#!/usr/bin/env python3
"""π0.5（lerobot）→ VEL 能量头 的桥。

VEL 要两样东西，这个文件负责从 π0.5 里把它们取出来：

    h_t^(-1)   backbone 最后一层 hidden，[B, S, 2048]
    a_BC       原策略输出的动作块，       [B, 50, 7]

**为什么 π0.5 特别适配 VEL。** 论文 §3.3 有一条最容易漏、漏了还看不出来的要求：
构造 Z_h 时必须把 backbone 自己的 action-slot token mask 掉，否则能量头能从里面
读到 backbone 的预测，等于抄答案。π0.5 里这件事**是结构性的**：

    prefix = SigLIP 图像 token + 语言 token      <- 我们取的就是这一段
    suffix = 带噪动作 + timestep -> 动作专家      <- 动作全在这边，天然不在 prefix 里

所以取 prefix 的输出就自动满足了那条要求，没有 mask 可漏。
实测 prefix 有效 556 个 token，其中视觉 512 个 = **92%**，和论文 §3.2 说的
「vision token 占 >85%」对得上。

**一次前向拿两样。** prefix 那一趟（SigLIP + 18 层 gemma 过 712 个 token）是
π0.5 推理里最贵的一段。跑两遍会让单帧从 142 ms 涨到约 280 ms，越过臂 250 ms 的
latch 线。所以这里复刻 `sample_actions` 的调用顺序，把它丢掉的 prefix_out 留下来，
后面的去噪循环复用同一份 KV cache。`selftest` 里有一条比对，保证这条路径和
`predict_action_chunk` 输出一致，防止上游改动后悄悄漂掉。
"""

from __future__ import annotations

import torch

# 归一化后动作的实际范围。**不是 [-1, 1]。**
#
# 论文 §3.5 最后一步是 clip(ã, -1, 1)，§7 第 2 条也要求真机的归一化区间和它对齐。
# 但这批数据 ACTION 用的是 QUANTILES 归一化，实测：
#
#     整体 [-1.680, +1.889]，7.46% 的元素在 [-1,1] 之外
#     维 0~5（位姿）越界 2~3%      <- q01/q99 映射到 ±1，两尾天然溢出
#     维 6（夹爪）  越界 36.93%    <- 夹爪双峰，quantile 把中间压扁、两峰甩出界
#
# 照抄 [-1,1] 会把夹爪三分之一的合法值裁掉 —— 那是破坏动作，不是修正。
# 所以 clip 的边界取**训练数据归一化后的实际范围**，留 5% 余量。
ACTION_LIMITS = (-1.8, 2.0)

# 夹爪那一维不参与能量修正，直通。
#
# 两条独立理由：
#   1. 论文 §7 第 3 条：食指开合是离散意图（抓/放）为主，让能量梯度去改它可能
#      破坏抓取时序。原文建议「先只对臂的位姿维度做修正，手那一维直通」。
#   2. 上面那组实测：这一维 36.93% 在 [-1,1] 外，能量面在那些区域没有校准过。
CORRECT_DIMS = slice(0, 6)


class Pi05Features:
    """从一个 PI05Policy 里取 (h, a_BC)。策略必须是 eval + 冻结的。"""

    def __init__(self, policy):
        self.policy = policy
        self.model = policy.model
        self.cfg = policy.config
        self.act_dim = policy.config.output_features["action"].shape[0]
        # sample_actions 里也是这么设的。SDPA 路径吃不了这里的 4D bool mask。
        self.model.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"  # noqa: SLF001

    @property
    def hidden_dim(self) -> int:
        return self.model.paligemma_with_expert.paligemma.config.text_config.hidden_size

    def __call__(self, batch, *, with_action: bool = True):
        """返回 (h, pad_mask, a_BC)。`with_action=False` 时 a_BC 为 None，只跑 prefix。

        h        [B, S, 2048]  bfloat16
        pad_mask [B, S]        bool，True = 有效 token
        a_BC     [B, 50, 7]    归一化空间里的动作块
        """
        h, pad, samples = self.sample(batch, n=1 if with_action else 0)
        return h, pad, (samples[:, 0] if with_action else None)

    def sample(self, batch, *, n: int = 1):
        """采 n 个动作块，**共享同一份 prefix**。返回 (h, pad_mask, [B, n, 50, 7])。

        流匹配每次从不同的噪声起点积分，所以 n 个样本是策略自己分布里的
        n 个不同抽样 —— 和「拿一个输出去加噪」完全是两回事。后者会离开
        策略流形、破坏运动学关系（论文 §6.3 否掉的 Best-of-N 就是那种）。

        **成本几乎全在第一个样本上。** prefix 那一趟（SigLIP + 18 层 gemma 过
        712 个 token）占推理时间的大头，n 个样本共用同一份 KV cache，每多一个
        只是重跑 num_inference_steps 步去噪循环。
        """
        m, cfg = self.model, self.cfg
        images, img_masks = self.policy._preprocess_images(batch)  # noqa: SLF001
        tokens = batch["observation.language.tokens"]
        masks = batch["observation.language.attention_mask"]

        prefix_embs, prefix_pad_masks, prefix_att_masks = m.embed_prefix(
            images, img_masks, tokens, masks
        )
        # use_cache=True：同一趟既拿 hidden 又留 KV 给去噪循环用
        h, past_key_values = m._forward_prefix_lm(  # noqa: SLF001
            prefix_embs, prefix_pad_masks, prefix_att_masks, use_cache=True
        )
        if n <= 0:
            return h, prefix_pad_masks, None

        bsize = tokens.shape[0]
        num_steps = cfg.num_inference_steps
        dt = -1.0 / num_steps
        out = []
        for _ in range(n):
            x_t = m.sample_noise(
                (bsize, cfg.chunk_size, cfg.max_action_dim), tokens.device
            )
            for step in range(num_steps):
                t = 1.0 + step * dt
                t_tensor = torch.tensor(t, dtype=torch.float32,
                                        device=tokens.device).expand(bsize)
                v_t = m.denoise_step(
                    prefix_pad_masks=prefix_pad_masks,
                    past_key_values=past_key_values,
                    x_t=x_t,
                    timestep=t_tensor,
                )
                x_t = x_t + dt * v_t
            out.append(x_t[:, :, : self.act_dim])
        return h, prefix_pad_masks, torch.stack(out, dim=1)


def correct(energy_head, h, a_bc, *, alpha: float = 0.1, gamma: float = 1.0,
            dims: slice = CORRECT_DIMS, limits=ACTION_LIMITS):
    """VEL §3.5 的一步修正。**只走一步** —— 见 §4.2，多步是有害的不是省算力。

        g = ∇_a E(h, a_BC)
        g ← g · min(1, γ/‖g‖₂)        梯度范数裁剪，防能量面陡峭处过冲
        ã = a_BC − α·g
        a = clip(ã, limits)

    `dims` 之外的维度原样直通（默认夹爪不动，见 CORRECT_DIMS）。
    """
    a = a_bc.detach().clone().float().requires_grad_(True)
    energy = energy_head(h.float(), a, reduce="sum")
    if isinstance(energy, tuple):
        energy = energy[0]
    (grad,) = torch.autograd.grad(energy.sum(), a)

    # **逐时刻裁剪，不是对整块裁。** 论文 §3.5：「chunk 里每个动作各自独立修正，
    # 然后整块提交给开环执行」。论文的 D=7 是**一个**动作的维数，γ 裁的是那 7 个
    # 数的范数。
    #
    # 按整块（50×6=300 个元素）裁会把步长稀释 √(50) ≈ 7 倍：实测每个元素只挪
    # 0.0040，是它到专家动作距离的 0.5%，净改善 −0.09% —— 方向对（58% 变近，
    # 随机是 50%）但幅度等于没修。逐时刻裁之后每个 7 维动作各自拿满 α。
    norm = grad.norm(dim=-1, keepdim=True).clamp_min(1e-6)      # [B, H, 1]
    step = alpha * grad * torch.clamp(gamma / norm, max=1.0)

    out = a_bc.detach().clone().float()
    out[:, :, dims] = out[:, :, dims] - step[:, :, dims]
    return out.clamp(*limits).to(a_bc.dtype)


SEGMENTS = ("grasp", "transit", "place", "retreat")

# 和 train_energy.segment_of_episode 的切分保持一致。
GRIPPER_CLOSED = 0.15
# 离线切「运送 / 放置」用的是 z 的峰值 —— **在线不能开天眼**，只能因果地判断。
# 规则：夹爪合上后，z 先比合上时高出 LIFT_ARM_MM（确认真的抬起来了）才「上膛」，
# 之后 z 连续 DESCEND_FRAMES 帧低于运行最大值 DESCEND_MM，才算进入放置段。
#
# 三个数是在 296 条示教上扫出来的。只用「跌破运行最大值」不加前两个条件，
# 61/296 条会在抬升途中被短暂回落误触发（最早的早了 145 帧）：
#
#     L(mm)  δ(mm)  持续帧   误触发   没触发   晚于真峰值(中位帧)
#       —      5      1      61/296     0            5
#      40     10      1       9/296     1            8
#      40     15      5     **2/296**   1           18
#      40     20     10       1/296     1           27
#
# 取 40/15/5：误触发 0.7%，代价是晚 18 帧（放置段中位 58 帧，还剩 40 帧）。
LIFT_ARM_MM = 40.0
DESCEND_MM = 15.0
DESCEND_FRAMES = 5
# 夹爪状态要连续 GRIP_FRAMES 帧才认。**一帧就切会出两个问题：**
#   1. 一次极短的失败抓取会白白切进运送段，而那几帧臂明明还在够螺栓
#   2. 更糟 —— 进运送段会把 z 基准重置成当下高度。抖动若发生在抬升途中，
#      基准被抬高，后面「高出 LIFT_ARM_MM」的门槛跟着抬高，放置段可能永不触发
# 297 条示教上量的：闭合连续段中位 111 帧、5% 分位 23 帧，短于 10 帧的只有 2 个；
# 闭合期间张开的段中位 52 帧，短于 10 帧的只有 1 个。10 帧把毛刺全滤掉、真段全留下。
GRIP_FRAMES = 10
# **夹爪「命令闭合」不等于「手里有东西」。** 典型失败：夹爪合上了、策略以为抓到了，
# 闭着运送一段，半路才发现手里是空的，退回去重抓。这段假运送会被判成 transit，
# 抬起再下探还会进 place —— 手里没东西却用放置头去修，分布外。
#
# 夹爪的**状态**读数能分开：夹住螺栓时手指停在 0.22 左右，合到空气里会掉到 0.04。
# 但它是**滞后**指标 —— 闭合后第 10 帧还分不开（失败只识别 23%），因为很多是
# 一开始真夹住了、半路才滑脱。所以不用它阻止进入 transit，用它**提前退出**：
#
#     整段最低值        成功 中位 0.192（10% 分位 0.151）   失败 中位 0.038
#     连续低于 0.12     成功 中位 0 帧（90% 分位也是 0）    失败 中位 11 帧
#     判据「连续 5 帧」  成功误报 1.7%                      失败识别 72%
GRIP_EMPTY = 0.12
GRIP_EMPTY_FRAMES = 5


class VelRunner:
    """`policy.select_action` 的替身，中间插一步能量修正。

    π0.5 一次预测 50 步塞进队列，之后 49 次调用只是出队。VEL 修的是**整块**，
    所以只在队列空、真做一次预测的时候修，和 backbone 的节奏一致。

    **按段挑头。** 能量头是分段训的（抓取段一个、运送段一个），拿抓取段的头
    去修运送段是分布外用法 —— 第一版就是这么跑的，那 30 轮里运送和退回两段
    用的都是只见过抓取段的头。这里改成按当前所处的段挑对应的头，**没有对应
    的头就原样放行**，不做修正。这样「只训了抓取段」也能干净地上机测：
    抓取段有修正、后面两段等同基线，测出来的差异就只归因于抓取段。

    **段是从夹爪状态和末端高度推的**，和训练时 `segment_of_episode` 对齐：

        grasp    夹爪张着，在够螺栓
        transit  夹爪持续闭合，抬起平移
        place    抬够了又开始持续下探
        retreat  在 place 里松手（= 放完了）

    **这是个环，不是一条单行道。** 任何时候持续闭合都会回到 transit：
    在 transit 里松手 = 掉了，退回 grasp 等重抓（示教里 31 次）；在 retreat
    里再闭合 = 放了没插进去、重新抓起来重试（示教里 0 次，因为示教全成功，
    但真机上 46% 的轮次是「抓到了插不进去」）。

    按帧数比例硬切不行 —— 示教长度 268~527 帧。调用方每步把**真实单位**的
    动作和末端位姿喂给 `observe()`（`select_action` 拿到的是归一化空间的，
    夹爪阈值和毫米都对不上）。

    ⚠️ **不能用 `torch.inference_mode()` 包 backbone 那一趟。** inference_mode
    产出的张量不允许参与 autograd，而修正要对动作求梯度，会直接报错。用
    `no_grad` —— 它同样不建图、同样省显存，但产物可以进 autograd。
    """

    def __init__(self, policy, heads, *, alpha=0.1, gamma: float = 1.0):
        """heads: {段名 -> 能量头}。段名 'all' 表示不分段，整条轨迹都用它。

        **alpha 可以是一个数，也可以是 {段名 -> 数}。** 最优步长
        α* = cos·d₀/√H 是**每个头各自**的量，实测抓取段 0.0073、放置段 0.0036，
        差一倍 —— 一个全局 α 必然有一段用错。
        """
        from collections import deque

        bad = set(heads) - set(SEGMENTS) - {"all"}
        if bad:
            raise ValueError(f"不认识的段名 {sorted(bad)}，只能是 {SEGMENTS} 或 'all'")
        if isinstance(alpha, dict):
            bad = set(alpha) - set(heads)
            if bad:
                raise ValueError(f"给了 {sorted(bad)} 的 α，却没给这几段的头")
        self.feat = Pi05Features(policy)
        self.heads = {k: v.eval() for k, v in heads.items()}
        self.alpha, self.gamma = alpha, gamma
        self.n_steps = policy.config.n_action_steps
        self.queue = deque()
        self.enabled = True            # A/B 的开关：关掉就只出 a_BC，不做修正
        self.reset()

    def reset(self):
        # **修正统计每轮清零。** 之前放在 __init__ 里，于是每轮打的「修正 N 块」
        # 其实是从开程序算起的累计值 —— 看起来随轮次单调增，完全没法用。
        self.corrections = []          # (段, 修正幅度)
        self.skipped = {s: 0 for s in SEGMENTS}   # 这一段没有头/没开，放行了几块
        self.lost = 0                  # 这一轮检测到几次「东西掉了」
        # **trace 不在这里清。** reset() 在按 q 时也会调用，而 q 之后才按 r/f 落盘 ——
        # 在这里清掉的话，凡是「先按 q 再记结果」的轮次曲线全丢（实测 9 轮只剩 3 轮）。
        # 它和 deltas / last_round 一样，活到写进 CSV 那一刻，由调用方清。
        if not hasattr(self, "trace"):
            self.trace = []
        self._t = 0                    # 已经吐出多少步
        self.queue.clear()
        self.segment = "grasp"
        self._z_close = self._z_max = None
        self._armed = False
        self._below = 0
        self._grip_run = 0          # 当前夹爪状态已经连续多少帧
        self._pending_z = None      # 首次读到闭合时的高度，确认后才当基准
        self._empty_run = 0         # 夹爪状态连续多少帧显示「手里是空的」

    def observe(self, action, ee_pose=None, grip_state=None) -> None:
        """吃一步**真实单位**的动作和末端位姿，更新当前所处的段。

        单调推进不回退：grasp -> transit -> place -> retreat。`ee_pose` 不给
        就退化成只按夹爪切，那样 transit 和 place 分不开，place 的头永远用不上。
        """
        grip = float(action[6])
        closed = grip < GRIPPER_CLOSED
        z = None if ee_pose is None else float(ee_pose[2]) * 1000

        # **张着手的两段（grasp / retreat）走同一条路：再次持续闭合就回到运送段。**
        # retreat 绝不能是吸收态 —— 「放了但没插进去、再抓起来重插」在真机上是
        # 主要失败模式（100 轮里 46% 是「抓到了插不进去」），卡死在 retreat
        # 会让之后的重抓、重送全程挂不上任何头。示教里 0 次，因为示教全是成功的。
        if self.segment in ("grasp", "retreat"):
            if not closed:
                self._grip_run, self._pending_z = 0, None
                return
            self._grip_run += 1
            if self._pending_z is None:
                self._pending_z = z          # **记首次闭合那一刻的高度**，不是确认时的
            if self._grip_run >= GRIP_FRAMES:
                self.segment = "transit"
                self._z_close = self._z_max = self._pending_z
                self._armed, self._below, self._grip_run = False, 0, 0
                self._pending_z = None
            return

        # 夹爪闭着，但状态说手里是空的 —— 当成「掉了」，不等策略去松手
        if grip_state is not None and closed:
            self._empty_run = self._empty_run + 1 if grip_state < GRIP_EMPTY else 0
            if self._empty_run >= GRIP_EMPTY_FRAMES:
                self.segment = "grasp"
                self.lost += 1
                self._grip_run = self._empty_run = 0
                self._pending_z = None
                return

        if not closed:
            self._grip_run += 1
            if self._grip_run >= GRIP_FRAMES:
                # **在运送途中松手不是「放完了」，是掉了。** 实测 32/297 条示教
                # 有这个：中位松 52 帧，之后重新去抓。所以退回抓取段等它重抓，
                # 而不是判成退回段（那样后面全程都挂不上头）。
                self.segment = "retreat" if self.segment == "place" else "grasp"
                self._grip_run, self._pending_z = 0, None
            return
        self._grip_run = 0
        if self.segment == "transit" and z is not None and self._z_max is not None:
            self._z_max = max(self._z_max, z)
            if not self._armed and self._z_max >= self._z_close + LIFT_ARM_MM:
                self._armed = True
            self._below = (self._below + 1
                           if self._armed and z < self._z_max - DESCEND_MM else 0)
            if self._below >= DESCEND_FRAMES:
                self.segment = "place"

    def head_for(self, segment: str):
        return self.heads.get(segment) or self.heads.get("all")

    def alpha_for(self, segment: str) -> float:
        if not isinstance(self.alpha, dict):
            return self.alpha
        if segment in self.alpha:
            return self.alpha[segment]
        return self.alpha.get("all", 0.0)

    def select_action(self, batch):
        if not self.queue:
            with torch.no_grad():
                h, _pad, a_bc = self.feat(batch)
            head = self.head_for(self.segment) if self.enabled else None
            if head is None:
                a = a_bc                       # 这一段没训头，等同基线
                self.skipped[self.segment] += 1
            else:
                a = correct(head, h, a_bc, alpha=self.alpha_for(self.segment),
                            gamma=self.gamma)
                self.corrections.append((
                    self.segment,
                    (a.float() - a_bc.float()).abs()[:, :, CORRECT_DIMS].mean().item(),
                ))
            row = {"step": self._t, "segment": self.segment,
                   "applied": "" if head is None else self.segment}
            for nm, hd in self.heads.items():
                with torch.no_grad():
                    row[f"E_{nm}_bc"] = float(hd(h.float(), a_bc.float()).mean())
                    row[f"E_{nm}_out"] = float(hd(h.float(), a.float()).mean())
            self.trace.append(row)
            self.queue.extend(a[:, : self.n_steps].transpose(0, 1))
        self._t += 1
        return self.queue.popleft()

    def counts(self) -> dict:
        """这一轮各段修正了几块。写进 CSV，用来做「头真的修了 vs 没修」的内部检验。"""
        out = {s: 0 for s in SEGMENTS}
        for seg, _m in self.corrections:
            out[seg] += 1
        return out

    def report(self) -> str:
        """跑完打一行，**确认分段真的生效了**。修正次数为 0 说明没走到那一段。"""
        import statistics

        parts = []
        for s in SEGMENTS:
            mags = [m for seg, m in self.corrections if seg == s]
            if mags:
                parts.append(f"{s} 修正 {len(mags)} 块 幅度中位 {statistics.median(mags):.4f}")
            elif self.skipped[s]:
                parts.append(f"{s} 放行 {self.skipped[s]} 块（无头）")
        if self.lost:
            parts.append(f"**检测到 {self.lost} 次脱手**（夹爪闭着但手里是空的）")
        return "  VEL  " + ("   ".join(parts) if parts else "这一轮没预测过")
