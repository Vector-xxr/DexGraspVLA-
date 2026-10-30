#!/usr/bin/env python3
"""VEL 第二阶段：训能量头。backbone 全程不参与，只读 precompute 存下来的 h。

    $L atom_robot/ar5/vel/precompute.py                      # 先跑这个
    $L atom_robot/ar5/vel/train_energy.py --steps 20000

**学什么。** 一批 B 个样本，每个是「场景 h_i + 该场景下专家的动作块 a_i」。
目标：E(h_i, a_i) 要比 E(h_i, a_j)（j≠i，别的场景的专家动作）低。

不用专门挖难负样本 —— 论文 §3.4：别的场景的专家动作即使数值上接近，在本场景里
也是不合适的，因为可行性取决于场景构型而非动作向量本身。同批天然就是难负样本。

**划分按 episode，不按帧。** 30 Hz 下相邻帧几乎一样，随机划分会让验证集里
混进训练帧的近邻，验证指标直接失真。

**看哪个指标。** loss 会一直降（场景固定），真正有意义的是验证集上的
**top-1 命中率** —— 每一行 B 个能量里，配对那个是不是最低。随机是 1/B。
还有一条论文 §4.1 的印证：专家动作的能量必须显著低于随机动作。
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# 2026-09-23 代码从 VEL 仓库搬到 Project_Atom，按目录层数猜根目录那套就错了 ——
# 而且错得**没有报错**，只是路径不存在、import 静默失败或者文档静默不检查。
# 改成走 atom_robot.VEL_REPO（可用环境变量 VEL_REPO 覆盖）。
from atom_robot import VEL_REPO as _ROOT
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "vla-scripts"))

DEFAULT_CACHE = Path.home() / "vel_cache" / "pi05_bolt_010000"


def energy_matrix(head, h, a, valid, row_block: int = 8):
    """E[i, j] = E(h_i, a_j)，[B, B]。

    两处省显存，缺一个都跑不了论文的 batch 64：

    1. **投影只做一次再展开。** 仓库里的 `energy_inbatch_swap_infonce` 是先把 h
       展开成 B² 份再进模型，那是 2048 维上的拷贝，B=32 就要 24 GB。这里先投到
       512 维再展开，数学等价。
    2. **按行分块 + 梯度检查点。** 512 维下 B=64 仍是 [4096,712,512]≈6 GB，加上
       反向装不下。分成 row_block 行一算、反向时重算，峰值只剩一块，代价是
       前向多跑一遍。
    """
    import torch.utils.checkpoint as cp

    B, S, _ = h.shape
    ctx = head.state_linear(h.float())                          # [B,S,512]
    act = head.pe_layer(head.action_linear(a.float()))          # [B,H,512]
    # nn.MultiheadAttention 的 key_padding_mask 是 True=忽略，缓存里的 valid 是 True=有效
    kpm = ~valid

    def block(ctx_i, kpm_i, act_all):
        b, S_, E = ctx_i.shape
        n, H = act_all.shape[0], act_all.shape[1]
        cr = ctx_i.unsqueeze(1).expand(b, n, S_, E).reshape(b * n, S_, E)   # [i][j]=ctx_i
        ar = act_all.unsqueeze(0).expand(b, n, H, E).reshape(b * n, H, E)   # [i][j]=act_j
        kr = kpm_i.unsqueeze(1).expand(b, n, S_).reshape(b * n, S_)
        Z, _ = head.cross(query=ar, key=cr, value=cr,
                          need_weights=False, key_padding_mask=kr)
        return head.pool(head.prediction_head(Z)).view(b, n)

    rows = [cp.checkpoint(block, ctx[i:i + row_block], kpm[i:i + row_block], act,
                          use_reentrant=False)
            for i in range(0, B, row_block)]
    return torch.cat(rows, 0)


def local_energy(head, h, a, valid, k: int, lo: float, hi: float, gen=None):
    """E(h_i, a_i + eps·d)，每个样本 k 个**自己的小扰动**版本。[B, k]

    **这是补 InfoNCE 缺的那一半。** 原本的负样本全是「别的场景的动作」，离 a_i
    很远，所以训练信号只约束了「远处要高」—— 从没有任何梯度告诉模型
    「a_i **附近**该长什么样」。实测后果（`energy_probe.py`）：

        扰动 0.02~0.10 时能量升高的比例只有 55%（随机是 50%）
        要到扰动 0.8 才升到 86%

    也就是说盆地宽到 0.8，而 VEL 的一步修正走的是 α=0.1 这个尺度 —— 局部是
    噪声，梯度就是噪声。实测那个头修正后离人类动作更近的只占 27%。

    加了这些近处负样本，梯度才会去塑造局部形状。扰动幅度在 [lo, hi] 里随机取，
    覆盖 VEL 实际工作的尺度，而不是只在一个半径上。

    ⚠️ 论文 §6 警告过「加噪声的动作是分布外的，critic 在那儿没校准」——
    但那条针对的是**推理时用噪声采样选动作**（Best-of-N）。训练时拿扰动当负样本
    是相反的用法：正是在教它「偏离一点就该变差」。
    """
    B, H, D = a.shape
    d = torch.randn(B, k, H, D, device=a.device, dtype=torch.float32, generator=gen)
    d = d / d.flatten(2).norm(dim=-1).clamp_min(1e-6).view(B, k, 1, 1)
    eps = torch.rand(B, k, 1, 1, device=a.device, generator=gen) * (hi - lo) + lo
    pert = (a.float().unsqueeze(1) + eps * d).reshape(B * k, H, D)

    ctx = head.state_linear(h.float())                                   # [B,S,512]
    S, E = ctx.shape[1], ctx.shape[2]
    ctx_r = ctx.unsqueeze(1).expand(B, k, S, E).reshape(B * k, S, E)
    kpm = (~valid).unsqueeze(1).expand(B, k, S).reshape(B * k, S)
    act = head.pe_layer(head.action_linear(pert))
    Z, _ = head.cross(query=act, key=ctx_r, value=ctx_r,
                      need_weights=False, key_padding_mask=kpm)
    return head.pool(head.prediction_head(Z)).view(B, k)


def _math_sdpa():
    """把 scaled_dot_product_attention 切到 math 后端的上下文管理器。

    只有 math 后端支持二阶导，`direction_loss` 要用。torch 2.3 起 API 换过一次，
    两个都试。
    """
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        return sdpa_kernel(SDPBackend.MATH)
    except ImportError:
        return torch.backends.cuda.sdp_kernel(
            enable_math=True, enable_flash=False, enable_mem_efficient=False)


def _energy_at(head, h, pert, valid, B, k):
    """在 B×k 个候选动作上算能量。和 local_energy 走同一条手写前向。"""
    ctx = head.state_linear(h.float())                                   # [B,S,512]
    S, E = ctx.shape[1], ctx.shape[2]
    ctx_r = ctx.unsqueeze(1).expand(B, k, S, E).reshape(B * k, S, E)
    kpm = (~valid).unsqueeze(1).expand(B, k, S).reshape(B * k, S)
    act = head.pe_layer(head.action_linear(pert))
    Z, _ = head.cross(query=act, key=ctx_r, value=ctx_r,
                      need_weights=False, key_padding_mask=kpm)
    return head.pool(head.prediction_head(Z)).view(B, k)


def direction_loss(head, h, a, valid, k, lo, hi, gen=None, sub=16):
    """**把「梯度指不指向专家动作」直接写进损失。** 返回 (loss, 当前 cos)。

        a_pert = a* + ε·d           d 随机单位方向，ε ∈ [lo, hi]
        L = 1 − cos( −∇_a E(h, a_pert),  a* − a_pert )

    **为什么需要这一项。** InfoNCE 只约束「64 个候选里配对的那个能量最低」——
    那是对**离散点的排序**，对 E 在任何一点的**斜率方向**没有任何要求。论文
    命题 1 说全局最优时 −∇E 会等于 score（于是自动指向专家），但那要求负样本
    数趋于无穷 + 真达到全局最优。实测五个头：命中率从 22.9% 涨到 54.2%，
    而 cos(−∇E, a*−a_BC) 始终卡在 0.010~0.024，天花板 cos²/2 ≤ 0.03%。
    **五轮改进优化的一直是排序，从没优化过方向。**

    **ε 的范围要覆盖 a_BC 实际所在的距离**（实测 d₀ = 2.3~2.6），否则塑造的
    是用不到的区域 —— 这和 local_negatives 踩过的坑是同一个。

    ⚠️ **出发点是随机扰动，不是真的 a_BC**（缓存里没存 a_BC）。所以这个实验
    同时在测「从随机方向学到的方向场能不能推广到策略真正所在的流形」——
    评估始终用 `geo_check` 在真实 a_BC 上做。三种结局各指向不同的下一步：

        训练 cos 上去、a_BC 上的也上去   -> InfoNCE 就是问题所在，这条路通
        训练 cos 上去、a_BC 上的没上去   -> 随机方向推广不过去，得存真 a_BC
        训练 cos 都上不去                -> 3.96M 的头表达不了这个方向场
    """
    # **只取一小批做方向项。** math 后端 + 二阶导很吃显存，整批 64 会 OOM；
    # 而这一项是逐样本独立的，抽 sub 个就够，batch 本来也是随机取的。
    if sub and sub < len(a):
        h, a, valid = h[:sub], a[:sub], valid[:sub]
    B, H, D = a.shape
    d = torch.randn(B, k, H, D, device=a.device, dtype=torch.float32, generator=gen)
    d = d / d.flatten(2).norm(dim=-1).clamp_min(1e-6).view(B, k, 1, 1)
    eps = torch.rand(B, k, 1, 1, device=a.device, generator=gen) * (hi - lo) + lo
    tgt = a.float().unsqueeze(1) - eps * d            # a* − a_pert，正是要对齐的方向
    pert = (a.float().unsqueeze(1) + eps * d).reshape(B * k, H, D).detach()
    pert.requires_grad_(True)

    # **必须强制 SDPA 走 math 后端。** efficient / flash 两个后端都没实现二阶导，
    # 直接报 `derivative for aten::_scaled_dot_product_efficient_attention_backward
    # is not implemented`。math 后端会把注意力矩阵实体化，这里
    # B·k=128 个 query × 712 个 key，约 150 MB，可以接受。
    with _math_sdpa():
        e = _energy_at(head, h, pert, valid, B, k)
        # create_graph=True：要对「梯度」再求梯度，才能反传到头的权重上
        (g,) = torch.autograd.grad(e.sum(), pert, create_graph=True)
    cos = F.cosine_similarity((-g).flatten(1), tgt.reshape(B * k, -1), dim=1)
    return (1 - cos).mean(), cos.mean().detach()


def rank_loss(head, h, a_star, a_bc, valid, beta: float = 1.0):
    """把**同一场景的 K 个策略采样**按「到 a* 的距离」排序。返回 (loss, Spearman)。

    **这一项对着一个实测为零的指标。** Best-of-N 量过：给能量头同一场景的
    8 个 π0.5 采样，让它挑最接近人类动作的那个 ——

        随机选 9.8 mm   能量头选 10.3 mm   上帝视角 4.6 mm
        E 与「到 a* 距离」的 Spearman 相关 = +0.012（>0 的占 50%）

    **场内判别能力是 0。** 而命中率 54.2% 量的是**跨场景**判别（64 个候选来自
    不同场景，长得完全不一样）—— 那个容易，且 InfoNCE 只奖励那个。

    根因同一个：a* 的能量盆地半径只有 0.05~0.10，而策略的采样全落在 2.5 那么远，
    **盆地外 25 倍的地方能量只认场景、不认距离**。

    这里用成对 margin 排序损失，margin 正比于距离差 —— 不只教「谁好谁坏」，
    还教「好多少」。一个在策略实际探索的方向上随距离单调的能量面，
    它的梯度自动指向 a*，所以这一项同时也在修 cos。
    """
    B, K, H, D = a_bc.shape
    cand = torch.cat([a_star.float().unsqueeze(1), a_bc.float()], dim=1)   # [B,K+1,H,D]
    n = K + 1
    d = (cand - a_star.float().unsqueeze(1)).flatten(2).norm(dim=-1)       # [B,n]，第 0 列是 0
    e = _energy_at(head, h, cand.reshape(B * n, H, D), valid, B, n)        # [B,n]

    # 所有 d_j < d_k 的对：要 E_j 比 E_k 低，且低出 beta·(d_k − d_j)
    dj, dk = d.unsqueeze(2), d.unsqueeze(1)                                # [B,n,n]
    ej, ek = e.unsqueeze(2), e.unsqueeze(1)
    mask = dj < dk
    viol = F.softplus(ej - ek + beta * (dk - dj))
    loss = (viol * mask).sum() / mask.sum().clamp_min(1)

    with torch.no_grad():
        rk = lambda x: x.argsort(dim=1).argsort(dim=1).float()             # noqa: E731
        re_, rd = rk(e) - (n - 1) / 2, rk(d) - (n - 1) / 2
        rho = (re_ * rd).sum(1) / (re_.norm(dim=1) * rd.norm(dim=1)).clamp_min(1e-6)
    return loss, rho.mean()


def direction_loss_at(head, h, a_star, a_bc, valid, sub=16):
    """方向项,**出发点是真的 a_BC**,不是 a* 的随机扰动。

    路线一在随机扰动点上训，cos 在那些点上从 0.02 涨到 0.199（命中率不掉、
    验证集同步），但在真实 a_BC 上量还是 0.017 —— **学得会，只是学的地方不对**。
    350 维里「a* 周围的随机方向」和「策略采样所在的低维流形」几乎正交。
    """
    if sub and sub < len(a_star):
        h, a_star, a_bc, valid = h[:sub], a_star[:sub], a_bc[:sub], valid[:sub]
    B, K, H, D = a_bc.shape
    k = 1                                    # 每步随机取 K 个采样里的一个，省显存
    pick = torch.randint(0, K, (B,), device=a_bc.device)
    pert = a_bc[torch.arange(B, device=a_bc.device), pick].float().detach()
    pert.requires_grad_(True)
    tgt = a_star.float() - pert.detach()

    with _math_sdpa():                       # 只有 math 后端支持二阶导
        e = _energy_at(head, h, pert.unsqueeze(1).reshape(B * k, H, D), valid, B, k)
        (g,) = torch.autograd.grad(e.sum(), pert, create_graph=True)
    cos = F.cosine_similarity((-g).flatten(1), tgt.flatten(1), dim=1)
    return (1 - cos).mean(), cos.mean().detach()


def infonce(head, h, a, valid, tau, row_block=8, local_k=0, local_lo=0.05, local_hi=0.30):
    E = energy_matrix(head, h, a, valid, row_block)
    B = len(E)
    labels = torch.arange(B, device=E.device)
    off = ~torch.eye(B, dtype=torch.bool, device=E.device)

    cols = [E]
    e_loc = None
    if local_k > 0:
        e_loc = local_energy(head, h, a, valid, local_k, local_lo, local_hi)
        cols.append(e_loc)                       # 每行末尾接上自己的 k 个近处负样本
    logits = -torch.cat(cols, dim=1) / tau       # [B, B+k]，正样本仍是第 i 列

    loss = F.cross_entropy(logits, labels)
    hit = (logits[:, :B].argmax(1) == labels).float().mean()   # 命中率只在跨场景那部分算，可比
    return (loss, hit, E.diagonal().mean(), E[off].mean(),
            e_loc.mean() if e_loc is not None else None)


SEGMENT_CODE = {"grasp": 0, "transit": 1, "place": 2, "retreat": 3}


def segment_of_episode(action, ee_pose):
    """一条 episode 的逐帧分段。0 去抓 / 1 平移运送 / 2 下探放置 / 3 放完退回。

    **两个边界都是物理的，不按帧数比例切**（示教长度 268~527 帧，比例切法
    会把不同阶段混在一起）：

      * **夹爪** —— 一合就是抓到了，一松就是放完了
      * **末端高度 z 的峰值** —— 夹爪合着这段里，臂先抬起再下探进螺套。
        峰值把它劈成「平移」和「放置」两半。实测 296/296 条都有这个形状：
        抬升中位 78.3 mm、下降中位 68.9 mm，峰值后占比中位 0.51

    **为什么要把放置从运送里分出来。** 放置和抓取同构 —— 下探、减速、对准，
    是精度决定成败的一刻；运送只是把已有的误差平移过去。

    一个已经推翻的旧结论：我曾量到「运送段动作块预测螺套位置 R² = −2.9，
    原理上不可学」。那是**旧 100 条数据窄随机化的假象** —— 那批螺套 x 的
    标准差只有 20 px，没有方差可预测。297 条宽随机化之后重测：

        螺套 x 标准差   transit R²   place R²
        20.1 px（旧）     −0.044      +0.034
        91.8 px（新）     +0.523      +0.532

    两段都可学，分开只是为了让每个头专注一种运动模式。

    **取最后一段连续合着的，不是「第一次合到最后一次合」。** 实测 32/297 条
    在合着期间松开过，而且中位一松就是 48 帧、夹爪读数到 0.557 —— 那不是
    传感器噪声，是真的掉了再重抓。跨过它就把「掉落 + 重抓」整段算成运送段。
    那些**中途的失败尝试标成 -1，不属于任何段**，训练时不选它，也不猜。
    """
    g, z = action[:, 6], ee_pose[:, 2]
    n = len(g)
    closed = g < 0.15
    s = np.full(n, SEGMENT_CODE["retreat"], np.int8)
    if not closed.any():
        s[:] = SEGMENT_CODE["grasp"]
        return s
    # 取**最后一段够长的**连续闭合。329 个闭合段里有 2 个短于 10 帧（毛刺），
    # 不滤掉的话，末尾一个 1 帧的抖动会被当成整个运送+放置段。
    runs, k = [], 0
    for i, c in enumerate(closed):
        if c:
            k += 1
        elif k:
            runs.append((i - k, i - 1)); k = 0
    if k:
        runs.append((n - k, n - 1))
    good = [r for r in runs if r[1] - r[0] + 1 >= 10] or runs
    i0, i1 = good[-1]
    first = runs[0][0]
    j = i0 + int(np.argmax(z[i0:i1 + 1]))         # 抬升最高点
    s[:first] = SEGMENT_CODE["grasp"]
    s[first:i0] = -1                              # 掉落 + 重抓，不属于任何段
    s[i0:j] = SEGMENT_CODE["transit"]
    s[j:i1 + 1] = SEGMENT_CODE["place"]
    return s


def sample_segment(cache_dir: Path, src: str = "~/atom_episodes"):
    """每个缓存样本属于哪一段。见 `segment_of_episode`。"""
    meta = json.loads((cache_dir / "meta.json").read_text())
    files = sorted(glob.glob(os.path.expanduser(src) + "/episode_*.npz"))
    seg, total, need = [], 0, meta["n"] * meta["stride"]
    for f in files:
        if total >= need:
            break
        d = np.load(f, allow_pickle=True)
        seg.append(segment_of_episode(d["action"], d["ee_pose"]))
        total += len(seg[-1])
    seg = np.concatenate(seg)
    idx = np.clip(np.arange(meta["n"]) * meta["stride"], 0, len(seg) - 1)
    return seg[idx]


def sample_phase(cache_dir: Path, src: str = "~/atom_episodes"):
    """每个缓存样本处在它那条 episode 的哪个位置（0~1），以及属于第几条。

    缓存只存了 h / a / mask，没存这个。按 `样本 i -> 数据集第 i*stride 帧` 反推，
    episode 边界从原始 npz 的帧数拿。缓存做好之后又采了新数据也没关系 ——
    只取够 n*stride 帧的那几条。
    """
    meta = json.loads((cache_dir / "meta.json").read_text())
    files = sorted(glob.glob(os.path.expanduser(src) + "/episode_*.npz"))
    lens, total, need = [], 0, meta["n"] * meta["stride"]
    for f in files:
        if total >= need:
            break
        n = len(np.load(f, allow_pickle=True)["state"])
        lens.append(n); total += n
    phase = np.concatenate([np.arange(n) / n for n in lens])
    ep = np.concatenate([[i] * n for i, n in enumerate(lens)])
    idx = np.arange(meta["n"]) * meta["stride"]
    idx = np.clip(idx, 0, len(phase) - 1)
    return phase[idx].astype(np.float32), ep[idx]


class Cache:
    def __init__(self, path: Path, device: str):
        self.meta = json.loads((path / "meta.json").read_text())
        self.h = np.load(path / "h.npy", mmap_mode="r")
        self.m = np.load(path / "mask.npy", mmap_mode="r")
        self.a = np.load(path / "a.npy", mmap_mode="r")
        # **π0.5 自己的采样**，路线二的核心。老缓存没有这个文件，那时只能在
        # a* 的随机扰动点上训 —— 实测学得会（cos 0.199）但用不上（真 a_BC 上 0.017）。
        bc = path / "a_bc.npy"
        self.bc = np.load(bc, mmap_mode="r") if bc.exists() else None
        self.device = device
        self.phase = None

    def take(self, idx, with_bc: bool = False):
        i = np.sort(np.asarray(idx))
        out = (torch.from_numpy(np.ascontiguousarray(self.h[i])).to(self.device),
               torch.from_numpy(np.ascontiguousarray(self.a[i])).to(self.device),
               torch.from_numpy(np.ascontiguousarray(self.m[i])).to(self.device).bool())
        if not with_bc:
            return out
        if self.bc is None:
            raise RuntimeError(
                "这份缓存没有 a_bc.npy —— 重跑 precompute.py 时加 --n-samples 4")
        return (*out, torch.from_numpy(np.ascontiguousarray(self.bc[i])).to(self.device))

    def batch_idx(self, pool, B, rng, window: float):
        """一批的下标。window=0 是全局随机；>0 就只在相近轨迹阶段里取。

        **为什么要这个开关。** 全局随机时，一批 64 个样本大多处在完全不同的
        轨迹阶段（有的刚起步、有的在放置），光看「臂现在动得快不快、往哪动」
        就能配对，根本不用看场景。实测命中率 57%，但梯度方向系统性错误 ——
        它学会的是**匹配阶段**，不是判断「动作和场景兼不兼容」。

        window>0 时全批处在同一阶段附近，阶段这条捷径被堵死，剩下唯一的区别
        就是场景本身。这一改会让命中率掉多少，直接说明场景信号到底有多少。
        """
        if window <= 0 or self.phase is None:
            return rng.choice(pool, B, replace=False)
        ph = self.phase[pool]
        for _ in range(40):                      # 找一个附近样本够多的锚点
            anchor = float(ph[rng.integers(len(pool))])
            near = pool[np.abs(ph - anchor) <= window]
            if len(near) >= B:
                return rng.choice(near, B, replace=False)
        return rng.choice(pool, B, replace=False)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default=str(DEFAULT_CACHE))
    ap.add_argument("--out", default=None, help="默认 <cache>/energy_head.pt")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=64, help="**同时是负样本数**，B-1 个。论文 Table 3 用 64")
    ap.add_argument("--row-block", type=int, default=8, help="B×B 能量矩阵按几行一块算，省显存")
    ap.add_argument("--lr", type=float, default=5e-4, help="论文 Table 3")
    ap.add_argument("--tau", type=float, default=0.5, help="InfoNCE 温度")
    ap.add_argument("--val-frac", type=float, default=0.15, help="按 episode 划，不按帧")
    ap.add_argument("--local-negatives", type=int, default=0, metavar="K",
                    help="每个样本额外加 K 个**自己的小扰动**当负样本。0 = 关（论文原版）。\n"
                         "补 InfoNCE 缺的局部形状 —— 见 local_energy 的说明。建议 8")
    ap.add_argument("--local-lo", type=float, default=0.05, help="扰动幅度下界")
    ap.add_argument("--local-hi", type=float, default=0.30, help="扰动幅度上界")
    ap.add_argument("--rank-weight", type=float, default=0.0, metavar="λ",
                    help="**路线二**：把同一场景的 K 个 π0.5 采样按到 a* 的距离排序。\n"
                         "要缓存里有 a_bc.npy（precompute 加 --n-samples）。\n"
                         "对着的是实测为 0 的场内判别能力（Spearman +0.012）")
    ap.add_argument("--rank-beta", type=float, default=1.0,
                    help="排序 margin 相对距离差的系数")
    ap.add_argument("--dir-at-bc", action="store_true",
                    help="**路线二**：方向项的出发点用真的 a_BC，不是 a* 的随机扰动。\n"
                         "随机扰动点上 cos 能训到 0.199，真 a_BC 上仍是 0.017")
    ap.add_argument("--dir-weight", type=float, default=0.0, metavar="λ",
                    help="把 cos(−∇E, a*−a_pert) 直接写进损失的权重。0 = 关。\n"
                         "**InfoNCE 只约束排序，从不约束梯度方向** —— 实测命中率\n"
                         "涨 2.4 倍而 cos 纹丝不动。建议先试 1.0")
    ap.add_argument("--dir-k", type=int, default=1, metavar="K",
                    help="每个样本取几个出发点。要二阶导，显存翻倍，别给大")
    ap.add_argument("--dir-batch", type=int, default=16, metavar="N",
                    help="方向项只在批里前 N 个样本上算。math 后端 + 二阶导很吃显存，\n"
                         "整批 64 会 OOM；这一项逐样本独立，抽一小批就够")
    ap.add_argument("--dir-lo", type=float, default=1.5,
                    help="出发点到 a* 的距离下界。**要覆盖 a_BC 实际所在的 2.3~2.6**")
    ap.add_argument("--dir-hi", type=float, default=3.5)
    ap.add_argument("--phase-window", type=float, default=0.0,
                    help="只在轨迹阶段相差 < 这个值的样本里取负样本。0 = 全局随机。\n"
                         "0.05 ≈ 同一阶段，堵死「靠运动阶段配对」这条捷径")
    ap.add_argument("--segment", default=None, choices=list(SEGMENT_CODE),
                    help="只用某一段训。grasp=去抓螺栓 transit=抬起平移\n"
                         "place=下探放进螺套 retreat=放完退回。\n"
                         "**按夹爪状态切，不是按时间比例。** 两段动作里含不含场景信息差别很大，\n"
                         "混训会让可学的那半被不可学的那半拖累 —— 见 sample_segment 的说明")
    ap.add_argument("--src", default="~/atom_episodes", help="算轨迹阶段/分段用，要和缓存对应")
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=1000)
    args = ap.parse_args()

    from energy.energy_model import EnergyModel

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    cache_dir = Path(args.cache).expanduser()
    cache = Cache(cache_dir, args.device)
    n, S, Dh = cache.h.shape
    chunk, adim = cache.a.shape[1], cache.a.shape[2]
    print(f"  缓存 {cache_dir}")
    print(f"  {n} 个样本   h [{S}, {Dh}]   动作 [{chunk}, {adim}]   "
          f"来自 {Path(cache.meta['ckpt']).parent.name}/{Path(cache.meta['ckpt']).name}")

    # 按 episode 划分。stride 是均匀的，所以样本顺序 = 时间顺序，
    # 尾部连续一段就是最后若干条 episode。
    n_val = int(n * args.val_frac)
    tr_idx, va_idx = np.arange(n - n_val), np.arange(n - n_val, n)

    if args.segment:
        code = SEGMENT_CODE[args.segment]
        seg = sample_segment(cache_dir, args.src)
        tr_idx = tr_idx[seg[tr_idx] == code]
        va_idx = va_idx[seg[va_idx] == code]
        share = {k: int((seg == v).sum()) for k, v in
                 SEGMENT_CODE.items()}
        print(f"  只用 **{args.segment}** 段（按夹爪状态切）   全体分布 {share}")

    print(f"  训练 {len(tr_idx)}   验证 {len(va_idx)}（尾部连续，按时间切，不随机）")
    if len(tr_idx) < args.batch or len(va_idx) < args.batch:
        print(f"  这一段的样本不够一个 batch（{args.batch}），换更小的 --batch 或别的段")
        return 1

    if args.phase_window > 0:
        cache.phase, ep = sample_phase(cache_dir, args.src)
        print(f"  负样本只在轨迹阶段 ±{args.phase_window:.2f} 内取"
              f"（覆盖 {ep.max()+1} 条 episode）—— 堵死「靠运动阶段配对」这条捷径")
    else:
        print(f"  负样本全局随机取（--phase-window 0）")

    head = EnergyModel(state_dim=Dh, act_dim=adim).to(args.device).float()
    nparam = sum(p.numel() for p in head.parameters())
    print(f"  能量头 {nparam/1e6:.2f}M 参数   batch {args.batch}（负样本 {args.batch-1} 个）"
          f"   随机基线命中率 {100/args.batch:.1f}%")

    opt = torch.optim.AdamW(head.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.LinearLR(opt, 1.0, 0.0, args.steps)   # 论文 Table 3: linear

    best, t0 = -1.0, time.perf_counter()
    out = Path(args.out).expanduser() if args.out else cache_dir / "energy_head.pt"
    for step in range(1, args.steps + 1):
        head.train()
        need_bc = args.rank_weight > 0 or (args.dir_weight > 0 and args.dir_at_bc)
        got = cache.take(cache.batch_idx(tr_idx, args.batch, rng, args.phase_window),
                         with_bc=need_bc)
        (h, a, v), bc = got[:3], (got[3] if need_bc else None)
        loss, hit, e_pos, e_neg, e_loc = infonce(head, h, a, v, args.tau, args.row_block,
                                        args.local_negatives, args.local_lo, args.local_hi)
        dcos = rho = None
        if args.rank_weight > 0:
            l_rank, rho = rank_loss(head, h, a, bc, v, args.rank_beta)
            loss = loss + args.rank_weight * l_rank
        if args.dir_weight > 0:
            if args.dir_at_bc:
                l_dir, dcos = direction_loss_at(head, h, a, bc, v, sub=args.dir_batch)
            else:
                l_dir, dcos = direction_loss(head, h, a, v, args.dir_k,
                                             args.dir_lo, args.dir_hi, sub=args.dir_batch)
            loss = loss + args.dir_weight * l_dir
        if not torch.isfinite(loss):
            print(f"\n  第 {step} 步 loss 变成 {loss.item()} —— 停。"
                  f"h 的量级是 [{cache.h[:64].min():.0f}, {cache.h[:64].max():.0f}]，"
                  f"发散的话要在 state_linear 前加归一化。")
            return 1
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)   # 论文 Table 3
        opt.step()
        sched.step()

        if step % args.eval_every == 0 or step == args.steps:
            head.eval()
            hits, losses, vcos, vrho = [], [], [], []
            for _ in range(12):
                vg = cache.take(cache.batch_idx(va_idx, args.batch, rng, args.phase_window),
                                with_bc=need_bc)
                (vh, va, vv), vbc = vg[:3], (vg[3] if need_bc else None)
                with torch.no_grad():
                    vl, vhit, *_ = infonce(head, vh, va, vv, args.tau, args.row_block,
                                       args.local_negatives, args.local_lo, args.local_hi)
                    if args.rank_weight > 0:
                        _rl, r = rank_loss(head, vh, va, vbc, vv, args.rank_beta)
                        vrho.append(float(r))
                hits.append(vhit.item()); losses.append(vl.item())
                if args.dir_weight > 0:
                    # **这里不能包 no_grad** —— 要对动作求梯度才能量方向
                    if args.dir_at_bc:
                        _dl, dc = direction_loss_at(head, vh, va, vbc, vv, sub=args.dir_batch)
                    else:
                        _dl, dc = direction_loss(head, vh, va, vv, args.dir_k,
                                                 args.dir_lo, args.dir_hi, sub=args.dir_batch)
                    vcos.append(float(dc))
            vhit, vloss = float(np.mean(hits)), float(np.mean(losses))
            mark = ""
            if vhit > best:
                best = vhit
                torch.save(head.state_dict(), out)
                mark = "  <- 存了"
            el = time.perf_counter() - t0
            loc = f" E近 {e_loc.item():+.3f}" if e_loc is not None else ""
            if args.dir_weight > 0:
                loc += f"   cos 训 {float(dcos):+.3f} 验 {float(np.mean(vcos)):+.3f}"
            if args.rank_weight > 0:
                loc += f"   ρ 训 {float(rho):+.3f} 验 {float(np.mean(vrho)):+.3f}"
            print(f"  {step:6d}/{args.steps}  训练 loss {loss.item():.4f} 命中 {hit.item()*100:4.0f}%"
                  f"   验证 loss {vloss:.4f} 命中 {vhit*100:5.1f}%"
                  f"   E+ {e_pos.item():+.3f} E- {e_neg.item():+.3f}{loc}"
                  f"   {step/el:.0f} 步/秒{mark}")

    # 论文 §4.1 的印证：专家动作的能量必须显著低于随机动作。
    # 这条不过就说明能量面没学到东西，不管 loss 多低。
    head.eval()
    head.load_state_dict(torch.load(out))
    with torch.no_grad():
        vh, va, vv = cache.take(cache.batch_idx(va_idx, min(32, len(va_idx)), rng, args.phase_window))
        e_expert = head(vh.float(), va.float(), ~vv).mean().item()
        lo, hi = float(cache.a[:].min()), float(cache.a[:].max())
        rand = torch.empty_like(va).uniform_(lo, hi)
        e_rand = head(vh.float(), rand.float(), ~vv).mean().item()
    print(f"\n  验证集最好命中率 {best*100:.1f}%   （随机基线 {100/args.batch:.1f}%）")
    print(f"  能量排序检查：专家 {e_expert:+.3f}  vs  随机动作 {e_rand:+.3f}   "
          f"{'✓ 专家更低' if e_expert < e_rand else '✗ 反了 —— 能量面没学对，别拿去部署'}")
    print(f"  存到 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
