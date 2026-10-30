#!/usr/bin/env python3
r"""不用梯度，只用能量头的**排序能力**：π0.5 采 N 个动作块，挑能量最低的执行。

    $L atom_robot/ar5/vel/best_of_n.py --ckpt ~/pi05_bolt_v2/checkpoints/040000 \
       --head ~/vel_heads/head_040000_grasp.pt --cache ~/vel_cache/pi05_v2_040000 \
       --segment grasp -n 8

**为什么值得试。** 五个能量头测下来，两种能力完全分家：

    排序（挑出对的）      命中率 54.2%，随机 1.6%        ✅ 一直很好
    方向（往哪推一点点）  cos 0.010~0.024，天花板 0.03%  ❌ 一直不行

VEL 的一步修正用的是后者。**Best-of-N 只用前者。** 而且 π0.5 的采样随机性
（5.4 mm）和它的系统误差（4.9 mm）一样大 —— 同一个观测下不同采样之间的差异，
跟策略本身的偏差是一个量级。从 N 个里挑一个好的，原理上能吃掉这一项相当一部分。

**这不是论文 §6.3 否掉的那个 Best-of-N。** 论文否的是「一个输出 + 加噪声 × N」：
在连续高维动作空间里盲目加噪会破坏基础策略建立的运动学关系，把末端推到分布外，
critic 在那儿没校准，于是选出伪极小（96.7 -> 95.7）。这里的 N 个候选是**流匹配
从不同噪声起点积分出来的，每个都是策略自己分布里的合法样本**，不离开流形。
区别是我的判断，不是论文验证过的 —— 所以要测。

输出三条线，单位毫米（和 `policy_error.py` 同一个量：整块位置增量之和）：

    随机选一个      = 现在的基线
    能量头选最低的   = Best-of-N 的实际收益
    上帝视角选最近的 = N 个里最多能拿到多少（天花板）

**看能量头那条离哪边近。** 贴着随机线 = 排序在这个尺度上没用；贴着天花板 =
可以上机。还会报 E 和「到 a* 的距离」的 Spearman 相关 —— 那是排序有没有用的
最直接的一个数。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

# 2026-09-23 代码从 VEL 仓库搬到 Project_Atom，按目录层数猜根目录那套就错了 ——
# 而且错得**没有报错**，只是路径不存在、import 静默失败或者文档静默不检查。
# 改成走 atom_robot.VEL_REPO（可用环境变量 VEL_REPO 覆盖）。
from atom_robot import VEL_REPO as _ROOT
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "vla-scripts"))


def spearman(x, y):
    """秩相关。样本少、关系非线性时比 Pearson 稳。"""
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    d = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / d) if d > 0 else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--cache", required=True, help="只用来推分段")
    ap.add_argument("--segment", default=None)
    ap.add_argument("--repo-id", default="vel_ar5_bolt_v2")
    ap.add_argument("--src", default="~/atom_episodes")
    ap.add_argument("-n", "--n-samples", type=int, default=8)
    ap.add_argument("--n-obs", type=int, default=40)
    ap.add_argument("--val-frac", type=float, default=0.15)
    args = ap.parse_args()

    from energy.energy_model import EnergyModel
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from atom_robot.ar5.eval_policy import load_policy
    from atom_robot.ar5.vel.pi05_bridge import Pi05Features
    from atom_robot.ar5.vel.train_energy import SEGMENT_CODE, sample_segment

    policy, pre, post, _md, kind = load_policy(Path(args.ckpt).expanduser(), "cuda")
    if kind != "pi05":
        print(f"  只支持 pi05，这个 checkpoint 是 {kind}")
        return 1
    policy.eval()
    for p in policy.parameters():
        p.requires_grad_(False)
    feat = Pi05Features(policy)

    meta = LeRobotDataset(args.repo_id, video_backend="pyav").meta
    ds = LeRobotDataset(args.repo_id,
                        delta_timestamps=resolve_delta_timestamps(policy.config, meta),
                        video_backend="pyav")
    cache = Path(args.cache).expanduser()
    stride = json.loads((cache / "meta.json").read_text())["stride"]
    seg = np.repeat(sample_segment(cache, args.src), stride)[:len(ds)]
    lo = int(len(ds) * (1 - args.val_frac))
    idx = np.arange(lo, len(ds))
    if args.segment:
        idx = idx[seg[lo:len(ds)][idx - lo] == SEGMENT_CODE[args.segment]]
    n_obs = min(args.n_obs, len(idx))
    sel = np.sort(np.random.default_rng(0).choice(idx, n_obs, replace=False))

    head = None
    N = args.n_samples
    rand_mm, best_mm, oracle_mm, worst_mm, rhos, picked = [], [], [], [], [], []
    e_all = []
    for i in range(0, len(sel), 4):
        rows = [ds[int(j)] for j in sel[i:i + 4]]
        b = {k: (torch.stack([r[k] for r in rows]).cuda() if torch.is_tensor(rows[0][k])
                 else [r[k] for r in rows]) for k in rows[0]
             if torch.is_tensor(rows[0][k]) or k == "task"}
        bb = pre(b)
        with torch.no_grad():
            h, _pad, samples = feat.sample(bb, n=N)          # [B, N, 50, 7]
        B, _, H, D = samples.shape
        if head is None:
            head = EnergyModel(state_dim=h.shape[2], act_dim=D).cuda().float().eval()
            head.load_state_dict(torch.load(Path(args.head).expanduser()))

        # 反归一化到真实单位，再按 policy_error 的口径折成毫米
        with torch.no_grad():
            real = post(samples.reshape(-1, D)).reshape(B, N, H, D).float().cpu().numpy()
        expert = torch.stack([r["action"] for r in rows]).numpy()[:, :H]
        d_mm = np.linalg.norm(real[:, :, :, :3].sum(2) * 1000
                              - (expert[:, :, :3].sum(1) * 1000)[:, None, :], axis=-1)

        with torch.no_grad():
            e = torch.stack([head(h.float(), samples[:, k].float()).squeeze(-1)
                             for k in range(N)], dim=1).cpu().numpy()      # [B, N]

        for bi in range(B):
            pick = int(np.argmin(e[bi]))
            rand_mm.append(float(d_mm[bi].mean()))       # 随机选 = N 个的期望
            best_mm.append(float(d_mm[bi, pick]))
            oracle_mm.append(float(d_mm[bi].min()))
            worst_mm.append(float(d_mm[bi].max()))
            rhos.append(spearman(e[bi], d_mm[bi]))
            picked.append(int(np.argsort(d_mm[bi]).tolist().index(pick)))   # 名次 0=最好
        e_all.append(e)

    rand_mm, best_mm = np.array(rand_mm), np.array(best_mm)
    oracle_mm, worst_mm = np.array(oracle_mm), np.array(worst_mm)
    picked = np.array(picked)

    print(f"\n  {Path(args.head).name}   段 {args.segment or '全部'}   "
          f"{len(rand_mm)} 个观测 × N={N}")
    print(f"\n  {'选法':<22}{'到人类动作的距离 中位':>22}")
    for lab, v in (("随机选（= 基线）", rand_mm), ("能量头选最低的", best_mm),
                   ("上帝视角选最近的", oracle_mm), ("（最差的那个）", worst_mm)):
        print(f"  {lab:<22}{np.median(v):18.1f} mm")
    gain = np.median(rand_mm) - np.median(best_mm)
    room = np.median(rand_mm) - np.median(oracle_mm)
    print(f"\n  **能量头带来的改善 {gain:+.1f} mm**，N={N} 的天花板是 {room:.1f} mm"
          f"（拿到 {100*gain/room if room > 0 else 0:.0f}%）")
    print(f"  挑中的名次中位 {np.median(picked):.0f}/{N-1}（0 = 挑到最好的，"
          f"随机期望 {(N-1)/2:.1f}）")
    print(f"  E 与「到 a* 距离」的 Spearman 相关 中位 {np.median(rhos):+.3f}"
          f"   >0 的占 {100*(np.array(rhos) > 0).mean():.0f}%")
    print(f"\n  参照：040000 的整块位置误差是 13.2 mm（系统 6.7 + 采样 6.5），"
          f"030000->060000 三万步训练一共才降了 2.9 mm。")
    print(f"  **改善若有 2~3 mm，就是真机上测得出来的量级。**")
    return 0


if __name__ == "__main__":
    sys.exit(main())
