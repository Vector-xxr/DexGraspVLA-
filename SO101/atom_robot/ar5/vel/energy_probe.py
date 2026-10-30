#!/usr/bin/env python3
"""能量头训完先跑这个，再谈上不上机。**只用缓存，不加载 π0.5，1 GB 显存就够。**

    $L atom_robot/ar5/vel/energy_probe.py --head ~/vel_heads/head_v2_grasp.pt --segment grasp

**为什么不能只看命中率。** 实测过一个命中率 57%（随机 6%，9 倍）的能量头，
拿去做修正的结果是：修正后离人类动作更近的只占 27% —— 比掷硬币还差。
「能从 64 个里挑出对的」和「知道往哪推一点点会更好」是两回事。

这个脚本查两件命中率查不出来的事：

  **1 人类动作是不是局部极小。** 在 a* 周围随机方向上小扰动，能量该升高。
    真极小接近 100%，随机是 50%。实测 grasp 头在扰动 0.02~0.10 时只有 51~56%
    —— **局部是平的**，而 VEL 的一步修正（α=0.1）正好走在这个尺度上。
    局部是噪声，梯度就是噪声。

  **2 能量面的盆地有多宽。** 扰动加大到多少能量才可靠上升。宽盆地说明它学会了
    「远处不好」（够用来排序、拿高命中率），但没学会「近处该往哪走」。

**根因是目标函数的结构。** InfoNCE 只约束「E(h_i,a_i) 要比别的场景的 a_j 低」，
而那些 a_j 离 a_i 很远 —— **从没有任何梯度告诉模型 a_i 附近该长什么样**。
论文命题 1 说全局最优会还原 p_data(a|h)（那样局部就对了），但那要求负样本数
趋于无穷 + 达到全局最优，我们离得很远。

对症的改法是训练时加**近处负样本**（a* 的小扰动），让梯度去塑造局部形状。
"""

from __future__ import annotations

import argparse
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


def main() -> int:
    # **这两个 import 必须在 argparse 之前。** SEGMENT_CODE 被用在 add_argument 的
    # choices 里；放到后面 Python 会把它当成局部变量，--segment 直接 UnboundLocalError。
    from energy.energy_model import EnergyModel

    from atom_robot.ar5.vel.train_energy import (
        SEGMENT_CODE, Cache, sample_segment)

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--head", required=True, help="能量头 .pt")
    ap.add_argument("--cache", default=str(Path.home() / "vel_cache" / "pi05_050000_v2"))
    ap.add_argument("--segment", default=None, choices=list(SEGMENT_CODE),
                    help="只在这一段的验证样本上探。不给就用全部")
    ap.add_argument("--src", default="~/atom_episodes")
    ap.add_argument("--n", type=int, default=48, help="取多少个验证样本")
    ap.add_argument("--dirs", type=int, default=8, help="每个样本试几个随机方向")
    ap.add_argument("--val-frac", type=float, default=0.15)
    args = ap.parse_args()

    cache_dir = Path(args.cache).expanduser()
    cache = Cache(cache_dir, "cuda")
    n = len(cache.h)
    va = np.arange(int(n * (1 - args.val_frac)), n)
    if args.segment:
        code = SEGMENT_CODE[args.segment]
        va = va[sample_segment(cache_dir, args.src)[va] == code]
    if len(va) < args.n:
        print(f"  这一段验证样本只有 {len(va)} 个，少于 --n {args.n}，用全部")
        args.n = len(va)

    head = EnergyModel(state_dim=cache.h.shape[2], act_dim=cache.a.shape[2]).cuda().float().eval()
    head.load_state_dict(torch.load(Path(args.head).expanduser()))

    rng = np.random.default_rng(0)
    sel = np.sort(rng.choice(va, args.n, replace=False))
    # 动作在归一化空间，典型幅度约 1，所以扰动 0.02 ≈ 2%，0.4 ≈ 40%
    scales = (0.02, 0.05, 0.1, 0.2, 0.4, 0.8)
    up = {s: [] for s in scales}
    with torch.no_grad():
        for i in range(0, len(sel), 8):
            h, a, v = cache.take(sel[i:i + 8])
            e0 = head(h.float(), a.float(), ~v).squeeze(-1)
            for s in scales:
                for _ in range(args.dirs):
                    d = torch.randn_like(a)
                    d = d / d.flatten(1).norm(dim=1).view(-1, 1, 1)
                    e1 = head(h.float(), (a + s * d).float(), ~v).squeeze(-1)
                    up[s].append((e1 > e0).float().cpu())

    print(f"  头 {Path(args.head).name}   段 {args.segment or '全部'}   "
          f"{len(sel)} 个验证样本 × {args.dirs} 个方向")
    print(f"\n  === 人类动作 a* 是不是局部极小 ===")
    print(f"  {'扰动幅度':>10} {'能量升高':>10}   （真极小≈100%，随机=50%）")
    basin = None
    for s in scales:
        r = float(torch.cat(up[s]).mean())
        mark = "✓" if r > 0.8 else ("~" if r > 0.65 else "✗")
        print(f"  {s:10.2f} {100*r:9.0f}%   {mark}")
        if basin is None and r > 0.8:
            basin = s
    print()
    small = float(torch.cat(up[0.05]).mean())
    if small < 0.65:
        print(f"  ✗ **小扰动下只有 {small*100:.0f}% —— a* 不是局部极小，局部是平的/噪声。**")
        print(f"    VEL 的一步修正走的就是这个尺度（α=0.1），梯度不可信，别上机。")
        print(f"    盆地要到扰动 {basin if basin else '>0.8'} 才收紧 —— 学会了「远处不好」"
              f"（够排序、命中率好看），没学会「近处往哪走」。")
        print(f"    对症改法：训练时加**近处负样本**（a* 的小扰动当负样本）。")
    else:
        print(f"  ✓ 小扰动下 {small*100:.0f}% 能量升高，a* 是局部极小。")
        print(f"    可以继续测那两条判据（E(人类)<E(π0.5) 的比例、修正后变近的比例）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
