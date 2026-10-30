#!/usr/bin/env python3
r"""能量头上机前的**决定性**检查：梯度到底指不指向人类动作，α 该给多少。

    $L atom_robot/ar5/vel/geo_check.py --ckpt ~/pi05_bolt_v2/checkpoints/040000 \
       --head ~/vel_heads/head_040000_grasp.pt --segment grasp

`train_energy` 报的命中率是**排序能力**（64 个里挑对），`energy_probe` 查的是
**局部形状**（a* 是不是极小）。两个都过了，修正照样可以完全无效 —— 它们都不看
「从 a_BC 出发的那一步走对没走对」。这个脚本只看那一件事。

**一步修正的改善有个和 α 无关的天花板。** 记 d₀ = |a_BC − a*|、
cos = cos(−∇E, a*−a_BC)、步长 S：

    d₁² = d₀² − 2·S·cos·d₀ + S²

对 S 求极值得 **S\* = cos·d₀**，代回去 **d₁ = d₀·√(1−cos²)**，所以

    最大改善 = 1 − √(1−cos²) ≈ cos²/2        ← 与 α 无关
    最优 α   = S*/√H = cos·d₀/√H             ← H 是 chunk 长度

实测过的（π0.5 + 抓取段）：

    backbone / 负样本范围        命中率   cos      天花板   最优 α
    050000 冻结 / 0.05~0.30      22.9%   +0.0101  0.005%   0.008
    030000 解冻 / 0.05~0.30      42.4%   +0.0183  0.016%   0.009
    030000 解冻 / 0.05~3.00      53.8%   +0.0243  0.030%   0.012

**命中率涨了 2.3 倍，cos 几乎没动。** 天花板 0.03% 意味着 9.3 mm 的系统误差
只能修掉 0.003 mm —— 上机测不出来，真机 A/B 也确认了（p = 0.785）。

所以看这个脚本的输出时：**cos 要到 0.1 量级才谈得上有用。**
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
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="**要和训能量头时那份缓存的 backbone 一致**")
    ap.add_argument("--head", required=True)
    ap.add_argument("--cache", required=True, help="只用来推分段，不读 h")
    ap.add_argument("--segment", default=None,
                    help="只在这一段的验证样本上量。不给就用全部")
    ap.add_argument("--repo-id", default="vel_ar5_bolt_v2")
    ap.add_argument("--src", default="~/atom_episodes")
    ap.add_argument("--n", type=int, default=32, help="取多少个验证样本")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--multistep", type=int, default=0, metavar="K",
                    help="再多走 K−1 步，报 d_k/d_0。**论文说多步有害**\n"
                         "（k=1: 96.6 -> k=4: 91.8，能量面 rugged 会冲进伪极小）。\n"
                         "梯度方向若不变，多步只等于换个方式调 α，天花板还是 cos²/2；\n"
                         "只有「走一步之后新位置的梯度更准」才可能真的更好 —— 这是可测的。")
    ap.add_argument("--prefix", action="store_true",
                    help="额外按「只修前 k 步」拆开看。整块 350 维会把信号稀释掉")
    args = ap.parse_args()

    from energy.energy_model import EnergyModel
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from atom_robot.ar5.eval_policy import load_policy
    from atom_robot.ar5.vel.pi05_bridge import Pi05Features, correct
    from atom_robot.ar5.vel.train_energy import SEGMENT_CODE, sample_segment

    policy, pre, _post, _md, kind = load_policy(Path(args.ckpt).expanduser(), "cuda")
    if kind != "pi05":
        print(f"  只支持 pi05，这个 checkpoint 是 {kind}")
        return 1
    policy.eval()
    for p in policy.parameters():
        p.requires_grad_(False)
    feat = Pi05Features(policy)
    norm = next(s for s in pre.steps if "ormaliz" in type(s).__name__)

    meta = LeRobotDataset(args.repo_id, video_backend="pyav").meta
    ds = LeRobotDataset(args.repo_id,
                        delta_timestamps=resolve_delta_timestamps(policy.config, meta),
                        video_backend="pyav")
    cache = Path(args.cache).expanduser()
    import json
    stride = json.loads((cache / "meta.json").read_text())["stride"]
    seg = np.repeat(sample_segment(cache, args.src), stride)[:len(ds)]
    lo = int(len(ds) * (1 - args.val_frac))
    idx = np.arange(lo, len(ds))
    if args.segment:
        idx = idx[seg[lo:len(ds)][idx - lo] == SEGMENT_CODE[args.segment]]
    if len(idx) < args.n:
        print(f"  这一段只有 {len(idx)} 个验证样本，全用上")
        args.n = len(idx)
    sel = np.sort(np.random.default_rng(0).choice(idx, args.n, replace=False))

    H = []
    ABC = []
    AST = []
    for i in range(0, len(sel), 4):
        rows = [ds[int(j)] for j in sel[i:i + 4]]
        b = {k: (torch.stack([r[k] for r in rows]).cuda() if torch.is_tensor(rows[0][k])
                 else [r[k] for r in rows]) for k in rows[0]
             if torch.is_tensor(rows[0][k]) or k == "task"}
        bb = pre(b)
        ast = norm({"action": torch.stack([r["action"] for r in rows]).cuda()})["action"]
        torch.manual_seed(0)
        with torch.no_grad():
            h, _pad, abc = feat(bb)
        H.append(h)
        ABC.append(abc)
        AST.append(ast[:, :abc.shape[1], :abc.shape[2]])

    head = EnergyModel(state_dim=H[0].shape[2], act_dim=ABC[0].shape[2]).cuda().float().eval()
    head.load_state_dict(torch.load(Path(args.head).expanduser()))

    import torch.nn.functional as F
    G, U = [], []
    for h, abc, ast in zip(H, ABC, AST):
        a = abc.detach().clone().float().requires_grad_(True)
        e = head(h.float(), a)
        (g,) = torch.autograd.grad(e.sum(), a)
        G.append((-g).cpu())
        U.append((ast.float() - abc.float()).cpu())
    G, U = torch.cat(G), torch.cat(U)
    chunk = U.shape[1]

    cos = F.cosine_similarity(U.flatten(1), G.flatten(1), dim=1)
    d0 = U.flatten(1).norm(dim=1)
    c, D = float(cos.median()), float(d0.median())
    ceil = 1 - (1 - c * c) ** 0.5
    a_star = c * D / chunk ** 0.5

    print(f"\n  头 {Path(args.head).name}   段 {args.segment or '全部'}   {len(sel)} 个验证样本")
    print(f"  cos(−∇E, a*−a_BC)  中位 {c:+.4f}   >0 的占 {100*(cos>0).float().mean():.0f}%")
    print(f"  |a_BC − a*|        中位 {D:.2f}")
    print(f"\n  **改善天花板 = 1−√(1−cos²) = {100*ceil:.3f}%**（与 α 无关）")
    print(f"  **最优 α = cos·d₀/√{chunk} = {a_star:.4f}**")

    print(f"\n  {'α':>7} {'步长 S':>9} {'需要 cos >':>12} {'实际变近比例':>13}")
    for al in (a_star, 0.002, 0.01, 0.05, 0.1):
        closer = []
        for h, abc, ast in zip(H, ABC, AST):
            an = correct(head, h, abc, alpha=al)
            dd0 = (abc.float() - ast.float()).flatten(1).norm(dim=1)
            dd1 = (an.float() - ast.float()).flatten(1).norm(dim=1)
            closer.append((dd1 < dd0).float().cpu())
        S = al * chunk ** 0.5
        mark = "  <- 最优" if abs(al - a_star) < 1e-9 else ""
        print(f"  {al:7.4f} {S:9.3f} {S/(2*D):12.4f} {100*torch.cat(closer).mean():12.0f}%{mark}")

    if args.multistep > 1:
        print(f"\n  多步下降（每步 α={a_star:.4f}）")
        print(f"  {'k':>3} {'d_k/d_0 中位':>13} {'比 k=1 更近的比例':>18} {'比原动作更近':>13}")
        cur = [abc.detach().clone() for abc in ABC]
        d1 = None
        for k in range(1, args.multistep + 1):
            cur = [correct(head, h, a, alpha=a_star) for h, a in zip(H, cur)]
            ratio, closer = [], []
            for c, abc, ast in zip(cur, ABC, AST):
                d0 = (abc.float() - ast.float()).flatten(1).norm(dim=1)
                dk = (c.float() - ast.float()).flatten(1).norm(dim=1)
                ratio.append((dk / d0).cpu())
                closer.append((dk < d0).float().cpu())
            r = torch.cat(ratio)
            if d1 is None:
                d1 = r
            better = 100 * (r < d1).float().mean()
            print(f"  {k:>3} {float(r.median()):13.6f} {better:17.0f}%"
                  f" {100*torch.cat(closer).mean():12.0f}%")
        print(f"  d_k/d_0 < 1 = 比原动作近。**比 k=1 更近的比例**要明显过半，"
              f"多步才算有意义。")

    if args.prefix:
        print(f"\n  只修前 k 步（整块 {chunk}×{U.shape[2]} 维会把信号稀释掉）")
        print(f"  {'前k步':>6} {'cos 中位':>10} {'>0比例':>8} {'天花板':>9} {'随机|cos|':>10}")
        for k in (1, 2, 5, 10, 25, chunk):
            u, s = U[:, :k].flatten(1), G[:, :k].flatten(1)
            ck = float(F.cosine_similarity(u, s, dim=1).median())
            pos = 100 * (F.cosine_similarity(u, s, dim=1) > 0).float().mean()
            dim = k * U.shape[2]
            print(f"  {k:6d} {ck:+10.4f} {pos:7.0f}% {100*(1-(1-ck*ck)**0.5):8.3f}%"
                  f" {1/dim**0.5:10.4f}")
        print(f"  「随机|cos|」= 该维度下两个随机向量夹角余弦的典型值。"
              f"实测的 cos 低于它，说明信号被噪声压着 —— 只有「>0 比例」证明它非零。")

    print(f"\n  **判据：cos 要到 0.1 量级才谈得上有用。** 0.02 量级的天花板是 0.03%，"
          f"真机测不出来（实测 A/B p = 0.785）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
