#!/usr/bin/env python3
"""VEL 第二阶段的第 0 步：把 backbone 的 h_t^(-1) 算一遍存下来。

    $L atom_robot/ar5/vel/precompute.py --ckpt ~/pi05_bolt/checkpoints/010000

**为什么要缓存。** 第二阶段 backbone 是冻结的，所以同一帧的 h 永远一样，
每个训练步重算是纯浪费。而它很贵：实测 prefix 前向 **15.8 样本/秒，且与 batch 无关**
（5070Ti，eager attention）。论文规模 200k 步 × batch 64 = 1280 万样本，
现算要 225 小时；缓存一次 10 分钟，之后训练只剩 3.96M 参数的能量头，是秒级的事。

**为什么抽稀。** 全量 36414 帧要 106 GB。30 Hz 下相邻帧几乎一样，`--stride 4`
（7.5 Hz）取 9104 帧、26.6 GB，信息损失可以忽略 —— InfoNCE 的负样本来自
**不同场景**，不是来自相邻帧。

存两样，`.npy` memmap：

    h.npy     [N, 712, 2048] float16   backbone 最后一层 hidden
    mask.npy  [N, 712]       uint8     True = 有效 token
    a.npy     [N,  50,    7] float32   专家动作块，**归一化空间**
    meta.json

动作必须存归一化后的：能量头和推理时的修正都活在归一化空间里，
存原始值会让训练和推理对不上。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

# 2026-09-23 代码从 VEL 仓库搬到 Project_Atom，按目录层数猜根目录那套就错了 ——
# 而且错得**没有报错**，只是路径不存在、import 静默失败或者文档静默不检查。
# 改成走 atom_robot.VEL_REPO（可用环境变量 VEL_REPO 覆盖）。
from atom_robot import VEL_REPO as _VEL
sys.path.insert(0, str(_VEL))

DEFAULT_OUT = Path.home() / "vel_cache" / "pi05_bolt_010000"


def normalizer_of(preprocessor):
    """从推理用的 preprocessor 里挑出归一化那一步，用它归一化专家动作。

    必须复用策略自己的那一份，不能另算一套统计量 —— 差一点，能量头学到的
    谷底就不在策略实际输出的那个空间里。
    """
    for step in preprocessor.steps:
        if "ormaliz" in type(step).__name__:
            return step
    raise RuntimeError("preprocessor 里没有归一化步骤，检查 lerobot 版本")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="~/pi05_bolt/checkpoints/010000")
    ap.add_argument("--repo-id", default="vel_ar5_bolt")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--stride", type=int, default=4, help="每几帧取一个（默认 4 = 7.5 Hz）")
    ap.add_argument("--n-samples", type=int, default=0, metavar="K",
                    help="每帧另存 K 个 **π0.5 自己的采样** 到 a_bc.npy。0 = 不存。\n"
                         "**这是让能量头在「策略真正所在的地方」受训的前提。**\n"
                         "实测：在 a* 周围的随机扰动点上训方向，cos 能到 0.199，\n"
                         "但在真实 a_BC 上量还是 0.017 —— 350 维里两处几乎正交。\n"
                         "成本很低：prefix 共享，每多一个采样只加一轮去噪循环。")
    ap.add_argument("--batch", type=int, default=8, help="吞吐和 batch 无关，这个只影响显存")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from atom_robot.ar5.eval_policy import load_policy
    from atom_robot.ar5.vel.pi05_bridge import Pi05Features

    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    policy, pre, _post, model_dir, kind = load_policy(Path(args.ckpt).expanduser(), args.device)
    if kind != "pi05":
        print(f"  这个脚本只处理 pi05，checkpoint 是 {kind}")
        return 1
    policy.eval()
    for p in policy.parameters():
        p.requires_grad_(False)
    feat = Pi05Features(policy)
    norm = normalizer_of(pre)

    # 用官方 helper 解出 delta_timestamps，保证动作块的切法和训练时逐帧一致
    meta = LeRobotDataset(args.repo_id, video_backend="pyav").meta
    dts = resolve_delta_timestamps(policy.config, meta)
    ds = LeRobotDataset(args.repo_id, delta_timestamps=dts, video_backend="pyav")
    idx = np.arange(0, len(ds), args.stride)
    n, chunk, adim = len(idx), policy.config.chunk_size, feat.act_dim
    print(f"  数据集 {len(ds)} 帧 -> 每 {args.stride} 帧取一个 = {n} 个样本")
    print(f"  动作块 [{chunk}, {adim}]   hidden [?, {feat.hidden_dim}]")

    h_mm = a_mm = m_mm = bc_mm = None
    t0 = time.perf_counter()
    for start in range(0, n, args.batch):
        sel = idx[start:start + args.batch]
        rows = [ds[int(i)] for i in sel]
        batch = {}
        for k in rows[0]:
            v = rows[0][k]
            if torch.is_tensor(v):
                batch[k] = torch.stack([r[k] for r in rows]).to(args.device)
            elif k == "task":
                batch[k] = [r[k] for r in rows]
        batch = pre(batch)

        with torch.inference_mode():
            h, pad, bc = feat.sample(batch, n=args.n_samples)
            a = norm({"action": torch.stack([r["action"] for r in rows]).to(args.device)})["action"]

        if h_mm is None:
            S, D = h.shape[1], h.shape[2]
            h_mm = np.lib.format.open_memmap(out / "h.npy", "w+", np.float16, (n, S, D))
            m_mm = np.lib.format.open_memmap(out / "mask.npy", "w+", np.uint8, (n, S))
            a_mm = np.lib.format.open_memmap(out / "a.npy", "w+", np.float32, (n, chunk, adim))
            print(f"  h [{n}, {S}, {D}] float16 = {n*S*D*2/1e9:.1f} GB -> {out}")
            if args.n_samples:
                K = args.n_samples
                bc_mm = np.lib.format.open_memmap(out / "a_bc.npy", "w+", np.float32,
                                                  (n, K, chunk, adim))
                print(f"  a_bc [{n}, {K}, {chunk}, {adim}] float32 = "
                      f"{n*K*chunk*adim*4/1e6:.0f} MB（相对 h 可忽略）")

        j = slice(start, start + len(sel))
        if args.n_samples:
            bc_mm[j] = bc[:, :, :chunk, :adim].float().cpu().numpy()
        h_mm[j] = h.float().cpu().numpy().astype(np.float16)
        m_mm[j] = pad.cpu().numpy().astype(np.uint8)
        a_mm[j] = a[:, :chunk, :adim].float().cpu().numpy()

        done = start + len(sel)
        if start % (args.batch * 20) == 0 or done == n:
            el = time.perf_counter() - t0
            print(f"\r  {done}/{n}   {done/max(el,1e-9):.1f} 样本/秒   "
                  f"剩 {(n-done)/max(done/max(el,1e-9),1e-9)/60:.1f} 分钟   ", end="", flush=True)

    for mm in (h_mm, m_mm, a_mm, bc_mm):
        if mm is not None:
            mm.flush()
    (out / "meta.json").write_text(json.dumps({
        "ckpt": str(Path(args.ckpt).expanduser()), "repo_id": args.repo_id,
        "stride": args.stride, "n": int(n),
        "hidden_dim": int(h_mm.shape[2]), "seq_len": int(h_mm.shape[1]),
        "chunk": int(chunk), "act_dim": int(adim), "n_samples": int(args.n_samples),
    }, indent=1))
    print(f"\n  写好了 {out}   用时 {(time.perf_counter()-t0)/60:.1f} 分钟")
    print(f"  下一步：$L atom_robot/ar5/vel/train_energy.py --cache {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
