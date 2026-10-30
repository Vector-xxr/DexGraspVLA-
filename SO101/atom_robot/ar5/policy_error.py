#!/usr/bin/env python3
"""不上真机，量一个 checkpoint 的动作误差有多大。**训完先跑这个再决定要不要上机。**

    $L atom_robot/ar5/policy_error.py --ckpt ~/pi05_bolt/checkpoints/030000
    $L atom_robot/ar5/policy_error.py --ckpt A --ckpt B --repo-id vel_ar5_bolt_v2

拿训练集里的观测喂回策略，把它输出的动作块和人类真实动作比。真机评测一轮十几分钟
还受场景漂移影响；这个几分钟出结果，而且**把误差拆成两半**：

    系统误差    多次采样的均值 vs 人类。这是欠拟合，加步数会降
    采样随机性  同一观测采多次，各次之间的散度。**加步数基本降不动** ——
                它是模型对「这一刻该往哪走」的不确定性。哪怕均值完全正确，
                单次执行也会随机偏这么多

实测（旧 100 条数据训的 π0.5）：

    010000  2.2 epoch   系统 25.4 mm   随机 21.1 mm
    030000  6.6 epoch   系统 19.1 mm   随机 17.5 mm

螺栓宽 25 mm。两项加起来超过它，真机上就是抓空 —— 和实测的 10 次全败对上。

ACT 是确定性回归，采样随机性恒为 0，只看系统误差。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


POS = slice(0, 3)      # 位置增量，米 -> 毫米
ROT = slice(3, 6)      # 轴角增量，弧度 -> 度


def measure(ckpt: Path, ds, pre, post, policy, n_obs: int, n_draw: int, seed: int):
    """返回 {位置, 姿态} 两组 (系统误差, 采样随机性, 策略幅度, 人类幅度)。

    **姿态一定要单独量**，因为它和位置的容差完全不同：位置容差是螺栓宽度/套筒
    间隙那个量级，姿态容差是插入的角度余量。只报一个位置数会看不出是哪一项在卡。

    实测下来姿态不是瓶颈（040000 只有 1.7°，人类整块也才转 6.8°），位置才是 ——
    见 main() 末尾那几行的容差反推。

    **一个我推翻过的结论。** 曾经因为「18.0 -> 13.2 mm 而整体成功率 36.7% ->
    35.0%」就断言位置指标失效了。那个比较站不住：36.7% 是 n=30 的数，区间
    [21.9, 54.5]，35.0% 完全落在里面 —— 两个数根本分不开，说明不了任何事。
    要用它下结论，两边都得有足够的轮次。
    """
    import torch

    rng = np.random.default_rng(seed)
    sel = np.sort(rng.choice(len(ds), n_obs, replace=False))
    out = {"位置": [[], [], [], []], "姿态": [[], [], [], []]}
    for i in range(0, len(sel), 4):
        rows = [ds[int(j)] for j in sel[i:i + 4]]
        batch = {}
        for k, v in rows[0].items():
            if torch.is_tensor(v):
                batch[k] = torch.stack([r[k] for r in rows]).cuda()
            elif k == "task":
                batch[k] = [r[k] for r in rows]
        bb = pre(batch)
        chunk = policy.config.chunk_size
        exp_raw = torch.stack([r["action"] for r in rows]).numpy()[:, :chunk]

        draws = []
        for s in range(n_draw):
            torch.manual_seed(1000 + s)          # 固定每一次抽样，跨 checkpoint 可比
            policy.reset()
            with torch.no_grad():
                ch = policy.predict_action_chunk(bb)
            a = post(ch.reshape(-1, ch.shape[-1])).reshape(len(rows), chunk, -1)
            draws.append(a.float().cpu().numpy())
        draws = np.array(draws)                   # [n_draw, B, chunk, dim]

        for name, dims, unit in (("位置", POS, 1000.0), ("姿态", ROT, 180.0 / np.pi)):
            # 整块累计增量。轴角这么加不是严格的旋转复合，但策略和人类用的是
            # **同一套加法**，比较的是同一个量，够用来看差多少。
            expert = exp_raw[:, :, dims].sum(1) * unit
            D = draws[:, :, :, dims].sum(2) * unit          # [n_draw, B, 3]
            acc = out[name]
            acc[0].append(np.linalg.norm(D.mean(0) - expert, axis=-1))
            acc[1].append(np.linalg.norm(D - D.mean(0, keepdims=True), axis=-1).mean(0))
            acc[2].append(np.linalg.norm(D.mean(0), axis=-1))
            acc[3].append(np.linalg.norm(expert, axis=-1))
    return {k: tuple(np.concatenate(c) for c in v) for k, v in out.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", action="append", required=True,
                    help="可以给多个，逐个测完并排比较")
    ap.add_argument("--repo-id", default="vel_ar5_bolt_v2",
                    help="**要和 checkpoint 训练时用的那份一致**，否则量的是分布外表现")
    ap.add_argument("--n-obs", type=int, default=40, help="取多少个观测")
    ap.add_argument("--n-draw", type=int, default=4, help="每个观测采几次（ACT 给 1 就够）")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import torch

    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from atom_robot.ar5.eval_policy import load_policy

    print(f"  数据集 {args.repo_id}   {args.n_obs} 个观测 × {args.n_draw} 次采样")
    rows = []
    ds = None
    for c in args.ckpt:
        ck = Path(c).expanduser()
        policy, pre, post, _model_dir, kind = load_policy(ck, args.device)
        policy.eval()
        if ds is None:
            meta = LeRobotDataset(args.repo_id, video_backend="pyav").meta
            ds = LeRobotDataset(args.repo_id,
                                delta_timestamps=resolve_delta_timestamps(policy.config, meta),
                                video_backend="pyav")
        draws = 1 if kind == "act" else args.n_draw     # ACT 是确定性的，采多次没意义
        res = measure(ck, ds, pre, post, policy, args.n_obs, draws, args.seed)
        rows.append((ck.name, kind, res))
        del policy
        torch.cuda.empty_cache()

    for which, unit, ref in (("位置", "mm", "螺栓宽约 25 mm"),
                             ("姿态", "°", "套筒插入的角度余量")):
        print(f"\n  === {which} ===")
        print(f"  {'checkpoint':>12} {'类型':>6} {'系统误差':>11} {'采样随机性':>12}"
              f" {'两项和':>9} {'整块量 策略/人类':>20}")
        for name, kind, res in rows:
            s, sp, mp, me = (np.median(x) for x in res[which])
            print(f"  {name:>12} {kind:>6} {s:8.1f} {unit:<2} {sp:9.1f} {unit:<2}"
                  f" {s+sp:7.1f} {unit:<2} {mp:8.1f} /{me:6.1f} {unit}")
        print(f"  参照：{ref}")

    print(f"\n  系统误差 = 多次采样的均值离人类动作多远。欠拟合，加步数会降。")
    print(f"  采样随机性 = 同一观测各次采样之间的散度。**加步数基本降不动。**")
    print(f"\n  **和什么比。** 抓取瞄的是 25 mm 宽的螺栓；插入的容差用示教反推："
          f"人松手时的位置散度 ±7 mm 而 296 条全成功，所以插入容差至少 ±7~10 mm。")
    print(f"  实测 040000（11.6 mm 那档之前）真机 100 轮：抓取 82.0%、插入|抓到 43.9%。"
          f"位置误差正卡在插入容差边缘，抓取那边则宽裕得多 —— 两个成功率的差距"
          f"和这个容差比对得上。")
    print(f"  **别拿相邻两档的小差值下结论。** n=40 个观测取中位，0.5 mm 上下的"
          f"差异分不出是真降还是噪声 —— 我照着 6.5 -> 6.1 判过一次「采样随机性"
          f"见底」，下一档 5.4 就打脸了。要判平台期，把 --n-obs 加到 120 以上。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
