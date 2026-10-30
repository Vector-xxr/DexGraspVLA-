#!/usr/bin/env python3
"""把 `eval_policy.py` 逐轮写的 CSV 统计成表。**在 vel-ar5 环境里跑，不用 GPU。**

    $V -m atom_robot.ar5.rollout_stats
    $V -m atom_robot.ar5.rollout_stats --ab fixed vel-fixed

分组按 (tag, checkpoint, vel)，三者任一不同就是不同的实验条件。

**每一段的成功率分开看。** 能量头是按段训的（抓取段一个、放置段一个）；
整体成功率里混着后面各段的失败，会把单段的改善稀释掉。老 CSV 那些列是空的
（那时候还没记），这些行只统计整体 —— 空不等于失败，不能猜。

置信区间用 Wilson 区间，不用正态近似 —— n=30、p 在 0.2~0.4 时正态近似会给出
超出 [0,1] 的下界。两组比较用 Fisher 精确检验，不用卡方 —— 30 轮的格子太小。

**先看区间宽度再看点估计。** n=30 时 95% 区间大约 ±17 个百分点，
两个条件差 5~10 个点基本没有意义。
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path


# 每段各算各的。**「插入」的分母是抓到了的那些轮**（CSV 里没抓到的行这一列
# 是空的），不是全部 —— 否则一个只改善了抓取段的头，会在插入段也显出「提升」，
# 因为它多送了一批本来根本走不到插入的轮次进去。
SEGMENT_FIELDS = ("抓取", "插入", "结果")
SEGMENT_LABELS = ("抓取", "插入|抓到", "整体")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 得分区间。n 小、p 靠近 0 或 1 时比正态近似可靠得多。"""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def fisher(a: int, b: int, c: int, d: int) -> float:
    """2×2 双尾 Fisher 精确检验。表格是 [[a, b], [c, d]]。"""
    from math import comb

    n = a + b + c + d
    if min(a + b, c + d, a + c, b + d) == 0:
        return 1.0

    def pt(x: int) -> float:
        b_, c_ = (a + b) - x, (a + c) - x
        d_ = n - x - b_ - c_
        if min(b_, c_, d_) < 0:
            return 0.0
        return comb(a + b, x) * comb(c + d, c_) / comb(n, a + c)

    p0 = pt(a)
    return min(1.0, sum(pt(x) for x in range(0, min(a + b, a + c) + 1)
                        if pt(x) <= p0 + 1e-12))


def load(path: Path):
    if not path.exists():
        print(f"  没有 {path} —— 先跑几轮 eval_policy，按 r/g/f 记结果")
        return None
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", default=str(Path.home() / "vel_runs" / "rollouts.csv"))
    ap.add_argument("--ab", nargs=2, metavar=("TAG_A", "TAG_B"),
                    help="对这两个 tag 做 Fisher 精确检验")
    args = ap.parse_args()

    rows = load(Path(args.log).expanduser())
    if rows is None:
        return 1

    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r.get("tag") or "—", r.get("checkpoint") or "?",
                r.get("vel") or "—")].append(r)

    print(f"  {len(rows)} 轮   {args.log}\n")
    print(f"  {'tag':<13}{'ckpt':>8}{'vel':>30}{'n':>4}"
          f"{'抓取':>17}{'插入|抓到':>17}{'整体':>17}")
    for key, rs in groups.items():
        cols = [f"{len(rs):>4}"]
        for field in SEGMENT_FIELDS:
            # 空格 = 这一段没发生（前一段就失败了），不能当失败算进分母。
            # 老 CSV 整列都是空的 —— 那是「没记」，同样不能猜成失败。
            sub = [r for r in rs if r.get(field)]
            if not sub:
                cols.append(f"{'未标注':>17}")
                continue
            k = sum(1 for r in sub if r[field] == "成功")
            lo, hi = wilson(k, len(sub))
            cols.append(f"{100*k/len(sub):5.1f}%[{100*lo:4.1f},{100*hi:4.1f}]n{len(sub):<3}")
        print(f"  {key[0]:<13}{key[1]:>8}{key[2]:>30}" + "".join(cols))

    if args.ab:
        print()
        for label, field in zip(SEGMENT_LABELS, SEGMENT_FIELDS):
            got = []
            for tag in args.ab:
                sub = [r for r in rows if (r.get("tag") or "—") == tag and r.get(field)]
                got.append((tag, sum(1 for r in sub if r[field] == "成功"), len(sub)))
            (ta, ka, na), (tb, kb, nb) = got
            if min(na, nb) == 0:
                print(f"  {label:<10} {ta if na == 0 else tb} 没有可用的标注，跳过")
                continue
            p = fisher(ka, na - ka, kb, nb - kb)
            verdict = "**有显著差异**" if p < 0.05 else "分不开（差异在噪声里）"
            print(f"  {label:<10} {ta} {ka}/{na} = {100*ka/na:5.1f}%   vs   "
                  f"{tb} {kb}/{nb} = {100*kb/nb:5.1f}%   "
                  f"p = {p:.3f}   {verdict}")

    print(f"\n  「插入|抓到」的分母是**抓到了的那些轮**，不是全部 —— 否则抓取段的"
          f"改善会伪装成插入段的改善。")
    print(f"  n=30 时 95% 区间约 ±17 个百分点。**先看区间宽度再看点估计。**")
    return 0


if __name__ == "__main__":
    sys.exit(main())
