#!/usr/bin/env python3
"""量螺栓和螺套在桌面上摆得够不够散。**每采 50 条跑一次，别等采完 300 条才发现。**

    $V -m atom_robot.ar5.object_spread                    # 全部
    $V -m atom_robot.ar5.object_spread --from 100         # 只看新采的
    $V -m atom_robot.ar5.object_spread --from 100 --save  # 顺便存检测图

看三件事：

  1. **各自的跨度**  螺栓和螺套分别摆了多大范围
  2. **两者的相关**  ⚠️ 这条最容易忽略。要是「螺栓越靠左、螺套就越靠右」，
     那两个物体只提供了一个自由度 —— 模型找到一个就知道另一个在哪。
     **两次摆放要互相独立，相关系数接近 0 才对。**
  3. **有没有空洞**  直方图里连续几格是 0，说明那一片从来没摆过

怎么检测（两个物体用两套方法，都实测 100/100）：

  螺套  蓝色，色相 95~128 直接找连通域
  螺栓  **背景差分**。白色六角头会跟桌面反光连成一片，纯颜色阈值实测只有
        62~91/100，而且漏的**系统性集中在右侧**（侧立时头看着更小更暗）——
        那样量出来的分布是偏的。改成拿 100 条首帧的中位图当静态背景
        （桌面、胶带、反光都是静止的，只有物体在动），差分出来 100/100。
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# 桌面上 1 像素约等于多少毫米。俯视机位固定，这个值只有换机位才要重标。
# 想标准的：拿尺量黄胶带外框的实际宽度，用 --tape-mm 传进来。
MM_PER_PX = 0.71


def first_frames(files, cam="cam_high"):
    import cv2

    out = []
    for f in files:
        cap = cv2.VideoCapture(f"{f[:-4]}_{cam}.mp4")
        ok, fr = cap.read()
        cap.release()
        out.append(fr if ok else None)
    return out


def tape_box(bgr):
    """黄胶带外框。用来换算 mm/px，也顺便看相机有没有动。"""
    import cv2

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (18, 110, 110), (35, 255, 255))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    c, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return cv2.boundingRect(max(c, key=cv2.contourArea)) if c else None


def find_sleeve(bgr):
    import cv2

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (95, 110, 60), (128, 255, 255))
    n, _l, st, ce = cv2.connectedComponentsWithStats(m, 8)
    b = max(range(1, n), key=lambda i: st[i, 4], default=None)
    if b is None or st[b, 4] < 300:
        return None
    return float(ce[b][0]), float(ce[b][1]), tuple(int(v) for v in st[b, :4])


def find_bolt(bgr, bg, ytop=140, thr=45):
    """背景差分。见模块 docstring —— 纯颜色阈值会被桌面反光带偏。"""
    import cv2

    d = cv2.absdiff(bgr, bg).max(2)
    d[:ytop] = 0                                  # 上方是机械臂
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    blue = cv2.dilate(cv2.inRange(hsv, (95, 90, 40), (130, 255, 255)),
                      np.ones((15, 15), np.uint8))
    d[blue > 0] = 0                               # 螺套也会动一点，排掉
    m = (d > thr).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    n, _l, st, ce = cv2.connectedComponentsWithStats(m, 8)
    cand = [i for i in range(1, n) if 600 <= st[i, 4] <= 15000]
    if not cand:
        return None
    b = max(cand, key=lambda i: st[i, 4])
    return float(ce[b][0]), float(ce[b][1]), tuple(int(v) for v in st[b, :4])


def _structural_r(s, b, n, trials=300):
    """顺序约束本身会造成多少正相关。蒙特卡洛，不是理论值。

    模拟：螺套在它实测的区间里均匀取；螺栓在「螺套 + 实测最小间距」到它实测的
    右边界之间均匀取。两次抽签互相独立 —— 这就是「理想独立摆法」，它的 r 就是
    躲不掉的那部分。实测 r 减去它，才是操作者引入的相关。

    顺序不成立（比如 y 方向两个物体没有固定上下关系）时返回 None，退回简单判据。
    """
    gap = float(np.min(b - s))
    if gap <= 0:                      # 没有稳定的顺序关系
        return None, None
    lo, hi = float(s.min()), float(s.max())
    top = float(b.max())
    rng = np.random.default_rng(0)
    out = []
    for _ in range(trials):
        ss = rng.uniform(lo, hi, n)
        bb = np.array([rng.uniform(min(x + gap, top - 1e-6), top) for x in ss])
        if bb.std() < 1e-6 or ss.std() < 1e-6:
            return None, None
        out.append(np.corrcoef(ss, bb)[0, 1])
    return float(np.mean(out)), float(np.std(out))


def bar(v, lo, hi, n=14, width=42):
    h, edges = np.histogram(v, bins=n, range=(lo, hi))
    top = max(h.max(), 1)
    lines = []
    for i in range(n):
        mark = "·" if h[i] == 0 else "█" * max(1, int(width * h[i] / top))
        lines.append(f"      {edges[i]:6.0f}~{edges[i+1]:6.0f}  {mark:<{width}} {h[i]:3d}")
    return lines


def report(name, xy, mmpp):
    x, y = xy[:, 0], xy[:, 1]
    print(f"\n  ── {name}（{len(xy)} 条）──")
    print(f"    x  跨度 {np.ptp(x):5.0f} px = {np.ptp(x)*mmpp:5.0f} mm    "
          f"std {x.std():5.1f} px = ±{x.std()*mmpp:4.0f} mm")
    print(f"    y  跨度 {np.ptp(y):5.0f} px = {np.ptp(y)*mmpp:5.0f} mm    "
          f"std {y.std():5.1f} px = ±{y.std()*mmpp:4.0f} mm")
    print(f"    x 分布:")
    for line in bar(x, x.min(), x.max()):
        print(line)
    empty = sum(1 for c in np.histogram(x, bins=14, range=(x.min(), x.max()))[0] if c == 0)
    if empty >= 3:
        print(f"    ⚠ 有 {empty}/14 格是空的 —— 那些位置从来没摆过")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="~/atom_episodes")
    ap.add_argument("--from", dest="start", type=int, default=0,
                    help="只看编号 >= 这个的（续采时看新的那批）")
    ap.add_argument("--tape-mm", type=float, default=None,
                    help="黄胶带外框实际宽度(mm)，给了就用它标定，不给用 %.2f mm/px" % MM_PER_PX)
    ap.add_argument("--save", action="store_true", help="存一张检测图，人眼核对")
    args = ap.parse_args()

    import cv2

    src = os.path.expanduser(args.src)
    files = sorted(glob.glob(f"{src}/episode_*.npz"))
    files = [f for f in files if int(os.path.basename(f)[8:12]) >= args.start]
    if len(files) < 5:
        print(f"  {src} 里编号 >= {args.start} 的只有 {len(files)} 条，太少")
        return 1
    print(f"  {src}   {len(files)} 条"
          f"（{os.path.basename(files[0])[:-4]} ~ {os.path.basename(files[-1])[:-4]}）")

    frames = first_frames(files)
    good = [f for f in frames if f is not None]
    if len(good) < len(frames):
        print(f"  ⚠ {len(frames)-len(good)} 条读不到 cam_high 首帧")
    bg = np.median(np.array(good), axis=0).astype(np.uint8)

    box = tape_box(bg)
    mmpp = MM_PER_PX
    if box and args.tape_mm:
        mmpp = args.tape_mm / box[2]
        print(f"  胶带外框 {box[2]} px = {args.tape_mm:.0f} mm  ->  {mmpp:.3f} mm/px")
    else:
        print(f"  按 {mmpp:.2f} mm/px 换算（想更准就量一下胶带外框宽度，用 --tape-mm 传）")

    S, B, miss = [], [], [0, 0]
    for fr in frames:
        if fr is None:
            continue
        s, b = find_sleeve(fr), find_bolt(fr, bg)
        S.append(s[:2] if s else None); miss[0] += s is None
        B.append(b[:2] if b else None); miss[1] += b is None
    Sx = np.array([p for p in S if p], float)
    Bx = np.array([p for p in B if p], float)
    print(f"  检出  螺套 {len(Sx)}/{len(frames)}   螺栓 {len(Bx)}/{len(frames)}")

    report("螺套（放置目标）", Sx, mmpp)
    report("螺栓（抓取目标）", Bx, mmpp)

    # ── 两个物体的位置是不是相关的 ──
    #
    # ⚠️ **不能拿 0 当目标。** 「螺套始终在螺栓左边」这个约束本身就强制 r 为正：
    # 螺套越靠右，螺栓被迫更靠右。所以基线要现算 —— 蒙特卡洛模拟一个完全独立
    # 的摆法（各自在自己的可行区间里均匀取，只保证顺序和最小间距），
    # 看实测比它高多少。实测 +0.383 vs 结构性 +0.632，是**好于**理想独立摆法的。
    both = [(s, b) for s, b in zip(S, B) if s and b]
    if len(both) >= 10:
        s_ = np.array([p[0] for p in both], float)
        b_ = np.array([p[1] for p in both], float)
        print(f"\n  ── 两者独立吗（{len(both)} 条都检出）──")
        for j, ax in enumerate("xy"):
            r = float(np.corrcoef(s_[:, j], b_[:, j])[0, 1])
            base, spread = _structural_r(s_[:, j], b_[:, j], len(both))
            if base is None:
                verdict = "✓ 独立" if abs(r) < 0.3 else "⚠ 有相关"
                print(f"    螺套 {ax}  vs  螺栓 {ax}    r = {r:+.3f}   {verdict}")
                continue
            # 2σ 起报。1σ 在 40 条一批的时候 σ≈0.10，误报率约 16% ——
            # 实测 +0.701 vs 基线 +0.597±0.098 就被误判成「相关得多」，
            # 而那只是 1.06σ 的抽样噪声。
            excess = r - base
            verdict = ("✓ 不差于理想独立摆法" if excess < 2 * spread else
                       "⚠ 比理想独立摆法相关得多，摆的时候两个别互相看" if excess < 4 * spread else
                       "✗ 强相关 —— 两个物体只提供了一个自由度")
            print(f"    螺套 {ax}  vs  螺栓 {ax}    r = {r:+.3f}   "
                  f"（顺序约束下的结构性基线 {base:+.3f}±{spread:.3f}）  {verdict}")
        print(f"    要点是别让两次摆放互相参考；顺序约束带来的那部分正相关是躲不掉的")

    if args.save:
        out = []
        for k in np.linspace(0, len(frames) - 1, 6).astype(int):
            fr = frames[k]
            if fr is None:
                continue
            v = fr.copy()
            s, b = find_sleeve(fr), find_bolt(fr, bg)
            if s:
                x, y, w, h = s[2]
                cv2.rectangle(v, (x, y), (x + w, y + h), (255, 0, 0), 2)
            if b:
                x, y, w, h = b[2]
                cv2.rectangle(v, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(v, os.path.basename(files[k])[:-4], (10, 28), 0, 0.8, (0, 255, 255), 2)
            out.append(cv2.resize(v, (426, 320)))
        p = Path(src).expanduser() / "object_spread.jpg"
        cv2.imwrite(str(p), np.vstack([np.hstack(out[:3]), np.hstack(out[3:6])]))
        print(f"\n  检测图 {p}   蓝框=螺套 红框=螺栓，**人眼核对一下框对不对**")
    return 0


if __name__ == "__main__":
    sys.exit(main())
