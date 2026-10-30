#!/usr/bin/env python3
"""采完一批数据跑一遍，把能自动查的都查掉。只读文件，不碰硬件。

    python -m atom_robot.ar5.qc_dataset
    python -m atom_robot.ar5.qc_dataset --src ~/atom_episodes --verbose

查九项：丢帧 / 重复帧 / 画面丢失 / 帧率 / 卡顿 / 相机漂移 / 人手误入 /
有效动作 / 场景多样性。

**有两件它查不出来，只能你自己回放看**：
  * 人手误入 —— 肤色阈值和黄胶带撞色，实测 100 条里报的全是胶带和白螺栓
  * 操作对不对 —— 抓没抓稳、放没放进去，这个没有自动判据
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def scan(files, names, verbose=False):
    """逐帧扫一遍。返回每条 episode 的指标。"""
    import cv2

    out = []
    for k, f in enumerate(files):
        d = np.load(f, allow_pickle=True)
        rec = {
            "name": os.path.basename(f)[:-4],
            "n": len(d["state"]),
            "state": d["state"], "action": d["action"], "ee": d["ee_pose"],
            "fps": float(d["fps"]),
            "fps_measured": float(d["fps_measured"]) if "fps_measured" in d else None,
            "stamps": d["timestamp"] if "timestamp" in d else None,
            "task": str(d["task"]),
            "cams": {},
        }
        for cam in names:
            cap = cv2.VideoCapture(f"{f[:-4]}_{cam}.mp4")
            diffs, bright, std, skin, first = [], [], [], [], None
            prev = None
            while True:
                ok, fr = cap.read()
                if not ok:
                    break
                s = cv2.resize(fr, (160, 120))
                g = cv2.cvtColor(s, cv2.COLOR_BGR2GRAY).astype(np.float32)
                if first is None:
                    first = g
                if prev is not None:
                    diffs.append(float(np.abs(g - prev).mean()))
                prev = g
                bright.append(float(g.mean()))
                std.append(float(g.std()))
                hsv = cv2.cvtColor(s, cv2.COLOR_BGR2HSV)
                skin.append(float((cv2.inRange(hsv, (0, 40, 60), (25, 180, 255)) > 0).mean()))
            cap.release()
            rec["cams"][cam] = {
                "n": len(bright),
                "diffs": np.array(diffs, np.float32),
                "bright": np.array(bright, np.float32),
                "std": np.array(std, np.float32),
                "skin": np.array(skin, np.float32),
                "first": first,
            }
        out.append(rec)
        if verbose:
            print(f"\r  扫描 {k+1}/{len(files)}", end="", flush=True)
    if verbose:
        print()
    return out


def landmark_xy(gray, lo, hi):
    """在缩略灰度图上找不到颜色，所以这个只在彩色首帧上用。留给 drift 用。"""
    return None


def check(recs, names, thresh_px=8.0):
    """把指标变成结论。返回 (问题清单, 摘要行)。"""
    import cv2

    problems = []
    n_eps = len(recs)
    print(f"\n  {n_eps} 条，共 {sum(r['n'] for r in recs)} 帧\n")

    # 1 丢帧
    bad = [r["name"] for r in recs
           if any(c["n"] != r["n"] for c in r["cams"].values())]
    print(f"  1 丢帧        npz 与各路 mp4 帧数：{'✓ 全部一致' if not bad else f'✗ {bad}'}")
    if bad:
        problems.append(f"帧数对不上：{bad}")

    L = np.array([r["n"] for r in recs])
    print(f"                帧数 {L.min()}~{L.max()}  中位 {int(np.median(L))}"
          f"  = {np.median(L)/30:.1f} 秒")

    # 2 重复帧：帧间差恰好为 0 = 同一 buffer 被读两次
    for cam in names:
        z = sum(int((r["cams"][cam]["diffs"] == 0).sum()) for r in recs)
        t = sum(len(r["cams"][cam]["diffs"]) for r in recs)
        flag = "✓" if z == 0 else f"✗ {z}"
        print(f"  2 重复帧      {cam:10} {flag} / {t} 帧")
        if z:
            problems.append(f"{cam} 有 {z} 帧重复")

    # 3 画面丢失
    for cam in names:
        blk = sum(int(((r["cams"][cam]["bright"] < 10) |
                       (r["cams"][cam]["std"] < 3)).sum()) for r in recs)
        b = np.concatenate([r["cams"][cam]["bright"] for r in recs])
        print(f"  3 画面丢失    {cam:10} {'✓ 0 黑帧' if not blk else f'✗ {blk} 黑帧'}"
              f"   亮度 {b.min():.0f}~{b.max():.0f}")
        if blk:
            problems.append(f"{cam} 有 {blk} 个黑帧")

    # 4 帧率
    have = [r for r in recs if r["fps_measured"]]
    if have:
        m = np.array([r["fps_measured"] for r in have])
        nominal = have[0]["fps"]
        print(f"  4 帧率        实测 {m.min():.2f}~{m.max():.2f} Hz  中位 {np.median(m):.2f}"
              f"   标称 {nominal:.0f}  达成 {np.median(m)/nominal*100:.1f}%")
        slow = [r["name"] for r in have if r["fps_measured"] < nominal * 0.9]
        if slow:
            problems.append(f"帧率低于标称 90% 的：{slow}")
        stamps = [np.diff(r["stamps"]) * 1000 for r in recs if r["stamps"] is not None]
        if stamps:
            dt = np.concatenate(stamps)
            over = (dt > 250).sum()
            print(f"                逐帧间隔 p50 {np.median(dt):.1f} p99 "
                  f"{np.percentile(dt,99):.1f} 最大 {dt.max():.0f} ms"
                  f"   >250ms（臂 latch 线）{over}")
            if over:
                problems.append(f"有 {over} 帧间隔超过 250 ms —— 臂可能 latch 过")
    else:
        print(f"  4 帧率        ⚠ 数据里没有 fps_measured，测不了")

    # 5 卡顿
    for cam in names:
        hit = 0
        for r in recs:
            d = r["cams"][cam]["diffs"] < 0.1
            c = 0
            for v in d:
                c = c + 1 if v else 0
                if c >= 5:
                    hit += 1
                    break
        print(f"  5 卡顿        {cam:10} 连续≥5 帧几乎不变的 episode "
              f"{'✓ 0' if not hit else f'✗ {hit}'} / {n_eps}")
        if hit:
            problems.append(f"{cam} 有 {hit} 条卡顿")

    # 6 相机漂移
    #
    # **只在「跨 episode 不变的像素」上量。** 整帧相位相关会被两样东西带偏：
    # 臂在首帧的位置（起始位姿本来就有 ±4.8° 的自然离散）和物体的摆位。
    # 实测 episode_0125 整帧报 73 px，其中上部（臂）73 px、下部（空桌面）0.5 px
    # —— 相机根本没动，而中位背景图比对只有 0.97 px。
    #
    # 静态区自己算出来：跨所有首帧的逐像素标准差低的地方就是不变的地方。
    # 不硬编码坐标，换机位、改布局都不用跟着改。
    print(f"  6 相机漂移    只在跨 episode 不变的像素上量（臂和物体的位置会带偏整帧）")
    for cam in names:
        firsts = [r["cams"][cam]["first"] for r in recs if r["cams"][cam]["first"] is not None]
        if len(firsts) < 5:
            continue
        stack = np.array(firsts, np.float32)
        static = stack.std(0) < np.percentile(stack.std(0), 40)   # 最稳的 40% 像素
        bg = np.median(stack, 0)
        shifts = []
        for r in recs:
            f = r["cams"][cam]["first"]
            if f is None:
                continue
            a = np.where(static, bg, 0).astype(np.float64)
            b = np.where(static, f, 0).astype(np.float64)
            (dx, dy), _ = cv2.phaseCorrelate(a, b)
            shifts.append((r["name"], np.hypot(dx, dy) * 4))      # 160x120 -> 640x480
        mag = np.array([s[1] for s in shifts])
        worst = max(shifts, key=lambda s: s[1])
        ok = mag.max() < thresh_px
        print(f"                {cam:10} 相对静态背景 中位 {np.median(mag):.1f} "
              f"最大 {mag.max():.1f} px  （静态像素 {100*static.mean():.0f}%）"
              f"  {'✓ 没动' if ok else f'⚠ 最大在 {worst[0]}'}")
        if not ok:
            problems.append(f"{cam} 首帧最大漂移 {mag.max():.0f} px（{worst[0]}）")

    # 7 人手误入
    for cam in names:
        mx = np.array([r["cams"][cam]["skin"].max() for r in recs])
        top = sorted(zip([r["name"] for r in recs], mx), key=lambda x: -x[1])[:3]
        print(f"  7 人手误入    {cam:10} 肤色峰值 中位 {np.median(mx)*100:.1f}% "
              f"最大 {mx.max()*100:.1f}%   最高三条 {[f'{n}:{v*100:.0f}%' for n,v in top]}")
    print(f"                ⚠ 这个判据不可靠 —— 肤色阈值和黄胶带撞色。**要靠回放确认**")

    # 8 有效动作
    lead, tail, mid, tot, useful = [], [], [], [], []
    for r in recs:
        a = r["action"]
        step = np.linalg.norm(a[:, :3], axis=1) * 1000
        dgrip = np.abs(np.diff(a[:, 6], prepend=a[0, 6]))
        act = (step > 0.4) | (dgrip > 0.002)
        k = np.convolve(act.astype(float), np.ones(5) / 5, mode="same") > 0.4
        n = len(a)
        if not k.any():
            lead.append(n); tail.append(0); mid.append(0); tot.append(n); useful.append(0)
            continue
        first, last = int(np.argmax(k)), n - 1 - int(np.argmax(k[::-1]))
        lead.append(first); tail.append(n - 1 - last)
        mid.append(int((~k[first:last + 1]).sum()))
        tot.append(n); useful.append(int(k.sum()))
    lead, tail, mid, tot, useful = map(np.array, (lead, tail, mid, tot, useful))
    print(f"  8 有效动作    {useful.sum()/tot.sum()*100:.1f}%"
          f"   开头空转 {lead.sum()/tot.sum()*100:.1f}%"
          f"  结尾 {tail.sum()/tot.sum()*100:.1f}%"
          f"  中间停顿 {mid.sum()/tot.sum()*100:.1f}%")

    # 9 场景多样性：起始末端位置的离散度。固定场景训不出泛化。
    S = np.array([r["ee"][0, :3] for r in recs]) * 1000
    print(f"  9 多样性      起始末端 std (x,y,z) = "
          f"({S[:,0].std():.0f}, {S[:,1].std():.0f}, {S[:,2].std():.0f}) mm"
          f"   跨度 ({np.ptp(S[:,0]):.0f}, {np.ptp(S[:,1]):.0f}, {np.ptp(S[:,2]):.0f}) mm")

    # 夹爪。**不要拿「有没有用满 0~1」当判据。**
    #
    # 原来那条报「只用到 0.79，上面 21% 量程没出现过」，查下来是误报：标定的
    # gripper_open_deg=+55.5 是够得着的（实测有一条到 0.932 = 扳机 +47.9°），
    # 只是螺栓不大，张到 79% 就够让开了。**任务用不到量程顶部不是缺陷。**
    #
    # 真正要查的是这三件：闭合端到底没有（抓取全靠它）、开合分不分得开、
    # 每条内部的行程够不够。量程顶部空着无所谓，归一化会自己适配。
    grip_max = np.array([r["action"][:, 6].max() for r in recs])
    grip_min = np.array([r["action"][:, 6].min() for r in recs])
    span = grip_max - grip_min
    allg = np.concatenate([r["action"][:, 6] for r in recs])
    closed_frac = float((allg < 0.05).mean())
    print(f"                夹爪 逐条行程 中位 {np.median(span):.2f}"
          f"（{np.median(grip_min):.2f} ~ {np.median(grip_max):.2f}）"
          f"   闭合到底的帧 {closed_frac*100:.0f}%")
    if np.median(span) < 0.35:
        problems.append(f"夹爪逐条行程只有 {np.median(span):.2f}，开合几乎没动过")
    if closed_frac < 0.03:
        problems.append(f"只有 {closed_frac*100:.1f}% 的帧夹爪闭到底 —— 抓取可能没真正合拢")
    if np.median(grip_min) > 0.15:
        problems.append(f"夹爪最小值中位 {np.median(grip_min):.2f}，从来没完全闭合过")

    tasks = {r["task"] for r in recs}
    if len(tasks) > 1:
        problems.append(f"任务描述不一致：{tasks}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--src", default="~/atom_episodes")
    parser.add_argument("--thresh", type=float, default=8.0, help="相机漂移阈值 px")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    src = os.path.expanduser(args.src)
    files = sorted(glob.glob(f"{src}/episode_*.npz"))
    if not files:
        print(f"  {src} 里没有数据")
        return 1

    first = np.load(files[0], allow_pickle=True)
    names = [str(k) for k in first["camera_keys"]]
    print(f"  {src}   相机 {names}")

    recs = scan(files, names, verbose=True)
    problems = check(recs, names, args.thresh)

    print()
    if problems:
        print(f"  ── 要处理的 {len(problems)} 项 ──")
        for p in problems:
            print(f"    ⚠ {p}")
    else:
        print(f"  ── 自动检查全过 ──")
    print(f"\n  剩下这两件只能回放看：人手误入、操作对不对")
    print(f"    python -m atom_robot.ar5.replay --all")
    return 0 if not problems else 2


if __name__ == "__main__":
    sys.exit(main())
