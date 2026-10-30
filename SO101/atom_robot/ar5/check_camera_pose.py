#!/usr/bin/env python3
"""相机被碰过之后，对着采数据时的画面量它移了多少、该往哪调回去。

    python -m atom_robot.ar5.check_camera_pose                  # 两路都查
    python -m atom_robot.ar5.check_camera_pose --cam cam_high   # 只查俯视
    python -m atom_robot.ar5.check_camera_pose --save-ref       # 重新固定好之后，把当前画面存成新基准

**为什么这件事非做不可**：策略是照着采集时的视角学的。相机挪了，同一个物理
场景在图像里的位置就变了，模型看到的是它没见过的分布 —— 表现会崩，而且看起来
像"模型不行"，不像"相机歪了"。

**两路相机都固定在架子上**（2026-09-07 起腕部改成正视），所以判据一样：拿现在
的画面和数据里首帧的**中位图**比，算仿射变换，解出平移 / 旋转 / 缩放。
臂不用上电，也不用摆到特定位姿。
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from atom_robot import SDK_ROOT  # noqa: E402
sys.path.insert(0, str(SDK_ROOT / "Release" / "linux"))

from atom_robot.ar5.config import AR5Config  # noqa: E402

EPISODES = os.path.expanduser("~/atom_episodes")
REF_DIR = Path(__file__).resolve().parent / "camera_reference"


def grab_live(source, width: int, height: int):
    """抓一张当前画面。丢掉前几帧 —— 相机刚开时曝光还没稳。"""
    # 和 collect 同一套打开逻辑：Orbbec 走 OBSENSOR。自己 cv2.VideoCapture(路径)
    # 就是 V4L2，开一次 OBSENSOR 就塌到 0.3 Hz。
    from atom_robot.ar5.drivers.cameras import SingleCamera

    cap = SingleCamera(source, width, height)
    frame = None
    for _ in range(12):
        ok, f = cap.read()
        if ok:
            frame = f
    cap.release()
    if frame is None:
        raise IOError(f"{source!r} 打开了但读不到画面")
    return frame


def reference(name: str, limit: int = 40):
    """某一路相机的基准：整组数据首帧的逐像素中位图。

    取中位而不是某一帧，是为了把机械臂洗掉 —— 臂在每条 episode 里经过的位置
    不同，中位剩下的就是不动的背景（桌面、胶带、物体）。

    两路相机现在都固定在架子上（2026-09-07 起腕部改成正视），所以这一个函数
    对两路都成立。以前腕部相机跟着臂动，得先把臂开到某个已知位姿、再在数据里
    找关节角最近的一帧当基准 —— 那套逻辑连带它的"差 5 度就不可比"限制一起
    去掉了。
    """
    import cv2

    files = sorted(glob.glob(f"{EPISODES}/episode_*.npz"))[:limit]
    if not files:
        return None, 0
    frames = []
    for f in files:
        cap = cv2.VideoCapture(f"{f[:-4]}_{name}.mp4")
        ok, fr = cap.read()
        cap.release()
        if ok:
            frames.append(fr.astype(np.float32))
    if not frames:
        return None, 0
    return np.median(np.stack(frames), axis=0).astype(np.uint8), len(frames)


# 场景里两个明确的色块。特征点匹配在视角变化大的时候会退化（实测只剩 4 个内点，
# 那个数量解出来的变换不能信），而色块定位不依赖纹理，视角变多少都还在。
LANDMARKS = (
    ("黄胶带", (20, 80, 80), (38, 255, 255), 800),
    ("蓝螺套", (95, 120, 50), (130, 255, 255), 150),
)


def find_landmarks(img):
    """按颜色找地标，返回 {名字: (cx, cy, 面积, 外接框)}。找不到的为 None。"""
    import cv2

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    out = {}
    for name, lo, hi, area_min in LANDMARKS:
        mask = cv2.morphologyEx(cv2.inRange(hsv, lo, hi), cv2.MORPH_OPEN,
                                np.ones((5, 5), np.uint8))
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = [c for c in cnts if cv2.contourArea(c) > area_min]
        if not cnts:
            out[name] = None
            continue
        pts = np.vstack(cnts)
        M = cv2.moments(pts)
        out[name] = (M["m10"] / M["m00"], M["m01"] / M["m00"],
                     sum(cv2.contourArea(c) for c in cnts), cv2.boundingRect(pts))
    return out


def landmark_shift(ref, live):
    """两幅图之间地标的平均位移。比特征匹配稳，因为不依赖纹理。"""
    R, L = find_landmarks(ref), find_landmarks(live)
    rows, shifts = [], []
    for name, _, _, _ in LANDMARKS:
        r, l = R.get(name), L.get(name)
        if r is None or l is None:
            rows.append((name, r, l, None, None))
            continue
        dx, dy = l[0] - r[0], l[1] - r[1]
        rows.append((name, r, l, dx, dy))
        shifts.append((dx, dy))
    if not shifts:
        return rows, None, None, None
    dx = float(np.mean([s[0] for s in shifts]))
    dy = float(np.mean([s[1] for s in shifts]))
    areas = [l[2] / r[2] for _, r, l, sx, _ in rows if r and l and sx is not None]
    return rows, dx, dy, (float(np.mean(areas)) if areas else None)


def align_live(config, name, ref, thresh):
    """实时对齐窗口。一边搬相机一边看误差，比照着文字描述搬靠谱得多。

    画面上叠三样东西：基准里地标该在的位置（十字）、现在在哪（圆圈）、
    以及把两者连起来的箭头。误差进阈值就变绿。
    """
    import cv2

    from atom_robot.ar5.drivers.cameras import SingleCamera

    src = config.cameras.sources[name]
    try:
        cap = SingleCamera(src, config.cameras.width, config.cameras.height, name=name)
    except IOError as exc:
        print(f"  打不开 {src!r}：{exc}")
        return False
    R = find_landmarks(ref)
    win = f"align {name}"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    print(f"\n  实时对齐 {name} —— 一边搬相机一边看。误差进 {thresh:.0f} px 变绿。")
    print("  q 退出   s 存一张")
    ok_now = False
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            L = find_landmarks(frame)
            shown = frame.copy()
            # 基准帧压在底下当参考影，一眼看出差多少
            shown = cv2.addWeighted(shown, 0.75, ref, 0.25, 0)
            errs = []
            for lname, _, _, _ in LANDMARKS:
                r, l = R.get(lname), L.get(lname)
                if r is None:
                    continue
                cv2.drawMarker(shown, (int(r[0]), int(r[1])), (0, 255, 255),
                               cv2.MARKER_CROSS, 30, 2)
                if l is None:
                    continue
                d = np.hypot(l[0] - r[0], l[1] - r[1])
                errs.append(d)
                colour = (0, 220, 0) if d < thresh else (0, 100, 255)
                cv2.circle(shown, (int(l[0]), int(l[1])), 12, colour, 2)
                cv2.arrowedLine(shown, (int(l[0]), int(l[1])), (int(r[0]), int(r[1])),
                                colour, 2, tipLength=0.25)
            err = float(np.mean(errs)) if errs else float("nan")
            ok_now = bool(errs) and err < thresh
            bar = (0, 160, 0) if ok_now else (0, 0, 0)
            cv2.rectangle(shown, (0, 0), (shown.shape[1], 30), bar, -1)
            msg = (f"{name}   误差 {err:6.1f} px" if errs else f"{name}   地标丢失（移出画面？）")
            cv2.putText(shown, msg + ("   OK" if ok_now else "   把圆圈拖到十字上"),
                        (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow(win, shown)
            k = cv2.waitKey(30) & 0xFF
            if k in (ord("q"), 27):
                break
            if k == ord("s"):
                cv2.imwrite(f"align_{name}.png", shown)
                print(f"  存了 align_{name}.png")
    finally:
        cap.release()
        cv2.destroyWindow(win)
    return ok_now


def compare(ref, live, name: str):
    """两帧之间的单应变换，解成人能照着调的量。"""
    import cv2

    g_ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    g_live = cv2.cvtColor(live, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(4000)
    k1, d1 = orb.detectAndCompute(g_ref, None)
    k2, d2 = orb.detectAndCompute(g_live, None)
    if d1 is None or d2 is None or len(k1) < 20 or len(k2) < 20:
        print(f"  {name}: 特征点太少（{len(k1) if k1 else 0} / {len(k2) if k2 else 0}），比不了")
        return None

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = matcher.knnMatch(d1, d2, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < 12:
        print(f"  {name}: 只有 {len(good)} 个可靠匹配，比不了 —— 画面变化太大？")
        return None

    src = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, mask = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                          ransacReprojThreshold=3.0)
    if M is None:
        print(f"  {name}: 解不出变换")
        return None

    inliers = int(mask.sum())
    dx, dy = float(M[0, 2]), float(M[1, 2])
    scale = float(np.hypot(M[0, 0], M[1, 0]))
    rot = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
    # 角点平均位移：比单看平移更能反映"画面整体挪了多少"，因为旋转和缩放
    # 也会让边缘大幅移动，而 dx/dy 只描述原点。
    h, w = g_ref.shape
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    moved = cv2.transform(corners, M).reshape(-1, 2) - corners.reshape(-1, 2)
    drift = float(np.linalg.norm(moved, axis=1).mean())
    # `M` 一起返回：那是"从采集时的画面到现在的画面"的 2x3 仿射矩阵。
    # camera_shift_impact.py 拿它去 warp 录好的图像，模拟"如果当时相机在现在
    # 这个位置会拍到什么"，这样不碰机器人就能量出策略退化多少。
    return {"dx": dx, "dy": dy, "scale": scale, "rot": rot,
            "drift": drift, "inliers": inliers, "matches": len(good),
            "w": w, "h": h, "M": M}


def report(name: str, r: dict, thresh_px: float):
    print(f"\n  ── {name} ──")
    print(f"    可靠匹配 {r['inliers']}/{r['matches']}")
    print(f"    平移      dx {r['dx']:+7.1f} px    dy {r['dy']:+7.1f} px")
    print(f"    旋转      {r['rot']:+6.2f}°")
    print(f"    缩放      {r['scale']:.4f}   （>1 = 相机离场景更近了）")
    print(f"    画面整体位移 {r['drift']:.1f} px   （画幅 {r['w']}x{r['h']}）")

    if r["drift"] < thresh_px:
        print(f"    ✓ 小于 {thresh_px:.0f} px，可以认为没动")
        return True

    print(f"    ⚠ 超过 {thresh_px:.0f} px —— 相机位置变了，要调回去")
    print(f"\n    怎么调（把相机朝这些方向搬，让画面回到基准）：")
    # 画面里的内容往 +dx 走 = 相机往 -x 方向挪了。要调回去，相机反着搬。
    if abs(r["dx"]) > 3:
        print(f"      左右：画面内容右移了 {r['dx']:+.0f} px  ->  相机往"
              f"{'右' if r['dx'] > 0 else '左'}挪 {abs(r['dx']):.0f} px 对应的距离")
    if abs(r["dy"]) > 3:
        print(f"      上下：画面内容下移了 {r['dy']:+.0f} px  ->  相机往"
              f"{'下' if r['dy'] > 0 else '上'}挪 {abs(r['dy']):.0f} px 对应的距离")
    if abs(r["rot"]) > 0.5:
        print(f"      旋转：画面转了 {r['rot']:+.2f}°  ->  相机反着转 "
              f"{abs(r['rot']):.2f}°（{'逆' if r['rot'] > 0 else '顺'}时针拧回去）")
    if abs(r["scale"] - 1.0) > 0.01:
        pct = (r["scale"] - 1.0) * 100
        print(f"      远近：画面{'放大' if pct > 0 else '缩小'}了 {abs(pct):.1f}%  ->  相机往"
              f"{'后' if pct > 0 else '前'}退，或者说离场景{'远' if pct > 0 else '近'}一点")
    return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--cam", choices=["cam_high", "cam_front"], help="只查一路")
    parser.add_argument("--thresh", type=float, default=8.0,
                        help="画面整体位移超过多少像素算「动了」（默认 8）")
    parser.add_argument("--save-ref", action="store_true",
                        help="把当前画面存成新基准。**只在确认相机已经装回正确位置后用**")
    parser.add_argument("--out", default=None, help="把对比图存到哪（默认当前目录）")
    parser.add_argument("--live", action="store_true",
                        help="打开实时对齐窗口：一边搬相机一边看误差，进阈值变绿")
    args = parser.parse_args()

    import cv2

    config = AR5Config()
    names = [args.cam] if args.cam else list(config.cameras.sources)
    out_dir = Path(args.out).expanduser() if args.out else Path.cwd()

    from atom_robot.ar5.drivers.cameras import check_sources
    problems = check_sources(config.cameras)
    if problems:
        print("  相机开不了：")
        for k, v in problems.items():
            print(f"    {k}: {v}")
        return 1

    n_eps = len(glob.glob(f"{EPISODES}/episode_*.npz"))
    if n_eps == 0:
        print(f"  {EPISODES} 里没有采集数据 —— 没有基准可比。")
        print(f"  这个脚本是用来判断「相机相对采集时有没有动」的，采第一批数据之前")
        print(f"  无从谈起。先采数据；之后相机被碰过再回来跑。")
        print(f"\n  只想看画面：view_camera --all")
        return 0
    print(f"  基准来自 {EPISODES} 的 {n_eps} 条采集数据")
    all_ok = True

    for name in names:
        source = config.cameras.sources[name]
        live = grab_live(source, config.cameras.width, config.cameras.height)

        ref, n = reference(name)
        if ref is None:
            print(f"\n  {name}: 找不到采集数据，没法比")
            all_ok = False
            continue
        print(f"\n  {name} 基准 = {n} 条 episode 首帧的中位图")

        # 先用地标（色块）量 —— 视角变化大时特征匹配会退化到几个内点，
        # 那种估计不能信；色块不依赖纹理，挪多远都还在。
        rows, ldx, ldy, area = landmark_shift(ref, live)
        print(f"\n  ── {name} ──")
        print(f"    {'地标':<8}{'采集时':>16}{'现在':>16}{'位移':>22}")
        lost = []
        for lname, rr, ll, dx, dy in rows:
            if rr is None:
                print(f"    {lname:<8}{'基准里就没有':>16}")
                continue
            if ll is None:
                lost.append(lname)
                print(f"    {lname:<8}({rr[0]:6.0f},{rr[1]:6.0f}){'已移出画面':>16}   ⚠")
                continue
            print(f"    {lname:<8}({rr[0]:6.0f},{rr[1]:6.0f})({ll[0]:6.0f},{ll[1]:6.0f})"
                  f"   dx {dx:+6.0f}  dy {dy:+6.0f}  共 {np.hypot(dx,dy):5.0f} px")

        r = compare(ref, live, name)
        if r is not None and r["inliers"] >= 15:
            report(name, r, args.thresh)
        elif r is not None:
            print(f"    （特征匹配只有 {r['inliers']} 个内点，旋转/缩放估计不可信，不采信）")

        if ldx is None:
            print("    两个地标都找不到 —— 画面变化太大，只能靠肉眼对着基准图调")
            ok = False
        else:
            shift = float(np.hypot(ldx, ldy))
            print(f"\n    画面内容整体位移 {shift:.0f} px"
                  + (f"，面积比 {area*100:.0f}%（≈距离{'变近' if area>1.05 else '变远' if area<0.95 else '没变'}）"
                     if area else ""))
            ok = shift < args.thresh
            if ok:
                print(f"    ✓ 小于 {args.thresh:.0f} px，可以认为没动")
            else:
                print(f"    ⚠ 相机动了。内容往{'右' if ldx>0 else '左'}"
                      f"{'下' if ldy>0 else '上'}跑了 ({ldx:+.0f}, {ldy:+.0f}) px")
                if lost:
                    print(f"    ⚠ {'、'.join(lost)} 已经跑出画幅 —— 工作区不完整，必须调回来")
                print(f"    调的时候用 --live，比照着数字搬准得多。")
        all_ok = all_ok and ok

        if args.live:
            if align_live(config, name, ref, args.thresh):
                print(f"    ✓ {name} 对齐了")
                all_ok = True
            else:
                print(f"    {name} 退出时还没对齐")
                all_ok = False

        panel = np.hstack([cv2.resize(ref, (480, 360)), cv2.resize(live, (480, 360)),
                           cv2.resize(cv2.absdiff(ref, live), (480, 360))])
        for i, label in enumerate(("采集时（基准）", "现在", "差异")):
            cv2.rectangle(panel, (i * 480, 0), (i * 480 + 480, 22), (0, 0, 0), -1)
            cv2.putText(panel, ["REFERENCE (dataset)", "NOW", "DIFF"][i],
                        (i * 480 + 8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1)
        path = out_dir / f"campose_{name}.jpg"
        cv2.imwrite(str(path), panel, [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(f"    对比图 {path}")

        if args.save_ref:
            REF_DIR.mkdir(exist_ok=True)
            cv2.imwrite(str(REF_DIR / f"{name}.png"), live)
            print(f"    已存为新基准 {REF_DIR / f'{name}.png'}")

    print()
    if all_ok:
        print("  两路都没动（或在阈值内），可以直接评测。")
    else:
        print("  有相机动了。按上面的提示调回去，再跑一次这个脚本确认。")
        print(f"  调不回原位的话，那 {n_eps} 条数据对新视角就不成立了 —— 要么重采，")
        print("  要么接受策略性能下降。先跑 camera_shift_impact.py 量一下代价。")
    return 0 if all_ok else 2


if __name__ == "__main__":
    sys.exit(main())
