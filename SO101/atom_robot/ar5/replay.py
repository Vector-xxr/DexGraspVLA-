#!/usr/bin/env python3
"""回放采好的数据，检查这条能不能用。只读文件，不碰硬件。

    python -m atom_robot.ar5.replay                    # 列出所有条目
    python -m atom_robot.ar5.replay --episode 0        # 回放第 0 条
    python -m atom_robot.ar5.replay --all              # 一条条看完
    python -m atom_robot.ar5.replay --episode 0 --cam cam_front   # 只看腕部

**默认所有相机并排显示**（俯视 + 腕部）。

窗口里：
    空格   暂停 / 继续
    ← →    上一帧 / 下一帧（暂停时用）
    a d    上一条 / 下一条
    q      退出

画面下面那条是手的开合，绿色越长手张得越开。左上角是这一帧的位移量。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def load_episode(npz_path: Path) -> dict:
    data = np.load(npz_path, allow_pickle=True)
    camera_keys = [str(k) for k in data["camera_keys"]]
    videos = {}
    for key in camera_keys:
        path = npz_path.with_name(f"{npz_path.stem}_{key}.mp4")
        videos[key] = path if path.exists() else None
    # 有实测帧率就用实测的：npz 里的 `fps` 是配置的标称值，回路没跑到那么快
    # 时它是错的，按它回放会比真实快。老数据没有这一项。
    nominal = float(data["fps"])
    measured = float(data["fps_measured"]) if "fps_measured" in data else None
    return {
        "path": npz_path,
        "state": data["state"],
        "action": data["action"],
        "ee_pose": data["ee_pose"],
        "task": str(data["task"]),
        "fps": measured or nominal,
        "fps_nominal": nominal,
        "fps_measured": measured,
        "camera_keys": camera_keys,
        "videos": videos,
    }


def summarise(ep: dict) -> str:
    action, pose = ep["action"], ep["ee_pose"]
    # 走过的**路径长度**，不是首尾直线距离。后者对 take & place 是错的判据：
    # 做完之后臂就在起点附近 —— 实测 8 条会误报 4 条。
    path_mm = float(np.linalg.norm(np.diff(pose[:, :3], axis=0), axis=1).sum()) * 1000
    step = np.linalg.norm(action[:, :3], axis=1) * 1000
    grip = action[:, 6]
    notes = []
    if np.ptp(grip) < 0.02:
        notes.append("手没动")
    if path_mm < 100:
        notes.append("臂基本没动")
    if len(action) < 10:
        notes.append("太短")
    flag = ("  ⚠ " + "、".join(notes)) if notes else ""
    return (f"{ep['path'].name:22s} {len(action):4d} 帧  {len(action)/ep['fps']:5.1f}s  "
            f"末端走 {path_mm:5.0f}mm  逐帧中位 {np.median(step):4.2f}mm  "
            f"手 {grip.min():.2f}~{grip.max():.2f}{flag}")


def read_frames(video_path) -> list:
    import cv2

    if video_path is None:
        return []
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)          # 保持 BGR，直接给 imshow
    cap.release()
    return frames


def panel(frame, name, height=480):
    """给一路画面加个名字条，并统一高度，方便并排。"""
    import cv2

    h, w = frame.shape[:2]
    if h != height:
        frame = cv2.resize(frame, (int(w * height / h), height))
    out = frame.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(out, name, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1)
    return out


def compose(frames_by_cam, order, i):
    """把这一帧的所有相机横着拼起来。缺帧的那路补黑。"""
    import cv2

    panels = []
    for name in order:
        frames = frames_by_cam.get(name) or []
        if i < len(frames):
            panels.append(panel(frames[i], name))
        else:
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank, f"{name}: 没有这一帧", (60, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60, 60, 200), 2)
            panels.append(panel(blank, name))
    return np.hstack(panels) if panels else np.zeros((480, 640, 3), dtype=np.uint8)


def annotate(frame, ep, i):
    import cv2

    out = frame.copy()
    h, w = out.shape[:2]
    action = ep["action"][i]
    step_mm = float(np.linalg.norm(action[:3])) * 1000
    grip = float(action[6])

    # 底部：进度、位移、手开合条。放底部是因为顶部让给了每路的相机名。
    bar_y = h - 30
    cv2.rectangle(out, (0, h - 52), (w, h), (0, 0, 0), -1)
    cv2.putText(out, f"{ep['path'].stem}   {i + 1}/{len(ep['action'])}   "
                     f"step {step_mm:5.2f} mm   grip {grip:.2f}",
                (8, h - 36), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.rectangle(out, (10, bar_y), (w - 10, bar_y + 12), (70, 70, 70), 1)
    fill = int((w - 22) * max(0.0, min(1.0, grip)))
    if fill > 0:
        cv2.rectangle(out, (11, bar_y + 1), (11 + fill, bar_y + 11), (0, 200, 0), -1)
    cv2.putText(out, "grip", (w - 46, bar_y + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (200, 200, 200), 1)

    prog = int(w * (i + 1) / len(ep["action"]))
    cv2.line(out, (0, h - 54), (prog, h - 54), (0, 165, 255), 2)
    return out


def play(ep: dict, only=None) -> str:
    """回放一条，所有相机并排。返回 'next' / 'prev' / 'quit'。"""
    import cv2

    print(f"\n  {ep['path'].name}   「{ep['task']}」")
    print(f"  {summarise(ep)}")
    if ep["fps_measured"]:
        print(f"  帧率 实测 {ep['fps_measured']:.1f} Hz（标称 {ep['fps_nominal']:.0f}）")
    else:
        print(f"  帧率 {ep['fps_nominal']:.0f} Hz（标称值，这条没记时间戳）")

    order = [k for k in ep["camera_keys"] if only is None or k == only]
    if only is not None and not order:
        print(f"  这条没有 '{only}'，有的是 {ep['camera_keys']}")
        return "next"

    frames_by_cam = {k: read_frames(ep["videos"].get(k)) for k in order}
    have = {k: len(v) for k, v in frames_by_cam.items()}
    print(f"  相机 {have}")
    missing = [k for k, v in have.items() if v == 0]
    if missing:
        print(f"  ⚠ 这几路没有视频文件：{missing}")

    n = len(ep["action"])
    longest = max(have.values(), default=0)
    if longest == 0:
        print("  没有视频，只能看数字")
        for i in range(0, n, max(1, n // 20)):
            a = ep["action"][i]
            print(f"    {i:4d}  位移 {np.linalg.norm(a[:3])*1000:5.2f}mm  手 {a[6]:.2f}")
        return "next"
    if longest < n:
        print(f"  视频最多只有 {longest} 帧，数据有 {n} 帧，按短的来")
        n = longest

    win = "replay"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    # 两路 640x480 并排是 1280x480，一开就撑满屏不好用，先缩到八成
    cv2.resizeWindow(win, min(1600, 640 * len(order)), 480)
    delay = max(1, int(1000 / ep["fps"]))
    i, paused = 0, False
    while True:
        cv2.imshow(win, annotate(compose(frames_by_cam, order, i), ep, i))
        key = cv2.waitKey(0 if paused else delay) & 0xFF

        if key in (ord("q"), 27):
            cv2.destroyWindow(win)
            return "quit"
        if key == ord("d"):
            cv2.destroyWindow(win)
            return "next"
        if key == ord("a"):
            cv2.destroyWindow(win)
            return "prev"
        if key == ord(" "):
            paused = not paused
        if key in (81, ord(",")):        # ←
            i = max(0, i - 1)
            paused = True
        if key in (83, ord(".")):        # →
            i = min(n - 1, i + 1)
            paused = True

        if not paused:
            i += 1
            if i >= n:
                cv2.destroyWindow(win)
                return "next"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--src", default="~/atom_episodes", help="数据在哪")
    parser.add_argument("--episode", type=int, help="回放第几条")
    parser.add_argument("--all", action="store_true", help="一条条看完")
    parser.add_argument("--cam", help="只看某一路（cam_high / cam_front）。默认全部并排")
    args = parser.parse_args()

    src = Path(args.src).expanduser()
    paths = sorted(src.glob("episode_*.npz"))
    if not paths:
        print(f"{src} 里没有数据")
        return 1

    if args.episode is None and not args.all:
        print(f"{src}   共 {len(paths)} 条\n")
        bad = 0
        for p in paths:
            line = summarise(load_episode(p))
            print("  " + line)
            if "⚠" in line:
                bad += 1
        print(f"\n  {len(paths)} 条里 {bad} 条有问题")
        print(f"  看某一条：--episode 0    挨个看：--all")
        return 0

    index = args.episode or 0
    while 0 <= index < len(paths):
        action = play(load_episode(paths[index]), only=args.cam)
        if action == "quit":
            break
        if action == "prev":
            index -= 1
        else:
            if not args.all:
                break
            index += 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
