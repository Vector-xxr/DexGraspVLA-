#!/usr/bin/env python3
"""把 collect.py 录的 .npz + .mp4 转成 lerobot 数据集。

**在 lerobot 环境里跑，不是 vel-ar5：**

    ~/miniforge3/envs/lerobot/bin/python \\
        atom_robot/ar5/to_lerobot.py --src ~/atom_episodes --repo-id vel_ar5_pick

不碰硬件，纯转格式。用 lerobot 自己的接口写，格式对不对由它保证。

字段按 VEL 训练要的来：
    observation.state   8 = 7 个关节角 + 手开合
    action              7 = 位置变化量 3 + 姿态变化量 3 + 手开合 1
    observation.ee_pose 6 = 末端绝对位姿（训练用不到，留着以后重算增量用）
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

JOINT_NAMES = [f"joint_{i}" for i in range(7)] + ["hand.opening"]
ACTION_NAMES = ["d.x", "d.y", "d.z", "d.rx", "d.ry", "d.rz", "hand.opening"]
EE_NAMES = ["ee.x", "ee.y", "ee.z", "ee.rx", "ee.ry", "ee.rz"]


def read_video(path: Path) -> list:
    import cv2

    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def build_features(camera_keys, height, width) -> dict:
    features = {
        "observation.state": {
            "dtype": "float32", "shape": (len(JOINT_NAMES),), "names": JOINT_NAMES,
        },
        "action": {
            "dtype": "float32", "shape": (len(ACTION_NAMES),), "names": ACTION_NAMES,
        },
        "observation.ee_pose": {
            "dtype": "float32", "shape": (len(EE_NAMES),), "names": EE_NAMES,
        },
    }
    for key in camera_keys:
        features[f"observation.images.{key}"] = {
            "dtype": "video",
            "shape": (height, width, 3),
            "names": ["height", "width", "channels"],
        }
    return features


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--src", required=True, help="collect.py 存的目录")
    parser.add_argument("--repo-id", required=True, help="数据集名字，比如 vel_ar5_pick")
    parser.add_argument("--root", default=None, help="存到哪，默认 ~/.cache/huggingface/lerobot/<repo-id>")
    parser.add_argument("--fps", type=float, default=None, help="覆盖 npz 里记的帧率")
    parser.add_argument("--robot-type", default="ar5_inspire")
    args = parser.parse_args()

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        print("这个环境里没有 lerobot。用：")
        print("  ~/miniforge3/envs/lerobot/bin/python " + __file__ + " ...")
        return 1

    src = Path(args.src).expanduser()
    episodes = sorted(src.glob("episode_*.npz"))
    if not episodes:
        print(f"{src} 里没有 episode_*.npz")
        return 1

    first = np.load(episodes[0], allow_pickle=True)
    camera_keys = [str(k) for k in first["camera_keys"]]
    # 优先用实测帧率。npz 里的 `fps` 是配置的标称值，回路没跑到那么快时它是
    # 错的 —— 视频会放快，动作增量对应的真实间隔也比 1/fps 长。老数据没有
    # fps_measured 这一项，退回标称值。
    measured = float(first["fps_measured"]) if "fps_measured" in first else None
    fps = float(args.fps or measured or first["fps"])
    if args.fps is None and measured and abs(measured - float(first["fps"])) > 1.0:
        print(f"注意    实测帧率 {measured:.1f} Hz，标称 {float(first['fps']):.0f} Hz，用实测的")

    height = width = None
    for key in camera_keys:
        video = episodes[0].with_name(f"{episodes[0].stem}_{key}.mp4")
        if video.exists():
            frames = read_video(video)
            if frames:
                height, width = frames[0].shape[:2]
                break
    if height is None:
        height, width = 480, 640
        if camera_keys:
            print(f"读不到视频，图像尺寸按 {width}x{height} 算")

    print(f"源目录  {src}")
    print(f"共      {len(episodes)} 条")
    print(f"相机    {camera_keys or '（无）'}")
    print(f"帧率    {fps:.0f}")

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=int(round(fps)),
        features=build_features(camera_keys, height, width),
        root=args.root,
        robot_type=args.robot_type,
        use_videos=bool(camera_keys),
    )

    for path in episodes:
        data = np.load(path, allow_pickle=True)
        state, action, ee_pose = data["state"], data["action"], data["ee_pose"]
        task = str(data["task"])

        videos = {}
        for key in camera_keys:
            video_path = path.with_name(f"{path.stem}_{key}.mp4")
            frames = read_video(video_path) if video_path.exists() else []
            if len(frames) < len(state):
                print(f"  {path.name}: {key} 只有 {len(frames)} 帧，数据有 {len(state)} 帧")
            videos[key] = frames

        n = len(state)
        for key, frames in videos.items():
            n = min(n, len(frames)) if frames else n

        for i in range(n):
            frame = {
                "observation.state": state[i].astype(np.float32),
                "action": action[i].astype(np.float32),
                "observation.ee_pose": ee_pose[i].astype(np.float32),
                "task": task,
            }
            for key in camera_keys:
                if videos[key]:
                    frame[f"observation.images.{key}"] = videos[key][i]
            dataset.add_frame(frame)

        dataset.save_episode()
        print(f"  {path.name}  ->  {n} 帧   「{task}」")

    root = dataset.root
    print(f"\n写好了：{root}")
    print(f"  检查一下：")
    print(f"    python -c \"from lerobot.datasets.lerobot_dataset import LeRobotDataset;"
          f" d=LeRobotDataset('{args.repo_id}'); print(d)\"")
    return 0


if __name__ == "__main__":
    sys.exit(main())
