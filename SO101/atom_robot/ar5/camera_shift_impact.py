#!/usr/bin/env python3
"""相机偏了之后，策略还能不能用 —— **不碰机器人就能量出来**。

**在 lerobot 环境里跑。**

    L=~/miniforge3/envs/lerobot/bin/python

    $L atom_robot/ar5/camera_shift_impact.py --ckpt ~/vel_runs/act_bolt_0904_1732/checkpoints/020000
    $L atom_robot/ar5/camera_shift_impact.py --ckpt ~/pi05_bolt/checkpoints/030000

思路：`check_camera_pose` 已经解出了"采集时的画面 → 现在的画面"的仿射矩阵。
拿它去 **warp 录好的图像**，就得到"如果当时相机在现在这个位置会拍到什么"。
把原图和 warp 图分别喂进策略，比较两组预测动作 —— 差多少，就是这次偏移的代价。

为什么值得这么做：真机评测一轮十几分钟，而且策略表现差的时候分不清是相机偏了
还是模型本来就不行。这个脚本几分钟出结果，而且**把两件事分开了**。

判据（先写死，避免事后找理由）：

    位置增量相关系数 > 0.90    扛得住，直接上机
                  0.70~0.90   能上机但要预期下降，--scale 0.5 起步
                  < 0.70      别上机，先解决机械固定

`--per-camera` 会把两路分开算，看是哪一路主导了退化。
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


class _Obs:
    """把图像和本体感觉包成 build_batch 要的样子。"""

    def __init__(self, images, qpos):
        self.observation = {"images": images, "qpos": qpos}


def measure_shift(config, names, arm_q=None):
    """量出每一路相机现在偏了多少，返回 {相机名: 仿射矩阵}。

    直接复用 check_camera_pose 的基准和求解，不另起一套 —— 两个脚本对"偏了
    多少"必须是同一个答案，否则一个说能用、一个说不能用就没法收场了。
    """
    from atom_robot.ar5.check_camera_pose import (
        compare,
        grab_live,
        reference_high,
        reference_wrist,
    )

    out = {}
    for name in names:
        live = grab_live(config.cameras.sources[name], config.cameras.width,
                         config.cameras.height)
        if name == "cam_high":
            ref, n = reference_high()
            where = f"{n} 条首帧的中位图"
        else:
            if arm_q is None:
                print(f"  {name}: 读不到臂的关节角，跳过")
                continue
            ref, dist, where = reference_wrist(arm_q)
            if ref is None or dist is None or dist > 5.0:
                print(f"  {name}: 找不到姿态匹配的基准帧"
                      f"（最近的差 {dist:.1f}°），跳过")
                continue
            where = f"{where}，关节角差 {dist:.1f}°"
        r = compare(ref, live, name)
        if r is None:
            continue
        print(f"  {name:<10} 基准 {where}")
        print(f"             平移 ({r['dx']:+.1f}, {r['dy']:+.1f}) px   "
              f"旋转 {r['rot']:+.2f}°   缩放 {r['scale']:.4f}   "
              f"内点 {r['inliers']}/{r['matches']}")
        out[name] = r["M"]
    return out


def read_episode(path: Path, names, limit: int):
    """一条 episode 的图像 + 状态。图像保持 RGB uint8，和训练时一致。"""
    import cv2

    data = np.load(path, allow_pickle=True)
    n = min(limit, len(data["state"]))
    caps = {k: cv2.VideoCapture(str(path.with_name(f"{path.stem}_{k}.mp4"))) for k in names}
    frames = {k: [] for k in names}
    for _ in range(n):
        for k, cap in caps.items():
            ok, fr = cap.read()
            if not ok:
                n = min(n, len(frames[k]))
                break
            frames[k].append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    for cap in caps.values():
        cap.release()
    n = min([n] + [len(v) for v in frames.values()])
    return ({k: v[:n] for k, v in frames.items()},
            data["state"][:n], data["action"][:n], str(data["task"]))


def warp(img, M):
    """按仿射矩阵把图像变形。边界用**复制边缘**填。

    不用黑边：黑色是画面里不会自然出现的东西，模型一眼就能认出"这里被动过"，
    那样量出来的退化里混着"黑边造成的分布外"，不是纯粹的视角变化。
    """
    import cv2

    h, w = img.shape[:2]
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def predict(policy, pre, post, frames, state, names, device, task, warps=None):
    """跑一整条，返回逐帧动作。warps 给了就先把对应相机的图 warp 掉。"""
    import torch

    from atom_robot.ar5.eval_policy import build_batch

    policy.reset()
    out = []
    for i in range(len(state)):
        images = {}
        for k in names:
            img = frames[k][i]
            if warps and k in warps:
                img = warp(img, warps[k])
            images[k] = img
        batch = pre(build_batch(_Obs(images, state[i]), names, device, task))
        with torch.inference_mode():
            a = policy.select_action(batch)
        out.append(post(a).squeeze(0).float().cpu().numpy())
    return np.asarray(out)


def report(label, base, shifted):
    """两组预测的差距。相关系数是主判据，MAE 给量级参考。"""
    pos_r = float(np.corrcoef(base[:, :3].ravel(), shifted[:, :3].ravel())[0, 1])
    rot_r = float(np.corrcoef(base[:, 3:6].ravel(), shifted[:, 3:6].ravel())[0, 1])
    grip_r = float(np.corrcoef(base[:, 6], shifted[:, 6])[0, 1])
    bm = np.linalg.norm(base[:, :3], axis=1) * 1000
    sm = np.linalg.norm(shifted[:, :3], axis=1) * 1000
    print(f"\n  ── {label} ──")
    print(f"    位置增量相关  r = {pos_r:+.3f}")
    print(f"    姿态增量相关  r = {rot_r:+.3f}")
    print(f"    夹爪相关      r = {grip_r:+.3f}")
    print(f"    逐帧位移      原图 中位 {np.median(bm):.2f} mm  ->  "
          f"偏移后 {np.median(sm):.2f} mm")
    print(f"    位置 MAE      {np.abs(base[:, :3] - shifted[:, :3]).mean()*1000:.3f} mm"
          f"   （原图动作均幅 {np.abs(base[:, :3]).mean()*1000:.3f} mm）")
    return pos_r


VERDICTS = ((0.90, "✓ 扛得住，可以直接上机"),
            (0.70, "⚠ 能上机但要预期下降，--scale 0.5 起步"),
            (-2.0, "✗ 别上机 —— 先解决机械固定"))


def verdict(r):
    for threshold, text in VERDICTS:
        if r > threshold:
            return text
    return VERDICTS[-1][1]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--episodes", type=int, default=3, help="用几条 episode 评估")
    parser.add_argument("--frames", type=int, default=150, help="每条取前多少帧")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--per-camera", action="store_true",
                        help="两路分开算，看是哪一路主导了退化")
    parser.add_argument("--src", default=EPISODES)
    args = parser.parse_args()

    import logging
    logging.basicConfig(level=logging.WARNING, format="  %(message)s")

    config = AR5Config()
    names = list(config.cameras.sources)

    from atom_robot.ar5.drivers.cameras import check_sources
    problems = check_sources(config.cameras)
    if problems:
        print("  相机开不了：")
        for k, v in problems.items():
            print(f"    {k}: {v}")
        return 1

    arm_q = None
    try:
        from atom_robot.ar5.drivers.rokae_arm import RokaeAR5
        arm = RokaeAR5(config.arm)
        arm.connect(power_on=False)
        arm_q = arm.get_joint_positions()
        arm.disconnect()
    except Exception as exc:  # noqa: BLE001
        print(f"  读不到臂的关节角（{str(exc)[:60]}）—— 腕部那一路会被跳过")

    print(f"\n  === 量当前偏移 ===")
    warps = measure_shift(config, names, arm_q)
    if not warps:
        print("\n  一路都量不出来，没法继续。")
        return 1

    print(f"\n  === 加载策略 ===")
    from atom_robot.ar5.eval_policy import load_policy

    policy, pre, post, model_dir, kind = load_policy(Path(args.ckpt).expanduser(),
                                                     args.device)
    print(f"  {kind}  {sum(p.numel() for p in policy.parameters())/1e6:.1f}M 参数")

    files = sorted(glob.glob(os.path.expanduser(args.src) + "/episode_*.npz"))[:args.episodes]
    if not files:
        print(f"  {args.src} 里没有数据")
        return 1

    # 每条分开跑再拼，而不是拼成一条 —— 策略有内部动作队列，跨 episode
    # 连着喂等于让它以为这是一条超长轨迹。
    combos = [("两路都偏（现状）", warps)]
    if args.per_camera:
        combos += [(f"只有 {k} 偏", {k: m}) for k, m in warps.items()]

    print(f"\n  === 跑 {len(files)} 条 × 最多 {args.frames} 帧 ===")
    results = {label: [] for label, _ in combos}
    baseline = []
    for path in files:
        frames, state, _, task = read_episode(Path(path), names, args.frames)
        if not len(state):
            continue
        base = predict(policy, pre, post, frames, state, names, args.device, task)
        baseline.append(base)
        for label, w in combos:
            results[label].append(
                predict(policy, pre, post, frames, state, names, args.device, task, w))
        print(f"    {os.path.basename(path)}  {len(state)} 帧")

    base_all = np.concatenate(baseline)
    print(f"\n  === 结果（共 {len(base_all)} 帧）===")
    main_r = None
    for label, _ in combos:
        r = report(label, base_all, np.concatenate(results[label]))
        print(f"    -> {verdict(r)}")
        if main_r is None:
            main_r = r

    print(f"\n  ── 结论 ──")
    print(f"  {verdict(main_r)}   （位置增量相关 r = {main_r:+.3f}）")
    print(f"\n  注意这量的是**相机偏移这一个变量**的影响：同一个策略、同一份数据，")
    print(f"  只有图像被 warp 过。所以它不回答'模型好不好'，只回答")
    print(f"  '相机这点偏移会不会毁掉它'。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
