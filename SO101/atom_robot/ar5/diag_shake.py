#!/usr/bin/env python3
"""查臂为什么抖。不动臂，只测。

    python -m atom_robot.ar5.diag_shake --leader-only
        只读主臂，机械臂完全不碰。测编码器抖动和回路周期，算出这些
        噪声按现在的映射会变成从臂多大的速度抖动。

    python -m atom_robot.ar5.diag_shake
        连上机械臂，松开状态下跑完整回路（一直下发「保持不动」的目标），
        分段计时：读主臂 / 下发目标 / 建观测 / 读编码器。臂不会动。

抖动只可能来自三个地方，这个脚本把三个都量了：

  1. 编码器量化。主臂舵机 0.088°/count，相对映射 1:1，一个 count 就是从臂
     1.53 mrad。位置上很小，但跟踪器要对目标求导算前馈速度，20 ms 内跳一个
     count 就是 0.077 rad/s ——速度上限的十分之一。
  2. 回路周期不稳。前馈 = 位移 / 时间。时间不稳,速度就不稳。
  3. 人手震颤。生理性震颤 8~12 Hz，主臂是反驱的，握着就在信号里。

第 1、3 项靠滤波压掉，第 2 项靠把回路里没人要的活删掉。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from atom_robot.ar5.config import AR5Config  # noqa: E402
from atom_robot.ar5.joint_mapping import (  # noqa: E402
    JOINT_PAIRS,
    JointMapConfig,
    JointMapper,
    OneEuro,
)
from atom_robot.ar5.leader_bus import SO_JOINTS, LeaderBus  # noqa: E402

COUNT_DEG = 360.0 / 4095.0  # STS3215 一个 count 多少度


def summarise(values, unit="ms", scale=1000.0):
    a = np.sort(np.asarray(values, dtype=np.float64)) * scale
    if a.size == 0:
        return "（没有样本）"
    return (f"p50 {a[a.size // 2]:6.2f}  p95 {a[int(a.size * 0.95)]:6.2f}  "
            f"最差 {a[-1]:6.2f} {unit}")


def leader_only(args) -> int:
    """只读主臂。不碰机械臂，也不碰手。"""
    leader = LeaderBus.with_calibration(args.leader_port, args.calib_id,
                                        baudrate=args.leader_baud)
    leader.connect()
    leader.assert_torque_disabled()

    cfg = JointMapConfig.load(relative=True)
    mapper = JointMapper(cfg)
    mapper.configure({name: leader.half_travel_deg(name) for name, _, _ in JOINT_PAIRS},
                     AR5Config().arm.joint_min, AR5Config().arm.joint_max,
                     margin=AR5Config().arm.joint_limit_margin)

    print(f"\n  主臂 {args.leader_port}   {args.hz:.0f} Hz   测 {args.seconds:.0f} 秒")
    print("  别碰主臂 —— 先测静止时的底噪。\n" if args.still
          else "  按你平时遥操作的速度和幅度动主臂。\n")

    period = 1.0 / args.hz
    periods, read_ms = [], []
    raw = {name: [] for name, _, _ in JOINT_PAIRS}
    filt = {name: [] for name, _, _ in JOINT_PAIRS}
    # 单独一套滤波器，和 mapper 里那套参数一样,用来对比开/关的差别
    filters = {name: OneEuro(cfg.filter_min_cutoff, cfg.filter_beta, cfg.filter_d_cutoff)
               for name, _, _ in JOINT_PAIRS}

    previous = None
    deadline = time.perf_counter() + args.seconds
    dropped = 0
    while time.perf_counter() < deadline:
        loop = time.perf_counter()
        if previous is not None:
            periods.append(loop - previous)
        dt = 0.0 if previous is None else loop - previous
        previous = loop

        t0 = time.perf_counter()
        joints = leader.read_positions()
        read_ms.append(time.perf_counter() - t0)
        if len(joints) < len(SO_JOINTS):
            dropped += 1
            continue

        for name, _, _ in JOINT_PAIRS:
            value = joints[name]
            raw[name].append(value)
            filt[name].append(filters[name](value, dt))

        sleep = period - (time.perf_counter() - loop)
        if sleep > 0:
            time.sleep(sleep)

    leader.disconnect()

    if not periods:
        print("  一个样本都没取到。")
        return 1

    ms = np.asarray(periods) * 1000.0
    p50 = float(np.median(ms))
    p95 = float(np.percentile(ms, 95))
    print(f"  回路周期   目标 {period * 1000:.1f} ms   {summarise(periods)}")
    print(f"  读主臂     {summarise(read_ms)}   （六个舵机串行，一个一个问）")
    print(f"  丢包       {dropped} 次")
    if p95 > 1.6 * p50:
        print(f"  ⚠ 周期 p95/p50 = {p95 / p50:.1f}。前馈速度 = 位移/这段时间，"
              "分母不稳,前馈就不稳。")

    print(f"\n  每个关节（{'静止' if args.still else '运动中'}）：")
    print(f"    {'关节':<14}{'峰峰值':>10}{'≈counts':>9}"
          f"{'从臂前馈速度 rad/s':>22}")
    print(f"    {'':14}{'':>10}{'':>9}{'不滤':>11}{'滤波后':>11}")
    for name, index, _ in JOINT_PAIRS:
        a = np.asarray(raw[name])
        f = np.asarray(filt[name])
        if a.size < 3:
            continue
        span = float(a.max() - a.min())
        # 前馈速度 = 相邻两次目标之差 / 时间差，再乘映射比例进到从臂
        gain = abs(mapper._signs[name] * mapper._scales[name]) * np.pi / 180.0
        dts = np.asarray(periods[:a.size - 1])
        v_raw = np.abs(np.diff(a)[:dts.size] / dts) * gain
        v_filt = np.abs(np.diff(f)[:dts.size] / dts) * gain
        print(f"    {name:<14}{span:>9.3f}°{span / COUNT_DEG:>9.1f}"
              f"{np.percentile(v_raw, 95):>11.3f}{np.percentile(v_filt, 95):>11.3f}")

    cap = AR5Config().arm.max_joint_speed
    print(f"\n  从臂速度上限 {cap:.2f} rad/s。上面「从臂前馈速度」那两列是 p95，")
    print("  是这些噪声按现在的映射会让跟踪器多追多少速度。")

    worst = max(
        (float(np.ptp(np.asarray(raw[name]))) for name, _, _ in JOINT_PAIRS
         if len(raw[name]) > 2), default=0.0)
    if args.still and worst < COUNT_DEG * 0.5:
        # 2026-09-03 实测：没人碰的时候，五个关节 600 个采样一个 count 都不跳。
        # 所以「静止时编码器底噪把臂抖起来」这条不成立，别再往这上面查。
        print("\n  静止时零抖动 —— 主臂这一侧是干净的，回路周期也稳。")
        print("  也就是说：抖不是从这里来的。剩下的可能只有")
        print("    · 握着主臂时的手抖（8~12 Hz，相对映射 1:1 全传过去）")
        print("    · 动起来之后的量化（每步只跨几个 count，求导后就是速度纹波）")
        print("    · 回路里臂那一侧的阻塞（去掉 --leader-only 测）")
        print("  第一、二项现在被滤波压着，第三项跑不带 --leader-only 的那条命令。")
    return 0


def with_arm(args) -> int:
    """连上臂，松开状态跑完整回路，分段计时。臂只会保持不动。"""
    from atom_robot.ar5.ar5_env import AR5Env
    from atom_robot.ar5.teleop_joint import hold_posture

    leader = LeaderBus.with_calibration(args.leader_port, args.calib_id,
                                        baudrate=args.leader_baud)
    leader.connect()
    leader.assert_torque_disabled()

    config = AR5Config()
    config.use_hand = not args.no_hand
    config.arm.use_realtime = True
    if args.no_cameras:
        config.cameras.sources = {}

    env = AR5Env(config)
    env.connect()
    if not getattr(env.arm, "_rt_active", False):
        print("\n  实时通道没起来，测不了。")
        env.close()
        leader.disconnect()
        return 1

    n = config.arm.num_joints
    print(f"\n  臂 {config.arm.ip}   手 {'开' if config.use_hand else '关'}   "
          f"相机 {list(config.cameras.sources) or '（无）'}")
    print(f"  {args.hz:.0f} Hz 跑 {args.seconds:.0f} 秒。全程下发「保持当前姿态」，臂不会动。\n")

    stage = {"读主臂": [], "下发目标": [], "建观测": [], "读编码器": []}
    periods = []
    period = 1.0 / args.hz
    sent0 = env.arm.rt_stats().get("sent", 0)
    late0 = env.arm.rt_stats().get("late_ticks", 0)

    previous = None
    deadline = time.perf_counter() + args.seconds
    try:
        while time.perf_counter() < deadline:
            loop = time.perf_counter()
            if previous is not None:
                periods.append(loop - previous)
            previous = loop

            t = time.perf_counter()
            leader.read_positions()
            stage["读主臂"].append(time.perf_counter() - t)

            hold = hold_posture(env.arm, n)

            t = time.perf_counter()
            env.step_joints(hold, observe=False)
            stage["下发目标"].append(time.perf_counter() - t)

            t = time.perf_counter()
            env.get_observation()
            stage["建观测"].append(time.perf_counter() - t)

            t = time.perf_counter()
            env.arm.get_joint_positions()
            stage["读编码器"].append(time.perf_counter() - t)

            sleep = period - (time.perf_counter() - loop)
            if sleep > 0:
                time.sleep(sleep)
    finally:
        stats = env.arm.rt_stats()
        env.close()
        leader.disconnect()

    print(f"  回路周期   目标 {period * 1000:.1f} ms   {summarise(periods)}\n")
    print("  每步各段耗时：")
    for name, values in stage.items():
        if values:
            print(f"    {name:<10}{summarise(values)}")
    print("\n  「建观测」和「读编码器」是遥操作里白花的 —— 返回值没人用。")
    print("  已经从 teleop_joint / collect 的回路里拿掉了；这里单列是为了看它值多少。")

    sent = stats.get("sent", 0) - sent0
    late = stats.get("late_ticks", 0) - late0
    if sent:
        print(f"\n  1 kHz 发送线程：这段时间发了 {sent} 拍，其中 {late} 拍没赶上 "
              f"（{100.0 * late / sent:.1f}%）")
        if late / sent > 0.02:
            print("  ⚠ 赶不上就是控制器那边断供一拍、下一拍补两个,这个直接就是抖。")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--leader-only", action="store_true",
                        help="只读主臂，完全不碰机械臂")
    parser.add_argument("--still", action="store_true",
                        help="配 --leader-only：测静止底噪，跑的时候别碰主臂")
    parser.add_argument("--leader-port", default="/dev/ttyACM0")
    parser.add_argument("--calib-id", default="single_leader_arm")
    parser.add_argument("--leader-baud", type=int, default=1_000_000)
    parser.add_argument("--hz", type=float, default=50.0)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--no-hand", action="store_true")
    parser.add_argument("--no-cameras", action="store_true")
    args = parser.parse_args()

    return leader_only(args) if args.leader_only else with_arm(args)


if __name__ == "__main__":
    sys.exit(main())
