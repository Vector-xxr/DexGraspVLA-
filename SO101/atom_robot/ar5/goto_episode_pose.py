#!/usr/bin/env python3
"""把臂开到某条 episode 某一帧的姿态，然后**一直保持在那儿**。

    python -m atom_robot.ar5.goto_episode_pose                      # episode 0 第 0 帧
    python -m atom_robot.ar5.goto_episode_pose --episode 12 --frame 200
    python -m atom_robot.ar5.goto_episode_pose --show               # 只打印姿态，不动臂

用来干什么：腕部相机装在腕上，画面完全由臂的姿态决定。要判断它有没有被碰过、
往哪调，就得让臂回到采集时的某个姿态，再拿当时录下的那一帧和现在的实时画面比。

**臂会保持在目标姿态不动，直到你按 q。** 这段时间你可以慢慢调相机 ——
脚本持续下发目标，所以 sender 不会因为 0.25 秒收不到目标而刹车 latch。

安全：目标姿态先过软限位和地板线检查（`AR5Env.goto_joints` 里），不合格直接
拒绝而不是硬走。移动是关节空间插值，步长压在跳变门以内。
"""

from __future__ import annotations

import argparse
import os
import select
import sys
import termios
import time
import tty
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from atom_robot import SDK_ROOT  # noqa: E402
sys.path.insert(0, str(SDK_ROOT / "Release" / "linux"))

from atom_robot.ar5.ar5_env import AR5Env  # noqa: E402
from atom_robot.ar5.config import AR5Config  # noqa: E402

EPISODES = os.path.expanduser("~/atom_episodes")


def _goto(args, q, opening):
    """把臂开到 q 并保持。episode 和 --joints 两条路共用这一段 ——
    软限位、地板线、关节空间插值、保持到按 q，都在这里面。"""

    config = AR5Config()
    config.use_hand = not args.no_hand
    config.arm.use_realtime = True
    # 相机这里不用，空着可以让别的窗口（view_camera / check_camera_pose --live）
    # 同时开着看画面 —— 一路相机只能给一个进程出流。
    config.cameras.sources = {}

    env = AR5Env(config)
    env.connect()
    if not getattr(env.arm, "_rt_active", False):
        print("\n  臂的实时通道没起来。通常是没在示教器上上电。")
        env.close()
        return 1
    env.arm.clear_latch()

    here = env.arm.get_joint_positions()
    delta = np.rad2deg(np.abs(q - here[:7])).max()
    print(f"\n  当前姿态 {np.rad2deg(here[:7]).round(1).tolist()}")
    print(f"  最大单关节要走 {delta:.1f}°")
    if delta > 60:
        print(f"  ⚠ 走的角度不小，看清楚周围没有障碍再继续。")
    if args.yes:
        answer = ""
    else:
        print(f"\n  回车开始移动，其它键取消：", end="", flush=True)
        try:
            answer = sys.stdin.readline()
        except KeyboardInterrupt:
            answer = "x"
    if answer.strip() not in ("", "y", "Y"):
        print("  取消。")
        env.close()
        return 0

    print("\n  移动中 ...")
    try:
        reached = env.goto_joints(q, timeout_s=40.0)
    except Exception as exc:  # noqa: BLE001
        print(f"\n  拒绝移动：{exc}")
        env.close()
        return 1

    now = env.arm.get_joint_positions()
    err = np.rad2deg(np.abs(q - now[:7])).max()
    print(f"  {'到位' if reached else '⚠ 超时，没完全到位'}，最大偏差 {err:.3f}°")
    if config.use_hand:
        env.hand.set_opening(opening)
        print(f"  手摆到 {opening:.2f}")

    if args.no_hold:
        print(f"\n  到位，退出。臂由控制器自己保持。")
        env.close()
        return 0

    print(f"\n  ── 臂保持在这个姿态，去调腕部相机 ──")
    print(f"  对照这一帧： replay --episode {args.episode} --cam cam_front  第 {args.frame} 帧")
    print(f"  或者直接用： check_camera_pose --cam cam_front --live")
    print(f"  调好按 q 结束（臂会留在原地，不回 home）\n")

    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        last = 0.0
        while True:
            # 持续下发目标。停 0.25 秒 sender 就会刹车并 latch —— 而调相机
            # 要花好几分钟。
            env.arm.servo_to_joints(q)
            if select.select([sys.stdin], [], [], 0)[0]:
                if sys.stdin.read(1) in ("q", "Q"):
                    break
            t = time.time()
            if t - last >= 2.0:
                last = t
                drift = np.rad2deg(np.abs(env.arm.get_joint_positions()[:7] - q)).max()
                print(f"\r  保持中 … 偏差 {drift:.3f}°   （q 结束）   ", end="", flush=True)
            time.sleep(0.02)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)
        print("\n  结束。臂留在当前姿态。")
        env.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--show", action="store_true", help="只打印姿态，不连臂不动")
    parser.add_argument("--no-hand", action="store_true",
                        help="不驱动手。默认会把手也摆成那一帧的开合度")
    parser.add_argument("--src", default=EPISODES)
    parser.add_argument("--no-hold", action="store_true",
                        help="到位就退出，不停在那里等 q。只是把臂挪过去时用")
    parser.add_argument("--yes", action="store_true", help="跳过确认")
    parser.add_argument("--joints", default="",
                        help="直接给 7 个关节角（度，逗号分隔），不读 episode。"
                             "用来回到某次测试的起始姿态 —— 那个姿态不在采集数据里，"
                             "是从遥操轨迹的第一帧读出来的。\n"
                             "**必须写成 --joints=\"-22.4,...\"**（带等号）—— "
                             "负数开头会被 argparse 当成参数名")
    args = parser.parse_args()

    import logging
    logging.basicConfig(level=logging.INFO, format="  %(message)s")

    if args.joints:
        vals = [float(v) for v in args.joints.replace(" ", "").split(",")]
        if len(vals) != 7:
            print(f"--joints 要 7 个值，给了 {len(vals)} 个")
            return 1
        q = np.deg2rad(np.array(vals, dtype=np.float64))
        opening = 1.0
        print(f"\n  直接指定的姿态")
        print(f"  关节角 deg  {np.round(vals, 2).tolist()}")
        # 顺手把末端位姿和刀具朝向算出来：回位之后要确认刀还是竖直的
        import json as _json

        from atom_robot.ar5.kinematics import AR5Kinematics
        from atom_robot.ar5.transforms import axis_angle_to_matrix  # noqa: F401
        _cfg = AR5Config().arm
        _dh = _json.loads((Path(__file__).resolve().parent / "ar5_dh.json").read_text())["dh"]
        _kin = AR5Kinematics(np.array(_dh).reshape(-1, 4), _cfg.num_joints,
                             _cfg.joint_min, _cfg.joint_max)
        T = _kin.fk(q)
        tilt, axis = min(
            (np.degrees(np.arccos(np.clip(-sgn * T[:3, i][2], -1, 1))), f"{lab}{nm}")
            for i, nm in enumerate("XYZ") for sgn, lab in ((1, "+"), (-1, "-")))
        print(f"  末端位置    {np.round(T[:3, 3], 4).tolist()}")
        print(f"  最接近竖直的轴 {axis}，偏 {tilt:.1f}°"
              f"（螺丝刀在法兰 +Y，<5° 才算摆正）")
        lo = np.array(_cfg.workspace_min)
        hi = np.array(_cfg.workspace_max)
        out = [f"{a} {'低于下限' if T[i, 3] < lo[i] else '超出上限'}"
               for i, a in enumerate("xyz") if T[i, 3] < lo[i] or T[i, 3] > hi[i]]
        print(f"  工作空间盒  {'；'.join(out) if out else '盒内 OK'}")
        if args.show:
            return 0
        return _goto(args, q, opening)

    path = Path(args.src).expanduser() / f"episode_{args.episode:04d}.npz"
    if not path.exists():
        print(f"找不到 {path}")
        return 1
    data = np.load(path, allow_pickle=True)
    state = data["state"]
    if not 0 <= args.frame < len(state):
        print(f"帧号超范围：这条只有 {len(state)} 帧")
        return 1

    q = np.asarray(state[args.frame][:7], dtype=np.float64)
    opening = float(state[args.frame][7])
    pose = data["ee_pose"][args.frame]

    print(f"\n  episode_{args.episode:04d} 第 {args.frame} 帧（共 {len(state)} 帧）")
    print(f"  关节角 deg  {np.rad2deg(q).round(2).tolist()}")
    print(f"  末端位姿    {np.asarray(pose).round(4).tolist()}")
    print(f"  手开合      {opening:.3f}")
    print(f"\n  对应的腕部画面：")
    print(f"    python -m atom_robot.ar5.replay --episode {args.episode} "
          f"--cam cam_front   然后按 → 翻到第 {args.frame} 帧")

    # --show 承诺「只打印姿态，不连臂不动」。把移动那段抽成 _goto 时这一句被漏掉过，
    # 结果 --show 会连臂**并且上电**。两条路（episode / --joints）都要有。
    if args.show:
        return 0

    return _goto(args, q, opening)




if __name__ == "__main__":
    sys.exit(main())
