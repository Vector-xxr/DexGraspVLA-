#!/usr/bin/env python3
"""手动调灵巧手。读 config.py 里的 HandConfig，和遥操作用的是同一份参数。

    python -m atom_robot.ar5.jog_hand --read              # 只看，不动
    python -m atom_robot.ar5.jog_hand --dof 4 300         # 只动第4维（拇指弯曲）
    python -m atom_robot.ar5.jog_hand --pose 0,0,0,1000,300,100
    python -m atom_robot.ar5.jog_hand --open              # 全张开
    python -m atom_robot.ar5.jog_hand --find-contact      # 改完拇指测接触点

六个维度：
    0 小指   1 无名指   2 中指   3 食指   4 拇指弯曲   5 拇指旋转

角度 1000 = 完全张开，0 = 完全握紧。

改了拇指（第 4 或第 5 维）就必须重测接触点 —— 拇指是挡住食指的那个东西，
它一动，食指能扣到哪就变了。跑 --find-contact，它会把新的 finger_close 算给你，
填进 config.py 的三个地方（finger_close、closed_pose、fixed_pose）。

力控和速度掉电不保存，这个脚本每次动之前都会按 config 重设一遍。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from atom_robot.ar5.config import AR5Config  # noqa: E402

DOF_NAMES = ["小指", "无名指", "中指", "食指", "拇指弯曲", "拇指旋转"]


def open_hand(cfg):
    """连上手，顺便把力控和速度设回 config 里的值。"""
    from atom_robot import vel_path as _vel
    vel_path = _vel("VEL", "src", "hand_teleop")
    if str(vel_path) not in sys.path:
        sys.path.insert(0, str(vel_path))
    from inspire_hand import InspireHand

    hand = InspireHand(port=cfg.port, baudrate=cfg.baudrate,
                       dev_id=cfg.dev_id, protocol=cfg.protocol)
    hand.connect()
    hand.set_force(cfg.force)
    hand.set_speed(cfg.speed)
    return hand


def show(hand, cfg, note=""):
    angles = hand.get_angles()
    print(f"\n  当前角度{note}：")
    for i, (name, value) in enumerate(zip(DOF_NAMES, angles)):
        tag = "  ← 食指" if i == cfg.index_dof else ""
        print(f"    {i}  {name:<8} {value:>5}{tag}")
    return angles


def find_contact(hand, cfg, settle: float, step: int):
    """食指一档一档往下扣，扣不动的那个位置就是拇指挡住它的地方。

    判据是「指令给下去了、实际没跟上」，不是靠看读数稳没稳 —— 手指走得慢的时候
    连续几次读数一样很正常，那样会把还在走的当成被挡住。所以每档固定等足时间。
    """
    print(f"\n  先摆姿态 {list(cfg.fixed_pose)}，食指张开")
    pose = list(cfg.fixed_pose)
    pose[cfg.index_dof] = cfg.finger_open
    hand.set_angles(pose)
    time.sleep(settle + 1.5)
    actual = hand.get_angles()
    print(f"  到位 {actual}   拇指 {actual[4]}/{actual[5]}")

    print(f"\n  食指往下扣（每档等 {settle:.1f}s）：")
    print(f"    {'指令':>6}{'实际':>8}{'偏差':>8}")
    contact = None
    for target in range(cfg.finger_open - 20, 150, -step):
        hand.set_finger(cfg.index_dof, target)
        time.sleep(settle)
        actual = hand.get_angles()[cfg.index_dof]
        deviation = actual - target
        mark = ""
        if abs(deviation) > 15 and contact is None:
            contact = actual
            mark = "   ← 扣不动了，这就是接触点"
        print(f"    {target:>6}{actual:>8}{deviation:>+8}{mark}")
        if contact is not None:
            break

    hand.set_finger(cfg.index_dof, cfg.finger_open)
    time.sleep(settle)
    print(f"\n  食指张开回 {hand.get_angles()[cfg.index_dof]}")

    if contact is None:
        print("\n  一路扣到底都没碰到拇指。拇指伸太开了，把第4维调大一点再试。")
        return
    close = contact - 14
    print(f"\n  接触点 {contact}   ->   finger_close = {close}")
    print(f"\n  填进 config.py 的 HandConfig（三个地方都要改）：")
    print(f"    finger_close: int = {close}")
    pose = list(cfg.fixed_pose)
    print(f"    fixed_pose  = {tuple(pose)}")
    pose[cfg.index_dof] = close
    print(f"    closed_pose = {tuple(pose)}")
    pose[cfg.index_dof] = cfg.finger_open
    print(f"    open_pose   = {tuple(pose)}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--read", action="store_true", help="只看当前角度")
    parser.add_argument("--apply", action="store_true",
                        help="按 config.py 里现在的参数把手摆好（张开状态）")
    parser.add_argument("--grip", type=float, metavar="开合度",
                        help="试某个开合度，1=张开 0=闭合。两指会一起动")
    parser.add_argument("--dof", nargs=2, type=int, metavar=("N", "角度"),
                        help="只动第 N 维，其余不动")
    parser.add_argument("--pose", help="六个值，逗号分隔")
    parser.add_argument("--open", action="store_true", help="全部张开")
    parser.add_argument("--find-contact", action="store_true",
                        help="测食指被拇指挡住的位置，算出新的 finger_close")
    parser.add_argument("--settle", type=float, default=2.5, help="每步等多少秒")
    parser.add_argument("--step", type=int, default=40, help="--find-contact 每档降多少")
    args = parser.parse_args()

    moves = [args.dof is not None, args.pose is not None, args.open,
             args.find_contact, args.apply, args.grip is not None]
    if sum(moves) > 1:
        raise SystemExit("一次只能给一个动作参数")

    cfg = AR5Config().hand
    print(f"  手 {cfg.port} @{cfg.baudrate} id={cfg.dev_id}")
    print(f"  力控 {cfg.force}g  速度 {cfg.speed}   固定姿态 {list(cfg.fixed_pose)}")
    print(f"  食指行程 {cfg.finger_close}（闭）~ {cfg.finger_open}（开）")

    hand = open_hand(cfg)
    try:
        if args.find_contact:
            find_contact(hand, cfg, args.settle, args.step)
            return 0

        if args.grip is not None:
            pose = list(cfg.pose_for(args.grip))
            hand.set_angles(pose)
            print(f"\n  开合度 {args.grip}  ->  下发 {pose}，等 {args.settle:.1f}s")
            time.sleep(args.settle + 1.0)
            actual = show(hand, cfg, f"（开合度 {args.grip}）")
            index_gap = actual[cfg.index_dof] - pose[cfg.index_dof]
            thumb_gap = actual[4] - pose[4]
            print(f"\n  食指 差 {index_gap:+d}    拇指 差 {thumb_gap:+d}")
            if abs(index_gap) > 15 or abs(thumb_gap) > 15:
                print("  有手指没走到位 —— 两指顶上了，或者夹住东西了")
            else:
                print("  两指都到位了，中间是空的（没夹到东西也没互相顶）")
        elif args.apply:
            pose = list(cfg.fixed_pose)
            hand.set_angles(pose)
            print(f"\n  按 config 摆姿态 {pose}，等 {args.settle:.1f}s")
            time.sleep(args.settle + 1.5)
            actual = show(hand, cfg, "（摆好后）")
            drift = [(i, DOF_NAMES[i], pose[i], actual[i])
                     for i in range(6) if abs(actual[i] - pose[i]) > 15]
            if drift:
                print("\n  下面这几维没走到位（被挡住，或者到机械限位了）：")
                for i, name, want, got in drift:
                    print(f"    {i} {name}: 要 {want}，实际 {got}")
            print(f"\n  拇指现在 bend={actual[4]} rot={actual[5]}，"
                  f"finger_close={cfg.finger_close}")
            print("  刚改过拇指的话，跑 --find-contact 重测 finger_close")
        elif args.open:
            hand.set_angles([cfg.finger_open] * 6)
            print(f"\n  全部张开，等 {args.settle:.1f}s")
            time.sleep(args.settle)
        elif args.pose:
            values = [int(v) for v in args.pose.split(",")]
            if len(values) != 6:
                raise SystemExit(f"--pose 要六个值，给了 {len(values)} 个")
            hand.set_angles(values)
            print(f"\n  下发 {values}，等 {args.settle:.1f}s")
            time.sleep(args.settle)
        elif args.dof:
            dof, angle = args.dof
            if not 0 <= dof <= 5:
                raise SystemExit(f"维度只能是 0..5，给了 {dof}")
            if dof in (4, 5):
                print(f"\n  注意：动的是拇指，动完要跑 --find-contact 重测 finger_close")
            hand.set_finger(dof, angle)
            print(f"\n  第{dof}维（{DOF_NAMES[dof]}）-> {angle}，等 {args.settle:.1f}s")
            time.sleep(args.settle)

        show(hand, cfg)
        print("\n  [诊断]")
        print(hand.diagnose())
    finally:
        hand.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
