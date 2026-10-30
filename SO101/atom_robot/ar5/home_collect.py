#!/usr/bin/env python3
"""把 AR5 送回**采集的 0 位**（映射中心 `centre_joints`，主臂全零时从臂该在的姿势）。

    cd ~/yuhang_workspace/算法/DexGraspVLA-/SO101
    PY=~/miniforge3/envs/dexgraspvla/bin/python
    PYTHONPATH= $PY -m atom_robot.ar5.home_collect            # 诊断 → 上电 → 打出要走多远 → 问一次再动
    PYTHONPATH= $PY -m atom_robot.ar5.home_collect --check    # 只诊断、打目标姿势，不上电、不动
    PYTHONPATH= $PY -m atom_robot.ar5.home_collect --yes      # 不问，直接动

参考 `数据_硬件/灵巧手遥操/VR设备/home_arm.py`：**不开 RT**，走排队式 MoveAbsJ，由控制器
自己规划。比 RT 逐步伺服温和，也不吃跳变门 —— 2026-10-20 采集里录制中按 q，RT 回位每一步
都被跳变门拒（指令跑在臂前面 0.193 rad），回不去；这条路没有这个问题。

和 home_arm.py 不一样的地方：

* 目标是**采集的 0 位** `centre_joints`，不是 VR 遥操的 `home_joints`（两者 j4 差 45.6°）。
* **有参数解析**：`--help` 不会让臂动（home_arm.py 不解析参数，--help 照样连臂回位）。
* **默认不改控制器里的负载**：home_arm 写死 1.0 kg 是因时手的；现在装的是 OmniPicker，
  重量没称过。要设就给 `--payload-kg`（和 `--payload-com-z`）。
* **上电前先诊断**：急停、示教器占着控制权（ec=-513）会说清楚是哪一种，上电交给 power_on.py。
* **到没到位看编码器**：MoveAbsJ 会被静默拒（10-09 实测 reachTarget=False、ec=0），
  `_wait_for_motion` 返回也不代表到了，所以动完再读关节角核对。
* 只连臂：不碰夹爪、相机。采集开着的时候拒绝运行（同一台臂不能两个会话驱动）。
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from atom_robot.ar5.config import AR5Config  # noqa: E402
from atom_robot.ar5.joint_mapping import JointMapConfig  # noqa: E402

ARRIVED_DEG = 0.5            # 每个关节离目标都在这以内才算到位


def collect_running():
    """有没有 collect / teleop 正开着（看命令行）。它们握着这台臂的会话。"""
    me = os.getpid()
    found = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == me:
            continue
        try:
            cmd = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            continue
        if any(m in cmd for m in ("atom_robot.ar5.collect", "atom_robot.ar5.teleop")) \
                and "home_collect" not in cmd:
            found.append(f"PID {proc.name}  {cmd.strip()[:100]}")
    return found


def report(tag, q, target, cfg):
    lo = np.asarray(cfg.joint_min, float) + cfg.joint_limit_margin
    hi = np.asarray(cfg.joint_max, float) - cfg.joint_limit_margin
    margin = np.minimum(q - lo, hi - q)
    delta = np.degrees(target - q)
    print(f"  [{tag}] 关节    " + " ".join(f"j{i}:{np.degrees(v):+7.1f}°" for i, v in enumerate(q)))
    print(f"  [{tag}] 离 0 位 " + " ".join(f"j{i}:{d:+7.1f}°" for i, d in enumerate(delta)))
    bad = [f"j{i}" for i in range(len(q)) if margin[i] < 0]
    if bad:
        print(f"  [{tag}] ⚠ **已在软限位外：{', '.join(bad)}** —— MoveAbsJ 会被静默拒，"
              "先用 灵巧手遥操/VR设备/softlimit_rescue.py 转回来")
    return float(np.max(np.abs(delta)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="只诊断、打目标姿势，不上电、不动")
    ap.add_argument("--yes", action="store_true", help="不问，直接动。确认臂的活动范围里没人再用")
    ap.add_argument("--timeout", type=float, default=40.0, metavar="秒", help="等到位的上限，默认 40")
    ap.add_argument("--payload-kg", type=float, default=None,
                    help="设控制器负载（kg）。默认不改，沿用控制器里存的")
    ap.add_argument("--payload-com-z", type=float, default=0.08, metavar="米",
                    help="负载质心离法兰的 z 距离，配 --payload-kg 用，默认 0.08")
    args = ap.parse_args()

    cfg = AR5Config()
    target = np.asarray(JointMapConfig.load(**cfg.collect.mapping_overrides).centre_joints, float)
    print(f"  采集 0 位（centre_joints）：{np.round(np.degrees(target), 1).tolist()} 度")

    others = collect_running()
    if others:
        print("\n  ✗ 采集 / 遥操还开着，它握着这台臂。去那个终端按 q 退出再来：")
        for line in others:
            print(f"      {line}")
        return 1

    from atom_robot.ar5.unseize import diagnose, power_back_on  # noqa: PLC0415
    state = diagnose(cfg.arm)
    if args.check:
        print("\n  --check：只看不动。要回位去掉 --check。")
        return 0
    if not state["reachable"]:
        print("\n  ✗ 控制器连不上。刚重启 / 碰撞停机后要约 40 秒才接受连接，等一会儿再来。")
        return 2
    if state["physical_stop"]:
        print("\n  ✗ 急停 / 安全门是物理的，软件清不掉：急停按钮顺时针转弹起来，或者去示教器复位。")
        return 1
    if state["seized"] or state["idle_off"]:
        if power_back_on() != 0:
            print("\n  ✗ 上电没成功，看上面 power_on 打的返回码：")
            print("    ec=-3    急停按下 —— 按钮转起来 / 示教器上清软急停")
            print("    ec=-513  切不到自动模式 —— 示教器占着控制权：在示教器上切到自动、退出手动操作")
            return 1

    from atom_robot.ar5.ar5_env import AR5Env  # noqa: PLC0415
    cfg.use_hand = False                 # 只要臂：不碰夹爪、相机
    cfg.cameras.sources = {}
    cfg.arm.use_realtime = False         # ← 关键：不开 RT，走排队式 MoveAbsJ，不吃跳变门
    env = AR5Env(cfg)
    print("\n  连 AR5（不开 RT）...", flush=True)
    env.connect()
    try:
        if args.payload_kg is not None:
            robot = env.arm._robot
            ec = {}
            robot.setLoad(env.arm.sdk.Load(args.payload_kg, [0.0, 0.0, args.payload_com_z],
                                           [0.0, 0.0, 0.0]), ec)
            print(f"  [负载] {args.payload_kg} kg、质心 z={args.payload_com_z} m（ec={ec.get('ec', 0)}）")

        q = np.asarray(env.arm.get_joint_positions(), float)
        far = report("回位前", q, target, cfg.arm)
        if far < ARRIVED_DEG:
            print("\n  已经在 0 位了，不用动。")
            return 0
        if not args.yes:
            if not sys.stdin.isatty():
                print("\n  ✗ 没有终端可以问你确认。确认活动范围里没人，加 --yes 再跑。")
                return 1
            answer = input(f"\n  臂要移动回采集 0 位（最大关节变化 {far:.1f}°），活动范围里没人吗？[y/N] ")
            if answer.strip().lower() not in ("y", "yes", "是"):
                print("  没动。")
                return 0

        print("  ⚠ 臂开始移动（排队式 MoveAbsJ）...", flush=True)
        env.goto_joints(target, blocking=True, timeout_s=args.timeout)
        # MoveAbsJ 会被静默拒、_wait_for_motion 也可能在动起来之前就返回 —— 以编码器为准
        deadline = time.time() + args.timeout
        while time.time() < deadline:
            q = np.asarray(env.arm.get_joint_positions(), float)
            if np.max(np.abs(np.degrees(target - q))) < ARRIVED_DEG:
                break
            time.sleep(0.2)
        far = report("回位后", q, target, cfg.arm)
        if far < ARRIVED_DEG:
            print("\n  ✓ 臂已回到采集 0 位。")
            return 0
        print(f"\n  ✗ {args.timeout:.0f}s 内没到位（最大还差 {far:.1f}°）。MoveAbsJ 可能被控制器拒了 ——"
              "看示教器的报警，关节在软限位外就先 softlimit_rescue.py。")
        return 1
    finally:
        env.close()


if __name__ == "__main__":
    sys.exit(main())
