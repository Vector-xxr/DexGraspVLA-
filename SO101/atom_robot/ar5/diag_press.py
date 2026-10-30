#!/usr/bin/env python3
"""量末端的接触力 —— 回答「位置环那 15 mm 的 lead 到底压出多少牛」。

    python -m atom_robot.ar5.diag_press              # 先归零，再持续读
    python -m atom_robot.ar5.diag_press --seconds 20 # 读 20 秒就退出
    python -m atom_robot.ar5.diag_press --no-tare    # 不归零，看原始值

**只读。** connect(power_on=False)，不上电、不发任何运动指令、不碰实时流。
可以在 teleop 跑着的时候开第二个终端跑它。

为什么必须归零
──────────────────────────────────────────────────────────────────────
`ArmConfig.payload_mass = 0`，而手加法兰实测 1.2~1.3 kg。控制器扣动力学模型的
时候按 0 kg 扣，于是那 12~13 N 的重力**整个留在 external_torque / cart_force 里**，
和真正的接触力混在一起。再加上传感器本身有零漂（两个姿态实测模长差 1.26 N）。

所以原始读数没有意义，**有意义的是变化量**：先在"没顶到东西"的姿态上取基线，
之后报的都是相对基线的增量。姿态一变基线就作废（重力在末端系的投影变了），
所以**归零之后不要再动臂的姿态，只做那一下往下压**。

cart_force 是「末端**受到**的力」，不是臂输出的力。往下压工件 → 工件反推末端
→ 归零后的 ΔFz 是**正**的。关心的是它的大小。
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from atom_robot.ar5.config import AR5Config  # noqa: E402
from atom_robot.ar5.drivers.rokae_arm import RokaeAR5  # noqa: E402

TARE_SAMPLES = 20


def force_control_handle(arm):
    """`getEndTorque` 挂在 `robot.forceControl()` 返回的 PyForceControl7 上，不在 robot 上。

    实测 `xMateErProRobot` 本身没有这个方法（AttributeError），只有
    PyForceControl5/6/7 有。取句柄本身不改控制模式 —— **不调 fcInit / fcStart**，
    那两个会真的进力控，这个脚本要保持只读。
    """
    getter = getattr(arm._robot, "forceControl", None)
    if getter is None:
        return None
    fc = getter()
    return fc if hasattr(fc, "getEndTorque") else None


def read_end_torque(fc, arm, frame):
    """getEndTorque 的一帧：(关节原始力矩, 关节外部力矩, 笛卡尔力矩, 笛卡尔力)。

    SDK 用出参传值：四个 PyTypeVectorDouble 进去，填好之后 .content() 取出来。
    """
    sdk = arm.sdk
    joint_raw = sdk.PyTypeVectorDouble()
    joint_ext = sdk.PyTypeVectorDouble()
    cart_torque = sdk.PyTypeVectorDouble()
    cart_force = sdk.PyTypeVectorDouble()
    arm._call(fc.getEndTorque, frame, joint_raw, joint_ext, cart_torque, cart_force)
    return (np.asarray(joint_raw.content(), dtype=np.float64),
            np.asarray(joint_ext.content(), dtype=np.float64),
            np.asarray(cart_torque.content(), dtype=np.float64),
            np.asarray(cart_force.content(), dtype=np.float64))


def force_from_jacobian(arm):
    """退路：关节力矩 + 雅可比反推末端力，`F = pinv(J^T) @ tau`。

    `jointTorque` 就挂在 robot 上，力控句柄拿不到时这条还在。算出来的是基座系的
    六维力旋量，取前三个分量。**和 getEndTorque 一样带着没扣掉的重力**，所以同样
    要归零；两路都能读到的时候互相当交叉验证。
    """
    kin = getattr(arm, "kinematics", None)
    if kin is None:
        return None
    tau = np.asarray(arm._call(arm._robot.jointTorque), dtype=np.float64)
    n = arm.config.num_joints
    if tau.size < n:
        return None
    J = kin.jacobian(arm.get_joint_positions())
    wrench = np.linalg.pinv(J.T) @ tau[:n]
    return wrench[:3]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ip", default="", help="覆盖控制器 IP")
    parser.add_argument("--seconds", type=float, default=0.0,
                        help="读多少秒，0 = 一直读到 Ctrl-C")
    parser.add_argument("--hz", type=float, default=5.0, help="采样率")
    parser.add_argument("--no-tare", action="store_true", help="不归零，直接看原始值")
    parser.add_argument("--frame", default="world", choices=("world", "flange", "tool"),
                        help="力的参考系。world 的 Z 才是「竖直」")
    args = parser.parse_args()

    config = AR5Config().arm
    if args.ip:
        config.ip = args.ip
    # 只读：不起实时流，connect 也不上电。
    config.use_realtime = False

    arm = RokaeAR5(config)
    arm.connect(power_on=False)
    frame = getattr(arm.sdk.FrameType, args.frame)
    fc = force_control_handle(arm)

    def sample():
        """一帧 (末端力 3 维, 关节外部力矩 或 None)。力控句柄没有就走雅可比。"""
        if fc is not None:
            _, ext, _, force = read_end_torque(fc, arm, frame)
            return force, ext
        return force_from_jacobian(arm), None

    # 在 try 外面定义：归零阶段就抛异常的话，finally 里还要用到它们。
    peak = 0.0
    peak_row = None

    try:
        pose = arm.get_ee_pose()
        print(f"  参考系 {args.frame}    末端 z = {pose[2]:.4f} m")
        if fc is not None:
            print("  力源 getEndTorque（robot.forceControl() 上的，不是 robot 上的）")
        else:
            print("  力源 雅可比反推 pinv(J^T)@tau —— 拿不到力控句柄，退到 jointTorque")
        print(f"  payload_mass = {config.payload_mass} kg（手实测 1.2~1.3 kg，"
              f"所以原始值里带着未扣掉的重力）")

        if sample()[0] is None:
            print("\n  两条路都读不到末端力。jointTorque 能读但没有 kinematics，"
                  "或者力控句柄不可用。")
            return 1

        base_force = np.zeros(3)
        base_ext = None
        if not args.no_tare:
            print(f"\n  归零中：取 {TARE_SAMPLES} 帧基线。**保持现在的姿态，不要顶到任何东西。**")
            samples_f, samples_e = [], []
            for _ in range(TARE_SAMPLES):
                force, ext = sample()
                samples_f.append(force)
                if ext is not None:
                    samples_e.append(ext)
                time.sleep(0.05)
            base_force = np.mean(samples_f, axis=0)
            base_ext = np.mean(samples_e, axis=0) if samples_e else None
            spread = np.ptp(samples_f, axis=0)
            print(f"    基线 F = {np.round(base_force, 2).tolist()} N"
                  f"   帧间抖动 {np.round(spread, 2).tolist()} N")
            print(f"    **噪声地板约 {np.max(spread):.2f} N —— 小于它的读数不算数**")
            print("\n  现在往下压。归零后不要再改姿态，姿态一变基线就作废。")

        print(f"\n  {'ΔFx':>8}{'ΔFy':>8}{'ΔFz':>8}{'|ΔF|':>8}   "
              f"{'z (m)':>8}   外部关节力矩最大的那个")
        deadline = time.time() + args.seconds if args.seconds > 0 else None
        period = 1.0 / max(args.hz, 0.1)

        while deadline is None or time.time() < deadline:
            force, ext = sample()
            d = force - base_force
            mag = float(np.linalg.norm(d))
            z = arm.get_ee_pose()[2]

            if ext is None:
                dext = np.array([])
            elif base_ext is not None and ext.size == base_ext.size:
                dext = ext - base_ext
            else:
                dext = ext
            j = int(np.argmax(np.abs(dext))) if dext.size else -1
            worst = f"J{j + 1} {dext[j]:+.2f} Nm" if j >= 0 else "--"

            if mag > peak:
                peak, peak_row = mag, (d.copy(), z)
            print(f"  {d[0]:>8.2f}{d[1]:>8.2f}{d[2]:>8.2f}{mag:>8.2f}   "
                  f"{z:>8.4f}   {worst}        ", end="\r", flush=True)
            time.sleep(period)

    except KeyboardInterrupt:
        pass
    finally:
        print()
        if peak_row is not None:
            d, z = peak_row
            print(f"\n  峰值 |ΔF| = {peak:.2f} N    "
                  f"分量 [{d[0]:+.2f}, {d[1]:+.2f}, {d[2]:+.2f}] N   在 z = {z:.4f}")
            print(f"  **往下压的那一路是 ΔFz = {d[2]:+.2f} N**")
            print(f"\n  换算：teleop 的 max_lead = {AR5Config().max_lead * 1000:.0f} mm，")
            print(f"  所以笛卡尔刚度约 {abs(d[2]) / (AR5Config().max_lead * 1000):.2f} N/mm。")
            print(f"  抬到 40 mm 大约给 {abs(d[2]) / (AR5Config().max_lead * 1000) * 40:.0f} N"
                  f"（上限是 resync_threshold = {AR5Config().resync_threshold * 1000:.0f} mm）。")
        arm.disconnect()
    return 0


if __name__ == "__main__":
    sys.exit(main())
