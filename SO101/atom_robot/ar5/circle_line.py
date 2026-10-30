"""末端画一个竖直圆，再沿基座 +Y 走直线，每走一段停一下。

    python -m atom_robot.ar5.circle_line --plane xz --from-home --dry-run   # 只算，不上电不动
    python -m atom_robot.ar5.circle_line --plane xz --from-home             # 输入 go 后真跑

起点要选好：10-30 臂停在 J7=47.2°（顶着腕部限位），+Y 超过 5mm、圆上的点控制器全拒。
--from-home 先 MoveAbsJ 回 config 的 home_joints（法兰约 400,0,250mm、竖直朝下），
离线算过从那里出发 J6/J7 最大约 25°。不加它就从当前姿态开始。

动作顺序（全程保持起始姿态，控制的是法兰中心，不是指尖）：
  1. 圆：在竖直面（--plane xz 或 yz）里画一整圈，半径 --radius。
     当前位置是圆的最低点，圆心在它正上方，所以整圈都不会低于起始高度。
     用两段 MoveC（各半圈）实现，走完回到起点。
  2. 直线：从起点沿基座 +Y 走 --line 米，每 --step 米停 --pause 秒。

暂停不用 robot.stop()：这台 AR5 上 stop() 之后 2 秒内会掉电（9-20 实测，
SDK 头文件说不掉电，和真机对不上）。这里每段单独下发，等它到位，再睡 --pause 秒。
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from atom_robot.ar5.config import AR5Config  # noqa: E402
from atom_robot.ar5.drivers.rokae_arm import RokaeAR5, SDKError  # noqa: E402

AXIS = {"x": 0, "y": 1, "z": 2}


def offset(start: np.ndarray, dx=0.0, dy=0.0, dz=0.0) -> np.ndarray:
    pose = start.copy()
    pose[:3] += (dx, dy, dz)
    return pose


def circle_points(start: np.ndarray, plane: str, radius: float):
    """两段半圆的 (辅助点, 目标点)，以及用来做可达性检查的整圈采样点。"""
    h = AXIS[plane[0]]  # 竖直面里的水平轴
    centre = offset(start, dz=radius)

    def on_circle(theta):
        # theta=0 在最低点（起点），逆着 h 轴正向往上转
        p = centre.copy()
        p[h] += radius * np.sin(theta)
        p[2] -= radius * np.cos(theta)
        return p

    halves = [(on_circle(np.pi / 2), on_circle(np.pi)),
              (on_circle(3 * np.pi / 2), on_circle(2 * np.pi))]
    samples = [on_circle(t) for t in np.linspace(0, 2 * np.pi, 37)]
    return halves, samples


def line_points(start: np.ndarray, length: float, step: float):
    n = int(round(length / step))
    return [offset(start, dy=step * (i + 1)) for i in range(n)]


def wait_arrive(arm, target: np.ndarray, timeout_s: float, tol: float = 0.001) -> np.ndarray:
    """等到臂不在运动、并且离目标不到 tol 米。

    只看 operationState 不够：moveStart 刚返回时状态可能还没变成 moving，
    这时直接判「停了」就会把下一段叠上去。所以同时要求位置到位。
    """
    sdk = arm.sdk
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        state = arm._call(arm._robot.operationState)
        pose = arm.get_ee_pose()
        if state != sdk.OperationState.moving and np.linalg.norm(pose[:3] - target[:3]) < tol:
            return pose
        time.sleep(0.02)
    raise SDKError(
        f"{timeout_s:.0f} s 内没到位：当前 {np.round(pose[:3] * 1000, 1).tolist()} mm，"
        f"目标 {np.round(target[:3] * 1000, 1).tolist()} mm，状态 {state}"
    )


def run(arm, cmds, cmd_id: str, target: np.ndarray, timeout_s: float) -> np.ndarray:
    sdk = arm.sdk
    arm._call(arm._robot.moveAppend, cmds, sdk.PyString(cmd_id))
    arm._call(arm._robot.moveStart)
    return wait_arrive(arm, target, timeout_s)


def go_home(arm, q_home: np.ndarray, speed: float, timeout_s: float = 60.0) -> None:
    """MoveAbsJ 回 home，按编码器核对到位（MoveAbsJ 会被静默拒，见 home_collect.py）。"""
    sdk = arm.sdk
    cmd = sdk.MoveAbsJCommand(sdk.JointPosition(q_home.tolist()), speed, 0.0)
    arm._call(arm._robot.moveAppend, [cmd], sdk.PyString("home"))
    arm._call(arm._robot.moveStart)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        q = np.asarray(arm._call(arm._robot.jointPos))[: len(q_home)]
        state = arm._call(arm._robot.operationState)
        if state != sdk.OperationState.moving and np.max(np.abs(q - q_home)) < np.radians(0.5):
            return
        time.sleep(0.05)
    raise SDKError(f"{timeout_s:.0f} s 内没回到 home，各轴还差 "
                   f"{np.round(np.degrees(q - q_home), 1).tolist()}°，状态 {state}")


def local_check(arm, q0: np.ndarray, checks) -> bool:
    """不动臂时用本地运动学（控制器标定后的 DH）逐点逆解，看 J6/J7 会不会顶限位。

    控制器的 checkPath 只能从当前姿态算，起点不是当前姿态时只能这样估。
    """
    from atom_robot.ar5.kinematics import AR5Kinematics

    kin = AR5Kinematics.from_robot(arm)
    T0 = kin.fk(q0)
    soft = np.radians(47.1)  # 实测 J6/J7 软限位，比 config 的 50° 紧
    q, peak = q0.copy(), np.zeros(2)
    for k, p in checks:
        T = T0.copy()
        T[:3, 3] = p[:3]
        sol, _ = kin.ik(T, q)
        if sol is None or np.any(np.abs(sol[5:7]) >= soft):
            print(f"  {k}上的 {np.round(p[:3] * 1000, 1).tolist()} mm 逆解不出或 J6/J7 顶限位")
            return False
        q = sol
        peak = np.maximum(peak, np.abs(sol[5:7]))
    print(f"  {len(checks)} 个点都能解，J6/J7 最大 {np.round(np.degrees(peak), 1).tolist()}°")
    return True


def controller_check(arm, config, checks) -> bool:
    lo = np.array(config.workspace_min)
    hi = np.array(config.workspace_max)
    outside = [(k, p) for k, p in checks if np.any(p[:3] < lo) or np.any(p[:3] > hi)]
    if outside:
        k, p = outside[0]
        print(f"  {len(outside)} 个点超出工作空间盒，第一个是{k}上的 "
              f"{np.round(p[:3] * 1000, 1).tolist()} mm")
        return False
    for k, p in checks:
        ok, detail = arm.check_reachable(p)
        if not ok:
            print(f"  {k}上的 {np.round(p[:3] * 1000, 1).tolist()} mm 到不了：{detail}")
            return False
    print(f"  {len(checks)} 个点控制器都说能到")
    return True


def plan(start: np.ndarray, args):
    halves, samples = circle_points(start, args.plane, args.radius)
    stops = line_points(start, args.line, args.step)
    print(f"起点（法兰）  {np.round(start[:3] * 1000, 1).tolist()} mm")
    print(f"圆   {args.plane} 面，半径 {args.radius * 1000:.0f} mm，"
          f"圆心 {np.round((start[:3] + [0, 0, args.radius]) * 1000, 1).tolist()} mm，"
          f"最高到 z={(start[2] + 2 * args.radius) * 1000:.1f} mm")
    print(f"直线 基座 +Y {args.line * 1000:.0f} mm，每 {args.step * 1000:.0f} mm 停 {args.pause:g} s，"
          f"终点 {np.round(stops[-1][:3] * 1000, 1).tolist()} mm")
    print(f"速度 {args.speed:g} mm/s")
    checks = [("圆", p) for p in samples] + [("直线", p) for p in stops]
    return halves, stops, checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plane", required=True, choices=("xz", "yz"), help="圆所在的竖直面")
    parser.add_argument("--radius", type=float, default=0.05, help="圆半径，米")
    parser.add_argument("--line", type=float, default=0.20, help="直线总长，米（沿基座 +Y）")
    parser.add_argument("--step", type=float, default=0.05, help="每走多少米停一次")
    parser.add_argument("--pause", type=float, default=1.0, help="每次停多少秒")
    parser.add_argument("--speed", type=float, default=75.0, help="末端速度，mm/s（回 home 也用它）")
    parser.add_argument("--from-home", action="store_true", help="先 MoveAbsJ 回 config 的 home_joints")
    parser.add_argument("--dry-run", action="store_true", help="只检查轨迹，不上电不动")
    parser.add_argument("--yes", action="store_true", help="跳过输入 go 的确认")
    args = parser.parse_args()

    config = AR5Config().arm
    config.use_realtime = False
    q_home = np.asarray(config.home_joints, dtype=np.float64)

    arm = RokaeAR5(config)
    arm.connect(power_on=False)
    sdk = arm.sdk
    try:
        if args.from_home:
            from atom_robot.ar5.kinematics import AR5Kinematics
            from atom_robot.ar5.transforms import matrix_to_axis_angle

            q_now = np.asarray(arm._call(arm._robot.jointPos))[: len(q_home)]
            print(f"当前关节 {np.round(np.degrees(q_now), 1).tolist()}°")
            print(f"先回 home {np.round(np.degrees(q_home), 1).tolist()}°，"
                  f"最大一轴转 {np.degrees(np.max(np.abs(q_home - q_now))):.0f}°")
            T = AR5Kinematics.from_robot(arm).fk(q_home)
            start = np.concatenate([T[:3, 3], matrix_to_axis_angle(T[:3, :3])])
        else:
            start = arm.get_ee_pose()
        halves, stops, checks = plan(start, args)

        if args.from_home:
            print("\n本地运动学逐点检查（从 home 出发）...")
            if not local_check(arm, q_home, checks):
                return 1
        else:
            print("\n逐点问控制器能不能到 ...")
            if not controller_check(arm, config, checks):
                return 1

        if args.dry_run:
            print("\n只做了检查，臂没动。")
            return 0
        if not args.yes and input("\n输入 go 上电开始跑：").strip() != "go":
            print("取消")
            return 1

        # 顺序不能换：先切 NRT，再 _power_on（它里面 setOperateMode 和 setPowerState 紧挨着）
        arm._call(arm._robot.setMotionControlMode, sdk.MotionControlMode.NrtCommandMode)
        arm._power_on()
        arm._call(arm._robot.moveReset)

        if args.from_home:
            print("\n回 home ...")
            go_home(arm, q_home, args.speed)
            # 以真到的位姿为准重新出轨迹，再让控制器从这里逐点核一遍
            start = arm.get_ee_pose()
            halves, stops, checks = plan(start, args)
            print("逐点问控制器能不能到 ...")
            if not controller_check(arm, config, checks):
                return 1

        # zone=0 = 每段都停稳再走（<1 是 fine 档），段与段之间不圆滑过渡
        speed, zone = args.speed, 0.0
        circumference = 2 * np.pi * args.radius
        circle_timeout = 10.0 + 3 * circumference * 1000 / speed

        print("\n画圆 ...")
        t0 = time.time()
        cmds = [sdk.MoveCCommand(arm._to_cartesian_position(target),
                                 arm._to_cartesian_position(aux), speed, zone)
                for aux, target in halves]
        end = run(arm, cmds, "circle", halves[-1][1], circle_timeout)
        print(f"  {time.time() - t0:.1f} s，回到起点误差 {np.linalg.norm(end[:3] - start[:3]) * 1000:.2f} mm")

        print("\n走直线 ...")
        seg_timeout = 10.0 + 3 * args.step * 1000 / speed
        for i, target in enumerate(stops, 1):
            cmd = sdk.MoveLCommand(arm._to_cartesian_position(target), speed, zone)
            t0 = time.time()
            pose = run(arm, [cmd], f"line{i}", target, seg_timeout)
            print(f"  第 {i} 段 {time.time() - t0:.1f} s 到位，已走 {(pose[1] - start[1]) * 1000:.1f} mm，"
                  f"误差 {np.linalg.norm(pose[:3] - target[:3]) * 1000:.2f} mm")
            time.sleep(args.pause)

        print(f"\n完成，停在 {np.round(arm.get_ee_pose()[:3] * 1000, 1).tolist()} mm"
              f"，电源 {arm._call(arm._robot.powerState)}")
    except KeyboardInterrupt:
        # 不调 stop()（会掉电）。队列里一次只有一段，臂会走完这一段停下，不会再接下一段；
        # 运动中 moveReset 会被拒，所以失败不算问题
        print("\n中断：臂走完当前这一段后停下，不再下发")
        try:
            arm._call(arm._robot.moveReset)
        except SDKError as exc:
            print(f"  moveReset 被拒（运动中正常）：{exc}")
    finally:
        arm.disconnect()
        print("已断开")
    return 0


if __name__ == "__main__":
    sys.exit(main())
