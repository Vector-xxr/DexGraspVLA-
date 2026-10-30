#!/usr/bin/env python3
"""验活：`jointTorque()` / `jointVel()` 在臂动起来的时候到底是不是活的。

    python -m atom_robot.ar5.diag_torque
        完整流程：静止基线 → 拖动模式（你用手拖）→ 判定 → 关拖动。

    python -m atom_robot.ar5.diag_torque --static-only
        只跑静止基线，不开拖动。臂全程不可被拖动。

**这个脚本一条运动指令都不下发**，不进 RtCommandMode，不碰限幅。
臂唯一会动的方式是你用手拖它。

────────────────────────────────────────────────────────────────────────
为什么必须先做这一步
────────────────────────────────────────────────────────────────────────

力矩**不在实时状态流里**。`RtSupportedFields` 全集只有 8 个字段：

    jointPos_m  tcpPose_m  tcpPoseAbc_m  elbow_m
    keypads  exJointPos_m  exJointVel_m  exMotor_m

没有力矩、没有主轴速度、没有电流。所以 `jointTorque()` 只能走 NRT 缓存
路径 —— 而 `drivers/rokae_arm.py` 里明确记着，同一套缓存上的 `jointPos()`
在没订阅状态流时会"serve a snapshot that never refreshes"：实测 184 个采样
读数逐位相同，而臂实际动了 8.8°。

力矩又恰恰**没法订阅**。所以"运动中 jointTorque 会不会返回冻结的旧值"
是未知的。如果它是冻结的，后面整轮力矩测量记下来的会是同一个数。

────────────────────────────────────────────────────────────────────────
判据
────────────────────────────────────────────────────────────────────────

关键指标不是峰峰值，是**变化率** = 相邻两个采样之间该向量发生变化的比例。

    拖动阶段   jointPos 变化率高、jointTorque 变化率 ≈ 0   →   力矩是冻结的
    拖动阶段   两者变化率都高                              →   力矩是活的

静止基线的作用是反证：静止时变化率低是正常的（因为真的没动），不能用它
判断冻结 —— 必须在确实动起来的时候比。

同时做一个 A/B：`updateRobotState()` 到底是不是力矩刷新的必要条件。

    变体 A   直接调 jointTorque()，不调 updateRobotState()
    变体 B   先 updateRobotState() 再调 jointTorque()

如果 A 冻结而 B 是活的，那后面的采集线程每次都必须先 updateRobotState。
顺带量出两种变体的单次耗时，决定采集线程能跑多快。
"""

from __future__ import annotations

import argparse
import datetime
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from atom_robot.ar5.config import ArmConfig  # noqa: E402
from atom_robot.ar5.drivers.rokae_arm import (  # noqa: E402
    load_sdk,
    resolve_robot_class,
)

NUM_JOINTS = 7


def call(fn, *args):
    """调一个用尾部 dict 报错的 SDK 方法。成功时 ec 也会被填成 {'ec': 0, ...}。"""
    ec = {}
    result = fn(*args, ec)
    code = ec.get("ec", 0) if isinstance(ec, dict) else 0
    if code:
        name = getattr(fn, "__name__", str(fn))
        raise RuntimeError(f"{name} 失败 (ec={code}): {ec.get('message', ec)}")
    return result


def try_call(fn, *args):
    """同上，但失败只返回 None，不抛。用于可选能力的探测。"""
    try:
        return call(fn, *args)
    except Exception as exc:  # noqa: BLE001
        return f"<失败: {exc}>"


# ── 采样 ────────────────────────────────────────────────────────────────────


def sample_once(robot, refresh: bool):
    """采一帧。返回 (位置, 速度, 力矩, 各自耗时秒)。取不到的量返回 None。"""
    t_refresh = 0.0
    if refresh:
        t0 = time.perf_counter()
        # 超时给 2 ms，和驱动里 refresh_state() 用的一样
        robot.updateRobotState(datetime.timedelta(milliseconds=2))
        t_refresh = time.perf_counter() - t0

    t0 = time.perf_counter()
    pos = call(robot.jointPos)
    t_pos = time.perf_counter() - t0

    t0 = time.perf_counter()
    try:
        vel = call(robot.jointVel)
    except Exception:  # noqa: BLE001
        vel = None
    t_vel = time.perf_counter() - t0

    t0 = time.perf_counter()
    try:
        trq = call(robot.jointTorque)
    except Exception:  # noqa: BLE001
        trq = None
    t_trq = time.perf_counter() - t0

    def trim(v):
        # jointPos / jointVel 会用外部轴槽位补长，真关节只有前 num_joints 个
        if v is None:
            return None
        a = np.asarray(v, dtype=np.float64)
        return a[:NUM_JOINTS] if a.size >= NUM_JOINTS else a

    return trim(pos), trim(vel), trim(trq), (t_refresh, t_pos, t_vel, t_trq)


def collect(robot, seconds: float, refresh: bool, label: str):
    """连采 `seconds` 秒，尽可能快。返回一个 dict。"""
    pos, vel, trq, times = [], [], [], []
    t_end = time.perf_counter() + seconds
    t_start = time.perf_counter()
    while time.perf_counter() < t_end:
        p, v, q, dt = sample_once(robot, refresh)
        pos.append(p)
        vel.append(v)
        trq.append(q)
        times.append(dt)
    elapsed = time.perf_counter() - t_start

    def stack(rows):
        good = [r for r in rows if r is not None]
        if not good or len(good) != len(rows):
            return None
        return np.vstack(good)

    return {
        "label": label,
        "refresh": refresh,
        "n": len(pos),
        "elapsed": elapsed,
        "hz": len(pos) / elapsed if elapsed > 0 else 0.0,
        "pos": stack(pos),
        "vel": stack(vel),
        "trq": stack(trq),
        "times": np.asarray(times, dtype=np.float64),
    }


# ── 统计 ────────────────────────────────────────────────────────────────────


def change_rate(arr) -> float:
    """相邻采样之间整个向量发生任何变化的比例。这是判冻结的核心指标。"""
    if arr is None or arr.shape[0] < 2:
        return float("nan")
    return float(np.mean(np.any(np.diff(arr, axis=0) != 0.0, axis=1)))


def distinct_rows(arr) -> int:
    if arr is None or arr.size == 0:
        return 0
    return len(np.unique(arr, axis=0))


def p2p(arr):
    if arr is None or arr.size == 0:
        return None
    return arr.max(axis=0) - arr.min(axis=0)


def describe(run) -> None:
    print(f"\n── {run['label']} " + "─" * max(0, 60 - len(run["label"])))
    print(f"   采样 {run['n']} 个 / {run['elapsed']:.2f} s = {run['hz']:.1f} Hz"
          f"   （updateRobotState: {'调了' if run['refresh'] else '没调'}）")

    t = run["times"]
    if t.size:
        names = ["updateRobotState", "jointPos", "jointVel", "jointTorque"]
        parts = []
        for i, name in enumerate(names):
            col = np.sort(t[:, i]) * 1000.0
            if col[-1] <= 0.0:
                continue
            parts.append(f"{name} p50 {col[col.size // 2]:.2f} p95 {col[int(col.size * 0.95)]:.2f} ms")
        if parts:
            print("   单次耗时  " + "   ".join(parts))

    for key, unit, label in (("pos", "rad", "jointPos   "),
                             ("vel", "rad/s", "jointVel   "),
                             ("trq", "N·m", "jointTorque")):
        arr = run[key]
        if arr is None:
            print(f"   {label}  ❌ 读不到（接口不存在或报错）")
            continue
        rate = change_rate(arr)
        span = p2p(arr)
        print(f"   {label}  变化率 {rate * 100:5.1f}%   不同取值 {distinct_rows(arr):4d}/{arr.shape[0]}"
              f"   峰峰值 [{', '.join(f'{x:.3f}' for x in span)}] {unit}")


def verdict(drag_runs) -> int:
    """按拖动阶段的变化率下判定。返回进程退出码。"""
    print("\n" + "=" * 72)
    print("判定")
    print("=" * 72)

    if not drag_runs:
        print("  没有拖动阶段的数据，无法判定。静止时读数不变是正常的，")
        print("  不能用来证明力矩冻结 —— 必须在臂确实动起来时比。")
        return 2

    worst = 0
    for run in drag_runs:
        tag = "先 updateRobotState" if run["refresh"] else "不调 updateRobotState"
        pos_rate = change_rate(run["pos"])
        trq_rate = change_rate(run["trq"]) if run["trq"] is not None else float("nan")
        vel_rate = change_rate(run["vel"]) if run["vel"] is not None else float("nan")

        print(f"\n  [{tag}]")
        if not (pos_rate > 0.2):
            print(f"    ⚠ jointPos 变化率只有 {pos_rate * 100:.1f}% —— 臂没被拖动，这一段不算数。")
            worst = max(worst, 2)
            continue

        for name, rate in (("jointTorque", trq_rate), ("jointVel", vel_rate)):
            if np.isnan(rate):
                print(f"    ❌ {name} 读不到")
                worst = max(worst, 3)
            elif rate < 0.02:
                print(f"    ❌ {name} 变化率 {rate * 100:.1f}%（位置 {pos_rate * 100:.1f}%）—— **冻结**")
                worst = max(worst, 3)
            elif rate < 0.5 * pos_rate:
                print(f"    ⚠ {name} 变化率 {rate * 100:.1f}%，明显低于位置的 {pos_rate * 100:.1f}%"
                      f" —— 疑似降频刷新，采集时要按这个实际速率算")
                worst = max(worst, 1)
            else:
                print(f"    ✅ {name} 变化率 {rate * 100:.1f}%（位置 {pos_rate * 100:.1f}%）—— 活的")

    print()
    if worst >= 3:
        print("  结论：力矩/速度这条路不通，后面的测量方案要换。")
    elif worst == 2:
        print("  结论：这一轮没测到有效数据，重跑并确保拖动幅度足够大。")
    elif worst == 1:
        print("  结论：能用，但刷新速率低于采样速率，采集线程按实际速率设计。")
    else:
        print("  结论：力矩和速度都是活的，可以往下做采集。")
    return worst


# ── 主流程 ──────────────────────────────────────────────────────────────────


def main() -> int:
    cfg = ArmConfig()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ip", default=cfg.ip)
    ap.add_argument("--local-ip", default=cfg.local_ip)
    ap.add_argument("--sdk-path", default=cfg.sdk_path)
    ap.add_argument("--seconds", type=float, default=6.0,
                    help="每个子阶段采多久（秒），默认 6")
    ap.add_argument("--static-only", action="store_true",
                    help="只跑静止基线，不开拖动")
    args = ap.parse_args()

    sdk = load_sdk(args.sdk_path)
    robot_cls = resolve_robot_class(sdk, cfg.robot_class)

    print("=" * 72)
    print("jointTorque / jointVel 验活   —— 本脚本不下发任何运动指令")
    print("=" * 72)
    print(f"  臂 {args.ip}   本机 {args.local_ip}")

    robot = robot_cls(args.ip, args.local_ip)
    call(robot.connectToRobot)

    info = call(robot.robotInfo)
    print(f"  型号 {info.type}   xCore {info.version}   {info.joint_num} 轴")
    if info.joint_num != NUM_JOINTS:
        print(f"  ⚠ 控制器报 {info.joint_num} 轴，脚本按 {NUM_JOINTS} 轴写的")

    drag_on = False
    try:
        # 已经上电就不要再上一次 —— 重复 setPowerState 没有意义，而且
        # 臂本来就是带电抱位的常态（power_off_on_disconnect 默认 False）。
        power = try_call(robot.powerState)
        state = try_call(robot.operationState)
        print(f"  电源状态 {power}   运行状态 {state}   操作模式 {try_call(robot.operateMode)}")

        if power != sdk.PowerState.on:
            try:
                call(robot.setOperateMode, sdk.OperateMode.automatic)
                call(robot.setPowerState, True)
                print("  已上电，automatic 模式")
            except Exception as exc:  # noqa: BLE001
                print(f"  ⚠ SDK 上电失败（{exc}）")
                print("    有外部使能开关/示教器的话要在那边使能。只读不受影响。")

        # 上一次会话如果没清干净，控制器会一直停在 rtControlling。
        # 这个残留状态会让 enableDrag 被拒，先退回 Nrt。
        if state == sdk.OperationState.rtControlling:
            print("  ⚠ 控制器停在 rtControlling（上次会话残留，当前无人连接）")
            print("    退回 NrtCommandMode ...")
            r = try_call(robot.setMotionControlMode, sdk.MotionControlMode.NrtCommandMode)
            print(f"    结果: {'ok' if r is None else r}   运行状态 {try_call(robot.operationState)}")

        # 订阅状态流。位置要靠它刷新；力矩没法订阅，这正是要验的点。
        #
        # ⚠ startReceiveRobotState / stopReceiveRobotState 是 SDK 里少数
        # **不带尾部 ec 字典**的方法，必须直接调，不能过 call()。
        # 签名是 (timedelta, list[str])，多传一个 dict 会 TypeError。
        robot.startReceiveRobotState(
            datetime.timedelta(milliseconds=1),
            [sdk.RtSupportedFields.jointPos_m, sdk.RtSupportedFields.tcpPose_m])
        print("  状态流已订阅 (jointPos_m, tcpPose_m)")

        runs, drag_runs = [], []

        # ── 阶段 1：静止基线 ──────────────────────────────────────────────
        print("\n" + "─" * 72)
        print("阶段 1 / 静止基线   —— 别碰臂")
        print("─" * 72)
        input("  准备好按回车（确认没人碰臂）...")
        for refresh in (False, True):
            runs.append(collect(robot, args.seconds / 2, refresh,
                                f"静止 · {'先 updateRobotState' if refresh else '不调 updateRobotState'}"))
            describe(runs[-1])

        # ── 阶段 2：拖动 ─────────────────────────────────────────────────
        if not args.static_only:
            print("\n" + "─" * 72)
            print("阶段 2 / 拖动模式   —— 你用手拖臂")
            print("─" * 72)
            print("  ⚠ 开拖动之后臂会变成可徒手推动。开之前确认：")
            print("     · 人站在臂的侧面，不在它倒下来的方向上")
            print("     · 急停在手边")
            print("     · payload_mass 目前是 0，而手+法兰约 1 kg ——")
            print("       重力补偿少这一块，松手时腕部会往下坠，别让它砸到台面")
            if input("  确认后输入 yes 回车（其他任何输入都会跳过拖动）: ").strip().lower() != "yes":
                print("  跳过拖动阶段。")
            else:
                call(robot.enableDrag,
                     sdk.DragParameterSpace.jointSpace,
                     sdk.DragParameterType.freely,
                     True)
                drag_on = True
                print("  拖动已打开。")

                for refresh in (False, True):
                    tag = "先 updateRobotState" if refresh else "不调 updateRobotState"
                    input(f"\n  ▶ 现在开始**持续大幅拖动臂**（七个关节都动一动），"
                          f"按回车开始采 {args.seconds / 2:.0f} 秒 [{tag}]...")
                    print("     采集中，别停手 ...")
                    run = collect(robot, args.seconds / 2, refresh, f"拖动 · {tag}")
                    runs.append(run)
                    drag_runs.append(run)
                    describe(run)

        code = verdict(drag_runs)

        # 存一份原始数据，后面要复核
        out = Path(__file__).resolve().parent / "_rokae_log_" / "diag_torque.npz"
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out, **{f"{i}_{k}": v for i, r in enumerate(runs)
                         for k, v in r.items()
                         if isinstance(v, np.ndarray)})
        print(f"\n  原始数据已存 {out}")
        return code

    finally:
        if drag_on:
            print("  关闭拖动 ...")
            try_call(robot.disableDrag)
        try:
            robot.stopReceiveRobotState()   # 同样不带 ec
        except Exception:  # noqa: BLE001
            pass
        try_call(robot.disconnectFromRobot)
        print("  已断开（没有断电，臂保持带电抱位）")


if __name__ == "__main__":
    sys.exit(main())
