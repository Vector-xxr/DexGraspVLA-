"""Leader-follower teleoperation: SO-101 leader drives the AR5.

The leader's tool motion is mapped to AR5 Cartesian deltas and pushed through
``AR5Env.step``, so teleoperation exercises exactly the path the policy will
later drive -- same clamps, same reachability gate, same action layout. What
gets recorded here is therefore directly usable as demonstration data.

``--calib-id`` picks which saved leader calibration to use. It is only optional
when exactly one is saved; with several, the run stops and lists them.

    # no follower needed: check the mapping and watch the actions
    python -m atom_robot.ar5.teleop --mock --calib-id <name>

    # simulated arm, real hand: the whole gripper path with nothing that collides
    python -m atom_robot.ar5.teleop --mock --real-hand --calib-id <name>

    # arm only, hand not wired yet
    python -m atom_robot.ar5.teleop --no-hand --calib-id <name>

    # everything
    python -m atom_robot.ar5.teleop --calib-id <name>

Clutch: motion only reaches the follower while engaged. Press Enter to toggle.
Start disengaged, move the leader, watch the printed actions, and only engage
once the numbers look right.

``--log-actions <文件.npz>`` 记逐步轨迹，66 列：

    [0]      时间戳
    [1]      离合（1=接合）
    [2:5]    主臂工具位置 xyz
    [5:12]   下发的 7 维动作
    [12:18]  从臂末端位姿（全六维：位置 + 轴角）
    [18:25]  q_goal    发布给跟踪器的关节目标
    [25:32]  q_cmd     跟踪器实际命令的关节角
    [32:39]  q_actual  读回来的关节角
    [39]     RT 发送计数
    [40:47]  关节力矩 joint_torque_measured，Nm（力传感器原始值，含重力）
    [47:54]  **关节空间外力估计** external_torque_measured，Nm
             （控制器按机器人模型扣掉动力学之后的各轴外部力矩）
    [54:57]  **末端六维力估计** cart_force，N（world 系）
    [57:60]  **末端六维力估计** cart_torque，Nm（world 系）
    [60:66]  末端力旋量的本地反推 pinv(J^T)·tau：[0:3] 力 N，[3:6] 力矩 Nm

[40:60] 来自 `forceControl().getEndTorque()` 一次调用，[60:66] 是本地算的。
**两套来路不同，可以互相验证**；而同源的列对得上是必然的，不算验证。

**句柄必须缓存。** 实测 `robot.forceControl()` 取一次句柄要 **41 ms**，而
`getEndTorque()` 只要 **0.30 ms** —— 那 42 ms 是取句柄的开销。2026-09-17 踩过：
每次重取 = 每秒饿死 1 kHz 发送线程，发送率 963/s -> 781/s，控制器把间隔当丢包、
**停止执行却继续收指令**，现象是 q_cmd 一直走、q_actual 冻住、日志一条错都没有。
驱动的 `get_end_wrench` 现在缓存句柄，这一路才能按控制频率采。

力**带着没扣掉的重力**（`payload_mass` 是 0，而手加工具约 1.2~1.3 kg，实测末端
Z 向 -18 N 就是它），`external_torque` 也一样（模型按 0 kg 扣）。所以有意义的是
**变化量**，分析时先拿一段空载的做基线。
"""

import argparse
import select
import sys
import termios
import time
import tty
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from atom_robot.ar5.ar5_env import AR5Env  # noqa: E402
from atom_robot.ar5.config import AR5Config  # noqa: E402
from atom_robot.ar5.leader_bus import SO_JOINTS, LeaderBus  # noqa: E402
from atom_robot.ar5.leader_mapping import (  # noqa: E402
    CALIBRATION_FILE,
    LeaderMapper,
    MappingConfig,
)


LIVE_FLUSH_ROWS = 5        # 攒 5 行刷一次盘；50 Hz 下画面最多晚 100 ms
RSC_J7_LIMIT = 17.0        # RSC 的 J7 传动力矩上限，超了就安全停止 + 掉电抱闸
FORCE_BASE_FRAMES = 40     # 状态行力读数的归零帧数（接合前的空载段）


def _fit(v, n: int) -> np.ndarray:
    """把 SDK 返回的向量裁/补到 n 个。

    几个读接口会用外部轴槽位把返回值补长（`jointPos` 就是），而 `cart_force`
    这类又固定是 3 个。定长拼进轨迹之前统一一下，否则某一帧多回来一个数，
    `np.concatenate` 出来的行长度就和别的行对不上，存盘时才炸。
    """
    a = np.asarray(v, dtype=np.float64).reshape(-1)
    if a.size >= n:
        return a[:n]
    return np.concatenate([a, np.zeros(n - a.size)])


class RawKeyboard:
    """Non-blocking single-key reads, restoring the terminal on the way out."""

    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.saved = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, *exc_info):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.saved)

    def key(self):
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return None


def main() -> int:
    from atom_robot.ar5.end_effector import add_end_effector_args, configure_end_effector
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_end_effector_args(parser)
    parser.add_argument("--leader-port", default="/dev/ttyACM0")
    parser.add_argument("--calib-id", default=AR5Config().collect.calib_id,
                        help="lerobot calibration id; blank uses the only one on disk")
    parser.add_argument("--leader-baud", type=int, default=1_000_000)
    parser.add_argument("--hz", type=float, default=20.0, help="Control rate")
    parser.add_argument("--position-scale", type=float, default=1.0)
    parser.add_argument("--rotation-scale", type=float, default=1.0)
    parser.add_argument("--no-saved-mapping", action="store_true",
                        help="Ignore leader_calibration.json and use the identity axis map")
    parser.add_argument("--rotation-axes", default="xyz", metavar="AXES",
                        help="保留哪几个旋转轴（基座系）。默认 xyz 全开。\n"
                             "**螺丝刀/插销这类要求工具保持竖直的任务用 `z`** ——\n"
                             "绕 X/Y 一动刀头就歪了，绕 Z 的自转才是拧的那一下。\n"
                             "--no-rotation 会把自转也一起关掉。\n"
                             "被关掉的轴每步置零，所以姿态不漂；但**起始姿态要自己摆正**")
    parser.add_argument("--no-rotation", action="store_true", help="Position-only control")
    parser.add_argument("--mock", action="store_true", help="Simulated follower")
    parser.add_argument("--no-hand", action="store_true", help="Run the arm without the dexterous hand")
    parser.add_argument("--real-hand", action="store_true",
                        help="With --mock: simulated arm, real hand. Exercises the whole "
                             "leader->action->hand path before the arm is on the network.")
    parser.add_argument("--no-cameras", action="store_true",
                        help="Skip the cameras. Teleoperation does not need images; recording does.")
    parser.add_argument("--max-step", type=float, default=0.01,
                        help="Per-step translation limit, metres. Start small.")
    parser.add_argument("--queued", action="store_true",
                        help="Use the old queued MoveL path instead of real-time streaming.")
    parser.add_argument("--home", action="store_true",
                        help="Move to the home posture in joint space first. Works from anywhere,\nincluding postures where inverse kinematics has stopped converging.")
    parser.add_argument("--recentre", action="store_true",
                        help="Walk the tool to the middle of the workspace box before starting.")
    parser.add_argument("--log-actions", default="",
                        help="Write per-step leader pose, action and follower pose to this .npz")
    parser.add_argument("--live", default="",
                        help="逐行把轨迹写成 CSV 供实时画图。配 live_plot 用："
                             "teleop --live /tmp/live.csv，另一个终端跑 live_plot。"
                             "**不要在这个进程里画图** —— 渲染会饿死 1 kHz 发送线程")
    args = parser.parse_args()

    mapping = MappingConfig.load(
        path=None if not args.no_saved_mapping else Path("/nonexistent"),
        position_scale=args.position_scale,
        rotation_scale=args.rotation_scale,
        enable_rotation=not args.no_rotation,
        rotation_axes=args.rotation_axes,
    )
    mapper = LeaderMapper(mapping)
    mapper.connect()
    print(f"  axis map: permutation {mapping.axis_permutation} signs {mapping.axis_signs}"
          f"{'' if CALIBRATION_FILE.exists() and not args.no_saved_mapping else '  (UNCALIBRATED)'}")

    leader = LeaderBus.with_calibration(
        args.leader_port, args.calib_id, baudrate=args.leader_baud
    )
    leader.connect()
    leader.assert_torque_disabled()

    if args.real_hand and not args.mock:
        raise SystemExit("--real-hand only means anything with --mock; without it the hand is real anyway")
    config = AR5Config(mock=args.mock, real_hand_in_mock=args.real_hand)
    config.use_hand = not args.no_hand and (not args.mock or args.real_hand)
    configure_end_effector(config, args)
    config.arm.max_step_translation = args.max_step
    # Real-time joint streaming. The queued path cannot keep up with a control
    # loop: it is a trajectory queue, and appending at 20 Hz outruns the rate
    # the controller retires segments until the arm is executing setpoints from
    # seconds ago.
    config.arm.use_realtime = not args.queued
    if args.no_cameras:
        config.cameras.sources = {}
    if args.mock:
        config.settle_time = 0.0

    env = AR5Env(config)
    env.connect()

    period = 1.0 / args.hz
    engaged = False
    latencies = []
    trace = []
    # tau[7] + external[7] + cart_force[3] + cart_torque[3] + 本地反推旋量[6]
    force_row = np.zeros(env.config.arm.num_joints * 2 + 12)
    force_warned = False
    tau = ext = cforce = ctrq = None
    # 力的基线：接合之前那一段空载。锁定之后不再更新 —— 一旦接合过，
    # 后面的"松开"可能是手还压在工件上，拿那个当零点就把接触力减掉了。
    force_base = np.zeros(4)          # [Fx, Fy, Fz, J7 力矩]
    force_base_acc = []
    force_base_locked = False
    live = open(args.live, "w", buffering=1 << 16) if args.live else None
    live_pending = 0
    last_unreachable = 0
    last_clamped = 0
    consecutive_refusals = 0
    box_lo = np.asarray(env.config.arm.workspace_min, dtype=np.float64)
    box_hi = np.asarray(env.config.arm.workspace_max, dtype=np.float64)
    # Cumulative distances since the clutch was last engaged. The per-step
    # delta is a few millimetres and rounds to nothing on screen, so totals
    # are what actually answer 'is the follower following'.
    leader_travel = 0.0
    commanded_travel = 0.0
    follower_travel = 0.0
    engage_origin = None
    prev_leader = None
    prev_follower = None

    print(f"\nleader   {args.leader_port} @ {args.leader_baud}")
    parts = []
    if not config.use_hand:
        parts.append("no hand")
    if not config.cameras.sources:
        parts.append("no cameras")
    suffix = f"  ({', '.join(parts)})" if parts else ""
    print(f"follower {'MOCK' if args.mock else config.arm.ip}{suffix}")
    print(f"rate     {args.hz} Hz   position scale {args.position_scale}x"
          f"   rotation {'off' if args.no_rotation else f'{args.rotation_scale}x 轴 {args.rotation_axes}'}")
    rt = getattr(env.arm, "_rt_active", False)
    print(f"control  {'real-time joint stream, 1 kHz' if rt else 'queued MoveL (no streaming)'}")
    print(f"limit    {args.max_step * 1000:.0f} mm per step")
    print("\n  ENTER  toggle clutch      q  quit")
    print("  Starting DISENGAGED -- move the leader and watch the action first.\n")

    try:
        env.reset(fake=True)

        if args.home:
            print("  homing in joint space ...", end=" ", flush=True)
            env.home_joints()
            print("done")

        if args.recentre:
            print("  recentring ...", end=" ", flush=True)
            print("done" if env.move_to_workspace_centre() else "timed out")
            env.reset(fake=True)

        # Posture health up front. A tool that sits nicely inside the workspace
        # box can still be on a singularity or have a wrist joint against its
        # stop, and both make inverse kinematics fail for every target in some
        # directions -- which looks exactly like the follower ignoring the
        # leader, with no error anywhere.
        report = env.posture_report()
        print(f"  posture  manipulability {report['manipulability']:.4f}, "
              f"worst joint j{report['worst_joint']} at {report['worst_usage'] * 100:.0f}% of travel")
        if report["near_singularity"]:
            print("  WARNING: close to a singularity. Inverse kinematics will stop converging")
            print("           and the arm will refuse most targets. Run with --home first.")
        elif report["worst_usage"] > 0.85:
            print(f"  WARNING: j{report['worst_joint']} has almost no travel left; run with --home.")

        # Input toward a face the arm is already on is silently absorbed by the
        # workspace clamp, which looks exactly like the follower ignoring the
        # leader.
        pinned = env.pinned_axes()
        # Being *outside* the box is a different failure from being pressed
        # against it, and much worse. The clamp pulls the target onto the face
        # on the very first step, the leash feeds that gap in at max_lead per
        # step, and the arm walks itself the whole way there with the operator
        # commanding nothing. Say that, rather than the "looks stuck" wording,
        # which is only right for a tool already inside.
        outside = [(a, f, c) for a, f, c in pinned if c < 0.0]
        if outside:
            print("\n  WARNING: the tool is OUTSIDE the workspace box:")
            for axis, face, clearance in outside:
                print(f"    {axis} past the {face} face by {-clearance * 1000:.0f} mm")
            worst = max(-c for _, _, c in outside)
            steps = int(np.ceil(worst / max(env.config.max_lead, 1e-9)))
            print(f"  Engaging the clutch will drive the arm back inside on its own --")
            print(f"  about {worst * 1000:.0f} mm over ~{steps} steps, with no input from you.")
            print("  Move the arm inside the box first, or widen workspace_min/workspace_max.\n")
        if [p for p in pinned if p[2] >= 0.0]:
            print("\n  WARNING: the tool is pressed against the workspace box:")
            for axis, face, clearance in pinned:
                if clearance < 0.0:
                    continue
                print(f"    {axis} {face}: {clearance * 1000:.0f} mm of clearance")
            print("  Motion in those directions will be clipped and the arm will look stuck.")
            print("  Re-run with --recentre, or lead it away from the face first.\n")

        with RawKeyboard() as kb:
            while True:
                loop_start = time.time()

                key = kb.key()
                if key in ("q", "Q"):
                    break
                if key in ("\n", "\r"):
                    if engaged:
                        mapper.disengage()
                        engaged = False
                    else:
                        engaged = True
                    print()  # leave the last status line in the scrollback

                joints = leader.read_positions()
                if len(joints) < len(SO_JOINTS):
                    missing = [n for n in SO_JOINTS if n not in joints]
                    print(f"\n  leader dropped {missing}; holding", end="")
                    time.sleep(period)
                    continue

                if engaged and not mapper.engaged:
                    # Re-reference on engage so the follower never jumps to
                    # wherever the leader drifted to while disengaged. Engaging
                    # is also the deliberate operator action that clears a
                    # streaming latch.
                    mapper.engage(joints)
                    env.recover()
                    engage_origin = env.arm.get_ee_pose()[:3].copy()
                    prev_leader = mapper.leader_pose(joints)[:3].copy()
                    prev_follower = engage_origin.copy()
                    leader_travel = commanded_travel = follower_travel = 0.0

                action = mapper.compute_action(joints)

                step_start = time.time()
                if engaged:
                    obs = env.step(action)
                else:
                    obs = None
                    # Keep publishing a zero-motion step while released. The
                    # sender brakes and latches when goals stop arriving, which
                    # is the right response to a dead loop but wrong for a
                    # deliberately idle one -- and it keeps the gripper live so
                    # it can be positioned before taking the arm.
                    env.step(np.concatenate([np.zeros(6), [action[6]]]))
                latencies.append(time.time() - step_start)

                # 力每步都采，**不管记不记轨迹** —— 状态行要显示它。
                # tau 0.2 ms + getEndTorque 0.3 ms（句柄已缓存），共约 0.5 ms。
                # 雅可比反推那一路要 0.7 ms，只在真的要写进轨迹时才算。
                arm = env.arm
                try:
                    tau = arm.get_joint_torque()
                    _, ext, ctrq, cforce = arm.get_end_wrench()
                    cforce, ctrq = _fit(cforce, 3), _fit(ctrq, 3)
                    # 基线：接合之前那一段空载。`payload_mass=0` 而手+工具约 1.3 kg，
                    # 那 18 N 重力在每一路里都没扣，不减基线屏幕上就只看得到重力。
                    if not engaged and not force_base_locked:
                        force_base_acc.append(np.concatenate([cforce, [tau[-1]]]))
                        if len(force_base_acc) >= FORCE_BASE_FRAMES:
                            force_base = np.mean(force_base_acc, axis=0)
                    elif engaged:
                        force_base_locked = True
                except Exception as exc:  # noqa: BLE001
                    tau = ext = None
                    if not force_warned:
                        print(f"\n  力读不到，状态行不显示：{exc}")
                        force_warned = True

                if args.log_actions:
                    # Joint-level fields too: an accepted goal that produces no
                    # motion is only diagnosable by comparing the published
                    # goal, what the tracker commanded, and what came back.
                    lock = getattr(arm, "_setpoint_lock", None)
                    if lock is not None and arm._q_goal is not None:
                        with lock:
                            q_goal, q_cmd = arm._q_goal.copy(), arm._q_cmd.copy()
                    else:
                        q_goal = q_cmd = np.zeros(env.config.arm.num_joints)
                    sent = arm.rt_stats()["sent"] if hasattr(arm, "rt_stats") else 0

                    # 这几路都是 **0.3 ms 级**的，可以按控制频率采。
                    #
                    # 2026-09-17 的教训：`get_end_wrench` 曾经每次都重取
                    # `robot.forceControl()` 句柄，而**取句柄本身就要 41 ms**
                    # （`getEndTorque()` 只要 0.30 ms）。那 41 ms 全程拿着 GIL，
                    # 把 1 kHz 发送流打出洞：963/s 掉到 781/s，控制器当丢包处理、
                    # **停止执行却继续收指令** —— q_cmd 一直走、q_actual 冻住、
                    # 控制器日志一条错都没有。句柄缓存之后这条路才成立。
                    #
                    # 两套末端力都记：控制器自己算的（扣了动力学模型）和本地
                    # `pinv(J^T)·tau` 反推的。它们来路不同，可以互相验证 ——
                    # 而同源的两列对得上是必然的，不算验证。
                    q_now = env.arm.get_joint_positions()
                    if tau is not None:
                        # 本地反推只在写轨迹时算：它要 0.7 ms，而且只用于和控制器
                        # 自己的估计交叉验证，屏幕上不需要。
                        kin = getattr(arm, "kinematics", None)
                        jac_wrench = (np.linalg.pinv(kin.jacobian(q_now).T) @ tau
                                      if kin is not None else np.zeros(6))
                        force_row = np.concatenate([
                            tau, _fit(ext, env.config.arm.num_joints),
                            cforce, ctrq, _fit(jac_wrench, 6),
                        ])

                    row = np.concatenate([
                        [time.time(), 1.0 if engaged else 0.0],
                        mapper.leader_pose(joints)[:3],
                        action,
                        # **位姿全六维，不是只有位置。** 「工具竖不竖直」要靠姿态
                        # 才看得出来，而那正是「绕世界 Z 自转 == 绕工具自转」的前提。
                        env.arm.get_ee_pose(),
                        q_goal, q_cmd, q_now,
                        [sent],
                        force_row,
                    ])
                    trace.append(row)

                    # 实时曲线走「这里写、另一个进程读」，**不在这个循环里画图** ——
                    # matplotlib 渲染会和当初的 getEndTorque 一样把 1 kHz 发送流打出洞。
                    # 写一行文本是微秒级的；不每行 flush，攒够再刷，代价只是画面晚
                    # 一点点，而 flush 是会阻塞的。
                    if live is not None:
                        # **时间戳要单独用定点格式。** `%.6g` 只有 6 位有效数字，
                        # 而 Unix 时间戳是 1.79e9 —— 会被压成 `1.78962e+09`，
                        # 秒以下全丢，整个文件的时间戳变成同一个值，曲线摊成一竖线。
                        # 2026-09-17 在预演里抓到的，真机上一样会犯。
                        live.write(f"{row[0]:.6f},")
                        live.write(",".join(f"{v:.6g}" for v in row[1:]))
                        live.write("\n")
                        live_pending += 1
                        if live_pending >= LIVE_FLUSH_ROWS:
                            live.flush()
                            live_pending = 0

                # Surface refusals as they happen. Reported only at the end,
                # a run of unsolvable targets is indistinguishable from a
                # follower that is simply not moving.
                if env.stats["unreachable"] > last_unreachable:
                    consecutive_refusals += env.stats["unreachable"] - last_unreachable
                    last_unreachable = env.stats["unreachable"]
                    if consecutive_refusals in (5, 25, 100):
                        r = env.posture_report()
                        print(f"\n  {consecutive_refusals} targets refused. "
                              f"manipulability {r['manipulability']:.4f}, "
                              f"worst joint j{r['worst_joint']} at {r['worst_usage'] * 100:.0f}%")
                        print("  The arm cannot solve for these targets from this posture. "
                              "Release the clutch and restart with --home.")
                else:
                    consecutive_refusals = 0

                if engaged:
                    here = mapper.leader_pose(joints)[:3]
                    if prev_leader is not None:
                        leader_travel += float(np.linalg.norm(here - prev_leader))
                    prev_leader = here.copy()
                    commanded_travel += float(np.linalg.norm(action[:3]))
                    # Path length, to compare like with like. Net displacement
                    # alone is misleading: a leader waved back and forth racks up
                    # metres of path while a perfectly obedient follower returns
                    # to where it began, reading as zero.
                    now_follower = (
                        obs.observation["ee_pose"][:3].astype(np.float64)
                        if obs is not None else env.arm.get_ee_pose()[:3]
                    )
                    if prev_follower is not None:
                        follower_travel += float(np.linalg.norm(now_follower - prev_follower))
                    prev_follower = now_follower.copy()

                # Hitting the workspace wall is silent otherwise, and a pinned
                # target looks exactly like a stuck arm: the operator keeps
                # moving, the follower stops, nothing is reported. Name the axis
                # that ran out of room so it is obvious which way to back off.
                # Computed from the target already in memory: an extra SDK read
                # here would cost the 1 kHz sender a GIL slot every step, which
                # is the jitter this loop was just cleaned up to avoid.
                wall = ""
                if env.stats["workspace_clamped"] > last_clamped:
                    last_clamped = env.stats["workspace_clamped"]
                    if env._target_pose is not None:
                        p = np.asarray(env._target_pose[:3], dtype=np.float64)
                        room = np.minimum(p - box_lo, box_hi - p)
                        i = int(np.argmin(room))
                        face = "max" if (box_hi[i] - p[i]) < (p[i] - box_lo[i]) else "min"
                        wall = f"  <WALL {'xyz'[i]}-{face}>"

                # 笛卡尔空间力/力矩，扣掉空载基线。打螺丝时屏幕上最该看的两样：
                # 压了多少牛、J7 离 17 Nm 的停机线还有多远。
                fx = ""
                if tau is not None:
                    dF = cforce - force_base[:3]
                    j7 = abs(float(tau[-1]))
                    pct = j7 / RSC_J7_LIMIT * 100
                    warn = "!!" if pct > 90 else (" !" if pct > 75 else "  ")
                    fx = (f"  F[{dF[0]:+5.1f} {dF[1]:+5.1f} {dF[2]:+5.1f}]N"
                          f" M[{ctrq[0]:+5.2f} {ctrq[1]:+5.2f} {ctrq[2]:+5.2f}]Nm"
                          f"  J7 {j7:4.1f}/{RSC_J7_LIMIT:.0f}{warn}")

                status = "ENGAGED " if engaged else "released"
                if engaged and engage_origin is not None:
                    ratio = follower_travel / commanded_travel if commanded_travel > 1e-9 else 0.0
                    print(f"  [{status}] leader {leader_travel * 1000:4.0f} "
                          f"cmd {commanded_travel * 1000:4.0f} "
                          f"fol {follower_travel * 1000:4.0f}mm "
                          f"({ratio * 100:3.0f}%){fx}{wall}   ", end="\r", flush=True)
                else:
                    # 松开时逐轴增量是验方向用的（接合之后这行就换成路径统计了），
                    # 力照样显示 —— 归零那几秒正好能看到噪声本底有多大。
                    print(f"  [{status}] d={np.round(action[:3] * 1000, 2).tolist()}mm "
                          f"grip={action[6]:.2f}{fx}   ", end="\r", flush=True)

                remaining = period - (time.time() - loop_start)
                if remaining > 0:
                    time.sleep(remaining)
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        print("\n\n=== session ===")
        if engage_origin is not None:
            net = float(np.linalg.norm(env.arm.get_ee_pose()[:3] - engage_origin))
            ratio = follower_travel / commanded_travel if commanded_travel > 1e-9 else 0.0
            print(f"  path length   leader {leader_travel * 1000:.0f} mm, "
                  f"commanded {commanded_travel * 1000:.0f} mm, "
                  f"follower {follower_travel * 1000:.0f} mm ({ratio * 100:.0f}% tracked)")
            print(f"  net displacement of the follower {net * 1000:.0f} mm "
                  f"(small is expected if the leader returned to where it started)")
        print(f"  env stats     {env.stats}")
        print(f"  mapper stats  {mapper.stats}")
        rt_stats = getattr(env.arm, "rt_stats", None)
        if rt_stats is not None and getattr(env.arm, "_rt_active", False):
            print(f"  rt stats      {rt_stats()}")
        if live is not None:
            live.flush()
            live.close()
        if args.log_actions and trace:
            # 列的定义见模块 docstring（61 列），自检里有一条守着它和实际拼出来的一致。
            np.savez(args.log_actions, trace=np.array(trace))
            print(f"  wrote {len(trace)} steps ({np.array(trace).shape[1]} cols) "
                  f"to {args.log_actions}")
        elif trace:
            # 跑完才发现没录，等于白跑一趟。说一声，并且把命令给出来。
            print("  ⚠ 这一轮**没有记轨迹** —— 要记得加 --log-actions <文件.npz>")
        if latencies:
            lat = np.array(latencies)
            print(f"  follower call mean {lat.mean() * 1000:.1f} ms, max {lat.max() * 1000:.1f} ms")
            if lat.mean() > period:
                print(f"  the follower call alone exceeds the {period * 1000:.0f} ms cycle; lower --hz")
        env.stop()
        env.close()
        leader.disconnect()
        print("  disconnected")
    return 0


if __name__ == "__main__":
    sys.exit(main())
