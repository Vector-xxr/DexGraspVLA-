"""Joint-for-joint teleoperation: each SO-101 joint drives one AR5 joint.

The Cartesian script (``teleop.py``) sends tool-space deltas through inverse
kinematics. This one skips all of that: leader joint angles become AR5 joint
angles directly, and the two AR5 joints with no leader counterpart are held
still. No IK, no workspace box, no reachability gate -- a joint goal is either
inside the soft limits or it is not.

    # verify the mapping first: the arm does NOT move until you engage
    python -m atom_robot.ar5.teleop_joint --calib-id single_leader_arm --no-hand --no-cameras

    # a joint moving the wrong way
    python -m atom_robot.ar5.teleop_joint --flip wrist_roll --flip shoulder_pan

    # keep the corrections
    python -m atom_robot.ar5.teleop_joint --flip wrist_roll --save

Released, the status line shows each leader joint's angle next to the AR5
joint's angle measured from its centre. Those two numbers should move together
and with the same sign. Any joint where they move oppositely needs --flip.
Nothing moves while released, so this is safe to do first, every time.

Engaging ramps the follower from where it is to the mapped posture over a couple
of seconds rather than demanding it at once.
"""

import argparse
import json
import logging
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
from atom_robot.ar5.joint_mapping import (  # noqa: E402
    CALIBRATION_FILE,
    DATASHEET_AXIS,
    JOINT_PAIRS,
    JointMapConfig,
    JointMapper,
)
from atom_robot.ar5.leader_bus import SO_JOINTS, LeaderBus  # noqa: E402

SHORT = {"shoulder_pan": "pan", "shoulder_lift": "lift", "elbow_flex": "elb",
         "wrist_roll": "roll", "wrist_flex": "flex"}


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


def hold_posture(arm, num_joints: int) -> np.ndarray:
    """The posture to keep commanding while released.

    The tracker's own setpoint, not the measurement: re-reading the encoders
    every tick and commanding that back would integrate measurement noise into
    the goal, and the goal is what the 1 kHz sender is chasing.
    """
    lock = getattr(arm, "_setpoint_lock", None)
    if lock is not None and getattr(arm, "_q_cmd", None) is not None:
        with lock:
            return arm._q_cmd.copy()
    return arm.get_joint_positions()


def main() -> int:
    from atom_robot.ar5.end_effector import add_end_effector_args, configure_end_effector
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_end_effector_args(parser)
    # 和 collect 同一个默认值（by-id）。写死 ttyACM0 会在 OmniPicker 的 USB-485 先插时开错设备。
    parser.add_argument("--leader-port", default=AR5Config().collect.leader_port)
    parser.add_argument("--calib-id", default=AR5Config().collect.calib_id,
                        help="lerobot calibration id; blank uses the only one on disk")
    parser.add_argument("--leader-baud", type=int, default=1_000_000)
    parser.add_argument("--hz", type=float, default=50.0,
                        help="Goal rate. The 1 kHz tracker accelerates and brakes between "
                             "goals, so a low rate reads as a stutter at the tool; 20 Hz was "
                             "visibly rough. Measured with the hand attached, the loop holds "
                             "100 Hz with no overruns.")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Multiplies every fitted scale. Below 1 makes a first run gentle.")
    parser.add_argument("--flip", action="append", default=[], metavar="JOINT",
                        help="Invert this leader joint's sign. Repeatable.")
    parser.add_argument("--ramp", type=float, default=2.5,
                        help="Seconds to blend from the current posture into the mapped one")
    parser.add_argument("--min-cutoff", type=float, default=None, metavar="HZ",
                        help="Leader filter cutoff while the arm is nearly still. Lower is "
                             "smoother and laggier; 0 turns the filter off entirely.")
    parser.add_argument("--beta", type=float, default=None,
                        help="How fast the filter opens up as the leader moves, Hz per deg/s. "
                             "Raise it if the arm feels laggy while you are moving.")
    parser.add_argument("--relative", action="store_true",
                        help="Drive the arm by how far the leader has moved since engaging, "
                             "instead of by where it is. Runs 1:1 -- absolute mode has to shrink "
                             "every scale to fit the leader's swing around the centre posture. "
                             "Release and re-engage to recentre the leader and carry on.")
    parser.add_argument("--centre", default="",
                        help="Override the centre posture: 7 comma-separated degrees")
    parser.add_argument("--save", action="store_true",
                        help="Write the sign and scale corrections to joint_calibration.json")
    parser.add_argument("--no-saved", action="store_true",
                        help="Ignore joint_calibration.json")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--print-hz", type=float, default=5.0,
                        help="Status line refreshes per second. Printing every cycle with "
                             "flush=True blocks the loop on a slow terminal (a remote desktop, "
                             "say), and a loop that stalls past goal_stale_timeout makes the "
                             "sender brake and latch mid-motion. 0 disables the status line.")
    parser.add_argument("--no-hand", action="store_true")
    parser.add_argument("--real-hand", action="store_true",
                        help="With --mock: simulated arm, real hand.")
    parser.add_argument("--no-cameras", action="store_true")
    parser.add_argument("--home", action="store_true",
                        help="Move to the centre posture in joint space before starting")
    args = parser.parse_args()

    overrides = {"ramp_seconds": args.ramp, "scale_multiplier": args.scale,
                 "relative": args.relative}
    if args.min_cutoff is not None:
        overrides["filter_min_cutoff"] = args.min_cutoff
    if args.beta is not None:
        overrides["filter_beta"] = args.beta
    if args.flip:
        known = {name for name, _, _ in JOINT_PAIRS}
        bad = [j for j in args.flip if j not in known]
        if bad:
            parser.error(f"--flip {bad} is not a mapped joint. Choose from {sorted(known)}")
        overrides["sign_overrides"] = {
            name: -sign for name, _, sign in JOINT_PAIRS if name in args.flip
        }
    if args.centre:
        values = [float(v) for v in args.centre.split(",")]
        if len(values) != 7:
            parser.error(f"--centre needs 7 comma-separated degrees, got {len(values)}")
        overrides["centre_joints"] = tuple(np.deg2rad(values))

    mapping = JointMapConfig.load(
        path=Path("/nonexistent") if args.no_saved else None, **overrides
    )
    mapper = JointMapper(mapping)

    # Surface the driver's own diagnostics; a real-time stream that failed to
    # start says so exactly once, here, and nowhere else.
    logging.basicConfig(level=logging.INFO, format="  %(message)s")

    leader = LeaderBus.with_calibration(
        args.leader_port, args.calib_id, baudrate=args.leader_baud
    )
    leader.connect()
    leader.assert_torque_disabled()

    config = AR5Config(mock=args.mock)
    config.real_hand_in_mock = args.real_hand
    config.use_hand = not args.no_hand and (not args.mock or args.real_hand)
    configure_end_effector(config, args)
    config.arm.use_realtime = True
    if args.no_cameras:
        config.cameras.sources = {}
    if args.mock:
        config.settle_time = 0.0

    mapper.configure(
        {name: leader.half_travel_deg(name) for name, _, _ in JOINT_PAIRS},
        config.arm.joint_min,
        config.arm.joint_max,
        margin=config.arm.joint_limit_margin,
    )

    env = AR5Env(config)
    try:
        env.connect()
    except BaseException:
        leader.disconnect()
        raise

    # The driver downgrades a failed real-time startup to a warning and keeps
    # going, so without this the first engaged step raises "requires the
    # real-time stream" -- the symptom, long after the cause scrolled past.
    if not args.mock and not getattr(env.arm, "_rt_active", False):
        print("\n  the real-time stream is not running; the arm cannot be driven.")
        print("  The reason was logged as a warning above. Usually the teach pendant")
        print("  is in manual mode or holds the enable; switch it to automatic.")
        env.close()
        leader.disconnect()
        return 1

    period = 1.0 / args.hz
    engaged = False
    centre = mapper.centre

    print(f"\nleader   {args.leader_port} @ {args.leader_baud}")
    print(f"follower {'MOCK' if args.mock else config.arm.ip}"
          f"   joint-for-joint, no IK and no workspace box")
    print(f"rate     {args.hz} Hz   scale x{args.scale}   ramp {args.ramp}s")
    print(f"floor    tool and wrist kept above z = {config.arm.joint_floor_z * 1000:.0f} mm")
    if not args.no_saved and CALIBRATION_FILE.exists():
        print(f"loaded   {CALIBRATION_FILE.name}")
    print()
    print(mapper.report())
    print()
    print("  ENTER  toggle clutch      q  quit")
    print("  Starting RELEASED. Move ONE leader joint at a time and check that the")
    print("  two numbers for it move together. Opposite signs mean --flip that joint.\n")

    try:
        if args.home:
            print("  moving to the centre posture ...", end=" ", flush=True)
            ok = True
            q_start = hold_posture(env.arm, config.arm.num_joints)
            steps = max(1, int(np.max(np.abs(centre - q_start)) / 0.01))
            for i in range(1, steps + 1):
                env.step_joints(q_start + (centre - q_start) * (i / steps), observe=False)
                time.sleep(0.02)
            for _ in range(60):
                env.step_joints(centre, observe=False)
                time.sleep(0.02)
            reached = env.arm.get_joint_positions()
            error = float(np.max(np.abs(reached - centre)))
            print(f"done (worst joint off by {np.rad2deg(error):.3f} deg)" if ok else "failed")
            if env.arm.kinematics is not None:
                tool = env.arm.kinematics.fk(reached)[:3, 3]
                print(f"  tool at {np.round(tool, 3).tolist()} m")

        env.arm.clear_latch()
        blocked_seen = 0
        print_period = -1.0 if args.print_hz <= 0 else 1.0 / args.print_hz
        last_print = 0.0

        # Loop period, measured. The tracker's feed-forward is the goal's own
        # speed over the elapsed time, so a period that wanders is a commanded
        # velocity that wanders -- which is what shake looks like. Summarised at
        # exit rather than printed live, since printing is itself a cost here.
        periods = []
        previous_tick = None

        with RawKeyboard() as kb:
            while True:
                loop_start = time.time()
                tick = time.perf_counter()
                if previous_tick is not None:
                    periods.append(tick - previous_tick)
                previous_tick = tick

                key = kb.key()
                if key in ("q", "Q"):
                    break
                if key in ("\n", "\r"):
                    if engaged:
                        mapper.disengage()
                        engaged = False
                    else:
                        mapper.engage(hold_posture(env.arm, config.arm.num_joints))
                        engaged = True
                    print()

                joints = leader.read_positions()
                if len(joints) < len(SO_JOINTS):
                    missing = [n for n in SO_JOINTS if n not in joints]
                    print(f"\n  leader dropped {missing}; holding", end="")
                    env.step_joints(hold_posture(env.arm, config.arm.num_joints),
                                    observe=False)
                    time.sleep(period)
                    continue

                q_now = hold_posture(env.arm, config.arm.num_joints)
                command = mapper.compute(joints, q_now)
                gripper = mapper.gripper_command(joints)

                # observe=False: nothing here reads the observation, and building
                # one costs a camera frame plus three blocking SDK round trips.
                # Paying that on every step is what made the loop period jump
                # around, and the tracker turns a jumpy loop period into a jumpy
                # commanded velocity.
                env.step_joints(command, gripper, observe=False)

                note = ""
                if env.stats["floor_blocked"] > blocked_seen:
                    blocked_seen = env.stats["floor_blocked"]
                    note = "  <FLOOR>"
                elif mapper.stats["clamped"]:
                    note = f"  <clamp {mapper.stats['clamped']}>"

                # Rate-limited: see --print-hz. A blocked write here stalls the
                # control loop, and the sender reads a stalled loop as a dead
                # one. Never skip a line carrying <FLOOR> or a clamp count --
                # those are the only warning the operator gets.
                now_print = time.perf_counter()
                if note or (print_period >= 0 and now_print - last_print >= print_period):
                    last_print = now_print
                    # Reading the encoders is a state refresh plus an SDK call,
                    # and the status line is its only consumer -- so it happens
                    # at the print rate, not the control rate.
                    measured = env.arm.get_joint_positions()
                    cells = []
                    for name, index, _ in JOINT_PAIRS:
                        lead = joints.get(name, float("nan"))
                        follow = np.rad2deg(measured[index] - centre[index])
                        cells.append(f"{SHORT[name]} {lead:+6.1f}|{follow:+6.1f}")

                    if engaged:
                        alpha = mapper.ramp_fraction()
                        tag = f"ENGAGED {alpha * 100:3.0f}%" if alpha < 1.0 else "ENGAGED     "
                    else:
                        tag = "released    "

                    print(f"  [{tag}] " + "  ".join(cells) +
                          f"  grip target={gripper:.2f} actual={env.hand.get_opening():.2f}{note}   ",
                          end="\r", flush=True)

                remaining = period - (time.time() - loop_start)
                if remaining > 0:
                    time.sleep(remaining)

    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        print("\n")
        print(f"  steps {mapper.stats['steps']}   clutch cycles {mapper.stats['clutch_cycles']}")
        print(f"  goals clamped to soft limits: {mapper.stats['clamped']}")
        print(f"  refused by the floor guard:   {env.stats['floor_blocked']}")
        print(f"  dropped leader replies:       {mapper.stats['missing']}")

        if periods:
            ms = np.sort(np.asarray(periods)) * 1000.0
            p50, p95, worst = ms[len(ms) // 2], ms[int(len(ms) * 0.95)], ms[-1]
            print(f"\n  控制周期  目标 {period * 1000:.1f} ms   实测 p50 {p50:.1f}  "
                  f"p95 {p95:.1f}  最差 {worst:.1f} ms")
            # The p95/p50 spread is the number that matters: the tracker divides
            # by this interval to work out how fast the goal is moving.
            if p95 > 1.6 * p50:
                print(f"  ⚠ 周期抖动 {p95 / p50:.1f} 倍。前馈速度是「目标位移 / 这段时间」，"
                      "时间不稳,速度就不稳,臂就抖。")
        if hasattr(env.arm, "rt_stats"):
            stats = env.arm.rt_stats()
            print(f"  rt {stats}")
            sent, late = stats.get("sent", 0), stats.get("late_ticks", 0)
            if sent and late / sent > 0.02:
                print(f"  ⚠ 1 kHz 发送线程有 {100.0 * late / sent:.1f}% 的周期没赶上。"
                      "赶不上就是控制器那边断供一拍再补一拍,直接表现为抖。")

        if args.save:
            result = {
                "sign_overrides": {name: mapper._signs[name] for name, _, _ in JOINT_PAIRS},
                "scale_overrides": {name: mapper._scales[name] for name, _, _ in JOINT_PAIRS},
                "centre_joints": [float(v) for v in mapper.centre],
                "gripper_closed_deg": mapping.gripper_closed_deg,
                "gripper_open_deg": mapping.gripper_open_deg,
                "measured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            CALIBRATION_FILE.write_text(json.dumps(result, indent=2) + "\n")
            print(f"\n  saved to {CALIBRATION_FILE}")

        env.close()
        leader.disconnect()
    return 0


if __name__ == "__main__":
    sys.exit(main())
