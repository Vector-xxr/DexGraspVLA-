"""Validate the real-time joint stream on hardware, in stages.

Each stage only runs once the previous one passed, and each is bounded so a
wrong result is a small wrong result. Nothing here exceeds a few degrees of
joint travel or a few centimetres at the tool.

    python -m atom_robot.ar5.diag_rt                 # stage 1 only: hold still
    python -m atom_robot.ar5.diag_rt --stage 2       # + single-joint ramp
    python -m atom_robot.ar5.diag_rt --stage 3       # + Cartesian sine through local IK

Stage 1 commands the arm's own pose, so a healthy setup does not move at all.
Keep a hand on the e-stop for stages 2 and 3.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from atom_robot.ar5.config import AR5Config  # noqa: E402
from atom_robot.ar5.drivers.rokae_arm import RokaeAR5  # noqa: E402
from atom_robot.ar5.transforms import matrix_to_axis_angle  # noqa: E402


def stage_hold(arm, seconds: float) -> bool:
    print(f"\n=== stage 1: hold still for {seconds:.0f} s ===")
    arm.clear_latch()
    print("  The goal is the arm's own posture, so it must not move.")
    q_start = arm.get_joint_positions()
    drift = []
    t0 = time.time()
    while time.time() - t0 < seconds:
        arm.servo_to_joints(q_start)
        drift.append(np.max(np.abs(arm.get_joint_positions() - q_start)))
        time.sleep(0.05)

    worst = np.rad2deg(max(drift))
    stats = arm.rt_stats()
    print(f"  drift        {worst:.4f} deg")
    print(f"  sent         {stats['sent']} commands, {stats['late_ticks']} late ticks")
    print(f"  rejected     {stats['rejected']}")
    if worst > 0.05:
        print("  *** The arm moved while told to hold. Stop here.")
        return False
    print("  holds position")
    return True


def stage_joint_ramp(arm, joint: int, degrees: float) -> bool:
    print(f"\n=== stage 2: ramp joint {joint} by {degrees:+.1f} deg ===")
    cfg = arm.config
    q0 = arm.get_joint_positions()
    target = q0.copy()
    target[joint] += np.deg2rad(degrees)

    lo, hi = cfg.joint_min[joint] + cfg.joint_limit_margin, cfg.joint_max[joint] - cfg.joint_limit_margin
    if not (lo <= target[joint] <= hi):
        print(f"  target {target[joint]:.4f} is outside [{lo:.4f}, {hi:.4f}]; pick another joint")
        return False

    if input("  type 'go' to move: ").strip() != "go":
        return False
    arm.clear_latch()

    # Publish in small increments so every published goal clears the jump gate.
    samples = []
    steps = 40
    t0 = time.time()
    for i in range(1, steps + 1):
        q = q0 + (target - q0) * (i / steps)
        if not arm.servo_to_joints(q):
            print(f"  goal {i} refused; see the rejection counters")
            return False
        samples.append((time.time() - t0, arm.get_joint_positions()[joint]))
        time.sleep(0.05)

    # Let the tracker finish closing the gap.
    settle = time.time()
    while time.time() - settle < 2.0:
        arm.servo_to_joints(target)
        samples.append((time.time() - t0, arm.get_joint_positions()[joint]))
        time.sleep(0.02)

    reached = arm.get_joint_positions()[joint]
    error = np.rad2deg(reached - target[joint])
    times = np.array([s[0] for s in samples])
    values = np.array([s[1] for s in samples])
    speed = np.abs(np.diff(values) / np.maximum(np.diff(times), 1e-6))

    print(f"  commanded    {np.rad2deg(target[joint] - q0[joint]):+.4f} deg")
    print(f"  measured     {np.rad2deg(reached - q0[joint]):+.4f} deg   (error {error:+.4f} deg)")
    print(f"  peak speed   {speed.max():.4f} rad/s   (limit {cfg.max_joint_speed})")
    others = np.rad2deg(np.abs(arm.get_joint_positions() - q0))
    others[joint] = 0
    print(f"  other joints moved at most {others.max():.4f} deg")

    ok = abs(error) < 0.15 and speed.max() < cfg.max_joint_speed * 1.3 and others.max() < 0.2
    print("  tracking is correct" if ok else "  *** MISMATCH")

    print("  returning")
    for i in range(steps, -1, -1):
        arm.servo_to_joints(q0 + (target - q0) * (i / steps))
        time.sleep(0.05)
    for _ in range(50):
        arm.servo_to_joints(q0)
        time.sleep(0.02)
    return ok


def stage_cartesian(arm, amplitude: float, seconds: float, hz: float) -> bool:
    print(f"\n=== stage 3: {amplitude * 1000:.0f} mm Cartesian sine through local IK ===")
    if input("  type 'go' to move: ").strip() != "go":
        return False
    arm.clear_latch()

    kin = arm.kinematics
    q0 = arm.get_joint_positions()
    T0 = kin.fk(q0)

    period = 1.0 / hz
    commanded, measured, ik_ms, refused = [], [], [], 0
    t0 = time.time()
    while True:
        elapsed = time.time() - t0
        if elapsed > seconds:
            break
        T = T0.copy()
        T[0, 3] += amplitude * np.sin(2 * np.pi * 0.25 * elapsed)

        tick = time.perf_counter()
        pose = np.concatenate([T[:3, 3], matrix_to_axis_angle(T[:3, :3])])
        accepted = arm.servo_to_pose_rt(pose)
        ik_ms.append((time.perf_counter() - tick) * 1000)
        if not accepted:
            refused += 1

        commanded.append(T[0, 3])
        measured.append(kin.fk(arm.get_joint_positions())[0, 3])
        time.sleep(max(0.0, period - (time.time() - t0 - elapsed)))

    commanded = np.array(commanded)
    measured = np.array(measured)
    error = np.abs(measured - commanded)
    span_cmd = (commanded.max() - commanded.min()) * 1000
    span_meas = (measured.max() - measured.min()) * 1000

    print(f"  updates      {len(commanded)} in {seconds:.0f} s ({len(commanded) / seconds:.1f} Hz)")
    print(f"  IK+publish   mean {np.mean(ik_ms):.2f} ms, max {np.max(ik_ms):.2f} ms")
    print(f"  commanded    {span_cmd:.1f} mm peak to peak")
    print(f"  measured     {span_meas:.1f} mm peak to peak")
    print(f"  error        mean {error.mean() * 1000:.2f} mm, max {error.max() * 1000:.2f} mm")
    print(f"  refused      {refused}")
    print(f"  rt stats     {arm.rt_stats()}")

    print("  returning to start")
    for _ in range(80):
        current = arm.get_joint_positions()
        arm.servo_to_joints(current + (q0 - current) * 0.2)
        time.sleep(0.05)

    ok = span_meas > span_cmd * 0.7 and error.mean() < 0.01
    print("  Cartesian tracking through local IK works" if ok else "  *** tracking is poor")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", type=int, default=1, choices=(1, 2, 3))
    parser.add_argument("--hold-seconds", type=float, default=5.0)
    parser.add_argument("--joint", type=int, default=0, help="Joint to ramp in stage 2")
    parser.add_argument("--degrees", type=float, default=3.0)
    parser.add_argument("--amplitude", type=float, default=0.02, help="Stage 3 sine amplitude, metres")
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--hz", type=float, default=20.0)
    args = parser.parse_args()

    config = AR5Config().arm
    config.use_realtime = True

    arm = RokaeAR5(config)
    print("connecting (this powers on the arm) ...")
    arm.connect(power_on=True)
    try:
        if not arm._rt_active:
            print("\nReal-time streaming did not start; see the warning above.")
            return 1
        print(f"real-time joint stream active, {config.max_joint_speed} rad/s cap")

        if not stage_hold(arm, args.hold_seconds):
            return 1
        if args.stage >= 2 and not stage_joint_ramp(arm, args.joint, args.degrees):
            return 1
        if args.stage >= 3 and not stage_cartesian(arm, args.amplitude, args.seconds, args.hz):
            return 1

        print(f"\nall stages up to {args.stage} passed")
    except KeyboardInterrupt:
        print("\ninterrupted")
    except BaseException:
        print("\nunexpected error -- stopping")
        arm.stop()
        raise
    finally:
        arm.stop()
        arm.disconnect()
        print("disconnected")
    return 0


if __name__ == "__main__":
    sys.exit(main())
