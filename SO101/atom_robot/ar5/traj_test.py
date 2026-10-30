"""Drive a continuous path and measure how well the arm tracks it.

Isolated probes answer "does a command arrive". They do not answer the question
that matters for VEL, which streams an 8-step action chunk every control cycle:
can this arm follow a *sequence* of small Cartesian steps at the control rate,
and how far behind does it run?

The path is generated, validated in full before anything moves, then executed
while commanded and measured poses are logged every step.

    python -m atom_robot.ar5.traj_test --shape square --size 0.04
    python -m atom_robot.ar5.traj_test --shape circle --size 0.03 --hz 10

Reported afterwards:
    tracking error   how far the measured pose trails the commanded one
    achieved rate    whether the loop kept up with --hz
    step latency     time spent inside servo_to_pose, which is the real limit

Every waypoint is checked with the controller's own reachability test before
the first move, so an unreachable path aborts before the arm is powered.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from atom_robot.ar5.config import AR5Config  # noqa: E402
from atom_robot.ar5.drivers.rokae_arm import RokaeAR5  # noqa: E402


def build_path(start: np.ndarray, shape: str, size: float, points: int) -> np.ndarray:
    """Waypoints in the XY plane at the starting height, returning to start."""
    poses = []
    if shape == "square":
        corners = np.array([[0, 0], [size, 0], [size, size], [0, size], [0, 0]], dtype=float)
        per_side = max(1, points // 4)
        for a, b in zip(corners[:-1], corners[1:]):
            for t in np.linspace(0, 1, per_side, endpoint=False):
                poses.append(a + (b - a) * t)
        poses.append(corners[-1])
    elif shape == "circle":
        r = size / 2
        for t in np.linspace(0, 2 * np.pi, points, endpoint=True):
            # Start and end at the same point, sweeping a full circle.
            poses.append([r * np.sin(t), r * (1 - np.cos(t))])
    elif shape == "line":
        for t in np.concatenate([np.linspace(0, 1, points // 2), np.linspace(1, 0, points // 2)]):
            poses.append([size * t, 0.0])
    else:
        raise ValueError(shape)

    out = np.tile(start, (len(poses), 1))
    out[:, 0] += np.array([p[0] for p in poses])
    out[:, 1] += np.array([p[1] for p in poses])
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shape", default="square", choices=("square", "circle", "line"))
    parser.add_argument("--size", type=float, default=0.04, help="Side length or diameter, metres")
    parser.add_argument("--points", type=int, default=40, help="Waypoints in the path")
    parser.add_argument("--hz", type=float, default=10.0, help="Target command rate")
    parser.add_argument("--speed", type=float, default=100.0, help="MoveL speed argument")
    parser.add_argument("--dry-run", action="store_true", help="Validate the path without moving")
    args = parser.parse_args()

    config = AR5Config().arm
    config.use_realtime = False
    config.move_speed = args.speed
    config.move_zone = 0.0

    arm = RokaeAR5(config)
    arm.connect(power_on=False)
    try:
        start = arm.get_ee_pose()
        path = build_path(start, args.shape, args.size, args.points)

        span = path[:, :3].max(axis=0) - path[:, :3].min(axis=0)
        steps = np.linalg.norm(np.diff(path[:, :3], axis=0), axis=1)
        print(f"path      : {args.shape}, {len(path)} waypoints")
        print(f"extent    : {np.round(span * 1000, 1).tolist()} mm")
        print(f"step size : {steps.min() * 1000:.2f} .. {steps.max() * 1000:.2f} mm")
        print(f"rate      : {args.hz} Hz  ->  {steps.mean() * args.hz * 1000:.0f} mm/s average")

        lo = np.array(config.workspace_min)
        hi = np.array(config.workspace_max)
        outside = [i for i, p in enumerate(path) if np.any(p[:3] < lo) or np.any(p[:3] > hi)]
        if outside:
            print(f"\n{len(outside)} waypoints fall outside the workspace box; shrink --size")
            return 1
        if steps.max() > config.max_step_translation:
            print(f"\nlargest step {steps.max() * 1000:.1f} mm exceeds the "
                  f"{config.max_step_translation * 1000:.0f} mm per-step limit; "
                  "raise --points or shrink --size")
            return 1

        print("\nvalidating every waypoint with the controller ...")
        for i, p in enumerate(path):
            ok, detail = arm.check_reachable(p)
            if not ok:
                print(f"  waypoint {i} {np.round(p[:3], 4).tolist()} refused: {detail}")
                return 1
        print(f"  all {len(path)} waypoints reachable")

        if args.dry_run:
            print("\nDry run: path validated, nothing moved.")
            return 0

        if input("\ntype 'go' to power on and run the path: ").strip() != "go":
            print("aborted")
            return 1

        arm._power_on()
        print(f"power: {arm._call(arm._robot.powerState)}")

        period = 1.0 / args.hz
        commanded, measured, latencies, timestamps = [], [], [], []
        t0 = time.time()
        for target in path:
            step_start = time.time()
            arm.servo_to_pose(target)
            latencies.append(time.time() - step_start)
            commanded.append(target[:3].copy())
            measured.append(arm.get_ee_pose()[:3])
            timestamps.append(time.time() - t0)

            remaining = period - (time.time() - step_start)
            if remaining > 0:
                time.sleep(remaining)

        elapsed = time.time() - t0
        commanded = np.array(commanded)
        measured = np.array(measured)
        latencies = np.array(latencies)
        errors = np.linalg.norm(measured - commanded, axis=1)

        print("\n=== tracking ===")
        print(f"  mean error   {errors.mean() * 1000:7.2f} mm")
        print(f"  max  error   {errors.max() * 1000:7.2f} mm   at waypoint {int(errors.argmax())}")
        print(f"  final error  {errors[-1] * 1000:7.2f} mm")

        print("\n=== timing ===")
        print(f"  requested    {args.hz:.1f} Hz")
        print(f"  achieved     {len(path) / elapsed:.1f} Hz over {elapsed:.1f} s")
        print(f"  servo call   mean {latencies.mean() * 1000:.1f} ms, max {latencies.max() * 1000:.1f} ms")
        max_hz = 1.0 / latencies.mean() if latencies.mean() > 0 else float("inf")
        print(f"  this caps the queued path at about {max_hz:.0f} Hz")

        print()
        if latencies.mean() > period:
            print("  The servo call alone exceeds the cycle time: queued MoveL cannot")
            print("  sustain this rate. Switch to the real-time loop (use_realtime=True)")
            print("  before driving the arm from the policy.")
        elif errors.mean() > 0.005:
            print("  Tracking lags by more than 5 mm on average. Lower --hz or --speed,")
            print("  or move to the real-time loop for smoother following.")
        else:
            print("  Tracking and rate are both healthy at this setting.")

        print("\nreturning to start")
        arm.servo_to_pose(start)
        time.sleep(1.0)
    except KeyboardInterrupt:
        print("\ninterrupted -- stopping")
        arm.stop()
    except BaseException:
        print("\nunexpected error -- stopping")
        arm.stop()
        raise
    finally:
        arm.disconnect()
        print("disconnected")
    return 0


if __name__ == "__main__":
    sys.exit(main())
