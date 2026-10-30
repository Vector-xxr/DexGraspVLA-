"""Work out how the leader's axes line up with the follower's.

`MappingConfig` starts with an identity axis map, which assumes the two arms'
base frames point the same way. They almost never do: the leader sits wherever
it fits on the bench. Getting this wrong makes the follower move confidently in
the wrong direction, which is exactly what "it moved but did not follow" looks
like.

This measures the correspondence instead of guessing it. For each of the
leader's three base axes you move the leader along it; the tool records which
follower axis that motion belongs on, and with which sign.

    python -m atom_robot.ar5.calibrate_mapping

The follower is never commanded. Only the leader is read, so nothing can crash
into anything while calibrating.

It also calibrates the gripper range, which needs the trigger squeezed through
its full travel.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from atom_robot.ar5.config import AR5Config  # noqa: E402
from atom_robot.ar5.leader_bus import SO_JOINTS, LeaderBus  # noqa: E402
from atom_robot.ar5.leader_mapping import (  # noqa: E402
    CALIBRATION_FILE,
    LeaderMapper,
    MappingConfig,
    mapping_matrix,
)

AXES = ("x", "y", "z")
# The follower axis each prompt asks you to move along, described physically so
# the instruction does not depend on knowing either base frame.
PROMPTS = (
    ("x", "AWAY from you (the direction you want the follower to reach forward)"),
    ("y", "to your LEFT (the direction you want the follower to move left)"),
    ("z", "UPWARD (the direction you want the follower to rise)"),
)

MIN_TRAVEL_M = 0.03


def capture(bus, mapper, seconds: float, label: str):
    """Record leader tool travel over a window, returning the net displacement."""
    print(f"\n  {label}")
    for n in (3, 2, 1):
        print(f"    starting in {n} ...", end="\r", flush=True)
        time.sleep(1.0)
    print("    MOVE NOW                    ")

    start = None
    samples = []
    deadline = time.time() + seconds
    while time.time() < deadline:
        joints = bus.read_positions()
        if len(joints) < len(SO_JOINTS):
            continue
        pose = mapper.leader_pose(joints)
        if start is None:
            start = pose
        samples.append(pose[:3] - start[:3])
        time.sleep(0.02)

    if not samples:
        raise RuntimeError("no leader samples captured")

    samples = np.array(samples)
    # The furthest point reached, not the final one: it survives the hand
    # drifting back toward where it started.
    travel = samples[int(np.argmax(np.linalg.norm(samples, axis=1)))]
    print(f"    leader moved {np.round(travel * 1000, 1).tolist()} mm "
          f"({np.linalg.norm(travel) * 1000:.0f} mm total)")
    return travel


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--leader-port", default="/dev/ttyACM0")
    parser.add_argument("--calib-id", default=AR5Config().collect.calib_id,
                        help="lerobot calibration id; blank uses the only one on disk")
    parser.add_argument("--seconds", type=float, default=4.0, help="Recording window per axis")
    parser.add_argument("--skip-gripper", action="store_true")
    args = parser.parse_args()

    bus = LeaderBus.with_calibration(args.leader_port, args.calib_id)
    bus.connect()
    mapper = LeaderMapper(MappingConfig())
    mapper.connect()

    print("=" * 66)
    print("Leader/follower axis calibration")
    print("=" * 66)
    print("\nThe follower is NOT connected and will NOT move.")
    print("For each prompt, move the leader's gripper a good distance -- at least")
    print(f"{MIN_TRAVEL_M * 1000:.0f} mm -- in the direction named, then hold still.")

    try:
        travels = {}
        for follower_axis, description in PROMPTS:
            input(f"\n[{follower_axis}] Press Enter, then move the leader {description}")
            travel = capture(bus, mapper, args.seconds, f"recording {follower_axis} ...")
            if np.linalg.norm(travel) < MIN_TRAVEL_M:
                print(f"    too little motion; move further and rerun")
                return 1
            travels[follower_axis] = travel

        print("\n" + "=" * 66)
        print("Result")
        print("=" * 66)

        # For each follower axis, the leader axis that carried the most motion.
        permutation, signs, ambiguous = [], [], []
        for follower_axis, _ in PROMPTS:
            travel = travels[follower_axis]
            dominant = int(np.argmax(np.abs(travel)))
            magnitude = np.abs(travel)
            share = magnitude[dominant] / magnitude.sum()
            permutation.append(dominant)
            signs.append(1.0 if travel[dominant] > 0 else -1.0)

            print(f"\n  follower {follower_axis}  <-  leader {AXES[dominant]}"
                  f"  sign {'+' if travel[dominant] > 0 else '-'}")
            print(f"    components {np.round(travel * 1000, 1).tolist()} mm, "
                  f"dominant axis carries {share * 100:.0f}%")
            if share < 0.7:
                ambiguous.append(follower_axis)
                print("    WARNING: the motion was not clearly along one axis. Either the")
                print("    two frames are rotated by something other than a right angle, or")
                print("    the movement wandered. Repeat this axis more deliberately.")

        if len(set(permutation)) != 3:
            print("\n  Two follower axes resolved to the same leader axis, so this is not a")
            print("  valid permutation. Redo the calibration, moving along one direction")
            print("  at a time and keeping the other two still.")
            return 1

        # axis_permutation is applied as delta[:3][perm], so entry i says which
        # leader axis feeds follower axis i -- exactly what was measured.
        print("\n" + "-" * 66)
        print(f"    axis_permutation = {tuple(permutation)}")
        print(f"    axis_signs = {tuple(signs)}")
        if ambiguous:
            print(f"\n  (axes {ambiguous} were ambiguous -- verify them before trusting this)")

        M = mapping_matrix(tuple(permutation), tuple(signs))
        if np.linalg.det(M) < 0:
            print("\n  This map mirrors the frame (its determinant is negative), which is")
            print("  normal when you sit facing the arm. Rotation deltas are negated to")
            print("  compensate, so the wrist still turns the way your hand does.")

        result = {
            "axis_permutation": [int(v) for v in permutation],
            "axis_signs": [float(v) for v in signs],
            "measured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "leader_port": args.leader_port,
            "calib_id": args.calib_id,
            "travel_mm": {k: np.round(v * 1000, 1).tolist() for k, v in travels.items()},
            "ambiguous_axes": ambiguous,
        }

        if not args.skip_gripper:
            print("\n" + "=" * 66)
            input("\nGripper: press Enter, then squeeze the trigger fully OPEN and CLOSED a few times")
            print("  recording for 5 s ...")
            values = []
            deadline = time.time() + 5.0
            while time.time() < deadline:
                joints = bus.read_positions()
                if "gripper" in joints:
                    values.append(joints["gripper"])
                time.sleep(0.02)
            if values:
                lo, hi = float(np.min(values)), float(np.max(values))
                print(f"  trigger range {lo:.1f} .. {hi:.1f} degrees")
                if hi - lo < 5.0:
                    print("  that is barely any travel; squeeze through the full range and rerun")
                else:
                    # Closed is the end the trigger reaches when squeezed. On the
                    # SO leader that is the more negative reading.
                    print(f"\n    gripper_closed_deg = {lo:.1f}")
                    print(f"    gripper_open_deg = {hi:.1f}")
                    result["gripper_closed_deg"] = round(lo, 1)
                    result["gripper_open_deg"] = round(hi, 1)

        CALIBRATION_FILE.write_text(json.dumps(result, indent=2) + "\n")
        print("\n" + "=" * 66)
        print(f"  saved to {CALIBRATION_FILE}")
        print("  teleop picks this up automatically -- nothing to paste.")
        if "gripper_closed_deg" not in result:
            print("  (gripper range not measured; it keeps its default)")
    except KeyboardInterrupt:
        print("\ninterrupted -- nothing was saved")
    finally:
        bus.disconnect()
    return 0


if __name__ == "__main__":
    sys.exit(main())
