"""Read the arm's current posture and print it as pasteable config.

`home_joints` is not a number you derive -- you put the arm where you want it to
start each episode, then record where that is. Drag the arm there (or jog it
from the teach pendant), then run this.

    python -m atom_robot.ar5.capture_pose

Read-only: it connects, reads, and disconnects. It never powers on the arm and
never commands motion, so it is safe to run at any time.

    --watch          print continuously, so you can drag the arm and watch the numbers
    --save-centre    record where the arm is now as the joint-teleop centre posture

`--save-centre` is the other half of "drive it there with the leader, then keep
it": run teleop_joint, put the arm where you want each episode to start, quit
(the arm stays put), then run this. Recording after quitting rather than from
inside the loop is deliberate -- redefining the centre while engaged would move
the arm the instant the mapping changed under it.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from atom_robot.ar5.config import AR5Config  # noqa: E402
from atom_robot.ar5.drivers.rokae_arm import RokaeAR5  # noqa: E402
from atom_robot.ar5.transforms import axis_angle_to_matrix, internal_pose_to_rpy  # noqa: E402


def tool_axis(pose) -> tuple:
    """The frame axis pointing closest to straight down: ``(tilt_deg, name, vector)``.

    Which frame axis a tool sits on depends on how it is bolted to the flange --
    on this arm the screwdriver is on +Y, 90 degrees off the axis one would
    assume -- so rather than hard-code one, report whichever currently comes
    closest to vertical and let the operator confirm it is the right one.
    """
    R = axis_angle_to_matrix(np.asarray(pose[3:], dtype=np.float64))
    candidates = []
    for i, name in enumerate("XYZ"):
        for sign, label in ((1.0, "+"), (-1.0, "-")):
            v = sign * R[:, i]
            candidates.append((np.degrees(np.arccos(np.clip(-v[2], -1.0, 1.0))), f"{label}{name}", v))
    return min(candidates)


def render_compact(arm, config) -> None:
    """One refreshing line, for watching while both hands are on the arm.

    The full block is 25 lines; at two prints a second it scrolls past faster
    than it can be read, which is exactly when it is needed. This keeps only
    what a repositioning move is steered by: where the tool is, which box face
    it is past and by how much, and whether the drag has tipped it off vertical.
    """
    pose = arm.get_ee_pose()
    trans, _ = internal_pose_to_rpy(pose)
    lo = np.array(config.workspace_min)
    hi = np.array(config.workspace_max)

    over = []
    for i, axis in enumerate("xyz"):
        if trans[i] < lo[i]:
            over.append(f"{axis} 低于下限 {(lo[i] - trans[i]) * 1000:.0f}mm")
        elif trans[i] > hi[i]:
            over.append(f"{axis} 超出上限 {(trans[i] - hi[i]) * 1000:.0f}mm")
    box = "  ".join(over) if over else "盒内 OK"

    tilt, axis_name, _ = tool_axis(pose)
    print(f"  x {trans[0]:+.3f}  y {trans[1]:+.3f}  z {trans[2]:+.3f}   "
          f"[{box}]   刀具 {axis_name} 偏竖直 {tilt:4.1f}°        ",
          end="\r", flush=True)


def render(arm, config) -> None:
    q = arm.get_joint_positions()
    pose = arm.get_ee_pose()
    trans, rpy = internal_pose_to_rpy(pose)

    print("\n=== Joint angles (radians) ===")
    print("  home_joints = (" + ", ".join(f"{v:.6f}" for v in q) + ")")

    print("\n=== End-effector pose ===")
    print(f"  translation  [x y z]     m   : {np.round(trans, 5).tolist()}")
    print(f"  RPY          [rx ry rz]  rad : {np.round(rpy, 5).tolist()}")
    print(f"  axis-angle   (internal)      : {np.round(pose[3:], 5).tolist()}")
    print(f"  elbow / arm angle        rad : {arm.get_elbow():.5f}")

    # Which way the tool points, for tasks that need it vertical -- a screwdriver
    # has to stay on the screw axis. Dragging the arm by hand moves orientation
    # as well as position, so this is worth watching while repositioning.
    tilt, axis_name, axis_vec = tool_axis(pose)
    print("\n=== Tool direction ===")
    print(f"  frame axis closest to straight down : {axis_name}  "
          f"({axis_vec[0]:+.3f}, {axis_vec[1]:+.3f}, {axis_vec[2]:+.3f})")
    print(f"  its tilt off vertical               : {tilt:.1f} deg")
    print("  (confirm this is the axis the tool is actually mounted on)")

    lo = np.array(config.workspace_min)
    hi = np.array(config.workspace_max)
    inside = bool(np.all(trans >= lo) and np.all(trans <= hi))
    print("\n=== Against the configured workspace box ===")
    print(f"  min {lo.tolist()}")
    print(f"  max {hi.tolist()}")
    if inside:
        print("  current position is INSIDE the box")
    else:
        print("  current position is OUTSIDE the box")
        for i, axis in enumerate("xyz"):
            if trans[i] < lo[i] or trans[i] > hi[i]:
                print(f"    {axis} = {trans[i]:.4f} is outside [{lo[i]}, {hi[i]}]")
        print("  A home posture outside the box would be clipped on the first step.")
        print("  Either move the arm inside, or widen workspace_min/workspace_max.")


def save_centre(q: np.ndarray) -> None:
    """Record the current posture as the joint-teleop centre.

    Writes only `centre_joints`, so signs and scales already measured with
    `teleop_joint --save` survive. The scales are fitted from how much travel
    each joint has left around the centre, so they are re-derived on the next
    run anyway -- worth re-checking the printed scale table after moving it far.
    """
    import json

    from atom_robot.ar5.joint_mapping import CALIBRATION_FILE, JointMapConfig

    existing = {}
    if CALIBRATION_FILE.exists():
        try:
            existing = json.loads(CALIBRATION_FILE.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            print(f"\n  ignoring unreadable {CALIBRATION_FILE.name}: {exc}")

    previous = np.asarray(JointMapConfig.load().centre_joints, dtype=np.float64)
    existing["centre_joints"] = [float(v) for v in q]
    existing["centre_measured_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    CALIBRATION_FILE.write_text(json.dumps(existing, indent=2) + "\n")

    print("\n=== Saved as the joint-teleop centre ===")
    print(f"  was  {np.round(np.rad2deg(previous), 1).tolist()} deg")
    print(f"  now  {np.round(np.rad2deg(q), 1).tolist()} deg")
    print(f"  file {CALIBRATION_FILE}")
    if "sign_overrides" not in existing:
        print("\n  Signs and scales are not in this file yet. Confirm them with")
        print("  `teleop_joint ... --save`, which keeps this centre.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ip", default="", help="Override the controller IP")
    parser.add_argument("--watch", action="store_true", help="Print continuously until Ctrl-C")
    parser.add_argument("--interval", type=float, default=0.5, help="Seconds between prints when watching")
    parser.add_argument("--save-centre", action="store_true",
                        help="Record the current posture as the joint-teleop centre")
    args = parser.parse_args()

    config = AR5Config().arm
    if args.ip:
        config.ip = args.ip
    # Read-only: no real-time loop, and connect() is told not to power on.
    config.use_realtime = False

    arm = RokaeAR5(config)
    arm.connect(power_on=False)
    try:
        if args.watch:
            print("Drag the arm to the posture you want. Ctrl-C to stop,")
            print("then re-run without --watch to get the pasteable home_joints.\n")
            while True:
                render_compact(arm, config)
                time.sleep(args.interval)
        else:
            render(arm, config)
            if args.save_centre:
                save_centre(arm.get_joint_positions())
            else:
                print("\nPaste the home_joints line into ArmConfig in config.py,")
                print("or pass --save-centre to record this as the joint-teleop centre.")
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        arm.disconnect()
    return 0


if __name__ == "__main__":
    sys.exit(main())
