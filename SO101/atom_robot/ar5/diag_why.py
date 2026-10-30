"""Ask the controller why it refuses to move.

Three trials eliminated motion control mode, blending zone, simulation mode and
speed override: moveAppend and moveStart both return success, OperationState
never leaves `idle`, and displacement is exactly zero.

Two SDK calls answer this directly rather than by elimination:

  checkPath()           validates a Cartesian target and puts the reason for
                        unreachability in the error dict. Pure computation --
                        it never moves the arm.

  queryControllerLog()  returns the controller's own log. Sampling it before
                        and after a move attempt shows what the controller
                        recorded when it dropped the command.

    python -m atom_robot.ar5.diag_why            # checks only, no motion
    python -m atom_robot.ar5.diag_why --attempt  # also try one move

Without --attempt nothing moves and the arm is never powered on.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from atom_robot.ar5.config import AR5Config  # noqa: E402
from atom_robot.ar5.drivers.rokae_arm import RokaeAR5  # noqa: E402


def dump_log(arm, label: str, count: int = 10):
    print(f"\n=== controller log: {label} ===")
    try:
        entries = arm._call(arm._robot.queryControllerLog, count, set())
    except Exception as exc:  # noqa: BLE001
        print(f"  unavailable: {exc}")
        return []
    if not entries:
        print("  (empty)")
        return []
    rendered = []
    for e in entries:
        fields = {a: getattr(e, a) for a in dir(e) if not a.startswith("_") and not callable(getattr(e, a))}
        rendered.append(fields)
        print(f"  {fields}")
    return rendered


def describe(cp, label):
    print(f"  {label}")
    print(f"    trans     {np.round(cp.trans, 5).tolist()}")
    print(f"    rpy       {np.round(cp.rpy, 5).tolist()}")
    print(f"    elbow     {cp.elbow:.5f}   hasElbow={cp.hasElbow}")
    print(f"    confData  {list(cp.confData)}")
    print(f"    external  {list(cp.external)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--axis", default="x", choices="xyz")
    parser.add_argument("--step", type=float, default=0.010, help="metres")
    parser.add_argument("--attempt", action="store_true", help="Also power on and try one move")
    args = parser.parse_args()

    config = AR5Config().arm
    config.use_realtime = False

    arm = RokaeAR5(config)
    arm.connect(power_on=False)
    sdk = arm.sdk
    robot = arm._robot
    try:
        q_full = arm._call(robot.jointPos)
        start_cp = arm._call(robot.cartPosture, sdk.CoordinateType.flangeInBase)

        print("=== current state ===")
        print(f"  joints (all {len(q_full)}) {np.round(q_full, 5).tolist()}")
        describe(start_cp, "cartPosture(flangeInBase)")

        # Soft limits: a target outside them is refused.
        try:
            limits = sdk.PyTypeVectorArrayDouble2()
            enabled = arm._call(robot.getSoftLimit, limits)
            print(f"\n  soft limits enabled: {enabled}")
            for i, lim in enumerate(limits.content()):
                mark = ""
                if i < len(q_full) and not (lim[0] <= q_full[i] <= lim[1]):
                    mark = "   <-- CURRENT JOINT IS OUTSIDE"
                print(f"    joint {i}: [{lim[0]:.4f}, {lim[1]:.4f}]{mark}")
        except Exception as exc:  # noqa: BLE001
            print(f"  soft limits unavailable: {exc}")

        # Build the target exactly as the driver does: a fresh CartesianPosition
        # carrying only trans and rpy.
        idx = "xyz".index(args.axis)
        target_pose = arm.get_ee_pose()
        target_pose[idx] += args.step
        target_bare = arm._to_cartesian_position(target_pose)

        # And a second variant that inherits elbow and confData from the current
        # pose. A 7-axis arm is redundant, so the arm angle is what picks one
        # solution out of the family sharing an end-effector pose.
        target_full = arm._to_cartesian_position(target_pose)
        target_full.elbow = start_cp.elbow
        target_full.hasElbow = True
        target_full.confData = list(start_cp.confData)

        print(f"\n=== checkPath: is {args.step * 1000:.0f} mm on {args.axis} reachable? ===")
        start_joint = list(q_full)
        for label, target in (("bare (trans + rpy only)", target_bare),
                              ("with elbow + confData", target_full)):
            ec = {}
            try:
                solution = robot.checkPath(start_cp, start_joint, target, ec)
                code = ec.get("ec", 0)
                if code:
                    print(f"  {label:<26} REFUSED  ec={code}  {ec.get('message')}")
                else:
                    print(f"  {label:<26} OK       target joints "
                          f"{np.round(list(solution)[:7], 5).tolist()}")
            except Exception as exc:  # noqa: BLE001
                print(f"  {label:<26} raised {type(exc).__name__}: {exc}")

        dump_log(arm, "before any attempt")

        if not args.attempt:
            print("\nChecks only. Re-run with --attempt to also try a move.")
            return 0

        if input("\ntype 'go' to power on and attempt one move: ").strip() != "go":
            print("aborted")
            return 1

        arm._power_on()
        arm._call(robot.setMotionControlMode, sdk.MotionControlMode.NrtCommandMode)
        print(f"power={arm._call(robot.powerState)}  mode set")

        cmd = sdk.MoveLCommand(target_full, 50.0, 0.0)
        arm._call(robot.moveAppend, [cmd], sdk.PyString("why"))
        arm._call(robot.moveStart)
        time.sleep(2.0)

        after = arm.get_ee_pose()
        print(f"\n  displacement: {np.linalg.norm(after[:3] - target_pose[:3] + args.step * np.eye(3)[idx]) * 1000:.2f} mm")
        print(f"  state: {arm._call(robot.operationState)}")
        print(f"  moveExecution: {arm._call(robot.queryEventInfo, sdk.Event.moveExecution)}")
        dump_log(arm, "after the attempt")
    except KeyboardInterrupt:
        print("\ninterrupted -- stopping")
        arm.stop()
    except BaseException:
        print("\nunexpected error -- stopping")
        arm.stop()
        raise
    finally:
        arm.disconnect()
        print("\ndisconnected")
    return 0


if __name__ == "__main__":
    sys.exit(main())
