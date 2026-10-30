"""Find out why a queued move does not execute.

The bring-up test commanded 5 mm and measured exactly zero displacement, with
no SDK error. That means the controller accepted the command and declined to
run it. Two candidates, which this separates:

  A. Motion control mode was never set. The driver only called
     setMotionControlMode on the real-time path, so a queued (Nrt) move may
     arrive while the controller sits in Idle.

  B. The blending zone swallowed the motion. move_zone defaults to 10 mm while
     the step was 5 mm, so the entire path lies inside the blend radius.

Each trial reports the OperationState transitions, which is the ground truth:
if the arm never enters `moving`, the command was rejected, not mistracked.

    python -m atom_robot.ar5.diag_motion            # 10 mm in +X
    python -m atom_robot.ar5.diag_motion --axis z --step 0.01

Read-only until you confirm. Keep a hand on the e-stop.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from atom_robot.ar5.config import AR5Config  # noqa: E402
from atom_robot.ar5.drivers.rokae_arm import RokaeAR5  # noqa: E402


def poll_state(arm, seconds: float = 3.0, interval: float = 0.05):
    """Record every OperationState change during a window."""
    transitions = []
    last = None
    deadline = time.time() + seconds
    while time.time() < deadline:
        state = arm._call(arm._robot.operationState)
        if state != last:
            transitions.append((round(time.time() % 1000, 2), str(state)))
            last = state
        time.sleep(interval)
    return transitions


def trial(arm, sdk, label, target_pose, speed, zone, set_mode):
    print(f"\n--- {label}")
    print(f"    speed={speed}  zone={zone}  set_nrt_mode={set_mode}")

    if set_mode:
        arm._call(arm._robot.setMotionControlMode, sdk.MotionControlMode.NrtCommandMode)
        print("    setMotionControlMode(NrtCommandMode) ok")

    before = arm.get_ee_pose()
    print(f"    state before : {arm._call(arm._robot.operationState)}")

    cmd = sdk.MoveLCommand(arm._to_cartesian_position(target_pose), speed, zone)
    arm._call(arm._robot.moveAppend, [cmd], sdk.PyString("diag"))
    arm._call(arm._robot.moveStart)
    print("    moveAppend + moveStart returned without error")

    transitions = poll_state(arm)
    print(f"    state transitions: {transitions if transitions else '(none observed)'}")

    after = arm.get_ee_pose()
    moved = float(np.linalg.norm(after[:3] - before[:3]))
    print(f"    displacement : {moved * 1000:.2f} mm")
    entered_moving = any("moving" in s for _, s in transitions)
    print(f"    entered `moving`: {entered_moving}")
    return moved, entered_moving


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--axis", default="x", choices="xyz")
    parser.add_argument("--step", type=float, default=0.010, help="metres")
    parser.add_argument("--speed", type=float, default=50.0)
    parser.add_argument("--disable-pendant", action="store_true",
                        help="Hand motion authority from the teach pendant to the SDK")
    parser.add_argument("--disable-sim", action="store_true",
                        help="Turn off input simulation mode")
    args = parser.parse_args()

    config = AR5Config().arm
    config.use_realtime = False

    arm = RokaeAR5(config)
    arm.connect(power_on=False)
    sdk = arm.sdk
    try:
        start = arm.get_ee_pose()
        idx = "xyz".index(args.axis)
        target = start.copy()
        target[idx] += args.step

        print(f"start  {np.round(start[:3], 5).tolist()}")
        print(f"target {np.round(target[:3], 5).tolist()}  ({args.step * 1000:.0f} mm on {args.axis})")

        lo = np.array(config.workspace_min)
        hi = np.array(config.workspace_max)
        if np.any(target[:3] < lo) or np.any(target[:3] > hi):
            print("target is outside the workspace box; pick a smaller step or another axis")
            return 1

        if input("\ntype 'go' to power on and run the trials: ").strip() != "go":
            print("aborted")
            return 1

        arm._power_on()
        print(f"power state: {arm._call(arm._robot.powerState)}")

        # While teach pendant mode is enabled the pendant holds motion
        # authority, and SDK commands are accepted but never executed -- which
        # is exactly "no error, OperationState stays idle, zero displacement".
        if args.disable_pendant:
            print("\n*** Disabling teach pendant mode: motion authority moves to the SDK.")
            print("    Nobody should be holding or relying on the pendant right now.")
            if input("    type 'yes' to continue: ").strip() != "yes":
                print("    skipped")
            else:
                try:
                    arm._call(arm._robot.setTeachPendantMode, False)
                    print("    teach pendant mode disabled")
                except Exception as exc:  # noqa: BLE001
                    print(f"    failed: {exc}")

        if args.disable_sim:
            try:
                arm._call(arm._robot.setSimulationMode, False)
                print("simulation mode disabled")
            except Exception as exc:  # noqa: BLE001
                print(f"could not disable simulation mode: {exc}")

        # A controller-side speed override of 0% produces exactly this
        # signature: the command is accepted and nothing moves.
        try:
            arm._call(arm._robot.adjustSpeedOnline, 0.3)
            print("speed override set to 30%")
        except Exception as exc:  # noqa: BLE001
            print(f"could not set speed override: {exc}")

        # Clear anything latched from earlier attempts.
        for name in ("moveReset", "clearServoAlarm"):
            try:
                arm._call(getattr(arm._robot, name))
                print(f"{name} ok")
            except Exception as exc:  # noqa: BLE001
                print(f"{name}: {exc}")

        # Any pending controller event usually names the real reason.
        for ev in ("moveExecution", "error", "safety"):
            event = getattr(sdk.Event, ev, None)
            if event is None:
                continue
            try:
                print(f"event[{ev}] = {arm._call(arm._robot.queryEventInfo, event)}")
            except Exception as exc:  # noqa: BLE001
                print(f"event[{ev}]: {exc}")

        results = {}
        # B alone: zone smaller than the motion, mode untouched.
        results["zone=0 only"] = trial(arm, sdk, "trial 1: zone=0, no mode change", target, args.speed, 0.0, False)
        arm.servo_to_pose(start); time.sleep(1.0)

        # A alone: original zone, but the Nrt mode explicitly set.
        results["mode only"] = trial(arm, sdk, "trial 2: NrtCommandMode, zone=10", target, args.speed, 10.0, True)
        arm.servo_to_pose(start); time.sleep(1.0)

        # A + B.
        results["mode + zone=0"] = trial(arm, sdk, "trial 3: NrtCommandMode, zone=0", target, args.speed, 0.0, True)
        arm.servo_to_pose(start); time.sleep(1.0)

        print("\n=== summary ===")
        for name, (moved, moving) in results.items():
            verdict = "MOVED" if moved > args.step * 0.5 else "no motion"
            print(f"  {name:<16} {moved * 1000:7.2f} mm   entered_moving={str(moving):<5}  {verdict}")

        movers = [n for n, (m, _) in results.items() if m > args.step * 0.5]
        print()
        if not movers:
            print("  Nothing moved. The cause is neither the mode nor the zone --")
            print("  check the teach pendant for an active fault, a held enable, or a")
            print("  speed override at 0%.")
        else:
            print(f"  Working configuration(s): {', '.join(movers)}")
    except KeyboardInterrupt:
        print("\ninterrupted -- stopping")
        arm.stop()
    except BaseException:
        print("\nunexpected error -- stopping")
        arm.stop()
        raise
    finally:
        if args.disable_pendant and arm._robot is not None:
            try:
                arm._call(arm._robot.setTeachPendantMode, True)
                print("teach pendant mode restored")
            except Exception as exc:  # noqa: BLE001
                print(f"could not restore teach pendant mode: {exc}")
        arm.disconnect()
        print("disconnected")
    return 0


if __name__ == "__main__":
    sys.exit(main())
