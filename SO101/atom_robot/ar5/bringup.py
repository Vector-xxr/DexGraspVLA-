"""Staged first-motion test for the AR5.

Every stage is gated: nothing moves until you type the confirmation for that
stage. Steps start at 5 mm and only grow once the measured displacement agrees
with what was commanded -- which is what catches a unit error or a wrong pose
convention before it becomes a collision.

    python -m atom_robot.ar5.bringup --dry-run   # rehearse, no motion
    python -m atom_robot.ar5.bringup             # the real thing

Keep a hand on the e-stop. Ctrl-C stops the arm and disconnects.

Stages
    0  connect and read           no power, no motion
    1  enable                     power on only
    2  probe each axis            +-step on X, Y, Z, returning to start each time
    3  probe rotation             a small rotation about Z
    4  report                     commanded vs measured, per axis
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from atom_robot.ar5.config import AR5Config  # noqa: E402
from atom_robot.ar5.drivers.rokae_arm import RokaeAR5, SDKError  # noqa: E402
from atom_robot.ar5.transforms import (  # noqa: E402
    clamp_delta,
    compose_pose,
    internal_pose_to_rpy,
    pose_delta,
)

# A commanded step counts as verified when the measured motion matches this
# closely. Loose enough for servo lag, tight enough to catch a factor-of-10
# unit error or an axis swap.
TOLERANCE_M = 0.002
TOLERANCE_RAD = 0.02


class Aborted(RuntimeError):
    pass


def confirm(prompt: str, expect: str = "go") -> None:
    """Gate a stage. Motion stages require typing a word, not just Enter."""
    got = input(f"\n{prompt}\n  type '{expect}' to proceed, anything else aborts: ").strip()
    if got != expect:
        raise Aborted("aborted at the prompt")


class BringUp:
    def __init__(self, arm, config, step: float, rot_step: float, dry_run: bool):
        self.arm = arm
        self.config = config
        self.step = step
        self.rot_step = rot_step
        self.dry_run = dry_run
        self.results = []

    # ------------------------------------------------------------- utilities

    def _inside_workspace(self, pose) -> bool:
        lo = np.array(self.config.workspace_min)
        hi = np.array(self.config.workspace_max)
        return bool(np.all(pose[:3] >= lo) and np.all(pose[:3] <= hi))

    def _goto(self, target, settle: float = 1.0):
        """Command one pose and return the measured pose after settling."""
        if not self._inside_workspace(target):
            raise Aborted(f"target {np.round(target[:3], 4).tolist()} is outside the workspace box")

        # Ask the controller whether it can reach this before committing. A
        # queued move that fails inverse kinematics is dropped silently -- the
        # only trace is a line in the controller log.
        check = getattr(self.arm, "check_reachable", None)
        if check is not None:
            ok, detail = check(target)
            if not ok:
                raise Aborted(f"controller refuses this target: {detail}")

        if self.dry_run:
            print(f"    [dry-run] would command {np.round(target[:3], 5).tolist()}")
            return self.arm.get_ee_pose()

        self.arm.servo_to_pose(target)
        time.sleep(settle)
        return self.arm.get_ee_pose()

    def _probe(self, label: str, delta: np.ndarray) -> None:
        """Command a delta, measure it, return to the start, and record."""
        start = self.arm.get_ee_pose()
        clamped, was_clamped = clamp_delta(
            delta, self.config.max_step_translation, self.config.max_step_rotation
        )
        if was_clamped:
            print(f"    note: delta clamped by the per-step limit")

        target = compose_pose(start, clamped)
        print(f"  {label}")
        print(f"    from     {np.round(start[:3], 5).tolist()}")
        print(f"    command  {np.round(target[:3], 5).tolist()}")

        confirm(f"  about to move: {label}")
        reached = self._goto(target)

        measured = pose_delta(start, reached)
        trans_err = float(np.linalg.norm(measured[:3] - clamped[:3]))
        rot_err = float(np.linalg.norm(measured[3:] - clamped[3:]))

        print(f"    measured {np.round(reached[:3], 5).tolist()}")
        print(f"    commanded delta {np.round(clamped, 5).tolist()}")
        print(f"    measured  delta {np.round(measured, 5).tolist()}")
        print(f"    error: {trans_err * 1000:.2f} mm, {rot_err:.4f} rad")

        ok = trans_err <= TOLERANCE_M and rot_err <= TOLERANCE_RAD
        if self.dry_run:
            ok = None
        elif not ok:
            print("    *** MISMATCH -- do not raise the step size.")
            print("        A large error here usually means a unit mismatch, an axis")
            print("        swap, or the wrong rotation convention.")
        self.results.append((label, clamped, measured, trans_err, rot_err, ok))

        # Always return to where we started, so each probe is independent.
        print("    returning to start")
        self._goto(start)

    # ---------------------------------------------------------------- stages

    def stage_read(self) -> np.ndarray:
        print("\n=== Stage 0: connect and read (no power, no motion) ===")
        q = self.arm.get_joint_positions()
        pose = self.arm.get_ee_pose()
        trans, rpy = internal_pose_to_rpy(pose)
        print(f"  joints    {np.round(q, 4).tolist()}")
        print(f"  position  {np.round(trans, 5).tolist()} m")
        print(f"  RPY       {np.round(rpy, 5).tolist()} rad")
        print(f"  elbow     {self.arm.get_elbow():.5f} rad")
        print(f"  inside workspace box: {self._inside_workspace(pose)}")
        if not self._inside_workspace(pose):
            raise Aborted(
                "the arm starts outside the workspace box; move it inside or widen the box"
            )
        return pose

    def stage_enable(self) -> None:
        print("\n=== Stage 1: enable ===")
        if self.dry_run:
            print("  [dry-run] skipping power-on")
            return
        confirm("Power on the arm? It will hold position, not move.", expect="power")
        self.arm._power_on()
        state = self.arm._call(self.arm._robot.powerState)
        print(f"  power state: {state}")
        if str(state).endswith("off"):
            raise Aborted(
                "the arm did not power on. If a teach pendant is attached, enable it there, "
                "then re-run."
            )

    def stage_axes(self) -> None:
        print(f"\n=== Stage 2: translation, {self.step * 1000:.0f} mm per axis ===")
        for i, axis in enumerate("xyz"):
            for sign in (1, -1):
                delta = np.zeros(6)
                delta[i] = sign * self.step
                self._probe(f"{'+' if sign > 0 else '-'}{axis}  {sign * self.step * 1000:+.0f} mm", delta)

    def stage_rotation(self) -> None:
        print(f"\n=== Stage 3: rotation, {self.rot_step:.3f} rad about Z ===")
        delta = np.zeros(6)
        delta[5] = self.rot_step
        self._probe(f"+rz  {self.rot_step:+.3f} rad", delta)

    def report(self) -> bool:
        print("\n=== Stage 4: report ===")
        print(f"  {'probe':<16} {'commanded':>12} {'measured':>12} {'error':>10}   result")
        all_ok = True
        for label, cmd, meas, terr, rerr, ok in self.results:
            is_rot = abs(cmd[5]) > 1e-9
            c = cmd[5] if is_rot else np.linalg.norm(cmd[:3]) * np.sign(cmd[:3].sum() or 1)
            m = meas[5] if is_rot else np.linalg.norm(meas[:3]) * np.sign(meas[:3].sum() or 1)
            err = f"{rerr:.4f}rad" if is_rot else f"{terr * 1000:.2f}mm"
            verdict = "n/a (dry-run)" if ok is None else ("ok" if ok else "MISMATCH")
            all_ok = all_ok and (ok is not False)
            print(f"  {label:<16} {c:>12.5f} {m:>12.5f} {err:>10}   {verdict}")

        print()
        if self.dry_run:
            print("  Dry run only. Re-run without --dry-run to actually move.")
        elif all_ok:
            print("  All probes matched. The pose convention and units are consistent.")
            print(f"  Next: raise --step towards {self.config.max_step_translation * 1000:.0f} mm,")
            print("  then set use_realtime=True and try the continuous loop.")
        else:
            print("  At least one probe mismatched. Do not raise the step size.")
        return all_ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ip", default="", help="Override the controller IP")
    parser.add_argument("--step", type=float, default=0.005, help="Translation step, metres")
    parser.add_argument("--rot-step", type=float, default=0.05, help="Rotation step, radians")
    parser.add_argument("--max-step", type=float, default=None,
                        help="Raise the per-step safety limit (metres). Defaults to the config value.")
    parser.add_argument("--dry-run", action="store_true", help="Rehearse without commanding motion")
    parser.add_argument("--skip-rotation", action="store_true", help="Translation probes only")
    args = parser.parse_args()

    config = AR5Config().arm
    if args.ip:
        config.ip = args.ip
    # Discrete queued moves for bring-up: bounded by construction, and easier to
    # reason about than a streaming loop.
    config.use_realtime = False
    # The probe never exceeds the safety limit, and the limit is only raised
    # when asked for explicitly -- so a large --step alone cannot widen it.
    if args.max_step is not None:
        config.max_step_translation = args.max_step
    config.max_step_translation = min(config.max_step_translation, max(args.step, 1e-9))
    config.max_step_rotation = min(config.max_step_rotation, max(args.rot_step, 1e-9))
    if args.step > config.max_step_translation + 1e-12:
        print(f"--step {args.step * 1000:.0f} mm exceeds the per-step limit "
              f"{config.max_step_translation * 1000:.0f} mm; pass --max-step to raise it.")
        return 1

    print(f"Controller     : {config.ip}")
    print(f"Step           : {args.step * 1000:.0f} mm, {args.rot_step:.3f} rad")
    print(f"Per-step limit : {config.max_step_translation * 1000:.0f} mm, "
          f"{config.max_step_rotation:.3f} rad")
    print(f"Workspace box  : {list(config.workspace_min)} .. {list(config.workspace_max)}")
    if args.dry_run:
        print("Mode           : DRY RUN, nothing will move")

    arm = RokaeAR5(config)
    arm.connect(power_on=False)
    stages = BringUp(arm, config, args.step, args.rot_step, args.dry_run)
    ok = False
    try:
        stages.stage_read()
        stages.stage_enable()
        stages.stage_axes()
        if not args.skip_rotation:
            stages.stage_rotation()
        ok = stages.report()
    except Aborted as exc:
        print(f"\nAborted: {exc}")
        arm.stop()
    except KeyboardInterrupt:
        print("\nInterrupted -- stopping the arm")
        arm.stop()
    except SDKError as exc:
        print(f"\nSDK error: {exc}")
        arm.stop()
    except BaseException:
        # Anything unexpected must still halt the arm before it propagates --
        # a bare TypeError mid-motion would otherwise leave it running.
        print("\nUnexpected error -- stopping the arm")
        arm.stop()
        raise
    finally:
        arm.disconnect()
        print("disconnected")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
