#!/usr/bin/env python3
"""Find the centre posture by eye: ask for a tool height, watch the arm go there.

`teleop_joint --centre ... --home` can do this too, but only one height per
run, and each run re-connects and re-powers the arm. This keeps one session
open and moves between candidates, which is what picking a height against a
real table actually needs.

    python -m atom_robot.ar5.tune_centre                  # interactive
    python -m atom_robot.ar5.tune_centre --dry-run        # no arm, just the numbers
    python -m atom_robot.ar5.tune_centre --hand-length 0.18

The centre posture is what `teleop_joint` maps the leader onto, so its height
sets where the whole workspace sits. Lowering it costs elbow travel: the elbow
bends further to bring the tool down, which leaves less room before its +145
degree limit, and the leader-to-arm scale is fitted to whatever room is left.
Each candidate prints that cost before anything moves.

Nothing moves without a typed `y`. Ctrl-C stops the arm and disconnects.

**The floor guard does not know about the hand.** It checks the DH chain tip --
the flange plus a 90 mm tool row -- while the fingers reach further. Pass
`--hand-length` to have the fingertip height printed alongside; it is a
reported number only, the guard itself still cannot see it.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from atom_robot.ar5.ar5_env import AR5Env  # noqa: E402
from atom_robot.ar5.config import AR5Config  # noqa: E402
from atom_robot.ar5.joint_mapping import (  # noqa: E402
    CALIBRATION_FILE,
    JOINT_PAIRS,
    JointMapConfig,
)

# Leader half-travel in degrees, measured on this SO-101 (HANDOFF section 6).
LEADER_HALF_DEG = {
    "shoulder_pan": 110.6,
    "shoulder_lift": 105.2,
    "elbow_flex": 97.2,
    "wrist_roll": 180.0,
    "wrist_flex": 101.8,
}


def solve_centre(kin, cfg, z_target: float, x_hint: float = 0.45):
    """Pick (j1, j3) putting the tool at `z_target`, keeping the most elbow travel.

    j0/j2/j4/j5/j6 stay at zero: j2 and j4 are the locked joints, and leaving
    the wrist at zero keeps the tool pointing the way the mapping assumes.
    Among the postures that hit the height, the one with the most room left
    before the elbow's limit wins, because that room is what the leader-to-arm
    scale is fitted from.
    """
    jmin = np.asarray(cfg.joint_min)
    jmax = np.asarray(cfg.joint_max)
    margin = cfg.joint_limit_margin

    def scan(j1_range, j3_range, tol):
        best = None
        for j1 in j1_range:
            for j3 in j3_range:
                q = np.deg2rad([0.0, j1, 0.0, j3, 0.0, 0.0, 0.0])
                if np.any(q < jmin + margin) or np.any(q > jmax - margin):
                    continue
                tool = kin.fk(q)[:3, 3]
                if abs(tool[2] - z_target) > tol:
                    continue
                # Hard window on reach. A soft penalty alone loses to the elbow
                # score and walks the solution out to the edge of the workspace,
                # where there is no room left to move forward.
                if abs(tool[0] - x_hint) > 0.08:
                    continue
                scales = fit_scales(q, cfg)
                # The wrist is capped by its own soft limit at every height, so
                # scoring on it would make all candidates look identical.
                score = min(v for k, v in scales.items() if k != "wrist_flex")
                # Weighted to actually bite: elbow scale varies by ~0.05 between
                # nearby candidates, so a weak reach penalty just gets ignored
                # and the solver drifts out to the edge of the workspace.
                score -= 1.0 * abs(tool[0] - x_hint)
                if best is None or score > best[0]:
                    best = (score, q, tool, scales)
        return best

    # Coarse then local: a 0.25 degree sweep of the whole j1/j3 plane is a third
    # of a million FK calls per height, which is slow enough to be unusable
    # interactively. One degree finds the basin, then a tight sweep refines it.
    coarse = scan(np.arange(-20.0, 100.0, 1.0), np.arange(-20.0, 145.0, 1.0), 0.02)
    if coarse is None:
        return None
    _score, q_c, _tool, _scales = coarse
    j1_c, j3_c = np.rad2deg(q_c[1]), np.rad2deg(q_c[3])
    fine = scan(np.arange(j1_c - 1.5, j1_c + 1.5, 0.05),
                np.arange(j3_c - 1.5, j3_c + 1.5, 0.05), 0.004)
    best = fine or coarse
    return best[1:]


def fit_scales(q: np.ndarray, cfg) -> dict:
    """Leader-to-arm scale each joint would get at this centre."""
    jmin = np.asarray(cfg.joint_min)
    jmax = np.asarray(cfg.joint_max)
    margin = cfg.joint_limit_margin
    out = {}
    for name, joint, _sign in JOINT_PAIRS:
        room = min(q[joint] - (jmin[joint] + margin), (jmax[joint] - margin) - q[joint])
        out[name] = float(min(1.0, room / np.deg2rad(LEADER_HALF_DEG[name])))
    return out


def lowest_point(kin, q: np.ndarray) -> float:
    """Same quantity the floor guard uses: tool tip and the outboard origins."""
    T = kin.fk(q)
    _, _, origins = kin._chain(q)
    return min([float(T[2, 3])] + [float(origins[i][2]) for i in range(3, len(origins))])


def fingertip_height(kin, q: np.ndarray, hand_length: float) -> float:
    """Where the fingertips sit, assuming the hand extends along the tool's z axis.

    That axis is the usual approach direction, but it is an assumption about how
    the hand is bolted on, not something the DH table knows -- the chain stops at
    the flange plus a 90 mm tool row. Worth checking against the real hand once.
    """
    T = kin.fk(q)
    return float(T[2, 3] + T[2, 2] * hand_length)


def describe(kin, cfg, q: np.ndarray, hand_length: float) -> None:
    tool = kin.fk(q)[:3, 3]
    low = lowest_point(kin, q)
    scales = fit_scales(q, cfg)
    # Where the tool points matters as much as where it is: the hand extends
    # along this axis, so an upward-pointing tool cannot reach a table no matter
    # how low the wrist sits.
    approach_z = float(kin.fk(q)[2, 2])
    facing = "down" if approach_z < -0.5 else ("level" if approach_z < 0.5 else "UP")
    print(f"    posture   {np.round(np.rad2deg(q), 1).tolist()} deg")
    print(f"    tool      ({tool[0]:.3f}, {tool[1]:.3f}, {tool[2]:.3f}) m")
    print(f"    approach  {approach_z:+.2f} ({facing})   -1 is straight down")
    print(f"    lowest    {low:.3f} m   (floor guard trips below {cfg.joint_floor_z:.3f})")
    if hand_length > 0:
        tip = fingertip_height(kin, q, hand_length)
        flag = "  <-- BELOW the floor guard, which cannot see it" if tip < cfg.joint_floor_z else ""
        print(f"    fingertip {tip:.3f} m   (along the tool z axis){flag}")
    print("    scale     " + "  ".join(f"{n.split('_')[-1]}={scales[n]:.2f}" for n, _, _ in JOINT_PAIRS))


def require_realtime(env) -> bool:
    """Fail loudly if the real-time stream did not come up.

    The driver downgrades any real-time startup failure to a warning and carries
    on with `_rt_active = False`. Every later `servo_to_joints` then raises
    "requires the real-time stream", which names the symptom and not the cause --
    the cause was printed once, at connect time, and scrolled past.
    """
    if getattr(env.arm, "_rt_active", False):
        return True
    print("\nThe real-time stream is not running, so the arm cannot be moved.")
    print("The reason was logged as a warning just above. Common causes:")
    print("  * the teach pendant is in manual mode or holds the enable")
    print("    -> switch it to automatic and make sure nothing else is connected")
    print("  * the arm is not powered (probe_rokae --connect --read-state shows this)")
    print("  * another process still owns the controller session")
    return False


def build_config() -> AR5Config:
    """The configuration this procedure runs under: bare arm, no cameras.

    Separated out so it can be asserted without hardware. Setting a field that
    does not exist on a dataclass silently creates an attribute nothing reads,
    so "I disabled the cameras" is not something to take on faith.
    """
    config = AR5Config()
    # Bare arm by design: this procedure is about where the flange goes, and
    # running it with the hand attached would put fingers near a table while
    # the floor guard still cannot see them.
    config.use_hand = False
    # Cameras are disabled by emptying the source map -- there is no
    # `use_cameras` flag, and setting one just creates an attribute nothing reads.
    config.cameras.sources = {}
    return config


def confirm(prompt: str) -> bool:
    try:
        return input(prompt).strip().lower() == "y"
    except (EOFError, KeyboardInterrupt):
        print()
        return False


def move_to(env, q_target: np.ndarray, step_rad: float = 0.01) -> np.ndarray:
    """Ramp to the target posture, then hold it long enough to settle."""
    q_start = env.arm.get_joint_positions()
    steps = max(1, int(np.max(np.abs(q_target - q_start)) / step_rad))
    for i in range(1, steps + 1):
        env.step_joints(q_start + (q_target - q_start) * (i / steps))
        time.sleep(0.02)
    for _ in range(60):
        env.step_joints(q_target)
        time.sleep(0.02)
    return env.arm.get_joint_positions()


def save_centre(q: np.ndarray) -> None:
    """Write only the centre, preserving any signs and scales already measured."""
    existing = {}
    if CALIBRATION_FILE.exists():
        try:
            existing = json.loads(CALIBRATION_FILE.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  ignoring unreadable {CALIBRATION_FILE.name}: {exc}")
    existing["centre_joints"] = [float(v) for v in q]
    existing["centre_measured_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    CALIBRATION_FILE.write_text(json.dumps(existing, indent=2) + "\n")
    print(f"\n  saved centre_joints to {CALIBRATION_FILE}")
    if "sign_overrides" not in existing:
        print("  (signs and scales are not in this file yet -- confirm them with")
        print("   `teleop_joint ... --save`, which keeps this centre.)")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Solve and print candidates without connecting to the arm")
    parser.add_argument("--hand-length", type=float, default=0.0,
                        help="Flange face to fingertip, metres. Only reported, not enforced")
    parser.add_argument("--x-hint", type=float, default=0.45,
                        help="Preferred tool reach in x; the solver stays near it")
    args = parser.parse_args()

    # Without this the driver's warnings go nowhere, and a real-time stream that
    # failed to start looks like an unexplained error several steps later.
    logging.basicConfig(level=logging.INFO, format="  %(message)s")

    config = build_config()
    cfg = config.arm

    current = np.asarray(JointMapConfig.load().centre_joints, dtype=np.float64)

    if args.dry_run:
        import json as _json

        from atom_robot.ar5.kinematics import AR5Kinematics
        dh = _json.loads((Path(__file__).resolve().parent / "ar5_dh.json").read_text())["dh"]
        kin = AR5Kinematics(np.array(dh).reshape(-1, 4), cfg.num_joints, cfg.joint_min, cfg.joint_max)
        print("\ncurrent centre:")
        describe(kin, cfg, current, args.hand_length)
        print("\ncandidates:")
        for z in (0.60, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20):
            found = solve_centre(kin, cfg, z, args.x_hint)
            if found is None:
                print(f"\n  z={z:.2f}: no posture reaches this height")
                continue
            q, _tool, _scales = found
            print(f"\n  z={z:.2f}:")
            describe(kin, cfg, q, args.hand_length)
        return 0

    print("\nConnecting. This powers the arm on.")
    if not confirm("  Area clear, hand on the e-stop? [y/N] "):
        print("  aborted")
        return 1

    env = AR5Env(config)
    env.connect()
    try:
        if not require_realtime(env):
            return 1

        kin = env.arm.kinematics
        if kin is None:
            print("No local kinematics; cannot solve for a height.")
            return 1

        q_now = env.arm.get_joint_positions()
        print(f"\n  arm is at {np.round(np.rad2deg(q_now), 1).tolist()} deg")
        print(f"  tool at   {np.round(kin.fk(q_now)[:3, 3], 3).tolist()} m")
        print("\ncurrent centre in the config:")
        describe(kin, cfg, current, args.hand_length)

        chosen = None
        print("\nEnter a target tool height in metres (e.g. 0.30), or 'q' to stop.")
        while True:
            try:
                raw = input("\n  target z (m) > ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not raw or raw.lower() in ("q", "quit"):
                break
            try:
                z = float(raw)
            except ValueError:
                print("  not a number")
                continue

            found = solve_centre(kin, cfg, z, args.x_hint)
            if found is None:
                print(f"  no posture puts the tool at z={z:.3f} with j2/j6 locked at zero")
                continue
            q_target, _tool, _scales = found
            print()
            describe(kin, cfg, q_target, args.hand_length)

            if not confirm("\n    move there? [y/N] "):
                continue
            reached = move_to(env, q_target)
            err = float(np.max(np.abs(reached - q_target)))
            tool = kin.fk(reached)[:3, 3]
            print(f"    arrived, worst joint off by {np.rad2deg(err):.3f} deg")
            print(f"    measured tool ({tool[0]:.3f}, {tool[1]:.3f}, {tool[2]:.3f}) m")
            chosen = reached if err < np.deg2rad(1.0) else q_target

            if confirm("    keep this as the centre? [y/N] "):
                save_centre(chosen)
                break

        print("\nDone. The arm stays where it is.")
    except KeyboardInterrupt:
        print("\n  interrupted; stopping the arm")
        env.arm.stop()
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
