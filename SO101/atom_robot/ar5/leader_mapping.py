"""Map an SO-101 leader arm onto the AR5's Cartesian action space.

The two arms share no kinematics: the leader has six joints and a ~0.3 m reach,
the AR5 has seven and ~0.68 m. Joint angles therefore cannot be copied across.
What does transfer is end-effector *motion*: run forward kinematics on the
leader, take the change in its tool pose, scale it, and emit that as the AR5's
Cartesian delta.

Working in deltas rather than absolute poses avoids ever having to calibrate one
arm's base frame against the other's, and it lands directly in the 7-dim action
layout that ``AR5Env.step`` and the VEL energy head already use:

    [0:3]  position delta, metres
    [3:6]  orientation delta, axis-angle radians
    [6]    gripper opening in [0, 1], 0 = closed

A clutch makes the workspaces independent. While disengaged the leader can be
repositioned freely and the follower holds still, so a small leader can drive a
large follower by "rowing" -- exactly how a mouse is lifted and replaced.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from .transforms import matrix_to_axis_angle

logger = logging.getLogger(__name__)

# Where `calibrate_mapping` leaves its result, and where `MappingConfig.load`
# looks for it. Keeping it on disk rather than in source means the measurement
# survives edits to this file and needs no copy-paste step to take effect.
CALIBRATION_FILE = Path(__file__).resolve().parent / "leader_calibration.json"
CALIBRATED_FIELDS = ("axis_permutation", "axis_signs", "gripper_closed_deg", "gripper_open_deg")


def mapping_matrix(permutation: Tuple[int, int, int], signs: Tuple[float, float, float]) -> np.ndarray:
    """The 3x3 frame change described by ``axis_permutation`` and ``axis_signs``.

    Row *i* picks source axis ``permutation[i]`` and scales it by ``signs[i]``,
    so ``M @ v`` reproduces the older ``v[perm] * signs`` exactly.
    """
    M = np.zeros((3, 3), dtype=np.float64)
    for i, src in enumerate(permutation):
        M[i, src] = float(signs[i])
    return M

SO_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
LEADER_EE_FRAME = "gripper_frame_link"


@dataclass
class MappingConfig:
    """How leader motion becomes follower motion."""

    urdf_path: str = ""          # Defaults to the bundled SO-101 FK model
    ee_frame: str = LEADER_EE_FRAME

    # Leader travel is multiplied by this to get follower travel. The AR5's
    # reach is roughly twice the leader's, so >1 lets the whole follower
    # workspace be covered; start at 1.0 while learning the feel.
    position_scale: float = 1.0
    rotation_scale: float = 1.0
    # 保留哪几个旋转轴（基座系）。"xyz" 全开；"z" 只留绕世界 Z 的自转。
    #
    # **为什么需要按轴关。** 螺丝刀这类任务要求工具始终竖直：绕 X/Y 一动刀头就歪了，
    # 而绕 Z 的自转正是拧螺丝那个自由度。全关（--no-rotation）会把自转也一起关掉，
    # 全开又保不住竖直。中间态只能按轴选。
    #
    # 姿态不会漂：被关掉的轴每步都置零，增量累加不到姿态上。**但起始姿态要自己摆正** ——
    # 它只保持，不纠正。用 goto_episode_pose / capture_pose 先摆到竖直。
    rotation_axes: str = "xyz"

    # Axis remap and sign flip, applied to the leader's delta before scaling.
    # Identity assumes both bases share an orientation; if the leader sits
    # rotated relative to the AR5, fix it here rather than in your wrist.
    axis_permutation: Tuple[int, int, int] = (0, 1, 2)
    axis_signs: Tuple[float, float, float] = (1.0, 1.0, 1.0)

    # Motion below these thresholds is treated as noise and dropped. The floor
    # is set by the servo encoders, not by the hand: one count is 0.088 deg,
    # which at the leader's ~0.22 m reach is about 0.34 mm at the tool, and
    # several joints dither at once. A deadband under that lets quantisation
    # through as commanded motion. Sub-threshold motion still accumulates, so
    # raising this costs resolution of speed, not of reach.
    position_deadband: float = 0.0015   # metres per step
    rotation_deadband: float = 0.004    # radians per step

    # Gripper servo degrees at fully closed and fully open, mapped to 0 and 1.
    # Read them off `probe_leader --watch` while squeezing the trigger.
    gripper_closed_deg: float = -60.0
    gripper_open_deg: float = 0.0

    # Orientation is the least intuitive part of cross-arm teleop; turning it
    # off gives position-only control, which is much easier to learn.
    enable_rotation: bool = True

    @classmethod
    def load(cls, path: Optional[Path] = None, **overrides) -> "MappingConfig":
        """Build a config with any saved calibration applied.

        Only the four measured fields come from disk; everything else stays at
        its default or whatever ``overrides`` says. Explicit overrides win over
        the file, so a command-line flag is never silently ignored.
        """
        path = Path(path) if path is not None else CALIBRATION_FILE
        values = {}
        if path.exists():
            try:
                saved = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning(f"[MappingConfig] ignoring unreadable {path}: {exc}")
                saved = {}
            for key in CALIBRATED_FIELDS:
                if key in saved:
                    values[key] = saved[key]
            if values:
                logger.info(
                    f"[MappingConfig] loaded calibration from {path.name} "
                    f"(measured {saved.get('measured_at', 'unknown time')})"
                )
        else:
            logger.warning(
                f"[MappingConfig] no {path.name}; axis map falls back to identity, "
                "which is right only if the leader's base points the same way as "
                "the arm's. Run `python -m atom_robot.ar5.calibrate_mapping`."
            )

        values.update(overrides)
        if "axis_permutation" in values:
            values["axis_permutation"] = tuple(int(v) for v in values["axis_permutation"])
        if "axis_signs" in values:
            values["axis_signs"] = tuple(float(v) for v in values["axis_signs"])
        return cls(**values)


class LeaderMapper:
    """Turns leader joint readings into AR5 actions."""

    def __init__(self, config: Optional[MappingConfig] = None):
        self.config = config or MappingConfig()
        self._robot = None
        self._prev_p: Optional[np.ndarray] = None
        self._prev_R: Optional[np.ndarray] = None
        self._engaged = False

        self._M = mapping_matrix(self.config.axis_permutation, self.config.axis_signs)
        # An axis remap with an odd number of sign flips mirrors the frame. A
        # rotation vector is a pseudo-vector, so under a mirror it picks up an
        # extra negation: R' = M R M^-1 = exp([det(M) M r]x). Without this a
        # mirrored setup -- which is what you get sitting across from the arm --
        # translates correctly while every rotation comes out backwards.
        self._rot_parity = float(np.sign(np.linalg.det(self._M)) or 1.0)

        self.stats = {"steps": 0, "deadbanded": 0, "clutch_cycles": 0}

    # ------------------------------------------------------------------ setup

    def connect(self) -> None:
        import placo

        urdf = self.config.urdf_path
        if not urdf:
            from pathlib import Path

            from atom_robot import VEL_REPO
            urdf = str(VEL_REPO.parent / "assets" / "so101" / "so101_fk.urdf")

        self._robot = placo.RobotWrapper(urdf)
        frames = list(self._robot.frame_names())
        if self.config.ee_frame not in frames:
            raise ValueError(
                f"Frame {self.config.ee_frame!r} is not in the URDF. Available: {frames}"
            )
        logger.info(f"[LeaderMapper] loaded {urdf}, tool frame {self.config.ee_frame}")

    # -------------------------------------------------------------- kinematics

    def leader_frame(self, joints_deg: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
        """Forward kinematics -> ``(position, rotation matrix)`` in the leader base frame.

        The rotation is kept as a matrix rather than an axis-angle vector because
        orientation *changes* have to be composed, not subtracted.
        """
        if self._robot is None:
            raise RuntimeError("LeaderMapper used before connect()")

        for name in SO_JOINTS:
            if name in joints_deg:
                self._robot.set_joint(name, np.deg2rad(joints_deg[name]))
        self._robot.update_kinematics()

        T = np.asarray(self._robot.get_T_world_frame(self.config.ee_frame))
        return T[:3, 3].copy(), T[:3, :3].copy()

    def leader_pose(self, joints_deg: Dict[str, float]) -> np.ndarray:
        """Forward kinematics -> ``[x, y, z, rx, ry, rz]`` in the leader base frame."""
        p, R = self.leader_frame(joints_deg)
        return np.concatenate([p, matrix_to_axis_angle(R)])

    # ------------------------------------------------------------------ clutch

    def engage(self, joints_deg: Dict[str, float]) -> None:
        """Start following. The current leader pose becomes the reference."""
        self._prev_p, self._prev_R = self.leader_frame(joints_deg)
        if not self._engaged:
            self.stats["clutch_cycles"] += 1
        self._engaged = True

    def disengage(self) -> None:
        """Stop following. The leader can be repositioned without moving the follower."""
        self._engaged = False
        self._prev_p = None
        self._prev_R = None

    @property
    def engaged(self) -> bool:
        return self._engaged

    # ------------------------------------------------------------------ action

    def gripper_command(self, joints_deg: Dict[str, float]) -> float:
        """Leader trigger angle -> opening in [0, 1], 0 = closed."""
        raw = joints_deg.get("gripper")
        if raw is None:
            return 1.0
        closed, opened = self.config.gripper_closed_deg, self.config.gripper_open_deg
        if abs(opened - closed) < 1e-6:
            return 1.0
        return float(np.clip((raw - closed) / (opened - closed), 0.0, 1.0))

    def compute_action(self, joints_deg: Dict[str, float]) -> np.ndarray:
        """One 7-dim AR5 action from the current leader reading.

        While disengaged the arm delta is zero but the gripper still tracks, so
        the hand can be opened and closed without driving the arm.
        """
        cfg = self.config
        action = np.zeros(7, dtype=np.float64)
        action[6] = self.gripper_command(joints_deg)

        if not self._engaged:
            return action

        p, R = self.leader_frame(joints_deg)
        if self._prev_p is None or self._prev_R is None:
            self._prev_p, self._prev_R = p, R
            return action

        dp = p - self._prev_p

        # The orientation change is a composition, not a difference. Subtracting
        # two axis-angle vectors is only ever an approximation, and it is a bad
        # one as soon as the wrist turns appreciably or either angle passes near
        # pi, where the representation wraps and the difference explodes.
        # `R @ prev.T` is the base-frame delta, matching how `compose_pose`
        # applies action[3:6] on the follower (R_delta @ R_current).
        dr = matrix_to_axis_angle(R @ self._prev_R.T)

        translation = (self._M @ dp) * cfg.position_scale

        rotation = np.zeros(3)
        if cfg.enable_rotation:
            rotation = (self._rot_parity * (self._M @ dr)) * cfg.rotation_scale
            for i, axis in enumerate("xyz"):
                if axis not in cfg.rotation_axes:
                    rotation[i] = 0.0

        # Deadband on the *leader's* raw motion, so the threshold means the same
        # thing regardless of how the scale is set.
        if np.linalg.norm(dp) < cfg.position_deadband:
            translation[:] = 0.0
        if np.linalg.norm(dr) < cfg.rotation_deadband:
            rotation[:] = 0.0
        if not translation.any() and not rotation.any():
            self.stats["deadbanded"] += 1

        action[:3] = translation
        action[3:6] = rotation

        # Advance the reference only by what was actually emitted, so motion
        # suppressed by the deadband accumulates instead of being lost.
        if translation.any():
            self._prev_p = p
        if rotation.any():
            self._prev_R = R

        self.stats["steps"] += 1
        return action
