"""In-process stand-ins for the arm, hand and cameras.

These make the full pipeline runnable with no hardware attached: the same
environment code, the same safety limits, the same observation dict. The arm
mock carries no kinematics -- it tracks the commanded Cartesian pose and
synthesizes plausible joint angles -- which is enough to exercise the control
loop, the action encoding and the client/server round trip.
"""

import logging
import threading

import numpy as np

from ..transforms import axis_angle_to_matrix
from .base import ArmDriver, HandDriver

logger = logging.getLogger(__name__)


class MockArm(ArmDriver):
    # 真臂上这是「收不到目标已经刹车、什么都不接了」。mock 永远不刹车，
    # 但**这个属性必须存在**：采集回路每帧都查一次 `env.arm.latched`，
    # 没有它第一帧就 AttributeError —— 于是整条 collect 在 mock 下根本跑不起来，
    # 也就没法不接硬件验按键、回 0 位、存盘这些逻辑。
    latched = False

    def __init__(self, config, home_pose=None):
        self.config = config
        self._home_pose = np.array(
            home_pose if home_pose is not None else [0.45, 0.0, 0.30, np.pi, 0.0, 0.0],
            dtype=np.float64,
        )
        self._pose = self._home_pose.copy()
        self._joints = np.array(config.home_joints, dtype=np.float64)
        self._connected = False
        self.command_count = 0

        # Enough of the real-time surface for the joint-space path to run
        # against. The driver exposes `_q_cmd` under `_setpoint_lock` as the
        # tracker's own setpoint, and callers read it rather than the encoders to
        # avoid feeding measurement noise back into the goal.
        self._setpoint_lock = threading.Lock()
        self._q_cmd = self._joints.copy()
        self._q_goal = self._joints.copy()
        self._rt_active = False
        self.kinematics = None
        self.joint_command_count = 0
        self.rejected_joint_goals = 0

    def connect(self) -> None:
        self._connected = True
        logger.info("[MockArm] connected (no hardware)")

    def disconnect(self) -> None:
        self._connected = False

    def _require_connection(self) -> None:
        if not self._connected:
            raise RuntimeError("Arm driver used before connect()")

    def get_joint_positions(self) -> np.ndarray:
        self._require_connection()
        return self._joints.copy()

    def get_ee_pose(self) -> np.ndarray:
        self._require_connection()
        return self._pose.copy()

    def servo_to_pose(self, pose: np.ndarray, wait: bool = True) -> None:
        self._require_connection()
        self._pose = np.asarray(pose, dtype=np.float64).reshape(6).copy()
        self.command_count += 1
        # Stand-in for inverse kinematics: a smooth map from the 6-vector pose to
        # however many joints this arm has. The values are not physically
        # meaningful, but they move continuously with the pose, which is what the
        # proprio channel needs. Extra joints beyond the first six (the AR series
        # is 7-DoF and redundant) get a phase-shifted combination.
        n = self.config.num_joints
        base = np.array(
            [
                self._pose[0] * 2.0,
                self._pose[1] * 2.0 - 0.3,
                self._pose[2] * 2.0 + 1.2,
                self._pose[3] * 0.5,
                self._pose[4] * 0.5 + 1.0,
                self._pose[5] * 0.5,
            ],
            dtype=np.float64,
        )
        if n <= 6:
            self._joints = base[:n]
        else:
            extra = np.array([base[i % 6] * 0.5 for i in range(6, n)], dtype=np.float64)
            self._joints = np.concatenate([base, extra])

    def move_to_joints(self, joints: np.ndarray, blocking: bool = True) -> None:
        self._require_connection()
        self._joints = np.asarray(joints, dtype=np.float64).reshape(self.config.num_joints).copy()
        self._pose = self._home_pose.copy()

    def servo_to_joints(self, joints: np.ndarray) -> bool:
        """Accept a joint goal, applying the same gates the real driver applies.

        Not a no-op that always succeeds: a mock that accepts anything would hide
        exactly the mistakes this path is most likely to make -- a goal outside
        the limits, or a jump the real arm would refuse.
        """
        self._require_connection()
        q = np.asarray(joints, dtype=np.float64).reshape(self.config.num_joints)

        if not np.all(np.isfinite(q)):
            self.rejected_joint_goals += 1
            return False

        lo = np.asarray(self.config.joint_min) + self.config.joint_limit_margin
        hi = np.asarray(self.config.joint_max) - self.config.joint_limit_margin
        if np.any(q < lo) or np.any(q > hi):
            self.rejected_joint_goals += 1
            return False

        if float(np.max(np.abs(q - self._q_cmd))) > self.config.joint_jump_max:
            self.rejected_joint_goals += 1
            return False

        with self._setpoint_lock:
            self._q_goal = q.copy()
            self._q_cmd = q.copy()
        self._joints = q.copy()
        self.joint_command_count += 1
        self.command_count += 1

        if self.kinematics is not None:
            T = self.kinematics.fk(q)
            self._pose[:3] = T[:3, 3]
        return True

    def clear_latch(self) -> None:
        pass

    def rt_stats(self) -> dict:
        return {
            "sent": self.joint_command_count,
            "late_ticks": 0,
            "limit_hits": 0,
            "goals": self.joint_command_count,
            "rejected": {"latched": 0, "nonfinite": 0, "limits": self.rejected_joint_goals,
                         "jump": 0},
            "latched": False,
        }

    def get_elbow(self) -> float:
        """Arm angle. Seven axes are redundant; the real arm reports this."""
        return float(self._joints[2]) if self._joints.size > 2 else 0.0

    def has_motion_error(self) -> bool:
        return False

    def recover_from_error(self) -> None:
        pass

    def solve_ik(self, pose: np.ndarray, arm_angle: float = 0.0):
        """Mock IK: everything inside the workspace box is declared reachable."""
        pose = np.asarray(pose, dtype=np.float64).reshape(6)
        lo = np.array(self.config.workspace_min)
        hi = np.array(self.config.workspace_max)
        if np.any(pose[:3] < lo) or np.any(pose[:3] > hi):
            return None
        return self._joints.copy()


class MockHand(HandDriver):
    def __init__(self, config):
        self.config = config
        self._opening = 1.0
        self._connected = False

    def connect(self) -> None:
        self._connected = True
        logger.info("[MockHand] connected (no hardware)")

    def disconnect(self) -> None:
        self._connected = False

    def set_opening(self, opening: float) -> None:
        if not self._connected:
            raise RuntimeError("Hand driver used before connect()")
        self._opening = float(np.clip(opening, 0.0, 1.0))

    def get_opening(self) -> float:
        return self._opening

    def get_joint_angles(self) -> np.ndarray:
        closed = np.array(self.config.closed_pose, dtype=np.float64)
        opened = np.array(self.config.open_pose, dtype=np.float64)
        return closed + self._opening * (opened - closed)


class NullHand(HandDriver):
    """Accepts gripper commands and drives nothing.

    For running the arm before the hand is wired. It reports back exactly what
    was commanded, so the proprio channel keeps its shape and the recorded
    actions stay comparable with later runs that do have a hand.
    """

    def __init__(self, config):
        self.config = config
        self._opening = 1.0

    def connect(self) -> None:
        logger.warning("[NullHand] no hand attached; gripper commands are recorded, not executed")

    def disconnect(self) -> None:
        pass

    def set_opening(self, opening: float) -> None:
        self._opening = float(np.clip(opening, 0.0, 1.0))

    def get_opening(self) -> float:
        return self._opening

    def get_joint_angles(self) -> np.ndarray:
        closed = np.array(self.config.closed_pose, dtype=np.float64)
        opened = np.array(self.config.open_pose, dtype=np.float64)
        return closed + self._opening * (opened - closed)


class MockCameras:
    """Synthetic image source.

    Renders a moving marker whose position tracks the end-effector, so a replay
    video from a mock run still shows whether the control loop is doing
    something sensible.
    """

    def __init__(self, config, arm=None):
        self.config = config
        self.arm = arm
        self._connected = False

    def connect(self) -> None:
        self._connected = True
        logger.info(f"[MockCameras] connected: {self.config.names}")

    def disconnect(self) -> None:
        self._connected = False

    def peek(self):
        """Same shape as OpenCVCameras.peek(): newest frames, no bookkeeping,
        empty dict rather than an exception when there is nothing to show."""
        if not self._connected:
            return {}
        return self.read()

    def read(self):
        if not self._connected:
            raise RuntimeError("Cameras used before connect()")

        h, w = self.config.height, self.config.width
        images = {}
        pose = self.arm.get_ee_pose() if self.arm is not None else np.zeros(6)

        for i, name in enumerate(self.config.names):
            img = np.zeros((h, w, 3), dtype=np.uint8)
            # Distinct background per view so mixed-up camera wiring is obvious.
            img[:, :, i % 3] = 40

            # Map the x/y of the end-effector into pixel coordinates.
            cx = int(np.clip((pose[0] - 0.2) / 0.5, 0, 1) * (w - 40)) + 20
            cy = int(np.clip((pose[1] + 0.35) / 0.7, 0, 1) * (h - 40)) + 20
            img[max(0, cy - 12) : cy + 12, max(0, cx - 12) : cx + 12] = (230, 230, 60)

            # Orientation marker: a bar whose length follows the rotation angle.
            R = axis_angle_to_matrix(pose[3:6])
            bar = int(abs(R[0, 0]) * 60)
            img[h - 30 : h - 20, 20 : 20 + max(1, bar)] = (60, 200, 220)

            images[name] = img

        return images
