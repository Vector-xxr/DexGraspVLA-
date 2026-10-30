"""Hardware interfaces the AR5 environment programs against.

Keeping the environment behind these two small protocols means the control
logic, the safety limits and the eval client are all exercisable with mock
backends before any hardware is connected.
"""

from abc import ABC, abstractmethod

import numpy as np


class ArmDriver(ABC):
    """A 6-DoF arm addressed in Cartesian space.

    Poses are ``[x, y, z, rx, ry, rz]``: meters and axis-angle radians in the
    robot base frame.
    """

    @abstractmethod
    def connect(self) -> None:
        """Open the connection and put the arm in a controllable state."""

    @abstractmethod
    def disconnect(self) -> None:
        """Release the connection. Must be safe to call twice."""

    @abstractmethod
    def get_joint_positions(self) -> np.ndarray:
        """Current joint angles, shape (6,), radians."""

    @abstractmethod
    def get_ee_pose(self) -> np.ndarray:
        """Current end-effector pose, shape (6,)."""

    @abstractmethod
    def servo_to_pose(self, pose: np.ndarray, wait: bool = True) -> None:
        """Command a new target pose. Non-blocking in real-time mode."""

    @abstractmethod
    def move_to_joints(self, joints: np.ndarray, blocking: bool = True) -> None:
        """Move to a joint configuration. Used for homing, not for policy steps."""

    def stop(self) -> None:
        """Halt motion immediately. Overridden where the SDK supports it."""


class HandDriver(ABC):
    """A gripper or dexterous hand driven by a single opening scalar.

    ``opening`` is in [0, 1] with 0 = closed and 1 = open, matching the gripper
    encoding that `k_step_energy_correction_seq` expects in action dimension 7.
    """

    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    @abstractmethod
    def set_opening(self, opening: float) -> None:
        """Command the hand. ``opening`` is clamped to [0, 1] by implementations."""

    @abstractmethod
    def get_opening(self) -> float:
        """Current opening in [0, 1], derived from the measured joint angles."""

    @abstractmethod
    def get_joint_angles(self) -> np.ndarray:
        """Raw per-actuator readback, shape (num_joints,)."""
