"""Minimal SO(3) helpers for the AR5 end-effector action space.

Poses are represented as a 6-vector ``[x, y, z, rx, ry, rz]`` where the
translation is in meters and the rotation is an axis-angle (rotation vector) in
radians.  This matches the LIBERO / OpenVLA convention that the VEL energy head
was trained on.

NumPy only -- no scipy -- so this module imports cleanly without the full
training stack, which keeps the mock-hardware tests fast.
"""

from typing import Tuple

import numpy as np

_EPS = 1e-8


def axis_angle_to_matrix(rotvec: np.ndarray) -> np.ndarray:
    """Rodrigues' formula: rotation vector (3,) -> rotation matrix (3, 3)."""
    rotvec = np.asarray(rotvec, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(rotvec))
    if theta < _EPS:
        return np.eye(3)
    k = rotvec / theta
    K = np.array(
        [
            [0.0, -k[2], k[1]],
            [k[2], 0.0, -k[0]],
            [-k[1], k[0], 0.0],
        ]
    )
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def matrix_to_axis_angle(R: np.ndarray) -> np.ndarray:
    """Rotation matrix (3, 3) -> rotation vector (3,), with theta in [0, pi]."""
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    # Clamp guards against tiny numerical drift pushing the argument out of domain.
    cos_theta = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = float(np.arccos(cos_theta))

    if theta < _EPS:
        return np.zeros(3)

    if np.pi - theta < 1e-5:
        # Near pi the skew-symmetric part vanishes; recover the axis from the
        # diagonal of (R + I) instead, which stays well-conditioned there.
        A = (R + np.eye(3)) / 2.0
        axis = np.sqrt(np.clip(np.diag(A), 0.0, None))
        # Fix the signs using the largest component as the reference.
        i = int(np.argmax(axis))
        if axis[i] < _EPS:
            return np.zeros(3)
        for j in range(3):
            if j != i:
                axis[j] = A[i, j] / axis[i]
        axis = axis / np.linalg.norm(axis)
        return axis * theta

    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return axis / (2.0 * np.sin(theta)) * theta


def rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    """ROKAE RPY [Rx, Ry, Rz] -> rotation matrix.

    The convention is Rz(rz) @ Ry(ry) @ Rx(rx) -- ZYX intrinsic, equivalently
    XYZ fixed-angle. Verified against the SDK's own
    ``utility.postureToTransArray`` over random angles to 3e-16.

    Note the SDK also exports ``utility.eulerToMatrix``, which uses a *different*
    convention and does not agree with the poses ``cartPosture`` returns. Do not
    mix the two.
    """
    rx, ry, rz = np.asarray(rpy, dtype=np.float64).reshape(3)
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def matrix_to_rpy(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> ROKAE RPY [Rx, Ry, Rz]. Inverse of :func:`rpy_to_matrix`."""
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)

    sy = -R[2, 0]
    # Gimbal lock: ry = +-pi/2 leaves only (rx - rz) or (rx + rz) determined.
    # Pin rz to 0 and fold the whole rotation into rx, which round-trips.
    if abs(sy) > 1.0 - 1e-9:
        ry = np.pi / 2 if sy > 0 else -np.pi / 2
        rx = np.arctan2(-R[1, 2], R[1, 1])
        rz = 0.0
    else:
        ry = np.arcsin(np.clip(sy, -1.0, 1.0))
        rx = np.arctan2(R[2, 1], R[2, 2])
        rz = np.arctan2(R[1, 0], R[0, 0])
    return np.array([rx, ry, rz], dtype=np.float64)


def rpy_pose_to_internal(trans, rpy) -> np.ndarray:
    """(translation, ROKAE RPY) -> our ``[x, y, z, rx, ry, rz]`` axis-angle pose."""
    out = np.empty(6, dtype=np.float64)
    out[:3] = np.asarray(trans, dtype=np.float64).reshape(3)
    out[3:6] = matrix_to_axis_angle(rpy_to_matrix(rpy))
    return out


def internal_pose_to_rpy(pose) -> tuple:
    """Our axis-angle pose -> (translation, ROKAE RPY), both length 3."""
    pose = np.asarray(pose, dtype=np.float64).reshape(6)
    return pose[:3].copy(), matrix_to_rpy(axis_angle_to_matrix(pose[3:6]))


def compose_pose(pose: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """Apply a pose delta in the base frame.

    Translation adds; rotation composes as ``R_delta @ R_current`` so the delta
    is interpreted in the fixed base frame, matching how delta end-effector
    actions are defined in LIBERO.
    """
    pose = np.asarray(pose, dtype=np.float64).reshape(6)
    delta = np.asarray(delta, dtype=np.float64).reshape(6)

    out = np.empty(6, dtype=np.float64)
    out[:3] = pose[:3] + delta[:3]

    R_cur = axis_angle_to_matrix(pose[3:6])
    R_delta = axis_angle_to_matrix(delta[3:6])
    out[3:6] = matrix_to_axis_angle(R_delta @ R_cur)
    return out


def pose_delta(from_pose: np.ndarray, to_pose: np.ndarray) -> np.ndarray:
    """Inverse of :func:`compose_pose`: the delta taking ``from_pose`` to ``to_pose``."""
    from_pose = np.asarray(from_pose, dtype=np.float64).reshape(6)
    to_pose = np.asarray(to_pose, dtype=np.float64).reshape(6)

    out = np.empty(6, dtype=np.float64)
    out[:3] = to_pose[:3] - from_pose[:3]

    R_from = axis_angle_to_matrix(from_pose[3:6])
    R_to = axis_angle_to_matrix(to_pose[3:6])
    out[3:6] = matrix_to_axis_angle(R_to @ R_from.T)
    return out


def clamp_delta(delta: np.ndarray, max_translation: float, max_rotation: float) -> Tuple[np.ndarray, bool]:
    """Scale a pose delta down until it respects the per-step safety limits.

    Translation and rotation are scaled by a single common factor so the motion
    direction is preserved -- clipping the components independently would bend
    the commanded path.

    Returns the clamped delta and whether any clamping occurred.
    """
    delta = np.asarray(delta, dtype=np.float64).reshape(6).copy()

    trans_norm = float(np.linalg.norm(delta[:3]))
    rot_norm = float(np.linalg.norm(delta[3:6]))

    scale = 1.0
    if max_translation > 0 and trans_norm > max_translation:
        scale = min(scale, max_translation / trans_norm)
    if max_rotation > 0 and rot_norm > max_rotation:
        scale = min(scale, max_rotation / rot_norm)

    if scale < 1.0:
        return delta * scale, True
    return delta, False
