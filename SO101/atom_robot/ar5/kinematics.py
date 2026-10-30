"""Local forward and inverse kinematics for the AR5, from the controller's DH table.

The SDK's own kinematics are unavailable on this arm -- ``robot.model()`` raises
"xMate模型库暂不支持的机型" -- and the one solver that does work, ``checkPath``,
is a 42.5 ms network round trip that holds the GIL for most of that time. A 1 kHz
command stream cannot share a process with it: measured, a background thread gets
only 13.6% of its normal timeslice while ``checkPath`` runs, which would blank
roughly 40 of every 48 milliseconds of the stream.

``getRobotCfg_DHparam`` sidesteps both problems by handing over the arm's
*calibrated* DH table, from which the kinematics can be rebuilt locally in NumPy:
microseconds per call, no network, no GIL held. Verified against the controller's
own ``cartPosture`` at the live pose: 0.32 mm and 0.053 degrees.

Convention: modified (Craig) DH. Each row contributes

    T_i = Rx(alpha_i) . Tx(a_i) . Rz(theta_i + theta_offset_i) . Tz(d_i)

Standard DH was tested against the same pose and missed by 887 mm, so the choice
is not a matter of taste here.

The table has one more row than the arm has joints; the trailing rows carry the
fixed flange transform and take no joint variable.
"""

import logging
from typing import Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _rot_x(t: float) -> np.ndarray:
    c, s = np.cos(t), np.sin(t)
    return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]], dtype=np.float64)


def _rot_z(t: float) -> np.ndarray:
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)


def _trans_x(a: float) -> np.ndarray:
    T = np.eye(4)
    T[0, 3] = a
    return T


def _trans_z(d: float) -> np.ndarray:
    T = np.eye(4)
    T[2, 3] = d
    return T


class AR5Kinematics:
    """Modified-DH kinematics for a redundant 7-axis arm.

    Angles in radians, lengths in metres. Poses are 4x4 homogeneous transforms
    in the robot base frame.
    """

    def __init__(
        self,
        dh_rows: Sequence[Sequence[float]],
        num_joints: int = 7,
        joint_min: Optional[Sequence[float]] = None,
        joint_max: Optional[Sequence[float]] = None,
    ):
        """dh_rows: one [alpha_deg, a_mm, d_mm, theta_deg] per row, as the SDK returns."""
        dh = np.asarray(dh_rows, dtype=np.float64).reshape(-1, 4)
        self.alpha = np.deg2rad(dh[:, 0])
        self.a = dh[:, 1] / 1000.0
        self.d = dh[:, 2] / 1000.0
        self.theta_offset = np.deg2rad(dh[:, 3])
        self.n_rows = dh.shape[0]
        self.num_joints = num_joints

        if self.n_rows < num_joints:
            raise ValueError(f"DH table has {self.n_rows} rows, fewer than {num_joints} joints")

        self.joint_min = np.asarray(joint_min, dtype=np.float64) if joint_min is not None else np.full(num_joints, -np.pi)
        self.joint_max = np.asarray(joint_max, dtype=np.float64) if joint_max is not None else np.full(num_joints, np.pi)
        self.joint_mid = (self.joint_min + self.joint_max) / 2.0
        self.joint_range = np.maximum(self.joint_max - self.joint_min, 1e-9)
        # Extra centering pull near a joint stop, cubic in normalised offset.
        # Off by default: on this arm the failures that looked like joints
        # walking into their limits were actually the manipulability decaying
        # toward a singularity (measured 0.028 -> 0.019 over 8 steps, then no
        # convergence), which a centering term cannot fix. Raising it fought
        # convergence without helping, so it stays disabled until something
        # measured justifies it.
        self.limit_urgency = 0.0

    # ------------------------------------------------------------ construction

    @classmethod
    def from_robot(cls, arm) -> "AR5Kinematics":
        """Read the calibrated DH table off the controller.

        ``get_nominal=False`` asks for the post-calibration values; on this arm
        they differ from nominal by up to 0.4, which at the tool is millimetres.
        """
        dh = arm._call(arm._robot.getRobotCfg_DHparam, False)
        config = arm.config
        return cls(
            np.asarray(dh, dtype=np.float64).reshape(-1, 4),
            num_joints=config.num_joints,
            joint_min=getattr(config, "joint_min", None),
            joint_max=getattr(config, "joint_max", None),
        )

    # ------------------------------------------------------------------- forward

    def _chain(self, q: np.ndarray):
        """Frames along the chain, plus each joint's axis and origin in base coordinates.

        Joint i turns about the z axis of the frame reached *after* that row's
        Rx and Tx but before its Rz -- which is where the geometric Jacobian
        needs its axis and origin.
        """
        q = np.asarray(q, dtype=np.float64).reshape(-1)
        T = np.eye(4)
        axes = np.zeros((self.num_joints, 3))
        origins = np.zeros((self.num_joints, 3))

        for i in range(self.n_rows):
            pre = T @ _rot_x(self.alpha[i]) @ _trans_x(self.a[i])
            if i < self.num_joints:
                axes[i] = pre[:3, 2]
                origins[i] = pre[:3, 3]
                theta = self.theta_offset[i] + q[i]
            else:
                theta = self.theta_offset[i]
            T = pre @ _rot_z(theta) @ _trans_z(self.d[i])

        return T, axes, origins

    def fk(self, q: np.ndarray) -> np.ndarray:
        """Joint angles -> 4x4 flange pose in the base frame."""
        return self._chain(q)[0]

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        """Geometric Jacobian, 6 x num_joints: [linear; angular] of the flange."""
        T, axes, origins = self._chain(q)
        p_end = T[:3, 3]
        J = np.zeros((6, self.num_joints))
        for i in range(self.num_joints):
            J[:3, i] = np.cross(axes[i], p_end - origins[i])
            J[3:, i] = axes[i]
        return J

    # -------------------------------------------------------------------- error

    @staticmethod
    def pose_error(T_current: np.ndarray, T_target: np.ndarray) -> np.ndarray:
        """6-vector [translation; rotation] taking current to target.

        The rotation part is the axis-angle of ``R_target R_current^T``, which
        is the angular velocity that would close the gap in unit time.
        """
        e = np.empty(6)
        e[:3] = T_target[:3, 3] - T_current[:3, 3]

        R = T_target[:3, :3] @ T_current[:3, :3].T
        angle = np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
        if angle < 1e-9:
            e[3:] = 0.0
        elif np.pi - angle < 1e-6:
            # Near pi the skew part vanishes; recover the axis from R + I.
            A = (R + np.eye(3)) / 2.0
            axis = np.sqrt(np.clip(np.diag(A), 0.0, None))
            k = int(np.argmax(axis))
            if axis[k] < 1e-9:
                e[3:] = 0.0
            else:
                for j in range(3):
                    if j != k:
                        axis[j] = A[k, j] / axis[k]
                e[3:] = axis / np.linalg.norm(axis) * angle
        else:
            axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
            e[3:] = axis / (2.0 * np.sin(angle)) * angle
        return e

    # ------------------------------------------------------------------ inverse

    def ik(
        self,
        T_target: np.ndarray,
        q_seed: np.ndarray,
        max_iters: int = 100,
        pos_tol: float = 1e-4,
        rot_tol: float = 1e-3,
        damping: float = 0.02,
        max_step: float = 0.15,
        null_space_gain: float = 0.05,
    ) -> Tuple[Optional[np.ndarray], dict]:
        """Damped least squares IK, seeded from the current posture.

        Seeding from where the arm already is does the work that ``elbow`` does
        for the controller: seven joints are redundant, so a pose has a family
        of solutions, and starting from the current one keeps the arm in the
        same branch instead of letting the elbow swing across.

        Damping keeps the step finite near singularities, where an undamped
        pseudo-inverse would demand enormous joint rates for a small Cartesian
        motion. The redundancy left over after tracking is spent pushing joints
        toward the middle of their travel, which is what keeps j5 and j6 -- only
        +-0.8727 rad wide -- away from their stops.

        `null_space_gain` is deliberately small. Damping makes the null-space
        projector approximate, so the centering term leaks into the tracked
        pose; measured, a gain of 0.3 fights convergence badly enough to drop
        the success rate from 100/100 to 10/100, while 0.05 costs nothing.

        Returns (q, info). ``q`` is None when the target was not reached; info
        always carries ``reason``, ``iters``, ``pos_error`` and ``rot_error``.
        """
        q = np.asarray(q_seed, dtype=np.float64).reshape(self.num_joints).copy()
        lo = self.joint_min
        hi = self.joint_max

        info = {"reason": "", "iters": 0, "pos_error": np.inf, "rot_error": np.inf}

        for it in range(max_iters):
            T = self.fk(q)
            e = self.pose_error(T, T_target)
            pos_err = float(np.linalg.norm(e[:3]))
            rot_err = float(np.linalg.norm(e[3:]))
            info.update(iters=it + 1, pos_error=pos_err, rot_error=rot_err)

            if pos_err < pos_tol and rot_err < rot_tol:
                if np.any(q < lo) or np.any(q > hi):
                    # Redundancy means a pose usually has a solution inside the
                    # limits even when this one landed outside, so try pushing
                    # into range and re-checking rather than giving up. Only a
                    # target that cannot be reached from inside the box fails.
                    q_in = np.clip(q, lo, hi)
                    e_in = self.pose_error(self.fk(q_in), T_target)
                    if np.linalg.norm(e_in[:3]) < pos_tol * 3 and np.linalg.norm(e_in[3:]) < rot_tol * 3:
                        info["reason"] = "converged after clamping into limits"
                        return q_in, info
                    info["reason"] = (
                        f"converged outside joint limits: "
                        f"{self.limit_report(q)}"
                    )
                    return None, info
                info["reason"] = "converged"
                return q, info

            J = self.jacobian(q)
            JJt = J @ J.T + (damping ** 2) * np.eye(6)
            try:
                primary = J.T @ np.linalg.solve(JJt, e)
            except np.linalg.LinAlgError:
                info["reason"] = "singular Jacobian"
                return None, info

            # Null-space: drift toward mid-range without disturbing the tool.
            #
            # Weighted by how close each joint is to its own stop, not by a flat
            # gain. j5 and j6 travel only +-0.8727 rad, 3.5x narrower than the
            # rest, so a uniform pull leaves them to be walked into their limits
            # one seeded solve at a time -- measured on hardware as j6 arriving
            # 0.02 deg from its stop, after which every subsequent target became
            # unsolvable and the arm froze while every counter read healthy.
            if null_space_gain > 0:
                J_pinv = J.T @ np.linalg.inv(JJt)
                null = np.eye(self.num_joints) - J_pinv @ J
                # Normalised distance from centre: 0 at mid-travel, 1 at a stop.
                offset = 2.0 * (q - self.joint_mid) / self.joint_range
                # Halved so `null_space_gain` keeps the magnitude it was tuned
                # at, when the term was (mid - q) / range.
                pull = -0.5 * offset
                # Cubic so the pull is negligible mid-range and strong near a stop.
                urgency = np.abs(offset) ** 3
                centering = null_space_gain * pull * (1.0 + self.limit_urgency * urgency)
                dq = primary + null @ centering
            else:
                dq = primary

            # A bounded step keeps a bad Jacobian from throwing the seed away.
            norm = np.linalg.norm(dq)
            if norm > max_step:
                dq *= max_step / norm

            q = np.clip(q + dq, lo, hi)

        info["reason"] = f"did not converge in {max_iters} iterations"
        return None, info

    # ------------------------------------------------------------------ helpers

    def near_singularity(self, q: np.ndarray, threshold: float = 0.01) -> bool:
        """Whether the Jacobian is too ill-conditioned to solve reliably.

        Damped least squares degrades gracefully rather than blowing up, so the
        symptom of an approaching singularity is not a wild solution but a
        solver that stops converging -- which upstream sees as an unexplained
        run of refusals.
        """
        return self.manipulability(q) < threshold

    def within_limits(self, q: np.ndarray, margin: float = 0.0) -> bool:
        q = np.asarray(q, dtype=np.float64).reshape(-1)
        return bool(np.all(q >= self.joint_min + margin) and np.all(q <= self.joint_max - margin))

    def limit_report(self, q: np.ndarray, margin: float = 0.0):
        """Which joints are outside the (optionally shrunk) limits, and by how much."""
        q = np.asarray(q, dtype=np.float64).reshape(-1)
        out = []
        for i, v in enumerate(q):
            lo, hi = self.joint_min[i] + margin, self.joint_max[i] - margin
            if v < lo:
                out.append((i, float(v), float(lo - v)))
            elif v > hi:
                out.append((i, float(v), float(v - hi)))
        return out

    def manipulability(self, q: np.ndarray) -> float:
        """Yoshikawa's measure. Near zero means a singularity is close."""
        J = self.jacobian(q)
        return float(np.sqrt(max(0.0, np.linalg.det(J @ J.T))))
