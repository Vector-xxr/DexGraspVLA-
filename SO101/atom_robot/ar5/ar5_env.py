"""Real-robot environment for a ROKAE AR5 with an Inspire dexterous hand.

Mirrors the interface of ``experiments/robot/aloha/real_env.py`` so the eval
client stays close to ``run_aloha_eval.py``:

    env.reset()                 -> AR5Observation
    env.step(action)            -> AR5Observation
    env.get_observation(t=...)  -> AR5Observation
    env.get_qpos()              -> np.ndarray

Action layout, 7 dimensions, matching LIBERO so the VEL energy head applies
unchanged:

    [0:3]  end-effector position delta, meters, base frame
    [3:6]  end-effector orientation delta, axis-angle radians, base frame
    [6]    hand opening in [0, 1]; 0 = closed, 1 = open

That last convention is not cosmetic. ``k_step_energy_correction_seq`` rounds
dimension 7 to {0, 1} before scoring, so demonstrations must be recorded with
the same encoding or the energy head learns an inverted gripper semantics.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from .config import AR5Config
from .transforms import clamp_delta, compose_pose

logger = logging.getLogger(__name__)

ACTION_DIM = 7
GRIPPER_IDX = 6


@dataclass
class AR5Observation:
    """Mirrors the ``dm_env.TimeStep`` shape the ALOHA client expects."""

    observation: Dict[str, Any] = field(default_factory=dict)
    reward: float = 0.0
    step_index: int = 0


class SafetyViolation(RuntimeError):
    """Raised when a command cannot be made safe by clamping alone."""


class AR5Env:
    def __init__(self, config: Optional[AR5Config] = None):
        self.config = config or AR5Config()

        self.arm = None
        self.hand = None
        self.cameras = None

        self._target_pose: Optional[np.ndarray] = None
        self._joint_target: Optional[np.ndarray] = None
        self._connected = False

        # Counters surfaced after a rollout so clamping is never silent.
        self.stats = {"steps": 0, "delta_clamped": 0, "workspace_clamped": 0, "unreachable": 0,
                      "resynced": 0, "leashed": 0, "motion_errors": 0,
                      "joint_clamped": 0, "floor_blocked": 0, "joint_rate_limited": 0}

    # ----------------------------------------------------------------- setup

    def connect(self) -> None:
        """Instantiate and connect the drivers. Mock backends need no hardware."""
        from .end_effector import make_real_hand
        if self.config.mock:
            from .drivers.mock import MockArm, MockCameras, MockHand

            self.arm = MockArm(self.config.arm)
            if self.config.real_cameras_in_mock and self.config.cameras.sources:
                from .drivers.cameras import OpenCVCameras

                self.cameras = OpenCVCameras(self.config.cameras)
                logger.warning("AR5Env: MOCK arm reading REAL cameras %s",
                               list(self.config.cameras.sources))
            else:
                self.cameras = MockCameras(self.config.cameras, arm=self.arm)
            # A simulated arm with the real hand: the rung between "hand alone"
            # and "both for real". It exercises the whole leader -> action ->
            # hand path, including the gripper channel, with nothing that can
            # collide. Opt-in only: plain `mock` must never move real fingers.
            if self.config.real_hand_in_mock and self.config.use_hand:
                self.hand = make_real_hand(self.config)
                logger.warning("AR5Env: MOCK arm driving REAL %s", self.config.end_effector)
            else:
                self.hand = MockHand(self.config.hand)
                logger.warning("AR5Env running in MOCK mode -- no hardware is being driven")
        else:
            from .drivers.cameras import OpenCVCameras
            from .drivers.rokae_arm import RokaeAR5

            self.arm = RokaeAR5(self.config.arm)
            self.cameras = OpenCVCameras(self.config.cameras)

            if self.config.use_hand:
                self.hand = make_real_hand(self.config)
            else:
                from .drivers.mock import NullHand

                self.hand = NullHand(self.config.hand)

        # Bring the devices up in order, and unwind whatever is already up if a
        # later one fails. Without this a missing hand or camera leaves the arm
        # powered with its 1 kHz real-time stream running and no owner: the
        # process dies on the traceback, nothing calls disconnect(), and the next
        # run inherits a controller that is still in RtCommandMode.
        started = []
        try:
            # 夹爪（HKVR、OmniPicker）先确认连通，再给臂上电，避免夹爪未接时启动机械臂。
            gripper_first = self.config.end_effector in ("hkvr", "omnipicker")
            devices = ((self.hand, self.arm, self.cameras) if gripper_first
                       else (self.arm, self.hand, self.cameras))
            for device in devices:
                device.connect()
                started.append(device)
        except BaseException:
            for device in reversed(started):
                try:
                    device.disconnect()
                except Exception as exc:  # noqa: BLE001
                    logger.error(f"While unwinding {type(device).__name__}: {exc}")
            raise

        self._connected = True

    def close(self) -> None:
        try:
            self.stop()
        except Exception as exc:
            logger.error("Error stopping arm during close: %s", exc)
        for device in (self.cameras, self.hand, self.arm):
            if device is not None:
                try:
                    device.disconnect()
                except Exception as exc:  # noqa: BLE001
                    logger.error(f"Error while disconnecting {type(device).__name__}: {exc}")
        self._connected = False

    def __enter__(self):
        if not self._connected:
            self.connect()
        return self

    def __exit__(self, *exc_info):
        self.close()

    def _require_connection(self) -> None:
        if not self._connected:
            raise RuntimeError("AR5Env used before connect()")
        if hasattr(self.hand, "check_health"):
            try:
                self.hand.check_health()
            except Exception:
                self.stop()
                raise

    # ----------------------------------------------------------- observations

    def get_qpos(self) -> np.ndarray:
        """Proprio vector: the arm joint angles plus the hand opening scalar.

        Width is ArmConfig.num_joints + 1, which must equal PROPRIO_DIM.
        """
        self._require_connection()
        return np.concatenate([self.arm.get_joint_positions(), [self.hand.get_opening()]]).astype(np.float32)

    def read_arm_stamped(self):
        """(时刻, 7 个关节角, 末端位姿)：一次状态刷新里读的，时刻取读的前后中点。OmniPicker 采集每圈调。"""
        self._require_connection()
        asked = time.perf_counter()
        reader = getattr(self.arm, "read_joints_and_pose", None)
        if reader is not None:
            q, ee = reader()
        else:
            q, ee = self.arm.get_joint_positions(), self.arm.get_ee_pose()
        return (asked + time.perf_counter()) / 2, np.asarray(q, np.float64), np.asarray(ee, np.float64)

    def get_observation(self, t: int = 0) -> AR5Observation:
        """一次观测。`stamps` 记每个来源的数据是什么时候拍 / 读的（time.perf_counter 的钟）：

        * 每路相机：这张图的拍摄时刻（V4L2 是内核时刻，OBSENSOR 只有程序拿到帧的时刻，见 cameras.py）；
        * ``arm``：读关节角那次 SDK 调用前后的中点（末端位姿紧接着再问一次控制器，晚 1~2 ms）；
        * ``gripper``：夹爪开合那个读数是什么时候读的（OmniPicker 有；别的末端退回到这里取值的时刻）。

        图是取帧线程里「最新的一张」，夹爪是它读写线程里上一次的读数，所以都比臂的读数早 ——
        collect 逐帧记进 npz，看同步就是看这几个时刻差多少。
        """
        self._require_connection()
        images = self.cameras.read()
        stamps = dict(getattr(self.cameras, "last_stamps", None) or {})
        asked = time.perf_counter()
        joints = self.arm.get_joint_positions()
        stamps["arm"] = (asked + time.perf_counter()) / 2
        stamped = getattr(self.hand, "get_opening_stamped", None)
        if stamped is not None:
            opening, stamps["gripper"] = stamped()
        else:
            opening, stamps["gripper"] = self.hand.get_opening(), time.perf_counter()
        return AR5Observation(
            observation={
                "images": images,
                "qpos": np.concatenate([joints, [opening]]).astype(np.float32),
                "ee_pose": self.arm.get_ee_pose().astype(np.float32),
                "elbow": np.float32(self.arm.get_elbow()),
                "stamps": stamps,
            },
            step_index=t,
        )

    # ---------------------------------------------------------------- control

    def _decode_action(self, action) -> tuple:
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if action.shape[0] != ACTION_DIM:
            raise ValueError(f"Expected a {ACTION_DIM}-dim action, got shape {action.shape}")
        if not np.all(np.isfinite(action)):
            raise SafetyViolation(f"Action contains non-finite values: {action}")
        return action[:GRIPPER_IDX], float(action[GRIPPER_IDX])

    def _clamp_into_workspace(self, pose: np.ndarray) -> np.ndarray:
        lo = np.array(self.config.arm.workspace_min, dtype=np.float64)
        hi = np.array(self.config.arm.workspace_max, dtype=np.float64)
        clamped = pose.copy()
        clamped[:3] = np.clip(pose[:3], lo, hi)
        if not np.allclose(clamped[:3], pose[:3]):
            self.stats["workspace_clamped"] += 1
            logger.warning(f"Target {pose[:3]} clipped to workspace -> {clamped[:3]}")
        return clamped

    def _base_pose(self, measured: Optional[np.ndarray] = None) -> np.ndarray:
        """The pose to integrate the next delta onto.

        Normally the last commanded setpoint: chaining measurements instead
        would fold tracking lag into the target and leave the arm creeping
        behind the policy. But a setpoint the arm never reached -- after a
        refused step, a collision stop, or a stall -- would otherwise run away
        from reality, so it is re-anchored once the gap gets large.
        """
        if measured is None:
            measured = self.arm.get_ee_pose()
        if self._target_pose is None:
            return measured

        drift = float(np.linalg.norm(self._target_pose[:3] - measured[:3]))
        if drift > self.config.resync_threshold:
            self.stats["resynced"] += 1
            logger.warning(
                f"Commanded setpoint is {drift * 1000:.1f} mm from the measured pose; "
                "re-anchoring to the arm. The arm is not keeping up with the policy."
            )
            return measured
        return self._target_pose

    def workspace_clearance(self) -> dict:
        """Distance from the tool to each face of the workspace box, in metres."""
        pose = self.arm.get_ee_pose()
        lo = np.array(self.config.arm.workspace_min, dtype=np.float64)
        hi = np.array(self.config.arm.workspace_max, dtype=np.float64)
        return {
            "position": pose[:3],
            "to_min": pose[:3] - lo,
            "to_max": hi - pose[:3],
            "centre": (lo + hi) / 2.0,
        }

    def pinned_axes(self, margin: float = 0.02):
        """Faces the tool is pressed against, as (axis, 'min'|'max', clearance).

        Motion toward a face the arm is already on is silently eaten by the
        workspace clamp, so half the operator's input can do nothing while
        everything reports healthy. Worth checking before a session rather than
        discovering it by pushing and seeing no response.
        """
        c = self.workspace_clearance()
        pinned = []
        for i, axis in enumerate("xyz"):
            if c["to_min"][i] < margin:
                pinned.append((axis, "min", float(c["to_min"][i])))
            if c["to_max"][i] < margin:
                pinned.append((axis, "max", float(c["to_max"][i])))
        return pinned

    def home_joints(self, blocking: bool = True, timeout_s: float = 30.0) -> bool:
        """Move to the configured home posture in JOINT space.

        Joint space on purpose: it needs no inverse kinematics, so it works from
        anywhere -- including the postures where IK has stopped converging,
        which is exactly when homing is needed. Recentring the tool in the
        workspace box does not help there, because a box-centred tool can still
        sit on a singularity or with a wrist joint against its stop.

        Which mechanism carries the move depends on the control mode. The queued
        MoveAbsJ command is rejected outright while the controller is in
        RtCommandMode ("运动控制模式错误", ec 262), and switching modes would
        mean tearing down the live stream, so under real-time control the move
        goes through the stream itself as a series of joint goals -- bounded by
        the same publish gate as everything else.
        """
        import time as _time

        return self.goto_joints(self.config.arm.home_joints,
                                blocking=blocking, timeout_s=timeout_s)

    def goto_joints(self, target, blocking: bool = True, timeout_s: float = 30.0) -> bool:
        """Move to an arbitrary joint posture, same mechanism as homing.

        Split out of `home_joints` so anything that needs a controlled
        joint-space move -- replaying a recorded posture to line a camera up
        against it, for instance -- gets the same jump-gated interpolation and
        the same final hold, rather than a second copy that drifts from it.
        """
        import time as _time

        self._require_connection()
        target = np.array(target, dtype=np.float64)
        if target.shape != (self.config.arm.num_joints,):
            raise ValueError(
                f"expected {self.config.arm.num_joints} joint angles, got {target.shape}"
            )
        lo = np.asarray(self.config.arm.joint_min) + self.config.arm.joint_limit_margin
        hi = np.asarray(self.config.arm.joint_max) - self.config.arm.joint_limit_margin
        if np.any(target < lo) or np.any(target > hi):
            bad = [i for i in range(len(target)) if target[i] < lo[i] or target[i] > hi[i]]
            raise SafetyViolation(
                f"target posture is outside the soft limits at joint(s) {bad}: "
                f"{np.rad2deg(target).round(2).tolist()} deg"
            )
        low = self._lowest_point(target)
        if low is not None and low < self.config.arm.joint_floor_z:
            raise SafetyViolation(
                f"target posture puts the lowest point at z={low:.4f} m, under the "
                f"floor at {self.config.arm.joint_floor_z} m"
            )

        clear = getattr(self.arm, "clear_latch", None)
        if clear is not None:
            clear()

        reached = True
        if getattr(self.arm, "_rt_active", False):
            # Step size stays under the jump gate so no goal is ever refused.
            max_step = self.config.arm.joint_jump_max * 0.5
            deadline = _time.time() + timeout_s
            while _time.time() < deadline:
                current = self.arm.get_joint_positions()
                error = target - current
                if np.max(np.abs(error)) < 0.005:
                    break
                scale = min(1.0, max_step / max(np.max(np.abs(error)), 1e-9))
                self.arm.servo_to_joints(current + error * scale)
                _time.sleep(0.05)
            else:
                reached = False
            # Hold the final goal so the sender does not brake on staleness --
            # until the tracker is actually at rest, not for a fixed count.
            for _ in range(20):
                self.arm.servo_to_joints(target)
                _time.sleep(0.02)
            hold = getattr(self.arm, "hold_until_rest", None)
            if hold is not None:
                hold(timeout_s=5.0)
        else:
            self.arm.move_to_joints(target, blocking=blocking)

        self._target_pose = self.arm.get_ee_pose()
        # **锚点必须跟着走。** 这个函数绕过 step_joints 直接指挥臂，而
        # `_joint_target`（限速器的锚点）只有 step_joints 会写 —— 不在这里同步，
        # 下一次遥操就会从一个几十度之外的锚点起步，然后被跳变门一路拒死。
        # 见 step_joints 里那段注释。
        self._joint_target = self._commanded_joints()
        return reached

    def posture_report(self) -> dict:
        """Joint headroom and how close the solver is to giving up."""
        q = self.arm.get_joint_positions()
        cfg = self.config.arm
        lo = np.array(cfg.joint_min) + cfg.joint_limit_margin
        hi = np.array(cfg.joint_max) - cfg.joint_limit_margin
        mid = (lo + hi) / 2.0
        span = np.maximum(hi - lo, 1e-9)
        usage = np.abs(2.0 * (q - mid) / span)

        kin = getattr(self.arm, "kinematics", None)
        manip = float(kin.manipulability(q)) if kin is not None else float("nan")
        return {
            "joint_usage": usage,
            "worst_joint": int(np.argmax(usage)),
            "worst_usage": float(usage.max()),
            "manipulability": manip,
            "near_singularity": bool(kin.near_singularity(q)) if kin is not None else False,
        }

    def move_to_workspace_centre(self, step: float = 0.004, timeout_s: float = 60.0) -> bool:
        """Walk the tool to the middle of the box, one bounded step at a time.

        Uses the ordinary step path so every clamp and gate still applies; a
        recentring move is not a reason to bypass the safety envelope.
        """
        import time as _time

        centre = self.workspace_clearance()["centre"]
        deadline = _time.time() + timeout_s
        reached = False
        while _time.time() < deadline:
            error = centre - self.arm.get_ee_pose()[:3]
            distance = float(np.linalg.norm(error))
            if distance < 0.005:
                reached = True
                break
            move = error / distance * min(step, distance)
            self.step(np.concatenate([move, np.zeros(3), [self.hand.get_opening()]]))
            _time.sleep(0.05)
        # Do not go quiet while the tracker is still settling: the sender would
        # read that as a dead publisher and latch (see hold_until_rest).
        hold = getattr(self.arm, "hold_until_rest", None)
        if hold is not None:
            hold(timeout_s=5.0)
        return reached

    def _leash(self, target: np.ndarray, measured: Optional[np.ndarray] = None) -> np.ndarray:
        """Keep the commanded setpoint within reach of where the arm actually is.

        The arm has a joint speed cap, so a fast operator can command faster
        than it can move and the setpoint runs ahead. Left alone that lead grows
        until the hard resync fires and throws it away, which feels like the arm
        ignoring the input. Pulling the target back along the line to the
        measured pose bounds the lead while preserving the commanded direction,
        so the arm keeps moving the way the operator asked -- just no further
        ahead of itself than it can recover from.
        """
        if measured is None:
            measured = self.arm.get_ee_pose()
        lead = target[:3] - measured[:3]
        distance = float(np.linalg.norm(lead))
        limit = self.config.max_lead
        if distance <= limit or distance < 1e-12:
            return target

        leashed = target.copy()
        leashed[:3] = measured[:3] + lead * (limit / distance)
        self.stats["leashed"] += 1
        return leashed

    def step(self, action) -> AR5Observation:
        """Apply one 7-dim action and return the resulting observation.

        Safety limits are enforced here rather than relying on the policy or on
        VEL's ``clip_frac``: Proposition 2 guarantees the correction lowers
        energy, not that the resulting motion is safe.

        A step the controller cannot reach holds position instead of raising.
        One bad action should not end a rollout -- but it is counted, and the
        count is reported, so it is never silent.
        """
        self._require_connection()
        delta, gripper = self._decode_action(action)

        delta, was_clamped = clamp_delta(
            delta,
            max_translation=self.config.arm.max_step_translation,
            max_rotation=self.config.arm.max_step_rotation,
        )
        if was_clamped:
            self.stats["delta_clamped"] += 1

        # One pose read per step, shared by the drift check and the leash.
        # Each SDK round trip holds the GIL for about half a millisecond, and the
        # 1 kHz sender needs it a thousand times a second: six reads per step
        # was showing up as 2% dropped ticks and visible jitter.
        measured = self.arm.get_ee_pose()
        target = self._clamp_into_workspace(compose_pose(self._base_pose(measured), delta))
        target = self._leash(target, measured)

        # Two very different paths. The real-time one resolves the target with
        # local kinematics (0.7 ms, no GIL held) and publishes a joint goal for
        # the 1 kHz sender. The queued fallback has to ask the controller
        # whether the target is reachable, which is a 42 ms round trip that
        # holds the GIL -- fine when it is the only thing running, fatal to a
        # streaming loop, which is why it is never used on the real-time path.
        if getattr(self.arm, "_rt_active", False):
            accepted = self.arm.servo_to_pose_rt(target)
            if not accepted:
                self.stats["unreachable"] += 1
                self.hand.set_opening(np.clip(gripper, 0.0, 1.0))
                self.stats["steps"] += 1
                return self.get_observation(t=self.stats["steps"])
        else:
            check = getattr(self.arm, "check_reachable", None)
            if check is not None:
                reachable, detail = check(target)
                if not reachable:
                    self.stats["unreachable"] += 1
                    logger.warning(
                        f"Holding position: controller refused "
                        f"{np.round(target[:3], 4).tolist()} ({detail})"
                    )
                    self.hand.set_opening(np.clip(gripper, 0.0, 1.0))
                    self.stats["steps"] += 1
                    return self.get_observation(t=self.stats["steps"])
            self.arm.servo_to_pose(target, wait=self.config.blocking_steps)

        self.hand.set_opening(np.clip(gripper, 0.0, 1.0))

        self._target_pose = target
        self.stats["steps"] += 1

        if self.arm.has_motion_error():
            self.stats["motion_errors"] += 1
            logger.error("Controller reports a motion error; call recover() before continuing")

        return self.get_observation(t=self.stats["steps"])

    def step_joints(self, q_target, gripper: Optional[float] = None,
                    observe: bool = True) -> Optional[AR5Observation]:
        """Command a joint posture directly, bypassing pose and IK entirely.

        ``observe=False`` skips building the observation and returns None. That
        is not a micro-optimisation: an observation is a camera frame per source
        plus three SDK round trips, each of which first blocks in
        updateRobotState waiting for a state frame. Teleoperation discards the
        return value entirely, so it was paying all of that -- and, worse, a
        *variable* amount of it -- on every step. The variability is the part
        that matters. The tracker's feed-forward is the goal's own speed,
        estimated as (goal - previous) / elapsed, so jitter in how long a step
        takes comes out as jitter in commanded velocity. It also crowds the
        1 kHz sender off the interpreter, which shows up as `late_ticks`.

        The joint-space teleoperation path. Nothing here consults the workspace
        box, because the box is a statement about tool *poses* and this path has
        no pose to clamp -- a joint goal is either inside the soft limits or it
        is not. That removes the two failure modes that dominate the Cartesian
        path, IK refusal and workspace clipping, and leaves the soft limits, the
        velocity and acceleration caps, and the jump gate doing the work.

        What the box was also quietly providing is a floor. Joint space has no
        equivalent, so the tool and the wrist are checked against one explicitly:
        a posture that would put either below ``joint_floor_z`` is refused, since
        the first thing under the arm is usually the table.
        """
        self._require_connection()
        q = np.asarray(q_target, dtype=np.float64).reshape(self.config.arm.num_joints)

        if not np.all(np.isfinite(q)):
            self.stats["unreachable"] += 1
            self.stats["steps"] += 1
            if gripper is not None:
                self.hand.set_opening(float(np.clip(gripper, 0.0, 1.0)))
            return self.get_observation(t=self.stats["steps"]) if observe else None

        lo = np.asarray(self.config.arm.joint_min) + self.config.arm.joint_limit_margin
        hi = np.asarray(self.config.arm.joint_max) - self.config.arm.joint_limit_margin
        clamped = np.clip(q, lo, hi)
        if not np.allclose(clamped, q):
            self.stats["joint_clamped"] += 1
        q = clamped

        # Rate-limit the goal's advance instead of letting the driver's jump gate
        # refuse it. That gate exists to catch a discontinuity of *intent* -- an
        # IK branch flip -- and the joint path has none: the map from leader
        # angles to follower angles is continuous. Here a big jump only ever
        # means the operator moved fast, and refusing it is the worst possible
        # response: the goal is dropped, the operator keeps moving, the gap grows,
        # and every subsequent goal is refused too. The arm stops for good.
        #
        # Leading the arm by more than one gate's width buys nothing anyway --
        # the 1 kHz tracker is capped at max_joint_speed, some twenty degrees a
        # second, far below the rate a hand can produce -- so clamping the
        # advance costs no responsiveness and removes the failure mode entirely.
        limit = self.config.arm.joint_goal_step
        anchor = self._joint_target
        commanded = self._commanded_joints()

        # 锚点过期了就以驱动的设定点为准。
        #
        # 2026-09-23 这里死锁过：`goto_joints()`（`s`/`r`/`q` 回 0 位走的就是它）
        # **直接调 `arm.servo_to_joints()`，绕过 step_joints**，于是臂到了中心位、
        # 驱动的 `_q_cmd` 也到了中心位，而 `_joint_target` 还停在上一条结束时那个
        # 目标上。再按空格，限速器从那个过期锚点出发每帧挪 0.075 rad，算出来的目标
        # 离 `_q_cmd` 有 0.7 rad —— 驱动的跳变门拒掉，**拒了又不更新
        # `_joint_target`**，下一帧一模一样，一路拒到底。现场表现是「按完 s 回位，
        # 再按空格臂就不动了」，日志里一行不变的
        # `goal jumps 0.465 rad ... refusing`。
        #
        # 修 `goto_joints` 一处不够 —— 任何绕过 step_joints 去动臂的路子都会再犯。
        # 这里让锚点自己对齐：差得比跳变门还远就说明它已经不代表「上一个目标」了，
        # 那就信驱动的。差得小的时候仍然用 `_joint_target`，因为被拒的目标不会写回
        # `_q_cmd`，用它才能保住「不许因为被拒就往前跳」这个性质。
        if (anchor is not None and commanded is not None
                and float(np.max(np.abs(anchor - commanded))) > self.config.arm.joint_jump_max):
            self.stats["anchor_resync"] = self.stats.get("anchor_resync", 0) + 1
            logger.warning(
                "[AR5Env] 限速锚点离驱动设定点 %.3f rad，已过期（多半是刚走过 "
                "goto_joints/回位）。改用驱动的设定点，否则每个目标都会被跳变门拒掉。",
                float(np.max(np.abs(anchor - commanded))))
            anchor = commanded
        if anchor is None:
            anchor = commanded
        if anchor is not None:
            # **整体按比例缩放，不是逐关节截断。**
            #
            # `np.clip(delta, -limit, limit)` 会改变运动方向：目标要 j0 走 0.30、
            # j1 走 0.05 时，逐关节截断给出 (0.075, 0.05) —— j1 走满、j0 只走了
            # 四分之一，于是 j1 相对 j0 早到六倍，臂走的不是操作员比划的那条线。
            # 现场表现是两件事，2026-09-23 用户都报了：
            #   ① 「会执行从来没做过的动作」—— 路径被掰弯了
            #   ② 「行程结束后有动作冗余」—— 主臂停下时快的关节已到位、慢的还在爬
            #
            # 按最大分量统一缩放就只影响**快慢**、不影响**方向**，落后时臂沿着
            # 同一条路慢慢追。`goto_joints` 里本来就是这么写的，两处不一致纯属遗漏。
            delta = q - anchor
            peak = float(np.max(np.abs(delta)))
            if peak > limit:
                self.stats["joint_rate_limited"] += 1
                delta = delta * (limit / peak)
            q = anchor + delta

        low = self._lowest_point(q)
        if low is not None and low < self.config.arm.joint_floor_z:
            self.stats["floor_blocked"] += 1
            self.stats["steps"] += 1
            # Refusing the goal must not mean publishing nothing. The 1 kHz
            # sender treats a gap in goals as a dead control loop and brakes
            # into a latch that only re-engaging clears -- so a blocked goal
            # would stop the arm permanently instead of just holding it at the
            # floor. Republish where the arm already is: the floor still blocks
            # the motion, but the loop stays visibly alive.
            hold = self._joint_target
            if hold is None:
                hold = self._commanded_joints()
            if hold is not None:
                self.arm.servo_to_joints(hold)
            if gripper is not None:
                self.hand.set_opening(float(np.clip(gripper, 0.0, 1.0)))
            return self.get_observation(t=self.stats["steps"]) if observe else None

        # The arm's goal goes out first, always. Its sender latches after
        # goal_stale_timeout without one, so nothing that can block -- and hand
        # I/O can -- belongs in front of this.
        if not self.arm.servo_to_joints(q):
            self.stats["unreachable"] += 1
        else:
            self._joint_target = q

        if gripper is not None:
            self.hand.set_opening(float(np.clip(gripper, 0.0, 1.0)))

        self.stats["steps"] += 1
        if self.arm.has_motion_error():
            self.stats["motion_errors"] += 1
            logger.error("Controller reports a motion error; call recover() before continuing")

        return self.get_observation(t=self.stats["steps"]) if observe else None

    def _commanded_joints(self) -> Optional[np.ndarray]:
        """The tracker's own setpoint, not the encoders.

        Anchoring the rate limit on a measurement would let tracking lag shrink
        every step: the arm trails the goal, so `measured + limit` is short of
        `commanded + limit` by exactly that lag.
        """
        lock = getattr(self.arm, "_setpoint_lock", None)
        q_cmd = getattr(self.arm, "_q_cmd", None)
        if lock is not None and q_cmd is not None:
            with lock:
                return self.arm._q_cmd.copy()
        try:
            return self.arm.get_joint_positions()
        except Exception:  # noqa: BLE001
            return None

    def _lowest_point(self, q: np.ndarray) -> Optional[float]:
        """Height of the lowest of the tool and the outboard joint origins.

        Only the outboard joints are worth checking: joints 0 to 2 sit at fixed
        heights close to the base and cannot be driven into the table.
        """
        kin = getattr(self.arm, "kinematics", None)
        if kin is None:
            return None
        try:
            T, _, origins = kin._chain(q)
        except Exception:  # noqa: BLE001
            return None
        heights = [float(T[2, 3])] + [float(origins[i][2]) for i in range(3, len(origins))]
        return min(heights)

    def recover(self) -> None:
        """Clear a controller fault and any streaming latch, then re-anchor.

        Deliberately explicit: the real-time sender latches after braking so
        motion cannot resume on its own, and only an operator action of this
        kind is allowed to re-arm it.
        """
        self._require_connection()
        # Fault clearing is conditional inside the driver: running it on a
        # healthy real-time stream silently stops the arm executing.
        self.arm.recover_from_error()
        clear = getattr(self.arm, "clear_latch", None)
        if clear is not None:
            clear()
        self._target_pose = self.arm.get_ee_pose()

    def reset(self, fake: bool = False) -> AR5Observation:
        """Return to the home posture and open the hand."""
        self._require_connection()
        if not fake:
            self.arm.move_to_joints(np.array(self.config.arm.home_joints, dtype=np.float64), blocking=True)
            self.hand.set_opening(1.0)
            time.sleep(self.config.settle_time)

        # Re-anchor the setpoint to where the arm actually is after homing.
        self._target_pose = self.arm.get_ee_pose()
        self.stats = {"steps": 0, "delta_clamped": 0, "workspace_clamped": 0, "unreachable": 0,
                      "resynced": 0, "leashed": 0, "motion_errors": 0,
                      "joint_clamped": 0, "floor_blocked": 0, "joint_rate_limited": 0}
        return self.get_observation(t=0)

    def stop(self) -> None:
        if self.arm is not None:
            self.arm.stop()


def make_ar5_env(config: Optional[AR5Config] = None, connect: bool = True) -> AR5Env:
    env = AR5Env(config)
    if connect:
        env.connect()
    return env
