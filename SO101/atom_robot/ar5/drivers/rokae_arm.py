"""ROKAE AR5 driver built on xCoreSDK-Python.

    https://github.com/RokaeRobot/xCoreSDK-Python

Written against a live AR5-5_0.8L-W4C5C11 on xCore v3.2.1 with SDK 0.7.1.
The SDK's conventions, all confirmed on the hardware:

1.  **Errors arrive in a trailing dict, not as exceptions.** The dict is filled
    on success too, as ``{'ec': 0, 'message': '操作成功完成'}``, so only a
    non-zero ``ec`` is a failure. Testing the dict for emptiness flags every
    successful call.

2.  **``jointPos()`` returns 13 values on this arm**, of which only the first
    ``joint_num`` (7) are real; the tail is external-axis padding.

3.  **Poses are translation plus RPY Euler angles**, not axis-angle and not a
    4x4 matrix. ``CartesianPosition.pos`` (the 16-element matrix field) is left
    zeroed by ``cartPosture``. The RPY convention is Rz(rz)Ry(ry)Rx(rx) --
    see ``transforms.rpy_to_matrix``.

4.  **The xMate model library does not support AR5**: ``robot.model()`` raises
    "xMate模型库暂不支持的机型". So there is no SDK forward/inverse kinematics
    or Jacobian for this arm. Tool frame and payload are set through the
    real-time controller instead, which does work.

5.  **Real-time control is push, not callback.** The SDK's callback scheduler
    (``setControlLoopCar`` + ``startLoop``) runs at 1 kHz but calls Python
    from its own thread, starving the interpreter, and ``stopLoop`` then
    deadlocks on the GIL. ``sendCommand`` from an ordinary Python thread
    avoids both; the SDK documents the loop functions as unnecessary then.

6.  **The real-time channel is UDP and needs a local bind address.** The
    constructor's second argument supplies it; without it, RT mode fails
    with "1338 UDP socket打开失败".

Until the SDK is installed, run everything with ``AR5Config(mock=True)``.
"""

import datetime
import logging
import threading
import time
from typing import List, Optional

import numpy as np

from ..transforms import axis_angle_to_matrix, internal_pose_to_rpy, rpy_pose_to_internal
from .base import ArmDriver

logger = logging.getLogger(__name__)

# Candidate SDK classes, most to least likely for a 7-DoF AR arm. Only used
# when ArmConfig.robot_class is left empty. Connecting with the wrong class
# fails cleanly with "机器人实例类型与连接的机型不符", so probing is safe.
_ROBOT_CLASS_CANDIDATES = (
    "xMateErProRobot",   # collaborative 7-axis -- AR5 connects as this
    "xMateCr5Robot",
    "xMateRobot",        # collaborative 6-axis
    "StandardRobot",     # industrial 6-axis
)


def load_sdk(sdk_path: str = ""):
    """Import the platform-specific xCoreSDK binding.

    The SDK is not on PyPI: it ships as a prebuilt library under `Release/`,
    which must be importable. `sdk_path` is prepended to sys.path when given.
    """
    import platform
    import sys

    if sdk_path and sdk_path not in sys.path:
        sys.path.insert(0, sdk_path)

    system = platform.system()
    try:
        if system == "Windows":
            from Release.windows import xCoreSDK_python
        elif system == "Linux":
            from Release.linux import xCoreSDK_python
        else:
            raise ImportError(f"xCoreSDK has no build for {system}")
    except ImportError as exc:
        raise ImportError(
            "xCoreSDK-Python is not importable.\n"
            "  1. git clone https://github.com/RokaeRobot/xCoreSDK-Python\n"
            "  2. download the Release matching Linux x86_64 + your Python version\n"
            "  3. unpack its Release/linux/*.so into the repo's Release/linux/\n"
            "  4. set ArmConfig.sdk_path to the directory holding Release/\n"
            "To develop without the arm, use AR5Config(mock=True)."
        ) from exc

    return xCoreSDK_python


def resolve_robot_class(sdk, requested: str = ""):
    """Pick the SDK class for this arm."""
    if requested:
        if not hasattr(sdk, requested):
            available = sorted(n for n in dir(sdk) if n.endswith("Robot"))
            raise AttributeError(
                f"The installed SDK exports no class {requested!r}. Available: {available}"
            )
        return getattr(sdk, requested)

    for name in _ROBOT_CLASS_CANDIDATES:
        if hasattr(sdk, name):
            logger.warning(
                f"[RokaeAR5] ArmConfig.robot_class is unset; auto-selected {name!r}. "
                "Confirm with `probe_rokae.py --connect` and set it explicitly."
            )
            return getattr(sdk, name)

    available = sorted(n for n in dir(sdk) if n.endswith("Robot"))
    raise AttributeError(f"No known robot class found in the SDK. Available: {available}")


class SDKError(RuntimeError):
    """An xCoreSDK call reported a non-zero error code."""


class RokaeAR5(ArmDriver):
    """Cartesian-space wrapper around a ROKAE AR-series arm.

    Poses crossing this boundary are always ``[x, y, z, rx, ry, rz]``: meters
    and axis-angle radians in the robot base frame. The SDK speaks translation
    plus RPY; conversion happens at this edge so the rest of the stack keeps the
    LIBERO-compatible axis-angle convention the energy head was trained on.
    """

    def __init__(self, config):
        self.config = config
        self.sdk = None
        self._robot = None
        self._rt = None
        self._connected = False
        self._rt_active = False

        # Shared with the 1 kHz sender thread, all under _setpoint_lock.
        self._setpoint_lock = threading.Lock()
        self._q_goal = None        # published joint goal
        self._q_cmd = None         # what was last commanded (sender-owned)
        self._v_cmd = None         # commanded joint velocity (sender-owned)
        self._a_cmd = None         # commanded joint acceleration (S-curve mode only)
        self._otg = None           # Ruckig online trajectory generator, built lazily
        self._otg_warned = False
        self._goal_seq = 0         # increments on every accepted publish
        self._goal_ticks = 0       # sender ticks since the goal last changed
        self._goal_rate = None     # estimated dq_goal/dt, for feed-forward
        self._goal_prev = None
        self._goal_time = None

        self._rt_thread = None
        self._rt_stop = None
        self._rt_latched = None    # set once motion is refused until re-engage
        self._rt_stopped = None
        self._rt_error = None
        self._kinematics = None
        self._state_subscribed = False

        self._joint_lo = np.asarray(config.joint_min, dtype=np.float64)
        self._joint_hi = np.asarray(config.joint_max, dtype=np.float64)
        margin = getattr(config, "joint_limit_margin", 0.0)
        self._joint_lo = self._joint_lo + margin
        self._joint_hi = self._joint_hi - margin

        # Counters, reported after a session so nothing degrades silently.
        self._sent = 0
        self._late = 0
        self._limit_hits = 0
        self._rejected = {"latched": 0, "nonfinite": 0, "limits": 0, "jump": 0}

    # ---------------------------------------------------------- SDK plumbing

    @staticmethod
    def _call(fn, *args):
        """Invoke an SDK method that reports failures through a trailing dict.

        The dict is always populated, including on success, where it reads
        ``{'ec': 0, 'message': '操作成功完成'}``.
        """
        ec = {}
        result = fn(*args, ec)
        code = ec.get("ec", 0) if isinstance(ec, dict) else 0
        if code:
            name = getattr(fn, "__name__", str(fn))
            raise SDKError(f"{name} failed (ec={code}): {ec.get('message', ec)}")
        return result

    def _require_connection(self):
        if not self._connected or self._robot is None:
            raise RuntimeError("Arm driver used before connect()")
        return self._robot

    # ------------------------------------------------------------------ setup

    def connect(self, power_on: bool = True) -> None:
        self.sdk = load_sdk(self.config.sdk_path)
        robot_cls = resolve_robot_class(self.sdk, self.config.robot_class)

        # The second argument is the local address the real-time UDP socket
        # binds to. Leaving it empty is fine for the queued API but makes
        # real-time mode fail to open its socket.
        local_ip = getattr(self.config, "local_ip", "") or ""
        self._robot = robot_cls(self.config.ip, local_ip)
        self._call(self._robot.connectToRobot)

        info = self._call(self._robot.robotInfo)
        if info.joint_num != self.config.num_joints:
            raise SDKError(
                f"Controller reports joint_num={info.joint_num} but ArmConfig.num_joints "
                f"is {self.config.num_joints}. Fix num_joints, and PROPRIO_DIM "
                f"(= joints + 1) in prismatic/vla/constants.py."
            )
        logger.info(
            f"[RokaeAR5] {info.type}  xCore {info.version}  {info.joint_num} joints"
        )

        if power_on:
            self._power_on()

        self._connected = True
        self._apply_collision_detection()

        if self.config.use_realtime:
            self._start_realtime()

    def _apply_collision_detection(self) -> None:
        """Re-arm collision detection with this project's parameters.

        2026-09-17: driving a screw produces enough reaction torque to trip the
        controller's default collision detection. The default behaviour is a
        hard stop, which **drops power** -- the arm brake-locks ("又抱死了") and,
        because the real-time session is killed mid-flight, the controller is
        left in `rtControlling`, after which every queued Move* is refused with
        `ec=262 运动控制模式错误`. One trip, both symptoms.

        Two changes: lower sensitivity so ordinary process torque does not trip
        it, and `suppleStop` so a genuine trip goes compliant instead of cutting
        power -- recoverable without a power cycle.

        The setting does not survive a controller restart, so it is applied on
        every connect. Enabling while already enabled returns `ec=264 重复操作`,
        so it has to be disabled first; that is not an error worth failing on.

        Tolerated failure: this is a safety *relaxation*, and refusing to run
        because it could not be applied would leave the arm stricter, not less
        safe. It is logged either way, never silent.
        """
        if not getattr(self.config, "collision_detection_enabled", True):
            try:
                self._call(self._robot.disableCollisionDetection)
                logger.warning(
                    "[RokaeAR5] controller-level collision detection DISABLED by config. "
                    "The RSC joint collision-force limit still applies and cannot be "
                    "changed from the SDK -- if the arm still stops, it is the RSC."
                )
            except SDKError as exc:
                logger.warning(f"[RokaeAR5] could not disable collision detection: {exc}")
            return

        sens = list(self.config.collision_sensitivity or ())
        if not sens:
            return
        if len(sens) != self.config.num_joints:
            logger.warning(
                f"[RokaeAR5] collision_sensitivity has {len(sens)} values, "
                f"need {self.config.num_joints}; leaving the controller default"
            )
            return
        behaviour = getattr(self.sdk.StopLevel, self.config.collision_behaviour, None)
        if behaviour is None:
            logger.warning(
                f"[RokaeAR5] unknown collision_behaviour "
                f"{self.config.collision_behaviour!r}; leaving the controller default"
            )
            return
        try:
            # Already-enabled is the normal case, so a failure here is expected
            # and ignored; only the enable that follows actually matters.
            try:
                self._call(self._robot.disableCollisionDetection)
            except SDKError:
                pass
            self._call(self._robot.enableCollisionDetection,
                       sens, behaviour, float(self.config.collision_fallback))
            logger.info(
                f"[RokaeAR5] collision detection: sensitivity {sens[0]} x{len(sens)}, "
                f"{self.config.collision_behaviour}, "
                f"fallback/compliance {self.config.collision_fallback}"
            )
        except SDKError as exc:
            logger.warning(f"[RokaeAR5] could not set collision detection: {exc}")

    def _power_on(self) -> None:
        """Enable the arm. Tolerated failure: a teach pendant can hold the enable."""
        try:
            self._call(self._robot.setOperateMode, self.sdk.OperateMode.automatic)
            self._call(self._robot.setPowerState, True)
            logger.info("[RokaeAR5] powered on, automatic mode")
        except SDKError as exc:
            logger.warning(
                f"[RokaeAR5] could not power on from the SDK ({exc}). Arms with an "
                "external enable switch or teach pendant must be enabled there. "
                "Reading state still works; motion will be refused."
            )

        # 上面是"容忍失败"，但**不能容忍它静默**。没上电的话后面每一步都会失败，
        # 而报出来的是下游的错（RT 起不来 -> 回退 -> Move* 被拒），和真原因隔着
        # 好几层。这里把真相说一次，用 ERROR 级别，别让它淹在 warning 里。
        try:
            state = str(self._call(self._robot.powerState))
        except SDKError:
            return
        if not state.endswith("on"):
            logger.error(
                f"[RokaeAR5] the arm is NOT powered (powerState={state}). Everything "
                "downstream will fail with a misleading error. Note powerState reports "
                "'off' even when an e-stop is latched. A collision stop also drops "
                "power -- run `python -m atom_robot.ar5.power_on --enable`, "
                "and if that reports ec=-3, an e-stop has to be released by hand."
            )

    def _start_realtime(self) -> None:
        """Open the real-time JOINT stream, falling back if unavailable.

        Joint space, not Cartesian: RT Cartesian is accepted by the controller
        and produces no motion at all (verified), while RT joint tracks exactly
        -- 2.000 deg commanded, 2.000 deg measured. Inverse kinematics is done
        locally from the controller's own DH table, which takes 0.7 ms and holds
        no GIL, instead of the 42.5 ms GIL-holding checkPath round trip that
        would blank 40 of every 48 ms of this stream.

        Push mode, not the callback scheduler: the SDK calls a registered
        callback from its own thread 1000x/s, which starves the interpreter, and
        stopLoop then deadlocks waiting on that thread while holding the GIL it
        needs.
        """
        try:
            self._call(self._robot.setRtNetworkTolerance, self.config.rt_network_tolerance)
            self._call(self._robot.setMotionControlMode, self.sdk.MotionControlMode.RtCommandMode)

            self._rt = self._robot.getRtMotionController()
            self._apply_rt_safety()

            if self._kinematics is None:
                from ..kinematics import AR5Kinematics

                self._kinematics = AR5Kinematics.from_robot(self)

            # Start from where the arm is, at rest, so the stream opens holding.
            q_now = self.get_joint_positions()
            with self._setpoint_lock:
                self._q_goal = q_now.copy()
                self._q_cmd = q_now.copy()
                self._v_cmd = np.zeros(self.config.num_joints)
                self._a_cmd = np.zeros(self.config.num_joints)
                self._goal_seq = 0
                self._goal_ticks = 0

            # Subscribe to the state stream. Two separate reasons, both
            # load-bearing in real-time mode:
            #   * jointPos() and cartPosture() otherwise serve a snapshot that
            #     never refreshes. Measured: joint readings bit-identical across
            #     184 samples while the tracker commanded 8.8 degrees of motion.
            #   * the SDK surfaces real-time motion faults only through
            #     updateRobotState(), so without it a stopped stream reports
            #     nothing at all.
            self._robot.startReceiveRobotState(
                datetime.timedelta(milliseconds=1),
                ["q_m", "pos_m"],
            )
            self._state_subscribed = True

            self._rt.startMove(self.sdk.RtControllerMode.jointPosition)

            self._rt_stop = threading.Event()
            self._rt_latched = threading.Event()
            self._rt_stopped = threading.Event()
            self._rt_error = None
            self._rt_thread = threading.Thread(
                target=self._rt_sender, name="ar5-rt-sender", daemon=True
            )
            self._rt_thread.start()

            time.sleep(0.1)
            if self._rt_error is not None:
                raise SDKError(self._rt_error)

            self._rt_active = True
            logger.info(
                f"[RokaeAR5] real-time joint streaming active "
                f"(1 kHz push, {self.config.max_joint_speed} rad/s cap)"
            )
        except Exception as exc:  # noqa: BLE001 - SDK exception types vary
            self._stop_rt_thread()
            self._rt = None
            self._rt_active = False

            # 回退到排队式之前**必须把控制模式切回去**。上面 try 的第一件事就是
            # 设 RtCommandMode；失败后不还原，控制器就还留在 RT 模式里，而排队式
            # 的 Move* 在 RT 模式下会被直接拒绝。于是这个"回退"根本不是回退 ——
            # 它把一个清楚的「RT 启动失败」，延迟成几秒后一句莫名其妙的
            # `moveAppend failed (ec=262) 运动控制模式错误`，而且报错位置在
            # AR5Env.step 里，离真正的原因十万八千里。
            #
            # 2026-09-17 实测：碰撞停机掉电之后每次都走这条路。
            restored = False
            try:
                self._call(self._robot.setMotionControlMode,
                           self.sdk.MotionControlMode.NrtCommandMode)
                restored = True
            except Exception as restore_exc:  # noqa: BLE001
                logger.error(
                    f"[RokaeAR5] real-time start failed AND the controller could not be "
                    f"put back into NrtCommandMode ({restore_exc}). Queued moves will be "
                    f"refused with ec=262 until it is. Run `power_on` to clear it."
                )
            logger.warning(
                f"[RokaeAR5] real-time control unavailable ({exc}); falling back to "
                f"queued moves{' (controller returned to NrtCommandMode)' if restored else ''}. "
                "Expect a lower control rate and coarser motion. "
                "The usual cause is the arm not being powered -- a collision stop drops "
                "power, and this is what you see on the next run."
            )

    # ------------------------------------------------------- real-time sender

    # ── Jerk-limited S-curve tracker (Ruckig), used when config.max_joint_jerk > 0 ──
    def _otg_ready(self) -> bool:
        if self.config.max_joint_jerk <= 0:
            return False
        if self._otg is not None:
            return True
        try:
            from ruckig import Ruckig, InputParameter, OutputParameter  # noqa: WPS433
        except Exception as exc:  # noqa: BLE001
            if not self._otg_warned:
                self._otg_warned = True
                logger.warning(f"[RokaeAR5] max_joint_jerk set but ruckig unavailable ({exc}); "
                               "using the second-order tracker")
            return False
        n = self.config.num_joints
        self._otg = Ruckig(n, self.config.rt_period)
        self._otg_in = InputParameter(n)
        self._otg_out = OutputParameter(n)
        self._otg_in.max_velocity = [float(self.config.max_joint_speed)] * n
        self._otg_in.max_acceleration = [float(self.config.max_joint_accel)] * n
        self._otg_in.max_jerk = [float(self.config.max_joint_jerk)] * n
        self._otg_ControlInterface = __import__("ruckig").ControlInterface
        logger.info(f"[RokaeAR5] tracker: jerk-limited S-curve "
                    f"(V {self.config.max_joint_speed} rad/s, A {self.config.max_joint_accel} rad/s², "
                    f"J {self.config.max_joint_jerk} rad/s³)")
        return True

    def _otg_tick(self, q_goal: Optional[np.ndarray], feed_forward: Optional[np.ndarray]) -> np.ndarray:
        """One millisecond of jerk-limited tracking. q_goal=None means brake to rest.

        Ruckig replans a time-optimal trajectory under the V/A/J caps every
        tick from the *commanded* state toward the latest goal (arriving with
        the goal's own velocity, so a moving target is tracked without a
        standing lag). Because jerk is bounded, a goal that jitters by a
        fraction of a millimetre at 50 Hz produces a bounded acceleration
        ripple instead of the ±A square wave of the second-order tracker.
        """
        inp, out = self._otg_in, self._otg_out
        V = self.config.max_joint_speed
        if self._a_cmd is None or self._a_cmd.shape != self._q_cmd.shape:
            self._a_cmd = np.zeros_like(self._q_cmd)
        inp.current_position = self._q_cmd.tolist()
        inp.current_velocity = self._v_cmd.tolist()
        inp.current_acceleration = self._a_cmd.tolist()
        if q_goal is None:
            inp.control_interface = self._otg_ControlInterface.Velocity
            inp.target_velocity = [0.0] * self.config.num_joints
            inp.target_acceleration = [0.0] * self.config.num_joints
        else:
            inp.control_interface = self._otg_ControlInterface.Position
            q_goal = np.where(np.isfinite(q_goal), q_goal, self._q_cmd)
            q_goal = np.clip(q_goal, self._joint_lo, self._joint_hi)
            inp.target_position = q_goal.tolist()
            v_ff = np.zeros_like(q_goal) if feed_forward is None else np.clip(feed_forward, -V, V)
            inp.target_velocity = v_ff.tolist()
            inp.target_acceleration = [0.0] * self.config.num_joints
        result = self._otg.update(inp, out)
        if int(result) < 0:
            # Invalid input (e.g. current state outside limits after an external
            # reset). Fall back to the second-order law for this tick and resync.
            if not self._otg_warned:
                self._otg_warned = True
                logger.warning(f"[RokaeAR5] ruckig returned {result}; second-order tick used")
            q_new = self._tick2(self._q_cmd.copy() if q_goal is None else q_goal, feed_forward)
            self._a_cmd[:] = 0.0
            return q_new
        q_new = np.asarray(out.new_position, dtype=np.float64)
        self._v_cmd = np.asarray(out.new_velocity, dtype=np.float64)
        self._a_cmd = np.asarray(out.new_acceleration, dtype=np.float64)
        # Same backstop as the second-order tracker: never leave the travel.
        lo, hi = self._joint_lo, self._joint_hi
        for i in range(q_new.shape[0]):
            if not (lo[i] <= q_new[i] <= hi[i]):
                q_new[i] = self._q_cmd[i] if not np.isfinite(q_new[i]) else min(max(q_new[i], lo[i]), hi[i])
                self._v_cmd[i] = 0.0
                self._a_cmd[i] = 0.0
                self._limit_hits += 1
        self._q_cmd = q_new
        return q_new

    def _tick(self, q_goal: np.ndarray, feed_forward: Optional[np.ndarray] = None) -> np.ndarray:
        if self._otg_ready():
            return self._otg_tick(q_goal, feed_forward)
        return self._tick2(q_goal, feed_forward)

    def _tick2(self, q_goal: np.ndarray, feed_forward: Optional[np.ndarray] = None) -> np.ndarray:
        """One millisecond of velocity- and acceleration-limited tracking.

        Per joint, aim at the speed that still allows a full-deceleration stop
        exactly on the goal -- v_des = sign(e) * min(V, sqrt(2*A*|e|)) -- then
        let the commanded velocity move toward that by at most A*dt. The square
        root is what makes the arm arrive at rest instead of overshooting.
        """
        dt = self.config.rt_period
        V = self.config.max_joint_speed
        A = self.config.max_joint_accel

        # Aim at the limit rather than through it. Letting the braking law
        # decelerate into the boundary is what keeps the approach smooth; if the
        # clamp below has to fire instead it zeroes the velocity in one tick,
        # which is an unbounded deceleration and reads as a knock on a
        # force-controlled arm. A non-finite goal is replaced by the current
        # command -- np.clip would propagate the NaN.
        q_goal = np.where(np.isfinite(q_goal), q_goal, self._q_cmd)
        q_goal = np.clip(q_goal, self._joint_lo, self._joint_hi)

        e = q_goal - self._q_cmd

        # Feed-forward: match the goal's own velocity, then let the error term
        # close whatever gap remains. Tracking a moving target on error alone
        # guarantees a standing lag; matching its speed removes the reason for
        # the lag to exist.
        if feed_forward is not None:
            v_ff = np.clip(feed_forward, -V, V)
        else:
            v_ff = np.zeros_like(e)

        # Three caps: the speed limit, the speed from which a stop still lands
        # on the goal, and the speed that exactly closes the remaining gap in
        # one tick.
        #
        # The braking curve is planned at half the available deceleration. In
        # continuous time sqrt(2*A*e) would be exact, but v_cmd can only change
        # by A*dt per tick, so it lags v_des on the way down and arrives with
        # speed left over -- which the position clamp then kills in a single
        # tick, an unbounded deceleration. Braking early leaves the headroom
        # that lag needs.
        A_brake = 0.5 * A
        v_err = np.sign(e) * np.minimum.reduce([
            np.full_like(e, V),
            np.sqrt(2.0 * A_brake * np.abs(e)),
            np.abs(e) / dt,
        ])
        # The sum is capped at the speed limit, so feed-forward buys
        # responsiveness without widening the safety envelope.
        v_des = np.clip(v_ff + v_err, -V, V)
        dv = np.clip(v_des - self._v_cmd, -A * dt, A * dt)
        self._v_cmd = self._v_cmd + dv
        q_new = self._q_cmd + self._v_cmd * dt

        # Written as `not (lo <= q <= hi)` on purpose: every comparison against
        # NaN is False, so the natural `if q < lo` form would wave a NaN through
        # and then poison v_cmd permanently.
        lo = self._joint_lo
        hi = self._joint_hi
        for i in range(q_new.shape[0]):
            if not (lo[i] <= q_new[i] <= hi[i]):
                # Reaching here means the braking law above did not hold the
                # joint inside its travel, so this is a backstop, not the normal
                # path. Zeroing the velocity is an unbounded deceleration; it is
                # accepted only because the alternative is leaving the limit.
                if not np.isfinite(q_new[i]):
                    q_new[i] = self._q_cmd[i]
                else:
                    q_new[i] = min(max(q_new[i], lo[i]), hi[i])
                self._v_cmd[i] = 0.0
                self._limit_hits += 1

        self._q_cmd = q_new
        return q_new

    def _brake_tick(self) -> np.ndarray:
        """Track toward a standstill at the current command."""
        if self._otg_ready():
            return self._otg_tick(None, None)
        return self._tick2(self._q_cmd.copy())

    def _at_rest(self) -> bool:
        return bool(np.max(np.abs(self._v_cmd)) < 1e-4)

    def hold_until_rest(self, timeout_s: float = 3.0) -> bool:
        """Keep the current goal alive until the tracker has actually stopped.

        The sender latches when goals stop arriving while it is still moving.
        A scripted move (homing, recentring) that publishes its final goal a
        fixed number of times and then goes quiet can trip that rule when the
        tracker is slower to settle than the script assumed -- measured
        2026-09-20 with the jerk-limited tracker: recentre finished, the caller
        went off to connect the headset, the arm was still finishing its
        S-curve, 0.25 s later it latched, the sender exited and the RT stream
        died (controller powered off in rtControlling). This touches only the
        staleness clock, never the goal, so it cannot move the arm anywhere
        the script did not ask for.
        """
        if not self._rt_active:
            return True
        deadline = time.perf_counter() + timeout_s
        while time.perf_counter() < deadline:
            with self._setpoint_lock:
                self._goal_ticks = 0
                v = self._v_cmd
            if v is None or bool(np.max(np.abs(v)) < 1e-4):
                return True
            time.sleep(0.02)
        logger.warning(f"[RokaeAR5] tracker still moving after {timeout_s:.1f} s hold")
        return False

    def _rt_sender(self) -> None:
        """Push one joint command per millisecond until told to stop.

        Every exit path brakes to a standstill first. Leaving the loop with a
        non-zero commanded velocity would strand the arm mid-motion with nobody
        driving it.
        """
        period = self.config.rt_period
        stale_after = self.config.goal_stale_timeout
        stale_ticks = max(1, int(stale_after / period))
        main_thread = threading.main_thread()
        braking = False
        next_deadline = time.perf_counter()

        try:
            while True:
                with self._setpoint_lock:
                    goal = self._q_goal.copy()
                    rate = None if self._goal_rate is None else self._goal_rate.copy()
                    self._goal_ticks += 1
                    ticks_since_goal = self._goal_ticks

                if self._rt_stop.is_set():
                    braking = True
                elif not main_thread.is_alive():
                    # The owner is gone. Do not just break -- that would leave
                    # the arm moving with the stream dead.
                    if not braking:
                        logger.error("[RokaeAR5] main thread gone; braking the arm")
                    braking = True
                elif self._rt_latched.is_set():
                    braking = True
                elif ticks_since_goal > stale_ticks:
                    # Goals dried up. Whether that is dangerous depends entirely
                    # on whether the arm was moving at the time.
                    if self._at_rest():
                        # Already stopped, so there is nothing to brake and
                        # nothing to protect anyone from. Latching here would
                        # punish every ordinary pause -- a scripted read loop, an
                        # operator thinking, a slow policy step -- and the arm
                        # would then refuse to move again until re-engaged.
                        pass
                    else:
                        if not braking:
                            logger.error(
                                f"[RokaeAR5] goals stopped arriving for {stale_after:.2f} s "
                                "while the arm was moving; braking and latching. Motion "
                                "cannot resume until the operator re-engages."
                            )
                        # Latching, not transient: an arm that goes still and
                        # then resumes on its own is how people get hurt
                        # standing inside it.
                        self._rt_latched.set()
                        braking = True

                q_cmd = self._brake_tick() if braking else self._tick(goal, rate)

                try:
                    self._rt.sendCommand(self.sdk.JointPosition(q_cmd.tolist()))
                    self._sent += 1
                except Exception as exc:  # noqa: BLE001
                    self._rt_error = f"real-time stream failed: {exc}"
                    logger.error(f"[RokaeAR5] {self._rt_error}")
                    self._rt_active = False
                    return

                if braking and self._at_rest():
                    return

                next_deadline += period
                remaining = next_deadline - time.perf_counter()
                if remaining > 0:
                    time.sleep(remaining)
                else:
                    # Behind schedule. Resynchronise rather than burst to catch
                    # up, which the controller reads as commands arriving early.
                    self._late += 1
                    next_deadline = time.perf_counter()
        finally:
            # Signals teardown that the thread is really gone, on every path
            # including the send failure above.
            self._rt_stopped.set()

    def _stop_rt_thread(self) -> None:
        """Ask the sender to brake and stop, and verify that it did.

        The join result is load-bearing. A sender that outlives this call would
        be joined by a second one started later, and two threads pushing into
        one RT session at a combined 2 kHz alternate setpoints from independent
        trackers.
        """
        if self._rt_stop is not None:
            self._rt_stop.set()
        thread = self._rt_thread
        if thread is not None and thread.is_alive():
            # Long enough for a full-speed brake (V/A) plus slack.
            timeout = self.config.max_joint_speed / self.config.max_joint_accel + 1.0
            thread.join(timeout=timeout)
            if thread.is_alive():
                self._rt_latched.set()
                self._rt_active = False
                raise SDKError(
                    "the real-time sender did not stop; refusing to touch the stream "
                    "again. Restart the process and stop the arm from the pendant."
                )
        self._rt_thread = None
        self._rt_stop = None

    def _apply_rt_safety(self) -> None:
        """Push limits into the controller, where they are enforced in hardware.

        Python-side clamping in AR5Env still applies; this is the second line,
        and the only one that survives a stall or a crash of this process.
        """
        # 这里原先有一段 `self._rt.setCollisionBehaviour(thresholds)`。
        # 2026-09-17 核实：**整个 SDK 里没有 setCollisionBehaviour 这个方法**，
        # 调到就是 AttributeError；只因为 collision_torque_thresholds 一直是空的
        # 才从没炸过。碰撞检测的真接口是 robot 上的 enableCollisionDetection，
        # 见 `_apply_collision_detection()`。

        if self.config.payload_mass > 0:
            load = self.sdk.Load()
            load.mass = self.config.payload_mass
            load.cog = list(self.config.payload_com)
            load.inertia = [0.0] * 3
            self._call(self._rt.setLoad, load)
            logger.info(f"[RokaeAR5] payload set to {self.config.payload_mass} kg")

        if any(self.config.tcp_offset):
            frame = self._pose_to_matrix16(np.array(self.config.tcp_offset, dtype=np.float64))
            self._call(self._rt.setEndEffectorFrame, frame)
            logger.info(f"[RokaeAR5] TCP frame set to {self.config.tcp_offset}")

    def disconnect(self) -> None:
        if self._robot is None:
            return
        self._stop_rt_thread()
        steps = [
            ("stop state stream", lambda: self._state_subscribed and self._robot.stopReceiveRobotState()),
            ("stop RT move", lambda: self._rt and self._rt.stopMove()),
            ("restore Nrt mode", lambda: self._call(
                self._robot.setMotionControlMode, self.sdk.MotionControlMode.NrtCommandMode)),
        ]
        # Powering off here used to be unconditional, which meant every eval
        # rollout ended with a de-energised arm and the next one died on
        # "network error" until someone ran power_on again. The arm has already
        # been taken out of RT mode above, so leaving it energised just holds
        # the pose -- the normal idle state between two runs.
        if getattr(self.config, "power_off_on_disconnect", False):
            steps.append(("power off", lambda: self._call(self._robot.setPowerState, False)))
        steps.append(("disconnect", lambda: self._call(self._robot.disconnectFromRobot)))
        for label, action in steps:
            try:
                action()
            except Exception as exc:  # noqa: BLE001
                logger.error(f"[RokaeAR5] {label} failed during disconnect: {exc}")

        self._robot = self._rt = None
        self._connected = False
        self._rt_active = False
        # 力控句柄是缓存的，绑在上一个 robot 对象上 —— 不清掉，重连之后会拿着
        # 一个指向已断开会话的句柄。
        self._fc = None

    # ------------------------------------------------------------------ state

    def refresh_state(self) -> None:
        """Pull one frame from the state stream, if one is subscribed.

        Also the only place a real-time motion fault becomes visible: the SDK
        raises it out of updateRobotState and nowhere else.
        """
        if not self._state_subscribed:
            return
        try:
            self._robot.updateRobotState(datetime.timedelta(milliseconds=2))
        except Exception as exc:  # noqa: BLE001
            self._rt_error = f"real-time motion fault: {exc}"
            logger.error(f"[RokaeAR5] {self._rt_error}")
            if self._rt_latched is not None:
                self._rt_latched.set()

    def get_joint_torque(self) -> np.ndarray:
        """Joint torques from the torque sensors, Nm, one per joint.

        This is the *raw* measurement, gravity and dynamics included -- it is
        `getEndTorque`'s `joint_torque_measured`, same quantity. The controller's
        model-subtracted `external_torque_measured` comes from `get_end_wrench`.
        """
        robot = self._require_connection()
        raw = np.asarray(self._call(robot.jointTorque), dtype=np.float64)
        return raw[:self.config.num_joints]

    def get_end_wrench(self, frame: str = "world"):
        """``(joint_raw, joint_external, cart_torque, cart_force)`` from getEndTorque.

        **The method lives on `robot.forceControl()`, not on the robot** -- calling
        `robot.getEndTorque` is an AttributeError on this arm. Taking that handle
        does not change the control mode; `fcInit`/`fcStart` would, and are not
        called here, so this stays a read.

        Forces are in newtons, torques in Nm. Both carry the un-subtracted weight
        of whatever is bolted to the flange (`payload_mass` is 0 while the hand is
        1.2-1.3 kg), so only *changes* are meaningful -- tare against a baseline.

        One synchronous round trip, so do not call it at the control rate; the
        1 kHz sender needs the GIL this holds.
        """
        robot = self._require_connection()
        # **句柄必须缓存。** 实测 2026-09-17：`robot.forceControl()` 本身要
        # **41 ms**，而 `getEndTorque()` 只要 **0.30 ms** —— 那 42 ms 从头到尾
        # 是取句柄的开销，不是读数据的。每次重取会把 1 kHz 发送流打出洞
        # （实测 963/s 掉到 781/s，控制器当丢包、停止执行却继续收指令）。
        # 缓存之后这一路可以按控制频率采。
        fc = getattr(self, "_fc", None)
        if fc is None:
            getter = getattr(robot, "forceControl", None)
            if getter is None:
                raise SDKError("this robot class has no forceControl()")
            fc = getter()
            if not hasattr(fc, "getEndTorque"):
                raise SDKError("forceControl() object has no getEndTorque()")
            self._fc = fc
        ref = getattr(self.sdk.FrameType, frame)
        out = [self.sdk.PyTypeVectorDouble() for _ in range(4)]
        self._call(fc.getEndTorque, ref, *out)
        return tuple(np.asarray(v.content(), dtype=np.float64) for v in out)

    def get_joint_positions(self) -> np.ndarray:
        """Current joint angles, radians.

        The SDK pads its return with external-axis slots, so the real joints
        are only the leading `num_joints` entries.
        """
        robot = self._require_connection()
        self.refresh_state()
        raw = np.asarray(self._call(robot.jointPos), dtype=np.float64)
        n = self.config.num_joints
        if raw.size < n:
            raise SDKError(f"jointPos returned {raw.size} values, need at least {n}")
        return raw[:n]

    def get_ee_pose(self) -> np.ndarray:
        """Current end-effector pose as ``[x, y, z, rx, ry, rz]`` (axis-angle)."""
        robot = self._require_connection()
        self.refresh_state()
        cp = self._call(robot.cartPosture, self.sdk.CoordinateType.flangeInBase)
        return rpy_pose_to_internal(cp.trans, cp.rpy)

    def read_joints_and_pose(self):
        """关节角（弧度）和末端位姿，**同一次状态刷新**里读出来的 —— 分开调 get_joint_positions、
        get_ee_pose 会各刷新一次，两样就不是同一刻的。OmniPicker 采集每圈都调。"""
        robot = self._require_connection()
        self.refresh_state()
        raw = np.asarray(self._call(robot.jointPos), dtype=np.float64)
        n = self.config.num_joints
        if raw.size < n:
            raise SDKError(f"jointPos returned {raw.size} values, need at least {n}")
        cp = self._call(robot.cartPosture, self.sdk.CoordinateType.flangeInBase)
        return raw[:n], rpy_pose_to_internal(cp.trans, cp.rpy)

    def get_elbow(self) -> float:
        """Arm angle. Seven axes are redundant, so this picks among IK solutions."""
        robot = self._require_connection()
        cp = self._call(robot.cartPosture, self.sdk.CoordinateType.flangeInBase)
        return float(cp.elbow)

    def has_motion_error(self) -> bool:
        return bool(self._rt.hasMotionError()) if self._rt is not None else False

    def recover_from_error(self, force: bool = False) -> None:
        """Clear a real-time motion fault, if there is one.

        Only when there is one. Calling automaticErrorRecovery on a healthy
        stream resets the controller's motion state: sendCommand keeps being
        accepted at 1 kHz, no error is reported anywhere, and the arm stops
        executing. Measured: an identical 40-step run moves 67.16 mm without
        this call and 0.00 mm with it.
        """
        if self._rt is None:
            return
        if not (force or self.has_motion_error() or self._rt_error is not None):
            return
        self._call(self._rt.automaticErrorRecovery)
        self._rt_error = None
        logger.info("[RokaeAR5] automatic error recovery ran")

    # ---------------------------------------------------------------- control

    def servo_to_pose(self, pose: np.ndarray, wait: bool = True) -> None:
        """Command one Cartesian setpoint.

        ``wait`` only applies to the queued fallback path. Waiting for each
        segment to finish is right for discrete moves, but it starves a
        teleoperation or policy loop: the call takes as long as the motion, so
        the next setpoint is computed from a stale observation and arrives late.
        Streaming callers pass wait=False.
        """
        robot = self._require_connection()
        pose = np.asarray(pose, dtype=np.float64).reshape(6)

        if self._rt_active:
            # Build the SDK object here, on the caller's thread: the 1 kHz sender
            # must not do allocation or conversion inside its cycle.
            command = self._to_cartesian_position(pose)
            with self._setpoint_lock:
                self._setpoint = pose
                self._setpoint_cmd = command
            return

        # A segment below the controller's minimum is rejected outright, so a
        # near-zero action becomes "hold position" rather than an error.
        if self.is_degenerate(pose):
            return

        cmd = self.sdk.MoveLCommand(
            self._to_cartesian_position(pose), self.config.move_speed, self.config.move_zone
        )
        # cmdID is an SDK PyString, not a Python str.
        self._call(robot.moveAppend, [cmd], self.sdk.PyString("vel_servo"))

        if wait:
            self._call(robot.moveStart)
            self._wait_for_motion()
            return

        # Streaming: the queue keeps executing on its own, and moveStart is
        # rejected with "机器人运动中" while it does. Start it only when the arm
        # has gone idle, otherwise just leave the new segment in the queue.
        if self._call(robot.operationState) != self.sdk.OperationState.moving:
            try:
                self._call(robot.moveStart)
            except SDKError as exc:
                # A race against the controller finishing the previous segment.
                # Harmless: the next call starts the queue.
                logger.debug(f"[RokaeAR5] moveStart skipped: {exc}")

    def move_to_joints(self, joints: np.ndarray, blocking: bool = True) -> None:
        robot = self._require_connection()
        joints = np.asarray(joints, dtype=np.float64).reshape(self.config.num_joints)

        cmd = self.sdk.MoveAbsJCommand(
            self.sdk.JointPosition(joints.tolist()), self.config.move_speed, self.config.move_zone
        )
        self._call(robot.moveAppend, [cmd], self.sdk.PyString("vel_home"))
        self._call(robot.moveStart)
        if blocking:
            self._wait_for_motion()

    def _wait_for_motion(self, timeout_s: float = 30.0) -> None:
        """Poll until the queued motion finishes. Move* calls are non-blocking."""
        import time

        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self._call(self._robot.operationState) != self.sdk.OperationState.moving:
                return
            time.sleep(0.01)
        raise SDKError(f"Motion did not complete within {timeout_s} s")

    def stop(self) -> None:
        if self._robot is None:
            return
        try:
            if self._rt is not None:
                self._stop_rt_thread()
                self._rt.stopMove()
                self._rt_active = False
            else:
                # stop() takes only the error dict; the stop level is not a parameter.
                self._call(self._robot.stop)
        except Exception as exc:  # noqa: BLE001
            logger.error(f"[RokaeAR5] stop() failed: {exc}")

    # --------------------------------------------------------------- publish

    @property
    def kinematics(self):
        """Local DH kinematics, built on first use from the controller's table."""
        if self._kinematics is None:
            from ..kinematics import AR5Kinematics

            self._kinematics = AR5Kinematics.from_robot(self)
        return self._kinematics

    @property
    def latched(self) -> bool:
        """True once the sender refused to keep moving and needs a re-engage."""
        return self._rt_latched is not None and self._rt_latched.is_set()

    def clear_latch(self) -> None:
        """Re-arm after a latch. Only in response to a deliberate operator action."""
        if self._rt_latched is not None and self._rt_latched.is_set():
            with self._setpoint_lock:
                if self._q_cmd is not None:
                    self._q_goal = self._q_cmd.copy()
                self._goal_ticks = 0
            self._rt_latched.clear()
            logger.info("[RokaeAR5] latch cleared; motion re-armed")

    def servo_to_joints(self, q_target) -> bool:
        """Publish a joint goal for the sender to track. Returns whether it was accepted.

        Four gates, in order. Each exists because rate limiting alone does not
        make a goal safe: it bounds how fast the arm travels there, not where
        "there" is.
        """
        if not self._rt_active:
            # Publishing into a slot nobody reads would look exactly like the
            # RT-Cartesian bug this design replaced: commands accepted, no motion.
            raise RuntimeError(
                "servo_to_joints requires the real-time stream; it is not running"
            )

        q = np.asarray(q_target, dtype=np.float64).reshape(-1)

        # Touch the staleness clock first, and on every path. A rejected goal
        # means the goal was bad, not that the publisher died -- and the sender
        # latches on silence. Without this, a brief run of unsolvable targets
        # freezes the arm permanently.
        with self._setpoint_lock:
            self._goal_ticks = 0

        if self.latched:
            self._rejected["latched"] += 1
            return False

        # NaN survives every ordering comparison, so check finiteness explicitly
        # and before anything else.
        if q.shape[0] != self.config.num_joints or not np.all(np.isfinite(q)):
            self._rejected["nonfinite"] += 1
            logger.error(f"[RokaeAR5] refusing a non-finite or wrong-width goal: {q}")
            return False

        if np.any(q < self._joint_lo) or np.any(q > self._joint_hi):
            self._rejected["limits"] += 1
            outside = [
                f"j{i}={q[i]:.4f} outside [{self._joint_lo[i]:.4f}, {self._joint_hi[i]:.4f}]"
                for i in range(q.shape[0])
                if not (self._joint_lo[i] <= q[i] <= self._joint_hi[i])
            ]
            logger.warning(f"[RokaeAR5] goal outside joint limits: {', '.join(outside)}")
            return False

        with self._setpoint_lock:
            reference = self._q_cmd if self._q_cmd is not None else q
            jump = float(np.max(np.abs(q - reference)))
            if jump > self.config.joint_jump_max:
                self._rejected["jump"] += 1
                logger.warning(
                    f"[RokaeAR5] goal jumps {jump:.3f} rad from the current command, over "
                    f"the {self.config.joint_jump_max} rad limit; refusing. This is what an "
                    "inverse-kinematics branch flip looks like."
                )
                return False

            # Estimate how fast the goal itself is moving. Without this the
            # tracker is driven by position error alone, which makes it a
            # first-order lag: it can only move as fast as the error it is
            # allowed to accumulate, so its speed is set by the leash rather
            # than by the operator. Measured before adding this, tool speed sat
            # at 23 mm/s and did not respond to raising either the velocity or
            # the acceleration limit.
            now = time.perf_counter()
            if self._goal_prev is not None and self._goal_time is not None:
                dt_goal = now - self._goal_time
                if 1e-4 < dt_goal < 0.5:
                    rate = (q - self._goal_prev) / dt_goal
                    if self._goal_rate is None:
                        self._goal_rate = rate
                    else:
                        # Light smoothing: the estimate comes from a 50 Hz
                        # signal and feeds a 1 kHz loop, so raw differences are
                        # noisy enough to buzz.
                        #
                        # A time constant, not a fixed weight. This used to be
                        # a flat 0.6/0.4 blend, which is only a defined amount
                        # of smoothing if the steps are evenly spaced -- and
                        # they are not: the caller's loop carries a six-servo
                        # serial read and, while collecting, a camera. A fixed
                        # weight smooths less on a long step than a short one,
                        # so loop-period jitter came straight out as
                        # feed-forward jitter, which is exactly what the
                        # smoothing was there to stop. At the nominal 50 Hz the
                        # default tau reproduces the old 0.4 exactly.
                        a = dt_goal / (dt_goal + max(1e-6, self.config.goal_rate_tau))
                        self._goal_rate = (1.0 - a) * self._goal_rate + a * rate
            self._goal_prev = q.copy()
            self._goal_time = now

            self._q_goal = q.copy()
            self._goal_seq += 1
            self._goal_ticks = 0
        return True

    def servo_to_pose_rt(self, pose) -> bool:
        """Cartesian target -> local IK -> published joint goal."""
        pose = np.asarray(pose, dtype=np.float64).reshape(6)
        T = np.eye(4)
        T[:3, :3] = axis_angle_to_matrix(pose[3:6])
        T[:3, 3] = pose[:3]

        with self._setpoint_lock:
            seed = self._q_cmd.copy() if self._q_cmd is not None else None
        if seed is None:
            seed = self.get_joint_positions()

        q, info = self.kinematics.ik(T, seed)
        if q is None:
            self._rejected["limits"] += 1
            logger.warning(f"[RokaeAR5] IK failed for {np.round(pose[:3], 4).tolist()}: {info['reason']}")
            return False
        return self.servo_to_joints(q)

    def rt_stats(self) -> dict:
        return {
            "sent": self._sent,
            "late_ticks": self._late,
            "limit_hits": self._limit_hits,
            "goals": self._goal_seq,
            "rejected": dict(self._rejected),
            "latched": self.latched,
        }

    # ------------------------------------------------------- representations

    def _to_cartesian_position(self, pose, reference=None):
        """Our axis-angle pose -> the SDK's CartesianPosition (translation + RPY).

        A 7-axis arm is redundant: one end-effector pose is reached by a whole
        family of joint solutions, and the arm angle picks one. A target built
        from translation and orientation alone leaves `elbow` at 0 with
        `hasElbow` false, the controller cannot resolve the inverse kinematics,
        and it drops the command -- silently, because moveAppend and moveStart
        only queue it. The failure surfaces only in the controller log, as
        "[SDK MoveL] 目标点超出运动范围" (id 50002).

        So the arm angle and axis configuration are carried over from a
        reference posture, by default the arm's current one. Small Cartesian
        steps stay in the same solution branch, which is what we want: the
        elbow should not swing while the end-effector moves a centimetre.
        """
        trans, rpy = internal_pose_to_rpy(pose)
        if reference is None:
            reference = self._call(
                self._robot.cartPosture, self.sdk.CoordinateType.flangeInBase
            )

        cp = self.sdk.CartesianPosition()
        cp.trans = trans.tolist()
        cp.rpy = rpy.tolist()
        cp.elbow = reference.elbow
        cp.hasElbow = True
        cp.confData = list(reference.confData)
        return cp

    # The controller refuses a segment shorter than roughly 0.05 mm with
    # "相邻两个目标点过近" (ec -50003). Measured on the AR5: 0.01 mm is refused,
    # 0.05 mm is accepted.
    MIN_SEGMENT_M = 5e-5

    def is_degenerate(self, pose: np.ndarray, reference: Optional[np.ndarray] = None) -> bool:
        """True when a target is too close to the current pose to be commanded.

        A policy that outputs a near-zero action would otherwise produce a
        command the controller rejects. Holding position is the right response,
        not an error.
        """
        current = self.get_ee_pose() if reference is None else reference
        return bool(np.linalg.norm(np.asarray(pose)[:3] - current[:3]) < self.MIN_SEGMENT_M)

    def check_reachable(self, pose: np.ndarray):
        """Validate a Cartesian target without moving. Returns (ok, detail).

        Pure computation on the controller, so it is safe to call before every
        commanded step. `detail` is the joint solution when reachable, and the
        controller's refusal message otherwise.
        """
        robot = self._require_connection()
        start_cp = self._call(robot.cartPosture, self.sdk.CoordinateType.flangeInBase)
        start_joint = list(self._call(robot.jointPos))

        # checkPath compares against the *current* pose, so a target that
        # coincides with it is refused as degenerate rather than unreachable.
        current = rpy_pose_to_internal(start_cp.trans, start_cp.rpy)
        if self.is_degenerate(pose, reference=current):
            return True, np.asarray(start_joint[: self.config.num_joints], dtype=np.float64)

        target = self._to_cartesian_position(pose, reference=start_cp)

        ec = {}
        try:
            solution = robot.checkPath(start_cp, start_joint, target, ec)
        except Exception as exc:  # noqa: BLE001
            return False, f"{type(exc).__name__}: {exc}"
        code = ec.get("ec", 0)
        if code:
            return False, f"ec={code}: {ec.get('message')}"
        return True, np.asarray(list(solution)[: self.config.num_joints], dtype=np.float64)

    @staticmethod
    def _pose_to_matrix16(pose) -> List[float]:
        """Our axis-angle pose -> 16 row-major doubles, for the frame setters."""
        from ..transforms import axis_angle_to_matrix

        pose = np.asarray(pose, dtype=np.float64).reshape(6)
        T = np.eye(4)
        T[:3, :3] = axis_angle_to_matrix(pose[3:6])
        T[:3, 3] = pose[:3]
        return T.reshape(-1).tolist()

    def solve_ik(self, pose: np.ndarray, arm_angle: float = 0.0) -> Optional[np.ndarray]:
        """Not available: the xMate model library does not support the AR series.

        ``robot.model()`` raises "xMate模型库暂不支持的机型" on this arm, so
        there is no SDK inverse kinematics to call. Reachability is instead
        bounded by the workspace box in AR5Env and by the controller-side
        Cartesian limit configured in `_apply_rt_safety`.
        """
        return None
