"""Map the SO-101 leader joint-for-joint onto the AR5.

The Cartesian route in :mod:`leader_mapping` runs the leader's tool motion
through inverse kinematics. That decouples the two arms' geometry, but it also
means the follower's *posture* is whatever IK picks: the elbow can sit somewhere
the leader's elbow is not, and every failure mode of IK -- singularities,
unreachable targets, the workspace box -- lands in the middle of teleoperation.

This module takes the other road. Each leader joint drives one AR5 joint
directly, and the AR5 joints with no counterpart are held still. There is no IK,
no workspace box, and no reachability gate: a joint command is either inside the
soft limits or it is not.

Correspondence, derived from both arms' own geometry rather than assumed
(SO-101 axes from its URDF, AR5 axes from the DH table the controller reports):

    SO-101            axis at rest         AR5   datasheet   axis at zero
    shoulder_pan      (0, 0, -1)  yaw       j0   axis 1      (0, 0, +1)  yaw
    shoulder_lift     (0, +1, 0)  pitch     j1   axis 2      (0, +1, 0)  pitch
    elbow_flex        (0, +1, 0)  pitch     j3   axis 4      (0, +1, 0)  pitch
    wrist_roll        (-1, 0, 0)  roll      j4   axis 5      (0, 0, +1)  roll
    wrist_flex        (0, +1, 0)  pitch     j5   axis 6      (0, +1, 0)  pitch
    --                                      j2   axis 3      (0, 0, +1)  roll   LOCKED
    --                                      j6   axis 7      (+1, 0, 0)  pitch  LOCKED

The AR5 is a seven-axis roll-pitch-roll-pitch-roll-pitch-pitch arm -- ROKAE
calls the last two a cross wrist -- so the two joints without a leader
counterpart are the upper-arm roll and the second wrist pitch. Locking those
leaves a five-axis arm whose structure is exactly the leader's.

Signs follow from the axes above. Both arms are assumed to sit facing the same
way, so a rotation is "the same" when its axis points the same way in the two
base frames:

  * pan: the leader's axis is -z, the AR5's is +z, so the sign flips.
  * lift, elbow, wrist_flex: axes agree at +y. No flip.
  * wrist_roll: on the leader the axis is -x while the forearm points +x, so
    positive is a *negative* roll about the tool axis. On the AR5 the axis is +z
    with the forearm along +z, so positive is a positive roll. The sign flips.

Two things the two arms cannot agree on, and which no calibration can fix:

  * Wrist order. The leader pitches then rolls (wrist_flex, then wrist_roll);
    the AR5 rolls then pitches (j4, then j5). So with the wrist rolled, the
    follower's pitch acts in a plane rotated by that roll. This is topological.
  * Wrist travel. Leader wrist_flex swings +-102 degrees; AR5 axis 6 is limited
    to +-50 by its soft limit (the datasheet allows +-60). Its scale is
    therefore about 0.49 -- half a turn of the leader's wrist gives a quarter on
    the follower.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

CALIBRATION_FILE = Path(__file__).resolve().parent / "joint_calibration.json"

# (leader joint, AR5 index, sign). Order is the leader's, base outward.
JOINT_PAIRS: Tuple[Tuple[str, int, float], ...] = (
    ("shoulder_pan", 0, -1.0),
    ("shoulder_lift", 1, +1.0),
    ("elbow_flex", 3, +1.0),
    # 主臂的 wrist_roll 驱动 **轴7**（索引 6），锁死的是轴3 和轴5。
    #
    # 主臂只有 5 个可用关节、AR5 有 7 个，必然锁死两个，选哪两个是取舍。
    # 这个选择来回改过三次，把账记在这儿免得再绕：
    #
    #   2026-09-22 之前   wrist_roll -> 轴7，锁 轴3 + 轴5
    #   2026-09-22        改成 wrist_roll -> 轴5，锁 轴3 + 轴7
    #                     理由：**轴7 软限位只有 ±50°，轴5 有 ±178°**，行程差 3.5 倍
    #   2026-09-23        按用户要求改回 wrist_roll -> 轴7，锁 轴3 + 轴5
    #
    # ⚠ 所以现在手腕自转的行程只有 100°（轴7 ±50°），是轴5 那版的 28%。
    # 主臂扳到头时从臂转不满，这不是故障。
    #
    # **换关节必须重新拟合缩放**，沿用另一根轴的缩放会让主臂转不到头就顶死，
    # 现象是「这个关节坏了」。selftest 有一条专门守这个。缩放由 configure()
    # 按当前这根轴的软限位现拟，所以不用手改 —— 前提是
    # joint_calibration.json 的 scale_overrides 里**没有** wrist_roll，
    # 2026-09-23 确认过确实没有。
    #
    # **符号没验过。** -1.0 是给轴5 标的，轴7 是法兰自转、转向未必一致。
    # 转反了用 `teleop_joint.py --flip wrist_roll --save` 存新符号。
    ("wrist_roll", 6, -1.0),
    ("wrist_flex", 5, +1.0),
)

# AR5 axis numbers as the datasheet labels them, for messages the operator reads.
DATASHEET_AXIS = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7}


@dataclass
class JointMapConfig:
    """How leader joint angles become AR5 joint angles."""

    # The AR5 posture the leader's centred pose maps to, radians. Found by
    # search: it is the pose that lets four of the five mapped joints run at 1:1
    # without ever hitting a soft limit, with the two locked joints at zero.
    # The tool sits at (0.41, 0.00, 0.68) -- arm up and forward -- and the
    # operator brings it down by lowering the leader's shoulder, which is the
    # direction the sign convention already gives.
    centre_joints: Tuple[float, ...] = (
        0.0, 0.244346, 0.0, 0.837758, 0.0, 0.0, 0.0
    )

    # Per-leader-joint overrides. Empty means "use the derived sign" and
    # "fit the scale to the available travel".
    sign_overrides: Dict[str, float] = field(default_factory=dict)
    scale_overrides: Dict[str, float] = field(default_factory=dict)

    # Applied on top of the fitted scales. Below 1.0 the follower moves less
    # than the leader everywhere, which is the easiest way to make a first run
    # gentle without changing the correspondence.
    scale_multiplier: float = 1.0

    # Fitted scales are never raised above this: 1.0 keeps a degree of leader
    # motion worth a degree of follower motion wherever there is room for it.
    max_scale: float = 1.0

    # AR5 joints with no leader counterpart, held at their centre value.
    # J3 和 J5。wrist_roll 改去驱动 J7 之后 J5 空出来。
    # 索引，不是轴号：索引 i 对应轴 i+1（见 DATASHEET_AXIS）。
    # (2, 6) = **锁轴3 和轴7**，放开轴5 给 wrist_roll 驱动。
    # 锁死的两根轴（索引）。**必须和 JOINT_PAIRS 互补**：映射到的 ∪ 锁死的
    # = 全部 7 根，且不重叠，selftest 有一条守着。
    #
    # 注意这个元组**本身不执行锁定** —— 真正的锁是「不在 JOINT_PAIRS 里的轴
    # 永远不被写」，它只用于报告和一致性校验。改 JOINT_PAIRS 忘了改这里，
    # 锁不会失效，但自检会红，而红的是一致性不是行为。
    #
    # 2026-09-23 从 (2, 6) 改成 (2, 4)：放开轴7、锁住轴5，跟着 wrist_roll 的
    # 目标轴一起动。
    locked_joints: Tuple[int, ...] = (2, 4)

    gripper_closed_deg: float = -55.5
    gripper_open_deg: float = 55.5

    # On engage the follower is somewhere else entirely, so the mapped target is
    # blended in over this long rather than demanded at once.
    ramp_seconds: float = 2.5

    # Drive the follower by how far the leader has moved since the clutch was
    # engaged, rather than by where the leader is.
    #
    # Absolute mapping ties reach to the centre posture: the scale is fitted so
    # the leader's whole swing stays inside the follower's soft limits from the
    # centre, so a centre near a limit -- which a low, table-facing posture
    # necessarily is -- shrinks every scale. Measured on this arm: elbow 0.57,
    # wrist 0.40, so most of the leader's travel buys very little arm motion.
    #
    # Relative mapping has no such coupling and runs 1:1. The trade is that
    # correspondence is no longer fixed: the same leader posture means different
    # arm postures depending on clutch history. That is also what makes it
    # useful -- release, recentre the leader, re-engage, and keep going, so the
    # reachable set stops being bounded by one armful of leader travel.
    #
    # CAVEAT, and it bites on this rig: `scale_overrides` wins over all of this.
    # `configure()` consults the overrides before it ever looks at `relative`,
    # so a saved calibration recorded in absolute mode keeps its fitted scales
    # here -- on this machine that is 0.40 to 1.00, not 1:1. That is deliberate
    # (an override the operator saved should mean what it says) but it does mean
    # "relative implies 1:1" is only true with no saved scales. `report()`
    # prints the scales actually in force; read that, not this comment.
    relative: bool = False

    # --- leader-side filtering (see OneEuro below) --------------------------
    #
    # Two things ride on the leader reading and neither is the operator:
    #
    #   * Quantisation, while moving. The STS3215 resolves 0.088 deg. At the 1:1
    #     scale relative mapping runs, one count is 1.53 mrad on the follower --
    #     small in position, but the tracker's feed-forward *differentiates* the
    #     goal, and one count across a 20 ms step reads as 0.077 rad/s, a tenth
    #     of the whole speed cap. A joint crossing count boundaries at 20 deg/s
    #     advances 4.5 counts a step, so the estimate carries a +-20% ripple.
    #
    #     Not at rest, though: measured 2026-09-03, five joints, 600 samples,
    #     nobody touching the leader -- zero counts of movement on all of them.
    #     Whatever the arm does while the operator holds still, the leader is
    #     not the source of it. `diag_shake.py --leader-only --still` re-checks.
    #   * Hand tremor. Physiological tremor is 8-12 Hz and the leader is
    #     back-drivable, so it is in the signal whenever someone is holding it.
    #     Absolute mapping used to attenuate it for free, since its fitted
    #     scales were 0.4-0.57; relative mapping runs 1:1 and does not.
    #
    # A fixed low-pass cannot separate either from the operator: cut low enough
    # to stop the tremor and the arm lags the hand. The 1-euro filter cuts low
    # only while the leader is nearly still -- where tremor and dither are the
    # entire signal -- and opens up as it moves, where lag is what hurts.
    # min_cutoff = 0 turns it off.
    filter_min_cutoff: float = 1.5     # Hz, the cutoff while the leader is still
    filter_beta: float = 0.05          # Hz gained per deg/s of leader speed
    filter_d_cutoff: float = 1.0       # Hz, low-pass on the speed estimate itself

    @classmethod
    def load(cls, path: Optional[Path] = None, **overrides) -> "JointMapConfig":
        """Build a config with any saved sign/scale corrections applied."""
        path = Path(path) if path is not None else CALIBRATION_FILE
        values: Dict[str, object] = {}
        if path.exists():
            try:
                saved = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning(f"[JointMapConfig] ignoring unreadable {path}: {exc}")
                saved = {}
            for key in ("sign_overrides", "scale_overrides", "centre_joints",
                        "gripper_closed_deg", "gripper_open_deg"):
                if key in saved:
                    values[key] = saved[key]
            if values:
                logger.info(
                    f"[JointMapConfig] loaded {path.name} "
                    f"(measured {saved.get('measured_at', 'unknown time')})"
                )

        values.update(overrides)
        if "centre_joints" in values:
            values["centre_joints"] = tuple(float(v) for v in values["centre_joints"])
        return cls(**values)  # type: ignore[arg-type]


class OneEuro:
    """Low-pass filter whose cutoff rises with the speed of the signal.

    Casiez, Roussel & Vogel, CHI 2012. One scalar channel; the mapper keeps one
    per leader joint.

    Both smoothing weights are computed from the *elapsed time*, not fixed per
    sample. That matters more here than the adaptive cutoff does: this loop's
    period is not constant -- it carries a serial read of six servos and,
    during collection, a camera -- and a fixed weight means a different amount
    of smoothing on every step, which is itself a source of the ripple this is
    meant to remove.
    """

    def __init__(self, min_cutoff: float = 1.5, beta: float = 0.05,
                 d_cutoff: float = 1.0):
        if min_cutoff < 0.0 or d_cutoff <= 0.0:
            raise ValueError(f"bad cutoffs: min={min_cutoff}, d={d_cutoff}")
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._x: Optional[float] = None
        self._dx = 0.0

    @staticmethod
    def alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return float(dt / (dt + tau))

    def reset(self) -> None:
        self._x = None
        self._dx = 0.0

    @property
    def speed(self) -> float:
        """Filtered rate of change, units per second."""
        return self._dx

    def __call__(self, x: float, dt: float) -> float:
        x = float(x)
        if self._x is None or dt <= 0.0 or not np.isfinite(dt):
            self._x = x
            return x
        # Speed first: the cutoff for this sample depends on it.
        a_d = self.alpha(self.d_cutoff, dt)
        self._dx += a_d * ((x - self._x) / dt - self._dx)
        cutoff = self.min_cutoff + self.beta * abs(self._dx)
        if cutoff <= 0.0:
            self._x = x
            return x
        a = self.alpha(cutoff, dt)
        self._x += a * (x - self._x)
        return self._x


class JointMapper:
    """Leader joint angles in, AR5 joint angles out."""

    def __init__(self, config: Optional[JointMapConfig] = None):
        self.config = config or JointMapConfig()
        self.centre = np.asarray(self.config.centre_joints, dtype=np.float64)

        self._signs: Dict[str, float] = {}
        self._scales: Dict[str, float] = {}
        self._leader_half: Dict[str, float] = {}
        self._joint_lo: Optional[np.ndarray] = None
        self._joint_hi: Optional[np.ndarray] = None

        self._engaged = False
        self._ramp_from: Optional[np.ndarray] = None
        self._ramp_t0: Optional[float] = None
        # One filter per leader joint, and the timestamp they share. Built even
        # when filtering is off so `--min-cutoff 0` and a restart differ in
        # nothing but the cutoff.
        self._filters: Dict[str, OneEuro] = {
            name: OneEuro(self.config.filter_min_cutoff, self.config.filter_beta,
                          self.config.filter_d_cutoff)
            for name, _, _ in JOINT_PAIRS
        }
        self._filter_t: Optional[float] = None
        # Relative mode anchors: where the arm was and where the leader was at
        # the moment the clutch closed. Both are re-taken on every engage.
        self._rel_arm: Optional[np.ndarray] = None
        self._rel_leader: Optional[Dict[str, float]] = None

        self.stats = {"steps": 0, "clamped": 0, "clutch_cycles": 0, "missing": 0}

    # ------------------------------------------------------------------ setup

    def configure(self, leader_half_travel: Dict[str, float], joint_min, joint_max,
                  margin: float = 0.0) -> None:
        """Fit the per-joint scales to the travel actually available.

        ``leader_half_travel`` is degrees either side of zero, from the lerobot
        calibration. The scale is the largest that keeps the leader's full swing
        inside the follower joint's soft limits from the centre posture, so a
        mapped goal is never clipped and the operator never finds a joint that
        stops responding partway through its range.
        """
        lo = np.asarray(joint_min, dtype=np.float64) + margin
        hi = np.asarray(joint_max, dtype=np.float64) - margin
        self._joint_lo, self._joint_hi = lo, hi
        self._leader_half = dict(leader_half_travel)

        centre = np.clip(self.centre, lo, hi)
        if not np.allclose(centre, self.centre):
            outside = [i for i in range(len(centre)) if not np.isclose(centre[i], self.centre[i])]
            raise ValueError(
                f"centre_joints is outside the soft limits at joint(s) {outside}: "
                f"{np.rad2deg(self.centre).round(1).tolist()} deg against "
                f"[{np.rad2deg(lo).round(1).tolist()}, {np.rad2deg(hi).round(1).tolist()}]"
            )

        # Applied to fitted and overridden scales alike: it is a "move the whole
        # follower more gently" knob, and one that skipped saved calibrations
        # would do nothing on exactly the runs that have one.
        multiplier = float(self.config.scale_multiplier)

        for name, index, derived_sign in JOINT_PAIRS:
            self._signs[name] = float(self.config.sign_overrides.get(name, derived_sign))

            # Checked before `relative` on purpose: a scale the operator saved
            # is a statement of intent and outranks the mode's default. The
            # consequence is that a calibration recorded in absolute mode keeps
            # its fitted scales in relative mode too, so `--relative` alone is
            # not 1:1 wherever such a file exists. `report()` prints what is
            # actually in force.
            if name in self.config.scale_overrides:
                self._scales[name] = float(self.config.scale_overrides[name]) * multiplier
                continue

            half = np.deg2rad(self._leader_half.get(name, 180.0))
            if half <= 1e-9:
                self._scales[name] = 0.0
                continue
            if self.config.relative:
                # Nothing to fit: the anchor moves with the clutch, so the
                # leader's swing is not required to fit around a fixed centre.
                # The soft limits still clamp, they just stop dictating reach.
                self._scales[name] = float(self.config.max_scale * multiplier)
                continue
            room = min(hi[index] - self.centre[index], self.centre[index] - lo[index])
            fitted = min(self.config.max_scale, max(0.0, room) / half)
            self._scales[name] = float(fitted * multiplier)

    def report(self) -> str:
        """The mapping table, for printing before anything moves."""
        lines = [
            f"  {'leader':16s} {'AR5':>4s} {'轴':>3s} {'sign':>5s} {'scale':>6s} "
            f"{'leader ±deg':>12s} {'follower ±deg':>14s}"
        ]
        for name, index, _ in JOINT_PAIRS:
            half = self._leader_half.get(name, float("nan"))
            scale = self._scales.get(name, float("nan"))
            lines.append(
                f"  {name:16s} {'j' + str(index):>4s} {DATASHEET_AXIS[index]:>3d} "
                f"{self._signs.get(name, float('nan')):>+5.0f} {scale:>6.3f} "
                f"{half:>12.1f} {half * scale:>14.1f}"
            )
        locked = ", ".join(
            f"j{i} (轴{DATASHEET_AXIS[i]}) at {np.rad2deg(self.centre[i]):+.1f}deg"
            for i in self.config.locked_joints
        )
        lines.append(f"  locked: {locked}")
        cut = self.config.filter_min_cutoff
        if cut > 0.0:
            lines.append(
                f"  filter: 1-euro, {cut:.2f} Hz at rest, +{self.config.filter_beta:.3f} Hz "
                f"per deg/s of leader speed"
            )
            lines.append(f"          (lag at rest {1000.0 / (2 * np.pi * cut):.0f} ms, "
                         "falling to nothing as you move)")
        else:
            lines.append("  filter: OFF -- encoder dither and hand tremor go straight through")
        if self.config.relative:
            lines.append("  mapping: RELATIVE -- the arm follows how far the leader has moved "
                         "since engaging.")
            lines.append("           Release and re-engage to recentre the leader and keep going;"
                         " the 'follower ±deg'")
            lines.append("           column is per clutch cycle, not a limit.")
        else:
            lines.append("  mapping: absolute -- leader posture maps to arm posture through the "
                         "centre.")
        return "\n".join(lines)

    # ------------------------------------------------------------------ clutch

    def engage(self, q_now: Sequence[float]) -> None:
        """Start driving, ramping in from wherever the follower is now."""
        self._ramp_from = np.asarray(q_now, dtype=np.float64).copy()
        self._ramp_t0 = time.perf_counter()
        if self.config.relative:
            # Anchor the arm here; the leader anchor is taken on the next
            # reading, since the clutch is toggled before one is available.
            self._rel_arm = np.asarray(q_now, dtype=np.float64).copy()
            self._rel_leader = None
        if not self._engaged:
            self.stats["clutch_cycles"] += 1
        self._engaged = True

    def disengage(self) -> None:
        self._engaged = False
        self._ramp_from = None
        self._rel_arm = None
        self._rel_leader = None
        self._ramp_t0 = None

    @property
    def engaged(self) -> bool:
        return self._engaged

    def ramp_fraction(self) -> float:
        """How much of the mapped target is being applied, 0 to 1."""
        if self._ramp_t0 is None or self.config.ramp_seconds <= 0:
            return 1.0
        t = (time.perf_counter() - self._ramp_t0) / self.config.ramp_seconds
        if t >= 1.0:
            return 1.0
        # Smoothstep: starts and ends at zero velocity, so the ramp neither
        # jerks at engage nor overshoots as it hands over to the operator.
        t = max(0.0, t)
        return float(t * t * (3.0 - 2.0 * t))

    # ------------------------------------------------------------------ mapping

    def gripper_command(self, joints_deg: Dict[str, float]) -> float:
        """Leader trigger angle -> opening in [0, 1], 0 = closed."""
        raw = joints_deg.get("gripper")
        if raw is None:
            return 1.0
        closed, opened = self.config.gripper_closed_deg, self.config.gripper_open_deg
        if abs(opened - closed) < 1e-6:
            return 1.0
        return float(np.clip((raw - closed) / (opened - closed), 0.0, 1.0))

    def filtered(self, joints_deg: Dict[str, float]) -> Dict[str, float]:
        """Run the leader reading through the per-joint filters.

        Called on every reading, engaged or not, for two reasons: the released
        status line should show the same numbers the arm would be driven by,
        and a filter that only ran while engaged would be cold at the moment of
        engaging -- which is the one moment its output is taken as an anchor.
        """
        # One timestamp for the whole reading: the six servos are read back to
        # back off one serial bus, so timing them individually would only
        # record the bus order.
        now = time.perf_counter()
        dt = 0.0 if self._filter_t is None else now - self._filter_t
        self._filter_t = now

        if self.config.filter_min_cutoff <= 0.0:
            return dict(joints_deg)

        out = dict(joints_deg)
        for name, _, _ in JOINT_PAIRS:
            value = joints_deg.get(name)
            if value is not None:
                out[name] = self._filters[name](value, dt)
        return out

    def leader_speed(self) -> Dict[str, float]:
        """Each filter's own speed estimate, deg/s. Diagnostics only."""
        return {name: self._filters[name].speed for name, _, _ in JOINT_PAIRS}

    def target_joints(self, joints_deg: Dict[str, float]) -> np.ndarray:
        """The absolute AR5 posture this leader reading maps to.

        Independent of the clutch and of where the follower currently is, so it
        can be printed while released to check the correspondence without
        anything moving.
        """
        if self._joint_lo is None:
            raise RuntimeError("JointMapper used before configure()")

        # Before the early return below, so the filters keep running while
        # released and are warm when the clutch closes.
        joints_deg = self.filtered(joints_deg)

        relative = self.config.relative
        if relative and self._rel_arm is None:
            # Released, or engage has not been seen yet: nothing to move from.
            return self.centre.copy() if self._ramp_from is None else self._ramp_from.copy()

        # Take the leader anchor on the first reading after the clutch closes,
        # so the first commanded posture equals the anchored one and engaging
        # never moves the arm. Anchored on the *filtered* reading, which is what
        # the mapping below consumes -- anchoring on the raw one would make the
        # arm step by the filter's lag the instant the clutch closed.
        if relative and self._rel_leader is None:
            self._rel_leader = {k: float(v) for k, v in joints_deg.items()}

        q = self._rel_arm.copy() if relative else self.centre.copy()
        for name, index, _ in JOINT_PAIRS:
            value = joints_deg.get(name)
            if value is None:
                # A dropped serial reply must hold the joint, not read as zero.
                self.stats["missing"] += 1
                continue
            if relative:
                anchor = self._rel_leader.get(name)
                if anchor is None:
                    continue
                q[index] = self._rel_arm[index] + self._signs[name] * self._scales[name] * np.deg2rad(value - anchor)
            else:
                q[index] = self.centre[index] + self._signs[name] * self._scales[name] * np.deg2rad(value)

        clamped = np.clip(q, self._joint_lo, self._joint_hi)
        if not np.allclose(clamped, q):
            self.stats["clamped"] += 1
        return clamped

    def compute(self, joints_deg: Dict[str, float], q_now: Sequence[float]) -> np.ndarray:
        """The posture to command this step, blended in if the ramp is running."""
        target = self.target_joints(joints_deg)
        self.stats["steps"] += 1

        if not self._engaged:
            return np.asarray(q_now, dtype=np.float64).copy()

        alpha = self.ramp_fraction()
        if alpha >= 1.0 or self._ramp_from is None:
            return target
        return (1.0 - alpha) * self._ramp_from + alpha * target
