"""主从臂**笛卡尔**映射：主臂末端怎么动，从臂末端就怎么动，7 个轴全交给逆解。

和 `joint_mapping.JointMapper` **接口一样**（engage / disengage / compute /
gripper_command / configure / report / centre / stats / config），collect 里换一个
对象就能切过去，P 键、回 0 位、限速、跳变门、地板检查、诊断、存盘全部照旧。

为什么要它
---------
主臂 SO-101 只有 5 个可用关节，AR5 有 7 个。关节一对一映射必然锁死两个轴
（现在是轴3 + 轴5），肘的形态永远固定在一种，轴7 那 ±50° 的行程也躲不开。
笛卡尔映射传的是**工具位姿**，7 轴对 6 维任务是冗余的，逆解会把运动分给
全部 7 个轴，冗余的那一维在零空间里调肘。

⚠ **它不会凭空多出自由度。** 主臂 5 个关节摆出来的位姿落在一个 5 维流形上，
6 个数里有一个组合被主臂的几何锁死 —— 你能独立指挥的还是 5 个。多出来的是
「7 个轴都参与」，不是「多了一个能控的方向」。

锚定式，不是逐帧累加
-------------------
按空格（或 P 重锚）那一刻记下两个锚点：主臂末端位姿、从臂末端位姿。之后

    目标位置 = 从臂锚点位置 + M · (主臂现在位置 − 主臂锚点位置) · 缩放
    目标姿态 = exp(M · 主臂转过的角度) · 从臂锚点姿态

`leader_mapping.LeaderMapper` 原来是**逐帧算增量再累加**。那种写法只要中间有一帧
被限速、被跳变门拒、或者逆解没解出来，那一帧的增量就丢了，目标和臂从此越漂
越远。锚定式每帧都从锚点重算，丢一帧不会累积。和关节相对模式同一个思路。

受控点在法兰
-----------
工具偏置 `tcp-z` 归过 0，所以转动是绕**法兰中心**转的，夹爪指尖会跟着甩。
等量出夹爪加转接板的长度，填进 tcp 就是绕指尖转了。
（2026-10-31：`CartesianMapConfig.tcp_z` 就是它，OmniPicker 量出来 0.13 m；默认仍是 0。）

两种逆解（2026-10-31）
---------------------
`ik="dls"`（默认，原来那套）够不到就整帧作废、臂卡住；姿态和位置一样硬跟，
主臂手腕一转就容易要出 AR5 手腕 ±50° 给不了的姿态。`ik="weighted"` 照
SO101-OpenArm_teleop（github.com/Vector-xxr/SO101-OpenArm_teleop，只在 MuJoCo 里
跑通过）的 mink 写法移植到我们的 DH 上：位置为主、姿态放软、限位当约束、每帧不要求
收敛，外加末端死区和每帧限幅。见 `AR5Kinematics.ik_weighted`。

`ik="sdk"`（同日加，试验模式，只用来对比手感）：映射完全一样，只是不在本地解，把法兰目标
放在 `sdk_target` 里，collect 交给 `movel_stream.MoveLStreamer` 发排队式 MoveL，控制器解逆解。
"""

from __future__ import annotations

import json
import logging
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

from .joint_mapping import JointMapConfig, JointMapper
from .kinematics import AR5Kinematics
from .leader_mapping import MappingConfig, mapping_matrix
from .transforms import axis_angle_to_matrix, matrix_to_axis_angle

logger = logging.getLogger(__name__)

ARM_CHAIN = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")


# ────────────────────────────────────────────────────────────── 主臂正运动学

def _rpy(r, p, y) -> np.ndarray:
    """URDF 的 rpy：R = Rz(y) · Ry(p) · Rx(r)。"""
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr],
    ])


def _axis_rot(axis: np.ndarray, q: float) -> np.ndarray:
    """绕单位轴转 q 弧度（Rodrigues）。"""
    a = axis / (np.linalg.norm(axis) or 1.0)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(q) * K + (1 - np.cos(q)) * (K @ K)


class So101FK:
    """SO-101 主臂的正运动学，直接从 URDF 读，**不用 placo**。

    placo 在这里只被用来做这一件事，而它是编译包、只有 py3.10 的轮子，
    装进 dexgraspvla（py3.11）有 numpy ABI 风险，会威胁到这个环境里配死的 torch。
    一条 5 关节串联链的正运动学三十行就够，就自己写了。

    和 placo 在 vel-ar5 环境里交叉验证过（见 selftest），关节角约定也和原来
    `LeaderMapper` 一样：主臂读数的度数直接换成弧度喂进 URDF 的关节。
    """

    def __init__(self, urdf: str, tip: str = "gripper_frame_link"):
        root = ET.parse(urdf).getroot()
        by_child = {}
        for j in root.findall("joint"):
            o = j.find("origin")
            xyz = np.array([float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()])
            rpy = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
            ax = j.find("axis")
            axis = np.array([float(v) for v in ax.get("xyz").split()]) if ax is not None else np.zeros(3)
            by_child[j.find("child").get("link")] = dict(
                name=j.get("name"), type=j.get("type"), parent=j.find("parent").get("link"),
                T=self._T(_rpy(*rpy), xyz), axis=axis)

        # 从末端往根走，再倒过来，就是从基座到末端的链
        chain, link = [], tip
        while link in by_child:
            chain.append(by_child[link])
            link = by_child[link]["parent"]
        self.chain = chain[::-1]
        self.base = link
        names = [j["name"] for j in self.chain if j["type"] != "fixed"]
        if tuple(names) != ARM_CHAIN:
            raise ValueError(f"URDF 里从 {self.base} 到 {tip} 的关节是 {names}，期望 {ARM_CHAIN}")

    @staticmethod
    def _T(R, p) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R, p
        return T

    def fk(self, joints_deg: Dict[str, float]) -> np.ndarray:
        """主臂关节角（度）-> 4x4 末端位姿，主臂基座系。"""
        T = np.eye(4)
        for j in self.chain:
            T = T @ j["T"]
            if j["type"] != "fixed":
                q = np.deg2rad(float(joints_deg.get(j["name"], 0.0)))
                T = T @ self._T(_axis_rot(j["axis"], q), np.zeros(3))
        return T


def default_urdf() -> str:
    from atom_robot import VEL_REPO  # noqa: PLC0415
    return str(VEL_REPO.parent / "assets" / "so101" / "so101_fk.urdf")


def default_kinematics(cfg) -> AR5Kinematics:
    """AR5 的运动学。用控制器标定过的 DH（ar5_dh.json），离线和 mock 下都能用。"""
    dh = json.loads((Path(__file__).resolve().parent / "ar5_dh.json").read_text())["dh"]
    return AR5Kinematics(np.array(dh).reshape(-1, 4), cfg.num_joints, cfg.joint_min, cfg.joint_max)


# ────────────────────────────────────────────────────────────── 映射

@dataclass
class CartesianMapConfig:
    position_scale: float = 1.0
    rotation_scale: float = 1.0
    # 保留哪几个转动轴（**AR5 基座系**）。"xyz" 全开；"z" 只留绕竖直轴的自转。
    rotation_axes: str = "xyz"
    # 锚定式在接合瞬间目标就等于臂当前位姿，不会跳，所以不需要渐入。
    # 留这个字段是因为 collect 的 P 键提示会读它。
    ramp_seconds: float = 0.0

    # 逆解用哪个（2026-10-31 加，**默认 dls = 原来那套，一点没变**）：
    #   "dls"       AR5Kinematics.ik：迭代到精确收敛，够不到就整帧作废、臂停在上一个解
    #   "weighted"  AR5Kinematics.ik_weighted：照 SO101-OpenArm_teleop 的 mink 写法移植，
    #               位置为主、姿态放软、限位和每帧关节步长当约束、每帧 8 步不要求收敛，
    #               永远有输出 —— 够不到时臂贴着可达边界走，不卡死
    #   "sdk"       不在本地解：只给出法兰目标位姿（`sdk_target`），由 collect 交给
    #               movel_stream 发排队式 MoveL，控制器自己解逆解。试验模式，只用来对比手感
    ik: str = "dls"
    # 受控点沿法兰 z 往外偏多少米。0 = 法兰中心（原来的行为）。
    # OmniPicker + 转接板到指尖 0.13（用户 10-31 量的）。
    tcp_z: float = 0.0
    # 下面几个只在 weighted 下用，数值照参考项目 config/default.yaml。
    position_cost: float = 50.0
    orientation_cost: float = 0.15
    posture_cost: float = 0.02
    damping_cost: float = 0.2
    ik_iters: int = 8
    # 末端死区：目标比上一帧的目标变得少于这么多就不动，压主臂编码器噪声
    deadband_m: float = 0.003
    deadband_rad: float = 0.03
    # 末端每帧最多走多少（50 Hz 下 8 mm = 0.4 m/s），挡主臂读数跳一下就把臂甩出去
    max_ee_step_m: float = 0.008
    max_ee_step_rad: float = 0.06
    # 解出来离目标还差这么多就记一帧「没跟上」，给 TrackingWatch 报
    lag_report_m: float = 0.005
    # 逐帧诊断记录写到哪个目录（collect 的 --cart-log）。None = 不记（默认）。
    # 每次接合到松开算一段，松开 / 重锚时存成一个 npz，按 R 丢掉的那条也留着。
    log_dir: Optional[str] = None

    def __post_init__(self):
        if self.ik not in ("dls", "weighted", "sdk"):
            raise ValueError(f"ik 只能是 dls、weighted 或 sdk，给的是 {self.ik!r}")
        if self.tcp_z < 0 or self.tcp_z > 0.4:
            raise ValueError(f"tcp_z 要在 0~0.4 m，给的是 {self.tcp_z}")


class CartesianMapper:
    """主臂末端位姿（相对锚点的变化）-> 从臂目标位姿 -> 逆解 -> 7 个关节角。"""

    def __init__(self, arm_cfg, workspace_min, workspace_max,
                 config: Optional[CartesianMapConfig] = None,
                 axis_cfg: Optional[MappingConfig] = None,
                 joint_cfg: Optional[JointMapConfig] = None,
                 urdf: Optional[str] = None):
        self.config = config or CartesianMapConfig()
        # 滤波、夹爪映射、0 位都沿用关节那一套：夹爪方向是 9/23 实测校过的，
        # 0 位（centre_joints）是回位要去的地方，两边不该各有一份。
        self._joint = JointMapper(joint_cfg or JointMapConfig.load(relative=True))
        self.centre = self._joint.centre
        self._posture = self.centre.copy()

        self._leader = So101FK(urdf or default_urdf())
        self._kin = default_kinematics(arm_cfg)
        self._ws_lo = np.asarray(workspace_min, dtype=np.float64)
        self._ws_hi = np.asarray(workspace_max, dtype=np.float64)

        # 主臂基座系 -> AR5 基座系的轴映射，读 leader_calibration.json
        # （calibrate_mapping 写的）。镜像时转动向量要多一个负号。
        axis_cfg = axis_cfg or MappingConfig.load()
        self._M = mapping_matrix(axis_cfg.axis_permutation, axis_cfg.axis_signs)
        self._parity = float(np.sign(np.linalg.det(self._M)) or 1.0)

        # 受控点：法兰 @ 工具。tcp_z=0 时是单位阵，和原来完全一样
        self._tool = np.eye(4)
        self._tool[2, 3] = self.config.tcp_z
        self._tool_inv = np.linalg.inv(self._tool)
        # weighted 每帧关节步长的上界，取 env 限速的 98%，免得解出来又被等比缩小。
        # 不取正好相等：解贴在上界上，舍入多出 1e-17 也会被 env 判成超限（config 里
        # joint_goal_step 的注释记过同一个坑），离线对比里白白多记了几百帧「被限速」。
        self._step_limit = 0.98 * float(getattr(arm_cfg, "joint_goal_step", 0.0)) or None
        # 重新对齐种子的阈值：上一帧的解离臂的设定点比这还远，说明中间被别的东西
        # （地板、跳变门、回位）拦过，从设定点重新起步
        self._resync = 0.5 * float(getattr(arm_cfg, "joint_jump_max", 0.15))
        if self.config.ik == "weighted":
            # ik_weighted 里懒加载 scipy，第一次要 100 多毫秒 —— 别让它落在接合后的第一帧
            import scipy.optimize  # noqa: F401, PLC0415

        self._engaged = False
        self._lead0: Optional[np.ndarray] = None       # 主臂锚点 4x4
        self._arm0: Optional[np.ndarray] = None        # 从臂锚点 4x4（受控点的位姿）
        self._last_q: Optional[np.ndarray] = None
        self._last_target: Optional[np.ndarray] = None  # weighted：上一帧真正用的目标
        self._last_ik_reason = ""
        self.stats = {"steps": 0, "clamped": 0, "missing": 0, "ik_fail": 0,
                      "clutch_cycles": 0, "ik_ms_max": 0.0,
                      "ik_lag": 0, "ik_lag_max_mm": 0.0, "seed_resync": 0}
        self._diag: dict = {}
        # ik="sdk"：这一帧要发给控制器的**法兰**目标（4x4），没有就是 None。每帧 compute 开头清掉
        self.sdk_target: Optional[np.ndarray] = None
        # 逐帧诊断：内存里攒一段，松开时才写盘，回路里不碰磁盘
        self._log: Optional[list] = [] if self.config.log_dir else None
        self._log_started = time.strftime("%m%d_%H%M%S")
        self._log_seq = 0
        self.log_files: list = []

    # ── 和 JointMapper 同名的接口 ────────────────────────────────────
    def configure(self, leader_half_travel, joint_min, joint_max, margin: float = 0.0) -> None:
        self._joint.configure(leader_half_travel, joint_min, joint_max, margin=margin)
        lo = np.asarray(joint_min, dtype=np.float64) + margin
        hi = np.asarray(joint_max, dtype=np.float64) - margin
        # 逆解在 margin 以内找解，和驱动 / step_joints 用同一个边界
        self._kin.joint_min, self._kin.joint_max = lo, hi
        self._kin.joint_mid = (lo + hi) / 2.0
        self._kin.joint_range = np.maximum(hi - lo, 1e-9)
        # weighted 的姿势项往 0 位拉（参考项目拉向启动姿势，这里的启动姿势就是 0 位）
        self._posture = np.clip(self.centre, lo, hi)

    def gripper_command(self, joints_deg: Dict[str, float]) -> float:
        return self._joint.gripper_command(joints_deg)

    @property
    def engaged(self) -> bool:
        return self._engaged

    def engage(self, q_now: Sequence[float]) -> None:
        """接合 / 重锚：从臂锚点取当前设定点的正运动学，主臂锚点下一帧再取。"""
        q_now = np.asarray(q_now, dtype=np.float64)
        self.flush_log()                    # 重锚也算新的一段
        self._arm0 = self._kin.fk(q_now) @ self._tool
        self._lead0 = None
        self._last_q = q_now.copy()
        self._last_target = self._arm0.copy()
        if not self._engaged:
            self.stats["clutch_cycles"] += 1
        self._engaged = True

    def disengage(self) -> None:
        self._engaged = False
        self._lead0 = self._arm0 = None
        self.flush_log()

    # ── 逐帧诊断记录（--cart-log）────────────────────────────────────
    def _log_frame(self, filtered, q_now, target_raw, q_out) -> None:
        d = self._diag or {}
        used = d.get("target_used", target_raw)
        self._log.append(np.concatenate([
            [time.perf_counter()],
            [float(filtered[n]) for n in ARM_CHAIN],                               # 1:6 主臂（滤波后，度）
            target_raw[:3, 3], matrix_to_axis_angle(target_raw[:3, :3]),         # 6:12 映射出的目标
            used[:3, 3], matrix_to_axis_angle(used[:3, :3]),                     # 12:18 死区/限幅后真正用的
            q_now, d.get("seed", q_now), q_out,                                  # 18:25 25:32 32:39
            [d.get("pos_error", np.nan), d.get("rot_error", np.nan), d.get("ms", np.nan),
             float(d.get("ok", True)), float(d.get("held", False)), float(d.get("step_clipped", False)),
             float(d.get("resync", False)), float(d.get("n_bounded", 0))],       # 39:47
        ]))

    LOG_COLUMNS = ("t", "leader[5]", "target_raw[6]", "target_used[6]", "q_now[7]", "seed[7]", "q_out[7]",
                   "pos_error", "rot_error", "ik_ms", "ok", "held", "step_clipped", "resync", "n_bounded")

    def flush_log(self) -> None:
        """把攒着的一段写成 npz。松开 / 重锚时调；没开 --cart-log 或这段是空的就什么都不做。"""
        if not self._log:
            return
        rows, self._log = np.array(self._log), []
        out = Path(self.config.log_dir).expanduser()
        path = out / f"cart_{self._log_started}_{self._log_seq:03d}.npz"
        self._log_seq += 1
        c = self.config
        try:
            out.mkdir(parents=True, exist_ok=True)
            np.savez(path, rows=rows, columns=np.array(self.LOG_COLUMNS),
                     tool=self._tool, ik=c.ik, position_scale=c.position_scale,
                     rotation_scale=c.rotation_scale, deadband=(c.deadband_m, c.deadband_rad),
                     max_ee_step=(c.max_ee_step_m, c.max_ee_step_rad), step_limit=self._step_limit or 0.0)
        except OSError as exc:
            # 在采集回路里被调（松开 / 重锚），诊断记录写不出来不能把采集带崩
            logger.warning("逐帧诊断记录没写出来（%s）：%s", path, exc)
            return
        self.log_files.append(str(path))

    def compute(self, joints_deg: Dict[str, float], q_now: Sequence[float]) -> np.ndarray:
        q_now = np.asarray(q_now, dtype=np.float64)
        self.stats["steps"] += 1
        self.sdk_target = None
        # 滤波和关节那套一样，先滤再做正运动学 —— 滤波器在松开时也跑，接合时是热的
        filtered = self._joint.filtered(joints_deg)
        if not self._engaged or self._arm0 is None:
            return q_now.copy()
        if any(n not in filtered for n in ARM_CHAIN):
            self.stats["missing"] += 1
            return (self._last_q if self._last_q is not None else q_now).copy()

        T_lead = self._leader.fk(filtered)
        if self._lead0 is None:
            self._lead0 = T_lead           # 第一帧就是锚点，目标 = 当前，不跳
            return q_now.copy()

        dp = self._M @ (T_lead[:3, 3] - self._lead0[:3, 3]) * self.config.position_scale
        dr = self._parity * (self._M @ matrix_to_axis_angle(T_lead[:3, :3] @ self._lead0[:3, :3].T))
        dr = dr * self.config.rotation_scale
        for i, ax in enumerate("xyz"):
            if ax not in self.config.rotation_axes:
                dr[i] = 0.0

        target = np.eye(4)
        target[:3, :3] = axis_angle_to_matrix(dr) @ self._arm0[:3, :3]
        p = self._arm0[:3, 3] + dp
        # 笛卡尔路径要看工作空间盒（关节路径不看）。z 下界是台面保护。
        clipped = np.clip(p, self._ws_lo, self._ws_hi)
        if not np.allclose(clipped, p):
            self.stats["clamped"] += 1
        target[:3, 3] = clipped

        if self.config.ik == "weighted":
            q = self._solve_weighted(target, q_now)
        elif self.config.ik == "sdk":
            # 不解逆解：受控点目标换回法兰目标交出去，关节原样返回（collect 这一路不发关节）
            self.sdk_target = target @ self._tool_inv
            self._diag = {"target_used": target, "seed": q_now, "ms": 0.0, "ok": True}
            q = q_now.copy()
        else:
            q = self._solve_dls(target, q_now)
        if self._log is not None:
            self._log_frame(filtered, q_now, target, q)
        return q

    def _solve_dls(self, target: np.ndarray, q_now: np.ndarray) -> np.ndarray:
        """原来那条路，逻辑一点没改（10-31 只是从 compute 里挪出来，好让两条路都能记诊断）。"""
        seed = self._last_q if self._last_q is not None else q_now
        t0 = time.perf_counter()
        # 原来的解法求的是法兰；有工具偏置时把受控点目标换回法兰目标，精确收敛下两者等价
        q, info = self._kin.ik(target @ self._tool_inv, seed)
        ms = (time.perf_counter() - t0) * 1000
        self.stats["ik_ms_max"] = max(self.stats["ik_ms_max"], ms)
        self._diag = {"target_used": target, "seed": seed, "ms": ms, "ok": q is not None,
                      "pos_error": info.get("pos_error", np.nan), "rot_error": info.get("rot_error", np.nan)}
        if q is None:
            # 够不到：停在上一个够得到的地方，别瞎动。TrackingWatch 会报出来。
            self.stats["ik_fail"] += 1
            self._last_ik_reason = info.get("reason", "")
            return (self._last_q if self._last_q is not None else q_now).copy()
        self._last_q = q
        return q

    def _solve_weighted(self, target: np.ndarray, q_now: np.ndarray) -> np.ndarray:
        """weighted 那条路：末端死区 + 每帧限幅（参考项目 se3.hold_if_small / clip_step），
        再走 ik_weighted。永远有输出，不往 ik_fail 里记。"""
        c = self.config
        prev = self._last_target if self._last_target is not None else target
        held = _hold_if_small(prev, target, c.deadband_m, c.deadband_rad)
        stepped = _clip_step(prev, held, c.max_ee_step_m, c.max_ee_step_rad)
        self._last_target = stepped
        diag = {"held": held is not target, "step_clipped": not np.allclose(stepped, held)}
        target = stepped

        seed = self._last_q if self._last_q is not None else q_now
        diag["resync"] = float(np.max(np.abs(seed - q_now))) > self._resync
        if diag["resync"]:
            self.stats["seed_resync"] += 1
            seed = q_now
        t0 = time.perf_counter()
        q, info = self._kin.ik_weighted(
            target, seed, q_posture=self._posture, step_limit=self._step_limit, tool=self._tool,
            iters=c.ik_iters, position_cost=c.position_cost, orientation_cost=c.orientation_cost,
            posture_cost=c.posture_cost, damping_cost=c.damping_cost)
        ms = (time.perf_counter() - t0) * 1000
        self.stats["ik_ms_max"] = max(self.stats["ik_ms_max"], ms)
        self._diag = dict(diag, target_used=target, seed=seed, ms=ms, ok=bool(np.all(np.isfinite(q))),
                          pos_error=info["pos_error"], rot_error=info["rot_error"],
                          n_bounded=len(info["bounded"]))
        if not np.all(np.isfinite(q)):
            # 数值炸了（参考项目也防这个）：原地保持
            self.stats["ik_fail"] += 1
            self._last_ik_reason = "non-finite solution"
            return seed.copy()
        if info["pos_error"] > c.lag_report_m:
            self.stats["ik_lag"] += 1
            self.stats["ik_lag_max_mm"] = max(self.stats["ik_lag_max_mm"], info["pos_error"] * 1000)
        self._last_q = q
        return q

    def report(self) -> str:
        c = self.config
        if c.ik == "weighted":
            ik_line = (f"    逆解 **weighted**（位置 {c.position_cost:g} / 姿态 {c.orientation_cost:g} / "
                       f"姿势 {c.posture_cost:g} / 阻尼 {c.damping_cost:g}，每帧 {c.ik_iters} 步不要求收敛）\n"
                       f"    末端死区 {c.deadband_m * 1000:.0f} mm / {np.rad2deg(c.deadband_rad):.1f}°   "
                       f"每帧最多 {c.max_ee_step_m * 1000:.0f} mm / {np.rad2deg(c.max_ee_step_rad):.1f}°   "
                       f"关节每帧最多 {self._step_limit} rad")
        elif c.ik == "sdk":
            ik_line = ("    逆解 **sdk**（不在本地解：发排队式 MoveL，控制器自己解逆解、做插补）"
                       "   ⚠ 试验模式，只用来对比手感")
        else:
            ik_line = "    逆解 dls（迭代到精确收敛，够不到就停在上一个解）"
        tcp_line = (f"    受控点沿法兰 z 偏 {c.tcp_z * 1000:.0f} mm（绕这一点转）" if c.tcp_z > 0
                    else "    受控点在法兰中心（tcp 归 0），转腕时夹爪指尖会甩")
        if c.log_dir:
            tcp_line += f"\n    逐帧诊断记录 -> {c.log_dir}"
        return "\n".join([
            "  映射：**笛卡尔**（主臂末端位姿 -> 逆解 -> 7 轴全动）",
            f"    位置缩放 x{c.position_scale}   转动缩放 x{c.rotation_scale}   转动轴 {c.rotation_axes}",
            ik_line,
            f"    主臂->AR5 轴映射 M =\n" + "\n".join("      " + str(r) for r in self._M.astype(int).tolist()),
            f"    工作空间盒 {self._ws_lo.tolist()} ~ {self._ws_hi.tolist()} m",
            tcp_line,
        ])


# ────────────────────────────────────────────────────────────── 末端指令预处理
# 照 SO101-OpenArm_teleop 的 se3.hold_if_small / clip_step。只给 weighted 用。

def _rot_angle(R: np.ndarray) -> float:
    return float(np.arccos(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)))


def _hold_if_small(T_prev: np.ndarray, T_next: np.ndarray, dead_m: float, dead_rad: float) -> np.ndarray:
    """位移和转角都低于死区就保持上一帧的目标。"""
    if not np.all(np.isfinite(T_next)):
        return T_prev.copy()
    dp = float(np.linalg.norm(T_next[:3, 3] - T_prev[:3, 3]))
    th = _rot_angle(T_next[:3, :3] @ T_prev[:3, :3].T)
    if dp < dead_m and th < dead_rad:
        return T_prev.copy()
    return T_next


def _clip_step(T_prev: np.ndarray, T_next: np.ndarray, max_m: float, max_rad: float) -> np.ndarray:
    """一帧的平移、转角各自按比例截到上限，方向不变。"""
    dp = T_next[:3, 3] - T_prev[:3, 3]
    n = float(np.linalg.norm(dp))
    if max_m > 0 and n > max_m:
        dp = dp * (max_m / n)
    dr = matrix_to_axis_angle(T_next[:3, :3] @ T_prev[:3, :3].T)
    th = float(np.linalg.norm(dr))
    if max_rad > 0 and th > max_rad:
        dr = dr * (max_rad / th)
    T = np.eye(4)
    T[:3, :3] = axis_angle_to_matrix(dr) @ T_prev[:3, :3]
    T[:3, 3] = T_prev[:3, 3] + dp
    return T
