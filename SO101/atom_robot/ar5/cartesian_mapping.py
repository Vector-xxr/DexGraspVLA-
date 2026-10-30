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

        self._leader = So101FK(urdf or default_urdf())
        self._kin = default_kinematics(arm_cfg)
        self._ws_lo = np.asarray(workspace_min, dtype=np.float64)
        self._ws_hi = np.asarray(workspace_max, dtype=np.float64)

        # 主臂基座系 -> AR5 基座系的轴映射，读 leader_calibration.json
        # （calibrate_mapping 写的）。镜像时转动向量要多一个负号。
        axis_cfg = axis_cfg or MappingConfig.load()
        self._M = mapping_matrix(axis_cfg.axis_permutation, axis_cfg.axis_signs)
        self._parity = float(np.sign(np.linalg.det(self._M)) or 1.0)

        self._engaged = False
        self._lead0: Optional[np.ndarray] = None       # 主臂锚点 4x4
        self._arm0: Optional[np.ndarray] = None        # 从臂锚点 4x4
        self._last_q: Optional[np.ndarray] = None
        self._last_ik_reason = ""
        self.stats = {"steps": 0, "clamped": 0, "missing": 0, "ik_fail": 0,
                      "clutch_cycles": 0, "ik_ms_max": 0.0}

    # ── 和 JointMapper 同名的接口 ────────────────────────────────────
    def configure(self, leader_half_travel, joint_min, joint_max, margin: float = 0.0) -> None:
        self._joint.configure(leader_half_travel, joint_min, joint_max, margin=margin)
        lo = np.asarray(joint_min, dtype=np.float64) + margin
        hi = np.asarray(joint_max, dtype=np.float64) - margin
        # 逆解在 margin 以内找解，和驱动 / step_joints 用同一个边界
        self._kin.joint_min, self._kin.joint_max = lo, hi
        self._kin.joint_mid = (lo + hi) / 2.0
        self._kin.joint_range = np.maximum(hi - lo, 1e-9)

    def gripper_command(self, joints_deg: Dict[str, float]) -> float:
        return self._joint.gripper_command(joints_deg)

    @property
    def engaged(self) -> bool:
        return self._engaged

    def engage(self, q_now: Sequence[float]) -> None:
        """接合 / 重锚：从臂锚点取当前设定点的正运动学，主臂锚点下一帧再取。"""
        q_now = np.asarray(q_now, dtype=np.float64)
        self._arm0 = self._kin.fk(q_now)
        self._lead0 = None
        self._last_q = q_now.copy()
        if not self._engaged:
            self.stats["clutch_cycles"] += 1
        self._engaged = True

    def disengage(self) -> None:
        self._engaged = False
        self._lead0 = self._arm0 = None

    def compute(self, joints_deg: Dict[str, float], q_now: Sequence[float]) -> np.ndarray:
        q_now = np.asarray(q_now, dtype=np.float64)
        self.stats["steps"] += 1
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

        seed = self._last_q if self._last_q is not None else q_now
        t0 = time.perf_counter()
        q, info = self._kin.ik(target, seed)
        self.stats["ik_ms_max"] = max(self.stats["ik_ms_max"], (time.perf_counter() - t0) * 1000)
        if q is None:
            # 够不到：停在上一个够得到的地方，别瞎动。TrackingWatch 会报出来。
            self.stats["ik_fail"] += 1
            self._last_ik_reason = info.get("reason", "")
            return (self._last_q if self._last_q is not None else q_now).copy()
        self._last_q = q
        return q

    def report(self) -> str:
        c = self.config
        return "\n".join([
            "  映射：**笛卡尔**（主臂末端位姿 -> 逆解 -> 7 轴全动）",
            f"    位置缩放 x{c.position_scale}   转动缩放 x{c.rotation_scale}   转动轴 {c.rotation_axes}",
            f"    主臂->AR5 轴映射 M =\n" + "\n".join("      " + str(r) for r in self._M.astype(int).tolist()),
            f"    工作空间盒 {self._ws_lo.tolist()} ~ {self._ws_hi.tolist()} m",
            "    受控点在法兰中心（tcp 归 0），转腕时夹爪指尖会甩",
        ])
