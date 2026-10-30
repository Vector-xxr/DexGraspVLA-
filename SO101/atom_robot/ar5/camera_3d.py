"""RGB-D 相机里的物体 -> 基座系 3D 位置（顶部 335 + 腕部 305）。

    from atom_robot.ar5.camera_3d import load_rig
    rig = load_rig()
    # 顶部：深度已对齐彩色，直接给
    r = rig.top.mask_to_base(mask, depth_mm)                      # mask 是彩色图上的 bool (480, 640)
    # 腕部：深度没对齐、还要当前关节角（相机跟着臂走）
    r = rig.wrist.mask_to_base(mask, depth_mm, q=q7)              # q7 是 7 个关节角，rad
    r["xyz"]   # 基座系，米；None 表示这一帧不可用（有效深度点太少）

数学（和 calib_* 脚本同一套约定）：
    顶部    p_base = Rᵀ (p_cam − t)                     R、t 来自 top_extrinsic.json（p_cam = R p_base + t）
    腕部    p_base = T_base_flange(q) · T_flange_cam · p_cam
            p_cam  = R_d2c · p_depth + t_d2c            深度系 -> 彩色系（出厂值，R≈I、t=0）
    p_cam  = z · K⁻¹ [u, v, 1]ᵀ（先去畸变）

标定文件都在 `算法/DexGraspVLA-/SO101/calib_poses_manual/`：top_extrinsic.json、wrist_handeye.json、
depth_params.json。出厂深度内参是 10-29 从 RobotJev 复制的一份。

**精度（10-29 实测，见 记忆 wrist-cam-handeye-calib）**：两台相机对同一个套筒，高度差 0.5 mm、
顶面中心水平差 12 mm；绝对 z 各有几毫米偏差（顶部台面读低 5.8、腕部读高 3.1，套筒读 46~47 mm
实际 45）。这是深度图自己的尺度误差，**这里没有修**。物体相对台面的高度可信，绝对位置按
±5~10 mm 看。

深度单位：调用方先换成**毫米**（335 原始 uint16 就是 mm；305 的 V4L2 原始值 ×0.1，
`DepthStream.get_mm()` 已经换好）。0 = 无效。

    python -m atom_robot.ar5.camera_3d                       # 不碰硬件的自检（往返一致）
    python -m atom_robot.ar5.camera_3d --capture cap.npz     # 用 rgbd 采集文件（见下）演示
"""

import json
import sys
import warnings
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

CALIB_DIR = Path(__file__).resolve().parents[2] / "calib_poses_manual"
DH_JSON = Path(__file__).resolve().parent / "ar5_dh.json"

MIN_POINTS = 30          # 掩码里有效深度点少于这个数就不给结果
MIN_VALID_RATIO = 0.15   # 掩码里有深度的像素占比低于这个数也不给（黑物体常这样）


def _T(R, t) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = np.asarray(R, dtype=np.float64)
    T[:3, 3] = np.asarray(t, dtype=np.float64).ravel()
    return T


def load_kinematics():
    """离线正运动学（ar5_dh.json），不连臂。"""
    from .config import ArmConfig
    from .kinematics import AR5Kinematics

    raw = json.load(open(DH_JSON))
    flat = raw["dh"] if isinstance(raw, dict) else raw
    cfg = ArmConfig()
    return AR5Kinematics(np.array(flat, dtype=np.float64).reshape(-1, 4).tolist(),
                         num_joints=cfg.num_joints, joint_min=list(cfg.joint_min),
                         joint_max=list(cfg.joint_max))


class Cam3D:
    """一台相机的 像素+深度 -> 基座系。顶部和腕部共用，差别只在 `T_base_cam` 怎么来、深度要不要先对齐。"""

    def __init__(self, name, K, dist, mm_per_unit=1.0, T_base_cam=None, T_flange_cam=None,
                 depth_intrinsic=None, depth_to_color=None, kin=None):
        self.name = name
        self.K = np.asarray(K, dtype=np.float64)
        self.dist = np.asarray(dist, dtype=np.float64).ravel()
        self.mm_per_unit = float(mm_per_unit)
        self._T_base_cam = None if T_base_cam is None else np.asarray(T_base_cam, dtype=np.float64)
        self._T_flange_cam = None if T_flange_cam is None else np.asarray(T_flange_cam, dtype=np.float64)
        self.depth_intrinsic = depth_intrinsic          # 只有深度没对齐的相机才有
        self.depth_to_color = depth_to_color            # (R 3x3, t 3)
        self._kin = kin

    # ------------------------------------------------------------------ 位姿

    @property
    def mounted_on_arm(self) -> bool:
        return self._T_flange_cam is not None

    def T_base_cam(self, q=None, T_base_flange=None) -> np.ndarray:
        """彩色相机在基座系的位姿。顶部是常量；腕部要关节角 q（rad，7 个）或直接给法兰位姿。"""
        if not self.mounted_on_arm:
            return self._T_base_cam
        if T_base_flange is None:
            if q is None:
                raise ValueError(f"{self.name} 装在臂上，要给 q（7 个关节角，rad）或 T_base_flange")
            if self._kin is None:
                self._kin = load_kinematics()
            T_base_flange = self._kin.fk(np.asarray(q, dtype=np.float64)[:7])
        return np.asarray(T_base_flange) @ self._T_flange_cam

    # ------------------------------------------------------------------ 深度对齐

    def align_depth(self, depth_mm: np.ndarray) -> np.ndarray:
        """把深度图变成「和彩色像素一一对应」的深度（mm，0=无效）。顶部本来就对齐，原样返回。

        腕部：每个深度像素先搬到彩色相机系，再投到彩色像素，同一像素取最近的那个（z-buffer）。
        两个视角几乎重合（R≈I、t=0），投不到的空洞用 3x3 邻域中位数补。
        """
        depth_mm = np.asarray(depth_mm, dtype=np.float32)
        if self.depth_intrinsic is None:
            return depth_mm
        h, w = depth_mm.shape
        di = self.depth_intrinsic
        v, u = np.nonzero(depth_mm > 0)
        z = depth_mm[v, u].astype(np.float64) / 1000.0
        pd = np.c_[(u - di["cx"]) / di["fx"] * z, (v - di["cy"]) / di["fy"] * z, z]
        Rd, td = self.depth_to_color
        pc = pd @ np.asarray(Rd).T + np.asarray(td)
        pr = cv2.projectPoints(pc, np.zeros(3), np.zeros(3), self.K, self.dist)[0].reshape(-1, 2)
        pi = np.round(pr).astype(int)
        ok = (pi[:, 0] >= 0) & (pi[:, 0] < w) & (pi[:, 1] >= 0) & (pi[:, 1] < h) & (pc[:, 2] > 0)
        flat = pi[ok, 1] * w + pi[ok, 0]
        buf = np.full(h * w, np.inf)
        np.minimum.at(buf, flat, pc[ok, 2] * 1000.0)
        out = buf.reshape(h, w)
        out[~np.isfinite(out)] = np.nan
        # 补空洞：只补周围至少 5 个邻居有值的像素
        pad = np.pad(out, 1, constant_values=np.nan)
        stack = np.stack([pad[dy:dy + h, dx:dx + w] for dy in range(3) for dx in range(3)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)      # 整块全 NaN 的邻域，下面按计数排除
            med = np.nanmedian(stack, axis=0)
        cnt = np.sum(~np.isnan(stack), axis=0)
        fill = np.isnan(out) & (cnt >= 5)
        out[fill] = med[fill]
        return np.nan_to_num(out, nan=0.0).astype(np.float32)

    # ------------------------------------------------------------------ 像素 <-> 基座

    def pixels_to_cam(self, uv, z_mm) -> np.ndarray:
        """彩色像素 + 深度(mm) -> 彩色相机系 3D 点（米）。"""
        uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
        z = np.asarray(z_mm, dtype=np.float64).reshape(-1) / 1000.0
        und = cv2.undistortPoints(uv.reshape(-1, 1, 2), self.K, self.dist).reshape(-1, 2)
        return np.c_[und * z[:, None], z]

    def pixels_to_base(self, uv, z_mm, q=None, T_base_flange=None) -> np.ndarray:
        """彩色像素 + **已对齐**的深度(mm) -> 基座系 3D 点（米）。"""
        T = self.T_base_cam(q, T_base_flange)
        pc = self.pixels_to_cam(uv, z_mm)
        return pc @ T[:3, :3].T + T[:3, 3]

    def base_to_pixels(self, p_base, q=None, T_base_flange=None):
        """基座系点 -> (彩色像素, 相机系深度 mm)。画叠加图、核对用。"""
        T = self.T_base_cam(q, T_base_flange)
        p = np.asarray(p_base, dtype=np.float64).reshape(-1, 3)
        pc = (p - T[:3, 3]) @ T[:3, :3]
        uv = cv2.projectPoints(pc, np.zeros(3), np.zeros(3), self.K, self.dist)[0].reshape(-1, 2)
        return uv, pc[:, 2] * 1000.0

    def mask_to_base(self, mask, depth_mm, q=None, T_base_flange=None, aligned=False,
                     mode="median", return_points=False) -> dict:
        """掩码（彩色图上的 bool）+ 深度 -> 物体的基座系位置。

        mode="median"  掩码内所有有效点的逐轴中位数（默认，稳）。
        mode="top"     最高那一层（最高 4 mm 内）的平均 —— 想要物体顶面时用。
                       **只适合干净的掩码**：混进一个离群点（夹爪、线缆）它就跟着跑，
                       中位数不会。10-29 演示里用粗糙的深色阈值当掩码，腕部的 top 就跑到了夹爪上。
        深度没对齐（腕部）就在这里先对齐；已经对齐过的传 aligned=True。

        返回 dict：xyz（米，或 None）、n_points、valid_ratio、reason（不可用时的原因）。
        """
        mask = np.asarray(mask, dtype=bool)
        d = np.asarray(depth_mm, dtype=np.float32)
        if not aligned:
            d = self.align_depth(d)
        if d.shape != mask.shape:
            raise ValueError(f"掩码 {mask.shape} 和深度 {d.shape} 对不上")
        total = int(mask.sum())
        v, u = np.nonzero(mask & (d > 0))
        res = {"xyz": None, "n_points": int(len(u)), "valid_ratio": (len(u) / total) if total else 0.0,
               "reason": None}
        if total == 0:
            res["reason"] = "掩码是空的"
            return res
        if len(u) < MIN_POINTS or res["valid_ratio"] < MIN_VALID_RATIO:
            res["reason"] = f"有效深度点太少（{len(u)} 个，占掩码 {res['valid_ratio'] * 100:.0f}%）"
            return res
        P = self.pixels_to_base(np.c_[u, v], d[v, u], q, T_base_flange)
        if mode == "top":
            sel = P[:, 2] > np.percentile(P[:, 2], 98) - 0.004
            res["xyz"] = P[sel].mean(0)
        elif mode == "median":
            res["xyz"] = np.median(P, axis=0)
        else:
            raise ValueError(f"mode 只能是 median / top，给的是 {mode!r}")
        if return_points:
            res["points"] = P
        return res


class Rig:
    """两台相机。`top` / `wrist` 各是一个 Cam3D。"""

    def __init__(self, top: Cam3D, wrist: Cam3D):
        self.top, self.wrist = top, wrist


def load_rig(calib_dir: Optional[Path] = None) -> Rig:
    d = Path(calib_dir) if calib_dir else CALIB_DIR
    top = json.load(open(d / "top_extrinsic.json"))
    wr = json.load(open(d / "wrist_handeye.json"))
    dp = json.load(open(d / "depth_params.json"))
    kin = load_kinematics()
    top_cam = Cam3D("top", top["K"], top["dist_k1_k2_p1_p2_k3"], dp["top"]["mm_per_unit"],
                    T_base_cam=top["T_base_cam"])
    wr_cam = Cam3D("wrist", wr["K"], wr["dist_k1_k2_p1_p2_k3"], dp["wrist"]["mm_per_unit"],
                   T_flange_cam=wr["T_flange_camera"],
                   depth_intrinsic=dp["wrist"]["depth_intrinsic"],
                   depth_to_color=(dp["wrist"]["depth_to_color"]["R"], dp["wrist"]["depth_to_color"]["t_m"]),
                   kin=kin)
    return Rig(top_cam, wr_cam)


# ---------------------------------------------------------------------- 自检 / 演示

def _selftest() -> int:
    rig = load_rig()
    rng = np.random.default_rng(0)
    q = np.radians([-30.4, 8.2, 0.8, 113.6, 7.8, 36.7, 47.1])
    worst = 0.0
    for cam, kw in ((rig.top, {}), (rig.wrist, {"q": q})):
        # 基座系里取一批点 -> 投到像素和深度 -> 再反投影，应当回到原处
        c = cam.T_base_cam(**kw)
        P = c[:3, 3] + (c[:3, :3] @ np.c_[rng.uniform(-.2, .2, 50), rng.uniform(-.15, .15, 50),
                                         rng.uniform(.3, .8, 50)].T).T
        uv, z = cam.base_to_pixels(P, **kw)
        back = cam.pixels_to_base(uv, z, **kw)
        err = np.abs(back - P).max()
        worst = max(worst, err)
        print(f"  {cam.name}：往返最大误差 {err * 1e6:.3f} µm")
    # 掩码接口：整个画面都是同一块水平板（距相机 0.5 m、正对），mask_to_base 应给出中位数
    cam = rig.top
    v, u = np.mgrid[0:480, 0:640]
    T = cam.T_base_cam()
    z0 = 600.0
    d = np.full((480, 640), z0, np.float32)
    m = np.zeros((480, 640), bool); m[200:260, 300:360] = True
    r = cam.mask_to_base(m, d, aligned=True)
    exp = np.median(cam.pixels_to_base(np.c_[u[m], v[m]], d[m]), axis=0)
    print(f"  mask_to_base：{np.round(r['xyz'], 4).tolist()}  期望 {np.round(exp, 4).tolist()}  点数 {r['n_points']}")
    assert np.abs(r["xyz"] - exp).max() < 1e-9
    assert cam.mask_to_base(np.zeros_like(m), d, aligned=True)["xyz"] is None
    assert cam.mask_to_base(m, np.zeros_like(d), aligned=True)["xyz"] is None
    # 腕部深度对齐：深度全 330 mm 的平面 -> 对齐后仍是 ~330 mm、孔洞很少
    dw = np.full((480, 640), 330.0, np.float32)
    al = rig.wrist.align_depth(dw)
    print(f"  腕部深度对齐：有效 {np.mean(al > 0) * 100:.1f}%，中位 {np.median(al[al > 0]):.1f} mm（深度系 fx 375 -> 彩色系 fx 371）")
    assert worst < 1e-6
    print("  自检通过")
    return 0


def _demo(path: str) -> int:
    """用采集文件里的两台相机各取一个套筒掩码（深色像素 + 种子点半径），打印基座系位置。"""
    cap = np.load(path)
    rig = load_rig()

    def med(s):
        s = np.asarray(s, np.float32).copy(); s[s <= 0] = np.nan
        with np.errstate(all="ignore"):
            return np.nan_to_num(np.nanmedian(s, axis=0), nan=0.0)

    for cam, rgb, depth, seed, rad, kw in (
            (rig.top, cap["top_rgb"][-1], med(cap["top_depth"]), (328, 215), 38, {}),
            (rig.wrist, cap["wrist_rgb"][-1], med(cap["wrist_depth_mm"]), (478, 190), 60,
             {"q": cap["q"]})):
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        yy, xx = np.mgrid[0:480, 0:640]
        m = (np.hypot(xx - seed[0], yy - seed[1]) < rad) & (gray < 70)
        for mode in ("median", "top"):
            r = cam.mask_to_base(m, depth, mode=mode, **kw)
            xyz = None if r["xyz"] is None else np.round(r["xyz"] * 1000, 1).tolist()
            print(f"  {cam.name:5s} {mode:6s}：基座系 {xyz} mm   点数 {r['n_points']}  有效占比 {r['valid_ratio'] * 100:.0f}%"
                  f"{'' if r['reason'] is None else '  ' + r['reason']}")
    return 0


if __name__ == "__main__":
    if "--capture" in sys.argv:
        sys.exit(_demo(sys.argv[sys.argv.index("--capture") + 1]))
    sys.exit(_selftest())
