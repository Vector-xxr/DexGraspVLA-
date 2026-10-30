"""顶部相机（cam_high，Gemini 335）外参：借已标好的板位置求相机在基座系的位姿。

前提：腕部手眼标定已做完（wrist_handeye.json 里有板在基座系的位姿），采集时板没动，
poses.npz 里存了每组位姿同时拍的顶部原图 images_high。板位置已知 → 顶部相机看板就是
一个 PnP，32 张的全部角点一起拟合，比单张稳。

内参不在这里估：板在顶部图里只占画面中间一小块（约 140x100 px），估不出内参。
用 RobotJev 从设备读出的出厂值（robojev/calibration/orbbec335_params.json 的 color_intrinsic）。

用法（在 算法/DexGraspVLA-/SO101 下，OpenCV 4.x + scipy，用 lerobot 环境）：
    python calib_poses_manual/calib_top_extrinsic.py

约定：p_cam = R · p_base + t（和 RobotJev 的 top335_extrinsic.json 同一写法）。
注意：RobotJev 那份是夹球标的、朝向固定下才成立，**这里是纯相机位姿，不能和它混用**。
"""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

SO101 = Path(__file__).resolve().parents[1]
ROOT = next(p for p in SO101.parents if (p / "算法").is_dir() and (p / "数据_硬件").is_dir())  # 工作区根
HERE = Path(__file__).resolve().parent
PARAMS = ROOT / "算法/Jev/RobotJev/robojev/calibration/orbbec335_params.json"
OUT = HERE / "top_extrinsic.json"
MIN_CORNERS = 20
BOARD_POS_SIGMA_M = 0.001      # 腕部标定给的板位置的不确定度，偏保守
BOARD_ROT_SIGMA_DEG = 0.3

wrist = json.load(open(HERE / "wrist_handeye.json"))
Tbb = np.array(wrist["T_base_board"])
SQ = wrist["square_len_m"]
par = json.load(open(PARAMS))
ci = par["color_intrinsic"]
K = np.array([[ci["fx"], 0, ci["cx"]], [0, ci["fy"], ci["cy"]], [0, 0, 1.]])
dist = np.array(par["color_distortion"]["opencv"][:5])

A = cv2.aruco
board = A.CharucoBoard((7, 9), SQ, 0.018, A.getPredefinedDictionary(A.DICT_5X5_50))
board.setLegacyPattern(False)
det = A.CharucoDetector(board)
P3 = board.getChessboardCorners().astype(np.float64)
Pb = (Tbb[:3, :3] @ P3.T).T + Tbb[:3, 3]                      # 板角点，基座系

imgs = np.load(HERE / "poses.npz")["images_high"]
obs = []
for im in imgs:
    ch, cid, _, _ = det.detectBoard(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
    if cid is not None and len(cid) >= MIN_CORNERS:
        obs.append((np.asarray(ch, np.float64).reshape(-1, 2), np.asarray(cid).ravel()))
print(f"顶部图 {len(imgs)} 张，角点>={MIN_CORNERS} 的 {len(obs)} 张，共 {sum(len(c) for _, c in obs)} 个角点")
print(f"内参来自 {PARAMS.relative_to(ROOT)}：fx={ci['fx']:.1f} fy={ci['fy']:.1f} cx={ci['cx']:.1f} cy={ci['cy']:.1f}")


def project(p, pts):
    pr, _ = cv2.projectPoints(pts, p[:3], p[3:], K, dist)
    return pr.reshape(-1, 2)


def resid(p, Pw):
    return np.concatenate([(project(p, Pw[c]) - ch).ravel() for ch, c in obs])


# 初值：第一张的 PnP（板系→相机系，再接上板→基座）
ok, r0, t0 = cv2.solvePnP(P3[obs[0][1]].astype(np.float32), obs[0][0].astype(np.float32), K, dist,
                          flags=cv2.SOLVEPNP_IPPE)
Tcb = np.eye(4); Tcb[:3, :3] = cv2.Rodrigues(r0)[0]; Tcb[:3, 3] = t0.ravel()
Tc0 = Tcb @ np.linalg.inv(Tbb)
p0 = np.concatenate([cv2.Rodrigues(Tc0[:3, :3])[0].ravel(), Tc0[:3, 3]])
sol = least_squares(resid, p0, args=(Pb,), loss="soft_l1", f_scale=1.0, x_scale="jac")
p = sol.x
r = resid(p, Pb).reshape(-1, 2)
rms = float(np.sqrt(np.mean(np.sum(r ** 2, axis=1))))
R = cv2.Rodrigues(p[:3])[0]; t = p[3:]
pos = -R.T @ t
dof = r.size - 6
cov = np.linalg.inv(sol.jac.T @ sol.jac) * float(resid(p, Pb) @ resid(p, Pb)) / dof
print(f"\n重投影 RMS {rms:.2f} px（最大 {np.max(np.linalg.norm(r, axis=1)):.2f}）")
print(f"相机在基座系位置 {np.round(pos * 1000, 1).tolist()} mm")
print(f"光轴(基座系) {np.round(R.T @ [0, 0, 1.], 3).tolist()}   画面向右 {np.round(R.T @ [1, 0, 0.], 3).tolist()}"
      f"   画面向下 {np.round(R.T @ [0, 1, 0.], 3).tolist()}")
print(f"光轴与竖直的夹角 {np.degrees(np.arccos(-(R.T @ [0, 0, 1.])[2])):.1f}°")
print(f"统计 1σ（角点噪声）：平移 {np.round(np.sqrt(np.diag(cov))[3:] * 1000, 2).tolist()} mm  旋转 {np.round(np.degrees(np.sqrt(np.diag(cov))[:3]), 3).tolist()}°")

# ── 板位置不确定度传进来的误差：扰动 Tbb 后重拟合，看相机位姿和「像素→桌面点」怎么变 ──
rng = np.random.default_rng(0)
Pt = np.vstack([ch for ch, _ in obs]); Ci = np.concatenate([c for _, c in obs])


def refit(Tb):
    Pw = (Tb[:3, :3] @ P3.T).T + Tb[:3, 3]
    s = least_squares(lambda q: (project(q, Pw[Ci]) - Pt).ravel(), p, x_scale="jac")
    return s.x


def pixel_to_plane(q, uv, z):
    """像素 -> 基座系里高度为 z 的平面上的点。"""
    Rq = cv2.Rodrigues(q[:3])[0]
    und = cv2.undistortPoints(uv.reshape(-1, 1, 2).astype(np.float64), K, dist).reshape(-1, 2)
    rays = (Rq.T @ np.c_[und, np.ones(len(und))].T).T
    c = -Rq.T @ q[3:]
    return c + rays * ((z - c[2]) / rays[:, 2])[:, None]


zb = Tbb[2, 3]
targets = np.array([[0.31, -0.27, zb], [0.45, -0.10, zb], [0.60, 0.05, zb], [0.25, -0.45, zb],
                    [0.31, -0.27, zb + 0.10]])
uv = np.array([project(p, x.reshape(1, 3))[0] for x in targets])
in_img = [(0 <= u[0] < 640 and 0 <= u[1] < 480) for u in uv]
shifts = []
for _ in range(100):
    ax = rng.normal(size=3); ax /= np.linalg.norm(ax)
    dT = np.eye(4)
    dT[:3, :3] = cv2.Rodrigues(ax * np.radians(rng.normal(0, BOARD_ROT_SIGMA_DEG)))[0]
    dT[:3, 3] = rng.normal(0, BOARD_POS_SIGMA_M, 3)
    q = refit(dT @ Tbb)
    shifts.append(np.array([pixel_to_plane(q, uv[k], targets[k][2])[0] - targets[k] for k in range(len(targets))]))
shifts = np.array(shifts)
print(f"\n板位置不确定度（{BOARD_POS_SIGMA_M * 1000:g} mm / {BOARD_ROT_SIGMA_DEG}°）传给「顶部像素→基座系点」的误差（100 次扰动，水平 RMS）：")
for k, x in enumerate(targets):
    h = np.sqrt(np.mean(np.sum(shifts[:, k, :2] ** 2, axis=1))) * 1000
    print(f"  基座系 ({x[0] * 1000:.0f}, {x[1] * 1000:.0f}, {x[2] * 1000:.0f}) mm → 像素 ({uv[k][0]:.0f}, {uv[k][1]:.0f})"
          f"{'' if in_img[k] else ' [在画面外]'}   误差 {h:.1f} mm")

result = {
    "camera": "top (orbbec:CP0BB53000FC, Gemini 335), color 640x480",
    "date": time.strftime("%Y-%m-%d %H:%M"),
    "convention": "p_cam = R @ p_base + t（米）；T_base_cam 是它的逆",
    "R": R.tolist(), "t": t.tolist(),
    "T_base_cam": np.linalg.inv(np.block([[R, t.reshape(3, 1)], [np.zeros((1, 3)), np.ones((1, 1))]])).tolist(),
    "camera_pos_base_mm": (pos * 1000).tolist(),
    "K": K.tolist(), "dist_k1_k2_p1_p2_k3": dist.tolist(),
    "intrinsics_source": str(PARAMS.relative_to(ROOT)) + "（出厂值，未独立标定）",
    "reproj_rms_px": rms, "n_images": len(obs),
    "board_in_base": "wrist_handeye.json 的 T_base_board",
    "note": "纯相机位姿；和 RobotJev 的 top335_extrinsic.json（夹球标、朝向固定）不是一回事，别混用",
}
OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False))
print("\n已写", OUT)
