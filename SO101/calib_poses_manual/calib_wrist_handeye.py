"""腕部相机手眼标定（眼在手上）：重投影联合优化。

未知量：X（相机系→法兰系，6）、板在基座系的位姿（6）、fx fy cx cy k1 k2（6）。
残差：每组位姿里每个 Charuco 角点的像素重投影误差。板是固定在桌上的，所以板位姿只有一份。
用法（在 算法/DexGraspVLA-/SO101 下）：
    python calib_poses_manual/calib_wrist_handeye.py --out calib_poses_manual/wrist_handeye.json \
        calib_poses/poses.npz calib_poses_manual/poses.npz
npz 要有 joints（>=7 列，取前 7）和 images（BGR 原图）。角点 <15 的组自动丢掉。
需要 OpenCV 4.x（calibrateHandEye；5.0 里没有）和 scipy。
"""
import argparse
import sys, json
from pathlib import Path
import numpy as np
import cv2
from scipy.optimize import least_squares

WS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WS))
from atom_robot.ar5.config import ArmConfig
from atom_robot.ar5.kinematics import AR5Kinematics

ap = argparse.ArgumentParser()
ap.add_argument("npz", nargs="+")
ap.add_argument("--out", required=True)
ap.add_argument("--square", type=float, default=0.025, help="方格边长 m（量实物）")
ap.add_argument("--min-corners", type=int, default=15)
args = ap.parse_args()
OUT, SQ = Path(args.out), args.square
parts = [np.load(f) for f in args.npz]
J = np.vstack([d["joints"][:, :7] for d in parts])
IMGS = np.concatenate([d["images"] for d in parts])
H, W = IMGS.shape[1:3]

raw = json.load(open(WS / "atom_robot/ar5/ar5_dh.json"))
flat = raw["dh"] if isinstance(raw, dict) else raw
cfg = ArmConfig()
kin = AR5Kinematics(np.array(flat, float).reshape(-1, 4).tolist(), num_joints=cfg.num_joints,
                    joint_min=list(cfg.joint_min), joint_max=list(cfg.joint_max))
Tbf = np.array([kin.fk(q) for q in J])

A = cv2.aruco
board = A.CharucoBoard((7, 9), SQ, 0.018, A.getPredefinedDictionary(A.DICT_5X5_50))
board.setLegacyPattern(False)
det = A.CharucoDetector(board)
P3 = board.getChessboardCorners().astype(np.float64)
obs = []
for im in IMGS:
    ch, cid, _, _ = det.detectBoard(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
    obs.append((np.asarray(ch, np.float64).reshape(-1, 2), np.asarray(cid).ravel()))
keep = [i for i, (_, c) in enumerate(obs) if len(c) >= args.min_corners]
print(f"读入 {len(obs)} 组，角点 >= {args.min_corners} 的 {len(keep)} 组")
J, IMGS, Tbf = J[keep], IMGS[keep], Tbf[keep]
obs = [obs[i] for i in keep]
N = len(obs)
print(f"{N} 组，方格边长 {SQ * 1000:g} mm，角点总数 {sum(len(c) for _, c in obs)}")


def T_of(R, t):
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = np.ravel(t); return T


def T_from(p):
    return T_of(cv2.Rodrigues(p[:3])[0], p[3:6])


# ── 初值：粗内参 + 闭式手眼(Park) ──
rms0, K0, d0, _, _ = cv2.calibrateCamera([P3[c].astype(np.float32) for _, c in obs],
                                         [p.astype(np.float32) for p, _ in obs], (W, H), None, None,
                                         flags=cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST)
Tcb = []
for p, c in obs:
    ok, r, t = cv2.solvePnP(P3[c], p, K0, d0, flags=cv2.SOLVEPNP_IPPE)
    Tcb.append(T_of(cv2.Rodrigues(r)[0], t))
Rx, tx = cv2.calibrateHandEye([T[:3, :3] for T in Tbf], [T[:3, 3] for T in Tbf],
                              [T[:3, :3] for T in Tcb], [T[:3, 3] for T in Tcb],
                              method=cv2.CALIB_HAND_EYE_PARK)
X0 = T_of(Rx, tx)
Tbb0 = np.mean([Tbf[i] @ X0 @ Tcb[i] for i in range(N)], axis=0)       # 位置取均值
Rm = Tbf[0][:3, :3] @ X0[:3, :3] @ Tcb[0][:3, :3]                      # 姿态取第一组（误差 <1°）
Tbb0[:3, :3] = Rm


def pack(X, Tbb, intr):
    return np.concatenate([cv2.Rodrigues(X[:3, :3])[0].ravel(), X[:3, 3],
                           cv2.Rodrigues(Tbb[:3, :3])[0].ravel(), Tbb[:3, 3], intr])


def residuals(p, idx):
    X, Tbb = T_from(p[0:6]), T_from(p[6:12])
    fx, fy, cx, cy, k1, k2 = p[12:18]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]])
    dist = np.array([k1, k2, 0, 0, 0.])
    res = []
    for i in idx:
        Tc = np.linalg.inv(Tbf[i] @ X) @ Tbb
        pr, _ = cv2.projectPoints(P3[obs[i][1]], cv2.Rodrigues(Tc[:3, :3])[0], Tc[:3, 3], K, dist)
        res.append((pr.reshape(-1, 2) - obs[i][0]).ravel())
    return np.concatenate(res)


intr0 = np.array([K0[0, 0], K0[1, 1], K0[0, 2], K0[1, 2], d0.ravel()[0], d0.ravel()[1]])
p0 = pack(X0, Tbb0, intr0)


def fit(idx, p_init):
    return least_squares(residuals, p_init, args=(idx,), loss="soft_l1", f_scale=1.5, x_scale="jac")


def rms(p, idx):
    r = residuals(p, idx).reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum(r ** 2, axis=1))))


allidx = list(range(N))
print(f"\n闭式手眼(Park)+粗内参 重投影 RMS = {rms(p0, allidx):.2f} px")
sol = fit(allidx, p0)
p = sol.x
print(f"联合优化后            重投影 RMS = {rms(p, allidx):.2f} px")

# ── 不确定度：高斯牛顿协方差 ──
r = residuals(p, allidx)
Jm = sol.jac
dof = len(r) - len(p)
sigma2 = float(r @ r) / dof
cov = np.linalg.inv(Jm.T @ Jm) * sigma2
sd = np.sqrt(np.diag(cov))
X = T_from(p[0:6]); Tbb = T_from(p[6:12])
print("\n── 结果 ──")
print(f"相机在法兰系的位置 t = {np.round(X[:3, 3] * 1000, 1).tolist()} mm   (±{np.round(sd[3:6] * 1000, 2).tolist()} 1σ)")
print(f"相机姿态旋转向量   = {np.round(np.degrees(p[0:3]), 2).tolist()} deg   (±{np.round(np.degrees(sd[0:3]), 2).tolist()} 1σ)")
fx, fy, cx, cy, k1, k2 = p[12:18]
print(f"内参 fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f} k1={k1:.4f} k2={k2:.4f}   (±fx {sd[12]:.1f}, cx {sd[14]:.1f}, cy {sd[15]:.1f})")
print(f"粗内参(只用这批图) fx={K0[0,0]:.1f} fy={K0[1,1]:.1f} cx={K0[0,2]:.1f} cy={K0[1,2]:.1f}")
print(f"板在基座系位置 = {np.round(Tbb[:3, 3] * 1000, 1).tolist()} mm   板法线(基座系) = {np.round(Tbb[:3, 2], 3).tolist()}")
for name, v in (("画面向右", X[:3, 0]), ("画面向下", X[:3, 1]), ("光轴", X[:3, 2])):
    print(f"  {name} 在法兰系 = {np.round(v, 3).tolist()}")

# ── 逐组残差，找坏帧 ──
per = []
for i in allidx:
    rr = residuals(p, [i]).reshape(-1, 2)
    per.append(float(np.sqrt(np.mean(np.sum(rr ** 2, axis=1)))))
per = np.array(per)
bad = np.where(per > 3 * np.median(per))[0]
print(f"\n逐组 RMS：中位 {np.median(per):.2f}  最大 {per.max():.2f} px（第 {int(per.argmax())} 组）；> 3×中位的组：{bad.tolist()}")

# ── 5 折交叉验证：留出的组用训练出的参数去投影 ──
rng = np.random.default_rng(0)
perm = rng.permutation(N)
folds = np.array_split(perm, 5)
ts, rs, held = [], [], []
for f in folds:
    train = [i for i in allidx if i not in set(f.tolist())]
    s = fit(train, p)
    ts.append(T_from(s.x[0:6])[:3, 3] * 1000)
    rs.append(np.degrees(s.x[0:3]))
    # 留出组：板位姿/内参/外参用训练出的，直接投影
    held.append(rms(s.x, list(f)))
ts, rs = np.array(ts), np.array(rs)
print(f"\n5 折交叉验证：留出组重投影 RMS = {np.round(held, 2).tolist()} px")
print(f"  相机平移各折 std = {np.round(ts.std(0), 2).tolist()} mm   极差 = {np.round(np.ptp(ts, 0), 2).tolist()} mm")
print(f"  相机旋转各折 std = {np.round(rs.std(0), 2).tolist()} deg")

# ── 换 SQ 的敏感度：平移按比例变 ──
print(f"\n方格边长每差 1 mm（{100 / (SQ * 1000):.0f}%），相机离板的距离相应差同样比例；"
      f"外参里平移主要由臂的运动定，受影响小。")

result = {
    "camera": "wrist (v4l2:/dev/ar5_cam_wrist), 640x480",
    "convention": "T_flange_camera：把相机系的点变到法兰系；法兰 = AR5 FK 末端（ar5_dh.json）",
    "T_flange_camera": X.tolist(),
    "t_mm": (X[:3, 3] * 1000).tolist(),
    "t_sigma_mm": (sd[3:6] * 1000).tolist(),
    "K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
    "dist_k1_k2_p1_p2_k3": [k1, k2, 0, 0, 0],
    "square_len_m": SQ,
    "n_poses": N,
    "reproj_rms_px": rms(p, allidx),
    "cv_heldout_rms_px": held,
    "T_base_board": Tbb.tolist(),
}
OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False))
print("\n已写", OUT)
