#!/usr/bin/env python3
"""验证 RGB-D → 基座系：用已知板位置检查两台深度反投影。

前提：板还在桌上（位置 = wrist_handeye.json 的 T_base_board），外参是今天标的。
每台各取一帧彩色+深度，在彩色图上检出板，取板区域内的深度点反投影到基座系，
看中位高度、平面法向、水平偏差。

顶部 335：OBSENSOR（深度已对齐彩色），用 top_extrinsic.json。
腕部 305：V4L2 彩色 + V4L2 Z16 深度（未对齐），用出厂 depth_to_color + wrist_handeye；
         关节角只读连臂（power_on=False），不上电、不动。

用法（在 算法/DexGraspVLA-/SO101 下，lerobot 环境）：
    python calib_poses_manual/verify_rgbd_base.py
    python calib_poses_manual/verify_rgbd_base.py --top-only
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

SO101 = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
ROOT = next(p for p in SO101.parents if (p / "算法").is_dir() and (p / "数据_硬件").is_dir())  # 工作区根
sys.path.insert(0, str(SO101))
sys.path.insert(0, str(ROOT / "算法/Jev/RobotDecision"))

from atom_robot.ar5.config import ArmConfig  # noqa: E402
from atom_robot.ar5.kinematics import AR5Kinematics  # noqa: E402

TOP_SERIAL = "CP0BB53000FC"
TOP_COLOR = "/dev/v4l/by-path/pci-0000:0d:00.0-usb-0:5:1.4-video-index0"
TOP_DEPTH = "/dev/v4l/by-path/pci-0000:0d:00.0-usb-0:5:1.0-video-index0"
WRIST_COLOR = "/dev/v4l/by-path/pci-0000:0d:00.0-usb-0:6:1.4-video-index0"
WRIST_DEPTH = "/dev/v4l/by-path/pci-0000:0d:00.0-usb-0:6:1.0-video-index0"
MIN_CORNERS = 20
SQ = 0.025
PARAMS335 = ROOT / "算法/Jev/RobotJev/robojev/calibration/orbbec335_params.json"
PARAMS305 = ROOT / "算法/Jev/RobotJev/robojev/calibration/orbbec_params.json"


def load_json(p):
    return json.load(open(p))


def charuco_mask(img_bgr, board, det):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    ch, cid, _, _ = det.detectBoard(gray)
    if cid is None or len(cid) < MIN_CORNERS:
        return None, None, None
    ch = np.asarray(ch, np.float64).reshape(-1, 2)
    cid = np.asarray(cid).ravel()
    hull = cv2.convexHull(ch.astype(np.float32))
    mask = np.zeros(gray.shape, np.uint8)
    cv2.fillConvexPoly(mask, hull.astype(np.int32), 255)
    # 缩一点，避开板边
    erode = cv2.erode(mask, np.ones((7, 7), np.uint8))
    return erode, ch, cid


def plane_fit(P):
    c = P.mean(0)
    _, _, vh = np.linalg.svd(P - c, full_matrices=False)
    n = vh[-1]
    if n[2] < 0:
        n = -n
    return c, n


def summarize(name, P, Tbb, save_prefix=None, color=None, mask=None):
    """P: Nx3 基座系点（米）。"""
    if len(P) < 50:
        print(f"[{name}] 有效深度点太少：{len(P)}")
        return None
    c, n = plane_fit(P)
    zb = float(Tbb[2, 3])
    board_xy = Tbb[:2, 3]
    # 板系 z 朝下（标定里板法向约 [0,0,-1]），板面高度 ≈ Tbb z
    dz = (P[:, 2] - zb) * 1000
    dxy = np.linalg.norm(P[:, :2] - board_xy, axis=1) * 1000
    tilt = float(np.degrees(np.arccos(np.clip(abs(n[2]), 0, 1))))
    print(f"[{name}] 点数 {len(P)}  中位深度点(基座) "
          f"{np.round(np.median(P, 0) * 1000, 1).tolist()} mm")
    print(f"  相对已知板面：Δz 中位 {np.median(dz):+.1f} mm，"
          f"IQR [{np.percentile(dz, 25):+.1f}, {np.percentile(dz, 75):+.1f}]，"
          f"水平到板中心中位 {np.median(dxy):.1f} mm")
    print(f"  拟合平面法向 {np.round(n, 3).tolist()}  与竖直夹角 {tilt:.1f}°"
          f"（板标定时法向 {[round(x,3) for x in Tbb[:3,2]]}）")
    print(f"  已知板中心 {np.round(Tbb[:3,3]*1000,1).tolist()} mm，z={zb*1000:.1f} mm")
    if save_prefix is not None and color is not None and mask is not None:
        vis = color.copy()
        vis[mask > 0] = (0.55 * vis[mask > 0] + np.array([0, 180, 0]) * 0.45).astype(np.uint8)
        cv2.imwrite(str(save_prefix) + "_color_mask.jpg", vis)
        # 俯视散点（x-y）和侧视（x-z）简单存成 npz，方便回头看
        np.savez(str(save_prefix) + "_points.npz", P=P, center=c, normal=n)
    return {"n": len(P), "dz_med_mm": float(np.median(dz)), "dxy_med_mm": float(np.median(dxy)),
            "tilt_deg": tilt, "pos_med_mm": (np.median(P, 0) * 1000).tolist()}


def capture_v4l2_rgbd(color_node, depth_node, mm_per_unit, label):
    """V4L2 彩色 + V4L2 Z16 深度（未对齐彩色）。"""
    import depth_v4l2 as D
    print(f"打开 {label}：V4L2 彩色 + V4L2 Z16 深度（未对齐）…")
    depth_s = D.DepthStream(depth_node, mm_per_unit, fps=30).start()
    cap = cv2.VideoCapture(color_node, cv2.CAP_V4L2)
    if not cap.isOpened():
        depth_s.stop()
        raise RuntimeError(f"打不开彩色 {color_node}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    bgr = mm = None
    t0 = time.time()
    while time.time() - t0 < 8:
        ok, frame = cap.read()
        d = depth_s.get_mm(max_age=0.5)
        if ok and frame is not None and d is not None:
            bgr, mm = frame, d
            for _ in range(15):
                ok, frame = cap.read()
                d = depth_s.get_mm(max_age=0.5)
                if ok and frame is not None and d is not None:
                    bgr, mm = frame, d
            break
        time.sleep(0.02)
    cap.release()
    depth_s.stop()
    if bgr is None:
        raise RuntimeError(f"{label} 8 秒内没拿到彩色+深度")
    v = mm[mm > 0]
    print(f"  {label} 帧 {bgr.shape}，深度有效 {v.size / mm.size:.0%}，中位 {np.median(v):.0f} mm"
          f"（×{mm_per_unit:g}）")
    return bgr, mm


def capture_top():
    return capture_v4l2_rgbd(TOP_COLOR, TOP_DEPTH, 1.0, "顶部335")


def capture_wrist():
    return capture_v4l2_rgbd(WRIST_COLOR, WRIST_DEPTH, 0.1, "腕部305")


def read_joints():
    from atom_robot.ar5.drivers.rokae_arm import RokaeAR5
    print("只读连臂（不上电）读关节角…")
    arm = RokaeAR5(ArmConfig())
    arm.connect(power_on=False)
    q = np.asarray(arm.get_joint_positions(), float)[:7]
    arm.disconnect()
    print(f"  q = {np.round(np.degrees(q), 1).tolist()} deg")
    return q


def backproject_depth_via_d2c(mm, Kd, Rd2c, td2c_m, Kc, dist_c, mask_color,
                              T_base_cam, z_min_mm=50, z_max_mm=2000, stride=2):
    """V4L2 深度未对齐：深度像素→深度系→彩色系，投影落在彩色板 mask 内的点再进基座。"""
    ys, xs = np.where((mm > z_min_mm) & (mm < z_max_mm))
    if stride > 1:
        ys, xs = ys[::stride], xs[::stride]
    if len(xs) == 0:
        return np.zeros((0, 3))
    uv_d = np.stack([xs, ys], axis=1).astype(np.float64)
    Z = mm[ys, xs] / 1000.0
    und = cv2.undistortPoints(uv_d.reshape(-1, 1, 2), Kd, None).reshape(-1, 2)
    Pd = np.c_[und * Z[:, None], Z]
    Pc = (Rd2c @ Pd.T).T + td2c_m
    # 投到彩色像素，筛板区域
    pr, _ = cv2.projectPoints(Pc, np.zeros(3), np.zeros(3), Kc, dist_c)
    uv_c = pr.reshape(-1, 2)
    u = np.rint(uv_c[:, 0]).astype(int)
    v = np.rint(uv_c[:, 1]).astype(int)
    H, W = mask_color.shape
    keep = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    keep[keep] = mask_color[v[keep], u[keep]] > 0
    Pc = Pc[keep]
    if len(Pc) == 0:
        return np.zeros((0, 3))
    R = T_base_cam[:3, :3]
    t = T_base_cam[:3, 3]
    return (R @ Pc.T).T + t


def load_d2c(path):
    d = load_json(path)
    di = d["depth_intrinsic"]
    Kd = np.array([[di["fx"], 0, di["cx"]], [0, di["fy"], di["cy"]], [0, 0, 1.]])
    Rd2c = np.array(d["depth_to_color"]["R"], float)
    td2c = np.array(d["depth_to_color"]["t_mm"], float) / 1000.0
    return Kd, Rd2c, td2c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-only", action="store_true")
    ap.add_argument("--wrist-only", action="store_true")
    ap.add_argument("--out", type=Path, default=HERE / "verify_rgbd_out")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    wrist = load_json(HERE / "wrist_handeye.json")
    top = load_json(HERE / "top_extrinsic.json")
    Tbb = np.array(wrist["T_base_board"])
    A = cv2.aruco
    board = A.CharucoBoard((7, 9), SQ, 0.018, A.getPredefinedDictionary(A.DICT_5X5_50))
    board.setLegacyPattern(False)
    det = A.CharucoDetector(board)

    results = {}

    do_top = not args.wrist_only
    do_wrist = not args.top_only

    if do_top:
        bgr, mm = capture_top()
        cv2.imwrite(str(args.out / "top_color.jpg"), bgr)
        mask, ch, cid = charuco_mask(bgr, board, det)
        if mask is None:
            print("[top] 彩色图里没检出板（角点不足），跳过。板还在桌上吗？")
        else:
            print(f"[top] 检出板 {len(cid)} 个角点")
            Kc = np.array(top["K"], float)
            dist = np.array(top["dist_k1_k2_p1_p2_k3"], float)
            T_base_cam = np.array(top["T_base_cam"], float)
            Kd, Rd2c, td2c = load_d2c(PARAMS335)
            print(f"  depth→color t={np.round(td2c*1000,2).tolist()} mm（V4L2 未对齐，走 d2c）")
            P = backproject_depth_via_d2c(mm, Kd, Rd2c, td2c, Kc, dist, mask, T_base_cam)
            results["top"] = summarize("top", P, Tbb, args.out / "top", bgr, mask)

    if do_wrist:
        try:
            q = read_joints()
        except Exception as e:
            print(f"[wrist] 读关节失败：{e} —— 跳过腕部（需要控制器在线）")
            q = None
        if q is not None:
            bgr, mm = capture_wrist()
            cv2.imwrite(str(args.out / "wrist_color.jpg"), bgr)
            mask, ch, cid = charuco_mask(bgr, board, det)
            if mask is None:
                print("[wrist] 彩色图里没检出板。当前姿态腕部可能看不到板。")
            else:
                print(f"[wrist] 检出板 {len(cid)} 个角点")
                raw = load_json(SO101 / "atom_robot/ar5/ar5_dh.json")
                flat = raw["dh"] if isinstance(raw, dict) else raw
                cfg = ArmConfig()
                kin = AR5Kinematics(np.array(flat, float).reshape(-1, 4).tolist(),
                                    num_joints=cfg.num_joints,
                                    joint_min=list(cfg.joint_min),
                                    joint_max=list(cfg.joint_max))
                T_bf = kin.fk(q)
                T_fc = np.array(wrist["T_flange_camera"], float)
                T_base_cam = T_bf @ T_fc
                Kc = np.array(wrist["K"], float)
                dist_c = np.array(wrist["dist_k1_k2_p1_p2_k3"], float)
                Kd, Rd2c, td2c = load_d2c(PARAMS305)
                print(f"  depth→color t={np.round(td2c*1000,2).tolist()} mm")
                P = backproject_depth_via_d2c(mm, Kd, Rd2c, td2c, Kc, dist_c, mask, T_base_cam)
                results["wrist"] = summarize("wrist", P, Tbb, args.out / "wrist", bgr, mask)
                (args.out / "wrist_T_base_cam.json").write_text(json.dumps({
                    "q_rad": q.tolist(),
                    "T_base_flange": T_bf.tolist(),
                    "T_base_cam": T_base_cam.tolist(),
                    "cam_pos_base_mm": (T_base_cam[:3, 3] * 1000).tolist(),
                }, indent=2))

    (args.out / "summary.json").write_text(
        json.dumps({"T_base_board_z_mm": Tbb[2, 3] * 1000, "results": results},
                   indent=2, ensure_ascii=False))
    print("\n结果已写", args.out)


if __name__ == "__main__":
    main()
