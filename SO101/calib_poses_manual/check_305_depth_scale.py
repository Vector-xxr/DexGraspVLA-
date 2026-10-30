#!/usr/bin/env python3
"""专查腕部 Gemini 305 的深度刻度。

真值：彩色相机看板做 PnP → 板平面在彩色相机系；深度像素经 depth_to_color 投到彩色，
与板平面求交得到期望距离 Z_true。比较 V4L2 原始 Z16 与 Z_true，拟合
    Z_true_mm ≈ a * raw + b
并对照出厂 scale 0.1 mm/单位。

可选：若 pyorbbecsdk 能开 305，再读一帧 SDK 深度作对照（单位由 get_depth_scale 给出）。

用法（lerobot 环境，算法/DexGraspVLA-/SO101 下）：
    python calib_poses_manual/check_305_depth_scale.py
"""
from __future__ import annotations

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

WRIST_COLOR = "/dev/v4l/by-path/pci-0000:0d:00.0-usb-0:6:1.4-video-index0"
WRIST_DEPTH = "/dev/v4l/by-path/pci-0000:0d:00.0-usb-0:6:1.0-video-index0"
WRIST_SN = "CV27864000V2"
OUT = HERE / "verify_rgbd_out"
SQ = 0.025
MIN_CORNERS = 15


def load(p):
    return json.load(open(p))


def read_joints():
    from atom_robot.ar5.drivers.rokae_arm import RokaeAR5
    arm = RokaeAR5(ArmConfig())
    arm.connect(power_on=False)
    q = np.asarray(arm.get_joint_positions(), float)[:7]
    arm.disconnect()
    return q


def capture_v4l2_raw():
    """返回 bgr, raw_uint16（不乘 scale）。"""
    import depth_v4l2 as D
    # 用 scale=1 拿到「按 mm 解释」的数，再除回去 = raw；更直接改读 raw
    depth_s = D.DepthStream(WRIST_DEPTH, 1.0, fps=30).start()  # 先当 1，后面用 latest 原始
    cap = cv2.VideoCapture(WRIST_COLOR, cv2.CAP_V4L2)
    if not cap.isOpened():
        depth_s.stop()
        raise RuntimeError("打不开腕部彩色")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    bgr = raw = None
    t0 = time.time()
    while time.time() - t0 < 8:
        ok, frame = cap.read()
        if depth_s.latest is not None and ok and frame is not None:
            bgr = frame
            raw = depth_s.latest.copy()
            for _ in range(20):
                ok, frame = cap.read()
                if ok and frame is not None and depth_s.latest is not None:
                    bgr, raw = frame, depth_s.latest.copy()
            break
        time.sleep(0.02)
    cap.release()
    depth_s.stop()
    if bgr is None or raw is None:
        raise RuntimeError("没拿到彩色+深度")
    return bgr, raw


def try_sdk_depth():
    """尽量用 pyorbbecsdk 读 305 一帧深度（mm）。失败返回 None。"""
    try:
        from pyorbbecsdk import Context, Pipeline, Config, OBSensorType, OBFormat
    except ImportError:
        print("[sdk] 无 pyorbbecsdk")
        return None
    try:
        ctx = Context()
        alist = ctx.query_devices()
        dev = None
        for i in range(alist.get_count()):
            d = alist.get_device_by_index(i)
            if d.get_device_info().get_serial_number() == WRIST_SN:
                dev = d
                break
        if dev is None:
            print(f"[sdk] 找不到序列号 {WRIST_SN}")
            return None
        scale = float(dev.get_device_info().get_device_name() and 0)  # placeholder
        # depth scale API
        try:
            scale = float(dev.get_depth_scale()) if hasattr(dev, "get_depth_scale") else None
        except Exception:
            scale = None
        pipe = Pipeline(dev)
        cfg = Config()
        try:
            profile_list = pipe.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
            profile = profile_list.get_video_stream_profile(640, 480, OBFormat.Y16, 30)
            cfg.enable_stream(profile)
        except Exception as e:
            print(f"[sdk] 配深度流失败：{e}")
            return None
        pipe.start(cfg)
        frame = None
        for _ in range(30):
            fs = pipe.wait_for_frames(1000)
            if fs is None:
                continue
            df = fs.get_depth_frame()
            if df is not None:
                frame = df
                break
        if frame is None:
            pipe.stop()
            print("[sdk] 等不到深度帧")
            return None
        # scale from frame if possible
        try:
            scale = float(frame.get_depth_scale())
        except Exception:
            pass
        data = np.frombuffer(frame.get_data(), dtype=np.uint16).reshape(frame.get_height(), frame.get_width())
        pipe.stop()
        print(f"[sdk] 拿到深度 {data.shape}，get_depth_scale={scale}")
        return data, scale
    except Exception as e:
        print(f"[sdk] 失败：{e}")
        return None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    wrist = load(HERE / "wrist_handeye.json")
    fac = load(ROOT / "算法/Jev/RobotJev/robojev/calibration/orbbec_params.json")
    factory_scale = float(fac["depth_scale_mm_per_unit"])
    print(f"出厂 depth_scale_mm_per_unit = {factory_scale}（SDK 读的）")

    print("只读连臂…")
    q = read_joints()
    raw_dh = load(SO101 / "atom_robot/ar5/ar5_dh.json")
    flat = raw_dh["dh"] if isinstance(raw_dh, dict) else raw_dh
    cfg = ArmConfig()
    kin = AR5Kinematics(np.array(flat, float).reshape(-1, 4).tolist(),
                        num_joints=cfg.num_joints,
                        joint_min=list(cfg.joint_min), joint_max=list(cfg.joint_max))
    T_base_cam = kin.fk(q) @ np.array(wrist["T_flange_camera"], float)
    print(f"  相机在基座 {np.round(T_base_cam[:3,3]*1000,1).tolist()} mm")

    print("采 V4L2 彩色 + 原始 Z16…")
    bgr, raw = capture_v4l2_raw()
    cv2.imwrite(str(OUT / "scale305_color.jpg"), bgr)
    np.save(OUT / "scale305_raw.npy", raw)
    valid = raw[raw > 0]
    print(f"  raw 有效 {valid.size/raw.size:.0%}，中位 {np.median(valid):.0f}，"
          f"×0.1 → {np.median(valid)*0.1:.0f} mm，×1 → {np.median(valid):.0f} mm")

    Kc = np.array(wrist["K"], float)
    dist = np.array(wrist["dist_k1_k2_p1_p2_k3"], float)
    di = fac["depth_intrinsic"]
    Kd = np.array([[di["fx"], 0, di["cx"]], [0, di["fy"], di["cy"]], [0, 0, 1.]])
    Rd2c = np.array(fac["depth_to_color"]["R"], float)
    td2c = np.array(fac["depth_to_color"]["t_mm"], float) / 1000.0

    A = cv2.aruco
    board = A.CharucoBoard((7, 9), SQ, 0.018, A.getPredefinedDictionary(A.DICT_5X5_50))
    board.setLegacyPattern(False)
    det = A.CharucoDetector(board)
    P3 = board.getChessboardCorners().astype(np.float64)
    ch, cid, _, _ = det.detectBoard(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
    if cid is None or len(cid) < MIN_CORNERS:
        raise RuntimeError(f"板角点不足：{0 if cid is None else len(cid)}（腕部要看得到板）")
    cid = np.asarray(cid).ravel()
    ch = np.asarray(ch, np.float64).reshape(-1, 2)
    print(f"  检出 {len(cid)} 个角点")

    ok, rvec, tvec = cv2.solvePnP(P3[cid].astype(np.float32), ch.astype(np.float32),
                                  Kc, dist, flags=cv2.SOLVEPNP_IPPE)
    Rcb = cv2.Rodrigues(rvec)[0]
    tcb = tvec.ravel()
    # 板平面（彩色相机系）：n·x = d；板 z 轴是法向
    n = Rcb[:, 2]
    d_plane = float(n @ tcb)
    print(f"  板原点深度 {np.linalg.norm(tcb)*1000:.1f} mm，法向 {np.round(n,3).tolist()}")

    # ── 1) 角点：期望 Z vs raw ──
    Pc = (Rcb @ P3[cid].T).T + tcb
    Z_true_c = Pc[:, 2] * 1000  # mm，彩色系 z ≈ 深度（光轴）
    Pd = (Rd2c.T @ (Pc - td2c).T).T
    uvd, _ = cv2.projectPoints(Pd, np.zeros(3), np.zeros(3), Kd, None)
    uvd = uvd.reshape(-1, 2)
    raw_c, z_c = [], []
    for (u, v), zt in zip(uvd, Z_true_c):
        ui, vi = int(round(u)), int(round(v))
        if not (0 <= ui < 640 and 0 <= vi < 480):
            continue
        patch = raw[max(0, vi - 2):vi + 3, max(0, ui - 2):ui + 3]
        patch = patch[patch > 0]
        if patch.size == 0:
            continue
        raw_c.append(float(np.median(patch)))
        z_c.append(float(zt))
    raw_c, z_c = np.array(raw_c), np.array(z_c)
    print(f"\n=== 角点对照（n={len(raw_c)}）===")
    for s in (0.1, 1.0, np.median(z_c / raw_c)):
        err = s * raw_c - z_c
        print(f"  scale={s:.4f} mm/unit：Δ 中位 {np.median(err):+.1f} mm，"
              f"RMS {np.sqrt(np.mean(err**2)):.1f} mm，比值中位 {np.median((s*raw_c)/z_c):.3f}")

    # ── 2) 板区域内所有深度点：射线-平面求交 ──
    # 深度像素 → 深度系 3D（先用单位方向，Z 用真值交点）→ 彩色系 → 与板求交得 Z_true
    ys, xs = np.where(raw > 0)
    # 降采样
    step = 2
    ys, xs = ys[::step], xs[::step]
    uv = np.stack([xs, ys], 1).astype(np.float64)
    und = cv2.undistortPoints(uv.reshape(-1, 1, 2), Kd, None).reshape(-1, 2)
    # 深度系单位射线
    dirs_d = np.c_[und, np.ones(len(und))]
    dirs_d /= np.linalg.norm(dirs_d, axis=1, keepdims=True)
    # 变到彩色系：x_c = R (λ * dir_d) + t ；与 n·x = d_plane 求交
    # n·(R λ dir + t) = d_plane → λ = (d_plane - n·t) / (n·R dir)
    Rt = Rd2c
    tt = td2c
    num = d_plane - n @ tt
    den = (n @ (Rt @ dirs_d.T))
    ok = np.abs(den) > 1e-6
    lam = np.full(len(dirs_d), np.nan)
    lam[ok] = num / den[ok]
    # 只要板前方、合理距离
    Pc_hit = (Rt @ (dirs_d * lam[:, None]).T).T + tt
    # 投到彩色，看是否在板凸包内
    hull = cv2.convexHull(ch.astype(np.float32))
    mask = np.zeros(bgr.shape[:2], np.uint8)
    cv2.fillConvexPoly(mask, hull.astype(np.int32), 255)
    mask = cv2.erode(mask, np.ones((9, 9), np.uint8))
    pr, _ = cv2.projectPoints(Pc_hit, np.zeros(3), np.zeros(3), Kc, dist)
    uv_c = pr.reshape(-1, 2)
    u_i = np.rint(uv_c[:, 0]).astype(int)
    v_i = np.rint(uv_c[:, 1]).astype(int)
    H, W = mask.shape
    on = (lam > 0.05) & (lam < 2.0) & (u_i >= 0) & (u_i < W) & (v_i >= 0) & (v_i < H)
    on[on] &= mask[v_i[on], u_i[on]] > 0
    # 深度传感器沿光轴的度量：对 Orbbec Z16，raw 通常是沿光轴的 Z（相机系 z），不是射线长
    # 深度系点 = und * Z_axis，所以期望 raw 对应的是 Pd_hit 的 z 分量 / scale
    Pd_hit = dirs_d * lam[:, None]  # 若 λ 是射线长；但上面 dirs 已单位化，λ=距离
    # 更正：深度图存的是 Z（光轴），反投影是 [x,y,1]*Z_axis。
    # 射线方向 dir_unnorm = [und,1]，点 = dir_unnorm * Z_axis，|dir_unnorm| 不一定 1。
    # 与平面相交应重做：
    dir_d = np.c_[und, np.ones(len(und))]          # 未归一化，点 = dir_d * Z
    # x_c = R (dir_d * Z) + t ；n·x_c = d_plane
    # Z * (n·R dir_d) = d_plane - n·t
    den2 = n @ (Rt @ dir_d.T)
    Z_axis = np.full(len(dir_d), np.nan)
    ok2 = np.abs(den2) > 1e-8
    Z_axis[ok2] = (d_plane - n @ tt) / den2[ok2]
    Pd2 = dir_d * Z_axis[:, None]
    Pc2 = (Rt @ Pd2.T).T + tt
    pr2, _ = cv2.projectPoints(Pc2, np.zeros(3), np.zeros(3), Kc, dist)
    uv2 = pr2.reshape(-1, 2)
    u2 = np.rint(uv2[:, 0]).astype(int)
    v2 = np.rint(uv2[:, 1]).astype(int)
    on2 = (Z_axis > 0.05) & (Z_axis < 2.0) & (u2 >= 0) & (u2 < W) & (v2 >= 0) & (v2 < H)
    on2[on2] &= mask[v2[on2], u2[on2]] > 0
    # 还要求命中点在板平面附近的「板坐标系」范围内：用角点包围盒放宽
    Pb_c = (Rcb @ P3.T).T + tcb
    # 用板系坐标：X_b = R^T (P - t)
    Xb = (Rcb.T @ (Pc2[on2] - tcb).T).T
    # Charuco 角点在板系大约 x∈[0,s*6], y∈[0,s*8]
    xs_b, ys_b = P3[:, 0], P3[:, 1]
    on_board = ((Xb[:, 0] > xs_b.min() - 0.01) & (Xb[:, 0] < xs_b.max() + 0.01) &
                (Xb[:, 1] > ys_b.min() - 0.01) & (Xb[:, 1] < ys_b.max() + 0.01) &
                (np.abs(Xb[:, 2]) < 0.01))
    idx = np.where(on2)[0][on_board]
    raw_b = raw[ys[idx], xs[idx]].astype(np.float64)
    z_b = Z_axis[idx] * 1000.0  # mm
    keep = raw_b > 0
    raw_b, z_b = raw_b[keep], z_b[keep]
    print(f"\n=== 板面深度像素对照（n={len(raw_b)}）===")
    if len(raw_b) < 50:
        print("  点太少，跳过拟合")
    else:
        # 纯比例
        a1 = float(np.median(z_b / raw_b))
        # 最小二乘 Z = a*raw + b
        A_ls = np.c_[raw_b, np.ones(len(raw_b))]
        a2, b2 = np.linalg.lstsq(A_ls, z_b, rcond=None)[0]
        for name, pred in (("出厂 0.1", 0.1 * raw_b),
                           (f"中位比 {a1:.4f}", a1 * raw_b),
                           (f"线性 {a2:.4f}*raw{b2:+.1f}", a2 * raw_b + b2)):
            err = pred - z_b
            print(f"  {name}：Δ 中位 {np.median(err):+.1f}，RMS {np.sqrt(np.mean(err**2)):.1f}，"
                  f"|Δ|<10mm 占比 {(np.abs(err)<10).mean():.0%}")

        # 分深度看比值是否稳定（区分比例误差 vs 固定偏置）
        print("  按期望深度分箱的 raw→mm 比值（中位）：")
        for lo, hi in [(200, 280), (280, 320), (320, 360), (360, 420), (420, 500)]:
            m = (z_b >= lo) & (z_b < hi)
            if m.sum() < 30:
                continue
            print(f"    Z_true [{lo},{hi}) mm：n={m.sum()}  ratio={np.median(z_b[m]/raw_b[m]):.4f}  "
                  f"0.1*raw 中位差 {np.median(0.1*raw_b[m]-z_b[m]):+.1f}")

    # 角点线性拟合
    if len(raw_c) >= 6:
        A_ls = np.c_[raw_c, np.ones(len(raw_c))]
        a_c, b_c = np.linalg.lstsq(A_ls, z_c, rcond=None)[0]
        print(f"\n角点最小二乘：Z_mm = {a_c:.4f} * raw + {b_c:.1f}")
        print(f"角点中位比值：{np.median(z_c/raw_c):.4f} mm/unit（出厂 0.1）")

    # SDK 对照（可能和 V4L2 抢设备，放最后；失败就跳过）
    print("\n尝试 SDK 读深度（可能失败/卡）…")
    sdk = try_sdk_depth()
    result = {
        "factory_scale_mm_per_unit": factory_scale,
        "n_corners": int(len(raw_c)),
        "corner_median_ratio_mm_per_raw": float(np.median(z_c / raw_c)) if len(raw_c) else None,
        "corner_median_err_with_0.1_mm": float(np.median(0.1 * raw_c - z_c)) if len(raw_c) else None,
        "n_board_pixels": int(len(raw_b)),
        "board_median_ratio_mm_per_raw": float(np.median(z_b / raw_b)) if len(raw_b) else None,
        "board_ls_a_b": [float(a2), float(b2)] if len(raw_b) >= 50 else None,
        "q_rad": q.tolist(),
        "raw_median": float(np.median(valid)),
        "sdk": None if sdk is None else {"scale": sdk[1], "raw_median": float(np.median(sdk[0][sdk[0] > 0]))},
    }
    if len(raw_b) >= 50:
        result["verdict"] = (
            f"V4L2 raw 应按 ~{np.median(z_b/raw_b):.4f} mm/unit 用，不是出厂 0.1；"
            f"或存在 ~{np.median(z_b-0.1*raw_b):.0f} mm 量级的系统偏差"
        )
    (OUT / "scale305_summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print("\n已写", OUT / "scale305_summary.json")
    print(result.get("verdict", ""))


if __name__ == "__main__":
    main()
