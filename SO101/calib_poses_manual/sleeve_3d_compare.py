#!/usr/bin/env python3
"""黑色套筒：两台相机 SAM 掩码 → 深度反投影到基座系，对比误差。

流程：
  1. 采顶部 + 腕部彩色/深度，只读关节角（不动臂）
  2. 每张图点一下套筒（左键正点，右键可加点排除，Enter 确认，r 重来）
  3. DexGraspVLA 的 SAM vit_h 出掩码
  4. 掩码内深度点 → 基座系，报中位坐标；两台互相比

用法（dexgraspvla 环境）：
    cd ~/yuhang_workspace/算法/DexGraspVLA-/SO101
    conda run -n dexgraspvla python calib_poses_manual/sleeve_3d_compare.py          # 鼠标点选
    conda run -n dexgraspvla python calib_poses_manual/sleeve_3d_compare.py --auto  # 自动找最黑的一块

若刚跑过 SDK/OBSENSOR，脚本会先复位 305 再开 V4L2。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

SO101 = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
ROOT = next(p for p in SO101.parents if (p / "算法").is_dir() and (p / "数据_硬件").is_dir())  # 工作区根
DEX = ROOT / "算法/DexGraspVLA-"
OUT = HERE / "sleeve_3d_out"
sys.path.insert(0, str(SO101))
sys.path.insert(0, str(ROOT / "算法/Jev/RobotDecision"))

from atom_robot.ar5.config import ArmConfig  # noqa: E402
from atom_robot.ar5.kinematics import AR5Kinematics  # noqa: E402

TOP_COLOR = "/dev/v4l/by-path/pci-0000:0d:00.0-usb-0:5:1.4-video-index0"
TOP_DEPTH = "/dev/v4l/by-path/pci-0000:0d:00.0-usb-0:5:1.0-video-index0"
WRIST_COLOR = "/dev/v4l/by-path/pci-0000:0d:00.0-usb-0:6:1.4-video-index0"
WRIST_DEPTH = "/dev/v4l/by-path/pci-0000:0d:00.0-usb-0:6:1.0-video-index0"
WRIST_SN = "CV27864000V2"
SAM_CKPT = DEX / "weights/sam_vit_h_4b8939.pth"


def load(p):
    return json.load(open(p))


def capture_rgbd(color_node, depth_node, mm_per_unit, label, n_warm=20):
    import depth_v4l2 as D
    print(f"打开 {label}…")
    ds = D.DepthStream(depth_node, 1.0, fps=30).start()  # 拿 raw，下面自己乘
    cap = cv2.VideoCapture(color_node, cv2.CAP_V4L2)
    if not cap.isOpened():
        ds.stop()
        raise RuntimeError(f"打不开 {color_node}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    bgr = raw = None
    t0 = time.time()
    while time.time() - t0 < 8:
        ok, frame = cap.read()
        if ok and frame is not None and ds.latest is not None:
            bgr, raw = frame, ds.latest.copy()
            for _ in range(n_warm):
                ok, frame = cap.read()
                if ok and frame is not None and ds.latest is not None:
                    bgr, raw = frame, ds.latest.copy()
            break
        time.sleep(0.02)
    cap.release()
    ds.stop()
    if bgr is None:
        raise RuntimeError(f"{label} 没拿到帧")
    mm = raw.astype(np.float32) * float(mm_per_unit)
    v = mm[mm > 0]
    print(f"  {label} 深度有效 {v.size/mm.size:.0%} 中位 {np.median(v):.0f} mm（×{mm_per_unit}）")
    return bgr, mm, raw


def read_joints():
    from atom_robot.ar5.drivers.rokae_arm import RokaeAR5
    arm = RokaeAR5(ArmConfig())
    arm.connect(power_on=False)
    q = np.asarray(arm.get_joint_positions(), float)[:7]
    arm.disconnect()
    return q


def auto_black_point(img_bgr, name):
    """在画面里找最黑的连通块中心，当作 SAM 正点（套筒是黑的）。"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # 排除画面边缘（腕部常有黑手指/外壳）
    h, w = gray.shape
    roi = np.zeros_like(gray)
    if name == "wrist":
        roi[int(0.05 * h):int(0.75 * h), int(0.15 * w):int(0.85 * w)] = 255
    else:
        roi[int(0.15 * h):int(0.85 * h), int(0.15 * w):int(0.85 * w)] = 255
    # 相对暗：比 ROI 内中位暗很多
    med = np.median(gray[roi > 0])
    thr = min(80, max(40, med * 0.55))
    dark = ((gray < thr) & (roi > 0)).astype(np.uint8) * 255
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, labels, stats, cents = cv2.connectedComponentsWithStats(dark, 8)
    if n <= 1:
        print(f"  [{name}] 自动没找到黑块（thr={thr:.0f}）")
        return None, None
    # 面积在 200～25000 之间的最黑块（用块内平均灰度）
    best = None
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 200 or area > 25000:
            continue
        mean_g = gray[labels == i].mean()
        score = (-mean_g, -area)  # 更黑优先，同黑取更大
        if best is None or score < best[0]:
            best = (score, cents[i], area, mean_g)
    if best is None:
        print(f"  [{name}] 黑块面积都不合适")
        return None, None
    cx, cy = best[1]
    print(f"  [{name}] 自动黑块中心 ({cx:.0f},{cy:.0f}) 面积 {best[2]} 灰度 {best[3]:.0f}")
    return np.array([[cx, cy]], float), np.array([1], int)


def click_points(img_bgr, title):
    """左键正点、右键负点、Enter 确认、r 清空、q 跳过。返回 (points Nx2, labels N)。"""
    vis0 = img_bgr.copy()
    state = {"pts": [], "labs": []}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["pts"].append([x, y])
            state["labs"].append(1)
        elif event == cv2.EVENT_RBUTTONDOWN:
            state["pts"].append([x, y])
            state["labs"].append(0)

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(title, on_mouse)
    print(f"  [{title}] 左键点套筒，右键排除，Enter 确认，r 重来，q 跳过")
    while True:
        vis = vis0.copy()
        for (x, y), lab in zip(state["pts"], state["labs"]):
            cv2.circle(vis, (x, y), 6, (0, 255, 0) if lab else (0, 0, 255), -1)
        cv2.putText(vis, "L:pos R:neg Enter:ok r:reset q:skip", (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
        cv2.imshow(title, vis)
        k = cv2.waitKey(20) & 0xFF
        if k in (13, 10):  # Enter
            break
        if k == ord("r"):
            state["pts"], state["labs"] = [], []
        if k == ord("q"):
            cv2.destroyWindow(title)
            return None, None
    cv2.destroyWindow(title)
    if not state["pts"]:
        return None, None
    return np.array(state["pts"], float), np.array(state["labs"], int)


def run_sam(predictor, img_bgr, points, labels):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(rgb)
    masks, scores, _ = predictor.predict(
        point_coords=points, point_labels=labels, multimask_output=True)
    i = int(np.argmax(scores))
    return masks[i].astype(bool), float(scores[i])


def backproject_via_d2c(mm, mask, Kd, Rd2c, td2c, Kc, dist, T_base_cam,
                        z_min=50, z_max=2000, stride=2):
    """深度未对齐彩色：全部有效深度点 → 彩色系 → 落在 SAM 掩码内的再进基座。"""
    ys, xs = np.where((mm > z_min) & (mm < z_max))
    if stride > 1:
        ys, xs = ys[::stride], xs[::stride]
    if len(xs) == 0:
        return np.zeros((0, 3))
    uv = np.stack([xs, ys], 1).astype(np.float64)
    Z = mm[ys, xs] / 1000.0
    und = cv2.undistortPoints(uv.reshape(-1, 1, 2), Kd, None).reshape(-1, 2)
    Pd = np.c_[und * Z[:, None], Z]
    Pc = (Rd2c @ Pd.T).T + td2c
    pr, _ = cv2.projectPoints(Pc, np.zeros(3), np.zeros(3), Kc, dist)
    uv_c = pr.reshape(-1, 2)
    u = np.rint(uv_c[:, 0]).astype(int)
    v = np.rint(uv_c[:, 1]).astype(int)
    H, W = mask.shape
    keep = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    keep[keep] = mask[v[keep], u[keep]]
    Pc = Pc[keep]
    if len(Pc) == 0:
        return np.zeros((0, 3))
    return (T_base_cam[:3, :3] @ Pc.T).T + T_base_cam[:3, 3]


def load_d2c(path):
    d = load(path)
    di = d["depth_intrinsic"]
    Kd = np.array([[di["fx"], 0, di["cx"]], [0, di["fy"], di["cy"]], [0, 0, 1.]])
    R = np.array(d["depth_to_color"]["R"], float)
    t = np.array(d["depth_to_color"]["t_mm"], float) / 1000.0
    return Kd, R, t


def summarize(name, P):
    if len(P) < 20:
        print(f"[{name}] 有效点太少：{len(P)}")
        return None
    med = np.median(P, 0)
    # 去掉离中位太远的飞点（深度噪声）
    d = np.linalg.norm(P - med, axis=1)
    thr = max(0.03, np.percentile(d, 80))
    P2 = P[d < thr]
    med = np.median(P2, 0)
    std = P2.std(0)
    print(f"[{name}] n={len(P2)}/{len(P)}  中位基座系 "
          f"{np.round(med*1000,1).tolist()} mm  1σ {np.round(std*1000,1).tolist()} mm")
    return {"n": int(len(P2)), "n_raw": int(len(P)),
            "median_mm": (med * 1000).tolist(), "std_mm": (std * 1000).tolist(),
            "P": P2}


def overlay_mask(img, mask, color=(0, 255, 0)):
    out = img.copy()
    out[mask] = (0.45 * out[mask] + 0.55 * np.array(color)).astype(np.uint8)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--auto", action="store_true", help="自动找最黑的连通块当 SAM 正点（黑套筒）")
    ap.add_argument("--no-reset", action="store_true", help="不复位 305")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    wrist = load(HERE / "wrist_handeye.json")
    top = load(HERE / "top_extrinsic.json")
    Kd335, R335, t335 = load_d2c(ROOT / "算法/Jev/RobotJev/robojev/calibration/orbbec335_params.json")
    Kd305, R305, t305 = load_d2c(ROOT / "算法/Jev/RobotJev/robojev/calibration/orbbec_params.json")

    if not args.no_reset:
        print("复位腕部 305（避免 V4L2 深度被污染）…")
        from atom_robot.ar5.drivers.cameras import _reset_orbbec
        _reset_orbbec(WRIST_SN, wait_s=8.0)

    print("只读关节角…")
    q = read_joints()
    print(f"  q_deg={np.round(np.degrees(q),1).tolist()}")
    raw_dh = load(SO101 / "atom_robot/ar5/ar5_dh.json")
    flat = raw_dh["dh"] if isinstance(raw_dh, dict) else raw_dh
    cfg = ArmConfig()
    kin = AR5Kinematics(np.array(flat, float).reshape(-1, 4).tolist(),
                        num_joints=cfg.num_joints,
                        joint_min=list(cfg.joint_min), joint_max=list(cfg.joint_max))
    T_base_cam_wrist = kin.fk(q) @ np.array(wrist["T_flange_camera"], float)
    T_base_cam_top = np.array(top["T_base_cam"], float)
    print(f"  腕部相机位置 {np.round(T_base_cam_wrist[:3,3]*1000,1).tolist()} mm")
    print(f"  顶部相机位置 {np.round(T_base_cam_top[:3,3]*1000,1).tolist()} mm")

    top_bgr, top_mm, top_raw = capture_rgbd(TOP_COLOR, TOP_DEPTH, 1.0, "顶部335")
    wrist_bgr, wrist_mm, wrist_raw = capture_rgbd(WRIST_COLOR, WRIST_DEPTH, 0.1, "腕部305")
    cv2.imwrite(str(OUT / "top_color.jpg"), top_bgr)
    cv2.imwrite(str(OUT / "wrist_color.jpg"), wrist_bgr)
    np.savez(OUT / "capture.npz", top_mm=top_mm, wrist_mm=wrist_mm,
             top_raw=top_raw, wrist_raw=wrist_raw, q=q,
             T_base_cam_wrist=T_base_cam_wrist, T_base_cam_top=T_base_cam_top)

    print("加载 SAM vit_h…")
    from segment_anything import SamPredictor, sam_model_registry
    sam = sam_model_registry["vit_h"](checkpoint=str(SAM_CKPT))
    sam.to(device="cuda")
    predictor = SamPredictor(sam)

    results = {}
    for name, bgr, mm, Kd, Rd2c, td2c, Kc, dist, Tbc in (
        ("top", top_bgr, top_mm, Kd335, R335, t335,
         np.array(top["K"]), np.array(top["dist_k1_k2_p1_p2_k3"]), T_base_cam_top),
        ("wrist", wrist_bgr, wrist_mm, Kd305, R305, t305,
         np.array(wrist["K"]), np.array(wrist["dist_k1_k2_p1_p2_k3"]), T_base_cam_wrist),
    ):
        if args.auto:
            pts, labs = auto_black_point(bgr, name)
        else:
            pts, labs = click_points(bgr, f"SAM-{name}")
        if pts is None:
            print(f"[{name}] 跳过")
            continue
        mask, score = run_sam(predictor, bgr, pts, labs)
        print(f"[{name}] SAM score={score:.3f} 掩码像素 {int(mask.sum())}")
        cv2.imwrite(str(OUT / f"{name}_mask.jpg"), overlay_mask(bgr, mask))
        np.save(OUT / f"{name}_mask.npy", mask)
        # 画提示点
        vis_p = overlay_mask(bgr, mask)
        for (x, y), lab in zip(pts, labs):
            cv2.circle(vis_p, (int(x), int(y)), 6, (0, 255, 0) if lab else (0, 0, 255), -1)
        cv2.imwrite(str(OUT / f"{name}_prompt.jpg"), vis_p)

        P = backproject_via_d2c(mm, mask, Kd, Rd2c, td2c, Kc, dist, Tbc)
        info = summarize(name, P)
        if info is not None:
            np.save(OUT / f"{name}_points.npy", info.pop("P"))
            results[name] = info
            med = np.array(info["median_mm"]) / 1000.0
            Pc = np.linalg.inv(Tbc)[:3, :3] @ med + np.linalg.inv(Tbc)[:3, 3]
            pr, _ = cv2.projectPoints(Pc.reshape(1, 3), np.zeros(3), np.zeros(3), Kc, dist)
            u, v = pr.reshape(2).astype(int)
            vis = overlay_mask(bgr, mask)
            cv2.drawMarker(vis, (u, v), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
            cv2.putText(vis, f"{np.round(info['median_mm'],1)}", (u + 8, v - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.imwrite(str(OUT / f"{name}_result.jpg"), vis)

    if "top" in results and "wrist" in results:
        a = np.array(results["top"]["median_mm"])
        b = np.array(results["wrist"]["median_mm"])
        d = b - a
        print(f"\n=== 对比（腕部 − 顶部）中位坐标差 {np.round(d,1).tolist()} mm，"
              f"|Δ|={np.linalg.norm(d):.1f} mm，水平 {np.linalg.norm(d[:2]):.1f} mm，竖向 {d[2]:+.1f} mm ===")
        results["delta_wrist_minus_top_mm"] = d.tolist()
        results["delta_norm_mm"] = float(np.linalg.norm(d))

    payload = {
        "auto": bool(args.auto),
        "q_rad": q.tolist(),
        "T_base_cam_wrist": T_base_cam_wrist.tolist(),
        "results": results,
        "note": "中位点是掩码内深度点（去飞点后）的中位；黑套筒可见表面，不是几何中心",
    }
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print("结果目录", OUT)
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass


if __name__ == "__main__":
    main()
