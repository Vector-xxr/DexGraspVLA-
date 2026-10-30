#!/usr/bin/env python3
"""开环 3D 增益验证（不动网络）。

Step1（默认，不动臂）：顶部 RGB-D + 点选 SAM → 基座系 p_base；可选腕部交叉验证；
      算出预抓取点建议（物体上方），只打印不运动。
Step2（--step 2，动臂）：读 step1 结果，关节插值到预抓取点上方再下降，量水平误差。

用法（dexgraspvla，有显示）：
    cd ~/yuhang_workspace/算法/DexGraspVLA-/SO101
    conda run -n dexgraspvla python calib_poses_manual/verify_openloop_3d.py --step 1
    conda run -n dexgraspvla python calib_poses_manual/verify_openloop_3d.py --step 2 --above 0.12
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
DEX = ROOT / "算法/DexGraspVLA-"
OUT = HERE / "openloop_3d_out"
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
STEP1_JSON = OUT / "step1_result.json"


def load(p):
    return json.load(open(p))


def load_d2c(path):
    d = load(path)
    di = d["depth_intrinsic"]
    Kd = np.array([[di["fx"], 0, di["cx"]], [0, di["fy"], di["cy"]], [0, 0, 1.]])
    R = np.array(d["depth_to_color"]["R"], float)
    t = np.array(d["depth_to_color"]["t_mm"], float) / 1000.0
    return Kd, R, t


def capture_rgbd(color_node, depth_node, mm_per_unit, label):
    import depth_v4l2 as D
    print(f"打开 {label}…")
    ds = D.DepthStream(depth_node, 1.0, fps=30).start()
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
            for _ in range(20):
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
    print(f"  深度有效 {v.size/mm.size:.0%} 中位 {np.median(v):.0f} mm")
    return bgr, mm


def read_joints(power_on=False):
    from atom_robot.ar5.drivers.rokae_arm import RokaeAR5
    arm = RokaeAR5(ArmConfig())
    arm.connect(power_on=power_on)
    q = np.asarray(arm.get_joint_positions(), float)[:7]
    return arm, q


def click_points(img_bgr, title):
    vis0 = img_bgr.copy()
    state = {"pts": [], "labs": []}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["pts"].append([x, y]); state["labs"].append(1)
        elif event == cv2.EVENT_RBUTTONDOWN:
            state["pts"].append([x, y]); state["labs"].append(0)

    # 标题只用 ASCII：OpenCV5+Qt 对非 ASCII 窗口名会 NULL window handler
    win = "".join(c if ord(c) < 128 else "_" for c in title) or "click"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.imshow(win, vis0)          # 先真正创建窗口再绑鼠标
    cv2.waitKey(1)
    cv2.setMouseCallback(win, on_mouse)
    print(f"  [{title}] 左键点目标，右键排除，Enter 确认，r 重来，q 跳过")
    while True:
        vis = vis0.copy()
        for (x, y), lab in zip(state["pts"], state["labs"]):
            cv2.circle(vis, (x, y), 6, (0, 255, 0) if lab else (0, 0, 255), -1)
        cv2.putText(vis, "L:pos R:neg Enter:ok r:reset q:skip", (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
        cv2.imshow(win, vis)
        k = cv2.waitKey(20) & 0xFF
        if k in (13, 10):
            break
        if k == ord("r"):
            state["pts"], state["labs"] = [], []
        if k == ord("q"):
            cv2.destroyWindow(win)
            return None, None
    cv2.destroyWindow(win)
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


def summarize_cloud(P, name):
    if len(P) < 20:
        print(f"[{name}] 有效点太少：{len(P)}")
        return None
    med = np.median(P, 0)
    d = np.linalg.norm(P - med, axis=1)
    thr = max(0.03, np.percentile(d, 80))
    P2 = P[d < thr]
    med = np.median(P2, 0)
    std = P2.std(0)
    # 掩码内同像素深度有效率粗看：用点云 z 离散
    print(f"[{name}] n={len(P2)}/{len(P)}  p_base={np.round(med*1000,1).tolist()} mm  "
          f"1σ={np.round(std*1000,1).tolist()} mm")
    return {"n": int(len(P2)), "n_raw": int(len(P)),
            "p_base_m": med.tolist(), "p_base_mm": (med * 1000).tolist(),
            "std_mm": (std * 1000).tolist()}


def overlay_mask(img, mask, color=(0, 255, 0)):
    out = img.copy()
    out[mask] = (0.45 * out[mask] + 0.55 * np.array(color)).astype(np.uint8)
    return out


def make_kin():
    raw = load(SO101 / "atom_robot/ar5/ar5_dh.json")
    flat = raw["dh"] if isinstance(raw, dict) else raw
    cfg = ArmConfig()
    return AR5Kinematics(np.array(flat, float).reshape(-1, 4).tolist(),
                         num_joints=cfg.num_joints,
                         joint_min=list(cfg.joint_min),
                         joint_max=list(cfg.joint_max)), cfg


def step1(args):
    OUT.mkdir(parents=True, exist_ok=True)
    wrist = load(HERE / "wrist_handeye.json")
    top = load(HERE / "top_extrinsic.json")
    Kd335, R335, t335 = load_d2c(ROOT / "算法/Jev/RobotJev/robojev/calibration/orbbec335_params.json")
    Kd305, R305, t305 = load_d2c(ROOT / "算法/Jev/RobotJev/robojev/calibration/orbbec_params.json")
    T_base_cam_top = np.array(top["T_base_cam"], float)

    if not args.no_reset:
        print("复位腕部 305…")
        from atom_robot.ar5.drivers.cameras import _reset_orbbec
        _reset_orbbec(WRIST_SN, wait_s=8.0)

    print("只读关节角（不上电）…")
    arm, q = read_joints(power_on=False)
    arm.disconnect()
    kin, _ = make_kin()
    T_bf = kin.fk(q)
    T_base_cam_wrist = T_bf @ np.array(wrist["T_flange_camera"], float)
    flange = T_bf[:3, 3]
    print(f"  法兰 {np.round(flange*1000,1).tolist()} mm")

    top_bgr, top_mm = capture_rgbd(TOP_COLOR, TOP_DEPTH, 1.0, "顶部335")
    wrist_bgr, wrist_mm = capture_rgbd(WRIST_COLOR, WRIST_DEPTH, 0.1, "腕部305")
    cv2.imwrite(str(OUT / "step1_top_color.jpg"), top_bgr)
    cv2.imwrite(str(OUT / "step1_wrist_color.jpg"), wrist_bgr)

    print("加载 SAM…")
    from segment_anything import SamPredictor, sam_model_registry
    sam = sam_model_registry["vit_h"](checkpoint=str(SAM_CKPT))
    sam.to(device="cuda")
    predictor = SamPredictor(sam)

    pts, labs = click_points(top_bgr, "Step1-TOP click target")
    if pts is None:
        raise SystemExit("顶部未点选，退出")
    mask, score = run_sam(predictor, top_bgr, pts, labs)
    print(f"[top] SAM score={score:.3f} 掩码 {int(mask.sum())} px")
    ys, xs = np.where(mask)
    depth_in = top_mm[ys, xs]
    valid = (depth_in > 50) & (depth_in < 2000)
    depth_valid_frac = float(valid.mean()) if len(depth_in) else 0.0
    depth_med = float(np.median(depth_in[valid])) if valid.any() else None
    print(f"[top] 掩码内深度有效 {depth_valid_frac:.0%} 中位 {depth_med if depth_med is not None else float('nan'):.0f} mm")
    cv2.imwrite(str(OUT / "step1_top_mask.jpg"), overlay_mask(top_bgr, mask))

    P = backproject_via_d2c(top_mm, mask, Kd335, R335, t335,
                            np.array(top["K"]), np.array(top["dist_k1_k2_p1_p2_k3"]),
                            T_base_cam_top)
    top_info = summarize_cloud(P, "top")
    if top_info is None:
        raise SystemExit("顶部 3D 失败")
    top_info["depth_valid_frac"] = depth_valid_frac
    top_info["depth_med_mm"] = depth_med
    top_info["sam_score"] = score
    top_info["mask_px"] = int(mask.sum())

    wrist_info = None
    if not args.top_only:
        pts_w, labs_w = click_points(wrist_bgr, "Step1-WRIST same object (q skip)")
        if pts_w is not None:
            mask_w, score_w = run_sam(predictor, wrist_bgr, pts_w, labs_w)
            print(f"[wrist] SAM score={score_w:.3f} 掩码 {int(mask_w.sum())} px")
            cv2.imwrite(str(OUT / "step1_wrist_mask.jpg"), overlay_mask(wrist_bgr, mask_w))
            Pw = backproject_via_d2c(wrist_mm, mask_w, Kd305, R305, t305,
                                     np.array(wrist["K"]), np.array(wrist["dist_k1_k2_p1_p2_k3"]),
                                     T_base_cam_wrist)
            wrist_info = summarize_cloud(Pw, "wrist")

    p = np.array(top_info["p_base_m"])
    above = float(args.above)
    pregrasp = p.copy()
    pregrasp[2] = p[2] + above
    # 朝向：沿用当前法兰姿态（只改位置）
    T_pre = np.eye(4)
    T_pre[:3, :3] = T_bf[:3, :3]
    T_pre[:3, 3] = pregrasp
    # 夹持中心近似 = 法兰；若以后有 gripper offset 再减
    q_pre, ok = None, False
    try:
        q_pre, ik_info = kin.ik(T_pre, q)
        ok = q_pre is not None and np.all(np.isfinite(q_pre))
        if not ok:
            print(f"  预抓取 IK 失败：{ik_info.get('reason')} "
                  f"pos_err={ik_info.get('pos_error'):.4f} rot_err={ik_info.get('rot_error'):.4f}")
        else:
            print(f"  预抓取 IK OK  pos_err={ik_info.get('pos_error'):.5f} m")
    except Exception as e:
        print(f"  预抓取 IK 异常：{e}")
        q_pre = None

    delta_tw = None
    if wrist_info is not None:
        d = np.array(wrist_info["p_base_mm"]) - np.array(top_info["p_base_mm"])
        delta_tw = d.tolist()
        print(f"\n交叉验证 腕−顶 = {np.round(d,1).tolist()} mm  |Δ|={np.linalg.norm(d):.1f} mm")

    horiz = np.linalg.norm(p[:2] - flange[:2]) * 1000
    print(f"\n=== Step1 结果 ===")
    print(f"  顶部 p_base     {np.round(p*1000,1).tolist()} mm")
    print(f"  建议预抓取点    {np.round(pregrasp*1000,1).tolist()} mm（物体上方 {above*1000:.0f} mm）")
    print(f"  当前法兰水平距目标 {horiz:.1f} mm")
    print(f"  预抓取 IK       {'OK ' + np.round(np.degrees(q_pre),1).tolist().__str__() if ok else '失败'}")

    result = {
        "step": 1,
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "q_rad": q.tolist(),
        "flange_mm": (flange * 1000).tolist(),
        "top": top_info,
        "wrist": wrist_info,
        "delta_wrist_minus_top_mm": delta_tw,
        "pregrasp_m": pregrasp.tolist(),
        "pregrasp_mm": (pregrasp * 1000).tolist(),
        "above_m": above,
        "ik_ok": bool(ok),
        "q_pregrasp_rad": q_pre.tolist() if ok else None,
        "note": "p_base=掩码深度点中位（可见表面）；预抓取=同 xy、z+above；Step2 用此文件动臂",
    }
    STEP1_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"已写 {STEP1_JSON}")
    print("Step1 完成。确认数字合理后跑：--step 2")
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass


def step2(args):
    if not STEP1_JSON.exists():
        raise SystemExit(f"没有 {STEP1_JSON}，先跑 --step 1")
    result = load(STEP1_JSON)
    if not result.get("ik_ok") or result.get("q_pregrasp_rad") is None:
        raise SystemExit("Step1 的预抓取 IK 失败，不能动臂。重新 Step1 或改 --above")
    p = np.array(result["top"]["p_base_m"])
    q_pre = np.array(result["q_pregrasp_rad"], float)
    above = float(args.above) if args.above else float(result["above_m"])
    if abs(above - result["above_m"]) > 1e-6:
        # 用户改了高度，重算 IK
        kin, _ = make_kin()
        arm, q = read_joints(power_on=False)
        arm.disconnect()
        T_bf = kin.fk(q)
        pregrasp = p.copy(); pregrasp[2] = p[2] + above
        T_pre = np.eye(4); T_pre[:3, :3] = T_bf[:3, :3]; T_pre[:3, 3] = pregrasp
        q_pre, ik_info = kin.ik(T_pre, q)
        if q_pre is None:
            raise SystemExit(f"新高度 IK 失败：{ik_info}")
        print(f"按新 above={above} 重算 IK")
        result["pregrasp_mm"] = (pregrasp * 1000).tolist()

    print("=== Step2：开环走到预抓取（会动臂）===")
    print(f"  目标 p_base {np.round(p*1000,1).tolist()} mm")
    print(f"  预抓取      {np.round(np.array(result['pregrasp_mm']),1).tolist()} mm")
    print(f"  q_pre deg   {np.round(np.degrees(q_pre),1).tolist()}")
    if not args.yes:
        ans = input("确认动臂？输入 yes 继续：").strip()
        if ans != "yes":
            print("已取消")
            return

    from atom_robot.ar5.drivers.rokae_arm import RokaeAR5
    cfg = ArmConfig()
    cfg.use_realtime = False   # 开环用队列 MoveAbsJ；RT 模式下 moveAppend 会 ec=262
    arm = RokaeAR5(cfg)
    arm.connect(power_on=True)
    # 上次若卡在 RT 模式，显式切回 NRT，否则 moveAppend → ec=262
    try:
        arm._call(arm._robot.setMotionControlMode, arm.sdk.MotionControlMode.NrtCommandMode)
    except Exception as e:
        print(f"  切 NrtCommandMode 警告：{e}")
    q0 = np.asarray(arm.get_joint_positions(), float)[:7]
    print(f"  当前 q deg {np.round(np.degrees(q0),1).tolist()}")
    print("  开始 move_to_joints → 预抓取…")
    arm.move_to_joints(q_pre, blocking=True)
    time.sleep(0.5)
    q1 = np.asarray(arm.get_joint_positions(), float)[:7]
    kin, _ = make_kin()
    flange1 = kin.fk(q1)[:3, 3]
    err_xy = np.linalg.norm(flange1[:2] - p[:2]) * 1000
    err_z = (flange1[2] - (p[2] + above)) * 1000
    print(f"  到位法兰 {np.round(flange1*1000,1).tolist()} mm")
    print(f"  相对目标水平误差 {err_xy:.1f} mm，相对预抓取高度 Δz {err_z:+.1f} mm")
    out = {
        "step": 2,
        "p_base_mm": (p * 1000).tolist(),
        "flange_after_mm": (flange1 * 1000).tolist(),
        "err_xy_mm": float(err_xy),
        "err_z_to_pregrasp_mm": float(err_z),
        "q_before_deg": np.degrees(q0).tolist(),
        "q_after_deg": np.degrees(q1).tolist(),
    }
    (OUT / "step2_result.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"已写 {OUT/'step2_result.json'}")
    print("Step2 只到预抓取，不合爪。不用就手动/脚本回零。")
    arm.disconnect()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, choices=[1, 2], default=1)
    ap.add_argument("--above", type=float, default=0.12, help="预抓取在物体上方多少米")
    ap.add_argument("--top-only", action="store_true", help="Step1 不点腕部")
    ap.add_argument("--no-reset", action="store_true")
    ap.add_argument("--yes", action="store_true", help="Step2 跳过交互确认")
    args = ap.parse_args()
    if args.step == 1:
        step1(args)
    else:
        step2(args)


if __name__ == "__main__":
    main()
