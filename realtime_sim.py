"""C2 / C3（待测试.md）：用录好的视频模拟实时流，测 SAM + Cutie + 控制器同时常驻时的速度、显存和掩码跟踪。

只读数据和检查点，不碰任何硬件。尽量直接用上游 inference.py 的代码：
- SAM / Cutie 的建法照 init_controller（inference.py:134-145），只把 SAM 权重路径换成本机的
- 首帧分割、逐帧跟踪直接调 RoboticsSystem.initialize_sam_cutie / get_mask：建一个不调 __init__
  （所以不连硬件）的 RoboticsSystem 对象，只填这两个方法要用的属性
- 控制器的加载、观测处理、预测和 offline_replay.py 一样

两部分：
1. 逐帧跟踪（C3）：每条第 0 帧用 zarr 里记的目标框做 SAM，之后每一帧都跑 Cutie，和数据集里的掩码比 IoU。
   数据集的掩码是转换时离线生成的，训练用的就是它；看推理时生成的和训练时的是不是一回事、整段跟没跟丢。
2. 实时节奏（C2）：照 inference.py 的 execute_grasping / execute_action 跑循环——
   读图 → 更新掩码 → 预测 64 步 → 执行前 6 步，每步之前再读图、更新一次掩码。
   「读图」按墙上时间取录像里对应的那一帧（30 Hz），循环慢了就会跳帧，和真机一样。
   录像里的臂不会跟着预测走（开环），所以只看节奏和跟踪，不看动作对不对。

用法（在仓库根目录）：
  PYTHONPATH= ~/miniforge3/envs/dexgraspvla/bin/python realtime_sim.py \\
      --ckpt weights/epoch_0124-train_loss_0.001.ckpt \\
      --zarr ~/H100_dexgraspvla/Datasets/atom_black_sleeve_20261021/black_sleeve.zarr
输出在 data/outputs/realtime_sim/<--name>/：summary.json、IoU 最低的几帧对照图、SAM 首帧图。
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import offline_replay  # noqa: E402  先 import：设好 TORCH_HOME、matplotlib 用 Agg

import cv2
import numpy as np
import torch
import zarr
from cutie.inference.inference_core import InferenceCore
from cutie.utils.get_default_model import get_default_model
from segment_anything import SamPredictor, sam_model_registry

from controller.common.pytorch_util import dict_apply
from inference import RoboticsSystem

CAM_HZ = 30          # 采集帧率，模拟实时流时按它取帧
EXEC_STEPS = offline_replay.EXEC_STEPS


def make_system(device, out_dir):
    rs = RoboticsSystem.__new__(RoboticsSystem)  # 不调 __init__：不连臂、手、相机
    rs.config = RoboticsSystem.load_inference_config(rs)
    rs.device = device
    rs.img_dir = str(out_dir)
    rs.log_file = open(out_dir / "run.log", "w")
    # 以下照 init_controller（inference.py:134-145）
    rs.sam = sam_model_registry[rs.config["sam"]["model_type"]](checkpoint=str(HERE / "weights" / "sam_vit_h_4b8939.pth"))
    rs.sam.to(device=device)
    rs.predictor = SamPredictor(rs.sam)
    rs.cutie = get_default_model()
    rs.processor = InferenceCore(rs.cutie, cfg=rs.cutie.cfg)
    rs.processor.max_internal_size = -1
    return rs


def iou(a, b):
    u = np.logical_or(a, b).sum()
    return 1.0 if u == 0 else float(np.logical_and(a, b).sum() / u)


def sam_init(rs, frame, bbox):
    """返回 (SAM 本身耗时, initialize_sam_cutie 整个耗时)。后者含上游存图和 plt.pause(0.5)。"""
    rs.third_color_image = frame
    torch.cuda.synchronize(); t0 = time.perf_counter()
    rs.predictor.set_image(frame[..., ::-1])
    rs.predictor.predict(box=bbox, multimask_output=True)
    torch.cuda.synchronize(); t_sam = time.perf_counter() - t0
    t0 = time.perf_counter()
    RoboticsSystem.initialize_sam_cutie(rs, bbox)
    torch.cuda.synchronize()
    return t_sam, time.perf_counter() - t0


def get_mask_timed(rs, frame):
    rs.third_color_image = frame
    torch.cuda.synchronize(); t0 = time.perf_counter()
    m = RoboticsSystem.get_mask(rs)
    torch.cuda.synchronize()
    return m, time.perf_counter() - t0


def save_pair(path, frame, m_data, m_online, title):
    a, b = frame.copy(), frame.copy()
    a[m_data > 0] = (0.5 * a[m_data > 0] + [127, 0, 0]).astype(np.uint8)
    b[m_online > 0] = (0.5 * b[m_online > 0] + [0, 0, 127]).astype(np.uint8)
    img = np.hstack([a, b])
    cv2.putText(img, f"{title}  left=dataset(blue) right=online(red)", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    cv2.imwrite(str(path), img)


def part_tracking(rs, z, starts, ends, names, boxes, eps, out):
    rgbm = z["data/rgbm"]
    per_ep, worst, t_mask, t_sam, t_init = [], [], [], [], []
    for ep in eps:
        s, e = int(starts[ep]), int(ends[ep])
        bbox = np.array(boxes[names[ep]], dtype=np.float32)
        ious, both_empty, online_empty_only, dataset_empty_only = [], 0, 0, 0
        for c0 in range(s, e, 64):
            chunk = rgbm[c0:min(c0 + 64, e)]
            for j, f in enumerate(chunk):
                frame, m_data = np.ascontiguousarray(f[..., :3]), f[..., 3]
                if c0 + j == s:
                    ts, ti = sam_init(rs, frame, bbox)
                    t_sam.append(ts); t_init.append(ti)
                m, dt = get_mask_timed(rs, frame)
                t_mask.append(dt)
                de, oe = m_data.sum() == 0, m.sum() == 0
                if de and oe:
                    both_empty += 1
                    continue
                if de != oe:
                    online_empty_only += int(oe); dataset_empty_only += int(de)
                v = iou(m_data > 0, m > 0)
                ious.append(v)
                worst.append((v, ep, c0 + j - s, frame, m_data, m))
                worst = sorted(worst, key=lambda x: x[0])[:6]
        ious = np.array(ious)
        per_ep.append(dict(ep=ep, frames=e - s, iou_mean=float(ious.mean()), iou_min=float(ious.min()),
                           iou_p5=float(np.percentile(ious, 5)), both_empty=both_empty,
                           online_empty_only=online_empty_only, dataset_empty_only=dataset_empty_only))
        print(f"  ep{ep:3d}  {e - s:3d} 帧  IoU 平均 {ious.mean():.3f}  最低 {ious.min():.3f}  "
              f"两边都空 {both_empty}  只有在线空 {online_empty_only}  只有数据集空 {dataset_empty_only}", flush=True)
    for n, (v, ep, t, frame, md, mo) in enumerate(w for w in worst if w[0] < 0.99):  # 全对上就不存图
        save_pair(out / f"worst{n}_ep{ep}_t{t}_iou{v:.2f}.jpg", frame, md, mo, f"ep{ep} t{t} IoU {v:.2f}")
    return per_ep, dict(mask_ms_median=float(np.median(t_mask) * 1000), mask_ms_p90=float(np.percentile(t_mask, 90) * 1000),
                        sam_ms_median=float(np.median(t_sam) * 1000), init_ms_median=float(np.median(t_init) * 1000))


def part_realtime(rs, policy, cfg, z, starts, ends, names, boxes, eps, device, out_dir):
    out = []
    for ep in eps:
        s, e = int(starts[ep]), int(ends[ep])
        T = e - s
        head = z["data/rgbm"][s:e]
        wrist = z["data/right_cam_img"][s:e]
        state = z["data/right_state"][s:e]
        bbox = np.array(boxes[names[ep]], dtype=np.float32)
        sam_init(rs, np.ascontiguousarray(head[0, ..., :3]), bbox)  # 抓取开始前框选、初始化，不计入节奏
        t_start = time.perf_counter()
        now_frame = lambda: min(T - 1, int((time.perf_counter() - t_start) * CAM_HZ))
        issue_t, issue_f, cycle_t, pred_t, mask_t, ious = [], [], [], [], [], []
        kinds = {"online_empty_only": 0, "dataset_empty_only": 0, "both_nonempty_lt_half": 0}
        worst = []

        def read_and_mask():
            i = now_frame()
            frame = np.ascontiguousarray(head[i, ..., :3])
            m, dt = get_mask_timed(rs, frame)
            mask_t.append(dt)
            md = head[i, ..., 3]
            if md.sum() or m.sum():
                v = iou(md > 0, m > 0)
                ious.append(v)
                if not m.sum():
                    kinds["online_empty_only"] += 1
                elif not md.sum():
                    kinds["dataset_empty_only"] += 1
                elif v < 0.5:
                    kinds["both_nonempty_lt_half"] += 1
                if v < 0.5 and len(worst) < 4 and all(abs(i - w[0]) > 15 for w in worst):
                    worst.append((i, v, frame.copy(), md.copy(), m.copy()))
            return i, frame, m

        while now_frame() < T - 1:
            c0 = time.perf_counter()
            # execute_grasping：get_state、get_mask、get_obs、process_obs、predict_action
            i, frame, m = read_and_mask()
            env_obs = {"right_cam_img": wrist[i:i + 1],
                       "rgbm": np.concatenate([frame, m[..., None]], axis=-1)[None],
                       "right_state": state[i:i + 1]}
            obs_np = RoboticsSystem.process_obs(None, env_obs, cfg.task.shape_meta)
            obs = dict_apply(obs_np, lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
            torch.cuda.synchronize(); p0 = time.perf_counter()
            with torch.no_grad():
                policy.predict_action(obs)
            torch.cuda.synchronize(); pred_t.append(time.perf_counter() - p0)
            # execute_action：6 步，每步前 get_state、get_mask、get_obs，然后下发目标
            for _ in range(EXEC_STEPS):
                i, _, _ = read_and_mask()
                issue_t.append(time.perf_counter() - t_start); issue_f.append(i)
            cycle_t.append(time.perf_counter() - c0)
        for i, v, frame, md, mo in worst:
            save_pair(out_dir / f"realtime_ep{ep}_t{i}_iou{v:.2f}.jpg", frame, md, mo, f"realtime ep{ep} t{i} IoU {v:.2f}")
        dur = issue_t[-1] - issue_t[0]
        gaps = np.diff(issue_t)
        r = dict(ep=ep, frames=T, video_s=T / CAM_HZ, cycles=len(cycle_t), actions=len(issue_t),
                 actions_per_s=float((len(issue_t) - 1) / dur),
                 cycle_ms_median=float(np.median(cycle_t) * 1000),
                 predict_ms_median=float(np.median(pred_t) * 1000),
                 mask_ms_median=float(np.median(mask_t) * 1000),
                 gap_within_6_ms_median=float(np.median(gaps[gaps < 0.2]) * 1000) if (gaps < 0.2).any() else None,
                 gap_between_cycles_ms_median=float(np.median(gaps[gaps >= 0.2]) * 1000) if (gaps >= 0.2).any() else None,
                 frames_seen=len(set(issue_f)), iou_mean=float(np.mean(ious)), iou_min=float(np.min(ious)),
                 iou_lt_half=int((np.array(ious) < 0.5).sum()), iou_n=len(ious), **kinds)
        out.append(r)
        print(f"  ep{ep:3d}  录像 {T / CAM_HZ:.1f} s  {len(cycle_t)} 轮预测  执行 {len(issue_t)} 步 = {r['actions_per_s']:.1f} 步/秒  "
              f"一轮 {r['cycle_ms_median']:.0f} ms（预测 {r['predict_ms_median']:.0f} + 掩码 7×{r['mask_ms_median']:.0f}）  "
              f"跳帧后 IoU 平均 {r['iou_mean']:.3f} 最低 {r['iou_min']:.3f}，低于 0.5 的 {r['iou_lt_half']}/{r['iou_n']} 帧"
              f"（在线空、数据集不空 {kinds['online_empty_only']}；数据集空、在线不空 {kinds['dataset_empty_only']}；"
              f"两边都不空但 IoU<0.5 {kinds['both_nonempty_lt_half']}）", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--zarr", required=True)
    ap.add_argument("--track-episodes", default="all", help="第 1 部分逐帧跟踪的条，逗号分隔或 all")
    ap.add_argument("--realtime-episodes", default="0,11,40", help="第 2 部分实时节奏的条")
    ap.add_argument("--name", default=None)
    a = ap.parse_args()

    device = torch.device("cuda:0")
    out = HERE / "data" / "outputs" / "realtime_sim" / (a.name or Path(a.ckpt).stem)
    out.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    policy, cfg, meta = offline_replay.load_policy(a.ckpt, device)
    rs = make_system(device, out)
    print(f"三样都已加载：显存 已分配 {torch.cuda.memory_allocated() / 2**20:.0f} MB")

    z = zarr.open(os.path.expanduser(a.zarr), "r")
    ends = z["meta/episode_ends"][:]
    starts = np.r_[0, ends[:-1]]
    attrs = dict(z.attrs)
    names, boxes = attrs["episodes"], attrs["head_bbox"]
    pick = lambda s: list(range(len(ends))) if s == "all" else [int(x) for x in s.split(",") if x != ""]

    summary = {"ckpt": a.ckpt, "ckpt_meta": meta, "zarr": a.zarr}
    if pick(a.track_episodes):
        print("\n== 1. 逐帧跟踪：在线掩码 vs 数据集掩码")
        per_ep, timing = part_tracking(rs, z, starts, ends, names, boxes, pick(a.track_episodes), out)
        summary["tracking"] = dict(timing, episodes=per_ep,
                                   iou_mean_all=float(np.mean([p["iou_mean"] for p in per_ep])),
                                   iou_min_all=float(min(p["iou_min"] for p in per_ep)))
        print(f"  每帧掩码中位 {timing['mask_ms_median']:.1f} ms（p90 {timing['mask_ms_p90']:.1f}）；"
              f"SAM 首帧 {timing['sam_ms_median']:.0f} ms；initialize_sam_cutie 整个 {timing['init_ms_median']:.0f} ms（含上游存图和 0.5 s 显示）")
    if pick(a.realtime_episodes):
        print("\n== 2. 实时节奏：照 execute_grasping 跑，按墙上时间取帧")
        summary["realtime"] = part_realtime(rs, policy, cfg, z, starts, ends, names, boxes, pick(a.realtime_episodes), device, out)
    summary["peak_vram_allocated_mb"] = torch.cuda.max_memory_allocated() / 2**20
    summary["peak_vram_reserved_mb"] = torch.cuda.max_memory_reserved() / 2**20
    print(f"\n显存峰值：已分配 {summary['peak_vram_allocated_mb']:.0f} MB，PyTorch 占用（reserved）{summary['peak_vram_reserved_mb']:.0f} MB")
    json.dump(summary, open(out / "summary.json", "w"), indent=1, ensure_ascii=False)
    print(f"输出：{out}")


if __name__ == "__main__":
    main()
