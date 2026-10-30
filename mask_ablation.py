"""掩码消融：同一帧、同一个随机种子，只换头部掩码，看模型预测变多少——掩码丢了 / 标错了对模型有没有影响。

只读数据和检查点，不碰硬件。加载、观测处理、预测和 offline_replay.py 完全一样，只改 rgbm 第 4 通道。
每个回放点的几种掩码用同一个种子（扩散采样的初始噪声相同），所以预测的差别只来自掩码。

A 组：数据集里掩码不空的帧
  真掩码 / 清空（模拟遮挡后丢失） / 挪位（同形状平移到白板上没有东西的地方，模拟标错目标）
B 组：数据集里掩码为空的帧
  保持空（训练时就是这样） / 补上这条之前最后一帧非空的掩码（README「已知问题」第 1 条的备选办法 1）

所有条都参与过训练：看的是「模型用不用掩码」，不是泛化。只有一个物体时，模型不看掩码也能找到它，
所以这里掩码影响小，不代表多物体时也小。

用法（在仓库根目录）：
  PYTHONPATH= ~/miniforge3/envs/dexgraspvla/bin/python mask_ablation.py \\
      --ckpt weights/epoch_0124-train_loss_0.001.ckpt \\
      --zarr ~/H100_dexgraspvla/Datasets/atom_black_sleeve_20261021/black_sleeve.zarr
输出在 data/outputs/mask_ablation/<--name>/：summary.json、preds.npz、几张挪位掩码的示意图。
"""
import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import offline_replay  # noqa: E402  先 import：设好 TORCH_HOME、matplotlib 用 Agg

import cv2
import numpy as np
import torch
import zarr

from controller.common.pytorch_util import dict_apply
from inference import RoboticsSystem

MOVING = [0, 1, 3, 5, 6]  # j3、j5 数据里全程不动，单独看没意义
SHIFT = (-150, 0)        # 挪位：往左 150 像素，落在白板左半边空白处（套筒都在白板中间偏右）


def shifted(mask, dx, dy):
    out = np.zeros_like(mask)
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    ys, xs = ys + dy, xs + dx
    keep = (ys >= 0) & (ys < h) & (xs >= 0) & (xs < w)
    out[ys[keep], xs[keep]] = 1
    return out


def predict(policy, cfg, head_bgr, mask, wrist, state, seed, device):
    env_obs = {"right_cam_img": wrist[None], "rgbm": np.concatenate([head_bgr, mask[..., None]], -1)[None],
               "right_state": state[None]}
    obs_np = RoboticsSystem.process_obs(None, env_obs, cfg.task.shape_meta)
    obs = dict_apply(obs_np, lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
    torch.manual_seed(seed)
    with torch.no_grad():
        return policy.predict_action(obs)[0].float().cpu().numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--zarr", required=True)
    ap.add_argument("--stride", type=int, default=20)
    ap.add_argument("--name", default=None)
    a = ap.parse_args()

    device = torch.device("cuda:0")
    out = HERE / "data" / "outputs" / "mask_ablation" / (a.name or Path(a.ckpt).stem)
    out.mkdir(parents=True, exist_ok=True)
    policy, cfg, meta = offline_replay.load_policy(a.ckpt, device)
    H = cfg.n_action_steps
    z = zarr.open(os.path.expanduser(a.zarr), "r")
    ends = z["meta/episode_ends"][:]
    starts = np.r_[0, ends[:-1]]
    lo, hi, _ = offline_replay.arm_limits(z)
    half = (hi - lo) / 2
    state_all, act_all = z["data/right_state"][:], z["data/action"][:]

    rec = {k: [] for k in ("group", "ep", "t", "gt", "real", "zero", "shift", "hold")}
    n_demo = 0
    q = 0
    for ep in range(len(ends)):
        s, e = int(starts[ep]), int(ends[ep])
        T = e - s
        head = z["data/rgbm"][s:e]
        wrist = z["data/right_cam_img"][s:e]
        nonempty = head[..., 3].reshape(T, -1).any(1)
        for t in range(0, T, a.stride):
            q += 1
            seed = 1000 + q
            frame, m = np.ascontiguousarray(head[t, ..., :3]), head[t, ..., 3]
            idx = s + np.minimum(np.arange(t, t + H), T - 1)
            gt = act_all[idx]
            args = (wrist[t], state_all[s + t], seed, device)
            real = predict(policy, cfg, frame, m, *args)
            if nonempty[t]:
                ms = shifted(m, *SHIFT)
                rec["group"].append("A"); rec["zero"].append(predict(policy, cfg, frame, np.zeros_like(m), *args))
                rec["shift"].append(predict(policy, cfg, frame, ms, *args)); rec["hold"].append(real * np.nan)
                if n_demo < 3 and t > 0:
                    img = frame.copy()
                    img[m > 0] = (0.4 * img[m > 0] + [150, 0, 0]).astype(np.uint8)
                    img[ms > 0] = (0.4 * img[ms > 0] + [0, 0, 150]).astype(np.uint8)
                    cv2.imwrite(str(out / f"shift_ep{ep}_t{t}.jpg"), img)
                    n_demo += 1
            else:
                prev = np.flatnonzero(nonempty[:t])
                if len(prev) == 0:
                    continue
                rec["group"].append("B"); rec["zero"].append(real * np.nan); rec["shift"].append(real * np.nan)
                rec["hold"].append(predict(policy, cfg, frame, head[prev[-1], ..., 3], *args))
            rec["ep"].append(ep); rec["t"].append(t); rec["gt"].append(gt); rec["real"].append(real)
        print(f"  ep{ep:3d}  累计 A {rec['group'].count('A')}  B {rec['group'].count('B')}", flush=True)

    R = {k: (np.array(v) if k in ("group",) else np.stack(v) if k not in ("ep", "t") else np.array(v)) for k, v in rec.items()}
    np.savez_compressed(out / "preds.npz", **R, joint_min=lo, joint_max=hi)

    def arm_err(p, sel, k):  # 会动的 5 个关节平均绝对误差（度）
        return float(np.degrees(np.abs(p[sel, :k, :7] - R["gt"][sel, :k, :7]) * half)[..., MOVING].mean())

    def grip_err(p, sel, k):
        return float(np.abs(p[sel, :k, 7] - R["gt"][sel, :k, 7]).mean())

    def change(p1, p2, sel, k):  # 两种掩码下预测本身差多少（不看真值）
        arm = float(np.degrees(np.abs(p1[sel, :k, :7] - p2[sel, :k, :7]) * half)[..., MOVING].mean())
        return arm, float(np.abs(p1[sel, :k, 7] - p2[sel, :k, 7]).mean())

    A, B = R["group"] == "A", R["group"] == "B"
    summ = {"ckpt": a.ckpt, "ckpt_meta": meta, "stride": a.stride, "shift_px": SHIFT, "n_A": int(A.sum()), "n_B": int(B.sum())}
    lines = []
    for k in (6, H):
        tag = f"前{k}步" if k == 6 else f"{k}步"
        for name, key, sel in (("A 真掩码", "real", A), ("A 清空", "zero", A), ("A 挪位", "shift", A),
                               ("B 保持空", "real", B), ("B 补上最后的掩码", "hold", B)):
            summ[f"{key}_{name[0]}_{k}"] = dict(arm=arm_err(R[key], sel, k), grip=grip_err(R[key], sel, k))
            lines.append(f"{tag:6s}{name:14s} 关节误差 {arm_err(R[key], sel, k):.3f}°  夹爪误差 {grip_err(R[key], sel, k):.4f}")
        for name, k1, k2, sel in (("A 清空 vs 真掩码", "zero", "real", A), ("A 挪位 vs 真掩码", "shift", "real", A),
                                  ("B 补上 vs 保持空", "hold", "real", B)):
            c = change(R[k1], R[k2], sel, k)
            summ[f"change_{k1}_{k2}_{k}"] = dict(arm=c[0], grip=c[1])
            lines.append(f"{tag:6s}{name:14s} 预测本身变了 关节 {c[0]:.3f}°  夹爪 {c[1]:.4f}")
    # 参照：同一个掩码、换个种子，预测本身会变多少（扩散采样的随机性）
    sel = np.flatnonzero(A)[:150]
    noise = []
    for n in sel:
        ep, t = int(R["ep"][n]), int(R["t"][n])
        s = int(starts[ep])
        f = z["data/rgbm"][s + t]
        p2 = predict(policy, cfg, np.ascontiguousarray(f[..., :3]), f[..., 3], z["data/right_cam_img"][s + t],
                     state_all[s + t], 999_000 + n, device)
        noise.append(p2)
    noise = np.stack(noise)
    for k in (6, H):
        arm = float(np.degrees(np.abs(noise[:, :k, :7] - R["real"][sel, :k, :7]) * half)[..., MOVING].mean())
        grip = float(np.abs(noise[:, :k, 7] - R["real"][sel, :k, 7]).mean())
        summ[f"change_seed_only_{k}"] = dict(arm=arm, grip=grip)
        lines.append(f"{'前6步' if k == 6 else f'{k}步':6s}{'参照：只换种子':14s} 预测本身变了 关节 {arm:.3f}°  夹爪 {grip:.4f}")
    json.dump(summ, open(out / "summary.json", "w"), indent=1, ensure_ascii=False)
    print(f"\nA 组（掩码不空）{A.sum()} 个回放点，B 组（掩码为空）{B.sum()} 个；会动的 5 个关节平均绝对误差")
    print("\n".join(lines))
    print(f"输出：{out}")


if __name__ == "__main__":
    main()
