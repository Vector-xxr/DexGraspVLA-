"""把送进 DiT 的条件采出来看：头部 RGB + 掩码（融成 1369 个头部 token）、腕部 RGB（1369 个 token）、状态（1 个）、扩散步数（1 个）。

只读数据和检查点，不碰硬件。观测处理照 inference.py（process_obs），编码器内部的几步照 obs_encoder.py
的 forward_head / forward_wrist 原样拆开算；注意力用上游 predict_action 自带的 gen_attn_map
（attention_map_visualizer.py 读的就是同一份数据），只是把 16 个扩散步 × 12 层 × 8 个头平均成一张。

每个时刻一张图：
  第一行  头部 RGB（模型收到的 518×518）| 掩码 | 头部 DINOv2 特征（PCA 成 3 维当颜色）| 掩码分支输出（每块的范数）
  第二行  腕部 RGB | 腕部 DINOv2 特征 | DiT 注意力落在头部哪里 | 落在腕部哪里
标题：状态、注意力在头部 / 腕部 / 状态 / 扩散步数四组之间的占比。

用法（在仓库根目录）：
  PYTHONPATH= ~/miniforge3/envs/dexgraspvla/bin/python dit_inputs_vis.py \\
      --ckpt weights/epoch_0124-train_loss_0.001.ckpt \\
      --h5 logs/ar5_black_sleeve/20261022_160646/data.h5 --steps 0,270,540,815
  也可以用训练数据：--zarr <…>.zarr --steps 0,100,200（全局帧号）
输出在 data/outputs/dit_inputs/<--name>/。
"""
import argparse
import os
import pickle
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import offline_replay  # noqa: E402  设好 TORCH_HOME、matplotlib 用 Agg、load_policy

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from controller.common.pytorch_util import dict_apply
from inference import RoboticsSystem

N_PATCH = 37  # 518 / 14


def load_frames(a):
    if a.h5:
        import h5py
        f = h5py.File(a.h5, "r")
        return [(i, f["rgbm"][i], f["right_cam_img"][i], f["state"][i]) for i in a.steps]
    import zarr
    z = zarr.open(os.path.expanduser(a.zarr), "r")
    return [(i, z["data/rgbm"][i], z["data/right_cam_img"][i], z["data/right_state"][i]) for i in a.steps]


def pca_rgb(feats):
    """feats: list of (1369, D)。一起拟合 3 个主成分，每张各自映射成 37×37×3 的颜色。"""
    X = np.concatenate(feats, 0)
    mu = X.mean(0)
    _, _, vt = np.linalg.svd(X - mu, full_matrices=False)
    out = []
    P = np.concatenate([(f - mu) @ vt[:3].T for f in feats], 0)
    lo, hi = np.percentile(P, 1, 0), np.percentile(P, 99, 0)
    for f in feats:
        p = ((f - mu) @ vt[:3].T - lo) / (hi - lo + 1e-8)
        out.append(np.clip(p, 0, 1).reshape(N_PATCH, N_PATCH, 3))
    return out


def overlay(img, heat):
    h = F.interpolate(torch.from_numpy(heat)[None, None].float(), size=img.shape[:2], mode="bilinear")[0, 0].numpy()
    h = (h - h.min()) / (h.max() - h.min() + 1e-12)
    cm = plt.get_cmap("jet")(h)[..., :3]
    return np.clip(0.45 * img + 0.55 * cm, 0, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--h5", default=None, help="deploy_ar5.py --save_deployment_data 存的 data.h5")
    ap.add_argument("--zarr", default=None)
    ap.add_argument("--steps", required=True)
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    a.steps = [int(s) for s in a.steps.split(",")]
    device = torch.device("cuda:0")
    out = HERE / "data" / "outputs" / "dit_inputs" / (a.name or Path(a.h5 or a.zarr).parent.name)
    out.mkdir(parents=True, exist_ok=True)

    policy, cfg, _ = offline_replay.load_policy(a.ckpt, device)
    enc = policy.obs_encoder
    frames = load_frames(a)
    lo = np.array([-3.1067, -2.0944, -3.1067, -1.0472, -3.1067, -0.8727, -0.8727])
    hi = np.array([3.1067, 2.0944, 3.1067, 2.5307, 3.1067, 0.8727, 0.8727])

    rec = []
    for step, rgbm, wrist, state in frames:
        env_obs = {"right_cam_img": wrist[None], "rgbm": rgbm[None], "right_state": state[None].astype(np.float32)}
        obs_np = RoboticsSystem.process_obs(None, env_obs, cfg.task.shape_meta)
        obs = dict_apply(obs_np, lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
        with torch.no_grad():
            # 照 obs_encoder.forward_head / forward_wrist（training=False：不做颜色扰动）
            r = obs["rgbm"]
            head_rgb = r[:, :, :3][:, :, [2, 1, 0]].reshape(1, 3, 518, 518)   # 编码器里把 BGR 翻成 RGB
            mask_in = r[:, :, 3:].reshape(1, 1, 518, 518)
            head_dino = enc.dino_head.get_intermediate_layers(enc.dino_transform(head_rgb), n=1)[0][0]
            mask_feat = enc.mask_process_net["transformer"](enc.mask_process_net["patch_embed"](mask_in))[0]
            wrist_rgb = obs["right_cam_img"][:, :, [2, 1, 0]].reshape(1, 3, 518, 518)
            wrist_dino = enc.dino_wrist.get_intermediate_layers(enc.dino_transform(wrist_rgb), n=1)[0][0]
            # 上游 predict_action 自带的注意力图（存成 pkl，再读回来）
            pkl = tempfile.mktemp(suffix=".pkl")
            torch.manual_seed(0)
            policy.predict_action(obs, pkl)
            maps = pickle.load(open(pkl, "rb"))["attention_maps"]
            os.remove(pkl)
        att = np.mean([np.mean([b[0].float().mean(0).cpu().numpy() if torch.is_tensor(b) else np.asarray(b[0]).mean(0)
                                for b in blocks], 0) for blocks in maps.values()], 0)           # (2740,)
        rec.append(dict(step=step, head_rgb=head_rgb[0].permute(1, 2, 0).cpu().numpy(),
                        mask=mask_in[0, 0].cpu().numpy(), wrist_rgb=wrist_rgb[0].permute(1, 2, 0).cpu().numpy(),
                        head_dino=head_dino.cpu().numpy(), wrist_dino=wrist_dino.cpu().numpy(),
                        mask_norm=mask_feat.norm(dim=-1).cpu().numpy().reshape(N_PATCH, N_PATCH),
                        att=att, state=state))
        print(f"第 {step} 步：注意力 头部 {att[:1369].sum():.1%}  腕部 {att[1369:2738].sum():.1%}  "
              f"状态 {att[2738]:.1%}  扩散步数 {att[2739]:.1%}", flush=True)

    head_pca = pca_rgb([r["head_dino"] for r in rec])
    wrist_pca = pca_rgb([r["wrist_dino"] for r in rec])
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "DejaVu Sans"]
    for r, hp, wp in zip(rec, head_pca, wrist_pca):
        fig, ax = plt.subplots(2, 4, figsize=(16, 8.6))
        q = np.degrees(r["state"][:7] * (hi - lo) / 2 + (hi + lo) / 2)
        att = r["att"]
        panels = [
            (r["head_rgb"], "头部 RGB（模型收到的 518×518）"), (r["mask"], "掩码（0/1）"),
            (hp, "头部 DINOv2 特征（PCA 3 维）"), (r["mask_norm"], "掩码分支输出（每块范数）"),
            (r["wrist_rgb"], "腕部 RGB"), (wp, "腕部 DINOv2 特征（PCA 3 维）"),
            (overlay(r["head_rgb"], att[:1369].reshape(N_PATCH, N_PATCH)), f"DiT 注意力：头部（占 {att[:1369].sum():.0%}）"),
            (overlay(r["wrist_rgb"], att[1369:2738].reshape(N_PATCH, N_PATCH)), f"DiT 注意力：腕部（占 {att[1369:2738].sum():.0%}）"),
        ]
        for axi, (img, title) in zip(ax.flat, panels):
            axi.imshow(img, cmap="gray" if img.ndim == 2 else None, interpolation="nearest")
            axi.set_title(title, fontsize=10)
            axi.axis("off")
        fig.suptitle(f"第 {r['step']} 步   关节(°) {np.round(q, 1)}   夹爪 {r['state'][7]:.2f}   "
                     f"注意力：头部 {att[:1369].sum():.0%} / 腕部 {att[1369:2738].sum():.0%} / "
                     f"状态 {att[2738]:.1%} / 扩散步数 {att[2739]:.1%}", fontsize=11)
        fig.tight_layout()
        fig.savefig(out / f"step_{r['step']:04d}.png", dpi=85)
        plt.close(fig)
    print(f"输出：{out}")


if __name__ == "__main__":
    main()
