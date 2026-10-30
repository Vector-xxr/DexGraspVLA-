"""C1 离线回放（待测试.md C1）：把采集好的 zarr 逐帧喂给训练好的控制器，对比预测动作和数据里的动作。

只读数据和检查点，不碰任何硬件。尽量直接用上游 inference.py 的代码：
- 观测处理：RoboticsSystem.process_obs（原样调用，不实例化 RoboticsSystem）
- 预测：policy.predict_action，torch.no_grad、fp32（inference.py 也没开 autocast）
- 关节还原：RoboticsSystem.scale，q = (x+1)/2·(hi−lo)+lo
- 部署时每预测一次只执行前 6 步（execute_action 里的 range(6)），所以重点看前 6 步
和上游不同的只有两处：
- 模型用检查点里存的配置建（inference.py 读仓库里的 yaml，那份是 13 维的），并清掉其中的 start_ckpt_path
  （H100 训练时填的是容器里的 /models/... 路径，本机没有）。官方检查点存的配置指向作者内部包名，
  建不出来，这时照 inference.py 读仓库 yaml
- 关节限位优先用 zarr attrs 里的 joint_min/joint_max（我们的数据：AR5 软限位）；
  官方数据没有这两项，就用 inference_utils/config.yaml 里上游那台臂的限位

对照组「原地不动」：64 步都等于当前状态。模型的误差要明显小于它，才说明学到了动作。
所有条都参与过训练，这里看的是「推理链路对不对、训练集上学得怎么样」，不是泛化。

用法（在仓库根目录）：
  PYTHONPATH= ~/miniforge3/envs/dexgraspvla/bin/python offline_replay.py \\
      --ckpt weights/epoch_0124-train_loss_0.001.ckpt \\
      --zarr ~/H100_dexgraspvla/Datasets/atom_black_sleeve_20261021/black_sleeve.zarr
输出在 data/outputs/offline_replay/<--name>/：summary.json、preds.npz、几条的曲线图。
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault("TORCH_HOME", str(HERE / "weights" / "torch_home"))
sys.path.insert(0, str(HERE))

import dill
import hydra
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "DejaVu Sans"]
import numpy as np
import torch
import yaml
import zarr
from omegaconf import open_dict

from controller.common.pytorch_util import dict_apply
from inference import RoboticsSystem  # 只用它的两个不碰硬件的方法
from inference_utils.utils import load_config

EXEC_STEPS = 6  # inference.py execute_action：每次预测执行前 6 步


def load_policy(ckpt_path, device):
    payload = torch.load(open(ckpt_path, "rb"), pickle_module=dill, map_location="cpu", weights_only=False)
    cfg = payload["cfg"]
    if not cfg._target_.startswith("controller."):
        # 官方检查点存的配置用的是作者内部包名 diffusion_policy.*（开源时改名成 controller.*），
        # 这时照 inference.py 的 init_controller 读仓库里的 yaml（那份就是官方的 13 维配置）
        cfg = load_config(main_config_path=HERE / "controller" / "config" / "train_dexgraspvla_controller_workspace.yaml",
                          task_config_path=HERE / "controller" / "config" / "task" / "grasp.yaml")
    with open_dict(cfg):
        cfg.policy.start_ckpt_path = None
    workspace = hydra.utils.get_class(cfg._target_)(cfg)
    workspace.load_payload(payload)
    policy = workspace.model.eval().to(device)
    meta = {k: dill.loads(payload["pickles"][k]) for k in ("epoch", "global_step") if k in payload["pickles"]}
    return policy, cfg, meta


def arm_limits(z):
    attrs = dict(z.attrs)
    if "joint_min" in attrs:
        return np.array(attrs["joint_min"]), np.array(attrs["joint_max"]), "zarr attrs（AR5 软限位）"
    lim = yaml.safe_load(open(HERE / "inference_utils" / "config.yaml"))["robot"]["dof_limits"]
    return np.array(lim["lower"]), np.array(lim["upper"]), "inference_utils/config.yaml（上游的臂）"


def roundtrip_check(z, lo, hi, starts, ends, eps):
    """action[t] = 下一帧 state；还原成弧度后应该等于原始 npz 里的关节角。只有我们的数据有原始 npz。"""
    attrs = dict(z.attrs)
    if "src" not in attrs or not Path(attrs["src"]).is_dir():
        return None
    act = z["data/action"]
    worst = 0.0
    for ep in eps:
        s, e = int(starts[ep]), int(ends[ep])
        raw = np.load(Path(attrs["src"]) / f"{attrs['episodes'][ep]}.npz")["state"][:, :7]
        q = RoboticsSystem.scale(None, act[s:e - 1, :7], lo, hi)
        worst = max(worst, float(np.abs(q - raw[1:e - s]).max()))
    return worst


def plot_episode(path, ep, T, ts, preds, gt_full, lo, hi, n_arm):
    """黑线：数据里的动作；红线：每个回放时刻预测的前 6 步；淡红：每 4 个时刻画一次完整 64 步。"""
    D = gt_full.shape[1]
    to_deg = lambda x, j: np.degrees(RoboticsSystem.scale(None, x, lo[j:j + 1], hi[j:j + 1])) if j < n_arm else x
    fig, axes = plt.subplots(D, 1, figsize=(12, 1.6 * D), sharex=True)
    for j in range(D):
        ax = axes[j]
        ax.plot(np.arange(T), to_deg(gt_full[:, j], j), "k-", lw=1)
        for n, (t, p) in enumerate(zip(ts, preds)):
            k = np.arange(t, t + p.shape[0])
            if n % 4 == 0:
                ax.plot(k, to_deg(p[:, j], j), "-", color=(1, 0.6, 0.6), lw=0.6)
            ax.plot(k[:EXEC_STEPS], to_deg(p[:EXEC_STEPS, j], j), "r-", lw=1.2)
        ax.set_ylabel(f"j{j + 1} °" if j < n_arm else f"末端{j - n_arm}", fontsize=8)
        ax.tick_params(labelsize=7)
    axes[-1].set_xlabel("帧")
    axes[0].set_title(f"episode {ep}：黑=数据里的动作  红=预测的前 {EXEC_STEPS} 步  淡红=完整 64 步", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--zarr", required=True)
    ap.add_argument("--episodes", default="all", help="逗号分隔的编号，或 all")
    ap.add_argument("--stride", type=int, default=10, help="每条隔多少帧回放一次")
    ap.add_argument("--plot-episodes", default="0", help="画曲线图的条，逗号分隔")
    ap.add_argument("--name", default=None, help="输出目录名，默认用检查点文件名")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    device = torch.device("cuda:0")
    out = HERE / "data" / "outputs" / "offline_replay" / (a.name or Path(a.ckpt).stem)
    out.mkdir(parents=True, exist_ok=True)

    policy, cfg, meta = load_policy(a.ckpt, device)
    H, D = cfg.n_action_steps, cfg.task.shape_meta.action.shape[0]
    print(f"检查点 {a.ckpt}：{meta}，动作 {D} 维 × {H} 步")

    z = zarr.open(os.path.expanduser(a.zarr), "r")
    ends = z["meta/episode_ends"][:]
    starts = np.r_[0, ends[:-1]]
    eps = list(range(len(ends))) if a.episodes == "all" else [int(x) for x in a.episodes.split(",")]
    plot_eps = {int(x) for x in a.plot_episodes.split(",") if x != ""}
    lo, hi, lim_src = arm_limits(z)
    n_arm = len(lo)
    half = (hi - lo) / 2  # 归一化 1 = 多少弧度
    print(f"数据 {a.zarr}：{len(ends)} 条 / {ends[-1]} 帧，回放 {len(eps)} 条、每 {a.stride} 帧一次；限位来自 {lim_src}")

    rt = roundtrip_check(z, lo, hi, starts, ends, eps)
    if rt is not None:
        print(f"限位还原核对：动作还原成弧度后和原始 npz 关节角最大差 {rt:.2e} rad")

    state_all = z["data/right_state"][:]
    act_all = z["data/action"][:]
    torch.manual_seed(a.seed)
    torch.cuda.reset_peak_memory_stats()

    rec = {"ep": [], "t": [], "pred": [], "gt": [], "hold": []}
    times = []
    for ep in eps:
        s, e = int(starts[ep]), int(ends[ep])
        T = e - s
        ts = list(range(0, T, a.stride))
        ep_preds = []
        for t in ts:
            i = s + t
            env_obs = {"right_cam_img": z["data/right_cam_img"][i:i + 1],
                       "rgbm": z["data/rgbm"][i:i + 1],
                       "right_state": state_all[i:i + 1]}
            obs_np = RoboticsSystem.process_obs(None, env_obs, cfg.task.shape_meta)
            obs = dict_apply(obs_np, lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                pred = policy.predict_action(obs)[0].float().cpu().numpy()
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
            idx = s + np.minimum(np.arange(t, t + H), T - 1)  # 和训练时一样：越过末尾的用最后一帧补
            rec["ep"].append(ep); rec["t"].append(t); rec["pred"].append(pred)
            rec["gt"].append(act_all[idx]); rec["hold"].append(np.repeat(state_all[i:i + 1], H, 0))
            ep_preds.append(pred)
        if ep in plot_eps:
            plot_episode(out / f"episode_{ep:04d}.png", ep, T, ts, ep_preds, act_all[s:e], lo, hi, n_arm)
        print(f"  ep{ep:3d}  {len(ts):3d} 次  累计 {len(times)}  单次中位 {np.median(times) * 1000:.0f} ms", flush=True)

    pred, gt, hold = (np.stack(rec[k]) for k in ("pred", "gt", "hold"))  # (N, H, D)
    np.savez_compressed(out / "preds.npz", ep=np.array(rec["ep"]), t=np.array(rec["t"]), pred=pred, gt=gt, hold=hold,
                        joint_min=lo, joint_max=hi)

    def err(x):  # 臂：度；末端：原值
        d = np.abs(x - gt)
        d[..., :n_arm] = np.degrees(d[..., :n_arm] * half)
        return d
    em, eh = err(pred), err(hold)
    dims = [f"j{j + 1}(°)" for j in range(n_arm)] + [f"末端{j}" for j in range(D - n_arm)]
    summary = {
        "ckpt": a.ckpt, "ckpt_meta": meta, "zarr": a.zarr, "episodes": len(eps), "predictions": len(times),
        "stride": a.stride, "limits_from": lim_src, "roundtrip_max_rad": rt,
        "predict_ms_median": float(np.median(times) * 1000), "predict_ms_p90": float(np.percentile(times, 90) * 1000),
        "peak_vram_mb": torch.cuda.max_memory_allocated() / 2**20,
        "dims": dims,
        "mae_first6_model": em[:, :EXEC_STEPS].mean((0, 1)).tolist(),
        "mae_first6_hold": eh[:, :EXEC_STEPS].mean((0, 1)).tolist(),
        "mae_all64_model": em.mean((0, 1)).tolist(),
        "mae_all64_hold": eh.mean((0, 1)).tolist(),
        "mae_by_step_model_arm": em[..., :n_arm].mean((0, 2)).tolist(),
        "mae_by_step_hold_arm": eh[..., :n_arm].mean((0, 2)).tolist(),
    }
    json.dump(summary, open(out / "summary.json", "w"), indent=1, ensure_ascii=False)

    print(f"\n回放 {len(times)} 次；预测一次中位 {summary['predict_ms_median']:.0f} ms、p90 {summary['predict_ms_p90']:.0f} ms；"
          f"显存峰值 {summary['peak_vram_mb']:.0f} MB")
    print(f"{'':10s}{'前6步 模型':>12s}{'前6步 不动':>12s}{'64步 模型':>12s}{'64步 不动':>12s}")
    for j, name in enumerate(dims):
        print(f"{name:10s}{summary['mae_first6_model'][j]:12.3f}{summary['mae_first6_hold'][j]:12.3f}"
              f"{summary['mae_all64_model'][j]:12.3f}{summary['mae_all64_hold'][j]:12.3f}")
    print(f"输出：{out}")


if __name__ == "__main__":
    main()
