#!/usr/bin/env bash
# DexGraspVLA controller 单卡训练（RTX 5070 Ti 16GB）
#
# 上游 train.sh 是 8 卡 accelerate launch + batch 48，这张卡上必 OOM。
# 这里不改上游文件，用 hydra 命令行覆盖：batch 12 × 梯度累积 4 = 等效 48。
# 实测（2026-10-08，真实 train.py）：显存峰值约 13.7 GB（几次实测在 13.5~13.7 之间），16 GB 的卡余量约 2 GB。
# bf16 写死在 workspace 代码里，单卡不需要 accelerate config。
#
# 用法（不用先 conda activate；脚本会先切到仓库根目录，相对路径按根目录算，稳妥起见写绝对路径）：
#   bash train_5070ti.sh                                # 默认数据 data/grasp_demo_example
#   bash train_5070ti.sh training.debug=true            # 冒烟：2 epoch × 3 步
#   bash train_5070ti.sh "policy.start_ckpt_path='/abs/xx.ckpt'"   # 从同维度检查点接着训（严格加载）
#   bash train_5070ti.sh "policy.start_ckpt_path='/abs/官方.ckpt'" +training.start_ckpt_skip_mismatch=true ...
#       ↑ 维度不同（8 维 vs 官方 13 维）：形状对得上的全加载，两端 4 个张量随机初始化
#   bash train_5070ti.sh "task.dataset_paths=['/abs/a.zarr','/abs/b.zarr']"
#       ↑ 路径外面要有一层单引号：项目路径里有中文（「算法」），不加 hydra 会报 LexerNoViableAltException
#   bash train_5070ti.sh ... 'task.shape_meta.obs.right_state.shape=[8]' 'task.shape_meta.action.shape=[8]'
#       ↑ 7 臂 + 1 夹爪（HKVR）的数据；obs_encoder.py 的 state 输入维已改成跟 shape_meta 走
# 输出：data/outputs/<日期>/<时间>_.../（checkpoints、normalizer.pkl、logs.json.txt）
set -euo pipefail

HERE="$(dirname "$(readlink -f "$0")")"
# DINOv2 的代码和权重用项目里的 weights/torch_home（普通文件，拷走就能断网训练），不去 ~/.cache/torch
export TORCH_HOME="${TORCH_HOME:-$HERE/weights/torch_home}"
cd "$HERE"

# wandb 的离线记录放工作区外：它会在当前目录建 wandb/latest-run 之类的软链接，
# 留在项目里会让 git、复制、拷进移动硬盘出问题。检查点和训练输出照旧在 data/outputs/。
export WANDB_DIR="${WANDB_DIR:-$HOME/wandb_dexgraspvla}"
mkdir -p "$WANDB_DIR"

# ROS 的 py3.12 PYTHONPATH 会污染这个 3.11 环境（见 环境说明.md 第 3 节）
PYTHONPATH= exec "$HOME/miniforge3/envs/dexgraspvla/bin/python" train.py \
    --config-name train_dexgraspvla_controller_workspace \
    dataloader.batch_size=12 \
    val_dataloader.batch_size=12 \
    training.gradient_accumulate_every=4 \
    "$@"
