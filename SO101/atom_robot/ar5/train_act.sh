#!/usr/bin/env bash
# 本机训练 ACT。数据要先用 to_lerobot.py 转好。
#
#   bash atom_robot/ar5/train_act.sh                 # 从头训
#   bash atom_robot/ar5/train_act.sh --resume        # 断点续训
#
# loss 曲线看 wandb，项目名 vel-ar5。终端每 log_freq 步也打一行。
#
# 数据集用 DATASET= 覆盖。默认 vel_ar5_bolt_v2（297 条 / 104616 帧）。
#   1 epoch @ batch8 = 13077 步   （旧的 100 条那份是 4552 步）
#
# 本机 5070Ti 16GB 实测（2026-09-04，100 条 / 41444 帧）：
#   batch 8   83 smp/s   3.73 GB   <- 最快，就用这个
#   batch 32  76 smp/s  13.08 GB   反而更慢
#   batch 64  OOM
# 10.3 步/秒，所以 10 万步约 2.7 小时。
#
# 三个必须的参数，少一个就跑不起来（都已经写在下面了）：
#   --dataset.video_backend=pyav   torchcodec 装了但加载不了（缺 FFmpeg 共享库）
#   unset ALL_PROXY                socks5 代理会让 httpx 报 socksio 缺失
#   accelerate 已装                lerobot 训练必需，原来没装
set -euo pipefail

PY=${PY:-$HOME/miniforge3/envs/lerobot/bin}
DATASET=${DATASET:-vel_ar5_bolt_v2}

# 可以用环境变量覆盖：STEPS=1000000 bash train_act.sh
STEPS=${STEPS:-100000}
BATCH=${BATCH:-8}
# 存盘间隔。checkpoint **不会自动清理**，每个约 0.62 GB（51.6M 参数 + 优化器
# 状态），所以步数拉大时这个必须跟着放大，否则磁盘会在训练中途满掉。
#   100k 步 / 5000  = 20 个 = 12 GB
#   1M   步 / 25000 = 40 个 = 25 GB
SAVE_FREQ=${SAVE_FREQ:-$(( STEPS / 20 < 5000 ? 5000 : STEPS / 20 ))}
ROOT="$HOME/.cache/huggingface/lerobot/$DATASET"
JOB=act_${DATASET#vel_ar5_}_$(date +%m%d_%H%M)
OUT="$HOME/vel_runs/$JOB"

# 这台机器配了 socks5 的 ALL_PROXY，httpx 没装 socksio 会直接报错，
# 而 http_proxy 本身是通的 —— 所以只去掉 ALL_PROXY。
unset ALL_PROXY all_proxy

# wandb 初始化默认只等 90 秒，走代理时偶尔不够 —— 超时会让整个训练进程直接
# 退出（实测撞过一次）。放宽到 5 分钟。
export WANDB_INIT_TIMEOUT=${WANDB_INIT_TIMEOUT:-300}

# --- 断点续训：找最近一次 run ---
EXTRA=()
if [[ "${1:-}" == "--resume" ]]; then
  LAST=$(ls -td "$HOME"/vel_runs/act_bolt_* 2>/dev/null | head -1 || true)
  [[ -z "$LAST" ]] && { echo "没有找到可续的 run（$HOME/vel_runs/act_bolt_*）"; exit 1; }
  OUT="$LAST"
  EXTRA=(--resume=true --config_path="$LAST/checkpoints/last/pretrained_model/train_config.json")
  echo "续训 $LAST"
fi

[[ -d "$ROOT" ]] || { echo "数据集不在 $ROOT，先跑 to_lerobot.py"; exit 1; }

echo "数据集  $ROOT"
echo "输出    $OUT"
echo "wandb   项目 vel-ar5，run 名 $JOB"
echo "步数    $STEPS   batch $BATCH   每 $SAVE_FREQ 步存一次（约 $(( STEPS / SAVE_FREQ )) 个 ckpt，$(( STEPS / SAVE_FREQ * 62 / 100 )) GB）"
echo

"$PY/lerobot-train" \
    --dataset.repo_id="$DATASET" \
    --dataset.root="$ROOT" \
    --dataset.video_backend=pyav \
    --policy.type=act \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --policy.chunk_size=100 \
    --policy.n_action_steps=100 \
    --policy.optimizer_lr=1e-5 \
    --output_dir="$OUT" \
    --job_name="$JOB" \
    --steps="$STEPS" \
    --batch_size="$BATCH" \
    --num_workers=8 \
    --save_freq="$SAVE_FREQ" \
    --log_freq=100 \
    --seed=1000 \
    --wandb.enable=true \
    --wandb.project=vel-ar5 \
    "${EXTRA[@]}"
