#!/usr/bin/env bash
# 本机冒烟一个 H100 数据集：和 H100 同一个 Docker 镜像、同样只读挂载、断网，
# 8 维从官方 ckpt 部分加载，training.debug=true（2 epoch × 3 步，含采样、attention map、存 ckpt）。
#
# 用法：
#   ./smoke_h100_dataset.sh <数据集名>            # Datasets/<数据集名>/ 下所有 *.zarr 一起训
#
# 成功判据：exit 0、打印 4 行 [skip_mismatch] random init、checkpoints/ 里有 ckpt。
# debug 的 ckpt 没用（每个 2.2 GB），确认落盘后删掉，日志留在 smoke_out_local/<数据集名>/。
# 5070 Ti 16 GB 所以 batch 12；H100 上用默认 48。
set -euo pipefail

[ $# -eq 1 ] || { sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }
NAME=$1
R=${H100_ROOT:-$HOME/H100_dexgraspvla}
IMG=dexgraspvla:cu128
CK=/models/dexgraspvla-controller-20250320/dexgraspvla-controller-20250320.ckpt
[ -d "$R/Datasets/$NAME" ] || { echo "  ✗ 没有 $R/Datasets/$NAME"; exit 2; }
ZS=$(cd "$R/Datasets/$NAME" && ls -d *.zarr 2>/dev/null) || { echo "  ✗ $NAME 下没有 .zarr"; exit 2; }
PATHS=$(printf "'/data/Datasets/$NAME/%s'," $ZS); PATHS="[${PATHS%,}]"

OUT=$R/smoke_out_local/$NAME
[ ! -e "$OUT" ] || { echo "  ✗ $OUT 已存在（上次冒烟的输出），先删掉"; exit 2; }
mkdir -p "$OUT"
echo "  数据：$PATHS"
echo "  输出：$OUT"

set +e
docker run --rm --gpus all --network none --ipc=host \
    -v "$R/Datasets:/data/Datasets:ro" -v "$R/Models:/models:ro" \
    -v "$R/DexGraspVLA:/workspace/DexGraspVLA:ro" -v "$R/smoke_out_local:/data/outputs" \
    -e WANDB_DIR=/data/outputs/$NAME -w /workspace/DexGraspVLA \
    "$IMG" \
    accelerate launch --num_processes=1 train.py \
        --config-name train_dexgraspvla_controller_workspace \
        training.debug=true dataloader.batch_size=12 val_dataloader.batch_size=12 \
        "task.dataset_paths=$PATHS" \
        'task.shape_meta.obs.right_state.shape=[8]' 'task.shape_meta.action.shape=[8]' \
        "policy.start_ckpt_path='$CK'" +training.start_ckpt_skip_mismatch=true \
        hydra.run.dir=/data/outputs/$NAME 2>&1 | tee "$OUT.log"
rc=${PIPESTATUS[0]}
set -e

echo
grep -h "skip_mismatch" "$OUT.log" || true
# 容器以 root 写的，删 ckpt 也得在容器里删
CKPTS=$(cd "$OUT" && find . -name '*.ckpt' 2>/dev/null | sort)
echo "  exit $rc   ckpt: ${CKPTS:-无}"
if [ "$rc" -eq 0 ] && [ -n "$CKPTS" ]; then
    docker run --rm -v "$R/smoke_out_local:/data/outputs" "$IMG" \
        sh -c "rm -rf /data/outputs/$NAME/checkpoints /data/outputs/$NAME/epoch_checkpoints && chmod -R a+rwX /data/outputs/$NAME"
    echo "  ✓ 冒烟通过（debug ckpt 已删，日志在 $OUT.log）"
else
    echo "  ✗ 冒烟失败，看 $OUT.log"
    exit 1
fi
