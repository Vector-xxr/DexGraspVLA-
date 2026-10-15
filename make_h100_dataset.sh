#!/usr/bin/env bash
# 把采集目录转成 DexGraspVLA 数据集，直接放进 H100 包的 Datasets/ 下，可以原样拷走。
#
# 用法（不用先 conda activate）：
#   ./make_h100_dataset.sh <数据集名> <任务目录> [<任务目录> ...]
#   ./make_h100_dataset.sh atom_grasp_v05_20261010 ~/atom_episodes/water_bottle ~/atom_episodes/pingpong_ball
#
# 每个任务目录要有 head_bbox.json（头部相机第 0 帧上的目标框）：
#   {"episode_0000": {"box": [x1, y1, x2, y2], "frames": 700}, ...}
# 缺框的 episode 会直接报错列出来。补框（弹窗手画，只弹没框的）：
#   env PYTHONPATH= ~/miniforge3/envs/dexgraspvla/bin/python ../../数据_硬件/灵巧手遥操/VR设备/to_dexgraspvla.py \
#       --src <任务目录> --bbox-only --wrist-cam cam_front --head-cam cam_high --action next-state
#
# 产出 ~/H100_dexgraspvla/Datasets/<数据集名>/：
#   <任务>.zarr/          一个任务一个，任务名 = 任务目录名
#   <任务>_mask_qc/       每条一张首/中/末帧叠掩码的质检图，**拷走前看一遍**
#   SHA256SUMS            整个目录的校验和，H100 上 sha256sum -c 核对
#
# 步骤：转换（SAM + Cutie 真掩码）→ check_dataset.py 全量体检 → 生成 SHA256SUMS。
# 冒烟用 ./smoke_h100_dataset.sh <数据集名>（和 H100 同一个 Docker 镜像，断网跑）。
# 包的位置可用环境变量 H100_ROOT 改，默认 ~/H100_dexgraspvla。
set -euo pipefail

if [ $# -lt 2 ]; then
    sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
fi
NAME=$1; shift
HERE="$(dirname "$(readlink -f "$0")")"
# Cutie 要的 resnet50 用项目里的 weights/torch_home，不去 ~/.cache/torch
export TORCH_HOME="${TORCH_HOME:-$HERE/weights/torch_home}"
R=${H100_ROOT:-$HOME/H100_dexgraspvla}
OUT=$R/Datasets/$NAME
CONVERT=$HERE/../../数据_硬件/灵巧手遥操/VR设备/to_dexgraspvla.py
CHECK=$R/scripts/check_dataset.py
SAM=$HERE/weights/sam_vit_h_4b8939.pth
# ROS 的 py3.12 PYTHONPATH 会污染这个 3.11 环境（见 环境说明.md 第 2 节）
PY="env PYTHONPATH= $HOME/miniforge3/envs/dexgraspvla/bin/python"

case $NAME in
    *[!A-Za-z0-9_.-]*) echo "  ✗ 数据集名只用字母数字和 _ . -（hydra 路径和远程目录都省事）：$NAME"; exit 2 ;;
esac
for SRC in "$@"; do
    [ -d "$SRC" ] || { echo "  ✗ 没有这个目录：$SRC"; exit 2; }
    [ -f "$SRC/head_bbox.json" ] || { echo "  ✗ $SRC 没有 head_bbox.json"; exit 2; }
    T=$(basename "$(readlink -f "$SRC")")
    [ ! -e "$OUT/$T.zarr" ] || { echo "  ✗ $OUT/$T.zarr 已存在。换个数据集名，或确认不要了再删"; exit 2; }
done
mkdir -p "$OUT"

ZARRS=()
for SRC in "$@"; do
    T=$(basename "$(readlink -f "$SRC")")
    echo; echo "===== 转换 $T  ($(date +%H:%M:%S))"
    $PY "$CONVERT" --src "$SRC" --dst "$OUT/$T.zarr" \
        --wrist-cam cam_front --head-cam cam_high --action next-state \
        --mask sam --bbox-file "$SRC/head_bbox.json" --sam-ckpt "$SAM"
    ZARRS+=("$OUT/$T.zarr")
done

echo; echo "===== 体检  ($(date +%H:%M:%S))"
$PY "$CHECK" "${ZARRS[@]}" --dim 8

echo; echo "===== 校验和  ($(date +%H:%M:%S))"
( cd "$OUT" && rm -f SHA256SUMS && find . -type f ! -name SHA256SUMS -print0 | sort -z \
    | xargs -0 sha256sum > SHA256SUMS )
echo "  $(wc -l < "$OUT/SHA256SUMS") 个文件，$(du -sh "$OUT" | cut -f1)"

PATHS=$(printf "'/data/Datasets/$NAME/%s'," $(cd "$OUT" && ls -d *.zarr)); PATHS="[${PATHS%,}]"
cat <<EOF

  做好了：$OUT
  1. 看质检图：$OUT/*_mask_qc/
  2. 本机冒烟：$HERE/smoke_h100_dataset.sh $NAME
  3. 拷到 H100 的 ~/H100_dexgraspvla/Datasets/$NAME/，在那边核对：
       cd ~/H100_dexgraspvla/Datasets/$NAME && sha256sum -c --quiet SHA256SUMS && echo ALL_OK
  4. 训练时：
       "task.dataset_paths=$PATHS"
       'task.shape_meta.obs.right_state.shape=[8]' 'task.shape_meta.action.shape=[8]'
EOF
