# DexGraspVLA-

在 [DexGraspVLA](https://github.com/Psi-Robot/DexGraspVLA) 的抓取控制器上做改进：适配自己的机器人（珞石 AR5 七轴臂 + 夹爪 / 灵巧手）和数据，在单张 RTX 5070 Ti（16 GB）上训练。

仓库结构和上游一样，上游代码在根目录；上游原来的英文说明见 [`README_DexGraspVLA.md`](README_DexGraspVLA.md)。

> [!NOTE]
> **放在 GitHub 上要知道的**
> - 模型权重（`weights/`）、数据集和训练输出（`data/`）、第三方库 Cutie（`third_party/`）都不在仓库里，下载和安装见 [`环境说明.md`](环境说明.md) 第 5 节和 [`requirements-5070ti.txt`](requirements-5070ti.txt)。
> - 自己采的数据怎么转成训练用的 zarr（采集、标框、SAM + Cutie 生成掩码、打包）不在本仓库：转换器在 [VisionPro_teleop_DexterousHand](https://github.com/Vector-xxr/VisionPro_teleop_DexterousHand) 的 `VR设备/to_dexgraspvla.py`，流程说明和打包脚本在原工作区的 `数据_硬件/SO101/`（没有公开）。这里只写训练要的数据格式，见下文「数据格式」。
> - 文中指向 `../../工作区说明.md`、`../../数据_硬件/…` 的链接是原工作区里的其他项目，在这里打不开。
> - 上游 DexGraspVLA 没有附许可证文件；出处、提交号和改动见 [`上游来源.md`](上游来源.md)。

## 相对上游改了什么

| | 改动 |
|---|---|
| `controller/model/vision/obs_encoder.py` | 状态输入维度改成跟 `shape_meta` 走（上游写死 13 维）。自己的数据是 8 维：7 臂 + 1 夹爪 |
| `controller/workspace/train_dexgraspvla_controller_workspace.py` | 加开关 `+training.start_ckpt_skip_mismatch=true`：从官方 13 维检查点起步时，形状对得上的权重全部加载，对不上的（8 维时正好是两端 4 个张量）随机初始化 |
| 新增 `train_5070ti.sh` | 单卡训练：batch 12 × 梯度累积 4 = 等效上游的 48，显存峰值约 13.7 GB |
| 新增 `requirements-5070ti.txt` | 这张卡（sm_120）要 torch 2.11 + cu128，上游钉的 torch 2.5.1 跑不了 |

两处代码改动的全文在 [`上游改动_DexGraspVLA.patch`](上游改动_DexGraspVLA.patch)，在仓库根目录跑 `git apply --check -R 上游改动_DexGraspVLA.patch` 没有输出，就说明当前代码和补丁对得上。

## 目录

```
DexGraspVLA-/
├── controller/  planner/  inference_utils/  assets/           上游代码（改了上面 2 个文件）
├── train.py  train.sh  inference.py  inference.sh  attention_map_visualizer.py  requirements.txt   上游
├── README_DexGraspVLA.md                                     上游的说明
├── train_5070ti.sh  requirements-5070ti.txt                  本机单卡训练
├── 环境说明.md  上游来源.md  上游改动_DexGraspVLA.patch
├── data/        不进 git：grasp_demo_example（官方小数据集）、outputs/（训练输出）
├── weights/     不进 git：官方控制器检查点、SAM、torch_home/（DINOv2 等）
└── third_party/ 不进 git：Cutie
```

## 环境

- conda 环境 `dexgraspvla`（Python 3.11、torch 2.11.0+cu128），确切版本见 [`requirements-5070ti.txt`](requirements-5070ti.txt)，搭建过程和踩过的坑见 [`环境说明.md`](环境说明.md)。本机这个环境同时给数据采集和遥操用，装、卸包前先想好。
- 命令前都加 `PYTHONPATH=`：本机 `~/.bashrc` 加载的 ROS 会污染这个环境。
- 训练脚本把 `TORCH_HOME` 指到 `weights/torch_home`，把 wandb 记录放到 `~/wandb_dexgraspvla`（wandb 会建软链接，不放进仓库）。

## 训练

在仓库根目录下：

```bash
# 冒烟：官方小数据集，2 轮 × 3 步
bash train_5070ti.sh training.debug=True

# 自己的 8 维数据，从官方检查点部分加载
CK=$PWD/weights/dexgraspvla-controller-20250320-opensource/dexgraspvla-controller-20250320.ckpt
D=<数据集目录>
bash train_5070ti.sh "policy.start_ckpt_path='$CK'" +training.start_ckpt_skip_mismatch=true \
    "task.dataset_paths=['$D/<任务A>.zarr','$D/<任务B>.zarr']" \
    'task.shape_meta.obs.right_state.shape=[8]' 'task.shape_meta.action.shape=[8]'
```

- 路径写绝对路径，外面再加一层单引号（`"policy.start_ckpt_path='…'"`）：路径里有中文时，不加 hydra 会报 `LexerNoViableAltException`。
- `+training.start_ckpt_skip_mismatch=true` 前面的 `+` 不能少；接着训自己存的 8 维检查点时不要加这个开关。
- 输出在 `data/outputs/<日期>/<时间>_train_dexgraspvla_controller_grasp/`（检查点每个 2.2 GB）。
- normalizer 每次按当前数据重新拟合，所以从官方检查点起步时 loss 偏高，不是出错（分析见 [`环境说明.md`](环境说明.md) 第 6 节）。
- 8 卡训练用上游的 `train.sh`；本机只有 1 卡，没验证过。

## 数据格式

和上游官方数据一样是 zarr，所有 episode 沿第 0 维首尾相接：

```text
<任务>.zarr/
├── data/
│   ├── rgbm           (N, 480, 640, 4) uint8    头部相机 BGR + 第 4 通道目标掩码 0/1
│   ├── right_cam_img  (N, 480, 640, 3) uint8    腕部相机 BGR
│   ├── right_state    (N, D) float32            臂关节归一化到 [-1, 1] + 末端开合
│   └── action         (N, D) float32            绝对目标，同 right_state 的约定
└── meta/
    └── episode_ends   (E,) int64                每条的累计结束帧号，最后一个等于 N
```

- **图像存 BGR，不要转 RGB**：上游在 `obs_encoder.py` 里进 DINOv2 前自己翻。颜色顺序存反了，loss 上看不出来（官方检查点对通道顺序不敏感），只能靠按约定存。
- **关节归一化**：`q_norm = clip((q − (hi+lo)/2) / ((hi−lo)/2), −1, 1)`，`hi`/`lo` 是关节限位，和上游 `inference.py` 的 `unscale()` 同一套。推理时要用同一套限位还原。
- **维度**：官方 13 维（7 臂 + 6 指）；自己的夹爪数据 8 维（7 臂 + 1 夹爪，夹爪 0 = 合拢、1 = 全开）。不是 13 维时，训练要覆盖 `shape_meta` 两行（见上）。
- **掩码**：用 SAM 在头部相机第 0 帧按目标框分割，再用 Cutie 往后逐帧跟踪，和上游推理时的做法一致。

## 现状

| 日期 | 进展 |
|---|---|
| 09-14 | 环境在 RTX 5070 Ti 上搭好 |
| 10-08 | SAM + Cutie 掩码流程和官方掩码平均 IoU 0.976；8 维从官方检查点部分加载、接着训都验证通过（[`环境说明.md`](环境说明.md) 第 7 节） |
| 10-10 | 第一个自采数据集（只用于冒烟）做好并拷到 H100 |
| 10-16 | 改成和上游一样的平铺结构；数据转换和打包挪出本仓库 |

还没做的：完整训练、看收敛效果；推理链路（planner 没装）和真机测试；13 维（灵巧手）数据；8 卡训练。

## 离线测试

```bash
bash train_5070ti.sh training.debug=True
```

正常：1 分钟左右结束，exit 0，`data/outputs/…/checkpoints/` 下存出一个检查点（看完可删）。中间 3 条 `xFormers is not available` 和几条 `RuntimeWarning: invalid value encountered in cast` 是正常的（原因见 [`环境说明.md`](环境说明.md) 第 3、4 节）。

## 文档

| 文档 | 内容 |
|---|---|
| [`环境说明.md`](环境说明.md) | conda 环境、为什么不能用上游的 requirements、显存与 batch、权重放哪、训练命令、验证记录 |
| [`待测试.md`](待测试.md) | 10-16 定的测试清单：软件（训练、数据、推理链路）和换夹爪后的硬件测试，做完一项勾一项 |
| [`上游来源.md`](上游来源.md) | 上游仓库和提交号、改了什么、哪些不进 git |
| [`上游改动_DexGraspVLA.patch`](上游改动_DexGraspVLA.patch) | 对上游 2 个文件的改动全文 |
| [`requirements-5070ti.txt`](requirements-5070ti.txt) | `dexgraspvla` 环境的确切版本，以及偏离上游的原因 |
| [`README_DexGraspVLA.md`](README_DexGraspVLA.md) | 上游的说明 |
| `数据_硬件/SO101/数据转换说明.md`（原工作区） | 自己的采集数据 → zarr 的全过程 |
| [`../../工作区说明.md`](../../工作区说明.md)（原工作区） | 整个工作区：结构、约定、迁移 |
