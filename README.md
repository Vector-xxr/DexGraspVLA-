# DexGraspVLA-（抓取控制器训练）

在 RTX 5070 Ti 上单卡训练 DexGraspVLA 的抓取控制器；把 SO-101 主从臂采集的数据（8 维：7 臂 + 1 夹爪）转成训练用的 zarr（SAM + Cutie 生成目标掩码），打包成 H100 离线训练用的数据集。

下面的命令都在项目根目录下跑：`cd ~/yuhang_workspace/算法/DexGraspVLA-`。

> [!NOTE]
> **放在 GitHub 上要知道的**：这个仓库原本是一个更大工作区里的项目（`<工作区>/算法/DexGraspVLA-/`）。
> - 模型权重（`weights/`、`third_party/Cutie/weights/`）、数据集和训练输出（`DexGraspVLA/data/`）不在仓库里，下载和摆放见 [`环境说明.md`](环境说明.md) 第 5 节。
> - 数据转换器 `to_dexgraspvla.py` 在另一个仓库 [VisionPro_teleop_DexterousHand](https://github.com/Vector-xxr/VisionPro_teleop_DexterousHand) 的 `VR设备/` 下；H100 离线包（`~/H100_dexgraspvla`）不在任何仓库里。
> - 文中指向 `../../工作区说明.md`、`../../数据_硬件/…` 的链接是原工作区里的其他项目，在这里打不开。
> - `DexGraspVLA/` 是上游 [Psi-Robot/DexGraspVLA](https://github.com/Psi-Robot/DexGraspVLA) 的代码（改了 2 个文件），`third_party/Cutie/` 是上游 [hkchengrex/Cutie](https://github.com/hkchengrex/Cutie)，出处和改动见 [`上游来源.md`](上游来源.md)。

## 目录地图

```
算法/DexGraspVLA-/                   本项目，也是 git 仓库根
├── README.md                      本文件
├── 环境说明.md                     conda 环境、显存、训练命令、验证记录
├── 数据转换说明.md                  采集数据 → zarr：标框、掩码、检查、交付
├── 上游来源.md                     上游仓库和提交号、改了什么、哪些不进 git
├── 上游改动_DexGraspVLA.patch       对上游 2 个文件的改动
├── requirements-5070ti.txt         dexgraspvla 环境的确切版本
├── train_5070ti.sh                 单卡训练
├── make_h100_dataset.sh            转换 + 体检 + 校验和，产出放进 H100 包
├── smoke_h100_dataset.sh           用和 H100 同一个 Docker 镜像断网冒烟
├── DexGraspVLA/                    上游代码（改了 2 个文件）
│   └── data/                       不进 git：grasp_demo_example（官方小数据集，3.7G）、outputs/（训练输出）
├── third_party/Cutie/              上游 Cutie，没改；weights/ 不进 git
└── weights/                        不进 git（6.1G）：SAM、官方控制器检查点、torch_home/（DINOv2、resnet18/50）
```

项目外面的：

| 位置 | 内容 |
|---|---|
| `数据_硬件/灵巧手遥操/VR设备/to_dexgraspvla.py`（工作区里，属于 `灵巧手遥操` 仓库） | 转换器：采集数据 → zarr |
| `~/atom_episodes/<任务>/` | 原始采集数据，目标框 `head_bbox.json` 也放在这里 |
| `~/H100_dexgraspvla/` | H100 离线包：`Datasets/`（做好的数据集）、`docs/`、`scripts/`（体检、重建环境）、`smoke_out_local/`（本机冒烟输出） |
| `~/wandb_dexgraspvla/` | 训练的 wandb 离线日志 |

## conda 环境

- `dexgraspvla`：本机训练和数据转换。SO101 采集和 VR 遥操也用它，别随便改。
- `dexgraspvla_cu128`：H100 离线包的环境，本机只用来冒烟 H100 包。

详见 [`环境说明.md`](环境说明.md) 第 1 节。

## 当前状态

| 日期 | 进展 |
|---|---|
| 09-14 | 环境搭好 |
| 10-08 | 补齐 SAM + Cutie 真掩码、8 维从官方检查点部分加载，训练链路验证通过（[`环境说明.md`](环境说明.md) 第 7 节）；做好 H100 离线包，本机断网冒烟通过 |
| 10-10 | 第一个自采数据集 `atom_grasp_v05_20261010` 做好（只用于冒烟），本机检查全部通过；H100 上官方数据冒烟通过（用户反馈） |
| 10-15 | 用户确认 v05 已拷到 H100；目录改名后，debug 训练和转换 2 条离线复验通过 |

还没做的：

- 完整训练，看收敛效果
- 推理链路（planner 没装）和任何真机测试
- 在 H100 上用自采数据冒烟、训练，以及 8 卡 DDP：本机没有记录（本机只有 1 卡，验证不了 8 卡）
- `--select-bbox` 补框弹窗没在真显示器上用过
- 采集时直接框目标（10-08 定的方向）：`collect.py` 还没有这个功能
- 13 维数据（Vision Pro 遥操 + 因时手）：转换器支持，但还没有这类数据；两个 H100 脚本只支持 8 维
- `held` 帧（P 键保持）和 30 Hz 长 episode（官方数据每条 75 帧）：转换器原样保留，留给采集端处理

## 流程

1. **补框**：新采的 episode 要在头部相机第 0 帧上框出目标，存进 `head_bbox.json`（细节见 [`数据转换说明.md`](数据转换说明.md) D 节）：
   ```bash
   env PYTHONPATH= ~/miniforge3/envs/dexgraspvla/bin/python ../../数据_硬件/灵巧手遥操/VR设备/to_dexgraspvla.py \
       --src ~/atom_episodes/<任务> --bbox-only --wrist-cam cam_front --head-cam cam_high --action next-state
   ```
2. **转换 + 体检 + 校验和**，产出在 `~/H100_dexgraspvla/Datasets/<数据集名>/`：
   ```bash
   bash make_h100_dataset.sh <数据集名> ~/atom_episodes/<任务A> ~/atom_episodes/<任务B>
   ```
3. **看质检图，再冒烟**。质检图在 `~/H100_dexgraspvla/Datasets/<数据集名>/<任务>_mask_qc/`，怎么看见 [`数据转换说明.md`](数据转换说明.md) G 节：
   ```bash
   bash smoke_h100_dataset.sh <数据集名>
   ```
4. **本机训练**。参数说明、接着训的写法见 [`环境说明.md`](环境说明.md) 第 6 节：
   ```bash
   CK=$PWD/weights/dexgraspvla-controller-20250320-opensource/dexgraspvla-controller-20250320.ckpt
   D=$HOME/H100_dexgraspvla/Datasets/<数据集名>
   bash train_5070ti.sh "policy.start_ckpt_path='$CK'" +training.start_ckpt_skip_mismatch=true \
       "task.dataset_paths=['$D/<任务A>.zarr','$D/<任务B>.zarr']" \
       'task.shape_meta.obs.right_state.shape=[8]' 'task.shape_meta.action.shape=[8]'
   ```
5. **上 H100**：把 `~/H100_dexgraspvla/Datasets/<数据集名>/` 整个目录拷过去，按 `~/H100_dexgraspvla/docs/自采数据集上传H100流程.md` 的 ⑤~⑨ 核对、冒烟、训练。拷完先在 H100 上核对：
   ```bash
   cd ~/H100_dexgraspvla/Datasets/<数据集名> && sha256sum -c --quiet SHA256SUMS && echo ALL_OK
   ```

## 离线测试

两项都不碰硬件，用的是项目里的模型。

**训练冒烟**（官方小数据集，2 轮 × 3 步）：

```bash
bash train_5070ti.sh training.debug=True
```

正常：1 分钟左右（10-15 实测）exit 0，wandb 汇总里 `global_step 6`，
`DexGraspVLA/data/outputs/<日期>/<时间>_train_dexgraspvla_controller_grasp/checkpoints/` 下存出一个检查点（2.2 GB，看完可删）。
中间这些警告是正常的：3 条 `xFormers is not available`，几条 `RuntimeWarning: invalid value encountered in cast`（原因见 [`环境说明.md`](环境说明.md) 第 4 节）。

**转换 2 条**（pingpong_ball 第 0、1 条，SAM + Cutie）：

```bash
mkdir -p /tmp/dexgraspvla_test
env PYTHONPATH= ~/miniforge3/envs/dexgraspvla/bin/python ../../数据_硬件/灵巧手遥操/VR设备/to_dexgraspvla.py \
    --src ~/atom_episodes/pingpong_ball --dst /tmp/dexgraspvla_test/pingpong_ball.zarr \
    --wrist-cam cam_front --head-cam cam_high --action next-state \
    --mask sam --bbox-file ~/atom_episodes/pingpong_ball/head_bbox.json --episodes 0,1
rm -rf /tmp/dexgraspvla_test      # 看完质检图再删
```

正常输出（10-15）：

```
  主从臂格式：action 用「下一帧的实测关节角」构造
  8 条   维度 8 = 7 臂 + 1 末端 ['gripper']
  ...
    SAM 分数 0.991  首帧掩码占比 0.005  掩码为空 0/491 帧
  episode_0000.npz -> 491 帧（累计 491）
    SAM 分数 0.990  首帧掩码占比 0.004  掩码为空 86/478 帧
  episode_0001.npz -> 478 帧（累计 969）
  ...
  2 条 / 969 帧
```

开头几条 Cutie 的 `FutureWarning`、结尾 `episode_0001：86/478 帧掩码为空`（球被提出了画面）和「维度是 8」的提示都是正常的。
质检图在 `/tmp/dexgraspvla_test/pingpong_ball_mask_qc/`。

想确认是真的断网也能跑，按 [`../../工作区说明.md`](../../工作区说明.md) 第 5 节的方法把代理和缓存指到别处再测。

## 已知的坑

- 跑 Python 前要加 `PYTHONPATH=`，否则 ROS 会污染环境（[`环境说明.md`](环境说明.md) 第 3 节）。
- hydra 参数里的路径要写绝对路径，外面再加一层单引号，像 `"policy.start_ckpt_path='…'"`，否则「算法」两个字会让 hydra 报错（[`环境说明.md`](环境说明.md) 第 6 节）。
- `+training.start_ckpt_skip_mismatch=true` 前面的 `+` 不能少；接着训自己存的检查点时不要加这个开关。
- normalizer 每次按当前数据重新拟合，所以从官方检查点起步时 loss 偏高，不是出错。
- 图像存 BGR，转换时不要转成 RGB（[`数据转换说明.md`](数据转换说明.md) C 节）。
- 主从臂数据必须给 `--action next-state`。
- 框文件靠帧数对上 episode，重排过编号就要重新框（[`数据转换说明.md`](数据转换说明.md) D 节）。
- Cutie 的两个权重、`weights/torch_home` 里的 resnet18 / resnet50 都要在，否则断网时 Cutie 起不来。
- 工作区换了位置，要改 `~/miniforge3/envs/dexgraspvla/lib/python3.11/site-packages/_editable_impl_cutie.pth` 里的 Cutie 路径（[`工作区说明.md`](../../工作区说明.md) 6.3 节）。
- H100 包里有一份上游代码副本 `~/H100_dexgraspvla/DexGraspVLA/`（10-15 核对和本地一致）。改了上游代码，要同步过去再重新冒烟。
- `make_h100_dataset.sh`、`smoke_h100_dataset.sh` 只支持 8 维；`smoke_out_local/<数据集名>/` 已存在时冒烟脚本拒跑，要先删掉。
- 上游 DexGraspVLA 没有许可证文件；本仓库 10-16 在知情的情况下决定公开，出处见 [`上游来源.md`](上游来源.md)。

## 文档索引

| 文档 | 内容 |
|---|---|
| [`环境说明.md`](环境说明.md) | conda 环境、显存、训练命令、验证记录 |
| [`数据转换说明.md`](数据转换说明.md) | 采集数据 → zarr 的全过程 |
| [`上游来源.md`](上游来源.md) | 上游仓库、提交号、改动、哪些不进 git |
| [`上游改动_DexGraspVLA.patch`](上游改动_DexGraspVLA.patch) | 对上游 2 个文件的改动全文 |
| [`requirements-5070ti.txt`](requirements-5070ti.txt) | `dexgraspvla` 环境的版本和偏离上游的原因 |
| `~/H100_dexgraspvla/docs/自采数据集上传H100流程.md` | 自采数据从本机到 H100 训练的全流程 |
| `~/H100_dexgraspvla/docs/DexGraspVLA_H100使用说明书.md` | 在 H100 上怎么跑：目录、启动模板、训练示例、常见问题 |
| `~/H100_dexgraspvla/docs/H100迁移DexGraspVLA说明书.md` | H100 离线包怎么做出来的：环境、Docker 镜像、传输、远程冒烟 |
| [`../../工作区说明.md`](../../工作区说明.md) | 整个工作区：结构、约定、离线测试、迁移 |
| [`DexGraspVLA/README.md`](DexGraspVLA/README.md) | 上游的说明 |
