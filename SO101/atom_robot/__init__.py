"""原子动作开发中心的机器人代码。

这套原来长在 VEL 仓库里（`算法/VEL/vel_ar5/openvla-energy/atom_robot/ar5/`），
模块路径是 `atom_robot.ar5.*` —— 那是 openvla/prismatic 那个仓库的目录
约定，跟本项目没关系。2026-09-23 搬过来，改成 `atom_robot.ar5.*`。

命令一律在 atom_robot 的上一级目录（工作区里的 `算法/DexGraspVLA-/SO101`）下跑：

    python -m atom_robot.ar5.collect --show-config
"""

import os
from pathlib import Path

# 搬过来之后，少数几处还要够到 **VEL 仓库自己的东西**：灵巧手遥操（`VEL/src/hand_teleop`）、
# SO-101 的 URDF、`prismatic`、`vla-scripts`。这些是 VEL 的资产，不该跟着搬，
# 但也不能再靠 `parents[3]` 数目录层数去猜 —— 搬完那个数就错了，而且错得没有报错，
# 只是路径不存在。
#
# 换成一个明确的常量，可以用环境变量 `VEL_REPO` 覆盖（换机器、换路径时用）。
#
# 默认值也不写死本机绝对路径：从本文件往上找「同时有 算法/ 和 数据_硬件/ 的目录」当工作区根。
# 整个工作区搬家、换用户、拷进移动硬盘都不用改；只把 SO101 单独拷走时用 YUHANG_WS 指到工作区根。


def _find_workspace() -> Path:
    env = os.environ.get("YUHANG_WS")
    if env:
        return Path(env).expanduser().resolve()
    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / "算法").is_dir() and (p / "数据_硬件").is_dir():
            return p
    return here.parents[4]   # 找不到就按现在的层级（算法/DexGraspVLA-/SO101/atom_robot/）推


WORKSPACE = _find_workspace()
VEL_REPO = Path(os.environ.get(
    "VEL_REPO", WORKSPACE / "算法" / "VEL" / "vel_ar5" / "openvla-energy")).expanduser()
# 珞石 xCoreSDK（Python 绑定在 Release/linux 下）；环境变量 XCORE_SDK 可覆盖
SDK_ROOT = Path(os.environ.get(
    "XCORE_SDK", WORKSPACE / "数据_硬件" / "xCoreSDK-Python")).expanduser()


def vel_path(*parts) -> Path:
    """VEL 仓库里的一个路径。找不到时**由调用方决定怎么办** —— 这里不抛，
    因为用到它的都是可选功能（灵巧手、策略评测），采集这条主路一个都不碰。"""
    return VEL_REPO.joinpath(*parts)
