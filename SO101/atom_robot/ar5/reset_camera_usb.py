#!/usr/bin/env python3
"""把一台 USB 相机软复位，等于拔下来再插回去。

    python -m atom_robot.ar5.reset_camera_usb              # 复位腕部 Orbbec
    python -m atom_robot.ar5.reset_camera_usb --port 1-10.4  # 顶部那台
    python -m atom_robot.ar5.reset_camera_usb --list       # 只看现状

**什么时候用**：相机打开之后 `read()` 一直卡住（裸 `v4l2-ctl --stream-mmap` 也卡），
或者 `by-path` 路径整个消失了。腕部那台 Orbbec 两种都犯过。

两种典型成因：

* **跑过 `pyorbbecsdk`**。它走 libuvc，会把内核 uvcvideo 从彩色接口上摘掉，
  于是 `/dev/videoN` 和对应的 by-path 链接一起消失。SDK 退出也不会自己还回来。
* **设备自己卡住**。枚举正常、`--list-formats` 也答得出来，但控制端点不回包
  （SDK 会打一串 `setXu failed`），一帧都出不来。

复位只碰指定的那一个设备，同一个 hub 上的别的设备不受影响 —— 本机主臂在
`1-10.1`、顶部相机在 `1-10.4`、腕部在 `1-10.3`，夹爪根本在另一条总线上。

不需要 sudo：`/dev/bus/usb/*` 在本机是 `crw-rw-rw-`。要是报权限不够，
那就只能物理拔插。
"""

from __future__ import annotations

import argparse
import fcntl
import os
import sys
import time
from pathlib import Path

# USBDEVFS_RESET，见 linux/usbdevice_fs.h。_IO('U', 20)，无参数。
USBDEVFS_RESET = ord("U") << 8 | 20

SYSFS = Path("/sys/bus/usb/devices")

# 本机默认：腕部 Orbbec Gemini 305。口号是物理插座，换口要改。
DEFAULT_PORT = "1-10.3"


def describe(port: str) -> str:
    """这个口上插着什么，以及它现在还剩几个 video 节点。"""
    base = SYSFS / port
    if not base.exists():
        return f"{port}: 这个口上没有设备"

    def read(name, default="?"):
        try:
            return (base / name).read_text().strip()
        except OSError:
            return default

    vid, pid = read("idVendor"), read("idProduct")
    product = read("product", "")
    line = f"{port}: {vid}:{pid} {product}  dev {read('devnum')}  {read('speed')}M"

    # 有几个接口、各自还挂着 video 节点没有。彩色接口丢了节点就是上面说的
    # 那个 libuvc 摘驱动的状态。
    nodes = []
    for iface in sorted(base.glob(f"{port}:*")):
        vids = sorted(p.name for p in (iface / "video4linux").glob("video*")) \
            if (iface / "video4linux").exists() else []
        if vids:
            nodes.append(f"{iface.name.split(':')[-1]}->{','.join(vids)}")
    line += "\n    video 节点: " + (" ".join(nodes) if nodes else "（一个都没有）")
    return line


def reset(port: str) -> None:
    base = SYSFS / port
    if not base.exists():
        raise SystemExit(f"{port} 上没有设备。`--list` 看看有哪些口。")

    # 设备号每次重新枚举都会变，所以现查，不能写死 —— 这里踩过：
    # 上一条命令里读到的 063，下一条命令时已经是 064 了，开出来是 ENODEV。
    busnum = int((base / "busnum").read_text())
    devnum = int((base / "devnum").read_text())
    node = f"/dev/bus/usb/{busnum:03d}/{devnum:03d}"

    try:
        fd = os.open(node, os.O_WRONLY)
    except PermissionError:
        raise SystemExit(f"打不开 {node}（权限）。只能物理拔插，或者 sudo 跑。")
    except FileNotFoundError:
        raise SystemExit(f"{node} 不在了 —— 设备刚好在这一刻重新枚举了，再跑一次。")
    try:
        fcntl.ioctl(fd, USBDEVFS_RESET, 0)
    finally:
        os.close(fd)
    print(f"  复位指令已发到 {node}（{port}）")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", default=DEFAULT_PORT,
                        help=f"sysfs 里的 USB 口号，默认 {DEFAULT_PORT}（腕部 Orbbec）")
    parser.add_argument("--list", action="store_true", help="只打印现状，不复位")
    parser.add_argument("--wait", type=float, default=5.0,
                        help="复位后等几秒再看结果。重新枚举要时间，别急着量")
    args = parser.parse_args()

    if args.list:
        # 只列插着相机的口：有 video4linux 子目录的才算
        found = False
        for base in sorted(SYSFS.glob("*-*")):
            if ":" in base.name:
                continue
            if any((i / "video4linux").exists() for i in base.glob(f"{base.name}:*")):
                print(describe(base.name)); found = True
        if not found:
            print("没找到任何 USB 相机")
        return 0

    print("复位前：")
    print("  " + describe(args.port).replace("\n", "\n  "))
    reset(args.port)
    time.sleep(args.wait)
    print("复位后：")
    print("  " + describe(args.port).replace("\n", "\n  "))
    print("\n  节点回来了就重测：python -m atom_robot.ar5.view_camera --all")
    return 0


if __name__ == "__main__":
    sys.exit(main())
