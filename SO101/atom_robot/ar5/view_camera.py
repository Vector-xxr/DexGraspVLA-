#!/usr/bin/env python3
"""看相机画面：先找到它，再摆好它。

    python -m atom_robot.ar5.view_camera --list        # 扫描有哪些相机
    python -m atom_robot.ar5.view_camera               # 看 cam_front（腕部 305，走 V4L2）
    python -m atom_robot.ar5.view_camera --name cam_high
    python -m atom_robot.ar5.view_camera --source 4    # 直接指定设备号
    python -m atom_robot.ar5.view_camera --source orbbec:CP0BB53000FC   # Orbbec 按序列号走 OBSENSOR（只对 335）
    python -m atom_robot.ar5.view_camera --all         # 所有配置的相机并排

窗口里的按键：
    q / Esc   退出
    s         存一张 PNG 到当前目录
    g         叠加九宫格，摆正视角用
    i         叠加信息条（分辨率 / 实测帧率）

没有图形界面时（ssh 进来的）加 --save-only，它只抓几帧存成文件。

俯视 335 按序列号写成 `orbbec:<序列号>`，走 OBSENSOR，换 USB 口也不用改。
腕部 305 **不能走 OBSENSOR**（一帧不出，还会把 V4L2 也弄坏，只能拔插），10-15 起 config
里写成 `v4l2:/dev/ar5_cam_wrist`，明确走 V4L2。别的 Orbbec 写法（裸路径、设备号）一律
改道 OBSENSOR，`--list` 也不用 V4L2 去碰 Orbbec（335 被碰过 OBSENSOR 就只剩 0.3 Hz）。
普通 USB 摄像头的索引取决于枚举顺序，用 --list 找 by-path 写法。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from atom_robot.ar5.config import AR5Config  # noqa: E402


def capture_nodes(limit: int) -> dict:
    """问内核每个 /dev/videoN 支持什么像素格式，不打开设备。

    深度相机（比如 Orbbec Gemini 335）一个设备会占好几个节点：彩色、深度、红外，
    外加几个**只报 metadata、没有任何像素格式**的节点。用 OpenCV 去打开后者会
    直接卡死 —— 实测扫 0..7 挂了三分钟被 kill。所以先用 v4l2 筛一遍，
    只把真能出像素的节点交给 OpenCV。
    """
    import subprocess

    nodes = {}
    for index in range(limit):
        path = Path(f"/dev/video{index}")
        if not path.exists():
            continue
        try:
            out = subprocess.run(["v4l2-ctl", "-d", str(path), "--list-formats"],
                                 capture_output=True, text=True, timeout=5).stdout
        except (OSError, subprocess.SubprocessError):
            # 没有 v4l2-ctl 就别猜，交给 OpenCV，让它自己碰运气
            nodes[index] = None
            continue
        formats = [line.split("'")[1].strip()
                   for line in out.splitlines() if "'" in line and "[" in line]
        if formats:
            nodes[index] = formats
    return nodes


# 能出彩色图的格式。深度(Z16)和红外(GREY/BA81)不是我们要预览的东西。
_COLOUR_FORMATS = {"YUYV", "MJPG", "MJPEG", "RGB3", "BGR3", "UYVY", "NV12", "H264"}


def scan(limit: int = 10) -> list:
    """探测哪些设备号能真的出图。

    只看 isOpened() 是不够的：有的后端对不存在的设备也返回 True，直到读帧才失败。
    """
    import cv2

    from atom_robot.ar5.drivers.cameras import orbbec_serial_of_node

    found = []
    for index, formats in capture_nodes(limit).items():
        # **Orbbec 的节点一个都不碰。** 用 V4L2 开一下，之后 OBSENSOR（collect、RobotJev
        # 都用它）就塌到 0.3 Hz 且不报错。Orbbec 在下面单独按序列号列。
        if orbbec_serial_of_node(index) is not None:
            continue
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            cap.release()
            continue
        ok, frame = cap.read()
        if ok and frame is not None:
            h, w = frame.shape[:2]
            fps = cap.get(cv2.CAP_PROP_FPS)
            found.append((index, w, h, fps, formats or []))
        cap.release()
    return found


def open_capture(source, width: int, height: int, fps: int, name: str = "camera"):
    """和 collect 走同一套打开逻辑（Orbbec 走 OBSENSOR，`v4l2:` 开头的走 V4L2），不自己 cv2.VideoCapture。"""
    from atom_robot.ar5.drivers.cameras import SingleCamera

    cap = SingleCamera(source, width, height, fps, name=name)
    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        raise IOError(f"设备 {source!r} 打开了但读不出画面（被别的进程占用？）")
    return cap


def draw_grid(frame) -> None:
    """三分线 + 中心十字。摆正腕部相机时对齐用。"""
    import cv2

    h, w = frame.shape[:2]
    for i in (1, 2):
        cv2.line(frame, (w * i // 3, 0), (w * i // 3, h), (0, 255, 0), 1)
        cv2.line(frame, (0, h * i // 3), (w, h * i // 3), (0, 255, 0), 1)
    cv2.drawMarker(frame, (w // 2, h // 2), (0, 0, 255), cv2.MARKER_CROSS, 28, 2)


def draw_info(frame, text: str) -> None:
    import cv2

    cv2.rectangle(frame, (0, 0), (frame.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(frame, text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--list", action="store_true", help="扫描能出图的设备号后退出")
    parser.add_argument("--name", default="cam_front", help="CameraConfig.sources 里的名字")
    parser.add_argument("--source", help="绕过配置，直接给设备号、路径或 orbbec:<序列号>")
    parser.add_argument("--all", action="store_true", help="配置里所有相机一起看")
    parser.add_argument("--save-only", type=int, metavar="N",
                        help="不开窗口，抓 N 帧存成 PNG（无图形界面时用）")
    parser.add_argument("--scan-limit", type=int, default=10, help="--list 扫到几号为止")
    args = parser.parse_args()

    try:
        import cv2
    except ImportError:
        print("没装 opencv：pip install opencv-python")
        return 1

    cfg = AR5Config().cameras

    if args.list:
        print(f"扫描设备 0..{args.scan_limit - 1} ...")
        found = scan(args.scan_limit)
        from atom_robot.ar5.drivers.cameras import orbbec_devices, orbbec_holders
        orbbecs = orbbec_devices()
        if orbbecs:
            model = {"0800": "Gemini 335", "0840": "Gemini 305"}
            print(f"\nOrbbec {len(orbbecs)} 台（走 OBSENSOR，按序列号认，这里不打开）：")
            for port, sn, pid in orbbecs:
                src = f"orbbec:{sn}"
                names = [n for n, s in cfg.sources.items() if s == src]
                tag = f"   <- 配置里是 {', '.join(names)}" if names else ""
                busy = orbbec_holders(sn)
                held = f"   ⚠ 被 PID {', '.join(str(p) for p, _ in busy)} 占着" if busy else ""
                print(f'  {model.get(pid, pid):<11} USB {port:<7} "{src}"{tag}{held}')
        if not found and not orbbecs:
            print("\n没有能出图的相机。")
            print("  - `ls /dev/video*` 有东西吗？没有就是没插上/没被内核识别")
            print("  - `lsusb` 里看得到摄像头吗？")
            print("  - 用户在 video 组里吗？（`groups`，改完要重新登录）")
            return 1
        if not found:
            return 0
        print(f"\n另外找到 {len(found)} 个能出图的普通 USB 相机节点：")
        colour = []
        for index, w, h, fps, formats in found:
            names = [n for n, s in cfg.sources.items() if s == index]
            is_colour = not formats or any(f in _COLOUR_FORMATS for f in formats)
            if is_colour:
                colour.append(index)
            kind = "彩色" if is_colour else "非彩色"
            fmt = ("/".join(formats)) if formats else "未知格式"
            tag = f"   <- 配置里是 {', '.join(names)}" if names else ""
            print(f"  设备 {index}:  {w}x{h}  {fps:.0f} fps  [{fmt}] {kind}{tag}")

        if colour:
            print(f"\n彩色流在设备 {colour}。深度相机会把深度/红外也报成 capture 设备，")
            print("预览和策略要的是彩色那个。")

            # 同型号的相机（尤其序列号也一样的）用索引会在重插后互换，
            # 把腕部画面喂成俯视画面。按 USB 端口写才稳。
            by_path = Path("/dev/v4l/by-path")
            if by_path.is_dir():
                stable = {}
                for link in sorted(by_path.iterdir()):
                    try:
                        target = link.resolve()
                    except OSError:
                        continue
                    if target.name.startswith("video") and target.name[5:].isdigit():
                        index = int(target.name[5:])
                        if index in colour and "usbv2" not in link.name:
                            stable.setdefault(index, str(link))
                if stable:
                    print("\n填进 config.py 时用下面这个写法，重插不会串：")
                    for index, path in sorted(stable.items()):
                        print(f'  设备 {index}:  "{path}"')
        configured = {s for s in cfg.sources.values() if isinstance(s, int)}
        wrong = {s for s in configured if s not in colour}
        if wrong:
            print(f"\n⚠ config.py 里配的设备 {sorted(wrong)} 不是彩色流（或压根没出图）。")
            print(f"  按上面的号改 CameraConfig.sources。")
        return 0

    # 决定要看哪些
    if args.source is not None:
        source = int(args.source) if args.source.isdigit() else args.source
        targets = [(args.name if args.name else "camera", source)]
    elif args.all:
        targets = list(cfg.sources.items())
        if not targets:
            print("CameraConfig.sources 是空的")
            return 1
    else:
        if args.name not in cfg.sources:
            print(f"配置里没有 '{args.name}'。已有：{list(cfg.sources)}")
            return 1
        targets = [(args.name, cfg.sources[args.name])]

    caps = {}
    try:
        for name, source in targets:
            try:
                caps[name] = open_capture(source, cfg.width, cfg.height, cfg.fps, name=name)
                print(f"  {name}: 设备 {source!r} 已打开")
            except IOError as exc:
                print(f"  {name}: {exc}")
        if not caps:
            print("\n一个相机都没打开。先跑 --list 看看有什么。")
            return 1

        if args.save_only:
            for i in range(args.save_only):
                for name, cap in caps.items():
                    ok, frame = cap.read()
                    if not ok:
                        print(f"  {name}: 第 {i} 帧读失败")
                        continue
                    path = Path(f"{name}_{i:02d}.png").resolve()
                    cv2.imwrite(str(path), frame)
                    print(f"  已存 {path}")
                time.sleep(0.1)
            return 0

        print("\n按键：q/Esc 退出   s 存图   g 网格   i 信息条")
        grid = False
        info = True
        frames_seen = 0
        t0 = time.time()
        measured = 0.0
        while True:
            for name, cap in caps.items():
                ok, frame = cap.read()
                if not ok:
                    print(f"\n{name}: 读帧失败，相机掉了？")
                    return 1
                shown = frame.copy()
                if grid:
                    draw_grid(shown)
                if info:
                    h, w = frame.shape[:2]
                    draw_info(shown, f"{name}  {w}x{h}  {measured:.1f} fps")
                cv2.imshow(name, shown)

            frames_seen += 1
            if frames_seen % 10 == 0:
                now = time.time()
                measured = 10.0 / (now - t0)
                t0 = now

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("g"):
                grid = not grid
            if key == ord("i"):
                info = not info
            if key == ord("s"):
                stamp = time.strftime("%H%M%S")
                for name, cap in caps.items():
                    ok, frame = cap.read()
                    if ok:
                        path = Path(f"{name}_{stamp}.png").resolve()
                        cv2.imwrite(str(path), frame)
                        print(f"  已存 {path}")
    finally:
        for cap in caps.values():
            cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
