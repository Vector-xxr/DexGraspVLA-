"""OpenCV camera capture for the AR5 setup.

Each camera is drained by its own thread and `read()` hands back whatever
arrived last. That is not about throughput -- one frame per step is all anyone
wants -- it is about *when* the step finishes.

`VideoCapture.read()` with a one-frame buffer blocks until the camera's next
frame is ready, so calling it inside the control loop welds the loop's period to
the camera's frame clock: up to 33 ms per camera at 30 fps, two cameras running
on independent clocks, and the wait varying from step to step as their phases
drift past each other. The arm's tracker works out how fast to move from
(goal - previous goal) / elapsed, so a loop period that wanders like that comes
out as a commanded velocity that wanders -- the arm shakes, and the camera is
the last place anyone looks.

The trade is that a loop faster than the camera sees the same frame twice.
That is the right way round: a repeated frame is visible in the data and
harmless, whereas jitter is invisible in the data and lands on the hardware.
`stale_reads` counts them.
"""

import logging
import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def who_holds(source) -> List[Tuple[int, str]]:
    """Which processes have this video device open, as (pid, command).

    Scanned out of /proc rather than shelled out to `fuser`, so it needs
    nothing installed and cannot hang. Only this user's processes are visible,
    which is the case that matters: the usual culprit is a preview window the
    operator left open in another terminal.
    """
    if not isinstance(source, (str, os.PathLike)):
        return []
    try:
        target = os.path.realpath(source)
    except OSError:
        return []

    holders = []
    me = os.getpid()
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        # **跳过自己。** 2026-09-28：相机"打开了但不出帧"时，connect() 报错里会查
        # 一遍占用 —— 而这时候占着它的正是报错的这个进程自己，于是提示写着
        # 「Held by: PID <自己>」，看起来像又开了一个 collect，把人往重复会话上引。
        if int(proc.name) == me:
            continue
        try:
            for fd in (proc / "fd").iterdir():
                if os.path.realpath(fd) != target:
                    continue
                command = (proc / "cmdline").read_bytes().replace(b"\0", b" ")
                holders.append((int(proc.name), command.decode(errors="replace").strip()))
                break
        except OSError:
            # The process exited mid-scan, or it is not ours to look into.
            continue
    return holders


def _open_failure_hint(name: str, source) -> str:
    """Why a camera would not open, in the operator's terms.

    "could not be opened" on its own sends people to check the wiring, when the
    two real causes need opposite fixes: a busy device wants a window closed, a
    missing by-path node means the camera moved ports or is unplugged.
    """
    lines = [f"Camera '{name}' (source {source!r}) could not be opened."]

    if isinstance(source, (str, os.PathLike)) and not os.path.exists(source):
        lines.append(
            "  That path does not exist. A by-path node is tied to the USB port, so "
            "moving the camera to another socket changes it. Re-run "
            "`python -m atom_robot.ar5.view_camera --list`, which prints the "
            "current by-path lines to paste into CameraConfig.sources."
        )
        return "\n".join(lines)

    holders = who_holds(source)
    if holders:
        lines.append(f"  It is open in {len(holders)} other process(es):")
        for pid, command in holders:
            lines.append(f"    PID {pid}  {command[:100]}")
        lines.append(
            "  Close that first -- a camera can only stream to one process. If it is "
            "a `view_camera` preview, press q in its terminal."
        )
        return "\n".join(lines)

    lines.append(
        "  Nothing else holds it, so this is the device itself: check `ls /dev/video*`, "
        "that the user is in the `video` group (`groups`), and that the node really "
        "carries pixels (`v4l2-ctl -d <node> --list-formats`) -- a depth camera "
        "publishes metadata-only nodes that open and never deliver."
    )
    return "\n".join(lines)


def check_sources(config) -> Dict[str, str]:
    """Look over the configured cameras without opening any. name -> problem.

    Worth doing before `AR5Env.connect()`, not during it: the arm comes up
    first, so a camera that is merely busy costs a power-on and an unwind of
    the real-time session before anyone finds out.
    """
    problems = {}
    present = None
    for name, source in config.sources.items():
        source, forced_v4l2 = strip_v4l2(source)
        serial = None if forced_v4l2 else (orbbec_serial(source) or orbbec_serial_of_node(source))
        if serial is not None:
            # 预检不打开相机：打开就是一次 OBSENSOR 会话，还得等它关干净
            if present is None:
                present = {sn for _, sn, _ in orbbec_devices()}
            if serial not in present:
                problems[name] = (f"USB 上找不到序列号 {serial} 的 Orbbec（没插，或者没插好）。"
                                  f"现有 Orbbec：{sorted(present) or '无'}")
                continue
            holders = orbbec_holders(serial)
            if holders:
                who = "；".join(f"PID {pid} {cmd[:60]}" for pid, cmd in holders)
                problems[name] = f"被占用（Orbbec {serial}）：{who}"
            continue
        if not isinstance(source, (str, os.PathLike)):
            continue                      # 整数设备号没法预检，只能打开才知道
        if not os.path.exists(source):
            problems[name] = f"路径不存在：{source}（相机换了 USB 口，或者没插）"
            continue
        holders = who_holds(source)
        if holders:
            who = "；".join(f"PID {pid} {cmd[:60]}" for pid, cmd in holders)
            problems[name] = f"被占用：{who}"
    return problems


def _disable_dynamic_framerate(source) -> None:
    """Stop the camera trading frame rate for exposure. Best effort.

    Measured on the two UGREEN cameras: with this control on, raw v4l2 streaming
    sits at ~19 fps at 640x480 in ordinary indoor light and does not recover at
    any exposure or pixel format; with it off it reaches ~30. It defaults to off
    in the UVC driver but comes up on here, so set it explicitly.
    """
    import subprocess  # noqa: PLC0415

    if not isinstance(source, (str, os.PathLike)):
        return
    try:
        subprocess.run(
            ["v4l2-ctl", "-d", str(source), "--set-ctrl", "exposure_dynamic_framerate=0"],
            capture_output=True, timeout=3,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug(f"[Cameras] could not clear exposure_dynamic_framerate: {exc}")


RECENT_FRAMES = 8


class _Grabber(threading.Thread):
    """Drains one camera, keeping only the newest frame."""

    def __init__(self, name: str, cap):
        super().__init__(name=f"ar5-cam-{name}", daemon=True)
        self.camera = name
        self._cap = cap
        self._frame = None
        self._seq = 0
        self._stamp = None
        # 最近几帧（序号, 时刻, 帧），OmniPicker 采集每圈取走新来的 —— 每一帧都要，不只是最新那张
        self._recent = deque(maxlen=RECENT_FRAMES)
        self._lock = threading.Lock()
        self._halt = threading.Event()
        self.errors = 0
        # 这一帧是什么时候拍的（time.perf_counter 的钟）。V4L2 给内核的缓冲区时刻（CLOCK_MONOTONIC，
        # 和 perf_counter 同一个钟）；10-20 实测腕部 305 程序拿到帧比它晚 24.4 ms、抖 0.5 ms。
        # 拿不到（返回 -1 或对不上）就退回「read 返回那一刻」，stamp_kind 记着是哪种。
        self.stamp_kind = None

    def run(self) -> None:
        import cv2  # noqa: PLC0415

        while not self._halt.is_set():
            ok, frame = self._cap.read()
            arrived = time.perf_counter()
            if not ok or frame is None:
                self.errors += 1
                # Do not spin on a camera that has gone away: a tight failing
                # loop would eat a core and starve the 1 kHz sender.
                self._halt.wait(0.02)
                continue
            try:
                kernel = self._cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            except Exception:                            # noqa: BLE001 —— 拿不到时刻不能把取帧线程弄死
                kernel = -1.0
            if 0.0 <= arrived - kernel < 1.0:
                stamp, kind = kernel, "kernel"
            else:
                stamp, kind = arrived, "arrival"
            # Converted here rather than in read(), so the control loop pays
            # nothing for it and a frame nobody reads costs only this thread.
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            with self._lock:
                self._frame, self._stamp = rgb, stamp
                self._seq += 1
                self.stamp_kind = kind
                self._recent.append((self._seq, stamp, rgb))

    def latest(self):
        with self._lock:
            return self._frame, self._seq

    def latest_stamped(self):
        """(帧, 序号, 拍摄时刻)，三样一起取，免得帧和时刻对不上。"""
        with self._lock:
            return self._frame, self._seq, self._stamp

    def frames_since(self, seq):
        """序号比 seq 新的帧 [(序号, 时刻, 帧), ...]；seq 为 None 给缓存里全部。"""
        with self._lock:
            return [item for item in self._recent if seq is None or item[0] > seq]

    def shutdown(self, timeout: float = 2.0) -> None:
        self._halt.set()
        if self.is_alive():
            # Bounded: a read already in flight can sit there for a frame time
            # or, on a camera that has been unplugged, considerably longer.
            self.join(timeout=timeout)


# ─────────────────────────────────────────────────────── Orbbec / OBSENSOR
#
# 2026-09-30 起两台 Orbbec 都走 OpenCV 自带的 OBSENSOR 后端，不走 V4L2。
#
# 为什么：OBSENSOR 和 V4L2 在同一台 Orbbec 上**互相污染**。RobotJev 用 OBSENSOR，
# collect 原来用 V4L2，谁后开谁倒霉：
#   * 跑过 OBSENSOR -> V4L2 彩色节点一帧不出（select() timeout），软复位 + 等 8 秒
#     也救不回来，只有物理拔插能救（9/30 实测）
#   * 跑过 V4L2 -> OBSENSOR 帧率从 ~25 Hz 塌到 **0.3 Hz，而且不报错**
# 两个程序都用 OBSENSOR 就不存在"切换"了。顺带 OBSENSOR 一次 grab 同时给彩色和深度。
#
# 源的写法：`"orbbec:<序列号>"`。按序列号认，不按 /dev/videoN、不按 USB 口 ——
# 两台同厂商，按厂商号会认错；插到 USB 3 口后两台的枚举顺序还会反过来。

ORBBEC_PREFIX = "orbbec:"

# 明确要走 V4L2 的写法：`"v4l2:<路径>"`，不做下面那套「Orbbec 一律改走 OBSENSOR」的改道。
# 为什么要有：腕部 Gemini 305 走 OBSENSOR 一帧不出（10-09 起未解决），走 V4L2 正常。
# RobotDecision 一直是用 V4L2 开这两台 Orbbec 的，2026-10-15 它改用本驱动后靠这个保持原样。
# 代价自己担：同一台 Orbbec 刚被 OBSENSOR 用过（比如 RobotJev），V4L2 可能一帧不出，要复位等 8 秒。
V4L2_PREFIX = "v4l2:"


def strip_v4l2(source):
    """`"v4l2:<路径>"` → (`<路径>`, True)；其他原样返回 (source, False)。"""
    if isinstance(source, str) and source.startswith(V4L2_PREFIX):
        return source[len(V4L2_PREFIX):], True
    return source, False
ORBBEC_VID = "2bc5"
# OBSENSOR 对 335 报彩色内参（深度已对齐彩色），对 305 报 nan。两台都插着时用它
# 核对「按序号打开的是不是要的那台」。产品号 -> 是否报内参。
_REPORTS_INTRINSICS = {"0800": True, "0840": False}      # 0800=Gemini 335, 0840=Gemini 305
# 原始深度一个单位是多少毫米。产品号 -> mm/单位。OBSENSOR 给的 335 深度图也是 1 mm。
_DEPTH_MM_PER_UNIT = {"0800": 1.0, "0840": 0.1}
# 低于这个帧率就当被 V4L2 污染了（污染后实测 0.3 Hz，正常 ~25 Hz）
_OBSENSOR_MIN_HZ = 10.0


def orbbec_serial(source):
    """`"orbbec:XXXX"` -> "XXXX"，不是 Orbbec 源返回 None。"""
    if isinstance(source, str) and source.startswith(ORBBEC_PREFIX):
        return source[len(ORBBEC_PREFIX):]
    return None


def orbbec_devices() -> List[Tuple[str, str, str]]:
    """USB 上所有 Orbbec：[(口号, 序列号, 产品号)]，按 USB 口路径排好序。"""
    import re  # noqa: PLC0415

    out = []
    for dev in Path("/sys/bus/usb/devices").iterdir():
        if ":" in dev.name or not (dev / "idVendor").exists():
            continue
        if (dev / "idVendor").read_text().strip() != ORBBEC_VID:
            continue
        rd = lambda f: (dev / f).read_text().strip() if (dev / f).exists() else ""  # noqa: E731
        out.append((dev.name, rd("serial"), rd("idProduct")))
    # OBSENSOR 按 **USB 口路径** 给设备编号，不是按 /dev/videoN（RobotJev 9/29 实测：
    # 第一版按 videoN 排，结果按 335 的序列号打开了 305）
    out.sort(key=lambda d: [int(x) for x in re.split(r"[-.]", d[0]) if x.isdigit()])
    return out


def orbbec_serial_of_node(source):
    """一个 V4L2 源（/dev/videoN、udev 链接、by-path、整数号）如果其实是 Orbbec，
    返回它的序列号；不是返回 None。

    用来兜底：谁要是还按路径给 Orbbec（旧 config、`--source /dev/ar5_cam_wrist`），
    照样改走 OBSENSOR，不让 V4L2 碰它一下 —— 碰一下 OBSENSOR 就塌到 0.3 Hz。
    """
    if isinstance(source, int):
        source = f"/dev/video{source}"
    if not isinstance(source, (str, os.PathLike)):
        return None
    node = Path(os.path.realpath(source)).name
    if not node.startswith("video"):
        return None
    try:
        usb = (Path("/sys/class/video4linux") / node / "device").resolve().parent
        if (usb / "idVendor").read_text().strip() != ORBBEC_VID:
            return None
        return (usb / "serial").read_text().strip() or None
    except OSError:
        return None


def orbbec_product(serial: str) -> str:
    """序列号 -> 产品号（"0800" / "0840"），USB 上没有返回 ""。"""
    return next((pid for _, sn, pid in orbbec_devices() if sn == serial), "")


def orbbec_depth_node(source):
    """一台 Orbbec 的彩色 V4L2 源 -> (同一台的深度节点, mm/单位)；不是 Orbbec 返回 None。

    深度节点 = 同一个 USB 设备上接口 `:1.0` 的第 0 个节点（Z16；第 1 个是元数据）。从彩色节点顺着
    sysfs 找，不写死 by-path —— 换 USB 口 by-path 就变，udev 链接（/dev/ar5_cam_wrist）只指向彩色。
    """
    if isinstance(source, int):
        source = f"/dev/video{source}"
    if not isinstance(source, (str, os.PathLike)):
        return None
    node = Path(os.path.realpath(source)).name
    try:
        usb = (Path("/sys/class/video4linux") / node / "device").resolve().parent
        if (usb / "idVendor").read_text().strip() != ORBBEC_VID:
            return None
        product = (usb / "idProduct").read_text().strip()
    except OSError:
        return None
    for v in sorted(Path("/sys/class/video4linux").glob("video*")):
        try:
            iface = (v / "device").resolve()
            if (iface.parent == usb and iface.name.endswith(":1.0")
                    and (v / "index").read_text().strip() == "0"):
                return f"/dev/{v.name}", _DEPTH_MM_PER_UNIT.get(product)
        except OSError:
            continue
    return None


def orbbec_nodes(serial: str) -> List[str]:
    """这台 Orbbec 名下所有 /dev/videoN（彩色、深度、元数据都算）。"""
    return sorted(f"/dev/{v.name}" for v in Path("/sys/class/video4linux").glob("video*")
                  if orbbec_serial_of_node(f"/dev/{v.name}") == serial)


def orbbec_holders(serial: str) -> List[Tuple[int, str]]:
    """谁开着这台 Orbbec。OBSENSOR 在 Linux 上底下也是开 /dev/videoN，所以查得到。"""
    seen = {}
    for node in orbbec_nodes(serial):
        for pid, cmd in who_holds(node):
            seen.setdefault(pid, cmd)
    return sorted(seen.items())


def _obsensor_index(serial: str):
    order = [sn for _, sn, _ in orbbec_devices()]
    return order.index(serial) if serial in order else None


def _reset_orbbec(serial: str, wait_s: float = 8.0) -> bool:
    """USB 复位这一台，等它重新枚举。**等待不能短**：等 3 秒彩色还是卡的，8 秒才活。"""
    port = next((port for port, sn, _ in orbbec_devices() if sn == serial), None)
    if port is None:
        return False
    try:
        from atom_robot.ar5.reset_camera_usb import reset  # noqa: PLC0415
        reset(port)
    except (ImportError, OSError, SystemExit) as exc:
        logger.warning(f"[Cameras] 复位 Orbbec {serial} 没发出去：{exc}")
        return False
    time.sleep(wait_s)
    return True


def _open_obsensor(serial: str):
    """按序列号打开一台 Orbbec，并核对打开的确实是它。返回 cv2.VideoCapture。"""
    import cv2  # noqa: PLC0415

    devs = orbbec_devices()
    idx = _obsensor_index(serial)
    if idx is None:
        raise IOError(f"USB 上找不到序列号 {serial} 的 Orbbec（现有：{devs}）")
    cap = cv2.VideoCapture(idx, cv2.CAP_OBSENSOR)
    if not cap.isOpened():
        cap.release()
        raise IOError(f"OBSENSOR 打不开序号 {idx}（序列号 {serial}）")
    product = next((pid for _, sn, pid in devs if sn == serial), "")
    want = _REPORTS_INTRINSICS.get(product)
    if len(devs) > 1 and want is not None:
        cap.grab()
        got = bool(np.isfinite(cap.get(cv2.CAP_PROP_OBSENSOR_INTRINSIC_FX)))
        if got != want:
            cap.release()
            raise IOError(
                f"OBSENSOR 序号 {idx} 打开的不是序列号 {serial} 那台（内参判据对不上）。"
                f"现有：{devs}")
    return cap


class _ObsensorGrabber(threading.Thread):
    """和 `_Grabber` 同一个接口（latest / shutdown / errors），读的是 OBSENSOR。

    `with_depth=True` 时每帧顺带取深度（uint16，单位看型号：335 是 1 mm，
    305 是 0.1 mm），`latest_depth()` 拿。不要深度就不取，省一次 retrieve。
    深度和彩色同一次 grab，序号、时刻都相同；`depth_since()` 和 `frames_since()` 一样取新来的。
    """

    def __init__(self, name: str, cap, with_depth: bool = False):
        super().__init__(name=f"ar5-cam-{name}", daemon=True)
        self.camera = name
        self._cap = cap
        self._with_depth = with_depth
        self._frame = None
        self._depth = None
        self._seq = 0
        self._stamp = None
        self._recent = deque(maxlen=RECENT_FRAMES)
        self._recent_depth = deque(maxlen=RECENT_FRAMES)
        self._lock = threading.Lock()
        self._halt = threading.Event()
        self.errors = 0
        # OBSENSOR 不给拍摄时刻（CAP_PROP_POS_MSEC 恒 -1，10-20 实测），只能记 grab() 返回那一刻 ——
        # 拍摄到这一刻之间的延迟量不到，是个固定偏差。
        self.stamp_kind = "arrival"

    def run(self) -> None:
        import cv2  # noqa: PLC0415

        while not self._halt.is_set():
            if not self._cap.grab():
                self.errors += 1
                self._halt.wait(0.02)
                continue
            arrived = time.perf_counter()      # 解码（retrieve）之前
            ok, bgr = self._cap.retrieve(None, cv2.CAP_OBSENSOR_BGR_IMAGE)
            if not ok or bgr is None:
                self.errors += 1
                continue
            depth = None
            if self._with_depth:
                ok_d, depth = self._cap.retrieve(None, cv2.CAP_OBSENSOR_DEPTH_MAP)
                if not ok_d:
                    depth = None
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            with self._lock:
                self._frame, self._depth, self._stamp = rgb, depth, arrived
                self._seq += 1
                self._recent.append((self._seq, arrived, rgb))
                if depth is not None:
                    self._recent_depth.append((self._seq, arrived, depth))

    def latest(self):
        with self._lock:
            return self._frame, self._seq

    def latest_stamped(self):
        with self._lock:
            return self._frame, self._seq, self._stamp

    def frames_since(self, seq):
        with self._lock:
            return [item for item in self._recent if seq is None or item[0] > seq]

    def latest_depth(self):
        with self._lock:
            return self._depth

    def depth_since(self, seq):
        """序号比 seq 新的深度 [(序号, 时刻, uint16 深度), ...]，序号和时刻与同一次 grab 的彩色相同。"""
        with self._lock:
            return [item for item in self._recent_depth if seq is None or item[0] > seq]

    def rate(self, window: float = 1.5) -> float:
        """量一段时间的实际帧率。"""
        _, s0 = self.latest()
        time.sleep(window)
        _, s1 = self.latest()
        return (s1 - s0) / window

    def shutdown(self, timeout: float = 2.0) -> None:
        self._halt.set()
        if self.is_alive():
            self.join(timeout=timeout)


class _V4l2DepthGrabber(threading.Thread):
    """Orbbec 的深度节点（Z16）走纯 V4L2：后台常驻 v4l2-ctl，从它的 stdout 按整帧读。

    为什么不用 OpenCV：OpenCV 的 V4L2 后端不认 Z16；腕部 305 走 OBSENSOR 一帧不出（见上）。纯 V4L2 和
    彩色同一个后端，不会互相污染（10-28 实测 335 OBSENSOR 彩色+深度、305 V4L2 彩色+深度 同开各 30 Hz）。

    时刻：`--verbose` 每帧在 stderr 打一行 `... ts: <秒> ... (ts-monotonic, ts-src-soe)`，是内核记的
    曝光开始时刻，和 perf_counter 同一个钟（10-30 核对）。按出现顺序和读到的帧一一配对；配不上
    就退回「读完这一帧那一刻」，stamp_kind 记着是哪种。

    **帧数据走单独的管道（`--stream-to=/dev/fd/3`），不走 stdout。** `--verbose` 会把 S_FMT/S_PARM
    那十几行文字打到 stdout，和帧混在一起，帧边界整个错开、读出来全是乱码（10-30 实测）。

    给的是深度传感器自己视角的原始值（uint16，单位见 mm_per_unit），**没有对齐到彩色**。
    """

    _PAIR_WAIT_S = 0.05

    def __init__(self, name: str, node: str, mm_per_unit, width: int, height: int, fps: int):
        super().__init__(name=f"ar5-depth-{name}", daemon=True)
        self.camera = name
        self.node = node
        self.mm_per_unit = mm_per_unit
        self._shape = (height, width)
        self._fps = fps
        self._proc = None
        self._data = None                   # 帧数据管道的读端
        self._depth = None
        self._seq = 0
        self._recent = deque(maxlen=RECENT_FRAMES)
        self._lock = threading.Lock()
        self._ts = {}                       # 第几帧 -> 内核时刻，stderr 线程写、取帧线程配对后删
        self._ts_cond = threading.Condition()
        self._halt = threading.Event()
        self.errors = 0
        self.stamp_kind = None

    def start(self):
        import subprocess  # noqa: PLC0415

        h, w = self._shape
        r, wr = os.pipe()
        cmd = ["v4l2-ctl", "-d", self.node, f"--set-fmt-video=width={w},height={h},pixelformat=Z16 ",
               f"--set-parm={self._fps}", "--stream-mmap=4", f"--stream-to=/dev/fd/{wr}", "--verbose"]
        try:
            self._proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                          pass_fds=(wr,), bufsize=0)
        finally:
            os.close(wr)                    # 写端只留给子进程，它一退出读端就读到 EOF
        self._data = os.fdopen(r, "rb", buffering=0)
        threading.Thread(target=self._read_stamps, name=f"ar5-depth-ts-{self.camera}", daemon=True).start()
        super().start()

    def _read_stamps(self) -> None:
        import re  # noqa: PLC0415

        pat = re.compile(rb"dqbuf:.*?\bts:\s*([0-9.]+)")
        k = 0
        for line in self._proc.stderr:
            m = pat.search(line)
            if not m:
                continue
            with self._ts_cond:
                self._ts[k] = float(m.group(1))
                k += 1
                for old in [i for i in self._ts if i < k - 4 * RECENT_FRAMES]:
                    del self._ts[old]
                self._ts_cond.notify_all()

    def run(self) -> None:
        h, w = self._shape
        n = h * w * 2
        out = self._data
        k = 0
        while not self._halt.is_set():
            buf, got = bytearray(n), 0
            while got < n:
                chunk = out.read(n - got)
                if not chunk:
                    self.errors += 1
                    return                  # v4l2-ctl 退出了（节点被拔、被占）
                buf[got:got + len(chunk)] = chunk
                got += len(chunk)
            arrived = time.perf_counter()
            depth = np.frombuffer(bytes(buf), dtype=np.uint16).reshape(h, w)
            with self._ts_cond:
                self._ts_cond.wait_for(lambda: k in self._ts, timeout=self._PAIR_WAIT_S)
                kernel = self._ts.pop(k, None)
            if kernel is not None and 0.0 <= arrived - kernel < 1.0:
                stamp, kind = kernel, "kernel"
            else:
                stamp, kind = arrived, "arrival"
            k += 1
            with self._lock:
                self._depth = depth
                self._seq += 1
                self.stamp_kind = kind
                self._recent.append((self._seq, stamp, depth))

    def latest(self):
        with self._lock:
            return self._depth, self._seq

    def latest_depth(self):
        with self._lock:
            return self._depth

    def depth_since(self, seq):
        """序号比 seq 新的深度 [(序号, 时刻, uint16 深度), ...]。"""
        with self._lock:
            return [item for item in self._recent if seq is None or item[0] > seq]

    def rate(self, window: float = 1.5) -> float:
        _, s0 = self.latest()
        time.sleep(window)
        _, s1 = self.latest()
        return (s1 - s0) / window

    def shutdown(self, timeout: float = 2.0) -> None:
        self._halt.set()
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=timeout)
            except Exception:  # noqa: BLE001
                self._proc.kill()
        if self.is_alive():
            self.join(timeout=timeout)


class OpenCVCameras:
    """Reads one frame per named source, returning RGB uint8 arrays."""

    def __init__(self, config):
        self.config = config
        self._caps: Dict[str, object] = {}
        self._grabbers: Dict[str, _Grabber] = {}
        self._last_seq: Dict[str, int] = {}
        # depth_sources 里各路的深度：OBSENSOR 那路就是彩色的 grabber 本身，V4L2 那路是单独的 _V4l2DepthGrabber。
        # depth_info[name] = mm_per_unit、是否已对齐彩色、从哪来 —— 存盘时一起写进去
        self._depth: Dict[str, object] = {}
        self.depth_info: Dict[str, dict] = {}
        # 最近一次 read() 交出去的每一帧是什么时候拍的（perf_counter 的钟），collect 逐帧记进 npz
        self.last_stamps: Dict[str, float] = {}
        # Tracked separately from _caps: configuring zero cameras is legal,
        # so an empty _caps must not read as 'never connected'.
        self._connected = False
        self.stale_reads = 0

    def connect(self) -> None:
        import cv2  # noqa: PLC0415

        if not self.config.sources:
            logger.warning("[Cameras] no sources configured; observations will carry no images")
            self._connected = True
            return

        for name, source in self.config.sources.items():
            source, forced_v4l2 = strip_v4l2(source)
            serial = None if forced_v4l2 else orbbec_serial(source)
            if serial is None and not forced_v4l2:
                serial = orbbec_serial_of_node(source)
                if serial is not None:
                    logger.warning(f"[Cameras] '{name}' 给的是路径 {source!r}，但那是 Orbbec "
                                   f"{serial}：改走 OBSENSOR（走 V4L2 会和 OBSENSOR 互相污染）")
            if serial is not None:
                try:
                    self._connect_orbbec(name, serial)
                except IOError:
                    self.disconnect()
                    raise
                continue
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                cap.release()
                self.disconnect()
                raise IOError(_open_failure_hint(name, source))

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
            cap.set(cv2.CAP_PROP_FPS, self.config.fps)
            # Two buffers, not one.
            #
            # A one-frame buffer halves the delivered rate on these cameras:
            # measured 15.5 Hz with BUFFERSIZE=1 against 29.3 Hz with 2, per
            # camera, both alone and with both streaming at once, and the same
            # either side of MJPG/YUYV. With a single buffer the driver has
            # nowhere to put the next frame between the grabber's dequeue and
            # requeue, so every other frame is dropped at the kernel.
            #
            # The staleness this was guarding against does not appear: the
            # grabber keeps only the newest frame, so the second buffer costs at
            # most one frame time and is drained immediately.
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

            # Auto-exposure is allowed to lengthen the exposure past the frame
            # period unless this is off, and then the sensor itself drops to
            # ~19 Hz in ordinary indoor light. It is a UVC control with no
            # OpenCV property, so it goes through v4l2-ctl; failure is not
            # fatal, it only costs frame rate.
            _disable_dynamic_framerate(source)

            self._caps[name] = cap
            grabber = _Grabber(name, cap)
            grabber.start()
            self._grabbers[name] = grabber
            self._last_seq[name] = 0
            logger.info(f"[Cameras] '{name}' opened from source {source!r}")

        # Wait for one frame from each before declaring the cameras up. A
        # camera that opens but never delivers is a real failure -- wrong
        # source, or a node that reports capture but carries no pixels -- and
        # it has to surface here, not as an IOError from the first read()
        # midway through a control loop.
        deadline = time.perf_counter() + 5.0
        for name, grabber in self._grabbers.items():
            if isinstance(grabber, _ObsensorGrabber):
                continue                 # 已在 _connect_orbbec 里等过首帧、量过帧率
            while grabber.latest()[0] is None:
                if time.perf_counter() > deadline:
                    source = strip_v4l2(self.config.sources[name])[0]
                    holders = who_holds(source)
                    detail = (
                        "  Held by: "
                        + "; ".join(f"PID {pid} {cmd[:60]}" for pid, cmd in holders)
                        if holders else
                        "  Nothing else holds it, so the node opens but carries no pixels -- "
                        "check `v4l2-ctl -d <node> --list-formats`."
                    )
                    self.disconnect()
                    raise IOError(
                        f"Camera '{name}' ({source!r}) opened but delivered no frame in 5 s "
                        f"({grabber.errors} failed reads).\n{detail}"
                    )
                time.sleep(0.01)

        try:
            self._connect_depth()
        except IOError:
            self.disconnect()
            raise

        # Only the zero-source path used to set this, so actually opening a
        # camera left the object looking unconnected and every read() raised
        # "used before connect()". Invisible until a camera is plugged in.
        self._connected = True

    def _connect_depth(self) -> None:
        """depth_sources 里的每一路都要出深度，出不来就报错 —— 要存深度却静默没存，比打不开更糟。

        OBSENSOR 那路（335）深度和彩色同一次 grab，已经在取，这里只等第一张；V4L2 那路（305）
        另开深度节点（同一台的 `:1.0`）。"""
        for name in tuple(getattr(self.config, "depth_sources", ()) or ()):
            grabber = self._grabbers.get(name)
            if grabber is None:
                continue                    # 这一路没开（--cameras 没点它）
            source, forced_v4l2 = strip_v4l2(self.config.sources[name])
            if isinstance(grabber, _ObsensorGrabber):
                serial = orbbec_serial(source) or orbbec_serial_of_node(source)
                depth, info = grabber, dict(mm_per_unit=_DEPTH_MM_PER_UNIT.get(orbbec_product(serial)),
                                            aligned_to_color=True, source=f"OBSENSOR {serial}")
            else:
                found = orbbec_depth_node(source)
                if found is None:
                    raise IOError(f"Camera '{name}' ({source!r}) 要深度，但它不是 Orbbec，找不到深度节点")
                node, mm = found
                depth = _V4l2DepthGrabber(name, node, mm, self.config.width, self.config.height,
                                          self.config.fps)
                depth.start()
                info = dict(mm_per_unit=mm, aligned_to_color=False, source=f"V4L2 {node}")
            self._depth[name] = depth
            deadline = time.perf_counter() + 5.0
            while depth.latest_depth() is None:
                if time.perf_counter() > deadline:
                    raise IOError(f"Camera '{name}' 的深度（{info['source']}）5 秒内没出帧。"
                                  "看是不是别的程序开着它（RobotDecision 的 depth_v4l2 之类）")
                time.sleep(0.02)
            info["stamp_kind"] = getattr(depth, "stamp_kind", None)
            self.depth_info[name] = info
            logger.info(f"[Cameras] '{name}' depth via {info['source']}, {info['mm_per_unit']} mm/unit")

    @property
    def depth_names(self) -> List[str]:
        """开着深度的那几路。"""
        return list(self._depth)

    def depth_since(self, name: str, seq):
        """这一路序号比 seq 新的深度 [(序号, 时刻, uint16 深度), ...]，缓存同 frames_since。"""
        return self._depth[name].depth_since(seq)

    def _connect_orbbec(self, name: str, serial: str) -> None:
        """打开一台 Orbbec；帧率不对就复位一次再开。

        不是每次都先复位（RobotJev 那样每次 8 秒，两台就是 16 秒）：先直接开、量 1.5 秒
        帧率，正常就用；没帧或者低于 10 Hz（被 V4L2 污染后是 0.3 Hz、不报错）才复位 +
        等 8 秒再开一次。**臂这时候已经上电但静止**，静止时 sender 不会因为没收到目标而锁死。
        """
        with_depth = name in tuple(getattr(self.config, "depth_sources", ()) or ())
        hz = 0.0
        for attempt in (1, 2):
            try:
                cap = _open_obsensor(serial)
            except IOError as exc:
                if attempt == 2:
                    raise
                logger.warning(f"[Cameras] '{name}' 打开失败（{exc}），USB 复位后等 8 秒再试一次 ...")
                _reset_orbbec(serial)
                continue
            grabber = _ObsensorGrabber(name, cap, with_depth=with_depth)
            grabber.start()
            deadline = time.perf_counter() + 5.0
            while grabber.latest()[0] is None and time.perf_counter() < deadline:
                time.sleep(0.02)
            hz = grabber.rate() if grabber.latest()[0] is not None else 0.0
            if hz >= _OBSENSOR_MIN_HZ:
                frame, _ = grabber.latest()
                if frame.shape[:2] != (self.config.height, self.config.width):
                    logger.warning(f"[Cameras] '{name}' 出的是 {frame.shape[1]}x{frame.shape[0]}，"
                                   f"config 要 {self.config.width}x{self.config.height}")
                self._caps[name] = cap
                self._grabbers[name] = grabber
                self._last_seq[name] = 0
                logger.info(f"[Cameras] '{name}' opened via OBSENSOR (serial {serial}), {hz:.1f} Hz")
                return
            grabber.shutdown()
            cap.release()
            if attempt == 1:
                logger.warning(f"[Cameras] '{name}' 只有 {hz:.1f} Hz（多半是被 V4L2 用过、"
                               "两个后端互相污染），USB 复位后等 8 秒再试一次 ...")
                _reset_orbbec(serial)
        holders = orbbec_holders(serial)
        if holders:
            raise IOError(
                f"Camera '{name}' (Orbbec {serial}) 被别的程序开着，只有 {hz:.1f} Hz：\n"
                + "\n".join(f"    PID {pid}  {cmd[:100]}" for pid, cmd in holders)
                + "\n  一台相机同一时刻只能给一个程序。先把它关掉。")
        raise IOError(
            f"Camera '{name}' (Orbbec {serial}) 复位后还是只有 {hz:.1f} Hz。\n"
            "  先确认没有别的程序开着它（RobotJev、view_camera、别的 collect），\n"
            "  还不行就把这台相机的 USB 线拔下来重插（插主机的 USB 3 口）。")

    def latest_depth(self, name: str):
        """某一路的最新深度图（uint16）。没开深度或者不是 Orbbec 返回 None。"""
        g = self._depth.get(name) or self._grabbers.get(name)
        return g.latest_depth() if hasattr(g, "latest_depth") else None

    def disconnect(self) -> None:
        for name, depth in self._depth.items():
            if depth is not self._grabbers.get(name):
                depth.shutdown()
        self._depth.clear()
        self.depth_info.clear()
        # Grabbers first: releasing a capture out from under a thread that is
        # blocked inside read() on it is a segfault, not an exception.
        for grabber in self._grabbers.values():
            grabber.shutdown()
        self._grabbers.clear()
        self._last_seq.clear()
        for cap in self._caps.values():
            cap.release()
        self._caps.clear()
        self._connected = False

    def peek(self) -> Dict[str, np.ndarray]:
        """Newest frame per camera, with no bookkeeping. Safe from any thread.

        `read()` is the control loop's accessor: it advances `_last_seq` and
        counts `stale_reads`, which is a data-quality signal (how often the loop
        saw a frame it had already recorded). A live preview pulling frames at
        its own rate would inflate that counter and corrupt the metric, so the
        preview uses this instead.

        Cameras that have not produced a frame are omitted rather than raising --
        a preview must never be the thing that takes the run down.
        """
        if not self._connected or not self._grabbers:
            return {}
        frames = {}
        for name, grabber in self._grabbers.items():
            frame, _seq = grabber.latest()
            if frame is not None:
                frames[name] = frame
        return frames

    def read(self) -> Dict[str, np.ndarray]:
        """Newest frame per camera. Does not block on the camera's frame clock."""
        if not self._connected:
            raise RuntimeError("Cameras used before connect()")
        if not self._grabbers:
            return {}

        frames, stamps = {}, {}
        for name, grabber in self._grabbers.items():
            frame, seq, stamp = grabber.latest_stamped()
            if frame is None:
                raise IOError(
                    f"Camera '{name}' has produced no frame ({grabber.errors} failed reads); "
                    "it was delivering at connect, so it has dropped off the bus since."
                )
            if seq == self._last_seq[name]:
                self.stale_reads += 1
            self._last_seq[name] = seq
            frames[name] = frame
            stamps[name] = stamp
        self.last_stamps = stamps
        return frames

    def frames_since(self, name: str, seq):
        """这一路序号比 seq 新的帧 [(序号, 时刻, RGB 帧), ...]。缓存只留 RECENT_FRAMES 帧（30 fps 下
        约 0.27 s），调用方隔得比这久就会漏帧 —— 回路卡这么久臂早 latch 了。"""
        return self._grabbers[name].frames_since(seq)

    def stamp_kinds(self) -> Dict[str, str]:
        """每一路的时刻是哪种：kernel = 驱动记的拍摄时刻，arrival = 程序拿到帧的时刻（偏晚）。"""
        return {name: g.stamp_kind for name, g in self._grabbers.items()}


class SingleCamera:
    """给一次性工具（看画面、对位）用的单路相机，用法像 `cv2.VideoCapture`：
    `read()` -> (ok, BGR 帧)、`release()`。打不开抛 IOError。

    底下就是 collect 用的 `OpenCVCameras`，所以 Orbbec 一样走 OBSENSOR、一样带
    「帧率不对就复位」。工具自己再去 `cv2.VideoCapture(路径)` 就是 V4L2，开一次
    就把 OBSENSOR 污染了，下一次 collect 只剩 0.3 Hz。
    """

    def __init__(self, source, width: int = 640, height: int = 480, fps: int = 30,
                 name: str = "camera"):
        from types import SimpleNamespace  # noqa: PLC0415

        self.name = name
        self._cams = OpenCVCameras(SimpleNamespace(
            sources={name: source}, width=width, height=height, fps=fps, depth_sources=()))
        self._cams.connect()
        self._seq = 0

    def isOpened(self) -> bool:  # noqa: N802 —— 和 cv2.VideoCapture 同名
        return self._cams._connected

    def read(self, timeout: float = 2.0):
        """等一帧**新**的再返回，和 VideoCapture.read() 一样；超时返回 (False, None)。"""
        import cv2  # noqa: PLC0415

        grabber = self._cams._grabbers.get(self.name)
        if grabber is None:
            return False, None
        deadline = time.perf_counter() + timeout
        while True:
            frame, seq = grabber.latest()
            if frame is not None and seq != self._seq:
                self._seq = seq
                return True, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if time.perf_counter() > deadline:
                return False, None
            time.sleep(0.003)

    def release(self) -> None:
        self._cams.disconnect()
