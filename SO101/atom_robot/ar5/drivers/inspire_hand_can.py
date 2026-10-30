"""Inspire RH56E over CAN -- the driver the hand on this rig actually speaks.

`inspire_hand.py` next to this file talks Modbus RTU. That was the assumption
carried over from the previous-generation RH56 manual; on this hardware it is
wrong and it does not answer. The hand is reached over CAN through a USB-CAN
adapter (Wuhan Ruobite) that enumerates as a CH340 serial port, so the port in
`HandConfig` is a `/dev/ttyUSB*` node and `baudrate` is the adapter's UART rate,
not the CAN rate.

Rather than reimplement the CAN codec here, this wraps the driver in
`VEL/src/hand_teleop/`, which is byte-for-byte verified against captures from
this exact hand (frame escaping included -- see `docs/inspire-can-protocol.md`).
This file is only the adapter between two interfaces:

    AR5 HandDriver          VEL InspireHand
    opening in [0, 1]  <->  index-finger angle in [finger_close, finger_open]
    0 = closed              356  (14 counts past the measured contact point 370)
    1 = open                1000

Only the index finger moves. The thumb is parked as a fixed backstop and the
other three fingers are curled out of the way, which is what makes a one-scalar
opening meaningful on a five-finger hand.
"""

import logging
import sys
import threading
from pathlib import Path

import numpy as np

from .base import HandDriver

logger = logging.getLogger(__name__)

# The verified CAN driver lives outside the openvla tree, in the hand repo.
# 2026-09-23 代码从 VEL 仓库搬到 Project_Atom，按目录层数猜根目录那套就错了 ——
# 而且错得**没有报错**，只是路径不存在、import 静默失败或者文档静默不检查。
# 改成走 atom_robot.VEL_REPO（可用环境变量 VEL_REPO 覆盖）。
from atom_robot import VEL_REPO as _VEL_REPO
_VEL_HAND = _VEL_REPO / "VEL" / "src" / "hand_teleop"


def _import_vel_driver():
    """Import the VEL hand driver, keeping sys.path edits local to this call."""
    if not _VEL_HAND.is_dir():
        raise RuntimeError(
            f"VEL hand driver not found at {_VEL_HAND}. It carries the verified CAN "
            "codec; clone github.com/Jiabo-Wang/VEL next to openvla-energy."
        )
    path = str(_VEL_HAND)
    if path not in sys.path:
        sys.path.insert(0, path)
    from inspire_hand import InspireHand  # noqa: PLC0415
    from inspire_hand import registers as reg  # noqa: PLC0415

    return InspireHand, reg


class InspireHandCan(HandDriver):
    """Index-finger opening control for an RH56E on CAN.

    The CAN I/O runs on its own thread. That is not a throughput optimisation --
    it is what keeps the hand from taking the arm down. A write is normally 13 ms,
    but a dropped frame costs a retry cycle: measured p99 219 ms, worst case
    (retries+1) x (2 x timeout) = 615 ms. `AR5Env.step_joints` drives the hand
    before it publishes the arm's goal, and the arm's 1 kHz sender brakes and
    latches after 250 ms without a goal. So one dropped hand frame used to stop
    the arm dead, mid-motion, until the operator re-engaged.

    Commands are therefore posted, not written: `set_opening` stores a target and
    returns immediately, and the worker writes the most recent one. Skipping
    superseded targets is correct here -- this is position control, and an
    opening that has already been overtaken has nothing to contribute.
    """

    def __init__(self, config):
        self.config = config
        self._hand = None
        self._reg = None
        # Last commanded opening. get_opening() prefers a real readback and only
        # falls back to this, so a transient bus error cannot stall the loop.
        self._opening = 1.0

        self._lock = threading.Lock()
        self._target = None            # posted opening the worker has yet to write
        self._measured = None          # last good readback, refreshed by the worker
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._worker = None
        self._io_errors = 0

    # ── lifecycle ───────────────────────────────────────────────────────────
    @staticmethod
    def _check_port(port: str) -> None:
        """开口之前先说清楚这个口是不是手的。

        2026-09-23：`port` 写死 `/dev/ttyUSB0` 害过一次。灵巧手换成夹爪之后，
        夹爪的 Prolific 转 485 占了 ttyUSB0，手的驱动照旧去开它 —— **开得成，
        但一个字节都收不到**，报的是「未扫到合法帧 (0 字节): 空」，看起来像手坏了
        或者 CAN_H/CAN_L 接反。真相是根本没跟手说话。

        所以这里在开口之前判两件事，各自给各自的话：口不存在（适配器没插）、
        口存在但属于别的设备（多半是夹爪）。判不出来就放行 —— 这是提示，不是闸。
        """
        import os  # noqa: PLC0415

        if not os.path.exists(port):
            raise RuntimeError(
                f"手的串口不存在：{port}\n"
                "  手的 USB-CAN 适配器没插，或者换了口。确认：\n"
                "    ls -l /dev/serial/by-id/        # 手是 usb-1a86_USB_Serial-if00-port0\n"
                "    lsusb | grep 1a86               # CH340 应该在\n"
                "  **手已经拆掉、现在装的是夹爪的话，加 `--end-effector hkvr`** ——\n"
                "  config 里 end_effector 的默认值还是 inspire。"
            )
        try:
            real = os.path.realpath(port)
            others = {}
            by_id = "/dev/serial/by-id"
            if os.path.isdir(by_id):
                for name in os.listdir(by_id):
                    target = os.path.realpath(os.path.join(by_id, name))
                    others.setdefault(target, []).append(name)
            names = others.get(real, [])
            # 同一个节点上挂着 Prolific 的名字 = 这是夹爪那根线，不是手的
            if any("Prolific" in n for n in names):
                raise RuntimeError(
                    f"{port} 指向 {real}，而那个节点是**夹爪**的 RS485 转换器"
                    f"（{', '.join(names)}），不是手的 USB-CAN。\n"
                    "  ttyUSBn 按插入顺序给，插拔一次就换位 —— 所以这里要写 by-id 路径。\n"
                    "  现在装的是夹爪的话，加 `--end-effector hkvr`。"
                )
        except OSError:
            pass                    # 判不出来就放行，别让一个提示挡住正常路径

    def connect(self) -> None:
        InspireHand, reg = _import_vel_driver()
        self._reg = reg

        cfg = self.config
        self._check_port(cfg.port)
        if cfg.index_dof >= reg.NUM_DOF:
            raise ValueError(f"index_dof {cfg.index_dof} outside 0..{reg.NUM_DOF - 1}")
        if len(cfg.fixed_pose) != reg.NUM_DOF:
            raise ValueError(f"fixed_pose needs {reg.NUM_DOF} entries, got {len(cfg.fixed_pose)}")

        self._hand = InspireHand(
            port=cfg.port,
            baudrate=cfg.baudrate,
            dev_id=cfg.dev_id,
            protocol=cfg.protocol,
            timeout=cfg.timeout,
        )
        try:
            self._hand.connect()
        except Exception as exc:
            self._hand = None
            raise RuntimeError(
                f"Could not open the hand on {cfg.port}: {exc}. Check the USB-CAN adapter "
                "is plugged in (lsusb | grep 1a86 -> 7523) and that the hand has its own supply."
            ) from exc

        # Prove the link before anything moves. A CAN read that comes back with
        # six plausible angles is the only evidence the protocol and wiring are
        # both right -- CAN_H/CAN_L swapped reads as silence, not as an error.
        try:
            angles = self._hand.get_angles()
        except Exception as exc:
            self.disconnect()
            raise RuntimeError(
                f"Hand did not answer on {cfg.port}: {exc}. If the port is right, check "
                "CAN_H/CAN_L are not swapped -- reversed wiring reads as no data, not an error."
            ) from exc
        logger.info("[InspireHandCan] %s @%d id=%d angles=%s", cfg.port, cfg.baudrate, cfg.dev_id, angles)

        self._arm_limits()

        self._measured = list(angles)
        self._stop.clear()
        self._worker = threading.Thread(target=self._io_loop, name="ar5-hand-io", daemon=True)
        self._worker.start()

    def _io_loop(self) -> None:
        """Write posted targets and refresh the readback, off the caller's thread."""
        while not self._stop.is_set():
            self._wake.wait(0.05)
            self._wake.clear()
            if self._stop.is_set():
                break

            with self._lock:
                target = self._target
                self._target = None

            try:
                if target is not None:
                    self._hand.set_angles(list(self.config.pose_for(target)))
                # Read on every pass, not just after a write. A write returns as
                # soon as the hand acknowledges it, while the finger takes a
                # second or two to actually travel -- so reading only after a
                # write captures the start of the motion and then never updates.
                # get_opening() served that stale value, which would land in the
                # recorded proprio as "the hand never moved".
                measured = self._hand.get_angles()
            except Exception as exc:  # noqa: BLE001
                # A dropped frame is normal on this bus and the next write
                # supersedes this one anyway. Never raise out of the worker:
                # that would kill the thread and silently freeze the hand.
                self._io_errors += 1
                if self._io_errors in (1, 10, 100) or self._io_errors % 500 == 0:
                    logger.warning("[InspireHandCan] hand I/O error #%d: %s", self._io_errors, exc)
                continue
            with self._lock:
                self._measured = measured

    def _arm_limits(self) -> None:
        """Force and speed are NOT retained across a power cycle -- the hand comes
        back at the factory 2000 g / 1000. Anything that moves fingers must set
        them first, or the index finger drives into the thumb at full force."""
        self._hand.set_force(int(self.config.force))
        self._hand.set_speed(int(self.config.speed))
        logger.info("[InspireHandCan] force=%dg speed=%d (not retained across power loss)",
                    self.config.force, self.config.speed)

    def disconnect(self) -> None:
        self._stop.set()
        self._wake.set()
        worker, self._worker = self._worker, None
        if worker is not None:
            # Bounded: a write already in flight can be mid-retry for a while,
            # and hanging the shutdown on it helps nobody.
            worker.join(timeout=2.0)
        if self._hand is not None:
            try:
                self._hand.close()
            finally:
                self._hand = None

    # ── control ─────────────────────────────────────────────────────────────
    def _pose_for(self, opening: float):
        """这个开合度对应的六维姿态。食指和拇指都会动。"""
        return self.config.pose_for(opening)

    def _angle_for(self, opening: float) -> int:
        """这个开合度下食指的角度。拇指的看 `_pose_for`。"""
        return self.config.pose_for(opening)[self.config.index_dof]

    def set_opening(self, opening: float) -> None:
        """Post an opening. Returns immediately -- the worker does the CAN write.

        Called from the arm's control loop, so it must not block: a synchronous
        write here is 13 ms normally but 219 ms at p99, and 250 ms without an arm
        goal makes the arm's sender brake and latch.
        """
        if self._hand is None:
            raise RuntimeError("Hand driver used before connect()")
        value = float(np.clip(opening, 0.0, 1.0))
        with self._lock:
            self._target = value
        self._opening = value
        self._wake.set()

    def get_opening(self) -> float:
        """Measured opening, from the index finger's last known angle.

        Served from what the worker last read, never by reading the bus here:
        the caller is the arm's control loop. Falls back to the last commanded
        value until a readback exists. Clipped to [0, 1] because the finger
        resting against the thumb sits slightly above `finger_close`.
        """
        if self._hand is None:
            raise RuntimeError("Hand driver used before connect()")
        cfg = self.config
        with self._lock:
            measured = self._measured
        if measured is None:
            return self._opening
        span = cfg.finger_open - cfg.finger_close
        if span == 0:
            return self._opening
        return float(np.clip((measured[cfg.index_dof] - cfg.finger_close) / span, 0.0, 1.0))

    def get_joint_angles(self) -> np.ndarray:
        """Last known six-DOF angles. Also served from the worker's readback."""
        if self._hand is None:
            raise RuntimeError("Hand driver used before connect()")
        with self._lock:
            measured = self._measured
        if measured is None:
            return np.asarray(self.config.fixed_pose, dtype=np.float64)
        return np.asarray(measured, dtype=np.float64)

    # ── diagnostics ─────────────────────────────────────────────────────────
    def diagnose(self) -> str:
        if self._hand is None:
            return "[InspireHandCan] not connected"
        return self._hand.diagnose()
