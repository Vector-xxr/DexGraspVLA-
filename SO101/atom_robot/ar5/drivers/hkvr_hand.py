"""HKVR-TG9801 RS485 / Ruobite CAN; serial I/O stays off the arm loop."""
import math
import threading
import time

import numpy as np

from .base import HandDriver
from .hkvr_protocol import HKVRGripper, MODE_FORCE_POS


class HKVRHand(HandDriver):
    def __init__(self, config):
        self.config = config
        self._device = None
        self._worker = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._target = None
        self._measured = None
        self._measured_at = 0.0
        self._error = None
        self._pending_force = None      # 运行中改力/速度：只登记，由串口线程在下次下发前写（见 set_grip_params）
        self._pending_speed = None
        self._force_readback = None

    def connect(self):
        cfg = self.config
        if self._device is not None:
            raise RuntimeError("夹爪已经连接")
        self._device = HKVRGripper(port=cfg.port, baud=cfg.baudrate,
                                  slave=cfg.slave_id, timeout=cfg.timeout,
                                  speed=cfg.speed, transport=cfg.transport,
                                  can_id=cfg.can_id, can_rx_id=cfg.can_rx_id)
        try:
            self._device.connect()
            if not self._device.set_mode(MODE_FORCE_POS):
                raise IOError("夹爪切力位模式未确认")
            if not self._device.set_force(cfg.force):
                raise IOError("夹爪设力未确认")
            # 初始化会走全行程，只在独立诊断命令显式执行；遥操启动不自动初始化。
            self._read_feedback()
            self._error = None
            self._target = None
            self._stop.clear()
            self._worker = threading.Thread(target=self._io_loop, name="hkvr-io", daemon=True)
            self._worker.start()
        except BaseException:
            self.disconnect()
            raise

    # 位置有效区间。上界不是量程本身，是量程 + 余量：闭合端会漂（见
    # hkvr_protocol 里 POS_CLOSED 那段），实测见过 1046。判死在 1000 的话，
    # 夹爪合到底时就会被判成"无效反馈" —— 2026-09-24 就是这么让 collect
    # 起不来的，而那句话读起来像串口断了，实际上通信好得很。
    POS_VALID_LO, POS_VALID_HI = -50, 1150

    def _read_feedback(self):
        info = self._device.info()
        if info is None:
            raise IOError("HKVR 读不到位置 —— 没有回包。检查 RS485 接线（A/B 别反）"
                          "、24V 供电、以及口是不是被别的进程占着")
        pos = info[0]
        if not self.POS_VALID_LO <= pos <= self.POS_VALID_HI:
            raise IOError(
                f"HKVR 位置读数 {pos} 超出有效区间 "
                f"[{self.POS_VALID_LO}, {self.POS_VALID_HI}] —— **通信是好的**，"
                "是这个数不对。多半是夹爪卡住或者位置寄存器跑飞了，"
                "跑一次 `jog_hkvr --open` 看它动不动")
        with self._lock:
            self._measured = info[0] / 1000.0
            self._measured_at = time.monotonic()

    # 2026-09-24 这里加过一版「目标没变也每 0.4 秒重发一次，直到走到位」，**已撤掉**。
    #
    # 那一版有量纲错误：这个类内部的单位是 `原始位置/1000`（`_measured = raw/1000`，
    # `set_opening` 存的目标是 `opening*1000` 直接当原始位置发），而到位判据用了
    # `1 - raw/1000`。结果全开、全闭两端**永远判成没到位**，闭合指令每 0.4 秒重发
    # 一次停不下来 —— 用户报「夹爪不合上了」。而且加的时候串口被占着，**没上真机验**
    # 就改了硬件行为。
    #
    # 就算量纲改对，「反复下同一个目标」对这台固件是什么效果也没验过（会不会每次都
    # 重启一遍抓取流程），所以回到原来「目标变了才发」，这是一直跑通的写法。

    def _io_loop(self):
        sent = None
        misses = 0
        # 漏几次才算故障。**单次漏读不是通信故障**：臂的实时发送线程是 1 kHz，
        # 而 xCoreSDK 的 Python 绑定在 C++ 调用期间不放 GIL，这个 30 Hz 的线程
        # 会被间歇饿住，串口读就会超时。原来漏一次就抛错停发，结果是遥操刚起步
        # （steps 1）就死在「HKVR 无有效位置反馈」，而夹爪本身好好的 ——
        # 单独压测 180 次零失败，一开臂就挂。
        #
        # 真正的断线判据用已有的 stale_timeout：反馈超过它没更新，上层自己会报。
        # 这里只保证「一直在试」，并且把上一次的好值留着不清掉。
        budget = max(3, int(self.config.stale_timeout * self.config.hz) - 1)
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                with self._lock:
                    target = self._target
                    self._target = None
                    pf, ps = self._pending_force, self._pending_speed
                    self._pending_force = self._pending_speed = None
                if pf is not None:
                    if not self._device.set_force(pf):
                        raise IOError("HKVR 设力未收到确认")
                    rb = self._device.target_force()
                    with self._lock:
                        self._force_readback = rb
                if ps is not None:
                    self._device.speed = int(ps)
                    sent = None                  # 速度跟着下一次位置指令一起写，强制重发
                if target is not None and target != sent:
                    if not self._device.move(target):
                        raise IOError("HKVR 目标位置未收到确认")
                    sent = target
                self._read_feedback()
                misses = 0
            except Exception as exc:
                misses += 1
                if misses < budget:
                    sent = None          # 重发一次目标，别让它停在半路
                    self._stop.wait(max(0, 1.0 / self.config.hz
                                        - (time.monotonic() - started)))
                    continue
                with self._lock:
                    self._error = exc
                    self._target = None
                # 断线后不自动恢复动作；上层下一周期收到错误并走原有停止/清理路径。
                return
            self._stop.wait(max(0, 1.0 / self.config.hz - (time.monotonic() - started)))

    def check_health(self):
        with self._lock:
            error, stamp = self._error, self._measured_at
        if error is not None:
            raise RuntimeError(f"夹爪通信故障，已停止发送：{error}") from error
        if self._device is None or time.monotonic() - stamp > self.config.stale_timeout:
            raise RuntimeError("夹爪反馈过期或尚未连接")

    def set_opening(self, opening):
        value = float(opening)
        if not math.isfinite(value):
            raise ValueError("夹爪开合必须为有限数")
        self.check_health()
        with self._lock:
            self._target = round(max(0.0, min(1.0, value)) * 1000)

    def set_grip_params(self, force=None, speed=None):
        """运行中改目标夹取力（1~100）和速度（200~1500）。**只登记**，由串口线程在下一次
        下发位置之前写入 —— 串口只归那个线程，主线程直接调 set_force 会和它抢着写、帧交错。
        读回的力在 force_readback() 里。"""
        self.check_health()
        with self._lock:
            if force is not None:
                self._pending_force = max(1, min(100, int(force)))
                self._force_readback = None
            if speed is not None:
                self._pending_speed = max(200, min(1500, int(speed)))

    def force_readback(self):
        with self._lock:
            return self._force_readback

    def get_opening(self):
        self.check_health()
        with self._lock:
            return self._measured

    def get_joint_angles(self):
        return np.array([self.get_opening() * 1000], dtype=np.float64)

    def disconnect(self):
        self._stop.set()
        if self._worker is not None:
            self._worker.join(timeout=2.0)
            if self._worker.is_alive():
                raise RuntimeError("HKVR 串口线程未退出")
            self._worker = None
        if self._device is not None:
            self._device.close()
            self._device = None
        # 退出不自动松夹；已下发的夹爪内部运动可能继续到目标。
