"""OmniPicker 驱动：自适应模式（位置 + 力上限），串口 I/O 在独立线程，不拖臂的回路。

接口和 atom_robot 的 `HandDriver` 一样（connect / disconnect / set_opening /
get_opening / get_joint_angles / check_health），另外多了调夹持力的几个方法。
opening 0 = 合拢、1 = 位置 `open_max`（换算基准）；张开目标另由 `open_limit` 封顶，
所以 set_opening(1) 实际只开到 open_limit，读回来的开度约 0.86。

**自适应**的意思：给「合到 0」它就合到碰上东西为止，然后按夹持力一直夹着，
不会越夹越紧；夹手和空夹的力矩读数一样（2026-10-19 实测）。所以主臂扳机捏到底
就是「抓」，夹多紧由夹持力决定，不由扳机决定。

这一层守的规矩（都是 2026-10-19 / 20 在实物上踩出来的，见 README）：

* 张开目标不超过 `open_limit`（180），张开方向的力不低于 `open_force`（64）——
  两者任一不守，都可能冲过头、卡在张开端或冲过 255 回绕，只能断电重启。
  （上限原来是 240，从夹紧状态松开时冲过头回绕过一次，10-20 实测后降到 210；
  10-30 在 210 又冲过头、卡在张开端两次，降到 180。）
* 位置一个周期内跳变超过 100 判回绕：停发、报错、要求断电重启。物理上全速一个
  周期（20 ms）最多走 6 个单位。
* 「移动中」、离目标还远、位置却不动，持续 `stuck_seconds` 判卡死，停发报错。
* 错误码非 0（过温、超速……）立刻停发报错。
* 只在目标变化超过 `deadband` 或夹持力变了才重发，不每周期重复下同一个目标。
* 连上时只读，不发任何运动指令；退出也不自动松开。
"""
import math
import threading
import time
from collections import deque

import numpy as np

from .config import OmniPickerConfig
from .protocol import (FAULTS, STATE_MOVING, STATES, ModbusError, OmniPickerBus)

MIN_FIRMWARE = 335           # 3.3.5 才有 Modbus RTU
WRAP_JUMP = 100              # 一个周期内位置跳变超过它就是回绕，不是运动


class SafetyStop(RuntimeError):
    """夹爪自己报错、位置回绕或卡死 —— 不重试，停发。"""


def _open_bus(cfg):
    return OmniPickerBus(cfg.port, cfg.addr, cfg.baudrate, cfg.timeout)


class OmniPickerGripper:
    def __init__(self, config: OmniPickerConfig = None, bus_factory=_open_bus):
        self.config = config or OmniPickerConfig()
        self._bus_factory = bus_factory
        self._bus = None
        self._worker = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._target = None
        self._grip_force = int(self.config.grip_force)
        self._speed = int(self.config.speed)
        self._status = None
        self._status_at = 0.0
        # 这次状态是什么时候读的（time.perf_counter 的钟，和相机、臂同一个）：发请求和收到回包的中点。
        # 夹爪大约在这两者之间采的位置；一次读写约 7 ms，误差在 ±4 ms 以内。
        self._status_mid = None
        # 最近的读数 (时刻, 位置)，OmniPicker 采集每圈取走新来的（100 Hz 下约 1.3 s）
        self._history = deque(maxlen=128)
        self._error = None
        self.firmware = None

    # ── 连接 ─────────────────────────────────────────────────────────

    def connect(self):
        if self._bus is not None:
            raise RuntimeError("夹爪已经连接")
        cfg = self.config
        try:
            self._bus = self._bus_factory(cfg)
        except OSError as exc:
            raise IOError(f"打不开 {cfg.port}：{exc}。转换器插上了吗？口被别的程序占着吗？") from exc
        try:
            try:
                self.firmware = self._bus.firmware()
                status = self._bus.status()
            except ModbusError as exc:
                raise IOError(
                    f"OmniPicker 没有回应（{exc}）。依次查：24V；转换器 A/B —— hkvtech 这块要反接，"
                    "夹爪 H 接 B、L 接 A；夹爪后盖拨码在 485（1、2 号 ON）；"
                    f"上位机里 rs485_mode=1、modbus_mode=True、地址 {cfg.addr}、波特率 {cfg.baudrate}") from exc
            if self.firmware < MIN_FIRMWARE:
                raise IOError(f"夹爪固件 {self.firmware} 低于 {MIN_FIRMWARE}，没有 Modbus RTU")
            if status["fault"]:
                raise IOError(f"夹爪报错 {FAULTS.get(status['fault'], status['fault'])}，"
                              "先断电重启（上电会自己合拢找零位，指间别放东西）")
            with self._lock:
                self._status, self._status_at = status, time.monotonic()
                self._error, self._target = None, None
            self._stop.clear()
            self._worker = threading.Thread(target=self._io_loop, name="omnipicker-io", daemon=True)
            self._worker.start()
        except BaseException:
            self.disconnect()
            raise

    def disconnect(self):
        self._stop.set()
        if self._worker is not None:
            self._worker.join(timeout=2.0)
            if self._worker.is_alive():
                raise RuntimeError("OmniPicker 串口线程未退出")
            self._worker = None
        if self._bus is not None:
            self._bus.close()
            self._bus = None
        # 退出不自动松开：夹着东西时松开会把它摔了。已下发的运动会继续走到目标。

    # ── I/O 线程 ─────────────────────────────────────────────────────

    def _io_loop(self):
        cfg = self.config
        period = 1.0 / cfg.hz
        # 单次读写失败不算断线：臂的 1 kHz 线程会间歇饿住这个线程（xCoreSDK 不放
        # GIL），串口读就超时。真断线靠 stale_timeout 由上层发现。和 HKVR 驱动一个道理。
        budget = max(3, int(cfg.stale_timeout * cfg.hz) - 1)
        misses = 0
        sent_target = sent_force = None
        prev_pos = None
        stuck_since = stuck_pos = None
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                with self._lock:
                    target, grip_force, speed = self._target, self._grip_force, self._speed
                    pos_now = self._status["pos"]
                # 速度变了不重发：它只影响下一次运动，重发会让正在走的这一段重新规划
                if target is not None and (
                        sent_target is None or abs(target - sent_target) >= cfg.deadband
                        or grip_force != sent_force):
                    opening = target > pos_now + cfg.deadband
                    force = max(cfg.open_force, grip_force) if opening else grip_force
                    self._bus.command(target, speed, force, cfg.acc, cfg.dec)
                    sent_target, sent_force = target, grip_force
                    stuck_since = None
                asked = time.perf_counter()
                status = self._bus.status()
                mid = (asked + time.perf_counter()) / 2
                now = time.monotonic()
                misses = 0

                if status["fault"]:
                    raise SafetyStop(f"夹爪报错 {FAULTS.get(status['fault'], status['fault'])}")
                if prev_pos is not None and abs(status["pos"] - prev_pos) > WRAP_JUMP:
                    raise SafetyStop(
                        f"位置一个周期从 {prev_pos} 跳到 {status['pos']}，是冲过 255 回绕了。"
                        "固件内部也跟着乱，软件拉不回来 —— **断电重启夹爪**（上电会自己合拢找零位）")
                prev_pos = status["pos"]

                far = sent_target is not None and abs(status["pos"] - sent_target) > 3
                if status["state"] == STATE_MOVING and far:
                    if stuck_since is None or abs(status["pos"] - stuck_pos) > 1:
                        stuck_since, stuck_pos = now, status["pos"]
                    elif now - stuck_since > cfg.stuck_seconds:
                        raise SafetyStop(
                            f"夹爪显示「移动中」，位置却停在 {status['pos']} 超过 {cfg.stuck_seconds}s"
                            f"（目标 {sent_target}）—— 像是机构卡住了，已停发")
                else:
                    stuck_since = None

                with self._lock:
                    self._status, self._status_at, self._status_mid = status, now, mid
                    self._history.append((mid, status["pos"]))
            except SafetyStop as exc:
                with self._lock:
                    self._error, self._target = exc, None
                return
            except Exception as exc:                        # noqa: BLE001
                misses += 1
                if misses < budget:
                    sent_target = None          # 下一轮重发目标，别停在半路
                    self._stop.wait(max(0.0, period - (time.monotonic() - started)))
                    continue
                with self._lock:
                    self._error, self._target = exc, None
                return
            self._stop.wait(max(0.0, period - (time.monotonic() - started)))

    # ── HandDriver 接口 ──────────────────────────────────────────────

    def check_health(self):
        with self._lock:
            error, stamp = self._error, self._status_at
        if error is not None:
            raise RuntimeError(f"OmniPicker 已停止发送：{error}") from error
        if self._bus is None or time.monotonic() - stamp > self.config.stale_timeout:
            raise RuntimeError("OmniPicker 反馈过期或尚未连接")

    def set_opening(self, opening):
        value = float(opening)
        if not math.isfinite(value):
            raise ValueError("夹爪开合必须为有限数")
        self.check_health()
        target = self.target_position(value)
        with self._lock:
            self._target = target

    def target_position(self, opening) -> int:
        """开度 0~1 实际会下发的位置：按 open_max 换算，再按 open_limit 封顶。"""
        cfg = self.config
        return min(round(max(0.0, min(1.0, float(opening))) * cfg.open_max), cfg.open_limit)

    def get_opening(self):
        self.check_health()
        with self._lock:
            pos = self._status["pos"]
        return max(0.0, min(1.0, pos / self.config.open_max))

    def get_opening_stamped(self):
        """(开合, 这个读数是什么时候读的)，一起取，免得中间换了一次状态对不上。"""
        self.check_health()
        with self._lock:
            pos, mid = self._status["pos"], self._status_mid
        return max(0.0, min(1.0, pos / self.config.open_max)), mid

    def openings_since(self, t):
        """读取时刻比 t 晚的读数 [(时刻, 开合 0~1), ...]；t 为 None 给缓存里全部。"""
        self.check_health()
        top = self.config.open_max
        with self._lock:
            return [(m, max(0.0, min(1.0, pos / top))) for m, pos in self._history if t is None or m > t]

    def get_joint_angles(self):
        self.check_health()
        with self._lock:
            return np.array([self._status["pos"]], dtype=np.float64)

    # ── 夹持力 ───────────────────────────────────────────────────────

    @property
    def grip_force(self) -> int:
        with self._lock:
            return self._grip_force

    def set_grip_force(self, value) -> int:
        """改夹持力，夹在 [grip_force_min, grip_force_max]。下一个 I/O 周期生效：
        已经夹着东西的话会用同一个目标、新的力重发一次。返回实际生效的值。"""
        cfg = self.config
        value = max(cfg.grip_force_min, min(cfg.grip_force_max, int(value)))
        with self._lock:
            self._grip_force = value
        return value

    def step_grip_force(self, steps: int) -> int:
        return self.set_grip_force(self.grip_force + steps * self.config.grip_force_step)

    # ── 速度 ─────────────────────────────────────────────────────────

    @property
    def speed(self) -> int:
        with self._lock:
            return self._speed

    def set_speed(self, value) -> int:
        """改开合速度，夹在 [speed_min, speed_max]。从下一条运动指令开始生效，
        不打断正在走的这一段。返回实际生效的值。"""
        cfg = self.config
        value = max(cfg.speed_min, min(cfg.speed_max, int(value)))
        with self._lock:
            self._speed = value
        return value

    def step_speed(self, steps: int) -> int:
        """按 config.speed_levels() 的档位走 steps 档。当前值不在档位上就先对到最近的一档。"""
        levels = self.config.speed_levels()
        now = self.speed
        index = min(range(len(levels)), key=lambda i: abs(levels[i] - now))
        index = max(0, min(len(levels) - 1, index + int(steps)))
        return self.set_speed(levels[index])

    def last_status(self) -> dict:
        """最近一次读到的状态，外加中文状态名。给界面用，不做健康检查。"""
        with self._lock:
            status = dict(self._status or {})
        if status:
            status["state_name"] = STATES.get(status["state"], str(status["state"]))
        return status
