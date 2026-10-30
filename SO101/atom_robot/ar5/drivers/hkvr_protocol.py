#!/usr/bin/env python3
"""航凯 TG9801 协议（厂家手册 Ver.1.0.0.2，第 4.5/4.6 节）。

直连默认 RS485 Modbus RTU，1000000 baud、8N1、从站 1。
CAN 封装保留用于已有适配盒；不能把因时 USB-CAN 直接当 USB-RS485。
位置 0..1000：0 闭合、1000 张开。遥操线程适配见 hkvr_hand.py。
"""
from __future__ import annotations

import time
import math

# ── 适配器层 ────────────────────────────────────────────────────────────────
HEAD, TAIL = b"\xAA\xAA", b"\x55\x55"
ESC, SPECIAL = 0xA5, (0x55, 0xAA, 0xA5)


def _escape(body: bytes) -> bytes:
    out = bytearray()
    for b in body:
        if b in SPECIAL:
            out.append(ESC)
        out.append(b)
    return bytes(out)


def _unescape(body: bytes) -> bytes:
    out, esc = bytearray(), False
    for b in body:
        if not esc and b == ESC:
            esc = True
            continue
        out.append(b)
        esc = False
    return bytes(out)


def build_frame(can_id: int, data: bytes, dlc: int) -> bytes:
    """裹成适配器认的串口帧。data 不足 8 字节右补零。"""
    payload = bytes(data[:8]).ljust(8, b"\x00")
    body = can_id.to_bytes(4, "little") + payload + bytes((dlc, 0x00, 0x01, 0x00))
    cks = sum(body) & 0xFF                      # body 此时正好是 frame[2:18]
    return HEAD + _escape(body + bytes((cks,))) + TAIL


def parse_frames(buf: bytearray):
    """从缓冲里切出所有完整帧，返回 [(can_id, data, dlc, 校验对不对)]，并就地消费。"""
    out = []
    while True:
        i = buf.find(HEAD)
        if i < 0:
            del buf[:-1]
            break
        # 尾标记必须未被转义，尤其校验字节恰好为 0x55 时。
        j = -1
        cursor = i + 2
        while cursor < len(buf) - 1:
            if buf[cursor] == ESC:
                cursor += 2
            elif buf[cursor:cursor + 2] == TAIL:
                j = cursor
                break
            else:
                cursor += 1
        if j < 0:
            if i:
                del buf[:i]
            break
        body = _unescape(bytes(buf[i + 2:j]))
        del buf[:j + 2]
        if len(body) != 17 or not 0 <= body[12] <= 8:
            continue
        ok = (sum(body[:16]) & 0xFF) == body[16]
        out.append((int.from_bytes(body[0:4], "little"), bytes(body[4:12]), body[12], ok))
    return out


# ── 夹爪层 ──────────────────────────────────────────────────────────────────
CMD_GRASP_ADAPTIVE = 0x01      # 自适应抓放
CMD_SPEED_ADAPTIVE = 0x02      # 自适应设速度
CMD_INFO           = 0x03      # 读位置 + 状态
CMD_INIT           = 0x04      # 指尖初始化
CMD_FORCE_SET      = 0x06      # 设目标夹取力
CMD_FORCE_GET      = 0x07      # 读目标夹取力
CMD_DEV_ID         = 0x08      # 改设备 ID
CMD_MODE           = 0x0E      # 切模式 0=自适应 1=力位
CMD_FORCE_POS      = 0x10      # 力位模式抓放：速度 + 目标位置
CMD_EEPROM         = 0x11      # EEPROM 写使能
CMD_SERIAL         = 0xEE      # 出厂编号（分包返回）

MODE_ADAPTIVE, MODE_FORCE_POS = 0x00, 0x01

# **方向和量程都是实测的，和手册不一样。** 2026-09-22 力位模式扫全程：
#
#     指令    0 → 位置   0     ← 实物确认这一端是**张开**
#     指令  250 → 位置 249
#     指令  500 → 位置 499 / 501（来回两次）
#     指令  750 → 位置 743
#     指令 1000 → 位置 858     ← 到顶了，再给也不动
#
# 所以：**位置 0 = 全开，数值越大夹得越紧**（合拢端当时量到 858，9-24 查明是卡住时量的，现在用 1000，见下）。
# 手册只说量程 0~1000、没说哪端是合拢，这两点都要靠实测。
#
# （9-22 当时的理由，已作废）量程上限用 858 不用 1000：上面那 14% 根本没有行程，按 1000 归一化的话
# opening 在 0~0.14 之间怎么给都一样，策略学到的"合拢"其实合不到底。
# 位置量程。**闭合端 2026-09-24 从 858 改成 1000。**
#
# 858 是 2026-09-22 实测的，但**那天夹爪是卡住的** —— 合到 858 就走不动了，
# 于是那个数被当成了闭合端。解卡之后重量：空载来回走三轮，闭合端停在
# 907 / 1010 / 995 / 995，而手册写的位置范围正是 0~1000。也就是说手册没错，
# 错的是在卡住状态下量的那一次。
#
# ⚠ **闭合端本身会漂 ±50 左右**（上面四次跨度 103，约量程的 10%）。根因是
# **指尖初始化在这台固件上从来没真正执行过** —— `init_fingers()` 返回 True、
# 位置纹丝不动（2026-09-24 又验了一次）。没有绝对参考，"合到底"就只是
# "推到推不动为止"，力和速度不同停的地方就不同。
#
# 后果：raw -> opening 的换算有几个百分点的误差，够用但别当精密值；
# 而**位置有效性判据必须留出这个余量**，否则合到底时会被判成"无效反馈"。
POS_OPEN, POS_CLOSED = 0, 1000
POS_MIN, POS_MAX = 0, 1000            # 协议允许的范围，不是实际行程
SPEED_MIN, SPEED_MAX = 200, 1500
STATUS = {0: "常态", 1: "夹取中", 2: "等待抓稳", 3: "抓稳"}


def _le16(v: int) -> bytes:
    return int(v).to_bytes(2, "little")


# ── RS485 / Modbus RTU 层（手册 4.5）────────────────────────────────────────
# 走 RS485 时没有适配器那层封装，直接是裸 Modbus RTU。
# **寄存器地址、数量、值都是大端；CRC16 低字节在前。**
REG_SPEED_ADAPTIVE = 0x000A   # 自适应速度
REG_GRASP_ADAPTIVE = 0x000B   # 自适应抓放 1=抓 0=放
REG_PROTOCOL       = 0x000C   # 1=RS485自定义 2=Modbus RTU
REG_DEV_ID         = 0x000D   # 1~255
REG_MODE           = 0x000E   # 0=自适应 1=力位
REG_FORCE_POS      = 0x000F   # 写 2 个寄存器：速度 + 目标位置（功能码 0x10）
REG_POS_FORCE_MODE = 0x0010   # 力位模式读位置 0~1000
REG_INIT           = 0x0011   # 写 1 触发指尖初始化
REG_FORCE          = 0x0012   # 目标夹取力 1~100
REG_SERIAL_NO      = 0x0013   # 出厂编号，功能码 0x04 读 6 个寄存器
REG_BAUD           = 0x0021   # 波特率，功能码 0x10 写 2 个寄存器
REG_STATE_ADAPTIVE = 0x0006   # 自适应模式状态
REG_POS_ADAPTIVE   = 0x0008   # 自适应模式位置
REG_STATE_FORCE    = 0x002F   # 力位模式状态 0=运动中 1=到位 2=夹到物体
REG_EEPROM         = 0x00EE   # EEPROM 写使能

STATE_FORCE_MODE = {0: "运动中", 1: "到位", 2: "夹到物体"}


def crc16(data: bytes) -> bytes:
    """Modbus RTU CRC-16，低字节在前。"""
    crc = 0xFFFF
    for ch in data:
        crc ^= ch
        for _ in range(8):
            crc = (crc >> 1) ^ 0xA001 if crc & 1 else crc >> 1
    return bytes((crc & 0xFF, crc >> 8))


def mb_read(slave: int, reg: int, n: int = 1, fn: int = 0x03) -> bytes:
    body = bytes((slave, fn)) + reg.to_bytes(2, "big") + int(n).to_bytes(2, "big")
    return body + crc16(body)


def mb_write1(slave: int, reg: int, value: int) -> bytes:
    """功能码 0x06，写单个寄存器。"""
    body = bytes((slave, 0x06)) + reg.to_bytes(2, "big") + int(value).to_bytes(2, "big")
    return body + crc16(body)


def mb_write_n(slave: int, reg: int, values) -> bytes:
    """功能码 0x10，写多个寄存器。力位抓放要用这个（速度 + 位置一起写）。"""
    vals = list(values)
    body = (bytes((slave, 0x10)) + reg.to_bytes(2, "big") + len(vals).to_bytes(2, "big")
            + bytes((len(vals) * 2,)) + b"".join(int(v).to_bytes(2, "big") for v in vals))
    return body + crc16(body)


def mb_parse(data: bytes, slave: int):
    """回包必须 CRC 通过才算数。返回 ("ok", [寄存器值...]) / ("echo", 原样) /
    ("exception", 异常码) / None。"""
    if len(data) < 5 or data[0] != slave:
        return None
    fn = data[1]
    if fn in (0x03, 0x04):
        n = data[2]
        fr = data[: 3 + n + 2]
        if n > 0 and n % 2 == 0 and len(fr) == 3 + n + 2 and crc16(fr[:-2]) == fr[-2:]:
            return "ok", [int.from_bytes(fr[3 + 2 * i: 5 + 2 * i], "big") for i in range(n // 2)]
    elif fn in (0x06, 0x10):
        fr = data[:8]
        if len(fr) == 8 and crc16(fr[:-2]) == fr[-2:]:
            return "echo", fr
    elif fn & 0x80:
        fr = data[:5]
        if len(fr) == 5 and crc16(fr[:-2]) == fr[-2:]:
            return "exception", data[2]
    return None


class HKVRGripper:
    """一问一答式。每条命令发一帧、等一帧回包。

    本类只负责同步协议；遥操通过 HKVRHand 的独立线程调用，避免超时阻塞臂。
    """

    def __init__(self, port="/dev/ttyUSB0", baud=None, can_id=0x0001,
                 timeout=0.05, speed=1000, transport="rs485", slave=1, can_rx_id=None):
        """transport="can" 走 USB-CAN 适配器；"rs485" 是直连 USB 转 485，裸 Modbus RTU。

        两条路的默认波特率不一样：走适配器时 `baud` 是**适配器的 UART 速率**
        （115200，和因时手那条一致）；走 RS485 时是**夹爪自己的串口速率**，
        手册默认 1000000。
        """
        if transport not in ("can", "rs485"):
            raise ValueError("transport 只能是 can 或 rs485")
        self.transport = transport
        self.port = port
        self.baud = int(baud) if baud else (115200 if transport == "can" else 1000000)
        self.can_id, self.slave = int(can_id), int(slave)
        self.can_rx_id = self.can_id if can_rx_id is None else int(can_rx_id)
        self.timeout, self.speed = timeout, int(speed)
        if not 1 <= self.slave <= 247 or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Modbus 从站必须为 1..247，timeout 必须为正数")
        self._ser = None
        self._buf = bytearray()
        self.connected = False

    # ── 连接 ──
    def connect(self):
        import serial
        self._ser = serial.Serial(self.port, self.baud, bytesize=8, parity="N",
                                  stopbits=1, timeout=self.timeout,
                                  write_timeout=self.timeout, exclusive=True)
        time.sleep(0.05)
        self._ser.reset_input_buffer()
        try:
            # 先读与控制模式无关的目标力，不用力位专属寄存器探测自适应模式设备。
            info = self.info() if self.transport == "can" else self.target_force()
            if info is None:
                raise IOError(f"{self.port} ({self.transport}) 无有效回包：检查 24V、接口类型、"
                              "地址与通信速率；RS485 要匹配 Modbus 模式；"
                              "CAN 要匹配适配器封装、总线速率、收发 ID 和夹爪侧转换接口。")
        except BaseException:
            self.close()
            raise
        self.connected = True
        return info

    def close(self):
        if self._ser is not None:
            try:
                self._ser.close()
            finally:
                self._ser = None
        self.connected = False

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *exc):
        self.close()

    # ── 收发 ──
    def _send(self, data: bytes, dlc: int):
        self._ser.write(build_frame(self.can_id, data, dlc))

    def _recv(self, want_cmd: int, tries=1):
        """等一个首字节是 want_cmd 的回包。别的帧丢掉（总线上可能还有别人）。"""
        deadline = time.monotonic() + self.timeout * tries
        while time.monotonic() < deadline:
            self._ser.timeout = max(0.001, deadline - time.monotonic())
            chunk = self._ser.read(max(1, self._ser.in_waiting))
            if chunk:
                self._buf += chunk
            for _cid, data, dlc, ok in parse_frames(self._buf):
                if ok and _cid == self.can_rx_id and dlc and data[0] == want_cmd:
                    return data[:dlc]
        return None

    def _cmd(self, code: int, payload: bytes = b"\x01\x00", tries=1):
        self._ser.reset_input_buffer()
        self._buf.clear()
        self._send(bytes((code,)) + payload, 1 + len(payload))
        return self._recv(code, tries)

    def _can_write(self, code, payload, tries=1):
        got = self._cmd(code, payload, tries)
        if got is None:
            return False
        if got != bytes((code, 0, 0)):
            raise IOError(f"HKVR CAN 未确认成功：{got.hex(' ')}")
        return True

    # ── Modbus 收发（transport="rs485"）──
    def _mb(self, frame: bytes, tries=1):
        for _ in range(tries):
            # Modbus RTU 在 >19200 baud 时使用固定 1.75ms 帧间隔。
            time.sleep(0.00175)
            self._ser.reset_input_buffer()
            self._ser.write(frame)
            deadline = time.monotonic() + self.timeout
            data = bytearray()
            while time.monotonic() < deadline:
                self._ser.timeout = max(0.001, deadline - time.monotonic())
                data.extend(self._ser.read(max(1, self._ser.in_waiting)))
                # 帧长已足够就返回，不能 read(64) 等满超时；支持分包和前导噪声。
                while len(data) >= 3:
                    if data[0] != self.slave or data[1] not in (frame[1], frame[1] | 0x80):
                        del data[0]
                        continue
                    fn = data[1]
                    size = 5 if fn & 0x80 else (5 + data[2] if fn in (3, 4) else 8)
                    if size > 256:
                        del data[0]
                        continue
                    if len(data) < size:
                        break
                    got = mb_parse(bytes(data[:size]), self.slave)
                    if got is None:
                        del data[0]
                        continue
                    del data[:size]
                    if got[0] == "exception":
                        raise IOError(f"HKVR Modbus 异常 0x{got[1]:02x}，请求 {frame.hex(' ')}")
                    if fn in (3, 4):
                        if len(got[1]) != int.from_bytes(frame[4:6], "big"):
                            continue
                    elif got[1][:6] != frame[:6]:
                        continue  # 旧回包或不匹配的寄存器/值，不能当成功
                    return got
        return None

    # ── 读 ──
    def info(self):
        """读位置和状态。返回 (位置 0~1000, 状态码, 状态名)，读不到就 None。"""
        if self.transport == "rs485":
            # **位置读 0x0008、状态读 0x0006，不是手册写的 0x0010 / 0x002F。**
            #
            # 2026-09-22 在这台夹爪上实测：力位模式（0x000E 读回 1）下，
            # 0x0010 和 0x002F **恒为 0**，而 0x0008 会跟着指令走、0x0006 会在
            # 0/1 之间翻。指令 200→0x0008 读 710，指令 900→爬到 771，回 400→710。
            # 读 0x0010 的后果是驱动判定「无有效位置反馈」直接停发，遥操起不来。
            #
            # 手册第 4.5 节把 0x0010/0x002F 列成力位模式的位置和状态，
            # 和这台固件对不上 —— 以手册为准会一直读到 0。
            p = self._mb(mb_read(self.slave, REG_POS_ADAPTIVE, 1))
            s = self._mb(mb_read(self.slave, REG_STATE_ADAPTIVE, 1))
            if p is None or p[0] != "ok" or s is None or s[0] != "ok":
                return None
            st = s[1][0]
            return p[1][0], st, STATUS.get(st, f"未知({st})")
        got = self._cmd(CMD_INFO, b"\x00\x00\x01\x01\x00\x00\x00")
        if got is None or len(got) != 5:
            return None
        pos = int.from_bytes(got[1:3], "little")
        st = int.from_bytes(got[3:5], "little")
        return pos, st, STATUS.get(st, f"未知({st})")

    def position(self):
        got = self.info()
        return None if got is None else got[0]

    def target_force(self):
        if self.transport == "rs485":
            got = self._mb(mb_read(self.slave, REG_FORCE, 1))
            return got[1][0] if got and got[0] == "ok" else None
        got = self._cmd(CMD_FORCE_GET)
        return None if got is None or len(got) < 3 else int.from_bytes(got[1:3], "little")

    # ── 写 ──
    def init_fingers(self):
        """指尖初始化。**上电后要做一次**，否则位置零点不准。

        它会走一遍全行程，**别在夹着东西的时候做**。
        """
        if self.transport == "rs485":
            return self._mb(mb_write1(self.slave, REG_INIT, 1)) is not None
        return self._can_write(CMD_INIT, b"\x01\x00", tries=200)

    def set_mode(self, mode: int):
        if self.transport == "rs485":
            return self._mb(mb_write1(self.slave, REG_MODE, mode)) is not None
        return self._can_write(CMD_MODE, bytes((mode, 0x00)))

    def set_force(self, force: int):
        """目标夹取力 1~100。**掉电不保存**，除非先开 EEPROM 使能 —— 这里不碰 EEPROM。"""
        f = max(1, min(100, int(force)))
        if self.transport == "rs485":
            return self._mb(mb_write1(self.slave, REG_FORCE, f)) is not None
        return self._can_write(CMD_FORCE_SET, _le16(f))

    def move(self, position: int, speed=None):
        """力位模式：走到目标位置。**position 0=全开、1000=合拢**（POS_CLOSED，实际停点会漂 ±50），speed 200~1500。"""
        pos = max(POS_MIN, min(POS_MAX, int(position)))
        spd = max(SPEED_MIN, min(SPEED_MAX, int(self.speed if speed is None else speed)))
        if self.transport == "rs485":
            # 速度和位置必须一次写完（功能码 0x10，两个寄存器），分两次写夹爪不认
            return self._mb(mb_write_n(self.slave, REG_FORCE_POS, (spd, pos))) is not None
        return self._can_write(CMD_FORCE_POS, _le16(spd) + _le16(pos))

    # ── 给上层用的归一化接口 ──
    def set_opening(self, opening: float):
        """0.0 合拢 ~ 1.0 全开。和 VEL 那套 `hand.set_opening()` 对齐。

        **和夹爪的原始位置是反的**：opening 1.0（全开）对应位置 0，
        opening 0.0（合拢）对应位置 POS_CLOSED = 1000（手册量程；9-24 前用的 858 是卡住时量的）。
        """
        if not math.isfinite(float(opening)):
            raise ValueError("夹爪开合必须为有限数")
        o = max(0.0, min(1.0, float(opening)))
        return self.move(round(POS_OPEN + (1.0 - o) * (POS_CLOSED - POS_OPEN)))

    def get_opening(self):
        pos = self.position()
        if pos is None:
            return None
        span = max(POS_CLOSED - POS_OPEN, 1)
        return float(max(0.0, min(1.0, 1.0 - (pos - POS_OPEN) / span)))


# ── 离线自检：拿手册里的示例帧逐字节对 ────────────────────────────────────
def _selftest() -> int:
    """`python hkvr_gripper.py` 直接跑。不碰硬件。

    每条都对着手册 4.6 节的示例字节。协议写错了这里就会红，不用等接上硬件。
    """
    cases = [
        ("获取夹爪信息",  bytes([0x03, 0, 0, 1, 1, 0, 0, 0]), 8),
        ("指尖初始化",    bytes([CMD_INIT, 0x01, 0x00]), 3),
        ("切力位模式",    bytes([CMD_MODE, MODE_FORCE_POS, 0x00]), 3),
        ("力位抓放",      bytes([CMD_FORCE_POS]) + _le16(1000) + _le16(1000), 5),
        ("自适应设速度",  bytes([CMD_SPEED_ADAPTIVE]) + _le16(1000), 3),
        ("设夹取力 50",   bytes([CMD_FORCE_SET]) + _le16(50), 3),
        ("EEPROM 使能开", bytes([CMD_EEPROM, 0x01, 0x00]), 3),
    ]
    ok = True
    print("  手册示例 → 本驱动生成的数据段：")
    for name, data, dlc in cases:
        print(f"    {name:14s} DLC={dlc}  {data[:dlc].hex(' ')}")
    # 手册 4.6 的原文字节
    expect = {
        "获取夹爪信息": "03 00 00 01 01 00 00 00",
        "指尖初始化":   "04 01 00",
        "切力位模式":   "0e 01 00",
        "力位抓放":     "10 e8 03 e8 03",
        "自适应设速度": "02 e8 03",
        "设夹取力 50":  "06 32 00",
        "EEPROM 使能开": "11 01 00",
    }
    for name, data, dlc in cases:
        got = data[:dlc].hex(" ")
        if got != expect[name]:
            print(f"    ✗ {name}: 生成 {got}，手册 {expect[name]}")
            ok = False

    # 适配器层：裹完再拆，要能还原
    frame = build_frame(0x0001, bytes([0x03, 0, 0, 1, 1, 0, 0, 0]), 8)
    back = parse_frames(bytearray(frame))
    if len(back) != 1 or back[0][0] != 1 or back[0][2] != 8 or not back[0][3]:
        print(f"    ✗ 适配器层裹拆不一致：{back}")
        ok = False

    # 转义：数据里带 0x55/0xAA/0xA5 时帧会变长，但拆出来必须一样
    tricky = bytes([CMD_FORCE_POS, 0xA5, 0x55, 0xAA, 0x03])
    f2 = build_frame(0x0001, tricky, 5)
    b2 = parse_frames(bytearray(f2))
    if len(b2) != 1 or b2[0][1][:5] != tricky or not b2[0][3]:
        print(f"    ✗ 转义后还原不出来：{b2}")
        ok = False
    else:
        print(f"    转义测试：{len(f2)} 字节帧（未转义是 21），拆回来一致 ✓")

    # ── Modbus RTU（手册 4.5）逐字节对 ──
    mb_cases = [
        ("读目标力",      mb_read(1, REG_FORCE, 1),              "01 03 00 12 00 01 24 0f"),
        ("设目标力 1",    mb_write1(1, REG_FORCE, 1),            "01 06 00 12 00 01 e8 0f"),
        ("EEPROM 使能开", mb_write1(1, REG_EEPROM, 1),           "01 06 00 ee 00 01 28 3f"),
        ("读出厂编号",    mb_read(1, REG_SERIAL_NO, 6, fn=0x04), "01 04 00 13 00 06 81 cd"),
        ("切力位模式",    mb_write1(1, REG_MODE, 1),             "01 06 00 0e 00 01 29 c9"),
        ("指尖初始化",    mb_write1(1, REG_INIT, 1),             "01 06 00 11 00 01 18 0f"),
        ("自适应设速度",  mb_write1(1, REG_SPEED_ADAPTIVE, 1000),"01 06 00 0a 03 e8 a9 76"),
        ("自适应抓取",    mb_write1(1, REG_GRASP_ADAPTIVE, 1),   "01 06 00 0b 00 01 39 c8"),
        ("读力位位置",    mb_read(1, REG_POS_FORCE_MODE, 1),     "01 03 00 10 00 01 85 cf"),
        ("读力位状态",    mb_read(1, REG_STATE_FORCE, 1),        "01 03 00 2f 00 01 b5 c3"),
        ("力位抓放",      mb_write_n(1, REG_FORCE_POS, (1000, 1000)),
                          "01 10 00 0f 00 02 04 03 e8 03 e8 33 21"),
        ("改 ID 为 2",    mb_write1(1, REG_DEV_ID, 2),           "01 06 00 0d 00 02 99 c8"),
        ("切协议",        mb_write1(1, REG_PROTOCOL, 1),         "01 06 00 0c 00 01 88 09"),
    ]
    print("\n  Modbus RTU（手册 4.5）：")
    for name, frame, want in mb_cases:
        got = frame.hex(" ")
        mark = "✓" if got == want else "✗"
        if got != want:
            ok = False
            print(f"    {mark} {name:14s} 生成 {got}")
            print(f"      {'':14s} 手册 {want}")
        else:
            print(f"    {mark} {name:14s} {got}")

    print("\n  " + ("全过 ✓" if ok else "有不一致 ✗"))
    return 0 if ok else 1


if __name__ == "__main__":
    import sys as _sys
    _sys.exit(_selftest())
