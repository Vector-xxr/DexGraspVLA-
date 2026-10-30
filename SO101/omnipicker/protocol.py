"""智元 OmniPicker 夹爪的 Modbus RTU 协议（固件 ≥ 3.3.5，`modbus_mode=True`）。

寄存器表来自智元文档 agibot.com.cn/DOCS/PM/X1 第 6.4.4 节，2026-10-19 在本机实测过：

    0        固件版本（实测 335）
    10~14    目标 位置 / 速度 / 力 / 加速度 / 减速度，只取低字节 0~255
    15       触发：写非零开始运动，固件开始执行后自动清零
    20~24    当前 错误码 / 状态 / 位置 / 速度 / 力矩

位置 0 = 合拢，255 = 全开。功能码只用 0x03（读）和 0x10（写多个）。

这一层只管收发和校验，**不做任何安全判断** —— 张开上限、张开方向最小力、
回绕检测这些都在 `gripper.py`。直接用这个类发指令的人要自己守那几条规矩，
见 README 的「实测规矩」。
"""
import serial

# 当前错误码（寄存器 20）
FAULTS = {0: "无故障", 1: "过温", 2: "超速", 3: "初始化故障", 4: "超限检测"}
# 当前状态（寄存器 21）。**比位置晚约 0.2 秒更新**，刚发完指令读到的是旧值。
STATES = {0: "已到达", 1: "移动中", 2: "堵转", 3: "物体掉落"}
STATE_ARRIVED, STATE_MOVING, STATE_BLOCKED, STATE_DROPPED = 0, 1, 2, 3

REG_FIRMWARE = 0
REG_TARGET = 10          # 10~15 一次写 6 个
REG_STATUS = 20          # 20~24 一次读 5 个


def crc16(data: bytes) -> bytes:
    """Modbus CRC16，低字节在前。"""
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ 0xA001 if crc & 1 else crc >> 1
    return bytes((crc & 0xFF, crc >> 8))


def read_frame(addr: int, start: int, count: int) -> bytes:
    body = bytes((addr, 0x03, start >> 8, start & 0xFF, count >> 8, count & 0xFF))
    return body + crc16(body)


def command_frame(addr: int, pos: int, vel: int, force: int, acc: int, dec: int) -> bytes:
    """写 10~15：目标位置、速度、力、加速度、减速度，外加触发位 1。"""
    values = (pos, vel, force, acc, dec)
    if any(not 0 <= int(v) <= 255 for v in values):
        raise ValueError(f"位置/速度/力/加减速都必须在 0..255：{values}")
    # SDK 头文件注明加减速为 0 会被固件拒绝
    if acc == 0 or dec == 0:
        raise ValueError("加速度/减速度不能为 0")
    body = bytes((addr, 0x10, 0, REG_TARGET, 0, 6, 12))
    body += b"".join(int(v).to_bytes(2, "big") for v in (*values, 1))
    return body + crc16(body)


class ModbusError(IOError):
    """没回包、回包不全、CRC 错、或者夹爪回了异常码。"""


class OmniPickerBus:
    """一个串口上的一台夹爪。**不是线程安全的**：串口只能归一个线程。"""

    def __init__(self, port: str, addr: int = 1, baudrate: int = 115200, timeout: float = 0.05):
        if not 1 <= addr <= 247:
            raise ValueError("Modbus 从站地址必须是 1..247（0 是广播）")
        self.addr = addr
        self.port_name = port
        self.ser = serial.Serial(port, baudrate, bytesize=8, parity="N", stopbits=1, timeout=timeout)

    def close(self):
        self.ser.close()

    def _transact(self, frame: bytes, reply_len: int) -> bytes:
        self.ser.reset_input_buffer()
        self.ser.write(frame)
        self.ser.flush()
        rx = self.ser.read(5)
        func = frame[1]
        if len(rx) == 5 and rx[1] == func | 0x80 and crc16(rx[:3]) == rx[3:5]:
            raise ModbusError(f"夹爪回了异常码 {rx[2]}（功能码 0x{func:02x}）")
        rx += self.ser.read(reply_len - len(rx))
        if not rx:
            raise ModbusError(f"{self.port_name} 没有回包")
        if len(rx) != reply_len:
            raise ModbusError(f"回包不全：{rx.hex(' ')}")
        if rx[0] != self.addr or rx[1] != func or crc16(rx[:-2]) != rx[-2:]:
            raise ModbusError(f"回包地址/功能码/CRC 不对：{rx.hex(' ')}")
        return rx

    def read_registers(self, start: int, count: int) -> list:
        rx = self._transact(read_frame(self.addr, start, count), 5 + 2 * count)
        if rx[2] != 2 * count:
            raise ModbusError(f"字节数不符：{rx.hex(' ')}")
        return [int.from_bytes(rx[3 + 2 * i:5 + 2 * i], "big") for i in range(count)]

    def firmware(self) -> int:
        return self.read_registers(REG_FIRMWARE, 1)[0]

    def status(self) -> dict:
        fault, state, pos, vel, force = self.read_registers(REG_STATUS, 5)
        return dict(fault=fault, state=state, pos=pos, vel=vel, force=force)

    def command(self, pos: int, vel: int, force: int, acc: int, dec: int) -> None:
        self._transact(command_frame(self.addr, pos, vel, force, acc, dec), 8)
