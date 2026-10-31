"""OmniPicker 驱动参数。要持久改就改这里，命令行只覆盖一次。

数值都是寄存器原值 0~255，**不是牛顿**。手册给的最大夹持力 30 N，但 0~255 和牛顿的
对应关系没给、也没标定过。
"""
import math
from dataclasses import dataclass


@dataclass
class OmniPickerConfig:
    # by-id，不要写 /dev/ttyACM0。这块 hkvtech 转换器是 CDC 串口，主臂 SO-101 也是，
    # 两个都插着时 ttyACMn 按插入顺序给，拿错了一个会把对方的协议灌进自己的总线。
    #
    # **这块转换器的 A/B 和智元文档是反的**：夹爪 4 脚（H）接转换器 B，3 脚（L）接 A。
    # 照文档接是零回包；它那盏绿灯是电源灯，发数据时不闪，别拿它判断有没有发出去。
    port: str = "/dev/serial/by-id/usb-hkvtech.cn_HKV_USB-RS485_CANFD_20260717-if00"
    addr: int = 1                       # 就是夹爪的 can_node_id
    baudrate: int = 115200              # 夹爪 uart_baudrate，停止位 1
    timeout: float = 0.05               # 单次读写超时；实测读 ~7 ms、写 ~8 ms
    # I/O 线程频率。平时只读不写（指令只在空格 / O 时发），一次读 ~8 ms，100 Hz 排得下；
    # 有写的那一圈 ~16 ms，偶尔超一圈无妨。10-21 从 50 改成 100：录一帧时夹爪读数从平均
    # 旧 14 ms（最多 24）降到 9 ms（最多 14）。一次读的 7.6 ms 里线上只占 ~2 ms，其余是夹爪
    # 处理 + 转换器攒整包，提高波特率最多省 1.5 ms。
    hz: float = 100.0
    stale_timeout: float = 0.5          # 反馈多久没更新就算断线

    # 夹持力：合拢方向的力上限，也就是夹住东西后一直保持的力。采集界面里 ↑↓ 调。
    # 2026-10-19 实测：设 32 夹住读数 ~24、设 64 读数 ~47；夹人手 32 停在被夹处、
    # 0.5 秒后报堵转，夹着不再变紧。范围先只放到测过的附近，要放宽改这里。
    grip_force: int = 64
    grip_force_min: int = 32
    grip_force_max: int = 128
    grip_force_step: int = 16

    # 张开方向的力**不能低于 64**：给 32 时张开跟不上、冲过 255 位置回绕，固件自己
    # 也跟着乱，卡在张开端只能断电重启（2026-10-19 复现两次）。64 张开十几次都干净。
    open_force: int = 64

    # 开合速度，夹紧和松开共用。采集界面里准备阶段 ← → 调，档位见 speed_levels()。
    # 实测全行程张开：64 ~2.2 s、128 ~1.4 s、255 ~0.9 s。**越快碰到东西那一下冲击越大**
    # （运动中的力矩峰值不受夹持力限制），夹易碎件往慢调。
    # 下限 32：再慢一个读数周期走不到 1 个单位，「移动中但位置不动」会被误判成卡死。
    speed: int = 128
    speed_min: int = 32
    speed_max: int = 255
    speed_step: int = 16
    acc: int = 128
    dec: int = 128

    # 张开上限 210（约 99 mm；全行程 255 ≈ 120 mm）。**从夹紧状态松开、目标又靠近张开端时会冲过头**，
    # 冲过 255 就回绕、固件跟着乱，只能断电重启。2026-10-20 实测（力 64、减速 128，先合到 0 顶 1 秒再松开）：
    #   目标 200 / 210：速度 128、176、255 都不冲（合拢用 176、255 也一样）
    #   目标 230：速度 128 就冲到 241（+11）
    #   目标 240：速度 176 冲过 255 回绕（采集时真出过一次）
    # 原来是 240（只看了从空位张开不冲）。要放宽先按同样的办法测，别只改数字。
    #
    # **10-30 起 open_max 只当开度的换算基准**（开度 = 位置 / 210），数据和部署都按它换算，不能再改；
    # 张开目标另由 open_limit 封顶。
    open_max: int = 210
    # 张开目标的上限 180（约 85 mm）。10-30 采集时目标 210 又出事两次：力 64、速度 128 从夹紧松开
    # 冲到 219、在 206~218 来回晃；采集中两次卡在张开端（213、209）报堵转、不响应合拢，断电才好。
    # 同日力 112、速度 160 从夹紧松开到 178：停在 177，不冲不晃。
    # 全张开在数据里记成 180/210 ≈ 0.86，不是 1。
    open_limit: int = 180
    deadband: int = 2                   # 目标变化小于这个就不重发，约 1 mm

    # 「移动中」但位置不动、离目标还远，持续这么久判卡死（航凯夹爪 9-30 的教训）
    stuck_seconds: float = 1.5

    def __post_init__(self):
        for name in ("grip_force", "grip_force_min", "grip_force_max", "open_force",
                     "speed", "acc", "dec", "open_max", "open_limit"):
            if not 1 <= int(getattr(self, name)) <= 255:
                raise ValueError(f"OmniPicker {name} 必须是 1..255")
        if not 1 <= self.addr <= 247:
            raise ValueError("OmniPicker addr 必须是 1..247")
        if self.open_force < 64:
            raise ValueError("open_force 不能低于 64：张开方向力小了会冲过头回绕，见注释")
        if self.open_max > 210:
            raise ValueError("open_max 不能超过 210：从夹紧状态松开、目标靠近张开端会冲过 255 回绕，见注释")
        if self.open_limit > self.open_max:
            raise ValueError(f"open_limit {self.open_limit} 不能超过 open_max {self.open_max}："
                             "目标靠近张开端会冲过头、卡在张开端，见注释")
        if not self.grip_force_min <= self.grip_force <= self.grip_force_max:
            raise ValueError(f"grip_force {self.grip_force} 不在 "
                             f"[{self.grip_force_min}, {self.grip_force_max}]")
        if self.speed_min < 32 or self.speed_max > 255 or not self.speed_min <= self.speed <= self.speed_max:
            raise ValueError(f"speed {self.speed} 不在 [{self.speed_min}, {self.speed_max}]，"
                             "下限不能低于 32（太慢会被误判成卡死）")
        if self.grip_force_step < 1 or self.speed_step < 1 or self.deadband < 1:
            raise ValueError("grip_force_step、speed_step、deadband 至少为 1")

    def speed_levels(self) -> list:
        """← → 走的档位：从 speed_min 起每 speed_step 一档，最后补上 speed_max。"""
        levels = list(range(self.speed_min, self.speed_max, self.speed_step))
        return levels + [self.speed_max]
        if (not all(math.isfinite(v) and v > 0 for v in
                    (self.timeout, self.hz, self.stale_timeout, self.stuck_seconds))
                or self.timeout > 0.2 or self.hz > 100
                or self.stale_timeout <= 3 * self.timeout + 1 / self.hz):
            raise ValueError("OmniPicker 通信周期/超时参数无效")
