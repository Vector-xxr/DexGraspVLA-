"""不连硬件的自检：帧格式、参数校验、驱动的几道保护。

    cd ~/yuhang_workspace/算法/DexGraspVLA-/SO101
    PYTHONPATH= ~/miniforge3/envs/dexgraspvla/bin/python -m omnipicker.selftest

用一个假夹爪（FakeBus）代替串口：位置按速度往目标走，可以放「物体」让它停住报堵转，
也可以注入错误码、回绕、卡死、丢包。全部通过返回 0。
"""
import sys
import time
import traceback

from .config import OmniPickerConfig
from .gripper import OmniPickerGripper
from .protocol import (STATE_ARRIVED, STATE_BLOCKED, STATE_MOVING, ModbusError,
                       command_frame, crc16, read_frame)

_results = []


def check(name):
    def wrap(fn):
        _results.append((name, fn))
        return fn
    return wrap


class FakeBus:
    """一台假夹爪。只被驱动的 I/O 线程调用，所以不加锁。"""

    def __init__(self, firmware=335, pos=0):
        self.fw = firmware
        self.pos = float(pos)
        self.target = None
        self.vel = 0
        self.force_cmd = 0
        self.fault = 0
        self.obstacle = None        # 合拢方向碰到这个位置就停住、报堵转
        self.frozen = False         # 卡死：报移动中但不动
        self.wrap_to = None         # 下一次读状态把位置改成这个（模拟回绕）
        self.fail_reads = 0         # 接下来这么多次读状态抛 ModbusError
        self.commands = []
        self.closed = False
        self._t = time.monotonic()

    def _advance(self):
        now = time.monotonic()
        dt, self._t = now - self._t, now
        if self.target is None or self.frozen:
            return
        step = self.vel * 2.0 * dt              # 速度 128 ≈ 256 单位/秒，够测试用
        if self.target < self.pos:
            floor = self.target if self.obstacle is None else max(self.target, self.obstacle)
            self.pos = max(floor, self.pos - step)
        else:
            self.pos = min(self.target, self.pos + step)

    def firmware(self):
        return self.fw

    def status(self):
        if self.fail_reads:
            self.fail_reads -= 1
            raise ModbusError("假夹爪：丢一次包")
        self._advance()
        if self.wrap_to is not None:
            self.pos, self.wrap_to = float(self.wrap_to), None
        if self.frozen:
            state = STATE_MOVING
        elif self.target is None or abs(self.pos - self.target) < 1:
            state = STATE_ARRIVED
        elif self.obstacle is not None and self.target < self.obstacle and abs(self.pos - self.obstacle) < 1:
            state = STATE_BLOCKED
        else:
            state = STATE_MOVING
        force = self.force_cmd * 3 // 4 if state == STATE_BLOCKED else 5
        return dict(fault=self.fault, state=state, pos=int(round(self.pos)), vel=0, force=force)

    def command(self, pos, vel, force, acc, dec):
        command_frame(1, pos, vel, force, acc, dec)     # 同样的范围检查
        self._advance()
        self.commands.append((pos, vel, force, acc, dec))
        self.target, self.vel, self.force_cmd = pos, vel, force

    def close(self):
        self.closed = True


def make(bus=None, **overrides):
    bus = bus or FakeBus()
    cfg = OmniPickerConfig(port="/dev/null-fake", hz=100.0, stale_timeout=0.5,
                           stuck_seconds=0.3, **overrides)
    gripper = OmniPickerGripper(cfg, bus_factory=lambda _cfg: bus)
    return gripper, bus


def wait_for(predicate, timeout=2.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if predicate():
            return True
        time.sleep(0.01)
    return False


# ── 帧格式 ──────────────────────────────────────────────────────────

@check("读状态帧和智元文档示例一字不差")
def _():
    assert read_frame(1, 20, 5) == bytes.fromhex("01 03 00 14 00 05 C5 CD")


@check("运动帧和智元文档示例一字不差（50% 行程、其余最大）")
def _():
    frame = command_frame(1, 0x7F, 0xFF, 0xFF, 0xFF, 0xFF)
    assert frame == bytes.fromhex("01 10 00 0A 00 06 0C 00 7F 00 FF 00 FF 00 FF 00 FF 00 01 B9 BA"), frame.hex(" ")
    assert crc16(frame[:-2]) == frame[-2:]


@check("运动帧拒绝越界值和 0 加减速")
def _():
    for bad in ((256, 1, 1, 1, 1), (0, 1, 1, 0, 1), (0, 1, 1, 1, 0), (-1, 1, 1, 1, 1)):
        try:
            command_frame(1, *bad)
        except ValueError:
            continue
        raise AssertionError(f"没拒绝 {bad}")


# ── 参数校验 ────────────────────────────────────────────────────────

@check("配置拒绝 open_force<64、open_max>210、open_limit>open_max、夹持力出界")
def _():
    for bad in (dict(open_force=32), dict(open_max=255), dict(open_max=220), dict(grip_force=200),
                dict(grip_force=16), dict(addr=0), dict(open_limit=211), dict(open_max=170)):
        try:
            OmniPickerConfig(**bad)
        except ValueError:
            continue
        raise AssertionError(f"没拒绝 {bad}")


# ── 驱动 ────────────────────────────────────────────────────────────

@check("连上只读不动：connect 不发任何运动指令")
def _():
    gripper, bus = make()
    gripper.connect()
    try:
        time.sleep(0.1)
        assert bus.commands == [], bus.commands
        gripper.check_health()
    finally:
        gripper.disconnect()
    assert bus.closed


@check("固件低于 335 拒绝连接，串口关掉")
def _():
    gripper, bus = make(FakeBus(firmware=141))
    try:
        gripper.connect()
    except IOError as exc:
        assert "335" in str(exc)
    else:
        raise AssertionError("旧固件没被拒绝")
    assert bus.closed and bus.commands == []


@check("连上时夹爪已报错就拒绝连接")
def _():
    bus = FakeBus()
    bus.fault = 1
    gripper, _ = make(bus)
    try:
        gripper.connect()
    except IOError as exc:
        assert "过温" in str(exc)
    else:
        raise AssertionError("带错误码没被拒绝")


@check("张开：目标封顶 open_limit 180，张开方向的力不低于 open_force")
def _():
    gripper, bus = make(grip_force=32)
    gripper.connect()
    try:
        gripper.set_opening(1.0)
        assert wait_for(lambda: bus.commands), "没发指令"
        pos, _vel, force, _acc, _dec = bus.commands[0]
        assert pos == 180, pos
        assert force == 64, f"张开用了力 {force}，应该是 open_force 64"
    finally:
        gripper.disconnect()


@check("开度仍按 open_max 210 换算（和已有数据一致），只有目标被 open_limit 封顶")
def _():
    gripper, bus = make(FakeBus(pos=123))
    assert gripper.target_position(0.5) == 105
    assert gripper.target_position(170 / 210) == 170
    assert gripper.target_position(0.9) == 180 and gripper.target_position(1.0) == 180
    gripper.connect()
    try:
        assert wait_for(lambda: gripper.last_status())
        assert abs(gripper.get_opening() - 123 / 210) < 1e-6, gripper.get_opening()
    finally:
        gripper.disconnect()


@check("合拢：用夹持力；碰到物体停住报堵转")
def _():
    bus = FakeBus(pos=240)
    bus.obstacle = 32
    gripper, _ = make(bus, grip_force=48)
    gripper.connect()
    try:
        gripper.set_opening(0.0)
        assert wait_for(lambda: gripper.last_status().get("state") == STATE_BLOCKED), gripper.last_status()
        pos, _vel, force, _acc, _dec = bus.commands[0]
        assert (pos, force) == (0, 48), bus.commands[0]
        assert abs(gripper.get_opening() - 32 / 210) < 0.01
        gripper.check_health()          # 堵转是正常的夹住，不算故障
    finally:
        gripper.disconnect()


@check("目标变化小于死区不重发；改夹持力用同一目标重发")
def _():
    gripper, bus = make(deadband=2)
    gripper.connect()
    try:
        gripper.set_opening(0.5)                    # 105
        assert wait_for(lambda: len(bus.commands) == 1)
        gripper.set_opening(106 / 210)              # 差 1，不发
        time.sleep(0.1)
        assert len(bus.commands) == 1, bus.commands
        gripper.set_opening(0.0)
        assert wait_for(lambda: len(bus.commands) == 2)
        new = gripper.step_grip_force(+1)
        assert new == 80, new
        assert wait_for(lambda: len(bus.commands) == 3), bus.commands
        assert bus.commands[-1][0] == 0 and bus.commands[-1][2] == 80, bus.commands[-1]
    finally:
        gripper.disconnect()


@check("夹持力夹在 [min, max]")
def _():
    gripper, _ = make()
    assert gripper.set_grip_force(999) == 128
    assert gripper.set_grip_force(0) == 32
    assert gripper.step_grip_force(+1) == 48
    assert gripper.step_grip_force(-5) == 32


@check("位置回绕：停发、报错要求断电重启，之后不再发任何指令")
def _():
    bus = FakeBus(pos=230)
    gripper, _ = make(bus)
    gripper.connect()
    try:
        gripper.set_opening(1.0)
        assert wait_for(lambda: bus.commands)
        bus.wrap_to = 3
        assert wait_for(lambda: gripper._error is not None), "回绕没被发现"
        assert "断电重启" in str(gripper._error)
        sent = len(bus.commands)
        try:
            gripper.set_opening(0.0)
        except RuntimeError:
            pass
        else:
            raise AssertionError("回绕之后 set_opening 还能下指令")
        time.sleep(0.1)
        assert len(bus.commands) == sent
    finally:
        gripper.disconnect()


@check("移动中但位置不动：判卡死、停发")
def _():
    bus = FakeBus(pos=100)
    gripper, _ = make(bus)
    gripper.connect()
    try:
        gripper.set_opening(0.0)
        assert wait_for(lambda: bus.commands)
        bus.frozen = True
        assert wait_for(lambda: gripper._error is not None, timeout=2.0), "卡死没被发现"
        assert "卡" in str(gripper._error)
    finally:
        gripper.disconnect()


@check("到位后状态一直显示移动中（固件真会这样）不误报卡死")
def _():
    bus = FakeBus(pos=0)
    gripper, _ = make(bus)
    gripper.connect()
    try:
        gripper.set_opening(1.0)
        assert wait_for(lambda: abs(bus.pos - gripper.config.open_limit) < 1)
        bus.frozen = True                           # 停在目标上、状态仍是移动中
        time.sleep(0.6)
        gripper.check_health()
    finally:
        gripper.disconnect()


@check("错误码：停发报错")
def _():
    gripper, bus = make()
    gripper.connect()
    try:
        bus.fault = 2
        assert wait_for(lambda: gripper._error is not None)
        assert "超速" in str(gripper._error)
    finally:
        gripper.disconnect()


@check("偶尔丢包不算断线；一直不回才报错")
def _():
    gripper, bus = make()
    gripper.connect()
    try:
        bus.fail_reads = 3
        time.sleep(0.15)
        gripper.check_health()
        bus.fail_reads = 10_000
        assert wait_for(lambda: gripper._error is not None, timeout=2.0)
    finally:
        gripper.disconnect()


# ── 采集按键流程 ────────────────────────────────────────────────────

def _flow(existing=None, saved=None):
    """existing: {任务名: dict(count, prompt, grip_force, speed)}，模拟 new/ 下已有的目录。
    目录在就算已有（一条没存也算）。saved: [当前数据集磁盘上的条数]，可以在测试里改。"""
    from .keyflow import KeyFlow
    existing = existing or {}

    def probe(name):
        if not name or "/" in name or name.startswith("."):
            return dict(error="任务名不合法")
        if name not in existing:
            return dict(error=None, exists=False, count=0, prompt=None)
        info = dict(count=0, prompt=None, grip_force=None, speed=None)
        info.update(existing[name])
        return dict(error=None, exists=True, **info)
    saved = saved if saved is not None else [0]
    return KeyFlow(probe, saved_count=lambda: saved[0])


def _type(flow, text):
    out = []
    for ch in text:
        out.append(flow.on_key(ch))
    return out


@check("按键流程：P → 任务名 → prompt → 测试力度速度 → 回车 → 回车开录 → 暂停 → 接着录 → S → 回车录下一条")
def _():
    flow = _flow()
    assert flow.phase == "idle" and "P 开始" in flow.help()
    assert flow.on_key("p")[0] == ["help", "prompt_line"] and flow.text_input
    _type(flow, "pick cup")                         # 空格、p、c 在输入时都是字
    assert flow.input_line().endswith("pick cup")
    assert flow.on_key("\n")[0] == ["help", "prompt_line"]
    assert flow.phase == "prompt" and flow.name == "pick_cup"
    _type(flow, "pick up the cup")
    assert flow.gripper_command is None, "测试之前不该自己发夹爪指令"
    assert flow.on_key("\r")[0] == ["open_dataset", "auto_open", "help"]
    assert flow.prompt == "pick up the cup" and flow.phase == "test" and not flow.text_input
    assert flow.gripper_command == 1.0, "进测试时自动张开一次（夹爪上电是合着的）"
    assert flow.on_key("UP")[0] == ["force_up"]
    assert flow.on_key("LEFT")[0] == ["speed_down"]
    actions, note = flow.on_key("o")
    assert actions == [] and "已经是松开" in note, "自动张开之后再按 O：不发指令，只提示"
    assert flow.on_key(" ")[0] == ["grip_close"] and flow.gripper_command == 0.0
    actions, note = flow.on_key(" ")
    assert actions == [] and "按 O" in note and flow.gripper_command == 0.0, \
        "空格只管夹紧：夹着再按空格不能变成松开（10-20 东西就是这样掉的）"
    assert flow.on_key("O")[0] == ["grip_open"] and flow.gripper_command == 1.0, "大写 O 也算"
    assert not flow.arm_follows and not flow.records
    assert flow.on_key("\n")[0] == ["armed_banner", "help"], "测试完回车只是结束测试"
    assert not flow.records, "结束测试不该开录"
    assert flow.on_key("\n")[0] == ["engage", "begin_record", "help"]
    assert flow.arm_follows and flow.records
    assert flow.on_key(" ")[0] == ["freeze_for_grip", "grip_close"], "录制中空格：先冻住臂再夹"
    assert flow.records, "冻住等夹爪时照常录帧"
    assert flow.on_key(" ")[0] == [], "录制中夹着再按空格：不冻臂、不动夹爪"
    assert flow.on_key("\n")[0] == ["disengage", "pause", "help"] and not flow.records
    for key in (" ", "o"):
        actions, note = flow.on_key(key)
        assert actions == [] and "暂停" in note, f"暂停时 {key!r} 不起作用（动了夹爪数据里会看出中断）"
    assert flow.gripper_command == 0.0, "暂停时按空格 / O 不该改夹爪指令"
    assert "空格、O 不起作用" in flow.help()
    assert flow.on_key("\n")[0] == ["engage", "resume", "help"] and flow.records
    assert flow.on_key("o")[0] == ["freeze_for_grip", "grip_open"], "录制中 O：也是先冻住臂再松"
    assert flow.on_key(" ")[0] == ["freeze_for_grip", "grip_close"]
    assert flow.on_key("S")[0] == ["disengage", "save", "home", "show_count", "help"]
    assert flow.phase == "armed" and flow.gripper_command == 0.0, "存完不改夹爪"
    assert flow.on_key("\n")[0] == ["engage", "begin_record", "help"], "待录时回车直接录下一条"


@check("按键流程：每一步的 help 都列出这一步的键")
def _():
    flow = _flow({"old": dict(count=3, prompt="old prompt", grip_force=64, speed=128)})
    seen = {}
    seen["idle"] = flow.help()
    flow.on_key("p")
    seen["name"] = flow.help()
    _type(flow, "new_one")
    flow.on_key("\n")
    seen["prompt"] = flow.help()
    _type(flow, "x")
    flow.on_key("\n")
    seen["test"] = flow.help()
    flow.on_key("\n")
    seen["armed"] = flow.help()
    flow.on_key("\n")
    seen["recording"] = flow.help()
    flow.on_key("\n")
    seen["paused"] = flow.help()
    flow.on_key("\n")
    flow.on_key("s")
    flow.on_key("p")
    _type(flow, "old")
    flow.on_key("\n")
    seen["confirm"] = flow.help()
    must = {"idle": ("P", "q"), "name": ("回车", "Esc"), "confirm": ("y", "续采", "3 条", "沿用"),
            "prompt": ("回车", "Esc"), "test": ("↑↓", "← →", "空格 夹紧", "O 松开", "回车"),
            "armed": ("回车", "空格 夹紧", "O 松开", "N", "P", "q"),
            "recording": ("回车", "S", "R", "空格 夹紧", "O 松开"),
            "paused": ("回车", "S", "R")}
    for phase, words in must.items():
        for word in words:
            assert word in seen[phase], (phase, word, seen[phase])


@check("按键流程：同名目录（哪怕 0 条）直接问续采；y 续采不再问 prompt，测过力速就直接待录")
def _():
    full = dict(count=5, prompt="pick up the cup", grip_force=80, speed=96)
    flow = _flow({"pick_cup": full, "empty": dict(prompt="p0")})
    flow.on_key("p")
    _type(flow, "pick cup")
    assert flow.on_key("\n")[0] == ["help"] and flow.phase == "confirm"
    actions, note = flow.on_key("n")
    assert flow.phase == "name" and flow.buffer == "" and note, "不按 y 重新输名字"
    _type(flow, "pick_cup")
    flow.on_key("\n")
    actions, _ = flow.on_key("y")
    assert actions == ["open_dataset", "restore_grip", "auto_open", "help"], actions
    assert flow.gripper_command == 1.0, "续采跳过测试也要先张开（用户：不管新采续采，初始都张开）"
    assert flow.on_key(" ")[0] == ["grip_close"], "张开之后空格是夹紧"
    assert flow.on_key("o")[0] == ["grip_open"]
    assert flow.phase == "armed" and flow.continuing and flow.prompt == "pick up the cup"
    assert flow.info["grip_force"] == 80 and flow.info["speed"] == 96
    assert flow.on_key("\n")[0] == ["engage", "begin_record", "help"], "续采后回车直接开录"

    # 目录在、一条没存、记了 prompt 但没测完力度速度（上次测试那步就退了）：只补测试
    flow = _flow({"empty": dict(count=0, prompt="p0")})
    flow.on_key("p")
    _type(flow, "empty")
    assert flow.on_key("\n")[0] == ["help"] and flow.phase == "confirm", "0 条也要问续采"
    assert flow.on_key("y")[0] == ["open_dataset", "auto_open", "help"] and flow.phase == "test"
    assert flow.prompt == "p0"


@check("按键流程：续采但没记下 prompt —— 只补问 prompt")
def _():
    flow = _flow({"bare": dict(count=2)})
    flow.on_key("p")
    _type(flow, "bare")
    flow.on_key("\n")
    actions, note = flow.on_key("y")
    assert flow.phase == "prompt" and note and actions == ["help", "prompt_line"]
    _type(flow, "pick it")
    assert flow.on_key("\n")[0] == ["open_dataset", "auto_open", "help"] and flow.phase == "test"
    assert flow.prompt == "pick it" and flow.continuing


@check("按键流程：输入时 q / s / 空格是字；Esc 取消；空名字、空 prompt 不收")
def _():
    flow = _flow()
    flow.on_key("p")
    for key in ("q", "s", " ", "P"):
        assert flow.on_key(key)[0] == ["echo"], key
    assert flow.input_line().endswith("qs P")
    flow.on_key("\x7f")
    assert flow.input_line().endswith("qs ")
    assert flow.on_key("\x1b")[0] == ["help"] and flow.phase == "idle"
    flow.on_key("p")
    actions, note = flow.on_key("\n")
    assert flow.phase == "name" and note, "空名字不该过"
    _type(flow, "a")
    flow.on_key("\n")
    actions, note = flow.on_key("\n")
    assert flow.phase == "prompt" and note, "空 prompt 不该过"


@check("按键流程：力和速度只在测试时调；录制中 N、P 不响应；待录时 N 删上一条、P 换数据集")
def _():
    flow = _flow()
    flow.on_key("p")
    _type(flow, "t")
    flow.on_key("\n")
    _type(flow, "x")
    flow.on_key("\n")
    flow.on_key("\n")                               # 测试 → 待录
    for key in ("UP", "RIGHT"):
        actions, note = flow.on_key(key)
        assert actions == [] and "测试" in note, (key, note)
    assert flow.on_key("n")[0] == ["ask_delete", "help"] and flow.phase == "delete"
    flow.set_delete_target(None, 0)                 # collect 查到一条都没有
    assert flow.phase == "armed"
    flow.on_key("\n")                               # 开录
    for key in ("n", "p", "UP", "LEFT"):
        assert flow.on_key(key)[0] == [], key
    assert flow.on_key("r")[0] == ["disengage", "discard", "home", "show_count", "help"]
    assert flow.on_key("p")[0] == ["help", "prompt_line"] and flow.phase == "name"
    flow.on_key("\x1b")
    assert flow.on_key("q")[0] == ["disengage", "home", "quit"]


@check("按键流程：待录时 T 回到测试重新调力和速度 —— 只在这个数据集一条都没存时")
def _():
    saved = [0]
    flow = _flow({"water_bottle": dict(count=0, prompt="pick up the water bottle", grip_force=64, speed=144)},
                 saved)
    flow.on_key("p")
    _type(flow, "water_bottle")
    flow.on_key("\n")
    assert flow.on_key("y")[0] == ["open_dataset", "restore_grip", "auto_open", "help"]
    assert flow.phase == "armed", "续采沿用了力和速度，直接待录（这就是 10-23 想调却调不了的场景）"
    assert "T 重新测试" in flow.help()
    actions, note = flow.on_key("UP")
    assert actions == [] and "按 T" in note, "还没存过时，提示按 T"
    assert flow.on_key("T")[0] == ["retest", "help"] and flow.phase == "test"
    assert flow.on_key("UP")[0] == ["force_up"] and flow.on_key("RIGHT")[0] == ["speed_up"]
    assert flow.on_key("\n")[0] == ["armed_banner", "help"], "调完回车：写回 dataset.json、回待录"
    assert flow.on_key("\n")[0] == ["engage", "begin_record", "help"]
    assert flow.on_key("t")[0] == [], "录制中 T 不响应"
    assert flow.on_key("s")[0][:2] == ["disengage", "save"]
    saved[0] = 1                                    # collect 存完把计数加 1
    assert "T 重新测试" not in flow.help()
    actions, note = flow.on_key("t")
    assert actions == [] and flow.phase == "armed" and "已经存了 1 条" in note and "P" in note
    actions, note = flow.on_key("DOWN")
    assert actions == [] and "按 T" not in note


@check("速度：档位 32、48 … 240、255，← → 一档一档走，到头停住，下一条指令用新速度、不打断当前这段")
def _():
    cfg = OmniPickerConfig()
    levels = cfg.speed_levels()
    assert levels[0] == 32 and levels[-1] == 255 and levels[-2] == 240 and len(levels) == 15, levels
    assert all(b - a == 16 for a, b in zip(levels[:-2], levels[1:-1]))
    gripper, bus = make()
    assert gripper.speed == 128
    assert gripper.step_speed(+1) == 144
    assert gripper.step_speed(+20) == 255
    assert gripper.step_speed(-1) == 240
    assert gripper.step_speed(-99) == 32
    gripper.set_speed(100)                       # 不在档位上：先对到最近的 96 再走
    assert gripper.step_speed(+1) == 112
    gripper.set_speed(64)
    gripper.connect()
    try:
        gripper.set_opening(0.5)
        assert wait_for(lambda: len(bus.commands) == 1)
        assert bus.commands[0][1] == 64, bus.commands[0]
        gripper.set_speed(200)
        time.sleep(0.1)
        assert len(bus.commands) == 1, "只改速度不该重发"
        gripper.set_opening(0.0)
        assert wait_for(lambda: len(bus.commands) == 2)
        assert bus.commands[1][1] == 200, bus.commands[1]
    finally:
        gripper.disconnect()


@check("配置拒绝速度低于 32")
def _():
    for bad in (dict(speed=16), dict(speed_min=16, speed=16)):
        try:
            OmniPickerConfig(**bad)
        except ValueError:
            continue
        raise AssertionError(f"没拒绝 {bad}")


@check("按键流程：N 可以连着往前删，每条要 y 确认；y 以外的键（含 q、回车）都是取消")
def _():
    flow = _flow()
    flow.on_key("p")
    _type(flow, "t")
    flow.on_key("\n")
    _type(flow, "x")
    flow.on_key("\n")
    flow.on_key("\n")                               # 待录
    for latest in (3, 2, 1):                        # 连着往前删三条
        assert flow.on_key("n")[0] == ["ask_delete", "help"]
        flow.set_delete_target(latest, latest + 1)
        assert f"episode_{latest:04d}" in flow.help() and "y 删除" in flow.help()
        assert flow.on_key("y")[0] == ["delete_latest", "show_count", "help"]
        assert flow.phase == "armed" and flow.delete_target == (latest, latest + 1), "collect 要读编号"
        flow.delete_target = None                   # collect 删完清掉
    for key in ("q", "\n", " ", "x"):
        flow.on_key("n")
        flow.set_delete_target(0, 1)
        actions, note = flow.on_key(key)
        assert actions == ["help"] and note == "没删" and flow.phase == "armed", key
        assert flow.delete_target is None


@check("等夹爪：合拢等「堵转」（位置停了但力没建起来不算）或空夹到底；张开等到位；都至少 0.5s、最多 4s")
def _():
    from .keyflow import GripWait
    from .protocol import STATE_ARRIVED, STATE_BLOCKED, STATE_MOVING
    assert GripWait.BLOCKED == STATE_BLOCKED

    # 夹套筒（10-20 实测的顺序）：一路合拢 → 碰到停在 37 → 约 0.5s 后才报堵转
    wait = GripWait(0.0, 0, closing=True)
    assert wait.update(0.1, 210, STATE_ARRIVED) is None, "按键前的旧状态不能信"
    pos, t = 210, 0.3
    while pos > 40:
        t += 0.02
        pos -= 6
        assert wait.update(t, pos, STATE_MOVING) is None, (t, pos)
    t_touch = t
    for i in range(25):                          # 位置停了 0.5s，状态还是「运动中」
        t += 0.02
        assert wait.update(t, 37 + (i % 2), STATE_MOVING) is None, \
            "碰到就放臂 = 力没建起来套筒被带歪（原来位置 0.25s 不变就放）"
    assert t - t_touch >= 0.48
    assert wait.update(t + 0.02, 37, STATE_BLOCKED) == "gripped"

    # 空夹：合到底、状态「到达」
    empty = GripWait(0.0, 0, closing=True)
    assert empty.update(0.4, 1, STATE_ARRIVED) is None, "0.5s 之前都不放"
    assert empty.update(0.6, 1, STATE_ARRIVED) == "arrived"

    # 张开：堵转不算（张开碰不到东西，读到的是夹着时的旧状态），到 210 才放
    opening = GripWait(0.0, 210, closing=False)
    assert opening.update(0.6, 37, STATE_BLOCKED) is None
    assert opening.update(0.9, 150, STATE_MOVING) is None
    assert opening.update(1.2, 208, STATE_ARRIVED) == "arrived"

    # 夹在半路既不报堵转也到不了底（卡住之类）：4s 超时
    stuck = GripWait(0.0, 0, closing=True)
    r, t = None, 0.0
    while r is None:
        t += 0.02
        r = stuck.update(t, 60, STATE_MOVING)
    assert r == "timeout" and 3.98 <= t <= 4.04, (r, t)


@check("开合读数带读取时刻（perf_counter 的钟），开合和时刻是同一次读的")
def _():
    gripper, bus = make(FakeBus(pos=105))
    gripper.connect()
    try:
        assert wait_for(lambda: gripper.get_opening_stamped()[1] is not None)
        opening, stamp = gripper.get_opening_stamped()
        assert abs(opening - 0.5) < 0.01, opening
        assert 0 <= time.perf_counter() - stamp < 0.05, "100 Hz 读写，读数最多旧一个周期再加一点"
        # 读数缓存：每圈取走新来的，不漏不重
        got, after = [], None
        for _ in range(10):
            new = gripper.openings_since(after)
            if new:
                after = new[-1][0]
                got.extend(t for t, _ in new)
            time.sleep(0.03)
        assert len(got) >= 15 and all(b > a for a, b in zip(got, got[1:])), len(got)
    finally:
        gripper.disconnect()


@check("暂停后等停稳：滑行这段照录、连续 2 帧不动才停；停着暂停只多录 2 帧；夹爪还在动不算停；最多 2s")
def _():
    import math
    from .keyflow import SettleWait
    rad = math.radians

    def frame(q0, deg, opening=1.0):
        return [q0 + rad(deg)] + [0.3] * 6 + [opening]

    # 10-20 black_sleeve 的接缝：暂停前每帧 0.7°，暂停后臂一路减速滑到最后一个目标
    slide = [0.7, 0.45, 0.25, 0.12, 0.05, 0.02, 0.008, 0.0003, 0.0002, 0.0001]
    wait = SettleWait(0.0, frame(0.0, 0.0))
    angle, t, got = 0.0, 0.0, []
    for step in slide:
        angle += step
        t += 1 / 30
        got.append(wait.update(t, frame(0.0, angle)))
        if got[-1]:
            break
    assert got[-1] == "settled" and wait.frames == 9, (got, wait.frames)
    assert got[:8] == [None] * 8, "还在滑（哪怕每帧只有 0.008°）就要接着录"

    # 臂本来就停着时暂停：拿暂停前最后一帧当起点，只多录 2 帧
    still = SettleWait(0.0, frame(0.0, 5.0))
    assert still.update(0.033, frame(0.0, 5.0)) is None
    assert still.update(0.067, frame(0.0, 5.0)) == "settled"

    # 臂停了、夹爪还在合（录制中夹紧没等完就暂停）：不算停稳
    gripping = SettleWait(0.0, frame(0.0, 0.0, 0.5))
    results = [gripping.update(k / 30, frame(0.0, 0.0, 0.5 - 0.02 * k)) for k in range(1, 6)]
    assert results == [None] * 5, results

    # 一直在动：2s 超时，也停
    moving = SettleWait(0.0)
    r, t, a = None, 0.0, 0.0
    while r is None:
        t += 1 / 30
        a += 0.1
        r = moving.update(t, frame(0.0, a))
    assert r == "timeout" and 1.99 <= t <= 2.04, (r, t)


def main():
    failed = 0
    for name, fn in _results:
        try:
            fn()
        except Exception:                                   # noqa: BLE001
            failed += 1
            print(f"✗ {name}")
            traceback.print_exc()
        else:
            print(f"✓ {name}")
    print(f"\n{len(_results) - failed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
