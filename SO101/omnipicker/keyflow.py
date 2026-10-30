"""OmniPicker 采集的按键流程（2026-10-20 用户定的）。纯状态机，不碰硬件、不读时钟、不碰磁盘。

**P 开始的是一个数据集，不是一条录制。** 每个数据集一个目录（任务名），整个数据集共用一句
语言 prompt、一组夹持力和速度。**主臂扳机全程不起作用**，夹爪只听键盘：**空格夹紧、O 松开**。

    待机 ─P─▶ 输任务名 ─回车─▶ 输语言prompt ─回车─▶ 测试力度速度
         ─回车─▶ 待录（「开始采集数据集，按回车开始采集：」）─回车─▶ 录制 ⇄ 暂停（回车）
    录制 / 暂停 ─S 存盘 / R 丢弃，臂回初始位置─▶ 待录 ─回车─▶ 下一条 …
    待录 ─P─▶ 输任务名（换下一个数据集）        任何时候 q 退出（输入文字时 q 是字，先 Esc）

    续采：任务名和已有目录同名（哪怕一条都还没存）就问「是否要续采」，y 续采 ——
    prompt、力和速度都沿用 dataset.json 里的，**不再问**，直接到待录。哪样没记下（上次没
    测完力度速度就退了）就只补问那一步。

    不管新采还是续采，开始时（进测试，或续采直接进待录时）夹爪都**自动张开一次**。

    测试    进来时夹爪自动张开一次；↑↓ 夹持力、← → 速度、空格 夹紧、O 松开 —— 只有这一步能调力和速度
    待录    空格 夹紧、O 松开（摆现场用）、N 删掉最后一条存盘的（y 确认，可以连着往前删）、
            T 回到测试重新调力和速度（只在这个数据集一条都没存时，用户 10-23）
    录制    空格 夹紧 / O 松开 —— 臂先冻住、只动夹爪，夹牢（报夹住或合到底）/ 张开到位后臂自动接着跟（照常录帧）；
            暂停时臂冻住、**不录帧**（臂滑到停稳那一小段照录，见 SettleWait），回车从主臂当前位置
            接着跟、接着录；**暂停时空格、O 不起作用**（动了夹爪，接着录时开合会跳一下，数据里看得出中断）

每进一步都要把这一步的按键打印出来 —— 动作里带 "help"，collect 打印 `help()`。
任务名查重要看磁盘，所以构造时传一个 `probe(name) -> dict`：
    error       不合法时的说明（None = 合法）
    exists      这个目录已经在了（一条没存也算）
    count       已有几条
    prompt      以前用的语言 prompt（读不到就 None）
    grip_force  以前测好的夹持力、速度（没测完就 None）
    speed
"""
import math

IDLE, NAME, CONFIRM, PROMPT, TEST, ARMED, RECORDING, PAUSED, DELETE = (
    "idle", "name", "confirm", "prompt", "test", "armed", "recording", "paused", "delete")
TEXT_PHASES = (NAME, PROMPT)
ENTER_KEYS = ("\n", "\r")
BACKSPACE_KEYS = ("\x7f", "\x08")
ESC = "\x1b"


class KeyFlow:
    def __init__(self, probe, saved_count=lambda: 0):
        self.probe = probe
        # 当前数据集磁盘上已经存了几条（collect 给）。T 回测试只在一条都没存时允许 —— 一个数据集一组力和速度
        self.saved_count = saved_count
        self.phase = IDLE
        self.grip = None            # None = 还没按过空格；"closed" / "open"
        self.buffer = ""            # 正在输入的文字
        self.name = None            # 当前数据集的任务名（= 目录名）
        self.prompt = None          # 当前数据集的语言 prompt
        self.continuing = False     # 是不是续采
        self.info = None            # probe 的结果：确认续采时显示，续采时从这里沿用 prompt / 力 / 速度
        self.delete_target = None   # N 之后等 y 确认时：(要删的编号, 磁盘上共几条)，collect 填

    # ── 给 collect 读的状态 ──────────────────────────────────────────

    @property
    def gripper_command(self):
        """给夹爪的开合：None = 不发指令（原地不动），0.0 = 夹紧，1.0 = 张开。"""
        return None if self.grip is None else (0.0 if self.grip == "closed" else 1.0)

    @property
    def arm_follows(self) -> bool:
        return self.phase == RECORDING

    @property
    def records(self) -> bool:
        """这一帧要不要录。暂停时不录。"""
        return self.phase == RECORDING

    @property
    def text_input(self) -> bool:
        """正在输入文字：状态行别刷，免得盖掉输入的那一行。"""
        return self.phase in TEXT_PHASES

    def input_line(self) -> str:
        label = "任务名称是:" if self.phase == NAME else "语言prompt是:"
        return f"  {label} {self.buffer}"

    def help(self) -> str:
        """这一步能按什么键。"""
        if self.phase == IDLE:
            return "【待机】P 开始一个新数据集    q 退出"
        if self.phase == NAME:
            return "【第 1 步 任务名】输入任务名称（也是目录名，空格会换成 _），回车确认    退格 删字    Esc 取消"
        if self.phase == CONFIRM:
            info = self.info or {}
            kept = [f"语言prompt「{info['prompt']}」"] if info.get("prompt") else []
            if self._kept_grip():
                kept.append(f"夹持力 {info['grip_force']}、速度 {info['speed']}")
            kept = "；沿用" + "、".join(kept) if kept else ""
            return (f"已经有这个数据集「{self.buffer_name}」（已存盘 {info.get('count', 0)} 条{kept}），"
                    "是否要续采？    y 续采    其他键 重新输入任务名")
        if self.phase == PROMPT:
            return "【第 2 步 语言prompt】输入这个数据集的语言指令，回车确认    退格 删字    Esc 取消"
        if self.phase == TEST:
            return ("【第 3 步 测试力度和速度】↑↓ 夹持力    ← → 速度    空格 夹紧    O 松开    "
                    "测试完按回车    q 退出")
        if self.phase == ARMED:
            retest = "    T 重新测试力度和速度" if self.saved_count() == 0 else ""
            return ("【采集】回车 开始录这一条    空格 夹紧    O 松开    N 删掉最后一条存盘的（可以连着往前删，"
                    f"每条按 y 确认）{retest}    P 换下一个数据集    q 退出")
        if self.phase == DELETE:
            index, total = self.delete_target or (0, 0)
            return (f"要删掉「{self.name}」的 episode_{index:04d} 吗（磁盘上共 {total} 条，删了找不回来）？"
                    "    y 删除    其他键 取消")
        if self.phase == RECORDING:
            return ("【录制中】回车 暂停    空格 夹紧    O 松开（都是臂先冻住，夹牢 / 张开到位后自动接着跟）    "
                    "S 存盘+回初始位置    R 丢掉这条+回初始位置    q 退出")
        return ("【暂停】臂冻住、不录帧    回车 从主臂当前位置接着录    "
                "S 存盘+回初始位置    R 丢掉这条+回初始位置    q 退出    （暂停时空格、O 不起作用）")

    # ── collect 在按键之外推进状态 ────────────────────────────────────

    def to_armed(self):
        """存完 / 丢完 / 到帧数上限：回到待录。夹爪状态保留 —— 不自动松开。"""
        self.phase = ARMED

    def dataset_done(self):
        """这个数据集采够了：回待机，等 P 开下一个。"""
        self.phase = IDLE
        self.name = self.prompt = None

    def set_delete_target(self, index, total):
        """N 之后 collect 查磁盘：有可删的就填进来等 y；一条都没有就传 None，回待录。"""
        if index is None:
            self.phase, self.delete_target = ARMED, None
        else:
            self.delete_target = (int(index), int(total))

    # ── 按键 ─────────────────────────────────────────────────────────

    def on_key(self, key):
        """返回 (动作列表, 提示)。动作按顺序执行；提示为 None 表示不用打印。

        动作：help（打印这一步的按键）、prompt_line（新起一行显示输入行）、echo（重画输入行）、
        open_dataset（name / prompt / continuing 已就绪）、restore_grip（续采：力和速度设回
        info 里的，直接待录）、auto_open（进测试时夹爪自动张开一次）、freeze_for_grip（录制中
        按空格 / O：先冻住臂，夹牢 / 张开到位再接着跟）、force_up/down、speed_up/down、
        grip_close/open、engage、disengage、begin_record、pause、resume、save、discard、
        home、ask_delete（N：collect 查磁盘后调 set_delete_target）、delete_latest（y 确认后删
        编号最大的那条）、show_count、armed_banner、quit。
        """
        if key is None:
            return [], None
        if self.phase in TEXT_PHASES:
            return self._text_key(key)
        k = key.lower() if len(key) == 1 else key

        if self.phase == CONFIRM:
            if k == "y":
                self.continuing = True
                self.name, self.buffer = self.buffer_name, ""
                self.prompt = (self.info or {}).get("prompt")
                if self.prompt:
                    return self._opened()
                self.phase = PROMPT             # 目录在、但没记下 prompt：只能问
                return ["help", "prompt_line"], "这个数据集没记下语言prompt，输一次"
            self.buffer, self.phase = "", NAME
            return ["help", "prompt_line"], "重新输入任务名"

        if self.phase == DELETE:
            # 只有 y 才删；q、回车、空格这些在这里都只是「取消」，免得手滑
            self.phase = ARMED
            if k == "y" and self.delete_target is not None:
                return ["delete_latest", "show_count", "help"], None
            self.delete_target = None
            return ["help"], "没删"

        if k == "q":
            return ["disengage", "home", "quit"], "q 退出，先回初始位置……"

        if self.phase == IDLE:
            if k == "p":
                self._start_name()
                return ["help", "prompt_line"], None
            return [], "按 P 开始一个新数据集，q 退出"

        if k in ("UP", "DOWN", "LEFT", "RIGHT"):
            if self.phase == TEST:
                return [{"UP": "force_up", "DOWN": "force_down",
                         "LEFT": "speed_down", "RIGHT": "speed_up"}[k]], None
            if self.phase == ARMED and self.saved_count() == 0:
                return [], "力和速度在测试那一步调：这个数据集还没存过，按 T 回去测试"
            return [], "力和速度只在测试那一步调，整个数据集都用那一组；要换就按 P 开新数据集"

        if k in (" ", "o"):
            # **空格只管夹紧、O 只管松开**（用户 10-20）。原来空格是夹 / 松交替：夹在物体上时
            # 夹爪停在半合，看着像没夹紧，再按空格想夹 —— 实际是松开，东西就掉了。
            want = "closed" if k == " " else "open"
            if self.phase == PAUSED:
                # 暂停不录帧，这时动夹爪，接着录时前后两帧开合就跳一下 —— 数据里能看出中断（用户 10-20）
                return [], "暂停时不能动夹爪（接着录会留下开合跳变）。回车接着录，录制中再按"
            if self.grip == want:
                return [], ("已经是夹紧了（要松开按 O）" if want == "closed"
                            else "已经是松开了（要夹紧按空格）")
            self.grip = want
            action = "grip_close" if want == "closed" else "grip_open"
            if self.phase == RECORDING:
                # 录制中：先冻住臂、只动夹爪，夹牢 / 张开到位后 collect 让臂自动接着跟（用户 10-20）。
                # 照常录帧 —— 录下来就是「停下、夹、再走」，没有断开的地方。
                return ["freeze_for_grip", action], None
            return [action], None

        if k in ENTER_KEYS:
            if self.phase == TEST:
                self.phase = ARMED
                return ["armed_banner", "help"], None
            if self.phase == ARMED:
                self.phase = RECORDING
                return ["engage", "begin_record", "help"], None
            if self.phase == RECORDING:
                self.phase = PAUSED
                return ["disengage", "pause", "help"], None
            self.phase = RECORDING
            return ["engage", "resume", "help"], None

        recording = self.phase in (RECORDING, PAUSED)
        if k == "s":
            if not recording:
                return [], "现在没在录，回车开始录这一条" if self.phase == ARMED else "测试完按回车"
            self.phase = ARMED
            return ["disengage", "save", "home", "show_count", "help"], None
        if k == "r":
            if not recording:
                return [], "现在没在录，没东西可丢"
            self.phase = ARMED
            return ["disengage", "discard", "home", "show_count", "help"], None
        if k == "n":
            if self.phase == ARMED:
                # 删的是磁盘上这个数据集**编号最大**的那条（以前几次运行存的也算），连按可以一直往前删
                # （用户 10-20）。每条都要 y 确认：删了找不回来。
                self.phase, self.delete_target = DELETE, None
                return ["ask_delete", "help"], None
            return [], "录制中 N 不动已存的数据，要丢这一条按 R" if recording else "N 在待录时用"
        if k == "t":
            # 待录时回到「测试力度和速度」（用户 10-23）：续采会沿用 dataset.json 里记的力和速度、跳过测试，
            # 开好了数据集才发现要调就没地方调。**只在一条都没存时**允许 —— 一个数据集一组力和速度
            if self.phase != ARMED:
                return [], "录制中不能调力和速度" if recording else None
            saved = self.saved_count()
            if saved:
                return [], (f"这个数据集已经存了 {saved} 条，力和速度要整个数据集一致，不能再调；"
                            "要换一组就按 P 开新数据集")
            self.phase = TEST
            return ["retest", "help"], None
        if k == "p":
            if self.phase == ARMED:
                self._start_name()
                return ["help", "prompt_line"], None
            return [], "录制中不能换数据集，先 S 存盘或 R 丢掉" if recording else "测试完按回车"
        return [], None

    def _start_name(self):
        self.phase, self.buffer = NAME, ""
        self.name = self.prompt = self.info = None
        self.continuing = False

    def _kept_grip(self) -> bool:
        info = self.info or {}
        return info.get("grip_force") is not None and info.get("speed") is not None

    def _opened(self):
        """任务名和 prompt 都有了。续采且以前测过力度速度就沿用、直接待录，否则去测试。

        **不管新采还是续采，开始时夹爪都自动张开一次**（用户 10-20）：夹爪上电会自己合拢找零位，
        程序又不主动动它，于是一开始总是合着的，而第一下空格是「夹紧」—— 看不出变化，要按两下
        才张开。张开之后夹爪记成「松开」，下一下空格就是夹紧。续采跳过测试时先把力和速度
        恢复成这个数据集的，再用那个速度张开。"""
        self.grip = "open"
        if self.continuing and self._kept_grip():
            self.phase = ARMED
            return ["open_dataset", "restore_grip", "auto_open", "help"], None
        self.phase = TEST
        return ["open_dataset", "auto_open", "help"], None

    @property
    def buffer_name(self) -> str:
        """输入的任务名规整成目录名：去两头空白，中间的空白换成 _。"""
        return "_".join(self.buffer.split())

    def _text_key(self, key):
        if key == ESC:
            self.phase, self.buffer = IDLE, ""
            return ["help"], "取消了，这个数据集没开"
        if key in BACKSPACE_KEYS:
            self.buffer = self.buffer[:-1]
            return ["echo"], None
        if key in ENTER_KEYS:
            return self._submit()
        if len(key) == 1 and key.isprintable():
            self.buffer += key
            return ["echo"], None
        return [], None              # 方向键、控制字符：输入时不管

    def _submit(self):
        if self.phase == NAME:
            name = self.buffer_name
            info = self.probe(name)
            if info.get("error"):
                self.buffer = ""
                return ["prompt_line"], info["error"]
            if info.get("exists"):
                self.info, self.phase = info, CONFIRM
                self.buffer = name
                return ["help"], None
            self.name, self.buffer, self.phase = name, "", PROMPT
            return ["help", "prompt_line"], None
        text = " ".join(self.buffer.split())
        if not text:
            return ["prompt_line"], "语言prompt 不能为空"
        self.prompt, self.buffer = text, ""
        return self._opened()


class GripWait:
    """录制中按空格 / O、臂冻住之后，判断夹爪「夹牢了 / 张开到位了」没有。纯逻辑，collect 每个回路喂一次。

    **合拢**：等夹爪报「堵转」（= 夹住，夹持力已经建起来），或者空夹合到了底（离目标 ≤3）。
    **张开**：等位置到目标（≤3）。
    原来是「位置 0.25 s 不变就算动完」—— 10-20 实测夹爪刚碰到套筒位置就不变了，力还没建起来
    （约 0.5 s 后才报夹住），臂这时就开始跟主臂，套筒被带歪没夹牢（用户：「没有夹」）。
    状态寄存器比位置晚约 0.2 s，所以至少等 MIN_SECONDS 再信它，免得读到按键之前的旧状态。
    """
    MIN_SECONDS = 0.5       # 指令发出后夹爪要 0.1~0.3 s 才开始动；状态又晚 0.2 s
    ARRIVED = 3             # 离目标这么近就算到了
    TIMEOUT_SECONDS = 4.0   # 最多冻这么久，到点也放开臂（collect 打一句提示）
    BLOCKED = 2             # 状态寄存器的「堵转」（protocol.STATE_BLOCKED）

    def __init__(self, t0: float, target: int, closing: bool):
        self.t0 = t0
        self.target = int(target)
        self.closing = bool(closing)

    def update(self, t: float, pos: int, state: int):
        """返回 None（还在等）、"gripped"（夹住了）、"arrived"（到位了）或 "timeout"。"""
        elapsed = t - self.t0
        if elapsed >= self.TIMEOUT_SECONDS:
            return "timeout"
        if elapsed < self.MIN_SECONDS:
            return None
        if self.closing and state == self.BLOCKED:
            return "gripped"
        if abs(pos - self.target) <= self.ARRIVED:
            return "arrived"
        return None


class SettleWait:
    """暂停（录制中回车）之后，等臂和夹爪真的停稳了才不录帧。纯逻辑，collect 每录一帧喂一次。

    暂停时臂的目标停在最后一条指令上，实测位置却落后它一截（跟踪滞后，动得快时更多）。
    原来一按暂停就不录，这一截滑行没录进去，接着录的第一帧臂已经在滑完的地方 —— 10-20 实测
    black_sleeve 每处接缝一帧跳 8~33 mm（平时每帧 1.5 mm）、关节 2.5~4.6°，视频跳一下，
    action 也是一个离群的大增量（用户：「视频还是有点跳变的感觉」）。
    现在暂停后照录，连续 STILL_FRAMES 帧关节每帧变化都 < STILL_DEG、开合不变才停。
    停着的臂每帧变化 ≤0.0003°，动着的 ≥0.1°。暂停时夹爪可能还在合（录制中夹紧没等完就暂停），
    所以开合也要停。最多等 TIMEOUT_SECONDS。
    """
    STILL_DEG = 0.005
    STILL_OPENING = 0.004   # 开合 0~1 对应位置 0~210，一个单位约 0.0048
    STILL_FRAMES = 2
    TIMEOUT_SECONDS = 2.0

    def __init__(self, t0: float, last_qpos=None):
        """last_qpos：暂停前最后录下的那帧（7 个关节弧度 + 开合），臂本来就停着时少录几帧。"""
        self.t0 = t0
        self._prev = None if last_qpos is None else [float(v) for v in last_qpos]
        self._still = 0
        self.frames = 0

    def update(self, t: float, qpos) -> str:
        """喂刚录下的一帧。返回 None（还在滑，接着录）、"settled"（停稳了）或 "timeout"。"""
        q = [float(v) for v in qpos]
        self.frames += 1
        if self._prev is not None:
            joints = max(abs(a - b) for a, b in zip(q[:-1], self._prev[:-1]))
            still = (math.degrees(joints) < self.STILL_DEG
                     and abs(q[-1] - self._prev[-1]) < self.STILL_OPENING)
            self._still = self._still + 1 if still else 0
        self._prev = q
        if self._still >= self.STILL_FRAMES:
            return "settled"
        if t - self.t0 >= self.TIMEOUT_SECONDS:
            return "timeout"
        return None
