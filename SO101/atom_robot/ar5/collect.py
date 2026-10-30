#!/usr/bin/env python3
"""遥操作采数据。每个 episode 存成一个 .npz + 每路相机一个 .mp4。

    python -m atom_robot.ar5.collect --show-config    # 先看参数，不碰硬件
    python -m atom_robot.ar5.collect
    python -m atom_robot.ar5.collect --task "换个任务的描述"

**参数都在 `config.py` 的 `CollectConfig` 里**，带注释，改了下次开机还在。
命令行参数只覆盖这一次，用来试，不用来记。每次启动都会把真正生效的值打出来。

按键：
    空格   开始录这一条（同时接合，从臂开始跟主臂）
    P      **保持 / 重新锚定**（只在录制中有效）。按一下臂原地不动、主臂随便摆；
           再按一下从主臂的新位置接着跟。**主臂行程不够时用它**，跟抬鼠标一样
    s      **这段做完了**：存盘 + 松开 + 臂回 0 位
    r      丢掉正在录的这条，重来。编号不变。臂也回 0 位
    ←      **删掉刚存的那条**。只删最近的一条，连按不会往回删
    q      退出。**正在录的那条不保存** —— 要留就先按 s

**`--end-effector omnipicker` 按数据集采、用另一套键**（2026-10-20 用户定的，状态机在
omnipicker/keyflow.py，每一步能按的键会在那一步打出来）：

    P        开一个数据集：输任务名（= `~/atom_episodes/new/任务名/`）→ 输语言 prompt。
             目录已经在（一条没存也算）就问要不要续采，y 续采：prompt、力和速度都沿用
             dataset.json 里的，不再问、不再测，直接到待录
    测试     ↑↓ 夹持力、← → 速度、空格 夹紧、O 松开；**力和速度只在这一步调，整个数据集用这一组**
             测试完按回车，打出「开始采集数据集，按回车开始采集：」
    回车     开始录（臂开始跟）→ 再按暂停（臂冻住、**不录帧**）→ 再按从主臂当前位置接着跟、接着录
    S / R    存盘 / 丢掉这条，臂回 0 位，回到待录；回车直接录下一条
    N        待录时删掉上一条存盘的        P  待录时换下一个数据集        q  退出

**主臂扳机全程不起作用**，夹爪只听键盘：空格夹紧、O 松开；存盘、丢弃之后夹爪停在原地，不自动松开。
数据集目录里另有 dataset.json：任务名、语言 prompt、这个数据集用的夹持力和速度。
这套键还没接到网页上，所以 OmniPicker 加 `--remote` 会直接拒绝启动。
航凯夹爪和因时手还是上面那套键，离合只管臂，夹爪 / 手一启动就跟着扳机跑。

加 `--remote` 就能在网页上按这几个键（本地键盘照常有效）：命令从 `{out}/control.json`
读，画面写 `{out}/live.jpg`，状态写 `{out}/status.json`，由工作站代理转给网页。
工作站上不用有人守着终端 —— 也是**被代理拉起来时唯一能用的方式**：那种情况下
stdin 不是终端，压根读不了按键。

**录着就是接合、不录就是松开**，两件事同进同出：不录的时候从臂跟着主臂动，
只会让你摆现场时臂在旁边乱晃，以及下一条开录时臂不在起点。

**P 是录制中的离合**，解决的是另一个问题：主臂行程不够。臂要往一个方向走很远时，
主臂先到头了 —— 按 P 把臂停住、把主臂收回来、再按 P，臂就从新的主臂位置接着往前。
保持期间臂原地不动、**夹爪也冻住**，这几帧在 npz 里逐帧标进 `held`。

**s / r / q 三个出口臂都回 0 位**（`centre_joints` = 主臂全零对应的姿势），
这样每条的起点都一样。← 不回 —— 它只删磁盘文件，臂在哪儿不该受影响。

第四个出口是**臂自己抱死**（latch）。它和上面三个不一样：那时 RT 流已经死了、
控制器已经掉电，回不了位，只能先上电再开新会话。所以收尾时会**问你一句**，
按 y 才动臂；不答、答别的、或者根本没有终端（网页拉起来的那种）就不动，
只打一条 `unseize --recover --home` 让你自己救。

整个过程的控制台输出留一份在 `{out}/collect.log` —— 抱死之后要查的就是它。

**每条都手动起停，存完不会自动开下一条。** 存完臂自己回初始位置，你摆好现场
再按空格。复位时间想用多久用多久，没有倒计时。

存完用 to_lerobot.py 转成 lerobot 数据集（在 lerobot 环境里跑）。
分两步是因为这个环境的 torch / TensorFlow 版本是配死的，装 lerobot 会动到
它们；而转换那步不碰硬件，放到别的环境里最省事。
"""

from __future__ import annotations

import argparse
import codecs
import json
import os
import select
import sys
import termios
import threading
import time
import tty
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from atom_robot.ar5.ar5_env import AR5Env  # noqa: E402
from atom_robot.ar5.config import AR5Config  # noqa: E402
from atom_robot.ar5.joint_mapping import (  # noqa: E402
    JOINT_PAIRS,
    JointMapConfig,
    JointMapper,
)
from atom_robot.ar5.leader_bus import SO_JOINTS, LeaderBus  # noqa: E402
from atom_robot.ar5.teleop_joint import hold_posture  # noqa: E402
from atom_robot.ar5 import sync_align  # noqa: E402
from atom_robot.ar5.transforms import pose_delta  # noqa: E402

# 主臂各关节的半行程从标定文件里读（和 teleop_joint 一样）。这里原来是一份
# 写死的字典，数值当时是对的 —— 但换主臂或者重新标定之后它不会跟着变，
# 采出来的数据就会和遥操作时的手感悄悄不一致。


# 方向键不是一个字节，是 ESC [ 后面跟一个字母。一次 read(1) 只会拿到 ESC。
_ARROWS = {"\x1b[A": "UP", "\x1b[B": "DOWN", "\x1b[C": "RIGHT", "\x1b[D": "LEFT"}


# 网页按钮 → 本脚本的按键。**必须和文件顶部「按键：」那一段逐条对上**：
# 改键位时两边一起改，否则网页上点的和脚本执行的是两回事。
REMOTE_KEYS = {
    "start":   " ",        # 空格  开录并接合
    "save":    "s",        # s     存盘 + 松开 + 臂回初始位置
    "discard": "r",        # r     丢掉正在录的这条，编号不变
    "undo":    "LEFT",     # ←     删掉刚存的那条
    "quit":    "q",        # q     退出，正在录的那条不保存
    # P 在这里是一个键来回切，网页上却是「暂停 / 继续」两个按钮，所以两条都落到 p。
    # 网页显示的保持状态以脚本发出去的 holding 为准，不靠自己数点了几次 ——
    # 否则有人在工作站键盘上按一下 P，两边就反过来了。
    "pause":   "p",        # p     保持：臂原地不动，主臂随便摆
    "resume":  "p",        # p     再按一下：从主臂新位置接着跟
    # 网页上还没有这两个按钮（网页在开发中心那边改），先把命令名定下来；
    # 当前值在 status.json 的 gripper_force 里。
    "force_up":   "UP",    # ↑     夹爪夹持力 +1 档
    "force_down": "DOWN",  # ↓     夹爪夹持力 -1 档
}


class RawKeyboard:
    """非阻塞单键读取，退出时恢复终端。

    直接读裸 fd（`os.read`），**不走 `sys.stdin`**。走 sys.stdin 的话，
    `select` 看的是文件描述符、而 `read()` 走的是 Python 的 TextIOWrapper
    用户态缓冲：读一个字符会把整串字节从内核拽进缓冲区，fd 随即变空，
    `select` 就报「没数据」，剩下的字节再也取不出来。

    对单字节的键没影响，所以空格 / s / q 一直是好的；但方向键是 ESC [ D
    三个字节，结果只拿到一个裸 ESC —— 左键完全没反应就是这么来的。
    """

    def __enter__(self):
        self._pending = ""
        # 增量解码：OmniPicker 要在这里输中文任务名 / prompt，一个汉字 3 个字节，
        # 一次 os.read 正好切在中间的话，整段 decode 会把它变成乱码。
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self.fd = None
        try:
            fd = sys.stdin.fileno()
            self.saved = termios.tcgetattr(fd)      # 不是终端就在这里抛
            tty.setcbreak(fd)
            self.fd = fd
        except Exception:                           # noqa: BLE001
            # 被工作站代理 / systemd 拉起来时 stdin 不是终端，tcgetattr 直接抛，
            # 以前这会让整个采集起不来。现在退化成「永远没有按键」：
            # 加 --remote 时命令走 control.json，照样能采。
            print("  提示：stdin 不是终端，本地键盘读不了 —— 请用 --remote 从网页操作。")
        return self

    def __exit__(self, *exc):
        if self.fd is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.saved)

    def key(self):
        """一次按键。方向键返回 "LEFT" / "RIGHT" / "UP" / "DOWN"。

        一次 os.read 可能带回好几个键（手快，或者按住不放），所以多出来的
        存在 _pending 里下次再给，不丢。
        """
        if self.fd is None:              # 没有终端可读，见 __enter__
            return None
        if not self._pending:
            if not select.select([self.fd], [], [], 0)[0]:
                return None
            try:
                data = os.read(self.fd, 32)
            except OSError:
                return None
            if not data:
                return None
            decoder = getattr(self, "_decoder", None)
            if decoder is None:          # 没走 __enter__ 造出来的（自检里就是）
                decoder = self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
            self._pending = decoder.decode(data)
            if not self._pending:        # 半个汉字，等下一次读到后半截
                return None

        # 方向键：ESC [ 后面一个字母。只有确实拿到三个字节才当方向键，
        # 否则单独按 Esc 会把后面第一个字符吃掉。
        if self._pending.startswith("\x1b[") and len(self._pending) >= 3:
            seq, self._pending = self._pending[:3], self._pending[3:]
            return _ARROWS.get(seq, seq)

        char, self._pending = self._pending[0], self._pending[1:]
        return char


class _Tee:
    """写终端的同时留一份到日志文件。终端那份一个字节都不改。

    **状态行不落盘。** 采集时那根 `● ep3  12.3s  接合  手 0.42` 是靠 `\\r`
    原地刷新的，约 5 Hz；原样写进文件的话一条 episode 就是几千行回车垃圾，
    真正要查的那几行反而找不着。只有它以 `\\r` 开头，所以判据就用这个。
    """

    def __init__(self, stream, sink):
        self._stream = stream
        self._sink = sink

    def write(self, data):
        n = self._stream.write(data)
        if not data.startswith("\r"):
            try:
                self._sink.write(data)
                self._sink.flush()   # 抱死时进程常是被 kill 的，不刷就什么都不剩
            except Exception:        # noqa: BLE001
                pass                 # 日志写不了也不能把采集带下去
        return n

    def flush(self):
        self._stream.flush()

    def __getattr__(self, name):     # fileno / isatty / encoding 都转给真的那个
        return getattr(self._stream, name)


def start_session_log(out_dir: Path) -> Path:
    """把控制台输出和 logging 都留一份到 `out_dir/collect.log`。

    2026-09-23 加的。抱死之后真正有用的那几行 —— 驱动的
    `goals stopped arriving ... braking and latching`、回路的
    `✗ 臂已刹车锁死` —— 原来只往 stdout/stderr 打，进程一退就没了。
    剩下能查的只有 SDK 自己那份日志，而**它区分不了「latch」和「人按了 q」**：
    两条路在日志里都是 stopMove → NrtCommand → disconnect，一模一样。
    """
    import logging                                           # noqa: PLC0415

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "collect.log"
    sink = path.open("a", encoding="utf-8", buffering=1)
    sink.write("\n===== %s  pid %d  %s =====\n"
               % (time.strftime("%Y-%m-%d %H:%M:%S"), os.getpid(), " ".join(sys.argv)))
    # 单独挂一个 handler 写 sink，而不是指望下面的 stderr 替换：basicConfig 建的
    # 那个 StreamHandler 在构造时就把当时的 sys.stderr 存住了，之后再换
    # sys.stderr 它也不会跟着走。两个 handler 各写各的，终端和文件都不缺。
    handler = logging.StreamHandler(sink)
    handler.setFormatter(logging.Formatter("  %(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(handler)
    sys.stdout = _Tee(sys.stdout, sink)
    sys.stderr = _Tee(sys.stderr, sink)
    return path


def confirm_keypress(prompt: str, timeout_s: float = 60.0) -> bool:
    """等一个按键：`y` = 是，别的键、超时、没有终端都算否。

    走裸 fd + select，理由和 `RawKeyboard` 那边一样（见它的 docstring）。
    **超时是必须的**：网页拉起来的那种跑法根本没有终端，`input()` 会一直堵着，
    而堵着的进程不放主臂串口，下一次 collect 会被 `who_holds_serial` 挡在门外 ——
    臂抱死本身不危险，堵住串口才是真麻烦。
    """
    try:
        fd = sys.stdin.fileno()
        saved = termios.tcgetattr(fd)        # 不是终端就在这里抛
    except Exception:                        # noqa: BLE001
        return False                         # 问不了人，就当没答应 —— 默认不动臂
    try:
        tty.setcbreak(fd)
        print(prompt, end="", flush=True)
        if not select.select([fd], [], [], timeout_s)[0]:
            print(f"\n  {timeout_s:.0f}s 没人按，当作「不动」。")
            return False
        try:
            key = os.read(fd, 1).decode(errors="replace")
        except OSError:
            return False
        print(key)
        return key.lower() == "y"
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)


def offer_home_after_latch(config, out_dir: Path) -> None:
    """latch 收尾：问一句，答应了就上电 + 回 0 位。**默认不动。**

    **为什么不能在回路里直接 `go_home()`。** `env.arm.latched` 为真的时候，
    RT 发送线程已经刹停并 `return` 了（`rokae_arm._rt_sender` 末尾），流一断
    控制器随即掉电抱闸 —— 这就是「抱死」本身。而 `go_home()` 走的
    `env.goto_joints()` 恰恰要那条已经死掉的流（见它 docstring 里「实时流开着
    的时候」那句），喂进去只会空等 20 秒再打个 ⚠。真要动臂必须**先上电、
    再开一个干净的会话**，也就是 `unseize` 那条路 —— 所以这一步只能放在
    `env.close()` 之后，不能放在 break 之前。

    这里没有把驱动「latch 不自清」那条规则改掉（`_rt_sender` 的注释：臂停了
    又自己动起来，是人站在里面时受伤的典型场景）。改的只是**由谁来解** ——
    原来是「退出重开」，现在是当场问操作员一句。不答就不动。
    """
    import gc                                                 # noqa: PLC0415

    from atom_robot.ar5 import unseize                        # noqa: PLC0415

    # env.close() 已经把驱动里的 `_robot` 置空了，但对象要等回收才真的放掉。
    # 下面 diagnose 和 go_home 各要再建一个 robot 打同一台控制器 —— unseize
    # 自己在 diagnose 末尾也是这么收的尾，理由见那儿的注释。
    gc.collect()

    manual = "python -m atom_robot.ar5.unseize --recover --home"
    print("\n  —— 臂现在是掉电抱闸：latch 把 RT 流掐了，控制器跟着掉电。")
    print(f"     这次的控制台输出留在 {out_dir / 'collect.log'}")
    if not confirm_keypress(
            "\n  要现在自动上电并回 0 位吗？**臂会移动，先确认活动范围里没人。**"
            "\n  按 y 回位，其它键不动（60s 不答也不动）： "):
        print(f"\n  没动臂。要自己救：{manual}")
        return

    # 控制器被掐掉 RT 会话之后要一会儿才重新接受连接（实测约 40 秒，和启动那段
    # 引的是同一条）。不等就直接连，报的是「网络异常: 网络连接错误」—— 和网线
    # 无关，而这套代码一直在防的就是这句话把人带去查网线。
    deadline = time.time() + 90.0
    while True:
        state = unseize.diagnose(config.arm, quiet=True)
        if state["reachable"] or time.time() >= deadline:
            break
        print(f"\r  等控制器放开上一个会话…… 还剩 {deadline - time.time():4.0f}s ",
              end="", flush=True)
        time.sleep(5.0)
    print()

    if not state["reachable"]:
        print(f"  ✗ 控制器还是连不上：{state['why']}")
        print(f"    再等一会儿手动跑：{manual}")
        return
    if state["physical_stop"]:
        print(f"  ✗ {state['why']}")
        print("    物理停机，**软件清不掉**。去把按钮 / 安全门恢复。")
        return

    if unseize.power_back_on() != 0:
        print("\n  ✗ 上电没成功，臂没动过。上面 power_on 自己打的返回码才是真原因；")
        print("    `ec=-3` 是急停 —— powerState() 在急停时也报 off，分不出来。")
        print(f"    手动：{manual}")
        return

    # go_home 自己建一个不带手、不带相机的 AR5Env（救援只关心臂）。这时候
    # collect 的 env 已经在 finally 里关掉了，不会有两个会话打同一台控制器。
    try:
        unseize.go_home(config)
    except Exception as exc:                                  # noqa: BLE001
        print(f"\n  ⚠ 回 0 位半路失败：{type(exc).__name__}: {exc}")
        print("    电是上着的，**臂可能停在半路上** —— 先自己看一眼它在哪儿，")
        print(f"    再决定要不要重来：{manual}")


class Recorder:
    """攒一条 episode 的数据，结束时落盘。

    动作记的是**位姿增量**，因为 VEL 的动作空间就是增量（前 6 维位置和姿态的变化量，
    第 7 维手的开合）。同时也把绝对位姿存下来，这样以后想换算法重算增量还有原始数据。
    """

    def __init__(self, out_dir: Path, camera_keys, config):
        self.out_dir = out_dir
        self.camera_keys = list(camera_keys)
        self.config = config          # CollectConfig：帧率、质量阈值、编码器
        self.fps = config.hz
        # 写盘在后台线程上。实测 600 帧 + 2 路相机存一次要 2.21 秒，而臂的
        # sender 0.25 秒收不到目标就刹车并 latch —— 在回路里同步存盘等于
        # 每存一条就有一次锁死机会。2026-09-03 就是这么废掉三条数据的：
        # 第一条存的时候臂正好停着（_at_rest 不 latch）侥幸活了，第二条存的
        # 时候臂还在动，锁死，之后 600 帧 × 2 条全是一动不动的死数据。
        self._writer = None
        self._write_error = None
        self.reset()

    def reset(self):
        self.state = []        # 7 个关节角 + 手开合 = 8
        self.ee_pose = []      # 末端绝对位姿 6
        self.action = []       # 位姿增量 6 + 手开合 1 = 7
        self.frames = {k: [] for k in self.camera_keys}
        # 每帧的真实时刻。没有这个就只能相信配置里那个标称 fps —— 2026-09-04
        # 查 50 条数据时才发现：回路到底跑了多快，事后完全无从得知。
        self.stamps = []
        # 每个来源各自的时刻（同一个钟、同样扣掉暂停），和 stamps 一一对应：每路相机这张图的拍摄时刻、
        # 臂读关节角的时刻、夹爪开合读数的时刻。存成 npz 的 t_<来源>，用来查同步（用户 10-20）。
        self.sources = list(self.camera_keys) + ["arm", "gripper"]
        self.source_stamps = {name: [] for name in self.sources}
        self._prev_pose = None
        # 录制期间离合器被松开过。松开的时候臂不跟随主臂，那几帧的动作和画面
        # 对不上 —— 数据看起来正常，但训出来的策略会学到「有时候动作不生效」。
        # P 保持期间的帧。臂不动、主臂在重新摆位，这几帧 action 接近 0、画面静止，
        # 两边一致 —— **不是坏数据**，只是演示里的停顿。逐帧标进 npz 的 `held`，
        # 训练时想丢就按它筛。
        self.held = []
        self.clutch_dropped = 0
        # OmniPicker 暂停（不录帧）的处理，见 mark_resume()
        self.resumed = []
        self.paused_seconds = 0.0
        self._pause_offset = 0.0
        self._resume_pending = False

    def __len__(self):
        return len(self.state)

    def mark_resume(self):
        """暂停（不录帧）之后接着录：下一帧的时间戳接在上一帧后面一个周期，暂停那段扣掉。

        OmniPicker 第二下回车是暂停、**不录帧**（用户 10-20）。不扣的话，暂停那几秒算进
        首尾跨度，`fps_measured` 就被拉低 —— 10-20 实测录制段 30.0 Hz、存下来写的是 17.6；
        而 to_lerobot 拿它当帧率转，视频放慢、动作间隔全错。扣了多少（paused_seconds）、
        哪一帧是接着录的（resumed）只留在内存里给存盘时打日志，**不写进 npz** —— 用户要求
        数据集里完全看不出采集时的中断。P 保持（其他末端）是照录帧的，不走这里。"""
        self._resume_pending = True

    def add(self, obs, hand_opening: float, held: bool = False):
        now = time.perf_counter()
        resumed = self._resume_pending and bool(self.stamps)
        if resumed:
            gap = now - self._pause_offset - self.stamps[-1]
            removed = max(0.0, gap - 1.0 / self.fps)
            self._pause_offset += removed
            self.paused_seconds += removed
        self._resume_pending = False
        self.resumed.append(resumed)
        self.stamps.append(now - self._pause_offset)
        # 同一帧的各个来源扣同样多的暂停 —— 接缝两边的相对关系不变
        got = obs.observation.get("stamps") or {}
        for name in self.sources:
            t = got.get(name)
            self.source_stamps[name].append(np.nan if t is None else t - self._pause_offset)
        self.held.append(bool(held))
        pose = np.asarray(obs.observation["ee_pose"], dtype=np.float64)
        if self._prev_pose is None:
            delta = np.zeros(6)
        else:
            delta = pose_delta(self._prev_pose, pose)
        self._prev_pose = pose

        self.state.append(np.asarray(obs.observation["qpos"], dtype=np.float32))
        self.ee_pose.append(pose.astype(np.float32))
        self.action.append(np.concatenate([delta, [hand_opening]]).astype(np.float32))
        for key in self.camera_keys:
            self.frames[key].append(obs.observation["images"][key])

    def warnings(self) -> list:
        """存之前看一眼这条能不能用。阈值在 CollectConfig 里，可以改。

        最常见的废数据是手全程没动 —— action 第 7 维是个常数，那一维就没有任何
        可学的东西。臂几乎没动同理。这些录完当场说，比训练时才发现强。
        """
        cfg = self.config
        notes = []
        if len(self) < cfg.min_frames:
            notes.append(f"只有 {len(self)} 帧，短于 {cfg.min_frames}")

        action = np.stack(self.action) if self.action else np.zeros((1, 7))
        if np.ptp(action[:, 6]) < cfg.min_gripper_travel:
            notes.append(f"手全程没动（一直是 {action[0, 6]:.2f}），gripper 那一维学不到东西")

        pose = np.stack(self.ee_pose) if self.ee_pose else np.zeros((1, 6))
        # 走过的**路径长度**，不是首尾直线距离。后者对 take & place 是错的
        # 判据：做完之后臂就在起点附近，首尾自然接近 —— 实测 8 条误报 4 条。
        path = float(np.linalg.norm(np.diff(pose[:, :3], axis=0), axis=1).sum()) * 1000
        if path < cfg.min_ee_path_mm:
            notes.append(f"末端一共只走了 {path:.0f} mm（阈值 {cfg.min_ee_path_mm:.0f}），"
                         "臂基本没动")

        # P 保持是正常操作（主臂行程不够时重新锚定），不当缺陷报。
        # 只有占比大到「这条大半时间臂没动」才值得说一句。
        held = int(np.count_nonzero(self.held))
        if held and len(self) and held / len(self) > 0.5:
            notes.append(f"{held}/{len(self)} 帧（{held / len(self) * 100:.0f}%）是 P 保持，"
                         "这条大半时间臂没动")

        # 实测帧率对不上标称的，说明回路没跑到设定速度。差得多的时候，存进
        # npz 的 fps 是错的：视频会放快，动作增量对应的间隔也比 1/fps 长。
        #
        # 跨度不足 1 秒就不判：几帧之间的间隔估帧率纯是噪声，正常一条是十几秒。
        measured = self.measured_fps()
        span = (self.stamps[-1] - self.stamps[0]) if len(self.stamps) >= 2 else 0.0
        if measured and span >= 1.0 and abs(measured - cfg.hz) > cfg.hz * 0.1:
            notes.append(f"实测帧率 {measured:.1f} Hz，标称 {cfg.hz:.0f} Hz，"
                         f"差 {abs(measured - cfg.hz) / cfg.hz * 100:.0f}%")
        return notes

    SYNC_TITLE = "同步（各来源比这一帧的录制时刻早多少）"

    def pause_summary(self):
        """(暂停了几次, 一共几秒)，存盘时打日志用。"""
        return sum(self.resumed), self.paused_seconds

    def last_qpos(self):
        """最近录下的那一帧的 7 关节角 + 开合；还没录就 None。"""
        return self.state[-1] if self.state else None

    def sync_report(self):
        """这一条里各来源的数据比这一帧的录制时刻早多少（ms），和同一帧里最早、最晚的数据差多少。

        「早」是正常的：图是取帧线程里最新的一张，夹爪是读写线程上一次的读数，录这一帧时它们都已经
        有一阵了。要看的是这个差有多大、稳不稳（抖得厉害 = 同一帧里的图和臂对不齐的程度在变）。
        返回 (每个来源一句, 同一帧最早最晚差几 ms 那一句或 None)。
        """
        if len(self.stamps) < 2:
            return [], None
        row = np.asarray(self.stamps)
        lines, cols, spread_line = [], [], None
        for name in self.sources:
            t = np.asarray(self.source_stamps[name], dtype=np.float64)
            ok = np.isfinite(t)
            if not ok.any():
                lines.append(f"{name} 没有时刻")
                continue
            age = (row[ok] - t[ok]) * 1000
            lines.append(f"{name} 早 {np.median(age):.0f} ms（{np.percentile(age, 5):.0f}~{np.percentile(age, 95):.0f}）")
            cols.append(t)
        if len(cols) >= 2:
            spread = (np.nanmax(cols, axis=0) - np.nanmin(cols, axis=0)) * 1000
            spread_line = (f"同一帧里最早和最晚的数据差 中位 {np.nanmedian(spread):.0f} ms、"
                           f"最大 {np.nanmax(spread):.0f} ms")
        return lines, spread_line

    def measured_fps(self):
        """实测帧率。不足两帧就没有间隔可算，返回 None。"""
        if len(self.stamps) < 2:
            return None
        span = self.stamps[-1] - self.stamps[0]
        return (len(self.stamps) - 1) / span if span > 0 else None

    def save(self, index: int, task: str) -> Path:
        """交给后台线程写盘，立刻返回。缓冲当场换新的，可以接着录。

        **绝对不能在回路里同步写。** 实测 600 帧 + 2 路相机是 2.21 秒，
        而臂的 sender 0.25 秒收不到目标就刹车并 latch。
        """
        self.flush()          # 上一条必须先写完，见 flush() 的说明
        payload, frames = self._payload(task)
        npz_path = self.out_dir / f"episode_{index:04d}.npz"
        self.reset()          # 换新缓冲，交出去的那份归 writer 了

        self._writer = threading.Thread(
            target=self._write, args=(npz_path, payload, frames),
            name=f"ar5-write-{index:04d}", daemon=False,
        )
        self._writer.start()
        return npz_path

    def _payload(self, task: str):
        """存进 npz 的数组和要写成视频的帧。"""
        stamps = np.asarray(self.stamps, dtype=np.float64)
        payload = dict(
            state=np.stack(self.state),
            ee_pose=np.stack(self.ee_pose),
            action=np.stack(self.action),
            task=task,
            fps=self.fps,                       # 标称值，来自 config
            # 每帧相对第一帧的秒数，实测。fps_measured 才是真实速率 ——
            # 转 lerobot、算动作间隔都该用它，不要用上面那个标称值。
            timestamp=stamps - stamps[0] if len(stamps) else stamps,
            fps_measured=float(self.measured_fps() or self.fps),
            camera_keys=np.array(self.camera_keys),
            # 这一帧是不是 P 保持中（臂原地不动，操作员在给主臂重新摆位）
            held=np.asarray(self.held, dtype=bool),
            # 各来源的时刻：t_cam_high、t_cam_front（这张图的拍摄时刻）、t_arm（读关节角）、
            # t_gripper（夹爪开合读数）。和 timestamp 同一个零点（第一帧的录制时刻）、同样扣掉了暂停；
            # 拿不到的是 nan。相机的时刻是哪种（内核拍摄时刻 / 程序收到）见 collect.log。
            **{f"t_{name}": (np.asarray(v, dtype=np.float64) - stamps[0] if len(stamps)
                             else np.asarray(v, dtype=np.float64))
               for name, v in self.source_stamps.items()},
            # OmniPicker 暂停的位置和时长**故意不进 npz**：用户 10-20 要求数据集里完全看不出
            # 采集时的中断。暂停那段已从 timestamp 里扣掉，停了几次、几秒只打在终端 / collect.log。
        )
        return payload, self.frames

    def _write(self, npz_path: Path, payload: dict, frames: dict) -> None:
        """后台写盘。异常不往外抛（没人接），存起来给 flush() 报。"""
        try:
            import cv2

            self.out_dir.mkdir(parents=True, exist_ok=True)
            stem = npz_path.stem
            for key, images in frames.items():
                if not images:
                    continue
                h, w = images[0].shape[:2]
                path = self.out_dir / f"{stem}_{key}.mp4"
                writer = cv2.VideoWriter(
                    str(path), cv2.VideoWriter_fourcc(*self.config.video_codec),
                    self.fps, (w, h))
                for frame in images:
                    # 相机驱动给的是 RGB，写视频要 BGR
                    writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                writer.release()
            np.savez_compressed(npz_path, **payload)
        except Exception as exc:  # noqa: BLE001
            self._write_error = f"{npz_path.name} 写盘失败：{exc}"

    def flush(self, timeout: float = 60.0, warn: bool = True) -> None:
        """等上一条写完。

        为什么必须等：一条 600 帧双相机在内存里是约 1.1 GB，两条同时挂着就
        2.2 GB。

        `warn` 只在回路里为真。`save()` 开头调这个，那里真等上了就是问题 ——
        回路被堵住，臂随时会 latch。退出时那次是必须等的，不算异常。
        """
        writer, self._writer = self._writer, None
        if writer is not None and writer.is_alive():
            waited = time.perf_counter()
            writer.join(timeout=timeout)
            waited = time.perf_counter() - waited
            if warn and waited > 0.2:
                print(f"\n  ⚠ 等上一条写盘等了 {waited:.1f}s —— 回路被堵这么久，"
                      "臂可能已经刹车 latch。上一条是不是特别长？")
            if writer.is_alive():
                self._write_error = f"写盘线程 {writer.name} 超过 {timeout:.0f}s 没结束"

    def is_writing(self) -> bool:
        """上一条还在后台写盘吗。

        下一次 save() 会在 flush() 里等它 —— 等的时候回路是停的，臂随时 latch。
        网页上看得见这个，才能把「第三条开始按空格没反应」和写盘堵塞对上号。
        """
        w = self._writer
        return bool(w is not None and w.is_alive())

    def take_write_error(self):
        error, self._write_error = self._write_error, None
        return error


# OmniPicker 的软件对齐（用户 10-21）：以腕部相机为基准。顶部 335 只有「程序收到」的时刻，
# 挑最近那张之前先扣掉它比腕部拍摄时刻晚的那一截（秒）。10-21 用 diag_camera_latency 开关灯 17 次量的：
# 平均晚 7.2 ms（±4 ms；单次 -15~+33，两台晶振不同、相位在漂）。换相机、换线、改分辨率后重量。
SYNC_MASTER = "cam_front"
SYNC_LATENCY = {"cam_high": 0.007}

# `--depth`（10-30 起）：每路相机的深度当成另一个来源「<相机>_depth」一起对齐，存进
# episode_XXXX_depth.zarr.zip（每路一个数组，(N, H, W) uint16 原始值，attrs 里是 mm_per_unit 等），
# 时刻是 npz 的 t_<相机>_depth。为什么不进 npz：np.savez_compressed 压一帧深度 12 ms，一条 600 帧
# 两路要 14 秒，下一条存盘时 flush() 会等它、回路被堵；Blosc 一帧 0.3 ms。
DEPTH_SUFFIX = "_depth"
DEPTH_FILE = "{stem}_depth.zarr.zip"


def write_depth(path: Path, depth: dict, info: dict) -> None:
    """深度写成一个 zip 里的 zarr：每路 (N, H, W) uint16，一帧一块。先写 .part 再改名，半截文件不会被当成数据。"""
    import zarr  # noqa: PLC0415
    from numcodecs import Blosc  # noqa: PLC0415

    codec = Blosc(cname="zstd", clevel=1, shuffle=Blosc.BITSHUFFLE)
    part = path.with_name(path.name + ".part")
    store = zarr.ZipStore(str(part), mode="w")
    try:
        root = zarr.group(store=store)
        root.attrs["note"] = ("原始深度，值 × mm_per_unit = 毫米，0 = 无效；aligned_to_color=False 的要用 "
                              "calib/depth_params.json 对齐到彩色；时刻见同名 npz 的 t_<相机>_depth")
        for name, frames in depth.items():
            if not frames:
                continue
            h, w = frames[0].shape
            arr = root.create_dataset(name, shape=(len(frames), h, w), chunks=(1, h, w),
                                      dtype=np.uint16, compressor=codec)
            for i in range(0, len(frames), 64):
                arr[i:i + 64] = np.stack(frames[i:i + 64])
            arr.attrs.update(info.get(name, {}))
    finally:
        store.close()
    part.rename(path)


CALIB_FILES = ("top_extrinsic.json", "wrist_handeye.json", "depth_params.json")


def copy_calibration(dataset_dir: Path) -> list:
    """把当前标定复制进数据集的 calib/，数据自带标定（`camera_3d.load_rig(数据集/calib)` 直接能读）。

    已有的不覆盖：续采时标定变了（相机碰歪重标过）就报出来 —— 同一个数据集里混两套标定，
    前后几条的深度换算到基座系就不一致。返回要打给操作员看的几句话。"""
    import shutil  # noqa: PLC0415

    from atom_robot.ar5.camera_3d import CALIB_DIR  # noqa: PLC0415

    dst = dataset_dir / "calib"
    dst.mkdir(exist_ok=True)
    notes = []
    for name in CALIB_FILES:
        src, out = CALIB_DIR / name, dst / name
        if not src.exists():
            notes.append(f"⚠ 没有标定文件 {src}，calib/ 里缺 {name}")
        elif not out.exists():
            shutil.copy2(src, out)
        elif out.read_bytes() != src.read_bytes():
            notes.append(f"⚠ 标定 {name} 和这个数据集开采时的不一样（相机重标过？）。"
                         f"calib/ 里留的是旧的；混着采前后条换算不一致，考虑换个任务名开新数据集")
    return notes


class SyncRecorder(Recorder):
    """OmniPicker 的录制器：采集时各来源全量带时刻记下来，存盘前按腕部相机对齐（见 sync_align.py）。

    回路**每圈**调一次 log()：读一次臂（关节角 + 末端位姿，同一次状态刷新）、取走每路相机新来的帧、
    夹爪读写线程里新来的读数、记下这一圈的夹爪指令。暂停时不调；接着录时 mark_resume() 开新的一段，
    对齐时每段单独做、不跨暂停插值。warnings()、存盘之前先 finalize() 对齐，对齐后的数组放进父类
    那几个列表，后面的检查、写盘都照旧。原来那种「30 Hz 节拍上取最新一张图 + 当下的臂」不再用。
    """
    SYNC_TITLE = "对齐"

    def __init__(self, out_dir, camera_keys, config, master=SYNC_MASTER, latency=None, depth_info=None):
        self.master = master
        self.latency = dict(SYNC_LATENCY if latency is None else latency)
        # 要存深度的那几路：名字 -> cameras.depth_info[名字]（mm_per_unit、是否对齐彩色、来源、时刻种类）
        self.depth_info = dict(depth_info or {})
        for name, info in self.depth_info.items():
            # 和彩色同一次 grab 的（335 OBSENSOR）时刻也是「程序收到」，扣同样的延迟才会挑中同一帧；
            # 有内核时刻的（305 V4L2）不扣
            same = info.get("stamp_kind") == "arrival"
            self.latency.setdefault(name + DEPTH_SUFFIX, self.latency.get(name, 0.0) if same else 0.0)
        super().__init__(out_dir, camera_keys, config)

    def reset(self):
        super().reset()
        self.segments = []
        self._new_segment = True
        self._seen = {k: None for k in self.camera_keys}
        self._seen_depth = {k: None for k in self.depth_info}
        self.depth_frames = {}
        self._grip_after = None
        self._last = None
        self._aligned = None
        self._master_frames = 0

    def __len__(self):
        return self._master_frames if self._aligned is None else len(self._aligned["timestamp"])

    def mark_resume(self):
        self._new_segment = True

    def log(self, env, command, held=False):
        """回路每圈一次（录制中，包括暂停后等停稳那一小段）。返回这一圈的 7 关节角 + 开合，给 SettleWait。"""
        if self._new_segment:
            self.segments.append(sync_align.new_segment())
            self._new_segment = False
        seg = self.segments[-1]
        self._aligned = None
        t, q, ee = env.read_arm_stamped()
        seg["arm_t"].append(t)
        seg["arm_q"].append(q)
        seg["arm_ee"].append(ee)
        cams = env.cameras
        if hasattr(cams, "frames_since"):
            for key in self.camera_keys:
                new = cams.frames_since(key, self._seen[key])
                if new:
                    self._seen[key] = new[-1][0]
                    seg["cams"].setdefault(key, []).extend((stamp, frame) for _, stamp, frame in new)
                    if key == self.master:
                        self._master_frames += len(new)
        else:                           # mock 相机没有帧缓存：每圈当来了一帧
            images, now = cams.read(), time.perf_counter()
            for key in self.camera_keys:
                seg["cams"].setdefault(key, []).append((now, images[key]))
            self._master_frames += 1
        for key in self.depth_info:
            new = cams.depth_since(key, self._seen_depth[key])
            if new:
                self._seen_depth[key] = new[-1][0]
                seg["cams"].setdefault(key + DEPTH_SUFFIX, []).extend((stamp, d) for _, stamp, d in new)
        hand = env.hand
        if hasattr(hand, "openings_since"):
            new = hand.openings_since(self._grip_after)
            if new:
                self._grip_after = new[-1][0]
                for stamp, opening in new:
                    seg["grip_t"].append(stamp)
                    seg["grip"].append(opening)
        else:
            seg["grip_t"].append(time.perf_counter())
            seg["grip"].append(hand.get_opening())
        seg["cmd_t"].append(t)
        seg["cmd"].append(np.nan if command is None else float(command))
        seg["held"].append(bool(held))
        opening = seg["grip"][-1] if seg["grip"] else (self._last[-1] if self._last is not None else 0.0)
        self._last = np.concatenate([q, [opening]])
        return self._last

    def last_qpos(self):
        return self._last

    def finalize(self):
        """对齐（只做一次，之后再 log 会作废重做）。对齐后的数组填进父类的列表。"""
        if self._aligned is None:
            a = sync_align.align(self.segments, self.master, self.fps, self.latency)
            self.state, self.ee_pose, self.action = list(a["state"]), list(a["ee_pose"]), list(a["action"])
            self.held, self.stamps = list(a["held"]), list(a["timestamp"])
            self.frames = {k: a["frames"].get(k, []) for k in self.camera_keys}
            self.depth_frames = {k: a["frames"].get(k + DEPTH_SUFFIX, []) for k in self.depth_info}
            self._aligned = a
        return self._aligned

    def warnings(self) -> list:
        self.finalize()
        return super().warnings() + self.depth_check()

    def pause_summary(self):
        segs = [s for s in self.segments if s["arm_t"]]
        gaps = [b["arm_t"][0] - a["arm_t"][-1] for a, b in zip(segs, segs[1:])]
        return len(gaps), float(sum(gaps))

    def sync_report(self):
        kernel = tuple(k + DEPTH_SUFFIX for k, v in self.depth_info.items() if v.get("stamp_kind") == "kernel")
        return sync_align.report(self.finalize()["stats"], self.master, self.latency, kernel), None

    def _payload(self, task: str):
        a = self.finalize()
        payload, frames = super()._payload(task)
        for key in [k for k in payload if k.startswith("t_")]:
            del payload[key]                # 父类那套「各来源比录制时刻早多少」的 t_*，这里换成对齐后的
        payload.update({k: v for k, v in a.items() if k.startswith("t_")})
        payload.update(a["raw"])
        if self.depth_info:
            frames = dict(frames)
            frames.update({k + DEPTH_SUFFIX: v for k, v in self.depth_frames.items()})
        return payload, frames

    def depth_check(self) -> list:
        """存盘前查深度：每路帧数要等于行数；和彩色同一次 grab 的，每行挑中的必须是同一帧。"""
        if not self.depth_info:
            return []
        a, n, bad = self.finalize(), len(self), []
        for name, info in self.depth_info.items():
            got = len(self.depth_frames.get(name, []))
            if got != n:
                bad.append(f"{name} 深度 {got} 帧，行数 {n}")
            elif info.get("stamp_kind") == "arrival" and name in self.camera_keys:
                t_c, t_d = a.get(f"t_{name}"), a.get(f"t_{name}{DEPTH_SUFFIX}")
                if t_c is not None and t_d is not None and not np.array_equal(t_c, t_d):
                    bad.append(f"{name} 深度和彩色有 {int(np.sum(t_c != t_d))} 行不是同一帧")
        return bad

    def _write(self, npz_path: Path, payload: dict, frames: dict) -> None:
        depth = {k[:-len(DEPTH_SUFFIX)]: frames.pop(k) for k in list(frames) if k.endswith(DEPTH_SUFFIX)}
        super()._write(npz_path, payload, frames)
        if not depth or self._write_error is not None:
            return
        try:
            write_depth(npz_path.with_name(DEPTH_FILE.format(stem=npz_path.stem)), depth, self.depth_info)
        except Exception as exc:  # noqa: BLE001
            self._write_error = f"{npz_path.name} 的深度写盘失败：{exc}"


def acquire_single_instance_lock():
    """只允许一个 collect 在跑。返回锁文件对象 —— **调用方必须一直持有它**，
    对象被回收锁就没了。

    为什么不用「主臂串口被谁占着」就够：那是**点检查**。两个 collect 几乎同时
    启动时，A 查串口的那一刻 B 还没打开它，于是 A 放行；等 A 走到相机那步，
    B 已经把相机抢走了。2026-09-24 实测撞上一次：**臂白上一次电**，而报出来的
    错指向相机（`opened but delivered no frame`），根本看不出是重复会话。

    `flock` 是内核给的原子锁，两个进程同时抢只有一个拿到。而且**进程一死内核
    自动释放** —— 不像 PID 文件，被 kill -9 之后会留下一个过期的锁，
    让下一次启动误以为还有人在跑。

    锁文件放 `/tmp`：换成 out_dir 的话，两个会话给了不同的 `--out` 就各锁各的，
    而它们抢的是同一台臂和同一组相机，和存到哪儿无关。
    """
    import fcntl  # noqa: PLC0415

    path = "/tmp/atom_ar5_collect.lock"
    handle = open(path, "w")            # noqa: SIM115  故意不关：关了锁就掉
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        return None
    handle.write(f"{os.getpid()}\n")
    handle.flush()
    return handle


def who_holds_serial(port):
    """哪些进程开着这个串口，返回 [(pid, 命令行)]。

    从 /proc 扫，不 shell 出去调 `fuser` —— 不用装东西，也不会卡住。
    只看得见当前用户的进程，而这正是要抓的那种：操作员在另一个终端里
    还开着一个 collect。
    """
    import os
    from pathlib import Path as _P

    try:
        target = os.path.realpath(port)
    except OSError:
        return []
    me = os.getpid()
    holders = []
    for proc in _P("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == me:
            continue
        try:
            for fd in (proc / "fd").iterdir():
                if os.path.realpath(fd) != target:
                    continue
                cmd = (proc / "cmdline").read_bytes().replace(b"\0", b" ")
                holders.append((int(proc.name), cmd.decode(errors="replace").strip()))
                break
        except OSError:
            continue          # 进程正好退了，或者不是我们的
    return holders


def camera_problems(config) -> dict:
    """配置里的相机有没有明显问题。不开设备，也不需要 opencv。"""
    from atom_robot.ar5.drivers.cameras import check_sources

    return check_sources(config.cameras)


def delete_episode(out_dir, index: int) -> int:
    """删掉一条 episode 的**全部**文件，返回删了几个。

    一条不是一个文件：一个 npz 加每路相机一个 mp4。只删 npz 会留下孤儿 mp4，
    后面的检查和转换就对不上了。
    """
    from pathlib import Path as _P

    out_dir = _P(out_dir)
    stem = f"episode_{index:04d}"
    gone = 0
    for path in (list(out_dir.glob(f"{stem}.npz")) + list(out_dir.glob(f"{stem}_*.mp4"))
                 + list(out_dir.glob(DEPTH_FILE.format(stem=stem)))):
        try:
            path.unlink()
            gone += 1
        except OSError as exc:
            print(f"    ⚠ 删不掉 {path.name}：{exc}")
    return gone


def next_episode_index(out_dir) -> int:
    """续采时下一条的编号 = **已有的最大编号 + 1**，不是文件数。

    原来用 `len(glob("episode_*.npz"))`。删过中间某条就会撞车：删掉
    episode_0150 之后还剩 199 个文件，下一条就编号 0199 —— 而 episode_0199
    已经存在，**直接覆盖，而且不报错**。挑掉一条坏数据（人手入境、抓空）
    是常规操作，不该有这种后果。
    """
    from pathlib import Path as _P

    out_dir = _P(out_dir)
    if not out_dir.exists():
        return 0
    nums = [int(p.stem[8:12]) for p in out_dir.glob("episode_*.npz")
            if len(p.stem) >= 12 and p.stem[8:12].isdigit()]
    return max(nums) + 1 if nums else 0


class EpisodeSchedule:
    """一条 episode 的起止和这一组的计数。

    抽出来是因为它决定了数据的边界 —— 一条从哪开始到哪结束、录了几条。
    不碰硬件、不读时钟，所以能离线测完。

    三个状态：`idle`（待机）→ `record`（录）→ 存 → `idle` → … → `done`。

    **存完不会自动开下一条。** 自动推进试过一版，问题是「下一条已经开录了但
    人还没摆好现场」，白录一条 —— 数据看起来正常，只是前几秒是空的。
    所以起停都由操作者按键决定，`episode_time_s` 只作为忘按 `s` 的兜底上限。

    键位（2026-09-22 改）：空格开录，`s` 这段做完了（存 + 臂回初始位置），
    `r` 丢掉正在录的重来，`←` 删掉刚存的那条。**没有离合键** —— 录着就是接合，
    不录就是松开，见 `start()` / `stop()` 的说明。
    """

    def __init__(self, config, start_index: int):
        self.config = config
        self.index = start_index      # 下一条的编号（接着磁盘上已有的编）
        self.done = 0                 # **这一次**已经存下的条数
        self.phase = "idle"
        self.finished = False
        # 刚存下的那条的编号，`←` 删的就是它。None = 没有可删的。
        # 只记一条：连按 `←` 不该顺着往回删，那是丢数据丢到没边。
        self.last_saved = None

    @property
    def target(self) -> int:
        return self.config.num_episodes

    @property
    def cap(self) -> int:
        """一条最多多少帧，0 = 不限。"""
        return self.config.frame_cap

    @property
    def recording(self) -> bool:
        return self.phase == "record"

    def start(self):
        """空格：开始录这一条，**同时接合**。

        接合和录制绑在一起，所以没有单独的离合键。理由是这两件事本来就该同进同出：
        不录的时候从臂跟着主臂动，只有两个后果 —— 你摆现场时臂在旁边乱晃，
        以及下一条开录时臂不在起始位置。
        """
        if self.phase != "idle":
            return None
        self.phase = "record"
        return "begin_record"

    def stop(self):
        """s：这段任务做完了 —— 存盘、松开、臂回初始位置。"""
        return "end_episode" if self.phase == "record" else None

    def discard(self):
        """r：丢掉正在录的这条，**回到待机**，编号不变。

        回待机而不是原地接着录：重录的前提是现场要重新布置，而布置的时候
        录进去的是一段「人在摆螺栓」的画面。回待机等于把复位时间交给操作者，
        摆好按空格再开始，想用多久用多久。
        """
        if self.phase != "record":
            return None
        self.phase = "idle"
        return "discard"

    def delete_last(self):
        """←：把**刚存下的那条**从磁盘上删掉，编号退回去。

        只能删最近存的一条，删完 `last_saved` 就清空。连按不会顺着往回删 ——
        「多按一下」和「少按一下」的代价差太远了。正在录的时候不响应，
        那种情况要的是 `r`。
        """
        if self.phase == "record":
            return "recording"
        if self.last_saved is None:
            return None
        index, self.last_saved = self.last_saved, None
        self.index = index            # 编号退回去，下一条重新占这个号
        self.done = max(0, self.done - 1)
        self.phase = "idle"
        self.finished = False
        return index

    def forget(self, next_index: int, this_run: bool):
        """OmniPicker 的 N：磁盘上编号最大的那条刚被删了（不一定是这次存的，可以连着往前删）。

        编号退到 `next_index`（= 剩下的最大编号 + 1）；删的是这次运行存的，就从 done 里减掉，
        这样 -n 的计数也跟着退。`last_saved` 清掉 —— 老的 ← 路径不再适用。
        """
        self.index = next_index
        if this_run:
            self.done = max(0, self.done - 1)
        self.last_saved = None
        self.finished = False
        if self.phase == "done":
            self.phase = "idle"

    def tick(self, frames: int):
        """只有设了上限才会自动结束 —— 按帧数，不按墙上时间。

        帧数才是数据里真正的长度；回路周期抖一下不该改变一条的长短。
        """
        if self.phase == "record" and self.cap and frames >= self.cap:
            return "end_episode"
        return None

    def episode_saved(self, saved: bool) -> bool:
        """存完（或放弃）之后调。返回是否该收工。"""
        if saved:
            self.last_saved = self.index      # ← 能删的就是它
            self.index += 1
            self.done += 1
            if self.target and self.done >= self.target:
                self.phase = "done"
                self.finished = True
                return True
        self.phase = "idle"
        return False


class TrackingWatch:
    """盯着「臂为什么不跟主臂了」，有话当场说。

    症状是「动着动着就停在那儿不动了，程序看着一切正常」。臂停下来一共四种原因，
    **只有第一种原来会报**：

    1. `latched` —— 回路堵了 0.25 秒以上，sender 刹车。回路里已经每帧查了。
    2. `clamped` —— 映射出来的目标撞上关节软限位，被 clip 住。**撞上的那个
       关节不动了，别的还跟**，所以看起来是「歪着不动」而不是全停。
    3. `rejected.jump` —— 一帧跳变超过 `joint_jump_max`（现在 8.6°/帧）被驱动拒了。
       **这个会雪崩**：拒掉之后参考点 `_q_cmd` 不前进，于是下一帧还是同样大的
       跳变，一路拒到底 —— 直到你把主臂搬回拒绝发生的地方。整条臂全停。
    4. `missing` —— 主臂某个舵机这一帧没回包，那个关节保持不动。零星几帧无所谓，
       一直涨说明串口在丢包。

    2、3、4 的计数器本来就一直在数（`mapper.stats` 和 `arm.rt_stats()`），
    只是从来没人看。这个类每秒比一次，涨了就打一行，并说出是哪个关节。
    """

    def __init__(self, mapper, arm, joint_lo, joint_hi, period=1.0, env=None):
        self.mapper, self.arm, self.env = mapper, arm, env
        self.lo, self.hi = np.asarray(joint_lo), np.asarray(joint_hi)
        self.period = period
        self.floor_z = getattr(getattr(env, "config", None), "arm", None)
        self.floor_z = getattr(self.floor_z, "joint_floor_z", 0.0)
        self._last = 0.0
        self._prev = self._snapshot()
        self.notes = []

    def _snapshot(self):
        stats = dict(self.mapper.stats)
        try:
            rejected = dict(self.arm.rt_stats().get("rejected", {}))
        except Exception:                                   # noqa: BLE001
            rejected = {}
        # `env.stats` 里还有几个会让臂停住的出口，原来一个都没看 —— 尤其 `floor_blocked`：
        # 目标低于地板时 step_joints **把 goal 换成原地保持**，臂停死而回路照常跑，
        # 症状就是「主臂怎么动从臂都不动」，屏幕上一个字都没有。
        env = dict(getattr(self.env, "stats", {}) or {})
        return {"ik_fail": stats.get("ik_fail", 0),
                "clamped": stats.get("clamped", 0), "missing": stats.get("missing", 0),
                "jump": rejected.get("jump", 0), "limits": rejected.get("limits", 0),
                "floor": env.get("floor_blocked", 0),
                "envclamp": env.get("joint_clamped", 0),
                "ratelimit": env.get("joint_rate_limited", 0),
                "resync": env.get("anchor_resync", 0)}

    def check(self, now, q_cmd):
        """每帧调，内部限流。返回要打的那行，没事返回 None。"""
        if now - self._last < self.period:
            return None
        self._last = now
        cur = self._snapshot()
        grew = {k: cur[k] - self._prev[k] for k in cur if cur[k] > self._prev[k]}
        self._prev = cur
        if not grew:
            return None
        parts = []
        if "ik_fail" in grew:
            parts.append(f"**目标够不到** {grew['ik_fail']} 帧（逆解不收敛）—— 臂停在上一个"
                         "够得到的位置。多半是主臂把末端带出了臂的可达范围，往回收一点就恢复")
        if "jump" in grew:
            parts.append(f"跳变被拒 {grew['jump']} 次 —— **这个会雪崩**，"
                         "把主臂慢慢搬回刚才的位置就能恢复")
        if "clamped" in grew:
            q = np.asarray(q_cmd, dtype=np.float64)
            at = [f"轴{i + 1}" for i in range(q.size)
                  if q[i] <= self.lo[i] + 1e-6 or q[i] >= self.hi[i] - 1e-6]
            parts.append(f"目标撞软限位 {grew['clamped']} 帧"
                         + (f"（{'、'.join(at)} 顶住了）" if at else ""))
        if "missing" in grew:
            parts.append(f"主臂丢包 {grew['missing']} 帧，那几帧对应关节不动")
        if "limits" in grew:
            parts.append(f"驱动按限位拒了 {grew['limits']} 次")
        if "floor" in grew:
            floor_mm = self.floor_z * 1000
            parts.append(f"**撞地板 {grew['floor']} 帧**（最低点不许低于 {floor_mm:.0f} mm）"
                         "—— 这几帧目标被换成原地保持，臂停着不动，"
                         "主臂再往下压也没用。把主臂抬起来就恢复")
        if "envclamp" in grew:
            parts.append(f"目标被 env 夹在关节限位上 {grew['envclamp']} 帧")
        if "resync" in grew:
            parts.append(f"限速锚点重新对齐 {grew['resync']} 次（刚回过位就是正常的）")
        # 限速只影响跟随速度、不影响能不能动，所以只在它**大面积**发生时才说 ——
        # 每条都报一次「被限速」只会把真正的故障淹掉。阈值至少 1，否则
        # period=0（单测里就是）会让 `0 >= 0` 成立而键根本不存在。
        rate = grew.get("ratelimit", 0)
        if rate >= max(1, int(self.period * 25)):
            parts.append(f"目标被限速 {rate} 帧（主臂比臂快，属正常滞后）")
        if not parts:
            return None
        line = "  ⚠ 臂没完全跟上：" + "；".join(parts)
        self.notes.append(line)
        return line


class LivePreview(threading.Thread):
    """采集时本地看画面。**独立线程，主循环一行都不碰。**

    为什么不在主回路里 imshow：臂的 sender 0.25 秒收不到目标就刹车并 latch，
    而 `cv2.waitKey` 在 X 卡一下就可能超过那个数 —— 拿一个纯展示功能去赌
    整条 episode，不划算。全部 GUI 调用都在本线程内，主回路只写一个状态字典。

    取图走 `cameras.peek()` 而不是 `read()`：后者会推进 `_last_seq` 并累加
    `stale_reads`，那是数据质量指标，预览按自己的节奏取图会把它污染掉。

    这和 `--remote` 那条不冲突：远程把画面写 live.jpg 交给网页，这个是本地窗口。
    """

    def __init__(self, cameras, camera_keys, state, hz=15.0):
        super().__init__(name="collect-preview", daemon=True)
        self.cameras, self.camera_keys, self.state = cameras, camera_keys, state
        self.hz = float(hz)
        # **不能叫 _stop**：Thread 自己有个 _stop()，join() 内部会调它。
        # 拿 Event 顶掉之后 join 抛 TypeError，而 join 在 finally 里 —— 收摊整条断。
        self._halt = threading.Event()
        self.error = None

    def stop(self):
        self._halt.set()

    def run(self):
        try:
            import cv2
        except Exception as exc:                        # noqa: BLE001
            self.error = f"没有 opencv，预览关掉：{exc}"
            return
        win, created = "collect preview", False
        period = 1.0 / max(self.hz, 1.0)
        try:
            while not self._halt.is_set():
                started = time.monotonic()
                frames = self.cameras.peek() if self.cameras is not None else {}
                # 名字和画面成对取。分两个 list 各自过滤，少一路相机就会错位。
                shown = [(k, frames[k]) for k in self.camera_keys
                         if frames.get(k) is not None]
                if shown:
                    canvas = _stack_bgr(cv2, [f for _, f in shown], [k for k, _ in shown])
                    _overlay_state(cv2, canvas, self.state)
                    if not created:
                        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
                        cv2.resizeWindow(win, min(1600, 640 * len(shown)), 520)
                        created = True
                    cv2.imshow(win, canvas)
                    cv2.waitKey(1)          # 必须调，否则窗口不刷新
                self._halt.wait(max(0.0, period - (time.monotonic() - started)))
        except Exception as exc:                        # noqa: BLE001
            self.error = f"预览线程退出：{type(exc).__name__}: {exc}"
        finally:
            try:
                if created:
                    cv2.destroyWindow(win); cv2.waitKey(1)
            except Exception:                           # noqa: BLE001
                pass


def _stack_bgr(cv2, frames, names):
    """横着拼，高度裁齐。驱动给的是 RGB，imshow 要 BGR。"""
    height = min(f.shape[0] for f in frames)
    out = []
    for frame, name in zip(frames, names):
        f = frame
        if f.shape[0] != height:
            f = cv2.resize(f, (int(f.shape[1] * height / f.shape[0]), height))
        f = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
        cv2.rectangle(f, (0, 0), (f.shape[1], 22), (0, 0, 0), -1)
        cv2.putText(f, name, (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1)
        out.append(f)
    return np.hstack(out) if len(out) > 1 else out[0]


def _overlay_state(cv2, canvas, state):
    """底部一条：在录 / 保持 / 待机、第几条、多少帧、夹爪开合。

    **全是 ASCII**：`cv2.putText` 只有 Hershey 矢量字体，没有中文字形，
    中文会一个字一个 `?` 地画出来。
    """
    h, w = canvas.shape[:2]
    rec, hold = bool(state.get("recording")), bool(state.get("holding"))
    cv2.rectangle(canvas, (0, h - 34), (w, h), (0, 0, 0), -1)
    # 保持中是黄的：还在录（红点会误以为臂在跟），但臂其实不动。
    colour = (0, 200, 255) if (rec and hold) else ((60, 60, 255) if rec else (180, 180, 180))
    cv2.circle(canvas, (18, h - 17), 8, colour, -1)
    if rec and hold:
        text = (f"REC/HOLD  ep{state.get('index', 0):04d}  "
                f"{state.get('frames', 0)} frames   arm parked, "
                f"press {state.get('resume_key', 'P')} to resume")
    elif rec:
        text = f"REC  ep{state.get('index', 0):04d}  {state.get('frames', 0)} frames"
    else:
        text = f"IDLE  next ep{state.get('index', 0):04d}"
    cv2.putText(canvas, text, (34, h - 11), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    force, speed = state.get("force"), state.get("speed")
    cv2.putText(canvas, f"gripper {state.get('gripper', 0.0):.2f}"
                        + (f"  F{force}" if force is not None else "")
                        + (f" V{speed}" if speed is not None else "")
                        + f"   {state.get('done', 0)}/{state.get('target', 0) or '-'}",
                (w - 290, h - 11), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)


def preview_image(obs, camera_keys):
    """拼一张给网页看的画面：一路就是它本身，多路横着拼（高度裁齐）。

    取不到就返回 None —— 画面是附加功能，任何情况下都不该反过来影响采集。
    """
    try:
        images = obs.observation["images"]
        frames = [images[k] for k in camera_keys if images.get(k) is not None]
    except Exception:                                  # noqa: BLE001
        return None
    if not frames:
        return None
    if len(frames) == 1:
        return frames[0]
    height = min(f.shape[0] for f in frames)
    return np.concatenate([f[:height] for f in frames], axis=1)


def print_config(config, out_dir: Path, episode: int = None) -> None:
    """把真正生效的参数打出来。

    分两栏是有意的：左边是改了会影响数据本身的，右边是环境。开采之前扫一眼
    左边，尤其是任务描述和幅度 —— 那两个错了整组数据都得重采。
    """
    cfg = config.collect
    by_dataset = config.use_hand and config.end_effector == "omnipicker"
    print()
    print("  ── 采集参数（改 config.py 的 CollectConfig 可持久化）──")
    if by_dataset:
        print("  任务      按 P 之后在终端里输入任务名和语言 prompt")
        print(f"  存到      {out_dir}/任务名/   每个任务一个目录，同名就是续采")
    else:
        print(f"  任务      {cfg.task}")
        print(f"  存到      {out_dir}" + (f"   已有 {episode} 条，从 {episode} 号接着编"
                                          if episode is not None else ""))
    print(f"  控制频率  {cfg.control_hz:.0f} Hz   给臂发目标；跟踪器滞后 ≈ 3 个周期 = "
          f"{3000 / cfg.control_hz:.0f} ms")
    print(f"  录制帧率  {cfg.hz:.0f} Hz" +
          ("   ⚠ 高于相机的 30 fps，会录到重复帧" if cfg.hz > config.cameras.fps else ""))

    cap = (f"每条上限 {cfg.episode_time_s:.0f}s（{cfg.frame_cap} 帧）"
           if cfg.frame_cap else "每条不限时长，靠 s 结束")
    print(f"  编排      {'每个数据集这次' if by_dataset else '这次'}采 "
          f"{cfg.num_episodes if cfg.num_episodes else '不限'} 条   {cap}")
    print("            **手动起停，存完臂自己回初始位置，不会自动开下一条**")

    print(f"  遥操空间  " + ("**笛卡尔**：主臂末端位姿 -> 逆解，7 个轴全动" if cfg.space == "cartesian"
                             else "关节：一对一映射，轴3、轴5 锁死"))
    print(f"  映射      {'相对' if cfg.relative else '绝对'}   幅度 x{cfg.scale}   "
          f"接合过渡 {cfg.ramp_seconds:.1f}s")
    cut = cfg.filter_min_cutoff
    beta = cfg.filter_beta
    print(f"  主臂滤波  截止 {'默认' if cut is None else f'{cut:.2f} Hz'}   "
          f"beta {'默认' if beta is None else f'{beta:.3f}'}")
    print(f"  质量阈值  最少 {cfg.min_frames} 帧   手至少动 {cfg.min_gripper_travel:.2f}   "
          f"末端至少走 {cfg.min_ee_path_mm:.0f} mm")
    print()
    print(f"  相机      {list(config.cameras.sources) or '（无 —— 数据没有图像，不能训练）'}")
    for name, problem in camera_problems(config).items():
        print(f"            ⚠ {name}: {problem}")
    print(f"  预览      " + (f"开 @{cfg.preview_hz:.0f} Hz（独立线程，别点那个窗口，"
                             "焦点跑了按键就不响应）" if cfg.preview else "关   --preview 打开"))
    follow = "扳机不起作用，空格夹紧、O 松开" if config.end_effector == "omnipicker" else "一启动就跟扳机"
    print(f"  手        " + (f"开（{config.end_effector}，{follow}）" if config.use_hand else "关"))
    print(f"  主臂      {cfg.leader_port} @{cfg.leader_baud}   标定 {cfg.calib_id}")
    print(f"  臂        {config.arm.ip}   地板 z ≥ {config.arm.joint_floor_z * 1000:.0f} mm")
    print(f"  视频      {cfg.video_codec}  {config.cameras.width}x{config.cameras.height}")
    print()
    if config.use_hand and config.end_effector == "omnipicker":
        # OmniPicker 用另一套键（2026-10-20 用户定的），流程见 omnipicker/keyflow.py
        print("  流程：P 开一个数据集 → 输任务名 → 输语言prompt → 测试力度和速度 → 回车 → "
              "回车开始录 → … → P 换下一个数据集 / q 退出")
        print("  每一步能按的键都会在那一步打出来。**主臂扳机不起作用**，夹爪只听空格（夹紧）和 O（松开）。")
        return
    print("  空格 开录     P 保持/重锚(录制中)     s 做完了(存+回0位)     "
          "r 丢掉重来     ← 删掉刚存的     q 退出")
    print("  P 是给**主臂行程不够**用的：按一下臂停住、主臂随便收回来，再按一下从新位置接着跟。")
    print("  录着就是接合，不录就是松开；录制中要临时停住只用 P。存完臂自己回 0 位。")


def main() -> int:
    from atom_robot.ar5.end_effector import add_end_effector_args, configure_end_effector
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # 默认值全部来自 config.py 的 CollectConfig，命令行只是临时覆盖。
    # 想让某个改动下次开机还在，改 config.py，不要靠记住命令行。
    defaults = AR5Config().collect
    parser.add_argument("--task", default=None,
                        help=f"这条数据在做什么，一句话（默认 {defaults.task!r}）")
    parser.add_argument("--out", default=None, help=f"存到哪（默认 {defaults.out_dir}）")
    parser.add_argument("--calib-id", default=None)
    parser.add_argument("--leader-port", default=None)
    parser.add_argument("--leader-baud", type=int, default=None)
    parser.add_argument("--hz", type=float, default=None,
                        help=f"录制帧率（默认 {defaults.hz:.0f}，相机是 30 fps）")
    parser.add_argument("--num-episodes", "-n", type=int, default=None, metavar="N",
                        help=f"这次采多少条新的，0 = 不限（默认 {defaults.num_episodes}）")
    parser.add_argument("--episode-time", type=float, default=None, metavar="秒",
                        help=f"一条最长录多少秒，到点自动存的兜底上限。"
                             f"0 = 不限，靠 s 结束（默认 {defaults.episode_time_s:.0f}）")
    # 这两个只为「存完自动开下一条」服务，那个功能按操作者要求去掉了（自动开录
    # 会赶在人摆好现场之前，白录一条）。收下但忽略：它们还在 shell 历史里，
    # 按上箭头就翻出来，硬报错会反复挡路。忽略但**大声说出来**，不静默。
    parser.add_argument("--reset-time", type=float, default=None, metavar="秒",
                        help="已废弃，无作用。原本是存完自动倒数再开下一条的时间")
    parser.add_argument("--warmup-time", type=float, default=None, metavar="秒",
                        help="已废弃，无作用。原本是第一条之前自动倒数的时间")
    parser.add_argument("--scale", type=float, default=None,
                        help="幅度系数，0.5 = 减半")
    parser.add_argument("--space", choices=("joint", "cartesian"), default=None,
                        help="joint = 关节一对一（轴3、轴5 锁死）；cartesian = 主臂末端位姿 -> 逆解，7 个轴全动")
    parser.add_argument("--relative", action="store_true", default=None,
                        help="按主臂走的位移驱动（默认开）")
    parser.add_argument("--absolute", dest="relative", action="store_false",
                        help="改成主臂在哪臂就在哪")
    parser.add_argument("--min-cutoff", type=float, default=None, metavar="HZ",
                        help="主臂滤波截止频率，抖就往小调。见 VEL/docs/arm-shake.md")
    parser.add_argument("--beta", type=float, default=None,
                        help="滤波随速度放开的快慢，发黏就往大调")
    parser.add_argument("--no-hand", action="store_true")
    parser.add_argument("--no-cameras", action="store_true",
                        help="不开相机。**采出来的数据没有图像，不能训练** —— 只用来试流程")
    parser.add_argument("--no-home-on-start", action="store_true",
                        help="开机不回 0 位。默认会回 —— 臂掉电/抱死之后停在哪儿是随机的，不回位第一条就从一个奇怪的地方起步")
    parser.add_argument("--control-hz", type=float, default=None, metavar="HZ",
                        help="给臂发目标的频率，和录制帧率分开。默认 50，和 VEL 的 "
                             "teleop_joint 一致。调低会让工具上一顿一顿的")
    parser.add_argument("--preview", action="store_true", default=None,
                        help="开一个本地窗口边采边看。独立线程，不影响采集回路。"
                             "**别点这个窗口** —— 按键是从终端读的，焦点跑了键就不响应")
    parser.add_argument("--cameras", default=None,
                        help="只用这几路，逗号分隔，如 `--cameras cam_high`。不给=用 config 里的。"
                             "**少一路的数据和全路的不能混成一个数据集**（相机路数不一样）")
    parser.add_argument("--depth", action="store_true",
                        help="每路相机的深度一起存（episode_XXXX_depth.zarr.zip），数据集目录里复制一份 "
                             "calib/ 标定。只在 --end-effector omnipicker 下可用")
    parser.add_argument("--show-config", action="store_true",
                        help="把生效的参数打出来就退出，不碰硬件")
    parser.add_argument("--remote", action="store_true",
                        help="网页驱动：命令从 {out}/control.json 读，画面写 {out}/live.jpg，"
                             "状态写 {out}/status.json，由工作站代理转给网页。"
                             "被代理拉起来时必须加 —— 那时 stdin 不是终端，读不了键盘")
    parser.add_argument("--remote-fps", type=float, default=None, metavar="FPS",
                        help="录制时回传画面的帧率（默认 10，0 = 不回传画面，只留按钮）")
    parser.add_argument("--remote-idle-fps", type=float, default=None, metavar="FPS",
                        help="待机时为刷新网页预览取画面的频率（默认 4，0 = 待机不取）。"
                             "待机本来不取画面，取一帧要一次相机 + 几次 SDK 往返")
    add_end_effector_args(parser)
    args = parser.parse_args()

    import logging
    logging.basicConfig(level=logging.INFO, format="  %(message)s")

    retired = [f for f, v in (("--reset-time", args.reset_time),
                              ("--warmup-time", args.warmup_time)) if v is not None]
    if retired:
        print(f"\n  注意：{' 和 '.join(retired)} 已废弃，本次忽略。")
        print("  现在每条都手动起停：空格 开录，s 做完了(存+回初始位置)，r 丢掉重来，← 删掉刚存的。")

    # 命令行覆盖 config。给了什么就覆盖什么，没给的保持 config 里的值。
    collect_cfg = defaults
    for flag, field_name in (("task", "task"), ("out", "out_dir"), ("hz", "hz"),
                             ("scale", "scale"), ("relative", "relative"), ("space", "space"),
                             ("calib_id", "calib_id"), ("leader_port", "leader_port"),
                             ("leader_baud", "leader_baud"),
                             ("min_cutoff", "filter_min_cutoff"), ("beta", "filter_beta"),
                             ("num_episodes", "num_episodes"),
                             ("episode_time", "episode_time_s"),
                             ("cameras", "cameras"), ("preview", "preview"),
                             ("control_hz", "control_hz"),
                             ("remote_fps", "remote_fps"), ("remote_idle_fps", "remote_idle_fps")):
        value = getattr(args, flag)
        if value is not None:
            setattr(collect_cfg, field_name, value)
    # 覆盖完再校验一次，否则 --hz 0 之类的会一路带到磁盘上
    collect_cfg.__post_init__()

    config = AR5Config()
    config.collect = collect_cfg
    config.use_hand = not args.no_hand
    configure_end_effector(config, args)
    omni_mode = config.use_hand and config.end_effector == "omnipicker"
    # OmniPicker 的按键是另一套（空格=夹爪、回车=录制）。网页那边有专门的 OmniPicker 面板，
    # 送来的是**原样按键**（key）和**一整段字**（text），走同一个 KeyFlow；
    # 老的「开录 / 存盘」按钮发的 start/save 在这个模式下一律不认（start 映射成空格就成了
    # 「夹紧夹爪」）—— 见回路里读网页命令那段。
    if omni_mode:
        # 按数据集采：out_dir 是根目录，每个任务一个子目录，按 P 之后才知道是哪个
        if args.out is None:
            collect_cfg.out_dir = collect_cfg.dataset_root
        if args.task is not None:
            print("  注意：OmniPicker 模式下语言 prompt 在终端里输入，--task 不用。")
    out_dir = Path(collect_cfg.out_dir).expanduser()
    config.arm.use_realtime = True
    if args.no_cameras:
        config.cameras.sources = {}
    elif getattr(collect_cfg, "cameras", ""):
        # 只留点名的那几路。用途是「另一路还没插 / 坏了，但想先用现有的这路采」——
        # config 里配着的那路打不开，env.connect() 会直接失败，臂白上一次电。
        want = [c.strip() for c in collect_cfg.cameras.split(",") if c.strip()]
        unknown = [c for c in want if c not in config.cameras.sources]
        if unknown:
            print(f"  ✗ --cameras 里这几个名字不存在：{unknown}")
            print(f"    config 里配的是：{list(config.cameras.sources)}")
            return 2
        config.cameras.sources = {k: v for k, v in config.cameras.sources.items()
                                  if k in want}
        print(f"  只用这几路相机：{list(config.cameras.sources)}")
    if args.depth:
        # 深度跟着对齐走，只有 OmniPicker 的 SyncRecorder 存它；别的末端加了也存不下，不如直接拒绝
        if not omni_mode:
            print("  ✗ --depth 只在 --end-effector omnipicker 下可用")
            return 2
        if SYNC_MASTER not in config.cameras.sources:
            print(f"  ✗ --depth 要以 {SYNC_MASTER} 为基准对齐，这次没开它（相机：{list(config.cameras.sources)}）")
            return 2
        config.cameras.depth_sources = tuple(config.cameras.sources)
        print(f"  深度一起存：{list(config.cameras.depth_sources)}")

    if args.show_config:
        print_config(config, out_dir)
        return 0

    # 从这里开始落盘。放在 --show-config 之后是不想让「只看一眼参数」也建出
    # 目录和日志；但要**早于任何碰硬件的步骤** —— 相机预检、串口占用检查、
    # 上电全在下面，而它们打的正是出问题时最该留下的几行。
    log_path = start_session_log(out_dir)

    # 另一个 collect 还开着的话，现在就停 —— 别等臂上电失败后 SDK 崩掉。
    #
    # 这个失败长得很不像它的原因：控制器已经被前一个会话占住，第二个上电时回
    # `ec=-20 机器人运动中`，那只是一行**警告**；紧接着 SDK 在这条失败路径上
    # `terminate called without an active exception` 直接 abort 并 core dump。
    # 真信息埋在警告里，看到的却是一个没头没尾的崩溃。
    #
    # 判据用的是主臂串口被谁占着：一个活着的 collect 必然握着它，而这比去问
    # 控制器"你被占了吗"可靠 —— 后者要先连上去，连上去本身就已经在抢了。
    # **先拿锁再碰任何硬件。** 下面那道串口检查留着 —— 它能说出是哪个 PID、
    # 哪条命令，比一句"锁被占着"好查；但挡竞态靠的是这把锁。
    lock = acquire_single_instance_lock()
    if lock is None:
        print("\n  ✗ 已经有一个 collect 在跑（/tmp/atom_ar5_collect.lock 被锁着）。")
        others = [f"PID {pid}  {cmd[:90]}" for pid, cmd in
                  who_holds_serial(collect_cfg.leader_port)]
        for line in others:
            print(f"      {line}")
        if not others:
            print("      （它还没打开主臂串口，所以列不出是哪个 —— 多半是刚启动）")
        print("\n  同一台臂不能被两个会话同时驱动。去那个终端按 q 正常退出"
              "（正在录的那条不保存，要留先按 s），再回来重开。")
        return 1

    holders = who_holds_serial(collect_cfg.leader_port)
    if holders:
        print(f"\n  ✗ {collect_cfg.leader_port} 已经被占着，说明**另一个 collect 还开着**：")
        for pid, cmd in holders:
            print(f"      PID {pid}  {cmd[:100]}")
        print("\n  同一台臂不能被两个会话同时驱动。去那个终端按 q 正常退出"
              "（正在录的那条不保存，要留先按 s），再回来重开。")
        print("  找不到是哪个终端就看上面的 PID：`ps -o tty= -p <PID>`。")
        return 1

    leader = LeaderBus.with_calibration(collect_cfg.leader_port, collect_cfg.calib_id,
                                        baudrate=collect_cfg.leader_baud)
    leader.connect()
    leader.assert_torque_disabled()

    if collect_cfg.space == "cartesian":
        # 接口和 JointMapper 一样，下面所有东西（P 键、回 0 位、限速、跳变门、
        # 地板检查、诊断、存盘）原样沿用。
        from atom_robot.ar5.cartesian_mapping import CartesianMapConfig, CartesianMapper  # noqa: PLC0415
        mapper = CartesianMapper(
            config.arm, config.arm.workspace_min, config.arm.workspace_max,
            config=CartesianMapConfig(position_scale=collect_cfg.scale),
            joint_cfg=JointMapConfig.load(**collect_cfg.mapping_overrides))
    else:
        mapper = JointMapper(JointMapConfig.load(**collect_cfg.mapping_overrides))
    # 半行程从标定文件读，不写死 —— 见文件顶部的注释
    mapper.configure({name: leader.half_travel_deg(name) for name, _, _ in JOINT_PAIRS},
                     config.arm.joint_min, config.arm.joint_max,
                     margin=config.arm.joint_limit_margin)

    # 相机预检。**放在锁后面** —— 2026-09-24 改的顺序。
    #
    # 原来在锁前面，那一步的结论在拿到锁之前就过期了：两个 collect 同时启动时
    # 都能查到"相机空着"（对方还在自己的上电流程里，十几秒后才开相机），
    # 于是双双放行。现在先拿锁再查，查到的就是最终状态。
    #
    # 锁已经挡掉了另一个 collect，这一步挡的是**不认锁的东西** ——
    # `view_camera` 的预览窗、别的窗口手跑的脚本。它们照样会占住相机，
    # 而这一步在**臂上电之前**，比等 env.connect() 失败后整套回收便宜得多。
    problems = camera_problems(config)
    if problems:
        print("\n  相机开不了，先解决再开采（臂还没上电）：")
        for name, problem in problems.items():
            print(f"    {name}: {problem}")
        print("\n  被占用的话，去那个终端按 q 退出预览窗口。")
        print("  换过 USB 口的话，跑 view_camera --list 拿新的 by-path 填进 config.py。")
        print("  只是想试流程、不要图像，加 --no-cameras。")
        return 1

    # ── 连臂之前先把它救起来 ────────────────────────────────────────────
    #
    # 臂在四种状态下起不来，而 `env.connect()` 对这四种给的是**同一句**
    # 「网络异常 / 实时模式异常」—— 和网络没关系，看着却像网线的事。
    # 这里先读一遍控制器，认出是哪一种，能自己救的就救：
    #
    #   没上电（off + idle）      会话正常退出就是这样 —— 直接上电
    #   碰撞硬停（off + rtControlling）  掉电且实时会话被半路打死 —— 上电 + 清残留
    #   急停 / 安全门             **物理的**，说清楚然后退出，别让人去查网线
    #   刚停不到 40 秒            控制器还没重新接受连接 —— 等
    #
    # **只上电、不动臂。** 上电只是使能伺服，不产生运动；而"回到某个位置"得等
    # 操作员看过现场 —— 碰撞硬停意味着撞到了东西，开机就往回走等于再撞一次。
    # 要连回位一起做用 `python -m atom_robot.ar5.unseize --recover --home`。
    from atom_robot.ar5.unseize import diagnose, power_back_on  # noqa: PLC0415

    state = diagnose(config.arm, quiet=True)
    if state["physical_stop"]:
        print(f"\n  ✗ {state['why']}")
        print("    去把按钮 / 安全门恢复，再开一次。**软件清不掉这个。**")
        leader.disconnect()
        return 1
    if state["reachable"] and (state["seized"] or state.get("idle_off")):
        print(f"\n  臂现在起不来：{state['why']}")
        print("  先自动上电（**只使能伺服，臂不会动**）……")
        if power_back_on() != 0:
            print("\n  ✗ 上电没成功，上面的返回码才是真原因。`ec=-3` 是急停 ——")
            print("    powerState() 在急停时也报 off，分不出来，只有试上电才知道。")
            leader.disconnect()
            return 1
    elif not state["reachable"]:
        print(f"\n  ⚠ 读不到控制器状态：{state['why']}")
        print("    碰撞停机之后它要**约 40 秒**才重新接受连接。照常往下试一次。")

    env = AR5Env(config)
    env.connect()
    if not getattr(env.arm, "_rt_active", False):
        print("\n  臂的实时通道还是没起来 —— 上面已经试过上电了。")
        print("    自己看一眼：python -m atom_robot.ar5.unseize")
        print("  SDK 在未上电时报的是「网络异常: 网络连接错误」或者「实时模式异常」，")
        print("  和网络无关 —— 别去查网线。")
        env.close()
        leader.disconnect()
        return 1

    camera_keys = list(config.cameras.sources)
    if not camera_keys:
        print("\n  没有配相机，录出来的数据没有图像。config.py 里 CameraConfig.sources 是空的。")

    # 回路按**控制频率**跑，录制按**录制帧率**单独计拍 —— 两者拆开的理由见
    # CollectConfig.control_hz。臂要的是密集的目标（1kHz 跟踪器在两个目标之间
    # 加速刹车，稀了就一顿一顿），相机只有 30 fps，多取也是重复帧。
    period = 1.0 / collect_cfg.control_hz
    record_period = 1.0 / collect_cfg.hz
    next_record = 0.0
    # OmniPicker：各来源全量带时刻记、存盘前按腕部相机软件对齐（用户 10-21）。别的末端照旧。
    if omni_mode and SYNC_MASTER in camera_keys:
        depth_info = dict(getattr(env.cameras, "depth_info", {}) or {}) if args.depth else {}
        recorder = SyncRecorder(out_dir, camera_keys, collect_cfg, depth_info=depth_info)
    else:
        recorder = Recorder(out_dir, camera_keys, collect_cfg)
    synced = isinstance(recorder, SyncRecorder)
    episode = next_episode_index(out_dir)

    # 网页驱动：命令从 control.json 来，画面与状态写回 out_dir，代理负责和网页对接。
    # OmniPicker 用另一套按键流程，按**数据集**采（2026-10-20 用户定的，见 omnipicker/keyflow.py）：
    # P 开一个数据集 → 输任务名（= out_dir 下的目录名）→ 输语言 prompt →
    # 测试力度速度 → 回车 → 回车开录 / 暂停 / 接着录 → S 存 / R 丢 / N 删上一条 → 回车录下一条。
    # 同名目录 y 续采：prompt、力、速度沿用 dataset.json，跳过前两步直接待录。
    # **主臂扳机全程不起作用**。航凯和因时手保持原来的键位，扳机照常控制。
    omni = omni_mode
    flow = None
    dataset_dir = None                  # 当前数据集的目录；P 之前是 None
    session_saved = {}                  # 这次运行每个数据集存了几条，退出时报
    # 当前数据集磁盘上有几条。不用编号算：N 可以连着往前删，编号中间会有空洞；也不每次去
    # glob —— 刚按 S 时 npz 还在后台写，glob 会少数一条。开数据集时数一次，存 / 删时加减。
    disk_count = 0
    saved_this_run = set()              # 这次运行在当前数据集存下的编号（N 删到它们时 -n 计数要退）

    def dataset_prompt(d: Path):
        """这个数据集以前用的语言 prompt：先看 dataset.json，没有就看最新一条 npz 的 task。"""
        try:
            return json.loads((d / "dataset.json").read_text(encoding="utf-8"))["prompt"]
        except (OSError, ValueError, KeyError):
            pass
        npzs = sorted(d.glob("episode_*.npz"))
        if npzs:
            try:
                with np.load(npzs[-1]) as z:
                    return str(z["task"]) if "task" in z.files else None
            except Exception:                       # noqa: BLE001
                return None
        return None

    def dataset_probe(name: str) -> dict:
        """任务名查重。**目录在就算已有**（一条没存也算，用户 10-20）—— 同名只可能是续采，
        直接问要不要续采；续采时 prompt、夹持力、速度都从 dataset.json 沿用。"""
        if not name or name.startswith(".") or "/" in name or "\\" in name:
            return dict(error="任务名不能为空、不能以 . 开头、不能带 / —— 它就是目录名，重新输入")
        d = out_dir / name
        if not d.is_dir():
            return dict(error=None, exists=False, count=0, prompt=None)
        try:
            meta = json.loads((d / "dataset.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            meta = {}
        grip_force, speed = meta.get("grip_force"), meta.get("speed")
        if force_ui and (grip_force is not None or speed is not None):
            # 记下的值不在现在的范围里（改过 config）就当没测过，回去测一次，不硬塞给夹爪
            hc = env.hand.config
            if not (isinstance(grip_force, int) and hc.grip_force_min <= grip_force <= hc.grip_force_max
                    and isinstance(speed, int) and hc.speed_min <= speed <= hc.speed_max):
                grip_force = speed = None
        return dict(error=None, exists=True, count=len(list(d.glob("episode_*.npz"))),
                    prompt=dataset_prompt(d), grip_force=grip_force, speed=speed)

    def write_dataset_meta(**fields):
        """dataset.json：任务名、语言 prompt、这个数据集用的夹持力和速度。只增改，不删已有字段。"""
        path = dataset_dir / "dataset.json"
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            meta = {"created": time.strftime("%Y-%m-%d %H:%M:%S")}
        meta.update(fields, updated=time.strftime("%Y-%m-%d %H:%M:%S"))
        path.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")

    def list_datasets():
        """待机时列出已有的数据集，一个一行：名字、几条、语言 prompt。输同名就是续采。"""
        dirs = sorted(p for p in out_dir.iterdir() if p.is_dir()) if out_dir.is_dir() else []
        if not dirs:
            print(f"  已有的数据集（{out_dir}）：还没有")
            return
        print(f"  已有的数据集（{out_dir}，输同名就是续采）：")
        width = max(len(d.name) for d in dirs)
        for d in dirs:
            count = len(list(d.glob("episode_*.npz")))
            prompt = dataset_prompt(d)
            print(f"    {d.name:<{width}}  {count:>3} 条" + (f"   {prompt}" if prompt else ""))

    # 录制中按空格 / O：臂冻住、只动夹爪，夹牢 / 张开到位后自动接着跟（用户 10-20）。冻着的时候这里是个 GripWait。
    grip_wait = None
    # 录制中回车暂停：臂还要滑到最后一个目标才停，这段照录，停稳了才不录帧。这时候这里是个 SettleWait。
    settle = None
    if omni:
        from omnipicker.keyflow import GripWait, KeyFlow, SettleWait      # noqa: PLC0415
        flow = KeyFlow(dataset_probe, saved_count=lambda: disk_count)
    force_ui = omni and hasattr(env.hand, "step_grip_force")
    preview_state = {"recording": False, "index": 0, "frames": 0,
                     "gripper": 0.0, "done": 0, "target": 0, "holding": False,
                     "force": env.hand.grip_force if force_ui else None,
                     "speed": env.hand.speed if force_ui else None,
                     "resume_key": "Enter" if omni else "P"}
    previewer = None
    link = None
    preview_interval = 0.0
    if args.remote:
        from atom_robot.ar5.remote_ctl import RemoteLink
        link = RemoteLink(out_dir, frame_fps=collect_cfg.remote_fps)
        preview_interval = 1.0 / collect_cfg.remote_idle_fps if collect_cfg.remote_idle_fps > 0 else 0.0

    print_config(config, out_dir, episode)
    print(mapper.report())
    if link is not None:
        print("  ▶ 远程模式：网页上可以直接开录 / 存盘 / 丢弃 / 删上一条，画面也回传到网页。")
        print("    网页命令 → 按键：" + "   ".join(
            "%s=%s" % (k, "空格" if v == " " else ("←" if v == "LEFT" else v))
            for k, v in REMOTE_KEYS.items()))
        print(f"    control.json / live.jpg / status.json 都在 {out_dir}")
    if force_ui:
        hand_cfg = env.hand.config
        print(f"  夹持力现在 {env.hand.grip_force}，范围 {hand_cfg.grip_force_min}~"
              f"{hand_cfg.grip_force_max}，每档 {hand_cfg.grip_force_step}（寄存器值，不是牛顿）；"
              f"速度现在 {env.hand.speed}，范围 {hand_cfg.speed_min}~{hand_cfg.speed_max}，"
              f"每档 {hand_cfg.speed_step}。在每个数据集的「测试力度和速度」那一步调")
    print()

    target = collect_cfg.num_episodes          # 这一次要采多少条新的，0 = 不限
    schedule = EpisodeSchedule(collect_cfg, episode)

    # 上一次跑锁死过的话，臂现在是 latch 状态，什么目标都不接。启动程序本身
    # 就是一次明确的操作者动作，所以在这里清一次 —— teleop_joint 一直这么做，
    # collect 漏了，结果是「开起来臂就是不动」。
    env.arm.clear_latch()

    if collect_cfg.preview and camera_keys:
        # 没有图形界面就当场说，否则 namedWindow 在线程里抛，错误要等到退出才看见。
        if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            print("  ⚠ 没有 DISPLAY，开不了预览窗口（ssh 加 -X，或用 --remote 在网页上看）。采集照常。")
        else:
            previewer = LivePreview(getattr(env, "cameras", None), camera_keys,
                                    preview_state, hz=collect_cfg.preview_hz)
            previewer.start()
            print("  ▶ 本地预览窗口已开。**别去点那个窗口** —— 按键是从终端 fd 读的，")
            print("    焦点一旦跑到预览窗上，空格 / s / r / ← 就全不响应了。")

    # 臂停下来原来只报 latch 一种，别的三种默默发生 —— 「动着动着就不跟了」查不出来。
    watch = TrackingWatch(mapper, env.arm,
                          np.asarray(config.arm.joint_min) + config.arm.joint_limit_margin,
                          np.asarray(config.arm.joint_max) - config.arm.joint_limit_margin,
                          env=env)
    print()

    engaged = False
    # P 保持：臂原地不动、主臂随便摆，解决**主臂行程不够** —— 跟抬鼠标一个道理。
    holding = False
    held_gripper = None
    # `gripper` 在回路后半段才算，P 的处理在前半段。正常时序下按 P 时它一定有值，
    # 但不赌时序 —— 一个 NameError 会把整条 episode 带走。
    gripper = 0.0
    last_line = 0.0
    last_omni_line = None               # OmniPicker 状态行上一次打的内容，变了才再打
    # 是不是从 latch 那个出口退的。q/s/r 三个出口在 break 之前就把臂送回 0 位了，
    # latch 这个出口做不到（流已经死了），所以要带个标记出去，收尾时再救。
    latched_exit = False

    def announce(event):
        if event == "begin_record":
            recorder.reset()
            cap = f"（上限 {schedule.cap} 帧）" if schedule.cap else ""
            print(f"\n  ● 开始录 episode {schedule.index}{cap}"
                  f"   s=做完了  r=丢掉重来")

    def go_home():
        """存完把臂送回初始位置。`s` 的后半段 —— 「这段做完了」就包括复位。

        回的是**映射中心 `centre_joints`**，不是 `config.arm.home_joints`。
        这两个是不同的位姿，实测差到 45.6°（j4）：`home_joints` 是笛卡尔工作空间
        盒的中心，而 `centre_joints` 是主从映射锚定的那个姿势 —— 主臂在中位时
        从臂就该在这儿。回错了的后果是每条都从偏 30~45° 的地方起步，
        主臂行程被不对称地吃掉一半，绝对模式下还会在接合瞬间跳一下。

        实时流开着的时候 `goto_joints` 走的是流本身（一步步喂目标，受同一道
        跳变门管），不会拆流也不会让 sender 断粮 latch，所以不用退出 RT。
        """
        if link is not None:
            # goto_joints 最长阻塞 20 秒，这期间回路不转：既不读网页命令，也不写 status。
            # 不打招呼的话网页会以为脚本死了（状态超时），随后的命令还会被判成「没接住」。
            link.publish_status(min_interval=0, busy="go_home", recording=False,
                                engaged=False, episode=schedule.index, samples=schedule.done)
        try:
            ok = env.goto_joints(mapper.centre, timeout_s=20.0)
            print("  臂已回 0 位（映射中心 = 主臂全零对应的姿势）" if ok is not False else
                  "  ⚠ 臂没在 20s 内回到 0 位，自己看一眼再继续")
        except Exception as exc:                       # noqa: BLE001
            print(f"  ⚠ 回初始位置失败：{type(exc).__name__}: {exc}")

    def end_episode() -> bool:
        """存下当前这条，推进 schedule。返回是否该收工。"""
        if len(recorder) == 0:
            print("\n  一帧都没录到，跳过")
            saved = False
        else:
            notes = recorder.warnings()
            seconds = len(recorder) / collect_cfg.hz
            frames = len(recorder)
            pauses, paused = recorder.pause_summary()      # save() 会清掉
            sync, spread = recorder.sync_report()
            # save() 把缓冲交给后台线程就返回，回路不会停 —— 同步写要 2.2 秒，
            # 而臂 0.25 秒收不到目标就刹车 latch。
            path = recorder.save(schedule.index, collect_cfg.task)
            print(f"\n  存好了 episode {schedule.index}：{frames} 帧 / {seconds:.1f}s"
                  f" -> {path.name}")
            if pauses:
                print(f"    暂停过 {pauses} 次、共 {paused:.1f}s，已从时间戳里扣掉，数据里看不出来")
            if sync:
                kinds = getattr(env.cameras, "stamp_kinds", dict)()
                said = {"kernel": "驱动拍摄时刻", "arrival": "程序收到帧的时刻，比拍摄晚"}
                note = "；".join(f"{k} 用的是{said.get(v, v)}" for k, v in kinds.items())
                print(f"    {recorder.SYNC_TITLE}：" + "、".join(sync))
                if spread:
                    print(f"    {spread}")
                if note:
                    print(f"    （{note}）")
            for note in notes:
                print(f"    ⚠ {note}")
            if notes:
                print(f"    要删就删 {path.name} 和对应的 .mp4")
            saved = True

        error = recorder.take_write_error()
        if error:
            print(f"    ⚠ {error}")

        if schedule.episode_saved(saved):
            if not omni:                # OmniPicker 采够一个数据集不收工，run_omni 自己说
                print(f"\n  采够 {target} 条，收工。")
            return True
        if saved and not omni:          # OmniPicker 由 show_count() 提示「P 准备下一条」
            print(f"    摆好现场按空格录下一条"
                  + (f"（本次 {schedule.done}/{target}）" if target else ""))
        return False

    # ── 开机先回 0 位 ─────────────────────────────────────────────────
    #
    # 臂掉电、抱死、或者上一轮被 Ctrl-C 打断之后，停在哪儿是随机的。不回位的话
    # 第一条 episode 就从那个随机位置起步，而 relative 模式的锚点正是接合那一刻
    # 臂的位置 —— 起点不一致，主臂行程被不对称地吃掉，还更容易撞软限位。
    #
    # `s` / `r` / `q` 三个出口本来就回 0 位，开机这一次补齐，四个时机就齐了。
    #
    # **这一步会让臂移动。** 2026-09-23 之前这里只上电不动，理由是「碰撞停机
    # 意味着撞到了东西，开机就往回走等于再撞一次」—— 这个顾虑仍然成立，
    # 所以动之前会打一行、并给 3 秒反悔窗口（Ctrl-C）。不想要就 --no-home-on-start。
    if not args.no_home_on_start:
        print("\n  开机先回 0 位（`--no-home-on-start` 可关掉）。")
        print("  **臂就要移动了** —— 刚碰撞停过的话先看一眼现场，3 秒内 Ctrl-C 可以打断。")
        try:
            time.sleep(3.0)
        except KeyboardInterrupt:
            print("\n  打断了，不回位。臂停在原地，第一条会从这儿起步。")
        else:
            go_home()

    def show_count():
        """OmniPicker：当前数据集存了多少条。S / R / N 之后都打。"""
        session = f"本次 {schedule.done}/{target}" if target else f"本次 {schedule.done} 条"
        print(f"\n  ○ 数据集「{flow.name}」已存盘 {disk_count} 条（{session}），下一条是 episode {schedule.index}")

    def note_saved(index: int):
        """存下一条之后记账：磁盘条数、这次存过的编号、退出时的汇总。"""
        nonlocal disk_count
        disk_count += 1
        saved_this_run.add(index)
        session_saved[flow.name] = session_saved.get(flow.name, 0) + 1

    def say_help():
        """每进一步都把这一步能按的键打出来（用户 10-20 的要求）。待机时下面再列已有的数据集。"""
        print(f"\n  {flow.help()}")
        if flow.phase == "idle":
            list_datasets()

    def open_dataset():
        """输完任务名和 prompt：建目录、换存盘位置、按这个目录已有的接着编号。"""
        nonlocal schedule, dataset_dir, disk_count, saved_this_run
        d = out_dir / flow.name
        d.mkdir(parents=True, exist_ok=True)
        recorder.flush(warn=False)          # 上一个数据集的最后一条可能还在后台写
        recorder.out_dir = d
        collect_cfg.out_dir = str(d)        # delete_episode 删的是这个目录里的
        collect_cfg.task = flow.prompt      # 写进每条 npz 的 task
        dataset_dir = d
        schedule = EpisodeSchedule(collect_cfg, next_episode_index(d))
        disk_count = len(list(d.glob("episode_*.npz")))
        saved_this_run = set()
        write_dataset_meta(task_name=flow.name, prompt=flow.prompt)
        calib_notes = []
        if synced and recorder.depth_info:
            calib_notes = copy_calibration(d)
            write_dataset_meta(depth=dict(file=DEPTH_FILE.format(stem="episode_XXXX"), calib="calib/",
                                          cameras=recorder.depth_info))
        session_saved.setdefault(flow.name, 0)
        print(f"\n  {'续采' if flow.continuing else '新建'}数据集「{flow.name}」：{d}")
        print(f"  语言prompt：{flow.prompt}")
        print(f"  已存盘 {disk_count} 条，这次从 episode {schedule.index} 接着编")
        if synced and recorder.depth_info:
            print(f"  深度一起存：{list(recorder.depth_info)}，标定已复制到 {d / 'calib'}")
        for note in calib_notes:
            print(f"  {note}")

    # 网页 OmniPicker 面板送来的键名 → KeyFlow 认的键。单个可见字符原样（p、s、y …）
    WEB_KEYS = {"ENTER": "\r", "ESC": "\x1b", "BACKSPACE": "\x7f", "SPACE": " ",
                "UP": "UP", "DOWN": "DOWN", "LEFT": "LEFT", "RIGHT": "RIGHT"}

    def omni_web_key(cmd, doc):
        """网页命令 → 交给 run_omni 的那个键（None = 这一下什么都不做）。"""
        if cmd == "key":
            raw = str(doc.get("key", ""))
            key = WEB_KEYS.get(raw.upper(), raw if len(raw) == 1 and raw.isprintable() else None)
            print("\n  [网页] 按键 %s" % (raw if raw.strip() else repr(raw)))
            return key
        if cmd == "text":
            text = str(doc.get("text", ""))
            print("\n  [网页] 输入「%s」" % text)
            if not flow.text_input:
                print("  现在不是输入文字的那一步，没收")
                return None
            # 先把已经输了一半的字退掉（可能是本地键盘敲的），再一个字一个字喂 —— 和终端里敲
            # 完全一样，查重、续采、空 prompt 都还是 KeyFlow 自己判。最后的回车交回去走正常路径
            for _ in range(len(flow.buffer)):
                run_omni("\x7f")
            for ch in text:
                if ch.isprintable():
                    run_omni(ch)
            return "\r"
        if cmd == "quit":
            print("\n  [网页] 退出")
            if flow.text_input:
                run_omni("\x1b")           # 输入文字时 q 是个字，先 Esc 出来
            return "q"
        # 老按钮（start / save / …）：在这里按老映射会变成别的键（start → 空格 = 夹紧夹爪），不执行
        print("\n  [网页] %s —— OmniPicker 模式下不认这个按钮，用网页上的 OmniPicker 面板" % cmd)
        return None

    def omni_status():
        """给网页的 KeyFlow 状态。help 就是终端里那句「这一步能按什么」，网页原样显示。"""
        st = env.hand.last_status()
        return {"phase": flow.phase, "help": flow.help(), "text_input": flow.text_input,
                "buffer": flow.buffer, "name": flow.name, "prompt": flow.prompt,
                "continuing": flow.continuing, "grip": flow.grip,
                "blocked": st.get("state_name") == "堵转",
                "force": env.hand.grip_force if force_ui else None,
                "speed": env.hand.speed if force_ui else None,
                "count": disk_count if flow.name else None,
                "delete": list(flow.delete_target) if flow.delete_target else None,
                "grip_wait": grip_wait is not None}

    def run_omni(key) -> bool:
        """按 OmniPicker 的键位执行一个按键（流程在 omnipicker/keyflow.py）。True = 该退出回路。"""
        nonlocal engaged, grip_wait, settle, disk_count
        actions, note = flow.on_key(key)
        if note:
            print(f"\n  {note}")
        finish = False
        for action in actions:
            if finish and action in ("show_count", "help"):
                continue                    # 这个数据集采够了，下面统一回待机
            if action == "help":
                say_help()
            elif action == "prompt_line":
                print("\n" + flow.input_line(), end="", flush=True)
            elif action == "echo":
                print("\r" + flow.input_line() + "\x1b[K", end="", flush=True)
            elif action == "open_dataset":
                open_dataset()
            elif action == "auto_open":
                # 夹爪上电会自己合拢，程序又不主动动它，进测试时先张开一次（用户 10-20）。
                # 真正的指令是回路里按 flow.gripper_command 下发的，这里只说一声。
                print(f"\n  夹爪自动张开（速度 {env.hand.speed}）。空格夹紧、O 松开")
            elif action == "armed_banner":
                write_dataset_meta(grip_force=env.hand.grip_force, speed=env.hand.speed)
                print(f"\n  测试完：夹持力 {env.hand.grip_force}、速度 {env.hand.speed}，整个数据集都用这一组")
                print("\n  开始采集数据集，按回车开始采集：")
            elif action == "retest":
                print(f"\n  回到测试力度和速度：现在夹持力 {env.hand.grip_force}、速度 {env.hand.speed}。"
                      "调好按回车，新的一组写回 dataset.json")
            elif action == "restore_grip":
                # 续采且上次测好了力度速度：直接沿用，跳过测试，到待录
                info = flow.info or {}
                force = env.hand.set_grip_force(info["grip_force"])
                speed = env.hand.set_speed(info["speed"])
                preview_state.update(force=force, speed=speed)
                print(f"\n  沿用这个数据集的夹持力 {force}、速度 {speed}（不再测试）")
                print("\n  开始采集数据集，按回车开始采集：")
            elif action in ("force_up", "force_down"):
                before = env.hand.grip_force
                now_force = env.hand.step_grip_force(+1 if action == "force_up" else -1)
                print(f"\n  夹持力 {before} → {now_force}" + ("（已到头）" if now_force == before else ""))
                preview_state["force"] = now_force
            elif action in ("speed_up", "speed_down"):
                before = env.hand.speed
                now_speed = env.hand.step_speed(+1 if action == "speed_up" else -1)
                print(f"\n  速度 {before} → {now_speed}" + ("（已到头）" if now_speed == before else "")
                      + "（下一次夹紧 / 松开起生效，空格试一下）")
                preview_state["speed"] = now_speed
            elif action == "freeze_for_grip":
                # 臂停在当前设定点（compute() 不接合时返回保持姿态），照常录帧；
                # 夹牢 / 张开到位后由回路里的 grip_wait 让臂重新接合。flow.grip 已经是新方向了。
                if engaged:
                    mapper.disengage()
                    engaged = False
                closing = flow.gripper_command == 0.0
                goal = round(flow.gripper_command * env.hand.config.open_max)
                grip_wait = GripWait(time.perf_counter(), goal, closing)
                print("\n  臂冻住，" + ("等夹爪夹牢" if closing else "等夹爪张开到位") + "再接着跟")
            elif action == "grip_close":
                print(f"\n  夹爪夹紧（夹持力 {env.hand.grip_force}，碰到东西就停住夹着）")
            elif action == "grip_open":
                print("\n  夹爪松开")
            elif action == "engage":
                mapper.engage(hold_posture(env.arm, config.arm.num_joints))
                engaged = True
            elif action == "disengage":
                mapper.disengage()
                engaged = False
            elif action == "begin_record":
                if schedule.start() == "begin_record":
                    recorder.reset()
                    print(f"\n  ● 开始录「{flow.name}」episode {schedule.index}")
            elif action == "pause":
                # 臂滑到停稳这一小段照录，不然接着录时这一截变成一帧跳过去（10-20 实测 8~33 mm）
                last = recorder.last_qpos()
                settle = SettleWait(time.perf_counter(), last)
                print("\n  ‖ 暂停：臂冻住，停稳后不录帧。主臂随便摆")
            elif action == "resume":
                recorder.mark_resume()      # 暂停那段从时间戳里扣掉，否则实测帧率被拉低
                ramp = mapper.config.ramp_seconds
                print("\n  ● 接着录" + (f"（{ramp:.1f}s 渐入，这段跟得慢一点）" if ramp > 0 else ""))
            elif action == "save":
                if schedule.stop() == "end_episode":
                    before, index = schedule.done, schedule.index
                    finish = end_episode()
                    if schedule.done > before:
                        note_saved(index)
            elif action == "discard":
                dropped = len(recorder)
                if schedule.discard() == "discard":
                    recorder.reset()
                    print(f"\n  R 丢掉这条（{dropped} 帧），编号不变")
            elif action == "home":
                go_home()
            elif action == "ask_delete":
                # N：找这个数据集磁盘上编号最大的那条，等 y 确认（用户 10-20：要能连着往前删）
                recorder.flush(warn=False)          # 刚存的可能还在后台写，写完再数
                latest = next_episode_index(dataset_dir) - 1
                if latest < 0:
                    flow.set_delete_target(None, 0)
                    print(f"\n  数据集「{flow.name}」里一条都没有，没东西可删")
                else:
                    flow.set_delete_target(latest, len(list(dataset_dir.glob("episode_*.npz"))))
            elif action == "delete_latest":
                index = flow.delete_target[0]
                flow.delete_target = None
                recorder.flush(warn=False)
                gone = delete_episode(dataset_dir, index)
                this_run = index in saved_this_run
                saved_this_run.discard(index)
                if gone:
                    disk_count = max(0, disk_count - 1)
                if this_run:
                    session_saved[flow.name] = max(0, session_saved.get(flow.name, 0) - 1)
                schedule.forget(next_episode_index(dataset_dir), this_run)
                print(f"\n  N 删掉 episode {index:04d}（{gone} 个文件）。再按 N 接着往前删")
            elif action == "show_count":
                show_count()
            elif action == "quit":
                return True
        if flow.phase != "recording":
            grip_wait = None                # 暂停 / 存盘 / 丢弃：等夹爪那件事作废，接合由各自的动作管
        if flow.phase != "paused":
            settle = None                   # 接着录 / 存盘 / 丢弃：停没停稳都不用再等
        if finish:
            # 这个数据集采够了 -n 条：不退出，回待机等 P 开下一个
            print(f"\n  数据集「{flow.name}」这次采够 {target} 条了。")
            flow.dataset_done()
            say_help()
        return False

    if omni:
        say_help()                      # 待机的按键，下面跟着列已有的数据集

    try:
        with RawKeyboard() as kb:
            while True:
                t0 = time.perf_counter()
                key = kb.key()

                # 网页来的命令和键盘按键走同一套分支：映射成同样的键值，下面一行都不用改
                if link is not None:
                    cmd = link.poll_command()
                    if cmd and omni:
                        key = omni_web_key(cmd, getattr(link, "last", None) or {})
                    elif cmd:
                        key = REMOTE_KEYS.get(cmd, key)
                        # 收到就打一行。网页上点了没反应时，看这里最快分清是
                        # 「命令没到」还是「到了但臂没跟上」
                        print("\n  [网页] %s" % cmd)

                # OmniPicker 整套按键走 run_omni，下面那套老键位不再看这个键
                if omni:
                    if run_omni(key):
                        break
                    key = None
                    # 录制中按空格 / O 冻住的臂：夹牢（报夹住或合到底）/ 张开到位就重新接合，接着跟主臂
                    if grip_wait is not None:
                        st = env.hand.last_status()
                        how = grip_wait.update(time.perf_counter(), st.get("pos", 0), st.get("state"))
                        if how:
                            mapper.engage(hold_posture(env.arm, config.arm.num_joints))
                            engaged, grip_wait = True, None
                            ramp = mapper.config.ramp_seconds
                            said = {"gripped": f"夹住了（停在开度 {st.get('pos', 0) / env.hand.config.open_max:.2f}）",
                                    "arrived": "夹爪到位",
                                    "timeout": f"⚠ 等了 {GripWait.TIMEOUT_SECONDS:.0f}s 夹爪还没夹牢 / 到位，不等了"}[how]
                            print(f"\n  {said}，臂接着跟" + (f"（{ramp:.1f}s 渐入）" if ramp > 0 else ""))

                if key in ("q", "Q"):
                    # 收工也回 0 位，别把臂丢在最后一刻停的地方。放在 break 之前、
                    # finally 的 env.close() 之前 —— close() 之后就没有臂可以指挥了。
                    print("\n  q 退出，先回 0 位……")
                    mapper.disengage()
                    engaged = False
                    holding, held_gripper = False, None
                    go_home()
                    break

                if key == " ":
                    if schedule.start() == "begin_record":
                        # 接合和开录绑在一起，没有单独的离合键
                        mapper.engage(hold_posture(env.arm, config.arm.num_joints))
                        engaged = True
                        # 保持状态不能从上一条串过来：串过来的话夹爪一直冻着、
                        # 每帧还都被标成 held，而且没有任何报错。
                        holding, held_gripper = False, None
                        announce("begin_record")
                    else:
                        print("\n  已经在录了。s=做完存盘  r=丢掉重来")

                if key in ("s", "S"):
                    if schedule.stop() == "end_episode":
                        # 先松开再存：存盘期间主臂还在动的话，从臂不该跟
                        mapper.disengage()
                        engaged = False
                        holding, held_gripper = False, None
                        if link is not None:
                            # 上一条还在写盘的话，这次 save() 会在 flush() 里等它，
                            # 回路停多久就有多大 latch 风险 —— 网页上要看得见
                            link.publish_status(min_interval=0, busy="saving", recording=False,
                                                engaged=False, episode=schedule.index,
                                                samples=schedule.done,
                                                writing=recorder.is_writing())
                        done = end_episode()
                        go_home()
                        if done:
                            break
                    else:
                        print("\n  现在没在录。按空格开录。")

                if key in ("p", "P"):
                    # **只在录制中有效。** 不录时臂本来就不跟主臂，再给个"保持"
                    # 只会让人以为按错了。
                    if not schedule.recording:
                        print("\n  现在没在录，P 不起作用。先按空格开录。")
                    elif holding:
                        # 重新锚定：臂锚在它现在待着的地方，主臂锚点下一帧再取 ——
                        # 主臂当前位置就成了新"零点"，接着推就是从这儿算增量。
                        mapper.engage(hold_posture(env.arm, config.arm.num_joints))
                        engaged, holding, held_gripper = True, False, None
                        if link is not None:
                            link.publish_status(min_interval=0, holding=False, engaged=True,
                                                recording=schedule.recording,
                                                episode=schedule.index, samples=schedule.done)
                        ramp = mapper.config.ramp_seconds
                        print("\n  P 松开，从这儿接着跟"
                              + (f"（{ramp:.1f}s 渐入，这段会跟得慢一点）" if ramp > 0 else ""))
                    else:
                        mapper.disengage()      # compute() 不接合时返回 q_now，即原地保持
                        engaged, holding = False, True
                        held_gripper = gripper  # 夹爪也一起冻住
                        if link is not None:
                            # 当场发，不等下一个 5 Hz 节拍 —— 网页上点了要立刻看见
                            link.publish_status(min_interval=0, holding=True, engaged=False,
                                                recording=schedule.recording,
                                                episode=schedule.index, samples=schedule.done)
                        print("\n  ● P 保持：臂不动了，主臂随便摆。再按 P 从新位置接着跟")

                if key in ("r", "R"):
                    dropped = len(recorder)
                    if schedule.discard() == "discard":
                        recorder.reset()
                        mapper.disengage()
                        engaged = False
                        holding, held_gripper = False, None
                        print(f"\n  r 丢掉已录的 {dropped} 帧，臂回 0 位……")
                        # 重录之前也回 0 位。不回的话起点和前面每一条都不一样，
                        # 而"起点一致"正是重录想要的东西。
                        go_home()
                        print(f"  摆好现场按空格，重录 episode {schedule.index}")
                    else:
                        print("\n  现在没在录，没东西可丢。")

                if key == "LEFT":
                    outcome = schedule.delete_last()
                    if outcome == "recording":
                        print("\n  正在录，← 不动已存的数据。要丢这条按 r。")
                    elif outcome is None:
                        print("\n  没有刚存下的条目可删。← 只删最近存的那一条。")
                    else:
                        recorder.flush(warn=False)      # 可能还在后台写，等它写完再删
                        gone = delete_episode(collect_cfg.out_dir, outcome)
                        print(f"\n  ← 删掉 episode {outcome:04d}（{gone} 个文件）。"
                              f"下一条重新占这个号"
                              + (f"（本次 {schedule.done}/{target}）" if target else ""))

                if key in ("UP", "DOWN"):
                    print(f"\n  {config.end_effector} 不支持在这里调力（只有 omnipicker 支持）")

                if schedule.tick(len(recorder)) == "end_episode":
                    print(f"\n  到上限 {schedule.cap} 帧，自动存")
                    if omni:
                        # 和按 S 一样：松开、存、回 0 位、回待录
                        mapper.disengage()
                        engaged = False
                        flow.to_armed()
                        before, index = schedule.done, schedule.index
                    done = end_episode()
                    if omni:
                        if schedule.done > before:
                            note_saved(index)
                        go_home()
                        if done:            # 这个数据集采够了：回待机等 P，不退出
                            print(f"\n  数据集「{flow.name}」这次采够 {target} 条了。")
                            flow.dataset_done()
                        else:
                            show_count()
                        say_help()
                    elif done:
                        break

                # 臂一旦 latch 就什么目标都不接，而回路照样往下跑 —— 2026-09-03
                # 就是这样录出两条 600 帧一动不动的死数据。这里每帧查一次，
                # 33 ms 内就发现，不用等 20 秒录完才从警告里看出来。
                if env.arm.latched:
                    if link is not None:
                        # 马上让网页知道，不然人还在那儿按空格
                        link.publish_status(min_interval=0, latched=True, recording=False,
                                            engaged=False, episode=schedule.index,
                                            samples=schedule.done)
                    print("\n\n  ✗ 臂已刹车锁死（latch），目标全被拒，再录都是死数据。")
                    print("    原因看上面有没有 `goals stopped arriving` —— 通常是")
                    print("    回路被阻塞超过 0.25s。退出重开会自动清 latch。")
                    if schedule.recording:
                        print(f"    正在录的 {len(recorder)} 帧不保存。")
                    # 回 0 位不在这儿做：这时候 RT 流已经死了、控制器已经掉电，
                    # go_home() 走的就是那条死掉的流。等 finally 把 env 关干净了
                    # 再问再救 —— 见 offer_home_after_latch()。
                    latched_exit = True
                    break

                joints = leader.read_positions()
                if len(joints) < len(SO_JOINTS):
                    env.step_joints(hold_posture(env.arm, config.arm.num_joints),
                                    observe=False)
                    time.sleep(period)
                    continue

                # The tracker's own setpoint, not a fresh encoder read: reading
                # the encoders is a blocking state refresh plus an SDK call on
                # the critical path, and anchoring on a measurement lets tracking
                # lag shrink every step.
                q_cmd = mapper.compute(joints, hold_posture(env.arm, config.arm.num_joints))
                if omni:
                    # 扳机不起作用，夹爪只听空格 / O。None = 两个都还没按过，不发指令、原地不动
                    gripper = flow.gripper_command
                else:
                    gripper = mapper.gripper_command(joints)
                    if holding and held_gripper is not None:
                        # 夹爪也冻住：重新摆主臂时扳机难免蹭到，不冻就会把一次
                        # 莫名其妙的开合录进数据，而画面上什么都没发生。
                        gripper = held_gripper
                note = watch.check(t0, q_cmd)
                if note:
                    print("\n" + note)
                    if omni and flow.text_input:    # 正在打字：把输入那一行重新显示出来
                        print("\n" + flow.input_line(), end="", flush=True)
                # Only build an observation when there is something to record
                # into. Idle, it is a camera frame and three SDK round trips
                # thrown away -- and the jitter they add lands on the arm.
                # 到录制节拍才记一帧。回路 50 Hz、录制 30 Hz，所以不是每帧都记。
                #
                # **节拍从上一个节拍往前推，不是从当前帧推。** 写成
                # `next = max(t0, next) + record_period` 的话，相位每次都被对到
                # 当前帧上、小数余量丢掉，50/30 就退化成「固定两帧录一次」= 25 Hz ——
                # 而 npz 里的 fps 仍写 30，视频会放快、动作间隔也对不上。
                # 实测过：那样写出来是 25.0 Hz。
                due = t0 >= next_record
                # OmniPicker 暂停（第二次回车）时 schedule 还在录这一条，但这几帧不录 ——
                # 除了刚暂停、臂还没停稳的那一小段（settle）。
                # OmniPicker 不按节拍取「最新一张 + 当下的臂」了：录制中**每圈**都记（log_now），
                # 存盘前按腕部相机对齐（SyncRecorder）。
                log_now = synced and schedule.recording and (flow.records or settle is not None)
                recording = (not synced) and schedule.recording and due and (
                    (flow.records or settle is not None) if omni else True)
                if due:
                    next_record += record_period
                    # 回路真卡过（存盘、go_home）就别追着补帧，重新对齐即可
                    if next_record < t0:
                        next_record = t0 + record_period
                # 待机时原本不取画面（省一次相机 + 三次 SDK 往返，抖动会落到臂上）。
                # 远程模式下人得看着网页摆物体，所以待机时按一个很低的频率取一帧：
                # 默认 4 Hz，比录制时的 30 Hz 轻得多，也只在真要发帧时才取。
                preview = (link is not None and not recording
                           and link.frame_due(preview_interval))
                # gripper 为 None 时 env 不碰夹爪（OmniPicker 还没按过空格）
                obs = env.step_joints(q_cmd, gripper, observe=recording or preview)

                qpos_now = None
                if log_now:
                    # 下发的夹爪指令原样记（None = 还没按过空格 / O，对齐时用实测开合）
                    qpos_now = recorder.log(env, gripper, held=holding)
                elif recording and obs is not None:
                    # action 的末端维记下发的指令；OmniPicker 还没按过空格就记实测开合
                    recorder.add(obs, gripper if gripper is not None else env.hand.get_opening(),
                                 held=holding)
                    qpos_now = obs.observation["qpos"]
                    if settle is None and not engaged and grip_wait is None:
                        recorder.clutch_dropped += 1    # 等夹爪时臂冻住是动作的一部分，不算
                if settle is not None and qpos_now is not None:
                    # 暂停后臂还在往最后一个目标滑：照录，停稳了才真的不录
                    how = settle.update(time.perf_counter(), qpos_now)
                    if how == "timeout":
                        print(f"\n  ⚠ 暂停后等了 {SettleWait.TIMEOUT_SECONDS:.0f}s 臂 / 夹爪还没停稳，"
                              "不录了（接着录时这里可能有一点跳）")
                    if how:
                        settle = None

                if link is not None and obs is not None:
                    # 相机给的是 RGB，让 link 转成 BGR 再编码；内部限流，多调不亏
                    link.publish_frame(preview_image(obs, camera_keys), rgb=True)

                # 状态行，约 5 Hz。写终端会阻塞，而阻塞的回路会被臂的 sender
                # 读成「回路死了」然后刹车 latch —— 所以不能每帧都打。
                if t0 - last_line >= 0.2:
                    last_line = t0
                    state = "‖保持" if holding else ("接合" if engaged else "松开")
                    if schedule.recording:
                        seconds = len(recorder) / collect_cfg.hz   # 录制帧率，不是控制频率
                        tag = (f"● ep{schedule.index} {seconds:5.1f}/"
                               f"{collect_cfg.episode_time_s:.0f}s" if schedule.cap
                               else f"● ep{schedule.index} {seconds:5.1f}s")
                    else:
                        tag = f"○ 待机 ep{schedule.index}"
                    grip_force = env.hand.grip_force if force_ui else None
                    if omni:
                        st = env.hand.last_status()
                        shown = min(1.0, st.get("pos", 0) / env.hand.config.open_max)
                        blocked = st.get("state_name") == "堵转"   # 自适应：碰到东西停住、按夹持力夹着
                        parked = flow.phase == "paused"
                        # 只在内容变了的时候打一行（用户 10-20：不要一直打印，改完一次打印一次）。
                        # 原来 5 Hz 用 \r 原地刷，一行带中文超过终端宽度就折行，\r 只回到最后一截
                        # 的行首，结果整屏都是重复的状态。开度、秒数这种一直在变的数不放进来，
                        # 否则夹爪一动又刷屏；录了多久存盘时会打出来。
                        phase = {"idle": "待机", "name": "输任务名", "confirm": "确认续采",
                                 "prompt": "输prompt", "test": "测试力度速度", "armed": "待录",
                                 "recording": "● 录制", "paused": "‖ 暂停",
                                 "delete": "确认删除"}.get(flow.phase, flow.phase)
                        if grip_wait is not None:
                            phase += "（臂冻住等夹爪）"
                        where = (f"「{flow.name}」已存{disk_count}条"
                                 if dataset_dir is not None and flow.name else "还没开数据集")
                        grip_txt = "夹住" if blocked else {None: "未动", "closed": "夹紧",
                                                          "open": "松开"}[flow.grip]
                        omni_line = (f"{phase}  {where}  夹爪{grip_txt}  力{grip_force}  "
                                     f"速{env.hand.speed}")
                        if omni_line != last_omni_line and not flow.text_input:
                            print(f"\n  ▸ {omni_line}", flush=True)
                            last_omni_line = omni_line
                    else:
                        shown = gripper
                        hand_txt = f"手 {gripper:.2f}"
                        parked = holding
                    # 预览线程读这个字典。只在这里写（约 5 Hz），dict 单键赋值
                    # 在 CPython 里是原子的，不用锁。
                    preview_state.update(
                        recording=schedule.recording, index=schedule.index,
                        frames=len(recorder), gripper=float(shown),
                        done=schedule.done, target=target, holding=parked,
                        force=grip_force, speed=env.hand.speed if force_ui else None)
                    progress = f"  {schedule.done}/{target}" if target else ""
                    if not omni:                # OmniPicker 的状态行在上面，变了才打
                        print(f"\r  {tag}   {state}   {hand_txt}{progress}   ",
                              end="", flush=True)
                    if link is not None:
                        link.publish_status(
                            min_interval=0,          # 外面这层 0.2s 已经限过流了
                            episode=schedule.index, samples=schedule.done,
                            target=target, recording=schedule.recording,
                            engaged=engaged, frames=len(recorder),
                            gripper=round(float(shown), 3),
                            fps=recorder.measured_fps() or collect_cfg.hz,
                            # 臂 latch 之后目标全被拒，网页上表现为「按了没反应」；
                            # writing 为真说明下一次存盘会在 flush 里等，回路要停
                            latched=bool(getattr(env.arm, "latched", False)),
                            writing=recorder.is_writing(),
                            holding=holding,     # P 保持中：臂不跟主臂，夹爪也冻着
                            gripper_force=grip_force,    # None = 末端不支持调力
                            # OmniPicker：网页凭它画面板（第几步、提示、输入框、力和速度…）
                            omni=omni_status() if omni else None,
                        )

                sleep = period - (time.perf_counter() - t0)
                if sleep > 0:
                    time.sleep(sleep)
    finally:
        print()
        # 预览先收：它每帧在 peek 相机，而下面 env.close() 要 release 掉
        # VideoCapture —— 释放一个还有人在读的 capture 是段错误，不是异常。
        if previewer is not None:
            previewer.stop()
            previewer.join(timeout=2.0)
            if previewer.error:
                print(f"  ⚠ {previewer.error}")
        if watch.notes:
            print(f"  本次臂有 {len(watch.notes)} 次没完全跟上，最后一条：")
            print("  " + watch.notes[-1].strip())
        if schedule.recording and len(recorder) > 0:
            # 录到一半按 q，意思是不要了。以前这里会偷偷存下来，结果是
            # 「我只录了一条，怎么出来两条」。要留就先按 s 正常结束。
            print(f"  这条录了 {len(recorder)} 帧，没按 s 结束，不保存")

        # 最后一条还在后台写。这里不等就会随进程一起消失 —— 硬件先放，
        # 因为写盘可能要几秒，臂和主臂没必要一直占着。
        env.close()
        leader.disconnect()
        recorder.flush(warn=False)     # 退出时这次等待是预期的，不是回路被堵
        error = recorder.take_write_error()
        if error:
            print(f"  ⚠ {error}")
        if omni:
            # 按数据集报：这次碰过哪几个、各存了几条、磁盘上一共几条
            print(f"\n  这次的数据集（根目录 {out_dir}）：")
            for name, saved in session_saved.items():
                total = len(list((out_dir / name).glob("episode_*.npz")))
                print(f"    {name}：这次存了 {saved} 条，磁盘上一共 {total} 条")
            if not session_saved:
                print("    一个都没开")
            print("  续采：再跑 collect，按 P 输入同一个任务名、按 y")
            print(f"  这次的控制台输出：{log_path}")
        else:
            print(f"\n  本次采了 {schedule.done} 条"
                  + (f"（目标 {target}）" if target else "")
                  + f"，磁盘上一共 {schedule.index} 条，在 {out_dir}")
            if link is not None:
                # 网页看到 finished 就知道这次采集结束了，会自己跳去预览
                link.close(episode=schedule.index, samples=schedule.done, target=target)
            if target and schedule.done < target:
                print(f"  再跑一次 collect 会从 episode {schedule.index} 接着编，不会覆盖。")
            print(f"  转成 lerobot 格式：")
            print(f"    ~/miniforge3/envs/lerobot/bin/python \\")
            print(f"        {Path(__file__).parent / 'to_lerobot.py'} --src {out_dir} --repo-id vel_ar5_bolt")
            print(f"  这次的控制台输出：{log_path}")

    # 放在 try/finally **之后**：要等 env.close() 把会话放干净，救援才连得上
    # 控制器（它自己会再开一个）。异常退出时走不到这儿，那是有意的 —— 那种
    # 情况下堆栈才是要看的东西，不该被一个「要不要回位」的提问盖住。
    if latched_exit:
        offer_home_after_latch(config, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
