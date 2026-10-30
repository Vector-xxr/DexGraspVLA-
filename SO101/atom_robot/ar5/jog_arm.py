"""键盘点动 AR5 末端，同时看两路 RGB 实时画面（顶部 335 + 腕部 305）。

    cd 算法/DexGraspVLA-/SO101
    python -m atom_robot.ar5.jog_arm --mock      # 模拟臂 + 模拟画面，先熟悉键位
    python -m atom_robot.ar5.jog_arm             # 真机（先确认急停有人守）

键位 —— 终端里按、画面窗口里按都行，按住 = 连续走（靠键盘自动重复）：

    w / s   +x / -x        i / k   绕 x 转 + / -
    a / d   +y / -y        j / l   绕 y 转 + / -
    r / f   +z / -z        u / o   绕 z 转 + / -
    t       基座系 ↔ 法兰系（平移和旋转一起切）
    1 2 3   步长档：细 0.5mm/0.25°、中 2mm/1°、粗 5mm/2.5°（每次按键）
    空格     等臂停稳后存一组：关节角、末端位姿、两路原图（手眼标定用）
    q       退出

臂走的是 `env.step()` —— 和 teleop、策略同一条路：实时流、本地逆解、跳变门、
工作空间盒、缰绳全都在。旋转绕法兰原点（TCP 在法兰上，tcp-z 已归 0）。

**空闲时也每拍发一个零增量**：发送线程 0.25 s 收不到目标就刹车 latch，
latch 之后臂掉电抱闸，只能退出重开。

**工具在工作空间盒外就不启动**：零增量也会被盒子夹回盒面，臂会自己走过去。

退出时和 teleop 一样 `env.stop()` + `env.close()`，臂停在原地掉电抱闸。

空格存的东西在 `--out` 目录：`poses.npz`（joints 7 维 rad、ee_pose、images=腕部
BGR 原图、images_high、n_corners）和每组两张 jpg。目录里已有 poses.npz 就接着往后存。
"""

import argparse
import os
import queue
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
from atom_robot.ar5.transforms import axis_angle_to_matrix  # noqa: E402

WRIST, HIGH = "cam_front", "cam_high"

# (平移 m, 旋转 rad) / 每次按键
STEPS = {"1": (0.0005, np.radians(0.25)), "2": (0.002, np.radians(1.0)), "3": (0.005, np.radians(2.5))}
MOVE_KEYS = {
    "w": (0, +1), "s": (0, -1), "a": (1, +1), "d": (1, -1), "r": (2, +1), "f": (2, -1),
    "i": (3, +1), "k": (3, -1), "j": (4, +1), "l": (4, -1), "u": (5, +1), "o": (5, -1),
}

# 和 auto_collect_poses.py 同一块板（board_probe 实测）。这里只数角点，边长不影响。
BOARD_SIZE, SQUARE_LEN, MARKER_LEN = (7, 9), 0.025, 0.018

REST_TOL = 2e-4        # rad，一拍里各关节变化都小于它算「没动」
REST_TICKS = 8         # 连续这么多拍没动才算停稳
FRAME_LAG = 0.25       # s，停稳之后再等一下，保证拿到的是停稳之后拍的帧


class RawKeyboard:
    """终端单键非阻塞读；stdin 不是终端就什么都不读（只靠窗口按键）。"""

    def __enter__(self):
        self.fd = sys.stdin.fileno()
        try:
            self.saved = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
        except termios.error:
            self.saved = None
        return self

    def __exit__(self, *exc_info):
        if self.saved is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.saved)

    def keys(self):
        out = []
        while self.saved is not None and select.select([self.fd], [], [], 0)[0]:
            out.append(sys.stdin.read(1))
        return out


class Preview(threading.Thread):
    """两路画面拼一张窗口。**GUI 全在本线程**，主回路只写 state、读 keys。

    窗口里的按键经 `cv2.waitKey` 放进 keys 队列，所以焦点在窗口上也能操作
    （collect 的预览只从终端读键，点了窗口就不响应，这里没有这个问题）。
    """

    def __init__(self, cameras, state, keys, hz=15.0):
        super().__init__(name="jog-preview", daemon=True)
        self.cameras, self.state, self.keys = cameras, state, keys
        self.period = 1.0 / hz
        self._halt = threading.Event()
        self.error = None

    def stop(self):
        self._halt.set()

    def run(self):
        import cv2

        aru = cv2.aruco
        board = aru.CharucoBoard(BOARD_SIZE, SQUARE_LEN, MARKER_LEN,
                                 aru.getPredefinedDictionary(aru.DICT_5X5_50))
        board.setLegacyPattern(False)
        detector = aru.CharucoDetector(board)
        n_max = (BOARD_SIZE[0] - 1) * (BOARD_SIZE[1] - 1)
        win, created = "jog_arm", False
        try:
            while not self._halt.is_set():
                started = time.monotonic()
                frames = self.cameras.peek() if self.cameras is not None else {}
                tiles = []
                for name in (HIGH, WRIST):
                    if frames.get(name) is None:
                        continue
                    bgr = cv2.cvtColor(frames[name], cv2.COLOR_RGB2BGR)
                    label = name
                    if name == WRIST:
                        ch, cid, _, _ = detector.detectBoard(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
                        n = 0 if cid is None else len(cid)
                        self.state["corners"] = n
                        if n:
                            aru.drawDetectedCornersCharuco(
                                bgr, np.asarray(ch, np.float32).reshape(-1, 1, 2),
                                np.asarray(cid, np.int32).reshape(-1, 1))
                        label = f"{name}  charuco {n}/{n_max}"
                    if tiles and bgr.shape[0] != tiles[0].shape[0]:
                        h = tiles[0].shape[0]
                        bgr = cv2.resize(bgr, (int(bgr.shape[1] * h / bgr.shape[0]), h))
                    cv2.rectangle(bgr, (0, 0), (bgr.shape[1], 24), (0, 0, 0), -1)
                    cv2.putText(bgr, label, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1)
                    tiles.append(bgr)
                if tiles:
                    canvas = np.hstack(tiles)
                    # Hershey 字体没有中文字形，状态条只写 ASCII。
                    h, w = canvas.shape[:2]
                    cv2.rectangle(canvas, (0, h - 26), (w, h), (0, 0, 0), -1)
                    cv2.putText(canvas, self.state.get("line", ""), (8, h - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                    if not created:
                        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
                        cv2.resizeWindow(win, min(1600, 640 * len(tiles)), 520)
                        created = True
                    cv2.imshow(win, canvas)
                code = cv2.waitKey(1)
                if code != -1 and 0 < (code & 0xFF) < 128:
                    self.keys.put(chr(code & 0xFF))
                self._halt.wait(max(0.0, self.period - (time.monotonic() - started)))
        except Exception as exc:  # noqa: BLE001
            self.error = f"预览线程退出：{type(exc).__name__}: {exc}"
        finally:
            try:
                if created:
                    cv2.destroyWindow(win)
                    cv2.waitKey(1)
            except Exception:  # noqa: BLE001
                pass


class Saver:
    """空格存一组。每存一次就把 poses.npz 整个重写，中途崩了也不丢前面的。"""

    def __init__(self, out_dir: Path):
        self.out = out_dir
        self.out.mkdir(parents=True, exist_ok=True)
        self.path = self.out / "poses.npz"
        self.rows = {"joints": [], "ee_pose": [], "images": [], "images_high": [], "n_corners": []}
        if self.path.exists():
            old = np.load(self.path)
            for k in self.rows:
                if k in old:
                    self.rows[k] = list(old[k])
            print(f"  {self.path} 里已有 {len(self.rows['joints'])} 组，接着往后存")

    def save(self, q, ee, frames, n_corners) -> str:
        import cv2

        k = len(self.rows["joints"])
        wrist = cv2.cvtColor(frames[WRIST], cv2.COLOR_RGB2BGR)
        high = (cv2.cvtColor(frames[HIGH], cv2.COLOR_RGB2BGR) if frames.get(HIGH) is not None
                else np.zeros_like(wrist))
        self.rows["joints"].append(np.asarray(q, np.float64))
        self.rows["ee_pose"].append(np.asarray(ee, np.float64))
        self.rows["images"].append(wrist)
        self.rows["images_high"].append(high)
        self.rows["n_corners"].append(int(n_corners))
        cv2.imwrite(str(self.out / f"pose_{k:02d}_n{n_corners}_wrist.jpg"), wrist)
        cv2.imwrite(str(self.out / f"pose_{k:02d}_high.jpg"), high)
        np.savez(self.path, **{key: np.array(v) for key, v in self.rows.items()})
        return f"第 {k} 组  角点 {n_corners}  -> {self.path}（共 {k + 1} 组）"


def to_base(delta, pose, tool_frame: bool):
    """法兰系下的增量换成 env.step 要的基座系增量：p_b = R·p_t，ω_b = R·ω_t。"""
    if not tool_frame:
        return delta
    R = axis_angle_to_matrix(np.asarray(pose[3:6], np.float64))
    return np.concatenate([R @ delta[:3], R @ delta[3:6]])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mock", action="store_true", help="模拟臂 + 模拟画面，不碰任何硬件")
    parser.add_argument("--hz", type=float, default=20.0, help="控制频率")
    parser.add_argument("--step", choices=sorted(STEPS), default="2", help="起始步长档")
    parser.add_argument("--max-step", type=float, default=0.01, help="每拍平移上限，米")
    parser.add_argument("--out", default="calib_poses_manual", help="空格存的东西放哪")
    parser.add_argument("--no-preview", action="store_true", help="不开画面窗口")
    args = parser.parse_args()

    config = AR5Config(mock=args.mock)
    config.use_hand = False                 # 不碰夹爪串口；gripper 维度照发、NullHand 不执行
    config.arm.use_realtime = True
    config.arm.max_step_translation = args.max_step
    if args.mock:
        config.settle_time = 0.0

    env = AR5Env(config)
    env.connect()
    keys: "queue.Queue[str]" = queue.Queue()
    state = {"line": "", "corners": 0}
    preview = None
    saver = Saver(Path(args.out))

    try:
        env.reset(fake=True)
        outside = [(a, f, c) for a, f, c in env.pinned_axes() if c < 0.0]
        if outside:
            print("\n  工具在工作空间盒外，不启动（零增量也会把臂拖回盒面）：")
            for axis, face, c in outside:
                print(f"    {axis} 超出 {face} 面 {-c * 1000:.0f} mm")
            print("  先把臂挪进盒子，或者改 config 的 workspace_min/workspace_max。")
            return 1
        report = env.posture_report()
        print(f"\n  姿态：可操作度 {report['manipulability']:.4f}，"
              f"最紧的 j{report['worst_joint'] + 1} 用掉行程 {report['worst_usage'] * 100:.0f}%")
        if report["near_singularity"]:
            print("  ⚠ 接近奇异，很多方向的目标会被逆解拒掉")

        # 没有图形界面时 Qt 会直接 abort 整个进程（try 接不住），臂还连着 —— 先查。
        if not args.no_preview and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            print("  ⚠ 没有 DISPLAY，开不了画面窗口（ssh 加 -X）。只用终端按键。")
        elif not args.no_preview:
            preview = Preview(env.cameras, state, keys)
            preview.start()

        print(__doc__[__doc__.index("键位 ——"):__doc__.index("臂走的是")])
        rt = getattr(env.arm, "_rt_active", False)
        print(f"  控制：{'实时关节流 1 kHz' if rt else '排队 MoveL'}  {args.hz:.0f} Hz"
              f"   {'模拟臂' if args.mock else config.arm.ip}\n")

        period = 1.0 / args.hz
        step_key, tool_frame = args.step, False
        capture = None              # None / {"still": 拍数, "q": 上一拍关节, "t_rest": 停稳时刻}
        last = dict(env.stats)
        note, last_print = "", 0.0
        with RawKeyboard() as kb:
            while True:
                t0 = time.time()
                pressed = kb.keys()
                while not keys.empty():
                    pressed.append(keys.get_nowait())

                delta = np.zeros(6)
                quit_ = False
                for ch in pressed:
                    ch = ch.lower()
                    if ch == "q":
                        quit_ = True
                    elif ch == "t":
                        tool_frame = not tool_frame
                    elif ch in STEPS:
                        step_key = ch
                    elif ch == " " and capture is None:
                        capture = {"still": 0, "q": None, "t_rest": None}
                        note = "等臂停稳……"
                    elif ch in MOVE_KEYS and capture is None:
                        axis, sign = MOVE_KEYS[ch]
                        trans, rot = STEPS[step_key]
                        delta[axis] += sign * (trans if axis < 3 else rot)
                if quit_:
                    break

                if np.any(delta):
                    delta = to_base(delta, env.arm.get_ee_pose(), tool_frame)
                # 空闲也要发：sender 收不到目标 0.25 s 就 latch。
                env.step(np.concatenate([delta, [1.0]]))

                if getattr(env.arm, "latched", False):
                    print("\n\n  ⚠ 发送线程 latch 了（回路卡住或被控制器停了），臂已不受控。退出重开。")
                    break

                for key, label in (("unreachable", "逆解/跳变门拒了"), ("workspace_clamped", "顶到工作空间盒"),
                                   ("delta_clamped", "单拍增量被截")):
                    if env.stats[key] > last.get(key, 0):
                        note = f"{label} {env.stats[key] - last.get(key, 0)} 次"
                last = dict(env.stats)

                if capture is not None:
                    q = env.arm.get_joint_positions()
                    if capture["q"] is not None and np.max(np.abs(q - capture["q"])) < REST_TOL:
                        capture["still"] += 1
                    else:
                        capture["still"], capture["t_rest"] = 0, None
                    capture["q"] = q
                    if capture["still"] >= REST_TICKS and capture["t_rest"] is None:
                        capture["t_rest"] = time.time()
                    if capture["t_rest"] is not None and time.time() - capture["t_rest"] > FRAME_LAG:
                        frames = env.cameras.peek()
                        if frames.get(WRIST) is None:
                            note = "腕部相机没帧，没存"
                        else:
                            note = saver.save(env.arm.get_joint_positions(), env.arm.get_ee_pose(),
                                              frames, state.get("corners", 0))
                            print(f"\n  ✓ 存了{note}")
                        capture = None

                if time.time() - last_print > 0.2:
                    last_print = time.time()
                    pose = env.arm.get_ee_pose()
                    trans, rot = STEPS[step_key]
                    r = env.posture_report()
                    frame_name = "FLANGE" if tool_frame else "BASE"
                    state["line"] = (f"{frame_name}  step {trans * 1000:g}mm/{np.degrees(rot):g}deg  "
                                     f"xyz {pose[0]:+.3f} {pose[1]:+.3f} {pose[2]:+.3f}  "
                                     f"j{r['worst_joint'] + 1} {r['worst_usage'] * 100:.0f}%  "
                                     f"saved {len(saver.rows['joints'])}"
                                     + ("  [WAIT STILL]" if capture else ""))
                    print(f"  [{'法兰系' if tool_frame else '基座系'}] 步长 {trans * 1000:g}mm/"
                          f"{np.degrees(rot):g}°  xyz {np.round(pose[:3] * 1000).astype(int).tolist()}mm  "
                          f"最紧 j{r['worst_joint'] + 1} {r['worst_usage'] * 100:.0f}%  "
                          f"角点 {state.get('corners', 0)}  已存 {len(saver.rows['joints'])}  {note}      ",
                          end="\r", flush=True)
                    if preview is not None and preview.error:
                        print(f"\n  ⚠ {preview.error}")
                        preview = None

                remaining = period - (time.time() - t0)
                if remaining > 0:
                    time.sleep(remaining)
    except KeyboardInterrupt:
        print("\n  中断")
    finally:
        if preview is not None:
            preview.stop()
            preview.join(timeout=2.0)
        print(f"\n\n  env stats {env.stats}")
        env.stop()
        env.close()
        print(f"  已断开。存了 {len(saver.rows['joints'])} 组在 {saver.path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
