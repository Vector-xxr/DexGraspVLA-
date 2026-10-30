"""C4 部署脚本（待测试.md C4）：AR5 + OmniPicker + 两台 Orbbec 跑 DexGraspVLA 控制器。

照上游 inference.py：继承 RoboticsSystem，**只重写碰硬件的方法**。SAM / Cutie、观测处理、预测、
每次执行前 6 步、手动框选、按回车结束、超时、存部署数据，全部用上游原来的代码。

和上游不同的地方（都是硬件逼的，原因写在各方法里；README「已知问题」也有）：
- 模型：用检查点里存的 8 维配置（上游读仓库 yaml，那是 13 维的）
- 关节限位、起始位姿、夹爪参数：inference_utils/config_ar5.yaml（上游 config.yaml 不改）
- 发关节目标：上游 low_level_control 的做法（每个新目标做三次样条、200 ms 走到）照留，但按 50 Hz
  （= 采集回路）取点，每个点经 atom_robot 的 step_joints 发出去 —— 和采集同一条路：软限位余量、
  每步限幅、台面地板检查、驱动的限速和跳变门都在
- AR5 的实时流 0.25 s 收不到目标就锁停：发送线程从连上臂起一直跑，上游「不发」的时候改成原地保持；
  模型和 SAM / Cutie 在连臂之前加载（上游是之后），免得加载的十几秒里断粮
- 相机：atom_robot 读出来是 RGB，转成 BGR（训练数据和上游都是 BGR）
- 夹爪：往合拢方向多给余量（README「部署：夹爪那一维要换算」）；每次开始前先张开
- 抓完：没有放置位就直接回起始位姿、不松手（上游是移到放置位松手）
- 只支持手动框选（--manual）；规划器（C6）还没接
- 和采集互斥：抢 collect 用的同一把锁 /tmp/atom_ar5_collect.lock，抢不到就不启动

模式（--mode）：
  replay   模拟臂 + 模拟夹爪 + 拿 zarr 里录好的一条当相机（按墙上时间放），整条链路离线跑通
  mockcam  模拟臂 + 模拟夹爪 + 真相机：真画面上框选、看在线掩码；臂、夹爪都不动
  real     真臂、真夹爪、真相机

用法（在仓库根目录；上真机前先跑 replay、mockcam）：
  PYTHONPATH= ~/miniforge3/envs/dexgraspvla/bin/python deploy_ar5.py --manual --mode replay --auto \\
      --replay-zarr ~/H100_dexgraspvla/Datasets/atom_black_sleeve_20261021/black_sleeve.zarr --replay-episode 0
"""
import argparse
import os
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import offline_replay  # noqa: E402  先 import：设好 TORCH_HOME、matplotlib 用 Agg、load_policy

import gc

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
import zarr
from cutie.inference.inference_core import InferenceCore
from cutie.utils.get_default_model import get_default_model
from segment_anything import SamPredictor, sam_model_registry

import inference
from inference import RoboticsSystem
from inference_utils.utils import cubic_spline_interpolation_7d


SO101 = HERE / "SO101"
sys.path.insert(0, str(SO101))  # atom_robot、omnipicker 两个包在这里（10-30 起就在本仓库里）
from atom_robot.ar5.ar5_env import AR5Env      # noqa: E402
from atom_robot.ar5.config import AR5Config    # noqa: E402

COLLECT_LOCK = "/tmp/atom_ar5_collect.lock"    # 和 collect.py 的 acquire_lock 同一把


def _merge(base, over):
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge(base[k], v)
        else:
            base[k] = v
    return base


def _timed(label, fn, timeout):
    """收尾的一步：后台线程里跑，最多等 timeout 秒，打印进度。返回是否按时做完。
    10-22 退出时卡住、看不出卡在哪一步，所以每一步都单独计时。"""
    done, err = threading.Event(), []

    def run():
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            err.append(exc)
        finally:
            done.set()

    threading.Thread(target=run, daemon=True).start()
    print(f"  收尾：{label} …", end="", flush=True)
    if done.wait(timeout):
        print(" 完成" + (f"（出错：{type(err[0]).__name__}: {err[0]}）" if err else ""), flush=True)
        return True
    print(f" {timeout:.0f} 秒没做完，跳过", flush=True)
    return False


def _serial_holders(port):
    """哪些进程开着这个串口（扫 /proc，和 collect.py 的 who_holds_serial 同一个办法）。只查不杀。"""
    try:
        target = os.path.realpath(port)
    except OSError:
        return []
    holders = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            if any(os.path.realpath(fd) == target for fd in (proc / "fd").iterdir()):
                holders.append((int(proc.name), (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()))
        except OSError:
            continue
    return holders


def _take_collect_lock():
    import fcntl
    handle = open(COLLECT_LOCK, "w")  # noqa: SIM115  故意不关：关了锁就掉
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        return None
    handle.write(f"{os.getpid()} deploy_ar5\n")
    handle.flush()
    return handle


class ReplayCameras:
    """录像当相机：按墙上时间（30 Hz）给 zarr 里一条的帧。接口同 atom_robot 的相机，read() 返回 RGB。
    start() 之前一直给第 0 帧（框选、初始化 SAM 用），start() 之后开始走。"""

    def __init__(self, zarr_path, episode, head_key, wrist_key, hz=30.0):
        z = zarr.open(os.path.expanduser(zarr_path), "r")
        ends = z["meta/episode_ends"][:]
        s, e = (0 if episode == 0 else int(ends[episode - 1])), int(ends[episode])
        self.head = z["data/rgbm"][s:e][..., :3]          # BGR
        self.wrist = z["data/right_cam_img"][s:e]           # BGR
        self.state = z["data/right_state"][s:e]
        attrs = dict(z.attrs)
        self.bbox = np.array(attrs["head_bbox"][attrs["episodes"][episode]], dtype=np.float32)
        self.T, self.hz, self.keys, self.t0 = e - s, hz, (head_key, wrist_key), None

    def connect(self):
        pass

    def disconnect(self):
        pass

    def start(self):
        self.t0 = time.perf_counter()

    def index(self):
        return 0 if self.t0 is None else min(self.T - 1, int((time.perf_counter() - self.t0) * self.hz))

    def seconds(self):
        return self.T / self.hz

    def read(self):
        i = self.index()
        return {self.keys[0]: np.ascontiguousarray(self.head[i][..., ::-1]),
                self.keys[1]: np.ascontiguousarray(self.wrist[i][..., ::-1])}

    peek = read


class ViewerFeed:
    """把最新的俯视画面、腕部画面、掩码和状态写进共享内存，给 deploy_viewer.py（独立进程）显示。
    这边只做内存拷贝；窗口在别的进程里，卡了也碰不到发关节目标的线程。"""

    def __init__(self, snapshot=None):
        import subprocess
        from multiprocessing import shared_memory

        import deploy_viewer as dv
        size = dv.HDR_BYTES + 2 * dv.IMG + dv.H * dv.W
        self.shm = shared_memory.SharedMemory(create=True, size=size)
        self.hdr, self.head, self.wrist, self.mask = dv.views(self.shm.buf)
        self.hdr[:] = 0
        self.hdr[1] = 1  # alive
        cmd = [sys.executable, str(HERE / "deploy_viewer.py"), self.shm.name] + ([snapshot] if snapshot else [])
        # 新会话：终端里的 Ctrl+C 不传给它（10-22 按 Ctrl+C 时窗口进程先崩了一大段报错）
        self.proc = subprocess.Popen(cmd, env={**os.environ, "PYTHONPATH": ""}, start_new_session=True)

    def write(self, head, wrist, mask, status):
        self.hdr[0] += 1                      # 奇数 = 正在写
        np.copyto(self.head, head)
        np.copyto(self.wrist, wrist)
        if mask is None:
            self.mask[:] = 0
        else:
            np.copyto(self.mask, mask)
        self.hdr[2:8] = status
        self.hdr[0] += 1                      # 偶数 = 写完了

    def close(self):
        self.hdr[1] = 0
        try:
            self.proc.wait(timeout=2)
        except Exception:  # noqa: BLE001
            self.proc.terminate()
        self.shm.close()
        self.shm.unlink()


class GripperAdapter:
    """上游 execute_action 调 self.right_hand.execute_action(action[k, 7:]) 拿手的目标。
    夹爪只有 1 维开合（0 合拢、1 全开）；换算（多给余量）在 low_level_hand_control 里做，因为要看实测开合。"""

    def __init__(self):
        self.issue_times = []  # 每执行一步记一次时刻，看实际执行频率

    def execute_action(self, a):
        self.issue_times.append(time.perf_counter())
        return np.array([float(np.clip(a[0], 0.0, 1.0))])


class AR5System(RoboticsSystem):
    def __init__(self, args):
        self.args = args
        self.config = self.load_inference_config()
        self.running = True
        self._arm_lock = threading.Lock()
        self._scripted = None          # move_to_target_joint_angle 的整段轨迹 (轨迹, 起始时刻, 每点间隔)
        self._traj = None              # 策略目标的样条（上游 arm_traj_interp）
        self._hold = None              # 最近发出去的目标
        self._last_grip_cmd = None
        self.publish_log, self.grip_log = [], []
        self.max_publish_gap = 0.0
        self._need_gui = args.bbox is None and not (args.mode == "replay" and args.auto)  # 要不要弹窗框选
        self._viewer, self._view_mask = None, None
        # 顺序和上游不同：模型先加载，再连臂（AR5 实时流 0.25 s 收不到目标就锁停，加载要十几秒）
        self.init_controller()
        print("init_controller done")
        # 模型、SAM、Cutie 这些常驻对象移出垃圾回收的扫描范围：之后主线程每次回收只扫新对象。
        # 不这样的话一次完整回收要 120~130 ms，期间占着 GIL，发送线程插不上手（离 0.25 s 锁停太近）。
        # 放在连臂之前：这一次完整回收本身也要 130 ms，那时实时流还没开
        gc.collect()
        gc.freeze()
        self.init_robot()
        print("init_robot done")
        self.init_robot_state()
        print("init_robot_state done")
        self.init_planner()
        self.init_utils_and_data()
        print("init_utils_and_data done")
        if not args.no_view:
            self._viewer = ViewerFeed(args.view_snapshot)
            print("实时画面窗口已打开（独立进程；关掉窗口不影响部署）")
        self.init_threads()
        print("init_threads done")

    # ── 配置、模型 ───────────────────────────────────────────────────

    def load_inference_config(self):
        config = RoboticsSystem.load_inference_config(self)  # 上游 config.yaml
        return _merge(config, yaml.safe_load(open(HERE / "inference_utils" / "config_ar5.yaml")))

    def init_controller(self):
        self.device = torch.device("cuda:0")
        # 上游读仓库 yaml 建模型（13 维）；我们用检查点里存的 8 维配置。--ckpt 优先于 config_ar5.yaml
        ck = Path(os.path.expanduser(self.args.ckpt or self.config["checkpoint"]))
        self.ckpt_path = ck if ck.is_absolute() else (Path.cwd() / ck if (Path.cwd() / ck).exists() else HERE / ck)
        self.policy, self.cfg, meta = offline_replay.load_policy(str(self.ckpt_path), self.device)
        print(f"控制器 {self.ckpt_path}：{meta}")
        # 以下照上游 init_controller（inference.py:134-145），SAM 权重路径换成本机的
        self.sam = sam_model_registry[self.config["sam"]["model_type"]](checkpoint=str(HERE / self.config["sam"]["checkpoint"]))
        self.sam.to(device=self.device)
        self.predictor = SamPredictor(self.sam)
        self.cutie = get_default_model()
        self.processor = InferenceCore(self.cutie, cfg=self.cutie.cfg)
        self.processor.max_internal_size = -1

    def init_planner(self):
        if not self.args.manual:
            raise SystemExit("规划器（C6）还没接，只支持 --manual")

    # ── 硬件 ─────────────────────────────────────────────────────────

    def init_robot(self):
        lim = self.config["robot"]["dof_limits"]
        self.dof_lower_limits, self.dof_upper_limits = list(lim["lower"]), list(lim["upper"])
        mode = self.args.mode

        if mode in ("real", "mockcam"):  # 臂、夹爪、相机都只能一个程序用
            self._collect_lock = _take_collect_lock()
            if self._collect_lock is None:
                raise SystemExit(f"{COLLECT_LOCK} 被占着：采集（collect）或另一个部署正在用臂 / 相机，先退出它")

        ac = AR5Config()                              # 相机、限速、限幅、地板：和采集同一套默认值
        ac.end_effector, ac.use_hand = "omnipicker", True
        ac.arm.use_realtime = True                    # 和 collect 一样走实时流
        from omnipicker import OmniPickerConfig
        g = self.config["gripper"]
        ac.omnipicker = OmniPickerConfig(grip_force=g["grip_force"], speed=g["speed"])
        if mode == "real":
            holders = _serial_holders(ac.omnipicker.port)
            if holders:
                raise SystemExit(f"夹爪串口 {ac.omnipicker.port} 被占着：{holders}。先退出那个程序（这里不替你关）")
        if self.args.max_joint_speed is not None:     # 首次真机可以压低；每步限幅按 50 Hz 跟着缩
            ac.arm.max_joint_speed = self.args.max_joint_speed
            ac.arm.joint_goal_step = min(ac.arm.joint_goal_step,
                                         self.args.max_joint_speed / self.config["control"]["publish_hz"])
        floor = self.config.get("arm", {}).get("joint_floor_z")
        if floor is not None:                         # 部署专用地板，见 config_ar5.yaml（采集不受影响）
            ac.arm.joint_floor_z = float(floor)
        ac.mock = mode in ("replay", "mockcam")
        ac.real_cameras_in_mock = mode == "mockcam"
        self.goal_stale_timeout = ac.arm.goal_stale_timeout
        self.head_key, self.wrist_key = self.config["cameras"]["head"], self.config["cameras"]["wrist"]

        self.env = AR5Env(ac)
        self.env.connect()
        if mode == "replay":
            self.env.cameras.disconnect()
            self.env.cameras = ReplayCameras(self.args.replay_zarr, self.args.replay_episode,
                                             self.head_key, self.wrist_key)
        print(f"模式 {mode}：臂 {type(self.env.arm).__name__}，夹爪 {type(self.env.hand).__name__}，"
              f"相机 {type(self.env.cameras).__name__}；限速 {ac.arm.max_joint_speed} rad/s，"
              f"每步限幅 {ac.arm.joint_goal_step} rad，地板 {ac.arm.joint_floor_z} m")

        self.right_hand, self.left_hand = GripperAdapter(), None
        self.right_hand_joint = np.array([self.env.hand.get_opening()], dtype=np.float32)
        self._grab_frames()
        # 发送线程马上起来：从这一刻起每 20 ms 一个目标，先原地保持
        self._hold = np.asarray(self.env.arm.get_joint_positions(), dtype=np.float64)
        self.init_low_level_control, self.update_target, self.in_post_run = True, False, True
        self.arm_control_thread = threading.Thread(target=self.low_level_control, daemon=True)
        self.arm_control_thread.start()

    def init_robot_state(self):
        r = self.config["robot"]["arms"]["right"]
        lo = np.array(self.dof_lower_limits[:7])
        hi = np.array(self.dof_upper_limits[:7])
        self.right_init_qpos = np.clip(np.array(r["init_qpos"], dtype=np.float64), lo, hi)
        pick = lambda k: None if r.get(k) is None else np.array(r[k], dtype=np.float64)
        self.right_placement_joint, self.right_return_medium_joint = pick("placement_joint"), pick("return_medium_joint")
        self.move_to_target_joint_angle(self.right_init_qpos)
        self.set_left_hand_open()
        self.set_right_hand_open()
        self.target_right_arm_joint = self.right_init_qpos[None, :]
        self.update_target = True
        self.in_post_run = True

    def init_utils_and_data(self):
        super().init_utils_and_data()
        self.state_buffer = np.zeros((self.cfg.n_obs_steps, self.cfg.task.shape_meta.obs.right_state.shape[0]))
        if self.args.mode == "replay" and self.args.auto:
            # 录像放完就该结束：超时设成录像长度
            self.config["control"]["monitor"]["max_episode_duration"] = self.env.cameras.seconds() + 0.5
        if self.args.max_seconds is not None:
            self.config["control"]["monitor"]["max_episode_duration"] = self.args.max_seconds

    def init_threads(self):
        # 照上游 init_threads，只是臂的发送线程已经在 init_robot 里起来了；自动跑时不听键盘
        self.hand_control_thread = threading.Thread(target=self.low_level_hand_control, daemon=True)
        self.hand_control_thread.start()
        self.camera_thread = threading.Thread(target=self.capture_images, daemon=True)
        self.camera_thread.start()
        if not self.args.auto:
            self.keyboard_listener_thread = threading.Thread(target=self.keyboard_listener, daemon=True)
            self.keyboard_listener_thread.start()
        self.monitor_execution_time_thread = threading.Thread(target=self.monitor_execution_time, daemon=True)
        self.monitor_execution_time_thread.start()

    def _grab_frames(self):
        imgs = self.env.cameras.read()
        # atom_robot 的相机给 RGB；训练数据和上游都是 BGR（collect 写视频时转过）
        self.right_first_color_image = cv2.cvtColor(imgs[self.wrist_key], cv2.COLOR_RGB2BGR)
        self.third_color_image = cv2.cvtColor(imgs[self.head_key], cv2.COLOR_RGB2BGR)

    def capture_images(self):
        next_view = 0.0
        while self.running:
            self._grab_frames()
            if self._viewer is not None and time.perf_counter() >= next_view:
                next_view = time.perf_counter() + 1.0 / 15
                try:
                    self._viewer.write(self.third_color_image, self.right_first_color_image,
                                       self._view_mask, self._view_status())
                except Exception as exc:  # noqa: BLE001  显示坏了就不显示，部署照常
                    print(f"⚠ 实时画面停了：{type(exc).__name__}: {exc}", flush=True)
                    self._viewer = None
            time.sleep(0.005)  # read() 直接给最新一帧、不阻塞（上游的 capture_rgb_frame 会等新帧）

    def _view_status(self):
        it = self.right_hand.issue_times
        now = time.perf_counter()
        recent = sum(1 for t in it[-60:] if now - t < 2.0) / 2.0
        want, cmd, gnow = self.grip_log[-1][1:] if self.grip_log else (0.0, 0.0, 0.0)
        area = int(self._view_mask.sum()) if self._view_mask is not None else 0
        return (0.0 if self.in_post_run else 1.0, len(it), area, gnow, cmd, recent)

    def get_mask(self):
        mask = super().get_mask()
        self._view_mask = mask  # 给实时画面用
        return mask

    def get_state(self):
        right_hand_joint = self.right_hand_joint.copy()
        right_arm_joint = self.unscale(self.env.arm.get_joint_positions(),
                                       self.dof_lower_limits[:7], self.dof_upper_limits[:7])
        return np.concatenate([right_arm_joint, right_hand_joint])

    def low_level_control(self):
        """上游：新目标来了就从当前插值点到它做三次样条（20 个点），每 10 ms 发一个，复位时不发。
        这里：同样的样条、按时间取点，但每 20 ms（50 Hz）经 step_joints 发一个；任何时候都在发，
        没有新东西就发上一个（AR5 0.25 s 收不到就锁停）。"""
        try:
            self._publish_loop()
        except Exception:  # noqa: BLE001  线程悄悄死掉的话，真臂 0.25 s 后锁停却看不出为什么
            import traceback
            traceback.print_exc()
            print("⚠ 发关节目标的线程出错退出了：真臂会在 0.25 s 后锁停。结束这一条", flush=True)
            self.in_post_run = True

    def _publish_loop(self):
        ctl = self.config["control"]
        period, dt_pt = 1.0 / ctl["publish_hz"], ctl["interpolation_period"]
        n = ctl["arm_trajectory"]["interpolation_num"]
        t_plan, last, next_t = None, None, time.perf_counter()
        while self.running:
            now = time.perf_counter()
            with self._arm_lock:
                scripted = self._scripted
                if scripted is None and self.update_target:
                    if self.init_low_level_control or self._traj is None:
                        start = np.asarray(self.env.arm.get_joint_positions(), dtype=np.float64)
                        self.init_low_level_control = False
                    else:
                        start = self._traj[min(int((now - t_plan) / dt_pt), n - 1)]
                    self._traj = cubic_spline_interpolation_7d(np.array([start, self.target_right_arm_joint[0]]),
                                                               step=1.0 / n)
                    t_plan, self.update_target = now, False
                if scripted is not None:
                    traj, t0, dt = scripted
                    q = traj[min(int((now - t0) / dt), len(traj) - 1)]
                elif self.in_post_run or self._traj is None:
                    q = self._hold
                else:
                    q = self._traj[min(int((now - t_plan) / dt_pt), n - 1)]  # 上游也是到第 n-1 个点为止
            self.env.step_joints(q, observe=False)
            self._hold = q
            if last is not None:
                self.max_publish_gap = max(self.max_publish_gap, now - last)
                if now - last > 0.1:
                    print(f"⚠ 发关节目标隔了 {(now - last) * 1000:.0f} ms（超过 {self.goal_stale_timeout * 1000:.0f} ms 真臂锁停）", flush=True)
            last = now
            self.publish_log.append((now, *q))
            if getattr(self.env.arm, "latched", False):
                print("⚠ 臂的实时流已锁停（收不到目标 / 碰撞 / 报错），结束这一条", flush=True)
                self.in_post_run = True
            next_t += period
            time.sleep(max(0.0, next_t - time.perf_counter()))
            if time.perf_counter() - next_t > period:
                next_t = time.perf_counter()

    def move_to_target_joint_angle(self, target_joint_angle, step=None):
        """上游：样条 21 个点、每 10 ms 一个（约 0.2 s，靠他们的臂自己限速），再发终点直到误差 < 阈值。
        这里：按 config 的 move_speed 拉长成整段轨迹交给发送线程（step_joints 的限速还在），**精确走到目标**
        （上游的 np.arange 不含终点，只到 95%），再等误差 < position_error_threshold。"""
        if getattr(self.env.arm, "latched", False):
            print("⚠ 臂的实时流已锁停，动不了，跳过这次移动。输入 q 退出，再重新启动程序", flush=True)
            return
        target = np.asarray(target_joint_angle, dtype=np.float64)
        start = np.asarray(self.env.arm.get_joint_positions(), dtype=np.float64)
        dur = max(0.2, float(np.max(np.abs(target - start))) / self.config["control"]["move_speed"])
        k = max(2, int(dur / 0.01))
        traj = np.vstack([cubic_spline_interpolation_7d(np.array([start, target]), step=1.0 / k), target[None]])
        with self._arm_lock:
            self._scripted = (traj, time.perf_counter(), dur / k)
        thr = self.config["control"]["arm_trajectory"]["position_error_threshold"]
        deadline = time.perf_counter() + dur + 20.0
        while time.perf_counter() < deadline:
            err = np.max(np.abs(np.asarray(self.env.arm.get_joint_positions()) - target))
            if time.perf_counter() > self._scripted[1] + dur and err < thr:
                break
            time.sleep(0.02)
        else:
            print(f"⚠ {dur + 20:.0f} s 内没走到目标（误差 {err:.3f} rad），停在原地")
        with self._arm_lock:
            self._hold = target if err < thr else np.asarray(self.env.arm.get_joint_positions(), dtype=np.float64)
            self._scripted, self._traj, self.init_low_level_control = None, None, True

    # ── 夹爪 ─────────────────────────────────────────────────────────

    def set_left_hand_open(self):
        pass  # 没有左手

    def set_right_hand_open(self):
        self.target_right_hand_joint = np.array([1.0])

    def _gripper_command(self, want, now):
        """README「部署：夹爪那一维要换算」：OmniPicker 目标正好落在物体表面会卸力，往合拢走时多给余量。"""
        g = self.config["gripper"]
        if want < now - g["close_threshold"]:
            cmd = max(0.0, want - g["close_margin"])            # 往合拢走：比模型说的再合一点
        elif self._last_grip_cmd is not None and self._last_grip_cmd < now and want <= now + g["hold_band"]:
            cmd = self._last_grip_cmd                            # 正夹着：输出比实测略大也不松
        else:
            cmd = min(1.0, want)
        self._last_grip_cmd = cmd
        return cmd

    def low_level_hand_control(self):
        try:
            while self.running:  # 上游 20 Hz
                now = float(self.env.hand.get_opening())
                want = float(self.target_right_hand_joint[0])
                cmd = self._gripper_command(want, now)
                self.env.hand.set_opening(cmd)
                self.right_hand_joint = np.array([now], dtype=np.float32)
                self.grip_log.append((time.perf_counter(), want, cmd, now))
                time.sleep(0.05)
        except Exception:  # noqa: BLE001  夹爪报错（堵死、回绕、断线）时驱动会停发，这里结束这一条
            import traceback
            traceback.print_exc()
            print("⚠ 夹爪线程出错退出了（看上面的原因，可能要断电重启夹爪）。结束这一条", flush=True)
            self.in_post_run = True

    # ── 一条一条跑 ───────────────────────────────────────────────────

    def mark_bbox_manual(self):
        if self.args.bbox is None and not (self.args.mode == "replay" and self.args.auto):
            return super().mark_bbox_manual()  # 上游：头部画面上点两下
        bbox = (np.array([float(v) for v in self.args.bbox.split(",")], dtype=np.float32)
                if self.args.bbox else self.env.cameras.bbox)
        self.save_image(self.third_color_image, "head_image_start")
        self.show_and_save_image_with_bbox(self.third_color_image[..., ::-1], bbox,
                                           f"{time.strftime('%Y%m%d_%H%M%S')}_head_image_with_bbox.png")
        return bbox

    def right_finish_reset(self):
        if self.right_placement_joint is not None:     # 上游：放置位松手、经中间位回起始位
            self.move_to_target_joint_angle(self.right_placement_joint)
            self.set_right_hand_open()
            if self.right_return_medium_joint is not None:
                self.move_to_target_joint_angle(self.right_return_medium_joint)
        self.move_to_target_joint_angle(self.right_init_qpos)

    def _open_gripper_for_start(self):
        """采集时每条都从夹爪全开开始（数据第 0 帧开合 0.995~1.0）。"""
        self.set_right_hand_open()
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < 4.0 and self.env.hand.get_opening() < 0.95:
            time.sleep(0.05)

    def execute_action(self, action):
        """上游每次预测只执行前 6 步、步与步之间不等（README「已知问题」第 4、7 条）。
        --exec-steps / --step-hz 都不给时直接用上游的，行为不变。给了的话循环体照抄上游，只改三处：
        ① 执行前 n 步；② 每步按 step_hz 等够间隔（上游同一段内实测 16 ms 一步，是 30 Hz 数据的两倍速）；
        ③ 每步前看一眼 in_post_run，执行 32 步时结束不会被拖 2 秒。
        10-26 black_box epoch 30/40 真机：每次预测的下降都在后半段、前 6 步几乎不动，只执行 6 步就重来，臂停在半路。"""
        n, hz = self.args.exec_steps, self.args.step_hz
        if n == 6 and not hz:
            return super().execute_action(action)
        t0 = time.perf_counter()
        for k in range(min(n, len(action))):
            if self.in_post_run:
                break
            if hz:
                wait = t0 + k / hz - time.perf_counter()
                if wait > 0:
                    time.sleep(wait)
            state = self.get_state()
            mask = self.get_mask()
            self.get_obs(state, mask)

            self.target_right_arm_joint = self.scale(action[k:k+1, 0:7], self.dof_lower_limits[:7], self.dof_upper_limits[:7])
            self.update_target = True
            self.target_right_hand_joint = self.right_hand.execute_action(action[k, 7:])

            if self.args.save_deployment_data:
                self.record_frame(
                    state=state,
                    action=action[k],
                    right_cam_img=self.right_first_color_image.copy(),
                    rgbm=self.third_color_image_with_mask
                )

    def run_manual(self):
        """照上游 run_manual；--auto 时不等回车、按 --episodes 跑几条。"""
        done = 0
        while True:
            if self.args.auto:
                if done >= self.args.episodes:
                    break
            else:
                try:
                    if not inference.get_start_command():
                        print("Bye.")
                        break
                except (KeyboardInterrupt, EOFError):  # 提示符按 Ctrl+C = 退出
                    print("\n（Ctrl+C）退出")
                    break
            done += 1
            self.time_step = 0
            self.init_episode(manual=True)
            self.log(f"控制器 {self.ckpt_path}；每次预测执行 {self.args.exec_steps} 步，"
                     f"{'每步间隔 %.0f ms' % (1000 / self.args.step_hz) if self.args.step_hz else '步间不等（同上游）'}",
                     message_type="info")
            self._open_gripper_for_start()
            if self._need_gui:
                plt.switch_backend("TkAgg")  # 上游框选用 plt.ginput 弹窗点两下
            bbox = self.mark_bbox_manual()
            self.initialize_sam_cutie(bbox)
            self._release_gui()
            gc.disable()  # 执行期间回收只在主线程做，见 _release_gui
            self.reset_flags()
            if not self.args.auto:
                inference.clear_input_buffer()
            self.right_hand.issue_times.clear()
            self.max_publish_gap, n_pub0 = 0.0, len(self.publish_log)
            if isinstance(self.env.cameras, ReplayCameras):
                self.env.cameras.start()
            t_start = time.perf_counter()
            self.log("Controller starts executing the current instruction.", message_type="info")
            self.execute_grasping()
            t_end = time.perf_counter()
            gc.enable()
            gc.collect()
            self.right_finish_reset()
            self.save_image(self.third_color_image, "head_image_episode_finish")
            self.log("Head camera image saved at the end of the episode.", message_type="info")
            self._report(t_start, t_end, n_pub0)
            self.close_episode()

    def _release_gui(self):
        """10-22 真机锁停的原因：框选、显示掩码开过 TkAgg 窗口，Tk 对象要等垃圾回收才释放；回收碰巧在
        发送线程里触发时，每个 Tk 对象等主线程约 1 s 再报「main thread is not in main loop」，几十个
        就是几十秒 —— 发送线程卡 40 s，臂 0.25 s 后锁停。所以用完窗口马上在主线程关图、回收、切回 Agg
        （本机复现：别的线程里回收卡 60 s；这样处理后 4 ms）。执行期间再把自动回收关掉，结束后主线程回收。"""
        plt.close("all")
        gc.collect()
        if plt.get_backend().lower() != "agg":
            plt.switch_backend("Agg")
            gc.collect()

    def _report(self, t_start, t_end, n_pub0):
        it = np.array(self.right_hand.issue_times)
        pubs = np.array(self.publish_log[n_pub0:])
        rate = (len(it) - 1) / (it[-1] - it[0]) if len(it) > 1 else 0.0
        msg = (f"这一条：执行 {t_end - t_start:.1f} s，下发 {len(it)} 步动作 = {rate:.1f} 步/秒；"
               f"发关节目标 {len(pubs)} 次，最长间隔 {self.max_publish_gap * 1000:.0f} ms"
               f"（超过 {self.goal_stale_timeout * 1000:.0f} ms 真臂会锁停）；atom_robot 统计 {self.env.stats}")
        print(msg, flush=True)
        self.log(msg, message_type="info")
        np.savez_compressed(os.path.join(self.current_episode_dir, "deploy_trace.npz"),
                            action_issue_t=it, publish=pubs, gripper=np.array(self.grip_log),
                            t_start=t_start, t_end=t_end)

    def close(self):
        """照 AR5Env.close 的顺序（stop → 相机 → 夹爪 → 臂），但每一步单独计时、超时就跳过。
        返回是否全部按时做完；没做完的话 main 会强制退出（正常退出要等卡住的线程）。"""
        self.running = False
        for name in ("arm_control_thread", "hand_control_thread", "camera_thread",
                     "keyboard_listener_thread", "monitor_execution_time_thread"):
            th = getattr(self, name, None)
            if th is not None:
                th.join(timeout=2)
        clean = True
        if getattr(self, "_viewer", None) is not None:
            clean &= _timed("关实时画面", self._viewer.close, 5)
        # 不自动松开夹爪：夹着东西时松开会把它摔了（和 OmniPicker 驱动的约定一样）
        env = getattr(self, "env", None)
        if env is not None:
            clean &= _timed("停臂的实时流（真臂上会掉电，和采集退出时一样）", env.stop, 15)
            for label, dev in (("断开相机", env.cameras), ("断开夹爪", env.hand), ("断开臂", env.arm)):
                if dev is not None:
                    clean &= _timed(label, dev.disconnect, 10)
            env._connected = False
        print("System shutdown complete." if clean else "⚠ 收尾有一步没做完（见上），强制退出", flush=True)
        return clean


def main():
    ap = argparse.ArgumentParser(description="DexGraspVLA 部署：AR5 + OmniPicker + 两台 Orbbec")
    ap.add_argument("--manual", action="store_true", help="手动框选目标（目前只支持这个）")
    ap.add_argument("--gen_attn_map", action="store_true", help="上游参数：存 attention map")
    ap.add_argument("--save_deployment_data", action="store_true", help="上游参数：存部署数据")
    ap.add_argument("--mode", choices=("replay", "mockcam", "real"), required=True)
    ap.add_argument("--auto", action="store_true",
                    help="不等回车、不听键盘。replay / mockcam 自测用；真机要再加 --bbox、--max-seconds、--operator-at-estop")
    ap.add_argument("--operator-at-estop", action="store_true", help="真机 --auto 时声明：有人守在急停旁")
    ap.add_argument("--episodes", type=int, default=1, help="--auto 时跑几条")
    ap.add_argument("--bbox", default=None, help="x1,y1,x2,y2：不弹窗，直接用这个框（头部相机 640×480 像素）")
    ap.add_argument("--replay-zarr", default=None)
    ap.add_argument("--replay-episode", type=int, default=0)
    ap.add_argument("--ckpt", default=None, help="控制器检查点，默认用 inference_utils/config_ar5.yaml 里的")
    ap.add_argument("--no-view", action="store_true", help="不开实时画面窗口（默认开：俯视 + 掩码、腕部）")
    ap.add_argument("--view-snapshot", default=None, help="实时画面存图：给文件只留最新一张，给目录每秒存一张带编号的")
    ap.add_argument("--max-seconds", type=float, default=None,
                    help="每条最多跑几秒就结束（上游 max_episode_duration，默认 100）")
    ap.add_argument("--max-joint-speed", type=float, default=None,
                    help="臂限速 rad/s，默认用 atom_robot 的（和采集一样 0.74）；首次真机可以压低")
    ap.add_argument("--exec-steps", type=int, default=6,
                    help="每次预测后执行前几步再重新预测（上游写死 6，最多 64）")
    ap.add_argument("--step-hz", type=float, default=0.0,
                    help="执行时每步至少隔 1/F 秒（上游不等，同一段内约 16 ms 一步；30 = 和采集数据同节奏）。0 = 同上游")
    args = ap.parse_args()
    if not 1 <= args.exec_steps <= 64:
        ap.error("--exec-steps 要在 1~64 之间（一次预测 64 步）")
    if args.step_hz < 0:
        ap.error("--step-hz 不能是负数")
    if args.mode == "replay" and not args.replay_zarr:
        ap.error("--mode replay 要给 --replay-zarr")
    if args.mode == "real" and args.auto and not (args.bbox and args.max_seconds and args.operator_at_estop):
        # 真机自动跑（用户 10-22 要 Claude 代跑）：没有回车可按，只能靠超时或急停停下，所以三样都要明确给
        ap.error("真机 --auto 必须同时给 --bbox、--max-seconds 和 --operator-at-estop（有人守在急停旁）")

    # 先建空对象再初始化：初始化到一半出错（臂已经连上了）也能走 close() 收尾
    system = AR5System.__new__(AR5System)
    clean = True
    try:
        system.__init__(args)
        system.run()
    finally:
        clean = system.close()
    if not clean:
        os._exit(1)  # 有收尾线程卡着：正常退出会一直等它，所以直接退


if __name__ == "__main__":
    main()
