#!/usr/bin/env python3
"""用训好的策略驱动真臂。**ACT 和 π0.5 共用这一个脚本。**

**在 lerobot 环境里跑，不是 vel-ar5。**

    L=~/miniforge3/envs/lerobot/bin/python
    CK=~/vel_runs/act_bolt_0907_1840/checkpoints/100000

    $L atom_robot/ar5/eval_policy.py --ckpt $CK --dry-run --autostart
    $L atom_robot/ar5/eval_policy.py --ckpt $CK --scale 0.5
    $L atom_robot/ar5/eval_policy.py --ckpt ~/pi05_bolt/checkpoints/030000

策略类型从 checkpoint 的 `config.json` 自动读，不用告诉它是 act 还是 pi05。

**先跑 --dry-run。** 相机和策略是真的、臂是模拟的，把每步想走多远打出来。
数字和训练数据一个量级（中位 1.3 mm 左右）再接臂。

**每一轮开跑前会先把臂和手开到任务起点**（100 条首帧的中位 state：关节
[-46.5 29.8 6.9 119.6 -16.3 -8.6 1.0] deg、夹爪 0.61）。采集是遥操作，起点是操作者
握主臂的预备位；从别处起跑，策略第一帧就吃到训练分布外的输入。`config.home_joints`
和这个起点差 46.8°，不要拿它当起点。`--no-goto-start` 可以关掉。

按键：
    空格   开始 / 暂停。**再按开跑时臂先回起点**，因为这里 step 归零、
           policy.reset()，等于新的一轮，不是接着上一轮
    q      **结束本轮，回起点，留在程序里**等下一轮。评测要连着跑十几轮记
           成功率，不该每轮都重启一次、重新加载几 GB 权重
    g      **抓取成功**（螺栓夹住了）。**发生的当下就按**，不等本轮结束
    i      **插入成功**（螺栓进套筒了）
    r      **本轮结束，全程成功**（写一行 CSV）
    f      **本轮结束，失败**
    x      退出程序（退出前也会回起点）。Ctrl-C 同样是干净的退出路径

**每一段的成功率都要能单独算出来。** 能量头是按段训的，只记整体成功率的话，
抓取段的改善会被后面的失败稀释掉，判断不了是哪个头起了作用。后面的里程碑蕴含
前面的（按 `i` 会补上 `g`，按 `r` 两个都补），漏按不当成失败。没抓到那轮的
「插入」在 CSV 里**留空而不是「失败」**—— 它根本没发生。
逐轮写进 `--log`（默认 `~/vel_runs/rollouts.csv`），用 `rollout_stats.py` 统计。

策略输出的是**位姿增量**（3 位置 + 3 轴角 + 1 手开合），和采集时记的是同一套，
`AR5Env.step()` 正好吃这个格式，限幅/工作空间/牵引绳/可达性检查都在它里面。

π0.5 特有的两件事：
  * 权重 9.4 GB，**训练要 A6000，但推理在本机 16 GB 上跑得动**。脚本加载后会报
    显存占用。
  * 它吃语言指令。任务描述默认从数据集元信息读，和训练时一致。
"""

from __future__ import annotations

import argparse
import json
import select
import sys
import termios
import time
import tty
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from atom_robot import SDK_ROOT  # noqa: E402
sys.path.insert(0, str(SDK_ROOT / "Release" / "linux"))

from atom_robot.ar5.ar5_env import AR5Env  # noqa: E402
from atom_robot.ar5.config import AR5Config  # noqa: E402


class RawKeyboard:
    """和 collect.py 同一套：直接读裸 fd，不走 sys.stdin 的缓冲。"""

    def __enter__(self):
        import os

        self.os = os
        self._pending = ""
        # stdin 不是终端就退化成"没有按键"。dry-run 常常被重定向或者放后台跑，
        # 那时候 tcgetattr 会抛 `Inappropriate ioctl for device`，
        # 为了个可选的键盘把整个评测搞崩不值得。
        self.tty = sys.stdin.isatty()
        if not self.tty:
            self.fd = self.saved = None
            print("  （stdin 不是终端，按键关闭；用 --autostart 无人值守跑）")
            return self
        self.fd = sys.stdin.fileno()
        self.saved = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, *exc):
        if self.tty and self.saved is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.saved)

    def key(self):
        if not self.tty:
            return None
        if not self._pending:
            if not select.select([self.fd], [], [], 0)[0]:
                return None
            try:
                data = self.os.read(self.fd, 32)
            except OSError:
                return None
            if not data:
                return None
            self._pending = data.decode(errors="replace")
        char, self._pending = self._pending[0], self._pending[1:]
        return char


def silence_shutdown_noise() -> None:
    """闭掉退出时那条吓人的 AttributeError。**它和臂没有任何关系。**

        Exception ignored in: <function ResourceTracker.__del__ ...>
        AttributeError: '_thread.RLock' object has no attribute '_recursion_count'

    是 `multiprocess`（dill 那个 multiprocessing 分支，lerobot 经 datasets 拉进来的）
    自己的 bug：`_stop_locked` 假设锁是 `_multiprocessing.SemLock`，而这里拿到的是
    普通 `_thread.RLock`，没有 `_recursion_count`。触发点在**解释器终结阶段**，
    比 `env.close()` 晚，所以臂早就干净断开了。

    但它长得像崩溃，会让人以为按 q 之后出了事 —— 实际那是另一回事（臂断电，
    见 `ArmConfig.power_off_on_disconnect`）。两个凑一起很容易看混，所以闭掉。
    """
    try:
        from multiprocess import resource_tracker as _rt
    except Exception:  # noqa: BLE001
        return
    tracker = getattr(_rt, "_resource_tracker", None)
    if tracker is None:
        return
    original = tracker._stop_locked

    def quiet(*args, **kwargs):
        try:
            return original(*args, **kwargs)
        except AttributeError as exc:
            if "_recursion_count" not in str(exc):
                raise      # 别的 AttributeError 是真问题，照抛
    try:
        tracker._stop_locked = quiet
    except Exception:  # noqa: BLE001
        pass


def round_stats(step: int, deltas) -> dict:
    """一轮的步数和逐帧位移摘要。空轮返回空字典。"""
    if not deltas:
        return {}
    return {"steps": step, "median": float(np.median(deltas)), "max": float(max(deltas))}


CSV_HEADER = ["时间", "轮次", "结果", "抓取", "插入", "步数", "位移中位mm", "位移最大mm",
              "tag", "checkpoint", "scale", "vel", "抓取修正块", "放置修正块", "脱手次数"]
CSV_RENAMED = {"运送": "插入"}      # 改过名的老列，迁移时按这个映射

# **里程碑式打标：发生了就按，不是最后猜一个结局。**
#
# 能量头是按段训的，所以每一段的成功率都要能单独算出来 —— 只记整体成功率的话，
# 抓取段的改善会被后面的失败稀释掉，判断不了是哪个头起了作用。
MILESTONES = {
    "g": ("grasp", "抓取成功"),      # 螺栓夹住了
    "i": ("insert", "插入成功"),     # 螺栓进套筒了
}
ENDINGS = {
    "r": (True, "全程成功"),         # 结束本轮，写一行 CSV
    "f": (False, "失败"),
}
# 后面的里程碑蕴含前面的：插入了就一定抓到过，全程成功就一定两个都过。
# 漏按不当成失败 —— 顺序是物理决定的，能补就补，并在屏幕上说一声。
IMPLIES = {"i": ("g",), "r": ("g", "i")}


def migrate_csv(path: Path) -> bool:
    """老 CSV 缺列就补上再追加，否则表头和数据错位。

    列是逐步加出来的、还改过名（「运送」-> 「插入」），所以按**列名**映射，
    不按位置。老行缺的列留空 —— 那时候没记，不能猜。

    **改过名的老列里如果真有数据，宁可报错也不改写。** 语义变了，
    把「运送成功」当成「插入成功」是在编数据。

    就地重写：先写临时文件再原子改名，中途断电也不会得到半个文件。
    """
    import csv

    if not path.exists():
        return False
    with path.open(newline="") as fh:
        rows = list(csv.reader(fh))
    if not rows or rows[0] == CSV_HEADER:
        return False
    old = rows[0]
    unknown = [c for c in old if c not in CSV_HEADER and c not in CSV_RENAMED]
    if unknown:
        raise RuntimeError(f"{path} 有本脚本不认识的列 {unknown}，不敢就地改写。"
                           f"改个 --log 路径重开一份")
    for src, dst in CSV_RENAMED.items():
        if src not in old:
            continue
        col = old.index(src)
        if any(len(r) > col and r[col] for r in rows[1:]):
            raise RuntimeError(
                f"{path} 的「{src}」列有数据，而它已经改名成「{dst}」、语义也变了。"
                f"两者不能混在一列里 —— 把老文件改名存档，用新的 --log 重开一份")
        old = [dst if c == src else c for c in old]
    tmp = path.with_suffix(".csv.tmp")
    with tmp.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(CSV_HEADER)
        for r in rows[1:]:
            d = dict(zip(old, r))
            w.writerow([d.get(c, "") for c in CSV_HEADER])
    tmp.replace(path)
    return True


def log_rollout(path: Path, stats, ok: bool, marks: dict, last_round, args, ev,
                vel_stats=None) -> None:
    """按 r / f 结束本轮时追加一行 CSV。**内存里的计数器是留不住的。**

    实测教训：跑完 50 轮想按「前 30 随机摆 / 后 20 固定」分组统计，结果只有
    退出时打的一行「X/Y 成功」—— 没有逐轮记录、没有时间戳，分不开。
    终端一关就彻底没了。

    `--tag` 用来标这一批的条件（random / fixed / vel-on ...），中途换条件
    直接重开一次 eval 换个 tag，或者事后按时间戳切。

    **每一段分开记。** 能量头是**按段训**的（抓取段一个、放置段一个），
    只记整体成功率的话，抓取段的改善会被后面的失败稀释掉，判断不了哪个头
    起了作用。里程碑在过程中按（见 `MILESTONES`），`r`/`f` 结束本轮时落盘。

    **没抓到那轮的「插入」留空，不是「失败」。** 那一段根本没发生过；
    算进分母会让抓取段的改善伪装成插入段的改善 —— 它只是多送了一批
    本来走不到插入的轮次进去。

    **步数/位移取的是「正在评的那一轮」**，不是某个残留变量。第一版把它们存在
    一个只在 q / 到 max_steps 时才更新的 last_round 里，于是按 r/f 的时机不同，
    整列要么空、要么整体错位一轮 —— 实测那 60 轮的第 1 行就是空的。
    """
    import csv
    import datetime as _dt

    grasp = "成功" if marks.get("grasp") else "失败"
    insert = ("成功" if marks.get("insert") else "失败") if marks.get("grasp") else ""
    new = not path.exists()
    with path.open("a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(CSV_HEADER)
        w.writerow([_dt.datetime.now().isoformat(timespec="seconds"),
                    stats["success"] + stats["fail"],
                    "成功" if ok else "失败", grasp, insert,
                    last_round.get("steps", ""),
                    f"{last_round['median']:.2f}" if last_round.get("median") else "",
                    f"{last_round['max']:.2f}" if last_round.get("max") else "",
                    args.tag, Path(args.ckpt).name, ev.scale, vel_column(args.vel),
                    *(vel_stats or ("", "", ""))])


def write_energy_trace(path: Path, trace, args, rollout: int, ok: bool) -> None:
    """逐块能量追加到 CSV。**每个加载的头都记，修正前后各一列。**

    论文 §6.4：修正主要发生在抓取/放置的关键接触段，逐时刻能量曲线上看得见。
    画曲线要的就是这个。基线用 `--vel-off` 跑，那时 E_*_out == E_*_bc。
    """
    import csv
    import datetime as _dt

    keys = [k for k in trace[0] if k.startswith("E_")]
    header = ["时间", "tag", "轮次", "结果", "步", "段", "生效的头", *keys]
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    now = _dt.datetime.now().isoformat(timespec="seconds")
    with path.open("a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(header)
        for r in trace:
            w.writerow([now, args.tag, rollout, "成功" if ok else "失败",
                        r["step"], r["segment"], r["applied"],
                        *[f"{r[k]:.4f}" for k in keys]])


def vel_column(specs) -> str:
    """CSV 里那一列要能区分「哪几段挂了头」，不能只记文件名。

    `grasp=a.pt` 和 `a.pt`（全程用 a.pt）是两个不同的实验条件，落到同一格
    就白测了 —— `rollout_stats.py` 正是按这一列分组的。
    """
    if not specs:
        return ""
    out = []
    for spec in specs:
        seg, _, raw = spec.rpartition("=")
        out.append(f"{seg or 'all'}={Path(raw).name}")
    return ";".join(sorted(out))


def dataset_extent(root: Path):
    """训练数据里末端到过哪些地方。用来判断工作空间盒够不够大。"""
    import glob
    import os

    files = sorted(glob.glob(os.path.expanduser("~/atom_episodes/episode_*.npz")))
    if not files:
        return None
    poses = np.concatenate([np.load(f, allow_pickle=True)["ee_pose"][:, :3] for f in files])
    return poses.min(axis=0), poses.max(axis=0), len(poses)


def task_start_state(src=None):
    """任务起点 = 100 条 episode 首帧的**中位** state，8 维（7 关节 + 夹爪）。

    **开跑前和跑完都要用它。** 采集是遥操作，起点是操作者握主臂的预备位；
    部署如果从别处起步，策略第一帧就看到训练分布外的输入。夹爪同理 —— 它是
    state 的第 8 维，策略直接看得到。

    不能用 `config.arm.home_joints`：那是给遥操作定的工作空间盒中心，
    和任务起点差 **46.8°**（j0 0.4 vs -46.5、j4 -47.7 vs -16.3，L2 60.6°）。

    取中位而不是某一条首帧：起点本身有自然离散
    （关节 std [4.8 3.6 0 1.4 3.2 3.7 0.1] deg，夹爪 std 0.100），中位是这一族的代表。
    """
    import glob
    import os

    files = sorted(glob.glob(os.path.expanduser(str(src or "~/atom_episodes") + "/episode_*.npz")))
    if not files:
        return None
    starts = np.array([np.load(f, allow_pickle=True)["state"][0][:8] for f in files])
    return np.median(starts, axis=0)


def task_start_pose(src=None):
    """只要关节那 7 维，给 `goto_joints` 用。"""
    s = task_start_state(src)
    return None if s is None else s[:7]


def goto_task_start(env, ev, label: str) -> bool:
    """把臂和手都开到任务起点。开跑前和跑完都走这一条路径。"""
    s = task_start_state()
    if s is None:
        print(f"  {label}：找不到示教数据，不知道该去哪，臂留在原地。")
        return False
    joints, grip = s[:7], float(s[7])
    now = env.arm.get_joint_positions()[:7]
    off = np.rad2deg(np.abs(now - joints)).max()
    print(f"  {label} {np.rad2deg(joints).round(1).tolist()} 夹爪 {grip:.2f}"
          f"（当前最大差 {off:.1f}°）...")
    try:
        env.goto_joints(joints, timeout_s=40.0)
        err = np.rad2deg(np.abs(env.arm.get_joint_positions()[:7] - joints)).max()
        if env.hand is not None:
            env.hand.set_opening(grip)
        print(f"  到位，偏差 {err:.3f}°")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"  {label} 失败：{exc}")
        return False


def check_workspace(config, widen: bool) -> bool:
    """示教轨迹在不在工作空间盒里。不在就拦住，别让人白测一场。

    这条真的会咬人：`AR5Env.step()` 走笛卡尔路径，会把目标夹进
    `workspace_min/max`。而这批数据是用**关节路径**采的，那条路径不看盒子 ——
    实测 100 条里 46.8% 的帧 y 在盒外。不改盒子就等于近一半时间臂拒绝跟随，
    看起来像"模型不行"，其实是配置不匹配。
    """
    extent = dataset_extent(None)
    if extent is None:
        print("  找不到 ~/atom_episodes，跳过工作空间检查")
        return True
    lo_d, hi_d, n = extent
    lo_b = np.array(config.arm.workspace_min, dtype=np.float64)
    hi_b = np.array(config.arm.workspace_max, dtype=np.float64)

    print(f"\n  工作空间检查（对着 {n} 帧示教数据）")
    print(f"    {'轴':>3}{'数据范围':>22}{'盒子':>22}")
    bad = []
    for i, ax in enumerate("xyz"):
        mark = ""
        if lo_d[i] < lo_b[i] or hi_d[i] > hi_b[i]:
            bad.append(ax)
            mark = "   ⚠ 数据在盒外"
        print(f"    {ax:>3}  [{lo_d[i]:+.3f}, {hi_d[i]:+.3f}]     "
              f"[{lo_b[i]:+.3f}, {hi_b[i]:+.3f}]{mark}")

    if not bad:
        print("    ✓ 示教轨迹完全在盒内")
        return True

    margin = 0.05
    # 只放宽**真的被越界**的那一侧。无脑给每个轴都减余量会把没问题的边界
    # 也一起放宽 —— 实测会把 z 下界从 0.08 降到 0.045，等于白白拆掉 35 mm
    # 的地板保护，而 z 方向的数据本来就全在盒内。放宽安全边界只能是有依据的。
    new_lo = np.where(lo_d < lo_b, np.minimum(lo_b, lo_d - margin), lo_b).round(3)
    new_hi = np.where(hi_d > hi_b, np.maximum(hi_b, hi_d + margin), hi_b).round(3)
    print(f"\n  ⚠ {','.join(bad)} 轴的示教数据超出工作空间盒。")
    print("    盒子是安全边界，`step()` 会把目标夹进去 —— 不改的话策略要去的地方")
    print("    臂到不了，评测结果没有意义。")
    print(f"\n    建议改成（数据范围各留 {margin*1000:.0f} mm 余量）：")
    print(f"      workspace_min = {tuple(new_lo)}")
    print(f"      workspace_max = {tuple(new_hi)}")
    print("\n    两个选择：")
    print("      1) 手改 config.py 的 ArmConfig（持久，推荐 —— 这是安全参数，")
    print("         应该有意识地改，而不是被脚本偷偷放宽）")
    print("      2) 加 --widen-workspace，只对这一次生效")

    if widen:
        config.arm.workspace_min = tuple(float(v) for v in new_lo)
        config.arm.workspace_max = tuple(float(v) for v in new_hi)
        print(f"\n  --widen-workspace：本次已放宽到 {config.arm.workspace_min} ~ "
              f"{config.arm.workspace_max}")
        return True
    print("\n  没有 --widen-workspace，就到这里。")
    return False


def load_policy(ckpt: Path, device: str):
    """加载策略 **以及它的归一化管线**。act / pi05 / 其它都走这条路。

    策略类型从 checkpoint 的 `config.json` 里的 `type` 字段读，不用手填 ——
    填错了会以一种很难看出来的方式失败（权重形状对不上或者静默乱跑）。

    两个管线不是可选的。checkpoint 里除了 model.safetensors 还有
    `policy_preprocessor_*`（把观测归一化）和 `policy_postprocessor_*`
    （把动作反归一化），只调 `select_action` 会把它们整个绕过去 ——
    模型吃到未归一化的输入、吐出未反归一化的输出。

    实测后果：拿训练集原样喂回去，预测位移 412 mm/步，真值 1.31 mm，差 300 倍。
    而且它**不会报任何错**，只是数值荒谬。
    """
    import json as _json

    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    model_dir = ckpt / "pretrained_model" if (ckpt / "pretrained_model").is_dir() else ckpt
    cfg_file = model_dir / "config.json"
    if not cfg_file.exists():
        raise SystemExit(f"{model_dir} 里没有 config.json，这不像一个 lerobot checkpoint")
    kind = _json.loads(cfg_file.read_text()).get("type")
    if not kind:
        raise SystemExit(f"{cfg_file} 里没有 type 字段，认不出是什么策略")

    policy = get_policy_class(kind).from_pretrained(model_dir)
    policy.to(device)
    policy.eval()

    pre, post = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(model_dir),
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    return policy, pre, post, model_dir, kind


def build_batch(obs, camera_keys, device, task: str):
    """把 AR5Observation 拼成策略要的 batch。

    键名和通道顺序必须和训练时一模一样，不然模型看到的是另一个世界：
    lerobot 存的是 RGB、CHW、float32 [0,1]，而相机驱动给的是 RGB uint8 HWC。
    """
    import torch

    batch = {}
    for key in camera_keys:
        img = np.asarray(obs.observation["images"][key])
        t = torch.from_numpy(img).to(device).permute(2, 0, 1).float() / 255.0
        batch[f"observation.images.{key}"] = t.unsqueeze(0)
    state = np.asarray(obs.observation["qpos"], dtype=np.float32)
    batch["observation.state"] = torch.from_numpy(state).to(device).unsqueeze(0)
    batch["task"] = [task]
    return batch


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--ckpt", required=True, help="checkpoint 目录，比如 .../checkpoints/020000")
    parser.add_argument("--task", default=None,
                        help="任务描述。默认从 checkpoint 旁边的数据集元信息里读")
    # 默认值全部来自 config.py 的 EvalConfig，命令行只是临时覆盖。
    # 想让某个改动下次开机还在，改 config.py，不要靠记住命令行。
    _ev = AR5Config().eval
    parser.add_argument("--hz", type=float, default=None,
                        help=f"控制频率，要和采集时一致（默认 {_ev.hz:.0f}）")
    parser.add_argument("--max-steps", type=int, default=None,
                        help=f"一轮最多跑多少步，到了自动停"
                             f"（默认 {_ev.max_steps} = {_ev.max_steps/_ev.hz:.0f} 秒）")
    parser.add_argument("--dry-run", action="store_true",
                        help="不连臂。连相机+策略，把每步的动作打出来看数值合不合理")
    parser.add_argument("--widen-workspace", action="store_true",
                        help="本次把工作空间盒放宽到覆盖示教数据。看清楚 §工作空间检查 再用")
    parser.add_argument("--no-hand", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--stay", action="store_true",
                        help="退出时不动臂，留在原地。默认回**任务起始位姿**，"
                             "不是 config 的 home（那个差 46.8 度）")
    parser.add_argument("--no-goto-start", action="store_true",
                        help="开跑前不把臂和手开到任务起点。默认会开 —— "
                             "从别处起跑，策略第一帧就是训练分布外的输入")
    parser.add_argument("--temporal-ensemble", type=float, default=None, metavar="COEFF",
                        help="只对 ACT。开时序集成：每一步都重新预测整块，"
                             "再把多个重叠 chunk 对同一时刻的预测加权平均。"
                             "ACT 原论文用 0.01。默认关（=一次预测 100 步全开环执行）")
    parser.add_argument("--tag", default="", metavar="NAME",
                        help="标这一批的条件（random / fixed / vel-on …），写进 CSV。"
                             "中途换条件就重开一次 eval 换个 tag")
    parser.add_argument("--log", default="~/vel_runs/rollouts.csv",
                        help="逐轮结果追加到这个 CSV。**内存里的计数器留不住**")
    parser.add_argument("--vel", action="append", default=None, metavar="[段=]HEAD.pt",
                        help="挂 VEL 能量头做一步残差修正。只支持 pi05。不给就是基线\n"
                             "（baseline / VEL 的 A/B 只差这一个开关）。\n"
                             "**能量头是分段训的，按段挂**，可以给多次：\n"
                             "  --vel grasp=head_grasp.pt --vel carry=head_carry.pt\n"
                             "没给头的段原样放行、等同基线 —— 所以「只训完抓取段」\n"
                             "就能干净地上机，测出的差异只归因于抓取段。\n"
                             "不写段名 = 整条轨迹都用它（第一版的行为，"
                             "会把抓取段的头用到运送段，属于分布外）")
    parser.add_argument("--vel-alpha", action="append", default=None,
                        metavar="[段=]α",
                        help="修正步长。可以给多次按段分别设：--vel-alpha grasp=0.0073\n"
                             "**最优 α 是每个头各自的量**，α*=cos·d₀/√50，用\n"
                             "vel/geo_check.py 量出来。实测 040000：抓取 0.0073、\n"
                             "放置 0.0036，差一倍。不给就用 geo_check 之前量的默认 0.01")
    parser.add_argument("--vel-off", action="store_true",
                        help="**加载能量头、照常记逐块能量，但不做修正。** 画基线能量曲线用。\n"
                             "不能拿 --vel-alpha 0 代替：correct() 末尾的 clip 仍会动值，\n"
                             "而真正的基线路径不 clip。")
    parser.add_argument("--energy-log", default=str(Path.home()/"vel_runs"/"energy_trace.csv"),
                        help="逐块能量写到这里，每块一行")
    parser.add_argument("--ab-interleave", action="store_true",
                        help="**同一场次里随机交替开/关 VEL 的盲测。** 日间差异、相机漂移、\n"
                             "现场摆放、操作习惯全部被消掉 —— 那是隔天对比消不掉的。\n"
                             "两臂走**同一条前向路径**，唯一差别就是做不做那一步修正。\n"
                             "**屏幕上不显示这一轮是哪一臂**（操作者知道了会影响摆放和\n"
                             "按键时机），只写进 CSV 的 tag 后缀 -on / -off。")
    parser.add_argument("--ab-seed", type=int, default=0, help="交替序列的随机种子，可复现")
    parser.add_argument("--vel-gamma", type=float, default=1.0,
                        help="梯度范数裁剪，防能量面陡峭处过冲")
    parser.add_argument("--autostart", action="store_true",
                        help="不等空格，直接开跑。给 --dry-run 或无人值守用")
    parser.add_argument("--scale", type=float, default=None,
                        help=f"动作整体缩放。第一次测建议 0.5（默认 {_ev.scale}）")
    args = parser.parse_args()

    import logging
    logging.basicConfig(level=logging.INFO, format="  %(message)s")

    ckpt = Path(args.ckpt).expanduser()
    if not ckpt.is_dir():
        print(f"找不到 checkpoint：{ckpt}")
        return 1

    config = AR5Config()
    # 命令行覆盖 config。给了什么就覆盖什么，没给的保持 config 里的值。
    for flag, field_name in (("hz", "hz"), ("max_steps", "max_steps"),
                             ("scale", "scale"), ("device", "device")):
        value = getattr(args, flag)
        if value is not None:
            setattr(config.eval, field_name, value)
    if args.stay:
        config.eval.return_to_start = False
    if args.no_goto_start:
        config.eval.goto_start = False
    ev = config.eval
    if ev.max_steps < 1 or ev.hz <= 0 or ev.scale <= 0:
        raise SystemExit(f"max_steps/hz/scale 都必须是正数，现在是 "
                         f"{ev.max_steps}/{ev.hz}/{ev.scale}")

    config.use_hand = not args.no_hand
    config.arm.use_realtime = True

    print(f"  checkpoint  {ckpt}")
    policy, preprocess, postprocess, model_dir, kind = load_policy(ckpt, ev.device)
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"  策略        {kind}  {n_params/1e6:.1f}M 参数  "
          f"chunk={policy.config.chunk_size} n_action_steps={policy.config.n_action_steps}")
    if n_params > 1e9:
        import torch as _t
        free, total = _t.cuda.mem_get_info()
        held = _t.cuda.memory_allocated()
        # 这里权重**已经在显存里了** —— 能走到这一行就说明装下了。拿 need 去比
        # free 是把权重算了两遍，π0.5 每次都会误报"装不下"，然后它照跑不误。
        # 真正要盯的是剩下的余量够不够激活值。
        print(f"  显存        权重占用 {held/1e9:.1f} GB，还空 {free/1e9:.1f} / "
              f"{total/1e9:.1f} GB")
        if free < 2e9:
            print(f"  ⚠ 余量不到 2 GB，前向可能 OOM。关掉别的 GPU 程序再跑。")

    # ACT 时序集成。**推理期开关，不用重训。**
    #
    # 默认（checkpoint 里 temporal_ensemble_coeff=None、n_action_steps=100）是
    # 一次预测 100 步、3.3 秒全开环执行、没有任何集成 —— ACT 最容易出问题的模式。
    # 症状：抓完之后又慢又飘，走没示范过的轨迹。原因是 ACT 是 L1 回归，观测一模糊
    # 就输出多条合理轨迹的**平均**，而示教长度 268~527 帧、时序差异极大，平均下来
    # 就是一条被抹平的低幅度轨迹。（π0.5 是流匹配，采样而非平均，所以没这个毛病。）
    #
    # 原论文的解法就是时序集成：每步重新预测整块，多个重叠 chunk 对同一时刻的
    # 预测按 exp(-coeff*i) 加权平均。ACT 前向只要 0.9 ms，30 Hz 下每步跑一次绰绰有余。
    if args.temporal_ensemble is not None:
        if kind != "act":
            print(f"\n  --temporal-ensemble 只对 act 有意义，这个 checkpoint 是 {kind}。")
            return 1
        from lerobot.policies.act.modeling_act import ACTTemporalEnsembler

        policy.config.temporal_ensemble_coeff = args.temporal_ensemble
        policy.config.n_action_steps = 1
        policy.temporal_ensembler = ACTTemporalEnsembler(
            args.temporal_ensemble, policy.config.chunk_size)
        policy.reset()
        print(f"  时序集成    coeff={args.temporal_ensemble}  "
              f"每步重新预测 {policy.config.chunk_size} 步并集成"
              f"（默认是一次预测 100 步全开环）")

    # VEL：给冻结的 backbone 外挂能量头，推理时对动作做一步残差修正。
    # baseline / VEL 的 A/B 只差这一个开关，别的什么都不变。
    vel = None
    if args.vel:
        if kind != "pi05":
            print(f"\n  --vel 目前只接 pi05，这个 checkpoint 是 {kind}。")
            return 1
        import torch as _t

        from atom_robot.ar5.vel.pi05_bridge import CORRECT_DIMS, SEGMENTS, VelRunner
        from atom_robot import vel_path as _vel
        sys.path.insert(0, str(_vel("vla-scripts")))
        from energy.energy_model import EnergyModel

        heads, head_names = {}, {}
        for spec in args.vel:
            seg, _, raw = spec.rpartition("=")
            seg = seg or "all"
            if seg not in (*SEGMENTS, "all"):
                print(f"\n  --vel {spec}：段名要是 {SEGMENTS} 或 all，不是 {seg!r}")
                return 1
            head_path = Path(raw).expanduser()
            if not head_path.is_file():
                print(f"\n  找不到能量头：{head_path}")
                print(f"  先跑 vel/precompute.py 再跑 vel/train_energy.py")
                return 1
            head = EnergyModel(state_dim=policy.model.paligemma_with_expert.paligemma.config
                               .text_config.hidden_size,
                               act_dim=policy.config.output_features["action"].shape[0])
            head.load_state_dict(_t.load(head_path, map_location=ev.device))
            heads[seg] = head.to(ev.device).float().eval()
            head_names[seg] = head_path.name
        alphas = {}
        flat = None
        for spec in (args.vel_alpha or ["0.01"]):
            seg, _, raw = spec.rpartition("=")
            if seg:
                alphas[seg] = float(raw)
            else:
                flat = float(raw)
        if alphas and flat is not None:
            print(f"\n  --vel-alpha 不能既给全局值又给分段值：{args.vel_alpha}")
            return 1
        alpha = alphas or flat
        vel = VelRunner(policy, heads, alpha=alpha, gamma=args.vel_gamma)
        vel.enabled = not args.vel_off
        if args.vel_off:
            print(f"  --vel-off：**只记能量，不做修正**（基线曲线）")
        n_par = sum(p.numel() for p in next(iter(heads.values())).parameters()) / 1e6
        print(f"  VEL         γ={args.vel_gamma}   {n_par:.2f}M 参数/头   "
              f"修正维 {CORRECT_DIMS.start}~{CORRECT_DIMS.stop-1}（夹爪直通）")
        for seg in (*SEGMENTS, "all"):
            if seg in heads:
                print(f"                {seg:<8} α={vel.alpha_for(seg):<8.4f} {head_names[seg]}")
        missing = [s for s in SEGMENTS if s not in heads and "all" not in heads]
        if missing:
            print(f"                {'、'.join(missing)} 没有头，**这几段原样放行**"
                  f"（等同基线）")

    task = args.task
    if task is None:
        meta = Path.home() / ".cache/huggingface/lerobot/vel_ar5_bolt/meta/tasks.parquet"
        try:
            import pandas as pd
            # 任务名在 **index** 里，列只有 task_index。iloc[0,0] 会读到那个 0。
            task = str(pd.read_parquet(meta).index[0])
        except Exception:
            task = "pick up the bolt and put it in the blue sleeve"
    print(f"  任务        {task}")

    camera_keys = list(config.cameras.sources)
    print(f"  相机        {camera_keys or '（无 —— 策略吃不到图像，没法评测）'}")
    if not camera_keys:
        print("\n  没有相机，ACT 必须有图像输入。检查 config.py 的 CameraConfig.sources。")
        return 1

    from atom_robot.ar5.collect import camera_problems
    problems = camera_problems(config)
    if problems:
        print("\n  相机开不了（臂还没上电）：")
        for name, why in problems.items():
            print(f"    {name}: {why}")
        print("\n  预览窗口开着的话去那个终端按 q。")
        return 1

    if not args.dry_run and not check_workspace(config, args.widen_workspace):
        return 1

    if args.dry_run:
        config.mock = True
        config.real_cameras_in_mock = True     # 模型必须看真画面，见 config 里的说明
        config.use_hand = False
        print("\n  ── DRY RUN：臂和手是模拟的，相机和策略是真的 ──")

    env = AR5Env(config)
    env.connect()
    if not args.dry_run and not getattr(env.arm, "_rt_active", False):
        print("\n  臂的实时通道没起来。**先上电**：")
        print("    python -m atom_robot.ar5.power_on --enable")
        print("  SDK 在未上电时报的是「网络异常: 网络连接错误」或者「实时模式异常」，")
        print("  和网络无关 —— 别去查网线。臂似乎有空闲超时，会自己断电。")
        env.close()
        return 1
    if not args.dry_run:
        # 上一次跑要是锁死过，这次开起来臂什么都不接。启动本身是一次明确的
        # 操作者动作，所以在这里清一次 —— collect 漏过这条，症状是"就是不动"。
        env.arm.clear_latch()

    period = 1.0 / ev.hz
    print(f"\n  控制频率 {ev.hz:.0f} Hz   动作缩放 x{ev.scale}   "
          f"一轮上限 {ev.max_steps} 步 = {ev.max_steps/ev.hz:.0f} 秒")
    median_demo = 409
    if ev.max_steps > median_demo * 3:
        print(f"  注意：示教每条中位 {median_demo} 帧 = {median_demo/ev.hz:.0f} 秒，"
              f"这一轮是它的 {ev.max_steps/median_demo:.0f} 倍。后半段策略在外推。")
    print("  空格 开始/暂停    q 结束本轮    x 退出")
    print("  过程中打标：" + "    ".join(f"{k} {v[1]}" for k, v in MILESTONES.items()))
    print("  本轮结束：" + "    ".join(f"{k} {v[1]}" for k, v in ENDINGS.items()) + "\n")

    stats = {"rollouts": 0, "success": 0, "fail": 0,
             "grasp": 0, "insert": 0, "insert_n": 0}
    marks = {v[0]: False for v in MILESTONES.values()}   # 本轮已经打过的里程碑
    ab_rng = np.random.default_rng(args.ab_seed)
    base_tag = args.tag

    def new_arm():
        """抽这一轮开不开 VEL。**随机而非严格交替** —— 严格交替时操作者数几轮
        就能推出当前是哪一臂，盲测就破了。种子固定，事后可复现。"""
        if not args.ab_interleave or vel is None:
            return
        vel.enabled = bool(ab_rng.integers(2))
        args.tag = f"{base_tag}-{'on' if vel.enabled else 'off'}"

    new_arm()
    last_round = {}                       # 上一轮的步数和位移，按 r/g/f 时写进 CSV
    rollout_csv = Path(args.log).expanduser()
    rollout_csv.parent.mkdir(parents=True, exist_ok=True)
    if migrate_csv(rollout_csv):
        print(f"  老 CSV 补齐了新增的列，历史行留空（当时没记，不能猜）")
    print(f"  逐轮结果 -> {rollout_csv}" + (f"   tag={args.tag}" if args.tag else ""))
    # 开跑前先回任务起点。采集是遥操作，起点是操作者的预备位；从别处起步，
    # 策略第一帧就看到训练分布外的输入（j0 差 46.8° 是实测过的）。
    if not args.dry_run and ev.goto_start:
        if not goto_task_start(env, ev, "开跑前回任务起点"):
            print("  没到起点就跑，第一帧是分布外输入。要坚持就加 --no-goto-start。")
            env.close()
            return 1
    elif not args.dry_run:
        print("  --no-goto-start：从当前位姿起跑，注意这可能是训练里没有的起点。")

    running = bool(args.autostart)
    if running:
        policy.reset()
        if vel is not None:
            vel.reset()
        print("  --autostart：直接开跑")
    step = 0
    deltas = []
    last_line = 0.0

    try:
        with RawKeyboard() as kb:
            while True:
                t0 = time.perf_counter()
                key = kb.key()

                if key in ("x", "X"):
                    break                       # 退出程序（finally 里回起点再断开）
                if key in ("q", "Q"):
                    # 结束本轮、回起点、**留在程序里**等下一轮。
                    # 原来 q 直接退整个进程，可评测本来就是要连着跑十几轮记
                    # 成功率的，每轮都重启一次要重新加载 4 GB 权重。
                    if deltas:
                        stats["rollouts"] += 1
                        last_round = round_stats(step, deltas)
                        print(f"\n\n  ■ 第 {stats['rollouts']} 轮结束（{step} 步）。"
                              f"逐帧位移 中位 {np.median(deltas):.2f} mm  "
                              f"最大 {max(deltas):.2f} mm")
                        if vel is not None and not args.ab_interleave:
                            print(vel.report())      # 盲测时不打，这行会泄露是哪一臂
                    else:
                        print("\n\n  ■ 本轮还没动过。")
                    running = False
                    if not args.dry_run and ev.goto_start:
                        goto_task_start(env, ev, "回起点")
                    policy.reset()
                    if vel is not None:
                        vel.reset()
                    step = 0
                    deltas = []
                    print("  " + " / ".join(f"{k} {v[1]}" for k, v in
                                           list(MILESTONES.items()) + list(ENDINGS.items()))
                          + "   /   空格 下一轮 / x 退出")
                if key == " ":
                    running = not running
                    if running:
                        # 这里 step 归零、policy.reset()，所以每次开跑都是新的一轮，
                        # 不是接着上一轮。臂被驱动过就得先回起点，否则新一轮的
                        # 第一帧从上一轮的终点开始 —— 训练里没有那种起点。
                        if deltas and not args.dry_run and ev.goto_start:
                            goto_task_start(env, ev, "新一轮，先回起点")
                        policy.reset()          # 清掉动作队列，从头预测
                        if vel is not None:
                            vel.reset()
                        step = 0
                        deltas = []
                        marks = {v[0]: False for v in MILESTONES.values()}
                        if vel is not None:
                            vel.trace.clear()    # 没记结果就重开，上一轮的曲线作废
                        new_arm()
                        print(f"\n  ● 开跑（第 {stats['rollouts']+1} 轮）")
                    else:
                        print("\n  ○ 暂停")
                # key 没按下时是 None —— 主循环每帧都走这里，不能直接 .lower()
                k = key.lower() if key is not None else None
                if k in MILESTONES:
                    # 里程碑：发生的当下就按，不等本轮结束。
                    name, label = MILESTONES[k]
                    if marks[name]:
                        print(f"\n  「{label}」已经记过了，忽略。")
                    else:
                        auto = [MILESTONES[p][1] for p in IMPLIES.get(k, ())
                                if not marks[MILESTONES[p][0]]]
                        for p in IMPLIES.get(k, ()):
                            marks[MILESTONES[p][0]] = True
                        marks[name] = True
                        print(f"\n  ✓ {label}" +
                              (f"（顺带补上漏按的：{'、'.join(auto)}）" if auto else ""))
                if k in ENDINGS:
                    ok, label = ENDINGS[k]
                    auto = [MILESTONES[p][1] for p in IMPLIES.get(k, ())
                            if not marks[MILESTONES[p][0]]]
                    if ok:
                        for p in IMPLIES["r"]:
                            marks[MILESTONES[p][0]] = True
                    stats["success" if ok else "fail"] += 1
                    stats["grasp"] += int(marks["grasp"])
                    stats["insert"] += int(marks["insert"])
                    stats["insert_n"] += int(marks["grasp"])   # 抓到了才谈得上插入
                    # 这一轮可能还没按 q 就先记了结果，那 deltas 里正是它的数据；
                    # 按过 q 的话 deltas 已清空，用 q 存下来的 last_round。
                    cur = round_stats(step, deltas) or last_round
                    vs = None
                    if vel is not None:
                        c = vel.counts()
                        vs = (c["grasp"], c["place"], vel.lost)
                    log_rollout(rollout_csv, stats, ok, marks, cur, args, ev, vel_stats=vs)
                    if vel is not None and vel.trace:
                        write_energy_trace(Path(args.energy_log).expanduser(), vel.trace,
                                           args, stats["success"] + stats["fail"], ok)
                    if vel is not None:
                        vel.trace.clear()        # 写过就清，下一轮从头记
                    if not cur:
                        print("\n  这一轮没有位移记录（臂没动过），CSV 里这几列留空。")
                    last_round = {}      # 用过就清，防止下一轮重复记上一轮的数
                    n = stats["success"] + stats["fail"]
                    inn = stats["insert_n"]
                    ins = (f"{stats['insert']}/{inn} = {100*stats['insert']/inn:.0f}%"
                           if inn else "—（还没抓到过）")
                    print(f"\n  ■ 本轮记「{label}」" +
                          (f"（顺带补上漏按的：{'、'.join(auto)}）" if auto else ""))
                    print(f"    抓取 {stats['grasp']}/{n} = {100*stats['grasp']/n:.0f}%"
                          f"    插入|抓到 {ins}"
                          f"    整体 {stats['success']}/{n}"
                          f" = {100*stats['success']/n:.0f}%   -> {rollout_csv.name}")
                    marks = {v[0]: False for v in MILESTONES.values()}   # 下一轮从头记

                if not args.dry_run and env.arm.latched:
                    print("\n\n  ✗ 臂刹车锁死（latch），停止。退出重开会自动清。")
                    break

                if not running:
                    if not args.dry_run:
                        # 暂停时也要持续下发目标，否则 0.25 秒没有目标 sender
                        # 就会刹车 latch —— 和采集时同一个坑。
                        env.step_joints(env.get_qpos()[:config.arm.num_joints], observe=False)
                    time.sleep(period)
                    continue

                obs = env.get_observation(t=step)
                import torch
                batch = preprocess(build_batch(obs, camera_keys, ev.device, task))
                if vel is not None:
                    # VEL 自己管 no_grad —— 修正要对动作求梯度，inference_mode 会禁掉
                    action = vel.select_action(batch)
                else:
                    with torch.inference_mode():
                        action = policy.select_action(batch)
                action = postprocess(action).squeeze(0).float().cpu().numpy()
                if ev.scale != 1.0:
                    action = action.copy()
                    action[:6] *= ev.scale
                if vel is not None:
                    # **必须喂真实单位的动作和末端位姿。** 段是按夹爪 <0.15 和
                    # 末端高度切的，select_action 里拿到的还是归一化空间，
                    # 阈值和毫米都对不上。
                    # qpos 最后一维是夹爪的**实际开合**（get_opening），不是命令值。
                    # 「命令闭合但状态显示手里是空的」= 脱手，见 GRIP_EMPTY。
                    qpos = obs.observation.get("qpos")
                    vel.observe(action, obs.observation.get("ee_pose"),
                                grip_state=None if qpos is None else float(qpos[-1]))

                deltas.append(float(np.linalg.norm(action[:3])) * 1000)
                if not args.dry_run:
                    env.step(action)

                step += 1
                if t0 - last_line >= 0.2:
                    last_line = t0
                    print(f"\r  步 {step:4d}/{ev.max_steps}   位移 {deltas[-1]:5.2f} mm   "
                          f"手 {action[6]:.2f}   中位 {np.median(deltas):4.2f} mm   ",
                          end="", flush=True)

                if step >= ev.max_steps:
                    running = False
                    stats["rollouts"] += 1
                    last_round = round_stats(step, deltas)
                    # 没有 tty 就没人能按键，再等下去是死循环。--autostart
                    # 常常配合重定向或后台跑，那种场景下跑完就该退出。
                    if not kb.tty:
                        print(f"\n\n  到 {ev.max_steps} 步，无 tty，退出。"
                              f"逐帧位移 中位 {np.median(deltas):.2f} mm  "
                              f"最大 {max(deltas):.2f} mm")
                        if vel is not None and not args.ab_interleave:
                            print(vel.report())      # 盲测时不打，这行会泄露是哪一臂
                        break
                    print(f"\n\n  到 {ev.max_steps} 步，停。"
                          f"逐帧位移 中位 {np.median(deltas):.2f} mm  最大 {max(deltas):.2f} mm")
                    if vel is not None and not args.ab_interleave:
                        print(vel.report())          # 盲测时不打，这行会泄露是哪一臂
                    print("  " + " / ".join(f"{k} {v[1]}" for k, v in
                                           list(MILESTONES.items()) + list(ENDINGS.items()))
                          + "   /   空格 下一轮 / x 退出")

                sleep = period - (time.perf_counter() - t0)
                if sleep > 0:
                    time.sleep(sleep)
    finally:
        print()
        if not args.dry_run and ev.return_to_start:
            goto_task_start(env, ev, "回任务起点")
        elif not ev.return_to_start:
            print("  --stay：臂留在原地。")
        env.close()
        if not args.dry_run:
            powered = "已断电，下次要先 power_on --enable" if \
                config.arm.power_off_on_disconnect else "仍带电，保持当前位姿"
            print(f"  臂{powered}")
        total = stats["success"] + stats["fail"]
        if total:
            print(f"\n  {stats['success']}/{total} 成功  ({stats['success']/total*100:.0f}%)")
        print(f"  env 统计 {env.stats}")
    return 0


if __name__ == "__main__":
    # 放最外层：加载阶段就 OOM 的时候 main 的 finally 根本走不到，
    # 而那正是最容易被那条 AttributeError 混淆视听的时刻。
    try:
        _code = main()
    finally:
        silence_shutdown_noise()
    sys.exit(_code)
