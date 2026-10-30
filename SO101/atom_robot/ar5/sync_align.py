"""OmniPicker 采集的软件对齐（2026-10-21 用户定）：以腕部相机的每一帧为一行，把臂、夹爪、另一路相机
对到这一帧的拍摄时刻。纯计算，不碰硬件，`SyncRecorder` 存盘前调一次。

为什么要这样：两台相机、臂、夹爪各走各的钟，没有硬件同步。原来回路 50 Hz 里每到 30 Hz 节拍录一行，
取「最新的那张图」配「当下的臂」，图比臂早 25~60 ms 且每行不一样，行间隔还是 40/40/20 ms 轮换。
现在采集时各来源全量带时刻记下来，这里再对齐：

* **基准** = 腕部 305 的每一帧，时刻是内核记的拍摄时刻（V4L2，和 perf_counter 同一个钟）。
  行间隔就是相机自己的 33.3 ms；控制器两路输入里它跟着臂动，对不齐最显眼。
* **臂的关节角、夹爪开合**：按时间加权插值（不是两点平均 —— 那等于取两点正中那一刻，最多偏半个采样间隔）。
* **末端姿态**：球面插值（不对轴角三个分量直接插）。
* **夹爪指令、held**：不连续，取这一刻正在生效的值（0 和 1 平均出 0.5 没有意义）。
* **另一路相机**（顶部 335）：图没法插值，取拍摄时刻最近的一张；它只有「程序收到」的时刻，
  `latency` 给了就先扣掉再比。
* **暂停**：每段单独对齐，不跨暂停插值；段与段在时间上接起来（下一段第一行 = 上一段最后一行 + 1/fps），
  原始数据也只留每段基准帧覆盖的那一截、同样平移 —— 数据里看不出停过（用户 10-20 的要求）。
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .transforms import axis_angle_to_matrix, matrix_to_axis_angle, pose_delta

# 段首尾的基准帧离最近的臂 / 夹爪读数在这以内，就用端点的值（段首是刚接着录、段尾是暂停后停稳，
# 臂这时都停着）；再远就丢掉这一帧，不外推。
EDGE_HOLD_S = 0.05


def new_segment() -> dict:
    """一段连续录制（两次暂停之间）的原始数据。时刻都是 time.perf_counter 的钟。"""
    return {"cams": {}, "arm_t": [], "arm_q": [], "arm_ee": [],
            "grip_t": [], "grip": [], "cmd_t": [], "cmd": [], "held": []}


def _slerp_rotvec(ts, rotvecs, t):
    """姿态（轴角）在时刻 t 的球面插值：R0 · exp(w · log(R0ᵀ R1))。超出两端就取端点。"""
    if len(ts) == 1:
        return np.asarray(rotvecs[0], dtype=np.float64)
    i = int(np.clip(np.searchsorted(ts, t, side="right") - 1, 0, len(ts) - 2))
    span = ts[i + 1] - ts[i]
    w = 0.0 if span <= 0 else float(np.clip((t - ts[i]) / span, 0.0, 1.0))
    R0 = axis_angle_to_matrix(rotvecs[i])
    R1 = axis_angle_to_matrix(rotvecs[i + 1])
    step = matrix_to_axis_angle(R0.T @ R1) * w
    return matrix_to_axis_angle(R0 @ axis_angle_to_matrix(step))


def _hold(ts, values, t):
    """t 这一刻正在生效的值：最后一个时刻 ≤ t 的；比第一个还早就用第一个。"""
    i = int(np.searchsorted(ts, t, side="right") - 1)
    return values[max(i, 0)]


def _gap_used(ts, t):
    """插值 t 用到的前后两个读数相隔多久（秒）。超出两端返回到端点的距离。"""
    i = int(np.searchsorted(ts, t, side="left"))
    if i == 0:
        return float(ts[0] - t)
    if i >= len(ts):
        return float(t - ts[-1])
    return float(ts[i] - ts[i - 1])


def align(segments: List[dict], master: str, fps: float,
          latency: Optional[Dict[str, float]] = None) -> dict:
    """把各段原始数据对齐到 `master` 相机的帧上，返回存盘用的数组和对齐统计。

    segments[k]["cams"][name] 是 [(拍摄或收到时刻, 帧), ...]，按时刻升序；其余键见 new_segment()。
    返回的 dict：
        timestamp (N,)、state (N,8)、ee_pose (N,6)、action (N,7)、held (N,)、
        frames {相机: [帧...]}、t_<相机> (N,)（基准相机就是 timestamp；别的是选中那张的时刻）、
        raw {raw_arm_t、raw_arm_q、raw_arm_ee、raw_gripper_t、raw_gripper、raw_cmd_t、raw_cmd}、
        stats（给存盘时打日志）。时刻全部相对第一行、扣掉了暂停。
    """
    latency = dict(latency or {})
    period = 1.0 / fps
    rows = {"t": [], "q": [], "op": [], "ee": [], "cmd": [], "held": []}
    frames: Dict[str, list] = {}
    others_t: Dict[str, list] = {}
    raw = {k: [] for k in ("raw_arm_t", "raw_arm_q", "raw_arm_ee", "raw_gripper_t", "raw_gripper",
                           "raw_cmd_t", "raw_cmd")}
    stats = {"rows": 0, "dropped_edge": 0, "stale": 0, "camera_gaps": 0, "arm_gap": [], "grip_gap": [],
             "other_offset": {}, "segments": 0}
    last_rel = None

    for seg in segments:
        cams = seg["cams"]
        base = cams.get(master) or []
        arm_t = np.asarray(seg["arm_t"], dtype=np.float64)
        grip_t = np.asarray(seg["grip_t"], dtype=np.float64)
        cmd_t = np.asarray(seg["cmd_t"], dtype=np.float64)
        if not base or len(arm_t) == 0 or len(cmd_t) == 0:
            stats["dropped_edge"] += len(base)
            continue
        arm_q = np.asarray(seg["arm_q"], dtype=np.float64)
        arm_ee = np.asarray(seg["arm_ee"], dtype=np.float64)
        grip = np.asarray(seg["grip"], dtype=np.float64)
        cmd = np.asarray(seg["cmd"], dtype=np.float64)
        held = np.asarray(seg["held"], dtype=bool)
        others = {name: (np.asarray([t - latency.get(name, 0.0) for t, _ in fr], dtype=np.float64), fr)
                  for name, fr in cams.items() if name != master}

        lo = max(arm_t[0], grip_t[0] if len(grip_t) else arm_t[0]) - EDGE_HOLD_S
        hi = min(arm_t[-1], grip_t[-1] if len(grip_t) else arm_t[-1]) + EDGE_HOLD_S
        keep = [(t, f) for t, f in base if lo <= t <= hi]
        if any(len(o[0]) == 0 for o in others.values()):
            keep = []                       # 这一段另一路相机一帧都没有：整段不要
        # 段首比第一次读臂还早的，是暂停时（或开录前）拍的、积在相机缓存里的旧帧，本来就不要，不算丢；
        # 10-21 第一条真数据 5 次暂停「丢」了 42 帧全是这种。真正丢的是录制开始之后拍、却没对上的
        stale = sum(1 for t, _ in base if t < lo)
        stats["stale"] += stale
        stats["dropped_edge"] += len(base) - len(keep) - (stale if keep else 0)
        if not keep:
            continue
        stats["segments"] += 1
        ts = np.asarray([t for t, _ in keep], dtype=np.float64)
        stats["camera_gaps"] += int(np.sum(np.diff(ts) > 1.5 * period))

        # 这一段平移多少：第一段第一行是 0，之后每段接在上一段最后一行后面一个周期
        shift = ts[0] - (0.0 if last_rel is None else last_rel + period)
        last_rel = ts[-1] - shift

        rows["t"].extend(ts - shift)
        rows["q"].extend(np.stack([np.interp(ts, arm_t, arm_q[:, j]) for j in range(arm_q.shape[1])], axis=1))
        if len(grip_t):
            rows["op"].extend(np.interp(ts, grip_t, grip))
            stats["grip_gap"].extend(_gap_used(grip_t, t) for t in ts)
        else:
            rows["op"].extend(np.full(len(ts), np.nan))
        pos = np.stack([np.interp(ts, arm_t, arm_ee[:, j]) for j in range(3)], axis=1)
        rot = np.stack([_slerp_rotvec(arm_t, arm_ee[:, 3:6], t) for t in ts])
        rows["ee"].extend(np.concatenate([pos, rot], axis=1))
        rows["cmd"].extend(_hold(cmd_t, cmd, t) for t in ts)
        rows["held"].extend(bool(_hold(cmd_t, held, t)) for t in ts)
        stats["arm_gap"].extend(_gap_used(arm_t, t) for t in ts)
        frames.setdefault(master, []).extend(f for _, f in keep)

        for name, (o_t, o_frames) in others.items():
            idx = np.clip(np.searchsorted(o_t, ts), 1, len(o_t) - 1) if len(o_t) > 1 else np.zeros(len(ts), int)
            if len(o_t) > 1:
                left = idx - 1
                idx = np.where(np.abs(o_t[left] - ts) <= np.abs(o_t[idx] - ts), left, idx)
            frames.setdefault(name, []).extend(o_frames[i][1] for i in idx)
            others_t.setdefault(name, []).extend(o_t[idx] - shift)
            stats["other_offset"].setdefault(name, []).extend(o_t[idx] - ts)

        # 原始数据：只留这一段基准帧覆盖的那一截、两头各多半个周期，同样平移。相邻两段正好在
        # 接缝处（上一段最后一行后半个周期）接上，读数间隔和平时一样，看不出停过
        w_lo, w_hi = ts[0] - period / 2, ts[-1] + period / 2
        a = (arm_t >= w_lo) & (arm_t < w_hi)
        raw["raw_arm_t"].extend(arm_t[a] - shift)
        raw["raw_arm_q"].extend(arm_q[a])
        raw["raw_arm_ee"].extend(arm_ee[a])
        g = (grip_t >= w_lo) & (grip_t < w_hi) if len(grip_t) else np.zeros(0, bool)
        raw["raw_gripper_t"].extend(grip_t[g] - shift)
        raw["raw_gripper"].extend(grip[g])
        c = (cmd_t >= w_lo) & (cmd_t < w_hi)
        raw["raw_cmd_t"].extend(cmd_t[c] - shift)
        raw["raw_cmd"].extend(cmd[c])

    n = len(rows["t"])
    stats["rows"] = n
    out = {"timestamp": np.asarray(rows["t"], dtype=np.float64), "frames": frames, "stats": stats}
    if n == 0:
        out.update(state=np.zeros((0, 8), np.float32), ee_pose=np.zeros((0, 6), np.float32),
                   action=np.zeros((0, 7), np.float32), held=np.zeros(0, bool))
    else:
        q = np.asarray(rows["q"], dtype=np.float64)
        op = np.asarray(rows["op"], dtype=np.float64)
        ee = np.asarray(rows["ee"], dtype=np.float64)
        cmd = np.asarray(rows["cmd"], dtype=np.float64)
        cmd = np.where(np.isnan(cmd), op, cmd)          # 还没按过空格 / O：和原来一样记实测开合
        delta = np.zeros((n, 6))
        for i in range(1, n):
            delta[i] = pose_delta(ee[i - 1], ee[i])
        out.update(
            state=np.concatenate([q, op[:, None]], axis=1).astype(np.float32),
            ee_pose=ee.astype(np.float32),
            action=np.concatenate([delta, cmd[:, None]], axis=1).astype(np.float32),
            held=np.asarray(rows["held"], dtype=bool),
        )
    out[f"t_{master}"] = out["timestamp"].copy()
    for name, v in others_t.items():
        out[f"t_{name}"] = np.asarray(v, dtype=np.float64)
    out["raw"] = {k: (np.asarray(v, dtype=np.float64) if v else np.zeros(0)) for k, v in raw.items()}
    return out


def report(stats: dict, master: str, latency: Optional[Dict[str, float]] = None,
           kernel: tuple = ()) -> List[str]:
    """对齐统计 → 存盘时打的几句话。`kernel`：时刻本来就是拍摄时刻的来源（不用扣延迟）。"""
    latency = latency or {}
    lines = [f"以 {master} 为基准 {stats['rows']} 行"
             + (f"（相机掉帧 {stats['camera_gaps']} 处）" if stats["camera_gaps"] else "")
             + (f"，⚠ 录制中有 {stats['dropped_edge']} 帧没对上、丢了" if stats["dropped_edge"] else "")]
    for name, key in (("臂", "arm_gap"), ("夹爪", "grip_gap")):
        gaps = np.asarray(stats[key]) * 1000
        if len(gaps):
            lines.append(f"{name}插值用的两点相隔 中位 {np.median(gaps):.0f} ms、最大 {gaps.max():.0f} ms")
    for name, off in stats["other_offset"].items():
        off = np.abs(np.asarray(off)) * 1000
        how = (f"扣了 {latency[name] * 1000:.0f} ms 延迟" if latency.get(name)
               else "按拍摄时刻" if name in kernel else "按收到时刻、没扣它的固定延迟")
        lines.append(f"{name} 取最近的一张，离基准 中位 {np.median(off):.0f} ms、最大 {off.max():.0f} ms（{how}）")
    return lines
