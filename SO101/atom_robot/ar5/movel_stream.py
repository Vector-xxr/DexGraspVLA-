"""排队式 MoveL 流：把末端目标位姿交给**控制器**去解逆解、做插补（collect 的 `--ik sdk`）。

和本地逆解那两条（`--ik dls / weighted`）唯一的区别是「目标位姿怎么变成关节运动」：
本地那两条在本机解出 7 个关节角、走实时关节流；这条只发法兰位姿，关节怎么转由控制柜定。
**试验模式，只用来对比手感**（2026-10-31 用户定）。

移植自 `数据_硬件/灵巧手遥操/VR设备/movel_stream.py`（2026-09-18~20 VR 遥操真机调过），
只搬了 collect 用得到的部分，并按记下来的坑改了三处：

* **不写工具设置。** `setToolset` 会持久化在控制器里，带负载时写它会掉电。这里只读一次，
  工具不是 0 就拒绝启动（目标按法兰发，工具不是 0 的话控制器会把它当成工具点，整体偏掉）。
  指尖偏移由调用方在本地换算成法兰目标。
* **`moveReset` 只在臂空闲时调。** 运动中调会被拒（ec=-20）。松开 / 冻结只是不再追加，
  队列在前导预算内（几厘米）自己走完；再接合前等空闲再 `start()`。
* **速度 600、转弯区 100**：`probe_movel_speed.py` 单段实测的结论 —— 设定速度决定加速度档，
  <500 落在低档；转弯区是分档百分比，3 落在 10% 档每点都停车。VR 代码里的默认 250 / 3
  早于那次探测。

真机上已知的行为（见 VR 那边的记录）：跟随落后 100~200 mm、延迟 240~500 ms 且随手速增长；
手一慢队列就走空，臂停下、重启约 170 ms，表现为「钝一下」。

10-31 第一次上 collect 真机：死区 1 mm / 0.5° 下一秒塞进几十条 0.1 mm 的碎段，每段却要 ~90 ms，
队列积到下发到走完中位 1.9 s，臂「基本不跟」。于是加了**按时间估算的积压闸门**（`backlog_ms`），
死区默认改成 3 mm / 2°，每次只发最新的那个目标。
"""
from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np

from .transforms import axis_angle_to_matrix, matrix_to_axis_angle, matrix_to_rpy, rpy_to_matrix

logger = logging.getLogger(__name__)

ELBOW_MODES = ("current", "start", "none")


def _rot_angle(R: np.ndarray) -> float:
    return float(np.arccos(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)))


class MoveLStreamer:
    """submit() 只进本地缓冲，flush() 才碰 SDK。目标都是**法兰**在基座系的 4x4 位姿。

    elbow（7 轴冗余的那一维怎么定）：
      "current"  每次下发前取臂当前的臂角和构型（VR 实测：只取一次时末端移开 20 cm 臂扭着走，
                 每段 250 ms；每段刷新 65~82 ms）
      "start"    开始跟随时取一次（VR 的 movel_stream.py 原样）
      "none"     不传，让控制器自己选解。⚠ 驱动里记过：不带臂角时控制器可能解不出、
                 **悄悄丢指令**，只在控制器日志里报 50002「目标点超出运动范围」
    """

    # 每条 MoveL 都要加速再减速，再短也要这么久。10-31 真机 SDK 日志：段长中位 0.1 mm 时
    # 控制器约 87 ms 才走完一条，和长短几乎无关
    MIN_SEGMENT_S = 0.09
    # 段时长 ≈ 2√(L/a)（probe_movel_speed 单段实测：10 mm 160 ms、80 mm 364 ms，a 取 1.5 偏保守）
    SEGMENT_ACCEL = 1.5

    # 批量模式：队列里剩不到这么多个点才发下一批
    REFILL_AT = 3
    # 一批里和前一个点几乎重合的剔掉：控制器拒「相邻两个目标点过近」（ec -50003，实测 0.01 mm 拒、0.05 mm 收）
    DEDUP_M, DEDUP_RAD = 1e-4, np.deg2rad(0.1)
    # 判断臂走到第几个点时，1 rad 的姿态差折算成多少米
    ROT_WEIGHT_M = 0.05

    def __init__(self, robot, sdk, *, speed: float = 600.0, zone: float = 100.0,
                 lead_ms: float = 50.0, min_step_mm: float = 3.0, min_rotation_deg: float = 2.0,
                 max_rotation_lead: float = 0.15, backlog_ms: float = 150.0, batch: int = 1,
                 merge_mm: float = 0.0, merge_deg: float = 2.0,
                 cache: int = 300, buffer_max: int = 64, elbow: str = "current"):
        vals = [speed, zone, lead_ms, min_step_mm, min_rotation_deg, max_rotation_lead, backlog_ms]
        if not np.all(np.isfinite(vals)) or speed <= 0 or lead_ms <= 0 or zone < 0 or backlog_ms <= 0:
            raise ValueError(f"MoveL 参数不对：speed={speed} zone={zone} lead_ms={lead_ms} backlog_ms={backlog_ms}")
        if min_step_mm < 0 or min_rotation_deg < 0 or max_rotation_lead <= 0:
            raise ValueError("死区不能是负数，姿态前导上限要是正数")
        if elbow not in ELBOW_MODES:
            raise ValueError(f"elbow 只能是 {ELBOW_MODES}，给的是 {elbow!r}")
        if not 1 <= int(batch) <= 50:
            raise ValueError(f"batch 要在 1~50，给的是 {batch}")
        # 一次 moveAppend 发几个点（10-31 用户要一次发 10 个）。1 = 逐条发、走上面的积压闸门；
        # >1 = 攒满一批一起发，按臂实测走到第几个点来续发（_flush_batch）
        self.batch = int(batch)
        self._inflight: list = []          # 已发、臂还没走到的点（批量模式）
        if merge_mm < 0 or merge_deg < 0:
            raise ValueError("merge_mm / merge_deg 不能是负数")
        # 批量模式下发之前把接近共线的点合掉（10-31 加）。真机：一次发 10 个 0.8 mm 的点，
        # 控制器每个点照样 ~64 ms（转弯区只把拐角磨圆、没把 10 段连成一段），一批 8 mm 走 0.6 s。
        # 控制器的耗时跟点数走、不跟距离走，就在本地把点数减下来。0 = 不合
        self.merge_m, self.merge_rad = merge_mm / 1000.0, np.deg2rad(merge_deg)
        # 合并之后每段长，排多了一样积压：只剩最后一段时就续发
        self.refill_at = 1 if self.merge_m > 0 else self.REFILL_AT
        self.robot, self.sdk = robot, sdk
        self.speed, self.zone = float(speed), float(zone)
        # 前导预算：已下发的目标最多领先臂多远。VR 实测「嫌延迟高要提速度，不是降预算」
        self.max_lead = self.speed * lead_ms / 1e6
        self.min_step = min_step_mm / 1000.0
        self.min_rotation = np.deg2rad(min_rotation_deg)
        self.max_rotation_lead = float(max_rotation_lead)
        # 积压闸门（10-31 加）：按时间估算队列里还有多久没走完，超过它就先不发。
        # 光有上面那个**距离**预算不够 —— 真机上一堆 0.1 mm 的碎段距离加起来很短、全被放行，
        # 每段却要 ~90 ms，队列积到下发到走完中位 1.9 s、最长 4.9 s，「基本不跟」
        self.max_backlog = backlog_ms / 1000.0
        self._busy_until = 0.0
        self.cache, self.buffer_max = int(cache), int(buffer_max)
        self.elbow = elbow
        self._elbow, self._conf = 0.0, None
        self._last_T: Optional[np.ndarray] = None       # 最后一个已下发的目标
        self._running = self._pending = False
        self._buf: list = []
        self.stats = dict.fromkeys((
            "samples", "appended", "skipped_tiny", "skipped_lead", "skipped_backlog", "decimated", "flushes",
            "batch_sum", "skipped_queue", "deduped", "merged",
            "batch_max", "restarts", "already_moving", "queue_empty", "start_failed",
            "execution_errors", "pauses", "halts"), 0)
        self.last_error = ""

    # ── SDK 调用 ──────────────────────────────────────────────────────
    def _call(self, method, *args):
        ec = {}
        result = method(*args, ec)
        if ec.get("ec", 0):
            raise RuntimeError(f"{getattr(method, '__name__', 'SDK')} 失败：{ec}")
        return result

    def is_idle(self) -> bool:
        # 只认 idle：error / unknown / drag 都不能当成「可以开始」
        return self._call(self.robot.operationState) == self.sdk.OperationState.idle

    def _pose(self):
        cp = self._call(self.robot.cartPosture, self.sdk.CoordinateType.flangeInBase)
        T = np.eye(4)
        T[:3, :3] = rpy_to_matrix(cp.rpy)
        T[:3, 3] = np.asarray(cp.trans, dtype=np.float64)
        return T, cp

    def wait_idle(self, timeout: float = 8.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.is_idle():
                return True
            time.sleep(0.05)
        return False

    # ── 进入排队模式 ──────────────────────────────────────────────────
    def configure(self, log=print) -> bool:
        """切到排队模式并上电。顺序照 VR 那版真机跑通的：切模式 → 自动模式 → **紧跟着**上电
        （中间插别的调用时 setPowerState 41 ms 返回成功但电机不上，真上电要 ~214 ms）。"""
        if not self.wait_idle():
            log("  [MoveL] 等控制器空闲超时，没切模式")
            return False
        try:
            self._call(self.robot.setMotionControlMode, self.sdk.MotionControlMode.NrtCommandMode)
            self._call(self.robot.setOperateMode, self.sdk.OperateMode.automatic)
            t = time.monotonic()
            self._call(self.robot.setPowerState, True)
            log(f"  [MoveL] 上电返回 {1000 * (time.monotonic() - t):.0f} ms")
            self.ensure_power(log=log)
            tool = self._call(self.robot.toolset)
            end = (list(tool.end.trans), list(tool.end.rpy))
            if not (np.allclose(end[0], 0.0, atol=1e-6) and np.allclose(end[1], 0.0, atol=1e-6)):
                # 不替用户改：setToolset 持久化、带负载写会掉电（见文件顶部）
                log(f"  [MoveL] ✗ 控制器的工具不是 0（{end}），目标按法兰发会整体偏掉。"
                    "不替你改 —— 用示教器或 VR 那边的工具把它归 0 再来")
                return False
            self._call(self.robot.setDefaultConfOpt, False)
            self._call(self.robot.setMaxCacheSize, self.cache)
            self._call(self.robot.setDefaultSpeed, self.speed)
            self._call(self.robot.setDefaultZone, self.zone)
            self.ensure_power(log=log)
        except Exception as exc:                # noqa: BLE001
            self.last_error = str(exc)
            log(f"  [MoveL] ✗ 配置失败：{exc}")
            return False
        return True

    def ensure_power(self, wait_s: float = 3.0, attempts: int = 3, log=print) -> None:
        """moveStart 前电机必须是开的（不然 ec=-17）。读回来确认，没开就重试几次。"""
        state = self._call(self.robot.powerState)
        if str(state).endswith("on"):
            return
        for i in range(attempts):
            log(f"  [MoveL] 电机 {state}，第 {i + 1}/{attempts} 次上电")
            self._call(self.robot.setOperateMode, self.sdk.OperateMode.automatic)
            self._call(self.robot.setPowerState, True)
            deadline = time.monotonic() + wait_s
            while time.monotonic() < deadline:
                time.sleep(0.1)
                state = self._call(self.robot.powerState)
                if str(state).endswith("on"):
                    return
            time.sleep(1.0)
        raise RuntimeError(f"上电后电机还是关的：{state}")

    # ── 起停 ──────────────────────────────────────────────────────────
    @property
    def running(self) -> bool:
        return self._running

    def start(self) -> None:
        """开始跟随。**要求臂空闲**（调用方先看 is_idle()）：清掉旧队列，从臂当前位姿起步。"""
        self.ensure_power()
        self._call(self.robot.moveReset)
        T, cp = self._pose()
        self._elbow, self._conf = float(cp.elbow), list(cp.confData)
        self._last_T = T
        self._running, self._pending = True, False
        self._buf = []
        self._busy_until = 0.0
        self._inflight = []

    def pause(self) -> None:
        """松开 / 冻结：不再追加，已排队的在前导预算内自己走完。不碰 SDK。"""
        if self._running:
            self.stats["pauses"] += 1
        self._running = self._pending = False
        self._buf = []

    def halt(self, timeout: float = 8.0) -> bool:
        """回位、退出之前：停追加、等臂停稳、清队列。**不用 robot.stop()** —— 这台 AR5 上它是掉电。"""
        self.pause()
        self.stats["halts"] += 1
        idle = self.wait_idle(timeout)
        if idle:
            self._call(self.robot.moveReset)
        return idle

    def step(self, engaged: bool, target: Optional[np.ndarray], on_start=None) -> str:
        """collect 每个回路调一次：接合着就跟、松开就停追加，然后 flush。

        **只有松开才 pause，「接合着但这一拍没有目标」什么都不做。** 映射器重锚后的第一拍
        在取主臂锚点、本来就不出目标 —— 2026-10-31 第一版把「没目标」也当成松开去 pause，
        于是 start → 重锚 → 下一拍没目标 → pause → start … 每两拍一循环，一条指令都没发出去，
        真机上「主臂怎么动臂都不动」。

        没在跟、臂又空闲了才起步（上一段队列走完、停稳），起步后调 on_start 让调用方从停稳的
        地方重新锚定 —— 不然新目标是按臂还在滑的时候的位置算的，起步那一下会往回拽。
        """
        if not engaged:
            if self._running:
                self.pause()
        elif self._running:
            if target is not None:
                self.submit(target)
        elif self.is_idle():
            self.start()
            if on_start is not None:
                on_start()
        return self.flush()

    # ── 下发 ──────────────────────────────────────────────────────────
    def submit(self, T_flange: np.ndarray) -> None:
        if not self._running:
            return
        T = np.asarray(T_flange, dtype=np.float64).reshape(4, 4)
        if not np.all(np.isfinite(T)):
            raise ValueError("MoveL 目标里有非有限数")
        self.stats["samples"] += 1
        self._buf.append(T.copy())
        if len(self._buf) > self.buffer_max:
            self._buf.pop(0)

    def _changed(self, T, ref) -> bool:
        dp = float(np.linalg.norm(T[:3, 3] - ref[:3, 3]))
        dr = _rot_angle(T[:3, :3] @ ref[:3, :3].T)
        return dp > max(self.min_step, 1e-9) or dr > max(self.min_rotation, 1e-7)

    def _poll_event(self) -> None:
        event_enum = getattr(self.sdk, "Event", None)
        if event_enum is None or not hasattr(self.robot, "queryEventInfo"):
            return
        event = self._call(self.robot.queryEventInfo, event_enum.moveExecution)
        if not event:
            return
        error = event.get("error", 0)
        if isinstance(error, dict):
            code = error.get("ec", error.get("value", 0))
        elif isinstance(error, (int, float)):
            code = error
        else:
            value = getattr(error, "value", 0)
            code = value() if callable(value) else value
        if code:
            self.stats["execution_errors"] += 1
            self.last_error = f"MoveL 执行失败：{event}"
            self.pause()
            raise RuntimeError(self.last_error)

    def flush(self) -> str:
        """每个回路调一次。返回这一拍干了什么（给诊断看）。"""
        if not self._running:
            return "stopped"
        self._poll_event()
        idle = self.is_idle()
        cur, cp = self._pose()
        if self.elbow == "current":
            self._elbow, self._conf = float(cp.elbow), list(cp.confData)
        if self.batch > 1:
            return self._flush_batch(idle, cur)
        if not self._buf:
            return self._try_start() if (self._pending and idle) else "empty"

        now = time.monotonic()
        if idle:
            self._busy_until = now          # 队列走空了，估算清零
        elif self._busy_until - now > self.max_backlog:
            # 按时间估算还有一段多没走完：再塞只会排队。留最新那个，下一拍再试
            self.stats["skipped_backlog"] += max(0, len(self._buf) - 1)
            self._buf = self._buf[-1:]
            return "backlog"

        lead = float(np.linalg.norm(self._last_T[:3, 3] - cur[:3, 3]))
        rot_lead = _rot_angle(self._last_T[:3, :3] @ cur[:3, :3].T)
        budget = self.max_lead - lead
        if (budget <= 1e-8 or rot_lead >= self.max_rotation_lead) and not idle:
            # 已经欠着够多了，现在发只会加深积压。留最新那个，下一拍再试
            self.stats["skipped_lead"] += max(0, len(self._buf) - 1)
            self._buf = self._buf[-1:]
            return "lead"

        # 只发最新的那个目标：中间那些点每个都要 ~90 ms，发出去只会排队。发的是绝对位姿，
        # 最新的已经包含了中间的全部信息。死区相对上一个**已下发**的点
        latest, dropped = self._buf[-1], len(self._buf) - 1
        self._buf = []
        if not self._changed(latest, self._last_T):
            self.stats["skipped_tiny"] += dropped + 1
            return self._try_start() if (self._pending and idle) else "tiny"
        self.stats["decimated"] += dropped
        keep = [latest]

        # 按前导预算截：发出去的折线不超过预算，超出的那个点只走到预算为止，剩下的留到下一拍
        dist_left = self.max_lead if idle else max(budget, 0.0)
        ang_left = self.max_rotation_lead if idle else max(self.max_rotation_lead - rot_lead, 0.0)
        anchor, bounded = self._last_T, []
        for T in keep:
            d = float(np.linalg.norm(T[:3, 3] - anchor[:3, 3]))
            rot = matrix_to_axis_angle(T[:3, :3] @ anchor[:3, :3].T)
            ang = float(np.linalg.norm(rot))
            ratio = min(1.0, dist_left / max(d, 1e-12), ang_left / max(ang, 1e-12))
            if ratio > 1e-8:
                P = np.eye(4)
                P[:3, :3] = axis_angle_to_matrix(rot * ratio) @ anchor[:3, :3]
                P[:3, 3] = anchor[:3, 3] + ratio * (T[:3, 3] - anchor[:3, 3])
                bounded.append(P)
            if ratio < 1.0 - 1e-9:
                self._buf = [keep[-1]]
                self.stats["decimated"] += len(keep) - len(bounded)
                break
            dist_left -= d
            ang_left -= ang
            anchor = T
        if not bounded:
            self._buf = keep[-1:]
            return "lead"

        cmds = [self._cmd(P) for P in bounded]
        self._call(self.robot.moveAppend, cmds, self.sdk.PyString("cart_sdk"))
        self.stats["appended"] += len(cmds)
        self.stats["flushes"] += 1
        self.stats["batch_max"] = max(self.stats["batch_max"], len(cmds))
        prev = self._last_T
        for P in bounded:
            self._busy_until = max(self._busy_until, now) + self.segment_seconds(
                float(np.linalg.norm(P[:3, 3] - prev[:3, 3])))
            prev = P
        self._last_T = bounded[-1]
        self._pending = True
        return f"批{len(cmds)}" + ("+" + self._try_start() if idle else "")

    def _remaining(self, cur: np.ndarray) -> int:
        """已发的点里臂还有几个没走到：找离臂当前位姿最近的那个，它和它后面的都算没走完
        （偏保守：转弯区混合时臂只从点旁边擦过，最近的那个可能其实已经过了）。"""
        if not self._inflight:
            return 0
        cost = [float(np.linalg.norm(P[:3, 3] - cur[:3, 3]))
                + self.ROT_WEIGHT_M * _rot_angle(P[:3, :3] @ cur[:3, :3].T) for P in self._inflight]
        j = int(np.argmin(cost))
        del self._inflight[:j]
        return len(self._inflight)

    def _flush_batch(self, idle: bool, cur: np.ndarray) -> str:
        """批量模式：攒满 batch 个点一次 moveAppend，队列里剩不到 REFILL_AT 个才续发。

        **闭环**：剩几个点是拿臂的实测位姿对着已发的点数出来的，不是估算 —— 逐条模式那个按时间
        估算的积压闸门 10-31 真机估得偏乐观，队列照样积到 9~22 条。
        攒不过来时只取最新的 batch 个，更早的丢掉（发的是绝对位姿，丢了不会累积偏差，
        只是那一段路径被一条长一点的线段取代）。臂停下（队列空）时不满一批也发，免得最后一截不走。
        """
        if idle:
            self._inflight = []
        remaining = self._remaining(cur)
        if remaining > self.refill_at:
            self.stats["skipped_queue"] += 1
            return "queue"
        if not self._buf or (len(self._buf) < self.batch and not idle):
            return self._try_start() if (self._pending and idle) else "wait"
        if not self._changed(self._buf[-1], self._last_T):
            # 整批的终点离上一个已发的点还不到死区：手没怎么动，先不发，接着攒
            return self._try_start() if (self._pending and idle) else "tiny"

        pts, ref = [], self._last_T
        for T in self._buf[-self.batch:]:
            if (np.linalg.norm(T[:3, 3] - ref[:3, 3]) > self.DEDUP_M
                    or _rot_angle(T[:3, :3] @ ref[:3, :3].T) > self.DEDUP_RAD):
                pts.append(T)
                ref = T
        self.stats["decimated"] += max(0, len(self._buf) - self.batch)
        self.stats["deduped"] += min(len(self._buf), self.batch) - len(pts)
        self._buf = []
        if not pts:
            return "tiny"
        if self.merge_m > 0 and len(pts) > 1:
            kept = self._simplify(self._last_T, pts)
            self.stats["merged"] += len(pts) - len(kept)
            pts = kept

        cmds = [self._cmd(P) for P in pts]
        self._call(self.robot.moveAppend, cmds, self.sdk.PyString("cart_sdk"))
        self.stats["appended"] += len(cmds)
        self.stats["flushes"] += 1
        self.stats["batch_sum"] += len(cmds)
        self.stats["batch_max"] = max(self.stats["batch_max"], len(cmds))
        self._inflight += pts
        self._last_T = pts[-1]
        self._pending = True
        return f"批{len(cmds)}" + ("+" + self._try_start() if idle else "")

    def _simplify(self, anchor: np.ndarray, pts: list) -> list:
        """道格拉斯-普克：从 anchor 出发经过 pts 的折线，删掉偏离「首尾连线」不到 merge_m、
        姿态偏离「首尾均匀插值」不到 merge_rad 的点。最后一个点永远留着。"""
        chain = [anchor] + list(pts)
        keep = {len(chain) - 1}

        def split(i, j):
            if j - i < 2:
                return
            a, b = chain[i][:3, 3], chain[j][:3, 3]
            ab = b - a
            L2 = float(ab @ ab)
            rel = matrix_to_axis_angle(chain[j][:3, :3] @ chain[i][:3, :3].T)
            worst, k_worst = 0.0, None
            for k in range(i + 1, j):
                p = chain[k][:3, 3]
                s = float(np.clip((p - a) @ ab / L2, 0.0, 1.0)) if L2 > 1e-12 else (k - i) / (j - i)
                dev_p = float(np.linalg.norm(p - (a + s * ab)))
                R_lin = axis_angle_to_matrix(rel * s) @ chain[i][:3, :3]
                dev_r = _rot_angle(chain[k][:3, :3] @ R_lin.T)
                # 两种偏离各按自己的容差折成「超了几倍」，取大的
                score = max(dev_p / max(self.merge_m, 1e-9), dev_r / max(self.merge_rad, 1e-9))
                if score > worst:
                    worst, k_worst = score, k
            if worst > 1.0:
                keep.add(k_worst)
                split(i, k_worst)
                split(k_worst, j)

        split(0, len(chain) - 1)
        return [chain[k] for k in sorted(keep)]

    @classmethod
    def segment_seconds(cls, length_m: float) -> float:
        """估一条 MoveL 要走多久：加速度受限的 2√(L/a)，再短也不少于 MIN_SEGMENT_S。偏保守，
        转弯区混合会让相邻段重叠、实际更快 —— 估多了只是少发几条，估少了才会积压。"""
        return max(cls.MIN_SEGMENT_S, 2.0 * np.sqrt(max(length_m, 0.0) / cls.SEGMENT_ACCEL))

    def _cmd(self, T: np.ndarray):
        cp = self.sdk.CartesianPosition()
        cp.trans = T[:3, 3].tolist()
        cp.rpy = matrix_to_rpy(T[:3, :3]).tolist()
        if self.elbow != "none" and self._conf is not None:
            cp.elbow, cp.hasElbow, cp.confData = self._elbow, True, list(self._conf)
        cmd = self.sdk.MoveLCommand(cp)
        cmd.speed = self.speed
        cmd.zone = self.zone
        return cmd

    def _try_start(self) -> str:
        """moveStart 的返回码只有两个是良性的：-20 运动中（追加的点会被接上执行）、
        768 队列空。它**不抛异常、只回 ec**，所以这里逐个认。"""
        ec = {}
        self.robot.moveStart(ec)
        code = ec.get("ec", 0)
        if code == -20:
            self.stats["already_moving"] += 1
            return "已在动"
        if code == 768:
            self._pending = False
            self.stats["queue_empty"] += 1
            return "队列空"
        if code:
            self.stats["start_failed"] += 1
            raise RuntimeError(f"moveStart 失败：{ec}")
        self._pending = False
        self.stats["restarts"] += 1
        return "启动"

    def report(self) -> str:
        s = self.stats
        batch = (f"一次最多 {self.batch} 点、平均每批 {s['batch_sum'] / max(1, s['flushes']):.1f} 点，"
                 f"队列满没发 {s['skipped_queue']} 拍，去重 {s['deduped']}，共线合并掉 {s['merged']}，"
                 if self.batch > 1 else f"积压挡掉 {s['skipped_backlog']}，")
        return (f"MoveL：收到 {s['samples']} 个目标，死区滤掉 {s['skipped_tiny']}，前导挡掉 {s['skipped_lead']}，"
                f"{batch}只发最新丢掉 {s['decimated']}，"
                f"下发 {s['appended']} 点 / {s['flushes']} 批（最大 {s['batch_max']}）；"
                f"启动 {s['restarts']} 次（每次都是队列走空后重启，≈ 臂停下再起步）；"
                f"启动失败 {s['start_failed']}，执行错误 {s['execution_errors']}")
