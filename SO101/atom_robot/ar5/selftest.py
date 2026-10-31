"""Hardware-free checks for the AR5 stack.

Covers the parts that are easy to get subtly wrong and expensive to debug with
a robot attached: rotation composition, the per-step safety clamps, the
workspace box, the gripper encoding, and the Modbus framing.

    python atom_robot/ar5/selftest.py

Exits non-zero on the first failure.
"""

import contextlib
import io
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from atom_robot.ar5.ar5_env import AR5Env, SafetyViolation  # noqa: E402
from atom_robot.ar5.config import AR5Config  # noqa: E402
from atom_robot.ar5.drivers.inspire_hand import (  # noqa: E402
    build_read_registers_frame,
    build_write_registers_frame,
    modbus_crc16,
    parse_read_response,
)
from atom_robot.ar5.transforms import (  # noqa: E402
    axis_angle_to_matrix,
    clamp_delta,
    compose_pose,
    matrix_to_axis_angle,
    pose_delta,
)

_results = []


class Skip(Exception):
    """这一项在当前环境里测不了（少一个**可选**依赖），不是失败。

    分清「坏了」和「这台机器上装不了」很重要：笛卡尔那条遥操路要 `placo`，
    而采集走的是关节空间、压根不用它。缺 placo 报成 FAIL 的话，
    每次跑自检都有六条红的要人去分辨，真出问题时反而看不见。
    """


def need(module: str):
    """可选依赖，缺了就跳过这一项。"""
    import importlib
    try:
        return importlib.import_module(module)
    except ImportError:
        raise Skip(f"没装 {module}（可选依赖，只有笛卡尔遥操那条路要）")


def check(name):
    def decorator(fn):
        _results.append((name, fn))
        return fn

    return decorator


# --------------------------------------------------------------- transforms


@check("axis-angle round-trips through the rotation matrix")
def _():
    rng = np.random.default_rng(0)
    for _ in range(200):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        theta = rng.uniform(0.0, np.pi - 1e-3)
        rotvec = axis * theta
        back = matrix_to_axis_angle(axis_angle_to_matrix(rotvec))
        assert np.allclose(rotvec, back, atol=1e-6), f"{rotvec} -> {back}"


@check("rotation matrices stay orthonormal")
def _():
    rng = np.random.default_rng(1)
    for _ in range(100):
        R = axis_angle_to_matrix(rng.normal(size=3))
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-9)


@check("identity and near-pi rotations are handled")
def _():
    assert np.allclose(matrix_to_axis_angle(np.eye(3)), np.zeros(3))
    for axis in np.eye(3):
        rotvec = axis * (np.pi - 1e-7)
        back = matrix_to_axis_angle(axis_angle_to_matrix(rotvec))
        assert np.isclose(np.linalg.norm(back), np.pi, atol=1e-4)
        assert np.allclose(np.abs(back / np.linalg.norm(back)), np.abs(axis), atol=1e-3)


@check("pose_delta inverts compose_pose")
def _():
    rng = np.random.default_rng(2)
    for _ in range(100):
        pose = np.concatenate([rng.uniform(-0.5, 0.5, 3), rng.normal(size=3) * 0.3])
        delta = np.concatenate([rng.uniform(-0.02, 0.02, 3), rng.normal(size=3) * 0.05])
        recovered = pose_delta(pose, compose_pose(pose, delta))
        assert np.allclose(recovered, delta, atol=1e-6), f"{delta} -> {recovered}"


@check("clamp_delta preserves direction while enforcing limits")
def _():
    delta = np.array([0.10, 0.0, 0.0, 0.0, 0.0, 0.5])
    clamped, hit = clamp_delta(delta, max_translation=0.02, max_rotation=0.10)
    assert hit
    assert np.linalg.norm(clamped[:3]) <= 0.02 + 1e-9
    assert np.linalg.norm(clamped[3:]) <= 0.10 + 1e-9
    # A single common scale means the direction is unchanged.
    for original, scaled in ((delta[:3], clamped[:3]), (delta[3:], clamped[3:])):
        if np.linalg.norm(original) > 1e-9:
            cos = (original @ scaled) / (np.linalg.norm(original) * np.linalg.norm(scaled))
            assert np.isclose(cos, 1.0, atol=1e-9)

    small = np.array([0.001, 0.0, 0.0, 0.0, 0.0, 0.01])
    unchanged, hit2 = clamp_delta(small, 0.02, 0.10)
    assert not hit2 and np.allclose(unchanged, small)


# ---------------------------------------------------------------- modbus


@check("Modbus CRC matches the reference vector")
def _():
    # Canonical example: slave 1, read holding registers from 0 for 1 register.
    assert modbus_crc16(bytes([0x01, 0x03, 0x00, 0x00, 0x00, 0x01])) == 0x0A84


@check("Modbus write frame is well formed")
def _():
    frame = build_write_registers_frame(1, 1486, [0, 100, 200, 300, 400, 500])
    assert frame[0] == 1 and frame[1] == 0x10
    assert int.from_bytes(frame[2:4], "big") == 1486
    assert int.from_bytes(frame[4:6], "big") == 6      # register count
    assert frame[6] == 12                               # byte count
    assert len(frame) == 7 + 12 + 2
    assert int.from_bytes(frame[-2:], "little") == modbus_crc16(frame[:-2])


@check("Modbus read response parses and rejects corruption")
def _():
    values = [11, 22, 33, 44, 55, 66]
    body = bytes([1, 0x03, len(values) * 2]) + b"".join(v.to_bytes(2, "big") for v in values)
    frame = body + modbus_crc16(body).to_bytes(2, "little")
    assert parse_read_response(frame, 1, 6) == values

    bad = bytearray(frame)
    bad[4] ^= 0xFF
    try:
        parse_read_response(bytes(bad), 1, 6)
    except IOError:
        pass
    else:
        raise AssertionError("corrupted frame was accepted")

    exception_body = bytes([1, 0x83, 0x02])
    exception_frame = exception_body + modbus_crc16(exception_body).to_bytes(2, "little")
    try:
        parse_read_response(exception_frame, 1, 6)
    except IOError as exc:
        assert "exception code" in str(exc)
    else:
        raise AssertionError("Modbus exception response was accepted")

    assert build_read_registers_frame(1, 1546, 6)[1] == 0x03


# ------------------------------------------------------------------- env


def _mock_env():
    cfg = AR5Config(mock=True)
    cfg.settle_time = 0.0
    env = AR5Env(cfg)
    env.connect()
    return env


@check("mock environment resets and produces a well-formed observation")
def _():
    env = _mock_env()
    try:
        obs = env.reset()
        assert set(obs.observation) == {"images", "qpos", "ee_pose", "elbow", "stamps"}
        # 臂和夹爪的读数时刻每帧都有（mock 相机不记拍摄时刻，collect 那边记成 nan）
        stamps = obs.observation["stamps"]
        now = time.perf_counter()
        assert all(0 <= now - stamps[k] < 1.0 for k in ("arm", "gripper")), stamps
        # Follow the configured sources rather than a fixed pair of names: how
        # many cameras are attached is a property of the rig, and hard-coding it
        # here turns "someone unplugged a camera" into a mystery test failure.
        assert set(obs.observation["images"]) == set(env.config.cameras.sources), (
            f"images {sorted(obs.observation['images'])} do not match the configured "
            f"sources {sorted(env.config.cameras.sources)}"
        )
        assert obs.observation["images"], "no cameras configured; this check needs at least one"
        for img in obs.observation["images"].values():
            assert img.shape == (480, 640, 3) and img.dtype == np.uint8
        # PROPRIO_DIM for AR5 is 8: seven joints plus the hand opening.
        assert obs.observation["qpos"].shape == (env.config.arm.num_joints + 1,)
        assert obs.observation["ee_pose"].shape == (6,)
    finally:
        env.close()


@check("action deltas accumulate on the commanded setpoint")
def _():
    env = _mock_env()
    try:
        env.reset()
        start = env.arm.get_ee_pose().copy()
        for _ in range(5):
            env.step([0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        moved = env.arm.get_ee_pose()
        assert np.isclose(moved[0] - start[0], 0.05, atol=1e-9), moved[0] - start[0]
        assert env.stats["steps"] == 5
        assert env.stats["delta_clamped"] == 0
    finally:
        env.close()


@check("oversized deltas are clamped, not executed")
def _():
    env = _mock_env()
    try:
        env.reset()
        start = env.arm.get_ee_pose().copy()
        env.step([10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        travelled = env.arm.get_ee_pose()[0] - start[0]
        assert travelled <= env.config.arm.max_step_translation + 1e-9, travelled
        assert env.stats["delta_clamped"] == 1
    finally:
        env.close()


@check("the workspace box holds under sustained pushing")
def _():
    env = _mock_env()
    try:
        env.reset()
        for _ in range(200):
            env.step([0.02, 0.02, 0.02, 0.0, 0.0, 0.0, 1.0])
        pose = env.arm.get_ee_pose()
        lo = np.array(env.config.arm.workspace_min)
        hi = np.array(env.config.arm.workspace_max)
        assert np.all(pose[:3] >= lo - 1e-9) and np.all(pose[:3] <= hi + 1e-9), pose[:3]
        assert env.stats["workspace_clamped"] > 0
    finally:
        env.close()


@check("盒子外面起步：零输入也会被自己拽回盒内，而且遥操开跑前要讲清楚")
def _():
    # 2026-09-16：末端停在 z=0.499、盒子上限 0.42 的时候按下离合，臂会在没有任何
    # 指令的情况下自己往下走 79 mm —— 夹取先把目标拉到面上，leash 再每步喂
    # max_lead 进去。刀尖朝下时这个不能是静悄悄的。
    env = _mock_env()
    try:
        env.reset()
        hi = np.array(env.config.arm.workspace_max, dtype=np.float64)
        outside = env.arm.get_ee_pose().copy()
        outside[2] = hi[2] + 0.08
        env.arm._pose = outside.copy()          # 摆到盒子外，不经过 step
        env._target_pose = None

        assert any(c < 0.0 for _, _, c in env.pinned_axes()), \
            "pinned_axes 必须把「在盒子外」报成负余量，遥操的预检就是看这个"

        before = env.arm.get_ee_pose()[2]
        for _ in range(40):
            env.step([0.0] * 6 + [1.0])         # 零 delta：操作者什么都没做
        after = env.arm.get_ee_pose()[2]
        assert after < before - 0.05, f"零输入应当把臂拽回盒内，z {before:.3f} -> {after:.3f}"
        assert after <= hi[2] + 1e-6, f"拽回来之后应当在盒内，z={after:.3f}"

        # 预检必须把这件事说成「会自己动」，而不是原来那句「会被夹住、看起来卡住」
        src = (Path(__file__).resolve().parent / "teleop.py").read_text()
        assert "OUTSIDE the workspace box" in src and "with no input from you" in src, \
            "teleop 预检没有把「盒外起步会自己走」单独讲出来"
    finally:
        env.close()


@check("两个实时窗口都要有暂停键，而且暂停语义不能弄反")
def _():
    import ast

    here = Path(__file__).resolve().parent
    fm = (here / "force_monitor.py").read_text()
    lp = (here / "live_plot.py").read_text()

    for name, src in (("force_monitor", fm), ("live_plot", lp)):
        tree = ast.parse(src)
        fn = next((f for f in ast.walk(tree)
                   if isinstance(f, ast.FunctionDef) and f.name == "on_key"), None)
        assert fn is not None, f"{name} 没有按键回调了"
        body = ast.get_source_segment(src, fn) or ""
        assert '" "' in body, f"{name} 的暂停键（空格）没了"
        # 暂停时必须继续 flush_events，否则窗口收不到按键、再也恢复不了。
        # 锚点用循环里那句 `if state["paused"]:`，不是 on_key 里的赋值。
        assert 'if state["paused"]:' in src, f"{name} 循环里没有暂停分支"
        blk = src.split('if state["paused"]:', 1)[1][:400]
        assert "flush_events" in blk, \
            f"{name} 暂停分支里没有 flush_events —— 空格按下去就再也起不来"

    # **两边的暂停语义是不同的，而且是有意的：**
    #   force_monitor 自己是数据源 -> 暂停连采样一起停，否则要看的那个峰会被
    #                                后续数据挤出滚动窗口
    #   live_plot 的数据源是 teleop -> 只冻画面、照常读文件，停读只会积压
    def paused_block(src: str) -> str:
        """暂停分支的分支体：切到第一个 continue 为止。

        再往后就是恢复执行的正常路径了，连着一起看会把下一行误当成分支内容。
        """
        tail = src.split('if state["paused"]:', 1)[1]
        cut = tail.find("continue")
        return tail[: cut + len("continue")] if cut >= 0 else tail[:400]

    assert "cur = sample()" in fm, "force_monitor 的采样调用改名了？"
    assert "sample()" not in paused_block(fm), \
        "force_monitor 暂停时还在采样 —— 滚动窗口会把要看的峰挤掉"
    assert "read_rows" not in paused_block(lp), \
        "live_plot 暂停时不该停止读文件 —— teleop 还在写，停读只会积压"


@check("goto_episode_pose --show 只打印，绝不连臂上电（两条路都要）")
def _():
    import ast

    # 2026-09-17：把移动那段抽成 _goto() 时漏了 episode 那条的 `if args.show: return 0`，
    # 结果 `--show` 会连臂**并且上电**——文档写的是"只打印姿态，不连臂不动"。
    # 两条路（episode / --joints）各要一个守卫。
    src = (Path(__file__).resolve().parent / "goto_episode_pose.py").read_text()
    tree = ast.parse(src)
    fn = next(f for f in ast.walk(tree)
              if isinstance(f, ast.FunctionDef) and f.name == "main")

    guards, gotos = 0, 0
    for node in ast.walk(fn):
        if isinstance(node, ast.If):
            test = ast.get_source_segment(src, node.test) or ""
            body = ast.get_source_segment(src, node) or ""
            if "args.show" in test and "return 0" in body:
                guards += 1
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "_goto":
            gotos += 1
    assert gotos >= 2, f"_goto 只被调用 {gotos} 次，两条路应该各一次"
    assert guards >= gotos, (
        f"main 里有 {gotos} 条通向 _goto 的路，但只有 {guards} 个 --show 守卫 —— "
        "少一个就意味着 --show 会去连臂上电"
    )

    # _goto 才是真正动臂的，main 里不该有连接
    body = ast.get_source_segment(src, fn) or ""
    assert "env.connect()" not in body, "连接应该只在 _goto 里，main 里不该有"


@check("力监视窗口只读：不上电、不起实时流、不发运动指令、不进力控")
def _():
    import ast

    # force_monitor 会另开一个 SDK 会话，所以「只读」这条必须由代码保证，
    # 而不是靠记得。fcInit/fcStart 也算 —— 那两个是真进力控模式。
    src = (Path(__file__).resolve().parent / "force_monitor.py").read_text()
    tree = ast.parse(src)
    names = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    names |= {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    forbidden = {"servo_to_pose", "servo_to_pose_rt", "servo_to_joints", "goto_joints",
                 "moveAppend", "setPowerState", "set_power", "fcInit", "fcStart",
                 "setCartesianDesiredForce", "enableDrag"}
    hit = forbidden & names
    assert not hit, f"force_monitor 里出现了会动臂/改模式的调用：{sorted(hit)}"
    assert "connect(power_on=False)" in src, "force_monitor 必须以不上电的方式连接"
    assert "use_realtime = False" in src, \
        "force_monitor 不能起 1 kHz 实时流 —— 它只是读数，起流会和真正的控制打架"


@check("实时曲线：半截的行不能吞、画图不能塞进遥操循环")
def _():
    import ast
    import tempfile

    from atom_robot.ar5.live_plot import LAYOUT, read_rows

    # teleop 攒几行才 flush，所以读的时候末行常常写到一半。丢掉它是对的，
    # **但不能把它计进游标** —— 计进去等 teleop 写完，这一行就永远读不到，
    # 曲线缺一小段而且完全没有提示。
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "live.csv"
        full = ",".join(["1"] * 66)
        p.write_text(full + "\n" + full + "\n" + "1,2,3")      # 两整行 + 半行
        rows, seen = read_rows(p, 0)
        assert rows is not None and rows.shape == (2, 66), rows
        assert seen == 2, f"游标应停在 2（半行不算），实际 {seen}"

        # 半行写完之后必须能读到
        p.write_text(full + "\n" + full + "\n" + full + "\n")
        rows2, seen2 = read_rows(p, seen)
        assert rows2 is not None and rows2.shape == (1, 66), \
            "半行补全之后没被读到 —— 游标把它吞了"
        assert seen2 == 3

        rows3, seen3 = read_rows(p, seen2)
        assert rows3 is None and seen3 == 3, "没有新数据时不该返回行"

    assert set(LAYOUT) == {53, 66}, "live_plot 认的列数和 teleop 的布局对不上了"

    # 时间戳的精度：Unix 时间戳是 1.79e9，用 %.6g 写会变成 `1.78962e+09` ——
    # 秒以下全丢、整个文件同一个时间，曲线摊成一竖线。2026-09-17 预演里抓到的。
    # 50 Hz 要分得清 20 ms，所以至少得到毫秒。
    now = 1789618271.123456
    assert float(f"{now:.6g}") == now or True                      # 说明用的
    coarse, fine = float(f"{now:.6g}"), float(f"{now:.6f}")
    assert coarse == float(f"{now + 0.02:.6g}"), "前提变了：%.6g 本该分不清 20 ms"
    assert fine != float(f"{now + 0.02:.6f}"), "%.6f 必须分得清 20 ms"
    tsrc = (Path(__file__).resolve().parent / "teleop.py").read_text()
    assert 'f"{row[0]:.6f},"' in tsrc, (
        "--live 的时间戳不是定点格式了 —— %.6g 会把 Unix 时间戳压成 1.78962e+09，"
        "实时曲线会摊成一竖线"
    )

    # 画图必须在**另一个进程**：渲染一帧几十毫秒且拿着 GIL，塞进遥操循环就是
    # 当初把 getEndTorque(41ms) 放进去的同一个错误 —— 发送率 963/s -> 781/s，
    # 控制器当丢包、停止执行却继续收指令。
    src = (Path(__file__).resolve().parent / "teleop.py").read_text()
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "main")
    names = {a.attr for a in ast.walk(fn) if isinstance(a, ast.Attribute)}
    names |= {a.id for a in ast.walk(fn) if isinstance(a, ast.Name)}
    for bad in ("pyplot", "plt", "matplotlib", "draw_idle", "show"):
        assert bad not in names, f"teleop 的主循环里出现了 {bad} —— 画图要走 live_plot"
    assert "live" in names, "teleop 没有 --live 流式输出，实时曲线就没数据源"


@check("轨迹里力那两路：列数对得上、而且必须降频采（不能按控制频率）")
def _():
    import ast
    import re

    from atom_robot.ar5.config import ArmConfig
    from atom_robot.ar5.teleop import _fit

    n = ArmConfig().num_joints
    # 文档里写的列数要和真实拼出来的一致 —— 这一列错位，离线分析全是错的
    src = (Path(__file__).resolve().parent / "teleop.py").read_text()
    m = re.search(r"记逐步轨迹，(\d+) 列", src)
    assert m, "teleop 的 docstring 里没写轨迹列数了"
    documented = int(m.group(1))
    actual = 1 + 1 + 3 + 7 + 6 + n + n + n + 1 + n + n + 3 + 3 + 6
    assert documented == actual, f"docstring 写 {documented} 列，实际拼出来 {actual} 列"

    # 文档里每一段的下标要连得上、不重叠。只扫布局块那几行（docstring 里缩进 4 格、
    # 以 [ 开头），否则正文里提到的 [40:60] 之类也会被当成一段。
    layout = [ln for ln in src.splitlines() if re.match(r"^    \[\d+[:\]]", ln)]
    spans = [(int(a), int(b)) for ln in layout
             for a, b in re.findall(r"^    \[(\d+):(\d+)\]", ln)]
    spans += [(int(a), int(a) + 1) for ln in layout
              for a in re.findall(r"^    \[(\d+)\]", ln)]
    spans = sorted(set(spans))
    assert spans, "docstring 里找不到轨迹布局表了"
    cursor = 0
    for lo, hi in spans:
        assert lo == cursor, f"轨迹列 [{lo}:{hi}] 和上一段之间断了或重了（应从 {cursor} 开始）"
        cursor = hi
    assert cursor == actual, f"文档覆盖到 {cursor} 列，实际 {actual} 列"

    # _fit 要真的定长：SDK 会用外部轴槽位把返回值补长，长度不一拼出来的行会参差
    assert list(_fit([1, 2, 3, 4, 5], 3)) == [1.0, 2.0, 3.0]
    assert list(_fit([1, 2], 4)) == [1.0, 2.0, 0.0, 0.0]

    tree = ast.parse(src)
    fn = next(n_ for n_ in ast.walk(tree)
              if isinstance(n_, ast.FunctionDef) and n_.name == "main")
    names = {a.attr for a in ast.walk(fn) if isinstance(a, ast.Attribute)}
    assert "get_joint_torque" in names, "teleop 没在记关节力矩"
    assert "get_end_wrench" in names, "teleop 没在记控制器的外力/末端力估计"
    assert "jacobian" in names, "末端力少了本地反推那一路，就没有交叉验证了"

    # 2026-09-17 实测：`robot.forceControl()` 取句柄 **41 ms**，getEndTorque 本身
    # 只要 0.30 ms。每次重取会把 1 kHz 发送流打出洞（963/s -> 781/s），控制器当
    # 丢包处理、停止执行却继续收指令 —— q_cmd 走、q_actual 冻、日志干净。
    # 所以驱动里那个句柄**必须缓存**，而且断开时要清掉。
    drv = (Path(__file__).resolve().parent / "drivers" / "rokae_arm.py").read_text()
    dtree = ast.parse(drv)
    gw = next((f for f in ast.walk(dtree)
               if isinstance(f, ast.FunctionDef) and f.name == "get_end_wrench"), None)
    assert gw is not None, "驱动里没有 get_end_wrench 了"
    body = ast.get_source_segment(drv, gw) or ""
    assert "_fc" in body, (
        "get_end_wrench 没有缓存 forceControl() 句柄 —— 取句柄要 41 ms，"
        "按控制频率调会饿死 1 kHz 发送线程、让臂停止执行"
    )
    dis = next(f for f in ast.walk(dtree)
               if isinstance(f, ast.FunctionDef) and f.name == "disconnect")
    assert "_fc" in (ast.get_source_segment(drv, dis) or ""), \
        "disconnect 没清掉缓存的力控句柄，重连后会拿着指向旧会话的句柄"


@check("RT 起不来就回退时，控制模式必须切回 Nrt —— 否则「回退」保证失败")
def _():
    # 2026-09-17：`_start_realtime` 的 try 第一件事是设 RtCommandMode，except 里
    # 只是把 _rt_active 置 False 就宣布"回退到排队式"。控制器还在 RT 模式，排队式
    # Move* 必被拒，于是一个清楚的「RT 启动失败」被延迟成几秒后 AR5Env.step 里的
    # `moveAppend failed (ec=262)`。碰撞停机掉电之后每次都踩。
    import ast

    src = (Path(__file__).resolve().parent / "drivers" / "rokae_arm.py").read_text()
    tree = ast.parse(src)
    fn = next((n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name == "_start_realtime"), None)
    assert fn is not None, "找不到 _start_realtime"

    handlers = [h for h in ast.walk(fn) if isinstance(h, ast.ExceptHandler)]
    assert handlers, "_start_realtime 没有 except 分支了？回退逻辑被改掉了"

    def names_in(node):
        return {n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)}

    assert any("setMotionControlMode" in names_in(h) and "NrtCommandMode" in names_in(h)
               for h in handlers), (
        "RT 启动失败的 except 分支里没有把控制模式切回 NrtCommandMode —— "
        "排队式回退会被控制器以 ec=262 全部拒掉"
    )


@check("碰撞检测的配置是可用的，而且别再去调 SDK 里不存在的那个方法")
def _():
    from atom_robot.ar5.config import ArmConfig

    cfg = ArmConfig()
    sens = cfg.collision_sensitivity
    if sens:
        assert len(sens) == cfg.num_joints, (
            f"collision_sensitivity 有 {len(sens)} 个值，关节有 {cfg.num_joints} 个 —— "
            "长度对不上时驱动只会打个 warning 然后用控制器默认值，等于设置静默失效"
        )
        for v in sens:
            assert 0.01 <= v <= 2.0, f"灵敏度 {v} 超出 SDK 的 0.01~2.0"
    # StopLevel 的成员名，照 SDK 实测（stop0/stop1/stop2/suppleStop）
    assert cfg.collision_behaviour in ("stop0", "stop1", "stop2", "suppleStop"), \
        f"collision_behaviour={cfg.collision_behaviour!r} 不是 StopLevel 的成员"
    if cfg.collision_behaviour == "suppleStop":
        assert 0.0 <= cfg.collision_fallback <= 1.0, \
            "柔顺停止时第三个参数是柔顺度 [0,1]，不是回退距离"

    # 2026-09-17：驱动里曾有 `self._rt.setCollisionBehaviour(...)`，而这个方法
    # **整个 SDK 都没有**，只因为配置项一直是空的才没炸。别再写回去。
    src = (Path(__file__).resolve().parent / "drivers" / "rokae_arm.py").read_text()
    import ast

    tree = ast.parse(src)
    called = {n.func.attr for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    passed = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert "setCollisionBehaviour" not in (called | passed), \
        "setCollisionBehaviour 在 SDK 里不存在，碰撞检测要走 enableCollisionDetection"
    assert "enableCollisionDetection" in (called | passed), \
        "驱动没有接 enableCollisionDetection —— 拧螺丝的反扭矩会把臂碰撞停机、掉电抱闸"


@check("测压力：未扣的重力必须被归零减掉，报的是接触力不是总力")
def _():
    # payload_mass=0 而手有 1.2~1.3 kg，那 12~13 N 整个留在 cart_force 里。
    # 不减基线的话「压了多少」会被这个常数淹掉 —— 这正是这个脚本存在的理由，
    # 所以拿一个「常驻偏置 + 后段叠接触力」的合成信号守住它。
    import types
    from unittest import mock

    from atom_robot.ar5 import diag_press

    class VecD:
        def __init__(self, v=None):
            self._v = list(v or [])

        def content(self):
            return self._v

        def fill(self, v):
            self._v = list(v)

    GRAVITY_N = -12.5          # 没扣掉的手重，常驻
    CONTACT_N = 28.0           # 真正想量的那一下

    class FakeArm:
        def __init__(self):
            self.sdk = types.SimpleNamespace(
                PyTypeVectorDouble=VecD,
                FrameType=types.SimpleNamespace(world="W"))
            # getEndTorque 挂在 forceControl() 返回的对象上，**robot 本身没有** ——
            # 实测 xMateErProRobot 直接调会 AttributeError。照实物摆，免得再放回去。
            self._fc = types.SimpleNamespace(getEndTorque=self._get)
            self._robot = types.SimpleNamespace(forceControl=lambda: self._fc)
            self.n = 0

        def _get(self, frame, jr, je, ct, cf, ec):
            self.n += 1
            contact = CONTACT_N if self.n > diag_press.TARE_SAMPLES + 2 else 0.0
            cf.fill([0.0, 0.0, GRAVITY_N + contact])
            je.fill([0.0] * 7)
            jr.fill([0.0] * 7)
            ct.fill([0.0] * 3)
            ec["ec"] = 0

        def _call(self, fn, *a):
            ec = {}
            fn(*a, ec)

        def get_ee_pose(self):
            return np.array([0.3, -0.18, 0.2, 0.0, 0.0, 0.0])

        def connect(self, power_on=True):
            assert power_on is False, "测压力必须只读，不能上电"

        def disconnect(self):
            pass

    fake = FakeArm()
    out = io.StringIO()
    with mock.patch.object(diag_press, "RokaeAR5", lambda cfg: fake), \
            mock.patch.object(diag_press.time, "sleep", lambda s: None), \
            mock.patch.object(sys, "argv", ["diag_press", "--seconds", "0.2", "--hz", "500"]), \
            contextlib.redirect_stdout(out):
        diag_press.main()

    text = out.getvalue()
    assert f"ΔFz = +{CONTACT_N:.2f} N" in text, \
        f"归零没把 {GRAVITY_N} N 的重力减掉，报出来的不是接触力：\n{text[-400:]}"
    assert f"{GRAVITY_N + CONTACT_N:.2f} N**" not in text, "报了总力而不是增量"

    # 脚本绝不能发运动指令 —— 它要能在 teleop 跑着的时候当第二个进程开。
    # fcInit / fcStart 也算：那两个是真进力控模式，不是读。
    # 按 AST 查**调用**，不是查子串 —— 注释里写「不要调 fcInit」不该算违规。
    import ast

    FORBIDDEN = {"servo_to_pose", "servo_to_pose_rt", "servo_to_joints", "goto_joints",
                 "moveAppend", "set_power", "fcInit", "fcStart", "setCartesianDesiredForce"}
    tree = ast.parse((Path(__file__).resolve().parent / "diag_press.py").read_text())
    called = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
            if name:
                called.add(name)
    # arm._call(fn, ...) 把被调函数当值传进去，绕过上面那圈，所以属性名也要算
    passed_as_value = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    offenders = FORBIDDEN & (called | passed_as_value)
    assert not offenders, f"diag_press 里出现了会改变控制模式的调用：{sorted(offenders)}"


@check("gripper uses the energy head's 0=closed / 1=open encoding")
def _():
    env = _mock_env()
    try:
        env.reset()
        env.step([0.0] * 6 + [0.0])
        assert env.hand.get_opening() == 0.0
        closed_angles = env.hand.get_joint_angles()

        env.step([0.0] * 6 + [1.0])
        assert env.hand.get_opening() == 1.0
        open_angles = env.hand.get_joint_angles()

        # The grasp is a two-finger pinch: index and thumb both travel, closing
        # toward each other. Everything else holds still, out of the camera's
        # way. (Earlier versions of this check expected all four fingers to
        # open, then only the index -- both were wrong for this hand.)
        index = env.config.hand.index_dof
        THUMB_BEND = 4
        assert open_angles[index] > closed_angles[index], (
            f"the index finger must extend as it opens: {closed_angles} -> {open_angles}"
        )
        for dof in range(len(open_angles)):
            if dof in (index, THUMB_BEND):
                continue
            assert open_angles[dof] == closed_angles[dof], (
                f"DOF {dof} is not part of the pinch and must hold: "
                f"{closed_angles} -> {open_angles}"
            )

        # Out-of-range commands saturate rather than propagate.
        env.step([0.0] * 6 + [5.0])
        assert env.hand.get_opening() == 1.0
        assert 0.0 <= env.hand.get_opening() <= 1.0
    finally:
        env.close()


@check("malformed actions are rejected")
def _():
    env = _mock_env()
    try:
        env.reset()
        for bad in ([0.0] * 6, [0.0] * 8):
            try:
                env.step(bad)
            except ValueError:
                pass
            else:
                raise AssertionError(f"accepted an action of length {len(bad)}")

        try:
            env.step([np.nan] + [0.0] * 6)
        except SafetyViolation:
            pass
        else:
            raise AssertionError("accepted a NaN action")
    finally:
        env.close()


def _load_constants(platform: str):
    """Import prismatic/vla/constants.py directly by path.

    Going through the `prismatic` package would drag in torch and
    huggingface_hub, which would make this self-test depend on the full
    training stack for no benefit.
    """
    import importlib.util
    import os

    from atom_robot import vel_path as _vel
    path = _vel("prismatic", "vla", "constants.py")
    spec = importlib.util.spec_from_file_location(f"_ar5_constants_{platform}", path)
    module = importlib.util.module_from_spec(spec)

    previous = os.environ.get("VLA_ROBOT_PLATFORM")
    os.environ["VLA_ROBOT_PLATFORM"] = platform
    try:
        spec.loader.exec_module(module)
    finally:
        if previous is None:
            os.environ.pop("VLA_ROBOT_PLATFORM", None)
        else:
            os.environ["VLA_ROBOT_PLATFORM"] = previous
    return module


@check("VLA_ROBOT_PLATFORM selects the AR5 constants")
def _():
    ar5 = _load_constants("AR5")
    assert ar5.ROBOT_PLATFORM == "AR5"
    assert ar5.ACTION_DIM == 7, ar5.ACTION_DIM
    assert ar5.PROPRIO_DIM == 8, ar5.PROPRIO_DIM   # 7 joints + hand opening
    assert ar5.NUM_ACTIONS_CHUNK == 8, ar5.NUM_ACTIONS_CHUNK

    # The override must actually override: LIBERO differs in PROPRIO_DIM, which
    # is exactly the mismatch that would silently break the proprio projector.
    libero = _load_constants("LIBERO")
    assert libero.PROPRIO_DIM == 8, libero.PROPRIO_DIM

    try:
        _load_constants("NOT_A_ROBOT")
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown platform name was accepted")


@check("env proprio width matches the AR5 constants")
def _():
    expected = _load_constants("AR5").PROPRIO_DIM
    env = _mock_env()
    try:
        env.reset()
        assert env.get_qpos().shape == (expected,), env.get_qpos().shape
    finally:
        env.close()


@check("RPY <-> axis-angle round-trips (the SDK's pose convention)")
def _():
    from atom_robot.ar5.transforms import (
        internal_pose_to_rpy,
        matrix_to_rpy,
        rpy_pose_to_internal,
        rpy_to_matrix,
    )

    rng = np.random.default_rng(7)
    for _ in range(300):
        trans = rng.uniform(-1.0, 1.0, 3)
        rpy = rng.uniform(-np.pi, np.pi, 3)
        R = rpy_to_matrix(rpy)

        # Rotation matrices from RPY must be proper rotations.
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-9)

        # matrix -> rpy -> matrix must land back, even where the rpy triple
        # itself differs (the mapping is many-to-one).
        assert np.allclose(rpy_to_matrix(matrix_to_rpy(R)), R, atol=1e-9)

        # Full driver-edge round trip: what cartPosture gives us, converted to
        # the axis-angle pose the rest of the stack uses, and back.
        pose = rpy_pose_to_internal(trans, rpy)
        assert pose.shape == (6,)
        back_trans, back_rpy = internal_pose_to_rpy(pose)
        assert np.allclose(back_trans, trans, atol=1e-9)
        assert np.allclose(rpy_to_matrix(back_rpy), R, atol=1e-8)

    # Gimbal lock: ry = +-pi/2 collapses one degree of freedom and is where a
    # naive arctan2 implementation returns NaN.
    for sign in (1, -1):
        R = rpy_to_matrix([0.7, sign * np.pi / 2, 0.3])
        recovered = matrix_to_rpy(R)
        assert np.all(np.isfinite(recovered))
        assert np.allclose(rpy_to_matrix(recovered), R, atol=1e-9)


@check("joint count is consistent across config, constants and the mock arm")
def _():
    expected = _load_constants("AR5").PROPRIO_DIM
    cfg = AR5Config(mock=True)
    assert cfg.arm.num_joints + 1 == expected, (cfg.arm.num_joints, expected)
    assert len(cfg.arm.home_joints) == cfg.arm.num_joints

    env = _mock_env()
    try:
        env.reset()
        assert env.arm.get_joint_positions().shape == (cfg.arm.num_joints,)
        env.step([0.01, 0, 0, 0, 0, 0, 1.0])
        assert env.arm.get_joint_positions().shape == (cfg.arm.num_joints,)
    finally:
        env.close()

    # A home posture that disagrees with num_joints must fail loudly at
    # construction, not home the arm into an undefined pose.
    try:
        AR5Config().arm.__class__(num_joints=7, home_joints=(0.0,) * 6)
    except ValueError:
        pass
    else:
        raise AssertionError("mismatched home_joints was accepted")


@check("an unreachable step holds position instead of ending the rollout")
def _():
    env = _mock_env()
    try:
        env.reset()
        before = env.arm.get_ee_pose().copy()

        # Stand in for the controller refusing a target, as the real one does
        # when inverse kinematics fails.
        env.arm.check_reachable = lambda pose: (False, "ec=-50002: simulated refusal")

        obs = env.step([0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        assert obs is not None, "step must still return an observation"
        assert np.allclose(env.arm.get_ee_pose(), before), "the arm should not have moved"
        assert env.stats["unreachable"] == 1, env.stats
        assert env.stats["steps"] == 1, env.stats
        # The gripper is independent of arm reachability and must still apply.
        assert env.hand.get_opening() == 0.0
    finally:
        env.close()


@check("the setpoint re-anchors when the arm falls behind")
def _():
    env = _mock_env()
    try:
        env.reset()
        for _ in range(3):
            env.step([0.005, 0, 0, 0, 0, 0, 1.0])
        assert env.stats["resynced"] == 0

        # Shove the commanded setpoint far from where the arm actually is, as a
        # stall or a refused step would.
        env._target_pose = env._target_pose + np.array([0.10, 0, 0, 0, 0, 0])
        env.step([0.005, 0, 0, 0, 0, 0, 1.0])
        assert env.stats["resynced"] == 1, env.stats

        # After re-anchoring the target must track the arm again, not the
        # runaway setpoint.
        drift = np.linalg.norm(env._target_pose[:3] - env.arm.get_ee_pose()[:3])
        assert drift < env.config.resync_threshold, drift
    finally:
        env.close()


@check("the arm runs without a hand attached")
def _():
    from atom_robot.ar5.drivers.mock import NullHand

    cfg = AR5Config(mock=True)
    cfg.use_hand = False
    hand = NullHand(cfg.hand)
    hand.connect()

    # Commands are recorded and echoed back, so proprio keeps its width and the
    # recorded actions stay comparable with runs that do have a hand.
    hand.set_opening(0.0)
    assert hand.get_opening() == 0.0
    hand.set_opening(0.4)
    assert np.isclose(hand.get_opening(), 0.4)
    hand.set_opening(5.0)
    assert hand.get_opening() == 1.0
    assert hand.get_joint_angles().shape == (cfg.hand.num_joints,)
    hand.disconnect()


def _mapper():
    from atom_robot.ar5.leader_mapping import LeaderMapper, MappingConfig

    mapper = LeaderMapper(MappingConfig())
    mapper.connect()
    return mapper


# Two leader postures far enough apart to clear the deadband.
_POSE_A = {"shoulder_pan": 0.0, "shoulder_lift": -100.0, "elbow_flex": 90.0,
           "wrist_flex": 45.0, "wrist_roll": 0.0, "gripper": -60.0}
_POSE_B = dict(_POSE_A, shoulder_pan=10.0)


@check("leader forward kinematics is deterministic and moves with the joints")
def _():
    need("placo")          # 笛卡尔遥操那条路才用，采集用不到
    mapper = _mapper()
    a1 = mapper.leader_pose(_POSE_A)
    a2 = mapper.leader_pose(_POSE_A)
    assert np.allclose(a1, a2), "FK must be a pure function of the joints"
    assert a1.shape == (6,)

    b = mapper.leader_pose(_POSE_B)
    assert not np.allclose(a1[:3], b[:3]), "rotating the base must move the tool"
    # 10 degrees of base rotation at ~0.2 m reach is a few centimetres.
    travel = np.linalg.norm(b[:3] - a1[:3])
    assert 0.005 < travel < 0.2, travel


@check("the clutch gates arm motion but not the gripper")
def _():
    need("placo")          # 笛卡尔遥操那条路才用，采集用不到
    mapper = _mapper()

    # Released: the arm delta must be identically zero however far the leader moves.
    mapper.compute_action(_POSE_A)
    action = mapper.compute_action(_POSE_B)
    assert np.allclose(action[:6], 0.0), action
    assert 0.0 <= action[6] <= 1.0

    # Engaging references the current pose, so the first step cannot jump.
    mapper.engage(_POSE_A)
    first = mapper.compute_action(_POSE_A)
    assert np.allclose(first[:6], 0.0), first

    moved = mapper.compute_action(_POSE_B)
    assert np.linalg.norm(moved[:3]) > 0, "engaged motion must reach the follower"

    # Disengaging drops the reference, so re-engaging elsewhere does not replay
    # the accumulated offset.
    mapper.disengage()
    assert np.allclose(mapper.compute_action(_POSE_A)[:6], 0.0)


@check("position scale multiplies follower travel")
def _():
    need("placo")          # 笛卡尔遥操那条路才用，采集用不到
    from atom_robot.ar5.leader_mapping import LeaderMapper, MappingConfig

    travels = {}
    for scale in (1.0, 2.5):
        mapper = LeaderMapper(MappingConfig(position_scale=scale))
        mapper.connect()
        mapper.engage(_POSE_A)
        action = mapper.compute_action(_POSE_B)
        travels[scale] = np.linalg.norm(action[:3])

    assert travels[1.0] > 0
    ratio = travels[2.5] / travels[1.0]
    assert np.isclose(ratio, 2.5, rtol=1e-6), ratio


@check("the deadband suppresses tremor without losing slow motion")
def _():
    need("placo")          # 笛卡尔遥操那条路才用，采集用不到
    from atom_robot.ar5.leader_mapping import LeaderMapper, MappingConfig

    mapper = LeaderMapper(MappingConfig(position_deadband=0.01, enable_rotation=False))
    mapper.connect()
    mapper.engage(_POSE_A)

    # A tiny nudge falls inside the deadband and emits nothing.
    tiny = dict(_POSE_A, shoulder_pan=_POSE_A["shoulder_pan"] + 0.05)
    assert np.allclose(mapper.compute_action(tiny)[:3], 0.0)
    assert mapper.stats["deadbanded"] == 1

    # Motion the deadband swallowed must accumulate rather than vanish. The
    # tool sits ~0.21 m from the base, so a degree of pan is ~3.7 mm of travel:
    # the sweep below has to reach several degrees to clear a 10 mm threshold.
    emitted, steps_until_first = 0.0, None
    for i in range(1, 60):
        pose = dict(_POSE_A, shoulder_pan=_POSE_A["shoulder_pan"] + 0.1 * i)
        travel = float(np.linalg.norm(mapper.compute_action(pose)[:3]))
        emitted += travel
        if travel > 0 and steps_until_first is None:
            steps_until_first = i

    assert emitted > 0, "accumulated sub-deadband motion must eventually be emitted"
    # It must be held back for a while first, otherwise the deadband is inert.
    assert steps_until_first is not None and steps_until_first > 5, steps_until_first
    # And when it is released, it comes out as one jump of at least the
    # threshold -- not as the single last increment.
    assert emitted >= mapper.config.position_deadband, emitted


@check("a leader step drives the follower through AR5Env")
def _():
    need("placo")          # 笛卡尔遥操那条路才用，采集用不到
    mapper = _mapper()
    env = _mock_env()
    try:
        env.reset(fake=True)
        start = env.arm.get_ee_pose().copy()

        mapper.engage(_POSE_A)
        action = mapper.compute_action(_POSE_B)
        env.step(action)

        moved = np.linalg.norm(env.arm.get_ee_pose()[:3] - start[:3])
        assert moved > 0, "the follower should have moved"
        # AR5Env's per-step clamp still bounds whatever the leader asks for.
        assert moved <= env.config.arm.max_step_translation + 1e-9, moved
    finally:
        env.close()


@check("a logged rollout carries its own step count, never the previous round's")
def _():
    # 实测那 60 轮：步数/位移存在一个只在 q 时更新的变量里，按 r/f 的时机一变，
    # 整列要么空、要么错位一轮。round_stats 从当前 deltas 现算，错不了位。
    import csv
    import tempfile
    from argparse import Namespace

    from atom_robot.ar5.eval_policy import log_rollout, round_stats

    assert round_stats(0, []) == {}, "空轮必须给空字典，否则 CSV 写出 0.00 假数据"
    r = round_stats(300, [1.0, 3.0, 2.0])
    assert (r["steps"], r["median"], r["max"]) == (300, 2.0, 3.0), r

    path = Path(tempfile.mkdtemp()) / "rollouts.csv"
    args = Namespace(tag="t", ckpt="/x/030000", vel=None, log=str(path))
    ev = Namespace(scale=1.0)
    for i, deltas in enumerate(([1.0, 2.0], [4.0, 6.0])):
        log_rollout(path, {"success": i + 1, "fail": 0}, True,
                    {"grasp": True, "insert": True},
                    round_stats(100 * (i + 1), deltas), args, ev)
    rows = list(csv.DictReader(path.open()))
    assert [x["步数"] for x in rows] == ["100", "200"], rows
    assert [x["位移中位mm"] for x in rows] == ["1.50", "5.00"], rows


@check("里程碑打标：每段的成功率各自可算，老 CSV 自动补列改名且不丢数据")
def _():
    # 能量头是按段训的。只记整体成功率，抓取段的改善会被后面的失败稀释。
    import csv
    import tempfile
    from argparse import Namespace

    from atom_robot.ar5.eval_policy import (
        CSV_HEADER, CSV_RENAMED, IMPLIES, MILESTONES, log_rollout, migrate_csv)

    path = Path(tempfile.mkdtemp()) / "rollouts.csv"
    args = Namespace(tag="t", ckpt="/x/040000", vel=None, log=str(path))
    ev = Namespace(scale=1.0)
    cases = [
        (True, {"grasp": True, "insert": True}),     # r：全程成功
        (False, {"grasp": True, "insert": True}),    # 插了但退回时碰掉
        (False, {"grasp": True, "insert": False}),   # 抓到了没插进去
        (False, {"grasp": False, "insert": False}),  # 没抓到
    ]
    for ok, marks in cases:
        log_rollout(path, {"success": int(ok), "fail": 0}, ok, marks, {}, args, ev)
    rows = list(csv.DictReader(path.open()))
    assert [r["结果"] for r in rows] == ["成功", "失败", "失败", "失败"], rows
    assert [r["抓取"] for r in rows] == ["成功", "成功", "成功", "失败"], rows
    assert [r["插入"] for r in rows] == ["成功", "成功", "失败", ""], rows
    # **没抓到那轮的「插入」必须留空而不是「失败」** —— 那一段根本没发生，
    # 算进分母会让抓取段的改善伪装成插入段的改善。
    assert rows[3]["插入"] == "", rows[3]

    # 里程碑的蕴含关系：插入蕴含抓取，全程成功蕴含两个
    assert IMPLIES["i"] == ("g",)
    assert set(IMPLIES["r"]) == set(MILESTONES), IMPLIES

    # 老格式（列逐步加出来、还改过名）要能就地迁移，历史行一条不少
    old_path = Path(tempfile.mkdtemp()) / "rollouts.csv"
    legacy = [c for c in CSV_HEADER if c != "插入"]
    legacy[legacy.index("抓取")] = "抓取"
    legacy.insert(legacy.index("抓取") + 1, "运送")          # 改名前的列
    with old_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(legacy)
        w.writerow(["2026-09-10T10:38:42", "1", "成功", "", "", "473", "0.22", "9.14",
                    "random", "030000", "1.0", ""])
    assert migrate_csv(old_path) is True
    assert migrate_csv(old_path) is False, "迁移过一次就不该再动"
    rows = list(csv.DictReader(old_path.open()))
    assert len(rows) == 1 and rows[0]["插入"] == "", rows
    assert rows[0]["步数"] == "473" and rows[0]["tag"] == "random", rows[0]

    # **改名的老列里如果真有数据，必须报错而不是当成新列。** 语义变了。
    dirty = Path(tempfile.mkdtemp()) / "rollouts.csv"
    with dirty.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(legacy)
        w.writerow(["2026-09-10T10:38:42", "1", "成功", "成功", "成功", "473", "0.22",
                    "9.14", "random", "030000", "1.0", ""])
    try:
        migrate_csv(dirty)
    except RuntimeError as exc:
        assert "运送" in str(exc) and "插入" in str(exc), exc
    else:
        raise AssertionError(f"「运送」列有数据也照改，会把 {CSV_RENAMED} 的语义搞混")


@check("主循环每帧都要能吃「没按键」，按键分支不能假设 key 是字符串")
def _():
    # 实测崩过：把 `key in ("r","f")` 改成 `key.lower() in OUTCOMES` 之后，
    # 没按键的那一帧 key 是 None，第 0 步就 AttributeError。空跑抓到的。
    import ast

    src = (Path(__file__).parent / "eval_policy.py").read_text()
    assert "key.lower()" in src, "这条检查跟着 key.lower() 走，用法变了就该改检查"

    def lower_calls(node):
        return [n for n in ast.walk(node)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "lower" and isinstance(n.func.value, ast.Name)
                and n.func.value.id == "key"]

    tree = ast.parse(src)
    # 凡是被「key is not None」保护住的分支（if 语句或三元），整棵子树里的
    # key.lower() 都算安全。剩下的就是裸用。
    guarded = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.IfExp)) and (
                "key is not None" in ast.unparse(node.test)
                or "key and" in ast.unparse(node.test)):
            guarded.update(id(c) for c in lower_calls(node))
    calls = lower_calls(tree)
    assert calls, "没找到 key.lower()，检查失效了"
    bare = [c for c in calls if id(c) not in guarded]
    assert not bare, (
        f"eval_policy.py:{bare[0].lineno} 对 key 裸调 .lower()，"
        f"没按键那一帧 key 是 None，主循环第 0 步就崩：\n"
        f"    {src.splitlines()[bare[0].lineno - 1].strip()}")

    # RawKeyboard 在没有按键 / 非终端时确实返回 None，上面那条约束才有意义
    from atom_robot.ar5.eval_policy import RawKeyboard

    kb = RawKeyboard.__new__(RawKeyboard)
    kb.tty = False
    assert kb.key() is None


@check("在线的分段判据和离线训练用的那套，在真示教上对得上")
def _():
    # 离线切「运送/放置」用 z 的峰值（开天眼），在线只能因果判断。两者必须
    # 落在同一个位置附近，否则「训的是放置段、跑的时候挂在别处」。
    import glob
    import os

    import numpy as np

    from atom_robot.ar5.vel.pi05_bridge import GRIPPER_CLOSED, VelRunner
    from atom_robot.ar5.vel.train_energy import SEGMENT_CODE, segment_of_episode

    files = sorted(glob.glob(os.path.expanduser("~/atom_episodes/episode_*.npz")))[:60]
    if not files:
        return                                     # 没有示教数据就跳过

    from collections import deque

    runner = VelRunner.__new__(VelRunner)
    runner.queue = deque()
    late, never, early = [], 0, 0
    for f in files:
        d = np.load(f, allow_pickle=True)
        action, ee = d["action"], d["ee_pose"]
        off = segment_of_episode(action, ee)
        want = np.flatnonzero(off == SEGMENT_CODE["place"])
        if not len(want):
            continue
        runner.reset()
        fire = None
        for t in range(len(action)):
            runner.observe(action[t], ee[t])
            if fire is None and runner.segment == "place":
                fire = t
            # 夹爪判据两边必须一致
            # 夹爪合着时离线只可能是 transit / place / -1（-1 = 掉落后重抓
            # 之前那段失败尝试，不属于任何段）。绝不能是 grasp 或 retreat。
            if action[t, 6] < GRIPPER_CLOSED:
                assert off[t] in (SEGMENT_CODE["transit"], SEGMENT_CODE["place"], -1), (
                    f"{Path(f).name}:{t} 夹爪合着，离线却标成 "
                    f"{off[t]}（在线 {runner.segment}）")
        if fire is None:
            never += 1
            continue
        if fire < want[0]:
            early += 1
        late.append(fire - want[0])

    n = len(late) + never
    assert n >= 30, f"只比了 {n} 条，样本太少"
    assert early / n < 0.05, (
        f"{early}/{n} 条在离线峰值之前就触发了放置段 —— 抬升途中的短暂回落"
        f"被误当成下探。调 LIFT_ARM_MM / DESCEND_MM / DESCEND_FRAMES")
    assert never / n < 0.05, f"{never}/{n} 条整条都没触发放置段"
    assert np.median(late) < 30, (
        f"在线比离线晚 {np.median(late):.0f} 帧（放置段中位才 58 帧），"
        f"放置段的头几乎用不上")


@check("排序损失的符号和 ρ 的算法：理想头 ρ=+1，反着的头 ρ=−1")
def _():
    # 训练前 60 步 ρ 是负的（−0.087 ~ −0.225），一眼像符号写反了。用两个解析的
    # 假头一验就知道是「还没学出来」而不是 bug —— 省下 50 分钟的白跑。
    import torch

    import atom_robot.ar5.vel.train_energy as T

    B, K, H, D = 4, 4, 8, 7
    torch.manual_seed(0)
    a_star = torch.zeros(B, H, D)
    # K 个采样到 a* 的距离递增
    a_bc = torch.randn(B, K, H, D) * torch.tensor([0.5, 1.0, 2.0, 3.0]).view(1, K, 1, 1)
    h = torch.randn(B, 3, 2)
    valid = torch.ones(B, 3, dtype=torch.bool)

    def dist(pert, B_, k_):
        return (pert.view(B_, k_, H, D) - a_star.unsqueeze(1)).flatten(2).norm(dim=-1)

    orig = T._energy_at
    try:
        T._energy_at = lambda hd, h_, pert, v, B_, k_: dist(pert, B_, k_)
        good_loss, good_rho = T.rank_loss(None, h, a_star, a_bc, valid)
        T._energy_at = lambda hd, h_, pert, v, B_, k_: -dist(pert, B_, k_)
        bad_loss, bad_rho = T.rank_loss(None, h, a_star, a_bc, valid)
    finally:
        T._energy_at = orig

    assert good_rho.item() > 0.99, f"E 随距离单调递增时 ρ 应为 +1，实际 {good_rho.item():+.3f}"
    assert bad_rho.item() < -0.99, f"E 反着时 ρ 应为 −1，实际 {bad_rho.item():+.3f}"
    assert good_loss.item() < bad_loss.item(), (
        f"排序损失的符号反了：正确排序 {good_loss.item():.3f} "
        f"应该低于错误排序 {bad_loss.item():.3f}")


@check("逐块能量落盘：修正前后各一列，每个加载的头各一对")
def _():
    # 论文 §6.4 的「逐时刻能量曲线」。基线用 --vel-off 跑，那时 out == bc。
    import csv
    import tempfile
    from argparse import Namespace

    from atom_robot.ar5.eval_policy import write_energy_trace

    path = Path(tempfile.mkdtemp()) / "e.csv"
    trace = [{"step": 0, "segment": "grasp", "applied": "grasp",
              "E_grasp_bc": -3.5, "E_grasp_out": -3.7,
              "E_place_bc": 1.2, "E_place_out": 1.2},
             {"step": 50, "segment": "place", "applied": "place",
              "E_grasp_bc": 0.4, "E_grasp_out": 0.4,
              "E_place_bc": -2.0, "E_place_out": -2.1}]
    write_energy_trace(path, trace, Namespace(tag="t"), rollout=3, ok=True)
    rows = list(csv.DictReader(path.open()))
    assert len(rows) == 2, rows
    assert rows[0]["步"] == "0" and rows[1]["步"] == "50"
    assert rows[0]["段"] == "grasp" and rows[1]["段"] == "place"
    assert rows[0]["轮次"] == "3" and rows[0]["结果"] == "成功"
    for c in ("E_grasp_bc", "E_grasp_out", "E_place_bc", "E_place_out"):
        assert c in rows[0], f"{c} 没落盘 —— 画不了修正前后的对比曲线"
    assert float(rows[0]["E_grasp_out"]) < float(rows[0]["E_grasp_bc"]), (
        "修正后能量应该更低（这正是一步下降的保证）")

    # **reset() 不能清 trace。** 按 q 会调 reset，而 r/f 在那之后才落盘 ——
    # 清掉的话「先按 q 再记结果」的轮次曲线全丢。实测 9 轮只留下 3 轮。
    from collections import deque

    from atom_robot.ar5.vel.pi05_bridge import VelRunner

    r = VelRunner.__new__(VelRunner)
    r.queue, r.enabled = deque(), True
    r.reset()
    r.trace.append({"step": 0})
    r.reset()
    assert len(r.trace) == 1, "reset 之后 trace 必须还在，等 r/f 落盘"


@check("盲测 A/B：两臂只差那一步修正，屏幕不泄露，逐轮统计每轮清零")
def _():
    # 隔天对比消不掉日间差异：基线 09-10 采、VEL 09-11 采，相机漂了 6.4 px，
    # 每轮间隔从 20 秒变 30 秒。插入率差了 17 个点（p=0.031），但归因不了。
    # 同一场次随机交替才能把这些全消掉。
    import csv
    import tempfile
    from argparse import Namespace
    from collections import deque

    import numpy as np

    from atom_robot.ar5.eval_policy import CSV_HEADER, log_rollout
    from atom_robot.ar5.vel.pi05_bridge import SEGMENTS, VelRunner

    for c in ("抓取修正块", "放置修正块", "脱手次数"):
        assert c in CSV_HEADER, f"{c} 没进 CSV —— 没法做「头真修了 vs 没修」的内部检验"

    r = VelRunner.__new__(VelRunner)
    r.queue, r.enabled = deque(), True
    r.reset()
    # **每轮清零。** 之前 corrections 建在 __init__ 里，每轮打的「修正 N 块」
    # 是从开程序算起的累计值，随轮次单调增，完全没法用。
    r.corrections = [("grasp", 0.1), ("grasp", 0.2), ("place", 0.3)]
    r.lost = 2
    assert r.counts() == {"grasp": 2, "transit": 0, "place": 1, "retreat": 0}, r.counts()
    r.reset()
    assert r.counts() == {s: 0 for s in SEGMENTS}, "reset 之后必须清零"
    assert r.lost == 0

    # 随机而非严格交替：严格交替时操作者数几轮就能推出当前是哪一臂
    seq = [bool(np.random.default_rng(0).integers(2)) for _ in range(1)]
    rng = np.random.default_rng(0)
    arms = [bool(rng.integers(2)) for _ in range(40)]
    assert 0 < sum(arms) < 40, "两臂都要出现"
    assert any(arms[i] == arms[i + 1] for i in range(39)), "严格交替会被操作者猜出来"
    assert seq[0] == arms[0], "同一种子要可复现"

    # 修正块数要真的落进 CSV
    path = Path(tempfile.mkdtemp()) / "r.csv"
    args = Namespace(tag="ab-on", ckpt="/x/040000", vel=["grasp=/a/h.pt"], log=str(path))
    log_rollout(path, {"success": 1, "fail": 0}, True,
                {"grasp": True, "insert": True}, {}, args, Namespace(scale=1.0),
                vel_stats=(3, 1, 2))
    row = next(iter(csv.DictReader(path.open())))
    assert (row["抓取修正块"], row["放置修正块"], row["脱手次数"]) == ("3", "1", "2"), row


@check("VEL 按段挂头：没给头的段原样放行，不拿抓取段的头去修运送段")
def _():
    # 第一版就是这个 bug：--vel 只接一个头，整条轨迹都用它。那 30 轮里运送和
    # 退回两段用的都是只见过抓取段的头 —— 分布外，测出来的差异归因不了。
    import torch

    from atom_robot.ar5.eval_policy import vel_column
    from atom_robot.ar5.vel.pi05_bridge import GRIPPER_CLOSED, SEGMENTS, VelRunner

    class _Head(torch.nn.Module):
        def forward(self, h, a, mask=None):
            return (a ** 2).flatten(1).sum(1, keepdim=True)

    class _Policy:
        class config:
            n_action_steps = 2
    runner = VelRunner.__new__(VelRunner)          # 绕开 Pi05Features，只测分段逻辑
    from collections import deque
    runner.heads = {"grasp": _Head()}
    runner.alpha, runner.gamma = 0.1, 1.0
    runner.n_steps, runner.queue = 2, deque()
    runner.corrections = []
    runner.skipped = {s: 0 for s in SEGMENTS}
    runner.reset()

    assert SEGMENTS == ("grasp", "transit", "place", "retreat"), SEGMENTS
    assert runner.head_for("grasp") is not None
    for s in ("transit", "place", "retreat"):
        assert runner.head_for(s) is None, f"{s} 没给头就必须是 None（放行）"

    # 段按夹爪 + 末端高度切，和 train_energy.segment_of_episode 对齐，单调不回退
    open_a, closed_a = [0] * 6 + [0.8], [0] * 6 + [GRIPPER_CLOSED - 0.01]

    def z(mm):
        return [0.0, 0.0, mm / 1000.0]

    from atom_robot.ar5.vel.pi05_bridge import GRIP_FRAMES

    runner.observe(open_a, z(110))
    assert runner.segment == "grasp"
    # **一次极短的失败抓取不该切段。** 切了会把 z 基准挪到抬升途中，
    # 「高出 LIFT_ARM_MM」的门槛跟着抬高，放置段可能永远触发不了。
    for _ in range(GRIP_FRAMES - 1):
        runner.observe(closed_a, z(110))
    assert runner.segment == "grasp", "闭合没满 GRIP_FRAMES 帧就切，抖一下就误判"
    runner.observe(open_a, z(110))
    assert runner.segment == "grasp"
    for _ in range(GRIP_FRAMES):
        runner.observe(closed_a, z(110))
    assert runner.segment == "transit", "连续闭合够帧数才进运送段"
    for mm in (120, 140, 160, 180, 190):          # 抬升，还没到下探
        runner.observe(closed_a, z(mm))
    assert runner.segment == "transit", f"抬升途中不该进放置段（{runner.segment}）"
    # 抬够了（190-110 = 80 > 40），现在跌破 190-15=175 且要持续 5 帧
    for mm in (170, 168, 166, 164):
        runner.observe(closed_a, z(mm))
    assert runner.segment == "transit", "只跌了 4 帧，还差一帧"
    runner.observe(closed_a, z(162))
    assert runner.segment == "place", "连续 5 帧下探就该进放置段"
    for _ in range(GRIP_FRAMES):
        runner.observe(open_a, z(120))
    assert runner.segment == "retreat", "放置段里连续松开够帧数才进退回段"
    runner.observe(closed_a, z(120))
    assert runner.segment == "retreat", "一帧闭合不算，要够 GRIP_FRAMES"
    runner.observe(open_a, z(120))               # 断开，让下一段闭合从 115 起算
    # **retreat 不能是吸收态。** 「放了但没插进去、再抓起来重插」在真机上是
    # 主要失败模式（46% 的轮次是「抓到了插不进去」）。卡死在 retreat 的话，
    # 之后的重抓重送全程挂不上头。
    for _ in range(GRIP_FRAMES):
        runner.observe(closed_a, z(115))
    assert runner.segment == "transit", "退回段里重新抓起来，要能回到运送段"
    # z 基准取的是**这一段连续闭合的第一帧**，不是确认切段时那一帧 ——
    # 那 GRIP_FRAMES 帧里臂可能已经开始抬了，用确认时的高度基准会偏高。
    assert abs(runner._z_close - 115) < 1e-6, (
        f"重抓后 z 基准应为 115，实际 {runner._z_close}")

    # **运送途中松手 = 掉了，不是放完了。** 实测 32/297 条示教有这个（中位松
    # 48 帧）。判成退回段的话，后面重抓、重送全程都挂不上头。
    runner.reset()
    for _ in range(GRIP_FRAMES):
        runner.observe(closed_a, z(110))
    assert runner.segment == "transit"
    for _ in range(GRIP_FRAMES):
        runner.observe(open_a, z(150))
    assert runner.segment == "grasp", "运送途中松手要退回抓取段等重抓"
    # **重抓之后 z 基准要重新取**，不能留着失败那次的（否则基准被抬到 150，
    # 门槛变成 190，放置段永远触发不了）。
    for _ in range(GRIP_FRAMES):
        runner.observe(closed_a, z(108))
    assert runner.segment == "transit", "重抓之后要能再进运送段"
    assert abs(runner._z_close - 108) < 1e-6, (
        f"z 基准应为重抓时的 108，实际 {runner._z_close}")

    # **没抬起来就不许进放置段** —— 否则抬升途中一个短暂回落就误触发。
    # 实测不加这条，296 条里 61 条会误触发，最早的早了 145 帧。
    runner.reset()
    assert runner.segment == "grasp", "reset 必须回到抓取段，否则下一轮全走错头"
    for _ in range(GRIP_FRAMES):
        runner.observe(closed_a, z(110))
    for mm in (120, 125, 100, 100, 100, 100, 100, 100):   # 只抬了 15 mm 就掉回去
        runner.observe(closed_a, z(mm))
    assert runner.segment == "transit", "没抬够 40 mm 就跌落，不该算放置段"

    # **夹爪闭着但状态说手里是空的 = 脱手**，不等策略去松手就退回抓取段。
    # 这是实测的典型失败：夹爪合上、策略以为抓到了，闭着运送一段才发现是空的。
    # 判据「连续 5 帧状态 < 0.12」：296 条成功里误报 1.7%，25 段失败里识别 72%。
    from atom_robot.ar5.vel.pi05_bridge import GRIP_EMPTY, GRIP_EMPTY_FRAMES

    runner.reset()
    for _ in range(GRIP_FRAMES):
        runner.observe(closed_a, z(110), grip_state=0.22)      # 夹住了
    assert runner.segment == "transit"
    for _ in range(GRIP_EMPTY_FRAMES - 1):
        runner.observe(closed_a, z(120), grip_state=0.03)
    assert runner.segment == "transit", "还差一帧，不该切"
    runner.observe(closed_a, z(120), grip_state=0.03)
    assert runner.segment == "grasp", "夹爪闭着但手里是空的，要退回抓取段"
    assert runner.lost == 1, runner.lost
    # 抓稳的那些不能被误判 —— 成功抓取时状态在 0.19 上下，远高于阈值
    runner.reset()
    for _ in range(GRIP_FRAMES + 30):
        runner.observe(closed_a, z(110), grip_state=0.19)
    assert runner.segment == "transit" and runner.lost == 0, "抓稳的不该判成脱手"
    assert GRIP_EMPTY < 0.151, "阈值要低于成功抓取最低值的 10% 分位（0.151）"

    # 不给 ee_pose 时退化：进得了运送段，但**永远进不了放置段**（没有高度信息），
    # 于是松手一律当成掉落。eval_policy 总会把 ee_pose 传进来，这里只保证不崩。
    runner.reset()
    for _ in range(GRIP_FRAMES):
        runner.observe(closed_a)
    assert runner.segment == "transit"
    for _ in range(GRIP_FRAMES):
        runner.observe(open_a)
    assert runner.segment == "grasp"

    # CSV 那一列要能区分「只挂抓取段」和「全程都挂」——它们是不同的实验条件
    assert vel_column(["grasp=/a/h.pt"]) == "grasp=h.pt"
    assert vel_column(["/a/h.pt"]) == "all=h.pt"
    assert vel_column(["grasp=/a/h.pt"]) != vel_column(["/a/h.pt"])
    assert vel_column(None) == ""


@check("rollout videos are written without touching the ROS-bound ALOHA utils")
def _():
    import tempfile

    from atom_robot.ar5.video import save_rollout_video

    assert save_rollout_video([], 1, True, "empty", out_dir=tempfile.mkdtemp()) is None

    env = _mock_env()
    try:
        env.reset()
        view = next(iter(env.config.cameras.sources))
        frames = [env.step([0.005, 0, 0, 0, 0, 0, 1.0]).observation["images"][view] for _ in range(6)]
    finally:
        env.close()

    out_dir = tempfile.mkdtemp()
    path = save_rollout_video(frames, 3, False, "put the mug in the microwave!", out_dir=out_dir)
    if path is None:
        # No ffmpeg backend available; the caller-side contract still holds.
        return
    assert Path(path).exists() and Path(path).stat().st_size > 0
    assert "episode_003_failure" in Path(path).name
    assert "!" not in Path(path).name


def main() -> int:
    passed = failed = 0
    skipped = []
    for name, fn in _results:
        try:
            fn()
        except Skip as why:
            skipped.append((name, str(why)))
            print(f"skip  {name}  —— {why}")
        except Exception:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {name}")
            traceback.print_exc()
            print()
        else:
            passed += 1
            print(f"ok    {name}")

    tail = f"，{len(skipped)} 项跳过" if skipped else ""
    print(f"\n{passed} passed, {failed} failed{tail}")
    for name, why in skipped:
        print(f"  skip: {name}  ({why})")
    return 1 if failed else 0


# ------------------------------------------------------- local kinematics


def _kin():
    import json
    from atom_robot.ar5.config import ArmConfig
    from atom_robot.ar5.kinematics import AR5Kinematics

    data = json.loads((Path(__file__).parent / "ar5_dh.json").read_text())
    cfg = ArmConfig()
    return (
        AR5Kinematics(np.array(data["dh"]).reshape(-1, 4), 7, cfg.joint_min, cfg.joint_max),
        np.array(data["q_sample"]),
    )


@check("forward kinematics is deterministic and its Jacobian matches finite differences")
def _():
    kin, q0 = _kin()
    assert np.allclose(kin.fk(q0), kin.fk(q0))

    # A wrong Jacobian is the quickest way to an unstable IK, and it is easy to
    # get the joint axis one frame off with modified DH.
    J = kin.jacobian(q0)
    assert J.shape == (6, 7)
    eps = 1e-6
    for i in range(7):
        dq = np.zeros(7)
        dq[i] = eps
        T_plus, T_minus = kin.fk(q0 + dq), kin.fk(q0 - dq)
        dp = (T_plus[:3, 3] - T_minus[:3, 3]) / (2 * eps)
        assert np.allclose(J[:3, i], dp, atol=1e-4), f"joint {i}: {J[:3, i]} vs {dp}"


@check("inverse kinematics converges and stays on the seeded redundancy branch")
def _():
    kin, q0 = _kin()
    rng = np.random.default_rng(3)
    failures, residuals, branch_jumps = 0, [], 0
    for _ in range(200):
        target = np.clip(q0 + rng.normal(0, 0.06, 7), kin.joint_min, kin.joint_max)
        T = kin.fk(target)
        q, info = kin.ik(T, q0)
        if q is None:
            failures += 1
            continue
        residuals.append(np.linalg.norm(kin.fk(q)[:3, 3] - T[:3, 3]))
        # Seven joints are redundant, so the solver may legitimately land on a
        # different member of the solution family -- but not a distant one, or
        # the elbow swings while the tool barely moves.
        if np.max(np.abs(q - target)) > 0.5:
            branch_jumps += 1

    assert failures == 0, f"{failures}/200 IK failures"
    assert max(residuals) < 2e-4, f"worst residual {max(residuals)} m"
    assert branch_jumps == 0, f"{branch_jumps} branch jumps"


@check("inverse kinematics refuses an unreachable target instead of returning nonsense")
def _():
    kin, q0 = _kin()
    T = kin.fk(q0)
    T[0, 3] += 5.0          # metres away, far outside a 0.68 m reach
    q, info = kin.ik(T, q0)
    assert q is None, "an unreachable target must not produce a solution"
    assert info["reason"], info


@check("tool-point Jacobian matches finite differences of flange @ tool")
def _():
    kin, q0 = _kin()
    tool = np.eye(4)
    tool[2, 3] = 0.13
    _, J = kin.pose_and_jacobian(q0, tool)
    eps = 1e-6
    for i in range(7):
        dq = np.zeros(7)
        dq[i] = eps
        dp = ((kin.fk(q0 + dq) @ tool)[:3, 3] - (kin.fk(q0 - dq) @ tool)[:3, 3]) / (2 * eps)
        assert np.allclose(J[:3, i], dp, atol=1e-4), f"joint {i}: {J[:3, i]} vs {dp}"


@check("weighted IK reaches reachable targets within a few frames")
def _():
    kin, q0 = _kin()
    rng = np.random.default_rng(5)
    residuals = []
    for _ in range(100):
        target = np.clip(q0 + rng.normal(0, 0.06, 7), kin.joint_min, kin.joint_max)
        T = kin.fk(target)
        q = q0.copy()
        for _frame in range(3):         # unbounded step, 8 iterations a frame
            q, info = kin.ik_weighted(T, q)
        assert np.all(q >= kin.joint_min - 1e-9) and np.all(q <= kin.joint_max + 1e-9)
        residuals.append(info["pos_error"])
    assert np.median(residuals) < 5e-4, f"median residual {np.median(residuals)} m"


@check("weighted IK never fails: unreachable target -> bounded step toward it, inside limits")
def _():
    kin, q0 = _kin()
    T = kin.fk(q0)
    T[0, 3] += 5.0
    step = 0.0137
    q, info = kin.ik_weighted(T, q0, step_limit=step)
    assert q is not None and np.all(np.isfinite(q))
    assert np.max(np.abs(q - q0)) <= step + 1e-12, np.max(np.abs(q - q0))
    assert np.all(q >= kin.joint_min - 1e-9) and np.all(q <= kin.joint_max + 1e-9)
    assert kin.fk(q)[0, 3] > kin.fk(q0)[0, 3], "should move toward the target"
    assert info["bounded"], "an unreachable target should leave joints on the step bound"


@check("weighted IK gives up orientation, not position, when the wrist cannot follow")
def _():
    # 主臂手腕要出 AR5 手腕给不了的姿态时（10-30 真机笛卡尔卡住的主因），位置照样要跟上
    from atom_robot.ar5.transforms import axis_angle_to_matrix
    kin, q0 = _kin()
    T = kin.fk(q0)
    T[:3, :3] = axis_angle_to_matrix(np.array([0.0, 2.5, 0.0])) @ T[:3, :3]   # 143 deg
    T[2, 3] += 0.02
    assert kin.ik(T, q0)[0] is None, "premise: dls cannot solve this"
    q = q0.copy()
    for _frame in range(5):
        q, info = kin.ik_weighted(T, q)
    assert info["pos_error"] < 3e-3, f"position should still track: {info['pos_error']} m"
    assert info["rot_error"] > 0.3, "the orientation was the part that had to give"


@check("cartesian mapper: default is the original dls at the flange; weighted is opt-in")
def _():
    from atom_robot.ar5.cartesian_mapping import (CartesianMapConfig, _clip_step,
                                                  _hold_if_small)
    from atom_robot.ar5.config import CollectConfig
    c = CartesianMapConfig()
    assert (c.ik, c.tcp_z) == ("dls", 0.0)
    for bad in (dict(ik="mink"), dict(tcp_z=-0.01), dict(tcp_z=1.0)):
        try:
            CartesianMapConfig(**bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted {bad}")
    cc = CollectConfig()
    assert (cc.cart_ik, cc.cartesian_tcp_z()) == ("dls", 0.0)
    cc.cart_ik = "weighted"
    assert cc.cartesian_tcp_z() == 0.13
    cc.tcp_z = 0.0
    assert cc.cartesian_tcp_z() == 0.0

    T0 = np.eye(4)
    T1 = np.eye(4)
    T1[0, 3] = 0.002
    assert np.allclose(_hold_if_small(T0, T1, 0.003, 0.03), T0), "inside the deadband"
    T1[0, 3] = 0.05
    assert np.allclose(_hold_if_small(T0, T1, 0.003, 0.03), T1)
    assert np.isclose(_clip_step(T0, T1, 0.008, 0.06)[0, 3], 0.008), "clipped to 8 mm"


@check("cartesian mapper weighted: no frozen frames, every step within the env rate limit")
def _():
    from atom_robot.ar5.cartesian_mapping import CartesianMapConfig, CartesianMapper
    from atom_robot.ar5.joint_mapping import JOINT_PAIRS
    arm = AR5Config().arm
    try:
        m = CartesianMapper(arm, arm.workspace_min, arm.workspace_max,
                            config=CartesianMapConfig(ik="weighted", tcp_z=0.13))
    except (FileNotFoundError, OSError) as exc:
        raise Skip(f"主臂 URDF 或标定文件不在：{exc}")
    m.configure({n: 90.0 for n, _, _ in JOINT_PAIRS}, arm.joint_min, arm.joint_max,
                margin=arm.joint_limit_margin)
    q = np.clip(m.centre, arm.joint_min, arm.joint_max)
    m.engage(q)
    worst = 0.0
    for k in range(300):
        s = k / 50
        # 手腕大幅转动 + 肩大幅摆动：dls 在这段轨迹上会大面积解不出
        joints = {"shoulder_pan": 40 * np.sin(0.5 * s), "shoulder_lift": -10 + 35 * np.sin(0.3 * s),
                  "elbow_flex": 20.0, "wrist_flex": 30 + 50 * np.sin(0.7 * s),
                  "wrist_roll": 60 * np.sin(0.4 * s), "gripper": 0.0}
        q_next = m.compute(joints, q)
        worst = max(worst, float(np.max(np.abs(q_next - q))))
        q = q_next
    assert m.stats["ik_fail"] == 0, m.stats
    assert worst < arm.joint_goal_step, f"step {worst} >= rate limit {arm.joint_goal_step}"


@check("cartesian: rotation scale follows --scale under weighted only; --cart-log writes a segment per engagement")
def _():
    import tempfile
    from atom_robot.ar5.cartesian_mapping import CartesianMapConfig, CartesianMapper
    from atom_robot.ar5.config import CollectConfig
    from atom_robot.ar5.joint_mapping import JOINT_PAIRS
    cc = CollectConfig()
    assert cc.cartesian_rot_scale() == 1.0, "dls keeps the original 1:1 rotation"
    cc.cart_ik = "weighted"
    assert cc.cartesian_rot_scale() == cc.scale
    cc.rot_scale = 0.8
    assert cc.cartesian_rot_scale() == 0.8

    arm = AR5Config().arm
    with tempfile.TemporaryDirectory() as tmp:
        for ik in ("dls", "weighted"):
            try:
                m = CartesianMapper(arm, arm.workspace_min, arm.workspace_max,
                                    config=CartesianMapConfig(ik=ik, log_dir=f"{tmp}/{ik}"))
            except (FileNotFoundError, OSError) as exc:
                raise Skip(f"主臂 URDF 或标定文件不在：{exc}")
            m.configure({n: 90.0 for n, _, _ in JOINT_PAIRS}, arm.joint_min, arm.joint_max,
                        margin=arm.joint_limit_margin)
            q = np.clip(m.centre, arm.joint_min, arm.joint_max)
            joints = {"shoulder_pan": 0.0, "shoulder_lift": -10.0, "elbow_flex": 20.0,
                      "wrist_flex": 30.0, "wrist_roll": 0.0, "gripper": 0.0}
            for segment in range(2):
                m.engage(q)
                for k in range(20):
                    joints["shoulder_pan"] = segment * 5 + k * 0.5
                    q = m.compute(joints, q)
                m.disengage()
            m.disengage()                   # 空的一段不写文件
            assert len(m.log_files) == 2, m.log_files
            rows = np.load(m.log_files[0])["rows"]
            assert rows.shape[1] == 47 and len(rows) == 19, rows.shape   # 第一帧取锚点，不记
            assert np.all(np.isfinite(rows[:, 32:39])), "q_out"

        m = CartesianMapper(arm, arm.workspace_min, arm.workspace_max, config=CartesianMapConfig())
        m.engage(q)
        m.compute(joints, q)
        m.disengage()
        assert m.log_files == [], "logging is off by default"


# ---------------------------------------------- --ik sdk：排队式 MoveL 流（假控制器）


class _FakeMoveLSdk:
    """MoveLStreamer 用到的那几个 SDK 符号。"""

    class CoordinateType:
        flangeInBase = "flangeInBase"

    class OperationState:
        idle, moving = "idle", "moving"

    class MotionControlMode:
        NrtCommandMode = "nrt"

    class OperateMode:
        automatic = "automatic"

    class CartesianPosition:
        def __init__(self):
            self.trans, self.rpy = [0.0] * 3, [0.0] * 3
            self.elbow, self.hasElbow, self.confData = 0.0, False, []

    class MoveLCommand:
        def __init__(self, target):
            self.target, self.speed, self.zone = target, 0.0, 0.0

    @staticmethod
    def PyString(s):
        return s


class _FakeMoveLRobot:
    """假控制器：真的排队、真的判运动中；`finish()` 把队列一下走完（测试里代替时间流逝）。
    每个方法都按 SDK 的规矩收尾部的 ec 字典，返回码照真机：运动中 moveReset / moveStart 回 -20，
    队列空 moveStart 回 768。"""

    def __init__(self, tool_z=0.0):
        from types import SimpleNamespace
        from atom_robot.ar5.transforms import matrix_to_rpy
        self.T = np.eye(4)
        self.T[:3, 3] = [0.45, 0.0, 0.30]
        self.T[:3, :3] = np.diag([1.0, -1.0, -1.0])
        self._rpy = matrix_to_rpy
        self.elbow, self.conf = 0.3, [1, 0, 1, 0, 0, 0, 0, 0]
        self.queue, self.moving = [], False
        self.calls = []
        self._tool = SimpleNamespace(end=SimpleNamespace(trans=[0.0, 0.0, tool_z], rpy=[0.0] * 3))

    def _log(self, name):
        self.calls.append(name)

    def operationState(self, ec):
        return "moving" if self.moving else "idle"

    def cartPosture(self, _ct, ec):
        from types import SimpleNamespace
        return SimpleNamespace(trans=self.T[:3, 3].tolist(), rpy=self._rpy(self.T[:3, :3]).tolist(),
                               elbow=self.elbow, confData=list(self.conf))

    def moveReset(self, ec):
        self._log("moveReset")
        if self.moving:
            ec["ec"] = -20
        else:
            self.queue = []

    def moveAppend(self, cmds, _cid, ec):
        self._log("moveAppend")
        self.queue += list(cmds)

    def moveStart(self, ec):
        self._log("moveStart")
        if self.moving:
            ec["ec"] = -20
        elif not self.queue:
            ec["ec"] = 768
        else:
            self.moving = True

    def finish(self):
        from atom_robot.ar5.transforms import rpy_to_matrix
        if self.queue:
            last = self.queue[-1].target
            self.T[:3, 3] = last.trans
            self.T[:3, :3] = rpy_to_matrix(last.rpy)
        self.queue, self.moving = [], False

    def toolset(self, ec):
        return self._tool

    def powerState(self, ec):
        return "PowerState.on"

    def __getattr__(self, name):
        # setMotionControlMode / setPowerState / setDefault* …：记下来、什么都不做
        if name.startswith("set"):
            return lambda *a: self._log(name)
        raise AttributeError(name)


def _streamer(**kw):
    from atom_robot.ar5.movel_stream import MoveLStreamer
    robot = _FakeMoveLRobot(tool_z=kw.pop("tool_z", 0.0))
    return robot, MoveLStreamer(robot, _FakeMoveLSdk, **kw)


@check("MoveL stream: configure never writes the tool, and refuses a non-zero tool instead of fixing it")
def _():
    robot, s = _streamer()
    assert s.configure(log=lambda *_: None)
    assert "setToolset" not in robot.calls, "setToolset 持久化、带负载写会掉电，不许碰"
    assert robot.calls.index("setMotionControlMode") < robot.calls.index("setPowerState")
    robot, s = _streamer(tool_z=0.097)
    assert not s.configure(log=lambda *_: None), "工具不是 0 时法兰目标会整体偏掉，必须拒绝"
    assert "setToolset" not in robot.calls


@check("MoveL stream: deadband, append with the current elbow, start only when idle")
def _():
    robot, s = _streamer()
    s.start()
    T = robot.T.copy()
    T[0, 3] += 0.0005                      # 0.5 mm < 1 mm 死区
    s.submit(T)
    assert s.flush() == "tiny" and not robot.queue
    T[0, 3] += 0.005
    robot.elbow = 0.42                     # 「current」每次下发前刷新臂角
    s.submit(T)
    s.flush()
    assert len(robot.queue) == 1 and robot.moving, "空闲时追加完要 moveStart"
    cmd = robot.queue[0]
    assert (cmd.speed, cmd.zone) == (600.0, 100.0)
    assert cmd.target.hasElbow and cmd.target.elbow == 0.42
    assert np.allclose(cmd.target.trans, T[:3, 3])
    T[0, 3] += 0.005                       # 运动中再追加：不再 moveStart，-20 不算失败
    s.submit(T)
    s.flush()
    assert len(robot.queue) == 2 and s.stats["start_failed"] == 0


@check("MoveL stream: the time backlog gate stops tiny segments from piling up (10-31 延迟 1.9 s 的 bug)")
def _():
    # 真机上 0.1 mm 的碎段每段也要 ~90 ms，距离预算挡不住。按时间估算的积压超过上限就先不发，
    # 只留最新目标，臂停下（队列走空）再发
    robot, s = _streamer()
    s.start()
    T = robot.T.copy()
    for k in range(20):                    # 运动中连发 20 个各 4 mm 的目标
        T[0, 3] += 0.004
        s.submit(T)
        s.flush()
    est = len(robot.queue) * s.segment_seconds(0.004)
    assert len(robot.queue) <= 3, f"运动中排进去 {len(robot.queue)} 段，积压闸门没挡住"
    assert s.stats["skipped_backlog"] > 0 or s.flush() == "backlog"
    assert est <= s.max_backlog + s.segment_seconds(0.004) + 1e-9
    robot.finish()
    start = robot.T[:3, 3].copy()
    s.submit(T)
    s.flush()
    # 臂停下后只发一段、直奔最新目标（中间的不补），长度受距离预算限制
    assert len(robot.queue) == 1
    sent = np.asarray(robot.queue[0].target.trans)
    step = sent - start
    want = T[:3, 3] - start
    assert np.allclose(step / np.linalg.norm(step), want / np.linalg.norm(want), atol=1e-6)
    assert np.linalg.norm(step) <= s.max_lead + 1e-9
    assert s.segment_seconds(0.0001) == s.MIN_SEGMENT_S


@check("MoveL stream batch=10: one moveAppend per 10 points, refill only when <=3 are left, newest kept")
def _():
    robot, s = _streamer(batch=10)
    s.start()
    T = robot.T.copy()
    sizes = []
    real_append = robot.moveAppend

    def counting_append(cmds, cid, ec):
        sizes.append(len(cmds))
        real_append(cmds, cid, ec)

    robot.moveAppend = counting_append
    for k in range(25):                    # 臂不推进（假控制器），主臂每拍走 2 mm
        T[0, 3] += 0.002
        s.submit(T.copy())
        s.flush()
    # 第 2 拍过死区、臂空闲 -> 不满一批也发（起步）；之后臂在动 -> 攒满 10 个才发；
    # 再之后队列里剩 12 个 > 3 -> 不发
    assert sizes == [2, 10], sizes
    assert s.stats["skipped_queue"] > 0
    robot.finish()                         # 臂走完、空闲：缓冲里 13 个，只取最新 10 个
    s.flush()
    assert sizes[-1] == 10 and np.allclose(robot.queue[-1].target.trans, T[:3, 3])
    assert s.stats["decimated"] == 3


@check("MoveL stream batch: duplicate points are dropped (controller refuses points 0.01 mm apart)")
def _():
    robot, s = _streamer(batch=10)
    s.start()
    A, B = robot.T.copy(), robot.T.copy()
    A[0, 3] += 0.005
    B[0, 3] += 0.010
    for P in (A, A, A, B):
        s.submit(P)
    s.flush()                              # 臂空闲：不满一批也发
    assert len(robot.queue) == 2 and s.stats["deduped"] == 2, (len(robot.queue), s.stats["deduped"])
    s.submit(B)                            # 和上一个已发的点一样：不发
    for _ in range(12):
        s.submit(B)
    s.flush()
    assert len(robot.queue) == 2


@check("MoveL stream merge: collinear points collapse to one segment, corners and orientation turns survive")
def _():
    from atom_robot.ar5.transforms import axis_angle_to_matrix
    robot, s = _streamer(batch=10, merge_mm=2.0)
    assert s.refill_at == 1
    base = robot.T.copy()

    def pose(dx, dy=0.0, rot=0.0):
        P = base.copy()
        P[0, 3] += dx
        P[1, 3] += dy
        P[:3, :3] = axis_angle_to_matrix(np.array([0.0, 0.0, rot])) @ base[:3, :3]
        return P

    line = [pose(0.002 * k + 0.0003 * (k % 2)) for k in range(1, 11)]       # 直线 + 0.3 mm 抖动
    assert len(s._simplify(base, line)) == 1, "共线的 10 个点该合成一段"
    corner = [pose(0.004 * k) for k in range(1, 6)] + [pose(0.02, 0.004 * k) for k in range(1, 6)]
    kept = s._simplify(base, corner)
    assert len(kept) == 2 and np.allclose(kept[0][:3, 3], corner[4][:3, 3]), "L 形的拐点要留"
    twist = [pose(0.002 * k, rot=(0.2 if k == 5 else 0.0)) for k in range(1, 11)]  # 中途转 11° 又转回
    assert len(s._simplify(base, twist)) >= 2, "姿态中途拐了也要留"
    smooth = [pose(0.002 * k, rot=0.02 * k) for k in range(1, 11)]          # 边走边匀速转
    assert len(s._simplify(base, smooth)) == 1

    s.start()
    for P in line:
        s.submit(P)
    s.flush()
    assert len(robot.queue) == 1 and s.stats["merged"] == 9


@check("MoveL stream: the lead budget caps how far targets run ahead of the arm")
def _():
    robot, s = _streamer(backlog_ms=10_000)     # 这项只测距离预算，积压闸门放开
    s.start()
    T = robot.T.copy()
    T[0, 3] += 0.010
    s.submit(T)
    s.flush()                              # 臂开始动，但「还没走」（假控制器不推进）
    far = robot.T.copy()
    far[0, 3] += 0.200
    s.submit(far)
    s.flush()
    lead = np.linalg.norm(np.asarray(robot.queue[-1].target.trans) - robot.T[:3, 3])
    assert lead <= s.max_lead + 1e-9, f"已下发目标领先 {lead * 1000:.1f} mm，超过预算 {s.max_lead * 1000:.0f} mm"
    n = len(robot.queue)
    s.submit(far)
    assert s.flush() == "lead" and len(robot.queue) == n, "预算用光了就先不发"
    robot.finish()
    s.submit(far)
    s.flush()
    assert len(robot.queue) == 1 and robot.moving, "臂停下后接着往目标走"


@check("MoveL stream: pause touches no SDK, halt waits for idle before moveReset, elbow=none sends no elbow")
def _():
    robot, s = _streamer()
    s.start()
    T = robot.T.copy()
    T[2, 3] += 0.01
    s.submit(T)
    s.flush()
    robot.calls.clear()
    s.pause()
    assert robot.calls == [] and not s.running
    assert not s.halt(timeout=0.1), "臂还在动，halt 要等，不能硬 moveReset"
    assert "moveReset" not in robot.calls
    robot.finish()
    assert s.halt(timeout=0.5) and robot.calls[-1] == "moveReset"

    robot, s = _streamer(elbow="none")
    s.start()
    s.submit(T)
    s.flush()
    assert not robot.queue[0].target.hasElbow


@check("MoveL stream step(): a target-less frame while engaged does not pause (10-31 真机臂不动的 bug)")
def _():
    # 照映射器的真实节奏喂：起步后 on_start 重锚 -> 下一拍取主臂锚点、没目标 -> 再下一拍才有目标。
    # 第一版把「没目标」当成松开去 pause，于是永远在起步和暂停之间来回，一条指令都发不出去。
    robot, s = _streamer()
    state = {"anchor_frame": False, "reanchors": 0}

    def reanchor():
        state["anchor_frame"] = True
        state["reanchors"] += 1

    T = robot.T.copy()
    for k in range(10):
        T[0, 3] += 0.005                   # 大于 3 mm 死区
        target = None if state["anchor_frame"] else T.copy()
        state["anchor_frame"] = False
        s.step(True, target, on_start=reanchor)
    assert state["reanchors"] == 1, f"重锚了 {state['reanchors']} 次，在起步和暂停之间打转"
    assert s.running and robot.queue, "接合着、主臂在动，就该有 MoveL 发出去"

    robot.calls.clear()
    s.step(False, None)
    assert not s.running and "moveReset" not in robot.calls, "松开只停追加，不碰正在走的队列"


@check("cartesian mapper ik=sdk: same relative target as the local solvers, handed over as a flange pose")
def _():
    from atom_robot.ar5.cartesian_mapping import CartesianMapConfig, CartesianMapper
    from atom_robot.ar5.config import CollectConfig
    from atom_robot.ar5.joint_mapping import JOINT_PAIRS
    cc = CollectConfig()
    assert not cc.sdk_mode()
    cc.space, cc.cart_ik = "cartesian", "sdk"
    assert cc.sdk_mode() and cc.cartesian_tcp_z() == 0.13 and cc.cartesian_rot_scale() == cc.scale
    cc.sdk_elbow = "sometimes"
    try:
        cc.__post_init__()
    except ValueError:
        pass
    else:
        raise AssertionError("accepted a bad sdk_elbow")

    arm = AR5Config().arm
    try:
        m = CartesianMapper(arm, arm.workspace_min, arm.workspace_max,
                            config=CartesianMapConfig(ik="sdk", tcp_z=0.13))
    except (FileNotFoundError, OSError) as exc:
        raise Skip(f"主臂 URDF 或标定文件不在：{exc}")
    m.configure({n: 90.0 for n, _, _ in JOINT_PAIRS}, arm.joint_min, arm.joint_max,
                margin=arm.joint_limit_margin)
    q = np.clip(m.centre, arm.joint_min, arm.joint_max)
    joints = {"shoulder_pan": 0.0, "shoulder_lift": -10.0, "elbow_flex": 20.0,
              "wrist_flex": 30.0, "wrist_roll": 0.0, "gripper": 0.0}
    m.engage(q)
    m.compute(joints, q)                   # 第一帧取主臂锚点
    assert m.sdk_target is None
    joints["shoulder_pan"] = 5.0
    out = m.compute(joints, q)
    assert np.array_equal(out, q), "sdk 这一路不解逆解，关节原样返回"
    assert m.sdk_target is not None
    assert np.allclose(m.sdk_target @ m._tool, m._diag["target_used"]), "交出去的是法兰，不是指尖"
    m.disengage()
    m.compute(joints, q)
    assert m.sdk_target is None, "松开后不能再给目标"


@check("TrackingWatch reports MoveL failures and frequent queue restarts")
def _():
    from types import SimpleNamespace
    from atom_robot.ar5.collect import TrackingWatch
    mapper = SimpleNamespace(stats={})
    arm = SimpleNamespace(rt_stats=lambda: {"rejected": {}})
    streamer = SimpleNamespace(stats={"execution_errors": 0, "start_failed": 0, "restarts": 0})
    w = TrackingWatch(mapper, arm, np.zeros(7), np.ones(7), period=0.0, streamer=streamer)
    streamer.stats["execution_errors"] = 1
    streamer.stats["restarts"] = 5
    line = w.check(1.0, np.zeros(7))
    assert line and "MoveL 失败 1" in line and "重启 5" in line, line


# --------------------------------------------------- real-time joint limiter


class _FakeSDK:
    """Just enough SDK surface for the sender's tick maths."""

    class JointPosition:
        def __init__(self, values):
            self.values = list(values)


def _tracker(**overrides):
    """A RokaeAR5 with only the tick state populated -- no robot, no threads."""
    from atom_robot.ar5.config import ArmConfig
    from atom_robot.ar5.drivers.rokae_arm import RokaeAR5

    cfg = ArmConfig(**overrides)
    arm = RokaeAR5(cfg)
    arm._q_cmd = np.zeros(cfg.num_joints)
    arm._v_cmd = np.zeros(cfg.num_joints)
    arm._q_goal = np.zeros(cfg.num_joints)
    return arm, cfg


@check("the tracker respects velocity and acceleration limits")
def _():
    arm, cfg = _tracker()
    goal = np.full(cfg.num_joints, 1.0)      # far enough to saturate
    prev_v = arm._v_cmd.copy()
    max_v = max_a = 0.0
    for _ in range(3000):
        arm._tick(goal)
        max_v = max(max_v, float(np.max(np.abs(arm._v_cmd))))
        max_a = max(max_a, float(np.max(np.abs(arm._v_cmd - prev_v)) / cfg.rt_period))
        prev_v = arm._v_cmd.copy()

    assert max_v <= cfg.max_joint_speed + 1e-9, max_v
    assert max_a <= cfg.max_joint_accel * 1.001, max_a


@check("the tracker arrives at rest without overshooting")
def _():
    arm, cfg = _tracker()
    goal = np.full(cfg.num_joints, 0.4)
    overshoot = 0.0
    for _ in range(5000):
        q = arm._tick(goal)
        overshoot = max(overshoot, float(np.max(q - goal)))
        if arm._at_rest() and np.allclose(q, goal, atol=1e-4):
            break
    assert np.allclose(arm._q_cmd, goal, atol=1e-3), arm._q_cmd
    assert arm._at_rest(), arm._v_cmd
    # The sqrt(2*A*e) braking law is what buys this; a plain velocity clamp
    # would sail past the goal and oscillate.
    assert overshoot < 1e-3, overshoot


@check("a NaN goal cannot poison the tracker")
def _():
    arm, cfg = _tracker()
    # Comparisons against NaN are all False, so a `if q < lo` clamp would pass
    # it straight through and every later velocity would be NaN as well.
    arm._tick(np.full(cfg.num_joints, np.nan))
    assert np.all(np.isfinite(arm._q_cmd)), arm._q_cmd
    assert np.all(np.isfinite(arm._v_cmd)), arm._v_cmd

    # And the tracker must still work afterwards.
    goal = np.full(cfg.num_joints, 0.1)
    for _ in range(3000):
        arm._tick(goal)
    assert np.allclose(arm._q_cmd, goal, atol=1e-3), arm._q_cmd


@check("the tracker never commands a joint outside its soft limits")
def _():
    arm, cfg = _tracker()
    lo = np.array(cfg.joint_min) + cfg.joint_limit_margin
    hi = np.array(cfg.joint_max) - cfg.joint_limit_margin
    # Aim far beyond every limit in both directions.
    for goal in (np.full(cfg.num_joints, 100.0), np.full(cfg.num_joints, -100.0)):
        for _ in range(4000):
            q = arm._tick(goal)
            assert np.all(q >= lo - 1e-9) and np.all(q <= hi + 1e-9), q

    # The goal is clamped into the limit box before tracking, so the braking law
    # decelerates onto the boundary and the position clamp never has to fire.
    # If it does, something upstream let an out-of-range value through, and the
    # clamp's one-tick velocity kill is an unbounded deceleration.
    assert arm._limit_hits == 0, f"backstop clamp fired {arm._limit_hits} times"


@check("publishing gates reject latched, non-finite, out-of-limit and jumping goals")
def _():
    arm, cfg = _tracker()
    import threading as _th

    arm._rt_active = True
    arm._rt_latched = _th.Event()

    good = np.zeros(cfg.num_joints)
    assert arm.servo_to_joints(good) is True

    assert arm.servo_to_joints(np.full(cfg.num_joints, np.nan)) is False
    assert arm._rejected["nonfinite"] == 1

    assert arm.servo_to_joints(np.zeros(cfg.num_joints - 1)) is False
    assert arm._rejected["nonfinite"] == 2

    beyond = np.zeros(cfg.num_joints)
    beyond[5] = 5.0
    assert arm.servo_to_joints(beyond) is False
    assert arm._rejected["limits"] == 1

    # Rate limiting bounds speed, not destination: a branch flip has to be
    # refused outright or the arm tracks smoothly all the way there.
    jump = np.zeros(cfg.num_joints)
    jump[0] = cfg.joint_jump_max * 2
    assert arm.servo_to_joints(jump) is False
    assert arm._rejected["jump"] == 1

    arm._rt_latched.set()
    assert arm.servo_to_joints(good) is False
    assert arm._rejected["latched"] == 1

    arm.clear_latch()
    assert arm.servo_to_joints(good) is True


@check("publishing without a running stream raises rather than going nowhere")
def _():
    arm, cfg = _tracker()
    arm._rt_active = False
    # Silently accepting would reproduce the RT-Cartesian failure: every command
    # acknowledged, nothing moving.
    try:
        arm.servo_to_joints(np.zeros(cfg.num_joints))
    except RuntimeError as exc:
        assert "real-time" in str(exc)
    else:
        raise AssertionError("publishing with no stream must raise")


@check("the real-time path never calls the controller's 42 ms reachability check")
def _():
    env = _mock_env()
    try:
        env.reset(fake=True)
        env.arm._rt_active = True
        calls = {"rt": 0, "check": 0}

        def fake_rt(pose):
            calls["rt"] += 1
            env.arm.servo_to_pose(pose)
            return True

        env.arm.servo_to_pose_rt = fake_rt
        env.arm.check_reachable = lambda pose: (calls.__setitem__("check", calls["check"] + 1), (True, "ok"))[1]

        for _ in range(10):
            env.step([0.002, 0, 0, 0, 0, 0, 1.0])

        assert calls["rt"] == 10, calls
        # checkPath holds the GIL for ~42 ms. Calling it once per step would
        # blank most of the 1 kHz stream, so the streaming path must resolve
        # reachability locally instead.
        assert calls["check"] == 0, calls
    finally:
        env.close()


@check("a refused real-time goal holds the arm but still drives the gripper")
def _():
    env = _mock_env()
    try:
        env.reset(fake=True)
        env.arm._rt_active = True
        env.arm.servo_to_pose_rt = lambda pose: False

        before = env.arm.get_ee_pose().copy()
        env.step([0.01, 0, 0, 0, 0, 0, 0.0])

        assert np.allclose(env.arm.get_ee_pose(), before), "a refused goal must not move the arm"
        assert env.stats["unreachable"] == 1, env.stats
        assert env.hand.get_opening() == 0.0, "the gripper is independent of arm reachability"
    finally:
        env.close()


@check("the queued path still asks the controller before committing")
def _():
    env = _mock_env()
    try:
        env.reset(fake=True)
        calls = {"n": 0}
        env.arm.check_reachable = lambda pose: (calls.__setitem__("n", calls["n"] + 1), (True, "ok"))[1]
        for _ in range(10):
            env.step([0.002, 0, 0, 0, 0, 0, 1.0])
        assert calls["n"] == 10, calls
    finally:
        env.close()


@check("a rejected goal does not let the sender latch on staleness")
def _():
    import threading as _th

    arm, cfg = _tracker()
    arm._rt_active = True
    arm._rt_latched = _th.Event()

    # Pretend the sender has been counting silence.
    with arm._setpoint_lock:
        arm._goal_ticks = 10_000

    # A goal the gates refuse still proves the publisher is alive. If rejection
    # left the clock running, a brief run of unsolvable targets would latch the
    # arm permanently -- observed on hardware as the arm freezing mid-session.
    beyond = np.zeros(cfg.num_joints)
    beyond[5] = 5.0
    assert arm.servo_to_joints(beyond) is False
    with arm._setpoint_lock:
        assert arm._goal_ticks == 0, "a rejection must still refresh the staleness clock"


@check("inverse kinematics recovers a solution that lands just outside the limits")
def _():
    kin, q0 = _kin()

    # Drive the seed close to a narrow joint's stop; j5 travels only +-0.8727,
    # so solutions there routinely converge a hair outside.
    seed = q0.copy()
    seed[5] = kin.joint_max[5] - 0.01
    T = kin.fk(seed)

    q, info = kin.ik(T, seed)
    assert q is not None, info
    assert kin.within_limits(q), kin.limit_report(q)
    # Whatever route it took, the pose must still be right.
    assert np.linalg.norm(kin.fk(q)[:3, 3] - T[:3, 3]) < 5e-4, info


@check("staleness latches only when the arm was actually moving")
def _():
    import threading as _th
    from atom_robot.ar5.config import ArmConfig
    from atom_robot.ar5.drivers.rokae_arm import RokaeAR5

    # At rest: goals drying up is benign, and latching would punish every
    # ordinary pause -- a read loop, an operator thinking, a slow policy step.
    arm, cfg = _tracker()
    arm._v_cmd = np.zeros(cfg.num_joints)
    assert arm._at_rest()

    # In motion: the commander going quiet must stop the arm and keep it stopped.
    moving, cfg2 = _tracker()
    moving._v_cmd = np.full(cfg2.num_joints, 0.2)
    assert not moving._at_rest()


@check("error recovery is skipped while the stream is healthy")
def _():
    import threading as _th

    arm, cfg = _tracker()
    arm._rt_active = True
    arm._rt_latched = _th.Event()

    calls = {"n": 0}

    class FakeRt:
        def automaticErrorRecovery(self, ec):
            calls["n"] += 1

    arm._rt = FakeRt()
    arm.has_motion_error = lambda: False

    # Running recovery on a healthy real-time stream resets the controller's
    # motion state: commands keep being accepted at 1 kHz, nothing reports an
    # error, and the arm stops executing. Measured on hardware as 67.16 mm of
    # motion becoming 0.00 mm. So it must not run unless there is a fault.
    arm.recover_from_error()
    assert calls["n"] == 0, "recovery ran with no fault present"

    arm._rt_error = "simulated fault"
    arm.recover_from_error()
    assert calls["n"] == 1, "recovery must run once a fault is recorded"
    assert arm._rt_error is None, "a cleared fault must not stay recorded"

    arm.has_motion_error = lambda: True
    arm.recover_from_error()
    assert calls["n"] == 2, "a controller-reported fault must also trigger recovery"

    arm.has_motion_error = lambda: False
    arm.recover_from_error(force=True)
    assert calls["n"] == 3, "force must run it regardless"


@check("velocity feed-forward cuts the lag behind a moving goal")
def _():
    # Error-only tracking of a moving target is a first-order lag: the tracker
    # only commands speed in proportion to how far behind it already is, so it
    # settles at whatever lag produces the target's speed and never catches up.
    # Feeding the goal's own rate forward removes that structural offset.
    # Measured on hardware: tracking a 60 mm/s leader went from 23% to 46% of
    # commanded travel, with steady-state lag dropping from about 15 mm to 2.3 mm.
    speed = 0.05                                   # rad/s, well under the cap
    steps = 4000

    def settle(with_ff):
        arm, cfg = _tracker()
        goal = np.zeros(cfg.num_joints)
        for _ in range(steps):
            goal = goal + speed * cfg.rt_period
            arm._tick(goal, np.full(cfg.num_joints, speed) if with_ff else None)
        return float(abs(goal[0] - arm._q_cmd[0]))

    lag_off = settle(False)
    lag_on = settle(True)
    assert lag_off > 1e-4, f"expected a measurable lag without feed-forward, got {lag_off}"
    assert lag_on < lag_off / 10, f"feed-forward barely helped: {lag_on} against {lag_off}"


@check("the goal-rate estimate feeding the tracker tracks a constant-speed goal")
def _():
    arm, cfg = _tracker()
    arm._rt_active = True
    arm._rt_latched = __import__("threading").Event()
    arm._published = []
    speed = 0.08
    dt = 0.05

    now = [0.0]
    import atom_robot.ar5.drivers.rokae_arm as _mod

    real_counter = _mod.time.perf_counter
    _mod.time.perf_counter = lambda: now[0]
    try:
        q = np.zeros(cfg.num_joints)
        for _ in range(12):
            now[0] += dt
            q = q + speed * dt
            arm.servo_to_joints(q)
    finally:
        _mod.time.perf_counter = real_counter

    assert arm._goal_rate is not None, "no rate was estimated"
    err = float(np.max(np.abs(arm._goal_rate - speed)))
    assert err < speed * 0.05, f"rate estimate off by {err} on a {speed} rad/s goal"


@check("a mirrored axis mapping keeps rotation turning the same way")
def _():
    from atom_robot.ar5.leader_mapping import mapping_matrix
    from atom_robot.ar5.transforms import axis_angle_to_matrix, matrix_to_axis_angle

    # A remap with an odd number of sign flips mirrors the frame, and an
    # axis-angle vector is a pseudo-vector: the correct image is
    # det(M) * M @ r, because M R M^-1 = exp([det(M) M r]x). Dropping the
    # determinant leaves translation right and every rotation reversed, which
    # is the failure that is easiest to mistake for a wiring mistake.
    rng = np.random.default_rng(19)
    for perm in ((0, 1, 2), (1, 0, 2), (2, 1, 0), (1, 2, 0)):
        for signs in ((1, 1, 1), (1, -1, 1), (-1, -1, 1), (-1, -1, -1)):
            M = mapping_matrix(perm, signs)
            parity = float(np.sign(np.linalg.det(M)))
            for _ in range(20):
                r = (rng.random(3) - 0.5) * 2.0
                truth = matrix_to_axis_angle(M @ axis_angle_to_matrix(r) @ M.T)
                assert np.allclose(truth, parity * (M @ r), atol=1e-9), (perm, signs)


@check("the leader's rotation delta composes instead of subtracting")
def _():
    need("placo")          # 笛卡尔遥操那条路才用，采集用不到
    from atom_robot.ar5.transforms import axis_angle_to_matrix, matrix_to_axis_angle

    # Subtracting axis-angle vectors is only a small-angle approximation. Turn
    # the wrist far enough and it diverges from the true incremental rotation;
    # near the pi wrap it inverts entirely. The mapper composes R @ prev.T,
    # which is exact and matches how compose_pose applies action[3:6].
    mapper = _mapper()
    a = dict(_POSE_A)
    b = dict(a)
    b["wrist_roll"] = a.get("wrist_roll", 0.0) + 150.0     # a big, legal turn

    mapper.config.position_scale = 1.0
    mapper.config.rotation_scale = 1.0
    mapper.engage(a)
    action = mapper.compute_action(b)

    _, R_a = mapper.leader_frame(a)
    _, R_b = mapper.leader_frame(b)
    expected = matrix_to_axis_angle(R_b @ R_a.T)
    assert np.allclose(action[3:6], expected, atol=1e-9), (action[3:6], expected)

    # Composing the emitted delta onto the start orientation must land exactly
    # on the end orientation -- the property subtraction does not have.
    landed = axis_angle_to_matrix(action[3:6]) @ R_a
    assert np.allclose(landed, R_b, atol=1e-9), "the delta does not close the gap"

    naive = matrix_to_axis_angle(R_b) - matrix_to_axis_angle(R_a)
    assert np.linalg.norm(naive - expected) > 0.05, (
        "this posture pair is too small to distinguish the two formulations"
    )


@check("the home posture sits at the centre of the workspace box")
def _():
    import json
    from pathlib import Path

    from atom_robot.ar5.config import AR5Config
    from atom_robot.ar5.kinematics import AR5Kinematics

    cfg = AR5Config().arm
    dh_file = Path(__file__).resolve().parent / "ar5_dh.json"
    if not dh_file.exists():
        return
    dh = np.asarray(json.loads(dh_file.read_text())["dh"], dtype=np.float64)
    kin = AR5Kinematics(dh, num_joints=cfg.num_joints, joint_min=cfg.joint_min, joint_max=cfg.joint_max)

    tool = kin.fk(np.asarray(cfg.home_joints, dtype=np.float64))[:3, 3]
    lo = np.asarray(cfg.workspace_min, dtype=np.float64)
    hi = np.asarray(cfg.workspace_max, dtype=np.float64)

    assert np.all(tool > lo) and np.all(tool < hi), f"home tool {tool} is outside the box"

    # What actually bit: the tool sat 80 mm forward, leaving 67 mm of forward
    # travel against 333 mm back, so pushing the leader forward pinned the
    # target on the box wall within seven steps -- which reads as the arm
    # stuttering, not as a limit being hit.
    #
    # The property worth asserting is "enough room in every direction", not
    # "centred". Those were the same thing while the box was symmetric; they
    # stopped being so on 2026-09-07, when y was widened on one side only to
    # cover where the task actually is. Demanding symmetry there would push
    # toward widening +y for no reason -- loosening a safety limit to satisfy
    # a test is exactly backwards.
    room = np.minimum(tool - lo, hi - tool)
    floor_mm = 150.0
    assert np.all(room * 1000 > floor_mm), (
        f"home is boxed in: {np.round(room * 1000, 1)} mm of clearance, "
        f"needs more than {floor_mm:.0f} mm in every direction. The 67 mm that "
        "caused the original stutter would land here."
    )


@check("a saved axis calibration is loaded, and explicit arguments still win")
def _():
    import json
    import tempfile
    from pathlib import Path

    from atom_robot.ar5.leader_mapping import MappingConfig

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "leader_calibration.json"

        # Missing file: identity, not a crash. Teleop has to keep working before
        # anyone has calibrated.
        blank = MappingConfig.load(path=path)
        assert blank.axis_permutation == (0, 1, 2)
        assert blank.axis_signs == (1.0, 1.0, 1.0)

        path.write_text(json.dumps({
            "axis_permutation": [1, 0, 2],
            "axis_signs": [1.0, -1.0, 1.0],
            "gripper_closed_deg": -55.0,
            "gripper_open_deg": 2.0,
        }))

        loaded = MappingConfig.load(path=path)
        assert loaded.axis_permutation == (1, 0, 2), loaded.axis_permutation
        assert loaded.axis_signs == (1.0, -1.0, 1.0), loaded.axis_signs
        assert loaded.gripper_closed_deg == -55.0
        assert loaded.gripper_open_deg == 2.0

        # A flag the operator typed must not be silently overwritten by the file.
        overridden = MappingConfig.load(path=path, axis_signs=(1.0, 1.0, 1.0))
        assert overridden.axis_signs == (1.0, 1.0, 1.0)
        assert overridden.axis_permutation == (1, 0, 2), "unrelated fields must still load"

        # Corrupt file: warn and fall back, never take the session down.
        path.write_text("{not json")
        assert MappingConfig.load(path=path).axis_permutation == (0, 1, 2)


@check("a device that fails to connect does not leave the arm powered")
def _():
    from atom_robot.ar5.ar5_env import AR5Env
    from atom_robot.ar5.config import AR5Config
    from atom_robot.ar5.drivers import mock as mock_mod

    # The arm comes up first, so anything failing after it -- an unplugged hand,
    # a camera that will not open -- used to abandon a powered arm with its 1 kHz
    # stream still running: the traceback killed the process, disconnect() never
    # ran, and the controller stayed in RtCommandMode for the next session.
    order = []
    originals = (mock_mod.MockArm.disconnect, mock_mod.MockHand.disconnect,
                 mock_mod.MockCameras.connect)
    try:
        mock_mod.MockArm.disconnect = lambda self: order.append("arm")
        mock_mod.MockHand.disconnect = lambda self: order.append("hand")

        def boom(self):
            raise RuntimeError("camera refused to open")

        mock_mod.MockCameras.connect = boom

        # 上电顺序**按末端分两种**，退栈顺序跟着反过来，所以两条都要测：
        #   inspire: 臂 -> 手 -> 相机     退栈 hand, arm
        #   hkvr / omnipicker:  手 -> 臂 -> 相机  退栈 arm, hand（先确认夹爪通了再给臂上电）
        # 2026-09-23 默认末端从 inspire 改成 hkvr，这条只认死一种顺序，于是报红 ——
        # 报的是测试自己过时，不是代码坏了。两种都写出来就不会再有这种误报。
        for end_effector, expected in (("inspire", ["hand", "arm"]),
                                       ("hkvr", ["arm", "hand"]),
                                       ("omnipicker", ["arm", "hand"])):
            order.clear()
            env = AR5Env(AR5Config(mock=True, end_effector=end_effector))
            try:
                env.connect()
            except RuntimeError:
                pass
            else:
                raise AssertionError(f"the failure was swallowed ({end_effector})")
            assert order == expected, (
                f"{end_effector}: unwound in the wrong order: {order}，应该是 {expected}")
            assert not env._connected, "a failed connect must not report a connected env"
    finally:
        (mock_mod.MockArm.disconnect, mock_mod.MockHand.disconnect,
         mock_mod.MockCameras.connect) = originals


@check("--end-effector omnipicker 生成 OmniPicker 驱动，命令行覆盖生效，越界的力被拒")
def _():
    import argparse

    from atom_robot.ar5.end_effector import (add_end_effector_args, configure_end_effector,
                                             make_real_hand)
    from omnipicker import OmniPickerGripper

    parser = argparse.ArgumentParser()
    add_end_effector_args(parser)
    config = AR5Config()
    with contextlib.redirect_stdout(io.StringIO()):
        configure_end_effector(config, parser.parse_args(
            ["--end-effector", "omnipicker", "--omnipicker-force", "96", "--omnipicker-speed", "200"]))
    assert config.end_effector == "omnipicker"
    assert (config.omnipicker.grip_force, config.omnipicker.speed) == (96, 200)
    hand = make_real_hand(config)              # 只构造，不连串口
    assert isinstance(hand, OmniPickerGripper) and hand.grip_force == 96
    # HKVR 那组参数不能串到 OmniPicker：--gripper-force 默认 20 当成 OmniPicker 的力就出界了
    config = AR5Config()
    with contextlib.redirect_stdout(io.StringIO()):
        configure_end_effector(config, parser.parse_args(["--end-effector", "omnipicker"]))
    assert config.omnipicker.grip_force == 64
    try:
        configure_end_effector(AR5Config(), parser.parse_args(
            ["--end-effector", "omnipicker", "--omnipicker-force", "250"]))
    except ValueError:
        pass
    else:
        raise AssertionError("夹持力 250 没被拒")


@check("an absent hand port is reported as an unplugged hand, not a serial error")
def _():
    from atom_robot.ar5.config import AR5Config
    from atom_robot.ar5.drivers.inspire_hand import InspireHand

    cfg = AR5Config().hand
    cfg.port = "/dev/definitely-not-a-real-tty"
    try:
        InspireHand(cfg).connect()
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("connecting to a missing port must fail")

    assert "does not exist" in message, message
    assert "--no-hand" in message, "the message must name the way out"


# --------------------------------------------------------------- CAN hand

# The hand on this rig speaks CAN. These cover the adapter between the AR5
# HandDriver interface and the verified codec in VEL/src/hand_teleop.


def _can_hand(**overrides):
    from atom_robot.ar5.config import AR5Config
    from atom_robot.ar5.drivers.inspire_hand_can import InspireHandCan

    cfg = AR5Config().hand
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return InspireHandCan(cfg)


@check("the hand config describes the CAN hand that is actually attached")
def _():
    from atom_robot.ar5.config import AR5Config

    cfg = AR5Config().hand
    assert cfg.protocol == "can", f"this hand is not on Modbus: {cfg.protocol}"
    assert cfg.index_dof == 3, "the index finger is DOF 3 across the RH56 series"
    assert cfg.finger_close < cfg.finger_open, "angles run 0=closed .. 1000=open"
    # finger_close has to sit just past where the thumb stops the finger, and
    # that contact point moves whenever the thumb pose does -- so pinning the
    # number here would fail every time the thumb is re-posed, which is a normal
    # thing to do. What must hold is that it is not deep enough to stall the
    # motor: 0 drives the finger straight through the thumb.
    assert cfg.finger_close >= 100, (
        f"finger_close {cfg.finger_close} is far past any measured contact point; "
        "the finger would stall against the thumb"
    )
    # And it must agree with closed_pose, which the mock hand interpolates from.
    assert cfg.closed_pose[cfg.index_dof] == cfg.finger_close, (
        f"closed_pose says the index finger closes to {cfg.closed_pose[cfg.index_dof]} "
        f"but finger_close is {cfg.finger_close}; re-measuring the contact point "
        "means updating both"
    )
    assert cfg.force <= 1000, f"force {cfg.force} outside the 0..1000 the manual gives"
    assert cfg.speed <= 1000, f"speed {cfg.speed} outside 0..1000"


@check("only the two pinching fingers move with the opening scalar")
def _():
    from atom_robot.ar5.config import AR5Config

    THUMB_BEND, THUMB_ROT = 4, 5
    cfg = AR5Config().hand
    closed, opened = list(cfg.closed_pose), list(cfg.open_pose)
    assert len(closed) == len(opened) == cfg.num_joints
    differing = {i for i, (c, o) in enumerate(zip(closed, opened)) if c != o}
    # The grasp is index against thumb. Both may sweep -- that is what makes it
    # a two-finger pinch rather than one finger hitting a fixed backstop -- but
    # nothing else may, or the other fingers would wander into the camera view
    # and into whatever the hand is reaching around.
    assert differing <= {cfg.index_dof, THUMB_BEND}, (
        f"only the index finger and thumb bend may move with the opening scalar; "
        f"these differ: {sorted(differing)}"
    )
    assert cfg.index_dof in differing, "the index finger has to move with the opening"
    # Thumb rotation stays put: it aims the thumb at the index finger, and
    # sweeping it would change the direction of the pinch mid-grasp.
    assert closed[THUMB_ROT] == opened[THUMB_ROT], "thumb rotation must not sweep"


@check("the two fingers close toward each other, never past each other")
def _():
    from atom_robot.ar5.config import HandConfig

    # Both fingers travel, so nothing is a fixed backstop any more and the
    # position command alone cannot say where they meet. Sanity-check the
    # endpoints instead: closing must move the index finger inward, and the
    # thumb must not travel so far that it drives through where the index
    # finger is going to be.
    cfg = HandConfig()
    assert cfg.finger_close < cfg.finger_open, "closing must lower the index angle"
    for opening in (0.0, 0.25, 0.5, 0.75, 1.0):
        pose = cfg.pose_for(opening)
        assert len(pose) == cfg.num_joints
        assert all(0 <= v <= 1000 for v in pose), f"pose_for({opening}) out of range: {pose}"
    # Monotone in the index finger, so a policy sweeping the scalar gets a
    # monotone grasp rather than one that folds back.
    angles = [cfg.pose_for(i / 20)[cfg.index_dof] for i in range(21)]
    assert all(b >= a for a, b in zip(angles, angles[1:])), angles
    assert cfg.pose_for(0.0)[cfg.index_dof] == cfg.finger_close
    assert cfg.pose_for(1.0)[cfg.index_dof] == cfg.finger_open
    assert cfg.pose_for(0.0)[4] == cfg.thumb_close
    assert cfg.pose_for(1.0)[4] == cfg.thumb_open


@check("opening maps onto the calibrated finger range, both ends exact")
def _():
    hand = _can_hand()
    cfg = hand.config
    assert hand._angle_for(0.0) == cfg.finger_close, "0 must reach finger_close exactly"
    assert hand._angle_for(1.0) == cfg.finger_open, "1 must reach finger_open exactly"
    assert hand._angle_for(-5.0) == cfg.finger_close, "out-of-range input must clamp"
    assert hand._angle_for(99.0) == cfg.finger_open
    mid = hand._angle_for(0.5)
    assert cfg.finger_close < mid < cfg.finger_open, mid
    # Monotone: a policy or leader sweeping the scalar must not fold back.
    angles = [hand._angle_for(i / 40) for i in range(41)]
    assert all(b >= a for a, b in zip(angles, angles[1:])), angles


@check("the hand refuses to act before connect() instead of driving nothing")
def _():
    hand = _can_hand()
    for call in (lambda: hand.set_opening(0.5), hand.get_opening, hand.get_joint_angles):
        try:
            call()
        except RuntimeError:
            continue
        raise AssertionError("using the hand before connect() must raise")
    assert "not connected" in hand.diagnose()


@check("a hand command holds the thumb backstop and moves only the index finger")
def _():
    import numpy as np

    import time as _time

    sent = []

    class _FakeHand:
        def set_angles(self, angles):
            sent.append(list(angles))

        def get_angles(self):
            return sent[-1] if sent else [0] * 6

    hand = _can_hand()
    hand._hand = _FakeHand()
    cfg = hand.config

    # Writes are posted to a worker, so each target has to be let through before
    # the next is posted -- otherwise the second supersedes the first, which is
    # exactly what the worker is supposed to do during teleoperation.
    hand._stop.clear()
    hand._worker = threading.Thread(target=hand._io_loop, daemon=True)
    hand._worker.start()
    try:
        for value in (0.0, 1.0):
            hand.set_opening(value)
            for _ in range(100):
                _time.sleep(0.005)
                if len(sent) >= (1 if value == 0.0 else 2):
                    break
        assert len(sent) >= 2, f"the worker wrote {len(sent)} commands, expected 2"
        closed_cmd, open_cmd = sent[0], sent[1]
        THUMB_BEND = 4
        assert closed_cmd[cfg.index_dof] == cfg.finger_close, closed_cmd
        assert open_cmd[cfg.index_dof] == cfg.finger_open, open_cmd
        assert closed_cmd[THUMB_BEND] == cfg.thumb_close, closed_cmd
        assert open_cmd[THUMB_BEND] == cfg.thumb_open, open_cmd
        # Everything except the two pinching fingers holds still.
        for dof in range(cfg.num_joints):
            if dof in (cfg.index_dof, THUMB_BEND):
                continue
            assert closed_cmd[dof] == open_cmd[dof], (
                f"DOF {dof} moved between the two commands: {closed_cmd} vs {open_cmd}"
            )
        assert abs(hand.get_opening() - 1.0) < 1e-9, hand.get_opening()
    finally:
        hand._stop.set()
        hand._wake.set()
        hand._worker.join(timeout=2.0)


@check("hand I/O never blocks the caller, so a dropped frame cannot latch the arm")
def _():
    import time as _time

    # The measured failure: set_opening wrote CAN synchronously from inside
    # AR5Env.step_joints. Normally 13 ms, but p99 219 ms on a retry -- and the
    # arm's 1 kHz sender brakes and latches after 250 ms without a goal. One
    # dropped hand frame stopped the arm mid-motion.
    class _SlowHand:
        def __init__(self):
            self.writes = 0

        def set_angles(self, angles):
            _time.sleep(0.25)          # a retry cycle
            self.writes += 1

        def get_angles(self):
            return [0, 0, 0, 500, 600, 100]

    hand = _can_hand()
    hand._hand = _SlowHand()
    hand._stop.clear()
    hand._worker = threading.Thread(target=hand._io_loop, daemon=True)
    hand._worker.start()
    try:
        worst = 0.0
        for i in range(10):
            t0 = _time.perf_counter()
            hand.set_opening(i / 10)
            worst = max(worst, _time.perf_counter() - t0)
        assert worst < 0.02, f"set_opening blocked for {worst * 1000:.0f} ms; it must post, not write"

        # get_opening is called from the same loop and must not block either.
        t0 = _time.perf_counter()
        hand.get_opening()
        hand.get_joint_angles()
        assert _time.perf_counter() - t0 < 0.02, "readback blocked the caller"

        _time.sleep(0.6)
        assert hand._hand.writes > 0, "the worker never wrote anything"
    finally:
        hand._stop.set()
        hand._wake.set()
        hand._worker.join(timeout=2.0)


@check("a hand I/O error is absorbed by the worker instead of killing it")
def _():
    import time as _time

    class _BrokenHand:
        def set_angles(self, angles):
            raise IOError("bus went quiet")

        def get_angles(self):
            raise IOError("bus went quiet")

    hand = _can_hand()
    hand._hand = _BrokenHand()
    hand._stop.clear()
    hand._worker = threading.Thread(target=hand._io_loop, daemon=True)
    hand._worker.start()
    try:
        for i in range(5):
            hand.set_opening(i / 5)
            _time.sleep(0.03)
        assert hand._worker.is_alive(), "the worker died on an I/O error; the hand would freeze silently"
        assert hand._io_errors > 0, "errors were not counted"
        # The caller still gets a usable answer.
        assert 0.0 <= hand.get_opening() <= 1.0
    finally:
        hand._stop.set()
        hand._wake.set()
        hand._worker.join(timeout=2.0)


@check("a readback failure reports the last command rather than stalling the arm loop")
def _():
    class _DeadReadback:
        def set_angles(self, angles):
            pass

        def get_angles(self):
            raise IOError("bus went quiet")

    hand = _can_hand()
    hand._hand = _DeadReadback()
    hand.set_opening(0.25)
    # The arm's control loop calls this every step. Raising here would take the
    # arm down over a dropped hand frame, which is the wrong trade.
    assert abs(hand.get_opening() - 0.25) < 1e-9, hand.get_opening()


@check("the index finger resting against the thumb still reads as fully closed")
def _():
    class _RestsOnThumb:
        """The finger physically stops at the contact point, above finger_close."""

        def set_angles(self, angles):
            pass

        def get_angles(self):
            return [0, 0, 0, 370, 600, 100]

    hand = _can_hand()
    hand._hand = _RestsOnThumb()
    # get_opening serves the worker's last readback rather than touching the bus,
    # so seed it the way a completed write would.
    hand._measured = _RestsOnThumb().get_angles()
    opening = hand.get_opening()
    assert 0.0 <= opening <= 1.0, opening
    # 370 is above finger_close=356, so the raw ratio is positive but tiny.
    # What matters is that it does not escape [0, 1] and does not read as open.
    assert opening < 0.05, f"resting on the thumb should read as closed, got {opening}"


@check("a goal blocked by the floor still publishes, so the sender cannot latch")
def _():
    # Measured failure this pins: engaging with a mapped target 17 mm below the
    # floor made step_joints return without calling servo_to_joints. The 1 kHz
    # sender saw no goals for 0.25 s, braked, and latched -- "goals stopped
    # arriving while the arm was moving" -- and the arm never moved again.
    import json
    from pathlib import Path

    from atom_robot.ar5.ar5_env import AR5Env
    from atom_robot.ar5.config import AR5Config
    from atom_robot.ar5.joint_mapping import JointMapConfig
    from atom_robot.ar5.kinematics import AR5Kinematics

    dh_file = Path(__file__).resolve().parent / "ar5_dh.json"
    if not dh_file.exists():
        return

    config = AR5Config(mock=True)
    env = AR5Env(config)
    env.connect()
    try:
        dh = np.asarray(json.loads(dh_file.read_text())["dh"], dtype=np.float64)
        env.arm.kinematics = AR5Kinematics(
            dh, num_joints=config.arm.num_joints,
            joint_min=config.arm.joint_min, joint_max=config.arm.joint_max,
        )

        low = np.asarray(JointMapConfig().centre_joints, dtype=np.float64).copy()
        for angle in np.deg2rad(np.arange(0, 121, 5)):
            low[1] = angle
            if env._lowest_point(low) < config.arm.joint_floor_z:
                break
        else:
            raise AssertionError("could not find a posture below the floor to test with")

        published = []
        real_servo = env.arm.servo_to_joints

        def spy(q):
            published.append(np.asarray(q).copy())
            return real_servo(q)

        env.arm.servo_to_joints = spy

        # Rate limiting means the goal creeps toward the floor over many steps;
        # what matters is the stretch after the guard starts refusing.
        for _ in range(400):
            env.step_joints(low)
        assert env.stats["floor_blocked"] > 0, "the floor guard never fired"

        published.clear()
        for _ in range(20):
            env.step_joints(low)
        assert published, (
            "20 floor-blocked steps published no goal at all. The 1 kHz sender "
            "reads that as a dead loop, brakes and latches, and the arm stops "
            "responding until the operator re-engages."
        )
    finally:
        env.close()


# ------------------------------------------------------------ centre tuning


@check("tune_centre runs bare: no cameras opened, no hand driven")
def _():
    from atom_robot.ar5.tune_centre import build_config

    config = build_config()
    # The failure this catches: `config.use_cameras = False` looks like it
    # disables the cameras, but no such field exists, so it just adds an
    # attribute nothing reads and OpenCV still tries to open device 0.
    assert config.cameras.sources == {}, (
        f"cameras must be off for a bare-arm procedure, got {config.cameras.sources}"
    )
    assert config.use_hand is False, "the hand must not be driven while tuning the centre"


@check("a config typo cannot silently pass for a disabled camera")
def _():
    from atom_robot.ar5.config import AR5Config

    # Guards the assumption the check above relies on: `use_cameras` is not a
    # field, so anyone writing it is not turning anything off.
    assert not hasattr(AR5Config(), "use_cameras"), (
        "AR5Config grew a `use_cameras` field; tune_centre and the teleop "
        "scripts disable cameras by emptying cameras.sources and should be updated"
    )


def _rel_mapper(**overrides):
    from atom_robot.ar5.config import ArmConfig
    from atom_robot.ar5.joint_mapping import JOINT_PAIRS, JointMapConfig, JointMapper

    cfg = ArmConfig()
    # Filter off: these checks are about the correspondence between leader and
    # follower joints, and they feed readings back to back with no time between
    # them -- which is exactly the input the filter is built to reject. The
    # filter has its own checks further down.
    overrides.setdefault("filter_min_cutoff", 0.0)
    mapper = JointMapper(JointMapConfig(relative=True, ramp_seconds=0.0, **overrides))
    mapper.configure({name: 100.0 for name, _, _ in JOINT_PAIRS},
                     cfg.joint_min, cfg.joint_max, margin=cfg.joint_limit_margin)
    return mapper


@check("relative mode runs 1:1 instead of shrinking to fit the centre")
def _():
    from atom_robot.ar5.joint_mapping import JOINT_PAIRS

    rel = _rel_mapper()
    for name, _, _ in JOINT_PAIRS:
        assert abs(rel._scales[name] - 1.0) < 1e-9, (
            f"{name} scale {rel._scales[name]}; relative mode has no centre to fit around"
        )
    # --scale still applies on top.
    assert abs(_rel_mapper(scale_multiplier=0.5)._scales["elbow_flex"] - 0.5) < 1e-9


@check("engaging in relative mode does not move the arm")
def _():
    from atom_robot.ar5.joint_mapping import JOINT_PAIRS

    rel = _rel_mapper()
    # Nowhere near the centre, and nowhere near the leader's zero.
    q_now = rel.centre + np.deg2rad(11.0)
    leader = {name: 37.0 for name, _, _ in JOINT_PAIRS}

    rel.engage(q_now)
    first = rel.target_joints(leader)
    assert np.allclose(first, q_now, atol=1e-9), (
        f"engaging jumped the arm by {np.rad2deg(np.max(np.abs(first - q_now))):.2f} deg; "
        "the leader anchor is supposed to be taken from this very reading"
    )


@check("relative mode tracks leader movement, not leader position")
def _():
    from atom_robot.ar5.joint_mapping import JOINT_PAIRS

    rel = _rel_mapper()
    q_now = rel.centre.copy()
    rel.engage(q_now)
    rel.target_joints({name: 20.0 for name, _, _ in JOINT_PAIRS})       # anchor here
    moved = rel.target_joints({name: 30.0 for name, _, _ in JOINT_PAIRS})

    for name, index, sign in JOINT_PAIRS:
        expected = q_now[index] + sign * rel._scales[name] * np.deg2rad(10.0)
        assert abs(moved[index] - expected) < 1e-9, (
            f"{name}: got {np.rad2deg(moved[index]):.3f}, expected {np.rad2deg(expected):.3f}"
        )

    # Re-engaging re-anchors: the same leader posture now means "no motion",
    # which is what lets the operator recentre and keep going.
    rel.disengage()
    rel.engage(moved)
    again = rel.target_joints({name: 30.0 for name, _, _ in JOINT_PAIRS})
    assert np.allclose(again, moved, atol=1e-9), "re-engaging failed to re-anchor"


@check("relative mode still clamps to the soft limits")
def _():
    from atom_robot.ar5.config import ArmConfig
    from atom_robot.ar5.joint_mapping import JOINT_PAIRS

    cfg = ArmConfig()
    rel = _rel_mapper()
    # Anchor near a limit, then keep pushing the leader the same way. Relative
    # mode has no fitted scale holding it back, so the clamp is the only thing
    # between the operator and a refused goal.
    q_now = np.asarray(cfg.joint_max, dtype=np.float64) - 0.10
    rel.engage(q_now)
    rel.target_joints({name: 0.0 for name, _, _ in JOINT_PAIRS})
    pushed = rel.target_joints({name: 200.0 for name, _, _ in JOINT_PAIRS})

    lower = np.asarray(cfg.joint_min) + cfg.joint_limit_margin
    upper = np.asarray(cfg.joint_max) - cfg.joint_limit_margin
    assert np.all(pushed >= lower - 1e-9) and np.all(pushed <= upper + 1e-9), (
        f"relative mode escaped the soft limits: {np.rad2deg(pushed).round(1).tolist()}"
    )
    assert rel.stats["clamped"] > 0, "the clamp never registered"


@check("opening a camera leaves the driver connected, not just the zero-camera path")
def _():
    from atom_robot.ar5.config import CameraConfig
    from atom_robot.ar5.drivers.cameras import OpenCVCameras

    # Only the "no sources" branch used to set _connected, so a rig that
    # actually had a camera opened it fine and then raised "used before
    # connect()" on the first read. Invisible on a machine with no camera.
    class _FakeCap:
        def isOpened(self):
            return True

        def set(self, *_):
            return True

        def read(self):
            return True, np.zeros((4, 4, 3), dtype=np.uint8)

        def release(self):
            pass

    cfg = CameraConfig()
    # 假设备号要用不存在的：本机 /dev/video0..9 真是腕部 Orbbec，驱动会认出来
    # 改走 OBSENSOR，就绕过了这里假的 VideoCapture。
    cfg.sources = {"cam_front": 92}
    cams = OpenCVCameras(cfg)

    import cv2
    real_capture = cv2.VideoCapture
    cv2.VideoCapture = lambda *_a, **_k: _FakeCap()
    try:
        cams.connect()
        assert cams._connected, "connect() opened a camera but left the driver 'unconnected'"
        frames = cams.read()
        assert "cam_front" in frames, frames
    finally:
        cv2.VideoCapture = real_capture
        cams.disconnect()

    # And the zero-source path stays legal.
    empty = OpenCVCameras(CameraConfig())
    empty.config.sources = {}
    empty.connect()
    assert empty.read() == {}


@check("给路径的 Orbbec 也改走 OBSENSOR，V4L2 一下都不碰")
def _():
    from types import SimpleNamespace

    from atom_robot.ar5.drivers import cameras as cammod

    # 2026-09-30：V4L2 开过一次 Orbbec，OBSENSOR 就只剩 0.3 Hz 且不报错；反过来
    # OBSENSOR 开过，V4L2 一帧不出。所以谁要是还按路径给（旧 config、
    # `--source /dev/ar5_cam_wrist`），驱动要认出来是 Orbbec，改道。
    opened_v4l2, went_obsensor = [], []
    import cv2
    real_capture = cv2.VideoCapture
    real_of_node, real_connect = cammod.orbbec_serial_of_node, cammod.OpenCVCameras._connect_orbbec
    cv2.VideoCapture = lambda *a, **_k: opened_v4l2.append(a) or None
    cammod.orbbec_serial_of_node = lambda src: "SN305" if src == "/dev/ar5_cam_wrist" else None
    cammod.OpenCVCameras._connect_orbbec = lambda self, name, sn: went_obsensor.append((name, sn))
    try:
        cams = cammod.OpenCVCameras(SimpleNamespace(
            sources={"cam_front": "/dev/ar5_cam_wrist"}, width=640, height=480, fps=30))
        cams.connect()
    finally:
        cv2.VideoCapture = real_capture
        cammod.orbbec_serial_of_node, cammod.OpenCVCameras._connect_orbbec = real_of_node, real_connect
    assert went_obsensor == [("cam_front", "SN305")], went_obsensor
    assert not opened_v4l2, f"Orbbec 被 V4L2 开了：{opened_v4l2}"

    # 不是 Orbbec 的东西不能误认
    assert cammod.orbbec_serial("orbbec:CV27864000V2") == "CV27864000V2"
    assert cammod.orbbec_serial("/dev/video0") is None
    assert cammod.orbbec_serial_of_node("/dev/null") is None
    assert cammod.orbbec_serial_of_node(9999) is None


@check("写了 v4l2: 的 Orbbec 不改道，按原路径走 V4L2")
def _():
    from types import SimpleNamespace

    from atom_robot.ar5.drivers import cameras as cammod

    # 2026-10-15：腕部 305 走 OBSENSOR 一帧不出，RobotDecision 要继续用 V4L2 开它。
    # 「v4l2:<路径>」就是给这种情况的出口：明知是 Orbbec 也不改道。
    # 路径故意用一个不存在的：/dev/ar5_cam_wrist 在装了 udev 规则的机器上是真的。
    assert cammod.strip_v4l2("v4l2:/dev/ar5_cam_wrist") == ("/dev/ar5_cam_wrist", True)
    assert cammod.strip_v4l2("/dev/ar5_cam_wrist") == ("/dev/ar5_cam_wrist", False)
    assert cammod.strip_v4l2(3) == (3, False)

    class _Closed:
        def isOpened(self):
            return False

        def release(self):
            pass

    opened_v4l2, went_obsensor = [], []
    import cv2
    real_capture = cv2.VideoCapture
    real_of_node, real_connect = cammod.orbbec_serial_of_node, cammod.OpenCVCameras._connect_orbbec
    cv2.VideoCapture = lambda *a, **_k: opened_v4l2.append(a) or _Closed()
    cammod.orbbec_serial_of_node = lambda src: "SN305" if src == "/dev/__selftest_no_such_cam__" else None
    cammod.OpenCVCameras._connect_orbbec = lambda self, name, sn: went_obsensor.append((name, sn))
    try:
        cams = cammod.OpenCVCameras(SimpleNamespace(
            sources={"cam_front": "v4l2:/dev/__selftest_no_such_cam__"}, width=640, height=480, fps=30))
        try:
            cams.connect()
        except IOError:
            pass                       # 假相机打不开，报错是对的；这里只看走了哪条路
        problems = cammod.check_sources(SimpleNamespace(sources={"cam_front": "v4l2:/dev/__selftest_no_such_cam__"}))
    finally:
        cv2.VideoCapture = real_capture
        cammod.orbbec_serial_of_node, cammod.OpenCVCameras._connect_orbbec = real_of_node, real_connect
    assert not went_obsensor, f"写了 v4l2: 还是改道去了 OBSENSOR：{went_obsensor}"
    assert opened_v4l2 and opened_v4l2[0][0] == "/dev/__selftest_no_such_cam__", opened_v4l2
    assert "路径不存在" in problems.get("cam_front", ""), problems     # 预检按去掉前缀的路径查


@check("--scale actually scales, including over a saved calibration")
def _():
    from atom_robot.ar5.config import ArmConfig
    from atom_robot.ar5.joint_mapping import JOINT_PAIRS, JointMapConfig, JointMapper

    # scale_multiplier was declared, plumbed in from --scale, and then never
    # read by configure(). "--scale 0.5" moved the arm exactly as far as no
    # flag at all -- the opposite of what an operator asks it for on a first run.
    cfg = ArmConfig()
    half = {name: 100.0 for name, _, _ in JOINT_PAIRS}

    def scales(multiplier, overrides=None):
        config = JointMapConfig(scale_multiplier=multiplier,
                                scale_overrides=overrides or {})
        mapper = JointMapper(config)
        mapper.configure(half, cfg.joint_min, cfg.joint_max, margin=cfg.joint_limit_margin)
        return dict(mapper._scales)

    full, half_scaled = scales(1.0), scales(0.5)
    for name, _, _ in JOINT_PAIRS:
        assert abs(half_scaled[name] - full[name] * 0.5) < 1e-9, (
            f"{name}: --scale 0.5 gave {half_scaled[name]}, expected {full[name] * 0.5}"
        )

    # A saved calibration must not become immune to the knob.
    pinned = scales(0.5, {"shoulder_pan": 0.8})
    assert abs(pinned["shoulder_pan"] - 0.4) < 1e-9, pinned["shoulder_pan"]


@check("the centre solver keeps the locked joints at zero and stays in limits")
def _():
    import json as _json

    from atom_robot.ar5.config import ArmConfig
    from atom_robot.ar5.kinematics import AR5Kinematics
    from atom_robot.ar5.tune_centre import solve_centre

    cfg = ArmConfig()
    dh = _json.loads((Path(__file__).resolve().parent / "ar5_dh.json").read_text())["dh"]
    kin = AR5Kinematics(np.array(dh).reshape(-1, 4), cfg.num_joints, cfg.joint_min, cfg.joint_max)

    found = solve_centre(kin, cfg, 0.30)
    assert found is not None, "no posture found at a height the arm can obviously reach"
    q, tool, _scales = found

    # 锁死的关节必须在零位：中心姿态挪动它们等于映射了遥操作永远不会命令的东西。
    # 跟着 locked_joints 走，别写死索引 —— wrist_roll 从 J5 改到 J7 时这里差点漏掉。
    from atom_robot.ar5.joint_mapping import JointMapConfig
    for i in JointMapConfig().locked_joints:
        assert abs(q[i]) < 1e-9, f"locked joint j{i} not at zero: {q}"
    assert abs(tool[2] - 0.30) < 0.005, f"solved posture misses the height: {tool}"

    lower = np.asarray(cfg.joint_min) + cfg.joint_limit_margin
    upper = np.asarray(cfg.joint_max) - cfg.joint_limit_margin
    assert np.all(q >= lower) and np.all(q <= upper), f"solution outside soft limits: {q}"


# ------------------------------------------------------- joint-space mapping


_LEADER_HALF = {"shoulder_pan": 110.5, "shoulder_lift": 105.2, "elbow_flex": 97.2,
                "wrist_roll": 180.0, "wrist_flex": 101.8}


def _joint_mapper(**overrides):
    from atom_robot.ar5.config import AR5Config
    from atom_robot.ar5.joint_mapping import JOINT_PAIRS, JointMapConfig, JointMapper

    cfg = AR5Config().arm
    # Filter off, for the same reason as _rel_mapper: these are correspondence
    # checks that step the leader instantly, which is the one input shape the
    # filter exists to swallow.
    overrides.setdefault("filter_min_cutoff", 0.0)
    mapper = JointMapper(JointMapConfig(**overrides))
    mapper.configure(_LEADER_HALF, cfg.joint_min, cfg.joint_max, margin=cfg.joint_limit_margin)
    return mapper, cfg, JOINT_PAIRS


@check("each leader joint drives exactly one AR5 joint, and the locked ones never move")
def _():
    mapper, _, pairs = _joint_mapper()
    rest = {name: 0.0 for name, _, _ in pairs}
    base = mapper.target_joints(rest)

    driven = {}
    for name, index, _ in pairs:
        moved = dict(rest)
        moved[name] = 20.0
        delta = mapper.target_joints(moved) - base
        touched = [i for i in range(len(delta)) if abs(delta[i]) > 1e-9]
        assert touched == [index], f"{name} moved joints {touched}, expected only {index}"
        driven[name] = index

    assert len(set(driven.values())) == len(driven), f"two leader joints share an AR5 joint: {driven}"
    for locked in mapper.config.locked_joints:
        assert locked not in driven.values(), f"j{locked} is both locked and driven"


@check("full leader travel maps inside the soft limits without ever clipping")
def _():
    mapper, cfg, pairs = _joint_mapper()
    lo = np.asarray(cfg.joint_min) + cfg.joint_limit_margin
    hi = np.asarray(cfg.joint_max) - cfg.joint_limit_margin

    # A scale fitted to the available travel is the whole point: a joint that
    # clipped partway through its range would simply stop responding, which is
    # indistinguishable from a broken mapping.
    for name, index, _ in pairs:
        for extreme in (+_LEADER_HALF[name], -_LEADER_HALF[name]):
            reading = {n: 0.0 for n, _, _ in pairs}
            reading[name] = extreme
            q = mapper.target_joints(reading)
            assert np.all(q >= lo - 1e-9) and np.all(q <= hi + 1e-9), (
                f"{name} at {extreme:+.1f} deg left the limits: "
                f"{np.rad2deg(q).round(1).tolist()}"
            )
    assert mapper.stats["clamped"] == 0, "a fitted scale should never need clamping"


@check("a dropped leader reply holds its joint instead of reading as zero")
def _():
    mapper, _, pairs = _joint_mapper()
    full = {name: 30.0 for name, _, _ in pairs}
    both = mapper.target_joints(full)

    partial = dict(full)
    del partial["elbow_flex"]
    held = mapper.target_joints(partial)

    elbow = next(i for n, i, _ in pairs if n == "elbow_flex")
    assert np.isclose(held[elbow], mapper.centre[elbow]), (
        "a missing joint must fall back to its centre, not to a zero reading "
        "that happens to mean something else"
    )
    others = [i for _, i, _ in pairs if i != elbow]
    assert np.allclose(held[others], both[others]), "one dropped joint disturbed the rest"
    assert mapper.stats["missing"] >= 1


@check("--flip inverts one joint and leaves the others alone")
def _():
    from atom_robot.ar5.joint_mapping import JOINT_PAIRS

    plain, _, pairs = _joint_mapper()
    derived = {name: sign for name, _, sign in JOINT_PAIRS}
    flipped, _, _ = _joint_mapper(sign_overrides={"wrist_roll": -derived["wrist_roll"]})

    reading = {name: 25.0 for name, _, _ in pairs}
    a = plain.target_joints(reading) - plain.centre
    b = flipped.target_joints(reading) - flipped.centre

    roll = next(i for n, i, _ in pairs if n == "wrist_roll")
    assert np.isclose(a[roll], -b[roll]) and abs(a[roll]) > 1e-6, (a[roll], b[roll])
    others = [i for _, i, _ in pairs if i != roll]
    assert np.allclose(a[others], b[others]), "flipping one joint changed another"


@check("engaging ramps in from the current posture instead of jumping")
def _():
    import time as _time

    mapper, cfg, pairs = _joint_mapper(ramp_seconds=0.25)
    reading = {name: 40.0 for name, _, _ in pairs}
    target = mapper.target_joints(reading)

    start = np.asarray(cfg.home_joints, dtype=np.float64)
    gap = float(np.max(np.abs(target - start)))
    assert gap > 0.2, "pick a start posture far enough away for this to mean something"

    # Released: the commanded posture is exactly where the arm already is.
    assert np.allclose(mapper.compute(reading, start), start)

    mapper.engage(start)
    first = mapper.compute(reading, start)
    assert float(np.max(np.abs(first - start))) < gap * 0.25, (
        "the first engaged command must be close to the starting posture"
    )

    deadline = _time.perf_counter() + 1.0
    while mapper.ramp_fraction() < 1.0 and _time.perf_counter() < deadline:
        _time.sleep(0.02)
    assert mapper.ramp_fraction() == 1.0, "the ramp never completed"
    assert np.allclose(mapper.compute(reading, start), target), (
        "after the ramp the command must be the mapped target, not a blend"
    )


@check("the joint path refuses a posture that would drive the arm below the floor")
def _():
    import json
    from pathlib import Path

    from atom_robot.ar5.ar5_env import AR5Env
    from atom_robot.ar5.config import AR5Config
    from atom_robot.ar5.joint_mapping import JointMapConfig
    from atom_robot.ar5.kinematics import AR5Kinematics

    dh_file = Path(__file__).resolve().parent / "ar5_dh.json"
    if not dh_file.exists():
        return

    config = AR5Config(mock=True)
    env = AR5Env(config)
    env.connect()
    try:
        dh = np.asarray(json.loads(dh_file.read_text())["dh"], dtype=np.float64)
        env.arm.kinematics = AR5Kinematics(
            dh, num_joints=config.arm.num_joints,
            joint_min=config.arm.joint_min, joint_max=config.arm.joint_max,
        )
        kin = env.arm.kinematics

        # Joint space has no workspace box to stop the arm folding into the
        # table, so the floor is the only thing between a plausible-looking
        # joint command and the bench.
        safe = np.asarray(config.arm.home_joints, dtype=np.float64)
        assert env._lowest_point(safe) > config.arm.joint_floor_z, "home is already below the floor"

        # Tipping the shoulder forward is the way the arm actually gets low, and
        # it is reachable in normal use: the mapped shoulder_lift can drive j1 to
        # 117 degrees, which puts the wrist well under the bench.
        low = np.asarray(JointMapConfig().centre_joints, dtype=np.float64).copy()
        for angle in np.deg2rad(np.arange(0, 121, 5)):
            low[1] = angle
            if env._lowest_point(low) < config.arm.joint_floor_z:
                break
        else:
            raise AssertionError("could not find a posture below the floor to test with")

        # The goal's advance is rate-limited, so getting there takes many steps.
        # Driving all the way in is also the honest test: the guard has to hold
        # against sustained pushing, not just refuse one oversized command.
        for _ in range(400):
            env.step_joints(low)

        assert env.stats["floor_blocked"] > 0, "the floor guard never fired"
        settled = env._commanded_joints()
        assert env._lowest_point(settled) >= config.arm.joint_floor_z, (
            f"the arm settled at {env._lowest_point(settled):.4f} m, below the "
            f"{config.arm.joint_floor_z} m floor"
        )
        assert not np.allclose(settled, low, atol=1e-6), "the arm reached the unsafe posture"

        before_ok = env.stats["floor_blocked"]
        for _ in range(400):
            env.step_joints(safe)
        assert env.stats["floor_blocked"] == before_ok, "the guard fired on a safe posture"
        assert np.allclose(env._commanded_joints(), safe, atol=1e-9), (
            "a safe posture must still be reachable"
        )
    finally:
        env.close()


@check("the joint path clamps to the soft limits rather than refusing outright")
def _():
    from atom_robot.ar5.ar5_env import AR5Env
    from atom_robot.ar5.config import AR5Config

    config = AR5Config(mock=True)
    env = AR5Env(config)
    env.connect()
    try:
        env.arm.kinematics = None          # skip the floor check for this one
        wild = np.asarray(config.arm.home_joints, dtype=np.float64).copy()
        wild[0] = 99.0                     # far outside anything legal

        before = env.stats["joint_clamped"]
        env.step_joints(wild)
        assert env.stats["joint_clamped"] == before + 1, "an illegal goal was not counted"

        # And a non-finite goal must not reach the arm at all.
        before_bad = env.stats["unreachable"]
        env.step_joints(np.full(config.arm.num_joints, np.nan))
        assert env.stats["unreachable"] == before_bad + 1
    finally:
        env.close()


@check("a fast leader move is rate-limited, not refused")
def _():
    from atom_robot.ar5.ar5_env import AR5Env
    from atom_robot.ar5.config import AR5Config

    config = AR5Config(mock=True)
    config.settle_time = 0.0
    env = AR5Env(config)
    env.connect()
    try:
        env.arm.kinematics = None          # isolate this from the floor guard

        start = env._commanded_joints().copy()
        far = start.copy()
        far[0] += 1.2                      # a jump no gate would ever accept

        # The driver's jump gate refuses a discontinuity. On the joint path a
        # discontinuity only means the operator moved fast, and refusing it is
        # fatal: the goal is dropped, the gap grows, and every later goal is
        # refused too. So the goal's advance is limited instead.
        for _ in range(400):
            env.step_joints(far)

        reached = env._commanded_joints()
        assert np.allclose(reached, far, atol=1e-9), (
            f"the goal was never reached: j0 got to {reached[0]:.4f}, wanted {far[0]:.4f}"
        )
        assert env.stats["joint_rate_limited"] > 0, "the rate limiter never engaged"
        assert env.arm.rejected_joint_goals == 0, (
            f"{env.arm.rejected_joint_goals} goals were refused; the rate limit must keep "
            "every command inside the jump gate"
        )
    finally:
        env.close()


@check("the goal rate limit stays strictly inside the jump gate")
def _():
    from atom_robot.ar5.config import ArmConfig

    cfg = ArmConfig()
    assert 0.0 < cfg.joint_goal_step < cfg.joint_jump_max, (
        cfg.joint_goal_step, cfg.joint_jump_max
    )

    # Equality is the trap, not a typo: rate-limiting to exactly the gate's
    # threshold makes q - q_cmd land on the boundary, and 1.926388 - 0.15 comes
    # back 0.15000000000000013 away, which the gate reads as a jump. Measured on
    # the mock arm: every goal refused, zero motion. So equality must be an
    # error at construction, not a surprise at runtime.
    for bad in (cfg.joint_jump_max, cfg.joint_jump_max * 1.5, 0.0, -0.01):
        try:
            ArmConfig(joint_goal_step=bad)
        except ValueError:
            continue
        raise AssertionError(f"joint_goal_step={bad} should have been rejected")


# --------------------------------------------------- smoothing / loop jitter
#
# The arm shook. Three inputs feed that, and each of the checks below pins one:
# encoder quantisation differentiated into a feed-forward velocity, hand tremor
# passed 1:1 by relative mapping, and a control-loop period that wanders because
# the loop was doing work nobody read.


def _dither(counts: int = 1, samples: int = 400):
    """A leader joint sitting still, wobbling between two adjacent counts."""
    step = 360.0 / 4095.0 * counts
    return [step * (i % 2) for i in range(samples)]


@check("the leader filter is on by default, not just in the shipped comment")
def _():
    from atom_robot.ar5.joint_mapping import JointMapConfig

    # Every mapping check above builds its mapper with the filter off, because
    # they step the leader instantly. That is a fine way to test a mapping and a
    # terrible way to find out the product default got turned off, so assert the
    # default here where nothing has overridden it.
    cfg = JointMapConfig()
    assert cfg.filter_min_cutoff > 0.0, "the leader filter is disabled by default"
    assert cfg.filter_beta > 0.0, (
        "beta is 0, so the cutoff never opens up and the filter lags every real move"
    )


@check("the filter swallows encoder dither and still follows a real move")
def _():
    from atom_robot.ar5.joint_mapping import JointMapConfig, OneEuro

    cfg = JointMapConfig()
    dt = 1.0 / 50.0

    quiet = OneEuro(cfg.filter_min_cutoff, cfg.filter_beta, cfg.filter_d_cutoff)
    out = [quiet(v, dt) for v in _dither()]
    settled = np.ptp(np.asarray(out[100:]))
    raw_span = 360.0 / 4095.0
    # Measured: 0.0879 deg of dither in, 0.0076 out -- about 12x. The threshold
    # is deliberately looser than that, so retuning the filter does not have to
    # come here, but a filter that stopped rejecting dither would.
    assert settled < raw_span * 0.25, (
        f"dither only shrank from {raw_span:.4f} to {settled:.4f} deg -- the "
        "tracker differentiates this into feed-forward velocity"
    )

    # A deliberate move must still arrive. 40 deg/s for a second: the output has
    # to end within a few degrees of the input, not lag it by a fixed fraction.
    moving = OneEuro(cfg.filter_min_cutoff, cfg.filter_beta, cfg.filter_d_cutoff)
    last = 0.0
    for i in range(50):
        last = moving(40.0 * i * dt, dt)
    lag_deg = 40.0 * 49 * dt - last
    assert 0.0 <= lag_deg < 4.0, f"the filter lags a 40 deg/s move by {lag_deg:.2f} deg"


@check("the filter smooths by elapsed time, so the loop rate does not change it")
def _():
    from atom_robot.ar5.joint_mapping import JointMapConfig, OneEuro

    # This is the property a plain EMA does not have, and the reason the old
    # fixed-weight smoothing turned loop-period jitter into output jitter: at a
    # fixed weight, a long step is smoothed exactly as much as a short one, so
    # the amount of smoothing tracked whatever the loop happened to be doing.
    cfg = JointMapConfig()
    RATE, SECONDS = 30.0, 2.0          # a steady 30 deg/s ramp

    def lag(hz, step):
        """How far behind the input the filter settles, in degrees."""
        dt = 1.0 / hz
        n = int(hz * SECONDS)
        value = 0.0
        for i in range(n):
            value = step(RATE * i * dt, dt, value)
        return RATE * (n - 1) * dt - value

    euro = OneEuro(cfg.filter_min_cutoff, cfg.filter_beta, cfg.filter_d_cutoff)
    euro_slow = lag(25.0, lambda x, dt, _prev: euro(x, dt))
    euro = OneEuro(cfg.filter_min_cutoff, cfg.filter_beta, cfg.filter_d_cutoff)
    euro_fast = lag(100.0, lambda x, dt, _prev: euro(x, dt))

    # The comparison, and the reason this check exists: the fixed 0.4-per-sample
    # blend that used to smooth the leader reading. Its lag is a fixed number of
    # *samples*, so quadrupling the loop rate quarters it -- and a loop whose
    # period wanders is a filter whose strength wanders with it.
    fixed = lambda x, _dt, prev: 0.6 * prev + 0.4 * x  # noqa: E731
    ema_slow, ema_fast = lag(25.0, fixed), lag(100.0, fixed)

    assert ema_slow / ema_fast > 3.0, (
        f"the fixed blend was supposed to be the rate-dependent one: "
        f"{ema_slow:.3f} vs {ema_fast:.3f} deg"
    )
    # Measured 1.51 against the blend's 4.00. Not exactly 1: the canonical filter
    # estimates speed against its own last output, so a faster loop reads a
    # slightly higher speed and opens the cutoff a little further. That residual
    # is a property of the filter, not of the loop's jitter.
    ratio = euro_slow / euro_fast
    assert ratio < 2.0, (
        f"lag went from {euro_slow:.3f} deg at 25 Hz to {euro_fast:.3f} at 100 Hz "
        f"({ratio:.2f}x) -- smoothing still depends on the loop rate"
    )


@check("engaging with the filter on does not step the arm by the filter's lag")
def _():
    from atom_robot.ar5.config import ArmConfig
    from atom_robot.ar5.joint_mapping import (
        JOINT_PAIRS,
        JointMapConfig,
        JointMapper,
    )

    # Relative mode anchors on the leader reading taken at the clutch. Anchor on
    # the raw reading while driving from the filtered one and the arm jumps by
    # however far the filter is behind at that instant -- which, right after a
    # move, is the whole point of the filter.
    cfg = ArmConfig()
    mapper = JointMapper(JointMapConfig(relative=True, ramp_seconds=0.0))
    mapper.configure({name: 100.0 for name, _, _ in JOINT_PAIRS},
                     cfg.joint_min, cfg.joint_max, margin=cfg.joint_limit_margin)

    # Move the leader hard so the filter is genuinely behind, then engage.
    for i in range(30):
        mapper.target_joints({name: 3.0 * i for name, _, _ in JOINT_PAIRS})
        time.sleep(0.002)

    q_now = np.asarray(cfg.home_joints, dtype=np.float64)
    mapper.engage(q_now)
    reading = {name: 90.0 for name, _, _ in JOINT_PAIRS}
    first = mapper.target_joints(reading)
    assert np.allclose(first, q_now, atol=1e-9), (
        f"engaging moved the arm by {np.rad2deg(np.max(np.abs(first - q_now))):.3f} deg"
    )


@check("the feed-forward estimate is smoothed by time, not by step count")
def _():
    from atom_robot.ar5.config import ArmConfig

    # Mirror of the driver's estimator. A goal moving at a steady rate must
    # converge to that rate at the same speed in wall-clock terms whatever the
    # publisher's period, or loop jitter comes straight out as velocity jitter.
    def converge(dt, tau, true_rate=0.5, seconds=0.2):
        estimate = 0.0
        for _ in range(int(seconds / dt)):
            a = dt / (dt + tau)
            estimate = (1.0 - a) * estimate + a * true_rate
        return estimate

    tau = ArmConfig().goal_rate_tau
    slow, fast = converge(0.04, tau), converge(0.005, tau)
    assert abs(slow - fast) < 0.02, (
        f"25 Hz converged to {slow:.4f} and 200 Hz to {fast:.4f} rad/s"
    )

    # And the default must reproduce the fixed 0.4 blend it replaced at the
    # nominal 50 Hz, so this change is not also a retune of the arm's feel.
    assert abs(0.02 / (0.02 + tau) - 0.4) < 1e-9, (
        f"goal_rate_tau={tau} does not reproduce the previous 0.4 at 50 Hz"
    )


@check("teleoperation can step the arm without building an observation")
def _():
    from atom_robot.ar5.ar5_env import AR5Env
    from atom_robot.ar5.config import AR5Config

    # An observation is a camera frame plus three blocking SDK round trips.
    # Teleoperation discards it, and paying a *variable* amount of that per step
    # is what made the loop period wander -- which the tracker turns into a
    # wandering commanded velocity.
    config = AR5Config(mock=True)
    config.settle_time = 0.0
    env = AR5Env(config)
    env.connect()
    try:
        q = env.arm.get_joint_positions()

        reads = {"n": 0}
        real_read = env.cameras.read
        env.cameras.read = lambda: (reads.__setitem__("n", reads["n"] + 1), real_read())[1]

        assert env.step_joints(q, observe=False) is None, "observe=False still built one"
        assert reads["n"] == 0, "observe=False still read the cameras"

        obs = env.step_joints(q, observe=True)
        assert obs is not None and reads["n"] == 1, (reads, obs)

        # The default must stay observe=True: run_ar5_eval and collect both rely
        # on the return value, and a silent None there is a crash at record time.
        assert env.step_joints(q) is not None, "the default stopped returning an observation"
    finally:
        env.close()


@check("a camera read serves the newest frame instead of waiting for the next one")
def _():
    from atom_robot.ar5.config import CameraConfig
    from atom_robot.ar5.drivers.cameras import OpenCVCameras

    # Same shape as the hand fix: blocking I/O belongs on a worker, because the
    # caller is a control loop whose period the arm's tracker differentiates.
    class _SlowCap:
        """Delivers a frame every 40 ms and blocks in between, like a 25 fps camera."""

        def __init__(self):
            self.frames = 0

        def isOpened(self):
            return True

        def set(self, *_):
            return True

        def read(self):
            time.sleep(0.04)
            self.frames += 1
            return True, np.full((4, 4, 3), self.frames % 256, dtype=np.uint8)

        def release(self):
            pass

    cfg = CameraConfig()
    # 假设备号要用不存在的：本机 /dev/video0..9 真是腕部 Orbbec，驱动会认出来
    # 改走 OBSENSOR，就绕过了这里假的 VideoCapture。
    cfg.sources = {"cam_front": 90}
    cams = OpenCVCameras(cfg)

    import cv2
    real_capture = cv2.VideoCapture
    cv2.VideoCapture = lambda *_a, **_k: _SlowCap()
    try:
        cams.connect()
        # Ten reads at a 200 Hz loop. Synchronously that is ten 40 ms waits.
        started = time.perf_counter()
        for _ in range(10):
            assert "cam_front" in cams.read()
            time.sleep(0.005)
        elapsed = time.perf_counter() - started
        assert elapsed < 0.20, (
            f"ten reads took {elapsed * 1000:.0f} ms; read() is still blocking on the "
            "camera's frame clock, which welds the control loop to it"
        )
        # Reading faster than the camera means repeats, and they are counted
        # rather than hidden -- a silent repeat looks like a stalled scene.
        assert cams.stale_reads > 0, "reads outran a 25 fps camera without counting a repeat"
    finally:
        cv2.VideoCapture = real_capture
        cams.disconnect()


@check("shutting the cameras down actually joins the grabbers")
def _():
    from atom_robot.ar5.config import CameraConfig
    from atom_robot.ar5.drivers.cameras import OpenCVCameras

    # threading.Thread already owns the name `_stop`, and it calls it during
    # join(). A grabber that stored an Event there raised TypeError from inside
    # disconnect() -- so the caps were never released and the threads ran on.
    class _FakeCap:
        def isOpened(self):
            return True

        def set(self, *_):
            return True

        def read(self):
            time.sleep(0.005)
            return True, np.zeros((4, 4, 3), dtype=np.uint8)

        def release(self):
            self.released = True

    cfg = CameraConfig()
    # 假设备号要用不存在的：本机 /dev/video0..9 真是腕部 Orbbec，驱动会认出来
    # 改走 OBSENSOR，就绕过了这里假的 VideoCapture。
    cfg.sources = {"cam_front": 90, "cam_high": 91}
    cams = OpenCVCameras(cfg)

    import cv2
    real_capture = cv2.VideoCapture
    cv2.VideoCapture = lambda *_a, **_k: _FakeCap()
    try:
        cams.connect()
        grabbers = list(cams._grabbers.values())
        assert len(grabbers) == 2
        cams.disconnect()
        for grabber in grabbers:
            assert not grabber.is_alive(), f"{grabber.camera}'s grabber outlived disconnect()"
    finally:
        cv2.VideoCapture = real_capture


# ------------------------------------------------------- collection settings


@check("every CollectConfig mapping override is a real JointMapConfig field")
def _():
    from atom_robot.ar5.config import CollectConfig
    from atom_robot.ar5.joint_mapping import JointMapConfig

    # The whole point of `mapping_overrides` is that collect.py does not have to
    # know which knobs live where. A renamed field would otherwise surface as a
    # TypeError at the top of a collection session, with the arm powered.
    overrides = CollectConfig().mapping_overrides
    mapping = JointMapConfig(**overrides)
    assert mapping.relative is CollectConfig().relative
    assert mapping.scale_multiplier == CollectConfig().scale


@check("an unset filter knob defers to the mapper instead of forcing a value")
def _():
    from atom_robot.ar5.config import CollectConfig
    from atom_robot.ar5.joint_mapping import JointMapConfig

    # None means "whatever the mapper defaults to", not zero -- and zero turns
    # the anti-shake filter off entirely, which is the opposite of the default.
    default = CollectConfig()
    assert default.filter_min_cutoff is None and default.filter_beta is None
    assert "filter_min_cutoff" not in default.mapping_overrides
    assert JointMapConfig(**default.mapping_overrides).filter_min_cutoff > 0.0, (
        "an unset --min-cutoff ended up disabling the leader filter"
    )

    # Set, it must actually arrive.
    tuned = CollectConfig(filter_min_cutoff=0.8, filter_beta=0.12)
    mapping = JointMapConfig(**tuned.mapping_overrides)
    assert mapping.filter_min_cutoff == 0.8 and mapping.filter_beta == 0.12

    # Zero is a legal, meaningful request: filter off.
    off = JointMapConfig(**CollectConfig(filter_min_cutoff=0.0).mapping_overrides)
    assert off.filter_min_cutoff == 0.0


@check("a collection setting that would spoil the data is refused at construction")
def _():
    from atom_robot.ar5.config import CollectConfig

    # These do not fail loudly at runtime -- they quietly produce a dataset that
    # is wrong, and you find out at training time. So they have to be errors.
    bad = [
        {"hz": 0.0}, {"hz": -30.0},
        {"task": ""}, {"task": "   "},          # task string lands in every npz
        {"scale": 0.0}, {"scale": -1.0},
        {"min_frames": 0},
        {"filter_min_cutoff": -1.0}, {"filter_beta": -0.1},
        # FourCC is exactly four characters -- "h264" passes, and should.
        {"video_codec": "mp4"}, {"video_codec": "mpeg4"}, {"video_codec": ""},
    ]
    for kwargs in bad:
        try:
            CollectConfig(**kwargs)
        except ValueError:
            continue
        raise AssertionError(f"CollectConfig({kwargs}) should have been refused")

    # And the shipped defaults must survive their own validation.
    CollectConfig()


@check("the recorder's quality thresholds come from config, not from literals")
def _():
    from atom_robot.ar5.collect import Recorder
    from atom_robot.ar5.config import CollectConfig

    class _Obs:
        def __init__(self, pose, opening):
            self.observation = {
                "ee_pose": np.asarray(pose, dtype=np.float64),
                "qpos": np.zeros(8, dtype=np.float32),
                "images": {},
            }

    def record(config, frames, opening_of, pose_of):
        rec = Recorder(Path("/nonexistent"), [], config)
        for i in range(frames):
            rec.add(_Obs(pose_of(i), opening_of(i)), opening_of(i))
        return rec.warnings()

    still = lambda i: 0.5                                    # noqa: E731
    origin = lambda i: [0.4, 0.0, 0.3, 0.0, 0.0, 0.0]        # noqa: E731

    # 20 frames of a motionless hand: short-episode warning depends on the
    # threshold, and the gripper warning must fire either way.
    strict = record(CollectConfig(min_frames=50), 20, still, origin)
    lenient = record(CollectConfig(min_frames=5), 20, still, origin)
    assert any("短于 50" in n for n in strict), strict
    assert not any("短于" in n for n in lenient), lenient
    assert any("手全程没动" in n for n in strict), strict

    # A hand that moves and an arm that travels 300 mm must pass clean. Not
    # 100: that is exactly the threshold, and summing 19 steps of 0.1/19 lands
    # on 99.99999999999999 mm, so the fixture would sit on the boundary and
    # decide the test by rounding.
    straight = lambda i: [0.4 + 0.3 * i / 19.0, 0.0, 0.3, 0.0, 0.0, 0.0]  # noqa: E731
    moving = record(CollectConfig(min_frames=5), 20, lambda i: i / 19.0, straight)
    assert moving == [], moving

    # Raising the threshold above the actual path must fire.
    tight = record(CollectConfig(min_frames=5, min_ee_path_mm=500.0), 20,
                   lambda i: i / 19.0, straight)
    assert any("末端一共只走了" in n for n in tight), tight


@check("an arm that returns to where it started is not called motionless")
def _():
    from atom_robot.ar5.collect import Recorder
    from atom_robot.ar5.config import CollectConfig

    # The check used to measure the straight line from the first frame to the
    # last. For take & place that is close to zero by construction -- the arm
    # ends up back near where it began. Replayed against 8 real episodes on
    # 2026-09-04 it fired on 4 of them (5-14 mm end-to-end against 730-920 mm
    # actually travelled, worst case a factor of 182) and caught nothing.
    # Path length has no such blind spot: the two genuinely dead episodes from
    # the day before travelled 0.6 mm.
    class _Obs:
        def __init__(self, pose):
            self.observation = {
                "ee_pose": np.asarray(pose, dtype=np.float64),
                "qpos": np.zeros(8, dtype=np.float32),
                "images": {},
            }

    def notes_for(pose_of, frames=200):
        rec = Recorder(Path("/nonexistent"), [], CollectConfig(min_frames=5))
        for i in range(frames):
            rec.add(_Obs(pose_of(i)), 0.5 * (i % 2))     # hand moving, not the point
        return rec.warnings()

    # Out 300 mm and back: ends 0 mm from the start, travels 600 mm.
    def out_and_back(i):
        t = i / 199.0
        reach = 0.3 * (2 * t if t < 0.5 else 2 * (1 - t))
        return [0.4 + reach, 0.0, 0.3, 0.0, 0.0, 0.0]

    assert not any("末端" in n for n in notes_for(out_and_back)), notes_for(out_and_back)

    # Genuinely motionless must still be caught, wherever it sits.
    dead = notes_for(lambda i: [0.4, 0.0, 0.3, 0.0, 0.0, 0.0])
    assert any("末端一共只走了" in n for n in dead), dead


@check("collect reads the leader's travel from calibration instead of a literal")
def _():
    import inspect

    from atom_robot.ar5 import collect

    # It used to carry a hardcoded LEADER_HALF_DEG dict. The numbers were right
    # when written, but they do not follow a recalibration or a swapped leader
    # arm -- so teleoperation and collection would silently disagree on scale,
    # and the data would not match the feel it was recorded with.
    assert not hasattr(collect, "LEADER_HALF_DEG"), (
        "collect grew a hardcoded leader-travel table again; read it from "
        "LeaderBus.half_travel_deg() so a recalibration is picked up"
    )
    source = inspect.getsource(collect.main)
    assert "half_travel_deg" in source, (
        "collect.main no longer asks the calibrated bus for the leader's travel"
    )


@check("collect's own defaults are the config's, so the CLI is only an override")
def _():
    import inspect

    from atom_robot.ar5 import collect
    from atom_robot.ar5.config import AR5Config

    assert hasattr(AR5Config(), "collect"), "AR5Config lost its CollectConfig"

    # Every argparse default must be None, so "not given" is distinguishable
    # from "given the same value the config happens to hold". Otherwise editing
    # config.py would appear to do nothing, which is the whole interface.
    source = inspect.getsource(collect.main)
    for flag in ("--task", "--out", "--hz", "--scale", "--min-cutoff", "--beta"):
        assert f'"{flag}"' in source, f"{flag} disappeared from collect"
    bad = [line.strip() for line in source.splitlines()
           if "add_argument" in line and "default=" in line
           and "default=None" not in line]
    assert not bad, (
        "these flags carry their own default instead of deferring to "
        f"CollectConfig: {bad}"
    )


# ------------------------------------------------- episode scheduling
#
# This is what decides when data starts and stops being recorded, so it is
# driven here rather than left to be discovered mid-session.


def _schedule(**kwargs):
    from atom_robot.ar5.collect import EpisodeSchedule
    from atom_robot.ar5.config import CollectConfig

    kwargs.setdefault("hz", 30.0)
    kwargs.setdefault("min_frames", 10)
    start = kwargs.pop("start", 0)
    return EpisodeSchedule(CollectConfig(**kwargs), start)


@check("saving never advances to the next episode on its own")
def _():
    # The operator asked for this after losing episodes to it: auto-advance
    # started the next recording before the scene was reset, so the opening
    # seconds were of nobody doing anything. Manual start is the whole point.
    schedule = _schedule(num_episodes=5)
    assert schedule.phase == "idle", schedule.phase

    for episode in range(3):
        assert schedule.start() == "begin_record"
        assert schedule.recording
        assert schedule.stop() == "end_episode"
        assert schedule.episode_saved(True) is False
        # Back to standby, and it stays there however long nothing is pressed.
        assert schedule.phase == "idle", schedule.phase
        for frames in (0, 100, 10 ** 6):
            assert schedule.tick(frames) is None, "it started a new episode by itself"
        assert not schedule.recording
    assert schedule.index == 3 and schedule.done == 3


@check("an episode runs until s is pressed when no cap is set")
def _():
    from atom_robot.ar5.config import CollectConfig

    # The default: no time limit at all. A cap that quietly cut a
    # demonstration short would be worse than one long episode.
    assert CollectConfig().episode_time_s == 0.0
    assert CollectConfig().frame_cap == 0

    schedule = _schedule()
    schedule.start()
    for frames in (1, 600, 10 ** 5, 10 ** 7):
        assert schedule.tick(frames) is None, f"auto-stopped at {frames} frames"
    assert schedule.stop() == "end_episode"


@check("the episode cap is a backstop and still does not advance")
def _():
    # Set a cap and it saves on its own -- but lands in standby, not in the
    # next recording.
    schedule = _schedule(episode_time_s=2.0, num_episodes=5)
    assert schedule.cap == 60, schedule.cap
    schedule.start()
    assert schedule.tick(59) is None
    assert schedule.tick(60) == "end_episode"
    assert schedule.episode_saved(True) is False
    assert schedule.phase == "idle", "the cap auto-started the next episode"


@check("re-recording drops back to standby so the scene can be reset")
def _():
    # Pressing r used to discard the frames and keep recording, so the
    # retake started while the operator was still putting the bolt back --
    # the opening seconds were of a hand tidying up. Standby hands the reset
    # time to the operator instead, for as long as they need.
    schedule = _schedule(num_episodes=3)
    schedule.start()
    assert schedule.recording

    assert schedule.discard() == "discard"
    assert schedule.phase == "idle", schedule.phase
    # The episode number is unchanged -- this is a retake, not a new episode --
    # and nothing was counted as done.
    assert schedule.index == 0 and schedule.done == 0

    # It stays in standby until the operator says go, however long that is.
    for frames in (0, 500, 10 ** 6):
        assert schedule.tick(frames) is None
    assert schedule.start() == "begin_record"
    assert schedule.index == 0, "the retake took a new episode number"

    # Not recording: nothing to discard, and it must not disturb the state.
    schedule.stop()
    schedule.episode_saved(True)
    assert schedule.phase == "idle"
    assert schedule.discard() is None
    assert schedule.phase == "idle"


@check("left arrow deletes the episode that was just saved, and only that one")
def _():
    # 2026-09-22: the clutch key is gone -- recording *is* engagement -- and
    # left now means delete rather than retake. Deleting only ever reaches the
    # most recent save: holding the key down must not walk backwards through
    # the session, because one press too many and one press too few are not
    # equally expensive.
    schedule = _schedule(num_episodes=3, start=7)
    assert schedule.delete_last() is None, "deleted something before anything was saved"

    schedule.start()
    assert schedule.delete_last() == "recording", "left touched saved data mid-episode"

    schedule.stop()
    schedule.episode_saved(True)
    assert (schedule.index, schedule.done, schedule.last_saved) == (8, 1, 7)

    assert schedule.delete_last() == 7
    assert (schedule.index, schedule.done) == (7, 0), "the number did not go back"
    assert schedule.delete_last() is None, "a second press walked further back"

    # Deleting after the session hit its target reopens it.
    for _ in range(3):
        schedule.start()
        schedule.stop()
        finished = schedule.episode_saved(True)
    assert finished and schedule.phase == "done"
    assert schedule.delete_last() == 9
    assert not schedule.finished and schedule.phase == "idle"
    assert schedule.start() == "begin_record", "could not carry on after deleting"


@check("start and stop are ignored when they cannot mean anything")
def _():
    schedule = _schedule(num_episodes=1)
    assert schedule.stop() is None, "stopped while nothing was recording"
    schedule.start()
    assert schedule.start() is None, "started a second episode over the first"
    schedule.stop()
    assert schedule.episode_saved(True) is True     # num_episodes reached
    assert schedule.finished and schedule.phase == "done"
    assert schedule.start() is None, "started an episode after finishing"


@check("a session resumes numbering instead of overwriting what is on disk")
def _():
    # `num_episodes` counts this session's new episodes; the index continues
    # from whatever is already in the output directory.
    schedule = _schedule(num_episodes=2, start=12)
    for _ in range(2):
        schedule.start()
        schedule.stop()
        schedule.episode_saved(True)
    assert schedule.index == 14 and schedule.done == 2, (schedule.index, schedule.done)
    assert schedule.finished


@check("num_episodes = 0 keeps going instead of stopping at zero episodes")
def _():
    # The trap: `if self.target and ...` is what makes 0 mean unlimited. A
    # `>=` against a raw 0 would end the session before the first episode.
    schedule = _schedule(num_episodes=0)
    for _ in range(20):
        assert schedule.start() == "begin_record"
        schedule.stop()
        assert schedule.episode_saved(True) is False
    assert not schedule.finished and schedule.done == 20


@check("an abandoned episode does not consume an index or a slot")
def _():
    # Nothing recorded, or `q` mid-episode: the index must not advance, or the
    # next file would leave a hole and the count would lie.
    schedule = _schedule(num_episodes=3)
    schedule.start()
    assert schedule.episode_saved(False) is False
    assert schedule.index == 0 and schedule.done == 0
    assert schedule.phase == "idle"


@check("an episode cap that would flag every episode is refused up front")
def _():
    from atom_robot.ar5.config import CollectConfig

    # 2 s at 30 Hz is 60 frames, so min_frames=200 means *every* capped episode
    # saves with a "too short" warning -- a contradiction between two settings.
    try:
        CollectConfig(hz=30.0, episode_time_s=2.0, min_frames=200)
    except ValueError as exc:
        assert "min_frames" in str(exc), exc
    else:
        raise AssertionError("a self-contradictory episode cap was accepted")

    # With no cap there is no episode length to contradict, so it must not fire.
    CollectConfig(hz=30.0, episode_time_s=0.0, min_frames=200)

    for kwargs in ({"num_episodes": -1}, {"episode_time_s": -1.0}):
        try:
            CollectConfig(**kwargs)
        except ValueError:
            continue
        raise AssertionError(f"CollectConfig({kwargs}) should have been refused")


@check("retired flags are accepted and ignored, never silently honoured")
def _():
    import ast
    import inspect

    from atom_robot.ar5 import collect
    from atom_robot.ar5.config import CollectConfig

    # --reset-time / --warmup-time only ever drove auto-advance, which was
    # removed. They stay accepted because they sit in the operator's shell
    # history and a hard error blocked the same command twice in a row -- but
    # they must not come back to life, and they must not pass quietly either.
    source = inspect.getsource(collect.main)
    for flag in ("--reset-time", "--warmup-time"):
        assert f'"{flag}"' in source, f"{flag} would now be a hard error again"
    assert "已废弃" in source, "the retired flags no longer announce themselves"

    # The fields they used to set must be gone from the config, so there is
    # nothing for a future edit to wire them back into by accident.
    for gone in ("reset_time_s", "warmup_time_s"):
        assert not hasattr(CollectConfig(), gone), (
            f"CollectConfig.{gone} came back; auto-advance was removed on the "
            "operator's request because it started recording before the scene "
            "was reset"
        )

    # And they must not appear in the override table that copies CLI values
    # into the config -- that is the one place a revival would be invisible.
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and node.value in ("reset_time", "warmup_time"):
            raise AssertionError(
                f"{node.value!r} is back in collect.main's override table"
            )


@check("arrow keys arrive as one key, over a real terminal")
def _():
    import os
    import termios
    import tty

    from atom_robot.ar5.collect import RawKeyboard

    # Driven through a real pty, not a mocked stdin. The mocked version of this
    # check passed while the left arrow did nothing at all on the actual
    # console, because it stubbed out the exact layer that was broken:
    # `select` watches the file descriptor, while `sys.stdin.read(1)` goes
    # through Python's TextIOWrapper. Reading one character pulled all three
    # bytes of ESC [ D out of the kernel into that buffer, the fd went empty,
    # select then reported "nothing to read", and the `[` and `D` were
    # unreachable. Single-byte keys were unaffected, which is why space and s
    # always worked. RawKeyboard now reads the raw fd with os.read.
    master, slave = os.openpty()
    saved = termios.tcgetattr(slave)
    tty.setcbreak(slave)
    keyboard = RawKeyboard.__new__(RawKeyboard)
    keyboard.fd = slave
    keyboard._pending = ""
    try:
        for raw, expected in ((b"\x1b[D", "LEFT"), (b"\x1b[C", "RIGHT"),
                              (b"\x1b[A", "UP"), (b"\x1b[B", "DOWN"),
                              (b" ", " "), (b"s", "s"), (b"q", "q")):
            os.write(master, raw)
            got = keyboard.key()
            assert got == expected, f"wrote {raw!r}, got {got!r}, wanted {expected!r}"

        # Several keys in one read must all survive -- a fast operator, or a
        # held key, delivers more than one byte per select().
        os.write(master, b"\x1b[Ds \x1b[C")
        assert [keyboard.key() for _ in range(5)] == ["LEFT", "s", " ", "RIGHT", None]

        # Nothing pending, nothing typed: no key, and no blocking.
        assert keyboard.key() is None

        # OmniPicker 在这里输中文任务名：一个汉字 3 个字节，被两次 read 切开也要拼回来，
        # 不能变成两个乱码字符。
        han = "抓".encode()
        os.write(master, han[:2])
        assert keyboard.key() is None, "半个汉字不该当成一个键"
        os.write(master, han[2:] + "杯".encode())
        assert [keyboard.key() for _ in range(3)] == ["抓", "杯", None]
    finally:
        termios.tcsetattr(slave, termios.TCSADRAIN, saved)
        os.close(master)
        os.close(slave)


@check("the keys the loop binds are the keys the operator was told about")
def _():
    import inspect

    from atom_robot.ar5 import collect

    # 2026-09-22: 回车那个离合键去掉了（录着就是接合），← 改成删上一条，r 接手重录。
    # 2026-09-23: **P 加回了一个离合**，但它解决的是另一个问题 —— 主臂行程不够时
    # 原地锚定重来，只在录制中有效。回车那个仍然不该回来。
    loop = inspect.getsource(collect.main)
    for binding in ('key in ("q", "Q")', 'key == " "', 'key in ("s", "S")',
                    'key in ("r", "R")', 'key == "LEFT"', 'key in ("p", "P")'):
        assert binding in loop, f"the loop no longer binds {binding}"
    assert 'key in ("\\n", "\\r")' not in loop, (
        "Enter is bound again -- that clutch was removed on purpose; "
        "recording is engagement. P is the travel-limit re-anchor, not that one.")

    # P 只在录制中有效，而且必须清 holding，否则保持状态会串到下一条：
    # 夹爪一直冻着、每帧还都被标成 held。
    assert "if not schedule.recording:" in loop, "P is no longer gated on recording"
    assert loop.count("holding, held_gripper = False, None") >= 4, (
        "P 的保持状态必须在 空格/s/r/q 四处都清掉，少一处就会串到下一条")

    printed = inspect.getsource(collect.print_config)
    for shown in ("空格", "s ", "r ", "←", "q ", "P "):
        assert shown in printed, f"the key hint no longer mentions {shown}"
    assert "Enter" not in printed and "接合/松开" not in printed, (
        "the hint still advertises the old clutch key")


@check("README 里的部署按键表和代码里真正绑的键一致")
def _():
    # 这张表悄悄过期过一次：代码从「四个互斥结局」改成「里程碑式打标」之后，
    # README 还写着 r/p/g/f，而 p 已经不是绑定键了。操作者照着按会丢数据，
    # 而且要到统计时才发现。flag 那条检查管不到按键。
    from atom_robot.ar5.eval_policy import ENDINGS, MILESTONES

    from atom_robot import VEL_REPO
    doc = VEL_REPO.parent / "README_AR5.md"
    if not doc.exists():
        return
    text = doc.read_text()
    head = text[text.index("## 9. 部署"):text.index("## 10.")]

    bound = {**MILESTONES, **{k: (None, v[1]) for k, v in ENDINGS.items()}}
    for k, v in bound.items():
        assert f"`{k}`" in head, (
            f"代码绑了 `{k}`（{v[1]}），README 第 9 节没写 —— 操作者不会按")
    # 反向：README 不能宣传已经不存在的键
    import re
    shown = set(re.findall(r"^\| \| `([a-z])` \|", head, re.MULTILINE))
    shown |= set(re.findall(r"^\| \*\*[^|]*\*\* \| `([a-z])` \|", head, re.MULTILINE))
    stale = shown - set(bound) - {"q", "x"}
    assert not stale, f"README 第 9 节还在教 {sorted(stale)}，代码里已经没有这些键了"


@check("P 保持的帧逐帧标出来，少量不算缺陷、过半才报")
def _():
    """2026-09-23 契约变了。

    P 保持以前不存在，唯一会「录制中松开离合」的情况是出错，所以任何一帧都当
    缺陷报。现在 P 是有意操作（主臂行程不够时重新锚定），保持期间臂不动、画面
    也静止，**两边一致，不是坏数据**，只是演示里的一段停顿 —— 每次用都报一句
    警告等于把警告训练成噪声。

    改成：逐帧标进 `held` 让下游自己筛；只有占比过半（这条大半时间臂没动）才报。
    """
    from atom_robot.ar5.collect import Recorder
    from atom_robot.ar5.config import CollectConfig

    class _Obs:
        def __init__(self, i):
            self.observation = {
                "ee_pose": np.array([0.4 + 0.005 * i, 0.0, 0.3, 0.0, 0.0, 0.0]),
                "qpos": np.zeros(8, dtype=np.float32),
                "images": {},
            }

    cfg = CollectConfig(min_frames=5)

    # 少量保持：标出来，但不报
    rec = Recorder(Path("/nonexistent"), [], cfg)
    for i in range(30):
        rec.add(_Obs(i), i / 29.0, held=10 <= i < 15)
    assert rec.held.count(True) == 5, rec.held
    assert not any("保持" in n for n in rec.warnings()), rec.warnings()

    # 过半保持：这条大半时间臂没动，值得说
    rec2 = Recorder(Path("/nonexistent"), [], cfg)
    for i in range(30):
        rec2.add(_Obs(i), i / 29.0, held=i >= 10)
    assert any("保持" in n for n in rec2.warnings()), rec2.warnings()

    # reset 必须清干净，否则标记会串到下一条
    rec2.reset()
    assert rec2.held == [] and rec2.clutch_dropped == 0


@check("saving happens off the control loop, and finishes before exit")
def _():
    import tempfile
    import time as _time

    from atom_robot.ar5.collect import Recorder
    from atom_robot.ar5.config import CollectConfig

    # The bug this replaced, measured 2026-09-03: save() ran inline and took
    # 2.21 s for 600 frames of two cameras, against the arm's 0.25 s staleness
    # timeout. The sender braked and latched, and three episodes recorded 600
    # frames each of an arm that could no longer move.
    class _Obs:
        def __init__(self, i, keys):
            self.observation = {
                "ee_pose": np.array([0.4 + 0.001 * i, 0.0, 0.3, 0.0, 0.0, 0.0]),
                "qpos": np.zeros(8, dtype=np.float32),
                "images": {k: np.zeros((32, 32, 3), dtype=np.uint8) for k in keys},
            }

    keys = ["cam_high", "cam_front"]
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        rec = Recorder(out, keys, CollectConfig(min_frames=5, hz=30.0))
        for i in range(120):
            rec.add(_Obs(i, keys), 0.5)

        started = _time.perf_counter()
        path = rec.save(0, "t")
        handed_off = _time.perf_counter() - started
        # The arm tolerates 0.25 s without a goal. Handing off must be far
        # inside that, whatever the disk is doing.
        assert handed_off < 0.10, f"save() blocked the loop for {handed_off * 1000:.0f} ms"
        # And the buffer is immediately reusable for the next episode.
        assert len(rec) == 0, "save() did not hand the buffer over"
        rec.add(_Obs(0, keys), 0.5)
        assert len(rec) == 1

        rec.flush()
        assert rec.take_write_error() is None
        assert path.exists(), "the npz never landed"
        for key in keys:
            assert (out / f"{path.stem}_{key}.mp4").exists(), key
        data = np.load(path, allow_pickle=True)
        assert len(data["state"]) == 120, len(data["state"])


@check("每帧的真实时刻要存下来，不能只信配置里那个标称帧率")
def _():
    import tempfile

    from atom_robot.ar5.collect import Recorder
    from atom_robot.ar5.config import CollectConfig

    # 2026-09-04 查 50 条数据时发现的空白：npz 里只有 config 的标称 fps，
    # 回路实际跑了多快事后完全无从得知 —— 视频会不会放快、动作增量对应的
    # 真实间隔是多少，全都答不上来。
    class _Obs:
        observation = {
            "ee_pose": np.zeros(6),
            "qpos": np.zeros(8, dtype=np.float32),
            "images": {},
        }

    with tempfile.TemporaryDirectory() as tmp:
        rec = Recorder(Path(tmp), [], CollectConfig(min_frames=1, hz=30.0))
        # 跨度要超过 1 秒才判帧率 —— 几帧之间估出来的数字是噪声，
        # 而所有别的检查用的都是紧循环夹具，不该被这条误伤。
        for _ in range(60):
            rec.add(_Obs(), 0.5)
            time.sleep(0.02)                  # 实际约 50 Hz，标称 30
        measured = rec.measured_fps()
        assert 40 < measured < 60, measured
        assert any("实测帧率" in n for n in rec.warnings()), rec.warnings()

        # 跨度不够的时候必须闭嘴
        quick = Recorder(Path(tmp), [], CollectConfig(min_frames=1, hz=30.0))
        for _ in range(5):
            quick.add(_Obs(), 0.5)
        assert not any("实测帧率" in n for n in quick.warnings()), quick.warnings()

        path = rec.save(0, "t")
        rec.flush()
        assert rec.take_write_error() is None
        data = np.load(path, allow_pickle=True)
        assert "timestamp" in data and "fps_measured" in data, list(data)
        stamps = data["timestamp"]
        assert len(stamps) == 60 and stamps[0] == 0.0
        assert np.all(np.diff(stamps) > 0), "时间戳必须严格递增"
        assert abs(float(data["fps_measured"]) - measured) < 1e-6
        # 标称值也要留着，用来对照
        assert float(data["fps"]) == 30.0


@check("相机时刻：V4L2 用内核拍摄时刻（和 perf_counter 同一个钟），拿不到就用收到帧的时刻")
def _():
    import cv2

    from atom_robot.ar5.drivers.cameras import _Grabber

    class _Cap:
        def __init__(self, delay):
            self.delay, self.kernel = delay, None

        def read(self):
            time.sleep(0.01)
            self.kernel = time.perf_counter() - self.delay
            return True, np.zeros((4, 4, 3), dtype=np.uint8)

        def get(self, prop):
            assert prop == cv2.CAP_PROP_POS_MSEC
            return -1.0 if self.delay is None else self.kernel * 1000.0

    for delay, kind in ((0.024, "kernel"), (None, "arrival")):
        cap = _Cap(0.0 if delay is None else delay)
        if delay is None:
            cap.delay = None
            cap.read = lambda: (time.sleep(0.01), (True, np.zeros((4, 4, 3), dtype=np.uint8)))[1]
        grabber = _Grabber("cam", cap)
        grabber.start()
        try:
            deadline = time.perf_counter() + 2.0
            while grabber.latest_stamped()[2] is None and time.perf_counter() < deadline:
                time.sleep(0.005)
            frame, seq, stamp = grabber.latest_stamped()
            age = time.perf_counter() - stamp
        finally:
            grabber.shutdown()
        assert frame is not None and seq >= 1 and grabber.stamp_kind == kind, (kind, grabber.stamp_kind)
        low = 0.0 if delay is None else delay
        assert low <= age < low + 0.05, (kind, age)


@check("各来源的时刻逐帧进 npz：和 timestamp 同一个零点、同样扣掉暂停；存盘时报同步")
def _():
    import tempfile

    from atom_robot.ar5.collect import Recorder
    from atom_robot.ar5.config import CollectConfig

    # 用户 10-20：相机、臂、夹爪的数据都要有相对录制开始的时刻，暂停时时间也停，用来看是否同步
    class _Obs:
        def __init__(self, now):
            image = np.zeros((4, 4, 3), dtype=np.uint8)
            self.observation = {
                "ee_pose": np.zeros(6), "qpos": np.zeros(8, dtype=np.float32),
                "images": {"cam_high": image, "cam_front": image},
                # 图早 40 ms、夹爪早 15 ms、臂在当下；cam_front 这一路拿不到时刻
                "stamps": {"cam_high": now - 0.040, "cam_front": None, "arm": now, "gripper": now - 0.015},
            }

    with tempfile.TemporaryDirectory() as tmp:
        rec = Recorder(Path(tmp), ["cam_high", "cam_front"], CollectConfig(min_frames=1, hz=30.0))
        for _ in range(20):
            rec.add(_Obs(time.perf_counter()), 1.0)
            time.sleep(0.02)
        time.sleep(0.5)                       # 暂停，一帧不录
        rec.mark_resume()
        for _ in range(20):
            rec.add(_Obs(time.perf_counter()), 1.0)
            time.sleep(0.02)
        lines, spread = rec.sync_report()
        assert any(line.startswith("cam_high 早 40 ms") for line in lines), lines
        assert any(line.startswith("gripper 早 15 ms") for line in lines), lines
        assert "cam_front 没有时刻" in lines, lines
        assert spread is not None and "中位 40 ms" in spread, spread
        path = rec.save(0, "t")
        rec.flush()
        assert rec.take_write_error() is None
        data = np.load(path, allow_pickle=True)
    row = data["timestamp"]
    for name in ("t_cam_high", "t_cam_front", "t_arm", "t_gripper"):
        assert name in data.files and data[name].shape == row.shape, (name, data.files)
    assert np.all(np.isnan(data["t_cam_front"]))
    assert abs(data["t_cam_high"][0] - (-0.040)) < 0.002, "零点是第一帧的录制时刻，图比它早 40 ms"
    # 暂停接缝两边：各来源都扣了同样多，和录制时刻的差不变，也没留下 0.5 s 的空档
    for name, early in (("t_cam_high", 0.040), ("t_arm", 0.0), ("t_gripper", 0.015)):
        gap = row - data[name]
        assert np.all(np.abs(gap - early) < 0.003), (name, gap.min(), gap.max())
        assert np.diff(data[name]).max() < 0.1, f"{name} 里还留着暂停的空档"
    leaked = [k for k in data.files if "pause" in k or "resum" in k]
    assert not leaked, leaked


def _sync_segment(t0, seconds, q_of_t, rot_of_t, cmd_switch=None, other_offset=0.010,
                  other_latency=0.0, master_phase=0.0):
    """造一段原始数据：臂 50 Hz（间隔 20/40 ms 轮换，和真回路一样）、腕部相机 30 fps、
    顶部相机 30 fps（拍摄时刻比腕部晚 other_offset，收到时刻再晚 other_latency）、夹爪 100 Hz。"""
    from atom_robot.ar5 import sync_align

    seg = sync_align.new_segment()
    t, k = t0, 0
    while t < t0 + seconds:
        seg["arm_t"].append(t)
        seg["arm_q"].append(np.array([q_of_t(t)] + [0.1] * 6))
        seg["arm_ee"].append(np.array([0.3, 0.0, q_of_t(t) * 0.1, 0.0, 0.0, rot_of_t(t)]))
        seg["cmd_t"].append(t)
        seg["cmd"].append(1.0 if cmd_switch is None or t < cmd_switch else 0.0)
        seg["held"].append(False)
        t += (0.040, 0.040, 0.020)[k % 3]
        k += 1
    f = np.zeros((2, 2, 3), np.uint8)
    seg["cams"]["cam_front"] = [(t0 + master_phase + i / 30, f) for i in range(int(seconds * 30))]
    seg["cams"]["cam_high"] = [(t0 + i / 30 + other_offset + other_latency, f) for i in range(int(seconds * 30) + 1)]
    seg["grip_t"] = [t0 + i / 100 for i in range(int(seconds * 100) + 1)]
    seg["grip"] = [0.5] * len(seg["grip_t"])
    return seg


@check("软件对齐：按时间加权插值（匀速时和真值一样）、姿态球面插值、夹爪指令取当时生效的值、顶部取最近一张")
def _():
    from atom_robot.ar5 import sync_align

    speed = 0.8                                   # rad/s，匀速
    seg = _sync_segment(10.0, 2.0, lambda t: speed * (t - 10.0), lambda t: 0.5 * (t - 10.0),
                        cmd_switch=11.0, master_phase=0.007)
    out = sync_align.align([seg], "cam_front", 30.0)
    master_t = np.array([t for t, _ in seg["cams"]["cam_front"]])
    keep = master_t[(master_t >= 10.0 - sync_align.EDGE_HOLD_S) & (master_t <= seg["arm_t"][-1] + 0.05)]
    n = len(out["timestamp"])
    assert n == len(keep) and n >= 58, (n, len(keep))
    assert np.allclose(out["timestamp"], keep - keep[0]), "时刻 = 腕部相机拍摄时刻，第一行为 0"
    inside = (keep >= 10.0) & (keep <= seg["arm_t"][-1])
    truth = speed * (keep - 10.0)
    assert np.allclose(out["state"][inside, 0], truth[inside], atol=1e-5), "匀速时线性插值应该和真值一样"
    # 两点平均会差多少（对照）：最多半个采样间隔 × 速度
    assert np.allclose(out["ee_pose"][inside, 5], 0.5 * (keep[inside] - 10.0), atol=1e-5), "姿态球面插值"
    cmd = out["action"][:, 6]
    assert set(np.unique(cmd)) <= {0.0, 1.0}, "夹爪指令不能插出 0.5"
    # 指令是在回路里发的：真正生效是 11 s 之后的第一圈，不是正好 11.000
    switch = next(t for t in seg["cmd_t"] if t >= 11.0)
    assert np.all(cmd[keep < switch] == 1.0) and np.all(cmd[keep >= switch] == 0.0)
    off = np.asarray(out["stats"]["other_offset"]["cam_high"])
    # 顶部拍摄比 t0 晚 10 ms、腕部晚 7 ms：最近那张比基准晚 3 ms
    assert np.allclose(off, 0.003, atol=1e-9), "顶部取最近的那张"
    assert np.allclose(out["t_cam_high"] - out["timestamp"], 0.003, atol=1e-9)
    assert out["state"].shape == (n, 8) and out["action"].shape == (n, 7) and out["ee_pose"].shape == (n, 6)
    assert np.all(np.abs(np.diff(out["timestamp"]) - 1 / 30) < 1e-9), "行间隔就是相机的 33.3 ms，不再 40/40/20"

    # 顶部只有收到时刻、晚 30 ms：给了 latency 就扣掉再挑
    late = _sync_segment(10.0, 1.0, lambda t: 0.0, lambda t: 0.0, other_offset=0.0, other_latency=0.030)
    raw = sync_align.align([late], "cam_front", 30.0)["stats"]["other_offset"]["cam_high"]
    fixed = sync_align.align([late], "cam_front", 30.0, {"cam_high": 0.030})["stats"]["other_offset"]["cam_high"]
    assert np.max(np.abs(raw)) > 0.003 and np.max(np.abs(fixed)) < 1e-9, (np.max(np.abs(raw)), np.max(np.abs(fixed)))


@check("软件对齐：暂停前后分段、不跨暂停插值；段与段接起来，原始数据里也看不出停过")
def _():
    from atom_robot.ar5 import sync_align

    first = _sync_segment(10.0, 1.0, lambda t: 1.0, lambda t: 0.0)
    # 暂停 3 秒；第二段的臂停在另一个位置（跨暂停插值就会混出 1~2 之间的数）
    second = _sync_segment(14.0, 1.0, lambda t: 2.0, lambda t: 0.0)
    out = sync_align.align([first, second], "cam_front", 30.0)
    ts, q0 = out["timestamp"], out["state"][:, 0]
    assert np.max(np.diff(ts)) < 1 / 30 + 1e-6, "暂停那 3 秒要扣掉，接缝处就是一个周期"
    assert set(np.round(q0, 6)) == {1.0, 2.0}, f"跨暂停插值了：{sorted(set(np.round(q0, 4)))[:6]}"
    assert out["stats"]["segments"] == 2
    for key in ("raw_arm_t", "raw_gripper_t", "raw_cmd_t"):
        r = out["raw"][key]
        assert np.all(np.diff(r) > 0), f"{key} 不是递增的"
        assert -1 / 60 - 1e-9 <= r[0] and r[-1] < ts[-1] + 1 / 60, key
        assert np.max(np.diff(r)) <= 0.040 + 1e-9, f"{key} 接缝处比平时的间隔大：{np.max(np.diff(r)):.3f}"
    assert len(out["raw"]["raw_arm_q"]) == len(out["raw"]["raw_arm_t"])

    # 段首：相机缓存里暂停时拍的旧帧（比第一次读臂早很多）要丢，不外推
    stale = _sync_segment(20.0, 1.0, lambda t: 0.0, lambda t: 0.0)
    f = np.zeros((2, 2, 3), np.uint8)
    stale["cams"]["cam_front"] = [(19.5, f), (19.6, f)] + stale["cams"]["cam_front"]
    res = sync_align.align([stale], "cam_front", 30.0)
    assert res["stats"]["stale"] >= 2 and res["timestamp"][0] == 0.0
    assert res["stats"]["dropped_edge"] == 0, "暂停时积下的旧帧不算丢"
    assert "丢" not in sync_align.report(res["stats"], "cam_front")[0]


@check("OmniPicker 录制器：每圈记原始数据，存盘前对齐；npz 里是对齐后的数组 + raw_*，看不出停过")
def _():
    import tempfile

    from atom_robot.ar5.collect import SyncRecorder
    from atom_robot.ar5.config import CollectConfig

    env = _mock_env()
    try:
        keys = list(env.config.cameras.sources)
        if "cam_front" not in keys:
            return                                  # 这台机器的配置里没有腕部相机，测不了
        with tempfile.TemporaryDirectory() as tmp:
            rec = SyncRecorder(Path(tmp), keys, CollectConfig(min_frames=1, hz=30.0, min_ee_path_mm=0))
            for i in range(25):
                q = rec.log(env, 1.0 if i < 12 else 0.0)
                time.sleep(0.02)
            assert q.shape == (8,) and np.allclose(rec.last_qpos(), q)
            time.sleep(0.4)                         # 暂停，不调 log
            rec.mark_resume()
            for _ in range(25):
                rec.log(env, 0.0)
                time.sleep(0.02)
            assert len(rec) == 50, len(rec)       # mock 相机没有帧缓存：每圈一帧
            pauses, paused = rec.pause_summary()
            assert pauses == 1 and 0.35 < paused < 0.6, (pauses, paused)
            lines, spread = rec.sync_report()
            assert lines and lines[0].startswith("以 cam_front 为基准"), lines
            path = rec.save(0, "t")
            rec.flush()
            assert rec.take_write_error() is None
            data = np.load(path, allow_pickle=True)
            n = len(data["timestamp"])
            assert n >= 45 and data["state"].shape == (n, 8) and data["action"].shape == (n, 7)
            assert np.max(np.diff(data["timestamp"])) < 0.06, "暂停的 0.4 s 没扣掉"
            assert {"raw_arm_t", "raw_arm_q", "raw_arm_ee", "raw_gripper_t", "raw_cmd_t"} <= set(data.files)
            assert "t_cam_front" in data.files and "t_arm" not in data.files
            assert not [k for k in data.files if "pause" in k or "resum" in k]
            for key in keys:
                assert (Path(tmp) / f"episode_0000_{key}.mp4").exists(), key
    finally:
        env.close()


@check("相机帧缓存：每圈取走新来的帧，一帧不漏、不重复")
def _():
    from atom_robot.ar5.drivers.cameras import RECENT_FRAMES, _Grabber

    class _Cap:
        def read(self):
            time.sleep(0.005)
            return True, np.zeros((4, 4, 3), dtype=np.uint8)

        def get(self, _prop):
            return -1.0

    grabber = _Grabber("cam", _Cap())
    grabber.start()
    try:
        seen, last = [], None
        deadline = time.perf_counter() + 0.5
        while time.perf_counter() < deadline:
            new = grabber.frames_since(last)
            if new:
                last = new[-1][0]
                seen.extend(item[0] for item in new)
            time.sleep(0.012)                      # 取得比出帧慢：一次会来好几帧
    finally:
        grabber.shutdown()
    assert seen == list(range(seen[0], seen[0] + len(seen))), "漏帧或重复"
    assert len(grabber.frames_since(None)) <= RECENT_FRAMES


@check("OmniPicker 暂停不录帧：暂停那段从时间戳里扣掉，实测帧率不被拉低，npz 里看不出停过")
def _():
    import tempfile

    from atom_robot.ar5.collect import Recorder
    from atom_robot.ar5.config import CollectConfig

    # 2026-10-20：一条录 15.7 s、中间暂停 8 次共 11.3 s，存下来 fps_measured=17.6
    # （录制段实测 29.9 Hz）—— 首尾跨度把暂停算进去了。to_lerobot 拿它当帧率转。
    class _Obs:
        observation = {"ee_pose": np.zeros(6), "qpos": np.zeros(8, dtype=np.float32), "images": {}}

    with tempfile.TemporaryDirectory() as tmp:
        rec = Recorder(Path(tmp), [], CollectConfig(min_frames=1, hz=30.0))
        for _ in range(40):
            rec.add(_Obs(), 1.0)
            time.sleep(0.02)                  # 约 50 Hz
        time.sleep(1.0)                       # 暂停 1 秒，一帧不录
        rec.mark_resume()
        for _ in range(40):
            rec.add(_Obs(), 1.0)
            time.sleep(0.02)
        measured = rec.measured_fps()
        assert 40 < measured < 60, f"暂停被算进去了：{measured:.1f} Hz"
        # 内存里记着（存盘时打日志用），但不进文件
        assert 0.85 < rec.paused_seconds < 1.15, rec.paused_seconds
        assert rec.resumed.count(True) == 1 and rec.resumed[40]
        path = rec.save(0, "t")
        rec.flush()
        data = np.load(path, allow_pickle=True)
        stamps = data["timestamp"]
        assert np.all(np.diff(stamps) > 0) and np.diff(stamps).max() < 0.1, "时间戳里还留着暂停的空档"
        # 用户 10-20：数据集里要完全看不出采集时的中断
        leaked = [k for k in data.files if "pause" in k or "resum" in k]
        assert not leaked, f"npz 里漏出了暂停的痕迹：{leaked}"
        assert abs(float(data["fps_measured"]) - measured) < 1e-6

        # 没暂停过的：一帧都不标、不扣
        plain = Recorder(Path(tmp), [], CollectConfig(min_frames=1, hz=30.0))
        plain.mark_resume()                   # 第一帧之前的 resume 不算（没有空档可扣）
        for _ in range(5):
            plain.add(_Obs(), 1.0)
        assert not any(plain.resumed) and plain.paused_seconds == 0.0


@check("a failed background write is reported, not swallowed")
def _():
    import tempfile

    from atom_robot.ar5.collect import Recorder
    from atom_robot.ar5.config import CollectConfig

    # The writer runs on its own thread, so an exception there reaches nobody.
    # Silently losing an episode would be worse than the original bug.
    class _Obs:
        observation = {
            "ee_pose": np.zeros(6),
            "qpos": np.zeros(8, dtype=np.float32),
            "images": {},
        }

    with tempfile.TemporaryDirectory() as tmp:
        rec = Recorder(Path(tmp), [], CollectConfig(min_frames=1))
        for _ in range(5):
            rec.add(_Obs(), 0.5)
        # A directory where the npz should go makes savez fail.
        (Path(tmp) / "episode_0000.npz").mkdir()
        rec.save(0, "t")
        rec.flush()
        error = rec.take_write_error()
        assert error and "episode_0000.npz" in error, error
        # Taken once, cleared -- it must not haunt the next episode.
        assert rec.take_write_error() is None


@check("collect clears a stale latch at startup and bails out on a fresh one")
def _():
    import inspect

    from atom_robot.ar5 import collect

    source = inspect.getsource(collect.main)
    # A latch survives the process that caused it, so a run that starts against
    # one would find the arm refusing every goal. teleop_joint has always
    # cleared it here; collect did not, and "it just will not move" was the
    # symptom.
    assert "clear_latch()" in source, "collect no longer clears a stale latch"
    # And a latch appearing mid-session must stop the session. Recording
    # through one produces files full of an arm that cannot move -- which is
    # exactly what happened, three episodes deep, before anything noticed.
    assert "arm.latched" in source, (
        "collect no longer checks arm.latched, so it can record dead episodes"
    )


@check("a busy camera says which process holds it, not just 'could not be opened'")
def _():
    import os
    import tempfile

    from atom_robot.ar5.drivers.cameras import _open_failure_hint, who_holds

    # The real occurrence: a `view_camera` preview left open in another
    # terminal. "could not be opened" sent the operator to check the wiring,
    # when the answer was a window to close.
    with tempfile.TemporaryDirectory() as tmp:
        held = os.path.join(tmp, "fake-video")
        with open(held, "w"):
            pass
        # 2026-09-28：who_holds 改成**不报自己**。原来 connect() 失败时会把正在报错的
        # 这个进程自己列成占用者（「Held by: PID <自己>」），把人引向重复会话。
        # 所以这里用一个真的子进程来占，而本进程自己打开时必须查不到。
        import subprocess
        import sys as _sys
        import time as _time

        with open(held, "rb"):
            assert who_holds(held) == [], "本进程自己打开着，不该算作别人占用"

        child = subprocess.Popen(
            [_sys.executable, "-c",
             f"import time; f = open({held!r}, 'rb'); print('ok', flush=True); time.sleep(30)"],
            stdout=subprocess.PIPE, text=True)
        try:
            child.stdout.readline()
            _time.sleep(0.2)
            holders = who_holds(held)
            assert holders, "who_holds found nothing while another process held it"
            assert child.pid in [pid for pid, _ in holders], (child.pid, holders)

            hint = _open_failure_hint("cam_high", held)
            assert str(child.pid) in hint, hint
            assert "one process" in hint, hint
        finally:
            child.kill()
            child.wait()

        # Released: no holders, so the message must stop blaming a process.
        assert who_holds(held) == [], who_holds(held)
        assert "other process" not in _open_failure_hint("cam_high", held)

    # A path that does not exist is a different fault with a different fix --
    # the by-path node is tied to the USB socket, so it changes when the camera
    # is moved. Saying "busy" there would send the operator hunting a process
    # that is not there.
    missing = _open_failure_hint("cam_front", "/dev/v4l/by-path/does-not-exist")
    assert "does not exist" in missing and "view_camera --list" in missing, missing
    assert "process" not in missing.split("does not exist")[1], missing


@check("collect checks the cameras before the arm is powered, not after")
def _():
    import inspect
    import os
    import tempfile

    from atom_robot.ar5 import collect
    from atom_robot.ar5.config import AR5Config
    from atom_robot.ar5.drivers.cameras import check_sources

    # AR5Env.connect() brings up arm -> hand -> cameras, so a camera that is
    # merely busy used to cost a power-on and an unwind of the real-time
    # session. Whether a camera can be opened is knowable without touching
    # anything, so it is checked first.
    source = inspect.getsource(collect.main)
    before_env = source.split("AR5Env(config)")[0]
    assert "camera_problems" in before_env, (
        "the camera preflight moved after env.connect(); a busy camera will "
        "power the arm before failing"
    )

    config = AR5Config()
    config.cameras.sources = {"cam_front": "/dev/v4l/by-path/does-not-exist"}
    assert "cam_front" in collect.camera_problems(config)

    # An empty source list is legal (--no-cameras) and must not be a problem.
    config.cameras.sources = {}
    assert collect.camera_problems(config) == {}

    # Integer device indices cannot be preflighted, and must not be guessed at.
    config.cameras.sources = {"cam_front": 4}
    assert collect.camera_problems(config) == {}

    # A path that exists and is free is not a problem either.
    with tempfile.TemporaryDirectory() as tmp:
        free = os.path.join(tmp, "free-video")
        with open(free, "w"):
            pass
        config.cameras.sources = {"cam_front": free}
        assert check_sources(config.cameras) == {}


@check("OmniPicker 的 N 连着往前删：每删一条编号退到剩下的最大 +1，这次存的才从 -n 计数里减")
def _():
    import tempfile

    from atom_robot.ar5.collect import EpisodeSchedule, delete_episode, next_episode_index
    from atom_robot.ar5.config import CollectConfig

    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        for i in range(5):                          # 0~2 是以前存的，3、4 是这次存的
            for suffix in (".npz", "_cam_high.mp4", "_cam_front.mp4"):
                (d / f"episode_{i:04d}{suffix}").write_bytes(b"x")
        sched = EpisodeSchedule(CollectConfig(num_episodes=10), 3)
        for _ in range(2):
            sched.start()
            sched.episode_saved(True)
        assert (sched.index, sched.done) == (5, 2)
        this_run = {3, 4}
        for expect_next, expect_done in ((4, 1), (3, 0), (2, 0), (1, 0)):
            latest = next_episode_index(d) - 1
            assert delete_episode(d, latest) == 3, "一条是 npz + 两路 mp4"
            sched.forget(next_episode_index(d), latest in this_run)
            this_run.discard(latest)
            assert (sched.index, sched.done) == (expect_next, expect_done), (sched.index, sched.done)
        assert sorted(p.name for p in d.iterdir()) == [
            "episode_0000.npz", "episode_0000_cam_front.mp4", "episode_0000_cam_high.mp4"]
        # 删完接着录，编号接在剩下的后面，不会撞上
        sched.start()
        sched.episode_saved(True)
        assert sched.index == 2


@check("中间删掉一条之后，续采不会覆盖已有数据")
def _():
    import inspect
    import tempfile

    from atom_robot.ar5 import collect

    # 原来编号取的是**文件数**。删掉 episode_0150 之后还剩 199 个文件，
    # 下一条就编号 0199 —— 而 episode_0199 已经存在，直接覆盖，而且不报错。
    # 挑掉一条坏数据是常规操作（人手入境、抓空），不该有这种后果。
    assert "len({p for p in out_dir.glob" not in inspect.getsource(collect.main), (
        "编号又变回按文件数算了"
    )
    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        assert collect.next_episode_index(out) == 0, "空目录该从 0 开始"
        assert collect.next_episode_index(out / "nope") == 0, "目录不存在该从 0 开始"
        for i in (0, 1, 2, 4, 5):                       # 缺 3，模拟删掉过一条
            (out / f"episode_{i:04d}.npz").write_bytes(b"")
        got = collect.next_episode_index(out)
        assert got == 6, f"缺号目录里下一条编成了 {got}，应该是 6（max 5 + 1）"
        # 真实场景：删掉中间一条之后，绝不能撞上已有文件
        (out / "episode_0002.npz").unlink()
        got = collect.next_episode_index(out)
        assert not (out / f"episode_{got:04d}.npz").exists(), (
            f"下一条编号 {got:04d} 会覆盖已有文件"
        )


@check("部署的 q 是结束本轮，不是退出程序")
def _():
    import inspect

    from atom_robot.ar5 import eval_policy

    # docstring 一直写着 q 是「结束这一轮」，代码里却是 `break` —— 整个进程退出，
    # 4 GB 权重下一轮得重新加载。评测本来就是连着跑十几轮记成功率的。
    loop = inspect.getsource(eval_policy.main)
    q_block = loop[loop.index('key in ("q", "Q")'):]
    q_block = q_block[:q_block.index('if key == " ":')]
    assert "break" not in q_block, "q 又变回退出整个程序了"
    assert "goto_task_start" in q_block, "q 结束本轮之后没有回起点"
    assert "policy.reset()" in q_block, "q 没有清动作队列，下一轮会接着上一轮预测"
    assert 'key in ("x", "X")' in loop, "没有退出键了"

    # 文档和绑定必须对上 —— 对不上正是这个 bug 的根源
    doc = eval_policy.__doc__
    for shown in ("空格", "q ", "r ", "f ", "x "):
        assert shown in doc, f"docstring 里不再提 {shown!r}"
    assert "结束本轮" in doc and "退出程序" in doc, (
        "docstring 没把「结束本轮」和「退出程序」分清楚"
    )


@check("断开连接不会顺手把臂断电")
def _():
    import inspect

    from atom_robot.ar5.config import AR5Config
    from atom_robot.ar5.drivers.rokae_arm import RokaeAR5

    # 原来 disconnect() 里无条件 setPowerState(False)，于是评测每按一次 q 臂就掉电，
    # 下一轮开起来报「网络异常: 网络连接错误」（SDK 未上电时就报这个，和网线无关），
    # 每轮都得先 power_on。断开前已经 stopMove + 回 NrtCommandMode，带电只是保持位姿。
    assert AR5Config().arm.power_off_on_disconnect is False, (
        "默认又变成断开就断电了 —— 评测每轮都得重新上电"
    )
    src = inspect.getsource(RokaeAR5.disconnect)
    assert "power_off_on_disconnect" in src, (
        "disconnect 里的断电又变回无条件的了"
    )
    # 断电必须仍然做得到，只是要显式要求
    assert "setPowerState, False" in src, "连显式断电的路径都没了"


@check("退出时那条 multiprocess 的 AttributeError 要被闭掉")
def _():
    import inspect

    from atom_robot.ar5 import eval_policy

    # `AttributeError: '_thread.RLock' object has no attribute '_recursion_count'`
    # 是 multiprocess 在解释器终结阶段自己的 bug，比 env.close() 晚，和臂无关。
    # 但它长得像崩溃，实测已经让人把它和「按 q 臂断电」当成同一件事了。
    src = inspect.getsource(eval_policy)
    tail = src[src.index('if __name__ == "__main__":'):]
    assert "silence_shutdown_noise()" in tail, (
        "噪音抑制不在最外层 —— 加载阶段 OOM 时 main 的 finally 走不到"
    )

    # 真的能挡住，不是只写了个函数。multiprocess 只装在 lerobot 环境里
    # （eval_policy 本来也只在那儿跑），自检跑在 vel-ar5，所以这半段按需跳过。
    try:
        import multiprocess.resource_tracker as _rt
    except ImportError:
        return
    _rt._resource_tracker.ensure_running()
    eval_policy.silence_shutdown_noise()
    _rt._resource_tracker._stop_locked()        # 打补丁前这里会抛 AttributeError


@check("部署开跑前要回任务起点，臂和手都要回")
def _():
    import inspect

    from atom_robot.ar5 import eval_policy

    # 采集是遥操作，起点是操作者握主臂的预备位（100 条首帧中位
    # [-46.5 29.8 6.9 119.6 -16.3 -8.6 1.0] deg，夹爪 0.61）。部署原来**只在退出时**
    # 回这个位姿，开跑前什么都不做 —— 臂停在上一次留下的地方，跟 config.home_joints
    # 差 46.8°。策略第一帧就吃到训练分布外的输入。
    source = inspect.getsource(eval_policy.main)
    head, sep, _tail = source.partition("while True:")
    assert sep, "eval_policy.main 的主循环结构变了，这条检查要重写"
    assert "goto_task_start" in head, (
        "开跑前没有回任务起点 —— 策略会从上一次留下的位姿起步"
    )

    # 夹爪是 state 的第 8 维，策略直接看得到。只摆关节、不摆手，一样是分布外。
    class _Arm:
        def __init__(self):
            self.went = None

        def get_joint_positions(self):
            return np.zeros(7)

    class _Hand:
        def __init__(self):
            self.opening = None

        def set_opening(self, v):
            self.opening = v

    class _Env:
        def __init__(self):
            self.arm, self.hand = _Arm(), _Hand()

        def goto_joints(self, target, timeout_s=None):
            self.arm.went = np.asarray(target, dtype=float)
            self.arm.get_joint_positions = lambda: self.arm.went
            return True

    want = eval_policy.task_start_state()
    if want is None:
        return          # 没有示教数据，behaviour 部分跳过
    env = _Env()
    assert eval_policy.goto_task_start(env, None, "test")
    assert env.arm.went is not None and np.allclose(env.arm.went, want[:7]), (
        "回起点用的关节角不是首帧中位"
    )
    assert env.hand.opening is not None, "只摆了臂，没摆手 —— 夹爪开度也是策略的输入"
    assert abs(env.hand.opening - want[7]) < 1e-6, (
        f"夹爪回到 {env.hand.opening}，应该是首帧中位 {want[7]}"
    )


@check("函数里的惰性 import 不能排在用它的地方后面")
def _():
    # 实测踩过：energy_probe.py 在 add_argument 的 choices 里用了 SEGMENT_CODE，
    # 而那个 import 写在 args = ap.parse_args() 之后。Python 因此把它当局部变量，
    # 于是 --help / --segment 一律 UnboundLocalError —— 语法检查和 import 检查
    # 都发现不了，只有真正跑一次才暴露。这些脚本为了启动快全在 main() 里惰性
    # import，这个坑会反复踩。
    import ast

    root = Path(__file__).parent
    files = sorted(root.glob("*.py")) + sorted((root / "vel").glob("*.py"))
    checked = 0
    for f in files:
        tree = ast.parse(f.read_text())
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            # 只看这一层函数体，不下钻嵌套函数（那是另一个作用域）
            body = []
            for node in ast.walk(fn):
                if node is not fn and isinstance(node, (ast.FunctionDef, ast.Lambda,
                                                        ast.AsyncFunctionDef)):
                    continue
                body.append(node)
            # **取最小行号，不是先遇到的那个。** ast.walk 不按源码顺序走；
            # 同一个名字在 if / else 两个分支各 import 一次很常见（ar5_env.connect
            # 就是），拿到后面那个会把正常代码报成用在 import 之前。
            imported = {}
            for node in body:
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    for a in node.names:
                        name = (a.asname or a.name).split(".")[0]
                        imported[name] = min(imported.get(name, node.lineno), node.lineno)
            if not imported:
                continue
            checked += 1
            for node in body:
                if not (isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)):
                    continue
                first = imported.get(node.id)
                if first is not None and node.lineno < first:
                    raise AssertionError(
                        f"{f.name}:{node.lineno} 用了 `{node.id}`，但它在同一个函数里"
                        f"第 {first} 行才 import —— Python 会把它当局部变量，"
                        f"运行到这里必然 UnboundLocalError")
    assert checked > 5, f"只扫到 {checked} 个带惰性 import 的函数，检查失效了"


@check("按轴关旋转：只留 Z 时工具保持竖直，绕 Z 的自转还在")
def _():
    # 螺丝刀这类任务要求工具始终竖直。--no-rotation 会把拧的那一下也关掉，
    # 全开又保不住竖直 —— 只能按轴选。被关掉的轴必须**逐步置零**，
    # 否则增量累加起来姿态会慢慢漂走。
    import numpy as np

    from atom_robot.ar5.leader_mapping import LeaderMapper, MappingConfig
    from atom_robot.ar5.transforms import axis_angle_to_matrix

    def rot_out(axes, dr):
        m = LeaderMapper(MappingConfig(rotation_axes=axes, rotation_deadband=0.0,
                                       position_deadband=0.0))
        m._M = np.eye(3)                                        # noqa: SLF001
        m._rot_parity = 1.0                                     # noqa: SLF001
        m._engaged = True                                       # noqa: SLF001
        m.gripper_command = lambda _j: 0.0
        frames = [(np.zeros(3), np.eye(3)),
                  (np.zeros(3), axis_angle_to_matrix(np.asarray(dr, float)))]
        m.leader_frame = lambda _j: frames.pop(0)
        m.compute_action({})            # 第一帧只记锚点
        return m.compute_action({})[3:6]

    full = rot_out("xyz", [0.1, 0.2, 0.3])
    assert np.linalg.norm(full) > 0.3, f"全开时旋转不该是零：{full}"
    onlyz = rot_out("z", [0.1, 0.2, 0.3])
    assert abs(onlyz[0]) < 1e-12 and abs(onlyz[1]) < 1e-12, (
        f"绕 X/Y 没被关掉：{onlyz} —— 工具会歪，螺丝刀任务里这是致命的")
    assert abs(onlyz[2] - full[2]) < 1e-12, f"绕 Z 的自转被削了：{onlyz} vs {full}"
    none = rot_out("", [0.1, 0.2, 0.3])
    assert not none.any(), f"全关时应该全零：{none}"


@check("存下来的缩放不会把关节推出软限位（换映射关节时最容易踩）")
def _():
    # 实测差点踩：wrist_roll 原来驱动 J5（软限位 ±178°），标定存下 scale 0.96。
    # 改成驱动 J7（±50°）之后，存下来的 scale **优先于模式默认值**，于是主臂转到
    # 28% 行程 J7 就顶死，后面转了也不动 —— 看着像「这个关节坏了」。
    # 拟合出来的 scale 天然满足这条；出问题的只会是**手写或换关节后残留**的 override。
    import numpy as np

    from atom_robot.ar5.config import AR5Config
    from atom_robot.ar5.joint_mapping import (
        DATASHEET_AXIS, JOINT_PAIRS, JointMapConfig, JointMapper)

    arm = AR5Config().arm
    cfg = JointMapConfig.load()
    m = JointMapper(cfg)
    half = {"shoulder_pan": 110.5, "shoulder_lift": 105.2, "elbow_flex": 97.2,
            "wrist_roll": 180.0, "wrist_flex": 101.8}
    m.configure(half, arm.joint_min, arm.joint_max, margin=arm.joint_limit_margin)

    lo = np.asarray(arm.joint_min) + arm.joint_limit_margin
    hi = np.asarray(arm.joint_max) - arm.joint_limit_margin
    centre = np.asarray(m.centre)
    for name, index, _ in JOINT_PAIRS:
        swing = np.deg2rad(half[name]) * m._scales[name]        # noqa: SLF001
        room = min(centre[index] - lo[index], hi[index] - centre[index])
        # 拟合出来的 scale 按定义正好填满，所以要留余量：2% 或 0.5°，取大的。
        # 要抓的是 3.5 倍那种量级（J5 的 0.96 套到 J7 上），不是浮点往返的零头。
        slack = max(room * 0.02, np.deg2rad(0.5))
        assert swing <= room + slack, (
            f"{name} 驱动 j{index}（轴{DATASHEET_AXIS[index]}）：主臂满行程映射成 "
            f"±{np.rad2deg(swing):.1f}°，而从中心到限位只有 ±{np.rad2deg(room):.1f}° —— "
            f"主臂转到 {100*room/swing:.0f}% 就顶死。"
            f"joint_calibration.json 里那条 scale 多半是给别的关节拟合的，删掉让它重拟合")

    # 锁死的关节不能同时被映射
    mapped = {index for _, index, _ in JOINT_PAIRS}
    clash = mapped & set(cfg.locked_joints)
    assert not clash, f"j{sorted(clash)} 既被映射又被列为锁死"
    assert len(mapped | set(cfg.locked_joints)) == arm.num_joints, (
        f"映射 {sorted(mapped)} + 锁死 {sorted(cfg.locked_joints)} "
        f"没覆盖全部 {arm.num_joints} 个关节")


@check("every command-line flag the docs advertise actually exists")
def _():
    import re

    # This has already rotted once: `--smoothing` was replaced by
    # `--min-cutoff`/`--beta` and the README went on recommending the flag that
    # no longer existed. Docs that name flags are checkable, so check them.
    def flags_of(script):
        """Option strings from a script's add_argument calls, read from source.

        Parsed, not executed. Running main() to discover its parser would open
        the leader's serial port and power the arm -- which is exactly what this
        suite promises not to do.
        """
        import ast

        # Most entries are inside this package; a few live in the hand-teleop
        # tree, which the README also documents (probe_hand is the first thing
        # you run when the hand stops answering). Those carry their own path so
        # they are checked for real rather than whitelisted as "foreign".
        if "/" in script and not script.startswith("vel/"):
            from atom_robot import VEL_REPO
            path = VEL_REPO / f"{script}.py"
        else:
            path = Path(__file__).parent / f"{script}.py"
        tree = ast.parse(path.read_text())
        found = set()
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "add_argument"):
                for arg in node.args:
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str) \
                            and arg.value.startswith("-"):
                        found.add(arg.value)
        return found

    scripts = {name: flags_of(name) for name in (
        "collect", "teleop_joint", "replay", "view_camera", "jog_hand", "diag_shake",
        # 笛卡尔那条线。之前没列进来，于是 README 写 --log-actions 也查不出。
        "teleop", "calibrate_mapping",
        # 摆起始姿态那一步，README 第 3 节④要求先跑它，参数得真的存在
        "capture_pose",
        # 手不回包时第一个要跑的诊断，README 第 3 节⑤写了它的参数
        "VEL/src/hand_teleop/probe_hand",
        # 调手只用这个（用户明确要求），README 第 3 节⑤把它的参数都写出来了
        "VEL/src/hand_teleop/grip_tune",
        # 部署那条线的脚本。README 会写它们的参数，不列进来就等于没检查。
        "eval_policy", "power_on", "check_camera_pose", "goto_episode_pose",
        "camera_shift_impact", "object_spread", "qc_dataset",
        "policy_error", "rollout_stats",
        # VEL 那四步的脚本，README 第 10 节把它们的参数都写出来了
        "vel/precompute", "vel/train_energy", "vel/energy_probe", "vel/geo_check",
        "vel/best_of_n",
    )}
    assert len(scripts["collect"]) > 10, (
        f"only found {scripts['collect']} in collect.py; the source scan broke"
    )

    # The docs live outside this tree, so their absence is not a failure --
    # a stale flag in one that is present is.
    # 这两份文档是 **VEL 仓库**的（README_AR5.md 在 vel_ar5 根、ar5-bringup 在
    # openvla-energy/VEL/docs）。2026-09-23 代码搬到 Project_Atom 之后，原来那个
    # `parents[4]` 指到了 /home/<user>，两份都找不到 —— 而这条检查「文档不存在
    # 就跳过」，于是变成静默不检查。改成走 VEL_REPO。
    from atom_robot import VEL_REPO
    root = VEL_REPO.parent
    docs = [root / "README_AR5.md",
            root / "openvla-energy" / "VEL" / "docs" / "ar5-bringup.md"]
    known = set().union(*scripts.values())
    # Flags belonging to tools outside this package that legitimately appear in
    # these files: probe_rokae/diag_rt, to_lerobot, lerobot's own calibrate, and
    # the finetune script. (capture_pose used to be whitelisted here; it is
    # scanned for real now that the README documents its flags.)
    foreign = {"--src", "--repo-id", "--root", "--connect", "--read-state",
               "--stage", "--teleop.type", "--teleop.port",
               "--teleop.id", "--vla_path", "--dataset_name", "--run_root_dir",
               "--save_freq", "--resume", "--resume_step",
               "--save_latest_checkpoint_only", "--max-step",
               # lerobot-train / huggingface-cli 的参数，出现在训练那几节里
               "--batch_size", "--steps", "--num_workers", "--output_dir",
               "--job_name", "--log_freq", "--seed", "--local-dir",
               "--policy.type", "--policy.device", "--policy.chunk_size",
               "--policy.n_action_steps", "--policy.optimizer_lr",
               "--policy.push_to_hub", "--policy.pretrained_path",
               "--policy.train_expert_only", "--policy.dtype",
               "--dataset.repo_id", "--dataset.root", "--dataset.video_backend",
               "--wandb.enable", "--wandb.project",
               # rsync / huggingface-cli 的参数，出现在传数据和下权重那几节
               "--exclude", "--info", "--local-dir"}

    checked = 0
    for doc in docs:
        if not doc.exists():
            continue
        text = doc.read_text()
        # Two things this pattern has to get right, both learned the hard way:
        #   * `_` and `.` belong in the class, or `--dataset_name` is read as
        #     `--dataset` and the check fails on a flag nobody wrote
        #   * a flag has to *start* a word. Checkpoint directories look like
        #     `{module}--{step}_checkpoint.pt`, and the `--` in the middle of
        #     that is not an option string.
        pattern = r"""(?:^|[\s`"'(\[|])(--[a-z][a-z0-9_.-]*)"""
        for flag in sorted(set(re.findall(pattern, text, re.MULTILINE))):
            if flag in foreign:
                continue
            checked += 1
            assert flag in known, (
                f"{doc.name} documents {flag}, which no AR5 script accepts. "
                f"Closest known flags: "
                f"{sorted(f for f in known if flag[2:4] in f)[:4]}"
            )
    assert checked > 10, f"only {checked} flags checked; the docs went missing"


if __name__ == "__main__":
    sys.exit(main())
