#!/usr/bin/env python3
"""遥操时的实时曲线。盯 J7 离 17 Nm 还有多远，别等抱死了才知道。

    # 终端 A：跑遥操，逐行把轨迹写出来
    python -m atom_robot.ar5.teleop --no-cameras --no-hand \
        --rotation-axes z --hz 50 --live /tmp/live.csv --log-actions ~/screw_x.npz

    # 终端 B：画
    python -m atom_robot.ar5.live_plot /tmp/live.csv

**它不碰机器人。** 只读那个 CSV，所以不会多开一个 SDK 连接、不会跟 1 kHz 发送线程
抢 GIL。画图放在遥操进程里是行不通的：matplotlib 渲染一帧几十毫秒且拿着 GIL，
和当初把 `getEndTorque`(41 ms) 放进循环是同一个错误 —— 发送率会从 963/s 掉到
781/s，控制器把间隔当丢包、**停止执行却继续收指令**（q_cmd 一直走、q_actual 冻住、
控制器日志一条错都没有）。

基线：接合之前那一段当零点，和 `analyse_runs` 一致。还没接合过就用已有的全部帧，
所以刚起来那几秒的数值会跳，属正常。

Fz 的符号统一到控制器的约定（`cart_force` = 末端**受到**的力），**正 = 在往下压**。
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

RSC_J7_LIMIT = 17.0
# 和 teleop 的 docstring 对齐。66 列才有控制器自己的估计；53 列只有本地反推。
LAYOUT = {
    53: {"tau": 40, "jac": 47, "est": None, "ext": None},
    66: {"tau": 40, "jac": 60, "est": 54, "ext": 47},
}


def read_rows(path: Path, seen: int):
    """从第 seen 行之后继续读。返回 (新行数组, 现在读到第几行)。

    **最后一行可能正写到一半**（teleop 攒几行才 flush）。文本不是以换行结尾时把
    末行整个丢掉、**而且不计进游标** —— 计进去的话等 teleop 把它写完，这一行就
    永远读不到了，曲线会缺一小段。
    """
    try:
        text = path.read_text()
    except OSError:
        return None, seen
    lines = text.splitlines()
    if lines and not text.endswith("\n"):
        lines = lines[:-1]
    if len(lines) <= seen:
        return None, seen
    rows = []
    for ln in lines[seen:]:
        parts = ln.split(",")
        if len(parts) not in LAYOUT:
            continue
        try:
            rows.append([float(v) for v in parts])
        except ValueError:
            continue
    return (np.array(rows) if rows else None), len(lines)


def channels(a: np.ndarray, lay: dict):
    """扣完基线的四路：(J7 力矩绝对值, 末端力 3, 末端力矩 3, 关节外力 7, 末端 z mm)。

    基线取接合之前那一段，和 `analyse_runs` 一致 —— `payload_mass` 是 0 而手加
    工具约 1.2~1.3 kg，那份重力在每一路里都没被扣掉，所以只有变化量有意义。
    还没接合过就拿已有的全部帧当基线，刚起来那几秒数值会跳，属正常。
    """
    eng = a[:, 1] > 0.5
    base = np.zeros(len(a), bool)
    if eng.any():
        base[: np.argmax(eng)] = True
    if base.sum() < 3:
        base[:] = True

    j7 = np.abs(a[:, lay["tau"] + 6])
    if lay["est"] is not None:
        wr = a[:, lay["est"]:lay["est"] + 6] - a[base, lay["est"]:lay["est"] + 6].mean(0)
        ext = a[:, lay["ext"]:lay["ext"] + 7] - a[base, lay["ext"]:lay["ext"] + 7].mean(0)
    else:
        # 53 列没有控制器的估计。本地反推是「臂施加的力」，取反统一到
        # 控制器约定（`cart_force` = 末端受到的力），正 = 在往下压。
        wr = -(a[:, lay["jac"]:lay["jac"] + 6] - a[base, lay["jac"]:lay["jac"] + 6].mean(0))
        ext = None
    return j7, wr[:, :3], wr[:, 3:], ext, a[:, 14] * 1000, eng


def run_print(path: Path, fps: float) -> int:
    """终端数字模式：不开窗口，按行刷新。没有 X11 时也能用。"""
    print(f"  盯着 {path}（--print 模式，不开窗口）")
    while not path.exists():
        time.sleep(0.3)
    buf, seen, lay = [], 0, None
    period = 1.0 / max(fps, 0.5)
    hdr = False
    try:
        while True:
            new, seen = read_rows(path, seen)
            if new is not None and len(new):
                if lay is None:
                    lay = LAYOUT[new.shape[1]]
                buf.append(new)
            if buf:
                a = np.vstack(buf)
                j7, F, M, ext, z, eng = channels(a, lay)
                if not hdr:
                    src = "控制器 cart_force" if lay["est"] else "本地 pinv(J^T)·τ（取反）"
                    print(f"  末端力来源：{src}"
                          + ("" if lay["est"] else "　（53 列，没有关节外力估计）"))
                    print(f"\n  {'J7/17.0':>12} {'末端力 Fx Fy Fz (N)':>26}"
                          f" {'末端力矩 Mx My Mz (Nm)':>26}"
                          + (f" {'关节外力 J1..J7 (Nm)':>40}" if ext is not None else ""))
                    hdr = True
                pct = j7[-1] / RSC_J7_LIMIT * 100
                mark = "!!" if pct > 90 else (" !" if pct > 75 else "  ")
                line = (f"  {j7[-1]:6.2f} {pct:3.0f}%{mark}"
                        f" {F[-1, 0]:+8.1f}{F[-1, 1]:+8.1f}{F[-1, 2]:+8.1f}"
                        f" {M[-1, 0]:+8.2f}{M[-1, 1]:+8.2f}{M[-1, 2]:+8.2f}")
                if ext is not None:
                    line += " " + "".join(f"{v:+6.1f}" for v in ext[-1])
                line += f"  {'接合' if eng[-1] else '松开'}"
                print(line, end="\r", flush=True)
            time.sleep(period)
    except KeyboardInterrupt:
        print("\n  停了")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?", default="/tmp/live.csv")
    ap.add_argument("--window", type=float, default=30.0, help="横轴显示多少秒")
    ap.add_argument("--fps", type=float, default=10.0, help="刷新率")
    ap.add_argument("--print", dest="print_only", action="store_true",
                    help="只在终端打数，不开窗口。没有图形界面 / 只想读数时用。这个模式没有暂停键，要停就 Ctrl-C")
    args = ap.parse_args()

    if args.print_only:
        return run_print(Path(args.path).expanduser(), args.fps)

    import matplotlib
    from matplotlib import pyplot as plt
    for cand in ("Noto Sans CJK JP", "WenQuanYi Micro Hei", "Droid Sans Fallback"):
        matplotlib.rcParams["font.sans-serif"] = [cand]
        break
    matplotlib.rcParams["axes.unicode_minus"] = False

    path = Path(args.path).expanduser()
    print(f"  盯着 {path}")
    print(f"  等 teleop 开始写 ...（teleop 要带 --live {path}）")
    while not path.exists():
        time.sleep(0.3)

    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    fig.canvas.manager.set_window_title("AR5 遥操实时曲线")
    l_j7 = axes[0].plot([], [], lw=1.2, color="tab:red")[0]
    l_F = [axes[1].plot([], [], lw=1.0, label=n)[0] for n in ("Fx", "Fy", "Fz")]
    l_M = [axes[2].plot([], [], lw=1.0, label=n)[0] for n in ("Mx", "My", "Mz")]
    l_E = [axes[3].plot([], [], lw=.9, label=f"J{j+1}")[0] for j in range(7)]

    axes[0].axhline(RSC_J7_LIMIT, color="crimson", ls="--", lw=1.6)
    axes[0].axhline(RSC_J7_LIMIT * 0.8, color="orange", ls=":", lw=1.2)
    axes[0].set_ylabel("|J7 力矩| (Nm)")
    axes[0].set_ylim(0, RSC_J7_LIMIT * 1.15)
    axes[1].set_ylabel("末端力 (N)\n正 Fz = 往下压")
    axes[2].set_ylabel("末端力矩 (Nm)")
    axes[3].set_ylabel("关节空间外力\n估计 (Nm)")
    axes[3].set_xlabel("时间 (s)　　空格 暂停/继续　q 退出")
    for ax in axes:
        ax.grid(alpha=.3)
    for ax in axes[1:3]:
        ax.legend(fontsize=8, ncol=3, loc="upper left")
    axes[3].legend(fontsize=7, ncol=7, loc="upper left")
    banner = axes[0].text(0.01, 0.94, "", transform=axes[0].transAxes,
                          fontsize=10, va="top", fontweight="bold")
    # 不用 monospace：那几个字体没有中文字形，中文会渲染成方框
    readout = fig.text(0.01, 0.005, "", fontsize=9, va="bottom")
    plt.tight_layout(rect=(0, 0.035, 1, 1))
    plt.show(block=False)

    # 空格暂停：**只冻画面，照常读文件**。和 force_monitor 不一样 ——
    # 那边暂停连采样一起停（它自己是数据源），这边 teleop 还在往文件里写，
    # 停读只会让积压越堆越多，恢复时一次性灌进来反而卡一下。
    state = {"paused": False, "quit": False}

    def on_key(evt):
        if evt.key == " ":
            state["paused"] = not state["paused"]
            print(f"\n  {'暂停（画面冻住，仍在读）' if state['paused'] else '继续'}",
                  flush=True)
        elif evt.key == "q":
            state["quit"] = True

    fig.canvas.mpl_connect("key_press_event", on_key)

    buf, seen, lay = [], 0, None
    period = 1.0 / max(args.fps, 0.5)
    try:
        while plt.fignum_exists(fig.number) and not state["quit"]:
            new, seen = read_rows(path, seen)
            if new is not None and len(new):
                if lay is None:
                    lay = LAYOUT[new.shape[1]]
                    src = "控制器 cart_force / external_torque" if lay["est"] \
                        else "本地 pinv(J^T)·τ 取反（53 列，无关节外力估计）"
                    axes[1].set_title(f"来源：{src}", fontsize=9)
                buf.append(new)
            if state["paused"]:
                # 暂停时也要 flush_events，否则收不到按键、空格再也按不动
                fig.canvas.flush_events()
                time.sleep(period)
                continue
            if buf:
                a = np.vstack(buf)
                t = a[:, 0] - a[0, 0]
                j7, F, M, ext, z, eng = channels(a, lay)
                hi = t[-1]
                lo = max(0.0, hi - args.window)
                m = t >= lo

                l_j7.set_data(t, j7)
                for k, ln in enumerate(l_F):
                    ln.set_data(t, F[:, k])
                for k, ln in enumerate(l_M):
                    ln.set_data(t, M[:, k])
                if ext is not None:
                    for k, ln in enumerate(l_E):
                        ln.set_data(t, ext[:, k])

                for ax in axes:
                    ax.set_xlim(lo, max(hi, lo + 1.0))
                if m.any():
                    def span(arr, pad, floor):
                        v = arr[m]
                        return (min(-floor, v.min() * pad), max(floor, v.max() * pad))
                    axes[1].set_ylim(*span(F, 1.15, 10))
                    axes[2].set_ylim(*span(M, 1.15, 2))
                    if ext is not None:
                        axes[3].set_ylim(*span(ext, 1.15, 5))

                pct = j7[-1] / RSC_J7_LIMIT * 100
                colour = "crimson" if pct > 90 else ("darkorange" if pct > 75 else "black")
                banner.set_text(f"J7 {j7[-1]:5.2f} Nm  {pct:3.0f}% of 17.0   "
                                f"峰值 {j7.max():5.2f}   z {z[-1]:6.1f} mm   "
                                f"{'接合' if eng[-1] else '松开'}")
                banner.set_color(colour)
                txt = (f"末端力  Fx{F[-1,0]:+7.1f}  Fy{F[-1,1]:+7.1f}  Fz{F[-1,2]:+7.1f} N     "
                       f"末端力矩  Mx{M[-1,0]:+6.2f}  My{M[-1,1]:+6.2f}  Mz{M[-1,2]:+6.2f} Nm")
                if ext is not None:
                    txt += "\n关节外力  " + "  ".join(
                        f"J{k+1}{ext[-1,k]:+6.1f}" for k in range(7)) + " Nm"
                readout.set_text(txt)
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            time.sleep(period)
    except KeyboardInterrupt:
        pass
    print("\n  停了")
    return 0


if __name__ == "__main__":
    sys.exit(main())
