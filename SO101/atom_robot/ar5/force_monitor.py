#!/usr/bin/env python3
"""独立的实时力监视窗口：直接连臂读，不需要 teleop。

    python -m atom_robot.ar5.force_monitor

四路都来自 `forceControl().getEndTorque()` 一次调用：

    joint_torque_measured    各轴测量力矩（原始，含重力）      Nm
    external_torque_measured 各轴外部力（控制器扣掉动力学模型）  Nm
    cart_force               笛卡尔空间力                      N
    cart_torque              笛卡尔空间力矩                    Nm

**只读。** connect(power_on=False)、不起实时流、不发任何运动指令，
`fcInit`/`fcStart` 也不调（实测 `getEndTorque` 不需要它们，而那两个会真的进力控）。

单次调用 0.3 ms —— 前提是**句柄缓存**，驱动的 `get_end_wrench` 已经做了。
`robot.forceControl()` 取一次句柄要 41 ms，每帧重取会把 1 kHz 流打出洞。

### 和 teleop 一起用？

这个脚本会**另开一个 SDK 会话**。遥操时想看曲线，优先用另一条路：

    teleop ... --live /tmp/live.csv        # 终端 A
    live_plot /tmp/live.csv                # 终端 B，只读文件，不碰臂

那条只有一个 SDK 连接，最稳。本脚本是给**不跑遥操时**用的 —— 手动推臂、
装夹检查、负载辨识摆姿态，都靠它看数。

### 归零

启动时自动取基线（默认 30 帧）。`payload_mass` 是 0 而手加工具约 1.2~1.3 kg，
那份重力在四路里**都没被扣掉**，所以绝对值没意义、只有变化量有意义。
**归零之后不要再改姿态** —— 姿态一变，重力在末端系的投影就变了，基线作废。
按 `t` 可以随时重新归零（窗口获得焦点时）。

### 按键

    空格   暂停 / 继续 —— **采样也一起停**，要看的那个峰不会被挤出滚动窗口
    t      重新归零
    q      退出
"""

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from atom_robot.ar5.config import AR5Config  # noqa: E402
from atom_robot.ar5.drivers.rokae_arm import RokaeAR5  # noqa: E402

RSC_J7_LIMIT = 17.0      # 控制器日志 id=30400 报的 J7 传动力矩上限
TARE_FRAMES = 30


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ip", default="", help="覆盖控制器 IP")
    ap.add_argument("--hz", type=float, default=20.0, help="采样率")
    ap.add_argument("--window", type=float, default=20.0, help="横轴显示多少秒")
    ap.add_argument("--frame", default="world", choices=("world", "flange", "tool"),
                    help="末端力的参考系。world 的 Z 才是「竖直」；"
                         "flange 可以用来看末端负载的重力落在哪根轴上（= 哪根轴朝下）")
    ap.add_argument("--no-tare", action="store_true", help="不归零，直接看原始值")
    ap.add_argument("--print", dest="print_only", action="store_true",
                    help="只在终端打数，不开窗口。没有图形界面时用；"
                         "按 Ctrl-C 停")
    ap.add_argument("--retries", type=int, default=15,
                    help="连不上时重试几次。上个进程刚退出时控制器要约 40 秒才放会话")
    args = ap.parse_args()

    if not args.print_only:
        # 只在要开窗时才导入 matplotlib：--print 得能在没有显示的机器上跑
        import matplotlib
        from matplotlib import pyplot as plt
        for cand in ("Noto Sans CJK JP", "WenQuanYi Micro Hei", "Droid Sans Fallback"):
            matplotlib.rcParams["font.sans-serif"] = [cand]
            break
        matplotlib.rcParams["axes.unicode_minus"] = False

    cfg = AR5Config().arm
    if args.ip:
        cfg.ip = args.ip
    cfg.use_realtime = False          # 只读：不起 1 kHz 流
    arm = RokaeAR5(cfg)
    for attempt in range(args.retries):
        try:
            arm.connect(power_on=False)
            break
        except RuntimeError as exc:
            print(f"  连接第 {attempt + 1} 次失败：{exc}", flush=True)
            time.sleep(10)
    else:
        print("  连不上控制器")
        return 1

    n = cfg.num_joints
    print(f"  参考系 {args.frame}   采样 {args.hz} Hz")
    print(f"  payload_mass = {cfg.payload_mass} kg（手+工具实测 1.2~1.3 kg，"
          f"所以四路都带着没扣的重力）")

    def sample():
        raw, ext, ctrq, cforce = arm.get_end_wrench(frame=args.frame)
        return (np.asarray(raw[:n]), np.asarray(ext[:n]),
                np.asarray(cforce[:3]), np.asarray(ctrq[:3]))

    period_print = 1.0 / max(args.hz, 0.5)
    base = [np.zeros(n), np.zeros(n), np.zeros(3), np.zeros(3)]
    if not args.no_tare:
        print(f"  归零：取 {TARE_FRAMES} 帧基线，**保持姿态、不要碰臂**...", flush=True)
        acc = [[], [], [], []]
        for _ in range(TARE_FRAMES):
            for k, v in enumerate(sample()):
                acc[k].append(v)
            time.sleep(0.03)
        base = [np.mean(x, axis=0) for x in acc]
        spread = np.ptp(acc[2], axis=0)
        print(f"    基线末端力 {np.round(base[2], 2).tolist()} N"
              f"   帧间抖动 {np.round(spread, 2).tolist()}")
        print(f"    **噪声地板约 {np.max(spread):.2f} N —— 小于它的读数不算数**")
        print("  归零后别再改姿态，姿态一变基线作废。窗口里按 t 可重新归零。")

    if args.print_only:
        print(f"\n  {'各轴外部力 J1..J7 (Nm)':<46}"
              f"{'笛卡尔力 XYZ (N)':>26}{'笛卡尔力矩 XYZ (Nm)':>26}   J7/17.0")
        peak = 0.0
        try:
            while True:
                cur = sample()
                dext = cur[1] - base[1]
                dF = cur[2] - base[2]
                dM = cur[3] - base[3]
                j7 = abs(float(cur[0][n - 1]))
                peak = max(peak, j7)
                pct = j7 / RSC_J7_LIMIT * 100
                warn = "!!" if pct > 90 else (" !" if pct > 75 else "  ")
                print("  " + "".join(f"{v:+6.1f}" for v in dext)
                      + "   " + "".join(f"{v:+8.1f}" for v in dF)
                      + "   " + "".join(f"{v:+8.2f}" for v in dM)
                      + f"   {j7:5.2f} {pct:3.0f}%{warn}峰{peak:5.2f}",
                      end="\r", flush=True)
                time.sleep(period_print)
        except KeyboardInterrupt:
            print(f"\n\n  停了。J7 峰值 {peak:.2f} Nm（{peak/RSC_J7_LIMIT*100:.0f}% of "
                  f"{RSC_J7_LIMIT}）")
        finally:
            arm.disconnect()
        return 0

    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    fig.canvas.manager.set_window_title(f"AR5 实时力监视（{args.frame} 系）")
    l_tau = [axes[0].plot([], [], lw=.9, label=f"J{j+1}")[0] for j in range(n)]
    l_ext = [axes[1].plot([], [], lw=.9, label=f"J{j+1}")[0] for j in range(n)]
    l_F = [axes[2].plot([], [], lw=1.0, label=x)[0] for x in ("Fx", "Fy", "Fz")]
    l_M = [axes[3].plot([], [], lw=1.0, label=x)[0] for x in ("Mx", "My", "Mz")]
    axes[0].set_ylabel("各轴测量力矩 Δ\n(Nm)")
    axes[1].set_ylabel("各轴外部力 Δ\n(Nm)")
    axes[2].set_ylabel("笛卡尔空间力 Δ\n(N)")
    axes[3].set_ylabel("笛卡尔空间力矩 Δ\n(Nm)")
    axes[3].set_xlabel("时间 (s)　　空格 暂停/继续　t 重新归零　q 退出")
    for ax in axes:
        ax.grid(alpha=.3)
    axes[0].legend(fontsize=7, ncol=n, loc="upper left")
    axes[2].legend(fontsize=8, ncol=3, loc="upper left")
    axes[3].legend(fontsize=8, ncol=3, loc="upper left")
    # 读数放在图外面：放进 axes[0] 会和那一排 J1..J7 的图例叠在一起
    banner = fig.text(0.5, 0.985, "", fontsize=11, ha="center", va="top",
                      fontweight="bold")
    readout = fig.text(0.01, 0.005, "", fontsize=9, va="bottom")
    plt.tight_layout(rect=(0, 0.045, 1, 0.968))

    state = {"tare": False, "quit": False, "paused": False}

    def on_key(evt):
        # 空格 = 冻住画面**并且停止采样**。只冻画面的话，要看的那个峰会被后续
        # 数据挤出滚动窗口 —— 暂停就是为了留住它。
        if evt.key == " ":
            state["paused"] = not state["paused"]
            print(f"\n  {'暂停' if state['paused'] else '继续'}", flush=True)
        elif evt.key == "t":
            state["tare"] = True
        elif evt.key == "q":
            state["quit"] = True

    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show(block=False)

    keep = int(max(args.window, 1.0) * args.hz) + 10
    hist = {k: deque(maxlen=keep) for k in ("t", "tau", "ext", "F", "M")}
    t0 = time.time()
    period = 1.0 / max(args.hz, 0.5)
    pause_text = ""

    try:
        while plt.fignum_exists(fig.number) and not state["quit"]:
            if state["paused"]:
                # 暂停时也要 flush_events，否则窗口收不到按键、空格再也按不动
                banner.set_text(pause_text or banner.get_text())
                banner.set_color("tab:blue")
                fig.canvas.flush_events()
                time.sleep(period)
                continue
            cur = sample()
            if state["tare"]:
                base = list(cur)
                state["tare"] = False
                print("\n  已重新归零")
            hist["t"].append(time.time() - t0)
            for key, v, b in zip(("tau", "ext", "F", "M"), cur, base):
                hist[key].append(v - b)

            t = np.array(hist["t"])
            dat = {k: np.array(hist[k]) for k in ("tau", "ext", "F", "M")}
            for lines, key in ((l_tau, "tau"), (l_ext, "ext"), (l_F, "F"), (l_M, "M")):
                for j, ln in enumerate(lines):
                    ln.set_data(t, dat[key][:, j])
            lo, hi = max(0.0, t[-1] - args.window), max(t[-1], args.window * 0.1)
            for ax, key, floor in ((axes[0], "tau", 2.0), (axes[1], "ext", 2.0),
                                   (axes[2], "F", 5.0), (axes[3], "M", 1.0)):
                ax.set_xlim(lo, max(hi, lo + 1.0))
                v = dat[key]
                ax.set_ylim(min(-floor, v.min() * 1.15), max(floor, v.max() * 1.15))

            j7 = abs(float(np.asarray(cur[0])[n - 1]))
            pct = j7 / RSC_J7_LIMIT * 100
            banner.set_text(f"J7 绝对力矩 {j7:5.2f} Nm　{pct:3.0f}% of {RSC_J7_LIMIT}"
                            f"　（RSC 超限就安全停止、掉电抱闸）")
            banner.set_color("crimson" if pct > 90 else
                             ("darkorange" if pct > 75 else "black"))
            pause_text = (f"⏸ 已暂停（空格继续）　J7 {j7:5.2f} Nm　{pct:3.0f}% of "
                          f"{RSC_J7_LIMIT}　—— 采样也停了，数据不再往前走")
            readout.set_text(
                "各轴外部力 " + " ".join(f"J{j+1}{dat['ext'][-1, j]:+6.1f}" for j in range(n))
                + " Nm\n笛卡尔 力 "
                + " ".join(f"{x}{dat['F'][-1, k]:+7.1f}" for k, x in enumerate("XYZ"))
                + " N　力矩 "
                + " ".join(f"{x}{dat['M'][-1, k]:+6.2f}" for k, x in enumerate("XYZ")) + " Nm")

            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            time.sleep(period)
    except KeyboardInterrupt:
        pass
    finally:
        arm.disconnect()
        print("\n  断开")
    return 0


if __name__ == "__main__":
    sys.exit(main())
