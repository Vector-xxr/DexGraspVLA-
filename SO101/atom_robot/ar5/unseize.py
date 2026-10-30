#!/usr/bin/env python3
"""臂抱死了自己把它救回来：认原因 → 上电 → 清残留 → 回 0 位。

    python -m atom_robot.ar5.unseize                # 只诊断，什么都不动
    python -m atom_robot.ar5.unseize --recover      # 上电 + 清残留，**不动臂**
    python -m atom_robot.ar5.unseize --recover --home   # 再回 0 位（会问你一次）
    python -m atom_robot.ar5.unseize --watch         # 守着，发现抱死就自动救

⚠ **臂抱死是有原因的。** 碰撞停机意味着它撞到了东西，自动上电回位就是朝那个
东西再走一次；急停和安全门是人按下的，更不该被软件绕过。所以：

* **诊断永远做**，不需要任何参数
* **上电**要 `--recover`
* **移动**要再加 `--home`，而且会停下来问一次（`--yes` 跳过）
* `--watch` 也一样 —— 不给 `--home` 它就只上电、不动臂

也就是说默认路径下这个脚本不会让臂动一毫米。

怎么判断「抱死」
---------------
抱死不是一种状态，是四种，救法完全不同：

    1. 电源 off          —— 碰撞硬停掉的电。软件能上回来
    2. 急停 / 安全门     —— **物理的，软件无能为力**。去把按钮旋起来
    3. rtControlling 残留 —— 上次实时会话没退干净。不清掉的话每个排队指令都被
                            拒，报 `ec=262 运动控制模式错误`，而且**熬得过控制器重启**
    4. sender latch      —— 回路断粮 0.25 秒，驱动自己刹的车。清一下就行，不用上电

**`powerState()` 不可靠**：实测急停按下时它报的是 `off` 而不是 `estop`，所以
第 2 种只能靠上电的返回码认（`ec=-3`）。这个坑在 `power_on.py` 里写着，
这里不重复判断，直接把上电那步交给它。

碰撞停机之后控制器要 **约 40 秒**才重新接受连接，`--watch` 会等。
"""

from __future__ import annotations

import argparse
import gc
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from atom_robot.ar5.config import AR5Config  # noqa: E402

PKG = "atom_robot.ar5"


def _sdk_and_robot(cfg):
    """连上控制器。连不上就抛 —— 调用方决定要不要重试。

    **必须走 `drivers.rokae_arm.load_sdk`**，不能自己 `sys.path` 里塞
    `Release/linux` 再裸 `import xCoreSDK_python`。那是 `power_on.py` 的写法，
    在它自己的进程里没问题；但 `load_sdk` 用的是带包名的
    `from Release.linux import xCoreSDK_python`，两种写法会把**同一个 .so 以两个
    模块名加载两遍**。同进程里先诊断再 `AR5Env.connect()` 就会踩到：
    先是 `ImportError: xCoreSDK-Python is not importable`，
    紧接着 `free(): invalid pointer` 核心转储（2026-09-23 实测）。
    """
    from atom_robot.ar5.drivers.rokae_arm import load_sdk  # noqa: PLC0415

    sdk = load_sdk(cfg.sdk_path)
    robot = getattr(sdk, cfg.robot_class)(cfg.ip, cfg.local_ip)
    return sdk, robot


def _name(value) -> str:
    return str(value).rsplit(".", 1)[-1]


def diagnose(cfg, quiet: bool = False) -> dict:
    """读一遍控制器，回报「现在是哪一种抱死」。只读，不动任何东西。"""
    out = {"reachable": False, "power": None, "operation": None,
           "rt_left_over": False, "physical_stop": False, "seized": False,
           "idle_off": False, "why": ""}
    try:
        sdk, robot = _sdk_and_robot(cfg)
    except Exception as exc:                                # noqa: BLE001
        out["why"] = f"连不上控制器：{type(exc).__name__}: {exc}"
        return out

    ec = {}
    try:
        power = robot.powerState(ec)
        out["reachable"] = True
        out["power"] = _name(power)
    except Exception as exc:                                # noqa: BLE001
        out["why"] = f"读电源状态失败：{exc}"
        return out

    try:
        op = robot.operationState(ec)
        out["operation"] = _name(op)
        # rtControlling 残留：上次实时会话没退干净。这个**熬得过控制器重启**，
        # 不清掉的话排队式 Move* 全被拒，错误码指向「运动控制模式」而不是真因。
        out["rt_left_over"] = "rtControl" in out["operation"]
    except Exception:                                       # noqa: BLE001
        pass

    # **掉电不等于抱死。** 2026-09-23 实测：collect 正常按 q 退出之后就是
    # `off + idle` —— 线程退出会掉电，这是设计如此。把它判成抱死的话，
    # `--watch` 会对着一个没坏的臂反复"抢救"。
    #
    # 两者的分界是 operationState：碰撞硬停**同时**掉电并把控制器留在
    # rtControlling（实时会话被半路打死），而正常退出会干净地回到 idle。
    if out["power"] in ("estop", "gstop"):
        out["physical_stop"] = True
        out["seized"] = True
        where = "急停按钮" if out["power"] == "estop" else "安全门 / 护栏回路"
        out["why"] = f"{out['power']} —— {where}是物理的，软件清不掉"
    elif out["power"] == "off" and out["rt_left_over"]:
        out["seized"] = True
        out["why"] = (f"掉电 + operationState={out['operation']} 残留 —— "
                      "碰撞硬停的典型样子（掉电且实时会话被半路打死）")
    elif out["rt_left_over"]:
        out["seized"] = True
        out["why"] = f"上电着，但 operationState={out['operation']} 残留，排队指令会被拒"
    elif out["power"] == "off":
        out["idle_off"] = True
        # 措辞在 2026-09-23 改过一次。原来这里写「**没抱死**，只是没上电」——
        # 那是控制器内部的区分，和操作员手里的体感是反的：**掉电抱闸就咬死**，
        # 臂推不动，看起来跟碰撞停机一模一样。跟人说"没抱死"只会让人以为
        # 工具查错了。真正有用的区分不是"算不算抱死"，而是"软件能不能救"。
        out["why"] = ("掉电，抱闸咬着 —— 臂是硬的，推不动。**软件能救**："
                      "上电就松闸（会话正常退出、空闲超时都会到这个状态）")
    else:
        out["why"] = f"电源 {out['power']}、operationState {out['operation']}，看起来正常"

    # 用完立刻放掉。**同进程里后面还要建 AR5Env 的那个 robot**，
    # 两个实例同时活着打同一台控制器不安全。
    del robot
    gc.collect()

    if not quiet:
        print(f"  控制器 {cfg.ip}")
        print(f"    电源            {out['power']}")
        print(f"    operationState  {out['operation']}")
        print(f"    判断            {out['why']}")
    return out


def power_back_on(keep_rt: bool = False) -> int:
    """上电。**转交给 `power_on.py`**，不在这儿重写。

    那个脚本已经处理了这条路上所有踩过的坑：急停恢复 `recoverState(1)`、
    清 `rtControlling`、以及「ec=-3 才是急停，powerState 说了不算」。
    复制一份出来只会让两处慢慢长歪。
    """
    # `--yes`：power_on 在非交互终端下不给这个就拒绝上电（EOF 不算确认）。
    # 这里给，是因为调用方已经各自把门设好了 —— unseize 要 --recover，
    # collect 则是操作员亲手启动的，人就在终端前。
    cmd = [sys.executable, "-m", f"{PKG}.power_on", "--enable", "--yes"]
    if keep_rt:
        cmd.append("--keep-rt")
    print(f"\n  上电：{' '.join(cmd[2:])}")
    sys.stdout.flush()      # 不刷的话子进程的输出会插到这几行前面，读起来因果颠倒
    # 收下来再自己打，**不让子进程直接写 fd 1**。collect 是把 sys.stdout 换成
    # 一个同时落盘的 tee 来留证据的，而子进程继承的是裸 fd、绕过那一层 ——
    # 结果就是上电失败时最关键的那行 `ec=…` 只在终端上闪一下，日志里没有。
    # power_on 一秒就跑完，攒完再打和边跑边打看不出区别。
    done = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, errors="replace")
    if done.stdout:
        print(done.stdout, end="" if done.stdout.endswith("\n") else "\n")
    return done.returncode


def go_home(cfg, timeout_s: float = 30.0) -> bool:
    """回 0 位（映射中心 `centre_joints`），走和采集同一条路。

    用 `AR5Env.goto_joints` 而不是自己发指令：那条路上有跳变门、地板检查、
    以及走完把限速锚点对齐（2026-09-23 修的那个死锁就在这儿）。
    """
    from atom_robot.ar5.ar5_env import AR5Env              # noqa: PLC0415
    from atom_robot.ar5.joint_mapping import JointMapConfig  # noqa: PLC0415

    centre = np.asarray(JointMapConfig.load(**cfg.collect.mapping_overrides).centre_joints)
    print(f"  目标 0 位：{np.round(np.degrees(centre), 1).tolist()} 度")

    # 不要手和相机：救援只关心臂，少连一样少一个失败点。
    cfg.use_hand = False
    cfg.cameras.sources = {}
    env = AR5Env(cfg)
    try:
        env.connect()
        ok = env.goto_joints(centre, timeout_s=timeout_s)
        print("  ✓ 已回 0 位" if ok is not False else
              f"  ⚠ {timeout_s:.0f}s 内没到位，自己看一眼")
        return bool(ok is not False)
    finally:
        env.close()


def confirm(prompt: str) -> bool:
    try:
        return input(prompt).strip().lower() in ("y", "yes", "是")
    except (EOFError, KeyboardInterrupt):
        print()
        return False


def recover_once(cfg, args) -> int:
    # cfg 是整份 AR5Config：diagnose 只要 cfg.arm，go_home 要 cfg.collect 和
    # cfg.use_hand。两处传不同的东西曾经让 --home 直接 AttributeError。
    state = diagnose(cfg.arm)

    if not state["reachable"]:
        print("\n  ✗ 控制器连不上。碰撞停机之后它要 **约 40 秒**才重新接受连接 ——")
        print("    刚停就跑这个脚本会看到这一条，等一会儿再来，或者用 --watch。")
        return 2
    if state["physical_stop"]:
        print("\n  ✗ 物理停机，软件救不了。去把按钮/安全门恢复，再跑一次。")
        return 2
    if not state["seized"] and not state.get("idle_off"):
        print("\n  臂看起来是好的，电也上着，没什么要救的。")
        # **但 --home 还是要做。** 2026-09-24：臂状态正常时这里直接 return，
        # 于是 `--recover --home` 一声不吭地什么都没干 —— 用户要的是"回 0 位"，
        # 而"不算抱死"是工具自己的分类，不该拿它否决一条明确的指令。
        # 这个毛病在同一个文件里犯过两次（上一次是 idle_off 那条），
        # 一律按「给了参数就照做」处理。
        if args.home:
            if not (args.yes or confirm("\n  臂要开始移动，回 0 位。活动范围里没人吗？[y/N] ")):
                print("  没确认，不动。")
                return 0
            return 0 if go_home(cfg, timeout_s=args.timeout) else 1
        return 0

    if state.get("idle_off"):
        print("\n  掉电抱闸 —— 臂是硬的，但**软件能救**：上电就松闸。")
        if not args.recover:
            print("  救：python -m atom_robot.ar5.unseize --recover --home")
            print("      （--recover 上电松闸，--home 再让它回 0 位）")
            return 0
        print("  给了 --recover，上电松闸。")

    if not args.recover:
        print("\n  只诊断，没动电源。要救就加 --recover（会上电，臂会变成活的）。")
        print("  想让它顺便回 0 位再加 --home。**确认臂的活动范围里没人。**")
        return 1

    rc = power_back_on(keep_rt=args.keep_rt)
    if rc != 0:
        print(f"\n  ✗ 上电没成功（power_on 返回 {rc}）。上面它自己打的返回码才是真原因；")
        print("    `ec=-3` 是急停 —— powerState() 在急停时也报 off，分不出来。")
        return rc

    after = diagnose(cfg.arm, quiet=True)
    print(f"  上电后：电源 {after['power']}、operationState {after['operation']}")

    if not args.home:
        print("\n  电上回来了，**臂没有动过**。要回 0 位加 --home。")
        return 0

    if not (args.yes or confirm("\n  臂要开始移动，回 0 位。活动范围里没人吗？[y/N] ")):
        print("  没确认，不动。")
        return 0
    return 0 if go_home(cfg, timeout_s=args.timeout) else 1


def watch(cfg, args) -> int:
    """守着，发现抱死就救。**不给 --home 就只上电、不动臂。**"""
    print(f"  守着 {cfg.arm.ip}，每 {args.interval:.0f}s 看一次。Ctrl-C 退出。")
    print(f"  发现抱死会：诊断 → 上电" + ("→ 回 0 位" if args.home else "（不动臂）"))
    if args.home and not args.yes:
        print("  ⚠ --watch 配 --home 时每次都会问你一次；要真全自动请加 --yes。")
    healthy = None
    while True:
        try:
            state = diagnose(cfg.arm, quiet=True)
        except Exception as exc:                            # noqa: BLE001
            print(f"  {time.strftime('%H:%M:%S')} 查状态出错：{exc}")
            time.sleep(args.interval)
            continue

        now = not state["seized"] and state["reachable"]
        was_healthy = healthy is True
        if now != healthy:                    # 只在状态翻转时打印，别刷屏
            stamp = time.strftime("%H:%M:%S")
            print(f"  {stamp} " + ("正常" if now else f"**抱死** —— {state['why']}"))
            healthy = now
        # **只对「本来好的、突然坏了」动手。** 启动时臂就是坏的，可能只是
        # 这会儿没人用（off+idle 已经不判成抱死了），也可能是上一轮没处理完的
        # 现场 —— 两种都该让人看一眼，而不是脚本自己上电。
        if was_healthy and not now and state["reachable"] and not state["physical_stop"]:
            recover_once(cfg, args)
            healthy = None                    # 下一轮重新判定
        time.sleep(args.interval)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recover", action="store_true",
                        help="真的上电（默认只诊断）。上电后臂是活的，但不会动")
    parser.add_argument("--home", action="store_true",
                        help="上电之后回 0 位。**臂会移动**，默认还会问你一次")
    parser.add_argument("--yes", action="store_true", help="不问，直接动。想清楚再用")
    parser.add_argument("--watch", action="store_true",
                        help="守着，发现抱死就按上面的开关救")
    parser.add_argument("--interval", type=float, default=5.0, metavar="秒",
                        help="--watch 的轮询间隔，默认 5")
    parser.add_argument("--timeout", type=float, default=30.0, metavar="秒",
                        help="回 0 位的超时，默认 30")
    parser.add_argument("--keep-rt", action="store_true",
                        help="不清残留的 rtControlling（默认清，见模块说明第 3 种）")
    args = parser.parse_args()

    if args.home and not args.recover:
        parser.error("--home 要配 --recover：不上电就动不了")

    cfg = AR5Config()
    try:
        return watch(cfg, args) if args.watch else recover_once(cfg, args)
    except KeyboardInterrupt:
        print("\n  退出，没有留下任何未完成的动作。")
        return 130


if __name__ == "__main__":
    sys.exit(main())
