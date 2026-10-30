#!/usr/bin/env python3
"""从 SDK 给臂上电，不用示教器。

    python -m atom_robot.ar5.power_on            # 只诊断，不动电源
    python -m atom_robot.ar5.power_on --enable   # 真的上电
    python -m atom_robot.ar5.power_on --off      # 下电

示教器是网页版、只能在 Windows 上跑，而 xCoreSDK 本身有 `setPowerState`。
`RokaeAR5._power_on()` 一直在调它，但失败时只打一句 warning 就过去了 ——
所以"代码上不了电"这个结论可能只是被那句 warning 骗了。这个脚本把控制器
真实的返回码打出来，成不成、为什么不成，都说清楚。

**软件清不掉物理急停。** `PowerState` 有五种：

    off      没使能 —— 软件通常能开，就是这个脚本要做的事
    on       已经上电了
    estop    急停按钮被按下 —— **必须去把按钮旋起来**，软件无能为力
    gstop    安全门/护栏回路断开 —— 同样是物理的
    unknown  读不到

上电只是使能伺服，不会产生任何运动。但上电之后臂就是活的了，
**动手之前确认没人在臂的活动范围里。**
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from atom_robot.ar5.config import AR5Config  # noqa: E402


def ec_of(ec) -> int:
    """SDK 成功时也会填 ec，只有非零才是失败 —— `if ec:` 会把成功当失败。"""
    if isinstance(ec, dict):
        return int(ec.get("ec", 0))
    return int(ec or 0)


def ec_msg(ec) -> str:
    if isinstance(ec, dict):
        return str(ec.get("message", ""))
    return ""


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--enable", action="store_true", help="真的上电")
    parser.add_argument("--yes", action="store_true",
                        help="不问「范围里没人吗」。**自动流程里必须显式给** —— 非交互终端下不给就拒绝上电")
    parser.add_argument("--off", action="store_true", help="下电")
    parser.add_argument("--mode", choices=["automatic", "manual"], default="automatic",
                        help="上电前把运行模式设成哪个（默认 automatic）")
    parser.add_argument("--no-recover", action="store_true",
                        help="不自动做急停恢复。默认撞上急停会先调 recoverState(1)")
    parser.add_argument("--keep-rt", action="store_true",
                        help="残留的 rtControlling 不清掉。默认会退回 NrtCommandMode")
    args = parser.parse_args()
    if args.enable and args.off:
        raise SystemExit("--enable 和 --off 只能给一个")

    cfg = AR5Config().arm
    sys.path.insert(0, cfg.sdk_path)
    sys.path.insert(0, str(Path(cfg.sdk_path) / "Release" / "linux"))
    import xCoreSDK_python as sdk

    print(f"  连 {cfg.ip}（本地 {cfg.local_ip}）...")
    # 构造函数吃普通 str。PyString 只有 cmdID 需要 —— 传错这里会直接
    # TypeError，倒是不会静默出错。localIP 必须给，RT 的 UDP socket 要绑它。
    robot = getattr(sdk, cfg.robot_class)(cfg.ip, cfg.local_ip)
    # SDK 用一个尾部 dict 报错，成功时也会填 {'ec': 0, 'message': '操作成功完成'}。
    # 所以判断要看 ec['ec'] != 0，不能写 if ec —— 那会把成功当失败。
    ec = {}

    state = robot.powerState(ec)
    mode = robot.operateMode(ec)
    print(f"  当前电源  {state}")
    print(f"  当前模式  {mode}")

    # 上一次会话的实时模式没退干净时，控制器会卡在 rtControlling。症状不是"连不上"，
    # 而是**排队式的 Move* 指令被拒**：`moveAppend failed (ec=262) 运动控制模式错误`。
    # 2026-09-17 实测它**熬得过一次断电重启** —— 是控制器里的状态，不是进程或 socket
    # 的残留，所以"重启一下"解决不了，必须显式退回 NrtCommandMode。
    #
    # 只换控制模式，不发任何运动指令；已经是 idle 就什么都不做。
    op_state = robot.operationState(ec) if hasattr(robot, "operationState") else None
    if op_state is not None and str(op_state).endswith("rtControlling"):
        print(f"\n  ⚠ operationState = {op_state} —— 上次会话的实时模式没退干净。")
        print(f"    不清掉的话排队式运动指令会被拒（ec=262 运动控制模式错误）。")
        if args.keep_rt:
            print(f"    --keep-rt 指定了，不动它。")
        else:
            rt = {}
            robot.setMotionControlMode(sdk.MotionControlMode.NrtCommandMode, rt)
            print(f"    setMotionControlMode(NrtCommandMode) -> "
                  f"ec={ec_of(rt)} {ec_msg(rt)}")
            print(f"    现在 operationState = {robot.operationState(ec)}")

    # ⚠️ powerState() 不可靠：实测急停按下时它报的是 `off`，不是 `estop`。
    # 所以下面这个检查只能挡住它**确实**报 estop/gstop 的情况，真正的判据是
    # 上电时 setOperateMode 的返回码。
    name = str(state).rsplit(".", 1)[-1]
    if name in ("estop", "gstop"):
        where = "急停按钮" if name == "estop" else "安全门 / 护栏回路"
        print(f"\n  ✗ 电源状态是 {name} —— 这是**物理**的，软件清不掉。")
        print(f"    去把{where}恢复，再跑这个脚本。")
        return 2
    if name == "unknown":
        print("\n  ✗ 读不到电源状态。控制器连上了但没回应，检查网络和控制器本身。")
        return 2

    if not (args.enable or args.off):
        print(f"\n  只诊断，没动电源。")
        if name == "off":
            print(f"  状态是 off。**但这不保证能上电** —— 急停按下时 powerState()")
            print(f"  照样报 off，只有真去 setOperateMode 才看得出来。")
            print(f"  确认臂的活动范围里没人，然后：")
            print(f"    python -m atom_robot.ar5.power_on --enable")
        else:
            print(f"  已经是 {name}，不用上电。")
        return 0

    if args.off:
        print(f"\n  下电 ...")
        robot.setPowerState(False, ec)
        code = ec_of(ec)
        print(f"    setPowerState(False) -> ec={code} {ec_msg(ec)}")
        print(f"  现在 {robot.powerState(ec)}")
        return 0 if code == 0 else 1

    # --- 上电 ---
    print(f"\n  ⚠ 就要给臂上电了。上电只使能伺服、不产生运动，但臂会变成活的。")
    if args.yes:
        print("  给了 --yes，不再确认。")
    elif not sys.stdin.isatty():
        # **EOF 不算确认。** 原来这里 `readline()` 在 EOF 返回 ""，而 "" 又落在
        # 「回车 = 确认」那一档里 —— 于是被 cron / 网页代理 / `< /dev/null` 调起来时，
        # 「没人回答」被读成了「是，范围里没人」。上电本身不产生运动，但这个默认
        # 方向是反的：没人应答时该拒绝，不是放行。
        print()
        print("  ✗ 不是交互终端，没法确认「臂的活动范围里没人」。")
        print("    自动流程里要上电请显式给 --yes —— 那是一次明确的声明，")
        print("    而不是让 EOF 替你回答。")
        return 3
    else:
        print(f"  臂的活动范围里没人的话按回车，其它键取消：", end="", flush=True)
        answer = sys.stdin.readline()
        if answer.strip() not in ("", "y", "Y"):
            print("  取消。")
            return 0

    print(f"\n  设运行模式 {args.mode} ...")
    robot.setOperateMode(getattr(sdk.OperateMode, args.mode), ec)
    code = ec_of(ec)
    print(f"    setOperateMode -> ec={code} {ec_msg(ec)}")

    # ec=-3 是急停。`recoverState(1)` 就是官方的急停恢复 —— SDK 文档写着
    # 「item (int): 恢复选项，1 表示急停恢复」。
    #
    # 这一条差点被漏掉：第一次试 recoverState 时只传了 ec、没传 item，抛了
    # 签名错误，我据此判断"这个接口不是这个用途"—— 错的。清安全回路的接口
    # 一直都在，是调用方式不对。
    #
    # 注意它只能恢复**可恢复的**急停源。真有一个物理按钮按着，回路是断的，
    # 这个调用会失败 —— 那种情况只能去现场。
    if code == -3 and not args.no_recover:
        print(f"\n  急停状态，试 recoverState(1)（官方的急停恢复）...")
        rec = {}
        robot.recoverState(1, rec)
        print(f"    recoverState(1) -> ec={ec_of(rec)} {ec_msg(rec)}")
        robot.setOperateMode(getattr(sdk.OperateMode, args.mode), ec)
        code = ec_of(ec)
        print(f"    重试 setOperateMode -> ec={code} {ec_msg(ec)}")
    if code != 0:
        # ec=-3 是急停。**不能靠 powerState() 判断急停** —— 实测按钮按下时它
        # 照样报 `off`，只有写路径的返回码说真话。所以急停要在这里认，而不是
        # 在前面那个状态检查里。
        if code == -3 or "急停" in ec_msg(ec):
            print(f"\n  ✗ 急停恢复失败（recoverState(1) 也没清掉）。")
            print(f"    说明安全回路是真的断着，软件恢复不了：")
            print(f"      - 有物理急停按钮按下 -> 顺时针旋转弹出来")
            print(f"      - 控制柜的安全回路输入端子开路 -> 没接急停时要短接")
            print(f"      - 示教器软件里持着一个软急停 -> 去那边清")
            return 2
        print(f"  ✗ 设模式失败 ec={code}。示教器可能还占着控制权（手动模式下 SDK 改不了）。")
        return 1

    print(f"  上电 ...")
    robot.setPowerState(True, ec)
    code = ec_of(ec)
    print(f"    setPowerState(True) -> ec={code} {ec_msg(ec)}")

    state = robot.powerState(ec)
    print(f"\n  现在电源  {state}")
    print(f"  现在模式  {robot.operateMode(ec)}")

    joints = robot.jointPos(ec)
    n = len(joints) if joints else 0
    if n:
        import numpy as np
        q = np.asarray(joints[:cfg.num_joints], dtype=float)
        print(f"  关节角    {np.rad2deg(q).round(2).tolist()} deg")
        print(f"\n  ✓ 上电成功，能读到关节角了。")
        return 0

    print(f"  关节角    读不到（返回 {n} 个值）")
    if code != 0:
        print(f"\n  ✗ 上电失败，ec={code}。常见原因：")
        print(f"    - 示教器还连着并且在手动模式，控制权不在 SDK 这边")
        print(f"    - 控制器有未清除的故障（跑 probe_rokae 看有没有报错）")
        print(f"    - 这台控制器配置成必须从示教器使能")
    return 1 if code != 0 else 0


if __name__ == "__main__":
    sys.exit(main())
