"""单独调 OmniPicker，不连 AR5、不连主臂。

    cd ~/yuhang_workspace/算法/DexGraspVLA-/SO101
    PY=~/miniforge3/envs/dexgraspvla/bin/python
    PYTHONPATH= $PY -m omnipicker.jog                       # 只读：固件、状态、读写延迟，不动
    PYTHONPATH= $PY -m omnipicker.jog --opening 0.5         # 走到一半开度
    PYTHONPATH= $PY -m omnipicker.jog --grip --force 32     # 合拢夹住，--hold 秒后自动松开

不给 --opening / --grip 就**一条运动指令都不发**。走的是和采集同一个驱动，
回绕、卡死、错误码的保护都在。会动的命令先确认指间是空的（或者是你要夹的东西）。
"""
import argparse
import statistics
import sys
import time

from .config import OmniPickerConfig
from .gripper import OmniPickerGripper
from .protocol import FAULTS, STATE_BLOCKED, OmniPickerBus


def _line(status, t0):
    return (f"  {time.monotonic() - t0:5.2f}s  位置 {status['pos']:>3}  速度 {status['vel']:>3}  "
            f"力矩 {status['force']:>3}  {status['state_name']}"
            + (f"  错误 {FAULTS.get(status['fault'])}" if status["fault"] else ""))


def _watch(gripper, seconds, until=None):
    t0 = time.monotonic()
    last = -1.0
    while (t := time.monotonic() - t0) < seconds:
        gripper.check_health()
        status = gripper.last_status()
        if t - last >= 0.25:
            print(_line(status, t0))
            last = t
        if until is not None and t > 0.4 and until(status):
            print(_line(status, t0))
            return status
        time.sleep(0.02)
    return gripper.last_status()


def readonly(cfg, count):
    bus = OmniPickerBus(cfg.port, cfg.addr, cfg.baudrate, cfg.timeout)
    try:
        print(f"  固件 {bus.firmware()}")
        lat = []
        for _ in range(count):
            t = time.perf_counter()
            status = bus.status()
            lat.append((time.perf_counter() - t) * 1000)
        lat.sort()
        print(f"  状态 {status}")
        print(f"  读 {count} 次：中位 {statistics.median(lat):.1f} ms，最大 {lat[-1]:.1f} ms")
    finally:
        bus.close()


def main():
    cfg = OmniPickerConfig()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default=cfg.port)
    ap.add_argument("--addr", type=int, default=cfg.addr)
    ap.add_argument("--baud", type=int, default=cfg.baudrate)
    ap.add_argument("--opening", type=float, default=None, help="走到这个开度（0 合拢，1 张开到 open_max）")
    ap.add_argument("--grip", action="store_true", help="合拢夹住，--hold 秒后自动张开")
    ap.add_argument("--hold", type=float, default=3.0, help="--grip 夹持多久，最多 10 秒")
    ap.add_argument("--change-force", type=int, default=None,
                    help="--grip 夹住后、夹持到一半时把夹持力改成这个值，看夹着时改力生不生效")
    ap.add_argument("--force", type=int, default=cfg.grip_force,
                    help=f"夹持力 {cfg.grip_force_min}~{cfg.grip_force_max}（寄存器值，不是牛顿）")
    ap.add_argument("--speed", type=int, default=cfg.speed,
                    help=f"{cfg.speed_min}~{cfg.speed_max}，默认 %(default)s")
    ap.add_argument("--count", type=int, default=50, help="只读模式读几次")
    args = ap.parse_args()
    if not 0 < args.hold <= 10:
        ap.error("--hold 要在 (0, 10]")

    cfg = OmniPickerConfig(port=args.port, addr=args.addr, baudrate=args.baud,
                           speed=args.speed, grip_force=args.force)
    if args.opening is None and not args.grip:
        readonly(cfg, args.count)
        return 0

    gripper = OmniPickerGripper(cfg)
    gripper.connect()
    try:
        print(f"  固件 {gripper.firmware}  夹持力 {gripper.grip_force}  速度 {cfg.speed}  "
              f"张开上限 {cfg.open_limit}（开度换算基准 {cfg.open_max}）")
        if args.opening is not None:
            target = gripper.target_position(args.opening)
            print(f"== 走到开度 {args.opening:.2f}（位置 {target}）")
            gripper.set_opening(args.opening)
            _watch(gripper, 4.0, until=lambda s: s["vel"] == 0 and abs(s["pos"] - target) <= 2)
        if args.grip:
            print(f"== 合拢夹取，夹持力 {gripper.grip_force}，{args.hold:.1f}s 后自动松开")
            gripper.set_opening(0.0)
            status = _watch(gripper, 4.0, until=lambda s: s["state"] == STATE_BLOCKED or
                            (s["vel"] == 0 and s["pos"] <= 3))
            if status["state"] == STATE_BLOCKED:
                print(f"  夹住了：停在位置 {status['pos']}")
            if args.change_force is None:
                _watch(gripper, args.hold)
            else:
                _watch(gripper, args.hold / 2)
                now_force = gripper.set_grip_force(args.change_force)
                print(f"== 夹着改夹持力 → {now_force}")
                _watch(gripper, args.hold / 2)
            print("== 松开")
            gripper.set_opening(1.0)
            _watch(gripper, 4.0, until=lambda s: s["vel"] == 0 and abs(s["pos"] - cfg.open_limit) <= 2)
    finally:
        gripper.disconnect()
    return 0


if __name__ == "__main__":
    sys.exit(main())
