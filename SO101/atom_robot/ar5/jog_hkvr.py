"""Read-only HKVR probe by default; explicit --init / --opening / --follow-leader moves gripper only."""
import argparse
import time

from .config import AR5Config
from .end_effector import add_end_effector_args, configure_end_effector
from .drivers.hkvr_protocol import HKVRGripper, MODE_FORCE_POS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_end_effector_args(parser)
    parser.set_defaults(end_effector="hkvr")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--init", action="store_true", help="指尖初始化，会走全行程")
    action.add_argument("--opening", type=float, help="目标开合 0=闭合、1=张开")
    action.add_argument("--follow-leader", action="store_true", help="SO101 扳机控制夹爪，不连接 AR5")
    parser.add_argument("--leader-port", default=AR5Config().collect.leader_port)
    parser.add_argument("--leader-baud", type=int, default=1_000_000)
    parser.add_argument("--calib-id", default=AR5Config().collect.calib_id)
    args = parser.parse_args()
    if args.end_effector != "hkvr":
        parser.error("这个入口只支持 hkvr")
    if args.opening is not None and not 0 <= args.opening <= 1:
        parser.error("--opening 必须为 0..1")
    config = AR5Config()
    configure_end_effector(config, args)
    cfg = config.hkvr
    if args.follow_leader:
        from .drivers.hkvr_hand import HKVRHand
        from .leader_bus import LeaderBus
        from .joint_mapping import JointMapper, JointMapConfig
        leader = LeaderBus.with_calibration(args.leader_port, args.calib_id,
                                           baudrate=args.leader_baud)
        hand = HKVRHand(cfg)
        mapper = JointMapper(JointMapConfig.load())
        try:
            leader.connect()
            leader.assert_torque_disabled()
            hand.connect()
            print("仅夹爪跟随 SO101 扳机；AR5 未连接。Ctrl+C 退出。")
            while True:
                joints = leader.read_positions()
                if "gripper" not in joints:
                    raise IOError("SO101 扳机读数丢失，停止发送夹爪目标")
                target = mapper.gripper_command(joints)
                hand.set_opening(target)
                print(f"\rtarget={target:.3f} actual={hand.get_opening():.3f}   ", end="", flush=True)
                time.sleep(0.05)
        except KeyboardInterrupt:
            print("\n已退出；不自动松夹。")
        finally:
            try:
                hand.disconnect()
            finally:
                leader.disconnect()
        return 0
    with HKVRGripper(port=cfg.port, baud=cfg.baudrate, slave=cfg.slave_id,
                     timeout=cfg.timeout, speed=cfg.speed, transport=cfg.transport,
                     can_id=cfg.can_id, can_rx_id=cfg.can_rx_id) as gripper:
        print(f"{cfg.transport} 读回目标力 = {gripper.target_force()}；通信已确认。")
        if args.init:
            if not gripper.init_fingers():
                raise IOError("初始化未收到确认；不要自动重发，先检查夹爪状态")
            print("初始化指令已确认，等待夹爪全行程结束后再启动遥操。")
        elif args.opening is not None:
            if not gripper.set_mode(MODE_FORCE_POS) or not gripper.set_force(cfg.force):
                raise IOError("模式/力设置未确认")
            if not gripper.set_opening(args.opening):
                raise IOError("目标位置未确认")
            for _ in range(30):
                info = gripper.info()
                if info is None:
                    raise IOError("夹爪反馈丢失")
                print(f"\rposition={info[0]} status={info[2]}   ", end="", flush=True)
                time.sleep(0.1)
            print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
