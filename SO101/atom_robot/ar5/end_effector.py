"""Shared CLI selection for SO101 teleoperation and collection."""
from .config import HKVRConfig


def add_end_effector_args(parser):
    # 默认 None，落到 `AR5Config.end_effector`（现在是 hkvr）。写死默认值的话，
    # 改 config.py 会显得没生效 —— 而 config 才是「现场装的是什么」的唯一说法。
    parser.add_argument("--end-effector", choices=("inspire", "hkvr", "omnipicker"), default=None)
    # 默认取 HKVRConfig 里那条 by-id 路径，别写死 ttyUSBn ——
    # 本机同时插着 CH340（USB-CAN）和 Prolific（USB 转 RS485），
    # **编号按插入顺序给，重插一次就互换**，接错口的表现是完全没有回包。
    parser.add_argument("--gripper-port", default=HKVRConfig.port)
    parser.add_argument("--gripper-transport", choices=("rs485", "can"), default="rs485",
                        help="can 仅支持仓库中武汉若比特串口透传封装；其他适配器需另接驱动")
    parser.add_argument("--gripper-baud", type=int, default=None,
                        help="RS485 默认 1M；CAN 适配器串口默认 115200，不是 CAN 总线位速率")
    parser.add_argument("--gripper-can-id", type=lambda s: int(s, 0), default=1,
                        help="主机发送 CAN ID（夹爪接收 ID）")
    parser.add_argument("--gripper-can-rx-id", type=lambda s: int(s, 0), default=1,
                        help="主机接收 CAN ID（夹爪发送 ID），按设备配置设置")
    parser.add_argument("--gripper-id", type=int, default=1)
    parser.add_argument("--gripper-speed", type=int, default=1000)
    parser.add_argument("--gripper-force", type=int, default=20,
                        help="HKVR 目标力寄存器值 1..100，非牛顿")
    # OmniPicker 单独一组：力的单位、范围都和 HKVR 不一样（0~255 寄存器值），
    # 共用 --gripper-force 的话默认的 20 会被当成 OmniPicker 的力，直接出界。
    # 默认 None = 用 SO101/omnipicker/config.py 里的值，那里才是持久的地方。
    parser.add_argument("--omnipicker-port", default=None,
                        help="默认 omnipicker/config.py 的 by-id 路径（hkvtech USB-485）")
    parser.add_argument("--omnipicker-force", type=int, default=None,
                        help="OmniPicker 初始夹持力，寄存器值，非牛顿；采集时 ↑↓ 还能调")
    parser.add_argument("--omnipicker-speed", type=int, default=None,
                        help="OmniPicker 初始开合速度 32..255；采集时准备阶段 ←→ 还能调")


def configure_end_effector(config, args):
    # 没给就保持 config 里的值，别拿 None 去顶掉它
    if getattr(args, "end_effector", None):
        config.end_effector = args.end_effector
    config.hkvr = HKVRConfig(port=args.gripper_port, baudrate=args.gripper_baud,
                            transport=args.gripper_transport, can_id=args.gripper_can_id,
                            can_rx_id=args.gripper_can_rx_id,
                            slave_id=args.gripper_id, speed=args.gripper_speed,
                            force=args.gripper_force)
    if config.end_effector == "omnipicker":
        # 只在选了它的时候才 import：omnipicker 包在 SO101 目录下，从别处跑
        # （VEL、RobotDecision）import atom_robot 时不该因为它找不到而起不来。
        from omnipicker import OmniPickerConfig     # noqa: PLC0415
        overrides = {k: v for k, v in (("port", getattr(args, "omnipicker_port", None)),
                                       ("grip_force", getattr(args, "omnipicker_force", None)),
                                       ("speed", getattr(args, "omnipicker_speed", None)))
                     if v is not None}
        config.omnipicker = OmniPickerConfig(**overrides)
        if config.use_hand:
            op = config.omnipicker
            print(f"  OmniPicker 485 {op.port} @{op.baudrate} addr={op.addr} "
                  f"夹持力={op.grip_force}（{op.grip_force_min}~{op.grip_force_max}，↑↓ 调） "
                  f"速度={op.speed}（{op.speed_min}~{op.speed_max}，←→ 调） 张开上限={op.open_max}")
    if config.use_hand and config.end_effector == "hkvr":
        print(f"  HKVR-TG9801 {config.hkvr.transport} {config.hkvr.port} "
              f"@{config.hkvr.baudrate} id={config.hkvr.slave_id} "
              f"speed={config.hkvr.speed} force={config.hkvr.force}")
        if config.hkvr.transport == "can":
            print(f"  CAN tx=0x{config.hkvr.can_id:x} rx=0x{config.hkvr.can_rx_id:x}；"
                  "UART 波特率不设置 CAN 总线速率；需要兼容的夹爪侧 CAN 接口/转换盒。")


def make_real_hand(config):
    if config.end_effector == "omnipicker":
        from omnipicker import OmniPickerConfig, OmniPickerGripper     # noqa: PLC0415
        return OmniPickerGripper(config.omnipicker or OmniPickerConfig())
    if config.end_effector == "hkvr":
        from .drivers.hkvr_hand import HKVRHand
        return HKVRHand(config.hkvr)
    if config.end_effector != "inspire":
        raise ValueError(f"未知末端设备 {config.end_effector}")
    if config.hand.protocol == "can":
        from .drivers.inspire_hand_can import InspireHandCan
        return InspireHandCan(config.hand)
    from .drivers.inspire_hand import InspireHand
    return InspireHand(config.hand)
