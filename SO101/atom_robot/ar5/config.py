"""Configuration for the AR5 + Inspire-hand real-robot setup.

Everything hardware-specific is collected here so the environment and the eval
client stay free of magic numbers.  Values marked VERIFY must be checked against
your controller and your hand's manual before the first powered run.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from atom_robot import SDK_ROOT  # 不写死本机路径，见 atom_robot/__init__.py


@dataclass
class ArmConfig:
    """ROKAE AR5 connection and motion limits.

    The AR series is a 7-DoF humanoid force-control arm (AR5: 5 kg payload,
    683 mm reach, joint torque sensors standard, xCore controller). Seven axes
    means the arm is kinematically redundant, so inverse kinematics takes an
    extra arm-angle parameter -- see `RokaeAR5.solve_ik`.
    """

    # --- connection ---
    ip: str = "192.168.2.160"           # Confirmed on the AR5-5_0.8L-W4C5C11
    # Local NIC address on the robot subnet. Required for real-time control:
    # the RT channel is UDP and the socket has to be bound to a local
    # address, otherwise it fails with "1338 UDP socket打开失败".
    local_ip: str = "192.168.2.222"
    sdk_path: str = str(SDK_ROOT)

    # SDK robot class. The xCoreSDK exposes one class per robot family:
    #   StandardRobot     -- industrial 6-axis
    #   xMateRobot        -- collaborative 6-axis
    #   xMateErProRobot   -- collaborative 7-axis   <- AR5 is 7-DoF
    #   PCB3Robot / PCB4Robot -- 3/4-axis PCB robots
    # A newer SDK build may ship a dedicated AR class. Run
    # `python -m atom_robot.ar5.probe_rokae --list` to see what your
    # installed SDK actually exports, then set this. Empty = auto-detect.
    robot_class: str = "xMateErProRobot"

    num_joints: int = 7                 # AR series is 7-DoF; keep in sync with PROPRIO_DIM

    # --- control ---
    control_hz: float = 20.0            # Command rate of the outer loop
    use_realtime: bool = True           # Prefer the SDK's real-time streaming interface
    rt_network_tolerance: int = 20      # Packet-loss tolerance for RtCommandMode
    # VERIFY: whether your controller has the real-time option licensed and is
    # on xCore >= v3.2.1. When unavailable the driver falls back to queued
    # blocking Cartesian moves: slower and coarser, but works on any controller.

    # --- per-step safety limits (applied before anything reaches the robot) ---
    max_step_translation: float = 0.02  # meters per control step
    max_step_rotation: float = 0.10     # radians per control step
    move_speed: float = 200.0           # Speed argument for queued Move*Commands
    move_zone: float = 10.0             # Blending zone radius for queued Move*Commands

    # --- Cartesian workspace box, base frame, meters ---
    # Sized against the measured reachable set rather than guessed. Sweeping the
    # local kinematics out of `home_joints` at fixed orientation reaches
    # x [0.188, 0.618], y [-0.317, 0.353], z [-0.305, 0.460]; this box sits
    # inside that with a margin, keeping the z floor high enough to clear a table
    # regardless of what the arm can physically do.
    #
    # The box is deliberately *centred on the home posture*. The previous one was
    # not: the tool sat at x = 0.483 against a ceiling of 0.55, leaving 67 mm of
    # travel forward against 333 mm back. Pushing the leader forward drove the
    # target onto that wall within seven steps, where it stopped advancing while
    # the operator kept moving -- which reads exactly like the arm stuttering.
    #
    # 2026-09-07：y 下界从 -0.28 放宽到 -0.516。原因是采集用的是**关节路径**
    # （不看这个盒子），而策略评测走**笛卡尔路径**（看）。100 条示教数据实测
    # 落在 y [-0.466, -0.168]，用原来的盒子会夹住 46.8% 的帧 —— 臂到不了策略
    # 要去的地方，评测结果没有意义，而且看起来像"模型不行"。
    #
    # 只动了越界的那一侧：x 数据在 [0.255, 0.464]、z 在 [0.095, 0.259]，
    # 本来就在盒内，边界一个不碰。**尤其 z 下界保持 0.08** —— 那是地板保护，
    # 没有依据不放宽。
    #
    # 2026-09-21 按用户要求放大上界。重新用本地运动学量了一遍：从 home
    # (0.400, 0.001, 0.250) 出发、保持 home 姿态、每轴 5mm 步进往外推，
    # 单轴能到的极限是 x [0.265, 0.645]、y [-0.369, 0.396]、z [-0.305, 0.470]；
    # 不限姿态的纯运动学包络是半径约 0.70 m、z [-0.345, 0.874]。
    #
    # 也就是说**上界一直卡在盒子上，不是卡在臂上**：x 早 65mm、y 早 116mm、
    # z 早 50mm 就被夹住了。三个上界各自推到单轴极限往里 15~20mm：
    #
    #     x_max 0.58 -> 0.63    y_max 0.28 -> 0.38    z_max 0.42 -> 0.46
    #
    # 下界一个不动：x_min 0.22 和 y_min -0.516 本来就在臂够不到的地方（够不到
    # 的目标由逆解拒掉，不是危险），z_min 0.08 是台面保护。
    #
    # 盒子比可达集大是正常的 —— 可达集接近球壳，方盒总会外溢。放大后网格可达率
    # 从 65% 降到 53%，溢出的那部分由逆解拒绝，不会动。
    #
    # 2026-09-23 再次按用户要求放大。用控制器标定过的 DH（ar5_dh.json）重量了
    # 一遍，两个判据：
    #
    #   保持 home 姿态单轴外推   x [0.235, 0.630]  y [-0.370, 0.415]  z [-0.325, 0.475]
    #   不限姿态的纯运动学包络   x,y ±0.70         z [-0.346, 0.875]   半径 ≤ 0.875
    #
    # 上一版三个上界离第一个判据分别差 0mm / 35mm / 15mm —— **x 已经顶死在
    # 可达极限上**。这一版直接越过「保持 home 姿态」那条线，按第二个判据取：
    #
    #     x_max 0.63 -> 0.66    y_max 0.38 -> 0.43    z_max 0.46 -> 0.50
    #     x_min 0.22 -> 0.20
    #
    # 越过去是安全的：盒子只负责「不许去」，够不够得到由逆解说了算，
    # 够不到的目标被拒掉而不是硬闯。盒子比可达集大是正常的 —— 可达集接近球壳，
    # 方盒总会外溢。
    #
    # **两个下界一个不动**：y_min -0.516 本来就远超单轴可达（-0.370），
    # z_min 0.08 是台面保护，没有实测依据不放宽。
    #
    # ⚠ **采集根本不看这个盒子。** 采集走关节路径（`step_joints`），盒子约束的是
    # 位姿，关节指令没有位姿可约束 —— 那条路上管事的是 `joint_floor_z`。
    # 这个盒子约束的是策略评测走的笛卡尔路径。
    #
    # 2026-09-23 第二轮，用户要 z 再高一点。单独量了 z：
    #
    #   保持 home 姿态（工具竖直向下）  整个盒子里 z 最高 0.440
    #   不限姿态、x/y 落在盒内          z 最高 0.838（全空间 0.874）
    #
    # 也就是说 **0.44 以上全靠把工具斜过来**，竖直向下的姿态到 0.44 就顶天了。
    # 上一版的 0.50 其实已经越过竖直可达上限。这一版取 **0.65** —— 在不限姿态
    # 的 0.838 里留了 190mm 余量，够不到的姿态由逆解拒掉。
    #
    # **z_min 0.08 不动。** 那是台面保护：台面约在 z = -0.08，而 `_lowest_point`
    # 只算到 DH 链末端（法兰 + 90mm 工具行），**夹爪的指尖在那底下，这道闸看不见**。
    # 要往下放得先量「夹爪加转接板从法兰面到指尖」有多长 —— 这个数现在没有。
    # 注意 `joint_floor_z = 0.10` 也是按**因时手**算的，换夹爪之后没重算过。
    workspace_min: Tuple[float, float, float] = (0.20, -0.516, 0.08)
    workspace_max: Tuple[float, float, float] = (0.66, 0.43, 0.65)

    # Floor for the joint-space path, metres in the base frame. The workspace
    # box does not apply there -- it constrains poses, and joint commands have
    # none -- so the tool and wrist are checked against this instead.
    #
    # The bench sits near z = -0.08: measured 2026-09-01 with the tool at
    # z = 0.080 and roughly 0.16 m of air beneath it.
    #
    # 2026-09-03, the hand is now bolted to the arm, so this had to go back up.
    # `_lowest_point` stops at the DH chain's tip -- flange plus a 90 mm tool row
    # -- and the fingers hang below that, invisible to the guard. The value has
    # to cover them instead:
    #
    #     floor = bench + hand_length + clearance
    #           = -0.08 + hand_length + 0.03
    #
    # Set to 0.10 on request, to give 50 mm more reach than the first pass. That
    # is the right number if the hand is 0.15 m from flange face to fingertip;
    # it was first set to 0.15 on a guess of 0.20 m, which has never been
    # measured. If the hand is actually longer than 0.15 m the fingers reach the
    # bench before the guard fires, so the first descent to this floor wants
    # watching, with a note of how much air is left under the fingertips.
    #
    #     fingertip height at the floor = 0.10 - hand_length
    #     bench is at -0.08, so anything under 0.18 m of hand has clearance
    #
    # Measuring the hand and filling in `tcp_offset` / `payload_mass` replaces
    # this whole compensation with the guard actually seeing the fingers.
    joint_floor_z: float = 0.10

    # --- joint soft limits, radians, read from the controller ---
    # getSoftLimit() reports these and the controller enforces them, but a
    # target outside them is refused rather than clamped, so the same numbers
    # are needed here to reject a bad goal before it is ever commanded.
    joint_min: Tuple[float, ...] = (-3.1067, -2.0944, -3.1067, -1.0472, -3.1067, -0.8727, -0.8727)
    joint_max: Tuple[float, ...] = (3.1067, 2.0944, 3.1067, 2.5307, 3.1067, 0.8727, 0.8727)

    # Stay this far inside the soft limits when commanding, so a joint never
    # arrives exactly on the boundary the controller is checking.
    joint_limit_margin: float = 0.05    # radians

    # --- home posture, `num_joints` angles in radians ---
    # Chosen by search over the local kinematics rather than by eye. It puts the
    # tool at the exact centre of the workspace box, (0.40, 0.00, 0.25), so the
    # operator has 180 mm of travel forward and back, 280 mm each way sideways,
    # and 170 mm up and down before anything clamps. The posture it replaced was
    # better conditioned in isolation (manipulability 0.049 against 0.040) but
    # offset 80 mm forward, which cost more in usable travel than the
    # conditioning was worth.
    #
    # No joint uses more than 68% of its travel, and the box corners nearest the
    # ceiling are out of reach at fixed orientation -- expected, since a box
    # always overhangs a roughly spherical reachable set, and an unreachable goal
    # is refused by IK rather than acted on.
    home_joints: Tuple[float, ...] = (
        0.006413, 0.330561, -0.017369, 1.926388, -0.831742, 0.066788, -0.143514
    )

    # 退出时给臂断电。**默认不断。**
    #
    # 原来是硬编码断电的，结果评测每按一次 q 臂就掉电，下一轮开起来报
    # 「网络异常: 网络连接错误」（SDK 在未上电时就是报这个，和网线无关），
    # 每轮都得先 power_on 一次。断开前已经 stopMove + 回 NrtCommandMode，
    # 臂只是带电保持位姿，这是机械臂两次任务之间的常态。
    #
    # 真要断电用 `power_on --off`，那是一次明确的操作者动作。
    power_off_on_disconnect: bool = False

    # --- tool frame and payload, for the dexterous hand ---
    # Both should be set: the hand moves the TCP and adds mass, and the arm's
    # force control depends on a correct dynamic model.
    tcp_offset: Tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)  # VERIFY: hand TCP
    payload_mass: float = 0.0           # VERIFY: hand mass in kg (RH56 is around 0.5 kg)
    payload_com: Tuple[float, float, float] = (0.0, 0.0, 0.0)       # VERIFY: centre of mass

    # --- 碰撞检测 ---
    # 2026-09-17：拧螺丝的反扭矩会触发控制器默认的碰撞检测，默认行为是硬停 ——
    # **掉电、抱闸**（现象就是"又抱死了"），同时 RT 会话被打断、控制器残留
    # `rtControlling`，之后所有排队式 Move* 都被拒（`ec=262 运动控制模式错误`）。
    # 一次触发，两个症状。
    #
    # `sensitivity` 范围 0.01~2.0，**越小越不灵敏**（触发阈值越高）。改成柔顺停止
    # 之后，真触发了也只是变柔顺、不掉电，不用重新上电。
    #
    # 每次 connect 都要重设 —— 这个配置活不过控制器重启。
    # 空元组 = 完全不碰，用控制器自己的默认值。
    #
    # 2026-09-21：0.3 -> 0.01，即 SDK 允许的**最不灵敏**一档（范围 0.01~2.0，越小
    # 阈值越高）。注意这个值当前**不生效** —— 下面 collision_detection_enabled
    # 是 False，驱动直接调 disableCollisionDetection 就 return 了，压根走不到这里。
    # 留最不灵敏的值是为了万一有人把检测打开，起点是最高阈值而不是 0.3。
    collision_sensitivity: Tuple[float, ...] = (0.01,) * 7
    # stop0 安全停止 / stop1 触发暂停 / stop2 / suppleStop 柔顺停止
    collision_behaviour: str = "suppleStop"
    # 安全停止/暂停时是回退距离(米)；柔顺停止时是柔顺度 [0.0, 1.0]
    collision_fallback: float = 0.5

    # 把**控制器层**的碰撞检测整个关掉。
    #
    # 2026-09-17：拧螺丝时 J7 传动力矩 17.119 Nm 顶破 17.000 Nm 的限制，臂停机掉电。
    # 但那个 17 Nm 是 **RSC（安全控制器）**的关节碰撞力限制，**SDK 里没有任何接口能改**
    # （穷举过所有类：只有 enableCollisionDetection 的灵敏度、和 setSoftLimit 的角度限位）。
    # 控制器也没开 web 端口（只有 22/873/6666/7777），只能用珞石的示教器软件改。
    #
    # 所以这个开关能做的只是关掉**控制器层**那道（日志 id=30400「触发柔顺停止」），
    # **RSC 那道（id=13036）关不掉**。两道报的是同一个 17 Nm，所以关了大概率还是会停 ——
    # 留这个开关是为了能一次测出来「到底是哪一层在拦」，不是因为它一定管用。
    # 2026-09-17 按用户要求关掉：拧螺丝反复触发，臂停机掉电没法干活。
    # 关的是控制器层那道，RSC 的关节保护仍在。
    #
    # 2026-09-21 又被要求「把碰撞阈值调高」，重新核实一遍：**软件这边已经到头了**。
    # 直接列了 SDK 的全部成员，和碰撞/力矩/限位沾边的只有
    # enableCollisionDetection / disableCollisionDetection（灵敏度 + 停止方式 +
    # 回退量）和 setSoftLimit / getSoftLimit（关节**角度**限位）。
    # **没有任何力矩阈值接口。** 关掉检测已经是软件能做到的最大放宽，
    # 再往上就是 RSC 那道 17 Nm，只能用珞石示教器软件改。
    collision_detection_enabled: bool = False

    # --- real-time joint streaming ---
    # The sender pushes one JointPosition per millisecond; the controller reads a
    # longer gap as packet loss and a shorter one as a servo fault.
    rt_period: float = 0.001            # seconds between commands

    # Joint speed cap. Derived from the tool speed rather than picked directly:
    # at the 0.68 m reach, w rad/s at the shoulder is 0.68*w m/s at the flange.
    #
    # Raised 2026-09-03 from 0.25 m/s (0.37 rad/s) to 0.5 m/s. The old value was
    # a conservative starting point, not something measured -- teleoperation at
    # 21 deg/s reads as the arm lagging the operator. 0.5 m/s is still under
    # ISO/TS 15066's 0.25 m/s *quasi-static contact* guidance only in the sense
    # that the operator is not meant to be inside the envelope while it runs:
    # keep hands clear and the e-stop within reach.
    #
    # Below this the tracker is unchanged; raising the cap does not make the arm
    # jumpy, it only stops it saturating when the leader moves quickly.
    max_tool_speed: float = 0.50        # m/s at the flange
    max_joint_speed: float = 0.74       # rad/s, consistent with the above
    # Left at 2.0 after raising the speed cap. Briefly 4.0, which made the arm
    # visibly shake: acceleration enters the tracker twice, and both make it
    # more aggressive near the goal --
    #     A_brake = 0.5*A  ->  sqrt(2*A_brake*e) allows sqrt(2)x the speed at
    #                          the same error, so it brakes later
    #     dv capped at A*dt -> commanded velocity can swing twice as fast
    # Raising the speed cap alone only stops the arm saturating; raising this
    # changes how hard it corrects, which is what turns tracking lag into
    # overshoot and then oscillation.
    max_joint_accel: float = 2.0        # rad/s^2

    # A published goal further than this from the current command is refused.
    # Rate limiting bounds how fast the arm travels, not where it ends up: an
    # inverse-kinematics branch flip would otherwise be tracked smoothly all the
    # way to a posture radians away. At 20 Hz and 0.37 rad/s a legitimate step is
    # under 0.019 rad, so this leaves a wide margin.
    joint_jump_max: float = 0.15        # radians, per joint

    # How far a joint-space goal may advance per control step. Deliberately well
    # below `joint_jump_max`: rate-limiting to exactly the gate's threshold puts
    # every command on the boundary, where rounding decides the outcome --
    # 1.926388 - 0.15 comes back as 0.15000000000000013 away, which the gate
    # reads as a jump and refuses. Measured: that rejected 100% of goals and the
    # arm never moved. At 20 Hz this still allows 86 deg/s, four times what the
    # tracker's own speed cap can execute, so it never limits responsiveness.
    # 2026-09-23 从 0.075 降到 0.014。原来的值允许目标每秒前进 129°，而臂自己的
    # 上限 `max_joint_speed` 只有 42°/s —— **目标可以跑在臂前面三倍**，差额存成
    # 跟踪误差；主臂一停，臂还得把这段差额走完。用户报的「行程结束后有动作冗余」
    # 就是它在展开。
    #
    #     0.74 rad/s ÷ 50 Hz = 0.0148 rad/帧
    #
    # 命令自己做不到的速度没有意义，只会把超出的部分变成事后才还的债。
    # 改 control_hz 或 max_joint_speed 的话这个数要跟着重算。
    joint_goal_step: float = 0.014      # radians, per joint, per control step

    # The sender brakes to rest if the goal stops being refreshed for this long,
    # then latches: motion may not resume without an explicit operator action.
    goal_stale_timeout: float = 0.25    # seconds

    # Time constant of the low-pass on the tracker's feed-forward estimate.
    # The estimate is (goal - previous goal) / elapsed, so it inherits both the
    # leader encoder's quantisation and the control loop's period jitter, and
    # the tracker acts on it at 1 kHz. 0.03 s reproduces the old fixed 0.4 blend
    # at the nominal 50 Hz while staying the same amount of smoothing when the
    # loop runs long. Raise it if the arm still ripples while tracking a steady
    # hand; the cost is that the arm takes longer to pick up a change of speed.
    goal_rate_tau: float = 0.03         # seconds

    # Jerk limit for the 1 kHz tracker, rad/s^3. 0 keeps the second-order
    # tracker (velocity + acceleration caps, sqrt braking law). > 0 switches the
    # sender to a jerk-limited online S-curve (Ruckig) replanned every tick
    # toward the latest goal. Motivation (2026-09-20, AVP teleop): the goal
    # arrives at 50 Hz with sub-mm noise, the second-order tracker flips its
    # acceleration between +A and -A on every goal update, and that square
    # wave is audible as motor buzz. A jerk cap turns the square wave into a
    # ramp. Requires the `ruckig` package; falls back to second order if absent.
    max_joint_jerk: float = 0.0         # rad/s^3

    def __post_init__(self):
        if not 0.0 < self.joint_goal_step < self.joint_jump_max:
            raise ValueError(
                f"joint_goal_step ({self.joint_goal_step}) must be positive and strictly "
                f"below joint_jump_max ({self.joint_jump_max}). A rate limit equal to the "
                "gate's threshold puts every command on the boundary, where floating-point "
                "rounding decides whether it is refused -- measured, that rejected every "
                "goal and the arm never moved."
            )

        for name in ("home_joints", "joint_min", "joint_max"):
            got = len(getattr(self, name))
            if got != self.num_joints:
                raise ValueError(
                    f"{name} has {got} entries but num_joints is {self.num_joints}"
                )
        # A home posture outside the soft limits cannot be reached, so reset()
        # would fail on every episode. Catch it at construction instead.
        outside = [
            (i, q) for i, q in enumerate(self.home_joints)
            if not (self.joint_min[i] <= q <= self.joint_max[i])
        ]
        if outside:
            detail = ", ".join(
                f"j{i}={q:.4f} outside [{self.joint_min[i]:.4f}, {self.joint_max[i]:.4f}]"
                for i, q in outside
            )
            raise ValueError(f"home_joints violates the joint soft limits: {detail}")


@dataclass
class HandConfig:
    """Inspire RH56E dexterous hand.

    **This hand speaks CAN, not Modbus RTU.** The link is
    ``PC --USB-- USB-CAN adapter (CH340) --CAN 1000K-- RH56E``, so ``port`` is a
    serial node and ``baudrate`` is the adapter's UART rate, not the CAN rate.
    The Modbus register block below is dead on this hardware and kept only for
    the older RS485 variants; the live driver is ``drivers/inspire_hand_can.py``,
    which delegates to the byte-for-byte verified codec in ``VEL/src/hand_teleop``.

    Every value in the "measured" block was calibrated against this physical
    hand on 2026-09-01. Changing ``fixed_pose``'s thumb entries invalidates
    ``finger_close`` -- the contact point moves with the thumb (measured: thumb
    at 400 -> contact 484; thumb at 600 -> contact 370, a 114-count shift).
    """

    # **by-id，不要写 ttyUSBn。** 2026-09-23 这里写死 `/dev/ttyUSB0` 害过一次：
    # 灵巧手换成夹爪之后，夹爪的 Prolific 转 485 占了 ttyUSB0，而手的驱动照旧
    # 去开 ttyUSB0 —— 开得成，但收不到一个字节，报的是
    # 「未扫到合法帧 (0 字节): 空」，看起来像手坏了或者 CAN_H/L 接反。
    #
    # `ttyUSBn` / `ttyACMn` **按插入顺序给**，插拔一次就换位。本机同时插着
    # 三个 CH340/Prolific 类的东西，靠编号区分是碰运气：
    #
    #   主臂 SO-101  usb-1a86_USB_Single_Serial_5B79051033-if00  -> ttyACM0
    #   手 USB-CAN   usb-1a86_USB_Serial-if00-port0              -> ttyUSB*
    #   夹爪 RS485   usb-Prolific_..._ATCNl2AC568-if00-port0     -> ttyUSB*
    #
    # 主臂和手都是 1a86（CH340），**产品串不一样**才分得开：手那个没有序列号、
    # 带 `-port0` 后缀。这个值和 `数据_硬件/灵巧手遥操/VR设备/avp_servoj_teleop.py`
    # 的 `--hand-port` 默认值是同一个，两条路不要各写各的。
    port: str = "/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0"
    baudrate: int = 115200              # Adapter UART rate. CAN runs at 1000K.
    slave_id: int = 1
    timeout: float = 0.1

    # ── CAN driver (live path) ──────────────────────────────────────────────
    protocol: str = "can"               # can / eb90 / modbus:byte / modbus:word
    dev_id: int = 1                     # Hand's CAN device id

    # ─────────────────────────────────────────────────────────────────────
    #  改手的姿态就改这一块，下面 fixed_pose / closed_pose / open_pose 会自动跟着变
    # ─────────────────────────────────────────────────────────────────────
    #
    # 角度都是 1000 = 完全张开，0 = 完全握紧。
    #
    # 两指对向捏取：opening 从 1 走到 0 的过程中，食指和拇指同时往中间收。
    # 每根手指有「张开」和「闭合」两个端点，中间线性插值。
    #
    # 想回到以前那种「拇指当挡板不动、只有食指去撞它」的方式，
    # 把 thumb_open 和 thumb_close 设成同一个数就行。
    #
    # 改完任何一个端点都要重新试捏，看两指闭合时是不是刚好压住东西
    # 而不是互相顶死或者够不着：
    #
    #     python -m atom_robot.ar5.jog_hand --apply        # 摆张开姿态
    #     python -m atom_robot.ar5.jog_hand --grip 0       # 试闭合
    #
    # 2026-09-03 两指对向实测：食指和拇指一起往里收，到「食指 495 / 拇指 273」
    # 时互相顶上（指令给的是 450/150，都没走到）。闭合端各压过约 15 个计数。
    #
    # 单指时代的数字（拇指不动、食指去撞）已经不适用了，留着做参考：
    #     拇指 600 -> 接触点 370      拇指 300 -> 接触点 521
    #     拇指 500 -> 接触点 490      单指的 finger_close 是 507
    thumb_open: int = 1000               # 张开时的拇指弯曲（opening = 1）
    thumb_close: int = 500              # 闭合时的拇指弯曲（opening = 0），接触点 273 再进 15
    thumb_rot: int = 100                # 拇指旋转，全程不动，实际停在 93 左右
    other_fingers: int = 0              # 小指/无名指/中指，收起来让开视野
    finger_open: int = 1000             # 食指张开到哪（opening = 1）
    finger_close: int = 480             # 食指闭合到哪（opening = 0），接触点 495 再进 15
    # 力控阈值，克。到这个力手指就停，所以它同时是「能抓多紧」和「顶到拇指时
    # 的保护」两件事。掉电不保存，每次连接自动重设。
    #
    # 2026-09-03：300 g 能夹住螺栓但提不起来 —— 手指一受力就判定到位停住了，
    # 夹持力不够克服重量。先提到 800，再按要求拉到上限 1000。
    #
    # 1000 是驱动允许的最大值。两指对向捏取时这个值也是唯一的过载保护：
    # 两根手指都在动，谁也不是不动的挡板，所以位置指令给到底时全靠力控停住。
    force: int = 1000
    speed: int = 300                    # 0..1000，同样掉电不保存
    # ─────────────────────────── 以下一般不用动 ───────────────────────────

    index_dof: int = 3                  # Index finger is DOF 3 across the RH56 series

    # Register map, cross-checked against the RH56 series user manual (section
    # 2.6). All the 6-wide blocks are 6 consecutive uint16 registers ordered
    # [little, ring, middle, index, thumb-bend, thumb-rotate].
    angle_set_register: int = 1486      # ANGLE_SET(m)   W/R  -1, 0-1000
    angle_act_register: int = 1546      # ANGLE_ACT(m)   R    0-1000
    force_set_register: int = 1498      # FORCE_SET(m)   W/R  0-3000
    force_act_register: int = 1582      # FORCE_ACT(m)   R
    speed_set_register: int = 1522      # SPEED_SET(m)   W/R  0-1000
    error_register: int = 1606          # ERROR(m)       R    6 bytes
    status_register: int = 1612         # STATUS(m)      R    6 bytes
    temp_register: int = 1618           # TEMP(m)        R    6 bytes
    clear_error_register: int = 1004    # CLEAR_ERROR    W/R  write 1

    num_joints: int = 6
    angle_min: int = 0                  # Fully flexed / closed
    angle_max: int = 1000               # Fully extended / open
    force_max: int = 3000               # Force registers use a wider range than angles
    hold_value: int = -1                # ANGLE_SET sentinel: leave this DOF where it is

    # Below this the commanded opening counts as "closed" for logging.
    close_threshold: float = 0.5

    # ── 下面三个是算出来的，不是配置项 ──────────────────────────────────
    # 以前它们是三个独立字段，改拇指要手动同步四处，漏一处就是配置和实物对不上。
    # 现在从上面那一块推出来，改不了也不会不一致。
    # 顺序固定：[小指, 无名指, 中指, 食指, 拇指弯曲, 拇指旋转]

    def pose_for(self, opening: float) -> Tuple[int, ...]:
        """opening（0=闭合，1=张开）对应的六维姿态。

        食指和拇指同时插值，所以两根手指是对着往中间收，而不是食指去撞一个
        不动的拇指。thumb_open == thumb_close 时拇指自然就不动了。
        """
        t = min(1.0, max(0.0, float(opening)))
        index = self.finger_close + t * (self.finger_open - self.finger_close)
        thumb = self.thumb_close + t * (self.thumb_open - self.thumb_close)
        return (self.other_fingers, self.other_fingers, self.other_fingers,
                int(round(index)), int(round(thumb)), self.thumb_rot)

    @property
    def fixed_pose(self) -> Tuple[int, ...]:
        """张开时的姿态。名字是历史遗留 —— 拇指现在也会动了。"""
        return self.pose_for(1.0)

    @property
    def closed_pose(self) -> Tuple[int, ...]:
        """opening = 0 对应的六维姿态。"""
        return self.pose_for(0.0)

    @property
    def open_pose(self) -> Tuple[int, ...]:
        """opening = 1 对应的六维姿态。"""
        return self.pose_for(1.0)

    def __post_init__(self):
        if not 0 <= self.finger_close < self.finger_open <= 1000:
            raise ValueError(
                f"食指行程不对：finger_close={self.finger_close} 必须小于 "
                f"finger_open={self.finger_open}，两个都要在 0..1000 之间"
            )
        if self.finger_close < 100:
            raise ValueError(
                f"finger_close={self.finger_close} 太小了，食指会顶死拇指持续堵转。"
                "先跑 jog_hand --find-contact 测出接触点，再取「接触点 - 14」"
            )
        for name in ("thumb_open", "thumb_close", "thumb_rot", "other_fingers",
                     "force", "speed"):
            value = getattr(self, name)
            if not 0 <= value <= 1000:
                raise ValueError(f"{name}={value} 超出 0..1000")


@dataclass
class CameraConfig:
    """Camera sources, keyed by the observation name the policy expects."""

    # OpenCV device index, GStreamer pipeline, or video file path.
    #
    # Two UGREEN 1080P cameras (measured 2026-09-03). Addressed by USB port, not
    # by index, because both report the same USB id AND the same serial
    # (SN0001) -- /dev/v4l/by-id/ holds only one of them, and video0 / video2
    # swap on a replug. Swapping them silently would feed the wrist view in as
    # the overhead view, which is not something a trained policy would survive.
    #
    # by-path is tied to the physical socket: keep each camera in its own port
    # and these stay correct. Move one and re-run
    # `python -m atom_robot.ar5.view_camera --list` to get the new path.
    #
    # Each camera also claims a second, metadata-only node (video1, video3);
    # opening those with OpenCV hangs. Only -index0 is the colour stream.
    sources: Dict[str, object] = field(
        default_factory=lambda: {
            # Overhead, looking down at the bench
            # **固定名字，由 udev 规则给**（`99-ar5-cameras.rules`，装在 /etc/udev/rules.d/）。
            #
            # 2026-09-24 从 by-path 换过来。by-path 绑的是**物理插座**，而腕部这台
            # Orbbec 卡死时唯一的救法就是拔插 —— 一拔它就从 10.3 口挪到 10.2，
            # 路径立刻失效、collect 报「路径不存在」。等于每救一次就要改一次 config。
            #
            # 为什么不用现成的 by-id：**彩色节点没有 by-id**。Orbbec 的六个 by-id
            # 链接指向接口 1.0 和 1.6（深度/元数据），偏偏漏掉 1.4 那两个彩色节点。
            #
            # **2026-09-30 起两台都走 OBSENSOR，写成 "orbbec:<序列号>"。** 原来走 V4L2
            # （上面那套 udev 固定名），但 RobotJev 用 OBSENSOR 开过同一台相机之后，
            # V4L2 就一帧不出，复位也救不回来 —— 两个后端互相污染。两边都用 OBSENSOR
            # 就没这回事了。udev 名字留着（`/dev/ar5_cam_high`、`/dev/ar5_cam_wrist`），
            # 想退回 V4L2 把这两行改回去就行。（腕部 305 在 10-15 已经改回 V4L2，见下面。）
            #
            # 顶部：Gemini 335（9/29 替掉了绿联，**cam_high 在 9/29 前后不是同一台相机**）
            "cam_high": "orbbec:CP0BB53000FC",
            # 腕部，跟着臂走。2026-09-22 换成 Orbbec Gemini 305，顶掉了原来
            # 插在这个口上的绿联侧视相机 —— 所以**这个键在新旧数据里不是同一个
            # 视角**，两批不能混（新的采到别的目录去）。键名没改是因为
            # `run_ar5_eval.py` 里它已经就是 wrist_image，而且 893 个已有文件
            # 也叫这个名字。
            #
            # **接口号是 `:1.4` 不是 `:1.0`。** Orbbec 一个设备占八个接口，
            # `:1.0` 是深度（Z16）、`:1.6` 也是深度，只有 `:1.4` 是彩色。
            # 填错的表现不是报错，是 `read()` 每次卡满 10 秒。
            #
            # 2026-09-23 实测这条路 640x480 @ 30.3 Hz，和顶部那台一样。
            # **不要改用 pyorbbecsdk**：它走 libuvc，会把内核驱动从彩色接口上
            # 摘掉、v4l2 节点整个消失（要再发一次 USB 复位才回来），而且默认档位
            # 是 848x530@10fps，实测只有 8.2 Hz —— 比这条路差，还多一个只装在
            # lerobot 里的依赖。
            # 腕部：Gemini 305
            #
            # **2026-10-15 起走 V4L2（`v4l2:` 前缀，不让驱动改道 OBSENSOR）。** 305 走 OBSENSOR
            # 10-09 起一帧不出，还会把 V4L2 也弄得不出帧，只能拔插救。10-15 真机核对：
            # 335 走 OBSENSOR + 305 走 V4L2 同时开，29.7 / 30.1 Hz，640x480 RGB，305 零读失败。
            # 别再改回 "orbbec:CV27864000V2"。代价：别的程序（比如 RobotJev）用 OBSENSOR 碰过
            # 305 之后，这里也会一帧不出，要拔插。
            "cam_front": "v4l2:/dev/ar5_cam_wrist",
        }
    )
    # 哪几路顺带取深度（只对 Orbbec 有效）。`orbbec:` 源同一次 grab 取深度（已对齐彩色）；
    # `v4l2:` 源另开同一台的深度节点（Z16，没对齐彩色）。取了能用 `cameras.latest_depth(name)` /
    # `depth_since()` 拿。collect 加 `--depth` 时设成全部相机并存盘（10-30 起，OmniPicker 模式）。
    depth_sources: Tuple[str, ...] = ()
    width: int = 640
    height: int = 480
    fps: int = 30

    @property
    def names(self) -> List[str]:
        return list(self.sources.keys())


@dataclass
class CollectConfig:
    """采数据的参数。`collect.py` 读这里，命令行参数可以临时覆盖。

    改这个文件是**持久**的，下次开机还在；命令行参数只影响这一次。
    一组数据采到一半不要改这里 —— 幅度、帧率、任务描述变了，
    不同 episode 的尺度就不一致了。
    """

    # ─────────────────────────────────────────────────────────────────────
    #  开采之前确认这一块
    # ─────────────────────────────────────────────────────────────────────

    # 这条数据在做什么，一句话。会原样写进每个 .npz。
    #
    # 用**英文小写祈使句**，和 OpenVLA / LIBERO 训练数据的说法保持一致
    # （"put the mug in the microwave" 那种）—— backbone 见过的就是这种句式。
    # 中文意思：将螺栓拿起放入蓝色螺套中。
    task: str = "pick up the bolt and put it in the blue sleeve"

    # 存到哪。每条 episode 一个 .npz + 每路相机一个 .mp4。
    # 已有几条就从几号接着编，不会覆盖。
    #
    # 2026-09-23 从 `~/vel_episodes` 改过来。**老数据还在 `~/vel_episodes`，
    # 没动也没删**，那里有 297 条 09-08 采的 —— 那批的 `cam_front` 是台架侧视，
    # 而现在是腕视，**同名不同视角，本来就不该续在一起**。要用老数据显式给 --out。
    out_dir: str = "~/atom_episodes"

    # `--end-effector omnipicker` 按**数据集**采（2026-10-20 用户定的）：按 P 输入任务名，
    # 存到 `dataset_root/任务名/`，每个任务一个目录，目录里另有 dataset.json 记任务名和
    # 语言 prompt。给了 --out 就用 --out 当根目录。其他末端还是直接存 out_dir。
    dataset_root: str = "~/atom_episodes/new"

    # 录制帧率。也是写进 mp4 和 .npz 的 fps。
    #
    # 相机是 30 fps，所以超过 30 只会录到重复帧（`stale_reads` 会涨）。
    # 低于 30 是可以的，数据小一点、动作看起来更连续。
    hz: float = 30.0

    # **控制频率**：多久给臂发一次目标。和上面的录制帧率是两回事，2026-09-23 拆开的。
    #
    # 原来两件事共用 `hz`，为了配相机压在 30 —— 遥操手感就是这么丢的。臂的 1 kHz
    # 跟踪器在两个目标之间**加速再刹车**，目标给得越稀，工具上越像一顿一顿的；
    # 跟踪器滞后实测约 3 个周期，30 Hz 是 100 ms，50 Hz 只有 60 ms。
    # VEL 的 `teleop_joint` 默认就是 50 Hz，手感差别主要在这儿。
    #
    # 拆开之后回路按 control_hz 跑，录制按 hz 的节拍取帧，两者互不牵扯。
    control_hz: float = 50.0

    # --- 一组数据怎么编排 -------------------------------------------------
    #
    # **每一条都手动起停，存完不会自动开下一条。** 空格开录、`s` 结束并保存，
    # 存完回到待机等你摆好现场再按空格。自动切换试过，问题是「下一条已经开录
    # 了但人还没准备好」，白录一条。
    #
    # 一条 take & place 大概 10~20 秒。

    # 这一次采够多少条就收工。0 = 不限，一直录到按 q。
    # 数的是**这一次采的新条数**，编号从磁盘上已有的接着走。
    num_episodes: int = 50

    # 一条最长录多少秒，到点自动存。**0 = 不限，完全靠 `s` 结束**（默认）。
    #
    # 只是个兜底，防止忘了按 s 录出一条几分钟的。设了也不会自动开下一条。
    episode_time_s: float = 0.0

    # --- 遥操作手感，和 teleop_joint 的同名参数一个意思 ---

    # True = 按主臂自接合起走的位移驱动，松开离合器可以重新抓握接着走。
    # False = 主臂在哪臂就在哪。
    #
    # ⚠️ 注意 relative 不等于 1:1：`joint_calibration.json` 里存的
    # scale_overrides 优先级更高，本机实际是 0.40~1.00。启动时打的表里
    # `scale` 那一列才是真的。
    relative: bool = True

    # 遥操走哪个空间。2026-09-24 加的。
    #   "joint"      关节一对一映射，主臂 5 个关节对从臂 5 个轴，**轴3、轴5 锁死**
    #   "cartesian"  主臂末端位姿 -> 逆解 -> **7 个轴全动**（cartesian_mapping.py）
    # 两条路存盘格式一样（state 关节角 / ee_pose / action 笛卡尔增量都有），
    # 但**数据分布不同**：joint 下轴3、轴5 恒定，cartesian 下它们会动。
    # 一批数据里别混着用。
    space: str = "joint"

    # 在上面那些比例之上再乘一个系数。0.5 = 幅度减半。
    # cartesian 下它就是位置缩放：主臂末端走 1 cm，从臂走 scale cm。
    #
    # **2026-10-20 从 1.0 改成 0.5**（用户：遥操太快）。主臂走多少从臂走一半，同样的手速
    # 从臂速度也减半；代价是主臂行程要两倍，推到头就暂停（OmniPicker 回车 / 其他末端 P）收回来。
    # **10-20 之前采的数据都是 1.0**，和之后的别混成一个数据集；要回到 1.0 给 --scale 1。
    scale: float = 0.5

    # 接合时从当前姿态平滑过渡到映射姿态用多少秒。
    ramp_seconds: float = 2.5

    # 主臂读数的 1-euro 滤波，压手抖和量化。None = 用 JointMapConfig 的默认
    # （1.5 Hz / 0.05）。抖就把 min_cutoff 往小调，发黏就把 beta 往大调。
    # 详见 VEL/docs/arm-shake.md。
    filter_min_cutoff: Optional[float] = None
    filter_beta: Optional[float] = None

    # --- 存盘之前的质量检查，不达标只是打 ⚠ 提醒，仍然会存 ---

    # 短于这个帧数就提醒。太短的 episode 学不到完整动作。
    min_frames: int = 10

    # 手的开合全程变化不到这个数就提醒。手全程没动 = action 第 7 维是常数，
    # gripper 那一维没有任何可学的东西。这是最常见的废数据。
    min_gripper_travel: float = 0.02

    # 末端**一共走过**多少毫米（逐帧位移累加）不到这个数就提醒。
    #
    # 原来量的是「首尾直线距离」，那是错的判据：take & place 做完之后臂往往
    # 就在起点附近，首尾自然接近。2026-09-04 用 8 条真实数据回测，那个判据
    # 误报 4 条（首尾 5~14 mm，实际路径 730~920 mm，最悬殊的差 182 倍），
    # 真阳性 0。路径长度没有这个问题：真死掉的那两条路径只有 0.6 mm。
    #
    # 实测正常一条是 730~1100 mm，所以 100 是个很宽的下限。
    min_ee_path_mm: float = 100.0

    # ─────────────────────────── 以下一般不用动 ───────────────────────────

    # 采集时开一个本地窗口实时看画面。**独立线程，不进主回路** ——
    # cv2.waitKey 在 X 卡一下就可能超过臂 0.25s 的 latch 线，不能拿它赌 episode。
    preview: bool = False
    preview_hz: float = 15.0        # 看画面够用，比录制的 30Hz 轻

    # 网页驱动（--remote）时回传画面的频率：录制中 / 待机刷新预览。0 = 不回传。
    remote_fps: float = 10.0
    remote_idle_fps: float = 4.0

    # 只用这几路相机，逗号分隔，如 "cam_high"。空 = config.cameras.sources 里配的全上。
    # 另一路没插时必须给，否则 env.connect() 会在**臂已经上电之后**才失败。
    # **少一路的数据和全路的不能混成一个数据集**（相机路数不一样）。
    cameras: str = ""

    calib_id: str = "single_leader_arm"     # 本机有两份主臂标定，必须指名
    # by-id，不是 /dev/ttyACM0。2026-10-19 起 OmniPicker 夹爪的 hkvtech USB-485 也是
    # CDC 串口（ttyACMn），和主臂一起插时编号按插入顺序给：写死 ttyACM0 的话，
    # 转换器先插就会把主臂的舵机协议灌进夹爪的 485 总线，主臂自己反而连不上。
    # 这条路径抄自 HandConfig.port 上方的注释（9-23 记的），改的时候主臂没插，
    # **插上后核对一次** `ls -l /dev/serial/by-id/`。
    leader_port: str = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B79051033-if00"
    leader_baud: int = 1_000_000

    # mp4v 到处都能播。想要更小的文件可以试 "avc1"，但不是每个 OpenCV
    # 构建都带 H.264 编码器，换了要先确认真的写出了非空文件。
    video_codec: str = "mp4v"

    def __post_init__(self):
        if self.hz <= 0:
            raise ValueError(f"hz 必须是正数，给的是 {self.hz}")
        if not self.task.strip():
            raise ValueError("task 不能是空的 —— 它会写进每个 .npz，训练时要用")
        if self.scale <= 0:
            raise ValueError(f"scale 必须是正数，给的是 {self.scale}")
        if self.min_frames < 1:
            raise ValueError(f"min_frames 至少是 1，给的是 {self.min_frames}")
        for name in ("filter_min_cutoff", "filter_beta"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} 不能是负数，给的是 {value}")
        if len(self.video_codec) != 4:
            raise ValueError(
                f"video_codec 必须是 4 个字符的 FourCC，给的是 {self.video_codec!r}"
            )

        if self.num_episodes < 0:
            raise ValueError(f"num_episodes 不能是负数，给的是 {self.num_episodes}")
        if self.episode_time_s < 0:
            raise ValueError(f"episode_time_s 不能是负数，给的是 {self.episode_time_s}")

        # 设了上限的话，录满都到不了 min_frames 就意味着**每一条**都会被打
        # 「太短」的警告 —— 那不是数据的问题，是这两个参数互相矛盾。
        if self.episode_time_s > 0:
            frames = self.episode_time_s * self.hz
            if frames < self.min_frames:
                raise ValueError(
                    f"episode_time_s={self.episode_time_s}s × hz={self.hz} 只有 "
                    f"{frames:.0f} 帧，少于 min_frames={self.min_frames} —— "
                    "这样每条都会被判定太短。把时间加长或者把 min_frames 降下来。"
                )

    # ── 派生值，不用手填 ───────────────────────────────────────────────────

    @property
    def frame_cap(self) -> int:
        """一条最多录多少帧。0 = 不限，靠 `s` 结束。"""
        return int(round(self.episode_time_s * self.hz)) if self.episode_time_s > 0 else 0

    @property
    def mapping_overrides(self) -> Dict[str, object]:
        """喂给 JointMapConfig.load() 的那几个键。None 的不传，用它自己的默认。"""
        out: Dict[str, object] = {
            "relative": self.relative,
            "scale_multiplier": self.scale,
            "ramp_seconds": self.ramp_seconds,
        }
        if self.filter_min_cutoff is not None:
            out["filter_min_cutoff"] = self.filter_min_cutoff
        if self.filter_beta is not None:
            out["filter_beta"] = self.filter_beta
        return out


@dataclass
class EvalConfig:
    """跑策略（`eval_policy.py`）的参数。命令行同名参数可以临时覆盖。

    改这个文件是**持久**的，下次开机还在；命令行只影响这一次。
    """

    # ─────────────────────────────────────────────────────────────────────
    #  开跑之前确认这一块
    # ─────────────────────────────────────────────────────────────────────

    # 一轮最多跑多少步，到了自动停。**30 Hz 下 = 步数 / 30 秒。**
    #
    #     600   =  20 秒
    #    1800   =   1 分钟
    #    6000   = 200 秒 = 3.3 分钟
    #
    # 参考：示教数据每条 315~674 帧、中位 409 帧 = 13.7 秒。跑得比示教长很多时，
    # 后半段策略是在它没见过的时间尺度上外推 —— 不危险，但别把那一段的表现
    # 当成"模型不行"的证据。
    max_steps: int = 6000

    # 控制频率。**要和采集时一致**，否则同一个动作增量对应的真实速度就变了。
    hz: float = 30.0

    # 动作整体缩放。第一次上机、或者换了 checkpoint，建议 0.5 起步。
    scale: float = 1.0

    # 每一轮**开跑前**把臂和手开到任务起点（100 条首帧的中位 state，
    # 7 关节 + 夹爪）。采集是遥操作，起点是操作者的预备位；从别处起跑，
    # 策略第一帧就吃到训练分布外的输入。实测 config.home_joints 和任务起点
    # 差 46.8°（L2 60.6°），夹爪起点中位是 0.61 不是 0。
    # 关掉只在「臂已经在起点、不想让它再动一次」时才有意义。
    goto_start: bool = True

    # 退出时回任务起始位姿。False = 留在原地。
    # 和 goto_start 是同一个目标位姿，走同一条代码路径。
    return_to_start: bool = True

    # ─────────────────────────── 以下一般不用动 ───────────────────────────

    device: str = "cuda"


@dataclass
class HKVRConfig:
    """TG9801 directly over RS485 Modbus; force is a register value, not N."""
    # by-id，不是 ttyUSBn。这台机器上同时插着两个 USB 串口设备：
    # CH340（原来接因时手的 USB-CAN 适配器）和这个 Prolific 的 USB 转 RS485，
    # **ttyUSB0/1 的编号按插入顺序给，重插一次就互换**，接错口的表现是完全没有回包。
    # by-id 绑的是芯片序列号，不会变。
    #
    # 2026-09-22 实测通过：这条线上 1 Mbps、从站 1、Modbus RTU 读到目标力 50。
    port: str = ("/dev/serial/by-id/"
                 "usb-Prolific_Technology_Inc._USB-Serial_Controller_ATCNl2AC568-if00-port0")
    transport: str = "rs485"
    baudrate: Optional[int] = None
    can_id: int = 1
    can_rx_id: int = 1
    slave_id: int = 1
    speed: int = 1000

    # 目标夹取力，寄存器值 1~100（**不是牛顿**；整机规格是单指 0.02~30 N）。
    #
    # 2026-09-24 从 20 提到 50。20 太低，夹爪在闭合途中一点摩擦就判定「抓稳」
    # 并就地停住 —— 实测手里空着也读到过 `(1046, 3, '抓稳')`。现象是
    # 「主臂都合住了，从臂夹爪不合」，而且时好时坏。
    #
    # 50 是这台夹爪上电时自己报的目标力（第一次读寄存器 0x0012 就是 50），
    # 拿它当基准比拍一个数有依据。要更大夹持力再往上调，但**先想清楚夹的是
    # 什么** —— 规格上限 30 N，易碎件要往小调。
    #
    # 注意这个值**掉电不保存**，每次连接时驱动会重新写一遍。
    force: int = 50
    timeout: float = 0.05
    hz: float = 30.0
    stale_timeout: float = 0.5

    def __post_init__(self):
        import math
        if self.transport not in ("rs485", "can"):
            raise ValueError("HKVR transport 必须是 rs485 或 can")
        if self.baudrate is None:
            self.baudrate = 115200 if self.transport == "can" else 1_000_000
        if not 0 <= self.can_id <= 0x7ff or not 0 <= self.can_rx_id <= 0x7ff:
            raise ValueError("HKVR CAN 标准帧 ID 必须为 0..0x7ff")
        if not 1 <= self.slave_id <= 247 or not 200 <= self.speed <= 1500 or not 1 <= self.force <= 100:
            raise ValueError("HKVR 要求 id=1..247、speed=200..1500、force=1..100")
        if (not all(math.isfinite(v) and v > 0 for v in
                    (self.timeout, self.hz, self.stale_timeout, self.baudrate))
                or self.timeout > 0.2 or self.hz > 100
                or self.stale_timeout <= 3 * self.timeout + 1 / self.hz):
            raise ValueError("HKVR 通信周期/超时参数无效")


@dataclass
class AR5Config:
    """Top-level bundle passed to :class:`AR5Env`."""

    arm: ArmConfig = field(default_factory=ArmConfig)
    hand: HandConfig = field(default_factory=HandConfig)
    # 末端装的是什么。**2026-09-23 从 "inspire" 改成 "hkvr"** —— 因时灵巧手已经
    # 从臂上拆了，现在装的是 HKVR-TG9801 夹爪。
    #
    # 改这个默认值的直接原因：默认值是 inspire 的时候，忘了加 `--end-effector hkvr`
    # 就会去连一只**不在臂上**的手，而且报的错完全指不到点子上 ——
    # 「未扫到合法帧 (0 字节): 空」，看起来像手坏了或者 CAN 线接反。
    # 默认值应该等于现场实际装的东西，不是历史上装过的东西。
    #
    # 手装回来的话把这里改回 "inspire"，或者命令行给 `--end-effector inspire`。
    end_effector: str = "hkvr"
    hkvr: HKVRConfig = field(default_factory=HKVRConfig)
    # 智元 OmniPicker（`--end-effector omnipicker`）。参数在 SO101/omnipicker/config.py，
    # 这里只留个位置，由 end_effector.configure_end_effector 填。不在这里 import 是因为
    # omnipicker 包在 SO101 目录下，别处 import atom_robot 时不该被它拖住。
    omnipicker: Optional[object] = None
    cameras: CameraConfig = field(default_factory=CameraConfig)
    collect: CollectConfig = field(default_factory=CollectConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    # When True every driver is replaced by an in-process simulator, so the
    # whole stack runs end to end with no hardware attached.
    mock: bool = False

    # Set False to run the arm alone. The gripper dimension of the action is
    # still accepted and logged, it just drives nothing -- which is what lets
    # arm-only rollouts happen before the hand is wired.
    use_hand: bool = True

    # Simulated arm, real hand: the rung between bringing the hand up alone and
    # driving both for real. Must be asked for explicitly -- `mock` on its own
    # always means nothing physical moves, and anything that quietly drove real
    # fingers under `--mock` would be a trap.
    real_hand_in_mock: bool = False

    # 模拟臂 + 真相机。给策略评测的 dry-run 用：模型看到的必须是真画面，
    # 否则"先空跑看看动作合不合理"这件事就没有意义了 —— 假图配出来的动作
    # 说明不了任何问题。和 real_hand_in_mock 一样是显式 opt-in。
    real_cameras_in_mock: bool = False

    # Whether each step waits for its motion to finish. False streams: the
    # command is queued and the loop continues, which is what a control loop
    # needs. True is only for scripted, one-move-at-a-time use.
    blocking_steps: bool = False

    # Seconds to wait after a reset before the first observation is read.
    settle_time: float = 1.5

    # How far the commanded setpoint may lead the measured pose. Beyond this
    # the target is pulled back along the line to the arm -- a leash, not a
    # reset. Re-anchoring instead would discard whatever lead the operator had
    # built up, which during fast teleoperation happens several times a second
    # and reads as the arm refusing to follow.
    max_lead: float = 0.015             # metres

    # Hard re-anchor threshold. Only for a setpoint that has genuinely lost
    # contact with reality -- a refused step, a collision stop, a stall -- not
    # for ordinary tracking lag, which the leash above absorbs.
    resync_threshold: float = 0.06
