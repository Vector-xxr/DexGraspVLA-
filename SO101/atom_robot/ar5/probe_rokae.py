"""Discover what the installed xCoreSDK actually exposes.

The SDK ships as a prebuilt binary whose exported names depend on the build, so
the robot class for a given arm is best answered by asking the library rather
than by reading documentation.

Three levels, each needing more than the last:

    # 1. SDK only, no robot, no network -- answers "which class?"
    python -m atom_robot.ar5.probe_rokae --list

    # 2. + a reachable controller -- answers "which class connects?"
    python -m atom_robot.ar5.probe_rokae --connect --ip 192.168.0.160

    # 3. + a powered arm -- answers "how many joints, and what pose convention?"
    python -m atom_robot.ar5.probe_rokae --connect --ip ... --read-state

Level 3 only reads; it never commands motion.
"""

import argparse
import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from atom_robot.ar5.drivers.rokae_arm import load_sdk  # noqa: E402


def _members(sdk, predicate):
    return sorted(n for n in dir(sdk) if not n.startswith("_") and predicate(n))


def list_symbols(sdk) -> None:
    """Print the classes and enums that matter for wiring up a driver."""
    robots = _members(sdk, lambda n: n.endswith("Robot"))
    print("\n=== Robot classes ===")
    if robots:
        for name in robots:
            print(f"  {name}")
    else:
        print("  (none found -- the SDK may name them differently)")

    print("\n  Known mapping:")
    print("    StandardRobot    industrial 6-axis")
    print("    xMateRobot       collaborative 6-axis")
    print("    xMateErProRobot  collaborative 7-axis   <- AR5 is 7-DoF")
    print("    PCB3/PCB4Robot   3/4-axis PCB robots")

    print("\n=== Motion command types ===")
    for name in _members(sdk, lambda n: "Command" in n or "Position" in n):
        print(f"  {name}")

    print("\n=== Real-time / follow control ===")
    for name in _members(sdk, lambda n: "Follow" in n or "RTmotion" in n or "RtMotion" in n):
        print(f"  {name}")

    print("\n=== Kinematic models ===")
    for name in _members(sdk, lambda n: "odel" in n):
        print(f"  {name}")

    for enum_name in ("OperateMode", "MotionControlMode", "OperationState", "RtSupportedFields"):
        enum = getattr(sdk, enum_name, None)
        if enum is None:
            print(f"\n=== {enum_name} === (not exported)")
            continue
        values = _members(enum, lambda n: True)
        print(f"\n=== {enum_name} ===")
        # RtSupportedFields is long; the rest are short enough to print whole.
        shown = values if enum_name != "RtSupportedFields" else [
            v for v in values if any(k in v for k in ("joint", "tcp", "Pos", "Pose", "Vel", "Torque"))
        ]
        print("  " + ", ".join(shown) if shown else "  (empty)")
        if enum_name == "RtSupportedFields" and len(values) > len(shown):
            print(f"  ... and {len(values) - len(shown)} more")


def try_connect(sdk, ip: str, requested: str = "", local_ip: str = ""):
    """Try each candidate class against a live controller; report which works.

    ``local_ip`` is not optional in practice: the real-time UDP socket binds to
    it, and without it any later ``startReceiveRobotState`` fails with
    "1338 UDP socket打开失败". Plain TCP calls work either way, which is what
    makes the omission easy to miss until the real-time path is touched.
    """
    from atom_robot.ar5.drivers.rokae_arm import _ROBOT_CLASS_CANDIDATES

    candidates = [requested] if requested else [
        n for n in _ROBOT_CLASS_CANDIDATES if hasattr(sdk, n)
    ]
    if not candidates:
        print("No candidate robot classes present in this SDK build.")
        return None

    print(f"\n=== Connecting to {ip} ===")
    for name in candidates:
        cls = getattr(sdk, name, None)
        if cls is None:
            print(f"  {name:20s} not exported by this SDK")
            continue
        try:
            robot = cls(ip, local_ip) if local_ip else cls(ip)
            ec = {}
            robot.connectToRobot(ec)
            # The SDK fills `ec` on success too, as {'ec': 0, 'message': ...}.
            # Only a non-zero code is a failure.
            code = ec.get("ec", 0) if isinstance(ec, dict) else 0
            if code:
                print(f"  {name:20s} FAILED  ec={code} {ec.get('message', '')}")
                continue
            print(f"  {name:20s} CONNECTED  <- set ArmConfig.robot_class to this")
            return robot, name
        except Exception as exc:  # noqa: BLE001
            print(f"  {name:20s} raised {type(exc).__name__}: {exc}")
    return None


def read_state(sdk, robot) -> None:
    """Read joints and TCP pose. Answers the joint-count and convention questions."""
    print("\n=== Reading state (no motion commanded) ===")
    ec = {}
    print(f"  power state    : {robot.powerState(ec)}")
    print(f"  operate mode   : {robot.operateMode(ec)}")
    for label, fn in (("model", "robotInfo"), ("controller", "controllerVersion")):
        getter = getattr(robot, fn, None)
        if getter is not None:
            try:
                print(f"  {label:<15}: {getter(ec)}")
            except Exception as exc:  # noqa: BLE001
                print(f"  {label:<15}: <{type(exc).__name__}>")

    # Both take a timedelta, not a bare number of milliseconds -- the binding
    # rejects an int outright. The subscription period is not free either: the
    # controller refuses anything but its own rate with "该频率不支持", so this
    # matches the 1 ms the driver uses. The update timeout is separate.
    fields = [sdk.RtSupportedFields.jointPos_m, sdk.RtSupportedFields.tcpPose_m]
    robot.startReceiveRobotState(datetime.timedelta(milliseconds=1), fields)
    robot.updateRobotState(datetime.timedelta(milliseconds=2))

    # Ask for more joints than any ROKAE arm has, then see how many come back.
    buf = sdk.PyTypeVectorDouble()
    robot.getStateData(sdk.RtSupportedFields.jointPos_m, buf, 8)
    joints = list(buf.content())
    print(f"\n  joints returned: {len(joints)}")
    print(f"  values         : {[round(j, 4) for j in joints]}")
    if joints:
        print(f"  --> set ArmConfig.num_joints = {len(joints)}")
        print(f"  --> set PROPRIO_DIM = {len(joints) + 1} in prismatic/vla/constants.py")
    else:
        # An unpowered arm streams no joint data. Reading that as "this arm has
        # zero joints" and writing it into the config would be a lot of damage
        # from one probe of a sleeping controller.
        print("  --> no joint data. The arm is almost certainly not powered")
        print("      (check `power state` above). Do NOT set num_joints from this.")

    buf = sdk.PyTypeVectorDouble()
    robot.getStateData(sdk.RtSupportedFields.tcpPose_m, buf, 16)
    pose = list(buf.content())
    print(f"\n  tcpPose length : {len(pose)}")
    if len(pose) == 16:
        rows = [pose[i : i + 4] for i in range(0, 16, 4)]
        print("  as 4x4 (row-major):")
        for row in rows:
            print("    " + "  ".join(f"{v:9.5f}" for v in row))
        print(f"  translation    : {[round(v, 5) for v in (rows[0][3], rows[1][3], rows[2][3])]}")
        print("  --> if the bottom row is [0 0 0 1] and translation looks like metres,")
        print("      the row-major reading in rokae_arm.py is correct.")
    else:
        print(f"  values         : {[round(v, 5) for v in pose]}")
        print("  --> NOT 16 values; rokae_arm._matrix_to_pose needs adjusting.")

    robot.stopReceiveRobotState()


def main() -> int:
    # Default to what the rest of the stack uses, so probing and running cannot
    # disagree about where the SDK is or which arm is being addressed.
    from atom_robot.ar5.config import ArmConfig

    defaults = ArmConfig()

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sdk-path", default=defaults.sdk_path,
                        help="Directory containing the SDK's Release/ folder")
    parser.add_argument("--list", action="store_true", help="List exported classes and enums (no robot needed)")
    parser.add_argument("--connect", action="store_true", help="Try connecting with each candidate class")
    parser.add_argument("--ip", default=defaults.ip, help="Controller IP")
    parser.add_argument("--local-ip", default=defaults.local_ip,
                        help="This machine's IP on the arm's subnet; required for the real-time UDP socket")
    parser.add_argument("--robot-class", default="", help="Only try this class")
    parser.add_argument("--read-state", action="store_true", help="After connecting, read joints and pose")
    args = parser.parse_args()

    if not (args.list or args.connect):
        args.list = True

    try:
        sdk = load_sdk(args.sdk_path)
    except ImportError as exc:
        print(exc)
        return 1

    print(f"SDK module: {getattr(sdk, '__file__', '<builtin>')}")

    if args.list:
        list_symbols(sdk)

    if args.connect:
        result = try_connect(sdk, args.ip, args.robot_class, args.local_ip)
        if result is None:
            print("\nNo class connected. Check the IP, the cabling, and that the controller is up.")
            return 1
        robot, name = result
        if args.read_state:
            try:
                read_state(sdk, robot)
            finally:
                ec = {}
                robot.disconnectFromRobot(ec)
        else:
            ec = {}
            robot.disconnectFromRobot(ec)
        print(f"\nResult: use robot_class = {name!r}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
