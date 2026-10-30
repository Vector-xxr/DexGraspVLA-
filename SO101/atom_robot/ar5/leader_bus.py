"""Minimal Feetech STS3215 bus reader for an SO-100/SO-101 leader arm.

lerobot's own FeetechMotorsBus would do this, but installing lerobot pulls a
whole training stack including its own pinned torch, which would disturb the
CUDA build this environment already has working. Reading six servo positions
needs none of that -- just the vendor SDK and the register map.

Read-only by design: torque is never enabled, so the leader stays
back-drivable and safe to hold while it is being read.
"""

import glob
import json
import logging
import os
from typing import Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

# SO-100 / SO-101 leader joints, in servo ID order (IDs are 1..6).
SO_JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

# STS3215 control table. Present_Position is a 2-byte little-endian register.
ADDR_TORQUE_ENABLE = 40
ADDR_PRESENT_POSITION = 56
ADDR_PRESENT_SPEED = 58
ADDR_PRESENT_LOAD = 60

# Raw encoder counts span one turn; lerobot treats 2048 as the centre.
STS3215_RESOLUTION = 4096
STS3215_CENTRE = 2048

# lerobot configures these servos at 1 Mbps. The slower rates are here because a
# factory-fresh or previously reconfigured servo may still be on one of them.
STS3215_BAUDRATES = (1_000_000, 500_000, 250_000, 115_200, 57_600)


LEROBOT_CALIB_DIR = os.path.expanduser(
    "~/.cache/huggingface/lerobot/calibration/teleoperators/so_leader"
)


def load_lerobot_calibration(calib_id: str = "") -> Dict[str, dict]:
    """Load a calibration written by ``lerobot-calibrate``.

    With no id, and exactly one calibration on disk, that one is used. With
    several, the id must be given -- silently picking one would apply another
    arm's joint ranges to this one.
    """
    paths = sorted(glob.glob(os.path.join(LEROBOT_CALIB_DIR, "*.json")))
    if not paths:
        raise FileNotFoundError(
            f"No calibration in {LEROBOT_CALIB_DIR}. Record one with:\n"
            "  lerobot-calibrate --teleop.type=so101_leader "
            "--teleop.port=/dev/ttyACM0 --teleop.id=<name>"
        )

    if calib_id:
        match = os.path.join(LEROBOT_CALIB_DIR, f"{calib_id}.json")
        if match not in paths:
            available = [os.path.basename(p)[:-5] for p in paths]
            raise FileNotFoundError(f"No calibration {calib_id!r}. Available: {available}")
        path = match
    elif len(paths) == 1:
        path = paths[0]
    else:
        available = [os.path.basename(p)[:-5] for p in paths]
        raise ValueError(
            f"{len(paths)} calibrations found; pass calib_id to choose. Available: {available}"
        )

    with open(path) as handle:
        calibration = json.load(handle)
    logger.info(f"[LeaderBus] using calibration {os.path.basename(path)}")
    return calibration


class LeaderBus:
    """Half-duplex Feetech serial bus, read side only."""

    def __init__(
        self,
        port: str,
        baudrate: int = 1_000_000,
        joints: Iterable[str] = SO_JOINTS,
        calibration: Optional[Dict[str, dict]] = None,
    ):
        self.port = port
        self.baudrate = baudrate
        self.joints = tuple(joints)
        # Servo IDs are 1-based and follow the joint order.
        self.ids = {name: i + 1 for i, name in enumerate(self.joints)}
        self.calibration = calibration or {}
        self._port_handler = None
        self._packet_handler = None

    @classmethod
    def with_calibration(cls, port: str, calib_id: str = "", **kwargs) -> "LeaderBus":
        """Build a bus using a calibration recorded by ``lerobot-calibrate``.

        Without it, joint zero is assumed to be the servo's own midpoint (2048),
        which is only true if every horn happened to be assembled there. The
        calibration records each joint's real travel, and its midpoint is the
        zero the URDF expects -- so skipping it biases forward kinematics by
        however far the assembly is off, which can be tens of degrees.
        """
        return cls(port, calibration=load_lerobot_calibration(calib_id), **kwargs)

    # ------------------------------------------------------------------ setup

    def connect(self) -> None:
        import scservo_sdk as scs

        self._port_handler = scs.PortHandler(self.port)
        self._packet_handler = scs.PacketHandler(0)  # protocol 0 = STS/SMS series

        if not self._port_handler.openPort():
            raise IOError(f"Could not open {self.port}")
        if not self._port_handler.setBaudRate(self.baudrate):
            self._port_handler.closePort()
            raise IOError(f"Could not set {self.port} to {self.baudrate} baud")

    def disconnect(self) -> None:
        if self._port_handler is not None:
            self._port_handler.closePort()
            self._port_handler = None
            self._packet_handler = None

    def _require(self):
        if self._port_handler is None:
            raise RuntimeError("LeaderBus used before connect()")

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *exc_info):
        self.disconnect()

    # ----------------------------------------------------------------- probing

    def scan_ids(self, candidates: Iterable[int] = range(1, 13)) -> List[int]:
        """Ping each candidate ID and return those that answer."""
        self._require()
        found = []
        for servo_id in candidates:
            _, comm, error = self._packet_handler.ping(self._port_handler, servo_id)
            if comm == 0 and error == 0:
                found.append(servo_id)
        return found

    # ----------------------------------------------------------------- reading

    def read_raw(self, name: str) -> Optional[int]:
        """Raw encoder counts for one joint, or None if it did not answer."""
        self._require()
        value, comm, error = self._packet_handler.read2ByteTxRx(
            self._port_handler, self.ids[name], ADDR_PRESENT_POSITION
        )
        if comm != 0 or error != 0:
            return None
        return value

    def read_positions(self) -> Dict[str, float]:
        """Every joint in degrees, centred so the servo midpoint reads 0.

        A joint that does not answer is omitted rather than reported as zero --
        a dropped reply must not look like a commanded position.
        """
        out = {}
        for name in self.joints:
            raw = self.read_raw(name)
            if raw is not None:
                out[name] = self.raw_to_degrees(raw, name)
        return out

    def read_raw_positions(self) -> Dict[str, int]:
        out = {}
        for name in self.joints:
            raw = self.read_raw(name)
            if raw is not None:
                out[name] = raw
        return out

    # -------------------------------------------------------------- conversion

    def centre_for(self, name: str) -> float:
        """The raw count that means zero degrees for this joint."""
        entry = self.calibration.get(name)
        if entry is None:
            return float(STS3215_CENTRE)
        # lerobot's DEGREES mode centres on the midpoint of the recorded travel.
        return (entry["range_min"] + entry["range_max"]) / 2.0

    def half_travel_deg(self, name: str) -> float:
        """How far this joint swings either side of zero, in degrees.

        Readings are centred on the midpoint of the recorded travel, so the
        reachable span is symmetric about zero by construction. Joint-space
        mapping needs this to scale the leader's travel into the follower's
        without ever running off the end.
        """
        entry = self.calibration.get(name)
        if entry is None:
            return 180.0
        span = float(entry["range_max"] - entry["range_min"])
        return span * 360.0 / (STS3215_RESOLUTION - 1) / 2.0

    def raw_to_degrees(self, raw: int, name: str = "") -> float:
        # Servos report Present_Position already reduced by their stored
        # Homing_Offset, so only the centre has to be applied here.
        return (raw - self.centre_for(name)) * 360.0 / (STS3215_RESOLUTION - 1)

    def degrees_to_raw(self, deg: float, name: str = "") -> int:
        return int(round(deg * (STS3215_RESOLUTION - 1) / 360.0 + self.centre_for(name)))

    # ----------------------------------------------------------------- safety

    def assert_torque_disabled(self) -> None:
        """Confirm no servo is holding torque, so the arm can be moved by hand."""
        self._require()
        for name, servo_id in self.ids.items():
            value, comm, error = self._packet_handler.read1ByteTxRx(
                self._port_handler, servo_id, ADDR_TORQUE_ENABLE
            )
            if comm == 0 and error == 0 and value:
                logger.warning(f"[LeaderBus] {name} (id {servo_id}) has torque enabled")
