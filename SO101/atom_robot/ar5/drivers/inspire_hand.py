"""Inspire RH56 dexterous hand over Modbus RTU (RS485).

The Modbus framing is implemented directly on pyserial rather than pulled from
a Modbus library: the hand only needs two function codes, and hand-rolling them
keeps the dependency surface at pyserial and makes the CRC unit-testable
without hardware.

Register addresses come from :class:`HandConfig` -- check them against the
manual for your specific RH56 variant before the first powered run.
"""

import logging
import struct
from typing import List, Optional

import numpy as np

from .base import HandDriver

logger = logging.getLogger(__name__)

_READ_HOLDING_REGISTERS = 0x03
_WRITE_MULTIPLE_REGISTERS = 0x10


def modbus_crc16(payload: bytes) -> int:
    """Standard Modbus RTU CRC-16, returned in transmission (little-endian) order."""
    crc = 0xFFFF
    for byte in payload:
        crc ^= byte
        for _ in range(8):
            if crc & 0x0001:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1
    return crc


def build_write_registers_frame(slave_id: int, address: int, values: List[int]) -> bytes:
    """Function 0x10 -- preset multiple holding registers."""
    body = struct.pack(">BBHHB", slave_id, _WRITE_MULTIPLE_REGISTERS, address, len(values), len(values) * 2)
    body += b"".join(struct.pack(">H", v & 0xFFFF) for v in values)
    return body + struct.pack("<H", modbus_crc16(body))


def build_read_registers_frame(slave_id: int, address: int, count: int) -> bytes:
    """Function 0x03 -- read holding registers."""
    body = struct.pack(">BBHH", slave_id, _READ_HOLDING_REGISTERS, address, count)
    return body + struct.pack("<H", modbus_crc16(body))


def parse_read_response(frame: bytes, expected_slave: int, expected_count: int) -> List[int]:
    """Validate a 0x03 response and return the register values."""
    if len(frame) < 5:
        raise IOError(f"Modbus response too short: {frame!r}")

    if struct.unpack("<H", frame[-2:])[0] != modbus_crc16(frame[:-2]):
        raise IOError(f"Modbus CRC mismatch: {frame!r}")

    slave, function = frame[0], frame[1]
    if slave != expected_slave:
        raise IOError(f"Modbus response from slave {slave}, expected {expected_slave}")
    if function & 0x80:
        raise IOError(f"Modbus exception code {frame[2]} for function {function & 0x7F:#04x}")
    if function != _READ_HOLDING_REGISTERS:
        raise IOError(f"Unexpected Modbus function {function:#04x} in response")

    byte_count = frame[2]
    if byte_count != expected_count * 2:
        raise IOError(f"Modbus returned {byte_count} bytes, expected {expected_count * 2}")

    data = frame[3 : 3 + byte_count]
    return [struct.unpack(">H", data[i : i + 2])[0] for i in range(0, byte_count, 2)]


class InspireHand(HandDriver):
    """RH56-series hand exposed as a single opening scalar in [0, 1]."""

    def __init__(self, config):
        self.config = config
        self._serial: Optional[object] = None
        self._last_opening = 1.0

    # ------------------------------------------------------------------ setup

    def connect(self) -> None:
        try:
            import serial  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "pyserial is required for the Inspire hand driver: pip install pyserial"
            ) from exc

        try:
            self._serial = serial.Serial(
                port=self.config.port,
                baudrate=self.config.baudrate,
                bytesize=8,
                parity="N",
                stopbits=1,
                timeout=self.config.timeout,
            )
        except serial.SerialException as exc:
            # The port not existing at all is worth its own message: it means
            # the hand is unplugged or on a different tty, not that the link is
            # misbehaving, and the fix is a flag rather than debugging.
            from pathlib import Path  # noqa: PLC0415

            hint = (
                f"{self.config.port} does not exist. The hand is unplugged, or it "
                "enumerated elsewhere -- check `ls /dev/ttyUSB* /dev/ttyACM*`. To run "
                "the arm without it, pass --no-hand (or set config.use_hand = False)."
                if not Path(self.config.port).exists()
                else f"could not open {self.config.port}: {exc}. Another process may hold "
                     "it, or the user may not be in the dialout group."
            )
            raise RuntimeError(f"[InspireHand] {hint}") from exc

        # Prove the link before anything moves.
        angles = self.get_joint_angles()
        logger.info(f"[InspireHand] connected on {self.config.port}, angles={angles.tolist()}")

    def disconnect(self) -> None:
        if self._serial is not None:
            self._serial.close()
            self._serial = None

    def _require_connection(self):
        if self._serial is None:
            raise RuntimeError("Hand driver used before connect()")
        return self._serial

    # ------------------------------------------------------------- transport

    def _write_registers(self, address: int, values: List[int]) -> None:
        ser = self._require_connection()
        ser.reset_input_buffer()
        ser.write(build_write_registers_frame(self.config.slave_id, address, values))
        # An 0x10 acknowledgement is a fixed 8 bytes. A short read means the
        # hand did not accept the frame, which we surface rather than ignore.
        ack = ser.read(8)
        if len(ack) != 8:
            raise IOError(f"No acknowledgement for register write at {address:#06x} (got {len(ack)} bytes)")

    def _read_registers(self, address: int, count: int) -> List[int]:
        ser = self._require_connection()
        ser.reset_input_buffer()
        ser.write(build_read_registers_frame(self.config.slave_id, address, count))
        # slave + function + byte-count + payload + CRC
        frame = ser.read(5 + count * 2)
        return parse_read_response(frame, self.config.slave_id, count)

    # ----------------------------------------------------------------- driver

    def set_opening(self, opening: float) -> None:
        opening = float(np.clip(opening, 0.0, 1.0))
        closed = np.array(self.config.closed_pose, dtype=np.float64)
        opened = np.array(self.config.open_pose, dtype=np.float64)

        targets = closed + opening * (opened - closed)
        targets = np.clip(targets, self.config.angle_min, self.config.angle_max)

        self._write_registers(self.config.angle_set_register, [int(round(v)) for v in targets])
        self._last_opening = opening

    def get_joint_angles(self) -> np.ndarray:
        values = self._read_registers(self.config.angle_act_register, self.config.num_joints)
        return np.array(values, dtype=np.float64)

    # ------------------------------------------------------------ diagnostics

    def get_errors(self) -> np.ndarray:
        """Per-DOF error codes. Non-zero means that actuator is faulted."""
        # ERROR(m) is 6 bytes = 3 registers; unpack back to per-DOF bytes.
        words = self._read_registers(self.config.error_register, 3)
        out = []
        for w in words:
            out.extend([(w >> 8) & 0xFF, w & 0xFF])
        return np.array(out[: self.config.num_joints], dtype=np.int32)

    def clear_errors(self) -> None:
        """Clear latched actuator faults.

        A stalled, over-current or comms-faulted actuator stays locked until
        cleared, which on a real hand looks exactly like "the finger stopped
        working". Over-temperature is the exception: it clears itself once the
        actuator cools.
        """
        self._write_registers(self.config.clear_error_register, [1])
        logger.info("[InspireHand] cleared latched actuator errors")

    def get_forces(self) -> np.ndarray:
        """Per-DOF measured force. Only meaningful on force-sensing variants."""
        return np.array(
            self._read_registers(self.config.force_act_register, self.config.num_joints),
            dtype=np.float64,
        )

    def set_speed_limit(self, speed: int) -> None:
        """Cap the speed of every DOF. Worth lowering during bring-up."""
        value = int(np.clip(speed, 0, 1000))
        self._write_registers(self.config.speed_set_register, [value] * self.config.num_joints)

    def set_force_limit(self, force: int) -> None:
        """Set the force-control threshold for every DOF (range 0-3000)."""
        value = int(np.clip(force, 0, self.config.force_max))
        self._write_registers(self.config.force_set_register, [value] * self.config.num_joints)

    # ----------------------------------------------------------------- driver

    def get_opening(self) -> float:
        """Project the measured joint angles back onto the closed->open segment.

        Uses a least-squares projection rather than one reference finger, so a
        single stalled actuator (a finger blocked by the grasped object) does
        not dominate the estimate.
        """
        try:
            angles = self.get_joint_angles()
        except IOError:
            # Readback is best-effort: fall back to the last command so a
            # transient bus error cannot stall the control loop.
            return self._last_opening

        closed = np.array(self.config.closed_pose, dtype=np.float64)
        opened = np.array(self.config.open_pose, dtype=np.float64)
        span = opened - closed
        denom = float(span @ span)
        if denom < 1e-9:
            return self._last_opening
        return float(np.clip(((angles - closed) @ span) / denom, 0.0, 1.0))
