"""Offline protocol, threading and SO101 integration regressions. No hardware."""
import threading
import time
import unittest
from unittest.mock import patch

from .config import AR5Config, HKVRConfig
from .drivers import hkvr_protocol as p
from .drivers.hkvr_hand import HKVRHand
from .joint_mapping import JointMapper, JointMapConfig


class Serial:
    def __init__(self, response):
        self.response = response
        self.data = bytearray()
        self.timeout = 0.02
        self.writes = []

    @property
    def in_waiting(self):
        return min(2, len(self.data))  # fragments

    def reset_input_buffer(self):
        self.data.clear()

    def write(self, request):
        self.writes.append(request)
        self.data.extend(self.response(request))

    def read(self, size):
        if not self.data:
            time.sleep(self.timeout)
        value = bytes(self.data[:size])
        del self.data[:size]
        return value


def frame(body):
    return body + p.crc16(body)


class ProtocolTests(unittest.TestCase):
    def test_can_escape_at_checksum_and_fragment_boundary(self):
        # Deliberately make checksum 0x55, right before the unescaped tail.
        encoded = p.build_frame(1, bytes([0x50]), 3)
        buf = bytearray()
        result = []
        for byte in encoded:
            buf.append(byte)
            result.extend(p.parse_frames(buf))
        self.assertEqual(len(result), 1)
        self.assertTrue(result[0][3])

    def test_can_feedback_checks_id_and_exact_dlc(self):
        device = p.HKVRGripper(transport="can", can_id=2, can_rx_id=1, timeout=0.02)
        device._ser = Serial(lambda r: p.build_frame(2, bytes.fromhex("03e8030100"), 5)
                             + p.build_frame(1, bytes.fromhex("03f4010100"), 5))
        self.assertEqual(device.info()[0], 500)
        # USB adapter's local echo of a query must not be mistaken for feedback.
        device._ser = Serial(lambda r: r)
        device.can_rx_id = 2
        self.assertIsNone(device.info())

    def test_can_nonzero_status_not_success(self):
        device = p.HKVRGripper(transport="can", timeout=0.02)
        device._ser = Serial(lambda r: p.build_frame(1, bytes.fromhex("0e0100"), 3))
        with self.assertRaisesRegex(IOError, "未确认成功"):
            device.set_mode(1)

    def test_can_position_ack(self):
        device = p.HKVRGripper(transport="can", timeout=0.02)
        device._ser = Serial(lambda r: p.build_frame(1, bytes.fromhex("100000"), 3))
        self.assertTrue(device.move(500))

    def device(self, response):
        device = p.HKVRGripper(timeout=0.02)
        device._ser = Serial(response)
        return device

    def test_manual_force_position_bytes(self):
        self.assertEqual(p.mb_write_n(1, p.REG_FORCE_POS, (1000, 1000)).hex(),
                         "0110000f00020403e803e83321")

    def test_exception_is_failure(self):
        device = self.device(lambda r: frame(bytes([1, r[1] | 0x80, 2])))
        for call in (lambda: device.move(500), lambda: device.set_mode(1),
                     lambda: device.set_force(20), device.init_fingers):
            with self.assertRaisesRegex(IOError, "异常"):
                call()

    def test_ack_must_match_written_register_and_value(self):
        device = self.device(lambda r: p.mb_write1(1, p.REG_FORCE, 30))
        self.assertFalse(device.set_mode(1))

    def test_fragmented_noisy_response_returns_without_full_timeout(self):
        device = self.device(lambda r: b"\x99\x88" + frame(bytes.fromhex("0103020032")))
        start = time.monotonic()
        self.assertEqual(device.target_force(), 50)
        self.assertLess(time.monotonic() - start, 0.02)

    def test_bad_crc_is_rejected(self):
        device = self.device(lambda r: bytes.fromhex("01030200320000"))
        self.assertIsNone(device.target_force())

    def test_wrong_register_count_is_rejected(self):
        device = self.device(lambda r: frame(bytes.fromhex("01030400320033")))
        self.assertIsNone(device.target_force())

    def test_single_atomic_speed_position_write(self):
        device = self.device(lambda r: frame(r[:6]))
        self.assertTrue(device.set_opening(0.5))
        # 速度和位置必须一次写完（功能码 0x10 两个寄存器），分两次写夹爪不认
        self.assertEqual(device._ser.writes,
                         [p.mb_write_n(1, 15, (1000, round(0.5 * p.POS_CLOSED)))])

    def test_opening_maps_to_measured_travel_and_direction(self):
        """opening 与原始位置是**反的**：opening 1.0（全开）= 位置 0，opening 0.0（合拢）= POS_CLOSED。

        2026-09-22 实测指令 0 → 位置 0，实物是**张开**。合拢端当时量到 858，9-24 查明那天夹爪卡住了，
        解卡后闭合端停在 907~1010，所以 POS_CLOSED 改回手册量程 1000（见 hkvr_protocol.py 的注释）。
        """
        for opening, want_pos in ((1.0, 0), (0.0, p.POS_CLOSED), (0.5, round(0.5 * p.POS_CLOSED))):
            device = self.device(lambda r: frame(r[:6]))
            self.assertTrue(device.set_opening(opening))
            self.assertEqual(device._ser.writes, [p.mb_write_n(1, 15, (1000, want_pos))],
                             f"opening={opening}")

    def test_get_opening_inverts_position(self):
        for pos, want in ((0, 1.0), (p.POS_CLOSED, 0.0)):
            device = self.device(lambda r: frame(
                bytes([1, 3, 2]) + pos.to_bytes(2, "big")))
            self.assertAlmostEqual(device.get_opening(), want, places=3)

    def test_nonfinite_target_does_not_write(self):
        device = self.device(lambda r: frame(r[:6]))
        with self.assertRaises(ValueError):
            device.set_opening(float("nan"))
        self.assertEqual(device._ser.writes, [])


class FakeDevice:
    def __init__(self, **kwargs):
        self.moves = []
        self.block = False
        self.entered = threading.Event()
        self.release = threading.Event()
        self.fail = False
        self.closed = False

    def connect(self): return 20
    def close(self): self.closed = True
    def set_mode(self, mode): return True
    def set_force(self, force): return True
    def move(self, position):
        self.moves.append(position)
        return not self.fail
    def info(self):
        if self.block:
            self.entered.set()
            self.release.wait(1.0)
        if self.fail:
            raise IOError("cable removed")
        return (250, 1, "到位")


class IntegrationTests(unittest.TestCase):
    def wait_for(self, predicate):
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.005)
        self.fail("worker did not reach expected state")

    def test_latest_only_nonblocking_and_measured_feedback(self):
        with patch("atom_robot.ar5.drivers.hkvr_hand.HKVRGripper", FakeDevice):
            hand = HKVRHand(HKVRConfig())
            hand.connect()
        device = hand._device
        try:
            device.block = True
            self.assertTrue(device.entered.wait(1))
            start = time.monotonic()
            for value in (0.1, 0.2, 0.9):
                hand.set_opening(value)
            self.assertLess(time.monotonic() - start, 0.02)
            self.assertEqual(hand.get_opening(), 0.25)  # actual, not last target
            device.block = False
            device.release.set()
            self.wait_for(lambda: bool(device.moves))
            self.assertEqual(device.moves, [900])
            device.fail = True
            self.wait_for(lambda: hand._error is not None)
            with self.assertRaisesRegex(RuntimeError, "通信故障"):
                hand.set_opening(1)
        finally:
            device.release.set()
            hand.disconnect()
        self.assertTrue(device.closed)

    def test_so101_trigger_to_hkvr_counts(self):
        mapper = JointMapper(JointMapConfig(gripper_closed_deg=-55.5, gripper_open_deg=55.5))
        for angle, counts in ((-55.5, 0), (0, 500), (55.5, 1000)):
            self.assertEqual(round(mapper.gripper_command({"gripper": angle}) * 1000), counts)

    def test_mock_selection_never_opens_real_serial(self):
        from .ar5_env import AR5Env
        cfg = AR5Config(mock=True, end_effector="hkvr")
        cfg.cameras.sources = {}
        with patch("atom_robot.ar5.drivers.hkvr_hand.HKVRGripper") as device:
            with AR5Env(cfg) as env:
                env.hand.set_opening(0.5)
                self.assertEqual(env.hand.get_opening(), 0.5)
            device.assert_not_called()

    def test_gripper_failure_stops_arm_even_without_observation(self):
        from .ar5_env import AR5Env
        cfg = AR5Config(mock=True)
        cfg.cameras.sources = {}
        with AR5Env(cfg) as env:
            env.hand.check_health = lambda: (_ for _ in ()).throw(RuntimeError("lost gripper"))
            with patch.object(env.arm, "stop") as stop:
                with self.assertRaisesRegex(RuntimeError, "lost gripper"):
                    env.step_joints(env.arm.get_joint_positions(), observe=False)
                stop.assert_called_once()

    def test_stale_feedback_is_never_a_valid_state(self):
        hand = HKVRHand(HKVRConfig())
        hand._device = FakeDevice()
        hand._measured = 0.5
        hand._measured_at = time.monotonic() - 2
        with self.assertRaisesRegex(RuntimeError, "过期"):
            hand.get_opening()


if __name__ == "__main__":
    unittest.main()
