"""Robotell configuration and HKVR CAN diagnostics; no gripper initialization or motion.

Protocol reference: python-can's can.interfaces.robotell implementation.
Config reads: channel FF, extended REMOTE frame. Bus queries: channel 0 DATA.
Default reads only. --bitrates temporarily changes adapter bitrate, verifies and restores it.
Retain all receive IDs and raw bytes in JSON. Never change gripper configuration.
"""
import argparse
import json
import time
from pathlib import Path

from .drivers.hkvr_protocol import _escape


def encode(can_id, data=b"", dlc=None, channel=0, extended=True, remote=False):
    if dlc is None:
        dlc = len(data)
    if not 0 <= dlc <= 8:
        raise ValueError("DLC outside 0..8")
    payload = bytes(8) if remote else data.ljust(8, b"\x00")
    body = can_id.to_bytes(4, "little") + payload + bytes((dlc, channel, int(extended), int(remote)))
    return b"\xaa\xaa" + _escape(body + bytes((sum(body) & 255,))) + b"\x55\x55"


def decode(buf):
    frames = []
    while True:
        start = buf.find(b"\xaa\xaa")
        if start < 0:
            del buf[:-1]
            return frames
        del buf[:start]
        body = bytearray()
        cursor = 2
        while len(body) < 17:
            if cursor >= len(buf):
                return frames
            if buf[cursor] == 0xa5:
                cursor += 1
                if cursor >= len(buf):
                    return frames
            body.append(buf[cursor])
            cursor += 1
        if len(buf) < cursor + 2:
            return frames
        if buf[cursor:cursor + 2] != b"\x55\x55" or sum(body[:16]) & 255 != body[16] or body[12] > 8:
            del buf[0]
            continue
        del buf[:cursor + 2]
        frames.append(dict(id=int.from_bytes(body[:4], "little"), dlc=body[12],
                           data=bytes(body[4:4 + body[12]]).hex(), channel=body[13],
                           extended=bool(body[14]), remote=bool(body[15])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="/dev/ttyUSB0")
    parser.add_argument("--uart-baud", type=int, default=115200)
    parser.add_argument("--ids", default="1,2", help="comma-separated IDs or inclusive start:end; receive every ID")
    parser.add_argument("--info-only", action="store_true", help="Only send HKVR read-info command during ID scan")
    parser.add_argument("--quiet", action="store_true", help="Print responses and progress only")
    parser.add_argument("--timeout", type=float, default=0.25)
    parser.add_argument("--out", default="/tmp/hkvr-can-probe.json")
    parser.add_argument("--bitrates", default="", help="Optional adapter bitrate sweep; restore verified original rate on exit")
    args = parser.parse_args()
    if ":" in args.ids:
        start, stop = (int(v, 0) for v in args.ids.split(":"))
        if not 0 <= start <= stop <= 0x7ff:
            parser.error("ID range must be within 0..0x7ff")
        ids = range(start, stop + 1)
    else:
        ids = [int(v, 0) for v in args.ids.split(",")]
    if any(not 0 <= v <= 0x7ff for v in ids) or not 0 < args.timeout <= 2:
        parser.error("IDs must be 0..0x7ff, timeout (0,2]")
    rates = [int(v) for v in args.bitrates.split(",")] if args.bitrates else []
    if any(v not in (1000000, 800000, 500000, 400000, 250000, 125000, 100000, 50000) for v in rates):
        parser.error("unsupported diagnostic bitrate")
    import serial
    report = dict(port=args.port, uart_baud=args.uart_baud, timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                  source="https://python-can.readthedocs.io/en/main/_modules/can/interfaces/robotell.html",
                  exchanges=[])
    buf = bytearray()
    def exchange(ser, name, packet=None, seconds=None):
        entry = dict(name=name, tx=packet.hex() if packet else "", rx="", frames=[])
        report["exchanges"].append(entry)
        if packet:
            ser.write(packet)
        deadline = time.monotonic() + (seconds if seconds is not None else args.timeout)
        raw = bytearray()
        while time.monotonic() < deadline:
            ser.timeout = min(0.02, max(0, deadline - time.monotonic()))
            chunk = ser.read(max(1, ser.in_waiting))
            raw.extend(chunk)
            buf.extend(chunk)
            entry["frames"].extend(decode(buf))
        entry["rx"] = raw.hex()
        if not args.quiet or raw or len(report['exchanges']) % 128 == 0:
            print(f"{name}: raw={len(raw)} bytes frames={entry['frames']} "
                  f"[{len(report['exchanges'])} queries]", flush=True)
        return entry
    def get_rate(ser):
        entry = exchange(ser, "verify_can_bitrate",
                         encode(0x01fffed0, dlc=4, channel=255, remote=True), seconds=0.3)
        found = [f for f in entry['frames'] if f['channel'] == 255 and f['id'] == 0x01fffed0
                 and not f['remote'] and f['dlc'] >= 4]
        if not found:
            raise RuntimeError("Cannot read adapter bitrate; will not change unknown configuration")
        return int.from_bytes(bytes.fromhex(found[-1]['data'])[:4], 'little')
    def set_rate(ser, rate):
        exchange(ser, f"SET_ADAPTER_BITRATE_{rate}",
                 encode(0x01fffed0, rate.to_bytes(4, 'little'), channel=255), seconds=0.15)
        actual = get_rate(ser)
        if actual != rate:
            raise RuntimeError(f"Adapter bitrate readback {actual} != requested {rate}")
    try:
        with serial.Serial(args.port, args.uart_baud, timeout=0.02, write_timeout=0.2, exclusive=True) as ser:
            exchange(ser, "passive", seconds=0.5)
            configs = [("can_bitrate", 0x01fffed0, 4), ("uart_bitrate", 0x01fffe90, 4),
                       ("auto_retransmit", 0x01fffea0, 1), ("auto_bus_management", 0x01fffeb0, 1),
                       ("serial_1", 0x01fffff0, 8), ("serial_2", 0x01fffff1, 8)]
            configs += [(f"filter_{i}", 0x01fffee0 + i, 8) for i in range(14)]
            for name, cid, dlc in configs:
                exchange(ser, name, encode(cid, dlc=dlc, channel=255, extended=True, remote=True))
            queries = [("info", bytes.fromhex("0300000101000000"))]
            if not args.info_only:
                queries.append(("force", bytes.fromhex("070100")))
            original = get_rate(ser) if rates else None
            try:
                for rate in rates or [None]:
                    if rate is not None:
                        set_rate(ser, rate)
                    for extended in (False, True):
                        for cid in ids:
                            for name, data in queries:
                                exchange(ser, f"{name} bitrate={rate} tx=0x{cid:x} {'extended' if extended else 'standard'}",
                                         encode(cid, data, extended=extended))
            finally:
                if original is not None:
                    set_rate(ser, original)
                    report['restored_bitrate'] = original
    finally:
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
        print(f"原始日志：{args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
