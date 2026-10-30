"""Compare Robotell standard adapter initialization, then HKVR read-only commands.

Only adapter initialization/config restoration may write. Never initialize gripper,
change its mode, set a target, or enable EEPROM. Keeps every incoming frame.
"""
import argparse
import json
import time
from pathlib import Path

from .probe_hkvr_can import encode, decode


CONFIGS = [("uart", 0x01fffe90, 4), ("retransmit", 0x01fffea0, 1),
           ("bus_management", 0x01fffeb0, 1), ("can_rate", 0x01fffed0, 4)]
CONFIGS += [(f"filter_{i}", 0x01fffee0 + i, 8) for i in range(14)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="/dev/ttyUSB0")
    parser.add_argument("--out", default="/tmp/hkvr-can-init-probe.json")
    args = parser.parse_args()
    import serial
    log = dict(exchanges=[], timestamp=time.strftime("%Y-%m-%d %H:%M:%S"))
    buf = bytearray()
    def exchange(ser, label, packet, duration=0.2):
        entry = dict(label=label, tx=packet.hex(), rx="", frames=[])
        log['exchanges'].append(entry)
        ser.write(packet)
        end = time.monotonic() + duration
        raw = bytearray()
        while time.monotonic() < end:
            ser.timeout = min(0.02, max(0, end - time.monotonic()))
            chunk = ser.read(max(1, ser.in_waiting))
            raw.extend(chunk)
            buf.extend(chunk)
            entry['frames'].extend(decode(buf))
        entry['rx'] = raw.hex()
        print(label, f"raw={len(raw)}", entry['frames'], flush=True)
        return entry['frames']
    def read_config(ser, name, cid, size):
        frames = exchange(ser, f"config_{name}", encode(cid, dlc=size, channel=255, remote=True))
        for frame in frames:
            if frame['id'] == cid and frame['channel'] == 255 and not frame['remote'] and frame['dlc'] >= size:
                return bytes.fromhex(frame['data'])[:size]
        raise IOError(f"Cannot read configuration {name}")
    snapshot = {}
    try:
        with serial.Serial(args.port, 115200, timeout=0.02, write_timeout=0.2, exclusive=True) as ser:
            for name, cid, size in CONFIGS:
                snapshot[cid] = read_config(ser, name, cid, size)
            log['before'] = {hex(k): v.hex() for k, v in snapshot.items()}
            try:
                # Matches python-can robotellBus.__init__ / _writeconfig(RESET_ID, 0).
                exchange(ser, 'adapter_initialize_not_gripper',
                         encode(0x01fffec0, bytes(8), channel=255), duration=0.5)
                after = {cid: read_config(ser, name, cid, size) for name, cid, size in CONFIGS}
                log['after_init'] = {hex(k): v.hex() for k, v in after.items()}
                if snapshot != after:
                    raise RuntimeError("Adapter initialization changed configuration; restoring before continuing")
                exchange(ser, 'adapter_firmware', encode(0x01ffffe0, dlc=8, channel=255, remote=True))
                for extended in (False, True):
                    for cid in (1, 2, 0):
                        for name, payload in (("position", "0300000101000000"),
                                              ("target_force", "070100"),
                                              ("serial_number", "ee0100")):
                            exchange(ser, f"HKVR_{name}_id{cid}_ext{int(extended)}",
                                     encode(cid, bytes.fromhex(payload), extended=extended), duration=0.7)
            finally:
                failures = []
                for name, cid, size in CONFIGS:
                    try:
                        actual = read_config(ser, name, cid, size)
                        if actual != snapshot[cid]:
                            exchange(ser, f"RESTORE_{name}", encode(cid, snapshot[cid], channel=255))
                            if read_config(ser, name, cid, size) != snapshot[cid]:
                                raise RuntimeError("restore readback mismatch")
                    except Exception as exc:
                        failures.append(f"{name}: {exc}")
                log['restore_failures'] = failures
                if failures:
                    raise RuntimeError(f"Configuration restoration needs attention: {failures}")
                log['all_original_configuration_verified'] = True
    finally:
        Path(args.out).write_text(json.dumps(log, indent=2) + "\n")
        print(args.out)


if __name__ == "__main__":
    main()
