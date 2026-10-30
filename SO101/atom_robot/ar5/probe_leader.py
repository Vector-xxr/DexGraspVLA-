"""Find the SO leader arm on the serial bus and read its joints.

The SO-100/SO-101 leader is six Feetech STS3215 servos at IDs 1-6 on one
half-duplex serial bus. Which /dev node it lands on depends on enumeration
order, and the same USB-serial chips are used for other devices, so the only
reliable identification is to ping the bus and see which IDs answer.

    python -m atom_robot.ar5.probe_leader              # scan every port
    python -m atom_robot.ar5.probe_leader --watch      # then stream joints

Read-only: it pings and reads positions. Torque is never enabled, so the arm
stays back-drivable and safe to hold.
"""

import argparse
import glob
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from atom_robot.ar5.leader_bus import (  # noqa: E402
    SO_JOINTS,
    STS3215_BAUDRATES,
    LeaderBus,
)


def scan_port(port: str, baudrates=STS3215_BAUDRATES):
    """Return (baudrate, [ids]) for the first baudrate where anything answers."""
    for baud in baudrates:
        try:
            bus = LeaderBus(port, baud)
            bus.connect()
        except Exception as exc:  # noqa: BLE001
            print(f"  {port} @ {baud:>7}  cannot open: {exc}")
            return None
        try:
            found = bus.scan_ids(range(1, 13))
        finally:
            bus.disconnect()
        if found:
            return baud, found
        print(f"  {port} @ {baud:>7}  no reply")
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", default="", help="Probe only this port")
    parser.add_argument("--baudrate", type=int, default=0, help="Probe only this baudrate")
    parser.add_argument("--watch", action="store_true", help="Stream joint positions after finding the arm")
    parser.add_argument("--hz", type=float, default=10.0)
    args = parser.parse_args()

    ports = [args.port] if args.port else sorted(glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*"))
    if not ports:
        print("No /dev/ttyACM* or /dev/ttyUSB* devices found. Is the arm plugged in?")
        return 1

    bauds = [args.baudrate] if args.baudrate else STS3215_BAUDRATES

    print(f"Scanning {len(ports)} port(s): {', '.join(ports)}")
    hits = []
    for port in ports:
        result = scan_port(port, bauds)
        if result:
            baud, ids = result
            print(f"  {port} @ {baud:>7}  IDs {ids}")
            hits.append((port, baud, ids))

    if not hits:
        print("\nNothing answered. Check the cable, the arm's power, and that no other\n"
              "process holds the port.")
        return 1

    print("\n=== found ===")
    leader = None
    for port, baud, ids in hits:
        expected = list(range(1, len(SO_JOINTS) + 1))
        if ids == expected:
            print(f"  {port}  SO leader: all {len(SO_JOINTS)} servos present")
            leader = (port, baud)
        else:
            missing = [i for i in expected if i not in ids]
            extra = [i for i in ids if i not in expected]
            print(f"  {port}  {len(ids)} servos, IDs {ids}"
                  + (f", missing {missing}" if missing else "")
                  + (f", unexpected {extra}" if extra else ""))
            if not leader and len(ids) >= 2:
                leader = (port, baud)

    if leader:
        print(f"\n  --> leader_port = {leader[0]!r}   baudrate = {leader[1]}")

    if args.watch and leader:
        port, baud = leader
        bus = LeaderBus(port, baud)
        bus.connect()
        print("\nMove the leader arm. Ctrl-C to stop.\n")
        try:
            header = "  " + "".join(f"{name:>14}" for name in SO_JOINTS)
            print(header)
            while True:
                pos = bus.read_positions()
                print("  " + "".join(f"{pos.get(n, float('nan')):>14.1f}" for n in SO_JOINTS), end="\r")
                time.sleep(1.0 / args.hz)
        except KeyboardInterrupt:
            print("\n\nstopped")
        finally:
            bus.disconnect()

    return 0


if __name__ == "__main__":
    sys.exit(main())
