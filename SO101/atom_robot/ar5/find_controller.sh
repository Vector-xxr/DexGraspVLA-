#!/usr/bin/env bash
# Locate a ROKAE controller on a directly-connected Ethernet link.
#
# The controller's subnet is not knowable in advance: ROKAE documentation gives
# 192.168.2.160 as the factory default, while the SDK examples use
# 192.168.0.160. A link that is up but silent usually means the PC and the
# controller are simply on different subnets, and ARP never crosses.
#
# This adds a temporary secondary address in each candidate subnet, sweeps it,
# and removes the address again. The interface's existing configuration is left
# untouched.
#
#   sudo ./find_controller.sh [interface]
#
set -u

IFACE="${1:-enp9s0}"
HOST_SUFFIX=222          # The address this PC borrows while sweeping
CANDIDATE_SUBNETS=(192.168.2 192.168.0 192.168.1 192.168.100 192.168.10 10.0.0)
LIKELY_HOSTS=(160 100 1 10 200)   # Controller addresses worth trying first

if [[ $EUID -ne 0 ]]; then
    echo "Needs root to add temporary addresses:  sudo $0 $IFACE" >&2
    exit 1
fi

if [[ ! -d /sys/class/net/$IFACE ]]; then
    echo "No such interface: $IFACE" >&2
    exit 1
fi

carrier=$(cat "/sys/class/net/$IFACE/carrier" 2>/dev/null || echo 0)
if [[ "$carrier" != "1" ]]; then
    echo "$IFACE has no link (carrier=$carrier). Check the cable before scanning." >&2
    exit 1
fi

echo "Interface : $IFACE"
echo "Link      : $(cat /sys/class/net/$IFACE/speed 2>/dev/null || echo '?') Mb/s"
echo "Existing  : $(ip -br -4 addr show "$IFACE" | awk '{$1=$2=""; print}' | xargs)"
echo

ADDED=""
cleanup() {
    if [[ -n "$ADDED" ]]; then
        ip addr del "$ADDED" dev "$IFACE" 2>/dev/null
    fi
}
trap cleanup EXIT INT TERM

FOUND=0

for net in "${CANDIDATE_SUBNETS[@]}"; do
    addr="$net.$HOST_SUFFIX/24"

    # Skip a subnet this interface already covers.
    if ip -4 addr show "$IFACE" | grep -q " $net\."; then
        echo "== $net.0/24  (already configured, sweeping directly)"
        ADDED=""
    else
        if ! ip addr add "$addr" dev "$IFACE" 2>/dev/null; then
            echo "== $net.0/24  (could not add $addr, skipping)"
            continue
        fi
        ADDED="$addr"
        echo "== $net.0/24  (borrowed $addr)"
    fi

    # Probe the addresses a controller is most likely to sit on first.
    for h in "${LIKELY_HOSTS[@]}"; do
        if ping -c 1 -W 1 -I "$IFACE" "$net.$h" >/dev/null 2>&1; then
            mac=$(ip neigh show "$net.$h" dev "$IFACE" | awk '{print $3}')
            echo "   *** REPLY from $net.$h   MAC ${mac:-unknown}"
            FOUND=1
        fi
    done

    # Then sweep the rest of the subnet in parallel.
    seq 1 254 | xargs -P 64 -I{} sh -c \
        "ping -c 1 -W 1 -I $IFACE $net.{} >/dev/null 2>&1 && echo '   host $net.{}'" 2>/dev/null

    # Anything that answered ARP is now in the neighbour table, even if it
    # ignored ICMP -- some controllers do exactly that.
    neigh=$(ip neigh show dev "$IFACE" | grep "^$net\." | grep -vE "FAILED|INCOMPLETE")
    if [[ -n "$neigh" ]]; then
        echo "   ARP replies in this subnet:"
        echo "$neigh" | sed 's/^/     /'
        FOUND=1
    fi

    cleanup
    ADDED=""
done

echo
if [[ $FOUND -eq 0 ]]; then
    cat <<'EOF'
Nothing answered on any candidate subnet.

The link is up but no device is speaking IP on it. Most likely causes, in order:

  1. The cable is in the wrong port on the controller. xCore boxes expose
     several: teach pendant, EtherCAT, and an external/SDK port. Only the last
     one runs a normal IP stack.
  2. The controller has not finished booting, or its network service is off.
  3. The controller is on a subnet outside the candidate list.

The authoritative answer is on the teach pendant: Control Panel ->
Communication -> External Communication shows the controller's IP and mask.
EOF
else
    cat <<'EOF'
Found a device. Make its subnet permanent on this PC, for example:

  sudo nmcli connection modify "有线连接 1" ipv4.method manual \
      ipv4.addresses <PC_ADDRESS>/24
  sudo nmcli connection up "有线连接 1"

then set ArmConfig.ip to the controller address.
EOF
fi
