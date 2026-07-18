#!/usr/bin/env bash
# fix_network.sh — pin the x86 box's WiFi to a static IP + working DNS.
#
# The WiFi connection "人形机器人" (wlp2s0) is on DHCP, so its IP changes.
# This switches it to a static (manual) IPv4 config using the values it
# currently has, and sets DNS (internal 10.0.3.46 + public fallbacks).
#
# Usage:
#   sudo bash fix_network.sh          # apply static IP + DNS
#   sudo bash fix_network.sh --revert # go back to DHCP (auto)
#
# Override any value via env vars, e.g.:
#   sudo CON="人形机器人" IPADDR=10.11.144.89/16 GW=10.11.255.254 bash fix_network.sh
set -euo pipefail

CON="${CON:-人形机器人}"
DEV="${DEV:-wlp2s0}"
IPADDR="${IPADDR:-10.11.144.89/16}"
GW="${GW:-10.11.255.254}"
# Internal DNS first (resolves lab hosts), then public fallbacks.
DNS="${DNS:-10.0.3.46 1.1.1.1 8.8.8.8}"

if [[ "${1:-}" == "--revert" ]]; then
  echo ">> Reverting '$CON' to DHCP (auto)..."
  nmcli con mod "$CON" ipv4.method auto ipv4.addresses "" ipv4.gateway "" \
    ipv4.dns "" ipv4.ignore-auto-dns no
  nmcli con up "$CON"
  echo ">> Done. Current address:"
  ip -4 addr show "$DEV" | grep inet || true
  exit 0
fi

echo ">> Pinning '$CON' ($DEV) to static:"
echo "     address: $IPADDR"
echo "     gateway: $GW"
echo "     dns    : $DNS"

nmcli con mod "$CON" \
  ipv4.method manual \
  ipv4.addresses "$IPADDR" \
  ipv4.gateway "$GW" \
  ipv4.dns "$DNS" \
  ipv4.ignore-auto-dns yes

echo ">> Re-applying connection (the link may blip for a second)..."
nmcli con up "$CON"

echo
echo ">> New address:"
ip -4 addr show "$DEV" | grep inet || true
echo ">> DNS in use:"
resolvectl dns "$DEV" 2>/dev/null || true
echo ">> Connectivity check:"
ping -c 2 -W 2 "$GW" >/dev/null 2>&1 && echo "   gateway OK" || echo "   gateway UNREACHABLE"
getent ahosts gitlab.com >/dev/null 2>&1 && echo "   DNS OK (gitlab.com resolved)" || echo "   DNS still failing"
echo ">> Done."
