#!/bin/bash
# Install the host side of the robot's clock as a systemd --user unit.
#
#   ./deploy_mac/lib/install_time_server.sh                # bind 192.168.123.x
#   ./deploy_mac/lib/install_time_server.sh 192.168.123.99
#
# Rootless on purpose: the unit runs as you, the server binds a high port, and
# there is nothing to escalate over. The Jetson side (deploy_mac/jetson/CLOCK.md)
# pulls from here every 30 s and simply keeps retrying if this is not up.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PORT=8123
ACTION="${ACTION:-install}"

# Binding the robot NIC rather than 0.0.0.0 keeps this off wifi and VPN links.
# The address is also what the Jetson has to be pointed at, so do not guess it
# silently: auto-detect only the robot subnet, and stop if there is no match.
BIND="${1:-}"
if [ -z "$BIND" ] && [ "$ACTION" != "uninstall" ]; then
  BIND="$(ip -4 -br addr 2>/dev/null \
    | awk '/192\.168\.123\./ { sub(/\/.*/, "", $3); print $3; exit }')"
fi
if [ -z "$BIND" ] && [ "$ACTION" != "uninstall" ]; then
  echo "install_time_server.sh: no 192.168.123.x address on this host." >&2
  echo "  is the robot's Ethernet plugged in? or pass the address explicitly." >&2
  exit 1
fi

UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
UNIT="$UNIT_DIR/go2-time-server.service"

if [ "$ACTION" = "uninstall" ]; then
  systemctl --user disable --now go2-time-server.service 2>/dev/null || true
  rm -f "$UNIT"
  systemctl --user daemon-reload
  echo "removed $UNIT"
  exit 0
fi

if ! command -v systemctl >/dev/null || ! systemctl --user is-system-running >/dev/null 2>&1; then
  echo "no systemd user manager here; run the server by hand instead:" >&2
  echo "  python3 $REPO/deploy_mac/lib/time_server.py $BIND $PORT" >&2
  exit 1
fi

mkdir -p "$UNIT_DIR"
sed -e "s|@REPO@|$REPO|g" -e "s|@BIND@|$BIND|g" -e "s|@PORT@|$PORT|g" \
  "$REPO/deploy_mac/lib/go2-time-server.service.in" > "$UNIT"

systemctl --user daemon-reload
# --now also covers the first start; if the port is still held by a manual
# instance this fails here, and RestartSec=5 makes the unit converge once the
# old process goes away.
systemctl --user enable --now go2-time-server.service

echo "unit:    $UNIT"
echo "enabled: $(systemctl --user is-enabled go2-time-server.service)"
echo "active:  $(systemctl --user is-active go2-time-server.service)"
echo "serving: $BIND:$PORT"
echo
echo "point the Jetson at it with:"
echo "  /etc/default/go2-clock-sync -> GO2_TIME_SERVER=$BIND:$PORT"
echo
echo "the unit stops when you log out; keep it up with:"
echo "  loginctl enable-linger \"\$USER\""
