#!/bin/bash
# Install the clock-sync client + timer on the robot's Jetson.
#
# This runs ON THE JETSON under sudo. From the deployment host:
#
#   mkdir -p /tmp/g2cs && scp deploy_mac/jetson/go2_sync_clock.py \
#     deploy_mac/jetson/go2-clock-sync.service deploy_mac/jetson/go2-clock-sync.timer \
#     deploy_mac/jetson/install_clock_sync.sh unitree@192.168.123.18:/tmp/g2cs/
#   ssh unitree@192.168.123.18 'echo 123 | sudo -S bash /tmp/g2cs/install_clock_sync.sh'
#
# The host that serves the time is $1 (default 192.168.123.99:8123) and lands in
# /etc/default/go2-clock-sync. Nothing here touches unitree_ros2, ~/go2, or any
# running node — a clock step does not move a motor.
set -euo pipefail

SERVER="${1:-192.168.123.99:8123}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "$(id -u)" -ne 0 ]; then
  echo "install_clock_sync.sh: must run as root (use sudo)" >&2
  exit 1
fi

# Root-owned, 0755: systemd runs it as root, so it must not be writable by
# unitree — a writable root-executed script is a free escalation path.
install -o root -g root -m 0755 "$HERE/go2_sync_clock.py" \
  /usr/local/sbin/go2_sync_clock.py
install -o root -g root -m 0644 "$HERE/go2-clock-sync.service" \
  "$HERE/go2-clock-sync.timer" /etc/systemd/system/

printf 'GO2_TIME_SERVER=%s\n' "$SERVER" > /etc/default/go2-clock-sync
chmod 0644 /etc/default/go2-clock-sync

systemctl daemon-reload
systemctl enable --now go2-clock-sync.timer >/dev/null

# The timer's OnBootSec is already in the past on a running system, but the
# first activation after enabling is not guaranteed, so run it once now.
systemctl start go2-clock-sync.service || true

echo "installed: /usr/local/sbin/go2_sync_clock.py"
echo "config:    /etc/default/go2-clock-sync (GO2_TIME_SERVER=$SERVER)"
echo "timer:     $(systemctl is-enabled go2-clock-sync.timer) / $(systemctl is-active go2-clock-sync.timer)"
echo "clock:     $(date -u '+%F %T') UTC"
echo "--- last run ---"
systemctl status go2-clock-sync.service --no-pager -n 5 2>&1 | tail -8
