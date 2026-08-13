#!/usr/bin/env bash
# Synchronize the RealSense host (Jetson) system clock to this Mac.
# The camera ROS driver publishes RealSense GLOBAL_TIME in the Jetson clock
# domain, so this is required before interpreting camera timestamps on the Mac.
set -e

JETSON_CLOCK_HOST="${JETSON_CLOCK_HOST:-unitree@192.168.123.222}"
_CLOCK_REPO="$(cd "$(dirname "$0")/../.." && pwd)"
_CLOCK_DIR="$(mktemp -d /tmp/go2-jetson-clock.XXXXXX)"
_CLOCK_SOCKET="$_CLOCK_DIR/ssh.sock"
_CLOCK_SERVER_PID=""

# macOS may strip DYLD_* while entering this executable script. Reconstruct the
# local rosidl library path for the final CycloneDDS metadata subscription.
_RS_MSG_LIB="$_CLOCK_REPO/install/realsense2_camera_msgs/lib"
if [ -d "$_RS_MSG_LIB" ]; then
  export DYLD_LIBRARY_PATH="$_RS_MSG_LIB:${DYLD_LIBRARY_PATH:-}"
  export DYLD_FALLBACK_LIBRARY_PATH="$_RS_MSG_LIB:${CONDA_PREFIX:+$CONDA_PREFIX/lib:}${DYLD_FALLBACK_LIBRARY_PATH:-}"
fi

cleanup_clock_sync() {
  if [ -n "$_CLOCK_SERVER_PID" ]; then
    kill "$_CLOCK_SERVER_PID" >/dev/null 2>&1 || true
    wait "$_CLOCK_SERVER_PID" >/dev/null 2>&1 || true
  fi
  ssh -S "$_CLOCK_SOCKET" -O exit "$JETSON_CLOCK_HOST" >/dev/null 2>&1 || true
  rmdir "$_CLOCK_DIR" >/dev/null 2>&1 || true
}
trap cleanup_clock_sync EXIT INT TERM

echo "== opening SSH connection to $JETSON_CLOCK_HOST"
echo "   (enter the Jetson login password if requested)"
ssh -M -S "$_CLOCK_SOCKET" -o ControlPersist=60 -o ConnectTimeout=8 \
  -o ServerAliveInterval=5 -fN "$JETSON_CLOCK_HOST"

echo "== authorizing Jetson clock update"
echo "   (enter the Jetson sudo password if requested)"

# Ubuntu can scope its sudo credential cache to one terminal. Keep sudo
# authentication and clock setting in the same SSH terminal, and serve a fresh
# Mac timestamp only after authentication so password-entry time is not
# included. The privileged Python process sets CLOCK_REALTIME directly after
# receiving the timestamp, avoiding an extra shell + `date` launch (~10 ms).
_CLOCK_MAC_IP="${JETSON_CLOCK_MAC_IP:-$(ipconfig getifaddr "${GO2_IFACE:-en13}")}"
_CLOCK_PORT="${JETSON_CLOCK_PORT:-45123}"
if [ -z "$_CLOCK_MAC_IP" ]; then
  echo "ERROR: cannot determine the Mac address on ${GO2_IFACE:-en13}" >&2
  exit 1
fi

python - "$_CLOCK_MAC_IP" "$_CLOCK_PORT" <<'PY' &
import socket
import sys
import time

host, port = sys.argv[1], int(sys.argv[2])
with socket.socket() as server:
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, port))
    server.listen(1)
    connection, _ = server.accept()
    with connection:
        connection.sendall(f"{time.time():.9f}\n".encode("ascii"))
PY
_CLOCK_SERVER_PID=$!
sleep 0.1
if ! kill -0 "$_CLOCK_SERVER_PID" >/dev/null 2>&1; then
  wait "$_CLOCK_SERVER_PID"
  exit 1
fi

# Preferred path: the root-installed helper with its scoped NOPASSWD sudoers
# line (deploy_mac/jetson/go2_set_clock.py + /etc/sudoers.d/go2-clock-sync) —
# fully non-interactive, works from background deploys with no terminal.
# Fallback: the original interactive sudo flow (sudo -v prompts in this ssh -t).
if ssh -S "$_CLOCK_SOCKET" "$JETSON_CLOCK_HOST" \
    "sudo -n python3 /usr/local/sbin/go2_set_clock.py $_CLOCK_MAC_IP $_CLOCK_PORT" 2>/dev/null; then
  :
else
  echo "   (passwordless helper unavailable — falling back to interactive sudo;"
  echo "    install per deploy_mac/jetson/INSTALL.md to make this non-interactive)"
  # The remote Python read happens only after `sudo -v` has succeeded.
  ssh -S "$_CLOCK_SOCKET" -t "$JETSON_CLOCK_HOST" \
    "sudo -v && sudo -n python3 -c 'import socket, time; s=socket.create_connection((\"$_CLOCK_MAC_IP\", $_CLOCK_PORT), 5); epoch=float(s.recv(64).decode().strip()); time.clock_settime(time.CLOCK_REALTIME, epoch)'"
fi
wait "$_CLOCK_SERVER_PID"
_CLOCK_SERVER_PID=""

export JETSON_CLOCK_SOCKET="$_CLOCK_SOCKET"
export JETSON_CLOCK_TARGET="$JETSON_CLOCK_HOST"
python - <<'PY'
import os
import statistics
import subprocess
import time

sock = os.environ["JETSON_CLOCK_SOCKET"]
target = os.environ["JETSON_CLOCK_TARGET"]
samples = []
for _ in range(12):
    t0 = time.time_ns()
    raw = subprocess.check_output(
        ["ssh", "-S", sock, target, "date +%s%N"], text=True).strip()
    t1 = time.time_ns()
    remote = int(raw)
    samples.append((t1 - t0, remote - ((t0 + t1) // 2)))

# The minimum-RTT exchange has the smallest scheduling/network uncertainty.
rtt_ns, offset_ns = min(samples)
print(f"== verified Jetson-Mac offset: {offset_ns / 1e6:+.3f} ms "
      f"(best RTT {rtt_ns / 1e6:.3f} ms)")
if abs(offset_ns) > 20_000_000:
    raise SystemExit("ERROR: residual clock offset exceeds 20 ms")
PY

# librealsense continuously maps its hardware clock to the Jetson system clock.
# Reset that mapping after a large system-clock correction; otherwise old
# samples can contaminate GLOBAL_TIME for several seconds.
if command -v ros2 >/dev/null 2>&1 && \
   ros2 param get /camera/camera enable_depth >/dev/null 2>&1; then
  echo "== refreshing RealSense GLOBAL_TIME mapping"
  ros2 param set /camera/camera enable_depth false >/dev/null
  sleep 0.5
  ros2 param set /camera/camera enable_depth true >/dev/null
  sleep 2
  python "$_CLOCK_REPO/deploy_mac/camera/check_camera_clock.py"
else
  echo "WARN: ROS camera node unavailable; restart it before measuring camera latency"
fi

echo "== Jetson clock synchronized"
