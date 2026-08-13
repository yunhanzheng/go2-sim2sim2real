#!/usr/bin/env bash
# Launch depth preprocessor + viewer on the Mac.
# Requires: conda activate ros_humble
# Robot must have the RealSense camera running and be on the same ROS_DOMAIN_ID.

set -e
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
SCRIPTS="$REPO/src/deploy_rl_policy/scripts"
export PYTHONPATH="$SCRIPTS:${PYTHONPATH:-}"

RAW_TOPIC="${RAW_TOPIC:-/camera/depth/image_rect_raw}"
BAG_DIR="${BAG_DIR:-$HOME/bags/depth_$(date +%Y-%m-%d_%H-%M-%S)}"
SNAP_DIR="${SNAP_DIR:-$HOME/snaps/depth_$(date +%Y-%m-%d_%H-%M-%S)}"
SNAP_EVERY="${SNAP_EVERY:-2.0}"

echo "[depth_viewer] PYTHONPATH=$PYTHONPATH"
echo "[depth_viewer] raw topic : $RAW_TOPIC"
echo "[depth_viewer] bag       : $BAG_DIR"
echo "[depth_viewer] snaps     : $SNAP_DIR (every ${SNAP_EVERY}s + Space)"

# Run preprocessor in background
python3 "$SCRIPTS/depth_camera_node.py" \
  --ros-args -p input_topic:="$RAW_TOPIC" &
PREPROC_PID=$!
echo "[depth_viewer] depth_camera_node PID=$PREPROC_PID"

cleanup() {
  echo "[depth_viewer] shutting down..."
  kill "$PREPROC_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# Viewer on foreground (owns main thread for cv2)
python3 "$SCRIPTS/depth_viewer_node.py" \
  --raw "$RAW_TOPIC" \
  --bag "$BAG_DIR" \
  --snap-dir "$SNAP_DIR" \
  --snap-every "$SNAP_EVERY"
