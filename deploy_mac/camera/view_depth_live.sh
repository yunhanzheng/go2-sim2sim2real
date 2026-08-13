#!/usr/bin/env bash
# Open a live window of the depth image the policy is fed (/rl/depth_image).
# Runs depth_camera_node (camera -> /rl/depth_image) + the viewer GUI. READ-ONLY:
# no low_level_ctrl, nothing on /lowcmd — the robot does not move.
# Requires the Jetson realsense service publishing /camera/depth/image_rect_raw.
#
#   ./deploy_mac/camera/view_depth_live.sh
# (no `set -u`: conda activation scripts reference unset vars)
HERE="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$HERE"
source deploy_mac/env.sh
export PYTHONPATH="$HERE/src/deploy_rl_policy/scripts:${PYTHONPATH:-}"

cleanup() { kill "${DC:-}" 2>/dev/null; ./deploy_mac/kill_ros.sh >/dev/null 2>&1; }
trap cleanup EXIT INT TERM

ros2 topic list 2>/dev/null | grep -q image_rect_raw \
  || { echo "ERR: /camera/depth/image_rect_raw not found — is the Jetson realsense-depth service up?"; exit 1; }

echo "starting depth_camera_node (/camera/depth/image_rect_raw -> /rl/depth_image)..."
python src/deploy_rl_policy/scripts/depth_camera_node.py \
  --ros-args -p input_topic:=/camera/depth/image_rect_raw >/tmp/depthcam.log 2>&1 & DC=$!

echo "opening depth viewer window (close it to stop)..."
python deploy_mac/camera/view_depth.py
