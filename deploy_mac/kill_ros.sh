#!/usr/bin/env bash
# Kill orphan ROS 2 nodes + the ros2 daemon left over from this stack.
# ALWAYS run this between launches / after Ctrl-C — the record launches have no
# on_exit shutdown (see ../DEPLOY.md) and orphaned nodes keep holding DDS topics,
# which makes the next run see stale publishers and behave unpredictably.
#
#   ./deploy_mac/kill_ros.sh

# NOTE: distill_recorder and joy_sequencer used to be missing here, so they
# orphaned (reparented to init) on every record run and kept publishing onto
# DDS topics — stale joy_sequencers in particular inject competing /joy engage
# + velocity commands into the next run. They are now included.
NODES="rl_distill_policy|rl_policy|low_level_ctrl|depth_encoder_node|depth_camera_node|depth_sim_node|mujoco_simulator|distill_runner|distill_recorder|joy_sequencer|lidar_runner|base_velocity_estimator|selftest_dds|_ros2_daemon|joy_node|target_pos_bridge"

ros2 daemon stop >/dev/null 2>&1 || true
pkill -f "$NODES" 2>/dev/null || true   # graceful first (clean DDS shutdown)
sleep 1
pkill -9 -f "$NODES" 2>/dev/null || true # then force any reparented stragglers
echo "killed orphan ros nodes. remaining ros-ish processes:"
pgrep -fl "ros2|_ros2_daemon|rmw|$NODES" 2>/dev/null || echo "  (none)"
