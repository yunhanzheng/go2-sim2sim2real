#!/bin/bash
# Watchdog for the wedged-but-alive RealSense failure mode: node running,
# topics advertised, zero frames delivered (observed 2026-07-27).
#
# v2 (2026-07-27): PERSISTENT rclpy subscriber. v1 spawned `ros2 topic echo`
# per probe — each spawn is a fresh DDS participant whose discovery storm
# STALLED the driver ~1.6 s every ~63 s on the loaded Jetson (measured in the
# first real-run bag; gaps vanished with the watchdog stopped). One long-lived
# subscriber costs discovery exactly once.
source /opt/ros/foxy/setup.bash
source /home/unitree/cyclonedds_ws/install/setup.bash
set -u
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=0
export CYCLONEDDS_URI=/home/unitree/cyclonedds_ws/cyclonedds.xml

exec python3 - <<'EOF'
import subprocess
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

TOPIC = "/camera/depth/image_rect_raw"
SERVICE = "go2-realsense.service"
STALL_S = 15.0            # no frame for this long => restart the driver
GRACE_S = 30.0            # settle time after a restart (driver + USB ~15 s)

rclpy.init()
node = Node("realsense_watchdog")
last = {"t": time.monotonic()}
node.create_subscription(Image, TOPIC, lambda m: last.__setitem__("t", time.monotonic()), 1)
node.get_logger().info(f"watchdog v2: persistent subscriber on {TOPIC}, stall threshold {STALL_S}s")

while rclpy.ok():
    rclpy.spin_once(node, timeout_sec=1.0)
    if time.monotonic() - last["t"] > STALL_S:
        node.get_logger().error(f"no frame for {STALL_S}s — restarting {SERVICE}")
        subprocess.run(["sudo", "/bin/systemctl", "restart", SERVICE])
        last["t"] = time.monotonic() + GRACE_S
EOF
