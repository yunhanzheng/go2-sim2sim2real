#!/bin/bash
# Watchdog for the wedged-but-alive RealSense failure mode: node running,
# topics advertised, zero frames delivered (observed on the Foxy image
# 2026-07-27; nothing about the Orin NX image rules it out).
#
# PERSISTENT rclpy subscriber, deliberately. The first version spawned a fresh
# `ros2 topic echo` per probe; each spawn is a new DDS participant whose
# discovery storm stalled the driver ~1.6 s every ~63 s on the loaded Jetson
# (visible as gaps in the first real-run bag, gone with the watchdog stopped).
# One long-lived subscriber pays discovery exactly once.
#
# Environment comes from setup_go2.sh — the same script the driver unit sources,
# so the watchdog discovers the driver through the identical CycloneDDS
# interface (enP8p1s0) instead of the default one.
source /home/unitree/unitree_ros2/setup_go2.sh
set -u

exec python3 - <<'EOF'
import subprocess
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

TOPIC = "/camera/depth/image_rect_raw"
SERVICE = "go2-realsense-humble.service"
STALL_S = 15.0             # no frame for this long => restart the driver
GRACE_S = 30.0             # settle time after a restart (driver + USB ~15 s)

rclpy.init()
node = Node("realsense_watchdog")
# Start the clock GRACE_S in the FUTURE, i.e. give the driver a full grace
# window before the first probe counts. Starting at now() would make a
# watchdog that comes up before (or alongside) the driver restart it for
# merely not having finished booting yet.
last = {"t": time.monotonic() + GRACE_S}
node.create_subscription(Image, TOPIC, lambda m: last.__setitem__("t", time.monotonic()), 1)
node.get_logger().info(
    f"watchdog: persistent subscriber on {TOPIC}, stall threshold {STALL_S}s, "
    f"first restart no sooner than {GRACE_S}s"
)

while rclpy.ok():
    rclpy.spin_once(node, timeout_sec=1.0)
    if time.monotonic() - last["t"] > STALL_S:
        node.get_logger().error(f"no frame for {STALL_S}s — restarting {SERVICE}")
        subprocess.run(["sudo", "/bin/systemctl", "restart", SERVICE])
        last["t"] = time.monotonic() + GRACE_S
EOF
