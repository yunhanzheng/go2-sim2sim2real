#!/usr/bin/env python3
"""Verify that RealSense GLOBAL_TIME is aligned with the Mac wall clock."""

import argparse
import json
import time

import rclpy
from rclpy import qos
from rclpy.node import Node
from realsense2_camera_msgs.msg import Metadata


class OneMetadata(Node):
    def __init__(self, topic: str):
        super().__init__("camera_clock_check")
        self.result = None
        self.create_subscription(Metadata, topic, self._cb,
                                 qos.qos_profile_sensor_data)

    def _cb(self, msg: Metadata):
        receive_wall_ns = time.time_ns()
        md = json.loads(msg.json_data)
        frame_ms = float(md["frame_timestamp"])
        hw_us = float(md["hw_timestamp"])
        sensor_us = float(md["sensor_timestamp"])
        exposure_ns = round(frame_ms * 1_000_000.0
                            - (hw_us - sensor_us) * 1_000.0)
        header_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        self.result = {
            "domain": md.get("clock_domain", "unknown"),
            "header_age_ms": (receive_wall_ns - header_ns) / 1e6,
            "exposure_age_ms": (receive_wall_ns - exposure_ns) / 1e6,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="/camera/depth/metadata")
    parser.add_argument("--timeout", type=float, default=8.0)
    args, ros_args = parser.parse_known_args()
    rclpy.init(args=ros_args)
    node = OneMetadata(args.topic)
    deadline = time.monotonic() + args.timeout
    while node.result is None and time.monotonic() < deadline and rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.25)
    result = node.result
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
    if result is None:
        raise SystemExit(f"ERROR: no metadata received from {args.topic}")
    print("== camera clock verification: "
          f"domain={result['domain']} "
          f"frame-to-Mac={result['header_age_ms']:.3f} ms "
          f"exposure-to-Mac={result['exposure_age_ms']:.3f} ms")
    if result["domain"] != "global_time":
        raise SystemExit("ERROR: RealSense metadata is not in GLOBAL_TIME")
    if not (-10.0 <= result["header_age_ms"] <= 1000.0):
        raise SystemExit("ERROR: camera and Mac clocks are not synchronized")


if __name__ == "__main__":
    main()
