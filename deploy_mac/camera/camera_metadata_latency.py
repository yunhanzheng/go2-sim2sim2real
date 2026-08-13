#!/usr/bin/env python3
"""Log RealSense exposure metadata for cross-host latency measurement.

The RealSense ROS wrapper publishes frame_timestamp in Jetson GLOBAL_TIME and
sensor/hardware timestamps in the camera clock. Their difference maps the
middle-of-exposure sensor timestamp into GLOBAL_TIME without needing the raw
camera clock offset. Synchronize the Jetson clock first with
sync_jetson_clock.sh, then pair this log with latency_depth_camera.tsv by the
shared ROS header stamp.
"""

import argparse
import json
import time
from pathlib import Path

import rclpy
from rclpy import qos
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from realsense2_camera_msgs.msg import Metadata


class MetadataLatencyLogger(Node):
    def __init__(self, topic: str, output: Path):
        super().__init__("camera_metadata_latency_logger")
        output.parent.mkdir(parents=True, exist_ok=True)
        self._fp = output.open("w", buffering=1)
        self._fp.write(
            "metadata_seq\tsource_sec\tsource_nsec\tframe_number\tclock_domain\t"
            "frame_timestamp_ms\thw_timestamp_us\tsensor_timestamp_us\t"
            "exposure_global_ns\tbackend_timestamp_ms\ttime_of_arrival_ms\t"
            "actual_exposure_us\treceive_mono_ns\treceive_wall_ns\n"
        )
        self._seq = 0
        self.create_subscription(Metadata, topic, self._on_metadata,
                                 qos.qos_profile_sensor_data)
        self.get_logger().info(f"logging {topic} exposure metadata -> {output}")

    def _on_metadata(self, msg: Metadata):
        receive_mono_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
        receive_wall_ns = time.time_ns()
        try:
            md = json.loads(msg.json_data)
            frame_ms = float(md["frame_timestamp"])
            hw_us = float(md["hw_timestamp"])
            sensor_us = float(md["sensor_timestamp"])
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            self.get_logger().warning(f"invalid RealSense metadata: {exc}")
            return

        # frame_timestamp and the ROS header are GLOBAL_TIME. hw_timestamp is
        # the beginning of readout; sensor_timestamp is the middle of exposure.
        exposure_global_ns = round(frame_ms * 1_000_000.0
                                   - (hw_us - sensor_us) * 1_000.0)
        values = (
            self._seq,
            msg.header.stamp.sec,
            msg.header.stamp.nanosec,
            md.get("frame_number", ""),
            md.get("clock_domain", ""),
            frame_ms,
            hw_us,
            sensor_us,
            exposure_global_ns,
            md.get("backend_timestamp", ""),
            md.get("time_of_arrival", ""),
            md.get("actual_exposure", ""),
            receive_mono_ns,
            receive_wall_ns,
        )
        self._fp.write("\t".join(map(str, values)) + "\n")
        self._seq += 1

    def destroy_node(self):
        self._fp.close()
        return super().destroy_node()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="/camera/depth/metadata")
    parser.add_argument("--output", required=True, type=Path)
    args, ros_args = parser.parse_known_args()
    rclpy.init(args=ros_args)
    node = MetadataLatencyLogger(args.topic, args.output)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
