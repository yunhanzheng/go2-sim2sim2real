#!/usr/bin/env python3
"""Live viewer for the preprocessed depth image the policy sees.

Subscribes to /rl/depth_image (32FC1, 58x87, values in [-0.5, 0.5]) and
shows a colourmap-scaled, upsampled window.  Also shows the raw camera topic
side-by-side when --raw is passed.

Pass --bag <output_dir> to simultaneously record a ros2 bag with both the
preprocessed and (if --raw) the raw depth topics, for later replay/analysis.

Usage (on robot or Mac with ROS2):
  python3 depth_viewer_node.py
  python3 depth_viewer_node.py --raw /camera/depth/image_raw --scale 8
  python3 depth_viewer_node.py --raw /camera/depth/image_raw --bag ~/bags/depth_run1
"""
import argparse
import signal
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray


CMAP = cv2.COLORMAP_MAGMA   # matches common depth visualisations; change to VIRIDIS if preferred


def depth_msg_to_uint8(msg: Image, vmin: float, vmax: float) -> np.ndarray:
    h, w = msg.height, msg.width
    if msg.encoding == "32FC1":
        arr = np.frombuffer(msg.data, dtype=np.float32).reshape(h, w)
    elif msg.encoding == "16UC1":
        arr = np.frombuffer(msg.data, dtype=np.uint16).reshape(h, w).astype(np.float32) * 1e-3
    else:
        raise ValueError(f"unsupported encoding: {msg.encoding}")
    arr = np.clip((arr - vmin) / (vmax - vmin + 1e-6), 0.0, 1.0)
    return (arr * 255).astype(np.uint8)


class DepthViewerNode(Node):
    def __init__(self, policy_topic: str, raw_topic: str | None, scale: int,
                 bag_proc=None, snap_dir=None, snap_every_s: float = 0.0):
        super().__init__("depth_viewer_node")
        self._bag_proc = bag_proc
        self._snap_dir = snap_dir
        self._snap_idx = 0
        self._snap_every_s = snap_every_s
        self._last_snap_t = 0.0
        self._scale = scale
        self._policy_img = None
        self._raw_img = None

        self.create_subscription(Image, policy_topic, self._on_policy, 10)
        self.get_logger().info(f"subscribing to {policy_topic}  (policy view, 58×87, [-0.5, 0.5])")

        if raw_topic:
            self.create_subscription(Image, raw_topic, self._on_raw, 10)
            self.get_logger().info(f"subscribing to {raw_topic}  (raw camera, metres)")

        # Latency diagnostics (all optional — the strip shows "--" until the
        # publishing node is up, and greys out when a stream goes stale).
        # /rl/depth_latency:      [elapsed, wait, achieved, budget, late_count, offset] ms
        # /rl/proprio_latency:    [lowstate_age, wait, achieved, budget] ms
        # /rl/depth_clock_offset: [offset_ms, transport_floor_ms, baseline_ready]
        self._depth_lat = None    # (values, recv_monotonic)
        self._proprio_lat = None
        self._clock_off = None
        self.create_subscription(Float32MultiArray, "/rl/depth_latency",
                                 lambda m: self._on_diag("_depth_lat", m), 10)
        self.create_subscription(Float32MultiArray, "/rl/proprio_latency",
                                 lambda m: self._on_diag("_proprio_lat", m), 10)
        self.create_subscription(Float32MultiArray, "/rl/depth_clock_offset",
                                 lambda m: self._on_diag("_clock_off", m), 10)

    def _on_diag(self, attr: str, msg: Float32MultiArray):
        setattr(self, attr, (list(msg.data), time.monotonic()))

    def _on_policy(self, msg: Image):
        h, w = msg.height, msg.width
        arr = np.frombuffer(msg.data, dtype=np.float32).reshape(h, w)
        # auto-contrast so the full grayscale range is used regardless of scene depth
        lo, hi = arr.min(), arr.max()
        normed = np.clip((arr - lo) / (hi - lo + 1e-6), 0.0, 1.0)
        u8 = (normed * 255).astype(np.uint8)
        bgr = cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)
        # overlay actual value range so you know what you're looking at
        cv2.putText(bgr, f"[{lo:.2f}, {hi:.2f}]", (2, h - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
        self._policy_img = bgr

    def _on_raw(self, msg: Image):
        u8 = depth_msg_to_uint8(msg, vmin=0.0, vmax=3.0)
        self._raw_img = cv2.applyColorMap(u8, CMAP)

    def get_frame(self):
        """Build the composite display frame; returns None if no data yet."""
        panels = []
        if self._policy_img is not None:
            p = cv2.resize(self._policy_img,
                           (self._policy_img.shape[1] * self._scale,
                            self._policy_img.shape[0] * self._scale),
                           interpolation=cv2.INTER_NEAREST)
            cv2.putText(p, "policy view (58x87, [-0.5,0.5])", (4, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            panels.append(p)
        if self._raw_img is not None:
            target_h = panels[0].shape[0] if panels else self._raw_img.shape[0] * self._scale
            aspect = self._raw_img.shape[1] / self._raw_img.shape[0]
            target_w = int(target_h * aspect)
            r = cv2.resize(self._raw_img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            cv2.putText(r, "raw depth (metres, 0-3 m)", (4, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            panels.append(r)
        if not panels:
            return None
        frame = np.hstack(panels) if len(panels) > 1 else panels[0]
        return np.vstack([frame, self._metrics_strip(frame.shape[1])])

    GREEN, YELLOW, RED, GREY = ((80, 220, 80), (60, 200, 240),
                                (60, 60, 255), (140, 140, 140))

    def _metrics_strip(self, width: int) -> np.ndarray:
        """One-line latency status bar rendered under the depth panels.

        depth: achieved vs budget (green within 5 ms of the enforced
        constant, yellow within 15, red beyond) + late-frame count;
        clk: Jetson-Mac clock offset from the passive drift tracker;
        prop: proprio receipt->publish vs its budget. Values grey out when
        their stream has been silent > 2 s (node down / not engaged).
        """
        strip = np.zeros((26, width, 3), dtype=np.uint8)
        now = time.monotonic()

        def entry(stored, fmt, label):
            if stored is None:
                return f"{label} --", self.GREY
            values, recv_t = stored
            text, color = fmt(values)
            return (text, self.GREY) if now - recv_t > 2.0 else (text, color)

        def fmt_depth(v):
            achieved, budget, late = v[2], v[3], int(v[4])
            err = abs(achieved - budget)
            color = (self.GREEN if err <= 5.0 else
                     self.YELLOW if err <= 15.0 else self.RED)
            if budget <= 0.0:
                color = self.YELLOW   # enforcement off — natural latency
            text = f"depth {achieved:5.1f}/{budget:.0f}ms"
            if late:
                text += f" late:{late}"
                color = self.RED
            return text, color

        def fmt_clock(v):
            offset, ready = v[0], v[2] >= 1.0
            color = ((self.GREEN if abs(offset) < 30.0 else self.YELLOW)
                     if ready else self.GREY)
            return (f"clk {offset:+5.1f}ms" + ("" if ready else " (warmup)"),
                    color)

        def fmt_prop(v):
            achieved, budget = v[2], v[3]
            err = abs(achieved - budget)
            color = (self.GREEN if err <= 3.0 else
                     self.YELLOW if err <= 8.0 else self.RED)
            if budget <= 0.0:
                color = self.YELLOW
            return f"prop {achieved:4.1f}/{budget:.0f}ms", color

        x = 6
        for stored, fmt, label in ((self._depth_lat, fmt_depth, "depth"),
                                   (self._clock_off, fmt_clock, "clk"),
                                   (self._proprio_lat, fmt_prop, "prop")):
            text, color = entry(stored, fmt, label)
            cv2.putText(strip, text, (x, 18), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, color, 1, cv2.LINE_AA)
            (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            x += tw + 18
        return strip

    def maybe_snap(self, frame):
        import time
        now_s = time.monotonic()
        do_snap = (self._snap_dir is not None and self._snap_every_s > 0
                   and (now_s - self._last_snap_t) >= self._snap_every_s)
        if do_snap:
            self._last_snap_t = now_s
        return do_snap

    def save_snap(self, frame):
        import time
        ts_ms = int(time.monotonic() * 1000)
        path = self._snap_dir / f"snap_{ts_ms}_{self._snap_idx:04d}.png"
        cv2.imwrite(str(path), frame)
        self._snap_idx += 1
        self.get_logger().info(f"snapshot → {path}")


def main():
    ap = argparse.ArgumentParser(description="Visualise the preprocessed depth the policy sees")
    ap.add_argument("--policy-topic", default="/rl/depth_image",
                    help="topic publishing the 58×87 preprocessed depth (default: /rl/depth_image)")
    ap.add_argument("--raw", metavar="TOPIC", default=None,
                    help="also show raw camera topic side-by-side (e.g. /camera/depth/image_raw)")
    ap.add_argument("--scale", type=int, default=8,
                    help="pixel scale factor for the policy view window (default: 8)")
    ap.add_argument("--bag", metavar="DIR", default=None,
                    help="record a ros2 bag to this directory (created automatically)")
    ap.add_argument("--snap-dir", metavar="DIR", default=None,
                    help="directory to save PNG snapshots (Space = manual, or use --snap-every)")
    ap.add_argument("--snap-every", metavar="SEC", type=float, default=0.0,
                    help="auto-save a snapshot every N seconds (requires --snap-dir)")
    args, ros_args = ap.parse_known_args()

    bag_proc = None
    if args.bag:
        topics = [args.policy_topic]
        if args.raw:
            topics.append(args.raw)
        out_dir = Path(args.bag)
        out_dir.parent.mkdir(parents=True, exist_ok=True)
        cmd = ["ros2", "bag", "record", "-o", str(out_dir)] + topics
        bag_proc = subprocess.Popen(cmd)
        print(f"[depth_viewer] rosbag recording → {out_dir}  topics: {topics}")

    snap_dir = None
    if args.snap_dir:
        snap_dir = Path(args.snap_dir)
        snap_dir.mkdir(parents=True, exist_ok=True)
        print(f"[depth_viewer] snapshots → {snap_dir}  (press Space)")

    rclpy.init(args=ros_args)
    node = DepthViewerNode(args.policy_topic, args.raw, args.scale,
                           bag_proc=bag_proc, snap_dir=snap_dir,
                           snap_every_s=args.snap_every)

    # Spin rclpy in a background thread so cv2.imshow can own the main thread
    # (required on macOS where Cocoa GUI must run on the main thread).
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    W, H = args.scale * 87, args.scale * 58
    placeholder = np.zeros((H, W, 3), dtype=np.uint8)
    cv2.putText(placeholder, "waiting for /rl/depth_image ...", (10, H // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.imshow("depth_viewer", placeholder)
    cv2.waitKey(1)

    try:
        while rclpy.ok():
            frame = node.get_frame()
            if frame is None:
                frame = placeholder
            else:
                if node.maybe_snap(frame):
                    node.save_snap(frame)
            cv2.imshow("depth_viewer", frame)
            key = cv2.waitKey(33) & 0xFF  # ~30 fps
            if key == ord(" ") and snap_dir is not None and frame is not None:
                node.save_snap(frame)
            elif key == ord("q") or key == 27:
                break
    finally:
        rclpy.shutdown()
        spin_thread.join(timeout=2.0)
        if bag_proc and bag_proc.poll() is None:
            bag_proc.send_signal(signal.SIGINT)
            bag_proc.wait()
        node.destroy_node()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
