#!/usr/bin/env python3
"""Snapshot the EXACT depth image fed into the policy during a dry run.

Subscribes /rl/depth_image — the 58x87 32FC1 image (normalized ~[-0.5, 0.5])
that depth_encoder_node reads straight into the encoder. Saves each captured
frame as a raw .npy (exact values the policy sees) + a viewable .png.

Usage (2nd terminal, while a dry run / real run is publishing /rl/depth_image):
  source deploy_mac/env.sh
  python deploy_mac/camera/snap_depth.py                       # every 10th frame, until Ctrl-C
  python deploy_mac/camera/snap_depth.py --every 1 --count 30  # 30 consecutive frames
  python deploy_mac/camera/snap_depth.py --out run_logs/depthsnaps/myrun
"""
import argparse
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

H, W = 58, 87
NEAR, FAR = -0.5, 0.5  # normalized range published on /rl/depth_image


class SnapDepth(Node):
    def __init__(self, out: Path, every: int, count: int, scale: int):
        super().__init__("snap_depth")
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.every = max(1, every)
        self.count = count
        self.scale = scale
        self.seen = 0
        self.saved = 0
        self.create_subscription(Image, "/rl/depth_image", self._on, 10)
        self.get_logger().info(
            f"saving every {self.every}th /rl/depth_image -> {self.out}"
            + (f" (stop after {count})" if count else " (Ctrl-C to stop)")
        )

    def _on(self, msg: Image):
        if msg.encoding != "32FC1":
            self.get_logger().warn(f"expected 32FC1, got {msg.encoding}")
            return
        self.seen += 1
        if self.seen % self.every:
            return
        d = np.frombuffer(bytes(msg.data), dtype="<f4").reshape(
            msg.height, msg.width).copy()
        tag = f"{self.saved:04d}"
        # exact values the policy sees:
        np.save(self.out / f"depth_{tag}.npy", d)
        # viewable png (fixed [-0.5,0.5] mapping; near=dark, far=bright):
        img = np.clip((d - NEAR) / (FAR - NEAR) * 255.0, 0, 255).astype(np.uint8)
        try:
            import cv2
            img = cv2.resize(img, (W * self.scale, H * self.scale),
                             interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(str(self.out / f"depth_{tag}.png"), img)
        except Exception as e:  # png is best-effort; .npy is the source of truth
            self.get_logger().warn(f"png skipped ({e})")
        self.saved += 1
        self.get_logger().info(
            f"saved {tag}  shape={d.shape}  "
            f"min={d.min():+.3f} max={d.max():+.3f} mean={d.mean():+.3f}"
        )
        if self.count and self.saved >= self.count:
            raise SystemExit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="run_logs/depthsnaps",
                    help="output dir for .npy/.png frames")
    ap.add_argument("--every", type=int, default=10,
                    help="save every Nth frame (depth arrives ~10 Hz)")
    ap.add_argument("--count", type=int, default=0,
                    help="stop after N saves (0 = until Ctrl-C)")
    ap.add_argument("--scale", type=int, default=6,
                    help="png upscale factor (58x87 -> Nx)")
    args = ap.parse_args()
    rclpy.init()
    node = SnapDepth(args.out, args.every, args.count, args.scale)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
