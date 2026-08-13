#!/usr/bin/env python3
"""Live viewer of the depth image fed to the policy.

By default shows /rl/depth_image — the 58x87 normalized depth (32FC1, ~[-0.5,0.5],
near=-0.5/0 m, far=+0.5/2 m) that depth_encoder_node actually ingests. That's
literally "what the policy sees". READ-ONLY (subscribe only; never commands the robot).

  source deploy_mac/env.sh
  python deploy_mac/camera/view_depth.py               # /rl/depth_image (policy input)
  python deploy_mac/camera/view_depth.py --raw         # /camera/depth/image_rect_raw (raw mm)
  python deploy_mac/camera/view_depth.py --selftest    # headless: confirm frames+shape, no GUI
"""
import argparse
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


def decode(m):
    enc = m.encoding.lower()
    if enc == "32fc1":
        dt = ">f4" if m.is_bigendian else "<f4"
    elif enc in ("16uc1", "mono16"):
        dt = ">u2" if m.is_bigendian else "<u2"
    else:
        dt = "<f4"
    a = np.frombuffer(bytes(m.data), dtype=dt)[: m.height * m.width]
    return a.reshape(m.height, m.width).astype(np.float32), enc


class Viewer(Node):
    def __init__(self, topic):
        super().__init__("view_depth")
        self.a = None
        self.enc = None
        self.n = 0
        self.create_subscription(Image, topic, self._cb, 10)

    def _cb(self, m):
        self.a, self.enc = decode(m)
        self.n += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", default=None)
    ap.add_argument("--raw", action="store_true", help="view /camera/depth/image_rect_raw instead")
    ap.add_argument("--selftest", action="store_true", help="headless read check, no GUI")
    args = ap.parse_args()
    topic = args.topic or ("/camera/depth/image_rect_raw" if args.raw else "/rl/depth_image")

    rclpy.init()
    node = Viewer(topic)
    print(f"subscribing {topic}")

    # wait for first frame
    end = time.time() + 8
    while time.time() < end and node.a is None:
        rclpy.spin_once(node, timeout_sec=0.1)

    if args.selftest:
        if node.a is None:
            print(f"FAIL: no frames on {topic} (is depth_camera_node / the camera running?)")
            node.destroy_node(); rclpy.shutdown(); sys.exit(1)
        print(f"PASS: {node.n} frames, shape={node.a.shape}, enc={node.enc}, "
              f"min={node.a.min():.3f} max={node.a.max():.3f} mean={node.a.mean():.3f}")
        node.destroy_node(); rclpy.shutdown(); sys.exit(0)

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    is_norm = (node.enc == "32fc1") or (not args.raw)
    vmin, vmax, note = (-0.5, 0.5, "policy input (normalized: -0.5≈near/0m, +0.5≈far/2m)") \
        if is_norm else (0.0, 2000.0, "raw depth [mm], 0..2000")
    init = node.a if node.a is not None else np.zeros((58, 87), np.float32)
    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(init, cmap="turbo", vmin=vmin, vmax=vmax, aspect="auto", interpolation="nearest")
    fig.colorbar(im, ax=ax, fraction=0.04)
    ax.set_title(f"Go2 depth fed to the policy — {topic}\n{note}")
    txt = ax.text(0.01, 0.99, "", transform=ax.transAxes, color="w", va="top", fontsize=9,
                  bbox=dict(facecolor="black", alpha=0.4, pad=2))

    def update(_):
        rclpy.spin_once(node, timeout_sec=0.0)
        if node.a is not None:
            im.set_data(node.a)
            txt.set_text(f"frames={node.n}  shape={node.a.shape}  "
                         f"range[{node.a.min():.2f},{node.a.max():.2f}]")
        return [im, txt]

    if node.a is None:
        print(f"WARN: no frames yet on {topic} — opening anyway; is the camera + depth_camera_node up?")
    _ani = FuncAnimation(fig, update, interval=50, blit=False, cache_frame_data=False)
    plt.show()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
