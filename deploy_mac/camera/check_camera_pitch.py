#!/usr/bin/env python3
"""Measure the depth camera's physical mount pitch on the real robot.

Why: the mount angle is the single largest documented sim2real gap on this
robot. With the same checkpoint and only the camera pitch changed, 20 deg gives
6/6 clean hurdle jumps and 7.6 deg gives 3/6 with feet striking the bar
(notes/analysis/sim2real_divergence_camera_pitch_2026-07-27.md). The bracket has
been re-mounted since, and `resources/go2/go2.xml` cannot answer the question —
it currently says 20 deg while its own comment says to keep it at the measured
7.6 until the hardware changes. So measure it.

Method (the same one that produced the 7.6 figure): deproject RAW depth with the
driver's own camera_info intrinsics into a point cloud, least-squares a floor
plane out of the lower part of the frame, and report the angle between that
plane's normal and the camera's own "up" axis. The fit also recovers the camera
height above the floor, which cross-checks it — a wrong pitch biases the height
too.

RUN IT WITH THE ROBOT STANDING ON FLAT FLOOR AND HOLDING STILL, and facing
something at least 1 m away. Lying down does not work: the camera sits ~0.1 m off
the ground, closer than the sensor's minimum range, and there is no floor in the
frame at all (the tool says so instead of inventing a number).

    deploy_mac/camera/check_camera_pitch.py            # 30 frames, ~2 s
    deploy_mac/camera/check_camera_pitch.py --frames 60 --min-height 0.5

Reads the IMU pitch from /lowstate too: the whole measurement is only meaningful
while `imu_pitch` is near zero, because a robot pitched by 5 deg tilts the camera
by the same 5 deg.

Exit 0 = a stable angle was measured; 1 = no floor found; 2 = topics missing.
"""
import argparse
import sys
import time

try:
    import numpy as np
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import CameraInfo, Image
    from unitree_go.msg import LowState
except ImportError as exc:
    print(f"check_camera_pitch: cannot run here ({exc}); source the ROS env first")
    raise SystemExit(2)

DEPTH_TOPIC = "/camera/depth/image_rect_raw"
INFO_TOPIC = "/camera/depth/camera_info"


def decode_depth(m):
    """16UC1 millimetres -> float32 metres."""
    h, w = int(m.height), int(m.width)
    if m.encoding == "16UC1":
        raw = np.frombuffer(bytes(m.data), dtype=np.uint8)
        rows = raw[:m.step * h].reshape(h, m.step)[:, :w * 2]
        return rows.copy().view(np.uint16).reshape(h, w).astype(np.float32) / 1000.0
    if m.encoding == "32FC1":
        return np.frombuffer(bytes(m.data), dtype=np.float32).reshape(h, w).copy()
    raise SystemExit(f"unexpected depth encoding {m.encoding}")


def fit_floor(depth, K, skip=4, top=0.40, bottom=0.92):
    """Deproject the lower band and least-squares a plane.

    Returns (pitch_deg, height_m, inlier_frac) or None. Camera frame is OpenCV's:
    x right, y DOWN, z forward, so a floor that is 20 deg below the optical axis
    puts its normal at (0, -cos20, -sin20).
    """
    h, w = depth.shape
    fx, fy, cx, cy = K[0], K[4], K[2], K[5]
    v0, v1 = int(h * top), int(h * bottom)
    d = depth[v0:v1:skip, ::skip]
    vv, uu = np.mgrid[v0:v1:skip, 0:w:skip]
    ok = (d > 0.20) & (d < 4.0)
    if ok.sum() < 200:
        return None
    z = d[ok]
    pts = np.stack([(uu[ok] - cx) * z / fx,
                    (vv[ok] - cy) * z / fy,
                    z], axis=1)
    # Least-squares plane through the centroid, normal = smallest singular vector.
    c = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    n = vt[2]
    n /= np.linalg.norm(n)
    if n @ np.array([0.0, -1.0, 0.0]) < 0.0:      # make it point "up"
        n = -n
    res = np.abs((pts - c) @ n)
    inliers = float((res < 0.05).mean())
    if inliers < 0.5:
        return None
    pitch = np.degrees(np.arccos(np.clip(n @ np.array([0.0, -1.0, 0.0]), -1, 1)))
    height = float(abs(c @ n))
    return pitch, height, inliers


class Probe(Node):
    def __init__(self):
        super().__init__("check_camera_pitch")
        self.K = None
        self.depth = None
        self.imu_pitch = None
        self.n_depth = 0
        self.create_subscription(CameraInfo, INFO_TOPIC, self.on_info, 10)
        self.create_subscription(Image, DEPTH_TOPIC, self.on_depth, 10)
        self.create_subscription(LowState, "/lowstate", self.on_low, 50)

    def on_info(self, m):
        self.K = list(m.k)

    def on_depth(self, m):
        self.n_depth += 1
        self.depth = m

    def on_low(self, m):
        self.imu_pitch = float(m.imu_state.rpy[1])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--frames", type=int, default=30,
                    help="depth frames to fit (default 30, ~0.5 s at 60 Hz)")
    ap.add_argument("--min-height", type=float, default=0.15,
                    help="reject a fit that puts the camera lower than this (default 0.15)")
    ap.add_argument("--max-height", type=float, default=0.8,
                    help="reject a fit that puts the camera higher than this (default 0.8)")
    opts = ap.parse_args()

    rclpy.init()
    node = Probe()
    end = time.time() + 10.0
    while (node.K is None or node.depth is None) and time.time() < end:
        rclpy.spin_once(node, timeout_sec=0.1)
    if node.K is None:
        print(f"check_camera_pitch: no {INFO_TOPIC}; is the driver running?")
        return 2
    if node.depth is None:
        print(f"check_camera_pitch: no {DEPTH_TOPIC} in 10 s")
        return 2

    fits = []
    for _ in range(opts.frames):
        rclpy.spin_once(node, timeout_sec=0.1)
        if node.depth is None:
            continue
        d = decode_depth(node.depth)
        r = fit_floor(d, node.K)
        if r is None:
            continue
        pitch, height, inl = r
        if not (opts.min_height <= height <= opts.max_height):
            continue
        fits.append((pitch, height, inl))

    if not fits:
        print("check_camera_pitch: no floor plane in the lower part of the frame.")
        print("  A lying robot puts the camera ~0.1 m off the ground, inside the")
        print("  sensor's blind zone — stand the robot up on flat floor first.")
        return 1

    p = np.array([f[0] for f in fits])
    hgt = np.array([f[1] for f in fits])
    print(f"frames   : {len(fits)}/{opts.frames} produced a floor fit")
    print(f"mount    : {np.median(p):.2f} deg  (std {p.std():.2f}, "
          f"p5 {np.percentile(p, 5):.2f} p95 {np.percentile(p, 95):.2f})")
    print(f"height   : {np.median(hgt):.3f} m above the floor")
    if node.imu_pitch is not None:
        print(f"imu pitch: {node.imu_pitch:+.3f} rad "
              f"({np.degrees(node.imu_pitch):+.2f} deg) — "
              + ("robot is level, the number above is the mount angle"
                 if abs(node.imu_pitch) < 0.05 else
                 "ROBOT IS NOT LEVEL, this measurement is biased"))
    m = float(np.median(p))
    print(f"verdict  : " + (
        "≈20 deg — matches training (resources/go2/go2.xml is already 20 deg)"
        if abs(m - 20.0) < 2.0 else
        f"≈7.6 deg — the hardware is still at the OLD mount angle; go2.xml claims "
        f"20 deg but must stay at 7.6 until the bracket is re-tilted"
        if abs(m - 7.6) < 2.0 else
        f"{m:.1f} deg — neither 20 nor 7.6; go2.xml has to be set to this value "
        f"before sim2sim predicts this robot"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
