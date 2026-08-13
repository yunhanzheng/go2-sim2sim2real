#!/usr/bin/env python3
"""Dry-run depth source.

Mirrors the paper's two-process split: a separate node that renders the
front_depth_cam from MuJoCo (the same camera distill_runner.py uses) and
publishes preprocessed depth on /rl/depth_image, exactly the topic
depth_encoder_node.py expects.

Avoids a parallel physics sim by subscribing to /mujoco/pos that
mujoco_simulator.py already publishes (qpos[:19]). Locally only mj_forward()
is called to set body transforms before rendering.
"""
import os
# EGL must be set before mujoco import for headless render to work.
os.environ.setdefault("MUJOCO_GL", "glfw")

from pathlib import Path
import argparse

import numpy as np
import rclpy
import torch
from rclpy.node import Node
from std_msgs.msg import Bool, Float32MultiArray
from sensor_msgs.msg import Image

import mujoco

from distill_common import (
    normalize_depth, TrainDepthDegrader, MujocoStereoDepth,
    TRAIN_DEPTH_CFG, TRAIN_DEPTH_SOURCE_COMMIT, DEPTH_RAW_H, DEPTH_RAW_W,
    DEPTH_TARGET_H, DEPTH_TARGET_W, DEPTH_FAR, DEPTH_HFOV_DEG,
    MUJOCO_EMPTY_BELOW_M, clip_mujoco_too_close_to_far,
    mujoco_depth_scene_option, radial_maxperp_mask,
    depth_radial_clip_enabled,
)
from depth_camera_node import metres_to_image_msg


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SCENE = REPO_ROOT / "resources" / "go2" / "scene_terrain.xml"


class DepthSimNode(Node):
    def __init__(self, scene: Path, depth_mode: str, rate_hz: float,
                 depth_noise_scale: float = 0.0, stereo_shadow=None,
                 seed=None, gate_on_engage: bool = False,
                 cam_roll_deg: float = 0.0):
        super().__init__("depth_sim_node")
        self.depth_mode = depth_mode
        self.rate_hz = rate_hz
        self.gate_on_engage = bool(gate_on_engage)
        self.policy_engaged = False
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        # Apply the latest cliffside Warp-camera image degradation: per-frame
        # bias, MGDP grain/row-runs, and Parkour-in-the-Wild edge/holes.
        self.depth_noise_scale = float(depth_noise_scale)
        self.degrader = (TrainDepthDegrader(noise_scale=self.depth_noise_scale)
                         if self.depth_noise_scale > 0.0 else None)

        self.get_logger().info(f"loading scene {scene}")
        self.m = mujoco.MjModel.from_xml_path(str(scene))
        self.d = mujoco.MjData(self.m)

        self.cam_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_CAMERA, "front_depth_cam")
        if self.cam_id < 0:
            raise RuntimeError("front_depth_cam not found in scene")
        self.cam_roll_deg = float(cam_roll_deg)
        if self.cam_roll_deg != 0.0:
            self._apply_camera_roll(self.cam_roll_deg)

        # The cliffside training camera always applies stereo geometry before
        # image-space noise. In auto mode it follows the noise-model toggle so
        # legacy clean simulations remain unchanged; explicit on supports the
        # training curriculum's stereo-but-image-clean s=0 endpoint.
        self.stereo_shadow = (self.depth_noise_scale > 0.0
                              if stereo_shadow is None else bool(stereo_shadow))
        if self.stereo_shadow and self.degrader is None:
            self.degrader = TrainDepthDegrader(noise_scale=0.0)
        self.stereo = (MujocoStereoDepth(
            self.m, self.cam_id,
            baseline=TRAIN_DEPTH_CFG["stereo_baseline"],
            eps=TRAIN_DEPTH_CFG["stereo_eps"],
            radial_clip=depth_radial_clip_enabled(),
        ) if self.stereo_shadow and self.depth_mode == "render" else None)

        if self.depth_mode == "render":
            self.renderer = mujoco.Renderer(self.m, height=DEPTH_RAW_H, width=DEPTH_RAW_W)
            self.scene_option = mujoco_depth_scene_option()
        else:
            self.renderer = None
            self.scene_option = None

        # Current training uses a planar perpendicular-depth far clip. The old
        # spherical corner vignette remains available only as a legacy opt-in.
        self.radial_clip = depth_radial_clip_enabled()
        self._radial_maxperp = (radial_maxperp_mask(
            DEPTH_RAW_H, DEPTH_RAW_W, DEPTH_HFOV_DEG, DEPTH_FAR)
            if self.radial_clip else None)

        self._latest_qpos = None
        self.create_subscription(Float32MultiArray, "/mujoco/pos", self._on_qpos, 10)
        self.create_subscription(Bool, "/policy_engaged", self._on_engaged, 10)
        self.pub = self.create_publisher(Image, "/rl/depth_image", 10)
        self.timer = self.create_timer(1.0 / rate_hz, self._tick)
        self.get_logger().info(
            f"depth_sim_node: mode={depth_mode} rate={rate_hz:.1f}Hz "
            f"seed={seed if seed is not None else 'random'} "
            f"depth_noise_scale={self.depth_noise_scale:.2f} "
            f"cam_roll_deg={self.cam_roll_deg:+.2f} "
            f"close_clip=<{MUJOCO_EMPTY_BELOW_M:.2f}m→far "
            f"stereo_shadow={'on' if self.stereo_shadow else 'off'} "
            f"radial_clip={'on' if self.radial_clip else 'off (planar)'} "
            f"training_depth={TRAIN_DEPTH_SOURCE_COMMIT} → /rl/depth_image")

    def _apply_camera_roll(self, cam_roll_deg: float):
        cam_quat = np.asarray(self.m.cam_quat[self.cam_id], dtype=np.float64)
        cam_mat = np.zeros(9, dtype=np.float64)
        mujoco.mju_quat2Mat(cam_mat, cam_quat)
        cam_rot = cam_mat.reshape(3, 3)

        roll = np.radians(cam_roll_deg)
        cz = np.cos(roll)
        sz = np.sin(roll)
        roll_local_z = np.array([
            [cz, -sz, 0.0],
            [sz,  cz, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        rotated = cam_rot @ roll_local_z
        quat = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quat, rotated.ravel())
        self.m.cam_quat[self.cam_id] = quat

    def _on_qpos(self, msg: Float32MultiArray):
        if len(msg.data) < 19:
            return
        self._latest_qpos = np.asarray(msg.data[:19], dtype=np.float64)

    def _on_engaged(self, msg: Bool):
        self.policy_engaged = bool(msg.data)

    def _tick(self):
        if self.gate_on_engage and not self.policy_engaged:
            return
        stamp = self.get_clock().now().to_msg()
        if self.depth_mode == "zeros":
            norm = np.zeros((DEPTH_TARGET_H, DEPTH_TARGET_W), dtype=np.float32)
            out = metres_to_image_msg(norm, stamp, frame_id="depth_sim")
            self.pub.publish(out)
            return

        if self.depth_mode == "black":
            # Feed a literal BLACK depth image (diagnostic). The depth inset maps
            # normalized depth via (norm+0.5)*255, so a black pixel (0) == norm
            # -0.5 == NEAR. So an all-black image = "everything point-blank in
            # front" everywhere. Constant, pose-independent.
            norm = np.full((DEPTH_TARGET_H, DEPTH_TARGET_W), -0.5, dtype=np.float32)
            out = metres_to_image_msg(norm, stamp, frame_id="depth_sim")
            self.pub.publish(out)
            return

        if self._latest_qpos is None:
            return  # wait for first /mujoco/pos
        self.d.qpos[:19] = self._latest_qpos
        mujoco.mj_forward(self.m, self.d)

        self.renderer.enable_depth_rendering()
        self.renderer.update_scene(
            self.d, camera="front_depth_cam", scene_option=self.scene_option)
        raw = self.renderer.render()
        self.renderer.disable_depth_rendering()

        # Stereo visibility is independent of the far-clip convention. Current
        # training and MuJoCo both use planar perpendicular-depth clipping.
        if self.stereo is not None:
            # Includes the right-FOV disparity bound and exact right-imager
            # half-occlusion test. Legacy radial clipping is opt-in above.
            raw = self.stereo(self.d, raw)
        elif self.radial_clip and raw.shape == self._radial_maxperp.shape:
            raw = np.where(raw > self._radial_maxperp,
                           np.float32(DEPTH_FAR), raw).astype(np.float32)

        # Emulate the depth sensor's unusable near field in simulation. Apply
        # before either the clean or degraded image pipeline so every MuJoCo
        # pixel below 15 cm is encoded as empty/far, never as a near obstacle.
        raw = clip_mujoco_too_close_to_far(raw)

        if self.degrader is not None:
            norm = self.degrader(raw).numpy()[0]  # training-matched degraded depth
        else:
            norm = normalize_depth(raw).numpy()[0]  # (58, 87) in [-0.5, 0.5]
        out = metres_to_image_msg(norm, stamp, frame_id="depth_sim")
        self.pub.publish(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    ap.add_argument("--depth_mode", choices=["render", "zeros", "black"], default="render")
    ap.add_argument("--rate", type=float, default=10.0,
                    help="depth publish rate (Hz); paper uses 10")
    ap.add_argument("--depth_noise_scale", type=float, default=0.0,
                    help="Latest training curriculum scale: 0=clean image-space "
                         "pipeline, 0.5=reality-tuned rates, 1=2x rates/amplitudes.")
    ap.add_argument("--stereo_shadow", choices=("auto", "on", "off"), default="auto",
                    help="D435 right-imager visibility model. auto enables it when "
                         "depth_noise_scale > 0; on also models training's stereo "
                         "geometry at image-space scale 0.")
    ap.add_argument("--seed", type=int, default=None,
                    help="Optional NumPy/PyTorch seed for repeatable depth degradation.")
    ap.add_argument("--gate_on_engage", type=int, choices=[0, 1], default=0,
                    help="1 avoids rendering and consuming noise draws before engagement.")
    ap.add_argument("--cam_roll_deg", type=float, default=0.0,
                    help="Rotate front_depth_cam about its optical axis by this many degrees.")
    # ignore ros-args
    args, _ = ap.parse_known_args()

    rclpy.init()
    if not 0.0 <= args.depth_noise_scale <= 1.0:
        ap.error("--depth_noise_scale must be in [0, 1]")
    seed = args.seed if args.seed is not None and args.seed >= 0 else None
    stereo_shadow = None if args.stereo_shadow == "auto" else args.stereo_shadow == "on"
    node = DepthSimNode(args.scene, args.depth_mode, args.rate,
                        depth_noise_scale=args.depth_noise_scale,
                        stereo_shadow=stereo_shadow, seed=seed,
                        gate_on_engage=bool(args.gate_on_engage),
                        cam_roll_deg=args.cam_roll_deg)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
