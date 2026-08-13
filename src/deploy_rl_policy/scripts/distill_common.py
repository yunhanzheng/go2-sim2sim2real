"""Shared constants and helpers for the extreme-parkour distillation deploy stack.

Lifted as-is from distill_runner.py so the ROS nodes (depth_sim, depth_camera,
depth_encoder, rl_distill_policy) and the standalone harness share one source
of truth for the AI pipeline.
"""

from pathlib import Path
import os
import time
import sys
import re

import numpy as np
import torch


# src/deploy_rl_policy/scripts -> repo root
REPO_ROOT = Path(__file__).resolve().parents[3]

# The training repo supplies the policy classes (Actor / Estimator /
# RecurrentDepthBackbone): https://github.com/Roman-089/go2-flyball-parkour
# Defaults to a sibling clone; override with $EP_ROOT or --ep_root.
EP_ROOT_DEFAULT = Path(
    os.environ.get("EP_ROOT", str(REPO_ROOT.parent / "go2-flyball-parkour"))
)

# Student checkpoint shipped in this repo.
CKPT_DEFAULT = (
    REPO_ROOT / "models" / "L00-02-flat-baseline-delay-distillation-noised" / "model_14500.pt"
)


def host_monotonic_ns() -> int:
    """Cross-process monotonic timestamp shared by Python and C++ on macOS."""
    return time.clock_gettime_ns(time.CLOCK_MONOTONIC)
import torch.nn.functional as F

# --- proprio / actor dimensions (frozen for P11-99-distillation) ---------
N_PROPRIO = 53
NUM_SCAN = 132
N_PRIV_EXPLICIT = 9
N_PRIV_LATENT = 29
HISTORY_LEN = 10
NUM_ACTIONS = 12

# Scan/depth latent dim emitted by RecurrentDepthBackbone.output_mlp.
# L00-flatstep student: yaw predictor head removed (was Linear(512, 32+2),
# now Linear(512, 32)). The encoder emits only the 32-dim scan latent; the
# actor's heading slots prop[6:8] are held at 0 instead of 1.5 * yaw_pred.
DEPTH_LATENT_DIM = 32

# obs scales (legged_robot_config.py:114-118 of the snapshot)
SCALE_ANG_VEL = 0.25
SCALE_DOF_POS = 1.0
SCALE_DOF_VEL = 0.05
CLIP_OBS = 100.0
# `normalization.clip_actions = 1.2` in training is the SCALED clip (in rad);
# the raw-action clip is clip_actions / action_scale = 1.2 / 0.25 = 4.8.
# Deploy was using 1.2 directly, giving motors 4× less authority than training.
CLIP_ACTIONS = 4.8
ACTION_SCALE = 0.25

# control (Go2Advanced2Cfg)
KP = 35.0
KD = 0.6
DECIMATION = 4

# depth pipeline (legged_robot_config.py:90-111 of the snapshot)
DEPTH_RAW_W = 106
DEPTH_RAW_H = 60
DEPTH_TARGET_H = 58
DEPTH_TARGET_W = 87
DEPTH_NEAR = 0.0
DEPTH_FAR = 2.0
# MuJoCo-only sensor-floor emulation.  Keep this out of normalize_depth(): the
# latter is also used by the real-camera node, whose close-range policy is set
# by the camera/driver rather than the simulator.
MUJOCO_EMPTY_BELOW_M = 0.15
UPDATE_INTERVAL_DEFAULT = 5
DEPTH_BUFFER_LEN = 2

# Horizontal FOV of the training depth camera (legged_robot_config.py
# depth.horizontal_fov). Kept for the legacy radial-clip compatibility path.
DEPTH_HFOV_DEG = 87.0


def depth_radial_clip_enabled() -> bool:
    """Whether to emulate the pre-8b8ba80 spherical far clip.

    Current training clips perpendicular depth at ``DEPTH_FAR`` (planar clip).
    Keep the old behavior available only for replaying legacy checkpoints.
    """
    return os.environ.get("EP_DEPTH_RADIAL_CLIP", "0") != "0"


def clip_mujoco_too_close_to_far(d: np.ndarray,
                                 threshold_m=MUJOCO_EMPTY_BELOW_M) -> np.ndarray:
    """Map simulated depth strictly closer than ``threshold_m`` to far/empty."""
    d = np.asarray(d)
    return np.where(d < threshold_m, np.float32(DEPTH_FAR), d).astype(np.float32)


def hole_fill(d: np.ndarray, mode: int = 1, max_iter: int = 8) -> np.ndarray:
    """Fill invalid (<=0) pixels in a positive-metres depth frame.

    Mirrors librealsense's rs2_hole_filling_filter:
      0 fill_from_left        row-wise forward fill from the last valid pixel
      1 farthest_from_around  iterative max of valid 8-neighbours  (SDK default)
      2 nearest_from_around   iterative min of valid 8-neighbours

    SINGLE SOURCE OF TRUTH for the deploy path (depth_camera_node) and the
    offline replay (deploy_mac/camera/recompute_depth_bag.py). The 2026-07-28 training
    recalibration was measured against THIS function, so the two must not drift:
    on 2397 real standing frames it takes smooth jitter 0.0128 -> 0.0092 m, big
    jumps 0.704 -> 0.364 %, and grid invalid 2.56 -> 1.04 %, which is what
    go2_cliffside_distill_config's dis_noise/pw_edge_prob/stereo_baseline now
    target. Note it does NOT clear every hole (large or border-touching regions
    survive max_iter), which is why the target is 1.04 % and not 0.

    Applied BEFORE the empty_below_m remap — afterwards there are no holes left
    to fill, they have already become far.
    """
    out = np.asarray(d).astype(np.float32, copy=True)
    if mode == 0:
        valid = out > 0
        idx = np.where(valid, np.arange(out.shape[1], dtype=np.int64)[None, :], -1)
        np.maximum.accumulate(idx, axis=1, out=idx)
        rows = np.arange(out.shape[0])[:, None]
        filled = out[rows, np.where(idx < 0, 0, idx)]
        return np.where(idx >= 0, filled, out).astype(np.float32)

    import cv2  # lazy: keeps cv2 off the import path of non-depth consumers
    k = np.ones((3, 3), np.uint8)
    for _ in range(max_iter):
        inv = out <= 0
        if not inv.any():
            break
        if mode == 1:                                  # farthest = max of neighbours
            cand = cv2.dilate(out, k)
        else:                                          # nearest = min of VALID neighbours
            tmp = np.where(out > 0, out, np.float32(1e6))
            cand = cv2.erode(tmp, k)
            cand = np.where(cand >= 1e6 - 1, np.float32(0.0), cand)
        nxt = np.where(inv & (cand > 0), cand, out)
        if np.array_equal(nxt, out):
            break
        out = nxt
    return out


def radial_maxperp_mask(h=DEPTH_RAW_H, w=DEPTH_RAW_W,
                        hfov_deg=DEPTH_HFOV_DEG, far=DEPTH_FAR):
    """Legacy per-pixel ceiling for the pre-8b8ba80 radial far clip.

    Isaac renders depth with a Warp ray-cast (warp_cam.py): a surface is seen only
    if its EUCLIDEAN range <= far_clip, and the stored value is range*cosθ (the
    perpendicular depth). So a pixel at angle θ off the optical axis saturates to
    `far` once its perpendicular depth z exceeds far*cosθ — a circular far-vignette
    at the corners. MuJoCo's depth render AND a real RealSense both report
    perpendicular z directly, so clipping z at far*cosθ reproduces it. Same pinhole
    model as warp_cam: focal from hfov on width, principal point at the grid centre.
    Apply on the raw render grid (DEPTH_RAW_H x DEPTH_RAW_W) BEFORE crop/resize.
    """
    focal = (w / 2.0) / np.tan(np.radians(hfov_deg) / 2.0)
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    uu, vv = np.meshgrid(np.arange(w), np.arange(h))   # (h, w): col, row indices
    a = (uu - cx) / focal      # +right
    b = (vv - cy) / focal      # +down
    cos_theta = 1.0 / np.sqrt(1.0 + a * a + b * b)     # dot(unit ray, optical axis)
    return (far * cos_theta).astype(np.float32)

# action_delay
ACTION_BUF_LEN = 8
ACTION_DELAY_VIEW = 1

# yaw prediction multiplier (extreme-parkour evaluate.py).
# DEPRECATED for the L00-flatstep student — the yaw predictor head was removed,
# so prop[6:8] is held at 0 and this multiplier is unused on the live path.
# Kept defined for legacy diagnostic tools (deploy_mac/diag/view_prediction.py) that
# still import it for replaying older 34-dim checkpoints.
YAW_PRED_MULTIPLIER = 1.5

# paper §4.1 latency budget
DEPTH_LATENCY_S = 0.080
# paper §4.1: proprioception latency fixed at 0.016 s (sleep after inference to
# enforce a constant sensor→publish latency; prevents jitter). Enforced by default.
PROPRIO_LATENCY_S = 0.016

# Identity permutation — kept for callers that still want the symbol.
# Note: training's proprio uses legged_robot.reindex(.) on Isaac-order joints,
# which produces order [FR, FL, RR, RL] — exactly MuJoCo's actuator block order
# (and the real Go2 SDK's motor_state order). So no permutation is needed when
# building proprio from MJ/SDK data: feed MJ-order inputs directly.
MJ_TO_POLICY = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], dtype=np.int64)

# Default joint angles in TRAINING-PROPRIO order = MJ actuator order =
# [FR, FL, RR, RL] × (hip, thigh, calf). Confirmed by inspecting
# legged_robot.py:121 (env.step does `actions = self.reindex(actions)` before
# storing/applying), legged_robot.py:684 (`self.reindex((dof_pos - default))`),
# go2.xml actuator block (FR,FL,RR,RL), and go2_advanced_config.py:36-51
# (named-joint defaults).
DEFAULT_DOF_POLICY = np.array(
    [-0.1, 0.8, -1.5,  # FR hip thigh calf
      0.1, 0.8, -1.5,  # FL
     -0.1, 1.0, -1.5,  # RR
      0.1, 1.0, -1.5], # RL
    dtype=np.float64,
)

# Legacy foot-force sensor order in go2.xml (FL,FR,RL,RR) → MJ/SDK order
# (FR,FL,RR,RL). Kept for older diagnostic runners; distill_runner and the ROS
# bridge now both compute resultant contact load directly from foot contacts.
FOOT_SENSOR_TO_MJ = np.array([1, 0, 3, 2], dtype=np.int64)


def add_extreme_parkour_to_path(ep_root: Path) -> None:
    """Make `from rsl_rl.modules.* import *` work without installing rsl_rl."""
    ep_root = Path(ep_root)
    rsl = str(ep_root / "rsl_rl")
    lg = str(ep_root / "legged_gym")
    if rsl not in sys.path:
        sys.path.insert(0, rsl)
    if lg not in sys.path:
        sys.path.insert(0, lg)
    # depth_backbone.py imports torchvision unconditionally but we don't use
    # the parts that need it; stub when ABI mismatched.
    if "torchvision" not in sys.modules:
        import types as _types
        sys.modules["torchvision"] = _types.ModuleType("torchvision")


def load_checkpoint(ckpt_path: Path):
    try:
        return torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except Exception:
        return torch.load(ckpt_path, map_location="cpu", weights_only=False)


def detect_action_delay(ckpt_path: Path) -> int:
    """Look for legged_gym_src/.../legged_robot_config.py next to the ckpt."""
    src_root = Path(ckpt_path).parent / "legged_gym_src"
    cfg = src_root / "envs" / "base" / "legged_robot_config.py"
    if not cfg.exists():
        return 0
    text = cfg.read_text()
    m = re.search(r"action_delay\s*=\s*(True|False)", text)
    if m is None:
        return 0
    return 1 if m.group(1) == "True" else 0


def mj_quat_to_roll_pitch_yaw(q_wxyz):
    """(w, x, y, z) → (roll, pitch, yaw) ZYX Euler."""
    w, x, y, z = q_wxyz
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def resize_depth_bicubic(t: torch.Tensor) -> torch.Tensor:
    """Resize a [N,1,H,W] (or [N,H,W] via caller) depth tensor to the policy's
    58x87 exactly like training's `torchvision.transforms.Resize(BICUBIC)`.

    ANTIALIAS: training runs torchvision 0.11.1 (install.sh of the extreme-
    parkour repo pins torch 1.10.0 / torchvision 0.11.1+cu113), where
    Resize on a TENSOR is a plain `interpolate(..., align_corners=False)` with
    no antialias filter — torchvision only gained tensor antialias support
    later and flipped its default to True in 0.17. So antialias=False is the
    training-matched setting and we state it explicitly instead of relying on
    the F.interpolate default. If a training pod is ever moved to
    torchvision >= 0.17, re-check with
        python -c "import torchvision; print(torchvision.__version__)"
    and flip this flag to keep deploy aligned.
    """
    return F.interpolate(t, size=(DEPTH_TARGET_H, DEPTH_TARGET_W),
                         mode="bicubic", align_corners=False, antialias=False)


def normalize_depth(d: np.ndarray, depth_noise_m: float = 0.0) -> torch.Tensor:
    """Raw positive-metres depth → normalized (1, 58, 87) tensor in [-0.5, 0.5].

    Mirrors the Isaac training pipeline in legged_robot.py:167-178:
      crop[:-2, 4:-4]  → +per-frame-scalar noise  → clip[near, far]
      → BICUBIC resize to (58, 87) → normalize / range - 0.5.

    Two corrections vs the older deploy implementation (2026-05-24):
      1. interpolation = BICUBIC (matches torchvision.Resize(BICUBIC) in
         legged_robot.py:101-102). Was bilinear, which produces smoother
         outputs at depth edges and shifts the encoder off-distribution.
      2. Noise is PER-FRAME SCALAR (one random shift applied to all
         pixels), matching legged_robot.py:170:
            `depth_image += dis_noise * 2 * (rand-0.5)[0]`
         The earlier per-pixel uniform noise was statistically wrong —
         training never saw pixel-independent jitter, only frame-level
         depth bias drift.

    `depth_noise_m` defaults to 0.0 to preserve byte-for-byte the
    clean-render behaviour; pass 0.02 to match the training cfg
    `cfg.depth.dis_noise = 0.02 m`.
    """
    if d.shape == (DEPTH_RAW_H, DEPTH_RAW_W):
        d = d[:-2, 4:-4]                          # crop first: (58, 98)
    # Preserve valid close-range readings, including pixels below 0.15 m. Only
    # non-positive no-return/invalid values become empty = far. Do this before
    # resize so actual sensor holes do not smear into mid-range values.
    d = np.where(d <= 0.0, np.float32(DEPTH_FAR), d).astype(np.float32)
    t = torch.from_numpy(d)[None, None]
    if depth_noise_m > 0.0:
        # Per-frame scalar — single random shift for all pixels in this frame.
        shift = depth_noise_m * 2.0 * (torch.rand(1).item() - 0.5)
        t = t + shift
    t = torch.clamp(t, DEPTH_NEAR, DEPTH_FAR)     # clip after noise (Isaac order)
    t = resize_depth_bicubic(t)
    # Second clip AFTER the resize, matching legged_robot.py:428 / :497 — bicubic
    # overshoots at depth edges, so without this a real/clean-sim frame can carry
    # pixels outside [near, far] (normalized outside [-0.5, 0.5]) that training,
    # which always re-clips here, never produced.
    t = torch.clamp(t, DEPTH_NEAR, DEPTH_FAR)
    t = t.squeeze(0)
    return t / (DEPTH_FAR - DEPTH_NEAR) - 0.5


# --- training-matched depth degradation (Warp camera noise) -----------------


class MujocoStereoDepth:
    """D435 stereo visibility model equivalent to WarpDepthCamera's 2nd ray.

    MuJoCo supplies the clean left-imager perpendicular depth. This class
    reconstructs each hit point, moves the ray origin ``baseline`` metres along
    camera +X (the physical right imager), and uses ``mj_multiRay`` to test
    whether that point is visible there. Points outside the right imager's FOV
    or hidden by nearer geometry are invalidated to ``far``.

    Current Warp training clips perpendicular depth, matching MuJoCo directly.
    ``radial_clip=True`` retains the pre-8b8ba80 spherical range cap solely for
    legacy checkpoint replay. Apply this model to the raw 60x106 render before
    crop/resize and image-space degradation.
    """

    def __init__(self, model, camera="front_depth_cam", height=DEPTH_RAW_H,
                 width=DEPTH_RAW_W, hfov_deg=DEPTH_HFOV_DEG,
                 far=DEPTH_FAR, baseline=0.060, eps=0.015,
                 radial_clip=False):
        import mujoco

        self._mujoco = mujoco
        self.model = model
        self.height = int(height)
        self.width = int(width)
        self.far = float(far)
        self.radial_clip = bool(radial_clip)
        self.baseline = float(baseline)
        self.eps = float(eps)
        if isinstance(camera, str):
            camera = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
        self.camera_id = int(camera)
        if self.camera_id < 0 or self.camera_id >= model.ncam:
            raise ValueError(f"invalid MuJoCo camera id: {self.camera_id}")

        self.focal = (self.width / 2.0) / np.tan(
            np.radians(float(hfov_deg)) / 2.0)
        cx, cy = (self.width - 1) / 2.0, (self.height - 1) / 2.0
        uu, vv = np.meshgrid(
            np.arange(self.width, dtype=np.float64),
            np.arange(self.height, dtype=np.float64),
        )
        self._uu = uu
        self._x_over_z = (uu - cx) / self.focal
        self._y_over_z = -(vv - cy) / self.focal
        self._range_factor = (np.sqrt(
            1.0 + self._x_over_z ** 2 + self._y_over_z ** 2)
            if self.radial_clip else None)
        self.last_fov_mask = np.zeros((self.height, self.width), dtype=bool)
        self.last_occlusion_mask = np.zeros_like(self.last_fov_mask)
        # Warp casts against the terrain mesh only. MuJoCo's robot visuals and
        # collision geoms are consistently assigned groups 2 and 3; environment
        # scenes use group 0 or 4.
        self._geomgroup = np.ascontiguousarray(
            np.array([1, 1, 0, 0, 1, 1], dtype=np.uint8))
        self._multi_ray_ok = True

    def _ray_distances(self, data, origin, directions, max_dist):
        """Return terrain hit distances from a shared origin for unit rays."""
        nray = int(directions.shape[0])
        hit_dist = np.empty(nray, dtype=np.float64)
        geom_id = np.empty(nray, dtype=np.int32)
        body_exclude = int(self.model.cam_bodyid[self.camera_id])
        origin = np.ascontiguousarray(origin, dtype=np.float64)
        flat_directions = np.ascontiguousarray(
            directions, dtype=np.float64).reshape(-1)
        if self._multi_ray_ok:
            try:
                ray_args = (
                    self.model, data, origin, flat_directions, self._geomgroup,
                    True, body_exclude, geom_id, hit_dist,
                )
                try:
                    # Newer bindings include an optional data argument before
                    # nray/cutoff.
                    self._mujoco.mj_multiRay(
                        *ray_args, None, nray, float(max_dist))
                except TypeError:
                    # MuJoCo 3.2 exposes the original 11-argument binding.
                    self._mujoco.mj_multiRay(
                        *ray_args, nray, float(max_dist))
                return hit_dist
            except TypeError:
                self._multi_ray_ok = False

        one_geom = np.empty(1, dtype=np.int32)
        for i, direction in enumerate(np.asarray(directions, dtype=np.float64)):
            ray_args = (
                self.model, data, origin, np.ascontiguousarray(direction),
                self._geomgroup, True, body_exclude, one_geom,
            )
            try:
                # MuJoCo >= 3.3 accepts an optional cutoff distance.
                hit_dist[i] = self._mujoco.mj_ray(*ray_args, None)
            except TypeError:
                # MuJoCo 3.2 and older expose the original 8-argument binding.
                hit_dist[i] = self._mujoco.mj_ray(*ray_args)
        return hit_dist

    def __call__(self, data, left_depth):
        depth = np.asarray(left_depth, dtype=np.float32)
        expected = (self.height, self.width)
        if depth.shape != expected:
            raise ValueError(f"expected raw depth shape {expected}, got {depth.shape}")
        depth = depth.copy()

        # Current training uses a planar perpendicular-depth far clip. The
        # optional range-factor branch reproduces the old spherical cap.
        left_hit = np.isfinite(depth) & (depth > 0.0) & (depth <= self.far)
        if self.radial_clip:
            left_hit &= depth * self._range_factor <= self.far
        depth[~left_hit] = np.float32(self.far)

        # Rectified stereo FOV constraint from the training kernel:
        # u_R = u_L - focal * baseline / z.
        with np.errstate(divide="ignore", invalid="ignore"):
            right_u = self._uu - self.focal * self.baseline / depth
        fov_invalid = left_hit & (right_u < 0.0)
        candidates = left_hit & ~fov_invalid

        self.last_fov_mask[:] = fov_invalid
        self.last_occlusion_mask[:] = False
        if not np.any(candidates):
            depth[fov_invalid] = np.float32(self.far)
            return depth

        z = depth[candidates].astype(np.float64)
        points_cam = np.column_stack((
            self._x_over_z[candidates] * z,
            self._y_over_z[candidates] * z,
            -z,
        ))
        cam_rot = np.asarray(data.cam_xmat[self.camera_id], dtype=np.float64).reshape(3, 3)
        left_origin = np.asarray(data.cam_xpos[self.camera_id], dtype=np.float64)
        points_world = left_origin + points_cam @ cam_rot.T
        right_origin = left_origin + cam_rot[:, 0] * self.baseline
        to_point = points_world - right_origin
        target_dist = np.linalg.norm(to_point, axis=1)
        directions = to_point / target_dist[:, None]

        hit_dist = self._ray_distances(
            data, right_origin, directions, float(target_dist.max()))
        occluded = (hit_dist >= 0.0) & (hit_dist < target_dist - self.eps)
        candidate_indices = np.flatnonzero(candidates)
        occluded_indices = candidate_indices[occluded]
        self.last_occlusion_mask.flat[occluded_indices] = True

        invalid = fov_invalid | self.last_occlusion_mask
        depth[invalid] = np.float32(self.far)
        return depth


def mujoco_depth_scene_option():
    """Render environment depth without robot visual/collision geometry.

    Excluding the robot matches TRAINING: Isaac's warp_cam ray-casts against the
    terrain triangle mesh only (warp_cam.py set_mesh/mesh_query_ray), so the
    student never saw its own body in a depth frame. The real D435 has no such
    exclusion, but a floor-plane check on the 2026-07-27 bags found essentially
    no self-occlusion in the standing pose (bottom 40 rows: 0.2% of frames read
    in front of the floor), so all three pipelines agree in practice.

    EP_DEPTH_SELF_VIEW=1 renders the robot's visual meshes (group 2) into its own
    depth image, to measure what self-occlusion would cost if the mount or pose
    ever put the legs in frame. Group 3 (collision) stays off — group 2 is the
    real silhouette.
    """
    import mujoco

    option = mujoco.MjvOption()
    option.geomgroup[:] = 1
    option.geomgroup[2] = 1 if os.environ.get("EP_DEPTH_SELF_VIEW", "0") != "0" else 0
    option.geomgroup[3] = 0
    return option


# Source of truth: extreme-parkour at 8b8ba80. These are the inherited
# Go2RoughCfg depth values plus go2_cliffside_distill_config.depth overrides.
# In particular the geometric stereo model replaces the old heuristic blind
# band, so pw_blindspot_cols stays zero.
# RESYNCED 2026-08-06 to extreme-parkour 8b8ba80 (planar far clipping).
# The image-noise constants remain the f76aa69 calibration used by A1. The
# previous 90eb887 values are kept in the comments below
# because every sim2sim run before this date used them:
#     dis_noise 0.05  noise_gaussian 0.006  noise_dropout 0.00125
#     pw_edge_prob 0.18  pw_hole_prob 0.002  stereo_baseline 0.060
# Those were measured (2026-07-28, hole-filled real bags at the 58x87 policy
# grid) to be 3-20x harsher than the real D435; the values below are the
# real-matched calibration that f76aa69 trains with.
TRAIN_DEPTH_SOURCE_COMMIT = "8b8ba80"
TRAIN_DEPTH_CFG = dict(
    dis_noise=0.002,
    noise_gaussian=0.005,
    noise_dropout=0.0006,
    dropout_cluster=True,
    dropout_cluster_ext=0.55,
    dropout_jitter=0.8,
    dropout_near_prob=0.4,
    pw_noise=True,
    pw_empty_min=0.0,
    pw_edge_thresh=0.50,
    pw_edge_prob=0.009,
    pw_edge_near_frac=0.03,
    pw_edge_far_frac=0.90,
    pw_edge_dilate=1,
    pw_hole_prob=0.001,
    pw_hole_scale=8,
    pw_hole_drift=0.1,
    pw_blindspot_cols=0,
    pw_blindspot_side="left",
    pw_blindspot_drift=0.05,
    pw_blindspot_slope_max=0.3,
    pw_blindspot_jitter=1.0,
    pw_blur_sigma=0.0,
    stereo_shadow=True,
    stereo_baseline=0.010,
    stereo_eps=0.015,
)

# EP_STEREO_BASELINE overrides the stereo baseline (metres) for BOTH the ROS
# depth_sim_node and render_sim2sim_noros, which each read this dict.
#
# WHY THIS EXISTS (measured 2026-07-28): the 0.060 m value is the D435's physical
# baseline and the model is geometrically correct, but it is ALGORITHMICALLY
# pessimistic — it invalidates every geometrically occluded pixel, whereas real
# stereo hardware recovers most of them by block-matching, and the real pipeline
# then dilutes what remains through the 4.4x downsample to the 106-wide grid.
# Measured on the 2026-07-27 bags vs the same 60x106 grid:
#     real D435   : 2.70% invalid at 480x270  ->  1.30% dominant on 60x106
#     sim @0.060 m: 4.86% mean (p95 11.6%, max 17.4%)   ~3.7x too much
# Sim invalidation vs baseline: .060->5.4%  .030->2.8%  .020->2.0%  .015->1.5%
# .010->1.1%, so ~0.0125 m reproduces the real rate. Leave the default at the
# training value; set the env var only for deliberate real-matched experiments.
_bl = os.environ.get("EP_STEREO_BASELINE")
if _bl:
    TRAIN_DEPTH_CFG["stereo_baseline"] = float(_bl)

# EP_DEPTH_CFG overrides ANY numeric key above, e.g. to reproduce a run made
# against the pre-2026-07-28 (90eb887) calibration:
#   EP_DEPTH_CFG='dis_noise=0.05,pw_edge_prob=0.18,stereo_baseline=0.060'
for _kv in filter(None, os.environ.get("EP_DEPTH_CFG", "").split(",")):
    _k, _, _v = _kv.partition("=")
    _k = _k.strip()
    if _k not in TRAIN_DEPTH_CFG:
        raise ValueError(f"EP_DEPTH_CFG: unknown key {_k!r}")
    TRAIN_DEPTH_CFG[_k] = type(TRAIN_DEPTH_CFG[_k])(_v)


class TrainDepthDegrader:
    """Apply the latest cliffside Warp depth degradation to one live frame.

    Call with raw positive-metres depth (DEPTH_RAW_H x DEPTH_RAW_W or already
    cropped); returns the normalized (1, 58, 87) tensor in [-0.5, 0.5], same
    contract as normalize_depth. ``noise_scale=0.5`` is the reality-tuned model;
    scales above 0.5 keep hard artifacts and raise rates/amplitudes up to 2x.

    Stereo visibility is geometry-dependent and must be applied to the raw
    MuJoCo render with :class:`MujocoStereoDepth` before calling this class.
    """

    def __init__(self, cfg=None, noise_scale: float = 1.0):
        self.cfg = dict(TRAIN_DEPTH_CFG)
        if cfg:
            self.cfg.update(cfg)
        if not 0.0 <= noise_scale <= 1.0:
            raise ValueError(f"noise_scale must be in [0, 1], got {noise_scale}")
        self.noise_scale = float(noise_scale)
        self.far = float(DEPTH_FAR)
        self.near = float(DEPTH_NEAR)
        self._pw_low = None          # drifting low-res hole field
        self._blur = None            # cached blur kernel
        self._blur_pad = 0
        self.bs_k = torch.zeros(1)   # blind-spot band width (drifts)
        sm = float(self.cfg["pw_blindspot_slope_max"])
        self.bs_slope = (torch.rand(1) * 2 - 1) * sm
        self.bs_side = torch.tensor([self.cfg["pw_blindspot_side"] == "right"])

    def _blur_batch(self, depth, sigma):
        r = max(1, int(round(3.0 * sigma)))
        if self._blur is None:
            x = torch.arange(-r, r + 1, dtype=depth.dtype)
            k1 = torch.exp(-(x ** 2) / (2.0 * sigma ** 2)); k1 = k1 / k1.sum()
            self._blur = (k1[:, None] * k1[None, :]).view(1, 1, 2 * r + 1, 2 * r + 1)
            self._blur_pad = r
        d = F.conv2d(depth.unsqueeze(1), self._blur, padding=self._blur_pad)
        return d.squeeze(1)

    def _hole_field(self, N, H, W, dtype):
        scale = max(1, int(self.cfg["pw_hole_scale"]))
        lh, lw = max(1, H // scale), max(1, W // scale)
        drift = float(self.cfg["pw_hole_drift"])
        new_low = torch.rand(N, 1, lh, lw, dtype=dtype)
        if self._pw_low is None or self._pw_low.shape != new_low.shape:
            self._pw_low = new_low
        else:
            self._pw_low = (1.0 - drift) * self._pw_low + drift * new_low
        return F.interpolate(self._pw_low, size=(H, W), mode="bilinear",
                             align_corners=False).squeeze(1)

    def _pw_noise(self, depth, rate_factor):  # depth [N,H,W], negative convention
        if not self.cfg["pw_noise"]:
            return depth
        far = -self.far; near = -self.near; N, H, W = depth.shape
        emin = float(self.cfg["pw_empty_min"])
        if emin > 0.0:
            depth = torch.where((depth.abs() < emin) | (depth == 0),
                                torch.full_like(depth, far), depth)
        et = float(self.cfg["pw_edge_thresh"])
        if et > 0.0:
            gx = torch.zeros_like(depth); gy = torch.zeros_like(depth)
            gx[:, :, 1:] = (depth[:, :, 1:] - depth[:, :, :-1]).abs()
            gy[:, 1:, :] = (depth[:, 1:, :] - depth[:, :-1, :]).abs()
            edge = (gx + gy) > et
            dil = int(self.cfg["pw_edge_dilate"])
            if dil > 0:
                edge = F.max_pool2d(
                    edge.float().unsqueeze(1), 2 * dil + 1,
                    stride=1, padding=dil).squeeze(1) > 0.5
            ep = min(1.0, float(self.cfg["pw_edge_prob"]) * rate_factor)
            nf = float(self.cfg["pw_edge_near_frac"]); ff = float(self.cfg["pw_edge_far_frac"])
            hit = edge & (torch.rand_like(depth) < ep); r = torch.rand_like(depth)
            to_near = hit & (r < nf)
            to_far = hit & (r >= nf) & (r < nf + ff)
            shuf = hit & (r >= nf + ff)
            rolled = torch.roll(depth, shifts=1, dims=2)
            depth = torch.where(to_near, torch.full_like(depth, near), depth)
            depth = torch.where(to_far, torch.full_like(depth, far), depth)
            depth = torch.where(shuf, rolled, depth)
        hp = float(self.cfg["pw_hole_prob"])
        if hp > 0.0:
            field = self._hole_field(N, H, W, depth.dtype)
            thr = torch.quantile(field.flatten(1), 1.0 - hp, dim=1).view(N, 1, 1)
            depth = torch.where(field > thr, torch.full_like(depth, far), depth)
        kmax = int(self.cfg["pw_blindspot_cols"])
        if kmax > 0:
            drift = float(self.cfg["pw_blindspot_drift"])
            target = torch.rand(N) * kmax
            self.bs_k[:N] = (1.0 - drift) * self.bs_k[:N] + drift * target
            rows = torch.arange(H, dtype=torch.float32)
            k_per_row = (self.bs_k[:N, None] + self.bs_slope[:N, None] * (rows[None, :] - H / 2.0))
            jrag = float(self.cfg["pw_blindspot_jitter"])
            if jrag > 0.0:
                k_per_row = k_per_row + (torch.rand(N, H) - 0.5) * 2.0 * jrag
            k_per_row = k_per_row.round().clamp(0, W)
            col = torch.arange(W, dtype=torch.float32)
            left = col[None, None, :] < k_per_row[:, :, None]
            right = col[None, None, :] >= (W - k_per_row[:, :, None])
            mask = torch.where(self.bs_side[:N, None, None], right, left)
            depth = torch.where(mask, torch.full_like(depth, far), depth)
        sigma = float(self.cfg["pw_blur_sigma"])
        if sigma > 0.0:
            depth = self._blur_batch(depth, sigma)
        return depth

    def _mgdp_noise(self, depth, rate_factor):  # [N,H,W], negative convention
        g = float(self.cfg["noise_gaussian"]); p = float(self.cfg["noise_dropout"])
        far_v = -self.far; near_v = -self.near
        if g > 0.0:
            depth = depth + torch.randn_like(depth) * (g * rate_factor * depth)
        if p > 0.0 and self.cfg["dropout_cluster"]:
            ext = float(self.cfg["dropout_cluster_ext"])
            valid = depth > (far_v + 0.05)
            seeds = valid & (torch.rand_like(depth) < p * rate_factor)
            add1 = seeds & (torch.rand_like(depth) < ext)
            add2 = add1 & (torch.rand_like(depth) < ext)
            blk = seeds & (torch.rand_like(depth) < float(self.cfg["dropout_near_prob"]))
            white = seeds.clone()
            white[..., :, 1:] |= add1[..., :, :-1]
            white[..., :, 2:] |= add2[..., :, :-2]
            black = torch.zeros_like(seeds)
            black[..., :, :-1] |= blk[..., :, 1:]
            black &= ~white
            jit = float(self.cfg["dropout_jitter"])
            white_fill = far_v + torch.rand_like(depth) * jit
            near_fill = near_v - torch.rand_like(depth) * jit
            depth = torch.where(white, white_fill, depth)
            depth = torch.where(black, near_fill, depth)
        elif p > 0.0:
            drop = torch.rand_like(depth) < p * rate_factor
            depth = torch.where(drop, torch.full_like(depth, far_v), depth)
        return depth

    def __call__(self, d: np.ndarray) -> torch.Tensor:
        if d.shape == (DEPTH_RAW_H, DEPTH_RAW_W):
            d = d[:-2, 4:-4]
        t = torch.from_numpy(np.ascontiguousarray(d)).float()[None]  # [1,h,w] positive
        dn = -t  # to Isaac negative-in-front
        clean = torch.clamp(dn, -self.far, -self.near)
        clean = resize_depth_bicubic(clean[None])[0]
        blend = min(2.0 * self.noise_scale, 1.0)
        rate_factor = max(2.0 * self.noise_scale, 1.0)
        dis = float(self.cfg["dis_noise"])
        if dis > 0.0:
            dn = dn + dis * rate_factor * 2.0 * (torch.rand(1).item() - 0.5)
        dn = torch.clamp(dn, -self.far, -self.near)
        dn = resize_depth_bicubic(dn[None])[0]
        dn = self._pw_noise(dn, rate_factor)
        dn = self._mgdp_noise(dn, rate_factor)
        dn = (1.0 - blend) * clean + blend * dn
        dn = torch.clamp(dn, -self.far, -self.near)
        dp = -dn  # back to positive metres
        return dp / (self.far - self.near) - 0.5


# Per-foot |F| threshold + OR with the previous policy step (matches Isaac
# legged_robot.py:332-334). Caller owns the previous-step state. The default
# is 2.0 N — the exact training value (legged_robot.py: contact_forces L2 norm
# > 2.0). This is only correct paired with mujoco_simulator publishing the NET
# ground-contact force (mj_contactForce over the foot geoms, ~0 in swing); the
# old go2.xml <force> sensor floored at ~2.4 N in swing, so a matching 2.0 N
# threshold read "always in contact". Parkour-class checkpoints (P10-*, P11-*,
# L00-01 wall variants) override per-foot; see distill_runner.py near
# `args.contact_threshold` for the full rationale.
CONTACT_FORCE_THRESHOLD_N = 2.0

# Default raw-sensor thresholds (FR, FL, RR, RL) for the real Go2. Keep this a
# four-value tuple so callers can pass it directly to the per-foot contact path.
# This is deliberately separate from CONTACT_FORCE_THRESHOLD_N=2.0, which is
# the MuJoCo/training-domain net-force threshold.
CONTACT_FORCE_THRESHOLD_PER_FOOT_REAL = (20.0, 20.0, 20.0, 20.0)


def parse_contact_threshold(spec):
    """Parse a --contact_threshold value into a scalar or length-4 per-foot list.

    Accepts a float, or a string that is either one value ("20") or four
    comma-separated per-foot values in FR,FL,RR,RL order ("10,20,20,17").
    Returns a float (global) or a list[float] of length 4 (per-foot), ready to
    pass straight to compute_contact_filt's `threshold_n`. Raises ValueError on
    any other count so a typo fails loudly instead of silently desyncing the
    encoder and the actor.
    """
    if spec is None:
        return CONTACT_FORCE_THRESHOLD_N
    if isinstance(spec, (int, float)):
        return float(spec)
    vals = [float(x) for x in str(spec).split(",") if x.strip() != ""]
    if len(vals) == 1:
        return vals[0]
    if len(vals) == 4:
        return vals
    raise ValueError(
        f"--contact_threshold must be 1 value (global) or 4 (FR,FL,RR,RL); got {spec!r}")


def compute_contact_filt(foot_forces: np.ndarray,
                         last_raw_contact: np.ndarray,
                         threshold_n=None):
    """Return Isaac-style filtered foot contact + the raw thresholded contact.

    Inputs are per-foot scalar force magnitudes (Newtons), MJ/SDK foot order
    (FR, FL, RR, RL), and the PREVIOUS step's RAW thresholded contact (not the
    filtered output — training stores `self.last_contacts = contact` at
    legged_robot.py:334, not `self.contact_filt`).

    `threshold_n` lets the caller override the module-level default — set
    this from a launch arg so the depth encoder and the policy node stay
    in lockstep on the threshold (a mismatch silently desyncs proprio[49:53]
    between the encoder's forward pass and the actor's). When omitted, the
    Isaac/MuJoCo net-contact value of 2.0 N is used. It may be a scalar (one global
    threshold) OR a length-4 array of per-foot thresholds in (FR, FL, RR, RL)
    order — see CONTACT_FORCE_THRESHOLD_PER_FOOT_REAL for the real-robot values.

    Returns `(filt, raw)` so callers can store `raw` for the next call. Using
    the filtered value latches the bit permanently once it ever fires.
    """
    t = (CONTACT_FORCE_THRESHOLD_N if threshold_n is None
         else np.asarray(threshold_n, dtype=np.float64))
    contact_now = (foot_forces > t).astype(np.float64)
    filt = np.maximum(contact_now, last_raw_contact.astype(np.float64))
    return filt, contact_now


# --- proprio observation noise (sim2sim only; OFF by default) ---------------
# EP_PROPRIO_NOISE = multiplier on the TRAINING noise vector. 0 (default) keeps
# the historic behaviour: sim2sim has always fed perfectly clean proprio, so
# every sweep before 2026-07-28 was implicitly an EP_PROPRIO_NOISE=0 run.
# 1.0 reproduces Isaac exactly.
#
# Isaac (legged_robot._get_proprio_noise_scale_vec + :1599) adds UNIFORM
# +-noise_scale_vec to the assembled proprio obs, once per step, in SCALED obs
# space. Only real sensors are perturbed; commands/flags/last-action/contact get
# zero. With the cliffside noise_scales (ang_vel .05, dof_pos .01, dof_vel .05,
# rotation .05) and obs_scales (ang_vel .25, dof_pos 1, dof_vel .05):
#     prop[0:3]   ang_vel  +-0.0125      (= 0.05 rad/s unscaled)
#     prop[3:5]   roll/pitch  +-0.05 rad (UNSCALED — obs carries raw roll/pitch)
#     prop[13:25] dof_pos  +-0.01 rad
#     prop[25:37] dof_vel  +-0.0025      (= 0.05 rad/s unscaled)
# CORRECTED 2026-07-28: rotation was mirrored here as 0.0, matching the BASE
# LeggedRobotCfg. Go2RoughCfg overrides it to 0.05 (go2_config.py:281) and it is
# applied to the roll/pitch slots (legged_robot.py:1625). The 2026-07-28 proprio
# sweep therefore ran with roll/pitch noise pinned at ZERO in every arm — its
# "flat across a 17x span" null does NOT cover this channel.
# NOTE this obs layout carries roll/pitch, not a gravity vector, so
# noise_scales.gravity (and .lin_vel) never reach this policy at all.
PROPRIO_NOISE_SCALE = float(os.environ.get("EP_PROPRIO_NOISE", "0.0"))
# Training values (go2 cliffside noise_scales), in RAW sensor units.
_NS = dict(ang_vel=0.05, rotation=0.05, dof_pos=0.01, dof_vel=0.05)
# EP_PROPRIO_NS overrides individual channels, e.g.
#   EP_PROPRIO_NS='dof_pos=0.0041,dof_vel=0.4323,ang_vel=0.2264'
# Needed because the real robot's noise mix is NOT a uniform multiple of the
# training vector: measured on the 2026-07-27 hurdle-jumping bags (6 windows,
# spectral floor 150-250 Hz, converted to a uniform half-width via std*sqrt(3)),
# dof_pos wants 0.4x the trained value while dof_vel wants 8.6x.
for _kv in filter(None, os.environ.get("EP_PROPRIO_NS", "").split(",")):
    _k, _, _v = _kv.partition("=")
    _k = _k.strip()
    if _k not in _NS:
        raise ValueError(f"EP_PROPRIO_NS: unknown channel {_k!r}; expected {sorted(_NS)}")
    _NS[_k] = float(_v)
_PROP_NOISE_CACHE = None


def _proprio_noise_vec() -> np.ndarray:
    global _PROP_NOISE_CACHE
    if _PROP_NOISE_CACHE is None:
        v = np.zeros(N_PROPRIO, dtype=np.float64)
        v[0:3]   = _NS["ang_vel"] * SCALE_ANG_VEL
        v[3:5]   = _NS["rotation"]
        v[13:25] = _NS["dof_pos"] * SCALE_DOF_POS
        v[25:37] = _NS["dof_vel"] * SCALE_DOF_VEL
        _PROP_NOISE_CACHE = v
    return _PROP_NOISE_CACHE


def build_prop(qj_policy: np.ndarray,
               dqj_policy: np.ndarray,
               ang_vel_body: np.ndarray,
               roll: float,
               pitch: float,
               last_action_pol: np.ndarray,
               contact_filt: np.ndarray,
               lin_vel_x: float,
               env_class_neq_17: float = 1.0,
               env_class_eq_17: float = 0.0) -> np.ndarray:
    """Assemble the 53-D proprio frame (yaw slots left zero — caller fills)."""
    prop = np.zeros(N_PROPRIO, dtype=np.float64)
    prop[0:3]   = ang_vel_body * SCALE_ANG_VEL
    prop[3]     = roll
    prop[4]     = pitch
    prop[5]     = 0.0                        # 0 * delta_yaw
    prop[6]     = 0.0                        # delta_yaw (filled by caller)
    prop[7]     = 0.0                        # delta_next_yaw (filled by caller)
    prop[8]     = 0.0                        # 0 * commands[0]
    prop[9]     = 0.0                        # 0 * commands[1]
    prop[10]    = lin_vel_x
    prop[11]    = env_class_neq_17
    prop[12]    = env_class_eq_17
    prop[13:25] = (qj_policy - DEFAULT_DOF_POLICY) * SCALE_DOF_POS
    prop[25:37] = dqj_policy * SCALE_DOF_VEL
    prop[37:49] = last_action_pol
    prop[49:53] = contact_filt - 0.5
    if PROPRIO_NOISE_SCALE > 0.0:
        prop += PROPRIO_NOISE_SCALE * _proprio_noise_vec() * (
            2.0 * np.random.rand(N_PROPRIO) - 1.0)
    return prop


def assemble_actor_obs(prop_t: torch.Tensor,
                       priv_explicit: torch.Tensor,
                       history_ring: torch.Tensor) -> torch.Tensor:
    """Build the 114-D actor input from its parts.

    Layout: [prop(53), zeros(132 scan), priv_expl(9), zeros(29 priv_lat),
             history(10*53)]. The actor consumes scandots_latent via kwarg
             and ignores the scan/priv_lat slots when hist_encoding=True.
    """
    total = N_PROPRIO + NUM_SCAN + N_PRIV_EXPLICIT + N_PRIV_LATENT + HISTORY_LEN * N_PROPRIO
    obs = torch.zeros(1, total, dtype=prop_t.dtype, device=prop_t.device)
    obs[:, 0:N_PROPRIO] = prop_t
    s = N_PROPRIO + NUM_SCAN
    obs[:, s:s + N_PRIV_EXPLICIT] = priv_explicit
    obs[:, -HISTORY_LEN * N_PROPRIO:] = history_ring.reshape(1, -1)
    return obs.clamp_(-CLIP_OBS, CLIP_OBS)
