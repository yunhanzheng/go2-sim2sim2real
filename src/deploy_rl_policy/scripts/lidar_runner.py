#!/usr/bin/env python3
"""
Sim-to-sim verification of the extreme-parkour BASE (teacher) policy in MuJoCo,
fed with raycast-derived scandots in place of the depth pipeline.

The base policy (`P11-40-base_4090` etc.) was trained with privileged 132-dim
height scandots arranged on a 12×11 body-frame grid (x ∈ [-0.45, 1.20], y ∈
[-0.75, 0.75], 0.15 m spacing). At deploy time the teacher's scandots are
unavailable on hardware; we approximate them here by raycasting the MuJoCo
scene straight down at each grid point. The resulting obs follows the Isaac
formula `clip(root_z - 0.3 - terrain_z, -1, 1)`.

This is a clean control variant for `distill_runner.py` (which feeds the
DISTILL student a depth-encoder latent). Use it to confirm whether observed
deploy-side issues come from the distillation pipeline or from the actor
itself.
"""

import argparse
import os
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]


# src/deploy_rl_policy/scripts -> repo root
_REPO_ROOT = Path(__file__).resolve().parents[3]
# The training repo supplies the policy classes:
# https://github.com/Roman-089/go2-flyball-parkour
# Defaults to a sibling clone; override with $EP_ROOT or --ep_root.
_EP_ROOT_DEFAULT = Path(
    os.environ.get("EP_ROOT", str(_REPO_ROOT.parent / "go2-flyball-parkour"))
)
_CKPT_DEFAULT = (
    _REPO_ROOT / "models" / "L00-02-flat-baseline-delay-distillation-noised" / "model_14500.pt"
)

def add_extreme_parkour_to_path(ep_root: Path) -> None:
    sys.path.insert(0, str(ep_root / "rsl_rl"))
    sys.path.insert(0, str(ep_root / "legged_gym"))


# --- proprio / actor dimensions (frozen for P11-40-base_4090) ------------
N_PROPRIO = 53
NUM_SCAN = 132
N_PRIV_EXPLICIT = 9
N_PRIV_LATENT = 29
HISTORY_LEN = 10
NUM_ACTIONS = 12

# obs scales
SCALE_ANG_VEL = 0.25
SCALE_DOF_POS = 1.0
SCALE_DOF_VEL = 0.05
CLIP_OBS = 100.0
# Isaac (legged_robot.py:136-137) clips raw actor outputs at
#   normalization.clip_actions / control.action_scale = 1.2 / 0.25 = 4.8.
# i.e. it caps the *post-scale* action at ±1.2. The earlier distill_runner.py
# used 1.2 here, which silently capped post-scale actions at ±0.3 — a 4×
# underactuation that the depth student tolerated but the base teacher (with
# larger action amplitudes on parkour) collapses under.
ACTION_SCALE = 0.25
CLIP_ACTIONS = 1.2 / ACTION_SCALE

# control (Go2CliffsideCfg / T04 teacher). Overridable per run with --kp/--kd:
# teachers trained under a different PD (e.g. T22-warm-orient-pd40.05-l40s, which
# trains at 40/0.5) MUST be evaluated at their own gains, or the eval measures a
# controller mismatch rather than the policy. Defaults keep every pre-2026-07-28
# sweep bit-reproducible.
KP = 35.0
KD = 0.6
DECIMATION = 4

# Real-deploy "dead-man" standing hold. When the policy is DISENGAGED
# (low_level_ctrl release-L1+R1 branch — see low_level_ctrl.cpp settle ramp via
# state_transform, or --disengage_after here), /lowcmd interpolates the target
# from the pose at release -> the fixed standing pose DEFAULT_DOF_POLICY (== the
# C++ standing_angels_) over ~2 s at these stiffer init_cmd gains kp=50/kd=1, NOT
# the policy's 35/0.6. Ramping from the current pose means kp=50 yields ~0 torque
# at release (no snap); the gains hold standing once the ramp completes.
STAND_KP = 50.0
STAND_KD = 1.0

# action_delay
ACTION_BUF_LEN = 8
ACTION_DELAY_VIEW = 1

# scandot grid (body frame, metres) — must match the checkpoint's Isaac config.
# T04-warm is a cliffside teacher and uses the forward-shifted, camera-footprint
# grid from go2_cliffside_config.py. The older parkour teacher grid is still
# available via --scandot_grid legacy.
MEASURED_X_LEGACY = np.array([-0.45, -0.3, -0.15, 0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2], dtype=np.float64)
MEASURED_X_CAMERA = np.array([0.45, 0.6, 0.75, 0.9, 1.05, 1.2, 1.35, 1.5, 1.65, 1.8, 1.95, 2.1], dtype=np.float64)
MEASURED_X = MEASURED_X_CAMERA.copy()
MEASURED_Y = np.array([-0.75, -0.6, -0.45, -0.3, -0.15, 0.0, 0.15, 0.3, 0.45, 0.6, 0.75], dtype=np.float64)


def make_scandot_body_xy(measured_x: np.ndarray) -> np.ndarray:
    # meshgrid order matches Isaac: torch.meshgrid(x, y) → grid_x.flatten(), grid_y.flatten()
    gx, gy = np.meshgrid(measured_x, MEASURED_Y, indexing="ij")
    body_xy = np.stack([gx.flatten(), gy.flatten()], axis=1)  # (132, 2)
    assert body_xy.shape[0] == NUM_SCAN
    return body_xy


SCANDOT_BODY_XY = make_scandot_body_xy(MEASURED_X)

# offset above robot base for ray origin; terrain max ~1 m, so 2 m clears it
RAY_HEIGHT_OFFSET = 2.0

# --- hurdle-aware scandot max-pooling (mirrors Isaac) --------------------
# go2_cliffside_config.py sets scandot_maxpool / scandot_maxpool_hurdles_only /
# scandot_maxpool_radius_x=8 / radius_y=0, and legged_robot._get_heights()
# replaces the min-of-3-neighbours lookup with F.max_pool2d over the height
# field for every dot whose footprint overlaps a recorded hurdle x-range. A 1 cm
# hurdle sitting between two nominal 15 cm samples is otherwise invisible to the
# teacher. Isaac pools over height-field CELLS (horizontal_scale = 0.01 m), so
# radius_x=8 is a ±8 cm / 17 cm-wide footprint in WORLD x — the same axis the
# height field is indexed on, not the body frame.
# Related method: Kuthirummal, Das & Samarasekera, "A Graph Traversal Based
# Algorithm for Obstacle Detection Using Lidar or Stereo", IROS 2011,
# DOI 10.1109/IROS.2011.6048725.
SCANDOT_POOL_CELL = 0.01          # Isaac terrain.horizontal_scale [m]
SCANDOT_CAMERA_OFFSET = np.array([0.33, 0.0, 0.08], dtype=np.float64)
SCANDOT_OCCLUSION_SAMPLES = 12
SCANDOT_OCCLUSION_EPS = 0.05

# Geom-group mask: only terrain geoms (group 0). Robot visuals are in group 2,
# collision shapes in group 3 (see go2.xml `class="visual"`/`class="collision"`).
# Without this mask the raycast intercepts the robot body and reports its own
# torso z as the terrain height for points that pass through the trunk.
TERRAIN_GEOMGROUP = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)

# 2026-06-07 fix: MuJoCo joint/sensor/actuator order in go2.xml is FR,FL,RR,RL,
# which IS the policy order (training reindex maps Isaac FL,FR,RL,RR -> FR,FL,RR,RL).
# The old MJ_TO_POLICY permutation therefore SWAPPED left/right legs on obs and
# action (mirrored controller, inconsistent with unmirrored body-frame channels
# and FL-first foot-force sensors). No joint permutation is needed at all.
# Kept only for reference; do not apply to joint data:
MJ_TO_POLICY = np.array([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8], dtype=np.int64)
# Foot force sensors in go2.xml are FL,FR,RL,RR -> policy wants FR,FL,RR,RL:
FOOT_MJ_TO_POLICY = np.array([1, 0, 3, 2], dtype=np.int64)

# default joint angles in TRUE policy order FR,FL,RR,RL (hip signs: FR/RR=-0.1, FL/RL=+0.1)
DEFAULT_DOF_POLICY = np.array(
    [-0.1, 0.8, -1.5,
     0.1, 0.8, -1.5,
     -0.1, 1.0, -1.5,
     0.1, 1.0, -1.5],
    dtype=np.float64,
)


def yaw_arrow_mat(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[0.0, -s,  c],
                     [0.0,  c,  s],
                     [-1.0, 0.0, 0.0]], dtype=np.float64)


def add_arrow(scene, pos, theta, length, radius, rgba):
    import mujoco
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        g,
        type=int(mujoco.mjtGeom.mjGEOM_ARROW),
        size=np.array([radius, radius, length], dtype=np.float64),
        pos=pos.astype(np.float64),
        mat=yaw_arrow_mat(theta).flatten(),
        rgba=np.array(rgba, dtype=np.float32),
    )
    scene.ngeom += 1


def add_sphere(scene, pos, radius, rgba):
    import mujoco
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        g,
        type=int(mujoco.mjtGeom.mjGEOM_SPHERE),
        size=np.array([radius, radius, radius], dtype=np.float64),
        pos=pos.astype(np.float64),
        mat=np.eye(3).flatten(),
        rgba=np.array(rgba, dtype=np.float32),
    )
    scene.ngeom += 1


def parse_goals(spec):
    """Parse '--goals x0,y0;x1,y1;...' into an (N,2) float array, or None."""
    if not spec:
        return None
    pts = []
    for chunk in spec.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        x, y = chunk.split(",")
        pts.append([float(x), float(y)])
    return np.array(pts, dtype=np.float64) if pts else None


def wall_foot_contact_forces(mujoco, m, d, foot_geom_ids, wall_geom_ids):
    """Return per-foot force from contacts specifically involving wall geoms."""
    forces = np.zeros(4, dtype=np.float64)
    wall_ids = set(int(x) for x in wall_geom_ids)
    for ci in range(d.ncon):
        contact = d.contact[ci]
        g1, g2 = int(contact.geom1), int(contact.geom2)
        foot_idx = -1
        if g1 in foot_geom_ids and g2 in wall_ids:
            foot_idx = int(np.where(foot_geom_ids == g1)[0][0])
        elif g2 in foot_geom_ids and g1 in wall_ids:
            foot_idx = int(np.where(foot_geom_ids == g2)[0][0])
        if foot_idx >= 0:
            cf = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(m, d, ci, cf)
            forces[foot_idx] = max(forces[foot_idx], float(np.linalg.norm(cf[:3])))
    return forces


def mj_quat_to_roll_pitch_yaw(q_wxyz):
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


def load_checkpoint(ckpt_path: Path):
    try:
        return torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except Exception:
        return torch.load(ckpt_path, map_location="cpu", weights_only=False)


def parse_x_ranges(spec):
    """Parse '--hurdle_x_ranges x0a,x0b;x1a,x1b' into an (N,2) array, or None."""
    if not spec:
        return None
    out = []
    for chunk in spec.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        a, b = chunk.split(",")
        out.append([float(a), float(b)])
    return np.array(out, dtype=np.float64) if out else None


def detect_hurdle_x_ranges(mujoco, m, d, name_substr="hurdle"):
    """World-frame x extents of every scene geom whose name contains `hurdle`.

    The MuJoCo mirror of Isaac's `terrain.flyball_hurdle_x_ranges_world`, which
    flyball_wall.py fills with (front_face_x, back_face_x) per hurdle slot and
    legged_robot.py reads to decide which scandots get pooled. Scene geoms are
    axis-aligned boxes in practice; a rotated geom falls back to the extent of
    its rotated half-axes, which is conservative (never under-covers the bar).
    """
    ranges = []
    for gid in range(m.ngeom):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        if name_substr not in name.lower():
            continue
        xmat = np.array(d.geom_xmat[gid], dtype=np.float64).reshape(3, 3)
        half_x = float(np.abs(xmat[0, :]) @ np.abs(m.geom_size[gid]))
        cx = float(d.geom_xpos[gid][0])
        ranges.append([cx - half_x, cx + half_x])
    if not ranges:
        return None
    ranges.sort()
    return np.array(ranges, dtype=np.float64)


def cast_scandots(mujoco, m, d, base_xyz, world_yaw, occlusion_mask=False,
                  pool_radius_x=0, pool_radius_y=0, hurdle_x_ranges=None,
                  pool_cell=SCANDOT_POOL_CELL):
    """Raycast scandots and optionally replace camera-occluded heights.

    With `pool_radius_x/y > 0` the height reported for a dot is the MAXIMUM over
    a (2*rx+1) x (2*ry+1) grid of `pool_cell`-spaced world-frame samples centred
    on it — the raycast analogue of Isaac's F.max_pool2d over the height field.
    When `hurdle_x_ranges` is given, only dots whose pooled x-footprint overlaps
    one of those ranges are pooled (Isaac's scandot_maxpool_hurdles_only); pass
    None to pool every dot. Pooling runs BEFORE the occlusion mask, matching
    legged_robot.py (pooling lives in _get_heights, masking is applied to its
    output).
    """
    cy, sy = np.cos(world_yaw), np.sin(world_yaw)
    xy_world = np.empty_like(SCANDOT_BODY_XY)
    xy_world[:, 0] = SCANDOT_BODY_XY[:, 0] * cy - SCANDOT_BODY_XY[:, 1] * sy + base_xyz[0]
    xy_world[:, 1] = SCANDOT_BODY_XY[:, 0] * sy + SCANDOT_BODY_XY[:, 1] * cy + base_xyz[1]

    origin_z = base_xyz[2] + RAY_HEIGHT_OFFSET
    vec = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    geomid_out = np.zeros(1, dtype=np.int32)
    terrain_z = np.empty(NUM_SCAN, dtype=np.float64)
    for i in range(NUM_SCAN):
        pnt = np.array([xy_world[i, 0], xy_world[i, 1], origin_z], dtype=np.float64)
        dist = mujoco.mj_ray(m, d, pnt, vec, TERRAIN_GEOMGROUP, 1, -1, geomid_out)
        if dist < 0.0:
            # ray missed everything (e.g. outside scene) — treat as floor at z=0
            terrain_z[i] = 0.0
        else:
            terrain_z[i] = origin_z - dist

    if pool_radius_x or pool_radius_y:
        if hurdle_x_ranges is None:
            pool_mask = np.ones(NUM_SCAN, dtype=bool)
        else:
            dot_lo = xy_world[:, 0] - pool_radius_x * pool_cell
            dot_hi = xy_world[:, 0] + pool_radius_x * pool_cell
            pool_mask = np.zeros(NUM_SCAN, dtype=bool)
            for lo, hi in np.asarray(hurdle_x_ranges, dtype=np.float64):
                if hi > lo:
                    pool_mask |= (dot_hi >= lo) & (dot_lo <= hi)
        offsets = [(ox * pool_cell, oy * pool_cell)
                   for ox in range(-pool_radius_x, pool_radius_x + 1)
                   for oy in range(-pool_radius_y, pool_radius_y + 1)
                   if not (ox == 0 and oy == 0)]
        for i in np.flatnonzero(pool_mask):
            h_max = terrain_z[i]
            for ox, oy in offsets:
                pnt = np.array([xy_world[i, 0] + ox, xy_world[i, 1] + oy, origin_z],
                               dtype=np.float64)
                dist = mujoco.mj_ray(m, d, pnt, vec, TERRAIN_GEOMGROUP, 1, -1, geomid_out)
                h = 0.0 if dist < 0.0 else origin_z - dist
                if h > h_max:
                    h_max = h
            terrain_z[i] = h_max

    if occlusion_mask:
        cam_xy = np.array([
            base_xyz[0] + SCANDOT_CAMERA_OFFSET[0] * cy,
            base_xyz[1] + SCANDOT_CAMERA_OFFSET[0] * sy,
        ])
        cam_z = base_xyz[2] + SCANDOT_CAMERA_OFFSET[2]
        seg_xy = xy_world - cam_xy
        seg_z = terrain_z - cam_z
        occluded = np.zeros(NUM_SCAN, dtype=bool)
        blocker_h = np.full(NUM_SCAN, -1.0e6, dtype=np.float64)
        for t in np.linspace(0.08, 0.92, SCANDOT_OCCLUSION_SAMPLES):
            sample_xy = cam_xy + t * seg_xy
            sample_z = cam_z + t * seg_z
            for i in range(NUM_SCAN):
                pnt = np.array([sample_xy[i, 0], sample_xy[i, 1], origin_z], dtype=np.float64)
                dist = mujoco.mj_ray(m, d, pnt, vec, TERRAIN_GEOMGROUP, 1, -1, geomid_out)
                h_terr = 0.0 if dist < 0.0 else origin_z - dist
                blocked = h_terr > sample_z[i] + SCANDOT_OCCLUSION_EPS
                if blocked:
                    occluded[i] = True
                    blocker_h[i] = max(blocker_h[i], h_terr)
        terrain_z = np.where(occluded, blocker_h, terrain_z)
    return terrain_z, xy_world


def write_run_meta(args, m) -> None:
    """Dump the full run configuration next to the recorded video.

    Without this the only record of how a clip was produced is its filename
    (same provenance hole run_real_remote.sh closed with meta.txt). Written
    BEFORE the sim loop so a crashed/killed run still leaves its config.
    """
    import json
    import subprocess

    def _git(*cmd):
        try:
            return subprocess.run(("git", "-C", str(REPO_ROOT)) + cmd,
                                  capture_output=True, text=True,
                                  timeout=5).stdout.strip() or None
        except Exception:
            return None

    meta = {
        "runner": "lidar_runner.py",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "argv": sys.argv[1:],
        "args": {k: (str(v) if isinstance(v, Path) else v)
                 for k, v in sorted(vars(args).items())},
        # hardcoded controller/actor constants the CLI cannot change
        "constants": {
            "kp": args.kp, "kd": args.kd,
            "stand_kp": STAND_KP, "stand_kd": STAND_KD,
            "decimation": DECIMATION, "action_scale": ACTION_SCALE,
            "clip_actions": CLIP_ACTIONS, "timestep": float(m.opt.timestep),
            "n_proprio": N_PROPRIO, "num_scan": NUM_SCAN,
            "action_delay_view": ACTION_DELAY_VIEW,
        },
        "git": {
            "sim2sim_commit": _git("rev-parse", "HEAD"),
            "sim2sim_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "sim2sim_dirty": bool(_git("status", "--porcelain")),
        },
        "env": {k: os.environ.get(k) for k in ("MUJOCO_GL", "KMP_DUPLICATE_LIB_OK")},
    }
    # The scene hash covers only the scene file; the robot model it <include>s
    # (contact solref/solimp, foot friction, cameras) changes the sim just as
    # much, so hash it too.
    for label, path in (("ckpt", args.ckpt), ("scene", args.scene),
                        ("robot_xml", REPO_ROOT / "resources" / "go2" / "go2.xml")):
        try:
            meta[f"{label}_sha256"] = __import__("hashlib").sha256(
                Path(path).read_bytes()).hexdigest()
        except Exception:
            meta[f"{label}_sha256"] = None

    meta_path = args.record.with_suffix(".meta.json")
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"[meta] wrote {meta_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path,
                    default=_CKPT_DEFAULT)
    ap.add_argument("--ep_root", type=Path,
                    default=_EP_ROOT_DEFAULT)
    ap.add_argument("--scene", type=Path, default=REPO_ROOT / "resources" / "go2" / "scene_flat.xml")
    ap.add_argument("--spawn_x", type=float, default=0.0,
                    help="initial robot base x position in world metres")
    ap.add_argument("--lin_vel_x", type=float, default=0.5)
    ap.add_argument("--kp", type=float, default=KP,
                    help="policy PD stiffness; must match the gains the checkpoint "
                         "was TRAINED at (default 35.0)")
    ap.add_argument("--kd", type=float, default=KD,
                    help="policy PD damping (default 0.6)")
    ap.add_argument("--cmd_pre_seconds", type=float, default=0.0,
                    help="hold the forward velocity command at 0 for this many seconds "
                         "at the start (the policy is engaged, just not commanded)")
    ap.add_argument("--stop_on_return_x", action="store_true",
                    help="zero the velocity command once the robot has driven out past "
                         "--return_arm_x and come back to --return_x, then run "
                         "--return_stop_hold_s more seconds and end the episode")
    ap.add_argument("--return_x", type=float, default=0.0,
                    help="x (m) the robot must return to for the hard stop")
    ap.add_argument("--return_arm_x", type=float, default=2.0,
                    help="x (m) that arms the return-x stop")
    ap.add_argument("--return_stop_hold_s", type=float, default=3.0,
                    help="seconds to hold cmd_x=0 after the return-x stop triggers")
    ap.add_argument("--vel_zero_after", type=float, default=None,
                    help="if set, drop the forward velocity command (prop[10]) to "
                         "0 after this many seconds of sim (the policy then holds "
                         "position; heading still tracks the goal sequence)")
    ap.add_argument("--disengage_after", type=float, default=None,
                    help="if set, DISENGAGE the policy after this many seconds of "
                         "sim: stop applying the actor's action and interpolate the "
                         "PD target from the pose at release -> the standing pose "
                         "(DEFAULT_DOF) over ~2 s at kp=50/kd=1, reproducing the "
                         "real-deploy dead-man (low_level_ctrl release-L1+R1 settle "
                         "ramp). Ramps from the current pose, so no snap.")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--viewer", action="store_true")
    ap.add_argument("--record", type=Path, default=None)
    ap.add_argument("--record_fps", type=int, default=50)
    ap.add_argument("--record_width", type=int, default=1280)
    ap.add_argument("--record_height", type=int, default=720)
    ap.add_argument("--record_camera", default="follow_cam")
    ap.add_argument("--action_delay", type=int, default=1, choices=[0, 1],
                    help="1 (default, changed 2026-07-28 to match the deploy/training "
                         "envelope): the fixed 1-step / 20 ms action delay every cliffside "
                         "policy after EP 6fb4d46 trains with, latched at fleet level 6. "
                         "Use 0 only for the older base policies trained without delay.")
    ap.add_argument("--proprio_latency_s", type=float, default=0.0,
                    help="DIAGNOSTIC, default 0 (off). Sensor-to-motor latency emulation "
                         "(mirrors distill_runner): physics keeps applying the previous PD "
                         "target for this long after the tick boundary, quantized to the "
                         "5 ms substep. Do NOT combine with --action_delay 1 for teachers "
                         "trained after EP 6fb4d46: there the 20 ms action delay already "
                         "models the whole proprio pipeline (the 16 ms enforced pad + ~3 ms "
                         "firmware/DDS), and randomize_proprio_latency is off precisely to "
                         "avoid double-counting it.")
    ap.add_argument("--friction_scale", type=float, default=1.0)
    ap.add_argument("--scandot_grid", choices=["camera", "legacy"], default="camera",
                    help="camera = T04/go2_cliffside forward grid x=0.45..2.10; "
                         "legacy = older grid x=-0.45..1.20")
    ap.add_argument("--draw_scandots", action="store_true",
                    help="overlay the 132 scandot ray hit points in the recorded video")
    ap.add_argument("--no_scandot_occlusion_mask", action="store_true",
                    help="disable Isaac-compatible camera line-of-sight masking")
    ap.add_argument("--scandot_maxpool", choices=["hurdles", "all", "off"],
                    default="hurdles",
                    help="Isaac scandot max-pooling (go2_cliffside_config: "
                         "scandot_maxpool=True, hurdles_only=True). hurdles = pool "
                         "only dots whose footprint overlaps a hurdle x-range "
                         "(scene geoms named *hurdle*, or --hurdle_x_ranges); "
                         "all = pool every dot; off = pre-2026-07-29 behaviour "
                         "(single ray per dot). NOTE: checkpoints trained before "
                         "the pooling change should be evaluated with 'off'.")
    ap.add_argument("--scandot_maxpool_radius_x", type=int, default=8,
                    help="pooling radius in 1 cm height-field cells along WORLD x "
                         "(Isaac scandot_maxpool_radius_x; 8 = 17 cm footprint)")
    ap.add_argument("--scandot_maxpool_radius_y", type=int, default=0,
                    help="pooling radius in 1 cm cells along WORLD y "
                         "(Isaac scandot_maxpool_radius_y; 0 = hurdle spans the "
                         "scan laterally, so no lateral pooling is needed)")
    ap.add_argument("--hurdle_x_ranges", type=str, default=None,
                    help="override the auto-detected hurdle x-ranges: "
                         "'front0,back0;front1,back1' in world metres")
    # --- Isaac goal-following heading (matches legged_robot.py delta_yaw) -----
    ap.add_argument("--goals", type=str, default=None,
                    help="Semicolon-separated world-frame goal xy list "
                         "'x0,y0;x1,y1;...'. When given, prop[6:8] are filled "
                         "with Isaac's delta_yaw/delta_next_yaw toward the goal "
                         "sequence (unwrapped, refreshed every 5 steps) instead "
                         "of the fixed +x heading.")
    ap.add_argument("--next_goal_threshold", type=float, default=0.5,
                    help="advance to the next goal when within this distance (m)")
    ap.add_argument("--reach_goal_delay", type=float, default=0.1,
                    help="dwell time (s) at a reached goal before advancing")
    ap.add_argument("--wall_goal_idx", type=int, default=-1,
                    help="goal index treated as the WoCoCo wall-contact goal; "
                         "the runner holds it because it has no plant FSM")
    ap.add_argument("--wall_front_x", type=float, default=float("nan"),
                    help="world x of the wall front face (for the wall-goal x-gate)")
    ap.add_argument("--wall_gate_margin", type=float, default=0.15,
                    help="legacy x-gate margin (used only with --legacy_wall_x_gate)")
    ap.add_argument("--legacy_wall_x_gate", action="store_true",
                    help="use the pre-training-compatible base-x wall shortcut")
    ap.add_argument("--contact_threshold", type=float, default=1.0,
                    help="foot-force threshold (N) for proprioceptive contact observations")
    # --- Isaac-parity ablation knobs -----------------------------------------
    ap.add_argument("--seed", type=int, default=0,
                    help="RNG seed for observation noise and the init jitter")
    ap.add_argument("--init_jitter", type=float, default=0.0,
                    help="trial-decorrelation only (NOT an Isaac feature): uniform "
                         "+-this many rad on the initial joint angles and "
                         "+-(this*20) deg on the initial base yaw, so repeated runs "
                         "of a deterministic config differ")
    ap.add_argument("--contact_or_filter", type=int, default=0, choices=[0, 1],
                    help="1 = mirror Isaac's contact_filt = contact OR last_contact "
                         "(legged_robot.py:680). 0 = instantaneous threshold only.")
    ap.add_argument("--obs_noise", type=int, default=0, choices=[0, 1],
                    help="1 = add Isaac's training observation noise "
                         "(_get_proprio_noise_scale_vec + height_measurements): "
                         "uniform +-0.0125 ang_vel, +-0.01 dof_pos, +-0.0025 dof_vel, "
                         "+-0.02 per scandot; roll/pitch get none (ns.rotation=0).")
    ap.add_argument("--wall_advance", choices=["frontfeet", "stage2"], default="frontfeet",
                    help="wall-goal release rule. frontfeet = legacy (both front feet "
                         ">=10 N on the wall). stage2 = P4L proxy for Isaac's "
                         "wococo_plant_complete_stage=2: front pair on the wall, then "
                         "the rear pair, then both front feet back on the FLOOR.")
    ap.add_argument("--result_tag", type=str, default=None,
                    help="print a machine-readable RESULT line with this tag")
    args = ap.parse_args()

    if args.record is not None and "MUJOCO_GL" not in os.environ:
        os.environ["MUJOCO_GL"] = "egl"

    # A bare --record filename (no directory) files itself under the checkpoint's
    # own folder: run_logs/sim2sim/<ckptdir>__<model>/<name>.mp4. An explicit
    # path with a directory is always honoured as given.
    if args.record is not None and args.record.parent == Path("."):
        policy = f"{args.ckpt.parent.name}__{args.ckpt.stem}"
        args.record = REPO_ROOT / "run_logs" / "sim2sim" / policy / args.record.name
        args.record.parent.mkdir(parents=True, exist_ok=True)
        print(f"[video] bare --record name -> {args.record}")

    global MEASURED_X, SCANDOT_BODY_XY
    MEASURED_X = MEASURED_X_CAMERA.copy() if args.scandot_grid == "camera" else MEASURED_X_LEGACY.copy()
    SCANDOT_BODY_XY = make_scandot_body_xy(MEASURED_X)
    use_scandot_occlusion_mask = args.scandot_grid == "camera" and not args.no_scandot_occlusion_mask
    print(f"[scandots] grid={args.scandot_grid} x=[{MEASURED_X[0]:.2f}..{MEASURED_X[-1]:.2f}] y=[{MEASURED_Y[0]:.2f}..{MEASURED_Y[-1]:.2f}] occlusion_mask={use_scandot_occlusion_mask}")

    add_extreme_parkour_to_path(args.ep_root)
    if "torchvision" not in sys.modules:
        import types as _types
        sys.modules["torchvision"] = _types.ModuleType("torchvision")
    import mujoco
    from rsl_rl.modules.actor_critic import Actor, get_activation
    from rsl_rl.modules.estimator import Estimator

    print(f"[ckpt] loading {args.ckpt}")
    ckpt = load_checkpoint(args.ckpt)
    print(f"[ckpt] iter = {ckpt.get('iter', '?')}")

    activation = get_activation("elu")
    actor = Actor(
        num_prop=N_PROPRIO, num_scan=NUM_SCAN, num_actions=NUM_ACTIONS,
        scan_encoder_dims=[128, 64, 32],
        actor_hidden_dims=[512, 256, 128],
        priv_encoder_dims=[64, 20],
        num_priv_latent=N_PRIV_LATENT, num_priv_explicit=N_PRIV_EXPLICIT,
        num_hist=HISTORY_LEN, activation=activation, tanh_encoder_output=False,
    )
    actor_state = {k[len("actor."):]: v for k, v in ckpt["model_state_dict"].items()
                   if k.startswith("actor.")}
    actor.load_state_dict(actor_state, strict=True)
    actor.eval()

    estimator = Estimator(input_dim=N_PROPRIO, output_dim=N_PRIV_EXPLICIT, hidden_dims=[128, 64])
    estimator.load_state_dict(ckpt["estimator_state_dict"], strict=True)
    estimator.eval()

    fc = actor.actor_backbone[0]
    assert fc.in_features == 114, f"actor first layer expected 114, got {fc.in_features}"
    print(f"[actor] in_features={fc.in_features}  scan_lat=32  priv_expl={N_PRIV_EXPLICIT}  hist_lat=20  → 114 ✓")

    print(f"[mj] loading {args.scene}")
    m = mujoco.MjModel.from_xml_path(str(args.scene))
    if args.friction_scale != 1.0:
        m.geom_friction[:, 0] *= args.friction_scale
        print(f"[mj] friction_scale={args.friction_scale}  sliding friction × {args.friction_scale}")
    d = mujoco.MjData(m)

    default_dof_mj = DEFAULT_DOF_POLICY  # MuJoCo qpos order == policy order
    rng = np.random.default_rng(args.seed)

    d.qpos[:3] = [args.spawn_x, 0.0, 0.42]
    d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    d.qpos[7:7 + 12] = default_dof_mj
    if args.init_jitter > 0.0:
        d.qpos[7:7 + 12] += rng.uniform(-args.init_jitter, args.init_jitter, 12)
        yaw0 = np.deg2rad(rng.uniform(-1.0, 1.0) * args.init_jitter * 20.0)
        d.qpos[3:7] = [np.cos(yaw0 / 2), 0.0, 0.0, np.sin(yaw0 / 2)]
        print(f"[init] jitter={args.init_jitter} seed={args.seed} yaw0={np.rad2deg(yaw0):+.2f} deg")
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)

    # --- Isaac scandot max-pooling setup ---------------------------------
    pool_rx = args.scandot_maxpool_radius_x if args.scandot_maxpool != "off" else 0
    pool_ry = args.scandot_maxpool_radius_y if args.scandot_maxpool != "off" else 0
    pool_hurdle_ranges = None
    if args.scandot_maxpool == "hurdles":
        pool_hurdle_ranges = parse_x_ranges(args.hurdle_x_ranges)
        if pool_hurdle_ranges is None:
            pool_hurdle_ranges = detect_hurdle_x_ranges(mujoco, m, d)
        if pool_hurdle_ranges is None:
            # No hurdle geom in this scene: hurdles-only pooling is a no-op,
            # exactly like Isaac's sentinel (-1, -1) x-ranges.
            pool_hurdle_ranges = np.zeros((0, 2), dtype=np.float64)
    if args.scandot_maxpool == "off":
        print("[scandots] maxpool=off (one ray per dot)")
    else:
        rng_str = ("all dots" if pool_hurdle_ranges is None else
                   " ".join(f"[{a:.3f},{b:.3f}]" for a, b in pool_hurdle_ranges) or "none")
        print(f"[scandots] maxpool={args.scandot_maxpool} radius=({pool_rx},{pool_ry}) cells "
              f"= ±{pool_rx * SCANDOT_POOL_CELL * 100:.0f}/±{pool_ry * SCANDOT_POOL_CELL * 100:.0f} cm  "
              f"hurdle_x_ranges: {rng_str}")

    # Isaac observation-noise vector (_get_proprio_noise_scale_vec), in the same
    # SCALED space as the obs. ns.rotation = 0.0, so roll/pitch stay clean;
    # commands, terrain flags, last action and contacts get no noise either.
    noise_vec = np.zeros(N_PROPRIO, dtype=np.float64)
    if args.obs_noise:
        noise_vec[0:3] = 0.05 * SCALE_ANG_VEL      # ns.ang_vel * s.ang_vel
        noise_vec[13:25] = 0.01 * SCALE_DOF_POS    # ns.dof_pos * s.dof_pos
        noise_vec[25:37] = 0.05 * SCALE_DOF_VEL    # ns.dof_vel * s.dof_vel
        print("[obs_noise] Isaac proprio noise on; scandot noise +-0.02 m")
    last_contact_raw = np.zeros(4, dtype=np.float64)   # for the Isaac OR-filter

    # MuJoCo-side mirror of Isaac's P4L contact FSM (FL, FR, RL, RR order).
    foot_body_ids = np.array([
        mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
        for name in ("FL_foot", "FR_foot", "RL_foot", "RR_foot")
    ], dtype=np.int32)
    foot_geom_ids = np.array([
        mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in ("FL", "FR", "RL", "RR")
    ], dtype=np.int32)
    # Wall-contact geoms. The flyball scenes name them flyball_wall_*, the
    # leanplate scenes lean_plate/room_wall; absent names resolve to -1 and are
    # dropped so either family works.
    wall_geom_ids = np.array([
        gid for gid in (
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in ("flyball_wall_ramp", "flyball_wall_back",
                         "lean_plate", "room_wall")
        ) if gid >= 0
    ], dtype=np.int32)
    p4l_stage = 0
    p4l_complete = False

    record_renderer = None
    if args.record is not None:
        record_renderer = mujoco.Renderer(m, height=args.record_height,
                                          width=args.record_width)

    viewer = None
    if args.viewer:
        from mujoco import viewer as mj_viewer
        viewer = mj_viewer.launch_passive(m, d)

    video_writer = None
    record_every = 1
    if args.record is not None:
        import imageio
        video_writer = imageio.get_writer(str(args.record), fps=args.record_fps, codec="libx264",
                                          quality=None, output_params=["-crf", "23", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                                                                       "-movflags", "+faststart"])
        record_every = max(1, int(round(50 / args.record_fps)))
        print(f"[video] recording {args.record} at {args.record_fps} fps ({args.record_width}×{args.record_height}, cam={args.record_camera})")
        write_run_meta(args, m)

    obs_full = torch.zeros(1, N_PROPRIO + NUM_SCAN + N_PRIV_EXPLICIT + N_PRIV_LATENT + HISTORY_LEN * N_PROPRIO)
    history_ring = None
    last_action = torch.zeros(1, NUM_ACTIONS)
    action_buf = deque([torch.zeros(1, NUM_ACTIONS) for _ in range(ACTION_BUF_LEN)], maxlen=ACTION_BUF_LEN)
    last_scandot_xy_world = None
    last_terrain_z = None
    scandot_cache = None
    disengage_tick = None      # global_counter at the first disengaged tick
    q_init_settle = None       # joint pose latched at release, for the settle ramp

    # --- Isaac goal-following heading state -------------------------------
    goals_xy = parse_goals(args.goals)
    cur_goal_idx = 0
    reach_goal_timer = 0
    delta_yaw_state = 0.0
    delta_next_yaw_state = 0.0
    target_yaw = 0.0
    next_target_yaw = 0.0
    reach_goal_delay_ticks = max(1, int(round(args.reach_goal_delay / (m.opt.timestep * DECIMATION))))
    if goals_xy is not None:
        print(f"[goals] {len(goals_xy)} goals, thr={args.next_goal_threshold}m "
              f"delay={reach_goal_delay_ticks}tk  wall_idx={args.wall_goal_idx} "
              f"wall_front_x={args.wall_front_x}")
        for gi, g in enumerate(goals_xy):
            print(f"        g{gi} = ({g[0]:+.2f}, {g[1]:+.2f})")

    n_policy_ticks = int(args.seconds / (m.opt.timestep * DECIMATION))
    print(f"[run] {n_policy_ticks} policy ticks at 50 Hz → {args.seconds:.1f} s of sim")

    # Constant-latency enforcement (EP paper 4.1) mirror. The teacher has no
    # depth path, so only the proprio (lowstate-receipt -> target) pad applies;
    # the 20 ms action delay is modelled by --action_delay 1.
    proprio_delay_steps = int(round(args.proprio_latency_s / m.opt.timestep))
    proprio_delay_steps = min(DECIMATION, max(0, proprio_delay_steps))
    print(f"[latency] proprio={proprio_delay_steps * m.opt.timestep * 1000:.0f} ms "
          f"({proprio_delay_steps}/{DECIMATION} substeps; requested "
          f"{args.proprio_latency_s * 1000:.0f} ms), "
          f"action_delay={'20 ms (1 tick)' if args.action_delay else 'off'}")
    if args.action_delay and proprio_delay_steps > 0:
        print(f"[latency] WARNING: --action_delay 1 and --proprio_latency_s "
              f"{args.proprio_latency_s:g} model the SAME physical loop delay — this run "
              f"applies {20 + proprio_delay_steps * m.opt.timestep * 1000:.0f} ms. Pick one.")
    held_target_dof_mj = DEFAULT_DOF_POLICY.copy()
    return_armed = False
    return_stop_tick = None
    return_hold_ticks = max(0, int(round(args.return_stop_hold_s /
                                         (m.opt.timestep * DECIMATION))))

    torque_max = 0.0
    base_x_max = -1e9
    base_z_min, base_z_max = 1.0, 0.0
    fwd_velocity_window = deque(maxlen=100)

    t0 = time.time()
    for global_counter in range(n_policy_ticks):
        sim_t = global_counter * (m.opt.timestep * DECIMATION)
        sd = d.sensordata
        qj_mj = sd[0:12].copy()
        dqj_mj = sd[12:24].copy()
        quat_wxyz = sd[36:40].copy()
        ang_vel_body = sd[40:43].copy()
        roll, pitch, world_yaw = mj_quat_to_roll_pitch_yaw(quat_wxyz)
        foot_forces = np.linalg.norm(sd[55:67].reshape(4, 3), axis=1)
        foot_forces = foot_forces[FOOT_MJ_TO_POLICY]  # FL,FR,RL,RR sensors -> FR,FL,RR,RL policy
        contact_raw = (foot_forces > args.contact_threshold).astype(np.float64)
        if args.contact_or_filter:
            # Isaac legged_robot.py:680 — contact_filt = contact OR last_contacts.
            contact_filt = np.maximum(contact_raw, last_contact_raw)
        else:
            contact_filt = contact_raw
        last_contact_raw = contact_raw

        # Wall-goal gate: advance as soon as both front feet contact the wall
        # at >=10 N. No rear-foot condition is required for goal progression.
        if args.wall_goal_idx >= 0 and cur_goal_idx == args.wall_goal_idx and not p4l_complete:
            wall_forces = wall_foot_contact_forces(mujoco, m, d, foot_geom_ids, wall_geom_ids)
            if args.wall_advance == "frontfeet":
                if p4l_stage == 0 and np.all(wall_forces[:2] >= 10.0):
                    p4l_stage = 1
                    p4l_complete = True
                    print(f"[wococo] t={sim_t:.2f}s front feet contacted wall (>=10 N); wall goal complete")
            else:
                # P4L stage-2 proxy for Isaac's wococo_plant_complete_stage=2:
                # front pair plants on the wall -> rear pair plants on the wall ->
                # LANDING, i.e. both front feet back on the floor (contact bit set
                # while no longer pressing the wall). foot order here is FL,FR,RL,RR.
                if p4l_stage == 0 and np.all(wall_forces[:2] >= 10.0):
                    p4l_stage = 1
                    print(f"[wococo] t={sim_t:.2f}s stage 1: front pair on wall")
                elif p4l_stage == 1 and np.all(wall_forces[2:] >= 10.0):
                    p4l_stage = 2
                    print(f"[wococo] t={sim_t:.2f}s stage 2: rear pair on wall")
                elif p4l_stage == 2:
                    front_on_floor = np.all(contact_raw[[0, 1]] > 0.0) and np.all(wall_forces[:2] < 1.0)
                    if front_on_floor:
                        p4l_complete = True
                        print(f"[wococo] t={sim_t:.2f}s LANDING: front feet back on floor; wall goal complete")

        # sensors are already in policy order (FR,FL,RR,RL) - no permutation
        qj_policy = qj_mj
        dqj_policy = dqj_mj
        last_action_pol = last_action.numpy().reshape(-1)

        # --- heading: Isaac goal-following delta_yaw, or fixed +x fallback ---
        if goals_xy is not None:
            base_xy = d.qpos[:2].copy()
            n_goals = len(goals_xy)
            # Goal advancement mirrors training. Normal goals use the distance
            # threshold. The flyball wall goal advances only after WoCoCo P4L
            # plant-complete stage 2; this renderer has no FSM, so it stays active.
            cur = goals_xy[min(cur_goal_idx, n_goals - 1)]
            is_wall = (cur_goal_idx == args.wall_goal_idx) and not np.isnan(args.wall_front_x)
            if is_wall and args.legacy_wall_x_gate:
                reached = base_xy[0] >= (args.wall_front_x - args.wall_gate_margin)
            elif is_wall:
                reached = p4l_complete
            else:
                reached = np.linalg.norm(cur - base_xy) < args.next_goal_threshold
            if reached:
                reach_goal_timer += 1
                if reach_goal_timer > reach_goal_delay_ticks and cur_goal_idx < n_goals - 1:
                    cur_goal_idx += 1
                    reach_goal_timer = 0
            else:
                reach_goal_timer = 0
            cur = goals_xy[min(cur_goal_idx, n_goals - 1)]
            nxt = goals_xy[min(cur_goal_idx + 1, n_goals - 1)]
            tgt_rel = cur - base_xy
            nxt_rel = nxt - base_xy
            target_yaw = float(np.arctan2(tgt_rel[1], tgt_rel[0]))
            next_target_yaw = float(np.arctan2(nxt_rel[1], nxt_rel[0]))
            # delta_yaw is UNWRAPPED and refreshed only every 5 steps (Isaac:
            # legged_robot.compute_observations, `if global_counter % 5 == 0`).
            if global_counter % 5 == 0:
                delta_yaw_state = target_yaw - world_yaw
                delta_next_yaw_state = next_target_yaw - world_yaw
            dy = delta_yaw_state
            dny = delta_next_yaw_state
        else:
            # Forward-only "goal": target world-frame yaw is 0 (look down +x).
            # Isaac's delta_yaw = target_yaw - yaw, so for a +x goal it = -yaw.
            dy = float(-world_yaw)
            dny = dy
            target_yaw = 0.0

        # forward velocity command: held at lin_vel_x, optionally zeroed after
        # --vel_zero_after seconds so the policy stops driving and holds position.
        cmd_vx = args.lin_vel_x
        if sim_t < args.cmd_pre_seconds:
            cmd_vx = 0.0            # pre-roll: policy engaged, no drive command
        if args.vel_zero_after is not None and sim_t >= args.vel_zero_after:
            cmd_vx = 0.0

        # Return-x hard stop (mirrors distill_runner / the joy_sequencer FSM):
        # arm once the robot has driven out past --return_arm_x, then zero the
        # command when it comes back to --return_x and hold for the dwell.
        if args.stop_on_return_x:
            base_x_now = float(d.qpos[0])
            if base_x_now >= args.return_arm_x:
                return_armed = True
            if (return_armed and return_stop_tick is None
                    and base_x_now <= args.return_x and sim_t >= args.cmd_pre_seconds):
                return_stop_tick = global_counter
                print(f"[return-x] cmd_x -> 0 at t={sim_t:.2f}s x={base_x_now:+.3f}; "
                      f"holding {args.return_stop_hold_s:.1f}s")
            if return_stop_tick is not None:
                cmd_vx = 0.0

        prop = np.zeros(N_PROPRIO, dtype=np.float64)
        prop[0:3]   = ang_vel_body * SCALE_ANG_VEL
        prop[3]     = roll
        prop[4]     = pitch
        prop[5]     = 0.0          # 0 * delta_yaw
        prop[6]     = dy           # delta_yaw
        prop[7]     = dny          # delta_next_yaw
        prop[8]     = 0.0       # 0 * cmd_x
        prop[9]     = 0.0       # 0 * cmd_y
        prop[10]    = cmd_vx
        prop[11]    = 1.0
        prop[12]    = 0.0
        prop[13:25] = (qj_policy - DEFAULT_DOF_POLICY) * SCALE_DOF_POS
        prop[25:37] = dqj_policy * SCALE_DOF_VEL
        prop[37:49] = last_action_pol
        prop[49:53] = contact_filt - 0.5

        if args.obs_noise:
            # Isaac: obs_buf += U(-1,1) * noise_scale_vec, applied BEFORE the
            # frame is pushed into the history buffer.
            prop = prop + rng.uniform(-1.0, 1.0, N_PROPRIO) * noise_vec

        prop_t = torch.from_numpy(prop).float().unsqueeze(0)

        # history (yaw-masked copy, matches Isaac obs_history_buf)
        # Isaac's compute_observations (legged_robot.py:723-730) initializes the
        # history with HISTORY_LEN stacked copies of the first proprio frame
        # when episode_length_buf <= 1, NOT zeros. Using zeros would mismatch
        # the training distribution at t=0 and the base policy is sensitive to
        # this — it loses balance within ~3 s when fed a zero history.
        prop_hist_frame = prop_t.clone()
        prop_hist_frame[:, 6:8] = 0.0
        if history_ring is None:
            history_ring = prop_hist_frame.unsqueeze(1).repeat(1, HISTORY_LEN, 1)
        else:
            history_ring = torch.cat([history_ring[:, 1:], prop_hist_frame.unsqueeze(1)], dim=1)

        # --- scandots from raycast ----------------------------------------
        base_xyz = d.qpos[:3].copy()
        # Isaac refreshes measured heights every depth.update_interval=5 ticks;
        # keep both the masked heights and their world positions between refreshes.
        if scandot_cache is None or global_counter % 5 == 0:
            scandot_cache = cast_scandots(
                mujoco, m, d, base_xyz, world_yaw,
                occlusion_mask=use_scandot_occlusion_mask,
                pool_radius_x=pool_rx, pool_radius_y=pool_ry,
                hurdle_x_ranges=pool_hurdle_ranges)
        terrain_z, xy_world = scandot_cache
        scan_obs = np.clip(base_xyz[2] - 0.3 - terrain_z, -1.0, 1.0)
        if args.obs_noise:
            # Isaac noise_scales.height_measurements = 0.02, noise_level 1.0
            scan_obs = scan_obs + rng.uniform(-1.0, 1.0, scan_obs.shape) * 0.02
        last_terrain_z = terrain_z
        last_scandot_xy_world = xy_world

        # --- assemble obs ------------------------------------------------
        obs_full[:, 0:N_PROPRIO] = prop_t
        obs_full[:, N_PROPRIO:N_PROPRIO + NUM_SCAN] = torch.from_numpy(scan_obs).float().unsqueeze(0)
        with torch.no_grad():
            priv_expl = estimator(prop_t)
        obs_full[:, N_PROPRIO + NUM_SCAN:N_PROPRIO + NUM_SCAN + N_PRIV_EXPLICIT] = priv_expl
        obs_full[:, N_PROPRIO + NUM_SCAN + N_PRIV_EXPLICIT:N_PROPRIO + NUM_SCAN + N_PRIV_EXPLICIT + N_PRIV_LATENT] = 0.0
        obs_full[:, -HISTORY_LEN * N_PROPRIO:] = history_ring.reshape(1, -1)
        obs_full.clamp_(-CLIP_OBS, CLIP_OBS)

        if global_counter == 0:
            print(f"[step-0] base_z={base_xyz[2]:.3f}  scan_obs[min,max,mean]=({scan_obs.min():+.3f},{scan_obs.max():+.3f},{scan_obs.mean():+.3f})")

        # --- actor (NO scandots_latent → scan_encoder runs on raw scandots)
        with torch.no_grad():
            hist_latent = actor.infer_hist_latent(obs_full)
            action_raw = actor(obs_full, hist_encoding=True, eval=False, scandots_latent=None)
        # Isaac (legged_robot.py:124) stores the *unclipped* raw action into
        # action_history_buf, then clips a separate copy for the simulator.
        # The proprio at the next step reads action_history_buf[:, -1], so it
        # must be the unclipped value or the actor sees a different obs than
        # at training time.
        last_action = action_raw.detach()
        action_clipped = action_raw.clamp(-CLIP_ACTIONS, CLIP_ACTIONS)

        action_buf.append(action_clipped.detach())
        action_to_apply = action_buf[-1 - ACTION_DELAY_VIEW] if args.action_delay else action_clipped

        action_pol = action_to_apply.squeeze(0).numpy()
        target_dof_policy = DEFAULT_DOF_POLICY + ACTION_SCALE * action_pol
        target_dof_mj = target_dof_policy  # MuJoCo actuator order == policy order

        # Dead-man: once disengaged, stop applying the actor's action and INTERPOLATE
        # the PD target from the pose at release -> standing over ~100 ticks (~2 s) at
        # kp=50/kd=1, mirroring low_level_ctrl's settle ramp (state_transform). No
        # snap: the ramp starts from the current pose, so kp=50 yields ~0 torque at
        # release. (This clean lockstep port skips the C++ ~5-tick q_init-capture
        # window — a DDS timing artifact — and latches q_init instantly instead.)
        disengaged = (args.disengage_after is not None and sim_t >= args.disengage_after)
        settle_pct = None
        if disengaged:
            if disengage_tick is None:
                disengage_tick = global_counter
                q_init_settle = d.qpos[7:7 + 12].copy()   # pose latched at release
            rate = min(1.0, (global_counter - disengage_tick) / 100.0)
            target_dof_mj = q_init_settle * (1.0 - rate) + DEFAULT_DOF_POLICY * rate
            kp_pd, kd_pd = STAND_KP, STAND_KD
            settle_pct = rate
        else:
            kp_pd, kd_pd = args.kp, args.kd

        for substep in range(DECIMATION):
            active_target = (held_target_dof_mj
                             if substep < proprio_delay_steps else target_dof_mj)
            torque = kp_pd * (active_target - d.qpos[7:7 + 12]) - kd_pd * d.qvel[6:6 + 12]
            d.ctrl[:] = torque
            mujoco.mj_step(m, d)
            if viewer is not None:
                viewer.sync()
        held_target_dof_mj = target_dof_mj.copy()

        if (return_stop_tick is not None
                and global_counter - return_stop_tick >= return_hold_ticks):
            print(f"[return-x] hold complete at t={sim_t:.2f}s "
                  f"pos=({float(d.qpos[0]):+.3f},{float(d.qpos[1]):+.3f})")
            break

        torque_max = max(torque_max, float(np.max(np.abs(torque))))
        base_x_max = max(base_x_max, float(d.qpos[0]))
        base_z = float(d.qpos[2])
        base_z_min = min(base_z_min, base_z)
        base_z_max = max(base_z_max, base_z)
        fwd_velocity_window.append(float(d.qvel[0]))

        if global_counter > 0 and global_counter % 50 == 0:
            mean_vx = float(np.mean(fwd_velocity_window))
            hl = hist_latent[0].detach()
            goal_str = ""
            if goals_xy is not None:
                gc = goals_xy[min(cur_goal_idx, len(goals_xy) - 1)]
                goal_str = (f"  goal[{cur_goal_idx}]→({gc[0]:+.2f},{gc[1]:+.2f})"
                            f"  dyaw={np.degrees(dy):+6.1f}°  dnext={np.degrees(dny):+6.1f}°")
            dis_str = "  [DISENGAGED]" if disengaged else ""
            print(f"[t={global_counter*0.02:5.2f}s] base_z={base_z:.3f}  pos=({float(d.qpos[0]):+5.2f},{float(d.qpos[1]):+5.2f})  vx_avg(2s)={mean_vx:+.3f}  τ_peak={torque_max:.1f}  scan_μ={scan_obs.mean():+.3f}  scan_σ={scan_obs.std():.3f}  world_yaw={np.degrees(world_yaw):+6.1f}°  hist|={float(hl.norm()):.2f}{goal_str}{dis_str}")

        # --- video frame -----------------------------------------------
        if video_writer is not None and global_counter % record_every == 0:
            record_renderer.update_scene(d, camera=args.record_camera)
            base_pos = base_xyz.copy()
            base_pos[2] += 0.20
            if goals_xy is not None:
                # Show the two goal-yaw inputs supplied to the policy.
                add_arrow(record_renderer.scene, base_pos + np.array([0.0, 0.0, 0.10]),
                          target_yaw, length=0.30, radius=0.010,
                          rgba=(0.95, 0.15, 0.90, 1.0))
                add_arrow(record_renderer.scene, base_pos + np.array([0.0, 0.0, 0.20]),
                          next_target_yaw, length=0.26, radius=0.010,
                          rgba=(0.10, 0.85, 1.0, 1.0))
                for gi, g in enumerate(goals_xy):
                    if gi == cur_goal_idx:
                        rgba, rad = (0.10, 1.0, 0.10, 0.95), 0.035  # current → bright green
                    elif gi < cur_goal_idx:
                        rgba, rad = (0.35, 0.35, 0.35, 0.55), 0.018 # passed → grey
                    else:
                        rgba, rad = (1.0, 0.80, 0.10, 0.75), 0.022  # upcoming → amber
                    add_sphere(record_renderer.scene,
                               np.array([g[0], g[1], 0.05]), rad, rgba)
            if args.draw_scandots and last_scandot_xy_world is not None:
                # color by clipped scan value: red = obstacle above, blue = pit below
                v = scan_obs
                for i in range(NUM_SCAN):
                    p = np.array([last_scandot_xy_world[i, 0],
                                  last_scandot_xy_world[i, 1],
                                  last_terrain_z[i] + 0.01])
                    t = float((v[i] + 1.0) * 0.5)  # 0=floor below, 0.5≈base, 1=above base
                    rgba = (1.0 - t, 0.4, t, 0.85)
                    add_sphere(record_renderer.scene, p, 0.018, rgba)
            frame = record_renderer.render().copy()
            # --- data overlay (cmd speed, checkpoint, GT velocity) -------
            # Body-frame GT velocity from MuJoCo (privileged), like play.py --data.
            try:
                import cv2
                quat = d.qpos[3:7]  # wxyz
                w, x, y, z = quat
                # world->body yaw-only is enough for vx readout; use full rot for honesty
                Rm = np.array([
                    [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
                    [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
                    [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
                ])
                v_body = Rm.T @ d.qvel[0:3]
                if disengaged and settle_pct is not None and settle_pct < 1.0:
                    policy_state = f"DISENGAGED - ramp to stand {settle_pct*100:.0f}% kp=50/kd=1"
                elif disengaged:
                    policy_state = "DISENGAGED - standing hold kp=50/kd=1"
                else:
                    policy_state = f"cmd_vx: {cmd_vx:+.2f} m/s"
                lines = [
                    f"ckpt: {args.ckpt.name}",
                    policy_state,
                    f"gt_v:  ({v_body[0]:+.2f}, {v_body[1]:+.2f}, {v_body[2]:+.2f}) m/s",
                    f"base_z: {base_z:.3f} m   t={global_counter*0.02:5.2f}s",
                ]
                if goals_xy is not None:
                    lines.append(f"goal {cur_goal_idx}/{len(goals_xy)-1}  "
                                 f"dyaw {np.degrees(dy):+.0f}  dnext {np.degrees(dny):+.0f}")
                y0 = 28
                for li, txt in enumerate(lines):
                    cv2.putText(frame, txt, (12, y0 + 26*li),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                                (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(frame, txt, (12, y0 + 26*li),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                                (60, 255, 60), 1, cv2.LINE_AA)
            except ImportError:
                pass  # no cv2 -> plain frames
            video_writer.append_data(frame)

    elapsed = time.time() - t0
    print(f"[done] {n_policy_ticks} ticks in {elapsed:.1f}s wall  base_z∈[{base_z_min:.3f},{base_z_max:.3f}]  τ_peak={torque_max:.1f}  vx_avg(last 2s)={float(np.mean(fwd_velocity_window)):+.3f}")

    if args.result_tag is not None:
        fx, fy, fz = float(d.qpos[0]), float(d.qpos[1]), float(d.qpos[2])
        upright = fz >= 0.20
        returned = bool(upright and fx <= args.return_x + 0.5)
        print(f"RESULT\ttag={args.result_tag}\tseed={args.seed}"
              f"\treturned={int(returned)}\tupright={int(upright)}"
              f"\tfinal_x={fx:.3f}\tfinal_y={fy:.3f}\tfinal_z={fz:.3f}"
              f"\tmax_x={base_x_max:.3f}\tmin_z={base_z_min:.3f}"
              f"\ttau_peak={torque_max:.1f}\tp4l_stage={p4l_stage}"
              f"\tp4l_complete={int(p4l_complete)}\tgoal_idx={cur_goal_idx}"
              f"\tsim_t={sim_t:.2f}")

    if viewer is not None:
        viewer.close()
    if video_writer is not None:
        video_writer.close()
        print(f"[video] wrote {args.record}")


if __name__ == "__main__":
    main()
