#!/usr/bin/env python3
"""
Sim-to-sim verification of the extreme-parkour distillation (student) policy in MuJoCo.

Self-contained, ROS-free. Loads the depth_actor + estimator + depth_encoder from a
trained extreme-parkour distillation checkpoint, runs them in a flat MuJoCo scene
at 50 Hz with 30/0.6 PD gains, and prints diagnostics.

See ../../../resources/go2/scene_flat.xml for the scene; see go2.xml for the
front_depth_cam used by the depth encoder.
"""

import argparse
import math
import os
import re
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import distill_common as depth_common

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


# --- proprio / actor dimensions (frozen for P11-99-distillation) ---------
N_PROPRIO = 53
NUM_SCAN = 132
N_PRIV_EXPLICIT = 9
N_PRIV_LATENT = 29
HISTORY_LEN = 10
NUM_ACTIONS = 12

# Scan/depth latent dim emitted by RecurrentDepthBackbone.output_mlp.
# L00-flatstep student: yaw predictor head removed (was 32 + 2), now 32 only.
DEPTH_LATENT_DIM = 32

# obs scales (legged_robot_config.py:114-118 of the snapshot)
SCALE_ANG_VEL = 0.25
SCALE_DOF_POS = 1.0
SCALE_DOF_VEL = 0.05
CLIP_OBS = 100.0
# `normalization.clip_actions = 1.2` is training's SCALED (rad) clip;
# the raw-action clip is 1.2 / action_scale = 1.2 / 0.25 = 4.8.
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
UPDATE_INTERVAL_DEFAULT = 5
DEPTH_BUFFER_LEN = 2

# action_delay
ACTION_BUF_LEN = 8
ACTION_DELAY_VIEW = 1

# Official Unitree RL Lab Go2HV torque-speed envelope. Motoring torque (effort
# and velocity in the same direction) is Y1; braking/regenerative torque is Y2.
# Both stay flat through X1 and decrease linearly to zero at X2.
GO2HV_X1 = 13.5       # rad/s, full-torque knee
GO2HV_X2 = 30.0       # rad/s, no-load speed
GO2HV_Y1 = 20.2       # N m, motoring peak
GO2HV_Y2 = 23.4       # N m, braking peak


def apply_go2hv_torque_speed_curve(torque, joint_vel):
    """Clip requested effort to Unitree's direction-dependent Go2HV envelope."""
    torque = np.asarray(torque, dtype=np.float64)
    joint_vel = np.asarray(joint_vel, dtype=np.float64)
    peak = np.where(joint_vel * torque > 0.0, GO2HV_Y1, GO2HV_Y2)
    speed = np.abs(joint_vel)
    limit = np.where(
        speed < GO2HV_X1,
        peak,
        np.maximum(0.0, peak * (GO2HV_X2 - speed) / (GO2HV_X2 - GO2HV_X1)),
    )
    return np.clip(torque, -limit, limit)

# MJ actuator order (FR,FL,RR,RL × hip,thigh,calf) is the SAME order the trained
# policy expects in proprio. Training builds proprio via legged_robot.reindex()
# on Isaac-order joints, and reindex swaps L/R pairs — making the result match
# MJ order. The reverse mapping was previously applied here (wrong direction):
# it L/R-flipped every joint reading and every action target. Kept as identity
# for callers that still reference it.
MJ_TO_POLICY = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], dtype=np.int64)
# Foot force sensors in go2.xml are declared in [FL, FR, RL, RR]. Reorder to
# MJ/SDK order [FR, FL, RR, RL] for the policy.
FOOT_SENSOR_TO_MJ = np.array([1, 0, 3, 2], dtype=np.int64)

# Default joint angles in MJ/SDK order = training proprio order.
DEFAULT_DOF_POLICY = np.array(
    [-0.1, 0.8, -1.5,  # FR hip thigh calf
      0.1, 0.8, -1.5,  # FL
     -0.1, 1.0, -1.5,  # RR
      0.1, 1.0, -1.5], # RL
    dtype=np.float64,
)

LAY_DOF_MJ = np.array(
    [-0.1, 1.2634956837, -2.72,
      0.1, 1.2486898899, -2.72,
     -0.1, 1.2936768532, -2.72,
      0.1, 1.2838366032, -2.72],
    dtype=np.float64,
)

Q_MIN_MJ = np.array(
    [-1.0472, -1.5708, -2.7227,
     -1.0472, -1.5708, -2.7227,
     -1.0472, -0.5236, -2.7227,
     -1.0472, -0.5236, -2.7227], dtype=np.float64)
Q_MAX_MJ = np.array(
    [1.0472, 3.4907, -0.83776,
     1.0472, 3.4907, -0.83776,
     1.0472, 4.5379, -0.83776,
     1.0472, 4.5379, -0.83776], dtype=np.float64)

PROPRIO_FIELD_NAMES = (
    [f"ang_vel_{a}*0.25" for a in "xyz"]
    + ["roll", "pitch"]
    + ["zero_dyaw", "delta_yaw", "delta_next_yaw"]
    + ["zero_cmdx", "zero_cmdy", "lin_vel_x"]
    + ["env_class!=17", "env_class==17"]
    + [f"dpos_{j}" for j in ("FLh","FLt","FLc","FRh","FRt","FRc","RLh","RLt","RLc","RRh","RRt","RRc")]
    + [f"dvel_{j}" for j in ("FLh","FLt","FLc","FRh","FRt","FRc","RLh","RLt","RLc","RRh","RRt","RRc")]
    + [f"act_{j}"  for j in ("FLh","FLt","FLc","FRh","FRt","FRc","RLh","RLt","RLc","RRh","RRt","RRc")]
    + ["c_FL-0.5", "c_FR-0.5", "c_RL-0.5", "c_RR-0.5"]
)
assert len(PROPRIO_FIELD_NAMES) == N_PROPRIO


# --- math utilities ------------------------------------------------------

def yaw_arrow_mat(theta: float) -> np.ndarray:
    """Row-major 3x3 rotation that takes the local +Z axis (mjGEOM_ARROW points along Z)
    to the horizontal world direction (cos θ, sin θ, 0)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[0.0, -s,  c],
                     [0.0,  c,  s],
                     [-1.0, 0.0, 0.0]], dtype=np.float64)


def add_arrow(scene, pos: np.ndarray, theta: float, length: float,
              radius: float, rgba: tuple) -> None:
    """Push a horizontal arrow into a user mjvScene. Silently no-ops if the geom
    pool is full."""
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


def mj_quat_to_roll_pitch_yaw(q_wxyz: np.ndarray) -> tuple:
    """MuJoCo IMU quat is (w, x, y, z); return (roll, pitch, yaw) ZYX Euler."""
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


# --- checkpoint loading --------------------------------------------------

def load_checkpoint(ckpt_path: Path):
    try:
        return torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except Exception:
        return torch.load(ckpt_path, map_location="cpu", weights_only=False)


def detect_action_delay(ckpt_path: Path) -> int:
    """Look for legged_gym_src/.../legged_robot_config.py next to the checkpoint and grep action_delay."""
    src_root = ckpt_path.parent / "legged_gym_src"
    cfg = src_root / "envs" / "base" / "legged_robot_config.py"
    if not cfg.exists():
        return 0
    text = cfg.read_text()
    m = re.search(r"action_delay\s*=\s*(True|False)", text)
    if m is None:
        return 0
    return 1 if m.group(1) == "True" else 0


# --- depth pipeline ------------------------------------------------------

def normalize_depth(d: np.ndarray, depth_noise_m: float = 0.0) -> torch.Tensor:
    """MuJoCo depth (positive metres) → normalized (1, 58, 87) tensor in [-0.5, 0.5].

    Mirrors Isaac's legged_robot.process_depth_image() exactly:
      crop → per-frame SCALAR additive noise → clip → BICUBIC resize → normalize.
    The noise is one scalar applied to every pixel (not per-pixel) — that's how
    training generated dis_noise=0.02.
    """
    d = d[:-2, 4:-4]                              # crop first (matches Isaac order)
    t = torch.from_numpy(d.astype(np.float32))[None, None]  # (1, 1, 58, 98)
    if depth_noise_m > 0.0:
        shift = depth_noise_m * 2.0 * (torch.rand(1).item() - 0.5)  # scalar per frame
        t = t + shift
    t = torch.clamp(t, DEPTH_NEAR, DEPTH_FAR)
    # BICUBIC, antialias off — matches torchvision 0.11.1 Resize(BICUBIC) on a
    # tensor (see distill_common.resize_depth_bicubic for the provenance).
    t = depth_common.resize_depth_bicubic(t)
    # Re-clip after the resize (legged_robot.py:428/:497): bicubic overshoots at
    # depth edges and training never lets those pixels through.
    t = torch.clamp(t, DEPTH_NEAR, DEPTH_FAR)
    t = t.squeeze(0)                              # (1, 58, 87)
    return t / (DEPTH_FAR - DEPTH_NEAR) - 0.5


def net_foot_contact_forces(m, d, foot_geom_ids) -> np.ndarray:
    """Per-foot resultant contact load, matching the ROS MuJoCo bridge."""
    import mujoco

    net = np.zeros((4, 3), dtype=np.float64)
    contact_force = np.zeros(6, dtype=np.float64)
    for contact_index in range(d.ncon):
        contact = d.contact[contact_index]
        for foot_index, geom_id in enumerate(foot_geom_ids):
            if geom_id >= 0 and (contact.geom1 == geom_id or contact.geom2 == geom_id):
                mujoco.mj_contactForce(m, d, contact_index, contact_force)
                net[foot_index] += contact.frame.reshape(3, 3).T @ contact_force[:3]
                break
    return np.linalg.norm(net, axis=1)


# --- main ----------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path,
                    default=_CKPT_DEFAULT)
    ap.add_argument("--ep_root", type=Path,
                    default=_EP_ROOT_DEFAULT)
    ap.add_argument("--scene", type=Path, default=REPO_ROOT / "resources" / "go2" / "scene_flat.xml")
    ap.add_argument("--lin_vel_x", type=float, default=0.5)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=None,
                    help="Optional NumPy/PyTorch seed for reproducible depth-noise trials.")
    ap.add_argument("--init_jitter", type=float, default=0.0,
                    help="trial-decorrelation only (mirrors lidar_runner): uniform "
                         "+-this many rad on the initial joint angles and "
                         "+-(this*20) deg on the initial base yaw")
    ap.add_argument("--result_tag", type=str, default=None,
                    help="print a machine-readable RESULT line with this tag "
                         "(same fields as lidar_runner where applicable)")
    ap.add_argument("--viewer", action="store_true")
    ap.add_argument("--record", type=Path, default=None, help="Write MP4 of the third-person view to this path; off-screen, no X11 needed")
    ap.add_argument("--record_fps", type=int, default=50)
    ap.add_argument("--record_width", type=int, default=1280)
    ap.add_argument("--record_height", type=int, default=720)
    ap.add_argument("--record_camera", default="follow_cam")
    ap.add_argument("--clean_frame", action="store_true",
                    help="Record the selected camera without heading arrows or the depth PiP overlay")
    ap.add_argument("--reference_crop_depth_pip", action="store_true",
                    help="Suppress heading arrows and place the processed policy-depth PiP at the "
                         "top-right of the final 2400x610 thesis-reference crop")
    # ---- loop latency: DEFAULTS ARE DEPLOY-FAITHFUL (changed 2026-07-28) ------
    # Every cliffside-distill student (post EP 6fb4d46) trains with the 20 ms
    # action delay latched from the first DAgger batch (--delay is REQUIRED for
    # distillation) and with depth.latency_range=[4,5] capture frames at
    # capture_interval=1, i.e. a hard 80-100 ms depth age with no maturity ramp.
    # The ROS/real deploy path enforces the same envelope (16 ms target-publish
    # pad + 80 ms exposure->latent). Running this evaluator with NO action delay is
    # off-distribution in the fresh direction and it is the single biggest
    # standalone-vs-ROS divergence there is — 24 scenes x 3 seeds, D22 m9000,
    # wall scene, paired (run_logs/dryrun/lat_decomp_2026-07-28):
    #     arm                       fell    fell <8s   upright frac   median t_fall
    #     zero latency (old default) 42/72   36/72        0.303          3.5 s
    #     15 ms action delay         20/72    5/72        0.685         12.4 s   p=0.0005
    #     80 ms depth latency only   48/72   43/72        0.291          3.5 s   p=0.39 (NULL)
    # i.e. the delay is what stops the faceplant AT THE HURDLE (36/72 -> 5/72);
    # depth latency is behaviourally neutral and defaults to 0.08 for FIDELITY only.
    # Every student sweep before 2026-07-28 ran undelayed and understated the
    # checkpoints; the teacher sweeps passed --action_delay 1 and did not.
    # Pass --action_delay 0 explicitly for PRE-6fb4d46 policies (T000/L00), which
    # trained undelayed.
    ap.add_argument("--action_delay", type=int, default=1, choices=[0, 1],
                    help="1 (default) = the 20 ms / 1-tick delay every cliffside "
                         "student trains with (latched at fleet level 6). Use 0 only "
                         "for pre-6fb4d46 policies (T000/L00) trained undelayed.")
    ap.add_argument("--proprio_latency_s", type=float, default=0.0,
                    help="Sensor-to-motor latency emulation. Physics keeps applying the "
                         "previous target until this delay expires; 0 disables it. "
                         "Default 0 because --action_delay 1 ALREADY models the whole "
                         "loop delay (the deploy stack's 16 ms publish pad + ~3 ms "
                         "firmware/DDS); setting both double-counts it.")
    ap.add_argument("--depth_latency_s", type=float, default=0.08,
                    help="Delay from depth capture to latent availability. Default 0.08 = "
                         "the enforced deploy constant and training's latency floor. 0 "
                         "selects the legacy Isaac depth_buffer[-2] path instead (which "
                         "is ~100 ms stale anyway, so 0 is NOT 'no latency').")
    ap.add_argument("--depth_mode", choices=["render", "zeros"], default="render")
    ap.add_argument("--update_interval", type=int, default=UPDATE_INTERVAL_DEFAULT)
    ap.add_argument("--ros_timing_emulation", type=int, default=0, choices=[0, 1],
                    help="Emulate the ROS stack's wall-clock latent timing: the "
                         "latent delivery delay drifts across [depth_latency_s, "
                         "+20 ms] on a slow (~7 s) beat — the phase drift between "
                         "the 10 Hz depth clock and the 50 Hz actor clock — plus "
                         "2 ms gaussian scheduler jitter, instead of the "
                         "tick-locked constant delay. Causal test for the "
                         "ROS-vs-standalone student gap (2026-07-27).")
    ap.add_argument("--no_yaw_override", action="store_true",
                    help="DEPRECATED no-op: the L00-flatstep student has no yaw head, so "
                         "prop[6:8] is already held at 0 on the default path. Kept so existing "
                         "scripts don't error on an unknown arg.")
    ap.add_argument("--synthetic_yaw_target_rad", type=float, default=float("nan"),
                    help="DIAGNOSTIC: set prop[6:8] = delta_yaw = target - world_yaw computed from "
                         "MuJoCo's ground-truth yaw (0.0 = face world +X). Off the L00-flatstep "
                         "student's training distribution (it was trained with prop[6:8]=0).")
    ap.add_argument("--friction_scale", type=float, default=1.0,
                    help="Multiply all geom sliding-friction coefficients by this factor after model load (e.g. 2.0 for 2× friction).")
    # L00-* era trained at 40/0.5 (training go2_config.py control block);
    # harness historically hardcoded 30/0.6 — flag added 2026-06-06 to A/B the era mismatch.
    ap.add_argument("--kp", type=float, default=KP,
                    help="PD position gain (stiffness). Default 35.0 (cliffside/T04 default). "
                         "L00-* checkpoints were trained at 40.0 (go2_config.py control block) — "
                         "pass --kp 40 --kd 0.5 to match the training era.")
    ap.add_argument("--kd", type=float, default=KD,
                    help="PD velocity gain (damping). Default 0.6 (legacy harness constant). "
                         "L00-* training-era value is 0.5.")
    ap.add_argument("--clamp_joint_targets", action="store_true",
                    help="Clamp policy targets to Go2 joint limits before PD, matching "
                         "low_level_ctrl. Off preserves the training/legacy standalone path.")
    ap.add_argument("--torque_log", type=Path, default=None,
                    help="If set, log per-substep PD torques (MuJoCo joint order) and times to this .npz file.")
    ap.add_argument("--go2_torque_speed_curve", action="store_true",
                    help="Clip every commanded joint torque to the official Unitree RL Lab Go2HV "
                         "curve (X1=13.5, X2=30 rad/s; Y1=20.2 motoring, Y2=23.4 N m braking).")
    ap.add_argument("--env_class_eq_17", type=int, choices=[0, 1], default=0,
                    help="If 1, set prop[11:13] indicator to (0, 1) — matches "
                         "parkour_flat training (go2_flat / L00-* exptids, env_class==17). "
                         "Default 0 keeps the (1, 0) indicator used by every other env_class.")
    ap.add_argument("--use_sim_truth_vel", type=int, choices=[0, 1], default=0,
                    help="If 1, replace estimator's priv_explicit[0:3] with MuJoCo body-frame qvel. "
                         "Diagnoses whether the learned estimator's velocity prediction is the dominant "
                         "sim2sim bottleneck — when ON, the actor sees a perfect velocity input.")
    ap.add_argument("--cmd_pre", type=float, default=None,
                    help="Pre-switch forward command (m/s). When set together with --switch_step, "
                         "the runner runs cmd_pre for steps [0, switch_step) and --lin_vel_x for "
                         "steps [switch_step, n_ticks). Matches play_deterministic.py's schedule "
                         "for sim2sim comparison.")
    ap.add_argument("--switch_step", type=int, default=-1,
                    help="Policy-tick index at which to switch from --cmd_pre to --lin_vel_x. "
                         "-1 disables switching (constant cmd).")
    ap.add_argument("--stop_on_return_x", action="store_true",
                    help="After base x reaches --return_arm_x, switch cmd_x to zero when "
                         "it returns through --return_x, hold for --return_stop_hold_s, "
                         "then end the run.")
    ap.add_argument("--return_x", type=float, default=0.0,
                    help="Return-x threshold used by --stop_on_return_x.")
    ap.add_argument("--return_arm_x", type=float, default=1.0,
                    help="Forward x that arms the return crossing detector.")
    ap.add_argument("--return_stop_hold_s", type=float, default=2.0,
                    help="Seconds to keep cmd_x=0 after the return crossing before ending.")
    ap.add_argument("--laydown_on_return", action="store_true",
                    help="After a return-x trigger, stop inference and execute the same "
                         "two-second interpolated laydown used by low_level_ctrl.")
    ap.add_argument("--laydown_hold_s", type=float, default=1.0,
                    help="Seconds to hold the lay pose after the return-triggered ramp.")
    ap.add_argument("--cam_pitch_deg", type=float, default=-1.0,
                    help="Override front_depth_cam downward pitch (deg). go2.xml currently ships "
                         "the MEASURED hardware 7.6; training rendered 20. -1 = keep XML value.")
    ap.add_argument("--armature", type=float, default=-1.0,
                    help="Override dof_armature (reflected rotor inertia, kg m^2) on the 12 leg "
                         "joints. go2.xml=0.01; Isaac training ran 0. -1 = keep XML value.")
    ap.add_argument("--joint_frictionloss", type=float, default=-1.0,
                    help="Override dof_frictionloss (dry gearbox friction, N m) on the 12 leg "
                         "joints. go2.xml=0.2; Isaac training ran 0 (URDF has no <dynamics>, and "
                         "randomize_joint_friction=False). -1 = keep XML value.")
    ap.add_argument("--joint_damping", type=float, default=-1.0,
                    help="Override dof_damping (viscous, N m s/rad) on the 12 leg joints. "
                         "go2.xml=0.01; Isaac training ran 0. -1 = keep XML value.")
    ap.add_argument("--foot_log", type=Path, default=None,
                    help="If set, dump per-tick world foot positions (4x3, order FR,FL,RR,RL) "
                         "to this .npz for geometric hurdle-strike scoring.")
    ap.add_argument("--log_npz", type=Path, default=None,
                    help="If set, dump per-tick {obs (753), depth_latent (32), action_raw (12), "
                         "cmd_x, base state} to this .npz for diff with the Isaac play_deterministic "
                         "trace.")
    ap.add_argument("--contact_threshold", type=float, default=10.0,
                    help="Foot-force threshold (N) for the contact bit prop[49:53] OR-filter. "
                         "Default 10.0 (parkour-wall tuning, matches videos/threshold_sweep/). "
                         "For L00-* flat baseline (parkour_flat / env_class==17) pass 1.0 — "
                         "10 N over-filters the flat-walk foot signal and the policy falls at "
                         "speeds it can otherwise track. The ROS depth_encoder_node uses 1.0.")
    ap.add_argument("--depth_noise_m", type=float, default=0.0,
                    help="Uniform per-pixel depth noise (±N meters, before normalize). "
                         "Training added ±0.02 m noise; deploy is clean. Adding it might "
                         "stabilize the recurrent encoder if it was trained to expect noise. "
                         "Apply BEFORE normalize_depth so the clip+normalize sees the noisy input.")
    ap.add_argument("--depth_noise_scale", type=float, default=0.0,
                    help="Latest cliffside training model: 0=clean, "
                         "0.5=reality-tuned, 1=2x artifact rates. Supersedes "
                         "--depth_noise_m when nonzero.")
    ap.add_argument("--stereo_shadow", choices=("auto", "on", "off"), default="auto",
                    help="D435 stereo geometry; auto enables with depth noise.")
    ap.add_argument("--init_x", type=float, default=0.0,
                    help="Override the initial base_x. Default 0 (scene origin). Use to "
                         "spawn the robot closer to obstacles so the depth camera sees them "
                         "from the first frame (e.g. --init_x 2.0 on a 2hurdles scene places "
                         "the robot ~1m before the first hurdle).")
    ap.add_argument("--init_y", type=float, default=0.0,
                    help="Override the initial base_y. Default 0 (scene centerline).")
    ap.add_argument("--init_z", type=float, default=0.42,
                    help="Override the initial base_z. Default 0.42 (matches Isaac training reset).")
    ap.add_argument("--pre_settle_seconds", type=float, default=0.0,
                    help="If >0, run kp=50/kd=1.0 PD holding DEFAULT_DOF_POLICY for this many "
                         "seconds BEFORE the policy loop starts (action=0, policy not ticking, "
                         "history/encoder state never updated). Reproduces the ROS FSM's "
                         "lay→stand→engage init condition — robot lands and settles to the "
                         "standing PD equilibrium with all four feet planted, then the policy "
                         "starts cold from there. Used to verify whether the in-air-drop reset "
                         "is what makes the standalone harness walk vs the ROS stack fall.")
    args = ap.parse_args()
    if not 0.0 <= args.depth_noise_scale <= 1.0:
        ap.error("--depth_noise_scale must be in [0, 1]")
    if args.proprio_latency_s < 0.0 or args.depth_latency_s < 0.0:
        ap.error("latencies must be non-negative")
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    # MuJoCo's offscreen depth renderer needs an OpenGL context. Default to EGL so
    # headless / SSH runs work; the user can override before invocation if they want
    # GLFW/OSMESA. Set this BEFORE importing mujoco.
    if (args.depth_mode == "render" or args.record is not None) and "MUJOCO_GL" not in os.environ:
        os.environ["MUJOCO_GL"] = "egl"

    add_extreme_parkour_to_path(args.ep_root)
    # depth_backbone.py does `import torchvision` at top but never uses it for the
    # parts we need; the env's torchvision is ABI-mismatched against torch, so stub
    # it so the import succeeds.
    if "torchvision" not in sys.modules:
        import types as _types
        sys.modules["torchvision"] = _types.ModuleType("torchvision")
    import mujoco
    from rsl_rl.modules.actor_critic import Actor, get_activation
    from rsl_rl.modules.depth_backbone import DepthOnlyFCBackbone58x87, RecurrentDepthBackbone
    from rsl_rl.modules.estimator import Estimator

    # ----- checkpoint
    print(f"[ckpt] loading {args.ckpt}")
    ckpt = load_checkpoint(args.ckpt)
    print(f"[ckpt] iter = {ckpt.get('iter', '?')}")
    encoder_output_dim = int(
        ckpt["depth_encoder_state_dict"]["output_mlp.0.weight"].shape[0])
    if encoder_output_dim not in (DEPTH_LATENT_DIM, DEPTH_LATENT_DIM + 2):
        raise ValueError(
            f"unsupported depth encoder output width {encoder_output_dim}; "
            f"expected {DEPTH_LATENT_DIM} (no yaw) or {DEPTH_LATENT_DIM + 2} (yaw)")
    has_yaw_head = encoder_output_dim == DEPTH_LATENT_DIM + 2
    print(f"[ckpt] depth output = {encoder_output_dim} "
          f"({'32 latent + 2 yaw' if has_yaw_head else '32 latent, no yaw'})")

    # ----- action delay (default 0; pass --action_delay 1 to enable)
    print(f"[cli] action_delay = {args.action_delay}")
    print(f"[cli] PD gains kp={args.kp} kd={args.kd}")
    if args.go2_torque_speed_curve:
        print("[cli] Go2HV torque-speed curve enabled: "
              "X1=13.5 X2=30.0 rad/s, Y1=20.2 Y2=23.4 N m")

    # ----- build policy modules (CPU)
    activation = get_activation("elu")
    actor = Actor(
        num_prop=N_PROPRIO, num_scan=NUM_SCAN, num_actions=NUM_ACTIONS,
        scan_encoder_dims=[128, 64, 32],
        actor_hidden_dims=[512, 256, 128],
        priv_encoder_dims=[64, 20],
        num_priv_latent=N_PRIV_LATENT, num_priv_explicit=N_PRIV_EXPLICIT,
        num_hist=HISTORY_LEN, activation=activation, tanh_encoder_output=False,
    )
    actor.load_state_dict(ckpt["depth_actor_state_dict"], strict=True)
    actor.eval()

    estimator = Estimator(input_dim=N_PROPRIO, output_dim=N_PRIV_EXPLICIT, hidden_dims=[128, 64])
    estimator.load_state_dict(ckpt["estimator_state_dict"], strict=True)
    estimator.eval()

    depth_backbone = DepthOnlyFCBackbone58x87(
        prop_dim=N_PROPRIO, scandots_output_dim=DEPTH_LATENT_DIM,
        hidden_state_dim=512, output_activation=None, num_frames=1,
    )
    # RecurrentDepthBackbone selects its 32-vs-34 output head from this toggle.
    # The image backbone itself always remains 32-D for both architectures.
    os.environ["EP_USE_YAW"] = "1" if has_yaw_head else "0"
    depth_encoder = RecurrentDepthBackbone(depth_backbone, env_cfg=None)
    depth_encoder.load_state_dict(ckpt["depth_encoder_state_dict"], strict=True)
    depth_encoder.eval()

    # actor sanity check
    fc = actor.actor_backbone[0]
    assert fc.in_features == 114, f"actor first layer expected 114, got {fc.in_features}"
    print(f"[actor] in_features={fc.in_features}  num_prop={N_PROPRIO}  scan_lat=32  priv_expl={N_PRIV_EXPLICIT}  hist_lat=20  → 114 ✓")

    # MJ_TO_POLICY is now identity — kept for compatibility with older callers.
    assert np.array_equal(MJ_TO_POLICY, np.arange(12)), "MJ_TO_POLICY expected to be identity after L/R-swap fix"
    print(f"[reindex] mj→policy = identity (training proprio is already in MJ order after legged_robot.reindex())")
    print(f"[default] MJ-order default DoF = {DEFAULT_DOF_POLICY.tolist()}")

    # ----- mujoco model
    print(f"[mj] loading {args.scene}")
    m = mujoco.MjModel.from_xml_path(str(args.scene))
    if args.friction_scale != 1.0:
        m.geom_friction[:, 0] *= args.friction_scale
        print(f"[mj] friction_scale={args.friction_scale}  sliding friction × {args.friction_scale}")
    # Actuator-drivetrain overrides on the 12 leg DOFs (dof 6..17; 0..5 are the
    # free joint). go2.xml ships armature=0.01 frictionloss=0.2 damping=0.01,
    # but Isaac training ran armature=0, and the URDF declares no joint
    # <dynamics>, so Isaac's joint friction/damping were 0 too. These flags let
    # a sweep span both references. -1 = leave the XML value untouched.
    for flag, field, label in ((args.armature, m.dof_armature, "armature"),
                               (args.joint_frictionloss, m.dof_frictionloss, "frictionloss"),
                               (args.joint_damping, m.dof_damping, "damping")):
        if flag >= 0.0:
            field[6:6 + 12] = flag
            print(f"[mj] dof_{label}[legs] = {flag}")
    d = mujoco.MjData(m)
    foot_geom_ids = [
        mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in ("FR", "FL", "RR", "RL")
    ]
    if any(geom_id < 0 for geom_id in foot_geom_ids):
        raise RuntimeError(f"missing foot collision geom(s): {foot_geom_ids}")
    cam_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "front_depth_cam")
    assert cam_id >= 0, "front_depth_cam not found in model"
    if args.cam_pitch_deg >= 0.0:
        # Rebuild the camera frame for a downward pitch p, matching the XML
        # convention xyaxes="0 -1 0   sin(p) 0 cos(p)" (camera looks along -z,
        # so -z = (cos p, 0, -sin p): forward and down by p).
        p = np.radians(args.cam_pitch_deg)
        xa = np.array([0.0, -1.0, 0.0])
        ya = np.array([np.sin(p), 0.0, np.cos(p)])
        za = np.cross(xa, ya)
        mat = np.column_stack((xa, ya, za)).ravel()
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, mat)
        m.cam_quat[cam_id] = quat
        print(f"[mj] front_depth_cam pitch = {args.cam_pitch_deg:.2f} deg down")

    # initialize qpos to a standing pose (DEFAULT_DOF_POLICY is in MJ order)
    default_dof_mj = DEFAULT_DOF_POLICY
    d.qpos[:3] = [args.init_x, args.init_y, args.init_z]   # default (0,0,0.42) = Isaac init_state.pos
    d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]                  # w, x, y, z (identity)
    d.qpos[7:7 + 12] = default_dof_mj
    if args.init_jitter > 0.0:
        d.qpos[7:7 + 12] += np.random.uniform(-args.init_jitter, args.init_jitter, 12)
        yaw0 = np.deg2rad(np.random.uniform(-1.0, 1.0) * args.init_jitter * 20.0)
        d.qpos[3:7] = [np.cos(yaw0 / 2), 0.0, 0.0, np.sin(yaw0 / 2)]
        print(f"[init] jitter={args.init_jitter} seed={args.seed} yaw0={np.rad2deg(yaw0):+.2f} deg")
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)

    # Optional pre-settle: hold the standing PD target for args.pre_settle_seconds
    # BEFORE the policy starts. action=0, policy not running, history_ring stays
    # None, depth encoder not called — this exactly mirrors the ROS FSM's
    # lay→stand→engage handoff (robot settles under standing PD before the
    # policy ticks). Used to test whether the in-air-drop reset is what makes
    # the standalone walk vs the ROS dry-run fall.
    if args.pre_settle_seconds > 0.0:
        n_settle = int(args.pre_settle_seconds / m.opt.timestep)
        kp_settle = np.full(12, 50.0)
        kd_settle = np.full(12, 1.0)
        for _ in range(n_settle):
            tau = kp_settle * (default_dof_mj - d.qpos[7:7 + 12]) - kd_settle * d.qvel[6:6 + 12]
            if args.go2_torque_speed_curve:
                tau = apply_go2hv_torque_speed_curve(tau, d.qvel[6:6 + 12])
            d.ctrl[:] = tau
            mujoco.mj_step(m, d)
        print(f"[pre-settle] held standing PD for {args.pre_settle_seconds:.2f}s "
              f"({n_settle} substeps) → base_z={float(d.qpos[2]):.3f}")

    # ----- two separate renderers so depth always renders at the exact Isaac
    # aspect (DEPTH_RAW_W × DEPTH_RAW_H = 106 × 60 → hFOV 87° at fovy=56.49°).
    # If we shared a 1280×720 renderer for both passes, the depth pass would
    # render at aspect 1.778 → hFOV ≈ 87.6°, a 0.6 % horizontal stretch the
    # policy never saw in training. Two renderers cost ~2× GL state at startup
    # and behave fine on EGL in practice.
    depth_renderer = None
    record_renderer = None
    if args.depth_mode == "render":
        depth_renderer = mujoco.Renderer(m, height=DEPTH_RAW_H, width=DEPTH_RAW_W)
        depth_renderer.enable_depth_rendering()
        depth_scene_option = depth_common.mujoco_depth_scene_option()
        stereo_enabled = (args.depth_noise_scale > 0.0
                          if args.stereo_shadow == "auto"
                          else args.stereo_shadow == "on")
        stereo_depth = (depth_common.MujocoStereoDepth(
            m, cam_id,
            baseline=depth_common.TRAIN_DEPTH_CFG["stereo_baseline"],
            eps=depth_common.TRAIN_DEPTH_CFG["stereo_eps"],
            radial_clip=depth_common.depth_radial_clip_enabled(),
        ) if stereo_enabled else None)
        depth_degrader = (depth_common.TrainDepthDegrader(
            noise_scale=args.depth_noise_scale)
            if args.depth_noise_scale > 0.0 or stereo_enabled else None)
        depth_radial_max = (depth_common.radial_maxperp_mask()
                            if depth_common.depth_radial_clip_enabled() else None)
        print(f"[depth] source={depth_common.TRAIN_DEPTH_SOURCE_COMMIT} "
              f"stereo={'on' if stereo_enabled else 'off'} "
              f"noise_scale={args.depth_noise_scale:.2f}")
    if args.record is not None:
        record_renderer = mujoco.Renderer(m, height=args.record_height,
                                          width=args.record_width)

    # ----- viewer
    viewer = None
    if args.viewer:
        from mujoco import viewer as mj_viewer
        viewer = mj_viewer.launch_passive(m, d)

    # ----- video recorder (off-screen, no X11)
    video_writer = None
    record_every = 1
    if args.record is not None:
        import imageio
        video_writer = imageio.get_writer(str(args.record), fps=args.record_fps, codec="libx264",
                                          quality=None, output_params=["-crf", "23", "-preset", "ultrafast", "-pix_fmt", "yuv420p"])
        # 50 Hz policy → capture at args.record_fps (default 50), so one frame per policy tick.
        record_every = max(1, int(round(50 / args.record_fps)))
        print(f"[video] recording {args.record} at {args.record_fps} fps ({args.record_width}×{args.record_height}, cam={args.record_camera})")

    # ----- runtime state
    obs_full = torch.zeros(1, N_PROPRIO + NUM_SCAN + N_PRIV_EXPLICIT + N_PRIV_LATENT + HISTORY_LEN * N_PROPRIO)
    history_ring = None     # (1, HISTORY_LEN, 53)
    last_action = torch.zeros(1, NUM_ACTIONS)        # in policy order
    action_buf = deque([torch.zeros(1, NUM_ACTIONS) for _ in range(ACTION_BUF_LEN)], maxlen=ACTION_BUF_LEN)
    depth_ring = deque(maxlen=DEPTH_BUFFER_LEN)
    pending_depth_latents = deque()
    cached_depth_latent = torch.zeros(1, DEPTH_LATENT_DIM)
    # Raw encoder yaw prediction. For 34-output checkpoints this is multiplied
    # by 1.5 before feeding proprio slots 6:8, exactly as in distillation.
    # It stays zero for the 32-output no-yaw checkpoints.
    cached_yaw_pred = torch.zeros(1, 2)
    encoder_seen = False
    last_depth_norm_np = None       # (58, 87) float32 in [-0.5, 0.5] for video PiP

    n_policy_ticks = int(args.seconds / (m.opt.timestep * DECIMATION))
    print(f"[run] {n_policy_ticks} policy ticks at 50 Hz → {args.seconds:.1f} s of sim")
    policy_dt = float(m.opt.timestep * DECIMATION)
    proprio_delay_steps = int(round(args.proprio_latency_s / m.opt.timestep))
    proprio_delay_steps = min(DECIMATION, max(0, proprio_delay_steps))
    depth_delay_ticks = int(round(args.depth_latency_s / policy_dt))
    print(
        f"[latency] action_delay={'20 ms (1 tick)' if args.action_delay else 'off'}, "
        f"proprio={proprio_delay_steps * m.opt.timestep * 1000:.0f} ms "
        f"({proprio_delay_steps}/{DECIMATION} substeps; requested "
        f"{args.proprio_latency_s * 1000:.0f} ms), depth={depth_delay_ticks * policy_dt * 1000:.0f} ms "
        f"({depth_delay_ticks} policy ticks; requested {args.depth_latency_s * 1000:.0f} ms)"
    )
    if args.action_delay and proprio_delay_steps > 0:
        print(f"[latency] WARNING: --action_delay 1 and --proprio_latency_s "
              f"{args.proprio_latency_s:g} both model the SAME physical loop delay — "
              f"this run applies "
              f"{20 + proprio_delay_steps * m.opt.timestep * 1000:.0f} ms of it. "
              f"Pick one.")
    held_target_dof_mj = DEFAULT_DOF_POLICY.copy()

    torque_max = 0.0
    base_z_min, base_z_max = 1.0, 0.0
    base_x_max = 0.0
    # Peak lateral excursion over the WHOLE episode. final_y only reports where
    # the run ended, so a trial that swings wide mid-course and comes back looks
    # identical to one that tracked the lane. Reported, never terminated on, so
    # the corridor threshold stays a scoring choice made after the sweep.
    base_absy_max = 0.0
    fwd_velocity_window = deque(maxlen=100)   # 2 s at 50 Hz
    torque_log = [] if args.torque_log is not None else None
    torque_time_log = [] if args.torque_log is not None else None
    last_contact_raw = np.zeros(4, dtype=np.float64)   # RAW prev contact for OR-filter (not the filtered output)
    return_armed = False
    return_stop_tick = None
    return_hold_ticks = max(0, int(round(args.return_stop_hold_s /
                                         (m.opt.timestep * DECIMATION))))
    completed_ticks = n_policy_ticks

    # --- sim2sim trace buffers ---------------------------------------
    obs_dim = N_PROPRIO + NUM_SCAN + N_PRIV_EXPLICIT + N_PRIV_LATENT + HISTORY_LEN * N_PROPRIO
    log_obs = np.zeros((n_policy_ticks, obs_dim), dtype=np.float32) if args.log_npz else None
    log_dlat = np.zeros((n_policy_ticks, DEPTH_LATENT_DIM), dtype=np.float32) if args.log_npz else None
    log_yaw_pred = np.zeros((n_policy_ticks, 2), dtype=np.float32) if args.log_npz else None
    log_act_raw = np.zeros((n_policy_ticks, NUM_ACTIONS), dtype=np.float32) if args.log_npz else None
    log_cmd_x = np.zeros((n_policy_ticks,), dtype=np.float32) if args.log_npz else None
    log_base_pos = np.zeros((n_policy_ticks, 3), dtype=np.float32) if args.log_npz else None
    log_base_quat = np.zeros((n_policy_ticks, 4), dtype=np.float32) if args.log_npz else None
    log_base_vel_world = np.zeros((n_policy_ticks, 3), dtype=np.float32) if args.log_npz else None
    log_world_yaw = np.zeros((n_policy_ticks,), dtype=np.float32) if args.log_npz else None
    log_foot_pos = np.zeros((n_policy_ticks, 4, 3), dtype=np.float32) if args.foot_log else None
    log_foot_base = np.zeros((n_policy_ticks, 3), dtype=np.float32) if args.foot_log else None

    t0 = time.time()
    for global_counter in range(n_policy_ticks):
        # -------- sensors → proprio ------------------------------------
        sd = d.sensordata
        # joints in MuJoCo order
        qj_mj = sd[0:12].copy()
        dqj_mj = sd[12:24].copy()
        # IMU (body frame): quat at 36:40, gyro at 40:43
        quat_wxyz = sd[36:40].copy()
        ang_vel_body = sd[40:43].copy()
        roll, pitch, world_yaw = mj_quat_to_roll_pitch_yaw(quat_wxyz)
        # Go2 LowState exposes one scalar contact-load value per foot. Use the
        # norm of the summed MuJoCo contact vector, as the ROS bridge does.
        foot_forces = net_foot_contact_forces(m, d, foot_geom_ids)
        # Threshold tunable via --contact_threshold (default 10.0); OR with RAW
        # previous-step contact (matches legged_robot.py:332-334; the filtered
        # version would latch permanently after first ground touch, so we OR
        # with raw_prev). Picking the right threshold is task-dependent:
        #   - Parkour walls / hurdles (P10-*, P11-*, L00-01 wall variants):
        #     10 N. The "8/15 honest clears vs 0/15 for 2 N" tuning in
        #     videos/threshold_sweep/ measured partial-weight contacts on a
        #     vertical wall; 2 N latches the bit through the whole stance and
        #     the policy loses gait-phase info.
        #   - L00-* flat baseline (parkour_flat / env_class==17): 1.0. The
        #     10 N value over-filters the flat-walk foot signal — stale
        #     contact bits → depth-encoder yaw_pred saturates at ~−56° → fall.
        #     Verified L00-02 ckpt 14500 on scene_flat.xml at 1.5 m/s:
        #     10 N → τ_peak 102, base_z 0.095 (face-plant); 1 N → τ_peak 31,
        #     base_z 0.28, vx_avg +1.29 m/s. The ROS depth_encoder_node uses
        #     1.0 N for this reason.
        contact_now = (foot_forces > args.contact_threshold).astype(np.float64)
        contact_filt = np.maximum(contact_now, last_contact_raw)
        last_contact_raw = contact_now

        # qj_mj/dqj_mj are already in MJ order, which equals training proprio order.
        qj_policy = qj_mj
        dqj_policy = dqj_mj
        last_action_pol = last_action.numpy().reshape(-1)

        prop = np.zeros(N_PROPRIO, dtype=np.float64)
        prop[0:3]   = ang_vel_body * SCALE_ANG_VEL
        prop[3]     = roll
        prop[4]     = pitch
        prop[5]     = 0.0                        # 0 * delta_yaw
        prop[6]     = 0.0                        # delta_yaw (overwritten below from yaw_pred)
        prop[7]     = 0.0                        # delta_next_yaw
        prop[8]     = 0.0                        # 0 * commands[0]
        prop[9]     = 0.0                        # 0 * commands[1]
        if return_stop_tick is not None:
            cmd_x_now = 0.0
        elif args.switch_step >= 0 and global_counter < args.switch_step and args.cmd_pre is not None:
            cmd_x_now = args.cmd_pre
        else:
            cmd_x_now = args.lin_vel_x
        prop[10]    = cmd_x_now                  # lin_vel_x (per-step, supports switch schedule)
        prop[11]    = 0.0 if args.env_class_eq_17 else 1.0   # env_class != 17
        prop[12]    = 1.0 if args.env_class_eq_17 else 0.0   # env_class == 17
        prop[13:25] = (qj_policy - DEFAULT_DOF_POLICY) * SCALE_DOF_POS
        prop[25:37] = dqj_policy * SCALE_DOF_VEL
        prop[37:49] = last_action_pol
        prop[49:53] = contact_filt - 0.5

        prop_t = torch.from_numpy(prop).float().unsqueeze(0)        # (1, 53)

        # -------- history buffer (yaw-masked copy) ---------------------
        prop_hist_frame = prop_t.clone()
        prop_hist_frame[:, 6:8] = 0.0
        if history_ring is None:
            # Match Isaac (legged_robot.py:466): obs_history_buf starts as all zeros
            # at episode reset, not as repeated copies of the first frame.
            history_ring = torch.zeros(1, HISTORY_LEN, N_PROPRIO)
        else:
            history_ring = torch.cat([history_ring[:, 1:], prop_hist_frame.unsqueeze(1)], dim=1)

        # -------- depth render + encoder (every update_interval ticks) -
        if global_counter % args.update_interval == 0:
            if args.depth_mode == "zeros":
                depth_in = torch.zeros(1, DEPTH_TARGET_H, DEPTH_TARGET_W)
            else:
                depth_renderer.update_scene(
                    d, camera="front_depth_cam", scene_option=depth_scene_option)
                raw = depth_renderer.render()                # (60, 106) positive metres
                if stereo_depth is not None:
                    raw = stereo_depth(d, raw)
                elif depth_radial_max is not None:
                    raw = np.where(
                        raw > depth_radial_max, np.float32(DEPTH_FAR), raw).astype(np.float32)
                raw = depth_common.clip_mujoco_too_close_to_far(raw)
                if depth_degrader is not None:
                    depth_in = depth_degrader(raw)
                else:
                    depth_in = normalize_depth(raw, depth_noise_m=args.depth_noise_m)
                last_depth_norm_np = depth_in.squeeze(0).numpy()
                if not depth_ring:
                    depth_ring.append(depth_in)
                depth_ring.append(depth_in)
            # Deployment publishes a latent after a fixed capture-to-publish
            # delay. The legacy path instead mirrors Isaac's depth_buffer[-2].
            depth_for_encoder = (depth_in if depth_delay_ticks > 0 else
                                 depth_ring[-DEPTH_BUFFER_LEN]
                                 if len(depth_ring) >= DEPTH_BUFFER_LEN else depth_in)
            obs_student = prop_t.clone()
            obs_student[:, 6:8] = 0.0
            with torch.no_grad():
                out = depth_encoder(depth_for_encoder, obs_student)
            next_depth_latent = out[:, :DEPTH_LATENT_DIM].detach()
            next_yaw_pred = (out[:, DEPTH_LATENT_DIM:DEPTH_LATENT_DIM + 2].detach()
                             if has_yaw_head else torch.zeros(1, 2))
            if depth_delay_ticks > 0:
                eff_delay_ticks = depth_delay_ticks
                if args.ros_timing_emulation:
                    t_s = global_counter * policy_dt
                    drift = 0.5 * (1.0 + math.sin(2.0 * math.pi * t_s / 7.0))
                    delay_s = (args.depth_latency_s + 0.020 * drift
                               + float(np.random.normal(0.0, 0.002)))
                    eff_delay_ticks = max(1, int(round(delay_s / policy_dt)))
                pending_depth_latents.append(
                    (global_counter + eff_delay_ticks,
                     next_depth_latent, next_yaw_pred))
            else:
                cached_depth_latent = next_depth_latent
                cached_yaw_pred = next_yaw_pred
                encoder_seen = True

        while (pending_depth_latents and
               pending_depth_latents[0][0] <= global_counter):
            _, cached_depth_latent, cached_yaw_pred = pending_depth_latents.popleft()
            encoder_seen = True

        # -------- heading slots prop[6:8] (every tick) -----------------
        # No-yaw students hold prop[6:8] at zero. Yaw-head students consume
        # 1.5 * the two encoder outputs, matching on_policy_runner.py. The
        # --synthetic_yaw_target_rad override remains as a diagnostic for
        # injecting a ground-truth-derived heading, but that is now OFF the
        # student's training distribution. --no_yaw_override is redundant.
        if encoder_seen:
            prop_t = prop_t.clone()
            if not np.isnan(args.synthetic_yaw_target_rad):
                # DIAGNOSTIC: override with ground-truth-derived delta_yaw =
                # target - world_yaw. Wrap to (-pi, pi] and clip to ±0.5 rad
                # (28°) so large drifts produce a *strong but in-range*
                # correction signal, not a wraparound spike.
                delta = args.synthetic_yaw_target_rad - world_yaw
                delta = (delta + np.pi) % (2 * np.pi) - np.pi
                delta = float(np.clip(delta, -0.5, 0.5))
                prop_t[:, 6:8] = delta
            elif has_yaw_head:
                prop_t[:, 6:8] = 1.5 * cached_yaw_pred
            else:
                prop_t[:, 6:8] = 0.0

        # -------- assemble full obs vector -----------------------------
        obs_full[:, 0:N_PROPRIO] = prop_t
        # scan slot (132): zeros — actor ignores this when scandots_latent is given
        obs_full[:, N_PROPRIO:N_PROPRIO + NUM_SCAN] = 0.0
        # priv_explicit slot (9): filled by estimator(prop)
        with torch.no_grad():
            priv_expl = estimator(prop_t)
        if args.use_sim_truth_vel:
            # Body-frame qvel * lin_vel obs scale (=2.0). Remaining 6 dims are
            # zero pads in training (legged_robot.py:689-691) — match exactly.
            vw = d.qvel[0:3].copy()
            cy, sy = np.cos(-world_yaw), np.sin(-world_yaw)
            gt_vx_body = cy * vw[0] - sy * vw[1]
            gt_vy_body = sy * vw[0] + cy * vw[1]
            gt_vz_body = float(vw[2])
            priv_expl = torch.zeros_like(priv_expl)
            priv_expl[0, 0] = float(gt_vx_body) * 2.0
            priv_expl[0, 1] = float(gt_vy_body) * 2.0
            priv_expl[0, 2] = gt_vz_body * 2.0
        obs_full[:, N_PROPRIO + NUM_SCAN:N_PROPRIO + NUM_SCAN + N_PRIV_EXPLICIT] = priv_expl
        # priv_latent slot (29): zeros — actor ignores this with hist_encoding=True
        obs_full[:, N_PROPRIO + NUM_SCAN + N_PRIV_EXPLICIT:N_PROPRIO + NUM_SCAN + N_PRIV_EXPLICIT + N_PRIV_LATENT] = 0.0
        # history (10 * 53)
        obs_full[:, -HISTORY_LEN * N_PROPRIO:] = history_ring.reshape(1, -1)
        obs_full.clamp_(-CLIP_OBS, CLIP_OBS)

        # -------- step-0 diagnostic ------------------------------------
        if global_counter == 0:
            arr = prop_t.squeeze(0).numpy()
            print("[step-0 proprio]")
            for name, v in zip(PROPRIO_FIELD_NAMES, arr):
                print(f"  {name:20s} = {v:+.4f}")

        # -------- actor ------------------------------------------------
        with torch.no_grad():
            # Pull out the history-encoder embedding (the deploy-time stand-in
            # for the privileged-info latent that encodes friction / mass / etc.).
            hist_latent = actor.infer_hist_latent(obs_full)
            action_raw = actor(obs_full, hist_encoding=True, eval=False, scandots_latent=cached_depth_latent)

        # -------- sim2sim trace logging --------------------------------
        if log_obs is not None:
            log_obs[global_counter] = obs_full[0].cpu().numpy()
            log_dlat[global_counter] = cached_depth_latent[0].cpu().numpy()
            log_yaw_pred[global_counter] = cached_yaw_pred[0].cpu().numpy()
            log_act_raw[global_counter] = action_raw[0].cpu().numpy()
            log_cmd_x[global_counter] = cmd_x_now
            log_base_pos[global_counter] = d.qpos[0:3]
            log_base_quat[global_counter] = d.qpos[3:7]   # (w, x, y, z)
            log_base_vel_world[global_counter] = d.qvel[0:3]
            log_world_yaw[global_counter] = world_yaw
        if log_foot_pos is not None:
            log_foot_pos[global_counter] = d.geom_xpos[foot_geom_ids]
            log_foot_base[global_counter] = d.qpos[0:3]
        action_clipped = action_raw.clamp(-CLIP_ACTIONS, CLIP_ACTIONS)
        # MATCH TRAINING: feed *unclipped* actor output back into proprio[37:49]
        # (training stores raw in action_history_buf at legged_robot.py:124).
        last_action = action_raw.detach()

        # -------- action delay -----------------------------------------
        action_buf.append(action_clipped.detach())
        if args.action_delay:
            action_to_apply = action_buf[-1 - ACTION_DELAY_VIEW]
        else:
            action_to_apply = action_clipped

        # -------- targets in MuJoCo order ------------------------------
        action_pol = action_to_apply.squeeze(0).numpy()
        # DEFAULT_DOF_POLICY is in MJ order; actor output is also in MJ order
        # (training applies reindex() inside env.step before motors see it).
        target_dof_mj = DEFAULT_DOF_POLICY + ACTION_SCALE * action_pol
        if args.clamp_joint_targets:
            target_dof_mj = np.clip(target_dof_mj, Q_MIN_MJ, Q_MAX_MJ)

        # -------- PD + 4 substeps --------------------------------------
        # ROS takes a sensor snapshot, runs inference, then publishes after the
        # configured proprioception latency while MuJoCo keeps advancing. Mirror
        # that by retaining the previous target for the delayed substeps.
        for substep in range(DECIMATION):
            active_target = (held_target_dof_mj
                             if substep < proprio_delay_steps else target_dof_mj)
            torque = args.kp * (active_target - d.qpos[7:7 + 12]) - args.kd * d.qvel[6:6 + 12]
            if args.go2_torque_speed_curve:
                torque = apply_go2hv_torque_speed_curve(torque, d.qvel[6:6 + 12])
            d.ctrl[:] = torque
            mujoco.mj_step(m, d)
            if torque_log is not None:
                torque_log.append(torque.copy())
                torque_time_log.append(float(d.time))
            if viewer is not None:
                viewer.sync()
        held_target_dof_mj = target_dof_mj.copy()

        # -------- diagnostics ------------------------------------------
        torque_max = max(torque_max, float(np.max(np.abs(torque))))
        base_z = float(d.qpos[2])
        base_z_min = min(base_z_min, base_z)
        base_x_max = max(base_x_max, float(d.qpos[0]))
        base_absy_max = max(base_absy_max, abs(float(d.qpos[1])))
        base_z_max = max(base_z_max, base_z)
        fwd_velocity_window.append(float(d.qvel[0]))   # world-frame x velocity

        if args.stop_on_return_x:
            base_x = float(d.qpos[0])
            if base_x >= args.return_arm_x:
                return_armed = True
            if return_armed and return_stop_tick is None and base_x <= args.return_x:
                return_stop_tick = global_counter + 1
                print(f"[return-x] hard stop triggered at tick={return_stop_tick} "
                      f"x={base_x:+.4f}; holding cmd_x=0 for "
                      f"{args.return_stop_hold_s:.2f}s")
            if (return_stop_tick is not None and
                    global_counter + 1 - return_stop_tick >= return_hold_ticks):
                completed_ticks = global_counter + 1
                print(f"[return-x] hold complete at tick={completed_ticks} "
                      f"pos=({float(d.qpos[0]):+.4f},{float(d.qpos[1]):+.4f})")
                break

        if global_counter > 0 and global_counter % 50 == 0:
            mean_vx = float(np.mean(fwd_velocity_window))
            dy_now = float(cached_yaw_pred[0, 0])
            dy_next = float(cached_yaw_pred[0, 1])
            hl = hist_latent[0].detach()
            # estimator vx prediction (body-frame, undo lin_vel scale=2.0)
            est_vx = float(priv_expl[0, 0]) / 2.0
            est_vy = float(priv_expl[0, 1]) / 2.0
            est_vz = float(priv_expl[0, 2]) / 2.0
            # ground-truth body-frame velocity from MuJoCo qvel rotated by -world_yaw
            vw = d.qvel[0:3].copy()
            cy, sy = np.cos(-world_yaw), np.sin(-world_yaw)
            gt_vx_body = cy * vw[0] - sy * vw[1]
            gt_vy_body = sy * vw[0] + cy * vw[1]
            print(f"[t={global_counter*0.02:5.2f}s] base_z={base_z:.3f}  pos=({float(d.qpos[0]):+5.2f},{float(d.qpos[1]):+5.2f})  vx_avg(2s)={mean_vx:+.3f}  est_vx={est_vx:+.3f}  gt_vx={gt_vx_body:+.3f}  est_vy={est_vy:+.3f}  gt_vy={gt_vy_body:+.3f}  est_vz={est_vz:+.3f}  τ_peak={torque_max:.1f}  d_lat|={float(cached_depth_latent.norm()):.2f}  yaw_pred=({np.degrees(dy_now):+6.1f}°,{np.degrees(dy_next):+6.1f}°)  world_yaw={np.degrees(world_yaw):+6.1f}°  hist|={float(hl.norm()):.2f}")

        # -------- video frame ------------------------------------------
        if video_writer is not None and global_counter % record_every == 0:
            record_renderer.update_scene(d, camera=args.record_camera)
            if not args.clean_frame and not args.reference_crop_depth_pip:
                # heading arrows: red = predicted delta_yaw (current step),
                # orange = predicted delta_next_yaw (next step). Both rendered as
                # the policy actually consumes them (× 1.5, see prop_t[:, 6:8] above).
                base_pos = d.qpos[:3].copy()
                base_pos[2] += 0.20
                dyaw_now = 1.5 * float(cached_yaw_pred[0, 0])
                dyaw_next = 1.5 * float(cached_yaw_pred[0, 1])
                add_arrow(record_renderer.scene, base_pos,
                          world_yaw + dyaw_now,
                          length=0.45, radius=0.018,
                          rgba=(0.95, 0.20, 0.15, 1.0))
                add_arrow(record_renderer.scene, base_pos,
                          world_yaw + dyaw_next,
                          length=0.35, radius=0.012,
                          rgba=(1.00, 0.65, 0.10, 0.85))
            frame = record_renderer.render()
            frame = frame.copy()
            # overlay PiP of what the robot's depth camera sees, top-right corner
            if not args.clean_frame and last_depth_norm_np is not None:
                if args.reference_crop_depth_pip:
                    # Desired final-crop geometry is 348x232 with 12 px margins.
                    # The reference pipeline first resizes to 2400x1350, then
                    # crops rows 513:1123, so position the raw overlay inside
                    # that region rather than at the top of the discarded 4K frame.
                    pip_w = int(round(348 * args.record_width / 2400.0))
                    pip_h = int(round(232 * args.record_height / 1350.0))
                    margin_x = int(round(12 * args.record_width / 2400.0))
                    y0 = int(round((513 + 12) * args.record_height / 1350.0))
                    margin = max(3, int(round(4 * args.record_width / 2400.0)))
                    x1 = args.record_width - margin_x
                    x0 = x1 - pip_w
                    y1 = y0 + pip_h
                else:
                    pip_h, pip_w = 232, 348                   # 4× upsample of (58, 87)
                    margin = 12
                    y0, y1 = margin, margin + pip_h
                    x0, x1 = args.record_width - margin - pip_w, args.record_width - margin
                # Isaac depth display convention (legged_robot.py:281): show
                # normalized depth shifted to [0,1] as greyscale -> near=black, far=white.
                d_n = (last_depth_norm_np + 0.5).clip(0.0, 1.0)        # 0=near, 1=far
                d_vis = (d_n * 255.0).astype(np.uint8)                 # near=0(black) far=255(white)
                d_t = torch.from_numpy(d_vis.astype(np.float32))[None, None]
                d_t = F.interpolate(d_t, size=(pip_h, pip_w), mode="nearest")
                d_up = d_t.squeeze().numpy().astype(np.uint8)
                rgb = np.stack([d_up, d_up, d_up], axis=-1)
                # black border + paste in the selected top-right region
                border = max(3, margin) if args.reference_crop_depth_pip else 3
                frame[y0-border:y1+border, x0-border:x1+border] = 0
                frame[y0:y1, x0:x1] = rgb
            video_writer.append_data(frame)

    if args.laydown_on_return and return_stop_tick is not None:
        q_start = d.qpos[7:19].copy()
        ramp_ticks = 100
        hold_ticks = max(0, int(round(args.laydown_hold_s * 50.0)))
        for lay_tick in range(ramp_ticks + hold_ticks):
            rate = min((lay_tick + 1) / ramp_ticks, 1.0)
            target = q_start * (1.0 - rate) + LAY_DOF_MJ * rate
            for _ in range(DECIMATION):
                torque = 50.0 * (target - d.qpos[7:19]) - 1.0 * d.qvel[6:18]
                if args.go2_torque_speed_curve:
                    torque = apply_go2hv_torque_speed_curve(torque, d.qvel[6:18])
                d.ctrl[:] = torque
                mujoco.mj_step(m, d)
                if torque_log is not None:
                    torque_log.append(torque.copy())
                    torque_time_log.append(float(d.time))
            if video_writer is not None and lay_tick % record_every == 0:
                record_renderer.update_scene(d, camera=args.record_camera)
                video_writer.append_data(record_renderer.render().copy())
        print(f"[laydown] ramped for 2.00s and held for {args.laydown_hold_s:.2f}s "
              f"at pos=({float(d.qpos[0]):+.3f},{float(d.qpos[1]):+.3f})")

    elapsed = time.time() - t0
    print(f"[done] {completed_ticks} ticks in {elapsed:.1f}s wall  base_z∈[{base_z_min:.3f},{base_z_max:.3f}]  τ_peak={torque_max:.1f}  vx_avg(last 2s)={float(np.mean(fwd_velocity_window)):+.3f}")

    if args.result_tag is not None:
        fx, fy, fz = float(d.qpos[0]), float(d.qpos[1]), float(d.qpos[2])
        upright = fz >= 0.20
        returned = bool(upright and return_stop_tick is not None)
        print(f"RESULT\ttag={args.result_tag}\tseed={args.seed}"
              f"\treturned={int(returned)}\tupright={int(upright)}"
              f"\tfinal_x={fx:.3f}\tfinal_y={fy:.3f}\tfinal_z={fz:.3f}"
              f"\tmax_x={base_x_max:.3f}\tmin_z={base_z_min:.3f}"
              f"\tmax_abs_y={base_absy_max:.3f}"
              f"\ttau_peak={torque_max:.1f}\treturn_armed={int(return_armed)}")

    if viewer is not None:
        viewer.close()
    if video_writer is not None:
        video_writer.close()
        print(f"[video] wrote {args.record}")
    if log_foot_pos is not None:
        np.savez(args.foot_log,
                 foot_pos=log_foot_pos[0:completed_ticks],
                 base_pos=log_foot_base[0:completed_ticks],
                 foot_order=np.array(["FR", "FL", "RR", "RL"]),
                 dt=float(m.opt.timestep * DECIMATION))
        print(f"[foot] wrote {completed_ticks} ticks → {args.foot_log}")
    if log_obs is not None:
        valid = slice(0, completed_ticks)
        np.savez(args.log_npz,
                 obs=log_obs[valid], depth_latent=log_dlat[valid],
                 yaw_pred=log_yaw_pred[valid], action_raw=log_act_raw[valid],
                 cmd_x=log_cmd_x[valid], base_pos=log_base_pos[valid],
                 base_quat=log_base_quat[valid],
                 base_vel_world=log_base_vel_world[valid],
                 world_yaw=log_world_yaw[valid],
                 dt=float(m.opt.timestep * DECIMATION),
                 switch_step=int(args.switch_step),
                 return_armed=bool(return_armed),
                 return_triggered=bool(return_stop_tick is not None))
        print(f"[log] wrote {args.log_npz}  "
              f"obs{log_obs[valid].shape}  act{log_act_raw[valid].shape}")
    if torque_log is not None:
        torque_arr = np.asarray(torque_log, dtype=np.float32)
        time_arr = np.asarray(torque_time_log, dtype=np.float32)
        np.savez(args.torque_log, torque=torque_arr, time=time_arr,
                 joint_order_mj=np.array([
                     "FR_hip", "FR_thigh", "FR_calf",
                     "FL_hip", "FL_thigh", "FL_calf",
                     "RR_hip", "RR_thigh", "RR_calf",
                     "RL_hip", "RL_thigh", "RL_calf"]),
                 max_torque_nm=np.array([23.7, 23.7, 35.5] * 4, dtype=np.float32))
        print(f"[torque] wrote {torque_arr.shape[0]} substep samples → {args.torque_log}")


if __name__ == "__main__":
    main()
