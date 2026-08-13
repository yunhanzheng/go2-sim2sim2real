#!/usr/bin/env python3
"""Single-process (NO ROS) MuJoCo sim2sim render — faithful port of the
lay2stand_record stack's ENGAGED phase, for comparing against the ROS run.

Why this exists
---------------
The normal render path (deploy_mac/sim/render_sim2sim.sh) runs 7 ROS nodes that talk
over DDS: mujoco_simulator <-> low_level_ctrl (PD/FSM) <-> rl_distill_policy
(actor) <-> depth_encoder <-> depth_sim <-> joy_sequencer (+ recorder). This
script collapses all of that numeric pipeline into ONE deterministic process so
we can check whether the multi-process timing of the ROS stack is responsible
for the trajectory (vs the policy itself).

Reproduced EXACTLY (verified against the node source):
  * engage air-snap: qpos[2]=0.42, identity quat, joints=DEFAULT_DOF, qvel=0,
    mj_forward, then 200 substeps of standing PD (kp=50, kd=1)  [mujoco_simulator.py:193-216]
  * 53-D proprio obs (build_prop), scales, env_class indicator (env17=0 -> 1,0)
  * yaw-head injection: prop[6:8] = 1.5 * yaw_pred (live frame), 0 in history    [rl_distill_policy.py:399-407]
  * estimator(prop_t) -> 9-D priv_explicit; actor(obs, hist_encoding=True,
    eval=False, scandots_latent=latent[:32])                                      [rl_distill_policy.py:443-515]
  * raw action -> proprio[37:49] feedback, clipped action -> motors;
    target = DEFAULT_DOF + 0.25*clipped, applied at kp=35/kd=0.6 with decimation 4 [mujoco_simulator.py:247-251]
  * latest cliffside stereo/noise model from front_depth_cam (60x106) at 10 Hz;
    GRU-recurrent depth encoder (RecurrentDepthBackbone, stateful)                [depth_backbone.py:39-45]
  * net per-foot ground-contact force @ 2.0 N (OR-with-previous-raw)
  * planar perpendicular-depth far clip before the standard crop/resize

The deploy latency ENVELOPE is reproduced since 2026-07-28 and is now the default:
`--action_delay 1` (20 ms, what every post-6fb4d46 policy trains with) and
`--depth_latency_ticks 4` (the enforced 80 ms exposure->latent constant). Running
it at zero was off-distribution in the fresh direction and cost real robustness —
6/9 falls vs 2/9 on the wall scene (run_logs/dryrun/lat_ab_2026-07-28).

Still deliberately NOT reproduced — the stack's wall-clock asynchrony (sim
free-running with jittery, non-integer substeps/tick; 10/50 Hz timer drift; GRU
warm-up during lay/stand). This port is otherwise a clean LOCKSTEP loop:
`decimation` sim substeps per 50 Hz policy tick, depth re-encoded every
`depth_every` ticks, GRU reset at engage. To isolate those timing artifacts from
the latency model, run with `--action_delay 0 --depth_latency_ticks 0`.
"""
import os
os.environ.setdefault("MUJOCO_GL", "glfw")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import datetime
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SCRIPTS = ROOT / "src" / "deploy_rl_policy" / "scripts"


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


def add_sphere(scene, pos, radius, rgba):
    import mujoco
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        g, type=int(mujoco.mjtGeom.mjGEOM_SPHERE),
        size=np.array([radius, radius, radius], dtype=np.float64),
        pos=np.asarray(pos, dtype=np.float64), mat=np.eye(3).flatten(),
        rgba=np.array(rgba, dtype=np.float32),
    )
    scene.ngeom += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "models/OD-fine-ns240-h005-w67/model_8000.pt"))
    ap.add_argument("--scene", default=str(ROOT / "resources/go2/scene_flyball35_2h_h10_noffset.xml"))
    ap.add_argument("--ep_root", default=str(ROOT.parent / "go2-flyball-parkour"))
    ap.add_argument("--camera", default="side_cam")
    ap.add_argument("--kp", type=float, default=35.0)
    ap.add_argument("--kd", type=float, default=0.6)
    ap.add_argument("--lin_vel_max", type=float, default=1.75)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--engage", type=float, default=20.0, help="engage seconds at full cmd")
    ap.add_argument("--slowdown", type=float, default=2.0, help="seconds ramping cmd to 0")
    ap.add_argument("--yaw_head", type=int, default=1)
    ap.add_argument("--env17", type=int, default=0)
    ap.add_argument("--contact", type=float, default=2.0)
    ap.add_argument("--decimation", type=int, default=4)
    ap.add_argument("--depth_every", type=int, default=5, help="policy ticks per depth encode (5 => 10Hz)")
    # ---- loop latency: DEPLOY-FAITHFUL DEFAULTS (2026-07-28) -----------------
    # Cliffside students/teachers after EP 6fb4d46 train with a latched 20 ms
    # action delay and depth.latency_range=[4,5] frames (80-100 ms, capture_interval=1);
    # the ROS/real stack enforces the same envelope. An undelayed lockstep loop is
    # off-distribution in the FRESH direction and it is the dominant
    # standalone-vs-ROS divergence: paired over 24 scenes x 3 seeds (D22 m9000,
    # wall scene, run_logs/dryrun/lat_decomp_2026-07-28) a 15 ms action-application
    # delay cut falls 42/72 -> 20/72 and falls-at-the-hurdle 36/72 -> 5/72
    # (p=0.0005), while 80 ms depth latency alone was null (p=0.39) — so the depth
    # default is there for FIDELITY, and the action delay is the one that matters.
    # Pass --action_delay 0 --depth_latency_ticks 0 to recover the old lockstep
    # behaviour (e.g. to isolate ROS wall-clock asynchrony from the latency model).
    ap.add_argument("--action_delay", type=int, default=1, choices=[0, 1],
                    help="1 (default) applies the previous tick's target for one 20 ms "
                         "tick — the delay every post-6fb4d46 policy trains with. "
                         "0 = old undelayed lockstep.")
    ap.add_argument("--depth_latency_ticks", type=int, default=4,
                    help="policy ticks between depth encode and latent availability "
                         "(4 = the enforced 80 ms deploy constant; 0 = old immediate "
                         "delivery). The first latent is always delivered immediately so "
                         "the actor never starts on a zero latent.")
    ap.add_argument("--depth_noise_scale", type=float, default=0.0,
                    help="0=clean, 0.5=reality-tuned, 1=2x artifact rates")
    ap.add_argument("--stereo_shadow", choices=("auto", "on", "off"), default="auto",
                    help="D435 stereo geometry; auto enables with depth noise")
    ap.add_argument("--name", default="OD-w67_m8000_flyball35_noros")
    ap.add_argument("--out", default="")
    ap.add_argument("--no_video", action="store_true")
    # --- oracle goal-following yaw (replaces the encoder yaw-head prediction) ---
    ap.add_argument("--oracle_yaw", type=int, default=0,
                    help="1 => override prop[6:8] with Isaac goal-following "
                         "delta_yaw/delta_next_yaw (computed from base xy + --goals) "
                         "instead of the depth-encoder's predicted yaw. The 32-D "
                         "scandots latent still comes from the encoder.")
    ap.add_argument("--goals", type=str, default=None,
                    help="world-frame goal xy list 'x0,y0;x1,y1;...' for --oracle_yaw")
    ap.add_argument("--next_goal_threshold", type=float, default=0.5)
    ap.add_argument("--reach_goal_delay", type=float, default=0.1)
    ap.add_argument("--wall_goal_idx", type=int, default=-1,
                    help="goal index advanced on base_x>=wall_front_x (env19 flyball=2)")
    ap.add_argument("--wall_front_x", type=float, default=float("nan"))
    ap.add_argument("--wall_gate_margin", type=float, default=0.15)
    args = ap.parse_args()
    if not 0.0 <= args.depth_noise_scale <= 1.0:
        ap.error("--depth_noise_scale must be in [0, 1]")

    # EP_FEED_PRED_YAW must be set BEFORE constructing RecurrentDepthBackbone
    # (its output dim 32 vs 34 is decided at __init__ from this env var).
    os.environ["EP_FEED_PRED_YAW"] = "1" if args.yaw_head else "0"

    import sys
    sys.path.insert(0, str(SCRIPTS))  # for distill_common (rclpy-free)
    import torch
    torch.set_num_threads(1)
    import mujoco
    import distill_common as dc
    from distill_common import (
        N_PROPRIO, NUM_ACTIONS, HISTORY_LEN, DEPTH_LATENT_DIM, N_PRIV_EXPLICIT,
        N_PRIV_LATENT, ACTION_SCALE, CLIP_ACTIONS, YAW_PRED_MULTIPLIER,
        DEFAULT_DOF_POLICY, DEPTH_RAW_H, DEPTH_RAW_W, DEPTH_TARGET_H, DEPTH_TARGET_W,
    )

    dc.add_extreme_parkour_to_path(Path(args.ep_root))
    from rsl_rl.modules.actor_critic import Actor, get_activation
    from rsl_rl.modules.estimator import Estimator
    from rsl_rl.modules.depth_backbone import DepthOnlyFCBackbone58x87, RecurrentDepthBackbone

    # ---- models (exact constructor args from the ROS nodes) -----------------
    ckpt = dc.load_checkpoint(Path(args.ckpt))
    activation = get_activation("elu")
    actor = Actor(
        num_prop=N_PROPRIO, num_scan=132, num_actions=NUM_ACTIONS,
        scan_encoder_dims=[128, 64, 32], actor_hidden_dims=[512, 256, 128],
        priv_encoder_dims=[64, 20], num_priv_latent=N_PRIV_LATENT,
        num_priv_explicit=N_PRIV_EXPLICIT, num_hist=HISTORY_LEN,
        activation=activation, tanh_encoder_output=False,
    )
    actor.load_state_dict(ckpt["depth_actor_state_dict"], strict=True)
    actor.eval()

    estimator = Estimator(input_dim=N_PROPRIO, output_dim=N_PRIV_EXPLICIT, hidden_dims=[128, 64])
    estimator.load_state_dict(ckpt["estimator_state_dict"], strict=True)
    estimator.eval()

    backbone = DepthOnlyFCBackbone58x87(
        prop_dim=N_PROPRIO, scandots_output_dim=DEPTH_LATENT_DIM,
        hidden_state_dim=512, output_activation=None, num_frames=1,
    )
    encoder = RecurrentDepthBackbone(backbone, env_cfg=None)
    encoder.load_state_dict(ckpt["depth_encoder_state_dict"], strict=True)
    encoder.eval()
    enc_out_dim = encoder.output_mlp[0].out_features
    print(f"loaded: actor/estimator/encoder OK | enc_out_dim={enc_out_dim} "
          f"(yaw_head={'on' if enc_out_dim == DEPTH_LATENT_DIM + 2 else 'off'})")

    # ---- mujoco -------------------------------------------------------------
    m = mujoco.MjModel.from_xml_path(args.scene)
    d = mujoco.MjData(m)
    m.opt.timestep = 0.005
    default_dof_mj = DEFAULT_DOF_POLICY.copy()
    foot_geom_ids = [
        mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in ("FR", "FL", "RR", "RL")
    ]
    if any(gid < 0 for gid in foot_geom_ids):
        raise RuntimeError(f"missing foot collision geom(s): {foot_geom_ids}")

    depth_renderer = mujoco.Renderer(m, height=DEPTH_RAW_H, width=DEPTH_RAW_W)
    depth_renderer.enable_depth_rendering()
    depth_scene_option = dc.mujoco_depth_scene_option()
    stereo_enabled = (args.depth_noise_scale > 0.0
                      if args.stereo_shadow == "auto"
                      else args.stereo_shadow == "on")
    stereo_depth = (dc.MujocoStereoDepth(
        m, "front_depth_cam",
        baseline=dc.TRAIN_DEPTH_CFG["stereo_baseline"],
        eps=dc.TRAIN_DEPTH_CFG["stereo_eps"],
        radial_clip=dc.depth_radial_clip_enabled(),
    ) if stereo_enabled else None)
    depth_degrader = (dc.TrainDepthDegrader(noise_scale=args.depth_noise_scale)
                      if args.depth_noise_scale > 0.0 or stereo_enabled else None)
    print(f"depth model: source={dc.TRAIN_DEPTH_SOURCE_COMMIT} "
          f"stereo={'on' if stereo_enabled else 'off'} "
          f"noise_scale={args.depth_noise_scale:.2f}")
    vid_renderer = None if args.no_video else mujoco.Renderer(m, height=720, width=1280)

    # ---- engage air-snap + 1 s settle (mujoco_simulator.py:193-216) ---------
    d.qacc_warmstart[:] = 0.0
    d.qpos[0:2] = [0.0, 0.0]
    d.qpos[2] = 0.42
    d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    d.qpos[7:19] = default_dof_mj
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)
    kp_air = np.full(12, 50.0)
    kd_air = np.full(12, 1.0)
    for _ in range(int(1.0 / m.opt.timestep)):
        tau = kp_air * (default_dof_mj - d.qpos[7:19]) - kd_air * d.qvel[6:18]
        d.ctrl[:] = tau
        mujoco.mj_step(m, d)
    print(f"air-snap settle done: base_z={float(d.qpos[2]):.3f} |qvel|={float(np.linalg.norm(d.qvel)):.4f}")

    # ---- runtime state ------------------------------------------------------
    env_neq17 = 0.0 if args.env17 else 1.0
    env_eq17 = 1.0 if args.env17 else 0.0
    history_ring = None                       # (1,10,53)
    last_action_raw = torch.zeros(1, NUM_ACTIONS)   # -> actor proprio[37:49]
    last_action_clipped = np.zeros(NUM_ACTIONS)     # -> motors + encoder proprio
    cached_latent = torch.zeros(1, DEPTH_LATENT_DIM)
    cached_yaw = torch.zeros(1, 2)
    have_yaw = False
    pending_latents = []                      # [(deliver_tick, latent, yaw, have_yaw)]
    held_target_dof = default_dof_mj.copy()   # --action_delay 1: previous tick's target
    pol_last_contact_raw = np.zeros(4)
    enc_last_contact_raw = np.zeros(4)
    encoder.hidden_states = None              # reset GRU at engage

    # --- oracle goal-following yaw state -------------------------------------
    goals_xy = parse_goals(args.goals)
    use_oracle_yaw = bool(args.oracle_yaw) and goals_xy is not None
    cur_goal_idx = 0
    reach_goal_timer = 0
    oracle_dy = 0.0
    oracle_dny = 0.0
    oracle_target_yaw = 0.0
    reach_goal_delay_ticks = max(1, int(round(args.reach_goal_delay / (args.decimation * m.opt.timestep))))
    if args.oracle_yaw and goals_xy is None:
        raise SystemExit("--oracle_yaw 1 requires --goals")
    if use_oracle_yaw:
        print(f"[oracle_yaw] ON — {len(goals_xy)} goals, thr={args.next_goal_threshold}m "
              f"delay={reach_goal_delay_ticks}tk wall_idx={args.wall_goal_idx} "
              f"wall_front_x={args.wall_front_x}; encoder yaw-head IGNORED")
        for gi, g in enumerate(goals_xy):
            print(f"            g{gi} = ({g[0]:+.2f}, {g[1]:+.2f})")

    def foot_contact_forces():
        """Net world-frame ground-contact magnitude in FR,FL,RR,RL order."""
        net = np.zeros((4, 3), dtype=np.float64)
        wrench = np.zeros(6, dtype=np.float64)
        for contact_idx in range(d.ncon):
            contact = d.contact[contact_idx]
            for foot_idx, geom_id in enumerate(foot_geom_ids):
                if contact.geom1 == geom_id or contact.geom2 == geom_id:
                    mujoco.mj_contactForce(m, d, contact_idx, wrench)
                    net[foot_idx] += contact.frame.reshape(3, 3).T @ wrench[:3]
                    break
        return np.linalg.norm(net, axis=1)

    def read_state():
        qj = d.qpos[7:19].astype(np.float64).copy()
        dqj = d.qvel[6:18].astype(np.float64).copy()
        quat = d.qpos[3:7].astype(np.float64).copy()
        roll, pitch, _ = dc.mj_quat_to_roll_pitch_yaw(quat)
        gyro = d.sensordata[40:43].astype(np.float64).copy()
        ff_mag = foot_contact_forces()
        foot = np.rint(ff_mag).astype(np.float64)
        return qj, dqj, roll, pitch, gyro, foot

    def render_depth():
        depth_renderer.update_scene(
            d, camera="front_depth_cam", scene_option=depth_scene_option)
        raw = depth_renderer.render().astype(np.float32)       # (60,106) positive metres
        if stereo_depth is not None:
            raw = stereo_depth(d, raw)
        elif dc.depth_radial_clip_enabled():
            radial_max = dc.radial_maxperp_mask(raw.shape[0], raw.shape[1])
            raw = np.where(raw > radial_max, np.float32(dc.DEPTH_FAR), raw).astype(np.float32)
        raw = dc.clip_mujoco_too_close_to_far(raw)
        if depth_degrader is not None:
            return depth_degrader(raw)
        return dc.normalize_depth(raw, 0.0)                    # (1,58,87) torch

    fps = 50
    n_engage = int(round(args.engage * fps))
    n_slow = int(round(args.slowdown * fps))
    n_ticks = n_engage + n_slow

    def cmd_vx_at(tick):
        if tick < n_engage:
            return args.lin_vel_max * args.speed
        frac = min(1.0, (tick - n_engage) / max(1, n_slow))
        return args.lin_vel_max * args.speed * (1.0 - frac)

    # ---- output paths -------------------------------------------------------
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    slug = args.name.replace(" ", "-").replace("/", "-")
    # Default output is grouped by policy: run_logs/sim2sim/<ckptdir>__<model>/.
    policy = f"{Path(args.ckpt).parent.name}__{Path(args.ckpt).stem}"
    outdir = (Path(args.out) if args.out
              else ROOT / "run_logs" / "sim2sim" / policy / f"{ts}-{slug}")
    outdir.mkdir(parents=True, exist_ok=True)
    mp4_path = outdir / f"{slug}.mp4"
    traj_path = outdir / f"{slug}.traj.tsv"
    json_path = outdir / f"{slug}.base_height.json"

    writer = None
    if not args.no_video:
        import imageio
        writer = imageio.get_writer(
            str(mp4_path), fps=fps, codec="libx264", format="FFMPEG",
            macro_block_size=None,
            ffmpeg_params=["-crf", "23", "-preset", "ultrafast", "-pix_fmt", "yuv420p"],
        )
        try:
            import cv2
        except Exception:
            cv2 = None

    base_z_hist = []
    traj_rows = []

    # ---- main lockstep loop -------------------------------------------------
    for tick in range(n_ticks):
        cmd_vx = cmd_vx_at(tick)

        # depth encode @ 10 Hz (incl. tick 0, so the actor never sees a zero latent)
        if tick % args.depth_every == 0:
            qj, dqj, roll, pitch, gyro, foot = read_state()
            efilt, eraw = dc.compute_contact_filt(foot, enc_last_contact_raw, threshold_n=args.contact)
            enc_last_contact_raw = eraw
            eprop = dc.build_prop(
                qj_policy=qj, dqj_policy=dqj, ang_vel_body=gyro, roll=roll, pitch=pitch,
                last_action_pol=last_action_clipped, contact_filt=efilt, lin_vel_x=cmd_vx,
                env_class_neq_17=env_neq17, env_class_eq_17=env_eq17,
            )
            eprop_t = torch.from_numpy(eprop).float().unsqueeze(0)
            eprop_t[:, 6:8] = 0.0                              # yaw mask before encoder
            depth_t = render_depth().float()                  # (1,58,87)
            with torch.no_grad():
                out = encoder(depth_t, eprop_t).squeeze(0)     # (34,)
            new_latent = out[:DEPTH_LATENT_DIM].unsqueeze(0)
            new_yaw, new_have_yaw = cached_yaw, have_yaw
            if enc_out_dim == DEPTH_LATENT_DIM + 2:
                new_yaw = out[DEPTH_LATENT_DIM:DEPTH_LATENT_DIM + 2].unsqueeze(0)
                new_have_yaw = True
            # Deploy publishes the latent a fixed 80 ms after capture. Deliver the
            # FIRST one immediately (the ROS stack blocks until it arrives, so the
            # actor never runs on a zero latent) and delay every later one.
            if args.depth_latency_ticks > 0 and tick > 0:
                pending_latents.append((tick + args.depth_latency_ticks,
                                        new_latent, new_yaw, new_have_yaw))
            else:
                cached_latent, cached_yaw, have_yaw = new_latent, new_yaw, new_have_yaw
        while pending_latents and pending_latents[0][0] <= tick:
            _, cached_latent, cached_yaw, have_yaw = pending_latents.pop(0)

        # --- policy tick ---
        qj, dqj, roll, pitch, gyro, foot = read_state()
        pfilt, praw = dc.compute_contact_filt(foot, pol_last_contact_raw, threshold_n=args.contact)
        pol_last_contact_raw = praw
        prop = dc.build_prop(
            qj_policy=qj, dqj_policy=dqj, ang_vel_body=gyro, roll=roll, pitch=pitch,
            last_action_pol=last_action_raw.numpy().reshape(-1), contact_filt=pfilt,
            lin_vel_x=cmd_vx, env_class_neq_17=env_neq17, env_class_eq_17=env_eq17,
        )
        prop_t = torch.from_numpy(prop).float().unsqueeze(0)

        hist_frame = prop_t.clone()
        hist_frame[:, 6:8] = 0.0
        if history_ring is None:
            history_ring = torch.zeros(1, HISTORY_LEN, N_PROPRIO)
        else:
            history_ring = torch.cat([history_ring[:, 1:], hist_frame.unsqueeze(1)], dim=1)

        if use_oracle_yaw:
            # ORACLE heading: Isaac goal-following delta_yaw/delta_next_yaw from
            # base xy + the goal sequence (unwrapped, refreshed every 5 ticks),
            # injected in place of the encoder's predicted yaw. The 32-D scandots
            # latent is still the encoder's (cached_latent) — only the yaw slots
            # are swapped for the oracle. History stays yaw-masked (set above).
            base_xy = d.qpos[:2].copy()
            _, _, world_yaw = dc.mj_quat_to_roll_pitch_yaw(d.qpos[3:7])
            n_goals = len(goals_xy)
            cur = goals_xy[min(cur_goal_idx, n_goals - 1)]
            is_wall = (cur_goal_idx == args.wall_goal_idx) and not np.isnan(args.wall_front_x)
            if is_wall:
                reached = base_xy[0] >= (args.wall_front_x - args.wall_gate_margin)
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
            oracle_target_yaw = float(np.arctan2(cur[1] - base_xy[1], cur[0] - base_xy[0]))
            next_target_yaw = float(np.arctan2(nxt[1] - base_xy[1], nxt[0] - base_xy[0]))
            if tick % 5 == 0:                       # Isaac refreshes delta_yaw every 5 steps
                oracle_dy = oracle_target_yaw - world_yaw     # UNWRAPPED
                oracle_dny = next_target_yaw - world_yaw
            prop_t[:, 6] = oracle_dy
            prop_t[:, 7] = oracle_dny
        elif have_yaw and args.yaw_head:
            prop_t[:, 6:8] = YAW_PRED_MULTIPLIER * cached_yaw
        else:
            prop_t[:, 6:8] = 0.0
        yaw_cmd = (float(prop_t[0, 6]), float(prop_t[0, 7]))

        with torch.no_grad():
            priv_expl = estimator(prop_t)
            obs = dc.assemble_actor_obs(prop_t, priv_expl, history_ring)
            action_raw = actor(obs, hist_encoding=True, eval=False, scandots_latent=cached_latent)
        est_vel = (priv_expl[0, 0:3].numpy() / 2.0)
        last_action_raw = action_raw.detach()
        action_clipped_t = action_raw.clamp(-CLIP_ACTIONS, CLIP_ACTIONS)
        last_action_clipped = action_clipped_t.squeeze(0).numpy()
        target_dof = DEFAULT_DOF_POLICY + ACTION_SCALE * last_action_clipped
        # --action_delay 1: the motors get the PREVIOUS tick's target for this tick
        # (Isaac's action_delay, latched at fleet level 6 for every cliffside run).
        applied_target_dof = held_target_dof if args.action_delay else target_dof
        held_target_dof = target_dof.copy()

        # PD with decimation (mujoco_simulator._step_decimation)
        for _ in range(args.decimation):
            tau = args.kp * (applied_target_dof - d.qpos[7:19]) - args.kd * d.qvel[6:18]
            d.ctrl[:] = tau
            mujoco.mj_step(m, d)

        bx, by, bz = float(d.qpos[0]), float(d.qpos[1]), float(d.qpos[2])
        base_z_hist.append(bz)
        dlat = float(cached_latent.norm())
        traj_rows.append((tick, tick / fps, bx, by, bz, cmd_vx,
                          float(est_vel[0]), float(est_vel[1]), float(est_vel[2]),
                          dlat, np.degrees(yaw_cmd[0]), np.degrees(yaw_cmd[1])))

        if tick % 50 == 0:
            gstr = (f"  goal[{cur_goal_idx}]" if use_oracle_yaw else "")
            yaw_src = "oracle" if use_oracle_yaw else ("pred" if (have_yaw and args.yaw_head) else "zero")
            print(f"t={tick/fps:5.2f}s  base_xyz=({bx:+.2f},{by:+.2f},{bz:.3f})  "
                  f"cmd_vx={cmd_vx:+.2f}  [est] vx={est_vel[0]:+.2f} vy={est_vel[1]:+.2f} "
                  f"vz={est_vel[2]:+.2f}  yaw_cmd({yaw_src})=({np.degrees(yaw_cmd[0]):+6.1f}°,"
                  f"{np.degrees(yaw_cmd[1]):+6.1f}°)  |dlat|={dlat:.2f}{gstr}")

        # video frame from side_cam
        if writer is not None:
            vid_renderer.update_scene(d, camera=args.camera)
            if use_oracle_yaw:
                for gi, g in enumerate(goals_xy):
                    if gi == cur_goal_idx:
                        rgba, rad = (0.10, 1.0, 0.10, 0.95), 0.035   # current → green
                    elif gi < cur_goal_idx:
                        rgba, rad = (0.35, 0.35, 0.35, 0.55), 0.018  # passed → grey
                    else:
                        rgba, rad = (1.0, 0.80, 0.10, 0.75), 0.022   # upcoming → amber
                    add_sphere(vid_renderer.scene, [g[0], g[1], 0.05], rad, rgba)
            frame = vid_renderer.render()
            if cv2 is not None:
                txt = [f"NO-ROS  t={tick/fps:4.1f}s  cmd_vx={cmd_vx:+.2f}",
                       f"base x={bx:+.2f} y={by:+.2f} z={bz:.3f}",
                       f"est_vx={est_vel[0]:+.2f}  |dlat|={dlat:.2f}"]
                if use_oracle_yaw:
                    txt.append(f"ORACLE yaw g{cur_goal_idx} dyaw={np.degrees(yaw_cmd[0]):+.0f}")
                frame = np.ascontiguousarray(frame)
                for i, line in enumerate(txt):
                    cv2.putText(frame, line, (16, 30 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (255, 255, 255), 2, cv2.LINE_AA)
            writer.append_data(frame)

    if writer is not None:
        writer.close()

    # ---- trajectory + base-height stats ------------------------------------
    with open(traj_path, "w") as f:
        f.write("tick\tt_s\tbase_x\tbase_y\tbase_z\tcmd_vx\test_vx\test_vy\test_vz\tdlat\tyaw0_deg\tyaw1_deg\n")
        for r in traj_rows:
            f.write("\t".join(f"{v:.5f}" if isinstance(v, float) else str(v) for v in r) + "\n")

    bz = np.array(base_z_hist)
    stats = {
        "ckpt": args.ckpt, "scene": args.scene, "camera": args.camera,
        "mode": "single_process_no_ros (lockstep)",
        "params": {"kp": args.kp, "kd": args.kd, "lin_vel_max": args.lin_vel_max,
                   "speed": args.speed, "decimation": args.decimation,
                   "depth_every": args.depth_every, "yaw_head": args.yaw_head, "env17": args.env17,
                   "contact": args.contact, "depth_noise_scale": args.depth_noise_scale,
                   "stereo_shadow": stereo_enabled,
                   "depth_source_commit": dc.TRAIN_DEPTH_SOURCE_COMMIT},
        "policy_engaged": {
            "frames": int(bz.size), "duration_s": bz.size / fps,
            "mean_m": float(bz.mean()), "std_m": float(bz.std()),
            "min_m": float(bz.min()), "max_m": float(bz.max()),
        },
        "final_base_xyz": [traj_rows[-1][2], traj_rows[-1][3], traj_rows[-1][4]],
        "max_base_x": float(max(r[2] for r in traj_rows)),
    }
    with open(json_path, "w") as f:
        json.dump(stats, f, indent=2)

    print("\n=== DONE (no-ROS single-process) ===")
    print(f"video : {mp4_path if writer is not None else '(skipped)'}")
    print(f"traj  : {traj_path}")
    print(f"json  : {json_path}")
    print(f"engaged base-z: mean={stats['policy_engaged']['mean_m']:.4f}  "
          f"min={stats['policy_engaged']['min_m']:.4f}  max={stats['policy_engaged']['max_m']:.4f}  "
          f"(n={bz.size})")
    print(f"final base_xyz={stats['final_base_xyz']}  max_base_x={stats['max_base_x']:.2f}")


if __name__ == "__main__":
    main()
