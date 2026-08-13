#!/usr/bin/env python3
"""Fused 50 Hz depth-encoder + actor node — one process, ONE proprio snapshot.

Replaces the depth_encoder_node.py + rl_distill_policy.py pair with a single
node. Numerically it runs the same pipeline; what changes is *structure*, and
the structure is the point.

WHY (the train-time contract)
-----------------------------
`RecurrentDepthBackbone.forward(depth_image, proprioception)` consumes proprio,
and distillation training feeds it the actor's own frame, same tick, same
tensor (rsl_rl/runners/on_policy_runner.py:312-319):

    obs_prop_depth = student_obs[:, :n_proprio].clone()
    obs_prop_depth[:, 6:8] = 0
    depth_out = self.alg.depth_encoder(infos["depth"].clone(), obs_prop_depth)
    ...
    obs_student[:, 6:8] = yaw.detach()          # encoder yaw -> actor obs

So the trained contract is: **one proprio frame per tick drives both networks**,
with the heading slots masked for the encoder and filled from the encoder's yaw
head for the actor. The two-node stack cannot honour that — the encoder builds
its own frame from its own /lowstate callback at depth-arrival time, the actor
builds another at its 50 Hz tick, and the two are up to a camera period apart.

Upstream EP split encoder/actor across a UDP socket (paper §4.1: depth backbone
10 Hz + base policy 50 Hz on a Jetson NX). The only public EP-on-Go2
deployment (change-every/Extreme-Parkour-Onboard) instead keeps the *image* on
the wire and runs both nets in one process off one `get_proprio()` — which is
what this node does.

WHAT THIS FIXES BY CONSTRUCTION
-------------------------------
The ROS-vs-standalone findings F01-F03 are all instances of
"the encoder's recurrent input disagrees with the actor's", and F12 showed that
correcting them one or two at a time does not recover — only all three together
do. Here they cannot occur:

  * F01 (GRU ran through lay/stand): the encoder only ever steps inside an
    engaged tick, and one reset path clears GRU + history + action buffer +
    contact together on the engage edge.
  * F02 (encoder filtered contact at 10 Hz, spanning 100 ms instead of 20 ms):
    there is one contact filter, advanced once per 50 Hz tick.
  * F03 (encoder saw clipped action, actor stored raw): there is one
    `last_action`, held in-process, raw — as training stores it.

It also deletes a whole class of flag-desync bug the two-node stack had to
defend against with matched CLI pairs (--contact_threshold, --env_class_eq_17,
--binary_vel, --vel_deadzone, --lin_vel_max, --force_lin_vel_x all had to be
passed identically to both nodes or prop[10]/prop[11:13]/prop[49:53] silently
diverged between the latent and the actor input). Each is now a single flag
consumed once.

DEPTH LATENCY: AGE GATE, NOT A SLEEP
------------------------------------
depth_encoder_node enforced the paper's constant 80 ms capture->latent budget by
*sleeping* the remainder before publishing. A fused node cannot sleep mid-tick
without missing the actor's 20 ms deadline, so the same invariant is enforced by
selection instead: queue incoming frames and encode the NEWEST frame that is
already at least `--depth_latency_s` old (camera-clock corrected). The encoder
therefore never sees a frame fresher than the budget, which is exactly what the
sleep guaranteed — and the *distribution* matches too, because the old actor
held each published latent until the next one arrived, giving an effective
latent age of budget..budget+camera_period. Nothing blocks.

Deliberately NOT changed: obs layout, scales, action clip/delay, target
assembly, PD path, the stale-input fail-safes, and the FSM contract with
low_level_ctrl (/policy_engaged in, /rl/target_pos out). Diagnostic topics the
viewer/recorder consume (/rl/depth_latent, /rl/last_action, /rl/contact_filt,
/rl/est_lin_vel, /rl/proprio_latency, /rl/depth_latency) are still published,
now as pure outputs — nothing in the loop reads them back.
"""
from collections import deque
from pathlib import Path
import argparse
import os
import sys
import time

import numpy as np
import rclpy
import torch
# Tiny MLPs: 24-thread BLAS is ~100x slower than single-thread because of spawn
# overhead (see rl_distill_policy.py). Pin before any forward runs.
torch.set_num_threads(1)
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32MultiArray
from geometry_msgs.msg import Twist
from unitree_go.msg import LowState

from distill_common import (
    add_extreme_parkour_to_path, load_checkpoint,
    mj_quat_to_roll_pitch_yaw, build_prop, assemble_actor_obs,
    compute_contact_filt, parse_contact_threshold,
    CONTACT_FORCE_THRESHOLD_N,
    MJ_TO_POLICY, DEFAULT_DOF_POLICY,
    NUM_ACTIONS, N_PROPRIO, N_PRIV_EXPLICIT, N_PRIV_LATENT,
    HISTORY_LEN, DEPTH_LATENT_DIM, YAW_PRED_MULTIPLIER,
    ACTION_BUF_LEN, ACTION_DELAY_VIEW, ACTION_SCALE, CLIP_ACTIONS,
    DEPTH_TARGET_H, DEPTH_TARGET_W,
    DEPTH_LATENCY_S, PROPRIO_LATENCY_S, UPDATE_INTERVAL_DEFAULT,
    host_monotonic_ns,
)
from xbox_command import XboxController


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

class DepthFrame:
    """One queued preprocessed depth frame, with the clocks needed to age it."""
    __slots__ = ("capture_s", "receive_mono_ns", "arr", "seq")

    def __init__(self, capture_s, receive_mono_ns, arr, seq):
        self.capture_s = capture_s
        self.receive_mono_ns = receive_mono_ns
        self.arr = arr
        self.seq = seq


class DistillFusedPolicy(Node):
    def __init__(self, ckpt_path: Path, ep_root: Path,
                 action_delay: int,
                 lin_vel_max: float,
                 force_lin_vel_x: float = float("nan"),
                 binary_vel: bool = False,
                 vel_deadzone: float = 0.2,
                 contact_threshold: float = CONTACT_FORCE_THRESHOLD_N,
                 env_class_eq_17: bool = False,
                 depth_latency_s: float = DEPTH_LATENCY_S,
                 proprio_latency_s: float = PROPRIO_LATENCY_S,
                 depth_update_interval: int = UPDATE_INTERVAL_DEFAULT,
                 reset_on_engage: bool = True,
                 zero_yaw_pred: bool = False,
                 use_true_yaw: bool = False,
                 use_sim_truth_vel: bool = False,
                 zero_student_velocity_obs: bool = False,
                 clip_applied_action: bool = True,
                 publish_latent: bool = True,
                 diagnostics: bool = True,
                 rate_hz: float = 50.0,
                 lowstate_max_age_s: float = 0.2,
                 depth_max_age_s: float = 0.6,
                 trace_log_path: Path = None,
                 action_log_path: Path = None,
                 latency_log_path: Path = None,
                 proprio_log_path: Path = None,
                 proprio_log_ticks: int = 100,
                 vel_log_path: Path = None,
                 vel_log_ticks: int = 2000):
        super().__init__("distill_fused_policy")
        self.action_delay = action_delay
        self.lin_vel_max = lin_vel_max
        self.force_lin_vel_x = force_lin_vel_x
        self.binary_vel = bool(binary_vel)
        self.vel_deadzone = vel_deadzone
        # Single owner of the contact threshold — the encoder/actor pair that
        # had to be kept in lockstep no longer exists.
        self.contact_threshold = parse_contact_threshold(contact_threshold)
        self.env_class_eq_17 = bool(env_class_eq_17)
        self.depth_latency_s = float(depth_latency_s)
        self.proprio_latency_s = float(proprio_latency_s)
        self.depth_update_interval = max(1, int(depth_update_interval))
        self.reset_on_engage = bool(reset_on_engage)
        self.zero_yaw_pred = bool(zero_yaw_pred)
        self.use_true_yaw = bool(use_true_yaw)
        self.use_sim_truth_vel = bool(use_sim_truth_vel)
        self.zero_student_velocity_obs = bool(zero_student_velocity_obs)
        self.clip_applied_action = bool(clip_applied_action)
        self.publish_latent = bool(publish_latent)
        # --diagnostics 0 drops every per-tick diagnostic publish (last_action,
        # contact_filt, est_lin_vel, latent, latency diags) from the hot 20 ms
        # tick. Only the viewer/recorder consume them; the control loop never
        # reads them back. Default on; run_real_remote.sh turns them off unless
        # a viewer/bag needs them.
        self.diagnostics = bool(diagnostics)
        self.lowstate_max_age_s = lowstate_max_age_s
        self.depth_max_age_s = depth_max_age_s
        self._stale_stopped = False

        # ----- module setup
        add_extreme_parkour_to_path(ep_root)
        from rsl_rl.modules.actor_critic import Actor, get_activation
        from rsl_rl.modules.estimator import Estimator
        from rsl_rl.modules.depth_backbone import (
            DepthOnlyFCBackbone58x87, RecurrentDepthBackbone,
        )

        self.get_logger().info(f"loading ckpt {ckpt_path}")
        ckpt = load_checkpoint(ckpt_path)

        activation = get_activation("elu")
        self.actor = Actor(
            num_prop=N_PROPRIO, num_scan=132, num_actions=NUM_ACTIONS,
            scan_encoder_dims=[128, 64, 32],
            actor_hidden_dims=[512, 256, 128],
            priv_encoder_dims=[64, 20],
            num_priv_latent=N_PRIV_LATENT, num_priv_explicit=N_PRIV_EXPLICIT,
            num_hist=HISTORY_LEN, activation=activation, tanh_encoder_output=False,
        )
        self.actor.load_state_dict(ckpt["depth_actor_state_dict"], strict=True)
        self.actor.eval()

        self.estimator = Estimator(
            input_dim=N_PROPRIO, output_dim=N_PRIV_EXPLICIT, hidden_dims=[128, 64],
        )
        self.estimator.load_state_dict(ckpt["estimator_state_dict"], strict=True)
        self.estimator.eval()

        backbone = DepthOnlyFCBackbone58x87(
            prop_dim=N_PROPRIO, scandots_output_dim=DEPTH_LATENT_DIM,
            hidden_state_dim=512, output_activation=None, num_frames=1,
        )
        self.encoder = RecurrentDepthBackbone(backbone, env_cfg=None)
        self.encoder.load_state_dict(ckpt["depth_encoder_state_dict"], strict=True)
        self.encoder.eval()
        # 32 (no yaw head) or 34 (32 latent + 2 regressed heading dims). A 34-D
        # checkpoint only loads strict=True when the gate is on. EP main renamed
        # that gate EP_FEED_PRED_YAW -> EP_USE_YAW (2026-07) and
        # RecurrentDepthBackbone now reads EP_USE_YAW only; the run scripts
        # export both. Log the effective one so a stale legacy-only export is
        # visible instead of silently producing a 32-D head.
        self._enc_out_dim = self.encoder.output_mlp[0].out_features
        self._has_yaw_head = self._enc_out_dim == DEPTH_LATENT_DIM + 2
        self.get_logger().info(
            f"depth encoder out_dim={self._enc_out_dim} "
            f"(yaw_head={'on' if self._has_yaw_head else 'off'}); "
            f"EP_USE_YAW={os.environ.get('EP_USE_YAW', '0')} "
            f"(legacy EP_FEED_PRED_YAW={os.environ.get('EP_FEED_PRED_YAW', '0')})")

        # ----- logs
        self.trace_log_fp = None
        self.trace_tick = 0
        if trace_log_path is not None:
            self.trace_log_fp = open(trace_log_path, "w", buffering=1)
            self.trace_log_fp.write(
                "tick\tnow_s\tcapture_s\tengaged\t" +
                "\t".join(f"prop_{i}" for i in range(N_PROPRIO)) + "\t" +
                "\t".join(f"latent_{i}" for i in range(DEPTH_LATENT_DIM)) + "\n")
        self.action_log_fp = None
        self.action_log_t0 = None
        if action_log_path is not None:
            self.action_log_fp = open(action_log_path, "w", buffering=1)
            self.action_log_fp.write(
                "tick\tt_s\tworld_x\tworld_y\tcmd_x\t" +
                "\t".join(f"contact_{i}" for i in range(4)) + "\t" +
                "\t".join(f"raw_{i}" for i in range(NUM_ACTIONS)) + "\t" +
                "\t".join(f"clipped_{i}" for i in range(NUM_ACTIONS)) + "\t" +
                "\t".join(f"target_{i}" for i in range(NUM_ACTIONS)) + "\n")
        self.latency_log_fp = None
        if latency_log_path is not None:
            self.latency_log_fp = open(latency_log_path, "w", buffering=1)
            self.latency_log_fp.write(
                "target_seq\tlowstate_tick\tlowstate_receive_mono_ns\tdepth_seq\t"
                "depth_capture_s\tdepth_age_ms\tencoder_ran\ttick_start_mono_ns\t"
                "encode_start_mono_ns\tencode_end_mono_ns\tinference_start_mono_ns\t"
                "inference_end_mono_ns\tsleep_start_mono_ns\tsleep_end_mono_ns\t"
                "publish_start_mono_ns\tpublish_end_mono_ns\n")
        self.proprio_log_fp = None
        self.proprio_log_ticks_remaining = proprio_log_ticks if proprio_log_path else 0
        self._proprio_log_written = 0
        self.proprio_log_t0 = None
        if proprio_log_path is not None:
            field_names = (
                ["ang_vel_x*0.25", "ang_vel_y*0.25", "ang_vel_z*0.25",
                 "roll", "pitch",
                 "zero_dyaw", "delta_yaw", "delta_next_yaw",
                 "zero_cmdx", "zero_cmdy", "lin_vel_x",
                 "env_class!=17", "env_class==17"]
                + [f"dpos_{leg}{j}" for leg in ("FR", "FL", "RR", "RL") for j in "htc"]
                + [f"dvel_{leg}{j}" for leg in ("FR", "FL", "RR", "RL") for j in "htc"]
                + [f"act_{leg}{j}" for leg in ("FR", "FL", "RR", "RL") for j in "htc"]
                + [f"c_{leg}-0.5" for leg in ("FR", "FL", "RR", "RL")]
            )
            assert len(field_names) == N_PROPRIO, f"got {len(field_names)} field names"
            self.proprio_log_fp = open(proprio_log_path, "w", buffering=1)
            self.proprio_log_fp.write(
                "tick\tt_s\t" + "\t".join(field_names) + "\t"
                + "\t".join(f"rawF_{leg}" for leg in ("FR", "FL", "RR", "RL")) + "\n")
        self.vel_log_fp = None
        self.vel_log_ticks_remaining = vel_log_ticks if vel_log_path else 0
        self._vel_log_written = 0
        self._vel_log_t0 = None
        if vel_log_path is not None:
            self.vel_log_fp = open(vel_log_path, "w", buffering=1)
            self.vel_log_fp.write(
                "tick\tt_s\tcmd_vx"
                "\tgt_vx\tgt_vy\tgt_vz"
                "\tekf_vx\tekf_vy\tekf_vz"
                "\test_vx\test_vy\test_vz"
                "\tworld_x\tworld_y\tworld_yaw_deg"
                "\tyaw_cmd0_deg\tyaw_cmd1_deg\n")

        # ----- runtime state
        self.latest_lowstate: LowState = LowState()
        self.latest_lowstate_receive_mono_ns = 0
        self.have_lowstate = False
        self.history_ring = None                       # (1, HISTORY_LEN, 53)
        self.last_action = torch.zeros(1, NUM_ACTIONS)  # RAW, as training stores
        self.action_buf = deque(
            [torch.zeros(1, NUM_ACTIONS) for _ in range(ACTION_BUF_LEN)],
            maxlen=ACTION_BUF_LEN,
        )
        self.cached_depth_latent = torch.zeros(1, DEPTH_LATENT_DIM)
        self.cached_yaw_pred = torch.zeros(1, 2)
        self.have_yaw_pred = False
        self.have_depth_latent = False
        self.last_contact_raw = np.zeros(4, dtype=np.float64)
        self._last_raw_foot_force = np.zeros(4)
        self._depth_q = deque(maxlen=32)
        self._depth_seq = 0
        self._last_capture_s = None   # source-frame identity for dedup
        self._dup_frame_count = 0     # republished duplicates dropped
        self._encoded_seq = -1
        self._early_count = 0        # frames encoded before reaching the budget
        self._skew_count = 0         # ticks where the capture stamp was distrusted
        self._proprio_late_count = 0  # ticks where the proprio pad could not be met
        self._out_of_band_count = 0   # encodes above the trained latency band
        # Width of the trained latency band above the budget. Mirrors Isaac's
        # cfg.depth.latency_range span in seconds: latency_range=[4,5] capture
        # frames at capture_interval=1 (20 ms) => 80-100 ms, so a 20 ms band.
        self.depth_latency_band_s = 0.02
        # Max tolerated disagreement between the capture stamp (camera-host clock,
        # offset-corrected) and local receipt (host monotonic) before the stamp is
        # distrusted. The honest difference is the capture->arrival transport, tens
        # of ms at worst. Tied to the budget it protects: an offset barely past
        # the budget (~100-400 ms) already makes EVERY frame read older than the
        # budget — the gate then takes the newest frame each tick and the
        # constant-latency contract is silently defeated. A fixed 0.5 s tolerance
        # caught only the loud (>0.5 s) failure and admitted that quiet regime.
        self.clock_skew_tol_s = max(2.0 * self.depth_latency_s, 0.1)
        # Age of the most recently encoded frame. Persisted across ticks because
        # the periodic status line lands on a non-encoder tick (the counter is
        # incremented before the log), where the per-tick value is still 0.
        self._last_depth_age_s = 0.0
        self._clock_offset_s = 0.0
        self._clock_offset_ready = False
        self._sim_truth_vel_body = np.zeros(3, dtype=np.float32)
        self._have_sim_truth_vel = False
        self._ekf_vel_body = np.zeros(3, dtype=np.float32)
        self._have_ekf_vel = False
        self._mj_world_xy_yaw = np.zeros(3, dtype=np.float32)
        self._last_yaw_cmd = (0.0, 0.0)
        self._policy_engaged = False
        self._tick_count = 0
        self._overrun_count = 0
        self._tick_budget_s = 1.0 / rate_hz

        # ----- topics
        self.xbox = XboxController(self)
        self.create_subscription(LowState, "/lowstate", self._on_lowstate, 10)
        self.create_subscription(LowState, "/mujoco/lowstate", self._on_lowstate, 10)
        self.create_subscription(Image, "/rl/depth_image", self._on_depth, 10)
        self.create_subscription(Float32MultiArray, "/rl/depth_clock_offset",
                                 self._on_clock_offset, 10)
        self.create_subscription(Bool, "/policy_engaged", self._on_engaged, 10)
        self.create_subscription(Float32MultiArray, "/mujoco/pos",
                                 self._on_mujoco_pos, 10)
        self.create_subscription(Twist, "/sim_truth/base_velocity",
                                 self._on_sim_truth_vel, 10)
        self.create_subscription(Twist, "/base_velocity", self._on_ekf_vel, 10)

        self.target_pub = self.create_publisher(Float32MultiArray, "/rl/target_pos", 10)
        # Pure outputs from here down: published for the viewer/recorder and for
        # offline traces. Nothing in this node reads them back — that round-trip
        # (the old /rl/last_action + /rl/contact_filt hops) was F02/F03.
        self.last_action_pub = self.create_publisher(Float32MultiArray, "/rl/last_action", 10)
        self.contact_filt_pub = self.create_publisher(Float32MultiArray, "/rl/contact_filt", 10)
        self.latent_pub = self.create_publisher(Float32MultiArray, "/rl/depth_latent", 10)
        self.est_lin_vel_pub = self.create_publisher(Float32MultiArray, "/rl/est_lin_vel", 10)
        self.proprio_diag_pub = self.create_publisher(
            Float32MultiArray, "/rl/proprio_latency", 10)
        self.latency_diag_pub = self.create_publisher(
            Float32MultiArray, "/rl/depth_latency", 10)

        self.timer = self.create_timer(self._tick_budget_s, self._tick)

        vel_mode = ("force_lin_vel_x" if not np.isnan(self.force_lin_vel_x)
                    else "binary" if self.binary_vel else "proportional")
        yaw_mode = ("use_true_yaw" if self.use_true_yaw
                    else "zero_yaw_pred" if self.zero_yaw_pred else "encoder")
        self.get_logger().info(
            f"distill_fused_policy ready — {rate_hz:g} Hz actor, encoder every "
            f"{self.depth_update_interval} ticks "
            f"({rate_hz / self.depth_update_interval:g} Hz), "
            f"depth budget={self.depth_latency_s * 1e3:.0f} ms (age gate), "
            f"proprio pad={self.proprio_latency_s * 1e3:.0f} ms, "
            f"action_delay={action_delay}, cmd_vx={vel_mode}, yaw={yaw_mode}, "
            f"env_class_eq_17={int(self.env_class_eq_17)}, "
            f"contact_threshold={self.contact_threshold}")
        self.get_logger().info(
            "ONE proprio frame per tick feeds encoder and actor — the "
            "encoder/actor flag-desync class is structurally absent")

    # ------------------------------------------------------------------
    # callbacks
    # ------------------------------------------------------------------
    def _on_lowstate(self, msg: LowState):
        self.latest_lowstate_receive_mono_ns = host_monotonic_ns()
        self.latest_lowstate = msg
        self.have_lowstate = True

    def _on_depth(self, msg: Image):
        """Enqueue only — the encoder runs inside the 50 Hz tick, never here."""
        if msg.encoding != "32FC1":
            self.get_logger().warning(
                f"depth expects 32FC1, got {msg.encoding}", throttle_duration_sec=2.0)
            return
        if (msg.height, msg.width) != (DEPTH_TARGET_H, DEPTH_TARGET_W):
            self.get_logger().warning(
                f"depth shape ({msg.height},{msg.width}) != "
                f"({DEPTH_TARGET_H},{DEPTH_TARGET_W})", throttle_duration_sec=2.0)
            return
        capture_s = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        # Reject REPUBLISHED duplicates of a capture we already hold.
        #
        # depth_camera_node caches the newest raw frame and republishes it from a
        # fixed-rate timer for up to max_frame_age_s (0.5 s) — and
        # _process_and_publish preserves the SOURCE header.stamp, so a stalled
        # camera yields several publications of ONE capture. Keying dedup on our own
        # arrival counter made each look like a new frame, breaking the contract
        # two ways:
        #   1. the recurrent encoder advanced ~5x on one capture. Isaac steps the
        #      GRU exactly once per distinct render (depth_buffer +
        #      update_interval), so duplicate steps are an off-contract observation
        #      SEQUENCE — the expensive error class per FINDINGS F12.
        #   2. each republication refreshed the newest-frame receipt time, so the
        #      stale-image guard could not trip until republishing stopped:
        #      ~0.5 s + depth_max_age_s of driving on frozen depth.
        # The capture stamp IS the source-frame identity, so dedup here fixes both.
        if capture_s == self._last_capture_s:
            self._dup_frame_count += 1
            self.get_logger().warning(
                f"duplicate depth capture (stamp {capture_s:.6f}) — camera stalled "
                f"and depth_camera_node is republishing; dropped so the GRU steps "
                f"once per distinct render (dup #{self._dup_frame_count})",
                throttle_duration_sec=2.0)
            return
        self._last_capture_s = capture_s
        arr = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
        self._depth_q.append(
            DepthFrame(capture_s, host_monotonic_ns(), arr.copy(), self._depth_seq))
        self._depth_seq = (self._depth_seq + 1) & 0xffffffff

    def _on_clock_offset(self, msg: Float32MultiArray):
        # [offset_ms, transport_floor_ms, baseline_ready] from depth_camera_node's
        # passive Jetson<->host drift tracker. Never published in sim.
        if len(msg.data) < 3:
            return
        offset_s = float(msg.data[0]) * 1e-3
        if abs(offset_s) > 0.5:
            self.get_logger().error(
                f"implausible clock offset {offset_s * 1e3:+.1f} ms ignored",
                throttle_duration_sec=5.0)
            return
        self._clock_offset_s = offset_s
        if not self._clock_offset_ready and float(msg.data[2]) >= 1.0:
            self._clock_offset_ready = True
            self.get_logger().info(
                f"clock-offset correction active ({offset_s * 1e3:+.2f} ms)")

    def _on_engaged(self, msg: Bool):
        was = self._policy_engaged
        self._policy_engaged = bool(msg.data)
        if self._policy_engaged and not was:
            if self.reset_on_engage:
                # ONE reset path. Matches an Isaac episode reset: no
                # pre-engagement recurrent, observation-history, action-history
                # or contact state may leak into the first policy observation.
                # In the two-node stack this had to happen consistently in two
                # processes off the same edge (F01).
                self.encoder.hidden_states = None
                self.history_ring = None
                self.last_action = torch.zeros(1, NUM_ACTIONS)
                self.action_buf = deque(
                    [torch.zeros(1, NUM_ACTIONS) for _ in range(ACTION_BUF_LEN)],
                    maxlen=ACTION_BUF_LEN,
                )
                self.cached_depth_latent.zero_()
                self.cached_yaw_pred.zero_()
                self.have_depth_latent = False
                self.have_yaw_pred = False
                self.last_contact_raw[:] = 0.0
                self._encoded_seq = -1
                self._depth_q.clear()
            self.get_logger().info(
                "/policy_engaged=True — reset encoder GRU + actor episode state")
        elif was and not self._policy_engaged:
            self.get_logger().info("/policy_engaged=False — RL inference paused")

    def _on_mujoco_pos(self, msg: Float32MultiArray):
        if len(msg.data) < 7:
            return
        q = np.asarray(msg.data[3:7], dtype=np.float64)
        _, _, yaw = mj_quat_to_roll_pitch_yaw(q)
        self._mj_world_xy_yaw[0] = float(msg.data[0])
        self._mj_world_xy_yaw[1] = float(msg.data[1])
        self._mj_world_xy_yaw[2] = yaw

    def _on_sim_truth_vel(self, msg: Twist):
        self._sim_truth_vel_body[:] = (msg.linear.x, msg.linear.y, msg.linear.z)
        if not self._have_sim_truth_vel:
            self._have_sim_truth_vel = True
            self.get_logger().info("first /sim_truth/base_velocity received")

    def _on_ekf_vel(self, msg: Twist):
        self._ekf_vel_body[:] = (msg.linear.x, msg.linear.y, msg.linear.z)
        if not self._have_ekf_vel:
            self._have_ekf_vel = True
            self.get_logger().info("first /base_velocity (EKF) received")

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _read_lin_vel_x(self) -> float:
        """Single owner of prop[10]. No mirrored copy to keep in lockstep."""
        if not np.isnan(self.force_lin_vel_x):
            return float(self.force_lin_vel_x)
        if not (self.xbox.axes and self.xbox.buttons):
            return 0.0
        lb_rb = self.xbox.is_pressed()
        if not (lb_rb[0] and lb_rb[1]):
            return 0.0
        lx, _ly = self.xbox.get_left_stick()
        if self.binary_vel:
            # bang-bang: forward push past the deadzone -> full lin_vel_max,
            # else 0 (backward / centred stick -> 0, no reverse).
            return self.lin_vel_max if float(lx) > self.vel_deadzone else 0.0
        return self.lin_vel_max * float(lx)

    def _quit_combo(self) -> bool:
        # Threshold, not == -1.0: analog triggers on some pads never report
        # exactly -1.0 when floored (matches the low_level_ctrl e-stop check).
        a = self.xbox.axes
        return bool(a) and len(a) >= 6 and a[2] < -0.9 and a[5] < -0.9

    def _advance_contact_filt(self) -> np.ndarray:
        """Advance Isaac's one-policy-step contact history. Once per tick."""
        ff = np.asarray(self.latest_lowstate.foot_force, dtype=np.float64)
        if ff.size == 4:
            contact_filt, raw_now = compute_contact_filt(
                ff, self.last_contact_raw, threshold_n=self.contact_threshold,
            )
            self.last_contact_raw = raw_now
            self._last_raw_foot_force = ff
        else:
            contact_filt = np.zeros(4)
            self._last_raw_foot_force = np.zeros(4)
        if self.diagnostics:
            msg = Float32MultiArray()
            msg.data = contact_filt.astype(np.float32).tolist()
            self.contact_filt_pub.publish(msg)
        return contact_filt

    def _frame_age_s(self, f, now_s, now_mono_ns, trusted=True):
        """Age of one queued frame, in seconds, using the trustworthy clock.

        HARDWARE HAZARD this guards: the capture stamp is RealSense GLOBAL_TIME in
        the *camera host* (Jetson) clock, while `now_s` is this host's ROS clock.
        They are reconciled by `_clock_offset_s` from depth_camera_node's drift
        tracker — but that topic does not exist in sim and does not exist on the
        real robot until the tracker has a baseline. With offset 0 and a Jetson
        clock lagging by even ~1 s, EVERY frame reads as older than the budget, so
        the gate always takes the newest frame and the constant-latency contract is
        silently defeated (no error, no warning, just no delay).

        Local receipt (`receive_mono_ns`, host monotonic) cannot be skewed. It
        under-reports the true age by the capture->arrival transport only, which is
        bounded and small (single-digit to tens of ms). So: prefer the capture
        stamp, but when the two disagree by more than the tolerance, distrust the
        cross-clock number and age by receipt instead.

        Trust is decided ONCE PER TICK on the newest frame (`_stamp_trusted`) and
        applied to the whole queue: skew is a property of the clock *stream*, not
        of individual frames, and mixing clocks within one queue walk breaks the
        oldest->newest age monotonicity the selection loop relies on.
        """
        if trusted:
            return now_s - f.capture_s - self._clock_offset_s
        return (now_mono_ns - f.receive_mono_ns) * 1e-9

    def _stamp_trusted(self, f, now_s, now_mono_ns):
        """Per-tick trust decision, made on one frame (the newest).

        Counts and logs once per tick — not per frame — so `skew=` on the status
        line is comparable to `early=` and the tick rate.
        """
        age_capture = now_s - f.capture_s - self._clock_offset_s
        age_recv = (now_mono_ns - f.receive_mono_ns) * 1e-9
        if abs(age_capture - age_recv) > self.clock_skew_tol_s:
            self._skew_count += 1
            self.get_logger().error(
                f"depth capture stamp disagrees with local receipt by "
                f"{(age_capture - age_recv) * 1e3:+.0f} ms (tol "
                f"{self.clock_skew_tol_s * 1e3:.0f} ms) — clock domains unreconciled"
                f"{'' if self._clock_offset_ready else ' (/rl/depth_clock_offset not ready)'}"
                f"; aging depth by RECEIPT instead. Check SYNC_JETSON_CLOCK.",
                throttle_duration_sec=5.0)
            return False
        return True

    def _select_depth_frame(self):
        """Newest queued frame already at least `depth_latency_s` old.

        This is the non-blocking form of the paper's constant-latency pad: the
        encoder never consumes a frame fresher than the budget. Frames older
        than the selected one are dropped (they can never be selected again).
        Returns (frame, age_s) or (None, 0.0) when the queue is empty.
        """
        if not self._depth_q:
            return None, 0.0
        now_s = self.get_clock().now().nanoseconds * 1e-9
        now_mono_ns = host_monotonic_ns()
        # One trust decision per tick, made on the newest frame, applied to the
        # whole queue: all frames share the clock stream, and a single clock
        # keeps ages monotonic oldest->newest (the `else: break` below relies
        # on that).
        trusted = self._stamp_trusted(self._depth_q[-1], now_s, now_mono_ns)
        if self.depth_latency_s <= 0.0:
            f = self._depth_q[-1]
            return f, self._frame_age_s(f, now_s, now_mono_ns, trusted)
        chosen = None
        chosen_age = 0.0
        for f in self._depth_q:                 # oldest -> newest
            age = self._frame_age_s(f, now_s, now_mono_ns, trusted)
            if age >= self.depth_latency_s:
                chosen, chosen_age = f, age     # advance to the newest qualifying
            else:
                break
        if chosen is None:
            # Every queued frame is still fresher than the budget -> encode NOTHING
            # this tick and keep the cached latent.
            #
            # This used to encode the oldest frame anyway "to keep the actor fed".
            # That was wrong: Isaac reads depth_buffer[-1-L], i.e. a frame that is
            # ALWAYS at least the configured delay old, and the encoder is
            # recurrent — one under-age observation does not just affect this tick,
            # it perturbs the GRU state for every tick after it. And it fired
            # predictably rather than rarely, because engagement clears the queue,
            # so the first post-engagement frame is always under-age.
            #
            # Waiting is already a supported state: before the first latent
            # `have_depth_latent` is False and the tick withholds /rl/target_pos
            # (low_level_ctrl's rl_stall_timeout covers a genuinely dead stream).
            # After the first latent, holding the previous one is exactly what
            # happens between encoder ticks anyway.
            self._early_count += 1
            self.get_logger().warning(
                f"no queued depth frame reached the "
                f"{self.depth_latency_s * 1e3:.0f} ms budget "
                f"(newest {self._frame_age_s(self._depth_q[-1], now_s, now_mono_ns, trusted) * 1e3:.0f} ms, "
                f"queue={len(self._depth_q)}) — skipping this encode and keeping "
                f"the cached latent (early #{self._early_count})",
                throttle_duration_sec=2.0)
            return None, 0.0
        # TRAINED-BAND DIAGNOSTIC. Isaac renders every capture_interval control
        # steps (go2_config.py: capture_interval=1 -> 20 ms) and the encoder reads
        # depth_buffer[-1-L] with L in cfg.depth.latency_range ([4,5] for the
        # cliffside distill students) — i.e. depth aged EXACTLY 80-100 ms at
        # encode. This gate only guarantees a floor, so with a slower deploy camera
        # the delivered age can sit far above the band: at 10 Hz the available ages
        # are ~68/168/268 ms, so an unlucky phase yields 168 ms, ~70 ms past the
        # trained cap, while the 68 ms frame is below the floor (training never saw
        # fresher — "never train on fresher depth than the pad allows").
        #
        # Nothing better can be SELECTED here: walking oldest->newest the ages
        # decrease, so `chosen` is already the closest frame above the floor, and
        # going fresher would break the floor. The cure is more frames per second
        # (depth_rate_hz -> 1/capture_interval, i.e. 50 Hz to match training), not
        # a different pick. So: measure and report the gap instead of hiding it.
        if self.depth_latency_s > 0.0:
            hi = self.depth_latency_s + self.depth_latency_band_s
            if chosen_age > hi:
                self._out_of_band_count += 1
                self.get_logger().warning(
                    f"depth age {chosen_age * 1e3:.0f} ms exceeds the TRAINED band "
                    f"[{self.depth_latency_s * 1e3:.0f}, {hi * 1e3:.0f}] ms by "
                    f"{(chosen_age - hi) * 1e3:.0f} ms — the camera is too slow for "
                    f"the band to be reachable (raise depth_rate_hz toward "
                    f"1/capture_interval); off-distribution vs training "
                    f"(#{self._out_of_band_count})",
                    throttle_duration_sec=5.0)
        # Drop everything older than the selection — it can never be chosen
        # again. Identity, not seq order, so the 32-bit seq wrap is a non-issue.
        while self._depth_q and self._depth_q[0] is not chosen:
            self._depth_q.popleft()
        return chosen, chosen_age

    # ------------------------------------------------------------------
    # main loop
    # ------------------------------------------------------------------
    def _tick(self):
        tick_start_ns = host_monotonic_ns()
        if self._quit_combo():
            sys.exit(0)
        if not self.have_lowstate or not self._policy_engaged:
            return

        # ---- stale-input fail-safe. Armed only while engaged, so lay/stand
        # bring-up (C++ owns the joints) is unaffected. Ceasing /rl/target_pos
        # lets low_level_ctrl's rl_stall_timeout latch a settle-to-stand.
        now_ns = host_monotonic_ns()
        if self.lowstate_max_age_s > 0:
            lowstate_age_s = (now_ns - self.latest_lowstate_receive_mono_ns) * 1e-9
            if lowstate_age_s > self.lowstate_max_age_s:
                self._stale_stopped = True
                self.get_logger().error(
                    f"STALE /lowstate ({lowstate_age_s:.2f} s > "
                    f"{self.lowstate_max_age_s:g} s) — ceasing /rl/target_pos",
                    throttle_duration_sec=1.0)
                return

        # ---- 1. ONE proprio frame. Everything downstream reads this.
        contact_filt = self._advance_contact_filt()

        ms = self.latest_lowstate.motor_state
        qj_unitree = np.array([ms[i].q for i in range(NUM_ACTIONS)], dtype=np.float64)
        dqj_unitree = np.array([ms[i].dq for i in range(NUM_ACTIONS)], dtype=np.float64)
        qj_policy = qj_unitree[MJ_TO_POLICY]
        dqj_policy = dqj_unitree[MJ_TO_POLICY]

        quat = np.asarray(self.latest_lowstate.imu_state.quaternion, dtype=np.float64)
        roll, pitch, cur_yaw = mj_quat_to_roll_pitch_yaw(quat)
        ang_vel_body = np.asarray(
            self.latest_lowstate.imu_state.gyroscope, dtype=np.float64)

        cmd_vx = self._read_lin_vel_x()
        prop = build_prop(
            qj_policy=qj_policy, dqj_policy=dqj_policy,
            ang_vel_body=ang_vel_body, roll=roll, pitch=pitch,
            last_action_pol=self.last_action.numpy().reshape(-1),
            contact_filt=contact_filt,
            lin_vel_x=cmd_vx,
            env_class_neq_17=0.0 if self.env_class_eq_17 else 1.0,
            env_class_eq_17=1.0 if self.env_class_eq_17 else 0.0,
        )
        # build_prop leaves prop[6:8] = 0, which is already the encoder's
        # yaw-masked form (on_policy_runner.py:313). The actor's copy gets the
        # heading written below; the encoder consumes this frame as-is.
        prop_t = torch.from_numpy(prop).float().unsqueeze(0)

        # ---- 2. encoder, every `depth_update_interval` ticks, on THIS prop.
        encoder_ran = 0
        encode_start_ns = encode_end_ns = tick_start_ns
        depth_capture_s = 0.0
        depth_age_s = 0.0
        used_depth_seq = self._encoded_seq
        if self._tick_count % self.depth_update_interval == 0:
            frame, depth_age_s = self._select_depth_frame()
            # Never step the GRU twice on the same frame. Training advances it
            # once per distinct render (depth.update_interval), so a duplicate
            # step is an off-contract observation *sequence* — the expensive
            # error class per F12 — whereas the resulting cadence jitter is the
            # cheap one (the handoff's hypothesis 3 measured delivery jitter as
            # harmless, 9/10). On a duplicate we keep the cached latent, exactly
            # as the two-node actor did between latent arrivals.
            if frame is not None and frame.seq == self._encoded_seq:
                frame = None
            if frame is not None:
                depth_capture_s = frame.capture_s
                depth_t = torch.from_numpy(frame.arr).float().unsqueeze(0)
                enc_prop = prop_t.clone()       # already yaw-masked
                encode_start_ns = host_monotonic_ns()
                with torch.no_grad():
                    out = self.encoder(depth_t, enc_prop)
                encode_end_ns = host_monotonic_ns()
                latent_full = out.squeeze(0)
                if not torch.isfinite(latent_full).all():
                    self.get_logger().error(
                        "non-finite depth latent — keeping previous",
                        throttle_duration_sec=1.0)
                else:
                    self.cached_depth_latent = latent_full[:DEPTH_LATENT_DIM].unsqueeze(0)
                    if self._has_yaw_head:
                        self.cached_yaw_pred = latent_full[
                            DEPTH_LATENT_DIM:DEPTH_LATENT_DIM + 2].unsqueeze(0)
                        self.have_yaw_pred = True
                    encoder_ran = 1
                    self._encoded_seq = frame.seq
                    used_depth_seq = frame.seq
                    self._last_depth_age_s = depth_age_s
                    if not self.have_depth_latent:
                        self.have_depth_latent = True
                        self.get_logger().info(
                            f"first depth latent (dim={self._enc_out_dim}); "
                            "policy unblocked")
                    if self.publish_latent and self.diagnostics:
                        lm = Float32MultiArray()
                        lm.data = latent_full.numpy().astype(np.float32).tolist()
                        lm.layout.data_offset = frame.seq
                        self.latent_pub.publish(lm)
                    if self.diagnostics:
                        diag = Float32MultiArray()
                        diag.data = [depth_age_s * 1e3, 0.0, depth_age_s * 1e3,
                                     self.depth_latency_s * 1e3,
                                     float(self._early_count),
                                     self._clock_offset_s * 1e3]
                        self.latency_diag_pub.publish(diag)
                    if self.trace_log_fp is not None:
                        now_s = self.get_clock().now().nanoseconds * 1e-9
                        vals = np.concatenate((
                            prop, self.cached_depth_latent.squeeze(0).numpy()))
                        self.trace_log_fp.write(
                            f"{self.trace_tick}\t{now_s:.9f}\t{depth_capture_s:.9f}\t"
                            f"{int(self._policy_engaged)}\t" +
                            "\t".join(f"{float(v):+.8f}" for v in vals) + "\n")
                        self.trace_tick += 1

        if not self.have_depth_latent:
            # Block startup rather than driving the motors from a zero latent:
            # in sim that unfreezes mujoco_simulator (which waits on the first
            # /mujoco/lowcmd) under garbage actions and tips the robot.
            self._tick_count += 1
            return

        if self.depth_max_age_s > 0 and self._depth_q:
            image_age_s = (now_ns - self._depth_q[-1].receive_mono_ns) * 1e-9
            if image_age_s > self.depth_max_age_s:
                self._stale_stopped = True
                self.get_logger().error(
                    f"STALE /rl/depth_image ({image_age_s:.2f} s > "
                    f"{self.depth_max_age_s:g} s) — ceasing /rl/target_pos",
                    throttle_duration_sec=1.0)
                return
        if self._stale_stopped:
            self._stale_stopped = False
            self.get_logger().info(
                "stale inputs recovered — resuming /rl/target_pos publication")

        # ---- 3. history ring (yaw-masked frames, as in training)
        prop_hist_frame = prop_t.clone()
        prop_hist_frame[:, 6:8] = 0.0
        if self.history_ring is None:
            # Match Isaac legged_robot.py:466 — zeros, not repeats of frame 0.
            self.history_ring = torch.zeros(1, HISTORY_LEN, N_PROPRIO)
        else:
            self.history_ring = torch.cat(
                [self.history_ring[:, 1:], prop_hist_frame.unsqueeze(1)], dim=1)

        # ---- 4. heading slots for the actor's live frame
        if self.use_true_yaw:
            # DIAGNOSTIC: true delta_yaw for a +x corridor (target heading 0).
            d = (-cur_yaw + np.pi) % (2 * np.pi) - np.pi
            prop_t[:, 6] = float(d)
            prop_t[:, 7] = float(d)
        elif self.have_yaw_pred and not self.zero_yaw_pred:
            # obs_student[:, 6:8] = 1.5 * yaw_pred (on_policy_runner.py:336).
            prop_t[:, 6:8] = YAW_PRED_MULTIPLIER * self.cached_yaw_pred
        else:
            prop_t[:, 6:8] = 0.0
        self._last_yaw_cmd = (float(prop_t[0, 6]), float(prop_t[0, 7]))

        # ---- 5. estimator + actor
        inference_start_ns = host_monotonic_ns()
        with torch.no_grad():
            priv_expl = self.estimator(prop_t)
        learned_est_lin_vel_body = priv_expl[0, 0:3].cpu().numpy() / 2.0
        if self.zero_student_velocity_obs:
            priv_expl = priv_expl.clone()
            priv_expl[0, 0:3] = 0.0
        elif self.use_sim_truth_vel:
            sim_vel = torch.from_numpy(self._sim_truth_vel_body).to(priv_expl.dtype)
            priv_expl = torch.zeros_like(priv_expl)
            priv_expl[0, 0:3] = sim_vel * 2.0
        est_lin_vel_body = priv_expl[0, 0:3].cpu().numpy() / 2.0

        obs = assemble_actor_obs(prop_t, priv_expl, self.history_ring)
        with torch.no_grad():
            action_raw = self.actor(
                obs, hist_encoding=True, eval=False,
                scandots_latent=self.cached_depth_latent,
            )
        action_clipped = (action_raw.clamp(-CLIP_ACTIONS, CLIP_ACTIONS)
                          if self.clip_applied_action else action_raw)
        # MATCH TRAINING: legged_robot.py:124 stores the RAW actor output in
        # action_history_buf and feeds it back unclipped into prop[37:49]. This
        # is now a plain attribute, not a topic round-trip (F03).
        self.last_action = action_raw.detach()

        self.action_buf.append(action_clipped.detach())
        action_to_apply = (self.action_buf[-1 - ACTION_DELAY_VIEW]
                           if self.action_delay else action_clipped)
        action_pol = action_to_apply.squeeze(0).numpy()
        target_dof_unitree = DEFAULT_DOF_POLICY + ACTION_SCALE * action_pol
        inference_end_ns = host_monotonic_ns()

        # ---- 6. proprio latency pad (paper §4.1: 16 ms, referenced to the
        # receipt of the lowstate this action was computed from).
        sleep_start_ns = host_monotonic_ns()
        _wait = 0.0
        if self.proprio_latency_s > 0.0:
            _elapsed = (sleep_start_ns - self.latest_lowstate_receive_mono_ns) * 1e-9
            _wait = self.proprio_latency_s - _elapsed
            if 0.0 < _wait < 0.5:
                # Hybrid precision sleep. Plain time.sleep() overshoots by
                # 3-4 ms on macOS; with /lowstate at 500 Hz the lowstate is
                # always fresh, so nearly the whole pad is slept inside the
                # 20 ms tick — the overshoot pushed the callback past its own
                # period and the 50 Hz timer slipped to ~44.7 Hz (measured
                # 2026-07-27 dryrun: sleep p50 18.9 ms for a 16 ms pad,
                # lowstate->publish 20.3 ms). Coarse-sleep to ~2 ms before
                # the deadline, then spin the tail: overshoot < 0.1 ms for
                # < 2 ms of burned CPU per tick.
                _deadline_ns = (self.latest_lowstate_receive_mono_ns
                                + int(self.proprio_latency_s * 1e9))
                _coarse = _wait - 1e-3
                if _coarse > 0.0:
                    time.sleep(_coarse)
                while host_monotonic_ns() < _deadline_ns:
                    pass
            elif _wait <= 0.0:
                # Inference already took longer than the pad, so receipt->publish
                # exceeds the configured constant and the pad is NOT enforced this
                # tick. Silent on the two-node path; surfaced here because on
                # hardware a persistent overrun means the actor is slower than its
                # own latency contract. Monotonic clocks, so no skew involved.
                self._proprio_late_count += 1
                self.get_logger().warning(
                    f"proprio pad not met: lowstate->publish already "
                    f"{_elapsed * 1e3:.1f} ms > {self.proprio_latency_s * 1e3:.0f} ms "
                    f"budget (late #{self._proprio_late_count})",
                    throttle_duration_sec=5.0)
        sleep_end_ns = host_monotonic_ns()

        out_msg = Float32MultiArray()
        out_msg.data = target_dof_unitree.astype(np.float32).tolist()
        out_msg.layout.data_offset = self._tick_count
        publish_start_ns = host_monotonic_ns()
        self.target_pub.publish(out_msg)
        publish_end_ns = host_monotonic_ns()

        # ---- 7. diagnostics / logs (pure outputs; --diagnostics 0 skips the
        # per-tick publishes entirely — nothing in the control loop reads them,
        # only the viewer/recorder do)
        if self.diagnostics:
            la = Float32MultiArray()
            la.data = action_raw.squeeze(0).numpy().astype(np.float32).tolist()
            self.last_action_pub.publish(la)

            elv = Float32MultiArray()
            elv.data = est_lin_vel_body.astype(np.float32).tolist()
            self.est_lin_vel_pub.publish(elv)

            diag = Float32MultiArray()
            diag.data = [
                (inference_start_ns - self.latest_lowstate_receive_mono_ns) * 1e-6,
                max(_wait, 0.0) * 1e3,
                (publish_end_ns - self.latest_lowstate_receive_mono_ns) * 1e-6,
                self.proprio_latency_s * 1e3,
            ]
            self.proprio_diag_pub.publish(diag)

        if self.latency_log_fp is not None:
            self.latency_log_fp.write(
                f"{self._tick_count}\t{int(self.latest_lowstate.tick)}\t"
                f"{self.latest_lowstate_receive_mono_ns}\t{used_depth_seq}\t"
                f"{depth_capture_s:.9f}\t{depth_age_s * 1e3:.3f}\t{encoder_ran}\t"
                f"{tick_start_ns}\t{encode_start_ns}\t{encode_end_ns}\t"
                f"{inference_start_ns}\t{inference_end_ns}\t{sleep_start_ns}\t"
                f"{sleep_end_ns}\t{publish_start_ns}\t{publish_end_ns}\n")

        if self.action_log_fp is not None:
            now = time.monotonic()
            if self.action_log_t0 is None:
                self.action_log_t0 = now
            values = np.concatenate((
                contact_filt,
                action_raw.squeeze(0).numpy(),
                action_clipped.squeeze(0).numpy(),
                target_dof_unitree,
            ))
            self.action_log_fp.write(
                f"{self._tick_count}\t{now - self.action_log_t0:.6f}\t"
                f"{self._mj_world_xy_yaw[0]:+.8f}\t{self._mj_world_xy_yaw[1]:+.8f}\t"
                f"{float(prop[10]):+.8f}\t" +
                "\t".join(f"{float(v):+.8f}" for v in values) + "\n")

        if self.proprio_log_fp is not None and self.proprio_log_ticks_remaining > 0:
            now = time.time()
            if self.proprio_log_t0 is None:
                self.proprio_log_t0 = now
            arr = prop_t.squeeze(0).numpy()
            self.proprio_log_fp.write(
                f"{self._proprio_log_written}\t{now - self.proprio_log_t0:.4f}\t"
                + "\t".join(f"{v:+.6f}" for v in arr) + "\t"
                + "\t".join(f"{v:+.6f}" for v in self._last_raw_foot_force) + "\n")
            self._proprio_log_written += 1
            self.proprio_log_ticks_remaining -= 1
            if self.proprio_log_ticks_remaining == 0:
                self.proprio_log_fp.close()
                self.proprio_log_fp = None
                self.get_logger().info(
                    f"proprio_log: wrote {self._proprio_log_written} ticks, closed")

        if self.vel_log_fp is not None and self.vel_log_ticks_remaining > 0:
            now = time.time()
            if self._vel_log_t0 is None:
                self._vel_log_t0 = now
            gt, ekf, lr = self._sim_truth_vel_body, self._ekf_vel_body, learned_est_lin_vel_body
            w = self._mj_world_xy_yaw
            yc0, yc1 = self._last_yaw_cmd
            self.vel_log_fp.write(
                f"{self._vel_log_written}\t{now - self._vel_log_t0:.4f}\t{cmd_vx:+.4f}"
                f"\t{gt[0]:+.4f}\t{gt[1]:+.4f}\t{gt[2]:+.4f}"
                f"\t{ekf[0]:+.4f}\t{ekf[1]:+.4f}\t{ekf[2]:+.4f}"
                f"\t{lr[0]:+.4f}\t{lr[1]:+.4f}\t{lr[2]:+.4f}"
                f"\t{w[0]:+.4f}\t{w[1]:+.4f}\t{np.degrees(w[2]):+.4f}"
                f"\t{np.degrees(yc0):+.4f}\t{np.degrees(yc1):+.4f}\n")
            self._vel_log_written += 1
            self.vel_log_ticks_remaining -= 1
            if self.vel_log_ticks_remaining == 0:
                self.vel_log_fp.close()
                self.vel_log_fp = None
                self.get_logger().info(
                    f"vel_log: wrote {self._vel_log_written} ticks, closed")

        self._tick_count += 1

        # Overrun watch: the fused tick pays encoder + actor every
        # depth_update_interval ticks. If that exceeds the 20 ms budget the
        # actor's cadence degrades, which is exactly the wall-clock effect F13
        # measured — so surface it loudly rather than letting it hide.
        spent_s = (host_monotonic_ns() - tick_start_ns) * 1e-9
        if spent_s > self._tick_budget_s:
            self._overrun_count += 1
            self.get_logger().warning(
                f"tick overrun {spent_s * 1e3:.1f} ms > "
                f"{self._tick_budget_s * 1e3:.1f} ms "
                f"(encoder_ran={encoder_ran}, #{self._overrun_count})",
                throttle_duration_sec=2.0)

        if self._tick_count % 50 == 0:
            yc0, yc1 = self._last_yaw_cmd
            self.get_logger().info(
                f"[cmd] vx={cmd_vx:+.2f} m/s  |  "
                f"[est] vx={est_lin_vel_body[0]:+.2f} vy={est_lin_vel_body[1]:+.2f} "
                f"vz={est_lin_vel_body[2]:+.2f}  yaw_cmd=({np.degrees(yc0):+6.1f}°, "
                f"{np.degrees(yc1):+6.1f}°)  "
                f"|dlat|={float(self.cached_depth_latent.norm()):.2f}  "
                f"depth_age={self._last_depth_age_s * 1e3:.0f}ms "
                f"early={self._early_count} skew={self._skew_count} oob={self._out_of_band_count} "
                f"clk_ofs={'rdy' if self._clock_offset_ready else 'n/a'}")

    def destroy_node(self):
        for attr in ("trace_log_fp", "action_log_fp", "latency_log_fp",
                     "proprio_log_fp", "vel_log_fp"):
            fp = getattr(self, attr, None)
            if fp is not None:
                fp.close()
                setattr(self, attr, None)
        return super().destroy_node()


def _opt_path(p):
    """Treat "" and "." as unset — launch files substitute empty strings."""
    if p is not None and str(p) in ("", "."):
        return None
    return p


# Flags that only ever existed to keep the two split processes in agreement.
# Fusing satisfies each one structurally, so accepting them silently would let a
# stale invocation *look* configured while the flag does nothing. argparse's
# parse_known_args (needed because ROS appends --ros-args) would swallow them, so
# reject them explicitly instead.
RETIRED_FLAGS = {
    "--use_shared_contact":
        "the actor and encoder now share one 50 Hz contact filter by "
        "construction (there is no second filter to point at).",
    "--publish_raw_last_action":
        "last_action is an in-process attribute and is always the RAW actor "
        "output, as training stores it; nothing round-trips through a topic.",
    "--gate_on_engage":
        "the encoder only ever steps inside an engaged tick, so there is no "
        "pre-engagement window to gate.",
    "--enforce_latency_s":
        "renamed to --depth_latency_s (accepted as an alias) — but note it is "
        "now enforced by an age gate over queued frames, not by sleeping.",
}


def _reject_retired_flags(argv):
    hits = [a for a in argv if a.split("=", 1)[0] in RETIRED_FLAGS
            and a.split("=", 1)[0] != "--enforce_latency_s"]
    if not hits:
        return
    lines = [f"distill_fused_policy: {len(hits)} retired flag(s) passed — "
             "these are no-ops here and would silently mislead:"]
    for h in hits:
        name = h.split("=", 1)[0]
        lines.append(f"  {name}: {RETIRED_FLAGS[name]}")
    lines.append("Remove them from the launch/script invocation.")
    print("\n".join(lines), file=sys.stderr)
    sys.exit(2)


def main():
    _reject_retired_flags(sys.argv[1:])
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--ep_root", type=Path,
                    default=_EP_ROOT_DEFAULT)
    ap.add_argument("--action_delay", type=int, choices=[0, 1], default=0)
    ap.add_argument("--lin_vel_max", type=float, default=0.7)
    ap.add_argument("--force_lin_vel_x", type=float, default=float("nan"),
                    help="bypass joystick gate; constant lin_vel_x in m/s")
    ap.add_argument("--binary_vel", type=int, choices=[0, 1], default=0,
                    help="1 = bang-bang: forward stick past --vel_deadzone commands a "
                         "CONSTANT lin_vel_max, else 0. Unlike the two-node stack this "
                         "is read once, so there is no encoder/actor copy to match.")
    ap.add_argument("--vel_deadzone", type=float, default=0.2)
    ap.add_argument("--contact_threshold", type=str, default=str(CONTACT_FORCE_THRESHOLD_N),
                    help="Foot-force threshold for prop[49:53]: one global value or four "
                         "comma-separated per-foot values in FR,FL,RR,RL order. Consumed "
                         "once (the encoder/actor lockstep requirement is gone).")
    ap.add_argument("--env_class_eq_17", type=int, choices=[0, 1], default=0,
                    help="1 sets prop[11:13] = (0, 1) for parkour_flat (env_class==17, "
                         "go2_flat / L00-* exptids); 0 gives (1, 0) for every other "
                         "env_class. Consumed once.")
    ap.add_argument("--depth_latency_s", "--enforce_latency_s", dest="depth_latency_s",
                    type=float, default=DEPTH_LATENCY_S,
                    help="Constant capture->encode budget (s). Enforced by selecting the "
                         "newest queued frame at least this old, NOT by sleeping. 0 = use "
                         "the freshest frame. Default 0.08 (paper §4.1).")
    ap.add_argument("--proprio_latency_s", type=float, default=PROPRIO_LATENCY_S,
                    help="Sleep after inference to enforce a fixed lowstate-receipt -> "
                         "target-publish latency (s). Default 0.016; 0 disables.")
    ap.add_argument("--depth_update_interval", type=int, default=UPDATE_INTERVAL_DEFAULT,
                    help="Run the encoder every N policy ticks. Default 5 = 10 Hz at a "
                         "50 Hz actor, matching training's depth.update_interval.")
    ap.add_argument("--reset_on_engage", type=int, choices=[0, 1], default=1,
                    help="1 clears encoder GRU, history, action buffer and contact "
                         "history on the engage edge (one path, all together).")
    ap.add_argument("--zero_yaw_pred", type=int, choices=[0, 1], default=0)
    ap.add_argument("--use_true_yaw", type=int, choices=[0, 1], default=0,
                    help="DIAGNOSTIC: overwrite prop[6:8] with true delta_yaw for a +x "
                         "corridor. Mutually exclusive with --zero_yaw_pred.")
    ap.add_argument("--use_sim_truth_vel", type=int, choices=[0, 1], default=0)
    ap.add_argument("--zero_student_velocity_obs", type=int, choices=[0, 1], default=0,
                    help="1 forces priv_explicit[0:3] to zero. Only for students trained "
                         "with zero_student_velocity_obs=True.")
    ap.add_argument("--clip_applied_action", type=int, choices=[0, 1], default=1)
    ap.add_argument("--diagnostics", type=int, choices=[0, 1], default=1,
                    help="0 = skip all per-tick diagnostic publishes "
                         "(last_action/contact_filt/est_lin_vel/latent/latency "
                         "diags) — use on real runs without a viewer/bag")
    ap.add_argument("--publish_latent", type=int, choices=[0, 1], default=1,
                    help="1 echoes the latent on /rl/depth_latent for viewers/recorders. "
                         "Nothing in this node consumes it.")
    ap.add_argument("--rate_hz", type=float, default=50.0)
    ap.add_argument("--lowstate_max_age_s", type=float, default=0.2,
                    help="Fail-safe: cease /rl/target_pos when /lowstate is older than "
                         "this. <= 0 disables.")
    ap.add_argument("--depth_max_age_s", "--latent_max_age_s", dest="depth_max_age_s",
                    type=float, default=0.6,
                    help="Fail-safe: cease /rl/target_pos when the newest /rl/depth_image "
                         "is older than this. <= 0 disables.")
    ap.add_argument("--trace_log", type=Path, default=None,
                    help="TSV of the encoder's proprio frame and latent, per encoder tick.")
    ap.add_argument("--action_log", type=Path, default=None)
    ap.add_argument("--latency_log", type=Path, default=None,
                    help="Per-tick timing TSV including depth age and encoder span.")
    ap.add_argument("--proprio_log", type=Path, default=None)
    ap.add_argument("--proprio_log_ticks", type=int, default=100)
    ap.add_argument("--vel_log", type=Path, default=None)
    ap.add_argument("--vel_log_ticks", type=int, default=2000)
    args, _ = ap.parse_known_args()

    rclpy.init()
    node = DistillFusedPolicy(
        args.ckpt, args.ep_root,
        action_delay=args.action_delay,
        lin_vel_max=args.lin_vel_max,
        force_lin_vel_x=args.force_lin_vel_x,
        binary_vel=bool(args.binary_vel),
        vel_deadzone=args.vel_deadzone,
        contact_threshold=args.contact_threshold,
        env_class_eq_17=bool(args.env_class_eq_17),
        depth_latency_s=args.depth_latency_s,
        proprio_latency_s=args.proprio_latency_s,
        depth_update_interval=args.depth_update_interval,
        reset_on_engage=bool(args.reset_on_engage),
        zero_yaw_pred=bool(args.zero_yaw_pred),
        use_true_yaw=bool(args.use_true_yaw),
        use_sim_truth_vel=bool(args.use_sim_truth_vel),
        zero_student_velocity_obs=bool(args.zero_student_velocity_obs),
        clip_applied_action=bool(args.clip_applied_action),
        publish_latent=bool(args.publish_latent),
        diagnostics=bool(args.diagnostics),
        rate_hz=args.rate_hz,
        lowstate_max_age_s=args.lowstate_max_age_s,
        depth_max_age_s=args.depth_max_age_s,
        trace_log_path=_opt_path(args.trace_log),
        action_log_path=_opt_path(args.action_log),
        latency_log_path=_opt_path(args.latency_log),
        proprio_log_path=_opt_path(args.proprio_log),
        proprio_log_ticks=args.proprio_log_ticks,
        vel_log_path=_opt_path(args.vel_log),
        vel_log_ticks=args.vel_log_ticks,
    )
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
