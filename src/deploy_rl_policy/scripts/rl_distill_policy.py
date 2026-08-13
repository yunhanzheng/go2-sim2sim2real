#!/usr/bin/env python3
"""50 Hz control node for the extreme-parkour distillation student.

Lifts the AI pipeline from distill_runner.py:
  - 53-D proprio assembly with the heading slots prop[6:8] zeroed
  - 10-frame yaw-masked history ring
  - Estimator forward → 9-D priv_explicit
  - 114-D actor input + Actor forward with scandots_latent=cached_depth_latent
  - clip → optional 1-tick action delay → default + 0.25 * action
  - reorder policy → MuJoCo/Unitree order and publish /rl/target_pos

The 32-D depth latent comes from /rl/depth_latent (depth_encoder_node).
The full /rl/last_action is published every tick so the encoder process can
include it in its next prop frame.

Architecture note (L00-flatstep student): the encoder's yaw/heading predictor
head was removed (RecurrentDepthBackbone.output_mlp Linear(512, 32+2) ->
Linear(512, 32)). The latent is now 32-D with no yaw_pred tail, and prop[6:8]
is held at 0 to match the student's depth-only training observation. The
--use_true_yaw diagnostic remains for A/B testing geometric yaw injection, but
note that any non-zero prop[6:8] is now off the student's training distribution.
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
# Tiny MLPs: 24-thread BLAS is ~100× slower than single-thread because of
# spawn overhead. Inside ROS where 6 nodes each spawn 24 threads, contention
# made the actor forward 85 ms. Pinning to 1 thread → 0.14 ms.
torch.set_num_threads(1)
from rclpy.node import Node
from std_msgs.msg import Bool, Float32MultiArray
from geometry_msgs.msg import Twist
from unitree_go.msg import LowState

from distill_common import (
    add_extreme_parkour_to_path, load_checkpoint, detect_action_delay,
    mj_quat_to_roll_pitch_yaw, build_prop, assemble_actor_obs,
    compute_contact_filt, CONTACT_FORCE_THRESHOLD_N, parse_contact_threshold,
    PROPRIO_LATENCY_S,
    MJ_TO_POLICY, DEFAULT_DOF_POLICY,
    NUM_ACTIONS, N_PROPRIO, N_PRIV_EXPLICIT, N_PRIV_LATENT,
    HISTORY_LEN, DEPTH_LATENT_DIM, YAW_PRED_MULTIPLIER,
    ACTION_BUF_LEN, ACTION_DELAY_VIEW, ACTION_SCALE, CLIP_ACTIONS,
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

class RLDistillPolicy(Node):
    def __init__(self, ckpt_path: Path, ep_root: Path,
                 action_delay: int, lin_vel_max: float,
                 force_lin_vel_x: float = float("nan"),
                 binary_vel: bool = False,
                 vel_deadzone: float = 0.2,
                 proprio_log_path: Path = None,
                 proprio_log_ticks: int = 100,
                 use_sim_truth_vel: bool = False,
                 zero_student_velocity_obs: bool = False,
                 vel_log_path: Path = None,
                 vel_log_ticks: int = 2000,
                 zero_yaw_pred: bool = False,
                 use_true_yaw: bool = False,
                 env_class_eq_17: bool = False,
                 contact_threshold: float = CONTACT_FORCE_THRESHOLD_N,
                 proprio_latency_s: float = PROPRIO_LATENCY_S,
                 publish_raw_last_action: bool = True,
                 clip_applied_action: bool = True,
                 action_log_path: Path = None,
                 latency_log_path: Path = None,
                 rate_hz: float = 50.0,
                 lowstate_max_age_s: float = 0.2,
                 latent_max_age_s: float = 0.6):
        super().__init__("rl_distill_policy")
        self.action_delay = action_delay
        # Stale-input fail-safe (see run_real_remote.sh watchdog note): if
        # /lowstate or /rl/depth_latent stops flowing, CEASE publishing
        # /rl/target_pos so low_level_ctrl's rl_stall_timeout latches its
        # settle-to-stand — instead of walking forever on cached inputs.
        # <= 0 disables the respective check.
        self.lowstate_max_age_s = lowstate_max_age_s
        self.latent_max_age_s = latent_max_age_s
        self._stale_stopped = False   # True while a stale guard is suppressing output
        self.proprio_latency_s = proprio_latency_s
        self.lin_vel_max = lin_vel_max
        self.force_lin_vel_x = force_lin_vel_x
        # binary_vel: command a CONSTANT lin_vel_max whenever the stick is pushed
        # forward past vel_deadzone (bang-bang), instead of scaling the command
        # proportionally with stick deflection. 0 otherwise (incl. backward push).
        self.binary_vel = binary_vel
        self.vel_deadzone = vel_deadzone
        # Foot-force threshold for proprio[49:53]. This node owns the single
        # 50 Hz contact filter used by both the actor and depth encoder.
        # scalar (one global threshold) or length-4 per-foot list (FR,FL,RR,RL);
        # parse_contact_threshold accepts a float, "20", or "10,20,20,17".
        self.contact_threshold = parse_contact_threshold(contact_threshold)
        self.publish_raw_last_action = bool(publish_raw_last_action)
        self.clip_applied_action = bool(clip_applied_action)
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
        self.lowstate_latency_log_fp = None
        if latency_log_path is not None:
            self.latency_log_fp = open(latency_log_path, "w", buffering=1)
            self.latency_log_fp.write(
                "target_seq\tlowstate_tick\tlowstate_receive_mono_ns\tdepth_seq\t"
                "latent_receive_mono_ns\ttick_start_mono_ns\tinference_start_mono_ns\t"
                "inference_end_mono_ns\tsleep_start_mono_ns\tsleep_end_mono_ns\t"
                "publish_start_mono_ns\tpublish_end_mono_ns\n"
            )
            lowstate_path = latency_log_path.with_name(
                latency_log_path.stem + "_lowstate" + latency_log_path.suffix)
            # /lowstate can arrive at 500 Hz. Buffer one-second batches so the
            # measurement itself does not add a filesystem write per sample.
            self.lowstate_latency_log_fp = open(lowstate_path, "w", buffering=262144)
            self.lowstate_latency_log_fp.write(
                "receive_seq\tlowstate_tick\treceive_mono_ns\treceive_wall_ns\n"
            )
        # When True, fill priv_explicit[0:3] from /sim_truth/base_velocity
        # (MuJoCo body-frame qvel) instead of the learned Estimator's prediction.
        # Tests whether velocity-estimate noise is the dominant sim2sim gap.
        self.use_sim_truth_vel = use_sim_truth_vel
        self.zero_student_velocity_obs = zero_student_velocity_obs
        self._sim_truth_vel_body = np.zeros(3, dtype=np.float32)
        self._have_sim_truth_vel = False
        # Latest EKF prediction from /base_velocity (Twist). Logger only —
        # we don't feed it back into the policy unless an explicit flag asks.
        self._ekf_vel_body = np.zeros(3, dtype=np.float32)
        self._have_ekf_vel = False
        # Retained for CLI/launch compatibility. The L00-flatstep student has
        # no yaw predictor head, so prop[6:8] is already held at 0 on the
        # default path — this flag is now a redundant no-op (kept so existing
        # launch files / scripts don't error on an unknown arg).
        self.zero_yaw_pred = zero_yaw_pred
        # When True, overwrite prop[6:8] with the *true* delta_yaw the policy
        # would see during straight-line walking along +x. Training defines
        # delta_yaw = target_yaw - current_yaw, with target_yaw=atan2(dy,dx)
        # toward the next goal. On the flat-strip corridor target_yaw=0 (the
        # corridor is +x aligned), so the ideal yaw command is just
        # -current_yaw wrapped to [-pi, pi]. DIAGNOSTIC ONLY for the L00-flatstep
        # student: that student was trained with prop[6:8]=0 (no yaw head), so
        # injecting a non-zero heading here is now OFF its training distribution.
        self.use_true_yaw = use_true_yaw
        # parkour_flat training (go2_flat task, L00-* exptids) runs on env_class==17,
        # so the proprio indicator bits at prop[11:13] are (0, 1) — distinct from
        # the (1, 0) packed for every other env_class. The L00-01 policy was trained
        # exclusively on parkour_flat; flipping these bits is the faithful indicator
        # for it.
        self.env_class_eq_17 = env_class_eq_17

        # 3-way velocity log: per-tick (t, cmd_vx, gt, ekf, est).
        self.vel_log_ticks_remaining = vel_log_ticks if vel_log_path else 0
        self.vel_log_fp = None
        if vel_log_path is not None:
            self.vel_log_fp = open(vel_log_path, "w")
            self.vel_log_fp.write(
                "tick\tt_s\tcmd_vx"
                "\tgt_vx\tgt_vy\tgt_vz"
                "\tekf_vx\tekf_vy\tekf_vz"
                "\test_vx\test_vy\test_vz"
                "\tworld_x\tworld_y\tworld_yaw_deg"
                "\tyaw_cmd0_deg\tyaw_cmd1_deg\n"
            )
            self._vel_log_t0 = None
            self.get_logger().info(
                f"vel_log: first {vel_log_ticks} ticks → {vel_log_path}"
            )
        self.proprio_log_ticks_remaining = proprio_log_ticks if proprio_log_path else 0
        self.proprio_log_fp = None
        if proprio_log_path is not None:
            field_names = (
                ["ang_vel_x*0.25", "ang_vel_y*0.25", "ang_vel_z*0.25",
                 "roll", "pitch",
                 "zero_dyaw", "delta_yaw", "delta_next_yaw",
                 "zero_cmdx", "zero_cmdy", "lin_vel_x",
                 "env_class!=17", "env_class==17"]
                + [f"dpos_{leg}{j}" for leg in ("FR","FL","RR","RL") for j in "htc"]
                + [f"dvel_{leg}{j}" for leg in ("FR","FL","RR","RL") for j in "htc"]
                + [f"act_{leg}{j}"  for leg in ("FR","FL","RR","RL") for j in "htc"]
                + [f"c_{leg}-0.5"   for leg in ("FR","FL","RR","RL")]
            )
            assert len(field_names) == 53, f"got {len(field_names)} field names"
            self.proprio_log_fp = open(proprio_log_path, "w")
            self.proprio_log_fp.write("tick\tt_s\t" + "\t".join(field_names) + "\t"
                                      + "\t".join(f"rawF_{leg}" for leg in ("FR","FL","RR","RL"))
                                      + "\n")
            self.proprio_log_t0 = None
            self.get_logger().info(
                f"proprio_log: writing first {proprio_log_ticks} ticks to {proprio_log_path}"
            )

        # ----- module setup
        add_extreme_parkour_to_path(ep_root)
        from rsl_rl.modules.actor_critic import Actor, get_activation
        from rsl_rl.modules.estimator import Estimator

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

        # ----- runtime state
        self.latest_lowstate: LowState = LowState()
        self.latest_lowstate_receive_mono_ns = 0
        self.lowstate_receive_seq = 0
        self.have_lowstate = False
        self.history_ring = None  # (1, HISTORY_LEN, 53)
        self.last_action = torch.zeros(1, NUM_ACTIONS)
        self.action_buf = deque(
            [torch.zeros(1, NUM_ACTIONS) for _ in range(ACTION_BUF_LEN)],
            maxlen=ACTION_BUF_LEN,
        )
        self.cached_depth_latent = torch.zeros(1, DEPTH_LATENT_DIM)
        # Yaw-head students (EP_FEED_PRED_YAW=1, e.g. the OD cliffside/flyball
        # student) emit a 34-D latent: 32 scan dims + 2 regressed-heading dims.
        # cached_yaw_pred holds that tail; have_yaw_pred is set once a 34-D
        # message is seen, switching prop[6:8] from 0 to 1.5 * yaw_pred.
        self.cached_yaw_pred = torch.zeros(1, 2)
        self.have_yaw_pred = False
        self.have_depth_latent = False
        self.latest_depth_seq = 0
        self.latest_latent_receive_mono_ns = 0
        # Persistent RAW contact (not filtered) for the Isaac-style OR with
        # previous step. Storing the filtered value would latch the bit
        # permanently — training stores raw at legged_robot.py:334.
        self.last_contact_raw = np.zeros(4, dtype=np.float64)

        # ----- joystick / topics
        self.xbox = XboxController(self)
        self.create_subscription(LowState, "/lowstate", self._on_lowstate, 10)
        self.create_subscription(LowState, "/mujoco/lowstate", self._on_lowstate, 10)
        self.create_subscription(Float32MultiArray, "/mujoco/pos",
                                 self._on_mujoco_pos, 10)
        self._mj_world_xy_yaw = np.zeros(3, dtype=np.float32)
        self.create_subscription(Float32MultiArray, "/rl/depth_latent",
                                 self._on_depth_latent, 10)
        # Always subscribe to /sim_truth/base_velocity and /base_velocity if
        # they're being published — even when use_sim_truth_vel is False, the
        # logger uses them. Cheap to ignore the messages when not needed.
        self.create_subscription(Twist, "/sim_truth/base_velocity",
                                 self._on_sim_truth_vel, 10)
        self.create_subscription(Twist, "/base_velocity",
                                 self._on_ekf_vel, 10)
        if self.use_sim_truth_vel:
            self.get_logger().info(
                "use_sim_truth_vel ON: priv_explicit[0:3] will be filled from "
                "/sim_truth/base_velocity, learned Estimator output ignored."
            )
        if self.zero_student_velocity_obs:
            self.get_logger().info(
                "zero_student_velocity_obs ON: priv_explicit[0:3] forced to zero"
            )
        # FSM gate from low_level_ctrl: only run the actor + publish target_pos
        # once the robot is standing AND LB+RB engage has been latched. Matches
        # real deploy where the policy doesn't run during lay/stand bring-up.
        self._policy_engaged = False
        self.create_subscription(Bool, "/policy_engaged",
                                 self._on_engaged, 10)
        self.target_pub = self.create_publisher(Float32MultiArray, "/rl/target_pos", 10)
        self.last_action_pub = self.create_publisher(Float32MultiArray, "/rl/last_action", 10)
        self.contact_filt_pub = self.create_publisher(Float32MultiArray, "/rl/contact_filt", 10)
        # Live-viewer diagnostics, published once per tick AFTER the target:
        # [lowstate_age_ms, wait_ms, achieved_ms, budget_ms] where achieved =
        # used-lowstate receipt -> target publish (the enforced constant).
        self.proprio_diag_pub = self.create_publisher(
            Float32MultiArray, "/rl/proprio_latency", 10)
        # Estimator's predicted body-frame linear velocity (m/s). priv_explicit[0:3]
        # is base_lin_vel * obs_scales.lin_vel(=2.0) — undo the scale here. The
        # remaining 6 dims of priv_explicit are zero pads in legged_robot.py:689-691.
        self.est_lin_vel_pub = self.create_publisher(Float32MultiArray, "/rl/est_lin_vel", 10)
        self._tick_count = 0

        self.timer = self.create_timer(1.0 / rate_hz, self._tick)
        self._last_yaw_cmd = (0.0, 0.0)
        yaw_mode = ("use_true_yaw" if use_true_yaw
                    else "zero_yaw_pred" if zero_yaw_pred
                    else "encoder")
        self.get_logger().info(
            f"rl_distill_policy ready (action_delay={action_delay}, lin_vel_max={lin_vel_max}, "
            f"yaw_mode={yaw_mode})"
        )

    # ------------------------------------------------------------------
    def _on_lowstate(self, msg: LowState):
        receive_mono_ns = host_monotonic_ns()
        receive_wall_ns = time.time_ns()
        self.latest_lowstate = msg
        self.latest_lowstate_receive_mono_ns = receive_mono_ns
        self.have_lowstate = True
        if self.lowstate_latency_log_fp is not None:
            self.lowstate_latency_log_fp.write(
                f"{self.lowstate_receive_seq}\t{int(msg.tick)}\t"
                f"{receive_mono_ns}\t{receive_wall_ns}\n"
            )
            if self.lowstate_receive_seq % 500 == 499:
                self.lowstate_latency_log_fp.flush()
        self.lowstate_receive_seq = (self.lowstate_receive_seq + 1) & 0xffffffff

    def _on_engaged(self, msg: Bool):
        was = self._policy_engaged
        self._policy_engaged = bool(msg.data)
        if self._policy_engaged and not was:
            # Match an Isaac episode reset. No pre-engagement recurrent,
            # observation-history, action-history, or contact state may leak
            # into the first policy observation.
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
            initial_action = Float32MultiArray()
            initial_action.data = np.zeros(NUM_ACTIONS, dtype=np.float32).tolist()
            self.last_action_pub.publish(initial_action)
            self._publish_contact_filt()
            self.get_logger().info(
                "/policy_engaged=True — reset actor episode state; RL inference on")
        elif was and not self._policy_engaged:
            self.get_logger().info("/policy_engaged=False — RL inference paused")

    def _on_sim_truth_vel(self, msg: Twist):
        self._sim_truth_vel_body[0] = msg.linear.x
        self._sim_truth_vel_body[1] = msg.linear.y
        self._sim_truth_vel_body[2] = msg.linear.z
        if not self._have_sim_truth_vel:
            self._have_sim_truth_vel = True
            self.get_logger().info("first /sim_truth/base_velocity received")

    def _on_mujoco_pos(self, msg: Float32MultiArray):
        if len(msg.data) < 7:
            return
        x = float(msg.data[0]); y = float(msg.data[1])
        q = np.asarray(msg.data[3:7], dtype=np.float64)
        _, _, yaw = mj_quat_to_roll_pitch_yaw(q)
        self._mj_world_xy_yaw[0] = x
        self._mj_world_xy_yaw[1] = y
        self._mj_world_xy_yaw[2] = yaw

    def _on_ekf_vel(self, msg: Twist):
        self._ekf_vel_body[0] = msg.linear.x
        self._ekf_vel_body[1] = msg.linear.y
        self._ekf_vel_body[2] = msg.linear.z
        if not self._have_ekf_vel:
            self._have_ekf_vel = True
            self.get_logger().info("first /base_velocity (EKF) received")

    def _on_depth_latent(self, msg: Float32MultiArray):
        receive_mono_ns = host_monotonic_ns()
        if not self._policy_engaged:
            return
        # Accept either the 32-D no-yaw-head latent (L00-flatstep student) or the
        # 34-D latent+yaw_pred emitted when the encoder has the yaw head
        # (EP_FEED_PRED_YAW=1 — the OD cliffside/flyball student). The 2-D yaw
        # tail, when present, is scaled by YAW_PRED_MULTIPLIER into prop[6:8].
        n = len(msg.data)
        if n != DEPTH_LATENT_DIM and n != DEPTH_LATENT_DIM + 2:
            return
        d = np.asarray(msg.data, dtype=np.float32)
        if not np.isfinite(d).all():
            # Source-level diagnostics: a NaN/inf latent would flow through the
            # actor into 12 non-finite targets (low_level_ctrl drops those, but
            # this log points at the encoder instead of 12 mystery NaNs there).
            # Keep the last good latent; if this is the FIRST latent the policy
            # simply stays blocked on have_depth_latent.
            self.get_logger().error(
                "non-finite /rl/depth_latent — keeping previous latent",
                throttle_duration_sec=1.0)
            return
        self.cached_depth_latent = torch.from_numpy(d[:DEPTH_LATENT_DIM]).unsqueeze(0)
        self.latest_depth_seq = int(msg.layout.data_offset)
        self.latest_latent_receive_mono_ns = receive_mono_ns
        if n == DEPTH_LATENT_DIM + 2:
            self.cached_yaw_pred = torch.from_numpy(
                d[DEPTH_LATENT_DIM:DEPTH_LATENT_DIM + 2]).unsqueeze(0)
            self.have_yaw_pred = True
        if not self.have_depth_latent:
            self.have_depth_latent = True
            self.get_logger().info(
                f"first /rl/depth_latent received (dim={n}); policy unblocked")

    def _read_lin_vel_x(self) -> float:
        if not np.isnan(self.force_lin_vel_x):
            return float(self.force_lin_vel_x)
        if not (self.xbox.axes and self.xbox.buttons):
            return 0.0
        lb_rb = self.xbox.is_pressed()
        if not (lb_rb[0] and lb_rb[1]):
            return 0.0
        lx, _ly = self.xbox.get_left_stick()
        if self.binary_vel:
            # bang-bang: forward push past the deadzone -> full lin_vel_max, else 0
            # (backward / centred stick -> 0, no reverse).
            return self.lin_vel_max if float(lx) > self.vel_deadzone else 0.0
        return self.lin_vel_max * float(lx)

    def _quit_combo(self) -> bool:
        # Threshold, not == -1.0: analog triggers on some pads never report
        # exactly -1.0 when floored (matches the low_level_ctrl e-stop check).
        a = self.xbox.axes
        return bool(a) and len(a) >= 6 and a[2] < -0.9 and a[5] < -0.9

    def _publish_contact_filt(self) -> np.ndarray:
        """Advance and publish Isaac's one-policy-step contact history."""
        ff = np.asarray(self.latest_lowstate.foot_force, dtype=np.float64)
        if ff.size == 4:
            contact_filt, raw_now = compute_contact_filt(
                ff, self.last_contact_raw, threshold_n=self.contact_threshold,
            )
            self.last_contact_raw = raw_now
        else:
            contact_filt = np.zeros(4)
        self._last_raw_foot_force = ff if ff.size == 4 else np.zeros(4)
        contact_msg = Float32MultiArray()
        contact_msg.data = contact_filt.astype(np.float32).tolist()
        self.contact_filt_pub.publish(contact_msg)
        return contact_filt

    # ------------------------------------------------------------------
    def _tick(self):
        tick_start_ns = host_monotonic_ns()
        if self._quit_combo():
            sys.exit(0)
        if not self.have_lowstate:
            return
        if not self._policy_engaged:
            return

        # ---- stale-input fail-safe. Guards run only while engaged (the two
        # returns above), so lay-down/stand bring-up — where C++ owns the
        # joints and the policy is idle — is unaffected. Ceasing /rl/target_pos
        # is the safe action: low_level_ctrl's rl_stall_timeout then latches a
        # settle-to-stand.
        now_ns = host_monotonic_ns()
        if self.lowstate_max_age_s > 0:
            # /lowstate is ~500 Hz (real robot and mujoco_simulator's 2 ms
            # timer alike) — 0.2 s means ~100 consecutive missed messages,
            # unreachable in normal operation.
            lowstate_age_s = (now_ns - self.latest_lowstate_receive_mono_ns) * 1e-9
            if lowstate_age_s > self.lowstate_max_age_s:
                self._stale_stopped = True
                self.get_logger().error(
                    f"STALE /lowstate ({lowstate_age_s:.2f} s > "
                    f"{self.lowstate_max_age_s:g} s) — ceasing /rl/target_pos "
                    "so low_level_ctrl's rl_stall_timeout settles the robot",
                    throttle_duration_sec=1.0)
                return

        # Publish the actor-owned 50 Hz contact filter before waiting for a
        # latent. This gives the encoder its first 20 ms history sample without
        # creating an actor-waits-for-encoder / encoder-waits-for-actor deadlock.
        # (Kept alive during a LATENT stall too, so a recovering camera/encoder
        # finds fresh shared contact and can restart the latent stream.)
        contact_filt = self._publish_contact_filt()

        if not self.have_depth_latent:
            # block startup: avoid publishing target_pos with cached_depth_latent=zeros,
            # which would unfreeze the simulator (waiting on first /mujoco/lowcmd) under
            # garbage actions and tip the robot before the encoder catches up.
            return

        if self.latent_max_age_s > 0:
            # Armed only once the first post-engagement latent has arrived
            # (have_depth_latent is reset False on every engage, checked
            # above) — the encoder gates GRU forwards on /policy_engaged
            # (gate_on_engage default 1), so latents legitimately don't flow
            # before engagement and must not trip this guard. Latents are
            # nominally 10 Hz with an 80-160 ms enforced capture->publish
            # latency; age is measured from RECEIPT, so normal inter-arrival
            # is ~100 ms — 0.6 s means ~6 consecutive missed latents.
            latent_age_s = (now_ns - self.latest_latent_receive_mono_ns) * 1e-9
            if latent_age_s > self.latent_max_age_s:
                self._stale_stopped = True
                self.get_logger().error(
                    f"STALE /rl/depth_latent ({latent_age_s:.2f} s > "
                    f"{self.latent_max_age_s:g} s) — ceasing /rl/target_pos "
                    "so low_level_ctrl's rl_stall_timeout settles the robot",
                    throttle_duration_sec=1.0)
                return
        if self._stale_stopped:
            self._stale_stopped = False
            self.get_logger().info(
                "stale inputs recovered — resuming /rl/target_pos publication")

        inference_start_ns = host_monotonic_ns()
        used_lowstate_tick = int(self.latest_lowstate.tick)
        used_lowstate_receive_ns = self.latest_lowstate_receive_mono_ns
        used_depth_seq = self.latest_depth_seq
        used_latent_receive_ns = self.latest_latent_receive_mono_ns

        # 1. proprio from /lowstate. motor_state, foot_force, and our target_pos
        # publish are all in MJ/SDK order (FR,FL,RR,RL × hip,thigh,calf), which
        # is also the order training's proprio was assembled in after
        # legged_robot.reindex(). No permutation needed.
        ms = self.latest_lowstate.motor_state
        qj_policy = np.array([ms[i].q for i in range(NUM_ACTIONS)], dtype=np.float64)
        dqj_policy = np.array([ms[i].dq for i in range(NUM_ACTIONS)], dtype=np.float64)

        quat = self.latest_lowstate.imu_state.quaternion
        roll, pitch, _ = mj_quat_to_roll_pitch_yaw(np.asarray(quat, dtype=np.float64))
        ang_vel_body = np.asarray(self.latest_lowstate.imu_state.gyroscope, dtype=np.float64)

        last_action_pol = self.last_action.numpy().reshape(-1)
        prop = build_prop(
            qj_policy=qj_policy, dqj_policy=dqj_policy,
            ang_vel_body=ang_vel_body, roll=roll, pitch=pitch,
            last_action_pol=last_action_pol,
            contact_filt=contact_filt,
            lin_vel_x=self._read_lin_vel_x(),
            env_class_neq_17=0.0 if self.env_class_eq_17 else 1.0,
            env_class_eq_17=1.0 if self.env_class_eq_17 else 0.0,
        )

        prop_t = torch.from_numpy(prop).float().unsqueeze(0)

        # 2. write yaw_pred into the live frame (yaw mask kept zero in history)
        prop_hist_frame = prop_t.clone()
        prop_hist_frame[:, 6:8] = 0.0
        if self.history_ring is None:
            # Match Isaac legged_robot.py:466 — reset obs_history_buf to zeros,
            # not repeated copies of the first frame. The repeat() variant
            # contaminates the first ~10 ticks with off-distribution data.
            self.history_ring = torch.zeros(1, HISTORY_LEN, N_PROPRIO)
        else:
            self.history_ring = torch.cat(
                [self.history_ring[:, 1:], prop_hist_frame.unsqueeze(1)], dim=1
            )
        if self.use_true_yaw:
            # DIAGNOSTIC: inject ground-truth-derived delta_yaw (off the
            # L00-flatstep student's training distribution — see __init__).
            # Compute current yaw from the lowstate quaternion (qpos[3:7]
            # broadcast through /lowstate). Target heading = 0 (+x corridor).
            q = np.asarray(self.latest_lowstate.imu_state.quaternion,
                           dtype=np.float64)
            _, _, cur_yaw = mj_quat_to_roll_pitch_yaw(q)
            d = -cur_yaw
            # wrap to [-pi, pi]
            d = (d + np.pi) % (2 * np.pi) - np.pi
            prop_t[:, 6] = float(d)
            prop_t[:, 7] = float(d)
        elif self.have_yaw_pred and not self.zero_yaw_pred:
            # Yaw-head student (EP_FEED_PRED_YAW=1): feed the encoder's regressed
            # heading into the live frame, scaled exactly as in training
            # (obs_student[:, 6:8] = 1.5 * yaw_pred). History stays yaw-masked.
            prop_t[:, 6:8] = YAW_PRED_MULTIPLIER * self.cached_yaw_pred
        else:
            # No yaw head (32-D latent) or --zero_yaw_pred: heading slots held at
            # 0 to match the no-yaw student's training observation.
            prop_t[:, 6:8] = 0.0
        # Capture what was actually written so HUD and vel_log reflect truth.
        self._last_yaw_cmd = (float(prop_t[0, 6]), float(prop_t[0, 7]))

        # Optional: log the full 53-D proprio frame the actor will see.
        if self.proprio_log_fp is not None and self.proprio_log_ticks_remaining > 0:
            import time as _time
            now = _time.time()
            if self.proprio_log_t0 is None:
                self.proprio_log_t0 = now
            tick_idx = (-self.proprio_log_ticks_remaining)  # negative=count-down; replace below
            # Cleaner: derive tick index from how many we've already logged.
            arr = prop_t.squeeze(0).numpy()
            written = (100 - self.proprio_log_ticks_remaining) if False else None
            # Compute count from initial budget vs remaining (simpler):
            # We store the initial value once.
            # Fall through to a simple counter using an attribute.
            if not hasattr(self, "_proprio_log_written"):
                self._proprio_log_written = 0
            i = self._proprio_log_written
            t_s = now - self.proprio_log_t0
            raw_ff = getattr(self, "_last_raw_foot_force", np.zeros(4))
            extra = "\t".join(f"{v:+.6f}" for v in raw_ff)
            self.proprio_log_fp.write(
                f"{i}\t{t_s:.4f}\t" + "\t".join(f"{v:+.6f}" for v in arr) + "\t" + extra + "\n"
            )
            self._proprio_log_written += 1
            self.proprio_log_ticks_remaining -= 1
            if self.proprio_log_ticks_remaining == 0:
                self.proprio_log_fp.flush()
                self.proprio_log_fp.close()
                self.proprio_log_fp = None
                self.get_logger().info(
                    f"proprio_log: wrote {self._proprio_log_written} ticks, file closed"
                )

        # 3. estimator + actor
        with torch.no_grad():
            priv_expl = self.estimator(prop_t)
        # Capture the learned-estimator prediction BEFORE any override so the
        # 3-way velocity log compares the actual estimator output (not the
        # sim-truth swap value).
        learned_est_lin_vel_body = priv_expl[0, 0:3].cpu().numpy() / 2.0
        if self.zero_student_velocity_obs:
            priv_expl = priv_expl.clone()
            priv_expl[0, 0:3] = 0.0
        elif self.use_sim_truth_vel:
            # Replace base_lin_vel slice with body-frame MuJoCo qvel * lin_vel
            # obs scale (=2.0). The remaining 6 dims of priv_explicit are zero
            # pads in training (legged_robot.py:689-691) — the estimator's
            # output for those dims gets overwritten too to match training.
            sim_vel = torch.from_numpy(self._sim_truth_vel_body).to(priv_expl.dtype)
            priv_expl = torch.zeros_like(priv_expl)
            priv_expl[0, 0:3] = sim_vel * 2.0
        # Publish the estimator's body-frame lin_vel (m/s) for debugging.
        est_lin_vel_body = priv_expl[0, 0:3].cpu().numpy() / 2.0  # undo obs_scales.lin_vel

        # 3-way velocity log (GT vs EKF vs learned estimator).
        if self.vel_log_fp is not None and self.vel_log_ticks_remaining > 0:
            import time as _time
            now = _time.time()
            if self._vel_log_t0 is None:
                self._vel_log_t0 = now
            t_s = now - self._vel_log_t0
            if not hasattr(self, "_vel_log_written"):
                self._vel_log_written = 0
            i = self._vel_log_written
            cmd_vx = self._read_lin_vel_x() if hasattr(self, "_read_lin_vel_x") else 0.0
            gt  = self._sim_truth_vel_body
            ekf = self._ekf_vel_body
            lr  = learned_est_lin_vel_body
            w = self._mj_world_xy_yaw
            yc0, yc1 = self._last_yaw_cmd
            self.vel_log_fp.write(
                f"{i}\t{t_s:.4f}\t{cmd_vx:+.4f}"
                f"\t{gt[0]:+.4f}\t{gt[1]:+.4f}\t{gt[2]:+.4f}"
                f"\t{ekf[0]:+.4f}\t{ekf[1]:+.4f}\t{ekf[2]:+.4f}"
                f"\t{lr[0]:+.4f}\t{lr[1]:+.4f}\t{lr[2]:+.4f}"
                f"\t{w[0]:+.4f}\t{w[1]:+.4f}\t{np.degrees(w[2]):+.4f}"
                f"\t{np.degrees(yc0):+.4f}\t{np.degrees(yc1):+.4f}\n"
            )
            self._vel_log_written += 1
            self.vel_log_ticks_remaining -= 1
            if self.vel_log_ticks_remaining == 0:
                self.vel_log_fp.flush(); self.vel_log_fp.close()
                self.vel_log_fp = None
                self.get_logger().info(
                    f"vel_log: wrote {self._vel_log_written} ticks, file closed"
                )
        elv_msg = Float32MultiArray()
        elv_msg.data = est_lin_vel_body.astype(np.float32).tolist()
        self.est_lin_vel_pub.publish(elv_msg)
        self._tick_count += 1
        if self._tick_count % 50 == 0:
            # yaw_cmd reflects what was written into prop[6:8]: 0 for the no-yaw
            # student, 1.5 * yaw_pred for a yaw-head student (EP_FEED_PRED_YAW=1).
            yc0, yc1 = self._last_yaw_cmd
            # cmd_vx is the exact forward velocity command fed to the policy as
            # proprio[10] this tick: lin_vel_max * stick while LB+RB held, else 0.
            cmd_vx = self._read_lin_vel_x()
            self.get_logger().info(
                f"[cmd] vx={cmd_vx:+.2f} m/s  |  "
                f"[est] vx={est_lin_vel_body[0]:+.2f} vy={est_lin_vel_body[1]:+.2f} "
                f"vz={est_lin_vel_body[2]:+.2f}  yaw_cmd=({np.degrees(yc0):+6.1f}°, "
                f"{np.degrees(yc1):+6.1f}°)  |dlat|={float(self.cached_depth_latent.norm()):.2f}"
            )
        obs = assemble_actor_obs(prop_t, priv_expl, self.history_ring)
        with torch.no_grad():
            action_raw = self.actor(
                obs, hist_encoding=True, eval=False,
                scandots_latent=self.cached_depth_latent,
            )
        action_clipped = (action_raw.clamp(-CLIP_ACTIONS, CLIP_ACTIONS)
                          if self.clip_applied_action else action_raw)
        # MATCH TRAINING: training stores the *raw* actor output in
        # action_history_buf (legged_robot.py:124) and feeds that — unclipped —
        # back into next-step proprio[37:49]. Storing the clipped value here
        # would shrink the feedback signal vs training.
        self.last_action = action_raw.detach()

        # 4. action delay buffer
        self.action_buf.append(action_clipped.detach())
        if self.action_delay:
            action_to_apply = self.action_buf[-1 - ACTION_DELAY_VIEW]
        else:
            action_to_apply = action_clipped

        # 5. target dof. The actor outputs in raw order, which == MJ/SDK order
        # because legged_robot.step() applies reindex() before motors see the
        # action. So target_dof_unitree[i] goes straight to motor i.
        action_pol = action_to_apply.squeeze(0).numpy()
        target_dof_unitree = DEFAULT_DOF_POLICY + ACTION_SCALE * action_pol
        inference_end_ns = host_monotonic_ns()

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

        # 6. Optional proprio latency enforcement (mirrors depth 80 ms budget).
        # EP paper §4.1 pads from *receiving* the observation, so the
        # reference is the receipt of the lowstate this action was computed
        # from — NOT the tick start. This also absorbs the 1-4 ms
        # lowstate-age-at-tick jitter into the constant: receipt -> publish
        # is proprio_latency_s regardless of where in the DDS cadence the
        # 50 Hz timer fired. A lowstate already older than the budget (rare
        # host stall) gives _wait <= 0 -> publish immediately, same as before.
        sleep_start_ns = host_monotonic_ns()
        _wait = 0.0
        if self.proprio_latency_s > 0.0:
            _elapsed = (sleep_start_ns - used_lowstate_receive_ns) * 1e-9
            _wait = self.proprio_latency_s - _elapsed
            if 0.0 < _wait < 0.5:
                time.sleep(_wait)
        sleep_end_ns = host_monotonic_ns()

        out = Float32MultiArray()
        out.data = target_dof_unitree.astype(np.float32).tolist()
        out.layout.data_offset = self._tick_count
        publish_start_ns = host_monotonic_ns()
        self.target_pub.publish(out)
        publish_end_ns = host_monotonic_ns()
        if self.latency_log_fp is not None:
            self.latency_log_fp.write(
                f"{self._tick_count}\t{used_lowstate_tick}\t{used_lowstate_receive_ns}\t"
                f"{used_depth_seq}\t{used_latent_receive_ns}\t{tick_start_ns}\t"
                f"{inference_start_ns}\t{inference_end_ns}\t{sleep_start_ns}\t"
                f"{sleep_end_ns}\t{publish_start_ns}\t{publish_end_ns}\n"
            )

        la = Float32MultiArray()
        action_feedback = action_raw if self.publish_raw_last_action else action_clipped
        la.data = action_feedback.squeeze(0).numpy().astype(np.float32).tolist()
        self.last_action_pub.publish(la)

        diag = Float32MultiArray()
        diag.data = [
            (inference_start_ns - used_lowstate_receive_ns) * 1e-6,
            max(_wait, 0.0) * 1e3,
            (publish_end_ns - used_lowstate_receive_ns) * 1e-6,
            self.proprio_latency_s * 1e3,
        ]
        self.proprio_diag_pub.publish(diag)

    def destroy_node(self):
        for attr in ("action_log_fp", "proprio_log_fp", "vel_log_fp",
                     "latency_log_fp", "lowstate_latency_log_fp"):
            fp = getattr(self, attr, None)
            if fp is not None:
                fp.close()
                setattr(self, attr, None)
        return super().destroy_node()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--ep_root", type=Path,
                    default=_EP_ROOT_DEFAULT)
    ap.add_argument("--action_delay", type=int, choices=[0, 1], default=0,
                    help="0=off (default), 1=on. Off: T000 trained with delay=0 and "
                         "paper §4.1 enforces no artificial action delay; real-robot "
                         "transport lag is physical, not added in software.")
    ap.add_argument("--proprio_latency_s", type=float, default=PROPRIO_LATENCY_S,
                    help="Sleep after inference to enforce fixed sensor→publish latency "
                         "(seconds). Default 0.016 = paper §4.1 proprioception latency; "
                         "pass 0.0 to disable.")
    ap.add_argument("--lin_vel_max", type=float, default=0.7)
    ap.add_argument("--force_lin_vel_x", type=float, default=float("nan"),
                    help="bypass joystick gate; constant lin_vel_x in m/s")
    ap.add_argument("--binary_vel", type=int, choices=[0, 1], default=0,
                    help="If 1, command a CONSTANT lin_vel_max whenever the left stick is "
                         "pushed forward past --vel_deadzone (bang-bang), instead of scaling "
                         "the command with stick deflection. 0 m/s otherwise (incl. backward).")
    ap.add_argument("--vel_deadzone", type=float, default=0.2,
                    help="Forward-stick fraction (0..1) above which --binary_vel commands "
                         "lin_vel_max. Default 0.2.")
    ap.add_argument("--proprio_log", type=Path, default=None,
                    help="If set, write every per-tick 53-D proprio frame to "
                         "this txt file (header row + tab-separated rows of "
                         "[tick, t_s, 53 values]). Stops after --proprio_log_ticks.")
    ap.add_argument("--proprio_log_ticks", type=int, default=100,
                    help="Number of ticks to record before closing the proprio log.")
    ap.add_argument("--use_sim_truth_vel", type=int, choices=[0, 1], default=0,
                    help="If 1, fill priv_explicit[0:3] from /sim_truth/base_velocity "
                         "(MuJoCo body-frame qvel) instead of the learned Estimator. "
                         "Used to test whether velocity-estimate noise is the sim2sim bottleneck.")
    ap.add_argument("--zero_student_velocity_obs", type=int, choices=[0, 1], default=0,
                    help="If 1, force actor priv_explicit[0:3] to zero. Use only for "
                         "students trained with zero_student_velocity_obs=True.")
    ap.add_argument("--vel_log", type=Path, default=None,
                    help="If set, write per-tick (t, cmd_vx, gt, ekf, est) TSV to this path. "
                         "Captures /sim_truth/base_velocity, /base_velocity (EKF), and the "
                         "learned estimator's prediction side-by-side for sim2sim comparison.")
    ap.add_argument("--vel_log_ticks", type=int, default=2000,
                    help="Number of policy ticks to log before closing the velocity log "
                         "(2000 = 40 s at 50 Hz, covers the full lay2stand2lay window).")
    ap.add_argument("--zero_yaw_pred", type=int, choices=[0, 1], default=0,
                    help="If 1, force prop[6:8] (the depth-encoder yaw_pred slice) to 0 "
                         "every tick. Diagnostic for cases where the depth latent / yaw "
                         "head is suspected of injecting phantom heading commands.")
    ap.add_argument("--use_true_yaw", type=int, choices=[0, 1], default=0,
                    help="If 1, overwrite prop[6:8] with the true delta_yaw computed "
                         "from the MuJoCo base quaternion (target heading = 0, +x "
                         "corridor). Mutually exclusive with --zero_yaw_pred.")
    ap.add_argument("--env_class_eq_17", type=int, choices=[0, 1], default=0,
                    help="If 1, set prop[11:13] indicator to (0, 1) — matches "
                         "parkour_flat training (env_class==17, used by go2_flat / L00-* "
                         "exptids). Default 0 leaves the (1, 0) indicator used by every "
                         "other env_class (parkour, wall-turn, etc.).")
    ap.add_argument("--contact_threshold", type=str, default=str(CONTACT_FORCE_THRESHOLD_N),
                    help="Foot-force threshold for proprio[49:53]: one global value OR "
                         "four comma-separated per-foot values in FR,FL,RR,RL order "
                         "(e.g. 10,20,20,17 for the real robot; see distill_common "
                         "CONTACT_FORCE_THRESHOLD_PER_FOOT_REAL). MUST match "
                         "depth_encoder_node --contact_threshold or the encoder and "
                         "actor will disagree on the gait-phase bit. Default 1.0 (L00-* "
                         "flat); raise to 10.0 for parkour wall checkpoints.")
    ap.add_argument("--publish_raw_last_action", type=int, choices=[0, 1], default=1,
                    help="1 publishes raw actor output to the encoder, matching training (default).")
    ap.add_argument("--clip_applied_action", type=int, choices=[0, 1], default=1,
                    help="1 clamps the motor-path raw action to +/-4.8 as in Isaac; "
                         "0 is a simulation-only ablation.")
    ap.add_argument("--action_log", type=Path, default=None,
                    help="Optional line-buffered TSV of contacts, actions, and targets.")
    ap.add_argument("--latency_log", type=Path, default=None,
                    help="Optional correlated actor and lowstate timing TSVs.")
    ap.add_argument("--rate_hz", type=float, default=50.0,
                    help="Wall-clock policy timer rate. Default 50 Hz.")
    ap.add_argument("--lowstate_max_age_s", type=float, default=0.2,
                    help="Fail-safe: cease /rl/target_pos when the latest /lowstate is "
                         "older than this (s), letting low_level_ctrl's rl_stall_timeout "
                         "settle the robot. /lowstate is ~500 Hz, so the default 0.2 "
                         "(~100 missed messages) cannot false-trigger. <= 0 disables.")
    ap.add_argument("--latent_max_age_s", type=float, default=0.6,
                    help="Fail-safe: cease /rl/target_pos when the latest /rl/depth_latent "
                         "is older than this (s). Latents are 10 Hz (with 80-160 ms "
                         "enforced latency), so the default 0.6 (~6 missed latents) cannot "
                         "false-trigger; only armed after the first post-engagement latent "
                         "(the encoder gates on /policy_engaged). <= 0 disables.")
    args, _ = ap.parse_known_args()

    rclpy.init()
    proprio_log_path = args.proprio_log
    if proprio_log_path is not None and str(proprio_log_path) in ("", "."):
        proprio_log_path = None
    vel_log_path = args.vel_log
    if vel_log_path is not None and str(vel_log_path) in ("", "."):
        vel_log_path = None
    action_log_path = args.action_log
    if action_log_path is not None and str(action_log_path) in ("", "."):
        action_log_path = None
    latency_log_path = args.latency_log
    if latency_log_path is not None and str(latency_log_path) in ("", "."):
        latency_log_path = None
    node = RLDistillPolicy(args.ckpt, args.ep_root,
                            args.action_delay, args.lin_vel_max,
                            args.force_lin_vel_x,
                            binary_vel=bool(args.binary_vel),
                            vel_deadzone=args.vel_deadzone,
                            proprio_log_path=proprio_log_path,
                            proprio_log_ticks=args.proprio_log_ticks,
                            use_sim_truth_vel=bool(args.use_sim_truth_vel),
                            zero_student_velocity_obs=bool(args.zero_student_velocity_obs),
                            vel_log_path=vel_log_path,
                            vel_log_ticks=args.vel_log_ticks,
                            zero_yaw_pred=bool(args.zero_yaw_pred),
                            use_true_yaw=bool(args.use_true_yaw),
                            env_class_eq_17=bool(args.env_class_eq_17),
                            contact_threshold=args.contact_threshold,
                            proprio_latency_s=args.proprio_latency_s,
                            publish_raw_last_action=bool(args.publish_raw_last_action),
                            clip_applied_action=bool(args.clip_applied_action),
                            action_log_path=action_log_path,
                            latency_log_path=latency_log_path,
                            rate_hz=args.rate_hz,
                            lowstate_max_age_s=args.lowstate_max_age_s,
                            latent_max_age_s=args.latent_max_age_s)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
