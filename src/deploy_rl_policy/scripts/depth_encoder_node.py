#!/usr/bin/env python3
"""Distillation-policy depth encoder.

Subscribes /rl/depth_image (preprocessed 58x87 32FC1) and /lowstate (proprio).
Runs RecurrentDepthBackbone forward at the depth-image arrival rate (~10 Hz),
enforces the paper's 80 ms total latency from depth-capture to latent-publish,
publishes std_msgs/Float32MultiArray (length 32: scan latent only) on
/rl/depth_latent.

Architecture note (L00-flatstep student): the yaw/heading predictor head was
removed from RecurrentDepthBackbone.output_mlp (Linear(512, 32+2) -> Linear(512,
32)). The encoder no longer estimates direction-to-goal; the actor's heading
slots prop[6:8] are zeroed at deploy to match the student's training obs.

The actor's proprio[37:49] (last_action) is sourced from /rl/last_action
which the policy node publishes every 50 Hz tick. lin_vel_x is sourced from
/joy through xbox_command.XboxController. Reading the same /joy stream is NOT
enough on its own: the command *mode* must also match. rl_distill_policy grew
--binary_vel/--vel_deadzone (bang-bang: forward stick past the deadzone ->
constant lin_vel_max, else 0), so this node takes the same flags and mirrors
that logic verbatim in _read_lin_vel_x. Run both nodes with identical
--binary_vel / --vel_deadzone / --lin_vel_max / --force_lin_vel_x or the
encoder's proprio[10] desyncs from the actor's (e.g. half stick: actor 1.0,
encoder 0.5) — same failure class as the env_class_eq_17 / contact-threshold
desyncs described below.
"""
from pathlib import Path
import argparse
import os
import time

import numpy as np
import rclpy
import torch
torch.set_num_threads(1)  # tiny MLPs; multi-thread overhead dominates (see rl_distill_policy.py)
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32MultiArray
from unitree_go.msg import LowState

from distill_common import (
    add_extreme_parkour_to_path, load_checkpoint,
    mj_quat_to_roll_pitch_yaw, build_prop, compute_contact_filt,
    MJ_TO_POLICY, NUM_ACTIONS, N_PROPRIO, DEPTH_LATENT_DIM,
    DEPTH_TARGET_H, DEPTH_TARGET_W, DEPTH_LATENCY_S,
    CONTACT_FORCE_THRESHOLD_N, parse_contact_threshold,
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

def image_msg_to_norm_depth(msg: Image) -> np.ndarray:
    h, w = msg.height, msg.width
    if msg.encoding != "32FC1":
        raise ValueError(f"depth_encoder expects 32FC1, got {msg.encoding}")
    return np.frombuffer(msg.data, dtype=np.float32).reshape(h, w)


class DepthEncoderNode(Node):
    def __init__(self, ckpt_path: Path, ep_root: Path,
                 enforce_latency_s: float, lin_vel_max: float,
                 force_lin_vel_x: float = float("nan"),
                 binary_vel: bool = False,
                 vel_deadzone: float = 0.2,
                 contact_threshold: float = CONTACT_FORCE_THRESHOLD_N,
                 env_class_eq_17: bool = False,
                 gate_on_engage: bool = True,
                 reset_on_engage: bool = True,
                 use_shared_contact: bool = True,
                 trace_log_path: Path = None,
                 latency_log_path: Path = None):
        super().__init__("depth_encoder_node")
        self.enforce_latency_s = enforce_latency_s
        self.lin_vel_max = lin_vel_max
        self.force_lin_vel_x = force_lin_vel_x
        # Bang-bang velocity command, mirroring rl_distill_policy._read_lin_vel_x
        # exactly. MUST match rl_distill_policy --binary_vel/--vel_deadzone: the
        # encoder embeds its own proprio copy into the depth latent, so a mode
        # mismatch silently desyncs prop[10] between the encoder forward and the
        # actor input (half stick: actor lin_vel_max, encoder lin_vel_max/2) —
        # same failure class as the env_class_eq_17 / contact-threshold desyncs.
        self.binary_vel = bool(binary_vel)
        self.vel_deadzone = vel_deadzone
        # scalar or length-4 per-foot list (FR,FL,RR,RL); MUST match the policy.
        self.contact_threshold = parse_contact_threshold(contact_threshold)
        # prop[11:13] env_class indicator. MUST match rl_distill_policy's
        # --env_class_eq_17: the encoder embeds its own proprio copy into the
        # depth latent, so a mismatch silently desyncs prop[11:13] between the
        # encoder forward and the actor input — same failure class as the
        # contact-threshold desync found 2026-05-23. Until 2026-06-06 this node
        # ignored the flag entirely and always packed (1, 0), wrong for every
        # L00-* / parkour_flat checkpoint.
        self.env_class_eq_17 = bool(env_class_eq_17)
        self.gate_on_engage = bool(gate_on_engage)
        self.reset_on_engage = bool(reset_on_engage)
        self.use_shared_contact = bool(use_shared_contact)
        # OR-filter previous-step raw contact, matching legged_robot.py:332-334
        # and the standalone harness. Without this the bit clears in the same
        # tick the foot leaves the ground, which under-represents the stance.
        self.last_contact_raw = np.zeros(4, dtype=np.float64)
        self.shared_contact_filt = np.zeros(4, dtype=np.float64)
        self.have_shared_contact = False
        self.trace_log_fp = None
        self.trace_tick = 0
        if trace_log_path is not None:
            self.trace_log_fp = open(trace_log_path, "w", buffering=1)
            self.trace_log_fp.write(
                "tick\tnow_s\tcapture_s\tengaged\t" +
                "\t".join(f"prop_{i}" for i in range(N_PROPRIO)) + "\t" +
                "\t".join(f"latent_{i}" for i in range(DEPTH_LATENT_DIM)) + "\n")
        self.latency_log_fp = None
        if latency_log_path is not None:
            self.latency_log_fp = open(latency_log_path, "w", buffering=1)
            self.latency_log_fp.write(
                "depth_seq\tsource_sec\tsource_nsec\treceive_mono_ns\treceive_wall_ns\t"
                "lowstate_tick\tlowstate_receive_mono_ns\tinference_start_mono_ns\t"
                "inference_end_mono_ns\twait_requested_ms\tsleep_start_mono_ns\t"
                "sleep_end_mono_ns\tpublish_start_mono_ns\tpublish_end_mono_ns\t"
                "clock_offset_ms\n"
            )

        # ----- module setup
        add_extreme_parkour_to_path(ep_root)
        from rsl_rl.modules.depth_backbone import (
            DepthOnlyFCBackbone58x87, RecurrentDepthBackbone,
        )

        self.get_logger().info(f"loading ckpt {ckpt_path}")
        ckpt = load_checkpoint(ckpt_path)
        backbone = DepthOnlyFCBackbone58x87(
            prop_dim=N_PROPRIO, scandots_output_dim=DEPTH_LATENT_DIM,
            hidden_state_dim=512, output_activation=None, num_frames=1,
        )
        self.encoder = RecurrentDepthBackbone(backbone, env_cfg=None)
        self.encoder.load_state_dict(ckpt["depth_encoder_state_dict"], strict=True)
        self.encoder.eval()
        # The RecurrentDepthBackbone output dim is set by EP_FEED_PRED_YAW at
        # construction: 32 (no yaw head) or 34 (32 latent + 2 yaw_pred). A 34-D
        # checkpoint only loads strict=True when EP_FEED_PRED_YAW=1. Publishing
        # is dim-agnostic (the full output goes out), so just log what we got.
        self._enc_out_dim = self.encoder.output_mlp[0].out_features
        self.get_logger().info(
            f"depth encoder out_dim={self._enc_out_dim} "
            f"(yaw_head={'on' if self._enc_out_dim == DEPTH_LATENT_DIM + 2 else 'off'}); "
            f"EP_FEED_PRED_YAW={os.environ.get('EP_FEED_PRED_YAW', '0')}")

        # ----- runtime state
        self.latest_lowstate: LowState = LowState()
        self.latest_lowstate_receive_mono_ns = 0
        self.last_action_pol = np.zeros(NUM_ACTIONS, dtype=np.float64)
        self.have_lowstate = False
        self.depth_seq = 0
        self.xbox = XboxController(self)

        # ----- topics
        self.create_subscription(LowState, "/lowstate", self._on_lowstate, 10)
        self.create_subscription(LowState, "/mujoco/lowstate", self._on_lowstate, 10)
        self.create_subscription(Float32MultiArray, "/rl/last_action", self._on_last_action, 10)
        self.create_subscription(Float32MultiArray, "/rl/contact_filt",
                                 self._on_shared_contact, 10)
        self.create_subscription(Image, "/rl/depth_image", self._on_depth, 10)
        self.pub = self.create_publisher(Float32MultiArray, "/rl/depth_latent", 10)
        # Jetson-Mac clock offset from depth_camera_node's passive drift
        # tracker (real path only; the topic never exists in sim, so the
        # correction stays 0 there and behavior is unchanged). The 80 ms
        # budget below compares the camera header stamp (Jetson clock) to the
        # Mac clock; subtracting this offset keeps the enforced latency
        # constant under the ~1 ms/min free-running drift measured 2026-07-24.
        self._clock_offset_s = 0.0
        self._clock_offset_ready = False
        self.create_subscription(Float32MultiArray, "/rl/depth_clock_offset",
                                 self._on_clock_offset, 10)
        # Diagnostics for the live viewer:
        # [elapsed_ms, wait_ms, achieved_ms, budget_ms, late_count, offset_ms]
        self._late_count = 0
        self.latency_diag_pub = self.create_publisher(
            Float32MultiArray, "/rl/depth_latency", 10)
        # FSM gate from low_level_ctrl: skip the depth encoder forward while
        # the robot is laying down or standing up so /rl/depth_latent (and the
        # yaw_pred carried in it) doesn't get updated until the policy is live.
        self._policy_engaged = False
        self.create_subscription(Bool, "/policy_engaged", self._on_engaged, 10)

        vel_mode = ("force_lin_vel_x" if not np.isnan(self.force_lin_vel_x)
                    else "binary" if self.binary_vel else "proportional")
        self.get_logger().info(
            f"cmd_vx mode={vel_mode} (binary_vel={int(self.binary_vel)}, "
            f"vel_deadzone={self.vel_deadzone}, lin_vel_max={self.lin_vel_max}, "
            f"force_lin_vel_x={self.force_lin_vel_x}) — MUST match rl_distill_policy"
        )
        self.get_logger().info(
            f"depth_encoder_node ready (latency budget={enforce_latency_s*1000:.0f} ms)"
        )

    def _on_lowstate(self, msg: LowState):
        self.latest_lowstate_receive_mono_ns = host_monotonic_ns()
        self.latest_lowstate = msg
        self.have_lowstate = True

    def _on_engaged(self, msg: Bool):
        was = self._policy_engaged
        self._policy_engaged = bool(msg.data)
        if self._policy_engaged and not was:
            if self.reset_on_engage:
                self.encoder.hidden_states = None
                self.last_contact_raw[:] = 0.0
                self.last_action_pol[:] = 0.0
                self.shared_contact_filt[:] = 0.0
                self.have_shared_contact = False
                self.get_logger().info(
                    "reset depth GRU, action history, and contact history")
            self.get_logger().info("/policy_engaged=True")
        elif was and not self._policy_engaged:
            self.get_logger().info("/policy_engaged=False")

    def _on_clock_offset(self, msg: Float32MultiArray):
        # [offset_ms, transport_floor_ms, baseline_ready]
        if len(msg.data) < 3:
            return
        offset_s = float(msg.data[0]) * 1e-3
        # Sanity clamp: the tracker corrects drift/steps of tens of ms. An
        # estimate beyond 0.5 s means broken clocks or a misconfigured
        # tracker (e.g. enabled against sim stamps) — refuse it and keep the
        # last sane value rather than corrupting the latency budget.
        if abs(offset_s) > 0.5:
            self.get_logger().error(
                f"implausible clock offset {offset_s * 1e3:+.1f} ms ignored",
                throttle_duration_sec=5.0)
            return
        self._clock_offset_s = offset_s
        if not self._clock_offset_ready and float(msg.data[2]) >= 1.0:
            self._clock_offset_ready = True
            self.get_logger().info(
                f"clock-offset correction active (offset "
                f"{offset_s * 1e3:+.2f} ms, transport floor "
                f"{float(msg.data[1]):.1f} ms)")

    def _on_last_action(self, msg: Float32MultiArray):
        if self._policy_engaged and len(msg.data) >= NUM_ACTIONS:
            self.last_action_pol = np.asarray(msg.data[:NUM_ACTIONS], dtype=np.float64)

    def _on_shared_contact(self, msg: Float32MultiArray):
        if self._policy_engaged and len(msg.data) >= 4:
            self.shared_contact_filt = np.asarray(msg.data[:4], dtype=np.float64)
            self.have_shared_contact = True

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
            # (backward / centred stick -> 0, no reverse). Verbatim copy of
            # rl_distill_policy._read_lin_vel_x — keep bit-identical.
            return self.lin_vel_max if float(lx) > self.vel_deadzone else 0.0
        return self.lin_vel_max * float(lx)

    def _build_prop(self) -> np.ndarray:
        ms = self.latest_lowstate.motor_state
        qj_unitree = np.array([ms[i].q for i in range(NUM_ACTIONS)], dtype=np.float64)
        dqj_unitree = np.array([ms[i].dq for i in range(NUM_ACTIONS)], dtype=np.float64)
        qj_policy = qj_unitree[MJ_TO_POLICY]
        dqj_policy = dqj_unitree[MJ_TO_POLICY]

        quat = self.latest_lowstate.imu_state.quaternion  # (w, x, y, z)
        roll, pitch, _ = mj_quat_to_roll_pitch_yaw(np.asarray(quat, dtype=np.float64))
        ang_vel_body = np.asarray(self.latest_lowstate.imu_state.gyroscope, dtype=np.float64)

        # Production uses the actor-owned 50 Hz filtered contact. The local
        # path remains only as an explicit diagnostic fallback.
        ff = np.asarray(self.latest_lowstate.foot_force, dtype=np.float64)
        if self.use_shared_contact:
            contact_filt = self.shared_contact_filt.copy()
        elif ff.size == 4:
            contact_filt, raw_now = compute_contact_filt(
                ff, self.last_contact_raw, threshold_n=self.contact_threshold,
            )
            self.last_contact_raw = raw_now
        else:
            contact_filt = np.zeros(4)

        return build_prop(
            qj_policy=qj_policy, dqj_policy=dqj_policy,
            ang_vel_body=ang_vel_body, roll=roll, pitch=pitch,
            last_action_pol=self.last_action_pol,
            contact_filt=contact_filt,
            lin_vel_x=self._read_lin_vel_x(),
            env_class_neq_17=0.0 if self.env_class_eq_17 else 1.0,
            env_class_eq_17=1.0 if self.env_class_eq_17 else 0.0,
        )

    def _on_depth(self, msg: Image):
        receive_mono_ns = host_monotonic_ns()
        receive_wall_ns = time.time_ns()
        if not self.have_lowstate:
            return  # wait until proprio is flowing
        if self.gate_on_engage and not self._policy_engaged:
            return
        if self.use_shared_contact and not self.have_shared_contact:
            # The actor owns the Isaac-style 50 Hz raw[t] OR raw[t-1]
            # contact filter. Never fall back to filtering at the encoder's
            # 10 Hz camera cadence, which would span 100 ms instead of 20 ms.
            return

        try:
            norm = image_msg_to_norm_depth(msg)
        except ValueError as e:
            self.get_logger().warning(str(e), throttle_duration_sec=2.0)
            return
        if norm.shape != (DEPTH_TARGET_H, DEPTH_TARGET_W):
            self.get_logger().warning(
                f"depth shape {norm.shape} != ({DEPTH_TARGET_H},{DEPTH_TARGET_W})",
                throttle_duration_sec=2.0,
            )
            return

        prop = self._build_prop()
        prop_t = torch.from_numpy(prop).float().unsqueeze(0)
        prop_t[:, 6:8] = 0.0  # yaw mask before encoder forward
        depth_t = torch.from_numpy(norm).float().unsqueeze(0)  # (1, 58, 87)

        inference_start_ns = host_monotonic_ns()
        with torch.no_grad():
            out = self.encoder(depth_t, prop_t)  # (1, 32) or (1, 34) with yaw head
        inference_end_ns = host_monotonic_ns()

        # Paper §4.1: enforce constant 80 ms from depth capture to publish.
        # The header stamp is in the camera-host (Jetson) clock domain;
        # _clock_offset_s (0 unless the depth_camera_node tracker is active)
        # maps it into this host's clock so drift/steps between the machines
        # don't silently stretch or shrink the enforced constant.
        capture_s = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        elapsed = (self.get_clock().now().nanoseconds * 1e-9
                   - capture_s - self._clock_offset_s)
        wait = self.enforce_latency_s - elapsed
        sleep_start_ns = host_monotonic_ns()
        if 0 < wait < 0.5:  # cap absurd waits (e.g. clock skew)
            time.sleep(wait)
        elif self.enforce_latency_s > 0.0 and wait <= 0:
            # Frame already older than the budget: publish immediately (a
            # late latent beats a starved actor — the stale-input guards
            # cover a truly dead stream) but count and warn.
            self._late_count += 1
            self.get_logger().warning(
                f"depth frame missed the {self.enforce_latency_s * 1e3:.0f} ms "
                f"budget by {-wait * 1e3:.1f} ms (late #{self._late_count})",
                throttle_duration_sec=2.0)
        sleep_end_ns = host_monotonic_ns()
        achieved = (self.get_clock().now().nanoseconds * 1e-9
                    - capture_s - self._clock_offset_s)
        diag = Float32MultiArray()
        diag.data = [elapsed * 1e3, max(wait, 0.0) * 1e3, achieved * 1e3,
                     self.enforce_latency_s * 1e3, float(self._late_count),
                     self._clock_offset_s * 1e3]
        self.latency_diag_pub.publish(diag)

        latent = out.squeeze(0).numpy().astype(np.float32)  # (32,) or (34,) latent[+yaw_pred]
        if self.trace_log_fp is not None:
            now_s = self.get_clock().now().nanoseconds * 1e-9
            values = np.concatenate((prop, latent[:DEPTH_LATENT_DIM]))
            self.trace_log_fp.write(
                f"{self.trace_tick}\t{now_s:.9f}\t{capture_s:.9f}\t"
                f"{int(self._policy_engaged)}\t" +
                "\t".join(f"{float(v):+.8f}" for v in values) + "\n")
            self.trace_tick += 1
        out_msg = Float32MultiArray()
        out_msg.data = latent.tolist()
        out_msg.layout.data_offset = self.depth_seq
        publish_start_ns = host_monotonic_ns()
        self.pub.publish(out_msg)
        publish_end_ns = host_monotonic_ns()
        if self.latency_log_fp is not None:
            self.latency_log_fp.write(
                f"{self.depth_seq}\t{msg.header.stamp.sec}\t{msg.header.stamp.nanosec}\t"
                f"{receive_mono_ns}\t{receive_wall_ns}\t{int(self.latest_lowstate.tick)}\t"
                f"{self.latest_lowstate_receive_mono_ns}\t{inference_start_ns}\t"
                f"{inference_end_ns}\t{wait * 1000.0:.6f}\t{sleep_start_ns}\t"
                f"{sleep_end_ns}\t{publish_start_ns}\t{publish_end_ns}\t"
                f"{self._clock_offset_s * 1000.0:.6f}\n"
            )
        self.depth_seq = (self.depth_seq + 1) & 0xffffffff

    def destroy_node(self):
        for attr in ("trace_log_fp", "latency_log_fp"):
            fp = getattr(self, attr)
            if fp is not None:
                fp.close()
                setattr(self, attr, None)
        return super().destroy_node()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--ep_root", type=Path,
                    default=_EP_ROOT_DEFAULT)
    ap.add_argument("--enforce_latency_s", type=float, default=DEPTH_LATENCY_S)
    ap.add_argument("--lin_vel_max", type=float, default=0.7)
    ap.add_argument("--force_lin_vel_x", type=float, default=float("nan"))
    ap.add_argument("--binary_vel", type=int, choices=[0, 1], default=0,
                    help="If 1, command a CONSTANT lin_vel_max whenever the left stick is "
                         "pushed forward past --vel_deadzone (bang-bang), instead of scaling "
                         "the command with stick deflection. 0 m/s otherwise (incl. backward). "
                         "MUST match rl_distill_policy --binary_vel or the encoder's "
                         "proprio[10] (cmd_vx) desyncs from the actor's — the latent is "
                         "proprio-driven, so this bites even for depth-blind checkpoints.")
    ap.add_argument("--vel_deadzone", type=float, default=0.2,
                    help="Forward-stick fraction (0..1) above which --binary_vel commands "
                         "lin_vel_max. Default 0.2. MUST match rl_distill_policy "
                         "--vel_deadzone.")
    ap.add_argument("--contact_threshold", type=str, default=str(CONTACT_FORCE_THRESHOLD_N),
                    help="Foot-force threshold for proprio[49:53]: one global value OR "
                         "four comma-separated per-foot values in FR,FL,RR,RL order "
                         "(e.g. 10,20,20,17 for the real robot). MUST match "
                         "rl_distill_policy --contact_threshold or the encoder and "
                         "actor will disagree on the gait-phase bit. Default 1.0 (L00-* "
                         "flat); raise to 10.0 for parkour wall checkpoints.")
    ap.add_argument("--env_class_eq_17", type=int, choices=[0, 1], default=0,
                    help="Set 1 when the checkpoint was trained on parkour_flat "
                         "(env_class==17, go2_flat / L00-* exptids) so prop[11:13] "
                         "= (0, 1). MUST match rl_distill_policy --env_class_eq_17 "
                         "or the encoder's proprio copy desyncs from the actor's "
                         "(latent is proprio-driven, so this bites even for "
                         "depth-blind checkpoints).")
    ap.add_argument("--gate_on_engage", type=int, choices=[0, 1], default=1,
                    help="1 skips depth-GRU forwards until policy engagement.")
    ap.add_argument("--reset_on_engage", type=int, choices=[0, 1], default=1,
                    help="1 clears depth-GRU hidden state on engagement.")
    ap.add_argument("--use_shared_contact", type=int, choices=[0, 1], default=1,
                    help="1 uses the actor's 50 Hz filtered contact topic (default).")
    ap.add_argument("--trace_log", type=Path, default=None,
                    help="Optional line-buffered TSV of encoder proprio and latent.")
    ap.add_argument("--latency_log", type=Path, default=None,
                    help="Optional correlated per-frame timing TSV.")
    args, _ = ap.parse_known_args()

    rclpy.init()
    trace_log_path = args.trace_log
    if trace_log_path is not None and str(trace_log_path) in ("", "."):
        trace_log_path = None
    latency_log_path = args.latency_log
    if latency_log_path is not None and str(latency_log_path) in ("", "."):
        latency_log_path = None
    node = DepthEncoderNode(args.ckpt, args.ep_root,
                             args.enforce_latency_s, args.lin_vel_max,
                             args.force_lin_vel_x,
                             binary_vel=bool(args.binary_vel),
                             vel_deadzone=args.vel_deadzone,
                             contact_threshold=args.contact_threshold,
                             env_class_eq_17=bool(args.env_class_eq_17),
                             gate_on_engage=bool(args.gate_on_engage),
                             reset_on_engage=bool(args.reset_on_engage),
                             use_shared_contact=bool(args.use_shared_contact),
                             trace_log_path=trace_log_path,
                             latency_log_path=latency_log_path)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
