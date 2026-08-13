#!/usr/bin/env python3
"""
Extreme Parkour base-policy deployment node.

Subscribes to /lowstate (real robot) or /mujoco/lowstate (sim) and publishes
target joint positions to /rl/target_pos at 50 Hz.

Obs layout (753-dim, matches save_jit_base.py / legged_robot_config.py):
  [0:3]    base_ang_vel × 0.25
  [3:5]    (roll, pitch)
  [5]      0  (masked delta_yaw in history)
  [6]      delta_yaw  (set to 0 → go straight)
  [7]      delta_next_yaw (set to 0)
  [8:10]   0  (masked cmd_xy)
  [10]     forward_vel_cmd (from config)
  [11]     1.0  (env_class != 17, i.e. not wall-turn terrain)
  [12]     0.0  (env_class == 17)
  [13:25]  (dof_pos − default) × 1.0   — Unitree hw order (FR,FL,RR,RL)
  [25:37]  dof_vel × 0.05              — Unitree hw order
  [37:49]  last_action                 — Isaac URDF order (FL,FR,RL,RR)
  [49:53]  foot_contact − 0.5          — Unitree foot order (FR,FL,RR,RL)
  [53:185] zeros (scandots — no lidar)
  [185:194] zeros (priv_explicit — filled by estimator inside JIT)
  [194:223] zeros (priv_latent — unused with hist_encoding)
  [223:753] rolling 10-step history (53-dim proprio, delta_yaw dims zeroed)

Joint ordering:
  Isaac URDF order: FL(0-2), FR(3-5), RL(6-8), RR(9-11)
  Unitree hw order: FR(0-2), FL(3-5), RR(6-8), RL(9-11)
  Permutation ISAAC_TO_UNITREE = [3,4,5,0,1,2,9,10,11,6,7,8]  (self-inverse)
"""

import argparse
import sys
import os
import math
import threading
import time
import numpy as np
import torch
import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor

# Allow int↔float coercion for numeric launch params.
_ANY_NUMERIC = ParameterDescriptor(dynamic_typing=True)
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from pathlib import Path
from unitree_go.msg import LowState
from xbox_command import XboxController
from std_msgs.msg import Float32MultiArray

project_root = Path(os.path.realpath(__file__)).parents[3]

# Isaac↔Unitree joint permutation (self-inverse)
ISAAC_TO_UNITREE = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]

# Obs dimensions
N_PROPRIO   = 53
NUM_SCAN    = 132
N_PRIV_EX   = 9
N_PRIV_LAT  = 29
HISTORY_LEN = 10
NUM_OBS     = N_PROPRIO + NUM_SCAN + N_PRIV_EX + N_PRIV_LAT + HISTORY_LEN * N_PROPRIO  # 753

# Foot contact threshold (Unitree foot_force, estimated Newtons)
CONTACT_THRESHOLD = 20


def quat_to_rpy(quat_wxyz):
    """Convert quaternion [w,x,y,z] to (roll, pitch) in radians."""
    w, x, y, z = quat_wxyz
    roll  = math.atan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    pitch = math.asin(max(-1.0, min(1.0, 2*(w*y - z*x))))
    return roll, pitch


def quat_to_yaw(quat_wxyz):
    """Yaw (z-axis rotation) in radians from a [w,x,y,z] quat."""
    w, x, y, z = quat_wxyz
    return math.atan2(2.0 * (w * z + x * y),
                      1.0 - 2.0 * (y * y + z * z))


def wrap_pi(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class EPRLPolicy(Node):
    def __init__(self, config_path, is_simulation):
        self.is_simulation = is_simulation
        super().__init__("ep_rl_policy")

        # ── Load config ───────────────────────────────────────────────────────
        import yaml
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        self.ang_vel_scale   = cfg["ang_vel_scale"]
        self.dof_pos_scale   = cfg["dof_pos_scale"]
        self.dof_vel_scale   = cfg["dof_vel_scale"]
        self.action_scale    = cfg["action_scale"]
        # ROS param overrides the yaml value if provided. Use 0.0 to stand in
        # place, 1.5 (or anything in the trained command range) to walk.
        self.forward_vel_cmd = float(
            self.declare_parameter("forward_vel_cmd", float(cfg["forward_vel_cmd"]), _ANY_NUMERIC).value
        )
        # If the sandbox training side wrote /tmp/train_dr_env0.npz, prefer
        # the forward_vel_cmd that play.py actually sampled — overriding the
        # ROS param. Without this, training and deploy disagree on obs[10] by
        # whatever the random sample was, which propagates into different
        # actions emitted at iter 0. Sim-only: only relevant when
        # /mujoco/base_lin_vel etc. are also wired up.
        # priv_latent (29 dims) defaults to zero. Training fills it with
        # [added_mass, com_offset(3), friction, motor_strength_p-1(12), motor_strength_d-1(12)].
        # In a sim2sim run we read those values from the sidecar so the actor's
        # priv_encoder sees the exact same vector training did. If no sidecar,
        # priv_latent stays zero (real-robot path).
        self.priv_latent = np.zeros(N_PRIV_LAT, dtype=np.float32)
        if is_simulation:
            try:
                import numpy as _np_sc
                _sc = _np_sc.load("/tmp/train_dr_env0.npz")
                if "forward_vel_cmd" in _sc.files:
                    sampled = float(_sc["forward_vel_cmd"])
                    self.get_logger().info(
                        f"Overriding forward_vel_cmd "
                        f"{self.forward_vel_cmd} → {sampled} (from /tmp/train_dr_env0.npz)"
                    )
                    self.forward_vel_cmd = sampled
                # Layout matches legged_robot.py compute_observations():
                #   priv_latent = cat(mass_params(4), friction(1), m_p-1(12), m_d-1(12))
                # mass_params = [added_mass, com_x, com_y, com_z]
                if all(k in _sc.files for k in ("added_mass", "com_offset", "friction",
                                                "motor_strength_p_isaac", "motor_strength_d_isaac")):
                    self.priv_latent[0]     = float(_sc["added_mass"])
                    self.priv_latent[1:4]   = _np_sc.asarray(_sc["com_offset"], dtype=np.float32)
                    self.priv_latent[4]     = float(_sc["friction"])
                    self.priv_latent[5:17]  = _np_sc.asarray(_sc["motor_strength_p_isaac"], dtype=np.float32) - 1.0
                    self.priv_latent[17:29] = _np_sc.asarray(_sc["motor_strength_d_isaac"], dtype=np.float32) - 1.0
                    self.get_logger().info(
                        f"Loaded priv_latent from sidecar: friction={self.priv_latent[4]:.3f}, "
                        f"added_mass={self.priv_latent[0]:.3f}"
                    )
            except FileNotFoundError:
                pass
        # When >0, write the first N policy iterations' obs+action vectors to
        # /tmp/deploy_obs_<i>.npz for slice-by-slice comparison against the
        # extreme-parkour training environment.
        self._dumps_left = int(self.declare_parameter("dump_obs", 0).value)
        self._dump_idx = 0
        # Debug switch: feed obs[priv_explicit] from MuJoCo's ground-truth
        # body-frame base_lin_vel instead of letting the BasePolicy estimator
        # predict it. The remaining 6 priv slots are zero in training (they're
        # padded), so we leave them zero. With this on, the actor sees exactly
        # the same priv distribution that play.py's act_inference does in
        # IsaacGym, isolating the physics gap from the estimator gap.
        self.use_ground_truth_priv = bool(
            self.declare_parameter("use_ground_truth_priv", False).value
        )
        # Separately opt-in to filling priv_latent[194:223] from the sidecar
        # (mass/friction/motor_strength). Off by default — the previous
        # behaviour (priv_latent stays zero) was tuned for stable walking. Turn
        # on only for byte-exact iter-0 obs comparison vs training. May change
        # actor output and destabilise the gait if the policy responds to
        # priv_latent at inference (e.g. if hist_encoding doesn't fully shadow it).
        self.use_priv_latent_sidecar = bool(
            self.declare_parameter("use_priv_latent_sidecar", False).value
        )

        # ── Deterministic capture mode ────────────────────────────────────────
        # When enabled, the policy:
        #   • overrides forward_vel_cmd from a 2-segment schedule
        #     (det_pre_cmd for steps [0, det_switch_step), det_post_cmd after)
        #   • buffers obs (NUM_OBS) and action (12) on every _run iteration
        #   • after det_total_steps, writes them to det_out and shuts down
        # Pair with a deterministic /tmp/train_dr_env0.npz (no forward_vel_cmd
        # field) and the matching Isaac play_deterministic.py rollout for
        # 1-to-1 obs/action comparison.
        self.det_capture = bool(
            self.declare_parameter("det_capture", False).value
        )
        self.det_total_steps = int(
            self.declare_parameter("det_total_steps", 500, _ANY_NUMERIC).value
        )
        self.det_switch_step = int(
            self.declare_parameter("det_switch_step", 250, _ANY_NUMERIC).value
        )
        self.det_pre_cmd = float(
            self.declare_parameter("det_pre_cmd", 0.0, _ANY_NUMERIC).value
        )
        self.det_post_cmd = float(
            self.declare_parameter("det_post_cmd", 0.5, _ANY_NUMERIC).value
        )
        self.det_out = str(
            self.declare_parameter("det_out", "/tmp/deploy_det.npz").value
        )
        # Goal tracking: place a fixed goal `goal_dist` m along the robot's
        # initial body-forward axis (captured from first /mujoco/pos). Each
        # _run computes delta_yaw to that goal and overrides proprio[6:8].
        # When off (default real-robot path) proprio[6:8] stay at hardcoded 0.
        self.track_goal = bool(
            self.declare_parameter("track_goal", False).value
        )
        self.goal_dist = float(
            self.declare_parameter("goal_dist", 5.0, _ANY_NUMERIC).value
        )
        # Hold-to-stand schedule: for the first N policy steps after the
        # first lowstate arrives, override forward_vel_cmd to 0 so the
        # robot stabilises in the standing pose before walking. After N
        # steps the configured forward_vel_cmd takes over. Independent of
        # det_capture (which has its own pre/post schedule). 0 = disabled.
        self.stand_for_steps = int(
            self.declare_parameter("stand_for_steps", 0, _ANY_NUMERIC).value
        )
        # Delay the very first policy step by this many wall-clock seconds
        # after the node's first lowstate is received. While we're delaying
        # we don't publish /rl/target_pos, so low_level_ctrl never publishes
        # /mujoco/lowcmd, so mujoco_simulator's step_simulation thread stays
        # idle — the robot sits frozen at the post-presettle standing pose.
        # Use this to give the MuJoCo viewer time to open over X11/VGL
        # before motion starts. 0 disables (default).
        self.startup_wait_s = float(
            self.declare_parameter("startup_wait_s", 0.0, _ANY_NUMERIC).value
        )
        self._first_lowstate_time = None  # filled when first /mujoco/lowstate arrives
        if self.startup_wait_s > 0:
            self.get_logger().info(
                f"startup_wait_s: holding the policy for the first "
                f"{self.startup_wait_s:.1f} s after first /mujoco/lowstate"
            )
        # The "post-stand" forward-vel command. We snapshot it here because
        # the schedule's fall-through case re-reads self.forward_vel_cmd; if
        # det_capture also messes with it, we still want the user's
        # forward_vel_cmd:=… launch arg to be the value we ramp into.
        self._post_stand_cmd = self.forward_vel_cmd
        if self.stand_for_steps > 0:
            self.get_logger().info(
                f"stand_for_steps: holding forward_vel_cmd=0 for the first "
                f"{self.stand_for_steps} policy steps "
                f"({self.stand_for_steps * 0.02:.1f} s), then "
                f"forward_vel_cmd={self._post_stand_cmd}"
            )
        if self.det_capture:
            self._det_obs_buf = np.zeros((self.det_total_steps, NUM_OBS), dtype=np.float32)
            self._det_act_buf = np.zeros((self.det_total_steps, 12), dtype=np.float32)
            self._det_cmd_buf = np.zeros((self.det_total_steps,), dtype=np.float32)
            self._det_step = 0
            self.get_logger().info(
                f"det_capture: cmd={self.det_pre_cmd} for {self.det_switch_step} steps, "
                f"then {self.det_post_cmd}; total={self.det_total_steps}; out={self.det_out}"
            )

        # Default joint angles in Unitree order (FR,FL,RR,RL) — for obs building
        self.default_unitree = np.array(cfg["default_angles"],       dtype=np.float32)
        # Default joint angles in Isaac order (FL,FR,RL,RR) — for action target
        self.default_isaac   = np.array(cfg["default_angles_isaac"], dtype=np.float32)

        # ── Load policy ───────────────────────────────────────────────────────
        # Two paths:
        #   use_sandbox_actor=True  → instantiate Actor+Estimator from
        #     ep_actor_sandbox.py and load model_*.pt state_dict directly.
        #     This is bit-identical to the training-side `policy(obs,
        #     hist_encoding=True)` call from play.py — no JIT trace in between.
        #   use_sandbox_actor=False → torch.jit.load the legacy traced model.
        script_dir = os.path.dirname(os.path.realpath(__file__))
        use_sandbox = bool(cfg.get("use_sandbox_actor", False))
        # ROS param can override yaml at launch time.
        use_sandbox = bool(self.declare_parameter("use_sandbox_actor", use_sandbox).value)

        if use_sandbox:
            sys.path.insert(0, script_dir)
            from ep_actor_sandbox import build_base_actor
            ckpt_rel  = cfg["checkpoint_path"]
            ckpt_path = ckpt_rel if os.path.isabs(ckpt_rel) else os.path.join(script_dir, ckpt_rel)
            self.policy = build_base_actor(ckpt_path, device="cpu")
            self.get_logger().info(f"Loaded sandbox policy from checkpoint {ckpt_path}")
        else:
            policy_path = os.path.join(script_dir, cfg["policy_path"])
            self.policy = torch.jit.load(policy_path, map_location="cpu")
            self.policy.eval()
            self.get_logger().info(f"Loaded JIT policy from {policy_path}")

        # ── State buffers ─────────────────────────────────────────────────────
        self.qj          = np.zeros(12, dtype=np.float32)  # Unitree order
        self.dqj         = np.zeros(12, dtype=np.float32)  # Unitree order
        self.last_action = np.zeros(12, dtype=np.float32)  # Isaac order
        # History buffer: shape (HISTORY_LEN, N_PROPRIO), with delta_yaw dims zeroed
        self.history_buf = np.zeros((HISTORY_LEN, N_PROPRIO), dtype=np.float32)
        self.history_initialised = False
        # Counter mirroring training's `episode_length_buf` for the
        # "episode_length<=1 → fill with 10 copies, else roll+append" logic.
        self._step_count = 0

        self.low_state = LowState()
        self._got_low_state = False
        # Ground-truth body-frame base_lin_vel from MuJoCo (3 floats). Filled
        # by /mujoco/base_lin_vel CB. Used only when use_ground_truth_priv=True.
        self.gt_base_lin_vel = np.zeros(3, dtype=np.float32)
        # Set every time _low_state_cb fires. _run() clears it after consuming
        # state and reading the latest snapshot, then waits for the next set
        # (i.e. the post-step lowstate that came back from THIS iter's
        # /rl/target_pos publish). Requires MultiThreadedExecutor so the CB
        # can fire while _run is blocked on .wait().
        self._fresh_state_event = threading.Event()
        # Tick (decim-batch counter from mujoco_simulator) we last consumed.
        # _low_state_cb only sets _fresh_state_event when msg.tick advances
        # past this — so a 500 Hz republish of the same tick can't fool _run.
        self._wait_for_tick = -1

        # Scandots: in sim, the simulator publishes 132 pre-computed obs values
        # (already in the form `clip(base_z - 0.3 - terrain_h, -1, 1)`). On the
        # real robot, no scandot source exists; the slice stays zero.
        # Initialise to the flat-ground signal: base_z=0.42 (training reset
        # height) → clip(0.42 - 0.3 - 0, -1, 1) = +0.12. This way the very
        # first policy iteration has in-distribution scandot obs even if
        # /mujoco/scandots hasn't fired yet. Real values arrive 20 ms later.
        self.scandots = np.full(NUM_SCAN, 0.12, dtype=np.float32)

        # ── Xbox controller ───────────────────────────────────────────────────
        self.cmd_sub = XboxController(self)

        # ── ROS topics ────────────────────────────────────────────────────────
        # Two callback groups so the timer doesn't race itself:
        #   timer_group  = MutuallyExclusive — only one _run() at a time. If
        #     a tick takes >20 ms (cold inference, slow chain), the next tick
        #     waits until the current returns. Without this, dumps overlap
        #     (saw deploy_obs_7 twice + deploy_obs_8 missing).
        #   io_group     = Reentrant — _low_state_cb can fire on a worker
        #     thread while _run is blocked on _fresh_state_event.wait().
        timer_group = MutuallyExclusiveCallbackGroup()
        io_group    = ReentrantCallbackGroup()
        topic = "/mujoco/lowstate" if is_simulation else "/lowstate"
        self.low_state_sub = self.create_subscription(
            LowState, topic, self._low_state_cb, 10, callback_group=io_group)
        if is_simulation:
            self.scandot_sub = self.create_subscription(
                Float32MultiArray, "/mujoco/scandots", self._scandot_cb, 10,
                callback_group=io_group)
            self.lin_vel_sub = self.create_subscription(
                Float32MultiArray, "/mujoco/base_lin_vel", self._lin_vel_cb, 10,
                callback_group=io_group)
            # /mujoco/pos publishes qpos[:19] = [x, y, z, w, qx, qy, qz, joints…].
            # We only need x, y here — the imu quaternion gives us yaw via
            # quat_to_rpy/yaw, so the goal-tracking signal can be computed
            # entirely from this + the existing low_state.imu_state.
            self.pos_sub = self.create_subscription(
                Float32MultiArray, "/mujoco/pos", self._pos_cb, 10,
                callback_group=io_group)
        self._got_pos = False
        self.base_xy = np.zeros(2, dtype=np.float32)
        self.base_yaw_world = 0.0
        # World-frame goal coords, populated at first /mujoco/pos receipt
        # when track_goal=True. Stays None on real robot / when disabled.
        self._goal_world = None
        self.target_pub = self.create_publisher(Float32MultiArray, "/rl/target_pos", 10)
        self._is_simulation = is_simulation
        self._timer_group = timer_group  # for create_timer below

        # ── Warm up the actor before the timer starts ─────────────────────────
        # The first torch.no_grad forward pass on a cold cache takes 50–200 ms
        # (autograd graph allocation + kernel cache). If iter 0 of _run() hits
        # that cold path, it blocks long enough that iter 1's 50 Hz timer
        # fires before iter 0's command has propagated through low_level_ctrl
        # → mujoco_simulator → /mujoco/lowstate. iter 1 then reads pre-step
        # state — the bit-exact iter-0/iter-1 obs lag we hunted in dumps.
        # Do 3 warmup forwards on a zero obs so subsequent inferences are fast.
        with torch.no_grad():
            warm_obs = torch.zeros(1, NUM_OBS)
            for _ in range(3):
                _ = self.policy(warm_obs)
        self.get_logger().info("Policy warmup done")

        # 50 Hz control loop in its own MutuallyExclusive group so a slow
        # _run can't get re-entered by the next 20 ms tick.
        self.timer = self.create_timer(0.02, self._run, callback_group=self._timer_group)
        self.get_logger().info(
            f"EP policy node ready ({'simulation' if is_simulation else 'real robot'}), "
            f"forward_vel={self.forward_vel_cmd}"
        )

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _low_state_cb(self, msg: LowState):
        self.low_state = msg
        if not self._got_low_state:
            self._first_lowstate_time = time.time()
        self._got_low_state = True
        # Only signal "fresh post-step state" when tick has advanced past
        # the value seen at our last publish. tick is mujoco_simulator's
        # decim-batch counter — between batches the 500 Hz publish_sensor_data
        # republishes the SAME tick, which we don't want to wake _run on.
        if msg.tick > self._wait_for_tick:
            self._fresh_state_event.set()

    def _scandot_cb(self, msg: Float32MultiArray):
        if len(msg.data) == NUM_SCAN:
            self.scandots = np.asarray(msg.data, dtype=np.float32)
            self._got_scandots = True

    def _pos_cb(self, msg: Float32MultiArray):
        # /mujoco/pos publishes qpos[:19]: [x, y, z, w, qx, qy, qz, joints…].
        # We only need x, y here for goal-tracking. Yaw comes from the same
        # imu quaternion the policy already reads via _low_state_cb.
        if len(msg.data) >= 7:
            self.base_xy[0] = float(msg.data[0])
            self.base_xy[1] = float(msg.data[1])
            qw, qx, qy, qz = (float(msg.data[3]), float(msg.data[4]),
                              float(msg.data[5]), float(msg.data[6]))
            self.base_yaw_world = quat_to_yaw((qw, qx, qy, qz))
            if not self._got_pos:
                self._got_pos = True
                if self.track_goal:
                    self._goal_world = (
                        float(self.base_xy[0])
                        + self.goal_dist * math.cos(self.base_yaw_world),
                        float(self.base_xy[1])
                        + self.goal_dist * math.sin(self.base_yaw_world),
                    )
                    self.get_logger().info(
                        f"track_goal: anchored at world ({self._goal_world[0]:.2f}, "
                        f"{self._goal_world[1]:.2f}) — {self.goal_dist} m forward "
                        f"of initial pose (yaw {math.degrees(self.base_yaw_world):.1f}°)"
                    )

    def _lin_vel_cb(self, msg: Float32MultiArray):
        if len(msg.data) == 3:
            self.gt_base_lin_vel = np.asarray(msg.data, dtype=np.float32)

    def _run(self):
        if self.cmd_sub.axes and self.cmd_sub.axes[2] == -1 and self.cmd_sub.axes[5] == -1:
            sys.exit()

        # Don't run inference until the first real /mujoco/lowstate has arrived.
        # The default-constructed LowState has all-zero motor_state which makes
        # the policy think every joint is at 0 (dof_pos - default ≈ -default),
        # producing garbage actions that crash the robot in the first 20 ms.
        if not self._got_low_state:
            return

        # Optional startup hold so the MuJoCo viewer can finish loading before
        # any motion happens. While we early-return here, low_level_ctrl never
        # sees a /rl/target_pos and the simulator's step_simulation thread
        # stays waiting on _step_event — the robot is frozen at the post-
        # presettle standing pose, viewer or no viewer.
        if self.startup_wait_s > 0 and self._first_lowstate_time is not None:
            elapsed = time.time() - self._first_lowstate_time
            if elapsed < self.startup_wait_s:
                if not getattr(self, "_startup_wait_logged_at", -1) > -1:
                    self.get_logger().info(
                        f"startup_wait_s active: {self.startup_wait_s - elapsed:.1f} s "
                        f"to go before first policy step"
                    )
                    self._startup_wait_logged_at = elapsed
                return

        # det_capture: overwrite forward_vel_cmd from the schedule before the
        # obs vector is built. Step 0 = first inference after first lowstate.
        if self.det_capture:
            i = self._det_step
            self.forward_vel_cmd = (
                self.det_pre_cmd if i < self.det_switch_step else self.det_post_cmd
            )
        elif self.stand_for_steps > 0:
            # Stand-then-walk: forward_vel=0 for the first N steps, then
            # the configured forward_vel_cmd kicks in.
            self.forward_vel_cmd = (
                0.0 if self._step_count < self.stand_for_steps
                else self._post_stand_cmd
            )

        state = self.low_state

        # ── Joint states (Unitree hw order: FR,FL,RR,RL) ─────────────────────
        for i in range(12):
            self.qj[i]  = state.motor_state[i].q
            self.dqj[i] = state.motor_state[i].dq

        # ── IMU ───────────────────────────────────────────────────────────────
        quat    = state.imu_state.quaternion   # [w,x,y,z]
        ang_vel = np.array(state.imu_state.gyroscope, dtype=np.float32)
        roll, pitch = quat_to_rpy(quat)

        # ── Foot contact (Unitree foot order: FR,FL,RR,RL) ────────────────────
        foot_force  = np.array([state.foot_force[i] for i in range(4)], dtype=np.float32)
        raw_contact = foot_force > CONTACT_THRESHOLD
        # Simple first-order filter: contact fires if current OR previous step
        if not hasattr(self, "_last_contact"):
            self._last_contact = raw_contact.copy()
        contact_filt = np.logical_or(raw_contact, self._last_contact).astype(np.float32)
        self._last_contact = raw_contact.copy()

        # ── Build current proprio (53-dim) ────────────────────────────────────
        proprio = np.zeros(N_PROPRIO, dtype=np.float32)
        # [0:3]  ang_vel
        proprio[0:3] = ang_vel * self.ang_vel_scale
        # [3:5]  roll, pitch
        proprio[3]   = roll
        proprio[4]   = pitch
        # [5]    masked delta_yaw (always 0)
        # [6]    delta_yaw to current goal (single fixed goal when track_goal=True)
        # [7]    delta_next_yaw to next goal (same as [6] — single goal)
        # [8:10] masked cmd_xy (always 0)
        # [10]   forward vel command
        if self.track_goal and self._goal_world is not None:
            tx = self._goal_world[0] - float(self.base_xy[0])
            ty = self._goal_world[1] - float(self.base_xy[1])
            target_yaw_world = math.atan2(ty, tx)
            delta_yaw = wrap_pi(target_yaw_world - self.base_yaw_world)
            proprio[6] = delta_yaw
            proprio[7] = delta_yaw
        proprio[10]  = self.forward_vel_cmd
        # [11]   env_class != 17 (parkour terrain, not wall-turn)
        proprio[11]  = 1.0
        # [12]   env_class == 17
        # [13:25] joint pos (Unitree order)
        proprio[13:25] = (self.qj - self.default_unitree) * self.dof_pos_scale
        # [25:37] joint vel (Unitree order)
        proprio[25:37] = self.dqj * self.dof_vel_scale
        # [37:49] last action — *Isaac order, raw policy output (pre-clip)*.
        # Training: env.step() reindexes actions (Isaac→Unitree) before
        # stuffing into action_history_buf, then compute_observations applies
        # `self.reindex(action_history_buf[-1])` again. Because the perm
        # [3,4,5,0,1,2,9,10,11,6,7,8] is SELF-INVERSE, applying it twice gives
        # back the original raw Isaac-order action. So obs[37:49] is in Isaac
        # order. (Earlier I mistakenly applied ISAAC_TO_UNITREE here — that
        # double-permuted relative to training and caused asymmetric actions.)
        proprio[37:49] = self.last_action
        # [49:53] foot contact - 0.5 (Unitree foot order)
        proprio[49:53] = contact_filt - 0.5

        # ── History on the first iteration: 10 stacked copies of proprio ─────
        # Training (legged_robot.py:683-690) updates obs_history_buf via the
        # `episode_length_buf <= 1` branch of torch.where on iter 0 — meaning
        # obs_history_buf is *already* 10 copies of current obs_buf when
        # compute_observations later concatenates it into the final obs.
        # The deploy code used to fill history at the END of _run, leaving
        # iter 0's obs[223:] all zeros — a structural sim2train mismatch
        # that yielded a ~0.15 step-0 action diff in the deterministic
        # comparison. Fill BEFORE obs assembly to match training.
        if self._step_count <= 0:
            hist_entry_init       = proprio.copy()
            hist_entry_init[6:8]  = 0.0
            for j in range(HISTORY_LEN):
                self.history_buf[j] = hist_entry_init

        # ── Assemble full 753-dim obs ─────────────────────────────────────────
        obs = np.zeros(NUM_OBS, dtype=np.float32)
        obs[0:N_PROPRIO]                              = proprio
        # [53:185]  scandots — filled from MuJoCo raycast in sim, zero on real robot
        obs[N_PROPRIO:N_PROPRIO + NUM_SCAN]           = self.scandots
        # [185:194] priv_explicit. Default = 0 (BasePolicy's estimator fills it
        # at inference time). When use_ground_truth_priv is True, fill the first
        # 3 slots with body-frame base_lin_vel * lin_vel_scale (=2.0) — exactly
        # what training's compute_observations writes via
        # `priv_explicit = cat(base_lin_vel * obs_scales.lin_vel, 0*lin, 0*lin)`.
        # The remaining 6 slots stay 0 (training pads them too).
        if self._is_simulation and self.use_ground_truth_priv:
            obs[N_PROPRIO + NUM_SCAN : N_PROPRIO + NUM_SCAN + 3] = self.gt_base_lin_vel * 2.0
        # [194:223] priv_latent. Default = 0 (real-robot path; the actor's
        # priv_encoder is bypassed when hist_encoding=True). For byte-exact
        # iter-0 obs match with training, opt in via use_priv_latent_sidecar.
        if self._is_simulation and self.use_priv_latent_sidecar:
            obs[N_PROPRIO + NUM_SCAN + N_PRIV_EX :
                N_PROPRIO + NUM_SCAN + N_PRIV_EX + N_PRIV_LAT] = self.priv_latent
        obs[N_PROPRIO + NUM_SCAN + N_PRIV_EX + N_PRIV_LAT:] = self.history_buf.flatten()
        obs = np.clip(obs, -100, 100)

        # ── Policy inference ──────────────────────────────────────────────────
        obs_tensor = torch.from_numpy(obs).unsqueeze(0)  # (1, 753)
        with torch.no_grad():
            if self.use_ground_truth_priv and hasattr(self.policy, "actor"):
                # Bypass BasePolicy.forward (which would overwrite obs[priv]
                # with estimator output) and call the raw Actor directly.
                # Matches play.py's act_inference path exactly: no estimator,
                # actor reads obs[priv_explicit] as-is.
                action = self.policy.actor(
                    obs_tensor, hist_encoding=True, eval=False, scandots_latent=None
                ).numpy().squeeze()
            else:
                action = self.policy(obs_tensor).numpy().squeeze()  # (12,) Isaac order

        # det_capture: buffer obs/action this step. After det_total_steps,
        # write the consolidated arrays and shut down so the wrapper can diff
        # against the matching Isaac trace.
        if self.det_capture:
            i = self._det_step
            self._det_obs_buf[i] = obs
            self._det_act_buf[i] = action
            self._det_cmd_buf[i] = self.forward_vel_cmd
            self._det_step += 1
            if self._det_step >= self.det_total_steps:
                np.savez(
                    self.det_out,
                    obs=self._det_obs_buf,
                    actions=self._det_act_buf,
                    cmd_x=self._det_cmd_buf,
                    dt=np.float32(0.02),
                    switch_step=np.int32(self.det_switch_step),
                )
                self.get_logger().info(
                    f"det_capture: wrote {self.det_out} "
                    f"(obs {self._det_obs_buf.shape}, actions {self._det_act_buf.shape})"
                )
                # Hard exit — clean rclpy.shutdown can deadlock with the timer
                # callback group; we have all the data we need.
                os._exit(0)

        # One-shot obs/action dump for sim2sim ↔ training perception diff. Run
        # extreme-parkour's play.py with the same checkpoint, log the same
        # arrays at env reset, then compare slice-by-slice.
        if self._dumps_left > 0:
            np.savez(
                f"/tmp/deploy_obs_{self._dump_idx}.npz",
                proprio=proprio,
                scandots=self.scandots,
                history_buf=self.history_buf,
                obs=obs,
                action_isaac=action,
                qj_unitree=self.qj,
                dqj_unitree=self.dqj,
                gt_base_lin_vel=self.gt_base_lin_vel,
                use_ground_truth_priv=np.array(self.use_ground_truth_priv, dtype=bool),
            )
            self.get_logger().info(
                f"dumped /tmp/deploy_obs_{self._dump_idx}.npz "
                f"(proprio mean={float(proprio.mean()):+.3f}, "
                f"action mean={float(action.mean()):+.3f})"
            )
            self._dump_idx += 1
            self._dumps_left -= 1

        self.last_action = action.copy()

        # ── Update history for next iter (matches training's roll branch) ────
        # The first-iter "10 stacked copies" branch already ran BEFORE obs
        # assembly above (so the policy sees the filled history at step 0).
        # Here we only do the roll+append used from step 1 onward.
        hist_entry       = proprio.copy()
        hist_entry[6:8]  = 0.0
        if self._step_count > 0:
            self.history_buf = np.roll(self.history_buf, shift=-1, axis=0)
            self.history_buf[-1] = hist_entry
        self._step_count += 1

        # ── Compute target joint positions ────────────────────────────────────
        # Action is in Isaac order → target_pos in Isaac order → convert to Unitree.
        # Clip the action to the same range training uses for `_compute_torques`:
        #   clip_actions = cfg.normalization.clip_actions / cfg.control.action_scale
        #                = 1.2 / 0.25 = 4.8
        # Without this clip the unbounded raw policy output produces target
        # positions ±2-3 rad away from default, which the PD chases to the
        # torque limit; training only ever saw targets at ±1.2 rad off default.
        clip = 1.2 / self.action_scale  # 4.8
        action_clipped = np.clip(action, -clip, clip)
        target_isaac   = self.default_isaac + action_clipped * self.action_scale
        target_unitree = target_isaac[ISAAC_TO_UNITREE]

        # ── Publish ───────────────────────────────────────────────────────────
        # Clear the fresh-state event BEFORE publishing so we ignore any
        # already-pending lowstate (which is from BEFORE this publish). Then
        # wait for the next CB after the publish — that lowstate is the
        # post-step state we want iter N+1 to read. Sim-only lockstep: in
        # real-robot mode the policy is wall-clock tied (500 Hz state stream)
        # and skipping a tick would just stale obs by 2 ms.
        if self._is_simulation:
            # Snapshot the current tick BEFORE publishing — we want the next
            # decim batch's tick (i.e. > current) to wake the wait.
            self._wait_for_tick = int(self.low_state.tick)
            self._fresh_state_event.clear()
        msg      = Float32MultiArray()
        msg.data = target_unitree.astype(np.float32).tolist()
        self.target_pub.publish(msg)
        if self._is_simulation:
            # 50 ms ceiling — the chain (low_level_ctrl → mujoco_simulator
            # 4 mj_steps → /mujoco/lowstate → CB) normally completes in
            # ~5–10 ms. If it doesn't fire in 50 ms, ROS plumbing is wedged
            # and waiting longer wouldn't help; let the next iter try again.
            self._fresh_state_event.wait(timeout=0.05)


def main():
    rclpy.init()
    config_path = project_root / "src" / "deploy_rl_policy" / "configs" / "go2_ep.yaml"
    node = EPRLPolicy(config_path=str(config_path), is_simulation=args.is_simulation)
    # MultiThreadedExecutor lets _low_state_cb fire on a worker thread while
    # _run() is blocked on _fresh_state_event.wait(). Two threads is enough
    # for the timer + lowstate + scandots fan-in (all in a ReentrantCallbackGroup).
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    import sys
    from rclpy.utilities import remove_ros_args
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_simulation", type=str, choices=["True", "False"], default="True")
    args = parser.parse_args(remove_ros_args(sys.argv)[1:])
    args.is_simulation = args.is_simulation == "True"
    main()
