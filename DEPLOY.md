# Deploying a depth-distillation policy on the Unitree Go2

The shipped **L00-02** flat-baseline student as a worked example. Flag *semantics*
generalise to any checkpoint; the *values* do not — observation layout is per-checkpoint.

## The ROS stack

Five nodes in the `deploy_rl_policy` package. `is_simulation` swaps only the first and
last stages; the policy is hardware-agnostic.

```
 RealSense ──/camera/depth/image_raw──▶ depth_camera_node ─┐   (sim: depth_sim_node renders front_depth_cam)
 Go2 SDK   ──/lowstate (500 Hz, DDS)──────────────────────┤   (sim: mujoco_simulator publishes /mujoco/lowstate)
                                                           ▼
                                          depth_encoder_node  (~10 Hz, 80 ms enforced latency)
                                          CNN+GRU → /rl/depth_latent  [32 latent + 2 yaw]
                                                           ▼
 /lowstate (joints, IMU, foot force) ────▶ rl_distill_policy (50 Hz)
                                          53-D proprio + latent → actor → /rl/target_pos [12 joint angles]
                                                           ▼
 /joy (gamepad) ──▶ low_level_ctrl (C++, 200 Hz)  ── lay/stand/engage FSM, PD: τ=kp·(q*−q)−kd·q̇
                                          → /lowcmd (motor cmds, DDS) ──▶ Go2 motors
                   + base_velocity_estimator (EKF: /imu + /contact → /base_velocity)
```

`depth_encoder_node` holds a constant 80 ms latency to match training.
`rl_distill_policy` assembles the 53-D proprio vector (body ang-vel, roll/pitch, command,
joint pos errors, joint vel, last action, foot contacts) every 20 ms, gated on
`/policy_engaged`. `low_level_ctrl` is the only node touching hardware and is the same
C++ node in sim and on the robot. `distill_fused_policy` runs encoder and actor in one
process so both see the same proprio frame per tick.

## L00-02 configuration

A **go2_flat** baseline (env_class 17), distilled with action delay + noise,
`--no_depth_encoder_loss`.

| flag | value | why |
| --- | --- | --- |
| `ckpt` | `models/L00-02-flat-baseline-delay-distillation-noised/model_14500.pt` | 46 MB, in-repo |
| `env_class_eq_17` | **`1`** | flat baseline → `prop[11:13]=(0,1)` |
| `contact_threshold` | **`1.0`** standalone / `2.0` ROS sim | `10.0` over-filters foot contacts → fall (τ_peak ~80–95) |
| `action_delay` | `1` | trained with `--delay`; the deploy default is `0` |
| `policy_kp` / `policy_kd` | `30` / `0.6` | matches training; τ_peak ~31 |
| `lin_vel_max` | `1.3` | m/s; below ~1.0 the student stands still |

Real runs additionally enforce `proprio_latency_s=0.016` and default `contact_threshold`
to the per-foot vector `10,20,20,14`.

## Running it

Environment, once per session:

```bash
source /opt/ros/humble/setup.bash
source <unitree_ros2>/setup.bash     # CycloneDDS on the robot NIC
conda activate <deploy env>          # ROS 2 + torch + mujoco
source install/setup.bash
export MUJOCO_GL=egl                 # headless offscreen render
```

Pre-flight without ROS, then sim, then hardware:

```bash
# 0. confirm the checkpoint walks — pass = base_z ∈ [0.20,0.40], vx tracks command, τ_peak < 35 Nm
python src/deploy_rl_policy/scripts/distill_runner.py \
  --ckpt models/L00-02-flat-baseline-delay-distillation-noised/model_14500.pt \
  --ep_root ../go2-flyball-parkour \
  --lin_vel_x 1.3 --seconds 12 --env_class_eq_17 1 --contact_threshold 1.0

# 1. sim
ros2 launch deploy_rl_policy distill.launch.py is_simulation:=true \
  ckpt:=models/L00-02-flat-baseline-delay-distillation-noised/model_14500.pt \
  env_class_eq_17:=1 contact_threshold:=2.0

# 1b. self-driving recording (lay→stand→walk→lay → MP4)
ros2 launch deploy_rl_policy lay2stand_record.launch.py \
  ckpt:=models/L00-02-flat-baseline-delay-distillation-noised/model_14500.pt \
  scene_xml:=resources/go2/scene_flat.xml camera:=isaac_view \
  env_class_eq_17:=1 contact_threshold:=2.0 lin_vel_max:=1.3

# 2. real robot — sport mode OFF, realsense2_camera publishing /camera/depth/image_rect_raw
ros2 launch deploy_rl_policy distill.launch.py is_simulation:=false \
  ckpt:=models/L00-02-flat-baseline-delay-distillation-noised/model_14500.pt \
  env_class_eq_17:=1 action_delay:=1
```

Gamepad: **B** lay→stand, hold **LB+RB** engage, **A** lay down. The record launches have
no `on_exit` shutdown, so clear orphans between runs: `pkill -f mujoco_simulator`.
Current hardware procedure: [`deploy_mac/runbooks/real_run_runbook.md`](deploy_mac/runbooks/real_run_runbook.md).

## Known limitations

Properties of the stack, not of any one checkpoint.

1. **Observation layout is per-checkpoint.** `env_class_eq_17`, the presence of a yaw head,
   and the contact threshold must match the checkpoint on **both** the encoder and the
   policy node. A mismatch does not error — it degrades the gait. Passing
   `env_class_eq_17=0` or `contact_threshold=10` makes L00-02 stumble or fall.
2. **No watchdog or dead-man's switch.** No command timeout, no stale-latent expiry, and
   the e-stop is an exact-combination joystick press. Keep physical power within reach and
   bring the robot up hoisted or tethered.
3. **Standalone and ROS paths can disagree.** They apply action delay and proprio latency
   differently, and the ROS sim runs its 200 Hz simulator asynchronously against the 50 Hz
   policy (`lockstep=False`), so the same checkpoint can rank differently. Compare like
   with like — [`notes/analysis/standalone_vs_ros_latency_divergence_2026-07-28.md`](notes/analysis/standalone_vs_ros_latency_divergence_2026-07-28.md).
4. **The depth pipeline is sensitive to camera geometry.** `depth_camera_node` FOV-matches
   and crops to the training render (`cam_hfov_deg`, `cam_vfov_deg`, `target_hz`); a
   different unit, FOV or mount pitch is off-distribution, and unfilled RealSense holes
   normalize toward a phantom obstacle —
   [`notes/analysis/sim2real_divergence_camera_pitch_2026-07-27.md`](notes/analysis/sim2real_divergence_camera_pitch_2026-07-27.md).
