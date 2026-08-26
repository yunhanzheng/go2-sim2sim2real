# go2-sim2sim2real

*The MuJoCo wall-turn scene: two hurdles and an angled plate against a wall.*

ROS 2 low-level control for the Unitree Go2. Validate a policy in MuJoCo (sim2sim),
then run the identical node stack on hardware by flipping `is_simulation:=false`
(sim2real) — including a native macOS (Apple Silicon) deploy path.

Policies come from the training repo
**[go2-flyball-parkour](https://github.com/Roman-089/go2-flyball-parkour)**, which
supplies the `Actor` / `Estimator` / `RecurrentDepthBackbone` classes this stack loads.
Point `ep_root` (or `$EP_ROOT`) at a clone of it; the default assumes a sibling directory.

*Go2 hardware on the same course: wall plant, flat ground and plank.*

Additions over the upstream framework this forks:

- **Depth-distillation deploy stack** — depth camera → recurrent encoder → 50 Hz base
policy, as separate nodes (`depth_encoder_node` + `rl_distill_policy`) or fused into
one process (`distill_fused_policy`) so both nets share a single proprio frame.
- **macOS (Apple Silicon) deploy** via RoboStack (`[deploy_mac/README.md](deploy_mac/README.md)`).
- **ROS-free MuJoCo harnesses** — `distill_runner.py` (depth student),
`lidar_runner.py` (privileged teacher, raycast scandots).
- **Scripted, reproducible runs** — `joy_sequencer` replaces the gamepad and scripts
lay → stand → engage → laydown while recording every topic of interest.
- **Depth pipeline faithful to training** — FOV-matched crop and resize to the sim
render's 58×87, decimated to the encoder's training cadence.

MIT (`[LICENSE](LICENSE)`), inherited from
[Deploy-an-RL-policy-on-the-Unitree-Go2-robot](https://github.com/Glowing-Torch/Deploy-an-RL-policy-on-the-Unitree-Go2-robot).
The policies are CC BY-NC 4.0 from the training repo.

## Quick start

### 1. Build

```bash
# ROS 2 Humble (or Foxy) + MuJoCo 3.2.3 + Pinocchio 3.4.0
source /opt/ros/humble/setup.bash
source <unitree_ros2>/setup.bash        # CycloneDDS on the robot NIC

git clone git@github.com:Roman-089/go2-sim2sim2real.git
git clone git@github.com:Roman-089/go2-flyball-parkour.git   # sibling: supplies the policy classes
cd go2-sim2sim2real && colcon build --symlink-install && source install/setup.bash
```

> [!NOTE]
> ROS 2 Foxy generates C typesupport only for Python 3.8; on another interpreter the
> build succeeds but imports fail at runtime. Fix:
> `export PYTHON_EXECUTABLE=/usr/bin/python3.8` before `colcon build`.

macOS: `[deploy_mac/README.md](deploy_mac/README.md)` (RoboStack env, linker workarounds
for the Unitree messages, Jetson depth path).

### 2. Checkpoint verification without ROS

MuJoCo only, no build — confirm a policy walks before it reaches hardware:

```bash
# depth student on the wall-turn course (also writes an MP4 under run_logs/sim2sim/)
python src/deploy_rl_policy/scripts/distill_runner.py \
    --ckpt models/L00-02-flat-baseline-delay-distillation-noised/model_14500.pt \
    --scene resources/go2/scene_two_hurdles_wall_turn.xml \
    --lin_vel_x 1.3 --seconds 12 --init_yaw_deg 90 --record wall_turn.mp4

# corridor student
python src/deploy_rl_policy/scripts/distill_runner.py \
  --ckpt models/student/model_10000.pt \
  --scene resources/go2/scene_flyball30_2h_h13_noffset.xml \
  --lin_vel_x 1.3 --seconds 15 --record flyball.mp4

# privileged teacher, raycast scandots instead of depth
python src/deploy_rl_policy/scripts/lidar_runner.py --ckpt <teacher>/model_3200.pt
```

Pass = `base_z ∈ [0.20, 0.40]`, `τ_peak < 35 Nm`, `vx` tracking the command, `d_lat|`
bounded. Commands are bimodal from training: `**--lin_vel_x` below ~1.0 reads as
stand-still**, so verify at 1.3, not at 0.5. Details in
`[README_distill.md](src/deploy_rl_policy/scripts/README_distill.md)`.

### 3. sim2sim

```bash
ros2 launch deploy_rl_policy distill.launch.py is_simulation:=true \
  use_keyboard:=true \
  ckpt:=models/student/model_10000.pt \
  scene_xml:=resources/go2/scene_flyball30_2h_h13_noffset.xml
```

`depth_mode:=render` renders `front_depth_cam` from the live qpos; `depth_mode:=zeros`
skips the offscreen renderer for a smoke test. Headless lay→stand→walk→lay to MP4:
`ros2 launch deploy_rl_policy lay2stand_record.launch.py ckpt:=<ckpt> camera:=isaac_view`.
Orphaned processes between launches: `./deploy_mac/kill_ros.sh`.

### 4. sim2real

**Turn the Go2's sport mode off first.**

```bash
ros2 launch deploy_rl_policy distill.launch.py is_simulation:=false ckpt:=<ckpt>
```

Gamepad: **B** lay→stand, hold **LB+RB** engage, **A** lay down. Real runs default
`contact_threshold` to the per-foot vector `10,20,20,14` — a low sim threshold on
hardware reads every foot as always-in-contact, so there is no gait phase.

There is no watchdog, command timeout or stale-latent expiry; the e-stop is a joystick
combo. Bring up hoisted or tethered — see
`[deploy_mac/runbooks/real_run_runbook.md](deploy_mac/runbooks/real_run_runbook.md)`.

## Node stack

Five ROS 2 nodes; sim and real differ only at the camera and robot ends — pipeline
diagram and per-node detail in `[DEPLOY.md](DEPLOY.md)`. `distill_fused_policy.py` runs
encoder and actor in one process so both see the same proprio frame per tick, matching
the distillation setup. The three-state FSM (`low_level_ctrl`, C++, 200 Hz) is the only
node touching hardware and is identical in sim and on the robot: **Laying Down →
Standing Up → Executing RL Policy**, with PD `τ = kp·(q*−q) − kd·q̇` on the motor MCUs.

> [!IMPORTANT]
> Checkpoints differ in observation layout (yaw head vs. none, `env_class_eq_17` flat vs.
> obstacle terrain). These flags must match the checkpoint on **both** the encoder and
> the policy node — a mismatch degrades the policy silently. See `[DEPLOY.md](DEPLOY.md)`.

24 MuJoCo scenes in `resources/go2/` (`scene_*.xml`): flat and rough strips, hurdle sets,
stepped corridors and switchbacks, flyball/lean-plate wall turns, and `scene_REAL_`*
replicas of the physical course. `src/deploy_rl_policy/scripts/generate_*_scene.py`
regenerate their scenes parametrically.


| checkpoint                                                             | contents                                                         |
| ---------------------------------------------------------------------- | ---------------------------------------------------------------- |
| `models/L00-02-flat-baseline-delay-distillation-noised/model_14500.pt` | flat-baseline depth student, distilled with action delay + noise |
| `models/S00-03-corridor-distillation/model_3800.pt`                    | corridor depth student                                           |
| `resources/policies/ep_base_jit.pt`                                    | TorchScript Extreme Parkour base policy                          |
| `resources/policies/kp_kd_random_policy.pt`, `policy_with_45_obs.pt`   | upstream 45-obs walking policies (`rl_policy.py`, EKF demo)      |


## Further documentation


| Document                                                              | Covers                                                                                                                                                                        |
| --------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `[DEPLOY.md](DEPLOY.md)`                                              | ROS stack in depth: node pipeline, validated flags, the `env_class_eq_17` / contact-threshold / action-delay gotchas                                                          |
| `[deploy_mac/README.md](deploy_mac/README.md)`                        | macOS ROS 2 via RoboStack; Jetson RealSense over DDS; live encoder-prediction viewer                                                                                          |
| `[deploy_mac/runbooks/](deploy_mac/runbooks/)`                        | Step-by-step dry-run and real-run procedures                                                                                                                                  |
| `[README_distill.md](src/deploy_rl_policy/scripts/README_distill.md)` | The ROS-free checkpoint harness                                                                                                                                               |
| `[notes/analysis/](notes/analysis/)`                                  | Measured sim2real findings: action delay is part of the observation contract, camera pitch and drivetrain armature tolerances, why rosbag write times are useless for latency |


`base_velocity_estimator` is an EKF with a kinematic measurement model via
[Pinocchio](https://github.com/stack-of-tasks/pinocchio), for policies needing a base
velocity that isn't observable onboard. No Go2 URDF ships here — pass one with
`--ros-args -p urdf_path:=/abs/path/to/go2.urdf`.