# Distillation policy sim-to-sim verification

`distill_runner.py` is a self-contained, ROS-free MuJoCo harness that loads the
extreme-parkour distillation (student) policy and runs it on a flat scene. Use it
to verify that a student checkpoint walks correctly under the deploy-side PD gains
before flashing it onto the real Go2.

## Usage

```bash
cd $REPO
python src/deploy_rl_policy/scripts/distill_runner.py --seconds 8 --lin_vel_x 1.3
```

The command is bimodal in training (zero, or a walk band above ~1 m/s), so the
`--lin_vel_x 0.5` the script still defaults to leaves the robot standing in place.
Verify at **1.3**. On `scene_flat` the shipped student holds a clean walk for ~8 s and
then yaws off and falls, so for a longer run use a course scene, e.g.
`--scene resources/go2/scene_two_hurdles_wall_turn.xml`.

That run uses `resources/go2/scene_flat.xml`, the shipped student checkpoint
`models/L00-02-flat-baseline-delay-distillation-noised/model_14500.pt`, and EGL for
offscreen depth rendering (auto-set when `--depth_mode render`).

Useful flags:

| flag | default | notes |
| --- | --- | --- |
| `--ckpt` | `models/L00-02-…/model_14500.pt` | swap to another distillation checkpoint without code changes |
| `--ep_root` | `../go2-flyball-parkour` (or `$EP_ROOT`) | training-repo root added to `sys.path` for `rsl_rl` imports |
| `--scene` | `resources/go2/scene_flat.xml` | flat MJCF scene including `go2.xml` with the `front_depth_cam` mounted on `base_link` |
| `--lin_vel_x` | `0.5` | forward command in m/s; below ~1.0 the student stands still — pass `1.3` |
| `--seconds` | `10.0` | sim duration |
| `--viewer` | off | opens `mujoco.viewer.launch_passive`; over SSH prefix the command with `vglrun` |
| `--action_delay {0,1}` | auto-detected from checkpoint | flips to 1 once the final delay-trained student is in |
| `--depth_mode {render,zeros}` | `render` | `zeros` skips the EGL render for fast smoke tests |
| `--update_interval` | `5` | depth-encoder cadence (every N policy ticks; matches training) |
| `--record` | off | write an MP4 of `follow_cam` off-screen (EGL, no X11) at `--record_width`×`--record_height` |

## Pass criteria

The script prints a one-liner every 1 s of sim and a final summary. The run is a
"verified" walk if all of:

- `base_z` stays in `[0.20, 0.40]` m (no fall, no stuck-flat)
- `vx_avg(last 2 s)` is within ±0.2 m/s of `--lin_vel_x`
- `τ_peak` < 35 Nm (Go2 motor headroom)
- `d_lat|` is finite and bounded between successive ticks

A second smoke run with `--lin_vel_x 0.0 --seconds 5` should keep the robot in
place (drift < 0.3 m).

If you change `--depth_pitch` or the terrain, re-confirm `d_lat|` stays bounded — a
sudden jump usually means a depth-pose or normalization bug.

## What the harness deliberately does not do

- It does not drive the real Go2. The ROS path does that: `rl_distill_policy.py` (or
  `distill_fused_policy.py`) for the distillation student, `rl_policy.py` for the
  45-obs walking policy. This harness shares the same pipeline via
  `distill_common.py`, so a checkpoint that passes here is worth launching there.
- It does not apply the ROS stack's latency model. Action delay and proprio latency
  are applied differently, which is enough to re-rank checkpoints — see
  [`notes/analysis/standalone_vs_ros_latency_divergence_2026-07-28.md`](../../../notes/analysis/standalone_vs_ros_latency_divergence_2026-07-28.md).
- It defaults to a flat scene. Pass `--scene` any of the terrain scenes
  (e.g. `resources/go2/scene_flyball70.xml`); the proprio/actor path stays the same.
