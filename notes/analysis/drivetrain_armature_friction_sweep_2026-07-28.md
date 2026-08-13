# Drivetrain sweep: reflected inertia (armature) vs dry friction (frictionloss)

**Headline: the policy tolerates armature up to ~0.01 kg m^2 and falls off a
cliff at 0.02. frictionloss is a weak-to-null axis. And at the deploy-faithful
encoder cadence the camera-pitch penalty does NOT reproduce.**

Follow-up to the falsified torque-clip hypothesis
(`memory/project_torque_saturation_not_root_cause.md`), which left gearbox
friction and reflected inertia as the remaining actuator-fidelity candidates.

## Setup (deploy-faithful)

512 runs on the workstation (24 cores, 22 workers, EGL, ~4 min wall). Zero failed runs.
Grid: `armature {0, 0.01, 0.02, 0.04}` x `frictionloss {0, 0.04, 0.2, 0.5}` x
`cam_pitch {20, 7.6} deg` x 16 seeds (1001-1016).

- ckpt `D22-nextgoal0.5-orient-4090/model_9000` (the checkpoint the 07-27 real runs used)
- scene `scene_REAL_h15_t20_w120_spawn160.xml`, kp 35 / kd 0.6, vel 1.0,
  `env_class_eq_17=0`, contact 2.0 N (sim equivalent of the real 20)
- depth noise 0.15, stereo on
- **latency matched to the real stack** (`run_logs/real/2026-07-27_20-57-17/rl_policy.log`:
  "50 Hz actor, encoder every 5 ticks (10 Hz), depth budget=80 ms, proprio pad=16 ms,
  action_delay=0"):
  `--update_interval 5 --depth_latency_s 0.08 --proprio_latency_s 0.016 --action_delay 0`
  → runner reports `proprio=15 ms (3/4 substeps; requested 16), depth=80 ms (4 ticks)`.
  Note the proprio pad quantises to **15 ms** — 16 ms is not representable on the
  5 ms substep grid.
- **success = completed out-and-back**: `--stop_on_return_x --return_x 0.3
  --return_arm_x 2.0`, upright, and the return crossing inside the hurdle's
  width (|y| < 0.6).

Data: the deploy-faithful sweep rows (regenerable output, not shipped).

> **A first pass used `--update_interval 1` (60 Hz encoder) and is superseded**
> (`rows_encoder60hz_SUPERSEDED.json`). 60 Hz is neither what training used
> (`UPDATE_INTERVAL_DEFAULT = 5`) nor what the deploy stack runs (encoder every
> 5 ticks). Every conclusion below changed when the cadence was fixed — including
> the sign of the pitch effect. Encoder cadence is not a free knob.

## Reference values (why this grid)

Isaac training ran **armature = 0 and joint friction = 0**:
`legged_robot_config.py:438` has `armature = 0.`, `go2_config.py` does not
override it, `randomize_joint_friction = False`, and the Go2 URDF declares no
`<dynamics>` at all. `resources/go2/go2.xml` ships `armature="0.01"`,
`frictionloss="0.2"`, `damping="0.01"`.

## Armature: a tolerance cliff between 0.01 and 0.02

| armature | returned | success | torque-sat | mean tau_max | mean feet striking |
|---|---|---|---|---|---|
| 0 (training) | 99/128 (77.3%) | 77/128 (60.2%) | 8.7% | 72.4 Nm | 0.37 |
| **0.01 (go2.xml)** | 93/128 (72.7%) | **69/128 (53.9%)** | 11.0% | 76.3 Nm | 0.20 |
| 0.02 | 35/128 (27.3%) | 15/128 (11.7%) | 22.8% | 88.6 Nm | 0.72 |
| 0.04 | 12/128 (9.4%) | 7/128 (5.5%) | 32.9% | 87.1 Nm | 1.07 |

Adjacent-level Fisher exact on `success`:
- 0 vs 0.01: **p = 0.38 — indistinguishable**
- 0.01 vs 0.02: **p < 1e-4 — the cliff**
- 0.02 vs 0.04: p = 0.12

Pooled: `<=0.01` 57.0% vs `>=0.02` 8.6%, **p = 8.6e-13**. corr(armature, success)
= **-0.47**.

**`go2.xml`'s armature = 0.01 is fine** — statistically identical to the trained
0. The finding is a *robustness threshold*: the policy collapses once reflected
inertia reaches ~0.02 kg m^2. Torque saturation and peak torque both climb
across the cliff (8.7% -> 33%, 72 -> 89 Nm), and foot strikes on the hurdle
triple — the signature is "feet low and late", which is what the real runs looked
like.

## Frictionloss: weak at most

| frictionloss | returned | success |
|---|---|---|
| 0 (training) | 68/128 (53.1%) | 47/128 (36.7%) |
| 0.04 | 65/128 (50.8%) | 49/128 (38.3%) |
| 0.2 (go2.xml) | 56/128 (43.8%) | 41/128 (32.0%) |
| 0.5 | 50/128 (39.1%) | 31/128 (24.2%) |

0 vs >0: `returned` p = 0.10, `success` p = 0.28. Only the extreme contrast
0 vs 0.5 reaches nominal significance (36.7% vs 24.2%, p = 0.041), uncorrected
for the multiple comparisons in this table. corr = -0.115. Treat as a weak
negative trend, not a lever, and **not** a candidate root cause.

## Camera pitch: the penalty does not reproduce at the deploy cadence

| pitch | returned | success |
|---|---|---|
| 20 deg (trained) | 100/256 (39.1%) | 77/256 (30.1%) |
| 7.6 deg (hardware) | 139/256 (54.3%) | 91/256 (35.5%) |

7.6 deg is **not worse** — it is *better* on `returned` (p = 7.5e-4) and
statistically indistinguishable on `success` (p = 0.22). The best single cell in
the whole grid is pitch **7.6**, arm 0, fric 0.2 (12/16).

This contradicts `notes/analysis/sim2real_divergence_camera_pitch_2026-07-27.md` (6/6 at
20 deg vs 3/6 at 7.6 deg). The likely reason: that A/B was run at a 60 Hz encoder
cadence, off-distribution for a policy trained at `update_interval 5`. The
mount-pitch *measurement* in that note (7.6 deg, two resolutions, cross-validated
by camera height and far-clip fraction) is unaffected — only the claim that the
pitch error *causes* the failures is in doubt. **Re-run that A/B at
`update_interval 5` before acting on "re-tilt the hardware".**

## The hurdle is not the bottleneck — the return leg is

Across both sweeps every run crossed the hurdle centreline *over* the obstacle
(0 stalled, 0 walked around its 1.2 m width). Failures are on the way back:
lateral drift, mean |y| at the return crossing ~1.2 m. Score the return leg, not
foot-strike counts.

## Hardware friction / torque: identification attempted

Fitting `tau_est ~ a*tau_cmd + b*sign(dq) + c` per joint on the three 07-27 bags
does **not** cleanly identify Coulomb friction: `b` flips sign across joints and
the calves come out with torque *along* motion (-1.5 to -1.9 N.m), i.e. gait-phase
correlation. Quote no friction number from it.

The attenuation-free (TLS) slope of `tau_est` on commanded PD torque is
**0.87-0.97**, robust to `/lowcmd`<->`/lowstate` pairing (0.3 ms age) — the real
drivetrain tracks commanded PD torque near-ideally, so there is no large torque
shortfall for friction to hide in.

## Actions

1. **Leave `go2.xml` armature at 0.01.** It is indistinguishable from the trained 0.
2. **Measure the real reflected inertia.** This is now the top open question:
   the policy is fine below ~0.015 and dead above ~0.02, and we do not know
   which side the hardware is on. A swing-leg free-response test (suspend the
   robot, step one joint, fit the decay) would settle it.
3. **If hardware lands above ~0.015, fix it in training** — add armature + a DR
   range to the Isaac config and re-distill. Do not paper over it in `go2.xml`.
4. **Do not tune `frictionloss`.**
5. **Re-run the camera-pitch A/B at `update_interval 5`** before re-tilting the mount.
6. Never sweep at an encoder cadence other than 5 without saying so — it changed
   every conclusion here.

## Renders

Deploy-faithful A/B renders at seed 1001, fric 0.2:

| file | result |
|---|---|
| `p20_arm0.0_s1001.mp4` | **returns cleanly**, stop at x = 0.298, final (0.274, -0.406), min base_z 0.216 |
| `p20_arm0.04_s1001.mp4` | collapses at x = 3.79, y = +1.88, final_z 0.088, min base_z 0.055 |
| `p7.6_arm0.04_s1001.mp4` | collapses at x = 3.80, y = +2.26, final_z 0.072, min base_z 0.056 |

## Tooling added (`src/deploy_rl_policy/scripts/distill_runner.py`, uncommitted)

`--armature`, `--joint_frictionloss`, `--joint_damping` (override the 12 leg DOFs
post-load; -1 = keep XML), `--cam_pitch_deg` (rebuilds the `front_depth_cam`
frame; reproduces the XML quat to 1e-6), `--foot_log` (per-tick world foot
positions + base position, for geometric scoring).

workstation harness: payload at `~/roman_dt_sweep`, env
the deploy conda env (torch 2.11 cpu, mujoco 3.8), `MUJOCO_GL=egl`
(EGL throws a harmless `__del__` EGLError at teardown — parse RESULT, ignore rc).
