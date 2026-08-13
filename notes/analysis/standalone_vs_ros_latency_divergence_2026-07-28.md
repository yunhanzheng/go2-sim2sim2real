# Why the ROS stack outperformed the standalone runner on the same student

**Root cause: the standalone evaluators applied the policy's joint target in the same
tick it was computed. The ROS/real stack applies it ~18.4 ms later, and every
cliffside student TRAINS with a latched 20 ms action delay. Zero loop delay is
off-distribution in the "too fresh" direction, and it makes the student faceplant at
the hurdle.** Nothing else about the two pipelines differs materially.

Checkpoint under test throughout: `models/D22-nextgoal0.5-orient-4090/model_9000.pt`,
wall scene `scene_leanplate60_wall3m425_1hurdle_h20_t25_spawn160_gap170.xml`,
camera pitch pinned at the trained 20°, depth noise 0.5, stereo on, kp 35 / kd 0.6.

## 1. The real delay, measured

Every 2026-07-27 run writes `latency_actor.tsv` (per tick: the lowstate frame used,
tick start, encode, inference, pad sleep, target publish) and `latency_low_level.tsv`
(target receipt → `/lowcmd` publish). Joining them on `target_seq` gives the true
host-side loop delay: **the proprio frame the actor used → `/lowcmd` on the wire.**

| run | n | p50 | p90 | p99 | max |
|---|---|---|---|---|---|
| 2026-07-27_18-08-25 | 1765 | 18.36 | 19.03 | 21.05 | 87.92 |
| 2026-07-27_18-13-25 | 460 | 17.64 | 18.80 | 21.81 | 60.09 |
| 2026-07-27_18-40-26 | 779 | 18.41 | 18.97 | 22.38 | 74.88 |
| 2026-07-27_18-48-46 | 949 | 18.69 | 18.94 | 19.36 | 45.37 |
| 2026-07-27_20-50-09 | 927 | 18.46 | 18.94 | 19.92 | 47.34 |
| 2026-07-27_20-57-17 | 405 | 17.88 | 18.91 | 19.27 | 53.75 |
| 2026-07-27_21-01-39 | 231 | 18.30 | 18.84 | 27.66 | 52.53 |
| **pooled** | **5516** | **18.40** | **18.94** | **20.51** | — |

It is this tight because the enforced 16 ms pad dominates (actor sleep p50 15.7–17.4 ms);
compute is 0.4 ms and the target→`/lowcmd` hop is 0.1 ms. Two segments remain
unobservable from host logs — sensor sampling → host receipt (~4–8 ms, measured
earlier from tick mapping) and `/lowcmd` → motor application (firmware). **Real total
is therefore ≥18.4 ms and most plausibly 23–27 ms.**

Training, for comparison: `action_delay_start_latched` + `--delay`, documented in
`go2_cliffside_distill_config.py` as "REQUIRED for distillation so the student never
trains on undelayed actions" → a fixed 1-step / **20 ms** delay from the first DAgger
batch. `randomize_proprio_latency = False`, deliberately, so the 20 ms is the whole
loop-delay model and must not be double-counted.

## 2. The divergence

| | standalone (as swept before today) | ROS sim2sim | real | training |
|---|---|---|---|---|
| action applied | **same tick (0 ms)** | 16 ms pad, `action_delay=0` | 18.4 ms measured | 20 ms latched |
| depth age at use | ~100 ms (`depth_buffer[-2]`) | 80–94 ms (age gate) | 81–96 ms | 80–100 ms |

Note the depth column: `--depth_latency_s 0` never meant "fresh depth". The runner
falls back to Isaac's legacy `depth_buffer[-2]`, which at a 10 Hz encoder cadence is
already ~100 ms stale. Depth was never the divergence.

## 3. Decomposition — 432 runs, 24 scenes × 3 seeds, paired per cell

`run_logs/dryrun/lat_decomp_2026-07-28` (local, mujoco 3.9.0). "early" = fell within 8 s,
i.e. at the hurdle; "upright" = fraction of the run with base_z > 0.20 m.

| arm | fell | early | mean xmax | upright | median t_fall | p(fell) vs zero |
|---|---|---|---|---|---|---|
| zero latency (old default) | 42/72 | 36/72 | 3.26 | 0.303 | 3.5 s | — |
| 80 ms depth latency only | 48/72 | 43/72 | 3.28 | 0.291 | 3.5 s | 0.39 **NULL** |
| 15 ms action delay | 20/72 | 5/72 | 3.54 | 0.685 | 12.4 s | **0.0005** |
| 20 ms action delay | 25/72 | 14/72 | 3.63 | 0.665 | 4.2 s | **0.0076** |
| 15 ms + 80 ms depth (= the ROS config) | 27/72 | 13/72 | 3.38 | 0.655 | 9.9 s | **0.024** |
| 20 ms + 80 ms depth (= training) | 24/72 | 11/72 | 3.70 | 0.697 | 10.8 s | **0.0064** |

## 4. Dose–response — 432 runs on the workstation, depth held at 80 ms

`run_logs/delay_sweep_2026-07-28` on the workstation (mujoco 3.8.0, 20 workers, 11 min).

| loop delay | early falls | p vs 0 ms | upright | Δupright p | mean xmax |
|---|---|---|---|---|---|
| 0 ms | 35/72 | — | 0.335 | — | 3.22 |
| 15 ms | 11/72 | 0.00007 | 0.648 | <1e-5 | 3.38 |
| 20 ms (trained) | 12/72 | 0.00029 | 0.675 | 0.00017 | 3.60 |
| 25 ms (real-measured) | 17/72 | 0.0029 | 0.638 | 0.0016 | 3.52 |
| 30 ms | 19/72 | 0.0037 | 0.552 | 0.00053 | 3.51 |
| 35 ms | 34/72 | 1.00 | 0.402 | 0.024 | 3.38 |

**Inverted U centred on 15–25 ms** — the trained 20 ms and the measured real 18–25 ms
sit in the flat optimum; by 35 ms the entire benefit is gone. Too fresh and too stale
both break it, which is the signature of a policy tuned to its own loop delay.
Cross-machine agreement is good on matched configs (15 ms + 80 ms depth: 27 falls
local vs 29 workstation; 0 ms + 80 ms: 48 vs 39) — but note `fell` over the full 45 s is a
noisy endpoint (the robot often survives the hurdle and falls later at the plate);
`early` and `upright` are the robust ones and agree on both machines.

## 5. Three-way confirmation — is the ROS stack really where the delayed arms are?

`run_logs/dryrun/threeway_2026-07-28`: 8 independent ROS episodes (non-deterministic) vs
8 seeds per standalone arm, one scene.

| stack | early falls | fell | upright | mean xmax |
|---|---|---|---|---|
| **ROS stack** | **1/8** | 2/8 | **0.809** | 3.53 |
| standalone 0 ms | 4/8 | 4/8 | 0.578 | 2.70 |
| standalone 15 ms | 2/8 | 3/8 | 0.758 | 3.29 |
| standalone 20 ms | 2/8 | 2/8 | 0.786 | 3.37 |
| standalone 20 ms + 80 ms depth | 1/8 | 5/8 | 0.742 | 3.39 |

The ROS stack sits with the delayed arms on every metric; zero-delay standalone is the
outlier. n=8, so this confirms rather than measures.

## 6. Mechanism

Measured during **healthy pre-hurdle walking** only (x < 1.4 m, base_z > 0.25 m), so it
is a cause and not a consequence of having already fallen — 72 runs per arm:

| | mean \|Δaction\| per tick | power above 12.5 Hz |
|---|---|---|
| zero delay | 0.516 | 5.79 % |
| 15 ms delay | 0.379 | 3.98 % |

Removing the trained lag adds phase lead: the loop chatters 36 % more per tick and
carries 46 % more high-frequency power, then diverges at the highest-gain moment —
hurdle touchdown.

## 7. Cleared (checked, not the cause)

- **Depth latency**: null on both machines (p = 0.39). Kept as an 80 ms default for
  fidelity only.
- **`foot_force` int16 quantization** (ROS `LowState` carries `int(round(F))`, so the
  2 N contact gate can flip): only **0.010 %** of foot-samples land in the (2.0, 2.5] N
  window where ROS and standalone disagree. Negligible.
- **Timestep / decimation**: `mujoco_simulator` forces `opt.timestep = 0.005` and the
  wall scene declares 0.005; both paths use decimation 4. Identical. (Watch this for
  scenes with no `<option timestep>`: ROS forces 0.005, standalone would take MuJoCo's
  0.002 default.)
- **GRU / history warm-up**: the fused policy logs "`/policy_engaged=True` — reset
  encoder GRU + actor episode state" at engage, and `depth_sim_node` gates rendering on
  engagement, so ROS starts as cold as standalone. History ring is zeros in both.
- **Initial pose**: ROS air-snaps to z = 0.42 and settles 1 s at engage; standalone does
  the same via `--init_z 0.42 --pre_settle_seconds 1.0`.
- **Real-time factor**: the ROS recorder captured 450 frames in 9.02 s = 49.9 Hz, so the
  sim is not running slow and diluting substeps per tick (the "~0.82× realtime" note in
  `render_sim2sim_noros.py` is stale).
- **Action→target math, clip, scale, PD**: identical code paths.
- **The 53-D proprio itself**: dumped from both stacks over a 300-tick engaged STANDING
  window (`run_logs/dryrun/obs_parity_2026-07-28`, ROS via `proprio_log:=`). The 12 constant
  slots (`zero_dyaw`, `delta_yaw`, `delta_next_yaw`, both `zero_cmd`, `lin_vel_x`, the
  env-class one-hot, all four contact bits) match EXACTLY; 32/53 channels agree to
  <0.01. The largest gaps are `last_action` (0.35) and `dof_pos` (0.08), i.e. the two
  stacks settle into slightly different standing postures — a state difference, not a
  scale/sign/offset bug. ROS is markedly jigglier while standing (act sd 0.21 vs 0.025,
  dof_pos sd 0.05 vs 0.003), consistent with the asynchronous 50 Hz timer plus the pad.

## 8. Consequences

- **The 2026-07-28 student sweeps ran undelayed** and understated the checkpoints —
  the D22 stage-1/2 sweeps (`student_sweep_newdepth`, `student_sweep_wall`) and the
  out-and-back wall sweep (`student_return_wall`), none of which passed a latency flag.
  The teacher sweeps passed `--action_delay 1` explicitly and are unaffected, so
  student-vs-teacher comparisons across those two sets were never like-for-like.
- **This does NOT overturn the 2026-07-27 "~2× ROS penalty".** That sweep ran standalone
  *with* the 16 ms pad and 80 ms depth (per its own note), i.e. already in-distribution,
  and still found D22 m3000 at 8/10 standalone vs 4/10 through ROS. So the two
  observations are consistent once the standalone config is pinned down: against a
  properly delayed standalone, ROS is *worse* (still unexplained, separate question);
  against a zero-delay standalone — which is what the 07-28 sweeps and the renders that
  prompted this investigation used — ROS looks *better*. Always record the latency flags
  with the result; the sign of the standalone↔ROS gap depends on them.
- **The resemblance between zero-delay standalone and the real robot was a
  coincidence, not validation.** Real failed because the camera mount is 7.6° instead
  of the trained 20° (see `sim2real_divergence_camera_pitch_2026-07-27.md`); standalone
  failed because its loop had no delay. Same symptom — feet into the hurdle — from two
  unrelated causes. Do not tune sim2sim to "match reality" on that basis.
- **`returned` is still a false metric for these students**: 32-D no-yaw, `prop[6:8] ≡ 0`,
  so nothing anchors heading. The "student walks back to x = 0" seen in the ROS renders
  is a heading random-walk after the plate plus a constant forward command, not a wall
  turn. Standalone runs reach |yaw| = 180° just as often.

## 8b. It mis-RANKED the students, not just the level (672 runs, workstation)

`<workstation>:.../run_logs/rerank_delay_2026-07-28` — 7 checkpoints × 24 scenes × 2 seeds at 0 ms
and at the measured 25 ms. "clear" = did not fall within 8 s, out of 48.

| ckpt | clear @ 0 ms | upright @ 0 ms | clear @ 25 ms | upright @ 25 ms |
|---|---|---|---|---|
| new_m4000 | 19/48 | 0.304 | 36/48 | 0.600 |
| **new_m5000** | 21/48 | 0.229 | **41/48** | **0.660** |
| new_m6000 | 14/48 | 0.176 | 26/48 | 0.391 |
| new_m7000 | 18/48 | 0.234 | 33/48 | 0.610 |
| new_m8000 | 12/48 | 0.225 | 33/48 | 0.498 |
| ref_m3000 | 26/48 | 0.464 | 28/48 | 0.558 |
| ref_m9000 | **27/48** | 0.300 | 35/48 | 0.633 |

Ranking by hurdle clears:
- **0 ms:** ref_m9000 · ref_m3000 · new_m5000 · new_m4000 · new_m7000 · new_m6000 · new_m8000
- **25 ms:** new_m5000 · new_m4000 · ref_m9000 · new_m7000 · new_m8000 · ref_m3000 · new_m6000

The zero-delay harness put both *reference* checkpoints on top and made every
new-depth-model checkpoint look like a regression. At the correct latency
`D22-newdepthmodel-4090/model_5000` is the **best** student by a clear margin (41/48 vs
ref_m9000's 35), and ref_m3000 falls from 2nd to 6th. The gain from the delay is
strongly checkpoint-specific (new_m8000 12→33, ref_m3000 26→28), so this is NOT a
uniform handicap that cancels in the ranking. **Any student selection made from a
zero-delay sweep must be redone.**

## 9. Defaults changed (commit f2617b9)

- `distill_runner.py`: `--action_delay` **1**, `--depth_latency_s` **0.08**,
  `--proprio_latency_s` stays 0.
- `lidar_runner.py`: `--action_delay` **1**.
- `render_sim2sim_noros.py`: new `--action_delay` (1) and `--depth_latency_ticks` (4).
- Both runners now warn when `--action_delay 1` and `--proprio_latency_s` are combined:
  they model the same physical delay. Combine them only to reach a specific total
  (e.g. 25 ms = `--action_delay 1 --proprio_latency_s 0.005`); the banner prints the sum.
- Pass `--action_delay 0` for pre-6fb4d46 policies (T000 / L00), which trained undelayed.

## 10. Still open

`render_sim2sim_noros.py` feeds the **clipped** action into the depth encoder's proprio
slot (`prop[37:49]`) while the actor gets the raw one — the F03 split that
`distill_fused_policy` fixed and that `distill_runner` never had (both feed raw, as
training does). Only bites when |action| > 4.8, i.e. during aggressive jumps. Left
unchanged for now because the pitch / proprio / stereo A-Bs in the notes were all run
against it and are paired, so a constant quirk largely cancels — but any new absolute
number from that runner is suspect until it is fixed.

## 11. Reproducing

```
run_logs/dryrun/lat_ab_2026-07-28/         # 36-run pilot
run_logs/dryrun/lat_decomp_2026-07-28/     # 6-arm decomposition, 432 runs
run_logs/dryrun/threeway_2026-07-28/       # 8 ROS episodes + 4 standalone arms
run_logs/dryrun/lat_renders_2026-07-28/    # 0 ms vs 15+80 ms, wall scene, seed 102
run_logs/dryrun/workstation_delay_renders{,2}/  # 0 ms vs 25 ms pairs rendered on the workstation
<workstation>:.../run_logs/delay_sweep_2026-07-28 # 0/15/20/25/30/35 ms dose-response, 432 runs
```
