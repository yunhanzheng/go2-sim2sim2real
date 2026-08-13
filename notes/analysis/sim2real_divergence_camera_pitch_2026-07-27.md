# Why the 2026-07-27 real runs failed while sim2sim passed

**Root cause: the depth camera's physical mount pitch is 7.6° down, not the 20° that
training and every sim2sim validation render. All sim2sim testing was done on a camera
geometry the robot does not have.**

Checkpoint under test: `models/D22-nextgoal0.5-orient-4090/model_9000.pt`.
Real runs: `run_logs/real/2026-07-27_{17-33..21-01}`. Bags with depth: `18-48-46` (640x480x30),
`20-57-17` (480x270x60).

## 1. The measurement

Least-squares floor-plane fit on RAW depth, deprojected with the driver's own
`camera_info` intrinsics, base pitch removed via IMU, standing frames only:

| bag | resolution | mount pitch (still) | mount pitch (engaged) | recovered cam height |
|---|---|---|---|---|
| 20-57-17 | 480x270 | **7.60 ± 0.31°** (n=4314) | 8.33 ± 1.57° | 0.347 m |
| 18-48-46 | 640x480 | **7.28 ± 0.02°** (n=749) | 7.90 ± 1.76° | 0.402 m |

Two different resolutions and intrinsics give the same answer. The fit simultaneously
recovers a plausible camera height, which cross-validates it — a wrong pitch biases the
height too.

Two further independent confirmations, at the **policy-input** level (`/rl/depth_image`,
58x87, i.e. after the whole crop/resize/normalize pipeline):

- **far-clip fraction**: real **0.559–0.645** on both bags, standing and engaged.
  Sim renders **0.594 at 7.6°** and **0.384 at 20°**.
- **ground-fade row**: real ~56–59% of image height. 7.6° predicts 56%, 20° predicts 36%.

The revert in `c84bc99` ("standing bag frames confirm ~20 deg mount") rested on reading
the floor fade as ~30% from the top. The bags put it at 56–59%. `780ea91` was right.

## 2. The consequence (sim2sim, causal A/B)

Same checkpoint, same seeds, only the camera pitch differs. Scoring is geometric: a
foot is "striking" if it enters the hurdle box (FK from logged qpos, 22 mm foot radius).

On the **real obstacle** (h15/t20/w120, confirmed by operator and by depth measurement
at 0.152 m), with depth aligned to the real camera (60 Hz, noise 0.15, stereo on):

| camera pitch | clean jumps | ≥2 feet strike | mean feet striking |
|---|---|---|---|
| 20° (what sim2sim always rendered) | **6/6** | 0/6 | **0.00** |
| 7.6° (what the hardware is) | **3/6** | 1/6 | 0.67 |

On the h20/t25/w280 scene used for validation all day:

| condition | pitch 20° | pitch 7.6° |
|---|---|---|
| noise 0.5, 10 Hz (the standard sim2sim config) | 7/12 clean | 5/12 clean, **2/12 stall at the hurdle** |
| noise 0.15, 60 Hz (real-aligned) | 5/6 clean | 3/6 clean, 2/6 ≥2-feet |
| clean depth + stereo on | 5/6 clean | 5/6 clean |

At 7.6° the failure modes that appear are exactly the observed real ones: feet striking
the hurdle, stalling at it, and lateral veering (one seed to |y| = 1.25 m).

**Action: physically re-tilt the mount to the trained 20°.** `resources/go2/go2.xml` is
left at the measured 7.6° so sim2sim stays an honest predictor; set it back to
`0.34202 0 0.93969` the moment the hardware is re-tilted.

## 3. Secondary divergences found (real, measured — none of them the cause)

- **Training/sim depth degradation is far noisier than the real camera.** Real per-pixel
  temporal noise 0.038 m with 0.29% invalid/far flicker. The `TrainDepthDegrader` at the
  "reality-tuned" 0.5 gives **0.128 m (3.4×) and 17.3% far-flicker (60×)**; even 0.25
  gives 0.065 m / 15.5%. Benign direction — it made sim2sim look *worse* than reality
  (clean-depth-with-stereo scores 5/6 vs 7/12 at 0.5) — but it means the standard
  sim2sim noise setting is not calibrated to this camera. ~0.15 is the matched value.
- **SECOND DIVERGENCE (2026-07-28): the sim/training stereo shadow is ~3.7x larger than
  the real camera's.** Measured on the same 60x106 grid:
    real D435  : 2.70% invalid at 480x270  ->  **1.30%** on the 60x106 grid
    sim @0.060m: **4.86%** mean (p50 4.20, p95 11.6, max 17.4)
  Only 62/38 of real invalid runs show the left-right asymmetry of a true occlusion
  shadow, so genuine real shadow is smaller still (~0.44% at native res). Two causes:
  (a) the model is geometrically right but ALGORITHMICALLY pessimistic — it invalidates
  every occluded pixel while real block-matching recovers most; (b) real shadows are
  2-3 px at 480 wide and are diluted by the 4.4x downsample to 106 wide, whereas the sim
  renders natively at 106 so its shadows stay >=1 px.
  Sim invalidation vs baseline: .060->5.4% .030->2.8% .020->2.0% .015->1.5% .010->1.1%,
  so **0.0125 m reproduces the real rate** (new `EP_STEREO_BASELINE` override; default
  unchanged).
  CONSEQUENCE (deterministic runner, 40 hard scenes, paired):
    | stereo            | mean final base_x | reached >2.6 m |
    | 0.060 m (trained) | 2.492 m           | 20/40          |
    | 0.0125 m (REAL)   | 2.228 m           | 17/40          |
    | off (0%)          | 2.143 m           | 12/40          |
  Monotone: less shadow = worse. Real-matched vs trained: -0.263 m mean, median -0.341,
  28/40 worse, **sign test p = 0.0166**. So this is a REAL and statistically supported
  divergence — but MODERATE (~10% of progress), not the catastrophic one. Rank it below
  the camera pitch, which flips 6/6 into multi-foot strikes on the real obstacle.
  Also: with `depth_stereo:=off` + no noise on the hard cell, D22 m9000 scores 0/3 clean
  with 3/3 striking >=2 feet and m3000 0/3 with 2/3 — the real-run symptom exactly.
  Never run a sim2sim comparison with stereo off; it is not a "clean input" baseline.
- **The 60 Hz camera change was a genuine improvement, and sim2sim is now the pessimistic
  one.** Real depth age 81–96 ms — inside the trained 80–100 ms band on every tick.
  Sim's 10 Hz render delivers **89–129 ms**, i.e. outside it. Run sim2sim with
  `depth_rate_hz:=60` to match.
- **Validation scene ≠ real obstacle.** All sim2sim testing used h20/t25/**w280**; the real
  hurdle is h15/t20/**w120**. Benign on its own (6/6 at 20°). Twin scene added as
  `resources/go2/scene_REAL_h15_t20_w120_spawn160.xml`.
- **Torque limits actually MATCH; the real calves saturate at the sim/Isaec value.**
  (Corrects a first reading of this data that took `tau_est` 47.9 Nm at face value.)
  `tau_est` is current-derived, correlates 0.84-0.93 with the PD demand with regression
  slope 0.77-0.91 — i.e. it falls *short* of demand. Three of four calves cap at
  32.7-37.1 Nm, right at the URDF's 35.55 Nm, and FR_calf shows demand 42.2 Nm →
  delivered 34.5 Nm, which is saturation observed directly. Only RL_calf reports 47.9 Nm
  — the same foot whose force sensor shows 21% false contact, so that is very likely a
  per-motor current-sense calibration error, not a stronger actuator. Thighs peak at
  21.9-28.6 Nm against a 23.7 Nm limit (RR_thigh +20%, same caveat).
  Real consequence that IS solid: the calves genuinely saturate during the jump attempt,
  and MuJoCo's `ctrlrange` saturates at the same place — so sim reproduces this, and it is
  not a sim2real gap. Real joint tracking error p95 0.46 rad, calf up to 0.80 rad, is
  largely this saturation plus commanding past the mechanical stop.

## 4. Explicitly cleared (checked, not the cause)

- **Contact obs / threshold 20 N**: well calibrated. Against FK-derived ground truth,
  false-contact 3.6–6.2% (one bad foot, RL, at 21%), miss rate 9–14%. The higher real
  contact duty (0.72 vs sim 0.43) is a *consequence* of not jumping, not a sensor fault.
- **Proprio noise**: cleared, but NOT for the reason first written here. The original
  claim ("real is far quieter than DR in every channel") was measured while STANDING and
  is WRONG — see section 6. In the impact regime dof_vel/gyro are UNDER-modelled 8.6x/4.5x.
  It is cleared because a 120-run sweep across a 17x span of proprio noise is flat, not
  because the levels match. (Accelerometer magnitude reads 9.42 vs 9.81 m/s², a 4%
  scale/bias error; affects only the gravity vector's length, not its direction.)
- **Control path**: `/lowcmd` gaps p50 20.0 ms, p99 23 ms, zero gaps >50 ms in the jump
  runs. The 2026-06-19 stall class is gone. DDS delivery target→low-level 0.25 ms mean.
- **kp/kd**: 35/0.6 identical on both paths; the 50/1.0 entries are the stand controller.
- **|dlat|**: real 2.1–3.8, sim 3.2–3.6 — comparable, depth is in-distribution.
- **Base-velocity obs**: both paths use the learned estimator (`use_sim_truth_vel`
  defaults to 0); no ground-truth leak in sim.

## 6. Proprio noise: measured, swept, NULL (2026-07-28)

Isaac DR vs the real robot, spectral noise floor (Welch 150-250 Hz, per-sample std,
RAW units), 39 windows / 21 bags:

| | dof_pos | dof_vel | gyro |
|---|---|---|---|
| July 27 (hurdle jumping, \|dq\|rms 4.1) | 0.00235 | 0.2496 | 0.1307 |
| June 17/19 (gentle, \|dq\|rms 1.0)      | 0.00049 | 0.0443 | 0.0137 |
| Isaac DR std (= ns/sqrt3)               | 0.00577 | 0.0289 | 0.0289 |

It scales with motion violence, so it is impact-excited structural ringing rather than a
fixed sensor floor. In the impact regime dof_pos is 2.5x OVER-modelled while dof_vel and
gyro are 8.6x / 4.5x UNDER-modelled.

Swept anyway (deterministic runner, 24 real-obstacle scenes paired per arm, pitch pinned
20 deg): clean 2.568 m / isaac 2.445 / july_half 2.521 / july 2.469 / july_2x 2.581, with
14/11/12/14/13 of 24 reaching x>2.6 m. A 17x span of dof_vel noise moves the mean inside a
5% band with no trend and every p >= 0.54; the NOISIEST arm scored best.

CONCLUSION: leave `noise_scales` alone. The mismatch is real but behaviourally
irrelevant, and a retrain buys nothing. Also note sim2sim had never injected proprio
noise at all, so every earlier sweep was implicitly zero-noise — and zero is
indistinguishable from trained DR, so none of them are invalidated.

## 5. Reproducing

```
# pitch A/B harness (edits go2.xml between conditions, restores nothing — check git)
run_logs/dryrun/pitch_ab_2026-07-27/       # noise 0.5, 10 Hz, h20/t25/w280
run_logs/dryrun/pitch_ab_nonoise_2026-07-27/   # noise 0, stereo OFF  <- the 0/6 collapse
run_logs/dryrun/pitch_ab_stereoclean_2026-07-27/ # noise 0, stereo ON
run_logs/dryrun/realaligned_2026-07-27/    # noise 0.15, 60 Hz, h20/t25/w280
run_logs/dryrun/realtwin_2026-07-27/       # noise 0.15, 60 Hz, REAL h15/t20/w120
```
