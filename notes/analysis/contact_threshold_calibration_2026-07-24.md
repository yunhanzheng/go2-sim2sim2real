# Per-foot CONTACT_THRESHOLD calibration — 2026-07-24

Derives evidence-based per-foot raw-sensor thresholds (FR,FL,RR,RL) for the
`CONTACT_THRESHOLD` env var (`deploy_mac/run/run_real_remote.sh` →
`rl_distill_policy.py` / `depth_encoder_node` → `distill_common.compute_contact_filt`,
proprio[49:53]) from all real Go2 foot-force data on disk. Training/sim uses a
clean 2.0 N net-contact-force threshold (Isaac / MuJoCo `CONTACT_FORCE_THRESHOLD_N`);
the real Unitree `foot_force` values are **uncalibrated raw units** with nonzero,
per-foot airborne offsets, so the sim value does not transfer.

## 1. Data inventory

What exists on disk (checked 2026-07-24):

| Source | Contents | Foot forces? |
|---|---|---|
| `run_logs/real/*/lowstate.csv` (22 dirs, 06-15…07-01) | joint q/dq, engaged, cmd_kp @50 Hz (`deploy_mac/analysis/monitor_deploy.py` schema) | **NO** — the CSV logger does not record `foot_force` |
| `run_logs/real/*/bag` | — | none exist (`full_log (rosbag): 0` in all metas) |
| `trash/repo_cleanup_2026-06-29/run_logs/real/` (21 dirs, 06-16…06-19) | 19 rosbags, ~28 GB total, `/lowstate` @500 Hz incl. `foot_force[4]`, `/policy_engaged`, motors | **YES** — the only raw real force data available |
| `trash/repo_cleanup_2026-06-29/run_logs/real_analysis/{summary,detailed}.json` | 2026-06-19 forensics outputs (whole-run mean/min/max per foot, contact rates @17/20/30/50) | derived stats |
| `media/analysis/foot_forces_stance_vs_airborne.png` | 40 s, 500 Hz lift-calibration plot (`log_foot_forces.py` session; source CSV `/tmp/foot_forces.csv` no longer exists) | plot only |
| `run_logs/dryrun/*/**.foot_forces.csv` (sim sweep rows not shipped) | MuJoCo sim foot forces (Newtons) by FSM state | sim only |

Consequences:

- **All real force evidence is from 2026-06-15…19** (T000-00 era, one robot, one
  venue). The 2026-06-29 and 2026-07-01 hardware sessions recorded **no foot
  forces at all** (no bag, no force columns in `lowstate.csv`).
- Bags analysed here (subsampled 500→100 Hz): the 8 active runs
  (`2026-06-17_17-30-34`, `2026-06-19_16-38-30/16-40-15/16-42-26/16-46-16/16-48-44/16-53-20/16-59-59`)
  plus the quiet-stand segments of `16-36-29` and `19-21-08`. Engaged walking
  total ≈ 358 s; quiet standing ≫ 1000 s.
- Prior per-foot attempts already exist in metas: `10,20,20,14`
  (2026-06-19 19:17/19:21 runs) and `10,20,20,20` (an earlier student run report).

Segmentation method: engaged = `/policy_engaged` state; quiet standing =
not-engaged ∧ ‖dq‖<0.5 ∧ ‖q−STAND‖<0.85; laying analogous with the LAY pose
(poses from `monitor_deploy.py`). Swing/stance inside engaged windows split
per foot by 1-D 2-means. Extraction script: sqlite-direct bag read, same
pattern as `tools/analyze_real_bags.py`.

## 2. Per-foot statistics (raw units, FR,FL,RR,RL)

### Airborne baseline (offset + noise)

Only one direct measurement exists: the lift-calibration plot
`media/analysis/foot_forces_stance_vs_airborne.png` (values read off the plot;
signal is ~integer-quantized and near-constant while hoisted):

| | FR | FL | RR | RL |
|---|---|---|---|---|
| airborne offset | 3–5 | 12–13 | 10 | 12–13 |

Matches the 06-19 forensics note. FR's offset is ~3x smaller than the other
three — a global threshold cannot be margin-equal across feet.

### Quiet-standing stance load (per-run p50 of the quiet-stand segment)

| run | FR | FL | RR | RL |
|---|---|---|---|---|
| 06-17_17-30-34 | 34 | 40 | 28 | 23 |
| 06-19_16-36-29 | 37 | 45 | 40 | 34 |
| 06-19_16-38-30 | 34 | 44 | 38 | 33 |
| 06-19_16-40-15 | 35 | 45 | 36 | 33 |
| 06-19_16-42-26 | 37 | 49 | 38 | 33 |
| 06-19_16-46-16 | 34 | 45 | 35 | 32 |
| 06-19_16-48-44 | 35 | 44 | 36 | 32 |
| 06-19_16-53-20 | 30 | 45 | 38 | 30 |
| 06-19_16-59-59 | 29 | 46 | 38 | 29 |
| 06-19_19-21-08 | **21** | 37 | 35 | **20** |
| median | 34 | 45 | 37 | 32 |

Persistent asymmetry: FL carries the most (~45), RL the least (~32), and FR/RL
drift strongly between sessions (FR 21–37, RL 20–34). In the 19-21-08 session
the **global threshold 20 classified RL as no-contact in 95% and FR in 42% of
quiet-standing samples** — a standing robot observed as a two/three-legged
stance, the exact failure mode from the 06-19 forensics.

### Engaged walking (per-foot percentiles pooled per run)

Engaged p10 / p50 ranges across the 8 active runs:

| | FR | FL | RR | RL |
|---|---|---|---|---|
| p10 | 9–33 | 17–40 | 11–30 | 8–18 |
| p50 | 16–41 | 42–52 | 26–41 | 20–29 |

Contact fraction while engaged at the deployed global 20:
FR 0.41–0.94 (0.41 in 16-59-59 — the run with the documented persistent FR
dropout), FL 0.88–0.98, RR 0.73–0.95, RL 0.46–0.87. The spread across feet at
a fixed threshold is the observation error the policy sees.

Note the sensor is slow/filtered: during trot, swing-phase readings only dip to
~8–18 (not to the 3–13 airborne offsets), so the swing/stance modes are much
closer together in walking than a hoisted recording suggests. Also one anomaly:
16-42-26 RL read 78+ in "stance" (robot likely handled/harness-loaded) —
excluded from the anchors below.

### Sim reference (MuJoCo dryrun, Newtons)

From `run_logs/dryrun/2026-07-24_12-59-51-D09_.../*.foot_forces.csv`: swing = exactly
0 N, quiet stance 35–47 N per foot → the 2.0 N sim threshold sits in a clean,
symmetric gap. Real data has no such gap; per-foot placement is required.

## 3. Recommended thresholds

Anchors per foot: `a` = airborne max (plot), `s` = worst-session quiet-stand
p50 (19-21-08). Placing every foot's threshold at the same normalized position
λ = (t − a)/(s − a) ≈ 0.42 of its usable band:

| | FR | FL | RR | RL |
|---|---|---|---|---|
| a (airborne max) | 5 | 13 | 10 | 13 |
| s (worst stand p50) | 21 | 37 | 35 | 20 |
| **recommended t** | **12** | **23** | **20** | **16** |

```
CONTACT_THRESHOLD=12,23,20,16
```

Checked against the data:

- Quiet standing: false no-contact ≈0–1% in the clean stand segments
  (16-36-29, 19-21-08), 5–16% in segments contaminated by load shifts —
  versus up to 42%/95% (FR/RL) at the current global 20.
- Engaged walking: per-foot contact fractions become far more uniform, e.g.
  16-59-59 FR 0.41→~0.79, 06-17 RL 0.50→~0.62; FL/RR nearly unchanged.
- Airborne margin: +7 (FR), +10 (FL), +10 (RR), +3 (RL) units above the
  measured airborne maxima. The prior attempt `10,20,20,14` sat only **1 unit**
  above RL's airborne reading — high false-positive risk when hoisted/swinging.

Sensitivity: FR/FL/RR are robust to ±2 units (standing-FN and engaged-duty move
by only a few percent). **RL is not**: its whole usable band in the worst
session is 13→20 (7 units), so ±1 unit matters and no RL value is simultaneously
safe against airborne false-positives and light-stance false-negatives. RL (and
the FR session drift, 21↔37 stance median) is why this recommendation is
provisional pending the recapture below.

Caveats:

- Raw uncalibrated units; no vendor scale, no cross-robot validity.
- Airborne anchors come from **one** 40 s lift on **one** day, surviving only
  as a PNG; drift of the airborne offset across sessions is unmeasured.
- All walking data is T000-00-era gait (June 2026); current checkpoints load
  feet differently in unknown ways — and their sessions logged no forces.
- The threshold interacts with the one-step OR filter
  (`compute_contact_filt`); all fractions above are per-sample raw crossings.

## 4. Follow-up capture protocol (required to make this rigorous)

1. **Hoisted airborne + stand recording, every hardware session** (existing
   tool, subscribe-only, safe):

   ```bash
   source deploy_mac/env.sh
   python deploy_mac/analysis/log_foot_forces.py --secs 90 --out run_logs/real/$(date +%F_%H-%M-%S)_foot_calib.csv
   ```

   Timeline: 0–20 s quiet stand (all feet loaded) → 20–60 s hoisted, feet fully
   airborne, gently swing each leg once → 60–90 s set down, quiet stand. Keep
   the CSV in the run dir (not `/tmp` — the previous one was lost). The script
   auto-reports the airborne segment; the per-foot recipe is then
   `t_i = round(airborne_p99_i + 0.42 * (stand_p50_i − airborne_p99_i))`.
2. **Log forces during runs again.** Either set `FULL_LOG=1` (rosbag) for at
   least one run per session, or extend `monitor_deploy.py` to add four
   `foot_force` columns to `lowstate.csv` (4 columns, no bandwidth concern).
   Without this, every new checkpoint deploy is force-blind.
3. **Validate contact bits against video** on a hoisted treadmill/step-in-place
   test: per-foot contact fraction should match the visible duty factor.
4. Longer term (already in the 06-19 forensics): per-session baseline
   subtraction + hysteresis, or retrain with a Unitree force-sensor model.
   `LowState.foot_force_est` is also recorded in the bags and unanalysed.

## Reproduction

Extraction script (sqlite bag read → per-segment per-foot percentiles JSON) was
run from the session scratchpad; it is disposable and reproducible from this
note's methodology (§1–2). Inputs: `trash/repo_cleanup_2026-06-29/run_logs/real/*/bag`,
`media/analysis/foot_forces_stance_vs_airborne.png`,
`trash/repo_cleanup_2026-06-29/run_logs/real_analysis/detailed.json`.
