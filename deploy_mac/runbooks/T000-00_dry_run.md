# Dry run — latest policy (T000-00 step-corridor distill, no-yaw-head)

Stationary verification of the **depth → encoder → policy** chain on the real robot
**before** any `/lowcmd` walk. No `low_level_ctrl` is started, so nothing publishes
`/lowcmd` → **the robot does NOT move from our stack.** Use this to read `|dlat|`,
the depth image, and the policy's velocity response while the dog is hoisted/laying.

Policy: latest checkpoint in `models/T000-00-step-corridor-distill/` (32-D
no-yaw-head; loads `strict=True` against `go2-flyball-parkour`@`L00-flatstep`).

> ⚠️ **Why this run exists.** The D01-99 dry run parked at `|dlat| ≈ 3.6` on the real
> RealSense (~2× sim, past the crash value) — off-distribution depth. T000-00 was
> trained on `S01-teacher-hurdles`, which does **NOT** include the Parkour-in-the-Wild
> depth degradation (that's on `L00-flatstep`, awaiting a re-distill), so the same risk
> applies. Clear this dry run (|dlat| near the sim band, clean depth) **before** running
> `deploy_mac/runbooks/T000-00_real_run.md`. See [[project_d01_real_dryrun_dlat]].

## Prereqs

- Robot **hoisted / laying**, facing a corridor or wall (NOT an open room — open rooms
  far-saturate the depth and inflate `|dlat|`).
- **Sport mode OFF** — otherwise the Go2 factory controller moves the robot from the
  remote, even though our stack does not:

  PYTHONPATH=$HOME/go2/unitree_sdk2_python $HOME/go2/venv/bin/python \
   $HOME/go2/disable_sport.py # --check to query, --force if standing

- Jetson RealSense publishing `/camera/depth/image_rect_raw`:

  ssh unitree@192.168.123.22
  ros2 launch realsense2_camera rs_launch.py \
   enable_color:=false enable_infra1:=false enable_infra2:=false

## Launch (camera + encoder + policy, NO /lowcmd)

```bash
SECS=60 ./deploy_mac/run/run_dryrun_remote.sh
```

Defaults to the **latest checkpoint** in `models/T000-00-step-corridor-distill/`
(override with `CKPT=...` or `CKPT_DIR=...`; the chosen one is logged in `meta.txt`).
Bakes in `contact_threshold=20,20,20,20` (real raw sensor units), `env_class_eq_17=1`. Starts depth_camera +
depth_encoder + rl_distill_policy + wireless_to_joy, self-publishes
`/policy_engaged=true`, and logs the policy response to `run_logs/dryrun/<ts>/policy_log.csv`.
(Override `CKPT`, `CONTACT_THRESHOLD`, `SECS`, `EP_ROOT` via env vars.
`RECORD_VIEW=1` also records the prediction view headless.)

On the remote: **hold `L1`+`R1` and push the LEFT STICK forward** to command velocity.
The robot will NOT move — we are only logging the policy's response.

## Live viewer (2nd terminal — depth + |dlat| + contacts)

```bash
source deploy_mac/env.sh                 # real robot uses en13 (default)
export MUJOCO_GL=glfw KMP_DUPLICATE_LIB_OK=TRUE
python deploy_mac/diag/view_prediction.py
```

## Depth sanity dump (3rd terminal)

```bash
python deploy_mac/camera/snap_depth.py          # captures /rl/depth_image to .npy/.png
```

Expect values clamped to **[−0.5, +0.5]**, frames live (changing), few far-saturated
(+0.5) pixels. Heavy far-saturation / values outside the clamp = off-distribution
depth (the D01-99 failure mode).

## Pass / fail

- **PASS:** `|dlat|` sits near the sim band (~**1.4–2.1**), depth clean, `yaw_cmd=(0,0)`
  (32-D head confirmed). → proceed to `T000-00_real_run.md`.
- **FAIL:** `|dlat|` parks high (~3.6 like D01-99) or depth is saturated/out-of-range.
  → **do not walk.** Fix depth first (re-distill with `pw_noise`, reposition to face a
  corridor, verify the A.1 invalid→far remap is active), then re-run this dry run.

## Teardown

```bash
# Ctrl-C the run (auto-runs kill_ros.sh), then make sure nothing lingers:
bash deploy_mac/kill_ros.sh && pkill -9 -f distill_recorder
# re-enable factory sport mode:
PYTHONPATH=$HOME/go2/unitree_sdk2_python $HOME/go2/venv/bin/python \
  $HOME/go2/enable_sport.py             # --mode ai for AI sport
```
