# Runbook — S00-03 dry-run + depth-encoder yaw-prediction viewer

Validate the depth encoder's **yaw (heading) prediction** on the real Go2, from the
MacBook, using model `S00-03-corridor-distillation`. The robot does **not** move
(no `/lowcmd` is published by our stack); we only run inference and visualize what
the encoder predicts vs. what the robot sees.

All paths below assume the repo at
`$REPO`.

---

## 0. Prerequisites

- Robot powered on, MacBook on the robot Ethernet (`en13`, `192.168.123.x`).
- Jetson RealSense service publishing `/camera/depth/image_rect_raw`.
- The two wrapper scripts source `deploy_mac/env.sh` themselves (conda `ros_humble`
  + CycloneDDS on `en13`), so you do **not** need to pre-source anything for them.

---

## 1. Disable sport mode (factory high-level controller OFF)

**Why:** with sport mode ON, the Go2 factory controller moves the robot in response
to the remote. Our dry-run never drives `/lowcmd`, but the factory controller would.

`~/go2/disable_sport.py` uses the Unitree SDK `MotionSwitcherClient`. It is **not**
pip-installed — run it with the `~/go2/venv` python and the SDK source on
`PYTHONPATH`:

```bash
# 1a. CHECK current mode + pose (read-only, safe, does NOT release):
PYTHONPATH=$HOME/go2/unitree_sdk2_python $HOME/go2/venv/bin/python \
  $HOME/go2/disable_sport.py --check
```

```bash
# 1b. Lay the robot down using the Unitree FACTORY remote first
#     (sport mode is still on at this point), THEN release it:
PYTHONPATH=$HOME/go2/unitree_sdk2_python $HOME/go2/venv/bin/python \
  $HOME/go2/disable_sport.py
```

Behaviour of `disable_sport.py`:
- `--check` — query only; prints the active mode + whether the robot is laying/standing.
- *(no flags)* — releases the active mode **only if the robot is laying** (refuses if
  standing, so it can't drop unexpectedly).
- `--force` — releases even while standing → **the robot settles/lowers; secure it first.**
- `--iface en13` — default NIC (matches `GO2_IFACE`).

Success looks like: `After release: code=0 result={}   (empty / no 'name' => sport mode OFF)`.

> If `--check` prints `No high-level mode active — sport mode is already OFF.`, you're done.

---

## 2. Launch the dry-run with model S00-03 (Terminal 1)

```bash
cd $REPO
CKPT=models/S00-03-corridor-distillation/model_3800.pt SECS=300 \
  ./deploy_mac/run/run_dryrun_remote.sh
```

`CKPT` is the **only** override needed. The script already sets the rest correctly
for S00-03 (verified against the S00-03 launch logs):
- `--env_class_eq_17 1` (hardcoded) — correct: S00-03 is a `parkour_flat`/env_class-17 policy.
- `CONTACT_THRESHOLD=20,20,20,20` (default) — the **real** per-foot foot-force sensor
  values FR,FL,RR,RL (2026-06-19 forensics; the `1.0` in *sim2sim* is for clean sim
  sensors; use the per-foot vector on hardware).
- `EP_ROOT=…/go2-flyball-parkour` (default) — the `rsl_rl`/`legged_gym` source S00-03 loads.
- `SECS=300` — keep the chain (and topics) alive 5 min; it tears down afterwards.

It starts `depth_camera_node → depth_encoder_node → rl_distill_policy`, forces
`/policy_engaged=true` at 5 Hz, and prints `== READY.` once `/rl/target_pos` is
flowing (model load on 2 nodes ≈ 20–40 s). Logs go to `run_logs/dryrun/<timestamp>/`.

> If it exits with `/camera/depth/image_rect_raw not found`, the Jetson camera isn't up.

---

## 3. Launch the yaw-prediction viewer (Terminal 2)

Once Terminal 1 prints `READY`:

```bash
cd $REPO
./deploy_mac/diag/view_prediction.sh
```

READ-ONLY (subscribe only). Use the default **flat** scene — the real corridor isn't
modeled in MuJoCo; the depth picture-in-picture is what the robot truly sees.

> **`distill.launch.py` shortcut:** when launching with `is_simulation:=false`, the
> viewer starts automatically in a new window. No separate terminal needed for that path.

Optional corridor backdrop (cosmetic only):
```bash
SCENE=resources/go2/scene_corridor_step_w2.xml ./deploy_mac/diag/view_prediction.sh
```

**Save a video of the live view:**
```bash
RECORD=1 ./deploy_mac/diag/view_prediction.sh              # -> run_logs/viewer/<ts>/yaw_prediction_<ts>.mp4
RECORD=/path/out.mp4 ./deploy_mac/diag/view_prediction.sh  # explicit path
```
Or have the dry-run itself record headless into its own run dir — add `RECORD_VIEW=1`:
```bash
CKPT=models/S00-03-corridor-distillation/model_3800.pt SECS=300 RECORD_VIEW=1 \
  ./deploy_mac/run/run_dryrun_remote.sh        # -> run_logs/dryrun/<ts>/yaw_prediction.mp4
```

**Viewer controls:** left-drag orbit · wheel / `+ -` zoom · `b`/`t`/`f` behind/top/front
(**`t` top-down** is best for reading the turn) · `y` yaw-lock · `space` pause · `q`/ESC quit.

---

## 4. Validate

- Bottom-left status line should read **OK** for `/lowstate`, `/rl/depth_latent`,
  `/rl/depth_image`. (`/rl/depth_latent` STALE ⇒ encoder not producing — check
  `run_logs/dryrun/<timestamp>/depth_encoder.log`.)
- Arrows appear as soon as depth + lowstate flow — **no need to hold L1+R1**. The
  encoder runs continuously (not gated by engagement). Engagement status is shown in
  the HUD but does not gate the arrows.
- Press **`t`**. **red** = `delta_yaw` (this step), **orange** = `delta_next_yaw`,
  **gray** = current facing. Point the robot so an obstacle is on one side and open
  space on the other: the red/orange arrows should swing toward the opening, matching
  the obstacle visible in the depth PiP (near=blue, far=red).
- Watch the **|dlat|** value in the HUD — it should stay ≈ 1.0 in a clear corridor.
  Values > 1.5 suggest the depth input is out-of-distribution (obstacle very close,
  bad pixels, or camera misalignment). A spike to ~2.0 just before a fall is a
  reliable crash precursor (observed 2026-06-16 real run).

---

## 5. Teardown

- Viewer: press **`q`** or close the window.
- Terminal 1: it stops after `SECS`, or `Ctrl-C` it. Cleanup runs `kill_ros.sh`.
- Manual kill of any stragglers:
  ```bash
  ./deploy_mac/kill_ros.sh
  ```

### Re-enable sport mode (restore factory control)

When you're done with low-level work, turn the factory high-level controller back
on with `~/go2/enable_sport.py` (counterpart to `disable_sport.py`).
**The robot becomes active and can move/stand — keep the area clear, hand on power.**

```bash
# Check current mode (read-only):
PYTHONPATH=$HOME/go2/unitree_sdk2_python $HOME/go2/venv/bin/python \
  $HOME/go2/enable_sport.py --check

# Turn sport mode ON (default 'normal' = classic sport mode; prompts before acting):
PYTHONPATH=$HOME/go2/unitree_sdk2_python $HOME/go2/venv/bin/python \
  $HOME/go2/enable_sport.py
```

Mode aliases (from the SDK example): `normal` (classic sport, default), `ai`
(AI sport mode), `advanced`, `ai-w` (wheeled). Restore the same name that
`disable_sport.py --check` reported before you released it, e.g.:
```bash
PYTHONPATH=$HOME/go2/unitree_sdk2_python $HOME/go2/venv/bin/python \
  $HOME/go2/enable_sport.py --mode ai
```
Success line: `After select: code=0 result={'name': 'normal', ...}  (… => sport mode ON)`.
Power-cycling the robot also restores its default mode.

---

## Variations

**Different model (e.g. L00-02 flat baseline):** drop the `CKPT` override (it's the
script default), or set `CKPT=models/L00-02-flat-baseline-delay-distillation-noised/model_14500.pt`.

**Sim2sim instead of the real robot** (MuJoCo stands in for hardware, no Jetson/robot):
the corridor sim stack publishes `/mujoco/lowstate`, which the viewer also subscribes
to — so `./deploy_mac/diag/view_prediction.sh` works there unchanged. (Launch is the
`*_record.launch.py` corridor stack, not `run_dryrun_remote.sh`.)

---

## Reference

**Topics the viewer reads** (all published by the dry-run):
| topic | type | used for |
|---|---|---|
| `/lowstate` (and `/mujoco/lowstate`) | `unitree_go/LowState` | pose: 12 joint angles + IMU quat; foot forces |
| `/rl/depth_latent` | `std_msgs/Float32MultiArray` (len 32) | scan latent only; `\|dlat\|` = L2 norm of `[0:32]`. L00-flatstep student has no yaw predictor head — heading is held forward (`prop[6:8]=0`). Legacy 34-dim checkpoints carry yaw pred at `[32:34]` ×1.5. |
| `/rl/depth_image` | `sensor_msgs/Image` (32FC1, 58×87) | depth picture-in-picture |
| `/policy_engaged` | `std_msgs/Bool` | engagement status display only (arrows are NOT gated on this) |

**Files involved**
- `deploy_mac/run/run_dryrun_remote.sh` — real-robot camera+policy dry-run (no `/lowcmd`).
- `deploy_mac/diag/view_prediction.py` / `view_prediction.sh` — the yaw-prediction viewer.
- `~/go2/disable_sport.py` — release the factory high-level controller.
