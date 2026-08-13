# Real run — latest policy (T000-00 step-corridor distill, no-yaw-head)

Disable sport mode (lay robot down with factory remote first):

PYTHONPATH=$HOME/go2/unitree_sdk2_python $HOME/go2/venv/bin/python $HOME/go2/disable_sport.py

ssh unitree@192.168.123.222

ros2 launch realsense2_camera rs_launch.py enable_color:=false enable_infra1:=false enable_infra2:=false

## Launch (full policy stack, remote-driven)

```bash
./deploy_mac/run/run_real_remote.sh
```

LIN_VEL_MAX=1.5 POLICY_KP=30.0 POLICY_KD=0.6 ./deploy_mac/run/run_real_remote.sh

CONTACT_THRESHOLD=20,20,20,20

Defaults to the **latest (highest-iteration) checkpoint** in
`models/T000-00-step-corridor-distill/` — no need to name it. The exact `model_*.pt`
chosen is recorded in `run_logs/real/<ts>/meta.txt`.

Bakes in the real deploy params: `contact_threshold=20,20,20,20`,
`lin_vel_max=0.8`, `policy_kp/kd=30/0.6`, `env_class_eq_17=1`, `action_delay=0`
(deactivated — T000-00 trained delay=0), `depth_latency_s=0.0`,
`proprio_latency_s=0.0` (no synthetic inference sleeps),
momentary-engage.
Override via env vars:

- `CKPT=models/.../model_NNNN.pt` — pin a specific checkpoint
- `CKPT_DIR=models/<other-policy>` — use the latest from a different policy dir
- e.g. `LIN_VEL_MAX=0.5 ./deploy_mac/run/run_real_remote.sh`
  Foreground = `monitor_deploy.py` (FSM, `/lowcmd`, velocity). Logs → `run_logs/real/<ts>/`.

### Full logging (on by default)

Besides the per-node `*.log`, `meta.txt`, git state and `lowstate.csv`, each run gets:

- **`bag/`** — a `ros2 bag` of every camera/policy/control topic (`FULL_LOG=1`, the
  default). Captures the **depth-camera video as data**: raw
  `/camera/depth/image_rect_raw` AND the 58×87 `/rl/depth_image` the policy sees, plus
  `/rl/depth_latent`, `/rl/est_lin_vel`, `/rl/last_action`, `/rl/target_pos`, `/joy`,
  `/wirelesscontroller`, `/policy_engaged`, `/lowstate`, `/lowcmd`, `/imu/data`,
  `/base_velocity`. Replay offline — no extra live load. Disable with `FULL_LOG=0`.
- **`yaw_prediction.mp4`** — watchable depth + `|dlat|` + contacts video, only if
  `RECORD_VIEW=1`. ⚠️ renders live, adds GPU load to the Mac running the 50 Hz policy.

Replay / export the depth video from the bag afterwards:

```bash
ros2 bag play run_logs/real/<ts>/bag        # then view_prediction.py / rqt_image_view
python deploy_mac/camera/snap_depth.py         # export /rl/depth_image frames while it plays
```

## Live viewer (2nd terminal — depth + |dlat| + contacts)

```bash
source deploy_mac/env.sh
export MUJOCO_GL=glfw KMP_DUPLICATE_LIB_OK=TRUE
python deploy_mac/diag/view_prediction.py
```

## Remote

`B` = stand · **hold `L1`+`R1`** = walk (release → stand) · `A` = lay · **`L2`+`R2`** = quit

## Watch / abort

- **First: stationary `|dlat|` check.** Hoisted, facing a corridor/wall, engage but do
  NOT walk. `|dlat|` should sit near the sim band (~1.4–2.1). If it parks at ~3.6 like
  the D01-99 dry run, the depth is still off-distribution — **do not walk**; fix depth
  (re-distill with `pw_noise`, or check the PiP for far-saturation) first.
- Ramp speed slowly (0.3 → 0.8 m/s).
- **Abort if `|dlat| > 1.5`** (crash precursor) or any depth anomaly in the PiP.

## Teardown

```bash
# Ctrl-C the launch (auto-runs kill_ros.sh), then make sure nothing lingers:
bash deploy_mac/kill_ros.sh && pkill -9 -f distill_recorder
# re-enable factory sport mode:
PYTHONPATH=$HOME/go2/unitree_sdk2_python $HOME/go2/venv/bin/python \
  $HOME/go2/enable_sport.py             # --mode ai for AI sport
```
