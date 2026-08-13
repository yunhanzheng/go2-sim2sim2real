# Runbook — REAL run on the Go2 (motors live, `/lowcmd`)

This is the **real** deployment: `low_level_ctrl` drives `/lowcmd` and **the robot
moves**. Unlike the dry-run (`run_dryrun_remote.sh`, no `/lowcmd`), every step here
can move the legs. Read §0 and §6 before powering anything.

Repo: `$REPO`.

> ⚠️ **Per `DEPLOY.md`, the full ROS policy stack is NOT free-walk-ready** (ROS-stack
> fall ~14 s, real depth pipeline unvalidated, no watchdog/e-stop). The only proven
> real milestone is **lay↔stand** (§3). Run the policy **hoisted/tethered only** (§4).

---

## 0. Safety pre-flight (every time)

- [ ] Robot **hoisted / tethered**, clear space around it.
- [ ] **Sport mode OFF** (see §2) — else the factory controller fights `/lowcmd`.
- [ ] **Hand on power.** E-stop = remote **L2+R2** (quits the controller). It's a
      fragile exact-combo, not a real dead-man switch — keep the physical power within reach.
- [ ] Expect a **small twitch at start**: `low_level_ctrl` immediately stiffens to the
      LAY pose at kp=50.
- [ ] MacBook on the robot Ethernet (`en13`, `192.168.123.x`).

---

## 1. One-time: build `low_level_ctrl` (C++)

It's the only node that touches hardware. It's already built at
`install/deploy_rl_policy/lib/deploy_rl_policy/low_level_ctrl`. Rebuild only if changed:

```bash
cd $REPO
source deploy_mac/env.sh
./deploy_mac/lib/build_cpp.sh      # macOS-specific fixes baked in
source deploy_mac/env.sh       # re-source to overlay the new install
```

---

## 2. Disable sport mode

Lay the robot down with the Unitree factory remote first, then release:

```bash
PYTHONPATH=$HOME/go2/unitree_sdk2_python $HOME/go2/venv/bin/python \
  $HOME/go2/disable_sport.py            # add --check first to query; --force if standing
```
(Full details in `deploy_mac/runbooks/yaw_prediction_runbook.md` §1. Re-enable afterwards: §5 here.)

---

## 3. Real lay ↔ stand (PROVEN, safest first — no policy, no camera)

This is the validated real milestone. No policy is loaded; it only exercises the
lay/stand FSM + PD. Do **not** press L1+R1 (engage) here.

```bash
cd $REPO
./deploy_mac/run/run_deploy.sh
```

- Starts `wireless_to_joy.py` (remote → `/joy`) + `low_level_ctrl`
  (`is_simulation:=false`, policy_kp=30 / kd=0.6; transition kp=50 / kd=1).
- Foreground = `monitor_deploy.py`: live FSM state, `/policy_engaged`, whether
  `/lowcmd` flows + its kp (50=bring-up, 30=policy), pose distances. Logs
  `run_logs/real/<timestamp>/lowstate.csv`.
- **Remote:** `B` → stand · `A` → lay · **L2+R2** → quit. (Do NOT press L1+R1.)
- Stop: `Ctrl-C` (cleanup runs `kill_ros.sh`). Logs in `run_logs/real/<timestamp>/`.

---

## 4. Full policy stack (HOISTED first-engage only — robot walks)

The 5-node stack with inference engaged. **Hoisted/tethered only** until the §6
blockers are cleared.

### 4a. Camera (Jetson) — if running the depth pipeline
```bash
# On the Jetson (CycloneDDS shell):
ros2 launch realsense2_camera rs_launch.py \
  enable_color:=false enable_infra1:=false enable_infra2:=false
```
> **Topic note:** the Jetson publishes RECTIFIED depth on
> **`/camera/depth/image_rect_raw`** (not `image_raw`). `distill.launch.py` now
> defaults `depth_camera_node`'s input to that topic via the `depth_input_topic`
> launch arg, so depth flows out of the box. Override with
> `depth_input_topic:=...` if your camera service uses a different name (e.g. the
> realsense2_camera v4.x `/camera/camera/...` double-namespace — check
> `ros2 topic list | grep image_rect_raw`).

### 4b. Launch (Mac) — Go2 remote, RECOMMENDED on this machine

Use the remote-driven script — it runs `wireless_to_joy` + the inference chain +
`low_level_ctrl`, with no `joy` package / USB gamepad needed:

```bash
cd $REPO
./deploy_mac/run/run_real_remote.sh          # default ckpt = S00-03 corridor
# L00-02 flat (validated real config):
CKPT=models/L00-02-flat-baseline-delay-distillation-noised/model_14500.pt LIN_VEL_MAX=1.3 \
  ./deploy_mac/run/run_real_remote.sh
```
- `contact_threshold` defaults to **`20,20,20,20`** (FR,FL,RR,RL real sensor
  units; 2026-06-19 forensics); `env_class_eq_17=1`, `action_delay=0` (deactivated —
  T000-00 trained delay=0), `DEPTH_LATENCY_S=0.0`, `PROPRIO_LATENCY_S=0.0`
  (no synthetic inference sleeps),
  `policy_kp/kd=30/0.6`, `lin_vel_max` synced across encoder+policy. Override with
  `CONTACT_THRESHOLD=`, `LIN_VEL_MAX=`, `POLICY_KP=`, `POLICY_KD=`, `EP_ROOT=`,
  `DEPTH_LATENCY_S=`, or `PROPRIO_LATENCY_S=`.
- Foreground is the live monitor + `run_logs/real/<ts>/lowstate.csv`. `low_level_ctrl`
  owns `/policy_engaged` (engage by holding L1+R1) — the script does not fake it.

For lightweight latency instrumentation without a rosbag:

```bash
LATENCY_LOG=1 ./deploy_mac/run/run_real_remote.sh
```

This adds correlated `latency_*.tsv` files and `latency_summary.txt` to the run
directory. It measures host-observable camera, encoder, actor, DDS, and `/lowcmd`
publication segments. Camera exposure-to-host and `/lowcmd`-to-motor-application
remain explicitly unobservable without synchronized hardware timestamps or a
firmware acknowledgement. Unlike `FULL_LOG=1`, this does not record image or
message payloads, and it does not add artificial latency.

> **Keyboard alternative — `distill.launch.py`:** The launch file requires no
> Go2 remote (use `use_keyboard:=true`; WASD move, Q/E yaw, hold Space = engage).
> It **automatically opens the yaw-prediction viewer window** (add `launch_viewer:=false`
> to suppress). Depth flows from the Jetson via `depth_camera_node`, which now
> defaults to `/camera/depth/image_rect_raw` (override with `depth_input_topic:=`).
> ```bash
> ros2 launch deploy_rl_policy distill.launch.py is_simulation:=false use_keyboard:=true \
>   ckpt:=models/S00-03-corridor-distillation/model_3800.pt \
>   ep_root:=$EP_ROOT \
>   env_class_eq_17:=1 lin_vel_max:=0.8
> ```

### 4b-S00-07. S00-07 edge-corridor-baseline-distillation — model_1000

**What it is:** distillation student trained exclusively on `parkour_flat` (env_class 17)
switchback corridors, width 1–3 m, amplitude 1.4 m, 6 bends, up to 58° corners.
Depth-guided (depth encoder required). **Not trained on steps.**

**Sim2sim findings (2026-06-16):**
- Walks at ~0.4–0.6 m/s actual in a 2 m wide walls-only switchback corridor.
- Reaches the first corner (~2.6 m) cleanly; stalls at the corner apex under sharp
  turns (58°). Recovers and re-attempts after a velocity ramp-up.
- model_900 with a stop-then-ramp profile (`8:1.0 → 4:0 → 2:0.5 → 2:0.8 → 8:1.0`)
  showed more robust corner recovery than a hard 0→1 jump.
- **Start conservatively:** `lin_vel_max=0.7` for the first real engage.

**Depth camera required** — same Jetson + RealSense setup as S00-03 (§4a caveat
about `/camera/depth/image_rect_raw` topic applies here too).

**Remote-driven (recommended):**

```bash
cd $REPO
CKPT=models/S00-07-edge-corridor-baseline-distillation/model_1000.pt \
  LIN_VEL_MAX=0.7 \
  ./deploy_mac/run/run_real_remote.sh
```

Bump to `LIN_VEL_MAX=0.8` once a clean hoisted walk is confirmed. Defaults
(`CONTACT_THRESHOLD=20,20,20,20`, `POLICY_KP=30`, `POLICY_KD=0.6`, `MOMENTARY_ENGAGE=true`)
are correct — no overrides needed.

**Keyboard alternative (`distill.launch.py`):**

```bash
ros2 launch deploy_rl_policy distill.launch.py is_simulation:=false use_keyboard:=true \
  ckpt:=models/S00-07-edge-corridor-baseline-distillation/model_1000.pt \
  ep_root:=$EP_ROOT \
  env_class_eq_17:=1 lin_vel_max:=0.7
```

**Parameter reference for S00-07 real:**

| param | real value | note |
|---|---|---|
| `env_class_eq_17` | 1 | sets prop[11:13]=(0,1) — required for parkour_flat |
| `contact_threshold` | 20,20,20,20 | FR,FL,RR,RL real sensor units (sim uses 2.0 N net contact) |
| `action_delay` | 0 | deactivated for deploy (match training; T000-00=0) |
| `depth_latency_s` | 0.0 | no artificial delay on real deploy; measure natural latency with `LATENCY_LOG=1` |
| `proprio_latency_s` | 0.0 | no artificial delay on real deploy; measure natural latency with `LATENCY_LOG=1` |
| `lin_vel_max` | 0.7 → 0.8 | start at 0.7; raise after first clean pass |
| `policy_kp / kd` | 30 / 0.6 | standard |
| `momentary_engage` | true | dead-man hold: release L1+R1 → stand immediately |

---

### 4c. Drive it (remote)
`run_real_remote.sh` runs `low_level_ctrl` with **`momentary_engage:=true`**, so the
policy is a **dead-man hold**:
- `B` → lay→stand
- **hold L1+R1** (the two upper triggers; "LB+RB" in Xbox terms) → policy walks
  (push left stick for speed). **Release L1+R1 → policy stops, robot returns to the
  standing pose** (held at kp=50/kd=1).
- `A` → lay down (also stops the policy)
- **L2+R2** (lower triggers) → quit/e-stop

To use the old latch behavior (engage stays on until A), run
`MOMENTARY_ENGAGE=false ./deploy_mac/run/run_real_remote.sh`.

> The momentary behavior lives in the C++ `low_level_ctrl` (param `momentary_engage`,
> default false). It's already built. If you ever edit the FSM, rebuild:
> `source deploy_mac/env.sh && ./deploy_mac/lib/build_cpp.sh && source deploy_mac/env.sh`.

### 4d. Watch it (second terminal, read-only)
```bash
source deploy_mac/env.sh
RUN=run_logs/real/$(date +%F_%H-%M-%S); mkdir -p "$RUN"
python deploy_mac/analysis/monitor_deploy.py --csv "$RUN/lowstate.csv"
```

**Yaw-prediction viewer:**

- **`distill.launch.py` path:** the viewer opens **automatically** in a new window.
  Add `launch_viewer:=false` to suppress it.
- **`run_real_remote.sh` path** (second terminal):
  ```bash
  ./deploy_mac/diag/view_prediction.sh                 # live window
  RECORD=1 ./deploy_mac/diag/view_prediction.sh        # live window + MP4 -> run_logs/viewer/<ts>/
  ```
- Arrows show **always** (not just when engaged) so you can sanity-check the depth
  encoder while standing. Policy engaged/not-engaged is shown in the HUD.
- Or record headless into the run dir (adds load to this Mac — watch for timing jitter):
  ```bash
  RECORD_VIEW=1 ./deploy_mac/run/run_real_remote.sh   # -> run_logs/real/<ts>/yaw_prediction.mp4
  ```

---

## 5. Teardown + re-enable sport mode

```bash
./deploy_mac/kill_ros.sh        # kill any ROS orphans
# also useful between MuJoCo launches:  pkill -f mujoco_simulator
```
Restore factory control (robot becomes active — clear area, hand on power):
```bash
PYTHONPATH=$HOME/go2/unitree_sdk2_python $HOME/go2/venv/bin/python \
  $HOME/go2/enable_sport.py       # default 'normal'; --mode ai for AI sport
```

---

## 6. Open blockers — DO NOT free-walk until cleared (from `DEPLOY.md`)

1. **ROS-stack fall (~14 s).** With the correct config the ROS lay2stand stack walks
   straight ~10 m then falls ~14 s, while the standalone harness runs clean to 20 s.
   Same commit/ckpt/flags → a ROS-path bug, not the policy. **Must fix first.**
2. **Real depth pipeline unvalidated** + `realsense2_camera` topic/preprocessing gaps
   (see §4a). RealSense holes can normalize to a phantom obstacle.
3. **No watchdog / dead-man switch** — no command timeout or stale-latent expiry;
   e-stop is a fragile exact-float joystick combo.

Recommended first hardware session: **camera bring-up + hoisted first-engage**, not
free walking.

---

## Reference

**PD gains:** transition (lay/stand) kp=50 / kd=1 ; policy kp=30 / kd=0.6.
**FSM / Go2 remote:** B=stand, A=disengage+lay, hold L1+R1 (=Xbox LB+RB)=engage, L2+R2=quit/e-stop.
**Deploy defaults:** `env_class_eq_17=1`, `contact_threshold=20,20,20,20`
(per-foot real, auto on real runs), `action_delay=0` (deactivated — match the
checkpoint's training; T000-00 trained delay=0), `depth_latency_s=0.0`,
`proprio_latency_s=0.0` (no synthetic sleeps), kp/kd=30/0.6.

**L00-02 (legacy, trained with `--delay`):** pass `action_delay:=1` explicitly — the
deploy default is now 0. Else `env_class_eq_17=1`, `contact_threshold=20,20,20,20`,
kp/kd=30/0.6, `lin_vel_max=1.3`.

**S00-07 corridor (sim2sim only, 2026-06-16):** `env_class_eq_17=1`,
`contact_threshold=20,20,20,20`, `action_delay=0`, kp/kd=30/0.6, `lin_vel_max=0.7`
(start). Walks ~0.4–0.6 m/s; stalls at sharp corners (>50°). Use ramp profile after a
stop (`0→0.5→0.8→1.0` over 4 s).

| node | role | topic |
|---|---|---|
| `low_level_ctrl` (C++) | lay/stand/engage FSM, PD → motors | `/joy` → `/lowcmd` |
| `rl_distill_policy` | 50 Hz actor → joint targets | `/lowstate`,`/rl/depth_latent` → `/rl/target_pos` |
| `depth_encoder_node` | depth → 32 latent (no-yaw-head) | `/rl/depth_image` → `/rl/depth_latent` |
| `depth_camera_node` | RealSense → 58×87 norm depth | `/camera/depth/image_rect_raw` → `/rl/depth_image` |

Related: `deploy_mac/runbooks/yaw_prediction_runbook.md` (dry-run + yaw viewer), `DEPLOY.md`
(full L00-02 deploy notes + blockers).
