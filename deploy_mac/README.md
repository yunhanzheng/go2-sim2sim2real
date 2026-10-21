# Running the ROS 2 inference stack on macOS (Apple Silicon) via RoboStack

Run the `go2-sim2sim2real` ROS 2 inference (`rl_distill_policy`) natively on a **MacBook
Air M3**, talking to the **Unitree Go2** over Ethernet — **no Linux, no VM, no Docker**.

This is the path we landed on after Asahi Linux turned out unviable on M3 (stable
installer refuses `Mac15,13`; M3 support is still tester-only / no-GPU as of 2026-06).


## Layout (2026-07-31)

The scripts are grouped by what they do; `env.sh` and `kill_ros.sh` stay at the
root because nearly every script sources or calls them.

| dir | what lives here |
| --- | --- |
| `run/` | robot + dry-run entry points (`run_real_remote.sh`, `run_dryrun_remote.sh`, `run_deploy.sh`, `run_inference_dryrun*.sh`, `keyboard_control*.sh`, `wireless_to_joy.py`) |
| `sim/` | sim2sim rendering (`render_sim2sim.sh`, `render_sim2sim_noros.py`, `verify_mujoco_depth_noise.py`) |
| `camera/` | depth camera + clock (`view_depth*`, `snap_depth.py`, `export_bag_depth.py`, `recompute_depth_bag.py`, `camera_metadata_latency.py`, `check_camera_clock.py`, `sync_jetson_clock.sh`, `fake_depth_latent.py`) |
| `analysis/` | post-run analysis (`analyze_run.py`, `analyze_latency_logs.py`, `log_*.py`, `monitor_deploy.py`, `read_target_pos.py`, `plot_*.py`) |
| `diag/` | pre-flight and debug (`check_deploy_contract.py`, `check_lowstate.sh`, `ros_lowstate_check.py`, `selftest_dds.py`, `decode_*.py`, `view_prediction.*`) |
| `lib/` | shared helpers (`log_git_state.sh`, `build_cpp.sh`, `cyclonedds_en13.xml`) |
| `runbooks/`, `stubs/`, `jetson/` | unchanged |

Runs are written to `run_logs/<kind>/<ckptdir>__<model>/<timestamp>/`, one
folder per policy.

## How it works

- **ROS 2 Humble** is installed natively in a conda env via **RoboStack**.
- ROS 2 speaks **DDS**; the Go2 firmware publishes on DDS topic `rt/lowstate`, which
  is exactly ROS 2's topic-naming convention — so a node subscribing to `/lowstate`
  receives the robot's data directly. No `unitree_ros2` bridge process is needed,
  only the `unitree_go` **message definitions** (built below) + CycloneDDS bound to
  the right NIC.
- The PD loop runs on the Go2's **motor MCUs** (position-mode `LowCmd`), so Python
  timing jitter on the Mac is not critical.

```
Go2 ──DDS rt/lowstate──▶ rl_distill_policy.py (rclpy, native on Mac)
                          builds 53-D proprio + depth latent → actor → /rl/target_pos
/rl/target_pos ─────────▶ low_level_ctrl (FSM + position-mode LowCmd) ──DDS rt/lowcmd──▶ Go2
```

## Network (one-time)

Mac on the robot's subnet, static IP:
- IP `192.168.123.99`, mask `255.255.255.0`, **router blank** (leaving a gateway here
  hijacks the default route and kills internet — keep it empty).
- Robot control board: `192.168.123.161`. Verify: `ping 192.168.123.161`.
- Onboard Jetson: `192.168.123.18` on the Orin NX image (`.222` on the older Foxy image).
  Verify: `ping 192.168.123.18`.
- Find the interface name: `route get 192.168.123.161 | grep interface` → here **en13**
  (on an Ubuntu host: `ip route get 192.168.123.18` → `dev eno1`).

## Environment setup (one-time)

Requires miniforge/conda already installed.

```bash
# 1. RoboStack ros-humble base + CycloneDDS RMW + colcon
mamba create -y -n ros_humble -c robostack-staging -c conda-forge \
  ros-humble-ros-base ros-humble-rmw-cyclonedds-cpp colcon-common-extensions

# 2. message-build deps
mamba install -y -n ros_humble -c robostack-staging -c conda-forge \
  ros-humble-rosidl-generator-dds-idl ninja "cmake<4"
#   - cmake<4: CMake 4 removed FindPythonInterp/Libs that Humble's rosidl still uses
#   - rosidl-generator-dds-idl: required by the unitree message packages

# 3. libatomic stub: ROS 2's C generator passes -latomic (a Linux/GCC flag);
#    macOS clang inlines atomics, so provide an empty lib to satisfy the linker.
source ~/miniforge3/etc/profile.d/conda.sh && conda activate ros_humble
printf 'void __dummy_atomic(void){}\n' | clang -x c -shared -o "$CONDA_PREFIX/lib/libatomic.dylib" -
```

### Build the unitree messages

```bash
mkdir -p ~/go2/ros2_ws/src && cd ~/go2/ros2_ws/src
git clone --depth 1 https://github.com/unitreerobotics/unitree_ros2.git
cd ~/go2/ros2_ws
conda activate ros_humble
colcon build --packages-select unitree_go unitree_api unitree_hg \
  --cmake-args -DCMAKE_BUILD_TYPE=Release \
    -DPython3_EXECUTABLE=$CONDA_PREFIX/bin/python \
    -DPython_EXECUTABLE=$CONDA_PREFIX/bin/python \
    -DCMAKE_SHARED_LINKER_FLAGS="-L$CONDA_PREFIX/lib" \
    -DCMAKE_EXE_LINKER_FLAGS="-L$CONDA_PREFIX/lib" \
    -DCMAKE_MODULE_LINKER_FLAGS="-L$CONDA_PREFIX/lib"
```
Why the flags:
- `Python*_EXECUTABLE`: CMake's `FindPython` doesn't auto-locate the conda Python.
- `*_LINKER_FLAGS=-L$CONDA_PREFIX/lib`: lets the linker find the libatomic stub. All
  three categories are needed — Python extension modules are CMake **MODULE** libs.

Verified working: `ros2 interface show unitree_go/msg/LowState`, `python -c "from
unitree_go.msg import LowState, LowCmd"`, and `ros2 doctor` reports
`rmw_cyclonedds_cpp`.

## Each session

```bash
cd <this repo>
source deploy_mac/env.sh        # conda env + messages + CycloneDDS on en13
./deploy_mac/diag/check_lowstate.sh  # confirm live data from the robot
```

Read-only diagnostics (never publish / never command motors):
```bash
python deploy_mac/diag/ros_lowstate_check.py    # one decoded /lowstate snapshot
python deploy_mac/analysis/plot_motors.py     # live plot of all 12 joints — move legs by hand to verify
python deploy_mac/analysis/plot_imu_feet.py   # live plot of IMU (rpy/gyro/accel) + 4 foot forces — tilt/press to verify
python deploy_mac/analysis/log_foot_forces.py --secs 40  # log foot forces to CSV; lift+hold robot to measure airborne noise floor
#   add --selftest to any plot_*.py for a headless read check (no GUI)
./deploy_mac/kill_ros.sh                   # ALWAYS run when done (kill orphan nodes)
```
If `plot_motors.py` opens no window, set a backend: `MPLBACKEND=macosx` (or `TkAgg`).

`env.sh` knobs: `GO2_IFACE` (default `en13`), `ROS2_WS` (default `~/go2/ros2_ws`).

## Running the inference

> ⚠️ First hardware runs must be **hoisted/tethered** — there is no watchdog or
> dead-man's switch (see `../DEPLOY.md`, "Known limitations"). Never free-walk on a
> first bring-up.

**Inference dry-run (no camera, NO motor commands — robot stays put):**
```bash
./deploy_mac/run/run_inference_dryrun.sh   # gate (zero depth latent + engage) + rl_distill_policy + read /rl/target_pos
```
This runs `rl_distill_policy` on live `/lowstate` with a zero depth latent and prints
the policy's `/rl/target_pos`. Nothing drives `/lowcmd`, so the robot does not move.
(Note: run it while the robot is **standing** for in-distribution targets — a laying
pose is out-of-distribution for the flat-walk policy and yields large target deltas.)

Manual equivalent: `source deploy_mac/env.sh`, run `fake_depth_latent.py`, then
`rl_distill_policy.py --ckpt <L00-02> --ep_root <go2-flyball-parkour> --env_class_eq_17 1 --contact_threshold 20,20,20,20 --proprio_latency_s 0.0`.

## Building `low_level_ctrl` (C++) on macOS

`low_level_ctrl` (C++) converts `/rl/target_pos` → `/lowcmd` and runs the
lay→stand→engage FSM — it's the node that **first moves the robot**. Build it with:

```bash
source deploy_mac/env.sh
./deploy_mac/lib/build_cpp.sh        # colcon build deploy_rl_policy with the macOS fixes
source deploy_mac/env.sh         # re-source to overlay the new install
```
`build_cpp.sh` encapsulates the macOS-specific fixes (each was needed):
`cmake<4` (env), **libatomic stub + `-L`**, **install-RPATH → `$CONDA_PREFIX/lib`**
(SIP strips `DYLD_LIBRARY_PATH`), and **linking `libpython`** (a conda ROS lib needs
Python C-API symbols in the flat namespace). Verified: node starts, publishes `/mujoco/lowcmd`.

### Driving the FSM from the Go2's own remote
The C++ node reads `/joy` (Xbox layout). Bridge the Go2 remote (`/wirelesscontroller`)
to it — no separate gamepad needed:
```bash
python deploy_mac/run/wireless_to_joy.py   # B=stand, A=lay, L1+R1=engage, L2+R2=quit
```

### Full stack (⚠️ REAL motors — hoist the robot, Go2 sport mode OFF)
```bash
source deploy_mac/env.sh
python deploy_mac/run/wireless_to_joy.py &                                   # remote -> /joy
ros2 run deploy_rl_policy low_level_ctrl --ros-args -p is_simulation:=false \
    -p policy_kp:=30.0 -p policy_kd:=0.6 &                               # FSM -> /lowcmd
python deploy_mac/camera/fake_depth_latent.py --no-engage &                     # zero depth (no camera); FSM owns /policy_engaged
python src/deploy_rl_policy/scripts/rl_distill_policy.py \
    --ckpt models/L00-02-flat-baseline-delay-distillation-noised/model_14500.pt \
    --ep_root $EP_ROOT \
    --env_class_eq_17 1 --contact_threshold 20,20,20,20 \
    --proprio_latency_s 0.0 &                                           # inference -> /rl/target_pos (FR,FL,RR,RL real sensor units)
# remote: B (lay->stand), hold L1+R1 (engage). L2+R2 = quit. Then: ./deploy_mac/kill_ros.sh
```
⚠️ The Python CRC is *not* in this path (C++ `motor_crc.cpp` is); but still verify the
robot accepts `/lowcmd` (motors respond when hoisted) before any free run.

## Status

- [x] RoboStack `ros_humble` env, CycloneDDS RMW
- [x] `unitree_go/api/hg` messages built + loadable
- [x] CycloneDDS bound to `en13`
- [x] live `/lowstate` confirmed from robot (2026-06-15, via ROS 2 / rmw_cyclonedds — see `ros_lowstate_check.py`)
- [x] `torch` installed in env (OpenMP fix in `env.sh`)
- [x] **inference dry-run** verified — `rl_distill_policy` on live `/lowstate` + zero depth
  latent → bounded `/rl/target_pos`, no motor commands (`run_inference_dryrun.sh`)
- [x] **C++ `low_level_ctrl` builds + runs on macOS** (`build_cpp.sh`); sim-verified (node up,
  publishes `/mujoco/lowcmd`). Go2-remote input via `wireless_to_joy.py`.
- [x] **Depth camera over DDS** — RealSense on the Jetson → `/camera/depth/image_rect_raw` →
  Mac → `depth_camera_node` → `/rl/depth_image` @15 Hz (no librealsense on macOS).
- [x] **hoisted lay→stand→engage** with full depth pipeline on hardware (2026-06-16,
  S00-03 corridor, contact_threshold=20, policy_kp/kd=30/0.6, lin_vel_max=0.8).
  Robot walked at ~0.9 m/s. Fall at ~9 s segment — crash preceded by `|dlat|` spike
  to 1.96 and yaw_pred → −51°. Investigation ongoing.

## Depth camera (RealSense D435i on the Jetson, over DDS)

The RealSense is plugged into the robot's **Jetson** (currently `192.168.123.18` on the
Orin NX / JetPack 6 image; older Foxy images answer on `.222`), not the Mac — so the
Jetson runs `realsense2_camera` and depth reaches the Mac **over DDS**. No
librealsense/pyrealsense2 on macOS. Verified 2026-10-20 on the Orin NX:
`/camera/depth/image_rect_raw` at **59.9 Hz** across DDS, 480x270x60.

Topology: Jetson `.18` (cyclonedds, domain 0) ── robot board `.161` ── Mac `.99`.
⚠️ Mac must NOT use `.18` (Jetson's IP — an earlier collision silently broke Mac↔Jetson DDS).

### One-time Jetson setup — Foxy image (it had no internet, a ~2 yr-wrong clock, and an expired ROS key)
```bash
# (a) internet via the Mac (NAT) — on the MAC, in Terminal.app (needs sudo password):
sudo sysctl -w net.inet.ip.forwarding=1
printf 'nat on en0 from 192.168.123.0/24 to any -> (en0)\n' > /tmp/go2-nat.conf   # en0=Wi-Fi
sudo pfctl -ef /tmp/go2-nat.conf
#     on the JETSON:
sudo ip route replace default via 192.168.123.99            # the Mac
echo "nameserver 1.1.1.1" | sudo tee /etc/resolv.conf
# (b) fix the Jetson clock (else TLS/apt "not yet valid"):
sudo date -u -s "<current UTC, e.g. 2026-06-15 12:35:00>"
# (c) current ROS key + OFFICIAL repo (tuna mirror was stale + expired-key):
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=arm64 signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu focal main" | sudo tee /etc/apt/sources.list.d/ros2-official.list
sudo apt update && sudo apt install -y ros-foxy-realsense2-camera   # Foxy image only;
# on JetPack 6 / Humble the package is `ros-humble-realsense2-camera` (already installed on
# the current robot — see deploy_mac/jetson/INSTALL.md)
# undo Mac NAT later:  sudo pfctl -d ;  sudo sysctl -w net.inet.ip.forwarding=0
```

### Run the camera

On the **Humble / JetPack 6** image the driver is a systemd service. Do not launch it by
hand — two `rs_launch` processes fight over `/dev/video0` and both crash-loop on
`resource busy`:

```bash
ssh unitree@192.168.123.18 sudo systemctl restart go2-realsense-humble
```

The unit pins the profile and flattens the namespace, so the topic is exactly
`/camera/depth/image_rect_raw` at **480x270x60**. See `jetson/INSTALL.md` for why that
profile, and for the three 4.58 parameter traps (`depth_module.profile` →
`depth_profile`, empty `camera_namespace` → `/`, `laser_power` 350 → 360).

Manual equivalent, debugging only:

```bash
ros2 launch realsense2_camera rs_launch.py \
  enable_color:=false enable_infra1:=false enable_infra2:=false \
  depth_module.depth_profile:=480x270x60 depth_module.infra_profile:=480x270x60 \
  camera_namespace:=/
# laser_power / emitter_enabled are NOT launch args on 4.58 — set them after the node is up:
ros2 param set /camera depth_module.emitter_enabled 1
ros2 param set /camera depth_module.laser_power 360.0     # 350 is illegal: step = 30
```

Then on the deployment host, preprocess to the policy's 58x87:

```bash
python src/deploy_rl_policy/scripts/depth_camera_node.py --ros-args -p input_topic:=/camera/depth/image_rect_raw
# encode the real depth latent instead of fake_depth_latent:
python src/deploy_rl_policy/scripts/depth_encoder_node.py --ckpt <L00-02> --ep_root <ep> --contact_threshold 20,20,20,20 --enforce_latency_s 0.0
```
Note the topic is `image_rect_raw` (not `image_raw`) → pass it via `--input_topic`. 480x270x60 is
picked for FOV and latency, not bandwidth: it frames the trained 87x56.49 with zero padding
and keeps depth age inside the trained 80–100 ms band. On the dock's Type-C port the camera is
SuperSpeed (`Device USB type: 3.2`, Bus 002); on a USB-A port it is a 480M link, where 848x480
would cap at 10 fps.

`depth_camera_node` is **faithful to training**: it auto-reads `camera_info`, derives the
stream's true FOV, and tan-space matches it to the render's hFOV 87°/vFOV 56.49° (crop when
wider, pad with `DEPTH_FAR` when narrower) → resizes to the **60×106** sim-render resolution → then
`normalize_depth` applies training's exact crop `[:-2,4:-4]` + resize to **58×87**, and it
**decimates to ~10 Hz** to match the recurrent encoder's training cadence. A naive 640×480→58×87
resize (the old behavior) was off-distribution (no crop, wrong aspect). Params: `auto_fov`
(default true; falls back to `cam_hfov_deg`/`cam_vfov_deg` when `camera_info` never arrives) and
`target_hz`.

The real wrapper has an opt-in lightweight timing mode:

```bash
LATENCY_LOG=1 ./deploy_mac/run/run_real_remote.sh
```

It writes correlated `latency_*.tsv` files and an automatic
`latency_summary.txt` under `run_logs/real/<timestamp>/`. This records timestamps and
sequence IDs only, not camera or ROS message payloads; `FULL_LOG=1` remains the
separate, high-I/O rosbag mode. Real deploy does not add artificial delay by
default (`DEPTH_LATENCY_S=0.0`, `PROPRIO_LATENCY_S=0.0`); timing mode measures
the naturally occurring host-visible latency without changing it.

Timing mode also synchronizes the USB-camera host (Jetson) clock to the Mac
before any motor-driving process starts (`SYNC_JETSON_CLOCK` defaults to the
value of `LATENCY_LOG`). It prompts for the Jetson SSH/sudo passwords and writes
the measured residual offset to `clock_sync.log`. Set `SYNC_JETSON_CLOCK=0` only
when another clock service is already synchronizing the machines. To synchronize
without launching the robot stack:

```bash
source deploy_mac/env.sh
./deploy_mac/camera/sync_jetson_clock.sh
```

`latency_camera_metadata.tsv` records the RealSense `GLOBAL_TIME`, middle-of-
exposure sensor timestamp, USB backend timestamp, and Mac receipt timestamp.
The summary then reports camera exposure → Jetson USB backend → Mac image
receipt. This is possible because the RealSense is physically attached to the
Jetson by USB; the remaining Jetson→Mac portion is DDS/network delivery.

## Validating the depth-encoder yaw prediction (live MuJoCo)

`view_prediction.py` opens an interactive MuJoCo window that shows, in real time,
**what the depth encoder predicts** during a dry-run — so you can confirm the
encoder ("YourPredictor") reacts to the scene before trusting it on hardware.
**READ-ONLY** (subscribe only; never touches `/lowcmd`).

It draws:
- the Go2 posed from `/lowstate` (12 joint angles + IMU quaternion),
- heading arrows above the robot, exactly as the policy consumes them.
  The L00-flatstep student has no yaw predictor head, so `/rl/depth_latent` is
  32-dim (scan latent only) and the heading is held forward (`prop[6:8]=0`) —
  the viewer draws a single **red** forward arrow. Legacy 34-dim checkpoints
  carry a yaw tail at `[32:34]` × `YAW_PRED_MULTIPLIER=1.5`, rendered as
  **red** = `delta_yaw` (this step), **orange** = `delta_next_yaw` (next step),
  **gray** = current facing (reference),
- a picture-in-picture of the normalized depth the robot truly sees
  (`/rl/depth_image`, 58×87; near=blue, far=red).

Run it in a **second terminal** while a dry-run is up (the dry-run publishes the
topics; this just visualizes them):
```bash
# terminal 1 — the dry-run (remote-driven, no /lowcmd):
SECS=120 ./deploy_mac/run/run_dryrun_remote.sh
# terminal 2 — the validator:
./deploy_mac/diag/view_prediction.sh
#   SCENE=resources/go2/scene_corridor_step_w2.xml ./deploy_mac/diag/view_prediction.sh
```
Controls: left-drag orbit · wheel / `+ -` zoom · `b`/`t`/`f` behind/top/front
(top is best for reading the turn) · `y` yaw-lock (face +x, keep roll/pitch) ·
`space` pause · `q`/ESC quit. Arrows show as soon as `/rl/depth_latent` is
fresh — **no engagement required** (the encoder runs continuously, not gated by
lay/stand). Engagement status is shown in the HUD for reference only.
The HUD also displays `|dlat|` (depth latent L2 norm; healthy ≈ 1.0, crash
precursor ≈ 2.0) and foot contacts. The bottom-left status line flags stale topics.

`distill.launch.py is_simulation:=false` **auto-launches the viewer** (add
`launch_viewer:=false` to suppress). For `run_real_remote.sh` / dry-run scripts,
use `./deploy_mac/diag/view_prediction.sh` in a second terminal.

To validate: point the robot at an obstacle on one side with open space on the
other — the red/orange arrows should swing toward the opening, and the depth PiP
should show the obstacle. Also works on a sim2sim run (it subscribes
`/mujoco/lowstate` too).
