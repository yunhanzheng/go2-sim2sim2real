# Runbook — REAL run on the Go2 (Ubuntu, system ROS, `/lowcmd`)

Ubuntu / system-ROS path. Mac / RoboStack → [`real_run_runbook.md`](real_run_runbook.md).
Same stack and safety rules; environment and launch commands below are for native Ubuntu
with **system** `/opt/ros/humble` plus a built `unitree_ros2` workspace (example:
`~/go2/ros2_ws`). This is **not** the RoboStack/conda-ROS path — use
[`deploy_mac/env.sh`](../env.sh) on macOS instead.

Repo: `$REPO` (default `~/data/go2-sim2sim2real`).

> ⚠️ **Hoist or tether the robot.** There is no real watchdog; the e-stop is a fragile
> gamepad combo. See [`DEPLOY.md`](../../DEPLOY.md).

Jetson RealSense one-time install (only if the service is not already installed):
[`../jetson/INSTALL.md`](../jetson/INSTALL.md). Depth pipeline overview:
[`../README.md`](../README.md).

---

## 0. One-time setup (verify)

- ROS 2 Humble installed (`ls /opt/ros/humble/setup.bash`)
- `unitree_go` / `unitree_api` built and sourced (e.g. `~/go2/ros2_ws`)
- This repo built: `cd ~/data/go2-sim2sim2real && colcon build --symlink-install`
- Training repo at `~/data/go2-flyball-parkour` (or set `$EP_ROOT`)
- **System Python 3.10** with Torch + MuJoCo — do **not** run ROS from the `parkour`
  conda env (Python 3.8); see §9
- Wired NIC to the Go2 identified (often `eno1`)
- Jetson `go2-realsense-humble.service` + `go2-realsense-humble-watchdog.service`
  active and enabled (`ssh unitree@192.168.123.18`)

---

## 1. Physical + network

- Robot hoisted/tethered, clear space, hand on power
- D435i USB cable seated on the **Jetson**, not on the Ubuntu host
- Camera pitch near training (~20°: floor fade ≈ 2/3 down the frame)

| Device | Typical IP |
| --- | --- |
| Go2 motion controller | `192.168.123.161` |
| Onboard Jetson (camera) | `192.168.123.18` (`.222` belongs to the old Foxy image; it does not answer) |
| Ubuntu host | e.g. `192.168.123.99` — **not** `.161`; **do not** use Jetson's `.18` |

```bash
ping -c 3 192.168.123.161
ping -c 3 192.168.123.18
ip -br addr | grep 192.168.123    # note NIC name, e.g. eno1
```

---

## 2. Sport mode off

Sport mode must be disabled or the factory controller will fight `/lowcmd`. Re-check
after every power cycle. Use your local `unitree_sdk2py` / `disable_sport` script;
success looks like `CheckMode: code=0` plus a release confirmation.

---

## 3. Environment (every terminal)

```bash
conda deactivate 2>/dev/null; conda deactivate 2>/dev/null
export PATH="/usr/bin:$PATH"
which python3 && python3 --version    # expect /usr/bin/python3, 3.10.x

REPO=/home/ubuntu22/data/go2-sim2sim2real
cd "$REPO"

source /opt/ros/humble/setup.bash
source ~/go2/ros2_ws/install/setup.bash
source "$REPO/install/setup.bash"

export EP_ROOT=/home/ubuntu22/data/go2-flyball-parkour
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=0

export GO2_IFACE=eno1   # replace with your wired NIC
export CYCLONEDDS_URI="<CycloneDDS><Domain><General><Interfaces>
  <NetworkInterface name=\"${GO2_IFACE}\" priority=\"default\" multicast=\"default\"/>
</Interfaces></General></Domain></CycloneDDS>"
```

Verify:

```bash
ros2 topic hz /lowstate                          # ~500 Hz
ros2 topic hz /camera/depth/image_rect_raw       # Jetson depth: ~60 Hz (480x270x60)
```

No depth → on the Jetson: `sudo systemctl restart go2-realsense-humble.service`, wait
~20 s. `ExecStartPost` re-applies `laser_power`/`emitter_enabled` on every start and the
unit stays `activating` until it succeeds (or its 60 s retry budget runs out) — depth is
often already streaming in that state. Do **not** start a second `rs_launch` while the
service is running: both fight over `/dev/video0` and crash-loop on `resource busy`.

The service flattens the node namespace to `/camera`, so the topic is exactly
`/camera/depth/image_rect_raw`. Launching the driver by hand instead gives the 4.x
double namespace `/camera/camera/depth/image_rect_raw`; confirm with
`ros2 topic list | grep image_rect_raw`. Two different knob names feed it:
`depth_input_topic:=...` under `distill.launch.py` (a launch arg), or
`--ros-args -p input_topic:=...` when running `depth_camera_node.py` directly.

---

## 4. Depth viewer (optional)

Raw `/camera/depth/image_rect_raw` is **not** the policy input. The policy consumes
preprocessed `/rl/depth_image` (58×87). To inspect it standalone:

```bash
# Terminal A — preprocessing (after §3)
python3 src/deploy_rl_policy/scripts/depth_camera_node.py

# Terminal B — popup window
python3 src/deploy_rl_policy/scripts/depth_viewer_node.py
```

`ros2 run deploy_rl_policy depth_viewer_node.py` may fail with `No executable found`;
use the `python3 src/...` form above.

A full real launch (`is_simulation:=false`) starts `depth_camera_node` automatically —
only Terminal B is needed. In sim, depth comes from `depth_sim_node`; the same viewer
works on `/rl/depth_image`.

---

## 5. Sim dry-run (optional)

```bash
ros2 launch deploy_rl_policy distill.launch.py is_simulation:=true \
  use_keyboard:=false \
  ckpt:=models/D31-fovscan_maxpool_fullforce_fx_2m-smoothfloor-l40s/model_9000.pt \
  scene_xml:=resources/go2/scene_flyball45_2h_h10_noffset.xml
```

Do **not** engage on hardware until hoisted.

---

## 6. Real: lay ↔ stand first

Verify FSM + PD **before** loading a walking policy. Hoisted; **do not engage** yet.

The default launch starts Xbox `joy_node`. For the **Unitree remote**, run the bridge
yourself:

```bash
# Terminal A — remote → /joy (wake the remote first)
python3 deploy_mac/run/wireless_to_joy.py

# Terminal B
ros2 run deploy_rl_policy low_level_ctrl --ros-args \
  -p is_simulation:=false \
  -p policy_kp:=35.0 -p policy_kd:=0.6
```

Heartbeat `msg count` in `wireless_to_joy` must increase.

| Input | Action |
| --- | --- |
| **B** | lay → stand |
| **A** | lay down |
| Hold **L1+R1** (Xbox: LB+RB; keyboard: space) | engage policy |
| Stick forward / **W** | walk (only while engaged) |
| **L2+R2** | exit |

---

## 7. Real: full policy stack

Hoisted. Sport off. Confirm `/camera/depth/image_rect_raw` and `/lowstate` are still
publishing. Start `wireless_to_joy.py` in a separate terminal if using the Unitree remote
(the launch file does not start it for you).

```bash
ros2 launch deploy_rl_policy distill.launch.py is_simulation:=false \
  use_joy:=false \
  ckpt:=models/D31-fovscan_maxpool_fullforce_fx_2m-smoothfloor-l40s/model_9000.pt \
  env_class_eq_17:=0 \
  action_delay:=1 \
  binary_vel:=1 \
  lin_vel_max:=1.0 \
  policy_kp:=35.0 \
  policy_kd:=0.6
```

This exact command is the field-verified configuration on the Go2 remote (2026-10-20:
B stands, hold L1+R1 and the policy runs without the engage flapping).

`use_joy` picks **which** `/joy` source runs, and the rule is that exactly one must
exist:

| Input device | Command | `/joy` publisher |
| --- | --- | --- |
| Physical pad (Xbox, LB+RB) | default `use_joy:=true` | `joy_node` |
| Unitree remote | `use_joy:=false` + run `wireless_to_joy.py` | `wireless_to_joy` |
| Keyboard | `use_keyboard:=true` | `keyboard_to_joy` |

If two of them publish at once they carry contradictory button states: the one you hold
says L1+R1 down, the idle one says all buttons up. Under the real-mode default
`momentary_engage:=true` that flips `/policy_engaged` off and on at the message rate
(~15-30 Hz). Each engage edge resets the encoder GRU and the actor buffer, and each
release re-arms the settle-to-stand ramp, whose first tick captures
`q_init_ = motor[i].q` — so the commanded target follows the sagging joints down instead
of driving to the stand pose. The observable symptom is "hold L1+R1 and the robot
crouches", and the `/rl` log shows dozens of `/policy_engaged=True/False` pairs per
second. Check with `ros2 topic info /joy` → `Publisher count: 1`.

Flat baseline (L00-02) example:

```bash
ros2 launch deploy_rl_policy distill.launch.py is_simulation:=false \
  ckpt:=models/L00-02-flat-baseline-delay-distillation-noised/model_14500.pt \
  env_class_eq_17:=1 \
  action_delay:=1
```

Real `contact_threshold` uses raw foot-force units — do **not** copy the sim value `2.0`.
See [`contact_threshold_calibration_2026-07-24.md`](../../notes/analysis/contact_threshold_calibration_2026-07-24.md).

**Sequence:** wake remote → **B** stand ~2 s → hold **L1+R1** to engage → stick forward →
release shoulder buttons to stand / **A** to lay; hard stop: cut power or **L2+R2**.

---

## 8. Between runs

```bash
bash "$REPO/deploy_mac/kill_ros.sh"
```

---

## 9. Troubleshooting

| Symptom | Fix |
| --- | --- |
| `No module named 'rclpy._rclpy_pybind11'` | Exit conda; `export PATH=/usr/bin:$PATH` |
| `depth_viewer_node` → `No executable found` | `python3 src/deploy_rl_policy/scripts/depth_viewer_node.py` |
| `wireless_to_joy` heartbeat stuck | Wake the Unitree remote |
| Crouches / collapses **on engage**, `/rl` log shows `engaged=True/False` flapping | Two `/joy` publishers: relaunch with `use_joy:=false`, and check `ros2 topic info /joy` is `Publisher count: 1` |
| Two `depth_camera_node` / two `distill_fused_policy` | Leftover stack: `ps -eo pid,args \| grep -E "depth_camera_node\|distill_fused_policy"`, kill the old PID; `/rl/target_pos` must show `Publisher count: 1` |
| No `/camera/depth/...` | `ssh unitree@192.168.123.18 sudo systemctl restart go2-realsense-humble`; then check `is-active`, an IP clash with `.18`, `GO2_IFACE`, and whether the topic picked up a nested namespace |
| Wrong checkpoint flags | Bad `env_class_eq_17` / contact / yaw **does not error** — the robot just walks badly |

**Do not:**

- Engage before lay↔stand works and the area is clear
- Send `/lowcmd` while sport mode is on
- Use `GO2_IFACE=lo` on the real robot
- Run ROS launch from the `parkour` (py3.8) env
- Run `joy_node` and `wireless_to_joy.py` at the same time (two `/joy` publishers →
  `/policy_engaged` flaps → the robot crouches the moment you engage)
- Skip `kill_ros.sh` between sessions
- Set the Ubuntu host IP to the Jetson's `.18`
