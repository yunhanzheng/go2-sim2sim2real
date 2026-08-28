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
- Jetson `go2-realsense.service` active and enabled (`ssh unitree@192.168.123.222`)

---

## 1. Physical + network

- Robot hoisted/tethered, clear space, hand on power
- D435i USB cable seated on the **Jetson**, not on the Ubuntu host
- Camera pitch near training (~20°: floor fade ≈ 2/3 down the frame)

| Device | Typical IP |
| --- | --- |
| Go2 motion controller | `192.168.123.161` |
| Onboard Jetson (camera) | `192.168.123.222` (sometimes `.18`) |
| Ubuntu host | e.g. `192.168.123.99` — **not** `.161`; **do not** use Jetson's `.222` |

```bash
ping -c 3 192.168.123.161
ping -c 3 192.168.123.222
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
ros2 topic hz /camera/depth/image_rect_raw       # Jetson depth: often ~15–60 Hz
```

No depth → on the Jetson: `sudo systemctl restart go2-realsense.service`, wait ~15 s.
If the topic uses a nested namespace, `ros2 topic list | grep image_rect_raw` and
override with `depth_input_topic:=...` at launch. Do **not** start a second `rs_launch`
while the service is already running.

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
  scene_xml:=resources/go2/scene_flyball70.xml
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
  ckpt:=models/D31-fovscan_maxpool_fullforce_fx_2m-smoothfloor-l40s/model_9000.pt \
  env_class_eq_17:=0 \
  action_delay:=1 \
  binary_vel:=1 \
  lin_vel_max:=1.0 \
  policy_kp:=35.0 \
  policy_kd:=0.6
```

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
| No `/camera/depth/...` | Restart Jetson `go2-realsense`; check IP clash with `.222`, `GO2_IFACE`, nested topic name |
| Wrong checkpoint flags | Bad `env_class_eq_17` / contact / yaw **does not error** — the robot just walks badly |

**Do not:**

- Engage before lay↔stand works and the area is clear
- Send `/lowcmd` while sport mode is on
- Use `GO2_IFACE=lo` on the real robot
- Run ROS launch from the `parkour` (py3.8) env
- Skip `kill_ros.sh` between sessions
- Set the Ubuntu host IP to the Jetson's `.222`
