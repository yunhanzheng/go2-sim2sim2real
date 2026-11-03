# Onboard run — all on the Jetson

Camera, policy and `/lowcmd` on the Orin NX; the laptop is only a terminal.
Camera driver: `INSTALL.md`.

**Hard smooth floor only, not carpet** — the checkpoint is trained at friction 1.0;
on carpet `|dlat|` goes 1.3 → 3.5 and the robot drifts backwards (see §4).

## 0. The full sequence

| # | Do | Where |
| --- | --- | --- |
| 1 | Cable laptop → Go2 ethernet port, power the robot on | — |
| 2 | `ping -c2 192.168.123.18`, then `ssh unitree@192.168.123.18` | laptop |
| 3 | Sport mode **OFF** in the app, wake the remote | app / remote |
| 4 | `go2 run` — self-check, launch, follow the log | Jetson |
| 5 | **B** — stand, wait for `[OK] stood up` | remote |
| 6 | Hold **L1+R1** — engage, watch for `[OK] \|dlat\|=1.3` | remote |
| 7 | `Ctrl-C` the log follower, then `exit` — the stack keeps running | Jetson |
| 8 | **Unplug the ethernet cable** | — |
| 9 | Drive with the remote | remote |
| 10 | **A** — lay down, then re-plug, ssh back, `go2 stop` | both |

**Never launch in a bare ssh foreground shell** — losing the link sends SIGHUP and
kills the stack, leaving nothing to hold a standing robot. Step 4's `tmux` is what
makes step 8 safe. Kill by hand: `tmux kill-session -t go2`, then
`pgrep -fc 'deploy_rl_policy/lib'` = 0 — press **A** first if standing, killing
`low_level_ctrl` drops the robot.

## 1. The cable

Safe and normal: everything — camera, policy, `low_level_ctrl`, remote — is on the
robot. You lose only the log and remote `kill`; `L2+R2` and the 0.5 s `/joy` dead-man
still stop it. Unplug **before walking** — the cable gets stepped on.

Missing host address: `sudo ip addr add 192.168.123.99/24 dev eno1 && sudo ip link set eno1 up`

## 2. Run

**Which model** — default is `smoothfloor-l40s/model_9000.pt`:

| checkpoint | steps | `env_class_eq_17` | |
| --- | --- | --- | --- |
| `D31-...-smoothfloor-l40s/model_9000.pt` | 9000 | 0 | **default** |
| `D31-...-smoothfloor-l40s/model_7000.pt` | 7000 | 0 | same `run_config`, earlier ckpt |
| `D31-...-l40s/model_9000.pt` | 9000 | 0 | no `run_config.yaml`; class assumed |
| `student/model_5000.pt` | 5000 | 0 | distilled, terrain lvl4 |
| `L00-02-flat-.../model_14500.pt` | 14500 | 1 | flat ground |

All D31-lineage weights share one architecture (29-dim proprio, same 7-layer depth
encoder), so any of them loads without changing a flag — only the path differs. New
weights go up from the laptop:

```bash
scp model_5000.pt unitree@192.168.123.18:~/go2_deploy/models/student/
```

**How** — robot lying down, one command:

```bash
go2 run                    # default
go2 run student            # distilled student
go2 run l00                # L00-02 preset
go2 run ~/go2_deploy/models/student/model_5000.pt    # any .pt
```

A `.pt` path runs at class 0; prefix `ENV17=1` for a flat-ground checkpoint.

Before launching it checks the camera service, the depth stream, `/lowstate` and a
stray `/joy` publisher, and refuses by name if one is missing. The banner must name
the file you asked for:

```
== ckpt : /home/unitree/go2_deploy/models/.../model_7000.pt
```

`Ctrl-C` stops the log follower, never the stack.

| | |
| --- | --- |
| `go2 watch` | follow the running stack |
| `go2 stop` | stop it; refuses while the robot is standing |
| `go2 log` | last ticks + error counters |

What it prints:

```
[--] lying            stand_diff=2.706
[..] standing up     stand_diff=0.82
[OK] stood up        stand_diff=0.21  <- hold L1+R1 to engage
[OK] |dlat|=1.30 age=54ms est vx=-0.01 oob=0
[!!] depth way off (|dlat|=3.2) - floor, or something within 2 m
```

**Without the wrapper** — the same launch, class 0 (D31 hurdle):

```bash
source ~/go2_deploy/env.sh
ros2 launch deploy_rl_policy distill.launch.py \
  is_simulation:=false use_wireless_joy:=true \
  ckpt:=$GO2_DEPLOY_ROOT/models/D31-fovscan_maxpool_fullforce_fx_2m-smoothfloor-l40s/model_9000.pt \
  ep_root:=$GO2_DEPLOY_ROOT/ep \
  env_class_eq_17:=0 action_delay:=1 binary_vel:=1 lin_vel_max:=1.0 \
  policy_kp:=35.0 policy_kd:=0.6
```

For L00-02 (flat, class 1) change two values:
`ckpt:=$GO2_DEPLOY_ROOT/models/L00-02-flat-baseline-delay-distillation-noised/model_14500.pt`
and `env_class_eq_17:=1`. The flag belongs to the checkpoint, not a free knob — a
wrong value does not error, the robot just walks badly.

Per 20 ms tick the actor sees `q`/`dq`, gyro, roll/pitch, four foot contacts, last
action and `vx`, plus a **32-dim depth latent** refreshed every 5th tick (10 Hz) from
a 58×87 frame normalised to [-0.5, 0.5] over 0–2 m. Output: 12 joint targets, PD at
kp 35 / kd 0.6.

## 3. Drive

| Input | Action |
| --- | --- |
| **B** | lay → stand (~2 s) |
| **L1+R1** hold | engage |
| stick forward | walk (only while engaged) |
| **A** | lay down |
| **L2+R2** | HOLD → PASSIVE → exit |

## 4. Check it

`[cmd]` prints once a second, `state_obs` every 5 s.

| Field | OK | Bad |
| --- | --- | --- |
| `\|dlat\|` | 1.28 – 1.41 on hard floor at rest (sim 1.24 – 1.30) | >2 scene, >3 carpet |
| `depth_age` | 50 – 66 ms standing idle, 85 – 97 ms while engaged | >150 ms |
| `oob` | flat | climbing every tick |
| `stand_diff` (5 s) | → <0.3 once standing | stuck near 2.6 = still lying |
| `est vx` at rest | ±0.02 | ±0.12+ |
| `tick overrun` | 0 – 3 over a whole run | several per minute |

Both `depth_age` values are normal — it rises when engaged, not a fault.

```bash
L=$(ls -t ~/go2_deploy/run_logs/real/*.log | head -1)
grep "\[cmd\]" $L | tail -5; grep -c "tick overrun" $L; top -bn1 | head -12
```

## 5. After editing code

`ws/install` holds **real copies** here, not symlinks:

```bash
S=~/go2_deploy/ws/src/deploy_rl_policy; I=~/go2_deploy/ws/install/deploy_rl_policy
cp $S/launch/distill.launch.py $I/share/deploy_rl_policy/launch/
cp $S/scripts/*.py $I/lib/deploy_rl_policy/
cd ~/go2_deploy/ws && colcon build --packages-select deploy_rl_policy \
  --cmake-args -DCMAKE_BUILD_TYPE=Release          # C++ only
```

## 6. Params — leave alone

```
is_simulation:=false  use_wireless_joy:=true  env_class_eq_17:=0
action_delay:=1  binary_vel:=1  lin_vel_max:=1.0  policy_kp:=35.0  policy_kd:=0.6
```

- `depth_latency_s` / `proprio_latency_s` — **omit**; launch forces 0/0 for real.
- `depth_rate_hz` — omit (10 Hz) or `0.0` (60 Hz). It selects which depth ages can
  exist, not throughput.
- `contact_threshold` 20 (raw N, not the sim's 2.0).
- One change per run.

## 7. Troubleshoot

| Symptom | Fix |
| --- | --- |
| ssh no route / hangs | cable, `ping 192.168.123.18`, host IP `.99` |
| drifts backwards, `\|dlat\|`≈3 | floor |
| `\|dlat\|`≈3 on wood too | obstacle within 2 m — turn away |
| engage does nothing | is `/rl/depth_image` publishing? |
| engaged flapping, crouches | two `/joy` publishers |
| `oob` climbing | `depth_rate_hz` |
| no depth topic | `sudo systemctl restart go2-realsense-humble` |
| `rclpy._rclpy_pybind11` | `export PATH=/usr/bin:$PATH` |
