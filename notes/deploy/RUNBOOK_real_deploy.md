# Go2 Real-Deploy Runbook (2026-07-27, fused stack)

Checkpoint: **D22-nextgoal0.5-orient-4090/model_9000** (5/6 fused ROS, 18/20 standalone).

## 0. One-time (already done on this robot — verify only)

- Jetson services installed: `go2-realsense.service` + watchdog (`deploy_mac/jetson/INSTALL.md`)
- Passwordless helpers: `/etc/sudoers.d/go2-realsense-watchdog`, `/etc/sudoers.d/go2-clock-sync`, `/usr/local/sbin/go2_set_clock.py`
- chrony disabled on the Jetson
- SSH key on the Jetson (`ssh unitree@192.168.123.222 echo ok` — no password)

## 1. Physical

- [ ] Robot hoisted/tethered, clear space, hand on power
- [ ] Camera cable seated (the snug one; watchdog covers flakiness, not disconnection)
- [ ] Camera pitch: viewer floor-fade ~2/3 down the frame ≈ trained 20°.
      43%-from-top = the 5° shallow mount → costs ~1 run in 6, re-tilt if possible

## 2. Robot bring-up

Sport mode OFF — run **on the Mac, in a CLEAN terminal (do NOT source env.sh
first**: the ros_humble env leaks conflicting CycloneDDS libs into the
standalone unitree_sdk2py and it segfaults before sending the release):

```bash
PYTHONPATH=$HOME/go2/unitree_sdk2_python $HOME/go2/venv/bin/python $HOME/go2/disable_sport.py
```

Success = a `CheckMode: code=0` line + release confirmation. A segfault after
only the `pose:` line means sport mode was NOT disabled — retry from a clean shell.

## 3. Mac environment + checks

Every terminal you use (checks, dry-run, real run):

```bash
cd $REPO
source deploy_mac/env.sh        # conda ros_humble + unitree msgs + CycloneDDS on GO2_IFACE
```

Defaults are correct for the real robot: `GO2_IFACE=en13` (the Ethernet link to
the Go2 — override only if the Mac uses a different interface: `GO2_IFACE=enX source ...`).
**Never `GO2_IFACE=lo0` here** — that is the sim-isolation setting.

Then verify:

```bash
ros2 topic hz /camera/depth/image_rect_raw    # ~30 Hz (Jetson service auto-starts)
ros2 topic hz /lowstate                       # ~500 Hz
```

Camera silent → `ssh unitree@192.168.123.222 sudo -n systemctl restart go2-realsense.service`, wait 15 s.

## 4. Dry-run first (no /lowcmd, robot cannot move)

```bash
SECS=60 ENV_CLASS_EQ_17=0 YAW_HEAD=0 BINARY_VEL=1 LIN_VEL_MAX=1.0 LATENCY_LOG=1 \
CKPT=models/D22-nextgoal0.5-orient-4090/model_9000.pt \
./deploy_mac/run/run_dryrun_remote.sh
```

Pass criteria (status line in rl_policy.log / latency_summary):

- clock sync succeeded non-interactively (no password prompt)
- `depth_age` ~80–115 ms, `early=0`, `skew=0`, no `>180 ms` entries
- actor ≈ 50 Hz, `|dlat|` finite, est velocities sane

## 5. Real run

```bash
ENV_CLASS_EQ_17=0 YAW_HEAD=0 BINARY_VEL=1 LIN_VEL_MAX=1.0 LATENCY_LOG=1 \
POLICY_KP=35.0 POLICY_KD=0.6 \
CKPT=models/D22-nextgoal0.5-orient-4090/model_9000.pt \
./deploy_mac/run/run_real_remote.sh
```

Defaults already correct: FUSED=1, CONTACT_THRESHOLD=20,20,20,20 (real units),
DEPTH_LATENCY_S=0.08, PROPRIO_LATENCY_S=0.016, SYNC_JETSON_CLOCK=1,
diagnostics off (auto — set RECORD_VIEW=1 or FULL_LOG=1 only if you need the viewer/bag,
costs ~0.5 Hz actor rate). `check_deploy_contract.py` aborts on ckpt/flag mismatch.

If the script prints a `renice` note: run the shown `sudo renice -10 -p <pid>` once
(kills the rare 35–52 ms scheduler stalls).

## 6. Operating

0. **Wake the remote first** — the firmware streams /wirelesscontroller only
   while the remote is awake; an asleep remote = no reaction to any button.
   Check `wireless_to_joy` heartbeats in the log: msg count must be GROWING.
1. **B** = stand
2. Hold **L1+R1** = engage → **hold standing ~2 s** (students train with a standstill prefix)
3. Left stick forward = walk (bang-bang past deadzone)
4. Release L1+R1 = settle to stand. **A** = lay. **L2+R2** = quit. Hard stop: both triggers floored.

## 7. Cleanup between runs

```bash
bash deploy_mac/kill_ros.sh    # kills every stack node by name (policy, camera, low_level, viewer, recorders)
```

Run it after every session or aborted launch — stale nodes silently steal DDS
topics from the next run. It also kills `distill_runner`/viewers, so never run
it while anything you care about is alive.

## 8. Live viewer (optional)

```bash
# separate terminal, env sourced:
MUJOCO_GL=glfw KMP_DUPLICATE_LIB_OK=TRUE python deploy_mac/diag/view_prediction.py
```

Dry-runs feed it out of the box. For a REAL run launch with `RECORD_VIEW=1`
(diagnostics are otherwise off). Costs ~0.5 Hz actor rate — fine for
commissioning, skip for max-performance runs.

## 9. Never / gotchas

- Never `lockstep:=true` on deploy-predictive runs (measured 1/10)
- Never a second `rs_launch` by hand — restart the service instead
- Never run sim batches/standalone evals on this Mac during a session (`kill_ros` kills by node name)
- `ENV_CLASS_EQ_17=0` is REQUIRED for D22 (cliffside); =1 silently breaks the policy
- `EP_USE_YAW` must be unset/0 (32-D no-yaw student)
- Logs land in `run_logs/dryrun/<timestamp>/`; `latency_summary.txt` is the post-run health check
