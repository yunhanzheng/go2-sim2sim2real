# Jetson RealSense service — one-time install (2026-07-27)

## Quick path (state as of 2026-07-27 session)

Files are already staged (`/tmp/` units + `~/go2/realsense_watchdog.sh`
installed, executable; old `realsense-depth.service` inactive with zero
camera processes). Only the sudo block remains — paste on the Jetson:

```bash
sudo systemctl disable realsense-depth.service
sudo cp /tmp/go2-realsense.service /tmp/go2-realsense-watchdog.service /etc/systemd/system/
echo 'unitree ALL=(root) NOPASSWD: /bin/systemctl restart go2-realsense.service' \
  | sudo tee /etc/sudoers.d/go2-realsense-watchdog
sudo systemctl daemon-reload
sudo systemctl enable --now go2-realsense.service go2-realsense-watchdog.service
```

Then verify (Jetson **and** Mac — Mac visibility is what the deploy script
gates on; on the Jetson export the CycloneDDS env first or trust the Mac check):

```bash
ros2 topic hz /camera/depth/image_rect_raw               # ~30 Hz
ros2 param get /camera/camera depth_module.laser_power   # 350.0
journalctl -u go2-realsense-watchdog -n 10 --no-pager
```

The full from-scratch procedure follows.

Replaces the old boot launch with a supervised setup that fixes, long-term:

1. **`laser_power=350` persisted** — was set at runtime on 2026-07-24 (fewer
   depth holes on low-texture) and silently lost on every reboot.
2. **Crash recovery** — `Restart=always` brings a dying driver back.
3. **Wedge recovery** — a watchdog probes real frame flow every 30 s and
   restarts the driver when the node is alive but delivers nothing (the
   advertised-topics-zero-frames state from 2026-07-27, which `Restart=`
   can never catch because the process doesn't exit).

## Install (from the Mac, on the robot network)

```bash
cd deploy_mac/jetson
scp go2-realsense.service go2-realsense-watchdog.service realsense_watchdog.sh \
    unitree@192.168.123.222:/tmp/
ssh unitree@192.168.123.222
```

Then on the Jetson:

```bash
# 1. FIRST find and disable whatever starts rs_launch today, or the old and
#    new launches will fight over the USB device (both crash-loop, 'resource busy'):
systemctl list-units --type=service | grep -i -E 'real|camera|rs'
grep -rl rs_launch /etc/systemd/system/ ~/.config/ /etc/rc.local 2>/dev/null
crontab -l | grep -i rs_launch
#    -> sudo systemctl disable --now <old-service>   (or remove the crontab/rc.local line)

# 2. install the new units
mkdir -p ~/go2
cp /tmp/realsense_watchdog.sh ~/go2/ && chmod +x ~/go2/realsense_watchdog.sh
sudo cp /tmp/go2-realsense.service /tmp/go2-realsense-watchdog.service /etc/systemd/system/

# 3. let the watchdog restart the driver without a password (visudo -f):
echo 'unitree ALL=(root) NOPASSWD: /bin/systemctl restart go2-realsense.service' \
  | sudo tee /etc/sudoers.d/go2-realsense-watchdog

# 4. enable + start
sudo systemctl daemon-reload
sudo systemctl enable --now go2-realsense.service go2-realsense-watchdog.service

# 5. verify
ros2 topic hz /camera/depth/image_rect_raw          # ~30 Hz
ros2 param get /camera/camera depth_module.laser_power   # 350.0
journalctl -u go2-realsense-watchdog -f              # probe heartbeat
```

Also verify from the **Mac** (`ros2 topic hz ...`) — Mac-side DDS visibility is
what `run_real_remote.sh` gates on.

## Clock-sync passwordless helper (added 2026-07-27, second sudo block)

`sync_jetson_clock.sh` runs a `time.clock_settime` as root on the Jetson; the
original flow prompts for sudo, which breaks non-interactive deploys. Install
the root-owned helper + scoped sudoers line (files staged in `/tmp` by the Mac):

```bash
sudo cp /tmp/go2_set_clock.py /usr/local/sbin/go2_set_clock.py
sudo chown root:root /usr/local/sbin/go2_set_clock.py
sudo chmod 755 /usr/local/sbin/go2_set_clock.py
echo 'unitree ALL=(root) NOPASSWD: /usr/bin/python3 /usr/local/sbin/go2_set_clock.py *' \
  | sudo tee /etc/sudoers.d/go2-clock-sync
```

(Root ownership is the point — a unitree-writable NOPASSWD target would be a
free root escalation.) `sync_jetson_clock.sh` tries this path first and falls
back to interactive sudo when absent.

## Operational notes

- Never launch a second `rs_launch` by hand while the service runs.
  To bounce the camera: `sudo systemctl restart go2-realsense`.
- The watchdog needs 2 consecutive missed probes (~40-70 s worst case) before
  restarting — a single dropped frame won't bounce the camera mid-run. A
  restart mid-run costs ~15 s of depth; the Mac-side stall guard (stale-latent
  cease + low_level settle) covers the gap.
- `SYNC_JETSON_CLOCK=1` in `run_real_remote.sh` is still required — this
  service does not manage clocks. The Jetson runs **chrony** (not timesyncd;
  checked 2026-07-27, active, NTPSynchronized=no) — it can step the clock
  mid-run and break the depth clock-offset tracker. Disable it once:
  `sudo systemctl disable --now chrony` (sync_jetson_clock.sh at run start
  is then the single clock authority).
