# Jetson RealSense service — install

Two images exist in the field and the install differs:

| Image | Unit | Driver |
| --- | --- | --- |
| **Orin NX, JetPack 6, Ubuntu 22.04, ROS 2 Humble** (current robot) | `go2-realsense-humble.service` | `ros-humble-realsense2-camera` 4.58.x |
| Jetson-Nano era, Ubuntu 18.04, ROS 2 Foxy (legacy, see the last section) | `go2-realsense.service` | `ros-foxy-realsense2-camera` |

Tell them apart with `ls /opt/ros/`.

The Jetson's address on the robot's internal network is **`192.168.123.18`**
(`enP8p1s0`). `192.168.123.222`, used throughout older notes, is stale — it does not
answer.

## Humble / JetPack 6 (current robot)

State as of 2026-10-20: installed, enabled, verified at 59.9 Hz across DDS from the
Ubuntu host (`eno1` ↔ `enP8p1s0`). Reinstall from scratch with the steps below.

### 0. Prerequisite check

```bash
ls /opt/ros/                                          # expect: humble
ls /opt/ros/humble/share | grep realsense2_camera     # expect: the package
rs-enumerate-devices -s                               # expect: Intel RealSense D435I
cat ~/unitree_ros2/setup_go2.sh                       # RMW + CYCLONEDDS_URI (enP8p1s0)
```

`setup_go2.sh` is the single source of the DDS environment. Both units `source` it, so
the services see exactly what an interactive shell sees. Do not re-declare
`CYCLONEDDS_URI` inside the unit — a second copy drifts.

`lsusb` seeing the camera only proves USB enumeration; it says nothing about whether
anything is publishing.

### 1. Stop anything that owns the camera

```bash
pkill -f realsense2_camera_node
systemctl list-unit-files | grep -iE 'real|camera'     # no old unit may remain enabled
```

Two `rs_launch` processes fight over `/dev/video0` and both crash-loop on `resource
busy`.

### 2. Install the units

Files in this directory: `go2-realsense-humble.service`,
`go2-realsense-humble-watchdog.service`, `go2-usb-role-host.service`,
`realsense_watchdog_humble.sh`.

```bash
mkdir -p ~/go2
cp realsense_watchdog_humble.sh ~/go2/
sudo install -m 644 go2-realsense-humble.service go2-realsense-humble-watchdog.service \
  go2-usb-role-host.service /etc/systemd/system/

echo 'unitree ALL=(root) NOPASSWD: /bin/systemctl restart go2-realsense-humble.service' \
  | sudo tee /etc/sudoers.d/go2-realsense-watchdog-humble
sudo chmod 440 /etc/sudoers.d/go2-realsense-watchdog-humble

sudo systemctl daemon-reload
sudo systemctl enable --now go2-usb-role-host.service      # port role, see the USB section
sudo systemctl enable --now go2-realsense-humble.service
```

`install -m 644` rather than `cp`: the source files are group-writable (`-rw-rw-r--`),
which systemd reports as an unsafe unit.

`chmod 440` is mandatory — sudo silently ignores a group/world-readable drop-in, and
the watchdog then fails to restart the camera with nothing in the journal.

### 3. Verify

```bash
sleep 25
systemctl is-active go2-realsense-humble          # active
journalctl -u go2-realsense-humble -b --no-pager | grep -E "Open profile|Node Is Up"
# expect: Open profile: ... Width: 480, Height: 270, FPS: 60
ros2 param get /camera depth_module.laser_power    # 360.0
```

`systemctl start` blocks on `ExecStartPost` (up to 60 s of retries applying
`laser_power`), so status can read `activating` while depth is already streaming. That
is not a failure, and the watchdog stays `inactive` until the driver's start job
finishes — its unit is `After=go2-realsense-humble.service`.

Then the watchdog, then a reboot to prove auto-start:

```bash
sudo systemctl enable --now go2-realsense-humble-watchdog.service
sudo reboot
```

Watchdog log should carry `persistent subscriber on /camera/depth/image_rect_raw ...
first restart no sooner than 30.0s`.

Cross-machine check from the Ubuntu host (`eno1` is that host's NIC):

```bash
ros2 topic hz /camera/depth/image_rect_raw        # ~60 Hz
```

## USB port: switch the Type-C role to host

The dock's `USB3.2 Type-C` port is a **dual-role (DRP) port**, and its role is not
`host` by default. That is the whole reason a camera on it produces **no dmesg output
at all** — the controller never sees a device, so no VBUS is raised either. One sysfs
write fixes it:

```bash
echo host | sudo tee /sys/class/usb_role/usb2-0-role-switch/role
cat /sys/class/usb_role/usb2-0-role-switch/role       # host
```

The camera then enumerates on **Bus 002** as a SuperSpeed device:

```
/:  Bus 02.Port 1: Dev 1, Class=root_hub, Driver=tegra-xusb/4p, 10000M
    |__ Port 1: Dev 2, Class=Video, Driver=uvcvideo, 5000M      <- 5 video ifs + 1 HID
```

and the driver logs `Device USB type: 3.2` (on the USB-A port it says `2.1 ...
Reduced performance is expected`). Measured 2026-10-20: **59.89 Hz, std dev 0.00013 s**
over 361 frames — tighter than the 480 Mbps path (0.00024 s).

There is exactly one role switch on this image:

```bash
ls /sys/class/usb_role/            # usb2-0-role-switch
```

### Persist it at boot

The role resets on every reboot, so `go2-usb-role-host.service` (in this directory)
makes the switch at start-up:

```
[Unit]
Description=Switch dock USB3.2 Type-C port to host role
Before=go2-realsense-humble.service

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/bin/sh -c 'echo host > /sys/class/usb_role/usb2-0-role-switch/role'

[Install]
WantedBy=multi-user.target
```

```bash
sudo install -m 644 go2-usb-role-host.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now go2-usb-role-host.service
systemctl status go2-usb-role-host.service --no-pager | head -6
```

Live test without rebooting — push the role back to `device`, then let the unit restore
it:

```bash
echo device | sudo tee /sys/class/usb_role/usb2-0-role-switch/role
sudo systemctl restart go2-usb-role-host.service
cat /sys/class/usb_role/usb2-0-role-switch/role      # host
```

Do **not** use `/etc/rc.local` for this:

- `systemctl enable rc-local.service` reads back `enabled-runtime`: the symlink lands in
  `/run/systemd/system`, so the enable does not survive a reboot.
- The unit ships static in `/lib/systemd/system/rc-local.service` (plus a `debian.conf`
  drop-in) with `ConditionFileIsExecutable=/etc/rc.local`. `rc-local`'s generator only
  adds the `multi-user.target.wants` symlink when `/etc/rc.local` is executable, so a
  truncated or non-`+x` file either never runs or fails with `Exec format error`.
  This image does not ship the file at all.

Verified across a reboot on 2026-10-20:

| Check | Value |
| --- | --- |
| `go2-usb-role-host.service` | `active (exited)`, `status=0/SUCCESS`, started 25 s into boot |
| `cat .../usb2-0-role-switch/role` | `host` |
| `lsusb -t` | `Bus 02 ... 10000M` → `Port 1: Dev 2 ... 5000M` ×6 |
| journal | `Device USB type: 3.2`, `Open profile: ... 480, Height: 270, FPS: 60`, `RealSense Node Is Up!` |
| `ros2 topic hz` | 59.885 Hz, std dev 0.00023 s, 481 frames |
| `laser_power` / `emitter_enabled` | `360.0` / `1` (from `ExecStartPost`) |
| `/etc/rc.local` | 0 bytes, `-rw-r--r--`, `rc-local.service` stays `inactive (dead)` |

Ordering is not what saves the boot: `After=network-online.target` in the camera unit
delayed its start to 12:13:05, with boot at 12:11 and the role set at 12:11:37 — nearly
90 s apart. `realsense2_camera` also retries while no device is present, so `Before=` is
belt-and-braces rather than a hard requirement. Expect `systemctl is-active
go2-realsense-humble` to read `activating` for the first ~2 min after boot — that is the
start job, not a failure. A restart is still deterministic:

```bash
sudo systemctl restart go2-realsense-humble
```

### Dead ends (all tested 2026-10-20 — do not repeat)

None of the following is the cause — every item was verified while the port stayed
silent:

| Lead | Result |
| --- | --- |
| `fusb301` probe failure at boot (`error -22`, I2C `0xffffff87`) | background noise; the same lines appear with the port working |
| PADCTL `0x02430030` via `devmem2 ... w 0x004`, run-time **and** at boot through `rc.local` | `Written 0x4; readback 0x4`; the value was already `0x004`, no effect |
| GPIO `PP.06` (`gpiochip0` offset 98 = global 446) | stays `out hi` the whole time; unrelated |
| `/sys/class/regulator/` | contains **no** USB VBUS domain, so the supply is not software-gated at all |

Unitree support's recipe (`/etc/rc.local` + `devmem2 0x02430030 w 0x004`) targets a
reset PINMUX, which was not the fault here.

The lesson: an empty `dmesg` on plug-in means the controller never saw a device. Check
the **port role** before chasing power.

### Tooling notes

- `/sys/class/gpio` does **not** exist on this image (GPIO sysfs disabled), so the
  `echo 446 > /sys/class/gpio/export` recipe from older notes cannot work. `gpioset`
  (libgpiod, preinstalled) is the only path; the image already runs
  `gpioset -m signal gpiochip0 98=1` at boot.
- `busybox devmem` takes `ADDR [WIDTH [VALUE]]`; the two-argument form fails with
  `devmem: invalid number '0x004'`.
- `devmem2` is in jammy/universe, but **this Jetson has no internet** — `apt install`
  dies with `Temporary failure resolving 'ports.ubuntu.com'`. Fetch the **arm64** `.deb`
  from `ports.ubuntu.com` on the host and `scp` it over; an `amd64` package will not
  install.

### Diagnosis rule

| `lsusb` | `dmesg` at plug-in | Meaning |
| --- | --- | --- |
| no `8086:0b3a` | no event at all | controller saw nothing — check port **role** first, then VBUS |
| no `8086:0b3a` | `failed to enumerate` / `error -71` | powered but link is bad — cable |
| `8086:0b3a` on `2-1` | `new SuperSpeed USB device` | working |

If the driver is already running when the camera powers up, `realsense2_camera` does
retry and will attach once the device appears, but a restart is deterministic:

```bash
sudo systemctl restart go2-realsense-humble
```

## Driver 4.58 traps (all hit on 2026-10-20)

| Symptom | Cause | Fix |
| --- | --- | --- |
| depth comes up 640x480@15 | `depth_module.profile` is no longer a parameter and is silently dropped | `depth_module.depth_profile` (+ `infra_profile`) |
| `malformed launch argument 'camera_namespace:='` | an empty namespace value is rejected | `camera_namespace:=/` |
| `depth_module.laser_power` stays `150.0` | not a launch arg on this driver | set it post-start, in `ExecStartPost` |
| setting 350 → `doesn't comply with floating point range` | descriptor declares `step = 30` over `[0, 360]` | use **360** (the Foxy unit's 350 is illegal here) |

Why 480x270x60: depth FOV at 480x270 is 89.2 x 58.0, which frames the trained
87 x 56.49 with zero padding (640x480 is 78.8 x 63.3 → 13% of every frame padded with
`DEPTH_FAR`), and 60 fps keeps depth age inside the trained 80–100 ms band (15 fps gives
80–147 ms). This is a FOV-and-latency choice, not a bandwidth one.

On the Type-C port the camera lands on `Bus 002` with `Device USB type: 3.2` (5 Gbps), so
nothing is bandwidth-limited. On a USB-A port it enumerates on `Bus 001` as `usb 1-3` and
the journal says `Device USB type: 2.1 ... Reduced performance is expected`; there USB 2
caps 848x480 at 10 fps and 640x480 at 30, and 480x270 is the only mode offering 60. Both
paths deliver the same 480x270x60 stream, so the contract holds either way.

`camera_namespace:=/` also matters for the deploy stack: it flattens the node to
`/camera`, so depth lands on `/camera/depth/image_rect_raw` — the name
`run_real_remote.sh` and the viewer scripts hardcode. Launch the driver by hand instead
and the topic becomes `/camera/camera/depth/image_rect_raw`. Two different knobs feed
the preprocessor: `depth_input_topic:=...` under `distill.launch.py` (a launch arg), or
`--ros-args -p input_topic:=...` when running `depth_camera_node.py` directly.

## Legacy: Foxy image

`go2-realsense.service`, `go2-realsense-watchdog.service`, `realsense_watchdog.sh` and
`go2_set_clock.py` in this directory belong to the older Ubuntu 18.04 / Foxy image
(`/opt/ros/foxy`, `~/cyclonedds_ws`). They use `depth_module.profile:=480x270x60` and
`laser_power=350`, both legal there and both rejected by 4.58. Their watch paths also
default to `~/cyclonedds_ws`. Never enable them alongside the Humble units.
