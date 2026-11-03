# Jetson clock — why it is wrong after every power-on, and the fix

State as of 2026-10-29: installed on the robot (`go2-clock-sync.timer`, enabled),
verified stepping ±2 h and −3 h back to the host clock within 30 s with no
manual step, and verified silent (`Result=success`, empty journal) while the
host is absent. The host side runs as the user unit `go2-time-server.service`
(enabled, `Linger=yes`).

## The failure

```
$ timedatectl
               RTC time: Thu 1970-01-01 22:51:52        <- no working RTC
System clock synchronized: no
```

The Orin's RTC reads 1970, and the robot's network has **no default route** — the
only route is `192.168.123.0/24` itself. `systemd-timesyncd` is enabled and
active but has no reachable server, so on every boot it falls back to the mtime
of `/var/lib/systemd/timesync/clock`, i.e. **the moment of the last shutdown**.

This is not "slightly off": every power cycle silently loses all elapsed time.
Observed 2026-10-29: 72 days behind at first inspection, then 62 minutes behind
after a reboot two hours later. Anything that compares timestamps across
machines (bag replay, `camera_metadata_latency.py`, latency logs) is wrong until
it is corrected, and TLS/apt fail with "not yet valid".

`tailscale0` exists but is logged out and has no route, so there is no internet
path to borrow either.

## The fix

The only reachable clock authority is the deployment host on
`192.168.123.0/24`, so the host serves its clock and the Jetson pulls:

```
host 192.168.123.99                                   jetson 192.168.123.18
  time_server.py :8123/tcp  ◀──── pull, every 30 s ──── go2_sync_clock.py
  (no root, high port)                                  (root, systemd timer)
```

The Jetson side converges whenever a host appears — at boot, or minutes later
when the operator turns the laptop on. Nothing to remember after a power cycle.

## Host side (one-time)

```bash
./deploy_mac/lib/install_time_server.sh        # binds the 192.168.123.x address it finds
# or pin it: ./deploy_mac/lib/install_time_server.sh 192.168.123.99
```

That renders `deploy_mac/lib/go2-time-server.service.in` into a `systemd --user`
unit, enables it and starts it as you — no root anywhere. It refuses to guess: if
no `192.168.123.x` address is present it stops rather than falling back to
`0.0.0.0`. The address is passed to `time_server.py` explicitly so the server
never answers on wifi or VPN interfaces.

For a single session the foreground form is enough:

```bash
python3 deploy_mac/lib/time_server.py 192.168.123.99 8123
```

A user unit stops when you log out; `loginctl enable-linger "$USER"` keeps it up
(enabled on the current host). Undo everything with
`ACTION=uninstall ./deploy_mac/lib/install_time_server.sh`.

## Jetson side (one-time)

```bash
mkdir -p /tmp/g2cs
scp deploy_mac/jetson/go2_sync_clock.py deploy_mac/jetson/go2-clock-sync.service \
    deploy_mac/jetson/go2-clock-sync.timer deploy_mac/jetson/install_clock_sync.sh \
    unitree@192.168.123.18:/tmp/g2cs/
ssh unitree@192.168.123.18 \
  'echo 123 | sudo -S bash /tmp/g2cs/install_clock_sync.sh 192.168.123.99:8123'
```

That installs `/usr/local/sbin/go2_sync_clock.py` (root:root 0755), the unit and
timer in `/etc/systemd/system/`, and `GO2_TIME_SERVER=192.168.123.99:8123` in
`/etc/default/go2-clock-sync`. Re-running it is safe.

## Checking it

```bash
cat /run/go2-clock-sync/state
# 2026-10-29 04:49:17 ok offset=+0.002s rtt=0.3ms 192.168.123.99:8123

systemctl list-timers go2-clock-sync.timer
date -u
```

`state` is one of:

| value | meaning |
| --- | --- |
| `ok offset=…` | synced; offset below the 50 ms no-op threshold |
| `stepped …` | the clock was just moved; the same line is in the journal as a warning |
| `unreachable` | no host answering — expected when the robot is on and the laptop is off |
| `refused …` | the server offered an offset over 3 years; nothing was applied |

Only a real step and a refusal produce journal lines. A robot left powered on
overnight with no host writes nothing at all (`Result=success`, not a failed
unit) — the reason `LogLevelMax=notice` is set and the no-op paths exit 0.

## Design notes

- **High port, not UDP/123.** Pointing the stock `systemd-timesyncd` at the host
  would be tidier but needs root on the host to bind 123, and `systemd --user`
  cannot take `AmbientCapabilities`. A high port keeps every piece unprivileged
  on the host side.
- **No sudoers entry.** `go2-clock-sync.service` runs as root already; a
  `NOPASSWD` target writable by `unitree` would be a free escalation path. The
  script is root-owned for the same reason.
- **Lowest RTT of 5 samples.** The server stamps the reply as it sends, so the
  remaining error is the one-way delay; high RTT means queuing, and queuing is
  exactly what corrupts that estimate. Measured 0.3 ms RTT on an idle link.
- **50 ms no-op threshold.** Keeps the 30 s tick from jittering the clock
  against itself; NTP-style slewing is not worth the complexity here.
- **`systemd-timesyncd` is left enabled.** It is inert without a route, and if
  the robot ever gets internet it becomes the better authority. Both target real
  UTC, so they agree.

## Caveats

- The first sync after a boot is a **discrete step**, not a slew. If the host
  only appears after the stack is already running, expect the clock to jump
  mid-session; the journal says how far.
- The RTC is dead, so **this recurs on every power cycle** — the timer is what
  makes that invisible, not a one-time correction.
- If the deployment host's address changes, update
  `/etc/default/go2-clock-sync` (comma-separated `host:port` list is accepted).
