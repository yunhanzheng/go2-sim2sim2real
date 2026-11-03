#!/usr/bin/env python3
"""Step this Jetson's clock from the deployment host.

Installed ROOT-OWNED at /usr/local/sbin/go2_sync_clock.py and run as root by
go2-clock-sync.service (there is no sudoers NOPASSWD entry: an
unitree-writable NOPASSWD target would be a free root escalation, and systemd
already gives us root without one).

Why a custom client instead of stock systemd-timesyncd: the Jetson's RTC reads
1970 and there is no default route, so timesyncd has no server to talk to. The
one reachable authority is the host on 192.168.123.0/24, which serves its clock
on a high port (deploy_mac/lib/time_server.py) because binding UDP/123 would
need root there.

Behaviour:
  * samples the server a few times and keeps the LOWEST-RTT sample — highest RTT
    means queuing, and queuing is exactly what corrupts a one-way estimate;
  * steps CLOCK_REALTIME only when the offset exceeds NOOP_THRESHOLD_S, so the
    steady-state timer does not jitter the clock against itself;
  * exits 0 and prints NOTHING on the two no-op paths (host unreachable, offset
    already small). A non-zero exit makes systemd log a unit failure — three
    lines every 30 s forever on a robot left powered on without a host — for a
    condition that is entirely expected. The outcome lands in
    /run/go2-clock-sync/state, which answers "is the clock synced?";
  * refuses an implausible offset loudly rather than trusting a bogus server.

Usage: sudo /usr/local/sbin/go2_sync_clock.py [host:port[,host:port...]]
Config: /etc/default/go2-clock-sync  ->  GO2_TIME_SERVER=192.168.123.99:8123
"""
import os
import socket
import sys
import time

DEFAULT_SERVERS = "192.168.123.99:8123"
CONFIG_PATH = "/etc/default/go2-clock-sync"
STATE_DIR = "/run/go2-clock-sync"
STATE_PATH = STATE_DIR + "/state"

NOOP_THRESHOLD_S = 0.050        # below this the clock is good enough
MAX_PLAUSIBLE_OFFSET_S = 3 * 365 * 24 * 3600
SAMPLES = 5
TIMEOUT_S = 2.0

# go2-clock-sync.service sets LogLevelMax=notice so a robot left powered on
# without the host does not write two systemd lines every 30 s forever. That
# also drops plain print() output, which is logged at info — so the lines an
# operator actually needs carry an explicit syslog priority (systemd strips the
# <4> prefix and files the line as a warning).
PRIORITY_WARNING = "<4>"


def report(msg: str) -> None:
    print(f"{PRIORITY_WARNING}{msg}", flush=True)


def write_state(text: str) -> None:
    """Record the outcome without touching the journal.

    The timer fires every 30 s whether or not a host is there, so the no-op
    outcomes must stay silent or the journal becomes useless on a robot left
    powered on overnight. This file keeps them observable instead.
    """
    try:
        os.makedirs(STATE_DIR, exist_ok=True)
        stamp = time.strftime("%F %T", time.gmtime())
        with open(STATE_PATH, "w") as fp:
            fp.write(f"{stamp} {text}\n")
    except OSError:
        pass


def parse_servers() -> list[tuple[str, int]]:
    raw = os.environ.get("GO2_TIME_SERVER", "")
    if not raw:
        try:
            with open(CONFIG_PATH) as fp:
                for line in fp:
                    line = line.split("#", 1)[0].strip()
                    if line.startswith("GO2_TIME_SERVER="):
                        raw = line.split("=", 1)[1].strip().strip("\"'")
                        break
        except OSError:
            pass
    if not raw:
        raw = DEFAULT_SERVERS
    out = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        host, _, port = item.partition(":")
        out.append((host, int(port) if port else 8123))
    return out


def sample(host: str, port: int) -> tuple[float, float]:
    """Return (offset_to_apply, rtt) for one request.

    The server stamps the reply as it sends it, so its clock at the instant we
    received the reply was `server_epoch + one_way_delay`. Assume a symmetric
    path and use half the bracketed RTT for that delay — the standard NTP
    assumption.
    """
    t0 = time.clock_gettime(time.CLOCK_REALTIME)
    with socket.create_connection((host, port), TIMEOUT_S) as sock:
        sock.settimeout(TIMEOUT_S)
        buf = b""
        while not buf.endswith(b"\n"):
            chunk = sock.recv(64)
            if not chunk:
                break
            buf += chunk
    t1 = time.clock_gettime(time.CLOCK_REALTIME)

    server_at_t1 = float(buf.decode().strip()) + (t1 - t0) / 2.0
    return server_at_t1 - t1, t1 - t0


def main() -> int:
    best = None                       # (rtt, offset, host, port)
    for host, port in parse_servers():
        rtts = []
        for _ in range(SAMPLES):
            try:
                offset, rtt = sample(host, port)
            except (OSError, ValueError):
                break
            rtts.append((rtt, offset))
        if len(rtts) < SAMPLES:
            continue
        rtt, offset = min(rtts)
        if best is None or rtt < best[0]:
            best = (rtt, offset, host, port)

    if best is None:
        # Host absent is an expected state, not an error. Exiting non-zero here
        # would make systemd mark the unit failed and log it on every tick.
        write_state("unreachable")
        return 0

    rtt, offset, host, port = best
    if abs(offset) > MAX_PLAUSIBLE_OFFSET_S:
        report(f"go2_sync_clock: refusing implausible offset {offset:+.0f} s "
               f"from {host}:{port}")
        write_state(f"refused {offset:+.0f}s from {host}:{port}")
        return 2
    if abs(offset) <= NOOP_THRESHOLD_S:
        write_state(f"ok offset={offset:+.3f}s rtt={rtt * 1e3:.1f}ms {host}:{port}")
        return 0

    before = time.clock_gettime(time.CLOCK_REALTIME)
    time.clock_settime(time.CLOCK_REALTIME, before + offset)
    # Loud only when it actually moved the clock: this is the line that tells an
    # operator a mid-session jump just happened.
    report(f"go2_sync_clock: stepped {offset:+.3f} s from {host}:{port} "
           f"(rtt {rtt * 1e3:.1f} ms) -> {time.strftime('%F %T', time.gmtime())} UTC")
    write_state(f"stepped {offset:+.3f}s rtt={rtt * 1e3:.1f}ms {host}:{port}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
