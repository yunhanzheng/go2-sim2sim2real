#!/usr/bin/env python3
"""Serve this host's clock to the robot's Jetson over TCP.

Why this exists: the Orin's RTC reads 1970 and the robot's network has no
default route, so `systemd-timesyncd` has nothing to sync against and the clock
restarts at whatever time the last shutdown happened (observed 2026-10-29:
62 minutes behind after a power cycle, 72 days behind before the first manual
fix). The only clock authority reachable from `192.168.123.18` is this machine
on `192.168.123.0/24`, so the Jetson pulls from here instead.

Deliberately unprivileged: it binds a high port, needs no root, keeps no state,
and only ever reveals what `date` would tell you. The Jetson side is
`deploy_mac/jetson/go2_sync_clock.py` (installed as
/usr/local/sbin/go2_sync_clock.py).

Usage: python3 time_server.py [bind_addr] [port]
       (default 0.0.0.0:8123; use 192.168.123.99 to stay off other interfaces)
"""
import socket
import sys
import threading
import time

DEFAULT_PORT = 8123


def serve_client(conn: socket.socket) -> None:
    # Stamp BEFORE the send: the number the client receives should describe the
    # instant the reply left, not the instant after a scheduling delay. Queue
    # time inside the host then shows up as RTT and gets halved by the client
    # instead of silently inflating the offset by the whole delay.
    epoch = time.time()
    try:
        conn.sendall(f"{epoch:.9f}\n".encode())
    except OSError:
        pass
    finally:
        conn.close()


def main() -> int:
    bind_addr = sys.argv[1] if len(sys.argv) > 1 else "0.0.0.0"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_PORT

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((bind_addr, port))
    srv.listen(8)
    print(f"time_server: {bind_addr}:{port}/tcp, serving this host's clock",
          flush=True)

    while True:
        try:
            conn, peer = srv.accept()
        except OSError:
            continue
        # A thread per connection: the Jetson retries every 30 s and a manual
        # probe should not have to wait behind it.
        threading.Thread(target=serve_client, args=(conn,), daemon=True).start()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        pass
