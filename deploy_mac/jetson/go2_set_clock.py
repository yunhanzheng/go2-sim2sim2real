#!/usr/bin/env python3
"""Set the Jetson CLOCK_REALTIME from the Mac's timestamp server.

Installed ROOT-OWNED at /usr/local/sbin/go2_set_clock.py and allowed
passwordless via /etc/sudoers.d/go2-clock-sync, so sync_jetson_clock.sh can run
non-interactively (background deploys have no terminal for a sudo prompt).
Root ownership matters: a unitree-writable NOPASSWD target would be a free
root escalation.

Usage: sudo python3 /usr/local/sbin/go2_set_clock.py <mac_ip> <port>
"""
import socket
import sys
import time

host, port = sys.argv[1], int(sys.argv[2])
s = socket.create_connection((host, port), 5)
epoch = float(s.recv(64).decode().strip())
time.clock_settime(time.CLOCK_REALTIME, epoch)
