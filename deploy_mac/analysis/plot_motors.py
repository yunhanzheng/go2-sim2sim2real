#!/usr/bin/env python3
"""Live plot of all 12 Go2 leg-motor positions from ROS 2 /lowstate (READ-ONLY).

Move the legs by hand and watch the lines to confirm each joint reads correctly.
Four subplots (one per leg), each with hip/thigh/calf.

  source deploy_mac/env.sh
  python deploy_mac/analysis/plot_motors.py              # live GUI plot
  python deploy_mac/analysis/plot_motors.py --window 15  # longer time window
  python deploy_mac/analysis/plot_motors.py --selftest   # headless 3s read check (no GUI)

Subscribe-only: it never publishes, never touches /lowcmd, never commands motors.
"""
import argparse
import sys
import time
from collections import deque

import rclpy
from rclpy.node import Node
from unitree_go.msg import LowState

NAMES = ["FR_hip", "FR_thigh", "FR_calf", "FL_hip", "FL_thigh", "FL_calf",
         "RR_hip", "RR_thigh", "RR_calf", "RL_hip", "RL_thigh", "RL_calf"]
LEGS = [("FR", [0, 1, 2]), ("FL", [3, 4, 5]), ("RR", [6, 7, 8]), ("RL", [9, 10, 11])]
JOINT = ["hip", "thigh", "calf"]


class MotorSub(Node):
    def __init__(self):
        super().__init__("plot_motors")
        self.q = [0.0] * 12
        self.have = False
        self.create_subscription(LowState, "lowstate", self._cb, 10)  # subscribe only

    def _cb(self, m):
        for i in range(12):
            self.q[i] = float(m.motor_state[i].q)
        self.have = True


def selftest(node, secs=3.0):
    lo = [9e9] * 12
    hi = [-9e9] * 12
    n = 0
    end = time.time() + secs
    while time.time() < end:
        rclpy.spin_once(node, timeout_sec=0.1)
        if node.have:
            n += 1
            for i in range(12):
                lo[i] = min(lo[i], node.q[i])
                hi[i] = max(hi[i], node.q[i])
    if n == 0:
        print("FAIL: no /lowstate samples (check robot link / GO2_IFACE)")
        return 1
    print(f"PASS: {n} /lowstate samples in {secs:.0f}s")
    for i, nm in enumerate(NAMES):
        print(f"  {nm:9s} now={node.q[i]:+.3f}  range[{lo[i]:+.3f},{hi[i]:+.3f}]")
    return 0


def live(node, window):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    t0 = time.time()
    bt = deque()
    bq = [deque() for _ in range(12)]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes = axes.flatten()
    lines = {}
    for ax, (leg, idxs) in zip(axes, LEGS):
        ax.set_title(leg)
        ax.set_ylabel("q [rad]")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-3.3, 1.9)
        for k, idx in enumerate(idxs):
            (ln,) = ax.plot([], [], label=JOINT[k], lw=1.5)
            lines[idx] = ln
        ax.legend(loc="upper right", fontsize=8)
    fig.suptitle("Go2 live motor positions  (/lowstate, READ-ONLY)  — move legs by hand to verify")
    fig.tight_layout()

    def update(_):
        rclpy.spin_once(node, timeout_sec=0.0)
        now = time.time() - t0
        bt.append(now)
        for i in range(12):
            bq[i].append(node.q[i])
        while bt and bt[0] < now - window:
            bt.popleft()
            for i in range(12):
                bq[i].popleft()
        ts = list(bt)
        for idx, ln in lines.items():
            ln.set_data(ts, list(bq[idx]))
        for ax in axes:
            ax.set_xlim(max(0, now - window), now + 0.2)
        return list(lines.values())

    _ani = FuncAnimation(fig, update, interval=50, blit=False, cache_frame_data=False)
    plt.show()
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=float, default=10.0, help="time window (s)")
    ap.add_argument("--selftest", action="store_true", help="headless read check, no GUI")
    args = ap.parse_args()
    rclpy.init()
    node = MotorSub()
    try:
        rc = selftest(node) if args.selftest else live(node, args.window)
    finally:
        node.destroy_node()
        rclpy.shutdown()
    sys.exit(rc)


if __name__ == "__main__":
    main()
