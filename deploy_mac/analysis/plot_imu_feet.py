#!/usr/bin/env python3
"""Live plot of Go2 IMU + foot-force from ROS 2 /lowstate (READ-ONLY).

Tilt/rotate the robot to verify orientation (roll/pitch/yaw) + gyro; press each
foot to verify the 4 foot-force sensors. These feed the policy proprio vector:
  roll,pitch -> prop[3:5],  body ang-vel (gyro) -> prop[0:3],  foot contact -> prop[49:53].

  source deploy_mac/env.sh
  python deploy_mac/analysis/plot_imu_feet.py            # live GUI
  python deploy_mac/analysis/plot_imu_feet.py --selftest # headless 3s read check (no GUI)

Subscribe-only: never publishes, never touches /lowcmd, never commands motors.
"""
import argparse
import sys
import time
from collections import deque

import rclpy
from rclpy.node import Node
from unitree_go.msg import LowState

KEYS = ["roll", "pitch", "yaw", "gx", "gy", "gz", "ax", "ay", "az", "FR", "FL", "RR", "RL"]


class Sub(Node):
    def __init__(self):
        super().__init__("plot_imu_feet")
        self.rpy = [0.0, 0.0, 0.0]
        self.gyro = [0.0, 0.0, 0.0]
        self.acc = [0.0, 0.0, 0.0]
        self.foot = [0.0, 0.0, 0.0, 0.0]
        self.have = False
        self.create_subscription(LowState, "lowstate", self._cb, 10)  # subscribe only

    def _cb(self, m):
        s = m.imu_state
        self.rpy = [float(x) for x in s.rpy]
        self.gyro = [float(x) for x in s.gyroscope]
        self.acc = [float(x) for x in s.accelerometer]
        self.foot = [float(x) for x in m.foot_force]   # order: FR, FL, RR, RL
        self.have = True

    def vals(self):
        return self.rpy + self.gyro + self.acc + self.foot


def selftest(node, secs=3.0):
    lo = {k: 9e9 for k in KEYS}
    hi = {k: -9e9 for k in KEYS}
    n = 0
    end = time.time() + secs
    while time.time() < end:
        rclpy.spin_once(node, timeout_sec=0.1)
        if node.have:
            n += 1
            for k, v in zip(KEYS, node.vals()):
                lo[k] = min(lo[k], v)
                hi[k] = max(hi[k], v)
    if n == 0:
        print("FAIL: no /lowstate samples (check robot link / GO2_IFACE)")
        return 1
    print(f"PASS: {n} /lowstate samples in {secs:.0f}s")
    print(f"  rpy  : roll[{lo['roll']:+.3f},{hi['roll']:+.3f}] "
          f"pitch[{lo['pitch']:+.3f},{hi['pitch']:+.3f}] yaw[{lo['yaw']:+.3f},{hi['yaw']:+.3f}]")
    print(f"  gyro : x[{lo['gx']:+.3f},{hi['gx']:+.3f}] y[{lo['gy']:+.3f},{hi['gy']:+.3f}] z[{lo['gz']:+.3f},{hi['gz']:+.3f}]")
    print(f"  accel: x[{lo['ax']:+.2f},{hi['ax']:+.2f}] y[{lo['ay']:+.2f},{hi['ay']:+.2f}] z[{lo['az']:+.2f},{hi['az']:+.2f}]")
    print(f"  foot : FR[{lo['FR']:.0f},{hi['FR']:.0f}] FL[{lo['FL']:.0f},{hi['FL']:.0f}] "
          f"RR[{lo['RR']:.0f},{hi['RR']:.0f}] RL[{lo['RL']:.0f},{hi['RL']:.0f}]")
    return 0


def live(node, window):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    t0 = time.time()
    bt = deque()
    series = {k: deque() for k in KEYS}
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    a_or, a_gy, a_ac, a_ft = axes.flatten()
    a_or.set_title("Orientation [rad]"); a_or.set_ylim(-3.3, 3.3)
    a_gy.set_title("Gyro [rad/s]"); a_gy.set_ylim(-4, 4)
    a_ac.set_title("Accel [m/s^2]"); a_ac.set_ylim(-12, 12)
    a_ft.set_title("Foot force [raw]"); a_ft.set_ylim(0, 60)
    for ax in (a_or, a_gy, a_ac, a_ft):
        ax.grid(True, alpha=0.3)
    spec = [("roll", "roll", a_or), ("pitch", "pitch", a_or), ("yaw", "yaw", a_or),
            ("gx", "x", a_gy), ("gy", "y", a_gy), ("gz", "z", a_gy),
            ("ax", "x", a_ac), ("ay", "y", a_ac), ("az", "z", a_ac),
            ("FR", "FR", a_ft), ("FL", "FL", a_ft), ("RR", "RR", a_ft), ("RL", "RL", a_ft)]
    L = {}
    for k, lab, ax in spec:
        (ln,) = ax.plot([], [], label=lab, lw=1.5)
        L[k] = ln
    for ax in (a_or, a_gy, a_ac, a_ft):
        ax.legend(loc="upper right", fontsize=8, ncol=2)
    fig.suptitle("Go2 IMU + foot force  (/lowstate, READ-ONLY)  — tilt robot / press feet to verify")
    fig.tight_layout()

    def update(_):
        rclpy.spin_once(node, timeout_sec=0.0)
        now = time.time() - t0
        bt.append(now)
        for k, v in zip(KEYS, node.vals()):
            series[k].append(v)
        while bt and bt[0] < now - window:
            bt.popleft()
            for k in series:
                series[k].popleft()
        ts = list(bt)
        for k, ln in L.items():
            ln.set_data(ts, list(series[k]))
        for ax in (a_or, a_gy, a_ac, a_ft):
            ax.set_xlim(max(0, now - window), now + 0.2)
        fmax = max([max(series[k]) for k in ("FR", "FL", "RR", "RL") if series[k]] or [1])
        a_ft.set_ylim(0, max(60, fmax * 1.15))
        return list(L.values())

    _ani = FuncAnimation(fig, update, interval=50, blit=False, cache_frame_data=False)
    plt.show()
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=float, default=10.0, help="time window (s)")
    ap.add_argument("--selftest", action="store_true", help="headless read check, no GUI")
    args = ap.parse_args()
    rclpy.init()
    node = Sub()
    try:
        rc = selftest(node) if args.selftest else live(node, args.window)
    finally:
        node.destroy_node()
        rclpy.shutdown()
    sys.exit(rc)


if __name__ == "__main__":
    main()
