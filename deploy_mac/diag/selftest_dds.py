#!/usr/bin/env python3
"""Robot-independent sanity check: round-trip a unitree_go/msg/LowState through
DDS in-process. Proves rclpy + CycloneDDS + the built message typesupport all
work on this machine, without needing the robot.

Run via deploy_mac/env.sh, but override the interface to loopback first, e.g.:
  CYCLONEDDS_URI='<CycloneDDS><Domain><General><Interfaces><NetworkInterface name="lo0"/></Interfaces></General></Domain></CycloneDDS>' \
  python deploy_mac/diag/selftest_dds.py
"""
import sys, time
import rclpy
from rclpy.node import Node
from unitree_go.msg import LowState  # noqa: F401  (LowCmd import = build check)


class Loopback(Node):
    def __init__(self):
        super().__init__("dds_selftest")
        self.got = None
        self.create_subscription(LowState, "dds_selftest", self._cb, 10)
        self.pub = self.create_publisher(LowState, "dds_selftest", 10)

    def _cb(self, msg):
        self.got = msg


def main():
    rclpy.init()
    n = Loopback()
    msg = LowState()
    msg.bandwidth = 12345
    msg.imu_state.rpy = [0.10, 0.20, 0.30]
    deadline = time.time() + 5.0
    while time.time() < deadline and n.got is None:
        n.pub.publish(msg)
        rclpy.spin_once(n, timeout_sec=0.1)
    ok = (n.got is not None
          and n.got.bandwidth == 12345
          and abs(float(n.got.imu_state.rpy[2]) - 0.30) < 1e-5)
    if ok:
        print(f"PASS: round-tripped unitree_go/LowState "
              f"(bandwidth={n.got.bandwidth}, rpy[2]={float(n.got.imu_state.rpy[2]):.3f})")
    else:
        print(f"FAIL: no/incorrect message received (got={n.got!r})")
    n.destroy_node()
    rclpy.shutdown()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
