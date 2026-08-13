#!/usr/bin/env python3
"""READ-ONLY: subscribe to /rl/target_pos (the policy's output) and print the 12
target joint angles next to the standing default. Used to verify the inference
dry-run. Never publishes, never commands motors.

  python deploy_mac/analysis/read_target_pos.py [timeout_s]
"""
import sys
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

NAMES = ["FR_hip", "FR_thigh", "FR_calf", "FL_hip", "FL_thigh", "FL_calf",
         "RR_hip", "RR_thigh", "RR_calf", "RL_hip", "RL_thigh", "RL_calf"]
# standing default (DEFAULT_DOF_POLICY, SDK/MJ order) from distill_common
DEF = [-0.1, 0.8, -1.5, 0.1, 0.8, -1.5, -0.1, 1.0, -1.5, 0.1, 1.0, -1.5]


class R(Node):
    def __init__(self):
        super().__init__("read_target_pos")
        self.d = None
        self.n = 0
        self.create_subscription(Float32MultiArray, "rl/target_pos", self._cb, 10)

    def _cb(self, m):
        self.d = list(m.data)
        self.n += 1


def main():
    to = float(sys.argv[1]) if len(sys.argv) > 1 else 15.0
    rclpy.init()
    node = R()
    deadline = time.time() + to
    while time.time() < deadline and node.d is None:
        rclpy.spin_once(node, timeout_sec=0.2)
    if node.d is None:
        print(f"FAIL: no /rl/target_pos in {to:.0f}s (policy not running or still gated)")
        node.destroy_node(); rclpy.shutdown(); sys.exit(1)
    # collect a few more to confirm it's streaming
    t2 = time.time() + 1.0
    while time.time() < t2:
        rclpy.spin_once(node, timeout_sec=0.1)
    print(f"PASS: /rl/target_pos streaming ({node.n} msgs). Target joint angles vs standing default:")
    maxdev = 0.0
    for i, nm in enumerate(NAMES):
        d = node.d[i] if i < len(node.d) else float("nan")
        dev = d - DEF[i]
        maxdev = max(maxdev, abs(dev))
        print(f"  {nm:9s} target={d:+.4f}  default={DEF[i]:+.3f}  delta={dev:+.4f}")
    print(f"max |delta from standing default| = {maxdev:.3f} rad")
    node.destroy_node(); rclpy.shutdown(); sys.exit(0)


if __name__ == "__main__":
    main()
