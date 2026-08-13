#!/usr/bin/env python3
"""READ-ONLY ROS 2 sanity check: subscribe to /lowstate from the real Go2 and
print decoded fields. Bounded (exits after one sample or timeout). Publishes
NOTHING — never touches /lowcmd, never commands motors.

Run via deploy_mac/env.sh with the robot connected:
  python deploy_mac/diag/ros_lowstate_check.py
"""
import sys, time
import rclpy
from rclpy.node import Node
from unitree_go.msg import LowState


class Listener(Node):
    def __init__(self):
        super().__init__("ros_lowstate_check")
        self.msg = None
        # NOTE: subscriber only. No publisher is created on purpose.
        self.create_subscription(LowState, "lowstate", self._cb, 10)

    def _cb(self, m):
        self.msg = m


def main():
    timeout = float(sys.argv[1]) if len(sys.argv) > 1 else 8.0
    rclpy.init()
    n = Listener()
    deadline = time.time() + timeout
    while time.time() < deadline and n.msg is None:
        rclpy.spin_once(n, timeout_sec=0.2)
    m = n.msg
    if m is None:
        print(f"FAIL: no /lowstate received in {timeout:.0f}s. "
              f"Check GO2_IFACE / robot link / domain.")
        n.destroy_node(); rclpy.shutdown(); sys.exit(1)

    imu = m.imu_state
    q = [round(m.motor_state[i].q, 3) for i in range(12)]
    print("PASS: live /lowstate received via ROS 2 (rmw_cyclonedds)")
    print(f"  battery : {m.power_v:.2f} V  {m.power_a:.2f} A   soc={m.bms_state.soc}%")
    print(f"  rpy     : roll={imu.rpy[0]:+.3f} pitch={imu.rpy[1]:+.3f} yaw={imu.rpy[2]:+.3f}")
    print(f"  gyro    : {imu.gyroscope[0]:+.3f} {imu.gyroscope[1]:+.3f} {imu.gyroscope[2]:+.3f}")
    print(f"  accel   : {imu.accelerometer[0]:+.2f} {imu.accelerometer[1]:+.2f} {imu.accelerometer[2]:+.2f}")
    print(f"  joints  : {q}")
    print(f"  foot    : {list(m.foot_force)}")
    n.destroy_node(); rclpy.shutdown(); sys.exit(0)


if __name__ == "__main__":
    main()
