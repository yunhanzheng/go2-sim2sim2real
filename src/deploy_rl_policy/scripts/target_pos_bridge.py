#!/usr/bin/env python3
"""Headless replacement for low_level_ctrl in dry-run mode.

Subscribes /rl/target_pos (Float32MultiArray, length 12 in Unitree order) and
forwards each entry to a unitree_go::msg::LowCmd on /mujoco/lowcmd with the
configured policy_kp / policy_kd. Skips the joystick state machine entirely so
the recorder pipeline can run headless without /joy. Not safe on real hardware
— for that path keep using low_level_ctrl.
"""
import argparse

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32MultiArray
from unitree_go.msg import LowCmd


class TargetPosBridge(Node):
    def __init__(self, kp: float, kd: float, output_topic: str):
        super().__init__("target_pos_bridge")
        self.kp = kp
        self.kd = kd
        self.cmd = LowCmd()
        for i in range(20):
            self.cmd.motor_cmd[i].mode = 0x01
            self.cmd.motor_cmd[i].q = 0.0
            self.cmd.motor_cmd[i].kp = 0.0
            self.cmd.motor_cmd[i].dq = 0.0
            self.cmd.motor_cmd[i].kd = 0.0
            self.cmd.motor_cmd[i].tau = 0.0

        self.pub = self.create_publisher(LowCmd, output_topic, 10)
        self.sub = self.create_subscription(
            Float32MultiArray, "/rl/target_pos", self._on_target, 10
        )
        self.engaged_pub = self.create_publisher(Bool, "/policy_engaged", 10)
        self.engaged_timer = self.create_timer(0.1, self._publish_engaged)
        self.get_logger().info(
            f"target_pos_bridge: /rl/target_pos → {output_topic} (kp={kp}, kd={kd})"
        )

    def _publish_engaged(self):
        msg = Bool()
        msg.data = True
        self.engaged_pub.publish(msg)

    def _on_target(self, msg: Float32MultiArray):
        if len(msg.data) < 12:
            return
        for i in range(12):
            self.cmd.motor_cmd[i].q = float(msg.data[i])
            self.cmd.motor_cmd[i].kp = self.kp
            self.cmd.motor_cmd[i].kd = self.kd
        self.pub.publish(self.cmd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kp", type=float, default=30.0)
    ap.add_argument("--kd", type=float, default=0.6)
    ap.add_argument("--output_topic", type=str, default="/mujoco/lowcmd")
    args, _ = ap.parse_known_args()

    rclpy.init()
    node = TargetPosBridge(args.kp, args.kd, args.output_topic)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
