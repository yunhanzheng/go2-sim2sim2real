#!/usr/bin/env python3
"""Publishes /joy continuously to drive the FSM without a physical pad.

Schedule (relative to node start):
  t < 1s        : all zeros (idle)
  1.0 < t < 2.0 : B held         → low_level_ctrl: stand up
  2.0 < t < 8.0 : all zeros      (let robot reach standing pose)
  8.0 < t < 9.0 : LB+RB held     → activate RL policy
  t > 9.0       : all zeros      (stay engaged)

Continuous publishing at 20 Hz means single-message drops don't lose the
command. Button "windows" (held for ~1s) are likewise robust — the FSM only
needs to see one message inside the window to flip state.
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
import time


# Button indices (Xbox layout, used by low_level_ctrl's joy_callback)
BTN_A, BTN_B, BTN_X, BTN_Y, BTN_LB, BTN_RB = 0, 1, 2, 3, 4, 5
NUM_BUTTONS = 11


class FakeJoy(Node):
    def __init__(self):
        super().__init__("fake_joy")
        self.pub = self.create_publisher(Joy, "/joy", 10)
        self._start = time.time()
        self._phase = None
        self.timer = self.create_timer(0.05, self._tick)  # 20 Hz

    def _make_msg(self, *pressed_btns):
        msg = Joy()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.axes = [0.0] * 8
        buttons = [0] * NUM_BUTTONS
        for b in pressed_btns:
            buttons[b] = 1
        msg.buttons = buttons
        return msg

    def _tick(self):
        elapsed = time.time() - self._start

        # Hold standing for 30 s, then engage the policy via LB+RB. The 30 s
        # gives time to verify the standing hold visually before the policy
        # takes over. LB+RB is held for 1 s to guarantee at least one message
        # reaches low_level_ctrl.
        if 30.0 <= elapsed < 31.0:
            new_phase = "LB+RB"
            msg = self._make_msg(BTN_LB, BTN_RB)
        else:
            new_phase = "idle"
            msg = self._make_msg()

        if new_phase != self._phase:
            self.get_logger().info(f"phase: {new_phase} (t={elapsed:.1f}s)")
            self._phase = new_phase

        self.pub.publish(msg)


def main():
    rclpy.init()
    node = FakeJoy()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
