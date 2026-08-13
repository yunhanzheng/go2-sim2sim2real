#!/usr/bin/env python3
"""Bridge the Go2's own remote (/wirelesscontroller) to /joy, so the UNMODIFIED C++
low_level_ctrl (and rl_distill_policy's xbox_command) can be driven by the Go2 remote
instead of a separate gamepad + joy_node.

/joy layout matches the Xbox mapping the C++/xbox_command expect:
  buttons: [0]=A [1]=B [2]=X [3]=Y [4]=L1 [5]=R1
  axes:    [0]=lx [1]=ly [2]=L2(-1 pressed / +1 rest) [3]=rx [4]=ry [5]=R2(-1/ +1)

Resulting FSM (handled by the C++ low_level_ctrl):
  B           lay -> stand           (buttons[1])
  A           stand -> lay           (buttons[0])
  L1 + R1     stand -> engage policy (buttons[4] & [5])
  L2 + R2     quit/e-stop            (axes[2]==-1 && axes[5]==-1)

READ-ONLY w.r.t. the robot: only republishes controller input.
"""
import rclpy
from rclpy.node import Node
from unitree_go.msg import WirelessController
from sensor_msgs.msg import Joy

K_R1, K_L1, K_R2, K_L2 = 0x0001, 0x0002, 0x0010, 0x0020
K_A, K_B, K_X, K_Y = 0x0100, 0x0200, 0x0400, 0x0800


class Bridge(Node):
    def __init__(self):
        super().__init__("wireless_to_joy")
        self.pub = self.create_publisher(Joy, "/joy", 10)
        self.create_subscription(WirelessController, "/wirelesscontroller", self._cb, 10)
        # Diagnosability (2026-07-27 real run: remote "stopped working", but the
        # only log line was the banner — cause unrecoverable). Log every key
        # transition, plus a 10 s heartbeat with the message count so a dead
        # /wirelesscontroller stream vs a dead bridge is distinguishable.
        self._msg_count = 0
        self._last_keys = None
        self.create_timer(10.0, self._heartbeat)

    def _heartbeat(self):
        self.get_logger().info(f"heartbeat: {self._msg_count} /wirelesscontroller msgs total "
                               f"(0 growth across heartbeats = remote stream dead)")

    def _cb(self, m):
        k = int(m.keys)
        self._msg_count += 1
        if k != self._last_keys:
            self.get_logger().info(f"keys 0x{(self._last_keys or 0):04x} -> 0x{k:04x} "
                                   f"(ly={m.ly:+.2f})")
            self._last_keys = k
        j = Joy()
        j.header.stamp = self.get_clock().now().to_msg()
        l2 = -1.0 if (k & K_L2) else 1.0
        r2 = -1.0 if (k & K_R2) else 1.0
        j.axes = [float(m.lx), float(m.ly), l2, float(m.rx), float(m.ry), r2]
        j.buttons = [
            1 if (k & K_A) else 0, 1 if (k & K_B) else 0, 1 if (k & K_X) else 0,
            1 if (k & K_Y) else 0, 1 if (k & K_L1) else 0, 1 if (k & K_R1) else 0,
        ]
        self.pub.publish(j)


def main():
    rclpy.init()
    n = Bridge()
    n.get_logger().info("wireless_to_joy: /wirelesscontroller -> /joy "
                        "(B=stand, A=lay, L1+R1=engage, L2+R2=quit)")
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        n.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
