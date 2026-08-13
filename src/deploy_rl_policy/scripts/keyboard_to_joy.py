#!/usr/bin/env python3
"""Keyboard -> sensor_msgs/Joy bridge: a software stand-in for the Xbox/8BitDo
remote so you can drive the robot in MuJoCo from the keyboard, using the exact
same deploy ROS stack (FSM + depth encoder + rl_distill_policy) — no gamepad.

Emulates the xpad axis/button layout the deploy stack expects (see
xbox_command.py + low_level_ctrl FSM). The keyboard keys are named after the
gamepad controls they stand in for, so A/B are the FSM face buttons (not WASD):

  BUTTONS      buttons[0] = A  Stand -> Lay ................. A
               buttons[1] = B  Lay -> Stand ................. B
               buttons[4]+[5] = LB+RB  engage policy ........ Space (hold)
  LEFT STICK   axes[1] = forward / back  (cmd_vel_x) ........ W / S
               axes[0] = strafe L / R    (cmd_vel_y) ........ ← / →
  RIGHT STICK  axes[3] = yaw L / R       (cmd_yaw)   ........ Q / E
  TRIGGERS     axes[2]=LT, axes[5]=RT    (idle +1, full -1) . Z + X (hold) = quit
  SPEED PRESET stick magnitude (scales top speed) .......... 1-9 = 0.1-0.9, 0 = 1.0

Pressing a digit latches the stick magnitude live, so you can change throttle
while driving (cmd_vx = lin_vel_max * magnitude). --speed sets the initial value.

Typical drive loop in sim:
  B  (lay -> stand)  ->  hold Space (LB+RB engage)  ->  W/S/←/→/Q/E to drive
Release Space to disengage (dead-man if momentary_engage), A to lay back down.

NOTE — which channels are *live* depends on the deployed policy:
  • rl_distill_policy reads forward only when LB+RB are held: cmd_vx = lin_vel_max
    * axes[1] (see _read_lin_vel_x). So W/S is the one channel that always drives.
  • The no-yaw-head student (L00 / T000) predicts yaw from the depth encoder and
    zeroes lateral in training, so A/D (axes[0]) and Q/E (axes[3]) are published
    faithfully but ignored by that policy. They're here so this node is a true
    full-remote emulator for any policy that *does* consume them.

Requires `pynput` (pip install pynput). On macOS grant the terminal app
Accessibility permission (System Settings -> Privacy & Security -> Accessibility)
or key events won't be captured. pynput captures key-release events so momentary
buttons (Space = LB+RB) behave correctly.

Run standalone against a live stack:
  ros2 run deploy_rl_policy keyboard_to_joy.py            # full-throttle sticks
  ros2 run deploy_rl_policy keyboard_to_joy.py --speed 0.5  # cap stick magnitude
or wired into the deploy-in-sim stack:
  ros2 launch deploy_rl_policy distill.launch.py is_simulation:=true use_keyboard:=true
  deploy_mac/run/keyboard_control.sh                          # Mac wrapper (env + paths)
"""
import argparse
import threading

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy

try:
    from pynput import keyboard
except ImportError as e:
    raise SystemExit("keyboard_to_joy needs pynput: pip install pynput") from e


AXIS_LY = 1   # left stick Y  -> forward cmd_vel_x (xbox_command.get_left_stick()[0])
AXIS_LX = 0   # left stick X  -> lateral cmd_vel_y (xbox_command.get_left_stick()[1])
AXIS_RX = 3   # right stick X -> yaw cmd          (xbox_command.get_right_stick())
AXIS_LT = 2   # idle +1, fully pressed -1
AXIS_RT = 5   # idle +1, fully pressed -1
BTN_A, BTN_B = 0, 1
BTN_LB, BTN_RB = 4, 5
NUM_AXES = 8
NUM_BUTTONS = 12
PUBLISH_HZ = 50.0


class KeyboardToJoy(Node):
    def __init__(self, args):
        super().__init__("keyboard_to_joy")
        self.pub = self.create_publisher(Joy, "/joy", 10)
        # Stick magnitude when a direction key is held. 1.0 = full deflection
        # (== a real stick pushed to the stop). cmd_vx = lin_vel_max * axes[1],
        # so --speed scales the top speed without touching lin_vel_max.
        self.mag = max(0.0, min(1.0, float(args.speed)))
        self._lock = threading.Lock()
        self._pressed = set()
        self._listener = keyboard.Listener(
            on_press=self._on_press, on_release=self._on_release
        )
        self._listener.daemon = True
        self._listener.start()
        self.create_timer(1.0 / PUBLISH_HZ, self._tick)
        self.get_logger().info(
            f"keyboard_to_joy ready (stick magnitude={self.mag:.2f}) -- "
            "A=lay  B=stand  Space(hold)=engage  |  "
            "W/S=fwd  ←/→=strafe  Q/E=yaw  |  1-9/0=speed preset  |  Z+X(hold)=quit"
        )

    def _key_id(self, key):
        if isinstance(key, keyboard.KeyCode) and key.char is not None:
            return key.char.lower()
        special = {
            keyboard.Key.space: "space",
            keyboard.Key.up: "up",
            keyboard.Key.down: "down",
            keyboard.Key.left: "left",
            keyboard.Key.right: "right",
        }
        return special.get(key)

    def _on_press(self, key):
        kid = self._key_id(key)
        if kid is None:
            return
        if kid.isdigit():
            # Speed preset: 1-9 -> 0.1-0.9, 0 -> 1.0 (full). Latches the stick
            # magnitude live so you can change throttle mid-drive. Not a held key.
            d = int(kid)
            self.mag = 1.0 if d == 0 else d / 10.0
            self.get_logger().info(f"stick magnitude -> {self.mag:.1f}")
            return
        with self._lock:
            self._pressed.add(kid)

    def _on_release(self, key):
        kid = self._key_id(key)
        if kid is not None:
            with self._lock:
                self._pressed.discard(kid)

    def _tick(self):
        with self._lock:
            keys = set(self._pressed)

        axes = [0.0] * NUM_AXES
        # Triggers idle at +1 in xpad convention; rl_distill_policy._quit_combo
        # treats axes[2]==axes[5]==-1 as the quit signal.
        axes[AXIS_LT] = 1.0
        axes[AXIS_RT] = 1.0
        buttons = [0] * NUM_BUTTONS

        # ── left stick: forward/back (W/S), strafe (← / →) ───────────────────
        if "w" in keys:
            axes[AXIS_LY] += self.mag
        if "s" in keys:
            axes[AXIS_LY] -= self.mag
        if "left" in keys:
            axes[AXIS_LX] += self.mag
        if "right" in keys:
            axes[AXIS_LX] -= self.mag
        # ── right stick: yaw (Q / E) ─────────────────────────────────────────
        if "q" in keys:
            axes[AXIS_RX] += self.mag
        if "e" in keys:
            axes[AXIS_RX] -= self.mag

        # ── FSM face buttons (named after the gamepad: A=lay, B=stand) ───────
        if "space" in keys:
            buttons[BTN_LB] = 1
            buttons[BTN_RB] = 1
        if "a" in keys:
            buttons[BTN_A] = 1
        if "b" in keys:
            buttons[BTN_B] = 1
        if "z" in keys and "x" in keys:
            axes[AXIS_LT] = -1.0
            axes[AXIS_RT] = -1.0

        msg = Joy()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.axes = axes
        msg.buttons = buttons
        self.pub.publish(msg)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--speed", type=float, default=1.0,
                   help="stick magnitude in [0,1] when a direction key is held "
                        "(scales top speed: cmd_vx = lin_vel_max * speed). "
                        "Default 1.0 = full deflection.")
    # ros2 launch / ros2 run inject '--ros-args ...'; ignore them.
    args, _ = p.parse_known_args()
    return args


def main():
    rclpy.init()
    node = KeyboardToJoy(parse_args())
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
