#!/usr/bin/env python3
"""Auto-pilot publisher to /joy: mimics the xbox controller's
lay->stand->engage->lay sequence so the distill stack can be exercised
without a real gamepad or keyboard hook.

Phases (timed from sequencer start, not from launch):
  [0, wait_init)                              : idle -- nodes wake up
  [wait_init, +press)                         : B held -- LayingDown -> StandingUp
  [+press, +settle)                           : idle -- wait for stand-up
  [+settle, +engage)                          : LB+RB held -- policy engaged;
                                                axes[1] = speed
  [+engage, +slowdown)                        : LB+RB held, axes[1] ramps
                                                linearly from speed -> 0
  [+slowdown, +laydown)                       : A held -- Standing -> LayDown
  [+laydown, end)                             : idle -- watch the lay state

Stop with Ctrl-C: sends a final all-zero Joy so the FSM disengages cleanly.

Run in a second terminal while distill.launch.py is up with use_keyboard:=false:
  ros2 run deploy_rl_policy joy_sequencer.py --speed 0.5
or directly:
  python3 .../scripts/joy_sequencer.py --speed 0.5 --duration 30
"""
import argparse
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from std_msgs.msg import Float32MultiArray


AXIS_LY = 1
AXIS_LT = 2
AXIS_RT = 5
BTN_A, BTN_B = 0, 1
BTN_LB, BTN_RB = 4, 5
NUM_AXES = 8
NUM_BUTTONS = 12
PUBLISH_HZ = 50.0


class JoySequencer(Node):
    def __init__(self, args):
        super().__init__("joy_sequencer")
        self.pub = self.create_publisher(Joy, "/joy", 10)
        self.args = args
        self.t0 = time.monotonic()
        self._last_phase = None
        self._return_armed = False
        self._return_stop_time = None
        if args.stop_on_return_x:
            self.create_subscription(Float32MultiArray, "/mujoco/pos", self._on_qpos, 10)
        # Optional velocity schedule during engage: list of
        # (duration_s, spd_start, spd_end) segments.  Two-part form "dur:spd"
        # is a constant step (spd_start == spd_end); three-part "dur:s0:s1"
        # is a linear ramp from s0 to s1 over dur seconds.  Empty => constant
        # args.speed (legacy behaviour).
        self._profile = []
        if getattr(args, "vel_profile", ""):
            for seg in args.vel_profile.split(","):
                parts = seg.split(":")
                if len(parts) == 2:
                    dur, s0 = float(parts[0]), float(parts[1])
                    self._profile.append((dur, s0, s0))
                elif len(parts) == 3:
                    dur, s0, s1 = float(parts[0]), float(parts[1]), float(parts[2])
                    self._profile.append((dur, s0, s1))
                else:
                    raise ValueError(f"vel_profile segment must be 'dur:spd' or 'dur:s0:s1', got {seg!r}")
        self.create_timer(1.0 / PUBLISH_HZ, self._tick)
        self.get_logger().info(
            f"joy_sequencer: wait_init={args.wait_init}s press={args.press}s "
            f"settle={args.settle}s speed={args.speed} duration={args.duration} "
            f"vel_profile={self._profile if self._profile else 'constant'} "
            f"stop_on_return_x={args.stop_on_return_x} return_x={args.return_x}"
        )

    def _on_qpos(self, msg):
        if not self.args.stop_on_return_x or not msg.data:
            return
        t = time.monotonic() - self.t0
        engage_start = self.args.wait_init + self.args.press + self.args.settle
        if t < engage_start:
            return
        x = float(msg.data[0])
        if x >= self.args.return_arm_x:
            self._return_armed = True
        if self._return_armed and self._return_stop_time is None and x <= self.args.return_x:
            self._return_stop_time = t
            self.get_logger().info(
                f"return-x hard stop triggered: x={x:+.4f} <= {self.args.return_x:+.4f} "
                f"at t={t:.2f}s"
            )

    def _engage_speed(self, t):
        """Forward stick command during the engage phase.

        With a vel_profile set, walks the segment list to find the active
        segment, then linearly interpolates within it (constant segments have
        spd_start == spd_end so interpolation is a no-op).
        Without a profile, returns the constant args.speed.
        """
        if not self._profile:
            return self.args.speed
        a = self.args
        t_engage_start = a.wait_init + a.press + a.settle
        te = t - t_engage_start            # elapsed within the engage window
        acc = 0.0
        for dur, s0, s1 in self._profile:
            if te < acc + dur:
                frac = (te - acc) / dur if dur > 0 else 1.0
                return s0 + (s1 - s0) * frac
            acc += dur
        return self._profile[-1][2]        # hold last segment's end value past the schedule

    def _phase(self, t):
        a = self.args
        t_press_end = a.wait_init + a.press         # B (lay→stand) released
        t_engage_start = t_press_end + a.settle     # LB+RB engage begins
        t_engage_end = t_engage_start + a.duration if a.duration > 0 else float("inf")
        t_slowdown_end = t_engage_end + a.slowdown  # speed → 0 ramp ends
        t_laydown_end = t_slowdown_end + a.laydown  # A (stand→lay) released
        t_done = t_laydown_end + a.tail             # tail = post-laydown idle
        if t < a.wait_init:
            return "init"
        if t < t_press_end:
            return "stand"
        if t < t_engage_start:
            return "settle"
        if self._return_stop_time is not None:
            stop_end = self._return_stop_time + a.return_stop_hold
            laydown_end = stop_end + a.laydown
            if t < self._return_stop_time:
                return "engage"
            if t < stop_end:
                return "hard_stop"
            if t < laydown_end:
                return "laydown"
            if t < laydown_end + a.tail:
                return "post_laydown"
            return "done"
        if t < t_engage_end:
            return "engage"
        if t < t_slowdown_end:
            return "slowdown"
        if t < t_laydown_end:
            return "laydown"
        if t < t_done:
            return "post_laydown"
        return "done"

    def _tick(self):
        t = time.monotonic() - self.t0
        phase = self._phase(t)
        if phase != self._last_phase:
            self.get_logger().info(f"phase -> {phase} (t={t:.2f}s)")
            self._last_phase = phase

        axes = [0.0] * NUM_AXES
        axes[AXIS_LT] = 1.0
        axes[AXIS_RT] = 1.0
        buttons = [0] * NUM_BUTTONS

        if phase == "stand":
            buttons[BTN_B] = 1
        elif phase == "engage":
            buttons[BTN_LB] = 1
            buttons[BTN_RB] = 1
            axes[AXIS_LY] = self._engage_speed(t)
        elif phase == "hard_stop":
            # Keep the policy engaged but step the command directly to zero.
            buttons[BTN_LB] = 1
            buttons[BTN_RB] = 1
            axes[AXIS_LY] = 0.0
        elif phase == "slowdown":
            # Linear ramp speed → 0 over args.slowdown seconds, LB+RB still
            # held so the policy stays engaged while we wind the stick down.
            a = self.args
            t_eng_end = a.wait_init + a.press + a.settle + a.duration
            frac = max(0.0, min(1.0, (t - t_eng_end) / max(a.slowdown, 1e-3)))
            buttons[BTN_LB] = 1
            buttons[BTN_RB] = 1
            axes[AXIS_LY] = self.args.speed * (1.0 - frac)
        elif phase == "laydown":
            buttons[BTN_A] = 1
        elif phase == "done":
            self.publish_idle()
            rclpy.shutdown()
            return
        # 'init', 'settle', 'post_laydown' fall through with idle axes/buttons

        self._publish(axes, buttons)

    def _publish(self, axes, buttons):
        msg = Joy()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.axes = axes
        msg.buttons = buttons
        self.pub.publish(msg)

    def publish_idle(self):
        axes = [0.0] * NUM_AXES
        axes[AXIS_LT] = 1.0
        axes[AXIS_RT] = 1.0
        self._publish(axes, [0] * NUM_BUTTONS)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--wait-init", type=float, default=2.0,
                   help="seconds of idle before pressing B")
    p.add_argument("--press", type=float, default=1.0,
                   help="seconds to hold B (lay->stand trigger)")
    p.add_argument("--settle", type=float, default=4.0,
                   help="seconds to wait after B before engaging policy")
    p.add_argument("--speed", type=float, default=0.0,
                   help="forward stick command in [-1, 1] while policy is engaged")
    p.add_argument("--vel_profile", type=str, default="",
                   help="Forward-speed schedule during engage as comma-separated "
                        "segments.  Two-part 'dur:spd' is a constant step; "
                        "three-part 'dur:s0:s1' is a linear ramp from s0 to s1.  "
                        "E.g. '6:1,2:0,4:0:1,8:1' = 6 s @ 1, 2 s @ 0, "
                        "4 s ramp 0→1, 8 s @ 1.  Overrides --speed.  "
                        "Total segment duration should equal --duration.")
    p.add_argument("--duration", type=float, default=0.0,
                   help="engage-phase seconds (LB+RB held); 0 = forever")
    p.add_argument("--stop-on-return-x", type=lambda v: v.lower() in ("1", "true", "yes"), default=False,
                   help="hard-stop after advancing, when base x returns to the threshold")
    p.add_argument("--return-x", type=float, default=0.0,
                   help="base-x threshold that triggers the hard stop")
    p.add_argument("--return-arm-x", type=float, default=1.0,
                   help="minimum forward base x required before the return trigger is armed")
    p.add_argument("--return-stop-hold", type=float, default=2.0,
                   help="seconds to stay engaged at zero velocity before laydown")
    p.add_argument("--slowdown", type=float, default=2.0,
                   help="seconds to linearly ramp axes[1] from speed → 0 "
                        "after engage and before laydown.")
    p.add_argument("--laydown", type=float, default=3.0,
                   help="seconds to hold A after engage (stand → lay).")
    p.add_argument("--tail", type=float, default=2.0,
                   help="idle seconds after laydown before shutdown.")
    # ros2 launch injects '--ros-args ...' on top of our CLI flags; ignore them.
    args, _ = p.parse_known_args()
    return args


def main():
    args = parse_args()
    rclpy.init()
    node = JoySequencer(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # one last all-released Joy so the FSM disengages politely
        try:
            node.publish_idle()
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
