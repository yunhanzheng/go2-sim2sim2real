#!/usr/bin/env python3
"""Publish /joy from the Go2's own remote, so the robot needs no laptop.

The Joy layout below is COPIED from deploy_mac/run/wireless_to_joy.py, not
re-invented, because two consumers are written against it and a drift in either
the index or the trigger sign is a silent behaviour change:

  low_level_ctrl (C++)            buttons[0..5] = A B X Y L1 R1
                                  is_standing_ && buttons[4] && buttons[5] = engage
                                  axes[2] < -0.9 && axes[5] < -0.9 = L2+R2 ladder
  distill_fused_policy (Python)   xbox_command.XboxController: is_pressed() =
                                  buttons[4]/[5] gate on the policy itself,
                                  get_left_stick()[0] = axes[1] = cmd_vx

That second consumer is why this has to be a /joy publisher rather than logic
inside low_level_ctrl: the sticks feed the actor, not just the FSM.

ONE SOURCE, AND WHY THE OTHER ONE IS GONE
-----------------------------------------
/wirelesscontroller (unitree_go/WirelessController) is the source. The main
board decodes the remote and publishes keys (uint16) + lx/ly/rx/ry (float,
-1..+1). Measured on the robot 2026-10-29 with the remote awake: 466 messages,
keys word identical to LowState.wireless_remote[2:4], lx reaching -1.00/+1.00,
ly +0.90 with the left stick pushed forward, ry +1.00 the same way on the right.

LowState.wireless_remote is NOT a second source, and using it as a fallback is a
safety bug rather than a convenience. Measured the same day: the board repeats
the last 40-byte frame at 500 Hz indefinitely, so the field keeps its old value
after the remote sleeps (22 501 identical frames in 45 s with the remote
silent). Liveness taken from it is therefore always true, /joy never stops, and
joy_timeout_s — the dead-man that exists precisely for a remote that dies —
never fires. The frame that would have been replayed is whatever was last seen,
L1+R1 and a forward stick included, so the failure mode is a robot that keeps
walking with nobody holding anything. Its layout also differs from the
documented one on this firmware (header 0x55 0x51, sticks are float32), so the
decoder this file used to carry was wrong on top of being unsafe. It is kept as
a printed diagnostic only.

A source counts as live for SOURCE_HOLD_S after its last message; past that /joy
stops being published entirely, which is what lets joy_timeout_s (0.5 s default)
fire. A remote that dies must look like a remote that dies, not like a remote
that keeps reporting "nothing pressed".

Verify what the remote is doing before trusting a new robot:

    ros2 run deploy_rl_policy wireless_joy.py --probe

which prints keys, sticks, and the LowState mirror for comparison, and never
publishes. Then drive with

    ros2 run deploy_rl_policy wireless_joy.py

and pass use_joy:=false to the launch file — two /joy publishers flap
/policy_engaged at the message rate and the robot collapses on engage
(deploy_mac/README.md, 2026-07-27 real run).

READ-ONLY w.r.t. the robot: it publishes /joy and nothing else.
"""
import argparse
import signal

import rclpy
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Joy
from unitree_go.msg import LowState, WirelessController

# Bit values of the 16-bit `keys` word, exactly as deploy_mac/run/wireless_to_joy.py
# defines them. Confirmed against the robot: 0x0001/0x0002/0x0010/0x0020 come back
# from R1/L1/L2/R2 and the same word appears at LowState.wireless_remote[2:4].
K_R1, K_L1, K_R2, K_L2 = 0x0001, 0x0002, 0x0010, 0x0020
K_A, K_B, K_X, K_Y = 0x0100, 0x0200, 0x0400, 0x0800

# Little-endian offset of that word inside LowState.wireless_remote. Diagnostic
# only (see the module docstring) — nothing is decoded from it.
MIRROR_KEYS_OFFSET = 2

# A source counts as live for this long after its last message. Past it, /joy
# stops being published entirely so joy_timeout_s (0.5 s default) still fires.
# The board sends /wirelesscontroller at a steady 20 Hz while the remote is awake
# (measured 2026-10-29, 40 s with the left stick held still: 100 messages in every
# 5 s window, no gaps), so 1.0 s is 20 missed messages — long enough that a held
# remote is never mistaken for a dead one, short enough to be a dead-man.
SOURCE_HOLD_S = 1.0


def joy_from(keys, lx, ly, rx, ry):
    msg = Joy()
    msg.buttons = [
        1 if (keys & K_A) else 0, 1 if (keys & K_B) else 0,
        1 if (keys & K_X) else 0, 1 if (keys & K_Y) else 0,
        1 if (keys & K_L1) else 0, 1 if (keys & K_R1) else 0,
    ]
    msg.axes = [
        float(lx), float(ly),
        -1.0 if (keys & K_L2) else 1.0,
        float(rx), float(ry),
        -1.0 if (keys & K_R2) else 1.0,
    ]
    return msg


class WirelessJoy(Node):
    def __init__(self, opts):
        super().__init__("wireless_joy")
        self._opts = opts
        self._pub = None if opts.probe else self.create_publisher(Joy, "/joy", 10)

        self._keys = 0
        self._sticks = (0.0, 0.0, 0.0, 0.0)
        self._wc_t = None             # last /wirelesscontroller message
        self._wc_n = 0
        self._remote_n = 0            # /lowstate frames seen (diagnostic)
        self._last_raw = None
        self._mirror_keys = 0
        self._last_print = None       # probe: last printed state
        self._last_probe_t = 0.0
        self._drops = 0
        self._last_drop_log_t = 0.0

        self.create_subscription(
            WirelessController, "/wirelesscontroller", self._on_wc, 10)
        self.create_subscription(LowState, "/lowstate", self._on_lowstate, 10)
        # Publishing from a timer rather than from the message callback: the FSM
        # reads /joy on its own schedule and its press-edge logic was written
        # against 20-50 Hz, so 50 Hz is both cheaper and closer to the behaviour
        # the stop ladder was tuned on.
        self.create_timer(1.0 / opts.rate_hz, self._tick)
        self.create_timer(10.0, self._heartbeat)

    # ---- source ------------------------------------------------------------
    def _on_wc(self, m):
        self._wc_n += 1
        self._wc_t = self.get_clock().now()
        self._keys = int(m.keys)
        self._sticks = (float(m.lx), float(m.ly), float(m.rx), float(m.ry))

    def _on_lowstate(self, m):
        # DIAGNOSTIC ONLY. Not a source: the board mirrors the last frame at
        # 500 Hz forever, so a message arriving here says nothing about the
        # remote still being alive. The [2:4] word is surfaced next to
        # /wirelesscontroller's keys so --probe can show whether the two agree.
        self._remote_n += 1
        raw = list(m.wireless_remote)
        self._last_raw = raw
        if len(raw) > MIRROR_KEYS_OFFSET + 1:
            self._mirror_keys = raw[MIRROR_KEYS_OFFSET] | (
                raw[MIRROR_KEYS_OFFSET + 1] << 8)

    def _live(self, t):
        if t is None:
            return False
        return (self.get_clock().now() - t).nanoseconds * 1e-9 <= SOURCE_HOLD_S

    # ---- output ------------------------------------------------------------
    def _tick(self):
        if not self._live(self._wc_t):
            # Silence is the signal: low_level_ctrl's joy_timeout_s (0.5 s) is
            # the dead-man, and it can only fire if this node stops talking.
            return
        if self._opts.probe:
            self._probe(f"keys=0x{self._keys:04x} sticks={_fmt(self._sticks)}"
                        f"  mirror[2:4]=0x{self._mirror_keys:04x}"
                        f" raw={_hex(self._last_raw)}")
            return
        msg = joy_from(self._keys, *self._sticks)
        msg.header.stamp = self.get_clock().now().to_msg()
        self._pub.publish(msg)

    def _probe(self, line, force=False):
        # Dedupe + a floor of 200 ms between lines: the sticks carry float noise
        # and a probe that floods the journal hides the one line the operator is
        # looking for.
        now = self.get_clock().now().nanoseconds * 1e-9
        if not force:
            if line == self._last_print or now - self._last_probe_t < 0.2:
                return
        self._last_print = line
        self._last_probe_t = now
        self.get_logger().info(f"probe: {line}")

    def note_drop(self, why):
        """Count a sample rclpy could not convert. See main() for why."""
        self._drops += 1
        now = self.get_clock().now().nanoseconds * 1e-9
        if now - self._last_drop_log_t > 5.0:
            self._last_drop_log_t = now
            self.get_logger().warning(
                f"dropped an undecodable sample ({why}); {self._drops} so far")

    def _heartbeat(self):
        # Kept from deploy_mac/run/wireless_to_joy.py: on 2026-07-27 the remote
        # "stopped working" and the only evidence was a banner, so a dead source
        # stream and a dead bridge had to be distinguishable.
        wc, rm = self._wc_n, self._remote_n
        self._wc_n = self._remote_n = 0
        parts = [f"wirelesscontroller {wc} msgs",
                 f"lowstate {rm} msgs (mirror keys 0x{self._mirror_keys:04x}, "
                 f"not a source)"]
        if self._drops:
            parts.append(f"{self._drops} undecodable")
            self._drops = 0
        if wc:
            tail = ""
        elif rm:
            tail = (" (no remote traffic; the lowstate mirror keeps moving even "
                    "while the remote sleeps)")
        else:
            tail = " (no traffic at all)"
        self.get_logger().info(f"heartbeat: {' | '.join(parts)}{tail}")


def _fmt(sticks):
    return "(" + ", ".join(f"{v:+.2f}" for v in sticks) + ")"


def _hex(raw):
    if not raw:
        return "-"
    return " ".join(f"{b:02x}" for b in raw[:12]) + (" ..." if len(raw) > 12 else "")


def _self_test() -> int:
    """Assert the mapping rather than trusting it.

    Two consumers read these indices and one of them is the stop ladder, so a
    drift in either direction is a silent behaviour change, not a bug report.
    No node, no DDS entities — but the ROS environment must be sourced, because
    the message types are imported at module level like every script here:

        ros2 run deploy_rl_policy wireless_joy.py --self-test
    """
    failed = []

    def check(name, got, want):
        if got != want:
            failed.append(f"{name}: {got!r} != {want!r}")

    every = K_A | K_B | K_X | K_Y | K_L1 | K_R1 | K_L2 | K_R2
    j = joy_from(every, 0.5, -0.25, 0.0, 0.0)
    check("buttons A B X Y L1 R1", list(j.buttons), [1, 1, 1, 1, 1, 1])
    check("axes lx ly L2 rx ry R2", [round(float(v), 3) for v in j.axes],
          [0.5, -0.25, -1.0, 0.0, 0.0, -1.0])

    engage = joy_from(K_L1 | K_R1, 0, 0, 0, 0)
    check("L1+R1 sets the engage pair", (engage.buttons[4], engage.buttons[5]),
          (1, 1))
    check("L1+R1 leaves the face buttons up", list(engage.buttons),
          [0, 0, 0, 0, 1, 1])

    rest = joy_from(0, 0, 0, 0, 0)
    # The consumers test "< -0.9", so rest has to be +1.0, never 0.
    check("rest sits at +1.0", (float(rest.axes[2]), float(rest.axes[5])),
          (1.0, 1.0))
    check("rest does not trip the stop ladder",
          rest.axes[2] < -0.9 and rest.axes[5] < -0.9, False)
    # The ladder needs BOTH shoulders held: L2 alone must leave R2's axis at
    # rest, or one stray press would walk the state machine down a rung.
    l2_only = joy_from(K_L2, 0, 0, 0, 0)
    check("L2 alone does not trip the stop ladder",
          (float(l2_only.axes[2]), float(l2_only.axes[5])), (-1.0, 1.0))
    stop = joy_from(K_L2 | K_R2, 0, 0, 0, 0)
    check("L2+R2 trips the stop ladder", (float(stop.axes[2]), float(stop.axes[5])),
          (-1.0, -1.0))
    for line in failed:
        print(f"FAIL {line}")
    print("self-test OK" if not failed else f"self-test FAILED ({len(failed)})")
    return 0 if not failed else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--probe", action="store_true",
                    help="print keys/sticks and the lowstate mirror, never publish")
    ap.add_argument("--self-test", action="store_true",
                    help="check the button/axis mapping against both consumers' "
                         "expectations and exit (needs the ROS env, no node)")
    ap.add_argument("--rate-hz", type=float, default=50.0,
                    help="publish rate while /wirelesscontroller is live (default 50)")
    # parse_known_args, not parse_args: `ros2 launch` appends
    # `--ros-args -r __node:=wireless_joy` and argparse would exit 2 on it.
    opts, _unknown = ap.parse_known_args()

    if opts.self_test:
        raise SystemExit(_self_test())

    rclpy.init()
    node = WirelessJoy(opts)
    node.get_logger().info(
        f"wireless_joy: source=/wirelesscontroller rate={opts.rate_hz:g} Hz "
        f"{'PROBE (publishing nothing)' if opts.probe else 'publishing /joy'}")
    try:
        # Hand-rolled spin instead of rclpy.spin: rclpy raises OUT of the
        # executor when a sample cannot be deserialized, with no callback to
        # catch it in — and the robot's 500 Hz /lowstate does that occasionally
        # under the bare-DDS publisher (2026-10-29: 1 in ~5000). The C++
        # subscriber just drops the sample; dying would take the /joy source
        # down mid-run, which is how a robot ends up with a frozen last command.
        executor = SingleThreadedExecutor()
        executor.add_node(node)
        try:
            while rclpy.ok():
                try:
                    executor.spin_once(timeout_sec=0.1)
                except ExternalShutdownException:
                    # Not a crash: rclpy turns an outside SIGTERM into a
                    # context shutdown (ros2 launch escalates to SIGTERM after
                    # its SIGINT grace), and raises it here to break the wait.
                    break
                except RuntimeError as exc:
                    # Two very different RuntimeErrors land here: a sample rclpy
                    # could not deserialize (drop it, keep going) and the wait
                    # set finding its context already shut down (SIGTERM raced
                    # the SIGINT handler — just leave, do not count it as a
                    # bad frame or the log claims a protocol problem that is
                    # really just shutdown).
                    if not rclpy.ok():
                        break
                    node.note_drop(str(exc))
        except KeyboardInterrupt:
            pass
        finally:
            executor.shutdown()
            # ros2 launch follows its first SIGINT with a second one; ignoring
            # it keeps the teardown from being cut in half (same fix as
            # distill_fused_policy.py).
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
    except KeyboardInterrupt:
        # Ctrl-C before the executor loop starts (rclpy.init/add_node window).
        pass


if __name__ == "__main__":
    main()
