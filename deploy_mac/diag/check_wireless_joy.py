#!/usr/bin/env python3
"""Cross-process contract check for deploy_rl_policy/scripts/wireless_joy.py.

`wireless_joy.py --self-test` proves the pure mapping. It cannot prove the part
that only exists once the node runs: that a frame arriving on /lowstate leaves
the process as a /joy message carrying the indices its two consumers read —
and that the node goes SILENT when the source stops, which is the precondition
for `joy_timeout_s` to fire (a node that keeps republishing its last state
holds an engage forever, and the dead-man never trips).

So this harness:
  * spawns the node as-is (it has exactly one source, `/wirelesscontroller`),
  * publishes synthetic WirelessController messages at ~20 Hz, which is what the
    board does while the remote is awake (measured: a steady 20 Hz even with the
    stick held still). That topic is the node's whole input surface, so nothing
    about the real robot is needed to run this,
  * asserts five things:
      1. L1+R1  -> buttons[4]&[5], no other button, L2/R2 resting at +1.0
      2. the left stick's ly -> axes[1]  (xbox_command.get_left_stick()[0] = cmd_vx)
      3. L2+R2  -> axes[2] < -0.9 and axes[5] < -0.9   (the stop ladder)
      4. messages stop -> /joy stops, measured over 1.5x SOURCE_HOLD_S
      5. two SIGINTs (what `ros2 launch` sends) shut it down clean: exit 0 and
         no traceback, which is the bug that showed up here on 2026-10-29 and
         had already been fixed once in distill_fused_policy.py

Safe by construction and therefore guarded: this publishes a fake
/wirelesscontroller, so it REFUSES domain 0. On the robot's own domain a
low_level_ctrl or distill_fused_policy would read it as a real remote and act
on the buttons it fabricates.

Usage:
    deploy_mac/diag/check_wireless_joy.py [--domain 99] [--node PATH] [--verbose]

    # on the Jetson stage
    GO2_DEPLOY_ROOT=~/go2_deploy deploy_mac/diag/check_wireless_joy.py

Exit 0 = contract OK; exit 1 = a check failed; exit 2 = cannot run here.
"""
import argparse
import importlib.util
import os
import signal
import subprocess
import sys
import threading
import time


def _prescan_domain(default="99"):
    """--domain must land in the environment before rclpy is imported."""
    argv = sys.argv[1:]
    for i, a in enumerate(argv):
        if a == "--domain" and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith("--domain="):
            return a.split("=", 1)[1]
    return default


_DOMAIN = _prescan_domain()
os.environ["ROS_DOMAIN_ID"] = _DOMAIN

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Joy
    from unitree_go.msg import WirelessController
except ImportError as exc:                      # no ROS env, or no unitree_go
    print(f"check_wireless_joy: cannot run here ({exc}); source the ROS env first")
    raise SystemExit(2)

_REPO = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))


def find_node():
    """Prefer the deployed copy, fall back to the source tree so this works
    before the package is installed anywhere."""
    cands = []
    root = os.environ.get("GO2_DEPLOY_ROOT")
    if root:
        cands.append(os.path.join(root, "ws/install/deploy_rl_policy/lib",
                                  "deploy_rl_policy", "wireless_joy.py"))
    try:
        from ament_index_python.packages import get_package_prefix
        cands.append(os.path.join(get_package_prefix("deploy_rl_policy"), "lib",
                                  "deploy_rl_policy", "wireless_joy.py"))
    except Exception:
        pass
    cands.append(os.path.join(_REPO, "src", "deploy_rl_policy", "scripts",
                              "wireless_joy.py"))
    for cand in cands:
        if os.path.isfile(cand):
            return os.path.normpath(cand)
    return None


def load_node_constants(path):
    """Import the node module (no rclpy.init, no node) to steal its constants."""
    spec = importlib.util.spec_from_file_location("wireless_joy_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Feeder(Node):
    """Stands in for the main board's /wirelesscontroller publisher."""

    def __init__(self):
        super().__init__("check_wireless_joy_feeder")
        self.pub = self.create_publisher(
            WirelessController, "/wirelesscontroller", 10)
        self.joy = []
        self.create_subscription(Joy, "/joy", lambda m: self.joy.append(m), 10)

    def pump(self, seconds, timeout_sec=0.05):
        end = time.time() + seconds
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=timeout_sec)

    def send(self, keys, ly=0.0, seconds=1.0):
        """Hold a state for `seconds` at ~20 Hz, spinning so /joy is collected.

        Repeated, not published once: the node treats SOURCE_HOLD_S of silence
        as a dead remote, and repeating is also what the board does (20 Hz).
        """
        end = time.time() + seconds
        while time.time() < end:
            msg = WirelessController()
            msg.keys = int(keys)
            msg.lx = 0.0
            msg.ly = float(ly)
            msg.rx = 0.0
            msg.ry = 0.0
            self.pub.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.02)
            time.sleep(0.05)

    def last(self):
        return self.joy[-1] if self.joy else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--domain", default=_DOMAIN,
                    help="ROS domain to run on (default 99; 0 is refused)")
    ap.add_argument("--node", help="path to wireless_joy.py (default: auto-detect)")
    ap.add_argument("--verbose", action="store_true", help="show the node's log")
    opts = ap.parse_args()

    if int(opts.domain) == 0:
        print("check_wireless_joy: refusing domain 0 — this publishes a synthetic "
              "/wirelesscontroller, which the robot's own /joy consumers would "
              "read as a real remote. Pick a sandbox domain.")
        return 2

    node_path = os.path.abspath(opts.node) if opts.node else find_node()
    if not node_path or not os.path.isfile(node_path):
        print("check_wireless_joy: wireless_joy.py not found "
              "(set GO2_DEPLOY_ROOT, or pass --node)")
        return 2
    mod = load_node_constants(node_path)
    hold = float(getattr(mod, "SOURCE_HOLD_S", 1.0))

    print(f"node   : {node_path}")
    print(f"domain : {os.environ['ROS_DOMAIN_ID']}"
          f"  (synthetic /wirelesscontroller; the robot is untouched)")
    print(f"hold   : /joy must stop within {hold:.2f} s of the last source message")

    proc = subprocess.Popen(
        [sys.executable, node_path],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    log_lines = []

    def drain():
        for line in proc.stdout:
            log_lines.append(line.rstrip())

    threading.Thread(target=drain, daemon=True).start()

    failures = []

    def report(name, ok, detail):
        print(f"  {'OK ' if ok else 'BAD'} {name}: {detail}")
        if not ok:
            failures.append(name)

    rclpy.init()
    feeder = Feeder()
    try:
        # Wait for the node's subscription to appear; a blind sleep would make
        # this flaky on a slow Jetson and hide a node that died on startup.
        deadline = time.time() + 20
        while time.time() < deadline and feeder.count_subscribers(
                "/wirelesscontroller") == 0:
            feeder.pump(0.1)
        feeder.pump(0.5)                        # let the /joy match resolve

        # 1. the engage gate, and nothing else moving
        feeder.joy.clear()
        feeder.send(mod.K_L1 | mod.K_R1, seconds=1.5)
        got = feeder.last()
        report("L1+R1 -> engage buttons",
               got is not None and list(got.buttons) == [0, 0, 0, 0, 1, 1]
               and got.axes[2] == 1.0 and got.axes[5] == 1.0,
               f"buttons={list(got.buttons) if got else None} "
               f"axes[2]={got.axes[2] if got else None} "
               f"axes[5]={got.axes[5] if got else None}")

        # 2. the axis that drives the robot: xbox_command.get_left_stick()
        #    returns (axes[1], axes[0]), so load('cmd_vx') reads axes[1].
        feeder.joy.clear()
        feeder.send(0, ly=0.4, seconds=1.0)
        got = feeder.last()
        want = 0.4
        report("left stick ly -> axes[1] (cmd_vx)",
               got is not None and abs(got.axes[1] - want) < 1e-3
               and got.axes[0] == 0.0 and not any(got.buttons),
               f"axes[0]={got.axes[0] if got else None} "
               f"axes[1]={got.axes[1] if got else None} (want 0.0, {want:.3f})")

        # 3. the stop ladder
        feeder.joy.clear()
        feeder.send(mod.K_L2 | mod.K_R2, seconds=1.0)
        got = feeder.last()
        report("L2+R2 -> stop ladder",
               got is not None and got.axes[2] < -0.9 and got.axes[5] < -0.9,
               f"axes[2]={got.axes[2] if got else None} "
               f"axes[5]={got.axes[5] if got else None}")

        # 4. silence. The subscription queue is depth 10 and nothing spins while
        #    this sleeps, so frames published BEFORE the silence are still queued:
        #    drain those first, then measure a window.
        time.sleep(hold + 0.5)
        feeder.pump(0.6)
        stale = len(feeder.joy)
        feeder.joy.clear()
        window = max(1.5, 1.5 * hold)
        feeder.pump(window)
        report("silent source -> /joy stops", not feeder.joy,
               f"{len(feeder.joy)} msgs in {window:.1f} s of silence (want 0); "
               f"{stale} drained before the window")

        report("node stayed up", proc.poll() is None,
               "alive" if proc.poll() is None else f"exited {proc.returncode}")

        # 5. shutdown path. `ros2 launch` follows its SIGINT with a second one
        #    as soon as the first is swallowed; the second used to land inside
        #    destroy_node() and print a traceback.
        proc.send_signal(signal.SIGINT)
        time.sleep(0.4)
        if proc.poll() is None:
            proc.send_signal(signal.SIGINT)
        try:
            rc = proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            rc = -9
        time.sleep(0.2)                         # let the drain thread finish
        tracebacks = [line for line in log_lines if "Traceback" in line]
        report("clean shutdown on double SIGINT", rc == 0 and not tracebacks,
               f"exit={rc} tracebacks={len(tracebacks)}")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        feeder.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        time.sleep(0.1)                         # let the drain thread finish

    if opts.verbose or failures:
        print("  --- node output ---")
        for line in log_lines[-20:]:
            print(f"    {line}")
    print("check_wireless_joy: OK" if not failures
          else f"check_wireless_joy: FAILED -> {', '.join(failures)}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
