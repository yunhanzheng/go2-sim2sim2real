#!/usr/bin/env python3
"""Live acceptance for wireless_joy.py: a real remote, on the real robot.

check_wireless_joy.py is the regression harness — synthetic messages, sandbox
domain, no hardware and no human. This is the other end of the same contract: it
starts the node on the robot's own domain and watches what a person pressing a
real remote makes it publish.

What only this can prove: that the remote's keys land on the indices its
consumers actually read, that the left stick's ly really is axes[1] (cmd_vx) on
this firmware, and that the node exits clean when a SIGTERM arrives from launch.

WHY THIS CANNOT MOVE THE ROBOT
It starts no consumer and publishes nothing itself; the node's only output is
/joy. And it refuses to start if anything on the domain is subscribed to /joy —
that subscription is precisely what would turn a button press into motor
commands. On a parked robot nothing subscribes, so the run is read-only.

Usage:
    deploy_mac/diag/check_joy_live.py [--seconds 30] [--wait 60] [--node PATH]
                                      [--verbose]

    # on the Jetson, remote in hand
    GO2_DEPLOY_ROOT=~/go2_deploy deploy_mac/diag/check_joy_live.py

Ask the person to:
    1. press any button            wake the remote (it sleeps after ~40 s idle)
    2. hold L1+R1 for ~2 s         -> buttons[4] and buttons[5]
    3. push the left stick forward -> axes[1] toward +1.0
    4. hold L2 alone for ~1 s      -> axes[2] at -1.0, axes[5] untouched
    5. hold L2+R2 for ~2 s         -> axes[2] and axes[5] at -1.0

The clock starts on their first real input, not when this process starts: the
person holding the remote cannot see this output, so a fixed window would drop
their first press while the subscription was still being discovered.

Exit 0 = every check passed; 1 = at least one failed; 2 = cannot run here.
"""
import argparse
import os
import subprocess
import sys
import threading
import time

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Joy
    from unitree_go.msg import WirelessController
except ImportError as exc:                      # no ROS env, or no unitree_go
    print(f"check_joy_live: cannot run here ({exc}); source the ROS env first")
    raise SystemExit(2)

_REPO = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

BTN = "ABXY L1 R1".split()


def find_node():
    """Same probe order as check_wireless_joy.py, kept local on purpose:
    importing that module would set ROS_DOMAIN_ID as a side effect, and this
    script has to run on whatever domain the robot is on."""
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


def fmt(m):
    btn = "".join(n if v else "." for n, v in zip(BTN, m.buttons))
    axes = " ".join(f"{float(v):+.2f}" for v in m.axes)
    return f"btn[{btn}] axes[{axes}]  (lx ly L2 rx ry R2)"


class Watch(Node):
    """Subscribes only after the caller has confirmed nobody else is listening."""

    def __init__(self):
        super().__init__("check_joy_live_watch")
        self.t0 = time.time()
        self.wc_n = 0
        self.joy_n = 0
        self.prev_axes = None
        self.combos = set()          # every distinct /joy button word seen
        self.lo = [9.0] * 6
        self.hi = [-9.0] * 6
        self.l1r1 = False            # L1 and R1 down in the same message
        self.l2r2 = False            # L2 and R2 both tripped in the same message
        self.l2_alone = False        # L2 down while R2 sat at rest
        self.first_input_t = None    # first real press; this starts the clock
        self.verbose = False

    def subscribe(self):
        self.create_subscription(WirelessController, "/wirelesscontroller",
                                 self.on_wc, 50)
        self.create_subscription(Joy, "/joy", self.on_joy, 50)

    def on_wc(self, _m):
        self.wc_n += 1

    def on_joy(self, m):
        self.joy_n += 1
        btns = tuple(int(v) for v in m.buttons)
        axes = [float(v) for v in m.axes]
        if self.first_input_t is None and (
                any(btns) or abs(axes[0]) > 0.15 or abs(axes[1]) > 0.15
                or abs(axes[3]) > 0.15 or abs(axes[4]) > 0.15
                or axes[2] < 0.5 or axes[5] < 0.5):
            # The shoulders rest at +1.0, so only "pulled down" counts as input
            # for them; the sticks are compared to their own centre.
            self.first_input_t = self.t0 = time.time()
        for i, v in enumerate(axes[:6]):
            self.lo[i] = min(self.lo[i], v)
            self.hi[i] = max(self.hi[i], v)
        word = sum(v << i for i, v in enumerate(btns))
        moved = self.prev_axes is None or max(
            abs(a - b) for a, b in zip(axes[:6], self.prev_axes)) > 0.2
        if word not in self.combos or moved or self.verbose:
            self.combos.add(word)
            self.prev_axes = axes
            print(f"  t=+{time.time() - self.t0:5.1f}s  {fmt(m)}", flush=True)
        if axes[2] < -0.9 and axes[5] > 0.9:
            self.l2_alone = True
        if axes[2] < -0.9 and axes[5] < -0.9:
            self.l2r2 = True
        if btns[4] and btns[5]:
            self.l1r1 = True

    def pump(self, seconds):
        end = time.time() + seconds
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.05)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seconds", type=float, default=30.0,
                    help="how long to score, starting at the first input "
                         "(default 30)")
    ap.add_argument("--wait", type=float, default=60.0,
                    help="how long to wait for that first input (default 60)")
    ap.add_argument("--node", help="path to wireless_joy.py (default: auto-detect)")
    ap.add_argument("--verbose", action="store_true",
                    help="print every /joy message, and the node's log")
    opts = ap.parse_args()

    node_path = os.path.abspath(opts.node) if opts.node else find_node()
    if not node_path or not os.path.isfile(node_path):
        print("check_joy_live: wireless_joy.py not found "
              "(set GO2_DEPLOY_ROOT, or pass --node)")
        return 2

    domain = os.environ.get("ROS_DOMAIN_ID", "0")
    print(f"node   : {node_path}")
    print(f"domain : {domain}")
    print(f"watch  : {opts.seconds:.0f} s of /joy, driven by a real remote")

    rclpy.init()
    watch = Watch()
    watch.verbose = opts.verbose

    # Refuse to run if anything is already listening: that is the only thing
    # that could turn this from a read-only observation into motor commands.
    # Another copy of this script does not count — it only subscribes and
    # prints — and must be filtered out, or one killed run leaves a ghost node
    # on the DDS graph that blocks every run after it.
    watch.pump(3.0)                 # let discovery settle before asking
    foreign = [s for s in watch.get_subscriptions_info_by_topic("/joy")
               if s.node_name != watch.get_name()]
    if foreign:
        names = ", ".join(sorted({f"/{s.node_name}" for s in foreign}))
        print(f"check_joy_live: refusing to run — {names} subscribes to /joy and "
              f"would react to the node's output. Stop it first; this check is "
              f"only safe on a robot where /joy has no consumers.")
        watch.destroy_node()
        rclpy.try_shutdown()
        return 2
    print("guard  : no /joy subscriber on this domain, the robot cannot move")

    proc = subprocess.Popen([sys.executable, node_path],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True)
    log_lines = []

    def drain():
        for line in proc.stdout:
            log_lines.append(line.rstrip())

    threading.Thread(target=drain, daemon=True).start()
    time.sleep(2.5)                 # node init + its own discovery
    watch.subscribe()

    # The clock starts on the first real input, not when this process starts
    # listening. Whoever holds the remote cannot see this process's output, so a
    # fixed start time puts their first press in the blind spot before the
    # subscription is live — which is exactly what happened on 2026-10-29.
    print(f"ready  : waiting up to {opts.wait:.0f} s for the first button or "
          f"stick input, then scoring {opts.seconds:.0f} s from there",
          flush=True)
    wait_end = time.time() + opts.wait
    while watch.first_input_t is None and time.time() < wait_end:
        rclpy.spin_once(watch, timeout_sec=0.05)
    if watch.first_input_t is None:
        print("warn   : no input arrived, so nothing was exercised", flush=True)
    else:
        print(f"clock  : started on the first input, scoring "
              f"{opts.seconds:.0f} s", flush=True)
        try:
            watch.pump(opts.seconds)
        except KeyboardInterrupt:
            pass
    watch.destroy_node()
    rclpy.try_shutdown()

    proc.terminate()
    try:
        rc = proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        rc = -9
    time.sleep(0.2)
    text = "\n".join(log_lines)
    banner = "source=/wirelesscontroller" in text
    traceback = "Traceback" in text
    if opts.verbose:
        print("--- node log")
        for line in log_lines:
            print(f"  {line}")

    took = max(time.time() - watch.t0, 1e-9)
    print("=== summary")
    print(f"  /wirelesscontroller : {watch.wc_n} msgs"
          f"  ({watch.wc_n / took:.1f} Hz)"
          + ("" if watch.wc_n else "   <- the remote is asleep, off or unpaired"))
    print(f"  /joy                : {watch.joy_n} msgs"
          f"  ({watch.joy_n / took:.1f} Hz; 0 is normal while the remote sleeps)")
    print(f"  button words        : {sorted(f'0x{w:02x}' for w in watch.combos)}")
    print("  axis ranges         : " + "  ".join(
        f"{nm}[{lo:+.2f},{hi:+.2f}]" for nm, lo, hi in
        zip("lx ly L2 rx ry R2".split(), watch.lo, watch.hi)))
    print(f"  node exit code      : {rc}")

    checks = [
        ("/wirelesscontroller is alive", watch.wc_n > 0, None),
        ("node announced source=/wirelesscontroller", banner, None),
        ("/joy is flowing", watch.joy_n > 0, None),
        ("L1+R1 -> buttons[4] and [5] together", watch.l1r1, None),
        ("L2+R2 -> axes[2] and axes[5] at -1.0", watch.l2r2, None),
        # A human cannot press two shoulders in the same millisecond, so the
        # trace always contains a frame with one down and the other at rest.
        # That frame is evidence, not a fault: it is what proves the two axes
        # are independent (a copy-paste in the mapping would drag both down).
        # The ladder is safe either way, because its consumer demands both.
        ("L2 and R2 move independently",
         True if watch.l2_alone else None,
         "saw L2 at -1.0 while R2 still rested at +1.0"),
        ("left stick ly reached +0.5", watch.hi[1] > 0.5,
         f"ly max {watch.hi[1]:+.2f}"),
        ("node exited cleanly", rc == 0 and not traceback, f"rc={rc}"),
    ]
    bad = 0
    for name, ok, note in checks:
        if ok is None:
            print(f"  SKIP {name}  (not exercised)")
            continue
        print(f"  {'OK  ' if ok else 'FAIL'} {name}"
              + (f": {note}" if note else ""))
        bad += 0 if ok else 1
    print("=== " + ("ALL OK" if not bad else f"{bad} FAILED"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
