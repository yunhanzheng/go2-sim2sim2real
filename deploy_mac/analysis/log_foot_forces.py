#!/usr/bin/env python3
"""READ-ONLY: log all 4 Go2 foot forces from /lowstate to CSV for a fixed window,
then report stats and auto-detect the AIRBORNE (out-of-contact) segment to measure
the per-foot noise floor. Use it to choose a contact threshold that sits safely
above the airborne noise (the policy thresholds foot_force at 1.0 N for L00-02).

Workflow:
  1. Stand the robot.
  2. python deploy_mac/analysis/log_foot_forces.py --secs 40
  3. A few seconds in, LIFT the robot and HOLD it in the air until the run ends.
  4. Read the AIRBORNE stats it prints (also dumps full CSV for re-analysis).

Subscribe-only: never publishes, never touches /lowcmd, never commands motors.
"""
import argparse
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from unitree_go.msg import LowState

FEET = ["FR", "FL", "RR", "RL"]   # /lowstate.foot_force order (SDK/MJ)


class Sub(Node):
    def __init__(self):
        super().__init__("log_foot_forces")
        self.rows = []
        self.t0 = None
        self.create_subscription(LowState, "lowstate", self._cb, 50)  # subscribe only

    def _cb(self, m):
        t = time.time()
        if self.t0 is None:
            self.t0 = t
        ff = [float(x) for x in m.foot_force]
        self.rows.append((t - self.t0, ff[0], ff[1], ff[2], ff[3]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--secs", type=float, default=40.0, help="recording duration")
    ap.add_argument("--out", default="/tmp/foot_forces.csv", help="CSV output path")
    ap.add_argument("--threshold", type=float, default=1.0,
                    help="contact threshold (N) to evaluate against airborne noise (L00-02 uses 1.0)")
    args = ap.parse_args()

    rclpy.init()
    node = Sub()
    print(f"recording foot forces for {args.secs:.0f}s ... lift + hold the robot partway through")
    end = time.time() + args.secs
    while time.time() < end:
        rclpy.spin_once(node, timeout_sec=0.1)
    rows = node.rows
    node.destroy_node()
    rclpy.shutdown()

    if not rows:
        print("FAIL: no /lowstate samples (robot connected + standing? GO2_IFACE correct?)")
        sys.exit(1)

    arr = np.array(rows)
    t = arr[:, 0]
    F = arr[:, 1:5]
    hz = len(rows) / max(t[-1], 1e-6)
    with open(args.out, "w") as f:
        f.write("t,FR,FL,RR,RL\n")
        for r in rows:
            f.write(f"{r[0]:.4f},{r[1]:.1f},{r[2]:.1f},{r[3]:.1f},{r[4]:.1f}\n")
    print(f"logged {len(rows)} samples over {t[-1]:.1f}s ({hz:.0f} Hz) -> {args.out}")

    print("\noverall per-foot raw force  (min / mean / max):")
    for i, nm in enumerate(FEET):
        print(f"  {nm}: {F[:, i].min():.0f} / {F[:, i].mean():.1f} / {F[:, i].max():.0f}")

    # Auto-detect airborne: total force across 4 feet drops low when lifted.
    tot = F.sum(axis=1)
    thr_tot = tot.min() + 0.25 * (tot.max() - tot.min())
    air = tot < thr_tot
    if air.sum() < 20:
        print("\nNo clear airborne window detected — make sure you lift + hold the robot "
              "during the recording. (Full CSV saved for manual inspection.)")
        sys.exit(0)

    Fa = F[air]
    ta = t[air]
    print(f"\nAIRBORNE window: {air.sum()} samples (~{ta.min():.1f}-{ta.max():.1f}s, "
          f"total-force < {thr_tot:.0f})")
    print("per-foot NOISE when out of contact  (min / mean / std / max):")
    for i, nm in enumerate(FEET):
        c = Fa[:, i]
        print(f"  {nm}: {c.min():.1f} / {c.mean():.2f} / {c.std():.2f} / {c.max():.1f}")
    overall_max = Fa.max()
    print(f"\n  max airborne force across all feet = {overall_max:.1f}")
    print(f"  at threshold = {args.threshold:.1f} N, airborne FALSE-contact rate per foot:")
    for i, nm in enumerate(FEET):
        pct = 100.0 * (Fa[:, i] > args.threshold).mean()
        print(f"    {nm}: {pct:.1f}% of airborne samples read as 'in contact'")
    print(f"  => pick a contact threshold safely above {overall_max:.1f} (and below standing load).")


if __name__ == "__main__":
    main()
