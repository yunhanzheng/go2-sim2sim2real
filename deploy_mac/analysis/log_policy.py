#!/usr/bin/env python3
"""Log ALL policy outputs during an inference dry-run, to CSV. READ-ONLY (subscribe
only; never commands the robot).

Subscribes:
  /rl/target_pos    (12 target joint angles — the policy's output)
  /rl/last_action   (12 raw actions)
  /rl/est_lin_vel   (3 estimator body velocity, m/s)
  /rl/depth_latent  (32 scan latent; L00-flatstep student has no yaw_pred tail —
                     the yaw_pred0/1 CSV columns are logged as NaN)
Writes one CSV row per /rl/target_pos at ~50 Hz.

  python deploy_mac/analysis/log_policy.py --out /tmp/policy_log.csv [--secs 0(forever)]
"""
import argparse
import time

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

JOINTS = ["FR_hip", "FR_thigh", "FR_calf", "FL_hip", "FL_thigh", "FL_calf",
          "RR_hip", "RR_thigh", "RR_calf", "RL_hip", "RL_thigh", "RL_calf"]


class PolicyLogger(Node):
    def __init__(self, out):
        super().__init__("log_policy")
        self.tp = self.la = self.ev = self.dl = None
        self.create_subscription(Float32MultiArray, "/rl/target_pos", lambda m: setattr(self, "tp", list(m.data)), 10)
        self.create_subscription(Float32MultiArray, "/rl/last_action", lambda m: setattr(self, "la", list(m.data)), 10)
        self.create_subscription(Float32MultiArray, "/rl/est_lin_vel", lambda m: setattr(self, "ev", list(m.data)), 10)
        self.create_subscription(Float32MultiArray, "/rl/depth_latent", lambda m: setattr(self, "dl", list(m.data)), 10)
        self.f = open(out, "w")
        cols = (["t"] + [f"target_{j}" for j in JOINTS] + [f"action_{i}" for i in range(12)]
                + ["est_vx", "est_vy", "est_vz", "dlat_norm", "yaw_pred0", "yaw_pred1"])
        self.f.write(",".join(cols) + "\n")
        self.n = 0
        self.t0 = time.time()
        self.create_timer(0.02, self._log)   # 50 Hz

    def _log(self):
        if self.tp is None or len(self.tp) < 12:
            return
        nan12 = [float("nan")] * 12
        la = self.la[:12] if self.la and len(self.la) >= 12 else nan12
        ev = self.ev[:3] if self.ev and len(self.ev) >= 3 else [float("nan")] * 3
        dl = self.dl or []
        dlat_norm = float(np.linalg.norm(dl[:32])) if len(dl) >= 32 else float("nan")
        yaw = dl[32:34] if len(dl) >= 34 else [float("nan"), float("nan")]
        row = [time.time() - self.t0] + self.tp[:12] + la + ev + [dlat_norm] + list(yaw)
        self.f.write(",".join(f"{x:.5f}" for x in row) + "\n")
        self.n += 1
        if self.n % 50 == 0:
            self.f.flush()
            self.get_logger().info(
                f"{self.n} rows | target[0:3]={[round(x,3) for x in self.tp[:3]]} "
                f"est_vx={ev[0]:+.2f} |dlat|={dlat_norm:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/policy_log.csv")
    ap.add_argument("--secs", type=float, default=0.0, help="0 = until Ctrl-C")
    args = ap.parse_args()
    rclpy.init()
    node = PolicyLogger(args.out)
    print(f"logging policy outputs -> {args.out} (waiting for /rl/target_pos)...")
    try:
        if args.secs > 0:
            end = time.time() + args.secs
            while time.time() < end:
                rclpy.spin_once(node, timeout_sec=0.1)
        else:
            rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.f.flush(); node.f.close()
        print(f"wrote {node.n} rows -> {args.out}")
        node.destroy_node(); rclpy.shutdown()


if __name__ == "__main__":
    main()
