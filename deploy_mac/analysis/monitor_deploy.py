#!/usr/bin/env python3
"""Live deploy monitor — prints the important state to the terminal and logs /lowstate
to CSV. READ-ONLY (subscribe only; never commands the robot).

Prints ~2 Hz:  FSM state (LAYING/STANDING/MOVING), /policy_engaged, whether /lowcmd is
flowing + its kp (50=bring-up/transition, 30=policy), pose distances, a couple joints.
Logs joint q/dq + engaged + cmd_kp to --csv at ~50 Hz.

  python deploy_mac/analysis/monitor_deploy.py --csv run_logs/real/<ts>/lowstate.csv
"""
import argparse
import math
import time

import rclpy
from rclpy.node import Node
from unitree_go.msg import LowState, LowCmd
from std_msgs.msg import Bool

JOINTS = ["FR_hip", "FR_thigh", "FR_calf", "FL_hip", "FL_thigh", "FL_calf",
          "RR_hip", "RR_thigh", "RR_calf", "RL_hip", "RL_thigh", "RL_calf"]
LAY = [-0.1, 1.26, -2.81, 0.1, 1.25, -2.80, -0.1, 1.29, -2.82, 0.1, 1.28, -2.81]
STAND = [-0.1, 0.8, -1.5, 0.1, 0.8, -1.5, -0.1, 1.0, -1.5, 0.1, 1.0, -1.5]


def dist(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


class Monitor(Node):
    def __init__(self, csv):
        super().__init__("monitor_deploy")
        self.q = [0.0] * 12
        self.dq = [0.0] * 12
        self.have = False
        self.engaged = False
        self.cmd_kp = None
        self.cmd_n = 0
        self.cmd_t0 = None
        self.cmd_last = 0.0
        self.create_subscription(LowState, "/lowstate", self._ls, 10)
        self.create_subscription(LowCmd, "/lowcmd", self._lc, 10)
        self.create_subscription(Bool, "/policy_engaged", self._eng, 10)
        self.f = open(csv, "w") if csv else None
        if self.f:
            self.f.write("t," + ",".join("q_" + j for j in JOINTS) + ","
                         + ",".join("dq_" + j for j in JOINTS) + ",engaged,cmd_kp\n")
        self.t0 = time.time()
        self.create_timer(0.02, self._logrow)   # 50 Hz CSV
        self.create_timer(0.5, self._print)      # 2 Hz print

    def _ls(self, m):
        for i in range(12):
            self.q[i] = m.motor_state[i].q
            self.dq[i] = m.motor_state[i].dq
        self.have = True

    def _lc(self, m):
        self.cmd_kp = float(m.motor_cmd[0].kp)
        self.cmd_n += 1
        self.cmd_last = time.time()
        if self.cmd_t0 is None:
            self.cmd_t0 = time.time()

    def _eng(self, m):
        self.engaged = bool(m.data)

    def _state(self):
        if not self.have:
            return "?(no /lowstate)"
        if math.sqrt(sum(x * x for x in self.dq)) > 0.5:
            return "MOVING"
        dl, ds = dist(self.q, LAY), dist(self.q, STAND)
        if ds < 0.85:
            return "STANDING"
        if dl < 0.85:
            return "LAYING"
        return "BETWEEN"

    def _logrow(self):
        if not self.have or not self.f:
            return
        t = time.time() - self.t0
        kp = f"{self.cmd_kp:.1f}" if self.cmd_kp is not None else "nan"
        self.f.write(f"{t:.4f}," + ",".join(f"{x:.4f}" for x in self.q) + ","
                     + ",".join(f"{x:.4f}" for x in self.dq) + f",{int(self.engaged)},{kp}\n")

    def _print(self):
        t = time.time() - self.t0
        cmd_alive = (time.time() - self.cmd_last) < 0.3
        rate = self.cmd_n / max(time.time() - (self.cmd_t0 or time.time()), 1e-9) if self.cmd_t0 else 0.0
        dl = dist(self.q, LAY) if self.have else float("nan")
        ds = dist(self.q, STAND) if self.have else float("nan")
        kp = f"{self.cmd_kp:.0f}" if self.cmd_kp is not None else "--"
        print(f"[t={t:5.1f}] state={self._state():14s} engaged={str(self.engaged):5s} "
              f"/lowcmd={'YES' if cmd_alive else 'NO ':3s}(~{rate:3.0f}Hz kp={kp})  "
              f"lay_d={dl:4.2f} stand_d={ds:4.2f}  FR_thigh={self.q[1]:+.2f} FR_calf={self.q[2]:+.2f}",
              flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()
    rclpy.init()
    node = Monitor(args.csv)
    print("monitor: state | engaged | /lowcmd(kp) | pose dists | joints   (READ-ONLY)", flush=True)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.f:
            node.f.flush(); node.f.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
