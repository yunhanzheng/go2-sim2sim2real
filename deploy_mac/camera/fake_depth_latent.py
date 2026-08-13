#!/usr/bin/env python3
"""Publish a ZERO depth latent (+ engage flag) so rl_distill_policy runs WITHOUT a
depth camera. For the no-camera dry-run: L00-02 is `--no_depth_encoder_loss`, so a
zero 32-dim scan latent gives weak depth influence. The L00-flatstep student has
no yaw predictor head, so the latent is 32-D only (no +2 yaw_pred tail) and
heading is held forward (prop[6:8]=0) by the policy node.

Publishes:
  /rl/depth_latent  (Float32MultiArray, 32 zeros) @ 10 Hz  -> unblocks have_depth_latent
  /policy_engaged   (Bool True)                   @  5 Hz  -> unblocks the FSM gate
        (low_level_ctrl normally sets /policy_engaged; the dry-run doesn't run it, so
         we set it here. Safe: nothing drives /lowcmd, so the robot does NOT move.
         Pass --no-engage to suppress, e.g. if you run low_level_ctrl yourself.)

These are internal ROS topics; the Go2 firmware does not consume them.
"""
import argparse
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Bool

DEPTH_LATENT_LEN = 32  # scan latent only (L00-flatstep: yaw predictor head removed)


class Gate(Node):
    def __init__(self, engage):
        super().__init__("fake_depth_latent")
        self.dl_pub = self.create_publisher(Float32MultiArray, "rl/depth_latent", 10)
        self.create_timer(0.1, self._pub_latent)            # 10 Hz
        self.eng_pub = None
        if engage:
            self.eng_pub = self.create_publisher(Bool, "policy_engaged", 10)
            self.create_timer(0.2, self._pub_engage)        # 5 Hz

    def _pub_latent(self):
        m = Float32MultiArray()
        m.data = [0.0] * DEPTH_LATENT_LEN
        self.dl_pub.publish(m)

    def _pub_engage(self):
        b = Bool()
        b.data = True
        self.eng_pub.publish(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-engage", action="store_true",
                    help="do not publish /policy_engaged (use if running low_level_ctrl)")
    args = ap.parse_args()
    rclpy.init()
    n = Gate(engage=not args.no_engage)
    extra = "" if args.no_engage else " + /policy_engaged=True @5Hz"
    print(f"fake_depth_latent: publishing zero /rl/depth_latent @10Hz{extra} (Ctrl-C to stop)")
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        n.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
