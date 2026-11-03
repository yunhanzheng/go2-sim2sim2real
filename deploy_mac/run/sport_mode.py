#!/usr/bin/env python3
"""Release or re-select the Go2's factory (sport) controller.

Why this exists: sport mode and `/lowcmd` cannot both own the motors. While it is
active the factory controller answers the remote itself, and every command this
repo sends is ignored or fought. The runbook used to say "run
~/go2/disable_sport.py", which needs `unitree_sdk2_python` plus a venv — neither
is installed on the Ubuntu host, so the step was quietly unexecutable. The
switch is only a request/response pair on `/api/motion_switcher/request` +
`/response` (the same topics unitree_ros2's MotionSwitcherClient uses), so this
speaks it with rclpy and nothing else.

    sport_mode.py --check        query only — publishes a read request, moves nothing
    sport_mode.py --off          release sport mode so /lowcmd can own the motors
    sport_mode.py --on NAME      select a factory mode again (name from --check)

`--off` REFUSES while the legs still look like the stand pose. Releasing drops
the motors for the moment before `/lowcmd` takes over, so a standing robot falls;
lay it down with the factory remote first (runbook §2). `--force` overrides.
The pose test reads the four calves, because this robot's own poses differ there
and nowhere else: lay is -2.72, stand is -1.5.

Exit 0 = the request was accepted (code 0); 1 = the robot answered with an error;
2 = cannot talk to the switcher at all.
"""
import argparse
import json
import os
import sys
import time


def _prescan(key, env_name):
    """--iface must be in the environment before rclpy (and DDS) come up."""
    argv = sys.argv[1:]
    for i, a in enumerate(argv):
        if a == f"--{key}" and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith(f"--{key}="):
            return a.split("=", 1)[1]
    return os.environ.get(env_name)


_IFACE = _prescan("iface", "GO2_IFACE")
if _IFACE and not os.environ.get("CYCLONEDDS_URI"):
    os.environ["CYCLONEDDS_URI"] = (
        f"<CycloneDDS><Domain><General><Interfaces>"
        f"<NetworkInterface name=\"{_IFACE}\" priority=\"default\" "
        f"multicast=\"default\"/></Interfaces></General></Domain></CycloneDDS>")

try:
    import rclpy
    from rclpy.node import Node
    from unitree_api.msg import Request, Response
    from unitree_go.msg import LowState
except ImportError as exc:                      # no ROS env, or ws not sourced
    print(f"sport_mode: cannot run here ({exc}); source /opt/ros/humble and "
          f"the unitree_ros2 workspace first")
    raise SystemExit(2)

API_CHECK_MODE = 1001
API_SELECT_MODE = 1002
API_RELEASE_MODE = 1003
REQ_TOPIC = "/api/motion_switcher/request"
RES_TOPIC = "/api/motion_switcher/response"
CALVERS = (2, 5, 8, 11)     # FR, FL, RR, RL calf joints in MotorState[20]
STAND_CALF = -2.1           # lay pose is -2.72, stand pose is -1.5


class Switcher(Node):
    def __init__(self):
        super().__init__("sport_mode")
        self.pub = self.create_publisher(Request, REQ_TOPIC, 10)
        self.resp = None
        self.create_subscription(Response, RES_TOPIC, self._on_resp, 20)
        self.lowstate = None
        self.create_subscription(LowState, "/lowstate", self._on_lowstate, 10)

    def _on_resp(self, m):
        self.resp = m

    def _on_lowstate(self, m):
        self.lowstate = m

    def spin(self, seconds):
        end = time.time() + seconds
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.05)

    def wait_peer(self, seconds=5.0):
        """The switcher only answers once both endpoints are discovered; without
        this the first request is published into the void and looks like a dead
        robot."""
        end = time.time() + seconds
        while time.time() < end:
            if (self.count_subscribers(REQ_TOPIC)
                    and self.count_publishers(RES_TOPIC)):
                return True
            self.spin(0.1)
        return False

    def call(self, api_id, parameter="", timeout=5.0):
        req = Request()
        req.header.identity.id = int(time.monotonic_ns())
        req.header.identity.api_id = int(api_id)
        req.parameter = parameter
        self.resp = None
        self.pub.publish(req)
        end = time.time() + timeout
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.05)
            if (self.resp is not None
                    and self.resp.header.identity.id == req.header.identity.id):
                return self.resp
        return None

    def pose(self):
        """Mean calf angle, or None if /lowstate never arrived."""
        self.spin(0.5)
        if self.lowstate is None:
            return None
        q = self.lowstate.motor_state
        return sum(float(q[i].q) for i in CALVERS) / len(CALVERS)


def describe(resp):
    if resp is None:
        return "no answer (timeout)"
    code = int(resp.header.status.code)
    if code != 0:
        return f"refused, code={code}"
    if not resp.data:
        return "ok, code=0 (empty answer)"
    try:
        return f"ok, code=0 {json.dumps(json.loads(resp.data), sort_keys=True)}"
    except ValueError:
        return f"ok, code=0 raw={resp.data!r}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true",
                   help="query the active mode (read-only)")
    g.add_argument("--off", action="store_true", help="release sport mode")
    g.add_argument("--on", metavar="NAME", help="select a factory mode by name")
    ap.add_argument("--force", action="store_true",
                    help="release even though the legs look like the stand pose")
    ap.add_argument("--iface", help="network interface (default: $GO2_IFACE)")
    opts = ap.parse_args()

    rclpy.init()
    node = Switcher()
    try:
        if not node.wait_peer():
            print(f"sport_mode: nothing answers on {REQ_TOPIC}. Power on, "
                  f"check the wired NIC and CYCLONEDDS_URI.")
            return 2
        print(f"iface : {_IFACE or '(CYCLONEDDS_URI as set)'}")

        calf = node.pose()
        if calf is None:
            print("pose  : /lowstate silent, cannot check the pose")
            laying = False
        else:
            laying = calf < STAND_CALF
            print(f"pose  : mean calf q = {calf:+.2f} "
                  f"({'laying' if laying else 'standing or loaded'})")

        if opts.check:
            print(f"check : {describe(node.call(API_CHECK_MODE))}")
            return 0

        if opts.off:
            if calf is not None and not laying and not opts.force:
                print("sport_mode: refusing — the legs are in the stand pose and "
                      "releasing drops them. Lay the robot down with the factory "
                      "remote first, or pass --force.")
                return 2
            print(f"release: {describe(node.call(API_RELEASE_MODE))}")
            print(f"confirm: {describe(node.call(API_CHECK_MODE))}")
            return 0

        param = json.dumps({"name": opts.on})
        print(f"select : {describe(node.call(API_SELECT_MODE, param))}")
        print(f"confirm: {describe(node.call(API_CHECK_MODE))}")
        return 0
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    sys.exit(main())
