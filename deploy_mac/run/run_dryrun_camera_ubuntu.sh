#!/usr/bin/env bash
# Ubuntu dry-run with the REAL depth camera. Starts NO low_level_ctrl, so nothing
# publishes /lowcmd and the robot cannot move from this stack.
#
# This is the Linux counterpart of run_inference_dryrun_camera.sh, which is Mac
# only (it sources deploy_mac/env.sh = RoboStack/conda/en13). Here we use the
# system ROS that real_run_runbook_ubuntu.md §3 sets up.
#
# What it proves, in order:
#   /camera/depth/image_rect_raw  ->  depth_camera_node  ->  /rl/depth_image
#   /rl/depth_image + /lowstate   ->  distill_fused_policy -> /rl/target_pos
# The last one is the number that matters: if the policy is being gated on
# depth (the "waiting for the actor's first step" failure), target_pos stays
# empty and the depth latent stays zero.
#
# !!! DO NOT TOUCH THE REMOTE'S STICKS WHILE THIS RUNS !!!
# Sport mode is still on. The factory controller reads the same remote, so a
# pushed stick moves the robot for real even though we publish no /lowcmd. That
# is also why this script deliberately starts no /joy source: the policy runs on
# a zero velocity command, which is what we want to observe anyway.
#
#   ./deploy_mac/run/run_dryrun_camera_ubuntu.sh
#   SECS=60 ./deploy_mac/run/run_dryrun_camera_ubuntu.sh
#   CKPT=models/L00-02-flat-baseline-delay-distillation-noised/model_14500.pt \
#     ENV_CLASS_EQ_17=1 ./deploy_mac/run/run_dryrun_camera_ubuntu.sh
#
# Logs: run_logs/dryrun/<ckptdir>__<model>/<timestamp>/
# (no `set -u`: ROS setup scripts reference unset vars)
# No `set -o pipefail` either: `ros2 topic list | grep -q` exits as soon as grep
# matches, grep closes the pipe, ros2 dies on SIGPIPE and pipefail then reports
# the whole pipeline as failed even though the topic WAS found (measured rc=120).
set +o pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

CKPT="${CKPT:-models/D31-fovscan_maxpool_fullforce_fx_2m-smoothfloor-l40s/model_9000.pt}"
EP="${EP_ROOT:-$(cd "$REPO/.." && pwd)/go2-flyball-parkour}"
SECS="${SECS:-45}"
CT="${CONTACT_THRESHOLD:-20,20,20,20}"   # real raw foot-force units
ENV17="${ENV_CLASS_EQ_17:-0}"            # 0 = parkour (D31/D08/D09), 1 = flat (T000/L00)
LV="${LIN_VEL_MAX:-1.0}"
AD="${ACTION_DELAY:-1}"
DEPTH_LATENCY_S="${DEPTH_LATENCY_S:-0.08}"
PROPRIO_LATENCY_S="${PROPRIO_LATENCY_S:-0.016}"

[ -f "$CKPT" ] || { echo "ERR: ckpt not found: $CKPT"; exit 1; }

# -- environment: system ROS, /usr/bin/python3 (conda's 3.13 cannot import rclpy)
conda deactivate 2>/dev/null; conda deactivate 2>/dev/null
export PATH="/usr/bin:$PATH"
source /opt/ros/humble/setup.bash
source "$HOME/go2/ros2_ws/install/setup.bash"
source "$REPO/install/setup.bash"
export EP_ROOT="$EP"
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=0
export GO2_IFACE="${GO2_IFACE:-eno1}"
export ROS_LOG_DIR="$REPO/.tmp_roslog"
export PYTHONUNBUFFERED=1     # node logs must reach $RUN/*.log even if we kill them
export PYTHONPATH="$EP/rsl_rl:$EP/legged_gym:$REPO/src/deploy_rl_policy/scripts:${PYTHONPATH:-}"

POLICY="$(basename "$(dirname "$CKPT")")__$(basename "$CKPT" .pt)"
RUN="$REPO/run_logs/dryrun/$POLICY/$(date +%Y-%m-%d_%H-%M-%S)"
mkdir -p "$RUN" "$ROS_LOG_DIR"

# -- preflight: the two inputs this run cannot fake. Wait for DDS discovery: a
# fresh process needs a few seconds before it can see the robot's topics.
wait_topic() {
  local t="$1" i list
  for i in $(seq 1 40); do
    list="$(ros2 topic list 2>/dev/null)"
    if grep -qx -- "$t" <<< "$list"; then return 0; fi
    sleep 0.5
  done
  return 1
}

wait_topic "/camera/depth/image_rect_raw" \
  || { echo "ERR: /camera/depth/image_rect_raw is missing after 20 s — the Jetson camera service is down"; exit 1; }
wait_topic "/lowstate" \
  || { echo "ERR: /lowstate is missing after 20 s — is the host on the robot's network ($GO2_IFACE)?"; exit 1; }

echo "== ckpt    : $CKPT"
echo "== config  : env_class_eq_17=$ENV17 action_delay=$AD binary_vel lin_vel_max=$LV ct=$CT"
echo "== latency : depth=${DEPTH_LATENCY_S}s proprio=${PROPRIO_LATENCY_S}s"
echo "== logs    : $RUN/"
echo "== NOTE    : no /lowcmd is published. Do NOT push the remote sticks — sport"
echo "==           mode is still on and the factory controller would drive the robot."

DC=""; P=""; G=""
cleanup() {
  kill "$P" "$DC" "$G" 2>/dev/null
  wait 2>/dev/null
  echo "== logs -> $RUN/"
}
trap cleanup EXIT INT TERM

# target_hz:=0.0  -> pass every 60 Hz frame; a 10 Hz feed makes the age gate beat
#                    against the encode tick (measured 28% vs 60% in-band).
# track_clock_offset:=true -> the stamps are in the Jetson clock domain; without
#                    this the gate ages frames by receipt and distrusts every one.
/usr/bin/python3 src/deploy_rl_policy/scripts/depth_camera_node.py \
  --ros-args \
  -p input_topic:=/camera/depth/image_rect_raw \
  -p target_hz:=0.0 \
  -p track_clock_offset:=true > "$RUN/depth_camera.log" 2>&1 & DC=$!

/usr/bin/python3 src/deploy_rl_policy/scripts/distill_fused_policy.py \
  --ckpt "$CKPT" --ep_root "$EP" --env_class_eq_17 "$ENV17" \
  --action_delay "$AD" --lin_vel_max "$LV" --binary_vel 1 --vel_deadzone 0.2 \
  --contact_threshold "$CT" \
  --depth_latency_s "$DEPTH_LATENCY_S" --proprio_latency_s "$PROPRIO_LATENCY_S" \
  --reset_on_engage 1 --clip_applied_action 1 \
  --latency_log "$RUN/latency_actor.tsv" > "$RUN/rl_policy.log" 2>&1 & P=$!

ros2 topic pub -r 5 /policy_engaged std_msgs/msg/Bool "{data: true}" \
  > "$RUN/engage.log" 2>&1 & G=$!

echo "== waiting up to 60 s for /rl/target_pos (model load + chain)..."
/usr/bin/python3 - "$RUN" "$SECS" <<'PY'
import statistics
import sys
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from unitree_go.msg import LowState

run, secs = sys.argv[1], float(sys.argv[2])


class Probe(Node):
    def __init__(self):
        super().__init__("dryrun_probe")
        self.img = self.lat = self.tgt = None
        self.n_img = self.n_lat = self.n_tgt = 0
        self.q = None
        self.dev = []
        self.create_subscription(Image, "/rl/depth_image", self.on_img, 10)
        self.create_subscription(Float32MultiArray, "/rl/depth_latent", self.on_lat, 10)
        self.create_subscription(Float32MultiArray, "/rl/target_pos", self.on_tgt, 20)
        self.create_subscription(LowState, "/lowstate", self.on_low, 50)

    def on_img(self, m):
        self.n_img += 1
        self.img = m

    def on_lat(self, m):
        self.n_lat += 1
        self.lat = list(m.data)

    def on_low(self, m):
        self.q = [s.q for s in m.motor_state[:12]]

    def on_tgt(self, m):
        self.n_tgt += 1
        self.tgt = list(m.data)
        if self.q and len(self.tgt) >= 12:
            self.dev.append(max(abs(a - b) for a, b in zip(self.tgt[:12], self.q)))


rclpy.init()
p = Probe()
end = time.time() + 60.0
while p.n_tgt == 0 and time.time() < end:
    rclpy.spin_once(p, timeout_sec=0.1)

if p.n_tgt == 0:
    print("RESULT: FAIL — no /rl/target_pos in 60 s.")
    print("        The policy never produced an action. Check rl_policy.log for a")
    print("        gate message ('waiting for ...', 'depth', 'startup_wait_s').")
    sys.exit(1)

print("== chain is up; sampling for %.0f s" % secs)
end = time.time() + secs
while time.time() < end:
    rclpy.spin_once(p, timeout_sec=0.1)

import numpy as np
ok = True
print("")
print("--- /rl/depth_image (policy's actual input) ---")
if p.img is None:
    print("  FAIL  nothing received")
    ok = False
else:
    a = np.frombuffer(bytes(p.img.data), dtype=np.float32)
    w, h = p.img.width, p.img.height
    a = a.reshape(h, w) if a.size == w * h else a
    print(f"  OK    {p.n_img} frames, {w}x{h} {p.img.encoding}")
    print(f"        range {a.min():+.3f} .. {a.max():+.3f}  (normalised ~[-0.5,0.5])")
    nz = float((a != 0).mean())
    print(f"        non-zero pixels {nz*100:.1f}%")
    if nz < 0.05:
        print("  FAIL  depth image is essentially empty")
        ok = False

print("--- /rl/depth_latent ---")
if p.lat is None:
    print("  FAIL  nothing received")
    ok = False
else:
    n = float(np.linalg.norm(p.lat))
    print(f"  OK    {p.n_lat} msgs, dim {len(p.lat)}, |latent| = {n:.3f}")
    if n < 1e-6:
        print("  FAIL  latent is all zeros — the encoder is not being fed")
        ok = False

print("--- /rl/target_pos ---")
print(f"  OK    {p.n_tgt} msgs, dim {len(p.tgt)}")
if p.q and len(p.tgt) >= 12:
    print(f"        target  : {' '.join(f'{v:+.2f}' for v in p.tgt[:12])}")
    print(f"        measured: {' '.join(f'{v:+.2f}' for v in p.q)}")
    print("        order   : FR h/t/c, FL h/t/c, RR h/t/c, RL h/t/c")
    print("        default : -0.10 +0.80 -1.50  +0.10 +0.80 -1.50  "
          "-0.10 +1.00 -1.50  +0.10 +1.00 -1.50")
if p.dev:
    print(f"        |target - measured q| max {max(p.dev):.3f} rad "
          f"(p50 {statistics.median(p.dev):.3f}, p95 {sorted(p.dev)[int(len(p.dev)*0.95)]:.3f})")
    # A LAYING robot and a stand-pose command differ by ~1.5 rad in every calf.
    # That gap is the expected result, not a fault: it is the policy asking to
    # stand up. Only a gap while ALREADY standing is suspicious.
    calf = (p.q[2] + p.q[5] + p.q[8] + p.q[11]) / 4.0
    if calf < -2.1:
        print(f"  note  robot is LYING (mean calf {calf:+.2f}) and the policy is")
        print(f"        commanding the stand pose — the gap is expected, and engaging")
        print(f"        would stand the robot.")
    elif max(p.dev) > 1.0:
        print("  WARN  target is far from the measured pose while the robot IS")
        print("        standing — the policy wants a posture it is not in.")
else:
    print("  WARN  no /lowstate overlap, cannot compare to the measured pose")

print("")
print("RESULT: " + ("PASS — depth is reaching the policy end to end" if ok
                    else "FAIL — see the lines marked FAIL above"))
sys.exit(0 if ok else 1)
PY
RC=$?

echo ""
echo "--- rl_policy.log tail ---"; tail -12 "$RUN/rl_policy.log"
echo "--- depth_camera.log tail ---"; tail -5 "$RUN/depth_camera.log"
exit $RC
