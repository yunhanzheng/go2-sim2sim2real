#!/usr/bin/env bash
# On-board dry-run: camera, preprocessing and policy ALL on the Jetson itself.
# No /lowcmd is published, so the robot cannot move from this stack.
#
# Why this exists: when the host drove the policy, every /lowcmd crossed the
# Ethernet link, and that link is the least reliable part of the loop. Running
# here removes it — the camera is on this board, the policy is on this board, and
# only /lowcmd leaves, over the internal NIC (enP8p1s0) to the robot MCU.
#
# The question this answers is NOT "is the chain wired up" (it is) but "can the
# Orin NX hold 50 Hz with the 80 ms depth budget". Read the numbers at the end:
#   depth_age   must sit in [80, 100) ms   <- the trained band
#   tick overrun  must never appear
#   early         may fire once at engage, must NOT keep climbing
#
#   ~/go2_deploy/run_dryrun_onboard.sh
#   SECS=60 ~/go2_deploy/run_dryrun_onboard.sh
#
# !!! DO NOT TOUCH THE REMOTE STICKS !!! Sport mode may still be armed and the
# factory controller reads the same remote.
# (no `set -u`: ROS setup scripts reference unset vars)
set +o pipefail

source "$HOME/go2_deploy/env.sh"

ROOT="$HOME/go2_deploy"
SCRIPTS="$ROOT/ws/src/deploy_rl_policy/scripts"
CKPT="${CKPT:-$ROOT/models/D31-fovscan_maxpool_fullforce_fx_2m-smoothfloor-l40s/model_9000.pt}"
SECS="${SECS:-45}"
CT="${CONTACT_THRESHOLD:-20,20,20,20}"
ENV17="${ENV_CLASS_EQ_17:-0}"
LV="${LIN_VEL_MAX:-1.0}"
AD="${ACTION_DELAY:-1}"
DEPTH_LATENCY_S="${DEPTH_LATENCY_S:-0.08}"
PROPRIO_LATENCY_S="${PROPRIO_LATENCY_S:-0.016}"

[ -f "$CKPT" ] || { echo "ERR: ckpt not found: $CKPT"; exit 1; }
[ -d "$SCRIPTS" ] || { echo "ERR: scripts not found: $SCRIPTS"; exit 1; }

export PYTHONPATH="$ROOT/ep/rsl_rl:$SCRIPTS:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
# Measured 2026-10-30 (A/B on the robot): limiting the thread pools is the WRONG
# lever here. With torch's default 8 threads the encoder tick was p50 9.89 /
# p95 18.42 / max 46.78 ms and 3.7% of ticks exceeded 20 ms. With OMP=2 the max
# dropped to 16.90 ms but p50 rose to 12.94 and 7.6% of ticks overran — and,
# because depth_camera_node inherits the same cap, its 480x270 -> 58x87 resample
# slowed enough to push depth_age to 149 ms with 100 frames out of band. The
# conv stack genuinely uses the cores. Leave the pools at their defaults; if the
# encoder has to get faster the answer is TensorRT/FP16, not fewer threads.
# (Export OMP_NUM_THREADS yourself if you ever want to re-run that A/B.)
# Same-machine stamps: the camera node and the policy share one clock, so the
# offset the host had to correct (42 ms) is structurally zero here.
RUN="$ROOT/run_logs/dryrun_onboard/$(date +%Y-%m-%d_%H-%M-%S)"
mkdir -p "$RUN"

wait_topic() {
  local t="$1" i list
  for i in $(seq 1 40); do
    list="$(ros2 topic list 2>/dev/null)"
    if grep -qx -- "$t" <<< "$list"; then return 0; fi
    sleep 0.5
  done
  return 1
}
wait_topic "/camera/depth/image_rect_raw" || { echo "ERR: no depth topic after 20 s"; exit 1; }
wait_topic "/lowstate"                  || { echo "ERR: no /lowstate after 20 s"; exit 1; }

echo "== ckpt  : $CKPT"
echo "== config: env_class_eq_17=$ENV17 action_delay=$AD lin_vel_max=$LV ct=$CT"
echo "== logs  : $RUN/"
echo "== NOTE  : no /lowcmd. Keep hands off the remote."

DC=""; P=""; G=""
cleanup() {
  kill "$P" "$DC" "$G" 2>/dev/null
  wait 2>/dev/null
  echo "== logs -> $RUN/"
}
trap cleanup EXIT INT TERM

python3 "$SCRIPTS/depth_camera_node.py" \
  --ros-args -p input_topic:=/camera/depth/image_rect_raw \
  -p target_hz:=0.0 -p track_clock_offset:=true > "$RUN/depth_camera.log" 2>&1 & DC=$!

python3 "$SCRIPTS/distill_fused_policy.py" \
  --ckpt "$CKPT" --ep_root "$ROOT/ep" --env_class_eq_17 "$ENV17" \
  --action_delay "$AD" --lin_vel_max "$LV" --binary_vel 1 --vel_deadzone 0.2 \
  --contact_threshold "$CT" \
  --depth_latency_s "$DEPTH_LATENCY_S" --proprio_latency_s "$PROPRIO_LATENCY_S" \
  --reset_on_engage 1 --clip_applied_action 1 \
  --latency_log "$RUN/latency_actor.tsv" > "$RUN/rl_policy.log" 2>&1 & P=$!

ros2 topic pub -r 5 /policy_engaged std_msgs/msg/Bool "{data: true}" \
  > "$RUN/engage.log" 2>&1 & G=$!

echo "== waiting up to 90 s for the first [cmd] line (torch load on Orin is slow)..."
for i in $(seq 1 90); do
  grep -q "\[cmd\]" "$RUN/rl_policy.log" 2>/dev/null && break
  sleep 1
done
if ! grep -q "\[cmd\]" "$RUN/rl_policy.log" 2>/dev/null; then
  echo "RESULT: FAIL — no [cmd] line in 90 s"
  echo "--- rl_policy.log ---"; tail -20 "$RUN/rl_policy.log"
  exit 1
fi

echo "== policy is ticking; observing for ${SECS}s"
sleep "$SECS"

echo ""
echo "=== depth age (last 10 samples) ==="
grep -o "depth_age=[0-9]*ms" "$RUN/rl_policy.log" | tail -10 | tr '\n' ' '; echo
echo "=== overruns / late pads (must be 0) ==="
grep -c "tick overrun" "$RUN/rl_policy.log" || true
echo "  tick overrun lines ^"
grep -c "proprio pad not met" "$RUN/rl_policy.log" || true
echo "  proprio-late lines ^"
echo "=== early (1 at engage is normal, climbing is not) ==="
grep -o "early=[0-9]*" "$RUN/rl_policy.log" | tail -5 | tr '\n' ' '; echo
echo ""
echo "--- rl_policy.log tail ---"; tail -6 "$RUN/rl_policy.log"
echo "--- depth_camera.log tail ---"; tail -4 "$RUN/depth_camera.log"
