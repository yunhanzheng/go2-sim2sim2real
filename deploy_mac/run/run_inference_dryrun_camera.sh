#!/usr/bin/env bash
# Inference DRY-RUN *with the real depth camera*. NO motor commands:
# low_level_ctrl is NOT started, so nothing drives /lowcmd and the robot stays put.
#
# Requires the Jetson to be publishing /camera/depth/image_rect_raw (realsense2_camera)
# and the robot board publishing /lowstate.
#
#   ./deploy_mac/run/run_inference_dryrun_camera.sh
#   DURATION=300 ./deploy_mac/run/run_inference_dryrun_camera.sh   # stay up 5 min
#   DURATION=0   ./deploy_mac/run/run_inference_dryrun_camera.sh   # stay up until Ctrl-C
#
# Chain: camera -> depth_camera_node -> /rl/depth_image -> depth_encoder_node ->
#        /rl/depth_latent  +  /lowstate  -> rl_distill_policy -> /rl/target_pos
# We publish /policy_engaged=true ourselves (low_level_ctrl normally does) to unblock
# the encoder + policy. Override: EP_ROOT, CKPT, CONTACT_THRESHOLD.
# (no `set -u`: conda activation scripts reference unset vars)
HERE="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$HERE"
source deploy_mac/env.sh

EP="${EP_ROOT:-$(cd "$HERE/.." && pwd)/go2-flyball-parkour}"
CKPT="${CKPT:-models/T000-00-step-corridor-distill/model_900.pt}"
CT="${CONTACT_THRESHOLD:-20,20,20,20}" # real raw sensor units, FR,FL,RR,RL
DEPTH_LATENCY_S="${DEPTH_LATENCY_S:-0.0}"
PROPRIO_LATENCY_S="${PROPRIO_LATENCY_S:-0.0}"
DURATION="${DURATION:-120}"  # seconds to stay engaged after PASS; 0 = until Ctrl-C
export PYTHONPATH="$HERE/deploy_mac/stubs:$EP/rsl_rl:$EP/legged_gym:$HERE/src/deploy_rl_policy/scripts:${PYTHONPATH:-}"

cleanup() { kill "${D:-}" "${E:-}" "${P:-}" "${G:-}" 2>/dev/null; ./deploy_mac/kill_ros.sh >/dev/null 2>&1; }
trap cleanup EXIT INT TERM

echo "== ckpt: $CKPT | ep: $EP | contact_threshold: $CT | synthetic latency: depth=${DEPTH_LATENCY_S}s proprio=${PROPRIO_LATENCY_S}s"
ros2 topic list 2>/dev/null | grep -q image_rect_raw \
  || { echo "ERR: /camera/depth/image_rect_raw not found — launch realsense2_camera on the Jetson"; exit 1; }

echo "== [1/4] depth_camera_node  (/camera/depth/image_rect_raw -> /rl/depth_image)"
python src/deploy_rl_policy/scripts/depth_camera_node.py \
  --ros-args -p input_topic:=/camera/depth/image_rect_raw >/tmp/depthcam.log 2>&1 & D=$!
echo "== [2/4] depth_encoder_node (-> /rl/depth_latent)  logs:/tmp/depthenc.log"
python src/deploy_rl_policy/scripts/depth_encoder_node.py \
  --ckpt "$CKPT" --ep_root "$EP" --contact_threshold "$CT" --env_class_eq_17 1 \
  --enforce_latency_s "$DEPTH_LATENCY_S" >/tmp/depthenc.log 2>&1 & E=$!
echo "== [3/4] rl_distill_policy  (-> /rl/target_pos)     logs:/tmp/rlpolicy.log"
python src/deploy_rl_policy/scripts/rl_distill_policy.py \
  --ckpt "$CKPT" --ep_root "$EP" --env_class_eq_17 1 --contact_threshold "$CT" \
  --proprio_latency_s "$PROPRIO_LATENCY_S" >/tmp/rlpolicy.log 2>&1 & P=$!
echo "== [4/4] /policy_engaged=true  (we stand in for low_level_ctrl; NO /lowcmd is sent)"
ros2 topic pub -r 5 /policy_engaged std_msgs/msg/Bool "{data: true}" >/tmp/engage.log 2>&1 & G=$!

echo "== waiting for /rl/target_pos (model load on 2 nodes + chain; up to 40s)..."
python deploy_mac/analysis/read_target_pos.py 40
RC=$?

echo "== depth latent sample (norm should be NON-zero => real camera feeding the policy):"
perl -e 'alarm 6; exec @ARGV' ros2 topic echo --once /rl/depth_latent 2>&1 | head -4
echo "--- depth_encoder log tail ---"; tail -4 /tmp/depthenc.log
echo "--- rl_policy log tail ---";    tail -4 /tmp/rlpolicy.log

if [ "$RC" -eq 0 ]; then
  if [ "${DURATION}" -eq 0 ] 2>/dev/null; then
    echo "== DURATION=0 — running until Ctrl-C (Ctrl-C to stop) =="
    wait
  else
    echo "== staying engaged for ${DURATION}s — Ctrl-C to stop early =="
    sleep "${DURATION}"
  fi
fi
exit $RC
