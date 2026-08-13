#!/usr/bin/env bash
# Inference DRY-RUN (no camera, NO motor commands):
# run rl_distill_policy on live /lowstate with a ZERO depth latent and confirm it
# outputs sane /rl/target_pos. Nothing drives /lowcmd, so the robot stays put.
#
#   ./deploy_mac/run/run_inference_dryrun.sh
#
# Override: EP_ROOT, CKPT.
# (no `set -u`: the RoboStack conda activation scripts reference unset vars)
HERE="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$HERE"
source deploy_mac/env.sh

EP="${EP_ROOT:-$(cd "$HERE/.." && pwd)/go2-flyball-parkour}"
CKPT="${CKPT:-models/L00-02-flat-baseline-delay-distillation-noised/model_14500.pt}"
PROPRIO_LATENCY_S="${PROPRIO_LATENCY_S:-0.0}"
# stubs first (torchvision), then rsl_rl + legged_gym, then the node's scripts dir
export PYTHONPATH="$HERE/deploy_mac/stubs:$EP/rsl_rl:$EP/legged_gym:$HERE/src/deploy_rl_policy/scripts:${PYTHONPATH:-}"

cleanup() { kill "${GATE:-}" "${POL:-}" 2>/dev/null; ./deploy_mac/kill_ros.sh >/dev/null 2>&1; }
trap cleanup EXIT INT TERM

echo "== ckpt: $CKPT"
echo "== ep_root: $EP"
echo "== [1/3] zero depth-latent + engage gate =="
python deploy_mac/camera/fake_depth_latent.py >/tmp/gate.log 2>&1 & GATE=$!
echo "== [2/3] rl_distill_policy (logs -> /tmp/rl_policy.log) =="
python src/deploy_rl_policy/scripts/rl_distill_policy.py \
  --ckpt "$CKPT" --ep_root "$EP" --env_class_eq_17 1 --contact_threshold "${CONTACT_THRESHOLD:-20,20,20,20}" \
  --proprio_latency_s "$PROPRIO_LATENCY_S" \
  >/tmp/rl_policy.log 2>&1 & POL=$!

echo "== [3/3] reading /rl/target_pos (waits up to 30s for model load + first publish) =="
python deploy_mac/analysis/read_target_pos.py 30
RC=$?

echo "--- rl_policy.log (tail) ---"
tail -20 /tmp/rl_policy.log
exit $RC
