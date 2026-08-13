#!/usr/bin/env bash
# Open the live MuJoCo yaw-prediction validator. Run this in a SECOND terminal
# while an inference dry-run is already up (run_dryrun_remote.sh /
# run_inference_dryrun_camera.sh), so /lowstate, /rl/depth_latent and
# /rl/depth_image are flowing.
#
# READ-ONLY: subscribe only, nothing on /lowcmd — the robot does not move.
#
#   ./deploy_mac/diag/view_prediction.sh                 # flat scene
#   SCENE=resources/go2/scene_corridor_step_w2.xml ./deploy_mac/diag/view_prediction.sh
#   RECORD=1 ./deploy_mac/diag/view_prediction.sh        # also save MP4 -> run_logs/viewer/<ts>/
#   RECORD=/path/out.mp4 ./deploy_mac/diag/view_prediction.sh
# (no `set -u`: conda activation scripts reference unset vars)
HERE="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$HERE"
source deploy_mac/env.sh
export PYTHONPATH="$HERE/src/deploy_rl_policy/scripts:${PYTHONPATH:-}"

ARGS=()
[ -n "${SCENE:-}" ] && ARGS+=(--scene "$SCENE")
if [ -n "${RECORD:-}" ]; then
  if [ "$RECORD" = "1" ]; then ARGS+=(--record); else ARGS+=(--record "$RECORD"); fi
fi

echo "opening yaw-prediction validator (close the window or press q to stop)..."
exec python deploy_mac/diag/view_prediction.py "${ARGS[@]}" "$@"
