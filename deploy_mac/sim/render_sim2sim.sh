#!/usr/bin/env bash
# One-shot sim2sim render wrapper for the lay2stand_record FSM stack.
#
# Handles ALL the env gotchas (conda ros_humble, GO2_IFACE=lo0, MUJOCO_GL=glfw,
# KMP_DUPLICATE_LIB_OK, sourcing), makes a timestamped output dir, cleans up
# stale ROS procs before AND after, and auto-stops the stack once the recorder
# has written the mp4 (the raw launch never exits on its own).
#
# Usage:
#   deploy_mac/sim/render_sim2sim.sh [options] [-- extra ros2-launch args]
#
# Options (all optional — defaults target the T000-00 corridor family):
#   --ckpt   PATH   checkpoint .pt  (path, or bare name under models/*/)   [model_2400.pt]
#   --scene  NAME   scene xml       (path, or bare name under resources/go2) [scene_switchback_easy.xml]
#   --name   LABEL  output dir/file label                                  [auto from scene+ckpt]
#   --speed  N      constant engage speed (ignored if --profile given)      [0.5]
#   --profile STR   vel schedule "dur:spd,dur:spd,..." (overrides --speed)   [none]
#   --engage N      engage seconds (auto = sum of --profile durations)       [20]
#   --kp N --kd N   policy gains                                             [35 / 0.6]
#   --contact N     contact_threshold (sim, net contact force)              [2.0]
#   --lin_vel_max N                                                         [1.0]
#   --env17  0|1    prop[11:13] indicator: 1=flat/env-17 (T000/L00),         [1]
#                   0=parkour/wall/cliffside (env_class!=17, e.g. ODFG flyball)
#   --yaw_head 0|1  1 => EP_FEED_PRED_YAW=1: 34-D encoder (32 latent + 2 yaw),   [0]
#                   feed 1.5*yaw_pred into prop[6:8] (OD cliffside/flyball ckpts)
#   --camera NAME   mujoco camera to record from                          [follow_cam]
#                   (follow_cam=trailing 3/4, side_cam=horizontal side profile,
#                    isaac_view=angled side, front_depth_cam=onboard depth)
#   --depth_noise   apply cliffside-distill training noise to depth input [off]
#   --depth_noise_scale N  latest curriculum: 0=clean, .5=reality, 1=2x    [0]
#   --depth_cam_roll_deg N  roll front_depth_cam about optical axis (deg)    [0]
#   --return_x N    zero the velocity command when the robot comes back    [0.0]
#                   through x=N (after having been out past --return_arm_x),
#                   hold 0 for --return_hold s, then lay down. ON by default.
#   --return_arm_x N  x the robot must reach first, so a run that never       [1.0]
#                   left the start box cannot self-trigger
#   --return_hold N seconds to hold cmd 0 while still engaged               [2.0]
#   --no-return-stop  record the full engage window instead (old behaviour)
#   --bag           also `ros2 bag record` the policy + control topics into
#                   <outdir>/bag (sim names: /mujoco/lowstate, /mujoco/lowcmd,
#                   /mujoco/pos, /rl/*, /contact, /imu/data, /joy,
#                   /policy_engaged, /sim_truth/base_velocity)
#   --metrics-only  capture base-height JSON without encoding an MP4
#   --no-stop       leave the stack running after the render (for the viewer)
#
# Examples:
#   deploy_mac/sim/render_sim2sim.sh
#   deploy_mac/sim/render_sim2sim.sh --scene scene_corridor_switchback.xml --speed 0.5
#   deploy_mac/sim/render_sim2sim.sh --ckpt model_900.pt --scene scene_switchback_easy_hurdle_low.xml \
#       --profile 10:0,5:0.5,4:0,10:0.8 --name hurdle006
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EP_ROOT="$(cd "$ROOT/.." && pwd)/go2-flyball-parkour"

# ---- defaults ---------------------------------------------------------------
CKPT="models/T000-00-step-corridor-distill/model_2400.pt"
SCENE="scene_switchback_easy.xml"
NAME=""
SPEED="0.5"
PROFILE=""
ENGAGE=""
KP="35.0"; KD="0.6"
CONTACT="2.0"
LIN_VEL_MAX="1.0"
ENV17="1"   # prop[11:13] indicator: 1=flat/env-17 (T000/L00), 0=parkour/wall/cliffside (env_class!=17)
YAW_HEAD="0"  # 1 => EP_FEED_PRED_YAW=1: 34-D yaw-head encoder (OD cliffside/flyball)
CAMERA="follow_cam"
DEPTH_NOISE="0"
DEPTH_CAM_ROLL_DEG="0"
STOP_AFTER=1
METRICS_ONLY=0
RETURN_STOP="true"   # zero the cmd when the robot comes home; see --no-return-stop
RETURN_X="0.0"
RETURN_ARM_X="1.0"
RETURN_HOLD="2.0"
RECORD_BAG=0
EXTRA=()
MAX_LAUNCH_ATTEMPTS=2

# ---- parse args -------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --ckpt) CKPT="$2"; shift 2;;
    --scene) SCENE="$2"; shift 2;;
    --name) NAME="$2"; shift 2;;
    --speed) SPEED="$2"; shift 2;;
    --profile) PROFILE="$2"; shift 2;;
    --engage) ENGAGE="$2"; shift 2;;
    --kp) KP="$2"; shift 2;;
    --kd) KD="$2"; shift 2;;
    --contact) CONTACT="$2"; shift 2;;
    --lin_vel_max) LIN_VEL_MAX="$2"; shift 2;;
    --env17) ENV17="$2"; shift 2;;
    --yaw_head) YAW_HEAD="$2"; shift 2;;
    --camera) CAMERA="$2"; shift 2;;
    --depth_noise) DEPTH_NOISE="1"; shift;;
    --depth_noise_scale) DEPTH_NOISE="$2"; shift 2;;
    --depth_cam_roll_deg) DEPTH_CAM_ROLL_DEG="$2"; shift 2;;
    --metrics-only) METRICS_ONLY=1; shift;;
    --return_x) RETURN_X="$2"; shift 2;;
    --return_arm_x) RETURN_ARM_X="$2"; shift 2;;
    --return_hold) RETURN_HOLD="$2"; shift 2;;
    --no-return-stop) RETURN_STOP="false"; shift;;
    --bag) RECORD_BAG=1; shift;;
    --no-stop) STOP_AFTER=0; shift;;
    --) shift; EXTRA=("$@"); break;;
    -h|--help) sed -n '2,40p' "${BASH_SOURCE[0]}"; exit 0;;
    *) echo "unknown option: $1" >&2; exit 1;;
  esac
done

# ---- resolve ckpt / scene (allow bare names) --------------------------------
[[ "$CKPT"  != /* && ! -f "$ROOT/$CKPT" ]] && CKPT="$(ls "$ROOT"/models/*/"$CKPT" 2>/dev/null | head -1)"
[[ "$CKPT"  != /* ]] && CKPT="$ROOT/$CKPT"
[[ "$SCENE" != /* && ! -f "$ROOT/$SCENE" ]] && SCENE="$ROOT/resources/go2/$SCENE"
[[ "$SCENE" != /* ]] && SCENE="$ROOT/$SCENE"
[[ -f "$CKPT"  ]] || { echo "ckpt not found: $CKPT" >&2; exit 1; }
[[ -f "$SCENE" ]] || { echo "scene not found: $SCENE" >&2; exit 1; }

# ---- engage duration from profile if not given -----------------------------
if [[ -z "$ENGAGE" ]]; then
  if [[ -n "$PROFILE" ]]; then
    ENGAGE="$(echo "$PROFILE" | tr ',' '\n' | cut -d: -f1 | paste -sd+ - | bc)"
  else
    ENGAGE="20.0"
  fi
fi

# ---- output dir/name --------------------------------------------------------
TS="$(date +%Y-%m-%d_%H-%M-%S)"
if [[ -z "$NAME" ]]; then
  NAME="$(basename "$SCENE" .xml)_$(basename "$(dirname "$CKPT")")_$(basename "$CKPT" .pt)"
fi
SLUG="$(echo "$NAME" | tr ' /' '--')"
# Every run lands under its OWN policy folder: run_logs/<kind>/<ckptdir>__<model>/.
# One checkpoint's whole history is then one `ls`, instead of thousands of
# timestamped dirs from every policy interleaved in one directory.
POLICY="$(basename "$(dirname "$CKPT")")__$(basename "$CKPT" .pt)"
OUTDIR="$ROOT/run_logs/dryrun/${POLICY}/${TS}-${SLUG}"
mkdir -p "$OUTDIR"
OUT="$OUTDIR/${SLUG}.mp4"
LOG="$OUTDIR/launch.log"

# ---- env (NOTE: no `set -u`; conda ros hook has unbound CONDA_BUILD) ---------
source ~/miniforge3/etc/profile.d/conda.sh 2>/dev/null \
  || source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null \
  || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate ros_humble
export GO2_IFACE=lo0
source "$ROOT/deploy_mac/env.sh"
export MUJOCO_GL=glfw
export KMP_DUPLICATE_LIB_OK=TRUE
export EP_FEED_PRED_YAW="$YAW_HEAD"   # 1 => RecurrentDepthBackbone builds the 34-D yaw head
source "$ROOT/install/setup.bash"

# ---- clean slate ------------------------------------------------------------
bash "$ROOT/deploy_mac/kill_ros.sh" >/dev/null 2>&1 || true
pkill -9 -f distill_recorder >/dev/null 2>&1 || true

echo "ckpt    : $CKPT"
echo "scene   : $SCENE"
echo "flags   : env17=$ENV17 yaw_head=$YAW_HEAD depth_noise=$DEPTH_NOISE depth_cam_roll_deg=$DEPTH_CAM_ROLL_DEG contact=$CONTACT kp=$KP kd=$KD lin_vel_max=$LIN_VEL_MAX camera=$CAMERA"
echo "profile : ${PROFILE:-(constant $SPEED)}  engage=${ENGAGE}s"
if [[ "$RETURN_STOP" == "true" ]]; then
  echo "return  : cmd->0 when x comes back through $RETURN_X (armed past $RETURN_ARM_X), hold ${RETURN_HOLD}s"
else
  echo "return  : OFF (full engage window)"
fi
echo "output  : $OUT"
echo "log     : $LOG"
echo "rendering... (stack auto-stops when the mp4 is written)"

# ---- optional rosbag alongside the render -----------------------------------
# Sim topic names, not the real-robot ones: mujoco_simulator publishes
# /mujoco/lowstate and consumes /mujoco/lowcmd (low_level_ctrl is remapped onto
# those), so a bag recorded here replays against the same analysis scripts as a
# real run only after accounting for that prefix.
BAG_PID=""
if [[ "$RECORD_BAG" == "1" ]]; then
  ros2 bag record -o "$OUTDIR/bag" \
    /mujoco/lowstate /mujoco/lowcmd /mujoco/pos /mujoco/force /mujoco/torque \
    /rl/depth_image /rl/depth_latent /rl/target_pos /rl/est_lin_vel /rl/last_action \
    /contact /lift_phase /imu/data /sim_truth/base_velocity \
    /joy /policy_engaged \
    > "$OUTDIR/rosbag.log" 2>&1 &
  BAG_PID=$!
  echo "bag     : $OUTDIR/bag"
fi
stop_bag() {
  [[ -n "$BAG_PID" ]] || return 0
  kill -INT "$BAG_PID" 2>/dev/null || true
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    kill -0 "$BAG_PID" 2>/dev/null || break
    sleep 1
  done
  kill -9 "$BAG_PID" 2>/dev/null || true
  # ros2 bag needs its own SIGINT to close the mcap cleanly; kill_ros.sh does not
  # know about it, so this runs BEFORE the teardown below.
  pkill -INT -f "ros2 bag record" >/dev/null 2>&1 || true
}

start_launch() {
  local log_path="$1"
  ros2 launch deploy_rl_policy lay2stand_record.launch.py \
    ckpt:="$CKPT" ep_root:="$EP_ROOT" scene_xml:="$SCENE" output:="$OUT" \
    lin_vel_max:="$LIN_VEL_MAX" speed:="$SPEED" \
    ${PROFILE:+vel_profile:="$PROFILE"} engage_duration:="$ENGAGE" \
    policy_kp:="$KP" policy_kd:="$KD" \
    env_class_eq_17:="$ENV17" contact_threshold:="$CONTACT" run_ekf:=0 \
    depth_noise_scale:="$DEPTH_NOISE" \
    depth_cam_roll_deg:="$DEPTH_CAM_ROLL_DEG" \
    stop_on_return_x:="$RETURN_STOP" return_x:="$RETURN_X" \
    return_arm_x:="$RETURN_ARM_X" return_stop_hold:="$RETURN_HOLD" \
    camera:="$CAMERA" \
    metrics_only:="$([[ "$METRICS_ONLY" == "1" ]] && echo true || echo false)" \
    title:="$NAME" \
    "${EXTRA[@]}" > "$log_path" 2>&1 &
  LAUNCH_PID=$!
}

should_retry_launch() {
  local log_path="$1"
  grep -q "rcl node's rmw handle is invalid" "$log_path" 2>/dev/null
}

if [[ "$STOP_AFTER" == "1" ]]; then
  for attempt in $(seq 1 "$MAX_LAUNCH_ATTEMPTS"); do
    ATTEMPT_LOG="$LOG"
    if [[ "$attempt" -gt 1 ]]; then
      ATTEMPT_LOG="${LOG%.log}.attempt_${attempt}.log"
      echo "retry   : launch attempt $attempt/$MAX_LAUNCH_ATTEMPTS after RMW init failure"
    fi
    start_launch "$ATTEMPT_LOG"

    # Wait until the recorder reports the mp4 is written, then tear down.
    until { [[ "$METRICS_ONLY" == "1" ]] && grep -q "metrics-only capture done" "$ATTEMPT_LOG" 2>/dev/null; } \
       || { [[ "$METRICS_ONLY" == "0" ]] && grep -q "wrote .* frames" "$ATTEMPT_LOG" 2>/dev/null; } \
       || ! kill -0 "$LAUNCH_PID" 2>/dev/null; do
      sleep 2
    done
    stop_bag
    bash "$ROOT/deploy_mac/kill_ros.sh" >/dev/null 2>&1 || true
    pkill -9 -f distill_recorder >/dev/null 2>&1 || true

    if [[ "$METRICS_ONLY" == "1" && -f "${OUT%.mp4}.base_height.json" ]]; then
      [[ "$ATTEMPT_LOG" != "$LOG" ]] && cp "$ATTEMPT_LOG" "$LOG"
      break
    elif [[ "$METRICS_ONLY" == "0" && -f "$OUT" ]]; then
      [[ "$ATTEMPT_LOG" != "$LOG" ]] && cp "$ATTEMPT_LOG" "$LOG"
      break
    elif [[ "$attempt" -lt "$MAX_LAUNCH_ATTEMPTS" ]] && should_retry_launch "$ATTEMPT_LOG"; then
      continue
    else
      [[ "$ATTEMPT_LOG" != "$LOG" ]] && cp "$ATTEMPT_LOG" "$LOG"
      break
    fi
  done

  if [[ -n "$BAG_PID" ]]; then
    echo "bag     : $(du -sh "$OUTDIR/bag" 2>/dev/null | cut -f1) in $OUTDIR/bag"
  fi
  if [[ "$METRICS_ONLY" == "1" && -f "${OUT%.mp4}.base_height.json" ]]; then
    echo "done -> ${OUT%.mp4}.base_height.json"
  elif [[ "$METRICS_ONLY" == "0" && -f "$OUT" ]]; then
    echo "done -> $OUT"
  else
    echo "WARNING: no mp4 produced — check $LOG" >&2
    tail -5 "$LOG" >&2
  fi
else
  echo "stack running (pid $LAUNCH_PID); stop with: deploy_mac/kill_ros.sh && pkill -9 -f distill_recorder"
  wait "$LAUNCH_PID"
fi
