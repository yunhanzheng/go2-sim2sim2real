#!/usr/bin/env bash
# Start the REAL deploy stack on the robot's own Jetson.
#
# This DOES bring up low_level_ctrl, which publishes /lowcmd. It is still safe to
# run unattended: low_level_ctrl boots into LAYDOWN, tracks /lowstate and holds
# the lying pose. Motion only begins when the operator presses B (stand) and then
# holds L1+R1 (policy engage). Never touch the sticks before that.
#
#   tmux new -s go2
#   bash ~/go2_deploy/start_real.sh
#
# (no `set -u`: ROS setup scripts reference unset vars)
set +o pipefail

source "$HOME/go2_deploy/env.sh"

ROOT="$HOME/go2_deploy"

# Checkpoint presets: $1 picks one, and the matching env_class_eq_17 follows with it.
# That flag is a property of the checkpoint, not a free knob — D31 is the hurdle env
# class (0), L00-02 the flat one (1) — and a wrong value does not raise an error, it
# just walks badly. Not left to the operator.
#   bash start_real.sh            # d31  (hurdle, default)
#   bash start_real.sh l00        # L00-02 (flat)
#   bash start_real.sh student    # distilled student (same env class as d31)
#   bash start_real.sh /path/model_9000.pt    # any checkpoint; env class 0
# CKPT= / ENV17= still override either half for a one-off.
MODEL="${1:-d31}"
case "$MODEL" in
  d31) CKPT="${CKPT:-$ROOT/models/D31-fovscan_maxpool_fullforce_fx_2m-smoothfloor-l40s/model_9000.pt}"
       ENV17="${ENV17:-0}" ;;
  l00) CKPT="${CKPT:-$ROOT/models/L00-02-flat-baseline-delay-distillation-noised/model_14500.pt}"
       ENV17="${ENV17:-1}" ;;
  # Distilled students are built from the same go2-flyball-parkour run as d31, so
  # they carry env_class 19 — i.e. eq_17=0 — even though the weights are new. Only
  # the terrain level (lvl4 in the name) differs, and that is baked into the weights.
  student) CKPT="${CKPT:-$ROOT/models/student/model_5000.pt}"
       ENV17="${ENV17:-0}" ;;
  # A full path to a .pt is taken as-is. The env class is assumed to be the hurdle
  # one (0) because that is what every D31-lineage checkpoint here uses; override
  # with ENV17=1 for a flat-ground checkpoint.
  *.pt) CKPT="$(realpath "$MODEL")"; ENV17="${ENV17:-0}" ;;
  *)   echo "ERR: unknown model '$MODEL' — use d31, l00, student, or a path to a .pt"; exit 1 ;;
esac
[ -f "$CKPT" ] || { echo "ERR: ckpt missing: $CKPT"; exit 1; }

mkdir -p "$ROOT/run_logs/real"
LOG="$ROOT/run_logs/real/$(date +%Y-%m-%d_%H-%M-%S)"
echo "== log: $LOG.log"
echo "== model: $MODEL  env_class_eq_17=$ENV17"
echo "== ckpt : $CKPT"
echo "== robot stays LYING until you press B. hands off the sticks."

# depth_rate_hz is intentionally NOT passed: this run tests the launch's own
# default (depth_camera_node target_hz = 10 Hz), i.e. exactly what the untouched
# upstream code does. Background for the A/B: at 10 Hz the queue only ever holds
# ages around 68/168/268 ms, so with a nonzero depth budget the 80-100 ms
# training band is unreachable (measured: 132-165 ms, out-of-band counter rising
# every tick). With the budget at 0 the code just takes the newest frame, so
# 10 Hz instead shows up as depth_age wandering over 0-100 ms rather than
# sitting still. Passing depth_rate_hz:=0.0 (~60 Hz passthrough) was the
# alternative and is the one to restore if this run looks noisier.
#
# depth_latency_s / proprio_latency_s are deliberately NOT set: the launch's
# real-mode default (0/0) is what the laptop run that worked used. An earlier
# attempt set them to the paper's training values (0.080/0.016) chasing a squat
# on engage; that squat turned out to be the CARPET (the checkpoint is trained
# on smooth hard floor, friction 1.0), not latency. On the wooden floor at 0/0
# the robot stands cleanly. Do not reintroduce those two args without an A/B.
ros2 launch deploy_rl_policy distill.launch.py \
  is_simulation:=false \
  use_wireless_joy:=true \
  ckpt:="$CKPT" \
  ep_root:="$ROOT/ep" \
  env_class_eq_17:="$ENV17" \
  action_delay:=1 \
  binary_vel:=1 \
  lin_vel_max:=1.0 \
  policy_kp:=35.0 \
  policy_kd:=0.6 2>&1 | tee "$LOG.log"
