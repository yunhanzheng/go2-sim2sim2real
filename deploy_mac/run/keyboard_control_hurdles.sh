#!/usr/bin/env bash
# Keyboard-drive the HURDLE student (K00-05) across the 10-hurdle terrain in
# MuJoCo, using the real deploy ROS stack (mujoco_simulator + low_level_ctrl FSM
# + depth_sim + depth_encoder + rl_distill_policy), with keyboard_to_joy standing
# in for the Xbox remote.
#
# This is keyboard_control.sh specialised for the hurdle student: the stock
# script bakes in the T000 corridor policy's env_class_eq_17:=1 / action_delay:=0,
# which are WRONG for K00-05. K00-05 is a pure parkour student (env_class 16 =>
# env_class_eq_17:=0) trained with action_delay:=1. See memory
# project_k00_05_works_and_waitinit + feedback_check_env_class_eq_17.
#
# Controls (focus the terminal running this script; macOS needs Accessibility
# permission for it — System Settings > Privacy & Security > Accessibility):
#   A            stand -> lay   (gamepad A button)
#   B            lay -> stand   (gamepad B button)
#   Space (hold) DEAD-MAN engage: policy runs only while held; release -> stand
#                (pass --latch for latch behaviour: stays engaged)
#   W / S        forward / back      (the one channel the no-yaw-head student drives)
#   ← / →        strafe left / right (published; ignored by no-yaw-head student)
#   Q / E        yaw  left / right   (published; ignored by no-yaw-head student)
#   1-9 / 0      speed preset: stick magnitude 0.1-0.9 / 1.0 (live, mid-drive)
#   Z + X (hold) quit the stack
#
# Typical flow: wait for the MuJoCo window + "pre-settle" to finish, press B to
# stand (watch for standing=1), then hold Space and tap a digit / W to walk.
#
# Usage:
#   deploy_mac/run/keyboard_control_hurdles.sh [options] [-- extra ros2-launch args]
#
# Options:
#   --ckpt   PATH   checkpoint .pt (path or bare name under models/*/)  [K00-05 model_25000.pt]
#   --scene  NAME   scene xml      (path or bare name under resources/go2) [scene_10hurdles.xml]
#   --speed  N      keyboard stick magnitude [0,1] (top speed = lin_vel_max*N) [0.7]
#   --lin_vel_max N                                                     [1.0]
#   --kp N --kd N   policy gains                                        [30 / 0.6]
#   --contact N     contact_threshold (net MuJoCo foot force)          [2.0]
#   --env17 0|1     env_class_eq_17 (0 = parkour/hurdle, 1 = flat)      [0]
#   --delay 0|1     action_delay                                       [1]
#   --latch         Space latches engage instead of dead-man hold
#
# Examples:
#   deploy_mac/run/keyboard_control_hurdles.sh
#   deploy_mac/run/keyboard_control_hurdles.sh --speed 0.5
#   deploy_mac/run/keyboard_control_hurdles.sh --scene scene_2hurdles_smooth.xml
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EP_ROOT="$(cd "$ROOT/.." && pwd)/go2-flyball-parkour"

# ---- defaults (hurdle student K00-05 on the 10-hurdle terrain) --------------
CKPT="models/K00-05-student-pure-dagger-yaw45/model_25000.pt"
SCENE="scene_10hurdles.xml"
SPEED="0.7"
LIN_VEL_MAX="1.0"
KP="35.0"; KD="0.6"
CONTACT="2.0"
ENV17="0"          # K00-05 is parkour env_class 16 -> env_class_eq_17:=0
DELAY="1"          # K00-05 trained with action_delay:=1
MOMENTARY="true"   # Space = dead-man hold by default; --latch flips to latch mode
EXTRA=()

# ---- parse args -------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --ckpt) CKPT="$2"; shift 2;;
    --scene) SCENE="$2"; shift 2;;
    --speed) SPEED="$2"; shift 2;;
    --lin_vel_max) LIN_VEL_MAX="$2"; shift 2;;
    --kp) KP="$2"; shift 2;;
    --kd) KD="$2"; shift 2;;
    --contact) CONTACT="$2"; shift 2;;
    --env17) ENV17="$2"; shift 2;;
    --delay) DELAY="$2"; shift 2;;
    --latch) MOMENTARY="false"; shift;;
    --) shift; EXTRA=("$@"); break;;
    -h|--help) sed -n '2,52p' "${BASH_SOURCE[0]}"; exit 0;;
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

# ---- env (NOTE: no `set -u`; conda ros hook has unbound CONDA_BUILD) ---------
source ~/miniforge3/etc/profile.d/conda.sh 2>/dev/null \
  || source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null \
  || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate ros_humble
export GO2_IFACE=lo0
source "$ROOT/deploy_mac/env.sh"
export MUJOCO_GL=glfw
export KMP_DUPLICATE_LIB_OK=TRUE
# low_level_ctrl (C++) links against libpython3.12.dylib via an rpath that a
# colcon rebuild can drop; the lib lives in the conda env, so make dyld fall
# back to it. Without this the FSM node dies on "Library not loaded".
export DYLD_FALLBACK_LIBRARY_PATH="$CONDA_PREFIX/lib:${DYLD_FALLBACK_LIBRARY_PATH:-}"
source "$ROOT/install/setup.bash"

# ---- clean slate ------------------------------------------------------------
bash "$ROOT/deploy_mac/kill_ros.sh" >/dev/null 2>&1 || true

cat <<EOF
ckpt    : $CKPT
scene   : $SCENE
env17   : $ENV17   action_delay: $DELAY   kp/kd: $KP/$KD   contact: $CONTACT
speed   : stick mag $SPEED  (top speed = lin_vel_max $LIN_VEL_MAX * $SPEED m/s)
controls: A=lay  B=stand  Space(hold)=engage[$([[ $MOMENTARY == true ]] && echo dead-man || echo latch)]  W/S=fwd  ←/→=strafe  Q/E=yaw  1-9/0=speed  Z+X=quit
          (focus this terminal; macOS needs Accessibility permission)
launching keyboard-driven hurdle stack in MuJoCo... Ctrl-C to stop.
EOF

# ---- launch (foreground; runs until Ctrl-C) ---------------------------------
exec ros2 launch deploy_rl_policy distill.launch.py \
  is_simulation:=true use_keyboard:=true kbd_speed:="$SPEED" \
  sim_prefix:=mjpython momentary_engage:="$MOMENTARY" \
  ckpt:="$CKPT" ep_root:="$EP_ROOT" scene_xml:="$SCENE" \
  lin_vel_max:="$LIN_VEL_MAX" action_delay:="$DELAY" \
  policy_kp:="$KP" policy_kd:="$KD" \
  env_class_eq_17:="$ENV17" contact_threshold:="$CONTACT" \
  "${EXTRA[@]}"
