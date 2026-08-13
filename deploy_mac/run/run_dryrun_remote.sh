#!/usr/bin/env bash
# Remote-driven inference DRY-RUN with the depth camera, logging ALL policy outputs.
# NO low_level_ctrl is started -> nothing publishes /lowcmd -> the robot does NOT move
# from our stack. The remote's left stick (held with L1+R1) feeds the policy's velocity
# command; we publish /policy_engaged=true ourselves (low_level_ctrl normally does).
#
# !!! REQUIRES sport mode OFF !!!  Otherwise the Go2 FACTORY controller will move the
# robot in response to the remote. Disable it first (~/go2/disable_sport.py).
#
#   SECS=60 ENV_CLASS_EQ_17=1 ./deploy_mac/run/run_dryrun_remote.sh
# Logs to run_logs/dryrun/<timestamp>/.
# ENV_CLASS_EQ_17 is REQUIRED (no default): 1 = flat/env-17 ckpts (T000, L00),
# 0 = parkour/cliffside ckpts (OD, D08, D09) — a wrong value silently breaks the
# policy. ZERO_VEL_OBS=1 is REQUIRED for ckpts trained with
# zero_student_velocity_obs=True (e.g. models/D09-latencydr-zerovelobs-l40s).
# (no `set -u`: conda activation scripts reference unset vars)
HERE="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$HERE"
source deploy_mac/env.sh

EP="${EP_ROOT:-$(cd "$HERE/.." && pwd)/go2-flyball-parkour}"
# Default = the latest (highest-iteration) checkpoint in the active policy dir.
CKPT_DIR="${CKPT_DIR:-models/T000-00-step-corridor-distill}"
CKPT="${CKPT:-$(ls "$CKPT_DIR"/model_*.pt 2>/dev/null | sort -t_ -k2 -n | tail -1)}"
[ -n "$CKPT" ] || { echo "ERR: no model_*.pt found in $CKPT_DIR (set CKPT or CKPT_DIR)"; exit 1; }
CT="${CONTACT_THRESHOLD:-20,20,20,20}" # real raw sensor units, FR,FL,RR,RL
# YAW_HEAD=1 => 34-D yaw-head encoder (OD/ODFG cliffside, e.g. w68); exported so
# the encoder subprocess builds the 32+2 head (must match the checkpoint).
# EP main renamed the gate EP_FEED_PRED_YAW -> EP_USE_YAW (2026-07): EP_USE_YAW
# is the CURRENT gate; export BOTH so older EP checkouts keep working.
YAW_HEAD="${YAW_HEAD:-0}"
export EP_USE_YAW="$YAW_HEAD"       # current EP gate name
export EP_FEED_PRED_YAW="$YAW_HEAD" # legacy EP gate name (older checkouts)
# prop[11:13] env-class indicator (BOTH nodes): 1=flat/env-17 (T000, L00),
# 0=parkour/cliffside (OD, D08, D09). NO DEFAULT: a wrong value silently breaks
# the policy (one-hot bits must match the checkpoint), so the caller MUST set it.
if [ -z "${ENV_CLASS_EQ_17:-}" ]; then
  echo "ERR: ENV_CLASS_EQ_17 is unset — set it explicitly (no safe default exists)." >&2
  echo "     The one-hot env-class bits prop[11:13] must match the checkpoint:" >&2
  echo "       ENV_CLASS_EQ_17=1  flat / env-17 students (T000, L00)" >&2
  echo "       ENV_CLASS_EQ_17=0  parkour / cliffside students (OD, D08, D09)" >&2
  echo "     A mismatch silently breaks the policy (encoder + actor must agree)." >&2
  exit 1
fi
ENV17="$ENV_CLASS_EQ_17"
# ZERO_YAW_PRED=1 => policy ignores the encoder yaw head and holds prop[6:8]=0
# (constant zero-heading command). Encoder still builds/publishes the 34-D head.
#   YAW_HEAD=1 ZERO_YAW_PRED=1 ENV_CLASS_EQ_17=0 LIN_VEL_MAX=1.5 \
#   CKPT=models/OD-fine-ns240-h005-w68/model_10000.pt ./deploy_mac/run/run_dryrun_remote.sh
ZERO_YAW_PRED="${ZERO_YAW_PRED:-0}"
# ZERO_VEL_OBS=1 => the actor forces priv_explicit[0:3] (estimated base lin vel)
# to zero. MUST be 1 for checkpoints trained with zero_student_velocity_obs=True
# (e.g. models/D09-latencydr-zerovelobs-l40s); leave 0 for all other checkpoints.
ZERO_VEL_OBS="${ZERO_VEL_OBS:-0}"
# BINARY_VEL=1 (default) => bang-bang velocity: stick forward past deadzone =>
# constant LIN_VEL_MAX (1.0), 0 on release. Set BINARY_VEL=0 for proportional.
BINARY_VEL="${BINARY_VEL:-1}"
VEL_DEADZONE="${VEL_DEADZONE:-0.2}"
LV="${LIN_VEL_MAX:-1.0}"                # two-node path: passed to BOTH encoder and policy (must match)
# FUSED=1 (DEFAULT since 2026-07-27) => single-process encoder+actor off ONE
# proprio frame per tick (the distillation training contract). This dry-run is the
# rehearsal path: it exercises the fused node against the LIVE RealSense and the
# /rl/depth_clock_offset correction — the one code path that never runs in sim —
# without driving motors. Set FUSED=0 for the legacy two-node split.
FUSED="${FUSED:-1}"
# Pads default to the REAL-RUN values (EP §4.1), not 0. They used to default off,
# which made this dry-run useless as a rehearsal: with depth_latency_s=0 the age
# gate takes the newest frame every tick, so the 80 ms contract the real run
# enforces was never exercised — and the bring-up runbook asks the operator to
# verify depth_age/early here. Pass 0.0 explicitly for a no-pad diagnostic.
DEPTH_LATENCY_S="${DEPTH_LATENCY_S:-0.08}"
PROPRIO_LATENCY_S="${PROPRIO_LATENCY_S:-0.016}"
# Jetson<->host clock sync, as in run_real_remote.sh: the depth pad is
# referenced to the camera-host stamp, so without this the fused skew guard
# distrusts the stamp and ages by receipt. 0 to skip (clocks synced elsewhere).
SYNC_JETSON_CLOCK="${SYNC_JETSON_CLOCK:-1}"
SECS="${SECS:-60}"

# Checkpoint/flag contract check (optional, runs BEFORE any node starts): the
# checker inspects the ckpt and rejects flag combinations that contradict it
# (exit 0 = OK, nonzero = mismatch + printed reasons). Skipped if absent.
if [ -f deploy_mac/diag/check_deploy_contract.py ]; then
  python deploy_mac/diag/check_deploy_contract.py --ckpt "$CKPT" \
    --env_class_eq_17 "$ENV17" --yaw_head "$YAW_HEAD" --zero_vel_obs "$ZERO_VEL_OBS" \
    --binary_vel "$BINARY_VEL" --lin_vel_max "$LV" \
    || { echo "ERR: deploy contract mismatch — see reasons above (deploy_mac/diag/check_deploy_contract.py). Aborting."; exit 1; }
fi

TS="$(date +%Y-%m-%d_%H-%M-%S)"
# Grouped by policy: run_logs/dryrun/<ckptdir>__<model>/<timestamp>/
POLICY="$(basename "$(dirname "$CKPT")")__$(basename "$CKPT" .pt)"
RUN="run_logs/dryrun/$POLICY/$TS"
mkdir -p "$RUN"
{ echo "timestamp:         $TS"
  echo "mode:              DRYRUN camera+policy (NO low_level_ctrl, no /lowcmd)"
  echo "ckpt:              $CKPT"
  echo "contact_threshold: $CT"
  echo "yaw_head:          $YAW_HEAD (EP_USE_YAW = current gate; legacy EP_FEED_PRED_YAW also exported)"
  echo "zero_yaw_pred:     $ZERO_YAW_PRED (1 = prop[6:8] held at constant 0)"
  echo "zero_vel_obs:      $ZERO_VEL_OBS (1 = priv_explicit[0:3] forced to zero; zero-vel-obs ckpts)"
  echo "binary_vel:        $BINARY_VEL (1 = stick-forward => $LV m/s, else 0; deadzone $VEL_DEADZONE)"
  echo "env_class_eq_17:   $ENV17"
  echo "lin_vel_max:       $LV"
  echo "depth_latency_s:   $DEPTH_LATENCY_S (synthetic enforced delay; 0 = off)"
  echo "proprio_latency_s: $PROPRIO_LATENCY_S (synthetic enforced delay; 0 = off)"
} > "$RUN/meta.txt"
# git provenance: commit/branch/dirty-state in meta.txt + git_diff.patch (uncommitted changes)
./deploy_mac/lib/log_git_state.sh "$RUN"
export PYTHONPATH="$HERE/deploy_mac/stubs:$EP/rsl_rl:$EP/legged_gym:$HERE/src/deploy_rl_policy/scripts:${PYTHONPATH:-}"

cleanup() { kill "${DC:-}" "${E:-}" "${P:-}" "${W:-}" "${G:-}" "${V:-}" 2>/dev/null; ./deploy_mac/kill_ros.sh >/dev/null 2>&1; echo "logs -> $RUN/"; }
trap cleanup EXIT INT TERM

# Sync the camera-host clock BEFORE any node starts, mirroring the real run: the
# depth pad is exposure-referenced, so an unsynced Jetson makes the fused skew
# guard distrust the stamp and fall back to receipt-based ageing.
if [ "$SYNC_JETSON_CLOCK" != "0" ] && [ -f deploy_mac/camera/sync_jetson_clock.sh ]; then
  ./deploy_mac/camera/sync_jetson_clock.sh 2>&1 | tee "$RUN/clock_sync.log"
fi

ros2 topic list 2>/dev/null | grep -q image_rect_raw \
  || { echo "ERR: /camera/depth/image_rect_raw not found — is the Jetson camera service up?"; exit 1; }

echo "== L00-02 | contact_threshold=$CT | $SECS s | logs -> $RUN/"
# track_clock_offset: publishes /rl/depth_clock_offset. REQUIRED for the pad to
# be referenced correctly (camera stamps are in the Jetson clock domain) and for
# the runbook gate `clk_ofs=rdy`. Was missing here, so the dry-run could never
# reach that state.
# target_hz:=0 = pass EVERY camera frame through (30 Hz). The old 10 Hz
# throttle matched the sleep-based encoder's cadence; with the fused AGE GATE
# the encode cadence is set by the policy node itself, and a 10 Hz feed makes
# the delivered depth age BEAT against the 10 Hz encode tick (measured
# 2026-07-27: in-band 60% vs 28% run-to-run at identical config). Full camera
# rate gives the gate 33 ms age granularity -> deterministic [80, 113) ms.
python src/deploy_rl_policy/scripts/depth_camera_node.py \
  --ros-args -p input_topic:=/camera/depth/image_rect_raw \
  -p target_hz:=0.0 \
  -p track_clock_offset:=true > "$RUN/depth_camera.log" 2>&1 & DC=$!
if [ "$FUSED" != "0" ]; then
  # Single-process encoder+actor (ONE proprio frame per 50 Hz tick). This is the
  # rehearsal path for FUSED=1 on the real robot: it exercises the same node with
  # the live RealSense + /rl/depth_clock_offset correction, but does not drive
  # motors. Check the node's status line for `depth_age` near the 80 ms budget,
  # `early=0`, and no "tick overrun" warnings before trusting a real run.
  echo "== FUSED policy: single-process encoder+actor (one proprio frame/tick)"
  python src/deploy_rl_policy/scripts/distill_fused_policy.py \
    --ckpt "$CKPT" --ep_root "$EP" --env_class_eq_17 "$ENV17" --contact_threshold "$CT" \
    --lin_vel_max "$LV" --depth_latency_s "$DEPTH_LATENCY_S" \
    --proprio_latency_s "$PROPRIO_LATENCY_S" \
    --zero_yaw_pred "$ZERO_YAW_PRED" --zero_student_velocity_obs "$ZERO_VEL_OBS" \
    --binary_vel "$BINARY_VEL" \
    --vel_deadzone "$VEL_DEADZONE" \
    --latency_log "$RUN/latency_actor.tsv" > "$RUN/rl_policy.log" 2>&1 & P=$!
  E=""
else
python src/deploy_rl_policy/scripts/depth_encoder_node.py \
  --ckpt "$CKPT" --ep_root "$EP" --contact_threshold "$CT" --env_class_eq_17 "$ENV17" \
  --lin_vel_max "$LV" --enforce_latency_s "$DEPTH_LATENCY_S" \
  --binary_vel "$BINARY_VEL" --vel_deadzone "$VEL_DEADZONE" > "$RUN/depth_encoder.log" 2>&1 & E=$!
python src/deploy_rl_policy/scripts/rl_distill_policy.py \
  --ckpt "$CKPT" --ep_root "$EP" --env_class_eq_17 "$ENV17" --contact_threshold "$CT" \
  --lin_vel_max "$LV" --proprio_latency_s "$PROPRIO_LATENCY_S" \
  --zero_yaw_pred "$ZERO_YAW_PRED" --zero_student_velocity_obs "$ZERO_VEL_OBS" \
  --binary_vel "$BINARY_VEL" \
  --vel_deadzone "$VEL_DEADZONE" > "$RUN/rl_policy.log" 2>&1 & P=$!
fi
# Scheduling hardening (mirrors run_real_remote.sh): remove macOS background
# throttling from the actor and try renice — the rare ~1 s actor stalls were
# measured in dryruns. Best-effort; renice without root just logs a note.
taskpolicy -B -p "$P" 2>/dev/null 1>&2 || true
renice -10 -p "$P" >/dev/null 2>&1 \
  || echo "note: renice -10 pid $P needs root — 'sudo renice -10 -p $P' for stall protection"
python deploy_mac/run/wireless_to_joy.py > "$RUN/wireless_to_joy.log" 2>&1 & W=$!
ros2 topic pub -r 5 /policy_engaged std_msgs/msg/Bool "{data: true}" > "$RUN/engage.log" 2>&1 & G=$!
# Optional: record the yaw-prediction view headless into the run dir (RECORD_VIEW=1).
V=""
if [ -n "${RECORD_VIEW:-}" ]; then
  SCENE_ARG=(); [ -n "${SCENE:-}" ] && SCENE_ARG=(--scene "$SCENE")
  echo "== recording yaw-prediction view -> $RUN/yaw_prediction.mp4 (headless)"
  python deploy_mac/diag/view_prediction.py --no-display \
    --record "$RUN/yaw_prediction.mp4" --seconds "$SECS" --fps 20 \
    --zero_yaw_pred "$ZERO_YAW_PRED" \
    "${SCENE_ARG[@]}" > "$RUN/view_prediction.log" 2>&1 & V=$!
fi

echo "== READY. On the remote: HOLD L1+R1 and push the LEFT STICK forward to command velocity."
echo "==        (robot will NOT move — no /lowcmd. Logging the policy's response.)"
python deploy_mac/analysis/log_policy.py --out "$RUN/policy_log.csv" --secs "$SECS"

echo "--- rl_policy log tail ---"; tail -6 "$RUN/rl_policy.log"
