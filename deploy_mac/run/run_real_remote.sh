#!/usr/bin/env bash
# REAL full policy stack, driven by the Go2's OWN remote (no `joy` package, no USB
# gamepad). This is run_dryrun_remote.sh + the C++ low_level_ctrl, so it DOES drive
# /lowcmd: *** THE ROBOT MOVES. ***  Use distill.launch.py only if you have the `joy`
# package + a USB gamepad; this script is the remote-driven path for this Mac.
#
# Prereqs (ALL of them):
#   - robot HOISTED / tethered, clear space, hand on power
#   - sport mode OFF        (~/go2/disable_sport.py)
#   - Jetson realsense publishing /camera/depth/image_rect_raw  (S00-03 needs depth)
#
# Chain (FUSED=1, the DEFAULT): wireless_to_joy(remote->/joy) + depth_camera ->
#        distill_fused_policy (encoder+actor in ONE process, one proprio frame per
#        50 Hz tick) -> low_level_ctrl(FSM, PD) -> /lowcmd -> motors.
#        With FUSED=0: depth_camera -> depth_encoder -> rl_distill_policy (legacy
#        two-node split). low_level_ctrl OWNS /policy_engaged (engage by holding
#        L1+R1) — we do NOT fake it here.
#
#   ENV_CLASS_EQ_17=1 ./deploy_mac/run/run_real_remote.sh
#   ENV_CLASS_EQ_17=1 CKPT=models/L00-02-flat-baseline-delay-distillation-noised/model_14500.pt ./deploy_mac/run/run_real_remote.sh
# Default CKPT = latest (highest-iteration) model_*.pt in CKPT_DIR
# (models/T000-00-step-corridor-distill); override CKPT or CKPT_DIR to change.
# Overridable: FUSED (0/1; DEFAULT 1 = single-process encoder+actor, ONE proprio
#                frame per tick — the distillation training contract. Set FUSED=0
#                for the legacy two-node split. Dry-run any new checkpoint first),
#              CKPT, CKPT_DIR, EP_ROOT, CONTACT_THRESHOLD, LIN_VEL_MAX, POLICY_KP, POLICY_KD,
#              YAW_HEAD (0/1; 1 => 34-D yaw-head OD/ODFG students),
#              ENV_CLASS_EQ_17 (0/1; REQUIRED, no default — one-hot env-class bits
#                prop[11:13] must match the ckpt: 1 = flat/env-17 (T000, L00),
#                0 = parkour/cliffside (OD, D08, D09); a wrong value silently
#                breaks the policy, so the script refuses to guess),
#              ZERO_VEL_OBS (0/1; MUST be 1 for ckpts trained with
#                zero_student_velocity_obs=True, e.g. models/D09-latencydr-zerovelobs-l40s
#                — the actor then forces priv_explicit[0:3] to zero; 0 for all others),
#              ZERO_YAW_PRED (0/1; 1 => ignore the encoder yaw head and feed a CONSTANT
#                ZERO heading into prop[6:8] — the "zero-heading command" scenario),
#              FULL_LOG (default 0 = off; set 1 to ros2-bag camera+policy+control topics),
#              LATENCY_LOG (default 0; set 1 for lightweight correlated timing TSVs),
#              SYNC_JETSON_CLOCK (default 1; sync RealSense host clock to Mac —
#                required by the exposure-referenced depth pad, see below),
#              DEPTH_LATENCY_S (default 0.08), PROPRIO_LATENCY_S (default 0.016)
#                (EP §4.1 constant-latency pads; 0.0 = off, natural latency),
#              RECORD_VIEW (set =1 to also save the live depth+|dlat| view mp4).
#
# Deploy an OD/ODFG yaw-head student (e.g. w68) on the real robot:
#   YAW_HEAD=1 ENV_CLASS_EQ_17=0 LIN_VEL_MAX=1.75 \
#   CKPT=models/OD-fine-ns240-h005-w68/model_6000.pt ./deploy_mac/run/run_real_remote.sh
#   (kp defaults to 35; contact defaults to 20,20,20,20.)
#
# Same model but with the HEADING PREDICTION SWAPPED FOR A CONSTANT ZERO command
# (the encoder still loads the 34-D head, but the policy ignores yaw_pred and holds
# prop[6:8]=0 — matches deploy_mac/sim/render_sim2sim.sh ... -- zero_yaw_pred:=1):
#   YAW_HEAD=1 ZERO_YAW_PRED=1 ENV_CLASS_EQ_17=0 LIN_VEL_MAX=1.5 \
#   CKPT=models/OD-fine-ns240-h005-w68/model_10000.pt ./deploy_mac/run/run_real_remote.sh
# (no `set -u`: conda activation scripts reference unset vars)
HERE="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$HERE"
source deploy_mac/env.sh

# low_level_ctrl (C++) links against libpython3.12.dylib via an rpath that a
# colcon rebuild can drop; the lib lives in the conda env, so make dyld fall
# back to it. Without this the FSM node dies on "Library not loaded" -> no
# /lowcmd -> the robot ignores all commands.
RS_MSG_LIB="$HERE/install/realsense2_camera_msgs/lib"
export DYLD_LIBRARY_PATH="$RS_MSG_LIB:${DYLD_LIBRARY_PATH:-}"
export DYLD_FALLBACK_LIBRARY_PATH="$RS_MSG_LIB:$CONDA_PREFIX/lib:${DYLD_FALLBACK_LIBRARY_PATH:-}"

EP="${EP_ROOT:-$(cd "$HERE/.." && pwd)/go2-flyball-parkour}"
# Default = the latest (highest-iteration) checkpoint in the active policy dir.
CKPT_DIR="${CKPT_DIR:-models/T000-00-step-corridor-distill}"
CKPT="${CKPT:-$(ls "$CKPT_DIR"/model_*.pt 2>/dev/null | sort -t_ -k2 -n | tail -1)}"
[ -n "$CKPT" ] || { echo "ERR: no model_*.pt found in $CKPT_DIR (set CKPT or CKPT_DIR)"; exit 1; }
CT="${CONTACT_THRESHOLD:-20,20,20,20}" # real raw sensor units, FR,FL,RR,RL; sim uses 2.0 N
LV="${LIN_VEL_MAX:-1.0}"               # two-node path: passed to BOTH encoder and policy (must match)
# FUSED=1 (DEFAULT since 2026-07-27) => one process runs the depth encoder AND
# the actor off ONE proprio frame per 50 Hz tick (the distillation training
# contract), instead of two nodes that each assemble their own frame up to a
# camera period apart. CT/LV/BINARY_VEL/VEL_DEADZONE are consumed exactly once,
# so they cannot desync between latent and actor input.
#
# Why default: the Mac A/B's original "fused 2/10" was a PARITY ARTIFACT. Rerun at
# parity it scored 8/10, and interleaved against the embedded arm it went 4/5 vs
# 3/5 — statistically identical in the fused-vs-embedded A/B.
# Frame-delay is also the more training-faithful contract (Isaac's
# depth_buffer[-1-L] paired with the CURRENT tick's prop).
#
# STILL NOT VALIDATED ON HARDWARE (no arm is). Two fused-specific things to watch
# on a first real run, both instrumented: the depth age gate's chosen-frame age
# (`depth_age` / `early=` in the node's status line; the gate is clock-corrected
# via /rl/depth_clock_offset, a path that only executes on the real robot), and
# "tick overrun" warnings — the fused tick pays encoder+actor every 5th tick in
# one process. Set FUSED=0 to fall back to the two-node split.
FUSED="${FUSED:-1}"
# YAW_HEAD=1 => 34-D yaw-head depth encoder (OD/ODFG cliffside students, e.g.
# w67/w68). The env var gates the RecurrentDepthBackbone to build the 32+2
# head; it MUST match the checkpoint (a 34-D ckpt strict-loads only when =1).
# EP main renamed the gate EP_FEED_PRED_YAW -> EP_USE_YAW (2026-07): EP_USE_YAW
# is the CURRENT gate (depth_backbone.py); we export BOTH so older EP checkouts
# keep working. Exported so the depth_encoder_node subprocess inherits them.
YAW_HEAD="${YAW_HEAD:-0}"
export EP_USE_YAW="$YAW_HEAD"       # current EP gate name
export EP_FEED_PRED_YAW="$YAW_HEAD" # legacy EP gate name (older checkouts)
# prop[11:13] env-class indicator, passed to BOTH encoder and policy (lockstep):
# 1 = flat / env-17 (T000, L00); 0 = parkour / cliffside (OD, D08, D09).
# NO DEFAULT: the one-hot bits must match the checkpoint's training terrain and
# a wrong value silently breaks the policy, so the caller MUST set it.
if [ -z "${ENV_CLASS_EQ_17:-}" ]; then
  echo "ERR: ENV_CLASS_EQ_17 is unset — set it explicitly (no safe default exists)." >&2
  echo "     The one-hot env-class bits prop[11:13] must match the checkpoint:" >&2
  echo "       ENV_CLASS_EQ_17=1  flat / env-17 students (T000, L00)" >&2
  echo "       ENV_CLASS_EQ_17=0  parkour / cliffside students (OD, D08, D09)" >&2
  echo "     A mismatch silently breaks the policy (encoder + actor must agree)." >&2
  exit 1
fi
ENV17="$ENV_CLASS_EQ_17"
# ZERO_YAW_PRED=1 => rl_distill_policy ignores the encoder's regressed heading and
# holds prop[6:8]=0 (constant zero-heading command). The encoder still builds the
# 34-D head (EP_FEED_PRED_YAW=$YAW_HEAD above) and publishes yaw_pred — only the
# policy's consumption is bypassed. No-op for no-yaw (32-D) students.
ZERO_YAW_PRED="${ZERO_YAW_PRED:-0}"
# ZERO_VEL_OBS=1 => the actor forces priv_explicit[0:3] (estimated base lin vel)
# to zero. MUST be 1 for checkpoints trained with zero_student_velocity_obs=True
# (e.g. models/D09-latencydr-zerovelobs-l40s); leave 0 for all other checkpoints.
ZERO_VEL_OBS="${ZERO_VEL_OBS:-0}"
# BINARY_VEL=1 (default) => bang-bang velocity: push the stick forward past the
# deadzone and the command is a CONSTANT LIN_VEL_MAX (1.0), 0 on release (no
# proportional control). Set BINARY_VEL=0 for proportional stick control.
BINARY_VEL="${BINARY_VEL:-1}"
VEL_DEADZONE="${VEL_DEADZONE:-0.2}"
# Force float notation — low_level_ctrl declares policy_kp/kd as double and
# ROS2 rejects integer literals (e.g. 40 fails, 40.0 works). Default 35.0/0.6.
KP="$(printf '%.1f' "${POLICY_KP:-35.0}")"
KD="$(printf '%.1f' "${POLICY_KD:-0.6}")"
ME="${MOMENTARY_ENGAGE:-true}"          # policy active only WHILE L1+R1 held; release -> stand
FULL_LOG="${FULL_LOG:-0}"               # 0 = no bag (default; 457MB bag I/O caused /lowcmd stalls
                                        # 2026-06-19 -> snap-backs). Set FULL_LOG=1 to re-enable.
LATENCY_LOG="${LATENCY_LOG:-0}"         # lightweight timing only; no image/message payload recording
# Default ON (was: tied to LATENCY_LOG): the depth constant-latency pad is
# exposure-referenced, and the passive drift tracker only corrects *drift
# since launch* — its baseline assumes the launch-time sync zeroed the
# absolute Jetson-Mac offset (verified to ±4 ms). Unsynced clocks would bias
# the enforced 80 ms by the whole launch-time offset.
SYNC_JETSON_CLOCK="${SYNC_JETSON_CLOCK:-1}"
# Constant-latency enforcement (EP paper §4.1): pad depth capture->latent to
# 0.08 s and lowstate receipt->target to 0.016 s so the real latencies match
# the constants training assumes, instead of jittering below them. Measured
# natural latencies (2026-07-24): depth exposure->latent ~40-75 ms, proprio
# receipt->publish ~1.5-2 ms — both comfortably under their budgets, so the
# pads engage on ~99.8% of cycles. Set 0.0 to disable (pre-2026-07-25 behavior).
# The depth budget is exposure-referenced across the Jetson-Mac clock pair and
# depends on the passive drift tracker (track_clock_offset below) staying on.
DEPTH_LATENCY_S="${DEPTH_LATENCY_S:-0.08}"
PROPRIO_LATENCY_S="${PROPRIO_LATENCY_S:-0.016}"

# Checkpoint/flag contract check (optional, runs BEFORE any node starts): the
# checker inspects the ckpt and rejects flag combinations that contradict it
# (exit 0 = OK, nonzero = mismatch + printed reasons). Skipped if absent.
if [ -f deploy_mac/diag/check_deploy_contract.py ]; then
  python deploy_mac/diag/check_deploy_contract.py --ckpt "$CKPT" \
    --env_class_eq_17 "$ENV17" --yaw_head "$YAW_HEAD" --zero_vel_obs "$ZERO_VEL_OBS" \
    --binary_vel "$BINARY_VEL" --lin_vel_max "$LV" \
    || { echo "ERR: deploy contract mismatch — see reasons above (deploy_mac/diag/check_deploy_contract.py). Aborting."; exit 1; }
fi

# Raising a process above the default macOS priority requires root. Authenticate
# before clock sync and, more importantly, before any motor-driving process is
# started; the later sudo -n call must never stop a live stack for a password.
echo "== authorizing local scheduler priority (sudo)"
sudo -v \
  || { echo "ERR: sudo authorization failed — cannot apply scheduler stall protection"; exit 1; }

TS="$(date +%Y-%m-%d_%H-%M-%S)"
# Grouped by policy: run_logs/real/<ckptdir>__<model>/<timestamp>/
POLICY="$(basename "$(dirname "$CKPT")")__$(basename "$CKPT" .pt)"
RUN="run_logs/real/$POLICY/$TS"
mkdir -p "$RUN"
# The USB camera driver converts RealSense GLOBAL_TIME into the Jetson system
# clock before publishing ROS headers. A synchronized Jetson clock is therefore
# required for exposure-to-Mac timing. Do this before any motor-driving process
# starts; set SYNC_JETSON_CLOCK=0 only when the clocks are synchronized elsewhere.
if [ "$SYNC_JETSON_CLOCK" != "0" ]; then
  ./deploy_mac/camera/sync_jetson_clock.sh 2>&1 | tee "$RUN/clock_sync.log"
  [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "ERR: Jetson clock synchronization failed"; exit 1; }
fi
{ echo "timestamp:         $TS"
  echo "mode:              REAL full stack (low_level_ctrl is_simulation:=false; /lowcmd LIVE)"
  echo "ckpt:              $CKPT"
  echo "contact_threshold: $CT"
  echo "lin_vel_max:       $LV"
  echo "yaw_head:          $YAW_HEAD (EP_USE_YAW = current gate; legacy EP_FEED_PRED_YAW also exported)"
  echo "zero_yaw_pred:     $ZERO_YAW_PRED (1 = prop[6:8] held at constant 0)"
  echo "zero_vel_obs:      $ZERO_VEL_OBS (1 = priv_explicit[0:3] forced to zero; zero-vel-obs ckpts)"
  echo "binary_vel:        $BINARY_VEL (1 = stick-forward => $LV m/s, else 0; deadzone $VEL_DEADZONE)"
  echo "env_class_eq_17:   $ENV17"
  echo "policy_kp/kd:      $KP / $KD"
  echo "momentary_engage:  $ME"
  echo "full_log (rosbag): $FULL_LOG"
  echo "latency_log:       $LATENCY_LOG"
  echo "sync_jetson_clock: $SYNC_JETSON_CLOCK"
  echo "depth_latency_s:   $DEPTH_LATENCY_S (constant-latency pad, exposure-referenced w/ clock-offset tracking; 0 = off)"
  echo "proprio_latency_s: $PROPRIO_LATENCY_S (constant-latency pad, lowstate-receipt-referenced; 0 = off)"
} > "$RUN/meta.txt"
./deploy_mac/lib/log_git_state.sh "$RUN"
export PYTHONPATH="$HERE/deploy_mac/stubs:$EP/rsl_rl:$EP/legged_gym:$HERE/src/deploy_rl_policy/scripts:${PYTHONPATH:-}"

DC_TIMING_ARGS=()
ENC_TIMING_ARGS=()
POLICY_TIMING_ARGS=()
LLC_TIMING_ARGS=()
if [ "$LATENCY_LOG" != "0" ]; then
  DC_TIMING_ARGS=(-p latency_log:="$RUN/latency_depth_camera.tsv")
  ENC_TIMING_ARGS=(--latency_log "$RUN/latency_depth_encoder.tsv")
  POLICY_TIMING_ARGS=(--latency_log "$RUN/latency_actor.tsv")
  LLC_TIMING_ARGS=(-p latency_log:="$RUN/latency_low_level.tsv")
fi

cleanup() {
  echo; echo "stopping stack ..."
  kill "${W:-}" "${M:-}" "${DC:-}" "${E:-}" "${P:-}" "${L:-}" "${V:-}" "${BAG:-}" "${WD:-}" 2>/dev/null
  ./deploy_mac/kill_ros.sh >/dev/null 2>&1
  if [ "$LATENCY_LOG" != "0" ]; then
    # Give the instrumented nodes a bounded window to close their line-buffered
    # TSVs. Never let diagnostics make teardown hang on a stuck process.
    for _ in $(seq 1 20); do
      live=0
      for pid in "${M:-}" "${DC:-}" "${E:-}" "${P:-}" "${L:-}"; do
        [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null && live=1
      done
      [ "$live" = "0" ] && break
      sleep 0.1
    done
    # </dev/null: cleanup runs after Ctrl-C has torn down the terminal's stdin;
    # python aborts at interpreter init ("init_sys_streams: Bad file descriptor")
    # without it (2026-07-24 run) and the summary comes out empty.
    python deploy_mac/analysis/analyze_latency_logs.py "$RUN" < /dev/null \
      > "$RUN/latency_summary.txt" 2> "$RUN/latency_analysis.log" || true
  fi
  echo "logs -> $RUN/"
}
trap cleanup EXIT INT TERM

ros2 topic list 2>/dev/null | grep -q image_rect_raw \
  || { echo "ERR: /camera/depth/image_rect_raw not found — is the Jetson camera service up?"; exit 1; }

echo "==================================================================="
echo " REAL RUN -> $RUN     ckpt=$CKPT"
echo " Robot WILL move.  HOISTED?  sport mode OFF?  hand on power?"
echo " Remote:  B = stand    HOLD L1+R1 = walk (release -> stand)    A = lay    L2+R2 = quit"
echo "==================================================================="

python deploy_mac/run/wireless_to_joy.py > "$RUN/wireless_to_joy.log" 2>&1 & W=$!
M=""
if [ "$LATENCY_LOG" != "0" ]; then
  python deploy_mac/camera/camera_metadata_latency.py \
    --output "$RUN/latency_camera_metadata.tsv" \
    > "$RUN/camera_metadata_latency.log" 2>&1 & M=$!
fi
# track_clock_offset: passive Jetson-Mac drift tracker (REAL path only — the
# camera header stamps are RealSense GLOBAL_TIME in the Jetson clock domain
# here; sim launches must keep the node default off). Publishes
# /rl/depth_clock_offset so the encoder's exposure-referenced 80 ms budget
# stays correct under clock drift (~1 ms/min) and NTP steps.
# target_hz:=0 = pass every camera frame (60 Hz since the 2026-07-27 switch to
# 480x270x60; was 30) — the fused age gate needs full capture rate for fine age
# granularity; a 10 Hz feed phase-beats against the 10 Hz encode tick (see
# run_dryrun_remote.sh note, measured 2026-07-27). Passthrough is what makes the
# camera rate reach the gate: the ENCODER still steps at 10 Hz
# (depth_update_interval), but it now picks from a 16.7 ms-spaced queue, so the
# delivered age lands in [80.0, 96.7] ms — inside the trained 80-100 ms band on
# every tick, vs 60% of ticks at 30 Hz.
python src/deploy_rl_policy/scripts/depth_camera_node.py \
  --ros-args -p input_topic:=/camera/depth/image_rect_raw \
  -p target_hz:=0.0 \
  -p track_clock_offset:=true "${DC_TIMING_ARGS[@]}" \
  > "$RUN/depth_camera.log" 2>&1 & DC=$!
if [ "$FUSED" != "0" ]; then
  # ONE process owns the depth encoder AND the actor, driven by ONE proprio
  # frame per 50 Hz tick — the contract distillation training uses
  # (on_policy_runner.py:312-319). Removes the encoder-vs-actor proprio skew and
  # the matched-flag-pair desync class; see FINDINGS F01-F03/F12 and
  # the fused-vs-two-node A/B.
  #
  # Two hardware-specific notes:
  #  * the 80 ms depth budget is enforced by an AGE GATE over timestamped frames
  #    (corrected by /rl/depth_clock_offset from depth_camera_node's tracker)
  #    instead of a sleep, because a fused tick cannot block without missing the
  #    20 ms actor deadline. That correction path is exercised only on the real
  #    robot — watch `depth_age` and `early=` in the node's status line.
  #  * the fused tick pays encoder + actor every 5th tick in one process. Watch
  #    for "tick overrun" warnings; if they are systematic on this host, the
  #    encoder needs to move to a worker thread (still fed the same snapshot).
  echo "== FUSED policy: single-process encoder+actor (one proprio frame/tick)"
  # Per-tick diagnostic publishes only when something consumes them (viewer or
  # bag) — otherwise they are dead serialization inside the 20 ms tick.
  DIAG=0
  { [ -n "${RECORD_VIEW:-}" ] || [ "${FULL_LOG:-0}" != "0" ]; } && DIAG=1
  python src/deploy_rl_policy/scripts/distill_fused_policy.py \
    --diagnostics "$DIAG" \
    --ckpt "$CKPT" --ep_root "$EP" --env_class_eq_17 "$ENV17" --contact_threshold "$CT" \
    --lin_vel_max "$LV" --depth_latency_s "$DEPTH_LATENCY_S" \
    --proprio_latency_s "$PROPRIO_LATENCY_S" \
    --zero_yaw_pred "$ZERO_YAW_PRED" --zero_student_velocity_obs "$ZERO_VEL_OBS" \
    --binary_vel "$BINARY_VEL" --vel_deadzone "$VEL_DEADZONE" "${POLICY_TIMING_ARGS[@]}" \
    > "$RUN/rl_policy.log" 2>&1 & P=$!
  E=""   # no separate encoder process; watchdog skips empty PIDs
else
python src/deploy_rl_policy/scripts/depth_encoder_node.py \
  --ckpt "$CKPT" --ep_root "$EP" --env_class_eq_17 "$ENV17" --contact_threshold "$CT" \
  --lin_vel_max "$LV" --enforce_latency_s "$DEPTH_LATENCY_S" \
  --binary_vel "$BINARY_VEL" --vel_deadzone "$VEL_DEADZONE" \
  "${ENC_TIMING_ARGS[@]}" > "$RUN/depth_encoder.log" 2>&1 & E=$!
python src/deploy_rl_policy/scripts/rl_distill_policy.py \
  --ckpt "$CKPT" --ep_root "$EP" --env_class_eq_17 "$ENV17" --contact_threshold "$CT" \
  --lin_vel_max "$LV" --proprio_latency_s "$PROPRIO_LATENCY_S" \
  --zero_yaw_pred "$ZERO_YAW_PRED" --zero_student_velocity_obs "$ZERO_VEL_OBS" \
  --binary_vel "$BINARY_VEL" --vel_deadzone "$VEL_DEADZONE" "${POLICY_TIMING_ARGS[@]}" \
  > "$RUN/rl_policy.log" 2>&1 & P=$!
fi
ros2 run deploy_rl_policy low_level_ctrl \
  --ros-args -p is_simulation:=false -p policy_kp:="$KP" -p policy_kd:="$KD" \
  -p momentary_engage:="$ME" "${LLC_TIMING_ARGS[@]}" \
  > "$RUN/low_level_ctrl.log" 2>&1 & L=$!
# Scheduling hardening: the actor and the FSM/PD loop are the two processes a
# macOS scheduling hiccup can hurt (measured: rare ~1 s actor stalls per dryrun;
# historic /lowcmd 732 ms stall -> motor watchdog). taskpolicy -t 4 marks the
# throughput tier. sudo was authorized before any live process was started, and
# -n guarantees this safety-critical section can never pause for a password.
for pid in $P $L; do
  taskpolicy -B -p "$pid" 2>/dev/null 1>&2 || true   # -B REMOVES background throttling (-b would ADD it)
done
sudo -n renice -10 -p "$P" "$L" > "$RUN/renice.log" 2>&1 \
  || { echo "ERR: failed to renice actor/controller — stopping stack (see $RUN/renice.log)"; exit 1; }
echo "== scheduler priority set: actor pid $P, controller pid $L (nice -10)"
# Full data log: ros2 bag of every camera/policy/control topic. The depth-camera
# "video" is captured as data here (raw /camera/depth/image_rect_raw + the 58x87
# /rl/depth_image the policy sees) — replay with `ros2 bag play`, or export frames
# offline so we don't load the Mac that's running the 50 Hz policy. Missing topics
# (e.g. /base_velocity when no EKF) are simply skipped.
BAG=""
if [ "$FULL_LOG" != "0" ]; then
  echo "== full logging: ros2 bag -> $RUN/bag (depth video + policy + control topics)"
  ros2 bag record -o "$RUN/bag" \
    /camera/depth/image_rect_raw /rl/depth_image /rl/depth_latent \
    /rl/est_lin_vel /rl/last_action /rl/target_pos \
    /joy /wirelesscontroller /policy_engaged \
    /lowstate /lowcmd /imu/data /base_velocity \
    > "$RUN/rosbag.log" 2>&1 & BAG=$!
fi
# Optional: record the yaw-prediction view headless into the run dir (RECORD_VIEW=1).
# Adds CPU/GPU load to THIS Mac, which is also running the 50 Hz policy — keep fps
# low and watch the monitor for timing jitter on the real robot.
V=""
if [ -n "${RECORD_VIEW:-}" ]; then
  echo "== recording yaw-prediction view -> $RUN/yaw_prediction.mp4 (headless, fps=15)"
  python deploy_mac/diag/view_prediction.py --no-display \
    --record "$RUN/yaw_prediction.mp4" --fps 15 --zero_yaw_pred "$ZERO_YAW_PRED" \
    > "$RUN/view_prediction.log" 2>&1 & V=$!
fi

# Watchdog: alert (terminal + $RUN/watchdog.log) if any stack process dies.
# ALERT-ONLY by design — killing low_level_ctrl on an actor death would
# silence /lowcmd -> motor watchdog -> the 2026-06-19 sag/snap-back. With the
# RL-stall timeout in low_level_ctrl the robot settles itself on actor death;
# this makes the failure visible so the operator releases L1+R1 and stops the
# run. NB: a dead camera/encoder is ALSO covered by input-age guards (2026-07-24):
# depth_camera stops republishing stale frames (max_frame_age_s) and the actor
# stops /rl/target_pos on stale lowstate/latent (--lowstate_max_age_s /
# --latent_max_age_s), so the C++ timeout latches a settle. This alert remains
# the only protection for a FROZEN-but-still-publishing node (staleness is
# detected by silence, not content).
( while sleep 1; do
    for spec in "wireless_to_joy:$W" "depth_camera:$DC" "depth_encoder:$E" \
                "rl_policy:$P" "low_level_ctrl:$L"; do
      name="${spec%%:*}"; pid="${spec#*:}"
      [ -n "$pid" ] || continue   # FUSED=1 has no separate depth_encoder PID
      if ! kill -0 "$pid" 2>/dev/null; then
        msg="!!! $name (pid $pid) DIED $(date +%H:%M:%S) — check $RUN/$name.log — release L1+R1, settle, then Ctrl-C !!!"
        echo; echo "$msg"; echo
        echo "$msg" >> "$RUN/watchdog.log"
        exit 0
      fi
    done
  done ) & WD=$!

echo "== stack up. Foreground = live monitor + lowstate.csv. Ctrl-C to stop."
python deploy_mac/analysis/monitor_deploy.py --csv "$RUN/lowstate.csv"
