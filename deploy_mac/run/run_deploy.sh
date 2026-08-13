#!/usr/bin/env bash
# REAL lay <-> stand via the C++ low_level_ctrl FSM, driven by the Go2 remote.
# *** THIS SENDS /lowcmd — THE ROBOT MOVES. ***
#
# Prereqs (do ALL before running):
#   - robot HOISTED / tethered, clear space
#   - sport mode OFF  (~/go2/disable_sport.py)
#   - hand on power; e-stop = L2+R2 (quits the controller)
#
# As soon as it starts, low_level_ctrl stiffens the legs to the LAY pose (kp=50) — expect
# a small twitch. Then on the remote:  B -> stand,  A -> lay.  Ctrl-C (or L2+R2) to stop.
# Do NOT press L1+R1 (engage) in this run — no policy is loaded; this is lay<->stand only.
# Logs everything to run_logs/real/<timestamp>/.
# (no `set -u`: conda activation scripts reference unset vars)
HERE="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$HERE"
source deploy_mac/env.sh

TS="$(date +%Y-%m-%d_%H-%M-%S)"
# No policy runs here (lay<->stand only), so it files under the _no-policy group
# of the same run_logs/real/<policy>/<timestamp>/ layout.
RUN="run_logs/real/_no-policy/$TS"
mkdir -p "$RUN"
{ echo "timestamp:   $TS"
  echo "mode:        REAL lay<->stand (low_level_ctrl, is_simulation:=false)"
  echo "gains:       transition kp=50/kd=1 ; policy kp=35/kd=0.6"
  echo "iface:       ${GO2_IFACE:-en13}"
} > "$RUN/meta.txt"
# git provenance: commit/branch/dirty-state in meta.txt + git_diff.patch (uncommitted changes)
./deploy_mac/lib/log_git_state.sh "$RUN"

cleanup() {
  echo; echo "stopping low_level_ctrl + wireless_to_joy ..."
  kill "${W:-}" "${L:-}" 2>/dev/null
  ./deploy_mac/kill_ros.sh >/dev/null 2>&1
  echo "logs -> $RUN/  (meta.txt, git_diff.patch, low_level_ctrl.log, wireless_to_joy.log, lowstate.csv)"
}
trap cleanup EXIT INT TERM

echo "==================================================================="
echo " REAL RUN -> $RUN"
echo " Robot WILL move.  HOISTED?  sport mode OFF?  hand on power?"
echo " Remote:  B = stand    A = lay    L2+R2 = quit   (do NOT press L1+R1)"
echo "==================================================================="

python deploy_mac/run/wireless_to_joy.py > "$RUN/wireless_to_joy.log" 2>&1 & W=$!
ros2 run deploy_rl_policy low_level_ctrl \
  --ros-args -p is_simulation:=false -p policy_kp:=35.0 -p policy_kd:=0.6 \
  > "$RUN/low_level_ctrl.log" 2>&1 & L=$!

# foreground: live status to YOUR terminal + lowstate trajectory log
python deploy_mac/analysis/monitor_deploy.py --csv "$RUN/lowstate.csv"
