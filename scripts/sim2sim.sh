#!/usr/bin/env bash
# Wrapper around `ros2 launch deploy_rl_policy sim2sim.launch.py` that:
#   1. cleans up zombie ROS nodes + stale Fast DDS shm from prior crashed runs
#   2. runs the launch in its own process group so SIGINT / shell-close /
#      OOM all terminate the entire group atomically — no orphans
#
# Usage: scripts/sim2sim.sh [launch args]
#   e.g. scripts/sim2sim.sh scene:=scene_flat.xml fake_joy:=true vgl:=true

set -uo pipefail

cleanup_stale() {
    # Kill leftover deploy nodes that aren't owned by an active launch.
    pkill -9 -x joy_node 2>/dev/null || true
    pkill -9 -f "deploy_rl_policy/lib/deploy_rl_policy/(low_level_ctrl|mujoco_simulator|ep_rl_policy|fake_joy)" 2>/dev/null || true
    # Remove unreferenced Fast DDS shm segments. Files held by a live
    # process (lsof returns 0) are skipped.
    for f in /dev/shm/fastrtps_* /dev/shm/sem.fastrtps_*; do
        [ -e "$f" ] || continue
        lsof "$f" >/dev/null 2>&1 || rm -f "$f"
    done
}

cleanup_stale

# `setsid` puts the launch in its own session/process group, distinct from the
# shell's. Trap SIGINT/SIGTERM/EXIT in this wrapper and propagate via negative
# PID (= signal entire group) so every child dies together.
setsid ros2 launch deploy_rl_policy sim2sim.launch.py "$@" &
PGID=$!
trap 'kill -TERM -"$PGID" 2>/dev/null; sleep 1; kill -KILL -"$PGID" 2>/dev/null; cleanup_stale' INT TERM EXIT
wait "$PGID"
