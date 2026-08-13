#!/usr/bin/env bash
# Source this to get a ROS 2 Humble (RoboStack) shell on macOS, configured to
# talk to the Unitree Go2 over Ethernet via CycloneDDS.
#
#   source deploy_mac/env.sh
#
# Overridable:
#   GO2_IFACE   ethernet interface to the robot   (default: en13)
#   ROS2_WS     unitree message workspace          (default: $HOME/go2/ros2_ws)
#   ROS_ENV     conda env name                     (default: ros_humble)

ROS_ENV="${ROS_ENV:-ros_humble}"
ROS2_WS="${ROS2_WS:-$HOME/go2/ros2_ws}"
GO2_IFACE="${GO2_IFACE:-en13}"

# 1. conda + RoboStack ros-humble (activation scripts pull in /opt-style ROS setup)
source "$HOME/miniforge3/etc/profile.d/conda.sh"
# Re-running `conda activate` while this RoboStack environment is already
# active re-enters its ament setup hooks under zsh and can lose the helper
# function `ament_zsh_to_array`. Keep sourcing this script idempotent.
if [ "${CONDA_DEFAULT_ENV:-}" != "$ROS_ENV" ]; then
  conda activate "$ROS_ENV"
fi

# 2. overlay the built unitree_go / unitree_api / unitree_hg messages
if [ -n "$ZSH_VERSION" ]; then
  [ -f "$ROS2_WS/install/setup.zsh" ] && source "$ROS2_WS/install/setup.zsh"
else
  [ -f "$ROS2_WS/install/setup.bash" ] && source "$ROS2_WS/install/setup.bash"
fi
[ -d "$ROS2_WS/install/unitree_go" ] || \
  echo "WARN: unitree messages not found in $ROS2_WS (see deploy_mac/README.md to build them)"

# 2b. overlay the local C++ build (deploy_rl_policy: low_level_ctrl), if built
if [ -n "$ZSH_VERSION" ]; then _ENVSH="${(%):-%x}"; else _ENVSH="${BASH_SOURCE[0]}"; fi
_REPO="$(cd "$(dirname "$_ENVSH")/.." && pwd)"
if [ -n "$ZSH_VERSION" ]; then
  [ -f "$_REPO/install/setup.zsh" ] && source "$_REPO/install/setup.zsh"
else
  [ -f "$_REPO/install/setup.bash" ] && source "$_REPO/install/setup.bash"
fi

# 3. DDS: CycloneDDS bound to the NIC connected to the robot (192.168.123.x)
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI="<CycloneDDS><Domain><General><Interfaces><NetworkInterface name=\"${GO2_IFACE}\" priority=\"default\" multicast=\"default\"/></Interfaces></General></Domain></CycloneDDS>"

# 4. torch + OpenMP: the env has multiple OpenMP runtimes (conda libomp + libiomp5
# + torch's bundled libomp); torch aborts on import without this. Safe here because
# the policy pins torch.set_num_threads(1).
export KMP_DUPLICATE_LIB_OK=TRUE

# 5. BLAS/OpenMP thread caps. torch is pinned in-process (set_num_threads(1)), but
# numpy's backend (Accelerate/vecLib on macOS, or OpenBLAS) + OpenMP/MKL otherwise
# spawn one thread PER CORE in EACH of the ~5 deploy Python processes → CPU
# oversubscription and scheduling jitter on the same Mac that drives the 200 Hz
# control + 50 Hz policy loops (a contributor to the /lowcmd stalls). Pin them to 1
# so the deploy stack stays single-threaded per process and the RT loops keep their
# slices. (Override by exporting these before sourcing env.sh if you ever need BLAS
# parallelism for an offline tool.)
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

# 6. low_level_ctrl (C++) links libpython3.12.dylib via an @rpath that a colcon
# rebuild drops (the lib lives in the conda env). Without this dyld fallback the
# FSM node aborts at launch ("Library not loaded: @rpath/libpython3.12.dylib") ->
# no /lowcmd in sim2sim OR real. Shared here so render_sim2sim.sh gets it too
# (previously only run_real_remote.sh set it). Needs the conda env active.
if [ -n "${CONDA_PREFIX:-}" ]; then
  export DYLD_FALLBACK_LIBRARY_PATH="$CONDA_PREFIX/lib:${DYLD_FALLBACK_LIBRARY_PATH:-}"
fi

echo "ros_humble ready | RMW=$RMW_IMPLEMENTATION | iface=$GO2_IFACE | ws=$ROS2_WS"
