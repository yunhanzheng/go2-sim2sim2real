#!/usr/bin/env bash
# Build the C++ deploy_rl_policy package (low_level_ctrl) on macOS via RoboStack.
# Encapsulates the macOS-specific fixes found during bring-up:
#   1. libatomic stub  - ROS C libs pass -latomic (Linux/GCC); macOS clang inlines
#      atomics, so provide an empty libatomic.dylib + point the linker at it.
#   2. install RPATH -> $CONDA_PREFIX/lib  - the binary's @rpath must include the
#      conda ROS libs; baking it in is SIP-safe (DYLD_LIBRARY_PATH gets stripped
#      crossing the shell on macOS).
#   3. link libpython  - a conda ROS lib references Python C-API symbols in the flat
#      namespace; a standalone C++ binary must load libpython or it aborts with
#      "symbol not found in flat namespace '_PyExc_RuntimeError'".
#
# Prereqs:  source deploy_mac/env.sh   (ros_humble + unitree_go messages built)
# Run from the repo root:  ./deploy_mac/lib/build_cpp.sh
set -e
: "${CONDA_PREFIX:?source deploy_mac/env.sh first}"
HERE="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$HERE"

# 1. libatomic stub (idempotent)
if [ ! -f "$CONDA_PREFIX/lib/libatomic.dylib" ]; then
  printf 'void __dummy_atomic(void){}\n' | clang -x c -shared -o "$CONDA_PREFIX/lib/libatomic.dylib" -
  echo "created libatomic stub"
fi

# 3. libpython link name (e.g. python3.12)
PYLIB=$(ls "$CONDA_PREFIX"/lib/libpython3.*.dylib 2>/dev/null | head -1 | sed -E 's#.*/lib(python3\.[0-9]+)\.dylib#\1#')
[ -n "$PYLIB" ] || { echo "libpython not found in $CONDA_PREFIX/lib"; exit 1; }

echo "building realsense2_camera_msgs + deploy_rl_policy (link -l$PYLIB, rpath -> $CONDA_PREFIX/lib)"
colcon build --packages-select realsense2_camera_msgs deploy_rl_policy \
  --cmake-args -DCMAKE_BUILD_TYPE=Release \
    -DPython3_EXECUTABLE="$CONDA_PREFIX/bin/python" \
    -DPython_EXECUTABLE="$CONDA_PREFIX/bin/python" \
    -DCMAKE_EXE_LINKER_FLAGS="-L$CONDA_PREFIX/lib -l$PYLIB" \
    -DCMAKE_SHARED_LINKER_FLAGS="-L$CONDA_PREFIX/lib" \
    -DCMAKE_MODULE_LINKER_FLAGS="-L$CONDA_PREFIX/lib" \
    -DCMAKE_INSTALL_RPATH="$CONDA_PREFIX/lib" \
    -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON

# rosidl's generated Python extensions do not inherit CMAKE_INSTALL_RPATH on
# macOS. Give them a relative path back to this package's install/lib directory
# so imports still work when SIP strips DYLD_LIBRARY_PATH at a script boundary.
while IFS= read -r typesupport_ext; do
  if ! otool -l "$typesupport_ext" | grep -Fq '@loader_path/../../..'; then
    install_name_tool -add_rpath '@loader_path/../../..' "$typesupport_ext"
  fi
done < <(find "$HERE/install/realsense2_camera_msgs/lib" \
  -name '*_s__rosidl_typesupport_*.so' -type f)

echo "done -> RealSense metadata messages + install/deploy_rl_policy/lib/deploy_rl_policy/low_level_ctrl"
echo "re-source the env to pick it up:  source deploy_mac/env.sh"
