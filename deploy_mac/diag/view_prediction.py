#!/usr/bin/env python3
"""Live MuJoCo viewer for the distillation policy during a dry-run / real run.

Run this on the MacBook ALONGSIDE an inference dry-run (run_dryrun_remote.sh /
run_inference_dryrun_camera.sh). It opens an interactive MuJoCo window that shows:

  * the Go2 posed in real time from /lowstate (12 joint angles + IMU quaternion),
  * a picture-in-picture of the normalized depth image the robot actually
    perceives (/rl/depth_image, 58x87),
  * the depth-latent norm |dlat| (health/OOD indicator), foot-contact state,
    policy-engaged status, and topic freshness.

NOTE: heading arrows are drawn for YAW-HEAD students (34-D latent: 32 scan +
2 yaw, e.g. OD/ODFG cliffside w67/w68) — red = predicted delta_yaw (current
step), orange = next step, at world_yaw + 1.5*yaw_pred (how the policy consumes
prop[6:8]). For no-yaw (32-D, L00-flatstep) students yaw_pred stays 0, so no
arrows are drawn.

ZERO-HEADING scenario (--zero_yaw_pred 1, matching the deploy/render flag of the
same name): the policy IGNORES the encoder yaw head and holds prop[6:8]=0, so the
commanded heading is just the robot's current world_yaw. The viewer then draws a
single cyan forward arrow representing that ACTUAL zeroed command (not the unused
prediction) and labels the heading "0° (zeroed)". Pass this whenever the policy
was launched with zero_yaw_pred so the arrows don't mislead.

READ-ONLY: subscribe only. It never publishes /lowcmd or anything that moves the
robot.

The robot base is pinned at the origin (real /lowstate carries no world position),
oriented by the IMU quaternion and dropped so the lowest foot rests on the floor.

Usage (in a second terminal, after `source deploy_mac/env.sh`):

    source deploy_mac/env.sh
    python deploy_mac/diag/view_prediction.py
    # or the wrapper:  ./deploy_mac/diag/view_prediction.sh

Record an MP4 of the live view (exactly what's shown):
    python deploy_mac/diag/view_prediction.py --record run_logs/.../yaw_prediction.mp4
    python deploy_mac/diag/view_prediction.py --record            # auto -> run_logs/viewer/<ts>/
    python deploy_mac/diag/view_prediction.py --no-display --record out.mp4 --seconds 60  # headless

Controls (focus the MuJoCo window):
    left-drag      orbit            mouse wheel / +,-   zoom
    b / t / f      behind / top-down / front preset     y  toggle yaw-lock
    space          pause            r  reset view        q / ESC  quit
"""
import argparse
import signal
import sys
import threading
import time
from pathlib import Path

import numpy as np

# Make `from distill_common import ...` resolve regardless of cwd / PYTHONPATH.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src" / "deploy_rl_policy" / "scripts"))

import cv2
import mujoco
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32MultiArray
from unitree_go.msg import LowState

from distill_common import (  # single source of truth, shared with the deploy nodes
    CONTACT_FORCE_THRESHOLD_N,
    CONTACT_FORCE_THRESHOLD_PER_FOOT_REAL,
    DEPTH_TARGET_H,
    DEPTH_TARGET_W,
    compute_contact_filt,
    mj_quat_to_roll_pitch_yaw,
    parse_contact_threshold,
)

# Foot body names used to drop the base so the robot rests on the floor.
FOOT_BODIES = ("FR_foot", "FL_foot", "RR_foot", "RL_foot")
FOOT_CLEARANCE = 0.022  # approx foot collision radius (m)


# --- arrow helpers (same recipe as distill_runner.py / distill_recorder.py) ---
def _yaw_arrow_mat(theta: float) -> np.ndarray:
    """Rotation taking the arrow's local +Z to the world dir (cos t, sin t, 0)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[0.0, -s, c],
                     [0.0, c, s],
                     [-1.0, 0.0, 0.0]], dtype=np.float64)


def _add_arrow(scene, pos, theta, length, radius, rgba) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom],
        type=int(mujoco.mjtGeom.mjGEOM_ARROW),
        size=np.array([radius, radius, length], dtype=np.float64),
        pos=np.asarray(pos, dtype=np.float64),
        mat=_yaw_arrow_mat(theta).flatten(),
        rgba=np.array(rgba, dtype=np.float32),
    )
    scene.ngeom += 1


def _add_foot_marker(scene, pos, rgba, radius=0.035) -> None:
    """Draw a small sphere at a foot position (world coords) into the scene."""
    if scene.ngeom >= scene.maxgeom:
        return
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom],
        type=int(mujoco.mjtGeom.mjGEOM_SPHERE),
        size=np.array([radius, radius, radius], dtype=np.float64),
        pos=np.asarray(pos, dtype=np.float64),
        mat=np.eye(3, dtype=np.float64).flatten(),
        rgba=np.array(rgba, dtype=np.float32),
    )
    scene.ngeom += 1


class _HzCounter:
    """Rolling 1-second message-rate counter."""
    def __init__(self):
        self._times: list[float] = []

    def tick(self):
        now = time.monotonic()
        self._times.append(now)
        cutoff = now - 1.0
        while self._times and self._times[0] < cutoff:
            self._times.pop(0)

    def hz(self) -> float:
        if len(self._times) < 2:
            return 0.0
        span = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / span if span > 0 else 0.0


class PredictionViewer(Node):
    """Subscribes the dry-run topics; caches the latest snapshot for the renderer."""

    def __init__(self, contact_threshold=CONTACT_FORCE_THRESHOLD_PER_FOOT_REAL):
        super().__init__("view_prediction")
        self._lock = threading.Lock()

        self.qj = None                       # (12,) joint angles, SDK/MJ order
        self.quat = np.array([1.0, 0.0, 0.0, 0.0])   # base (w,x,y,z) from IMU
        self.yaw_pred = np.zeros(2, dtype=np.float32)
        self.dlat_norm = 0.0                 # L2 norm of depth latent[0:32]; ~1.0 healthy, >1.5 warning
        self.depth_norm = None               # (58, 87) float32 in [-0.5, 0.5]
        self.engaged = False
        self.foot_force = np.zeros(4, dtype=np.float32)   # raw foot forces from /lowstate

        # The EXACT contact bool the policy consumes (prop[49:53]): the Isaac-style
        # filtered contact from compute_contact_filt, NOT a raw force threshold.
        # We mirror the deploy node here — same threshold, same OR-with-previous
        # latch — and update it per /lowstate message so it tracks the policy.
        # scalar (one global threshold) or length-4 per-foot (FR, FL, RR, RL)
        self.contact_threshold = np.asarray(contact_threshold, dtype=np.float64)
        self.contact_filt = np.zeros(4, dtype=np.float64)   # bool fed to the policy
        self.last_contact_raw = np.zeros(4, dtype=np.float64)

        self.t_lowstate = 0.0
        self.t_latent = 0.0
        self.t_depth = 0.0
        self.n_lowstate = 0
        self.n_latent = 0
        self.n_depth = 0
        self._hz_lowstate = _HzCounter()
        self._hz_latent = _HzCounter()
        self._hz_depth = _HzCounter()

        # Real robot publishes /lowstate; the sim bridge publishes /mujoco/lowstate.
        # Subscribe to both so this works for a real dry-run AND a sim2sim run.
        self.create_subscription(LowState, "/lowstate", self._on_lowstate, 10)
        self.create_subscription(LowState, "/mujoco/lowstate", self._on_lowstate, 10)
        self.create_subscription(Float32MultiArray, "/rl/depth_latent", self._on_latent, 10)
        self.create_subscription(Image, "/rl/depth_image", self._on_depth, 10)
        self.create_subscription(Bool, "/policy_engaged", self._on_engaged, 10)

        # Constant-latency enforcement diagnostics (2026-07-25 deploy stack).
        # Stored as (values, receive_monotonic); tuple assignment is atomic so
        # the render thread reads without the lock. All optional — the HUD
        # shows "--" until a publisher exists and greys out stale streams:
        #   /rl/depth_latency      [elapsed, wait, achieved, budget, late, offset] ms
        #   /rl/proprio_latency    [lowstate_age, wait, achieved, budget] ms
        #   /rl/depth_clock_offset [offset_ms, transport_floor_ms, baseline_ready]
        #     (real path only — depth_camera_node's Jetson-Mac drift tracker)
        self.depth_lat = None
        self.proprio_lat = None
        self.clock_off = None
        self.create_subscription(
            Float32MultiArray, "/rl/depth_latency",
            lambda m: setattr(self, "depth_lat", (list(m.data), time.monotonic())), 10)
        self.create_subscription(
            Float32MultiArray, "/rl/proprio_latency",
            lambda m: setattr(self, "proprio_lat", (list(m.data), time.monotonic())), 10)
        self.create_subscription(
            Float32MultiArray, "/rl/depth_clock_offset",
            lambda m: setattr(self, "clock_off", (list(m.data), time.monotonic())), 10)

    def _on_lowstate(self, msg: LowState):
        qj = np.fromiter((msg.motor_state[i].q for i in range(12)),
                         dtype=np.float64, count=12)
        quat = np.asarray(msg.imu_state.quaternion, dtype=np.float64)  # (w,x,y,z)
        ff = np.asarray(msg.foot_force, dtype=np.float32)
        with self._lock:
            self.qj = qj
            if quat.shape == (4,) and np.any(quat):
                self.quat = quat
            if ff.shape == (4,):
                self.foot_force = ff
                # exact policy contact bool: filtered, OR-ed with previous raw
                filt, raw_now = compute_contact_filt(
                    ff.astype(np.float64), self.last_contact_raw,
                    threshold_n=self.contact_threshold)
                self.contact_filt = filt
                self.last_contact_raw = raw_now
            self.t_lowstate = time.monotonic()
            self.n_lowstate += 1
            self._hz_lowstate.tick()

    def _on_latent(self, msg: Float32MultiArray):
        n = len(msg.data)
        # L00-flatstep student: 32-D scan latent, no yaw head -> yaw_pred stays
        # zero (heading is forward / +x). Legacy 34-D checkpoints still carry a
        # 2-D yaw tail at [32:34], which we display as before.
        if n in (32, 34):
            arr = np.asarray(msg.data, dtype=np.float32)
            with self._lock:
                self.yaw_pred = arr[32:34] if n == 34 else np.zeros(2, dtype=np.float32)
                self.dlat_norm = float(np.linalg.norm(arr[:32]))
                self.t_latent = time.monotonic()
                self.n_latent += 1
                self._hz_latent.tick()

    def _on_depth(self, msg: Image):
        if msg.encoding != "32FC1" or msg.height != DEPTH_TARGET_H or msg.width != DEPTH_TARGET_W:
            return
        d = np.frombuffer(bytes(msg.data), dtype="<f4").reshape(
            DEPTH_TARGET_H, DEPTH_TARGET_W).copy()
        with self._lock:
            self.depth_norm = d
            self.t_depth = time.monotonic()
            self.n_depth += 1
            self._hz_depth.tick()

    def _on_engaged(self, msg: Bool):
        with self._lock:
            self.engaged = bool(msg.data)

    def snapshot(self):
        with self._lock:
            return (
                None if self.qj is None else self.qj.copy(),
                self.quat.copy(),
                self.yaw_pred.copy(),
                self.dlat_norm,
                None if self.depth_norm is None else self.depth_norm.copy(),
                self.engaged,
                self.foot_force.copy(),
                self.contact_filt.copy(),
                (self.t_lowstate, self.t_latent, self.t_depth),
                (self.n_lowstate, self.n_latent, self.n_depth),
                (self._hz_lowstate.hz(), self._hz_latent.hz(), self._hz_depth.hz()),
            )


# --- rendering ----------------------------------------------------------------
class CamState:
    """Free-orbit camera driven by cv2 mouse/keyboard, lookat pinned to robot."""

    PRESETS = {
        # azimuth, elevation, distance. "behind" is an elevated 3/4 chase view so
        # the horizontal heading arrows aren't seen edge-on; "top" is best for
        # reading the predicted turn direction.
        "behind": (135.0, -35.0, 2.3),
        "top": (90.0, -89.0, 3.0),
        "front": (-45.0, -20.0, 2.3),
    }

    def __init__(self, m):
        self.cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(m, self.cam)
        self.cam.lookat[:] = [0.0, 0.0, 0.30]
        self.reset("behind")
        self._drag = False
        self._lx = 0
        self._ly = 0

    def reset(self, preset="behind"):
        az, el, dist = self.PRESETS[preset]
        self.cam.azimuth = az
        self.cam.elevation = el
        self.cam.distance = dist

    def on_mouse(self, event, x, y, flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._drag, self._lx, self._ly = True, x, y
        elif event == cv2.EVENT_LBUTTONUP:
            self._drag = False
        elif event == cv2.EVENT_MOUSEMOVE and self._drag:
            self.cam.azimuth -= (x - self._lx) * 0.4
            self.cam.elevation = float(np.clip(self.cam.elevation - (y - self._ly) * 0.4, -89.0, 89.0))
            self._lx, self._ly = x, y
        elif event == cv2.EVENT_MOUSEWHEEL:
            # wheel delta lives in the high word of `flags` (no getMouseWheelDelta
            # in this cv2 build); positive = scroll up = zoom in.
            delta = (flags >> 16) & 0xFFFF
            if delta > 0x7FFF:
                delta -= 0x10000
            self.zoom(0.9 if delta > 0 else 1.1)

    def zoom(self, factor):
        self.cam.distance = float(np.clip(self.cam.distance * factor, 0.6, 12.0))


def _ground_robot(m, d):
    """Drop the base so the lowest foot sits ~on the floor (real lowstate has no z)."""
    mujoco.mj_forward(m, d)
    zs = []
    for name in FOOT_BODIES:
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid >= 0:
            zs.append(float(d.xpos[bid, 2]))
    if zs:
        d.qpos[2] += FOOT_CLEARANCE - min(zs)
        mujoco.mj_forward(m, d)


def _depth_pip(depth_norm, pip_w, pip_h):
    """Normalized depth (58,87) in [-0.5,0.5] -> BGR greyscale image (pip_h,pip_w,3).

    Matches Isaac training's depth display (legged_robot.py:281, ``cv2.imshow(...,
    depth_buffer + 0.5)``): the normalized depth is shifted to [0,1] and shown as
    greyscale -- near=black, far=white -- with no colormap.
    """
    u = ((depth_norm + 0.5).clip(0.0, 1.0) * 255.0).astype(np.uint8)  # near=0(black) far=255(white)
    grey = cv2.resize(u, (pip_w, pip_h), interpolation=cv2.INTER_NEAREST)
    return cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)


def _put(img, text, org, color, scale=0.6, thick=2):
    """Shadow-outlined text for contrast (BGR)."""
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def _hud_panel(img, x, y, w, h, alpha=0.55):
    """Draw a semi-transparent dark rect as HUD background."""
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def _dlat_bar(img, x, y, value, w=160, h=12):
    """Horizontal gauge for |dlat|. Green < 1.4, amber < 1.7, red ≥ 1.7. Max=2.5."""
    MAX = 2.5
    cv2.rectangle(img, (x, y), (x + w, y + h), (60, 60, 60), -1)
    fill = int(min(value / MAX, 1.0) * w)
    color = (60, 200, 60) if value < 1.4 else ((40, 180, 250) if value < 1.7 else (40, 40, 235))
    if fill > 0:
        cv2.rectangle(img, (x, y), (x + fill, y + h), color, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (100, 100, 100), 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", type=Path,
                    default=REPO_ROOT / "resources" / "go2" / "scene_flat.xml",
                    help="MuJoCo scene (default flat; the real world isn't modeled, "
                         "the depth PiP is what the robot truly sees).")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30, help="render/display rate cap")
    ap.add_argument("--base_z", type=float, default=0.40,
                    help="nominal base height before grounding (m)")
    ap.add_argument("--stale_s", type=float, default=0.5,
                    help="flag a topic stale in the HUD after this many seconds")
    ap.add_argument("--record", nargs="?", const="__auto__", default=None,
                    help="Save an MP4 of the live view (exactly what's shown). Give a "
                         "path, or pass --record with no value to auto-save to "
                         "run_logs/viewer/<ts>/yaw_prediction_<ts>.mp4.")
    ap.add_argument("--no-display", action="store_true", dest="no_display",
                    help="Headless: render+record without a window (for backgrounded "
                         "logging from the run scripts). Implies --record.")
    ap.add_argument("--seconds", type=float, default=0.0,
                    help="Auto-stop after N seconds (0 = until window closed / Ctrl-C).")
    ap.add_argument("--cam", choices=["behind", "top", "front"], default="behind",
                    help="Camera preset (the headless default, and the 'r' reset target).")
    ap.add_argument("--zero_yaw_pred", type=int, choices=[0, 1], default=0,
                    help="Match the deploy/render zero_yaw_pred flag: when 1, the policy "
                         "holds prop[6:8]=0, so draw a single forward arrow at the robot's "
                         "current heading (the actual zeroed command) instead of the "
                         "unused encoder yaw prediction, and label the heading 'zeroed'.")
    _per_foot_default = ",".join(f"{v:g}" for v in CONTACT_FORCE_THRESHOLD_PER_FOOT_REAL)
    ap.add_argument("--contact_threshold", type=str, default=_per_foot_default,
                    help="Foot-force threshold(s) for the contact bool fed to the policy "
                         "(prop[49:53]). Either one global value, or four comma-separated "
                         "per-foot values in FR,FL,RR,RL order. MUST match the deploy "
                         f"node. Real default per-foot = {_per_foot_default} (from "
                         f"2026-06-19 forensics); sim2sim uses {CONTACT_FORCE_THRESHOLD_N:g}.")
    args = ap.parse_args()
    try:
        args.contact_threshold = parse_contact_threshold(args.contact_threshold)
    except ValueError as e:
        ap.error(str(e))
    if args.no_display and args.record is None:
        args.record = "__auto__"   # headless with no output would render to nothing

    GREEN, GRAY, AMBER, RED, ORANGE, CYAN = (
        (60, 200, 60), (190, 190, 190), (40, 180, 250),
        (40, 40, 235), (20, 150, 250), (230, 200, 60))

    print(f"[mj] loading {args.scene}")
    m = mujoco.MjModel.from_xml_path(str(args.scene))
    d = mujoco.MjData(m)
    renderer = mujoco.Renderer(m, height=args.height, width=args.width)

    default_qpos = m.qpos0.copy()  # fallback pose until /lowstate arrives

    rclpy.init()
    node = PredictionViewer(contact_threshold=args.contact_threshold)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spin = threading.Thread(target=executor.spin, daemon=True)
    spin.start()
    print("[ros] subscribing /lowstate /mujoco/lowstate /rl/depth_latent "
          "/rl/depth_image /policy_engaged")

    cams = CamState(m)
    cams.reset(args.cam)
    win = "Go2 depth-encoder yaw-prediction validator"
    if not args.no_display:
        cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(win, cams.on_mouse)

    # --- optional MP4 recording of the composited live view ----------
    writer = None
    record_path = None
    if args.record is not None:
        if args.record == "__auto__":
            ts = time.strftime("%Y-%m-%d_%H-%M-%S")
            record_path = REPO_ROOT / "runs" / "viewer" / ts / f"yaw_prediction_{ts}.mp4"
        else:
            record_path = Path(args.record)
        record_path.parent.mkdir(parents=True, exist_ok=True)
        # Prefer H.264 (avc1): mp4v (MPEG-4 Part 2) files are unplayable in
        # QuickTime/browsers and needed a manual re-encode after every run
        # (2026-07-26). OpenCV builds without an H.264 encoder fail
        # isOpened(), so fall back to mp4v with a loud hint.
        writer = None
        for fourcc in ("avc1", "mp4v"):
            writer = cv2.VideoWriter(str(record_path),
                                     cv2.VideoWriter_fourcc(*fourcc),
                                     float(args.fps), (args.width, args.height))
            if writer.isOpened():
                if fourcc == "mp4v":
                    print("[record] WARN: no H.264 encoder in this OpenCV — "
                          "writing mp4v (re-encode with ffmpeg for playback)")
                break
            writer = None
        if writer is None:
            print(f"[record] WARN: could not open VideoWriter for {record_path}")
        else:
            print(f"[record] saving live view -> {record_path} "
                  f"({args.width}x{args.height}@{args.fps}fps)")

    pip_w, pip_h = 4 * DEPTH_TARGET_W, 4 * DEPTH_TARGET_H  # 348 x 232
    paused = False
    yaw_lock = False
    frame_dt = 1.0 / max(1, args.fps)
    t_start = time.monotonic()

    # Graceful SIGTERM (the run scripts `kill` this when headless) so the MP4 is
    # finalized via the finally-block instead of left truncated/unplayable.
    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda _s, _f: stop.__setitem__("flag", True))

    try:
        while rclpy.ok():
            if stop["flag"]:
                break
            t0 = time.monotonic()
            qj, quat, yaw_pred, dlat_norm, depth_norm, engaged, foot_force, contact_filt, ts, ns, hzs = node.snapshot()

            # --- pose the robot from the latest lowstate -----------------
            d.qpos[:] = default_qpos
            d.qvel[:] = 0.0
            d.qpos[0:3] = [0.0, 0.0, args.base_z]
            d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
            world_yaw = 0.0
            if qj is not None:
                d.qpos[7:19] = qj
                if yaw_lock:
                    # keep only roll/pitch (gravity-referenced), face world +x
                    roll, pitch, _ = mj_quat_to_roll_pitch_yaw(quat)
                    cy, cp, cr = np.cos(0), np.cos(pitch / 2), np.cos(roll / 2)
                    sy, sp, sr = np.sin(0), np.sin(pitch / 2), np.sin(roll / 2)
                    d.qpos[3] = cr * cp * cy + sr * sp * sy
                    d.qpos[4] = sr * cp * cy - cr * sp * sy
                    d.qpos[5] = cr * sp * cy + sr * cp * sy
                    d.qpos[6] = cr * cp * sy - sr * sp * cy
                else:
                    d.qpos[3:7] = quat
                _ground_robot(m, d)
                _, _, world_yaw = mj_quat_to_roll_pitch_yaw(d.qpos[3:7])
            else:
                mujoco.mj_forward(m, d)

            # --- render robot --------------------------------------------
            renderer.update_scene(d, camera=cams.cam)

            # --- heading arrow(s) -----------------------------------------
            # Zero-heading scenario (--zero_yaw_pred, matching the deploy flag):
            # the policy IGNORES the encoder yaw head and holds prop[6:8]=0, so the
            # commanded heading is just the robot's current world_yaw. Draw a single
            # cyan forward arrow at world_yaw so the marker reflects the ACTUAL
            # zeroed command rather than the unused prediction.
            # Otherwise (yaw-head students): red = predicted delta_yaw (current
            # step), orange = next step, at world_yaw + 1.5*yaw_pred (exactly how the
            # policy consumes prop[6:8], see rl_distill_policy / distill_runner).
            # yaw_pred stays zero for no-yaw (32-D) students, so that branch is a
            # no-op for them.
            if qj is not None and args.zero_yaw_pred:
                arrow_pos = d.qpos[:3].copy()
                arrow_pos[2] += 0.20
                _add_arrow(renderer.scene, arrow_pos, world_yaw,
                           length=0.45, radius=0.018, rgba=(0.30, 0.75, 1.00, 1.0))
            elif qj is not None and yaw_pred is not None and \
                    float(np.linalg.norm(yaw_pred)) > 1e-6:
                arrow_pos = d.qpos[:3].copy()
                arrow_pos[2] += 0.20
                dyaw_now = 1.5 * float(yaw_pred[0])
                dyaw_next = 1.5 * float(yaw_pred[1])
                _add_arrow(renderer.scene, arrow_pos, world_yaw + dyaw_now,
                           length=0.45, radius=0.018, rgba=(0.95, 0.20, 0.15, 1.0))
                _add_arrow(renderer.scene, arrow_pos, world_yaw + dyaw_next,
                           length=0.35, radius=0.012, rgba=(1.00, 0.65, 0.10, 0.85))

            # --- foot-contact markers: the EXACT bool fed to the policy ----
            # Green sphere = in contact (prop bit = 1), red = swing (bit = 0).
            # Only meaningful once a real lowstate has posed the robot.
            if qj is not None:
                for i, name in enumerate(FOOT_BODIES):
                    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
                    if bid < 0:
                        continue
                    in_contact = bool(contact_filt[i] >= 0.5)
                    rgba = (0.2, 0.85, 0.2, 0.9) if in_contact else (0.9, 0.15, 0.15, 0.9)
                    _add_foot_marker(renderer.scene, d.xpos[bid], rgba)

            now = time.monotonic()
            latent_fresh = (now - ts[1]) < args.stale_s and ns[1] > 0
            frame = renderer.render()  # RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # --- depth picture-in-picture (top-right) --------------------
            if depth_norm is not None:
                pip = _depth_pip(depth_norm, pip_w, pip_h)
                mx = 12
                y0, y1 = mx, mx + pip_h
                x0, x1 = args.width - mx - pip_w, args.width - mx
                frame[y0 - 3:y1 + 3, x0 - 3:x1 + 3] = 0
                frame[y0:y1, x0:x1] = pip
                _put(frame, "DEPTH the robot sees  (near=black  far=white)",
                     (x0, y1 + 22), (235, 235, 235), 0.5, 1)

            # --- HUD (top-left) ------------------------------------------
            lowstate_fresh = (now - ts[0]) < args.stale_s and ns[0] > 0
            depth_fresh = (now - ts[2]) < args.stale_s and ns[2] > 0
            elapsed = now - t_start

            # semi-transparent background panel
            _hud_panel(frame, 8, 8, 480, 210)

            # title + elapsed
            _put(frame, "Go2  live viewer", (16, 34), AMBER, 0.7, 2)
            _put(frame, f"t = {elapsed:6.1f} s", (290, 34), GRAY, 0.6, 1)

            # policy engagement
            eng_txt = "ENGAGED" if engaged else "not engaged"
            _put(frame, f"policy  {eng_txt}", (16, 62), GREEN if engaged else GRAY, 0.6, 1)

            # heading-command mode: what the policy actually feeds into prop[6:8].
            if args.zero_yaw_pred:
                _put(frame, "heading  0deg (zeroed)", (250, 62), CYAN, 0.55, 1)
            elif yaw_pred is not None and float(np.linalg.norm(yaw_pred)) > 1e-6:
                _put(frame, f"heading  {np.degrees(1.5 * float(yaw_pred[0])):+.0f}deg (pred)",
                     (250, 62), ORANGE, 0.55, 1)

            # roll / pitch from IMU
            if qj is not None:
                roll, pitch, _ = mj_quat_to_roll_pitch_yaw(quat)
                _put(frame, f"roll {np.degrees(roll):+.1f}°  pitch {np.degrees(pitch):+.1f}°",
                     (16, 88), GRAY, 0.55, 1)

            # |dlat| gauge
            if latent_fresh:
                dlat_color = GREEN if dlat_norm < 1.4 else (AMBER if dlat_norm < 1.7 else RED)
                _put(frame, f"|dlat|  {dlat_norm:.3f}", (16, 116), dlat_color, 0.6, 1)
                _dlat_bar(frame, 150, 103, dlat_norm)
                _put(frame, "healthy < 1.4    OOD > 1.7", (16, 136), GRAY, 0.42, 1)
            else:
                _put(frame, "waiting for /rl/depth_latent ...", (16, 116), CYAN, 0.55, 1)

            # foot contacts
            if lowstate_fresh:
                symbols = ["●" if float(contact_filt[i]) >= 0.5 else "○" for i in range(4)]
                cols = [GREEN if float(contact_filt[i]) >= 0.5 else RED for i in range(4)]
                labels = ["FR", "FL", "RR", "RL"]
                x_cur = 16
                _put(frame, "feet", (x_cur, 164), GRAY, 0.55, 1)
                x_cur += 52
                for i in range(4):
                    _put(frame, f"{labels[i]}{symbols[i]}", (x_cur, 164), cols[i], 0.55, 1)
                    x_cur += 68
                thr = np.broadcast_to(node.contact_threshold, (4,))
                _put(frame,
                     f"raw [{foot_force[0]:.0f} {foot_force[1]:.0f} "
                     f"{foot_force[2]:.0f} {foot_force[3]:.0f}]  "
                     f"thr [{thr[0]:.0f} {thr[1]:.0f} {thr[2]:.0f} {thr[3]:.0f}]",
                     (16, 188), GRAY, 0.42, 1)

            # topic status bar (bottom-left)
            _hud_panel(frame, 8, args.height - 88, 420, 80)

            def _topic_line(label, fresh, n, hz, y):
                dot = "●" if fresh else "○"
                col = GREEN if fresh else RED
                _put(frame, f"{dot} {label}  {hz:.1f} Hz  ({n})", (16, y), col, 0.45, 1)

            _topic_line("/lowstate      ", lowstate_fresh, ns[0], hzs[0], args.height - 66)
            _topic_line("/rl/depth_latent", latent_fresh,  ns[1], hzs[1], args.height - 46)
            _topic_line("/rl/depth_image ", depth_fresh,   ns[2], hzs[2], args.height - 26)

            # latency-enforcement status (bottom-right) — mirrors the strip in
            # depth_viewer_node: achieved vs enforced budget for the depth and
            # proprio pads, plus the Jetson-Mac clock offset (real path only).
            _hud_panel(frame, args.width - 268, args.height - 88, 260, 80)

            def _lat_line(stored, fmt, label, y):
                if stored is None:
                    _put(frame, f"{label} --", (args.width - 256, y), GRAY, 0.45, 1)
                    return
                values, recv_t = stored
                text, col = fmt(values)
                if now - recv_t > 2.0:
                    col = GRAY
                _put(frame, text, (args.width - 256, y), col, 0.45, 1)

            def _fmt_depth(v):
                achieved, budget, late = v[2], v[3], int(v[4])
                err = abs(achieved - budget)
                col = (GREEN if err <= 5.0 else AMBER if err <= 15.0 else RED)
                if budget <= 0.0:
                    col = AMBER   # enforcement off — natural latency
                text = f"depth {achieved:5.1f}/{budget:.0f}ms"
                if late:
                    text += f" late:{late}"
                    col = RED
                return text, col

            def _fmt_clock(v):
                offset, ready = v[0], v[2] >= 1.0
                col = (GREEN if abs(offset) < 30.0 else AMBER) if ready else GRAY
                return (f"clk   {offset:+5.1f}ms" + ("" if ready else " (warmup)"), col)

            def _fmt_prop(v):
                achieved, budget = v[2], v[3]
                err = abs(achieved - budget)
                col = (GREEN if err <= 3.0 else AMBER if err <= 8.0 else RED)
                if budget <= 0.0:
                    col = AMBER
                return f"prop  {achieved:4.1f}/{budget:.0f}ms", col

            _lat_line(node.depth_lat,   _fmt_depth, "depth", args.height - 66)
            _lat_line(node.clock_off,   _fmt_clock, "clk",   args.height - 46)
            _lat_line(node.proprio_lat, _fmt_prop,  "prop",  args.height - 26)

            if qj is None:
                _put(frame, "waiting for /lowstate ... (is the dry-run / robot up?)",
                     (16, args.height // 2), CYAN, 0.8, 2)
            if yaw_lock:
                _put(frame, "[yaw-lock]", (args.width - 170, args.height - 26), CYAN, 0.5, 1)
            if paused:
                _put(frame, "[paused]", (args.width // 2 - 60, 40), CYAN, 0.8, 2)

            # --- record exactly what's shown -----------------------------
            if writer is not None and not paused:
                writer.write(frame)

            # --- display + input (skipped when headless) -----------------
            if not args.no_display:
                cv2.imshow(win, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                elif key == ord(" "):
                    paused = not paused
                elif key == ord("b"):
                    cams.reset("behind")
                elif key == ord("t"):
                    cams.reset("top")
                elif key == ord("f"):
                    cams.reset("front")
                elif key == ord("r"):
                    cams.reset(args.cam)
                elif key == ord("y"):
                    yaw_lock = not yaw_lock
                elif key in (ord("+"), ord("=")):
                    cams.zoom(0.9)
                elif key in (ord("-"), ord("_")):
                    cams.zoom(1.1)
                if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                    break

            # --- auto-stop after --seconds -------------------------------
            if args.seconds > 0 and (time.monotonic() - t_start) >= args.seconds:
                break

            if paused:
                time.sleep(0.05)
            else:
                dt = time.monotonic() - t0
                if dt < frame_dt:
                    time.sleep(frame_dt - dt)
    except KeyboardInterrupt:
        pass
    finally:
        if not args.no_display:
            cv2.destroyAllWindows()
        if writer is not None:
            writer.release()
            print(f"[record] wrote {record_path}")
        executor.shutdown()
        spin.join(timeout=1.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        print("[done] viewer closed")


if __name__ == "__main__":
    main()
