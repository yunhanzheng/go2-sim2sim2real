#!/usr/bin/env python3
"""Video recorder for the ROS distillation dry-run.

Two-phase design:
  1. CAPTURE phase — at every /rl/target_pos tick (50 Hz in lockstep), snapshot
     (qpos, cached yaw_pred, latest depth image) into an in-memory list. This
     work is cheap; no rendering happens here so we cannot back-pressure the
     policy/simulator pipeline.
  2. RENDER phase — once we have target_frames snapshots, render each frame at
     leisure with mujoco.Renderer + heading arrows + depth PiP, and write the
     MP4. Then shut down.

This guarantees every video frame represents *exactly* one policy tick, so an
N-frame video at 50 fps plays back as N/50 sim-seconds at real-time speed,
without the wall-clock-timer jitter the previous design suffered from.
"""
import os
os.environ.setdefault("MUJOCO_GL", "glfw")

import argparse
import json
from pathlib import Path

import numpy as np
import rclpy
import torch
import torch.nn.functional as F
from rclpy.node import Node
from sensor_msgs.msg import Image, Joy
from std_msgs.msg import Bool, Float32MultiArray

import mujoco

try:
    import cv2  # for text overlay
    _HAVE_CV2 = True
except ImportError:
    _HAVE_CV2 = False

from distill_common import (
    DEPTH_TARGET_H, DEPTH_TARGET_W, DEPTH_LATENT_DIM,
    YAW_PRED_MULTIPLIER, mj_quat_to_roll_pitch_yaw,
)


# --- arrow helpers (lifted from distill_runner.py:101-126) ---------------
def _yaw_arrow_mat(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[0.0, -s,  c],
                     [0.0,  c,  s],
                     [-1.0, 0.0, 0.0]], dtype=np.float64)


def _add_arrow(scene, pos: np.ndarray, theta: float, length: float,
               radius: float, rgba: tuple) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        g,
        type=int(mujoco.mjtGeom.mjGEOM_ARROW),
        size=np.array([radius, radius, length], dtype=np.float64),
        pos=pos.astype(np.float64),
        mat=_yaw_arrow_mat(theta).flatten(),
        rgba=np.array(rgba, dtype=np.float32),
    )
    scene.ngeom += 1


class DistillRecorder(Node):
    def __init__(self, scene: Path, out_path: Path, duration_s: float,
                 fps: int, width: int, height: int, camera: str,
                 lin_vel_max: float, title: str = "",
                 metrics_only: bool = False,
                 clean_frame: bool = False,
                 end_on_still: bool = True,
                 still_hold_s: float = 1.0,
                 still_eps_m: float = 0.02):
        super().__init__("distill_recorder")
        self.scene = scene
        self.out_path = out_path
        self.duration_s = duration_s
        self.fps = fps
        self.width = width
        self.height = height
        self.camera = camera
        self.lin_vel_max = lin_vel_max
        self.title = title
        self.metrics_only = metrics_only
        self.clean_frame = bool(clean_frame)
        self.target_frames = int(round(duration_s * fps))

        # ---- early finish: stop capturing once the run is actually OVER -------
        # Whichever comes first:
        #   1. the policy has DISENGAGED (after having been engaged) and the base
        #      has come to rest — i.e. the laydown, or whatever the FSM did after
        #      the policy released, has finished. This is the end of a
        #      lay->stand->engage->laydown clip.
        #   2. the policy is still engaged but the command is back to zero and the
        #      base has stopped, AND a forward command was seen earlier in the run.
        #      That is the return-x hard stop: cut at the standstill rather than
        #      filming the laydown that follows. The forward-command gate is what
        #      keeps the 2 s standstill PREFIX from ending the clip immediately.
        # Stillness is measured on the full base xyz over the trailing
        # still_hold_s, so a descending laydown counts as motion.
        # Without this the recorder always filled duration_s * fps frames, so an
        # early return-x stop left many seconds of a lying, idle robot on the end
        # of every clip.
        self.end_on_still = bool(end_on_still)
        self.still_hold_s = float(still_hold_s)
        self.still_eps_m = float(still_eps_m)
        self._was_engaged = False
        self._saw_forward_cmd = False
        self._disengage_frame = None
        self._still_frames = 0
        self._still_need = max(1, int(round(still_hold_s * fps)))
        self._finished = False

        # Cached snoop state
        self._latest_qpos = None
        self._cached_yaw_pred = np.zeros(2, dtype=np.float32)
        self._last_depth_norm = None  # (58, 87)
        self._latest_contact = None   # /contact: 4-vec foot-force magnitudes (FL,FR,RL,RR)
        self._latest_lift_phase = 0   # /lift_phase: lift-test phase index (0 when off)
        self._latest_dlat = 0.0       # |dlat| = L2 norm of /rl/depth_latent (in-dist ~1.6)
        self._recording = False
        # FSM state + joystick command for the on-screen text overlay.
        self._policy_engaged = False
        self._joy_axes = []
        self._joy_buttons = []

        # CAPTURE phase storage: (qpos[19], yaw_pred[2], depth[58,87], cmd_vx,
        # engaged, lb_rb_held). One entry per video frame.
        self._snapshots = []
        # Parallel per-frame foot-force log, index-aligned with _snapshots.
        # Each entry is a 4-vec in policy order FR,FL,RR,RL (Newtons).
        self._contact_snaps = []
        # Parallel per-frame lift-test phase index, index-aligned with _snapshots.
        self._lift_phase_snaps = []
        # Parallel per-frame |dlat| (depth-latent norm), index-aligned with _snapshots.
        self._dlat_snaps = []

        self.create_subscription(Float32MultiArray, "/mujoco/pos", self._on_qpos, 10)
        self.create_subscription(Float32MultiArray, "/contact", self._on_contact, 10)
        self.create_subscription(Float32MultiArray, "/lift_phase", self._on_lift_phase, 10)
        self.create_subscription(Float32MultiArray, "/rl/depth_latent",
                                 self._on_depth_latent, 10)
        self.create_subscription(Image, "/rl/depth_image", self._on_depth, 10)
        # /rl/target_pos arrives only during the engaged phase (RL inference is
        # FSM-gated), so we cannot use it as the capture trigger. Capture on a
        # wall-clock timer at fps Hz instead, starting from boot.
        self.create_subscription(Bool, "/policy_engaged", self._on_engaged, 10)
        self.create_subscription(Joy, "/joy", self._on_joy, 10)
        self._capture_timer = self.create_timer(1.0 / fps, self._capture_tick)

        self.get_logger().info(
            f"distill_recorder: cam={camera} {width}x{height}@{fps}fps "
            f"target {self.target_frames} frames (lin_vel_max={lin_vel_max})"
        )

    # ---------------- subscriptions ----------------------------------
    def _on_qpos(self, msg: Float32MultiArray):
        if len(msg.data) >= 19:
            self._latest_qpos = np.asarray(msg.data[:19], dtype=np.float64)

    def _on_depth_latent(self, msg: Float32MultiArray):
        # |dlat| = L2 norm of the depth-encoder output (matches
        # measured on real bags): in-distribution ~1.6, off-distribution higher.
        values = np.asarray(msg.data, dtype=np.float64)
        if values.size > 0:
            self._latest_dlat = float(np.linalg.norm(values))
        if values.size >= DEPTH_LATENT_DIM + 2:
            self._cached_yaw_pred = values[
                DEPTH_LATENT_DIM:DEPTH_LATENT_DIM + 2
            ].astype(np.float32)
        else:
            self._cached_yaw_pred.fill(0.0)

    def _on_depth(self, msg: Image):
        if msg.encoding != "32FC1" or msg.height != DEPTH_TARGET_H or msg.width != DEPTH_TARGET_W:
            return
        self._last_depth_norm = np.frombuffer(
            msg.data, dtype=np.float32
        ).reshape(DEPTH_TARGET_H, DEPTH_TARGET_W).copy()

    def _on_contact(self, msg: Float32MultiArray):
        # mujoco_simulator publishes /contact as foot-force magnitudes in
        # go2.xml sensor order FL,FR,RL,RR (float Newtons, not rounded).
        if len(msg.data) >= 4:
            self._latest_contact = np.asarray(msg.data[:4], dtype=np.float64)

    def _on_lift_phase(self, msg: Float32MultiArray):
        if len(msg.data) >= 1:
            self._latest_lift_phase = int(round(float(msg.data[0])))

    def _on_engaged(self, msg: Bool):
        self._policy_engaged = bool(msg.data)

    def _on_joy(self, msg: Joy):
        self._joy_axes = list(msg.axes)
        self._joy_buttons = list(msg.buttons)

    def _read_cmd_vx(self) -> float:
        """Mirror rl_distill_policy._read_lin_vel_x to compute on-screen cmd."""
        if not self._policy_engaged:
            return 0.0
        if len(self._joy_axes) < 2 or len(self._joy_buttons) < 6:
            return 0.0
        if not (self._joy_buttons[4] and self._joy_buttons[5]):  # LB+RB
            return 0.0
        # joy_sequencer writes the forward command into axis 1 (left-Y); the
        # XboxController class also exposes get_left_stick() but we read raw.
        return self.lin_vel_max * float(self._joy_axes[1])

    def _fsm_label(self) -> str:
        if self._policy_engaged:
            return "engaged"
        if self._latest_qpos is None:
            return "init"
        # Crude lay/stand split by base z: standing pose is ~0.30 m, lay ~0.10 m.
        if float(self._latest_qpos[2]) > 0.22:
            return "stand"
        return "lay"

    def _capture_tick(self):
        if not self._recording:
            self._recording = True
            self.get_logger().info("recording started (capture timer)")
        if self._latest_qpos is None:
            return
        if len(self._snapshots) >= self.target_frames:
            return
        depth_copy = None if self._last_depth_norm is None else self._last_depth_norm.copy()
        cmd_vx = self._read_cmd_vx()
        engaged = self._policy_engaged
        fsm = self._fsm_label()
        # /contact is FL,FR,RL,RR (go2.xml sensor order); reorder to policy
        # order FR,FL,RR,RL (same as LowState.foot_force) for per-foot logging.
        if self._latest_contact is None:
            contact_pol = np.full(4, np.nan, dtype=np.float64)
        else:
            contact_pol = self._latest_contact[[1, 0, 3, 2]]
        self._snapshots.append((
            self._latest_qpos.copy(),
            self._cached_yaw_pred.copy(),
            depth_copy,
            cmd_vx,
            engaged,
            fsm,
        ))
        self._contact_snaps.append(contact_pol)
        self._lift_phase_snaps.append(self._latest_lift_phase)
        self._dlat_snaps.append(self._latest_dlat)
        # ---- early-finish bookkeeping (see __init__) -------------------------
        if self.end_on_still and not self._finished:
            if engaged and abs(cmd_vx) > 1e-6:
                self._saw_forward_cmd = True
            if engaged:
                self._was_engaged = True
            # Base stillness over the trailing still_hold_s window (xy + z, so the
            # descent of a laydown counts as motion).
            moved = None
            if len(self._snapshots) > self._still_need:
                back = np.asarray(self._snapshots[-1 - self._still_need][0])
                moved = float(np.linalg.norm(self._latest_qpos[:3] - back[:3]))
            still = moved is not None and moved < self.still_eps_m
            if self._was_engaged and not engaged and self._disengage_frame is None:
                self._disengage_frame = len(self._snapshots)
            reason = None
            if (self._disengage_frame is not None and still
                    and len(self._snapshots) - self._disengage_frame >= self._still_need):
                # Disengaged AND at rest, with the whole stillness window measured
                # AFTER the disengage — so a robot that was merely standing still
                # when the policy released does not end the clip before the laydown
                # it is about to perform. The descent counts as motion, so this
                # fires once the robot is down and settled.
                reason = (f"disengaged and base at rest "
                          f"({moved * 100:.1f} cm over {self.still_hold_s:g} s)")
            elif (self._saw_forward_cmd and engaged
                  and abs(cmd_vx) <= 1e-6 and still):
                # Still engaged but the command is back to zero and the robot has
                # stopped — the return-x hard stop. Cut here rather than filming
                # the laydown that follows.
                reason = (f"cmd 0 and base still "
                          f"({moved * 100:.1f} cm over {self.still_hold_s:g} s)")
            if reason is not None:
                self._finished = True
                self.get_logger().info(
                    f"early finish after {len(self._snapshots)}/{self.target_frames} "
                    f"frames — {reason}")
                self.target_frames = len(self._snapshots)
                self._render_and_finish()
                return

        n = len(self._snapshots)
        if n % 50 == 0:
            self.get_logger().info(
                f"captured {n}/{self.target_frames}  base_xyz=({float(self._latest_qpos[0]):+.2f},{float(self._latest_qpos[1]):+.2f},{float(self._latest_qpos[2]):.3f})  cmd_vx={cmd_vx:+.2f}  fsm={fsm}"
            )
        if n == self.target_frames:
            self._render_and_finish()

    # ---------------- render phase -----------------------------------
    def _render_and_finish(self):
        self.get_logger().info(f"capture done; rendering {self.target_frames} frames")
        base_z = np.asarray(
            [float(s[0][2]) for s in self._snapshots], dtype=np.float64
        )
        engaged = np.asarray(
            [bool(s[4]) for s in self._snapshots], dtype=bool
        )

        def _height_stats(values: np.ndarray) -> dict:
            if values.size == 0:
                # lift_test runs never engage the policy, so the
                # "policy_engaged" slice is empty — don't crash on reductions.
                return {"frames": 0, "duration_s": 0.0, "mean_m": None,
                        "std_m": None, "min_m": None, "max_m": None}
            return {
                "frames": int(values.size),
                "duration_s": float(values.size / self.fps),
                "mean_m": float(np.mean(values)),
                "std_m": float(np.std(values)),
                "min_m": float(np.min(values)),
                "max_m": float(np.max(values)),
            }

        metrics = {
            "source_video": str(self.out_path),
            "fps": self.fps,
            "all_frames": _height_stats(base_z),
            "policy_engaged": _height_stats(base_z[engaged]),
        }
        metrics_path = self.out_path.with_suffix(".base_height.json")
        metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
        eng_mean = metrics['policy_engaged']['mean_m']
        eng_str = f"{eng_mean:.6f}" if eng_mean is not None else "n/a"
        self.get_logger().info(
            "base height: "
            f"engaged mean={eng_str} m "
            f"(n={metrics['policy_engaged']['frames']}), "
            f"full mean={metrics['all_frames']['mean_m']:.6f} m "
            f"→ {metrics_path}"
        )
        # ---- per-foot foot-force log -----------------------------------
        # One row per captured frame; forces in policy order FR,FL,RR,RL
        # (Newtons, from MuJoCo /contact). fsm column splits lay/stand/engaged.
        ff_path = self.out_path.with_suffix(".foot_forces.csv")
        with open(ff_path, "w") as fh:
            fh.write("frame,t_s,fsm,lift_phase,dlat,base_z,FR,FL,RR,RL\n")
            for i, snap in enumerate(self._snapshots):
                qz = float(snap[0][2])
                fsm = snap[5]
                t_s = i / float(self.fps)
                ff = (self._contact_snaps[i] if i < len(self._contact_snaps)
                      else np.full(4, np.nan))
                lp = (self._lift_phase_snaps[i]
                      if i < len(self._lift_phase_snaps) else 0)
                dl = (self._dlat_snaps[i] if i < len(self._dlat_snaps) else 0.0)
                fh.write(
                    f"{i},{t_s:.4f},{fsm},{lp},{dl:.4f},{qz:.4f},"
                    f"{ff[0]:.4f},{ff[1]:.4f},{ff[2]:.4f},{ff[3]:.4f}\n"
                )
        self.get_logger().info(f"wrote foot forces → {ff_path}")

        # Trajectory dump for offline ablation replays: per-frame qpos[19] so the
        # depth + contact can be re-rendered/re-derived under any sensor/clip/noise
        # variant on the SAME poses (fully paired, no trajectory divergence).
        try:
            traj_path = self.out_path.with_suffix(".traj.npz")
            np.savez(
                traj_path,
                qpos=np.array([s[0] for s in self._snapshots], dtype=np.float64),
                fsm=np.array([s[5] for s in self._snapshots]),
                engaged=np.array([bool(s[4]) for s in self._snapshots]),
            )
            self.get_logger().info(f"wrote trajectory → {traj_path}")
        except Exception as e:  # pragma: no cover
            self.get_logger().warning(f"traj dump failed: {e}")

        if self.metrics_only:
            self.get_logger().info("metrics-only capture done; skipping video render")
            rclpy.shutdown()
            return

        m = mujoco.MjModel.from_xml_path(str(self.scene))
        d = mujoco.MjData(m)
        renderer = mujoco.Renderer(m, height=self.height, width=self.width)
        import imageio
        writer = imageio.get_writer(
            str(self.out_path), fps=self.fps, codec="libx264", quality=None,
            macro_block_size=1,
            output_params=["-crf", "23", "-preset", "ultrafast", "-pix_fmt", "yuv420p"],
        )

        # Pre-compute ground-truth body-frame forward velocity by finite-diff
        # of qpos[:2] and rotating by the negative yaw. Smooth over a small
        # window so the readout doesn't flicker frame-to-frame.
        win = max(3, int(round(self.fps * 0.1)))   # ~100 ms moving average
        dt = 1.0 / float(self.fps)
        positions = np.array([s[0][:2] for s in self._snapshots], dtype=np.float64)   # (N, 2)
        quats = np.array([s[0][3:7] for s in self._snapshots], dtype=np.float64)      # (N, 4)
        # finite-diff over `win` frames → (N, 2) world-frame velocity
        N = len(self._snapshots)
        world_v = np.zeros((N, 2), dtype=np.float64)
        for i in range(N):
            lo = max(0, i - win // 2)
            hi = min(N - 1, i + win // 2)
            if hi > lo:
                world_v[i] = (positions[hi] - positions[lo]) / ((hi - lo) * dt)

        for idx, snap in enumerate(self._snapshots):
            qpos, yaw_pred, depth_norm, cmd_vx, engaged, fsm = snap
            d.qpos[:19] = qpos
            mujoco.mj_forward(m, d)
            renderer.update_scene(d, camera=self.camera)

            if not self.clean_frame and float(np.linalg.norm(yaw_pred)) > 1e-6:
                _r, _p, world_yaw = mj_quat_to_roll_pitch_yaw(qpos[3:7])
                arrow_pos = qpos[:3].copy()
                arrow_pos[2] += 0.20
                _add_arrow(
                    renderer.scene, arrow_pos,
                    world_yaw + YAW_PRED_MULTIPLIER * float(yaw_pred[0]),
                    length=0.45, radius=0.018,
                    rgba=(0.95, 0.20, 0.15, 1.0),
                )
                _add_arrow(
                    renderer.scene, arrow_pos,
                    world_yaw + YAW_PRED_MULTIPLIER * float(yaw_pred[1]),
                    length=0.35, radius=0.012,
                    rgba=(1.00, 0.65, 0.10, 0.85),
                )

            frame = renderer.render().copy()
            overlay_scale = min(self.width / 1280.0, self.height / 720.0)

            # command + state overlay (top-left).
            if _HAVE_CV2 and not self.clean_frame:
                # Body-frame GT velocity from finite-diff'd qpos rotated by -yaw.
                # vx_body is the "forward speed the policy is actually achieving".
                _r, _p, world_yaw = mj_quat_to_roll_pitch_yaw(qpos[3:7])
                wx, wy = float(world_v[idx, 0]), float(world_v[idx, 1])
                cy, sy = np.cos(-world_yaw), np.sin(-world_yaw)
                gt_vx_body = cy * wx - sy * wy
                gt_vy_body = sy * wx + cy * wy
                gt_speed_world = float(np.hypot(wx, wy))
                t_s = idx / float(self.fps)
                base_z = float(qpos[2])
                world_x = float(qpos[0])
                world_y = float(qpos[1])
                lines = [
                    f"FSM: {fsm:<7s}  t={t_s:5.1f}s",
                    f"cmd_vx:     {cmd_vx:+.2f} m/s",
                    f"gt_vx_body: {gt_vx_body:+.2f} m/s",
                    f"gt_speed:   {gt_speed_world:+.2f} m/s",
                    f"base_z:      {base_z:.2f} m",
                    f"world_x:    {world_x:+.2f} m",
                    f"world_y:    {world_y:+.2f} m",
                ]
                x0 = max(8, int(round(16 * overlay_scale)))
                y = max(16, int(round(32 * overlay_scale)))
                title_font = max(0.4, 0.8 * overlay_scale)
                text_font = max(0.35, 0.7 * overlay_scale)
                title_outline = max(2, int(round(5 * overlay_scale)))
                title_stroke = max(1, int(round(2 * overlay_scale)))
                text_outline = max(2, int(round(4 * overlay_scale)))
                text_stroke = max(1, int(round(2 * overlay_scale)))
                title_step = max(20, int(round(38 * overlay_scale)))
                line_step = max(14, int(round(28 * overlay_scale)))
                # optional title header (scene / policy descriptor)
                if self.title:
                    cv2.putText(frame, self.title, (x0, y), cv2.FONT_HERSHEY_SIMPLEX,
                                title_font, (0, 0, 0), title_outline, cv2.LINE_AA)
                    cv2.putText(frame, self.title, (x0, y), cv2.FONT_HERSHEY_SIMPLEX,
                                title_font, (60, 200, 255), title_stroke, cv2.LINE_AA)  # BGR amber
                    y += title_step
                for txt in lines:
                    # black outline for contrast on the gradient sky
                    cv2.putText(frame, txt, (x0, y), cv2.FONT_HERSHEY_SIMPLEX,
                                text_font, (0, 0, 0), text_outline, cv2.LINE_AA)
                    color = (40, 200, 60) if engaged else (220, 220, 220)
                    cv2.putText(frame, txt, (x0, y), cv2.FONT_HERSHEY_SIMPLEX,
                                text_font, color, text_stroke, cv2.LINE_AA)
                    y += line_step

            # depth PiP overlay (top-right)
            if depth_norm is not None and not self.clean_frame:
                # Keep the inset at the same relative size as the original
                # 348x232 inset in a 1280x720 render. Fixed pixel dimensions
                # made it occupy most of a 640x360 fast render.
                pip_h = max(DEPTH_TARGET_H, int(round(232 * overlay_scale)))
                pip_w = max(DEPTH_TARGET_W, int(round(348 * overlay_scale)))
                margin = max(4, int(round(12 * overlay_scale)))
                border = max(1, int(round(3 * overlay_scale)))
                # Isaac depth display convention (legged_robot.py:281): greyscale
                # of normalized depth shifted to [0,1] -> near=black, far=white.
                d_n = (depth_norm + 0.5).clip(0.0, 1.0)
                d_vis = (d_n * 255.0).astype(np.uint8)
                d_t = torch.from_numpy(d_vis.astype(np.float32))[None, None]
                d_t = F.interpolate(d_t, size=(pip_h, pip_w), mode="nearest")
                d_up = d_t.squeeze().numpy().astype(np.uint8)
                rgb = np.stack([d_up, d_up, d_up], axis=-1)
                y0, y1 = margin, margin + pip_h
                x0, x1 = self.width - margin - pip_w, self.width - margin
                frame[y0-border:y1+border, x0-border:x1+border] = 0
                frame[y0:y1, x0:x1] = rgb

            writer.append_data(frame)
            if (idx + 1) % 50 == 0:
                self.get_logger().info(
                    f"rendered {idx+1}/{self.target_frames}  base_z={float(qpos[2]):.3f}"
                )

        writer.close()
        self.get_logger().info(f"wrote {self.target_frames} frames → {self.out_path}")
        rclpy.shutdown()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--duration", type=float, default=8.0)
    ap.add_argument("--fps", type=int, default=50)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--camera", type=str, default="follow_cam")
    ap.add_argument("--lin_vel_max", type=float, default=0.7,
                    help="must match the value passed to rl_distill_policy / "
                         "depth_encoder_node so the on-screen cmd_vx is correct.")
    ap.add_argument("--title", type=str, default="",
                    help="optional header text drawn at the top-left of every "
                         "frame (e.g. policy id + scene descriptor).")
    ap.add_argument("--metrics-only", action="store_true",
                    help="write base-height JSON after capture and skip MP4 rendering.")
    ap.add_argument("--clean-frame", action="store_true",
                    help="render only the MuJoCo scene, without telemetry, depth PiP, "
                         "or yaw arrows.")
    ap.add_argument("--no_end_on_still", action="store_true",
                    help="keep capturing for the full --duration even after the robot "
                         "has finished and stopped. Default is to end the clip at the "
                         "engage->disengage edge or once the command is 0 and the base "
                         "has been still for --still_hold_s, whichever comes first.")
    ap.add_argument("--still_hold_s", type=float, default=1.0,
                    help="seconds of no base motion (< --still_eps_m) required before "
                         "an early finish, once the command is back to 0.")
    ap.add_argument("--still_eps_m", type=float, default=0.02,
                    help="base xy displacement over --still_hold_s counted as 'still'.")
    args, _ = ap.parse_known_args()

    rclpy.init()
    node = DistillRecorder(args.scene, args.output, args.duration,
                            args.fps, args.width, args.height, args.camera,
                            args.lin_vel_max, args.title, args.metrics_only,
                            args.clean_frame,
                            end_on_still=not args.no_end_on_still,
                            still_hold_s=args.still_hold_s,
                            still_eps_m=args.still_eps_m)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
