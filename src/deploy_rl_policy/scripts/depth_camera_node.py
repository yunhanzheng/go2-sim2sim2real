#!/usr/bin/env python3
"""Real-robot depth preprocessor (faithful to the training pipeline).

Subscribes /camera/depth/image_raw (sensor_msgs/Image, 16UC1 mm or 32FC1 m) and
republishes sensor_msgs/Image (32FC1, 58x87) on /rl/depth_image — exactly the
depth the policy was trained on — preserving header.stamp so the encoder can
enforce the 80 ms latency budget.

WHY THE EXTRA STEP (vs naive resize):
The policy was trained on a 60x106 sim render (hFOV 87 deg, vFOV 56.49 deg), then
normalize_depth crops [:-2, 4:-4] -> 58x98 and resizes to 58x87. A RealSense D435i
publishes 640x480 (FOV ~87 x 58 deg, 4:3). Feeding that straight into normalize_depth
skips the crop (it only triggers for the exact 60x106 shape) and squashes 4:3 -> 58x87,
i.e. a different FOV/aspect than training -> off-distribution depth for the encoder.

So here we first map the RealSense frame to the *sim render geometry*:
  1. center-crop to the training render's FOV (vFOV 56.49 of the cam's 58; hFOV matches),
  2. bicubic-resize to 60x106 (the render resolution),
then normalize_depth runs its standard (60,106) path. We also decimate to ~10 Hz to
match the recurrent encoder's training update rate (it was trained at 10 Hz, not 15).

Camera FOV is a parameter (cam_hfov_deg / cam_vfov_deg) in case your unit differs.
"""
import array
import math
import time
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float32MultiArray
import numpy as np
import torch
import torch.nn.functional as F

from distill_common import (normalize_depth, DEPTH_TARGET_H, DEPTH_TARGET_W,
                            DEPTH_RAW_H, DEPTH_RAW_W, DEPTH_FAR,
                            MUJOCO_EMPTY_BELOW_M,
                            radial_maxperp_mask, depth_radial_clip_enabled,
                            host_monotonic_ns, hole_fill)

# Training sim-render FOV (legged_robot_config snapshot; see distill_runner.py).
SIM_HFOV_DEG = 87.0
SIM_VFOV_DEG = 56.49


def image_msg_to_metres(msg: Image, hole_fill_mode: int = 1,
                        hole_fill_iters: int = 8) -> np.ndarray:
    h, w = msg.height, msg.width
    if msg.encoding == "32FC1":
        m = np.frombuffer(msg.data, dtype=np.float32).reshape(h, w).astype(np.float32, copy=False)
    elif msg.encoding == "16UC1":
        m = np.frombuffer(msg.data, dtype=np.uint16).reshape(h, w).astype(np.float32) * 1e-3
    else:
        raise ValueError(f"unsupported depth encoding: {msg.encoding}")
    # Hole filling, ON BY DEFAULT since 2026-07-28. Must run BEFORE the
    # empty_below/invalid remap below: afterwards every hole has already become
    # far and there is nothing left to fill. Shares distill_common.hole_fill with
    # the offline replay, which is what the training recalibration was measured
    # against — training (stereo_baseline 0.010 / pw_edge_prob 0.025 /
    # dis_noise 0.016) now assumes the FILLED distribution, so turning this off
    # without retraining leaves the student ~2x cleaner than it trained for.
    if hole_fill_mode >= 0:
        m = hole_fill(m, hole_fill_mode, hole_fill_iters)
    # Invalid (<= 0 no-return) AND sub-15 cm readings both become far, applied
    # before crop+resize so they don't bicubic-smear into mid-range values.
    # The 0.15 m threshold matches training (cfg.depth.empty_below_m, added
    # 2026-07-25) and the MuJoCo depth_sim_node (MUJOCO_EMPTY_BELOW_M): the
    # student never sees sub-15 cm depth in training, so a rare valid
    # close-range RealSense pixel must read far/empty here too, not survive
    # as an off-distribution near value. (Supersedes the 2026-07-24
    # "preserve valid close-range pixels" relaxation.)
    return np.where(m < MUJOCO_EMPTY_BELOW_M,
                    np.float32(DEPTH_FAR), m).astype(np.float32)


def to_sim_render(metres: np.ndarray, cam_hfov: float, cam_vfov: float) -> np.ndarray:
    """Map a real depth frame (HxW, covering cam_hfov x cam_vfov) onto the
    training render geometry (DEPTH_RAW_H x DEPTH_RAW_W = 60x106, hFOV 87 /
    vFOV 56.49).

    Aligns FOV per axis in tan (pinhole) space:
      * where the real camera is WIDER than the sim FOV, center-crop the excess;
      * where it is NARROWER (e.g. a D435 depth stream at 640x480, hFOV ~75 <
        87), the real image only spans the central part of the sim frame, so it
        is resized into that sub-region and the borders are padded with
        DEPTH_FAR. Angles the real camera never saw read empty=far, exactly as
        off-FOV pixels do in the sim stereo/radial model.
    Reduces to a plain bicubic 60x106 resize when the FOVs already match.
    """
    h, w = metres.shape
    ts_h = math.tan(math.radians(SIM_HFOV_DEG) / 2.0)   # sim half-FOV in tan space
    tc_h = math.tan(math.radians(cam_hfov) / 2.0)       # real half-FOV
    ts_v = math.tan(math.radians(SIM_VFOV_DEG) / 2.0)
    tc_v = math.tan(math.radians(cam_vfov) / 2.0)
    # 1) crop the real image to the central portion lying within the sim FOV
    #    (fraction <1 only when the real camera is wider than sim on that axis).
    keep_w = min(1.0, ts_h / tc_h)
    keep_h = min(1.0, ts_v / tc_v)
    cw = max(1, int(round(w * keep_w)))
    ch = max(1, int(round(h * keep_h)))
    x0, y0 = (w - cw) // 2, (h - ch) // 2
    crop = metres[y0:y0 + ch, x0:x0 + cw]
    # 2) resize the (cropped) real content into the fraction of the sim frame it
    #    actually covers; <1 only when the real camera is narrower than sim.
    cover_w = min(1.0, tc_h / ts_h)
    cover_v = min(1.0, tc_v / ts_v)
    inner_w = max(1, min(DEPTH_RAW_W, int(round(DEPTH_RAW_W * cover_w))))
    inner_h = max(1, min(DEPTH_RAW_H, int(round(DEPTH_RAW_H * cover_v))))
    t = torch.from_numpy(np.ascontiguousarray(crop, dtype=np.float32))[None, None]
    t = F.interpolate(t, size=(inner_h, inner_w), mode="bicubic", align_corners=False)
    # 3) pad the uncovered border (real FOV < sim FOV) with far = "unseen/empty".
    if (inner_h, inner_w) != (DEPTH_RAW_H, DEPTH_RAW_W):
        pad_l = (DEPTH_RAW_W - inner_w) // 2
        pad_r = DEPTH_RAW_W - inner_w - pad_l
        pad_t = (DEPTH_RAW_H - inner_h) // 2
        pad_b = DEPTH_RAW_H - inner_h - pad_t
        t = F.pad(t, (pad_l, pad_r, pad_t, pad_b),
                  mode="constant", value=float(DEPTH_FAR))
    return t.squeeze(0).squeeze(0).numpy()


def metres_to_image_msg(arr: np.ndarray, stamp, frame_id: str = "depth") -> Image:
    h, w = arr.shape
    msg = Image()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height, msg.width = h, w
    msg.encoding = "32FC1"
    msg.is_bigendian = 0
    msg.step = w * 4
    # array.array('B'), not raw bytes: rclpy's uint8[] setter validates a
    # `bytes` value element-by-element (1.4 ms for 58x87 f32; 57 ms for a full
    # 848x480 depth frame). array.array takes the generated fast path.
    msg.data = array.array('B', arr.astype(np.float32).tobytes())
    return msg


class DepthCameraNode(Node):
    def __init__(self):
        super().__init__("depth_camera_node")
        self.declare_parameter("input_topic", "/camera/depth/image_rect_raw")
        self.declare_parameter("output_topic", "/rl/depth_image")
        self.declare_parameter("target_hz", 10.0)      # match training encoder cadence
        # cam_hfov/vfov are FALLBACK values only. The true depth-stream FOV
        # depends on the resolution (a D435 depth stream at 640x480 is ~75x62,
        # NOT the 87x58 native spec), so by default we override these from the
        # live camera_info intrinsics (auto_fov). Set auto_fov:=0 to force the
        # params, or info_topic to point at a non-default camera_info topic.
        self.declare_parameter("cam_hfov_deg", 87.0)   # fallback RealSense D435 depth FOV
        self.declare_parameter("cam_vfov_deg", 58.0)
        self.declare_parameter("auto_fov", True)
        self.declare_parameter("info_topic", "")       # "" => derive from input_topic
        # Hole filling: DEFAULT ON (mode 1 = librealsense farthest_from_around).
        # -1 disables. The 2026-07-28 training recalibration targets the FILLED
        # distribution, so -1 requires a student retrained on raw depth.
        self.declare_parameter("hole_fill", 1)
        self.declare_parameter("hole_fill_iters", 8)
        # Fail-safe: max age of the cached input frame before the timer STOPS
        # republishing it (see run_real_remote.sh watchdog note — a dead camera
        # must stop /rl/depth_image so the encoder stops /rl/depth_latent and
        # the actor stops /rl/target_pos, letting low_level_ctrl's
        # rl_stall_timeout latch a settle). Camera is nominally ~30 Hz, so
        # 0.5 s = ~15 missed frames — never trips in normal operation.
        # <= 0 disables the check (old republish-forever behavior).
        self.declare_parameter("max_frame_age_s", 0.5)
        self.declare_parameter("latency_log", "")
        # Passive Jetson-Mac clock-drift tracker (REAL path only; default off).
        # The camera header stamp is RealSense GLOBAL_TIME in the *Jetson* clock
        # domain; the encoder's 80 ms budget compares it against the Mac clock.
        # sync_jetson_clock.sh steps the Jetson clock once at launch, after
        # which the clocks free-run apart (~17 ppm measured 2026-07-24, i.e.
        # ~1 ms/min) and NTP on the Jetson can step it mid-run (+50 ms seen).
        # The 2026-07-24 real runs showed the exposure->Mac transport FLOOR is
        # constant to <1 ms (p95), so the rolling MINIMUM of
        # (Mac wall receipt - header stamp) tracks the clock offset exactly:
        #   offset(t) = min over trailing window  -  min over launch baseline.
        # Published on /rl/depth_clock_offset for the encoder to subtract.
        # MUST stay off (default) in sim launches: there the header stamp is
        # not in a Mac-comparable clock domain and the "offset" would be
        # garbage that corrupts the encoder's latency enforcement.
        self.declare_parameter("track_clock_offset", False)
        self.declare_parameter("offset_baseline_s", 5.0)   # launch calibration window
        self.declare_parameter("offset_window_s", 10.0)    # trailing rolling-min window
        self.declare_parameter("offset_step_warn_ms", 25.0)
        gp = self.get_parameter
        in_topic = gp("input_topic").get_parameter_value().string_value
        out_topic = gp("output_topic").get_parameter_value().string_value
        self.target_hz = gp("target_hz").get_parameter_value().double_value
        self.cam_hfov = gp("cam_hfov_deg").get_parameter_value().double_value
        self.cam_vfov = gp("cam_vfov_deg").get_parameter_value().double_value
        self.auto_fov = gp("auto_fov").get_parameter_value().bool_value
        info_topic = gp("info_topic").get_parameter_value().string_value
        self.hole_fill_mode = int(gp("hole_fill").get_parameter_value().integer_value)
        self.hole_fill_iters = int(gp("hole_fill_iters").get_parameter_value().integer_value)
        self._fov_from_info = False
        self.max_frame_age_s = gp("max_frame_age_s").get_parameter_value().double_value
        latency_log = gp("latency_log").get_parameter_value().string_value
        self.track_clock_offset = gp("track_clock_offset").get_parameter_value().bool_value
        self.offset_baseline_s = gp("offset_baseline_s").get_parameter_value().double_value
        self.offset_window_s = gp("offset_window_s").get_parameter_value().double_value
        self.offset_step_warn_ms = gp("offset_step_warn_ms").get_parameter_value().double_value
        # (receive_mono_ns, raw_delta_ns) samples; raw_delta = Mac wall - stamp.
        self._offset_samples: deque = deque()
        self._offset_baseline_ns = None      # transport floor at launch (offset ~ 0)
        self._offset_first_mono_ns = None
        self._offset_ns = 0                  # current estimate (0 until baseline ready)
        self._offset_pub = None
        self._offset_step_warned = False
        if self.track_clock_offset:
            self._offset_pub = self.create_publisher(
                Float32MultiArray, "/rl/depth_clock_offset", 10)
        # Fixed-rate output: cache the latest camera frame and publish it from a
        # timer at EXACTLY target_hz, rather than gating on camera-frame arrival.
        # The old arrival-gate quantized to camera frame times — a 27 Hz camera
        # decimated to ~9 Hz, never a clean 10. A timer gives dead-on target_hz
        # (and matches Isaac, which updates depth on a fixed sim-step cadence).
        # Processing then runs at target_hz too (not the full camera rate).
        # target_hz <= 0 => pass through every camera frame (no timer).
        self._latest_msg = None
        self._stale_dropping = False   # True while the fail-safe is suppressing output
        self._raw_seq = 0
        self._latency_log_fp = None
        if latency_log:
            self._latency_log_fp = open(latency_log, "w", buffering=1)
            self._latency_log_fp.write(
                "raw_seq\tsource_sec\tsource_nsec\treceive_mono_ns\treceive_wall_ns\t"
                "process_start_mono_ns\tprocess_end_mono_ns\t"
                "publish_start_mono_ns\tpublish_end_mono_ns\tclock_offset_ns\n"
            )

        # Current training and RealSense both use planar perpendicular-depth
        # clipping. Retain the old spherical vignette only for legacy replays.
        self.radial_clip = depth_radial_clip_enabled()
        self._radial_maxperp = (radial_maxperp_mask(DEPTH_RAW_H, DEPTH_RAW_W)
                                if self.radial_clip else None)

        self.sub = self.create_subscription(Image, in_topic, self._on_image, 10)
        self.pub = self.create_publisher(Image, out_topic, 10)
        # Auto-detect the true depth-stream FOV from camera_info (see cam_hfov
        # comment). Intrinsics are static, so the callback unsubscribes after the
        # first valid message. Falls back to the params if no info ever arrives.
        self._info_sub = None
        if self.auto_fov:
            if not info_topic:
                info_topic = in_topic.rsplit("/", 1)[0] + "/camera_info"
            self._info_sub = self.create_subscription(
                CameraInfo, info_topic, self._on_camera_info, 10)
        self.timer = (self.create_timer(1.0 / self.target_hz, self._on_timer)
                      if self.target_hz > 0 else None)
        self.get_logger().info(
            f"depth_camera_node: {in_topic} -> {out_topic} | "
            f"{'timer @ %.1f Hz' % self.target_hz if self.timer else 'passthrough (every frame)'} | "
            f"cam FOV {self.cam_hfov:.1f}x{self.cam_vfov:.1f} "
            f"({'awaiting camera_info on %s' % info_topic if self._info_sub else 'param'}) "
            f"-> sim render 60x106 -> 58x87 | "
            f"radial_clip={'on' if self.radial_clip else 'off (planar)'} | "
            f"max_frame_age_s={self.max_frame_age_s:g}"
            f"{'' if self.max_frame_age_s > 0 else ' (stale fail-safe OFF)'} | "
            f"clock_offset_tracking="
            f"{'on (baseline %gs, window %gs)' % (self.offset_baseline_s, self.offset_window_s) if self.track_clock_offset else 'off'}")

    def _on_camera_info(self, msg: CameraInfo):
        fx, fy = float(msg.k[0]), float(msg.k[4])
        w, h = int(msg.width), int(msg.height)
        if fx <= 0.0 or fy <= 0.0 or w <= 0 or h <= 0:
            return
        hfov = math.degrees(2.0 * math.atan(w / (2.0 * fx)))
        vfov = math.degrees(2.0 * math.atan(h / (2.0 * fy)))
        self.get_logger().info(
            f"camera_info FOV {hfov:.1f}x{vfov:.1f} deg "
            f"(fx={fx:.1f} fy={fy:.1f} cx={msg.k[2]:.1f} cy={msg.k[5]:.1f} {w}x{h}) "
            f"replaces param {self.cam_hfov:.1f}x{self.cam_vfov:.1f}")
        self.cam_hfov, self.cam_vfov = hfov, vfov
        self._fov_from_info = True
        # Intrinsics are static — one message is enough; stop listening.
        if self._info_sub is not None:
            self.destroy_subscription(self._info_sub)
            self._info_sub = None

    def _update_clock_offset(self, msg: Image,
                             receive_mono_ns: int, receive_wall_ns: int):
        """Passive drift tracker (see track_clock_offset parameter comment).

        Runs on every camera arrival (~30 Hz). All operations are O(window)
        over ~300 samples — well under 0.1 ms, negligible next to the 10 Hz
        image preprocessing.
        """
        stamp_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        if stamp_ns <= 0:
            return
        raw_delta_ns = receive_wall_ns - stamp_ns
        samples = self._offset_samples
        samples.append((receive_mono_ns, raw_delta_ns))
        window_ns = int(self.offset_window_s * 1e9)
        while samples and receive_mono_ns - samples[0][0] > window_ns:
            samples.popleft()
        if self._offset_first_mono_ns is None:
            self._offset_first_mono_ns = receive_mono_ns
        if self._offset_baseline_ns is None:
            # Launch calibration: sync_jetson_clock.sh has just stepped the
            # Jetson clock (verified residual < ~4 ms), so the minimum raw
            # delta over the first offset_baseline_s IS the transport floor.
            if (receive_mono_ns - self._offset_first_mono_ns
                    < self.offset_baseline_s * 1e9):
                return  # still collecting; offset stays 0 (clock just synced)
            self._offset_baseline_ns = min(d for _, d in samples)
            self.get_logger().info(
                f"clock-offset baseline: transport floor "
                f"{self._offset_baseline_ns / 1e6:.2f} ms over "
                f"{len(samples)} frames — publishing /rl/depth_clock_offset")
        floor_ns = min(d for _, d in samples)
        self._offset_ns = floor_ns - self._offset_baseline_ns
        # Step detection: the floor over the most recent ~1 s vs the window
        # floor. Transport jitter is one-sided (delays only, p99 +4 ms), so a
        # sustained gap means the Jetson clock stepped (e.g. NTP) — the
        # rolling window self-corrects within offset_window_s, but warn so
        # the operator knows enforcement is degraded until then.
        recent_ns = [d for t, d in samples if receive_mono_ns - t < 1e9]
        if len(recent_ns) >= 5:
            gap_ms = (min(recent_ns) - floor_ns) / 1e6
            if gap_ms > self.offset_step_warn_ms:
                if not self._offset_step_warned:
                    self._offset_step_warned = True
                    self.get_logger().warning(
                        f"clock STEP suspected (+{gap_ms:.1f} ms vs window floor) "
                        f"— offset estimate degraded for up to "
                        f"{self.offset_window_s:g} s until the window refills")
            elif self._offset_step_warned and gap_ms < self.offset_step_warn_ms / 2:
                self._offset_step_warned = False
                self.get_logger().info("clock-offset floor consistent again")

    def _publish_clock_offset(self):
        if self._offset_pub is None:
            return
        msg = Float32MultiArray()
        # [offset_ms, transport_floor_ms, baseline_ready]
        floor_ms = (float("nan") if not self._offset_samples
                    else min(d for _, d in self._offset_samples) / 1e6)
        msg.data = [
            float(self._offset_ns) / 1e6,
            float(floor_ms),
            1.0 if self._offset_baseline_ns is not None else 0.0,
        ]
        self._offset_pub.publish(msg)

    def _on_image(self, msg: Image):
        receive_mono_ns = host_monotonic_ns()
        receive_wall_ns = time.time_ns()
        raw_seq = self._raw_seq
        self._raw_seq += 1
        if self.track_clock_offset:
            self._update_clock_offset(msg, receive_mono_ns, receive_wall_ns)
        if self.timer is not None:
            self._latest_msg = (msg, raw_seq, receive_mono_ns, receive_wall_ns)
            return
        self._process_and_publish(msg, raw_seq, receive_mono_ns, receive_wall_ns)

    def _on_timer(self):
        latest = self._latest_msg
        if latest is None:
            return
        if self.max_frame_age_s > 0:
            # Fail-safe: never republish a stale cached frame. If the input
            # topic has gone silent (camera death/stall), STOP publishing so
            # the downstream encoder -> actor -> low_level_ctrl rl_stall_timeout
            # chain settles the robot instead of walking on frozen depth.
            # Resumes automatically as soon as a fresh frame arrives.
            age_s = (host_monotonic_ns() - latest[2]) * 1e-9   # latest[2] = receive_mono_ns
            if age_s > self.max_frame_age_s:
                self._stale_dropping = True
                self.get_logger().error(
                    f"STALE input frame ({age_s:.2f} s > max_frame_age_s="
                    f"{self.max_frame_age_s:g}) — suppressing /rl/depth_image "
                    "so the actor stops and low_level_ctrl settles",
                    throttle_duration_sec=1.0)
                return
            if self._stale_dropping:
                self._stale_dropping = False
                self.get_logger().info(
                    f"input frames fresh again (age {age_s:.3f} s) — "
                    "resuming /rl/depth_image publication")
        self._process_and_publish(*latest)

    def _process_and_publish(self, msg: Image, raw_seq: int,
                             receive_mono_ns: int, receive_wall_ns: int):
        process_start_ns = host_monotonic_ns()
        try:
            metres = image_msg_to_metres(msg, self.hole_fill_mode, self.hole_fill_iters)
        except ValueError as e:
            self.get_logger().warning(str(e), throttle_duration_sec=2.0)
            return
        if metres.shape != (DEPTH_RAW_H, DEPTH_RAW_W):
            metres = to_sim_render(metres, self.cam_hfov, self.cam_vfov)
        # Legacy radial far-clip on the (60,106) render grid. Current training
        # leaves this off and clips perpendicular depth uniformly.
        if self.radial_clip and metres.shape == self._radial_maxperp.shape:
            metres = np.where(metres > self._radial_maxperp,
                              np.float32(DEPTH_FAR), metres).astype(np.float32)
        norm = normalize_depth(metres).numpy()[0]       # (58, 87) in [-0.5, 0.5]
        assert norm.shape == (DEPTH_TARGET_H, DEPTH_TARGET_W), norm.shape
        process_end_ns = host_monotonic_ns()
        out = metres_to_image_msg(norm, msg.header.stamp, frame_id=msg.header.frame_id)
        publish_start_ns = host_monotonic_ns()
        self.pub.publish(out)
        publish_end_ns = host_monotonic_ns()
        # Offset published right after each processed frame (10 Hz), so the
        # encoder always holds an estimate at most one frame old.
        self._publish_clock_offset()
        if self._latency_log_fp is not None:
            self._latency_log_fp.write(
                f"{raw_seq}\t{msg.header.stamp.sec}\t{msg.header.stamp.nanosec}\t"
                f"{receive_mono_ns}\t{receive_wall_ns}\t{process_start_ns}\t"
                f"{process_end_ns}\t{publish_start_ns}\t{publish_end_ns}\t"
                f"{self._offset_ns}\n"
            )

    def destroy_node(self):
        if self._latency_log_fp is not None:
            self._latency_log_fp.close()
            self._latency_log_fp = None
        return super().destroy_node()


def main():
    rclpy.init()
    node = DepthCameraNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
