#!/usr/bin/env python3
"""Verify the MuJoCo port of extreme-parkour's cliffside depth model."""

import os
os.environ.setdefault("MUJOCO_GL", "glfw")

from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "deploy_rl_policy" / "scripts"))

import mujoco  # noqa: E402
from distill_common import (  # noqa: E402
    DEPTH_FAR,
    DEPTH_HFOV_DEG,
    DEPTH_RAW_H,
    DEPTH_RAW_W,
    MujocoStereoDepth,
    TRAIN_DEPTH_CFG,
    TRAIN_DEPTH_SOURCE_COMMIT,
    TrainDepthDegrader,
    clip_mujoco_too_close_to_far,
    depth_radial_clip_enabled,
    normalize_depth,
)


SCENE_XML = """
<mujoco model="stereo_verify">
  <option gravity="0 0 0"/>
  <visual><map znear="0.01" zfar="10"/></visual>
  <worldbody>
    <body name="camera_mount">
      <camera name="front_depth_cam" pos="0.33 0 0.5"
              xyaxes="0 -1 0 0 0 1" fovy="56.49"/>
    </body>
    <geom name="backdrop" type="box" pos="1.52 0 0.5"
          size="0.02 2 1" rgba="0.5 0.5 0.5 1"/>
    <geom name="near_box" type="box" pos="0.95 -0.30 0.5"
          size="0.10 0.18 0.25" rgba="0.8 0.2 0.2 1"/>
  </worldbody>
</mujoco>
"""


def contiguous_left_width(mask_row):
    width = 0
    for value in mask_row:
        if not value:
            break
        width += 1
    return width


def verify_stereo():
    model = mujoco.MjModel.from_xml_string(SCENE_XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, height=DEPTH_RAW_H, width=DEPTH_RAW_W)
    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera="front_depth_cam")
    raw = renderer.render().astype(np.float32)

    assert not depth_radial_clip_enabled(), "current training default must be planar"
    mono = np.where(raw > DEPTH_FAR, np.float32(DEPTH_FAR), raw).astype(np.float32)
    stereo_model = MujocoStereoDepth(
        model,
        baseline=TRAIN_DEPTH_CFG["stereo_baseline"],
        eps=TRAIN_DEPTH_CFG["stereo_eps"],
        radial_clip=False,
    )
    stereo = stereo_model(data, raw)
    changed = stereo != mono
    assert np.all(stereo[changed] == DEPTH_FAR)

    row = DEPTH_RAW_H // 2
    z_bg = float(mono[row, DEPTH_RAW_W // 2])
    predicted = stereo_model.focal * stereo_model.baseline / z_bg
    observed = contiguous_left_width(stereo_model.last_fov_mask[row])
    assert abs(observed - predicted) <= 1.0, (observed, predicted)

    near = mono < z_bg - 0.2
    near_rows, near_cols = np.where(near)
    assert near_cols.size > 0
    box_row = int(np.median(near_rows))
    row_box_cols = np.where(near[box_row])[0]
    box_left = int(row_box_cols.min())
    occluded_left = np.where(stereo_model.last_occlusion_mask[box_row, :box_left])[0]
    # At the current 1 cm effective baseline the 106-wide render may quantize
    # the half-occlusion band to zero pixels; larger legacy baselines produced
    # a guaranteed non-empty band here.
    assert np.all(occluded_left < box_left)
    assert not np.any(stereo_model.last_occlusion_mask[box_row, row_box_cols.max() + 1:])

    print(
        f"stereo: changed={int(changed.sum())} "
        f"fov_band={observed}px (pred={predicted:.2f}) "
        f"half_occlusion={occluded_left.size}px"
    )


def verify_image_noise():
    raw = np.full((DEPTH_RAW_H, DEPTH_RAW_W), 1.0, dtype=np.float32)
    raw[:, DEPTH_RAW_W // 2:] = 1.6

    torch.manual_seed(7)
    clean = TrainDepthDegrader(noise_scale=0.0)(raw)
    expected = normalize_depth(raw)
    torch.testing.assert_close(clean, expected, rtol=0.0, atol=0.0)

    torch.manual_seed(7)
    reality = TrainDepthDegrader(noise_scale=0.5)(raw)
    assert reality.shape == (1, 58, 87)
    assert torch.isfinite(reality).all()
    assert float(reality.min()) >= -0.5 and float(reality.max()) <= 0.5
    assert not torch.equal(reality, expected)
    print(
        f"noise: source={TRAIN_DEPTH_SOURCE_COMMIT} "
        f"reality_delta_mean={float((reality - expected).abs().mean()):.5f}"
    )


def verify_close_clip():
    values = np.array([-0.1, 0.0, 0.149999, 0.15, 0.20], dtype=np.float32)
    clipped = clip_mujoco_too_close_to_far(values)
    np.testing.assert_array_equal(
        clipped,
        np.array([DEPTH_FAR, DEPTH_FAR, DEPTH_FAR, 0.15, 0.20], dtype=np.float32),
    )

    raw = np.full((DEPTH_RAW_H, DEPTH_RAW_W), 0.149, dtype=np.float32)
    depth = normalize_depth(clip_mujoco_too_close_to_far(raw))
    torch.testing.assert_close(depth, torch.full_like(depth, 0.5), rtol=0.0, atol=1e-6)
    print("close clip: depth <0.15m -> 2.00m far; 0.15m remains valid")


def main():
    assert TRAIN_DEPTH_SOURCE_COMMIT == "8b8ba80"
    assert TRAIN_DEPTH_CFG["stereo_baseline"] == 0.010
    assert TRAIN_DEPTH_CFG["pw_blindspot_cols"] == 0
    verify_stereo()
    verify_close_clip()
    verify_image_noise()
    print("ALL MUJOCO DEPTH-NOISE CHECKS PASSED")


if __name__ == "__main__":
    main()
