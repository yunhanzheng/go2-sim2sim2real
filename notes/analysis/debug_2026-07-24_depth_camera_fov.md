# Debug session — 2026-07-24: depth camera near-clip + FOV alignment

## Scope
Verify the depth-camera "near→far" clipping change, then investigate why sub-15 cm
pixels still read far on the live viewer, then align the real RealSense depth stream
to the MuJoCo/Isaac training render. Ended with a concrete fix (FOV auto-detection +
crop/pad) landed in `depth_camera_node.py`.

---

## 1. Near-clip change — verified correct
Files: `src/deploy_rl_policy/scripts/depth_camera_node.py`, `distill_common.py`

- **Change:** the old near gate `np.where(m < EMPTY_BELOW_M(=0.15), DEPTH_FAR, m)` was
  replaced by `np.where(m <= 0.0, DEPTH_FAR, m)`. `EMPTY_BELOW_M` fully removed
  (no lingering refs; grep clean).
- **Both mirror sites updated:** `image_msg_to_metres` (deploy) and `normalize_depth`
  (shared/sim), applied before crop+resize (holes don't bicubic-smear).
- **Validation:** training config `TRAIN_DEPTH_CFG["pw_empty_min"] = 0.0` → training
  never thresholds close pixels to far. So the new `<=0.0` rule *matches* training;
  the old 0.15 gate had no training counterpart. Change is correct for distribution.

## 2. Why sub-15 cm still reads far (user's viewer observation)
- On real hardware the **D435 returns `0` for pixels below its min-Z** (no valid stereo
  disparity). `0 <= 0.0` → `DEPTH_FAR`. So near surfaces still go to far — not because
  of a 0.15 gate (gone), but because the sensor emits `0` and the surviving rule maps
  `0`→far. Observation is expected, not a bug.
- The code change only preserves pixels the sensor *actually measures* in (0, 0.15) m,
  which at factory min-Z there usually are none. On sim/render paths (true depth, no
  min-Z), the change genuinely preserves close geometry.

## 3. Live camera settings (Jetson, `ros2 param get /camera/camera`)
Factory defaults, no close-range tuning:
- `clip_distance = -2.0` (disabled → driver not far-clipping; all far-mapping is our Python)
- `json_file_path = ""` (no advanced-mode preset → disparity shift ~0 → factory min-Z)
- `depth_module.profile = 640x480x30` (USB-3)
- `depth_module.visual_preset = 0` (Custom = factory table)
- `emitter_enabled = 1`, `laser_power = 150` → **raised to 350 this session** (fewer holes on low-texture)
- all filters OFF (decimation/spatial/temporal/hole_filling/disparity) → raw depth reaches our node = matches training assumption
- `align_depth = False`

**Action taken:** `ros2 param set /camera/camera depth_module.laser_power 350.0`
(runtime only — add `-p depth_module.laser_power:=350.0` to the `rs_launch.py` line to persist).

## 4. How the sim renders depth (training target)
From `distill_common.py` + `resources/go2/go2.xml:97`:
- Camera `front_depth_cam`: pos (0.33, 0, 0.08) trunk frame, pitched **20° down**, `fovy=56.49`, hFOV **87°**, grid **60×106**, centered pinhole.
- Range near 0.0 / far 2.0 m; **radial (spherical) far-clip** (`radial_maxperp_mask`) → circular corner vignette.
- **Stereo shadow** (`MujocoStereoDepth`): baseline **0.060 m**, eps 0.015, right-imager occlusion → far.
- **Parkour-in-the-Wild noise** (`TrainDepthDegrader`, `TRAIN_DEPTH_CFG` @ commit 90eb887): edge holes, drifting holes, `dis_noise=0.05`, `pw_empty_min=0.0`.
- Normalize: clip[0,2] → bicubic → 58×87 → `/2 − 0.5`.

**Already aligned (leave alone):** radial clip, far=2.0, filters off, `clip_distance` off, 1 mm depth units, bicubic resize, near-clip = `pw_empty_min` 0.0.

---

## 5. FINDING — real depth FOV ≠ assumed FOV  ⭐ main result
`/camera/depth/camera_info` (Foxy; used `| head`, `--once` unsupported):
```
width=640 height=480
k = [fx=389.4287, 0, cx=324.2679,
     0, fy=389.4287, cy=236.5089,
     0, 0, 1]
```
Computed:
- **Real depth FOV = 78.82° × 63.29°**  (hFOV = 2·atan(640/2fx), vFOV = 2·atan(480/2fy))
- Sim FOV = 87.00° × 56.49°
- Principal point offset: +4.8 px / −3.0 px (~0.7% → negligible)

**Impact of the old hard-coded `cam_hfov_deg=87, cam_vfov_deg=58`:**
`to_sim_render` only ever *cropped* (`min(1, SIM/cam)`), so it stretched the narrower
78.8° real image to fill sim's 87° frame → **+15.5% horizontal magnification** on
EVERY real deploy run. Vertical also wrong (assumed 58°, real 63.3°, cropped against
wrong reference). Silent sim2real distribution shift on the encoder's most sensitive input.

---

## 6. FIX 1 — implemented in `depth_camera_node.py`
1. **`to_sim_render` rewrite:** tan-space, per-axis **crop-OR-pad**.
   - Real wider than sim on an axis → center-crop the excess.
   - Real narrower → resize into central sub-region + **pad border with `DEPTH_FAR`**
     (angles never seen = empty = far, matching sim off-FOV behavior).
   - Reduces to plain bicubic 60×106 resize when FOVs match (backward compatible).
2. **Auto-FOV from `camera_info`:** new `auto_fov` param (default True) + optional
   `info_topic` (defaults to `<input>/../camera_info`). Subscribes, computes FOV from
   K, logs the override, unsubscribes (intrinsics static). Hard-coded 87×58 now a fallback.

For the measured 78.8×63.3 the new path: horizontal inner_w=92/106, **pad 14 px (7/side)**;
vertical crop 480→418 rows, inner_h=60.

Verified: `py_compile` OK; sizing/pad geometry validated numerically (exact-match →
no crop/pad; D435 640×480 → pad H / crop V; wider-both → crop only).

---

## Open items / next steps
- [ ] **Fix 0 gate:** confirm the deployed ckpt (OD-cliffside / D08 / D09) is
      depth-encoder-supervised, NOT a `no_enc_loss` family (go2.xml:84 warns the latent
      can be byte-identical for zero/near/far/random depth). Quick zeros-vs-real
      latent-diff test. If unsupervised, none of the depth alignment moves `|dlat|`.
- [ ] Launch stack; confirm startup log prints `camera_info FOV 78.8×63.3 … replaces param 87.0×58.0`.
- [ ] Record a bag; compare encoder `|dlat|` vs sim baseline (target ~1.5–1.7).
- [ ] Persist `laser_power=350` in the launch line.
- [ ] (Fix 4, training-side, low pri) sim stereo baseline 0.060 vs physical D435 ~0.050
      → occlusion shadows ~20% too wide; fix at next training render.
- [ ] (Fix 5, low pri) downsample character: real bicubic-averages 640×480→60×106
      (smoother) vs Warp single-ray (sharp); A/B area vs bicubic vs nearest by `|dlat|`.

## Files touched
- `src/deploy_rl_policy/scripts/depth_camera_node.py` — FOV auto-detect + crop/pad `to_sim_render` + math/CameraInfo imports
- `src/deploy_rl_policy/scripts/distill_common.py` — (near-clip change, pre-session)
- Memory: `project_realsense_depth_fov.md`
