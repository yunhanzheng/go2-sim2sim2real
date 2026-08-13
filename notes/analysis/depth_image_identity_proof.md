# Depth Image Identity Proof

**Claim:** the depth PiP in `view_prediction.py` shows the exact tensor fed to the depth encoder.

---

## Data flow

```
RealSense D435i
  → /camera/depth/image_rect_raw  (640×480, 16UC1, mm)
  → depth_camera_node.py          FOV crop → resize to 60×106 → normalize_depth → 58×87 float32 [-0.5, 0.5]
  → /rl/depth_image               (58×87, 32FC1)
        │                                │
        ▼                                ▼
  view_prediction.py            depth_encoder_node.py
  (display only)                (policy input)
```

Both nodes subscribe to **the same topic** and decode **identically**.

---

## Decoder comparison

**Encoder** (`depth_encoder_node.py:45–49, 200`):
```python
norm = np.frombuffer(msg.data, dtype=np.float32).reshape(58, 87)
depth_t = torch.from_numpy(norm).float().unsqueeze(0)  # (1, 58, 87) → encoder forward
```

**Viewer** (`view_prediction.py:_on_depth + _depth_pip`):
```python
d = np.frombuffer(bytes(msg.data), dtype="<f4").reshape(58, 87)  # stored as self.depth_norm
# display only — self.depth_norm is never modified:
u = ((depth_norm + 0.5).clip(0.0, 1.0) * 255).astype(np.uint8)
grey = cv2.resize(u, (pip_w, pip_h), interpolation=cv2.INTER_NEAREST)
```

---

## Why they are identical

| Property | Encoder | Viewer |
|---|---|---|
| Source topic | `/rl/depth_image` | `/rl/depth_image` |
| Encoding check | asserts `32FC1` | asserts `32FC1`, 58×87 |
| Decode | `frombuffer float32 reshape(58,87)` | `frombuffer float32 reshape(58,87)` |
| Shape fed forward | `(1, 58, 87)` torch tensor | displayed as-is |
| Interpolation | none | `INTER_NEAREST` (pixel replication only) |
| Value modification | none | `+0.5` shift for brightness **display only** |

The `+0.5` shift moves `[-0.5, 0.5]` → `[0, 1]` so near=black and far=white are visible.
It matches Isaac Lab's own depth display: `cv2.imshow(..., depth_buffer + 0.5)` in `legged_robot.py:281`.
`self.depth_norm` is never mutated before the encoder receives it.

---

## What to verify in Isaac

1. **normalize_depth output range**: confirm the training depth buffer is `float32` in `[-0.5, 0.5]` before it enters the encoder. Check `distill_common.py:normalize_depth` matches what Isaac writes.
2. **Crop/resize path**: Isaac renders at 60×106, crops `[:-2, 4:-4]` → 58×98, bicubic-resizes to 58×87. Confirm `legged_robot.py` uses the same crop before passing to the encoder.
3. **No-return pixel handling**: deploy remaps pixels `< 0.15 m` → `DEPTH_FAR` (A.1 fix, 2026-06-17). Check whether Isaac training did the same or left them as 0 / near.
4. **Display convention**: Isaac's `cv2.imshow(..., depth_buffer + 0.5)` should look identical to the viewer PiP. If they differ spatially, the FOV crop is off.
