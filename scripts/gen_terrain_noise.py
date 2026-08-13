"""Generate the heightfield PNG used by scene_2hurdles.xml.

Approximates the Isaac approach-floor roughness from `flyball_wall_terrain`
(at the easy end of `roughness_height = (0.02, 0.06)` with
`downsampled_scale = 0.075`). Heights are positive-only so the existing flat
plane and the hurdle/wall boxes (with bottoms at z=0) don't need shifting.

Outputs:  resources/go2/terrain_noise.png   grayscale uint8
"""
import numpy as np
from PIL import Image

# ----- footprint covered by the hfield (matches scene_2hurdles.xml) -----
X0, X1 = -0.5, 6.5      # m, world x — clears the wall at x≥6.65
Y0, Y1 =  -2.0, 2.0     # m, world y — wider than the corridor

# Spatial cell size ~ Isaac's downsampled_scale.
CELL = 0.075                                                # m
NCOL = int(round((X1 - X0) / CELL))                         # along x
NROW = int(round((Y1 - Y0) / CELL))                         # along y

# Vertical range. Isaac samples per-env from (0.02, 0.06) symmetric, so the
# total peak-to-peak is 0.04..0.12. We pick the easy end and use one-sided
# bumps so we don't need to lift the obstacles or shift the plane.
ZMAX = 0.025                                                # m
STEP = 0.005                                                # m vertical quant
N_LEVELS = int(round(ZMAX / STEP)) + 1                      # 6 levels

# ----- generate noise -----
rng = np.random.default_rng(0)
heights = rng.uniform(0.0, 1.0, size=(NROW, NCOL))          # [0, 1]
# Quantize to 5 mm steps within [0, ZMAX]
heights = np.round(heights * (N_LEVELS - 1)) / (N_LEVELS - 1) * ZMAX

# ----- punch flat patches under each obstacle so they sit flush -----
def x_to_col(x):
    return int(round((x - X0) / (X1 - X0) * (NCOL - 1)))

def clear_x_band(x_lo, x_hi, pad_cells=1):
    c_lo = max(0, x_to_col(x_lo) - pad_cells)
    c_hi = min(NCOL, x_to_col(x_hi) + pad_cells + 1)
    heights[:, c_lo:c_hi] = 0.0

# Each hurdle is 0.18 m thick centred at world x=2.0 / 4.0
clear_x_band(2.0 - 0.09, 2.0 + 0.09)        # hurdle 1
clear_x_band(4.0 - 0.09, 4.0 + 0.09)        # hurdle 2

# ----- write PNG (uint8 0..255 mapped linearly to 0..ZMAX) -----
png = (heights / ZMAX * 255.0).clip(0, 255).astype(np.uint8)

import pathlib
out = pathlib.Path(__file__).resolve().parent.parent / "resources" / "go2" / "assets" / "terrain_noise.png"
out.parent.mkdir(parents=True, exist_ok=True)
Image.fromarray(png, mode="L").save(out)
print(f"[wrote] {out}  shape=({NROW}, {NCOL})  ZMAX={ZMAX}m  cells={CELL}m  levels={N_LEVELS}")
