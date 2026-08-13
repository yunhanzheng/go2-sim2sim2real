#!/usr/bin/env python3
"""Generate a MuJoCo sim2sim scene mirroring ONE max-level tile of the Isaac
`parkour_flat` SHARP-SWITCHBACK corridor (extreme-parkour terrain.py, env_class
17, go2_flat, 2026-06-16 edit).

Why an hfield instead of boxes
------------------------------
The corridor centerline is a triangle wave (sharp ~90 deg corners) and the free
width pinches along the corridor — exactly the kind of geometry a MuJoCo
heightfield represents losslessly (it IS what Isaac stores internally). So we
replicate the Isaac carve here in pure numpy, dump it to a grayscale PNG, and
reference it as a `<hfield>`. The depth camera then sees byte-faithful walls,
corners and chokepoints — the whole point of sim2sim verification. Surface
roughness is intentionally omitted (clean walls/floor), matching the
scene_corridor_step / scene_flat_strip precedent (user: "we don't need surface
roughness").

Coordinate convention (matches the CURRENT go2.xml: base_link at 0 0 0.42,
quat 1 0 0 0 == identity, so the robot faces world +x):

    x_world = x_tile - 1     (robot spawns at tile-x 1 m -> world x 0, +x fwd)
    y_world = y_tile - 2     (corridor centred at tile-y 2 m -> world y 0)
    z_world = z_tile

Isaac tile: 18 m (x, forward) x 4 m (y, lateral), wall height 0.8 m, optional
full-width step-up (0.25 m) with its edge at tile-x 9 m. Defaults below mirror
go2_flat_config.py at difficulty = 1.0 (max curriculum level).

Usage
-----
    python generate_corridor_switchback_scene.py                 # width 1.5 m
    python generate_corridor_switchback_scene.py --width 1.0     # sharpest ~88 deg corners
    python generate_corridor_switchback_scene.py --no_step --no_pinch
"""

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image


# ---- Isaac terrain constants (mirror go2_flat_config.py / terrain.py) --------
TERRAIN_X = 18.0          # tile length along forward x [m]
TERRAIN_Y = 4.0           # tile width along lateral y [m]
HS        = 0.05          # horizontal_scale [m/px]
PLATFORM  = 2.5           # spawn platform_len [m]
SPAWN_TILE_X = 1.0        # env_origin_x within tile [m]
SPAWN_TILE_Y = 2.0        # env_origin_y within tile (centre) [m]


WALL_BOX_STEP = 5    # pixels between wall-box sample points (= 0.25 m at HS=0.05)
WALL_BOX_THK  = 0.2  # wall box thickness [m]


def _wall_boxes_xml(lo, hi, nx, wall_height):
    """Generate <geom type="box"> XML for left and right walls from lo/hi edge arrays.

    lo[x], hi[x] are inner-edge pixel coordinates (right/left wall respectively).
    Returns a string of XML lines (no surrounding tags).
    """
    thk_half = WALL_BOX_THK / 2.0
    h_half   = wall_height / 2.0
    lines = []

    xs = list(range(0, nx, WALL_BOX_STEP))
    if xs[-1] != nx - 1:
        xs.append(nx - 1)

    for side, edge, sign in (('L', hi, +1), ('R', lo, -1)):
        for i in range(len(xs) - 1):
            x1, x2 = xs[i], xs[i + 1]
            y1, y2  = float(edge[x1]), float(edge[x2])

            # Tile → world
            wx1 = x1 * HS - SPAWN_TILE_X;  wy1 = y1 * HS - SPAWN_TILE_Y
            wx2 = x2 * HS - SPAWN_TILE_X;  wy2 = y2 * HS - SPAWN_TILE_Y

            dx, dy = wx2 - wx1, wy2 - wy1
            seg = math.sqrt(dx * dx + dy * dy)
            if seg < 1e-6:
                continue

            # Outward normal: CCW (+1) for left wall, CW (-1) for right wall
            nx_n = -dy / seg * sign
            ny_n =  dx / seg * sign

            # Box centre = segment midpoint shifted outward by thk/2
            cx = (wx1 + wx2) / 2 + thk_half * nx_n
            cy = (wy1 + wy2) / 2 + thk_half * ny_n

            ang = math.atan2(dy, dx)
            qw  = math.cos(ang / 2);  qz = math.sin(ang / 2)

            lines.append(
                f'    <geom name="wall_{side}_{i:03d}" type="box" '
                f'pos="{cx:.4f} {cy:.4f} {h_half:.4f}" '
                f'quat="{qw:.6f} 0 0 {qz:.6f}" '
                f'size="{seg/2:.4f} {thk_half:.4f} {h_half:.4f}" '
                f'material="wall_face" friction="1.0 0.005 0.01"/>'
            )
    return '\n'.join(lines) + '\n'


def carve(width, amplitude, bends, pinch_amount, pinch_count, min_width,
          wall_height, step_height, difficulty, direction, seed):
    """Replicate the Isaac parkour_flat switchback carve for one tile.

    Returns (hf, goals_tile, lo, hi) where hf is (nx, ny) elevation in metres
    [axis0 = forward x, axis1 = lateral y], goals_tile is (8, 3) in tile
    metres (x, y, z), and lo/hi are inner-edge pixel arrays (shape nx)."""
    rng = np.random.RandomState(seed)
    nx = int(round(TERRAIN_X / HS))            # 360
    ny = int(round(TERRAIN_Y / HS))            # 80
    my = ny // 2                               # 40 (centerline px)
    plat_px = int(round(PLATFORM / HS))        # 50
    L = max(1, nx - plat_px)                   # 310
    half_px = int(round((width / 2.0) / HS))

    hf = np.zeros((nx, ny), dtype=np.float64)
    # Step-up: Isaac raises the whole width for x >= nx//2 (floor starts at 0,
    # so the raised floor == step_height), THEN walls are set absolute on top.
    step_h = step_height * difficulty
    sx = nx // 2
    if step_h > 0:
        hf[sx:, :] = step_h

    amp_px = amplitude * difficulty / HS
    amp_px = min(amp_px, max(0.0, float(my - half_px - 1)))
    if amp_px > 0:
        slope_est = 2.0 * amp_px * bends / L
        amp_px = min(amp_px, max(0.0, my - half_px * math.sqrt(1.0 + slope_est ** 2) - 1.0))

    goals_px = None
    if amp_px > 0:
        d = float(direction) if direction in (-1, 1) else float(rng.choice([-1.0, 1.0]))
        xs = np.arange(nx)
        t = np.clip((xs - plat_px) / L, 0.0, 1.0)
        tri = (2.0 / np.pi) * np.arcsin(np.sin(np.pi * bends * t))   # unit triangle, sharp corners
        cy = my + d * amp_px * tri
        # Per-column perpendicular half-width with along-corridor pinch.
        half_arr = np.full(nx, float(half_px))
        if pinch_amount > 0:
            ph = rng.uniform(0.0, 2.0 * np.pi)
            prof = 1.0 - pinch_amount * difficulty * (0.5 + 0.5 * np.cos(2.0 * np.pi * pinch_count * t + ph))
            min_half = (min_width / 2.0) / HS
            half_arr = np.maximum(half_px * prof, min_half)
        half_arr[xs < plat_px] = half_px                            # spawn platform full width
        # Distance-to-path TUBE: widen the horizontal band by sqrt(1+slope^2)
        # so perpendicular width survives the steep diagonal ramps.
        dl = np.abs(cy - np.roll(cy, 1))
        dr = np.abs(np.roll(cy, -1) - cy)
        slope = np.maximum(dl, dr)
        slope[0] = slope[-1] = 0.0
        horiz = half_arr * np.sqrt(1.0 + slope ** 2)
        lo = np.clip(np.round(cy - horiz).astype(int), 0, ny)
        hi = np.clip(np.round(cy + horiz).astype(int), 0, ny)
        ys = np.arange(ny)[None, :]
        wall_mask = (ys < lo[:, None]) | (ys >= hi[:, None])
        hf[wall_mask] = wall_height                                 # absolute (overwrites step)
        # Goals pinned to the switchback corners (+ start + end).
        gx = [float(plat_px - 1)]
        gy = [float(my)]
        for k in range(bends):
            gx.append(plat_px + ((k + 0.5) / bends) * L)
            sgn = 1.0 if (k % 2 == 0) else -1.0
            gy.append(my + d * amp_px * sgn)
        gx.append(float(nx - 1))
        gy.append(float(my))
        goals_px = np.stack([np.asarray(gx), np.asarray(gy)], axis=1)
    else:
        lo_val = max(0, my - half_px)
        hi_val = min(ny, my + half_px)
        lo = np.full(nx, lo_val, dtype=int)
        hi = np.full(nx, hi_val, dtype=int)
        hf[:, :lo_val] = wall_height
        hf[:, hi_val:] = wall_height
        # straight: 8 goals on the centerline, evenly along the corridor
        gx = np.linspace(plat_px - 1, nx - 1, 8)
        goals_px = np.stack([gx, np.full(8, float(my))], axis=1)

    # Pad / trim to exactly 8 goals (tail = last vertex), resolve z from hf.
    n_goals = 8
    g = np.repeat(goals_px[-1:], n_goals, axis=0).astype(float)
    m = min(goals_px.shape[0], n_goals)
    g[:m] = goals_px[:m]
    goals_tile = np.zeros((n_goals, 3))
    for i, (px, py) in enumerate(g):
        xi = int(np.clip(round(px), 0, nx - 1))
        yi = int(np.clip(round(py), 0, ny - 1))
        goals_tile[i] = [px * HS, py * HS, hf[xi, yi]]
    return hf, goals_tile, lo, hi


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--difficulty", type=float, default=1.0, help="curriculum scalar [0,1]; 1.0 = max level")
    p.add_argument("--width", type=float, default=1.5, help="free corridor width [m] (training samples [1,3])")
    p.add_argument("--amplitude", type=float, default=1.4, help="flat_corridor_curve_amplitude [m]")
    p.add_argument("--bends", type=int, default=6, help="flat_corridor_curve_bends (# corners = # interior goals)")
    p.add_argument("--pinch_amount", type=float, default=0.5, help="flat_corridor_pinch_amount (0 disables)")
    p.add_argument("--pinch_count", type=int, default=2, help="flat_corridor_pinch_count")
    p.add_argument("--min_width", type=float, default=0.8, help="flat_corridor_min_width [m] (pinch floor)")
    p.add_argument("--wall_height", type=float, default=0.8, help="flat_corridor_wall_height [m]")
    p.add_argument("--step_height", type=float, default=0.25, help="flat_step_height_max [m]; 0 disables")
    p.add_argument("--no_step", action="store_true", help="disable the mid-corridor step")
    p.add_argument("--no_pinch", action="store_true", help="disable the along-corridor width pinch")
    p.add_argument("--walls_only", action="store_true",
                   help="use box geoms for walls instead of hfield — floor is pure MuJoCo plane")
    p.add_argument("--direction", type=int, default=1, choices=(-1, 1), help="swing left(+1)/right(-1) first")
    p.add_argument("--seed", type=int, default=0, help="RNG seed (only affects pinch phase / random dir)")
    p.add_argument("--hurdle_x", type=float, nargs="*", default=None,
                   help="world-x position(s) [m] of step-over hurdle bar(s) across the corridor; "
                        "mirrors flat_corridor_hurdle (training: 2 bars, height<=0.15 m, thk 0.18 m). "
                        "Omit for no hurdle.")
    p.add_argument("--hurdle_height", type=float, default=0.12,
                   help="hurdle bar height [m] (flat_corridor_hurdle_height_max=0.15)")
    p.add_argument("--hurdle_thickness", type=float, default=0.18,
                   help="hurdle bar thickness along x [m] (flat_corridor_hurdle_thickness=0.18)")
    p.add_argument("--no_show_goals", dest="show_goals", action="store_false", default=True)
    repo = Path(__file__).resolve().parents[3]
    p.add_argument("--out", type=str, default=str(repo / "resources" / "go2" / "scene_corridor_switchback.xml"))
    p.add_argument("--png", type=str, default=str(repo / "resources" / "go2" / "assets" / "corridor_switchback.png"))
    args = p.parse_args()

    step_h = 0.0 if args.no_step else args.step_height
    pinch  = 0.0 if args.no_pinch else args.pinch_amount

    hf, goals_tile, lo, hi = carve(
        width=args.width, amplitude=args.amplitude, bends=args.bends,
        pinch_amount=pinch, pinch_count=args.pinch_count, min_width=args.min_width,
        wall_height=args.wall_height, step_height=step_h,
        difficulty=max(0.0, min(1.0, args.difficulty)),
        direction=args.direction, seed=args.seed,
    )
    nx, ny = hf.shape

    # ---- goal sites (tile -> world) -------------------------------------------
    goal_xml = ""
    if args.show_goals:
        n = goals_tile.shape[0]
        for i, (gxt, gyt, gzt) in enumerate(goals_tile):
            gx = gxt - SPAWN_TILE_X
            gy = gyt - SPAWN_TILE_Y
            gz = max(gzt, 0.0) + 0.03
            f = i / max(1, n - 1)
            r, g, b = (0.15, 0.55 * (1 - f) + 0.15, 0.30 + 0.55 * f)
            if i == 0:
                r, g, b = 0.1, 0.1, 0.1
            goal_xml += (f'    <site name="g{i}" pos="{gx:.3f} {gy:.3f} {gz:.3f}" '
                         f'size="0.07" rgba="{r:.2f} {g:.2f} {b:.2f} 0.9" group="3"/>\n')

    # ---- hurdle bars (tile -> world) ------------------------------------------
    # Thin full-tile-width boxes laid across the corridor; the robot must step
    # over them. z sits the bar on the floor (z=0 plane / hfield base). Mirrors
    # the Isaac flat_corridor_hurdle obstacles the policy trained on.
    hurdle_xml = ""
    if args.hurdle_x:
        h_thk_half = args.hurdle_thickness / 2.0
        h_half_z = args.hurdle_height / 2.0
        h_half_y = TERRAIN_Y / 2.0          # span the full tile width
        for j, hx in enumerate(args.hurdle_x):
            hurdle_xml += (
                f'    <geom name="hurdle_{j:02d}" type="box" '
                f'pos="{hx:.4f} 0.0000 {h_half_z:.4f}" '
                f'size="{h_thk_half:.4f} {h_half_y:.4f} {h_half_z:.4f}" '
                f'rgba="0.75 0.45 0.20 1.0" friction="1.0 0.005 0.01"/>\n'
            )

    corner_deg = "n/a"
    if goals_tile.shape[0] >= 3:
        angs = []
        for i in range(1, goals_tile.shape[0] - 1):
            v1 = goals_tile[i, :2] - goals_tile[i - 1, :2]
            v2 = goals_tile[i + 1, :2] - goals_tile[i, :2]
            if np.linalg.norm(v1) < 1e-6 or np.linalg.norm(v2) < 1e-6:
                continue
            c = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            angs.append(math.degrees(math.acos(max(-1.0, min(1.0, c)))))
        if angs:
            corner_deg = f"{np.mean(angs):.0f} (max {np.max(angs):.0f})"

    if args.walls_only:
        # ---- Box-geom walls: pure MuJoCo plane floor, two wall ribbons -------
        wall_geoms = _wall_boxes_xml(lo, hi, nx, args.wall_height)
        xml = f"""<mujoco model="go2 switchback corridor (walls only, difficulty={args.difficulty:.2f})">
  <!--
    Auto-generated by generate_corridor_switchback_scene.py — DO NOT hand-edit.

    Walls-only variant (--walls_only): box geoms approximate the curved switchback
    walls; floor is the MuJoCo infinite plane only (no hfield floor tiles).

    difficulty={args.difficulty:.2f}  width={args.width:.2f} m  amplitude={args.amplitude:.2f} m
    bends={args.bends}  corner turn={corner_deg} deg  step={step_h:.2f} m  pinch={pinch:.2f}
    wall height={args.wall_height:.2f} m  box thickness={WALL_BOX_THK:.2f} m  step={WALL_BOX_STEP * HS:.2f} m/seg
  -->
  <include file="go2.xml"/>

  <option timestep="0.005"/>
  <statistic center="7 0 0.3" extent="7"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="-130" elevation="-35" offwidth="1280" offheight="720"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
      markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
    <material name="wall_face" rgba="0.45 0.45 0.55 1.0"/>
  </asset>

  <worldbody>
    <light pos="6 0 3" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane" friction="1.0 0.005 0.01"/>

    <!-- Switchback corridor walls as box geom segments (no hfield floor). -->
{wall_geoms}
{hurdle_xml}{goal_xml}  </worldbody>
</mujoco>
"""
    else:
        # ---- hfield PNG -------------------------------------------------------
        png_arr = (hf.T / args.wall_height * 255.0).clip(0, 255).astype(np.uint8)[::-1, :]
        out_png = Path(args.png)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(png_arr, mode="L").save(out_png)

        radius_x = nx * HS / 2.0
        radius_y = ny * HS / 2.0
        center_x = (TERRAIN_X / 2.0) - SPAWN_TILE_X
        center_y = (TERRAIN_Y / 2.0) - SPAWN_TILE_Y
        png_name = out_png.name

        xml = f"""<mujoco model="go2 switchback corridor (max level)">
  <!--
    Auto-generated by generate_corridor_switchback_scene.py — DO NOT hand-edit;
    rerun the generator to change parameters.

    Mirrors ONE max-level (difficulty={args.difficulty:.2f}) tile of the Isaac
    parkour_flat sharp-switchback corridor (go2_flat, env_class 17):
      corridor width   = {args.width:.2f} m   (training samples [1,3])
      curve amplitude  = {args.amplitude:.2f} m,  {args.bends} switchback corners
      mean corner turn = {corner_deg} deg
      width pinch      = {pinch:.2f} (count {args.pinch_count}, floor {args.min_width:.2f} m)
      step-up          = {step_h:.2f} m at tile-x 9 m
      wall height      = {args.wall_height:.2f} m
    World mapping: x_world = x_tile - 1, y_world = y_tile - 2 (robot faces +x).
    Surface roughness intentionally omitted (clean walls/floor).
  -->
  <include file="go2.xml"/>

  <option timestep="0.005"/>
  <statistic center="7 0 0.3" extent="7"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="-130" elevation="-35" offwidth="1280" offheight="720"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
      markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
    <material name="corridor" rgba="0.50 0.50 0.58 1.0"/>
    <!-- size = radius_x radius_y elevation_z base_z ; data PNG normalised by wall_height -->
    <hfield name="corridor_switchback" file="{png_name}"
            size="{radius_x:.4f} {radius_y:.4f} {args.wall_height:.4f} 0.001"/>
  </asset>

  <worldbody>
    <light pos="6 0 3" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane" friction="1.0 0.005 0.01"/>

    <!-- Switchback corridor walls + floor (+ step) as a byte-faithful Isaac hfield. -->
    <geom name="corridor_switchback" type="hfield" hfield="corridor_switchback"
          pos="{center_x:.4f} {center_y:.4f} 0" material="corridor"
          friction="1.0 0.005 0.01"/>

{hurdle_xml}{goal_xml}  </worldbody>
</mujoco>
"""
        print(f"[wrote] {out_png}   shape=({ny},{nx})  (row=y, col=x)")
        print(f"  hfield: size=({radius_x:.2f} {radius_y:.2f} {args.wall_height:.2f} 0.001)  geom pos=({center_x:.2f} {center_y:.2f} 0)")

    out_xml = Path(args.out)
    out_xml.parent.mkdir(parents=True, exist_ok=True)
    out_xml.write_text(xml)

    print(f"[wrote] {out_xml}")
    print(f"  corridor width={args.width:.2f} m  amplitude={args.amplitude:.2f} m  bends={args.bends}  "
          f"mean corner turn={corner_deg} deg  step={step_h:.2f} m  pinch={pinch:.2f}  walls_only={args.walls_only}")
    print(f"  goals (world x,y,z):")
    for i, (gxt, gyt, gzt) in enumerate(goals_tile):
        print(f"    g{i}: x={gxt - SPAWN_TILE_X:+.2f}  y={gyt - SPAWN_TILE_Y:+.2f}  z={gzt:.2f}")


if __name__ == "__main__":
    main()
