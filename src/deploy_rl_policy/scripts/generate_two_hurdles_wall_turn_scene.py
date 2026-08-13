#!/usr/bin/env python3
"""Generate a MuJoCo scene that mirrors `parkour_two_hurdles_wall_turn_terrain`
(env_class==15) from extreme-parkour terrain.py.

Coordinate convention
---------------------
go2.xml places the robot at world (0, 0, 0.42) with quat (0.707, 0, 0, 0.707) —
a 90 deg yaw about +z. So the robot's body +x (forward) maps to world +y.
We therefore lay the terrain corridor along world +y, with the start platform
centered on the world origin so the robot sits on it. Mapping:

    terrain.x (forward) -> world.y
    terrain.y (lateral) -> world.x
    terrain.z            -> world.z

Simplifications relative to the Isaac heightfield
-------------------------------------------------
* Approach roughness (the +/- a few cm noise across the floor) is dropped.
* The wall is modelled as a single tilted thick box for the inclined surface.
  The flat top behind the ramp peak is not modelled (the robot never reaches
  it). The small `delta_z` front lip is opt-in via --include_delta_z.

Usage
-----
    python generate_two_hurdles_wall_turn_scene.py --difficulty 0.5
    python generate_two_hurdles_wall_turn_scene.py --difficulty 0.9 \
        --wall_angle_deg_range 70 90 --include_delta_z
"""

import argparse
import math
from pathlib import Path


def lerp(a, b, t):
    return a + (b - a) * t


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--difficulty", type=float, default=0.5,
                   help="curriculum scalar in [0, 1]; drives all linear interpolations")
    p.add_argument("--gap2", type=float, default=2.15,
                   help="centre-to-centre distance hurdle1 -> hurdle2 [m] "
                        "(in training this is uniform(1.5, 2.8); we fix it for reproducibility)")
    # ---- terrain function defaults (must mirror parkour_two_hurdles_wall_turn_terrain) ----
    p.add_argument("--platform_len", type=float, default=2.5)
    p.add_argument("--hurdle_thickness", type=float, default=0.18)
    p.add_argument("--hurdle_width", type=float, default=2.0)
    p.add_argument("--hurdle_height_range", nargs=2, type=float, default=(0.10, 0.30))
    p.add_argument("--hurdle_gap_range", nargs=2, type=float, default=(1.1, 2.4))
    p.add_argument("--wall_height", type=float, default=1.0)
    p.add_argument("--wall_width", type=float, default=3.0)
    p.add_argument("--wall_thickness", type=float, default=0.35,
                   help="kept for API parity; only used as box body half-thickness here")
    p.add_argument("--wall_angle_deg_range", nargs=2, type=float, default=(50.0, 90.0))
    p.add_argument("--wall_goal_height", type=float, default=0.26)
    p.add_argument("--runup_dist_range", nargs=2, type=float, default=(3.5, 5.5))
    p.add_argument("--lateral_offset", type=float, default=0.4)
    p.add_argument("--landing_extra_lateral_offset", type=float, default=-0.10)
    p.add_argument("--include_delta_z", action="store_true",
                   help="add a small front-lip step of delta_z m before the inclined ramp")
    p.add_argument("--delta_z_range", nargs=2, type=float, default=(0.05, 0.15))
    p.add_argument("--show_goals", dest="show_goals", action="store_true", default=True,
                   help="add coloured visual sites at the 8 goals (default)")
    p.add_argument("--no_show_goals", dest="show_goals", action="store_false",
                   help="omit goal sites")
    p.add_argument("--out", type=str,
                   default=str(Path(__file__).parents[3] / "resources" / "go2"
                               / "scene_two_hurdles_wall_turn.xml"))
    args = p.parse_args()

    d = max(0.0, min(1.0, args.difficulty))

    # ---- curriculum-driven scalars (deterministic; no jitter) ----
    hurdle_h1 = lerp(args.hurdle_height_range[0], args.hurdle_height_range[1], d)
    hurdle_h2 = lerp(args.hurdle_height_range[0], args.hurdle_height_range[1], d)
    gap1      = lerp(args.hurdle_gap_range[0],    args.hurdle_gap_range[1],    d)
    gap2      = args.gap2
    runup     = lerp(args.runup_dist_range[0],    args.runup_dist_range[1],    d)
    theta_deg = lerp(args.wall_angle_deg_range[0], args.wall_angle_deg_range[1], d)
    theta     = math.radians(theta_deg)
    delta_z   = lerp(args.delta_z_range[0], args.delta_z_range[1], d) if args.include_delta_z else 0.0

    tan_t = max(math.tan(theta), 1e-6)
    sin_t = max(math.sin(theta), 1e-6)
    cos_t = math.cos(theta)

    # ---- corridor layout along world +y; platform centred on origin ----
    p_len = args.platform_len
    platform_y0 = -p_len / 2.0
    platform_y1 =  p_len / 2.0
    h_thk_half = args.hurdle_thickness / 2.0

    h1_centre_y = platform_y1 + gap1                  # = p_len/2 + gap1 (matches terrain: p_len + gap1 from start)
    h2_centre_y = h1_centre_y + gap2
    h2_back_y   = h2_centre_y + h_thk_half
    wall_front_y = h2_back_y + runup

    # ---- wall as a tilted thick box ----
    # The inclined surface goes from (y=wall_front_y, z=delta_z) up to
    # (y=wall_front_y + bottom_len, z=delta_z + wall_height).
    bottom_len  = args.wall_height / tan_t
    surface_len = args.wall_height / sin_t
    half_lat    = args.wall_width / 2.0
    half_len    = surface_len / 2.0
    half_thk    = 0.15  # body half-thickness perpendicular to surface

    # surface midpoint in world coords
    surf_mid_y = wall_front_y + bottom_len / 2.0
    surf_mid_z = delta_z + args.wall_height / 2.0
    # surface normal (toward approaching robot): (0, -sin t, +cos t)
    # box centre = surf_mid + (-normal) * half_thk
    cx = 0.0
    cy = surf_mid_y + sin_t * half_thk
    cz = surf_mid_z - cos_t * half_thk

    # box-local axes: +x = lateral (world +x), +y = up the ramp, +z = surface normal.
    # Rotation about world +x by +theta gives this mapping.
    qw = math.cos(theta / 2.0)
    qx = math.sin(theta / 2.0)

    wall_geom = (
        f'    <geom name="wall_ramp" type="box" group="4" '
        f'pos="{cx:.4f} {cy:.4f} {cz:.4f}" '
        f'size="{half_lat:.4f} {half_len:.4f} {half_thk:.4f}" '
        f'quat="{qw:.6f} {qx:.6f} 0 0" '
        f'rgba="0.55 0.55 0.60 1" friction="1.0 0.005 0.0001"/>\n'
    )

    # ---- optional front lip (matches the delta_z step in the heightfield) ----
    lip_geom = ""
    if args.include_delta_z and delta_z > 1e-4:
        lip_thk_half = args.wall_thickness / 2.0
        lip_z_half   = delta_z / 2.0
        lip_pos_y    = wall_front_y + lip_thk_half
        lip_pos_z    = delta_z / 2.0
        lip_geom = (
            f'    <geom name="wall_lip" type="box" group="4" '
            f'pos="0 {lip_pos_y:.4f} {lip_pos_z:.4f}" '
            f'size="{half_lat:.4f} {lip_thk_half:.4f} {lip_z_half:.4f}" '
            f'rgba="0.45 0.45 0.50 1"/>\n'
        )

    # ---- hurdles ----
    h1_z_half = hurdle_h1 / 2.0
    h2_z_half = hurdle_h2 / 2.0
    hurdle1_geom = (
        f'    <geom name="hurdle1" type="box" group="4" '
        f'pos="0 {h1_centre_y:.4f} {h1_z_half:.4f}" '
        f'size="{args.hurdle_width/2:.4f} {h_thk_half:.4f} {h1_z_half:.4f}" '
        f'rgba="0.85 0.55 0.20 1"/>\n'
    )
    hurdle2_geom = (
        f'    <geom name="hurdle2" type="box" group="4" '
        f'pos="0 {h2_centre_y:.4f} {h2_z_half:.4f}" '
        f'size="{args.hurdle_width/2:.4f} {h_thk_half:.4f} {h2_z_half:.4f}" '
        f'rgba="0.85 0.55 0.20 1"/>\n'
    )

    # ---- 8 goals (analytic, mirroring terrain code) ----
    yC = 0.0
    base_z = 0.0
    g0_y = (h1_centre_y - h_thk_half) - 0.6
    g2_y = (h2_centre_y + h_thk_half) + 0.8
    g1_y = (h1_centre_y + h2_centre_y) / 2.0
    g3_y = wall_front_y - 1.0  # terrain uses (-pit_width - 1.0); env_class==15 has no pit
    contact_dy = args.wall_goal_height / tan_t
    g4 = (yC,                                        wall_front_y + contact_dy, delta_z + args.wall_goal_height)
    g0 = (yC,                                        g0_y,                      base_z)
    g1 = (yC,                                        g1_y,                      base_z)
    g2 = (yC,                                        g2_y,                      base_z)
    g3 = (-args.lateral_offset,                      g3_y,                      base_z)
    g5 = (args.lateral_offset + args.landing_extra_lateral_offset,
          g3_y, base_z)
    g6 = g1
    g7 = g0
    goals = [g0, g1, g2, g3, g4, g5, g6, g7]
    goal_colors = [
        (0.20, 0.20, 0.20),  # g0 black     — start
        (0.30, 0.55, 0.85),  # g1 blue      — between hurdles
        (0.30, 0.55, 0.85),  # g2 blue      — after hurdle 2
        (0.95, 0.75, 0.10),  # g3 yellow    — wp_takeoff
        (0.90, 0.25, 0.20),  # g4 red       — wp_contact (on wall)
        (0.15, 0.75, 0.40),  # g5 green     — wp_land
        (0.30, 0.55, 0.85),  # g6 blue      — return
        (0.20, 0.20, 0.20),  # g7 black     — return start
    ]
    goal_xml = ""
    if args.show_goals:
        for i, ((gx, gy, gz), (r, gg, b)) in enumerate(zip(goals, goal_colors)):
            goal_xml += (
                f'    <site name="g{i}" pos="{gx:.3f} {gy:.3f} {max(gz, 0.02):.3f}" '
                f'size="0.07" rgba="{r:.2f} {gg:.2f} {b:.2f} 0.9" group="3"/>\n'
            )

    # ---- assemble scene ----
    xml = f"""<mujoco model="go2 two_hurdles_wall_turn">
  <include file="go2.xml" />

  <statistic center="0 5 0.5" extent="6" />

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0" />
    <rgba haze="0.15 0.25 0.35 1" />
    <global azimuth="-90" elevation="-25" />
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072" />
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300" />
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2" />
  </asset>

  <worldbody>
    <light pos="0 5 4" dir="0 0 -1" directional="true" />
    <geom name="floor" group="4" size="0 0 0.05" type="plane" material="groundplane" />

    <!-- difficulty={d:.2f}  hurdle_h={hurdle_h1:.2f}m  gap1={gap1:.2f}m  gap2={gap2:.2f}m  runup={runup:.2f}m  wall_angle={theta_deg:.1f}deg  delta_z={delta_z:.3f}m -->
{hurdle1_geom}{hurdle2_geom}{lip_geom}{wall_geom}{goal_xml}  </worldbody>
</mujoco>
"""

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(xml)

    print(f"Wrote {out_path}")
    print()
    print(f"Layout (along world +y, robot at origin facing +y):")
    print(f"  start platform : y in [{platform_y0:+.2f}, {platform_y1:+.2f}]")
    print(f"  hurdle 1       : y={h1_centre_y:+.2f}  h={hurdle_h1:.2f} m")
    print(f"  hurdle 2       : y={h2_centre_y:+.2f}  h={hurdle_h2:.2f} m")
    print(f"  wall front     : y={wall_front_y:+.2f}  delta_z={delta_z:.3f} m")
    print(f"  wall angle     : {theta_deg:.1f} deg   wall_height={args.wall_height:.2f} m")
    print(f"  ramp surface   : len={surface_len:.2f} m   bottom_len={bottom_len:.2f} m")
    print()
    print(f"Goals (world frame, x=lateral, y=forward, z=height):")
    for i, (gx, gy, gz) in enumerate(goals):
        print(f"  g{i}: x={gx:+.2f}  y={gy:+.2f}  z={gz:.2f}")


if __name__ == "__main__":
    main()
