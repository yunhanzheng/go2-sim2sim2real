#!/usr/bin/env python3
"""Recompute /rl/depth_image OFFLINE from a bag's recorded raw depth, optionally
with extra filtering, and write a new bag holding the ORIGINAL and the RECOMPUTED
stream side by side so they can be compared frame-for-frame in Foxglove.

The real deploy depth path is deterministic (no noise is injected on the real
path: normalize_depth's depth_noise_m defaults to 0.0), so replaying the recorded
/camera/depth/image_rect_raw through the same code reproduces /rl/depth_image
EXACTLY. The tool checks that first (--validate) and reports the max abs
difference; a nonzero result means the replay does not model what the node did,
and any conclusion drawn from a filtered variant would be unsound.

Pipeline replayed (depth_camera_node.py + distill_common.normalize_depth):
    decode 16UC1 -> metres
    [--holes-fill]  <-- inserted HERE, where the RealSense SDK filter would sit:
                        before the <0.15 m -> FAR remap, because after it there
                        are no holes left to fill (they have become FAR).
    < empty_below_m -> FAR
    to_sim_render   tan-space crop / resize / pad  -> 60x106
    radial far clip
    normalize_depth crop[:-2,4:-4] -> clamp -> bicubic 58x87 -> clamp -> /2 - 0.5

  python deploy_mac/camera/recompute_depth_bag.py RUN/bag --holes-fill 1 -o RUN/depth_holefill_compare
  python deploy_mac/camera/recompute_depth_bag.py RUN/bag --validate-only

  # append the PRE-hole-fill stream to the recorded bag itself, so Foxglove shows
  # /rl/depth_image (filled, what the policy saw) next to the unfilled variant:
  python deploy_mac/camera/recompute_depth_bag.py RUN/bag --holes-fill -1 \
      --in-place --in-place-topic /rl/depth_image_prehole

Hole-fill modes mirror librealsense rs2_hole_filling_filter:
 -1 none                  no filling (the pre-filter stream)
  0 fill_from_left        row-wise forward fill from the last valid pixel
  1 farthest_from_around  iterative max of valid 8-neighbours   (SDK default)
  2 nearest_from_around   iterative min of valid 8-neighbours

VALIDATION always replays with --recorded-holes, i.e. what depth_camera_node's
`hole_fill` param was set to for the run (default 1, ON since 2026-07-28). It must
reproduce the recorded /rl/depth_image bit-exactly; if it does not, the replay does
not model the node and the recomputed variant is not comparable to it.
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src" / "deploy_rl_policy" / "scripts"))

import cv2                                              # noqa: E402
import distill_common as C                              # noqa: E402
import depth_camera_node as N                           # noqa: E402

RAW_TOPIC = "/camera/depth/image_rect_raw"
OUT_TOPIC = "/rl/depth_image"


# --------------------------------------------------------------------------- io
def _bagfile(p: Path) -> Path:
    if p.is_dir():
        c = sorted(p.glob("*.db3"))
        if not c:
            raise SystemExit(f"no .db3 inside {p}")
        return c[0]
    return p


def read_msgs(db3: Path, topic: str):
    """Yield (bag_ts_ns, header_stamp_ns, deserialized msg)."""
    import sqlite3
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    con = sqlite3.connect(str(db3))
    row = con.execute("select id, type from topics where name=?", (topic,)).fetchone()
    if row is None:
        raise SystemExit(f"topic {topic} not in bag")
    tid, ttype = row
    M = get_message(ttype)
    for ts, blob in con.execute(
            "select timestamp, data from messages where topic_id=? order by timestamp", (tid,)):
        m = deserialize_message(bytes(blob), M)
        hs = None
        if hasattr(m, "header"):
            hs = m.header.stamp.sec * 1_000_000_000 + m.header.stamp.nanosec
        yield ts, hs, m


# ------------------------------------------------------------------- hole fill
# Imported from distill_common so the offline replay and the live deploy node
# (depth_camera_node, hole_fill ON by default) are byte-identical — the 2026-07-28
# training recalibration was measured through this exact function.
from distill_common import hole_fill  # noqa: E402


# -------------------------------------------------------------------- pipeline
def pipeline(raw_mm: np.ndarray, hfov: float, vfov: float,
             holes: int | None, holes_iter: int, radial: bool,
             radial_mask: np.ndarray) -> np.ndarray:
    m = raw_mm.astype(np.float32) * 1e-3
    if holes is not None:
        m = hole_fill(m, holes, holes_iter)
    # depth_camera_node.image_msg_to_metres:65
    m = np.where(m < C.MUJOCO_EMPTY_BELOW_M, np.float32(C.DEPTH_FAR), m).astype(np.float32)
    if m.shape != (C.DEPTH_RAW_H, C.DEPTH_RAW_W):
        m = N.to_sim_render(m, hfov, vfov)
    if radial and m.shape == radial_mask.shape:
        m = np.where(m > radial_mask, np.float32(C.DEPTH_FAR), m).astype(np.float32)
    return C.normalize_depth(m).numpy()[0]


# ------------------------------------------------------------------- in-place
def write_in_place(db3: Path, topic: str, rec: dict, fov_for, holes, holes_iter,
                   radial: bool, radial_mask: np.ndarray, limit=None):
    """APPEND a recomputed depth stream to the recorded bag as a new topic.

    Append-only by construction: one INSERT into `topics`, N INSERTs into
    `messages`, nothing existing is read-modify-written. Each new message reuses
    the BAG TIMESTAMP of the recorded /rl/depth_image frame it pairs with, so
    Foxglove shows the two images on the same playback tick.

    ALL frames are computed first and written in one executemany. Interleaving a
    read connection with an open write transaction on the same multi-GB .db3 is
    pathologically slow (~0.7 s/row observed); the compute itself is ~6 s for a
    full run, so the read/write split is what makes this finish in minutes.
    """
    import sqlite3
    from rclpy.serialization import serialize_message

    # ---- phase 1: read + compute (no write connection open)
    rows, changed, n = [], [], 0
    for _, hs, msg in read_msgs(db3, RAW_TOPIC):
        if limit and n >= limit:
            break
        if hs not in rec:
            continue
        ref_ts, ref = rec[hs]
        raw = np.frombuffer(bytes(msg.data), np.uint16).reshape(msg.height, msg.width)
        fh, fv = fov_for(ref_ts)
        new = pipeline(raw, fh, fv, holes, holes_iter, radial, radial_mask)
        gold = np.frombuffer(bytes(ref.data), np.float32).reshape(ref.height, ref.width)
        changed.append(float(np.abs(new - gold).max()))
        out = N.metres_to_image_msg(new, msg.header.stamp, frame_id=msg.header.frame_id)
        rows.append((ref_ts, sqlite3.Binary(serialize_message(out))))
        n += 1
    print(f"computed {n} frames, writing...")

    # ---- phase 2: single write transaction
    con = sqlite3.connect(str(db3))
    try:
        if con.execute("select 1 from topics where name=?", (topic,)).fetchone():
            raise SystemExit(f"{topic} already exists in {db3.name} — pick another "
                             "--in-place-topic (this tool never overwrites)")
        qos = con.execute("select offered_qos_profiles from topics where name=?",
                          (OUT_TOPIC,)).fetchone()[0]
        tid = con.execute("select coalesce(max(id),0)+1 from topics").fetchone()[0]
        con.execute("insert into topics(id,name,type,serialization_format,"
                    "offered_qos_profiles) values (?,?,?,?,?)",
                    (tid, topic, "sensor_msgs/msg/Image", "cdr", qos))
        con.executemany("insert into messages(topic_id,timestamp,data) values (?,?,?)",
                        [(tid, ts, blob) for ts, blob in rows])
        con.commit()
    finally:
        con.close()
    patch_metadata(db3, topic, n)

    ch = np.array(changed)
    print(f"appended {n} frames to {db3.name} as {topic}")
    print(f"  vs recorded {OUT_TOPIC}: mean max|Δ| = {ch.mean():.4f}  "
          f"p99 = {np.percentile(ch, 99):.4f}  max = {ch.max():.4f}  (normalized units)")
    print(f"  frames that differ at all: {(ch > 0).sum()}/{n}")


def patch_metadata(db3: Path, topic: str, count: int):
    """Add the new topic to metadata.yaml so `ros2 bag play/info` see it.

    Also repairs relative_file_paths/files[].path if the .db3 was renamed after
    recording (rosbag2 writes the original name and never revisits it).
    """
    import yaml
    meta_path = db3.parent / "metadata.yaml"
    if not meta_path.exists():
        print(f"  NOTE: no metadata.yaml beside {db3.name} — nothing to patch")
        return
    meta = yaml.safe_load(meta_path.read_text())
    info = meta["rosbag2_bagfile_information"]
    qos = next(t["topic_metadata"]["offered_qos_profiles"]
               for t in info["topics_with_message_count"]
               if t["topic_metadata"]["name"] == OUT_TOPIC)
    info["topics_with_message_count"].append({
        "topic_metadata": {"name": topic, "type": "sensor_msgs/msg/Image",
                           "serialization_format": "cdr",
                           "offered_qos_profiles": qos},
        "message_count": count})
    info["message_count"] += count
    for f in info["files"]:
        f["message_count"] += count
        f["path"] = db3.name
    info["relative_file_paths"] = [db3.name]
    meta_path.write_text(yaml.safe_dump(meta, sort_keys=False, default_flow_style=False))
    print(f"  metadata.yaml updated (+{count} msgs, file path -> {db3.name})")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag", type=Path, help="input bag dir or .db3")
    ap.add_argument("-o", "--out", type=Path, help="output bag directory")
    ap.add_argument("--holes-fill", type=int, choices=[-1, 0, 1, 2], default=1,
                    help="hole-fill mode for the recomputed stream (-1 = none, default 1)")
    ap.add_argument("--holes-iter", type=int, default=8)
    ap.add_argument("--recorded-holes", type=int, choices=[-1, 0, 1, 2], default=1,
                    help="hole_fill mode depth_camera_node RAN with for this bag "
                         "(-1 = none). Used for the validation replay only.")
    ap.add_argument("--no-radial", action="store_true", help="disable the radial far clip")
    ap.add_argument("--hfov", type=float, help="override camera hFOV (deg)")
    ap.add_argument("--vfov", type=float, help="override camera vFOV (deg)")
    ap.add_argument("--fallback-until-ns", type=int, default=None,
                    help="bag ts before which the node still used the fallback FOV")
    ap.add_argument("--fallback-hfov", type=float, default=87.0)
    ap.add_argument("--fallback-vfov", type=float, default=58.0)
    ap.add_argument("--with-baseline", action="store_true",
                    help="also write /rl/depth_image_recomputed (current code, NO hole fill). "
                         "Use when the recorded stream came from a different code version: "
                         "then baseline-vs-holefill isolates the filter, while "
                         "original-vs-baseline shows the code drift.")
    ap.add_argument("--in-place", action="store_true",
                    help="APPEND the recomputed stream to the INPUT bag as a new topic "
                         "instead of writing a new bag. Append-only: existing topics and "
                         "messages are untouched; metadata.yaml is patched to match.")
    ap.add_argument("--in-place-topic", default=None,
                    help="topic name for --in-place (default /rl/depth_image_holefill<N>, "
                         "or /rl/depth_image_prehole when --holes-fill -1)")
    ap.add_argument("--validate-only", action="store_true",
                    help="only check the unfiltered replay against the recorded stream")
    ap.add_argument("--validate-n", type=int, default=300)
    ap.add_argument("--limit", type=int, help="process at most N frames")
    args = ap.parse_args()

    db3 = _bagfile(args.bag)
    radial_mask = C.radial_maxperp_mask(C.DEPTH_RAW_H, C.DEPTH_RAW_W)
    radial = not args.no_radial

    # FOV: default to what a D435 reports for the recorded resolution.
    probe = next(read_msgs(db3, RAW_TOPIC))[2]
    W, H = probe.width, probe.height
    default_fov = {(480, 270): (89.2, 58.0), (640, 480): (78.8, 63.3)}.get((W, H))
    if args.hfov and args.vfov:
        hfov, vfov = args.hfov, args.vfov
    elif default_fov:
        hfov, vfov = default_fov
    else:
        raise SystemExit(f"unknown resolution {W}x{H}: pass --hfov/--vfov")
    print(f"raw {W}x{H}  camera FOV {hfov}x{vfov}  radial_clip={'on' if radial else 'off'}")

    def fov_for(ts):
        if args.fallback_until_ns and ts < args.fallback_until_ns:
            return args.fallback_hfov, args.fallback_vfov
        return hfov, vfov

    # -1 => "no hole filling" for pipeline(), which takes None
    holes_new = None if args.holes_fill < 0 else args.holes_fill
    holes_rec = None if args.recorded_holes < 0 else args.recorded_holes

    # ---- validation: replay AS THE NODE RAN must reproduce the recorded stream
    rec = {hs: (ts, m) for ts, hs, m in read_msgs(db3, OUT_TOPIC)}
    print(f"recorded {OUT_TOPIC}: {len(rec)} frames "
          f"(node hole_fill={args.recorded_holes})")
    diffs, checked, skipped = [], 0, 0
    for ts, hs, msg in read_msgs(db3, RAW_TOPIC):
        if checked >= args.validate_n:
            break
        if hs not in rec:
            skipped += 1
            continue
        ref = rec[hs][1]
        raw = np.frombuffer(bytes(msg.data), np.uint16).reshape(msg.height, msg.width)
        fh, fv = fov_for(ts)
        mine = pipeline(raw, fh, fv, holes_rec, args.holes_iter, radial, radial_mask)
        gold = np.frombuffer(bytes(ref.data), np.float32).reshape(ref.height, ref.width)
        diffs.append(float(np.abs(mine - gold).max()))
        checked += 1
    if checked:
        d = np.array(diffs)
        print(f"VALIDATION on {checked} frames "
              f"(replay at hole_fill={args.recorded_holes} vs recorded):")
        print(f"  max abs diff = {d.max():.3e}   frames exact = {(d == 0).sum()}/{checked}")
        if d.max() > 1e-6:
            print("  WARNING: replay does not reproduce the recorded stream exactly.")
    else:
        print("VALIDATION: no header-stamp matches — cannot verify")
    if args.validate_only:
        return

    if args.in_place:
        topic = args.in_place_topic or (
            "/rl/depth_image_prehole" if holes_new is None
            else f"/rl/depth_image_holefill{args.holes_fill}")
        write_in_place(db3, topic, rec, fov_for, holes_new, args.holes_iter,
                       radial, radial_mask, args.limit)
        return

    if not args.out:
        raise SystemExit("pass -o/--out to write a bag")

    # ---- write the comparison bag
    import rosbag2_py
    from rclpy.serialization import serialize_message
    from sensor_msgs.msg import Image
    outdir = args.out
    if outdir.exists():
        raise SystemExit(f"{outdir} exists — remove it or choose another name")
    writer = rosbag2_py.SequentialWriter()
    writer.open(rosbag2_py.StorageOptions(uri=str(outdir), storage_id="sqlite3"),
                rosbag2_py.ConverterOptions("", ""))
    NEW = f"/rl/depth_image_holefill{args.holes_fill}"
    BASE = "/rl/depth_image_recomputed"
    topics = [OUT_TOPIC, NEW] + ([BASE] if args.with_baseline else [])
    for t in topics:
        writer.create_topic(rosbag2_py.TopicMetadata(
            name=t, type="sensor_msgs/msg/Image", serialization_format="cdr"))
    writer.create_topic(rosbag2_py.TopicMetadata(
        name="/policy_engaged", type="std_msgs/msg/Bool", serialization_format="cdr"))

    n = 0
    changed = []
    for ts, hs, msg in read_msgs(db3, RAW_TOPIC):
        if args.limit and n >= args.limit:
            break
        ref = rec.get(hs)
        if ref is None:
            continue
        raw = np.frombuffer(bytes(msg.data), np.uint16).reshape(msg.height, msg.width)
        fh, fv = fov_for(ts)
        new = pipeline(raw, fh, fv, args.holes_fill, args.holes_iter, radial, radial_mask)
        base = (pipeline(raw, fh, fv, None, 0, radial, radial_mask)
                if args.with_baseline else None)
        # compare the filter against the like-for-like baseline when we have it,
        # else against the recorded stream
        gold = base if base is not None else np.frombuffer(
            bytes(ref.data), np.float32).reshape(ref.height, ref.width)
        changed.append(float(np.abs(new - gold).max()))
        out = N.metres_to_image_msg(new, msg.header.stamp, frame_id=msg.header.frame_id)
        writer.write(OUT_TOPIC, serialize_message(ref), ts)
        writer.write(NEW, serialize_message(out), ts)
        if base is not None:
            writer.write(BASE, serialize_message(
                N.metres_to_image_msg(base, msg.header.stamp,
                                      frame_id=msg.header.frame_id)), ts)
        n += 1
    for ts, _, m in read_msgs(db3, "/policy_engaged"):
        writer.write("/policy_engaged", serialize_message(m), ts)
    del writer

    ch = np.array(changed)
    print(f"wrote {n} frame pairs -> {outdir}")
    print(f"  topics: {OUT_TOPIC} (original)  |  {NEW} (recomputed, holes-fill="
          f"{args.holes_fill})  |  /policy_engaged")
    print(f"  effect of hole filling: mean max|Δ| = {ch.mean():.4f}  "
          f"p99 = {np.percentile(ch, 99):.4f}  max = {ch.max():.4f}  (normalized units)")
    print(f"  frames altered at all: {(ch > 0).sum()}/{n}")


if __name__ == "__main__":
    main()
