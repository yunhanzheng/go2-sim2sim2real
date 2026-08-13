#!/usr/bin/env python3
"""View / export the depth images recorded in a rosbag — OFFLINE, no ROS graph.

Reads the bag's sqlite file directly, so nothing needs to be running and you do
not have to replay in real time. Works on both depth topics a real run records:

  /camera/depth/image_rect_raw   raw 16UC1 millimetres from the D435 (0 = invalid)
  /rl/depth_image                the 58x87 32FC1 policy input, normalized [-0.5, 0.5]

Colour mapping is shared by both: near = dark, far = bright, invalid = magenta,
so a hole in the raw stream is visually distinct from a genuine far reading.

  # contact sheet of the engaged window (quickest look)
  python deploy_mac/camera/export_bag_depth.py RUN/bag/bag_0.db3 --grid out.png

  # scrubbable video of the policy input
  python deploy_mac/camera/export_bag_depth.py RUN/bag/bag_0.db3 --mp4 depth.mp4

  # raw stream instead, only while the policy was engaged, as PNGs + .npy
  python deploy_mac/camera/export_bag_depth.py RUN/bag/bag_0.db3 --raw --engaged --png outdir/

  # list what is in the bag and stop
  python deploy_mac/camera/export_bag_depth.py RUN/bag/bag_0.db3 --list

`--engaged` restricts to the windows where /policy_engaged was true — usually the
only part you care about, and it is what the encoder actually consumed.
"""
import argparse
import sqlite3
from pathlib import Path

import numpy as np

RAW_TOPIC = "/camera/depth/image_rect_raw"
POLICY_TOPIC = "/rl/depth_image"


def _connect(bag: Path):
    if bag.is_dir():                       # accept the bag directory too
        cands = sorted(bag.glob("*.db3"))
        if not cands:
            raise SystemExit(f"no .db3 inside {bag}")
        bag = cands[0]
    if not bag.exists():
        raise SystemExit(f"no such bag: {bag}")
    return sqlite3.connect(str(bag)), bag


def list_topics(con):
    rows = con.execute(
        "select t.name, t.type, count(m.id), min(m.timestamp), max(m.timestamp) "
        "from topics t left join messages m on m.topic_id = t.id group by t.id"
    ).fetchall()
    print(f"{'topic':<34}{'count':>8}  {'dur_s':>7}  {'Hz':>6}  type")
    for name, typ, n, t0, t1 in sorted(rows, key=lambda r: -(r[2] or 0)):
        dur = (t1 - t0) / 1e9 if n and t1 > t0 else 0.0
        hz = n / dur if dur > 0 else 0.0
        print(f"{name:<34}{n or 0:>8}  {dur:>7.1f}  {hz:>6.1f}  {typ}")


def engaged_windows(con):
    """Windows (ns) where /policy_engaged was True. Empty list if not recorded."""
    row = con.execute("select id from topics where name='/policy_engaged'").fetchone()
    if row is None:
        return []
    # std_msgs/Bool CDR: 4-byte encapsulation header, then one byte payload.
    rows = con.execute(
        "select timestamp, data from messages where topic_id=? order by timestamp", (row[0],)
    ).fetchall()
    wins, t0, prev = [], None, None
    for ts, blob in rows:
        val = bool(bytes(blob)[4])
        if val != prev:
            if val:
                t0 = ts
            elif t0 is not None:
                wins.append((t0, ts))
            prev = val
    if prev and t0 is not None:            # still engaged at end of bag
        wins.append((t0, rows[-1][0]))
    return wins


def read_images(con, topic, wins=None):
    """Yield (timestamp_ns, HxW float array, is_raw). Decodes Image without ROS."""
    row = con.execute("select id, type from topics where name=?", (topic,)).fetchone()
    if row is None:
        raise SystemExit(f"topic {topic} not in bag (try --list)")
    tid = row[0]
    q = "select timestamp, data from messages where topic_id=? order by timestamp"
    for ts, blob in con.execute(q, (tid,)):
        if wins and not any(a <= ts <= b for a, b in wins):
            continue
        yield ts, _decode_image(bytes(blob))


def _decode_image(buf):
    """Minimal sensor_msgs/Image CDR parse: enough for 16UC1 / 32FC1."""
    o = 4                                                  # encapsulation header
    o += 8                                                 # header.stamp sec+nsec
    n = int.from_bytes(buf[o:o + 4], "little"); o += 4      # frame_id length
    o += n
    o = (o + 3) & ~3                                       # align to 4 for uint32
    height = int.from_bytes(buf[o:o + 4], "little"); o += 4
    width = int.from_bytes(buf[o:o + 4], "little"); o += 4
    n = int.from_bytes(buf[o:o + 4], "little"); o += 4      # encoding length
    enc = buf[o:o + n - 1].decode()
    # is_bigendian / step / the data sequence length follow, with alignment rules
    # that are fiddly to track by hand. height, width and encoding are enough to
    # know the payload size exactly, and the pixel data is the LAST thing in the
    # message — so slice it off the end instead of walking the remaining fields.
    dt = {"16UC1": np.uint16, "32FC1": np.float32}.get(enc)
    if dt is None:
        raise SystemExit(f"unhandled encoding {enc!r} on this topic")
    nbytes = height * width * np.dtype(dt).itemsize
    a = np.frombuffer(buf[len(buf) - nbytes:], dt).reshape(height, width)
    return a.astype(np.float32) * (1e-3 if dt is np.uint16 else 1.0)


def colourize(a, is_raw, far):
    """near dark -> far bright; invalid (raw 0) magenta so holes are obvious."""
    if is_raw:
        invalid = a <= 0.0
        v = np.clip(a / far, 0, 1)
    else:
        invalid = np.zeros_like(a, bool)
        v = np.clip(a + 0.5, 0, 1)          # normalized [-0.5,0.5] -> [0,1]
    g = (v * 255).astype(np.uint8)
    rgb = np.dstack([g, g, g])
    rgb[invalid] = (255, 0, 255)
    return rgb


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("bag", type=Path, help="path to bag_0.db3 (or the bag directory)")
    p.add_argument("--raw", action="store_true",
                   help=f"use {RAW_TOPIC} instead of {POLICY_TOPIC}")
    p.add_argument("--topic", help="explicit topic name, overrides --raw")
    p.add_argument("--engaged", action="store_true",
                   help="only frames inside /policy_engaged==True windows")
    p.add_argument("--every", type=int, default=1, help="keep every Nth frame")
    p.add_argument("--limit", type=int, help="stop after N kept frames")
    p.add_argument("--far", type=float, default=2.0,
                   help="metres mapped to white for --raw (default 2.0 = policy far clip)")
    p.add_argument("--scale", type=int, default=4, help="integer upscale for output")
    p.add_argument("--fps", type=int, default=20, help="--mp4 frame rate")
    p.add_argument("--cols", type=int, default=4, help="--grid columns")
    p.add_argument("--list", action="store_true", help="list topics and exit")
    p.add_argument("--mp4", type=Path, help="write a video here")
    p.add_argument("--grid", type=Path, help="write a contact sheet PNG here")
    p.add_argument("--png", type=Path, help="write per-frame PNG (+ .npy) into this dir")
    args = p.parse_args()

    con, bagpath = _connect(args.bag)
    if args.list:
        list_topics(con)
        return

    topic = args.topic or (RAW_TOPIC if args.raw else POLICY_TOPIC)
    is_raw = topic == RAW_TOPIC
    wins = engaged_windows(con) if args.engaged else None
    if args.engaged:
        if not wins:
            raise SystemExit("--engaged: no /policy_engaged windows found in this bag")
        total = sum(b - a for a, b in wins) / 1e9
        print(f"{len(wins)} engaged window(s), {total:.1f} s total")

    frames, stamps = [], []
    for i, (ts, img) in enumerate(read_images(con, topic, wins)):
        if i % args.every:
            continue
        frames.append(img)
        stamps.append(ts)
        if args.limit and len(frames) >= args.limit:
            break
    if not frames:
        raise SystemExit("no frames matched")
    h, w = frames[0].shape
    print(f"{topic}: {len(frames)} frames kept, {w}x{h}")
    if not (args.mp4 or args.grid or args.png):
        print("(nothing written — pass --grid, --mp4 or --png)")
        return

    from PIL import Image as PImage

    def tile(a):
        rgb = colourize(a, is_raw, args.far)
        im = PImage.fromarray(rgb)
        return im.resize((w * args.scale, h * args.scale), PImage.NEAREST)

    if args.png:
        args.png.mkdir(parents=True, exist_ok=True)
        for k, (ts, a) in enumerate(zip(stamps, frames)):
            tile(a).save(args.png / f"{k:05d}.png")
            np.save(args.png / f"{k:05d}.npy", a)
        print(f"wrote {len(frames)} png+npy -> {args.png}")

    if args.grid:
        tiles = [np.array(tile(a)) for a in frames]
        cols = max(1, args.cols)
        rows = [np.hstack(tiles[i:i + cols])
                for i in range(0, len(tiles) - cols + 1, cols)]
        if rows:
            PImage.fromarray(np.vstack(rows)).save(args.grid)
            print(f"wrote contact sheet -> {args.grid}")
        else:
            PImage.fromarray(tiles[0]).save(args.grid)
            print(f"wrote single frame -> {args.grid} (fewer frames than --cols)")

    if args.mp4:
        import imageio.v2 as imageio
        t0 = stamps[0]
        with imageio.get_writer(str(args.mp4), fps=args.fps, macro_block_size=1) as wr:
            for ts, a in zip(stamps, frames):
                wr.append_data(np.array(tile(a)))
        print(f"wrote {args.mp4}  ({len(frames)} frames @ {args.fps} fps, "
              f"spans {(stamps[-1]-t0)/1e9:.1f} s of run time)")


if __name__ == "__main__":
    main()
