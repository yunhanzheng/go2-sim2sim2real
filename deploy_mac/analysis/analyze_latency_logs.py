#!/usr/bin/env python3
"""Summarize the lightweight timing TSVs produced by LATENCY_LOG=1."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean


NS_TO_MS = 1e-6


def read_tsv(path: Path) -> list[dict[str, int | float]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    rows = []
    with path.open(newline="") as fp:
        for raw in csv.DictReader(fp, delimiter="\t"):
            row = {}
            for key, value in raw.items():
                if value is None or value == "":
                    row[key] = math.nan
                else:
                    try:
                        row[key] = (float(value)
                                    if any(ch in value for ch in ".eE")
                                    else int(value))
                    except ValueError:
                        row[key] = value
            rows.append(row)
    return rows


def percentile(values: list[float], fraction: float) -> float:
    values = sorted(v for v in values if math.isfinite(v))
    if not values:
        return math.nan
    position = fraction * (len(values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def print_stats(label: str, values: list[float]) -> None:
    values = [v for v in values if math.isfinite(v)]
    if not values:
        print(f"{label:46s} n=0")
        return
    print(
        f"{label:46s} n={len(values):5d}  mean={mean(values):9.3f}  "
        f"p50={percentile(values, 0.50):9.3f}  "
        f"p90={percentile(values, 0.90):9.3f}  "
        f"p99={percentile(values, 0.99):9.3f}  max={max(values):9.3f} ms"
    )


def delta_ms(rows, end: str, start: str) -> list[float]:
    return [(float(row[end]) - float(row[start])) * NS_TO_MS for row in rows]


def pair_camera_encoder(camera, encoder):
    camera_by_stamp = defaultdict(list)
    for row in camera:
        camera_by_stamp[(row["source_sec"], row["source_nsec"])].append(row)
    occurrence = defaultdict(int)
    pairs = []
    camera_for_depth_seq = {}
    for enc in encoder:
        key = (enc["source_sec"], enc["source_nsec"])
        index = occurrence[key]
        occurrence[key] += 1
        choices = camera_by_stamp.get(key, [])
        if index < len(choices):
            cam = choices[index]
            pairs.append((cam, enc))
            camera_for_depth_seq[int(enc["depth_seq"])] = cam
    return pairs, camera_for_depth_seq


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    root = args.run_dir

    camera = read_tsv(root / "latency_depth_camera.tsv")
    camera_metadata = read_tsv(root / "latency_camera_metadata.tsv")
    encoder = read_tsv(root / "latency_depth_encoder.tsv")
    actor = read_tsv(root / "latency_actor.tsv")
    lowstate = read_tsv(root / "latency_actor_lowstate.tsv")
    lowlevel = read_tsv(root / "latency_low_level.tsv")

    print(f"Latency summary: {root}")
    print(
        "Rows: camera={} camera_metadata={} encoder={} actor={} actor_lowstate={} low_level={}".format(
            len(camera), len(camera_metadata), len(encoder), len(actor),
            len(lowstate), len(lowlevel)
        )
    )
    print("\nObservable host-side segments")
    print_stats("camera queue: receipt -> processing", delta_ms(
        camera, "process_start_mono_ns", "receive_mono_ns"))
    print_stats("camera preprocessing", delta_ms(
        camera, "process_end_mono_ns", "process_start_mono_ns"))
    print_stats("camera node: receipt -> image publish", delta_ms(
        camera, "publish_end_mono_ns", "receive_mono_ns"))

    metadata_by_stamp = {
        (row["source_sec"], row["source_nsec"]): row
        for row in camera_metadata
    }
    camera_metadata_pairs = [
        (cam, metadata_by_stamp[(cam["source_sec"], cam["source_nsec"])])
        for cam in camera
        if (cam["source_sec"], cam["source_nsec"]) in metadata_by_stamp
    ]
    exposure_to_host = [
        (cam["receive_wall_ns"] - md["exposure_global_ns"]) * NS_TO_MS
        for cam, md in camera_metadata_pairs
    ]
    print_stats("camera exposure -> Mac image receipt", exposure_to_host)
    print_stats("camera exposure -> Jetson USB backend", [
        (md["backend_timestamp_ms"] * 1_000_000 - md["exposure_global_ns"])
        * NS_TO_MS for _, md in camera_metadata_pairs
    ])
    print_stats("Jetson USB backend -> Mac image receipt", [
        (cam["receive_wall_ns"] - md["backend_timestamp_ms"] * 1_000_000)
        * NS_TO_MS for cam, md in camera_metadata_pairs
    ])

    camera_encoder_pairs, camera_for_depth_seq = pair_camera_encoder(camera, encoder)
    print_stats("processed image DDS delivery", [
        (enc["receive_mono_ns"] - cam["publish_end_mono_ns"]) * NS_TO_MS
        for cam, enc in camera_encoder_pairs
    ])
    print_stats("depth encoder inference", delta_ms(
        encoder, "inference_end_mono_ns", "inference_start_mono_ns"))
    print_stats("depth encoder actual sleep", delta_ms(
        encoder, "sleep_end_mono_ns", "sleep_start_mono_ns"))
    print_stats("encoder callback -> latent publish", delta_ms(
        encoder, "publish_end_mono_ns", "receive_mono_ns"))

    encoder_by_seq = {int(row["depth_seq"]): row for row in encoder}
    actor_by_target = {int(row["target_seq"]): row for row in actor}
    lowlevel_by_target = {int(row["target_seq"]): row for row in lowlevel}
    first_actor_by_depth = {}
    for row in actor:
        first_actor_by_depth.setdefault(int(row["depth_seq"]), row)
    print_stats("latent DDS delivery to actor", [
        (row["latent_receive_mono_ns"] - encoder_by_seq[seq]["publish_end_mono_ns"])
        * NS_TO_MS
        for seq, row in first_actor_by_depth.items() if seq in encoder_by_seq
    ])
    print_stats("actor selected lowstate age at target", [
        (row["publish_end_mono_ns"] - row["lowstate_receive_mono_ns"]) * NS_TO_MS
        for row in actor
    ])
    print_stats("actor compute before latency wait", delta_ms(
        actor, "inference_end_mono_ns", "inference_start_mono_ns"))
    print_stats("actor actual sleep", delta_ms(
        actor, "sleep_end_mono_ns", "sleep_start_mono_ns"))
    print_stats("actor timer entry -> target publish", delta_ms(
        actor, "publish_end_mono_ns", "tick_start_mono_ns"))
    if len(actor) > 1:
        print_stats("actor target publication period", [
            (actor[i]["publish_end_mono_ns"] - actor[i - 1]["publish_end_mono_ns"])
            * NS_TO_MS for i in range(1, len(actor))
        ])

    joined_targets = [
        (actor_by_target[seq], lowlevel_by_target[seq])
        for seq in actor_by_target.keys() & lowlevel_by_target.keys()
    ]
    print_stats("target DDS delivery to low-level", [
        (ll["target_receive_mono_ns"] - act["publish_end_mono_ns"]) * NS_TO_MS
        for act, ll in joined_targets
    ])
    print_stats("low-level target callback -> /lowcmd", [
        (ll["lowcmd_publish_end_mono_ns"] - ll["target_receive_mono_ns"]) * NS_TO_MS
        for _, ll in joined_targets
    ])
    print_stats("selected lowstate receipt -> /lowcmd", [
        (ll["lowcmd_publish_end_mono_ns"] - act["lowstate_receive_mono_ns"])
        * NS_TO_MS for act, ll in joined_targets
    ])

    depth_to_target = []
    depth_to_lowcmd = []
    for act in actor:
        cam = camera_for_depth_seq.get(int(act["depth_seq"]))
        if cam is None:
            continue
        depth_to_target.append(
            (act["publish_end_mono_ns"] - cam["receive_mono_ns"]) * NS_TO_MS)
        ll = lowlevel_by_target.get(int(act["target_seq"]))
        if ll is not None:
            depth_to_lowcmd.append(
                (ll["lowcmd_publish_end_mono_ns"] - cam["receive_mono_ns"])
                * NS_TO_MS)
    print_stats("depth host receipt -> actor target", depth_to_target)
    print_stats("depth host receipt -> /lowcmd", depth_to_lowcmd)

    print("\nClock and sampling diagnostics")
    header_age = [
        (row["receive_wall_ns"] -
         (row["source_sec"] * 1_000_000_000 + row["source_nsec"])) * NS_TO_MS
        for row in camera
    ]
    print_stats("camera header -> host receipt (cross-clock)", header_age)
    if header_age and abs(percentile(header_age, 0.50)) > 1000.0:
        print("WARNING: camera and host timestamps are not in a usable common clock domain.")
    elif exposure_to_host:
        p50_exposure = percentile(exposure_to_host, 0.50)
        if p50_exposure < 0.0 or p50_exposure > 1000.0:
            print("WARNING: exposure-to-host result is implausible; verify clock_sync.log.")
        else:
            print("Camera GLOBAL_TIME is usable; residual uncertainty is bounded by clock-sync/network jitter.")
    if len(lowstate) > 1:
        print_stats("observed /lowstate callback period", [
            (lowstate[i]["receive_mono_ns"] - lowstate[i - 1]["receive_mono_ns"])
            * NS_TO_MS for i in range(1, len(lowstate))
        ])

    print("\nNot observable from these logs")
    if not camera_metadata_pairs:
        print("- Camera exposure -> host requires latency_camera_metadata.tsv and synchronized clocks.")
    print("- Robot state sampling -> host receipt requires mapping the robot tick to host time.")
    print("- /lowcmd publication -> motor application requires a firmware acknowledgement or hardware measurement.")


if __name__ == "__main__":
    main()
