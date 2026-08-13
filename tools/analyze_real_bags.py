#!/usr/bin/env python3
"""Offline analysis of real Go2 ROS 2 bags.

Reads bag SQLite databases directly so large raw camera topics do not need to be
replayed or scanned. Produces a JSON summary and depth-image contact sheets.

Run from the repository root after sourcing deploy_mac/env.sh:
    python tools/analyze_real_bags.py
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


ROOT = Path(__file__).resolve().parents[1]
REAL_ROOT = ROOT / "runs" / "real"
OUT_ROOT = ROOT / "runs" / "real_analysis"
FEET = ("FR", "FL", "RR", "RL")


def percentile(values, qs=(0, 1, 5, 25, 50, 75, 95, 99, 100)):
    if not values:
        return {}
    a = np.asarray(values, dtype=np.float64)
    return {str(q): float(np.percentile(a, q)) for q in qs}


def vector_stats(rows):
    if not rows:
        return {}
    a = np.asarray(rows, dtype=np.float64)
    return {
        "count": int(len(a)),
        "mean": a.mean(axis=0).tolist(),
        "std": a.std(axis=0).tolist(),
        "min": a.min(axis=0).tolist(),
        "max": a.max(axis=0).tolist(),
        "p95_abs": np.percentile(np.abs(a), 95, axis=0).tolist(),
        "max_abs": np.max(np.abs(a), axis=0).tolist(),
    }


def parse_meta(path):
    result = {}
    if not path.exists():
        return result
    for line in path.read_text(errors="replace").splitlines():
        if ":" not in line or line.startswith(" "):
            continue
        key, value = line.split(":", 1)
        result[key.strip()] = value.strip()
    return result


def parse_policy_log(path):
    result = {"engagement_events": [], "estimates": []}
    if not path.exists():
        return result
    epoch_re = re.compile(r"\[(\d+\.\d+)\]")
    est_re = re.compile(
        r"(?:\[cmd\]\s+vx=([+-]?[\d.]+)\s+m/s\s+\|\s+)?"
        r"\[est\]\s+vx=([+-]?[\d.]+)\s+vy=([+-]?[\d.]+)\s+vz=([+-]?[\d.]+)"
        r".*?\|dlat\|=([\d.]+)"
    )
    for line in path.read_text(errors="replace").splitlines():
        tm = epoch_re.search(line)
        if not tm:
            continue
        t = float(tm.group(1))
        if "policy_engaged=True" in line:
            result["engagement_events"].append([t, True])
        elif "policy_engaged=False" in line:
            result["engagement_events"].append([t, False])
        em = est_re.search(line)
        if em:
            cmd = float(em.group(1)) if em.group(1) is not None else None
            result["estimates"].append(
                [t, cmd, float(em.group(2)), float(em.group(3)),
                 float(em.group(4)), float(em.group(5))]
            )
    return result


class Bag:
    def __init__(self, db_path):
        self.path = db_path
        self.db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        rows = self.db.execute(
            "SELECT id, name, type FROM topics ORDER BY id"
        ).fetchall()
        self.topics = {name: {"id": tid, "type": typ} for tid, name, typ in rows}
        self.types = {}

    def close(self):
        self.db.close()

    def count(self, topic):
        info = self.topics.get(topic)
        if not info:
            return 0
        return self.db.execute(
            "SELECT COUNT(*) FROM messages WHERE topic_id=?", (info["id"],)
        ).fetchone()[0]

    def bounds(self, topic):
        info = self.topics.get(topic)
        if not info:
            return None
        row = self.db.execute(
            "SELECT MIN(timestamp), MAX(timestamp) FROM messages WHERE topic_id=?",
            (info["id"],),
        ).fetchone()
        if row[0] is None:
            return None
        return [row[0] / 1e9, row[1] / 1e9]

    def all_stats(self):
        rows = self.db.execute(
            "SELECT topic_id, COUNT(*), MIN(timestamp), MAX(timestamp) "
            "FROM messages GROUP BY topic_id"
        ).fetchall()
        by_id = {v["id"]: name for name, v in self.topics.items()}
        counts = {name: 0 for name in self.topics}
        bounds = {name: None for name in self.topics}
        for topic_id, count, t_min, t_max in rows:
            name = by_id[topic_id]
            counts[name] = int(count)
            bounds[name] = [t_min / 1e9, t_max / 1e9]
        return counts, bounds

    def iter_topics(self, topic_names, strides=None):
        strides = strides or {}
        selected = {self.topics[name]["id"]: name for name in topic_names if name in self.topics}
        if not selected:
            return
        placeholders = ",".join("?" for _ in selected)
        query = (
            f"SELECT topic_id, timestamp, data FROM messages "
            f"WHERE topic_id IN ({placeholders}) ORDER BY id"
        )
        classes = {
            topic_id: self.types.setdefault(
                self.topics[name]["type"], get_message(self.topics[name]["type"])
            )
            for topic_id, name in selected.items()
        }
        indexes = {}
        for topic_id, timestamp, data in self.db.execute(query, tuple(selected)):
            name = selected[topic_id]
            index = indexes.get(topic_id, 0)
            indexes[topic_id] = index + 1
            if index % int(strides.get(name, 1)):
                continue
            yield (
                name,
                timestamp / 1e9,
                deserialize_message(data, classes[topic_id]),
            )

    def iter_messages(self, topic, stride=1):
        info = self.topics.get(topic)
        if not info:
            return
        cls = self.types.setdefault(info["type"], get_message(info["type"]))
        query = (
            "SELECT timestamp, data FROM messages WHERE topic_id=? "
            "ORDER BY id"
        )
        for index, (timestamp, data) in enumerate(self.db.execute(query, (info["id"],))):
            if index % stride:
                continue
            yield timestamp / 1e9, deserialize_message(data, cls)

    def selected_messages(self, topic, count=12):
        info = self.topics.get(topic)
        total = self.count(topic)
        if not info or total == 0:
            return []
        cls = self.types.setdefault(info["type"], get_message(info["type"]))
        indexes = sorted(set(np.linspace(0, total - 1, min(count, total)).astype(int)))
        result = []
        for offset in indexes:
            row = self.db.execute(
                "SELECT timestamp, data FROM messages WHERE topic_id=? "
                "ORDER BY id LIMIT 1 OFFSET ?",
                (info["id"], int(offset)),
            ).fetchone()
            result.append((row[0] / 1e9, deserialize_message(row[1], cls)))
        return result


def compress_bool_events(events):
    compressed = []
    for t, value in events:
        value = bool(value)
        if not compressed or compressed[-1][1] != value:
            compressed.append([float(t), value])
    return compressed


def engagement_windows(events, end_time=None):
    windows = []
    start = None
    for t, value in compress_bool_events(events):
        if value and start is None:
            start = t
        elif not value and start is not None:
            windows.append([start, t])
            start = None
    if start is not None and end_time is not None:
        windows.append([start, end_time])
    return windows


def state_at(events, timestamp):
    value = False
    for t, v in events:
        if t > timestamp:
            break
        value = bool(v)
    return value


def decode_image(msg):
    enc = msg.encoding.lower()
    if enc in ("32fc1", "32fc"):
        a = np.frombuffer(msg.data, dtype=np.float32)
    elif enc in ("16uc1", "mono16"):
        a = np.frombuffer(msg.data, dtype=np.uint16).astype(np.float32)
    elif enc in ("8uc1", "mono8"):
        a = np.frombuffer(msg.data, dtype=np.uint8).astype(np.float32)
    else:
        return None
    expected = int(msg.height) * int(msg.width)
    if a.size < expected:
        return None
    return a[:expected].reshape(int(msg.height), int(msg.width))


def render_depth_sheet(run_name, frames, out_path):
    decoded = [(t, decode_image(msg)) for t, msg in frames]
    decoded = [(t, a) for t, a in decoded if a is not None]
    if not decoded:
        return None
    finite = np.concatenate([a[np.isfinite(a)] for _, a in decoded])
    vmin, vmax = np.percentile(finite, [2, 98]) if finite.size else (0, 1)
    cols = 4
    rows = math.ceil(len(decoded) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(12, 2.5 * rows), squeeze=False)
    t0 = decoded[0][0]
    for ax, (t, a) in zip(axes.flat, decoded):
        ax.imshow(a, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
        zeros = 100.0 * np.mean(a == 0)
        invalid = 100.0 * np.mean(~np.isfinite(a))
        ax.set_title(f"t={t-t0:.1f}s  zero={zeros:.1f}% invalid={invalid:.1f}%", fontsize=8)
        ax.axis("off")
    for ax in axes.flat[len(decoded):]:
        ax.axis("off")
    fig.suptitle(f"{run_name}: /rl/depth_image", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return str(out_path.relative_to(ROOT))


def summarize_window(window):
    force = np.asarray(window["force"], dtype=np.float64)
    return {
        "start": window["start"],
        "end": window["end"],
        "duration": window["end"] - window["start"],
        "samples": len(window["rpy"]),
        "rpy": vector_stats(window["rpy"]),
        "gyro": vector_stats(window["gyro"]),
        "force": vector_stats(window["force"]),
        "dq_norm": percentile(window["dq_norm"]),
        "tau_abs": percentile(window["tau_abs"]),
        "est_velocity": vector_stats(window["est"]),
        "contact20": {
            FEET[i]: float(np.mean(force[:, i] > 20)) for i in range(4)
        } if force.size else {},
    }


def analyze_bag(run_dir, image_count):
    dbs = sorted((run_dir / "bag").glob("*.db3"))
    if not dbs:
        return None
    bag = Bag(dbs[0])
    try:
        topic_counts, topic_bounds = bag.all_stats()

        policy_log = parse_policy_log(run_dir / "rl_policy.log")
        bool_events = []
        low_bounds = topic_bounds.get("/lowstate")

        force_all = []
        force_engaged = []
        force_idle = []
        force_est_all = []
        rpy_all = []
        rpy_engaged = []
        gyro_engaged = []
        accel_engaged = []
        dq_norm_engaged = []
        tau_abs_engaged = []
        temp_engaged = []
        samples_engaged = 0
        latent_norm = []
        latent_times = []
        est_rows = []
        est_engaged = []
        action_norm = []
        action_delta = []
        previous = None
        frames = []
        image_total = topic_counts.get("/rl/depth_image", 0)
        image_targets = set(
            np.linspace(0, max(image_total - 1, 0), min(image_count, image_total)).astype(int)
        ) if image_total else set()
        indexes = {}
        selected_topics = {
            "/policy_engaged", "/lowstate", "/rl/depth_latent",
            "/rl/est_lin_vel", "/rl/last_action", "/rl/depth_image",
        }
        window_details = []
        current_window = None
        for topic, t, msg in bag.iter_topics(
            selected_topics, strides={"/lowstate": 10}
        ) or []:
            index = indexes.get(topic, 0)
            indexes[topic] = index + 1
            if topic == "/policy_engaged":
                value = bool(msg.data)
                if not bool_events or bool_events[-1][1] != value:
                    bool_events.append([t, value])
                    if value:
                        current_window = {
                            "start": t, "end": t, "rpy": [], "gyro": [],
                            "force": [], "dq_norm": [], "tau_abs": [], "est": [],
                        }
                    elif current_window is not None:
                        current_window["end"] = t
                        window_details.append(summarize_window(current_window))
                        current_window = None
            elif topic == "/lowstate":
                # /lowstate is ~500 Hz; retain policy-rate (50 Hz) samples.
                engaged = state_at(bool_events, t)
                ff = [float(x) for x in msg.foot_force]
                ffe = [float(x) for x in msg.foot_force_est]
                rpy = [float(x) for x in msg.imu_state.rpy]
                force_all.append(ff)
                force_est_all.append(ffe)
                rpy_all.append(rpy)
                if engaged:
                    samples_engaged += 1
                    force_engaged.append(ff)
                    rpy_engaged.append(rpy)
                    gyro_engaged.append([float(x) for x in msg.imu_state.gyroscope])
                    accel_engaged.append([float(x) for x in msg.imu_state.accelerometer])
                    motors = msg.motor_state[:12]
                    dq_value = math.sqrt(sum(float(m.dq) ** 2 for m in motors))
                    tau_values = [abs(float(m.tau_est)) for m in motors]
                    dq_norm_engaged.append(dq_value)
                    tau_abs_engaged.extend(tau_values)
                    temp_engaged.extend(float(m.temperature) for m in motors)
                    if current_window is not None:
                        current_window["end"] = t
                        current_window["rpy"].append(rpy)
                        current_window["gyro"].append(
                            [float(x) for x in msg.imu_state.gyroscope]
                        )
                        current_window["force"].append(ff)
                        current_window["dq_norm"].append(dq_value)
                        current_window["tau_abs"].extend(tau_values)
                else:
                    force_idle.append(ff)
            elif topic == "/rl/depth_latent":
                latent_times.append(t)
                latent_norm.append(float(np.linalg.norm(np.asarray(msg.data, dtype=np.float64))))
            elif topic == "/rl/est_lin_vel":
                row = [float(x) for x in msg.data[:3]]
                est_rows.append(row)
                if state_at(bool_events, t):
                    est_engaged.append(row)
                    if current_window is not None:
                        current_window["est"].append(row)
            elif topic == "/rl/last_action":
                a = np.asarray(msg.data, dtype=np.float64)
                action_norm.append(float(np.linalg.norm(a)))
                if previous is not None and previous.shape == a.shape:
                    action_delta.append(float(np.linalg.norm(a - previous)))
                previous = a
            elif topic == "/rl/depth_image" and index in image_targets:
                frames.append((t, msg))

        if current_window is not None:
            current_window["end"] = low_bounds[1] if low_bounds else current_window["end"]
            window_details.append(summarize_window(current_window))
        if not bool_events:
            bool_events = compress_bool_events(policy_log["engagement_events"])
        windows = engagement_windows(bool_events, low_bounds[1] if low_bounds else None)

        force_arr = np.asarray(force_engaged, dtype=np.float64)
        contact_rates = {}
        if force_arr.size:
            for threshold in (17, 20, 30, 50):
                contact_rates[str(threshold)] = {
                    FEET[i]: float(np.mean(force_arr[:, i] > threshold))
                    for i in range(4)
                }
                contact_rates[str(threshold)]["all_airborne"] = float(
                    np.mean(np.all(force_arr <= threshold, axis=1))
                )
                contact_rates[str(threshold)]["all_contact"] = float(
                    np.mean(np.all(force_arr > threshold, axis=1))
                )

        image_stats = []
        for t, msg in frames:
            a = decode_image(msg)
            if a is None:
                continue
            finite = a[np.isfinite(a)]
            image_stats.append({
                "time": t,
                "encoding": msg.encoding,
                "shape": [int(msg.height), int(msg.width)],
                "zero_fraction": float(np.mean(a == 0)),
                "invalid_fraction": float(np.mean(~np.isfinite(a))),
                "min": float(finite.min()) if finite.size else None,
                "median": float(np.median(finite)) if finite.size else None,
                "max": float(finite.max()) if finite.size else None,
            })
        sheet_path = OUT_ROOT / f"{run_dir.name}_depth_sheet.png"
        sheet = render_depth_sheet(run_dir.name, frames, sheet_path)

        log_est = policy_log["estimates"]
        log_cmd = [r[1] for r in log_est if r[1] is not None]
        log_vel = [[r[2], r[3], r[4]] for r in log_est]
        log_dlat = [r[5] for r in log_est]

        return {
            "run": run_dir.name,
            "meta": parse_meta(run_dir / "meta.txt"),
            "bag_size_bytes": dbs[0].stat().st_size,
            "topic_counts": topic_counts,
            "topic_bounds": topic_bounds,
            "engagement_events": bool_events,
            "engagement_windows": [
                {"start": s, "end": e, "duration": e - s} for s, e in windows
            ],
            "engagement_window_details": window_details,
            "engaged_lowstate_samples": samples_engaged,
            "foot_force": {
                "all": vector_stats(force_all),
                "idle": vector_stats(force_idle),
                "engaged": vector_stats(force_engaged),
                "estimated_all": vector_stats(force_est_all),
                "contact_rates": contact_rates,
            },
            "imu": {
                "rpy_all": vector_stats(rpy_all),
                "rpy_engaged": vector_stats(rpy_engaged),
                "gyro_engaged": vector_stats(gyro_engaged),
                "accel_engaged": vector_stats(accel_engaged),
            },
            "motors_engaged": {
                "dq_norm": percentile(dq_norm_engaged),
                "tau_abs": percentile(tau_abs_engaged),
                "temperature": percentile(temp_engaged),
            },
            "policy_bag": {
                "est_velocity_all": vector_stats(est_rows),
                "est_velocity_engaged": vector_stats(est_engaged),
                "latent_norm": percentile(latent_norm),
                "action_norm": percentile(action_norm),
                "action_delta_norm": percentile(action_delta),
            },
            "policy_log": {
                "estimate_count": len(log_est),
                "command_vx": percentile(log_cmd),
                "est_velocity": vector_stats(log_vel),
                "latent_norm": percentile(log_dlat),
            },
            "depth_images": {
                "sampled": image_stats,
                "contact_sheet": sheet,
            },
        }
    finally:
        bag.close()


def analyze_csv_only(run_dir):
    path = run_dir / "lowstate.csv"
    if not path.exists():
        return {"run": run_dir.name, "meta": parse_meta(run_dir / "meta.txt")}
    rows = 0
    engaged = 0
    times = []
    dq_norm = []
    with path.open(errors="replace") as f:
        next(f, None)
        for line in f:
            cols = line.rstrip().split(",")
            if len(cols) < 27:
                continue
            try:
                t = float(cols[0])
                dq = np.asarray(cols[13:25], dtype=np.float64)
                is_engaged = int(cols[25])
            except ValueError:
                continue
            rows += 1
            engaged += is_engaged
            times.append(t)
            if is_engaged:
                dq_norm.append(float(np.linalg.norm(dq)))
    return {
        "run": run_dir.name,
        "meta": parse_meta(run_dir / "meta.txt"),
        "csv_only": True,
        "csv_rows": rows,
        "duration": (times[-1] - times[0]) if len(times) > 1 else 0,
        "engaged_samples": engaged,
        "engaged_dq_norm": percentile(dq_norm),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-root", type=Path, default=REAL_ROOT)
    parser.add_argument("--out", type=Path, default=OUT_ROOT / "summary.json")
    parser.add_argument("--image-count", type=int, default=12)
    parser.add_argument("--run", action="append", default=[])
    args = parser.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    selected = set(args.run)
    summaries = []
    for run_dir in sorted(p for p in args.real_root.iterdir() if p.is_dir()):
        if selected and run_dir.name not in selected:
            continue
        print(f"analyzing {run_dir.name}", flush=True)
        if (run_dir / "bag").exists():
            summaries.append(analyze_bag(run_dir, args.image_count))
        else:
            summaries.append(analyze_csv_only(run_dir))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summaries, indent=2, allow_nan=False))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
