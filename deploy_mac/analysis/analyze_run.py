#!/usr/bin/env python3
"""Analyse and plot a real/sim run captured under run_logs/.

Reads:
  lowstate.csv         joint q/dq (12 each), engaged flag, cmd_kp  (relative time t)
  rl_policy.log        per-second [est] vx/vy/vz, yaw_pred, |dlat|  (ROS epoch)
  low_level_ctrl.log   FSM state_obs lay_diff/stand_diff/dq, laydown/standing
Outputs:
  analysis.png         multi-panel timeline
  prints a text summary

Usage: python deploy_mac/analysis/analyze_run.py <run_dir>
"""
import sys, os, re, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

JOINTS = ["FR_hip","FR_thigh","FR_calf","FL_hip","FL_thigh","FL_calf",
          "RR_hip","RR_thigh","RR_calf","RL_hip","RL_thigh","RL_calf"]
EPOCH_RE = re.compile(r"\[(\d+\.\d+)\]")


def load_csv(path):
    t, q, dq, eng, kp = [], [], [], [], []
    with open(path) as f:
        r = csv.reader(f); next(r)
        for row in r:
            if len(row) < 27:
                continue
            t.append(float(row[0]))
            q.append([float(x) for x in row[1:13]])
            dq.append([float(x) for x in row[13:25]])
            eng.append(int(row[25]))
            kp.append(float(row[26]) if row[26] != "nan" else np.nan)
    return (np.array(t), np.array(q), np.array(dq),
            np.array(eng), np.array(kp))


def parse_rl(path):
    """returns dict of arrays keyed by epoch for [est] lines + engage events."""
    est = []  # (epoch, vx, vy, vz, yaw1, yaw2, dlat)
    events = []  # (epoch, engaged_bool)
    pat = re.compile(
        r"vx=([+-][\d.]+) vy=([+-][\d.]+) vz=([+-][\d.]+)\s+"
        r"yaw_pred=\(\s*([+-][\d.]+).?,\s*([+-][\d.]+).?\)\s+\|dlat\|=([\d.]+)")
    with open(path) as f:
        for line in f:
            m = EPOCH_RE.search(line)
            if not m:
                continue
            ep = float(m.group(1))
            if "policy_engaged=True" in line:
                events.append((ep, 1))
            elif "policy_engaged=False" in line:
                events.append((ep, 0))
            me = pat.search(line)
            if me:
                est.append((ep,) + tuple(float(g) for g in me.groups()))
    return np.array(est), events


def parse_fsm(path):
    rows = []  # (epoch, lay_diff, stand_diff, dq, laydown, standing)
    pat = re.compile(
        r"lay_diff=([\d.]+) stand_diff=([\d.]+) dq=([\d.]+)\s+"
        r"uncontrolled=(\d) laydown=(\d) standing=(\d)")
    with open(path) as f:
        for line in f:
            me = pat.search(line)
            if not me:
                continue
            m = EPOCH_RE.search(line)
            if not m:
                continue
            rows.append((float(m.group(1)),
                         float(me.group(1)), float(me.group(2)),
                         float(me.group(3)), int(me.group(5)), int(me.group(6))))
    return np.array(rows)


def engaged_windows(t, eng):
    wins = []
    d = np.diff(eng.astype(int))
    starts = t[1:][d == 1]
    ends = t[1:][d == -1]
    if eng[0] == 1:
        starts = np.insert(starts, 0, t[0])
    if eng[-1] == 1:
        ends = np.append(ends, t[-1])
    for s, e in zip(starts, ends):
        wins.append((s, e))
    return wins


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else "."
    t, q, dq, eng, kp = load_csv(os.path.join(run, "lowstate.csv"))
    est, rl_events = parse_rl(os.path.join(run, "rl_policy.log"))
    fsm = parse_fsm(os.path.join(run, "low_level_ctrl.log"))

    wins = engaged_windows(t, eng)

    # --- epoch -> relative-t alignment: anchor on first engage-on event ---
    first_eng_t = wins[0][0] if wins else 0.0
    first_eng_ep = next((ep for ep, v in rl_events if v == 1), None)
    offset = (first_eng_ep - first_eng_t) if first_eng_ep else \
             (fsm[0, 0] if len(fsm) else 0.0)

    # ---------------- text summary ----------------
    print(f"\n=== Run: {run} ===")
    print(f"duration: {t[-1]-t[0]:.1f} s  ({len(t)} samples, "
          f"{len(t)/(t[-1]-t[0]):.1f} Hz)")
    print(f"engaged windows (RL policy active): {len(wins)}")
    for i, (s, e) in enumerate(wins):
        print(f"  #{i+1}: t={s:6.1f}-{e:6.1f}s  ({e-s:5.1f}s)")
    if len(fsm):
        ft = fsm[:, 0] - offset
        stand_on = ft[np.diff(np.insert(fsm[:, 5], 0, 0)) == 1]
        if len(stand_on):
            print(f"first STANDING reached at t={stand_on[0]:.1f}s")
    if len(est):
        et = est[:, 0] - offset
        for i, (s, e) in enumerate(wins):
            m = (et >= s) & (et <= e)
            if m.sum() == 0:
                continue
            vx, vy, vz = est[m, 1], est[m, 2], est[m, 3]
            dl = est[m, 6]
            print(f"  win#{i+1} est: vx={vx.mean():+.2f}±{vx.std():.2f} "
                  f"(max {vx.max():+.2f})  |vy|max={np.abs(vy).max():.2f}  "
                  f"|dlat| {dl.min():.2f}-{dl.max():.2f}")
        # flag blow-up
        bad = np.abs(est[:, 1]) > 1.0
        if bad.any():
            print(f"  !! {bad.sum()} samples |vx|>1.0 (instability), "
                  f"first at t={et[bad][0]:.1f}s vx={est[bad,1][0]:+.2f}")
    print()

    # ---------------- figure ----------------
    fig, ax = plt.subplots(5, 1, figsize=(15, 14), sharex=True)

    def shade(a):
        for s, e in wins:
            a.axvspan(s, e, color="tab:green", alpha=0.12, lw=0)

    # 1: FSM state + cmd_kp
    a = ax[0]
    if len(fsm):
        ft = fsm[:, 0] - offset
        a.plot(ft, fsm[:, 1], label="lay_diff", lw=1)
        a.plot(ft, fsm[:, 2], label="stand_diff", lw=1)
    a.plot(t, kp/50.0, color="tab:red", lw=1, alpha=.7, label="cmd_kp/50")
    a.set_ylabel("FSM diffs")
    a.set_title(f"{os.path.basename(os.path.abspath(run))}  —  "
                f"green = RL policy engaged", fontsize=11)
    shade(a); a.legend(loc="upper right", ncol=4, fontsize=8); a.grid(alpha=.3)

    # 2: joint positions
    a = ax[1]
    for i, name in enumerate(JOINTS):
        a.plot(t, q[:, i], lw=.6, alpha=.8)
    a.set_ylabel("q [rad]\n(12 joints)")
    shade(a); a.grid(alpha=.3)

    # 3: joint velocities (motion intensity)
    a = ax[2]
    a.plot(t, np.linalg.norm(dq, axis=1), color="k", lw=.8,
           label="||dq|| (all joints)")
    a.set_ylabel("joint speed\n[rad/s]")
    shade(a); a.legend(loc="upper right", fontsize=8); a.grid(alpha=.3)

    # 4: estimated base velocity
    a = ax[3]
    if len(est):
        et = est[:, 0] - offset
        a.plot(et, est[:, 1], ".-", ms=4, label="vx")
        a.plot(et, est[:, 2], ".-", ms=4, label="vy")
        a.plot(et, est[:, 3], ".-", ms=4, label="vz")
        a.axhline(0.8, color="grey", ls="--", lw=.7, label="lin_vel_max")
        a.axhline(-0.8, color="grey", ls="--", lw=.7)
    a.set_ylabel("est. base\nvel [m/s]")
    shade(a); a.legend(loc="upper right", ncol=4, fontsize=8); a.grid(alpha=.3)

    # 5: yaw prediction + depth-latent norm
    a = ax[4]
    if len(est):
        a.plot(et, est[:, 4], ".-", ms=4, label="yaw_pred[0] (°)")
        a.plot(et, est[:, 5], ".-", ms=4, label="yaw_pred[1] (°)")
        a2 = a.twinx()
        a2.plot(et, est[:, 6], "x-", color="tab:purple", ms=4,
                alpha=.6, label="|dlat|")
        a2.set_ylabel("|depth latent|")
        a2.legend(loc="lower right", fontsize=8)
    a.set_ylabel("yaw pred [°]")
    a.set_xlabel("time [s]")
    shade(a); a.legend(loc="upper right", ncol=2, fontsize=8); a.grid(alpha=.3)

    fig.tight_layout()
    out = os.path.join(run, "analysis.png")
    fig.savefig(out, dpi=130)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
