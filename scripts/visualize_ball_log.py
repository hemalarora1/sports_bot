#!/usr/bin/env python3
"""Visualize ball tracking logs from logs/*.trace.jsonl"""

import json
import os
import sys
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

LOG_DIR = Path(__file__).parent.parent / "logs"


def load_log(path):
    events = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def parse_events(events):
    t0 = events[0]["t_mono"] if events else 0

    ball_t, ball_x, ball_y, ball_z = [], [], [], []
    fit_t, fit_x, fit_y, fit_z = [], [], [], []
    fit_vx, fit_vy, fit_vz = [], [], []
    intercept_t, int_x, int_y, int_z = [], [], [], []
    commit_starts, commit_dones = [], []

    for ev in events:
        t = ev["t_mono"] - t0

        if ev["event"] == "tracker_tick":
            rb = ev.get("raw_ball")
            if rb:
                ball_t.append(t)
                ball_x.append(rb[0])
                ball_y.append(rb[1])
                ball_z.append(rb[2])

            tr = ev.get("tracker", {})
            fp = tr.get("fit_pos_W")
            fv = tr.get("fit_vel_W")
            if fp and fv:
                fit_t.append(t)
                fit_x.append(fp[0])
                fit_y.append(fp[1])
                fit_z.append(fp[2])
                fit_vx.append(fv[0])
                fit_vy.append(fv[1])
                fit_vz.append(fv[2])

            ic = ev.get("intercept")
            if ic and ev.get("reject_reason") is None:
                intercept_t.append(t)
                p = ic["position_W"]
                int_x.append(p[0])
                int_y.append(p[1])
                int_z.append(p[2])

        elif ev["event"] == "commit_start":
            commit_starts.append(t)
        elif ev["event"] == "commit_done":
            commit_dones.append(t)

    return {
        "ball": (np.array(ball_t), np.array(ball_x), np.array(ball_y), np.array(ball_z)),
        "fit": (np.array(fit_t), np.array(fit_x), np.array(fit_y), np.array(fit_z),
                np.array(fit_vx), np.array(fit_vy), np.array(fit_vz)),
        "intercept": (np.array(intercept_t), np.array(int_x), np.array(int_y), np.array(int_z)),
        "commit_starts": commit_starts,
        "commit_dones": commit_dones,
        "duration": events[-1]["t_mono"] - events[0]["t_mono"] if events else 0,
    }


def plot(data, title):
    bt, bx, by, bz = data["ball"]
    ft, fx, fy, fz, fvx, fvy, fvz = data["fit"]
    it, ix, iy, iz = data["intercept"]
    commits = data["commit_starts"]

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(title, fontsize=12)
    gs = GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    # --- 3D trajectory ---
    ax3d = fig.add_subplot(gs[:2, 0], projection="3d")
    if len(bx):
        sc = ax3d.scatter(bx, by, bz, c=bt, cmap="plasma", s=2, alpha=0.5, label="raw ball")
    if len(fx):
        ax3d.plot(fx, fy, fz, "g-", lw=0.8, alpha=0.7, label="fit pos")
    if len(ix):
        ax3d.scatter(ix, iy, iz, c="red", marker="x", s=40, zorder=5, label="intercept")
    ax3d.set_xlabel("X (m)"); ax3d.set_ylabel("Y (m)"); ax3d.set_zlabel("Z (m)")
    ax3d.set_title("3-D Ball Trajectory")
    ax3d.legend(fontsize=7, loc="upper right")

    # --- Height over time ---
    ax_z = fig.add_subplot(gs[0, 1])
    if len(bt):
        ax_z.plot(bt, bz, "b.", ms=1.5, alpha=0.4, label="raw Z")
    if len(ft):
        ax_z.plot(ft, fz, "g-", lw=0.8, label="fit Z")
    if len(it):
        ax_z.scatter(it, iz, color="red", marker="x", s=30, zorder=5, label="intercept Z")
    for cs in commits:
        ax_z.axvline(cs, color="orange", lw=0.8, alpha=0.8)
    ax_z.set_xlabel("t (s)"); ax_z.set_ylabel("Z height (m)")
    ax_z.set_title("Ball Height Over Time")
    ax_z.legend(fontsize=7)

    # --- XY plane ---
    ax_xy = fig.add_subplot(gs[1, 1])
    if len(bx):
        ax_xy.scatter(bx, by, c=bt, cmap="plasma", s=2, alpha=0.4)
    if len(ix):
        ax_xy.scatter(ix, iy, color="red", marker="x", s=40, zorder=5, label="intercept")
    for cs in commits:
        pass  # hard to show on XY without time axis
    ax_xy.set_xlabel("X (m)"); ax_xy.set_ylabel("Y (m)")
    ax_xy.set_title("Ball XY Path (top-down)")
    ax_xy.set_aspect("equal", adjustable="datalim")
    ax_xy.legend(fontsize=7)

    # --- Speed over time ---
    ax_spd = fig.add_subplot(gs[2, :])
    if len(ft):
        speed = np.sqrt(fvx**2 + fvy**2 + fvz**2)
        ax_spd.plot(ft, speed, "m-", lw=0.8, label="|v| (m/s)")
        ax_spd.plot(ft, fvx, lw=0.5, alpha=0.6, label="vx")
        ax_spd.plot(ft, fvy, lw=0.5, alpha=0.6, label="vy")
        ax_spd.plot(ft, fvz, lw=0.5, alpha=0.6, label="vz")
    for cs in commits:
        ax_spd.axvline(cs, color="orange", lw=1.0, alpha=0.9, label="_commit")
    if commits:
        ax_spd.axvline(commits[0], color="orange", lw=1.0, alpha=0.9, label="commit_start")
    ax_spd.set_xlabel("t (s)"); ax_spd.set_ylabel("m/s")
    ax_spd.set_title("Ball Velocity Over Time")
    ax_spd.legend(fontsize=7, ncol=3)

    plt.show()


def pick_log(log_dir):
    files = sorted(log_dir.glob("*.trace.jsonl"))
    if not files:
        sys.exit(f"No .trace.jsonl files found in {log_dir}")
    print("Available logs:")
    for i, f in enumerate(files):
        size = f.stat().st_size // 1024
        print(f"  [{i}] {f.name}  ({size} KB)")
    idx = input(f"Pick log [0-{len(files)-1}, default=latest]: ").strip()
    if idx == "":
        return files[-1]
    return files[int(idx)]


def main():
    parser = argparse.ArgumentParser(description="Visualize ball tracking log")
    parser.add_argument("log", nargs="?", help="path to .trace.jsonl file (omit to pick interactively)")
    args = parser.parse_args()

    if args.log:
        path = Path(args.log)
    else:
        path = pick_log(LOG_DIR)

    print(f"Loading {path.name} ...")
    events = load_log(path)
    print(f"  {len(events)} events")

    data = parse_events(events)
    bt = data["ball"][0]
    print(f"  {len(bt)} ball samples, duration {data['duration']:.1f}s, "
          f"{len(data['commit_starts'])} commits")

    plot(data, path.name)


if __name__ == "__main__":
    main()
