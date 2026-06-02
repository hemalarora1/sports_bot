#!/usr/bin/env python3
"""Interactive calibration of T_B_C — the static offset between the OptiTrack
rigid-body local frame on the cart and the TidyBot odometry control-point frame.

Why this exists
---------------
Three calibration unknowns exist when you want world-frame commands to actually
land at world-frame goals:

    T_W_M    Motive room → world.       Solved in world_calibration.json.
    T_B_C    rigid-body → odometry.     Solved here, once per marker mounting.
    T_W_R    world → robot odometry.    Solved per-session in base_bridge.py
                                        from a stationary snapshot, given T_B_C.

Once T_B_C is known, the per-session bringup is a 2-second snapshot — no
calibration drive needed at the start of every workshop session.

What it does
------------
1. Connects to Redis. Verifies OptiTrack streamer and TidyBot driver are alive.
2. You manually drive the cart to N waypoints (default 5) covering a mix of
   rotations and translations across the workspace. At each waypoint, hold
   still and press Enter to capture.
3. Each capture averages ~1 s of OptiTrack and hb1::current_pose readings
   into a single pose pair (T_W_B, T_R_C).
4. After capture, solves the 2D hand-eye problem AX=XB jointly for T_B_C and
   T_W_R via Levenberg-Marquardt.
5. Reports per-waypoint and aggregate residuals. Saves T_B_C to
   sports_bot/optitrack/robot_marker_calibration.json. T_W_R is discarded
   (per-session, useless tomorrow).

Usage
-----
    conda activate opensai
    python sports_bot/scripts/calibrate_robot_marker.py \\
        --robot-rigid-body-id <ID>

Prerequisites (run in order):
    1.  redis-server
    2.  python sports_bot/optitrack/StreamDataSkeleton.py ...
    3.  TidyBot redis_driver.py (any pose — odometry just has to be live)
    4.  This script.

Tips for good waypoint coverage:
    - Spread waypoints across the workspace (≥0.5 m apart).
    - Mix rotations AND translations between successive waypoints — pure
      translation underdetermines T_B_C's rotation, pure rotation underdetermines
      its translation.
    - Five waypoints is the default; more is fine, fewer than three is
      brittle.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import List, Optional, Tuple

import numpy as np
import redis

# Make `sports_bot.*` importable when running this script directly.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SPORTS_BOT_DIR = os.path.dirname(_THIS_DIR)
_OPENSAI_DIR = os.path.dirname(_SPORTS_BOT_DIR)
if _OPENSAI_DIR not in sys.path:
    sys.path.insert(0, _OPENSAI_DIR)

from sports_bot.utils.frames import (  # noqa: E402
    SE2,
    average_angles,
    robot_marker_calibration_path,
    save_robot_marker_calibration,
    solve_hand_eye_se2,
    yaw_from_quat,
)


# ---------- Redis keys --------------------------------------------------------

OPTI_POS_PREFIX = "sai2::optitrack::rigid_body_pos::"   # [x,y,z]      world frame
OPTI_ORI_PREFIX = "sai2::optitrack::rigid_body_ori::"   # [qx,qy,qz,qw] world frame
HB1_CURRENT_POSE = "hb1::current_pose"


# ---------- Sampling primitives ----------------------------------------------

def _read_optitrack_pose_W(
    r: redis.Redis, rb_id: int
) -> Optional[Tuple[SE2, float]]:
    """Returns ((x_W, y_W, yaw_W), z_W) from the world-frame OptiTrack keys.
    Returns None if either key is missing or malformed."""
    raw_pos = r.get(OPTI_POS_PREFIX + str(rb_id))
    raw_ori = r.get(OPTI_ORI_PREFIX + str(rb_id))
    if raw_pos is None or raw_ori is None:
        return None
    try:
        pos = json.loads(raw_pos)
        ori = json.loads(raw_ori)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    if len(pos) != 3 or len(ori) != 4:
        return None
    yaw_W = yaw_from_quat(float(ori[0]), float(ori[1]), float(ori[2]), float(ori[3]))
    return (float(pos[0]), float(pos[1]), yaw_W), float(pos[2])


def _read_hb1_pose(r: redis.Redis) -> Optional[SE2]:
    raw = r.get(HB1_CURRENT_POSE)
    if raw is None:
        return None
    try:
        p = json.loads(raw)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    if len(p) < 3:
        return None
    return (float(p[0]), float(p[1]), float(p[2]))


def sample_pose_pair(
    r: redis.Redis,
    rb_id: int,
    duration_s: float = 1.0,
    rate_hz: float = 60.0,
) -> Tuple[SE2, SE2, dict]:
    """Average OptiTrack + hb1::current_pose over `duration_s`.

    Returns (T_W_B, T_R_C, stats) where stats has jitter / sample-count info
    so the caller can warn if the cart moved during capture.
    """
    dt = 1.0 / rate_hz
    xs_W: List[float] = []
    ys_W: List[float] = []
    zs_W: List[float] = []
    yaws_W: List[float] = []
    xs_R: List[float] = []
    ys_R: List[float] = []
    yaws_R: List[float] = []

    t_end = time.perf_counter() + duration_s
    last_warn = 0.0
    while time.perf_counter() < t_end:
        opti = _read_optitrack_pose_W(r, rb_id)
        hb1 = _read_hb1_pose(r)
        if opti is not None and hb1 is not None:
            (x_W, y_W, yaw_W), z_W = opti
            xs_W.append(x_W); ys_W.append(y_W); zs_W.append(z_W); yaws_W.append(yaw_W)
            xs_R.append(hb1[0]); ys_R.append(hb1[1]); yaws_R.append(hb1[2])
        else:
            now = time.perf_counter()
            if now - last_warn > 1.0:
                missing = []
                if opti is None:
                    missing.append("OptiTrack")
                if hb1 is None:
                    missing.append("hb1::current_pose")
                print(f"  ...waiting for {' + '.join(missing)}")
                last_warn = now
        time.sleep(dt)

    if not xs_W:
        raise RuntimeError("captured zero samples — is OptiTrack and/or TidyBot driver running?")

    x_W = float(np.mean(xs_W))
    y_W = float(np.mean(ys_W))
    z_W = float(np.mean(zs_W))
    yaw_W = average_angles(yaws_W)
    x_R = float(np.mean(xs_R))
    y_R = float(np.mean(ys_R))
    yaw_R = average_angles(yaws_R)

    # jitter: position is std-dev of (x, y) magnitude, yaw is std-dev around circular mean
    pos_jitter_mm_W = 1000.0 * float(np.sqrt(np.std(xs_W) ** 2 + np.std(ys_W) ** 2))
    pos_jitter_mm_R = 1000.0 * float(np.sqrt(np.std(xs_R) ** 2 + np.std(ys_R) ** 2))
    yaw_jitter_deg_W = math.degrees(float(np.std([math.atan2(math.sin(a - yaw_W), math.cos(a - yaw_W)) for a in yaws_W])))
    yaw_jitter_deg_R = math.degrees(float(np.std([math.atan2(math.sin(a - yaw_R), math.cos(a - yaw_R)) for a in yaws_R])))

    stats = {
        "n_samples": len(xs_W),
        "z_W_mean_m": z_W,
        "pos_jitter_mm_W": pos_jitter_mm_W,
        "pos_jitter_mm_R": pos_jitter_mm_R,
        "yaw_jitter_deg_W": yaw_jitter_deg_W,
        "yaw_jitter_deg_R": yaw_jitter_deg_R,
    }
    return (x_W, y_W, yaw_W), (x_R, y_R, yaw_R), stats


# ---------- Diversity / sanity --------------------------------------------------

def check_diversity(samples: List[Tuple[SE2, SE2]]) -> Optional[str]:
    """Returns a warning string if the captured waypoints don't cover enough
    rotation+translation diversity to constrain the calibration. Returns
    None if they look fine."""
    if len(samples) < 2:
        return "fewer than 2 waypoints — cannot solve"
    xs = np.array([s[1][0] for s in samples])
    ys = np.array([s[1][1] for s in samples])
    yaws = np.array([s[1][2] for s in samples])
    pos_span = float(np.sqrt((xs.max() - xs.min()) ** 2 + (ys.max() - ys.min()) ** 2))
    # yaw diversity: range across the unit-circle representation
    cosines = np.cos(yaws)
    sines = np.sin(yaws)
    yaw_span_deg = math.degrees(2.0 * math.asin(min(1.0, 0.5 * math.sqrt(
        (cosines.max() - cosines.min()) ** 2 + (sines.max() - sines.min()) ** 2
    ))))
    warnings = []
    if pos_span < 0.30:
        warnings.append(f"translation span only {pos_span * 100:.1f} cm (recommend ≥30 cm)")
    if yaw_span_deg < 25.0:
        warnings.append(f"yaw span only {yaw_span_deg:.1f}° (recommend ≥30°)")
    return "; ".join(warnings) if warnings else None


# ---------- Per-waypoint residual reporting -----------------------------------

def report_residuals(
    samples: List[Tuple[SE2, SE2]],
    T_B_C: SE2,
    T_W_R: SE2,
) -> None:
    from sports_bot.utils.frames import se2_compose, wrap_angle
    print(f"\n  per-waypoint residual (LHS − RHS of  T_W_B·T_B_C = T_W_R·T_R_C):")
    print(f"    {'#':>3}  {'dx (mm)':>10}  {'dy (mm)':>10}  {'dθ (deg)':>10}")
    print(f"    {'-' * 3}  {'-' * 10}  {'-' * 10}  {'-' * 10}")
    for i, (T_W_B, T_R_C) in enumerate(samples):
        lhs = se2_compose(T_W_B, T_B_C)
        rhs = se2_compose(T_W_R, T_R_C)
        dx_mm = 1000.0 * (lhs[0] - rhs[0])
        dy_mm = 1000.0 * (lhs[1] - rhs[1])
        dth_deg = math.degrees(wrap_angle(lhs[2] - rhs[2]))
        print(f"    {i:>3}  {dx_mm:>10.2f}  {dy_mm:>10.2f}  {dth_deg:>10.3f}")


# ---------- Main flow ---------------------------------------------------------

def run(args: argparse.Namespace) -> int:
    r = redis.Redis(host=args.redis_host, port=args.redis_port, decode_responses=True)
    try:
        r.ping()
    except redis.exceptions.ConnectionError as e:
        print(f"[calibrate] Cannot reach Redis at {args.redis_host}:{args.redis_port}: {e}")
        return 1

    # Liveness check
    if _read_optitrack_pose_W(r, args.robot_rigid_body_id) is None:
        print(f"[calibrate] No OptiTrack data for rigid body {args.robot_rigid_body_id}.")
        print(f"            Start StreamDataSkeleton.py and confirm the asset is visible.")
        return 1
    if _read_hb1_pose(r) is None:
        print(f"[calibrate] No hb1::current_pose in Redis.")
        print(f"            Start the TidyBot redis_driver.py.")
        return 1

    out_path = args.output or robot_marker_calibration_path()

    print("\n" + "=" * 72)
    print("Robot-marker (T_B_C) calibration")
    print("=" * 72)
    print(f"  rigid body ID    : {args.robot_rigid_body_id}")
    print(f"  waypoints        : {args.waypoints}")
    print(f"  capture duration : {args.capture_seconds:.1f} s per waypoint")
    print(f"  output           : {out_path}")
    print()
    print("Drive the cart to each waypoint, hold still, then press Enter to capture.")
    print("Cover a MIX of translations AND rotations between waypoints (≥30 cm /")
    print("≥30° between successive ones). Press Ctrl-C to abort.")
    print()

    samples: List[Tuple[SE2, SE2]] = []
    for i in range(args.waypoints):
        prompt = (
            f"  Waypoint {i + 1}/{args.waypoints}: hold the cart still, "
            f"then press Enter (or 's' to skip) > "
        )
        try:
            ans = input(prompt).strip().lower()
        except EOFError:
            print("\n[calibrate] EOF on input — aborting.")
            return 1
        if ans == "s":
            print(f"    skipped.")
            continue

        try:
            T_W_B, T_R_C, stats = sample_pose_pair(
                r, args.robot_rigid_body_id,
                duration_s=args.capture_seconds,
            )
        except RuntimeError as e:
            print(f"    capture failed: {e}")
            continue

        print(f"    T_W_B  = ({T_W_B[0]:+.4f}, {T_W_B[1]:+.4f}, "
              f"{math.degrees(T_W_B[2]):+.2f}°)")
        print(f"    T_R_C  = ({T_R_C[0]:+.4f}, {T_R_C[1]:+.4f}, "
              f"{math.degrees(T_R_C[2]):+.2f}°)")
        print(f"    z_W    = {stats['z_W_mean_m']:+.4f} m  (sanity: marker height above floor)")
        print(f"    jitter (pos / yaw): "
              f"W {stats['pos_jitter_mm_W']:.2f} mm / {stats['yaw_jitter_deg_W']:.3f}°   "
              f"R {stats['pos_jitter_mm_R']:.2f} mm / {stats['yaw_jitter_deg_R']:.3f}°   "
              f"({stats['n_samples']} samples)")
        if stats["pos_jitter_mm_W"] > 5.0 or stats["yaw_jitter_deg_W"] > 0.3:
            print(f"    WARNING: high jitter — cart may have moved. Consider redoing this waypoint.")
        samples.append((T_W_B, T_R_C))

    if len(samples) < 2:
        print(f"\n[calibrate] Need ≥2 captured waypoints, got {len(samples)}. Aborting.")
        return 1

    diversity_warning = check_diversity(samples)
    if diversity_warning:
        print(f"\n[calibrate] WARNING: poor waypoint diversity — {diversity_warning}.")
        print(f"            Solution may be ill-conditioned; consider re-running with")
        print(f"            more separated waypoints.")

    print(f"\n[calibrate] Solving 2D hand-eye over {len(samples)} waypoints...")
    T_B_C, T_W_R, pos_rms, yaw_rms_deg = solve_hand_eye_se2(samples)

    print()
    print(f"  T_B_C (PERSIST)   = ({T_B_C[0]:+.4f} m, {T_B_C[1]:+.4f} m, "
          f"{math.degrees(T_B_C[2]):+.2f}°)")
    print(f"  T_W_R (this session, discarded) = "
          f"({T_W_R[0]:+.4f} m, {T_W_R[1]:+.4f} m, {math.degrees(T_W_R[2]):+.2f}°)")
    print()
    print(f"  residual RMS  : {pos_rms * 1000:.2f} mm / {yaw_rms_deg:.3f}°")

    if pos_rms > 0.02 or yaw_rms_deg > 1.0:
        print(f"  WARNING: residuals are high. Likely causes:")
        print(f"    - cart drifted during a capture (high jitter warning earlier)")
        print(f"    - markers loose / shifting on the cart")
        print(f"    - wheel slip during the drive between waypoints")
        print(f"    - too little rotational diversity in the captured set")

    report_residuals(samples, T_B_C, T_W_R)

    metadata = {
        "calibrated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "rigid_body_id": args.robot_rigid_body_id,
        "n_waypoints": len(samples),
        "residual_pos_rms_mm": round(pos_rms * 1000.0, 3),
        "residual_yaw_rms_deg": round(yaw_rms_deg, 4),
        "world_calibration": "world_calibration.json",
        "samples": [
            {
                "T_W_B": list(T_W_B),
                "T_R_C": list(T_R_C),
            }
            for T_W_B, T_R_C in samples
        ],
    }

    if args.dry_run:
        print(f"\n[calibrate] --dry-run set; NOT writing {out_path}.")
        return 0

    if os.path.isfile(out_path) and not args.force:
        print(f"\n[calibrate] {out_path} already exists.")
        try:
            ans = input("            overwrite? [y/N] ").strip().lower()
        except EOFError:
            ans = ""
        if ans != "y":
            print("            not written.")
            return 0

    save_robot_marker_calibration(out_path, T_B_C, metadata)
    print(f"\n[calibrate] Wrote {out_path}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate T_B_C — the static marker-to-odometry offset.",
    )
    parser.add_argument(
        "--robot-rigid-body-id", type=int, required=True, metavar="ID",
        help="Motive Streaming ID of the TidyBot rigid body.",
    )
    parser.add_argument(
        "--waypoints", type=int, default=5,
        help="Number of waypoints to capture (default: 5; minimum 2; recommend 5+).",
    )
    parser.add_argument(
        "--capture-seconds", type=float, default=1.0,
        help="How long to average each waypoint (default: 1.0 s).",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Where to write the calibration JSON. Default: "
             "sports_bot/optitrack/robot_marker_calibration.json",
    )
    parser.add_argument("--redis-host", default="localhost")
    parser.add_argument("--redis-port", type=int, default=6379)
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite an existing calibration file without prompting.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Solve and report, but do not write the calibration file.",
    )
    args = parser.parse_args()

    try:
        rc = run(args)
    except KeyboardInterrupt:
        print("\n[calibrate] interrupted.")
        rc = 130
    sys.exit(rc)


if __name__ == "__main__":
    main()
