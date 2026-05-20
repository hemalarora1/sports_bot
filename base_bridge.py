#!/usr/bin/env python3
"""Bridge between the pickleball FSM base commands and the TidyBot wheel driver.

The problem
-----------
- FSM writes base goals in the WORLD frame W
  (sports_bot::cmd::base::goal_pose = [x, y, theta], origin = OptiTrack floor
  marker).
- TidyBot driver expects goals in the ROBOT ODOMETRY frame R
  (hb1::desired_pose = [x, y, theta], origin = wherever the cart was parked
  when redis_driver.py was started).

W and R differ by a per-session rigid transform T_W_R that is only knowable
at runtime, once we can read the robot's pose from OptiTrack.

The geometry
------------
Three frames live on the cart side of the picture:

    B    OptiTrack rigid-body local frame, glued to the cart's markers.
         Live pose in W comes from OptiTrack each frame as T_W_B(t).
    C    TidyBot odometry control-point frame, glued to the cart at
         whatever pivot the firmware tracks. Live pose in R comes from
         hb1::current_pose as T_R_C(t).
    R    Robot odometry origin — floor-fixed for the session, anchored at
         driver startup. Re-anchors every driver restart.

B and C are both rigidly attached to the cart, so the transform between
them — T_B_C — is a static property of how the markers happen to sit on the
cart. It does NOT depend on where the cart is or how it's facing.

R is also floor-fixed for the session, so T_W_R is also static for the
session — but unique per session, because R re-anchors at driver startup.

At every instant the same point on the cart has two world-pose expressions:

    T_W_B(t) ⊕ T_B_C  =  T_W_R ⊕ T_R_C(t)

Once T_B_C is known (calibrated once, persisted across sessions), the
per-session bringup needs only one stationary snapshot:

    T_W_R = T_W_B(t0) ⊕ T_B_C ⊕ T_R_C(t0)^-1                 (eq. 1)

This bridge does (eq. 1) at startup, then transforms each FSM goal:

    goal_R = T_W_R^-1 ⊕ goal_W                                (eq. 2)

Prerequisites (run in order)
----------------------------
1. redis-server
2. python sports_bot/optitrack/StreamDataSkeleton.py ...
3. TidyBot redis_driver.py (any pose — we read its current odometry)
4. THIS script

T_B_C must already be calibrated:
    python sports_bot/scripts/calibrate_robot_marker.py --robot-rigid-body-id <ID>

Usage
-----
    conda activate opensai
    python sports_bot/base_bridge.py --robot-rigid-body-id <ID>
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Optional, Tuple

import numpy as np
import redis

# Importable as `python -m sports_bot.base_bridge` from the OpenSai root,
# or as `python sports_bot/base_bridge.py` (adds the OpenSai root to sys.path).
_THIS_FILE = os.path.abspath(__file__)
_SPORTS_BOT_DIR = os.path.dirname(_THIS_FILE)
_OPENSAI_DIR = os.path.dirname(_SPORTS_BOT_DIR)
if _OPENSAI_DIR not in sys.path:
    sys.path.insert(0, _OPENSAI_DIR)

from sports_bot.utils.frames import (  # noqa: E402
    SE2,
    average_angles,
    load_robot_marker_calibration,
    robot_marker_calibration_path,
    se2_compose,
    se2_inverse,
    wrap_angle,
    yaw_from_quat,
)


# ---------- Redis keys --------------------------------------------------------

OPTI_POS_PREFIX = "sai2::optitrack::rigid_body_pos::"    # [x, y, z]      world frame
OPTI_ORI_PREFIX = "sai2::optitrack::rigid_body_ori::"    # [qx,qy,qz,qw]  world frame

FSM_BASE_GOAL    = "sports_bot::cmd::base::goal_pose"   # [x, y, theta]  world frame
HB1_DESIRED_POSE = "hb1::desired_pose"                  # [x, y, theta]  R frame
HB1_CURRENT_POSE = "hb1::current_pose"                  # [x, y, theta]  R frame


# ---------- Pose readers -------------------------------------------------------

def _read_optitrack_pose_W(
    r: redis.Redis, rb_id: int
) -> Optional[Tuple[SE2, float]]:
    """Returns ((x_W, y_W, yaw_W), z_W) from the world-frame Redis keys
    published by StreamDataSkeleton.py. Returns None if either key is
    missing or malformed."""
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


def _read_hb1_pose(r: redis.Redis, key: str) -> Optional[SE2]:
    raw = r.get(key)
    if raw is None:
        return None
    try:
        p = json.loads(raw)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    if len(p) < 3:
        return None
    return (float(p[0]), float(p[1]), float(p[2]))


# ---------- Snapshot capture ---------------------------------------------------

def snapshot_pose_pair(
    r: redis.Redis,
    rb_id: int,
    duration_s: float,
    rate_hz: float = 60.0,
) -> Tuple[SE2, SE2, dict]:
    """Average T_W_B(t) and T_R_C(t) over `duration_s` while the cart sits still.

    Returns (T_W_B, T_R_C, stats). Raises RuntimeError if no samples captured.
    """
    dt = 1.0 / rate_hz
    xs_W, ys_W, zs_W, yaws_W = [], [], [], []
    xs_R, ys_R, yaws_R = [], [], []

    t_end = time.perf_counter() + duration_s
    last_warn = 0.0
    while time.perf_counter() < t_end:
        opti = _read_optitrack_pose_W(r, rb_id)
        hb1 = _read_hb1_pose(r, HB1_CURRENT_POSE)
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
        raise RuntimeError("captured zero samples during snapshot window")

    T_W_B = (float(np.mean(xs_W)), float(np.mean(ys_W)), average_angles(yaws_W))
    T_R_C = (float(np.mean(xs_R)), float(np.mean(ys_R)), average_angles(yaws_R))

    pos_jitter_mm_W = 1000.0 * float(np.sqrt(np.std(xs_W) ** 2 + np.std(ys_W) ** 2))
    yaw_jitter_deg_W = math.degrees(float(np.std(
        [wrap_angle(a - T_W_B[2]) for a in yaws_W]
    )))
    pos_jitter_mm_R = 1000.0 * float(np.sqrt(np.std(xs_R) ** 2 + np.std(ys_R) ** 2))
    yaw_jitter_deg_R = math.degrees(float(np.std(
        [wrap_angle(a - T_R_C[2]) for a in yaws_R]
    )))

    stats = {
        "n_samples": len(xs_W),
        "z_W_mean_m": float(np.mean(zs_W)),
        "pos_jitter_mm_W": pos_jitter_mm_W,
        "yaw_jitter_deg_W": yaw_jitter_deg_W,
        "pos_jitter_mm_R": pos_jitter_mm_R,
        "yaw_jitter_deg_R": yaw_jitter_deg_R,
    }
    return T_W_B, T_R_C, stats


# ---------- Per-session calibration -------------------------------------------

def derive_T_W_R(T_W_B: SE2, T_B_C: SE2, T_R_C: SE2) -> SE2:
    """Equation (1) above:  T_W_R = T_W_B ⊕ T_B_C ⊕ T_R_C^-1."""
    return se2_compose(se2_compose(T_W_B, T_B_C), se2_inverse(T_R_C))


def world_to_robot(goal_W: SE2, T_R_W: SE2) -> SE2:
    """Equation (2) above:  goal_R = T_R_W ⊕ goal_W."""
    return se2_compose(T_R_W, goal_W)


# ---------- Main loop ----------------------------------------------------------

def run(
    r: redis.Redis,
    rb_id: int,
    T_B_C: SE2,
    T_W_R: SE2,
    rate_hz: float,
    sanity_interval_s: float,
) -> None:
    T_R_W = se2_inverse(T_W_R)

    print(f"[base_bridge] Bridging FSM goals → TidyBot at {rate_hz:.0f} Hz")
    print(f"[base_bridge]   {FSM_BASE_GOAL}  →  {HB1_DESIRED_POSE}")
    print(f"[base_bridge]   T_W_R = ({T_W_R[0]:+.4f}, {T_W_R[1]:+.4f}, "
          f"{math.degrees(T_W_R[2]):+.2f}°)")
    print()

    dt = 1.0 / rate_hz
    prev_goal_raw = None
    last_sanity = 0.0

    while True:
        t0 = time.perf_counter()

        # Goal forwarding
        goal_raw = r.get(FSM_BASE_GOAL)
        if goal_raw is not None and goal_raw != prev_goal_raw:
            try:
                goal_W_list = json.loads(goal_raw)
                if len(goal_W_list) < 3:
                    raise ValueError("goal has <3 components")
                goal_W: SE2 = (float(goal_W_list[0]), float(goal_W_list[1]), float(goal_W_list[2]))
            except (json.JSONDecodeError, TypeError, ValueError) as e:
                print(f"[base_bridge] malformed goal in {FSM_BASE_GOAL}: {e}")
            else:
                goal_R = world_to_robot(goal_W, T_R_W)
                r.set(HB1_DESIRED_POSE, json.dumps(list(goal_R)))
                print(
                    f"[base_bridge] goal  W=[{goal_W[0]:+.3f}, {goal_W[1]:+.3f}, "
                    f"{math.degrees(goal_W[2]):+6.1f}°]  →  "
                    f"R=[{goal_R[0]:+.3f}, {goal_R[1]:+.3f}, "
                    f"{math.degrees(goal_R[2]):+6.1f}°]"
                )
            prev_goal_raw = goal_raw

        # Periodic sanity print: cross-check OptiTrack chain vs odometry chain.
        # Big divergence = wheel slip, marker shift, or odometry drift.
        if sanity_interval_s > 0 and (t0 - last_sanity) >= sanity_interval_s:
            last_sanity = t0
            opti = _read_optitrack_pose_W(r, rb_id)
            hb1 = _read_hb1_pose(r, HB1_CURRENT_POSE)
            if opti is not None and hb1 is not None:
                T_W_B_now, _ = opti
                T_R_C_now = hb1
                lhs = se2_compose(T_W_B_now, T_B_C)              # T_W_C via OT
                rhs = se2_compose(T_W_R, T_R_C_now)              # T_W_C via odom
                dx_mm = 1000.0 * (lhs[0] - rhs[0])
                dy_mm = 1000.0 * (lhs[1] - rhs[1])
                dth_deg = math.degrees(wrap_angle(lhs[2] - rhs[2]))
                print(
                    f"[base_bridge] sanity  W_via_OT=[{lhs[0]:+.3f}, {lhs[1]:+.3f}, "
                    f"{math.degrees(lhs[2]):+6.1f}°]  "
                    f"W_via_odom=[{rhs[0]:+.3f}, {rhs[1]:+.3f}, "
                    f"{math.degrees(rhs[2]):+6.1f}°]  "
                    f"Δ=[{dx_mm:+.1f}, {dy_mm:+.1f}] mm, {dth_deg:+.2f}°"
                )

        elapsed = time.perf_counter() - t0
        sleep = dt - elapsed
        if sleep > 0:
            time.sleep(sleep)


# ---------- Entry point --------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Forward FSM world-frame base goals to the TidyBot driver in odometry frame.",
    )
    parser.add_argument(
        "--robot-rigid-body-id", type=int, required=True, metavar="ID",
        help="Motive Streaming ID of the TidyBot rigid body.",
    )
    parser.add_argument("--redis-host", default="localhost")
    parser.add_argument("--redis-port", type=int, default=6379)
    parser.add_argument(
        "--rate-hz", type=float, default=50.0,
        help="Bridge loop rate in Hz (default: 50).",
    )
    parser.add_argument(
        "--snapshot-seconds", type=float, default=1.0,
        help="How long to average the startup snapshot (default: 1.0 s). "
             "Cart must sit still for this long.",
    )
    parser.add_argument(
        "--sanity-interval-s", type=float, default=2.0,
        help="Print OT-vs-odom cross-check every N seconds (0 = off, default: 2).",
    )
    parser.add_argument(
        "--robot-marker-calibration", type=str, default=None,
        help="Path to T_B_C calibration JSON. Default: "
             "sports_bot/optitrack/robot_marker_calibration.json",
    )
    parser.add_argument(
        "--allow-nonzero-current-pose", action="store_true",
        help="Skip the hb1::current_pose ≈ [0,0,0] check. Use this only if the "
             "driver was started a while ago and the cart has since been moved; "
             "the snapshot still works as long as both readings are simultaneous.",
    )
    args = parser.parse_args()

    r = redis.Redis(host=args.redis_host, port=args.redis_port, decode_responses=True)
    try:
        r.ping()
    except redis.exceptions.ConnectionError as e:
        print(f"[base_bridge] Cannot reach Redis at {args.redis_host}:{args.redis_port}: {e}")
        sys.exit(1)

    cal_path = args.robot_marker_calibration or robot_marker_calibration_path()
    if not os.path.isfile(cal_path):
        print(f"[base_bridge] Missing T_B_C calibration at {cal_path}.")
        print(f"              Run sports_bot/scripts/calibrate_robot_marker.py first.")
        sys.exit(1)
    T_B_C = load_robot_marker_calibration(cal_path)
    print(f"[base_bridge] Loaded T_B_C from {cal_path}:")
    print(f"              ({T_B_C[0]:+.4f} m, {T_B_C[1]:+.4f} m, {math.degrees(T_B_C[2]):+.2f}°)")

    print(f"\n[base_bridge] Snapshotting pose for {args.snapshot_seconds:.1f} s — keep the cart still...")
    try:
        T_W_B, T_R_C, stats = snapshot_pose_pair(
            r, args.robot_rigid_body_id,
            duration_s=args.snapshot_seconds,
        )
    except RuntimeError as e:
        print(f"[base_bridge] snapshot failed: {e}")
        sys.exit(1)

    print(f"  T_W_B = ({T_W_B[0]:+.4f}, {T_W_B[1]:+.4f}, {math.degrees(T_W_B[2]):+.2f}°)  "
          f"(z = {stats['z_W_mean_m']:+.3f} m)")
    print(f"  T_R_C = ({T_R_C[0]:+.4f}, {T_R_C[1]:+.4f}, {math.degrees(T_R_C[2]):+.2f}°)")
    print(f"  jitter (pos / yaw): "
          f"W {stats['pos_jitter_mm_W']:.2f} mm / {stats['yaw_jitter_deg_W']:.3f}°   "
          f"R {stats['pos_jitter_mm_R']:.2f} mm / {stats['yaw_jitter_deg_R']:.3f}°   "
          f"({stats['n_samples']} samples)")
    if stats["pos_jitter_mm_W"] > 5.0 or stats["yaw_jitter_deg_W"] > 0.3:
        print(f"[base_bridge] WARNING: high snapshot jitter — cart may have moved. T_W_R may be off.")

    # Sanity: warn if the driver doesn't appear freshly started.
    if not args.allow_nonzero_current_pose:
        odom_mag = math.sqrt(T_R_C[0] ** 2 + T_R_C[1] ** 2)
        if odom_mag > 0.05 or abs(T_R_C[2]) > math.radians(2.0):
            print(f"[base_bridge] NOTE: hb1::current_pose is "
                  f"({T_R_C[0]:+.3f}, {T_R_C[1]:+.3f}, {math.degrees(T_R_C[2]):+.2f}°) — "
                  f"not at the odometry origin.")
            print(f"              That's fine — T_W_R is computed from the simultaneous "
                  f"(T_W_B, T_R_C) pair — but confirm the driver wasn't restarted between "
                  f"OptiTrack frames. Pass --allow-nonzero-current-pose to silence this.")

    T_W_R = derive_T_W_R(T_W_B, T_B_C, T_R_C)
    print(f"\n[base_bridge] Derived T_W_R = T_W_B ⊕ T_B_C ⊕ T_R_C⁻¹")
    print(f"              ({T_W_R[0]:+.4f} m, {T_W_R[1]:+.4f} m, {math.degrees(T_W_R[2]):+.2f}°)\n")

    try:
        run(
            r, args.robot_rigid_body_id,
            T_B_C, T_W_R,
            rate_hz=args.rate_hz,
            sanity_interval_s=args.sanity_interval_s,
        )
    except KeyboardInterrupt:
        print("\n[base_bridge] stopped.")


if __name__ == "__main__":
    main()
