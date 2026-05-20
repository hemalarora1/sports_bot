#!/usr/bin/env python3
"""Send a world-frame base goal and report convergence.

Used to test base_bridge.py end-to-end: this writes
sports_bot::cmd::base::goal_pose (world frame), then polls the cart's
OptiTrack world pose and reports how close it is to the goal.

Pre-reqs: redis-server, StreamDataSkeleton.py, TidyBot driver, AND
base_bridge.py are all already running.

Usage examples:

    # Read current world pose, exit (sanity check before sending anything)
    python sports_bot/scripts/send_base_goal.py --robot-rigid-body-id 11 --read

    # Absolute goal: drive to (0.5, 0, 0) in world
    python sports_bot/scripts/send_base_goal.py --robot-rigid-body-id 11 \\
        --x 0.5 --y 0.0 --yaw-deg 0

    # Relative goal: 10 cm forward in world X, no rotation change
    python sports_bot/scripts/send_base_goal.py --robot-rigid-body-id 11 \\
        --relative --x 0.10 --y 0.0 --yaw-deg 0
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import redis

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SPORTS_BOT_DIR = os.path.dirname(_THIS_DIR)
_OPENSAI_DIR = os.path.dirname(_SPORTS_BOT_DIR)
if _OPENSAI_DIR not in sys.path:
    sys.path.insert(0, _OPENSAI_DIR)

from sports_bot.utils.frames import (  # noqa: E402
    load_robot_marker_calibration,
    robot_marker_calibration_path,
    se2_compose,
    wrap_angle,
    yaw_from_quat,
)


OPTI_POS_PREFIX = "sai2::optitrack::rigid_body_pos::"
OPTI_ORI_PREFIX = "sai2::optitrack::rigid_body_ori::"
FSM_BASE_GOAL   = "sports_bot::cmd::base::goal_pose"
HB1_CURRENT     = "hb1::current_pose"
HB1_DESIRED     = "hb1::desired_pose"


def read_world_pose_B(r: redis.Redis, rb_id: int):
    """Read the cart's MARKER pose T_W_B from OptiTrack."""
    raw_pos = r.get(OPTI_POS_PREFIX + str(rb_id))
    raw_ori = r.get(OPTI_ORI_PREFIX + str(rb_id))
    if raw_pos is None or raw_ori is None:
        return None
    pos = json.loads(raw_pos)
    ori = json.loads(raw_ori)
    yaw = yaw_from_quat(float(ori[0]), float(ori[1]), float(ori[2]), float(ori[3]))
    return float(pos[0]), float(pos[1]), yaw


def read_world_pose_C(r: redis.Redis, rb_id: int, T_B_C):
    """Read the cart's CONTROL-POINT pose T_W_C = T_W_B ⊕ T_B_C in world.

    This is the pose the bridge actually commands against — the cart's
    odometry pivot expressed in world. Use this for relative offsets and
    convergence checks so they're consistent with what the driver is doing.
    """
    T_W_B = read_world_pose_B(r, rb_id)
    if T_W_B is None:
        return None
    return se2_compose(T_W_B, T_B_C)


def read_odom(r: redis.Redis):
    raw = r.get(HB1_CURRENT)
    if raw is None:
        return None
    p = json.loads(raw)
    return float(p[0]), float(p[1]), float(p[2])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--robot-rigid-body-id", type=int, required=True)
    parser.add_argument("--x", type=float, default=0.0, help="World X (m). Default 0.")
    parser.add_argument("--y", type=float, default=0.0, help="World Y (m). Default 0.")
    parser.add_argument("--yaw-deg", type=float, default=0.0,
                        help="World heading (degrees). Default 0.")
    parser.add_argument("--relative", action="store_true",
                        help="Treat (x, y, yaw-deg) as a delta from current world pose.")
    parser.add_argument("--read", action="store_true",
                        help="Print current world+odom pose and exit; don't send a goal.")
    parser.add_argument("--watch", type=float, default=10.0,
                        help="Seconds to poll for convergence (default 10).")
    parser.add_argument("--pos-tol-mm", type=float, default=30.0,
                        help="Position tolerance to declare arrival (default 30 mm).")
    parser.add_argument("--yaw-tol-deg", type=float, default=2.0,
                        help="Yaw tolerance to declare arrival (default 2°).")
    parser.add_argument("--robot-marker-calibration", type=str, default=None,
                        help="Path to T_B_C calibration JSON. Default: "
                             "sports_bot/optitrack/robot_marker_calibration.json")
    parser.add_argument("--redis-host", default="localhost")
    parser.add_argument("--redis-port", type=int, default=6379)
    args = parser.parse_args()

    r = redis.Redis(host=args.redis_host, port=args.redis_port, decode_responses=True)
    try:
        r.ping()
    except redis.exceptions.ConnectionError as e:
        print(f"Cannot reach Redis: {e}")
        sys.exit(1)

    # Goals live in the cart's control-point frame C in world (= what
    # base_bridge.py commands), NOT the marker frame B. So we use
    # T_W_C = T_W_B ⊕ T_B_C everywhere — same convention as the bridge's
    # sanity print.
    cal_path = args.robot_marker_calibration or robot_marker_calibration_path()
    if not os.path.isfile(cal_path):
        print(f"Missing T_B_C calibration at {cal_path}. "
              f"Run calibrate_robot_marker.py first.")
        sys.exit(1)
    T_B_C = load_robot_marker_calibration(cal_path)

    current_B = read_world_pose_B(r, args.robot_rigid_body_id)
    if current_B is None:
        print(f"No OptiTrack data for rigid body {args.robot_rigid_body_id}. "
              "Is the streamer running?")
        sys.exit(1)
    current = se2_compose(current_B, T_B_C)
    odom = read_odom(r)

    print(f"Current cart (C in W):  x={current[0]:+.3f}  y={current[1]:+.3f}  "
          f"θ={math.degrees(current[2]):+.1f}°")
    print(f"  (marker B in W:       x={current_B[0]:+.3f}  y={current_B[1]:+.3f}  "
          f"θ={math.degrees(current_B[2]):+.1f}°)")
    if odom is not None:
        print(f"Current odom (R):       x={odom[0]:+.3f}  y={odom[1]:+.3f}  "
              f"θ={math.degrees(odom[2]):+.1f}°")
    else:
        print(f"Current odom (R):       (no hb1::current_pose — is the driver up?)")

    if args.read:
        return

    if args.relative:
        gx = current[0] + args.x
        gy = current[1] + args.y
        gth = wrap_angle(current[2] + math.radians(args.yaw_deg))
        print(f"Goal (relative to C):   dx={args.x:+.3f}  dy={args.y:+.3f}  "
              f"dθ={args.yaw_deg:+.1f}°")
    else:
        gx, gy = args.x, args.y
        gth = math.radians(args.yaw_deg)

    print(f"Goal (C in W):          x={gx:+.3f}  y={gy:+.3f}  θ={math.degrees(gth):+.1f}°")

    goal_W = [gx, gy, gth]
    print(f"Writing {FSM_BASE_GOAL} = {goal_W}")
    r.set(FSM_BASE_GOAL, json.dumps(goal_W))

    print(f"\nPolling for up to {args.watch:.1f} s (pos tol {args.pos_tol_mm:.0f} mm, "
          f"yaw tol {args.yaw_tol_deg:.1f}°)...\n")
    t0 = time.perf_counter()
    last_print = 0.0
    converged = False
    while time.perf_counter() - t0 < args.watch:
        now = read_world_pose_C(r, args.robot_rigid_body_id, T_B_C)
        if now is not None:
            err_m = math.hypot(now[0] - gx, now[1] - gy)
            err_yaw = math.degrees(wrap_angle(now[2] - gth))
            t_elapsed = time.perf_counter() - t0
            if (time.perf_counter() - last_print) > 0.5:
                print(f"  t={t_elapsed:4.1f}s   "
                      f"pos err={err_m * 1000:6.0f} mm   "
                      f"yaw err={err_yaw:+6.2f}°   "
                      f"at  x={now[0]:+.3f}  y={now[1]:+.3f}  θ={math.degrees(now[2]):+6.1f}°")
                last_print = time.perf_counter()
            if err_m < args.pos_tol_mm / 1000.0 and abs(err_yaw) < args.yaw_tol_deg:
                t_elapsed = time.perf_counter() - t0
                print(f"\n  ✓ converged at t={t_elapsed:.2f}s: "
                      f"pos err {err_m * 1000:.0f} mm, yaw err {err_yaw:+.2f}°")
                converged = True
                break
        time.sleep(0.05)

    if not converged:
        print(f"\n  timeout — cart didn't reach tolerance within {args.watch:.1f} s.")
        print(f"  Either tolerance is too tight, the driver is slow, or something's wrong.")


if __name__ == "__main__":
    main()
