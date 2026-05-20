#!/usr/bin/env python3
"""Bridge between the pickleball FSM base commands and the TidyBot wheel driver.

The problem:
  - The pickleball FSM writes base goals in the WORLD frame
    (sports_bot::cmd::base::goal_pose = [x, y, theta], world origin = OptiTrack
    floor marker).
  - The TidyBot driver expects goals in the TIDYBOT frame
    (hb1::desired_pose = [x, y, theta], origin = wherever the cart was parked
    when redis_driver.py was started).

These two frames differ by a fixed offset that is only known at runtime, once
we can read the robot's OptiTrack position.

What this script does:
  1. CALIBRATION at startup: reads the TidyBot's OptiTrack rigid body pose to
     compute T_world_robot — the robot's position and heading in the world frame
     at the moment this script starts.  This is the transform between frames.
  2. GOAL BRIDGE in a loop: whenever the FSM writes a new
     sports_bot::cmd::base::goal_pose (world frame), transforms it into the
     TidyBot frame and writes it to hb1::desired_pose.

Prerequisites (run in order):
  1.  redis-server
  2.  python optitrack/StreamDataSkeleton.py   <- publishes the robot rigid body
  3.  TidyBot redis_driver.py                  <- starts with robot at its origin
  4.  THIS script (immediately after step 3, before the cart moves)

Usage:
    conda activate opensai
    python base_bridge.py --robot-rigid-body-id <ID>

    The rigid body ID is the Motive Streaming ID for the TidyBot asset.
    Check Motive > Assets pane > click the TidyBot body > Properties > Streaming ID.
    (PickleBall is 8. The robot is a different number, often 1 or 2.)
"""

import argparse
import json
import math
import sys
import time

import numpy as np
import redis


# ---------- Redis keys ---------------------------------------------------------

OPTI_POS_PREFIX = "sai2::optitrack::rigid_body_pos::"   # [x, y, z]  world frame
OPTI_ORI_PREFIX = "sai2::optitrack::rigid_body_ori::"   # [qx, qy, qz, qw] world

FSM_BASE_GOAL   = "sports_bot::cmd::base::goal_pose"    # [x, y, theta] world frame
HB1_DESIRED     = "hb1::desired_pose"                   # [x, y, theta] TidyBot frame
HB1_CURRENT     = "hb1::current_pose"                   # [x, y, theta] TidyBot frame (odometry)


# ---------- Helpers ------------------------------------------------------------

def quat_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    """Yaw (Z-axis rotation, radians) from a unit quaternion."""
    return math.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )


def read_optitrack_pose(r: redis.Redis, rb_id: int) -> "tuple[float,float,float] | None":
    """Read robot's (x, y, theta) in world frame from OptiTrack, or None."""
    raw_pos = r.get(OPTI_POS_PREFIX + str(rb_id))
    raw_ori = r.get(OPTI_ORI_PREFIX + str(rb_id))
    if raw_pos is None or raw_ori is None:
        return None
    try:
        pos = json.loads(raw_pos)
        ori = json.loads(raw_ori)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    if len(pos) < 2 or len(ori) != 4:
        return None
    theta = quat_to_yaw(*ori)
    return float(pos[0]), float(pos[1]), theta


def world_to_robot(goal_world: list, x0: float, y0: float, th0: float) -> list:
    """Transform a [x, y, theta] goal from world frame into TidyBot frame.

    At startup, the TidyBot was at (x0, y0, th0) in world frame.
    Its internal odometry started at [0, 0, 0].
    So every world-frame point must be rotated and translated accordingly.

    Rigid body 2D transform. Subtract the origin offset, then rotate to align axes
    """
    dx = goal_world[0] - x0 # displacement from robot's startup position to the goal, in the world frame
    dy = goal_world[1] - y0
    x_r =  math.cos(th0) * dx + math.sin(th0) * dy
    y_r = -math.sin(th0) * dx + math.cos(th0) * dy
    th_r = goal_world[2] - th0
    return [x_r, y_r, th_r]


# ---------- Calibration step ---------------------------------------------------

def calibrate(r: redis.Redis, rb_id: int) -> "tuple[float,float,float]":
    """Block until we get a valid OptiTrack reading for the robot.

    Returns (x0, y0, theta0) — the robot's pose in world frame right now.
    This must be captured while the TidyBot driver is freshly started
    (i.e. its odometry is at [0,0,0]).
    """
    print(f"[base_bridge] Waiting for OptiTrack data for rigid body {rb_id}...")
    print(f"[base_bridge] Make sure StreamDataSkeleton.py is running and the robot is visible.")
    while True:
        pose = read_optitrack_pose(r, rb_id)
        if pose is not None:
            x0, y0, th0 = pose
            print(f"[base_bridge] Robot initial pose in world frame:")
            print(f"              x = {x0:.3f} m")
            print(f"              y = {y0:.3f} m")
            print(f"              θ = {math.degrees(th0):.1f}°")
            print(f"[base_bridge] Frame calibration complete.\n")
            return x0, y0, th0
        time.sleep(0.05)


# ---------- Main loop ----------------------------------------------------------

def run(r: redis.Redis, rb_id: int, x0: float, y0: float, th0: float,
        rate_hz: float = 50.0) -> None:
    dt = 1.0 / rate_hz
    prev_goal_raw = None
    last_status_time = 0.0
    status_interval = 1.0  # print pose every second

    print(f"[base_bridge] Bridging FSM goals → TidyBot at {rate_hz:.0f} Hz")
    print(f"[base_bridge] {FSM_BASE_GOAL}  ->  {HB1_DESIRED}\n")

    while True:
        t0 = time.perf_counter()

        goal_raw = r.get(FSM_BASE_GOAL)
        if goal_raw is not None and goal_raw != prev_goal_raw:
            try:
                goal_world = json.loads(goal_raw)
                assert len(goal_world) == 3
            except (json.JSONDecodeError, TypeError, ValueError, AssertionError):
                goal_raw = prev_goal_raw  # ignore malformed
            else:
                goal_robot = world_to_robot(goal_world, x0, y0, th0)
                r.set(HB1_DESIRED, json.dumps(goal_robot))
                print(
                    f"[base_bridge] goal  world=[{goal_world[0]:.3f}, {goal_world[1]:.3f}, "
                    f"{math.degrees(goal_world[2]):.1f}°]  "
                    f"->  robot=[{goal_robot[0]:.3f}, {goal_robot[1]:.3f}, "
                    f"{math.degrees(goal_robot[2]):.1f}°]"
                )
                prev_goal_raw = goal_raw

        # Periodic pose status print
        if t0 - last_status_time >= status_interval:
            last_status_time = t0

            world_pose = read_optitrack_pose(r, rb_id)
            raw_robot = r.get(HB1_CURRENT)

            if world_pose is not None:
                wx, wy, wth = world_pose
                print(f"[base_bridge] world  x={wx:.3f} m  y={wy:.3f} m  θ={math.degrees(wth):.1f}°")
            else:
                print("[base_bridge] world  (no OptiTrack data)")

            if raw_robot is not None:
                try:
                    rp = json.loads(raw_robot)
                    print(f"[base_bridge] robot  x={rp[0]:.3f} m  y={rp[1]:.3f} m  θ={math.degrees(rp[2]):.1f}°")
                except Exception:
                    print("[base_bridge] robot  (malformed hb1::current_pose)")
            else:
                print("[base_bridge] robot  (no hb1::current_pose — is TidyBot driver running?)")

        elapsed = time.perf_counter() - t0
        sleep = dt - elapsed
        if sleep > 0:
            time.sleep(sleep)


# ---------- Entry point --------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transform FSM world-frame base goals into TidyBot frame and forward to hb1::desired_pose."
    )
    parser.add_argument(
        "--robot-rigid-body-id",
        type=int,
        required=True,
        metavar="ID",
        help=(
            "Motive Streaming ID of the TidyBot rigid body. "
            "Check Motive > Assets pane > click the TidyBot body > Properties > Streaming ID."
        ),
    )
    parser.add_argument("--redis-host", default="localhost")
    parser.add_argument("--redis-port", type=int, default=6379)
    parser.add_argument(
        "--rate-hz", type=float, default=50.0,
        help="Bridge loop rate in Hz (default: 50)",
    )
    args = parser.parse_args()

    r = redis.Redis(
        host=args.redis_host,
        port=args.redis_port,
        decode_responses=True,
    )
    try:
        r.ping()
    except redis.exceptions.ConnectionError as e:
        print(f"[base_bridge] Cannot connect to Redis at {args.redis_host}:{args.redis_port}: {e}")
        sys.exit(1)

    # Capture the robot's world-frame pose right now (TidyBot odometry = [0,0,0] at this moment)
    x0, y0, th0 = calibrate(r, args.robot_rigid_body_id)

    try:
        run(r, args.robot_rigid_body_id, x0, y0, th0, rate_hz=args.rate_hz)
    except KeyboardInterrupt:
        print("\n[base_bridge] stopped.")


if __name__ == "__main__":
    main()
