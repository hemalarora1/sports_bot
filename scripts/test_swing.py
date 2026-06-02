#!/usr/bin/env python3
"""Drive the Panda arm through a predefined swing: wind-up → strike → follow-through.

Bypasses the FSM entirely — no ball tracking needed. Useful for verifying the
arm moves to the right poses before integrating with live ball data.

Run from the OpenSai root:
    python -m sports_bot.scripts.test_swing --dry-run         # preview poses, no movement
    python -m sports_bot.scripts.test_swing                   # standard forward swing
    python -m sports_bot.scripts.test_swing --underhand --dry-run  # underhand preview

Pre-reqs: Redis running + OpenSai cartesian controller running (not needed for --dry-run).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import redis

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SPORTS_BOT_DIR = os.path.dirname(_THIS_DIR)
_OPENSAI_DIR = os.path.dirname(_SPORTS_BOT_DIR)
if _OPENSAI_DIR not in sys.path:
    sys.path.insert(0, _OPENSAI_DIR)

from sports_bot.state_machine.ball_tracker import Intercept   # noqa: E402
from sports_bot.state_machine.config import PickleballConfig  # noqa: E402
from sports_bot.state_machine.redis_keys import RedisKeys     # noqa: E402
from sports_bot.state_machine.swing_planner import SwingPlan, SwingPlanner  # noqa: E402


# ---------- helpers ------------------------------------------------------------

def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-9:
        raise ValueError("Cannot normalize a zero vector")
    return v / n


def _rotation_from_normal(face_normal: np.ndarray) -> np.ndarray:
    """Build a 3x3 rotation [face_right, face_up, face_normal] from a face normal."""
    n = _normalize(face_normal)
    world_up = np.array([0.0, 0.0, 1.0])
    up_proj = world_up - np.dot(world_up, n) * n
    if np.linalg.norm(up_proj) < 1e-6:
        up_proj = np.array([1.0, 0.0, 0.0]) - np.dot(np.array([1.0, 0.0, 0.0]), n) * n
    face_up = _normalize(up_proj)
    face_right = np.cross(face_up, n)
    return np.column_stack([face_right, face_up, n])


def write_racket_goal(
    r: redis.Redis,
    keys: RedisKeys,
    position: np.ndarray,
    orientation: np.ndarray,
    velocity: np.ndarray | None = None,
) -> None:
    vel = velocity if velocity is not None else np.zeros(3)
    r.set(keys.opensai.goal_position,        json.dumps(position.tolist()))
    r.set(keys.opensai.goal_orientation,     json.dumps(orientation.tolist()))
    r.set(keys.opensai.goal_linear_velocity, json.dumps(vel.tolist()))


# ---------- main ---------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])

    # --- strike point ---
    parser.add_argument(
        "--strike-x", type=float, default=None,
        help="Strike plane X (m). Defaults to strike_plane_x in config (0.60).",
    )
    parser.add_argument(
        "--strike-y", type=float, default=0.0,
        help="Lateral offset of strike point (m). Default 0.",
    )
    parser.add_argument(
        "--strike-z", type=float, default=0.90,
        help="Height of strike point (m). Default 0.90.",
    )

    # --- underhand options ---
    parser.add_argument(
        "--underhand", action="store_true",
        help="Use an underhand swing: arm sweeps upward through the strike point.",
    )
    parser.add_argument(
        "--swing-angle", type=float, default=30.0,
        help="(--underhand only) Angle of the swing direction above horizontal, "
             "in degrees. 0 = purely forward, 90 = purely upward. Default 60.",
    )
    parser.add_argument(
        "--swing-offset", type=float, default=None,
        help="Distance (m) for wind-up and follow-through offset from the strike point. "
             "Default 0.15 for --underhand, 0.25 otherwise (matches config).",
    )

    # --- timing ---
    parser.add_argument(
        "--windup-hold", type=float, default=3.0,
        help="Seconds to hold wind-up pose before striking. Default 3.",
    )
    parser.add_argument(
        "--strike-hold", type=float, default=0.5,
        help="Seconds to hold strike pose. Default 0.5.",
    )
    parser.add_argument(
        "--follow-hold", type=float, default=2.0,
        help="Seconds to hold follow-through pose before exiting. Default 2.",
    )

    # --- misc ---
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print computed poses without connecting to Redis or moving the robot.",
    )
    parser.add_argument("--redis-host", default="localhost")
    parser.add_argument("--redis-port", type=int, default=6379)
    args = parser.parse_args()

    # Only connect to Redis if we're actually going to send commands.
    r: redis.Redis | None = None
    if not args.dry_run:
        r = redis.Redis(host=args.redis_host, port=args.redis_port)
        try:
            r.ping()
        except redis.exceptions.ConnectionError as e:
            print(f"[test_swing] Cannot reach Redis at {args.redis_host}:{args.redis_port}: {e}")
            sys.exit(1)

    cfg = PickleballConfig()
    from sports_bot.state_machine.redis_keys import OpenSaiCartesianKeys
    keys = RedisKeys(robot_backend="opensai", opensai=OpenSaiCartesianKeys(robot_name="FrankaRobot"))
    planner = SwingPlanner(cfg.court, cfg.racket)

    strike_x = args.strike_x if args.strike_x is not None else cfg.court.strike_plane_x
    strike_point = np.array([strike_x, args.strike_y, args.strike_z])

    # Use planner for bounds checking and base pose; we may override pose geometry below.
    intercept = Intercept(position=strike_point, velocity=np.zeros(3), time_to_impact=1.0)
    plan = planner.plan(intercept)

    if plan is None:
        print(
            f"[test_swing] Strike point {strike_point.tolist()} is outside reachable bounds.\n"
            f"  strike_z must be between {cfg.court.strike_z_min} and {cfg.court.strike_z_max} m\n"
            f"  strike_y must be between {cfg.court.base_y_min - 0.5} and {cfg.court.base_y_max + 0.5} m"
        )
        sys.exit(1)

    if args.underhand:
        # Swing direction: forward (+X) and upward (+Z) at the requested angle.
        # Wind-up is below-and-behind the strike point; follow is above-and-forward.
        angle_rad = math.radians(args.swing_angle)
        swing_dir = np.array([math.cos(angle_rad), 0.0, math.sin(angle_rad)])
        offset = args.swing_offset if args.swing_offset is not None else 0.15

        wind_up_pos  = strike_point - offset * swing_dir
        follow_pos   = strike_point + offset * swing_dir
        strike_vel   = cfg.racket.impact_speed * swing_dir
        # Racket face points in the swing direction so the ball is redirected forward-up.
        ori = _rotation_from_normal(swing_dir)

        plan = SwingPlan(
            base_pose=plan.base_pose,
            wind_up_position=wind_up_pos,
            wind_up_orientation=ori,
            strike_position=strike_point,
            strike_orientation=ori,
            follow_position=follow_pos,
            follow_orientation=ori,
            strike_velocity=strike_vel,
            time_to_impact=plan.time_to_impact,
        )
        print(f"[test_swing] Mode        : UNDERHAND (angle={args.swing_angle}°, offset={offset}m)")
    else:
        offset = args.swing_offset
        if offset is not None:
            # Re-compute standard plan with custom offset.
            face_normal = _normalize(plan.strike_orientation[:, 2])
            plan = SwingPlan(
                base_pose=plan.base_pose,
                wind_up_position=strike_point - offset * face_normal,
                wind_up_orientation=plan.wind_up_orientation,
                strike_position=strike_point,
                strike_orientation=plan.strike_orientation,
                follow_position=strike_point + offset * face_normal,
                follow_orientation=plan.follow_orientation,
                strike_velocity=plan.strike_velocity,
                time_to_impact=plan.time_to_impact,
            )
        print(f"[test_swing] Mode        : STANDARD forward swing")

    print(f"[test_swing] Strike point : {strike_point.tolist()}")
    print(f"[test_swing] Wind-up pos  : {plan.wind_up_position.tolist()}")
    print(f"[test_swing] Strike pos   : {plan.strike_position.tolist()}")
    print(f"[test_swing] Follow pos   : {plan.follow_position.tolist()}")
    print(f"[test_swing] Strike vel   : {plan.strike_velocity.tolist()}")
    print(f"[test_swing] Strike ori   :\n{plan.strike_orientation}")
    print()

    if args.dry_run:
        print("[test_swing] DRY RUN — no Redis commands sent. Remove --dry-run to move the arm.")
        return

    print(f"[test_swing] WIND-UP — sending pose, holding for {args.windup_hold:.1f}s ...")
    write_racket_goal(r, keys, plan.wind_up_position, plan.wind_up_orientation)
    time.sleep(args.windup_hold)

    print(f"[test_swing] STRIKE — sending pose + velocity ...")
    write_racket_goal(r, keys, plan.strike_position, plan.strike_orientation, plan.strike_velocity)
    time.sleep(args.strike_hold)

    print(f"[test_swing] FOLLOW-THROUGH — sending pose, holding for {args.follow_hold:.1f}s ...")
    write_racket_goal(r, keys, plan.follow_position, plan.follow_orientation)
    time.sleep(args.follow_hold)

    print("[test_swing] Done.")


if __name__ == "__main__":
    main()
