#!/usr/bin/env python3
"""Base-only ball-intercept loop. No arm, no full FSM.

Pipeline
--------
    OptiTrack ball  ──▶  BallTracker.predict_intercept(x = strike_plane_x)
                                       │
                                       ▼
                         clamp y_pred to [base_y_min, base_y_max]
                                       │
                                       ▼
            sports_bot::cmd::base::goal_pose  =  [ready_x, y_pred, ready_yaw]
                                       │
                                       ▼
                              base_bridge.py  ──▶  hb1::desired_pose

Strategy
--------
- Sits at ``ready_base_pose`` until a ball is detected.
- When the tracker returns an incoming intercept at the strike plane, the cart
  is commanded to align its centerline (only Y changes; X and yaw stay at
  ready). The arm is untouched.
- When no usable prediction has arrived for ``hold_after_lost_s`` seconds, the
  cart returns to the ready pose.

What this script DOES NOT do (on purpose)
-----------------------------------------
- No commit-time / swing trigger — the cart just continuously tracks the
  predicted Y.
- No arm control. Whatever has the arm task active keeps it; this script
  doesn't write to any racket / cartesian / joint keys.
- No feasibility check on whether the base can actually arrive on time. It
  prints predicted-arrival vs. predicted-impact for inspection but commands
  the goal regardless. Tune ``--min-lookahead`` / ``--max-lookahead`` to gate.

Prereqs (in order)
------------------
1. redis-server.
2. OptiTrack streamer publishing ball + cart rigid bodies.
3. TidyBot redis_driver.py.
4. sports_bot/base_bridge.py --robot-rigid-body-id <CART_ID>.
5. THIS script.

Usage
-----
    conda activate opensai
    # measure your workspace first (see --help) and pass real limits
    python sports_bot/scripts/base_intercept.py \
        --ball-rigid-body-id 8 \
        --strike-plane-x 0.60 \
        --ready-x 0.0 --ready-y 0.0 --ready-yaw-deg 0 \
        --base-y-min -1.0 --base-y-max 1.0
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

_THIS_FILE = os.path.abspath(__file__)
_OPENSAI_DIR = os.path.dirname(os.path.dirname(os.path.dirname(_THIS_FILE)))
if _OPENSAI_DIR not in sys.path:
    sys.path.insert(0, _OPENSAI_DIR)

from sports_bot.state_machine.ball_tracker import BallTracker  # noqa: E402
from sports_bot.state_machine.config import BallTrackerConfig  # noqa: E402
from sports_bot.state_machine.redis_keys import RedisKeys      # noqa: E402


FSM_BASE_GOAL = "sports_bot::cmd::base::goal_pose"


def _clamp(v: float, lo: float, hi: float) -> Tuple[float, bool]:
    if v < lo:
        return lo, True
    if v > hi:
        return hi, True
    return v, False


def _write_goal(r: redis.Redis, pose: Tuple[float, float, float]) -> None:
    r.set(FSM_BASE_GOAL, json.dumps([float(pose[0]), float(pose[1]), float(pose[2])]))


def run(
    r: redis.Redis,
    tracker: BallTracker,
    ready_pose: Tuple[float, float, float],
    strike_plane_x: float,
    base_y_min: float,
    base_y_max: float,
    rate_hz: float,
    hold_after_lost_s: float,
    print_interval_s: float,
) -> None:
    dt = 1.0 / rate_hz
    last_print = 0.0
    last_good_t = -math.inf
    last_goal: Optional[Tuple[float, float, float]] = None
    at_ready = False

    # Park at ready immediately so the cart settles while we wait for the ball.
    _write_goal(r, ready_pose)
    last_goal = ready_pose
    at_ready = True
    print(f"[base_intercept] ready_pose = ({ready_pose[0]:+.3f}, "
          f"{ready_pose[1]:+.3f}, {math.degrees(ready_pose[2]):+.1f}°)")
    print(f"[base_intercept] strike_plane_x = {strike_plane_x:+.3f} m,  "
          f"y limits = [{base_y_min:+.3f}, {base_y_max:+.3f}]")
    print(f"[base_intercept] writing {FSM_BASE_GOAL} at {rate_hz:.0f} Hz "
          f"(Ctrl-C to stop)\n")

    while True:
        t0 = time.perf_counter()

        tracker.update()
        intercept = tracker.predict_intercept(strike_plane_x)

        if intercept is not None:
            y_raw = float(intercept.position[1])
            y_clamped, clamped = _clamp(y_raw, base_y_min, base_y_max)
            goal = (ready_pose[0], y_clamped, ready_pose[2])
            if last_goal is None or any(
                abs(goal[i] - last_goal[i]) > 1e-4 for i in range(3)
            ):
                _write_goal(r, goal)
                last_goal = goal
            at_ready = False
            last_good_t = t0

            if t0 - last_print >= print_interval_s:
                last_print = t0
                tag = " (clamped)" if clamped else ""
                z = float(intercept.position[2])
                tti = float(intercept.time_to_impact)
                vy = float(intercept.velocity[1])
                print(
                    f"[base_intercept] intercept y={y_raw:+.3f}{tag}  "
                    f"z={z:+.2f}  tti={tti:+.3f}s  v_y={vy:+.2f} m/s  "
                    f"→ goal_W=({goal[0]:+.3f}, {goal[1]:+.3f})"
                )
        else:
            # No usable prediction this tick. After we've been without one for
            # hold_after_lost_s, fall back to ready.
            since_good = t0 - last_good_t
            if since_good > hold_after_lost_s and not at_ready:
                _write_goal(r, ready_pose)
                last_goal = ready_pose
                at_ready = True
                reason = getattr(tracker, "last_reject_reason", "") or "no_data"
                print(f"[base_intercept] lost for {since_good:.2f}s "
                      f"(reason={reason}) → return to ready")

        elapsed = time.perf_counter() - t0
        sleep = dt - elapsed
        if sleep > 0:
            time.sleep(sleep)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Base-only ball intercept driver (writes "
                    "sports_bot::cmd::base::goal_pose; needs base_bridge.py "
                    "running to forward to the cart).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ball-rigid-body-id", type=int, default=8,
                   help="Motive Streaming ID of the pickleball.")
    p.add_argument("--redis-host", default="localhost")
    p.add_argument("--redis-port", type=int, default=6379)

    # Workspace measurements — REPLACE WITH REAL VALUES after walking the cart.
    p.add_argument("--strike-plane-x", type=float, default=0.60,
                   help="World-X (m) at which the predicted intercept is "
                        "evaluated. Set to ~the cart's forward standoff in "
                        "your bay.")
    p.add_argument("--ready-x", type=float, default=0.0,
                   help="World-X of the ready / rest base pose.")
    p.add_argument("--ready-y", type=float, default=0.0,
                   help="World-Y of the ready / rest base pose.")
    p.add_argument("--ready-yaw-deg", type=float, default=0.0,
                   help="World yaw (deg) of the ready / rest base pose.")
    p.add_argument("--base-y-min", type=float, default=-1.0,
                   help="Min world-Y the base is allowed to reach. "
                        "MEASURE this in the actual workspace.")
    p.add_argument("--base-y-max", type=float, default=1.0,
                   help="Max world-Y the base is allowed to reach. "
                        "MEASURE this in the actual workspace.")

    # Loop / behavior knobs.
    p.add_argument("--rate-hz", type=float, default=50.0,
                   help="Control loop rate.")
    p.add_argument("--hold-after-lost-s", type=float, default=0.6,
                   help="How long to hold the last intercept goal after the "
                        "tracker stops returning usable predictions before "
                        "falling back to ready.")
    p.add_argument("--print-interval-s", type=float, default=0.1,
                   help="Throttle for intercept prints (the loop runs faster).")

    # Tracker lookahead overrides — the default config caps at 1.5 s and 0.05 s,
    # which is usually fine; expose them for tuning without editing config.py.
    p.add_argument("--min-lookahead", type=float, default=None,
                   help="Override BallTrackerConfig.min_lookahead (s).")
    p.add_argument("--max-lookahead", type=float, default=None,
                   help="Override BallTrackerConfig.max_lookahead (s).")

    args = p.parse_args()

    if args.base_y_min >= args.base_y_max:
        p.error("--base-y-min must be < --base-y-max")
    if not (args.base_y_min <= args.ready_y <= args.base_y_max):
        print(f"[base_intercept] WARNING: ready_y={args.ready_y} is outside "
              f"[{args.base_y_min}, {args.base_y_max}].")

    r = redis.Redis(host=args.redis_host, port=args.redis_port, decode_responses=True)
    try:
        r.ping()
    except redis.exceptions.ConnectionError as e:
        print(f"[base_intercept] cannot reach Redis "
              f"at {args.redis_host}:{args.redis_port}: {e}")
        sys.exit(1)

    cfg = BallTrackerConfig()
    if args.min_lookahead is not None:
        cfg.min_lookahead = args.min_lookahead
    if args.max_lookahead is not None:
        cfg.max_lookahead = args.max_lookahead

    keys = RedisKeys(ball_source="optitrack")
    # Override the rigid-body id without recreating the BallKeys default.
    keys.ball.__dict__["optitrack_rigid_body_id"] = args.ball_rigid_body_id

    # Sanity: do we actually see ball positions?
    ball_key = keys.ball.optitrack_position
    if r.get(ball_key) is None:
        print(f"[base_intercept] WARNING: {ball_key} is empty. "
              f"Is the OptiTrack streamer running?")
    else:
        print(f"[base_intercept] reading ball from {ball_key}")

    tracker = BallTracker(r, keys, cfg)

    ready_pose = (args.ready_x, args.ready_y, math.radians(args.ready_yaw_deg))

    try:
        run(
            r=r,
            tracker=tracker,
            ready_pose=ready_pose,
            strike_plane_x=args.strike_plane_x,
            base_y_min=args.base_y_min,
            base_y_max=args.base_y_max,
            rate_hz=args.rate_hz,
            hold_after_lost_s=args.hold_after_lost_s,
            print_interval_s=args.print_interval_s,
        )
    except KeyboardInterrupt:
        print("\n[base_intercept] stopping — leaving last goal in Redis.")
        print(f"               (write {FSM_BASE_GOAL} or restart the bridge "
              f"to reset.)")


if __name__ == "__main__":
    main()
