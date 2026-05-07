#!/usr/bin/env python3
"""Trigger a pickleball launch in the running simviz_mmp_panda simulator.

The C++ simviz polls a launch counter in redis. We write the desired pose and
velocity, then bump the counter. simviz teleports the ball and applies the
linear velocity on the next tick. The python FSM (with --ball-source either
optitrack or opensai) sees the resulting motion the same way it would on the
real cart.

Examples:

    # default lob from opponent court toward the robot
    python -m sports_bot.mmp_panda.launch_ball

    # custom shot: position [x y z], velocity [vx vy vz]
    python -m sports_bot.mmp_panda.launch_ball \
        --pos 5.0 0.3 1.7 --vel -6.0 -0.4 1.5
"""

from __future__ import annotations

import argparse
import json
import sys

import redis


# Keep these in sync with sports_bot/mmp_panda/redis_keys.h.
BALL_LAUNCH_COUNTER_KEY  = "sports_bot::sim::ball::launch_counter"
BALL_LAUNCH_POSE_KEY     = "sports_bot::sim::ball::launch_pose"
BALL_LAUNCH_VELOCITY_KEY = "sports_bot::sim::ball::launch_velocity"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Launch a pickleball in simviz.")
    p.add_argument("--pos", nargs=3, type=float, metavar=("X", "Y", "Z"),
                   default=[5.0, 0.0, 1.6],
                   help="Starting position in world frame (m). Default: 5,0,1.6")
    p.add_argument("--vel", nargs=3, type=float, metavar=("VX", "VY", "VZ"),
                   default=[-5.5, 0.0, 1.0],
                   help="Starting velocity in world frame (m/s). Default: -5.5,0,1.0")
    p.add_argument("--redis-host", default="localhost")
    p.add_argument("--redis-port", type=int, default=6379)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    r = redis.Redis(host=args.redis_host, port=args.redis_port)

    r.set(BALL_LAUNCH_POSE_KEY, json.dumps(list(args.pos)))
    r.set(BALL_LAUNCH_VELOCITY_KEY, json.dumps(list(args.vel)))

    # Read the current counter (simviz initializes it to 0). Increment.
    raw = r.get(BALL_LAUNCH_COUNTER_KEY)
    counter = 0
    if raw is not None:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list) and parsed:
                counter = int(parsed[0])
            elif isinstance(parsed, (int, float)):
                counter = int(parsed)
        except (ValueError, json.JSONDecodeError):
            counter = 0
    counter += 1
    r.set(BALL_LAUNCH_COUNTER_KEY, json.dumps([counter]))

    print(f"[launch_ball] launch #{counter}: pos={args.pos} vel={args.vel}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
