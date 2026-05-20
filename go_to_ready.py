#!/usr/bin/env python3
"""Send the robot to the ReadyPose once and wait for it to arrive.

Usage (opensai backend, default):
    python go_to_ready.py

Usage (cs225a backend):
    python go_to_ready.py --robot-backend cs225a

Make sure the controller (and sim/hardware) are already running before
you execute this script.
"""

import argparse
import json
import sys
import time

import numpy as np
import redis

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))

from state_machine.config import PickleballConfig
from state_machine.redis_keys import RedisKeys


def send_ready_pose(r: redis.Redis, keys: RedisKeys, cfg: PickleballConfig) -> None:
    ready = cfg.ready

    if keys.robot_backend == "opensai":
        pos_key = keys.opensai.goal_position
        ori_key = keys.opensai.goal_orientation
        cur_pos_key = keys.opensai.current_position
        cur_ori_key = keys.opensai.current_orientation
    else:
        pos_key = keys.cs225a.racket_goal_position
        ori_key = keys.cs225a.racket_goal_orientation
        cur_pos_key = keys.cs225a.racket_current_position
        cur_ori_key = keys.cs225a.racket_current_orientation

    print(f"[go_to_ready] Writing ready pose to Redis ({keys.robot_backend} backend)...")
    print(f"  racket position   : {ready.racket_position}")
    print(f"  base pose         : {ready.base_pose}  (cs225a only)")

    r.set(pos_key, json.dumps(ready.racket_position.tolist()))
    r.set(ori_key, json.dumps(ready.racket_orientation.tolist()))

    if keys.robot_backend == "cs225a":
        r.set(keys.cs225a.base_goal_pose, json.dumps(ready.base_pose.tolist()))

    # ---- poll until arrived (or timeout) --------------------------------------
    pos_tol = 0.05   # metres
    timeout = 10.0   # seconds
    t0 = time.perf_counter()

    print("[go_to_ready] Waiting for robot to arrive (timeout 10 s)...")

    while True:
        elapsed = time.perf_counter() - t0
        if elapsed > timeout:
            print("[go_to_ready] WARNING: timed out waiting for convergence.")
            break

        raw_pos = r.get(cur_pos_key)
        if raw_pos is None:
            time.sleep(0.05)
            continue

        cur_pos = np.array(json.loads(raw_pos))
        err = float(np.linalg.norm(cur_pos - ready.racket_position))

        print(f"\r  position error: {err:.4f} m    ", end="", flush=True)

        if err < pos_tol:
            print(f"\n[go_to_ready] Arrived! (err={err:.4f} m, t={elapsed:.2f} s)")
            break

        time.sleep(0.05)


def main() -> None:
    parser = argparse.ArgumentParser(description="Move robot to ReadyPose.")
    parser.add_argument(
        "--robot-backend",
        default="opensai",
        choices=["opensai", "cs225a"],
        help="Which controller backend is running (default: opensai)",
    )
    parser.add_argument("--redis-host", default="localhost")
    parser.add_argument("--redis-port", type=int, default=6379)
    args = parser.parse_args()

    r = redis.Redis(host=args.redis_host, port=args.redis_port, decode_responses=True)
    try:
        r.ping()
    except redis.exceptions.ConnectionError as e:
        print(f"[go_to_ready] Cannot connect to Redis at {args.redis_host}:{args.redis_port}: {e}")
        sys.exit(1)

    keys = RedisKeys(robot_backend=args.robot_backend)
    cfg = PickleballConfig()

    send_ready_pose(r, keys, cfg)


if __name__ == "__main__":
    main()
