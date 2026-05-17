#!/usr/bin/env python3
"""Simple Redis consumer that reads OptiTrack rigid-body pose keys and prints them.

Usage:
  python3 robot_tracker.py --id 1 --rate 50

It reads keys written by StreamDataSkeleton.py:
  - sai2::optitrack::rigid_body_pos::<id>
  - sai2::optitrack::rigid_body_ori::<id>

Position is expected as a bracketed CSV string like "[x, y, z]".
Orientation is stored similarly (quaternion or rotation vector depending on NatNet client).
"""
import time
import redis
import argparse
from typing import Optional, Tuple


def parse_bracket_list(s: str) -> Optional[Tuple[float, ...]]:
    if s is None:
        return None
    s = s.strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    if len(s) == 0:
        return None
    try:
        parts = [float(p.strip()) for p in s.split(",")]
        return tuple(parts)
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=int, default=1, help="Rigid body ID to track")
    parser.add_argument("--rate", type=float, default=50.0, help="Poll rate (Hz)")
    parser.add_argument("--host", type=str, default="localhost", help="Redis host")
    parser.add_argument("--port", type=int, default=6379, help="Redis port")
    args = parser.parse_args()

    r = redis.Redis(host=args.host, port=args.port, decode_responses=True)

    pos_key = f"sai2::optitrack::rigid_body_pos::{args.id}"
    ori_key = f"sai2::optitrack::rigid_body_ori::{args.id}"

    interval = 1.0 / float(args.rate) if args.rate > 0 else 0.02

    print(f"Tracking rigid body {args.id} (pos: {pos_key}, ori: {ori_key}) @ {args.rate} Hz")
    try:
        while True:
            pos_s = r.get(pos_key)
            ori_s = r.get(ori_key)

            pos = parse_bracket_list(pos_s)
            ori = parse_bracket_list(ori_s)

            ts = time.time()
            if pos is not None:
                print(f"{ts:.3f} POS {pos}")
            else:
                print(f"{ts:.3f} POS <missing>")

            if ori is not None:
                print(f"{ts:.3f} ORI {ori}")
            else:
                print(f"{ts:.3f} ORI <missing>")

            time.sleep(interval)
    except KeyboardInterrupt:
        print("exiting")


if __name__ == "__main__":
    main()
