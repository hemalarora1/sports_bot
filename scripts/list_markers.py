#!/usr/bin/env python3
"""Live-print every labeled OptiTrack marker visible to the streamer.

Use this to figure out which `(model_id, marker_id)` corresponds to which
physical marker on the cart — wave your hand over a single marker (or pop
it off briefly), watch which key's value changes / disappears, and write
those IDs into arm_marker_calibration.json.

`model_id` is the rigid-body asset the marker belongs to (= the asset's
Motive Streaming ID; 11 for the cart). `marker_id` is the per-asset marker
index Motive assigns (1, 2, 3, ...).

Prereqs:
  1. Redis running.
  2. StreamDataSkeleton.py running with labeled-marker streaming enabled.
  3. Motive: Application Settings → Streaming → Labeled Markers = ENABLED.

Usage (from OpenSai root):
    python sports_bot/scripts/list_markers.py
    python sports_bot/scripts/list_markers.py --model-id 11    # filter to cart asset only
    python sports_bot/scripts/list_markers.py --rate-hz 5      # slower refresh
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import redis

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SPORTS_BOT_DIR = os.path.dirname(_THIS_DIR)
_OPENSAI_DIR = os.path.dirname(_SPORTS_BOT_DIR)
if _OPENSAI_DIR not in sys.path:
    sys.path.insert(0, _OPENSAI_DIR)


MARKER_POS_PATTERN = "sai2::optitrack::marker_pos::*"


def _parse_key(key: str):
    # Format: sai2::optitrack::marker_pos::<model_id>::<marker_id>
    try:
        suffix = key.split("sai2::optitrack::marker_pos::", 1)[1]
        model_id_str, marker_id_str = suffix.split("::", 1)
        return int(model_id_str), int(marker_id_str)
    except (IndexError, ValueError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--redis-host", default="localhost")
    parser.add_argument("--redis-port", type=int, default=6379)
    parser.add_argument("--model-id", type=int, default=None,
                        help="If set, only show markers belonging to this "
                             "asset / rigid body (e.g., 11 for the cart).")
    parser.add_argument("--rate-hz", type=float, default=2.0,
                        help="Refresh rate (Hz). Default 2.")
    args = parser.parse_args()

    r = redis.Redis(host=args.redis_host, port=args.redis_port, decode_responses=True)
    try:
        r.ping()
    except redis.exceptions.ConnectionError as e:
        print(f"[list_markers] Cannot reach Redis: {e}")
        sys.exit(1)

    dt = 1.0 / args.rate_hz
    print(f"[list_markers] watching {MARKER_POS_PATTERN}"
          + (f" (filtered to model_id={args.model_id})" if args.model_id is not None else "")
          + " — Ctrl-C to stop\n")

    try:
        while True:
            keys = sorted(r.scan_iter(match=MARKER_POS_PATTERN))
            entries = []
            for key in keys:
                parsed = _parse_key(key)
                if parsed is None:
                    continue
                model_id, marker_id = parsed
                if args.model_id is not None and model_id != args.model_id:
                    continue
                raw = r.get(key)
                if raw is None:
                    continue
                try:
                    pos = json.loads(raw)
                except (json.JSONDecodeError, TypeError, ValueError):
                    continue
                if len(pos) != 3:
                    continue
                entries.append((model_id, marker_id, pos))

            # Clear-screen + header
            sys.stdout.write("\x1b[2J\x1b[H")
            sys.stdout.write(f"=== {time.strftime('%H:%M:%S')} — "
                             f"{len(entries)} marker(s) visible ===\n")
            sys.stdout.write(f"{'model_id':>9}  {'marker_id':>9}  "
                             f"{'x (m)':>9}  {'y (m)':>9}  {'z (m)':>9}\n")
            sys.stdout.write("-" * 53 + "\n")
            for model_id, marker_id, pos in entries:
                sys.stdout.write(
                    f"{model_id:>9}  {marker_id:>9}  "
                    f"{pos[0]:>+9.3f}  {pos[1]:>+9.3f}  {pos[2]:>+9.3f}\n"
                )
            sys.stdout.flush()
            time.sleep(dt)
    except KeyboardInterrupt:
        print("\n[list_markers] stopped.")


if __name__ == "__main__":
    main()
