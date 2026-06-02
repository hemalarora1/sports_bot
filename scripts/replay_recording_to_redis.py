#!/usr/bin/env python3
"""Replay a recorded OptiTrack ball trajectory back into Redis in real time.

This publishes the same key that the live OptiTrack streamer publishes:

    sai2::optitrack::rigid_body_pos::<id>

Use it to test the live BallTracker/J6 timing path without a physical ball and
without impact/reflex events. The recording format is the .npz produced by:

    python -m sports_bot.state_machine.ball_tracker_test record ...

Examples from OpenSai root:

    python sports_bot/scripts/replay_recording_to_redis.py sports_bot/recordings/throws_*.npz --rigid-body-id 1
    python sports_bot/scripts/replay_recording_to_redis.py sports_bot/recordings/throws_20260529_012038.npz --segment 2 --rigid-body-id 1
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import redis


POS_KEY_PREFIX = "sai2::optitrack::rigid_body_pos::"
ORI_KEY_PREFIX = "sai2::optitrack::rigid_body_ori::"
RAW_POS_KEY_PREFIX = "sai2::optitrack::raw::rigid_body_pos::"
RAW_ORI_KEY_PREFIX = "sai2::optitrack::raw::rigid_body_ori::"
IDENTITY_QUAT = [0.0, 0.0, 0.0, 1.0]


def _segments(ts: np.ndarray, gap_s: float, min_samples: int) -> list[tuple[int, int]]:
    if len(ts) == 0:
        return []
    starts = [0]
    for i in range(1, len(ts)):
        if float(ts[i] - ts[i - 1]) > gap_s:
            starts.append(i)
    starts.append(len(ts))
    out: list[tuple[int, int]] = []
    for a, b in zip(starts[:-1], starts[1:]):
        if b - a >= min_samples:
            out.append((a, b))
    return out


def _fmt(v: np.ndarray) -> str:
    return f"[{v[0]:+.3f},{v[1]:+.3f},{v[2]:+.3f}]"


def _publish(r: redis.Redis, rb_id: int, pos: np.ndarray, publish_raw: bool) -> None:
    pos_json = json.dumps(np.asarray(pos, dtype=float).reshape(3).tolist())
    quat_json = json.dumps(IDENTITY_QUAT)
    r.set(POS_KEY_PREFIX + str(rb_id), pos_json)
    r.set(ORI_KEY_PREFIX + str(rb_id), quat_json)
    if publish_raw:
        r.set(RAW_POS_KEY_PREFIX + str(rb_id), pos_json)
        r.set(RAW_ORI_KEY_PREFIX + str(rb_id), quat_json)


def _clear(r: redis.Redis, rb_id: int) -> None:
    keys = [
        POS_KEY_PREFIX + str(rb_id),
        ORI_KEY_PREFIX + str(rb_id),
        RAW_POS_KEY_PREFIX + str(rb_id),
        RAW_ORI_KEY_PREFIX + str(rb_id),
    ]
    r.delete(*keys)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Replay a ball_tracker_test .npz recording into Redis as a live OptiTrack ball."
    )
    ap.add_argument("recording", help="Path to .npz recording from ball_tracker_test record.")
    ap.add_argument("--rigid-body-id", type=int, default=None,
                    help="Redis rigid body ID to publish. Default: ID stored in recording, else 1.")
    ap.add_argument("--redis-host", default="localhost")
    ap.add_argument("--redis-port", type=int, default=6379)
    ap.add_argument("--segment", type=int, default=None,
                    help="Replay only one gap-separated segment, 1-indexed.")
    ap.add_argument("--segment-gap-s", type=float, default=0.4,
                    help="Time gap that separates recorded throws.")
    ap.add_argument("--segment-min-samples", type=int, default=5)
    ap.add_argument("--time-scale", type=float, default=1.0,
                    help="1.0 = real time; 0.5 = twice as fast; 2.0 = half speed.")
    ap.add_argument("--lead-s", type=float, default=1.0,
                    help="Seconds to wait after clearing the key before publishing the first sample.")
    ap.add_argument("--tail-hold-s", type=float, default=0.0,
                    help="Keep publishing the final sample for this many seconds after replay.")
    ap.add_argument("--loop", action="store_true", help="Replay repeatedly until Ctrl-C.")
    ap.add_argument("--loop-gap-s", type=float, default=2.0)
    ap.add_argument("--keep-key", action="store_true",
                    help="Leave final position in Redis on exit instead of deleting the replayed ball key.")
    ap.add_argument("--publish-raw", action="store_true",
                    help="Also publish raw OptiTrack debug keys for this rigid body ID.")
    ap.add_argument("--dry-run", action="store_true", help="Print selected segments but do not publish.")
    args = ap.parse_args()

    if args.time_scale <= 0:
        ap.error("--time-scale must be positive")

    rec_path = Path(args.recording)
    if not rec_path.is_file():
        sys.exit(f"[replay] recording not found: {rec_path}")

    data = np.load(rec_path, allow_pickle=False)
    ts = np.asarray(data["timestamps"], dtype=float)
    ps = np.asarray(data["positions"], dtype=float)
    if ps.ndim != 2 or ps.shape[1] != 3 or len(ts) != len(ps):
        sys.exit(f"[replay] malformed recording: timestamps {ts.shape}, positions {ps.shape}")
    if len(ts) < 2:
        sys.exit("[replay] recording has fewer than 2 samples")

    rb_id = args.rigid_body_id
    if rb_id is None:
        try:
            rb_id = int(np.asarray(data["optitrack_rigid_body_id"]).item())
        except Exception:
            rb_id = 1

    segs = _segments(ts, args.segment_gap_s, args.segment_min_samples)
    if not segs:
        sys.exit("[replay] no usable segments found")
    if args.segment is not None:
        if args.segment < 1 or args.segment > len(segs):
            sys.exit(f"[replay] --segment must be 1..{len(segs)}")
        selected = [segs[args.segment - 1]]
    else:
        selected = segs

    print(f"[replay] recording: {rec_path}")
    print(f"[replay] publish key: {POS_KEY_PREFIX}{rb_id}")
    print(f"[replay] segments: {len(segs)} total; replaying "
          + (f"segment {args.segment}" if args.segment is not None else "all segments"))
    for i, (a, b) in enumerate(segs, start=1):
        dur = float(ts[b - 1] - ts[a])
        mark = "*" if (a, b) in selected else " "
        print(f"[replay] {mark} segment {i:02d}: samples={b-a:4d} duration={dur:5.2f}s "
              f"start={_fmt(ps[a])} end={_fmt(ps[b-1])}")

    if args.dry_run:
        return 0

    r = redis.Redis(host=args.redis_host, port=args.redis_port, decode_responses=True)
    try:
        r.ping()
    except redis.exceptions.ConnectionError as exc:
        sys.exit(f"[replay] cannot reach Redis at {args.redis_host}:{args.redis_port}: {exc}")

    try:
        pass_idx = 0
        while True:
            pass_idx += 1
            print(f"[replay] pass {pass_idx}: clearing key, lead {args.lead_s:.1f}s")
            _clear(r, rb_id)
            time.sleep(max(0.0, args.lead_s))

            for seg_idx, (a, b) in enumerate(selected, start=1):
                local_ts = ts[a:b] - ts[a]
                local_ps = ps[a:b]
                print(f"[replay] segment {seg_idx}/{len(selected)}: publishing {b-a} samples")
                wall0 = time.perf_counter()
                for rel_t, pos in zip(local_ts, local_ps):
                    target_t = wall0 + float(rel_t) * args.time_scale
                    sleep_for = target_t - time.perf_counter()
                    if sleep_for > 0:
                        time.sleep(sleep_for)
                    _publish(r, rb_id, pos, args.publish_raw)
                if args.tail_hold_s > 0:
                    end_t = time.perf_counter() + args.tail_hold_s
                    while time.perf_counter() < end_t:
                        _publish(r, rb_id, local_ps[-1], args.publish_raw)
                        time.sleep(0.02)
                if seg_idx != len(selected):
                    _clear(r, rb_id)
                    time.sleep(max(0.0, args.loop_gap_s))

            if not args.loop:
                break
            _clear(r, rb_id)
            print(f"[replay] loop gap {args.loop_gap_s:.1f}s")
            time.sleep(max(0.0, args.loop_gap_s))
    except KeyboardInterrupt:
        print("\n[replay] stopped")
    finally:
        if not args.keep_key:
            _clear(r, rb_id)
            print("[replay] cleared replayed ball key")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
