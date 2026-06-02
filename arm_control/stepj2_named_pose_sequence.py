#!/usr/bin/env python3
"""
Step J2: Named joint-pose sequence.

Prereqs:
  - Step J0 joint hold passes.
  - Step J1 wrist/joint nudges pass with the joint-control XML.

This is the first repeatable motion primitive after switching away from
Cartesian control. It anchors on the measured current joint posture, then runs
a small named sequence of relative joint-space poses:

  home -> prep -> strike -> follow -> home

The default preset is intentionally conservative and visual. It is not yet a
ball-strike planner; it is the bridge from "joint nudges are stable" to
"we can command repeatable swing-like motions".

Run from OpenSai root:
    python sports_bot/arm_control/stepj2_named_pose_sequence.py --pause-between
    python sports_bot/arm_control/stepj2_named_pose_sequence.py --preset wrist-only --pause-between
"""
import argparse
import math
import sys

import numpy as np
import redis

from stepj1_joint_nudge import (
    ACTIVE_CONTROLLER,
    CONTROLLER_TO_USE,
    GOAL_JOINTS,
    SENSOR_JOINTS,
    check_joint_limits,
    decode_redis_value,
    ensure_joint_controller,
    fmt_joints_deg,
    get_vec,
    run_segment,
    set_vec,
)


PRESETS_DEG = {
    # Small swing-like cycle using the joints that behaved well in J1, plus q7
    # for visible paddle-face motion. These are relative to measured start.
    "micro-strike": [
        ("prep",   [0.0, 0.0, +3.0, 0.0, 0.0, 0.0, -5.0]),
        ("strike", [+3.0, 0.0, -2.0, 0.0, 0.0, 0.0,  0.0]),
        ("follow", [+5.0, 0.0, -4.0, 0.0, 0.0, 0.0, +5.0]),
    ],
    # Wrist-only visual check. Good for verifying paddle-face repeatability.
    "wrist-only": [
        ("face_open",  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -5.0]),
        ("face_close", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, +5.0]),
        ("face_ref",   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  0.0]),
    ],
    # Wrist posture cycle for q5/q6/q7 after the wrist-stiff gain test.
    "wrist-posture": [
        ("wrist_prep",   [0.0, 0.0, 0.0, 0.0, -3.0, -3.0, -5.0]),
        ("wrist_strike", [0.0, 0.0, 0.0, 0.0,  0.0,  0.0,  0.0]),
        ("wrist_follow", [0.0, 0.0, 0.0, 0.0, +3.0, +3.0, +5.0]),
    ],
}


def parse_pose_delta(text):
    vals = [float(x) for x in text.replace(",", " ").split()]
    if len(vals) != 7:
        raise argparse.ArgumentTypeError("custom pose deltas must have 7 degree values")
    return vals


def max_abs_delta_deg(sequence):
    if not sequence:
        return 0.0
    return max(max(abs(v) for v in delta) for _, delta in sequence)


def build_sequence(args):
    if args.custom_pose:
        sequence = []
        for i, delta in enumerate(args.custom_pose):
            sequence.append((f"custom_{i+1}", delta))
        return sequence
    return PRESETS_DEG[args.preset]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--preset", choices=sorted(PRESETS_DEG), default="micro-strike")
    ap.add_argument(
        "--custom-pose",
        action="append",
        type=parse_pose_delta,
        help=(
            "Add a custom relative pose as 7 degree deltas, e.g. "
            "--custom-pose '0 0 3 0 0 0 -5'. May be passed multiple times."
        ),
    )
    ap.add_argument("--scale", type=float, default=1.0,
                    help="Scale all relative pose deltas.")
    ap.add_argument("--move-s", type=float, default=2.5,
                    help="Smooth interpolation duration for each segment.")
    ap.add_argument("--hold-s", type=float, default=1.0,
                    help="Hold time at each named pose.")
    ap.add_argument("--home-hold-s", type=float, default=2.0,
                    help="Hold time after returning home.")
    ap.add_argument("--publish-hz", type=float, default=100.0)
    ap.add_argument("--switch-timeout", type=float, default=1.0)
    ap.add_argument("--max-delta-deg", type=float, default=8.0,
                    help="Refuse larger relative deltas unless --force is passed.")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--pause-between", action="store_true")
    ap.add_argument("--no-return-home", action="store_true",
                    help="Leave the arm at the final named pose.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print planned poses without writing Redis.")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    sequence_deg = build_sequence(args)
    scaled_sequence_deg = [
        (name, [args.scale * v for v in delta])
        for name, delta in sequence_deg
    ]

    max_delta = max_abs_delta_deg(scaled_sequence_deg)
    if max_delta > args.max_delta_deg and not args.force:
        sys.exit(
            f"ERROR: largest requested relative delta is {max_delta:.1f} deg, "
            f"above --max-delta-deg {args.max_delta_deg:.1f}. Use --force only "
            "after smaller sequences pass."
        )

    r = redis.Redis(host=args.host, port=6379, decode_responses=True)
    try:
        r.ping()
    except Exception:
        sys.exit("ERROR: Redis not reachable")

    q_home = get_vec(r, SENSOR_JOINTS, expected_len=7)
    if q_home is None:
        sys.exit(f"ERROR: no {SENSOR_JOINTS} in Redis - is OpenSai running?")

    active_before = decode_redis_value(r.get(ACTIVE_CONTROLLER))
    print(f"active_controller before: {active_before!r}")
    print(f"joint_goal_key: {GOAL_JOINTS}")
    print("home_joints deg: " + fmt_joints_deg(q_home))
    print(f"preset: {args.preset if not args.custom_pose else 'custom'}  scale={args.scale:.2f}")

    planned = [("home", q_home)]
    for name, delta_deg in scaled_sequence_deg:
        q_target = q_home + np.radians(np.asarray(delta_deg, dtype=float))
        bad_limits = check_joint_limits(q_target)
        if bad_limits:
            for ji, q, lo, hi in bad_limits:
                print(f"ERROR {name}: q{ji} target {q:.1f} deg outside [{lo:.1f}, {hi:.1f}] deg")
            sys.exit(2)
        planned.append((name, q_target))
    if not args.no_return_home:
        planned.append(("return_home", q_home))

    print("\nPLANNED POSES")
    for name, q in planned:
        print(f"  {name:<12} " + fmt_joints_deg(q))

    if args.dry_run:
        print("\ndry-run: no Redis writes performed")
        return

    set_vec(r, GOAL_JOINTS, q_home)
    print("\nseeded current joints as joint_controller goal before switch")
    ensure_joint_controller(r, timeout_s=args.switch_timeout)
    print(f"active_controller: requested {CONTROLLER_TO_USE!r}")

    summaries = []
    q_prev = q_home.copy()
    for i, (name, q_target) in enumerate(planned[1:], start=1):
        hold_s = args.home_hold_s if name == "return_home" else args.hold_s
        q_actual = get_vec(r, SENSOR_JOINTS, expected_len=7)
        q_start = q_actual if q_actual is not None else q_prev
        summaries.append(run_segment(
            r,
            name,
            q_start,
            q_target,
            move_s=args.move_s,
            hold_s=hold_s,
            publish_hz=max(1.0, args.publish_hz),
            verbose=args.verbose,
        ))
        q_prev = q_target.copy()
        if args.pause_between and i < len(planned) - 1:
            input("Press Enter for next named pose...")

    sep = "-" * 72
    print(f"\n{sep}")
    print("STEP J2 NAMED POSE SUMMARY")
    print(f"  {'label':<16} {'err_deg':>8} {'qdot_max':>10} {'status':>8}")
    for item in summaries:
        status = "WARN" if item["vel_warn"] else "OK"
        print(
            f"  {item['label']:<16} {item['goal_err_max_deg']:>8.3f} "
            f"{item['qdot_max_deg_s']:>10.1f} {status:>8}"
        )
    print(sep)


if __name__ == "__main__":
    main()
