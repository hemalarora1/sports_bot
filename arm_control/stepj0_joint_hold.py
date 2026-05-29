#!/usr/bin/env python3
"""
Step J0: Joint-controller hold test.

This is the first gate for pivoting away from OpenSai Cartesian control.
It does not move anywhere on purpose:

  1. Read current Franka joints.
  2. Write those joints to joint_controller/joint_task/goal_position.
  3. Switch active_controller to joint_controller.
  4. Keep refreshing the same joint goal.
  5. Report joint drift and velocity.

Run from OpenSai root:
    python sports_bot/arm_control/stepj0_joint_hold.py
    python sports_bot/arm_control/stepj0_joint_hold.py --duration 12 --verbose

If this is not boring, do not continue to joint-space IK tests.
"""
import argparse
import json
import math
import sys
import time

import numpy as np
import redis

ROBOT_NAME = "FrankaRobot"
CONTROLLER_TO_USE = "joint_controller"
TASK_NAME = "joint_task"

ACTIVE_CONTROLLER = f"opensai::controllers::{ROBOT_NAME}::active_controller_name"
GOAL_JOINTS = f"opensai::controllers::{ROBOT_NAME}::{CONTROLLER_TO_USE}::{TASK_NAME}::goal_position"
SENSOR_JOINTS = f"opensai::sensors::{ROBOT_NAME}::joint_positions"
SENSOR_JOINT_VELS = f"opensai::sensors::{ROBOT_NAME}::joint_velocities"

# Franka hardware velocity limits, rad/s -> deg/s.
JOINT_VEL_LIMIT_DEGS = np.degrees([2.175, 2.175, 2.175, 2.175, 2.610, 2.610, 2.610])
WARN_VEL_FRAC = 0.50


def decode_redis_value(value):
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def get_vec(r, key, expected_len=None):
    raw = r.get(key)
    if raw is None:
        return None
    try:
        arr = np.asarray(json.loads(raw), dtype=float)
    except Exception:
        return None
    if expected_len is not None and arr.shape != (expected_len,):
        return None
    return arr


def set_vec(r, key, vec):
    r.set(key, json.dumps(np.asarray(vec, dtype=float).reshape(-1).tolist()))


def ensure_joint_controller(r, timeout_s=1.0):
    t0 = time.monotonic()
    while True:
        active = decode_redis_value(r.get(ACTIVE_CONTROLLER))
        if active == CONTROLLER_TO_USE:
            return True
        if time.monotonic() - t0 > timeout_s:
            print(
                f"WARNING: could not switch active_controller from {active!r} "
                f"to {CONTROLLER_TO_USE!r} within {timeout_s:.1f}s; continuing anyway",
                file=sys.stderr,
            )
            return False
        r.set(ACTIVE_CONTROLLER, CONTROLLER_TO_USE)
        time.sleep(0.02)


def fmt_joints_deg(q_rad):
    q_deg = np.degrees(q_rad)
    return "  ".join(f"q{i+1}={q_deg[i]:+.1f}" for i in range(len(q_deg)))


def fmt_vec_deg(v_rad):
    v_deg = np.degrees(v_rad)
    return "  ".join(f"q{i+1}={v_deg[i]:+.1f}" for i in range(len(v_deg)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--duration", type=float, default=8.0,
                    help="How long to hold current joints after switching.")
    ap.add_argument("--publish-hz", type=float, default=100.0,
                    help="Joint goal refresh rate.")
    ap.add_argument("--switch-timeout", type=float, default=1.0)
    ap.add_argument("--drift-tol-deg", type=float, default=1.0,
                    help="Pass if max absolute joint drift stays below this.")
    ap.add_argument("--vel-warn-frac", type=float, default=WARN_VEL_FRAC,
                    help="Warn if any joint speed exceeds this fraction of limit.")
    ap.add_argument("--no-switch-controller", action="store_true",
                    help="Write the joint goal but do not set active_controller.")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    r = redis.Redis(host=args.host, port=6379, decode_responses=True)
    try:
        r.ping()
    except Exception:
        sys.exit("ERROR: Redis not reachable")

    q_hold = get_vec(r, SENSOR_JOINTS, expected_len=7)
    if q_hold is None:
        sys.exit(f"ERROR: no {SENSOR_JOINTS} in Redis - is OpenSai running?")

    active_before = decode_redis_value(r.get(ACTIVE_CONTROLLER))
    print(f"active_controller before: {active_before!r}")
    print(f"joint_goal_key: {GOAL_JOINTS}")
    print("hold_joints deg: " + fmt_joints_deg(q_hold))

    # Critical ordering: seed the joint goal before switching to avoid stale-goal lurches.
    set_vec(r, GOAL_JOINTS, q_hold)
    print("seeded current joints as joint_controller goal before switch")

    if not args.no_switch_controller:
        ensure_joint_controller(r, timeout_s=args.switch_timeout)
        print(f"active_controller: requested {CONTROLLER_TO_USE!r}")
    else:
        print("active_controller: not changed (--no-switch-controller)")

    publish_period_s = 1.0 / max(1.0, args.publish_hz)
    sample_period_s = 0.02
    t0 = time.perf_counter()
    next_publish = t0
    next_sample = t0
    prev_q = None
    prev_t = None

    max_abs_drift = np.zeros(7)
    qdot_peak_degs = np.zeros(7)
    qdot_source = "finite-diff"
    last_q = q_hold.copy()

    while True:
        now = time.perf_counter()
        elapsed = now - t0
        if now >= next_publish:
            set_vec(r, GOAL_JOINTS, q_hold)
            next_publish = now + publish_period_s

        if now >= next_sample:
            q = get_vec(r, SENSOR_JOINTS, expected_len=7)
            if q is not None:
                last_q = q
                drift = q - q_hold
                max_abs_drift = np.maximum(max_abs_drift, np.abs(drift))

                dq_sensor = get_vec(r, SENSOR_JOINT_VELS, expected_len=7)
                if dq_sensor is not None:
                    qdot_peak_degs = np.maximum(qdot_peak_degs, np.abs(np.degrees(dq_sensor)))
                    qdot_source = "sensor"
                elif prev_q is not None and prev_t is not None:
                    dt = now - prev_t
                    if dt > 1e-6:
                        qdot_peak_degs = np.maximum(
                            qdot_peak_degs, np.abs(np.degrees((q - prev_q) / dt))
                        )
                    qdot_source = "finite-diff"
                prev_q = q
                prev_t = now

                if args.verbose:
                    drift_deg = np.degrees(drift)
                    print(
                        f"  t={elapsed:5.2f}s  max_drift={np.max(np.abs(drift_deg)):.3f}deg  "
                        f"qdot_max={np.max(qdot_peak_degs):.1f}deg/s",
                        end="\r",
                    )
            next_sample = now + sample_period_s

        if elapsed >= args.duration:
            if args.verbose:
                print()
            break
        time.sleep(0.005)

    active_after = decode_redis_value(r.get(ACTIVE_CONTROLLER))
    final_drift_deg = np.degrees(last_q - q_hold)
    max_drift_deg = np.degrees(max_abs_drift)
    qdot_max_i = int(np.argmax(qdot_peak_degs))
    qdot_max = float(qdot_peak_degs[qdot_max_i])
    qdot_lim = float(JOINT_VEL_LIMIT_DEGS[qdot_max_i])
    vel_warn = qdot_max > args.vel_warn_frac * qdot_lim
    drift_pass = float(np.max(max_drift_deg)) <= args.drift_tol_deg
    status = "PASS" if drift_pass and not vel_warn else "WARN"

    sep = "-" * 72
    print(sep)
    print(f"STEP J0 JOINT HOLD    STATUS: {status}")
    print(f"  active_controller after: {active_after!r}")
    print(f"  duration:     {args.duration:.1f} s")
    print("  hold_joints:  " + fmt_joints_deg(q_hold))
    print("  final_joints: " + fmt_joints_deg(last_q))
    print("  final_drift: " + "  ".join(f"q{i+1}={final_drift_deg[i]:+.3f}" for i in range(7)) + " deg")
    print("  max_drift:   " + "  ".join(f"q{i+1}={max_drift_deg[i]:.3f}" for i in range(7)) + " deg")
    print(f"  drift_gate:  max <= {args.drift_tol_deg:.2f} deg")
    print("  qdot_peak:   " + "  ".join(f"q{i+1}={qdot_peak_degs[i]:.1f}" for i in range(7)) + f" deg/s ({qdot_source})")
    print(
        f"  qdot_max:    {qdot_max:.1f} deg/s @ q{qdot_max_i+1} "
        f"({100*qdot_max/qdot_lim:.0f}% of {qdot_lim:.0f} limit)"
        + ("  <-- WARN" if vel_warn else "")
    )
    print(sep)

    if status != "PASS":
        print("J0 did not look boring. Do not continue to joint nudges/IK until this is understood.")


if __name__ == "__main__":
    main()
