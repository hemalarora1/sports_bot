#!/usr/bin/env python3
"""
Step J1: Tiny joint-space nudge test.

Prereq: Step J0 joint hold passes.

This script moves one or more joints by a small angle using a smooth joint-goal
trajectory through OpenSai's joint_controller. It seeds the current joint goal
before switching controllers, then streams intermediate joint goals.

Run from OpenSai root:
    python sports_bot/arm_control/stepj1_joint_nudge.py --joint 1 --deg 2 --pause-between
    python sports_bot/arm_control/stepj1_joint_nudge.py --sweep --pause-between

Default sweep is deliberately gentle: q1, q3, q7 at +/-2 deg, returning to the
measured start pose after each nudge.
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
COMMAND_TORQUES = f"opensai::commands::{ROBOT_NAME}::control_torques"
SENSED_TORQUES = f"opensai::sensors::{ROBOT_NAME}::joint_torques"
SENT_TORQUES = f"opensai::redis_driver::{ROBOT_NAME}::safety_controller::sent_torques"
SAFETY_TORQUES = f"opensai::redis_driver::{ROBOT_NAME}::safety_controller::safety_torques"

CURR_POS = "opensai::controllers::FrankaRobot::cartesian_controller::cartesian_task::current_position"

JOINT_VEL_LIMIT_DEGS = np.degrees([2.175, 2.175, 2.175, 2.175, 2.610, 2.610, 2.610])
JOINT_LIMITS_RAD = np.array([
    [-2.8973,  2.8973],
    [-1.7628,  1.7628],
    [-2.8973,  2.8973],
    [-3.0718, -0.0698],
    [-2.8973,  2.8973],
    [-0.0175,  3.7525],
    [-2.8973,  2.8973],
])

DEFAULT_SWEEP_JOINTS = [1, 3, 7]
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


def hold_current_joints(r):
    q_cur = get_vec(r, SENSOR_JOINTS, expected_len=7)
    if q_cur is not None:
        set_vec(r, GOAL_JOINTS, q_cur)
        print("\n[J1] interrupted — holding current measured joints")
    return q_cur


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


def fmt_torque_vec(tau):
    return "  ".join(f"q{i+1}={tau[i]:+.2f}" for i in range(len(tau)))


def smoothstep(s):
    s = float(np.clip(s, 0.0, 1.0))
    return s * s * (3.0 - 2.0 * s)


def check_joint_limits(q_target):
    bad = []
    for i, q in enumerate(q_target):
        lo, hi = JOINT_LIMITS_RAD[i]
        if q < lo or q > hi:
            bad.append((i + 1, math.degrees(q), math.degrees(lo), math.degrees(hi)))
    return bad


def sample_state(r, q_goal, q_start, qdot_peak_degs, prev_q, prev_t):
    now = time.perf_counter()
    q = get_vec(r, SENSOR_JOINTS, expected_len=7)
    if q is None:
        return None, qdot_peak_degs, prev_q, prev_t

    dq_sensor = get_vec(r, SENSOR_JOINT_VELS, expected_len=7)
    if dq_sensor is not None:
        qdot_peak_degs = np.maximum(qdot_peak_degs, np.abs(np.degrees(dq_sensor)))
    elif prev_q is not None and prev_t is not None:
        dt = now - prev_t
        if dt > 1e-6:
            qdot_peak_degs = np.maximum(qdot_peak_degs, np.abs(np.degrees((q - prev_q) / dt)))

    info = {
        "t": now,
        "q": q,
        "err_deg": np.degrees(q_goal - q),
        "drift_from_start_deg": np.degrees(q - q_start),
    }
    return info, qdot_peak_degs, q.copy(), now


def run_segment(r, label, q_start, q_target, move_s, hold_s, publish_hz,
                verbose=False):
    qdot_peak_degs = np.zeros(7)
    torque_peak = {
        "cmd": np.zeros(7),
        "sent": np.zeros(7),
        "safety": np.zeros(7),
        "sensed": np.zeros(7),
    }
    torque_seen = {key: False for key in torque_peak}
    torque_latest = {key: None for key in torque_peak}
    prev_q = None
    prev_t = None
    publish_period_s = 1.0 / publish_hz
    sample_period_s = 0.02
    t0 = time.perf_counter()
    next_publish = t0
    next_sample = t0
    final_info = None
    cart_start = get_vec(r, CURR_POS, expected_len=3)
    cart_final = None

    print(f"\n[running] {label}")
    print("  start: " + fmt_joints_deg(q_start))
    print("  target:" + fmt_joints_deg(q_target))

    while True:
        now = time.perf_counter()
        elapsed = now - t0
        total_s = move_s + hold_s

        if elapsed <= move_s:
            alpha = smoothstep(elapsed / max(move_s, 1e-6))
            q_goal = q_start + alpha * (q_target - q_start)
        else:
            q_goal = q_target

        if now >= next_publish:
            set_vec(r, GOAL_JOINTS, q_goal)
            next_publish = now + publish_period_s

        if now >= next_sample:
            final_info, qdot_peak_degs, prev_q, prev_t = sample_state(
                r, q_goal, q_start, qdot_peak_degs, prev_q, prev_t
            )
            torque_reads = {
                "cmd": get_vec(r, COMMAND_TORQUES, expected_len=7),
                "sent": get_vec(r, SENT_TORQUES, expected_len=7),
                "safety": get_vec(r, SAFETY_TORQUES, expected_len=7),
                "sensed": get_vec(r, SENSED_TORQUES, expected_len=7),
            }
            for key, tau in torque_reads.items():
                if tau is not None:
                    torque_seen[key] = True
                    torque_latest[key] = tau
                    torque_peak[key] = np.maximum(torque_peak[key], np.abs(tau))
            if verbose and final_info is not None:
                err_max = float(np.max(np.abs(final_info["err_deg"])))
                qdot_max = float(np.max(qdot_peak_degs))
                print(
                    f"  t={elapsed:5.2f}s  goal_err_max={err_max:.3f}deg  "
                    f"qdot_max={qdot_max:.1f}deg/s",
                    end="\r",
                )
            next_sample = now + sample_period_s

        if elapsed >= total_s:
            if verbose:
                print()
            break
        time.sleep(0.005)

    cart_final = get_vec(r, CURR_POS, expected_len=3)
    q_final = final_info["q"] if final_info is not None else get_vec(r, SENSOR_JOINTS, expected_len=7)
    if q_final is None:
        q_final = q_target.copy()
    final_err_deg = np.degrees(q_target - q_final)
    moved_deg = np.degrees(q_final - q_start)
    qdot_max_i = int(np.argmax(qdot_peak_degs))
    qdot_max = float(qdot_peak_degs[qdot_max_i])
    qdot_lim = float(JOINT_VEL_LIMIT_DEGS[qdot_max_i])
    vel_warn = qdot_max > WARN_VEL_FRAC * qdot_lim

    cart_delta_mm = None
    if cart_start is not None and cart_final is not None:
        cart_delta_mm = (cart_final - cart_start) * 1000.0

    sep = "-" * 72
    print(sep)
    print(f"SEGMENT: {label}")
    print("  final:      " + fmt_joints_deg(q_final))
    print("  moved_deg:  " + "  ".join(f"q{i+1}={moved_deg[i]:+.2f}" for i in range(7)))
    print("  goal_err:   " + "  ".join(f"q{i+1}={final_err_deg[i]:+.3f}" for i in range(7)) + " deg")
    if cart_delta_mm is not None:
        print(
            f"  cart_delta: [{cart_delta_mm[0]:+.1f}, {cart_delta_mm[1]:+.1f}, "
            f"{cart_delta_mm[2]:+.1f}] mm  (if current_position is being updated)"
        )
    print("  qdot_peak:  " + "  ".join(f"q{i+1}={qdot_peak_degs[i]:.1f}" for i in range(7)) + " deg/s")
    print(
        f"  qdot_max:   {qdot_max:.1f} deg/s @ q{qdot_max_i+1} "
        f"({100*qdot_max/qdot_lim:.0f}% of {qdot_lim:.0f} limit)"
        + ("  <-- WARN" if vel_warn else "")
    )
    for torque_name in ("cmd", "sent", "safety", "sensed"):
        if torque_seen[torque_name]:
            latest = torque_latest[torque_name]
            print(f"  tau_{torque_name}_latest: " + fmt_torque_vec(latest))
            print(f"  tau_{torque_name}_peak:   " + fmt_torque_vec(torque_peak[torque_name]))
    print(sep)

    return {
        "label": label,
        "goal_err_max_deg": float(np.max(np.abs(final_err_deg))),
        "qdot_max_deg_s": qdot_max,
        "vel_warn": vel_warn,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--joint", type=int, choices=range(1, 8),
                    help="Single 1-indexed joint to nudge.")
    ap.add_argument("--joints", nargs="+", type=int, choices=range(1, 8),
                    help="Joints to sweep. Default with --sweep: q1 q3 q7.")
    ap.add_argument("--deg", type=float, default=2.0,
                    help="Nudge magnitude in degrees.")
    ap.add_argument("--direction", choices=["both", "plus", "minus"], default="both")
    ap.add_argument("--sweep", action="store_true",
                    help="Run +/- nudges for multiple joints.")
    ap.add_argument("--move-s", type=float, default=1.5,
                    help="Smooth interpolation duration for each nudge.")
    ap.add_argument("--hold-s", type=float, default=1.0,
                    help="Hold time at the segment target.")
    ap.add_argument("--publish-hz", type=float, default=100.0)
    ap.add_argument("--switch-timeout", type=float, default=1.0)
    ap.add_argument("--max-deg", type=float, default=5.0,
                    help="Refuse larger nudges unless --force is passed.")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--pause-between", action="store_true")
    ap.add_argument("--no-return", action="store_true",
                    help="Do not return to the measured start after each nudge.")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if abs(args.deg) > args.max_deg and not args.force:
        sys.exit(
            f"ERROR: requested {args.deg:.1f} deg exceeds --max-deg {args.max_deg:.1f}. "
            "Use --force only after smaller nudges pass."
        )

    r = redis.Redis(host=args.host, port=6379, decode_responses=True)
    try:
        r.ping()
    except Exception:
        sys.exit("ERROR: Redis not reachable")

    q_start = get_vec(r, SENSOR_JOINTS, expected_len=7)
    if q_start is None:
        sys.exit(f"ERROR: no {SENSOR_JOINTS} in Redis - is OpenSai running?")

    active_before = decode_redis_value(r.get(ACTIVE_CONTROLLER))
    print(f"active_controller before: {active_before!r}")
    print(f"joint_goal_key: {GOAL_JOINTS}")
    print("start_joints deg: " + fmt_joints_deg(q_start))

    set_vec(r, GOAL_JOINTS, q_start)
    print("seeded current joints as joint_controller goal before switch")
    ensure_joint_controller(r, timeout_s=args.switch_timeout)
    print(f"active_controller: requested {CONTROLLER_TO_USE!r}")

    if args.sweep:
        joints = args.joints or DEFAULT_SWEEP_JOINTS
    elif args.joint is not None:
        joints = [args.joint]
    else:
        joints = [1]

    signs = []
    if args.direction in ("both", "plus"):
        signs.append(1.0)
    if args.direction in ("both", "minus"):
        signs.append(-1.0)

    summaries = []
    current_home = q_start.copy()
    try:
        for j in joints:
            for sign in signs:
                delta = np.zeros(7)
                delta[j - 1] = math.radians(sign * abs(args.deg))
                q_target = current_home + delta
                bad_limits = check_joint_limits(q_target)
                if bad_limits:
                    for ji, q, lo, hi in bad_limits:
                        print(f"SKIP q{ji}: target {q:.1f} deg outside [{lo:.1f}, {hi:.1f}] deg")
                    continue
    
                label = f"q{j}_{'plus' if sign > 0 else 'minus'}_{abs(args.deg):.1f}deg"
                summaries.append(run_segment(
                    r, label, current_home, q_target,
                    move_s=args.move_s, hold_s=args.hold_s,
                    publish_hz=max(1.0, args.publish_hz),
                    verbose=args.verbose,
                ))
    
                if args.pause_between:
                    input("Press Enter to return/continue...")
    
                if not args.no_return:
                    summaries.append(run_segment(
                        r, f"return_after_{label}", q_target, current_home,
                        move_s=args.move_s, hold_s=args.hold_s,
                        publish_hz=max(1.0, args.publish_hz),
                        verbose=args.verbose,
                    ))
                    if args.pause_between:
                        input("Press Enter for next nudge...")
                else:
                    current_home = q_target.copy()
    
    except KeyboardInterrupt:
        hold_current_joints(r)
        return

    sep = "-" * 72
    print(f"\n{sep}")
    print("STEP J1 JOINT NUDGE SUMMARY")
    print(f"  {'label':<28} {'err_deg':>8} {'qdot_max':>10} {'status':>8}")
    for item in summaries:
        status = "WARN" if item["vel_warn"] else "OK"
        print(
            f"  {item['label']:<28} {item['goal_err_max_deg']:>8.3f} "
            f"{item['qdot_max_deg_s']:>10.1f} {status:>8}"
        )
    print(sep)


if __name__ == "__main__":
    main()
