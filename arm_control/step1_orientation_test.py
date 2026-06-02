#!/usr/bin/env python3
"""
Step 1: Paddle-normal orientation micro-test.

No OptiTrack. This is the next gate after step0_controller_test.py:
use the same pre-ready pattern as Step 0, keep the OpenSai cartesian_controller active,
and sweep small paddle orientation changes around the verified pickleball
reference orientation.

Run from OpenSai root:
    python sports_bot/arm_control/step1_orientation_test.py --pre-ready --sweep --axis yaw --deg 5
    python sports_bot/arm_control/step1_orientation_test.py --pre-ready --sweep --axis yaw --deg 5 10
    python sports_bot/arm_control/step1_orientation_test.py --pre-ready --sweep

All positions are in arm BASE FRAME (A), meters. Orientation commands are
relative to the measured startup orientation by default:
    rx: roll around the face-normal / handle sway
    ry: pitch the paddle face up/down
    rz: yaw the paddle face left/right

Pass --reference pickleball-ref only when you explicitly want to snap to the
verified pickleball reference orientation first. That can be a large motion.

Safety intent:
    - Always writes paired goal_position and goal_orientation.
    - Refreshes the paired goal at 100 Hz while monitoring, matching Step 0.
    - Does not support position-only mode.
    - Defaults to relaxed gates: 15 mm position, 5 deg angle.
"""
import argparse
import json
import math
import sys
import time

import numpy as np
import redis

GOAL_POS = "opensai::controllers::FrankaRobot::cartesian_controller::cartesian_task::goal_position"
GOAL_ORI = "opensai::controllers::FrankaRobot::cartesian_controller::cartesian_task::goal_orientation"
CURR_POS = "opensai::controllers::FrankaRobot::cartesian_controller::cartesian_task::current_position"
CURR_ORI = "opensai::controllers::FrankaRobot::cartesian_controller::cartesian_task::current_orientation"
JOINTS = "opensai::sensors::FrankaRobot::joint_positions"
ACTIVE_CONTROLLER = "opensai::controllers::FrankaRobot::active_controller_name"
CONTROLLER_TO_USE = "cartesian_controller"

READY_POS = np.array([0.55, 0.00, 0.50])

# face +X opponent, handle -Z world -- verified 2026-05-26
_S2 = math.sqrt(0.5)
R_W_E_REF = np.array([
    [_S2,  _S2,  0.0],
    [_S2, -_S2,  0.0],
    [0.0,  0.0, -1.0],
])
FACE_NORMAL_E = np.array([_S2, _S2, 0.0])
HANDLE_AXIS_E = np.array([0.0, 0.0, 1.0])
DEFAULT_TCP_OFFSET_M = 0.35

JOINT_VEL_LIMIT_DEGS = np.degrees([2.175, 2.175, 2.175, 2.175, 2.610, 2.610, 2.610])
WARN_VEL_FRAC = 0.70


def decode_redis_value(value):
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def get(r, key):
    v = r.get(key)
    return np.array(json.loads(v)) if v else None


def ensure_cartesian_controller(r, timeout_s=1.0):
    t0 = time.monotonic()
    while True:
        active = decode_redis_value(r.get(ACTIVE_CONTROLLER))
        if active == CONTROLLER_TO_USE:
            return True
        if time.monotonic() - t0 > timeout_s:
            print(
                f"WARNING: could not switch active_controller from {active!r} "
                f"to {CONTROLLER_TO_USE!r} within {timeout_s:.1f}s; publishing goals anyway",
                file=sys.stderr,
            )
            return False
        r.set(ACTIVE_CONTROLLER, CONTROLLER_TO_USE)
        time.sleep(0.02)


def Rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def Ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def Rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def orientation_from_rpy_deg(rx_deg, ry_deg, rz_deg, base_ori):
    rx, ry, rz = [math.radians(v) for v in (rx_deg, ry_deg, rz_deg)]
    return Rz(rz) @ Ry(ry) @ Rx(rx) @ base_ori


def rot_err_deg(R_curr, R_goal):
    cos_a = float(np.clip((np.trace(R_goal @ R_curr.T) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_a)))


def fmt_vec(v):
    return f"[{v[0]:+.2f},{v[1]:+.2f},{v[2]:+.2f}]"


def ready_target_for_controlled_point(base_ori, ready_mode, tcp_offset_m):
    if ready_mode == "sweetspot":
        return READY_POS.copy()
    if ready_mode == "flange-from-sweetspot":
        return READY_POS - base_ori @ np.array([0.0, 0.0, tcp_offset_m])
    raise ValueError(f"unknown ready_mode: {ready_mode}")


def describe_goal_orientation(goal_ori, rpy_deg):
    face_A = goal_ori @ FACE_NORMAL_E
    handle_A = goal_ori @ HANDLE_AXIS_E
    delta_deg = float(np.linalg.norm(rpy_deg))
    arc_mm = DEFAULT_TCP_OFFSET_M * math.sin(math.radians(delta_deg)) * 1000.0
    print(
        f"  goal_face_A={fmt_vec(face_A)}  goal_handle_A={fmt_vec(handle_A)}  "
        f"tcp_arc_hint~{arc_mm:.0f}mm for {delta_deg:.1f}deg"
    )


def publish_cartesian_goal(r, goal_pos, goal_ori, switch_controller=True):
    if switch_controller:
        ensure_cartesian_controller(r)
    pipe = r.pipeline(transaction=False)
    pipe.set(GOAL_POS, json.dumps(np.asarray(goal_pos, dtype=float).reshape(3).tolist()))
    pipe.set(GOAL_ORI, json.dumps(np.asarray(goal_ori, dtype=float).reshape(3, 3).tolist()))
    pipe.execute()


def wait_converge(r, goal_pos, goal_ori, timeout_s, pos_tol_m, ang_tol_deg,
                  publish_fn, publish_period_s, verbose_progress=False):
    t0 = time.perf_counter()
    settled_since = None
    prev_joints = None
    prev_t = None
    qdot_peak = np.zeros(7)
    final_pos = None
    final_ori = None
    final_joints = None
    next_publish = t0

    while True:
        now = time.perf_counter()
        if now >= next_publish:
            publish_fn()
            next_publish = now + publish_period_s
        elapsed = now - t0

        curr = get(r, CURR_POS)
        curr_ori = get(r, CURR_ORI)
        curr_joints = get(r, JOINTS)
        if curr is None:
            time.sleep(0.05)
            continue

        final_pos = curr
        final_ori = curr_ori
        final_joints = curr_joints

        pos_err = float(np.linalg.norm(curr - goal_pos))
        ang_err = math.inf
        if curr_ori is not None:
            ang_err = rot_err_deg(curr_ori.reshape(3, 3), goal_ori)

        if curr_joints is not None and prev_joints is not None and prev_t is not None:
            dt = now - prev_t
            if dt > 0.01:
                qdot_peak = np.maximum(qdot_peak, np.abs((curr_joints - prev_joints) / dt))
        prev_joints = curr_joints
        prev_t = now

        if verbose_progress:
            print(
                f"  t={elapsed:5.2f}s  pos={pos_err*1000:6.1f}mm  "
                f"ang={ang_err:5.1f}deg  curr=[{curr[0]:.3f},{curr[1]:.3f},{curr[2]:.3f}]",
                end="\r",
            )

        in_tol = pos_err < pos_tol_m and ang_err < ang_tol_deg
        if in_tol:
            if settled_since is None:
                settled_since = now
            elif now - settled_since >= 0.5:
                if verbose_progress:
                    print()
                return "PASS", elapsed, final_pos, final_ori, np.degrees(qdot_peak), final_joints
        else:
            settled_since = None

        if elapsed >= timeout_s:
            if verbose_progress:
                print()
            return f"TIMEOUT({timeout_s:.0f}s)", elapsed, final_pos, final_ori, np.degrees(qdot_peak), final_joints

        time.sleep(0.05)


def print_result(label, status, goal_pos, rpy_deg, goal_ori, final_pos,
                 final_ori, t_settle, qdot_peak_degs, final_joints):
    sep = "-" * 72
    print(sep)
    print(f"TEST: {label}    STATUS: {status}")
    print(f"  goal_rpy:   rx={rpy_deg[0]:+.1f} deg  ry={rpy_deg[1]:+.1f} deg  rz={rpy_deg[2]:+.1f} deg")
    print(f"  goal_pos:   [{goal_pos[0]:.4f}, {goal_pos[1]:.4f}, {goal_pos[2]:.4f}]  (arm frame, m)")
    if final_pos is not None:
        delta = goal_pos - final_pos
        err_mm = float(np.linalg.norm(delta) * 1000.0)
        print(f"  curr_pos:   [{final_pos[0]:.4f}, {final_pos[1]:.4f}, {final_pos[2]:.4f}]")
        print(f"  pos_err:    {err_mm:.1f} mm")
        print(f"  delta_xyz:  [{delta[0]*1000:.1f}, {delta[1]*1000:.1f}, {delta[2]*1000:.1f}] mm")
    if final_ori is not None:
        print(f"  ang_err:    {rot_err_deg(final_ori.reshape(3, 3), goal_ori):.1f} deg")
    else:
        print("  ang_err:    n/a (current_orientation missing)")
    print(f"  t_settle:   {t_settle:.2f} s")
    if final_joints is not None:
        jdeg = np.degrees(final_joints)
        print("  joints deg: " + "  ".join(f"q{i+1}={jdeg[i]:.1f}" for i in range(len(jdeg))))
    if qdot_peak_degs is not None:
        max_i = int(np.argmax(qdot_peak_degs))
        max_v = float(qdot_peak_degs[max_i])
        lim = float(JOINT_VEL_LIMIT_DEGS[max_i])
        flag = "  <-- NEAR LIMIT" if max_v > WARN_VEL_FRAC * lim else ""
        print("  qdot_peak:  " + "  ".join(f"q{i+1}={qdot_peak_degs[i]:.0f}" for i in range(len(qdot_peak_degs))) + "  deg/s")
        print(f"  qdot_max:   {max_v:.0f} deg/s @ q{max_i+1}  ({100*max_v/lim:.0f}% of {lim:.0f} limit){flag}")
    print(sep)


def run_test(r, label, goal_pos, rpy_deg, base_ori, timeout_s, pos_tol_m,
             ang_tol_deg, publish_hz, switch_controller, verbose_progress):
    goal_ori = orientation_from_rpy_deg(*rpy_deg, base_ori=base_ori)
    publish_period_s = 1.0 / publish_hz

    def publish_fn():
        publish_cartesian_goal(r, goal_pos, goal_ori, switch_controller=switch_controller)

    print(
        f"\n[running] {label}  pos=[{goal_pos[0]:.3f},{goal_pos[1]:.3f},{goal_pos[2]:.3f}]  "
        f"rpy=[{rpy_deg[0]:+.1f},{rpy_deg[1]:+.1f},{rpy_deg[2]:+.1f}]"
    )
    describe_goal_orientation(goal_ori, rpy_deg)
    publish_fn()
    status, t_settle, final_pos, final_ori, qdot_peak_degs, final_joints = wait_converge(
        r, goal_pos, goal_ori, timeout_s, pos_tol_m, ang_tol_deg,
        publish_fn=publish_fn, publish_period_s=publish_period_s,
        verbose_progress=verbose_progress,
    )
    print_result(label, status, goal_pos, rpy_deg, goal_ori, final_pos,
                 final_ori, t_settle, qdot_peak_degs, final_joints)

    err_mm = float(np.linalg.norm(final_pos - goal_pos) * 1000.0) if final_pos is not None else math.inf
    ang = rot_err_deg(final_ori.reshape(3, 3), goal_ori) if final_ori is not None else math.inf
    qmax = float(np.max(qdot_peak_degs)) if qdot_peak_degs is not None else 0.0
    return status.startswith("PASS"), err_mm, ang, t_settle, qmax


def build_sweep(axis, degrees):
    tests = [(np.array([0.0, 0.0, 0.0]), "ref-0")]
    axes = ["pitch", "yaw"] if axis == "both" else [axis]
    for ax in axes:
        for mag in degrees:
            for sign in (1.0, -1.0):
                rpy = np.zeros(3)
                if ax == "roll":
                    rpy[0] = sign * mag
                elif ax == "pitch":
                    rpy[1] = sign * mag
                elif ax == "yaw":
                    rpy[2] = sign * mag
                tests.append((rpy, f"{ax}-{sign * mag:+.0f}deg"))
        tests.append((np.array([0.0, 0.0, 0.0]), f"{ax}-back-to-ref"))
    return tests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true",
                    help="Run a repeatable orientation sweep around the held position")
    ap.add_argument("--ori", nargs=3, type=float, metavar=("RX", "RY", "RZ"),
                    help="Single orientation relative to reference, degrees")
    ap.add_argument("--axis", choices=["pitch", "yaw", "roll", "both"], default="both",
                    help="Axis to sweep. Default intentionally excludes roll.")
    ap.add_argument("--deg", nargs="+", type=float, default=[5.0, 10.0],
                    help="Positive magnitudes to test for each swept axis")
    ap.add_argument("--goal", nargs=3, type=float, metavar=("X", "Y", "Z"),
                    help="Held sweet-spot position in arm frame. Default current_position; with --pre-ready, READY_POS is used only as the staging target.")
    ap.add_argument("--pre-ready", action="store_true",
                    help="First command READY_POS, then refresh to the measured post-ready pose like Step 0")
    ap.add_argument("--ready-mode", choices=["sweetspot", "flange-from-sweetspot"],
                    default="sweetspot",
                    help="Use sweetspot for normal 0.35 compliantFrame XML. Use flange-from-sweetspot with arm_test_picklebot_no_tcp.xml.")
    ap.add_argument("--tcp-offset-m", type=float, default=DEFAULT_TCP_OFFSET_M,
                    help="Physical flange-to-sweet-spot offset used for flange-from-sweetspot ready target.")
    ap.add_argument("--max-pre-ready-z", type=float, default=0.70,
                    help="Refuse --pre-ready targets above this arm-frame Z unless --allow-high-pre-ready is passed.")
    ap.add_argument("--allow-high-pre-ready", action="store_true",
                    help="Allow high --pre-ready targets. Use only with deliberate supervision.")
    ap.add_argument("--reference", choices=["current", "pickleball-ref"], default="current",
                    help="Base orientation for rpy commands. current is safer; pickleball-ref can be a large snap.")
    ap.add_argument("--timeout", type=float, default=8.0)
    ap.add_argument("--pos-tol-mm", type=float, default=15.0)
    ap.add_argument("--ang-tol-deg", type=float, default=5.0)
    ap.add_argument("--publish-hz", type=float, default=100.0)
    ap.add_argument("--no-switch-controller", action="store_true")
    ap.add_argument("--verbose-progress", action="store_true")
    ap.add_argument("--pause-between", action="store_true",
                    help="Wait for Enter after each test so you can visually inspect the held pose.")
    ap.add_argument("--host", default="localhost")
    args = ap.parse_args()

    r = redis.Redis(host=args.host, port=6379, decode_responses=True)
    try:
        r.ping()
    except Exception:
        sys.exit("ERROR: Redis not reachable")

    curr = get(r, CURR_POS)
    if curr is None:
        sys.exit("ERROR: no current_position in Redis - is OpenSai running with arm_test_picklebot.xml?")
    curr_ori = get(r, CURR_ORI)
    if curr_ori is None:
        sys.exit("ERROR: no current_orientation in Redis - cannot run paired orientation test safely")

    joints = get(r, JOINTS)
    print(f"start_pos:    [{curr[0]:.4f}, {curr[1]:.4f}, {curr[2]:.4f}]")
    if joints is not None:
        jdeg = np.degrees(joints)
        print("start_joints: " + "  ".join(f"q{i+1}={jdeg[i]:.1f}" for i in range(len(jdeg))))
    base_ori = curr_ori.reshape(3, 3) if args.reference == "current" else R_W_E_REF.copy()
    print(f"orientation_reference: {args.reference}")
    if args.reference == "pickleball-ref":
        start_to_ref = rot_err_deg(curr_ori.reshape(3, 3), R_W_E_REF)
        print(f"start_to_pickleball_ref_ang: {start_to_ref:.1f} deg")
        if start_to_ref > 15.0:
            print("WARNING: pickleball-ref is more than 15 deg from current orientation; expect a large reorientation.")

    switch_controller = not args.no_switch_controller
    if switch_controller:
        ensure_cartesian_controller(r)
        print(f"active_controller: ensured {CONTROLLER_TO_USE!r}")
    else:
        print("active_controller: not changed (--no-switch-controller)")

    goal_pos_user_supplied = args.goal is not None
    if goal_pos_user_supplied:
        goal_pos = np.array(args.goal, dtype=float)
        goal_source = "--goal"
    else:
        goal_pos = curr.copy()
        goal_source = "current_position"
    pos_tol_m = args.pos_tol_mm / 1000.0
    publish_hz = max(1.0, args.publish_hz)
    print(f"held_pos:     [{goal_pos[0]:.4f}, {goal_pos[1]:.4f}, {goal_pos[2]:.4f}]  source={goal_source}")
    pre_ready_target = ready_target_for_controlled_point(base_ori, args.ready_mode, args.tcp_offset_m)
    if args.pre_ready:
        print(f"pre_ready_target: [{pre_ready_target[0]:.4f}, {pre_ready_target[1]:.4f}, {pre_ready_target[2]:.4f}]  mode={args.ready_mode}")
        if pre_ready_target[2] > args.max_pre_ready_z and not args.allow_high_pre_ready:
            sys.exit(
                f"ERROR: pre_ready_target z={pre_ready_target[2]:.3f} m exceeds --max-pre-ready-z "
                f"{args.max_pre_ready_z:.3f} m. For no-TCP diagnostics, omit --pre-ready to test "
                "around the current flange pose, or pass an explicit low --goal. "
                "Use --allow-high-pre-ready only if you deliberately want the flange high."
            )
    print(f"tolerances:   pos<{args.pos_tol_mm:.1f} mm  ang<{args.ang_tol_deg:.1f} deg")
    print(f"goal_publish: refreshing paired pos+ori at {publish_hz:.1f} Hz")

    results = []
    if args.pre_ready:
        ok, err, ang, t, qmax = run_test(
            r, "pre-ready-hold-orientation", pre_ready_target, np.array([0.0, 0.0, 0.0]), base_ori,
            args.timeout, pos_tol_m, args.ang_tol_deg, publish_hz,
            switch_controller, args.verbose_progress,
        )
        results.append(("pre-ready-hold-orientation", ok, err, ang, t, qmax))
        time.sleep(1.0)
        refreshed_pos = get(r, CURR_POS)
        if refreshed_pos is not None and not goal_pos_user_supplied:
            goal_pos = refreshed_pos.copy()
            goal_source = "measured post-ready pose"
            print(f"held_pos: refreshed to [{goal_pos[0]:.4f}, {goal_pos[1]:.4f}, {goal_pos[2]:.4f}]  source={goal_source}")
        if args.reference == "current":
            refreshed_ori = get(r, CURR_ORI)
            if refreshed_ori is not None:
                base_ori = refreshed_ori.reshape(3, 3)
                print("orientation_reference: refreshed to measured post-ready orientation")

    if args.ori is not None:
        tests = [(np.array(args.ori, dtype=float), "single-ori")]
    elif args.sweep:
        tests = build_sweep(args.axis, args.deg)
    else:
        tests = [(np.array([0.0, 0.0, 0.0]), "hold-ref")]

    for rpy, label in tests:
        ok, err, ang, t, qmax = run_test(
            r, label, goal_pos, rpy, base_ori, args.timeout, pos_tol_m,
            args.ang_tol_deg, publish_hz, switch_controller,
            args.verbose_progress,
        )
        results.append((label, ok, err, ang, t, qmax))
        if args.pause_between:
            input("Press Enter for next test...")
        else:
            time.sleep(1.0)

    sep = "-" * 72
    print(f"\n{sep}")
    print("STEP 1 ORIENTATION SUMMARY")
    print(f"  {'label':<20} {'status':<8} {'err_mm':>8} {'ang_deg':>8} {'time_s':>8} {'qdot_max':>10}")
    for label, ok, err, ang, t, qmax in results:
        print(
            f"  {label:<20} {'PASS' if ok else 'FAIL':<8} "
            f"{err:>8.1f} {ang:>8.1f} {t:>8.2f} {qmax:>10.0f}"
        )
    print(sep)


if __name__ == "__main__":
    main()
