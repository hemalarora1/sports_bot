#!/usr/bin/env python3
"""
Step 0: Bare controller convergence test.

No OptiTrack. Writes goal_position (and optionally goal_orientation) ONCE to Redis,
then measures whether the arm reaches it. Run this first — if this fails, don't
debug anything else in the stack.

Run from OpenSai root:
    python sports_bot/arm_control/step0_controller_test.py
    python sports_bot/arm_control/step0_controller_test.py --sweep
    python sports_bot/arm_control/step0_controller_test.py --goal 0.60 0.0 0.35
    python sports_bot/arm_control/step0_controller_test.py --goal 0.60 0.0 0.35 --with-ori

All positions in arm BASE FRAME (A), meters. current_position = sweet-spot (compliantFrame
xyz="0 0 0.35" in picklebot.xml). Pass: pos_err < 8 mm held for 0.5 s.

Output is structured for easy copy-paste to Claude for debugging.
"""
import argparse
import json
import sys
import time

import numpy as np
import redis

GOAL_POS = "opensai::controllers::FrankaRobot::cartesian_controller::cartesian_task::goal_position"
GOAL_ORI = "opensai::controllers::FrankaRobot::cartesian_controller::cartesian_task::goal_orientation"
CURR_POS = "opensai::controllers::FrankaRobot::cartesian_controller::cartesian_task::current_position"
CURR_ORI = "opensai::controllers::FrankaRobot::cartesian_controller::cartesian_task::current_orientation"
JOINTS   = "opensai::sensors::FrankaRobot::joint_positions"

# face +X opponent, handle -Z world — verified 2026-05-26
R_W_E_REF = np.array([
    [ 0.7071068,  0.7071068, 0.0],
    [ 0.7071068, -0.7071068, 0.0],
    [ 0.0,        0.0,      -1.0],
])

# Franka hardware velocity limits (rad/s → deg/s)
JOINT_VEL_LIMIT_DEGS = np.degrees([2.175, 2.175, 2.175, 2.175, 2.610, 2.610, 2.610])
WARN_VEL_FRAC = 0.70  # flag if qdot > 70% of limit

SWEEP_GOALS = [
    (np.array([0.62,  0.00, 0.32]), "center-low"),
    (np.array([0.60,  0.00, 0.36]), "center-mid"),
    (np.array([0.58,  0.10, 0.36]), "left-small"),
    (np.array([0.58, -0.10, 0.36]), "right-small"),
    (np.array([0.64,  0.00, 0.40]), "center-higher"),
    (np.array([0.62,  0.00, 0.32]), "back-to-start"),
]


def get(r, key):
    v = r.get(key)
    return np.array(json.loads(v)) if v else None


def rot_err_deg(R_curr, R_goal):
    cos_a = float(np.clip((np.trace(R_goal @ R_curr.T) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_a)))


def print_result_block(label, status, goal_pos, goal_ori, final_pos, final_ori,
                       t_settle, qdot_peak_degs, final_joints):
    sep = "━" * 64
    print(sep)
    print(f"TEST: {label}    STATUS: {status}")
    print(f"  goal_pos:   [{goal_pos[0]:.4f}, {goal_pos[1]:.4f}, {goal_pos[2]:.4f}]  (arm frame, m)")
    if final_pos is not None:
        delta = goal_pos - final_pos
        err_mm = np.linalg.norm(delta) * 1000
        print(f"  curr_pos:   [{final_pos[0]:.4f}, {final_pos[1]:.4f}, {final_pos[2]:.4f}]")
        print(f"  pos_err:    {err_mm:.1f} mm")
        print(f"  delta_xyz:  [{delta[0]*1000:.1f}, {delta[1]*1000:.1f}, {delta[2]*1000:.1f}] mm  (goal-curr, sign shows direction to move)")
    if goal_ori is not None and final_ori is not None:
        ang_err = rot_err_deg(final_ori.reshape(3, 3), goal_ori)
        print(f"  ang_err:    {ang_err:.1f}°")
    else:
        print(f"  ang_err:    n/a (no ori commanded)")
    print(f"  t_settle:   {t_settle:.2f} s")
    if final_joints is not None:
        jdeg = np.degrees(final_joints)
        print(f"  joints deg: " + "  ".join(f"q{i+1}={jdeg[i]:.1f}" for i in range(len(jdeg))))
    if qdot_peak_degs is not None:
        max_i = int(np.argmax(qdot_peak_degs))
        max_v = qdot_peak_degs[max_i]
        lim = JOINT_VEL_LIMIT_DEGS[max_i]
        pct = 100 * max_v / lim
        flag = "  ← NEAR LIMIT" if pct > WARN_VEL_FRAC * 100 else ""
        print(f"  qdot_peak:  " + "  ".join(f"q{i+1}={qdot_peak_degs[i]:.0f}" for i in range(len(qdot_peak_degs))) + "  deg/s")
        print(f"  qdot_max:   {max_v:.0f} deg/s @ q{max_i+1}  ({pct:.0f}% of {lim:.0f} limit){flag}")
    print(sep)


def wait_converge(r, goal_pos, goal_ori, timeout_s, pos_tol_m, ang_tol_deg, settle_s):
    t0 = time.perf_counter()
    settled_since = None
    prev_joints = None
    prev_t = None
    qdot_peak = np.zeros(7)
    final_pos = None
    final_ori = None
    final_joints = None

    while True:
        now = time.perf_counter()
        elapsed = now - t0

        curr = get(r, CURR_POS)
        curr_joints = get(r, JOINTS)

        if curr is None:
            time.sleep(0.05)
            continue

        final_pos = curr
        final_joints = curr_joints

        pos_err = float(np.linalg.norm(curr - goal_pos))
        ang_err = 0.0
        curr_ori_arr = None
        if goal_ori is not None:
            c_ori = get(r, CURR_ORI)
            if c_ori is not None:
                curr_ori_arr = c_ori
                final_ori = c_ori
                ang_err = rot_err_deg(c_ori.reshape(3, 3), goal_ori)

        # joint velocity estimate
        if curr_joints is not None and prev_joints is not None and prev_t is not None:
            dt = now - prev_t
            if dt > 0.01:
                qdot = np.abs((curr_joints - prev_joints) / dt)
                qdot_peak = np.maximum(qdot_peak, qdot)
        prev_joints = curr_joints
        prev_t = now

        in_tol = pos_err < pos_tol_m and ang_err < ang_tol_deg
        print(
            f"  t={elapsed:5.2f}s  pos={pos_err*1000:6.1f}mm  ang={ang_err:5.1f}°  "
            f"curr=[{curr[0]:.3f},{curr[1]:.3f},{curr[2]:.3f}]",
            end="\r",
        )

        if in_tol:
            if settled_since is None:
                settled_since = now
            elif now - settled_since >= settle_s:
                print()
                return "PASS", elapsed, final_pos, final_ori, np.degrees(qdot_peak), final_joints
        else:
            settled_since = None

        if elapsed >= timeout_s:
            print()
            return f"TIMEOUT({timeout_s:.0f}s)", elapsed, final_pos, final_ori, np.degrees(qdot_peak), final_joints

        time.sleep(0.05)


def run_test(r, goal_pos, label, goal_ori, timeout_s, pos_tol_m, ang_tol_deg):
    print(f"\n[running] {label}  goal=[{goal_pos[0]:.3f},{goal_pos[1]:.3f},{goal_pos[2]:.3f}]")
    r.set(GOAL_POS, json.dumps(goal_pos.tolist()))  # write ONCE
    if goal_ori is not None:
        r.set(GOAL_ORI, json.dumps(goal_ori.tolist()))

    status, t_settle, final_pos, final_ori, qdot_peak_degs, final_joints = wait_converge(
        r, goal_pos, goal_ori, timeout_s, pos_tol_m, ang_tol_deg, settle_s=0.5
    )
    print_result_block(label, status, goal_pos, goal_ori, final_pos, final_ori,
                       t_settle, qdot_peak_degs, final_joints)

    err_mm = np.linalg.norm(final_pos - goal_pos) * 1000 if final_pos is not None else -1
    return status.startswith("PASS"), err_mm, t_settle, qdot_peak_degs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--goal", nargs=3, type=float, metavar=("X", "Y", "Z"),
                    help="Single goal in arm base frame (m)")
    ap.add_argument("--sweep", action="store_true", help="Run 6-point grid test")
    ap.add_argument("--with-ori", action="store_true", help="Also command R_W_E_REF orientation")
    ap.add_argument("--timeout", type=float, default=8.0)
    ap.add_argument("--pos-tol-mm", type=float, default=8.0)
    ap.add_argument("--ang-tol-deg", type=float, default=3.0)
    ap.add_argument("--host", default="localhost")
    args = ap.parse_args()

    r = redis.Redis(host=args.host, port=6379, decode_responses=True)
    try:
        r.ping()
    except Exception:
        sys.exit("ERROR: Redis not reachable")

    curr = get(r, CURR_POS)
    if curr is None:
        sys.exit("ERROR: no current_position in Redis — is OpenSai running with picklebot.xml?")

    joints = get(r, JOINTS)
    print(f"start_pos:    [{curr[0]:.4f}, {curr[1]:.4f}, {curr[2]:.4f}]")
    if joints is not None:
        jdeg = np.degrees(joints)
        print(f"start_joints: " + "  ".join(f"q{i+1}={jdeg[i]:.1f}" for i in range(len(jdeg))))

    goal_ori = R_W_E_REF if args.with_ori else None
    pos_tol = args.pos_tol_mm / 1000.0

    if args.sweep:
        results = []
        for gp, label in SWEEP_GOALS:
            ok, err, t, qdot = run_test(r, gp, label, goal_ori, args.timeout, pos_tol, args.ang_tol_deg)
            results.append((label, ok, err, t, qdot))
            time.sleep(1.5)

        sep = "━" * 64
        print(f"\n{sep}")
        print(f"SWEEP SUMMARY  (xml: check which picklebot_*.xml you launched)")
        print(f"  {'label':<20} {'status':<12} {'err_mm':>7} {'time_s':>7} {'qdot_max':>12}")
        for label, ok, err, t, qdot in results:
            status = "PASS" if ok else "FAIL"
            if qdot is not None:
                mi = int(np.argmax(qdot))
                qmax = f"{qdot[mi]:.0f}°/s@q{mi+1}"
                flag = " ←VEL" if qdot[mi] > WARN_VEL_FRAC * JOINT_VEL_LIMIT_DEGS[mi] else ""
            else:
                qmax, flag = "n/a", ""
            print(f"  {label:<20} {status:<12} {err:>7.1f} {t:>7.2f} {qmax:>12}{flag}")
        print(sep)
    else:
        goal_pos = np.array(args.goal) if args.goal else np.array([0.60, 0.00, 0.35])
        run_test(r, goal_pos, "single", goal_ori, args.timeout, pos_tol, args.ang_tol_deg)


if __name__ == "__main__":
    main()
