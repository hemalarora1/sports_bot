#!/usr/bin/env python3
"""
Step 1: Interactive arm-frame position + orientation commander.

No OptiTrack. Commands goal_position in arm BASE FRAME (A). Writes the goal ONCE per
command (not every tick) so OTG can complete its trajectory without being reset.
After each command, monitors convergence and prints a structured result block.

Run from OpenSai root:
    python sports_bot/arm_control/step1_arm_frame.py

Commands at the > prompt:
    X Y Z            absolute goal in arm base frame (m)
    r dX dY dZ       relative nudge from current goal
    ori              snap to reference orientation (face +X, handle -Z world)
    ori rx ry rz     orientation relative to reference, ZYX Euler degrees
    home             safe home: [0.40, 0.00, 0.60]
    s                show current state without moving
    q                quit (last goals stay in Redis)
"""
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

HOME = np.array([0.40, 0.00, 0.60])

# face +X opponent, handle -Z world — verified 2026-05-26
R_W_E_REF = np.array([
    [ 0.7071068,  0.7071068, 0.0],
    [ 0.7071068, -0.7071068, 0.0],
    [ 0.0,        0.0,      -1.0],
])

# Franka hardware velocity limits deg/s
JOINT_VEL_LIMIT_DEGS = np.degrees([2.175, 2.175, 2.175, 2.175, 2.610, 2.610, 2.610])
WARN_VEL_FRAC = 0.70

MONITOR_S   = 6.0
POLL_S      = 0.05
POS_TOL_M   = 0.008
ANG_TOL_DEG = 3.0
SETTLE_S    = 0.5


def get(r, key):
    v = r.get(key)
    return np.array(json.loads(v)) if v else None


def rot_err_deg(R_curr, R_goal):
    cos_a = float(np.clip((np.trace(R_goal @ R_curr.T) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_a)))


def euler_zyx_deg(rx, ry, rz, R_ref):
    rx, ry, rz = np.radians([rx, ry, rz])
    Rx = np.array([[1,0,0],[0,np.cos(rx),-np.sin(rx)],[0,np.sin(rx),np.cos(rx)]])
    Ry = np.array([[np.cos(ry),0,np.sin(ry)],[0,1,0],[-np.sin(ry),0,np.cos(ry)]])
    Rz = np.array([[np.cos(rz),-np.sin(rz),0],[np.sin(rz),np.cos(rz),0],[0,0,1]])
    return Rz @ Ry @ Rx @ R_ref


def print_result(cmd_str, status, goal_pos, goal_ori, final_pos, final_ori, t_elapsed, qdot_peak_degs, final_joints):
    sep = "━" * 64
    print(sep)
    print(f"CMD: {cmd_str}    STATUS: {status}")
    if final_pos is not None:
        delta = goal_pos - final_pos
        err_mm = np.linalg.norm(delta) * 1000
        print(f"  goal_pos:   [{goal_pos[0]:.4f}, {goal_pos[1]:.4f}, {goal_pos[2]:.4f}]  (arm frame, m)")
        print(f"  curr_pos:   [{final_pos[0]:.4f}, {final_pos[1]:.4f}, {final_pos[2]:.4f}]")
        print(f"  pos_err:    {err_mm:.1f} mm")
        print(f"  delta_xyz:  [{delta[0]*1000:.1f}, {delta[1]*1000:.1f}, {delta[2]*1000:.1f}] mm  (goal-curr)")
    if goal_ori is not None and final_ori is not None:
        ang_err = rot_err_deg(final_ori.reshape(3, 3), goal_ori)
        print(f"  ang_err:    {ang_err:.1f}°")
    elif goal_ori is None:
        print(f"  ang_err:    n/a (no ori commanded)")
    print(f"  t_elapsed:  {t_elapsed:.2f} s")
    if final_joints is not None:
        jdeg = np.degrees(final_joints)
        print(f"  joints deg: " + "  ".join(f"q{i+1}={jdeg[i]:.1f}" for i in range(len(jdeg))))
    if qdot_peak_degs is not None and np.any(qdot_peak_degs > 0):
        mi = int(np.argmax(qdot_peak_degs))
        mv = qdot_peak_degs[mi]
        lim = JOINT_VEL_LIMIT_DEGS[mi]
        flag = "  ← NEAR LIMIT" if mv > WARN_VEL_FRAC * lim else ""
        print(f"  qdot_peak:  " + "  ".join(f"q{i+1}={qdot_peak_degs[i]:.0f}" for i in range(len(qdot_peak_degs))) + "  deg/s")
        print(f"  qdot_max:   {mv:.0f} deg/s @ q{mi+1}  ({100*mv/lim:.0f}% of {lim:.0f} limit){flag}")
    print(sep)


def monitor(r, goal_pos, goal_ori, cmd_str):
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
            time.sleep(POLL_S)
            continue

        final_pos = curr
        final_joints = curr_joints

        pos_err = float(np.linalg.norm(curr - goal_pos))
        ang_err = 0.0
        if goal_ori is not None:
            c_ori = get(r, CURR_ORI)
            if c_ori is not None:
                final_ori = c_ori
                ang_err = rot_err_deg(c_ori.reshape(3, 3), goal_ori)

        if curr_joints is not None and prev_joints is not None and prev_t is not None:
            dt = now - prev_t
            if dt > 0.01:
                qdot_peak = np.maximum(qdot_peak, np.abs((curr_joints - prev_joints) / dt))
        prev_joints = curr_joints
        prev_t = now

        in_tol = pos_err < POS_TOL_M and ang_err < ANG_TOL_DEG
        print(
            f"  t={elapsed:.1f}s  pos={pos_err*1000:.1f}mm  ang={ang_err:.1f}°"
            f"  curr=[{curr[0]:.3f},{curr[1]:.3f},{curr[2]:.3f}]",
            end="\r",
        )

        if in_tol:
            if settled_since is None:
                settled_since = now
            elif now - settled_since >= SETTLE_S:
                print()
                print_result(cmd_str, "CONVERGED", goal_pos, goal_ori, final_pos, final_ori,
                             elapsed, np.degrees(qdot_peak), final_joints)
                return
        else:
            settled_since = None

        if elapsed >= MONITOR_S:
            print()
            print_result(cmd_str, f"TIMEOUT({MONITOR_S:.0f}s)", goal_pos, goal_ori, final_pos, final_ori,
                         elapsed, np.degrees(qdot_peak), final_joints)
            return

        time.sleep(POLL_S)


def send_goal(r, goal_pos, goal_ori, cmd_str):
    print(f"  → [{goal_pos[0]:.3f},{goal_pos[1]:.3f},{goal_pos[2]:.3f}]")
    r.set(GOAL_POS, json.dumps(goal_pos.tolist()))  # write ONCE
    if goal_ori is not None:
        r.set(GOAL_ORI, json.dumps(goal_ori.tolist()))
    monitor(r, goal_pos, goal_ori, cmd_str)


def show_state(r, goal_pos, goal_ori):
    curr = get(r, CURR_POS)
    joints = get(r, JOINTS)
    if curr is None:
        print("  (no current_position)")
        return
    err = np.linalg.norm(curr - goal_pos) * 1000
    delta = goal_pos - curr
    ang_err = 0.0
    if goal_ori is not None:
        c_ori = get(r, CURR_ORI)
        if c_ori is not None:
            ang_err = rot_err_deg(c_ori.reshape(3, 3), goal_ori)
    print(f"  curr:      [{curr[0]:.4f}, {curr[1]:.4f}, {curr[2]:.4f}]")
    print(f"  goal:      [{goal_pos[0]:.4f}, {goal_pos[1]:.4f}, {goal_pos[2]:.4f}]")
    print(f"  pos_err:   {err:.1f} mm  delta=[{delta[0]*1000:.1f},{delta[1]*1000:.1f},{delta[2]*1000:.1f}] mm")
    if goal_ori is not None:
        print(f"  ang_err:   {ang_err:.1f}°")
    if joints is not None:
        jdeg = np.degrees(joints)
        print(f"  joints deg: " + "  ".join(f"q{i+1}={jdeg[i]:.1f}" for i in range(len(jdeg))))


def main():
    r = redis.Redis(host="localhost", port=6379, decode_responses=True)
    try:
        r.ping()
    except Exception:
        sys.exit("ERROR: Redis not reachable")

    curr = get(r, CURR_POS)
    if curr is None:
        sys.exit("ERROR: no current_position — is OpenSai running with picklebot.xml?")

    goal_pos = curr.copy()
    goal_ori = None

    joints = get(r, JOINTS)
    print(f"start_pos:    [{curr[0]:.4f}, {curr[1]:.4f}, {curr[2]:.4f}]")
    if joints is not None:
        jdeg = np.degrees(joints)
        print(f"start_joints: " + "  ".join(f"q{i+1}={jdeg[i]:.1f}" for i in range(len(jdeg))))
    print("Commands: 'X Y Z' | 'r dX dY dZ' | 'ori [rx ry rz]' | 'home' | 's' | 'q'")
    print(f"goal_pos written ONCE per command. tol={POS_TOL_M*1000:.0f}mm/{ANG_TOL_DEG:.0f}°  monitor={MONITOR_S:.0f}s")

    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue
        parts = line.split()
        cmd = parts[0].lower()

        if cmd == "q":
            break

        elif cmd == "s":
            show_state(r, goal_pos, goal_ori)

        elif cmd == "home":
            goal_pos = HOME.copy()
            send_goal(r, goal_pos, goal_ori, "home")

        elif cmd == "ori":
            if len(parts) == 1:
                goal_ori = R_W_E_REF.copy()
                r.set(GOAL_ORI, json.dumps(goal_ori.tolist()))
                print("  ori → R_W_E_REF (face +X opponent, handle -Z)")
                monitor(r, goal_pos, goal_ori, "ori")
            elif len(parts) == 4:
                try:
                    rx, ry, rz = float(parts[1]), float(parts[2]), float(parts[3])
                    goal_ori = euler_zyx_deg(rx, ry, rz, R_W_E_REF)
                    r.set(GOAL_ORI, json.dumps(goal_ori.tolist()))
                    monitor(r, goal_pos, goal_ori, f"ori {rx:.0f} {ry:.0f} {rz:.0f}")
                except ValueError:
                    print("  usage: ori rx ry rz (degrees, ZYX relative to R_W_E_REF)")
            else:
                print("  usage: ori   OR   ori rx ry rz")

        elif cmd == "r" and len(parts) == 4:
            try:
                delta = np.array([float(p) for p in parts[1:]])
                goal_pos = goal_pos + delta
                send_goal(r, goal_pos, goal_ori, f"r {delta[0]:.3f} {delta[1]:.3f} {delta[2]:.3f}")
            except ValueError:
                print("  usage: r dX dY dZ")

        elif len(parts) == 3:
            try:
                goal_pos = np.array([float(p) for p in parts])
                send_goal(r, goal_pos, goal_ori, f"{goal_pos[0]:.3f} {goal_pos[1]:.3f} {goal_pos[2]:.3f}")
            except ValueError:
                print("  usage: X Y Z (meters, arm base frame)")

        else:
            print("  unknown. Try: 0.4 0.0 0.6 | r 0.05 0 0 | ori | ori 0 20 0 | home | s | q")

    print("Exiting. Last goals remain in Redis.")


if __name__ == "__main__":
    main()
