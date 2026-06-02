#!/usr/bin/env python3
"""
Step 2: Interactive world-frame arm commander using base rigid body offset.

Reads T_W_A from the cart's OptiTrack rigid body + one-time calibrated offset
(arm_base_offset_calibration.json). Converts world-frame commands to arm-frame
goals, writes ONCE per command — OTG completes without interruption.

This is the first script in the incremental ladder that exercises the world-frame
pipeline. For pure arm-frame testing (no OptiTrack) use step1_arm_frame.py.
For the full production loop with continuous re-writes and rich diagnostics use
cmd_arm_world_clean.py.

Run from OpenSai root:
    python sports_bot/arm_control/step2_world_frame.py

Prerequisites:
    redis-server
    StreamDataSkeleton.py (rigid bodies streaming — labeled markers NOT needed)
    OpenSai cartesian_controller (picklebot.xml)
    arm_base_offset_calibration.json (run calibrate_arm_base_offset.py first)

Commands at the > prompt (all positions in world frame, metres):
    X Y Z                absolute sweet-spot position
    r dX dY dZ           relative nudge
    ori                  snap to reference orientation (face +X, handle -Z)
    ori rx ry rz         set orientation relative to reference (ZYX euler, degrees)
    ori r drx dry drz    nudge orientation by deltas
    s                    show current state in world frame
    q                    quit (last goals remain in Redis)

Frame note:
    W  world frame (+X toward opponent, +Y left, +Z up; floor tape origin)
    A  arm base frame (derived from cart rigid body + calibrated offset)
    goal_position / current_position are in A — this script does the conversion.
"""
import json
import math
import os
import sys
import time

import numpy as np
import redis

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SPORTS_BOT_DIR = os.path.dirname(_THIS_DIR)
_OPENSAI_DIR = os.path.dirname(_SPORTS_BOT_DIR)
if _OPENSAI_DIR not in sys.path:
    sys.path.insert(0, _OPENSAI_DIR)

from sports_bot.utils.frames import (  # noqa: E402
    arm_base_offset_calibration_path,
    compute_T_W_A_from_base_offset,
    load_arm_base_offset_calibration,
    read_rigid_body_pose_W,
)

# ---------------------------------------------------------------------------
# Redis keys
# ---------------------------------------------------------------------------
GOAL_POS = "opensai::controllers::FrankaRobot::cartesian_controller::cartesian_task::goal_position"
GOAL_ORI = "opensai::controllers::FrankaRobot::cartesian_controller::cartesian_task::goal_orientation"
CURR_POS = "opensai::controllers::FrankaRobot::cartesian_controller::cartesian_task::current_position"
CURR_ORI = "opensai::controllers::FrankaRobot::cartesian_controller::cartesian_task::current_orientation"
JOINTS   = "opensai::sensors::FrankaRobot::joint_positions"

# ---------------------------------------------------------------------------
# Reference orientation: face +X opponent, handle -Z world — verified 2026-05-26
# ---------------------------------------------------------------------------
_S2 = math.sqrt(0.5)
R_W_E_REF = np.array([
    [_S2,  _S2,  0.],
    [_S2, -_S2,  0.],
    [0.,   0.,  -1.],
])

MONITOR_S   = 8.0
POLL_S      = 0.05
POS_TOL_M   = 0.008
ANG_TOL_DEG = 3.0
SETTLE_S    = 0.5

JOINT_VEL_LIMIT_DEGS = np.degrees([2.175, 2.175, 2.175, 2.175, 2.610, 2.610, 2.610])
WARN_VEL_FRAC = 0.70


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get(r, key):
    v = r.get(key)
    return np.array(json.loads(v)) if v else None


def _rot_err_deg(R_curr, R_goal):
    cos_a = float(np.clip((np.trace(R_goal @ R_curr.T) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_a)))


def _Rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1., 0., 0.], [0., c, -s], [0., s, c]])


def _Ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])


def _Rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def _ori_from_euler_deg(rx, ry, rz, R_ref=R_W_E_REF):
    """R_W_E = Rz(rz) @ Ry(ry) @ Rx(rx) @ R_ref."""
    return _Rz(math.radians(rz)) @ _Ry(math.radians(ry)) @ _Rx(math.radians(rx)) @ R_ref


def _read_T_W_A(r, cal):
    """Return (R_W_A, t_W_A) or None if rigid body missing."""
    T_W_B = read_rigid_body_pose_W(r, cal.base_rigid_body_id)
    if T_W_B is None:
        return None
    return compute_T_W_A_from_base_offset(T_W_B, cal)


def _current_world_pos(r, cal):
    """Forward-kinematics sweet-spot position in world frame, or None."""
    T_W_A = _read_T_W_A(r, cal)
    if T_W_A is None:
        return None
    R_W_A, t_W_A = T_W_A
    t_A_E = _get(r, CURR_POS)
    if t_A_E is None:
        return None
    return R_W_A @ t_A_E + t_W_A


# ---------------------------------------------------------------------------
# Goal writing
# ---------------------------------------------------------------------------

def _write_goal(r, cal, t_W_goal, R_W_E_goal):
    """Convert world-frame goal to arm frame using live T_W_A; write once."""
    T_W_A = _read_T_W_A(r, cal)
    if T_W_A is None:
        print(f'  [WARN] cart rigid body {cal.base_rigid_body_id} missing — goal not written')
        return False
    R_W_A, t_W_A = T_W_A
    t_A_goal = R_W_A.T @ (t_W_goal - t_W_A)
    R_A_E_goal = R_W_A.T @ R_W_E_goal
    r.set(GOAL_POS, json.dumps(t_A_goal.tolist()))
    r.set(GOAL_ORI, json.dumps(R_A_E_goal.tolist()))
    yaw_A = math.degrees(math.atan2(float(R_W_A[1, 0]), float(R_W_A[0, 0])))
    print(f'  → arm_goal_pos=[{t_A_goal[0]:.3f},{t_A_goal[1]:.3f},{t_A_goal[2]:.3f}]  '
          f'T_W_A yaw={yaw_A:+.1f}°')
    return True


# ---------------------------------------------------------------------------
# Convergence monitor
# ---------------------------------------------------------------------------

def _monitor(r, cal, t_W_goal, R_W_E_goal, cmd_str):
    t0 = time.perf_counter()
    settled_since = None
    prev_joints = None
    prev_t = None
    qdot_peak = np.zeros(7)
    final_pos_W = None
    final_joints = None

    while True:
        now = time.perf_counter()
        elapsed = now - t0

        T_W_A = _read_T_W_A(r, cal)
        t_A_E = _get(r, CURR_POS)
        R_A_E = _get(r, CURR_ORI)
        curr_joints = _get(r, JOINTS)

        if T_W_A is None or t_A_E is None:
            time.sleep(POLL_S)
            continue

        R_W_A, t_W_A = T_W_A
        t_W_ss = R_W_A @ t_A_E + t_W_A
        final_pos_W = t_W_ss
        final_joints = curr_joints

        pos_err = float(np.linalg.norm(t_W_ss - t_W_goal))

        ang_err = 0.0
        if R_A_E is not None:
            R_W_E_now = R_W_A @ R_A_E.reshape(3, 3)
            ang_err = _rot_err_deg(R_W_E_now, R_W_E_goal)

        if curr_joints is not None and prev_joints is not None and prev_t is not None:
            dt = now - prev_t
            if dt > 0.01:
                qdot_peak = np.maximum(qdot_peak, np.abs((curr_joints - prev_joints) / dt))
        prev_joints = curr_joints
        prev_t = now

        in_tol = pos_err < POS_TOL_M and ang_err < ANG_TOL_DEG
        print(
            f'  t={elapsed:.1f}s  pos={pos_err*1000:.1f}mm  ang={ang_err:.1f}°'
            f'  sweet_W=[{t_W_ss[0]:.3f},{t_W_ss[1]:.3f},{t_W_ss[2]:.3f}]',
            end='\r',
        )

        if in_tol:
            if settled_since is None:
                settled_since = now
            elif now - settled_since >= SETTLE_S:
                print()
                _print_result(cmd_str, 'CONVERGED', t_W_goal, R_W_E_goal,
                               final_pos_W, elapsed, np.degrees(qdot_peak), final_joints)
                return
        else:
            settled_since = None

        if elapsed >= MONITOR_S:
            print()
            _print_result(cmd_str, f'TIMEOUT({MONITOR_S:.0f}s)', t_W_goal, R_W_E_goal,
                           final_pos_W, elapsed, np.degrees(qdot_peak), final_joints)
            return

        time.sleep(POLL_S)


def _print_result(cmd_str, status, t_W_goal, R_W_E_goal, final_pos_W,
                  t_elapsed, qdot_peak_degs, final_joints):
    sep = '━' * 64
    print(sep)
    print(f'CMD: {cmd_str}    STATUS: {status}')
    print(f'  goal_W:    [{t_W_goal[0]:.4f}, {t_W_goal[1]:.4f}, {t_W_goal[2]:.4f}]  (world, m)')
    if final_pos_W is not None:
        delta = t_W_goal - final_pos_W
        err_mm = np.linalg.norm(delta) * 1000
        print(f'  curr_W:    [{final_pos_W[0]:.4f}, {final_pos_W[1]:.4f}, {final_pos_W[2]:.4f}]')
        print(f'  pos_err:   {err_mm:.1f} mm  '
              f'delta=[{delta[0]*1000:.1f},{delta[1]*1000:.1f},{delta[2]*1000:.1f}] mm')
    print(f'  t_elapsed: {t_elapsed:.2f} s')
    if final_joints is not None:
        jdeg = np.degrees(final_joints)
        print(f'  joints:    ' + '  '.join(f'q{i+1}={jdeg[i]:.1f}' for i in range(len(jdeg))))
    if qdot_peak_degs is not None and np.any(qdot_peak_degs > 0):
        mi = int(np.argmax(qdot_peak_degs))
        mv = qdot_peak_degs[mi]
        lim = JOINT_VEL_LIMIT_DEGS[mi]
        flag = '  ← NEAR LIMIT' if mv > WARN_VEL_FRAC * lim else ''
        print(f'  qdot_max:  {mv:.0f} deg/s @ q{mi+1}  ({100*mv/lim:.0f}% of {lim:.0f} limit){flag}')
    print(sep)


# ---------------------------------------------------------------------------
# State display
# ---------------------------------------------------------------------------

def _show_state(r, cal, t_W_goal, R_W_E_goal):
    T_W_A = _read_T_W_A(r, cal)
    t_A_E = _get(r, CURR_POS)
    R_A_E = _get(r, CURR_ORI)
    joints = _get(r, JOINTS)

    if T_W_A is None:
        print(f'  [WARN] cart rigid body {cal.base_rigid_body_id} not visible')
        return
    if t_A_E is None:
        print('  (no current_position — is OpenSai running?)')
        return

    R_W_A, t_W_A = T_W_A
    t_W_ss = R_W_A @ t_A_E + t_W_A
    yaw_A = math.degrees(math.atan2(float(R_W_A[1, 0]), float(R_W_A[0, 0])))
    pos_err = np.linalg.norm(t_W_ss - t_W_goal) * 1000

    print(f'  T_W_A:     pos=[{t_W_A[0]:.3f},{t_W_A[1]:.3f},{t_W_A[2]:.3f}]  yaw={yaw_A:+.1f}°')
    print(f'  curr_W:    [{t_W_ss[0]:.4f}, {t_W_ss[1]:.4f}, {t_W_ss[2]:.4f}]')
    print(f'  goal_W:    [{t_W_goal[0]:.4f}, {t_W_goal[1]:.4f}, {t_W_goal[2]:.4f}]')
    print(f'  pos_err:   {pos_err:.1f} mm')
    if R_A_E is not None:
        R_W_E_now = R_W_A @ R_A_E.reshape(3, 3)
        ang_err = _rot_err_deg(R_W_E_now, R_W_E_goal)
        print(f'  ang_err:   {ang_err:.1f}°')
    if joints is not None:
        jdeg = np.degrees(joints)
        print(f'  joints:    ' + '  '.join(f'q{i+1}={jdeg[i]:.1f}' for i in range(len(jdeg))))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cal_path = arm_base_offset_calibration_path()
    if not os.path.isfile(cal_path):
        sys.exit(
            f'ERROR: {cal_path} not found.\n'
            f'Run: python sports_bot/scripts/calibrate_arm_base_offset.py'
        )
    cal = load_arm_base_offset_calibration(cal_path)

    r = redis.Redis(host='localhost', port=6379, decode_responses=True)
    try:
        r.ping()
    except Exception:
        sys.exit('ERROR: Redis not reachable')

    T_W_A_init = _read_T_W_A(r, cal)
    if T_W_A_init is None:
        sys.exit(f'ERROR: cart rigid body {cal.base_rigid_body_id} not visible — '
                 f'is OptiTrack streamer running?')

    t_A_E_init = _get(r, CURR_POS)
    if t_A_E_init is None:
        sys.exit('ERROR: no current_position — is OpenSai running with picklebot.xml?')

    R_W_A_init, t_W_A_init = T_W_A_init
    t_W_ss_init = R_W_A_init @ t_A_E_init + t_W_A_init
    yaw_A_init = math.degrees(math.atan2(float(R_W_A_init[1, 0]), float(R_W_A_init[0, 0])))

    print(f'[step2] base rigid body ID: {cal.base_rigid_body_id}')
    print(f'[step2] T_W_A:  pos=[{t_W_A_init[0]:.3f},{t_W_A_init[1]:.3f},{t_W_A_init[2]:.3f}]  '
          f'yaw={yaw_A_init:+.1f}°')
    print(f'[step2] sweet-spot (world): [{t_W_ss_init[0]:.3f},{t_W_ss_init[1]:.3f},{t_W_ss_init[2]:.3f}]')
    print(f'[step2] goal writes ONCE per command  |  tol={POS_TOL_M*1000:.0f}mm/{ANG_TOL_DEG:.0f}°  '
          f'monitor={MONITOR_S:.0f}s')
    print()
    print('Commands (world frame, metres):')
    print('  X Y Z                — absolute sweet-spot position')
    print('  r dX dY dZ           — relative nudge')
    print('  ori                  — snap to reference (face +X, handle -Z)')
    print('  ori rx ry rz         — set orientation (° from reference, ZYX)')
    print('  ori r drx dry drz    — nudge orientation by deltas (°)')
    print('  s                    — show current state')
    print('  q                    — quit')

    # Start at current sweet-spot world position
    t_W_goal = t_W_ss_init.copy()
    R_W_E_goal = R_W_A_init @ _get(r, CURR_ORI).reshape(3, 3) if _get(r, CURR_ORI) is not None else R_W_E_REF.copy()
    rx_goal = ry_goal = rz_goal = 0.0  # relative to R_W_E_REF

    while True:
        try:
            line = input('\n> ').strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue
        parts = line.split()
        cmd = parts[0].lower()

        if cmd == 'q':
            break

        elif cmd == 's':
            _show_state(r, cal, t_W_goal, R_W_E_goal)

        elif cmd == 'ori':
            rel = len(parts) > 1 and parts[1].lower() in ('r', 'rel')
            nums_start = 2 if rel else 1
            nums = parts[nums_start:]

            if len(nums) == 0 and not rel:
                rx_goal = ry_goal = rz_goal = 0.0
                R_W_E_goal = R_W_E_REF.copy()
                r.set(GOAL_ORI, json.dumps((R_W_A_init.T @ R_W_E_goal).tolist()))
                print('  ori → reference (face +X opponent, handle -Z)')
                _monitor(r, cal, t_W_goal, R_W_E_goal, 'ori')
            elif len(nums) == 3:
                try:
                    drx, dry, drz = float(nums[0]), float(nums[1]), float(nums[2])
                    if rel:
                        rx_goal += drx; ry_goal += dry; rz_goal += drz
                    else:
                        rx_goal, ry_goal, rz_goal = drx, dry, drz
                    R_W_E_goal = _ori_from_euler_deg(rx_goal, ry_goal, rz_goal)
                    if _write_goal(r, cal, t_W_goal, R_W_E_goal):
                        tag = ' (rel)' if rel else ''
                        _monitor(r, cal, t_W_goal, R_W_E_goal,
                                 f'ori{tag} {rx_goal:.0f} {ry_goal:.0f} {rz_goal:.0f}')
                except ValueError:
                    print('  usage: ori [r] rx ry rz')
            else:
                print('  usage: ori   |   ori rx ry rz   |   ori r drx dry drz')

        elif cmd == 'r' and len(parts) == 4:
            try:
                delta = np.array([float(p) for p in parts[1:]])
                t_W_goal = t_W_goal + delta
                if _write_goal(r, cal, t_W_goal, R_W_E_goal):
                    _monitor(r, cal, t_W_goal, R_W_E_goal,
                             f'r {delta[0]:.3f} {delta[1]:.3f} {delta[2]:.3f}')
            except ValueError:
                print('  usage: r dX dY dZ')

        elif len(parts) == 3:
            try:
                t_W_goal = np.array([float(p) for p in parts])
                if _write_goal(r, cal, t_W_goal, R_W_E_goal):
                    _monitor(r, cal, t_W_goal, R_W_E_goal,
                             f'{t_W_goal[0]:.3f} {t_W_goal[1]:.3f} {t_W_goal[2]:.3f}')
            except ValueError:
                print('  usage: X Y Z (metres, world frame)')

        else:
            print('  unknown. Try: 0.3 0.0 0.8 | r 0.1 0 0 | ori | ori 0 20 0 | s | q')

    print('Exiting. Last goals remain in Redis.')


if __name__ == '__main__':
    main()
