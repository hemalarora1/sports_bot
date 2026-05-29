#!/usr/bin/env python3
"""cmd_arm_world_clean.py — World-frame arm commander for the pickleball robot.

Prereqs: Redis, OptiTrack streamer (labeled-marker streaming enabled in
Motive), OpenSai cartesian_controller running (picklebot.xml).

Design (post 2026-05-27 rewrite — read this if data looks wrong)
-----------------------------------------------------------------
The old version of this script wrote `goal_position`, `goal_orientation`,
AND `goal_linear_velocity` to Redis every 20 ms. With OpenSai's
`<otg type="acceleration">` enabled, that meant the OTG was being asked
to plan a new "decelerate-and-stop-at-goal" trajectory 50 times per
second, off a goal that jittered every tick because it was recomputed
from a live, noisy T_W_A. Result: persistent steady-state errors
(17–50 mm), arm visibly jittery, orientation changes barely tracked.

This version:

  * Writes the goal only when (a) the user changes it, or (b) the
    derived goal in the arm frame drifts past a threshold relative to
    what we last wrote (default 3 mm / 0.5°). Cart-still + user-silent
    means zero rewrites — the controller can converge.
  * Does NOT write `goal_linear_velocity` at all (leave it as whatever
    OpenSai/picklebot.xml configures).
  * No null-space joint task. The `<jointTask>` block was removed from
    `picklebot.xml`'s cartesian_controller as part of the same change.
  * Auto-assigns marker roles each tick: of the two configured markers,
    the one with smaller world Y is treated as the -Y side, the one
    with larger world Y as the +Y side. marker_specs order in the JSON
    is informational only.
  * Prints a single diagnostic block every 2 s with rolling-window
    stats for marker jitter, T_W_A jitter, goal_A jitter, pos/ang
    error history, joint velocities, and write rate. The intent is
    that pasting one block back into a chat is enough to root-cause
    any remaining tracking misbehavior.

Frame conventions
-----------------
  W   world frame  (+X toward opponent, +Y left, +Z up; floor tape origin)
  A   Franka arm base frame  (derived from two OptiTrack markers each tick)
  E   EE / compliant frame   (sweet spot; picklebot.xml compliantFrame
                              xyz="0 0 0.35" — OpenSai's current_position
                              IS the sweet spot)

Reference orientation (ori 0 0 0)
-----------------------------------
  Face toward opponent  (+X world)
  Handle pointing down  (-Z world)

  Franka link8 vs panda_hand: panda_hand_joint has rpy="0 0 -π/4", so
  panda_hand = Rz(-45°) @ link8. OpenSai's EE frame = link8 (the joint
  BEFORE that rotation). Paddle face = panda_hand +Y = [1/√2, 1/√2, 0]
  in link8 coords.

  R_W_E_REF = [[1/√2,  1/√2,  0],
               [1/√2, -1/√2,  0],
               [0,     0,    -1]]

ori command convention
-----------------------
  ori rx ry rz  (degrees, extrinsic rotations applied on top of reference)
    R_W_E_goal = Rz(rz) @ Ry(ry) @ Rx(rx) @ R_W_E_ref
    R_A_E_goal = R_W_A.T @ R_W_E_goal

  Physical meaning of small deltas from ori 0 0 0:
    rx  — handle sways sideways  (roll around face-normal / world +X)
    ry  — face pitches down (-) / up (+) from horizontal
    rz  — face yaws toward +Y (left) / -Y (right) in horizontal plane

Commands
--------
  x y z                      absolute position; keep orientation
  r dx dy dz                 relative position delta; keep orientation
  ori rx ry rz               set orientation from reference (°)
  ori r drx dry drz          nudge orientation by deltas (°)
  x y z rx ry rz             set position + set orientation
  r dx dy dz rx ry rz        relative position + set orientation
  x y z r drx dry drz        set position + nudge orientation
  r dx dy dz r drx dry drz   relative position + nudge orientation
  q / quit                   exit (last written goals remain in Redis)

Diagnostic block (every 2 s)
----------------------------
  [markers]  per-marker raw position + jitter σ + missing count over
             rolling 2 s window; role assignment (-Y / +Y side).
  [T_W_A]    arm base pose + jitter σ + yaw drift.
  [goal_W]   user-set world goal + seconds since last user change.
  [goal_A]   computed arm-frame goal + drift since last write +
             write count + CLIPPED flag if workspace clip kicked in.
  [current]  FK sweet-spot world pos, position error per-axis (world),
             angle error, and rolling min/max/mean/σ for pos and ang
             error over the last ~2 s.
  [joints]   joint angles (°) + joint speeds (°/s) + max speed.
  [diag]     measured loop Hz, total writes, marker jumps, FK misses.

Examples
--------
  python sports_bot/scripts/cmd_arm_world_clean.py            # start at current pose
  python sports_bot/scripts/cmd_arm_world_clean.py --x 0.5 --y 0.0
  python sports_bot/scripts/cmd_arm_world_clean.py --write-pos-tol-mm 5 --write-ang-tol-deg 1.0
"""
from __future__ import annotations

import argparse
import json
import math
import os
import queue
import sys
import threading
import time
from collections import deque
from typing import Deque, List, Optional, Tuple

import numpy as np
import redis

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SPORTS_BOT_DIR = os.path.dirname(_THIS_DIR)
_OPENSAI_DIR = os.path.dirname(_SPORTS_BOT_DIR)
if _OPENSAI_DIR not in sys.path:
    sys.path.insert(0, _OPENSAI_DIR)

from sports_bot.state_machine.redis_keys import OpenSaiCartesianKeys  # noqa: E402
from sports_bot.utils.frames import (  # noqa: E402
    FRANKA_DEFAULT_REACH_M,
    FRANKA_DEFAULT_Z_MAX_M,
    FRANKA_DEFAULT_Z_MIN_M,
    ArmBaseOffsetCalibration,
    arm_base_offset_calibration_path,
    clip_to_arm_workspace,
    compute_T_W_A_from_base_offset,
    load_arm_base_offset_calibration,
    read_rigid_body_pose_W,
)

# ---------------------------------------------------------------------------
# Reference orientation (see docstring)
# ---------------------------------------------------------------------------
_S2 = math.sqrt(0.5)
R_W_E_REF: np.ndarray = np.array([
    [_S2,  _S2,  0.],
    [_S2, -_S2,  0.],
    [0.,   0.,  -1.],
])

# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------
_WARN_INTERVAL_S = 1.0        # de-dupe warning printing
_DEBUG_INTERVAL_S = 2.0       # diag block cadence
_STATS_WINDOW_S = 2.0         # rolling stats window
_HEARTBEAT_S = math.inf       # don't auto-rewrite goals on a timer; pure threshold

# ---------------------------------------------------------------------------
# Rotation helpers
# ---------------------------------------------------------------------------

def _Rx(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1., 0., 0.], [0., c, -s], [0., s, c]])

def _Ry(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])

def _Rz(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])

def _R_from_rpy_rad(rx: float, ry: float, rz: float) -> np.ndarray:
    return _Rz(rz) @ _Ry(ry) @ _Rx(rx)

def _rpy_deg_from_R(R: np.ndarray) -> Tuple[float, float, float]:
    """ZYX Euler decomposition of R = Rz(rz)@Ry(ry)@Rx(rx)."""
    sy = float(-R[2, 0])
    cy = math.sqrt(max(0.0, float(R[2, 1])**2 + float(R[2, 2])**2))
    ry = math.atan2(sy, cy)
    if cy < 1e-6:
        rx = math.atan2(-float(R[1, 2]), float(R[1, 1]))
        rz = 0.0
    else:
        rx = math.atan2(float(R[2, 1]) / cy, float(R[2, 2]) / cy)
        rz = math.atan2(float(R[1, 0]) / cy, float(R[0, 0]) / cy)
    return math.degrees(rx), math.degrees(ry), math.degrees(rz)

def _ori_deg_from_R_W_E(R_W_E: np.ndarray) -> Tuple[float, float, float]:
    """Inverse of the ori-command convention: given R_W_E, return (rx,ry,rz)°."""
    return _rpy_deg_from_R(R_W_E @ R_W_E_REF.T)

def _angle_between_R(Ra: np.ndarray, Rb: np.ndarray) -> float:
    """Geodesic angle (rad) between two rotation matrices."""
    dR = Ra @ Rb.T
    cos_th = max(-1.0, min(1.0, (float(np.trace(dR)) - 1.0) / 2.0))
    return math.acos(cos_th)

# ---------------------------------------------------------------------------
# Redis helpers
# ---------------------------------------------------------------------------

def _read_ee_pose(
    r: redis.Redis, keys: OpenSaiCartesianKeys
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Read OpenSai's current EE (compliant-frame / sweet-spot) pose in arm frame."""
    pos_raw = r.get(keys.current_position)
    ori_raw = r.get(keys.current_orientation)
    if pos_raw is None or ori_raw is None:
        return None
    try:
        t = np.asarray(json.loads(pos_raw), dtype=float)
        R = np.asarray(json.loads(ori_raw), dtype=float)
    except Exception:
        return None
    if t.shape != (3,) or R.shape != (3, 3):
        return None
    return R, t


def _write_goal_pose(
    r: redis.Redis, keys: OpenSaiCartesianKeys,
    t_A: np.ndarray, R_A_E: np.ndarray,
) -> None:
    """Write sweet-spot position + orientation to Redis. No velocity write —
    OpenSai's OTG handles velocity profile internally given a pose target."""
    r.set(keys.goal_position,
          json.dumps([float(v) for v in t_A]))
    r.set(keys.goal_orientation,
          json.dumps([[float(v) for v in row] for row in R_A_E]))

# ---------------------------------------------------------------------------
# Command parsing (unchanged from previous version)
# ---------------------------------------------------------------------------
_ParseResult = Tuple[str, Optional[np.ndarray], bool,
                     Optional[float], Optional[float], Optional[float], bool]


def _parse_cmd(line: str) -> Optional[_ParseResult]:
    parts = line.split()
    if not parts:
        return None

    if parts[0].lower() == 'ori':
        if len(parts) == 4:
            try:
                rx, ry, rz = float(parts[1]), float(parts[2]), float(parts[3])
                return ('ori', None, False, rx, ry, rz, False)
            except ValueError:
                return None
        if len(parts) == 5 and parts[1].lower() in ('r', 'rel'):
            try:
                rx, ry, rz = float(parts[2]), float(parts[3]), float(parts[4])
                return ('ori', None, False, rx, ry, rz, True)
            except ValueError:
                return None
        return None

    relative = parts[0].lower() in ('r', 'rel')
    nums = parts[1:] if relative else parts[:]
    try:
        if len(nums) == 3:
            t = np.array([float(v) for v in nums])
            return ('pos', t, relative, None, None, None, False)
        if len(nums) == 6:
            t = np.array([float(v) for v in nums[:3]])
            rx, ry, rz = float(nums[3]), float(nums[4]), float(nums[5])
            return ('both', t, relative, rx, ry, rz, False)
        if len(nums) == 7 and nums[3].lower() in ('r', 'rel'):
            t = np.array([float(v) for v in nums[:3]])
            rx, ry, rz = float(nums[4]), float(nums[5]), float(nums[6])
            return ('both', t, relative, rx, ry, rz, True)
    except ValueError:
        pass
    return None


def _stdin_reader(q: 'queue.Queue[str]') -> None:
    try:
        for line in sys.stdin:
            q.put(line.strip())
    except EOFError:
        pass
    q.put('q')

# ---------------------------------------------------------------------------
# Rolling-stats helpers
# ---------------------------------------------------------------------------

def _stats_of(samples: List[float]) -> Tuple[float, float, float, float]:
    """min, max, mean, σ. Empty → all zeros."""
    if not samples:
        return 0.0, 0.0, 0.0, 0.0
    arr = np.asarray(samples, dtype=float)
    return (float(arr.min()), float(arr.max()),
            float(arr.mean()), float(arr.std()))


def _vec_jitter(samples: List[np.ndarray]) -> Tuple[float, float, float]:
    """Per-axis σ (in mm) of a list of 3-vectors. Empty → zeros."""
    if len(samples) < 2:
        return 0.0, 0.0, 0.0
    arr = np.stack(samples)
    sig = arr.std(axis=0) * 1000.0
    return float(sig[0]), float(sig[1]), float(sig[2])

# ---------------------------------------------------------------------------
# Main control loop
# ---------------------------------------------------------------------------

def run(
    r: redis.Redis,
    keys: OpenSaiCartesianKeys,
    base_rb_id: int,
    cal_offset: ArmBaseOffsetCalibration,
    t_W_goal: np.ndarray,
    rx_goal: float, ry_goal: float, rz_goal: float,
    rate_hz: float,
    reach_m: float,
    z_min_m: float,
    z_max_m: float,
    robot_name: str,
    write_pos_tol_m: float,
    write_ang_tol_rad: float,
) -> None:
    cmd_q: 'queue.Queue[str]' = queue.Queue()
    threading.Thread(target=_stdin_reader, args=(cmd_q,), daemon=True).start()

    dt = 1.0 / rate_hz
    win_samples = int(round(_STATS_WINDOW_S * rate_hz))

    # ---- Base rigid body tracking ----
    base_rb_pos_hist: Deque[np.ndarray] = deque(maxlen=win_samples)
    base_rb_yaw_hist: Deque[float] = deque(maxlen=win_samples)
    base_rb_miss_total: int = 0
    base_rb_warn_t: float = -math.inf
    # Keep latest T_W_B for diagnostics (populated once loop starts)
    t_W_B_latest: np.ndarray = np.zeros(3)
    yaw_B_latest: float = 0.0

    # ---- T_W_A and goal_A rolling histories ----
    t_W_A_hist: Deque[np.ndarray] = deque(maxlen=win_samples)
    yaw_W_A_hist: Deque[float] = deque(maxlen=win_samples)
    t_A_goal_hist: Deque[np.ndarray] = deque(maxlen=win_samples)

    # ---- Error histories ----
    pos_err_hist: Deque[float] = deque(maxlen=win_samples)
    ang_err_hist_deg: Deque[float] = deque(maxlen=win_samples)
    pos_err_world_hist: Deque[np.ndarray] = deque(maxlen=win_samples)
    ee_miss_total = 0

    # ---- Joint tracking ----
    _sensor_key = f'opensai::sensors::{robot_name}::joint_positions'
    q_prev: Optional[np.ndarray] = None
    q_prev_t: float = -math.inf
    qdot_latest: Optional[np.ndarray] = None   # most recent finite-difference

    # ---- Write tracking ----
    last_written_t_A: Optional[np.ndarray] = None
    last_written_R_A_E: Optional[np.ndarray] = None
    last_write_t: float = -math.inf
    writes_total: int = 0
    writes_since_diag: int = 0
    last_user_change_t: float = time.perf_counter()

    # ---- Loop / diag state ----
    last_debug_t: float = -math.inf
    loop_tick_count_since_diag: int = 0
    loop_window_start_t: float = time.perf_counter()
    last_was_clipped: bool = False

    _print_prompt()

    while True:
        t_loop = time.perf_counter()
        loop_tick_count_since_diag += 1

        # ----------------------------------------------------------------
        # 1. Drain user commands
        # ----------------------------------------------------------------
        user_changed_goal = False
        try:
            while True:
                line = cmd_q.get_nowait()
                if line.lower() in ('q', 'quit', 'exit'):
                    print('\n[cmd_arm] exit — last written goals remain in Redis.')
                    return
                if line == '':
                    _print_prompt()
                    continue
                parsed = _parse_cmd(line)
                if parsed is None:
                    print(f'[cmd_arm] unrecognised: "{line}"')
                    print('  x y z                      — absolute position')
                    print('  r dx dy dz                 — relative position')
                    print('  ori rx ry rz               — set orientation (° from ref)')
                    print('  ori r drx dry drz          — nudge orientation (° deltas)')
                    print('  x y z rx ry rz             — position + set orientation')
                    print('  r dx dy dz rx ry rz        — relative pos + set orientation')
                    print('  x y z r drx dry drz        — pos + nudge orientation')
                    print('  r dx dy dz r drx dry drz   — relative pos + nudge ori')
                    _print_prompt()
                    continue

                kind, t_delta, is_rel, rx_new, ry_new, rz_new, is_ori_rel = parsed
                if kind in ('pos', 'both'):
                    assert t_delta is not None
                    if is_rel:
                        t_W_goal = t_W_goal + t_delta
                    else:
                        t_W_goal = t_delta.copy()
                if kind in ('ori', 'both'):
                    if is_ori_rel:
                        rx_goal += rx_new  # type: ignore[assignment]
                        ry_goal += ry_new  # type: ignore[assignment]
                        rz_goal += rz_new  # type: ignore[assignment]
                    else:
                        rx_goal = rx_new  # type: ignore[assignment]
                        ry_goal = ry_new  # type: ignore[assignment]
                        rz_goal = rz_new  # type: ignore[assignment]
                user_changed_goal = True
                last_user_change_t = t_loop

                pos_tag = '  (rel pos)' if is_rel else ''
                ori_tag = '  (rel ori)' if (is_ori_rel and kind in ('ori', 'both')) else ''
                print(
                    f'[user]    goal_W pos=[{t_W_goal[0]:+.3f},'
                    f'{t_W_goal[1]:+.3f},{t_W_goal[2]:+.3f}]'
                    f'  ori=[{rx_goal:+.1f}°,{ry_goal:+.1f}°,{rz_goal:+.1f}°]'
                    f'{pos_tag}{ori_tag}'
                )
                _print_prompt()
        except queue.Empty:
            pass

        # ----------------------------------------------------------------
        # 2. Read cart rigid body → T_W_A via calibrated offset
        # ----------------------------------------------------------------
        T_W_B = read_rigid_body_pose_W(r, base_rb_id)
        if T_W_B is None:
            base_rb_miss_total += 1
            if t_loop - base_rb_warn_t > _WARN_INTERVAL_S:
                base_rb_warn_t = t_loop
                print(f'[WARN] cart rigid body {base_rb_id} missing from Redis')
            time.sleep(max(0.0, dt - (time.perf_counter() - t_loop)))
            continue

        R_W_B, t_W_B_latest = T_W_B
        yaw_B_latest = math.atan2(float(R_W_B[1, 0]), float(R_W_B[0, 0]))
        base_rb_pos_hist.append(t_W_B_latest.copy())
        base_rb_yaw_hist.append(yaw_B_latest)

        # ----------------------------------------------------------------
        # 3. Apply offset calibration → T_W_A
        # ----------------------------------------------------------------
        R_W_A, t_W_A = compute_T_W_A_from_base_offset(T_W_B, cal_offset)
        yaw_W_A = math.atan2(float(R_W_A[1, 0]), float(R_W_A[0, 0]))

        t_W_A_hist.append(t_W_A.copy())
        yaw_W_A_hist.append(yaw_W_A)

        # ----------------------------------------------------------------
        # 4. Compute the *would-be* arm-frame goal this tick
        # ----------------------------------------------------------------
        t_A_raw = R_W_A.T @ (t_W_goal - t_W_A)
        t_A_goal, was_clipped = clip_to_arm_workspace(
            t_A_raw, r_max=reach_m, z_min=z_min_m, z_max=z_max_m)
        last_was_clipped = was_clipped

        R_W_E_goal = (_R_from_rpy_rad(
            math.radians(rx_goal),
            math.radians(ry_goal),
            math.radians(rz_goal),
        ) @ R_W_E_REF)
        R_A_E_goal = R_W_A.T @ R_W_E_goal

        t_A_goal_hist.append(t_A_goal.copy())

        # ----------------------------------------------------------------
        # 5. Rate-limited goal write
        # ----------------------------------------------------------------
        should_write = False
        if user_changed_goal:
            should_write = True
            write_reason = 'user'
        elif last_written_t_A is None or last_written_R_A_E is None:
            should_write = True
            write_reason = 'first'
        else:
            d_pos = float(np.linalg.norm(t_A_goal - last_written_t_A))
            d_ang = _angle_between_R(R_A_E_goal, last_written_R_A_E)
            if d_pos > write_pos_tol_m or d_ang > write_ang_tol_rad:
                should_write = True
                write_reason = f'drift(pos={d_pos*1000:.1f}mm,ang={math.degrees(d_ang):.2f}°)'
            else:
                write_reason = ''

        if should_write:
            _write_goal_pose(r, keys, t_A_goal, R_A_E_goal)
            last_written_t_A = t_A_goal.copy()
            last_written_R_A_E = R_A_E_goal.copy()
            last_write_t = t_loop
            writes_total += 1
            writes_since_diag += 1

        # ----------------------------------------------------------------
        # 6. Read current EE pose + joints; update error / velocity stats
        # ----------------------------------------------------------------
        ee = _read_ee_pose(r, keys)
        if ee is None:
            ee_miss_total += 1
            R_A_E_now, t_A_E_now = None, None
        else:
            R_A_E_now, t_A_E_now = ee
            t_W_ss = R_W_A @ t_A_E_now + t_W_A
            pos_err_world = t_W_ss - t_W_goal
            pos_err_m = float(np.linalg.norm(pos_err_world))
            ang_err_deg = math.degrees(_angle_between_R(R_A_E_goal, R_A_E_now))
            pos_err_hist.append(pos_err_m)
            ang_err_hist_deg.append(ang_err_deg)
            pos_err_world_hist.append(pos_err_world.copy())

        q_raw_now = r.get(_sensor_key)
        q_vals_now: Optional[List[float]] = None
        if q_raw_now is not None:
            try:
                q_parsed = json.loads(q_raw_now)
                if len(q_parsed) == 7:
                    q_vals_now = q_parsed
            except Exception:
                pass
        if q_vals_now is not None:
            q_arr = np.asarray(q_vals_now, dtype=float)
            if q_prev is not None and q_prev_t > -math.inf:
                tau = t_loop - q_prev_t
                if tau > 1e-6:
                    qdot_latest = (q_arr - q_prev) / tau
            q_prev = q_arr
            q_prev_t = t_loop

        # ----------------------------------------------------------------
        # 7. Diagnostic block every _DEBUG_INTERVAL_S
        # ----------------------------------------------------------------
        if t_loop - last_debug_t >= _DEBUG_INTERVAL_S:
            window_dt = t_loop - loop_window_start_t
            loop_hz = (loop_tick_count_since_diag / window_dt
                       if window_dt > 0 else 0.0)
            last_debug_t = t_loop

            # --- base rigid body
            rb_arr = np.stack(list(base_rb_pos_hist)) if base_rb_pos_hist else np.zeros((1, 3))
            rb_sig = rb_arr.std(axis=0) * 1000.0
            yaw_rb_arr = np.asarray(list(base_rb_yaw_hist)) if base_rb_yaw_hist else np.zeros(1)
            yaw_rb_sig_deg = math.degrees(float(yaw_rb_arr.std()))

            # --- T_W_A jitter
            twa_arr = np.stack(list(t_W_A_hist)) if t_W_A_hist else np.zeros((1, 3))
            twa_sig = twa_arr.std(axis=0) * 1000.0
            yaw_arr = np.asarray(list(yaw_W_A_hist)) if yaw_W_A_hist else np.zeros(1)
            yaw_sig_deg = math.degrees(float(yaw_arr.std()))

            # --- goal_A jitter
            gA_arr = np.stack(list(t_A_goal_hist)) if t_A_goal_hist else np.zeros((1, 3))
            gA_sig = gA_arr.std(axis=0) * 1000.0

            # --- errors
            p_min, p_max, p_mean, p_sd = _stats_of(list(pos_err_hist))
            a_min, a_max, a_mean, a_sd = _stats_of(list(ang_err_hist_deg))
            pew = (np.stack(list(pos_err_world_hist))
                   if pos_err_world_hist else np.zeros((1, 3)))
            pew_mean = pew.mean(axis=0) * 1000.0  # mm, per world axis

            # --- joints / qdot
            if q_vals_now is not None:
                q_deg = [math.degrees(v) for v in q_vals_now]
                q_line = '[joints]   ' + '  '.join(
                    f'q{i+1}={q_deg[i]:+6.1f}°' for i in range(7))
            else:
                q_line = '[joints]   unavailable'

            if qdot_latest is not None:
                qd_deg = [math.degrees(v) for v in qdot_latest.tolist()]
                qd_max = max(abs(v) for v in qd_deg)
                qd_line = ('[qdot]     ' + '  '.join(
                    f'q{i+1}={qd_deg[i]:+6.1f}°/s' for i in range(7))
                    + f'  |max|={qd_max:.1f}°/s')
            else:
                qd_line = '[qdot]     unavailable'

            # --- last-write info
            if last_written_t_A is not None and last_written_R_A_E is not None:
                d_pos_since = float(np.linalg.norm(t_A_goal - last_written_t_A))
                d_ang_since_deg = math.degrees(
                    _angle_between_R(R_A_E_goal, last_written_R_A_E))
                age = t_loop - last_write_t
                write_info = (
                    f'last_write {age:.1f}s ago  '
                    f'drift_since=(Δpos={d_pos_since*1000:.2f}mm,'
                    f'Δang={d_ang_since_deg:.2f}°)  '
                    f'writes/2s={writes_since_diag}  writes_total={writes_total}'
                )
            else:
                write_info = 'no writes yet'

            clip_tag = '  CLIPPED' if last_was_clipped else ''

            # --- current state line
            if R_A_E_now is not None and t_A_E_now is not None:
                t_W_ss = R_W_A @ t_A_E_now + t_W_A
                R_W_E_now = R_W_A @ R_A_E_now
                crx, cry, crz = _ori_deg_from_R_W_E(R_W_E_now)
                cur_lines = [
                    f'[current]  sweet_W=[{t_W_ss[0]:+.3f},{t_W_ss[1]:+.3f},{t_W_ss[2]:+.3f}]'
                    f'  ori=[{crx:+.1f}°,{cry:+.1f}°,{crz:+.1f}°]',
                    f'           pos_err_W mean=[{pew_mean[0]:+.1f},{pew_mean[1]:+.1f},{pew_mean[2]:+.1f}]mm'
                    f'  total: min={p_min*1000:.1f} max={p_max*1000:.1f} '
                    f'mean={p_mean*1000:.1f} σ={p_sd*1000:.1f} mm  (n={len(pos_err_hist)})',
                    f'           ang_err total: min={a_min:.2f} max={a_max:.2f} '
                    f'mean={a_mean:.2f} σ={a_sd:.2f} deg',
                ]
            else:
                cur_lines = ['[current]  EE pose unavailable (FK keys missing)']

            since_user_s = t_loop - last_user_change_t

            print()
            print(f'=== diag t≈{t_loop - loop_window_start_t + (last_debug_t - loop_window_start_t):.0f}s '
                  f'(loop {loop_hz:.1f} Hz over {window_dt:.1f}s) ===')
            print(
                f'[base_rb]  id={base_rb_id}  '
                f'pos=[{t_W_B_latest[0]:+.3f},{t_W_B_latest[1]:+.3f},{t_W_B_latest[2]:+.3f}]  '
                f'yaw={math.degrees(yaw_B_latest):+.2f}°  '
                f'σ_xyz=({rb_sig[0]:.2f},{rb_sig[1]:.2f},{rb_sig[2]:.2f})mm  '
                f'σ_yaw={yaw_rb_sig_deg:.3f}°  miss_total={base_rb_miss_total}'
            )
            print(
                f'[T_W_A]    pos=[{t_W_A[0]:+.3f},{t_W_A[1]:+.3f},{t_W_A[2]:+.3f}]'
                f'  σ_xyz=({twa_sig[0]:.2f},{twa_sig[1]:.2f},{twa_sig[2]:.2f})mm  '
                f'yaw={math.degrees(yaw_W_A):+.2f}° σ_yaw={yaw_sig_deg:.3f}°'
            )
            print(
                f'[goal_W]   pos=[{t_W_goal[0]:+.3f},{t_W_goal[1]:+.3f},{t_W_goal[2]:+.3f}]'
                f'  ori=[{rx_goal:+.1f}°,{ry_goal:+.1f}°,{rz_goal:+.1f}°]'
                f'  (user idle {since_user_s:.1f}s)'
            )
            print(
                f'[goal_A]   pos=[{t_A_goal[0]:+.3f},{t_A_goal[1]:+.3f},{t_A_goal[2]:+.3f}]'
                f'{clip_tag}  σ_xyz=({gA_sig[0]:.2f},{gA_sig[1]:.2f},{gA_sig[2]:.2f})mm'
            )
            print(f'[write]    {write_info}')
            for ln in cur_lines:
                print(ln)
            print(q_line)
            print(qd_line)
            print(
                f'[diag]     write_thresh=(pos>{write_pos_tol_m*1000:.1f}mm,'
                f'ang>{math.degrees(write_ang_tol_rad):.2f}°)  '
                f'FK_misses_total={ee_miss_total}'
            )
            _print_prompt()

            loop_tick_count_since_diag = 0
            loop_window_start_t = t_loop
            writes_since_diag = 0

        time.sleep(max(0.0, dt - (time.perf_counter() - t_loop)))


def _print_prompt() -> None:
    print('  > ', end='', flush=True)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description='World-frame arm commander — rate-limited writes, no null space.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument('--x', type=float, default=None, help='Initial sweet-spot world X (m).')
    ap.add_argument('--y', type=float, default=None, help='Initial sweet-spot world Y (m).')
    ap.add_argument('--z', type=float, default=None, help='Initial sweet-spot world Z (m).')
    ap.add_argument('--ori-rx', type=float, default=None, metavar='DEG')
    ap.add_argument('--ori-ry', type=float, default=None, metavar='DEG')
    ap.add_argument('--ori-rz', type=float, default=None, metavar='DEG')
    ap.add_argument('--rate-hz', type=float, default=50.0)
    ap.add_argument('--reach-m', type=float, default=FRANKA_DEFAULT_REACH_M)
    ap.add_argument('--z-min-m', type=float, default=FRANKA_DEFAULT_Z_MIN_M)
    ap.add_argument('--z-max-m', type=float, default=FRANKA_DEFAULT_Z_MAX_M)
    ap.add_argument('--robot-name', default='FrankaRobot')
    ap.add_argument('--calibration', default=None,
                    help='Path to arm_base_offset_calibration.json. '
                         'Default: canonical sports_bot/optitrack/ location.')
    ap.add_argument('--redis-host', default='localhost')
    ap.add_argument('--redis-port', type=int, default=6379)
    ap.add_argument('--write-pos-tol-mm', type=float, default=3.0,
                    help='Rewrite the goal when computed arm-frame position '
                         'drifts by more than this (mm) from the last write.')
    ap.add_argument('--write-ang-tol-deg', type=float, default=0.5,
                    help='Rewrite the goal when goal orientation drifts by more '
                         'than this (°) from the last write.')
    args = ap.parse_args()

    cal_path = args.calibration or arm_base_offset_calibration_path()
    if not os.path.isfile(cal_path):
        print(f'[cmd_arm] calibration not found: {cal_path}')
        print(f'[cmd_arm] Run: python sports_bot/scripts/calibrate_arm_base_offset.py')
        sys.exit(1)
    cal = load_arm_base_offset_calibration(cal_path)
    print(f'[cmd_arm] calibration: {cal_path}')
    yaw_offset_deg = math.degrees(math.atan2(float(cal.R_B_A[1, 0]), float(cal.R_B_A[0, 0])))
    print(f'[cmd_arm] base rigid body ID: {cal.base_rigid_body_id}  '
          f'yaw_offset={yaw_offset_deg:+.2f}°')

    r = redis.Redis(host=args.redis_host, port=args.redis_port,
                    decode_responses=True)
    try:
        r.ping()
    except redis.exceptions.ConnectionError as exc:
        print(f'[cmd_arm] cannot reach Redis at '
              f'{args.redis_host}:{args.redis_port}: {exc}')
        sys.exit(1)

    keys = OpenSaiCartesianKeys(robot_name=args.robot_name)

    # Verify cart rigid body is visible; compute initial T_W_A.
    T_W_B_init = read_rigid_body_pose_W(r, cal.base_rigid_body_id)
    if T_W_B_init is None:
        print(f'[cmd_arm] cart rigid body {cal.base_rigid_body_id} not visible in Redis. '
              f'Check OptiTrack streamer is running.')
        sys.exit(1)
    R_W_A_init, t_W_A_init = compute_T_W_A_from_base_offset(T_W_B_init, cal)
    yaw_init = math.degrees(
        math.atan2(float(R_W_A_init[1, 0]), float(R_W_A_init[0, 0])))
    print(f'[cmd_arm] arm base (world): '
          f'pos=[{t_W_A_init[0]:+.3f},{t_W_A_init[1]:+.3f},'
          f'{t_W_A_init[2]:+.3f}]  yaw={yaw_init:+.1f}°')

    ee_init = _read_ee_pose(r, keys)
    if ee_init is None:
        print(f'[cmd_arm] cannot read EE pose from Redis. '
              f'Is cartesian_controller running for "{args.robot_name}"?')
        sys.exit(1)
    R_A_E_init, t_A_E_init = ee_init
    t_W_ss_init = R_W_A_init @ t_A_E_init + t_W_A_init
    R_W_E_init = R_W_A_init @ R_A_E_init
    crx_init, cry_init, crz_init = _ori_deg_from_R_W_E(R_W_E_init)

    print(f'[cmd_arm] current sweet-spot (world): '
          f'[{t_W_ss_init[0]:+.3f},{t_W_ss_init[1]:+.3f},'
          f'{t_W_ss_init[2]:+.3f}]')
    print(f'[cmd_arm] current orientation equiv: '
          f'ori {crx_init:+.1f} {cry_init:+.1f} {crz_init:+.1f}')
    print(f'[cmd_arm] reference (ori 0 0 0): '
          f'face toward +X opponent, handle pointing down (-Z)')

    t_W_goal = np.array([
        args.x if args.x is not None else t_W_ss_init[0],
        args.y if args.y is not None else t_W_ss_init[1],
        args.z if args.z is not None else t_W_ss_init[2],
    ])
    rx_goal = args.ori_rx if args.ori_rx is not None else crx_init
    ry_goal = args.ori_ry if args.ori_ry is not None else cry_init
    rz_goal = args.ori_rz if args.ori_rz is not None else crz_init

    print(f'\n[cmd_arm] initial goal: '
          f'pos=[{t_W_goal[0]:+.3f},{t_W_goal[1]:+.3f},{t_W_goal[2]:+.3f}]'
          f'  ori=[{rx_goal:+.1f}°,{ry_goal:+.1f}°,{rz_goal:+.1f}°]')
    print(f'[cmd_arm] workspace (arm frame): '
          f'r≤{args.reach_m:.2f}m  '
          f'z∈[{args.z_min_m:+.2f},{args.z_max_m:+.2f}]m')
    print(f'[cmd_arm] rate: {args.rate_hz:.0f} Hz  |  '
          f'goal-write threshold: pos>{args.write_pos_tol_mm:.1f}mm '
          f'OR ang>{args.write_ang_tol_deg:.2f}°')
    print(f'[cmd_arm] no null-space task (cartesian_controller has '
          f'<motionForceTask> only; jointTask removed from picklebot.xml)')
    print(f'[cmd_arm] T_W_A source: rigid body {cal.base_rigid_body_id} + offset calibration')

    print()
    print('Commands (metres / degrees, world frame):')
    print('  x y z                      — absolute position')
    print('  r dx dy dz                 — relative position')
    print('  ori rx ry rz               — set orientation from reference (°)')
    print('  ori r drx dry drz          — nudge orientation by deltas (°)')
    print('  x y z rx ry rz             — position + set orientation')
    print('  r dx dy dz rx ry rz        — relative pos + set orientation')
    print('  x y z r drx dry drz        — position + nudge orientation')
    print('  r dx dy dz r drx dry drz   — relative pos + nudge orientation')
    print('  q                          — quit (last goals remain in Redis)')
    print()

    try:
        run(
            r=r,
            keys=keys,
            base_rb_id=cal.base_rigid_body_id,
            cal_offset=cal,
            t_W_goal=t_W_goal,
            rx_goal=rx_goal,
            ry_goal=ry_goal,
            rz_goal=rz_goal,
            rate_hz=args.rate_hz,
            reach_m=args.reach_m,
            z_min_m=args.z_min_m,
            z_max_m=args.z_max_m,
            robot_name=args.robot_name,
            write_pos_tol_m=args.write_pos_tol_mm / 1000.0,
            write_ang_tol_rad=math.radians(args.write_ang_tol_deg),
        )
    except KeyboardInterrupt:
        print('\n[cmd_arm] interrupted — last written goals remain in Redis.')


if __name__ == '__main__':
    main()
