#!/usr/bin/env python3
"""cmd_arm_world_clean.py — Clean world-frame arm commander for pickleball robot.

Prereqs: Redis, OptiTrack streamer (labeled-marker streaming enabled in
Motive), OpenSai cartesian_controller running (picklebot.xml or equivalent).

IMPORTANT — compliantFrame assumption
--------------------------------------
This script assumes picklebot.xml has::

    <compliantFrame xyz="0 0 0.35" rpy="0 0 0" />

With that setting, OpenSai's MotionForceTask reports and accepts the sweet-spot
position (35 cm along EE +Z from the flange) directly via::

    current_position / goal_position   ← sweet-spot in arm base frame A
    current_orientation / goal_orientation ← EE frame orientation in A

T_E_P is intentionally NOT used here.  After the compliantFrame fix the
controller already tracks the sweet spot, so there is no further offset to apply.

Frame conventions
-----------------
  W   world frame  (+X toward opponent, +Y left, +Z up; floor tape origin)
  A   Franka arm base frame  (derived from two OptiTrack markers every tick)
  E   EE / compliant frame   (at sweet spot; +X = face normal, +Z = handle)

Reference orientation (ori 0 0 0)
-----------------------------------
  Face toward opponent  (+X world)
  Handle pointing down  (-Z world)

  Internally: R_W_E_ref = Rx(180°) = diag(1, -1, -1)

ori command convention
-----------------------
  ori rx ry rz  (degrees, extrinsic rotations around fixed world axes,
                 applied on top of the reference)

  R_W_E_goal = Rz(rz) @ Ry(ry) @ Rx(rx) @ R_W_E_ref
  R_A_E_goal = R_W_A.T @ R_W_E_goal

  Physical meaning of small deltas from ori 0 0 0:
    rx  — handle sways sideways  (roll around face-normal / world +X axis)
    ry  — face pitches down (-) / up (+) from horizontal
    rz  — face yaws toward +Y (left) / -Y (right) in horizontal plane

  At startup the script reads the arm's current orientation from Redis and
  displays its equivalent "ori rx ry rz" so you know where you are.
  The arm holds that orientation until you send the first ori command.

Commands (all metres / degrees, world frame)
--------------------------------------------
  x y z                   absolute position; keep orientation
  r dx dy dz              relative position delta; keep orientation
  ori rx ry rz            set orientation (° from reference); keep position
  x y z rx ry rz          set position + orientation
  r dx dy dz rx ry rz     relative position + set orientation
  q / quit                exit (last goals remain in Redis)

Debug output
------------
  Every 2 s a summary is printed:
    [T_W_A]   — live arm-base world position and yaw
    [goal_W]  — current world-frame position + orientation targets
    [goal_A]  — computed arm-frame position goal (+ CLIPPED flag if hit limit)
    [current] — sweet-spot world position from FK, position error, angle error,
                and current orientation expressed as "ori rx ry rz"

  Marker warnings printed immediately whenever a marker is missing or jumps
  more than 4 cm in a single 20 ms tick (likely OptiTrack ID swap or dropout).
  The script still uses whatever T_W_A it can compute; warnings are diagnostic
  only, not suppressive.

Examples
--------
  # Start at current arm pose:
  python sports_bot/scripts/cmd_arm_world_clean.py

  # Start at specific world XY, read Z from current pose:
  python sports_bot/scripts/cmd_arm_world_clean.py --x 0.5 --y 0.0

  # At the prompt:
  0.5 0.0 0.9             # move sweet spot to (0.5, 0, 0.9) world
  r 0.0 0.0 -0.1          # nudge 10 cm down
  ori 0 10 0              # tilt face 10° downward from reference
  0.5 0.0 0.9 0 10 0      # move + set orientation simultaneously
  q                       # quit
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
from typing import List, Optional, Tuple

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
    arm_marker_calibration_path,
    clip_to_arm_workspace,
    load_arm_marker_calibration,
    read_marker_position_W,
)

# ---------------------------------------------------------------------------
# Reference orientation
# ---------------------------------------------------------------------------
# Face toward +X opponent, handle pointing down (-Z world).
# This equals Rx(180°) = diag(1, -1, -1).
# ori 0 0 0 reproduces this exactly; all RPY commands are applied on top.
R_W_E_REF: np.ndarray = np.array([
    [1.,  0.,  0.],
    [0., -1.,  0.],
    [0.,  0., -1.],
])

# ---------------------------------------------------------------------------
# Marker jump / missing detection
# ---------------------------------------------------------------------------
_MARKER_JUMP_M = 0.04    # position jump threshold per tick (m); 4 cm in 20 ms
                          # is physically impossible for the cart
_WARN_INTERVAL_S = 1.0   # suppress repeated warnings within this window (s)

# ---------------------------------------------------------------------------
# Rotation helpers
# ---------------------------------------------------------------------------

def _Rx(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1., 0., 0.],
                     [0.,  c, -s],
                     [0.,  s,  c]])


def _Ry(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[ c, 0., s],
                     [0., 1., 0.],
                     [-s, 0., c]])


def _Rz(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.],
                     [s,  c, 0.],
                     [0., 0., 1.]])


def _R_from_rpy_rad(rx: float, ry: float, rz: float) -> np.ndarray:
    """Extrinsic world-frame RPY: R = Rz(rz) @ Ry(ry) @ Rx(rx)."""
    return _Rz(rz) @ _Ry(ry) @ _Rx(rx)


def _rpy_deg_from_R(R: np.ndarray) -> Tuple[float, float, float]:
    """Extract (rx, ry, rz) in degrees from R = Rz(rz)@Ry(ry)@Rx(rx).

    Standard ZYX Euler decomposition.  Handles gimbal lock at ry = ±90°
    by fixing rz = 0 and solving rx only (rare in practice for paddle poses).
    """
    # ry from R[2,0] = -sin(ry)
    sy = float(-R[2, 0])
    cy = math.sqrt(max(0.0, float(R[2, 1])**2 + float(R[2, 2])**2))
    ry = math.atan2(sy, cy)
    if cy < 1e-6:
        # Gimbal lock: fix rz = 0, solve rx only.
        rx = math.atan2(-float(R[1, 2]), float(R[1, 1]))
        rz = 0.0
    else:
        rx = math.atan2(float(R[2, 1]) / cy, float(R[2, 2]) / cy)
        rz = math.atan2(float(R[1, 0]) / cy, float(R[0, 0]) / cy)
    return math.degrees(rx), math.degrees(ry), math.degrees(rz)


def _ori_deg_from_R_W_E(R_W_E: np.ndarray) -> Tuple[float, float, float]:
    """Return the (rx, ry, rz) degrees that reproduce R_W_E from the reference:
        R_W_E = Rz(rz)@Ry(ry)@Rx(rx) @ R_W_E_ref
    Equivalent: extract RPY of R_W_E @ R_W_E_ref.T."""
    return _rpy_deg_from_R(R_W_E @ R_W_E_REF.T)


# ---------------------------------------------------------------------------
# Redis helpers
# ---------------------------------------------------------------------------

def _read_ee_pose(
    r: redis.Redis, keys: OpenSaiCartesianKeys
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Read current EE (sweet-spot) pose from Redis.
    Returns (R_A_E, t_A_E) or None if keys are absent / malformed."""
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


def _write_goal(
    r: redis.Redis, keys: OpenSaiCartesianKeys,
    t_A: np.ndarray, R_A_E: np.ndarray,
) -> None:
    """Write sweet-spot position + orientation goals to Redis."""
    r.set(keys.goal_position,
          json.dumps([float(v) for v in t_A]))
    r.set(keys.goal_orientation,
          json.dumps([[float(v) for v in row] for row in R_A_E]))
    r.set(keys.goal_linear_velocity,
          json.dumps([0.0, 0.0, 0.0]))


# ---------------------------------------------------------------------------
# Command parsing
# ---------------------------------------------------------------------------
# Return type: (kind, t_delta_or_None, is_relative, rx_or_None, ry, rz)
#   kind in {'pos', 'ori', 'both'}
_ParseResult = Tuple[str, Optional[np.ndarray], bool,
                     Optional[float], Optional[float], Optional[float]]


def _parse_cmd(line: str) -> Optional[_ParseResult]:
    """Parse one user command line.

    Accepted formats::

      x y z                  → ('pos', [x,y,z], False, None,  None, None)
      r dx dy dz             → ('pos', [dx,dy,dz], True, None,  None, None)
      ori rx ry rz           → ('ori', None, False, rx, ry, rz)
      x y z rx ry rz         → ('both', [x,y,z], False, rx, ry, rz)
      r dx dy dz rx ry rz    → ('both', [dx,dy,dz], True,  rx, ry, rz)
    """
    parts = line.split()
    if not parts:
        return None

    # --- orientation-only ---
    if parts[0].lower() == 'ori':
        if len(parts) != 4:
            return None
        try:
            rx, ry, rz = float(parts[1]), float(parts[2]), float(parts[3])
            return ('ori', None, False, rx, ry, rz)
        except ValueError:
            return None

    # --- position (+ optional orientation) ---
    relative = parts[0].lower() in ('r', 'rel')
    nums = parts[1:] if relative else parts[:]
    try:
        if len(nums) == 3:
            t = np.array([float(v) for v in nums])
            return ('pos', t, relative, None, None, None)
        if len(nums) == 6:
            t = np.array([float(v) for v in nums[:3]])
            rx, ry, rz = float(nums[3]), float(nums[4]), float(nums[5])
            return ('both', t, relative, rx, ry, rz)
    except ValueError:
        pass
    return None


# ---------------------------------------------------------------------------
# Stdin reader thread
# ---------------------------------------------------------------------------

def _stdin_reader(q: 'queue.Queue[str]') -> None:
    try:
        for line in sys.stdin:
            q.put(line.strip())
    except EOFError:
        pass
    q.put('q')


# ---------------------------------------------------------------------------
# Main control loop
# ---------------------------------------------------------------------------

def run(
    r: redis.Redis,
    keys: OpenSaiCartesianKeys,
    marker_specs: List,
    t_W_goal: np.ndarray,
    rx_goal: float, ry_goal: float, rz_goal: float,
    rate_hz: float,
    reach_m: float,
    z_min_m: float,
    z_max_m: float,
) -> None:

    cmd_q: 'queue.Queue[str]' = queue.Queue()
    threading.Thread(
        target=_stdin_reader, args=(cmd_q,), daemon=True
    ).start()

    dt = 1.0 / rate_hz
    last_debug_t: float = -math.inf
    debug_interval_s = 2.0

    # Per-marker tracking for jump detection.
    prev_pos: List[Optional[np.ndarray]] = [None, None]
    last_warn_t: List[float] = [-math.inf, -math.inf]
    last_both_ok_t: float = -math.inf   # last time both markers were seen

    _print_prompt()

    while True:
        t_loop = time.perf_counter()

        # ----------------------------------------------------------------
        # 1. Drain user commands
        # ----------------------------------------------------------------
        try:
            while True:
                line = cmd_q.get_nowait()
                if line.lower() in ('q', 'quit', 'exit'):
                    print('\n[cmd_arm] exit — last goals remain in Redis.')
                    return
                if line == '':
                    _print_prompt()
                    continue

                parsed = _parse_cmd(line)
                if parsed is None:
                    print(f'[cmd_arm] unrecognised: "{line}"')
                    print('  x y z                 — absolute position')
                    print('  r dx dy dz            — relative position')
                    print('  ori rx ry rz          — orientation (° from ref)')
                    print('  x y z rx ry rz        — position + orientation')
                    print('  r dx dy dz rx ry rz   — relative pos + orientation')
                    _print_prompt()
                    continue

                kind, t_delta, is_rel, rx_new, ry_new, rz_new = parsed

                if kind in ('pos', 'both'):
                    assert t_delta is not None
                    if is_rel:
                        t_W_goal = t_W_goal + t_delta
                    else:
                        t_W_goal = t_delta.copy()

                if kind in ('ori', 'both'):
                    rx_goal = rx_new  # type: ignore[assignment]
                    ry_goal = ry_new  # type: ignore[assignment]
                    rz_goal = rz_new  # type: ignore[assignment]

                rel_tag = '  (relative)' if is_rel else ''
                print(
                    f'[cmd_arm] goal_W  pos=[{t_W_goal[0]:+.3f},'
                    f'{t_W_goal[1]:+.3f},{t_W_goal[2]:+.3f}]'
                    f'  ori=[{rx_goal:+.1f}°,{ry_goal:+.1f}°,'
                    f'{rz_goal:+.1f}°]{rel_tag}'
                )
                _print_prompt()

        except queue.Empty:
            pass

        # ----------------------------------------------------------------
        # 2. Read individual markers + jump / missing detection
        # ----------------------------------------------------------------
        p: List[Optional[np.ndarray]] = [None, None]
        for i, spec in enumerate(marker_specs):
            pos = read_marker_position_W(r, spec)
            p[i] = pos

            if pos is None:
                absent_s = (t_loop - last_both_ok_t
                            if last_both_ok_t > -math.inf else 0.0)
                if t_loop - last_warn_t[i] > _WARN_INTERVAL_S:
                    last_warn_t[i] = t_loop
                    print(
                        f'[WARN] marker {spec} missing'
                        + (f' (both absent {absent_s:.1f}s)'
                           if absent_s > 0.5 else '')
                    )
            else:
                prev = prev_pos[i]
                if prev is not None:
                    jump = float(np.linalg.norm(pos - prev))
                    if jump > _MARKER_JUMP_M:
                        if t_loop - last_warn_t[i] > _WARN_INTERVAL_S:
                            last_warn_t[i] = t_loop
                            print(
                                f'[WARN] marker {spec} jumped '
                                f'{jump * 100:.1f} cm in one tick '
                                f'— possible ID swap or dropout'
                            )
                prev_pos[i] = pos.copy()

        # ----------------------------------------------------------------
        # 3. Build T_W_A from markers
        # ----------------------------------------------------------------
        if p[0] is None or p[1] is None:
            # Can't compute arm base pose — skip write, hold last Redis goal.
            time.sleep(max(0.0, dt - (time.perf_counter() - t_loop)))
            continue

        last_both_ok_t = t_loop

        # Midpoint = arm base origin.
        t_W_A: np.ndarray = 0.5 * (p[0] + p[1])

        # spec[0] → spec[1] direction (horizontally projected) = Franka +Y.
        y_raw = p[1] - p[0]
        z_world = np.array([0.0, 0.0, 1.0])
        y_horiz = y_raw - float(np.dot(y_raw, z_world)) * z_world
        ny = float(np.linalg.norm(y_horiz))
        if ny < 1e-6:
            # Markers stacked vertically — degenerate; can't recover +Y axis.
            if t_loop - last_warn_t[0] > _WARN_INTERVAL_S:
                last_warn_t[0] = t_loop
                print('[WARN] markers nearly co-vertical — cannot form T_W_A')
            time.sleep(max(0.0, dt - (time.perf_counter() - t_loop)))
            continue

        y_hat = y_horiz / ny
        x_hat = np.cross(y_hat, z_world)  # right-handed: X = Y × Z
        R_W_A: np.ndarray = np.column_stack([x_hat, y_hat, z_world])

        # ----------------------------------------------------------------
        # 4. Position goal: world → arm frame
        # ----------------------------------------------------------------
        t_A_raw = R_W_A.T @ (t_W_goal - t_W_A)
        t_A_goal, was_clipped = clip_to_arm_workspace(
            t_A_raw, r_max=reach_m, z_min=z_min_m, z_max=z_max_m)

        # ----------------------------------------------------------------
        # 5. Orientation goal: RPY from reference → arm frame
        # ----------------------------------------------------------------
        R_W_E_goal = (_R_from_rpy_rad(
            math.radians(rx_goal),
            math.radians(ry_goal),
            math.radians(rz_goal),
        ) @ R_W_E_REF)
        R_A_E_goal = R_W_A.T @ R_W_E_goal

        # ----------------------------------------------------------------
        # 6. Write both goals to Redis (every tick)
        # ----------------------------------------------------------------
        _write_goal(r, keys, t_A_goal, R_A_E_goal)

        # ----------------------------------------------------------------
        # 7. Debug summary every 2 s
        # ----------------------------------------------------------------
        if t_loop - last_debug_t >= debug_interval_s:
            last_debug_t = t_loop

            yaw_deg = math.degrees(math.atan2(
                float(R_W_A[1, 0]), float(R_W_A[0, 0])))
            clip_tag = '  [CLIPPED]' if was_clipped else ''

            # Current sweet-spot from FK (Redis current_position).
            ee = _read_ee_pose(r, keys)
            if ee is not None:
                R_A_E_now, t_A_E_now = ee
                t_W_ss = R_W_A @ t_A_E_now + t_W_A
                pos_err_m = float(np.linalg.norm(t_W_ss - t_W_goal))

                # Orientation error: angle between goal and current R_A_E.
                dR = R_A_E_goal @ R_A_E_now.T
                cos_th = max(-1.0, min(1.0,
                             (float(np.trace(dR)) - 1.0) / 2.0))
                ang_err_deg = math.degrees(math.acos(cos_th))

                # Current orientation as equivalent ori degrees.
                R_W_E_now = R_W_A @ R_A_E_now
                crx, cry, crz = _ori_deg_from_R_W_E(R_W_E_now)

                cur_line = (
                    f'[current] sweet_W=['
                    f'{t_W_ss[0]:+.3f},{t_W_ss[1]:+.3f},{t_W_ss[2]:+.3f}]'
                    f'  ori=[{crx:+.1f}°,{cry:+.1f}°,{crz:+.1f}°]'
                    f'  pos_err={pos_err_m * 1000:.1f}mm'
                    f'  ang_err={ang_err_deg:.1f}°'
                )
            else:
                cur_line = '[current] EE pose unavailable (FK keys missing)'

            print(
                f'\n[T_W_A]   pos=[{t_W_A[0]:+.3f},{t_W_A[1]:+.3f},'
                f'{t_W_A[2]:+.3f}]  yaw={yaw_deg:+.1f}°\n'
                f'[goal_W]  pos=[{t_W_goal[0]:+.3f},{t_W_goal[1]:+.3f},'
                f'{t_W_goal[2]:+.3f}]'
                f'  ori=[{rx_goal:+.1f}°,{ry_goal:+.1f}°,{rz_goal:+.1f}°]\n'
                f'[goal_A]  pos=[{t_A_goal[0]:+.3f},{t_A_goal[1]:+.3f},'
                f'{t_A_goal[2]:+.3f}]{clip_tag}\n'
                f'{cur_line}'
            )
            _print_prompt()

        time.sleep(max(0.0, dt - (time.perf_counter() - t_loop)))


def _print_prompt() -> None:
    print('  > ', end='', flush=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description='World-frame arm commander — clean, no filtering.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument('--x', type=float, default=None,
                    help='Initial sweet-spot world X (m). '
                         'Default: read from current arm pose.')
    ap.add_argument('--y', type=float, default=None,
                    help='Initial sweet-spot world Y (m). '
                         'Default: read from current arm pose.')
    ap.add_argument('--z', type=float, default=None,
                    help='Initial sweet-spot world Z (m). '
                         'Default: read from current arm pose.')
    ap.add_argument('--ori-rx', type=float, default=None, metavar='DEG',
                    help='Initial roll  (°). Default: current arm orientation.')
    ap.add_argument('--ori-ry', type=float, default=None, metavar='DEG',
                    help='Initial pitch (°). Default: current arm orientation.')
    ap.add_argument('--ori-rz', type=float, default=None, metavar='DEG',
                    help='Initial yaw   (°). Default: current arm orientation.')
    ap.add_argument('--rate-hz', type=float, default=50.0,
                    help='Control-loop rate (Hz).')
    ap.add_argument('--reach-m', type=float, default=FRANKA_DEFAULT_REACH_M,
                    help='Max XY reach from arm base (m, arm frame).')
    ap.add_argument('--z-min-m', type=float, default=FRANKA_DEFAULT_Z_MIN_M,
                    help='Min EE height (m, arm frame).')
    ap.add_argument('--z-max-m', type=float, default=FRANKA_DEFAULT_Z_MAX_M,
                    help='Max EE height (m, arm frame).')
    ap.add_argument('--robot-name', default='FrankaRobot',
                    help='Robot name in Redis key namespace.')
    ap.add_argument('--calibration', default=None,
                    help='Path to arm_marker_calibration.json. '
                         'Default: canonical sports_bot/optitrack/ location.')
    ap.add_argument('--redis-host', default='localhost')
    ap.add_argument('--redis-port', type=int, default=6379)
    args = ap.parse_args()

    # ------------------------------------------------------------------
    # Load calibration (only marker_specs needed; T_E_P not used).
    # ------------------------------------------------------------------
    cal_path = args.calibration or arm_marker_calibration_path()
    if not os.path.isfile(cal_path):
        print(f'[cmd_arm] calibration not found: {cal_path}')
        sys.exit(1)
    cal = load_arm_marker_calibration(cal_path)
    print(f'[cmd_arm] calibration: {cal_path}')
    print(f'[cmd_arm] marker specs: {cal.marker_specs}  '
          f'(spec[0]=left/-Y side, spec[1]=right/+Y side)')

    # ------------------------------------------------------------------
    # Redis connection.
    # ------------------------------------------------------------------
    r = redis.Redis(host=args.redis_host, port=args.redis_port,
                    decode_responses=True)
    try:
        r.ping()
    except redis.exceptions.ConnectionError as exc:
        print(f'[cmd_arm] cannot reach Redis at '
              f'{args.redis_host}:{args.redis_port}: {exc}')
        sys.exit(1)

    keys = OpenSaiCartesianKeys(robot_name=args.robot_name)

    # ------------------------------------------------------------------
    # Initial arm base pose from markers.
    # ------------------------------------------------------------------
    # Import here so the module works even if frames is partially loaded.
    from sports_bot.utils.frames import compute_T_W_A_from_markers  # noqa

    T_W_A_init = compute_T_W_A_from_markers(r, cal.marker_specs)
    if T_W_A_init is None:
        print('[cmd_arm] arm base markers not visible. '
              'Check OptiTrack streamer and Motive labeled-marker setting.')
        sys.exit(1)
    R_W_A_init, t_W_A_init = T_W_A_init
    yaw_init = math.degrees(
        math.atan2(float(R_W_A_init[1, 0]), float(R_W_A_init[0, 0])))
    print(f'[cmd_arm] arm base (world): '
          f'pos=[{t_W_A_init[0]:+.3f},{t_W_A_init[1]:+.3f},'
          f'{t_W_A_init[2]:+.3f}]  yaw={yaw_init:+.1f}°')

    # ------------------------------------------------------------------
    # Initial EE pose from Redis (no startup lurch).
    # ------------------------------------------------------------------
    ee_init = _read_ee_pose(r, keys)
    if ee_init is None:
        print(f'[cmd_arm] cannot read EE pose from Redis. '
              f'Is the cartesian_controller running for "{args.robot_name}"?')
        sys.exit(1)
    R_A_E_init, t_A_E_init = ee_init

    # Current sweet-spot world position.
    t_W_ss_init = R_W_A_init @ t_A_E_init + t_W_A_init

    # Current world-frame orientation as equivalent "ori rx ry rz".
    R_W_E_init = R_W_A_init @ R_A_E_init
    crx_init, cry_init, crz_init = _ori_deg_from_R_W_E(R_W_E_init)

    print(f'[cmd_arm] current sweet-spot (world): '
          f'[{t_W_ss_init[0]:+.3f},{t_W_ss_init[1]:+.3f},'
          f'{t_W_ss_init[2]:+.3f}]')
    print(f'[cmd_arm] current orientation equiv: '
          f'ori {crx_init:+.1f} {cry_init:+.1f} {crz_init:+.1f}')
    print(f'[cmd_arm] reference (ori 0 0 0): '
          f'face toward +X opponent, handle pointing down (-Z)')

    # ------------------------------------------------------------------
    # Initial goals: current pose unless CLI args override.
    # ------------------------------------------------------------------
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
          f'marker jump threshold: {_MARKER_JUMP_M * 100:.0f} cm/tick')
    print()
    print('Commands (metres / degrees, world frame):')
    print('  x y z                   — absolute position')
    print('  r dx dy dz              — relative position')
    print('  ori rx ry rz            — orientation from reference (°)')
    print('  x y z rx ry rz          — position + orientation')
    print('  r dx dy dz rx ry rz     — relative position + orientation')
    print('  q                       — quit (last goals remain in Redis)')
    print()

    try:
        run(
            r=r,
            keys=keys,
            marker_specs=cal.marker_specs,
            t_W_goal=t_W_goal,
            rx_goal=rx_goal,
            ry_goal=ry_goal,
            rz_goal=rz_goal,
            rate_hz=args.rate_hz,
            reach_m=args.reach_m,
            z_min_m=args.z_min_m,
            z_max_m=args.z_max_m,
        )
    except KeyboardInterrupt:
        print('\n[cmd_arm] interrupted — last goals remain in Redis.')


if __name__ == '__main__':
    main()
