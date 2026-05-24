#!/usr/bin/env python3
"""Command the Franka arm to world-frame paddle sweet-spot positions.

Runs a continuous tracking loop (cart-motion compensation via OptiTrack markers)
and lets you type new target positions at any time. The arm moves to the new
world-frame target and holds it, compensating for cart drift.

Prompt usage
------------
  x y z              move sweet spot to (x,y,z) world frame (absolute), face toward +X
  x y z nx ny nz     same, with explicit face normal (nx, ny, nz)
  r dx dy dz         move sweet spot by (dx,dy,dz) relative to current goal
  r dx dy dz nx ny nz  relative move with explicit face normal
  q / quit           stop

All units in metres, world frame:
  +X  toward opponent
  +Y  left
  +Z  up (origin at court floor)

Examples
--------
  # Interactive (default):
  python sports_bot/scripts/cmd_arm_world.py

  # Start at a specific position:
  python sports_bot/scripts/cmd_arm_world.py --x 0.5 --y 0.0 --z 0.9

  # At the prompt — absolute then relative:
  0.5 0.0 0.9          # go to world (0.5, 0, 0.9)
  r 0.1 0 0            # nudge +10 cm in X
  r 0 0 -0.05          # nudge -5 cm in Z
  r 0 0.1 0 0 1 0      # nudge +10 cm in Y, change face to +Y

Prereqs: Redis, OptiTrack streamer, Franka driver, OpenSai cartesian_controller.
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
from typing import Optional

import numpy as np
import redis

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
    SE3,
    arm_marker_calibration_path,
    clip_to_arm_workspace,
    compute_T_W_A_from_markers,
    enforce_paddle_floor,
    load_arm_marker_calibration,
    se3_compose,
    world_racket_to_arm_ee,
)

_MARKER_DROPOUT_RESET_S = 3.0   # re-seed EMA if markers absent this long
_MARKER_MAX_JUMP_M = 0.10       # discard T_W_A reading if position jumps this far


# ---------- Helpers -----------------------------------------------------------

def _R_from_face_normal(n: np.ndarray) -> np.ndarray:
    """Build R_W_P with face normal as +Z_P (SwingPlanner convention)."""
    n = n / max(1e-9, float(np.linalg.norm(n)))
    world_up = np.array([0.0, 0.0, 1.0])
    up_proj = world_up - float(np.dot(world_up, n)) * n
    if float(np.linalg.norm(up_proj)) < 1e-6:
        ref = np.array([1.0, 0.0, 0.0])
        up_proj = ref - float(np.dot(ref, n)) * n
    face_up = up_proj / float(np.linalg.norm(up_proj))
    face_right = np.cross(face_up, n)
    return np.column_stack([face_right, face_up, n])


def _read_T_A_E(r: redis.Redis, keys: OpenSaiCartesianKeys) -> Optional[SE3]:
    pos_raw = r.get(keys.current_position)
    ori_raw = r.get(keys.current_orientation)
    if pos_raw is None or ori_raw is None:
        return None
    try:
        pos = np.asarray(json.loads(pos_raw), dtype=float)
        ori = np.asarray(json.loads(ori_raw), dtype=float)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    if pos.shape != (3,) or ori.shape != (3, 3):
        return None
    return (ori, pos)


def _write_arm_goal(r: redis.Redis, keys: OpenSaiCartesianKeys, T_A_E: SE3) -> None:
    R, t = T_A_E
    r.set(keys.goal_position, json.dumps(t.tolist()))
    r.set(keys.goal_orientation,
          json.dumps([[float(v) for v in row] for row in R]))
    r.set(keys.goal_linear_velocity, json.dumps([0.0, 0.0, 0.0]))


def _parse_target(line: str) -> Optional[tuple[np.ndarray, np.ndarray, bool]]:
    """Parse a position command. Returns (t_W_P, face_normal, is_relative) or None.

    Formats accepted:
      x y z              absolute position, default face normal (+X)
      x y z nx ny nz     absolute position, explicit face normal
      r x y z            relative delta, default face normal
      r x y z nx ny nz   relative delta, explicit face normal
    """
    parts = line.split()
    if not parts:
        return None
    relative = parts[0].lower() in ("r", "rel")
    nums = parts[1:] if relative else parts
    try:
        if len(nums) == 3:
            t = np.array([float(p) for p in nums])
            n = np.array([1.0, 0.0, 0.0])
            return t, n, relative
        elif len(nums) == 6:
            t = np.array([float(p) for p in nums[:3]])
            n = np.array([float(p) for p in nums[3:]])
            return t, n, relative
    except ValueError:
        pass
    return None


# ---------- Stdin reader thread -----------------------------------------------

def _stdin_reader(cmd_q: "queue.Queue[str]") -> None:
    """Reads lines from stdin and puts them on cmd_q. Runs in a daemon thread."""
    try:
        for line in sys.stdin:
            cmd_q.put(line.strip())
    except EOFError:
        pass
    cmd_q.put("q")


# ---------- Main loop ---------------------------------------------------------

def run(
    r: redis.Redis,
    keys: OpenSaiCartesianKeys,
    cal,
    initial_T_W_P: SE3,
    rate_hz: float,
    reach_m: float,
    z_min: float,
    z_max: float,
    filter_alpha: float,
    goal_deadband_m: float,
) -> None:
    cmd_q: queue.Queue[str] = queue.Queue()
    t_stdin = threading.Thread(target=_stdin_reader, args=(cmd_q,), daemon=True)
    t_stdin.start()

    T_W_P_goal = initial_T_W_P
    R_g, t_g = T_W_P_goal
    print(f"\n[cmd_arm] holding sweet spot at world t="
          f"[{t_g[0]:+.3f}, {t_g[1]:+.3f}, {t_g[2]:+.3f}]")
    _print_prompt()

    dt = 1.0 / rate_hz
    last_warn = -math.inf
    last_print = -math.inf
    print_interval_s = 2.0

    t_smooth: Optional[np.ndarray] = None
    R_smooth: Optional[np.ndarray] = None
    last_sent_t: Optional[np.ndarray] = None
    last_marker_t: float = -math.inf

    while True:
        t_loop = time.perf_counter()

        # --- Check for new user command ---
        try:
            while True:
                cmd = cmd_q.get_nowait()
                if cmd.lower() in ("q", "quit", "exit"):
                    print("[cmd_arm] stopping.")
                    return
                if cmd == "":
                    _print_prompt()
                    continue
                parsed = _parse_target(cmd)
                if parsed is None:
                    print(f"[cmd_arm] unrecognised: '{cmd}' — "
                          f"expected 'x y z', 'r dx dy dz', "
                          f"'x y z nx ny nz', 'r dx dy dz nx ny nz', or 'q'")
                    _print_prompt()
                else:
                    t_new, n_new, is_relative = parsed
                    if is_relative:
                        t_new = T_W_P_goal[1] + t_new
                    T_W_P_goal = (_R_from_face_normal(n_new), t_new)
                    last_sent_t = None  # force immediate write to new target
                    tag = "relative→" if is_relative else ""
                    print(f"[cmd_arm] new target ({tag}world t="
                          f"[{t_new[0]:+.3f}, {t_new[1]:+.3f}, {t_new[2]:+.3f}])  "
                          f"face_normal=[{n_new[0]:+.2f},{n_new[1]:+.2f},"
                          f"{n_new[2]:+.2f}]")
                    _print_prompt()
        except queue.Empty:
            pass

        # --- Tracking loop ---
        T_W_A_raw = compute_T_W_A_from_markers(r, cal.marker_specs)
        if T_W_A_raw is None:
            if t_loop - last_warn > 1.0:
                last_warn = t_loop
                absent = (t_loop - last_marker_t
                          if last_marker_t > -math.inf else 0.0)
                print(f"[cmd_arm] markers missing ({absent:.0f}s) — "
                      f"holding last goal.")
        else:
            R_raw, t_raw = T_W_A_raw
            dropout = (t_smooth is None or
                       t_loop - last_marker_t > _MARKER_DROPOUT_RESET_S)
            if (not dropout and
                    float(np.linalg.norm(t_raw - t_smooth)) > _MARKER_MAX_JUMP_M):
                # Looks like a bad frame or Motive marker-ID swap — discard.
                if t_loop - last_warn > 1.0:
                    last_warn = t_loop
                    delta = float(np.linalg.norm(t_raw - t_smooth))
                    print(f"[cmd_arm] T_W_A jumped {delta*100:.1f} cm in one tick "
                          f"— discarding bad frame, EMA resets on next valid read.")
                last_marker_t = -math.inf
            else:
                if dropout:
                    t_smooth = t_raw.copy()
                    R_smooth = R_raw.copy()
                    last_sent_t = None  # force write after re-seed
                else:
                    t_smooth = (filter_alpha * t_raw
                                + (1.0 - filter_alpha) * t_smooth)
                    R_smooth = (filter_alpha * R_raw
                                + (1.0 - filter_alpha) * R_smooth)
                last_marker_t = t_loop
                T_W_A = (R_smooth, t_smooth)

                R_A_E, t_A_E_raw = world_racket_to_arm_ee(
                    T_W_P_goal, T_W_A, cal.T_E_P)
                t_A_E_floored, floor_adj = enforce_paddle_floor(
                    t_A_E_raw, R_A_E, T_W_A)
                t_A_E_clipped, was_clipped = clip_to_arm_workspace(
                    t_A_E_floored, r_max=reach_m, z_min=z_min, z_max=z_max)
                was_clipped = was_clipped or floor_adj

                moved = (last_sent_t is None or
                         float(np.linalg.norm(t_A_E_clipped - last_sent_t))
                         > goal_deadband_m)
                if moved:
                    _write_arm_goal(r, keys, (R_A_E, t_A_E_clipped))
                    last_sent_t = t_A_E_clipped.copy()

                if t_loop - last_print >= print_interval_s:
                    last_print = t_loop
                    R_g, t_g = T_W_P_goal
                    tag = " CLIPPED" if was_clipped else ""
                    print(f"[cmd_arm] cart=[{t_smooth[0]:+.3f},{t_smooth[1]:+.3f},"
                          f"{t_smooth[2]:+.3f}]  "
                          f"target_W=[{t_g[0]:+.3f},{t_g[1]:+.3f},{t_g[2]:+.3f}]  "
                          f"goal_A=[{t_A_E_clipped[0]:+.3f},"
                          f"{t_A_E_clipped[1]:+.3f},"
                          f"{t_A_E_clipped[2]:+.3f}]{tag}")

        elapsed = time.perf_counter() - t_loop
        sleep_t = dt - elapsed
        if sleep_t > 0:
            time.sleep(sleep_t)


def _print_prompt() -> None:
    print("  > ", end="", flush=True)


# ---------- Entry point -------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--x", type=float, default=None,
                   help="Initial sweet-spot world X (m). If omitted, uses "
                        "the arm's current world X.")
    p.add_argument("--y", type=float, default=None,
                   help="Initial sweet-spot world Y (m). If omitted, uses "
                        "the arm's current world Y.")
    p.add_argument("--z", type=float, default=None,
                   help="Initial sweet-spot world Z (m). If omitted, uses "
                        "the arm's current world Z.")
    p.add_argument("--face-normal-x", type=float, default=1.0)
    p.add_argument("--face-normal-y", type=float, default=0.0)
    p.add_argument("--face-normal-z", type=float, default=0.0)
    p.add_argument("--rate-hz", type=float, default=50.0)
    p.add_argument("--goal-deadband-mm", type=float, default=3.0,
                   help="Only write a new arm goal when the arm-frame position "
                        "has moved more than this many mm from the last sent "
                        "goal. Suppresses vibration from residual marker noise.")
    p.add_argument("--marker-filter-alpha", type=float, default=0.15,
                   help="EMA smoothing on T_W_A (0<α≤1; lower=smoother). "
                        "Default 0.15 ≈ 130ms time constant at 50Hz.")
    p.add_argument("--reach-m", type=float, default=FRANKA_DEFAULT_REACH_M)
    p.add_argument("--z-min-m", type=float, default=FRANKA_DEFAULT_Z_MIN_M)
    p.add_argument("--z-max-m", type=float, default=FRANKA_DEFAULT_Z_MAX_M)
    p.add_argument("--robot-name", default="FrankaRobot")
    p.add_argument("--calibration", default=None)
    p.add_argument("--redis-host", default="localhost")
    p.add_argument("--redis-port", type=int, default=6379)
    args = p.parse_args()

    cal_path = args.calibration or arm_marker_calibration_path()
    if not os.path.isfile(cal_path):
        print(f"[cmd_arm] missing calibration at {cal_path}.")
        sys.exit(1)
    cal = load_arm_marker_calibration(cal_path)
    print(f"[cmd_arm] loaded {cal_path}")

    keys = OpenSaiCartesianKeys(robot_name=args.robot_name)
    r = redis.Redis(host=args.redis_host, port=args.redis_port,
                    decode_responses=True)
    try:
        r.ping()
    except redis.exceptions.ConnectionError as e:
        print(f"[cmd_arm] cannot reach Redis: {e}")
        sys.exit(1)

    # Check markers visible.
    T_W_A_init = compute_T_W_A_from_markers(r, cal.marker_specs)
    if T_W_A_init is None:
        print("[cmd_arm] markers not visible — check streamer and Motive.")
        sys.exit(1)
    print(f"[cmd_arm] arm base (world): t=[{T_W_A_init[1][0]:+.3f}, "
          f"{T_W_A_init[1][1]:+.3f}, {T_W_A_init[1][2]:+.3f}]")

    # Build initial target — read current world pose for any coord not specified.
    if args.x is None or args.y is None or args.z is None:
        T_A_E = _read_T_A_E(r, keys)
        if T_A_E is None:
            print(f"[cmd_arm] cannot read EE pose — is OpenSai cartesian_controller "
                  f"running on '{args.robot_name}'?")
            sys.exit(1)
        T_W_E = se3_compose(T_W_A_init, T_A_E)
        T_W_P_current = se3_compose(T_W_E, cal.T_E_P)
        t_current = T_W_P_current[1].copy()
        t_init = np.array([
            args.x if args.x is not None else t_current[0],
            args.y if args.y is not None else t_current[1],
            args.z if args.z is not None else t_current[2],
        ])
        # Use the current paddle orientation so the inverse-compose in the first
        # loop tick yields exactly the current EE position — no startup lurch.
        # (world_racket_to_arm_ee derives EE position from both sweet-spot
        # position and orientation; a mismatched orientation shifts the EE goal
        # by up to 2×|t_EP| = 0.70 m even when the position is correct.)
        initial_T_W_P: SE3 = (T_W_P_current[0], t_init)
    else:
        t_init = np.array([args.x, args.y, args.z])
        face_normal = np.array([args.face_normal_x, args.face_normal_y,
                                args.face_normal_z])
        initial_T_W_P = (_R_from_face_normal(face_normal), t_init)

    print(f"[cmd_arm] initial target: world t=[{t_init[0]:+.3f}, "
          f"{t_init[1]:+.3f}, {t_init[2]:+.3f}]")
    print(f"[cmd_arm] filter α={args.marker_filter_alpha:.2f}, "
          f"rate={args.rate_hz:.0f}Hz")
    print(f"[cmd_arm] workspace: r≤{args.reach_m:.2f}m, "
          f"z∈[{args.z_min_m:+.2f},{args.z_max_m:+.2f}]m (arm frame)")
    print()
    print("Commands (all metres, world frame):")
    print("  x y z              — absolute position, face toward +X")
    print("  x y z nx ny nz     — absolute position, explicit face normal")
    print("  r dx dy dz         — relative nudge from current goal")
    print("  r dx dy dz nx ny nz — relative nudge + new face normal")
    print("  q / quit           — exit (last goal stays in Redis)")

    try:
        run(
            r=r,
            keys=keys,
            cal=cal,
            initial_T_W_P=initial_T_W_P,
            rate_hz=args.rate_hz,
            reach_m=args.reach_m,
            z_min=args.z_min_m,
            z_max=args.z_max_m,
            filter_alpha=args.marker_filter_alpha,
            goal_deadband_m=args.goal_deadband_mm / 1000.0,
        )
    except KeyboardInterrupt:
        print(f"\n[cmd_arm] interrupted — last goal stays in Redis.")


if __name__ == "__main__":
    main()
