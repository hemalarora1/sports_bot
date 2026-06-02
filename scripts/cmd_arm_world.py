#!/usr/bin/env python3
"""Command the Franka arm to world-frame paddle sweet-spot positions.

Runs a continuous tracking loop (cart-motion compensation via OptiTrack markers)
and lets you type new target positions at any time. The arm moves to the new
world-frame target and holds it, compensating for cart drift.

Prompt usage
------------
  x y z              move sweet spot to (x,y,z) world frame; PRESERVE current orientation
  x y z nx ny nz     same, with explicit face normal (nx,ny,nz); roll chosen by world-up
  r dx dy dz         relative nudge from current goal; PRESERVE current orientation
  r dx dy dz nx ny nz  relative nudge + explicit face normal
  q / quit           stop

Orientation note
----------------
The script starts with the arm's **current** paddle orientation from Redis.
Position-only commands (3 numbers) always preserve the current orientation —
no orientation snap occurs. Only 6-number commands change the face normal, and
roll is then filled by the world-up heuristic (keep paddle roughly upright).

All units in metres, world frame:
  +X  toward opponent
  +Y  left
  +Z  up (origin at court floor)

Examples
--------
  # Interactive (default — starts at current arm world pose):
  python sports_bot/scripts/cmd_arm_world.py

  # Start at a specific XY position, read Z and orientation from current arm pose:
  python sports_bot/scripts/cmd_arm_world.py --x 0.5 --y 0.0

  # At the prompt:
  0.5 0.0 0.9          # go to world (0.5, 0, 0.9), keep orientation
  r 0.1 0 0            # nudge +10 cm in X, keep orientation
  r 0 0 -0.05          # nudge -5 cm in Z, keep orientation
  0.5 0.0 0.9 0 1 0    # move + change face normal to world +Y

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
# Orientation jump threshold: Frobenius |R_raw - R_smooth|.
# Formula: 2*sqrt(1 - cos(θ)) where θ is the rotation angle between them.
#   θ = 25° → 0.44   (legitimate fast cart rotation, allow)
#   θ = 45° → 0.77   (physically impossible in one 20ms tick, discard)
# A Motive marker-ID swap flips the y-hat vector ~180° → |diff|_F ≈ 2.83.
# We use 0.6 (~20° equivalent lag tolerance) as the threshold.
_MARKER_MAX_ORI_JUMP = 0.6


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


def _parse_target(
    line: str,
) -> Optional[tuple[np.ndarray, Optional[np.ndarray], bool]]:
    """Parse a position command.

    Returns ``(t_W_P, face_normal_or_None, is_relative)``.

    ``face_normal`` is ``None`` when the user gave only 3 numbers — the caller
    should preserve the current paddle orientation rather than snapping to a
    synthesised one.  A 6-number command provides an explicit face normal.

    Formats accepted:
      x y z              absolute position, preserve current orientation
      x y z nx ny nz     absolute position, explicit face normal
      r x y z            relative delta, preserve current orientation
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
            return t, None, relative       # face_normal=None → preserve orientation
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
    lock_orientation: bool = False,
    max_goal_vel_ms: float = 0.3,
) -> None:
    cmd_q: queue.Queue[str] = queue.Queue()
    t_stdin = threading.Thread(target=_stdin_reader, args=(cmd_q,), daemon=True)
    t_stdin.start()

    T_W_P_goal = initial_T_W_P
    R_g, t_g = T_W_P_goal
    print(f"\n[cmd_arm] holding sweet spot at world t="
          f"[{t_g[0]:+.3f}, {t_g[1]:+.3f}, {t_g[2]:+.3f}]")

    # Orientation lock: when set, this arm-frame rotation is written every tick
    # instead of the R_A_E re-derived from T_W_A.  This eliminates the ~0.2°
    # per-command orientation twitch caused by OptiTrack marker angular noise.
    # Position tracking (world→arm via T_W_A) continues normally when locked.
    # When the cart rotates significantly, unlock to let world-frame orientation
    # compensation kick back in.
    locked_R_A_E: Optional[np.ndarray] = None  # None = unlocked

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

                # Orientation lock/unlock controls.
                if cmd.lower() == "lock":
                    if locked_R_A_E is not None:
                        print("[cmd_arm] orientation already locked.")
                    else:
                        print("[cmd_arm] orientation will be locked on the "
                              "next marker frame.")
                        lock_orientation = True
                    _print_prompt()
                    continue
                if cmd.lower() == "unlock":
                    locked_R_A_E = None
                    lock_orientation = False
                    last_sent_t = None  # force rewrite with live R_A_E
                    print("[cmd_arm] orientation unlocked — re-deriving from "
                          "T_W_A each tick.")
                    _print_prompt()
                    continue
                if cmd.lower() == "status":
                    ori_state = ("LOCKED" if locked_R_A_E is not None
                                 else "live (unlocked)")
                    _, t_g = T_W_P_goal
                    print(f"[cmd_arm] target_W=[{t_g[0]:+.3f},{t_g[1]:+.3f},"
                          f"{t_g[2]:+.3f}]  orientation={ori_state}")
                    _print_prompt()
                    continue

                parsed = _parse_target(cmd)
                if parsed is None:
                    print(f"[cmd_arm] unrecognised: '{cmd}'\n"
                          f"          position : x y z  |  r dx dy dz\n"
                          f"          with normal: x y z nx ny nz  |  "
                          f"r dx dy dz nx ny nz\n"
                          f"          controls  : lock | unlock | status | q")
                    _print_prompt()
                else:
                    t_new, n_new, is_relative = parsed
                    if is_relative:
                        t_new = T_W_P_goal[1] + t_new
                    if n_new is None:
                        # Position-only command: preserve current orientation.
                        # This is the common case — do NOT snap to a synthesised
                        # face normal (that was the bug causing arm lurches).
                        T_W_P_goal = (T_W_P_goal[0], t_new)
                        ori_tag = (f"ori={'LOCKED' if locked_R_A_E is not None else 'preserved'}")
                    else:
                        # Explicit face normal given: synthesise orientation.
                        # Roll is chosen by the world-up heuristic.  Also clear
                        # any orientation lock so the new face normal takes effect.
                        T_W_P_goal = (_R_from_face_normal(n_new), t_new)
                        locked_R_A_E = None
                        lock_orientation = False
                        ori_tag = (f"face_normal=[{n_new[0]:+.2f},"
                                   f"{n_new[1]:+.2f},{n_new[2]:+.2f}]"
                                   f" (roll from world-up heuristic; lock cleared)")
                    last_sent_t = None  # force immediate write to new target
                    tag = "relative→" if is_relative else ""
                    print(f"[cmd_arm] new target ({tag}world t="
                          f"[{t_new[0]:+.3f}, {t_new[1]:+.3f}, {t_new[2]:+.3f}])  "
                          f"{ori_tag}")
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
            if not dropout:
                pos_jump = float(np.linalg.norm(t_raw - t_smooth))
                # Orientation jump: Frobenius |R_raw - R_smooth|.
                # Catches Motive marker-ID swaps, which barely move the midpoint
                # (t_W_A) but flip the y-hat vector ~180°, making R_W_A garbage.
                # The position-only check above misses this entirely.
                ori_jump = float(np.linalg.norm(R_raw - R_smooth, 'fro'))
                bad_frame = (pos_jump > _MARKER_MAX_JUMP_M
                             or ori_jump > _MARKER_MAX_ORI_JUMP)
            else:
                bad_frame = False
            if bad_frame:
                if t_loop - last_warn > 1.0:
                    last_warn = t_loop
                    print(f"[cmd_arm] T_W_A bad frame: "
                          f"pos_jump={pos_jump*100:.1f}cm  "
                          f"ori_jump={ori_jump:.2f} (>{_MARKER_MAX_ORI_JUMP:.2f}) "
                          f"— discarding, arm holds last goal.")
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
                    # Re-orthogonalise R_smooth: linear EMA of rotation matrices
                    # accumulates numerical drift (det drifts from 1, axes lose
                    # orthogonality).  SVD projects back to the nearest SO(3)
                    # element every tick — cheap at 3×3.
                    U, _, Vt = np.linalg.svd(R_smooth)
                    if np.linalg.det(U @ Vt) < 0:
                        U[:, -1] *= -1   # ensure proper rotation, not reflection
                    R_smooth = U @ Vt
                last_marker_t = t_loop
                T_W_A = (R_smooth, t_smooth)

                R_A_E, t_A_E_raw = world_racket_to_arm_ee(
                    T_W_P_goal, T_W_A, cal.T_E_P)

                # Capture orientation lock on first valid frame after `lock` cmd.
                if lock_orientation and locked_R_A_E is None:
                    locked_R_A_E = R_A_E.copy()
                    print(f"[cmd_arm] orientation LOCKED (arm-frame R_A_E "
                          f"captured). Type 'unlock' to release.")
                    _print_prompt()

                # Use locked orientation when set; otherwise live R_A_E.
                R_to_write = (locked_R_A_E if locked_R_A_E is not None
                              else R_A_E)

                t_A_E_floored, floor_adj = enforce_paddle_floor(
                    t_A_E_raw, R_to_write, T_W_A)
                t_A_E_clipped, was_clipped = clip_to_arm_workspace(
                    t_A_E_floored, r_max=reach_m, z_min=z_min, z_max=z_max)
                was_clipped = was_clipped or floor_adj

                # --- Goal velocity clamping -----------------------------------
                # Cap how fast the arm-frame position goal can change per tick.
                # Without this, a large/fast cart motion (real or marker glitch)
                # instantly commands a large arm jump → violent motion.
                # We ramp toward the desired goal at max_goal_vel_ms m/s.
                vel_clamped = False
                if last_sent_t is not None and max_goal_vel_ms > 0:
                    max_step = max_goal_vel_ms / rate_hz
                    delta = t_A_E_clipped - last_sent_t
                    dist = float(np.linalg.norm(delta))
                    if dist > max_step:
                        t_A_E_clipped = last_sent_t + delta * (max_step / dist)
                        vel_clamped = True

                moved = (last_sent_t is None or
                         float(np.linalg.norm(t_A_E_clipped - last_sent_t))
                         > goal_deadband_m)
                if moved:
                    _write_arm_goal(r, keys, (R_to_write, t_A_E_clipped))
                    last_sent_t = t_A_E_clipped.copy()

                if t_loop - last_print >= print_interval_s:
                    last_print = t_loop
                    R_g, t_g = T_W_P_goal
                    tag = " CLIPPED" if was_clipped else ""
                    tag += " VEL-CLAMPED" if vel_clamped else ""
                    lock_tag = " [ORI LOCKED]" if locked_R_A_E is not None else ""
                    print(f"[cmd_arm] cart=[{t_smooth[0]:+.3f},{t_smooth[1]:+.3f},"
                          f"{t_smooth[2]:+.3f}]  "
                          f"target_W=[{t_g[0]:+.3f},{t_g[1]:+.3f},{t_g[2]:+.3f}]  "
                          f"goal_A=[{t_A_E_clipped[0]:+.3f},"
                          f"{t_A_E_clipped[1]:+.3f},"
                          f"{t_A_E_clipped[2]:+.3f}]{tag}{lock_tag}")

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
    p.add_argument("--lock-orientation", action="store_true",
                   help="Lock arm-frame orientation immediately at startup. "
                        "Eliminates the ~0.2° per-command twitch from OptiTrack "
                        "marker angular noise. Position tracking in world frame "
                        "continues normally. Use 'unlock' at the prompt to "
                        "re-enable live orientation. Equivalent to typing 'lock' "
                        "as the first command.")
    p.add_argument("--max-goal-vel-ms", type=float, default=0.3,
                   help="Maximum rate at which the arm-frame position goal can "
                        "change (m/s). Prevents violent arm motion when the cart "
                        "is moved quickly or a marker glitch causes a large "
                        "T_W_A jump. The goal ramps toward the desired value at "
                        "this speed. Set 0 to disable (unsafe with fast cart "
                        "motion). Default 0.3 m/s → 6 mm/tick at 50 Hz.")
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

    # Build initial target — always read current EE pose to get the current
    # paddle orientation.  This prevents a startup lurch: world_racket_to_arm_ee
    # derives the EE position from both sweet-spot position AND orientation, so a
    # mismatched orientation shifts the EE goal by up to 2×|t_EP| ≈ 0.70 m even
    # when the position is exactly right.
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
    # Always use the live paddle orientation — preserves arm posture.
    initial_T_W_P: SE3 = (T_W_P_current[0], t_init)

    print(f"[cmd_arm] initial target: world t=[{t_init[0]:+.3f}, "
          f"{t_init[1]:+.3f}, {t_init[2]:+.3f}]")
    print(f"[cmd_arm] filter α={args.marker_filter_alpha:.2f}, "
          f"rate={args.rate_hz:.0f}Hz")
    print(f"[cmd_arm] workspace: r≤{args.reach_m:.2f}m, "
          f"z∈[{args.z_min_m:+.2f},{args.z_max_m:+.2f}]m (arm frame)")
    print()
    print("Commands (all metres, world frame):")
    print("  x y z              — absolute position, preserve orientation")
    print("  x y z nx ny nz     — absolute position + explicit face normal")
    print("                       (roll filled by world-up heuristic)")
    print("  r dx dy dz         — relative nudge, preserve orientation")
    print("  r dx dy dz nx ny nz — relative nudge + new face normal")
    print("  lock               — lock arm-frame orientation (stops marker twitch)")
    print("  unlock             — re-enable live orientation from T_W_A")
    print("  status             — print current target and lock state")
    print("  q / quit           — exit (last goal stays in Redis)")
    if args.lock_orientation:
        print("\n[cmd_arm] --lock-orientation: will lock on first marker frame.")
    print(f"[cmd_arm] goal velocity cap: "
          f"{args.max_goal_vel_ms:.2f} m/s "
          f"({'disabled' if args.max_goal_vel_ms <= 0 else f'{args.max_goal_vel_ms/args.rate_hz*1000:.1f} mm/tick'})")

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
            lock_orientation=args.lock_orientation,
            max_goal_vel_ms=args.max_goal_vel_ms,
        )
    except KeyboardInterrupt:
        print(f"\n[cmd_arm] interrupted — last goal stays in Redis.")


if __name__ == "__main__":
    main()
