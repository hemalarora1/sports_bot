#!/usr/bin/env python3
"""Drive the Franka arm to hold a fixed world-frame racket pose, compensating
for cart motion via the live marker-based arm calibration.

This is the foundational capability `base_intercept.py` will use to track
ball intercepts: given a world-frame target pose for the racket sweet-spot,
compute the corresponding EE goal in the Franka arm base frame each tick
and write it to the OpenSai cartesian controller. As the cart moves and
T_W_A changes, the EE goal in the arm frame updates so the racket stays at
the *same world point*.

Pipeline (per tick)
-------------------
    OptiTrack markers (cart) ──▶ compute_T_W_A_from_markers ──▶ T_W_A(t)
                                                                 │
                              T_E_P (from arm calibration)       │
                                          │                      │
                                          ▼                      ▼
        T_A_E_desired  =  T_W_A(t)⁻¹  ⊕  T_W_P_goal  ⊕  T_E_P⁻¹
                                          │
                                          ▼
              opensai::controllers::FrankaRobot::cartesian_controller::
                  cartesian_task::goal_{position, orientation, linear_velocity}

Strategy
--------
- Build the target world racket pose T_W_P_goal once (from --target-* flags,
  or from the arm's current world pose with --from-current).
- Loop at --rate-hz: re-derive T_W_A from live markers, re-solve the EE goal
  in the arm frame, push to OpenSai. Goal in WORLD stays fixed; goal in ARM
  drifts as the cart moves.
- On marker dropout (one or both spheres occluded), hold the last written
  goal silently — OpenSai keeps tracking it.

What this script does NOT do
----------------------------
- No swing motion / velocity goal: just a static pose hold.
- No collision / reachability check beyond a coarse first-step warning if
  the initial EE goal is far from the arm's current pose. OpenSai's
  cartesian task clips internally; if the goal is unreachable the arm will
  reach as close as it can.
- No controller switch: assumes the cartesian_controller is already the
  active controller (use test_flick.py / test_swing.py if you need to flip).

Prereqs (in order)
------------------
1. redis-server
2. OptiTrack streamer publishing the cart's two arm-base markers
   (sai2::optitrack::marker_pos::<model_id>::<marker_id>) — labeled markers
   must be enabled in Motive.
3. OpenSai cartesian_controller running on the Franka.
4. sports_bot/optitrack/arm_marker_calibration.json present (sanity-check it
   first with `python sports_bot/scripts/verify_arm_calibration.py`).
5. THIS script.

Usage (from OpenSai root)
-------------------------
    conda activate opensai

    # Dry-run preview: what would we send each tick, no Redis writes.
    python sports_bot/scripts/test_arm_world_track.py \
        --target-x 0.6 --target-y 0.0 --target-z 0.9 --dry-run

    # Hold the racket at the arm's current world pose. Push the cart by hand
    # and watch the arm move to keep the racket where it was.
    python sports_bot/scripts/test_arm_world_track.py --from-current

    # Hold at an absolute world target, face-normal pointing toward the
    # opponent (world +X).
    python sports_bot/scripts/test_arm_world_track.py \
        --target-x 0.6 --target-y 0.0 --target-z 0.9 \
        --face-normal-x 1 --face-normal-y 0 --face-normal-z 0
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
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
    axis_angle_from_R,
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

def _R_world_racket_from_face_normal(
    face_normal: np.ndarray,
    world_up: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Build a 3x3 R_W_P whose columns are [face_right, face_up, face_normal].
    Mirrors sports_bot/state_machine/swing_planner.py:_rotation_from_normal —
    the same racket-frame convention (+Z_P = face normal) used by the FSM."""
    n = face_normal / max(1e-9, float(np.linalg.norm(face_normal)))
    if world_up is None:
        world_up = np.array([0.0, 0.0, 1.0])
    up_proj = world_up - float(np.dot(world_up, n)) * n
    if float(np.linalg.norm(up_proj)) < 1e-6:
        # Degenerate (face_normal parallel to world_up) — pick world +X as ref.
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


def _write_arm_goal(
    r: redis.Redis,
    keys: OpenSaiCartesianKeys,
    T_A_E: SE3,
) -> None:
    R, t = T_A_E
    r.set(keys.goal_position, json.dumps(t.tolist()))
    r.set(keys.goal_orientation,
          json.dumps([[float(v) for v in row] for row in R]))
    # Always publish a zero velocity goal so we don't inherit a stale impact
    # velocity from a previous test_swing run.
    r.set(keys.goal_linear_velocity, json.dumps([0.0, 0.0, 0.0]))


def _format_se3(label: str, T: SE3) -> str:
    R, t = T
    omega = axis_angle_from_R(R)
    angle_deg = math.degrees(float(np.linalg.norm(omega)))
    return (f"  {label}: "
            f"t=[{t[0]:+.3f}, {t[1]:+.3f}, {t[2]:+.3f}] m, "
            f"|ω|={angle_deg:+.1f}°")


# ---------- Main loop ---------------------------------------------------------

def run(
    r: redis.Redis,
    keys: OpenSaiCartesianKeys,
    cal,
    T_W_P_goal: SE3,
    rate_hz: float,
    print_interval_s: float,
    dry_run: bool,
    reach_m: float,
    z_min: float,
    z_max: float,
    filter_alpha: float,
    goal_deadband_m: float,
) -> None:
    dt = 1.0 / rate_hz
    last_print = -math.inf
    last_warn = -math.inf
    last_clip_warn = -math.inf

    print(f"[arm_track] T_W_P_goal (held fixed in WORLD frame):")
    R_goal, t_goal = T_W_P_goal
    print(f"            t = [{t_goal[0]:+.4f}, {t_goal[1]:+.4f}, "
          f"{t_goal[2]:+.4f}] m")
    print(f"            R_W_P columns = "
          f"face_right={R_goal[:,0].round(3).tolist()}  "
          f"face_up={R_goal[:,1].round(3).tolist()}  "
          f"face_normal={R_goal[:,2].round(3).tolist()}")
    print(f"[arm_track] workspace clip: r ≤ {reach_m:.2f} m, "
          f"z ∈ [{z_min:+.2f}, {z_max:+.2f}] m (arm base frame)")
    print(f"[arm_track] marker filter alpha={filter_alpha:.2f} "
          f"(τ≈{1.0/(filter_alpha*rate_hz+1e-9):.0f}ms), "
          f"goal deadband={goal_deadband_m*1000:.1f}mm")
    print(f"[arm_track] loop rate {rate_hz:.0f} Hz, "
          f"writing {keys.goal_position}{' (DRY RUN)' if dry_run else ''}")
    print(f"[arm_track] Ctrl-C to stop. Last goal stays in Redis on exit.\n")

    # EMA state for T_W_A — seeded on first valid read.
    t_smooth: Optional[np.ndarray] = None
    R_smooth: Optional[np.ndarray] = None
    last_sent_t: Optional[np.ndarray] = None  # last arm-frame position written
    last_marker_t: float = -math.inf

    while True:
        t_loop = time.perf_counter()

        T_W_A_raw = compute_T_W_A_from_markers(r, cal.marker_specs)
        if T_W_A_raw is None:
            if t_loop - last_warn > 1.0:
                last_warn = t_loop
                absent = (t_loop - last_marker_t
                          if last_marker_t > -math.inf else 0.0)
                print(f"[arm_track] markers missing ({absent:.0f}s) — "
                      f"holding last goal (no Redis write this tick).")
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
                    print(f"[arm_track] T_W_A jumped {delta*100:.1f} cm in one "
                          f"tick — discarding bad frame, EMA resets on next "
                          f"valid read.")
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
                    t_A_E_floored, r_max=reach_m, z_min=z_min, z_max=z_max,
                )
                was_clipped = was_clipped or floor_adj

                moved = (last_sent_t is None or
                         float(np.linalg.norm(t_A_E_clipped - last_sent_t))
                         > goal_deadband_m)
                if moved:
                    if not dry_run:
                        _write_arm_goal(r, keys, (R_A_E, t_A_E_clipped))
                    last_sent_t = t_A_E_clipped.copy()

                if was_clipped and t_loop - last_clip_warn > 1.0:
                    last_clip_warn = t_loop
                    r_raw = float(np.linalg.norm(t_A_E_raw[:2]))
                    print(f"[arm_track] CLIPPED: raw arm-frame target "
                          f"[{t_A_E_raw[0]:+.3f}, {t_A_E_raw[1]:+.3f}, "
                          f"{t_A_E_raw[2]:+.3f}] (r_xy={r_raw:.3f} m) → "
                          f"[{t_A_E_clipped[0]:+.3f}, {t_A_E_clipped[1]:+.3f}, "
                          f"{t_A_E_clipped[2]:+.3f}]")

                if t_loop - last_print >= print_interval_s:
                    last_print = t_loop
                    tag = " (clipped)" if was_clipped else ""
                    print(f"[arm_track] T_W_A.t=[{t_smooth[0]:+.3f},"
                          f"{t_smooth[1]:+.3f},{t_smooth[2]:+.3f}]  "
                          f"→ goal_A.t=[{t_A_E_clipped[0]:+.3f},"
                          f"{t_A_E_clipped[1]:+.3f},"
                          f"{t_A_E_clipped[2]:+.3f}]{tag}")

        elapsed = time.perf_counter() - t_loop
        sleep = dt - elapsed
        if sleep > 0:
            time.sleep(sleep)


# ---------- Entry point -------------------------------------------------------

def _build_target_world_pose(
    args: argparse.Namespace,
    r: Optional[redis.Redis],
    keys: OpenSaiCartesianKeys,
    cal,
) -> SE3:
    """Build the world-frame racket goal T_W_P_goal from the CLI flags.

    Orientation policy
    ------------------
    When Redis is reachable (i.e. not --dry-run), we read the arm's current
    EE pose and derive the current paddle world orientation via the forward
    kinematic chain::

        T_W_P = T_W_A ⊕ T_A_E ⊕ T_E_P

    That orientation is used as R_W_P, regardless of --face-normal-*.  This
    prevents a startup lurch: world_racket_to_arm_ee derives the EE *position*
    from both the sweet-spot position AND its orientation, so a mismatched
    orientation shifts the EE goal by up to 2×|t_EP| ≈ 0.70 m.

    The --face-normal-* args are kept as a fallback for --dry-run (no Redis),
    and they are still the orientation source for the old --from-current path
    if the FK read somehow fails.
    """
    face_normal = np.array([args.face_normal_x,
                            args.face_normal_y,
                            args.face_normal_z], dtype=float)
    if float(np.linalg.norm(face_normal)) < 1e-6:
        raise SystemExit("[arm_track] --face-normal vector is zero")
    R_W_P_synthesised = _R_world_racket_from_face_normal(face_normal)

    if args.from_current:
        if r is None:
            raise SystemExit("[arm_track] --from-current cannot be combined "
                             "with --dry-run (need live Redis reads)")
        T_A_E = _read_T_A_E(r, keys)
        if T_A_E is None:
            raise SystemExit(
                f"[arm_track] cannot read EE pose from {keys.current_position} "
                f"— is OpenSai cartesian_controller running on robot "
                f"'{keys.robot_name}'?"
            )
        T_W_A = compute_T_W_A_from_markers(r, cal.marker_specs)
        if T_W_A is None:
            raise SystemExit("[arm_track] cannot compute T_W_A — markers not "
                             "visible at startup")
        # Current racket world pose = T_W_A ⊕ T_A_E ⊕ T_E_P
        T_W_E = se3_compose(T_W_A, T_A_E)
        T_W_P_current = se3_compose(T_W_E, cal.T_E_P)
        t_goal = T_W_P_current[1].copy()
        # Use the current racket orientation rather than the synthesized
        # face-normal one — preserves the arm posture.
        R_goal = T_W_P_current[0].copy()
        # Optional offset from current.
        t_goal += np.array([args.offset_x, args.offset_y, args.offset_z])
        return (R_goal, t_goal)

    # --target-* mode: position is fully specified; determine orientation.
    t_goal = np.array([args.target_x, args.target_y, args.target_z], dtype=float)

    # Prefer live orientation from Redis — avoids a startup lurch caused by
    # the world-up-heuristic roll in R_W_P_synthesised not matching the
    # physical T_E_P mount roll.
    if r is not None:
        T_W_A = compute_T_W_A_from_markers(r, cal.marker_specs)
        T_A_E = _read_T_A_E(r, keys)
        if T_W_A is not None and T_A_E is not None:
            T_W_E = se3_compose(T_W_A, T_A_E)
            T_W_P_current = se3_compose(T_W_E, cal.T_E_P)
            R_goal = T_W_P_current[0].copy()
            print("[arm_track] using current paddle orientation to avoid "
                  "startup lurch (face-normal synthesis would shift EE goal).")
            return (R_goal, t_goal)
        else:
            print("[arm_track] WARNING: cannot read current EE pose — falling "
                  "back to face-normal orientation (may cause startup motion).")

    # Fallback (--dry-run or FK read failed): synthesised from --face-normal-*.
    return (R_W_P_synthesised, t_goal)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Target spec — pick one mode.
    p.add_argument("--from-current", action="store_true",
                   help="Read the current racket world pose and hold there. "
                        "Useful for the 'push the cart by hand' test. Ignores "
                        "--target-*. Can be offset via --offset-*.")
    p.add_argument("--target-x", type=float, default=0.6,
                   help="World-X (m) of the held racket sweet-spot.")
    p.add_argument("--target-y", type=float, default=0.0,
                   help="World-Y (m) of the held racket sweet-spot.")
    p.add_argument("--target-z", type=float, default=0.9,
                   help="World-Z (m) of the held racket sweet-spot.")
    p.add_argument("--face-normal-x", type=float, default=1.0,
                   help="World-X component of the racket face normal direction.")
    p.add_argument("--face-normal-y", type=float, default=0.0,
                   help="World-Y component of the racket face normal direction.")
    p.add_argument("--face-normal-z", type=float, default=0.0,
                   help="World-Z component of the racket face normal direction.")
    p.add_argument("--offset-x", type=float, default=0.0,
                   help="(--from-current only) World-X offset from current.")
    p.add_argument("--offset-y", type=float, default=0.0,
                   help="(--from-current only) World-Y offset from current.")
    p.add_argument("--offset-z", type=float, default=0.0,
                   help="(--from-current only) World-Z offset from current.")

    # Loop / behavior.
    p.add_argument("--rate-hz", type=float, default=50.0,
                   help="Control loop rate.")
    p.add_argument("--print-interval-s", type=float, default=0.5,
                   help="Throttle for status prints (loop runs faster).")
    p.add_argument("--max-startup-jump-m", type=float, default=0.20,
                   help="Safety: refuse to start if the initial arm-frame EE "
                        "goal is more than this far from the arm's current "
                        "EE pose. Prevents accidental large jumps when the "
                        "target is set far from where the arm currently is. "
                        "Set 0 to disable the check.")
    p.add_argument("--reach-m", type=float, default=FRANKA_DEFAULT_REACH_M,
                   help="Conservative max horizontal reach (m) in the arm "
                        "base frame. EE position goals beyond this are "
                        "radially clipped to the cylinder boundary each tick.")
    p.add_argument("--z-min-m", type=float, default=FRANKA_DEFAULT_Z_MIN_M,
                   help="Min EE Z (m) in the arm base frame. Goals below "
                        "this are clamped each tick.")
    p.add_argument("--z-max-m", type=float, default=FRANKA_DEFAULT_Z_MAX_M,
                   help="Max EE Z (m) in the arm base frame. Goals above "
                        "this are clamped each tick.")

    # Config / connectivity.
    p.add_argument("--robot-name", default="FrankaRobot",
                   help="OpenSai robot name for the cartesian goal keys.")
    p.add_argument("--calibration", default=None,
                   help="Path to arm_marker_calibration.json. Default: "
                        "sports_bot/optitrack/arm_marker_calibration.json")
    p.add_argument("--goal-deadband-mm", type=float, default=3.0,
                   help="Only write a new arm goal when the arm-frame position "
                        "has moved more than this many mm from the last sent "
                        "goal. Suppresses vibration from residual marker noise. "
                        "Set 0 to disable.")
    p.add_argument("--marker-filter-alpha", type=float, default=0.15,
                   help="EMA smoothing factor for T_W_A (0<α≤1). Lower = "
                        "smoother but more lag. 1.0 = no filter. Default 0.15 "
                        "gives ~130ms time constant at 50Hz, filtering OptiTrack "
                        "marker noise while tracking cart motion.")
    p.add_argument("--redis-host", default="localhost")
    p.add_argument("--redis-port", type=int, default=6379)
    p.add_argument("--dry-run", action="store_true",
                   help="Print computed goals each tick without writing to "
                        "Redis. Skips the EE pose read so OpenSai doesn't "
                        "need to be running.")

    args = p.parse_args()

    cal_path = args.calibration or arm_marker_calibration_path()
    if not os.path.isfile(cal_path):
        print(f"[arm_track] missing arm calibration at {cal_path}.")
        print(f"            run `verify_arm_calibration.py --init` first.")
        sys.exit(1)
    cal = load_arm_marker_calibration(cal_path)
    print(f"[arm_track] loaded {cal_path}")
    print(f"            marker_specs = {cal.marker_specs}")
    a, b = cal.marker_specs
    print(f"            (Franka +Y direction = vector from marker {a} → {b})")

    keys = OpenSaiCartesianKeys(robot_name=args.robot_name)

    r: Optional[redis.Redis] = None
    if not args.dry_run or args.from_current:
        r = redis.Redis(host=args.redis_host, port=args.redis_port,
                        decode_responses=True)
        try:
            r.ping()
        except redis.exceptions.ConnectionError as e:
            print(f"[arm_track] cannot reach Redis "
                  f"at {args.redis_host}:{args.redis_port}: {e}")
            sys.exit(1)

    # Pre-flight visibility check.
    if r is not None:
        T_W_A_initial = compute_T_W_A_from_markers(r, cal.marker_specs)
        if T_W_A_initial is None:
            print(f"[arm_track] markers not visible at startup. Check Motive + "
                  f"streamer; run list_markers.py to confirm IDs "
                  f"{cal.marker_specs} are streaming.")
            sys.exit(1)
        print(_format_se3("T_W_A startup", T_W_A_initial))

    T_W_P_goal = _build_target_world_pose(args, r, keys, cal)
    print(_format_se3("T_W_P_goal (held fixed)", T_W_P_goal))

    # Safety: warn / abort on a big startup jump (using the CLIPPED first
    # goal — clipping reflects what we'd actually send).
    if r is not None and not args.dry_run and args.max_startup_jump_m > 0:
        T_A_E_now = _read_T_A_E(r, keys)
        if T_A_E_now is None:
            print(f"[arm_track] cannot read current EE pose from "
                  f"{keys.current_position} — is OpenSai cartesian_controller "
                  f"running on robot '{args.robot_name}'?")
            sys.exit(1)
        # Re-read T_W_A to be sure (it's basically free).
        T_W_A = compute_T_W_A_from_markers(r, cal.marker_specs)
        R_A_E, t_A_E_raw = world_racket_to_arm_ee(
            T_W_P_goal, T_W_A, cal.T_E_P)
        t_A_E_clipped, was_clipped = clip_to_arm_workspace(
            t_A_E_raw,
            r_max=args.reach_m, z_min=args.z_min_m, z_max=args.z_max_m,
        )
        jump = float(np.linalg.norm(t_A_E_clipped - T_A_E_now[1]))
        print(_format_se3("T_A_E current", T_A_E_now))
        print(_format_se3("T_A_E first goal (raw)", (R_A_E, t_A_E_raw)))
        if was_clipped:
            print(_format_se3("T_A_E first goal (clipped)",
                              (R_A_E, t_A_E_clipped)))
        print(f"[arm_track] startup jump (arm-frame, clipped) = "
              f"{jump*1000:.1f} mm  "
              f"(limit {args.max_startup_jump_m*1000:.0f} mm)")
        if jump > args.max_startup_jump_m:
            print(f"[arm_track] ABORT: first goal is {jump:.3f} m from the "
                  f"current EE pose. Move the arm closer first, or pass "
                  f"--max-startup-jump-m {jump + 0.05:.2f} to override.")
            sys.exit(1)

    try:
        run(
            r=r,
            keys=keys,
            cal=cal,
            T_W_P_goal=T_W_P_goal,
            rate_hz=args.rate_hz,
            print_interval_s=args.print_interval_s,
            dry_run=args.dry_run,
            reach_m=args.reach_m,
            z_min=args.z_min_m,
            z_max=args.z_max_m,
            filter_alpha=args.marker_filter_alpha,
            goal_deadband_m=args.goal_deadband_mm / 1000.0,
        )
    except KeyboardInterrupt:
        print(f"\n[arm_track] stopping — leaving last goal in Redis "
              f"({keys.goal_position}). Re-run or write a new goal to "
              f"change it.")


if __name__ == "__main__":
    main()
