#!/usr/bin/env python3
"""Base-only ball-intercept loop. No arm, no full FSM.

Pipeline
--------
    OptiTrack ball  ──▶  BallTracker.predict_intercept(x = strike_plane_x)
                                       │
                                       ▼
                         clamp y_pred to [base_y_min, base_y_max]
                                       │
                                       ▼
            sports_bot::cmd::base::goal_pose  =  [ready_x, y_pred, ready_yaw]
                                       │
                                       ▼
                              base_bridge.py  ──▶  hb1::desired_pose

Strategy
--------
- Sits at ``ready_base_pose`` until a ball is detected.
- When the tracker returns an incoming intercept at the strike plane, the cart
  is commanded to align its centerline (only Y changes; X and yaw stay at
  ready). The arm is untouched.
- When no usable prediction has arrived for ``hold_after_lost_s`` seconds, the
  cart returns to the ready pose.

What this script DOES NOT do (on purpose)
-----------------------------------------
- No commit-time / swing trigger — the cart just continuously tracks the
  predicted Y.
- No arm control. Whatever has the arm task active keeps it; this script
  doesn't write to any racket / cartesian / joint keys.
- No feasibility check on whether the base can actually arrive on time. It
  prints predicted-arrival vs. predicted-impact for inspection but commands
  the goal regardless. Tune ``--min-lookahead`` / ``--max-lookahead`` to gate.

Prereqs (in order)
------------------
1. redis-server.
2. OptiTrack streamer publishing ball + cart rigid bodies.
3. TidyBot redis_driver.py.
4. sports_bot/base_bridge.py --robot-rigid-body-id <CART_ID>.
5. THIS script.

Usage
-----
    conda activate opensai
    # measure your workspace first (see --help) and pass real limits
    python sports_bot/scripts/base_intercept.py \
        --ball-rigid-body-id 8 \
        --strike-plane-x 0.60 \
        --ready-x 0.0 --ready-y 0.0 --ready-yaw-deg 0 \
        --base-y-min -1.0 --base-y-max 1.0
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Optional, Tuple

import numpy as np
import redis

_THIS_FILE = os.path.abspath(__file__)
_OPENSAI_DIR = os.path.dirname(os.path.dirname(os.path.dirname(_THIS_FILE)))
if _OPENSAI_DIR not in sys.path:
    sys.path.insert(0, _OPENSAI_DIR)

from sports_bot.state_machine.ball_tracker import BallTracker  # noqa: E402
from sports_bot.state_machine.config import BallTrackerConfig  # noqa: E402
from sports_bot.state_machine.redis_keys import (                # noqa: E402
    OpenSaiCartesianKeys,
    RedisKeys,
)
from sports_bot.utils.frames import (                            # noqa: E402
    FRANKA_DEFAULT_REACH_M,
    FRANKA_DEFAULT_Z_MAX_M,
    FRANKA_DEFAULT_Z_MIN_M,
    SE3,
    arm_marker_calibration_path,
    clip_to_arm_workspace,
    compute_T_W_A_from_markers,
    load_arm_marker_calibration,
    load_robot_marker_calibration,
    quat_to_R,
    robot_marker_calibration_path,
    se2_compose,
    world_racket_to_arm_ee,
)


FSM_BASE_GOAL = "sports_bot::cmd::base::goal_pose"

OPTI_POS_PREFIX = "sai2::optitrack::rigid_body_pos::"
OPTI_ORI_PREFIX = "sai2::optitrack::rigid_body_ori::"


def _clamp(v: float, lo: float, hi: float) -> Tuple[float, bool]:
    if v < lo:
        return lo, True
    if v > hi:
        return hi, True
    return v, False


def _write_goal(r: redis.Redis, pose: Tuple[float, float, float]) -> None:
    r.set(FSM_BASE_GOAL, json.dumps([float(pose[0]), float(pose[1]), float(pose[2])]))


def _read_T_W_C(
    r: redis.Redis, cart_rb_id: int, T_B_C
) -> Optional[Tuple[float, float, float]]:
    """Live world-frame pose of the cart's odometry control point C:
    T_W_C = T_W_B ⊕ T_B_C. Used for the world-frame tracking-error
    diagnostic — same composition `send_base_goal.py` / base_bridge sanity
    line use."""
    pos_raw = r.get(OPTI_POS_PREFIX + str(cart_rb_id))
    ori_raw = r.get(OPTI_ORI_PREFIX + str(cart_rb_id))
    if pos_raw is None or ori_raw is None:
        return None
    try:
        pos = json.loads(pos_raw)
        ori = json.loads(ori_raw)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    if len(pos) != 3 or len(ori) != 4:
        return None
    R = quat_to_R(float(ori[0]), float(ori[1]), float(ori[2]), float(ori[3]))
    # Cart yaw from rotation matrix (same projection yaw_from_quat uses).
    yaw_W_B = math.atan2(R[1, 0], R[0, 0])
    T_W_B: Tuple[float, float, float] = (float(pos[0]), float(pos[1]), yaw_W_B)
    return se2_compose(T_W_B, T_B_C)


def _R_world_racket_from_face_normal(face_normal: np.ndarray) -> np.ndarray:
    """Build a 3x3 R_W_P with +Z_P = face_normal, +Y_P as close to world +Z
    as possible. Same convention as SwingPlanner / test_arm_world_track."""
    n = face_normal / max(1e-9, float(np.linalg.norm(face_normal)))
    world_up = np.array([0.0, 0.0, 1.0])
    up_proj = world_up - float(np.dot(world_up, n)) * n
    if float(np.linalg.norm(up_proj)) < 1e-6:
        ref = np.array([1.0, 0.0, 0.0])
        up_proj = ref - float(np.dot(ref, n)) * n
    face_up = up_proj / float(np.linalg.norm(up_proj))
    face_right = np.cross(face_up, n)
    return np.column_stack([face_right, face_up, n])


def _write_arm_goal(
    r: redis.Redis,
    keys: OpenSaiCartesianKeys,
    T_A_E: SE3,
) -> None:
    R, t = T_A_E
    r.set(keys.goal_position, json.dumps(t.tolist()))
    r.set(keys.goal_orientation,
          json.dumps([[float(v) for v in row] for row in R]))
    # Zero velocity goal — pose hold, no stale impact velocity carried over.
    r.set(keys.goal_linear_velocity, json.dumps([0.0, 0.0, 0.0]))


def run(
    r: redis.Redis,
    tracker: BallTracker,
    ready_pose: Tuple[float, float, float],
    strike_plane_x: float,
    base_y_min: float,
    base_y_max: float,
    rate_hz: float,
    hold_after_lost_s: float,
    print_interval_s: float,
    # Diagnostic: world-frame tracking error (cart C vs commanded goal_W).
    cart_rb_id: Optional[int] = None,
    T_B_C: Optional[Tuple[float, float, float]] = None,
    # Arm tracking (optional): hold the racket at the predicted intercept
    # point as the cart chases, clipped to the arm workspace.
    arm_cal=None,
    arm_keys: Optional[OpenSaiCartesianKeys] = None,
    arm_face_normal: Optional[np.ndarray] = None,
    arm_reach_m: float = FRANKA_DEFAULT_REACH_M,
    arm_z_min: float = FRANKA_DEFAULT_Z_MIN_M,
    arm_z_max: float = FRANKA_DEFAULT_Z_MAX_M,
) -> None:
    dt = 1.0 / rate_hz
    last_print = 0.0
    last_good_t = -math.inf
    last_goal: Optional[Tuple[float, float, float]] = None
    at_ready = False
    last_arm_warn = -math.inf
    arm_track_on = (arm_cal is not None and arm_keys is not None
                    and arm_face_normal is not None)

    # Park at ready immediately so the cart settles while we wait for the ball.
    _write_goal(r, ready_pose)
    last_goal = ready_pose
    at_ready = True
    print(f"[base_intercept] ready_pose = ({ready_pose[0]:+.3f}, "
          f"{ready_pose[1]:+.3f}, {math.degrees(ready_pose[2]):+.1f}°)")
    print(f"[base_intercept] strike_plane_x = {strike_plane_x:+.3f} m,  "
          f"y limits = [{base_y_min:+.3f}, {base_y_max:+.3f}]")
    if cart_rb_id is not None and T_B_C is not None:
        print(f"[base_intercept] tracking-error diagnostic: comparing "
              f"OT T_W_C against last commanded goal_W")
    if arm_track_on:
        print(f"[base_intercept] arm tracking ON: holding racket sweet-spot at "
              f"intercept, face_normal={arm_face_normal.round(3).tolist()}, "
              f"workspace r ≤ {arm_reach_m:.2f} m, "
              f"z ∈ [{arm_z_min:+.2f}, {arm_z_max:+.2f}] m")
    print(f"[base_intercept] writing {FSM_BASE_GOAL} at {rate_hz:.0f} Hz "
          f"(Ctrl-C to stop)\n")

    R_W_P_const = (_R_world_racket_from_face_normal(arm_face_normal)
                   if arm_track_on else None)

    while True:
        t0 = time.perf_counter()

        tracker.update()
        intercept = tracker.predict_intercept(strike_plane_x)

        if intercept is not None:
            y_raw = float(intercept.position[1])
            y_clamped, clamped = _clamp(y_raw, base_y_min, base_y_max)
            goal = (ready_pose[0], y_clamped, ready_pose[2])
            if last_goal is None or any(
                abs(goal[i] - last_goal[i]) > 1e-4 for i in range(3)
            ):
                _write_goal(r, goal)
                last_goal = goal
            at_ready = False
            last_good_t = t0

            # Arm tracking: command the racket sweet-spot to the predicted
            # intercept (not the clamped Y — the arm should reach beyond
            # the base limits if the cart hasn't caught up yet, and clipping
            # to the arm workspace handles the case where the cart is too
            # far away). Always-write so the controller stays warm.
            arm_clip_note = ""
            if arm_track_on:
                T_W_A = compute_T_W_A_from_markers(r, arm_cal.marker_specs)
                if T_W_A is None:
                    if t0 - last_arm_warn > 1.0:
                        last_arm_warn = t0
                        print(f"[base_intercept] arm markers missing — "
                              f"holding last arm goal")
                else:
                    t_W_P = np.asarray(intercept.position, dtype=float)
                    T_W_P_goal: SE3 = (R_W_P_const, t_W_P)
                    R_A_E, t_A_E_raw = world_racket_to_arm_ee(
                        T_W_P_goal, T_W_A, arm_cal.T_E_P)
                    t_A_E_clipped, was_clipped = clip_to_arm_workspace(
                        t_A_E_raw,
                        r_max=arm_reach_m, z_min=arm_z_min, z_max=arm_z_max,
                    )
                    _write_arm_goal(r, arm_keys, (R_A_E, t_A_E_clipped))
                    if was_clipped:
                        r_raw = float(np.linalg.norm(t_A_E_raw[:2]))
                        arm_clip_note = (
                            f"  [arm CLIPPED r_xy={r_raw:.2f}→{arm_reach_m:.2f}]"
                        )

            if t0 - last_print >= print_interval_s:
                last_print = t0
                tag = " (clamped)" if clamped else ""
                z = float(intercept.position[2])
                tti = float(intercept.time_to_impact)
                vy = float(intercept.velocity[1])

                # World-frame tracking error: cart C pose from OT vs last
                # commanded goal_W (only base XY — yaw is held at ready_yaw).
                track_note = ""
                if cart_rb_id is not None and T_B_C is not None:
                    T_W_C = _read_T_W_C(r, cart_rb_id, T_B_C)
                    if T_W_C is not None:
                        dx_mm = 1000.0 * (T_W_C[0] - goal[0])
                        dy_mm = 1000.0 * (T_W_C[1] - goal[1])
                        track_note = (
                            f"  track_err W=[{dx_mm:+.0f}, {dy_mm:+.0f}] mm"
                        )

                print(
                    f"[base_intercept] intercept y={y_raw:+.3f}{tag}  "
                    f"z={z:+.2f}  tti={tti:+.3f}s  v_y={vy:+.2f} m/s  "
                    f"→ goal_W=({goal[0]:+.3f}, {goal[1]:+.3f})"
                    f"{track_note}{arm_clip_note}"
                )
        else:
            # No usable prediction this tick. After we've been without one for
            # hold_after_lost_s, fall back to ready.
            since_good = t0 - last_good_t
            if since_good > hold_after_lost_s and not at_ready:
                _write_goal(r, ready_pose)
                last_goal = ready_pose
                at_ready = True
                reason = getattr(tracker, "last_reject_reason", "") or "no_data"
                print(f"[base_intercept] lost for {since_good:.2f}s "
                      f"(reason={reason}) → return to ready")
                # NOTE: arm goal is intentionally left at its last value when
                # the ball is lost — we don't have a "racket ready pose" here
                # (that's the FSM's job). Whatever last intercept arm goal
                # OpenSai is holding stays put.

        elapsed = time.perf_counter() - t0
        sleep = dt - elapsed
        if sleep > 0:
            time.sleep(sleep)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Base-only ball intercept driver (writes "
                    "sports_bot::cmd::base::goal_pose; needs base_bridge.py "
                    "running to forward to the cart).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ball-rigid-body-id", type=int, default=8,
                   help="Motive Streaming ID of the pickleball.")
    p.add_argument("--redis-host", default="localhost")
    p.add_argument("--redis-port", type=int, default=6379)

    # Workspace measurements — REPLACE WITH REAL VALUES after walking the cart.
    p.add_argument("--strike-plane-x", type=float, default=0.60,
                   help="World-X (m) at which the predicted intercept is "
                        "evaluated. Set to ~the cart's forward standoff in "
                        "your bay.")
    p.add_argument("--ready-x", type=float, default=0.0,
                   help="World-X of the ready / rest base pose.")
    p.add_argument("--ready-y", type=float, default=0.0,
                   help="World-Y of the ready / rest base pose.")
    p.add_argument("--ready-yaw-deg", type=float, default=0.0,
                   help="World yaw (deg) of the ready / rest base pose.")
    p.add_argument("--base-y-min", type=float, default=-1.0,
                   help="Min world-Y the base is allowed to reach. "
                        "MEASURE this in the actual workspace.")
    p.add_argument("--base-y-max", type=float, default=1.0,
                   help="Max world-Y the base is allowed to reach. "
                        "MEASURE this in the actual workspace.")

    # Loop / behavior knobs.
    p.add_argument("--rate-hz", type=float, default=50.0,
                   help="Control loop rate.")
    p.add_argument("--hold-after-lost-s", type=float, default=0.6,
                   help="How long to hold the last intercept goal after the "
                        "tracker stops returning usable predictions before "
                        "falling back to ready.")
    p.add_argument("--print-interval-s", type=float, default=0.1,
                   help="Throttle for intercept prints (the loop runs faster).")

    # Tracker lookahead overrides — the default config caps at 1.5 s and 0.05 s,
    # which is usually fine; expose them for tuning without editing config.py.
    p.add_argument("--min-lookahead", type=float, default=None,
                   help="Override BallTrackerConfig.min_lookahead (s).")
    p.add_argument("--max-lookahead", type=float, default=None,
                   help="Override BallTrackerConfig.max_lookahead (s).")

    # World-frame tracking diagnostic (cart C from OT vs last commanded goal_W).
    # Needs the cart rigid body ID and T_B_C — same inputs base_bridge uses.
    p.add_argument("--robot-rigid-body-id", type=int, default=11,
                   help="Cart Streaming ID — used for the world-frame "
                        "tracking-error diagnostic. Set to 0 to disable the "
                        "diagnostic.")
    p.add_argument("--robot-marker-calibration", type=str, default=None,
                   help="Path to T_B_C calibration JSON. Default: "
                        "sports_bot/optitrack/robot_marker_calibration.json")

    # Optional arm tracking — racket sweet-spot held at the predicted
    # intercept point in world frame, clipped to the Franka workspace.
    p.add_argument("--arm-track", action="store_true",
                   help="Also command the Franka arm to track the predicted "
                        "intercept point in world frame. Requires the arm "
                        "marker calibration JSON and an active OpenSai "
                        "cartesian_controller.")
    p.add_argument("--arm-calibration", type=str, default=None,
                   help="Path to arm_marker_calibration.json. Default: "
                        "sports_bot/optitrack/arm_marker_calibration.json")
    p.add_argument("--arm-robot-name", default="FrankaRobot",
                   help="OpenSai robot name for the cartesian goal keys.")
    p.add_argument("--arm-face-normal-x", type=float, default=1.0,
                   help="World-X component of the racket face normal.")
    p.add_argument("--arm-face-normal-y", type=float, default=0.0,
                   help="World-Y component of the racket face normal.")
    p.add_argument("--arm-face-normal-z", type=float, default=0.0,
                   help="World-Z component of the racket face normal.")
    p.add_argument("--arm-reach-m", type=float, default=FRANKA_DEFAULT_REACH_M,
                   help="Max horizontal arm reach (m, arm base frame) for "
                        "workspace clipping.")
    p.add_argument("--arm-z-min-m", type=float, default=FRANKA_DEFAULT_Z_MIN_M,
                   help="Min EE Z (m, arm base frame) for workspace clipping.")
    p.add_argument("--arm-z-max-m", type=float, default=FRANKA_DEFAULT_Z_MAX_M,
                   help="Max EE Z (m, arm base frame) for workspace clipping.")

    args = p.parse_args()

    if args.base_y_min >= args.base_y_max:
        p.error("--base-y-min must be < --base-y-max")
    if not (args.base_y_min <= args.ready_y <= args.base_y_max):
        print(f"[base_intercept] WARNING: ready_y={args.ready_y} is outside "
              f"[{args.base_y_min}, {args.base_y_max}].")

    r = redis.Redis(host=args.redis_host, port=args.redis_port, decode_responses=True)
    try:
        r.ping()
    except redis.exceptions.ConnectionError as e:
        print(f"[base_intercept] cannot reach Redis "
              f"at {args.redis_host}:{args.redis_port}: {e}")
        sys.exit(1)

    cfg = BallTrackerConfig()
    if args.min_lookahead is not None:
        cfg.min_lookahead = args.min_lookahead
    if args.max_lookahead is not None:
        cfg.max_lookahead = args.max_lookahead

    keys = RedisKeys(ball_source="optitrack")
    # Override the rigid-body id without recreating the BallKeys default.
    keys.ball.__dict__["optitrack_rigid_body_id"] = args.ball_rigid_body_id

    # Sanity: do we actually see ball positions?
    ball_key = keys.ball.optitrack_position
    if r.get(ball_key) is None:
        print(f"[base_intercept] WARNING: {ball_key} is empty. "
              f"Is the OptiTrack streamer running?")
    else:
        print(f"[base_intercept] reading ball from {ball_key}")

    tracker = BallTracker(r, keys, cfg)

    ready_pose = (args.ready_x, args.ready_y, math.radians(args.ready_yaw_deg))

    # World-frame tracking-error diagnostic: load T_B_C if available.
    cart_rb_id: Optional[int] = None
    T_B_C = None
    if args.robot_rigid_body_id > 0:
        tbc_path = (args.robot_marker_calibration
                    or robot_marker_calibration_path())
        if os.path.isfile(tbc_path):
            T_B_C = load_robot_marker_calibration(tbc_path)
            cart_rb_id = args.robot_rigid_body_id
            print(f"[base_intercept] loaded T_B_C from {tbc_path} "
                  f"(cart rigid body {cart_rb_id}) for tracking diagnostic")
        else:
            print(f"[base_intercept] no T_B_C at {tbc_path} — "
                  f"tracking-error diagnostic disabled")

    # Optional arm tracking.
    arm_cal = None
    arm_keys: Optional[OpenSaiCartesianKeys] = None
    arm_face_normal = None
    if args.arm_track:
        arm_cal_path = args.arm_calibration or arm_marker_calibration_path()
        if not os.path.isfile(arm_cal_path):
            print(f"[base_intercept] --arm-track requested but no arm "
                  f"calibration at {arm_cal_path}. Run "
                  f"`verify_arm_calibration.py --init` first.")
            sys.exit(1)
        arm_cal = load_arm_marker_calibration(arm_cal_path)
        arm_keys = OpenSaiCartesianKeys(robot_name=args.arm_robot_name)
        arm_face_normal = np.array([
            args.arm_face_normal_x,
            args.arm_face_normal_y,
            args.arm_face_normal_z,
        ], dtype=float)
        if float(np.linalg.norm(arm_face_normal)) < 1e-6:
            print(f"[base_intercept] --arm-face-normal vector is zero")
            sys.exit(1)
        # Pre-flight: confirm OpenSai is reachable.
        if r.get(arm_keys.current_position) is None:
            print(f"[base_intercept] --arm-track requested but no EE pose at "
                  f"{arm_keys.current_position}. Is OpenSai cartesian_"
                  f"controller running on robot '{args.arm_robot_name}'?")
            sys.exit(1)
        # Pre-flight: confirm arm markers are visible.
        if compute_T_W_A_from_markers(r, arm_cal.marker_specs) is None:
            print(f"[base_intercept] --arm-track requested but arm markers "
                  f"{arm_cal.marker_specs} aren't currently visible. "
                  f"Check Motive / streamer / list_markers.py.")
            sys.exit(1)
        print(f"[base_intercept] arm calibration loaded from {arm_cal_path}")

    try:
        run(
            r=r,
            tracker=tracker,
            ready_pose=ready_pose,
            strike_plane_x=args.strike_plane_x,
            base_y_min=args.base_y_min,
            base_y_max=args.base_y_max,
            rate_hz=args.rate_hz,
            hold_after_lost_s=args.hold_after_lost_s,
            print_interval_s=args.print_interval_s,
            cart_rb_id=cart_rb_id,
            T_B_C=T_B_C,
            arm_cal=arm_cal,
            arm_keys=arm_keys,
            arm_face_normal=arm_face_normal,
            arm_reach_m=args.arm_reach_m,
            arm_z_min=args.arm_z_min_m,
            arm_z_max=args.arm_z_max_m,
        )
    except KeyboardInterrupt:
        print("\n[base_intercept] stopping — leaving last goal in Redis.")
        print(f"               (write {FSM_BASE_GOAL} or restart the bridge "
              f"to reset.)")


if __name__ == "__main__":
    main()
