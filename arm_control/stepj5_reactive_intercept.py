#!/usr/bin/env python3
"""
Step J5: Reactive ball intercept (IK-based, joint controller only).

Architecture
------------
IDLE  →  TRACKING  →  [blocking swing]  →  IDLE

  TRACKING (100 Hz, non-blocking):
    - tracker.update() + predict_intercept(strike_plane_x_world)
    - T_W_A from cart rigid body + arm_base_offset_calibration.json
    - World intercept → arm-frame target  (same transform as cmd_arm_world_clean)
    - Clip to conservative workspace
    - IK-solve for wind-up sweet-spot; write GOAL_JOINTS each tick

  COMMIT (blocking, triggered when TTI ≤ commit_tti):
    - Solve IK for strike + follow positions
    - run_segment(q_cur → q_strike, swing_s)       ~0.3 s
    - run_segment(q_strike → q_follow, swing_s)    ~0.3 s
    - run_segment(q_follow → q_home, return_s)     ~2.0 s

IK sweet-spot formulation
--------------------------
  Standard ikpy FK gives link7 tip.  The paddle sweet spot is 35 cm along
  EE +Z from the flange — a fixed offset *in link7 frame* (calibrated once
  at startup from OpenSai's current_position + sensor joints).

    sweet_spot_A = T[:3,3] + T[:3,:3] @ offset_link7

  IK cost: ||sweet_spot_A(q) - t_A_target||² + w_reg ||q - q_init||²

  t_A_target = R_W_A.T @ (t_W_intercept - t_W_A)   (same as cmd_arm_world_clean)

Safety gates (every IK call, every tick)
-----------------------------------------
  1. Workspace clip: r_xy ≤ reach_m, z ∈ [z_min, z_max]  (arm base frame)
  2. IK convergence: err < ik_tol_m   (skip write, hold last goal otherwise)
  3. Joint limits:   all q within Franka hardware limits
  4. Delta guard:    max per-joint delta from q_cur < max_delta_deg  (tracking only)
  5. Safety torques: abort swing immediately if tripped; slow-return home

z_min = 0.20 m (arm frame) → sweet spot ≥ 0.64 m world Z → paddle tip clears
floor even with handle pointing straight down (tip = 0.54 m > 0.05 m clearance).

Prereqs
-------
  redis-server
  OptiTrack streamer (cart rigid body streaming)
  arm_base_offset_calibration.json  (run calibrate_arm_base_offset.py once)
  OpenSai running with joint_controller AND cartesian_controller in the XML
    (cartesian_controller is only read at startup for offset calibration)

Run from OpenSai root:
  python sports_bot/arm_control/stepj5_reactive_intercept.py
  python sports_bot/arm_control/stepj5_reactive_intercept.py --no-commit
  python sports_bot/arm_control/stepj5_reactive_intercept.py --print-cal-only
  python sports_bot/arm_control/stepj5_reactive_intercept.py --swing-s 0.5
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
import warnings
from enum import Enum, auto

import numpy as np
import redis
from scipy.optimize import minimize

warnings.filterwarnings("ignore")
import ikpy.chain  # noqa: E402

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_OPENSAI_DIR = os.path.dirname(os.path.dirname(_THIS_DIR))
if _OPENSAI_DIR not in sys.path:
    sys.path.insert(0, _OPENSAI_DIR)

# Reuse run_segment and Redis key/helper definitions from J1
from stepj1_joint_nudge import (  # noqa: E402
    ACTIVE_CONTROLLER,
    CURR_POS,
    GOAL_JOINTS,
    SAFETY_TORQUES,
    SENSOR_JOINTS,
    get_vec,
    run_segment,
    set_vec,
)

from sports_bot.state_machine.ball_tracker import BallTracker  # noqa: E402
from sports_bot.state_machine.config import BallTrackerConfig  # noqa: E402
from sports_bot.state_machine.redis_keys import RedisKeys      # noqa: E402
from sports_bot.utils.frames import (                          # noqa: E402
    ArmBaseOffsetCalibration,
    arm_base_offset_calibration_path,
    clip_to_arm_workspace,
    compute_T_W_A_from_base_offset,
    load_arm_base_offset_calibration,
    read_rigid_body_pose_W,
)

# ---------------------------------------------------------------------------
# Robot constants
# ---------------------------------------------------------------------------
ROBOT_NAME = "FrankaRobot"
NS         = f"opensai::controllers::{ROBOT_NAME}"
CART_CTRL  = "cartesian_controller"
JOINT_CTRL = "joint_controller"
URDF_PATH  = "drivers/FrankaPanda/model/panda_arm.urdf"

# Franka hardware joint limits (rad)
Q_LO = np.radians([-166, -101, -166, -176, -166,  -1, -166])
Q_HI = np.radians([ 166,  101,  166,   -4,  166, 215,  166])

# ---------------------------------------------------------------------------
# J5 conservative workspace  (sweet-spot position in arm base frame)
# ---------------------------------------------------------------------------
# Horizontal reach: Franka HW is ~85 cm; 60 cm gives ~25 cm of margin.
# z_min = 0.20 m: sweet spot stays ≥ 0.64 m world-Z, so paddle tip (10 cm
#   past sweet spot) is ≥ 0.54 m — well clear of the floor in any EE pose.
# z_max = 0.80 m: well within comfortable range, avoids arm fully extended up.
# Home / ready pose — free-driven to a forward-reach position, paddle face
# toward opponent, elbow up, comfortable for mid-height volleys (2026-05-28).
# Arm extends forward-left, paddle at roughly shoulder height.
Q_HOME_RAD = np.array([
    -0.0603075, -0.690094, -0.0084536, -2.06741, 0.00632398, 1.48626, -0.793569,
])

J5_REACH_M       = 0.60
J5_Z_MIN_M       = 0.20
J5_Z_MAX_M       = 0.80
J5_IK_TOL_M      = 0.008   # 8 mm — same pass threshold as J3
J5_MAX_DELTA_DEG = 6.0     # per-tick guard against noisy IK target jumps
J5_ACQUIRE_MAX_DELTA_DEG = 20.0  # allow first move from home into tracking
J5_COMMIT_MAX_DELTA_DEG = 15.0  # refuse large blocking strike jumps
J5_WORLD_Z_MIN_M = 0.05       # reject obvious tracker/world-frame outliers
J5_WORLD_Z_MAX_M = 1.40
J5_TARGET_JUMP_MAX_M = 0.04   # max tracking-target jump accepted per tick
J5_TRACKING_STEP_DEG = 0.25    # max commanded joint-goal step per 100 Hz tick
J5_SWING_VEL_FRAC = 0.45      # cap blocking swing peak qdot to this fraction of XML limits
J5_FIXED_ARM_Z_M = 0.45       # J5 bringup: trust lateral prediction, hold Z steady

# joint_controller XML velocity limits currently used by picklebot_j5.xml / picklebot.xml
J5_XML_VEL_LIMIT_RAD_S = np.array([1.2, 1.4, 1.6, 1.8, 1.0, 1.1, 1.2])


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
class State(Enum):
    IDLE     = auto()
    TRACKING = auto()


# ---------------------------------------------------------------------------
# IK helpers
# ---------------------------------------------------------------------------
def build_chain() -> ikpy.chain.Chain:
    return ikpy.chain.Chain.from_urdf_file(URDF_PATH, base_elements=["link0"])


def _fk(chain: ikpy.chain.Chain, q7: np.ndarray) -> np.ndarray:
    return chain.forward_kinematics(np.concatenate([[0.0], q7]))


def _sweet_spot_A(T4x4: np.ndarray, offset_link7: np.ndarray) -> np.ndarray:
    """Sweet-spot position in arm base frame from 4×4 ikpy FK matrix."""
    return T4x4[:3, 3] + T4x4[:3, :3] @ offset_link7


def ik_solve(
    chain: ikpy.chain.Chain,
    t_A_target: np.ndarray,
    q_init: np.ndarray,
    offset_link7: np.ndarray,
    w_reg: float = 0.001,
) -> tuple[np.ndarray, float]:
    """Solve IK so that the sweet spot reaches t_A_target (arm base frame).

    Regularization toward q_init keeps solutions smooth across ticks and
    avoids wrist flips.  Returns (q_rad_7, err_m).
    """
    def cost(q: np.ndarray) -> float:
        T = chain.forward_kinematics(np.concatenate([[0.0], q]))
        ss = T[:3, 3] + T[:3, :3] @ offset_link7
        return float(np.sum((ss - t_A_target) ** 2) + w_reg * np.sum((q - q_init) ** 2))

    result = minimize(cost, q_init, method="L-BFGS-B",
                      bounds=list(zip(Q_LO, Q_HI)),
                      options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-8})
    q_out = result.x
    T = chain.forward_kinematics(np.concatenate([[0.0], q_out]))
    ss = T[:3, 3] + T[:3, :3] @ offset_link7
    return q_out, float(np.linalg.norm(ss - t_A_target))


def _joints_ok(q: np.ndarray) -> bool:
    return bool(np.all(q >= Q_LO) and np.all(q <= Q_HI))


def _delta_ok(q_new: np.ndarray, q_cur: np.ndarray, max_deg: float) -> bool:
    return float(np.max(np.abs(np.degrees(q_new - q_cur)))) <= max_deg


def _rate_limited_goal(q_target: np.ndarray, q_cur: np.ndarray, max_step_deg: float) -> np.ndarray:
    step = math.radians(max_step_deg)
    return q_cur + np.clip(q_target - q_cur, -step, step)


def _move_s_with_velocity_floor(
    q_start: np.ndarray,
    q_target: np.ndarray,
    requested_s: float,
    vel_frac: float,
    label: str,
) -> float:
    """Stretch smoothstep segment duration so peak qdot stays under a limit.

    run_segment uses smoothstep, whose peak slope is 1.5 / move_s.
    """
    vel_cap = np.maximum(J5_XML_VEL_LIMIT_RAD_S * max(vel_frac, 1e-3), 1e-6)
    required_s_by_joint = 1.5 * np.abs(q_target - q_start) / vel_cap
    required_s = float(np.max(required_s_by_joint))
    move_s = max(float(requested_s), required_s)
    if move_s > requested_s + 1e-3:
        i = int(np.argmax(required_s_by_joint))
        peak_deg_s = math.degrees(1.5 * abs(q_target[i] - q_start[i]) / move_s)
        cap_deg_s = math.degrees(vel_cap[i])
        print(
            f"[J5] {label}: stretching move_s {requested_s:.2f}s → {move_s:.2f}s "
            f"for q{i+1} velocity cap ({peak_deg_s:.1f}/{cap_deg_s:.1f} deg/s)"
        )
    return move_s


def _fmt_q(q: np.ndarray) -> str:
    return "  ".join(f"q{i+1}={math.degrees(v):+.1f}" for i, v in enumerate(q))


# ---------------------------------------------------------------------------
# Controller switch
# ---------------------------------------------------------------------------
def switch_ctrl(r: redis.Redis, name: str, timeout: float = 2.0) -> bool:
    t0 = time.monotonic()
    while True:
        r.set(ACTIVE_CONTROLLER, name)
        got = r.get(ACTIVE_CONTROLLER)
        if isinstance(got, bytes):
            got = got.decode()
        if got == name:
            return True
        if time.monotonic() - t0 > timeout:
            print(f"  WARNING: switch to {name!r} timed out", file=sys.stderr)
            return False
        time.sleep(0.02)


# ---------------------------------------------------------------------------
# Startup: calibrate offset_link7
# ---------------------------------------------------------------------------
def calibrate_offset_link7(
    r: redis.Redis,
    chain: ikpy.chain.Chain,
) -> tuple[np.ndarray, np.ndarray]:
    """Derive the sweet-spot offset in ikpy link7 frame.

    Switches to cartesian_controller (read-only), reads current_position
    (the sweet spot, since picklebot.xml has compliantFrame xyz="0 0 0.35")
    and sensor joints simultaneously, then switches back to joint_controller.

    Returns (q_at_cal_rad, offset_link7).  offset_link7 is constant for the
    session — it encodes the 35 cm handle geometry in link7 frame.
    """
    print("[J5 cal] switching to cartesian_controller (read-only) ...")
    # Redis can retain old Cartesian FK values across controller/XML restarts.
    # Clear before switching so calibration pairs sensor joints with a fresh
    # current_position, not a stale value from a previous pose.
    r.delete(CURR_POS)
    switch_ctrl(r, CART_CTRL)

    t_sweet_A = None
    t_wait = time.monotonic()
    while time.monotonic() - t_wait < 2.0:
        t_sweet_A = get_vec(r, CURR_POS, 3)
        if t_sweet_A is not None:
            break
        time.sleep(0.02)

    q_at_cal = get_vec(r, SENSOR_JOINTS, 7)
    if q_at_cal is None:
        sys.exit(f"ERROR: {SENSOR_JOINTS} not available — is OpenSai running?")

    if t_sweet_A is None:
        sys.exit(
            f"ERROR: {CURR_POS} was not freshly published after switching to "
            f"{CART_CTRL}. Is OpenSai running with a cartesian_controller?"
        )

    T_at_cal = _fk(chain, q_at_cal)
    offset_link7 = T_at_cal[:3, :3].T @ (t_sweet_A - T_at_cal[:3, 3])
    mag_m = float(np.linalg.norm(offset_link7))

    print(f"[J5 cal]   q_at_cal:     {_fmt_q(q_at_cal)}")
    print(f"[J5 cal]   sweet_spot_A: [{t_sweet_A[0]:+.4f}, {t_sweet_A[1]:+.4f}, {t_sweet_A[2]:+.4f}] m")
    print(f"[J5 cal]   link7_tip_A:  [{T_at_cal[0,3]:+.4f}, {T_at_cal[1,3]:+.4f}, {T_at_cal[2,3]:+.4f}] m")
    print(f"[J5 cal]   offset_link7: [{offset_link7[0]:+.4f}, {offset_link7[1]:+.4f}, {offset_link7[2]:+.4f}] m  |{mag_m*100:.1f} cm|")

    # Expect ~35-50 cm (sweet spot + any URDF flange offset)
    if not (0.25 < mag_m < 0.60):
        print(
            f"  WARNING: offset magnitude {mag_m*100:.1f} cm is outside expected "
            f"25–60 cm — check URDF or mount", file=sys.stderr,
        )

    # Switch to joint_controller, seed goal = current joints (no lurch)
    print("[J5 cal] switching to joint_controller ...")
    set_vec(r, GOAL_JOINTS, q_at_cal)
    switch_ctrl(r, JOINT_CTRL)
    print("[J5 cal] calibration done.\n")
    return q_at_cal, offset_link7




# ---------------------------------------------------------------------------
# Safety torque check
# ---------------------------------------------------------------------------
def _safety_tripped(r: redis.Redis) -> bool:
    tau = get_vec(r, SAFETY_TORQUES, 7)
    return tau is not None and bool(np.any(np.abs(tau) > 0.01))


# ---------------------------------------------------------------------------
# Workspace-clipped IK target
# ---------------------------------------------------------------------------
def _safe_target(
    t_A_raw: np.ndarray,
    reach_m: float,
    z_min: float,
    z_max: float,
) -> tuple[np.ndarray, bool]:
    return clip_to_arm_workspace(t_A_raw, r_max=reach_m, z_min=z_min, z_max=z_max)


# ---------------------------------------------------------------------------
# Main intercept loop
# ---------------------------------------------------------------------------
def run_loop(
    r: redis.Redis,
    chain: ikpy.chain.Chain,
    tracker: BallTracker,
    q_home_rad: np.ndarray,
    offset_link7: np.ndarray,
    cal: ArmBaseOffsetCalibration,
    *,
    strike_plane_x_world: float,
    commit_tti: float,
    wind_up_offset_m: float,
    follow_offset_m: float,
    swing_s: float,
    return_s: float,
    rate_hz: float,
    no_commit: bool,
    reach_m: float,
    z_min: float,
    z_max: float,
    commit_max_delta_deg: float,
    world_z_min_m: float,
    world_z_max_m: float,
    target_jump_max_m: float,
    tracking_step_deg: float,
    swing_vel_frac: float,
    z_mode: str,
    fixed_arm_z_m: float,
) -> None:
    dt = 1.0 / rate_hz
    state = State.IDLE
    last_good_t = float("-inf")
    hold_after_lost_s = 0.5
    last_print_t = float("-inf")
    print_interval_s = 0.20
    last_wu_q: np.ndarray | None = None  # last valid wind-up IK solution
    last_track_target_A: np.ndarray | None = None

    switch_ctrl(r, JOINT_CTRL)
    set_vec(r, GOAL_JOINTS, q_home_rad)

    print(f"[J5] running at {rate_hz:.0f} Hz  —  Ctrl-C to stop")
    print(f"[J5] strike_plane_x={strike_plane_x_world:+.3f} m  "
          f"commit_tti={commit_tti:.3f} s  swing_s={swing_s:.2f} s")
    print(f"[J5] workspace (arm frame):  r_xy ≤ {reach_m:.2f} m  "
          f"z ∈ [{z_min:+.2f}, {z_max:+.2f}] m")
    print(f"[J5] commit guards: max Δ={commit_max_delta_deg:.1f}°  "
          f"world_z ∈ [{world_z_min_m:.2f}, {world_z_max_m:.2f}] m  no clipped commits")
    print(f"[J5] tracking guards: acquire Δ≤{J5_ACQUIRE_MAX_DELTA_DEG:.1f}°  "
          f"track Δ≤{J5_MAX_DELTA_DEG:.1f}°/tick  target jump ≤ {target_jump_max_m*100:.0f} cm  "
          f"goal step≤{tracking_step_deg:.2f}°/tick  "
          f"swing vel≤{swing_vel_frac*100:.0f}% XML  "
          f"z_mode={z_mode}" + (f"({fixed_arm_z_m:.2f}m A)" if z_mode == "fixed-arm" else ""))
    if no_commit:
        print("[J5] --no-commit: tracking only, swing disabled")
    print()

    while True:
        t0 = time.perf_counter()

        # ----------------------------------------------------------------
        # 1. Ball tracking
        # ----------------------------------------------------------------
        tracker.update()
        intercept = tracker.predict_intercept(strike_plane_x_world)

        # ----------------------------------------------------------------
        # 2. Arm base frame (cart rigid body + calibrated offset)
        # ----------------------------------------------------------------
        T_W_B = read_rigid_body_pose_W(r, cal.base_rigid_body_id)
        if T_W_B is None:
            time.sleep(max(0.0, dt - (time.perf_counter() - t0)))
            continue
        R_W_A, t_W_A = compute_T_W_A_from_base_offset(T_W_B, cal)

        # ----------------------------------------------------------------
        # 3. Ball lost — decay to home
        # ----------------------------------------------------------------
        if intercept is None:
            since_good = t0 - last_good_t
            if state == State.TRACKING and since_good > hold_after_lost_s:
                set_vec(r, GOAL_JOINTS, q_home_rad)
                reason = getattr(tracker, "last_reject_reason", "") or "no_data"
                print(f"[J5] ball lost {since_good:.2f}s (reason={reason}) → home")
                state = State.IDLE
                last_wu_q = None
                last_track_target_A = None
            elif state == State.IDLE:
                set_vec(r, GOAL_JOINTS, q_home_rad)
            time.sleep(max(0.0, dt - (time.perf_counter() - t0)))
            continue

        last_good_t = t0
        tti = float(intercept.time_to_impact)
        t_W_strike = np.asarray(intercept.position, dtype=float)

        # ----------------------------------------------------------------
        # 4. World → arm-base frame  (identical to cmd_arm_world_clean)
        # ----------------------------------------------------------------
        t_A_strike_raw = R_W_A.T @ (t_W_strike - t_W_A)
        if z_mode == "fixed-arm":
            t_A_strike_raw[2] = fixed_arm_z_m
        t_A_windup_raw = t_A_strike_raw - np.array([wind_up_offset_m, 0.0, 0.0])

        t_A_windup, wu_clipped  = _safe_target(t_A_windup_raw,  reach_m, z_min, z_max)
        t_A_strike, st_clipped  = _safe_target(t_A_strike_raw,  reach_m, z_min, z_max)
        t_A_follow, fw_clipped  = _safe_target(
            t_A_strike + np.array([follow_offset_m, 0.0, 0.0]), reach_m, z_min, z_max)

        # ----------------------------------------------------------------
        # 5. Diagnostic print
        # ----------------------------------------------------------------
        if t0 - last_print_t >= print_interval_s:
            last_print_t = t0
            clip_tag = " [wu CLIP]" if wu_clipped else ""
            print(
                f"[J5] {state.name:8s}  tti={tti:+.3f}s  "
                f"strike_W=[{t_W_strike[0]:+.3f},{t_W_strike[1]:+.3f},{t_W_strike[2]:+.3f}]  "
                f"wu_A=[{t_A_windup[0]:+.3f},{t_A_windup[1]:+.3f},{t_A_windup[2]:+.3f}]"
                f"{clip_tag}"
            )

        q_cur = get_vec(r, SENSOR_JOINTS, 7)
        if q_cur is None:
            time.sleep(max(0.0, dt - (time.perf_counter() - t0)))
            continue

        # ----------------------------------------------------------------
        # 6a. TRACKING — solve IK for wind-up, write directly each tick
        # ----------------------------------------------------------------
        if tti > commit_tti or no_commit:
            if (last_track_target_A is not None
                    and np.linalg.norm(t_A_windup - last_track_target_A) > target_jump_max_m):
                time.sleep(max(0.0, dt - (time.perf_counter() - t0)))
                continue

            q_seed = last_wu_q if last_wu_q is not None else q_cur
            q_wu, err_wu = ik_solve(chain, t_A_windup, q_seed, offset_link7)

            max_delta_deg = J5_ACQUIRE_MAX_DELTA_DEG if state == State.IDLE else J5_MAX_DELTA_DEG
            if (err_wu < J5_IK_TOL_M
                    and _joints_ok(q_wu)
                    and _delta_ok(q_wu, q_cur, max_delta_deg)):
                q_cmd = _rate_limited_goal(q_wu, q_cur, tracking_step_deg)
                set_vec(r, GOAL_JOINTS, q_cmd)
                last_wu_q = q_wu
                last_track_target_A = t_A_windup.copy()
                state = State.TRACKING

        # ----------------------------------------------------------------
        # 6b. COMMIT — blocking swing sequence
        # ----------------------------------------------------------------
        else:
            if state != State.TRACKING or last_wu_q is None:
                q_wu, err_wu = ik_solve(chain, t_A_windup, q_cur, offset_link7)
                if (err_wu < J5_IK_TOL_M
                        and _joints_ok(q_wu)
                        and _delta_ok(q_wu, q_cur, J5_ACQUIRE_MAX_DELTA_DEG)):
                    q_cmd = _rate_limited_goal(q_wu, q_cur, tracking_step_deg)
                    set_vec(r, GOAL_JOINTS, q_cmd)
                    last_wu_q = q_wu
                    last_track_target_A = t_A_windup.copy()
                    state = State.TRACKING
                else:
                    set_vec(r, GOAL_JOINTS, q_home_rad)
                    state = State.IDLE
                    last_wu_q = None
                time.sleep(max(0.0, dt - (time.perf_counter() - t0)))
                continue

            if not (world_z_min_m <= t_W_strike[2] <= world_z_max_m):
                print(
                    f"[J5] COMMIT rejected: strike_W.z={t_W_strike[2]:+.3f} m "
                    f"outside [{world_z_min_m:.2f}, {world_z_max_m:.2f}]"
                )
                set_vec(r, GOAL_JOINTS, q_home_rad)
                state = State.IDLE
                last_wu_q = None
                last_track_target_A = None
                time.sleep(max(0.0, dt - (time.perf_counter() - t0)))
                continue

            if wu_clipped or st_clipped or fw_clipped:
                clips = []
                if wu_clipped:
                    clips.append("windup")
                if st_clipped:
                    clips.append("strike")
                if fw_clipped:
                    clips.append("follow")
                print(f"[J5] COMMIT rejected: clipped target(s) {','.join(clips)}")
                set_vec(r, GOAL_JOINTS, q_home_rad)
                state = State.IDLE
                last_wu_q = None
                last_track_target_A = None
                time.sleep(max(0.0, dt - (time.perf_counter() - t0)))
                continue

            # Solve strike IK seeded from last wind-up solution (smooth)
            q_seed = last_wu_q if last_wu_q is not None else q_cur
            q_strike, err_st = ik_solve(chain, t_A_strike, q_seed, offset_link7)
            q_follow, err_fw = ik_solve(chain, t_A_follow, q_strike, offset_link7)
            commit_delta_deg = float(np.degrees(np.abs(q_strike - q_cur)).max())

            if not _joints_ok(q_strike) or err_st > J5_IK_TOL_M * 3:
                print(f"[J5] COMMIT IK failed (err={err_st*1000:.1f} mm) — aborting")
                time.sleep(max(0.0, dt - (time.perf_counter() - t0)))
                continue
            if commit_delta_deg > commit_max_delta_deg:
                print(
                    f"[J5] COMMIT rejected: q_cur→q_strike max Δ={commit_delta_deg:.1f}° "
                    f"> {commit_max_delta_deg:.1f}°"
                )
                set_vec(r, GOAL_JOINTS, q_home_rad)
                state = State.IDLE
                last_wu_q = None
                last_track_target_A = None
                time.sleep(max(0.0, dt - (time.perf_counter() - t0)))
                continue

            print(f"\n{'='*60}")
            print(f"[J5] COMMIT  tti={tti:.3f}s")
            print(f"[J5]   strike_A=[{t_A_strike[0]:+.3f},{t_A_strike[1]:+.3f},{t_A_strike[2]:+.3f}]  "
                  f"err={err_st*1000:.1f} mm")
            print(f"[J5]   follow_A=[{t_A_follow[0]:+.3f},{t_A_follow[1]:+.3f},{t_A_follow[2]:+.3f}]  "
                  f"err={err_fw*1000:.1f} mm")
            print(f"[J5]   q_cur  → q_strike max Δ={commit_delta_deg:.1f}°")
            print(f"{'='*60}\n")

            switch_ctrl(r, JOINT_CTRL)

            # Wind-up → strike
            wu_strike_s = _move_s_with_velocity_floor(q_cur, q_strike, swing_s, swing_vel_frac, "wu→strike")
            seg = run_segment(r, "wu→strike", q_cur, q_strike,
                              move_s=wu_strike_s, hold_s=0.10, publish_hz=200.0)

            if seg["goal_err_max_deg"] > 5.0:
                print(
                    f"[J5] *** wu→strike did not track "
                    f"(goal_err={seg['goal_err_max_deg']:.1f}°) — holding current pose, no follow-through ***"
                )
                q_post = get_vec(r, SENSOR_JOINTS, 7)
                if q_post is not None:
                    set_vec(r, GOAL_JOINTS, q_post)
                state = State.IDLE
                last_wu_q = None
                last_track_target_A = None
                continue

            if _safety_tripped(r):
                print("[J5] *** SAFETY TORQUES after strike — aborting, slow return ***")
                q_post = get_vec(r, SENSOR_JOINTS, 7)
                if q_post is None:
                    q_post = q_strike
                safe_return_s = _move_s_with_velocity_floor(q_post, q_home_rad, return_s * 2, swing_vel_frac, "safe_return")
                run_segment(r, "safe_return", q_post, q_home_rad,
                            move_s=safe_return_s, hold_s=1.0, publish_hz=100.0)
                state = State.IDLE
                last_wu_q = None
                set_vec(r, GOAL_JOINTS, q_home_rad)
                continue

            # Strike → follow-through
            strike_follow_s = _move_s_with_velocity_floor(q_strike, q_follow, swing_s, swing_vel_frac, "strike→follow")
            run_segment(r, "strike→follow", q_strike, q_follow,
                        move_s=strike_follow_s, hold_s=0.30, publish_hz=200.0)

            # Follow → home
            follow_home_s = _move_s_with_velocity_floor(q_follow, q_home_rad, return_s, swing_vel_frac, "follow→home")
            run_segment(r, "follow→home", q_follow, q_home_rad,
                        move_s=follow_home_s, hold_s=1.0, publish_hz=100.0)

            state = State.IDLE
            last_wu_q = None
            last_track_target_A = None
            set_vec(r, GOAL_JOINTS, q_home_rad)

        time.sleep(max(0.0, dt - (time.perf_counter() - t0)))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="J5: Reactive IK-based ball intercept (joint controller).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Ball / geometry
    ap.add_argument("--ball-rigid-body-id", type=int, default=1,
                    help="OptiTrack streaming ID for the pickleball.")
    ap.add_argument("--strike-plane-x", type=float, default=0.65,
                    help="World-X of the strike plane (m).")
    ap.add_argument("--wind-up-offset", type=float, default=0.15,
                    help="Pull-back behind strike in arm +X (m).")
    ap.add_argument("--follow-offset", type=float, default=0.12,
                    help="Follow-through past strike in arm +X (m).")

    # Timing
    ap.add_argument("--swing-s", type=float, default=0.30,
                    help="Move time for wu→strike and strike→follow (s).")
    ap.add_argument("--commit-tti", type=float, default=None,
                    help="TTI threshold to commit to swing (s). "
                         "Default: swing_s + 0.05.")
    ap.add_argument("--return-s", type=float, default=2.0,
                    help="Move time for follow→home (s).")

    # Workspace safety
    ap.add_argument("--reach-m", type=float, default=J5_REACH_M,
                    help="Max horizontal reach for IK targets in arm frame (m).")
    ap.add_argument("--z-min-m", type=float, default=J5_Z_MIN_M,
                    help="Min arm-frame Z for IK targets (m).")
    ap.add_argument("--z-max-m", type=float, default=J5_Z_MAX_M,
                    help="Max arm-frame Z for IK targets (m).")
    ap.add_argument("--commit-max-delta-deg", type=float, default=J5_COMMIT_MAX_DELTA_DEG,
                    help="Refuse blocking swing if any joint is farther than this from strike.")
    ap.add_argument("--world-z-min-m", type=float, default=J5_WORLD_Z_MIN_M,
                    help="Reject commit if predicted world-frame strike Z is below this.")
    ap.add_argument("--world-z-max-m", type=float, default=J5_WORLD_Z_MAX_M,
                    help="Reject commit if predicted world-frame strike Z is above this.")
    ap.add_argument("--target-jump-max-m", type=float, default=J5_TARGET_JUMP_MAX_M,
                    help="Reject tracking updates whose arm-frame target jumps farther than this.")
    ap.add_argument("--tracking-step-deg", type=float, default=J5_TRACKING_STEP_DEG,
                    help="Max per-tick joint-goal step during non-blocking tracking.")
    ap.add_argument("--swing-vel-frac", type=float, default=J5_SWING_VEL_FRAC,
                    help="Max blocking-swing peak velocity as a fraction of XML velocity limits.")
    ap.add_argument("--z-mode", choices=["fixed-arm", "predicted"], default="fixed-arm",
                    help="Use fixed arm-frame Z for J5 bringup, or raw predicted Z.")
    ap.add_argument("--fixed-arm-z-m", type=float, default=J5_FIXED_ARM_Z_M,
                    help="Arm-frame strike Z used when --z-mode=fixed-arm.")

    # Loop behaviour
    ap.add_argument("--rate-hz", type=float, default=100.0,
                    help="Outer loop rate.")
    ap.add_argument("--no-commit", action="store_true",
                    help="Tracking only — never execute the swing. "
                         "Good for verifying IK/tracking before first throw.")
    ap.add_argument("--print-cal-only", action="store_true",
                    help="Calibrate offset_link7, print it, and exit.")

    # Calibration / infra
    ap.add_argument("--calibration", default=None,
                    help="Path to arm_base_offset_calibration.json.")
    ap.add_argument("--robot-name", default="FrankaRobot")
    ap.add_argument("--redis-host", default="localhost")
    ap.add_argument("--redis-port", type=int, default=6379)

    # Ball tracker overrides
    ap.add_argument("--min-lookahead", type=float, default=None)
    ap.add_argument("--max-lookahead", type=float, default=None)

    args = ap.parse_args()

    commit_tti = args.commit_tti if args.commit_tti is not None else args.swing_s + 0.05
    if commit_tti <= 0:
        ap.error("commit_tti must be positive")

    # ---- Redis ----
    r = redis.Redis(host=args.redis_host, port=args.redis_port,
                    decode_responses=True)
    try:
        r.ping()
    except redis.exceptions.ConnectionError as e:
        sys.exit(f"[J5] cannot reach Redis at {args.redis_host}:{args.redis_port}: {e}")

    # ---- Calibration file ----
    cal_path = args.calibration or arm_base_offset_calibration_path()
    if not os.path.isfile(cal_path):
        sys.exit(
            f"[J5] arm_base_offset_calibration.json not found: {cal_path}\n"
            f"      Run: python sports_bot/scripts/calibrate_arm_base_offset.py"
        )
    cal = load_arm_base_offset_calibration(cal_path)
    print(f"[J5] arm base calibration: {cal_path}  (rigid body {cal.base_rigid_body_id})")

    # Verify cart is visible before doing anything
    if read_rigid_body_pose_W(r, cal.base_rigid_body_id) is None:
        sys.exit(f"[J5] cart rigid body {cal.base_rigid_body_id} not visible in Redis. "
                 f"Check OptiTrack streamer.")

    # ---- Build ikpy chain ----
    if not os.path.isfile(URDF_PATH):
        sys.exit(f"[J5] URDF not found: {URDF_PATH}  (run from OpenSai root)")
    chain = build_chain()

    # ---- Calibrate offset_link7 at current pose ----
    q_at_cal, offset_link7 = calibrate_offset_link7(r, chain)

    if args.print_cal_only:
        print("[J5] --print-cal-only: done.")
        return

    # ---- Move to home before tracking ----
    print("[J5] moving to home pose ...")
    run_segment(r, "→home", q_at_cal, Q_HOME_RAD,
                move_s=3.0, hold_s=1.0, publish_hz=100.0)
    q_home_rad = Q_HOME_RAD
    set_vec(r, GOAL_JOINTS, q_home_rad)
    print("[J5] at home.\n")

    # ---- Ball tracker ----
    cfg = BallTrackerConfig()
    if args.min_lookahead is not None:
        cfg.min_lookahead = args.min_lookahead
    if args.max_lookahead is not None:
        cfg.max_lookahead = args.max_lookahead

    keys = RedisKeys(ball_source="optitrack")
    keys.ball.__dict__["optitrack_rigid_body_id"] = args.ball_rigid_body_id

    ball_key = keys.ball.optitrack_position
    if r.get(ball_key) is None:
        print(f"[J5] WARNING: {ball_key} is empty — is the OptiTrack streamer running?")
    else:
        print(f"[J5] reading ball from {ball_key}")

    tracker = BallTracker(r, keys, cfg)

    # ---- Go ----
    try:
        run_loop(
            r=r,
            chain=chain,
            tracker=tracker,
            q_home_rad=q_home_rad,
            offset_link7=offset_link7,
            cal=cal,
            strike_plane_x_world=args.strike_plane_x,
            commit_tti=commit_tti,
            wind_up_offset_m=args.wind_up_offset,
            follow_offset_m=args.follow_offset,
            swing_s=args.swing_s,
            return_s=args.return_s,
            rate_hz=args.rate_hz,
            no_commit=args.no_commit,
            reach_m=args.reach_m,
            z_min=args.z_min_m,
            z_max=args.z_max_m,
            commit_max_delta_deg=args.commit_max_delta_deg,
            world_z_min_m=args.world_z_min_m,
            world_z_max_m=args.world_z_max_m,
            target_jump_max_m=args.target_jump_max_m,
            tracking_step_deg=args.tracking_step_deg,
            swing_vel_frac=args.swing_vel_frac,
            z_mode=args.z_mode,
            fixed_arm_z_m=args.fixed_arm_z_m,
        )
    except KeyboardInterrupt:
        print("\n[J5] stopped — leaving last joint goal in Redis.")


if __name__ == "__main__":
    main()
