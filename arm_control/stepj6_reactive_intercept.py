#!/usr/bin/env python3
"""
Step J6: Reactive ball intercept (IK-based, joint controller only).

Fork of ``stepj5_reactive_intercept.py`` — edit this file to iterate without
changing the J5 baseline.

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

  IK cost: ||sweet_spot_A(q) - t_A_target||² + w_ori·angle(R, R_home)² + w_reg||q-q_init||²

  By default R_home is link7 orientation at Q_HOME_RAD — same paddle face / net
  parallelism as the resting backswing pose. Pass --no-fixed-paddle-ori to revert
  to position-only IK.

  t_A_target = R_W_A.T @ (t_W_intercept - t_W_A)   (same as cmd_arm_world_clean)

Safety gates (every IK call, every tick)
-----------------------------------------
  1. Workspace clip: r_xy ≤ reach_m (default 0.78 m), z ∈ [z_min, z_max]  (arm base frame)
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
  python sports_bot/arm_control/stepj6_reactive_intercept.py
  python sports_bot/arm_control/stepj6_reactive_intercept.py --no-commit
  python sports_bot/arm_control/stepj6_reactive_intercept.py --print-cal-only
  python sports_bot/arm_control/stepj6_reactive_intercept.py --swing-s 0.5

Mock intercept (no ball tracker / optional no OptiTrack):
  python sports_bot/arm_control/stepj6_reactive_intercept.py --no-commit \\
      --mock-intercept 0.65 0.0 0.85 --mock-tti 0.8
  python sports_bot/arm_control/stepj6_reactive_intercept.py --no-commit \\
      --mock-intercept 0.45 0.0 0.45 --mock-tti 0.8 --mock-identity-base
  python sports_bot/arm_control/stepj6_reactive_intercept.py \\
      --mock-intercept 0.65 0.0 0.85 --mock-tti 0.25   # triggers swing
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
    ensure_joint_controller,
    get_vec,
    run_segment,
    set_vec,
)

from sports_bot.state_machine.ball_tracker import BallTracker, Intercept  # noqa: E402
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
# J6 conservative workspace  (sweet-spot position in arm base frame)
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

J6_REACH_M       = 0.78   # Franka HW ~0.85 m; 0.78 keeps ~7 cm margin vs 0.60 bring-up default
J6_Z_MIN_M       = 0.15
J6_Z_MAX_M       = 0.85
J6_IK_TOL_M      = 0.012  # 12 mm — relaxed from 8 mm so clipped/near-limit targets track
J6_MAX_DELTA_DEG = 10.0   # per-tick guard against noisy IK target jumps
J6_ACQUIRE_MAX_DELTA_DEG = 40.0  # first move from home into tracking (was 20° — caused IDLE)
J6_COMMIT_MAX_DELTA_DEG = 15.0  # refuse large blocking strike jumps
J6_WORLD_Z_MIN_M = 0.05       # reject obvious tracker/world-frame outliers
J6_WORLD_Z_MAX_M = 1.40
J6_TARGET_JUMP_MAX_M = 0.08   # max tracking-target jump accepted per tick
J6_TRACKING_STEP_DEG = 0.25    # max commanded joint-goal step per 100 Hz tick
J6_SWING_VEL_FRAC = 0.45      # cap blocking swing peak qdot to this fraction of XML limits
J6_HOME_VEL_FRAC = 0.85       # startup / return-home velocity cap fraction
J6_HOME_SETTLE_TOL_DEG = 5.0  # accept settled joints as home if within this of nominal
J6_FIXED_ARM_Z_M = 0.45       # J6 bringup: trust lateral prediction, hold Z steady
J6_W_ORI = 2.0                # IK weight on (link7 rotation error from home)²
J6_MAX_ORI_ERR_DEG = 8.0      # reject IK if link7 rotates farther than this from home

# joint_controller XML velocity limits currently used by picklebot_j5.xml / picklebot.xml
J6_XML_VEL_LIMIT_RAD_S = np.array([1.2, 1.4, 1.6, 1.8, 1.0, 1.1, 1.2])


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


def _rot_angle_rad(R_from: np.ndarray, R_to: np.ndarray) -> float:
    """Geodesic angle (rad) from R_from to R_to."""
    R_err = R_from.T @ R_to
    c = (float(np.trace(R_err)) - 1.0) * 0.5
    c = float(np.clip(c, -1.0, 1.0))
    return float(np.arccos(c))


def _link7_R_A(chain: ikpy.chain.Chain, q7: np.ndarray) -> np.ndarray:
    return _fk(chain, q7)[:3, :3]


def ik_solve(
    chain: ikpy.chain.Chain,
    t_A_target: np.ndarray,
    q_init: np.ndarray,
    offset_link7: np.ndarray,
    w_reg: float = 0.001,
    R_A_link7_target: np.ndarray | None = None,
    w_ori: float = J6_W_ORI,
) -> tuple[np.ndarray, float, float]:
    """Solve IK so that the sweet spot reaches t_A_target (arm base frame).

    When ``R_A_link7_target`` is set, penalize deviation of link7 orientation
    from that matrix so the paddle keeps the same face/net alignment as home.

    Returns (q_rad_7, pos_err_m, ori_err_deg).
    """
    def cost(q: np.ndarray) -> float:
        T = chain.forward_kinematics(np.concatenate([[0.0], q]))
        ss = T[:3, 3] + T[:3, :3] @ offset_link7
        pos_cost = float(np.sum((ss - t_A_target) ** 2))
        reg = w_reg * float(np.sum((q - q_init) ** 2))
        ori_cost = 0.0
        if R_A_link7_target is not None:
            ang = _rot_angle_rad(R_A_link7_target, T[:3, :3])
            ori_cost = w_ori * (ang * ang)
        return pos_cost + reg + ori_cost

    result = minimize(cost, q_init, method="L-BFGS-B",
                      bounds=list(zip(Q_LO, Q_HI)),
                      options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-8})
    q_out = result.x
    T = chain.forward_kinematics(np.concatenate([[0.0], q_out]))
    ss = T[:3, 3] + T[:3, :3] @ offset_link7
    pos_err = float(np.linalg.norm(ss - t_A_target))
    ori_err_deg = 0.0
    if R_A_link7_target is not None:
        ori_err_deg = math.degrees(_rot_angle_rad(R_A_link7_target, T[:3, :3]))
    return q_out, pos_err, ori_err_deg


def _ik_ok(
    pos_err: float,
    ori_err_deg: float,
    pos_tol_m: float,
    max_ori_err_deg: float,
    R_A_link7_target: np.ndarray | None,
) -> bool:
    if pos_err >= pos_tol_m:
        return False
    if R_A_link7_target is not None and ori_err_deg > max_ori_err_deg:
        return False
    return True


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
    vel_cap = np.maximum(J6_XML_VEL_LIMIT_RAD_S * max(vel_frac, 1e-3), 1e-6)
    required_s_by_joint = 1.5 * np.abs(q_target - q_start) / vel_cap
    required_s = float(np.max(required_s_by_joint))
    move_s = max(float(requested_s), required_s)
    if move_s > requested_s + 1e-3:
        i = int(np.argmax(required_s_by_joint))
        peak_deg_s = math.degrees(1.5 * abs(q_target[i] - q_start[i]) / move_s)
        cap_deg_s = math.degrees(vel_cap[i])
        print(
            f"[J6] {label}: stretching move_s {requested_s:.2f}s → {move_s:.2f}s "
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
    print("[J6 cal] switching to cartesian_controller (read-only) ...")
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

    print(f"[J6 cal]   q_at_cal:     {_fmt_q(q_at_cal)}")
    print(f"[J6 cal]   sweet_spot_A: [{t_sweet_A[0]:+.4f}, {t_sweet_A[1]:+.4f}, {t_sweet_A[2]:+.4f}] m")
    print(f"[J6 cal]   link7_tip_A:  [{T_at_cal[0,3]:+.4f}, {T_at_cal[1,3]:+.4f}, {T_at_cal[2,3]:+.4f}] m")
    print(f"[J6 cal]   offset_link7: [{offset_link7[0]:+.4f}, {offset_link7[1]:+.4f}, {offset_link7[2]:+.4f}] m  |{mag_m*100:.1f} cm|")

    # Expect ~35-50 cm (sweet spot + any URDF flange offset)
    if not (0.25 < mag_m < 0.60):
        print(
            f"  WARNING: offset magnitude {mag_m*100:.1f} cm is outside expected "
            f"25–60 cm — check URDF or mount", file=sys.stderr,
        )

    # Switch to joint_controller, seed goal = current joints (no lurch)
    print("[J6 cal] switching to joint_controller ...")
    set_vec(r, GOAL_JOINTS, q_at_cal)
    if not ensure_joint_controller(r, timeout_s=2.0):
        sys.exit(
            f"ERROR: could not activate {JOINT_CTRL} after calibration — "
            "is OpenSai running with joint_controller in the XML?"
        )
    print("[J6 cal] calibration done.\n")
    return q_at_cal, offset_link7


def move_to_home_pose(
    r: redis.Redis,
    q_nominal_home: np.ndarray,
    *,
    move_s: float,
    hold_s: float,
    vel_frac: float,
) -> np.ndarray:
    """Smoothstep move from live sensor joints to nominal home.

    Re-reads sensors (not stale calibration samples), ensures joint_controller
    is active, and stretches duration for large joint deltas.  Never treats a
    failed move as home — keeps commanding Q_HOME until settled within tol.
    """
    q_cur = get_vec(r, SENSOR_JOINTS, 7)
    if q_cur is None:
        sys.exit(f"[J6] cannot read {SENSOR_JOINTS} for home move")

    max_delta_deg = float(np.max(np.abs(np.degrees(q_nominal_home - q_cur))))
    print(f"[J6] home move: max joint delta {max_delta_deg:.1f}° from nominal")
    if max_delta_deg < 2.0:
        print("[J6] already near home — holding current joints")
        set_vec(r, GOAL_JOINTS, q_cur)
        ensure_joint_controller(r, timeout_s=2.0)
        return q_nominal_home.copy()

    set_vec(r, GOAL_JOINTS, q_cur)
    if not ensure_joint_controller(r, timeout_s=2.0):
        sys.exit(
            f"[J6] {JOINT_CTRL} not active — cannot move to home. "
            f"Check OpenSai and active_controller_name in Redis."
        )
    time.sleep(0.05)

    home_s = _move_s_with_velocity_floor(
        q_cur, q_nominal_home, move_s, vel_frac, "→home",
    )
    run_segment(
        r, "→home", q_cur, q_nominal_home,
        move_s=home_s, hold_s=hold_s, publish_hz=100.0,
    )

    q_settled = get_vec(r, SENSOR_JOINTS, 7)
    if q_settled is None:
        q_settled = q_nominal_home.copy()

    err_deg = float(np.max(np.abs(np.degrees(q_nominal_home - q_settled))))
    if err_deg > J6_HOME_SETTLE_TOL_DEG:
        print(
            f"[J6] WARNING: home move incomplete — max joint error {err_deg:.1f}° "
            f"(need ≤ {J6_HOME_SETTLE_TOL_DEG:.1f}°). "
            "Keeping nominal home as goal; tracking may be degraded.",
            file=sys.stderr,
        )
        set_vec(r, GOAL_JOINTS, q_nominal_home)
        return q_nominal_home.copy()

    set_vec(r, GOAL_JOINTS, q_settled)
    print(f"[J6] at home (max joint error {err_deg:.2f}° from nominal).\n")
    return q_settled.copy()




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
    tracker: BallTracker | None,
    q_home_rad: np.ndarray,
    offset_link7: np.ndarray,
    R_A_link7_home: np.ndarray | None,
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
    acquire_max_delta_deg: float,
    max_delta_deg: float,
    ik_tol_m: float,
    z_mode: str,
    fixed_arm_z_m: float,
    w_ori: float,
    max_ori_err_deg: float,
    mock_intercepts: list[np.ndarray] | None = None,
    mock_tti: float = 0.50,
    mock_cycle_s: float = 0.0,
    mock_identity_base: bool = False,
) -> None:
    dt = 1.0 / rate_hz
    state = State.IDLE
    last_good_t = float("-inf")
    hold_after_lost_s = 0.5
    last_print_t = float("-inf")
    print_interval_s = 0.20
    last_wu_q: np.ndarray | None = None  # last valid wind-up IK solution
    last_track_target_A: np.ndarray | None = None
    mock_t0 = time.monotonic()
    using_mock = mock_intercepts is not None and len(mock_intercepts) > 0

    switch_ctrl(r, JOINT_CTRL)
    set_vec(r, GOAL_JOINTS, q_home_rad)

    print(f"[J6] running at {rate_hz:.0f} Hz  —  Ctrl-C to stop")
    print(f"[J6] strike_plane_x={strike_plane_x_world:+.3f} m  "
          f"commit_tti={commit_tti:.3f} s  swing_s={swing_s:.2f} s")
    print(f"[J6] workspace (arm frame):  r_xy ≤ {reach_m:.2f} m  "
          f"z ∈ [{z_min:+.2f}, {z_max:+.2f}] m")
    print(f"[J6] commit guards: max Δ={commit_max_delta_deg:.1f}°  "
          f"world_z ∈ [{world_z_min_m:.2f}, {world_z_max_m:.2f}] m  no clipped commits")
    print(f"[J6] tracking guards: acquire Δ≤{acquire_max_delta_deg:.1f}°  "
          f"track Δ≤{max_delta_deg:.1f}°/tick  ik_tol={ik_tol_m*1000:.0f} mm  "
          f"target jump ≤ {target_jump_max_m*100:.0f} cm  "
          f"goal step≤{tracking_step_deg:.2f}°/tick  "
          f"swing vel≤{swing_vel_frac*100:.0f}% XML  "
          f"z_mode={z_mode}" + (f"({fixed_arm_z_m:.2f}m A)" if z_mode == "fixed-arm" else ""))
    if no_commit:
        print("[J6] --no-commit: tracking only, swing disabled")
    if R_A_link7_home is not None:
        face_A = R_A_link7_home @ (offset_link7 / max(np.linalg.norm(offset_link7), 1e-9))
        print("[J6] fixed paddle orientation: link7 rotation locked to home")
        print(f"[J6]   strike-face normal_A ≈ [{face_A[0]:+.3f},{face_A[1]:+.3f},{face_A[2]:+.3f}]  "
              f"w_ori={w_ori:.2f}  max_ori_err={max_ori_err_deg:.1f}°")
    else:
        print("[J6] paddle orientation: position-only IK (--no-fixed-paddle-ori)")
    if using_mock:
        pts = ", ".join(
            f"[{p[0]:+.2f},{p[1]:+.2f},{p[2]:+.2f}]" for p in mock_intercepts
        )
        print(f"[J6] MOCK intercept(s) world (m): {pts}")
        print(f"[J6]   mock_tti={mock_tti:.3f}s  cycle={mock_cycle_s:.1f}s  "
              f"identity_base={mock_identity_base}")
    print()

    while True:
        t0 = time.perf_counter()

        # ----------------------------------------------------------------
        # 1. Ball tracking (or mock intercept)
        # ----------------------------------------------------------------
        if using_mock:
            idx = 0
            if mock_cycle_s > 0.0 and len(mock_intercepts) > 1:
                idx = int((time.monotonic() - mock_t0) / mock_cycle_s) % len(mock_intercepts)
            t_W_strike = np.asarray(mock_intercepts[idx], dtype=float)
            intercept = Intercept(
                position=t_W_strike.copy(),
                velocity=np.array([-5.0, 0.0, 0.0]),
                time_to_impact=float(mock_tti),
                n_bounces=0,
            )
        else:
            assert tracker is not None
            tracker.update()
            intercept = tracker.predict_intercept(strike_plane_x_world)

        # ----------------------------------------------------------------
        # 2. Arm base frame (cart rigid body + calibrated offset)
        # ----------------------------------------------------------------
        if mock_identity_base:
            R_W_A = np.eye(3)
            t_W_A = np.zeros(3)
        else:
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
                reason = ""
                if tracker is not None:
                    reason = getattr(tracker, "last_reject_reason", "") or "no_data"
                print(f"[J6] ball lost {since_good:.2f}s (reason={reason}) → home")
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
                f"[J6] {state.name:8s}  tti={tti:+.3f}s  "
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
            q_wu, err_wu, ori_wu = ik_solve(
                chain, t_A_windup, q_seed, offset_link7,
                R_A_link7_target=R_A_link7_home, w_ori=w_ori,
            )

            tick_max_delta_deg = acquire_max_delta_deg if state == State.IDLE else max_delta_deg
            if (_ik_ok(err_wu, ori_wu, ik_tol_m, max_ori_err_deg, R_A_link7_home)
                    and _joints_ok(q_wu)
                    and _delta_ok(q_wu, q_cur, tick_max_delta_deg)):
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
                q_wu, err_wu, ori_wu = ik_solve(
                    chain, t_A_windup, q_cur, offset_link7,
                    R_A_link7_target=R_A_link7_home, w_ori=w_ori,
                )
                if (_ik_ok(err_wu, ori_wu, ik_tol_m, max_ori_err_deg, R_A_link7_home)
                        and _joints_ok(q_wu)
                        and _delta_ok(q_wu, q_cur, acquire_max_delta_deg)):
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
                    f"[J6] COMMIT rejected: strike_W.z={t_W_strike[2]:+.3f} m "
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
                print(f"[J6] COMMIT rejected: clipped target(s) {','.join(clips)}")
                set_vec(r, GOAL_JOINTS, q_home_rad)
                state = State.IDLE
                last_wu_q = None
                last_track_target_A = None
                time.sleep(max(0.0, dt - (time.perf_counter() - t0)))
                continue

            # Solve strike IK seeded from last wind-up solution (smooth)
            q_seed = last_wu_q if last_wu_q is not None else q_cur
            q_strike, err_st, ori_st = ik_solve(
                chain, t_A_strike, q_seed, offset_link7,
                R_A_link7_target=R_A_link7_home, w_ori=w_ori,
            )
            q_follow, err_fw, ori_fw = ik_solve(
                chain, t_A_follow, q_strike, offset_link7,
                R_A_link7_target=R_A_link7_home, w_ori=w_ori,
            )
            commit_delta_deg = float(np.degrees(np.abs(q_strike - q_cur)).max())

            if (not _joints_ok(q_strike)
                    or not _ik_ok(err_st, ori_st, ik_tol_m * 3, max_ori_err_deg, R_A_link7_home)):
                print(f"[J6] COMMIT IK failed (pos={err_st*1000:.1f} mm, ori={ori_st:.1f}°) — aborting")
                time.sleep(max(0.0, dt - (time.perf_counter() - t0)))
                continue
            if commit_delta_deg > commit_max_delta_deg:
                print(
                    f"[J6] COMMIT rejected: q_cur→q_strike max Δ={commit_delta_deg:.1f}° "
                    f"> {commit_max_delta_deg:.1f}°"
                )
                set_vec(r, GOAL_JOINTS, q_home_rad)
                state = State.IDLE
                last_wu_q = None
                last_track_target_A = None
                time.sleep(max(0.0, dt - (time.perf_counter() - t0)))
                continue

            print(f"\n{'='*60}")
            print(f"[J6] COMMIT  tti={tti:.3f}s")
            print(f"[J6]   strike_A=[{t_A_strike[0]:+.3f},{t_A_strike[1]:+.3f},{t_A_strike[2]:+.3f}]  "
                  f"err={err_st*1000:.1f} mm  ori={ori_st:.1f}°")
            print(f"[J6]   follow_A=[{t_A_follow[0]:+.3f},{t_A_follow[1]:+.3f},{t_A_follow[2]:+.3f}]  "
                  f"err={err_fw*1000:.1f} mm  ori={ori_fw:.1f}°")
            print(f"[J6]   q_cur  → q_strike max Δ={commit_delta_deg:.1f}°")
            print(f"{'='*60}\n")

            switch_ctrl(r, JOINT_CTRL)

            # Wind-up → strike
            wu_strike_s = _move_s_with_velocity_floor(q_cur, q_strike, swing_s, swing_vel_frac, "wu→strike")
            seg = run_segment(r, "wu→strike", q_cur, q_strike,
                              move_s=wu_strike_s, hold_s=0.10, publish_hz=200.0)

            if seg["goal_err_max_deg"] > 5.0:
                print(
                    f"[J6] *** wu→strike did not track "
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
                print("[J6] *** SAFETY TORQUES after strike — aborting, slow return ***")
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
        description="J6: Reactive IK-based ball intercept (joint controller).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Ball / geometry
    ap.add_argument("--ball-rigid-body-id", type=int, default=1,
                    help="OptiTrack streaming ID for the pickleball.")
    ap.add_argument("--mock-intercept", nargs=3, type=float, action="append",
                    metavar=("X", "Y", "Z"),
                    help="World-frame mock strike point (m). Repeat for multiple "
                         "waypoints. Skips ball tracker.")
    ap.add_argument("--mock-tti", type=float, default=0.50,
                    help="Fixed time-to-impact for --mock-intercept (s). "
                         "Use > commit_tti for tracking-only; smaller triggers swing.")
    ap.add_argument("--mock-cycle-s", type=float, default=0.0,
                    help="Cycle through mock intercepts every N seconds (0 = first only).")
    ap.add_argument("--mock-identity-base", action="store_true",
                    help="Skip cart OptiTrack; treat arm base frame as world "
                         "(origin-aligned). Use with --mock-intercept when "
                         "streamer is off — coords are then arm-frame targets.")
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
    ap.add_argument("--home-s", type=float, default=3.0,
                    help="Minimum move time for startup →home (s); stretched for large deltas.")
    ap.add_argument("--home-hold-s", type=float, default=1.0,
                    help="Hold time at home after startup move (s).")
    ap.add_argument("--home-vel-frac", type=float, default=J6_HOME_VEL_FRAC,
                    help="Velocity cap fraction for startup →home.")

    # Workspace safety
    ap.add_argument("--reach-m", type=float, default=J6_REACH_M,
                    help="Max horizontal reach for IK targets in arm frame (m).")
    ap.add_argument("--z-min-m", type=float, default=J6_Z_MIN_M,
                    help="Min arm-frame Z for IK targets (m).")
    ap.add_argument("--z-max-m", type=float, default=J6_Z_MAX_M,
                    help="Max arm-frame Z for IK targets (m).")
    ap.add_argument("--commit-max-delta-deg", type=float, default=J6_COMMIT_MAX_DELTA_DEG,
                    help="Refuse blocking swing if any joint is farther than this from strike.")
    ap.add_argument("--world-z-min-m", type=float, default=J6_WORLD_Z_MIN_M,
                    help="Reject commit if predicted world-frame strike Z is below this.")
    ap.add_argument("--world-z-max-m", type=float, default=J6_WORLD_Z_MAX_M,
                    help="Reject commit if predicted world-frame strike Z is above this.")
    ap.add_argument("--target-jump-max-m", type=float, default=J6_TARGET_JUMP_MAX_M,
                    help="Reject tracking updates whose arm-frame target jumps farther than this.")
    ap.add_argument("--acquire-max-delta-deg", type=float, default=J6_ACQUIRE_MAX_DELTA_DEG,
                    help="Max per-joint delta from current pose to enter TRACKING from IDLE.")
    ap.add_argument("--max-delta-deg", type=float, default=J6_MAX_DELTA_DEG,
                    help="Max per-joint IK goal delta from current pose while TRACKING.")
    ap.add_argument("--ik-tol-mm", type=float, default=J6_IK_TOL_M * 1000.0,
                    help="Reject IK solutions whose sweet-spot error exceeds this (mm).")
    ap.add_argument("--tracking-step-deg", type=float, default=J6_TRACKING_STEP_DEG,
                    help="Max per-tick joint-goal step during non-blocking tracking.")
    ap.add_argument("--swing-vel-frac", type=float, default=J6_SWING_VEL_FRAC,
                    help="Max blocking-swing peak velocity as a fraction of XML velocity limits.")
    ap.add_argument("--z-mode", choices=["fixed-arm", "predicted"], default="fixed-arm",
                    help="Use fixed arm-frame Z for J6 bringup, or raw predicted Z.")
    ap.add_argument("--fixed-arm-z-m", type=float, default=J6_FIXED_ARM_Z_M,
                    help="Arm-frame strike Z used when --z-mode=fixed-arm.")
    ap.add_argument("--no-fixed-paddle-ori", action="store_true",
                    help="Position-only IK (old behavior). Default locks link7 "
                         "orientation to home so the paddle face stays toward the net.")
    ap.add_argument("--w-ori", type=float, default=J6_W_ORI,
                    help="IK weight on squared link7 rotation error from home (rad²).")
    ap.add_argument("--max-ori-err-deg", type=float, default=J6_MAX_ORI_ERR_DEG,
                    help="Reject IK solutions whose link7 orientation deviates "
                         "farther than this from home.")

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
        sys.exit(f"[J6] cannot reach Redis at {args.redis_host}:{args.redis_port}: {e}")

    # ---- Calibration file ----
    cal_path = args.calibration or arm_base_offset_calibration_path()
    if not os.path.isfile(cal_path):
        sys.exit(
            f"[J6] arm_base_offset_calibration.json not found: {cal_path}\n"
            f"      Run: python sports_bot/scripts/calibrate_arm_base_offset.py"
        )
    cal = load_arm_base_offset_calibration(cal_path)
    print(f"[J6] arm base calibration: {cal_path}  (rigid body {cal.base_rigid_body_id})")

    mock_intercepts: list[np.ndarray] | None = None
    if args.mock_intercept:
        mock_intercepts = [np.array(triple, dtype=float) for triple in args.mock_intercept]

    # Verify cart is visible before doing anything (unless mock + identity base)
    if not (mock_intercepts and args.mock_identity_base):
        if read_rigid_body_pose_W(r, cal.base_rigid_body_id) is None:
            sys.exit(f"[J6] cart rigid body {cal.base_rigid_body_id} not visible in Redis. "
                     f"Check OptiTrack streamer, or use --mock-identity-base.")

    # ---- Build ikpy chain ----
    if not os.path.isfile(URDF_PATH):
        sys.exit(f"[J6] URDF not found: {URDF_PATH}  (run from OpenSai root)")
    chain = build_chain()

    # ---- Calibrate offset_link7 at current pose ----
    q_at_cal, offset_link7 = calibrate_offset_link7(r, chain)

    if args.print_cal_only:
        print("[J6] --print-cal-only: done.")
        return

    # ---- Move to home before tracking ----
    print("[J6] moving to home pose ...")
    q_home_rad = move_to_home_pose(
        r,
        Q_HOME_RAD,
        move_s=args.home_s,
        hold_s=args.home_hold_s,
        vel_frac=args.home_vel_frac,
    )

    R_A_link7_home = None
    if not args.no_fixed_paddle_ori:
        # Paddle face / net alignment comes from the designed home pose, not wherever
        # the arm happened to stop if the move was only partial.
        R_A_link7_home = _link7_R_A(chain, Q_HOME_RAD)
        face_A = R_A_link7_home @ (offset_link7 / max(np.linalg.norm(offset_link7), 1e-9))
        print("[J6] home link7 orientation from nominal Q_HOME")
        print(f"[J6]   strike-face normal_A ≈ [{face_A[0]:+.3f},{face_A[1]:+.3f},{face_A[2]:+.3f}]")

    # ---- Ball tracker (skipped in mock mode) ----
    tracker: BallTracker | None = None
    if mock_intercepts is None:
        cfg = BallTrackerConfig()
        if args.min_lookahead is not None:
            cfg.min_lookahead = args.min_lookahead
        if args.max_lookahead is not None:
            cfg.max_lookahead = args.max_lookahead

        keys = RedisKeys(ball_source="optitrack")
        keys.ball.__dict__["optitrack_rigid_body_id"] = args.ball_rigid_body_id

        ball_key = keys.ball.optitrack_position
        if r.get(ball_key) is None:
            print(f"[J6] WARNING: {ball_key} is empty — is the OptiTrack streamer running?")
        else:
            print(f"[J6] reading ball from {ball_key}")

        tracker = BallTracker(r, keys, cfg)

    # ---- Go ----
    try:
        run_loop(
            r=r,
            chain=chain,
            tracker=tracker,
            q_home_rad=q_home_rad,
            offset_link7=offset_link7,
            R_A_link7_home=R_A_link7_home,
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
            acquire_max_delta_deg=args.acquire_max_delta_deg,
            max_delta_deg=args.max_delta_deg,
            ik_tol_m=args.ik_tol_mm / 1000.0,
            z_mode=args.z_mode,
            fixed_arm_z_m=args.fixed_arm_z_m,
            w_ori=args.w_ori,
            max_ori_err_deg=args.max_ori_err_deg,
            mock_intercepts=mock_intercepts,
            mock_tti=args.mock_tti,
            mock_cycle_s=args.mock_cycle_s,
            mock_identity_base=args.mock_identity_base,
        )
    except KeyboardInterrupt:
        print("\n[J6] stopped — leaving last joint goal in Redis.")


if __name__ == "__main__":
    main()
