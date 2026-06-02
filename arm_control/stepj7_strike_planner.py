#!/usr/bin/env python3
"""
Step J7: Strike planner for live/replayed pickleball hits (IK joint controller).

Fork of ``stepj6_reactive_intercept.py`` focused on actual hitting: dynamic
commit timing, upward follow-through, and optional open paddle face.

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

IK sweet-spot formulationgit 
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
  OpenSai running with joint_controller in the XML (cartesian_controller only
    needed for live offset cal — use --skip-cal if that switch kills OpenSai)

Run from OpenSai root:
  python sports_bot/arm_control/stepj6_reactive_intercept.py --skip-cal --no-commit
  python sports_bot/arm_control/stepj6_reactive_intercept.py --skip-cal --gentle-swing
  python sports_bot/arm_control/stepj6_reactive_intercept.py --skip-cal --gentle-swing --mini-swing
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
import contextlib
import io
import json
import math
import os
import sys
import time
import warnings
from enum import Enum, auto


class _Tee:
    """Write to both the original stdout and a log file simultaneously."""
    def __init__(self, log_path: str) -> None:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self._file = open(log_path, "a", buffering=1)
        self._stdout = sys.stdout
    def write(self, data: str) -> int:
        self._stdout.write(data)
        self._file.write(data)
        return len(data)
    def flush(self) -> None:
        self._stdout.flush()
        self._file.flush()
    def fileno(self) -> int:
        return self._stdout.fileno()
    def close(self) -> None:
        self._file.close()

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
    COMMAND_TORQUES,
    GOAL_JOINTS,
    SAFETY_TORQUES,
    SENSED_TORQUES,
    SENT_TORQUES,
    SENSOR_JOINTS,
    SENSOR_JOINT_VELS,
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
# J7 conservative workspace  (sweet-spot position in arm base frame)
# ---------------------------------------------------------------------------
# Horizontal reach: Franka HW is ~85 cm; 60 cm gives ~25 cm of margin.
# z_min = 0.20 m: sweet spot stays ≥ 0.64 m world-Z, so paddle tip (10 cm
#   past sweet spot) is ≥ 0.54 m — well clear of the floor in any EE pose.
# z_max = 0.80 m: well within comfortable range, avoids arm fully extended up.
# Home / ready pose — free-driven to a forward-reach position, paddle face
# toward opponent, elbow up, comfortable for mid-height volleys (2026-05-28).
# Old reliable home — arm extends forward, paddle at mid-height.
# Higher than the 2026-05-31 centroid home; better coverage for tonight's
# typical throws (z_arm 0.40–0.60m) without the rightward bias.
# Centroid home (kept for reference):
#   [+2.1°, -55.4°, +15.8°, -134.6°, +11.9°, +86.6°, -28.4°]
#   (too low: z_arm≈0.35m, biased right, misses most of tonight's balls)
Q_HOME_RAD = np.array([
    -0.06108652, -0.68940505, -0.00872665, -2.06821516, 0.00698132, 1.48702052, -0.79412481,
])
# Degrees: q1=-3.5  q2=-39.5  q3=-0.5  q4=-118.5  q5=+0.4  q6=+85.2  q7=-45.5

J6_REACH_M       = 0.78   # Franka HW ~0.85 m; 0.78 keeps ~7 cm margin vs 0.60 bring-up default
J6_Z_MIN_M       = 0.15
J6_Z_MAX_M       = 0.85
J6_IK_TOL_M      = 0.012  # 12 mm — relaxed from 8 mm so clipped/near-limit targets track
J6_MAX_DELTA_DEG = 10.0   # per-tick guard against noisy IK target jumps
J6_ACQUIRE_MAX_DELTA_DEG = 60.0  # first reachable strike-plane target; command is still rate-limited
J6_COMMIT_MAX_DELTA_DEG = 22.0  # tonight-hit mode: commit earlier, still bounded by velocity floor
J6_WORLD_Z_MIN_M = 0.05       # reject obvious tracker/world-frame outliers
J6_WORLD_Z_MAX_M = 0.95       # high throws pull the arm toward poor conditioning; skip them tonight
J6_TARGET_JUMP_MAX_M = 0.08   # max tracking-target jump accepted per tick
J6_TRACKING_STEP_DEG = 0.25    # max commanded joint-goal step per 100 Hz tick
J6_SWING_VEL_FRAC = 0.45      # cap blocking swing peak qdot to this fraction of XML limits
J6_PRE_SWING_SETTLE_S = 0.05   # was 0.20s; that made Step 5 arrive late
J6_COMMIT_MARGIN_S = 0.12      # default commit_tti = swing + settle + margin
J6_STRIKE_A_X_MIN_M = 0.14     # arm-frame strike box, raw sweet-spot target before clipping
J6_STRIKE_A_X_MAX_M = 0.44
J6_STRIKE_A_Y_ABS_MAX_M = 0.34
J6_STRIKE_A_Z_MIN_M = 0.14
J6_STRIKE_A_Z_MAX_M = 0.50
# --gentle-swing preset (slow poke, keeps orientation; pair with --mini-swing)
J6_GENTLE_SWING_S = 0.55
J6_GENTLE_SWING_VEL_FRAC = 0.25
J6_GENTLE_FOLLOW_OFFSET_M = 0.05
J6_GENTLE_COMMIT_TTI_S = 0.50
J6_GENTLE_RETURN_S = 2.5
J6_HOME_VEL_FRAC = 0.85       # startup / return-home velocity cap fraction
J6_HOME_SETTLE_TOL_DEG = 5.0  # accept settled joints as home if within this of nominal
J6_FIXED_ARM_Z_M = 0.45       # J7 bringup: trust lateral prediction, hold Z steady
J6_W_ORI = 2.0                # IK weight on (link7 rotation error from home)²
J6_MAX_ORI_ERR_DEG = 8.0      # reject IK if link7 rotates farther than this from home
J6_LOCK_RELEASE_AFTER_IMPACT_S = 0.6

# joint_controller XML velocity limits currently used by picklebot_j5.xml / picklebot.xml
J6_XML_VEL_LIMIT_RAD_S = np.array([1.2, 1.4, 1.6, 1.8, 1.0, 1.1, 1.2])

# picklebot.xml compliantFrame z=0.35 m + URDF flange → ~0.43 m in ikpy link7 +Z.
# Matches live cartesian cal on tidybot01; use --skip-cal when that switch kills OpenSai.
J6_DEFAULT_OFFSET_LINK7 = np.array([0.0, 0.0, 0.43])


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


def _rot_y_rad(angle_rad: float) -> np.ndarray:
    """Arm-frame Y-axis rotation matrix."""
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


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


def _rate_accel_limited_goal(
    q_target: np.ndarray,
    q_base: np.ndarray,
    qdot_cmd_prev: np.ndarray,
    max_speed_deg_s: float,
    max_accel_deg_s2: float,
    dt_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Rate and acceleration limited command update from the previous goal."""
    dt_s = max(float(dt_s), 1e-4)
    max_speed = math.radians(max_speed_deg_s)
    max_accel = math.radians(max_accel_deg_s2)
    delta = q_target - q_base
    qdot_des = np.clip(delta / dt_s, -max_speed, max_speed)
    qdot_cmd = qdot_cmd_prev + np.clip(
        qdot_des - qdot_cmd_prev, -max_accel * dt_s, max_accel * dt_s,
    )
    qdot_cmd = np.clip(qdot_cmd, -max_speed, max_speed)
    step = qdot_cmd * dt_s
    overshoot = np.abs(step) > np.abs(delta)
    if np.any(overshoot):
        step[overshoot] = delta[overshoot]
        qdot_cmd[overshoot] = 0.0
    return q_base + step, qdot_cmd


def _hold_current_joints(r: redis.Redis) -> np.ndarray | None:
    q_cur = get_vec(r, SENSOR_JOINTS, 7)
    if q_cur is not None:
        set_vec(r, GOAL_JOINTS, q_cur)
    return q_cur


def _hold_current_joints_for(
    r: redis.Redis,
    duration_s: float,
    publish_hz: float = 100.0,
) -> np.ndarray | None:
    q_hold = get_vec(r, SENSOR_JOINTS, 7)
    if q_hold is None:
        return None
    set_vec(r, GOAL_JOINTS, q_hold)
    _hold_joint_controller(r)
    period_s = 1.0 / max(1.0, publish_hz)
    t_end = time.perf_counter() + max(0.0, duration_s)
    while time.perf_counter() < t_end:
        set_vec(r, GOAL_JOINTS, q_hold)
        time.sleep(period_s)
    return q_hold


def _soft_home_goal(r: redis.Redis, q_home_rad: np.ndarray, max_step_deg: float) -> None:
    q_cur = get_vec(r, SENSOR_JOINTS, 7)
    if q_cur is None:
        return
    set_vec(r, GOAL_JOINTS, _rate_limited_goal(q_home_rad, q_cur, max_step_deg))


def _home_error_deg(r: redis.Redis, q_home_rad: np.ndarray) -> float:
    q_cur = get_vec(r, SENSOR_JOINTS, 7)
    if q_cur is None:
        return float("inf")
    return float(np.max(np.abs(np.degrees(q_home_rad - q_cur))))


def _return_home_blocking(
    r: redis.Redis,
    q_home_rad: np.ndarray,
    *,
    move_s: float,
    vel_frac: float,
    label: str,
    hold_s: float = 0.35,
    publish_hz: float = 100.0,
    full_segment_logs: bool = True,
    trace_cb=None,
) -> bool:
    """Actively return to home; soft one-tick nudges are not enough between throws."""
    q_start = get_vec(r, SENSOR_JOINTS, 7)
    if q_start is None:
        print(f"[J7] {label}: cannot read joints for home return")
        return False
    err0 = float(np.max(np.abs(np.degrees(q_home_rad - q_start))))
    if err0 <= 2.0:
        set_vec(r, GOAL_JOINTS, q_home_rad)
        _hold_joint_controller(r)
        return True

    set_vec(r, GOAL_JOINTS, q_start)
    _hold_joint_controller(r)
    time.sleep(0.05)
    print(f"[J7] {label}: returning home from {err0:.1f}° max error")
    home_s = _move_s_with_velocity_floor(q_start, q_home_rad, move_s, vel_frac, label)
    if full_segment_logs:
        seg = run_segment(
            r, label, q_start, q_home_rad,
            move_s=home_s, hold_s=hold_s, publish_hz=publish_hz,
        )
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            seg = run_segment(
                r, label, q_start, q_home_rad,
                move_s=home_s, hold_s=hold_s, publish_hz=publish_hz,
            )
        print(
            f"[J7 segment] {label}: move_s={home_s:.3f}s hold_s={hold_s:.2f}s "
            f"goal_err={seg['goal_err_max_deg']:.2f}° "
            f"qdot_max={seg['qdot_max_deg_s']:.1f}°/s"
        )
    err1 = _home_error_deg(r, q_home_rad)
    if err1 > 5.0:
        q_after = get_vec(r, SENSOR_JOINTS, 7)
        if q_after is not None:
            set_vec(r, GOAL_JOINTS, q_after)
        active = r.get(ACTIVE_CONTROLLER)
        tau_cmd = get_vec(r, COMMAND_TORQUES, 7)
        tau_sensed = get_vec(r, SENSED_TORQUES, 7)
        tau_cmd_txt = "n/a" if tau_cmd is None else f"{_max_abs_with_joint(tau_cmd)[0]:.1f}@q{_max_abs_with_joint(tau_cmd)[1]}"
        tau_sensed_txt = "n/a" if tau_sensed is None else f"{_max_abs_with_joint(tau_sensed)[0]:.1f}@q{_max_abs_with_joint(tau_sensed)[1]}"
        print(
            f"[J7] WARNING: {label} failed to reach home "
            f"(err {err1:.1f}°, qdot_max {seg['qdot_max_deg_s']:.1f}°/s, "
            f"active={active!r}, tau_cmd={tau_cmd_txt}, tau_sensed={tau_sensed_txt}). "
            "If qdot was near zero, OpenSai/Franka control is likely not accepting torques; relaunch driver/OpenSai."
        )
        return False

    set_vec(r, GOAL_JOINTS, q_home_rad)
    print(f"[J7] {label}: home reached (err {err1:.2f}°)")
    return True


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
            f"[J7] {label}: stretching move_s {requested_s:.2f}s → {move_s:.2f}s "
            f"for q{i+1} velocity cap ({peak_deg_s:.1f}/{cap_deg_s:.1f} deg/s)"
        )
    return move_s


def _move_s_estimate(
    q_start: np.ndarray,
    q_target: np.ndarray,
    requested_s: float,
    vel_frac: float,
) -> float:
    """Non-printing sibling of _move_s_with_velocity_floor for commit timing."""
    vel_cap = np.maximum(J6_XML_VEL_LIMIT_RAD_S * max(vel_frac, 1e-3), 1e-6)
    required_s_by_joint = 1.5 * np.abs(q_target - q_start) / vel_cap
    return max(float(requested_s), float(np.max(required_s_by_joint)))


def _fmt_q(q: np.ndarray) -> str:
    return "  ".join(f"q{i+1}={math.degrees(v):+.1f}" for i, v in enumerate(q))


def _fmt_v3(v: np.ndarray) -> str:
    return f"[{v[0]:+.3f},{v[1]:+.3f},{v[2]:+.3f}]"


def _fmt_deg_vec(q: np.ndarray) -> str:
    return "[" + ",".join(f"{math.degrees(v):+.1f}" for v in q) + "]"


def _max_abs_deg_with_joint(v_rad: np.ndarray) -> tuple[float, int]:
    vals = np.abs(np.degrees(v_rad))
    i = int(np.argmax(vals))
    return float(vals[i]), i + 1


def _max_abs_with_joint(v: np.ndarray) -> tuple[float, int]:
    vals = np.abs(v)
    i = int(np.argmax(vals))
    return float(vals[i]), i + 1


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


def _hold_joint_controller(r: redis.Redis) -> None:
    """Re-assert joint_controller if OpenSai reverted to cartesian (goals then ignored)."""
    got = r.get(ACTIVE_CONTROLLER)
    if isinstance(got, bytes):
        got = got.decode()
    if got != JOINT_CTRL:
        switch_ctrl(r, JOINT_CTRL)


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
    print("[J7 cal] switching to cartesian_controller (read-only) ...")
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

    print(f"[J7 cal]   q_at_cal:     {_fmt_q(q_at_cal)}")
    print(f"[J7 cal]   sweet_spot_A: [{t_sweet_A[0]:+.4f}, {t_sweet_A[1]:+.4f}, {t_sweet_A[2]:+.4f}] m")
    print(f"[J7 cal]   link7_tip_A:  [{T_at_cal[0,3]:+.4f}, {T_at_cal[1,3]:+.4f}, {T_at_cal[2,3]:+.4f}] m")
    print(f"[J7 cal]   offset_link7: [{offset_link7[0]:+.4f}, {offset_link7[1]:+.4f}, {offset_link7[2]:+.4f}] m  |{mag_m*100:.1f} cm|")

    # Expect ~35-50 cm (sweet spot + any URDF flange offset)
    if not (0.25 < mag_m < 0.60):
        print(
            f"  WARNING: offset magnitude {mag_m*100:.1f} cm is outside expected "
            f"25–60 cm — check URDF or mount", file=sys.stderr,
        )

    # Switch to joint_controller, seed goal = current joints (no lurch)
    print("[J7 cal] switching to joint_controller ...")
    set_vec(r, GOAL_JOINTS, q_at_cal)
    if not ensure_joint_controller(r, timeout_s=2.0):
        sys.exit(
            f"ERROR: could not activate {JOINT_CTRL} after calibration — "
            "is OpenSai running with joint_controller in the XML?"
        )
    print("[J7 cal] calibration done.\n")
    return q_at_cal, offset_link7


def seed_offset_link7_no_cartesian(
    r: redis.Redis,
    offset_link7: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Use a fixed link7 sweet-spot offset — no cartesian_controller switch.

    Switching active_controller to cartesian_controller for live cal can fault
    or kill OpenSai on some bring-up setups.  This path stays on joint_controller.
    """
    q_cur = get_vec(r, SENSOR_JOINTS, 7)
    if q_cur is None:
        sys.exit(f"ERROR: {SENSOR_JOINTS} not available — is OpenSai running?")

    mag_m = float(np.linalg.norm(offset_link7))
    print("[J7 cal] --skip-cal: using fixed offset_link7 (no cartesian switch)")
    print(f"[J7 cal]   q_cur:        {_fmt_q(q_cur)}")
    print(f"[J7 cal]   offset_link7: [{offset_link7[0]:+.4f}, {offset_link7[1]:+.4f}, "
          f"{offset_link7[2]:+.4f}] m  |{mag_m*100:.1f} cm|")

    set_vec(r, GOAL_JOINTS, q_cur)
    if not ensure_joint_controller(r, timeout_s=2.0):
        sys.exit(
            f"ERROR: could not activate {JOINT_CTRL} — "
            "is OpenSai running with joint_controller in the XML?"
        )
    print("[J7 cal] joint_controller seeded at current joints.\n")
    return q_cur, offset_link7.copy()


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
        sys.exit(f"[J7] cannot read {SENSOR_JOINTS} for home move")

    max_delta_deg = float(np.max(np.abs(np.degrees(q_nominal_home - q_cur))))
    print(f"[J7] home move: max joint delta {max_delta_deg:.1f}° from nominal")
    if max_delta_deg < 2.0:
        print("[J7] already near home — holding current joints")
        set_vec(r, GOAL_JOINTS, q_cur)
        ensure_joint_controller(r, timeout_s=2.0)
        return q_cur.copy()

    set_vec(r, GOAL_JOINTS, q_cur)
    if not ensure_joint_controller(r, timeout_s=2.0):
        sys.exit(
            f"[J7] {JOINT_CTRL} not active — cannot move to home. "
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
            f"[J7] WARNING: home move incomplete — max joint error {err_deg:.1f}° "
            f"(need ≤ {J6_HOME_SETTLE_TOL_DEG:.1f}°). "
            "Holding settled joints; tracking may be degraded.",
            file=sys.stderr,
        )
        set_vec(r, GOAL_JOINTS, q_settled)
        return q_settled.copy()

    set_vec(r, GOAL_JOINTS, q_settled)
    print(f"[J7] at home (max joint error {err_deg:.2f}° from nominal).\n")
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


def _strike_box_reject_reason(
    t_A_strike_raw: np.ndarray,
    *,
    x_min_m: float,
    x_max_m: float,
    y_abs_max_m: float,
    z_min_m: float,
    z_max_m: float,
) -> str | None:
    """Reject strike targets before workspace clipping hides an awkward pose."""
    x, y, z = map(float, t_A_strike_raw)
    if not (x_min_m <= x <= x_max_m):
        return f"strike_A.x={x:+.3f} outside [{x_min_m:.2f},{x_max_m:.2f}]"
    if abs(y) > y_abs_max_m:
        return f"|strike_A.y|={abs(y):.3f} > {y_abs_max_m:.2f}"
    if not (z_min_m <= z <= z_max_m):
        return f"strike_A.z={z:+.3f} outside [{z_min_m:.2f},{z_max_m:.2f}]"
    return None


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
    follow_up_offset_m: float,
    swing_s: float,
    contact_margin_s: float,
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
    tracking_accel_deg_s2: float,
    swing_vel_frac: float,
    acquire_max_delta_deg: float,
    max_delta_deg: float,
    ik_tol_m: float,
    z_mode: str,
    fixed_arm_z_m: float,
    w_ori: float,
    max_ori_err_deg: float,
    pre_swing_settle_s: float,
    strike_arm_x_min_m: float,
    strike_arm_x_max_m: float,
    strike_arm_y_abs_max_m: float,
    strike_arm_z_min_m: float,
    strike_arm_z_max_m: float,
    mock_intercepts: list[np.ndarray] | None = None,
    mock_tti: float = 0.50,
    mock_cycle_s: float = 0.0,
    mock_identity_base: bool = False,
    direct_tracking: bool = False,
    verbose_tracking: bool = False,
    full_segment_logs: bool = False,
    mini_swing: bool = False,
    log_period_s: float = 0.50,
    debug_joints: bool = False,
    quiet_after_settle_s: float = 0.0,
    quiet_settle_deadband_deg: float = 1.0,
    lock_first_safe_prediction: bool = True,
    lock_tti_max: float = 0.0,
    lock_release_after_impact_s: float = J6_LOCK_RELEASE_AFTER_IMPACT_S,
    post_impact_idle_s: float = 1.5,
    wu_hold_s: float = 0.10,
    max_swings: int = 0,
    flick_q6_deg: float = 0.0,
    flick_s: float = 0.25,
    flick_vel_frac: float = 1.0,
    flick_wu_delta_deg: float = 8.0,
) -> None:
    dt = 1.0 / rate_hz
    state = State.IDLE
    idle_until_t: float = 0.0  # no new locks accepted before this time
    last_good_t = float("-inf")
    hold_after_lost_s = 0.5
    last_print_t = float("-inf")
    print_interval_s = max(0.05, float(log_period_s))
    last_wu_q: np.ndarray | None = None  # last valid wind-up IK solution
    last_track_target_A: np.ndarray | None = None
    last_q_cmd: np.ndarray | None = None
    last_q_cmd_vel = np.zeros(7)
    last_goal_write_t = time.perf_counter()
    last_effective_step_deg = tracking_step_deg
    locked_wu_q: np.ndarray | None = None
    locked_ik_err_mm = 0.0
    locked_ori_err_deg = 0.0
    mock_t0 = time.monotonic()
    using_mock = mock_intercepts is not None and len(mock_intercepts) > 0
    locked_intercept: Intercept | None = None
    locked_at_s = 0.0
    locked_throw_id = 0
    lock_goal_writes = 0
    lock_ik_failures = 0
    lock_rejections = 0
    pending_lock_rejections = 0
    lock_peak_goal_err_deg = 0.0
    last_lock_reject_print_t = float("-inf")
    settled_since_t: float | None = None
    quiet_hold_active = False
    quiet_hold_printed = False
    quiet_hold_goal: np.ndarray | None = None
    home_retry_after_t = 0.0

    audit_lock_strike_W: np.ndarray | None = None
    audit_lock_wu_A: np.ndarray | None = None
    audit_lock_tti_s: float | None = None
    audit_lock_target_delta_deg: float | None = None
    audit_tracker_rejects: dict[str, int] | None = None
    tracker_reject_counts: dict[str, int] = {}
    audit_commit_delta_deg: float | None = None
    audit_timing_margin_s: float | None = None
    audit_wu_goal_err_deg: float | None = None
    audit_follow_goal_err_deg: float | None = None
    audit_home_ok: bool | None = None

    def finish_lock(reason: str, status: str | None = None) -> None:
        nonlocal locked_intercept, locked_at_s, locked_wu_q, lock_goal_writes, lock_ik_failures
        nonlocal lock_rejections, lock_peak_goal_err_deg, last_q_cmd_vel
        nonlocal settled_since_t, quiet_hold_active, quiet_hold_printed, quiet_hold_goal
        nonlocal audit_lock_strike_W, audit_lock_wu_A, audit_lock_tti_s, audit_lock_target_delta_deg
        nonlocal audit_tracker_rejects, tracker_reject_counts
        nonlocal audit_commit_delta_deg, audit_timing_margin_s, audit_wu_goal_err_deg
        nonlocal audit_follow_goal_err_deg, audit_home_ok
        if locked_intercept is None:
            return
        if status is None:
            status = "OK" if lock_goal_writes > 0 and lock_ik_failures == 0 else "WARN"
        print(
            f"[J7 lock] throw {locked_throw_id} {status}: {reason}; "
            f"goal_writes={lock_goal_writes}  prelock_rejects={lock_rejections}  "
            f"ik_failures={lock_ik_failures}  peak_goal_err={lock_peak_goal_err_deg:.2f}°"
        )
        if "COMMIT rejected" in reason or "no locked throw" in reason:
            outcome = "TRACKER_OR_GATE"
        elif "ball passed" in reason or (audit_timing_margin_s is not None and audit_timing_margin_s < -0.08):
            outcome = "TIMING_HEADROOM"
        elif "did not track" in reason:
            outcome = "ARM_EXECUTION"
        elif "safety torques" in reason:
            outcome = "CONTACT_REFLEX"
        elif status == "OK":
            outcome = "OK"
        else:
            outcome = "NEEDS_REVIEW"

        def _num(v: float | None, fmt: str) -> str:
            return "n/a" if v is None else format(v, fmt)

        def _counts_text(counts: dict[str, int] | None) -> str:
            if not counts:
                return "none"
            return ",".join(
                f"{k}:{v}" for k, v in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:4]
            )

        lock_w = "n/a" if audit_lock_strike_W is None else _fmt_v3(audit_lock_strike_W)
        lock_a = "n/a" if audit_lock_wu_A is None else _fmt_v3(audit_lock_wu_A)
        print(
            f"[J7 audit] throw={locked_throw_id} outcome={outcome} "
            f"tracker[strike_W={lock_w} wu_A={lock_a} "
            f"lock_tti={_num(audit_lock_tti_s, '.3f')}s "
            f"target_delta={_num(audit_lock_target_delta_deg, '.1f')}° "
            f"pre_rejects={lock_rejections} ik_fail={lock_ik_failures} "
            f"tracker_none={_counts_text(audit_tracker_rejects)}] "
            f"arm[commit_delta={_num(audit_commit_delta_deg, '.1f')}° "
            f"margin={_num(audit_timing_margin_s, '+.3f')}s "
            f"wu_err={_num(audit_wu_goal_err_deg, '.2f')}° "
            f"follow_err={_num(audit_follow_goal_err_deg, '.2f')}° "
            f"home_ok={audit_home_ok}]"
        )
        locked_intercept = None
        locked_wu_q = None
        locked_at_s = 0.0
        lock_goal_writes = 0
        lock_ik_failures = 0
        lock_rejections = 0
        lock_peak_goal_err_deg = 0.0
        settled_since_t = None
        quiet_hold_active = False
        quiet_hold_printed = False
        quiet_hold_goal = None
        audit_lock_strike_W = None
        audit_lock_wu_A = None
        audit_lock_tti_s = None
        audit_lock_target_delta_deg = None
        audit_tracker_rejects = None
        tracker_reject_counts = {}
        audit_commit_delta_deg = None
        audit_timing_margin_s = None
        audit_wu_goal_err_deg = None
        audit_follow_goal_err_deg = None
        audit_home_ok = None
        last_q_cmd_vel = np.zeros(7)

    def print_lock_wait(reason: str) -> None:
        nonlocal last_lock_reject_print_t, pending_lock_rejections
        pending_lock_rejections += 1
        now = time.perf_counter()
        if now - last_lock_reject_print_t >= 0.50:
            last_lock_reject_print_t = now
            print(f"[J7 lock] waiting for safe prediction: {reason}")

    def segment(label: str, q_start: np.ndarray, q_target: np.ndarray, *,
                move_s: float, hold_s: float, publish_hz: float) -> dict:
        if full_segment_logs:
            return run_segment(
                r, label, q_start, q_target,
                move_s=move_s, hold_s=hold_s, publish_hz=publish_hz,
            )
        with contextlib.redirect_stdout(io.StringIO()):
            seg = run_segment(
                r, label, q_start, q_target,
                move_s=move_s, hold_s=hold_s, publish_hz=publish_hz,
            )
        status_txt = "OK" if seg["goal_err_max_deg"] <= 5.0 else "WARN"
        print(
            f"[J7 segment] {label}: {status_txt} "
            f"move_s={move_s:.3f}s hold_s={hold_s:.2f}s "
            f"goal_err={seg['goal_err_max_deg']:.2f}° "
            f"qdot_max={seg['qdot_max_deg_s']:.1f}°/s"
        )
        return seg

    def recover_home_blocking(label: str, *, idle_after: bool = True) -> bool:
        nonlocal state, last_wu_q, locked_wu_q, last_track_target_A
        nonlocal last_q_cmd, last_q_cmd_vel, last_goal_write_t, idle_until_t, home_retry_after_t
        ok = _return_home_blocking(
            r, q_home_rad,
            move_s=return_s, vel_frac=max(swing_vel_frac, 0.35),
            label=label, hold_s=0.35, publish_hz=100.0,
            full_segment_logs=full_segment_logs,
        )
        state = State.IDLE
        last_wu_q = None
        locked_wu_q = None
        last_track_target_A = None
        last_q_cmd = get_vec(r, SENSOR_JOINTS, 7)
        last_q_cmd_vel = np.zeros(7)
        last_goal_write_t = time.perf_counter()
        if ok:
            home_retry_after_t = 0.0
        else:
            home_retry_after_t = time.monotonic() + 5.0
        if idle_after:
            idle_until_t = time.monotonic() + post_impact_idle_s
        return ok

    switch_ctrl(r, JOINT_CTRL)
    _hold_current_joints(r)

    print(f"[J7] running at {rate_hz:.0f} Hz  —  Ctrl-C to stop")
    flick_info = f"  flick_q6={flick_q6_deg:+.1f}° flick_s={flick_s:.2f}s" if flick_q6_deg > 0 else ""
    print(f"[J7] strike_plane_x={strike_plane_x_world:+.3f} m  "
          f"commit_tti_max={commit_tti:.3f} s  swing_s={swing_s:.2f} s  "
          f"contact_margin={contact_margin_s:+.3f} s  pre_settle={pre_swing_settle_s:.2f} s  "
          f"wu_hold={wu_hold_s:.2f} s  max_swings={'∞' if max_swings == 0 else max_swings}"
          f"{flick_info}")
    print(f"[J7] swing vector: windup={wind_up_offset_m*100:.1f} cm  "
          f"follow=+{follow_offset_m*100:.1f} cm forward, +{follow_up_offset_m*100:.1f} cm up")
    print(f"[J7] workspace (arm frame):  r_xy ≤ {reach_m:.2f} m  "
          f"z ∈ [{z_min:+.2f}, {z_max:+.2f}] m")
    print(f"[J7] strike box (arm frame, unclipped): "
          f"x=[{strike_arm_x_min_m:+.2f},{strike_arm_x_max_m:+.2f}] m  "
          f"|y|≤{strike_arm_y_abs_max_m:.2f} m  "
          f"z=[{strike_arm_z_min_m:+.2f},{strike_arm_z_max_m:+.2f}] m")
    print(f"[J7] commit guards: max Δ={commit_max_delta_deg:.1f}°  "
          f"world_z ∈ [{world_z_min_m:.2f}, {world_z_max_m:.2f}] m  no clipped commits")
    tracking_goal_speed_deg_s = tracking_step_deg * rate_hz
    xml_vel_deg_s = np.degrees(J6_XML_VEL_LIMIT_RAD_S)
    xml_min_deg_s = float(np.min(xml_vel_deg_s))
    xml_max_deg_s = float(np.max(xml_vel_deg_s))
    print(f"[J7] tracking guards: acquire Δ≤{acquire_max_delta_deg:.1f}°  "
          f"track Δ≤{max_delta_deg:.1f}° IK-jump  ik_tol={ik_tol_m*1000:.0f} mm  "
          f"target jump≤{target_jump_max_m*100:.0f} cm  "
          f"goal step≤{tracking_step_deg:.2f}°/tick≈{tracking_goal_speed_deg_s:.0f}°/s  "
          f"XML vel=[{xml_min_deg_s:.0f},{xml_max_deg_s:.0f}]°/s  "
          f"swing vel≤{swing_vel_frac*100:.0f}% XML  "
          f"z_mode={z_mode}" + (f"({fixed_arm_z_m:.2f}m A)" if z_mode == "fixed-arm" else ""))
    if no_commit:
        print("[J7] --no-commit: tracking only, swing disabled")
    elif mini_swing:
        print("[J7] --mini-swing: wu→strike→home (no follow-through segment)")
    if R_A_link7_home is not None:
        face_A = R_A_link7_home @ (offset_link7 / max(np.linalg.norm(offset_link7), 1e-9))
        print("[J7] fixed paddle orientation: link7 rotation locked to home")
        print(f"[J7]   strike-face normal_A ≈ [{face_A[0]:+.3f},{face_A[1]:+.3f},{face_A[2]:+.3f}]  "
              f"w_ori={w_ori:.2f}  max_ori_err={max_ori_err_deg:.1f}°")
    else:
        print("[J7] paddle orientation: position-only IK (--no-fixed-paddle-ori)")
    if using_mock:
        pts = ", ".join(
            f"[{p[0]:+.2f},{p[1]:+.2f},{p[2]:+.2f}]" for p in mock_intercepts
        )
        print(f"[J7] MOCK intercept(s) world/arm if identity (m): {pts}")
        print(f"[J7]   mock_tti={mock_tti:.3f}s  cycle={mock_cycle_s:.1f}s  "
              f"identity_base={mock_identity_base}")
        if not mock_identity_base:
            print("[J7]   mock needs cart rigid body in Redis (or pass --mock-identity-base)")
    if lock_first_safe_prediction:
        print(
            "[J7 lock] enabled: first safe prediction per throw is frozen; "
            "later tracker jitter is ignored"
        )
    else:
        print("[J7 lock] disabled: tracking will chase each new prediction")
    print("[J7] tracking limiter is time-based; slow Python/IK loops still command requested deg/s")
    if tracking_accel_deg_s2 > 0.0:
        print(f"[J7] tracking acceleration cap: {tracking_accel_deg_s2:.0f}°/s² command ramp")
    if direct_tracking:
        print("[J7] --direct-tracking: writing full IK goal each tick (no rate limit)")
    print()

    while True:
        t0 = time.perf_counter()
        _hold_joint_controller(r)

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
            if intercept is None and locked_intercept is None:
                reason = getattr(tracker, "last_reject_reason", "") or "no_prediction"
                tracker_reject_counts[reason] = tracker_reject_counts.get(reason, 0) + 1

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
        # 3. Prediction lock / ball lost handling
        # ----------------------------------------------------------------
        if locked_intercept is not None:
            lock_age_s = time.monotonic() - locked_at_s
            locked_tti = float(locked_intercept.time_to_impact - lock_age_s)
            keep_static_mock_lock = using_mock and no_commit and mock_cycle_s <= 0.0
            if locked_tti < -lock_release_after_impact_s and not keep_static_mock_lock:
                home_ok = recover_home_blocking("impact→home", idle_after=True)
                finish_lock(f"impact window elapsed; home_ok={home_ok}")
                time.sleep(max(0.0, dt - (time.perf_counter() - t0)))
                continue
            intercept = Intercept(
                position=locked_intercept.position.copy(),
                velocity=locked_intercept.velocity.copy(),
                time_to_impact=locked_tti,
                n_bounces=locked_intercept.n_bounces,
                position_cov=locked_intercept.position_cov,
            )
        elif intercept is None:
            since_good = t0 - last_good_t
            if state == State.TRACKING and since_good > hold_after_lost_s:
                reason = ""
                if tracker is not None:
                    reason = getattr(tracker, "last_reject_reason", "") or "no_data"
                print(f"[J7] ball lost {since_good:.2f}s (reason={reason}) → home")
                recover_home_blocking("lost-ball→home", idle_after=False)
            elif state == State.IDLE:
                if _home_error_deg(r, q_home_rad) > 4.0:
                    if time.monotonic() >= home_retry_after_t:
                        recover_home_blocking("idle→home", idle_after=False)
                    else:
                        _hold_current_joints(r)
                else:
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
        strike_box_reason = _strike_box_reject_reason(
            t_A_strike_raw,
            x_min_m=strike_arm_x_min_m,
            x_max_m=strike_arm_x_max_m,
            y_abs_max_m=strike_arm_y_abs_max_m,
            z_min_m=strike_arm_z_min_m,
            z_max_m=strike_arm_z_max_m,
        )

        t_A_windup, wu_clipped  = _safe_target(t_A_windup_raw,  reach_m, z_min, z_max)
        t_A_strike, st_clipped  = _safe_target(t_A_strike_raw,  reach_m, z_min, z_max)
        t_A_follow, fw_clipped  = _safe_target(
            t_A_strike + np.array([follow_offset_m, 0.0, follow_up_offset_m]), reach_m, z_min, z_max)

        q_cur = get_vec(r, SENSOR_JOINTS, 7)
        if q_cur is None:
            time.sleep(max(0.0, dt - (time.perf_counter() - t0)))
            continue

        # pre_positioning: prediction is geometrically valid but TTI is still too large
        # to trust the extrapolation. Arm tracks toward the predicted wu to pre-position
        # (closes most of the joint gap early), but does NOT lock — lock fires once
        # TTI drops to lock_tti_max and the LS fit has converged on a stable intercept.
        _pre_positioning = False
        if lock_first_safe_prediction and locked_intercept is None:
            lock_reject_reason = None
            if not (world_z_min_m <= t_W_strike[2] <= world_z_max_m):
                lock_reject_reason = (
                    f"strike_W.z={t_W_strike[2]:+.3f} outside "
                    f"[{world_z_min_m:.2f},{world_z_max_m:.2f}]"
                )
            elif strike_box_reason is not None:
                lock_reject_reason = strike_box_reason
            elif wu_clipped or st_clipped:
                clips = []
                if wu_clipped:
                    clips.append("windup")
                if st_clipped:
                    clips.append("strike")
                lock_reject_reason = "clipped target(s) " + ",".join(clips)
            if lock_reject_reason is not None:
                print_lock_wait(lock_reject_reason)
                time.sleep(max(0.0, dt - (time.perf_counter() - t0)))
                continue
            # Prediction passes geometric checks. If TTI is still above lock_tti_max,
            # pre-position toward the predicted wu without committing to a lock.
            if lock_tti_max > 0.0 and tti > lock_tti_max:
                _pre_positioning = True
                print_lock_wait(
                    f"pre-positioning tti={tti:.3f}s > lock_tti_max={lock_tti_max:.3f}s"
                )

        # ----------------------------------------------------------------
        # 6a. Dynamic commit decision — J7 commits when the estimated strike
        #     duration lines up with the current TTI, not at one fixed TTI.
        # ----------------------------------------------------------------
        commit_due = (tti <= commit_tti)
        dynamic_commit_txt = ""
        if (not no_commit and locked_intercept is not None
                and state == State.TRACKING and last_wu_q is not None):
            if flick_q6_deg > 0.0:
                # Flick commit: estimate settle-to-wu + q6 flick time
                settle_est_s = _move_s_estimate(q_cur, last_wu_q, 0.02, swing_vel_frac)
                q_flick_probe = last_wu_q.copy()
                q_flick_probe[5] += math.radians(flick_q6_deg)
                flick_est_s = _move_s_estimate(last_wu_q, q_flick_probe, flick_s, flick_vel_frac)
                est_strike_s = settle_est_s + flick_est_s
                dynamic_threshold_s = max(0.0, est_strike_s + contact_margin_s)
                commit_due = (tti <= dynamic_threshold_s) or (tti <= 0.0)
                dynamic_commit_txt = (
                    f"  dyn_commit≤{dynamic_threshold_s:.3f}s"
                    f" est_settle={settle_est_s:.3f}s est_flick={flick_est_s:.3f}s"
                )
            else:
                q_probe_seed = last_wu_q
                q_strike_probe, err_probe, ori_probe = ik_solve(
                    chain, t_A_strike, q_probe_seed, offset_link7,
                    R_A_link7_target=R_A_link7_home, w_ori=w_ori,
                )
                if (_joints_ok(q_strike_probe)
                        and _ik_ok(err_probe, ori_probe, ik_tol_m * 3, max_ori_err_deg, R_A_link7_home)):
                    est_strike_s = _move_s_estimate(q_cur, q_strike_probe, swing_s, swing_vel_frac)
                    dynamic_threshold_s = max(0.0, est_strike_s + max(0.0, pre_swing_settle_s) + contact_margin_s)
                    commit_due = (tti <= dynamic_threshold_s) or (tti <= 0.0)
                    dynamic_commit_txt = (
                        f"  dyn_commit≤{dynamic_threshold_s:.3f}s"
                        f" est_swing={est_strike_s:.3f}s"
                    )

        # ----------------------------------------------------------------
        # 6b. TRACKING — solve IK for wind-up, write directly each tick
        # ----------------------------------------------------------------
        # If no prediction has been locked yet, still allow the tracking path
        # to acquire the first safe wind-up even when the TTI is already inside
        # the commit window. Otherwise a late-but-hittable throw gets stuck in
        # "commit due, but no locked throw" and the arm never moves.
        if (not commit_due) or no_commit or (lock_first_safe_prediction and locked_intercept is None):
            # Post-impact idle: hold home instead of chasing the next intercept.
            # The lock gate already blocks new locks during this window; this
            # prevents the arm from pre-positioning toward a wind-up before the
            # idle expires, so it visibly returns home between shots.
            if time.monotonic() < idle_until_t and locked_intercept is None:
                _soft_home_goal(r, q_home_rad, tracking_step_deg)
                time.sleep(max(0.0, dt - (time.perf_counter() - t0)))
                continue
            if (last_track_target_A is not None
                    and np.linalg.norm(t_A_windup - last_track_target_A) > target_jump_max_m):
                jump_m = float(np.linalg.norm(t_A_windup - last_track_target_A))
                if lock_first_safe_prediction and locked_intercept is None:
                    print_lock_wait(
                        f"target jump {jump_m*100:.1f} cm > {target_jump_max_m*100:.1f} cm"
                    )
                elif locked_intercept is not None:
                    lock_ik_failures += 1
                time.sleep(max(0.0, dt - (time.perf_counter() - t0)))
                continue

            if locked_intercept is not None and locked_wu_q is not None:
                q_wu = locked_wu_q.copy()
                err_wu = locked_ik_err_mm / 1000.0
                ori_wu = locked_ori_err_deg
            else:
                q_seed = last_wu_q if last_wu_q is not None else q_cur
                q_wu, err_wu, ori_wu = ik_solve(
                    chain, t_A_windup, q_seed, offset_link7,
                    R_A_link7_target=R_A_link7_home, w_ori=w_ori,
                )

            tick_max_delta_deg = acquire_max_delta_deg if state == State.IDLE else max_delta_deg
            ik_good = _ik_ok(err_wu, ori_wu, ik_tol_m, max_ori_err_deg, R_A_link7_home)
            joints_good = _joints_ok(q_wu)
            if direct_tracking or state == State.IDLE:
                delta_ref = q_cur
                delta_label = "delta"
            else:
                # With rate-limited tracking, the commanded goal only moves by
                # tracking_step_deg. Guard IK branch jumps, not remaining distance.
                delta_ref = last_wu_q if last_wu_q is not None else q_cur
                delta_label = "ik_jump"
            delta_good = _delta_ok(q_wu, delta_ref, tick_max_delta_deg)
            if ik_good and joints_good and delta_good:
                if lock_first_safe_prediction and locked_intercept is None and not _pre_positioning:
                    if time.monotonic() < idle_until_t:
                        print_lock_wait(
                            f"post-impact idle ({idle_until_t - time.monotonic():.1f}s remaining)"
                        )
                        time.sleep(max(0.0, dt - (time.perf_counter() - t0)))
                        continue
                    # Reachability pre-screen: reject locks where the arm physically
                    # can't reach wu and complete the flick before the ball arrives.
                    # This keeps the arm at home and ready for the next throw instead
                    # of chasing an impossible target for 0.5s then aborting.
                    if flick_q6_deg > 0.0:
                        _settle_est = _move_s_estimate(q_cur, q_wu, 0.02, swing_vel_frac)
                        _q_fp = q_wu.copy()
                        _q_fp[5] += math.radians(flick_q6_deg)
                        _flick_est = _move_s_estimate(q_wu, _q_fp, flick_s, flick_vel_frac)
                        _total_needed = _settle_est + _flick_est + contact_margin_s
                        # Reject only if settle ALONE exceeds full TTI — clearly impossible.
                        # Borderline cases (total > tti but settle < tti) are allowed through;
                        # the dynamic commit and proximity gate handle the tight timing.
                        if _settle_est > tti:
                            _d_deg, _d_j = _max_abs_deg_with_joint(q_wu - q_cur)
                            print_lock_wait(
                                f"unreachable: settle={_settle_est:.3f}s > tti={tti:.3f}s  "
                                f"(+flick={_flick_est:.3f}s total={_total_needed:.3f}s)  "
                                f"Δ={_d_deg:.1f}°@q{_d_j}"
                            )
                            time.sleep(max(0.0, dt - (time.perf_counter() - t0)))
                            continue
                    locked_throw_id += 1
                    locked_intercept = Intercept(
                        position=t_W_strike.copy(),
                        velocity=np.asarray(intercept.velocity, dtype=float).copy(),
                        time_to_impact=float(intercept.time_to_impact),
                        n_bounces=intercept.n_bounces,
                        position_cov=intercept.position_cov,
                    )
                    locked_at_s = time.monotonic()
                    locked_wu_q = q_wu.copy()
                    last_q_cmd = q_cur.copy()
                    last_q_cmd_vel = np.zeros(7)
                    last_goal_write_t = time.perf_counter()
                    locked_ik_err_mm = err_wu * 1000.0
                    locked_ori_err_deg = ori_wu
                    lock_goal_writes = 0
                    lock_ik_failures = 0
                    lock_rejections = pending_lock_rejections
                    pending_lock_rejections = 0
                    lock_peak_goal_err_deg = 0.0
                    lock_delta_deg, lock_delta_joint = _max_abs_deg_with_joint(q_wu - q_cur)
                    audit_lock_strike_W = t_W_strike.copy()
                    audit_lock_wu_A = t_A_windup.copy()
                    audit_lock_tti_s = float(tti)
                    audit_lock_target_delta_deg = lock_delta_deg
                    audit_tracker_rejects = tracker_reject_counts.copy()
                    tracker_reject_counts = {}
                    audit_commit_delta_deg = None
                    audit_timing_margin_s = None
                    audit_wu_goal_err_deg = None
                    audit_follow_goal_err_deg = None
                    audit_home_ok = None
                    print(
                        f"[J7 lock] throw {locked_throw_id} LOCKED: "
                        f"strike_W={_fmt_v3(t_W_strike)}  wu_A={_fmt_v3(t_A_windup)}  "
                        f"time={'mock_hold' if (using_mock and no_commit and mock_cycle_s <= 0.0) else f'{tti:.3f}s'}  "
                        f"ik={err_wu*1000:.1f}mm  ori={ori_wu:.1f}°  "
                        f"target_delta={lock_delta_deg:.1f}°@q{lock_delta_joint}"
                    )
                    if debug_joints:
                        print(f"[J7 lock]   q_cur_deg={_fmt_deg_vec(q_cur)}")
                        print(f"[J7 lock]   q_tgt_deg={_fmt_deg_vec(q_wu)}")
                now_goal_t = time.perf_counter()
                elapsed_goal_s = max(0.0, now_goal_t - last_goal_write_t)
                step_scale = max(1.0, elapsed_goal_s * rate_hz)
                effective_step_deg = tracking_step_deg * step_scale
                last_effective_step_deg = effective_step_deg
                q_target_remaining_deg, _ = _max_abs_deg_with_joint(q_wu - q_cur)
                q_cmd_remaining_deg = 0.0
                if last_q_cmd is not None:
                    q_cmd_remaining_deg, _ = _max_abs_deg_with_joint(q_wu - last_q_cmd)

                settle_ready = (
                    quiet_after_settle_s > 0.0
                    and locked_intercept is not None
                    and q_target_remaining_deg <= quiet_settle_deadband_deg
                    and q_cmd_remaining_deg <= quiet_settle_deadband_deg
                )
                if settle_ready:
                    if settled_since_t is None:
                        settled_since_t = now_goal_t
                    if now_goal_t - settled_since_t >= quiet_after_settle_s:
                        quiet_hold_active = True
                else:
                    settled_since_t = None
                    quiet_hold_active = False
                    quiet_hold_printed = False
                    quiet_hold_goal = None

                if quiet_hold_active:
                    if quiet_hold_goal is None:
                        quiet_hold_goal = q_cur.copy()
                    q_cmd = quiet_hold_goal.copy()
                    last_q_cmd_vel = np.zeros(7)
                    effective_step_deg = 0.0
                    last_effective_step_deg = effective_step_deg
                    if not quiet_hold_printed:
                        print(
                            f"[J7 quiet] settled within {quiet_settle_deadband_deg:.1f}° "
                            f"for {quiet_after_settle_s:.2f}s — holding fixed measured snapshot"
                        )
                        quiet_hold_printed = True
                elif direct_tracking:
                    q_cmd = q_wu
                    last_q_cmd_vel = np.zeros(7)
                else:
                    q_rate_base = last_q_cmd if last_q_cmd is not None else q_cur
                    if tracking_accel_deg_s2 > 0.0:
                        q_cmd, last_q_cmd_vel = _rate_accel_limited_goal(
                            q_wu, q_rate_base, last_q_cmd_vel,
                            tracking_goal_speed_deg_s, tracking_accel_deg_s2, elapsed_goal_s,
                        )
                        effective_step_deg, _ = _max_abs_deg_with_joint(q_cmd - q_rate_base)
                        last_effective_step_deg = effective_step_deg
                    else:
                        q_cmd = _rate_limited_goal(q_wu, q_rate_base, effective_step_deg)
                set_vec(r, GOAL_JOINTS, q_cmd)
                last_goal_write_t = now_goal_t
                if locked_intercept is not None:
                    lock_goal_writes += 1
                last_q_cmd = q_cmd.copy()
                last_wu_q = q_wu
                last_track_target_A = t_A_windup.copy()
                # Stay in IDLE during pre-positioning (unlocked tracking) so the
                # 60° acquire gate applies — early predictions can shift significantly
                # in z as more samples arrive. Only transition to TRACKING on lock.
                if locked_intercept is not None:
                    state = State.TRACKING
            else:
                reasons = []
                if not ik_good:
                    reasons.append(f"ik pos={err_wu*1000:.1f}mm ori={ori_wu:.1f}°")
                if not joints_good:
                    reasons.append("joint limit")
                if not delta_good:
                    max_delta = float(np.max(np.abs(np.degrees(q_wu - delta_ref))))
                    reasons.append(f"{delta_label} {max_delta:.1f}° > {tick_max_delta_deg:.1f}°")
                reason = "; ".join(reasons)
                if lock_first_safe_prediction and locked_intercept is None:
                    print_lock_wait(reason)
                elif locked_intercept is not None:
                    lock_ik_failures += 1

        # ----------------------------------------------------------------
        # 5. Diagnostic print (after IK so goal_err reflects this tick)
        # ----------------------------------------------------------------
        if t0 - last_print_t >= print_interval_s:
            last_print_t = t0
            clip_tag = " [wu CLIP]" if wu_clipped else ""
            extra = ""
            if verbose_tracking and last_q_cmd is not None:
                goal_err_deg, goal_err_joint = _max_abs_deg_with_joint(last_q_cmd - q_cur)
                target_remaining_deg = None
                command_remaining_deg = None
                target_remaining_joint = None
                command_remaining_joint = None
                if last_wu_q is not None:
                    target_remaining_deg, target_remaining_joint = _max_abs_deg_with_joint(last_wu_q - q_cur)
                    command_remaining_deg, command_remaining_joint = _max_abs_deg_with_joint(last_wu_q - last_q_cmd)
                qdot = get_vec(r, SENSOR_JOINT_VELS, 7)
                qdot_max_deg_s = None
                qdot_joint = None
                if qdot is not None:
                    qdot_max_deg_s, qdot_joint = _max_abs_deg_with_joint(qdot)
                tau_cmd = get_vec(r, COMMAND_TORQUES, 7)
                tau_sent = get_vec(r, SENT_TORQUES, 7)
                tau_sensed = get_vec(r, SENSED_TORQUES, 7)
                tau_safety = get_vec(r, SAFETY_TORQUES, 7)
                tau_cmd_max = _max_abs_with_joint(tau_cmd) if tau_cmd is not None else None
                tau_sent_max = _max_abs_with_joint(tau_sent) if tau_sent is not None else None
                tau_sensed_max = _max_abs_with_joint(tau_sensed) if tau_sensed is not None else None
                tau_safety_max = _max_abs_with_joint(tau_safety) if tau_safety is not None else None
                if locked_intercept is not None:
                    lock_peak_goal_err_deg = max(lock_peak_goal_err_deg, goal_err_deg)
                active = r.get(ACTIVE_CONTROLLER)
                extra = f"  err={goal_err_deg:.1f}°@q{goal_err_joint}"
                if target_remaining_deg is not None and command_remaining_deg is not None:
                    if quiet_hold_active:
                        extra += (
                            f"  target_rem={target_remaining_deg:.1f}°@q{target_remaining_joint}"
                            f"  target_gap={command_remaining_deg:.1f}°@q{command_remaining_joint}"
                            "  QUIET_HOLD"
                        )
                    else:
                        settled = target_remaining_deg < max(0.5, 2.0 * tracking_step_deg)
                        extra += (
                            f"  rem={target_remaining_deg:.1f}°@q{target_remaining_joint}"
                            f"  cmd_rem={command_remaining_deg:.1f}°@q{command_remaining_joint}"
                            f"  {'SETTLED' if settled else 'MOVING'}"
                        )
                extra += f"  cmd≤{tracking_goal_speed_deg_s:.0f}°/s step={last_effective_step_deg:.1f}°"
                if qdot_max_deg_s is not None:
                    extra += f"  qdot={qdot_max_deg_s:.1f}°/s@q{qdot_joint}"
                torque_bits = []
                if tau_cmd_max is not None:
                    torque_bits.append(f"cmd={tau_cmd_max[0]:.1f}@q{tau_cmd_max[1]}")
                if tau_sent_max is not None:
                    torque_bits.append(f"sent={tau_sent_max[0]:.1f}@q{tau_sent_max[1]}")
                if tau_sensed_max is not None:
                    torque_bits.append(f"sense={tau_sensed_max[0]:.1f}@q{tau_sensed_max[1]}")
                if tau_safety_max is not None and tau_safety_max[0] > 0.01:
                    torque_bits.append(f"SAFETY={tau_safety_max[0]:.1f}@q{tau_safety_max[1]}")
                if torque_bits:
                    extra += "  tau[" + " ".join(torque_bits) + "]"
                if goal_err_deg > 5.0 and qdot_max_deg_s is not None and qdot_max_deg_s < 1.0:
                    cmd_mag = tau_cmd_max[0] if tau_cmd_max is not None else float("nan")
                    if not np.isfinite(cmd_mag) or cmd_mag < 0.2:
                        extra += "  DIAG=goal_written_but_controller_not_generating_torque"
                    else:
                        extra += "  DIAG=torque_commanded_but_arm_not_moving"
                extra += f"  active={active!r}"
                if dynamic_commit_txt:
                    extra += dynamic_commit_txt
            if debug_joints and verbose_tracking and last_q_cmd is not None:
                extra += f"  q_cur={_fmt_deg_vec(q_cur)}  q_goal={_fmt_deg_vec(last_q_cmd)}"
                if last_wu_q is not None:
                    extra += f"  q_tgt={_fmt_deg_vec(last_wu_q)}"
            if locked_intercept is not None:
                extra += f"  lock=throw{locked_throw_id}"
            time_label = "mock_hold" if (using_mock and no_commit and mock_cycle_s <= 0.0) else f"tti={tti:+.3f}s"
            print(
                f"[J7] {state.name:8s}  {time_label}  "
                f"strike_W=[{t_W_strike[0]:+.3f},{t_W_strike[1]:+.3f},{t_W_strike[2]:+.3f}]  "
                f"wu_A=[{t_A_windup[0]:+.3f},{t_A_windup[1]:+.3f},{t_A_windup[2]:+.3f}]"
                f"{clip_tag}{extra}"
            )

        # ----------------------------------------------------------------
        # 6c. COMMIT — blocking swing sequence
        # ----------------------------------------------------------------
        if commit_due and not no_commit:
            if locked_intercept is None:
                print_lock_wait(
                    f"commit due at tti={tti:.3f}s, but no safe lock yet"
                )
                time.sleep(max(0.0, dt - (time.perf_counter() - t0)))
                continue

            if state != State.TRACKING or last_wu_q is None:
                reason = f"COMMIT skipped: locked throw has no valid wind-up state at tti={tti:.3f}s"
                print(f"[J7] {reason}")
                home_ok = recover_home_blocking("commit-no-windup→home", idle_after=True)
                finish_lock(reason + f"; home_ok={home_ok}", status="WARN")
                time.sleep(max(0.0, dt - (time.perf_counter() - t0)))
                continue

            if not (world_z_min_m <= t_W_strike[2] <= world_z_max_m):
                reason = (
                    f"COMMIT rejected: strike_W.z={t_W_strike[2]:+.3f} m "
                    f"outside [{world_z_min_m:.2f}, {world_z_max_m:.2f}]"
                )
                print(f"[J7] {reason}")
                home_ok = recover_home_blocking("commit-reject→home", idle_after=True)
                finish_lock(reason + f"; home_ok={home_ok}", status="WARN")
                time.sleep(max(0.0, dt - (time.perf_counter() - t0)))
                continue

            if strike_box_reason is not None:
                reason = f"COMMIT rejected: {strike_box_reason}"
                print(f"[J7] {reason}")
                home_ok = recover_home_blocking("commit-reject→home", idle_after=True)
                finish_lock(reason + f"; home_ok={home_ok}", status="WARN")
                time.sleep(max(0.0, dt - (time.perf_counter() - t0)))
                continue

            if wu_clipped or st_clipped or (fw_clipped and not mini_swing):
                clips = []
                if wu_clipped:
                    clips.append("windup")
                if st_clipped:
                    clips.append("strike")
                if fw_clipped and not mini_swing:
                    clips.append("follow")
                reason = f"COMMIT rejected: clipped target(s) {','.join(clips)}"
                print(f"[J7] {reason}")
                home_ok = recover_home_blocking("commit-reject→home", idle_after=True)
                finish_lock(reason + f"; home_ok={home_ok}", status="WARN")
                time.sleep(max(0.0, dt - (time.perf_counter() - t0)))
                continue

            # Solve strike IK seeded from last wind-up solution (smooth)
            q_seed = last_wu_q if last_wu_q is not None else q_cur
            q_strike, err_st, ori_st = ik_solve(
                chain, t_A_strike, q_seed, offset_link7,
                R_A_link7_target=R_A_link7_home, w_ori=w_ori,
            )
            if mini_swing:
                q_follow, err_fw, ori_fw = q_strike, 0.0, 0.0
            else:
                q_follow, err_fw, ori_fw = ik_solve(
                    chain, t_A_follow, q_strike, offset_link7,
                    R_A_link7_target=R_A_link7_home, w_ori=w_ori,
                )
            commit_delta_deg = float(np.degrees(np.abs(q_strike - q_cur)).max())
            audit_commit_delta_deg = commit_delta_deg

            if (not _joints_ok(q_strike)
                    or not _ik_ok(err_st, ori_st, ik_tol_m * 3, max_ori_err_deg, R_A_link7_home)):
                reason = f"COMMIT IK failed (pos={err_st*1000:.1f} mm, ori={ori_st:.1f}°)"
                print(f"[J7] {reason} — aborting")
                home_ok = recover_home_blocking("commit-ik-failed→home", idle_after=True)
                audit_home_ok = home_ok
                finish_lock(reason + f"; home_ok={home_ok}", status="WARN")
                time.sleep(max(0.0, dt - (time.perf_counter() - t0)))
                continue
            # For flick mode use a tighter windup-proximity gate so the arm is
            # near q_wu when the flick fires (settle step stays short).
            effective_delta_max = (
                flick_wu_delta_deg if (flick_q6_deg > 0.0 and flick_wu_delta_deg > 0.0)
                else commit_max_delta_deg
            )
            if commit_delta_deg > effective_delta_max:
                if tti < 0.0:
                    # Ball has passed and arm never made it to wind-up — abort.
                    reason = (
                        f"COMMIT aborted: ball passed (tti={tti:.3f}s) before arm "
                        f"reached wind-up (Δ={commit_delta_deg:.1f}°)"
                    )
                    print(f"[J7] {reason}")
                    home_ok = recover_home_blocking("commit-abort→home", idle_after=True)
                    audit_home_ok = home_ok
                    finish_lock(reason + f"; home_ok={home_ok}", status="WARN")
                else:
                    # Arm still en route to wind-up — keep tracking, don't oscillate.
                    now_goal_t = time.perf_counter()
                    elapsed_goal_s = max(0.0, now_goal_t - last_goal_write_t)
                    q_rate_base = last_q_cmd if last_q_cmd is not None else q_cur
                    if tracking_accel_deg_s2 > 0.0:
                        q_cmd, last_q_cmd_vel = _rate_accel_limited_goal(
                            last_wu_q, q_rate_base, last_q_cmd_vel,
                            tracking_goal_speed_deg_s, tracking_accel_deg_s2, elapsed_goal_s,
                        )
                    else:
                        step_scale = max(1.0, elapsed_goal_s * rate_hz)
                        q_cmd = _rate_limited_goal(
                            last_wu_q, q_rate_base, tracking_step_deg * step_scale,
                        )
                    set_vec(r, GOAL_JOINTS, q_cmd)
                    last_q_cmd = q_cmd.copy()
                    last_goal_write_t = now_goal_t
                time.sleep(max(0.0, dt - (time.perf_counter() - t0)))
                continue

            print(f"\n{'='*60}")
            print(f"[J7] COMMIT  tti={tti:.3f}s")
            print(f"[J7]   strike_A=[{t_A_strike[0]:+.3f},{t_A_strike[1]:+.3f},{t_A_strike[2]:+.3f}]  "
                  f"err={err_st*1000:.1f} mm  ori={ori_st:.1f}°")
            if not mini_swing:
                print(f"[J7]   follow_A=[{t_A_follow[0]:+.3f},{t_A_follow[1]:+.3f},{t_A_follow[2]:+.3f}]  "
                      f"err={err_fw*1000:.1f} mm  ori={ori_fw:.1f}°")
            print(f"[J7]   q_cur  → q_strike max Δ={commit_delta_deg:.1f}°")
            print(f"{'='*60}\n")
            commit_wall_t = time.monotonic()

            switch_ctrl(r, JOINT_CTRL)

            # Active settle at wind-up: hold the IK wind-up goal (not current joints)
            # so the arm actively converges toward q_wu before swinging.  Passive hold
            # (current joints) freezes the arm at its gravitational equilibrium, which
            # may still be far from the IK target if any joint hasn't converged yet.
            settle_goal = last_wu_q if last_wu_q is not None else q_cur
            settle_end = time.perf_counter() + max(0.0, pre_swing_settle_s)
            while time.perf_counter() < settle_end:
                set_vec(r, GOAL_JOINTS, settle_goal)
                time.sleep(0.005)
            q_cur_settled = get_vec(r, SENSOR_JOINTS, 7)
            if q_cur_settled is not None:
                q_cur = q_cur_settled

            # ---- Wrist flick path ----
            if flick_q6_deg > 0.0:
                q_wu_target = last_wu_q if last_wu_q is not None else q_cur
                # Settle to windup position first
                settle_to_wu_s = _move_s_with_velocity_floor(q_cur, q_wu_target, 0.02, swing_vel_frac, "settle→wu")
                tti_elapsed = time.monotonic() - commit_wall_t
                print(
                    f"[J7 flick] settle={settle_to_wu_s:.3f}s  flick_q6={flick_q6_deg:+.1f}°  "
                    f"tti_remaining≈{tti - tti_elapsed:.3f}s"
                )
                settle_seg = segment("settle→wu", q_cur, q_wu_target,
                                     move_s=settle_to_wu_s, hold_s=0.0, publish_hz=200.0)
                audit_wu_goal_err_deg = settle_seg["goal_err_max_deg"]

                q_wu_settled = get_vec(r, SENSOR_JOINTS, 7)
                if q_wu_settled is None:
                    q_wu_settled = q_wu_target

                # Flick: single q6 rotation from windup pose
                q_flick = q_wu_settled.copy()
                q_flick[5] += math.radians(flick_q6_deg)
                flick_s_actual = _move_s_with_velocity_floor(
                    q_wu_settled, q_flick, flick_s, flick_vel_frac, "flick"
                )
                tti_at_flick_start = tti - (time.monotonic() - commit_wall_t)
                flick_margin_s = tti_at_flick_start - flick_s_actual
                audit_timing_margin_s = flick_margin_s
                print(
                    f"[J7 flick] flick_s={flick_s_actual:.3f}s  "
                    f"impact_in≈{tti_at_flick_start:.3f}s  margin={flick_margin_s:+.3f}s"
                )
                flick_seg = segment("flick", q_wu_settled, q_flick,
                                    move_s=flick_s_actual, hold_s=0.05, publish_hz=200.0)
                audit_follow_goal_err_deg = flick_seg["goal_err_max_deg"]

                if flick_seg["goal_err_max_deg"] > 5.0:
                    reason = f"flick did not track (goal_err={flick_seg['goal_err_max_deg']:.1f}°)"
                    print(f"[J7] *** {reason} — recovering home ***")
                    home_ok = recover_home_blocking("flick-failed→home", idle_after=True)
                    audit_home_ok = home_ok
                    finish_lock(reason + f"; home_ok={home_ok}", status="WARN")
                    if max_swings > 0 and locked_throw_id >= max_swings:
                        print(f"[J7] --max-swings={max_swings} reached; exiting")
                        break
                    continue

                home_ok = recover_home_blocking("post-flick→home", idle_after=True)
                audit_home_ok = home_ok
                finish_lock(f"flick completed; home_ok={home_ok}", status="OK" if home_ok else "WARN")
                if max_swings > 0 and locked_throw_id >= max_swings:
                    print(f"[J7] --max-swings={max_swings} reached after throw {locked_throw_id}; exiting")
                    break
                continue

            # ---- Standard swing path ----
            wu_strike_s = _move_s_with_velocity_floor(q_cur, q_strike, swing_s, swing_vel_frac, "wu→strike")
            tti_at_swing_start = tti - (time.monotonic() - commit_wall_t)
            arrival_margin_s = tti_at_swing_start - wu_strike_s
            audit_timing_margin_s = arrival_margin_s
            print(
                f"[J7 timing] impact_in≈{tti_at_swing_start:.3f}s at swing start; "
                f"commanded_strike_arrival≈{wu_strike_s:.3f}s; "
                f"margin={arrival_margin_s:+.3f}s "
                f"({'ON_TIME' if arrival_margin_s >= -0.03 else 'LATE'})"
            )
            seg = segment("wu→strike", q_cur, q_strike,
                          move_s=wu_strike_s, hold_s=wu_hold_s, publish_hz=200.0)
            audit_wu_goal_err_deg = seg["goal_err_max_deg"]

            if seg["goal_err_max_deg"] > 5.0:
                reason = f"wu→strike did not track (goal_err={seg['goal_err_max_deg']:.1f}°)"
                print(f"[J7] *** {reason} — recovering home, no follow-through ***")
                home_ok = recover_home_blocking("strike-failed→home", idle_after=True)
                audit_home_ok = home_ok
                finish_lock(reason + f"; home_ok={home_ok}", status="WARN")
                continue

            if _safety_tripped(r):
                print("[J7] *** SAFETY TORQUES after strike — aborting, slow return ***")
                home_ok = recover_home_blocking("safety→home", idle_after=True)
                audit_home_ok = home_ok
                finish_lock(f"safety torques after strike; home_ok={home_ok}", status="WARN")
                continue

            # Read actual post-strike position to avoid a setpoint snap at the start of
            # the return move (arm may have tracking error from wu→strike).
            q_post_strike = get_vec(r, SENSOR_JOINTS, 7)
            if q_post_strike is None:
                q_post_strike = q_strike

            # Optional follow-through, then one canonical checked recovery home.
            if not mini_swing:
                strike_follow_s = _move_s_with_velocity_floor(
                    q_post_strike, q_follow, swing_s, swing_vel_frac, "strike→follow",
                )
                follow_seg = segment(
                    "strike→follow", q_post_strike, q_follow,
                    move_s=strike_follow_s, hold_s=0.30, publish_hz=200.0,
                )
                audit_follow_goal_err_deg = follow_seg["goal_err_max_deg"]
                if follow_seg["goal_err_max_deg"] > 5.0:
                    reason = f"strike→follow did not track (goal_err={follow_seg['goal_err_max_deg']:.1f}°)"
                    print(f"[J7] *** {reason} — recovering home ***")
                    home_ok = recover_home_blocking("follow-failed→home", idle_after=True)
                    audit_home_ok = home_ok
                    finish_lock(reason + f"; home_ok={home_ok}", status="WARN")
                    continue

            home_ok = recover_home_blocking("post-swing→home", idle_after=True)
            audit_home_ok = home_ok
            finish_lock(f"swing sequence completed; home_ok={home_ok}", status="OK" if home_ok else "WARN")
            if max_swings > 0 and locked_throw_id >= max_swings:
                print(f"[J7] --max-swings={max_swings} reached after throw {locked_throw_id}; exiting")
                break

        time.sleep(max(0.0, dt - (time.perf_counter() - t0)))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="J7: Reactive IK-based ball intercept (joint controller).",
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
    ap.add_argument("--follow-up-offset", type=float, default=0.025,
                    help="J7 follow-through upward component in arm +Z (m).")
    ap.add_argument("--paddle-open-deg", type=float, default=0.0,
                    help="Open the fixed paddle orientation by this many degrees. "
                         "Positive rotates the face normal toward arm +X and slightly upward/lifted.")

    # Timing
    ap.add_argument("--swing-s", type=float, default=0.30,
                    help="Move time for wu→strike and strike→follow (s).")
    ap.add_argument("--gentle-swing", action="store_true",
                    help="Soft first swing: slower timing, 25%% vel cap, shorter "
                         "follow. Does not disable --no-commit.")
    ap.add_argument("--mini-swing", action="store_true",
                    help="Small swing only: wu→strike→home (skip follow-through). "
                         "Best when tracking already holds the wind-up pose.")
    ap.add_argument("--commit-tti", type=float, default=None,
                    help="TTI threshold to commit to swing (s). "
                         "Default: swing_s + pre_swing_settle_s + commit_margin_s.")
    ap.add_argument("--pre-swing-settle-s", type=float, default=J6_PRE_SWING_SETTLE_S,
                    help="Hold wind-up goal for this long immediately before wu→strike. "
                         "Keep small for real-time hitting; old bring-up value was 0.20s.")
    ap.add_argument("--commit-margin-s", type=float, default=J6_COMMIT_MARGIN_S,
                    help="Extra timing margin added to the default commit_tti.")
    ap.add_argument("--contact-margin-s", type=float, default=-0.04,
                    help="J7 dynamic timing target: commit when TTI <= estimated_swing_time + this margin. "
                         "Negative keeps the paddle moving through contact instead of arriving early.")
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
    ap.add_argument("--strike-arm-x-min-m", type=float, default=J6_STRIKE_A_X_MIN_M,
                    help="Minimum arm-frame strike target X before clipping.")
    ap.add_argument("--strike-arm-x-max-m", type=float, default=J6_STRIKE_A_X_MAX_M,
                    help="Maximum arm-frame strike target X before clipping.")
    ap.add_argument("--strike-arm-y-abs-max-m", type=float, default=J6_STRIKE_A_Y_ABS_MAX_M,
                    help="Maximum absolute arm-frame strike target Y before clipping.")
    ap.add_argument("--strike-arm-z-min-m", type=float, default=J6_STRIKE_A_Z_MIN_M,
                    help="Minimum arm-frame strike target Z before clipping.")
    ap.add_argument("--strike-arm-z-max-m", type=float, default=J6_STRIKE_A_Z_MAX_M,
                    help="Maximum arm-frame strike target Z before clipping.")
    ap.add_argument("--acquire-max-delta-deg", type=float, default=J6_ACQUIRE_MAX_DELTA_DEG,
                    help="Max per-joint delta from current pose to enter TRACKING from IDLE.")
    ap.add_argument("--max-delta-deg", type=float, default=J6_MAX_DELTA_DEG,
                    help="Max per-joint IK goal delta from current pose while TRACKING.")
    ap.add_argument("--ik-tol-mm", type=float, default=J6_IK_TOL_M * 1000.0,
                    help="Reject IK solutions whose sweet-spot error exceeds this (mm).")
    ap.add_argument("--tracking-step-deg", type=float, default=J6_TRACKING_STEP_DEG,
                    help="Max per-tick joint-goal step during non-blocking tracking.")
    ap.add_argument("--tracking-accel-deg-s2", type=float, default=0.0,
                    help="Optional acceleration cap for the joint-goal tracking ramp. "
                         "Use 0 to keep pure rate limiting.")
    ap.add_argument("--direct-tracking", action="store_true",
                    help="Write full IK wind-up goal each tick (no rate limit). "
                         "Use for mock/bring-up when 0.25°/tick looks like no motion.")
    ap.add_argument("--verbose-tracking", action="store_true",
                    help="Print target remaining, command speed, qdot and active controller while tracking.")
    ap.add_argument("--full-segment-logs", action="store_true",
                    help="Print full run_segment tables. Default J7 output is compact for easy pasteback.")
    ap.add_argument("--log-period-s", type=float, default=0.50,
                    help="Seconds between tracking diagnostic prints.")
    ap.add_argument("--log-file", type=str, default=None, metavar="PATH",
                    help="Append all stdout to PATH in addition to the terminal. "
                         "Use 'auto' to create sports_bot/logs/j7_YYYYMMDD_HHMMSS.log "
                         "automatically.")
    ap.add_argument("--debug-joints", action="store_true",
                    help="Also print q_cur/q_goal/q_target joint vectors in degrees.")
    ap.add_argument("--quiet-after-settle-s", type=float, default=0.0,
                    help="After a locked target has settled for this long, command "
                         "one fixed measured-joint snapshot to reduce hold buzz (0 disables).")
    ap.add_argument("--quiet-settle-deadband-deg", type=float, default=1.0,
                    help="Joint deadband used by --quiet-after-settle-s.")
    ap.add_argument("--shutdown-hold-s", type=float, default=2.0,
                    help="On Ctrl-C, refresh one measured joint-goal snapshot for this long before exiting. \
                         Use 0 to exit immediately after one hold write.")
    ap.add_argument("--no-lock-prediction", dest="lock_prediction", action="store_false",
                    help="Chase each new ball prediction instead of freezing the first safe one.")
    ap.set_defaults(lock_prediction=True)
    ap.add_argument("--lock-tti-max", type=float, default=0.0,
                    help="Only lock a prediction when TTI ≤ this value (seconds). "
                         "0 = lock as soon as a valid prediction exists (default). "
                         "Set to e.g. 0.65 to wait for a mature, accurate prediction "
                         "before freezing — early extrapolations are noisy.")
    ap.add_argument("--lock-release-after-impact-s", type=float,
                    default=J6_LOCK_RELEASE_AFTER_IMPACT_S,
                    help="When prediction locking is enabled, release a locked throw this long after impact.")
    ap.add_argument("--post-impact-idle-s", type=float, default=1.5,
                    help="After a lock releases, block new locks for this many seconds so the arm "
                         "can return to home before the next throw (default: 1.5s).")
    ap.add_argument("--swing-vel-frac", type=float, default=J6_SWING_VEL_FRAC,
                    help="Max blocking-swing peak velocity as a fraction of XML velocity limits.")
    ap.add_argument("--z-mode", choices=["fixed-arm", "predicted"], default="fixed-arm",
                    help="Use fixed arm-frame Z for J7 bringup, or raw predicted Z.")
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
    ap.add_argument("--skip-cal", action="store_true",
                    help="Skip cartesian_controller cal (avoids OpenSai fault on "
                         "controller switch). Uses --offset-link7 or 0.43 m default.")
    ap.add_argument("--offset-link7", nargs=3, type=float, metavar=("X", "Y", "Z"),
                    default=None,
                    help="Sweet-spot offset in link7 frame (m). Default with --skip-cal: "
                         "[0, 0, 0.43].")

    # Calibration / infra
    ap.add_argument("--calibration", default=None,
                    help="Path to arm_base_offset_calibration.json.")
    ap.add_argument("--robot-name", default="FrankaRobot")
    ap.add_argument("--redis-host", default="localhost")
    ap.add_argument("--redis-port", type=int, default=6379)

    # Ball tracker overrides
    ap.add_argument("--min-lookahead", type=float, default=None)
    ap.add_argument("--max-lookahead", type=float, default=None)
    ap.add_argument("--tracker-history-size", type=int, default=None,
                    help="Override BallTrackerConfig.history_size for the LS fit.")
    ap.add_argument("--tracker-history-max-age-s", type=float, default=None,
                    help="Override BallTrackerConfig.history_max_age_s for the LS fit.")
    ap.add_argument("--tracker-min-history", type=int, default=None,
                    help="Minimum history samples before producing an intercept.")
    ap.add_argument("--tracker-median-window", type=int, default=None,
                    help="Raw position median filter window; use 1 to disable.")
    ap.add_argument("--tracker-max-implied-speed-mps", type=float, default=None,
                    help="Reject raw OptiTrack samples that imply more than this speed.")
    ap.add_argument("--tracker-stale-position-eps-m", type=float, default=None,
                    help="Position-change threshold used to detect frozen Redis ball keys.")
    ap.add_argument("--tracker-stale-timeout-s", type=float, default=None,
                    help="How long an unchanged Redis ball key may be sampled before reset/reject.")
    ap.add_argument("--wu-hold-s", type=float, default=0.10,
                    help="Hold at the strike pose before starting follow-through (s). "
                         "Default 0.10 matches original behavior; use 0 for arm-moving-at-contact.")
    ap.add_argument("--max-swings", type=int, default=0,
                    help="Exit cleanly after N successful swings (0 = unlimited). "
                         "Use 1 for single-throw replay tests.")
    ap.add_argument("--flick-q6-deg", type=float, default=0.0,
                    help="Wrist-flick mode: after tracking to windup, snap q6 by this many "
                         "degrees instead of the multi-joint swing. 0 = disabled (use normal swing). "
                         "Try 25–35° for a strong flick. Requires picklebot_j7.xml for higher q6 vel limit.")
    ap.add_argument("--flick-s", type=float, default=0.25,
                    help="Requested move time for the q6 flick (s). Will be stretched to "
                         "respect the q6 velocity limit. With picklebot_j7.xml q6 limit of "
                         "2.5 rad/s, a 30° flick needs ~0.21s minimum.")
    ap.add_argument("--flick-vel-frac", type=float, default=1.0,
                    help="Velocity cap fraction for the flick segment (applied to q6 XML limit). "
                         "1.0 = use full q6 velocity limit for maximum flick speed.")
    ap.add_argument("--flick-wu-delta-deg", type=float, default=8.0,
                    help="Flick mode only: commit only when q_cur→q_strike delta is below this "
                         "(degrees). Tighter than commit_max_delta_deg so arm is near q_wu before "
                         "the flick fires and settle stays short. Default 8°.")

    args = ap.parse_args()

    if args.log_file is not None:
        log_path = args.log_file
        if log_path == "auto":
            log_dir = os.path.join(_THIS_DIR, "..", "logs")
            log_path = os.path.join(log_dir, f"j7_{time.strftime('%Y%m%d_%H%M%S')}.log")
        log_path = os.path.abspath(log_path)
        sys.stdout = _Tee(log_path)
        print(f"[J7] logging to {log_path}")

    if args.gentle_swing:
        args.swing_s = J6_GENTLE_SWING_S
        args.swing_vel_frac = J6_GENTLE_SWING_VEL_FRAC
        args.follow_offset = J6_GENTLE_FOLLOW_OFFSET_M
        args.return_s = J6_GENTLE_RETURN_S
        if args.commit_tti is None:
            args.commit_tti = J6_GENTLE_COMMIT_TTI_S
        print("[J7] --gentle-swing: "
              f"swing_s={args.swing_s:.2f}s  vel={args.swing_vel_frac:.0%}  "
              f"follow={args.follow_offset:.2f}m  commit_tti={args.commit_tti:.2f}s")

    commit_tti = (
        args.commit_tti if args.commit_tti is not None
        else args.swing_s + max(0.0, args.pre_swing_settle_s) + args.commit_margin_s
    )
    if commit_tti <= 0:
        ap.error("commit_tti must be positive")

    # ---- Redis ----
    r = redis.Redis(host=args.redis_host, port=args.redis_port,
                    decode_responses=True)
    try:
        r.ping()
    except redis.exceptions.ConnectionError as e:
        sys.exit(f"[J7] cannot reach Redis at {args.redis_host}:{args.redis_port}: {e}")

    # Read actual velocity limits from OpenSai so flick/swing timing matches whichever
    # XML was loaded (e.g. picklebot_j7.xml has q6=2.5 rad/s vs picklebot.xml q6=1.1 rad/s).
    global J6_XML_VEL_LIMIT_RAD_S
    _vel_key = f"{NS}::{JOINT_CTRL}::joint_task::velocity_saturation_limit"
    _vel_raw = r.get(_vel_key)
    if _vel_raw is not None:
        try:
            _limits = np.array(json.loads(_vel_raw), dtype=float)
            if len(_limits) == 7:
                J6_XML_VEL_LIMIT_RAD_S = _limits
                print(f"[J7] velocity limits from OpenSai: "
                      f"{np.degrees(J6_XML_VEL_LIMIT_RAD_S).round(1).tolist()} °/s")
            else:
                print(f"[J7] WARNING: velocity limits key has {len(_limits)} values (expected 7), "
                      f"using hardcoded defaults")
        except Exception:
            print(f"[J7] WARNING: could not parse velocity limits from Redis, "
                  f"using hardcoded defaults")
    else:
        print(f"[J7] WARNING: velocity limits key not found ({_vel_key}), "
              f"OpenSai may not be running — using hardcoded defaults")

    # ---- Calibration file ----
    cal_path = args.calibration or arm_base_offset_calibration_path()
    if not os.path.isfile(cal_path):
        sys.exit(
            f"[J7] arm_base_offset_calibration.json not found: {cal_path}\n"
            f"      Run: python sports_bot/scripts/calibrate_arm_base_offset.py"
        )
    cal = load_arm_base_offset_calibration(cal_path)
    print(f"[J7] arm base calibration: {cal_path}  (rigid body {cal.base_rigid_body_id})")

    mock_intercepts: list[np.ndarray] | None = None
    if args.mock_intercept:
        mock_intercepts = [np.array(triple, dtype=float) for triple in args.mock_intercept]

    # Verify cart is visible before doing anything (unless mock + identity base)
    if not (mock_intercepts and args.mock_identity_base):
        if read_rigid_body_pose_W(r, cal.base_rigid_body_id) is None:
            sys.exit(f"[J7] cart rigid body {cal.base_rigid_body_id} not visible in Redis. "
                     f"Check OptiTrack streamer, or use --mock-identity-base.")

    # ---- Build ikpy chain ----
    if not os.path.isfile(URDF_PATH):
        sys.exit(f"[J7] URDF not found: {URDF_PATH}  (run from OpenSai root)")
    chain = build_chain()

    # ---- Calibrate offset_link7 at current pose ----
    if args.skip_cal:
        if args.offset_link7 is not None:
            offset_link7 = np.asarray(args.offset_link7, dtype=float)
        else:
            offset_link7 = J6_DEFAULT_OFFSET_LINK7.copy()
        q_at_cal, offset_link7 = seed_offset_link7_no_cartesian(r, offset_link7)
    else:
        q_at_cal, offset_link7 = calibrate_offset_link7(r, chain)

    if args.print_cal_only:
        print("[J7] --print-cal-only: done.")
        return

    # ---- Move to home before tracking ----
    print("[J7] moving to home pose ...")
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
        R_home_nominal = _link7_R_A(chain, Q_HOME_RAD)
        R_A_link7_home = _rot_y_rad(math.radians(-args.paddle_open_deg)) @ R_home_nominal
        face_A = R_A_link7_home @ (offset_link7 / max(np.linalg.norm(offset_link7), 1e-9))
        print("[J7] home link7 orientation from nominal Q_HOME")
        print(f"[J7]   paddle_open={args.paddle_open_deg:+.1f}°  "
              f"strike-face normal_A ≈ [{face_A[0]:+.3f},{face_A[1]:+.3f},{face_A[2]:+.3f}]")

    # ---- Ball tracker (skipped in mock mode) ----
    tracker: BallTracker | None = None
    if mock_intercepts is None:
        cfg = BallTrackerConfig()
        if args.min_lookahead is not None:
            cfg.min_lookahead = args.min_lookahead
        if args.max_lookahead is not None:
            cfg.max_lookahead = args.max_lookahead
        if args.tracker_history_size is not None:
            cfg.history_size = args.tracker_history_size
        if args.tracker_history_max_age_s is not None:
            cfg.history_max_age_s = args.tracker_history_max_age_s
        if args.tracker_min_history is not None:
            cfg.min_history_for_prediction = args.tracker_min_history
        if args.tracker_median_window is not None:
            cfg.median_filter_window = args.tracker_median_window
        if args.tracker_max_implied_speed_mps is not None:
            cfg.max_implied_speed_mps = args.tracker_max_implied_speed_mps
        if args.tracker_stale_position_eps_m is not None:
            cfg.stale_position_epsilon_m = args.tracker_stale_position_eps_m
        if args.tracker_stale_timeout_s is not None:
            cfg.stale_position_timeout_s = args.tracker_stale_timeout_s
        print(
            f"[J7 tracker] LS volley predictor: hist={cfg.history_size} samples/{cfg.history_max_age_s:.2f}s "
            f"min_hist={cfg.min_history_for_prediction} median={cfg.median_filter_window} "
            f"speed_gate≤{cfg.max_implied_speed_mps:.1f}m/s stale≤{cfg.stale_position_timeout_s:.2f}s@{cfg.stale_position_epsilon_m*1000:.1f}mm "
            f"lookahead=[{cfg.min_lookahead:.2f},{cfg.max_lookahead:.2f}]s"
        )

        keys = RedisKeys(ball_source="optitrack")
        keys.ball.__dict__["optitrack_rigid_body_id"] = args.ball_rigid_body_id

        ball_key = keys.ball.optitrack_position
        if r.get(ball_key) is None:
            print(f"[J7] WARNING: {ball_key} is empty — is the OptiTrack streamer running?")
        else:
            print(f"[J7] reading ball from {ball_key}")

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
            follow_up_offset_m=args.follow_up_offset,
            swing_s=args.swing_s,
            contact_margin_s=args.contact_margin_s,
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
            tracking_accel_deg_s2=args.tracking_accel_deg_s2,
            swing_vel_frac=args.swing_vel_frac,
            acquire_max_delta_deg=args.acquire_max_delta_deg,
            max_delta_deg=args.max_delta_deg,
            ik_tol_m=args.ik_tol_mm / 1000.0,
            z_mode=args.z_mode,
            fixed_arm_z_m=args.fixed_arm_z_m,
            w_ori=args.w_ori,
            max_ori_err_deg=args.max_ori_err_deg,
            pre_swing_settle_s=args.pre_swing_settle_s,
            strike_arm_x_min_m=args.strike_arm_x_min_m,
            strike_arm_x_max_m=args.strike_arm_x_max_m,
            strike_arm_y_abs_max_m=args.strike_arm_y_abs_max_m,
            strike_arm_z_min_m=args.strike_arm_z_min_m,
            strike_arm_z_max_m=args.strike_arm_z_max_m,
            mock_intercepts=mock_intercepts,
            mock_tti=args.mock_tti,
            mock_cycle_s=args.mock_cycle_s,
            mock_identity_base=args.mock_identity_base,
            direct_tracking=args.direct_tracking,
            verbose_tracking=args.verbose_tracking,
            full_segment_logs=args.full_segment_logs,
            mini_swing=args.mini_swing,
            log_period_s=args.log_period_s,
            debug_joints=args.debug_joints,
            quiet_after_settle_s=args.quiet_after_settle_s,
            quiet_settle_deadband_deg=args.quiet_settle_deadband_deg,
            lock_first_safe_prediction=args.lock_prediction,
            lock_tti_max=args.lock_tti_max,
            lock_release_after_impact_s=args.lock_release_after_impact_s,
            post_impact_idle_s=args.post_impact_idle_s,
            wu_hold_s=args.wu_hold_s,
            max_swings=args.max_swings,
            flick_q6_deg=args.flick_q6_deg,
            flick_s=args.flick_s,
            flick_vel_frac=args.flick_vel_frac,
            flick_wu_delta_deg=args.flick_wu_delta_deg,
        )
    except KeyboardInterrupt:
        if args.shutdown_hold_s > 0.0:
            _hold_current_joints_for(r, args.shutdown_hold_s, publish_hz=100.0)
            print(
                f"\n[J7] stopped — refreshed current joint hold for "
                f"{args.shutdown_hold_s:.1f}s."
            )
        else:
            _hold_current_joints(r)
            print("\n[J7] stopped — holding current joint position in Redis.")


if __name__ == "__main__":
    main()
