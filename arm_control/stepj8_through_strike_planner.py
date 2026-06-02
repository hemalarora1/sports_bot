#!/usr/bin/env python3
"""
Step J8: Through-strike pickleball planner with mobile base coordination.

Arm strike
----------
J8 keeps the robust J7 plumbing (OptiTrack -> LS predictor, arm-base transform,
sweet-spot IK, Redis joint controller) but removes the wind-up-then-flick state
machine.  A valid throw is handled as one timed waypoint problem:

    q_now  ->  q_strike at impact time  ->  q_follow  ->  home

Before the hard commit, predictions may keep improving.  The planner retargets
only when the new strike point is close enough and the full timing budget stays
feasible.  At hard commit, the impact waypoint freezes and the arm executes a
cubic Hermite trajectory that passes through q_strike with nonzero velocity.

Mobile base coordination (pass --base-ready-x to enable)
---------------------------------------------------------
1. Lateral tracking (pre-commit): while the arm is planning its swing, the base
   is continuously commanded to align its Y position with the predicted ball
   intercept Y (clamped to --base-y-min / --base-y-max).  X and yaw stay fixed
   at the ready pose.  When the ball is lost, the base returns to ready after
   --hold-base-after-lost-s seconds.

2. Forward thrust (at commit): the moment J8 commits to a swing, the base is
   commanded forward --base-thrust-m metres (default 0.5 m) along the arm's
   world-frame +X axis.  This fires concurrently with the arm strike, so the
   base is accelerating while the arm swings, adding forward momentum to the
   hit.  After the strike completes the base is commanded back to ready.

3. Live IK correction during the strike (100 Hz): because the base is moving
   while the arm swings, the arm's joint trajectory is updated at 100 Hz using
   live OptiTrack base pose data.  Each correction re-solves IK for the fixed
   world-frame ball target in the current arm frame, so the arm tracks the ball
   even as the base moves.  Joint velocity limits are not affected by base
   translation (limits are angular, not world-frame linear).

Prereqs (in addition to the normal J8 stack):
   - base_bridge.py running (forwards sports_bot::cmd::base::goal_pose to the
     TidyBot driver in its odometry frame).
   - OptiTrack streaming both the ball and the cart rigid bodies.
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
from dataclasses import dataclass

import numpy as np
import redis

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_OPENSAI_DIR = os.path.dirname(os.path.dirname(_THIS_DIR))
for _path in (_THIS_DIR, _OPENSAI_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from stepj1_joint_nudge import (  # noqa: E402
    ACTIVE_CONTROLLER,
    COMMAND_TORQUES,
    GOAL_JOINTS,
    SAFETY_TORQUES,
    SENSED_TORQUES,
    SENSOR_JOINTS,
    SENSOR_JOINT_VELS,
    SENT_TORQUES,
    ensure_joint_controller,
    get_vec,
    set_vec,
)

from stepj7_strike_planner import (  # noqa: E402
    CART_CTRL,
    JOINT_CTRL,
    J6_ACQUIRE_MAX_DELTA_DEG,
    J6_DEFAULT_OFFSET_LINK7,
    J6_HOME_SETTLE_TOL_DEG,
    J6_HOME_VEL_FRAC,
    J6_IK_TOL_M,
    J6_LOCK_RELEASE_AFTER_IMPACT_S,
    J6_MAX_DELTA_DEG,
    J6_MAX_ORI_ERR_DEG,
    J6_REACH_M,
    J6_STRIKE_A_X_MAX_M,
    J6_STRIKE_A_X_MIN_M,
    J6_STRIKE_A_Y_ABS_MAX_M,
    J6_STRIKE_A_Z_MAX_M,
    J6_STRIKE_A_Z_MIN_M,
    J6_TARGET_JUMP_MAX_M,
    J6_W_ORI,
    J6_WORLD_Z_MAX_M,
    J6_WORLD_Z_MIN_M,
    J6_XML_VEL_LIMIT_RAD_S,
    J6_Z_MAX_M,
    J6_Z_MIN_M,
    NS,
    Q_HI,
    Q_HOME_RAD,
    Q_LO,
    URDF_PATH,
    _Tee,
    _fmt_q,
    _fmt_v3,
    _hold_current_joints,
    _hold_current_joints_for,
    _home_error_deg,
    _ik_ok,
    _joints_ok,
    _link7_R_A,
    _max_abs_deg_with_joint,
    _max_abs_with_joint,
    _move_s_with_velocity_floor,
    _rate_accel_limited_goal,
    _rate_limited_goal,
    _return_home_blocking,
    _rot_y_rad,
    _safe_target,
    _safety_tripped,
    _strike_box_reject_reason,
    build_chain,
    calibrate_offset_link7,
    ik_solve,
    move_to_home_pose,
    seed_offset_link7_no_cartesian,
    switch_ctrl,
)

from sports_bot.state_machine.ball_tracker import BallTracker, Intercept  # noqa: E402
from sports_bot.state_machine.config import BallTrackerConfig  # noqa: E402
from sports_bot.state_machine.redis_keys import RedisKeys  # noqa: E402
from sports_bot.utils.frames import (  # noqa: E402
    ArmBaseOffsetCalibration,
    arm_base_offset_calibration_path,
    compute_T_W_A_from_base_offset,
    load_arm_base_offset_calibration,
    read_rigid_body_pose_W,
)


@dataclass
class StrikeCandidate:
    strike_W: np.ndarray
    strike_A: np.ndarray
    pre_A: np.ndarray
    follow_A: np.ndarray
    q_pre: np.ndarray
    q_strike: np.ndarray
    q_follow: np.ndarray
    v_strike: np.ndarray
    tti_s: float
    observed_t: float
    min_strike_s: float
    strike_time_s: float
    budget_margin_s: float
    cmd_peak_frac: float
    ik_err_mm: float
    ori_err_deg: float
    follow_ik_err_mm: float
    q6_follow_deg: float
    follow_duration_s: float

    def aged_tti(self) -> float:
        return self.tti_s - (time.monotonic() - self.observed_t)


def _fmt_counts(counts: dict[str, int]) -> str:
    if not counts:
        return "none"
    return ",".join(
        f"{k}:{v}" for k, v in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:5]
    )


def _jsonable(value):
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value.astype(float).tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


class _TraceLogger:
    def __init__(self, path: str | None) -> None:
        self.path = path
        self._file = None
        if path:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self._file = open(path, "a", buffering=1)

    def write(self, event: str, **fields) -> None:
        if self._file is None:
            return
        row = {"t_mono": time.monotonic(), "event": event}
        row.update({k: _jsonable(v) for k, v in fields.items()})
        self._file.write(json.dumps(row, separators=(",", ":")) + "\n")

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


def _parse_raw_json(raw: str | bytes | None):
    if raw is None:
        return None
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    try:
        return json.loads(raw)
    except Exception:
        return raw


def _tracker_snapshot(tracker: BallTracker | None) -> dict:
    if tracker is None:
        return {}
    hist = list(getattr(tracker, "_history", []))
    raw_buffer = list(getattr(tracker, "_raw_buffer", []))
    snap = {
        "history_len": len(hist),
        "raw_buffer_len": len(raw_buffer),
        "last_reject_reason": getattr(tracker, "last_reject_reason", ""),
        "time_since_last_seen_s": tracker.time_since_last_seen(),
    }
    if hist:
        snap["latest_history_t"] = hist[-1].t
        snap["latest_history_pos_W"] = hist[-1].pos
        snap["history_first_t"] = hist[0].t
        snap["history_first_pos_W"] = hist[0].pos
    fit_fn = getattr(tracker, "_fit_state", None)
    if callable(fit_fn):
        try:
            fit = fit_fn()
            if fit is not None:
                p0, v0, fit_t = fit
                snap["fit_pos_W"] = p0
                snap["fit_vel_W"] = v0
                snap["fit_t"] = fit_t
        except Exception as exc:
            snap["fit_error"] = str(exc)
    return snap


def _intercept_snapshot(intercept: Intercept | None) -> dict | None:
    if intercept is None:
        return None
    return {
        "position_W": intercept.position,
        "velocity_W": intercept.velocity,
        "tti_s": intercept.time_to_impact,
        "n_bounces": intercept.n_bounces,
    }


def _candidate_snapshot(cand: StrikeCandidate | None) -> dict | None:
    if cand is None:
        return None
    aged_tti = cand.aged_tti()
    return {
        "strike_W": cand.strike_W,
        "strike_A": cand.strike_A,
        "pre_A": cand.pre_A,
        "follow_A": cand.follow_A,
        "q_pre_deg": np.degrees(cand.q_pre),
        "q_strike_deg": np.degrees(cand.q_strike),
        "q_follow_deg": np.degrees(cand.q_follow),
        "v_strike_deg_s": np.degrees(cand.v_strike),
        "tti_original_s": cand.tti_s,
        "tti_aged_s": aged_tti,
        "min_strike_s": cand.min_strike_s,
        "strike_time_s": cand.strike_time_s,
        "budget_margin_s": cand.budget_margin_s,
        "cmd_peak_frac": cand.cmd_peak_frac,
        "ik_err_mm": cand.ik_err_mm,
        "ori_err_deg": cand.ori_err_deg,
        "follow_ik_err_mm": cand.follow_ik_err_mm,
        "q6_follow_deg": cand.q6_follow_deg,
        "follow_duration_s": cand.follow_duration_s,
    }


def _read_xml_velocity_limits(r: redis.Redis) -> np.ndarray:
    vel_limits = J6_XML_VEL_LIMIT_RAD_S.copy()
    key = f"{NS}::{JOINT_CTRL}::joint_task::velocity_saturation_limit"
    raw = r.get(key)
    if raw is None:
        print(f"[J8] WARNING: velocity limits key not found ({key}); using hardcoded defaults")
        return vel_limits
    try:
        parsed = np.asarray(json.loads(raw), dtype=float)
        if parsed.shape == (7,):
            print(f"[J8] velocity limits from OpenSai: {np.degrees(parsed).round(1).tolist()} °/s")
            return parsed
    except Exception:
        pass
    print("[J8] WARNING: could not parse velocity limits from Redis; using hardcoded defaults")
    return vel_limits


def _hermite(q0: np.ndarray, v0: np.ndarray, q1: np.ndarray, v1: np.ndarray,
             t: float, T: float) -> tuple[np.ndarray, np.ndarray]:
    T = max(float(T), 1e-6)
    s = float(np.clip(t / T, 0.0, 1.0))
    h00 = 2.0 * s**3 - 3.0 * s**2 + 1.0
    h10 = s**3 - 2.0 * s**2 + s
    h01 = -2.0 * s**3 + 3.0 * s**2
    h11 = s**3 - s**2
    q = h00 * q0 + h10 * T * v0 + h01 * q1 + h11 * T * v1

    dh00 = (6.0 * s**2 - 6.0 * s) / T
    dh10 = 3.0 * s**2 - 4.0 * s + 1.0
    dh01 = (-6.0 * s**2 + 6.0 * s) / T
    dh11 = 3.0 * s**2 - 2.0 * s
    qdot = dh00 * q0 + dh10 * v0 + dh01 * q1 + dh11 * v1
    return q, qdot


def _trajectory_sample(
    q_start: np.ndarray,
    q_strike: np.ndarray,
    q_follow: np.ndarray,
    v_strike: np.ndarray,
    strike_s: float,
    follow_s: float,
    *,
    samples: int = 180,
) -> tuple[np.ndarray, np.ndarray]:
    total_s = max(1e-3, strike_s + follow_s)
    ts = np.linspace(0.0, total_s, max(8, samples))
    q_goals = np.zeros((len(ts), 7))
    qdots = np.zeros((len(ts), 7))
    zeros = np.zeros(7)
    for i, t in enumerate(ts):
        if t <= strike_s:
            q, qdot = _hermite(q_start, zeros, q_strike, v_strike, t, strike_s)
        else:
            q, qdot = _hermite(q_strike, v_strike, q_follow, zeros, t - strike_s, follow_s)
        q_goals[i] = q
        qdots[i] = qdot
    return q_goals, qdots


def _trajectory_peak_frac(
    q_start: np.ndarray,
    q_strike: np.ndarray,
    q_follow: np.ndarray,
    v_strike: np.ndarray,
    strike_s: float,
    follow_s: float,
    vel_cap: np.ndarray,
) -> float:
    peak, _idx, frac = _trajectory_peak_profile(
        q_start, q_strike, q_follow, v_strike, strike_s, follow_s, vel_cap,
    )
    del peak
    return frac


def _trajectory_peak_profile(
    q_start: np.ndarray,
    q_strike: np.ndarray,
    q_follow: np.ndarray,
    v_strike: np.ndarray,
    strike_s: float,
    follow_s: float,
    vel_cap: np.ndarray,
) -> tuple[np.ndarray, int, float]:
    _, qdots = _trajectory_sample(q_start, q_strike, q_follow, v_strike, strike_s, follow_s)
    peak = np.max(np.abs(qdots), axis=0)
    ratios = peak / np.maximum(vel_cap, 1e-6)
    idx = int(np.argmax(ratios))
    return peak, idx, float(ratios[idx])


def _minimum_feasible_strike_time(
    q_start: np.ndarray,
    q_strike: np.ndarray,
    q_follow: np.ndarray,
    v_strike: np.ndarray,
    follow_s: float,
    vel_cap: np.ndarray,
) -> float:
    lo = 0.035
    hi = max(0.08, _move_s_estimate_local(q_start, q_strike, vel_cap) * 1.4)
    while _trajectory_peak_frac(q_start, q_strike, q_follow, v_strike, hi, follow_s, vel_cap) > 1.0:
        hi *= 1.35
        if hi > 2.5:
            return hi
    for _ in range(24):
        mid = 0.5 * (lo + hi)
        if _trajectory_peak_frac(q_start, q_strike, q_follow, v_strike, mid, follow_s, vel_cap) <= 1.0:
            hi = mid
        else:
            lo = mid
    return hi


def _move_s_estimate_local(q_start: np.ndarray, q_target: np.ndarray, vel_cap: np.ndarray) -> float:
    return float(np.max(1.5 * np.abs(q_target - q_start) / np.maximum(vel_cap, 1e-6)))


def _compact_segment(
    r: redis.Redis,
    label: str,
    q_start: np.ndarray,
    q_target: np.ndarray,
    *,
    move_s: float,
    vel_frac: float,
    hold_s: float,
    publish_hz: float,
    full_segment_logs: bool,
) -> dict:
    from stepj1_joint_nudge import run_segment

    move_s_actual = _move_s_with_velocity_floor(q_start, q_target, move_s, vel_frac, label)
    if full_segment_logs:
        return run_segment(
            r, label, q_start, q_target,
            move_s=move_s_actual, hold_s=hold_s, publish_hz=publish_hz,
        )
    with contextlib.redirect_stdout(io.StringIO()):
        seg = run_segment(
            r, label, q_start, q_target,
            move_s=move_s_actual, hold_s=hold_s, publish_hz=publish_hz,
        )
    status = "OK" if seg["goal_err_max_deg"] <= 5.0 else "WARN"
    print(
        f"[J8 segment] {label}: {status} move_s={move_s_actual:.3f}s "
        f"hold_s={hold_s:.2f}s goal_err={seg['goal_err_max_deg']:.2f}° "
        f"qdot_max={seg['qdot_max_deg_s']:.1f}°/s"
    )
    return seg


def _run_through_strike(
    r: redis.Redis,
    cand: StrikeCandidate,
    *,
    publish_hz: float,
    full_segment_logs: bool,
    chain=None,
    offset_link7: np.ndarray | None = None,
    R_A_link7_home: np.ndarray | None = None,
    w_ori: float = 0.0,
    ik_tol_m: float = 0.005,
    cal=None,
    base_correction_hz: float = 100.0,
) -> dict:
    live_correction = (
        cal is not None and chain is not None and offset_link7 is not None
    )
    q_start = get_vec(r, SENSOR_JOINTS, 7)
    if q_start is None:
        q_start = cand.q_pre.copy()
    q_strike_live = cand.q_strike.copy()
    follow_s = max(0.05, cand.follow_duration_s)
    total_s = cand.strike_time_s + follow_s
    period_s = 1.0 / max(1.0, publish_hz)
    sample_period_s = 0.01
    base_ik_period = 1.0 / max(1.0, base_correction_hz)
    qdot_peak = np.zeros(7)
    qdot_cmd_peak = np.zeros(7)
    strike_err_deg = float("nan")
    follow_err_deg = float("nan")
    strike_sampled = False
    prev_q = None
    prev_t = None
    next_publish = time.perf_counter()
    next_sample = next_publish
    next_base_ik = next_publish
    t0 = time.perf_counter()
    zeros = np.zeros(7)

    print(
        f"[J8 through] start: strike_s={cand.strike_time_s:.3f}s "
        f"follow_s={follow_s:.3f}s q6_follow={cand.q6_follow_deg:+.1f}°"
        + (" [live base correction ON]" if live_correction else "")
    )
    if full_segment_logs:
        print("[J8 through] q_start:  " + _fmt_q(q_start))
        print("[J8 through] q_strike: " + _fmt_q(cand.q_strike))
        print("[J8 through] q_follow: " + _fmt_q(cand.q_follow))

    while True:
        now = time.perf_counter()
        elapsed = now - t0

        # Re-solve strike IK against the current base position so the arm
        # tracks the fixed world-frame ball target even as the base moves.
        # Only during the approach (before impact); freeze after contact.
        if live_correction and now >= next_base_ik and elapsed < cand.strike_time_s:
            T_W_B = read_rigid_body_pose_W(r, cal.base_rigid_body_id)
            if T_W_B is not None:
                R_W_A_live, t_W_A_live = compute_T_W_A_from_base_offset(T_W_B, cal)
                t_A_strike_live = R_W_A_live.T @ (cand.strike_W - t_W_A_live)
                q_new, err, _ = ik_solve(
                    chain, t_A_strike_live, q_strike_live, offset_link7,
                    R_A_link7_target=R_A_link7_home, w_ori=w_ori,
                )
                if _joints_ok(q_new) and err < ik_tol_m * 2.0:
                    q_strike_live = q_new
            next_base_ik = now + base_ik_period

        if elapsed <= cand.strike_time_s:
            q_goal, qdot_goal = _hermite(q_start, zeros, q_strike_live, cand.v_strike,
                                         elapsed, cand.strike_time_s)
        else:
            q_goal, qdot_goal = _hermite(q_strike_live, cand.v_strike, cand.q_follow, zeros,
                                         elapsed - cand.strike_time_s, follow_s)
        qdot_cmd_peak = np.maximum(qdot_cmd_peak, np.abs(np.degrees(qdot_goal)))

        if now >= next_publish:
            set_vec(r, GOAL_JOINTS, q_goal)
            next_publish = now + period_s

        if (not strike_sampled) and elapsed >= cand.strike_time_s:
            q_hit = get_vec(r, SENSOR_JOINTS, 7)
            if q_hit is not None:
                strike_err_deg = float(np.max(np.abs(np.degrees(cand.q_strike - q_hit))))
            strike_sampled = True

        if now >= next_sample:
            q = get_vec(r, SENSOR_JOINTS, 7)
            dq = get_vec(r, SENSOR_JOINT_VELS, 7)
            if dq is not None:
                qdot_peak = np.maximum(qdot_peak, np.abs(np.degrees(dq)))
            elif q is not None and prev_q is not None and prev_t is not None:
                dt = max(1e-6, now - prev_t)
                qdot_peak = np.maximum(qdot_peak, np.abs(np.degrees((q - prev_q) / dt)))
            if q is not None:
                prev_q = q.copy()
                prev_t = now
            next_sample = now + sample_period_s

        if elapsed >= total_s:
            break
        time.sleep(0.002)

    q_final = get_vec(r, SENSOR_JOINTS, 7)
    if q_final is not None:
        follow_err_deg = float(np.max(np.abs(np.degrees(cand.q_follow - q_final))))
    qdot_i = int(np.argmax(qdot_peak))
    qdot_cmd_i = int(np.argmax(qdot_cmd_peak))
    qdot_max = float(qdot_peak[qdot_i])
    qdot_cmd_max = float(qdot_cmd_peak[qdot_cmd_i])
    status = "OK" if (not math.isfinite(strike_err_deg) or strike_err_deg <= 7.0) else "WARN"
    print(
        f"[J8 segment] through-strike: {status} strike_err={strike_err_deg:.2f}° "
        f"follow_err={follow_err_deg:.2f}° qdot_max={qdot_max:.1f}°/s@q{qdot_i+1} "
        f"cmd_peak={qdot_cmd_max:.1f}°/s@q{qdot_cmd_i+1}"
    )
    return {
        "strike_err_deg": strike_err_deg,
        "follow_err_deg": follow_err_deg,
        "qdot_max_deg_s": qdot_max,
        "qdot_max_joint": qdot_i + 1,
        "cmd_qdot_max_deg_s": qdot_cmd_max,
        "cmd_qdot_max_joint": qdot_cmd_i + 1,
    }



def _make_candidate(
    *,
    chain,
    q_cur: np.ndarray,
    q_seed: np.ndarray,
    intercept: Intercept,
    R_W_A: np.ndarray,
    t_W_A: np.ndarray,
    offset_link7: np.ndarray,
    R_A_link7_home: np.ndarray | None,
    strike_plane_x_world: float,
    pre_offset_m: float,
    follow_offset_m: float,
    follow_up_offset_m: float,
    through_q6_deg: float,
    follow_s: float,
    contact_margin_s: float,
    timing_guard_s: float,
    vel_cap: np.ndarray,
    reach_m: float,
    z_min: float,
    z_max: float,
    world_z_min_m: float,
    world_z_max_m: float,
    strike_arm_x_min_m: float,
    strike_arm_x_max_m: float,
    strike_arm_y_abs_max_m: float,
    strike_arm_z_min_m: float,
    strike_arm_z_max_m: float,
    ik_tol_m: float,
    max_ori_err_deg: float,
    w_ori: float,
    z_mode: str,
    fixed_arm_z_m: float,
    impact_vel_frac: float,
) -> tuple[StrikeCandidate | None, str]:
    del strike_plane_x_world
    t_W_strike = np.asarray(intercept.position, dtype=float)
    tti = float(intercept.time_to_impact)
    if tti <= 0.0:
        return None, f"past impact tti={tti:.3f}s"
    if not (world_z_min_m <= t_W_strike[2] <= world_z_max_m):
        return None, f"strike_W.z={t_W_strike[2]:+.3f} outside [{world_z_min_m:.2f},{world_z_max_m:.2f}]"

    t_A_strike_raw = R_W_A.T @ (t_W_strike - t_W_A)
    if z_mode == "fixed-arm":
        t_A_strike_raw[2] = fixed_arm_z_m

    strike_box_reason = _strike_box_reject_reason(
        t_A_strike_raw,
        x_min_m=strike_arm_x_min_m,
        x_max_m=strike_arm_x_max_m,
        y_abs_max_m=strike_arm_y_abs_max_m,
        z_min_m=strike_arm_z_min_m,
        z_max_m=strike_arm_z_max_m,
    )
    if strike_box_reason is not None:
        return None, strike_box_reason

    t_A_pre_raw = t_A_strike_raw - np.array([pre_offset_m, 0.0, 0.0])
    t_A_follow_raw = t_A_strike_raw + np.array([follow_offset_m, 0.0, follow_up_offset_m])
    t_A_pre, pre_clipped = _safe_target(t_A_pre_raw, reach_m, z_min, z_max)
    t_A_strike, st_clipped = _safe_target(t_A_strike_raw, reach_m, z_min, z_max)
    t_A_follow, fw_clipped = _safe_target(t_A_follow_raw, reach_m, z_min, z_max)
    if pre_clipped or st_clipped:
        clips = []
        if pre_clipped:
            clips.append("pre")
        if st_clipped:
            clips.append("strike")
        return None, "clipped target(s) " + ",".join(clips)

    q_pre, err_pre, ori_pre = ik_solve(
        chain, t_A_pre, q_seed, offset_link7,
        R_A_link7_target=R_A_link7_home, w_ori=w_ori,
    )
    if (not _joints_ok(q_pre)) or (not _ik_ok(err_pre, ori_pre, ik_tol_m, max_ori_err_deg, R_A_link7_home)):
        return None, f"pre IK pos={err_pre*1000:.1f}mm ori={ori_pre:.1f}°"

    q_strike, err_st, ori_st = ik_solve(
        chain, t_A_strike, q_pre, offset_link7,
        R_A_link7_target=R_A_link7_home, w_ori=w_ori,
    )
    if (not _joints_ok(q_strike)) or (not _ik_ok(err_st, ori_st, ik_tol_m, max_ori_err_deg, R_A_link7_home)):
        return None, f"strike IK pos={err_st*1000:.1f}mm ori={ori_st:.1f}°"

    q_follow_cart, err_fw, ori_fw = ik_solve(
        chain, t_A_follow, q_strike, offset_link7,
        R_A_link7_target=R_A_link7_home, w_ori=w_ori,
    )
    if fw_clipped or (not _joints_ok(q_follow_cart)) or err_fw > ik_tol_m * 4.0:
        q_follow_cart = q_strike.copy()
        err_fw = 0.0

    q_follow = q_follow_cart.copy()
    q6_requested = math.radians(through_q6_deg)
    q6_target = float(np.clip(q_follow[5] + q6_requested, Q_LO[5] + 0.02, Q_HI[5] - 0.02))
    q6_actual_deg = math.degrees(q6_target - q_follow[5])
    q_follow[5] = q6_target
    if not _joints_ok(q_follow):
        return None, "follow joint limit"

    v_strike = np.zeros(7)
    if follow_s > 1e-3:
        v_strike = (q_follow - q_strike) / follow_s
    v_cap = np.maximum(vel_cap * max(0.05, impact_vel_frac), 1e-6)
    v_strike = np.clip(v_strike, -v_cap, v_cap)

    strike_time_s = tti + contact_margin_s - timing_guard_s
    if strike_time_s <= 0.035:
        return None, f"too late: strike_time={strike_time_s:.3f}s"

    min_s = _minimum_feasible_strike_time(q_cur, q_strike, q_follow, v_strike, follow_s, vel_cap)
    peak_frac = _trajectory_peak_frac(q_cur, q_strike, q_follow, v_strike, strike_time_s, follow_s, vel_cap)
    margin = strike_time_s - min_s
    cand = StrikeCandidate(
        strike_W=t_W_strike.copy(),
        strike_A=t_A_strike.copy(),
        pre_A=t_A_pre.copy(),
        follow_A=t_A_follow.copy(),
        q_pre=q_pre,
        q_strike=q_strike,
        q_follow=q_follow,
        v_strike=v_strike,
        tti_s=tti,
        observed_t=time.monotonic(),
        min_strike_s=min_s,
        strike_time_s=strike_time_s,
        budget_margin_s=margin,
        cmd_peak_frac=peak_frac,
        ik_err_mm=err_st * 1000.0,
        ori_err_deg=ori_st,
        follow_ik_err_mm=err_fw * 1000.0,
        q6_follow_deg=q6_actual_deg,
        follow_duration_s=follow_s,
    )
    return cand, "ok"



def _make_candidate_fast(
    *,
    chain,
    q_cur: np.ndarray,
    q_seed: np.ndarray,
    intercept: Intercept,
    R_W_A: np.ndarray,
    t_W_A: np.ndarray,
    offset_link7: np.ndarray,
    R_A_link7_home: np.ndarray | None,
    strike_plane_x_world: float,
    pre_offset_m: float,
    follow_offset_m: float,
    follow_up_offset_m: float,
    through_q6_deg: float,
    follow_s: float,
    contact_margin_s: float,
    timing_guard_s: float,
    vel_cap: np.ndarray,
    reach_m: float,
    z_min: float,
    z_max: float,
    world_z_min_m: float,
    world_z_max_m: float,
    strike_arm_x_min_m: float,
    strike_arm_x_max_m: float,
    strike_arm_y_abs_max_m: float,
    strike_arm_z_min_m: float,
    strike_arm_z_max_m: float,
    ik_tol_m: float,
    max_ori_err_deg: float,
    w_ori: float,
    z_mode: str,
    fixed_arm_z_m: float,
    impact_vel_frac: float,
) -> tuple[StrikeCandidate | None, str]:
    # Fast/simple hot path: one strike IK only. The old J8 path solved pre,
    # strike, follow, plus adaptive variants; in live traces that consumed the
    # entire ball flight. This version treats the strike point as the tracking
    # target and builds follow-through in joint space.
    del strike_plane_x_world, pre_offset_m, follow_offset_m, follow_up_offset_m
    t_W_strike = np.asarray(intercept.position, dtype=float)
    tti = float(intercept.time_to_impact)
    if tti <= 0.0:
        return None, f"past impact tti={tti:.3f}s"
    if not (world_z_min_m <= t_W_strike[2] <= world_z_max_m):
        return None, f"strike_W.z={t_W_strike[2]:+.3f} outside [{world_z_min_m:.2f},{world_z_max_m:.2f}]"

    t_A_strike_raw = R_W_A.T @ (t_W_strike - t_W_A)
    if z_mode == "fixed-arm":
        t_A_strike_raw[2] = fixed_arm_z_m

    strike_box_reason = _strike_box_reject_reason(
        t_A_strike_raw,
        x_min_m=strike_arm_x_min_m,
        x_max_m=strike_arm_x_max_m,
        y_abs_max_m=strike_arm_y_abs_max_m,
        z_min_m=strike_arm_z_min_m,
        z_max_m=strike_arm_z_max_m,
    )
    if strike_box_reason is not None:
        return None, strike_box_reason

    t_A_strike, st_clipped = _safe_target(t_A_strike_raw, reach_m, z_min, z_max)
    if st_clipped:
        return None, "clipped target(s) strike"

    q_strike, err_st, ori_st = ik_solve(
        chain, t_A_strike, q_seed, offset_link7,
        R_A_link7_target=R_A_link7_home, w_ori=w_ori,
    )
    if (not _joints_ok(q_strike)) or (not _ik_ok(err_st, ori_st, ik_tol_m, max_ori_err_deg, R_A_link7_home)):
        return None, f"strike IK pos={err_st*1000:.1f}mm ori={ori_st:.1f}°"

    q_pre = q_strike.copy()
    q_follow = q_strike.copy()
    q6_requested = math.radians(through_q6_deg)
    q6_target = float(np.clip(q_follow[5] + q6_requested, Q_LO[5] + 0.02, Q_HI[5] - 0.02))
    q6_actual_deg = math.degrees(q6_target - q_follow[5])
    q_follow[5] = q6_target
    if not _joints_ok(q_follow):
        return None, "follow joint limit"

    v_strike = np.zeros(7)
    if follow_s > 1e-3:
        v_strike = (q_follow - q_strike) / follow_s
    v_cap = np.maximum(vel_cap * max(0.05, impact_vel_frac), 1e-6)
    v_strike = np.clip(v_strike, -v_cap, v_cap)

    strike_time_s = tti + contact_margin_s - timing_guard_s
    if strike_time_s <= 0.035:
        return None, f"too late: strike_time={strike_time_s:.3f}s"

    min_s = _minimum_feasible_strike_time(q_cur, q_strike, q_follow, v_strike, follow_s, vel_cap)
    peak_frac = _trajectory_peak_frac(q_cur, q_strike, q_follow, v_strike, strike_time_s, follow_s, vel_cap)
    margin = strike_time_s - min_s
    return StrikeCandidate(
        strike_W=t_W_strike.copy(),
        strike_A=t_A_strike.copy(),
        pre_A=t_A_strike.copy(),
        follow_A=t_A_strike.copy(),
        q_pre=q_pre,
        q_strike=q_strike,
        q_follow=q_follow,
        v_strike=v_strike,
        tti_s=tti,
        observed_t=time.monotonic(),
        min_strike_s=min_s,
        strike_time_s=strike_time_s,
        budget_margin_s=margin,
        cmd_peak_frac=peak_frac,
        ik_err_mm=err_st * 1000.0,
        ori_err_deg=ori_st,
        follow_ik_err_mm=0.0,
        q6_follow_deg=q6_actual_deg,
        follow_duration_s=follow_s,
    ), "ok"

def _retarget_ok(new: StrikeCandidate, old: StrikeCandidate | None,
                 max_target_jump_m: float, max_joint_jump_deg: float) -> tuple[bool, str]:
    if old is None:
        return True, "first"
    target_jump = float(np.linalg.norm(new.strike_A - old.strike_A))
    q_jump, q_joint = _max_abs_deg_with_joint(new.q_strike - old.q_strike)
    if target_jump > max_target_jump_m:
        return False, f"target_jump={target_jump*100:.1f}cm"
    if q_jump > max_joint_jump_deg:
        return False, f"q_jump={q_jump:.1f}°@q{q_joint}"
    return True, f"retarget {target_jump*100:.1f}cm {q_jump:.1f}°"




def _candidate_runtime_feasible(cand: StrikeCandidate, q_cur: np.ndarray,
                                contact_margin_s: float, timing_guard_s: float,
                                follow_s: float, vel_cap: np.ndarray) -> tuple[bool, float, float]:
    strike_time_s = cand.aged_tti() + contact_margin_s - timing_guard_s
    margin_s = strike_time_s - cand.min_strike_s
    cmd_frac = _trajectory_peak_frac(
        q_cur, cand.q_strike, cand.q_follow, cand.v_strike,
        max(0.035, strike_time_s), follow_s, vel_cap,
    )
    feasible = strike_time_s > 0.035 and margin_s >= 0.0 and cmd_frac <= 1.0
    return feasible, margin_s, cmd_frac


def _choose_more_reachable_candidate(candidates: list[StrikeCandidate]) -> StrikeCandidate:
    # Prefer positive timing margin, then lower command peak, then shorter minimum time.
    return max(
        candidates,
        key=lambda c: (c.budget_margin_s, -c.cmd_peak_frac, -c.min_strike_s),
    )

def run_loop(
    *,
    r: redis.Redis,
    chain,
    tracker: BallTracker | None,
    q_home_rad: np.ndarray,
    offset_link7: np.ndarray,
    R_A_link7_home: np.ndarray | None,
    cal: ArmBaseOffsetCalibration,
    strike_plane_x_world: float,
    pre_offset_m: float,
    follow_offset_m: float,
    follow_up_offset_m: float,
    through_q6_deg: float,
    follow_s: float,
    commit_tti: float,
    launch_margin_s: float,
    contact_margin_s: float,
    timing_guard_s: float,
    return_s: float,
    return_vel_frac: float,
    rate_hz: float,
    no_commit: bool,
    reach_m: float,
    z_min: float,
    z_max: float,
    world_z_min_m: float,
    world_z_max_m: float,
    target_jump_max_m: float,
    retarget_joint_jump_deg: float,
    tracking_step_deg: float,
    tracking_accel_deg_s2: float,
    swing_vel_frac: float,
    ik_tol_m: float,
    z_mode: str,
    fixed_arm_z_m: float,
    w_ori: float,
    max_ori_err_deg: float,
    strike_arm_x_min_m: float,
    strike_arm_x_max_m: float,
    strike_arm_y_abs_max_m: float,
    strike_arm_z_min_m: float,
    strike_arm_z_max_m: float,
    impact_vel_frac: float,
    adaptive_trajectory: bool,
    relaxed_ori_w_scale: float,
    relaxed_ori_extra_deg: float,
    adaptive_q6_deg: float,
    adaptive_follow_s: float,
    simple_fast: bool,
    try_late_slack_s: float,
    mock_intercepts: list[np.ndarray] | None,
    mock_tti: float,
    mock_cycle_s: float,
    mock_identity_base: bool,
    full_segment_logs: bool,
    verbose_tracking: bool,
    log_period_s: float,
    max_swings: int,
    post_impact_idle_s: float,
    lock_release_after_impact_s: float,
    xml_vel_limits: np.ndarray,
    ready_label: str,
    ready_arm_A: np.ndarray | None,
    trace: _TraceLogger | None,
    ball_key: str | None,
    base_goal_key: str,
    base_ready_pose: tuple[float, float, float] | None,
    base_y_min: float,
    base_y_max: float,
    base_thrust_m: float,
    hold_base_after_lost_s: float,
) -> None:
    dt = 1.0 / max(1.0, rate_hz)
    using_mock = mock_intercepts is not None and len(mock_intercepts) > 0
    mock_t0 = time.monotonic()
    last_print_t = float("-inf")
    last_reject_print_t = float("-inf")
    reject_counts: dict[str, int] = {}
    active_cand: StrikeCandidate | None = None
    active_throw_id = 0
    swings_done = 0
    last_q_cmd: np.ndarray | None = get_vec(r, SENSOR_JOINTS, 7)
    last_q_cmd_vel = np.zeros(7)
    last_goal_write_t = time.perf_counter()
    idle_until_t = 0.0
    home_retry_after_t = 0.0
    vel_cap = np.maximum(xml_vel_limits * max(0.05, swing_vel_frac), 1e-6)
    tick_i = 0
    max_goal_gap_s = max(0.03, 2.5 * dt)
    last_base_good_t = float("-inf")
    base_at_ready = base_ready_pose is None
    if base_ready_pose is not None:
        r.set(base_goal_key, json.dumps(list(base_ready_pose)))

    def tr(event: str, **fields) -> None:
        if trace is not None:
            trace.write(event, **fields)

    def print_reject(reason: str) -> None:
        nonlocal last_reject_print_t
        reject_counts[reason] = reject_counts.get(reason, 0) + 1
        now = time.perf_counter()
        tr("reject", tick=tick_i, reason=reason, count=reject_counts.get(reason, 0))
        if now - last_reject_print_t >= 0.40:
            last_reject_print_t = now
            print(f"[J8 wait] {reason}")

    def recover_home(label: str, idle_after: bool) -> bool:
        nonlocal active_cand, last_q_cmd, last_q_cmd_vel, last_goal_write_t
        nonlocal idle_until_t, home_retry_after_t
        tr("recover_home_start", tick=tick_i, label=label, idle_after=idle_after, active_candidate=_candidate_snapshot(active_cand))
        ok = _return_home_blocking(
            r, q_home_rad,
            move_s=return_s,
            vel_frac=return_vel_frac,
            label=label,
            hold_s=0.35,
            publish_hz=100.0,
            full_segment_logs=full_segment_logs,
        )
        active_cand = None
        if ok:
            home_retry_after_t = 0.0
            last_q_cmd = get_vec(r, SENSOR_JOINTS, 7)
        else:
            last_q_cmd = _hold_current_joints(r)
            home_retry_after_t = time.monotonic() + 5.0
        if last_q_cmd is None:
            last_q_cmd = get_vec(r, SENSOR_JOINTS, 7)
        last_q_cmd_vel = np.zeros(7)
        last_goal_write_t = time.perf_counter()
        if idle_after:
            idle_until_t = time.monotonic() + post_impact_idle_s
        tr("recover_home_done", tick=tick_i, label=label, ok=ok, idle_until_t=idle_until_t, home_retry_after_t=home_retry_after_t)
        return ok

    def publish_limited_goal(target: np.ndarray, q_cur: np.ndarray, mode: str) -> None:
        nonlocal last_q_cmd, last_q_cmd_vel, last_goal_write_t
        now_goal_t = time.perf_counter()
        raw_elapsed_s = max(0.0, now_goal_t - last_goal_write_t)
        stale_goal = last_q_cmd is None or raw_elapsed_s > max_goal_gap_s
        cmd_state_error_deg = (
            float(np.max(np.abs(np.degrees(last_q_cmd - q_cur))))
            if last_q_cmd is not None else float("inf")
        )
        measured_state_reset = stale_goal or cmd_state_error_deg > max(2.0 * tracking_step_deg, 3.0)
        if measured_state_reset:
            # If the arm lags behind the last commanded goal, restart from
            # measured q. This avoids post-reflex/home-failure command jumps.
            q_rate_base = q_cur
            last_q_cmd_vel = np.zeros(7)
            elapsed_goal_s = dt
        else:
            q_rate_base = last_q_cmd
            elapsed_goal_s = max(dt, min(raw_elapsed_s, max_goal_gap_s))
        if tracking_accel_deg_s2 > 0.0:
            q_cmd, last_q_cmd_vel = _rate_accel_limited_goal(
                target, q_rate_base, last_q_cmd_vel,
                tracking_step_deg * rate_hz, tracking_accel_deg_s2, elapsed_goal_s,
            )
        else:
            q_cmd = _rate_limited_goal(
                target, q_rate_base, tracking_step_deg * max(1.0, elapsed_goal_s * rate_hz),
            )
        set_vec(r, GOAL_JOINTS, q_cmd)
        tr(
            "goal_write",
            tick=tick_i, target_mode=mode, stale_goal_reset=stale_goal,
            measured_state_reset=measured_state_reset,
            cmd_state_error_deg=cmd_state_error_deg,
            raw_elapsed_goal_s=raw_elapsed_s, elapsed_goal_s=elapsed_goal_s,
            q_target_deg=np.degrees(target), q_cmd_deg=np.degrees(q_cmd),
            q_cur_deg=np.degrees(q_cur), last_q_cmd_vel_deg_s=np.degrees(last_q_cmd_vel),
        )
        last_q_cmd = q_cmd.copy()
        last_goal_write_t = now_goal_t

    switch_ctrl(r, JOINT_CTRL)
    _hold_current_joints(r)

    print(f"[J8] running at {rate_hz:.0f} Hz  -- through-strike mode")
    print(
        f"[J8] strike_plane_x={strike_plane_x_world:+.3f}m commit_tti={commit_tti:.3f}s "
        f"launch_margin={launch_margin_s:.3f}s guard={timing_guard_s:.3f}s "
        f"contact_margin={contact_margin_s:+.3f}s"
    )
    print(
        f"[J8] pre={pre_offset_m*100:.1f}cm follow=+{follow_offset_m*100:.1f}cm/"
        f"+{follow_up_offset_m*100:.1f}cm q6_follow={through_q6_deg:+.1f}° follow_s={follow_s:.2f}s"
    )
    print(
        f"[J8] commit rule: valid geometry + IK + sampled trajectory peak <= "
        f"{swing_vel_frac*100:.0f}% XML before impact"
    )
    if simple_fast:
        print(f"[J8] simple-fast: one strike IK per tick, no adaptive IK variants, late-try slack={try_late_slack_s:.3f}s")
    elif adaptive_trajectory:
        print(
            f"[J8] adaptive: fallback q6={adaptive_q6_deg:+.1f}° "
            f"follow_s={adaptive_follow_s:.2f}s ori_w_scale={relaxed_ori_w_scale:.2f} "
            f"ori_extra={relaxed_ori_extra_deg:.1f}°"
        )
    print(
        f"[J8] strike box A: x=[{strike_arm_x_min_m:+.2f},{strike_arm_x_max_m:+.2f}] "
        f"|y|<={strike_arm_y_abs_max_m:.2f} z=[{strike_arm_z_min_m:+.2f},{strike_arm_z_max_m:+.2f}] "
        f"z_mode={z_mode}" + (f"({fixed_arm_z_m:.2f}m)" if z_mode == "fixed-arm" else "")
    )
    if ready_arm_A is not None:
        print(f"[J8] ready pose '{ready_label}' targets arm_A={_fmt_v3(ready_arm_A)}")
    else:
        print(f"[J8] ready pose '{ready_label}' uses imported Q_HOME_RAD")
    if base_ready_pose is not None:
        print(
            f"[J8] base tracking ON: ready=({base_ready_pose[0]:+.3f},{base_ready_pose[1]:+.3f},"
            f"{math.degrees(base_ready_pose[2]):+.1f}°) "
            f"y∈[{base_y_min:.2f},{base_y_max:.2f}] thrust={base_thrust_m:.2f}m"
        )
    print()

    while True:
        tick_i += 1
        loop_t = time.perf_counter()
        raw_ball = r.get(ball_key) if ball_key else None
        raw_ball_parsed = _parse_raw_json(raw_ball)
        switch_ctrl(r, JOINT_CTRL)

        if using_mock:
            idx = 0
            if mock_cycle_s > 0.0 and len(mock_intercepts or []) > 1:
                idx = int((time.monotonic() - mock_t0) / mock_cycle_s) % len(mock_intercepts or [])
            t_W_strike = np.asarray((mock_intercepts or [np.zeros(3)])[idx], dtype=float)
            intercept = Intercept(t_W_strike.copy(), np.array([-5.0, 0.0, 0.0]), float(mock_tti), 0)
        else:
            assert tracker is not None
            sample = tracker.update()
            intercept = tracker.predict_intercept(strike_plane_x_world)
            reason = getattr(tracker, "last_reject_reason", "") or "no_prediction"
            tr(
                "tracker_tick",
                tick=tick_i,
                raw_ball=raw_ball_parsed,
                sample_pos_W=None if sample is None else sample.pos,
                sample_t=None if sample is None else sample.t,
                tracker=_tracker_snapshot(tracker),
                reject_reason=None if intercept is not None else reason,
                intercept=_intercept_snapshot(intercept),
                active_candidate=_candidate_snapshot(active_cand),
            )
            if intercept is None:
                print_reject(reason)

        if mock_identity_base:
            R_W_A = np.eye(3)
            t_W_A = np.zeros(3)
        else:
            T_W_B = read_rigid_body_pose_W(r, cal.base_rigid_body_id)
            if T_W_B is None:
                print_reject(f"cart rigid body {cal.base_rigid_body_id} not visible")
                tr("cart_missing", tick=tick_i, rigid_body_id=cal.base_rigid_body_id)
                time.sleep(max(0.0, dt - (time.perf_counter() - loop_t)))
                continue
            R_W_A, t_W_A = compute_T_W_A_from_base_offset(T_W_B, cal)

        q_cur = get_vec(r, SENSOR_JOINTS, 7)
        if q_cur is None:
            print_reject("no joint sensor")
            tr("no_joint_sensor", tick=tick_i)
            time.sleep(max(0.0, dt - (time.perf_counter() - loop_t)))
            continue

        if time.monotonic() < idle_until_t:
            publish_limited_goal(q_home_rad, q_cur, "post_impact_ready")
            tr("post_impact_idle", tick=tick_i, q_cur_deg=np.degrees(q_cur), idle_until_t=idle_until_t)
            time.sleep(max(0.0, dt - (time.perf_counter() - loop_t)))
            continue

        if intercept is None and active_cand is not None:
            aged = active_cand.aged_tti()
            if aged > 0.02 and active_cand.budget_margin_s > -0.20 and active_cand.cmd_peak_frac < 1.50:
                intercept = Intercept(active_cand.strike_W.copy(), np.zeros(3), aged, 0)
                tr("prediction_gap_keep_active", tick=tick_i, aged_tti_s=aged, synthetic_intercept=_intercept_snapshot(intercept))
            else:
                tr("prediction_gap_expired", tick=tick_i, aged_tti_s=aged, active_candidate=_candidate_snapshot(active_cand))
                active_cand = None

        if intercept is None:
            home_err = _home_error_deg(r, q_home_rad)
            tr("no_intercept_idle", tick=tick_i, home_err_deg=home_err, q_cur_deg=np.degrees(q_cur), raw_ball=raw_ball_parsed)
            publish_limited_goal(q_home_rad, q_cur, "idle_ready")
            if base_ready_pose is not None and not base_at_ready:
                if time.monotonic() - last_base_good_t > hold_base_after_lost_s:
                    r.set(base_goal_key, json.dumps(list(base_ready_pose)))
                    base_at_ready = True
            time.sleep(max(0.0, dt - (time.perf_counter() - loop_t)))
            continue

        q_seed = active_cand.q_strike if active_cand is not None else q_cur

        def build_candidate(q6_deg: float, cand_follow_s: float, cand_w_ori: float,
                            cand_max_ori_err_deg: float) -> tuple[StrikeCandidate | None, str]:
            maker = _make_candidate_fast if simple_fast else _make_candidate
            return maker(
                chain=chain,
                q_cur=q_cur,
                q_seed=q_seed,
                intercept=intercept,
                R_W_A=R_W_A,
                t_W_A=t_W_A,
                offset_link7=offset_link7,
                R_A_link7_home=R_A_link7_home,
                strike_plane_x_world=strike_plane_x_world,
                pre_offset_m=pre_offset_m,
                follow_offset_m=follow_offset_m,
                follow_up_offset_m=follow_up_offset_m,
                through_q6_deg=q6_deg,
                follow_s=cand_follow_s,
                contact_margin_s=contact_margin_s,
                timing_guard_s=timing_guard_s,
                vel_cap=vel_cap,
                reach_m=reach_m,
                z_min=z_min,
                z_max=z_max,
                world_z_min_m=world_z_min_m,
                world_z_max_m=world_z_max_m,
                strike_arm_x_min_m=strike_arm_x_min_m,
                strike_arm_x_max_m=strike_arm_x_max_m,
                strike_arm_y_abs_max_m=strike_arm_y_abs_max_m,
                strike_arm_z_min_m=strike_arm_z_min_m,
                strike_arm_z_max_m=strike_arm_z_max_m,
                ik_tol_m=ik_tol_m,
                max_ori_err_deg=cand_max_ori_err_deg,
                w_ori=cand_w_ori,
                z_mode=z_mode,
                fixed_arm_z_m=fixed_arm_z_m,
                impact_vel_frac=impact_vel_frac,
            )

        cand, reason = build_candidate(through_q6_deg, follow_s, w_ori, max_ori_err_deg)
        tr("candidate_base", tick=tick_i, reason=reason, candidate=_candidate_snapshot(cand), intercept=_intercept_snapshot(intercept))
        variant_txt = "base"
        candidates = [] if cand is None else [cand]
        if (not simple_fast) and adaptive_trajectory and (cand is None or cand.budget_margin_s < 0.08 or cand.cmd_peak_frac > 0.92):
            q6_options = [adaptive_q6_deg]
            if through_q6_deg > adaptive_q6_deg + 1.0:
                q6_options.append(0.5 * (through_q6_deg + adaptive_q6_deg))
            follow_options = [adaptive_follow_s] if adaptive_follow_s > 0.0 else []
            if follow_s not in follow_options:
                follow_options.append(follow_s)
            for q6_opt in q6_options:
                for follow_opt in follow_options:
                    for ori_scale, ori_extra in ((1.0, 0.0), (relaxed_ori_w_scale, relaxed_ori_extra_deg)):
                        cand_i, _reason_i = build_candidate(
                            q6_opt, follow_opt,
                            max(0.0, w_ori * ori_scale),
                            max_ori_err_deg + max(0.0, ori_extra),
                        )
                        tr(
                            "candidate_variant",
                            tick=tick_i, q6_deg=q6_opt, follow_s=follow_opt,
                            ori_scale=ori_scale, ori_extra_deg=ori_extra,
                            reason=_reason_i, candidate=_candidate_snapshot(cand_i),
                        )
                        if cand_i is not None:
                            candidates.append(cand_i)
        if candidates:
            cand = _choose_more_reachable_candidate(candidates)
            if abs(cand.q6_follow_deg - through_q6_deg) > 0.5 or abs(cand.follow_duration_s - follow_s) > 1e-3:
                variant_txt = f"adapt q6={cand.q6_follow_deg:+.0f}° follow_s={cand.follow_duration_s:.2f}"
        if cand is None:
            print_reject(reason)
            tr("candidate_none", tick=tick_i, reason=reason, q_cur_deg=np.degrees(q_cur), intercept=_intercept_snapshot(intercept))
            time.sleep(max(0.0, dt - (time.perf_counter() - loop_t)))
            continue

        ok_retarget, retarget_reason = _retarget_ok(
            cand, active_cand, target_jump_max_m, retarget_joint_jump_deg,
        )
        old_feasible = False
        old_margin = float("-inf")
        if active_cand is not None:
            old_feasible, old_margin, _old_cmd = _candidate_runtime_feasible(
                active_cand, q_cur, contact_margin_s, timing_guard_s, follow_s, vel_cap,
            )
        replace_stale = (
            active_cand is not None
            and not old_feasible
            and cand.budget_margin_s > old_margin + 0.04
        )
        if ok_retarget or active_cand is None or active_cand.aged_tti() <= 0.0 or replace_stale:
            active_cand = cand
            retarget_txt = retarget_reason if not replace_stale else f"replace_stale {retarget_reason}"
        else:
            retarget_txt = "keep_prev " + retarget_reason
        if variant_txt != "base":
            retarget_txt += " " + variant_txt
        tr(
            "retarget_decision",
            tick=tick_i, ok_retarget=ok_retarget, retarget_reason=retarget_reason,
            replace_stale=replace_stale, old_feasible=old_feasible, old_margin_s=old_margin,
            chosen_text=retarget_txt, chosen_candidate=_candidate_snapshot(active_cand),
            new_candidate=_candidate_snapshot(cand),
        )

        cand_use = active_cand
        assert cand_use is not None
        aged_tti = cand_use.aged_tti()
        strike_time_now = aged_tti + contact_margin_s - timing_guard_s
        cand_use.strike_time_s = strike_time_now
        cand_use.budget_margin_s = strike_time_now - cand_use.min_strike_s
        cand_use.cmd_peak_frac = _trajectory_peak_frac(
            q_cur, cand_use.q_strike, cand_use.q_follow, cand_use.v_strike,
            max(0.035, strike_time_now), cand_use.follow_duration_s, vel_cap,
        )

        feasible = (
            strike_time_now > 0.035
            and cand_use.budget_margin_s >= 0.0
            and cand_use.cmd_peak_frac <= 1.0
        )
        peak_trace, peak_idx_trace, peak_frac_trace = _trajectory_peak_profile(
            q_cur, cand_use.q_strike, cand_use.q_follow, cand_use.v_strike,
            max(0.035, strike_time_now), cand_use.follow_duration_s, vel_cap,
        )
        q_delta_trace, q_joint_trace = _max_abs_deg_with_joint(cand_use.q_strike - q_cur)
        tr(
            "candidate_runtime",
            tick=tick_i, feasible=feasible, strike_time_s=strike_time_now,
            aged_tti_s=aged_tti, margin_s=cand_use.budget_margin_s,
            cmd_peak_frac=cand_use.cmd_peak_frac, bottleneck_joint=peak_idx_trace + 1,
            bottleneck_cmd_deg_s=math.degrees(peak_trace[peak_idx_trace]),
            q_to_strike_deg=q_delta_trace, q_to_strike_joint=q_joint_trace,
            q_cur_deg=np.degrees(q_cur), active_candidate=_candidate_snapshot(cand_use),
        )

        trackable = (
            aged_tti > 0.02
            and cand_use.cmd_peak_frac < (2.00 if simple_fast else 1.50)
            and cand_use.budget_margin_s > (-(try_late_slack_s + 0.10) if simple_fast else -0.25)
        )
        if not trackable:
            tr(
                "candidate_not_tracked", tick=tick_i, aged_tti_s=aged_tti,
                margin_s=cand_use.budget_margin_s, cmd_peak_frac=cand_use.cmd_peak_frac,
                active_candidate=_candidate_snapshot(cand_use),
            )
            active_cand = None
            publish_limited_goal(q_home_rad, q_cur, "reject_ready")
            if base_ready_pose is not None and not base_at_ready:
                if time.monotonic() - last_base_good_t > hold_base_after_lost_s:
                    r.set(base_goal_key, json.dumps(list(base_ready_pose)))
                    base_at_ready = True
            time.sleep(max(0.0, dt - (time.perf_counter() - loop_t)))
            continue

        if loop_t - last_print_t >= max(0.05, log_period_s):
            last_print_t = loop_t
            launch_ready = feasible and aged_tti <= commit_tti and cand_use.budget_margin_s <= launch_margin_s
            state = "SOFT" if aged_tti <= commit_tti else "PREVIEW"
            if feasible and aged_tti <= commit_tti:
                state = "LAUNCH" if launch_ready else "AIMING"
            print(
                f"[J8] {state:7s} tti={aged_tti:+.3f}s strike_W={_fmt_v3(cand_use.strike_W)} "
                f"strike_A={_fmt_v3(cand_use.strike_A)} Tmin={cand_use.min_strike_s:.3f}s "
                f"strike_s={strike_time_now:.3f}s margin={cand_use.budget_margin_s:+.3f}s "
                f"cmd={cand_use.cmd_peak_frac*100:.0f}% {retarget_txt}"
            )
            if verbose_tracking:
                q_delta, q_joint = _max_abs_deg_with_joint(cand_use.q_strike - q_cur)
                qdot = get_vec(r, SENSOR_JOINT_VELS, 7)
                qdot_txt = ""
                if qdot is not None:
                    qdot_deg, qdot_joint = _max_abs_deg_with_joint(qdot)
                    qdot_txt = f" qdot={qdot_deg:.1f}°/s@q{qdot_joint}"
                peak, peak_idx, peak_frac = _trajectory_peak_profile(
                    q_cur, cand_use.q_strike, cand_use.q_follow, cand_use.v_strike,
                    max(0.035, strike_time_now), cand_use.follow_duration_s, vel_cap,
                )
                print(
                    f"[J8]   q_to_strike={q_delta:.1f}°@q{q_joint} "
                    f"peak_cmd={math.degrees(peak[peak_idx]):.1f}°/s@q{peak_idx+1} "
                    f"({peak_frac*100:.0f}% cap) "
                    f"ik={cand_use.ik_err_mm:.1f}mm ori={cand_use.ori_err_deg:.1f}° "
                    f"rejects={_fmt_counts(reject_counts)}{qdot_txt}"
                )

        launch_ready = feasible and aged_tti <= commit_tti and cand_use.budget_margin_s <= launch_margin_s
        late_try = (
            simple_fast
            and not feasible
            and aged_tti <= commit_tti
            and strike_time_now > 0.045
            and cand_use.budget_margin_s >= -try_late_slack_s
        )
        commit_now = (not no_commit) and (launch_ready or late_try)
        if commit_now:
            if late_try and not feasible:
                cand_use.strike_time_s = max(0.08, cand_use.min_strike_s)
                cand_use.budget_margin_s = cand_use.strike_time_s - cand_use.min_strike_s
                cand_use.cmd_peak_frac = _trajectory_peak_frac(
                    q_cur, cand_use.q_strike, cand_use.q_follow, cand_use.v_strike,
                    cand_use.strike_time_s, cand_use.follow_duration_s, vel_cap,
                )
                print(
                    f"[J8] LATE_TRY: ball_time={strike_time_now:.3f}s < Tmin={cand_use.min_strike_s:.3f}s; "
                    f"swinging safely over {cand_use.strike_time_s:.3f}s"
                )
            tr("commit_start", tick=tick_i, throw_id=active_throw_id + 1, active_candidate=_candidate_snapshot(cand_use), late_try=late_try)
            active_throw_id += 1
            print("\n" + "=" * 64)
            print(
                f"[J8] COMMIT throw {active_throw_id}: tti={aged_tti:.3f}s "
                f"strike_s={strike_time_now:.3f}s Tmin={cand_use.min_strike_s:.3f}s "
                f"margin={cand_use.budget_margin_s:+.3f}s cmd_peak={cand_use.cmd_peak_frac*100:.0f}%"
            )
            print(
                f"[J8]   strike_W={_fmt_v3(cand_use.strike_W)} "
                f"strike_A={_fmt_v3(cand_use.strike_A)} "
                f"pre_A={_fmt_v3(cand_use.pre_A)} follow_A={_fmt_v3(cand_use.follow_A)}"
            )
            peak, peak_idx, peak_frac = _trajectory_peak_profile(
                q_cur, cand_use.q_strike, cand_use.q_follow, cand_use.v_strike,
                max(0.035, strike_time_now), cand_use.follow_duration_s, vel_cap,
            )
            print(
                f"[J8]   IK strike={cand_use.ik_err_mm:.1f}mm ori={cand_use.ori_err_deg:.1f}° "
                f"follow_ik={cand_use.follow_ik_err_mm:.1f}mm q6_follow={cand_use.q6_follow_deg:+.1f}°"
            )
            print(
                f"[J8]   bottleneck q{peak_idx+1}: cmd={math.degrees(peak[peak_idx]):.1f}°/s "
                f"= {peak_frac*100:.0f}% of budget"
            )
            print("=" * 64 + "\n")

            if base_ready_pose is not None and base_thrust_m > 0.0:
                arm_fwd_W = R_W_A[:, 0]
                arm_fwd_xy = arm_fwd_W[:2]
                arm_fwd_len = float(np.linalg.norm(arm_fwd_xy))
                if arm_fwd_len > 1e-6:
                    arm_fwd_xy = arm_fwd_xy / arm_fwd_len
                thrust_x = float(t_W_A[0] + arm_fwd_xy[0] * base_thrust_m)
                thrust_y = float(t_W_A[1] + arm_fwd_xy[1] * base_thrust_m)
                base_yaw_now = math.atan2(float(R_W_A[1, 0]), float(R_W_A[0, 0]))
                r.set(base_goal_key, json.dumps([thrust_x, thrust_y, base_yaw_now]))
                base_at_ready = False
                print(
                    f"[J8 base] THRUST +{base_thrust_m:.1f}m → "
                    f"({thrust_x:+.3f},{thrust_y:+.3f},{math.degrees(base_yaw_now):+.1f}°)"
                )

            result = _run_through_strike(
                r, cand_use,
                publish_hz=200.0,
                full_segment_logs=full_segment_logs,
                chain=chain,
                offset_link7=offset_link7,
                R_A_link7_home=R_A_link7_home,
                w_ori=w_ori,
                ik_tol_m=ik_tol_m,
                cal=cal if not mock_identity_base else None,
            )
            if base_ready_pose is not None:
                r.set(base_goal_key, json.dumps(list(base_ready_pose)))
                base_at_ready = True
            if _safety_tripped(r):
                print("[J8] safety torque after through-strike; recovering slowly")
            q_post = get_vec(r, SENSOR_JOINTS, 7)
            if q_post is None:
                q_post = cand_use.q_follow
            home_ok = recover_home("post-strike->home", idle_after=True)
            swings_done += 1
            outcome = "OK" if home_ok and result["strike_err_deg"] <= 7.0 else "WARN"
            print(
                f"[J8 audit] throw={active_throw_id} outcome={outcome} "
                f"timing[tti={aged_tti:.3f}s strike_s={strike_time_now:.3f}s "
                f"Tmin={cand_use.min_strike_s:.3f}s margin={cand_use.budget_margin_s:+.3f}s] "
                f"arm[strike_err={result['strike_err_deg']:.2f}° "
                f"follow_err={result['follow_err_deg']:.2f}° home_ok={home_ok}] "
                f"tracker_rejects={_fmt_counts(reject_counts)}"
            )
            tr("commit_done", tick=tick_i, throw_id=active_throw_id, outcome=outcome, result=result, home_ok=home_ok, reject_counts=reject_counts)
            reject_counts = {}
            active_cand = None
            if max_swings > 0 and swings_done >= max_swings:
                print(f"[J8] --max-swings={max_swings} reached; exiting")
                break
            time.sleep(max(0.0, dt - (time.perf_counter() - loop_t)))
            continue

        # Buy timing budget aggressively: if the current plan is behind budget,
        # move toward strike directly instead of spending travel on a pre pose.
        if cand_use.budget_margin_s < 0.08 or aged_tti <= commit_tti + 0.12:
            target = cand_use.q_strike
            target_mode = "strike"
        else:
            target = cand_use.q_pre
            target_mode = "pre"
        publish_limited_goal(target, q_cur, target_mode)

        if base_ready_pose is not None:
            y_base = float(np.clip(cand_use.strike_W[1], base_y_min, base_y_max))
            r.set(base_goal_key, json.dumps([float(base_ready_pose[0]), y_base, float(base_ready_pose[2])]))
            last_base_good_t = time.monotonic()
            base_at_ready = False

        if aged_tti < -lock_release_after_impact_s:
            tr("missed_window", tick=tick_i, aged_tti_s=aged_tti, active_candidate=_candidate_snapshot(active_cand))
            active_cand = None
            idle_until_t = time.monotonic() + post_impact_idle_s

        time.sleep(max(0.0, dt - (time.perf_counter() - loop_t)))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="J8: simplified through-strike pickleball intercept planner.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--ball-rigid-body-id", type=int, default=13)
    ap.add_argument("--strike-plane-x", type=float, default=-0.55)
    ap.add_argument("--mock-intercept", nargs=3, type=float, action="append")
    ap.add_argument("--mock-tti", type=float, default=0.60)
    ap.add_argument("--mock-cycle-s", type=float, default=0.0)
    ap.add_argument("--mock-identity-base", action="store_true")

    ap.add_argument("--pre-offset", "--wind-up-offset", dest="pre_offset", type=float, default=0.060)
    ap.add_argument("--follow-offset", type=float, default=0.040)
    ap.add_argument("--follow-up-offset", type=float, default=0.030)
    ap.add_argument("--through-q6-deg", type=float, default=24.0)
    ap.add_argument("--follow-s", type=float, default=0.28)
    ap.add_argument("--paddle-open-deg", type=float, default=6.0)

    ap.add_argument("--commit-tti", type=float, default=0.62)
    ap.add_argument("--launch-margin-s", type=float, default=0.060,
                    help="Retarget until budget_margin_s <= this value, then hard-commit. Smaller waits longer for better ball predictions.")
    ap.add_argument("--contact-margin-s", type=float, default=0.0)
    ap.add_argument("--timing-guard-s", type=float, default=0.045)
    ap.add_argument("--swing-vel-frac", type=float, default=0.85)
    ap.add_argument("--impact-vel-frac", type=float, default=0.75)
    ap.add_argument("--adaptive-trajectory", dest="adaptive_trajectory", action="store_true",
                    help="Try lower-q6/slower-follow/relaxed-orientation variants when the base plan is tight. Disabled by default in simple-fast mode.")
    ap.add_argument("--no-adaptive-trajectory", dest="adaptive_trajectory", action="store_false")
    ap.set_defaults(adaptive_trajectory=False)
    ap.add_argument("--simple-fast", dest="simple_fast", action="store_true",
                    help="Use the J8 fast hot path: one strike IK, joint-space follow, no live adaptive IK variant explosion.")
    ap.add_argument("--no-simple-fast", dest="simple_fast", action="store_false")
    ap.set_defaults(simple_fast=True)
    ap.add_argument("--try-late-slack-s", type=float, default=0.18,
                    help="In simple-fast mode, still commit as a safe late try if Tmin exceeds ball-time by up to this many seconds.")
    ap.add_argument("--adaptive-q6-deg", type=float, default=14.0,
                    help="Fallback q6 follow-through used when full through-q6 is too expensive.")
    ap.add_argument("--adaptive-follow-s", type=float, default=0.36,
                    help="Fallback follow duration used to lower through-strike velocity peaks.")
    ap.add_argument("--relaxed-ori-w-scale", type=float, default=0.25,
                    help="Fallback multiplier for orientation IK weight when timing is tight.")
    ap.add_argument("--relaxed-ori-extra-deg", type=float, default=8.0,
                    help="Extra orientation error allowed for fallback timing variants.")
    ap.add_argument("--return-s", type=float, default=4.0)
    ap.add_argument("--return-vel-frac", type=float, default=0.45)
    ap.add_argument("--home-s", type=float, default=3.0)
    ap.add_argument("--home-hold-s", type=float, default=1.0)
    ap.add_argument("--home-vel-frac", type=float, default=J6_HOME_VEL_FRAC)
    ap.add_argument("--ready-pose", choices=["home", "strike-center"], default="strike-center",
                    help="Startup/recovery posture. strike-center solves IK near the middle of the J8 strike box.")
    ap.add_argument("--ready-arm-x", type=float, default=0.28,
                    help="Arm-frame x for --ready-pose=strike-center.")
    ap.add_argument("--ready-arm-y", type=float, default=0.0,
                    help="Arm-frame y for --ready-pose=strike-center.")
    ap.add_argument("--ready-arm-z", type=float, default=0.40,
                    help="Arm-frame z for --ready-pose=strike-center. 0.40m keeps recent low/mid throws inside budget while centered balls stay easy.")

    ap.add_argument("--reach-m", type=float, default=J6_REACH_M)
    ap.add_argument("--z-min-m", type=float, default=J6_Z_MIN_M)
    ap.add_argument("--z-max-m", type=float, default=J6_Z_MAX_M)
    ap.add_argument("--world-z-min-m", type=float, default=J6_WORLD_Z_MIN_M)
    ap.add_argument("--world-z-max-m", type=float, default=1.10)
    ap.add_argument("--strike-arm-x-min-m", type=float, default=J6_STRIKE_A_X_MIN_M)
    ap.add_argument("--strike-arm-x-max-m", type=float, default=J6_STRIKE_A_X_MAX_M)
    ap.add_argument("--strike-arm-y-abs-max-m", type=float, default=0.38)
    ap.add_argument("--strike-arm-z-min-m", type=float, default=J6_STRIKE_A_Z_MIN_M)
    ap.add_argument("--strike-arm-z-max-m", type=float, default=0.62)
    ap.add_argument("--target-jump-max-m", type=float, default=J6_TARGET_JUMP_MAX_M)
    ap.add_argument("--retarget-joint-jump-deg", type=float, default=12.0)
    ap.add_argument("--ik-tol-mm", type=float, default=J6_IK_TOL_M * 1000.0)
    ap.add_argument("--z-mode", choices=["fixed-arm", "predicted"], default="predicted")
    ap.add_argument("--fixed-arm-z-m", type=float, default=0.45)
    ap.add_argument("--no-fixed-paddle-ori", action="store_true")
    ap.add_argument("--w-ori", type=float, default=J6_W_ORI)
    ap.add_argument("--max-ori-err-deg", type=float, default=J6_MAX_ORI_ERR_DEG)

    ap.add_argument("--tracking-step-deg", type=float, default=1.8)
    ap.add_argument("--tracking-accel-deg-s2", type=float, default=1200.0)
    ap.add_argument("--rate-hz", type=float, default=100.0)
    ap.add_argument("--post-impact-idle-s", type=float, default=0.5)
    ap.add_argument("--lock-release-after-impact-s", type=float, default=J6_LOCK_RELEASE_AFTER_IMPACT_S)
    ap.add_argument("--max-swings", type=int, default=0)
    ap.add_argument("--no-commit", action="store_true")
    ap.add_argument("--verbose-tracking", action="store_true")
    ap.add_argument("--full-segment-logs", action="store_true")
    ap.add_argument("--log-period-s", type=float, default=0.20)
    ap.add_argument("--log-file", type=str, default=None)
    ap.add_argument("--trace-file", type=str, default="auto",
                    help="Write per-tick JSONL trace. Use 'auto' for logs/j8_YYYYMMDD_HHMMSS.trace.jsonl, or 'off' to disable.")
    ap.add_argument("--shutdown-hold-s", type=float, default=0.0)
    ap.add_argument("--allow-zero-joint-start", action="store_true",
                    help="Allow startup when sensed joints are exactly zero. This is normally a stale/disconnected driver signal on the real robot.")

    ap.add_argument("--skip-cal", action="store_true")
    ap.add_argument("--offset-link7", nargs=3, type=float, default=None)
    ap.add_argument("--print-cal-only", action="store_true")
    ap.add_argument("--calibration", default=None)
    ap.add_argument("--redis-host", default="localhost")
    ap.add_argument("--redis-port", type=int, default=6379)

    ap.add_argument("--min-lookahead", type=float, default=None)
    ap.add_argument("--max-lookahead", type=float, default=None)
    ap.add_argument("--tracker-history-size", type=int, default=None)
    ap.add_argument("--tracker-history-max-age-s", type=float, default=None)
    ap.add_argument("--tracker-min-history", type=int, default=None)
    ap.add_argument("--tracker-median-window", type=int, default=None)
    ap.add_argument("--tracker-max-implied-speed-mps", type=float, default=None)
    ap.add_argument("--tracker-stale-position-eps-m", type=float, default=None)
    ap.add_argument("--tracker-stale-timeout-s", type=float, default=None)

    ap.add_argument("--base-ready-x", type=float, default=None,
                    help="World X of the base ready pose. If set, enables base lateral tracking and strike thrust.")
    ap.add_argument("--base-ready-y", type=float, default=0.0,
                    help="World Y of the base ready pose.")
    ap.add_argument("--base-ready-yaw-deg", type=float, default=0.0,
                    help="World yaw (deg) of the base ready pose.")
    ap.add_argument("--base-y-min", type=float, default=-1.5,
                    help="Min world Y the base may reach when tracking the ball laterally.")
    ap.add_argument("--base-y-max", type=float, default=1.5,
                    help="Max world Y the base may reach when tracking the ball laterally.")
    ap.add_argument("--base-thrust-m", type=float, default=0.5,
                    help="Drive base forward this many meters at strike commit to add impact velocity (0 = off).")
    ap.add_argument("--base-goal-key", type=str, default="sports_bot::cmd::base::goal_pose",
                    help="Redis key for world-frame base goal [x, y, theta]. Must be read by base_bridge.py.")
    ap.add_argument("--hold-base-after-lost-s", type=float, default=0.6,
                    help="Hold last base goal this many seconds after the ball is lost before returning to ready.")

    args = ap.parse_args()
    run_stamp = time.strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.abspath(os.path.join(_THIS_DIR, "..", "logs"))

    if args.log_file is not None:
        log_path = args.log_file
        if log_path == "auto":
            log_path = os.path.join(log_dir, f"j8_{run_stamp}.log")
        log_path = os.path.abspath(log_path)
        sys.stdout = _Tee(log_path)
        print(f"[J8] logging to {log_path}")

    trace_path = None
    if args.trace_file.lower() != "off":
        trace_path = args.trace_file
        if trace_path == "auto":
            trace_path = os.path.join(log_dir, f"j8_{run_stamp}.trace.jsonl")
        trace_path = os.path.abspath(trace_path)
    trace = _TraceLogger(trace_path)
    if trace_path:
        print(f"[J8] trace logging to {trace_path}")

    r = redis.Redis(host=args.redis_host, port=args.redis_port, decode_responses=True)
    try:
        r.ping()
    except redis.exceptions.ConnectionError as e:
        sys.exit(f"[J8] cannot reach Redis at {args.redis_host}:{args.redis_port}: {e}")

    xml_vel_limits = _read_xml_velocity_limits(r)
    q_start_check = get_vec(r, SENSOR_JOINTS, 7)
    if (
        q_start_check is None
        or (not args.allow_zero_joint_start and np.max(np.abs(q_start_check)) < 1e-6)
    ):
        sys.exit(
            "[J8] refusing to start: sensed joint state is missing or exactly zero. "
            "After a libfranka reflex abort, relaunch/recover the Franka driver/OpenSai first; "
            "otherwise the planner will command from a fake zero pose."
        )

    cal_path = args.calibration or arm_base_offset_calibration_path()
    if not os.path.isfile(cal_path):
        sys.exit(f"[J8] arm base calibration not found: {cal_path}")
    cal = load_arm_base_offset_calibration(cal_path)
    print(f"[J8] arm base calibration: {cal_path} (rigid body {cal.base_rigid_body_id})")

    mock_intercepts = None
    if args.mock_intercept:
        mock_intercepts = [np.asarray(p, dtype=float) for p in args.mock_intercept]
    if not (mock_intercepts and args.mock_identity_base):
        if read_rigid_body_pose_W(r, cal.base_rigid_body_id) is None:
            sys.exit(f"[J8] cart rigid body {cal.base_rigid_body_id} not visible in Redis")

    if not os.path.isfile(URDF_PATH):
        sys.exit(f"[J8] URDF not found: {URDF_PATH} (run from OpenSai root)")
    chain = build_chain()

    if args.skip_cal:
        offset_link7 = (
            np.asarray(args.offset_link7, dtype=float)
            if args.offset_link7 is not None
            else J6_DEFAULT_OFFSET_LINK7.copy()
        )
        _, offset_link7 = seed_offset_link7_no_cartesian(r, offset_link7)
    else:
        _, offset_link7 = calibrate_offset_link7(r, chain)

    if args.print_cal_only:
        print("[J8] --print-cal-only: done.")
        return

    R_A_link7_home = None
    if not args.no_fixed_paddle_ori:
        R_home_nominal = _link7_R_A(chain, Q_HOME_RAD)
        R_A_link7_home = _rot_y_rad(math.radians(-args.paddle_open_deg)) @ R_home_nominal
        face_A = R_A_link7_home @ (offset_link7 / max(np.linalg.norm(offset_link7), 1e-9))
        print("[J8] fixed paddle orientation from nominal Q_HOME")
        print(
            f"[J8]   paddle_open={args.paddle_open_deg:+.1f}° "
            f"strike-face normal_A≈[{face_A[0]:+.3f},{face_A[1]:+.3f},{face_A[2]:+.3f}]"
        )

    ready_arm_A = None
    q_ready_seed = Q_HOME_RAD.copy()
    q_ready = Q_HOME_RAD.copy()
    if args.ready_pose == "strike-center":
        ready_arm_A = np.array([args.ready_arm_x, args.ready_arm_y, args.ready_arm_z], dtype=float)
        q_ready, ready_err, ready_ori = ik_solve(
            chain, ready_arm_A, q_ready_seed, offset_link7,
            R_A_link7_target=R_A_link7_home, w_ori=args.w_ori,
        )
        if (not _joints_ok(q_ready)) or (not _ik_ok(
            ready_err, ready_ori, args.ik_tol_mm / 1000.0, args.max_ori_err_deg, R_A_link7_home,
        )):
            print(
                f"[J8] WARNING: strike-center ready IK failed "
                f"pos={ready_err*1000:.1f}mm ori={ready_ori:.1f}°; falling back to Q_HOME_RAD"
            )
            ready_arm_A = None
            q_ready = Q_HOME_RAD.copy()
        else:
            q_delta, q_joint = _max_abs_deg_with_joint(q_ready - Q_HOME_RAD)
            print(
                f"[J8] strike-center ready IK: arm_A={_fmt_v3(ready_arm_A)} "
                f"pos={ready_err*1000:.1f}mm ori={ready_ori:.1f}° "
                f"Δhome={q_delta:.1f}°@q{q_joint}"
            )

    print(f"[J8] moving to ready pose ({args.ready_pose}) ...")
    q_home_rad = move_to_home_pose(
        r, q_ready,
        move_s=args.home_s,
        hold_s=args.home_hold_s,
        vel_frac=args.home_vel_frac,
    )

    tracker = None
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
            f"[J8 tracker] hist={cfg.history_size}/{cfg.history_max_age_s:.2f}s "
            f"min_hist={cfg.min_history_for_prediction} median={cfg.median_filter_window} "
            f"lookahead=[{cfg.min_lookahead:.2f},{cfg.max_lookahead:.2f}]s"
        )
        keys = RedisKeys(ball_source="optitrack")
        keys.ball.__dict__["optitrack_rigid_body_id"] = args.ball_rigid_body_id
        ball_key = keys.ball.optitrack_position
        if r.get(ball_key) is None:
            print(f"[J8] WARNING: {ball_key} is empty")
        else:
            print(f"[J8] reading ball from {ball_key}")
        tracker = BallTracker(r, keys, cfg)

    base_ready_pose = None
    if args.base_ready_x is not None:
        base_ready_pose = (args.base_ready_x, args.base_ready_y, math.radians(args.base_ready_yaw_deg))

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
            pre_offset_m=args.pre_offset,
            follow_offset_m=args.follow_offset,
            follow_up_offset_m=args.follow_up_offset,
            through_q6_deg=args.through_q6_deg,
            follow_s=args.follow_s,
            commit_tti=args.commit_tti,
            launch_margin_s=args.launch_margin_s,
            contact_margin_s=args.contact_margin_s,
            timing_guard_s=args.timing_guard_s,
            return_s=args.return_s,
            return_vel_frac=args.return_vel_frac,
            rate_hz=args.rate_hz,
            no_commit=args.no_commit,
            reach_m=args.reach_m,
            z_min=args.z_min_m,
            z_max=args.z_max_m,
            world_z_min_m=args.world_z_min_m,
            world_z_max_m=args.world_z_max_m,
            target_jump_max_m=args.target_jump_max_m,
            retarget_joint_jump_deg=args.retarget_joint_jump_deg,
            tracking_step_deg=args.tracking_step_deg,
            tracking_accel_deg_s2=args.tracking_accel_deg_s2,
            swing_vel_frac=args.swing_vel_frac,
            ik_tol_m=args.ik_tol_mm / 1000.0,
            z_mode=args.z_mode,
            fixed_arm_z_m=args.fixed_arm_z_m,
            w_ori=args.w_ori,
            max_ori_err_deg=args.max_ori_err_deg,
            strike_arm_x_min_m=args.strike_arm_x_min_m,
            strike_arm_x_max_m=args.strike_arm_x_max_m,
            strike_arm_y_abs_max_m=args.strike_arm_y_abs_max_m,
            strike_arm_z_min_m=args.strike_arm_z_min_m,
            strike_arm_z_max_m=args.strike_arm_z_max_m,
            impact_vel_frac=args.impact_vel_frac,
            adaptive_trajectory=args.adaptive_trajectory,
            relaxed_ori_w_scale=args.relaxed_ori_w_scale,
            relaxed_ori_extra_deg=args.relaxed_ori_extra_deg,
            adaptive_q6_deg=args.adaptive_q6_deg,
            adaptive_follow_s=args.adaptive_follow_s,
            simple_fast=args.simple_fast,
            try_late_slack_s=args.try_late_slack_s,
            mock_intercepts=mock_intercepts,
            mock_tti=args.mock_tti,
            mock_cycle_s=args.mock_cycle_s,
            mock_identity_base=args.mock_identity_base,
            full_segment_logs=args.full_segment_logs,
            verbose_tracking=args.verbose_tracking,
            log_period_s=args.log_period_s,
            max_swings=args.max_swings,
            post_impact_idle_s=args.post_impact_idle_s,
            lock_release_after_impact_s=args.lock_release_after_impact_s,
            xml_vel_limits=xml_vel_limits,
            ready_label=args.ready_pose,
            ready_arm_A=ready_arm_A,
            trace=trace,
            ball_key=ball_key if mock_intercepts is None else None,
            base_goal_key=args.base_goal_key,
            base_ready_pose=base_ready_pose,
            base_y_min=args.base_y_min,
            base_y_max=args.base_y_max,
            base_thrust_m=args.base_thrust_m,
            hold_base_after_lost_s=args.hold_base_after_lost_s,
        )
    except KeyboardInterrupt:
        if args.shutdown_hold_s > 0.0:
            _hold_current_joints_for(r, args.shutdown_hold_s, publish_hz=100.0)
            print(f"\n[J8] stopped -- refreshed current joint hold for {args.shutdown_hold_s:.1f}s.")
        else:
            _hold_current_joints(r)
            print("\n[J8] stopped -- holding current joint position in Redis.")


if __name__ == "__main__":
    main()
