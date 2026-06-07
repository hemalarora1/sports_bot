#!/usr/bin/env python3
"""Step J11: preload behind the strike pose, then swing through early with joint PD.

J11 keeps the useful pieces from J9 -- foam-ball tracking, strike IK, and the
Jacobian solve for impact velocity -- but removes the timed Hermite impact
segment.  The swing primitive is deliberately small:

    offset    = swing_gain * qdot_strike * (kv / kp)
    q_pre     = q_strike - offset
    q_through = q_strike + offset

During tracking J11 rate-limits the joint goal toward q_pre.  As the ball enters
the commit window, it publishes q_through directly for a short window.  It
prefers being well preloaded, but it will still swing if it is merely close
enough; the point is to move fast through the ball rather than wait for a
fragile perfect timing instant.  The joint PD controller should carry the arm through q_strike with a
velocity in the qdot_strike direction.  After the through window, J11 returns
home with the existing slow measured-current -> home helper.
"""
from __future__ import annotations

import argparse
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
    GOAL_JOINTS,
    SENSOR_JOINTS,
    SENSOR_JOINT_VELS,
    get_vec,
    set_vec,
)
from stepj7_strike_planner import (  # noqa: E402
    J6_DEFAULT_OFFSET_LINK7,
    J6_HOME_VEL_FRAC,
    J6_IK_TOL_M,
    J6_LOCK_RELEASE_AFTER_IMPACT_S,
    J6_MAX_ORI_ERR_DEG,
    J6_REACH_M,
    J6_STRIKE_A_X_MAX_M,
    J6_STRIKE_A_X_MIN_M,
    J6_STRIKE_A_Z_MAX_M,
    J6_STRIKE_A_Z_MIN_M,
    J6_W_ORI,
    J6_WORLD_Z_MIN_M,
    J6_XML_VEL_LIMIT_RAD_S,
    J6_Z_MAX_M,
    J6_Z_MIN_M,
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
    _rate_accel_limited_goal,
    _rate_limited_goal,
    _return_home_blocking,
    _rot_y_rad,
    build_chain,
    calibrate_offset_link7,
    ik_solve,
    move_to_home_pose,
    seed_offset_link7_no_cartesian,
    switch_ctrl,
)
from stepj9_impact_velocity_planner import (  # noqa: E402
    J9_TRACKER_DRAG_COEFFICIENT,
    J9_TRACKER_GRAVITY_MPS2,
    J9_TRACKER_MIN_HISTORY,
    J9_TRACKER_SIMULATION_DT_S,
    _TraceLogger,
    _candidate_snapshot,
    _driver_diag_snapshot,
    _fmt_counts,
    _fr3_driver_velocity_bands_rad,
    _intercept_snapshot,
    _jsonable,
    _load_config_yaml,
    _make_impact_candidate,
    _parse_raw_json,
    _read_xml_velocity_limits,
    _safety_tripped,
    _sweet_spot_jacobian_A,
    _tracker_config_snapshot,
    _tracker_snapshot,
    _velocity_band_snapshot,
)
from sports_bot.foam_ball import FoamBallTracker  # noqa: E402
from sports_bot.foam_ball.config import FoamBallConfig  # noqa: E402
from sports_bot.state_machine.ball_tracker import Intercept  # noqa: E402
from sports_bot.state_machine.redis_keys import RedisKeys  # noqa: E402
from sports_bot.utils.frames import (  # noqa: E402
    arm_base_offset_calibration_path,
    compute_T_W_A_from_base_offset,
    load_arm_base_offset_calibration,
    read_rigid_body_pose_W,
)

# Joint gains in config_folder/xml_config_files/picklebot_j9_fast.xml.
J11_KP = np.array([200.0, 200.0, 200.0, 200.0, 400.0, 600.0, 400.0])
J11_KV = np.array([20.0, 20.0, 20.0, 20.0, 28.0, 28.0, 28.0])
J11_GAIN_TIME = J11_KV / J11_KP


@dataclass
class J11Candidate:
    base: object
    q_pre: np.ndarray
    q_through: np.ndarray
    offset: np.ndarray
    offset_scale: float
    pre_err_deg: float = float("inf")
    commit_tti_s: float = float("nan")

    def aged_tti(self) -> float:
        return self.base.aged_tti()


def _j11_candidate_snapshot(cand: J11Candidate | None) -> dict | None:
    if cand is None:
        return None
    snap = _candidate_snapshot(cand.base)
    if snap is None:
        snap = {}
    snap.update({
        "q_pre_deg": np.degrees(cand.q_pre),
        "q_through_deg": np.degrees(cand.q_through),
        "offset_deg": np.degrees(cand.offset),
        "offset_scale": cand.offset_scale,
        "pre_err_deg": cand.pre_err_deg,
        "commit_tti_s": cand.commit_tti_s,
    })
    return snap


def _limit_scale_for_joint_bounds(q_strike: np.ndarray, offset: np.ndarray, margin_rad: float) -> float:
    scale = 1.0
    lo = Q_LO + margin_rad
    hi = Q_HI - margin_rad
    for i, off in enumerate(offset):
        if abs(off) < 1e-9:
            continue
        if off > 0.0:
            scale = min(scale, max(0.0, (hi[i] - q_strike[i]) / off))
            scale = min(scale, max(0.0, (q_strike[i] - lo[i]) / off))
        else:
            scale = min(scale, max(0.0, (q_strike[i] - lo[i]) / (-off)))
            scale = min(scale, max(0.0, (hi[i] - q_strike[i]) / (-off)))
    return float(np.clip(scale, 0.0, 1.0))


def _limit_scale_for_velocity_band(q: np.ndarray, qdot: np.ndarray, hard_frac: float) -> float:
    bands = _fr3_driver_velocity_bands_rad(q)
    allowed = np.where(qdot >= 0.0, bands["hard_max"], -bands["hard_min"])
    allowed = np.maximum(allowed * max(0.05, hard_frac), 1e-6)
    ratios = np.abs(qdot) / allowed
    worst = float(np.max(ratios))
    if worst <= 1.0:
        return 1.0
    return max(0.0, 1.0 / worst)


def _make_j11_candidate(
    base,
    *,
    swing_gain: float,
    max_offset_deg: float,
    joint_margin_deg: float,
    hard_velocity_frac: float,
) -> tuple[J11Candidate | None, str]:
    offset = np.asarray(base.qdot_strike, dtype=float) * J11_GAIN_TIME * max(0.0, swing_gain)
    max_offset = math.radians(max_offset_deg)
    offset_abs = float(np.max(np.abs(offset)))
    scale = 1.0
    if offset_abs > max_offset > 0.0:
        scale = min(scale, max_offset / offset_abs)
    scale = min(scale, _limit_scale_for_joint_bounds(base.q_strike, offset, math.radians(joint_margin_deg)))
    scale = min(scale, _limit_scale_for_velocity_band(base.q_strike, base.qdot_strike, hard_velocity_frac))
    if scale <= 1e-3:
        return None, "j11 offset scaled to zero by joint/velocity bounds"
    offset = offset * scale
    q_pre = base.q_strike - offset
    q_through = base.q_strike + offset
    if not (_joints_ok(q_pre) and _joints_ok(q_through)):
        return None, "j11 q_pre/q_through outside joint limits"
    return J11Candidate(
        base=base,
        q_pre=q_pre,
        q_through=q_through,
        offset=offset,
        offset_scale=scale,
    ), "ok"


def _run_through_swing(
    r: redis.Redis,
    chain,
    cand: J11Candidate,
    offset_link7: np.ndarray,
    R_W_A: np.ndarray,
    *,
    publish_hz: float,
    through_s: float,
    swing_step_deg: float,
    trace: _TraceLogger | None,
) -> dict:
    period_s = 1.0 / max(1.0, publish_hz)
    t0 = time.perf_counter()
    next_publish = t0
    next_sample = t0
    next_trace_sample = t0
    qdot_peak = np.zeros(7)
    q_cross = None
    dq_cross = None
    best_alpha_err = float("inf")
    measured_v_W = np.full(3, np.nan)
    q_start_live = get_vec(r, SENSOR_JOINTS, 7)
    q_goal_cmd = cand.q_pre.copy() if q_start_live is None else q_start_live.copy()
    swing_axis = cand.q_through - cand.q_pre
    swing_norm2 = float(np.dot(swing_axis, swing_axis))

    def tr(event: str, **fields) -> None:
        if trace is not None:
            trace.write(event, **fields)

    print(
        f"[J11 swing] through goal for {through_s:.3f}s "
        f"offset={np.max(np.abs(np.degrees(cand.offset))):.1f}deg "
        f"step={swing_step_deg:.2f}deg/tick tti={cand.commit_tti_s:.3f}s"
    )
    tr("swing_start", candidate=_j11_candidate_snapshot(cand), diagnostic=_driver_diag_snapshot(r))

    while True:
        now = time.perf_counter()
        elapsed = now - t0
        if now >= next_publish:
            q_goal_cmd = _rate_limited_goal(cand.q_through, q_goal_cmd, swing_step_deg)
            set_vec(r, GOAL_JOINTS, q_goal_cmd)
            next_publish = now + period_s

        if now >= next_sample:
            q = get_vec(r, SENSOR_JOINTS, 7)
            dq = get_vec(r, SENSOR_JOINT_VELS, 7)
            if dq is not None:
                qdot_peak = np.maximum(qdot_peak, np.abs(np.degrees(dq)))
            if q is not None and swing_norm2 > 1e-12:
                alpha = float(np.dot(q - cand.q_pre, swing_axis) / swing_norm2)
                alpha_err = abs(alpha - 0.5)
                if alpha_err < best_alpha_err:
                    best_alpha_err = alpha_err
                    q_cross = q.copy()
                    dq_cross = None if dq is None else dq.copy()
            if trace is not None and now >= next_trace_sample:
                tr(
                    "swing_sample",
                    elapsed_s=elapsed,
                    q_goal_deg=np.degrees(q_goal_cmd),
                    q_actual_deg=None if q is None else np.degrees(q),
                    dq_actual_deg_s=None if dq is None else np.degrees(dq),
                    actual_fr3_velocity=None if q is None else _velocity_band_snapshot(q, dq),
                    diagnostic=_driver_diag_snapshot(r),
                )
                next_trace_sample = now + 0.02
            next_sample = now + 0.005

        if elapsed >= through_s:
            break
        time.sleep(0.0015)

    if q_cross is not None and dq_cross is not None:
        J_A = _sweet_spot_jacobian_A(chain, q_cross, offset_link7)
        measured_v_W = R_W_A @ (J_A @ dq_cross)
    q_now = get_vec(r, SENSOR_JOINTS, 7)
    strike_err_deg = float("nan") if q_cross is None else float(np.max(np.abs(np.degrees(cand.base.q_strike - q_cross))))
    through_err_deg = float("nan") if q_now is None else float(np.max(np.abs(np.degrees(cand.q_through - q_now))))
    qdot_i = int(np.argmax(qdot_peak))
    result = {
        "strike_err_deg": strike_err_deg,
        "through_err_deg": through_err_deg,
        "qdot_max_deg_s": float(qdot_peak[qdot_i]),
        "qdot_max_joint": qdot_i + 1,
        "measured_v_W": measured_v_W,
        "best_alpha_err": best_alpha_err,
        "offset_scale": cand.offset_scale,
    }
    print(
        f"[J11 segment] strike_err={strike_err_deg:.2f}deg through_err={through_err_deg:.2f}deg "
        f"qdot_max={qdot_peak[qdot_i]:.1f}deg/s@q{qdot_i+1} "
        f"v_meas_W=[{measured_v_W[0]:+.2f},{measured_v_W[1]:+.2f},{measured_v_W[2]:+.2f}]"
    )
    tr("swing_done", result=result, diagnostic=_driver_diag_snapshot(r))
    return result


def run_loop(
    *,
    r: redis.Redis,
    chain,
    tracker: FoamBallTracker | None,
    q_home_rad: np.ndarray,
    offset_link7: np.ndarray,
    R_A_link7_home: np.ndarray | None,
    cal,
    strike_plane_x_world: float,
    forward_speed_mps: float,
    up_speed_mps: float,
    qdot_damping: float,
    impact_vel_frac: float,
    swing_vel_frac: float,
    xml_vel_limits: np.ndarray,
    return_s: float,
    return_vel_frac: float,
    tracking_step_deg: float,
    tracking_accel_deg_s2: float,
    tracking_leash_deg: float,
    rate_hz: float,
    post_impact_idle_s: float,
    lock_release_after_impact_s: float,
    max_swings: int,
    no_commit: bool,
    log_period_s: float,
    verbose_tracking: bool,
    full_segment_logs: bool,
    mock_intercepts: list[np.ndarray] | None,
    mock_tti: float,
    mock_cycle_s: float,
    mock_identity_base: bool,
    ball_key: str | None,
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
    trace: _TraceLogger | None,
    j11_commit_tti: float,
    j11_pre_err_deg: float,
    j11_max_pre_err_deg: float,
    j11_max_through_err_deg: float,
    j11_swing_gain: float,
    j11_through_s: float,
    j11_swing_step_deg: float,
    j11_max_offset_deg: float,
    j11_joint_margin_deg: float,
    j11_hard_velocity_frac: float,
    j11_candidate_jump_hold_deg: float,
    j11_candidate_replace_margin_deg: float,
) -> None:
    dt = 1.0 / max(1.0, rate_hz)
    vel_cap = np.maximum(xml_vel_limits * max(0.05, swing_vel_frac), 1e-6)
    last_q_cmd = get_vec(r, SENSOR_JOINTS, 7)
    last_q_cmd_vel = np.zeros(7)
    last_goal_write_t = time.perf_counter()
    last_print_t = 0.0
    idle_until_t = 0.0
    active_cand: J11Candidate | None = None
    reject_counts: dict[str, int] = {}
    tick_i = 0
    swings_done = 0
    active_throw_id = 0
    mock_t0 = time.monotonic()
    using_mock = mock_intercepts is not None

    def tr(event: str, **fields) -> None:
        if trace is not None:
            trace.write(event, **fields)

    def print_reject(reason: str) -> None:
        reject_counts[reason] = reject_counts.get(reason, 0) + 1

    def stale_active_candidate() -> bool:
        if active_cand is None:
            return True
        return active_cand.aged_tti() < -lock_release_after_impact_s

    def track_active_candidate(q_cur: np.ndarray, reason: str) -> bool:
        """Keep moving toward the last good preload target through brief tracker gaps."""
        nonlocal active_cand, idle_until_t
        if active_cand is None:
            return False
        aged_tti = active_cand.aged_tti()
        if aged_tti < -lock_release_after_impact_s:
            tr("missed_window", tick=tick_i, aged_tti_s=aged_tti, candidate=_j11_candidate_snapshot(active_cand), reason=reason)
            active_cand = None
            idle_until_t = time.monotonic() + post_impact_idle_s
            return False
        pre_err, pre_joint = _max_abs_deg_with_joint(active_cand.q_pre - q_cur)
        through_err, through_joint = _max_abs_deg_with_joint(active_cand.q_through - q_cur)
        tr(
            "hold_active_candidate", tick=tick_i, reason=reason, aged_tti_s=aged_tti,
            pre_err_deg=pre_err, pre_err_joint=pre_joint,
            through_err_deg=through_err, through_err_joint=through_joint,
            candidate=_j11_candidate_snapshot(active_cand),
        )
        publish_limited_goal(active_cand.q_pre, q_cur, "track_active_pre")
        return True

    def choose_tracking_candidate(new_cand: J11Candidate, q_cur: np.ndarray) -> J11Candidate:
        nonlocal active_cand
        if active_cand is None:
            tr("candidate_update", tick=tick_i, decision="new", candidate=_j11_candidate_snapshot(new_cand))
            return new_cand
        jump_deg, jump_joint = _max_abs_deg_with_joint(new_cand.q_pre - active_cand.q_pre)
        new_pre_err, new_pre_joint = _max_abs_deg_with_joint(new_cand.q_pre - q_cur)
        old_pre_err, old_pre_joint = _max_abs_deg_with_joint(active_cand.q_pre - q_cur)
        clearly_better = new_pre_err + j11_candidate_replace_margin_deg < old_pre_err
        hold_jump = (
            j11_candidate_jump_hold_deg > 0.0
            and jump_deg > j11_candidate_jump_hold_deg
            and not clearly_better
            and active_cand.aged_tti() > 0.05
        )
        decision = "hold_jump" if hold_jump else "replace"
        tr(
            "candidate_update", tick=tick_i, decision=decision,
            jump_deg=jump_deg, jump_joint=jump_joint,
            old_pre_err_deg=old_pre_err, old_pre_err_joint=old_pre_joint,
            new_pre_err_deg=new_pre_err, new_pre_err_joint=new_pre_joint,
            old_candidate=_j11_candidate_snapshot(active_cand),
            new_candidate=_j11_candidate_snapshot(new_cand),
        )
        return active_cand if hold_jump else new_cand

    def publish_limited_goal(target: np.ndarray, q_cur: np.ndarray, mode: str) -> None:
        nonlocal last_q_cmd, last_q_cmd_vel, last_goal_write_t
        now_goal_t = time.perf_counter()
        raw_elapsed_s = now_goal_t - last_goal_write_t
        max_goal_gap_s = 0.05
        stale_goal = last_q_cmd is None or raw_elapsed_s > max_goal_gap_s
        cmd_state_error_deg = (
            float(np.max(np.abs(np.degrees(last_q_cmd - q_cur))))
            if last_q_cmd is not None else float("inf")
        )
        measured_reset = stale_goal
        if measured_reset:
            q_base = q_cur
            last_q_cmd_vel = np.zeros(7)
            elapsed_goal_s = dt
        else:
            q_base = last_q_cmd
            elapsed_goal_s = max(dt, min(raw_elapsed_s, max_goal_gap_s))
        if tracking_accel_deg_s2 > 0.0:
            q_cmd, last_q_cmd_vel = _rate_accel_limited_goal(
                target, q_base, last_q_cmd_vel,
                tracking_step_deg * rate_hz, tracking_accel_deg_s2, elapsed_goal_s,
            )
        else:
            q_cmd = _rate_limited_goal(target, q_base, tracking_step_deg * max(1.0, elapsed_goal_s * rate_hz))

        leash_clamped = False
        leash_error_deg = float("nan")
        if tracking_leash_deg > 0.0:
            delta = q_cmd - q_cur
            leash_error_deg = float(np.max(np.abs(np.degrees(delta))))
            if leash_error_deg > tracking_leash_deg:
                q_cmd = q_cur + delta * (tracking_leash_deg / max(leash_error_deg, 1e-9))
                leash_clamped = True

        set_vec(r, GOAL_JOINTS, q_cmd)
        tr(
            "goal_write", tick=tick_i, target_mode=mode,
            stale_goal_reset=stale_goal, measured_state_reset=measured_reset,
            leash_clamped=leash_clamped, leash_error_deg=leash_error_deg,
            cmd_state_error_deg=cmd_state_error_deg,
            q_target_deg=np.degrees(target), q_cmd_deg=np.degrees(q_cmd),
            q_cur_deg=np.degrees(q_cur), last_q_cmd_vel_deg_s=np.degrees(last_q_cmd_vel),
        )
        last_q_cmd = q_cmd.copy()
        last_goal_write_t = now_goal_t

    def recover_home(label: str, idle_after: bool) -> bool:
        nonlocal active_cand, last_q_cmd, last_q_cmd_vel, last_goal_write_t, idle_until_t
        tr("recover_home_start", tick=tick_i, label=label, active_candidate=_j11_candidate_snapshot(active_cand), diagnostic=_driver_diag_snapshot(r))
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
        last_q_cmd = get_vec(r, SENSOR_JOINTS, 7) if ok else _hold_current_joints(r)
        if last_q_cmd is None:
            last_q_cmd = get_vec(r, SENSOR_JOINTS, 7)
        last_q_cmd_vel = np.zeros(7)
        last_goal_write_t = time.perf_counter()
        if idle_after:
            idle_until_t = time.monotonic() + post_impact_idle_s
        tr("recover_home_done", tick=tick_i, label=label, ok=ok, idle_until_t=idle_until_t, diagnostic=_driver_diag_snapshot(r))
        return ok

    switch_ctrl(r, "joint_controller")
    _hold_current_joints(r)
    print(f"[J11] running at {rate_hz:.0f} Hz -- preload-through PD striker")
    print(
        f"[J11] strike_plane_x={strike_plane_x_world:+.3f}m j11_commit_tti={j11_commit_tti:.3f}s "
        f"pre_err ideal<={j11_pre_err_deg:.1f}deg max<={j11_max_pre_err_deg:.1f}deg "
        f"through_err<={j11_max_through_err_deg:.1f}deg through_s={j11_through_s:.3f}s"
    )
    print(
        f"[J11] velocity target W: forward=+{forward_speed_mps:.2f} m/s, up=+{up_speed_mps:.2f} m/s "
        f"swing_gain={j11_swing_gain:.2f} max_offset={j11_max_offset_deg:.1f}deg"
    )
    print("[J11] policy: track q_pre, swing early-through, publish q_through; no Hermite, no q_stop, no decel phase")
    print()

    while True:
        tick_i += 1
        loop_t = time.perf_counter()
        raw_ball = r.get(ball_key) if ball_key else None
        raw_ball_parsed = _parse_raw_json(raw_ball)
        switch_ctrl(r, "joint_controller")

        if using_mock:
            idx = 0
            if mock_cycle_s > 0.0 and len(mock_intercepts or []) > 1:
                idx = int((time.monotonic() - mock_t0) / mock_cycle_s) % len(mock_intercepts or [])
            t_W_strike = np.asarray((mock_intercepts or [np.zeros(3)])[idx], dtype=float)
            intercept = Intercept(t_W_strike.copy(), np.array([-5.0, 0.0, 0.0]), float(mock_tti), 0)
            sample = None
            reason = "mock"
        else:
            assert tracker is not None
            sample = tracker.update()
            intercept = tracker.predict_intercept(strike_plane_x_world)
            reason = getattr(tracker, "last_reject_reason", "") or "no_prediction"

        tr(
            "tracker_tick", tick=tick_i, raw_ball=raw_ball_parsed,
            sample_pos_W=None if sample is None else sample.pos,
            sample_t=None if sample is None else sample.t,
            tracker={} if using_mock else _tracker_snapshot(tracker),
            reject_reason=None if intercept is not None else reason,
            intercept=_intercept_snapshot(intercept),
            active_candidate=_j11_candidate_snapshot(active_cand),
        )

        if mock_identity_base:
            R_W_A = np.eye(3)
            t_W_A = np.zeros(3)
        else:
            T_W_B = read_rigid_body_pose_W(r, cal.base_rigid_body_id)
            if T_W_B is None:
                print_reject(f"cart rigid body {cal.base_rigid_body_id} not visible")
                time.sleep(max(0.0, dt - (time.perf_counter() - loop_t)))
                continue
            R_W_A, t_W_A = compute_T_W_A_from_base_offset(T_W_B, cal)

        q_cur = get_vec(r, SENSOR_JOINTS, 7)
        if q_cur is None:
            print_reject("no joint sensor")
            time.sleep(max(0.0, dt - (time.perf_counter() - loop_t)))
            continue

        if time.monotonic() < idle_until_t:
            publish_limited_goal(q_home_rad, q_cur, "post_impact_ready")
            tr("post_impact_idle", tick=tick_i, idle_until_t=idle_until_t, q_cur_deg=np.degrees(q_cur))
            time.sleep(max(0.0, dt - (time.perf_counter() - loop_t)))
            continue

        if intercept is None:
            print_reject(reason)
            tr("no_intercept_idle", tick=tick_i, home_err_deg=_home_error_deg(r, q_home_rad), q_cur_deg=np.degrees(q_cur))
            if track_active_candidate(q_cur, f"no_intercept:{reason}"):
                time.sleep(max(0.0, dt - (time.perf_counter() - loop_t)))
                continue
            publish_limited_goal(q_home_rad, q_cur, "idle_ready")
            time.sleep(max(0.0, dt - (time.perf_counter() - loop_t)))
            continue

        if stale_active_candidate():
            active_cand = None

        q_seed = active_cand.base.q_strike if active_cand is not None else q_cur
        base, base_reason = _make_impact_candidate(
            chain=chain,
            q_cur=q_cur,
            q_seed=q_seed,
            intercept=intercept,
            R_W_A=R_W_A,
            t_W_A=t_W_A,
            offset_link7=offset_link7,
            R_A_link7_home=R_A_link7_home,
            forward_speed_mps=forward_speed_mps,
            up_speed_mps=up_speed_mps,
            qdot_damping=qdot_damping,
            decel_s=0.10,
            contact_margin_s=0.0,
            timing_guard_s=0.0,
            vel_cap=vel_cap,
            impact_vel_frac=impact_vel_frac,
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
            max_ori_err_deg=max_ori_err_deg,
            w_ori=w_ori,
            z_mode=z_mode,
            fixed_arm_z_m=fixed_arm_z_m,
        )
        if base is None:
            print_reject(base_reason)
            tr("candidate", tick=tick_i, reason=base_reason, candidate=None, intercept=_intercept_snapshot(intercept))
            if track_active_candidate(q_cur, f"base_reject:{base_reason}"):
                time.sleep(max(0.0, dt - (time.perf_counter() - loop_t)))
                continue
            publish_limited_goal(q_home_rad, q_cur, "reject_ready")
            time.sleep(max(0.0, dt - (time.perf_counter() - loop_t)))
            continue

        cand, cand_reason = _make_j11_candidate(
            base,
            swing_gain=j11_swing_gain,
            max_offset_deg=j11_max_offset_deg,
            joint_margin_deg=j11_joint_margin_deg,
            hard_velocity_frac=j11_hard_velocity_frac,
        )
        tr("candidate", tick=tick_i, reason=cand_reason, candidate=_j11_candidate_snapshot(cand), intercept=_intercept_snapshot(intercept))
        if cand is None:
            print_reject(cand_reason)
            if track_active_candidate(q_cur, f"candidate_reject:{cand_reason}"):
                time.sleep(max(0.0, dt - (time.perf_counter() - loop_t)))
                continue
            publish_limited_goal(q_home_rad, q_cur, "reject_ready")
            time.sleep(max(0.0, dt - (time.perf_counter() - loop_t)))
            continue

        cand = choose_tracking_candidate(cand, q_cur)
        active_cand = cand
        aged_tti = cand.aged_tti()
        pre_err, pre_joint = _max_abs_deg_with_joint(cand.q_pre - q_cur)
        cand.pre_err_deg = pre_err
        cand.commit_tti_s = aged_tti
        offset_deg, offset_joint = _max_abs_deg_with_joint(cand.offset)
        qdot = get_vec(r, SENSOR_JOINT_VELS, 7)
        qdot_txt = ""
        if qdot is not None:
            qdot_deg, qdot_joint = _max_abs_deg_with_joint(qdot)
            qdot_txt = f" qdot={qdot_deg:.1f}deg/s@q{qdot_joint}"
        ready = pre_err <= j11_pre_err_deg
        through_err, through_joint = _max_abs_deg_with_joint(cand.q_through - q_cur)
        close_enough = pre_err <= j11_max_pre_err_deg and through_err <= j11_max_through_err_deg
        too_late = aged_tti < -lock_release_after_impact_s
        commit_now = (not no_commit) and close_enough and aged_tti <= j11_commit_tti and not too_late
        state = "READY" if ready else ("CHASE" if close_enough else "PRELOAD")
        if commit_now:
            state = "COMMIT" if ready else "GO_EARLY"
        elif too_late:
            state = "MISSED"

        tr(
            "candidate_runtime", tick=tick_i, state=state, ready=ready, close_enough=close_enough,
            aged_tti_s=aged_tti, pre_err_deg=pre_err, pre_err_joint=pre_joint,
            through_err_deg=through_err, through_err_joint=through_joint,
            offset_deg=offset_deg, offset_joint=offset_joint,
            q_cur_deg=np.degrees(q_cur), candidate=_j11_candidate_snapshot(cand),
            q_pre_fr3_velocity=_velocity_band_snapshot(cand.q_pre, cand.base.qdot_strike),
            q_strike_fr3_velocity=_velocity_band_snapshot(cand.base.q_strike, cand.base.qdot_strike),
        )

        if loop_t - last_print_t >= max(0.05, log_period_s):
            last_print_t = loop_t
            print(
                f"[J11] {state:7s} tti={aged_tti:+.3f}s pre_err={pre_err:.1f}deg@q{pre_joint} "
                f"through_err={through_err:.1f}deg@q{through_joint} "
                f"offset={offset_deg:.1f}deg@q{offset_joint} scale={cand.offset_scale:.2f} "
                f"strike_W={_fmt_v3(cand.base.strike_W)} vW=[{cand.base.forward_mps:+.2f},{cand.base.up_mps:+.2f}]"
            )
            if verbose_tracking:
                print(
                    f"[J11]   strike_A={_fmt_v3(cand.base.strike_A)} "
                    f"ik={cand.base.ik_err_mm:.1f}mm ori={cand.base.ori_err_deg:.1f}deg "
                    f"rejects={_fmt_counts(reject_counts)}{qdot_txt}"
                )

        if commit_now:
            active_throw_id += 1
            tr("commit_start", tick=tick_i, throw_id=active_throw_id, candidate=_j11_candidate_snapshot(cand), diagnostic=_driver_diag_snapshot(r))
            print("\n" + "=" * 64)
            print(
                f"[J11] COMMIT throw {active_throw_id}: tti={aged_tti:.3f}s pre_err={pre_err:.1f}deg "
                f"offset={offset_deg:.1f}deg scale={cand.offset_scale:.2f}"
            )
            print(
                f"[J11]   q_pre:     {_fmt_q(cand.q_pre)}\n"
                f"[J11]   q_strike:  {_fmt_q(cand.base.q_strike)}\n"
                f"[J11]   q_through: {_fmt_q(cand.q_through)}"
            )
            print("=" * 64 + "\n")
            result = _run_through_swing(
                r, chain, cand, offset_link7, R_W_A,
                publish_hz=200.0,
                through_s=j11_through_s,
                swing_step_deg=j11_swing_step_deg,
                trace=trace,
            )
            if _safety_tripped(r):
                print("[J11] safety torque after swing; recovering slowly")
            home_ok = recover_home("post-through->home", idle_after=True)
            swings_done += 1
            outcome = "OK" if home_ok and (not math.isfinite(result["strike_err_deg"]) or result["strike_err_deg"] <= 10.0) else "WARN"
            print(
                f"[J11 audit] throw={active_throw_id} outcome={outcome} "
                f"tti={aged_tti:.3f}s pre_err={pre_err:.1f}deg "
                f"strike_err={result['strike_err_deg']:.2f}deg through_err={result['through_err_deg']:.2f}deg "
                f"home_ok={home_ok} rejects={_fmt_counts(reject_counts)}"
            )
            tr("commit_done", tick=tick_i, throw_id=active_throw_id, outcome=outcome, result=result, home_ok=home_ok, reject_counts=reject_counts)
            reject_counts = {}
            active_cand = None
            if max_swings > 0 and swings_done >= max_swings:
                print(f"[J11] --max-swings={max_swings} reached; exiting")
                break
            time.sleep(max(0.0, dt - (time.perf_counter() - loop_t)))
            continue

        publish_limited_goal(cand.q_pre, q_cur, "track_pre")
        if too_late:
            tr("missed_window", tick=tick_i, aged_tti_s=aged_tti, candidate=_j11_candidate_snapshot(cand))
            active_cand = None
            idle_until_t = time.monotonic() + post_impact_idle_s
        time.sleep(max(0.0, dt - (time.perf_counter() - loop_t)))


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="J11: preload q_pre, then publish q_through for a fast PD swing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--config", default=None, metavar="YAML", help="YAML config file; CLI flags override.")
    ap.add_argument("--ball-rigid-body-id", type=int, default=13)
    ap.add_argument("--strike-plane-x", type=float, default=-0.30)
    ap.add_argument("--mock-intercept", nargs=3, type=float, action="append")
    ap.add_argument("--mock-tti", type=float, default=0.60)
    ap.add_argument("--mock-cycle-s", type=float, default=0.0)
    ap.add_argument("--mock-identity-base", action="store_true")

    ap.add_argument("--paddle-open-deg", type=float, default=8.0)
    ap.add_argument("--forward-speed-mps", type=float, default=1.50)
    ap.add_argument("--up-speed-mps", type=float, default=0.40)
    ap.add_argument("--impact-vel-frac", type=float, default=1.00)
    ap.add_argument("--qdot-damping", type=float, default=0.03)
    ap.add_argument("--swing-vel-frac", type=float, default=1.00)
    ap.add_argument("--vel-cap-rad-s", nargs=7, type=float, default=None)

    # J9's commit_tti may appear in reused YAMLs. Keep J11 timing independent.
    ap.add_argument("--j11-commit-tti", type=float, default=0.45)
    ap.add_argument("--j11-pre-err-deg", type=float, default=8.0)
    ap.add_argument("--j11-max-pre-err-deg", type=float, default=12.0)
    ap.add_argument("--j11-max-through-err-deg", type=float, default=18.0)
    ap.add_argument("--j11-swing-gain", type=float, default=0.65)
    ap.add_argument("--j11-through-s", type=float, default=0.24)
    ap.add_argument("--j11-swing-step-deg", type=float, default=0.8)
    ap.add_argument("--j11-max-offset-deg", type=float, default=10.0)
    ap.add_argument("--j11-joint-margin-deg", type=float, default=2.0)
    ap.add_argument("--j11-hard-velocity-frac", type=float, default=0.85)
    ap.add_argument("--j11-candidate-jump-hold-deg", type=float, default=18.0)
    ap.add_argument("--j11-candidate-replace-margin-deg", type=float, default=6.0)

    # Accepted for compatibility with J9 YAMLs; not used by J11's swing primitive.
    ap.add_argument("--commit-tti", type=float, default=0.90, help=argparse.SUPPRESS)
    ap.add_argument("--try-late-slack-s", type=float, default=0.0, help=argparse.SUPPRESS)
    ap.add_argument("--max-stretched-strike-s", type=float, default=0.0, help=argparse.SUPPRESS)
    ap.add_argument("--contact-margin-s", type=float, default=0.0, help=argparse.SUPPRESS)
    ap.add_argument("--timing-guard-s", type=float, default=0.0, help=argparse.SUPPRESS)
    ap.add_argument("--decel-s", type=float, default=0.0, help=argparse.SUPPRESS)

    ap.add_argument("--return-s", type=float, default=4.0)
    ap.add_argument("--return-vel-frac", type=float, default=0.35)
    ap.add_argument("--home-s", type=float, default=3.0)
    ap.add_argument("--home-hold-s", type=float, default=1.0)
    ap.add_argument("--home-vel-frac", type=float, default=J6_HOME_VEL_FRAC)
    ap.add_argument("--home-joints-deg", nargs=7, type=float, default=None)
    ap.add_argument("--ready-pose", choices=["home", "strike-center"], default="home")
    ap.add_argument("--ready-arm-x", type=float, default=0.28)
    ap.add_argument("--ready-arm-y", type=float, default=0.0)
    ap.add_argument("--ready-arm-z", type=float, default=0.35)

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
    ap.add_argument("--ik-tol-mm", type=float, default=J6_IK_TOL_M * 1000.0)
    ap.add_argument("--z-mode", choices=["fixed-arm", "predicted"], default="predicted")
    ap.add_argument("--fixed-arm-z-m", type=float, default=0.45)
    ap.add_argument("--no-fixed-paddle-ori", action="store_true")
    ap.add_argument("--w-ori", type=float, default=J6_W_ORI)
    ap.add_argument("--max-ori-err-deg", type=float, default=J6_MAX_ORI_ERR_DEG)

    ap.add_argument("--tracking-step-deg", type=float, default=1.0)
    ap.add_argument("--tracking-accel-deg-s2", type=float, default=0.0)
    ap.add_argument("--tracking-leash-deg", type=float, default=3.0)
    ap.add_argument("--rate-hz", type=float, default=100.0)
    ap.add_argument("--post-impact-idle-s", type=float, default=0.4)
    ap.add_argument("--lock-release-after-impact-s", type=float, default=J6_LOCK_RELEASE_AFTER_IMPACT_S)
    ap.add_argument("--max-swings", type=int, default=0)
    ap.add_argument("--no-commit", action="store_true")
    ap.add_argument("--verbose-tracking", action="store_true")
    ap.add_argument("--full-segment-logs", action="store_true")
    ap.add_argument("--log-period-s", type=float, default=0.20)
    ap.add_argument("--log-file", type=str, default=None)
    ap.add_argument("--trace-file", type=str, default="auto")
    ap.add_argument("--shutdown-hold-s", type=float, default=0.0)
    ap.add_argument("--allow-zero-joint-start", action="store_true")

    ap.add_argument("--skip-cal", action="store_true")
    ap.add_argument("--offset-link7", nargs=3, type=float, default=None)
    ap.add_argument("--print-cal-only", action="store_true")
    ap.add_argument("--calibration", default=None)
    ap.add_argument("--redis-host", default="localhost")
    ap.add_argument("--redis-port", type=int, default=6379)

    ap.add_argument("--min-lookahead", type=float, default=None)
    ap.add_argument("--max-lookahead", type=float, default=None)
    ap.add_argument("--tracker-gravity", type=float, default=J9_TRACKER_GRAVITY_MPS2)
    ap.add_argument("--tracker-drag-coefficient", type=float, default=J9_TRACKER_DRAG_COEFFICIENT)
    ap.add_argument("--tracker-simulation-dt", type=float, default=J9_TRACKER_SIMULATION_DT_S)
    ap.add_argument("--tracker-history-size", type=int, default=None)
    ap.add_argument("--tracker-history-max-age-s", type=float, default=None)
    ap.add_argument("--tracker-min-history", type=int, default=J9_TRACKER_MIN_HISTORY)
    ap.add_argument("--tracker-median-window", type=int, default=None)
    ap.add_argument("--tracker-max-implied-speed-mps", type=float, default=None)
    ap.add_argument("--tracker-stale-position-eps-m", type=float, default=None)
    ap.add_argument("--tracker-stale-timeout-s", type=float, default=None)
    return ap


def main() -> None:
    ap = _build_parser()
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    pre_args, _ = pre.parse_known_args()
    if pre_args.config is not None:
        ap.set_defaults(**_load_config_yaml(pre_args.config))
    args = ap.parse_args()

    run_stamp = time.strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.abspath(os.path.join(_THIS_DIR, "..", "logs"))
    if args.log_file is not None:
        log_path = args.log_file
        if log_path == "auto":
            log_path = os.path.join(log_dir, f"j11_{run_stamp}.log")
        log_path = os.path.abspath(log_path)
        sys.stdout = _Tee(log_path)
        print(f"[J11] logging to {log_path}")

    trace_path = None
    if args.trace_file.lower() != "off":
        trace_path = args.trace_file
        if trace_path == "auto":
            trace_path = os.path.join(log_dir, f"j11_{run_stamp}.trace.jsonl")
        trace_path = os.path.abspath(trace_path)
    trace = _TraceLogger(trace_path)
    if trace_path:
        print(f"[J11] trace logging to {trace_path}")

    r = redis.Redis(host=args.redis_host, port=args.redis_port, decode_responses=True)
    try:
        r.ping()
    except redis.exceptions.ConnectionError as exc:
        sys.exit(f"[J11] cannot reach Redis at {args.redis_host}:{args.redis_port}: {exc}")

    xml_vel_limits = _read_xml_velocity_limits(r)
    if args.vel_cap_rad_s is not None:
        xml_vel_limits = np.asarray(args.vel_cap_rad_s, dtype=float)
        print(f"[J11] vel_cap override (--vel-cap-rad-s): {np.degrees(xml_vel_limits).round(1).tolist()} deg/s")

    q_start_check = get_vec(r, SENSOR_JOINTS, 7)
    if q_start_check is None or (not args.allow_zero_joint_start and np.max(np.abs(q_start_check)) < 1e-6):
        sys.exit("[J11] refusing to start: sensed joint state is missing or exactly zero. Recover/relaunch driver/OpenSai first.")

    cal_path = args.calibration or arm_base_offset_calibration_path()
    if not os.path.isfile(cal_path):
        sys.exit(f"[J11] arm base calibration not found: {cal_path}")
    cal = load_arm_base_offset_calibration(cal_path)
    print(f"[J11] arm base calibration: {cal_path} (rigid body {cal.base_rigid_body_id})")

    mock_intercepts = None
    if args.mock_intercept:
        mock_intercepts = [np.asarray(p, dtype=float) for p in args.mock_intercept]
    if not (mock_intercepts and args.mock_identity_base):
        if read_rigid_body_pose_W(r, cal.base_rigid_body_id) is None:
            sys.exit(f"[J11] cart rigid body {cal.base_rigid_body_id} not visible in Redis")

    if not os.path.isfile(URDF_PATH):
        sys.exit(f"[J11] URDF not found: {URDF_PATH} (run from OpenSai root)")
    chain = build_chain()

    if args.skip_cal:
        offset_link7 = np.asarray(args.offset_link7, dtype=float) if args.offset_link7 is not None else J6_DEFAULT_OFFSET_LINK7.copy()
        _, offset_link7 = seed_offset_link7_no_cartesian(r, offset_link7)
    else:
        _, offset_link7 = calibrate_offset_link7(r, chain)

    if args.print_cal_only:
        print("[J11] --print-cal-only: done.")
        return

    q_ready_seed = Q_HOME_RAD.copy()
    if args.home_joints_deg is not None:
        q_ready_seed = np.radians(np.asarray(args.home_joints_deg, dtype=float))
        q_delta, q_joint = _max_abs_deg_with_joint(q_ready_seed - Q_HOME_RAD)
        print(f"[J11] custom home pose: {_fmt_q(q_ready_seed)} (delta imported home {q_delta:.1f}deg@q{q_joint})")

    R_A_link7_home = None
    if not args.no_fixed_paddle_ori:
        R_home_nominal = _link7_R_A(chain, q_ready_seed)
        R_A_link7_home = _rot_y_rad(math.radians(-args.paddle_open_deg)) @ R_home_nominal
        face_A = R_A_link7_home @ (offset_link7 / max(np.linalg.norm(offset_link7), 1e-9))
        print("[J11] fixed paddle orientation from ready/home pose")
        print(f"[J11]   paddle_open={args.paddle_open_deg:+.1f}deg strike-face normal_A=[{face_A[0]:+.3f},{face_A[1]:+.3f},{face_A[2]:+.3f}]")

    q_ready = q_ready_seed.copy()
    if args.ready_pose == "strike-center":
        ready_arm_A = np.array([args.ready_arm_x, args.ready_arm_y, args.ready_arm_z], dtype=float)
        q_candidate, ready_err, ready_ori = ik_solve(
            chain, ready_arm_A, q_ready, offset_link7,
            R_A_link7_target=R_A_link7_home, w_ori=args.w_ori,
        )
        if _joints_ok(q_candidate) and _ik_ok(ready_err, ready_ori, args.ik_tol_mm / 1000.0, args.max_ori_err_deg, R_A_link7_home):
            q_ready = q_candidate
        else:
            print(f"[J11] WARNING: strike-center ready IK failed pos={ready_err*1000:.1f}mm ori={ready_ori:.1f}deg; using home pose")

    print(f"[J11] moving to ready pose ({args.ready_pose}) ...")
    q_home_rad = move_to_home_pose(r, q_ready, move_s=args.home_s, hold_s=args.home_hold_s, vel_frac=args.home_vel_frac)

    tracker = None
    ball_key = None
    if mock_intercepts is None:
        cfg = FoamBallConfig()
        if args.tracker_gravity is not None:
            cfg.gravity = args.tracker_gravity
        if args.tracker_drag_coefficient is not None:
            cfg.drag_coefficient = args.tracker_drag_coefficient
        if args.tracker_simulation_dt is not None:
            cfg.simulation_dt = args.tracker_simulation_dt
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
            f"[J11 tracker] model=foam_drag gravity={cfg.gravity:.2f}m/s^2 "
            f"drag_k={cfg.drag_coefficient:.3f} sim_dt={cfg.simulation_dt:.4f}s "
            f"hist={cfg.history_size}/{cfg.history_max_age_s:.2f}s "
            f"min_hist={cfg.min_history_for_prediction} median={cfg.median_filter_window} "
            f"lookahead=[{cfg.min_lookahead:.2f},{cfg.max_lookahead:.2f}]s"
        )
        keys = RedisKeys(ball_source="optitrack")
        keys.ball.__dict__["optitrack_rigid_body_id"] = args.ball_rigid_body_id
        ball_key = keys.ball.optitrack_position
        if r.get(ball_key) is None:
            print(f"[J11] WARNING: {ball_key} is empty")
        else:
            print(f"[J11] reading ball from {ball_key}")
        tracker = FoamBallTracker(r, keys, cfg)
        if trace is not None:
            trace.write("tracker_config", ball_key=ball_key, ball_rigid_body_id=args.ball_rigid_body_id, tracker_config=_tracker_config_snapshot(tracker))

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
            forward_speed_mps=args.forward_speed_mps,
            up_speed_mps=args.up_speed_mps,
            qdot_damping=args.qdot_damping,
            impact_vel_frac=args.impact_vel_frac,
            swing_vel_frac=args.swing_vel_frac,
            xml_vel_limits=xml_vel_limits,
            return_s=args.return_s,
            return_vel_frac=args.return_vel_frac,
            tracking_step_deg=args.tracking_step_deg,
            tracking_accel_deg_s2=args.tracking_accel_deg_s2,
            tracking_leash_deg=args.tracking_leash_deg,
            rate_hz=args.rate_hz,
            post_impact_idle_s=args.post_impact_idle_s,
            lock_release_after_impact_s=args.lock_release_after_impact_s,
            max_swings=args.max_swings,
            no_commit=args.no_commit,
            log_period_s=args.log_period_s,
            verbose_tracking=args.verbose_tracking,
            full_segment_logs=args.full_segment_logs,
            mock_intercepts=mock_intercepts,
            mock_tti=args.mock_tti,
            mock_cycle_s=args.mock_cycle_s,
            mock_identity_base=args.mock_identity_base,
            ball_key=ball_key,
            reach_m=args.reach_m,
            z_min=args.z_min_m,
            z_max=args.z_max_m,
            world_z_min_m=args.world_z_min_m,
            world_z_max_m=args.world_z_max_m,
            strike_arm_x_min_m=args.strike_arm_x_min_m,
            strike_arm_x_max_m=args.strike_arm_x_max_m,
            strike_arm_y_abs_max_m=args.strike_arm_y_abs_max_m,
            strike_arm_z_min_m=args.strike_arm_z_min_m,
            strike_arm_z_max_m=args.strike_arm_z_max_m,
            ik_tol_m=args.ik_tol_mm / 1000.0,
            max_ori_err_deg=args.max_ori_err_deg,
            w_ori=args.w_ori,
            z_mode=args.z_mode,
            fixed_arm_z_m=args.fixed_arm_z_m,
            trace=trace,
            j11_commit_tti=args.j11_commit_tti,
            j11_pre_err_deg=args.j11_pre_err_deg,
            j11_max_pre_err_deg=args.j11_max_pre_err_deg,
            j11_max_through_err_deg=args.j11_max_through_err_deg,
            j11_swing_gain=args.j11_swing_gain,
            j11_through_s=args.j11_through_s,
            j11_swing_step_deg=args.j11_swing_step_deg,
            j11_max_offset_deg=args.j11_max_offset_deg,
            j11_joint_margin_deg=args.j11_joint_margin_deg,
            j11_hard_velocity_frac=args.j11_hard_velocity_frac,
            j11_candidate_jump_hold_deg=args.j11_candidate_jump_hold_deg,
            j11_candidate_replace_margin_deg=args.j11_candidate_replace_margin_deg,
        )
    finally:
        trace.close()
        if args.shutdown_hold_s > 0.0:
            _hold_current_joints_for(r, args.shutdown_hold_s, publish_hz=100.0)
            print(f"\n[J11] stopped -- refreshed current joint hold for {args.shutdown_hold_s:.1f}s.")
        else:
            _hold_current_joints(r)
            print("\n[J11] stopped -- holding current joint position in Redis.")


if __name__ == "__main__":
    main()
