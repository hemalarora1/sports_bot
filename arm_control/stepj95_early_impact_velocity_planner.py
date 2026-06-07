#!/usr/bin/env python3
"""
Step J9.5: early smooth impact-velocity striker.

Goal: commit often, react quickly, and keep the swing primitive simple.
A valid throw is handled as one timed impact problem:

    q_now -> q_strike at contact with qdot_strike -> short decel -> home

J9 continuously aims at the latest predicted strike IK before commit. At commit
it freezes the latest q_strike and executes a cubic Hermite segment that reaches
that pose with a joint velocity chosen to make the paddle sweet spot move mostly
forward, with a small upward component.
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
    SENT_TORQUES,
    SENSOR_JOINTS,
    SENSOR_JOINT_VELS,
    get_vec,
    set_vec,
)
from stepj7_strike_planner import (  # noqa: E402
    JOINT_CTRL,
    J6_DEFAULT_OFFSET_LINK7,
    J6_HOME_VEL_FRAC,
    J6_IK_TOL_M,
    J6_LOCK_RELEASE_AFTER_IMPACT_S,
    J6_MAX_ORI_ERR_DEG,
    J6_REACH_M,
    J6_STRIKE_A_X_MAX_M,
    J6_STRIKE_A_X_MIN_M,
    J6_STRIKE_A_Y_ABS_MAX_M,
    J6_STRIKE_A_Z_MAX_M,
    J6_STRIKE_A_Z_MIN_M,
    J6_W_ORI,
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
    _fk,
    _fmt_q,
    _fmt_v3,
    _hold_current_joints,
    _hold_current_joints_for,
    _home_error_deg,
    _ik_ok,
    _joints_ok,
    _link7_R_A,
    _max_abs_deg_with_joint,
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
from stepj8_through_strike_planner import (  # noqa: E402
    _TraceLogger,
    _fmt_counts,
    _hermite,
    _intercept_snapshot,
    _jsonable,
    _minimum_feasible_strike_time,
    _parse_raw_json,
    _tracker_snapshot as _base_tracker_snapshot,
    _trajectory_peak_profile,
)
from sports_bot.state_machine.ball_tracker import Intercept  # noqa: E402
from sports_bot.foam_ball import FoamBallTracker  # noqa: E402
from sports_bot.foam_ball.config import FoamBallConfig  # noqa: E402
from sports_bot.state_machine.redis_keys import RedisKeys  # noqa: E402
from sports_bot.utils.frames import (  # noqa: E402
    arm_base_offset_calibration_path,
    compute_T_W_A_from_base_offset,
    load_arm_base_offset_calibration,
    read_rigid_body_pose_W,
)


# Fitted from clean new_ball recordings on 2026-06-02 at strike_plane_x=-0.30.
# The mild gravity boost plus mild drag matched early and commit-window Z best.
J95_TRACKER_GRAVITY_MPS2 = 11.0
J95_TRACKER_DRAG_COEFFICIENT = 0.10
J95_TRACKER_SIMULATION_DT_S = 0.001
J95_TRACKER_MIN_HISTORY = 4


@dataclass
class ImpactCandidate:
    strike_W: np.ndarray
    strike_A: np.ndarray
    q_strike: np.ndarray
    q_stop: np.ndarray
    qdot_strike: np.ndarray
    desired_v_W: np.ndarray
    achieved_v_W: np.ndarray
    tti_s: float
    observed_t: float
    min_strike_s: float
    strike_time_s: float
    budget_margin_s: float
    cmd_peak_frac: float
    ik_err_mm: float
    ori_err_deg: float
    decel_s: float
    forward_mps: float
    up_mps: float

    def aged_tti(self) -> float:
        return self.tti_s - (time.monotonic() - self.observed_t)


def _tracker_config_snapshot(tracker: FoamBallTracker | None) -> dict:
    if tracker is None:
        return {}
    cfg = getattr(tracker, "_cfg", None)
    if cfg is None:
        return {}
    return {
        "model": "foam_drag",
        "gravity": getattr(cfg, "gravity", None),
        "drag_coefficient": getattr(cfg, "drag_coefficient", None),
        "simulation_dt": getattr(cfg, "simulation_dt", None),
        "history_size": getattr(cfg, "history_size", None),
        "history_max_age_s": getattr(cfg, "history_max_age_s", None),
        "min_history_for_prediction": getattr(cfg, "min_history_for_prediction", None),
        "median_filter_window": getattr(cfg, "median_filter_window", None),
        "min_lookahead": getattr(cfg, "min_lookahead", None),
        "max_lookahead": getattr(cfg, "max_lookahead", None),
        "min_incoming_speed": getattr(cfg, "min_incoming_speed", None),
        "max_implied_speed_mps": getattr(cfg, "max_implied_speed_mps", None),
        "max_bounces": getattr(cfg, "max_bounces", None),
        "stale_position_epsilon_m": getattr(cfg, "stale_position_epsilon_m", None),
        "stale_position_timeout_s": getattr(cfg, "stale_position_timeout_s", None),
    }


def _tracker_snapshot(tracker: FoamBallTracker | None) -> dict:
    snap = _base_tracker_snapshot(tracker)
    if tracker is not None:
        snap["config"] = _tracker_config_snapshot(tracker)
    return snap


def _candidate_snapshot(cand: ImpactCandidate | None) -> dict | None:
    if cand is None:
        return None
    return {
        "strike_W": cand.strike_W,
        "strike_A": cand.strike_A,
        "q_strike_deg": np.degrees(cand.q_strike),
        "q_stop_deg": np.degrees(cand.q_stop),
        "qdot_strike_deg_s": np.degrees(cand.qdot_strike),
        "desired_v_W": cand.desired_v_W,
        "achieved_v_W": cand.achieved_v_W,
        "forward_mps": cand.forward_mps,
        "up_mps": cand.up_mps,
        "tti_original_s": cand.tti_s,
        "tti_aged_s": cand.aged_tti(),
        "min_strike_s": cand.min_strike_s,
        "strike_time_s": cand.strike_time_s,
        "budget_margin_s": cand.budget_margin_s,
        "cmd_peak_frac": cand.cmd_peak_frac,
        "ik_err_mm": cand.ik_err_mm,
        "ori_err_deg": cand.ori_err_deg,
        "decel_s": cand.decel_s,
    }


def _read_xml_velocity_limits(r: redis.Redis) -> np.ndarray:
    limits = J6_XML_VEL_LIMIT_RAD_S.copy()
    key = f"{NS}::{JOINT_CTRL}::joint_task::velocity_saturation_limit"
    raw = r.get(key)
    if raw is None:
        print(f"[J9.5] WARNING: velocity limits key not found ({key}); using hardcoded defaults")
        return limits
    try:
        parsed = np.asarray(json.loads(raw), dtype=float)
        if parsed.shape == (7,):
            print(f"[J9.5] velocity limits from OpenSai: {np.degrees(parsed).round(1).tolist()} deg/s")
            return parsed
    except Exception:
        pass
    print("[J9.5] WARNING: could not parse velocity limits from Redis; using hardcoded defaults")
    return limits


def _sweet_spot_A(chain, q: np.ndarray, offset_link7: np.ndarray) -> np.ndarray:
    T = _fk(chain, q)
    return T[:3, 3] + T[:3, :3] @ offset_link7


def _fr3_raw_velocity_limits_rad(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    q = np.asarray(q, dtype=float)
    dq_max = np.empty(7)
    dq_max[0] = min(3.0, max(0.0, -0.3 + math.sqrt(max(0.0, 12.0 * (2.75010 - q[0])))))
    dq_max[1] = min(3.0, max(0.0, -0.2 + math.sqrt(max(0.0, 5.17 * (1.79180 - q[1])))))
    dq_max[2] = min(3.0, max(0.0, -0.2 + math.sqrt(max(0.0, 7.00 * (2.90650 - q[2])))))
    dq_max[3] = min(3.0, max(0.0, -0.3 + math.sqrt(max(0.0, 8.00 * (-0.1458 - q[3])))))
    dq_max[4] = min(3.0, max(0.0, -0.35 + math.sqrt(max(0.0, 34.0 * (2.81010 - q[4])))))
    dq_max[5] = min(3.0, max(0.0, -0.35 + math.sqrt(max(0.0, 11.0 * (4.52050 - q[5])))))
    dq_max[6] = min(3.0, max(0.0, -0.35 + math.sqrt(max(0.0, 34.0 * (3.01960 - q[6])))))

    dq_min = np.empty(7)
    dq_min[0] = max(-3.0, min(0.0, 0.3 - math.sqrt(max(0.0, 12.0 * (2.75010 + q[0])))))
    dq_min[1] = max(-3.0, min(0.0, 0.2 - math.sqrt(max(0.0, 5.17 * (1.79180 + q[1])))))
    dq_min[2] = max(-3.0, min(0.0, 0.2 - math.sqrt(max(0.0, 7.00 * (2.90650 + q[2])))))
    dq_min[3] = max(-3.0, min(0.0, 0.3 - math.sqrt(max(0.0, 8.00 * (3.04810 + q[3])))))
    dq_min[4] = max(-3.0, min(0.0, 0.35 - math.sqrt(max(0.0, 34.0 * (2.81010 + q[4])))))
    dq_min[5] = max(-3.0, min(0.0, 0.35 - math.sqrt(max(0.0, 11.0 * (-0.54092 + q[5])))))
    dq_min[6] = max(-3.0, min(0.0, 0.35 - math.sqrt(max(0.0, 34.0 * (3.01960 + q[6])))))
    return dq_min, dq_max


def _fr3_driver_velocity_bands_rad(q: np.ndarray) -> dict:
    raw_min, raw_max = _fr3_raw_velocity_limits_rad(q)
    # Mirrors drivers/FrankaPanda/redis_driver/main.cpp for FR3 with dynamic bounds:
    # hard zone = 6 * 0.1 rad/s, soft zone = 9 * 0.1 rad/s.
    hard = 0.6
    soft = 0.9
    return {
        "raw_min": raw_min,
        "hard_min": raw_min + hard,
        "soft_min": raw_min + soft,
        "soft_max": raw_max - soft,
        "hard_max": raw_max - hard,
        "raw_max": raw_max,
    }


def _velocity_band_snapshot(q: np.ndarray | None, qdot: np.ndarray | None = None) -> dict | None:
    if q is None:
        return None
    bands = _fr3_driver_velocity_bands_rad(q)
    snap = {f"{k}_deg_s": np.degrees(v) for k, v in bands.items()}
    if qdot is not None:
        allowed_hard = np.where(qdot >= 0.0, bands["hard_max"], -bands["hard_min"])
        allowed_soft = np.where(qdot >= 0.0, bands["soft_max"], -bands["soft_min"])
        hard_ratio = np.abs(qdot) / np.maximum(allowed_hard, 1e-6)
        soft_ratio = np.abs(qdot) / np.maximum(allowed_soft, 1e-6)
        jh = int(np.argmax(hard_ratio))
        js = int(np.argmax(soft_ratio))
        snap.update({
            "qdot_deg_s": np.degrees(qdot),
            "hard_ratio": float(hard_ratio[jh]),
            "hard_joint": jh + 1,
            "soft_ratio": float(soft_ratio[js]),
            "soft_joint": js + 1,
        })
    return snap


def _driver_diag_snapshot(r: redis.Redis) -> dict:
    q = get_vec(r, SENSOR_JOINTS, 7)
    dq = get_vec(r, SENSOR_JOINT_VELS, 7)
    return {
        "active_controller": r.get(ACTIVE_CONTROLLER),
        "q_deg": None if q is None else np.degrees(q),
        "dq_deg_s": None if dq is None else np.degrees(dq),
        "fr3_velocity": _velocity_band_snapshot(q, dq),
        "tau_cmd": get_vec(r, COMMAND_TORQUES, 7),
        "tau_sent": get_vec(r, SENT_TORQUES, 7),
        "tau_safety": get_vec(r, SAFETY_TORQUES, 7),
        "tau_sensed": get_vec(r, SENSED_TORQUES, 7),
        "tau_desired": get_vec(r, f"{NS}::sensors::FrankaRobot::joint_torques_desired", 7),
        "tau_ext_hat_filtered": get_vec(r, f"{NS}::sensors::FrankaRobot::tau_ext_hat_filtered", 7),
    }


def _trajectory_diagnostics(chain, q_start: np.ndarray, cand: ImpactCandidate, offset_link7: np.ndarray) -> dict:
    p_start = _sweet_spot_A(chain, q_start, offset_link7)
    p_strike = _sweet_spot_A(chain, cand.q_strike, offset_link7)
    p_stop = _sweet_spot_A(chain, cand.q_stop, offset_link7)

    worst_hard = {"ratio": 0.0}
    worst_soft = {"ratio": 0.0}
    phases = (
        ("approach", max(0.045, cand.strike_time_s), q_start, np.zeros(7), cand.q_strike, cand.qdot_strike),
        ("decel", max(0.05, cand.decel_s), cand.q_strike, cand.qdot_strike, cand.q_stop, np.zeros(7)),
    )
    for phase, T, q0, v0, q1, v1 in phases:
        for t in np.linspace(0.0, T, 160):
            q, qdot = _hermite(q0, v0, q1, v1, float(t), T)
            bands = _fr3_driver_velocity_bands_rad(q)
            hard_allowed = np.where(qdot >= 0.0, bands["hard_max"], -bands["hard_min"])
            soft_allowed = np.where(qdot >= 0.0, bands["soft_max"], -bands["soft_min"])
            hard_ratio = np.abs(qdot) / np.maximum(hard_allowed, 1e-6)
            soft_ratio = np.abs(qdot) / np.maximum(soft_allowed, 1e-6)
            jh = int(np.argmax(hard_ratio))
            js = int(np.argmax(soft_ratio))
            if float(hard_ratio[jh]) > float(worst_hard["ratio"]):
                worst_hard = {
                    "ratio": float(hard_ratio[jh]),
                    "joint": jh + 1,
                    "phase": phase,
                    "t_s": float(t),
                    "q_deg": float(math.degrees(q[jh])),
                    "qdot_deg_s": float(math.degrees(qdot[jh])),
                    "hard_min_deg_s": float(math.degrees(bands["hard_min"][jh])),
                    "hard_max_deg_s": float(math.degrees(bands["hard_max"][jh])),
                }
            if float(soft_ratio[js]) > float(worst_soft["ratio"]):
                worst_soft = {
                    "ratio": float(soft_ratio[js]),
                    "joint": js + 1,
                    "phase": phase,
                    "t_s": float(t),
                    "q_deg": float(math.degrees(q[js])),
                    "qdot_deg_s": float(math.degrees(qdot[js])),
                    "soft_min_deg_s": float(math.degrees(bands["soft_min"][js])),
                    "soft_max_deg_s": float(math.degrees(bands["soft_max"][js])),
                }

    return {
        "sweet_start_A": p_start,
        "sweet_strike_A": p_strike,
        "sweet_stop_A": p_stop,
        "sweet_dx_pre_m": float(p_strike[0] - p_start[0]),
        "sweet_dx_after_m": float(p_stop[0] - p_strike[0]),
        "sweet_dz_after_m": float(p_stop[2] - p_strike[2]),
        "fr3_worst_hard": worst_hard,
        "fr3_worst_soft": worst_soft,
        "fr3_at_strike": _velocity_band_snapshot(cand.q_strike, cand.qdot_strike),
    }


def _sweet_spot_jacobian_A(chain, q: np.ndarray, offset_link7: np.ndarray) -> np.ndarray:
    J = np.zeros((3, 7))
    eps = 1e-4
    for i in range(7):
        dq = np.zeros(7)
        dq[i] = eps
        p_plus = _sweet_spot_A(chain, q + dq, offset_link7)
        p_minus = _sweet_spot_A(chain, q - dq, offset_link7)
        J[:, i] = (p_plus - p_minus) / (2.0 * eps)
    return J


def _solve_qdot_for_velocity(
    J_A: np.ndarray,
    desired_v_A: np.ndarray,
    qdot_cap: np.ndarray,
    damping: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    damp2 = max(1e-6, damping) ** 2
    qdot = J_A.T @ np.linalg.solve(J_A @ J_A.T + damp2 * np.eye(3), desired_v_A)
    peak = float(np.max(np.abs(qdot) / np.maximum(qdot_cap, 1e-6)))
    if peak > 1.0:
        qdot = qdot / peak
    qdot = np.clip(qdot, -qdot_cap, qdot_cap)
    achieved_v_A = J_A @ qdot
    return qdot, achieved_v_A, peak


def _make_impact_candidate(
    *,
    chain,
    q_cur: np.ndarray,
    q_seed: np.ndarray,
    intercept: Intercept,
    R_W_A: np.ndarray,
    t_W_A: np.ndarray,
    offset_link7: np.ndarray,
    R_A_link7_home: np.ndarray | None,
    forward_speed_mps: float,
    up_speed_mps: float,
    qdot_damping: float,
    decel_s: float,
    contact_margin_s: float,
    timing_guard_s: float,
    vel_cap: np.ndarray,
    impact_vel_frac: float,
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
) -> tuple[ImpactCandidate | None, str]:
    t_W_strike = np.asarray(intercept.position, dtype=float)
    tti = float(intercept.time_to_impact)
    if tti <= 0.0:
        return None, f"past impact tti={tti:.3f}s"
    if not (world_z_min_m <= t_W_strike[2] <= world_z_max_m):
        return None, f"strike_W.z={t_W_strike[2]:+.3f} outside [{world_z_min_m:.2f},{world_z_max_m:.2f}]"

    t_A_strike_raw = R_W_A.T @ (t_W_strike - t_W_A)
    if z_mode == "fixed-arm":
        t_A_strike_raw[2] = fixed_arm_z_m

    box_reason = _strike_box_reject_reason(
        t_A_strike_raw,
        x_min_m=strike_arm_x_min_m,
        x_max_m=strike_arm_x_max_m,
        y_abs_max_m=strike_arm_y_abs_max_m,
        z_min_m=strike_arm_z_min_m,
        z_max_m=strike_arm_z_max_m,
    )
    if box_reason is not None:
        return None, box_reason

    t_A_strike, clipped = _safe_target(t_A_strike_raw, reach_m, z_min, z_max)
    if clipped:
        return None, "clipped target strike"

    q_strike, err_st, ori_st = ik_solve(
        chain, t_A_strike, q_seed, offset_link7,
        R_A_link7_target=R_A_link7_home, w_ori=w_ori,
    )
    if (not _joints_ok(q_strike)) or (not _ik_ok(err_st, ori_st, ik_tol_m, max_ori_err_deg, R_A_link7_home)):
        return None, f"strike IK pos={err_st*1000:.1f}mm ori={ori_st:.1f}deg"

    desired_v_W = np.array([forward_speed_mps, 0.0, up_speed_mps], dtype=float)
    desired_v_A = R_W_A.T @ desired_v_W
    qdot_cap = np.maximum(vel_cap * max(0.05, impact_vel_frac), 1e-6)
    J_A = _sweet_spot_jacobian_A(chain, q_strike, offset_link7)
    qdot_strike, achieved_v_A, _raw_peak = _solve_qdot_for_velocity(
        J_A, desired_v_A, qdot_cap, qdot_damping,
    )
    achieved_v_W = R_W_A @ achieved_v_A

    # Short, disposable decel target. The impact velocity is the objective;
    # q_stop exists only to bleed off the swing after contact.
    q_stop = q_strike + 0.5 * qdot_strike * max(0.05, decel_s)
    q_stop = np.clip(q_stop, Q_LO + 0.02, Q_HI - 0.02)

    strike_time_s = tti + contact_margin_s - timing_guard_s
    if strike_time_s <= 0.035:
        return None, f"too late: strike_time={strike_time_s:.3f}s"

    min_s = _minimum_feasible_strike_time(q_cur, q_strike, q_stop, qdot_strike, decel_s, vel_cap)
    peak, peak_idx, peak_frac = _trajectory_peak_profile(
        q_cur, q_strike, q_stop, qdot_strike, max(0.035, strike_time_s), decel_s, vel_cap,
    )
    del peak, peak_idx
    return ImpactCandidate(
        strike_W=t_W_strike.copy(),
        strike_A=t_A_strike.copy(),
        q_strike=q_strike,
        q_stop=q_stop,
        qdot_strike=qdot_strike,
        desired_v_W=desired_v_W,
        achieved_v_W=achieved_v_W,
        tti_s=tti,
        observed_t=time.monotonic(),
        min_strike_s=min_s,
        strike_time_s=strike_time_s,
        budget_margin_s=strike_time_s - min_s,
        cmd_peak_frac=peak_frac,
        ik_err_mm=err_st * 1000.0,
        ori_err_deg=ori_st,
        decel_s=decel_s,
        forward_mps=float(achieved_v_W[0]),
        up_mps=float(achieved_v_W[2]),
    ), "ok"


def _run_impact_strike(
    r: redis.Redis,
    chain,
    cand: ImpactCandidate,
    offset_link7: np.ndarray,
    R_W_A: np.ndarray,
    *,
    publish_hz: float,
    full_segment_logs: bool,
    trace: _TraceLogger | None = None,
) -> dict:
    q_start = get_vec(r, SENSOR_JOINTS, 7)
    if q_start is None:
        q_start = cand.q_strike.copy()
    strike_s = max(0.045, cand.strike_time_s)
    decel_s = max(0.05, cand.decel_s)

    # --- Anti-hook clamp ---
    # A cubic Hermite from (q_start, v=0) to (q_strike, qdot_strike) has a "hook"
    # when sign(qdot_strike[j]) opposes sign(q_strike[j]-q_start[j]): the joint
    # must overshoot backward first, creating velocity spikes that trigger
    # joint_velocity_violation and large acceleration discontinuities.
    # Fix: zero the conflicting qdot_strike components so the arm goes straight in.
    disp = cand.q_strike - q_start
    # Clamp only hooks ≥ 0.1 rad (5.7°). Smaller opposing displacements create
    # backward velocity peaks well under the FR3 hard zone (e.g., 5° at 2 rad/s
    # over 0.5s → ~0.9 rad/s; FR3 hard limit ~2.4 rad/s). The old 5e-3 threshold
    # was clamping q6 for 1.1° conflicts, zeroing the main velocity contributor.
    hook_mask = (disp * cand.qdot_strike < 0) & (np.abs(disp) > 0.10)
    if np.any(hook_mask):
        cand.qdot_strike = cand.qdot_strike.copy()
        cand.qdot_strike[hook_mask] = 0.0
        hook_details = ", ".join(
            f"q{j+1}({math.degrees(disp[j]):+.1f}°)"
            for j, h in enumerate(hook_mask) if h
        )
        print(f"[J9.5 safe] anti-hook clamp on [{hook_details}]")

    # --- Initial velocity from sensors ---
    # During the aim phase J9 is already moving the arm toward q_strike, so at
    # commit the arm has real velocity. Using zeros as v0 throws that away and
    # forces the Hermite to restart from rest. Instead, read actual joint vels
    # and keep only the components already moving in the displacement direction
    # (toward q_strike). Opposing components are zeroed so we don't start with
    # momentum going the wrong way.
    v_sensor = get_vec(r, SENSOR_JOINT_VELS, 7)
    if v_sensor is not None:
        v_start = np.where(v_sensor * disp > 0, v_sensor, 0.0)
    else:
        v_start = np.zeros(7)

    # --- Acceleration-continuous q_stop ---
    # The approach Hermite's terminal acceleration at t=strike_s and the decel
    # Hermite's initial acceleration can differ sharply, triggering the
    # joint_motion_generator_acceleration_discontinuity reflex.
    # Choose q_stop so the decel Hermite begins with the same acceleration as
    # the approach Hermite ends with, giving C2 continuity at the transition.
    qddot_approach_end = (
        -6.0 * (cand.q_strike - q_start) / strike_s**2
        + 2.0 * v_start / strike_s
        + 4.0 * cand.qdot_strike / strike_s
    )
    q_stop_cont = cand.q_strike + (
        (qddot_approach_end + 4.0 * cand.qdot_strike / decel_s) * decel_s**2 / 6.0
    )
    cand.q_stop = np.clip(q_stop_cont, Q_LO + 0.02, Q_HI - 0.02)

    total_s = strike_s + decel_s
    period_s = 1.0 / max(1.0, publish_hz)
    sample_period_s = 0.01
    zeros = np.zeros(7)
    qdot_peak = np.zeros(7)
    qdot_cmd_peak = np.zeros(7)
    strike_err_deg = float("nan")
    stop_err_deg = float("nan")
    measured_v_W = np.full(3, np.nan)
    strike_sampled = False
    prev_q = None
    prev_t = None
    next_publish = time.perf_counter()
    next_sample = next_publish
    t0 = time.perf_counter()
    next_trace_sample = t0

    def tr(event: str, **fields) -> None:
        if trace is not None:
            trace.write(event, **fields)

    tr("impact_start", candidate=_candidate_snapshot(cand), diagnostic=_driver_diag_snapshot(r))

    print(
        f"[J9.5 impact] start: strike_s={strike_s:.3f}s decel_s={decel_s:.3f}s "
        f"v_cmd_W=[{cand.achieved_v_W[0]:+.2f},{cand.achieved_v_W[1]:+.2f},{cand.achieved_v_W[2]:+.2f}] m/s"
    )
    if full_segment_logs:
        print("[J9.5 impact] q_start:  " + _fmt_q(q_start))
        print("[J9.5 impact] v_start:  " + _fmt_q(v_start))
        print("[J9.5 impact] q_strike: " + _fmt_q(cand.q_strike))
        print("[J9.5 impact] qdot_hit: " + _fmt_q(cand.qdot_strike))
        print("[J9.5 impact] q_stop:   " + _fmt_q(cand.q_stop))

    while True:
        now = time.perf_counter()
        elapsed = now - t0
        if elapsed <= strike_s:
            q_goal, qdot_goal = _hermite(q_start, v_start, cand.q_strike, cand.qdot_strike, elapsed, strike_s)
        else:
            q_goal, qdot_goal = _hermite(cand.q_strike, cand.qdot_strike, cand.q_stop, zeros, elapsed - strike_s, decel_s)
        qdot_cmd_peak = np.maximum(qdot_cmd_peak, np.abs(np.degrees(qdot_goal)))

        if now >= next_publish:
            set_vec(r, GOAL_JOINTS, q_goal)
            next_publish = now + period_s

        if (not strike_sampled) and elapsed >= strike_s:
            q_hit = get_vec(r, SENSOR_JOINTS, 7)
            dq_hit = get_vec(r, SENSOR_JOINT_VELS, 7)
            if q_hit is not None:
                strike_err_deg = float(np.max(np.abs(np.degrees(cand.q_strike - q_hit))))
                if dq_hit is not None:
                    J_A = _sweet_spot_jacobian_A(chain, q_hit, offset_link7)
                    measured_v_W = R_W_A @ (J_A @ dq_hit)
            strike_sampled = True

        if now >= next_sample:
            q = get_vec(r, SENSOR_JOINTS, 7)
            dq = get_vec(r, SENSOR_JOINT_VELS, 7)
            if dq is not None:
                qdot_peak = np.maximum(qdot_peak, np.abs(np.degrees(dq)))
            elif q is not None and prev_q is not None and prev_t is not None:
                dt = max(1e-6, now - prev_t)
                qdot_peak = np.maximum(qdot_peak, np.abs(np.degrees((q - prev_q) / dt)))
            if trace is not None and now >= next_trace_sample:
                tr(
                    "impact_sample",
                    elapsed_s=elapsed,
                    phase="approach" if elapsed <= strike_s else "decel",
                    q_goal_deg=np.degrees(q_goal),
                    qdot_goal_deg_s=np.degrees(qdot_goal),
                    q_actual_deg=None if q is None else np.degrees(q),
                    dq_actual_deg_s=None if dq is None else np.degrees(dq),
                    cmd_fr3_velocity=_velocity_band_snapshot(q_goal, qdot_goal),
                    actual_fr3_velocity=None if q is None else _velocity_band_snapshot(q, dq),
                    diagnostic=_driver_diag_snapshot(r),
                )
                next_trace_sample = now + 0.02
            if q is not None:
                prev_q = q.copy()
                prev_t = now
            next_sample = now + sample_period_s

        if elapsed >= total_s:
            break
        time.sleep(0.002)

    q_final = get_vec(r, SENSOR_JOINTS, 7)
    if q_final is not None:
        stop_err_deg = float(np.max(np.abs(np.degrees(cand.q_stop - q_final))))
    qdot_i = int(np.argmax(qdot_peak))
    qdot_cmd_i = int(np.argmax(qdot_cmd_peak))
    status = "OK" if (not math.isfinite(strike_err_deg) or strike_err_deg <= 8.0) else "WARN"
    print(
        f"[J9.5 segment] impact: {status} strike_err={strike_err_deg:.2f}deg "
        f"stop_err={stop_err_deg:.2f}deg qdot_max={qdot_peak[qdot_i]:.1f}deg/s@q{qdot_i+1} "
        f"cmd_peak={qdot_cmd_peak[qdot_cmd_i]:.1f}deg/s@q{qdot_cmd_i+1} "
        f"v_meas_W=[{measured_v_W[0]:+.2f},{measured_v_W[1]:+.2f},{measured_v_W[2]:+.2f}]"
    )
    result = {
        "strike_err_deg": strike_err_deg,
        "stop_err_deg": stop_err_deg,
        "qdot_max_deg_s": float(qdot_peak[qdot_i]),
        "qdot_max_joint": qdot_i + 1,
        "cmd_qdot_max_deg_s": float(qdot_cmd_peak[qdot_cmd_i]),
        "cmd_qdot_max_joint": qdot_cmd_i + 1,
        "measured_v_W": measured_v_W,
    }
    tr("impact_done", result=result, diagnostic=_driver_diag_snapshot(r))
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
    decel_s: float,
    commit_tti: float,
    contact_margin_s: float,
    timing_guard_s: float,
    early_strike_lead_s: float,
    min_planned_strike_s: float,
    max_planned_strike_s: float,
    try_late_slack_s: float,
    max_stretched_strike_s: float,
    return_s: float,
    return_vel_frac: float,
    rate_hz: float,
    no_commit: bool,
    reach_m: float,
    z_min: float,
    z_max: float,
    world_z_min_m: float,
    world_z_max_m: float,
    tracking_step_deg: float,
    tracking_accel_deg_s2: float,
    swing_vel_frac: float,
    impact_vel_frac: float,
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
    trace: _TraceLogger | None,
    ball_key: str | None,
) -> None:
    dt = 1.0 / max(1.0, rate_hz)
    using_mock = mock_intercepts is not None and len(mock_intercepts) > 0
    mock_t0 = time.monotonic()
    vel_cap = np.maximum(xml_vel_limits * max(0.05, swing_vel_frac), 1e-6)
    last_q_cmd: np.ndarray | None = get_vec(r, SENSOR_JOINTS, 7)
    last_q_cmd_vel = np.zeros(7)
    last_goal_write_t = time.perf_counter()
    max_goal_gap_s = max(0.03, 2.5 * dt)
    last_print_t = float("-inf")
    last_reject_print_t = float("-inf")
    reject_counts: dict[str, int] = {}
    active_cand: ImpactCandidate | None = None
    active_throw_id = 0
    swings_done = 0
    idle_until_t = 0.0
    tick_i = 0

    def tr(event: str, **fields) -> None:
        if trace is not None:
            trace.write(event, **fields)

    def print_reject(reason: str) -> None:
        nonlocal last_reject_print_t
        reject_counts[reason] = reject_counts.get(reason, 0) + 1
        tr("reject", tick=tick_i, reason=reason, count=reject_counts.get(reason, 0))
        now = time.perf_counter()
        if now - last_reject_print_t >= 0.35:
            last_reject_print_t = now
            print(f"[J9.5 wait] {reason}")

    def publish_limited_goal(target: np.ndarray, q_cur: np.ndarray, mode: str) -> None:
        nonlocal last_q_cmd, last_q_cmd_vel, last_goal_write_t
        now_goal_t = time.perf_counter()
        raw_elapsed_s = max(0.0, now_goal_t - last_goal_write_t)
        stale_goal = last_q_cmd is None or raw_elapsed_s > max_goal_gap_s
        cmd_state_error_deg = (
            float(np.max(np.abs(np.degrees(last_q_cmd - q_cur))))
            if last_q_cmd is not None else float("inf")
        )
        measured_reset = stale_goal or cmd_state_error_deg > max(2.0 * tracking_step_deg, 3.0)
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
            q_cmd = _rate_limited_goal(
                target, q_base, tracking_step_deg * max(1.0, elapsed_goal_s * rate_hz),
            )
        set_vec(r, GOAL_JOINTS, q_cmd)
        tr(
            "goal_write", tick=tick_i, target_mode=mode,
            stale_goal_reset=stale_goal, measured_state_reset=measured_reset,
            cmd_state_error_deg=cmd_state_error_deg,
            q_target_deg=np.degrees(target), q_cmd_deg=np.degrees(q_cmd),
            q_cur_deg=np.degrees(q_cur), last_q_cmd_vel_deg_s=np.degrees(last_q_cmd_vel),
        )
        last_q_cmd = q_cmd.copy()
        last_goal_write_t = now_goal_t

    def recover_home(label: str, idle_after: bool) -> bool:
        nonlocal active_cand, last_q_cmd, last_q_cmd_vel, last_goal_write_t, idle_until_t
        tr(
            "recover_home_start", tick=tick_i, label=label,
            active_candidate=_candidate_snapshot(active_cand),
            diagnostic=_driver_diag_snapshot(r),
        )

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

    switch_ctrl(r, JOINT_CTRL)
    _hold_current_joints(r)
    print(f"[J9.5] running at {rate_hz:.0f} Hz -- high-commit impact-velocity striker")
    print(
        f"[J9.5] strike_plane_x={strike_plane_x_world:+.3f}m commit_tti={commit_tti:.3f}s "
        f"lead={early_strike_lead_s:.3f}s planned_s=[{min_planned_strike_s:.3f},{max_planned_strike_s:.3f}] "
        f"guard={timing_guard_s:.3f}s"
    )
    print(
        f"[J9.5] impact velocity target W: forward=+{forward_speed_mps:.2f} m/s, "
        f"up=+{up_speed_mps:.2f} m/s, decel_s={decel_s:.2f}s"
    )
    print(
        f"[J9.5] strike box A: x=[{strike_arm_x_min_m:+.2f},{strike_arm_x_max_m:+.2f}] "
        f"|y|<={strike_arm_y_abs_max_m:.2f} z=[{strike_arm_z_min_m:+.2f},{strike_arm_z_max_m:+.2f}] z_mode={z_mode}"
    )
    print("[J9.5] policy: aim immediately, freeze early, execute a shorter smooth swing-through")
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
            active_candidate=_candidate_snapshot(active_cand),
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
            publish_limited_goal(q_home_rad, q_cur, "idle_ready")
            active_cand = None
            time.sleep(max(0.0, dt - (time.perf_counter() - loop_t)))
            continue

        q_seed = active_cand.q_strike if active_cand is not None else q_cur
        cand, cand_reason = _make_impact_candidate(
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
            decel_s=decel_s,
            contact_margin_s=contact_margin_s,
            timing_guard_s=timing_guard_s,
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
        tr("candidate", tick=tick_i, reason=cand_reason, candidate=_candidate_snapshot(cand), intercept=_intercept_snapshot(intercept))
        if cand is None:
            print_reject(cand_reason)
            publish_limited_goal(q_home_rad, q_cur, "reject_ready")
            time.sleep(max(0.0, dt - (time.perf_counter() - loop_t)))
            continue

        active_cand = cand
        aged_tti = cand.aged_tti()
        ball_strike_time_now = aged_tti + contact_margin_s - timing_guard_s
        strike_time_now = ball_strike_time_now - max(0.0, early_strike_lead_s)
        if max_planned_strike_s > 0.0:
            strike_time_now = min(strike_time_now, max_planned_strike_s)
        strike_time_now = max(max(0.045, min_planned_strike_s), strike_time_now)
        cand.strike_time_s = strike_time_now
        cand.min_strike_s = _minimum_feasible_strike_time(
            q_cur, cand.q_strike, cand.q_stop, cand.qdot_strike, cand.decel_s, vel_cap,
        )
        cand.budget_margin_s = strike_time_now - cand.min_strike_s
        peak, peak_idx, peak_frac = _trajectory_peak_profile(
            q_cur, cand.q_strike, cand.q_stop, cand.qdot_strike,
            max(0.035, strike_time_now), cand.decel_s, vel_cap,
        )
        cand.cmd_peak_frac = peak_frac
        traj_diag = _trajectory_diagnostics(chain, q_cur, cand, offset_link7)
        q_delta, q_joint = _max_abs_deg_with_joint(cand.q_strike - q_cur)
        hw_hard_ratio = traj_diag["fr3_worst_hard"]["ratio"]
        hw_ok = hw_hard_ratio <= 1.0
        feasible = strike_time_now > 0.045 and cand.budget_margin_s >= 0.0 and peak_frac <= 1.0 and hw_ok
        late_by_s = cand.min_strike_s - strike_time_now
        late_try = (
            strike_time_now > 0.045
            and aged_tti <= commit_tti
            and cand.min_strike_s <= max_stretched_strike_s
            and (try_late_slack_s <= 0.0 or late_by_s <= try_late_slack_s)
        )
        commit_now = (not no_commit) and aged_tti <= commit_tti and (feasible or late_try) and hw_ok
        state = "AIM"
        if aged_tti <= commit_tti:
            if not hw_ok:
                state = "HW_LIMIT"
            elif feasible:
                state = "COMMIT"
            elif late_try:
                state = "TRY"
            else:
                state = "LATE"

        tr(
            "candidate_runtime", tick=tick_i, state=state, feasible=feasible, late_try=late_try,
            aged_tti_s=aged_tti, ball_strike_time_s=ball_strike_time_now, strike_time_s=strike_time_now,
            early_strike_lead_s=early_strike_lead_s, min_strike_s=cand.min_strike_s, margin_s=cand.budget_margin_s,
            late_by_s=late_by_s,
            cmd_peak_frac=peak_frac, bottleneck_joint=peak_idx + 1,
            bottleneck_cmd_deg_s=math.degrees(peak[peak_idx]), q_to_strike_deg=q_delta,
            q_to_strike_joint=q_joint, q_cur_deg=np.degrees(q_cur),
            trajectory_diagnostics=traj_diag,
            candidate=_candidate_snapshot(cand),
        )

        if loop_t - last_print_t >= max(0.05, log_period_s):
            last_print_t = loop_t
            qdot = get_vec(r, SENSOR_JOINT_VELS, 7)
            qdot_txt = ""
            if verbose_tracking and qdot is not None:
                qdot_deg, qdot_joint = _max_abs_deg_with_joint(qdot)
                qdot_txt = f" qdot={qdot_deg:.1f}deg/s@q{qdot_joint}"
            print(
                f"[J9.5] {state:6s} tti={aged_tti:+.3f}s strike_W={_fmt_v3(cand.strike_W)} "
                f"strike_A={_fmt_v3(cand.strike_A)} ball_s={ball_strike_time_now:.3f}s Tmin={cand.min_strike_s:.3f}s "
                f"swing_s={strike_time_now:.3f}s margin={cand.budget_margin_s:+.3f}s "
                f"cmd={peak_frac*100:.0f}% vW=[{cand.forward_mps:+.2f},{cand.up_mps:+.2f}]"
            )
            if verbose_tracking:
                hard = traj_diag["fr3_worst_hard"]
                soft = traj_diag["fr3_worst_soft"]
                print(
                    f"[J9.5]   q_to_strike={q_delta:.1f}deg@q{q_joint} "
                    f"peak_cmd={math.degrees(peak[peak_idx]):.1f}deg/s@q{peak_idx+1} "
                    f"ik={cand.ik_err_mm:.1f}mm ori={cand.ori_err_deg:.1f}deg "
                    f"path_dx={100*traj_diag['sweet_dx_pre_m']:+.1f}->{100*traj_diag['sweet_dx_after_m']:+.1f}cm "
                    f"fr3_hard={100*hard['ratio']:.0f}%@q{hard['joint']} "
                    f"soft={100*soft['ratio']:.0f}%@q{soft['joint']} "
                    f"rejects={_fmt_counts(reject_counts)}{qdot_txt}"
                )

        if commit_now:
            if late_try and not feasible:
                old_strike_s = strike_time_now
                cand.strike_time_s = cand.min_strike_s
                cand.budget_margin_s = 0.0
                peak, peak_idx, peak_frac = _trajectory_peak_profile(
                    q_cur, cand.q_strike, cand.q_stop, cand.qdot_strike,
                    cand.strike_time_s, cand.decel_s, vel_cap,
                )
                cand.cmd_peak_frac = peak_frac
                traj_diag = _trajectory_diagnostics(chain, q_cur, cand, offset_link7)
                print(
                    f"[J9.5] TRY_STRETCH: planned_s={old_strike_s:.3f}s < Tmin={cand.min_strike_s:.3f}s; "
                    f"swinging safely over {cand.strike_time_s:.3f}s"
                )
            active_throw_id += 1
            tr(
                "commit_start", tick=tick_i, throw_id=active_throw_id,
                candidate=_candidate_snapshot(cand), late_try=late_try,
                trajectory_diagnostics=traj_diag, diagnostic=_driver_diag_snapshot(r),
            )
            print("\n" + "=" * 64)
            print(
                f"[J9.5] COMMIT throw {active_throw_id}: tti={aged_tti:.3f}s "
                f"ball_s={ball_strike_time_now:.3f}s swing_s={cand.strike_time_s:.3f}s Tmin={cand.min_strike_s:.3f}s "
                f"margin={cand.budget_margin_s:+.3f}s cmd_peak={cand.cmd_peak_frac*100:.0f}%"
            )
            print(
                f"[J9.5]   strike_W={_fmt_v3(cand.strike_W)} strike_A={_fmt_v3(cand.strike_A)} "
                f"v_W desired={_fmt_v3(cand.desired_v_W)} achieved={_fmt_v3(cand.achieved_v_W)}"
            )
            hard = traj_diag["fr3_worst_hard"]
            soft = traj_diag["fr3_worst_soft"]
            print(
                f"[J9.5]   q_to_strike={q_delta:.1f}deg@q{q_joint} "
                f"bottleneck q{peak_idx+1}: cmd={math.degrees(peak[peak_idx]):.1f}deg/s"
            )
            print(
                f"[J9.5]   path_A start={_fmt_v3(traj_diag['sweet_start_A'])} "
                f"strike={_fmt_v3(traj_diag['sweet_strike_A'])} stop={_fmt_v3(traj_diag['sweet_stop_A'])} "
                f"dx={100*traj_diag['sweet_dx_pre_m']:+.1f}->{100*traj_diag['sweet_dx_after_m']:+.1f}cm "
                f"fr3_hard={100*hard['ratio']:.0f}%@q{hard['joint']} "
                f"soft={100*soft['ratio']:.0f}%@q{soft['joint']}"
            )
            print("=" * 64 + "\n")
            result = _run_impact_strike(
                r, chain, cand, offset_link7, R_W_A,
                publish_hz=200.0,
                full_segment_logs=full_segment_logs,
                trace=trace,
            )
            if _safety_tripped(r):
                print("[J9.5] safety torque after impact; recovering slowly")
            home_ok = recover_home("post-impact->home", idle_after=True)
            swings_done += 1
            outcome = "OK" if home_ok and result["strike_err_deg"] <= 8.0 else "WARN"
            print(
                f"[J9.5 audit] throw={active_throw_id} outcome={outcome} "
                f"timing[tti={aged_tti:.3f}s ball_s={ball_strike_time_now:.3f}s swing_s={cand.strike_time_s:.3f}s "
                f"Tmin={cand.min_strike_s:.3f}s margin={cand.budget_margin_s:+.3f}s] "
                f"arm[strike_err={result['strike_err_deg']:.2f}deg stop_err={result['stop_err_deg']:.2f}deg "
                f"home_ok={home_ok}] tracker_rejects={_fmt_counts(reject_counts)}"
            )
            tr("commit_done", tick=tick_i, throw_id=active_throw_id, outcome=outcome, result=result, home_ok=home_ok, reject_counts=reject_counts)
            reject_counts = {}
            active_cand = None
            if max_swings > 0 and swings_done >= max_swings:
                print(f"[J9.5] --max-swings={max_swings} reached; exiting")
                break
            time.sleep(max(0.0, dt - (time.perf_counter() - loop_t)))
            continue

        # High-commit J9: always aim directly at the latest strike pose. No pre-pose.
        publish_limited_goal(cand.q_strike, q_cur, "aim_strike")
        if aged_tti < -lock_release_after_impact_s:
            tr("missed_window", tick=tick_i, aged_tti_s=aged_tti, candidate=_candidate_snapshot(cand))
            active_cand = None
            idle_until_t = time.monotonic() + post_impact_idle_s
        time.sleep(max(0.0, dt - (time.perf_counter() - loop_t)))


def _load_config_yaml(path: str) -> dict:
    try:
        import yaml
    except ImportError:
        sys.exit("[J9.5] --config requires PyYAML: pip install pyyaml")
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    # YAML keys use underscores matching argparse dest names
    return {str(k): v for k, v in data.items()}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="J9.5: early smooth impact-velocity pickleball striker.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--config", default=None, metavar="YAML",
                    help="YAML config file; sets argument defaults (CLI flags override).")
    ap.add_argument("--ball-rigid-body-id", type=int, default=13)  # RigidBody002
    ap.add_argument("--strike-plane-x", type=float, default=-0.28)
    ap.add_argument("--mock-intercept", nargs=3, type=float, action="append")
    ap.add_argument("--mock-tti", type=float, default=0.60)
    ap.add_argument("--mock-cycle-s", type=float, default=0.0)
    ap.add_argument("--mock-identity-base", action="store_true")

    ap.add_argument("--paddle-open-deg", type=float, default=8.0)
    ap.add_argument("--forward-speed-mps", type=float, default=1.50)
    ap.add_argument("--up-speed-mps", type=float, default=0.45)
    ap.add_argument("--impact-vel-frac", type=float, default=0.95)
    ap.add_argument("--qdot-damping", type=float, default=0.03)
    ap.add_argument("--decel-s", type=float, default=0.18)
    ap.add_argument("--commit-tti", type=float, default=0.75)
    ap.add_argument("--try-late-slack-s", type=float, default=999.0)
    ap.add_argument("--max-stretched-strike-s", type=float, default=0.90)
    ap.add_argument("--contact-margin-s", type=float, default=0.0)
    ap.add_argument("--timing-guard-s", type=float, default=0.015)
    ap.add_argument("--early-strike-lead-s", type=float, default=0.10,
                    help="Plan the smooth swing to pass q_strike this much before predicted plane time.")
    ap.add_argument("--min-planned-strike-s", type=float, default=0.20,
                    help="Minimum approach duration for the committed smooth swing.")
    ap.add_argument("--max-planned-strike-s", type=float, default=0.42,
                    help="Maximum approach duration; shorter than ball TTI makes the swing faster once frozen.")
    ap.add_argument("--swing-vel-frac", type=float, default=1.00)
    ap.add_argument("--vel-cap-rad-s", nargs=7, type=float, default=None,
                    metavar=("Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7"),
                    help="Override joint velocity caps (rad/s) used for trajectory planning. "
                         "Required when using picklebot_j9_fast.xml (vel sat disabled) to push "
                         "past the conservative XML defaults. Target ~2.0-2.5 rad/s from home pose.")

    ap.add_argument("--return-s", type=float, default=5.0)
    ap.add_argument("--return-vel-frac", type=float, default=0.30)
    ap.add_argument("--home-s", type=float, default=3.0)
    ap.add_argument("--home-hold-s", type=float, default=1.0)
    ap.add_argument("--home-vel-frac", type=float, default=J6_HOME_VEL_FRAC)
    ap.add_argument("--home-joints-deg", nargs=7, type=float, default=None,
                    help="Override the imported J7 home/ready pose with seven joint angles in degrees.")
    ap.add_argument("--ready-pose", choices=["home", "strike-center"], default="strike-center")
    ap.add_argument("--ready-arm-x", type=float, default=0.52)
    ap.add_argument("--ready-arm-y", type=float, default=0.0)
    ap.add_argument("--ready-arm-z", type=float, default=0.24)

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
    ap.add_argument("--max-ori-err-deg", type=float, default=15.0)

    ap.add_argument("--tracking-step-deg", type=float, default=1.5)
    ap.add_argument("--tracking-accel-deg-s2", type=float, default=700.0)
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
    ap.add_argument("--tracker-gravity", type=float, default=J95_TRACKER_GRAVITY_MPS2,
                    help="Foam-ball prediction gravity; tuned hybrid default from clean J9 recordings.")
    ap.add_argument("--tracker-drag-coefficient", type=float, default=J95_TRACKER_DRAG_COEFFICIENT,
                    help="Foam-ball drag coefficient k in dv/dt = -k*|v|*v.")
    ap.add_argument("--tracker-simulation-dt", type=float, default=J95_TRACKER_SIMULATION_DT_S,
                    help="Drag propagator integration dt in seconds.")
    ap.add_argument("--tracker-history-size", type=int, default=None)
    ap.add_argument("--tracker-history-max-age-s", type=float, default=None)
    ap.add_argument("--tracker-min-history", type=int, default=J95_TRACKER_MIN_HISTORY)
    ap.add_argument("--tracker-median-window", type=int, default=None)
    ap.add_argument("--tracker-max-implied-speed-mps", type=float, default=None)
    ap.add_argument("--tracker-stale-position-eps-m", type=float, default=None)
    ap.add_argument("--tracker-stale-timeout-s", type=float, default=None)

    # Load YAML defaults before full parse so explicit CLI flags still win
    _pre = argparse.ArgumentParser(add_help=False)
    _pre.add_argument("--config", default=None)
    _pre_args, _ = _pre.parse_known_args()
    if _pre_args.config is not None:
        ap.set_defaults(**_load_config_yaml(_pre_args.config))

    args = ap.parse_args()
    run_stamp = time.strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.abspath(os.path.join(_THIS_DIR, "..", "logs"))

    if args.log_file is not None:
        log_path = args.log_file
        if log_path == "auto":
            log_path = os.path.join(log_dir, f"j95_{run_stamp}.log")
        log_path = os.path.abspath(log_path)
        sys.stdout = _Tee(log_path)
        print(f"[J9.5] logging to {log_path}")

    trace_path = None
    if args.trace_file.lower() != "off":
        trace_path = args.trace_file
        if trace_path == "auto":
            trace_path = os.path.join(log_dir, f"j95_{run_stamp}.trace.jsonl")
        trace_path = os.path.abspath(trace_path)
    trace = _TraceLogger(trace_path)
    if trace_path:
        print(f"[J9.5] trace logging to {trace_path}")

    r = redis.Redis(host=args.redis_host, port=args.redis_port, decode_responses=True)
    try:
        r.ping()
    except redis.exceptions.ConnectionError as exc:
        sys.exit(f"[J9.5] cannot reach Redis at {args.redis_host}:{args.redis_port}: {exc}")

    xml_vel_limits = _read_xml_velocity_limits(r)
    if args.vel_cap_rad_s is not None:
        xml_vel_limits = np.asarray(args.vel_cap_rad_s, dtype=float)
        print(
            f"[J9.5] vel_cap override (--vel-cap-rad-s): "
            f"{np.degrees(xml_vel_limits).round(1).tolist()} deg/s"
        )
    q_start_check = get_vec(r, SENSOR_JOINTS, 7)
    if q_start_check is None or (not args.allow_zero_joint_start and np.max(np.abs(q_start_check)) < 1e-6):
        sys.exit(
            "[J9.5] refusing to start: sensed joint state is missing or exactly zero. "
            "Recover/relaunch the Franka driver/OpenSai after a reflex abort first."
        )

    cal_path = args.calibration or arm_base_offset_calibration_path()
    if not os.path.isfile(cal_path):
        sys.exit(f"[J9.5] arm base calibration not found: {cal_path}")
    cal = load_arm_base_offset_calibration(cal_path)
    print(f"[J9.5] arm base calibration: {cal_path} (rigid body {cal.base_rigid_body_id})")

    mock_intercepts = None
    if args.mock_intercept:
        mock_intercepts = [np.asarray(p, dtype=float) for p in args.mock_intercept]
    if not (mock_intercepts and args.mock_identity_base):
        if read_rigid_body_pose_W(r, cal.base_rigid_body_id) is None:
            sys.exit(f"[J9.5] cart rigid body {cal.base_rigid_body_id} not visible in Redis")

    if not os.path.isfile(URDF_PATH):
        sys.exit(f"[J9.5] URDF not found: {URDF_PATH} (run from OpenSai root)")
    chain = build_chain()

    if args.skip_cal:
        offset_link7 = np.asarray(args.offset_link7, dtype=float) if args.offset_link7 is not None else J6_DEFAULT_OFFSET_LINK7.copy()
        _, offset_link7 = seed_offset_link7_no_cartesian(r, offset_link7)
    else:
        _, offset_link7 = calibrate_offset_link7(r, chain)

    if args.print_cal_only:
        print("[J9.5] --print-cal-only: done.")
        return

    q_ready_seed = Q_HOME_RAD.copy()
    if args.home_joints_deg is not None:
        q_ready_seed = np.radians(np.asarray(args.home_joints_deg, dtype=float))
        q_delta, q_joint = _max_abs_deg_with_joint(q_ready_seed - Q_HOME_RAD)
        print(
            f"[J9.5] custom home pose: {_fmt_q(q_ready_seed)} "
            f"(delta imported home {q_delta:.1f}deg@q{q_joint})"
        )

    R_A_link7_home = None
    if not args.no_fixed_paddle_ori:
        R_home_nominal = _link7_R_A(chain, q_ready_seed)
        R_A_link7_home = _rot_y_rad(math.radians(-args.paddle_open_deg)) @ R_home_nominal
        face_A = R_A_link7_home @ (offset_link7 / max(np.linalg.norm(offset_link7), 1e-9))
        print("[J9.5] fixed paddle orientation from ready/home pose")
        print(
            f"[J9.5]   paddle_open={args.paddle_open_deg:+.1f}deg "
            f"strike-face normal_A=[{face_A[0]:+.3f},{face_A[1]:+.3f},{face_A[2]:+.3f}]"
        )

    ready_arm_A = None
    q_ready = q_ready_seed.copy()
    if args.ready_pose == "strike-center":
        ready_arm_A = np.array([args.ready_arm_x, args.ready_arm_y, args.ready_arm_z], dtype=float)
        q_candidate, ready_err, ready_ori = ik_solve(
            chain, ready_arm_A, q_ready, offset_link7,
            R_A_link7_target=R_A_link7_home, w_ori=args.w_ori,
        )
        if _joints_ok(q_candidate) and _ik_ok(ready_err, ready_ori, args.ik_tol_mm / 1000.0, args.max_ori_err_deg, R_A_link7_home):
            q_ready = q_candidate
            q_delta, q_joint = _max_abs_deg_with_joint(q_ready - Q_HOME_RAD)
            print(
                f"[J9.5] strike-center ready IK: arm_A={_fmt_v3(ready_arm_A)} "
                f"pos={ready_err*1000:.1f}mm ori={ready_ori:.1f}deg delta_home={q_delta:.1f}deg@q{q_joint}"
            )
        else:
            print(
                f"[J9.5] WARNING: strike-center ready IK failed pos={ready_err*1000:.1f}mm "
                f"ori={ready_ori:.1f}deg; using home pose"
            )
            ready_arm_A = None

    print(f"[J9.5] moving to ready pose ({args.ready_pose}) ...")
    q_home_rad = move_to_home_pose(
        r, q_ready,
        move_s=args.home_s,
        hold_s=args.home_hold_s,
        vel_frac=args.home_vel_frac,
    )

    tracker = None
    ball_key = None
    if mock_intercepts is None:
        cfg = FoamBallConfig()
        if args.tracker_gravity is not None:
            if args.tracker_gravity <= 0.0:
                raise ValueError("--tracker-gravity must be positive")
            cfg.gravity = args.tracker_gravity
        if args.tracker_drag_coefficient is not None:
            if args.tracker_drag_coefficient < 0.0:
                raise ValueError("--tracker-drag-coefficient must be non-negative")
            cfg.drag_coefficient = args.tracker_drag_coefficient
        if args.tracker_simulation_dt is not None:
            if args.tracker_simulation_dt <= 0.0:
                raise ValueError("--tracker-simulation-dt must be positive")
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
            f"[J9.5 tracker] model=foam_drag gravity={cfg.gravity:.2f}m/s^2 "
            f"drag_k={cfg.drag_coefficient:.3f} sim_dt={cfg.simulation_dt:.4f}s "
            f"hist={cfg.history_size}/{cfg.history_max_age_s:.2f}s "
            f"min_hist={cfg.min_history_for_prediction} median={cfg.median_filter_window} "
            f"lookahead=[{cfg.min_lookahead:.2f},{cfg.max_lookahead:.2f}]s"
        )
        keys = RedisKeys(ball_source="optitrack")
        keys.ball.__dict__["optitrack_rigid_body_id"] = args.ball_rigid_body_id
        ball_key = keys.ball.optitrack_position
        if r.get(ball_key) is None:
            print(f"[J9.5] WARNING: {ball_key} is empty")
        else:
            print(f"[J9.5] reading ball from {ball_key}")
        tracker = FoamBallTracker(r, keys, cfg)
        if trace is not None:
            trace.write(
                "tracker_config",
                ball_key=ball_key,
                ball_rigid_body_id=args.ball_rigid_body_id,
                tracker_config=_tracker_config_snapshot(tracker),
            )

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
            decel_s=args.decel_s,
            commit_tti=args.commit_tti,
            contact_margin_s=args.contact_margin_s,
            timing_guard_s=args.timing_guard_s,
            early_strike_lead_s=args.early_strike_lead_s,
            min_planned_strike_s=args.min_planned_strike_s,
            max_planned_strike_s=args.max_planned_strike_s,
            try_late_slack_s=args.try_late_slack_s,
            max_stretched_strike_s=args.max_stretched_strike_s,
            return_s=args.return_s,
            return_vel_frac=args.return_vel_frac,
            rate_hz=args.rate_hz,
            no_commit=args.no_commit,
            reach_m=args.reach_m,
            z_min=args.z_min_m,
            z_max=args.z_max_m,
            world_z_min_m=args.world_z_min_m,
            world_z_max_m=args.world_z_max_m,
            tracking_step_deg=args.tracking_step_deg,
            tracking_accel_deg_s2=args.tracking_accel_deg_s2,
            swing_vel_frac=args.swing_vel_frac,
            impact_vel_frac=args.impact_vel_frac,
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
            trace=trace,
            ball_key=ball_key,
        )
    except KeyboardInterrupt:
        if args.shutdown_hold_s > 0.0:
            _hold_current_joints_for(r, args.shutdown_hold_s, publish_hz=100.0)
            print(f"\n[J9.5] stopped -- refreshed current joint hold for {args.shutdown_hold_s:.1f}s.")
        else:
            _hold_current_joints(r)
            print("\n[J9.5] stopped -- holding current joint position in Redis.")
    finally:
        trace.close()


if __name__ == "__main__":
    main()
