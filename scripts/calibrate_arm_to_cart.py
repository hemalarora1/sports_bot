#!/usr/bin/env python3
"""Automatic calibration of T_C_A + T_E_P via SE(3) hand-eye.

What this solves
----------------
Two static SE(3) transforms needed to convert a world-frame racket goal into
a Franka EE goal:

    T_C_A    cart odom control point C  →  Franka arm base A
             (depends on how the arm is bolted to the cart; persists across
             sessions until the arm gets re-mounted)
    T_E_P    Franka EE flange E  →  racket sweet-spot P
             (depends on how the racket+marker is attached to the flange;
             persists until the racket or its OptiTrack rigid body changes)

The constraint at every captured waypoint is

    T_W_A  ⊕  T_A_E_i  ⊕  T_E_P  =  T_W_P_i

with T_W_A constant across samples (cart is stationary during the whole
calibration). Solved jointly via Levenberg-Marquardt over 12 unknowns; see
`solve_hand_eye_se3` in `utils/frames.py`. T_C_A is back-solved from the
fitted T_W_A using the *live* T_W_C = T_W_B ⊕ T_B_C at calibration time.

How it works
------------
1. Pre-flight: ping Redis; confirm cart + racket rigid bodies visible;
   confirm OpenSai cartesian + joint controllers exposed; read the current
   joint config as 'home'.
2. For each waypoint in a hardcoded delta tour (default 10):
     a. Validate target joints are within Franka limits; skip if not.
     b. Switch to joint_controller, drive arm to target, wait for settle.
     c. Average T_A_E (from OpenSai cartesian_task::current_*) and T_W_P
        (from OptiTrack rigid body on the racket) over `--capture-seconds`.
     d. Skip the waypoint if either reading is missing >70 % of the window
        (racket marker occluded by arm, etc).
3. Drive arm back to home; switch back to cartesian_controller.
4. Solve hand-eye over the captured samples.
5. Back-solve T_C_A = T_W_C⁻¹ ⊕ T_W_A using the snapshot T_W_C.
6. Save T_C_A + T_E_P to `sports_bot/optitrack/arm_calibration.json`.

Prerequisites (run in order)
----------------------------
1. redis-server
2. OptiTrack streamer (cart marker AND racket marker both visible)
3. OpenSai cartesian controller for the Franka (FrankaRobot)
4. TidyBot driver — optional, only needed for the wheel base; the arm
   calibration itself doesn't depend on it. (T_W_C is derived from OptiTrack +
   T_B_C, not from hb1::current_pose.)
5. T_B_C calibration: `sports_bot/optitrack/robot_marker_calibration.json`
6. CART MUST BE STATIONARY throughout the whole calibration run.

Usage
-----
    conda activate opensai
    # Dry-run first to see the waypoint tour without moving the arm:
    python sports_bot/scripts/calibrate_arm_to_cart.py \\
        --robot-rigid-body-id 11 --racket-rigid-body-id 9 --dry-run

    # Live run:
    python sports_bot/scripts/calibrate_arm_to_cart.py \\
        --robot-rigid-body-id 11 --racket-rigid-body-id 9
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import redis

# Importable as `python sports_bot/scripts/calibrate_arm_to_cart.py` from
# the OpenSai root.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SPORTS_BOT_DIR = os.path.dirname(_THIS_DIR)
_OPENSAI_DIR = os.path.dirname(_SPORTS_BOT_DIR)
if _OPENSAI_DIR not in sys.path:
    sys.path.insert(0, _OPENSAI_DIR)

from sports_bot.utils.frames import (  # noqa: E402
    SE3,
    R_from_axis_angle,
    arm_calibration_path,
    average_quats,
    axis_angle_from_R,
    load_robot_marker_calibration,
    quat_to_R,
    robot_marker_calibration_path,
    save_arm_calibration,
    se3_compose,
    se3_from_se2,
    se3_inverse,
    solve_hand_eye_se3,
)


# ---------- Redis keys --------------------------------------------------------

OPTI_POS_PREFIX = "sai2::optitrack::rigid_body_pos::"
OPTI_ORI_PREFIX = "sai2::optitrack::rigid_body_ori::"

# OpenSai cartesian_controller (where we read EE pose from).
EE_POS_FMT = ("opensai::controllers::{robot}::cartesian_controller::"
              "cartesian_task::current_position")
EE_ORI_FMT = ("opensai::controllers::{robot}::cartesian_controller::"
              "cartesian_task::current_orientation")

# OpenSai joint_controller (where we drive the arm during the tour).
JOINT_CURRENT_FMT = ("opensai::controllers::{robot}::joint_controller::"
                     "joint_task::current_position")
JOINT_GOAL_FMT    = ("opensai::controllers::{robot}::joint_controller::"
                     "joint_task::goal_position")
OTG_MAX_VEL_FMT   = ("opensai::controllers::{robot}::joint_controller::"
                     "joint_task::otg_max_velocity")
ACTIVE_CTRL_FMT   = "opensai::controllers::{robot}::active_controller_name"
CARTESIAN_CTRL    = "cartesian_controller"
JOINT_CTRL        = "joint_controller"


# ---------- Franka joint limits ----------------------------------------------
# From the official Franka Emika Panda spec sheet. The script clamps candidate
# joint configurations and skips waypoints that would exceed these.

FRANKA_J_MIN = np.array([-2.8973, -1.7628, -2.8973, -3.0718,
                         -2.8973, -0.0175, -2.8973])
FRANKA_J_MAX = np.array([ 2.8973,  1.7628,  2.8973, -0.0698,
                          2.8973,  3.7525,  2.8973])
# Tour-mode velocity cap (rad/s). Conservative so the arm moves slowly during
# the calibration tour — we're not in a hurry, and slow motion makes the OT
# settle gate trivial. Restored to whatever was there before, on exit.
FRANKA_OTG_VEL_TOUR = np.array([0.6, 0.6, 0.6, 0.6, 0.8, 0.8, 1.0])


# ---------- Default waypoint tour --------------------------------------------
# Each row is a joint-space delta applied on top of the operator-positioned
# 'home' configuration. Designed to maximize SE(3) pose diversity of the EE
# while staying within Franka limits from a reasonable upright home.
# Strategy: mix translation-inducing joints (1, 2, 4) with wrist-orientation
# joints (5, 6, 7) so each waypoint contributes a novel rotation axis.
#
# If a waypoint would exceed joint limits relative to YOUR home, the script
# will skip it and warn. As long as ≥6 valid waypoints survive with good
# rotational + translational diversity, the solver is fine.

WAYPOINT_DELTAS_RAD: List[np.ndarray] = [
    np.array([+0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0]),  # home (reference)
    np.array([+0.4, -0.2, +0.0, +0.2, +0.0, +0.3, +0.5]),  # swing left + tilt
    np.array([-0.4, -0.2, +0.0, +0.2, +0.0, -0.3, -0.5]),  # swing right + tilt
    np.array([+0.3, +0.2, +0.0, +0.2, +0.0, +0.0, +0.8]),  # forward-up + roll
    np.array([-0.3, +0.2, +0.0, +0.2, +0.0, +0.0, -0.8]),  # forward-up + roll
    np.array([+0.0, -0.3, +0.0, -0.3, +0.0, +0.5, +0.0]),  # arm reach + pitch
    np.array([+0.0, +0.3, +0.0, +0.3, +0.0, -0.5, +0.0]),  # arm down + pitch
    np.array([+0.3, +0.0, +0.0, +0.0, +0.0, +0.5, -0.8]),  # mixed
    np.array([-0.3, +0.0, +0.0, +0.0, +0.0, -0.5, +0.8]),  # mixed (mirror)
    np.array([+0.0, +0.0, +0.0, +0.0, +0.0, +0.4, +0.4]),  # wrist-only diversity
]


# ---------- Helpers -----------------------------------------------------------

@dataclass
class OpenSaiKeys:
    robot_name: str

    @property
    def ee_pos(self) -> str:     return EE_POS_FMT.format(robot=self.robot_name)
    @property
    def ee_ori(self) -> str:     return EE_ORI_FMT.format(robot=self.robot_name)
    @property
    def joint_cur(self) -> str:  return JOINT_CURRENT_FMT.format(robot=self.robot_name)
    @property
    def joint_goal(self) -> str: return JOINT_GOAL_FMT.format(robot=self.robot_name)
    @property
    def otg_max_vel(self) -> str:return OTG_MAX_VEL_FMT.format(robot=self.robot_name)
    @property
    def active(self) -> str:     return ACTIVE_CTRL_FMT.format(robot=self.robot_name)


def _read_json_array(r: redis.Redis, key: str) -> Optional[np.ndarray]:
    raw = r.get(key)
    if raw is None:
        return None
    try:
        return np.asarray(json.loads(raw), dtype=float)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def read_T_A_E(r: redis.Redis, keys: OpenSaiKeys) -> Optional[SE3]:
    """OpenSai exposes EE pose in the arm base frame A as (3-vec pos, 3x3 R)."""
    pos = _read_json_array(r, keys.ee_pos)
    ori = _read_json_array(r, keys.ee_ori)
    if pos is None or ori is None:
        return None
    if pos.shape != (3,) or ori.shape != (3, 3):
        return None
    return (ori, pos)


def read_T_W_rigidbody(r: redis.Redis, rb_id: int) -> Optional[SE3]:
    """Full SE(3) pose of an OptiTrack rigid body in world frame from the
    world-frame Redis keys published by StreamDataSkeleton.py."""
    raw_pos = r.get(OPTI_POS_PREFIX + str(rb_id))
    raw_ori = r.get(OPTI_ORI_PREFIX + str(rb_id))
    if raw_pos is None or raw_ori is None:
        return None
    try:
        pos = json.loads(raw_pos)
        ori = json.loads(raw_ori)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    if len(pos) != 3 or len(ori) != 4:
        return None
    R = quat_to_R(float(ori[0]), float(ori[1]), float(ori[2]), float(ori[3]))
    t = np.array([float(pos[0]), float(pos[1]), float(pos[2])])
    return (R, t)


def read_T_W_C(r: redis.Redis, cart_rb_id: int, T_B_C_2d) -> Optional[SE3]:
    """T_W_C = T_W_B (full SE(3) from OT) ⊕ T_B_C (SE(2), lifted to SE(3)
    at z=0). Returns None if cart marker isn't visible."""
    T_W_B = read_T_W_rigidbody(r, cart_rb_id)
    if T_W_B is None:
        return None
    T_B_C_3d = se3_from_se2(T_B_C_2d, z=0.0)
    return se3_compose(T_W_B, T_B_C_3d)


# ---------- Motion primitives -------------------------------------------------

def switch_controller(r: redis.Redis, keys: OpenSaiKeys, controller: str) -> None:
    r.set(keys.active, controller)


def seed_joint_goal_to_current(r: redis.Redis, keys: OpenSaiKeys) -> bool:
    """Before switching to joint_controller, copy current joint positions into
    the goal so the controller doesn't lurch when it becomes active."""
    cur = _read_json_array(r, keys.joint_cur)
    if cur is None or cur.shape != (7,):
        return False
    r.set(keys.joint_goal, json.dumps(cur.tolist()))
    return True


def drive_to_joints(
    r: redis.Redis,
    keys: OpenSaiKeys,
    target: np.ndarray,
    *,
    settle_tol_rad: float,
    settle_sustain_s: float,
    timeout_s: float,
) -> bool:
    """Command target joints and wait for settle. Returns True if the arm
    converged within tolerance for `settle_sustain_s`, False on timeout."""
    r.set(keys.joint_goal, json.dumps(target.tolist()))
    t_start = time.perf_counter()
    t_below = None
    while time.perf_counter() - t_start < timeout_s:
        cur = _read_json_array(r, keys.joint_cur)
        if cur is None or cur.shape != (7,):
            time.sleep(0.05)
            continue
        err = float(np.max(np.abs(cur - target)))
        if err < settle_tol_rad:
            if t_below is None:
                t_below = time.perf_counter()
            elif time.perf_counter() - t_below >= settle_sustain_s:
                return True
        else:
            t_below = None
        time.sleep(0.02)
    return False


def capture_pose_pair(
    r: redis.Redis,
    keys: OpenSaiKeys,
    racket_rb_id: int,
    duration_s: float,
    rate_hz: float = 60.0,
) -> Tuple[Optional[SE3], Optional[SE3], dict]:
    """Average T_A_E (Franka FK from OpenSai) and T_W_P (OptiTrack racket
    rigid body in world) over `duration_s` while the arm sits still.

    Returns (T_A_E_avg, T_W_P_avg, stats). Either component may be None if
    too few valid samples were captured.
    """
    dt = 1.0 / rate_hz
    A_E_pos: List[np.ndarray] = []
    A_E_qx: List[Tuple[float, float, float, float]] = []
    W_P_pos: List[np.ndarray] = []
    W_P_qx: List[Tuple[float, float, float, float]] = []

    t_end = time.perf_counter() + duration_s
    n_attempts = 0
    while time.perf_counter() < t_end:
        n_attempts += 1
        T_A_E = read_T_A_E(r, keys)
        T_W_P = read_T_W_rigidbody(r, racket_rb_id)
        if T_A_E is not None:
            R_AE, t_AE = T_A_E
            A_E_pos.append(t_AE)
            A_E_qx.append(_R_to_quat_tuple(R_AE))
        if T_W_P is not None:
            R_WP, t_WP = T_W_P
            W_P_pos.append(t_WP)
            W_P_qx.append(_R_to_quat_tuple(R_WP))
        time.sleep(dt)

    stats = {
        "n_attempts": n_attempts,
        "n_A_E": len(A_E_pos),
        "n_W_P": len(W_P_pos),
        "yield_A_E": len(A_E_pos) / max(1, n_attempts),
        "yield_W_P": len(W_P_pos) / max(1, n_attempts),
    }

    T_A_E_avg = None
    T_W_P_avg = None
    if len(A_E_pos) >= 5:
        T_A_E_avg = (_quat_tuple_to_R(average_quats(A_E_qx)),
                     np.mean(np.asarray(A_E_pos), axis=0))
        stats["pos_jitter_mm_A_E"] = float(1000.0 * np.mean(np.std(np.asarray(A_E_pos), axis=0)))
    if len(W_P_pos) >= 5:
        T_W_P_avg = (_quat_tuple_to_R(average_quats(W_P_qx)),
                     np.mean(np.asarray(W_P_pos), axis=0))
        stats["pos_jitter_mm_W_P"] = float(1000.0 * np.mean(np.std(np.asarray(W_P_pos), axis=0)))

    return T_A_E_avg, T_W_P_avg, stats


def _R_to_quat_tuple(R: np.ndarray) -> Tuple[float, float, float, float]:
    from sports_bot.utils.frames import R_to_quat
    return R_to_quat(R)


def _quat_tuple_to_R(q: Tuple[float, float, float, float]) -> np.ndarray:
    return quat_to_R(q[0], q[1], q[2], q[3])


# ---------- Diversity check ---------------------------------------------------

def check_diversity(samples: List[Tuple[SE3, SE3]]) -> Optional[str]:
    """Warn if the captured T_W_P poses lack translation or rotation spread.
    The solver underdetermines T_E_P's rotation if T_A_E only rotates about
    a single axis across samples, and underdetermines T_E_P's translation
    if T_A_E only translates.
    """
    if len(samples) < 3:
        return f"only {len(samples)} valid sample(s) — need ≥3"
    # Translation span of measured T_W_P
    W_P_pos = np.asarray([s[1][1] for s in samples])
    pos_span = float(np.max(np.linalg.norm(W_P_pos - np.mean(W_P_pos, axis=0), axis=1)))

    # Rotation span: collect axis-angle of T_A_E_i relative to sample 0, and
    # check that the principal axis is reasonably distributed (no single
    # rotation axis dominates).
    R_ref = samples[0][0][0]
    rel_omegas = np.asarray([
        axis_angle_from_R(s[0][0] @ R_ref.T) for s in samples
    ])
    rel_norms = np.linalg.norm(rel_omegas, axis=1)
    rot_span_deg = math.degrees(float(np.max(rel_norms)))

    warnings = []
    if pos_span < 0.10:
        warnings.append(f"T_W_P translation span only {pos_span*100:.1f} cm "
                        f"(recommend ≥10 cm)")
    if rot_span_deg < 30.0:
        warnings.append(f"EE rotation span only {rot_span_deg:.1f}° from reference "
                        f"(recommend ≥30°)")
    return "; ".join(warnings) if warnings else None


# ---------- Per-waypoint reporting --------------------------------------------

def report_residuals(
    samples: List[Tuple[SE3, SE3]],
    T_W_A: SE3,
    T_E_P: SE3,
) -> None:
    print(f"\n  per-waypoint residual (predicted T_W_P − measured T_W_P):")
    print(f"    {'#':>3}  {'dpos (mm)':>10}  {'dori (deg)':>10}")
    print(f"    {'-' * 3}  {'-' * 10}  {'-' * 10}")
    for i, (T_A_E, T_W_P_meas) in enumerate(samples):
        T_W_P_pred = se3_compose(se3_compose(T_W_A, T_A_E), T_E_P)
        dpos_mm = 1000.0 * float(np.linalg.norm(T_W_P_pred[1] - T_W_P_meas[1]))
        R_err = T_W_P_pred[0] @ T_W_P_meas[0].T
        dori_deg = math.degrees(float(np.linalg.norm(axis_angle_from_R(R_err))))
        print(f"    {i:>3}  {dpos_mm:>10.2f}  {dori_deg:>10.3f}")


# ---------- Main flow ---------------------------------------------------------

def run(args: argparse.Namespace) -> int:
    keys = OpenSaiKeys(robot_name=args.robot_name)

    # --- Connect ---
    r = redis.Redis(host=args.redis_host, port=args.redis_port, decode_responses=True)
    try:
        r.ping()
    except redis.exceptions.ConnectionError as e:
        print(f"[arm_calib] Cannot reach Redis at {args.redis_host}:{args.redis_port}: {e}")
        return 1

    # --- Load T_B_C ---
    tbc_path = args.robot_marker_calibration or robot_marker_calibration_path()
    if not os.path.isfile(tbc_path):
        print(f"[arm_calib] Missing T_B_C calibration at {tbc_path}.")
        print(f"            Run sports_bot/scripts/calibrate_robot_marker.py first.")
        return 1
    T_B_C_2d = load_robot_marker_calibration(tbc_path)
    print(f"[arm_calib] Loaded T_B_C: "
          f"({T_B_C_2d[0]:+.4f} m, {T_B_C_2d[1]:+.4f} m, {math.degrees(T_B_C_2d[2]):+.2f}°)")

    # --- Pre-flight liveness checks ---
    if read_T_W_rigidbody(r, args.robot_rigid_body_id) is None:
        print(f"[arm_calib] No OptiTrack data for cart rigid body "
              f"{args.robot_rigid_body_id}. Is StreamDataSkeleton.py running?")
        return 1
    if read_T_W_rigidbody(r, args.racket_rigid_body_id) is None:
        print(f"[arm_calib] No OptiTrack data for racket rigid body "
              f"{args.racket_rigid_body_id}. Did you create the rigid body in "
              f"Motive and is it visible to the cameras?")
        return 1
    if read_T_A_E(r, keys) is None:
        print(f"[arm_calib] Cannot read EE pose from "
              f"{keys.ee_pos}. Is OpenSai cartesian_controller running on "
              f"robot '{args.robot_name}'?")
        return 1
    joints_home = _read_json_array(r, keys.joint_cur)
    if joints_home is None or joints_home.shape != (7,):
        print(f"[arm_calib] Cannot read joint positions from {keys.joint_cur}.")
        return 1
    print(f"[arm_calib] Pre-flight OK. Home joints (rad): "
          f"{np.round(joints_home, 3).tolist()}")

    # --- Snapshot T_W_C at calibration time ---
    T_W_C = read_T_W_C(r, args.robot_rigid_body_id, T_B_C_2d)
    if T_W_C is None:
        print(f"[arm_calib] Failed to read T_W_C.")
        return 1
    print(f"[arm_calib] T_W_C snapshot:")
    print(f"            t = {T_W_C[1].tolist()}")
    print(f"            ω = {axis_angle_from_R(T_W_C[0]).tolist()}")

    # --- Plan waypoints (joint-space, deltas-from-home) ---
    deltas = WAYPOINT_DELTAS_RAD
    if args.max_waypoints is not None:
        deltas = deltas[: args.max_waypoints]
    planned: List[Tuple[int, np.ndarray, str]] = []   # (idx, target_joints, status)
    for i, delta in enumerate(deltas):
        target = joints_home + delta
        if np.any(target < FRANKA_J_MIN) or np.any(target > FRANKA_J_MAX):
            offending = [(j, target[j], FRANKA_J_MIN[j], FRANKA_J_MAX[j])
                         for j in range(7)
                         if target[j] < FRANKA_J_MIN[j] or target[j] > FRANKA_J_MAX[j]]
            planned.append((i, target, f"OUT_OF_LIMITS ({offending})"))
        else:
            planned.append((i, target, "ok"))

    print(f"\n[arm_calib] Planned waypoint tour ({len(planned)} entries):")
    for i, target, status in planned:
        delta = target - joints_home
        delta_str = ", ".join(f"{d:+5.2f}" for d in delta)
        print(f"  WP{i:02d}  Δ=[{delta_str}]  {status}")
    n_ok = sum(1 for _, _, s in planned if s == "ok")
    if n_ok < 3:
        print(f"\n[arm_calib] Only {n_ok} waypoint(s) within joint limits — "
              f"need ≥3. Move the arm to a more central home posture and re-run.")
        return 1

    if args.dry_run:
        print(f"\n[arm_calib] --dry-run set — not switching controllers, not moving the arm.")
        print(f"            {n_ok}/{len(planned)} waypoints would execute.")
        return 0

    # --- Switch to joint controller ---
    print(f"\n[arm_calib] Switching to joint_controller (will restore at end).")
    if not seed_joint_goal_to_current(r, keys):
        print(f"[arm_calib] Failed to seed joint goal. Aborting before controller switch.")
        return 1
    time.sleep(0.1)
    switch_controller(r, keys, JOINT_CTRL)
    time.sleep(0.5)

    # Slow tour velocity. Snapshot prior OTG cap to restore later.
    prior_otg = _read_json_array(r, keys.otg_max_vel)
    r.set(keys.otg_max_vel, json.dumps(FRANKA_OTG_VEL_TOUR.tolist()))

    # --- Run the tour ---
    samples: List[Tuple[SE3, SE3]] = []
    capture_log: List[dict] = []
    aborted = False
    try:
        for i, target, status in planned:
            if status != "ok":
                print(f"\n[arm_calib] WP{i:02d} skipped: {status}")
                continue
            print(f"\n[arm_calib] WP{i:02d}: driving to target ...")
            settled = drive_to_joints(
                r, keys, target,
                settle_tol_rad=args.settle_tol_rad,
                settle_sustain_s=args.settle_sustain_s,
                timeout_s=args.settle_timeout_s,
            )
            if not settled:
                print(f"            did NOT settle in {args.settle_timeout_s:.1f}s "
                      f"— skipping capture.")
                continue
            print(f"            settled. Capturing for {args.capture_seconds:.1f}s "
                  f"(racket must stay visible to OT)...")
            T_A_E, T_W_P, stats = capture_pose_pair(
                r, keys, args.racket_rigid_body_id,
                duration_s=args.capture_seconds,
            )
            log_entry = {"wp": i, **stats}
            if T_A_E is None or T_W_P is None:
                why = []
                if T_A_E is None: why.append(f"A_E yield {stats['yield_A_E']:.0%}")
                if T_W_P is None: why.append(f"W_P yield {stats['yield_W_P']:.0%}")
                print(f"            CAPTURE FAILED ({'; '.join(why)}). "
                      f"Skipping waypoint.")
                log_entry["skipped"] = True
                capture_log.append(log_entry)
                continue
            if stats["yield_W_P"] < 0.30:
                print(f"            W_P yield {stats['yield_W_P']:.0%} — marker "
                      f"mostly occluded. Skipping (would poison the fit).")
                log_entry["skipped"] = True
                capture_log.append(log_entry)
                continue
            samples.append((T_A_E, T_W_P))
            capture_log.append(log_entry)
            jitter_str = ""
            if "pos_jitter_mm_A_E" in stats:
                jitter_str += f"A_E {stats['pos_jitter_mm_A_E']:.2f} mm  "
            if "pos_jitter_mm_W_P" in stats:
                jitter_str += f"W_P {stats['pos_jitter_mm_W_P']:.2f} mm"
            print(f"            captured. yields A_E {stats['yield_A_E']:.0%} / "
                  f"W_P {stats['yield_W_P']:.0%}.  jitter {jitter_str}")

    except KeyboardInterrupt:
        print(f"\n[arm_calib] interrupted during tour.")
        aborted = True

    # --- Return to home + cleanup ---
    print(f"\n[arm_calib] Returning to home joints ...")
    drive_to_joints(
        r, keys, joints_home,
        settle_tol_rad=args.settle_tol_rad,
        settle_sustain_s=args.settle_sustain_s,
        timeout_s=args.settle_timeout_s,
    )
    if prior_otg is not None:
        r.set(keys.otg_max_vel, json.dumps(prior_otg.tolist()))
    print(f"[arm_calib] Switching back to cartesian_controller ...")
    switch_controller(r, keys, CARTESIAN_CTRL)

    if aborted and len(samples) < 3:
        print(f"[arm_calib] Aborted with only {len(samples)} valid samples — "
              f"not solving.")
        return 130
    if len(samples) < 3:
        print(f"[arm_calib] Only {len(samples)} valid samples — need ≥3. "
              f"Common causes: racket markers occluded by arm at many poses; "
              f"joint limits too restrictive from this home pose.")
        return 1

    # --- Diversity check ---
    diversity_warning = check_diversity(samples)
    if diversity_warning:
        print(f"\n[arm_calib] WARNING: poor sample diversity — {diversity_warning}.")

    # --- Solve hand-eye ---
    print(f"\n[arm_calib] Solving SE(3) hand-eye over {len(samples)} samples ...")
    T_W_A, T_E_P, pos_rms, ori_rms_deg = solve_hand_eye_se3(samples)

    print()
    print(f"  T_W_A (this session, used to back-solve T_C_A):")
    print(f"    t = {T_W_A[1].tolist()}")
    print(f"    ω = {axis_angle_from_R(T_W_A[0]).tolist()} rad")
    print(f"  T_E_P (PERSIST):")
    print(f"    t = {T_E_P[1].tolist()}  (≈ flange→sweet-spot offset, m)")
    print(f"    ω = {axis_angle_from_R(T_E_P[0]).tolist()} rad")
    print()
    print(f"  residual RMS  : {pos_rms*1000:.2f} mm / {ori_rms_deg:.3f}°")

    if pos_rms > 0.01 or ori_rms_deg > 1.0:
        print(f"  WARNING: residuals are high. Likely causes:")
        print(f"    - cart moved during the tour (check ground / wheel locks)")
        print(f"    - racket marker shifted between captures (markers loose?)")
        print(f"    - too little EE rotational diversity in surviving waypoints")
        print(f"    - OpenSai EE pose noisy / lagged (joint controller still settling)")

    report_residuals(samples, T_W_A, T_E_P)

    # --- Back-solve T_C_A ---
    T_C_A = se3_compose(se3_inverse(T_W_C), T_W_A)
    print(f"\n  T_C_A (PERSIST):")
    print(f"    t = {T_C_A[1].tolist()}  (≈ cart-pivot→arm-base offset, m)")
    print(f"    ω = {axis_angle_from_R(T_C_A[0]).tolist()} rad")

    # --- Save ---
    out_path = args.output or arm_calibration_path()
    metadata = {
        "calibrated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "robot_name": args.robot_name,
        "cart_rigid_body_id": args.robot_rigid_body_id,
        "racket_rigid_body_id": args.racket_rigid_body_id,
        "n_samples": len(samples),
        "residual_pos_rms_mm": round(pos_rms * 1000.0, 3),
        "residual_ori_rms_deg": round(ori_rms_deg, 4),
        "T_W_C_at_calibration": {
            "translation_m": T_W_C[1].tolist(),
            "rotation_matrix": [list(row) for row in T_W_C[0]],
        },
        "robot_marker_calibration": tbc_path,
        "capture_log": capture_log,
    }

    if args.dry_run:
        print(f"\n[arm_calib] --dry-run set — not writing {out_path}.")
        return 0
    if os.path.isfile(out_path) and not args.force:
        print(f"\n[arm_calib] {out_path} already exists.")
        try:
            ans = input("            overwrite? [y/N] ").strip().lower()
        except EOFError:
            ans = ""
        if ans != "y":
            print("            not written.")
            return 0

    save_arm_calibration(out_path, T_C_A, T_E_P, metadata)
    print(f"\n[arm_calib] Wrote {out_path}")
    return 0


# ---------- Entry point -------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Auto-calibrate T_C_A (arm mount on cart) + T_E_P (racket on flange).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--robot-rigid-body-id", type=int, required=True, metavar="ID",
        help="Motive Streaming ID of the TidyBot rigid body.",
    )
    parser.add_argument(
        "--racket-rigid-body-id", type=int, required=True, metavar="ID",
        help="Motive Streaming ID of the racket rigid body. Must be defined "
             "with pivot at the paddle sweet spot and +Z aligned with the "
             "face normal (matches the SwingPlanner convention).",
    )
    parser.add_argument(
        "--robot-name", default="FrankaRobot",
        help="OpenSai robot name (used to build the Redis key prefixes). "
             "Matches what test_swing.py uses.",
    )
    parser.add_argument(
        "--robot-marker-calibration", type=str, default=None,
        help="Path to T_B_C calibration JSON. Default: "
             "sports_bot/optitrack/robot_marker_calibration.json",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Where to write the arm calibration JSON. Default: "
             "sports_bot/optitrack/arm_calibration.json",
    )
    parser.add_argument("--redis-host", default="localhost")
    parser.add_argument("--redis-port", type=int, default=6379)

    parser.add_argument(
        "--capture-seconds", type=float, default=1.0,
        help="How long to average pose readings at each waypoint.",
    )
    parser.add_argument(
        "--settle-tol-rad", type=float, default=0.01,
        help="Joint error threshold (rad) for declaring the arm settled.",
    )
    parser.add_argument(
        "--settle-sustain-s", type=float, default=0.3,
        help="How long the joint error must stay below tolerance before "
             "capture starts.",
    )
    parser.add_argument(
        "--settle-timeout-s", type=float, default=8.0,
        help="Max time to wait for settle before giving up on a waypoint.",
    )
    parser.add_argument(
        "--max-waypoints", type=int, default=None,
        help="If set, only use the first N waypoints from the default tour. "
             "Useful for quick smoke-tests.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite an existing arm_calibration.json without prompting.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Pre-flight + print the planned tour, but don't switch controllers "
             "or move the arm.",
    )
    args = parser.parse_args()

    try:
        rc = run(args)
    except KeyboardInterrupt:
        print("\n[arm_calib] interrupted.")
        rc = 130
    sys.exit(rc)


if __name__ == "__main__":
    main()
