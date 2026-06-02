#!/usr/bin/env python3
"""
Step J3: IK round-trip test (ikpy → joint_controller, no cartesian_controller).

Pipeline:
  1. Calibrate FK offset: read SENSOR_JOINTS + OpenSai current_position
     simultaneously (arm stationary, cartesian_controller active for reading only).
     This measures the fixed vector from ikpy's link7 frame to OpenSai's
     end-effector frame in the base frame at the current config.
  2. For each target pose:
       a. Solve IK (position-only) for (target - fk_offset) using ikpy.
       b. Safety-check: max joint delta from current < delta_limit.
       c. Send q_ik to joint_controller, wait to settle.
       d. Read settled SENSOR_JOINTS, run ikpy FK + offset.
       e. Report Cartesian residual (pass: < pos_tol_mm).
  3. Return to home joints.

Does NOT command any motion via cartesian_controller.

Run from OpenSai root:
    python sports_bot/arm_control/stepj3_ik_roundtrip.py
    python sports_bot/arm_control/stepj3_ik_roundtrip.py --pose all --verbose
    python sports_bot/arm_control/stepj3_ik_roundtrip.py --pose strike_mid --pause
"""
import argparse
import json
import sys
import time
import warnings

import numpy as np
import redis
from scipy.optimize import minimize

warnings.filterwarnings("ignore")  # suppress ikpy UserWarning on fixed links
import ikpy.chain  # noqa: E402

# Reuse the smoothstep interpolation runner from J1/J2.
# run_segment(r, name, q_start, q_target, move_s, hold_s, publish_hz, verbose)
from stepj1_joint_nudge import run_segment, SENSOR_JOINTS as _SJ  # noqa: E402 (verify same key)

# ---------------------------------------------------------------------------
# URDF and robot constants
# ---------------------------------------------------------------------------
URDF_PATH = "drivers/FrankaPanda/model/panda_arm.urdf"

ROBOT_NAME  = "FrankaRobot"
NS          = f"opensai::controllers::{ROBOT_NAME}"
CART_CTRL   = "cartesian_controller"
JOINT_CTRL  = "joint_controller"

ACTIVE_CTRL   = f"opensai::controllers::{ROBOT_NAME}::active_controller_name"
SENSOR_JOINTS = f"opensai::sensors::{ROBOT_NAME}::joint_positions"
SENSOR_VELS   = f"opensai::sensors::{ROBOT_NAME}::joint_velocities"
CURR_POS      = f"{NS}::{CART_CTRL}::cartesian_task::current_position"
GOAL_JOINTS   = f"{NS}::{JOINT_CTRL}::joint_task::goal_position"

# Franka soft joint limits (rad)
Q_LO = np.radians([-166, -101, -166, -176,  -166,  -1,  -166])
Q_HI = np.radians([ 166,  101,  166,   -4,   166, 215,   166])

# Target poses in OpenSai end-effector frame (Franka base coords, metres).
# x = forward, y = left, z = up (from Franka base mount at 0.445m above floor).
# Derived from calibration: home EE is approximately [0.41, 0.16, 0.56] m.
# Strike zone: x~0.55-0.65m, z~0.35-0.70m (ball heights 0.80-1.15m above floor).
POSES = {
    "near_home": {
        "pos": np.array([0.44, 0.12, 0.54]),
        "desc": "tiny move from home (~3cm) — IK/FK sanity check",
    },
    "strike_mid": {
        "pos": np.array([0.55, 0.05, 0.52]),
        "desc": "mid-height strike zone, center",
    },
    "strike_high": {
        "pos": np.array([0.52, 0.00, 0.65]),
        "desc": "high strike zone",
    },
    "strike_low": {
        "pos": np.array([0.55, 0.05, 0.38]),
        "desc": "low strike zone",
    },
}

DEFAULT_SEQUENCE = ["near_home", "strike_mid", "strike_high"]


# ---------------------------------------------------------------------------
# Redis helpers
# ---------------------------------------------------------------------------
def get_vec(r, key, n=None):
    raw = r.get(key)
    if raw is None:
        return None
    try:
        arr = np.array(json.loads(raw), dtype=float)
    except Exception:
        return None
    if n is not None and arr.shape != (n,):
        return None
    return arr


def set_vec(r, key, v):
    r.set(key, json.dumps(np.asarray(v, dtype=float).tolist()))


def decode(r, key):
    v = r.get(key)
    return v.decode() if isinstance(v, bytes) else v


def switch_ctrl(r, name, timeout=2.0):
    t0 = time.monotonic()
    while True:
        r.set(ACTIVE_CTRL, name)
        if decode(r, ACTIVE_CTRL) == name:
            return True
        if time.monotonic() - t0 > timeout:
            print(f"  WARNING: controller switch to {name!r} timed out", file=sys.stderr)
            return False
        time.sleep(0.02)


def hold(r, duration, hz=100.0, joint_goal=None, cart_goal=None):
    dt = 1.0 / max(1.0, hz)
    t0 = time.monotonic()
    qdot_peak = np.zeros(7)
    while time.monotonic() - t0 < duration:
        if joint_goal is not None:
            set_vec(r, GOAL_JOINTS, joint_goal)
        qdot = get_vec(r, SENSOR_VELS, 7)
        if qdot is not None:
            qdot_peak = np.maximum(qdot_peak, np.abs(qdot))
        time.sleep(dt)
    return qdot_peak


def fmt_q(q_rad):
    return "  ".join(f"q{i+1}={np.degrees(v):+.1f}" for i, v in enumerate(q_rad))


def fmt_pos(p):
    return f"[{p[0]:+.4f}, {p[1]:+.4f}, {p[2]:+.4f}]"


# ---------------------------------------------------------------------------
# IK helpers
# ---------------------------------------------------------------------------
def build_chain():
    return ikpy.chain.Chain.from_urdf_file(URDF_PATH, base_elements=["link0"])


def fk(chain, q_rad_7):
    """Return 4x4 FK matrix for 7 joint angles (rad)."""
    q_ikpy = np.concatenate([[0.0], q_rad_7])
    return chain.forward_kinematics(q_ikpy)


def ik_solve(chain, target_pos_ikpy, q_init_rad_7, w_reg=0.001):
    """
    Regularized position-only IK via scipy L-BFGS-B.

    Minimizes:  ||FK(q) - target||^2 + w_reg * ||q - q_init||^2

    The regularization term keeps the solution near the current joint
    configuration, which:
      - avoids wrist flips and other null-space surprises
      - produces smooth trajectories when targets move incrementally
      - is the right behaviour for real-time FSM use

    Returns (q_rad_7, fk_err_m).
    """
    def cost(q):
        q_ikpy = np.concatenate([[0.0], q])
        pos = chain.forward_kinematics(q_ikpy)[:3, 3]
        pos_err = np.sum((pos - target_pos_ikpy) ** 2)
        reg     = np.sum((q - q_init_rad_7) ** 2)
        return pos_err + w_reg * reg

    bounds = list(zip(Q_LO, Q_HI))
    result = minimize(cost, q_init_rad_7, method="L-BFGS-B", bounds=bounds,
                      options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-8})
    q_out = result.x
    q_ikpy = np.concatenate([[0.0], q_out])
    fk_result = chain.forward_kinematics(q_ikpy)
    err_m = np.linalg.norm(fk_result[:3, 3] - target_pos_ikpy)
    return q_out, err_m


# ---------------------------------------------------------------------------
# Calibrate: measure fk_offset = opensai_curr_pos - ikpy_link7_pos
# ---------------------------------------------------------------------------
def calibrate_fk_offset(r, chain, verbose=False):
    """
    Read SENSOR_JOINTS and CURR_POS simultaneously (arm must be in
    cartesian_controller so CURR_POS is fresh). Returns fk_offset (3-vec).
    """
    # Ensure cartesian_controller active so CURR_POS is updated
    prev_ctrl = decode(r, ACTIVE_CTRL)
    switch_ctrl(r, CART_CTRL)
    time.sleep(0.1)  # let OpenSai update CURR_POS

    q_rad = get_vec(r, SENSOR_JOINTS, 7)
    cp    = get_vec(r, CURR_POS, 3)

    if q_rad is None:
        sys.exit(f"ERROR: {SENSOR_JOINTS} not available")
    if cp is None:
        sys.exit(f"ERROR: {CURR_POS} not available — is cartesian_controller in the XML?")

    fk_mat = fk(chain, q_rad)
    link7_pos = fk_mat[:3, 3]
    offset = cp - link7_pos

    print(f"\n[calibration]")
    print(f"  joints (deg):  {fmt_q(q_rad)}")
    print(f"  ikpy link7:    {fmt_pos(link7_pos)} m")
    print(f"  opensai ee:    {fmt_pos(cp)} m")
    print(f"  fk_offset:     {fmt_pos(offset)} m  (|{np.linalg.norm(offset)*1000:.1f}| mm)")

    expected_range = (0.03, 0.18)  # flange offset ~0.107m but projected
    if not (expected_range[0] < np.linalg.norm(offset) < expected_range[1]):
        print(f"  WARNING: offset magnitude {np.linalg.norm(offset)*1000:.1f} mm outside expected range — check URDF/frame", file=sys.stderr)

    return q_rad, offset


# ---------------------------------------------------------------------------
# Single pose test
# ---------------------------------------------------------------------------
def test_pose(r, chain, pose_name, pose, q_home_rad, fk_offset,
              settle_s=5.0, move_s=2.0, delta_limit_deg=30.0, pos_tol_m=0.005,
              pause=False, verbose=False):

    target_opensai = pose["pos"]
    target_ikpy    = target_opensai - fk_offset

    print(f"\n{'─'*68}")
    print(f"POSE: {pose_name!r}  —  {pose['desc']}")
    print(f"  target (opensai ee): {fmt_pos(target_opensai)} m")

    # --- IK solve ---
    q_current = get_vec(r, SENSOR_JOINTS, 7)
    if q_current is None:
        print("  ERROR: no sensor joints", file=sys.stderr)
        return None

    q_ik, ik_fk_err = ik_solve(chain, target_ikpy, q_current)
    print(f"  IK FK err (internal): {ik_fk_err*1000:.2f} mm")

    if ik_fk_err > pos_tol_m * 2:
        print(f"  SKIP: IK did not converge ({ik_fk_err*1000:.1f} mm > {pos_tol_m*2000:.0f} mm threshold)")
        return None

    # --- Joint limit check ---
    violations = []
    for i in range(7):
        if q_ik[i] < Q_LO[i] or q_ik[i] > Q_HI[i]:
            violations.append(f"q{i+1}={np.degrees(q_ik[i]):.1f}°")
    if violations:
        print(f"  SKIP: joint limit violation: {violations}")
        return None

    # --- Delta check ---
    delta_deg = np.degrees(np.abs(q_ik - q_current))
    max_delta = delta_deg.max()
    if max_delta > delta_limit_deg:
        print(f"  SKIP: max joint delta {max_delta:.1f}° > {delta_limit_deg:.0f}° limit "
              f"(q{int(delta_deg.argmax())+1} = {max_delta:.1f}°)")
        return None

    print(f"  IK joints (deg): {fmt_q(q_ik)}")
    print(f"  max delta from current: {max_delta:.1f}°")

    if pause:
        input(f"  [pause] press Enter to move smoothly to IK target ({move_s:.1f}s interpolation) ...")

    # --- Execute: smoothstep interpolation via run_segment (same as J2) ---
    # Seed current joints as goal, ensure joint_controller active, then interpolate.
    q_start = get_vec(r, SENSOR_JOINTS, 7)
    if q_start is None:
        q_start = q_current
    set_vec(r, GOAL_JOINTS, q_start.tolist())
    switch_ctrl(r, JOINT_CTRL)

    print(f"  [executing] smoothstep {move_s:.1f}s move + {settle_s:.0f}s hold ...")
    summary = run_segment(r, pose_name, q_start, q_ik,
                          move_s=move_s, hold_s=settle_s,
                          publish_hz=100.0, verbose=verbose)
    qdot_peak = np.zeros(7)  # run_segment reports internally

    # --- Measure result ---
    q_settled = get_vec(r, SENSOR_JOINTS, 7)
    if q_settled is None:
        print("  ERROR: lost sensor joints during hold", file=sys.stderr)
        return None

    q_err_deg = np.degrees(q_settled - q_ik)
    fk_settled = fk(chain, q_settled)
    ee_settled  = fk_settled[:3, 3] + fk_offset
    cart_err_m  = np.linalg.norm(ee_settled - target_opensai)

    passed = cart_err_m <= pos_tol_m

    print(f"  settled q (deg): {fmt_q(q_settled)}")
    print(f"  joint err (deg): " + "  ".join(f"q{i+1}={v:+.2f}" for i, v in enumerate(q_err_deg)))
    print(f"  settled ee pos:  {fmt_pos(ee_settled)} m")
    print(f"  cart residual:   {cart_err_m*1000:.1f} mm  (tol={pos_tol_m*1000:.0f} mm)")
    print(f"  qdot_peak (°/s): " + "  ".join(f"q{i+1}={np.degrees(v):.1f}" for i, v in enumerate(qdot_peak)))
    print(f"  result:          {'PASS ✓' if passed else 'FAIL ✗'}")

    return cart_err_m


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose", default="sequence",
                    choices=list(POSES.keys()) + ["sequence", "all"],
                    help="Pose(s) to test. 'sequence' runs default set, 'all' runs every pose.")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--settle-s", type=float, default=3.0)
    ap.add_argument("--move-s", type=float, default=2.0,
                    help="Smoothstep interpolation time per pose (default 2.0s).")
    ap.add_argument("--pos-tol-mm", type=float, default=8.0,
                    help="Cartesian residual pass threshold in mm (default 8mm for J3).")
    ap.add_argument("--delta-limit-deg", type=float, default=40.0,
                    help="Max allowed joint change from current per pose (safety gate).")
    ap.add_argument("--pause", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    r = redis.Redis(host=args.host, port=6379, decode_responses=True)
    try:
        r.ping()
    except Exception:
        sys.exit("ERROR: Redis not reachable")

    chain = build_chain()

    # Calibrate FK offset (arm stationary)
    q_home_rad, fk_offset = calibrate_fk_offset(r, chain, verbose=args.verbose)

    # Pose list
    if args.pose == "sequence":
        poses = [(n, POSES[n]) for n in DEFAULT_SEQUENCE if n in POSES]
    elif args.pose == "all":
        poses = list(POSES.items())
    else:
        poses = [(args.pose, POSES[args.pose])]

    pos_tol_m = args.pos_tol_mm / 1000.0
    results = {}

    print(f"\nStep J3 — testing {len(poses)} pose(s): {[p for p,_ in poses]}")
    print(f"  tol={args.pos_tol_mm:.0f}mm  move={args.move_s:.1f}s  settle={args.settle_s:.0f}s  delta_limit={args.delta_limit_deg:.0f}°")

    for pose_name, pose in poses:
        err = test_pose(
            r, chain, pose_name, pose, q_home_rad, fk_offset,
            settle_s=args.settle_s,
            move_s=args.move_s,
            delta_limit_deg=args.delta_limit_deg,
            pos_tol_m=pos_tol_m,
            pause=args.pause,
            verbose=args.verbose,
        )
        results[pose_name] = err

    # Return to home — interpolated, same as any other segment
    print(f"\n[return home]")
    q_current = get_vec(r, SENSOR_JOINTS, 7)
    if q_current is not None:
        switch_ctrl(r, JOINT_CTRL)
        run_segment(r, "return_home", q_current, q_home_rad,
                    move_s=args.move_s, hold_s=args.settle_s,
                    publish_hz=100.0, verbose=False)
    q_final = get_vec(r, SENSOR_JOINTS, 7)
    if q_final is not None:
        home_err = np.degrees(np.abs(q_final - q_home_rad)).max()
        print(f"  home err: {home_err:.2f}°")

    # Summary
    print(f"\n{'='*68}")
    print("STEP J3 SUMMARY")
    print(f"  fk_offset magnitude: {np.linalg.norm(fk_offset)*1000:.1f} mm")
    any_pass = False
    for pose_name, err in results.items():
        if err is None:
            status = "SKIP"
        elif err <= pos_tol_m:
            status = f"PASS  ({err*1000:.1f} mm)"
            any_pass = True
        else:
            status = f"FAIL  ({err*1000:.1f} mm > {args.pos_tol_mm:.0f} mm tol)"
        print(f"  {pose_name:20s}  {status}")
    all_pass = all(e is not None and e <= pos_tol_m for e in results.values())
    print(f"\n  OVERALL: {'PASS — IK pipeline validated, ready for J4' if all_pass else 'needs review (see above)'}")
    print(f"{'='*68}")

    # Leave in joint_controller at home
    switch_ctrl(r, JOINT_CTRL)
    set_vec(r, GOAL_JOINTS, q_home_rad.tolist())


if __name__ == "__main__":
    main()
