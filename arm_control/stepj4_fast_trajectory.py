#!/usr/bin/env python3
"""
Step J4: Fast trajectory test.

Validates the joint controller at swing-relevant move times before wiring up
a reactive intercept loop (J5). Two phases:

  Phase A — speed sweep: IK-solve wind-up and strike positions, then drive
    wind-up → strike at decreasing move times (default: 2.0 → 1.0 → 0.5 → 0.3 s).
    Returns to wind-up slowly between runs. Reports tracking error and torques
    at each speed so we know the fastest clean tier.

  Phase B — mock swing: home → wind-up (slow) → strike (fast, short hold) →
    follow-through (fast) → home (slow). Not a real swing shape; purely a
    three-waypoint motion primitive to stress the wind-up → strike → follow
    transition in one continuous sequence.

Run from OpenSai root:
    python sports_bot/arm_control/stepj4_fast_trajectory.py
    python sports_bot/arm_control/stepj4_fast_trajectory.py --move-s 0.5
    python sports_bot/arm_control/stepj4_fast_trajectory.py --sweep-only
    python sports_bot/arm_control/stepj4_fast_trajectory.py --swing-only
    python sports_bot/arm_control/stepj4_fast_trajectory.py --verbose
"""
import argparse
import json
import sys
import time
import warnings

import numpy as np
import redis
from scipy.optimize import minimize

warnings.filterwarnings("ignore")
import ikpy.chain  # noqa: E402

from stepj1_joint_nudge import run_segment  # noqa: E402

# ---------------------------------------------------------------------------
# Redis keys
# ---------------------------------------------------------------------------
ROBOT_NAME     = "FrankaRobot"
NS             = f"opensai::controllers::{ROBOT_NAME}"
CART_CTRL      = "cartesian_controller"
JOINT_CTRL     = "joint_controller"
ACTIVE_CTRL    = f"opensai::controllers::{ROBOT_NAME}::active_controller_name"
SENSOR_JOINTS  = f"opensai::sensors::{ROBOT_NAME}::joint_positions"
CURR_POS       = f"{NS}::{CART_CTRL}::cartesian_task::current_position"
GOAL_JOINTS    = f"{NS}::{JOINT_CTRL}::joint_task::goal_position"
SAFETY_TORQUES = f"opensai::redis_driver::{ROBOT_NAME}::safety_controller::safety_torques"

URDF_PATH = "drivers/FrankaPanda/model/panda_arm.urdf"

Q_LO = np.radians([-166, -101, -166, -176, -166,  -1, -166])
Q_HI = np.radians([ 166,  101,  166,   -4,  166, 215,  166])

DEFAULT_SPEEDS = [2.0, 1.0, 0.5, 0.3]

# ---------------------------------------------------------------------------
# Helpers
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


def fmt_q(q_rad):
    return "  ".join(f"q{i+1}={np.degrees(v):+.1f}" for i, v in enumerate(q_rad))


def fmt_pos(p):
    return f"[{p[0]:+.4f}, {p[1]:+.4f}, {p[2]:+.4f}]"


def safety_tripped(r):
    tau = get_vec(r, SAFETY_TORQUES, 7)
    if tau is None:
        return False
    return bool(np.any(np.abs(tau) > 0.01))


# ---------------------------------------------------------------------------
# IK (same solver as J3)
# ---------------------------------------------------------------------------
def build_chain():
    return ikpy.chain.Chain.from_urdf_file(URDF_PATH, base_elements=["link0"])


def fk(chain, q_rad_7):
    q_ikpy = np.concatenate([[0.0], q_rad_7])
    return chain.forward_kinematics(q_ikpy)


def ik_solve(chain, target_pos_ikpy, q_init_rad_7, w_reg=0.001):
    """Position-only IK with regularization toward q_init (same as J3)."""
    def cost(q):
        q_ikpy = np.concatenate([[0.0], q])
        pos = chain.forward_kinematics(q_ikpy)[:3, 3]
        return np.sum((pos - target_pos_ikpy) ** 2) + w_reg * np.sum((q - q_init_rad_7) ** 2)
    bounds = list(zip(Q_LO, Q_HI))
    result = minimize(cost, q_init_rad_7, method="L-BFGS-B", bounds=bounds,
                      options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-8})
    q_out = result.x
    q_ikpy = np.concatenate([[0.0], q_out])
    err_m = np.linalg.norm(chain.forward_kinematics(q_ikpy)[:3, 3] - target_pos_ikpy)
    return q_out, err_m


# ---------------------------------------------------------------------------
# FK-offset calibration (same as J3: read SENSOR_JOINTS + CURR_POS together)
# ---------------------------------------------------------------------------
def calibrate_fk_offset(r, chain):
    switch_ctrl(r, CART_CTRL)
    time.sleep(0.1)
    q_rad = get_vec(r, SENSOR_JOINTS, 7)
    cp    = get_vec(r, CURR_POS, 3)
    if q_rad is None:
        sys.exit(f"ERROR: {SENSOR_JOINTS} not available")
    if cp is None:
        sys.exit(f"ERROR: {CURR_POS} not available — is cartesian_controller in the XML?")
    link7_pos = fk(chain, q_rad)[:3, 3]
    offset = cp - link7_pos
    print(f"\n[calibration]")
    print(f"  joints (deg):  {fmt_q(q_rad)}")
    print(f"  ikpy link7:    {fmt_pos(link7_pos)} m")
    print(f"  opensai ee:    {fmt_pos(cp)} m")
    print(f"  fk_offset:     {fmt_pos(offset)} m  (|{np.linalg.norm(offset)*1000:.1f}| mm)")
    return q_rad, offset


# ---------------------------------------------------------------------------
# Solve one IK waypoint, report, check delta
# ---------------------------------------------------------------------------
def solve_waypoint(chain, name, target_opensai, fk_offset, q_ref,
                   delta_limit_deg, ik_tol_m=0.005):
    target_ikpy = target_opensai - fk_offset
    q_ik, err_m = ik_solve(chain, target_ikpy, q_ref)
    delta_deg = np.degrees(np.abs(q_ik - q_ref))
    max_delta = float(delta_deg.max())
    fk_check = fk(chain, q_ik)[:3, 3] + fk_offset
    print(f"  {name:12s}  target={fmt_pos(target_opensai)}  "
          f"IK_err={err_m*1000:.2f} mm  max_Δ={max_delta:.1f}°")
    if err_m > ik_tol_m * 2:
        print(f"             WARNING: IK did not converge tightly ({err_m*1000:.1f} mm)", file=sys.stderr)
    if max_delta > delta_limit_deg:
        sys.exit(f"ERROR: {name} IK requires {max_delta:.1f}° from reference, "
                 f"exceeds --delta-limit-deg {delta_limit_deg:.0f}°. "
                 "Adjust strike position or offset.")
    return q_ik, err_m


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="J4: fast trajectory and mock swing speed test")
    ap.add_argument("--host", default="localhost")
    # Geometry
    ap.add_argument("--strike-x", type=float, default=0.55,
                    help="Strike EE position x in arm base frame (m)")
    ap.add_argument("--strike-y", type=float, default=0.05)
    ap.add_argument("--strike-z", type=float, default=0.52)
    ap.add_argument("--swing-dir", nargs=3, type=float, default=[1.0, 0.0, 0.0],
                    metavar=("DX", "DY", "DZ"),
                    help="Swing direction in arm base frame (default: pure +X forward)")
    ap.add_argument("--wind-up-offset", type=float, default=0.12,
                    help="Distance behind strike for wind-up (m, default 0.12)")
    ap.add_argument("--follow-offset", type=float, default=0.12,
                    help="Distance past strike for follow-through (m, default 0.12)")
    # Timing
    ap.add_argument("--speeds", nargs="+", type=float, default=DEFAULT_SPEEDS,
                    help="Move times (s) for the Phase A speed sweep (slow → fast)")
    ap.add_argument("--move-s", type=float, default=0.3,
                    help="Fast move time for Phase B wind-up→strike and strike→follow (default 0.3s)")
    ap.add_argument("--approach-s", type=float, default=2.0,
                    help="Slow move time for approach / return moves (default 2.0s)")
    ap.add_argument("--hold-s", type=float, default=0.8,
                    help="Hold time at each waypoint during sweep (default 0.8s)")
    ap.add_argument("--strike-hold-s", type=float, default=0.15,
                    help="Hold time at strike in Phase B mock swing (default 0.15s — brief)")
    ap.add_argument("--settle-s", type=float, default=2.0,
                    help="Hold time at home before and after each phase (default 2.0s)")
    # Safety / control
    ap.add_argument("--delta-limit-deg", type=float, default=50.0,
                    help="Max joint delta between any two consecutive waypoints (safety gate)")
    ap.add_argument("--sweep-only", action="store_true",
                    help="Only run Phase A (speed sweep); skip Phase B mock swing")
    ap.add_argument("--swing-only", action="store_true",
                    help="Only run Phase B (mock swing); skip Phase A sweep")
    ap.add_argument("--pause", action="store_true",
                    help="Pause for Enter before each move")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    r = redis.Redis(host=args.host, port=6379, decode_responses=True)
    try:
        r.ping()
    except Exception:
        sys.exit("ERROR: Redis not reachable")

    chain = build_chain()

    # Calibrate FK offset
    q_home_rad, fk_offset = calibrate_fk_offset(r, chain)

    # Normalize swing direction
    swing_dir = np.array(args.swing_dir, dtype=float)
    norm = np.linalg.norm(swing_dir)
    if norm < 1e-6:
        sys.exit("ERROR: --swing-dir cannot be zero vector")
    swing_dir /= norm

    # Compute target positions
    p_strike  = np.array([args.strike_x, args.strike_y, args.strike_z])
    p_wind_up = p_strike - args.wind_up_offset * swing_dir
    p_follow  = p_strike + args.follow_offset  * swing_dir

    print(f"\n[waypoints]")
    print(f"  swing_dir:  {swing_dir}")
    print(f"  wind-up:    {fmt_pos(p_wind_up)} m")
    print(f"  strike:     {fmt_pos(p_strike)} m")
    print(f"  follow:     {fmt_pos(p_follow)} m")
    print(f"  offsets:    wind-up {args.wind_up_offset*100:.0f} cm | follow {args.follow_offset*100:.0f} cm")

    # Solve IK for all three waypoints (chained from home so solutions stay close)
    print(f"\n[IK solve]  (chained from home → wind-up → strike → follow)")
    q_wind_up, err_wu = solve_waypoint(chain, "wind_up",  p_wind_up, fk_offset,
                                        q_home_rad, args.delta_limit_deg)
    q_strike,  err_st = solve_waypoint(chain, "strike",   p_strike,  fk_offset,
                                        q_wind_up,  args.delta_limit_deg)
    q_follow,  err_fw = solve_waypoint(chain, "follow",   p_follow,  fk_offset,
                                        q_strike,   args.delta_limit_deg)

    wu_to_strike_deg = float(np.degrees(np.abs(q_strike - q_wind_up)).max())
    print(f"\n  wind-up → strike max joint delta: {wu_to_strike_deg:.1f}°")
    print(f"  speeds to test: {sorted(args.speeds)} s")
    print(f"  (at 0.3s that's ~{wu_to_strike_deg/0.3:.0f} deg/s peak — "
          f"{100*wu_to_strike_deg/0.3/125:.0f}% of 125 deg/s limit)")

    # -------------------------------------------------------------------------
    # Helper: ensure joint controller, seed goal, optionally pause
    # -------------------------------------------------------------------------
    def prep_move(label):
        q_cur = get_vec(r, SENSOR_JOINTS, 7)
        if q_cur is None:
            sys.exit(f"ERROR: lost sensor joints before {label}")
        set_vec(r, GOAL_JOINTS, q_cur)
        switch_ctrl(r, JOINT_CTRL)
        if args.pause:
            input(f"  [pause] press Enter to run {label!r} ...")
        return q_cur

    # -------------------------------------------------------------------------
    # Phase A: speed sweep
    # -------------------------------------------------------------------------
    sweep_results = []
    if not args.swing_only:
        print(f"\n{'='*68}")
        print(f"PHASE A — speed sweep: wind-up → strike")
        speeds_sorted = sorted(args.speeds, reverse=True)  # slow first → fast last
        print(f"  order: {speeds_sorted} s (slow → fast)")

        # Approach wind-up from home
        q_cur = prep_move("approach to wind-up")
        run_segment(r, "home→wind_up", q_cur, q_wind_up,
                    move_s=args.approach_s, hold_s=args.hold_s,
                    publish_hz=100.0, verbose=args.verbose)

        for move_s in speeds_sorted:
            label = f"wu→strike  {move_s:.1f}s"
            if args.pause:
                input(f"  [pause] press Enter to run {label!r} ...")
            res = run_segment(r, label, q_wind_up, q_strike,
                              move_s=move_s, hold_s=args.hold_s,
                              publish_hz=100.0, verbose=args.verbose)
            res["move_s"] = move_s
            res["safety_tripped"] = safety_tripped(r)
            sweep_results.append(res)

            # Slow return to wind-up for the next tier
            run_segment(r, f"return_wu  {move_s:.1f}s", q_strike, q_wind_up,
                        move_s=args.approach_s, hold_s=args.hold_s,
                        publish_hz=100.0, verbose=args.verbose)

        # Return home after sweep
        run_segment(r, "sweep→home", q_wind_up, q_home_rad,
                    move_s=args.approach_s, hold_s=args.settle_s,
                    publish_hz=100.0, verbose=args.verbose)

    # -------------------------------------------------------------------------
    # Phase B: mock swing (home → wu → strike → follow → home)
    # -------------------------------------------------------------------------
    swing_results = []
    if not args.sweep_only:
        print(f"\n{'='*68}")
        print(f"PHASE B — mock swing at {args.move_s:.1f}s per fast segment")
        print(f"  home → wind-up ({args.approach_s:.1f}s) → strike ({args.move_s:.1f}s, "
              f"{args.strike_hold_s:.2f}s hold) → follow ({args.move_s:.1f}s) → home ({args.approach_s:.1f}s)")

        q_cur = prep_move("home → wind-up")
        run_segment(r, "home→wind_up", q_cur, q_wind_up,
                    move_s=args.approach_s, hold_s=args.hold_s,
                    publish_hz=100.0, verbose=args.verbose)

        if args.pause:
            input("  [pause] press Enter to execute the fast swing sequence ...")

        r1 = run_segment(r, "wind_up→strike", q_wind_up, q_strike,
                         move_s=args.move_s, hold_s=args.strike_hold_s,
                         publish_hz=100.0, verbose=args.verbose)
        r1["safety_tripped"] = safety_tripped(r)
        swing_results.append(r1)

        r2 = run_segment(r, "strike→follow", q_strike, q_follow,
                         move_s=args.move_s, hold_s=args.hold_s,
                         publish_hz=100.0, verbose=args.verbose)
        r2["safety_tripped"] = safety_tripped(r)
        swing_results.append(r2)

        run_segment(r, "follow→home", q_follow, q_home_rad,
                    move_s=args.approach_s, hold_s=args.settle_s,
                    publish_hz=100.0, verbose=args.verbose)

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print(f"\n{'='*68}")
    print("STEP J4 SUMMARY")
    print(f"  strike:   {fmt_pos(p_strike)} m")
    print(f"  wind-up:  {fmt_pos(p_wind_up)} m  (IK err {err_wu*1000:.1f} mm)")
    print(f"  follow:   {fmt_pos(p_follow)} m  (IK err {err_fw*1000:.1f} mm)")
    print(f"  wu→strike max joint delta: {wu_to_strike_deg:.1f}°")

    if sweep_results:
        print(f"\n  Phase A — wu → strike speed sweep:")
        hdr = f"  {'move_s':>7}  {'qdot_max':>12}  {'goal_err':>9}  {'safety':>8}  {'status':>7}"
        print(hdr)
        print(f"  {'-'*7}  {'-'*12}  {'-'*9}  {'-'*8}  {'-'*7}")
        for res in sweep_results:
            safety_s = "TRIP!" if res["safety_tripped"] else "clean"
            ok = not res["vel_warn"] and not res["safety_tripped"]
            status = "OK" if ok else "WARN"
            print(f"  {res['move_s']:>7.1f}s  "
                  f"{res['qdot_max_deg_s']:>10.1f}°/s  "
                  f"{res['goal_err_max_deg']:>9.3f}°  "
                  f"{safety_s:>8}  "
                  f"{status:>7}")

    if swing_results:
        print(f"\n  Phase B — mock swing ({args.move_s:.1f}s fast moves):")
        for res in swing_results:
            safety_s = "TRIP!" if res["safety_tripped"] else "clean"
            ok = not res["vel_warn"] and not res["safety_tripped"]
            status = "OK" if ok else "WARN"
            print(f"    {res['label']:22s}  "
                  f"qdot={res['qdot_max_deg_s']:.1f}°/s  "
                  f"err={res['goal_err_max_deg']:.3f}°  "
                  f"safety={safety_s}  [{status}]")

    all_results = sweep_results + swing_results
    any_tripped = any(res["safety_tripped"] for res in all_results)
    any_vel_warn = any(res["vel_warn"] for res in all_results)

    fastest_clean = None
    if sweep_results:
        clean = [r for r in sweep_results if not r["safety_tripped"] and not r["vel_warn"]]
        if clean:
            fastest_clean = min(r["move_s"] for r in clean)

    print()
    if any_tripped:
        print("  *** SAFETY TORQUES TRIPPED — do not proceed to J5 until resolved ***")
    elif any_vel_warn:
        print("  Velocity limit warnings present — review qdot_max above before J5")
    else:
        print("  Safety torques: clean throughout")

    if fastest_clean is not None:
        if fastest_clean <= 0.5:
            print(f"  Fastest clean speed: {fastest_clean:.1f}s move"
                  f"  →  PASS, ready for J5 (reactive intercept)")
        else:
            print(f"  Fastest clean speed: {fastest_clean:.1f}s move"
                  f"  →  consider tuning gains before J5")
    elif not any_tripped and sweep_results:
        print("  Check vel_warn flags above; if only marginal, J5 is likely fine")

    print(f"{'='*68}")

    # Leave joint controller active, arm at home
    switch_ctrl(r, JOINT_CTRL)
    set_vec(r, GOAL_JOINTS, q_home_rad)


if __name__ == "__main__":
    main()
