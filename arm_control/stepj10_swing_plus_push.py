#!/usr/bin/env python3
"""
Step J10: J9 impact-velocity swing + mobile base push forward.

Same arm pipeline as ``stepj9_impact_velocity_planner.py``, but also writes
``sports_bot::cmd::base::goal_pose`` each control tick so the TidyBot cart
nudges forward in world +X while J9 is tracking (default +0.50 m).  Requires
``base_bridge.py`` and ``redis_driver.py`` running.

Prereqs (in order)
------------------
1. redis-server, OpenSai, OptiTrack streamer (ball + cart).
2. TidyBot ``redis_driver.py``.
3. ``python sports_bot/base_bridge.py --robot-rigid-body-id <CART_ID>``.
4. This script (run from OpenSai root on tidybot01).

Usage
-----
    python sports_bot/arm_control/stepj10_swing_plus_push.py --skip-cal \\
        --ball-rigid-body-id 1 --strike-plane-x 0.65 --w-ori 10

    # Custom forward nudge (default 0.50 m):
    python sports_bot/arm_control/stepj10_swing_plus_push.py --skip-cal \\
        --base-forward-nudge-m 0.50 ...
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import redis

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_OPENSAI_DIR = os.path.dirname(os.path.dirname(_THIS_DIR))
for _path in (_THIS_DIR, _OPENSAI_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from stepj9_base_push import (  # noqa: E402
    BasePusher,
    FSM_BASE_GOAL,
    _parse_flag_float,
    _parse_flag_int,
    _parse_flag_str,
    _read_cart_control_pose_W,
)
from sports_bot.utils.frames import (  # noqa: E402
    arm_base_offset_calibration_path,
    load_arm_base_offset_calibration,
    load_robot_marker_calibration,
    robot_marker_calibration_path,
)


def _build_j10_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="J10: J9 impact-velocity swing + base forward push.",
        add_help=False,
    )
    ap.add_argument("--ready-base-x", type=float, default=0.0,
                    help="World-frame base X while waiting (m).")
    ap.add_argument("--ready-base-y", type=float, default=0.0,
                    help="World-frame base Y while waiting (m).")
    ap.add_argument("--ready-base-yaw-deg", type=float, default=0.0,
                    help="Base yaw at ready (deg); ignored with --ready-from-cart.")
    ap.add_argument("--ready-from-cart", dest="ready_from_cart", action="store_true", default=True,
                    help="Set ready pose from current cart OptiTrack pose at startup (default).")
    ap.add_argument("--no-ready-from-cart", dest="ready_from_cart", action="store_false",
                    help="Use --ready-base-x/y/yaw-deg literally (may lurch if not where cart is).")
    ap.add_argument("--base-x-min", type=float, default=-0.30,
                    help="Min world X for base goals (m).")
    ap.add_argument("--base-x-max", type=float, default=0.60,
                    help="Max world X for base goals (m).")
    ap.add_argument("--base-y-min", type=float, default=-0.80,
                    help="Min world Y for base goals (m).")
    ap.add_argument("--base-y-max", type=float, default=0.80,
                    help="Max world Y for base goals (m).")
    ap.add_argument("--base-x-gain", type=float, default=1.0,
                    help="Scale forward push: goal_x = ready_x + gain*(strike_x - strike_plane_x).")
    ap.add_argument("--base-hold-after-lost-s", type=float, default=0.5,
                    help="Return base to ready after this long without tracking.")
    ap.add_argument("--base-forward-nudge-m", type=float, default=0.50,
                    help="While J9 is tracking, command ready + this world +X offset (default).")
    ap.add_argument("--base-track-intercept", action="store_true",
                    help="Track strike X/Y instead of a fixed forward nudge.")
    ap.add_argument("--verbose-base", action="store_true",
                    help="Print base goal updates ~4 Hz.")
    ap.add_argument("--no-base-push", action="store_true",
                    help="Run J9 arm planner only (no base goal writes).")
    ap.add_argument("-h", "--help", action="store_true",
                    help="Show J9 help (pass --help after J10-only flags).")
    return ap


def main() -> None:
    j10_ap = _build_j10_parser()
    j10_args, remaining = j10_ap.parse_known_args()

    if j10_args.help:
        import stepj9_impact_velocity_planner as j9  # noqa: WPS433

        sys.argv = [sys.argv[0], "--help"]
        j9.main()
        return

    strike_plane_x = _parse_flag_float(remaining, "--strike-plane-x", -0.55)
    ready_yaw = math.radians(j10_args.ready_base_yaw_deg)
    ready_pose = (j10_args.ready_base_x, j10_args.ready_base_y, ready_yaw)

    import stepj9_impact_velocity_planner as j9  # noqa: WPS433

    base: BasePusher | None = None
    if not j10_args.no_base_push:
        redis_host = _parse_flag_str(remaining, "--redis-host", "localhost")
        redis_port = _parse_flag_int(remaining, "--redis-port", 6379)
        r = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        try:
            r.ping()
        except redis.exceptions.ConnectionError as e:
            sys.exit(f"[J10] cannot reach Redis: {e}")

        if j10_args.ready_from_cart:
            arm_cal = load_arm_base_offset_calibration(arm_base_offset_calibration_path())
            T_B_C = load_robot_marker_calibration(robot_marker_calibration_path())
            cart_pose = _read_cart_control_pose_W(r, arm_cal.base_rigid_body_id, T_B_C)
            if cart_pose is None:
                sys.exit(
                    f"[J10] --ready-from-cart: cart rigid body {arm_cal.base_rigid_body_id} "
                    f"not visible in OptiTrack. Start the streamer or pass --no-ready-from-cart "
                    f"with explicit --ready-base-x/y."
                )
            ready_pose = cart_pose
            print(
                f"[J10] ready from cart rb={arm_cal.base_rigid_body_id}: "
                f"({ready_pose[0]:+.3f}, {ready_pose[1]:+.3f}, {math.degrees(ready_pose[2]):+.1f}°)"
            )

        base = BasePusher(
            r,
            ready_pose=ready_pose,
            strike_plane_x=strike_plane_x,
            base_x_min=j10_args.base_x_min,
            base_x_max=j10_args.base_x_max,
            base_y_min=j10_args.base_y_min,
            base_y_max=j10_args.base_y_max,
            x_gain=j10_args.base_x_gain,
            hold_after_lost_s=j10_args.base_hold_after_lost_s,
            forward_nudge_m=j10_args.base_forward_nudge_m,
            track_intercept=j10_args.base_track_intercept,
            verbose=j10_args.verbose_base,
        )
        base.write_ready()
        if j10_args.base_track_intercept:
            mode_txt = (
                f"track intercept  x∈[{j10_args.base_x_min:+.2f},{j10_args.base_x_max:+.2f}] "
                f"y∈[{j10_args.base_y_min:+.2f},{j10_args.base_y_max:+.2f}]  "
                f"x_gain={j10_args.base_x_gain:.2f}  strike_plane_x={strike_plane_x:+.3f}"
            )
        else:
            mode_txt = f"forward nudge +{j10_args.base_forward_nudge_m:.2f} m (world +X)"
        print(
            f"[J10] base push ON → {FSM_BASE_GOAL}  "
            f"ready=({ready_pose[0]:+.3f}, {ready_pose[1]:+.3f}, "
            f"{math.degrees(ready_pose[2]):+.1f}°)  "
            f"{mode_txt}"
        )
        print("[J10] requires base_bridge.py + TidyBot redis_driver.py")
    else:
        print("[J10] --no-base-push: arm-only (same as J9)")

    orig_run_loop = j9.run_loop
    loop_hook = base.on_loop_tick if base is not None else None

    def run_loop_with_base(**kwargs):
        kwargs["loop_hook"] = loop_hook
        return orig_run_loop(**kwargs)

    j9.run_loop = run_loop_with_base

    old_argv = sys.argv
    sys.argv = [old_argv[0]] + remaining
    try:
        j9.main()
    finally:
        sys.argv = old_argv
        j9.run_loop = orig_run_loop


if __name__ == "__main__":
    main()
