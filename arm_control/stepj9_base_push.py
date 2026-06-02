#!/usr/bin/env python3
"""
Step J9: J8 through-strike planner + mobile base push toward intercept.

Same arm pipeline as ``stepj8_through_strike_planner.py``, but also writes
``sports_bot::cmd::base::goal_pose`` each control tick so the TidyBot cart
drives toward the predicted strike (forward + lateral).  Requires
``base_bridge.py`` and ``redis_driver.py`` running.

Prereqs (in order)
------------------
1. redis-server, OpenSai, OptiTrack streamer (ball + cart).
2. TidyBot ``redis_driver.py``.
3. ``python sports_bot/base_bridge.py --robot-rigid-body-id <CART_ID>``.
4. This script (run from OpenSai root on tidybot01).

Usage
-----
    python sports_bot/arm_control/stepj9_base_push.py --skip-cal \\
        --ball-rigid-body-id 1 --strike-plane-x 0.65 --w-ori 10

    python sports_bot/arm_control/stepj9_base_push.py --skip-cal --no-commit \\
        --ball-rigid-body-id 1 --strike-plane-x 0.65 --verbose-tracking
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time

import numpy as np
import redis

FSM_BASE_GOAL = "sports_bot::cmd::base::goal_pose"


def _parse_flag_str(argv: list[str], flag: str, default: str) -> str:
    for i, tok in enumerate(argv):
        if tok == flag and i + 1 < len(argv):
            return argv[i + 1]
    return default


def _parse_flag_int(argv: list[str], flag: str, default: int) -> int:
    for i, tok in enumerate(argv):
        if tok == flag and i + 1 < len(argv):
            try:
                return int(argv[i + 1])
            except ValueError:
                return default
    return default


def _parse_flag_float(argv: list[str], flag: str, default: float) -> float:
    for i, tok in enumerate(argv):
        if tok == flag and i + 1 < len(argv):
            try:
                return float(argv[i + 1])
            except ValueError:
                return default
    return default


class BasePusher:
    """Write world-frame base goals from J8 loop hook state."""

    def __init__(
        self,
        r: redis.Redis,
        *,
        ready_pose: tuple[float, float, float],
        strike_plane_x: float,
        base_x_min: float,
        base_x_max: float,
        base_y_min: float,
        base_y_max: float,
        x_gain: float = 1.0,
        hold_after_lost_s: float = 0.5,
        verbose: bool = False,
    ) -> None:
        self.r = r
        self.ready = (float(ready_pose[0]), float(ready_pose[1]), float(ready_pose[2]))
        self.strike_plane_x = float(strike_plane_x)
        self.base_x_min = float(base_x_min)
        self.base_x_max = float(base_x_max)
        self.base_y_min = float(base_y_min)
        self.base_y_max = float(base_y_max)
        self.x_gain = float(x_gain)
        self.hold_after_lost_s = float(hold_after_lost_s)
        self.verbose = verbose
        self.last_goal: tuple[float, float, float] | None = None
        self.last_good_t = float("-inf")
        self.last_print_t = float("-inf")

    @staticmethod
    def _clamp(v: float, lo: float, hi: float) -> tuple[float, bool]:
        if v < lo:
            return lo, True
        if v > hi:
            return hi, True
        return v, False

    def goal_for_strike_W(self, strike_W: np.ndarray) -> tuple[tuple[float, float, float], bool]:
        """Map strike point to base SE2 goal (same idea as SwingPlanner._base_pose_for_strike)."""
        strike_W = np.asarray(strike_W, dtype=float).reshape(3)
        x_raw = self.ready[0] + self.x_gain * (float(strike_W[0]) - self.strike_plane_x)
        y_raw = float(strike_W[1])
        x, x_clip = self._clamp(x_raw, self.base_x_min, self.base_x_max)
        y, y_clip = self._clamp(y_raw, self.base_y_min, self.base_y_max)
        theta = self.ready[2]
        return (x, y, theta), (x_clip or y_clip)

    def write_goal(self, pose: tuple[float, float, float]) -> None:
        pose = (float(pose[0]), float(pose[1]), float(pose[2]))
        if self.last_goal is None or any(abs(pose[i] - self.last_goal[i]) > 1e-4 for i in range(3)):
            self.r.set(FSM_BASE_GOAL, json.dumps([pose[0], pose[1], pose[2]]))
            self.last_goal = pose

    def write_ready(self) -> None:
        self.write_goal(self.ready)

    def on_loop_tick(self, state: dict) -> None:
        mode = str(state.get("mode", "idle"))
        intercept = state.get("intercept")
        active_cand = state.get("active_cand")

        strike_W = None
        if active_cand is not None:
            strike_W = np.asarray(active_cand.strike_W, dtype=float)
        elif intercept is not None:
            strike_W = np.asarray(intercept.position, dtype=float)

        now = time.perf_counter()
        if strike_W is not None and mode in ("tracking", "reject", "post_impact"):
            goal, clipped = self.goal_for_strike_W(strike_W)
            self.write_goal(goal)
            self.last_good_t = now
            if self.verbose and now - self.last_print_t >= 0.25:
                self.last_print_t = now
                clip_tag = " [CLIP]" if clipped else ""
                print(
                    f"[J9 base] {mode:8s} goal_W=({goal[0]:+.3f}, {goal[1]:+.3f}, "
                    f"{math.degrees(goal[2]):+.1f}°) strike_W={strike_W.round(3).tolist()}"
                    f"{clip_tag}"
                )
            return

        if mode in ("idle", "post_impact") or now - self.last_good_t > self.hold_after_lost_s:
            self.write_ready()
            if self.verbose and now - self.last_print_t >= 0.50:
                self.last_print_t = now
                print(f"[J9 base] {mode:8s} → ready ({self.ready[0]:+.3f}, {self.ready[1]:+.3f})")


def _build_j9_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="J9: J8 through-strike + base push toward intercept.",
        add_help=False,
    )
    ap.add_argument("--ready-base-x", type=float, default=0.0,
                    help="World-frame base X while waiting (m).")
    ap.add_argument("--ready-base-y", type=float, default=0.0,
                    help="World-frame base Y while waiting (m).")
    ap.add_argument("--ready-base-yaw-deg", type=float, default=0.0,
                    help="Base yaw at ready (deg).")
    ap.add_argument("--base-x-min", type=float, default=-0.30,
                    help="Min world X for base goals (m).")
    ap.add_argument("--base-x-max", type=float, default=0.30,
                    help="Max world X for base goals (m).")
    ap.add_argument("--base-y-min", type=float, default=-0.80,
                    help="Min world Y for base goals (m).")
    ap.add_argument("--base-y-max", type=float, default=0.80,
                    help="Max world Y for base goals (m).")
    ap.add_argument("--base-x-gain", type=float, default=1.0,
                    help="Scale forward push: goal_x = ready_x + gain*(strike_x - strike_plane_x).")
    ap.add_argument("--base-hold-after-lost-s", type=float, default=0.5,
                    help="Return base to ready after this long without a strike target.")
    ap.add_argument("--verbose-base", action="store_true",
                    help="Print base goal updates ~4 Hz.")
    ap.add_argument("--no-base-push", action="store_true",
                    help="Run J8 arm planner only (no base goal writes).")
    ap.add_argument("-h", "--help", action="store_true",
                    help="Show J8 help (pass --help after J9-only flags).")
    return ap


def main() -> None:
    j9_ap = _build_j9_parser()
    j9_args, remaining = j9_ap.parse_known_args()

    if j9_args.help:
        import stepj8_through_strike_planner as j8  # noqa: WPS433

        sys.argv = [sys.argv[0], "--help"]
        j8.main()
        return

    strike_plane_x = _parse_flag_float(remaining, "--strike-plane-x", -0.55)
    ready_yaw = math.radians(j9_args.ready_base_yaw_deg)
    ready_pose = (j9_args.ready_base_x, j9_args.ready_base_y, ready_yaw)

    import stepj8_through_strike_planner as j8  # noqa: WPS433

    base: BasePusher | None = None
    if not j9_args.no_base_push:
        redis_host = _parse_flag_str(remaining, "--redis-host", "localhost")
        redis_port = _parse_flag_int(remaining, "--redis-port", 6379)
        r = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        try:
            r.ping()
        except redis.exceptions.ConnectionError as e:
            sys.exit(f"[J9] cannot reach Redis: {e}")

        base = BasePusher(
            r,
            ready_pose=ready_pose,
            strike_plane_x=strike_plane_x,
            base_x_min=j9_args.base_x_min,
            base_x_max=j9_args.base_x_max,
            base_y_min=j9_args.base_y_min,
            base_y_max=j9_args.base_y_max,
            x_gain=j9_args.base_x_gain,
            hold_after_lost_s=j9_args.base_hold_after_lost_s,
            verbose=j9_args.verbose_base,
        )
        base.write_ready()
        print(
            f"[J9] base push ON → {FSM_BASE_GOAL}  "
            f"ready=({ready_pose[0]:+.3f}, {ready_pose[1]:+.3f}, "
            f"{math.degrees(ready_pose[2]):+.1f}°)  "
            f"x∈[{j9_args.base_x_min:+.2f},{j9_args.base_x_max:+.2f}] "
            f"y∈[{j9_args.base_y_min:+.2f},{j9_args.base_y_max:+.2f}]  "
            f"x_gain={j9_args.base_x_gain:.2f}  strike_plane_x={strike_plane_x:+.3f}"
        )
        print("[J9] requires base_bridge.py + TidyBot redis_driver.py")
    else:
        print("[J9] --no-base-push: arm-only (same as J8)")

    orig_run_loop = j8.run_loop
    loop_hook = base.on_loop_tick if base is not None else None

    def run_loop_with_base(**kwargs):
        kwargs["loop_hook"] = loop_hook
        return orig_run_loop(**kwargs)

    j8.run_loop = run_loop_with_base

    old_argv = sys.argv
    sys.argv = [old_argv[0]] + remaining
    try:
        j8.main()
    finally:
        sys.argv = old_argv
        j8.run_loop = orig_run_loop


if __name__ == "__main__":
    main()
