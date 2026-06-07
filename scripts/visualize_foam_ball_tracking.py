"""Replay foam-ball recordings through the drag-aware tracker and show Viser.

This is the foam-ball companion to ``state_machine.ball_tracker_test analyze``.
It uses the same recording format and Viser UI, but predictions and the green
fit curve come from ``FoamBallTracker``:

    dv/dt = [0, 0, -g] - k * ||v|| * v

Example:
    python sports_bot/scripts/visualize_foam_ball_tracking.py \
        sports_bot/recordings/new_ball_20260601_083510.npz \
        --strike-plane-x -0.30 \
        --gravity 11.0 \
        --drag-coefficient 0.10
"""

from __future__ import annotations

import argparse
import collections
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np


def _repo_root() -> Path:
    # This file lives at <repo>/sports_bot/scripts/visualize_foam_ball_tracking.py.
    return Path(__file__).resolve().parents[2]


REPO_ROOT = _repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sports_bot.foam_ball import FoamBallTracker  # noqa: E402
from sports_bot.foam_ball.config import FoamBallConfig  # noqa: E402
from sports_bot.state_machine import ball_tracker_test as analyzer  # noqa: E402
from sports_bot.state_machine.ball_tracker import BallSample, REJECT_NONE  # noqa: E402
from sports_bot.state_machine.config import PickleballConfig  # noqa: E402


class _ReplayFoamBallTracker(FoamBallTracker):
    """FoamBallTracker driven by explicit recording timestamps and positions."""

    def __init__(self, cfg: FoamBallConfig) -> None:
        # Avoid Redis wiring; mirror BallTracker.__init__ fields used by
        # ingest, _fit_state, reset, and FoamBallTracker.predict_intercept.
        self._redis = None
        self._keys = None
        self._cfg = cfg
        self._history = collections.deque(maxlen=cfg.history_size)
        med_w = max(1, cfg.median_filter_window)
        self._raw_buffer = collections.deque(maxlen=med_w)
        self._t0 = 0.0
        self._last_seen_t = None
        self._last_raw_pos = None
        self._last_raw_change_t = None
        self.last_reject_reason = REJECT_NONE

    def ingest(self, t: float, pos: np.ndarray) -> Optional[BallSample]:
        if pos is None or pos.shape != (3,):
            return None
        pos = pos.astype(float)

        if self._cfg.median_filter_window >= 2:
            self._raw_buffer.append(pos.copy())
            if len(self._raw_buffer) >= self._cfg.median_filter_window:
                pos = np.median(np.stack(self._raw_buffer), axis=0)

        while self._history and (float(t) - self._history[0].t) > self._cfg.history_max_age_s:
            self._history.popleft()

        if self._history:
            last = self._history[-1]
            dist = float(np.linalg.norm(pos - last.pos))
            if dist > self._cfg.max_position_jump:
                return None
            if self._cfg.max_implied_speed_mps > 0:
                dt_sample = float(t) - last.t
                if dt_sample > 1e-6 and dist / dt_sample > self._cfg.max_implied_speed_mps:
                    return None

        sample = BallSample(t=float(t), pos=pos)
        self._history.append(sample)
        self._last_seen_t = float(t)
        if self._cfg.online_bounce_pruning:
            self._try_prune_pre_bounce()
        return sample


def _replay_foam(
    timestamps: np.ndarray,
    positions: np.ndarray,
    cfg: PickleballConfig,
) -> List[analyzer.TickAnalysis]:
    tracker = _ReplayFoamBallTracker(cfg.tracker)
    strike_plane_x = cfg.court.strike_plane_x
    out: List[analyzer.TickAnalysis] = []

    last_t: Optional[float] = None
    last_pos: Optional[np.ndarray] = None
    reset_gap_s = max(cfg.tracker.history_max_age_s, 0.3)
    min_incoming = cfg.tracker.min_incoming_speed
    not_incoming_streak = 0
    not_incoming_streak_limit = 5

    for t, pos in zip(timestamps, positions):
        t = float(t)
        if last_t is not None:
            dt = t - last_t
            if dt > reset_gap_s:
                tracker.reset()
                not_incoming_streak = 0
            elif last_pos is not None and dt > 1e-6:
                v_x = float((pos[0] - last_pos[0]) / dt)
                if v_x >= -min_incoming:
                    not_incoming_streak += 1
                else:
                    if not_incoming_streak >= not_incoming_streak_limit:
                        tracker.reset()
                    not_incoming_streak = 0
        last_t = t
        last_pos = pos

        accepted = tracker.ingest(t, pos) is not None
        rec = analyzer.TickAnalysis(
            t=t,
            ball_pos=pos.copy(),
            accepted=accepted,
            history_size=len(tracker._history),
        )

        if accepted:
            fit = tracker._fit_state()
            if fit is not None:
                p0, v0, _ = fit
                rec.fit_p0 = p0
                rec.fit_v0 = v0
                rec.is_incoming = bool(v0[0] < -tracker._cfg.min_incoming_speed)
            intercept = tracker.predict_intercept(strike_plane_x)
            rec.reject_reason = getattr(tracker, "last_reject_reason", REJECT_NONE)
            if intercept is not None:
                rec.intercept_pos = intercept.position
                rec.intercept_vel = intercept.velocity
                rec.time_to_impact = float(intercept.time_to_impact)
                rec.n_bounces_in_prediction = int(intercept.n_bounces)

        out.append(rec)
    return out


def _drag_path(
    p0: np.ndarray,
    v0: np.ndarray,
    target_x: float,
    cfg: PickleballConfig,
    n_per_segment: int = 25,
) -> np.ndarray:
    """Polyline of the drag+gravity prediction used by FoamBallTracker."""
    points: List[np.ndarray] = [p0.astype(float).copy()]
    p = p0.astype(float).copy()
    v = v0.astype(float).copy()
    g = float(cfg.tracker.gravity)
    k = float(getattr(cfg.tracker, "drag_coefficient", 0.0))
    dt = float(getattr(cfg.tracker, "simulation_dt", 0.001))
    n_bounces = 0
    t_total = 0.0
    max_t = cfg.tracker.max_lookahead + dt
    samples_per_step = max(1, int(np.ceil(n_per_segment / 25)))

    if v[0] >= 0 or p[0] <= target_x:
        return np.asarray(points)

    while t_total < max_t:
        s = float(np.linalg.norm(v))
        v_new = np.array([
            v[0] - k * s * v[0] * dt,
            v[1] - k * s * v[1] * dt,
            v[2] - (g + k * s * v[2]) * dt,
        ])
        p_new = p + v_new * dt

        if p_new[0] <= target_x and p[0] > target_x:
            denom = max(p[0] - p_new[0], 1e-12)
            alpha = float(np.clip((p[0] - target_x) / denom, 0.0, 1.0))
            p_cross = np.array([
                target_x,
                p[1] + (p_new[1] - p[1]) * alpha,
                p[2] + (p_new[2] - p[2]) * alpha,
            ])
            points.append(p_cross)
            return np.asarray(points)

        if (
            p[2] > cfg.tracker.floor_epsilon
            and p_new[2] <= cfg.tracker.floor_epsilon
            and v_new[2] <= 0
        ):
            if n_bounces >= cfg.tracker.max_bounces:
                points.append(p_new)
                return np.asarray(points)
            denom = max(p[2] - p_new[2], 1e-12)
            alpha = float(np.clip((p[2] - cfg.tracker.floor_epsilon) / denom, 0.0, 1.0))
            p_bounce = np.array([
                p[0] + (p_new[0] - p[0]) * alpha,
                p[1] + (p_new[1] - p[1]) * alpha,
                cfg.tracker.floor_epsilon,
            ])
            points.append(p_bounce)
            p = p_bounce
            v = np.array([
                v_new[0] * cfg.tracker.bounce_tangential_damping,
                v_new[1] * cfg.tracker.bounce_tangential_damping,
                -cfg.tracker.bounce_restitution * v_new[2],
            ])
            n_bounces += 1
            t_total += alpha * dt
            continue

        if p_new[2] <= cfg.tracker.floor_epsilon and v_new[2] <= 0:
            points.append(p_new)
            return np.asarray(points)

        if len(points) == 1 or samples_per_step == 1 or len(points) % samples_per_step == 0:
            points.append(p_new)
        p = p_new
        v = v_new
        t_total += dt

    return np.asarray(points)


def _build_cfg(args: argparse.Namespace) -> PickleballConfig:
    cfg = PickleballConfig()
    cfg.tracker = FoamBallConfig()
    cfg.court.strike_plane_x = args.strike_plane_x
    cfg.tracker.gravity = args.gravity
    cfg.tracker.drag_coefficient = args.drag_coefficient
    cfg.tracker.simulation_dt = args.simulation_dt
    cfg.tracker.history_size = args.history_size
    cfg.tracker.history_max_age_s = args.history_max_age_s
    cfg.tracker.min_history_for_prediction = args.min_history_for_prediction
    cfg.tracker.median_filter_window = args.median_filter_window
    cfg.tracker.max_bounces = args.max_bounces
    cfg.tracker.online_bounce_pruning = not args.no_online_bounce_pruning
    cfg.tracker.min_incoming_speed = args.min_incoming_speed
    cfg.tracker.max_implied_speed_mps = args.max_implied_speed_mps
    cfg.tracker.max_position_jump = args.max_position_jump
    cfg.tracker.min_lookahead = args.min_lookahead
    cfg.tracker.max_lookahead = args.max_lookahead
    return cfg


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replay .npz ball recordings through FoamBallTracker and visualize in Viser.",
    )
    parser.add_argument("recording", help="Path to a .npz recording.")
    parser.add_argument("--viser-port", type=int, default=8080)
    parser.add_argument("--no-viser", action="store_true")
    parser.add_argument("--strike-plane-x", type=float, default=-0.30)
    parser.add_argument("--gravity", "--tracker-gravity", dest="gravity", type=float, default=11.0)
    parser.add_argument("--drag-coefficient", "--tracker-drag-coefficient", dest="drag_coefficient", type=float, default=0.10)
    parser.add_argument("--simulation-dt", "--tracker-simulation-dt", dest="simulation_dt", type=float, default=0.001)
    parser.add_argument("--history-size", "--tracker-history-size", dest="history_size", type=int, default=12)
    parser.add_argument("--history-max-age-s", "--tracker-history-max-age-s", dest="history_max_age_s", type=float, default=0.15)
    parser.add_argument("--min-history-for-prediction", "--tracker-min-history", dest="min_history_for_prediction", type=int, default=4)
    parser.add_argument("--median-filter-window", "--tracker-median-window", dest="median_filter_window", type=int, default=3)
    parser.add_argument("--max-bounces", type=int, default=0)
    parser.add_argument("--no-online-bounce-pruning", action="store_true")
    parser.add_argument("--min-incoming-speed", type=float, default=0.5)
    parser.add_argument("--max-implied-speed-mps", "--tracker-max-implied-speed-mps", dest="max_implied_speed_mps", type=float, default=15.0)
    parser.add_argument("--max-position-jump", type=float, default=0.5)
    parser.add_argument("--min-lookahead", type=float, default=0.05)
    parser.add_argument("--max-lookahead", type=float, default=1.5)
    parser.add_argument("--segment-gap-s", type=float, default=0.4)
    parser.add_argument("--segment-min-samples", type=int, default=5)
    parser.add_argument("--segment-min-incoming-speed", type=float, default=None)
    parser.add_argument("--segment-lead-pad", type=int, default=20)
    parser.add_argument("--no-bounce-detection", action="store_true")
    args = parser.parse_args()

    rec_path = Path(args.recording)
    if not rec_path.exists():
        print(f"[foam-analyze] not found: {rec_path}", file=sys.stderr)
        return 1

    data = np.load(rec_path, allow_pickle=False)
    timestamps = np.asarray(data["timestamps"], dtype=float)
    positions = np.asarray(data["positions"], dtype=float)
    if timestamps.ndim != 1 or positions.ndim != 2 or positions.shape[1] != 3:
        print(
            f"[foam-analyze] malformed recording: timestamps {timestamps.shape}, "
            f"positions {positions.shape}",
            file=sys.stderr,
        )
        return 1
    if len(timestamps) == 0:
        print("[foam-analyze] recording is empty", file=sys.stderr)
        return 1

    cfg = _build_cfg(args)
    ball_source = str(data["ball_source"]) if "ball_source" in data.files else "unknown"

    print(f"[foam-analyze] {rec_path} ({ball_source})")
    print(f"[foam-analyze] {len(timestamps)} samples, duration {timestamps[-1] - timestamps[0]:.2f}s")
    print(
        "[foam-analyze] tracker=foam_drag "
        f"g={cfg.tracker.gravity:.2f}m/s^2 "
        f"drag_k={cfg.tracker.drag_coefficient:.3f} "
        f"sim_dt={cfg.tracker.simulation_dt:.4f}s "
        f"history={cfg.tracker.history_size}/{cfg.tracker.history_max_age_s:.2f}s "
        f"min_hist={cfg.tracker.min_history_for_prediction} "
        f"median={cfg.tracker.median_filter_window} "
        f"min_incoming={cfg.tracker.min_incoming_speed:.2f}m/s"
    )
    print(f"[foam-analyze] strike_plane_x = {cfg.court.strike_plane_x:.3f}")

    ticks = _replay_foam(timestamps, positions, cfg)
    seg_min_incoming = (
        args.segment_min_incoming_speed
        if args.segment_min_incoming_speed is not None
        else cfg.tracker.min_incoming_speed
    )
    throws = analyzer._segment_throws(
        timestamps,
        positions,
        ticks,
        cfg.court.strike_plane_x,
        gap_s=args.segment_gap_s,
        min_samples=args.segment_min_samples,
        detect_bounces=not args.no_bounce_detection,
        bounce_gravity=cfg.tracker.gravity,
        min_incoming_speed=seg_min_incoming,
        lead_pad_samples=args.segment_lead_pad,
    )
    print(f"[foam-analyze] segmented into {len(throws)} throw(s)")

    for throw in throws:
        analyzer._print_throw_summary(throw, cfg)
    analyzer._print_bounce_aggregate(throws)

    if not throws or args.no_viser:
        return 0

    try:
        import viser  # noqa: F401
    except ImportError:
        print(
            "[foam-analyze] viser not installed; skipping visualization.\n"
            "               install with: pip install viser",
            file=sys.stderr,
        )
        return 0

    # Reuse the existing UI while replacing its green fit curve with the
    # drag-aware trajectory that FoamBallTracker actually predicts.
    analyzer._bouncing_path = _drag_path
    analyzer._run_viser(throws, cfg, args.viser_port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
