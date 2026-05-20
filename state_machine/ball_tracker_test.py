"""Test / tuning harness for BallTracker intercept prediction.

The production BallTracker is wired straight into the FSM, which makes it
inconvenient to evaluate on real OptiTrack throws. This script lets you:

  1. RECORD a ball trajectory from Redis (OptiTrack or OpenSai sim) to a .npz
     file. Just stand next to the bay, throw the ball, repeat. Hit Ctrl-C.

  2. ANALYZE a recording offline by replaying it through the *real*
     BallTracker code (no reimplementation) and visualizing every per-tick
     prediction in viser alongside the ground-truth trajectory and the
     actual strike-plane crossing. Tracker parameters can be overridden on
     the command line so you can A/B the same recording under different
     filter settings without re-throwing.

Examples:

  # Record throws from OptiTrack (current PickleBall ID is 8)
  python -m sports_bot.state_machine.ball_tracker_test record \\
      --ball-source optitrack --optitrack-rigid-body-id 8 \\
      --output recordings/throws_2026-05-17.npz

  # Visualize + print summary
  python -m sports_bot.state_machine.ball_tracker_test analyze \\
      recordings/throws_2026-05-17.npz

  # Re-analyze with tweaked tracker parameters
  python -m sports_bot.state_machine.ball_tracker_test analyze \\
      recordings/throws_2026-05-17.npz \\
      --history-size 8 --max-position-jump 0.3 --gravity 9.81

  # Skip viser (just print stats)
  python -m sports_bot.state_machine.ball_tracker_test analyze \\
      recordings/throws_2026-05-17.npz --no-viser
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import redis

from .ball_tracker import BallSample, BallTracker
from .config import PickleballConfig
from .ekf_ball_tracker import EKFBallTracker
from .redis_keys import RedisKeys


# =============================================================================
# Shared helpers
# =============================================================================

def _read_ball_position(r: redis.Redis, keys: RedisKeys) -> Optional[np.ndarray]:
    """Mirror BallTracker._read_position so the recorder doesn't need a tracker."""
    if keys.ball_source == "optitrack":
        raw = r.get(keys.ball.optitrack_position)
        if raw is None:
            return None
        try:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            arr = np.array(json.loads(raw))
            return arr if arr.shape == (3,) else None
        except (ValueError, json.JSONDecodeError):
            return None

    if keys.ball_source == "opensai":
        raw = r.get(keys.ball.opensai_object_pose)
        if raw is None:
            return None
        try:
            pose = np.array(json.loads(raw))
            if pose.shape == (4, 4):
                return pose[0:3, 3]
            if pose.shape == (3,):
                return pose
        except (ValueError, json.JSONDecodeError):
            return None
    return None


def _make_keys(ball_source: str, optitrack_id: int) -> RedisKeys:
    keys = RedisKeys(ball_source=ball_source)
    if ball_source == "optitrack":
        keys.ball = dataclasses.replace(keys.ball, optitrack_rigid_body_id=optitrack_id)
    return keys


def _resolved_ball_key(keys: RedisKeys) -> str:
    if keys.ball_source == "optitrack":
        return keys.ball.optitrack_position
    return keys.ball.opensai_object_pose


# =============================================================================
# RECORD
# =============================================================================

def _default_recording_path() -> Path:
    """<repo>/sports_bot/recordings/throws_<timestamp>.npz, derived from this file's location."""
    # This file lives at <repo>/sports_bot/state_machine/ball_tracker_test.py
    sports_bot_root = Path(__file__).resolve().parents[1]
    return sports_bot_root / "recordings" / f"throws_{time.strftime('%Y%m%d_%H%M%S')}.npz"


def cmd_record(args: argparse.Namespace) -> int:
    if args.output is None:
        args.output = str(_default_recording_path())

    keys = _make_keys(args.ball_source, args.optitrack_rigid_body_id)
    redis_client = redis.Redis(host=args.redis_host, port=args.redis_port)
    ball_key = _resolved_ball_key(keys)

    if redis_client.get(ball_key) is None:
        print(f"[record] WARNING: nothing yet at {ball_key} (will keep polling)")
    else:
        print(f"[record] reading from {ball_key}")
    print(f"[record] sample rate: {1.0 / args.dt:.0f} Hz, min movement {args.min_movement*1000:.1f} mm")
    print(f"[record] saving to {args.output}")
    print("[record] press Ctrl-C to stop and save")

    timestamps: List[float] = []
    positions: List[np.ndarray] = []
    last_recorded: Optional[np.ndarray] = None
    last_print_t = -1.0

    running = [True]
    def _sigint(*_):
        running[0] = False
    signal.signal(signal.SIGINT, _sigint)
    signal.signal(signal.SIGTERM, _sigint)

    init_time = time.perf_counter()
    loop_time = 0.0
    dt = args.dt
    t0 = time.perf_counter()

    while running[0]:
        loop_time += dt
        sleep_for = loop_time - (time.perf_counter() - init_time)
        if sleep_for > 0:
            time.sleep(sleep_for)

        now = time.perf_counter() - t0
        pos = _read_ball_position(redis_client, keys)
        if pos is None:
            continue

        # Skip samples where the ball hasn't moved (OptiTrack publishes at
        # ~120 Hz even when the ball is stationary; we don't want to record
        # thousands of identical sitting-on-the-floor samples).
        if last_recorded is not None and np.linalg.norm(pos - last_recorded) < args.min_movement:
            continue
        last_recorded = pos.copy()

        timestamps.append(now)
        positions.append(pos)

        if now - last_print_t > 0.5:
            last_print_t = now
            print(f"[record] t={now:7.2f}s pos=[{pos[0]:+.3f} {pos[1]:+.3f} {pos[2]:+.3f}] "
                  f"(N={len(timestamps)})")

    if not timestamps:
        print("\n[record] no samples captured; not writing file")
        return 1

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        timestamps=np.asarray(timestamps, dtype=np.float64),
        positions=np.asarray(positions, dtype=np.float64),
        ball_source=args.ball_source,
        optitrack_rigid_body_id=args.optitrack_rigid_body_id,
    )
    duration = timestamps[-1] - timestamps[0]
    print(f"\n[record] saved {len(timestamps)} samples ({duration:.2f}s) to {out_path}")
    return 0


# =============================================================================
# ANALYZE
# =============================================================================

@dataclass
class TickAnalysis:
    """Per-tick output of replaying the tracker."""
    t: float
    ball_pos: np.ndarray
    accepted: bool                                # passed update()'s filter
    history_size: int = 0
    fit_p0: Optional[np.ndarray] = None
    fit_v0: Optional[np.ndarray] = None
    is_incoming: bool = False
    intercept_pos: Optional[np.ndarray] = None
    intercept_vel: Optional[np.ndarray] = None
    time_to_impact: Optional[float] = None
    n_bounces_in_prediction: int = 0              # bounces the predictor simulated
    # Set only by the EKF tracker. 3x3 position covariance at the predicted
    # intercept — diagonal sqrt entries are the per-axis position 1σ.
    intercept_pos_cov: Optional[np.ndarray] = None


@dataclass
class Bounce:
    """A bounce detected in a recorded trajectory."""
    t: float                # bounce time (s, relative to recording start)
    position: np.ndarray    # 3-vector approx bounce position (sample at min z)
    v_before: np.ndarray    # 3-velocity just before, fitted from local window
    v_after: np.ndarray     # 3-velocity just after, fitted from local window
    restitution: float      # e = -v_z_after / v_z_before
    mu_t: float             # ||v_xy_after|| / ||v_xy_before||


def _fit_velocity_at_time(
    ts: np.ndarray, ps: np.ndarray, t_eval: float, gravity: float
) -> np.ndarray:
    """Ballistic-aware least-squares velocity (xy linear, z free-fall) at t_eval.

    Same physics model as BallTracker._fit_state, parameterized so we can fit a
    short window before / after a candidate bounce and recover the velocity at
    the bounce instant (not at the window center).
    """
    dt = ts - t_eval
    z_corrected = ps[:, 2] + 0.5 * gravity * dt * dt
    A = np.stack([np.ones_like(dt), dt], axis=1)
    sx, *_ = np.linalg.lstsq(A, ps[:, 0], rcond=None)
    sy, *_ = np.linalg.lstsq(A, ps[:, 1], rcond=None)
    sz, *_ = np.linalg.lstsq(A, z_corrected, rcond=None)
    return np.array([sx[1], sy[1], sz[1]])


def _detect_bounces(
    timestamps: np.ndarray,
    positions: np.ndarray,
    *,
    gravity: float = 9.81,
    z_threshold_m: float = 0.10,
    pre_window_s: float = 0.12,
    post_window_s: float = 0.12,
    gap_s: float = 0.02,
    min_v_z_before: float = 0.5,
    min_v_z_after: float = 0.5,
    debounce_samples: int = 15,
) -> List[Bounce]:
    """Detect floor bounces as local z minima below z_threshold_m, with v_z
    going from clearly downward to clearly upward.

    For each candidate, fit ballistic velocity on a short window before and
    after (excluding ``gap_s`` around the bounce itself) and extrapolate to
    the bounce instant. Returns the resulting (e, μ_t) per bounce.
    """
    if len(timestamps) < 6:
        return []
    z = positions[:, 2]
    bounces: List[Bounce] = []
    last_bounce_idx = -10_000
    for i in range(2, len(z) - 2):
        if i - last_bounce_idx < debounce_samples:
            continue
        if z[i] >= z_threshold_m:
            continue
        if not (z[i] <= z[i - 1] and z[i] <= z[i + 1]):
            continue  # not a local min in z

        t_b = float(timestamps[i])
        m_before = (timestamps >= t_b - pre_window_s) & (timestamps <= t_b - gap_s)
        m_after = (timestamps >= t_b + gap_s) & (timestamps <= t_b + post_window_s)
        if m_before.sum() < 3 or m_after.sum() < 3:
            continue

        v_before = _fit_velocity_at_time(timestamps[m_before], positions[m_before], t_b, gravity)
        v_after = _fit_velocity_at_time(timestamps[m_after], positions[m_after], t_b, gravity)
        if v_before[2] >= -min_v_z_before:
            continue
        if v_after[2] <= min_v_z_after:
            continue

        e = float(-v_after[2] / v_before[2])
        vxy_b = float(np.linalg.norm(v_before[:2]))
        vxy_a = float(np.linalg.norm(v_after[:2]))
        mu_t = float(vxy_a / vxy_b) if vxy_b > 0.1 else 1.0

        bounces.append(Bounce(
            t=t_b,
            position=positions[i].copy(),
            v_before=v_before,
            v_after=v_after,
            restitution=e,
            mu_t=mu_t,
        ))
        last_bounce_idx = i
    return bounces


def _bouncing_path(
    p0: np.ndarray,
    v0: np.ndarray,
    target_x: float,
    cfg,
    n_per_segment: int = 25,
) -> np.ndarray:
    """Polyline of the predicted trajectory from (p0, v0) to where it crosses
    x = target_x, bending at predicted floor bounces. Same physics as
    BallTracker._propagate_to_plane; just collects the per-step points."""
    points: List[np.ndarray] = [p0.astype(float).copy()]
    p = p0.astype(float).copy()
    v = v0.astype(float).copy()
    g = cfg.tracker.gravity
    for _ in range(cfg.tracker.max_bounces + 1):
        if v[0] >= 0:
            return np.asarray(points)
        t_to_target = (target_x - p[0]) / v[0]
        if t_to_target < 0:
            return np.asarray(points)
        if p[2] <= cfg.tracker.floor_epsilon and v[2] <= 0:
            return np.asarray(points)
        disc = v[2] * v[2] + 2.0 * g * p[2]
        t_to_ground = (v[2] + np.sqrt(disc)) / g if disc >= 0 else float("inf")
        seg_dt = min(t_to_target, t_to_ground)
        ts = np.linspace(0.0, seg_dt, n_per_segment)
        seg = np.empty((n_per_segment, 3))
        seg[:, 0] = p[0] + v[0] * ts
        seg[:, 1] = p[1] + v[1] * ts
        seg[:, 2] = p[2] + v[2] * ts - 0.5 * g * ts * ts
        points.extend(list(seg[1:]))
        if t_to_target <= t_to_ground:
            return np.asarray(points)
        v_z_before = v[2] - g * t_to_ground
        p = np.array([seg[-1, 0], seg[-1, 1], 0.0])
        v = np.array([
            v[0] * cfg.tracker.bounce_tangential_damping,
            v[1] * cfg.tracker.bounce_tangential_damping,
            -cfg.tracker.bounce_restitution * v_z_before,
        ])
    return np.asarray(points)


class _ReplayTracker(BallTracker):
    """BallTracker driven by explicit (t, pos) samples instead of Redis + wall clock.

    Mirrors the filtering logic in BallTracker.update() so the prediction
    code path under test (_fit_state, predict_intercept) is byte-for-byte the
    production one — only the ingest layer differs.
    """

    def __init__(self, cfg) -> None:  # noqa: D401
        # Skip the parent __init__ to avoid requiring a Redis client.
        self._redis = None
        self._keys = None
        self._cfg = cfg
        self._history = collections.deque(maxlen=cfg.history_size)
        self._t0 = 0.0
        self._last_seen_t = None

    def ingest(self, t: float, pos: np.ndarray) -> Optional[BallSample]:
        if pos is None or pos.shape != (3,):
            return None
        # Drop stale samples first — same order as update().
        while self._history and (t - self._history[0].t) > self._cfg.history_max_age_s:
            self._history.popleft()
        if self._history:
            last = self._history[-1]
            if np.linalg.norm(pos - last.pos) > self._cfg.max_position_jump:
                return None
        sample = BallSample(t=float(t), pos=pos.astype(float))
        self._history.append(sample)
        self._last_seen_t = float(t)
        # Mirror BallTracker.update()'s bounce-pruning step so the replay
        # exercises the same code path the FSM will see.
        if self._cfg.online_bounce_pruning:
            self._try_prune_pre_bounce()
        return sample


class _ReplayEKFTracker(EKFBallTracker):
    """EKFBallTracker driven by explicit (t, pos) samples. Replay analog of
    ``_ReplayTracker`` for the experimental probabilistic tracker.

    The EKF's wall-clock ``update()`` is unused here — we feed the recorder's
    timestamps directly into ``_ingest`` so a replay of a recording sees the
    same dt sequence the recorder saw.
    """

    def __init__(self, cfg) -> None:
        super().__init__(redis_client=None, keys=None, cfg=cfg)

    def ingest(self, t: float, pos: np.ndarray) -> Optional[BallSample]:
        if pos is None or pos.shape != (3,):
            return None
        return self._ingest(float(t), pos)


def _replay(
    timestamps: np.ndarray,
    positions: np.ndarray,
    cfg: PickleballConfig,
    tracker_kind: str = "leastsq",
) -> List[TickAnalysis]:
    if tracker_kind == "ekf":
        tracker = _ReplayEKFTracker(cfg.tracker)
    elif tracker_kind == "leastsq":
        tracker = _ReplayTracker(cfg.tracker)
    else:
        raise ValueError(f"unknown tracker kind: {tracker_kind!r}")
    strike_plane_x = cfg.court.strike_plane_x

    out: List[TickAnalysis] = []
    last_t: Optional[float] = None
    # Gap above which we consider the tracker to have "lost the ball" — same
    # threshold the FSM uses to bail to RECOVER. Reset the tracker explicitly
    # so the EKF doesn't carry a 3 s dt step into the next throw (the LS
    # tracker is self-resetting via history_max_age_s, so this only changes
    # EKF behavior in practice — but applying it uniformly keeps the A/B fair).
    reset_gap_s = max(cfg.tracker.history_max_age_s, 0.3)
    for t, pos in zip(timestamps, positions):
        if last_t is not None and (t - last_t) > reset_gap_s:
            tracker.reset()
        last_t = float(t)
        accepted = tracker.ingest(float(t), pos) is not None
        rec = TickAnalysis(
            t=float(t),
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
            if intercept is not None:
                rec.intercept_pos = intercept.position
                rec.intercept_vel = intercept.velocity
                rec.time_to_impact = float(intercept.time_to_impact)
                rec.n_bounces_in_prediction = int(intercept.n_bounces)
                if intercept.position_cov is not None:
                    rec.intercept_pos_cov = intercept.position_cov.copy()

        out.append(rec)
    return out


def _find_crossings(
    timestamps: np.ndarray,
    positions: np.ndarray,
    strike_plane_x: float,
) -> List[Tuple[float, np.ndarray]]:
    """Where the recorded ball actually crossed x=strike_plane_x (linear interp)."""
    out: List[Tuple[float, np.ndarray]] = []
    for i in range(len(timestamps) - 1):
        x0 = positions[i, 0]
        x1 = positions[i + 1, 0]
        if x0 == x1:
            continue
        if (x0 - strike_plane_x) * (x1 - strike_plane_x) <= 0.0:
            alpha = (strike_plane_x - x0) / (x1 - x0)
            if 0.0 <= alpha <= 1.0:
                t_cross = timestamps[i] + alpha * (timestamps[i + 1] - timestamps[i])
                p_cross = positions[i] + alpha * (positions[i + 1] - positions[i])
                out.append((float(t_cross), p_cross))
    return out


@dataclass
class Throw:
    idx: int
    t_start: float
    t_end: float
    timestamps: np.ndarray
    positions: np.ndarray
    ticks: List[TickAnalysis]
    actual_crossing: Optional[Tuple[float, np.ndarray]] = None  # (t, pos)
    bounces: List[Bounce] = field(default_factory=list)


def _segment_throws(
    timestamps: np.ndarray,
    positions: np.ndarray,
    ticks: List[TickAnalysis],
    strike_plane_x: float,
    gap_s: float,
    min_samples: int,
    detect_bounces: bool = True,
    bounce_gravity: float = 9.81,
) -> List[Throw]:
    """Split into per-throw segments by time gaps in the recorded stream.

    With the recorder's min-movement filter, a stationary ball produces no
    samples — so two throws separated by a "ball at rest" period show up as
    a single time gap larger than gap_s.
    """
    if len(timestamps) == 0:
        return []
    segments: List[Tuple[int, int]] = []
    cur_start = 0
    for i in range(1, len(timestamps)):
        if timestamps[i] - timestamps[i - 1] > gap_s:
            segments.append((cur_start, i))
            cur_start = i
    segments.append((cur_start, len(timestamps)))

    throws: List[Throw] = []
    for a, b in segments:
        if b - a < min_samples:
            continue
        ts = timestamps[a:b]
        ps = positions[a:b]
        crossings = _find_crossings(ts, ps, strike_plane_x)
        bounces = (
            _detect_bounces(ts, ps, gravity=bounce_gravity) if detect_bounces else []
        )
        throws.append(Throw(
            idx=len(throws),
            t_start=float(ts[0]),
            t_end=float(ts[-1]),
            timestamps=ts,
            positions=ps,
            ticks=ticks[a:b],
            actual_crossing=crossings[0] if crossings else None,
            bounces=bounces,
        ))
    return throws


def _print_throw_summary(throw: Throw, cfg: PickleballConfig) -> None:
    plane = cfg.court.strike_plane_x
    n_pred = sum(1 for t in throw.ticks if t.intercept_pos is not None)
    n_pred_with_bounce = sum(1 for t in throw.ticks if t.n_bounces_in_prediction > 0)
    duration = throw.t_end - throw.t_start
    print(f"\n--- throw {throw.idx} | {len(throw.timestamps)} samples, "
          f"{duration:.2f}s | predictions: {n_pred} "
          f"(of which {n_pred_with_bounce} through ≥1 bounce) ---")

    if throw.bounces:
        print(f"  detected {len(throw.bounces)} bounce(s):")
        for j, b in enumerate(throw.bounces):
            print(f"    bounce {j}: t={b.t:.3f}s  "
                  f"pos=[{b.position[0]:+.3f} {b.position[1]:+.3f} {b.position[2]:+.3f}]  "
                  f"v_before=[{b.v_before[0]:+.2f} {b.v_before[1]:+.2f} {b.v_before[2]:+.2f}]  "
                  f"v_after=[{b.v_after[0]:+.2f} {b.v_after[1]:+.2f} {b.v_after[2]:+.2f}]  "
                  f"e={b.restitution:.3f}  μ_t={b.mu_t:.3f}")

    if n_pred == 0:
        print("  no intercept predictions produced (tracker rejected fit / not incoming / lookahead window)")
        return

    if throw.actual_crossing is None:
        print(f"  ball never crossed x={plane:.3f} -- nothing to compare predictions to")
        return

    _, p_cross = throw.actual_crossing
    print(f"  actual crossing : [{p_cross[0]:+.3f} {p_cross[1]:+.3f} {p_cross[2]:+.3f}] m")

    # Bucket predictions by time-to-impact remaining at the moment of prediction.
    # (upper, lower) seconds; predictions with tti in [lower, upper).
    buckets = [
        (1.50, 1.00),
        (1.00, 0.70),
        (0.70, 0.50),
        (0.50, 0.30),
        (0.30, 0.15),
        (0.15, 0.00),
    ]
    print(f"  {'tti window (s)':>15}  {'n':>3}  {'mean err (m)':>13}  {'max err (m)':>12}  "
          f"{'mean dz (m)':>12}  {'mean dy (m)':>12}")
    for upper, lower in buckets:
        bucket = [
            t for t in throw.ticks
            if t.intercept_pos is not None and lower <= t.time_to_impact < upper
        ]
        if not bucket:
            print(f"  [{lower:.2f}, {upper:.2f})    {'-':>3}  {'-':>13}  {'-':>12}  "
                  f"{'-':>12}  {'-':>12}")
            continue
        errs = np.array([np.linalg.norm(t.intercept_pos - p_cross) for t in bucket])
        dys = np.array([t.intercept_pos[1] - p_cross[1] for t in bucket])
        dzs = np.array([t.intercept_pos[2] - p_cross[2] for t in bucket])
        print(f"  [{lower:.2f}, {upper:.2f})    {len(bucket):>3d}  {errs.mean():>13.3f}  "
              f"{errs.max():>12.3f}  {dzs.mean():>+12.3f}  {dys.mean():>+12.3f}")


def _print_bounce_aggregate(throws: List[Throw]) -> None:
    """Roll up (e, μ_t) statistics across every detected bounce in every throw,
    and suggest config values."""
    all_bounces: List[Bounce] = [b for throw in throws for b in throw.bounces]
    if not all_bounces:
        print("\n=== aggregate bounces ===")
        print("  no bounces detected across the recording")
        return

    es = np.array([b.restitution for b in all_bounces])
    mus = np.array([b.mu_t for b in all_bounces])
    print(f"\n=== aggregate bounces ({len(all_bounces)} across {sum(1 for t in throws if t.bounces)} throw(s)) ===")
    print(f"  restitution  e  : mean={es.mean():.3f}  median={np.median(es):.3f}  "
          f"std={es.std():.3f}  [{es.min():.3f}, {es.max():.3f}]")
    print(f"  tangential μ_t  : mean={mus.mean():.3f}  median={np.median(mus):.3f}  "
          f"std={mus.std():.3f}  [{mus.min():.3f}, {mus.max():.3f}]")
    print(f"  suggested config (median):")
    print(f"    bounce_restitution        = {np.median(es):.3f}")
    print(f"    bounce_tangential_damping = {np.median(mus):.3f}")


def cmd_analyze(args: argparse.Namespace) -> int:
    rec_path = Path(args.recording)
    if not rec_path.exists():
        print(f"[analyze] not found: {rec_path}", file=sys.stderr)
        return 1
    data = np.load(rec_path, allow_pickle=False)
    timestamps = np.asarray(data["timestamps"])
    positions = np.asarray(data["positions"])
    ball_source = str(data["ball_source"]) if "ball_source" in data.files else "unknown"

    cfg = PickleballConfig()
    if args.strike_plane_x is not None:
        cfg.court.strike_plane_x = args.strike_plane_x
    if args.history_size is not None:
        cfg.tracker.history_size = args.history_size
    if args.history_max_age_s is not None:
        cfg.tracker.history_max_age_s = args.history_max_age_s
    if args.max_position_jump is not None:
        cfg.tracker.max_position_jump = args.max_position_jump
    if args.min_incoming_speed is not None:
        cfg.tracker.min_incoming_speed = args.min_incoming_speed
    if args.gravity is not None:
        cfg.tracker.gravity = args.gravity
    if args.bounce_restitution is not None:
        cfg.tracker.bounce_restitution = args.bounce_restitution
    if args.bounce_tangential_damping is not None:
        cfg.tracker.bounce_tangential_damping = args.bounce_tangential_damping
    if args.max_bounces is not None:
        cfg.tracker.max_bounces = args.max_bounces
    if args.no_online_bounce_pruning:
        cfg.tracker.online_bounce_pruning = False

    # EKF-specific overrides (no-ops for the LS tracker).
    if args.ekf_process_accel_std_xy is not None:
        cfg.tracker.ekf.process_accel_std_xy = args.ekf_process_accel_std_xy
    if args.ekf_process_accel_std_z is not None:
        cfg.tracker.ekf.process_accel_std_z = args.ekf_process_accel_std_z
    if args.ekf_measurement_pos_std is not None:
        cfg.tracker.ekf.measurement_pos_std = args.ekf_measurement_pos_std
    if args.ekf_seed_samples is not None:
        cfg.tracker.ekf.seed_samples = args.ekf_seed_samples
    if args.no_ekf_bounce_handling:
        cfg.tracker.ekf.online_bounce_handling = False

    print(f"[analyze] {rec_path} ({ball_source})")
    print(f"[analyze] {len(timestamps)} samples, duration {timestamps[-1] - timestamps[0]:.2f}s")
    print(f"[analyze] tracker = {args.tracker}")
    print(f"[analyze] tracker: history_size={cfg.tracker.history_size}, "
          f"max_age={cfg.tracker.history_max_age_s:.2f}s, "
          f"max_jump={cfg.tracker.max_position_jump:.3f}m, "
          f"min_incoming={cfg.tracker.min_incoming_speed:.2f}m/s, "
          f"lookahead=[{cfg.tracker.min_lookahead:.2f}, {cfg.tracker.max_lookahead:.2f}]s, "
          f"g={cfg.tracker.gravity:.2f}m/s^2")
    print(f"[analyze] bounce model: e={cfg.tracker.bounce_restitution:.3f}, "
          f"μ_t={cfg.tracker.bounce_tangential_damping:.3f}, "
          f"max_bounces={cfg.tracker.max_bounces}, "
          f"online_pruning={cfg.tracker.online_bounce_pruning}")
    if args.tracker == "ekf":
        ekf = cfg.tracker.ekf
        print(f"[analyze] ekf: σ_a_xy={ekf.process_accel_std_xy:.2f} m/s², "
              f"σ_a_z={ekf.process_accel_std_z:.2f} m/s², "
              f"σ_meas={ekf.measurement_pos_std*1000:.1f} mm, "
              f"seed_samples={ekf.seed_samples}, "
              f"online_bounce_handling={ekf.online_bounce_handling}")
    print(f"[analyze] strike_plane_x = {cfg.court.strike_plane_x:.3f}")

    ticks = _replay(timestamps, positions, cfg, tracker_kind=args.tracker)
    throws = _segment_throws(
        timestamps, positions, ticks,
        cfg.court.strike_plane_x,
        gap_s=args.segment_gap_s,
        min_samples=args.segment_min_samples,
        detect_bounces=not args.no_bounce_detection,
        bounce_gravity=cfg.tracker.gravity,
    )
    print(f"[analyze] segmented into {len(throws)} throw(s)")

    for throw in throws:
        _print_throw_summary(throw, cfg)

    _print_bounce_aggregate(throws)

    if not throws:
        return 0
    if args.no_viser:
        return 0

    try:
        import viser  # type: ignore
    except ImportError:
        print("[analyze] viser not installed; skipping visualization.\n"
              "          install with: pip install viser", file=sys.stderr)
        return 0

    _run_viser(throws, cfg, args.viser_port)
    return 0


# -----------------------------------------------------------------------------
# Viser visualization
# -----------------------------------------------------------------------------

def _ttis_to_colors(ttis: np.ndarray, t_max: float) -> np.ndarray:
    """Red (small tti, close to impact) -> blue (large tti, far from impact)."""
    t_max = max(float(t_max), 0.01)
    norm = np.clip(ttis / t_max, 0.0, 1.0)
    colors = np.empty((len(ttis), 3), dtype=np.uint8)
    colors[:, 0] = (255 * (1.0 - norm)).astype(np.uint8)
    colors[:, 1] = (80 * (1.0 - np.abs(2 * norm - 1.0))).astype(np.uint8)
    colors[:, 2] = (255 * norm).astype(np.uint8)
    return colors


def _line_segments_from_polyline(points: np.ndarray) -> np.ndarray:
    """(N, 3) polyline -> (N-1, 2, 3) line-segments array."""
    return np.stack([points[:-1], points[1:]], axis=1)


def _run_viser(throws: List[Throw], cfg: PickleballConfig, port: int) -> None:
    import viser  # local import; we already verified availability

    server = viser.ViserServer(host="0.0.0.0", port=port)
    print(f"[analyze] viser running at http://localhost:{port}")
    print("[analyze] (Ctrl-C to exit)")

    # Static scene -----------------------------------------------------------
    server.scene.add_frame("/world", show_axes=True, axes_length=0.3, axes_radius=0.01)
    server.scene.add_grid(
        "/floor",
        width=10.0, height=10.0,
        plane="xy",
        cell_size=0.5, section_size=1.0,
        position=(2.0, 0.0, 0.0),
    )
    strike_plane_x = cfg.court.strike_plane_x
    server.scene.add_box(
        "/strike_plane",
        position=(strike_plane_x, 0.0, 1.0),
        dimensions=(0.005, 4.0, 2.0),
        color=(255, 215, 0),
        opacity=0.25,
        side="double",
    )
    # Z range hints (strike-reachable band on the strike plane).
    for z_label, z_value, color in [
        ("z_min", cfg.court.strike_z_min, (90, 90, 90)),
        ("z_max", cfg.court.strike_z_max, (90, 90, 90)),
    ]:
        server.scene.add_box(
            f"/strike_plane/{z_label}",
            position=(strike_plane_x, 0.0, z_value),
            dimensions=(0.008, 4.0, 0.005),
            color=color,
        )

    # Mutable per-throw scene -------------------------------------------------
    handles: dict = {}

    def _clear() -> None:
        for h in list(handles.values()):
            try:
                h.remove()
            except Exception:  # noqa: BLE001
                pass
        handles.clear()

    state = {"throw_idx": 0, "tick_idx": len(throws[0].ticks) - 1}

    # GUI --------------------------------------------------------------------
    with server.gui.add_folder("Recording"):
        throw_dd = server.gui.add_dropdown(
            "Throw",
            options=[
                f"{t.idx}: N={len(t.timestamps)} dur={t.t_end - t.t_start:.2f}s"
                for t in throws
            ],
            initial_value=f"0: N={len(throws[0].timestamps)} dur={throws[0].t_end - throws[0].t_start:.2f}s",
        )
        tick_slider = server.gui.add_slider(
            "Tick", min=0, max=max(0, len(throws[0].ticks) - 1),
            step=1, initial_value=max(0, len(throws[0].ticks) - 1),
        )

    with server.gui.add_folder("Display"):
        show_all_preds = server.gui.add_checkbox("All predictions (point cloud)", True)
        show_fit_curve = server.gui.add_checkbox("Current fit ballistic curve", True)
        show_fit_window = server.gui.add_checkbox("Current fit-window samples", True)
        show_bounces = server.gui.add_checkbox("Detected bounces (orange)", True)

    with server.gui.add_folder("Tick info"):
        info_t = server.gui.add_text("t (s)", "—", disabled=True)
        info_pos = server.gui.add_text("ball pos", "—", disabled=True)
        info_v0 = server.gui.add_text("fit v0", "—", disabled=True)
        info_pred = server.gui.add_text("pred. intercept", "—", disabled=True)
        info_tti = server.gui.add_text("t to impact", "—", disabled=True)

    with server.gui.add_folder("Ground truth"):
        info_actual = server.gui.add_text("actual crossing", "—", disabled=True)
        info_err = server.gui.add_text("|pred - actual|", "—", disabled=True)
        info_err_xyz = server.gui.add_text("(dx, dy, dz)", "—", disabled=True)

    # Render -----------------------------------------------------------------
    def _redraw() -> None:
        _clear()
        throw = throws[state["throw_idx"]]
        i = int(state["tick_idx"])
        i = max(0, min(i, len(throw.ticks) - 1))
        tick = throw.ticks[i]

        # Full recorded trajectory as line segments (faithful to data).
        if len(throw.positions) >= 2:
            segs = _line_segments_from_polyline(throw.positions)
            handles["traj"] = server.scene.add_line_segments(
                "/throw/trajectory",
                points=segs.astype(np.float32),
                colors=(170, 170, 170),
                line_width=2.0,
            )

        # All recorded sample positions as faint dots (so user can see sampling density).
        handles["samples"] = server.scene.add_point_cloud(
            "/throw/samples",
            points=throw.positions.astype(np.float32),
            colors=np.tile(np.array([120, 120, 120], dtype=np.uint8),
                           (len(throw.positions), 1)),
            point_size=0.008,
            point_shape="circle",
        )

        # Current ball position.
        handles["ball"] = server.scene.add_icosphere(
            "/throw/ball",
            radius=0.035,
            color=(255, 80, 80),
            position=tuple(tick.ball_pos.tolist()),
        )

        # Fit-window samples (the ones that would currently feed _fit_state).
        if show_fit_window.value:
            window_pts: List[np.ndarray] = []
            t_now = tick.t
            max_age = cfg.tracker.history_max_age_s
            max_n = cfg.tracker.history_size
            for j in range(i, -1, -1):
                if (t_now - throw.timestamps[j]) > max_age:
                    break
                window_pts.append(throw.positions[j])
                if len(window_pts) >= max_n:
                    break
            if window_pts:
                pts = np.asarray(window_pts, dtype=np.float32)
                handles["fit_window"] = server.scene.add_point_cloud(
                    "/throw/fit_window",
                    points=pts,
                    colors=np.tile(np.array([0, 200, 255], dtype=np.uint8),
                                   (len(pts), 1)),
                    point_size=0.018,
                    point_shape="circle",
                )

        # Current fit's predicted trajectory (with bouncing). Bends at predicted
        # floor bounces using cfg.tracker.bounce_restitution / bounce_tangential_damping.
        if (
            show_fit_curve.value
            and tick.fit_p0 is not None
            and tick.fit_v0 is not None
            and tick.time_to_impact is not None
        ):
            curve = _bouncing_path(
                tick.fit_p0, tick.fit_v0,
                cfg.court.strike_plane_x,
                cfg,
                n_per_segment=25,
            )
            handles["fit_curve"] = server.scene.add_spline_catmull_rom(
                "/throw/fit_curve",
                points=curve.astype(np.float32),
                color=(0, 220, 60),
                line_width=2.5,
            )

        # Current predicted intercept (highlight).
        if tick.intercept_pos is not None:
            handles["pred_now"] = server.scene.add_icosphere(
                "/throw/predicted_now",
                radius=0.05,
                color=(0, 255, 0),
                position=tuple(tick.intercept_pos.tolist()),
            )
            # If the tracker provided a position covariance, render a
            # translucent 1-σ ellipsoid around the predicted intercept.
            # Useful as a fast eyeball check of "how tight is the EKF right
            # now?"
            if tick.intercept_pos_cov is not None:
                try:
                    eigvals, eigvecs = np.linalg.eigh(tick.intercept_pos_cov)
                    sigmas = np.sqrt(np.clip(eigvals, 1e-9, None))
                    # Box dimensions = 2σ (full extent ≈ 1σ each side).
                    dims = tuple(float(2.0 * s) for s in sigmas)
                    # Encode rotation by adding a frame at the intercept and
                    # nesting the box inside it. quat = wxyz from R = eigvecs.
                    R = eigvecs
                    # Ensure right-handed (otherwise wxyz will flip).
                    if np.linalg.det(R) < 0:
                        R[:, -1] *= -1
                    # R -> quaternion (wxyz). Standard formula.
                    tr = R[0, 0] + R[1, 1] + R[2, 2]
                    if tr > 0:
                        S = 2.0 * np.sqrt(tr + 1.0)
                        qw = 0.25 * S
                        qx = (R[2, 1] - R[1, 2]) / S
                        qy = (R[0, 2] - R[2, 0]) / S
                        qz = (R[1, 0] - R[0, 1]) / S
                    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
                        S = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
                        qw = (R[2, 1] - R[1, 2]) / S
                        qx = 0.25 * S
                        qy = (R[0, 1] + R[1, 0]) / S
                        qz = (R[0, 2] + R[2, 0]) / S
                    elif R[1, 1] > R[2, 2]:
                        S = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
                        qw = (R[0, 2] - R[2, 0]) / S
                        qx = (R[0, 1] + R[1, 0]) / S
                        qy = 0.25 * S
                        qz = (R[1, 2] + R[2, 1]) / S
                    else:
                        S = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
                        qw = (R[1, 0] - R[0, 1]) / S
                        qx = (R[0, 2] + R[2, 0]) / S
                        qy = (R[1, 2] + R[2, 1]) / S
                        qz = 0.25 * S
                    handles["pred_cov"] = server.scene.add_box(
                        "/throw/predicted_now_cov",
                        position=tuple(tick.intercept_pos.tolist()),
                        wxyz=(float(qw), float(qx), float(qy), float(qz)),
                        dimensions=dims,
                        color=(0, 255, 0),
                        opacity=0.18,
                    )
                except np.linalg.LinAlgError:
                    pass

        # All predictions across the throw, color-coded by t-to-impact.
        if show_all_preds.value:
            pred_pts = np.array(
                [t.intercept_pos for t in throw.ticks if t.intercept_pos is not None],
                dtype=np.float32,
            )
            pred_tti = np.array(
                [t.time_to_impact for t in throw.ticks if t.intercept_pos is not None],
                dtype=np.float32,
            )
            if len(pred_pts) > 0:
                colors = _ttis_to_colors(pred_tti, t_max=cfg.tracker.max_lookahead)
                handles["all_preds"] = server.scene.add_point_cloud(
                    "/throw/all_predictions",
                    points=pred_pts,
                    colors=colors,
                    point_size=0.022,
                    point_shape="circle",
                )

        # Actual strike-plane crossing.
        if throw.actual_crossing is not None:
            _, p_cross = throw.actual_crossing
            handles["actual"] = server.scene.add_icosphere(
                "/throw/actual_crossing",
                radius=0.05,
                color=(255, 230, 0),
                position=tuple(p_cross.tolist()),
            )

        # Detected bounces in the recorded trajectory (orange spheres).
        if show_bounces.value:
            for bi, b in enumerate(throw.bounces):
                handles[f"bounce_{bi}"] = server.scene.add_icosphere(
                    f"/throw/bounce_{bi}",
                    radius=0.04,
                    color=(255, 140, 0),
                    position=tuple(b.position.tolist()),
                )
                # Label with measured e and μ_t, perched above the bounce point.
                handles[f"bounce_label_{bi}"] = server.scene.add_label(
                    f"/throw/bounce_{bi}/label",
                    text=f"e={b.restitution:.2f}  μ_t={b.mu_t:.2f}",
                    position=(float(b.position[0]), float(b.position[1]), float(b.position[2]) + 0.12),
                )

        # GUI text panels --------------------------------------------------
        info_t.value = f"{tick.t:.3f}"
        info_pos.value = "[{:+.3f}, {:+.3f}, {:+.3f}]".format(*tick.ball_pos.tolist())
        if tick.fit_v0 is not None:
            info_v0.value = "[{:+.2f}, {:+.2f}, {:+.2f}] m/s".format(*tick.fit_v0.tolist())
        else:
            info_v0.value = "—"
        if tick.intercept_pos is not None and tick.time_to_impact is not None:
            bounce_suffix = (
                f"  (via {tick.n_bounces_in_prediction} bounce)"
                if tick.n_bounces_in_prediction > 0 else ""
            )
            info_pred.value = (
                "[{:+.3f}, {:+.3f}, {:+.3f}]".format(*tick.intercept_pos.tolist())
                + bounce_suffix
            )
            info_tti.value = f"{tick.time_to_impact:.3f} s"
        else:
            info_pred.value = "—"
            info_tti.value = "—"

        if throw.actual_crossing is not None:
            t_cross, p_cross = throw.actual_crossing
            info_actual.value = "[{:+.3f}, {:+.3f}, {:+.3f}] @ t={:.3f}s".format(
                p_cross[0], p_cross[1], p_cross[2], t_cross
            )
            if tick.intercept_pos is not None:
                d = tick.intercept_pos - p_cross
                info_err.value = f"{np.linalg.norm(d):.3f} m"
                info_err_xyz.value = "({:+.3f}, {:+.3f}, {:+.3f}) m".format(*d.tolist())
            else:
                info_err.value = "—"
                info_err_xyz.value = "—"
        else:
            info_actual.value = "(ball never crossed strike plane)"
            info_err.value = "—"
            info_err_xyz.value = "—"

    # Wire up callbacks ------------------------------------------------------
    @throw_dd.on_update
    def _on_throw(_):  # noqa: ANN001
        state["throw_idx"] = int(throw_dd.value.split(":")[0])
        throw = throws[state["throw_idx"]]
        new_max = max(0, len(throw.ticks) - 1)
        tick_slider.max = new_max
        tick_slider.value = new_max
        state["tick_idx"] = new_max
        _redraw()

    @tick_slider.on_update
    def _on_tick(_):  # noqa: ANN001
        state["tick_idx"] = int(tick_slider.value)
        _redraw()

    @show_all_preds.on_update
    def _(_):
        _redraw()

    @show_fit_curve.on_update
    def _(_):
        _redraw()

    @show_fit_window.on_update
    def _(_):
        _redraw()

    @show_bounces.on_update
    def _(_):
        _redraw()

    _redraw()

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[analyze] bye")


# =============================================================================
# CLI
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        prog="ball_tracker_test",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # ----- record
    rec = sub.add_parser("record", help="Capture ball trajectory from Redis to NPZ.")
    rec.add_argument("--ball-source", choices=["opensai", "optitrack"], default="optitrack")
    rec.add_argument("--optitrack-rigid-body-id", type=int, default=1)
    rec.add_argument("--redis-host", default="localhost")
    rec.add_argument("--redis-port", type=int, default=6379)
    rec.add_argument("--dt", type=float, default=0.005,
                     help="Polling period (s). Default 0.005 = 200 Hz, faster than OptiTrack's 120 Hz so we don't miss frames.")
    rec.add_argument("--min-movement", type=float, default=0.001,
                     help="Minimum distance (m) from the last recorded sample for a new sample to be saved. "
                          "Filters out OptiTrack chatter when the ball is stationary.")
    rec.add_argument(
        "--output", "-o", default=None,
        help="Output .npz path. Default: <repo>/sports_bot/recordings/throws_<YYYYMMDD_HHMMSS>.npz",
    )

    # ----- analyze
    an = sub.add_parser("analyze", help="Replay a recording through BallTracker and visualize.")
    an.add_argument("recording", help="Path to a .npz produced by `record`.")
    an.add_argument("--no-viser", action="store_true",
                    help="Print summary only; skip viser visualization.")
    an.add_argument("--viser-port", type=int, default=8080)

    an.add_argument("--segment-gap-s", type=float, default=0.4,
                    help="Time gap that separates one throw from the next.")
    an.add_argument("--segment-min-samples", type=int, default=5,
                    help="Drop segments shorter than this many samples.")

    an.add_argument("--strike-plane-x", type=float, default=None,
                    help="Override CourtConfig.strike_plane_x.")
    an.add_argument("--history-size", type=int, default=None,
                    help="Override BallTrackerConfig.history_size.")
    an.add_argument("--history-max-age-s", type=float, default=None,
                    help="Override BallTrackerConfig.history_max_age_s.")
    an.add_argument("--max-position-jump", type=float, default=None,
                    help="Override BallTrackerConfig.max_position_jump.")
    an.add_argument("--min-incoming-speed", type=float, default=None,
                    help="Override BallTrackerConfig.min_incoming_speed.")
    an.add_argument("--gravity", type=float, default=None,
                    help="Override BallTrackerConfig.gravity.")
    an.add_argument("--bounce-restitution", type=float, default=None,
                    help="Override BallTrackerConfig.bounce_restitution (e = -v_z_after / v_z_before).")
    an.add_argument("--bounce-tangential-damping", type=float, default=None,
                    help="Override BallTrackerConfig.bounce_tangential_damping (μ_t).")
    an.add_argument("--max-bounces", type=int, default=None,
                    help="Override BallTrackerConfig.max_bounces (0 = volley-only).")
    an.add_argument("--no-bounce-detection", action="store_true",
                    help="Skip the analyzer's bounce detection (does not affect the predictor's bounce model — that's "
                         "controlled by --max-bounces).")
    an.add_argument("--no-online-bounce-pruning", action="store_true",
                    help="Disable BallTracker's online bounce-triggered history pruning (3a). For A/B with the "
                         "pre-3a behavior on the same recording.")

    # ----- tracker selection + EKF overrides
    an.add_argument("--tracker", choices=["leastsq", "ekf"], default="leastsq",
                    help="Which tracker implementation to replay against the recording. "
                         "'leastsq' is the production sliding-window LS fitter; 'ekf' is the experimental "
                         "Kalman-filter-with-bounce-jumps variant.")
    an.add_argument("--ekf-process-accel-std-xy", type=float, default=None,
                    help="EKF only — unmodeled acceleration std (m/s²) in xy (Magnus + drag).")
    an.add_argument("--ekf-process-accel-std-z", type=float, default=None,
                    help="EKF only — unmodeled acceleration std (m/s²) in z (gravity already in dynamics).")
    an.add_argument("--ekf-measurement-pos-std", type=float, default=None,
                    help="EKF only — OptiTrack position measurement std (m).")
    an.add_argument("--ekf-seed-samples", type=int, default=None,
                    help="EKF only — number of samples to LS-seed the filter from before recursive updates.")
    an.add_argument("--no-ekf-bounce-handling", action="store_true",
                    help="EKF only — disable online bounce-triggered re-seeding (filter just keeps integrating).")

    args = parser.parse_args()

    if args.cmd == "record":
        return cmd_record(args)
    if args.cmd == "analyze":
        return cmd_analyze(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
