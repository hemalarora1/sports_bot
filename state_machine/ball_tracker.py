"""Ball position acquisition + ballistic intercept prediction.

The tracker keeps a small sliding window of (timestamp, position) samples,
fits a ballistic model

    x(t) = x0 + vx * t
    y(t) = y0 + vy * t
    z(t) = z0 + vz * t - 0.5 * g * t^2

and solves for the time-of-flight to a fixed strike plane (x = strike_plane_x).
"""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple

import numpy as np
import redis

from .config import BallTrackerConfig
from .redis_keys import RedisKeys


@dataclass
class BallSample:
    t: float                # seconds since tracker start
    pos: np.ndarray         # world-frame position (m)


@dataclass
class Intercept:
    """Predicted intercept point on the strike plane."""

    position: np.ndarray    # 3-vector, world frame, on x = strike_plane_x
    velocity: np.ndarray    # 3-vector ball velocity at intercept (world frame)
    time_to_impact: float   # seconds from "now" to predicted impact
    n_bounces: int = 0      # bounces simulated before reaching the strike plane
    # Optional 3x3 position covariance at the intercept. Populated by
    # probabilistic estimators (EKF); None for the production least-squares
    # tracker, which has no notion of uncertainty.
    position_cov: Optional[np.ndarray] = None


def _propagate_to_plane(
    p0: np.ndarray,
    v0: np.ndarray,
    target_x: float,
    cfg: "BallTrackerConfig",
) -> Optional[Tuple[float, np.ndarray, np.ndarray, int]]:
    """Propagate a ballistic point with optional bouncing to where it crosses
    x = target_x. Returns (t_total, p_at_target, v_at_target, n_bounces) or
    None if no valid intercept exists within the bounce budget.

    Physics: between bounces, free-fall in z + constant velocity in xy. On
    contact with z=0, v_z reflects with -cfg.bounce_restitution and v_xy
    scales by cfg.bounce_tangential_damping. Up to cfg.max_bounces bounces.
    """
    p = p0.astype(float).copy()
    v = v0.astype(float).copy()
    g = cfg.gravity
    t_total = 0.0

    # +1 so we get one extra "no bounce remaining" iteration that can still
    # reach the plane without bouncing again.
    for n_bounces in range(cfg.max_bounces + 1):
        if v[0] >= 0:
            return None  # not moving toward plane
        t_to_target = (target_x - p[0]) / v[0]
        if t_to_target < 0:
            return None  # already past
        # At-or-below-floor and not rising → ball is stuck on the ground, the
        # ballistic model can't predict anything useful. After our own
        # simulated bounce p[2]==0 with v[2]>0, which must be allowed.
        if p[2] <= cfg.floor_epsilon and v[2] <= 0:
            return None

        # Solve 0.5*g*t² - v_z*t - p_z = 0 for the positive root.
        discriminant = v[2] * v[2] + 2.0 * g * p[2]
        if discriminant < 0:
            t_to_ground = float("inf")
        else:
            t_to_ground = (v[2] + np.sqrt(discriminant)) / g

        if t_to_target <= t_to_ground:
            # Reach plane before the next bounce.
            t_total += t_to_target
            p_final = np.array([
                target_x,
                p[1] + v[1] * t_to_target,
                p[2] + v[2] * t_to_target - 0.5 * g * t_to_target * t_to_target,
            ])
            v_final = np.array([v[0], v[1], v[2] - g * t_to_target])
            return t_total, p_final, v_final, n_bounces

        # Ground first; bounce if budget remains.
        if n_bounces >= cfg.max_bounces:
            return None  # would need another bounce we're not allowed
        t_total += t_to_ground
        x_at_ground = p[0] + v[0] * t_to_ground
        y_at_ground = p[1] + v[1] * t_to_ground
        v_z_before = v[2] - g * t_to_ground
        p = np.array([x_at_ground, y_at_ground, 0.0])
        v = np.array([
            v[0] * cfg.bounce_tangential_damping,
            v[1] * cfg.bounce_tangential_damping,
            -cfg.bounce_restitution * v_z_before,
        ])

    return None


class BallTracker:
    """Reads the ball position from Redis and predicts the intercept point.

    Designed to be called once per FSM tick.
    """

    def __init__(
        self,
        redis_client: redis.Redis,
        keys: RedisKeys,
        cfg: BallTrackerConfig,
    ):
        self._redis = redis_client
        self._keys = keys
        self._cfg = cfg
        self._history: Deque[BallSample] = deque(maxlen=cfg.history_size)
        self._t0 = time.perf_counter()
        self._last_seen_t: Optional[float] = None

    # ------------------------------------------------------------------ I/O

    def _read_position(self) -> Optional[np.ndarray]:
        """Pull the latest ball position from Redis. Returns None if unavailable."""
        if self._keys.ball_source == "opensai":
            raw = self._redis.get(self._keys.ball.opensai_object_pose)
            if raw is None:
                return None
            try:
                pose = np.array(json.loads(raw))
                # 4x4 homogeneous transform; translation is the top-right column.
                if pose.shape == (4, 4):
                    return pose[0:3, 3]
                # Fall back to raw 3-vector.
                if pose.shape == (3,):
                    return pose
            except (ValueError, json.JSONDecodeError):
                return None
            return None

        if self._keys.ball_source == "optitrack":
            raw = self._redis.get(self._keys.ball.optitrack_position)
            if raw is None:
                return None
            try:
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8")
                return np.array(json.loads(raw))
            except (ValueError, json.JSONDecodeError):
                return None

        return None

    # ------------------------------------------------------------ history

    def update(self) -> Optional[BallSample]:
        """Sample the ball once. Returns the new sample or None if rejected."""
        now = time.perf_counter() - self._t0
        pos = self._read_position()
        if pos is None or pos.shape != (3,):
            return None

        # Drop stale samples FIRST so a teleporting ball (sim relaunch /
        # OptiTrack regaining lock after a long dropout) doesn't get its
        # samples rejected forever against a frozen-old "last" position.
        while self._history and (now - self._history[0].t) > self._cfg.history_max_age_s:
            self._history.popleft()

        # Reject blatant jumps (OptiTrack mis-labeled marker, etc.) — but only
        # against a recent sample.
        if self._history:
            last = self._history[-1]
            if np.linalg.norm(pos - last.pos) > self._cfg.max_position_jump:
                return None

        sample = BallSample(t=now, pos=pos.astype(float))
        self._history.append(sample)
        self._last_seen_t = now

        # 3a: if a floor bounce is now clearly inside the rolling window, drop
        # pre-bounce samples so the next _fit_state runs on the post-bounce arc
        # only. Predictions during the few ticks straddling a bounce go from
        # garbage (averaged pre/post velocity) to "no prediction yet" (history
        # too short) and then to clean post-bounce predictions.
        if self._cfg.online_bounce_pruning:
            self._try_prune_pre_bounce()

        return sample

    def _try_prune_pre_bounce(self) -> bool:
        """Look for a floor bounce inside the rolling history. If one is
        clearly present, drop all samples before it. Returns True on prune.

        Detection is intentionally conservative — we want zero false positives
        on clean arcs, since dropping samples truncates the fit window. A
        bounce is "clearly present" when there's a local-min z below the
        floor threshold with strictly-greater z two samples on each side.
        """
        if len(self._history) < 6:
            return False
        samples = list(self._history)
        n = len(samples)
        zs = np.fromiter((s.pos[2] for s in samples), dtype=float, count=n)
        z_thr = self._cfg.online_bounce_z_threshold
        # Earliest bounce wins — that's the one whose pre-bounce data is
        # poisoning the fit right now.
        for i in range(2, n - 2):
            if zs[i] >= z_thr:
                continue
            if not (zs[i] <= zs[i - 1] and zs[i] <= zs[i + 1]):
                continue
            # Clear v-shape: z[i±2] strictly above the local min.
            if zs[i - 2] <= zs[i] or zs[i + 2] <= zs[i]:
                continue
            for _ in range(i):
                self._history.popleft()
            return True
        return False

    def time_since_last_seen(self) -> float:
        if self._last_seen_t is None:
            return float("inf")
        return (time.perf_counter() - self._t0) - self._last_seen_t

    def latest_position(self) -> Optional[np.ndarray]:
        return self._history[-1].pos if self._history else None

    def reset(self) -> None:
        self._history.clear()
        self._last_seen_t = None

    # ----------------------------------------------------------- estimation

    def _fit_state(self) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
        """Least-squares fit of (p0, v0) under constant +/- gravity.

        Uses the most recent sample as the time origin so that the returned
        position p0 is the *current* ball position and v0 is the *current*
        velocity. Returns (p0, v0, t_now_in_history).
        """
        if len(self._history) < 3:
            return None

        ts = np.array([s.t for s in self._history])
        pts = np.array([s.pos for s in self._history])
        t_now = ts[-1]
        dt = ts - t_now  # all <= 0

        # Subtract gravity contribution from z so we can solve linearly.
        z_corrected = pts[:, 2] + 0.5 * self._cfg.gravity * dt * dt

        # Solve [1, dt] [p0; v] = pos for each axis.
        A = np.stack([np.ones_like(dt), dt], axis=1)
        try:
            sol_x, _, _, _ = np.linalg.lstsq(A, pts[:, 0], rcond=None)
            sol_y, _, _, _ = np.linalg.lstsq(A, pts[:, 1], rcond=None)
            sol_z, _, _, _ = np.linalg.lstsq(A, z_corrected, rcond=None)
        except np.linalg.LinAlgError:
            return None

        p0 = np.array([sol_x[0], sol_y[0], sol_z[0]])
        v0 = np.array([sol_x[1], sol_y[1], sol_z[1]])
        return p0, v0, t_now

    def is_incoming(self) -> bool:
        """True if the ball is currently moving toward the robot (-X)."""
        fit = self._fit_state()
        if fit is None:
            return False
        _, v0, _ = fit
        return v0[0] < -self._cfg.min_incoming_speed

    def predict_intercept(self, strike_plane_x: float) -> Optional[Intercept]:
        """Solve for ball state at x = strike_plane_x.

        Propagates the fitted ballistic state forward, simulating up to
        cfg.max_bounces floor bounces if z would hit 0 before x reaches the
        strike plane. Returns None if the fit is unusable, the ball is not
        incoming, the predicted intercept would require more than the allowed
        number of bounces, or the time-to-impact is outside the configured
        lookahead window.
        """
        fit = self._fit_state()
        if fit is None:
            return None
        p0, v0, _ = fit

        # Need a non-trivial -X velocity to define a future intercept.
        if v0[0] >= -self._cfg.min_incoming_speed:
            return None

        result = _propagate_to_plane(p0, v0, strike_plane_x, self._cfg)
        if result is None:
            return None
        t_impact, p_impact, v_impact, n_bounces = result

        if not np.isfinite(t_impact):
            return None
        if t_impact < self._cfg.min_lookahead or t_impact > self._cfg.max_lookahead:
            return None

        return Intercept(
            position=p_impact,
            velocity=v_impact,
            time_to_impact=float(t_impact),
            n_bounces=int(n_bounces),
        )
