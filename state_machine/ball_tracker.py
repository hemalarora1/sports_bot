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

        return sample

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

        Returns None if the fit is unusable, the ball is not incoming, the
        intercept lies in the past, or the time-to-impact is outside the
        configured lookahead window.
        """
        fit = self._fit_state()
        if fit is None:
            return None
        p0, v0, _ = fit

        # Need a non-trivial -X velocity to define a future intercept on the plane.
        if v0[0] >= -self._cfg.min_incoming_speed:
            return None

        # Solve x0 + vx * t = strike_plane_x.
        t_impact = (strike_plane_x - p0[0]) / v0[0]
        if not np.isfinite(t_impact):
            return None
        if t_impact < self._cfg.min_lookahead or t_impact > self._cfg.max_lookahead:
            return None

        x = p0[0] + v0[0] * t_impact
        y = p0[1] + v0[1] * t_impact
        z = p0[2] + v0[2] * t_impact - 0.5 * self._cfg.gravity * t_impact * t_impact

        vx = v0[0]
        vy = v0[1]
        vz = v0[2] - self._cfg.gravity * t_impact

        return Intercept(
            position=np.array([x, y, z]),
            velocity=np.array([vx, vy, vz]),
            time_to_impact=t_impact,
        )
