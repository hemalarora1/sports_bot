"""Foam ball tracker with aerodynamic drag physics.

Extends BallTracker with a drag-aware propagation model. The state estimation
(fitting p0, v0 from OptiTrack history) is inherited unchanged from
BallTracker — the LS fit over a short window (~100 ms) is accurate even with
drag because the drag correction is second-order over that timescale. The drag
model is applied during forward propagation to the strike plane, where flight
times of 0.1–1.5 s make the effect significant for a foam ball.

Physics (continuous)
--------------------
    p_dot = v

    v_dot_x = -k * ||v|| * v_x
    v_dot_y = -k * ||v|| * v_y
    v_dot_z = -g - k * ||v|| * v_z

    where ||v|| = sqrt(v_x^2 + v_y^2 + v_z^2)
    and g = 9.81 m/s^2, k = drag coefficient (m^-1, tune from data)

Discrete update (semi-implicit Euler, one time step dt)
-------------------------------------------------------
    s        = ||v||
    v_x_new  = v_x - k * s * v_x * dt
    v_y_new  = v_y - k * s * v_y * dt
    v_z_new  = v_z - (g + k * s * v_z) * dt
    x_new    = x   + v_x_new * dt
    y_new    = y   + v_y_new * dt
    z_new    = z   + v_z_new * dt

Usage (drop-in replacement for BallTracker)
-------------------------------------------
    from foam_ball.config import FoamBallConfig
    from foam_ball.tracker import FoamBallTracker

    cfg = FoamBallConfig(drag_coefficient=0.3)  # tune from real throws
    tracker = FoamBallTracker(redis_client, keys, cfg)
    # tracker.update() / tracker.predict_intercept() / tracker.reset()
    # behave identically to BallTracker.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import redis

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from state_machine.ball_tracker import (
    BallTracker,
    Intercept,
    REJECT_NONE,
    REJECT_INSUFFICIENT_HISTORY,
    REJECT_NOT_INCOMING,
    REJECT_PAST_PLANE,
    REJECT_ON_FLOOR,
    REJECT_WOULD_BOUNCE,
    REJECT_INFINITE_TTI,
    REJECT_TTI_TOO_SHORT,
    REJECT_TTI_TOO_LONG,
)
from state_machine.redis_keys import RedisKeys
from .config import FoamBallConfig


def _propagate_drag_to_plane(
    p0: np.ndarray,
    v0: np.ndarray,
    target_x: float,
    cfg: FoamBallConfig,
):
    """Propagate a foam ball trajectory to x = target_x using drag physics.

    Integrates the drag ODE with semi-implicit Euler at cfg.simulation_dt
    steps. Handles up to cfg.max_bounces floor bounces with the same
    restitution / tangential-damping model as the original propagator.

    Returns
    -------
    (t_total, p_at_target, v_at_target, n_bounces) on success.
    A REJECT_* string on failure — same codes as ball_tracker.py so the
    FSM and offline analyser see identical diagnostics.
    """
    p = p0.astype(float).copy()
    v = v0.astype(float).copy()
    g = cfg.gravity
    k = cfg.drag_coefficient
    dt = cfg.simulation_dt
    t_total = 0.0
    n_bounces = 0

    # Initial sanity checks (mirror _propagate_to_plane)
    if v[0] >= 0:
        return REJECT_NOT_INCOMING
    if p[0] <= target_x:
        return REJECT_PAST_PLANE
    if p[2] <= cfg.floor_epsilon and v[2] <= 0:
        return REJECT_ON_FLOOR

    max_t = cfg.max_lookahead + dt  # one extra step of slack

    while t_total < max_t:
        # --- Semi-implicit Euler step ---
        s = float(np.linalg.norm(v))
        v_new = np.array([
            v[0] - k * s * v[0] * dt,
            v[1] - k * s * v[1] * dt,
            v[2] - (g + k * s * v[2]) * dt,
        ])
        p_new = p + v_new * dt

        # --- Did ball cross the strike plane (x = target_x) this step? ---
        if p_new[0] <= target_x and p[0] > target_x:
            # Linear interpolation to the exact crossing instant
            denom = max(p[0] - p_new[0], 1e-12)
            alpha = float(np.clip((p[0] - target_x) / denom, 0.0, 1.0))
            t_cross = alpha * dt
            p_cross = np.array([
                target_x,
                p[1] + v_new[1] * t_cross,
                p[2] + v_new[2] * t_cross,
            ])
            t_at_impact = t_total + t_cross
            if not np.isfinite(t_at_impact):
                return REJECT_INFINITE_TTI
            return t_at_impact, p_cross, v_new, n_bounces

        # --- Did ball cross the floor (z = 0) this step? ---
        if p[2] > cfg.floor_epsilon and p_new[2] <= cfg.floor_epsilon and v_new[2] <= 0:
            if n_bounces >= cfg.max_bounces:
                return REJECT_WOULD_BOUNCE
            # Linear interpolation to the exact floor-crossing instant
            denom = max(p[2] - p_new[2], 1e-12)
            alpha = float(np.clip((p[2] - cfg.floor_epsilon) / denom, 0.0, 1.0))
            t_ground = alpha * dt
            t_total += t_ground
            p_bounce = np.array([
                p[0] + v_new[0] * t_ground,
                p[1] + v_new[1] * t_ground,
                cfg.floor_epsilon,
            ])
            v_z_before = v_new[2]
            p = p_bounce
            v = np.array([
                v_new[0] * cfg.bounce_tangential_damping,
                v_new[1] * cfg.bounce_tangential_damping,
                -cfg.bounce_restitution * v_z_before,
            ])
            n_bounces += 1
            continue

        # Ball on floor and not rising — trajectory stuck
        if p_new[2] <= cfg.floor_epsilon and v_new[2] <= 0:
            return REJECT_ON_FLOOR

        p = p_new
        v = v_new
        t_total += dt

    return REJECT_TTI_TOO_LONG


class FoamBallTracker(BallTracker):
    """Drop-in replacement for BallTracker using drag-aware trajectory propagation.

    All data management (Redis I/O, sliding window, median filter, LS fit,
    stale detection, bounce pruning) is inherited from BallTracker unchanged.
    Only predict_intercept is overridden to call _propagate_drag_to_plane
    instead of _propagate_to_plane.

    Parameters
    ----------
    redis_client : redis.Redis
    keys : RedisKeys
    cfg : FoamBallConfig
        Must be a FoamBallConfig (not bare BallTrackerConfig) so the drag
        parameters are present. BallTracker accepts it via Liskov
        substitution since FoamBallConfig extends BallTrackerConfig.
    """

    def __init__(
        self,
        redis_client: redis.Redis,
        keys: RedisKeys,
        cfg: FoamBallConfig,
    ):
        super().__init__(redis_client, keys, cfg)

    def predict_intercept(self, strike_plane_x: float) -> Optional[Intercept]:
        """Solve for ball state at x = strike_plane_x using drag physics.

        Identical logic to BallTracker.predict_intercept but substitutes
        _propagate_drag_to_plane for _propagate_to_plane.
        """
        self.last_reject_reason = REJECT_NONE
        min_hist = max(3, self._cfg.min_history_for_prediction)
        if len(self._history) < min_hist:
            self.last_reject_reason = REJECT_INSUFFICIENT_HISTORY
            return None

        fit = self._fit_state()
        if fit is None:
            self.last_reject_reason = REJECT_INSUFFICIENT_HISTORY
            return None
        p0, v0, _ = fit

        if v0[0] >= -self._cfg.min_incoming_speed:
            self.last_reject_reason = REJECT_NOT_INCOMING
            return None

        result = _propagate_drag_to_plane(p0, v0, strike_plane_x, self._cfg)
        if isinstance(result, str):
            self.last_reject_reason = result
            return None
        t_impact, p_impact, v_impact, n_bounces = result

        if not np.isfinite(t_impact):
            self.last_reject_reason = REJECT_INFINITE_TTI
            return None
        if t_impact < self._cfg.min_lookahead:
            self.last_reject_reason = REJECT_TTI_TOO_SHORT
            return None
        if t_impact > self._cfg.max_lookahead:
            self.last_reject_reason = REJECT_TTI_TOO_LONG
            return None

        return Intercept(
            position=p_impact,
            velocity=v_impact,
            time_to_impact=float(t_impact),
            n_bounces=int(n_bounces),
        )
