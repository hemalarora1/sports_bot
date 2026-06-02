"""Foam ball trajectory tracking with aerodynamic drag physics.

Quick start
-----------
    import redis
    from foam_ball import make_foam_tracker

    r = redis.Redis()
    tracker = make_foam_tracker(r)          # uses RigidBody002 rigid-body ID 14 + fitted k=0.20
    tracker.update()                         # same interface as BallTracker
    intercept = tracker.predict_intercept(strike_plane_x=0.60)

Manual wiring (if you need custom redis keys or config tweaks):

    import dataclasses, redis
    from state_machine.redis_keys import RedisKeys
    from foam_ball import FoamBallConfig, FoamBallTracker

    r = redis.Redis()
    keys = RedisKeys(ball_source="optitrack")
    keys.ball = dataclasses.replace(keys.ball, optitrack_rigid_body_id=14)  # RigidBody002
    cfg = FoamBallConfig(drag_coefficient=0.20)
    tracker = FoamBallTracker(r, keys, cfg)
"""

from __future__ import annotations

import dataclasses

import redis as _redis

from .config import FoamBallConfig
from .tracker import FoamBallTracker

# Foam ball OptiTrack rigid-body ID (Motive object name: "RigidBody002").
FOAM_BALL_RIGID_BODY_ID: int = 13


def make_foam_tracker(
    redis_client: _redis.Redis,
    *,
    rigid_body_id: int = FOAM_BALL_RIGID_BODY_ID,
    drag_coefficient: float | None = None,
    **cfg_overrides,
) -> FoamBallTracker:
    """Create a FoamBallTracker wired to the foam ball OptiTrack rigid body.

    Parameters
    ----------
    redis_client:
        Shared Redis connection.
    rigid_body_id:
        OptiTrack Motive rigid-body ID for RigidBody002 (default: 14).
    drag_coefficient:
        Override the fitted k (m^-1). Leave None to use the default from
        FoamBallConfig (currently 0.20, fitted from 2026-06-01 recordings).
    **cfg_overrides:
        Any other FoamBallConfig fields to override (e.g. max_bounces=1).
    """
    from state_machine.redis_keys import RedisKeys

    keys = RedisKeys(ball_source="optitrack")
    keys.ball = dataclasses.replace(keys.ball, optitrack_rigid_body_id=rigid_body_id)

    cfg_kwargs = cfg_overrides
    if drag_coefficient is not None:
        cfg_kwargs["drag_coefficient"] = drag_coefficient
    cfg = FoamBallConfig(**cfg_kwargs)

    return FoamBallTracker(redis_client, keys, cfg)


__all__ = [
    "FoamBallConfig",
    "FoamBallTracker",
    "make_foam_tracker",
    "FOAM_BALL_RIGID_BODY_ID",
]
