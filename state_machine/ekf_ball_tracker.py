"""Probabilistic ball tracker — Kalman-filter-with-bounce-jumps variant.

This is an experimental alternative to the production sliding-window
least-squares tracker in ``ball_tracker.py``. It exposes the same public
interface (``update``, ``is_incoming``, ``predict_intercept``,
``time_since_last_seen``, ``latest_position``, ``reset``, ``_fit_state``) so
the offline analyzer (``ball_tracker_test.py``) can swap one for the other
without code changes to anything downstream.

Why this might be better
========================

The least-squares fitter solves for ``(p0, v0)`` over a window of the last
~12 samples on every tick. Two costs of that approach:

1. Each tick's estimate ignores everything older than the window.
2. The fit weights samples equally — a noisy sample 0.05 s ago has the same
   influence as a clean sample 0.30 s ago.

A recursive estimator addresses both: it carries a *running* posterior
``(x̂, P)`` that accumulates information from all past samples, weighted
inverse-variance against sensor noise + process noise. In the absence of
bounces this is *strictly* more sample-efficient than a sliding-window LS
fit.

Bounces stay first-class
========================

We retain the discrete bounce model from the LS tracker:

* On a *detected* floor bounce inside the rolling sample history, re-seed
  the filter from the post-bounce samples so the contaminated pre-bounce
  posterior is discarded. (Symmetric to ``BallTracker._try_prune_pre_bounce``.)
* On a *predicted* floor crossing inside ``predict_intercept`` (i.e. the
  ball would cross z=0 before x = strike_plane_x), apply a deterministic
  state jump with covariance inflation by ``(σ_e, σ_μ)``, then continue.

The result is a predicted intercept with both a mean *and* a position
covariance — useful for any future commit logic that wants "swing when the
prediction is tight enough" rather than "swing when time-to-impact crosses
a hardcoded threshold."

Scope
=====

This file does not touch the FSM. The production ``BallTracker`` is
unchanged, the production ``Intercept`` gains an optional ``position_cov``
field (defaults to None), and CLI plumbing lives in ``ball_tracker_test.py``.
Run an A/B over an existing recording with::

    python -m sports_bot.state_machine.ball_tracker_test analyze \
        sports_bot/recordings/<file>.npz --tracker ekf

State vector layout
===================

    x = [px, py, pz, vx, vy, vz]  (6,)
    P : (6, 6)
"""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple

import numpy as np
import redis

from .ball_tracker import BallSample, Intercept
from .config import BallTrackerConfig, EKFConfig
from .redis_keys import RedisKeys


# ---------------------------------------------------------------------------
# Linear dynamics helpers
# ---------------------------------------------------------------------------

def _transition_matrix(dt: float) -> np.ndarray:
    """F such that x' = F @ x + control(g, dt). Linear in the state."""
    F = np.eye(6)
    F[0, 3] = dt
    F[1, 4] = dt
    F[2, 5] = dt
    return F


def _gravity_input(dt: float, g: float) -> np.ndarray:
    """Deterministic gravity contribution to x' = F x + u. Acts on z only."""
    u = np.zeros(6)
    u[2] = -0.5 * g * dt * dt
    u[5] = -g * dt
    return u


def _process_noise(dt: float, ekf_cfg: EKFConfig) -> np.ndarray:
    """Discretized white-acceleration process noise.

    For each axis with state (p, v) and an unmodeled acceleration with std
    σ_a, the discrete-time noise covariance is

        Q_axis = σ_a² · [[ dt⁴/4,  dt³/2 ],
                         [ dt³/2,  dt²   ]]

    The 6-D Q embeds three of those, one per axis, with σ_a possibly
    differing between (xy) and z.
    """
    Q = np.zeros((6, 6))
    sx = ekf_cfg.process_accel_std_xy
    sy = ekf_cfg.process_accel_std_xy
    sz = ekf_cfg.process_accel_std_z
    sigmas = [sx, sy, sz]
    dt2 = dt * dt
    dt3 = dt2 * dt
    dt4 = dt3 * dt
    for axis, sigma in enumerate(sigmas):
        s2 = sigma * sigma
        Q[axis, axis] = s2 * dt4 / 4.0
        Q[axis, axis + 3] = s2 * dt3 / 2.0
        Q[axis + 3, axis] = s2 * dt3 / 2.0
        Q[axis + 3, axis + 3] = s2 * dt2
    return Q


# H = [I_3 | 0_3]: we observe position only.
_H = np.zeros((3, 6))
_H[:3, :3] = np.eye(3)


# ---------------------------------------------------------------------------
# Bounce propagation with covariance
# ---------------------------------------------------------------------------

def _propagate_segment(
    x: np.ndarray, P: np.ndarray, dt: float, cfg: BallTrackerConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Advance ``(x, P)`` by ``dt`` under the no-bounce linear dynamics."""
    F = _transition_matrix(dt)
    x_next = F @ x + _gravity_input(dt, cfg.gravity)
    P_next = F @ P @ F.T + _process_noise(dt, cfg.ekf)
    return x_next, P_next


def _apply_bounce(
    x: np.ndarray, P: np.ndarray, cfg: BallTrackerConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply a floor bounce: v_z' = −e·v_z, v_xy' = μ_t·v_xy. p unchanged.

    Covariance picks up two contributions:
      (a) The deterministic linearization through the bounce Jacobian B.
      (b) Parameter uncertainty in (e, μ_t), treating them as random with
          known std. This inflates only the velocity block.
    """
    e = cfg.bounce_restitution
    mu = cfg.bounce_tangential_damping
    B = np.diag([1.0, 1.0, 1.0, mu, mu, -e])
    x_next = B @ x
    x_next[2] = 0.0  # clamp to floor; numerical drift can leave a tiny residual

    P_next = B @ P @ B.T
    # Parameter-uncertainty inflation. v'_x = μ·v_x  →  Δσ²(v'_x) ≈ (v_x · σ_μ)²
    sig_mu = cfg.ekf.bounce_tangential_damping_std
    sig_e = cfg.ekf.bounce_restitution_std
    vx, vy, vz = x[3], x[4], x[5]
    P_next[3, 3] += (vx * sig_mu) ** 2
    P_next[4, 4] += (vy * sig_mu) ** 2
    P_next[5, 5] += (vz * sig_e) ** 2
    return x_next, P_next


def _ekf_propagate_to_plane(
    x: np.ndarray, P: np.ndarray, target_x: float, cfg: BallTrackerConfig,
) -> Optional[Tuple[float, np.ndarray, np.ndarray, np.ndarray, int]]:
    """Propagate ``(x, P)`` forward (with bounces) to ``px = target_x``.

    Returns ``(t_total, x_at_plane, P_at_plane, _, n_bounces)`` or None.

    The implementation mirrors ``ball_tracker._propagate_to_plane`` exactly
    for the mean trajectory; the only difference is that we also evolve P
    along each segment and apply the bounce-Jacobian + parameter-noise
    inflation at every modeled bounce.
    """
    x = x.astype(float).copy()
    P = P.astype(float).copy()
    g = cfg.gravity
    t_total = 0.0

    for n_bounces in range(cfg.max_bounces + 1):
        if x[3] >= 0:  # not moving toward plane
            return None
        t_to_target = (target_x - x[0]) / x[3]
        if t_to_target < 0:
            return None
        # Stuck on floor with no rebound velocity — model breaks down.
        if x[2] <= cfg.floor_epsilon and x[5] <= 0:
            return None

        disc = x[5] * x[5] + 2.0 * g * x[2]
        if disc < 0:
            t_to_ground = float("inf")
        else:
            t_to_ground = (x[5] + np.sqrt(disc)) / g

        if t_to_target <= t_to_ground:
            x_final, P_final = _propagate_segment(x, P, t_to_target, cfg)
            # Force x onto the plane exactly to wipe out propagation slop.
            x_final[0] = target_x
            t_total += t_to_target
            return t_total, x_final, P_final, x_final[3:6].copy(), n_bounces

        if n_bounces >= cfg.max_bounces:
            return None

        x, P = _propagate_segment(x, P, t_to_ground, cfg)
        # The integrated dynamics can leave x[2] slightly above or below 0
        # depending on dt; clamp before bouncing for numerical sanity.
        x[2] = 0.0
        x, P = _apply_bounce(x, P, cfg)
        t_total += t_to_ground

    return None


# ---------------------------------------------------------------------------
# EKFBallTracker
# ---------------------------------------------------------------------------

class EKFBallTracker:
    """Recursive ball state estimator with discrete bounce jumps.

    Same public surface as ``ball_tracker.BallTracker``: ``update()`` once
    per tick, then read ``is_incoming()`` / ``predict_intercept()``. The
    analyzer also pokes at ``_fit_state()`` (which returns the *current*
    posterior mean re-keyed to match the LS tracker's convention).
    """

    def __init__(
        self,
        redis_client: Optional[redis.Redis],
        keys: Optional[RedisKeys],
        cfg: BallTrackerConfig,
    ):
        self._redis = redis_client
        self._keys = keys
        self._cfg = cfg
        self._ekf_cfg = cfg.ekf

        # Posterior state — None until seeded.
        self._x: Optional[np.ndarray] = None
        self._P: Optional[np.ndarray] = None
        self._t_last: Optional[float] = None  # tracker-time of the last update

        # Seed buffer + history. The history mirrors BallTracker's deque so
        # we can reuse the same conservative bounce detector — same input,
        # same trigger conditions, just a different recovery action
        # (re-seed instead of drop-pre-bounce).
        self._seed: list[BallSample] = []
        self._history: Deque[BallSample] = deque(maxlen=cfg.history_size)

        self._t0 = time.perf_counter()
        self._last_seen_t: Optional[float] = None

    # ----- redis I/O (mirrors BallTracker exactly) ----------------------

    def _read_position(self) -> Optional[np.ndarray]:
        if self._redis is None or self._keys is None:
            return None
        if self._keys.ball_source == "opensai":
            raw = self._redis.get(self._keys.ball.opensai_object_pose)
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

    # ----- main update --------------------------------------------------

    def update(self) -> Optional[BallSample]:
        """Pull the latest measurement and fold it into the posterior."""
        now = time.perf_counter() - self._t0
        pos = self._read_position()
        if pos is None or pos.shape != (3,):
            return None
        return self._ingest(now, pos)

    def _ingest(self, t: float, pos: np.ndarray) -> Optional[BallSample]:
        """Filter + history update for one measurement. Returns None if the
        sample was rejected (stale duplicate, position jump, or Mahalanobis
        outlier against the current posterior)."""
        # Drop stale samples from the (small) rolling history first, like the
        # LS tracker, so a teleporting ball doesn't get rejected forever
        # against a frozen-old "last" position.
        while self._history and (t - self._history[0].t) > self._cfg.history_max_age_s:
            self._history.popleft()

        if self._history:
            last = self._history[-1]
            if np.linalg.norm(pos - last.pos) > self._cfg.max_position_jump:
                return None

        sample = BallSample(t=float(t), pos=pos.astype(float))

        # Two regimes: seeding (filter not initialized yet) vs filtering.
        if self._x is None:
            self._seed.append(sample)
            if len(self._seed) >= max(2, self._ekf_cfg.seed_samples):
                self._initialize_from_seed()
        else:
            self._predict_and_update(sample)

        self._history.append(sample)
        self._last_seen_t = float(t)

        if self._ekf_cfg.online_bounce_handling:
            self._maybe_reseed_for_bounce()

        return sample

    # ----- seeding ------------------------------------------------------

    def _initialize_from_seed(self) -> None:
        """Initialize ``x`` and ``P`` from a least-squares fit of the seed
        window. Conceptually: kick the recursive filter off with a sensible
        first estimate rather than waiting for it to converge from
        zero-velocity prior."""
        ts = np.array([s.t for s in self._seed])
        ps = np.array([s.pos for s in self._seed])
        t_now = ts[-1]
        dt = ts - t_now

        # Solve [1, dt] @ [p0; v0] = positions (z corrected for gravity).
        A = np.stack([np.ones_like(dt), dt], axis=1)
        g = self._cfg.gravity
        z_corrected = ps[:, 2] + 0.5 * g * dt * dt
        sx, *_ = np.linalg.lstsq(A, ps[:, 0], rcond=None)
        sy, *_ = np.linalg.lstsq(A, ps[:, 1], rcond=None)
        sz, *_ = np.linalg.lstsq(A, z_corrected, rcond=None)

        self._x = np.array([sx[0], sy[0], sz[0], sx[1], sy[1], sz[1]])
        # Initial covariance: small on position (seed is a tight LS fit),
        # generous on velocity until updates tighten it.
        self._P = np.diag(
            [self._ekf_cfg.initial_pos_std ** 2] * 3
            + [self._ekf_cfg.initial_vel_std ** 2] * 3
        )
        self._t_last = float(t_now)
        # The seed buffer has served its purpose.
        self._seed = []

    # ----- predict + update --------------------------------------------

    def _predict_and_update(self, sample: BallSample) -> None:
        """Standard linear KF predict + measurement update."""
        assert self._x is not None and self._P is not None and self._t_last is not None
        dt = sample.t - self._t_last
        if dt <= 0:
            # Out-of-order or duplicate sample — skip the propagation step,
            # but still try the measurement update with zero dt.
            dt = 0.0

        # Predict.
        x_pred, P_pred = _propagate_segment(self._x, self._P, dt, self._cfg)

        # Innovation.
        z = sample.pos
        y = z - _H @ x_pred
        R = (self._ekf_cfg.measurement_pos_std ** 2) * np.eye(3)
        S = _H @ P_pred @ _H.T + R

        # Outlier rejection via Mahalanobis distance.
        try:
            mahal = float(np.sqrt(y @ np.linalg.solve(S, y)))
        except np.linalg.LinAlgError:
            mahal = float("inf")
        if mahal > self._ekf_cfg.outlier_mahalanobis_thresh:
            # Keep the prediction, don't fold this measurement in. The next
            # clean sample will pull the state back.
            self._x = x_pred
            self._P = P_pred
            self._t_last = sample.t
            return

        # Joseph form for numerical stability of the covariance update.
        K = P_pred @ _H.T @ np.linalg.inv(S)
        I_KH = np.eye(6) - K @ _H
        self._x = x_pred + K @ y
        self._P = I_KH @ P_pred @ I_KH.T + K @ R @ K.T
        self._t_last = sample.t

    # ----- online bounce handling --------------------------------------

    def _maybe_reseed_for_bounce(self) -> bool:
        """If a clear floor bounce sits in the rolling history, drop the
        pre-bounce samples and re-seed the filter from the post-bounce ones.
        Returns True iff a re-seed was performed.

        Detection logic is identical to BallTracker._try_prune_pre_bounce so
        the EKF and LS variants see the same trigger conditions in an A/B.
        """
        if len(self._history) < 6:
            return False
        samples = list(self._history)
        n = len(samples)
        zs = np.fromiter((s.pos[2] for s in samples), dtype=float, count=n)
        z_thr = self._ekf_cfg.online_bounce_z_threshold
        for i in range(2, n - 2):
            if zs[i] >= z_thr:
                continue
            if not (zs[i] <= zs[i - 1] and zs[i] <= zs[i + 1]):
                continue
            if zs[i - 2] <= zs[i] or zs[i + 2] <= zs[i]:
                continue
            # Clear bounce at sample index i. Keep [i, n) only and re-seed.
            post = samples[i:]
            for _ in range(i):
                self._history.popleft()
            self._x = None
            self._P = None
            self._t_last = None
            self._seed = list(post)
            if len(self._seed) >= max(2, self._ekf_cfg.seed_samples):
                self._initialize_from_seed()
            return True
        return False

    # ----- read-side API mirroring BallTracker --------------------------

    def time_since_last_seen(self) -> float:
        if self._last_seen_t is None:
            return float("inf")
        return (time.perf_counter() - self._t0) - self._last_seen_t

    def latest_position(self) -> Optional[np.ndarray]:
        if self._x is not None:
            return self._x[:3].copy()
        if self._history:
            return self._history[-1].pos
        return None

    def reset(self) -> None:
        self._x = None
        self._P = None
        self._t_last = None
        self._seed = []
        self._history.clear()
        self._last_seen_t = None

    # ----- LS-tracker-shaped query interface ---------------------------

    def _fit_state(self) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
        """Return ``(p_current, v_current, t_now_in_tracker_time)``.

        Naming mirrors ``BallTracker._fit_state`` so the analyzer's
        ``_replay`` loop can be shared. The "fit" here is just the current
        posterior mean — there's no per-tick least-squares solve.
        """
        if self._x is None or self._t_last is None:
            return None
        return self._x[:3].copy(), self._x[3:].copy(), float(self._t_last)

    def is_incoming(self) -> bool:
        if self._x is None:
            return False
        return bool(self._x[3] < -self._cfg.min_incoming_speed)

    def predict_intercept(self, strike_plane_x: float) -> Optional[Intercept]:
        if self._x is None or self._P is None:
            return None
        if self._x[3] >= -self._cfg.min_incoming_speed:
            return None

        result = _ekf_propagate_to_plane(
            self._x, self._P, strike_plane_x, self._cfg,
        )
        if result is None:
            return None
        t_impact, x_final, P_final, v_final, n_bounces = result
        if not np.isfinite(t_impact):
            return None
        if t_impact < self._cfg.min_lookahead or t_impact > self._cfg.max_lookahead:
            return None

        return Intercept(
            position=x_final[:3].copy(),
            velocity=v_final.copy(),
            time_to_impact=float(t_impact),
            n_bounces=int(n_bounces),
            position_cov=P_final[:3, :3].copy(),
        )

    # ----- diagnostic accessor (used by viser in the analyzer) ---------

    def posterior(self) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
        """Current posterior ``(x, P, t)`` or None if not yet seeded.

        Exposed for the analyzer to plot the uncertainty ellipsoid; the FSM
        doesn't use this.
        """
        if self._x is None or self._P is None or self._t_last is None:
            return None
        return self._x.copy(), self._P.copy(), float(self._t_last)
