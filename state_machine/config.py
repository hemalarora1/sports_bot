"""Tunable parameters for the pickleball state machine.

World frame convention (used everywhere unless noted):
    +X : pointing forward, toward the net / opponent.
    +Y : lateral (sideline).
    +Z : up.
The robot's "home" base position is the origin.
"""

from dataclasses import dataclass, field
import math
import numpy as np


DEG_TO_RAD = math.pi / 180.0


@dataclass
class CourtConfig:
    """Court / strike geometry, all in world frame meters."""

    # X position (forward distance from home base) at which the FSM commits to
    # hitting the ball. The intercept is solved on this plane.
    strike_plane_x: float = 0.60

    # Allowed lateral range the base can shift to chase a ball.
    base_y_min: float = -1.5
    base_y_max: float = 1.5

    # Allowed forward range of the base (kept small for safety in sim bring-up).
    base_x_min: float = -0.30
    base_x_max: float = 0.30

    # Strike point height range. Balls predicted outside this range are rejected.
    strike_z_min: float = 0.30
    strike_z_max: float = 1.40

    # Where we want returns to land in world frame (used to orient the racket).
    return_target_xyz: np.ndarray = field(
        default_factory=lambda: np.array([4.0, 0.0, 0.05])
    )


@dataclass
class RacketConfig:
    """Geometry of the MTEN MT-01 paddle as mounted on the panda flange.

    Reference values (used by the C++ controller as the controlled-frame setup
    for the paddle TCP); the FSM does not consume these directly.
    """

    # Sweet-spot offset in the link7 frame. Flange is at link7 +z = 0.107 m;
    # paddle face center is another 0.261 m along link7 +z.
    sweet_spot_in_flange: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.368])
    )

    # Paddle face normal expressed in the link7 frame. Paddle is mounted so that
    # link7 +x is the face-strike direction.
    face_normal_in_flange: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0])
    )

    # Distance to pull the racket back behind the strike point for the wind-up.
    wind_up_offset: float = 0.25

    # Distance to push the racket past the strike point on follow-through.
    follow_through_offset: float = 0.25

    # Desired racket linear speed at impact (m/s). Pickleball returns are
    # typically 5-10 m/s; start conservative for sim bring-up.
    impact_speed: float = 4.0


@dataclass
class ReadyPose:
    """Home / ready pose the robot returns to between hits."""

    # Mobile base [x, y, theta] in world frame.
    base_pose: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0])
    )

    # Racket sweet-spot position in world frame. Pinned to the controller's
    # natural rest pose at the configured arm posture so INIT settles at zero
    # cartesian error. The FSM's APPROACH state drives the racket UP to the
    # predicted ball intercept (typically z>1 m); the controller's OTG limits
    # (6 m/s, 30 m/s^2) traverse that in well under the ball flight time.
    racket_position: np.ndarray = field(
        default_factory=lambda: np.array([0.675, 0.0, 0.753])
    )

    # Racket orientation as a 3x3 rotation. Columns are
    # [face_right, face_up, face_normal] in world. Tuned to match the
    # controller's wrist orientation at the configured posture: face_normal ≈
    # +X (toward opponent), face_up ≈ -Z (paddle hanging down from wrist —
    # joint 7 limits don't allow a clean +Z face_up at this posture; the
    # cartesian controller still drives the TCP wherever the FSM commands).
    racket_orientation: np.ndarray = field(
        default_factory=lambda: np.array([
            [ 0.0,    0.077,  0.997 ],   # row 0 (world X components)
            [-1.0,    0.0,    0.0   ],   # row 1 (world Y components)
            [ 0.0,   -0.997,  0.077 ],   # row 2 (world Z components)
        ])
    )


@dataclass
class EKFConfig:
    """Parameters for the probabilistic (EKF) ball tracker variant.

    See sports_bot/state_machine/ekf_ball_tracker.py. These are only consumed
    when the EKF tracker is selected — the production least-squares tracker
    ignores them.

    State vector: x = [px, py, pz, vx, vy, vz].  Dynamics between bounces are
    constant-velocity in xy and constant-acceleration (−g) in z. Bounces are
    modeled as discrete state jumps with parameter uncertainty inflating the
    velocity covariance.
    """

    # Process-noise acceleration spectral density (m/s²). Models unmodeled
    # accelerations between samples — drag, Magnus on a spinning ball, sensor
    # timing jitter. Tuned independently for xy (more Magnus + drag effect)
    # and z (mostly drag — gravity itself is in the dynamics, not noise).
    #
    # Set generously: pickleball drag is ~1-2 m/s² at typical speeds and
    # Magnus adds another 1-3 m/s² for a spinning ball, but the more
    # important constraint is that σ_a must be *large enough that the
    # filter can adapt when its model diverges from the data*. Set too low
    # and the filter becomes overconfident, rejects legitimate measurements
    # via outlier gating, and locks into the seed velocity. Empirically
    # σ_a_xy ≈ 8 keeps the filter responsive without obvious oversmoothing
    # on the SRC Kitchen recordings.
    process_accel_std_xy: float = 8.0
    process_accel_std_z: float = 5.0

    # Position measurement noise (m). OptiTrack pose is sub-mm under good
    # marker visibility — but the *effective* measurement noise also has to
    # absorb model error not captured by the process noise (e.g., marker
    # mislabeling for a few ticks, transient occlusion). 5 mm is a
    # pessimistic-but-stable default; tighten it once we trust the rig.
    measurement_pos_std: float = 0.005

    # Initial covariance after seeding the filter with the first few samples.
    initial_pos_std: float = 0.02
    initial_vel_std: float = 3.0

    # Mahalanobis distance above which an incoming sample is treated as an
    # outlier and skipped. Set conservatively high — the dumb
    # ``max_position_jump`` filter (shared with the LS tracker) already
    # catches gross teleports, and a tight Mahalanobis gate can pathologically
    # reject *all* measurements when σ_meas is small and the filter's mean is
    # mildly biased (e.g. just after a bounce reseed with imperfect velocity
    # seed) — empirically observed as 95% rejection rates.
    outlier_mahalanobis_thresh: float = 20.0

    # Number of seed samples used to initialize position + velocity via a
    # quick least-squares fit before the recursive filter takes over. Six
    # samples (~50 ms at OptiTrack's 120 Hz) gives a markedly better velocity
    # estimate than 4 — and the recursive update doesn't fix bad initial v
    # very quickly when σ_meas is small.
    seed_samples: int = 6

    # Same online bounce-detection idea as the LS version, but instead of
    # dropping pre-bounce samples, the EKF re-seeds from the post-bounce
    # samples (its filter state was contaminated by pre-bounce data).
    # Disable to compare against a "naive" EKF that just keeps integrating.
    online_bounce_handling: bool = True
    online_bounce_z_threshold: float = 0.10

    # Additional velocity-covariance inflation applied at every *predicted*
    # bounce inside predict_intercept's forward propagation, to account for
    # the empirical std of e and μ_t observed on the SRC Kitchen recordings.
    # (e = 0.70 ± 0.058,  μ_t = 0.62 ± 0.127.)
    bounce_restitution_std: float = 0.058
    bounce_tangential_damping_std: float = 0.127


@dataclass
class BallTrackerConfig:
    """Parameters for ball trajectory estimation.

    **Volley-only mode (current defaults, SRC Kitchen bring-up 2026-05-19).**
    The tracker fits a single ballistic arc on a rolling window and lets
    ``predict_intercept`` propagate that arc forward to the strike plane.
    With ``max_bounces = 0`` any propagation that would cross z = 0 before
    reaching the strike plane is rejected — i.e. groundstrokes and anything
    that bounces in front of the robot return ``None`` to the FSM, which
    just stays in READY. This is intentional: bounces are harder to predict
    well, and volleys are a sufficient demonstration of the full pipeline.

    **To re-enable bounce handling (ground-stroke mode):**

        cfg.tracker.max_bounces = 1
        cfg.tracker.online_bounce_pruning = True
        cfg.tracker.history_size = 12
        cfg.tracker.history_max_age_s = 0.30

    The (e, μ_t) values below are already measured for SRC Kitchen
    (Phase 1, 11 real bounces), so flipping back requires no further
    tuning. See context.md → "What we're debugging now" for the A/B
    numbers under each mode.
    """

    gravity: float = 9.81

    # Sliding window for the LS fit. Two caps interact:
    #   - history_size: max samples kept (binding at high sample rates)
    #   - history_max_age_s: max sample age (binding at low / sparse rates)
    # We pick values such that the *typical* window is dominated by
    # ``history_size`` at OptiTrack's 120 / 240 Hz rates, while
    # ``history_max_age_s`` is permissive enough to keep the fit robust
    # when the sample stream becomes sparse (carry-around periods,
    # occlusion-induced low effective Hz).
    #
    # Sweep across all 2026-05-19 SRC Kitchen recordings (12 sessions, 22
    # throws) at the commit-moment bucket (tti ∈ [0.00, 0.15) s):
    #   size= 8 / age=0.20 : 5.3 cm  (the previously-best 120 Hz setting)
    #   size=10 / age=0.15 : 5.4 cm
    #   size=12 / age=0.15 : 5.8 cm  ◀ chosen — small regression at 120 Hz
    #   size=16 / age=0.10 : 13.7 cm (age too tight → noisy v on sparse stretches)
    #   size=16 / age=0.15 : 6.6 cm
    # The "longer window = smoother fit" intuition is wrong: the LS velocity
    # is a window-average, so a longer window adds lag against the true
    # current v. ``size=12`` is chosen to anticipate the 240 Hz upgrade —
    # at 240 Hz this gives a 50 ms window (richer than size=8's 33 ms),
    # while at 120 Hz it gives 100 ms (slightly long but acceptable). When
    # 240 Hz is live, re-sweep on real cart data and fine-tune. Reset to
    # ``size=12, age=0.30`` if switching back to bounce mode (longer
    # window needed to span clean post-bounce arcs).
    history_size: int = 12
    history_max_age_s: float = 0.15

    # Minimum forward speed (m/s, world -X direction) for a ball to be
    # considered "incoming". Below this we ignore it.
    min_incoming_speed: float = 0.5

    # Reject jumps in position larger than this between consecutive samples
    # (m). Filters out OptiTrack glitches / dropouts.
    max_position_jump: float = 0.5

    # Per-axis median filter applied to raw OptiTrack positions *before* they
    # hit the rolling history. Smooths out single-sample mislabels / marker
    # swaps / brief reflection artifacts at the cost of ~1 sample of lag in
    # the reported "current" position (which is below sensor noise at
    # 120/240 Hz). Set to 0 or 1 to disable; 3 is a sensible default
    # (single-outlier rejection without smearing real velocity).
    median_filter_window: int = 3

    # Time horizon (s) within which a predicted intercept is considered usable.
    max_lookahead: float = 1.5
    min_lookahead: float = 0.05

    # When true, BallTracker.update() detects floor bounces inside the rolling
    # history and drops pre-bounce samples so the next fit only sees the new
    # ballistic arc (Phase 3a). Disabled in volley mode for two reasons:
    # (a) by definition there are no bounces to prune, (b) a false trigger
    # (e.g. low-z noise at the end of flight, or ball briefly sitting on the
    # floor before being picked up) would wipe the fit window and produce a
    # silent prediction stall. Re-enable in bounce mode.
    online_bounce_pruning: bool = False
    # z threshold (m) below which a local minimum is treated as a floor bounce
    # for the online pruner. Unused while online_bounce_pruning = False.
    online_bounce_z_threshold: float = 0.10

    # ----- Bounce model -----
    # Used by predict_intercept when the fitted ballistic arc would cross z=0
    # before reaching the strike plane. Values below are fitted from 11 real
    # bounces in the SRC Kitchen recordings on 2026-05-17 (median used; see
    # context.md "What we're debugging now" for the per-recording breakdown).
    # Re-measure per-bay or after court changes via:
    #   python -m sports_bot.state_machine.ball_tracker_test analyze <rec>
    #
    # max_bounces is the volley/bounce-mode toggle:
    #   0 = volley-only — reject any predicted trajectory that bounces.
    #   1 = ground-strokes allowed — propagate through one floor bounce.
    bounce_restitution: float = 0.70         # e = -v_z_after / v_z_before
    bounce_tangential_damping: float = 0.62  # μ_t = ||v_xy_after|| / ||v_xy_before||
    max_bounces: int = 0                     # volley-only by default; flip to 1 for bounces
    # Treat the ball as "at/below floor" (model breaks down) below this z.
    floor_epsilon: float = 1e-3

    # Parameters for the experimental EKF tracker variant. Ignored by the
    # production least-squares tracker.
    ekf: EKFConfig = field(default_factory=EKFConfig)


@dataclass
class FsmConfig:
    """Top-level FSM tuning."""

    control_dt: float = 0.01           # 100 Hz loop
    pos_tol: float = 0.05              # meters — loose enough that base+racket settle in finite time
    ori_tol: float = 0.20              # Frobenius norm of (R_des - R_now); ~10° all-axis
    base_pos_tol: float = 0.05

    # If we lose the ball for this long during TRACK / APPROACH we abort.
    ball_timeout_s: float = 0.40

    # Time before predicted impact at which we commit to SWING.
    swing_commit_time_s: float = 0.20

    # After SWING, the time we hold the follow-through pose before recovering.
    follow_through_hold_s: float = 0.20

    # Hard cap on RECOVER duration. If the racket+base haven't settled within
    # tolerance by then, force-transition back to READY anyway — better to be
    # ready for the next ball than to deadlock chasing sub-cm convergence on a
    # 64 kg base.
    recover_max_s: float = 1.5

    # Verbose state-transition prints.
    verbose: bool = True


@dataclass
class PickleballConfig:
    court: CourtConfig = field(default_factory=CourtConfig)
    racket: RacketConfig = field(default_factory=RacketConfig)
    ready: ReadyPose = field(default_factory=ReadyPose)
    tracker: BallTrackerConfig = field(default_factory=BallTrackerConfig)
    fsm: FsmConfig = field(default_factory=FsmConfig)
