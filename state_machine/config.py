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

    # Racket sweet-spot position in world frame.
    racket_position: np.ndarray = field(
        default_factory=lambda: np.array([0.45, 0.0, 0.95])
    )

    # Racket orientation as a 3x3 rotation. Default: racket face normal points
    # toward the opponent (+X world), face vertical so the up-direction on the
    # face is world up (+Z), face-right is +Y world.
    # Columns are [face_right, face_up, face_normal]_world.
    racket_orientation: np.ndarray = field(
        default_factory=lambda: np.array([
            # face_right  face_up  face_normal
            [    0.0,       0.0,      1.0   ],   # row 0 (world X)
            [    1.0,       0.0,      0.0   ],   # row 1 (world Y)
            [    0.0,       1.0,      0.0   ],   # row 2 (world Z)
        ])
    )


@dataclass
class BallTrackerConfig:
    """Parameters for ball trajectory estimation."""

    gravity: float = 9.81

    # Sliding window size (samples) used to fit the ballistic trajectory.
    history_size: int = 12

    # Reject samples older than this when fitting.
    history_max_age_s: float = 0.30

    # Minimum forward speed (m/s, world -X direction) for a ball to be
    # considered "incoming". Below this we ignore it.
    min_incoming_speed: float = 0.5

    # Reject jumps in position larger than this between consecutive samples
    # (m). Filters out OptiTrack glitches / dropouts.
    max_position_jump: float = 0.5

    # Time horizon (s) within which a predicted intercept is considered usable.
    max_lookahead: float = 1.5
    min_lookahead: float = 0.05


@dataclass
class FsmConfig:
    """Top-level FSM tuning."""

    control_dt: float = 0.01           # 100 Hz loop
    pos_tol: float = 0.02              # meters
    ori_tol: float = 0.05              # Frobenius norm of (R_des - R_now)
    base_pos_tol: float = 0.03

    # If we lose the ball for this long during TRACK / APPROACH we abort.
    ball_timeout_s: float = 0.40

    # Time before predicted impact at which we commit to SWING.
    swing_commit_time_s: float = 0.20

    # After SWING, the time we hold the follow-through pose before recovering.
    follow_through_hold_s: float = 0.20

    # Verbose state-transition prints.
    verbose: bool = True


@dataclass
class PickleballConfig:
    court: CourtConfig = field(default_factory=CourtConfig)
    racket: RacketConfig = field(default_factory=RacketConfig)
    ready: ReadyPose = field(default_factory=ReadyPose)
    tracker: BallTrackerConfig = field(default_factory=BallTrackerConfig)
    fsm: FsmConfig = field(default_factory=FsmConfig)
