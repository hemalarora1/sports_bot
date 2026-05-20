"""Plan the racket pose / swing trajectory for a predicted intercept.

Given an intercept point P_impact in world frame and a desired return target
P_target, the planner produces:

  * SwingPlan.base_pose     : world [x, y, theta] for the mobile base.
  * SwingPlan.wind_up_pose  : (R, p) for the racket sweet-spot before swinging.
  * SwingPlan.strike_pose   : (R, p) for the racket sweet-spot at impact.
  * SwingPlan.follow_pose   : (R, p) for the racket sweet-spot after impact.
  * SwingPlan.strike_velocity: desired racket linear velocity at impact.

All racket poses use the convention that the rotation matrix's columns are
[face_right, face_up, face_normal] in world frame, i.e. the racket face normal
is the +Z column.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .ball_tracker import Intercept
from .config import CourtConfig, RacketConfig


@dataclass
class SwingPlan:
    base_pose: np.ndarray         # [x, y, theta] in world frame
    wind_up_position: np.ndarray  # 3-vector
    wind_up_orientation: np.ndarray  # 3x3
    strike_position: np.ndarray   # 3-vector
    strike_orientation: np.ndarray  # 3x3
    follow_position: np.ndarray   # 3-vector
    follow_orientation: np.ndarray  # 3x3
    strike_velocity: np.ndarray   # 3-vector world-frame linear velocity at impact
    time_to_impact: float


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-9:
        raise ValueError("Cannot normalize a zero vector")
    return v / n


def _rotation_from_normal(face_normal: np.ndarray, world_up: Optional[np.ndarray] = None) -> np.ndarray:
    """Build a 3x3 rotation whose columns are [face_right, face_up, face_normal]
    such that the third column is `face_normal` and the second column is as
    close as possible to `world_up`."""
    n = _normalize(face_normal)
    if world_up is None:
        world_up = np.array([0.0, 1.0, 0.0])
    # Project world_up onto the plane orthogonal to n.
    up_proj = world_up - np.dot(world_up, n) * n
    if np.linalg.norm(up_proj) < 1e-6:
        # Degenerate: fall back to using world +X as a reference.
        ref = np.array([1.0, 0.0, 0.0])
        up_proj = ref - np.dot(ref, n) * n
    face_up = _normalize(up_proj)
    face_right = np.cross(face_up, n)
    R = np.column_stack([face_right, face_up, n])
    return R


class SwingPlanner:
    def __init__(self, court: CourtConfig, racket: RacketConfig):
        self._court = court
        self._racket = racket

    # -------------------------------------------------------------- helpers

    def _is_strike_reachable(self, p_strike: np.ndarray) -> bool:
        if not (self._court.strike_z_min <= p_strike[2] <= self._court.strike_z_max):
            return False
        if not (self._court.base_y_min - 0.5 <= p_strike[1] <= self._court.base_y_max + 0.5):
            return False
        return True

    def _base_pose_for_strike(self, p_strike: np.ndarray) -> np.ndarray:
        """Place the base directly behind the strike point, clamped to the court."""
        x = float(np.clip(p_strike[0] - self._court.strike_plane_x, self._court.base_x_min, self._court.base_x_max))
        y = float(np.clip(p_strike[1], self._court.base_y_min, self._court.base_y_max))
        # Yaw the base toward the strike point (and the opponent).
        theta = 0.0  # base facing +X is fine for a strike plane parallel to YZ
        return np.array([x, y, theta])

    # ----------------------------------------------------------- main API

    def plan(self, intercept: Intercept) -> Optional[SwingPlan]:
        p_strike = intercept.position.copy()
        if not self._is_strike_reachable(p_strike):
            return None

        # Racket face normal points from the strike point toward the return target.
        target_dir = self._court.return_target_xyz - p_strike
        # Drop most of the vertical component so the face stays mostly upright;
        # keep a small upward bias so the ball clears the net.
        net_clear_dir = target_dir.copy()
        net_clear_dir[2] = max(net_clear_dir[2], 0.2 * np.linalg.norm(target_dir[:2]))
        try:
            face_normal = _normalize(net_clear_dir)
        except ValueError:
            face_normal = np.array([1.0, 0.0, 0.0])

        R_strike = _rotation_from_normal(face_normal)

        # Wind-up: pull the racket back along -face_normal by wind_up_offset.
        wind_up_pos = p_strike - self._racket.wind_up_offset * face_normal
        # Follow-through: push along +face_normal.
        follow_pos = p_strike + self._racket.follow_through_offset * face_normal

        # Desired racket velocity at impact: along +face_normal at impact_speed.
        strike_velocity = self._racket.impact_speed * face_normal

        # Base sits directly behind the strike point.
        base_pose = self._base_pose_for_strike(p_strike)

        return SwingPlan(
            base_pose=base_pose,
            wind_up_position=wind_up_pos,
            wind_up_orientation=R_strike,
            strike_position=p_strike,
            strike_orientation=R_strike,
            follow_position=follow_pos,
            follow_orientation=R_strike,
            strike_velocity=strike_velocity,
            time_to_impact=intercept.time_to_impact,
        )
