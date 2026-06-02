"""Tracker configuration for a foam pickleball.

Extends BallTrackerConfig with aerodynamic drag parameters.

The foam ball drag model (continuous time):

    dv/dt = [0, 0, -g] - k * ||v|| * v

where k (m^-1) is the only new parameter to tune experimentally. All other
fields are inherited from BallTrackerConfig with the same defaults.

How to tune k
-------------
1. Record several throws with OptiTrack at >= 120 Hz.
2. Use scripts/fit_drag_coefficient.py (or manually sweep) to find the k
   that minimises strike-plane position prediction error on held-out throws.
3. For a 60 mm foam ball at 3–8 m/s, expect k ≈ 0.3–0.6 m^-1.
   Set k = 0.0 to reproduce the zero-drag parabolic baseline.
"""

from __future__ import annotations

from dataclasses import dataclass

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from state_machine.config import BallTrackerConfig


@dataclass
class FoamBallConfig(BallTrackerConfig):
    """BallTrackerConfig extended with foam-ball aerodynamic drag.

    New fields
    ----------
    drag_coefficient : float
        k in ``dv/dt = [0, 0, -g] - k * ||v|| * v``  (units: m^-1).
        Start at 0.0 and increase in small increments (0.05) while
        comparing predicted vs. observed strike-plane positions.

    simulation_dt : float
        Euler integration time step (s) used by _propagate_drag_to_plane.
        1 ms gives sub-mm position error for flights up to 1.5 s at 10 m/s.
        Do not exceed 5 ms.
    """

    # Drag coefficient k (m^-1). Tune from real OptiTrack data.
    drag_coefficient: float = 0.2000

    # Euler integration step (s) for drag propagation.
    simulation_dt: float = 0.001
