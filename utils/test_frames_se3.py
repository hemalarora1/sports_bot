"""Synthetic tests for the SE(3) algebra in frames.py.

Run from the OpenSai root:
    python -m sports_bot.utils.test_frames_se3

Exits non-zero on the first failed assertion. Pure synthetic data — no Redis,
no robot, no OptiTrack. Catches regressions in the math before they hit the
real rig.
"""
from __future__ import annotations

import math
import sys

import numpy as np

from sports_bot.utils.frames import (
    R_from_axis_angle,
    axis_angle_from_R,
    se3_compose,
    se3_inverse,
)


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    omega = rng.normal(size=3)
    omega *= rng.uniform(0.05, 2.5) / max(1e-9, np.linalg.norm(omega))
    return R_from_axis_angle(omega)


def _random_se3(rng: np.random.Generator, t_scale: float = 0.5):
    return (_random_rotation(rng), rng.normal(scale=t_scale, size=3))


def _se3_close(a, b, *, pos_tol: float = 1e-6, rot_tol_rad: float = 1e-6) -> bool:
    Ra, ta = a
    Rb, tb = b
    if float(np.linalg.norm(ta - tb)) > pos_tol:
        return False
    R_err = Ra @ Rb.T
    omega = axis_angle_from_R(R_err)
    return float(np.linalg.norm(omega)) <= rot_tol_rad


def test_axis_angle_roundtrip() -> None:
    rng = np.random.default_rng(0)
    for _ in range(50):
        omega = rng.normal(size=3)
        omega *= rng.uniform(0.0, 2.9) / max(1e-9, np.linalg.norm(omega))
        R = R_from_axis_angle(omega)
        omega_back = axis_angle_from_R(R)
        R_back = R_from_axis_angle(omega_back)
        # ω and -ω are the same rotation when |ω| = π — compare R, not ω.
        assert _se3_close((R, np.zeros(3)), (R_back, np.zeros(3))), \
            f"axis-angle round-trip failed for ω={omega}"
    # Special cases
    assert _se3_close((R_from_axis_angle(np.zeros(3)), np.zeros(3)),
                      (np.eye(3), np.zeros(3)))
    # Exact rotation by π about z
    R = R_from_axis_angle(np.array([0.0, 0.0, math.pi]))
    expected = np.diag([-1.0, -1.0, 1.0])
    assert np.allclose(R, expected, atol=1e-10), f"R_z(π) mismatch:\n{R}"
    omega_back = axis_angle_from_R(R)
    R_back = R_from_axis_angle(omega_back)
    assert _se3_close((R, np.zeros(3)), (R_back, np.zeros(3)))
    print("[ok] axis_angle round-trip (incl. θ → π edge)")


def test_se3_compose_inverse() -> None:
    rng = np.random.default_rng(1)
    identity = (np.eye(3), np.zeros(3))
    for _ in range(50):
        T = _random_se3(rng)
        T_inv = se3_inverse(T)
        assert _se3_close(se3_compose(T, T_inv), identity), "T ⊕ T⁻¹ ≠ I"
        assert _se3_close(se3_compose(T_inv, T), identity), "T⁻¹ ⊕ T ≠ I"
    print("[ok] se3 compose / inverse")


def main() -> int:
    test_axis_angle_roundtrip()
    test_se3_compose_inverse()
    print("\nall tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
