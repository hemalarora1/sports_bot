"""Synthetic tests for the SE(3) algebra + hand-eye solver in frames.py.

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
    se3_from_xi,
    se3_identity,
    se3_inverse,
    se3_to_xi,
    solve_hand_eye_se3,
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
    for _ in range(50):
        T = _random_se3(rng)
        T_inv = se3_inverse(T)
        TT_inv = se3_compose(T, T_inv)
        T_invT = se3_compose(T_inv, T)
        assert _se3_close(TT_inv, se3_identity()), "T ⊕ T⁻¹ ≠ I"
        assert _se3_close(T_invT, se3_identity()), "T⁻¹ ⊕ T ≠ I"
    # xi packing round-trip
    for _ in range(50):
        T = _random_se3(rng)
        T_back = se3_from_xi(se3_to_xi(T))
        assert _se3_close(T, T_back), "se3 ↔ xi round-trip failed"
    print("[ok] se3 compose / inverse / xi round-trip")


def test_hand_eye_recovers_ground_truth_noise_free() -> None:
    """Generate N synthetic samples from known (T_W_A, T_E_P, joint poses T_A_E)
    and verify the solver recovers the unknowns to machine precision."""
    rng = np.random.default_rng(42)

    # Realistic-magnitude ground truth.
    T_W_A_true = (R_from_axis_angle(np.array([0.0, 0.0, 0.3])),
                  np.array([0.4, -0.1, 0.55]))
    T_E_P_true = (R_from_axis_angle(np.array([0.0, math.pi / 2, 0.0])),
                  np.array([0.02, 0.0, 0.38]))

    # 10 diverse T_A_E samples (the "arm visits these EE poses in its own base frame").
    samples = []
    for _ in range(10):
        T_A_E = _random_se3(rng, t_scale=0.4)
        T_W_P = se3_compose(se3_compose(T_W_A_true, T_A_E), T_E_P_true)
        samples.append((T_A_E, T_W_P))

    T_W_A_est, T_E_P_est, pos_rms, ori_rms_deg = solve_hand_eye_se3(samples)

    assert pos_rms < 1e-8, f"pos_rms {pos_rms:.3e} too large for noise-free data"
    assert ori_rms_deg < 1e-5, f"ori_rms {ori_rms_deg:.3e} deg too large"
    assert _se3_close(T_W_A_est, T_W_A_true, pos_tol=1e-6, rot_tol_rad=1e-6), \
        "solver did not recover T_W_A"
    assert _se3_close(T_E_P_est, T_E_P_true, pos_tol=1e-6, rot_tol_rad=1e-6), \
        "solver did not recover T_E_P"
    print(f"[ok] hand-eye recovers noise-free ground truth "
          f"(pos_rms={pos_rms:.2e} m, ori_rms={ori_rms_deg:.2e} deg)")


def test_hand_eye_robust_to_noise() -> None:
    """Add OptiTrack-scale noise (1 mm position, 0.1° orientation) to the
    measured T_W_P at each sample, plus 0.5 mm / 0.05° on T_A_E (Franka FK +
    quantization), and check the solver still recovers within sensor floor."""
    rng = np.random.default_rng(7)

    T_W_A_true = (R_from_axis_angle(np.array([0.05, -0.03, 0.4])),
                  np.array([0.42, -0.08, 0.56]))
    T_E_P_true = (R_from_axis_angle(np.array([0.01, 1.55, 0.02])),
                  np.array([0.005, -0.003, 0.382]))

    pos_noise_W = 1e-3   # 1 mm OT
    ori_noise_W = math.radians(0.1)
    pos_noise_A = 5e-4   # 0.5 mm Franka FK
    ori_noise_A = math.radians(0.05)

    samples = []
    for _ in range(12):
        T_A_E = _random_se3(rng, t_scale=0.4)
        T_W_P = se3_compose(se3_compose(T_W_A_true, T_A_E), T_E_P_true)
        # Perturb the measured T_A_E
        T_A_E_meas = se3_compose(
            T_A_E,
            (R_from_axis_angle(rng.normal(scale=ori_noise_A, size=3)),
             rng.normal(scale=pos_noise_A, size=3)),
        )
        # Perturb the measured T_W_P
        T_W_P_meas = se3_compose(
            T_W_P,
            (R_from_axis_angle(rng.normal(scale=ori_noise_W, size=3)),
             rng.normal(scale=pos_noise_W, size=3)),
        )
        samples.append((T_A_E_meas, T_W_P_meas))

    T_W_A_est, T_E_P_est, pos_rms, ori_rms_deg = solve_hand_eye_se3(samples)
    # With sensor-scale noise on 12 well-spread samples, recovered transforms
    # should be within a few mm / 0.3° of ground truth.
    err_W_A_pos = float(np.linalg.norm(T_W_A_est[1] - T_W_A_true[1]))
    err_E_P_pos = float(np.linalg.norm(T_E_P_est[1] - T_E_P_true[1]))
    err_W_A_rot_deg = math.degrees(float(np.linalg.norm(
        axis_angle_from_R(T_W_A_est[0] @ T_W_A_true[0].T))))
    err_E_P_rot_deg = math.degrees(float(np.linalg.norm(
        axis_angle_from_R(T_E_P_est[0] @ T_E_P_true[0].T))))
    assert err_W_A_pos < 0.005, f"T_W_A pos error {err_W_A_pos*1000:.2f} mm > 5 mm"
    assert err_E_P_pos < 0.005, f"T_E_P pos error {err_E_P_pos*1000:.2f} mm > 5 mm"
    assert err_W_A_rot_deg < 0.5, f"T_W_A rot error {err_W_A_rot_deg:.3f}° > 0.5°"
    assert err_E_P_rot_deg < 0.5, f"T_E_P rot error {err_E_P_rot_deg:.3f}° > 0.5°"
    print(
        f"[ok] hand-eye robust to sensor noise "
        f"(T_W_A err {err_W_A_pos*1000:.2f} mm / {err_W_A_rot_deg:.3f}°, "
        f"T_E_P err {err_E_P_pos*1000:.2f} mm / {err_E_P_rot_deg:.3f}°, "
        f"residual {pos_rms*1000:.2f} mm / {ori_rms_deg:.3f}°)"
    )


def main() -> int:
    test_axis_angle_roundtrip()
    test_se3_compose_inverse()
    test_hand_eye_recovers_ground_truth_noise_free()
    test_hand_eye_robust_to_noise()
    print("\nall tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
