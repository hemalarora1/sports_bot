"""SE(2) frame helpers for the sports_bot calibration pipeline.

Frame nomenclature (also see sports_bot/context.md and base_bridge.py):

    W       world frame — floor-tape origin, +X forward, +Y left, +Z up. The
            frame the FSM and ball tracker speak.
    M       Motive streaming "room" frame — Z-up (Motive's *streaming* up axis;
            independent of the Y-up display axis the UI shows — see
            context.md). Origin / yaw are whatever Motive picked.
    B       OptiTrack rigid-body local frame for the cart. Glued to the
            markers. Z-up like M (rigid bodies inherit the streaming up axis
            convention).
    R       Robot odometry origin. Floor-fixed for the duration of the
            TidyBot driver process. Re-anchors every driver restart.
    C       Cart "control point" frame — the pivot the TidyBot firmware
            tracks in `hb1::current_pose`. Glued to the cart.

Static across the rig:
    T_W_M     world calibration                  (world_calibration.json)
    T_B_C     marker-mounting offset             (robot_marker_calibration.json)

Live, streamed every frame:
    T_W_B(t)  from OptiTrack via R_W_M, T_W_M (position) and a horizontal-axis
              projection out of the body rotation (yaw)
    T_R_C(t)  from hb1::current_pose

Per-session, derived at bringup:
    T_W_R = T_W_B(snapshot) * T_B_C * T_R_C(snapshot)^-1
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np


SE2 = Tuple[float, float, float]   # (x, y, theta)
SE3 = Tuple[np.ndarray, np.ndarray]   # (R 3x3, t 3-vec)


# ---------- Quaternion / rotation matrix --------------------------------------

def quat_to_R(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Quaternion (qx, qy, qz, qw) → 3x3 rotation matrix.

    Convention: returned R maps vectors from body coords to the frame in
    which the quaternion is expressed (e.g. Motive room for raw streams,
    world for post-calibration streams).
    """
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if n < 1e-12:
        return np.eye(3)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz
    return np.array([
        [1 - 2 * (yy + zz), 2 * (xy - wz),     2 * (xz + wy)],
        [2 * (xy + wz),     1 - 2 * (xx + zz), 2 * (yz - wx)],
        [2 * (xz - wy),     2 * (yz + wx),     1 - 2 * (xx + yy)],
    ])


def R_to_quat(R: np.ndarray) -> Tuple[float, float, float, float]:
    """3x3 rotation matrix → quaternion (qx, qy, qz, qw). Shepperd's method:
    pick the largest of (1+tr, 1+2R00-tr, 1+2R11-tr, 1+2R22-tr) for numerical
    stability."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = 2.0 * math.sqrt(tr + 1.0)
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    return float(qx), float(qy), float(qz), float(qw)


def rotate_quat(R_target_source: np.ndarray, quat_source: Sequence[float]) -> Tuple[float, float, float, float]:
    """Rotate a quaternion by a fixed rotation matrix. If `quat_source` expresses
    a body's orientation in frame S, and R_target_source maps S → T, this
    returns the same body's orientation in frame T."""
    R_S_B = quat_to_R(quat_source[0], quat_source[1], quat_source[2], quat_source[3])
    R_T_B = R_target_source @ R_S_B
    return R_to_quat(R_T_B)


# ---------- World calibration loader ------------------------------------------

@dataclass
class WorldCalibration:
    R_W_M: np.ndarray   # 3x3, world ← Motive room
    t_W_M: np.ndarray   # 3-vector
    yaw_W_M: float      # planar yaw of R_W_M about Z (radians)

    @classmethod
    def load(cls, path: str) -> "WorldCalibration":
        if not os.path.isfile(path):
            print(f"[frames] no world calibration at {path}, using identity")
            return cls(np.eye(3), np.zeros(3), 0.0)
        with open(path, "r") as f:
            data = json.load(f)
        R = np.asarray(data.get("rotation", np.eye(3).tolist()), dtype=float)
        t = np.asarray(data.get("translation", [0.0, 0.0, 0.0]), dtype=float)
        if R.shape != (3, 3):
            raise ValueError(f"world calibration rotation has shape {R.shape}, expected (3, 3)")
        if t.shape != (3,):
            raise ValueError(f"world calibration translation has shape {t.shape}, expected (3,)")
        yaw = math.atan2(R[1, 0], R[0, 0])
        return cls(R, t, yaw)

    def world_pos_from_raw(self, raw_pos_M: Sequence[float]) -> np.ndarray:
        return self.R_W_M @ np.asarray(raw_pos_M, dtype=float) + self.t_W_M


# ---------- Y-up body → world-yaw extraction ----------------------------------

def yaw_from_quat(qx: float, qy: float, qz: float, qw: float) -> float:
    """Yaw of the body's +X axis projected into the XY plane of whatever frame
    the quaternion expresses.

    Body B is Z-up (Motive's streaming up-axis convention), so B's +X is a
    horizontal axis in B-local coords. We define "cart yaw" as the angle of
    B's +X axis projected into the XY plane of the surrounding frame.
    Whatever angular offset exists between B's +X and the cart's actual
    forward driving direction is absorbed into T_B_C — we just need *a*
    consistent yaw definition.

    Robust to small marker tilt (markers not perfectly level on the cart):
    the projection works for any cart attitude, where the textbook
    `atan2(2(wz + xy), 1 - 2(yy + zz))` formula assumes pure-Z rotation and
    silently absorbs tilt as bogus yaw.
    """
    R = quat_to_R(qx, qy, qz, qw)
    return math.atan2(R[1, 0], R[0, 0])


# ---------- SE(2) algebra ------------------------------------------------------

def wrap_angle(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def se2_compose(a: SE2, b: SE2) -> SE2:
    """Pose composition: T_a ⊕ T_b. Returns the SE(2) element representing
    'first apply T_a, then T_b' — i.e. if T_a is X-in-Y and T_b is Z-in-X,
    the result is Z-in-Y."""
    ax, ay, ath = a
    bx, by, bth = b
    c, s = math.cos(ath), math.sin(ath)
    return (ax + c * bx - s * by, ay + s * bx + c * by, wrap_angle(ath + bth))


def se2_inverse(a: SE2) -> SE2:
    ax, ay, ath = a
    c, s = math.cos(ath), math.sin(ath)
    return (-c * ax - s * ay, s * ax - c * ay, wrap_angle(-ath))


def se2_apply_point(T: SE2, p: Sequence[float]) -> Tuple[float, float]:
    ax, ay, ath = T
    px, py = p
    c, s = math.cos(ath), math.sin(ath)
    return (ax + c * px - s * py, ay + s * px + c * py)


def se2_apply_pose(T: SE2, pose: SE2) -> SE2:
    return se2_compose(T, pose)


# ---------- Hand-eye 2D solver -------------------------------------------------

def solve_hand_eye_se2(
    samples: Sequence[Tuple[SE2, SE2]],
) -> Tuple[SE2, SE2, float, float]:
    """Solve T_B_C and T_W_R jointly from a list of pose pairs (T_W_B_i, T_R_C_i).

    The constraint at every sample is

        T_W_B_i  ⊕  T_B_C  =  T_W_R  ⊕  T_R_C_i

    (same point on the cart, two ways to express its world pose). Stacks 3·N
    residuals — three per sample (x, y, yaw with angle-wrap) — and minimises
    via Levenberg-Marquardt over the 6 unknowns (T_B_C, T_W_R).

    Returns: (T_B_C, T_W_R, pos_rms_metres, yaw_rms_degrees).

    Requires ≥2 samples with reasonable rotational and translational diversity.
    With only translation between samples, T_B_C's rotation is undetermined.
    """
    try:
        from scipy.optimize import least_squares
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "solve_hand_eye_se2 needs scipy.optimize. "
            "Install scipy or activate the opensai conda env."
        ) from exc

    if len(samples) < 2:
        raise ValueError(f"need ≥2 samples, got {len(samples)}")

    def residuals(params: np.ndarray) -> np.ndarray:
        T_B_C = (float(params[0]), float(params[1]), float(params[2]))
        T_W_R = (float(params[3]), float(params[4]), float(params[5]))
        out = np.empty(3 * len(samples))
        for i, (T_W_B, T_R_C) in enumerate(samples):
            lhs = se2_compose(T_W_B, T_B_C)
            rhs = se2_compose(T_W_R, T_R_C)
            out[3 * i + 0] = lhs[0] - rhs[0]
            out[3 * i + 1] = lhs[1] - rhs[1]
            out[3 * i + 2] = wrap_angle(lhs[2] - rhs[2])
        return out

    T_W_B_0 = samples[0][0]
    T_R_C_0 = samples[0][1]
    # Initial guess: T_B_C = identity, T_W_R = T_W_B_0 ⊕ T_R_C_0^-1.
    # This is exactly the back-solve we'd do with T_B_C known to be identity —
    # a good warm start whether or not the markers actually sit at the cart's
    # control point.
    T_W_R_init = se2_compose(T_W_B_0, se2_inverse(T_R_C_0))
    x0 = np.array([0.0, 0.0, 0.0, T_W_R_init[0], T_W_R_init[1], T_W_R_init[2]])

    result = least_squares(residuals, x0, method="lm", max_nfev=200)
    T_B_C = (float(result.x[0]), float(result.x[1]), wrap_angle(float(result.x[2])))
    T_W_R = (float(result.x[3]), float(result.x[4]), wrap_angle(float(result.x[5])))

    r = result.fun.reshape(-1, 3)
    pos_rms = float(np.sqrt(np.mean(r[:, :2] ** 2)))
    yaw_rms_deg = float(np.degrees(np.sqrt(np.mean(r[:, 2] ** 2))))
    return T_B_C, T_W_R, pos_rms, yaw_rms_deg


# ---------- Robot-marker calibration file I/O ---------------------------------

def robot_marker_calibration_path() -> str:
    """Conventional location next to world_calibration.json."""
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "optitrack",
        "robot_marker_calibration.json",
    )


def save_robot_marker_calibration(
    path: str,
    T_B_C: SE2,
    metadata: Optional[dict] = None,
) -> None:
    payload = {
        "T_B_C": {
            "x": T_B_C[0],
            "y": T_B_C[1],
            "theta_rad": T_B_C[2],
            "theta_deg": math.degrees(T_B_C[2]),
        },
        "comment": (
            "Static SE(2) offset between the OptiTrack rigid-body local frame "
            "(B, Y-up, glued to the cart markers) and the TidyBot odometry "
            "control-point frame (C, glued to the cart at wherever the firmware "
            "tracks). Solved via 2D hand-eye (AX=XB) from drive-around pose pairs. "
            "Re-solve only when markers are re-stuck or bumped."
        ),
    }
    if metadata:
        payload.update(metadata)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def load_robot_marker_calibration(path: str) -> SE2:
    with open(path, "r") as f:
        data = json.load(f)
    t = data["T_B_C"]
    return (float(t["x"]), float(t["y"]), float(t["theta_rad"]))


# ---------- Angle-aware averaging ---------------------------------------------

def average_angles(angles: Sequence[float]) -> float:
    """Circular mean — atan2 of the summed unit vectors. Robust across the
    ±π wrap."""
    s = sum(math.sin(a) for a in angles)
    c = sum(math.cos(a) for a in angles)
    return math.atan2(s, c)


def average_quats(quats: Sequence[Sequence[float]]) -> Tuple[float, float, float, float]:
    """Hemisphere-aligned mean of a list of (qx, qy, qz, qw) quaternions, then
    renormalized. Accurate for small angular spread (< few degrees) — which is
    the regime when averaging a held-still OptiTrack capture. Returns the mean
    quaternion as (qx, qy, qz, qw)."""
    arr = np.asarray(quats, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 4 or arr.shape[0] == 0:
        raise ValueError(f"average_quats expects (N, 4) input, got {arr.shape}")
    ref = arr[0].copy()
    # Flip any quat in the opposite hemisphere so the linear mean is meaningful.
    for i in range(1, arr.shape[0]):
        if float(np.dot(arr[i], ref)) < 0.0:
            arr[i] = -arr[i]
    mean = np.mean(arr, axis=0)
    n = float(np.linalg.norm(mean))
    if n < 1e-12:
        # Antipodal samples cancelled out — fall back to the first sample.
        return float(ref[0]), float(ref[1]), float(ref[2]), float(ref[3])
    mean = mean / n
    return float(mean[0]), float(mean[1]), float(mean[2]), float(mean[3])


# ---------- SE(3) algebra ------------------------------------------------------

def _skew(v: np.ndarray) -> np.ndarray:
    """3-vec → 3x3 skew-symmetric matrix."""
    return np.array([
        [0.0,   -v[2],  v[1]],
        [v[2],   0.0,  -v[0]],
        [-v[1],  v[0],  0.0],
    ])


def R_from_axis_angle(omega: np.ndarray) -> np.ndarray:
    """Rodrigues formula. `omega` is a 3-vector with norm = rotation angle (rad)
    and direction = rotation axis. Returns the 3x3 rotation matrix."""
    omega = np.asarray(omega, dtype=float)
    theta = float(np.linalg.norm(omega))
    if theta < 1e-12:
        # First-order: R ≈ I + [omega]_×
        return np.eye(3) + _skew(omega)
    k = omega / theta
    K = _skew(k)
    return np.eye(3) + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)


def axis_angle_from_R(R: np.ndarray) -> np.ndarray:
    """Inverse of R_from_axis_angle. Returns the 3-vector ω such that
    `R_from_axis_angle(ω) == R`. Branch: ||ω|| ∈ [0, π]."""
    R = np.asarray(R, dtype=float)
    tr = float(R[0, 0] + R[1, 1] + R[2, 2])
    c = max(-1.0, min(1.0, (tr - 1.0) / 2.0))
    theta = math.acos(c)
    if theta < 1e-9:
        # Near identity — use the skew-symmetric part directly.
        return 0.5 * np.array([R[2, 1] - R[1, 2],
                               R[0, 2] - R[2, 0],
                               R[1, 0] - R[0, 1]])
    if abs(theta - math.pi) < 1e-6:
        # Rotation by π — sin θ = 0, the formula below blows up. Use the
        # symmetric part M = (R + I) / 2 = k k^T; pick the column with largest
        # diagonal to dodge sign-cancellation.
        M = 0.5 * (R + np.eye(3))
        diag = np.array([M[0, 0], M[1, 1], M[2, 2]])
        i = int(np.argmax(diag))
        if diag[i] < 0.0:
            # Numerically degenerate; fall back to identity.
            return np.zeros(3)
        k = M[:, i] / math.sqrt(diag[i])
        # Sign of k is ambiguous; either branch is a valid axis-angle.
        return theta * k
    coeff = theta / (2.0 * math.sin(theta))
    return coeff * np.array([R[2, 1] - R[1, 2],
                             R[0, 2] - R[2, 0],
                             R[1, 0] - R[0, 1]])


def se3_identity() -> SE3:
    return (np.eye(3), np.zeros(3))


def se3_compose(a: SE3, b: SE3) -> SE3:
    """Pose composition: T_a ⊕ T_b. Same semantics as se2_compose — if
    T_a is X-in-Y and T_b is Z-in-X, the result is Z-in-Y."""
    Ra, ta = a
    Rb, tb = b
    return (Ra @ Rb, Ra @ tb + ta)


def se3_inverse(a: SE3) -> SE3:
    R, t = a
    Rt = R.T
    return (Rt, -Rt @ t)


def se3_apply_point(T: SE3, p: Sequence[float]) -> np.ndarray:
    R, t = T
    return R @ np.asarray(p, dtype=float) + t


def se3_from_xi(xi: np.ndarray) -> SE3:
    """Pack a 6-vector (tx, ty, tz, ωx, ωy, ωz) into an SE(3) element. Uses
    the *decoupled* parameterization t-separate-from-axis-angle (NOT the
    matrix exponential / twist). This is the parameterization the LM solver
    uses; matches `se3_to_xi`."""
    t = np.asarray(xi[:3], dtype=float)
    R = R_from_axis_angle(np.asarray(xi[3:6], dtype=float))
    return (R, t)


def se3_to_xi(T: SE3) -> np.ndarray:
    """Inverse of se3_from_xi: SE(3) → 6-vec (tx, ty, tz, ωx, ωy, ωz)."""
    R, t = T
    omega = axis_angle_from_R(R)
    return np.concatenate([np.asarray(t, dtype=float), omega])


def se3_from_se2(T2: SE2, z: float = 0.0) -> SE3:
    """Lift an SE(2) element (x, y, theta) to SE(3) at height `z`, with
    rotation about world +Z. Used to compose the SE(2) cart-frame transforms
    (T_B_C, T_W_C) with full-3D arm transforms."""
    x, y, th = T2
    c, s = math.cos(th), math.sin(th)
    R = np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ])
    t = np.array([x, y, z])
    return (R, t)


# ---------- Hand-eye 3D solver -------------------------------------------------

def solve_hand_eye_se3(
    samples: Sequence[Tuple[SE3, SE3]],
    *,
    initial: Optional[Tuple[SE3, SE3]] = None,
) -> Tuple[SE3, SE3, float, float]:
    """Solve T_W_A and T_E_P jointly from a list of pose pairs (T_A_E_i, T_W_P_i).

    The constraint at every sample is

        T_W_A  ⊕  T_A_E_i  ⊕  T_E_P  =  T_W_P_i

    Stacks 6·N residuals (3 position + 3 axis-angle per sample) and minimises
    via Levenberg-Marquardt over the 12 unknowns (T_W_A, T_E_P), each
    parameterized as (tx, ty, tz, ωx, ωy, ωz) per se3_from_xi.

    `samples[i] = (T_A_E_i, T_W_P_i)`.

    Returns: (T_W_A, T_E_P, pos_rms_m, ori_rms_deg).

    Requires ≥3 samples with reasonable rotational *and* translational
    diversity. Pure translation across samples leaves T_E_P's rotation
    underdetermined (the racket rolls with the EE consistently in every
    sample); pure rotation underdetermines its translation.
    """
    try:
        from scipy.optimize import least_squares
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "solve_hand_eye_se3 needs scipy.optimize. "
            "Install scipy or activate the opensai conda env."
        ) from exc

    if len(samples) < 3:
        raise ValueError(f"need ≥3 samples, got {len(samples)}")

    def residuals(params: np.ndarray) -> np.ndarray:
        T_W_A = se3_from_xi(params[:6])
        T_E_P = se3_from_xi(params[6:12])
        out = np.empty(6 * len(samples))
        for i, (T_A_E, T_W_P_meas) in enumerate(samples):
            T_W_P_pred = se3_compose(se3_compose(T_W_A, T_A_E), T_E_P)
            R_pred, t_pred = T_W_P_pred
            R_meas, t_meas = T_W_P_meas
            out[6 * i + 0:6 * i + 3] = t_pred - t_meas
            # Rotation residual: axis-angle of R_err = R_pred · R_meas⁻¹,
            # which is near identity when the prediction matches.
            R_err = R_pred @ R_meas.T
            out[6 * i + 3:6 * i + 6] = axis_angle_from_R(R_err)
        return out

    # Initial guess: assume T_E_P = identity and back-solve T_W_A from sample 0.
    # If the racket frame happens to be near-identity in EE (as it should be
    # when we set the Motive pivot at the sweet spot with +Z = face normal),
    # this is already very close.
    if initial is not None:
        T_W_A_init, T_E_P_init = initial
    else:
        T_A_E_0, T_W_P_0 = samples[0]
        T_W_A_init = se3_compose(T_W_P_0, se3_inverse(T_A_E_0))
        T_E_P_init = se3_identity()
    x0 = np.concatenate([se3_to_xi(T_W_A_init), se3_to_xi(T_E_P_init)])

    result = least_squares(residuals, x0, method="lm", max_nfev=400)
    T_W_A = se3_from_xi(result.x[:6])
    T_E_P = se3_from_xi(result.x[6:12])

    r = result.fun.reshape(-1, 6)
    pos_rms = float(np.sqrt(np.mean(r[:, :3] ** 2)))
    ori_rms_deg = float(np.degrees(np.sqrt(np.mean(r[:, 3:] ** 2))))
    return T_W_A, T_E_P, pos_rms, ori_rms_deg


# ---------- Arm calibration file I/O ------------------------------------------

def arm_calibration_path() -> str:
    """Conventional location next to robot_marker_calibration.json."""
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "optitrack",
        "arm_calibration.json",
    )


def _se3_to_dict(T: SE3) -> dict:
    R, t = T
    omega = axis_angle_from_R(R)
    return {
        "translation_m": [float(t[0]), float(t[1]), float(t[2])],
        "rotation_matrix": [[float(v) for v in row] for row in R],
        "axis_angle_rad": [float(omega[0]), float(omega[1]), float(omega[2])],
    }


def _se3_from_dict(d: dict) -> SE3:
    t = np.asarray(d["translation_m"], dtype=float)
    if "rotation_matrix" in d:
        R = np.asarray(d["rotation_matrix"], dtype=float)
    else:
        R = R_from_axis_angle(np.asarray(d["axis_angle_rad"], dtype=float))
    return (R, t)


def save_arm_calibration(
    path: str,
    T_C_A: SE3,
    T_E_P: SE3,
    metadata: Optional[dict] = None,
) -> None:
    """Persist the two static arm transforms.

    T_C_A : SE(3) offset from cart odom control-point frame C to Franka arm
            base frame A. Re-solve when the arm gets re-mounted on the cart.
    T_E_P : SE(3) offset from Franka end-effector frame E (whatever OpenSai's
            cartesian_task reports) to the racket sweet-spot frame P
            (origin = paddle sweet spot, +Z = face normal, per SwingPlanner
            convention). Re-solve when the racket gets re-mounted or the
            paddle's OptiTrack rigid body is rebuilt.
    """
    payload = {
        "T_C_A": _se3_to_dict(T_C_A),
        "T_E_P": _se3_to_dict(T_E_P),
        "comment": (
            "Static arm-mount + racket-mount transforms for the sports_bot rig. "
            "T_C_A = cart odometry control point C → Franka arm base A. "
            "T_E_P = Franka EE flange E → racket sweet-spot P (with +Z_P = "
            "paddle face normal, matching SwingPlanner convention). Both are "
            "solved jointly via SE(3) hand-eye in "
            "scripts/calibrate_arm_to_cart.py from N (T_A_E, T_W_P) snapshots "
            "captured while the cart sits still and the arm visits a tour of "
            "joint waypoints. The script back-solves T_C_A from T_W_A using "
            "T_W_C = T_W_B ⊕ T_B_C at calibration time."
        ),
    }
    if metadata:
        payload.update(metadata)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def load_arm_calibration(path: str) -> Tuple[SE3, SE3]:
    """Returns (T_C_A, T_E_P)."""
    with open(path, "r") as f:
        data = json.load(f)
    return _se3_from_dict(data["T_C_A"]), _se3_from_dict(data["T_E_P"])


# ---------- Marker-based arm calibration (lightweight alternative) ------------
#
# Instead of solving for T_C_A via SE(3) hand-eye, we exploit the fact that the
# cart has labeled OptiTrack markers in known positions relative to the Franka
# arm base. A subset of those markers (4 by default) is averaged to give the
# arm base TRANSLATION in world; the cart rigid body's orientation is reused
# (with an optional constant correction) for the arm base ORIENTATION. T_E_P
# (flange → racket sweet spot) is supplied by mechanical measurement.
#
# Schema persisted to `sports_bot/optitrack/arm_marker_calibration.json`:
#
#   {
#     "marker_specs":   [[model_id, marker_id], ...],   # which markers define the arm base
#     "cart_rigid_body_id": <int>,                       # for the orientation lookup
#     "R_cart_to_arm":  [[3x3 rotation]],                # constant cart-rb-frame → arm-base-frame
#     "T_E_P":          {translation_m, rotation_matrix} # measured / CAD'd, racket sweet spot
#   }

MarkerSpec = Tuple[int, int]   # (model_id, marker_id) — as decoded by NatNetClient


def marker_world_position_key(model_id: int, marker_id: int) -> str:
    """Redis key under which StreamDataSkeleton.py publishes a labeled
    marker's calibrated world-frame [x, y, z] position."""
    return f"sai2::optitrack::marker_pos::{int(model_id)}::{int(marker_id)}"


def read_marker_position_W(
    r,
    spec: MarkerSpec,
) -> Optional[np.ndarray]:
    """Live world-frame position of a single labeled marker, or None if the
    key is missing / malformed (occluded, marker not in the current frame)."""
    raw = r.get(marker_world_position_key(spec[0], spec[1]))
    if raw is None:
        return None
    try:
        p = json.loads(raw)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    if len(p) != 3:
        return None
    return np.array([float(p[0]), float(p[1]), float(p[2])])


def read_marker_centroid_W(
    r,
    specs: Sequence[MarkerSpec],
    *,
    min_visible: Optional[int] = None,
) -> Optional[np.ndarray]:
    """Centroid (mean) of N labeled marker world positions. Returns None if
    fewer than `min_visible` markers are currently published; defaults to
    requiring all of them, since a missing marker biases the centroid in a
    way the runtime usually can't recover from."""
    if min_visible is None:
        min_visible = len(specs)
    positions = []
    for spec in specs:
        p = read_marker_position_W(r, spec)
        if p is not None:
            positions.append(p)
    if len(positions) < min_visible:
        return None
    return np.mean(np.asarray(positions), axis=0)


def _read_cart_orientation_W(r, cart_rb_id: int) -> Optional[np.ndarray]:
    """Read the cart rigid body's world-frame rotation as a 3x3 matrix from
    `sai2::optitrack::rigid_body_ori::<id>` (which is a quaternion). Returns
    None if the key is missing."""
    raw = r.get(f"sai2::optitrack::rigid_body_ori::{int(cart_rb_id)}")
    if raw is None:
        return None
    try:
        q = json.loads(raw)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    if len(q) != 4:
        return None
    return quat_to_R(float(q[0]), float(q[1]), float(q[2]), float(q[3]))


def compute_T_W_A_from_markers(
    r,
    specs: Sequence[MarkerSpec],
    cart_rb_id: int,
    *,
    R_cart_to_arm: Optional[np.ndarray] = None,
    min_visible: Optional[int] = None,
) -> Optional[SE3]:
    """Live arm base pose T_W_A, derived from:
      - translation : centroid of the `specs` marker positions in world frame
      - rotation    : cart rigid body's R_W_B, post-multiplied by an optional
                      constant `R_cart_to_arm` to align the cart-rb local axes
                      with Franka base axes.

    Returns None if the markers or the cart rigid body aren't currently
    visible (caller should retry / fall back to previous estimate).
    """
    t_W_A = read_marker_centroid_W(r, specs, min_visible=min_visible)
    if t_W_A is None:
        return None
    R_W_B = _read_cart_orientation_W(r, cart_rb_id)
    if R_W_B is None:
        return None
    if R_cart_to_arm is None:
        R_cart_to_arm = np.eye(3)
    R_W_A = R_W_B @ R_cart_to_arm
    return (R_W_A, t_W_A)


# ---------- Arm-marker calibration file I/O -----------------------------------

def arm_marker_calibration_path() -> str:
    """Conventional location next to robot_marker_calibration.json."""
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "optitrack",
        "arm_marker_calibration.json",
    )


def save_arm_marker_calibration(
    path: str,
    *,
    marker_specs: Sequence[MarkerSpec],
    cart_rigid_body_id: int,
    R_cart_to_arm: np.ndarray,
    T_E_P: SE3,
    metadata: Optional[dict] = None,
) -> None:
    payload = {
        "comment": (
            "Marker-based arm calibration for the sports_bot rig. Arm base "
            "position T_W_A is computed live as the centroid of `marker_specs` "
            "(world-frame positions published by StreamDataSkeleton.py). "
            "Orientation is taken from cart rigid body `cart_rigid_body_id` "
            "via its rigid_body_ori key, post-multiplied by `R_cart_to_arm` "
            "to handle any constant rotation between the cart-rb local frame "
            "and the Franka arm base frame. T_E_P is the static EE→racket "
            "sweet-spot transform (measured, not solved) used by the "
            "SwingPlanner."
        ),
        "marker_specs": [
            [int(spec[0]), int(spec[1])] for spec in marker_specs
        ],
        "cart_rigid_body_id": int(cart_rigid_body_id),
        "R_cart_to_arm": [[float(v) for v in row] for row in R_cart_to_arm],
        "T_E_P": _se3_to_dict(T_E_P),
    }
    if metadata:
        payload.update(metadata)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


@dataclass
class ArmMarkerCalibration:
    """Materialized form of the arm-marker calibration."""
    marker_specs: List[MarkerSpec]
    cart_rigid_body_id: int
    R_cart_to_arm: np.ndarray
    T_E_P: SE3


def load_arm_marker_calibration(path: str) -> ArmMarkerCalibration:
    with open(path, "r") as f:
        data = json.load(f)
    specs = [(int(s[0]), int(s[1])) for s in data["marker_specs"]]
    R_cart_to_arm = np.asarray(data.get("R_cart_to_arm", np.eye(3).tolist()),
                               dtype=float)
    if R_cart_to_arm.shape != (3, 3):
        raise ValueError(f"R_cart_to_arm has shape {R_cart_to_arm.shape}, "
                         f"expected (3, 3)")
    return ArmMarkerCalibration(
        marker_specs=specs,
        cart_rigid_body_id=int(data["cart_rigid_body_id"]),
        R_cart_to_arm=R_cart_to_arm,
        T_E_P=_se3_from_dict(data["T_E_P"]),
    )


def world_racket_to_arm_ee(
    T_W_P_goal: SE3,
    T_W_A: SE3,
    T_E_P: SE3,
) -> SE3:
    """Inverse-compose a desired world-frame racket sweet-spot pose into an
    arm-base-frame EE pose that the OpenSai cartesian controller can consume:

        T_A_E_desired = T_W_A⁻¹ ⊕ T_W_P_goal ⊕ T_E_P⁻¹

    This is the runtime mapping the FSM / arm_intercept loop calls each tick
    after the SwingPlanner produces a racket pose in world frame.
    """
    return se3_compose(se3_compose(se3_inverse(T_W_A), T_W_P_goal),
                       se3_inverse(T_E_P))
