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
    returns the same body's orientation in frame T.

    Used by StreamDataSkeleton.py to convert raw Motive-room-frame rigid body
    quaternions into world-frame ones via the world calibration."""
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


# ---------- SE(3) <-> JSON helpers --------------------------------------------

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


# Arm base pose T_W_A is derived live from two labeled OptiTrack markers on
# the cart, placed equidistantly on the left (-Y) and right (+Y) sides of the
# Franka base. The midpoint of the marker pair = arm base origin; the pair
# direction = Franka +Y; world +Z (flat-floor assumption) = Franka +Z;
# Franka +X = Y × Z closes the right-handed frame.
#
# Schema persisted to `sports_bot/optitrack/arm_marker_calibration.json`:
#
#   {
#     "marker_specs": [[model_id, marker_id], [model_id, marker_id]],
#                     # exactly 2 specs; spec[0] on -Y side, spec[1] on +Y side
#                     # (i.e. vector spec[0]→spec[1] = Franka +Y)
#     "T_E_P":        {translation_m, rotation_matrix}
#                     # measured / CAD'd EE flange → racket sweet spot
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


def compute_T_W_A_from_markers(
    r,
    specs: Sequence[MarkerSpec],
) -> Optional[SE3]:
    """Live arm base pose T_W_A from a 2-marker spec.

    Translation: midpoint (centroid) of the two markers in world frame —
                 this is the *origin* of the returned arm-base frame.
    Rotation:    Franka +Y = horiz_project(p_spec1 − p_spec0); Franka +Z =
                 world +Z (flat-floor upright-arm assumption); Franka +X =
                 Y × Z closes the right-handed frame.

    Returns None if either marker isn't currently visible, or if the two
    markers are stacked vertically (degenerate Franka +Y).
    """
    if len(specs) != 2:
        raise ValueError(
            f"compute_T_W_A_from_markers expects exactly 2 marker specs, "
            f"got {len(specs)}"
        )
    p0 = read_marker_position_W(r, specs[0])
    p1 = read_marker_position_W(r, specs[1])
    if p0 is None or p1 is None:
        return None

    # Centroid → arm base origin.
    t_W_A = 0.5 * (p0 + p1)

    # Pair direction → Franka +Y (after horizontal projection so a slight Z
    # mismatch between the two spheres doesn't tilt the inferred +Y axis).
    y_raw = p1 - p0
    z_world = np.array([0.0, 0.0, 1.0])
    y_horiz = y_raw - float(np.dot(y_raw, z_world)) * z_world
    ny = float(np.linalg.norm(y_horiz))
    if ny < 1e-6:
        # Markers stacked vertically — can't recover Franka +Y this way.
        return None
    y_hat = y_horiz / ny
    z_hat = z_world
    # Right-handed: X = Y × Z.
    x_hat = np.cross(y_hat, z_hat)
    R_W_A = np.column_stack([x_hat, y_hat, z_hat])

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
    T_E_P: SE3,
    metadata: Optional[dict] = None,
) -> None:
    """Persist the marker-based arm calibration. `marker_specs` must have
    exactly 2 entries: spec[0] on the -Y side of the Franka base, spec[1] on
    the +Y side (i.e. vector spec[0]→spec[1] is treated as Franka +Y).
    `T_E_P` is the measured EE flange → racket sweet-spot transform."""
    if len(marker_specs) != 2:
        raise ValueError(
            f"save_arm_marker_calibration: marker_specs must have exactly 2 "
            f"entries (spec[0] on -Y side, spec[1] on +Y side; vector "
            f"spec[0]→spec[1] = Franka +Y), got {len(marker_specs)}."
        )
    payload = {
        "comment": (
            "Marker-based arm calibration for the sports_bot rig. Arm base "
            "origin T_W_A.t is the midpoint of the two markers in marker_specs "
            "(world-frame positions published by StreamDataSkeleton.py). "
            "Arm base orientation: spec[0] on the -Y side of the Franka base, "
            "spec[1] on the +Y side, so vector spec[0]→spec[1] is treated as "
            "Franka +Y (after horizontal projection); world +Z is Franka +Z "
            "(assumes upright cart); Franka +X = Y × Z closes the right-handed "
            "frame. T_E_P is the measured EE flange → racket sweet-spot "
            "transform used by the SwingPlanner."
        ),
        "marker_specs": [
            [int(spec[0]), int(spec[1])] for spec in marker_specs
        ],
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
    T_E_P: SE3


def load_arm_marker_calibration(path: str) -> ArmMarkerCalibration:
    with open(path, "r") as f:
        data = json.load(f)
    specs = [(int(s[0]), int(s[1])) for s in data["marker_specs"]]
    if len(specs) != 2:
        raise ValueError(
            f"{path}: marker_specs must have exactly 2 entries, got "
            f"{len(specs)}."
        )
    return ArmMarkerCalibration(
        marker_specs=specs,
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


# Conservative Franka Panda workspace for the FLANGE in the arm base frame.
# Hardware reach is ~85 cm; leave ~10 cm headroom so orientation goals have
# room to settle without the cartesian controller saturating. Z range is set
# for a paddle-mounted upright Franka — adjust if your mount geometry differs.
FRANKA_DEFAULT_REACH_M = 0.75
FRANKA_DEFAULT_Z_MIN_M = -0.10
FRANKA_DEFAULT_Z_MAX_M = 1.00

# Distance from the sweet spot (compliant frame, 35 cm along EE +Z from
# flange) to the far tip of the paddle along EE +Z.
# Physical layout: flange → 35 cm → sweet spot / compliant frame
#                              → 10 cm → paddle tip (far end of face)
# Total flange→tip = 45 cm; sweet-spot→tip = 10 cm.
# Used by enforce_paddle_floor to ensure the paddle tip stays above the floor.
PADDLE_TIP_OFFSET_M = 0.10


def clip_to_arm_workspace(
    t_A_E: np.ndarray,
    *,
    r_max: float = FRANKA_DEFAULT_REACH_M,
    z_min: float = FRANKA_DEFAULT_Z_MIN_M,
    z_max: float = FRANKA_DEFAULT_Z_MAX_M,
) -> Tuple[np.ndarray, bool]:
    """Clip an EE position (in the arm base frame) to a conservative reachable
    cylinder. Returns (clipped_position, was_clipped).

    Strategy: clamp Z to [z_min, z_max], then radially shrink xy to r_max if
    it overshoots. Orientation is left to the caller — OpenSai's cartesian
    task generally handles unreachable orientations gracefully via task-space
    prioritization, while position needs to stay in-workspace or the
    controller saturates and the arm sits in a compromise pose that *looks*
    like tracking but isn't.

    The point is that during a chase the cart may not have caught up yet, so
    the FSM-level world target can be far outside the current arm workspace.
    Clipping makes the arm reach as far as it can toward the target rather
    than committing to a goal it can't fulfill.
    """
    clipped = False
    t = np.asarray(t_A_E, dtype=float).copy()
    if t[2] < z_min:
        t[2] = z_min
        clipped = True
    elif t[2] > z_max:
        t[2] = z_max
        clipped = True
    r_xy = float(np.linalg.norm(t[:2]))
    if r_xy > r_max:
        t[:2] = t[:2] * (r_max / r_xy)
        clipped = True
    return t, clipped


def enforce_paddle_floor(
    t_A_E: np.ndarray,
    R_A_E: np.ndarray,
    T_W_A: SE3,
    *,
    paddle_tip_offset: float = PADDLE_TIP_OFFSET_M,
    clearance: float = 0.05,
    floor_z_world: float = 0.0,
) -> Tuple[np.ndarray, bool]:
    """Raise the commanded EE goal in arm-frame Z if the paddle tip would be
    below the world floor. Returns (adjusted_t_A_E, was_adjusted).

    The world-frame Z of the paddle tip at the commanded goal is computed exactly
    from the live arm base pose (T_W_A from OptiTrack) and the commanded EE
    orientation (R_A_E). Assumes an upright cart so arm-frame Z = world Z
    (Franka +Z = world +Z), which is guaranteed by compute_T_W_A_from_markers.

        z_tip_world = T_W_A.t[2] + t_A_E[2] + R_A_E[2,2] * paddle_tip_offset

    Call this before clip_to_arm_workspace so the floor constraint is applied
    in world-frame terms; the hardware clip then handles reach/height limits.
    """
    arm_base_z = float(T_W_A[1][2])
    z_tip_world = arm_base_z + float(t_A_E[2]) + float(R_A_E[2, 2]) * paddle_tip_offset
    floor_limit = floor_z_world + clearance
    if z_tip_world >= floor_limit:
        return np.asarray(t_A_E, dtype=float), False
    t = np.asarray(t_A_E, dtype=float).copy()
    t[2] += floor_limit - z_tip_world
    return t, True
