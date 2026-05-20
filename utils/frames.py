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
from typing import Optional, Sequence, Tuple

import numpy as np


SE2 = Tuple[float, float, float]   # (x, y, theta)


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
