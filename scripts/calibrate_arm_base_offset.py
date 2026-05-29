#!/usr/bin/env python3
"""One-shot calibration of the arm base offset from the cart rigid body.

Captures ~2 seconds of simultaneous OptiTrack data from:
  - The cart's OptiTrack rigid body (camera-tracked, ID given by --base-rb-id)
  - The arm's labeled markers (IDs from arm_marker_calibration.json)

Computes the fixed 3D vector v_B from the rigid body centroid to the arm base
origin (in the base body's yaw-projected frame) and the fixed yaw offset R_B_A.
Saves to arm_base_offset_calibration.json.

After calibration, T_W_A is derived from the cart rigid body alone at runtime —
no arm markers needed. The source is pure OptiTrack optical tracking (~120 Hz),
not wheel odometry, so it is immune to wheel slip.

Re-run only when the arm is physically re-mounted on the cart.

Prerequisites:
  redis-server
  StreamDataSkeleton.py (rigid bodies AND labeled markers streaming)
  arm_marker_calibration.json (correct marker IDs; run verify_arm_calibration.py --init)

Usage:
  python sports_bot/scripts/calibrate_arm_base_offset.py
  python sports_bot/scripts/calibrate_arm_base_offset.py --base-rb-id 11
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from typing import List, Tuple

import numpy as np
import redis

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SPORTS_BOT_DIR = os.path.dirname(_THIS_DIR)
_OPENSAI_DIR = os.path.dirname(_SPORTS_BOT_DIR)
if _OPENSAI_DIR not in sys.path:
    sys.path.insert(0, _OPENSAI_DIR)

from sports_bot.utils.frames import (  # noqa: E402
    ArmBaseOffsetCalibration,
    arm_base_offset_calibration_path,
    arm_marker_calibration_path,
    average_angles,
    compute_T_W_A_from_base_offset,
    load_arm_marker_calibration,
    read_marker_position_W,
    read_rigid_body_pose_W,
    save_arm_base_offset_calibration,
)


def _capture(
    r: redis.Redis,
    base_rb_id: int,
    marker_specs,
    n_samples: int,
    rate_hz: float = 50.0,
) -> Tuple[List[np.ndarray], List[float], List[np.ndarray], List[np.ndarray]]:
    """Capture n_samples of simultaneous rigid body + marker readings.

    Returns (rb_positions, rb_yaws, m0_positions, m1_positions).
    Only includes ticks where all three sources are available.
    """
    dt = 1.0 / rate_hz
    rb_positions: List[np.ndarray] = []
    rb_yaws: List[float] = []
    m0_positions: List[np.ndarray] = []
    m1_positions: List[np.ndarray] = []
    last_warn = -math.inf

    while len(rb_positions) < n_samples:
        t0 = time.perf_counter()

        T_W_B = read_rigid_body_pose_W(r, base_rb_id)
        p0 = read_marker_position_W(r, marker_specs[0])
        p1 = read_marker_position_W(r, marker_specs[1])

        if T_W_B is None or p0 is None or p1 is None:
            now = time.perf_counter()
            if now - last_warn > 1.0:
                missing = []
                if T_W_B is None:
                    missing.append(f'rigid_body {base_rb_id}')
                if p0 is None:
                    missing.append(f'marker {marker_specs[0]}')
                if p1 is None:
                    missing.append(f'marker {marker_specs[1]}')
                print(f'  [WARN] waiting for: {", ".join(missing)}')
                last_warn = now
        else:
            R_W_B, t_W_B = T_W_B
            yaw_B = math.atan2(float(R_W_B[1, 0]), float(R_W_B[0, 0]))
            rb_positions.append(t_W_B.copy())
            rb_yaws.append(yaw_B)
            m0_positions.append(p0.copy())
            m1_positions.append(p1.copy())
            print(f'  {len(rb_positions)}/{n_samples}', end='\r', flush=True)

        time.sleep(max(0.0, dt - (time.perf_counter() - t0)))

    print()
    return rb_positions, rb_yaws, m0_positions, m1_positions


def main() -> None:
    ap = argparse.ArgumentParser(
        description='One-shot calibration: arm base offset from cart rigid body.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument('--base-rb-id', type=int, default=11,
                    help='OptiTrack rigid body streaming ID for the cart.')
    ap.add_argument('--n-samples', type=int, default=100,
                    help='Samples to average (50 Hz → 2 s).')
    ap.add_argument('--marker-calibration', default=None,
                    help='arm_marker_calibration.json path. Default: canonical location.')
    ap.add_argument('--output', default=None,
                    help='Output path. Default: optitrack/arm_base_offset_calibration.json')
    ap.add_argument('--redis-host', default='localhost')
    ap.add_argument('--redis-port', type=int, default=6379)
    ap.add_argument('--force', action='store_true',
                    help='Overwrite existing calibration without prompting.')
    args = ap.parse_args()

    r = redis.Redis(host=args.redis_host, port=args.redis_port,
                    decode_responses=True)
    try:
        r.ping()
    except redis.exceptions.ConnectionError as e:
        print(f'[calibrate] Cannot reach Redis: {e}')
        sys.exit(1)

    marker_cal_path = args.marker_calibration or arm_marker_calibration_path()
    if not os.path.isfile(marker_cal_path):
        print(f'[calibrate] arm_marker_calibration.json not found: {marker_cal_path}')
        print('  Run: python sports_bot/scripts/verify_arm_calibration.py --init')
        sys.exit(1)
    arm_cal = load_arm_marker_calibration(marker_cal_path)

    out_path = args.output or arm_base_offset_calibration_path()

    print(f'[calibrate] cart rigid body ID : {args.base_rb_id}')
    print(f'[calibrate] arm marker IDs     : {arm_cal.marker_specs}')
    print(f'[calibrate] output             : {out_path}')
    print()
    print('Hold the cart still. Cart rigid body AND both arm markers must be visible.')
    input('Press Enter to start capture...')
    print()

    rb_pos, rb_yaws, m0_pos, m1_pos = _capture(
        r, args.base_rb_id, arm_cal.marker_specs, n_samples=args.n_samples)

    # --- Average captured data
    t_W_B_mean = np.mean(np.stack(rb_pos), axis=0)
    yaw_B_mean = average_angles(rb_yaws)
    t_m0_mean = np.mean(np.stack(m0_pos), axis=0)
    t_m1_mean = np.mean(np.stack(m1_pos), axis=0)

    # --- Jitter stats
    rb_sig_mm = np.stack(rb_pos).std(axis=0) * 1000.0
    rb_pos_rms_mm = float(np.linalg.norm(rb_sig_mm))
    yaw_sig_deg = math.degrees(float(np.asarray(rb_yaws).std()))

    # --- Build T_W_A from averaged marker positions (Z-up convention)
    y_raw = t_m1_mean - t_m0_mean
    z_world = np.array([0.0, 0.0, 1.0])
    y_horiz = y_raw - float(np.dot(y_raw, z_world)) * z_world
    ny = float(np.linalg.norm(y_horiz))
    if ny < 1e-6:
        print('[calibrate] ERROR: arm markers appear co-vertical — check placement.')
        sys.exit(1)
    y_hat = y_horiz / ny
    x_hat = np.cross(y_hat, z_world)
    R_W_A = np.column_stack([x_hat, y_hat, z_world])
    t_W_A = 0.5 * (t_m0_mean + t_m1_mean)
    yaw_A = math.atan2(float(R_W_A[1, 0]), float(R_W_A[0, 0]))

    # --- Build T_W_B_planar (yaw-only, Z-up) from averaged rigid body data
    cb, sb = math.cos(yaw_B_mean), math.sin(yaw_B_mean)
    R_W_B_planar = np.array([[cb, -sb, 0.], [sb, cb, 0.], [0., 0., 1.]])

    # --- Calibration
    v_B = R_W_B_planar.T @ (t_W_A - t_W_B_mean)
    R_B_A = R_W_B_planar.T @ R_W_A  # = Rz(yaw_A - yaw_B)
    yaw_offset = math.atan2(float(R_B_A[1, 0]), float(R_B_A[0, 0]))

    # --- Live cross-check: read fresh sample and compare both derivations of T_W_A
    time.sleep(0.05)
    T_W_B_live = read_rigid_body_pose_W(r, args.base_rb_id)
    p0_live = read_marker_position_W(r, arm_cal.marker_specs[0])
    p1_live = read_marker_position_W(r, arm_cal.marker_specs[1])
    live_diff_mm = None
    if T_W_B_live is not None and p0_live is not None and p1_live is not None:
        cal_tmp = ArmBaseOffsetCalibration(
            base_rigid_body_id=args.base_rb_id, v_B=v_B, R_B_A=R_B_A)
        _, t_W_A_from_rb = compute_T_W_A_from_base_offset(T_W_B_live, cal_tmp)
        t_W_A_from_markers = 0.5 * (p0_live + p1_live)
        live_diff_mm = float(np.linalg.norm(t_W_A_from_rb - t_W_A_from_markers)) * 1000.0

    # --- Report
    print(f'  Cart rigid body (averaged):')
    print(f'    pos   = [{t_W_B_mean[0]:+.4f}, {t_W_B_mean[1]:+.4f}, {t_W_B_mean[2]:+.4f}] m')
    print(f'    yaw   = {math.degrees(yaw_B_mean):+.2f}°')
    print(f'    σ_pos = ({rb_sig_mm[0]:.2f}, {rb_sig_mm[1]:.2f}, {rb_sig_mm[2]:.2f}) mm  '
          f'rms={rb_pos_rms_mm:.2f} mm    σ_yaw = {yaw_sig_deg:.3f}°    '
          f'({len(rb_pos)} samples)')
    print(f'  Arm base (from markers, averaged):')
    print(f'    pos   = [{t_W_A[0]:+.4f}, {t_W_A[1]:+.4f}, {t_W_A[2]:+.4f}] m')
    print(f'    yaw   = {math.degrees(yaw_A):+.2f}°')
    print(f'  Calibration result:')
    print(f'    v_B       = [{v_B[0]:+.4f}, {v_B[1]:+.4f}, {v_B[2]:+.4f}] m  '
          f'(arm base origin in base body frame)')
    print(f'    yaw_B_A   = {math.degrees(yaw_offset):+.2f}°  '
          f'(arm base yaw relative to cart body; expect ≈0° if arm faces cart +X)')

    if live_diff_mm is not None:
        tag = '  ← HIGH — cart may have shifted' if live_diff_mm > 15.0 else ''
        print(f'  Live cross-check (rb vs markers): {live_diff_mm:.1f} mm{tag}')
    else:
        print('  Live cross-check: skipped (source unavailable)')

    if rb_pos_rms_mm > 5.0 or yaw_sig_deg > 0.3:
        print()
        print(f'  WARNING: high capture jitter ({rb_pos_rms_mm:.1f} mm / {yaw_sig_deg:.2f}°).')
        print('  Consider re-running after the cart has been stationary longer.')

    print()

    if os.path.isfile(out_path) and not args.force:
        try:
            ans = input(f'  {out_path} already exists. Overwrite? [y/N] ').strip().lower()
        except EOFError:
            ans = ''
        if ans != 'y':
            print('  Not written.')
            sys.exit(0)

    save_arm_base_offset_calibration(
        out_path,
        base_rigid_body_id=args.base_rb_id,
        v_B=v_B,
        R_B_A=R_B_A,
        metadata={
            'calibrated_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'n_samples': len(rb_pos),
            'rb_pos_sigma_rms_mm': round(rb_pos_rms_mm, 3),
            'rb_yaw_sigma_deg': round(yaw_sig_deg, 4),
            'live_cross_check_mm': round(live_diff_mm, 2) if live_diff_mm is not None else None,
            'arm_marker_calibration': os.path.basename(marker_cal_path),
            'marker_specs_used': [list(s) for s in arm_cal.marker_specs],
        },
    )
    print(f'[calibrate] Wrote {out_path}')


if __name__ == '__main__':
    main()
