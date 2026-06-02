"""Fit the foam ball drag coefficient k from recorded OptiTrack trajectories.

Loads every new_ball_*.npz in <repo>/recordings/, segments the incoming-throw
portions, and sweeps k over [0.0, 0.8] in 0.05 increments. For each k it
propagates the per-tick LS state estimates to the strike plane using the drag
model and compares the predicted crossing to the actual one. Prints a summary
table and recommends the k that minimises mean error in the commit window
(TTI 0.0–0.5 s), then writes the result into foam_ball/config.py.

Usage (from sports_bot root):
    python scripts/fit_drag_coefficient.py
    python scripts/fit_drag_coefficient.py --recordings-dir recordings/ \\
                                           --strike-plane-x 0.60 \\
                                           --apply            # write k back to config
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from foam_ball.config import FoamBallConfig
from foam_ball.tracker import _propagate_drag_to_plane
from state_machine.ball_tracker import BallSample, REJECT_NONE


# ---------------------------------------------------------------------------
# Replay harness  (FoamBallTracker without Redis, driven by recorded samples)
# ---------------------------------------------------------------------------

class _ReplayFoamTracker:
    """FoamBallTracker driven by explicit (t, pos) samples for offline analysis.

    Mirrors the filtering logic in BallTracker.update() so the replay sees
    exactly the same sample stream the FSM would see. Exposes _fit_state()
    and last_reject_reason so the fitting loop can inspect tracker internals.
    """

    def __init__(self, cfg: FoamBallConfig) -> None:
        self._cfg = cfg
        self._history: collections.deque = collections.deque(maxlen=cfg.history_size)
        med_w = max(1, cfg.median_filter_window)
        self._raw_buffer: collections.deque = collections.deque(maxlen=med_w)
        self.last_reject_reason: str = REJECT_NONE

    def reset(self) -> None:
        self._history.clear()
        self._raw_buffer.clear()
        self.last_reject_reason = REJECT_NONE

    def ingest(self, t: float, pos: np.ndarray) -> Optional[BallSample]:
        if pos is None or pos.shape != (3,):
            return None
        pos = pos.astype(float)
        if self._cfg.median_filter_window >= 2:
            self._raw_buffer.append(pos.copy())
            if len(self._raw_buffer) >= self._cfg.median_filter_window:
                pos = np.median(np.stack(self._raw_buffer), axis=0)
        while self._history and (t - self._history[0].t) > self._cfg.history_max_age_s:
            self._history.popleft()
        if self._history:
            last = self._history[-1]
            dist = float(np.linalg.norm(pos - last.pos))
            if dist > self._cfg.max_position_jump:
                return None
            if self._cfg.max_implied_speed_mps > 0:
                dt_s = float(t) - last.t
                if dt_s > 1e-6 and dist / dt_s > self._cfg.max_implied_speed_mps:
                    return None
        sample = BallSample(t=float(t), pos=pos)
        self._history.append(sample)
        return sample

    def _fit_state(self) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
        """Least-squares fit identical to BallTracker._fit_state."""
        if len(self._history) < max(3, self._cfg.min_history_for_prediction):
            return None
        ts = np.array([s.t for s in self._history])
        pts = np.array([s.pos for s in self._history])
        t_now = ts[-1]
        dt = ts - t_now
        z_corr = pts[:, 2] + 0.5 * self._cfg.gravity * dt * dt
        A = np.stack([np.ones_like(dt), dt], axis=1)
        try:
            sx, *_ = np.linalg.lstsq(A, pts[:, 0], rcond=None)
            sy, *_ = np.linalg.lstsq(A, pts[:, 1], rcond=None)
            sz, *_ = np.linalg.lstsq(A, z_corr, rcond=None)
        except np.linalg.LinAlgError:
            return None
        p0 = np.array([sx[0], sy[0], sz[0]])
        v0 = np.array([sx[1], sy[1], sz[1]])
        return p0, v0, t_now


# ---------------------------------------------------------------------------
# Recording helpers
# ---------------------------------------------------------------------------

def _find_crossing(
    ts: np.ndarray,
    pos: np.ndarray,
    target_x: float,
) -> Optional[np.ndarray]:
    """Linear interpolation to find where ball first crossed x = target_x."""
    for i in range(len(ts) - 1):
        x0, x1 = pos[i, 0], pos[i + 1, 0]
        if (x0 - target_x) * (x1 - target_x) <= 0.0 and x0 != x1:
            alpha = (target_x - x0) / (x1 - x0)
            if 0.0 <= alpha <= 1.0:
                return pos[i] + alpha * (pos[i + 1] - pos[i])
    return None


def _segment_throws(
    ts: np.ndarray,
    pos: np.ndarray,
    target_x: float,
    gap_s: float = 0.4,
    min_samples: int = 10,
    min_incoming_speed: float = 0.5,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Segment recording into (timestamps, positions, actual_crossing) tuples.

    Only keeps segments that contain incoming-ball samples AND actually cross
    target_x. Returns one tuple per throw.
    """
    segs: List[Tuple[int, int]] = []
    cur = 0
    for i in range(1, len(ts)):
        if ts[i] - ts[i - 1] > gap_s:
            segs.append((cur, i))
            cur = i
    segs.append((cur, len(ts)))

    throws = []
    for a, b in segs:
        if b - a < min_samples:
            continue
        t_seg = ts[a:b]
        p_seg = pos[a:b]
        # Must have at least some incoming-ball samples.
        dx = np.diff(p_seg[:, 0])
        dt_s = np.maximum(np.diff(t_seg), 1e-6)
        vx = dx / dt_s
        if not (vx < -min_incoming_speed).any():
            continue
        # Must actually cross the strike plane.
        crossing = _find_crossing(t_seg, p_seg, target_x)
        if crossing is None:
            continue
        throws.append((t_seg, p_seg, crossing))
    return throws


# ---------------------------------------------------------------------------
# Core sweep
# ---------------------------------------------------------------------------

def _collect_state_estimates(
    ts: np.ndarray,
    pos: np.ndarray,
    cfg: FoamBallConfig,
    target_x: float,
    actual_crossing: np.ndarray,
    tti_min: float,
    tti_max: float,
) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    """Replay a single throw through the LS tracker and collect per-tick state.

    Returns a list of (p0, v0, tti_estimated) for ticks where the ball is
    incoming, the fit is valid, and tti falls within [tti_min, tti_max].

    tti_estimated is computed analytically from the zero-drag model (used only
    for bucketing; the drag-corrected tti is computed in the sweep loop).
    """
    tracker = _ReplayFoamTracker(cfg)
    last_t: Optional[float] = None
    last_pos: Optional[np.ndarray] = None
    not_incoming_streak = 0
    NOT_INCOMING_STREAK = 5
    out = []

    for t, p in zip(ts, pos):
        if last_t is not None:
            dt = float(t) - last_t
            if dt > max(cfg.history_max_age_s, 0.3):
                tracker.reset()
                not_incoming_streak = 0
            elif last_pos is not None and dt > 1e-6:
                vx = (p[0] - last_pos[0]) / dt
                if vx >= -cfg.min_incoming_speed:
                    not_incoming_streak += 1
                else:
                    if not_incoming_streak >= NOT_INCOMING_STREAK:
                        tracker.reset()
                    not_incoming_streak = 0
        last_t = float(t)
        last_pos = p

        if tracker.ingest(float(t), p) is None:
            continue
        fit = tracker._fit_state()
        if fit is None:
            continue
        p0, v0, _ = fit

        # Only keep incoming ticks.
        if v0[0] >= -cfg.min_incoming_speed:
            continue

        # Rough TTI from no-drag model for bucketing.
        if v0[0] >= 0:
            continue
        t_rough = (target_x - p0[0]) / v0[0]
        if not (tti_min <= t_rough <= tti_max):
            continue

        out.append((p0.copy(), v0.copy(), float(t_rough)))
    return out


def sweep_drag(
    recordings_dir: Path,
    target_x: float,
    k_values: np.ndarray,
    tti_min: float = 0.05,
    tti_max: float = 1.5,
    commit_tti_max: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Sweep drag coefficient over k_values and return per-k mean errors.

    Returns
    -------
    errs_all : (len(k_values),) — mean 3-D error (m) over all TTI
    errs_commit : (len(k_values),) — mean 3-D error (m) in commit window
    n_throws : int — number of throws used
    """
    base_cfg = FoamBallConfig(drag_coefficient=0.0, max_bounces=1)

    all_estimates: List[Tuple[np.ndarray, np.ndarray, float, np.ndarray]] = []

    npz_files = sorted(recordings_dir.glob("new_ball_*.npz"))
    if not npz_files:
        print(f"[fit] no new_ball_*.npz files found in {recordings_dir}", file=sys.stderr)
        return np.zeros(len(k_values)), np.zeros(len(k_values)), 0

    n_throws = 0
    for rec in npz_files:
        data = np.load(rec, allow_pickle=False)
        ts = np.asarray(data["timestamps"])
        pos = np.asarray(data["positions"])
        throws = _segment_throws(ts, pos, target_x)
        for t_seg, p_seg, crossing in throws:
            estimates = _collect_state_estimates(
                t_seg, p_seg, base_cfg, target_x, crossing,
                tti_min=tti_min, tti_max=tti_max,
            )
            for p0, v0, tti in estimates:
                all_estimates.append((p0, v0, tti, crossing))
            if estimates:
                n_throws += 1

    if not all_estimates:
        print("[fit] no usable state estimates extracted from recordings.", file=sys.stderr)
        return np.zeros(len(k_values)), np.zeros(len(k_values)), 0

    errs_all = np.zeros(len(k_values))
    errs_commit = np.zeros(len(k_values))
    counts_all = np.zeros(len(k_values), dtype=int)
    counts_commit = np.zeros(len(k_values), dtype=int)

    for ki, k in enumerate(k_values):
        cfg_k = FoamBallConfig(drag_coefficient=float(k), max_bounces=1)
        for p0, v0, tti_rough, crossing in all_estimates:
            result = _propagate_drag_to_plane(p0, v0, target_x, cfg_k)
            if isinstance(result, str):
                continue
            t_impact, p_impact, _, _ = result
            err = float(np.linalg.norm(p_impact - crossing))
            errs_all[ki] += err
            counts_all[ki] += 1
            if t_impact <= commit_tti_max:
                errs_commit[ki] += err
                counts_commit[ki] += 1

    # Means (avoid divide-by-zero)
    with np.errstate(invalid="ignore", divide="ignore"):
        errs_all = np.where(counts_all > 0, errs_all / counts_all, np.nan)
        errs_commit = np.where(counts_commit > 0, errs_commit / counts_commit, np.nan)

    return errs_all, errs_commit, n_throws


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--recordings-dir", default=str(REPO / "recordings"),
        help="Directory containing new_ball_*.npz files (default: <repo>/recordings/)",
    )
    parser.add_argument(
        "--strike-plane-x", type=float, default=0.60,
        help="Strike plane x coordinate in world frame (default: 0.60 m)",
    )
    parser.add_argument(
        "--k-min", type=float, default=0.0,
        help="Minimum drag coefficient to try (default: 0.0)",
    )
    parser.add_argument(
        "--k-max", type=float, default=0.80,
        help="Maximum drag coefficient to try (default: 0.80)",
    )
    parser.add_argument(
        "--k-step", type=float, default=0.05,
        help="Drag coefficient step size (default: 0.05)",
    )
    parser.add_argument(
        "--apply", action="store_true",
        help="Write the fitted drag_coefficient back to foam_ball/config.py",
    )
    args = parser.parse_args()

    rec_dir = Path(args.recordings_dir)
    k_values = np.round(np.arange(args.k_min, args.k_max + 1e-9, args.k_step), 4)

    print(f"[fit] recordings : {rec_dir}")
    print(f"[fit] strike plane x = {args.strike_plane_x:.3f} m")
    print(f"[fit] k sweep : {k_values[0]:.3f} … {k_values[-1]:.3f}  ({len(k_values)} values)")
    print()

    errs_all, errs_commit, n_throws = sweep_drag(
        rec_dir, args.strike_plane_x, k_values,
    )

    if n_throws == 0:
        print("[fit] no throws found — check recordings directory and file names.")
        return 1

    print(f"[fit] {n_throws} throw segment(s) used\n")
    print(f"{'k':>6s}  {'mean err (all TTI)':>19s}  {'mean err (commit ≤0.5s)':>23s}")
    print("-" * 55)
    for ki, k in enumerate(k_values):
        all_s = f"{errs_all[ki]*100:6.1f} cm" if not np.isnan(errs_all[ki]) else "  —    "
        com_s = f"{errs_commit[ki]*100:6.1f} cm" if not np.isnan(errs_commit[ki]) else "  —    "
        marker = " ◀" if not np.isnan(errs_commit[ki]) and errs_commit[ki] == np.nanmin(errs_commit) else ""
        print(f"  {k:.3f}  {all_s:>19s}  {com_s:>23s}{marker}")

    # Recommend k that minimises error in the commit window.
    if not np.all(np.isnan(errs_commit)):
        best_idx = int(np.nanargmin(errs_commit))
        best_k = float(k_values[best_idx])
        print(f"\n[fit] recommended drag_coefficient = {best_k:.3f}")
        print(f"      mean prediction error in commit window: {errs_commit[best_idx]*100:.1f} cm")

        if args.apply:
            _write_drag_coefficient(best_k)
    else:
        print("\n[fit] could not determine a best k (no predictions in commit window).")
        best_k = None

    return 0


def _write_drag_coefficient(k: float) -> None:
    """Patch drag_coefficient in foam_ball/config.py in-place."""
    config_path = REPO / "foam_ball" / "config.py"
    text = config_path.read_text()
    import re
    new_text = re.sub(
        r"(drag_coefficient:\s*float\s*=\s*)[0-9.]+",
        f"\\g<1>{k:.4f}",
        text,
    )
    if new_text == text:
        print(f"[fit] WARNING: could not find 'drag_coefficient: float = ...' in {config_path}")
        return
    config_path.write_text(new_text)
    print(f"[fit] wrote drag_coefficient = {k:.4f} to {config_path}")


if __name__ == "__main__":
    sys.exit(main())
