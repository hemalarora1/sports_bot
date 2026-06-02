"""Volley-mode A/B between the LS and EKF ball trackers.

Replays every recording in `sports_bot/recordings/` through both trackers
configured for volleys only (`max_bounces = 0` — any predicted trajectory
that would bounce before reaching the strike plane is rejected). Aggregates
the per-tick predicted-vs-actual error by time-to-impact bucket so we can
see whether the EKF is meaningfully better than the LS fitter when bounces
are removed from the picture.

For the EKF, also prints the mean predicted-intercept `σ_yz` per bucket —
i.e., the lateral + vertical uncertainty on the strike plane. This is the
quantity an uncertainty-aware FSM commit rule would gate on.

Run:
    /opt/homebrew/Caskroom/miniconda/base/envs/opensai/bin/python \
        sports_bot/scripts/compare_trackers_volley.py
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from sports_bot.state_machine.ball_tracker_test import _replay, _segment_throws  # noqa: E402
from sports_bot.state_machine.config import PickleballConfig  # noqa: E402


BUCKETS = [(1.50, 1.00), (1.00, 0.70), (0.70, 0.50), (0.50, 0.30),
           (0.30, 0.15), (0.15, 0.00)]
RECORDINGS_DIR = REPO / "sports_bot" / "recordings"


def run(tracker_kind, *, max_bounces, history_size=None, history_max_age_s=None,
        ekf_bounce_handling=True, process_accel_std_xy=None,
        process_accel_std_z=None):
    cfg = PickleballConfig()
    cfg.tracker.max_bounces = max_bounces
    if history_size is not None:
        cfg.tracker.history_size = history_size
    if history_max_age_s is not None:
        cfg.tracker.history_max_age_s = history_max_age_s
    cfg.tracker.ekf.online_bounce_handling = ekf_bounce_handling
    if process_accel_std_xy is not None:
        cfg.tracker.ekf.process_accel_std_xy = process_accel_std_xy
    if process_accel_std_z is not None:
        cfg.tracker.ekf.process_accel_std_z = process_accel_std_z
    errs = defaultdict(list)
    sigmas = defaultdict(list)
    n_pred = 0
    for rec in sorted(RECORDINGS_DIR.glob("throws_*.npz")):
        data = np.load(rec, allow_pickle=False)
        ts = np.asarray(data["timestamps"])
        ps = np.asarray(data["positions"])
        ticks = _replay(ts, ps, cfg, tracker_kind=tracker_kind)
        throws = _segment_throws(
            ts, ps, ticks, cfg.court.strike_plane_x,
            gap_s=0.4, min_samples=5,
        )
        for throw in throws:
            if throw.actual_crossing is None:
                continue
            _, p_cross = throw.actual_crossing
            for t in throw.ticks:
                if t.intercept_pos is None or t.time_to_impact is None:
                    continue
                n_pred += 1
                err = float(np.linalg.norm(t.intercept_pos - p_cross))
                for upper, lower in BUCKETS:
                    if lower <= t.time_to_impact < upper:
                        errs[(lower, upper)].append(err)
                        if t.intercept_pos_cov is not None:
                            sigma_yz = float(np.sqrt(
                                t.intercept_pos_cov[1, 1]
                                + t.intercept_pos_cov[2, 2]
                            ))
                            sigmas[(lower, upper)].append(sigma_yz)
                        break
    return errs, sigmas, n_pred


def print_table(label, ls_errs, ek_errs, ek_sigmas):
    print(f"\n{label}")
    print(f"{'tti window':>16s}  {'LS  n':>6s} {'LS mean':>9s} {'LS max':>8s}"
          f"  {'EKF n':>6s} {'EKF mean':>9s} {'EKF max':>8s}"
          f"   Δmean   EKF σ_yz")
    for upper, lower in BUCKETS:
        ls = np.array(ls_errs[(lower, upper)])
        ek = np.array(ek_errs[(lower, upper)])
        if len(ls) == 0 and len(ek) == 0:
            continue
        sig = np.array(ek_sigmas[(lower, upper)])
        sig_str = f"{sig.mean()*100:5.1f}cm" if len(sig) else "  —   "
        ls_n = len(ls)
        ek_n = len(ek)
        ls_mean = ls.mean()*100 if ls_n else float("nan")
        ek_mean = ek.mean()*100 if ek_n else float("nan")
        ls_max = ls.max()*100 if ls_n else float("nan")
        ek_max = ek.max()*100 if ek_n else float("nan")
        d = (ek_mean - ls_mean) if (ls_n and ek_n) else float("nan")
        print(f"[{lower:.2f}, {upper:.2f})   "
              f"{ls_n:>6d} {ls_mean:>8.1f}cm {ls_max:>6.1f}cm  "
              f"{ek_n:>6d} {ek_mean:>8.1f}cm {ek_max:>6.1f}cm  "
              f"{d:+6.1f}cm   {sig_str}")


def main() -> int:
    print("=" * 92)
    print("Volley-only A/B  —  max_bounces = 0 on both trackers")
    print("=" * 92)
    ls, _, ls_n = run("leastsq", max_bounces=0)
    ek, ek_sig, ek_n = run("ekf", max_bounces=0)
    print(f"\nTotal predictions: LS={ls_n}, EKF={ek_n}")
    print_table("Default config:", ls, ek, ek_sig)

    print("\n" + "=" * 92)
    print("EKF: bounce-handling off (we don't need it for volleys)")
    print("=" * 92)
    ek2, ek2_sig, _ = run("ekf", max_bounces=0, ekf_bounce_handling=False)
    print_table("Bounce handling off:", ls, ek2, ek2_sig)

    print("\n" + "=" * 92)
    print("EKF: bounce-handling off + tighter process noise (σ_a_xy=3, σ_a_z=2)")
    print("=" * 92)
    ek3, ek3_sig, _ = run("ekf", max_bounces=0, ekf_bounce_handling=False,
                          process_accel_std_xy=3.0, process_accel_std_z=2.0)
    print_table("Tighter process noise:", ls, ek3, ek3_sig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
