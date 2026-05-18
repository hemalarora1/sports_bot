"""
debug_tracker.py — validate BallTracker with live OptiTrack data.

No robot commands are sent. Safe to run at any time while the streamer is up.

Usage (from the sports_bot directory):
    python debug_tracker.py

Workflow:
  1. Script starts a 5-second countdown immediately.
  2. Throw the ball any time during that window.
  3. After the countdown a 3-D plot pops up showing:
       - Blue dots  : ALL actual OptiTrack positions for the 5-second window
       - Red line   : predicted ballistic arc (fit from moment of peak speed),
                      extending to the strike plane so you can compare directly
       - Green X    : predicted intercept on the strike plane
       - Grey plane : strike plane (x = STRIKE_PLANE_X)
  4. Close the plot window and a new 5-second countdown starts automatically.

Adjust the configuration block below as needed.
"""

import sys
import time
from typing import List, Optional, Tuple

sys.path.insert(0, ".")

import matplotlib.pyplot as plt
import numpy as np
import redis
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3-d projection

from state_machine.ball_tracker import BallTracker, Intercept
from state_machine.config import BallTrackerConfig
from state_machine.redis_keys import RedisKeys
from enum import Enum, auto

# ── configuration ──────────────────────────────────────────────────────────────
BALL_ID        = 8      # rigid body ID in Motive
REDIS_HOST     = "localhost"
REDIS_PORT     = 6379
STRIKE_PLANE_X = 0.60   # metres — FSM strike plane
POLL_HZ        = 120    # how fast we poll Redis (match Motive frame rate)
RECORD_SECONDS = 5      # record for this many seconds, then plot
GRAVITY        = 9.81   # m/s²
# ───────────────────────────────────────────────────────────────────────────────


class _State(Enum):
    RECORDING = auto()   # collecting samples until countdown ends
    PLOTTING  = auto()   # rendering the plot (blocking)


# ── Redis + tracker setup ──────────────────────────────────────────────────────
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

keys = RedisKeys(ball_source="optitrack")
keys.ball = type(keys.ball)(optitrack_rigid_body_id=BALL_ID)

cfg     = BallTrackerConfig()
tracker = BallTracker(r, keys, cfg)


# ── helpers ────────────────────────────────────────────────────────────────────

def _read_raw() -> Optional[str]:
    return r.get(f"sai2::optitrack::rigid_body_pos::{BALL_ID}")


def _ballistic_arc(
    p0: np.ndarray,
    v0: np.ndarray,
    t_max: float,
    n: int = 200,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (x, y, z) arrays for the predicted arc over [0, t_max]."""
    ts  = np.linspace(0, t_max, n)
    xs  = p0[0] + v0[0] * ts
    ys  = p0[1] + v0[1] * ts
    zs  = p0[2] + v0[2] * ts - 0.5 * GRAVITY * ts * ts
    return xs, ys, zs


def _plot_throw(
    actual: List[np.ndarray],
    fit_p0: np.ndarray,
    fit_v0: np.ndarray,
    intercept: Optional[Intercept],
    throw_num: int,
) -> None:
    """Render the 3-D comparison plot and block until the window is closed."""

    actual_arr = np.array(actual)   # shape (N, 3)

    # Predicted arc from peak-speed moment to strike plane (or 1.5 s fallback).
    t_arc = intercept.time_to_impact + 0.3 if intercept else 1.5
    pred_x, pred_y, pred_z = _ballistic_arc(fit_p0, fit_v0, t_arc)

    fig = plt.figure(figsize=(10, 7))
    ax  = fig.add_subplot(111, projection="3d")

    # 1. Predicted arc — red solid line, drawn first.
    # ax.plot(pred_x, pred_y, pred_z, "r-", linewidth=1.5, alpha=0.6,
    #         label="Predicted arc")

    # 2. Actual positions — small blue dots, drawn on top.
    n = len(actual_arr)
    ax.scatter(
        actual_arr[:, 0], actual_arr[:, 1], actual_arr[:, 2],
        c="steelblue", s=10, depthshade=False, edgecolors="white", linewidths=0.3,
        label=f"Actual positions ({n} pts)",
    )

    # 3. Green X — predicted intercept on the strike plane.
    if intercept:
        ip = intercept.position
        ax.scatter(
            [ip[0]], [ip[1]], [ip[2]],
            c="lime", s=200, marker="X", depthshade=False, edgecolors="darkgreen", linewidths=1.5,
            label=f"Intercept  z={ip[2]:+.3f}m  t={intercept.time_to_impact:.3f}s",
        )

    # 4. Strike plane (grey, semi-transparent).
    all_pts = np.vstack([actual_arr, np.column_stack([pred_x, pred_y, pred_z])])
    y_lo = all_pts[:, 1].min() - 0.2
    y_hi = all_pts[:, 1].max() + 0.2
    z_lo = max(all_pts[:, 2].min() - 0.1, 0.0)
    z_hi = all_pts[:, 2].max() + 0.2
    YY, ZZ = np.meshgrid(np.linspace(y_lo, y_hi, 4), np.linspace(z_lo, z_hi, 4))
    ax.plot_surface(np.full_like(YY, STRIKE_PLANE_X), YY, ZZ,
                    alpha=0.12, color="silver")
    ax.text(STRIKE_PLANE_X, y_lo, z_hi, f"strike x={STRIKE_PLANE_X}m",
            fontsize=8, color="grey")

    spd = float(np.linalg.norm(fit_v0))
    ax.set_xlabel("X — forward (m)")
    ax.set_ylabel("Y — lateral (m)")
    ax.set_zlabel("Z — up (m)")
    ax.set_title(
        f"Round #{throw_num}  |  peak v=({fit_v0[0]:+.2f}, {fit_v0[1]:+.2f}, {fit_v0[2]:+.2f}) m/s  speed={spd:.2f}"
    )
    ax.legend(loc="upper left", fontsize=8)

    # Equal aspect on all axes so the parabola isn't distorted.
    ranges = all_pts.max(axis=0) - all_pts.min(axis=0)
    mid    = (all_pts.max(axis=0) + all_pts.min(axis=0)) / 2
    half   = max(ranges.max() / 2, 0.3)
    ax.set_xlim(mid[0] - half, mid[0] + half)
    ax.set_ylim(mid[1] - half, mid[1] + half)
    ax.set_zlim(max(mid[2] - half, 0), mid[2] + half)

    plt.tight_layout()
    plt.show()   # blocks until the window is closed


# ── main loop ─────────────────────────────────────────────────────────────────

def _new_round(round_num: int) -> tuple:
    """Print the round header and return fresh recording state."""
    print(f"\n── Round #{round_num} ── recording for {RECORD_SECONDS}s — throw now! ──")
    HEADER = (
        f"  {'secs_left':>9}  {'pos_x':>7} {'pos_y':>7} {'pos_z':>7}"
        f"  {'speed':>6}  {'incoming':<8}  {'z_impact':>9}"
    )
    DIVIDER = "  " + "-" * (len(HEADER) - 2)
    print(HEADER)
    print(DIVIDER)
    return (
        [],       # flight_positions
        None,     # fit_at_peak  (p0, v0) taken when speed is highest
        None,     # intercept_best
        0.0,      # peak_speed
        time.perf_counter(),  # round_start_t
        HEADER,
        DIVIDER,
        0,        # line counter
    )


def main() -> None:
    print(f"\nBall tracker debug  —  ID={BALL_ID}  strike plane x={STRIKE_PLANE_X} m")
    print(f"Redis: {REDIS_HOST}:{REDIS_PORT}")
    print(f"Each round records {RECORD_SECONDS}s then plots. Ctrl-C to quit.")

    dt        = 1.0 / POLL_HZ
    round_num = 1

    (flight_positions, fit_at_peak, intercept_best,
     peak_speed, round_start_t, HEADER, DIVIDER, line) = _new_round(round_num)

    try:
        while True:
            now     = time.perf_counter()
            raw_now = _read_raw()

            # ---- update tracker ----------------------------------------------
            tracker.update()
            pos = tracker.latest_position()

            secs_left = RECORD_SECONDS - (now - round_start_t)

            if pos is None:
                print(f"  {secs_left:>9.1f}  [ball not visible]")
                time.sleep(dt)
                # still need to check end-of-round below
            else:
                fit       = tracker._fit_state()
                speed     = float(np.linalg.norm(fit[1])) if fit else 0.0
                incoming  = tracker.is_incoming()
                intercept = tracker.predict_intercept(STRIKE_PLANE_X) if fit else None

                # Record every sample during the round
                flight_positions.append(pos.copy())

                # Keep the fit from the moment of highest speed (mid-throw)
                if fit and speed > peak_speed:
                    peak_speed     = speed
                    fit_at_peak    = (fit[0].copy(), fit[1].copy())
                    intercept_best = intercept

                z_str = f"{intercept.position[2]:+.3f}m" if intercept else "      ---"
                if line % 40 == 0 and line > 0:
                    print(HEADER)
                    print(DIVIDER)
                print(
                    f"  {secs_left:>9.1f}  {pos[0]:>+7.3f} {pos[1]:>+7.3f} {pos[2]:>+7.3f}"
                    f"  {speed:>6.2f}  {str(incoming):<8}  {z_str:>9}"
                )
                line += 1

            # ---- end of round ------------------------------------------------
            if RECORD_SECONDS - (now - round_start_t) <= 0:
                print(f"\n  Round #{round_num} done — {len(flight_positions)} samples collected.")
                if fit_at_peak and len(flight_positions) >= 3:
                    p0, v0 = fit_at_peak
                    _plot_throw(flight_positions, p0, v0, intercept_best, round_num)
                else:
                    print("  [not enough movement to plot — was the ball tracked?]")

                round_num += 1
                tracker.reset()
                (flight_positions, fit_at_peak, intercept_best,
                 peak_speed, round_start_t, HEADER, DIVIDER, line) = _new_round(round_num)

            time.sleep(dt)

    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
