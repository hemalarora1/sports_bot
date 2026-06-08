"""Post-process J9 trace files and visualize in Viser.

Shows per throw:
  • Ball trajectory colored by rejection reason
  • Arm sweet-spot path during pre-commit tracking (blue)
  • Arm actual swing path from impact_sample data (orange)
  • Planned cubic Hermite trajectory (green)
  • Key markers: arm@commit, planned contact (q_strike), q_stop, actual contact

Usage (run from OpenSai root):
    python sports_bot/scripts/visualize_j9_throw.py \\
        sports_bot/logs/j9_20260608_030436.trace.jsonl

    python sports_bot/scripts/visualize_j9_throw.py \\
        sports_bot/logs/j9_20260608_030436.trace.jsonl --viser-port 8081
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

# ── repo root on sys.path ──────────────────────────────────────────────────────
_SCRIPT = Path(__file__).resolve()
REPO_ROOT = _SCRIPT.parents[2]  # OpenSai/
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_URDF_PATH = REPO_ROOT / "drivers/FrankaPanda/model/panda_arm.urdf"
_OFFSET_L7 = np.array([0.0, 0.0, 0.39])

# Colours match ball_tracker_test.py
_REJECT_COLORS: dict[str, tuple] = {
    "":                     (0,   220,  60),   # prediction OK → green
    "insufficient_history": (130, 130, 130),   # grey
    "not_incoming":         (200, 140,   0),   # amber
    "would_bounce":         (210,  30,  30),   # red
    "tti_too_long":         ( 30,  80, 210),   # blue
    "tti_too_short":        (200,  30, 200),   # magenta
    "past_plane":           (110,   0, 160),   # purple
    "on_floor":             (100,  60,  20),   # brown
}
_DEFAULT_REJECT_COLOR = (120, 120, 120)


# ── data structures ───────────────────────────────────────────────────────────
@dataclass
class BallTick:
    t_mono: float
    ball_pos: np.ndarray           # world frame (3,)
    reject_reason: str
    intercept_pos_W: Optional[np.ndarray] = None
    tti_s: Optional[float] = None
    q_arm_deg: Optional[np.ndarray] = None  # interpolated from goal_write


@dataclass
class SwingSample:
    elapsed_s: float
    t_mono: float
    q_goal_deg: np.ndarray
    q_actual_deg: np.ndarray
    phase: str


@dataclass
class ThrowData:
    throw_id: int
    ball_ticks: List[BallTick] = field(default_factory=list)
    arm_tracking: List[tuple] = field(default_factory=list)  # (t_mono, q_deg)
    commit: Optional[dict] = None
    commit_t: Optional[float] = None
    swing_samples: List[SwingSample] = field(default_factory=list)
    impact_done: Optional[dict] = None
    t_W_A: Optional[np.ndarray] = None    # arm base in world frame


# ── trace parser ─────────────────────────────────────────────────────────────
def _parse_trace(path: Path) -> List[ThrowData]:
    events: List[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
                # json.loads may leave Infinity as a float — that's fine
                events.append(ev)
            except json.JSONDecodeError:
                continue

    # Locate commit_start events; each defines one throw
    commit_indices = [i for i, ev in enumerate(events) if ev.get("event") == "commit_start"]
    if not commit_indices:
        return []

    throws: List[ThrowData] = []
    t_W_A_global: Optional[np.ndarray] = None

    for ci, commit_idx in enumerate(commit_indices):
        commit_ev = events[commit_idx]
        throw_id = int(commit_ev.get("throw_id", ci + 1))
        throw = ThrowData(throw_id=throw_id)
        throw.commit = commit_ev
        throw.commit_t = float(commit_ev["t_mono"])

        # Derive arm base in world frame from strike_W − strike_A (R_W_A=I for yaw=0)
        cand = commit_ev.get("candidate", {})
        s_W = cand.get("strike_W")
        s_A = cand.get("strike_A")
        if s_W and s_A:
            throw.t_W_A = np.array(s_W, dtype=float) - np.array(s_A, dtype=float)
            if t_W_A_global is None:
                t_W_A_global = throw.t_W_A.copy()

        # Tracking phase: events between previous impact_done (or file start) and commit_start
        track_start = 0
        for j in range(commit_idx - 1, -1, -1):
            if events[j].get("event") == "impact_done":
                track_start = j + 1
                break

        for j in range(track_start, commit_idx):
            ev = events[j]
            event = ev.get("event", "")

            if event == "tracker_tick":
                pos = ev.get("raw_ball") or ev.get("sample_pos_W")
                if pos is None:
                    continue
                pos = np.array(pos, dtype=float)
                if pos.shape != (3,) or not np.all(np.isfinite(pos)):
                    continue
                reject = ev.get("reject_reason") or ""
                intercept_pos = None
                tti = None
                intercept = ev.get("intercept")
                if isinstance(intercept, dict):
                    pw = intercept.get("position_W")
                    if pw:
                        intercept_pos = np.array(pw, dtype=float)
                    tti = intercept.get("tti_s")
                throw.ball_ticks.append(BallTick(
                    t_mono=float(ev["t_mono"]),
                    ball_pos=pos,
                    reject_reason=reject,
                    intercept_pos_W=intercept_pos,
                    tti_s=float(tti) if tti is not None else None,
                ))

            elif event == "goal_write":
                q = ev.get("q_cmd_deg")
                if q is not None:
                    throw.arm_tracking.append((float(ev["t_mono"]), np.array(q, dtype=float)))

        # Swing phase: between commit_start and impact_done
        next_commit = commit_indices[ci + 1] if ci + 1 < len(commit_indices) else len(events)
        for j in range(commit_idx + 1, min(next_commit, len(events))):
            ev = events[j]
            event = ev.get("event", "")
            if event == "impact_sample":
                throw.swing_samples.append(SwingSample(
                    elapsed_s=float(ev.get("elapsed_s", 0.0)),
                    t_mono=float(ev["t_mono"]),
                    q_goal_deg=np.array(ev.get("q_goal_deg", [0.0]*7), dtype=float),
                    q_actual_deg=np.array(ev.get("q_actual_deg", [0.0]*7), dtype=float),
                    phase=str(ev.get("phase", "")),
                ))
            elif event == "impact_done":
                throw.impact_done = ev
                break

        throws.append(throw)

    # Interpolate arm position into each ball tick
    for throw in throws:
        if throw.t_W_A is None and t_W_A_global is not None:
            throw.t_W_A = t_W_A_global.copy()
        if not throw.arm_tracking:
            continue
        gw_times = np.array([gw[0] for gw in throw.arm_tracking])
        for tick in throw.ball_ticks:
            idx = int(np.searchsorted(gw_times, tick.t_mono))
            idx = min(idx, len(gw_times) - 1)
            tick.q_arm_deg = throw.arm_tracking[idx][1]

    return throws


# ── FK ────────────────────────────────────────────────────────────────────────
def _build_chain():
    import ikpy.chain
    return ikpy.chain.Chain.from_urdf_file(str(_URDF_PATH), base_elements=["link0"])


def _fk_sweet_spot_W(chain, q_deg: np.ndarray, t_W_A: np.ndarray) -> np.ndarray:
    T = chain.forward_kinematics(np.concatenate([[0.0], np.deg2rad(q_deg)]))
    p_A = T[:3, 3] + T[:3, :3] @ _OFFSET_L7
    return t_W_A + p_A


def _fk_batch(chain, q_list: list, t_W_A: np.ndarray, step: int = 1) -> np.ndarray:
    pts = [_fk_sweet_spot_W(chain, q, t_W_A) for q in q_list[::step]]
    return np.array(pts, dtype=np.float32) if pts else np.zeros((0, 3), dtype=np.float32)


# ── Hermite planned trajectory ────────────────────────────────────────────────
def _planned_path(chain, t_W_A: np.ndarray, cand: dict, q_cur_deg: np.ndarray,
                  n_approach: int = 35, n_decel: int = 18) -> np.ndarray:
    """Sample the planned cubic Hermite (approach + decel) in world frame."""
    q_cur    = np.deg2rad(q_cur_deg)
    q_strike = np.deg2rad(np.array(cand["q_strike_deg"], dtype=float))
    q_stop   = np.deg2rad(np.array(cand["q_stop_deg"],   dtype=float))
    qdot     = np.deg2rad(np.array(cand["qdot_strike_deg_s"], dtype=float))
    T_approach = float(cand.get("strike_time_s", 0.5))
    T_decel    = float(cand.get("decel_s", 0.20))

    def _hermite(q0, q1, v0, v1, T, n):
        ts = np.linspace(0.0, T, n)
        out = []
        for t in ts:
            tau = t / T
            h00 = 2*tau**3 - 3*tau**2 + 1
            h10 = tau**3  - 2*tau**2  + tau
            h01 = -2*tau**3 + 3*tau**2
            h11 = tau**3  - tau**2
            out.append(h00*q0 + h10*(T*v0) + h01*q1 + h11*(T*v1))
        return out

    approach_q = _hermite(q_cur,    q_strike, np.zeros(7), qdot, max(0.01, T_approach), n_approach)
    decel_q    = _hermite(q_strike, q_stop,   qdot, np.zeros(7), max(0.01, T_decel),    n_decel)
    all_q = approach_q + decel_q[1:]  # skip duplicate at seam

    pts = []
    for q in all_q:
        T_mat = chain.forward_kinematics(np.concatenate([[0.0], q]))
        p_A = T_mat[:3, 3] + T_mat[:3, :3] @ _OFFSET_L7
        pts.append(t_W_A + p_A)
    return np.array(pts, dtype=np.float32)


# ── geometry helpers ──────────────────────────────────────────────────────────
def _segs(pts: np.ndarray) -> np.ndarray:
    """Nx3 polyline → (N-1, 2, 3) line-segment array for add_line_segments."""
    return np.stack([pts[:-1], pts[1:]], axis=1)


# ── Viser UI ──────────────────────────────────────────────────────────────────
def _run_viser(throws: List[ThrowData], chain, port: int) -> None:
    import viser

    server = viser.ViserServer(port=port)
    print(f"[j9-viz] Viser running at http://localhost:{port}")
    print(f"[j9-viz] {len(throws)} throw(s) loaded — select in the GUI")

    # ── static scene ──────────────────────────────────────────────────────────
    server.scene.add_frame("/world", show_axes=True, axes_length=0.3, axes_radius=0.01)
    server.scene.add_grid(
        "/floor", width=5.0, height=4.0, plane="xy",
        cell_size=0.5, section_size=1.0, position=(0.0, 0.0, 0.0),
    )

    # Strike plane from first throw
    strike_x = -0.22
    if throws and throws[0].commit:
        sw = (throws[0].commit.get("candidate") or {}).get("strike_W")
        if sw:
            strike_x = float(sw[0])
    server.scene.add_box(
        "/strike_plane",
        position=(strike_x, 0.0, 0.75),
        dimensions=(0.005, 3.0, 1.5),
        color=(255, 215, 0), opacity=0.25,
    )

    # Arm base origin marker
    if throws and throws[0].t_W_A is not None:
        ab = throws[0].t_W_A
        server.scene.add_frame(
            "/arm_base",
            position=(float(ab[0]), float(ab[1]), float(ab[2])),
            show_axes=True, axes_length=0.12, axes_radius=0.005,
        )

    # ── GUI ───────────────────────────────────────────────────────────────────
    with server.gui.add_folder("Throw"):
        _throw_labels = [
            (f"Throw {th.throw_id}  "
             f"tti={((th.commit or {}).get('candidate') or {}).get('tti_original_s', 0):.2f}s  "
             f"err={(((th.impact_done or {}).get('result') or {}).get('strike_err_deg', 0)):.1f}°")
            for th in throws
        ]
        throw_dd = server.gui.add_dropdown(
            "Select throw",
            options=_throw_labels,
            initial_value=_throw_labels[0],
        )
        n0 = max(0, len(throws[0].ball_ticks) - 1)
        tick_slider = server.gui.add_slider(
            "Ball tick (tracking phase)", min=0, max=n0, step=1, initial_value=n0,
        )

    with server.gui.add_folder("Display"):
        cb_ball       = server.gui.add_checkbox("Ball trajectory",               True)
        cb_reject_col = server.gui.add_checkbox("Rejection-reason colours",      True)
        cb_tracking   = server.gui.add_checkbox("Arm tracking path (pre-commit)", True)
        cb_swing      = server.gui.add_checkbox("Arm actual swing path",          True)
        cb_planned    = server.gui.add_checkbox("Planned Hermite path",           True)
        cb_intercepts = server.gui.add_checkbox("Intercept cloud (all ticks)",    False)

    with server.gui.add_folder("Rejection legend"):
        server.gui.add_text("● green",   "prediction OK",        disabled=True)
        server.gui.add_text("● grey",    "insufficient_history", disabled=True)
        server.gui.add_text("● amber",   "not_incoming",         disabled=True)
        server.gui.add_text("● red",     "would_bounce",         disabled=True)
        server.gui.add_text("● blue",    "tti_too_long",         disabled=True)
        server.gui.add_text("● magenta", "tti_too_short",        disabled=True)
        server.gui.add_text("● purple",  "past_plane",           disabled=True)

    with server.gui.add_folder("Commit / Impact"):
        ui_commit_tti  = server.gui.add_text("Commit TTI",       "—", disabled=True)
        ui_strike_s    = server.gui.add_text("Strike time",      "—", disabled=True)
        ui_tmin        = server.gui.add_text("Tmin",             "—", disabled=True)
        ui_margin      = server.gui.add_text("Margin",           "—", disabled=True)
        ui_strike_err  = server.gui.add_text("Strike err",       "—", disabled=True)
        ui_v_desired   = server.gui.add_text("V desired [x,z]",  "—", disabled=True)
        ui_v_achieved  = server.gui.add_text("V IK-achieved",    "—", disabled=True)
        ui_v_measured  = server.gui.add_text("V measured",       "—", disabled=True)

    with server.gui.add_folder("Current ball tick"):
        ui_tick_t      = server.gui.add_text("t_mono",           "—", disabled=True)
        ui_ball_pos    = server.gui.add_text("Ball pos (W)",     "—", disabled=True)
        ui_pred_tti    = server.gui.add_text("Predicted TTI",    "—", disabled=True)
        ui_reject      = server.gui.add_text("Reject reason",    "—", disabled=True)

    # ── mutable scene handles ─────────────────────────────────────────────────
    _handles: dict = {}
    _state: dict = {"throw_idx": 0, "tick_idx": n0}

    def _clear() -> None:
        for h in list(_handles.values()):
            try:
                h.remove()
            except Exception:
                pass
        _handles.clear()

    def _redraw() -> None:  # noqa: C901
        _clear()
        ti_throw = _state["throw_idx"]
        throw = throws[ti_throw]
        t_W_A = throw.t_W_A if throw.t_W_A is not None else np.zeros(3)
        cand  = (throw.commit or {}).get("candidate", {})
        ti    = max(0, min(_state["tick_idx"], len(throw.ball_ticks) - 1))

        # ── info panels ───────────────────────────────────────────────────────
        tti_orig = cand.get("tti_original_s") or cand.get("tti_aged_s")
        ui_commit_tti.value  = f"{tti_orig:.3f}s"          if tti_orig  is not None else "—"
        ui_strike_s.value    = f"{cand.get('strike_time_s', 0):.3f}s"
        ui_tmin.value        = f"{cand.get('min_strike_s',  0):.3f}s"
        ui_margin.value      = f"{cand.get('budget_margin_s', 0):+.3f}s"

        dv = cand.get("desired_v_W")
        av = cand.get("achieved_v_W")
        if dv:
            ui_v_desired.value  = f"[{dv[0]:+.2f}, {dv[1]:+.2f}, {dv[2]:+.2f}] m/s"
        if av:
            ui_v_achieved.value = f"[{av[0]:+.2f}, {av[1]:+.2f}, {av[2]:+.2f}] m/s"

        if throw.impact_done:
            res = (throw.impact_done or {}).get("result", {})
            ui_strike_err.value = f"{res.get('strike_err_deg', 0):.1f}°"
            mv = res.get("measured_v_W")
            if mv:
                ui_v_measured.value = f"[{mv[0]:+.2f}, {mv[1]:+.2f}, {mv[2]:+.2f}] m/s"

        # ── ball trajectory ───────────────────────────────────────────────────
        if cb_ball.value and throw.ball_ticks:
            positions = np.array([tk.ball_pos for tk in throw.ball_ticks], dtype=np.float32)
            if cb_reject_col.value:
                n = len(positions)
                colors = np.zeros((n, 3), dtype=np.uint8)
                for j, tk in enumerate(throw.ball_ticks):
                    colors[j] = _REJECT_COLORS.get(tk.reject_reason, _DEFAULT_REJECT_COLOR)
            else:
                colors = np.tile(np.uint8([160, 160, 160]), (len(positions), 1))
            _handles["ball_pts"] = server.scene.add_point_cloud(
                "/throw/ball_pts", points=positions, colors=colors, point_size=0.016,
            )
            if len(positions) >= 2:
                _handles["ball_line"] = server.scene.add_line_segments(
                    "/throw/ball_line", points=_segs(positions),
                    colors=(140, 140, 140), line_width=1.5,
                )

        # ── current ball tick ──────────────────────────────────────────────────
        if throw.ball_ticks:
            tick = throw.ball_ticks[ti]
            _handles["ball_cur"] = server.scene.add_icosphere(
                "/throw/ball_cur", radius=0.055, color=(255, 60, 60),
                position=tuple(tick.ball_pos.tolist()),
            )
            ui_tick_t.value   = f"{tick.t_mono:.3f}"
            ui_ball_pos.value = "[{:+.3f}, {:+.3f}, {:+.3f}]".format(*tick.ball_pos.tolist())
            ui_pred_tti.value = f"{tick.tti_s:.3f}s" if tick.tti_s is not None else "—"
            ui_reject.value   = tick.reject_reason or "OK"

            # Arm at this tick (from interpolated goal_write)
            if tick.q_arm_deg is not None:
                p_arm = _fk_sweet_spot_W(chain, tick.q_arm_deg, t_W_A)
                _handles["arm_cur"] = server.scene.add_icosphere(
                    "/throw/arm_cur", radius=0.045, color=(80, 150, 255),
                    position=tuple(p_arm.tolist()),
                )

        # ── intercept prediction cloud ─────────────────────────────────────────
        if cb_intercepts.value and throw.ball_ticks:
            pred_pts, pred_ttis = [], []
            for tk in throw.ball_ticks:
                if tk.intercept_pos_W is not None and tk.tti_s is not None:
                    pred_pts.append(tk.intercept_pos_W)
                    pred_ttis.append(tk.tti_s)
            if pred_pts:
                pts_arr = np.array(pred_pts, dtype=np.float32)
                t_norm = np.clip(np.array(pred_ttis) / 1.5, 0, 1).astype(np.float32)
                col = np.zeros((len(t_norm), 3), dtype=np.uint8)
                col[:, 0] = (255 * t_norm).astype(np.uint8)
                col[:, 2] = (255 * (1 - t_norm)).astype(np.uint8)
                _handles["intercept_cloud"] = server.scene.add_point_cloud(
                    "/throw/intercept_cloud", points=pts_arr, colors=col, point_size=0.018,
                )

        # ── commit: frozen intercept (planned contact point) ──────────────────
        s_W = cand.get("strike_W")
        if s_W:
            _handles["planned_contact"] = server.scene.add_icosphere(
                "/throw/planned_contact", radius=0.06, color=(0, 255, 80),
                position=tuple(float(v) for v in s_W),
            )
            _handles["planned_contact_lbl"] = server.scene.add_label(
                "/throw/planned_contact/lbl", text="planned contact",
                position=(float(s_W[0]), float(s_W[1]), float(s_W[2]) + 0.11),
            )

        # ── arm pre-commit tracking path ───────────────────────────────────────
        if cb_tracking.value and throw.arm_tracking:
            step = max(1, len(throw.arm_tracking) // 120)
            tracking_pts = _fk_batch(chain, [gw[1] for gw in throw.arm_tracking], t_W_A, step)
            if len(tracking_pts) >= 2:
                _handles["arm_tracking"] = server.scene.add_line_segments(
                    "/throw/arm_tracking", points=_segs(tracking_pts),
                    colors=(80, 130, 220), line_width=2.0,
                )

        # ── arm actual swing path ─────────────────────────────────────────────
        if cb_swing.value and throw.swing_samples:
            step = max(1, len(throw.swing_samples) // 120)
            swing_pts = _fk_batch(chain, [s.q_actual_deg for s in throw.swing_samples], t_W_A, step)
            if len(swing_pts) >= 2:
                _handles["arm_swing"] = server.scene.add_line_segments(
                    "/throw/arm_swing", points=_segs(swing_pts),
                    colors=(255, 110, 40), line_width=3.0,
                )

            # q_strike position
            if cand.get("q_strike_deg"):
                p_qs = _fk_sweet_spot_W(chain, np.array(cand["q_strike_deg"], dtype=float), t_W_A)
                _handles["q_strike"] = server.scene.add_icosphere(
                    "/throw/q_strike", radius=0.05, color=(255, 210, 0),
                    position=tuple(p_qs.tolist()),
                )
                _handles["q_strike_lbl"] = server.scene.add_label(
                    "/throw/q_strike/lbl", text="q_strike",
                    position=(float(p_qs[0]), float(p_qs[1]), float(p_qs[2]) + 0.10),
                )

            # q_stop position
            if cand.get("q_stop_deg"):
                p_stop = _fk_sweet_spot_W(chain, np.array(cand["q_stop_deg"], dtype=float), t_W_A)
                _handles["q_stop"] = server.scene.add_icosphere(
                    "/throw/q_stop", radius=0.04, color=(200, 80, 210),
                    position=tuple(p_stop.tolist()),
                )
                _handles["q_stop_lbl"] = server.scene.add_label(
                    "/throw/q_stop/lbl", text="q_stop",
                    position=(float(p_stop[0]), float(p_stop[1]), float(p_stop[2]) + 0.10),
                )

            # Actual end-of-swing arm position (from impact_done diagnostic)
            if throw.impact_done:
                q_final = (throw.impact_done.get("diagnostic") or {}).get("q_deg")
                if q_final:
                    p_final = _fk_sweet_spot_W(chain, np.array(q_final, dtype=float), t_W_A)
                    res = (throw.impact_done.get("result") or {})
                    err_lbl = f"contact (err={res.get('strike_err_deg', 0):.1f}°)"
                    _handles["q_final"] = server.scene.add_icosphere(
                        "/throw/q_final", radius=0.045, color=(255, 50, 50),
                        position=tuple(p_final.tolist()),
                    )
                    _handles["q_final_lbl"] = server.scene.add_label(
                        "/throw/q_final/lbl", text=err_lbl,
                        position=(float(p_final[0]), float(p_final[1]), float(p_final[2]) + 0.10),
                    )

        # ── planned Hermite trajectory ─────────────────────────────────────────
        if cb_planned.value and cand.get("q_strike_deg"):
            q_cur_at_commit = (throw.commit.get("diagnostic") or {}).get("q_deg")
            if q_cur_at_commit is not None:
                try:
                    planned_pts = _planned_path(
                        chain, t_W_A, cand, np.array(q_cur_at_commit, dtype=float),
                    )
                    if len(planned_pts) >= 2:
                        _handles["planned"] = server.scene.add_line_segments(
                            "/throw/planned", points=_segs(planned_pts),
                            colors=(0, 210, 140), line_width=2.5,
                        )
                    # Arm position at commit start
                    p_arm_commit = _fk_sweet_spot_W(
                        chain, np.array(q_cur_at_commit, dtype=float), t_W_A,
                    )
                    _handles["arm_at_commit"] = server.scene.add_icosphere(
                        "/throw/arm_at_commit", radius=0.04, color=(0, 200, 100),
                        position=tuple(p_arm_commit.tolist()),
                    )
                    _handles["arm_at_commit_lbl"] = server.scene.add_label(
                        "/throw/arm_at_commit/lbl", text="arm@commit",
                        position=(float(p_arm_commit[0]), float(p_arm_commit[1]),
                                  float(p_arm_commit[2]) + 0.10),
                    )
                except Exception:
                    pass

    # ── callbacks ──────────────────────────────────────────────────────────────
    @throw_dd.on_update
    def _on_throw(_):
        label = throw_dd.value
        idx = next((i for i, l in enumerate(_throw_labels) if l == label), 0)
        _state["throw_idx"] = idx
        n = max(0, len(throws[idx].ball_ticks) - 1)
        tick_slider.max = n
        tick_slider.value = n
        _state["tick_idx"] = n
        _redraw()

    @tick_slider.on_update
    def _on_tick(_):
        _state["tick_idx"] = int(tick_slider.value)
        _redraw()

    for _cb in (cb_ball, cb_reject_col, cb_tracking, cb_swing, cb_planned, cb_intercepts):
        @_cb.on_update
        def _(_): _redraw()

    _redraw()

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[j9-viz] bye")


# ── CLI ───────────────────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Visualize J9 throw trace (.trace.jsonl) in Viser.",
    )
    parser.add_argument("trace", help="Path to a j9_*.trace.jsonl file.")
    parser.add_argument("--viser-port", type=int, default=8081,
                        help="Viser web port (default 8081, avoids clash with foam-ball viz at 8080).")
    parser.add_argument("--no-viser", action="store_true",
                        help="Print throw summary and exit without launching Viser.")
    args = parser.parse_args()

    path = Path(args.trace)
    if not path.exists():
        print(f"[j9-viz] not found: {path}", file=sys.stderr)
        return 1

    print(f"[j9-viz] parsing {path.name} ...")
    throws = _parse_trace(path)
    if not throws:
        print("[j9-viz] no throws found in trace file", file=sys.stderr)
        return 1

    print(f"[j9-viz] {len(throws)} throw(s):")
    for th in throws:
        cand = (th.commit or {}).get("candidate", {})
        res  = (th.impact_done or {}).get("result", {})
        mv   = res.get("measured_v_W") or [0, 0, 0]
        tti  = cand.get("tti_original_s") or cand.get("tti_aged_s") or 0
        print(
            f"  #{th.throw_id:2d}  "
            f"ball_ticks={len(th.ball_ticks):4d}  "
            f"swing_samples={len(th.swing_samples):4d}  "
            f"commit_tti={tti:.3f}s  "
            f"strike_err={res.get('strike_err_deg', '?'):.1f}°  "
            f"v_meas=[{mv[0]:+.2f},{mv[1]:+.2f},{mv[2]:+.2f}]"
            if th.impact_done else
            f"  #{th.throw_id:2d}  ball_ticks={len(th.ball_ticks):4d}  (no impact_done)"
        )

    if args.no_viser:
        return 0

    if not _URDF_PATH.exists():
        print(f"[j9-viz] URDF not found at {_URDF_PATH} — run from OpenSai root", file=sys.stderr)
        return 1

    try:
        import viser  # noqa: F401
    except ImportError:
        print("[j9-viz] viser not installed: pip install viser", file=sys.stderr)
        return 1

    import os
    os.chdir(REPO_ROOT)  # ensures URDF_PATH resolves for ikpy

    print("[j9-viz] building FK chain ...")
    chain = _build_chain()
    print("[j9-viz] chain ready")

    _run_viser(throws, chain, args.viser_port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
