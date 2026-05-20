"""Pickleball state machine for the mobile-manipulator Panda.

Reads ball position from OpenSai sim (object_pose) or OptiTrack (NatNet
streamer) via Redis, predicts the intercept on a fixed strike plane, plans a
swing, and commands the racket sweet-spot pose / mobile base pose through the
controller-side Redis keys (see redis_keys.py for the available backends).

Run with:
    python -m sports_bot.state_machine.pickleball_fsm

Optional CLI overrides:
    --robot-backend {opensai,cs225a}
    --ball-source   {opensai,optitrack}
    --config-file   <opensai-config-file-name>   # only checked when robot-backend=opensai
"""

from __future__ import annotations

import argparse
import json
import math
import signal
import sys
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

import numpy as np
import redis

from .ball_tracker import BallTracker, Intercept
from .config import PickleballConfig
from .redis_keys import RedisKeys
from .swing_planner import SwingPlan, SwingPlanner


# ---------- States -------------------------------------------------------------

class State(Enum):
    INIT = auto()         # Drive base + racket to the home / ready pose.
    READY = auto()        # Hold ready pose, wait for an incoming ball.
    TRACK = auto()        # Ball detected and incoming; gather samples.
    APPROACH = auto()     # Have a swing plan; drive to wind-up pose.
    SWING = auto()        # Execute the swing through the strike point.
    RECOVER = auto()      # Hold follow-through, then return to ready.
    SAFE_STOP = auto()    # Error: hold a safe pose and bail.


# ---------- Robot Redis adapter ------------------------------------------------

class RobotRedisAdapter:
    """Thin wrapper over Redis that exposes a uniform interface to the FSM
    regardless of which robot backend is in use."""

    def __init__(self, redis_client: redis.Redis, keys: RedisKeys):
        self._r = redis_client
        self._keys = keys

    # ----- reads

    def read_racket_pose(self) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """Return (position, rotation) of the racket sweet-spot in world frame,
        or None if not available."""
        try:
            if self._keys.robot_backend == "opensai":
                pos = np.array(json.loads(self._r.get(self._keys.opensai.current_position)))
                rot = np.array(json.loads(self._r.get(self._keys.opensai.current_orientation)))
            else:
                pos = np.array(json.loads(self._r.get(self._keys.cs225a.racket_current_position)))
                rot = np.array(json.loads(self._r.get(self._keys.cs225a.racket_current_orientation)))
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
        if pos.shape != (3,) or rot.shape != (3, 3):
            return None
        return pos, rot

    def read_base_pose(self) -> Optional[np.ndarray]:
        if self._keys.robot_backend != "cs225a":
            return None
        raw = self._r.get(self._keys.cs225a.base_current_pose)
        if raw is None:
            return None
        try:
            pose = np.array(json.loads(raw))
        except (ValueError, json.JSONDecodeError):
            return None
        return pose if pose.shape == (3,) else None

    # ----- writes

    def write_racket_goal(
        self,
        position: np.ndarray,
        orientation: np.ndarray,
        linear_velocity: Optional[np.ndarray] = None,
    ) -> None:
        # Always publish a velocity goal so the controller doesn't see a stale
        # impact velocity after a SWING transitions back to RECOVER / READY.
        velocity = linear_velocity if linear_velocity is not None else np.zeros(3)
        if self._keys.robot_backend == "opensai":
            self._r.set(self._keys.opensai.goal_position, json.dumps(position.tolist()))
            self._r.set(self._keys.opensai.goal_orientation, json.dumps(orientation.tolist()))
            self._r.set(
                self._keys.opensai.goal_linear_velocity,
                json.dumps(velocity.tolist()),
            )
        else:
            self._r.set(
                self._keys.cs225a.racket_goal_position,
                json.dumps(position.tolist()),
            )
            self._r.set(
                self._keys.cs225a.racket_goal_orientation,
                json.dumps(orientation.tolist()),
            )
            self._r.set(
                self._keys.cs225a.racket_goal_linear_velocity,
                json.dumps(velocity.tolist()),
            )

    def write_base_goal(self, pose: np.ndarray) -> None:
        if self._keys.robot_backend == "cs225a":
            self._r.set(self._keys.cs225a.base_goal_pose, json.dumps(pose.tolist()))

    def read_joint_positions(self) -> Optional[np.ndarray]:
        if self._keys.robot_backend != "opensai":
            return None
        try:
            q = np.array(json.loads(self._r.get(self._keys.opensai.joint_current_position)))
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
        return q if q.shape == (7,) else None

    def switch_to_joint_controller(self) -> None:
        if self._keys.robot_backend != "opensai":
            return
        self._r.set(self._keys.opensai.active_controller, self._keys.opensai.joint_controller_name)

    def switch_to_cartesian_controller(self) -> None:
        if self._keys.robot_backend != "opensai":
            return
        self._r.set(self._keys.opensai.active_controller, self._keys.opensai.controller_name)

    def write_joint_goal(self, positions: np.ndarray) -> None:
        if self._keys.robot_backend != "opensai":
            return
        self._r.set(self._keys.opensai.joint_goal_position, json.dumps(positions.tolist()))

    def publish_state(self, state: State) -> None:
        if self._keys.robot_backend == "cs225a":
            self._r.set(self._keys.cs225a.fsm_state, state.name)

    # ----- guards

    def check_opensai_guards(self, expected_config: Optional[str]) -> None:
        if self._keys.robot_backend != "opensai":
            return
        if expected_config is None:
            return
        cfg = self._r.get(self._keys.opensai.config_file_name)
        if cfg is None:
            raise RuntimeError(
                f"Could not read OpenSai config file name from "
                f"{self._keys.opensai.config_file_name}; is OpenSai running?"
            )
        cfg = cfg.decode("utf-8") if isinstance(cfg, bytes) else cfg
        if cfg != expected_config:
            raise RuntimeError(
                f"OpenSai is running config '{cfg}', expected '{expected_config}'"
            )
        # Make sure the cartesian controller is the active one.
        active = self._r.get(self._keys.opensai.active_controller)
        active = active.decode("utf-8") if isinstance(active, bytes) else active
        if active != self._keys.opensai.controller_name:
            self._r.set(
                self._keys.opensai.active_controller,
                self._keys.opensai.controller_name,
            )


# ---------- Helpers ------------------------------------------------------------

def joints_close(q_now: np.ndarray, q_goal: np.ndarray, tol: float) -> bool:
    return bool(np.max(np.abs(q_now - q_goal)) < tol)


def pose_close(
    pos_now: np.ndarray, pos_goal: np.ndarray,
    rot_now: np.ndarray, rot_goal: np.ndarray,
    pos_tol: float, ori_tol: float,
) -> bool:
    return (
        np.linalg.norm(pos_now - pos_goal) < pos_tol
        and np.linalg.norm(rot_now - rot_goal) < ori_tol
    )


# ---------- The FSM ------------------------------------------------------------

class PickleballFSM:

    def __init__(
        self,
        redis_client: redis.Redis,
        keys: RedisKeys,
        cfg: PickleballConfig,
    ):
        self._cfg = cfg
        self._keys = keys
        self._robot = RobotRedisAdapter(redis_client, keys)
        self._tracker = BallTracker(redis_client, keys, cfg.tracker)
        self._planner = SwingPlanner(cfg.court, cfg.racket)

        self._state = State.INIT
        self._plan: Optional[SwingPlan] = None
        self._t_state_entered: float = time.perf_counter()
        self._t_swing_started: Optional[float] = None
        self._t_predicted_impact: Optional[float] = None
        self._n_hits = 0
        self._running = True

    # -------- lifecycle

    def request_stop(self) -> None:
        self._running = False

    def _enter(self, new_state: State) -> None:
        if new_state == self._state:
            return
        if self._cfg.fsm.verbose:
            print(f"[FSM] {self._state.name} -> {new_state.name}")
        self._state = new_state
        self._t_state_entered = time.perf_counter()
        self._robot.publish_state(new_state)
        if new_state == State.RECOVER:
            self._robot.switch_to_joint_controller()
        elif new_state in (State.APPROACH, State.SWING):
            self._robot.switch_to_cartesian_controller()

    def _time_in_state(self) -> float:
        return time.perf_counter() - self._t_state_entered

    # -------- per-state handlers

    def _step_init(self, racket_pos: np.ndarray, racket_rot: np.ndarray) -> None:
        ready = self._cfg.ready
        self._robot.write_base_goal(ready.base_pose)
        self._robot.write_joint_goal(ready.joint_positions)

        q = self._robot.read_joint_positions()
        if q is not None and joints_close(q, ready.joint_positions, self._cfg.fsm.joint_tol):
            self._enter(State.READY)

    def _step_ready(self) -> None:
        # Hold the ready pose. Look for an incoming ball.
        if not self._tracker.is_incoming():
            return
        intercept = self._tracker.predict_intercept(self._cfg.court.strike_plane_x)
        if intercept is None:
            return
        plan = self._planner.plan(intercept)
        if plan is None:
            return
        self._plan = plan
        self._t_predicted_impact = time.perf_counter() + plan.time_to_impact
        self._enter(State.TRACK)

    def _step_track(self) -> None:
        # Refine the swing plan as more ball samples come in.
        if self._tracker.time_since_last_seen() > self._cfg.fsm.ball_timeout_s:
            self._enter(State.RECOVER)
            return
        intercept = self._tracker.predict_intercept(self._cfg.court.strike_plane_x)
        if intercept is None:
            # Don't have a fresh prediction yet; keep the previous plan.
            return
        plan = self._planner.plan(intercept)
        if plan is None:
            self._enter(State.RECOVER)
            return
        self._plan = plan
        self._t_predicted_impact = time.perf_counter() + plan.time_to_impact
        self._enter(State.APPROACH)

    def _step_approach(self, racket_pos: np.ndarray, racket_rot: np.ndarray) -> None:
        plan = self._plan
        if plan is None or self._t_predicted_impact is None:
            self._enter(State.RECOVER)
            return

        # Refine the plan continuously while we move into position.
        if self._tracker.time_since_last_seen() <= self._cfg.fsm.ball_timeout_s:
            intercept = self._tracker.predict_intercept(self._cfg.court.strike_plane_x)
            if intercept is not None:
                refined = self._planner.plan(intercept)
                if refined is not None:
                    plan = refined
                    self._plan = refined
                    self._t_predicted_impact = time.perf_counter() + refined.time_to_impact

        # Command the wind-up pose + base pose.
        self._robot.write_base_goal(plan.base_pose)
        self._robot.write_racket_goal(plan.wind_up_position, plan.wind_up_orientation)

        # Commit to the swing once we're inside the commit window.
        time_to_impact = self._t_predicted_impact - time.perf_counter()
        if time_to_impact <= self._cfg.fsm.swing_commit_time_s:
            self._enter(State.SWING)
            return

        # If the ball drops out of tracking, abort.
        if self._tracker.time_since_last_seen() > self._cfg.fsm.ball_timeout_s:
            self._enter(State.RECOVER)

    def _step_swing(self) -> None:
        plan = self._plan
        if plan is None:
            self._enter(State.RECOVER)
            return

        if self._t_swing_started is None:
            self._t_swing_started = time.perf_counter()
            # Send the strike pose with the desired strike velocity.
            self._robot.write_racket_goal(
                plan.strike_position,
                plan.strike_orientation,
                plan.strike_velocity,
            )

        # After impact, start heading to the ready pose immediately so the arm
        # decelerates toward home rather than arcing through a follow-through point.
        if (
            self._t_predicted_impact is not None
            and time.perf_counter() >= self._t_predicted_impact
        ):
            ready = self._cfg.ready
            self._robot.write_racket_goal(
                ready.racket_position,
                ready.racket_orientation,
            )
            self._robot.write_base_goal(ready.base_pose)

        # Hold the follow-through briefly, then recover.
        if self._time_in_state() >= self._cfg.fsm.follow_through_hold_s + max(
            0.0, (self._t_predicted_impact or 0.0) - self._t_swing_started
        ):
            self._n_hits += 1
            if self._cfg.fsm.verbose:
                print(f"[FSM] swing complete (hit #{self._n_hits})")
            self._t_swing_started = None
            self._enter(State.RECOVER)

    def _step_recover(self, racket_pos: np.ndarray, racket_rot: np.ndarray) -> None:
        ready = self._cfg.ready
        self._robot.write_base_goal(ready.base_pose)
        self._robot.write_joint_goal(ready.joint_positions)

        q = self._robot.read_joint_positions()
        settled = q is not None and joints_close(q, ready.joint_positions, self._cfg.fsm.joint_tol)
        timed_out = self._time_in_state() >= self._cfg.fsm.recover_max_s

        if settled or timed_out:
            if timed_out and not settled and self._cfg.fsm.verbose:
                print(f"[FSM] RECOVER timed out after {self._cfg.fsm.recover_max_s:.1f}s "
                      f"(pos err {np.linalg.norm(racket_pos - ready.racket_position):.3f} m); "
                      f"forcing READY")
            self._plan = None
            self._t_predicted_impact = None
            self._tracker.reset()
            self._enter(State.READY)

    def _step_safe_stop(self) -> None:
        # Just hold the ready pose and stop driving the base.
        ready = self._cfg.ready
        self._robot.write_base_goal(ready.base_pose)
        self._robot.write_racket_goal(ready.racket_position, ready.racket_orientation)

    # -------- main loop

    def run(self) -> None:
        # Pace the loop the same way panda_left_right.py does.
        dt = self._cfg.fsm.control_dt
        loop_time = 0.0
        time.sleep(0.01)
        init_time = time.perf_counter_ns() * 1e-9
        self._t_state_entered = time.perf_counter()
        self._robot.publish_state(self._state)
        self._robot.switch_to_joint_controller()

        while self._running:
            loop_time += dt
            sleep_for = loop_time - (time.perf_counter_ns() * 1e-9 - init_time)
            if sleep_for > 0:
                time.sleep(sleep_for)

            # Sample the ball every tick.
            self._tracker.update()

            pose = self._robot.read_racket_pose()
            if pose is None:
                # Without robot feedback we can't safely transition anywhere.
                continue
            racket_pos, racket_rot = pose

            try:
                if self._state == State.INIT:
                    self._step_init(racket_pos, racket_rot)
                elif self._state == State.READY:
                    self._step_ready()
                elif self._state == State.TRACK:
                    self._step_track()
                elif self._state == State.APPROACH:
                    self._step_approach(racket_pos, racket_rot)
                elif self._state == State.SWING:
                    self._step_swing()
                elif self._state == State.RECOVER:
                    self._step_recover(racket_pos, racket_rot)
                elif self._state == State.SAFE_STOP:
                    self._step_safe_stop()
            except Exception as exc:  # noqa: BLE001 - we want to log and bail safely
                print(f"[FSM] unhandled error in state {self._state.name}: {exc}")
                self._enter(State.SAFE_STOP)


# ---------- Entry point --------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pickleball FSM for mmp_panda")
    parser.add_argument("--robot-backend", choices=["opensai", "cs225a"], default="opensai")
    parser.add_argument("--ball-source", choices=["opensai", "optitrack"], default="opensai")
    parser.add_argument(
        "--config-file",
        default=None,
        help="Expected OpenSai config file name (only used when --robot-backend=opensai)",
    )
    parser.add_argument(
        "--optitrack-rigid-body-id",
        type=int,
        default=1,
        help="OptiTrack rigid body ID for the pickleball",
    )
    parser.add_argument("--redis-host", default="localhost")
    parser.add_argument("--redis-port", type=int, default=6379)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    keys = RedisKeys(
        robot_backend=args.robot_backend,
        ball_source=args.ball_source,
        expected_config_file=args.config_file,
    )
    keys.ball = type(keys.ball)(
        opensai_object_pose=keys.ball.opensai_object_pose,
        optitrack_rigid_body_id=args.optitrack_rigid_body_id,
        optitrack_pos_prefix=keys.ball.optitrack_pos_prefix,
        optitrack_ori_prefix=keys.ball.optitrack_ori_prefix,
    )

    redis_client = redis.Redis(host=args.redis_host, port=args.redis_port)
    cfg = PickleballConfig()

    fsm = PickleballFSM(redis_client, keys, cfg)

    # Sanity-check the OpenSai backend before driving the robot.
    fsm._robot.check_opensai_guards(args.config_file)

    def _sigint(_signum, _frame):
        print("\n[FSM] SIGINT received, stopping...")
        fsm.request_stop()

    signal.signal(signal.SIGINT, _sigint)
    signal.signal(signal.SIGTERM, _sigint)

    try:
        fsm.run()
    except KeyboardInterrupt:
        print("[FSM] keyboard interrupt")
    except Exception as exc:  # noqa: BLE001
        print(f"[FSM] fatal error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
