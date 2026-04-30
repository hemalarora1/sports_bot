"""Redis key profiles for the pickleball FSM.

Two backends are supported:

  ROBOT_BACKEND = "opensai"  -> use OpenSai's cartesian_controller keys (works
                                 out of the box with the python_examples config
                                 files like single_panda.xml).
  ROBOT_BACKEND = "cs225a"   -> use the custom cs225a-style controller in
                                 sports_bot/mmp_panda/controller.cpp, which
                                 exposes joint-level keys plus high-level
                                 racket goal keys we add on top.

Two ball sources are supported:

  BALL_SOURCE = "opensai"    -> ball pose comes from an OpenSai dynamic_object
                                 (4x4 homogeneous transform).
  BALL_SOURCE = "optitrack"  -> ball position comes from the OptiTrack streamer
                                 (StreamDataSkeleton.py) under the
                                 sai2::optitrack::rigid_body_pos::<id> key.
"""

from dataclasses import dataclass, field
from typing import Optional


# ---------- OpenSai cartesian_controller keys (sim path) -----------------------

@dataclass(frozen=True)
class OpenSaiCartesianKeys:
    """Matches python_examples/panda_*.py.

    Robot name and controller name are templated so the same dataclass can be
    reused for the panda or the mobile manipulator once an OpenSai config is
    written for it.
    """

    robot_name: str = "Panda"
    controller_name: str = "cartesian_controller"
    task_name: str = "cartesian_task"

    @property
    def goal_position(self) -> str:
        return f"opensai::controllers::{self.robot_name}::{self.controller_name}::{self.task_name}::goal_position"

    @property
    def goal_orientation(self) -> str:
        return f"opensai::controllers::{self.robot_name}::{self.controller_name}::{self.task_name}::goal_orientation"

    @property
    def goal_linear_velocity(self) -> str:
        return f"opensai::controllers::{self.robot_name}::{self.controller_name}::{self.task_name}::goal_linear_velocity"

    @property
    def current_position(self) -> str:
        return f"opensai::controllers::{self.robot_name}::{self.controller_name}::{self.task_name}::current_position"

    @property
    def current_orientation(self) -> str:
        return f"opensai::controllers::{self.robot_name}::{self.controller_name}::{self.task_name}::current_orientation"

    @property
    def current_linear_velocity(self) -> str:
        return f"opensai::controllers::{self.robot_name}::{self.controller_name}::{self.task_name}::current_linear_velocity"

    @property
    def active_controller(self) -> str:
        return f"opensai::controllers::{self.robot_name}::active_controller_name"

    @property
    def config_file_name(self) -> str:
        return "::sai-interfaces-webui::config_file_name"


# ---------- cs225a-style custom controller keys (mmp_panda) -------------------

@dataclass(frozen=True)
class Cs225aMmpPandaKeys:
    """Joint-level interface to sports_bot/mmp_panda/controller.cpp plus the
    high-level racket / base goal keys the FSM expects the controller to read.

    These extra keys are NOT yet in controller.cpp; you'll add them in the C++
    controller as part of bringing it up. They are the contract between the FSM
    and whatever low-level controller is running.
    """

    joint_angles: str = "sai::sim::mmp_panda::sensors::q"
    joint_velocities: str = "sai::sim::mmp_panda::sensors::dq"
    joint_torques_commanded: str = "sai::sim::mmp_panda::actuators::fgc"
    controller_running: str = "sai::sim::mmp_panda::controller"

    # Racket cartesian task (controller.cpp should consume these).
    racket_goal_position: str = "sports_bot::cmd::racket::goal_position"
    racket_goal_orientation: str = "sports_bot::cmd::racket::goal_orientation"
    racket_goal_linear_velocity: str = "sports_bot::cmd::racket::goal_linear_velocity"
    racket_current_position: str = "sports_bot::state::racket::current_position"
    racket_current_orientation: str = "sports_bot::state::racket::current_orientation"
    racket_current_linear_velocity: str = "sports_bot::state::racket::current_linear_velocity"

    # Mobile base task ([x, y, theta] in world frame).
    base_goal_pose: str = "sports_bot::cmd::base::goal_pose"
    base_current_pose: str = "sports_bot::state::base::current_pose"

    # FSM <-> controller handshake.
    fsm_state: str = "sports_bot::fsm::state"
    fsm_request: str = "sports_bot::fsm::request"  # e.g. "pause", "estop"


# ---------- Ball source keys ---------------------------------------------------

@dataclass(frozen=True)
class BallKeys:
    """Where to read the pickleball position from."""

    # OpenSai dynamic_object pose (4x4 homogeneous matrix as JSON list-of-lists).
    opensai_object_pose: str = "opensai::sensors::Ball::object_pose"

    # OptiTrack rigid body position written by StreamDataSkeleton.py (3-vector).
    # The rigid body id has to match what's set in Motive for the pickleball.
    optitrack_rigid_body_id: int = 1
    optitrack_pos_prefix: str = "sai2::optitrack::rigid_body_pos::"
    optitrack_ori_prefix: str = "sai2::optitrack::rigid_body_ori::"

    @property
    def optitrack_position(self) -> str:
        return f"{self.optitrack_pos_prefix}{self.optitrack_rigid_body_id}"

    @property
    def optitrack_orientation(self) -> str:
        return f"{self.optitrack_ori_prefix}{self.optitrack_rigid_body_id}"


# ---------- Bundle -------------------------------------------------------------

@dataclass
class RedisKeys:
    """The set of keys the FSM uses, selected by backend strings."""

    robot_backend: str = "opensai"        # "opensai" | "cs225a"
    ball_source: str = "opensai"          # "opensai" | "optitrack"

    opensai: OpenSaiCartesianKeys = field(default_factory=OpenSaiCartesianKeys)
    cs225a: Cs225aMmpPandaKeys = field(default_factory=Cs225aMmpPandaKeys)
    ball: BallKeys = field(default_factory=BallKeys)

    # Optional override of the OpenSai config file name guard.
    expected_config_file: Optional[str] = None
