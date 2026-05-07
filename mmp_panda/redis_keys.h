/**
 * @file redis_keys.h
 * @brief Redis key contract shared by simviz_mmp_panda, controller_mmp_panda,
 *        and the python pickleball FSM (state_machine/redis_keys.py).
 *
 * Conventions:
 *   - Positions are JSON 3-vectors in world frame (meters).
 *   - Orientations are JSON 3x3 row-major rotation matrices in world frame.
 *   - All "current_*" keys are written by the controller; "goal_*" keys are
 *     written by the FSM and read by the controller.
 *   - The ball is mirrored to the OptiTrack-shaped key so the FSM can run with
 *     --ball-source optitrack against either the simulator or the real cart.
 */

#pragma once

#include <string>

// ---- Joint-level state (sim <-> controller) ---------------------------------
const std::string JOINT_ANGLES_KEY              = "sai::sim::mmp_panda::sensors::q";
const std::string JOINT_VELOCITIES_KEY          = "sai::sim::mmp_panda::sensors::dq";
const std::string JOINT_TORQUES_COMMANDED_KEY   = "sai::sim::mmp_panda::actuators::fgc";
const std::string CONTROLLER_RUNNING_KEY        = "sai::sim::mmp_panda::controller";

// ---- Racket task (FSM <-> controller) ---------------------------------------
// Goal keys: written by the python FSM, read by controller.cpp every cycle.
const std::string RACKET_GOAL_POSITION_KEY        = "sports_bot::cmd::racket::goal_position";
const std::string RACKET_GOAL_ORIENTATION_KEY     = "sports_bot::cmd::racket::goal_orientation";
const std::string RACKET_GOAL_LINEAR_VELOCITY_KEY = "sports_bot::cmd::racket::goal_linear_velocity";

// State keys: written by controller.cpp, read by the FSM.
const std::string RACKET_CURRENT_POSITION_KEY        = "sports_bot::state::racket::current_position";
const std::string RACKET_CURRENT_ORIENTATION_KEY     = "sports_bot::state::racket::current_orientation";
const std::string RACKET_CURRENT_LINEAR_VELOCITY_KEY = "sports_bot::state::racket::current_linear_velocity";

// ---- Mobile base task (FSM <-> controller) ----------------------------------
// 3-vector [x, y, theta] in world frame.
const std::string BASE_GOAL_POSE_KEY     = "sports_bot::cmd::base::goal_pose";
const std::string BASE_CURRENT_POSE_KEY  = "sports_bot::state::base::current_pose";

// ---- FSM <-> controller handshake -------------------------------------------
const std::string FSM_STATE_KEY    = "sports_bot::fsm::state";    // string, e.g. "READY"
const std::string FSM_REQUEST_KEY  = "sports_bot::fsm::request";  // e.g. "pause", "estop"

// ---- Ball state (sim -> FSM, mirrors what OptiTrack would publish) ----------
const std::string BALL_OPTITRACK_POS_KEY   = "sai2::optitrack::rigid_body_pos::1";
const std::string BALL_OPTITRACK_ORI_KEY   = "sai2::optitrack::rigid_body_ori::1";
const std::string BALL_OPENSAI_POSE_KEY    = "opensai::sensors::Ball::object_pose";
const std::string BALL_LIN_VEL_KEY         = "sports_bot::sim::ball::linear_velocity";

// ---- Ball launcher (test scripts -> simviz) ---------------------------------
// Counter is incremented by launch_ball.py (or any test driver). simviz polls
// the counter; on change it teleports the ball to BALL_LAUNCH_POSE_KEY and
// applies BALL_LAUNCH_VELOCITY_KEY as the new linear velocity.
const std::string BALL_LAUNCH_COUNTER_KEY  = "sports_bot::sim::ball::launch_counter";
const std::string BALL_LAUNCH_POSE_KEY     = "sports_bot::sim::ball::launch_pose";      // 3-vec
const std::string BALL_LAUNCH_VELOCITY_KEY = "sports_bot::sim::ball::launch_velocity";  // 3-vec
