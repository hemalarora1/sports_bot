/**
 * @file controller.cpp
 * @brief Cartesian + base controller for the mobile-manipulator panda with the
 *        MTEN pickleball paddle.
 *
 * Reads the desired racket sweet-spot pose / linear velocity and the desired
 * base [x, y, theta] from redis (see redis_keys.h), regulates them with a
 * hierarchical SAI task stack (base > racket pose > posture damping), and
 * publishes the current racket and base state back so the python FSM can close
 * its outer loop. Base goes first so lateral ball-chasing happens through the
 * wheels rather than through arm reach (which jiggles).
 *
 * Controlled frame:
 *   The MotionForceTask is parameterized on link7 with a compliant_frame that
 *   places the control point at the paddle TCP (link7 +z = 0.107 m flange,
 *   then +z = 0.261 m to face center) and rotates body axes so that
 *   body z = paddle face normal, body y = handle->tip, body x = face right.
 *   This matches the convention used by sports_bot/state_machine/swing_planner.py
 *   ("columns of R are [face_right, face_up, face_normal] in world").
 */

#include <SaiModel.h>
#include "SaiPrimitives.h"
#include "redis/RedisClient.h"
#include "timer/LoopTimer.h"

#include <iostream>
#include <string>

using namespace std;
using namespace Eigen;
using namespace SaiPrimitives;

#include <signal.h>
bool runloop = false;
void sighandler(int){runloop = false;}

#include "redis_keys.h"

// Paddle TCP in link7 frame (translation only).
//   flange offset: 0.107 m along link7 +z (matches panda_arm_hand.urdf)
//   paddle face center: another 0.261 m along link7 +z
static const Vector3d PADDLE_TCP_IN_LINK7(0.0, 0.0, 0.368);

// Rotation R_link7_ctrl: maps a vector expressed in the controlled frame to
// the same vector expressed in link7. Chosen so that the controlled-frame
// body z-axis is the paddle face normal (link7 +x), body y-axis is the handle
// -> tip direction (link7 +z), body x-axis is "across the face" (link7 +y).
static Matrix3d paddleControlFrameRotation() {
	Matrix3d R;
	R << 0, 0, 1,
		 1, 0, 0,
		 0, 1, 0;
	return R;
}

int main() {
	static const string robot_file = string(PICKLEBALL_FOLDER) + "/urdf/mmp_panda/mmp_panda_measured.urdf";

	auto redis_client = SaiCommon::RedisClient();
	redis_client.connect();

	signal(SIGABRT, &sighandler);
	signal(SIGTERM, &sighandler);
	signal(SIGINT, &sighandler);

	auto robot = std::make_shared<SaiModel::SaiModel>(robot_file, false);
	robot->setQ(redis_client.getEigen(JOINT_ANGLES_KEY));
	robot->setDq(redis_client.getEigen(JOINT_VELOCITIES_KEY));
	robot->updateModel();

	const int dof = robot->dof();
	VectorXd command_torques = VectorXd::Zero(dof);
	MatrixXd N_prec = MatrixXd::Identity(dof, dof);

	// Racket pose task on the controlled frame at paddle TCP.
	const string control_link = "link7";
	Affine3d compliant_frame = Affine3d::Identity();
	compliant_frame.translation() = PADDLE_TCP_IN_LINK7;
	compliant_frame.linear() = paddleControlFrameRotation();
	auto racket_task = std::make_shared<SaiPrimitives::MotionForceTask>(
		robot, control_link, compliant_frame, "racket_task");
	racket_task->setPosControlGains(300.0, 35.0, 0.0);
	racket_task->setOriControlGains(300.0, 35.0, 0.0);
	// SAI's internal OTG defaults (0.3 m/s linear, 2 m/s^2 accel) are far too
	// slow for pickleball — the racket lags behind the planned strike pose and
	// balls end up hitting the handle/wrist instead of the face. Bump to limits
	// matched to a real swing.
	racket_task->enableInternalOtgAccelerationLimited(
		/*max_lin_vel=*/   6.0,            // m/s
		/*max_lin_accel=*/ 30.0,           // m/s^2
		/*max_ang_vel=*/   4.0 * M_PI,     // rad/s
		/*max_ang_accel=*/ 20.0 * M_PI);   // rad/s^2

	// Base partial-joint task on (x, y, yaw) of the mobile base. Putting the
	// base task ABOVE the racket task means lateral chase motions go through
	// the wheels first (low arm jiggle), and the racket task only handles the
	// fine-grain pose error left in the base nullspace.
	MatrixXd base_selection = MatrixXd::Zero(3, dof);
	base_selection(0, 0) = 1.0;
	base_selection(1, 1) = 1.0;
	base_selection(2, 2) = 1.0;
	auto base_task = std::make_shared<SaiPrimitives::JointTask>(robot, base_selection);
	base_task->setGains(80.0, 20.0, 0.0);

	// Posture / nullspace joint task. Small P-gain keeps the redundant joint
	// (1 DOF after base+racket use 9 of the 10 robot DOFs) close to q_posture
	// so the arm doesn't drift into joint limits in the racket task's
	// nullspace. Too high here will fight the cartesian task when the arm
	// needs to lift / lower to ball height — 8 P / 6 D is the working range.
	auto joint_task = std::make_shared<SaiPrimitives::JointTask>(robot);
	joint_task->setGains(8.0, 6.0, 0.0);
	VectorXd q_posture(dof);
	q_posture.setZero();
	// joint 1 = -90° aligns the arm with world +X (forward to opponent).
	// Must match simviz's initial_q so the controller starts at zero error.
	q_posture.tail(7) << -90.0, -15.0, 0.0, -100.0, 0.0, 90.0, 0.0;
	q_posture.tail(7) *= M_PI / 180.0;
	joint_task->setGoalPosition(q_posture);

	// Seed redis goals from the current state so a controller running without
	// the FSM holds station, and the FSM has valid keys to read on first tick.
	const Vector3d racket_pos0 = racket_task->getCurrentPosition();
	const Matrix3d racket_ori0 = racket_task->getCurrentOrientation();
	const Vector3d base_pose0 = robot->q().head(3);
	redis_client.setEigen(RACKET_GOAL_POSITION_KEY, racket_pos0);
	redis_client.setEigen(RACKET_GOAL_ORIENTATION_KEY, racket_ori0);
	redis_client.setEigen(RACKET_GOAL_LINEAR_VELOCITY_KEY, Vector3d::Zero());
	redis_client.setEigen(BASE_GOAL_POSE_KEY, base_pose0);

	redis_client.setEigen(RACKET_CURRENT_POSITION_KEY, racket_pos0);
	redis_client.setEigen(RACKET_CURRENT_ORIENTATION_KEY, racket_ori0);
	redis_client.setEigen(RACKET_CURRENT_LINEAR_VELOCITY_KEY, Vector3d::Zero());
	redis_client.setEigen(BASE_CURRENT_POSE_KEY, base_pose0);

	runloop = true;
	const double control_freq = 1000.0;
	SaiCommon::LoopTimer timer(control_freq, 1e6);

	while (runloop) {
		timer.waitForNextLoop();

		// ---- Update robot model from sim/hardware -----------------------------
		robot->setQ(redis_client.getEigen(JOINT_ANGLES_KEY));
		robot->setDq(redis_client.getEigen(JOINT_VELOCITIES_KEY));
		robot->updateModel();

		// ---- Read FSM goals ---------------------------------------------------
		Vector3d racket_goal_pos = redis_client.getEigen(RACKET_GOAL_POSITION_KEY);
		Matrix3d racket_goal_ori = redis_client.getEigen(RACKET_GOAL_ORIENTATION_KEY);
		Vector3d racket_goal_vel = redis_client.getEigen(RACKET_GOAL_LINEAR_VELOCITY_KEY);
		Vector3d base_goal_pose  = redis_client.getEigen(BASE_GOAL_POSE_KEY);

		racket_task->setGoalPosition(racket_goal_pos);
		racket_task->setGoalOrientation(racket_goal_ori);
		racket_task->setGoalLinearVelocity(racket_goal_vel);

		// JointTask with a 3-row selection expects a 3-vector goal.
		base_task->setGoalPosition(base_goal_pose);

		// ---- Compute hierarchical torques ------------------------------------
		// Base has top priority for bulk lateral motion when chasing balls;
		// racket pose runs in the base nullspace for fine cartesian tracking;
		// arm posture damps the residual nullspace.
		N_prec.setIdentity();
		base_task->updateTaskModel(N_prec);
		racket_task->updateTaskModel(base_task->getTaskAndPreviousNullspace());
		joint_task->updateTaskModel(racket_task->getTaskAndPreviousNullspace());

		command_torques = base_task->computeTorques()
						+ racket_task->computeTorques()
						+ joint_task->computeTorques();

		// ---- Publish current state to the FSM --------------------------------
		redis_client.setEigen(RACKET_CURRENT_POSITION_KEY,        racket_task->getCurrentPosition());
		redis_client.setEigen(RACKET_CURRENT_ORIENTATION_KEY,     racket_task->getCurrentOrientation());
		redis_client.setEigen(RACKET_CURRENT_LINEAR_VELOCITY_KEY, racket_task->getCurrentLinearVelocity());
		redis_client.setEigen(BASE_CURRENT_POSE_KEY,              robot->q().head(3));

		// ---- Send torques -----------------------------------------------------
		redis_client.setEigen(JOINT_TORQUES_COMMANDED_KEY, command_torques);
	}

	timer.stop();
	cout << "\nController loop timer stats:\n";
	timer.printInfoPostRun();
	redis_client.setEigen(JOINT_TORQUES_COMMANDED_KEY, VectorXd::Zero(dof));

	return 0;
}
