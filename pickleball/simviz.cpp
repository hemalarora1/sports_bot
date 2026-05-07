/**
 * @file simviz.cpp
 * @brief Simulation + visualization for the mobile-manipulator panda with the
 *        MTEN pickleball paddle and a pickleball.
 *
 * Responsibilities (beyond the base SAI sim/render loop):
 *   - Load mmp_panda_measured.urdf and start in a non-singular posture.
 *   - Mirror the ball's pose to the OptiTrack-shaped redis key so the python
 *     FSM can run with --ball-source optitrack against either sim or hardware.
 *   - Listen on a launch counter and teleport+kick the ball when test scripts
 *     ask for a new shot.
 */

#include <math.h>
#include <signal.h>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <vector>
#include <typeinfo>
#include <random>

#include "SaiGraphics.h"
#include "SaiModel.h"
#include "SaiSimulation.h"
#include "SaiPrimitives.h"
#include "redis/RedisClient.h"
#include "timer/LoopTimer.h"
#include "logger/Logger.h"

bool fSimulationRunning = true;
void sighandler(int){fSimulationRunning = false;}

#include "redis_keys.h"

using namespace Eigen;
using namespace std;

// mutex and globals
VectorXd ui_torques;
mutex mutex_torques, mutex_update;

// specify urdf and robots
static const string robot_name = "mmp_panda";
static const string camera_name = "camera_fixed";
static const string ball_name = "ball";

// dynamic objects information
const vector<std::string> object_names = {ball_name};
vector<Affine3d> object_poses;
vector<VectorXd> object_velocities;
const int n_objects = object_names.size();

// initial joint posture (radians). joint 1 = -90° aligns the arm with world
// +X (toward the opponent). joint 7 stays at 0 — full ±π wrist roll is outside
// the Franka limits, so the FSM's home orientation is matched to whatever
// face_up direction this posture happens to produce (see config.py).
static VectorXd initial_q(int dof) {
	VectorXd q = VectorXd::Zero(dof);
	q.tail(7) << -90.0, -15.0, 0.0, -100.0, 0.0, 90.0, 0.0;
	q.tail(7) *= M_PI / 180.0;
	return q;
}

// simulation thread
void simulation(std::shared_ptr<SaiSimulation::SaiSimulation> sim);

int main() {
	// Make our in-repo URDF tree (sports_bot/pickleball/urdf/) discoverable to
	// SAI under the ${PICKLEBALL_URDF_FOLDER} substitution variable used by
	// world_mmp_panda.urdf. Keep the old CS225A_URDF_FOLDER too so any future
	// world references to project_starter assets still resolve.
	SaiModel::URDF_FOLDERS["CS225A_URDF_FOLDER"]      = string(CS225A_URDF_FOLDER);
	SaiModel::URDF_FOLDERS["PICKLEBALL_URDF_FOLDER"]  = string(PICKLEBALL_FOLDER) + "/urdf";

	static const string robot_file = string(PICKLEBALL_FOLDER) + "/urdf/mmp_panda/mmp_panda_measured.urdf";
	static const string world_file = string(PICKLEBALL_FOLDER) + "/world_mmp_panda.urdf";
	std::cout << "Loading URDF world model file: " << world_file << endl;

	// start redis client
	auto redis_client = SaiCommon::RedisClient();
	redis_client.connect();

	// set up signal handler
	signal(SIGABRT, &sighandler);
	signal(SIGTERM, &sighandler);
	signal(SIGINT, &sighandler);

	// load graphics scene
	auto graphics = std::make_shared<SaiGraphics::SaiGraphics>(world_file, camera_name, false);
	graphics->setBackgroundColor(0.04, 0.30, 0.55);  // pickleball-court blue
	// uncomment to debug paddle frame alignment:
	// graphics->showLinkFrame(true, robot_name, "link7", 0.15);
	// graphics->showLinkFrame(true, robot_name, "paddle_tcp", 0.15);
	graphics->addUIForceInteraction(robot_name);
	graphics->addUIForceInteraction(ball_name);

	// load robot model and put it in a non-singular start pose.
	auto robot = std::make_shared<SaiModel::SaiModel>(robot_file, false);
	const VectorXd q0 = initial_q(robot->dof());
	robot->setQ(q0);
	robot->setDq(VectorXd::Zero(robot->dof()));
	robot->updateModel();
	ui_torques = VectorXd::Zero(robot->dof());

	// load simulation world
	auto sim = std::make_shared<SaiSimulation::SaiSimulation>(world_file, false);
	sim->setJointPositions(robot_name, robot->q());
	sim->setJointVelocities(robot_name, robot->dq());

	// Capture initial dynamic-object state.
	for (int i = 0; i < n_objects; ++i) {
		object_poses.push_back(sim->getObjectPose(object_names[i]));
		object_velocities.push_back(sim->getObjectVelocity(object_names[i]));
	}

	// Contact tuning.
	//   - Robot links shouldn't bounce, but the paddle face should so the ball
	//     deflects realistically when struck.
	//   - Ball-floor restitution ~0.6 (pickleball loses ~30% energy per floor
	//     bounce in practice).
	sim->setCollisionRestitution(0.0);                           // global default
	sim->setCollisionRestitution(0.6, ball_name);                // ball ↔ anything
	sim->setCollisionRestitution(0.85, robot_name, "paddle_face"); // paddle striking surface

	// A small amount of friction so the ball doesn't slide forever on contact.
	sim->setCoeffFrictionStatic(0.4);
	sim->setCoeffFrictionDynamic(0.3);

	// Init redis for joint state and ball state so consumers always read valid keys.
	redis_client.setEigen(JOINT_ANGLES_KEY, robot->q());
	redis_client.setEigen(JOINT_VELOCITIES_KEY, robot->dq());
	redis_client.setEigen(JOINT_TORQUES_COMMANDED_KEY, VectorXd::Zero(robot->dof()));

	const Affine3d ball_pose0 = sim->getObjectPose(ball_name);
	redis_client.setEigen(BALL_OPTITRACK_POS_KEY, ball_pose0.translation());
	redis_client.setEigen(BALL_OPTITRACK_ORI_KEY, ball_pose0.linear());
	redis_client.setEigen(BALL_OPENSAI_POSE_KEY, ball_pose0.matrix());
	redis_client.setEigen(BALL_LIN_VEL_KEY, Vector3d::Zero());

	// Initialize the launch trio so the launcher script just has to bump the
	// counter; default pose/velocity match the URDF starting state.
	redis_client.setEigen(BALL_LAUNCH_COUNTER_KEY, VectorXd::Zero(1));
	redis_client.setEigen(BALL_LAUNCH_POSE_KEY, ball_pose0.translation());
	redis_client.setEigen(BALL_LAUNCH_VELOCITY_KEY, Vector3d::Zero());

	// start simulation thread
	thread sim_thread(simulation, sim);

	while (graphics->isWindowOpen() && fSimulationRunning) {
		graphics->updateRobotGraphics(robot_name, redis_client.getEigen(JOINT_ANGLES_KEY));
		{
			lock_guard<mutex> lock(mutex_update);
			for (int i = 0; i < n_objects; ++i) {
				graphics->updateObjectGraphics(object_names[i], object_poses[i]);
			}
		}
		graphics->renderGraphicsWorld();
		{
			lock_guard<mutex> lock(mutex_torques);
			ui_torques = graphics->getUITorques(robot_name);
		}
	}

	fSimulationRunning = false;
	sim_thread.join();

	return 0;
}

//------------------------------------------------------------------------------
void simulation(std::shared_ptr<SaiSimulation::SaiSimulation> sim) {
	auto redis_client = SaiCommon::RedisClient();
	redis_client.connect();

	double sim_freq = 2000;
	SaiCommon::LoopTimer timer(sim_freq);

	sim->setTimestep(1.0 / sim_freq);
	sim->enableGravityCompensation(true);
	sim->enableJointLimits(robot_name);

	int last_launch_counter = 0;

	while (fSimulationRunning) {
		timer.waitForNextLoop();

		// Pull commanded torques from the controller.
		VectorXd control_torques = redis_client.getEigen(JOINT_TORQUES_COMMANDED_KEY);
		{
			lock_guard<mutex> lock(mutex_torques);
			sim->setJointTorques(robot_name, control_torques + ui_torques);
		}

		// Honor any pending ball-launch request from a test driver.
		// We treat the counter as a single double; the launcher increments it.
		const VectorXd launch_counter_vec = redis_client.getEigen(BALL_LAUNCH_COUNTER_KEY);
		const int launch_counter = (launch_counter_vec.size() > 0)
									 ? static_cast<int>(launch_counter_vec(0))
									 : 0;
		if (launch_counter != last_launch_counter) {
			const Vector3d launch_pos = redis_client.getEigen(BALL_LAUNCH_POSE_KEY);
			const Vector3d launch_vel = redis_client.getEigen(BALL_LAUNCH_VELOCITY_KEY);
			Affine3d new_pose = Affine3d::Identity();
			new_pose.translation() = launch_pos;
			sim->setObjectPose(ball_name, new_pose);
			sim->setObjectVelocity(ball_name, launch_vel, Vector3d::Zero());
			last_launch_counter = launch_counter;
			std::cout << "[simviz] ball launched: pos=" << launch_pos.transpose()
					  << " vel=" << launch_vel.transpose() << std::endl;
		}

		sim->integrate();

		// Publish robot state.
		redis_client.setEigen(JOINT_ANGLES_KEY, sim->getJointPositions(robot_name));
		redis_client.setEigen(JOINT_VELOCITIES_KEY, sim->getJointVelocities(robot_name));

		// Sync objects to the graphics thread + publish to redis.
		{
			lock_guard<mutex> lock(mutex_update);
			for (int i = 0; i < n_objects; ++i) {
				object_poses[i] = sim->getObjectPose(object_names[i]);
				object_velocities[i] = sim->getObjectVelocity(object_names[i]);
			}
		}
		const Affine3d ball_pose = sim->getObjectPose(ball_name);
		const VectorXd ball_vel6 = sim->getObjectVelocity(ball_name);
		redis_client.setEigen(BALL_OPTITRACK_POS_KEY, ball_pose.translation());
		redis_client.setEigen(BALL_OPTITRACK_ORI_KEY, ball_pose.linear());
		redis_client.setEigen(BALL_OPENSAI_POSE_KEY, ball_pose.matrix());
		if (ball_vel6.size() >= 3) {
			redis_client.setEigen(BALL_LIN_VEL_KEY, ball_vel6.head<3>());
		}
	}
	timer.stop();
	cout << "\nSimulation loop timer stats:\n";
	timer.printInfoPostRun();
}
