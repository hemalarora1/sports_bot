# mmp_panda — pickleball sim & controller

Simulation, visualization, and a hierarchical cartesian + base controller for
the mobile-manipulator panda with the MTEN MT-01 paddle. Designed to run
end-to-end with the python pickleball state machine in
[`../state_machine/`](../state_machine/) and to mirror the OptiTrack data path
the real cart uses — same redis keys, same FSM, sim or hardware.

## Files

| File | Purpose |
|---|---|
| `world_mmp_panda.urdf` | Sim world: robot + pickleball + visual net + court floor + strike-plane marker. |
| `simviz.cpp` | SAI sim/render loop. Publishes ball pose to the OptiTrack-shaped redis key and listens on a launch counter. |
| `controller.cpp` | Reads racket cartesian goals + base goal from redis, regulates with a hierarchical task stack, publishes current state. |
| `redis_keys.h` | The redis contract. Keep in sync with `state_machine/redis_keys.py`. |
| `launch_ball.py` | CLI for kicking off a ball trajectory in sim. |

## Coordinate frames

* World frame: +X forward (toward the net), +Y lateral, +Z up. Robot base at
  the origin. Floor top surface at z=0.
* Net: visual only at x = 6.71 m (USA Pickleball half-court).
* Strike plane: x = 0.60 m (matches `CourtConfig.strike_plane_x`).
* Racket controlled frame (paddle TCP, body axes):
  * body x = "across the face"
  * body y = handle base → tip
  * body z = face normal (the strike direction)

  This means the orientation matrix the FSM commands has columns
  `[face_right, face_up, face_normal]` interpreted as world-frame unit vectors,
  matching `swing_planner._rotation_from_normal`.

## Building

`sports_bot/mmp_panda` is a sub-CMakeLists that depends on `CS225A_COMMON_*`
variables defined by `cs225a/CMakeLists.txt`. The fastest way to wire it in is
to add one line at the bottom of `cs225a/CMakeLists.txt`:

```cmake
add_subdirectory(${PROJECT_SOURCE_DIR}/../sports_bot)
```

Targets are named `simviz_sports_bot_mmp_panda` and `controller_sports_bot_mmp_panda`
and land under `cs225a/bin/sports_bot_mmp_panda/` so they don't collide with
the `cs225a/project_starter/mmp_panda` versions.

```bash
cd cs225a/build && cmake .. && make -j simviz_sports_bot_mmp_panda controller_sports_bot_mmp_panda
```

## Launch sequence

You need **four** things up at the same time (five if you're on the real cart):

```bash
# 1. Redis (in its own terminal — only needed once per machine).
redis-server

# 2. Simulator + visualizer.
./cs225a/bin/sports_bot_mmp_panda/simviz_sports_bot_mmp_panda

# 3. Controller. Holds the current pose until the FSM starts driving.
./cs225a/bin/sports_bot_mmp_panda/controller_sports_bot_mmp_panda

# 4. Pickleball state machine. Use the cs225a backend so it talks to the
#    controller above, and pull ball position from the optitrack-shaped key
#    that simviz mirrors.
python -m sports_bot.state_machine.pickleball_fsm \
    --robot-backend cs225a \
    --ball-source optitrack

# 5. (real cart only) OptiTrack streamer. Edit world_calibration.json to the
#    measured T_world_optitrack first.
python sports_bot/optitrack/StreamDataSkeleton.py
```

Once the four sim processes are running and the FSM has settled in `READY`,
fire a ball:

```bash
python -m sports_bot.mmp_panda.launch_ball
# or
python -m sports_bot.mmp_panda.launch_ball \
    --pos 5.0 0.3 1.7 --vel -6.0 -0.4 1.5
```

The FSM will watch the ball cross the strike plane, plan a swing, drive the
base + racket through wind-up → strike → follow-through, then return to
`READY` for the next shot.

## Running against the real cart

Swap step 2 (`simviz_mmp_panda`) for whatever bridge publishes the real
robot's joint state to redis under `sai::sim::mmp_panda::sensors::*`, and run
`StreamDataSkeleton.py` in place of the launch-ball CLI. Nothing in the
controller or FSM changes — the contract is the same redis keys.

If the ball position in the FSM's frame looks off, edit
`sports_bot/optitrack/world_calibration.json` (created by the streamer if
absent) with the measured `T_world_optitrack` (rotation + translation). The
streamer publishes a calibrated `sai2::optitrack::rigid_body_pos::<id>` plus
a raw `sai2::optitrack::raw::rigid_body_pos::<id>` so you can debug both.

## Tuning knobs that move first

* `state_machine/config.py` — `CourtConfig.strike_plane_x`, `RacketConfig.impact_speed`,
  `BallTrackerConfig.history_size`, `FsmConfig.swing_commit_time_s`.
* `controller.cpp` — racket / base / posture gains. The defaults (400/40 racket,
  200/30 base, 40/12 posture) are conservative; bump racket gains as needed for
  swing crispness.
* `simviz.cpp` — ball restitution (default 0.6) and paddle restitution
  (default 0.85). Ball clipping the paddle face at high speed → bump sim
  frequency from 2000 Hz or stiffen contact.

## Debugging tips

* Uncomment the `graphics->showLinkFrame(...)` lines in `simviz.cpp` to render
  link7 and `paddle_tcp` axes — fastest way to verify the paddle is oriented
  the way the swing planner expects.
* `redis-cli get sports_bot::fsm::state` shows the current FSM state.
* `redis-cli get sai2::optitrack::rigid_body_pos::1` shows the ball's
  world-frame position whether sim or OptiTrack is feeding it.
