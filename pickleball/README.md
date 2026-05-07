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

`sports_bot/pickleball` is a sub-CMakeLists that depends on `CS225A_COMMON_*`
variables defined by `cs225a/CMakeLists.txt`. The fastest way to wire it in is
to add one line at the bottom of `cs225a/CMakeLists.txt`:

```cmake
add_subdirectory(${PROJECT_SOURCE_DIR}/../sports_bot)
```

Targets are named `simviz_pickleball` and `controller_pickleball`
and land under `cs225a/bin/pickleball/` so they don't collide with
the `cs225a/project_starter/mmp_panda` versions.

```bash
cd cs225a/build && cmake .. && make -j simviz_pickleball controller_pickleball
```

## Quick launch (`run.sh`)

For interactive bring-up + ball-launching from a single terminal:

```bash
./sports_bot/pickleball/run.sh start          # brings up sim+controller+FSM, drops into REPL
```

REPL commands:
- `ball` — launch the default test ball (5 m out, -7 m/s, slight arc)
- `ball 5 0 2 -7 0 1.5` — custom pose+velocity
- `shoot` — `ball` + watch FSM transitions for 5 s
- `state` — process + redis snapshot
- `watch [seconds]` — log every FSM state change
- `tail simviz|controller|fsm` — last 60 lines of a log
- `restart` / `stop` / `start` / `quit`

CLI subcommands also available outside the REPL: `./run.sh ball`, `./run.sh status`,
`./run.sh tail fsm`, `./run.sh stop`, `./run.sh watch 5`. Logs land in
`/tmp/sports_bot/`. Override the python interpreter with `PICKLEBALL_PYTHON=...`.

## Launch sequence (verified end-to-end)

This is the exact sequence that produces a `INIT → READY → TRACK → APPROACH →
SWING → RECOVER → READY` cycle with the paddle making contact with the ball.
Run each command in its own terminal, in order. Wait a beat between processes
so each one finishes initializing.

```bash
# 1. Redis. Leave running. ("redis-cli ping" returns PONG when it's up.)
redis-server

# 2. Simulator + visualizer. Opens the GUI window, publishes joint state,
#    ball pose, and listens for ball-launch requests. Robot starts in a
#    forward-facing posture (joint1 = -90°, arm extended along world +X).
./cs225a/bin/pickleball/simviz_pickleball

# 3. Controller. Reads racket + base goals from redis, drives the hierarchical
#    task stack. Until the FSM starts, this just holds the startup pose.
./cs225a/bin/pickleball/controller_pickleball

# 4. Pickleball state machine (Python — make sure the opensai conda env
#    is active so redis + numpy are importable).
conda activate opensai
python -m sports_bot.state_machine.pickleball_fsm \
    --robot-backend cs225a \
    --ball-source optitrack
```

Within ~2 seconds the FSM should print `[FSM] INIT -> READY`. Now fire a ball:

```bash
# Defaults give a flat shot from 5 m out at -5.5 m/s. That trajectory bounces
# on the floor before reaching the strike plane and the simple ballistic
# predictor rejects it. For a clean swing test, pass a higher arc:
python -m sports_bot.pickleball.launch_ball --pos 5.0 0.0 2.0 --vel -7.0 0.0 1.5
```

Expected console output (from the FSM):
```
[FSM] INIT -> READY
[FSM] READY -> TRACK
[FSM] TRACK -> APPROACH
[FSM] APPROACH -> SWING
[FSM] swing complete (hit #1)
[FSM] SWING -> RECOVER
[FSM] RECOVER -> READY
```

You can fire more balls — the FSM stays running and cycles through the states
for each one. Edit `--pos`/`--vel` to test different shots.

### What to watch in the GUI

* The dark blue paddle on the end of the arm. At rest, the paddle face faces
  the +X (away-from-camera) direction.
* The yellow pickleball — initially parked above the opponent court at
  `(5.0, 0.0, 1.5)`. After `launch_ball`, it flies toward the robot.
* When the FSM enters `SWING`, the racket TCP drives toward the predicted
  intercept point and the ball gets deflected.

### Live state inspection (any terminal)

```bash
redis-cli get sports_bot::fsm::state                  # INIT/READY/TRACK/APPROACH/SWING/RECOVER
redis-cli get sai2::optitrack::rigid_body_pos::1      # ball world-frame position
redis-cli get sports_bot::state::racket::current_position
redis-cli get sports_bot::cmd::racket::goal_position  # what the FSM is asking for
redis-cli get sports_bot::sim::ball::linear_velocity
```

### (Real cart only) — step 5

Swap step 2 (`simviz_pickleball`) for whatever bridge publishes the
real robot's joint state to redis under `sai::sim::mmp_panda::sensors::*`, and
run the OptiTrack streamer in place of the launch-ball CLI:

```bash
python sports_bot/optitrack/StreamDataSkeleton.py
```

Edit `sports_bot/optitrack/world_calibration.json` first with your measured
`T_world_optitrack` (rotation + translation). Without it the streamer publishes
the OptiTrack room frame as the world frame.

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

* `state_machine/config.py` — `CourtConfig.strike_plane_x`,
  `RacketConfig.impact_speed`, `BallTrackerConfig.history_size`,
  `FsmConfig.swing_commit_time_s`. `ReadyPose.racket_position/orientation` is
  pinned to the controller's natural rest pose at the configured arm posture;
  if you change `q_posture` in `controller.cpp`, re-measure the rest pose
  (start sim+controller, read `sports_bot::state::racket::current_*`) and
  update `ReadyPose` to match.
* `controller.cpp` — racket / base / posture gains (currently 120/25 racket,
  60/18 base, 20/8 posture). The cs225a-starter defaults (400/40) saturate the
  joint torque limits given the heavy mobile base + paddle; SAI's internal
  OTG (0.3 m/s, 2 m/s²) ramps the goal so lower P gains still track cleanly.
  Bump racket gains carefully if you need crisper swings.
* `simviz.cpp` — ball restitution (0.6) and paddle restitution (0.85). Ball
  clipping the paddle face at high speed → bump sim frequency past 2000 Hz or
  stiffen contact.

## Debugging tips

* Uncomment the `graphics->showLinkFrame(...)` lines in `simviz.cpp` to render
  link7 and `paddle_tcp` axes — fastest way to verify the paddle is oriented
  the way the swing planner expects.
* `redis-cli get sports_bot::fsm::state` shows the current FSM state.
* `redis-cli get sai2::optitrack::rigid_body_pos::1` shows the ball's
  world-frame position whether sim or OptiTrack is feeding it.
