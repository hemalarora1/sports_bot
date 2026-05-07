# sports_bot

CS225A final project — a mobile-manipulator panda with a 3D-printed pickleball
paddle, plus the python state machine that tracks an incoming ball and plans a
swing.

## Directory layout

```
sports_bot/
├── pickleball/      ← OUR pickleball implementation (C++ sim, controller,
│                      world URDF, run.sh launcher)
│   └── urdf/        ← in-repo robot URDF + visual meshes (no cs225a dep)
├── state_machine/   ← OUR python FSM, ball tracker, swing planner, config
├── optitrack/       ← third-party NatNet drivers + our world-frame
│                      calibration scaffolding (StreamDataSkeleton.py)
├── panda/           ← unmodified copy of cs225a/project_starter/panda,
│                      kept for reference; NOT built
├── my_project/      ← unmodified copy of cs225a/project_starter/my_project,
│                      kept for reference; NOT built
├── CMakeLists.txt   ← only builds pickleball/
└── README.md        ← this file
```

The `panda/` and `my_project/` subdirectories are starter-template copies kept
here as reference. They are **not** part of the pickleball build (their target
names would collide with the equivalents in `cs225a/project_starter/`). All
work for this project lives under `pickleball/`, `state_machine/`, and
`optitrack/`.

## Quick start

Pre-reqs: redis-server, the SAI core libraries, and a python with `redis +
numpy` (the `opensai` conda env in this repo).

```bash
# 1. one-time wiring into the cs225a build (already done in cs225a/CMakeLists.txt):
#       add_subdirectory(${PROJECT_SOURCE_DIR}/../sports_bot)

# 2. build the pickleball binaries:
cd cs225a/build && cmake .. && make -j simviz_pickleball controller_pickleball

# 3. one-shot launcher + interactive REPL:
./sports_bot/pickleball/run.sh start
```

Inside the REPL: `ball`, `state`, `watch`, `tail fsm`, `quit`, etc.
See [`pickleball/README.md`](pickleball/README.md) for full launch sequence,
tuning knobs, and debugging notes.

## What lives where

| File / dir | Role |
|---|---|
| [`pickleball/urdf/mmp_panda/mmp_panda_measured.urdf`](pickleball/urdf/mmp_panda/mmp_panda_measured.urdf) | Robot URDF: measured TidyBot base + Panda arm + MTEN paddle. Visual meshes live alongside in `pickleball/urdf/panda/meshes/visual/`. |
| [`pickleball/world_mmp_panda.urdf`](pickleball/world_mmp_panda.urdf) | Sim world: loads the robot URDF + pickleball + visual net + court floor + strike-plane marker. |
| [`pickleball/simviz.cpp`](pickleball/simviz.cpp) | SAI sim/render loop. Mirrors ball pose to the OptiTrack-shaped redis key. Listens for ball-launch requests. |
| [`pickleball/controller.cpp`](pickleball/controller.cpp) | Hierarchical task controller (base ▶ racket ▶ posture). Reads goals from the FSM, publishes current state. |
| [`pickleball/redis_keys.h`](pickleball/redis_keys.h) | The C-side redis contract; mirror of `state_machine/redis_keys.py`. |
| [`pickleball/launch_ball.py`](pickleball/launch_ball.py) | CLI: bumps the launch counter to teleport+kick the sim ball. |
| [`pickleball/run.sh`](pickleball/run.sh) | One-shot launcher + interactive REPL for sim+controller+FSM. |
| [`state_machine/pickleball_fsm.py`](state_machine/pickleball_fsm.py) | INIT → READY → TRACK → APPROACH → SWING → RECOVER state machine, 100 Hz. |
| [`state_machine/ball_tracker.py`](state_machine/ball_tracker.py) | Ballistic trajectory fit + intercept-on-strike-plane prediction. |
| [`state_machine/swing_planner.py`](state_machine/swing_planner.py) | Maps an intercept point + return target into wind-up / strike / follow-through poses. |
| [`state_machine/config.py`](state_machine/config.py) | All tunables (court geometry, ready pose, racket geometry, FSM timings). |
| [`state_machine/redis_keys.py`](state_machine/redis_keys.py) | Redis key contract for OpenSai vs cs225a backends and OpenSai vs OptiTrack ball sources. |
| [`optitrack/StreamDataSkeleton.py`](optitrack/StreamDataSkeleton.py) | NatNet → redis bridge with a hot-loadable world calibration. |
