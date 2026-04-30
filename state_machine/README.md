# Pickleball state machine

A clean, modular FSM that drives the mobile-manipulator Panda (`mmp_panda`)
through one full pickleball rally cycle:

```
INIT ──► READY ──► TRACK ──► APPROACH ──► SWING ──► RECOVER ──► READY ...
                                            │
                                            ▼
                                       SAFE_STOP (any error)
```

The FSM follows the same conventions as the scripts in `python_examples/`:
fixed 100 Hz loop paced with `time.sleep(max(0, ...))`, `Enum` states, all
robot/ball I/O over Redis.

## State summary

| State       | What it does                                                                 | Exit condition                                            |
|-------------|------------------------------------------------------------------------------|-----------------------------------------------------------|
| `INIT`      | Drive base + arm to the home pose; racket up, base centred.                  | Racket within `pos_tol` / `ori_tol` of the ready pose.    |
| `READY`     | Hold ready pose; sample the ball.                                            | Ball is incoming and a feasible swing plan exists.        |
| `TRACK`     | Confirm the trajectory with more samples.                                    | Refined plan stays feasible (→ APPROACH) or aborts.       |
| `APPROACH`  | Drive base + racket to the wind-up pose; keep refining the plan.             | `time_to_impact ≤ swing_commit_time_s` (→ SWING).         |
| `SWING`     | Command the strike pose with `goal_linear_velocity`, then follow-through.    | After follow-through hold (→ RECOVER).                    |
| `RECOVER`   | Return to the ready pose.                                                    | Racket back at ready (→ READY).                           |
| `SAFE_STOP` | Hold ready pose and stop driving the base; entered on unhandled errors.     | Manual restart.                                           |

## File layout

```
sports_bot/state_machine/
    __init__.py
    redis_keys.py       # OpenSai vs cs225a robot keys; OpenSai vs OptiTrack ball keys
    config.py           # Tunable parameters (court, racket, ready pose, tracker, FSM)
    ball_tracker.py     # Sliding-window ballistic fit + intercept-plane solve
    swing_planner.py    # Wind-up / strike / follow-through poses and base placement
    pickleball_fsm.py   # Main 100 Hz state machine loop (entry point)
    README.md
```

## World frame convention

All world-frame coordinates use:

* **+X**: forward, toward the net / opponent.
* **+Y**: lateral (sideline).
* **+Z**: up.

The robot's home base position is the world origin. The strike plane is
`x = config.court.strike_plane_x` (default 0.60 m), and the ball intercept is
solved on this plane.

## Redis interface

Two robot backends are available; pick one with `--robot-backend`.

### `--robot-backend opensai` (default)

Uses the standard OpenSai `cartesian_controller` keys, identical to the ones
in `python_examples/panda_left_right.py`:

* `opensai::controllers::Panda::cartesian_controller::cartesian_task::goal_position`
* `... ::goal_orientation`
* `... ::goal_linear_velocity`
* `... ::current_position`
* `... ::current_orientation`

This is the path for OpenSai sim validation as soon as a config XML for the
mmp_panda is in place. To validate the swing motion alone you can already point
this at `single_panda.xml`:

```bash
# in OpenSai workspace
./bin/sai-interfaces-server config_folder/xml_config_files/single_panda.xml
# in another shell
python -m sports_bot.state_machine.pickleball_fsm \
    --robot-backend opensai \
    --ball-source opensai \
    --config-file single_panda.xml
```

### `--robot-backend cs225a`

Uses the joint-level keys exposed by `sports_bot/mmp_panda/controller.cpp`
(`sai::sim::mmp_panda::sensors::q`, etc.) plus a small set of high-level keys
the FSM expects the controller to read/write:

| Key                                              | Direction      | Type            |
|--------------------------------------------------|----------------|-----------------|
| `sports_bot::cmd::racket::goal_position`         | FSM → ctrl     | 3-vector        |
| `sports_bot::cmd::racket::goal_orientation`      | FSM → ctrl     | 3×3 rotation    |
| `sports_bot::cmd::racket::goal_linear_velocity`  | FSM → ctrl     | 3-vector (opt.) |
| `sports_bot::cmd::base::goal_pose`               | FSM → ctrl     | `[x, y, theta]` |
| `sports_bot::state::racket::current_position`    | ctrl → FSM     | 3-vector        |
| `sports_bot::state::racket::current_orientation` | ctrl → FSM     | 3×3 rotation    |
| `sports_bot::state::base::current_pose`          | ctrl → FSM     | `[x, y, theta]` |
| `sports_bot::fsm::state`                         | FSM → world    | string          |

Adding the racket and base goals to `sports_bot/mmp_panda/controller.cpp` is
the bridge between this FSM and the cs225a low-level controller, and the same
contract works for the real robot driver.

## Ball source

Two sources are supported, picked with `--ball-source`:

* `--ball-source opensai`: reads the 4×4 pose written by an OpenSai
  `dynamic_object` named `Ball` (see e.g. how `panda_gripper_pick_place.py`
  reads the box pose at `opensai::sensors::Box::object_pose`).
* `--ball-source optitrack`: reads the 3-vector pushed by
  `sports_bot/optitrack/StreamDataSkeleton.py` at
  `sai2::optitrack::rigid_body_pos::<id>`. Use `--optitrack-rigid-body-id`
  to match the rigid body ID assigned to the pickleball in Motive.

## Tuning

Everything mechanical / geometric lives in `config.py`. The most useful knobs:

* `CourtConfig.strike_plane_x` — how far in front of the robot we commit to hitting.
* `CourtConfig.return_target_xyz` — where we want returns to land (sets the racket face normal).
* `RacketConfig.sweet_spot_in_flange` — geometry of the 3D-printed racket mount; update once the CAD is final.
* `RacketConfig.impact_speed` — desired racket speed at impact.
* `BallTrackerConfig.history_size` / `history_max_age_s` — sliding-window length for the ballistic fit.
* `FsmConfig.swing_commit_time_s` — how long before predicted impact we commit to the swing.

## Bring-up plan

1. **Plan-only test.** Run with `--ball-source opensai` and a stationary ball
   placed in front of the robot to verify the FSM transitions and goals
   without launching a real swing.
2. **Single-panda swing.** Point at `single_panda.xml`, drop the mmp parts of
   the plan (the planner will still command racket-tip goals), and check
   that the racket follows the wind-up → strike → follow-through sequence.
3. **mmp_panda sim.** Add a `world_pickleball.urdf` and OpenSai config that
   includes the pickleball as a `dynamic_object`, and use `--robot-backend
   opensai` once the mobile manipulator is exposed through OpenSai.
4. **Real robot.** Switch to `--robot-backend cs225a --ball-source optitrack`,
   start `StreamDataSkeleton.py`, and run the FSM unchanged.
