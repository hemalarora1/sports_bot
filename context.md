# sports_bot integration context

Living reference for the pickleball robot bringup. Append as we learn — don't rewrite
unless something is wrong. Each section has a date so we can see what's fresh.

---

## The system at a glance

```
Motive PC (bay)  ──NatNet UDP──▶  StreamDataSkeleton.py  ──redis.set──▶  Redis
                                  (NatNet client,                         sai2::optitrack::
                                   laptop or mini-PC)                     rigid_body_pos::<id>
                                                                               │
                                                                               ▼
Redis  ◀──get──  pickleball_fsm.py  (BallTracker reads ball pose;
                                     SwingPlanner plans a hit;
                                     writes racket+base goals)
                                                                               │
                                                                               ▼
Redis  ──get──▶  OpenSai (mini-PC)            ──torques──▶  Franka arm
       ──get──▶  TidyBot redis_driver.py      ──vel──────▶  mobile base
```

Two parties only talk through Redis.

---

## Quick start: see the ball position on your laptop

Setup as of 2026-05-19: laptop on **SRC wifi**, static IP `172.24.68.204`, Motive Kitchen PC in **multicast** mode.

```bash
cd "$(git rev-parse --show-toplevel)/sports_bot/optitrack"
conda activate opensai
PYTHONPATH=drivers/PythonClient python -u StreamDataSkeleton.py \
    172.24.69.102 172.24.68.204 m    # 'm'=multicast, 'u'=unicast
```

Preferred — recorder wrapper (starts streamer, tears it down on exit):

```bash
./sports_bot/scripts/record_throws.sh                   # defaults: multicast, ID 8
STREAMER_MODE=u ./sports_bot/scripts/record_throws.sh   # only if Motive is in Unicast
```

Live view of all rigid bodies in Redis:

```bash
./sports_bot/scripts/watch_ball.sh          # all rigid bodies
./sports_bot/scripts/watch_ball.sh 8        # just the pickleball
```

To get your laptop's IP: `ifconfig en0 | grep "inet "`.

---

## Network & ports

| Item                   | Value           | Notes                                                       |
| ---------------------- | --------------- | ----------------------------------------------------------- |
| Kitchen Motive server  | `172.24.69.102` | Other bays have different IPs.                              |
| NatNet command port    | `1510`          | UDP, bidirectional. Survives cross-subnet routing.          |
| NatNet data port       | `1511`          | UDP. Multicast group `239.255.42.99:1511`.                  |
| VRPN port              | `3883`          | Different protocol, disabled. Often confused with NatNet.   |

**Multicast does NOT route across subnets** — requires laptop MAC registered with Zen (`zyaskawa@stanford.edu`) for a static SRC IP and association with `SRC` wifi SSID. Stanford wifi → unicast only.

---

## NatNet protocol cheatsheet

Two independent UDP channels:
- **Command (`:1510`)** — bidirectional unicast. Handshakes, DataDescriptions (names+IDs). Always works cross-subnet.
- **Data (`:1511`)** — server→client per-frame positions. Multicast (same subnet only) or unicast (routes anywhere), set in Motive's Streaming → Transmission Type.

If command channel works but no positions arrive: data channel mismatch (server multicasting, client can't hear). Per-frame stream carries **numeric IDs** only; names come from DataDescriptions fetched separately.

---

## Motive configuration (Kitchen, as of 2026-05-19)

- Streaming → NatNet — **enabled**, Transmission Type **Multicast** (group `239.255.42.99`, port `1511`).
- Streaming → VRPN — disabled.
- Streaming → **Labeled Markers — enabled** (required for `marker_pos::*` keys).
- Assets: `PickleBall` rigid body, **Streaming ID = 8**; TidyBot cart, **Streaming ID = 11**.
- KVM access: `SRC-KVM-Kitchen.stanford.edu`. Credentials from Zen.

**Critical gotcha — display vs streaming Up Axis are independent.** `View → Up Axis` is cosmetic; `Edit → Application Settings → Streaming → Up Axis` controls what NatNet publishes. These can silently disagree. Always confirm *streaming* is Z-up at session start. Sanity check: place ball on origin floor marker, read `sai2::optitrack::raw::rigid_body_pos::8` — the Z component should be small (~ball radius).

---

## Redis key schema

Published by `StreamDataSkeleton.py` (~120 Hz):

| Key                                                         | Format                  | Frame              |
| ----------------------------------------------------------- | ----------------------- | ------------------ |
| `sai2::optitrack::rigid_body_pos::<id>`                     | JSON `[x, y, z]`        | World (calibrated) |
| `sai2::optitrack::rigid_body_ori::<id>`                     | JSON `[qx, qy, qz, qw]` | World quat         |
| `sai2::optitrack::raw::rigid_body_pos::<id>`                | JSON `[x, y, z]`        | Motive room frame  |
| `sai2::optitrack::marker_pos::<model_id>::<marker_id>`      | JSON `[x, y, z]`        | World (calibrated) |
| `sai2::optitrack::raw::marker_pos::<model_id>::<marker_id>` | JSON `[x, y, z]`        | Motive room frame  |

`marker_pos::*` keys require **Labeled Markers enabled** in Motive. `model_id` = rigid-body Streaming ID the marker belongs to (0 = standalone/unaffiliated).

World transform: `R_WORLD_OPTI · p_opti + T_WORLD_OPTI` from `sports_bot/optitrack/world_calibration.json`.

**Frame conventions:**
- Motive streaming frame: **Z-up**, right-handed.
- World frame: **Z-up**, right-handed, origin at robot home. +X toward opponent, +Y left, +Z up.
- `world_calibration.json` is yaw + translation only (no axis swap needed).

**Current calibration (2026-05-17, SRC Kitchen):** 2D Procrustes from 3 floor markers. Max residual 0.34 mm horizontal, 1.87 mm vertical.

Other relevant Redis keys:

| Key                                                                                   | Owner               | Purpose                            |
| ------------------------------------------------------------------------------------- | ------------------- | ---------------------------------- |
| `opensai::controllers::FrankaRobot::cartesian_controller::cartesian_task::goal_position`    | us → OpenSai  | EE goal pos (arm base frame)       |
| `opensai::controllers::FrankaRobot::cartesian_controller::cartesian_task::goal_orientation` | us → OpenSai  | EE goal rot (3×3 matrix)           |
| `opensai::controllers::FrankaRobot::cartesian_controller::cartesian_task::current_position` | OpenSai → us  | Current EE pos (FK from encoders)  |
| `opensai::controllers::FrankaRobot::cartesian_controller::cartesian_task::current_orientation` | OpenSai → us | Current EE rot                    |
| `sports_bot::cmd::base::goal_pose`                                                    | FSM → base_bridge   | `[x, y, theta]` world-frame goal   |
| `hb1::desired_pose`                                                                   | base_bridge → TidyBot | `[x, y, theta]` odom-frame goal  |
| `hb1::current_pose` / `hb1::current_vel`                                              | TidyBot → us        | Base feedback                      |
| `hb1::kill` / `hb1::stop`                                                             | us → TidyBot        | "kill" terminates; "stop" decels   |

---

## Gotchas hit so far

- **Two copies of `StreamDataSkeleton.py`.** Use `sports_bot/optitrack/StreamDataSkeleton.py` (has world calibration + rigid_body_listener). The copy in `drivers/PythonClient/` is the vanilla SDK sample. Needs `PYTHONPATH=drivers/PythonClient`.
- **`python -u` matters** for stdout in tail logs / background runs.
- **`request_data_descriptions` crashes** on UTF-8 decode of marker names (NatNetClient.py:965). Doesn't affect streaming. Live with it.
- **`set_use_multicast(False)` is not enough** if Motive is in Multicast — both sides must agree.
- **macOS multicast bind silently drops frames** (patched 2026-05-19). Data socket must bind to `('', 1511)` (INADDR_ANY), not `(local_ip, 1511)`. Symptom: command handshake works (`resetting requested version to 4 2 0 0`), no Redis keys appear.
- **`sports_bot/` is its own git repo** nested inside OpenSai. `git rev-parse --show-toplevel` from within it returns the `sports_bot/` dir, not the OpenSai root. Scripts resolve the OpenSai root from `$0`. Run modules as `python -m sports_bot.<module>` from the OpenSai root.

---

## Ball tracker / intercept prediction

`state_machine/ball_tracker_test.py` — `record` + `analyze` subcommands.  
`scripts/record_throws.sh` — starts streamer, runs recorder, cleans up.  
`scripts/watch_ball.sh` — macOS-safe live viewer.  
`recordings/` — `throws_<YYYYMMDD_HHMMSS>.npz`.

```bash
conda activate opensai
./sports_bot/scripts/record_throws.sh                      # ID 8, multicast
python -m sports_bot.state_machine.ball_tracker_test analyze \
    "$(ls -t sports_bot/recordings/throws_*.npz | head -1)"
# A/B tracker params on same recording:
python -m sports_bot.state_machine.ball_tracker_test analyze <file>.npz \
    --tracker {leastsq,ekf} --history-size 8 --bounce-restitution 0.70
```

**Current production defaults (volley-only mode for SRC Kitchen bring-up):**
`max_bounces=0`, `online_bounce_pruning=False`, `history_size=8`, `history_max_age_s=0.20`. With `max_bounces=0` the predictor rejects any trajectory that would cross z=0 before the strike plane — volleys only, no FSM changes needed.

**Measured bounce parameters (SRC Kitchen, 11 bounces / 19 throws):**

| metric | restitution `e` | tangential `μ_t` |
|---|---|---|
| median | 0.690 | 0.606 |
| mean | 0.712 | 0.640 |
| std | 0.058 | 0.127 |

Config defaults: `bounce_restitution=0.70`, `bounce_tangential_damping=0.62`. Re-enable bounce mode by setting `max_bounces=1`, `online_bounce_pruning=True`.

**EKF tracker** (`state_machine/ekf_ball_tracker.py`) — same public interface as `BallTracker`, A/B-able via `--tracker ekf`. At swing-commit essentially tied with LS+pruning (5.0 cm LS vs 5.8 cm EKF mean). The payoff is uncertainty-aware commit via `Intercept.position_cov` — FSM not yet wired for this.

**Commit-window accuracy (tti ∈ [0.00, 0.15) s):** 5.0 cm mean / 31 cm max (LS + bounce pruning, SRC Kitchen recordings).

---

## Base frame calibration (FSM goals → TidyBot driver)

Four frames:

| frame | what |
|---|---|
| W | world; floor tape origin |
| B | OptiTrack rigid-body frame, glued to cart markers |
| C | TidyBot odometry control-point (what `hb1::current_pose` tracks) |
| R | robot odometry origin; anchors at driver start |

Two unknowns: `T_B_C` (geometric, persists across sessions) and `T_W_R` (per-session, derived at bridge startup). Constraint: `T_W_B ⊕ T_B_C = T_W_R ⊕ T_R_C`.

**Files:**

| Path | Purpose |
|---|---|
| `sports_bot/utils/frames.py` | SE(2)+(3) algebra, hand-eye solver, calibration I/O |
| `scripts/calibrate_robot_marker.py` | Interactive N-waypoint capture, solves `(T_B_C, T_W_R)` via LM |
| `optitrack/robot_marker_calibration.json` | Persisted `T_B_C`. Re-solve only when markers re-stuck. |
| `sports_bot/base_bridge.py` | Loads `T_B_C`, derives `T_W_R` from 1-s snapshot, forwards goals, periodic OT-vs-odom sanity print. |

**Current calibration (SRC Kitchen, 2026-05-19):** `T_B_C = (-0.0351 m, +0.0086 m, -7.16°)`, RMS 3.71 mm / 0.37°.

`goal_W` is the desired pose of **C** (odometry control point), not marker centroid B. Use `T_W_C = T_W_B ⊕ T_B_C` when querying current world pose for control purposes.

**Odom drift:** `T_W_R` is snapshot-once at bridge startup. After ~5 m of travel, drift can reach a few cm. Restart the bridge to re-anchor; the sanity Δ print is the watchdog. `--periodic-refresh-s 0.5` (default) re-derives `T_W_R` every 0.5 s and re-emits the last goal — eliminates held-goal drift.

---

## Arm world-frame tracking (marker-based, 2026-05-22)

OpenSai's cartesian controller speaks arm base frame **A** (all `current_*` / `goal_*` keys are in A). We need `T_W_A(t)` live to command world-frame intercept targets.

**Approach:** two standalone spherical OptiTrack markers on the cart, left (−Y) and right (+Y) of the Franka base, equidistant from center.

```
t_W_A   = 0.5 * (p_left + p_right)               # midpoint = arm base origin
y_hat_W = horiz_project(p_right − p_left) / ||…|| # Franka +Y
z_hat_W = [0, 0, 1]                               # Franka +Z (upright cart)
x_hat_W = y_hat × z_hat                           # Franka +X (right-handed)
```

No hand-eye calibration needed.

**Frames:**

| frame | what |
|---|---|
| A | Franka arm base. +X forward, +Y left, +Z up |
| E | Franka EE flange. OpenSai FK publishes `current_position/orientation` in A |
| P | Racket sweet-spot. +Z_P = face normal (SwingPlanner convention) |

**Runtime composition:**
```
T_W_P(t)      = T_W_A(t) ⊕ T_A_E(t) ⊕ T_E_P          # racket pose in world
T_A_E_desired = T_W_A(t)⁻¹ ⊕ T_W_P_goal ⊕ T_E_P⁻¹    # world goal → EE goal
```

`world_racket_to_arm_ee` in `frames.py` does the inverse-compose.

**Paddle geometry (MTEN MT-01, as mounted 2026-05-23):**
- Handle mounted straight along EE +Z (= link8 +Z = panda_hand +Z) — no rotation adapter.
- OpenSai EE frame = Franka **link8** frame. `panda_hand_joint` has `rpy="0 0 -0.7854"` (-45° Z), so panda_hand = Rz(-45°) @ link8.
- Paddle face = **panda_hand +Y** = `[1/√2, 1/√2, 0]` in link8 frame (45° between link8 +X and +Y).
- Natural operating pose: face toward +X opponent, handle pointing down (-Z world).
- `R_W_E_REF = [[1/√2, 1/√2, 0], [1/√2, -1/√2, 0], [0, 0, -1]]` — verified 2026-05-26.
- `picklebot.xml` `compliantFrame xyz="0 0 0.35"` — OpenSai's `current_position`/`goal_position`
  track the **sweet spot** (35 cm along EE +Z from the flange) directly. No code offset needed.
- `T_E_P` in `arm_marker_calibration.json`: translation `[0, 0, 0]` (zeroed after compliantFrame fix);
  rotation unchanged (maps P's +Z to EE +X — kept for legacy `world_racket_to_arm_ee` callers).
- Paddle tip (far end of paddle) is ~**45 cm** along EE +Z from the flange;
  **10 cm** past the sweet spot (`PADDLE_TIP_OFFSET_M = 0.10`).

`verify_arm_calibration.py --init` writes these as the starter defaults.
After the 2026-05-26 fix: `T_W_E ≈ T_W_P` when verifying (sweet spot = compliant frame).

**Floor collision avoidance:** `enforce_paddle_floor(t_A_E, R_A_E, T_W_A)` in `frames.py` computes the world-frame Z of the commanded paddle tip exactly:
```
z_tip_world = T_W_A.t[2] + t_A_E[2] + R_A_E[2,2] * PADDLE_TIP_OFFSET_M
```
If below floor + 5 cm clearance, raises the EE goal in arm-frame Z. Called before `clip_to_arm_workspace` in `test_arm_world_track.py` and `base_intercept.py`. Uses actual EE orientation (R_A_E[2,2]) — not worst-case — and gets arm base height from live T_W_A, no config constant needed.

**Files:**

| Path | Purpose |
|---|---|
| `optitrack/drivers/PythonClient/NatNetClient.py` | Patched: `labeled_marker_listener` callback |
| `optitrack/StreamDataSkeleton.py` | Publishes `marker_pos::<model>::<id>` per frame |
| `utils/frames.py` | SE(3) algebra; `compute_T_W_A_from_markers`, `world_racket_to_arm_ee`, `enforce_paddle_floor`, `clip_to_arm_workspace` |
| `optitrack/arm_marker_calibration.json` | 2 marker IDs + `T_E_P`. Re-do only when spheres move or paddle mount changes. |
| `scripts/list_markers.py` | Live table of all labeled markers; tap one to see its row blink |
| `scripts/verify_arm_calibration.py` | `--init` writes starter JSON; bare run prints T_W_A / T_W_E / T_W_P live |
| `scripts/test_arm_world_track.py` | Hold a fixed world racket pose as cart moves |
| `scripts/base_intercept.py` | Ball intercept loop; `--arm-track` adds arm tracking to base-only mode |
| `scripts/cmd_arm_world_clean.py` | Clean interactive world-frame arm commander; no filtering/locking |

**Motive setup (one-time):** Mount two spherical markers (not flat stickers — visible from any angle) standalone — do NOT add to cart rigid body 11. Enable Labeled Markers in Motive. Manually label both in the Markers pane to lock IDs.

**Per-session arm bringup:**
```bash
# Confirm markers streaming
python sports_bot/scripts/list_markers.py --model-id 0
# Suppose IDs are (0,5) and (0,6).

# First time only:
python sports_bot/scripts/verify_arm_calibration.py --init
# Edit arm_marker_calibration.json: set marker_specs [[0,5],[0,6]]
# spec[0]=left/-Y, spec[1]=right/+Y

# Sanity check (needs OpenSai running):
python sports_bot/scripts/verify_arm_calibration.py
# T_W_P[:,2] (face normal in world) should point toward opponent.
# Use --no-ee to check T_W_A only.

# World-frame arm hold test:
python sports_bot/scripts/test_arm_world_track.py --from-current
# Push cart by hand — arm should keep paddle at same world point.
# If it moves opposite, swap marker_specs[0] and [1] in JSON.
```

**Arm gotchas:**
- **Labeled markers must be enabled in Motive** — symptom: `list_markers.py` shows 0 markers but rigid-body keys still flow.
- **Symmetric sphere placement** — Motive may swap IDs frame-to-frame if markers are too close. Manual label locks them.
- **Floor must be flat** — `z_hat_W = [0,0,1]` is hardcoded (upright cart). Significant tilt breaks it.
- `marker_specs[0]` = left/−Y, `marker_specs[1]` = right/+Y. Wrong order → Franka +Y inferred backwards → swap to fix.

**EE frame gotcha (2026-05-26):**  OpenSai uses the Franka **link8** frame, not `panda_hand`. `panda_hand_joint` has `rpy="0 0 -0.7854"` — a built-in -45° Z rotation — so the paddle face direction (panda_hand +Y) is `[1/√2, 1/√2, 0]` in link8. Using either pure `+X` or pure `+Y` as the face normal gives a ±40–45° `rz` error in `cmd_arm_world_clean.py`. The correct `R_W_E_REF` is `[[1/√2, 1/√2, 0],[1/√2, -1/√2, 0],[0, 0, -1]]`.

**cmd_arm_world_clean.py — verification test procedure (2026-05-26):**

Pre-conditions: Redis running, OptiTrack streaming, OpenSai cartesian_controller running, arm free-driven to a comfortable mid-range pose facing opponent.

```bash
python sports_bot/scripts/cmd_arm_world_clean.py
```

Startup print should show `current orientation equiv: ori ~0 ~0 ~0` (within ±5° of all zeros if paddle is roughly facing opponent). If it shows a large rz (e.g. ±45°), the EE frame is wrong — check `R_W_E_REF` in the script.

**Orientation test sequence** — send at the `>` prompt, wait for `[current]` to show `ang_err < 3°`:

| Command | Expected physical motion | `[current]` ori target |
|---|---|---|
| `ori 0 0 0` | face toward opponent, handle down | `[+0°, +0°, +0°]` |
| `ori 0 20 0` | face tilts 20° downward (ry pitch) | `[~0°, ~+20°, ~0°]` |
| `ori 0 -20 0` | face tilts 20° upward | `[~0°, ~-20°, ~0°]` |
| `ori 0 0 20` | face yaws 20° left (+Y world) | `[~0°, ~0°, ~+20°]` |
| `ori 0 0 -20` | face yaws 20° right (-Y world) | `[~0°, ~0°, ~-20°]` |
| `ori 20 0 0` | handle sways sideways (roll) | `[~+20°, ~0°, ~0°]` |
| `ori 0 0 0` | back to reference | `[~0°, ~0°, ~0°]` |

**Position test sequence** — start from reference, each command after the previous settles:

| Command | Expected motion | Pass criterion |
|---|---|---|
| `r 0.1 0 0` | 10 cm toward opponent (+X) | `pos_err < 8 mm` at steady state |
| `r -0.1 0 0` | 10 cm back | returns to start |
| `r 0 0.1 0` | 10 cm left (+Y) | `pos_err < 8 mm` |
| `r 0 -0.1 0` | 10 cm right | returns to start |
| `r 0 0 0.1` | 10 cm up (+Z) | `pos_err < 8 mm` |
| `r 0 0 -0.1` | 10 cm down | returns to start |

**What to paste back for verification:** the full terminal block from startup through the first `[current]` steady-state line after each command (includes `[goal_W]`, `[goal_A]`, `[current]`, `[joints]`, `[ns_goal]`).

---

## Day-to-day bringup (SRC Kitchen, TidyBot `tidybot01`)

All services run on `tidybot01` → all default to `localhost`. Known IDs: Ball=8, TidyBot=11.

### Pre-flight checklist (easy to forget)

Before trusting Redis keys or running arm scripts (`stepj*`, `cmd_arm_world_clean.py`, etc.):

1. **Franka Desk → Execution mode** (not Programming). If the arm is in Programming, OpenSai may publish **stale** `joint_positions` and ignore goals.
2. **Motive → Streaming → Transmission Type → Unicast** when `tidybot01` is on SRC wifi / cross-subnet. Match the streamer flag: `u` not `m` (see Step 2).
3. **TidyBot base driver:** `cd ~/tidybot2 && conda activate tidybot2 && sh launch_driver.sh` (or `python redis_driver.py` if that's what the session uses).
4. **OpenSai arm controller:** `cd ~/OpenSai && ./scripts/launch.sh sports_bot/picklebot.xml` (starts Redis if needed + `OpenSai_main`).
5. **Sanity — sensors must be live**, not frozen:
   ```bash
   watch -n 0.2 "redis-cli GET 'opensai::sensors::FrankaRobot::joint_positions'"
   ```
   Move the arm by hand; values should change every frame. If identical across reads, OpenSai is not connected — fix steps 1 and 4 before debugging Python.

**Step 1 — Redis:**
```bash
redis-server
redis-cli ping  # → PONG
```

**Step 2 — OptiTrack streamer:**

On `tidybot01` (SRC subnet): Motive **Unicast** + streamer mode `u`. Multicast (`m`) only when laptop and Motive share the same L2 subnet *and* Motive is set to Multicast.

```bash
cd ~/OpenSai/sports_bot/optitrack && conda activate opensai
PYTHONPATH=drivers/PythonClient python -u StreamDataSkeleton.py \
    172.24.69.102 <tidybot01-IP> u    # 'u'=unicast (default on tidybot01), 'm'=multicast
# Or:
STREAMER_MODE=u ./sports_bot/scripts/record_throws.sh
# Sanity:
redis-cli get sai2::optitrack::rigid_body_pos::8   # ball
redis-cli get sai2::optitrack::rigid_body_pos::11  # cart
```

**Step 3 — TidyBot driver:**
```bash
cd ~/tidybot2 && conda activate tidybot2
sh launch_driver.sh          # preferred one-shot launcher
# or: python redis_driver.py
```

**Step 4 — Base bridge:**
```bash
cd ~/OpenSai && conda activate opensai
redis-cli del sports_bot::cmd::base::goal_pose   # avoid stale lurch
python sports_bot/base_bridge.py --robot-rigid-body-id 11
# Watch for T_W_R = (...) and Δ near zero in first sanity line.
```

**Step 5 — Send base goals:**
```bash
python sports_bot/scripts/send_base_goal.py --robot-rigid-body-id 11 --read
python sports_bot/scripts/send_base_goal.py --robot-rigid-body-id 11 \
    --x 1.0 --y 0.5 --yaw-deg 0
python sports_bot/scripts/send_base_goal.py --robot-rigid-body-id 11 \
    --relative --x 0.20 --y 0.10 --yaw-deg 15
```
Default tolerances: 100 mm / 5°. Tighten with `--pos-tol-mm 20 --yaw-tol-deg 2` after fresh bridge restart.

**Step 6 — OpenSai controller** (arm sessions only; Franka in **Execution** first):
```bash
cd ~/OpenSai
./scripts/launch.sh sports_bot/picklebot.xml
# launch.sh starts redis-server if not running, runs ./bin/OpenSai_main,
# and opens the web UI via tmux. Ctrl-C kills OpenSai_main cleanly.
#
# To run without the UI (background / headless):
#   ./bin/OpenSai_main sports_bot/picklebot.xml
#
# Verify controller is live:
redis-cli get opensai::controllers::FrankaRobot::cartesian_controller::cartesian_task::current_position
redis-cli get opensai::controllers::FrankaRobot::active_controller_name  # → "cartesian_controller"
```

**Step 7 — Verify arm calibration:**
```bash
python sports_bot/scripts/verify_arm_calibration.py  # prints T_W_A, T_W_E, T_W_P
# After the 2026-05-26 compliantFrame fix: T_W_E ≈ T_W_P (they coincide,
# T_E_P translation is now 0 — sweet spot IS the compliant frame).
```

**Step 8 — Manual arm world-frame commanding (diagnostic / setup):**
```bash
cd ~/OpenSai && conda activate opensai
python sports_bot/scripts/cmd_arm_world_clean.py
# Starts at current arm pose — no lurch.
# Prints current orientation as "ori rx ry rz" equivalent.
#
# Reference orientation (ori 0 0 0):
#   face toward +X opponent, handle pointing down (-Z world)
#
# Useful first commands:
#   0.3 0.0 0.8          move sweet spot to (0.3, 0, 0.8) world, keep ori
#   ori 0 0 0            snap to reference pose (face +X, handle down)
#   ori 0 10 0           pitch face 10° toward floor
#   0.3 0.0 0.8 0 0 0    move + set reference orientation simultaneously
#   r 0.0 0.0 -0.1       nudge 10 cm down, keep orientation
#   q                    quit (last goals stay in Redis)
```

**Step 9 — Arm + intercept loop:**
```bash
python sports_bot/scripts/base_intercept.py \
    --ball-rigid-body-id 8 --strike-plane-x 0.60 \
    --ready-x 0.0 --ready-y 0.0 --ready-yaw-deg 0 \
    --base-y-min -1.0 --base-y-max 1.0 \
    --arm-track     # omit for base-only
```

---

## Open items for integration

- **Daily calibration sanity check.** Place marker at origin floor mark, read raw pos, apply world transform, expect (0,0,0) within ~5 mm.
- **Re-derive `T_W_R` continuously in bridge.** `--periodic-refresh-s 0.5` handles held-goal drift; intra-move drift still accumulates. Per-tick recompute with OT-dropout guard is the proper fix.
- **Decide PickleBall's permanent Streaming ID** — easier to standardize on `1` than keep passing `--optitrack-rigid-body-id 8` everywhere.
- **EKF commit rule.** Wire FSM commit decision to `Intercept.position_cov` σ threshold (replace fixed `swing_commit_time_s = 0.20`). Plumbing is in place; FSM change not done.
- **Re-measure bounce params** after ≥50 more throws to tighten `μ_t` estimate (current std ≈ 13%).
- **Confirm `tidybot01` is on SRC subnet** — cleanest architecture: streamer + Redis + OpenSai all on cart; laptop reads remotely.

---

## Useful one-liners

```bash
redis-cli ping
redis-cli --scan --pattern 'sai2::optitrack::*' | xargs -r -I{} redis-cli del {}
watch -n 0.1 'redis-cli get sai2::optitrack::rigid_body_pos::8'
ping -c 2 172.24.69.102
ifconfig en0 | awk '/inet / {print $2}'
redis-cli get opensai::controllers::FrankaRobot::cartesian_controller::cartesian_task::current_position
redis-cli get opensai::controllers::FrankaRobot::cartesian_controller::cartesian_task::current_orientation
```

---

## Change log

- **2026-05-30** — Pre-flight checklist added to day-to-day bringup: Franka Execution mode, Motive unicast on tidybot01, `sh launch_driver.sh`, `./scripts/launch.sh`, live `joint_positions` sanity check.
- **2026-05-26** — Double-counted EE offset bug fixed. `picklebot.xml` `compliantFrame` moved from `xyz="0 0 0"` to `xyz="0 0 0.35"` — OpenSai now tracks the sweet spot directly in `current_position`/`goal_position`. `T_E_P.translation_m` zeroed in `arm_marker_calibration.json` (rotation kept for legacy callers). `PADDLE_TIP_OFFSET_M` corrected to `0.10` (from sweet spot, not raw flange). New `cmd_arm_world_clean.py`: no EMA/velocity-clamping/orientation-lock/T_E_P in loop; world-frame RPY commands (`ori rx ry rz`) with reference = face toward +X opponent, handle down; per-marker jump detection; both goals written every tick. OpenSai launch command (`./scripts/launch.sh sports_bot/picklebot.xml`) documented in bringup Step 6.
- **2026-05-23** — T_E_P corrected: translation `[0,0,0.368]→[0,0,0.35]`; rotation unchanged (face normal = EE +X — handle is along EE +Z, face is perpendicular). `PADDLE_TIP_OFFSET_M=0.45` added. `enforce_paddle_floor` replaces worst-case z_min computation with exact world-Z of commanded paddle tip using live T_W_A and R_A_E.
- **2026-05-22** — arm world-frame tracking landed. Marker-pair midpoint → T_W_A; T_E_P persisted in `arm_marker_calibration.json`; `world_racket_to_arm_ee` + `clip_to_arm_workspace` in `frames.py`. `base_bridge.py` `--periodic-refresh-s` added to kill held-goal drift. `base_intercept.py` `--arm-track` mode + world-frame tracking-error diagnostic.
- **2026-05-19** — base-frame calibration captured on cart. `T_B_C=(-0.035, +0.009, -7.16°)`, RMS 3.71 mm/0.37°. Bridge verified end-to-end; `send_base_goal.py` added. Switched to multicast streaming (laptop static IP `172.24.68.204`); patched macOS NatNet data socket bind bug. Phase 3b EKF implemented as analyzer tool (not FSM swap). Tracker switched to volley-only defaults for bring-up.
- **2026-05-18** — Phase 3a: online bounce-triggered history pruning. Commit-window mean 8.2→5.0 cm, max 73→31 cm.
- **2026-05-17** — Phases 1+2: bounce detection + empirical (e, μ_t) measurement; bounce-aware `predict_intercept`. Measured e=0.71, μ_t=0.64 (SRC Kitchen). BallTracker test harness + `record_throws.sh` + `watch_ball.sh`. `world_calibration.json` for SRC Kitchen (Z-up streaming confirmed). First OptiTrack streaming verified (unicast path from Stanford wifi).
