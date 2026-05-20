# sports_bot integration context

Living reference for the pickleball robot bringup. Append as we learn — don't rewrite
unless something is wrong. Each section has a date so we can see what's fresh.

---

## The system at a glance

```
Motive PC (bay)  ──NatNet UDP──▶  StreamDataSkeleton.py  ──redis.set──▶  Redis
                                  (NatNet client, your                   sai2::optitrack::
                                   laptop or mini-PC)                    rigid_body_pos::<id>
                                                                              │
                                                                              ▼
Redis  ◀──get──  pickleball_fsm.py  (BallTracker reads ball pose;
                                     SwingPlanner plans a hit;
                                     writes racket+base goals)
                                                                              │
                                                                              ▼
Redis  ──get──▶  OpenSai (mini-PC)            ──torques──▶  Franka arm
       ──get──▶  TidyBot redis_driver.py      ──vel──────▶  mobile base
                 (consumes hb1::desired_pose)
```

Two parties only talk through Redis. Bringing up streaming, the FSM, or the controller
side independently is fine — that's the whole point of the bus.

---

## Quick start: see the ball position on your laptop

Setup as of 2026-05-19: laptop on **SRC wifi** with static IP `172.24.68.204`,
Motive Kitchen PC in **multicast** mode. Run from the laptop:

```bash
cd "$(git rev-parse --show-toplevel)/sports_bot/optitrack"
conda activate opensai
PYTHONPATH=drivers/PythonClient python -u StreamDataSkeleton.py \
    172.24.69.102        \
    172.24.68.204        \
    m                    # 'u' = unicast, 'm' = multicast
```

Or — preferred — just use the recorder wrapper which brings the streamer up
itself and tears it down on exit:

```bash
./sports_bot/scripts/record_throws.sh                   # defaults: multicast, ID 8
STREAMER_MODE=u ./sports_bot/scripts/record_throws.sh   # only if Motive is in Unicast
```

Live view of every rigid body Redis is publishing (second terminal):

```bash
while true; do
  clear
  printf '=== %s ===\n' "$(date '+%H:%M:%S.%3N')"
  for k in $(redis-cli --scan --pattern 'sai2::optitrack::rigid_body_pos::*' | sort); do
    printf '%-50s %s\n' "$k" "$(redis-cli get "$k")"
  done
  sleep 0.1
done
```

To get your laptop's IP: `ifconfig en0 | grep "inet "`.

---

## Network & ports


| Item                   | Value           | Notes                                                                                                       |
| ---------------------- | --------------- | ----------------------------------------------------------------------------------------------------------- |
| Kitchen Motive server  | `172.24.69.102` | from src_mocap/README. Other bays have different IPs.                                                       |
| NatNet command port    | `1510`          | UDP, bidirectional. Survives cross-subnet routing.                                                          |
| NatNet data port       | `1511`          | UDP. In multicast, listened to on `239.255.42.99:1511`.                                                     |
| NatNet multicast group | `239.255.42.99` | Default. Multicast does NOT route across subnets.                                                           |
| VRPN port              | `3883`          | A *different* protocol. Disabled in Motive currently. The UI puts it next to NatNet which causes confusion. |


**Stanford wifi → SRC subnet routing:** unicast UDP works (we get ~8 ms ping to
`172.24.69.102`). Multicast does **not** — multicast packets stay inside the L2
domain of the SRC switch.

If you want multicast (the supported long-term path), you need:

1. Your laptop's MAC registered with Zen (`zyaskawa@stanford.edu`) to get a static
  IP on the SRC subnet, **and**
2. To be associated with the `SRC` wifi SSID.

---

## NatNet protocol cheatsheet

NatNet has **two independent UDP channels**:

- **Command channel (`:1510`)** — bidirectional unicast. Handshakes, queries,
DataDescriptions ("list of rigid bodies + names + IDs"). Always works
cross-subnet.
- **Data channel (`:1511`)** — server → client, per-frame rigid body / marker /
skeleton data. Delivered as either **multicast** (one copy on the wire, same
subnet only) or **unicast** (one copy per client, routes anywhere). Picked in
Motive's Streaming → Transmission Type.

If you can `nc` the command port and get DataDescriptions back but no positions
ever arrive, it's almost always the data channel: server is multicasting and you
can't hear the group.

The per-frame stream carries **numeric IDs**, not names. Names live in
DataDescriptions, which is fetched separately. That's why a frame-level viewer
shows `rigid_body 8` instead of `PickleBall`.

---

## Motive configuration (Kitchen, as of 2026-05-19)

- Streaming → NatNet — **enabled**, Transmission Type **Multicast** (group
`239.255.42.99`, data port `1511`). Switched back from Unicast on
2026-05-19 once Zen issued a static SRC IP for the laptop
(`172.24.68.204`).
- Streaming → VRPN — disabled. Its "Broadcast Port: 3883" is a red herring; it
belongs to VRPN, not NatNet.
- Assets pane — `PickleBall` rigid body, **Streaming ID = 8**. (Default Streaming
ID is the asset's row position in the pane. Can be overridden in the asset's
Properties → User Data field.)
- KVM access for the Kitchen Motive PC: `SRC-KVM-Kitchen.stanford.edu`. Credentials
from Zen.

**Heads up:** multicast packets stay inside the L2 domain of the SRC switch —
they do **not** route across subnets. The laptop must be associated with SRC
wifi *and* using its registered static IP for multicast to work. If you ever
need to record from Stanford-wifi again (off-subnet), flip Motive's
Transmission Type to Unicast and run the streamer with `STREAMER_MODE=u`.

---

## Redis key schema

Published by `StreamDataSkeleton.py` (per rigid body per frame, ~120 Hz):


| Key                                          | Format                  | Frame                  |
| -------------------------------------------- | ----------------------- | ---------------------- |
| `sai2::optitrack::rigid_body_pos::<id>`      | JSON `[x, y, z]`        | **World** (calibrated) |
| `sai2::optitrack::rigid_body_ori::<id>`      | JSON `[qx, qy, qz, qw]` | World quat             |
| `sai2::optitrack::raw::rigid_body_pos::<id>` | JSON `[x, y, z]`        | Motive room frame      |
| `sai2::optitrack::raw::rigid_body_ori::<id>` | JSON `[qx, qy, qz, qw]` | Room quat              |


World transform is `R_WORLD_OPTI · p_opti + T_WORLD_OPTI`, loaded from
`sports_bot/optitrack/world_calibration.json`. If the file is missing, identity
is used and the printout `[optitrack] no calibration file …, using identity`
fires at startup.

**Frame conventions:**

- OptiTrack (Motive) **streaming** frame: configured to **Z-up**, right-handed
(see gotcha below — this is independent of Motive's *display* Up Axis).
- World frame (what the FSM, controllers, and URDF use): **Z-up**, right-handed,
origin at robot home. +X = toward opponent, +Y = robot's left, +Z = up. The
FSM constants in `state_machine/config.py` all assume this convention.
- `world_calibration.json` is just yaw + translation — both frames are Z-up so
no axis swap is needed.

**Critical Motive gotcha — display vs streaming Up Axis are independent.**
Motive has *two* "Up Axis" settings:

1. `View → Up Axis` controls what the 3D perspective view shows. Cosmetic.
2. `Edit → Application Settings → Streaming → Up Axis` controls what the NatNet
  stream actually publishes. **This is the one that matters for Redis.**

These can disagree silently — the perspective view will happily show Y-up while
streaming emits Z-up, and nothing in the UI warns you. Always confirm the
*streaming* setting at session start; if someone flipped it, the world
calibration will silently produce garbage. The cleanest sanity check is to
place the ball on the origin floor marker and read
`sai2::optitrack::raw::rigid_body_pos::8` — the vertical component should be
the small one (~ball radius), and the other two should be ~2.5 m and ~1.2 m.

**Current calibration (2026-05-17, SRC Kitchen):** 2D Procrustes from the
3 floor markers, assuming Motive streaming Z-up. Max horizontal residual 0.34 mm,
max vertical residual 1.87 mm (floor unevenness, not noise).

Consumed by `pickleball_fsm.py` via `BallTracker`:

```
--ball-source optitrack --optitrack-rigid-body-id 8
```

Reads `sai2::optitrack::rigid_body_pos::8` (world frame, not raw).

Other relevant keys (for full system integration):


| Key                                                                                   | Owner               | Purpose                                         |
| ------------------------------------------------------------------------------------- | ------------------- | ----------------------------------------------- |
| `opensai::controllers::Panda::cartesian_controller::cartesian_task::goal_position`    | FSM → OpenSai       | Racket goal pos (opensai backend)               |
| `opensai::controllers::Panda::cartesian_controller::cartesian_task::goal_orientation` | FSM → OpenSai       | Racket goal rot                                 |
| `sports_bot::cmd::base::goal_pose`                                                    | FSM → controller    | `[x, y, theta]` base goal (cs225a backend)      |
| `hb1::desired_pose`                                                                   | TidyBot driver      | `[x, y, theta]` the wheel driver actually reads |
| `hb1::current_pose` / `hb1::current_vel`                                              | TidyBot driver → us | Base feedback                                   |
| `hb1::kill` / `hb1::stop`                                                             | us → TidyBot driver | "kill" terminates driver; "stop" decelerates    |


---

## Gotchas hit so far

- **Two copies of `StreamDataSkeleton.py`** in the repo. The one in
`sports_bot/optitrack/StreamDataSkeleton.py` is the maintained version (has
world calibration + rigid_body_listener already enabled). The one in
`sports_bot/optitrack/drivers/PythonClient/StreamDataSkeleton.py` is the
vanilla NatNet SDK sample — older, has `rigid_body_listener` commented out.
Use the top-level one. It needs `PYTHONPATH=drivers/PythonClient` because the
NatNet SDK modules live there.
- `**python -u` matters** if you want to see prints in tail logs / background
runs; without it Python buffers stdout aggressively when not on a TTY.
- `**request_data_descriptions`** crashes the NatNet SDK on a UTF-8 decode of
marker names (NatNetClient.py:965). Doesn't affect frame streaming. Live with
it for now.
- `**set_use_multicast(False)` is not enough** if Motive itself is in Multicast
mode — the client requests unicast but Motive just doesn't send it. Both sides
have to agree.
- **macOS multicast bind silently drops frames** (patched 2026-05-19). The
vendored NatNet client used to bind the data socket to
`(self.local_ip_address, 1511)`. On BSD-derived stacks (macOS) that prevents
delivery of multicast packets even though the IGMP join succeeds — `NAT_CONNECT`
on the command port handshakes fine, server version arrives, but no data
frames ever do. Fixed by binding to `('', 1511)` (INADDR_ANY) instead — the
`IP_ADD_MEMBERSHIP` `setsockopt` above the bind still pins which interface
joins the group, so we don't lose any selectivity. Patch is in
`drivers/PythonClient/NatNetClient.py:__create_data_socket`. Symptom if it
ever regresses: streamer log shows `resetting requested version to 4 2 0 0
from 0 0 0 0` (= command handshake worked) but no rigid-body Redis keys
appear; switch to `STREAMER_MODE=u` (with Motive flipped to Unicast) to
confirm.
- **`sports_bot/` is its own git repo nested inside OpenSai.** So
`git rev-parse --show-toplevel` from anywhere under `sports_bot/` returns
`<OpenSai>/sports_bot`, not `<OpenSai>`. Helper scripts that need the OpenSai
root must resolve it from `$0` (e.g. `$(cd "$(dirname "$0")/../.." && pwd)`),
not from git. Running `python -m sports_bot.<module>` requires `cwd` to be
the OpenSai root.

---

## Ball tracker / intercept-prediction test harness

`state_machine/ball_tracker_test.py` lets us record ball trajectories from
Redis to disk and replay them through the *real* `BallTracker` to evaluate
the intercept prediction visually + numerically. The FSM is not involved —
this is a pure tracker/predictor test bench.

Files:

| Path | Purpose |
|---|---|
| `state_machine/ball_tracker_test.py` | `record` + `analyze` subcommands. |
| `scripts/record_throws.sh`           | One-shot: starts the OptiTrack streamer if it isn't already up, runs the recorder, cleans up the streamer on exit. |
| `scripts/watch_ball.sh`              | macOS-friendly replacement for `watch -n 0.1 redis-cli ...` — no GNU `watch` or `date %N` needed. |
| `recordings/`                        | Default save directory (auto-created). Filenames `throws_<YYYYMMDD_HHMMSS>.npz`. |

**Record a session:**

```bash
conda activate opensai
./sports_bot/scripts/record_throws.sh                      # ID 8, auto-named output
RIGID_BODY_ID=1 ./sports_bot/scripts/record_throws.sh      # different rigid body
./sports_bot/scripts/record_throws.sh -o myrun.npz         # explicit path
./sports_bot/scripts/record_throws.sh --min-movement 0.002 # any record flag passes through
```

Hold the ball still for ~1 s between picking it up and throwing, and between
throws — creates segmenter-friendly time gaps so walking-with-ball samples
don't leak into a throw's fit window.

**Sanity check the Redis stream (macOS-safe):**

```bash
./sports_bot/scripts/watch_ball.sh 8        # just the pickleball
./sports_bot/scripts/watch_ball.sh          # all rigid bodies
```

**Analyze a recording:**

```bash
# Most recent recording, with viser viewer at http://localhost:8080
python -m sports_bot.state_machine.ball_tracker_test analyze \
    "$(ls -t sports_bot/recordings/throws_*.npz | head -1)"

# Stats only, no viewer
python -m sports_bot.state_machine.ball_tracker_test analyze \
    sports_bot/recordings/<file>.npz --no-viser

# A/B-tune tracker params on the same recording (no re-throwing)
python -m sports_bot.state_machine.ball_tracker_test analyze \
    sports_bot/recordings/<file>.npz \
    --history-size 8 --max-position-jump 0.3 --gravity 9.81
```

Throws are auto-segmented from one continuous recording by 0.4 s time gaps
(`--segment-gap-s`). Output: a per-throw error table bucketed by
time-to-impact at the moment of prediction. The `[0.00, 0.15) s` bucket is the
one the FSM actually acts on (it commits to SWING ~0.20 s before impact).

**Reading the viser plot:**

- Yellow translucent rectangle = strike plane (`court.strike_plane_x`, default 0.60 m).
- Grey dots = recorded ball trajectory.
- Cyan dots = fit-window samples driving `_fit_state` at the current tick.
- Green spline + green sphere = current predicted ballistic arc + intercept.
- Yellow sphere = actual recorded crossing of the strike plane.
- Red→blue point cloud = every per-tick predicted intercept, colored by
  time-to-impact at the time of prediction (**red = close to impact** /
  short lookahead, **blue = far from impact** / long lookahead).

The `_ReplayTracker` subclass in `ball_tracker_test.py` shares
`_fit_state` / `predict_intercept` / `is_incoming` with the production
`BallTracker` and only overrides the ingest layer (so we test the production
code path, not a copy of it).

---

## What we're debugging now (2026-05-17)

**Bounce-blind trajectory fitter.** `BallTracker.predict_intercept` fits a
single ballistic arc (constant v in xy, free-fall in z) over the last ≤12
samples within a 0.30 s window. This breaks the moment the recorded
trajectory spans a bounce: the linear least-squares fit averages pre- and
post-bounce velocities, producing garbage. The fitter is also floor-blind —
the parabola is happily extrapolated through z = 0, which is why many
predicted intercepts in the viser plots end up well below the floor. Verified
visually 2026-05-17: scrubbing into the post-bounce period of a recorded
throw lands the cyan fit-window across the bounce, and the green prediction
dives below z = 0.

Real pickleball returns are mostly post-bounce groundstrokes, so this has to
be fixed before the robot can do anything beyond hitting volleys.

Planned work:

- [x] **Phase 1 — bounce detection + empirical parameter measurement.**
      `_detect_bounces` in `ball_tracker_test.py` finds floor bounces as
      local z-minima below ~10 cm with a downward → upward `v_z` flip,
      then fits ballistic velocity windows on either side (using a
      gravity-aware quadratic-in-z linear fit, same physics model as
      `_fit_state`). Per-bounce `(e, μ_t)` is printed in the throw summary
      and an aggregate `mean / median / std` is reported with suggested
      config values. Bounces show as orange spheres in viser with `(e,
      μ_t)` labels. Verified on synthetic ground-truth data: recovers
      `e=0.70` and `μ_t=0.90` to within 2 % across 4 bounces.

- [x] **Phase 2 — bounce-aware `predict_intercept`.**
      `_propagate_to_plane` in `ball_tracker.py` propagates the fitted
      ballistic state forward; if `z` would hit 0 before `x` reaches the
      strike plane, reflects `v_z` with `cfg.bounce_restitution` and
      scales `v_xy` with `cfg.bounce_tangential_damping`, then continues
      until either the plane is reached or `cfg.max_bounces` is exhausted.
      `Intercept` now carries `n_bounces`. `max_bounces=1` by default.
      Verified on a synthetic groundstroke that reaches `x=0.6` after one
      bounce: 3× as many usable predictions vs. the no-bounce baseline,
      and the final `[0.00, 0.15) s` bucket lands at 3-5 mm error.

      Known limitation (the Phase 3 problem): predictions made while the
      fit window straddles a bounce are still poisoned, because the
      least-squares fit averages pre- and post-bounce velocities. Bucket
      errors of 150-170 mm mean / 700-800 mm max in the `[0.30, 0.50) s`
      window are this exact failure mode.

**Measured bounce parameters (SRC Kitchen, 2026-05-17).** Aggregated `e`
and `μ_t` across all recordings in `sports_bot/recordings/`:

| metric | restitution `e` | tangential `μ_t` |
|---|---|---|
| n bounces | 11 (across 19 throws, 12 recordings, 12,116 samples) | same |
| mean      | 0.712 | 0.640 |
| median    | 0.690 | 0.606 |
| std       | 0.058 | 0.127 |
| min, max  | 0.666, 0.841 | 0.492, 0.932 |
| trimmed mean (10 % each side) | 0.703 | 0.624 |

`e` is tight (std ≈ 6 %, one outlier at 0.841 likely a glancing /
mis-fit bounce). `μ_t` is much noisier (std ≈ 13 %), which is expected —
it lumps friction with spin coupling and is highly throw-dependent.
**Defaults updated in `config.py`: `bounce_restitution = 0.70`,
`bounce_tangential_damping = 0.62`** (medians, rounded). Worth
re-measuring after every ~50 new throws to tighten the estimate.

- [x] **Phase 3a — online bounce-triggered history pruning.** Implemented
      in `BallTracker._try_prune_pre_bounce`: every tick after a sample is
      appended, scan the rolling history for a clear floor-bounce
      local-min (z below `online_bounce_z_threshold = 0.10 m`,
      strictly-greater z 2 samples on either side). If found, drop all
      pre-bounce samples from `_history`. The next `_fit_state` then
      operates only on the new ballistic arc. Conservative detection on
      purpose — false positives truncate the fit window. Toggle with
      `cfg.tracker.online_bounce_pruning` (default `True`) or the
      analyzer's `--no-online-bounce-pruning` for A/B.

      Measured impact on the SRC Kitchen recording set (12 recordings,
      19 throws with crossings):

      | tti window | mean err 3a-off → 3a-on | max err 3a-off → 3a-on |
      |---|---|---|
      | `[0.30, 0.50) s` | 0.160 → 0.144 m | 1.24 → 0.86 m |
      | `[0.15, 0.30) s` | 0.170 → 0.117 m | 1.02 → 1.42 m |
      | `[0.00, 0.15) s` (commit-to-swing) | **0.082 → 0.050 m** | **0.73 → 0.31 m** |

      Commit-window mean drops 40 %, max drops 58 %. The 5 cm mean error
      at swing-commit is now under the paddle face. Early buckets
      (`tti ≥ 0.5 s`) are unchanged because the bounce isn't in history
      yet — that's the regime Phase 3b/EKF would help.

- [x] **Phase 3b — Bayesian state estimator (KF over `(p, v)` with
      bounce jumps).** Implemented in
      `state_machine/ekf_ball_tracker.py` (`EKFBallTracker`) on 2026-05-19,
      with the **same public interface** as the LS `BallTracker`
      (`update` / `is_incoming` / `predict_intercept` / `_fit_state` /
      `reset`) so the FSM can swap backends without code changes. The FSM
      is **not** swapped yet — it's a testing tool A/B-able from the
      analyzer. State = `[px, py, pz, vx, vy, vz]`; dynamics are linear
      between bounces (constant-v in xy, free-fall in z), so the "EKF"
      is really a linear KF with discrete bounce jumps. In
      `predict_intercept`, each predicted floor crossing applies the
      same `(e, μ_t)` state jump as the LS code path, *plus* a
      covariance inflation from `σ_e ≈ 0.058`, `σ_{μ_t} ≈ 0.127`
      (measured Phase 1). Online bounce detection inside `update`
      mirrors the LS pruner — instead of dropping pre-bounce samples,
      it re-seeds the filter from the post-bounce ones. `Intercept`
      gained an optional `position_cov: Optional[np.ndarray]` (3×3) for
      downstream use; the LS tracker leaves it `None`. New
      `BallTrackerConfig.ekf: EKFConfig` carries all the EKF tunables.

      A/B from the analyzer:

      ```bash
      python -m sports_bot.state_machine.ball_tracker_test analyze <rec> \
          --tracker {leastsq,ekf}
      ```

      with overrides for `--ekf-process-accel-std-{xy,z}`,
      `--ekf-measurement-pos-std`, `--ekf-seed-samples`, and
      `--no-ekf-bounce-handling`. viser renders a translucent 1σ
      position ellipsoid at the predicted intercept when
      `--tracker ekf` is active.

      **A/B on the recording set (12 recordings, 19 throws, default
      EKF config):** essentially tied with the LS+pruning baseline at
      the swing-commit bucket the FSM actually acts on.

      | tti window | LS mean | EKF mean | Δ |
      |---|---|---|---|
      | [0.00, 0.15) s (commit) | **5.0 cm** | 5.8 cm | +0.8 cm |
      | [0.15, 0.30) s          | 11.7 cm | 13.5 cm | +1.8 cm |
      | [0.30, 0.50) s          | 14.4 cm | 20.6 cm | +6.3 cm |
      | [0.50, 0.70) s          | 28.3 cm | 27.3 cm | −1.0 cm |
      | [0.70, 1.00) s          | 25.2 cm | 20.8 cm | **−4.4 cm** |
      | [1.00, 1.50) s          | 24.7 cm | 32.5 cm | +7.8 cm |

      So the EKF on its own is **not** an obvious improvement at
      swing-commit on this dataset — within 1 cm of the tuned
      LS+pruning result. The headline payoff the EKF unlocks is what's
      *enabled*: an uncertainty-aware commit rule (swing when the
      predicted intercept's position σ drops below a paddle tolerance,
      instead of on a fixed `swing_commit_time_s = 0.20`). The
      `Intercept.position_cov` plumbing is in place; the FSM change is
      not done.

      **Tuning gotchas hit during bring-up (worth remembering before
      tuning further):**

      - *Mahalanobis outlier gating self-destructs at low `σ_meas`.*
        With `σ_meas = 2 mm`, a slightly biased seed velocity makes
        innovation cov `S = P + R` tiny, and innovations of just
        20–30 mm trip a 5.5σ gate. Empirically observed 95% rejection
        rate on real recordings → filter locks into the seed velocity
        and never recovers. Defaults are now `σ_meas = 5 mm` and
        threshold `= 20`; the shared `max_position_jump` filter still
        catches gross teleports.
      - *Seed quality matters more than expected.* A 4-sample LS seed
        is too noisy in velocity, and a low-`σ_meas` recursive update
        corrects `v` slowly (poor observability of `v` from `p` alone
        — the position-velocity correlation P[0,3] collapses fast).
        Bumped `seed_samples` default 4 → 6 (~50 ms at 120 Hz).
      - *No automatic state expiry across idle gaps.* The LS tracker
        self-resets via `history_max_age_s`; the EKF carries the
        posterior forward forever. The analyzer now explicitly calls
        `tracker.reset()` on any inter-sample gap > 0.30 s. If the FSM
        is ever swapped to the EKF, `_step_recover` already calls the
        equivalent reset, so we're fine there — but anywhere else long
        gaps can happen would need the same.

      Open work if we want to push the EKF further:

      - Apply the bounce *jump* (with covariance inflation) to the
        existing posterior instead of re-seeding from post-bounce
        samples — preserves pre-bounce xy information that the current
        approach throws away.
      - Wire the FSM commit decision to a covariance-based rule
        (replace `swing_commit_time_s` with a position-σ threshold).
      - Re-measure `(σ_e, σ_μ_t)` once we have ≥ 50 bounces; the
        current 13% std on `μ_t` inflates the predicted intercept
        ellipsoid more than it probably should.

---

## Base frame calibration (FSM goals → TidyBot driver)

The FSM speaks world frame W (floor tape). The TidyBot driver speaks robot
odometry frame R (origin = wherever the cart was parked when
`redis_driver.py` started; re-anchors every driver restart). `base_bridge.py`
sits between them. To do its job it needs the per-session transform
`T_W_R`. We split that into two pieces.

**Four frames, two unknown transforms:**

| frame | what / where |
|---|---|
| W | world; floor tape origin; FSM speaks this |
| B | OptiTrack rigid-body local frame, glued to the markers on the cart; pose in W comes from `sai2::optitrack::rigid_body_pos::<id>` + `…ori::<id>` |
| C | TidyBot odometry control-point frame, glued to the cart at whatever pivot the firmware tracks; pose in R comes from `hb1::current_pose` |
| R | robot odometry origin; floor-fixed for the session |

`T_W_B(t)` and `T_R_C(t)` are streamed live. The two unknowns are
`T_B_C` (static — depends on where the markers sit on the cart vs.
the firmware's odometry pivot) and `T_W_R` (static per session — depends on
where the cart was parked at driver start). They satisfy

```
T_W_B(t) ⊕ T_B_C  =  T_W_R ⊕ T_R_C(t)        for all t
```

(same point on the cart, two ways to express its world pose).

**Why split:** `T_B_C` is geometric and persists across sessions; `T_W_R`
re-anchors every driver restart and is meaningless tomorrow. So we
calibrate `T_B_C` once (the slow drive-around) and back-solve `T_W_R` from
one stationary snapshot every session (the fast bringup).

**Files:**

| Path | Purpose |
|---|---|
| `sports_bot/utils/frames.py` | SE(2) algebra, quat ↔ R, projection-based yaw extraction, 2D hand-eye AX=XB solver, calibration file I/O. |
| `sports_bot/scripts/calibrate_robot_marker.py` | Interactive N-waypoint capture; solves `(T_B_C, T_W_R)` jointly via Levenberg-Marquardt; persists only `T_B_C`. |
| `sports_bot/optitrack/robot_marker_calibration.json` | Persisted `T_B_C`. Re-solve only when markers are re-stuck. |
| `sports_bot/base_bridge.py` | Loads `T_B_C`, snapshots `(T_W_B, T_R_C)` for 1 s, derives `T_W_R = T_W_B ⊕ T_B_C ⊕ T_R_C⁻¹`, forwards goals, prints OT-vs-odom cross-check every 2 s. |

**One-time `T_B_C` calibration** (re-run when markers get re-stuck):

```bash
conda activate opensai
# 1. start redis, OptiTrack streamer, TidyBot driver
# 2. drive cart to 5 waypoints — mix translation AND rotation between them
python sports_bot/scripts/calibrate_robot_marker.py --robot-rigid-body-id <ID>
```

Each waypoint: hold the cart still, press Enter, 1 s average is captured.
Aim for ≥30 cm translation and ≥30° rotation span across the set —
the script warns if diversity is poor. Residual RMS should land under
~1 cm / 0.5°; the script flags higher values. Persists to
`sports_bot/optitrack/robot_marker_calibration.json`. Commit it.

**Per-session bringup:**

```bash
python sports_bot/base_bridge.py --robot-rigid-body-id <ID>
```

2-second pose snapshot (cart sits still) → derives `T_W_R` → runs the
bridge. Every ~2 s it prints a sanity line comparing `T_W_C` computed two
ways:

```
[base_bridge] sanity  W_via_OT=[+1.234, +0.567, +12.3°]  W_via_odom=[+1.235, +0.566, +12.2°]  Δ=[+1.0, -1.0] mm, -0.1°
```

Growing Δ over time ≈ wheel slip or marker shift. Free runtime watchdog;
not load-bearing, but useful for debugging.

**Quat passthrough is fixed**: `_opti_to_world_quat()` now actually
rotates the quaternion through `R_WORLD_OPTI`, so
`sai2::optitrack::rigid_body_ori::<id>` is genuinely in W. Both the
calibration script and the bridge read this world-frame key directly.

---

## Open items for integration

- ~~`world_calibration.json` for whichever bay we end up using~~ — done for
SRC Kitchen 2026-05-17. Re-do per bay (and after any Motive ground-plane
or camera recalibration).
- **Daily calibration sanity check.** Place a marker at the origin floor
mark, read `sai2::optitrack::raw::rigid_body_pos::<id>` and run it through
`R, t`; expect world-frame `(0, 0, 0)` within ~5 mm. If it drifts,
re-solve. (Better: group the 3 floor markers into a single `FloorRef`
rigid body and read its world pose each session.)
- ~~`_opti_to_world_quat()` is a passthrough~~ — fixed 2026-05-19. The
function now applies `R_WORLD_OPTI` to the quaternion via
`sports_bot.utils.frames.rotate_quat`, so `sai2::optitrack::rigid_body_ori::<id>`
is honestly in world frame. Downstream consumers (base bridge, marker
calibration) read this key directly instead of re-rotating raw OptiTrack
quats themselves.
- Make the FSM write to `hb1::desired_pose` instead of
`sports_bot::cmd::base::goal_pose` (or add a one-line bridge).
- Decide PickleBall's permanent Streaming ID — easier to standardize on `1`
in Motive than to keep passing `--optitrack-rigid-body-id 8` everywhere.
- ~~Register laptop MAC addresses with Zen → static SRC IPs → switch
Motive back to multicast~~ — done 2026-05-19. Laptop static IP is
`172.24.68.204` on the SRC subnet; Motive's Streaming →
Transmission Type is set to Multicast. The streamer defaults to
multicast (`scripts/record_throws.sh` uses `STREAMER_MODE=m` unless
overridden). Override with `STREAMER_MODE=u` if Motive is ever
flipped back to Unicast.
- Confirm whether the cart's mini-PC is on SRC and could run the streamer
itself (cleanest architecture: streamer + Redis + OpenSai all on the cart;
laptop reads from cart's Redis remotely).

---

## Useful one-liners

```bash
# Is Redis up?
redis-cli ping

# Clear stale OptiTrack keys
redis-cli --scan --pattern 'sai2::optitrack::*' | xargs -r -I{} redis-cli del {}

# Watch a single rigid body
watch -n 0.1 'redis-cli get sai2::optitrack::rigid_body_pos::8'   # GNU watch, brew install if missing

# Confirm cross-subnet routing works (should be a few ms RTT)
ping -c 2 172.24.69.102

# What's my wifi IP?
ifconfig en0 | awk '/inet / {print $2}'
```

---

## Change log

- 2026-05-19 — **base-frame calibration system landed.** Split the
  FSM↔TidyBot frame problem into two unknowns: `T_B_C` (geometric,
  marker-vs-odometry-pivot offset; persisted in
  `sports_bot/optitrack/robot_marker_calibration.json`) and `T_W_R`
  (per-session, where the cart was parked at driver start; derived from
  a 1-s snapshot at bridge startup). New `sports_bot/utils/frames.py`
  carries SE(2) algebra + a 2D AX=XB hand-eye solver via
  Levenberg-Marquardt — solver verified to machine precision against
  synthetic ground truth. New interactive
  `sports_bot/scripts/calibrate_robot_marker.py` captures N waypoints
  (default 5; mix rotation+translation) and reports per-waypoint
  residuals. Refactored `sports_bot/base_bridge.py` to load `T_B_C`,
  back-solve `T_W_R`, and print a periodic OT-vs-odom cross-check
  residual as a wheel-slip / marker-drift watchdog. Also fixed
  `_opti_to_world_quat()` in `StreamDataSkeleton.py` (was a passthrough)
  via `sports_bot.utils.frames.rotate_quat`, so
  `sai2::optitrack::rigid_body_ori::<id>` is now honestly world-frame.

- 2026-05-17 — first draft. Got OptiTrack streaming working from Stanford wifi to
Kitchen bay by switching Motive to Unicast. Confirmed PickleBall = Streaming
ID 8. Identified VRPN port (3883) vs NatNet (1511) confusion in Motive UI.
Streamer + Redis end-to-end verified; FSM and robot integration untouched.
- 2026-05-17 — added `world_calibration.json` for SRC Kitchen. Constrained
Procrustes from 3 taped floor markers (origin / 1 m forward / 1 m right).
Motive is Y-up, our world is Z-up; the calibration absorbs the axis swap so
FSM/sim/URDF stay Z-up unchanged.
- 2026-05-17 — discovered Motive's *streaming* Up Axis was Z, not Y (the View
Up Axis was Y, which is what we'd been looking at). Re-solved
`world_calibration.json` for Z-up streaming: now just yaw + translation, no
axis swap. Added the "display vs streaming Up Axis" gotcha to the conventions
section.
- 2026-05-17 — built the BallTracker test harness
(`state_machine/ball_tracker_test.py` + `scripts/record_throws.sh` +
`scripts/watch_ball.sh`). Recorded the first dataset of real throws against
SRC Kitchen OptiTrack. Confirmed visually in viser that the production
`predict_intercept` is bounce-blind and floor-blind: post-bounce extrapolation
of a single-arc ballistic fit produces predicted intercepts well below `z=0`.
Opened the Phase 1 → Phase 3 plan in "What we're debugging now".
- 2026-05-17 — landed **Phase 1** (bounce detection + empirical `(e, μ_t)`
measurement in the analyzer) and **Phase 2** (bounce-aware
`predict_intercept` in production). Added `bounce_restitution`,
`bounce_tangential_damping`, `max_bounces`, `floor_epsilon` to
`BallTrackerConfig`; added `n_bounces` to `Intercept`. New CLI flags on
`analyze`: `--bounce-restitution`, `--bounce-tangential-damping`,
`--max-bounces`, `--no-bounce-detection`. viser scene now shows detected
bounces as orange spheres with `(e, μ_t)` labels, and the predicted-arc
spline bends at predicted bounces. Verified on synthetic data; need to
run on the real recordings next to nail down SRC Kitchen `(e, μ_t)`.
- 2026-05-17 — measured `(e, μ_t)` on the SRC Kitchen recording set
(11 bounces / 19 throws / 12 recordings) and updated `config.py`
defaults to `bounce_restitution = 0.70`, `bounce_tangential_damping =
0.62` (medians of measurements). See the table under "What we're
debugging now" for the full distribution.
- 2026-05-18 — landed **Phase 3a** (online bounce-triggered history
pruning) in `BallTracker._try_prune_pre_bounce`. New config knobs:
`online_bounce_pruning` (default `True`), `online_bounce_z_threshold`.
Measured on the recording set: commit-to-swing window mean error
dropped 8.2 → 5.0 cm and max dropped 73 → 31 cm. Phase 3b (full EKF
with uncertainty ellipsoids) deferred until the rest of the FSM is
end-to-end on the cart.
- 2026-05-19 — **switched to multicast streaming.** Zen issued a static
SRC IP for the recording laptop (`172.24.68.204`). Motive's
Streaming → Transmission Type flipped from Unicast back to Multicast
(group `239.255.42.99`, port `1511`). `scripts/record_throws.sh`
default is now `STREAMER_MODE=m`; flip to `STREAMER_MODE=u` if
Motive is ever switched back. Also made `MY_IP` env-overridable for
the rare case `en0` isn't the SRC interface. No calibration impact —
the world-frame transform is independent of transmission type.
First multicast run hit a macOS-specific bug in the vendored NatNet
client (data socket bound to `(local_ip, 1511)` doesn't receive
multicast on BSD stacks — command channel handshakes but no frames
arrive). Patched `NatNetClient.__create_data_socket` to bind to
`('', 1511)` instead; `IP_ADD_MEMBERSHIP` still pins the interface
joining the group. Multicast confirmed end-to-end the same day.
Hardened `record_throws.sh` cleanup along the way: reap stale
`StreamDataSkeleton.py` processes at startup, escalate SIGTERM →
SIGKILL if a Python child is stuck in a socket-retry loop, and bail
early if the streamer dies before publishing rather than waiting the
full 10 s.
- 2026-05-19 — **diagnostic + filtering pass on the LS tracker** ahead of
the OptiTrack 240 Hz bump. Three changes, all behind config flags so
the existing FSM path is unchanged unless explicitly overridden:
   1. **Rejection reasons surfaced.** `BallTracker.predict_intercept`
      and `EKFBallTracker.predict_intercept` now set
      ``self.last_reject_reason`` to one of
      ``insufficient_history / not_incoming / past_plane / on_floor /
      would_bounce / tti_too_short / tti_too_long`` whenever they
      return `None`. The FSM ignores this; the analyzer surfaces it in
      a new GUI text panel ("reject reason") and color-codes the
      recorded trajectory by per-tick reason
      (green = OK, red = volley filter, amber = not incoming, blue =
      tti too long, etc.). New checkbox "Trajectory: color by
      rejection reason" in the Display folder; legend folder lists
      the color → reason mapping. Internals: ``_propagate_to_plane``
      now returns a `REJECT_*` string on failure instead of bare
      `None` so the caller can distinguish bounce-blocked from
      past-plane.
   2. **Per-axis median filter on raw OptiTrack samples** before they
      enter `_history`. New `BallTrackerConfig.median_filter_window`
      (default 3, 0/1 disables); applies to both LS and EKF trackers.
      Kills single-sample marker swaps / reflection artifacts at the
      cost of ~1 sample of lag in reported "current" position. On
      clean recordings the effect is in the noise (3.3 → 4.0 cm at
      commit on the test recording — slight regression because there
      were no outliers to filter); on real-cart data with marker
      occlusion, expected to be a clear win. Analyzer override flags:
      `--median-filter-window N` and `--no-median-filter`.
   3. **History defaults retuned for 240 Hz prep.** Previous
      `history_size = 8, history_max_age_s = 0.20` (120 Hz sweet
      spot) gives a 33 ms LS window at 240 Hz — too short, throws
      away the sample-rate benefit. New: `history_size = 12,
      history_max_age_s = 0.15`. At 240 Hz that gives 50 ms / 12
      samples (size-binding); at 120 Hz it gives 100 ms / 12 samples
      (slightly longer than the 67 ms 120 Hz optimum, but commit-
      moment error only regresses 5.3 → 5.8 cm across the recording
      set). Briefly tried `age = 0.07` to bind purely by time —
      catastrophic on recordings with sparse / bursty sample
      density (3-sample windows → noisy `v_z` → 65 cm dz bias on
      otherwise-clean throws). Lesson: `history_max_age_s` is a
      safety floor for sample density; `history_size` should remain
      the typical binding cap. Re-sweep on real 240 Hz cart data
      once it's live.

   Also updated `_ReplayTracker` (in the analyzer) to apply the same
   median filter and initialize `last_reject_reason`, and propagated
   `position_cov / last_reject_reason` through the new `TickAnalysis`
   fields so the viser overlay can read them.
- 2026-05-19 — **tuned `BallTrackerConfig.history_size` for fast
volleys**. After the first SRC Kitchen recording session showed
"meaningful predictions only at ~400 ms to impact," swept window
sizes 4–20 across 22 throws (mix of fully-tracked + ones with
mid-flight OptiTrack dropouts). The 167 ms window (size=20) was
introducing window-average lag in the LS velocity estimate — i.e.,
the fit's `v` was biased toward older samples, so forward
propagation aimed slightly off the true intercept. Shrinking to 8
samples (~67 ms at 120 Hz) drops commit-moment mean error
(tti ∈ [0.10, 0.20) s) from ~10 cm → ~7 cm and earlier-flight
buckets stay within 1 cm of the size-20 result. Post-commit buckets
(tti < 0.10 s) are slightly *worse* (size-20 had a 3 cm edge there)
but the FSM has already frozen the plan by then, so those don't
matter. Also added `--segment-min-incoming-speed` to
the analyzer (default `cfg.tracker.min_incoming_speed`) so its
segmentation matches the FSM's gating exactly: a segment is
"incoming" iff signed `v_x < -min_incoming_speed`. Carry-around /
sideways-wave samples are now trimmed away in viser so the throw
trajectory dominates the visualization. `_replay` also resets the
tracker on slow→fast transitions (5+ non-incoming samples followed
by an incoming one) so the LS history doesn't span the carry-prefix.
- 2026-05-19 — **switched the production tracker to volley-only mode**
for SRC Kitchen bring-up. `BallTrackerConfig` defaults changed:
`max_bounces = 0`, `online_bounce_pruning = False`, `history_size = 8`
(originally bumped to 20 in the same session, then dialed back after a
sweep — see below), `history_max_age_s = 0.20`. With `max_bounces = 0` the LS predictor
rejects any propagation that would cross `z = 0` before the strike
plane, so groundstrokes return `None` and the FSM stays in READY —
volley filtering is enforced *at the predictor*, no FSM changes
needed. Longer history window is safe in volley mode (no bounce risk
to straddle). A/B numbers across the existing recording set with
these defaults: ~11 cm mean error at the commit moment (tti ≈ 0.20 s
bucket); ~3.5 cm mean error on the last 150 ms (would be the
operative number if SWING re-issues the strike command during
execution — currently it doesn't, clean follow-up). Bounce mode
re-enabled by setting those four knobs back per the docstring in
`config.py`; (e, μ_t) defaults are already measured so no retuning is
needed when we flip back.
- 2026-05-19 — implemented **Phase 3b** as a testing tool, not an FSM
swap. New `state_machine/ekf_ball_tracker.py` (`EKFBallTracker`);
identical public interface to `BallTracker` so the FSM could swap
without code changes. Added `EKFConfig` nested under
`BallTrackerConfig.ekf`, optional `position_cov` field on `Intercept`
(backwards-compatible — LS leaves it `None`), and analyzer flags
`--tracker {leastsq,ekf}` plus EKF-specific overrides
(`--ekf-process-accel-std-{xy,z}`, `--ekf-measurement-pos-std`,
`--ekf-seed-samples`, `--no-ekf-bounce-handling`). viser draws a
translucent 1σ ellipsoid at the predicted intercept when
`--tracker ekf`. A/B across the full recording set: essentially tied
with the LS+pruning baseline at the swing-commit bucket
(5.0 cm LS → 5.8 cm EKF mean; full table in the **Phase 3b** block
under "What we're debugging now"). FSM is **not** switched — the
unlock the EKF actually buys is a covariance-based commit rule (swing
when σ on the strike plane is tight enough), not point-estimate
accuracy at commit. Tuning gotchas worth a callout: Mahalanobis
outlier gate self-destructs at small `σ_meas` (95% rejection rate
observed, locks filter into seed velocity — defaulted to
`σ_meas = 5 mm`, threshold `= 20`); 4-sample LS seed is too noisy for
the low-`σ_meas` recursive update to correct (bumped to 6); the EKF
posterior doesn't self-expire across idle gaps the way LS history
does (analyzer explicitly resets on gap > 0.30 s).

