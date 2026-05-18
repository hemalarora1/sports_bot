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

Today's setup (2026-05-17): your laptop is on Stanford wifi (not SRC), Motive Kitchen
PC is in **unicast** mode. Run from the laptop:

```bash
cd "$(git rev-parse --show-toplevel)/sports_bot/optitrack"
conda activate opensai
PYTHONPATH=drivers/PythonClient python -u StreamDataSkeleton.py \
    172.24.69.102        \
    <your-laptop-IP>     \
    u                    # 'u' = unicast, 'm' = multicast
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

| Item | Value | Notes |
|---|---|---|
| Kitchen Motive server | `172.24.69.102` | from src_mocap/README. Other bays have different IPs. |
| NatNet command port   | `1510` | UDP, bidirectional. Survives cross-subnet routing. |
| NatNet data port      | `1511` | UDP. In multicast, listened to on `239.255.42.99:1511`. |
| NatNet multicast group| `239.255.42.99` | Default. Multicast does NOT route across subnets. |
| VRPN port             | `3883` | A *different* protocol. Disabled in Motive currently. The UI puts it next to NatNet which causes confusion. |

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

## Motive configuration (Kitchen, as of 2026-05-17)

- Streaming → NatNet — **enabled**, Transmission Type **Unicast** (was Multicast
  before today's session).
- Streaming → VRPN — disabled. Its "Broadcast Port: 3883" is a red herring; it
  belongs to VRPN, not NatNet.
- Assets pane — `PickleBall` rigid body, **Streaming ID = 8**. (Default Streaming
  ID is the asset's row position in the pane. Can be overridden in the asset's
  Properties → User Data field.)
- KVM access for the Kitchen Motive PC: `SRC-KVM-Kitchen.stanford.edu`. Credentials
  from Zen.

**Heads up:** if anyone changes Transmission Type back to Multicast and you're not
on SRC, streaming will go silent without errors — the command channel still works,
so your script will *look* connected.

---

## Redis key schema

Published by `StreamDataSkeleton.py` (per rigid body per frame, ~120 Hz):

| Key | Format | Frame |
|---|---|---|
| `sai2::optitrack::rigid_body_pos::<id>`      | JSON `[x, y, z]`        | **World** (calibrated) |
| `sai2::optitrack::rigid_body_ori::<id>`      | JSON `[qx, qy, qz, qw]` | World quat |
| `sai2::optitrack::raw::rigid_body_pos::<id>` | JSON `[x, y, z]`        | Motive room frame |
| `sai2::optitrack::raw::rigid_body_ori::<id>` | JSON `[qx, qy, qz, qw]` | Room quat |

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

| Key | Owner | Purpose |
|---|---|---|
| `opensai::controllers::Panda::cartesian_controller::cartesian_task::goal_position` | FSM → OpenSai | Racket goal pos (opensai backend) |
| `opensai::controllers::Panda::cartesian_controller::cartesian_task::goal_orientation` | FSM → OpenSai | Racket goal rot |
| `sports_bot::cmd::base::goal_pose` | FSM → controller | `[x, y, theta]` base goal (cs225a backend) |
| `hb1::desired_pose` | TidyBot driver | `[x, y, theta]` the wheel driver actually reads |
| `hb1::current_pose` / `hb1::current_vel` | TidyBot driver → us | Base feedback |
| `hb1::kill` / `hb1::stop` | us → TidyBot driver | "kill" terminates driver; "stop" decelerates |

---

## Gotchas hit so far

- **Two copies of `StreamDataSkeleton.py`** in the repo. The one in
  `sports_bot/optitrack/StreamDataSkeleton.py` is the maintained version (has
  world calibration + rigid_body_listener already enabled). The one in
  `sports_bot/optitrack/drivers/PythonClient/StreamDataSkeleton.py` is the
  vanilla NatNet SDK sample — older, has `rigid_body_listener` commented out.
  Use the top-level one. It needs `PYTHONPATH=drivers/PythonClient` because the
  NatNet SDK modules live there.
- **`python -u` matters** if you want to see prints in tail logs / background
  runs; without it Python buffers stdout aggressively when not on a TTY.
- **`request_data_descriptions`** crashes the NatNet SDK on a UTF-8 decode of
  marker names (NatNetClient.py:965). Doesn't affect frame streaming. Live with
  it for now.
- **`set_use_multicast(False)` is not enough** if Motive itself is in Multicast
  mode — the client requests unicast but Motive just doesn't send it. Both sides
  have to agree.

---

## Open items for integration

- [x] ~~`world_calibration.json` for whichever bay we end up using~~ — done for
      SRC Kitchen 2026-05-17. Re-do per bay (and after any Motive ground-plane
      or camera recalibration).
- [ ] **Daily calibration sanity check.** Place a marker at the origin floor
      mark, read `sai2::optitrack::raw::rigid_body_pos::<id>` and run it through
      `R, t`; expect world-frame `(0, 0, 0)` within ~5 mm. If it drifts,
      re-solve. (Better: group the 3 floor markers into a single `FloorRef`
      rigid body and read its world pose each session.)
- [ ] **`_opti_to_world_quat()` is a passthrough.** Position is calibrated;
      orientation isn't. Fine for the ball (sphere). Fix before relying on
      OptiTrack for the cart's heading.
- [ ] Make the FSM write to `hb1::desired_pose` instead of
      `sports_bot::cmd::base::goal_pose` (or add a one-line bridge).
- [ ] Decide PickleBall's permanent Streaming ID — easier to standardize on `1`
      in Motive than to keep passing `--optitrack-rigid-body-id 8` everywhere.
- [ ] Register laptop MAC addresses with Zen → static SRC IPs → switch Motive
      back to multicast for the demo (lower bandwidth, supports many viewers).
- [ ] Confirm whether the cart's mini-PC is on SRC and could run the streamer
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
