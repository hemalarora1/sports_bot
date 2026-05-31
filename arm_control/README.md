# Arm control bring-up notes

This folder holds hardware bring-up scripts for the OpenSai Franka Cartesian
controller. These notes are from the 2026-05-28 pickleball TCP tests.

## Current best hardware test state

Use `config_folder/xml_config_files/arm_test_picklebot.xml` with:

```xml
<compliantFrame xyz="0 0 0.35" rpy="0 0 0" />
<positionGains kp="200.0" kv="20.0" ki="0.0"/>
<orientationGains kp="100.0" kv="20.0" ki="0.0"/>
<gains kp="15.0" kv="14.0" ki="0.0" />  <!-- cartesian_controller joint_task -->
```

`step0_controller_test.py` currently uses:

```python
READY_POS = np.array([0.55, 0.00, 0.50])
```

Before running hardware tests, relaunch OpenSai after XML edits. OpenSai loads
`compliantFrame` and gains at controller startup.

## Safe Step 0 commands

```bash
python sports_bot/arm_control/step0_controller_test.py --sweep --pre-ready --pos-tol-mm 10
python sports_bot/arm_control/step0_controller_test.py --delta-mm -10 0 0 --pre-ready --pos-tol-mm 10
```

`step0_controller_test.py` now ensures `cartesian_controller` is active and
refreshes paired Cartesian goals at 100 Hz, matching the ZitiBot Redis pattern.

## What we observed

- With the softened null-space posture task (`kp=15 kv=14`), the arm can perform
  most 2 cm micro-sweep moves around ready in under 1 s with joint speeds well
  below limits.
- The remaining weak direction is `x-minus`: it visibly moves backward, then
  plateaus with a steady residual instead of converging all the way.
- The residual grows with command size: a -10 mm request plateaued around 11.6 mm
  position error, and a -30 mm request plateaued around 25.7 mm position error.
- The plateau had low joint velocities, so this looks like a task/posture
  equilibrium rather than actuator saturation.
- Offset sweeps were directionally useful but not perfectly clean because the
  helper edits the XML in place and OpenSai must be manually relaunched each
  time. Treat those logs as comparative hints, not final evidence.

## Known unsafe / bad tests

Do not run `--position-only` on hardware with the long pickleball TCP unless you
have a very specific safety reason and use `--force-position-only`. With
`compliantFrame xyz="0 0 0.35"`, disabling orientation caused a hard jolt and a
Franka joint velocity violation.

Also avoid trying to diagnose the x-minus plateau by simply softening or zeroing
orientation gains on the real arm. Lower orientation gains (`kp=80 kv=15`) also
triggered a joint velocity violation in this setup. Restore normal orientation
control before continuing.

## Working hypothesis

The arm is dynamically promising for pickleball: most small Cartesian moves are
fast and low effort. The issue is local kinematic/task conditioning around the
current ready pose, especially for negative arm-frame X motion with a long TCP
and posture task active. Next useful diagnostics should be cautious and should
prefer posture/ready-pose exploration over underconstraining orientation.

---

## Step J2 results (2026-05-28, `arm_test_picklebot_no_tcp_wrist_stiff.xml`)

**Script:** `stepj2_named_pose_sequence.py --preset micro-strike --scale 1.0`

### q_start bug — found and fixed

The original script passed `q_prev` (the *planned* prior target) as the
interpolation start for each segment. If the arm hadn't fully converged to the
planned position, the next segment's setpoint jumped discontinuously at t=0.
Fixed in `stepj2_named_pose_sequence.py`: now reads actual `SENSOR_JOINTS` at
the start of each segment and uses that as `q_start`.

### q1 tracking failure

After the fix, `prep` (moves q3+q7 only) converges cleanly (≤0.4 deg error).
But `strike`, `follow`, and `return_home` all show ~2 deg residual error on q1:

| Segment | q1 target delta | q1 actually moved | tracking |
|---------|----------------|-------------------|---------|
| strike  | +2.7 deg       | +0.56 deg         | 21%     |
| follow  | +4.1 deg       | +1.61 deg         | 39%     |
| return  | −2.5 deg       | −0.52 deg         | 21%     |

Peak velocity on all joints: ~4 deg/s (3% of the 150 deg/s limit). Torques are
well under limits; safety torques stayed at zero throughout.

**Root cause:** same pattern as q5/q6/q7 before the wrist-stiff fix — `kp=200`
with no explicit `velocityLimit` produces only ~0.7 Nm on q1 for a 2 deg error
(expected ~7 Nm). Likely the default velocity saturation cap is extremely
conservative for q1.

**Fix applied to XML (2026-05-28 run 1):** `kp` q1: 200 → 400, `kv` q1: 20 → 28. Also added
explicit `velocityLimit="0.6 0.7 0.8 0.9 1.0 1.1 1.2"` rad/s.

**Result after relaunch (run 2):** q1 improved to 59-61% tracking — still undertracking.
Also found q3 undertracking: 82% in prep, 61% in strike/follow (accumulated drift amplifies
effective stroke). Both q1 and q3 show same symptom: tau_cmd_peak ~0.8-0.9 Nm regardless
of error magnitude, consistent with velocity saturation capping at ~kv * vLimit.

**Fix applied (run 2):** q1 and q3 both escalated to kp=800, kv=40, matching wrist joints.

### J2 complete (2026-05-28 run 3 — kp=800/kv=40 on q1 and q3)

After boosting both q1 and q3 to kp=800/kv=40:

| Segment    | err_deg |
|------------|---------|
| prep       | 0.189   |
| strike     | 0.593   |
| follow     | 0.506   |
| return_home| 0.715   |

q3 now at 91-97% tracking. q1 at 80-84% (still the lagging joint — higher
rotational inertia than wrist). At 0.5 m reach, 0.7 deg → <6 mm Cartesian
error, sufficient for pickleball. **J2 complete.**

### Next step: J3

`stepj3_ik_roundtrip.py` — validates ikpy → joint_controller pipeline.
Does NOT use cartesian_controller for motion.

1. Calibrate FK offset: read SENSOR_JOINTS + OpenSai `current_position` simultaneously
   (arm stationary) → measures link7-to-end-effector offset.
2. For each target: solve IK (position-only), safety-check joint deltas, execute
   via joint_controller, measure Cartesian residual via ikpy FK + offset.
3. Pass: residual < 8 mm.

```bash
# Start here (near_home → strike_mid → strike_high):
python sports_bot/arm_control/stepj3_ik_roundtrip.py

# Single pose with pauses:
python sports_bot/arm_control/stepj3_ik_roundtrip.py --pose near_home --pause

# All four poses:
python sports_bot/arm_control/stepj3_ik_roundtrip.py --pose all
```

---

## J6 bring-up complete (2026-05-31)

**Script:** `stepj6_reactive_intercept.py` — reactive IK-based ball intercept, joint controller only.

### Confirmed working XML: `picklebot.xml`

```xml
<!-- joint_controller jointTask -->
<gains kp="200 200 200 200 400 600 400"
       kv=" 20  20  20  20  28  28  28"
       ki="  0   0   0   0   0   0   0" />
<velocitySaturation enabled="true"
    velocityLimit="1.2 1.4 1.6 1.8 1.0 1.1 1.2" />
<otg type="disabled"/>
```

`picklebot_clean.xml` (kp=100 uniform) cannot execute the strike swing — insufficient torque to pull q6 through the 10° wrist arc. Do not use it for J6.

### Three code fixes applied

1. **Commit keep-tracking**: when commit fires but arm is still en route to wind-up (`commit_delta_deg > max` but `tti > 0`), the loop keeps tracking toward q_wu instead of oscillating home.
2. **strike→home actual-position start**: `run_segment("strike→home", ...)` now uses `get_vec(SENSOR_JOINTS)` as start — avoids a setpoint snap caused by wu→strike tracking error.
3. **Active pre-swing settle**: before wu→strike, the loop holds `last_wu_q` (IK wind-up goal) for 200 ms at 200 Hz. This lets all joints converge — especially q6 which is the slowest — before the swing fires. Passive settle (hold current joints) left q6 at gravitational equilibrium, 9.5° from target.

### Verified performance (mock-identity-base, z=0.45)

| Metric | Value |
|---|---|
| Home move max error | ≤ 2.2° |
| Tracking settled residual | 1.1°@q5 |
| wu→strike qdot_max | 12–20°/s (16% of hardware limit) |
| wu→strike goal_err | ~2.8°@q4, ~1.9°@q6 |
| Live cart T_W_A (OptiTrack) | SETTLED 1.1°@q5 |

### Standard bring-up command (mock, with commit)

```bash
python sports_bot/arm_control/stepj6_reactive_intercept.py \
  --skip-cal --offset-link7 0 0 0.49 \
  --mock-intercept 0.45 0.0 0.45 \
  --mock-tti 0.25 \
  --mock-identity-base \
  --gentle-swing --mini-swing \
  --wind-up-offset 0.02 \
  --fixed-arm-z-m 0.45 \
  --tracking-step-deg 0.75 \
  --tracking-accel-deg-s2 250 \
  --verbose-tracking \
  --log-period-s 0.25
```

### Live ball intercept (next milestone)

```bash
# Step 1: live cart, mock ball, no commit
python sports_bot/arm_control/stepj6_reactive_intercept.py \
  --skip-cal --offset-link7 0 0 0.49 \
  --no-commit \
  --mock-intercept 0.45 0.0 0.45 --mock-tti 1.0 \
  --wind-up-offset 0.02 --fixed-arm-z-m 0.45 \
  --tracking-step-deg 0.75 --tracking-accel-deg-s2 250 \
  --verbose-tracking --log-period-s 0.25
# Push cart by hand — wu_A should stay constant in world frame.

# Step 2: live cart, live ball, no commit (ball ID=1)
python sports_bot/arm_control/stepj6_reactive_intercept.py \
  --skip-cal --offset-link7 0 0 0.49 --no-commit \
  --ball-rigid-body-id 1 --strike-plane-x 0.45 \
  --wind-up-offset 0.02 --tracking-step-deg 0.75 \
  --tracking-accel-deg-s2 250 --verbose-tracking --log-period-s 0.10

# Step 3: live ball with commit
python sports_bot/arm_control/stepj6_reactive_intercept.py \
  --skip-cal --offset-link7 0 0 0.49 \
  --ball-rigid-body-id 1 --strike-plane-x 0.45 \
  --wind-up-offset 0.02 --gentle-swing --mini-swing \
  --tracking-step-deg 0.75 --tracking-accel-deg-s2 250 \
  --verbose-tracking --log-period-s 0.10
```

---

---

## J7 — Production strike planner (2026-05-31)

**Script:** `stepj7_strike_planner.py` — replaces J6 for real ball intercept. Reads ball from Redis, LS-predicts intercept, locks first safe prediction, tracks arm to wind-up, commits swing.

**Full context and commands:** See `CLAUDE.md` at repo root and `sports_bot/context.md`.

### Confirmed working state (replay tests 2026-05-31)

- XML: **`picklebot.xml`** (standard swing) / **`picklebot_j7.xml`** (wrist flick — q6 vel raised to 2.5 rad/s)
- Ball rigid body ID: **31**
- Replay test outcome: `target_delta` 26° → 10° with new Q_HOME_RAD; both 2A (replay23) and 2B (replay10) outcome=OK

### Known issues

| Issue | Symptom | Status |
|---|---|---|
| Standard swing is soft | qdot_max=35-44°/s, paddle ~0.09 m/s at contact | By design — forward push = tiny angular sweep |
| Q6 barely moves in standard swing | User observation: "no strong q6 swing" | Correct — q6 locked for orientation; use flick mode |
| wu_hold_s=0 fails 5° goal_err | `WARN goal_err=5.72° > 5.00°` → no follow-through | Keep wu_hold_s=0.10 |

### Wrist flick mode

```bash
# Load picklebot_j7.xml, then add to J7 command:
--flick-q6-deg 30 --flick-s 0.25 --flick-vel-frac 1.0 --flick-wu-delta-deg 8
```
- 30° q6 rotation at 143°/s → paddle tip ~1 m/s (vs ~0.09 m/s from forward swing)
- NOT yet tested on hardware (2026-05-31)
- Requires `picklebot_j7.xml` — without it, q6 vel limit 63°/s stretches flick to 0.71s

---

## Safe Step J1/J2 commands

```bash
# Validate joint hold (J0)
python sports_bot/arm_control/stepj0_joint_hold.py

# Small nudges (J1)
python sports_bot/arm_control/stepj1_joint_nudge.py --joint 1 --delta-deg 2 --pause

# Named pose sequence (J2)
python sports_bot/arm_control/stepj2_named_pose_sequence.py --preset micro-strike --scale 1.0
python sports_bot/arm_control/stepj2_named_pose_sequence.py --preset micro-strike --scale 1.0 --pause-between
python sports_bot/arm_control/stepj2_named_pose_sequence.py --preset wrist-only --pause-between
```
