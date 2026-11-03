# L2+R2 stop: from `rclcpp::shutdown()` to a latched HOLD / PASSIVE ladder

**Headline: the operator's "press L2+R2 and the robot twitches once and pushes
itself up into a stand, sometimes onto its back" is not a control-loop bug — it
is the *absence* of a control loop. The old handler shut the process down and
left the last `/lowcmd` frame on the wire with nobody to overwrite it. The fix
replaces the exit with two latched states that own `/lowcmd` and keep streaming,
so a stale target can never be executed again.**

Modeled on `wty-yy/unitree_cpp_deploy` (`deploy/include/FSM/State_Passive.h` and
`deploy/robots/go2/config/config.yaml`), which is the only one of the two
candidate references that actually contains deployed robot code (the other,
`JIAlonglong/REAL`, is training-only — 89 files, no C++ / ROS / lowcmd).

## Symptom

Press L2+R2 while standing (or while standing up, or while walking):

1. one sharp motion, then
2. the body gets pushed *upward* toward a stand — often not a clean stance, and
3. sometimes it ends up inverted.

Nothing in the PD path can produce an upward push out of thin air, so the
question was always "what target are the motors still executing?".

## Root cause: the last frame stayed on the wire

At HEAD (`17d7ec2`) the button handler was:

```cpp
// E-stop: both triggers fully pressed.
if (msg->axes[2] < -0.9 && msg->axes[5] < -0.9)
    rclcpp::shutdown();
```

`cmd_msg_` is a persistent member and `/lowcmd` is republished from it. Shutting
the node down stops the publisher but **does not retract the last frame**. The
motor driver then keeps executing the last `(kp, kd, q)` it received until its
own command watchdog fires — and on this arm the watchdog is not a "drop to
zero gains" event, it is a timeout whose behaviour inside the firmware is not
documented. So the robot keeps doing whatever the last frame said.

What the last frame actually contains depends on which of the three writers ran
last:

| Last writer | `kp` | `q` | Consequence of freezing it |
|---|---|---|---|
| `run_policy()` (engaged) | `policy_kp` = 30 | actor target | holds a mid-gait pose — one step finishes, then static |
| `state_transform()` (ramp / hold) | `kp[i]` = 50 | interpolated toward `standing_angels_` | **keeps ramping toward the stand** |
| fallback `else` branch | 50 | `standing_angels_` | **is a stand command** |

The third row is the reported one. `state_machine()`'s final `else` publishes
`state_transform(standing_angels_)` every 20 ms whenever the robot is not
engaged, and `state_transform()` keeps ramping `q_init_ → standing_angels_` at
kp=50 for `rate_count_ 0..100` (~2 s) before it becomes a plain hold. Press
L2+R2 during those two seconds — or simply while standing, since the fallback
re-publishes the stand target continuously — and the frozen frame is *a command
to stand up*. That is the upward push. The inversion follows from freezing
whichever leg was furthest from the target.

So the failure is not "the stop did something wrong". It is "the stop did
nothing at all, and something else was still driving the joints".

## What the reference does

`State_Passive.h` is small enough to quote in full (base64-decoded, 1202 bytes):

```cpp
class State_Passive : public FSMState
{
public:
    State_Passive(int state, std::string state_string = "Passive")
    : FSMState(state, state_string)
    {
        auto motor_mode = param::config["FSM"]["Passive"]["mode"];
        if (motor_mode.IsDefined()) {
            auto values = motor_mode.as<std::vector<int>>();
            for (int i(0); i < values.size(); ++i)
                lowcmd->msg_.motor_cmd()[i].mode() = values[i];
        }
    }

    void enter()          // gains, set once
    {
        static auto kd = param::config["FSM"]["Passive"]["kd"].as<std::vector<float>>();
        for (int i(0); i < kd.size(); ++i) {
            auto &motor = lowcmd->msg_.motor_cmd()[i];
            motor.kp()  = 0;
            motor.kd()  = kd[i];
            motor.dq()  = 0;
            motor.tau() = 0;
        }
    }

    void run()            // called every FSM tick, dt = 0.001 s
    {
        for (int i(0); i < lowcmd->msg_.motor_cmd().size(); ++i)
            lowcmd->msg_.motor_cmd()[i].q() = lowstate->msg_.motor_state()[i].q();
    }
};
```

Three properties matter:

1. **The process never exits.** "Stop" is an FSM state; `CtrlFSM::run_()` keeps
   calling `pre_run/run/post_run` at 1 kHz, so `/lowcmd` never goes stale and the
   motors never fall back to their own watchdog. `config.yaml` has
   `Passive: { mode: [1]*12, kd: [3]*12 }` — `kp` is unset, i.e. 0.
2. `kd` is written **once**, in `enter()`, and `run()` never touches gains. The
   damping is a property of the state, not something ramped or recomputed.
3. `q` is copied from the **live measurement every tick**. With `kp = 0` the `q`
   field is inert; it is rewritten for the sole purpose of keeping any previously
   commanded pose out of the message the motors are being fed.

The rest of the reference's safety shape points the same way: `LT + B` →
`Passive` from every state, `FixSquat.auto_transition: Passive`, and a
`registered_checks` entry calling `isaaclab::mdp::bad_orientation(env, 2.0)` that
forces `Passive` whenever `|acos(-g_z)| > 2.0 rad`. Safety is a *state*, never an
exit path.

## Design: two rungs, each one strictly weaker than the last

The reference answers "how do I stop safely" but not "how do I stop *without it
moving*". On a position-controlled joint those are different questions:

> Torque = `kp * (q_cmd - q_meas) + kd * (0 - dq)`.
> `kp = 0` therefore deletes the holding torque, and a standing robot drops.

Freezing `q_cmd` instead keeps the same torque and keeps the robot where it is —
but only if `q_cmd` is a pose the robot is actually standing in. Freezing a trot
target means freezing two feet in the air, which is a fall. Hence two rungs and
one gate:

| Press | Robot state at the press | State | What the motors get |
|---|---|---|---|
| 1 | settled (`max|dq| < 0.5`, `|roll|,|pitch| < 0.35 rad`) | **HOLD** | the last frame, byte for byte |
| 1 | moving / tilted | **PASSIVE** | `kp → 0`, `kd → 3`, `q → measured` |
| 2 | in HOLD | **PASSIVE** | as above |
| 2 | in PASSIVE | exit | already limp, so cutting the stream is now meaningless |
| 3 | in PASSIVE (after HOLD) | exit | same |

### Rung 1 — HOLD is the literal "do not move"

`hold_tick()` is one line: `publish_lowcmd()`. No field is rewritten, so `mode`,
`kp`, `kd`, `q`, `tau` and the CRC over them are identical to the frame the press
interrupted. From the motor's side nothing happened, which is exactly "keep
still" on a position-servo: the same position error keeps producing the same
holding torque, and the robot stays in the pose it was already holding.

The residual motion is bounded by the frozen frame's distance from the measured
pose — 0.02–0.15 rad in steady standing, and that error *is* the gravity-sup-
porting torque. So HOLD is not "no torque"; it is "no new target".

The `settled` gate is what makes this safe. It needs `/lowstate` (so
`have_lowstate_`), `max|dq| < stop_dq_max`, and `|roll|`, `|pitch| <
stop_tilt_max`. Anything else gets PASSIVE — including the case where L2+R2 is
pressed before the first `/lowstate`.

### Rung 2 — PASSIVE, but faded instead of cut

The reference slams `kp` from 60/80 to 0 inside one 1 ms tick. On a standing
robot that is an instantaneous removal of the holding torque and the body drops
onto its knees. Same end state, but reached continuously:

```
a    = clamp(elapsed / passive_ramp_s, 0, 1)
kp   = kp0 * (1 - a)                      // kp0 = the gain that was live at the press
kd   = kd0 + (passive_kd - kd0) * a
q    = (a >= 1) ? measured : q0           // q0 = the target that was live at the press
dq   = 0, tau = 0, mode = 1
```

Both torque terms fade along `a`, so the first passive frame is torque-identical
to the last active one, and at `a == 1` the message is *exactly*
`State_Passive` (`kp 0`, `kd passive_kd`, `dq`/`tau` 0, `q` measured). The
target is held at `q0` while `kp` is still non-zero — that is what is holding the
robot up during the fade — and handed over to the measurement only once `kp`
reaches 0 and the field is inert.

### The edge latch (why this needed a second pass)

The pad autorepeats `/joy` at 20 Hz while a trigger is held. Handling every
message would walk the ladder — HOLD → PASSIVE → exit — inside 100 ms of one
press. `stop_btn_held_` latches on the rising edge of `axes[2] < -0.9 &&
axes[5] < -0.9` and is cleared only when the triggers are released, so one press
advances exactly one rung.

## Implementation

`src/deploy_rl_policy/src/low_level_ctrl.cpp`, `src/deploy_rl_policy/include/deploy_real/low_level_ctrl.hpp`:

- `enter_stop()` — the settled test and the choice between the two rungs.
- `enter_passive()` — snapshot `stop_q0_/stop_kp0_/stop_kd0_` from `cmd_msg_`
  (so the fade starts where the previous frame ended) and stamp `stop_time_`.
- `hold_tick()` — republish `cmd_msg_` untouched.
- `passive_tick()` — the fade above.
- `state_machine()` — a latched stop is checked **before** the `have_lowstate_`
  gate, publishes `/policy_engaged = false` (that early return would otherwise
  leave the RL stack inferring against a stale `true`), then `hold_tick()` or
  `passive_tick()` and returns. Nothing else ever runs again.
- `keepalive_tick()` — publishes unconditionally in HOLD/PASSIVE, so the 200 Hz
  stream continues even when `keepalive_ms` is otherwise gated on engagement.
- `joy_callback()` — the ladder is handled first, so no other button can dirty
  the FSM flags while a stop is latched; `if (hold_ || passive_) return;` after
  the edge check.
- `state_callback()` — stores `imu_state.rpy` for the settled test.

## Parameters

| Name | Default | Meaning |
|---|---|---|
| `passive_kd` | `3.0` | PASSIVE damping; the reference's `config.yaml` value |
| `passive_ramp_s` | `0.30` | fade `kp → 0` over this long (`0` = reference behaviour, instant) |
| `stop_tilt_max` | `0.35` | rad; `|roll|`,`|pitch|` below this counts as settled (~20°) |
| `stop_dq_max` | `0.5` | rad/s; `max|dq|` below this counts as settled |

## Verification

Sim-side only (`is_simulation:=true`, so it touches `/mujoco/*` and can never
reach a real robot): a probe publishes `/mujoco/lowstate` at 500 Hz (standing
pose, `dq` configurable) and `/joy` at 20 Hz, and inspects `/mujoco/lowcmd`.

Settled robot, two presses 0.15 s long (three autorepeated `/joy` messages each —
which also proves the edge latch):

```
[baseline]      mode/kp/kd/q/tau = (1, 50.0, 1.0, -0.100000001, 0.0)
[after press 1] mode/kp/kd/q/tau = (1, 50.0, 1.0, -0.100000001, 0.0)   <- identical
[after press 2] mode/kp/kd/q/tau = (1,  0.0, 3.0, -0.100000001, 0.0)
still streaming in PASSIVE: 75 frames in 0.3 s
[WARN] L2+R2 — HOLD (settled: max|dq| 0.02 < 0.50, |roll| 0.01 |pitch| -0.01)
[WARN] L2+R2 — PASSIVE: kp 50.0 -> 0 over 0.30 s, kd -> 3.00, q = measured
```

Moving robot (`dq = 2.0`, i.e. mid-gait), one press:

```
[after press 1] mode/kp/kd/q/tau = (1, 0.0, 3.0, -0.100000001, 0.0)
[WARN] L2+R2 — not settled (max|dq| 2.00, |roll| 0.01 |pitch| -0.01) -> PASSIVE:
       freezing a moving target would strand a foot in the air, so the limb lets go instead.
```

`colcon build --packages-select deploy_rl_policy` clean; the new parameter names
are present in `install/.../low_level_ctrl`.

## Boundaries — read before trusting this on hardware

1. **HOLD is not a power cut.** The robot is still standing, still powered, still
   holding with kp = 30/50, just executing an unchanging target. Only physical
   power removal is a true hard stop.
2. **HOLD only engages when the robot was already still and upright.** Press it
   while walking and the robot gets PASSIVE (limb goes limp) by design — freezing
   a trot target is the failure mode this work exists to remove.
3. **The firmware's own command-timeout behaviour is still undocumented.** The
   design sidesteps it (something is always publishing) but does not characterise
   it. `/lowstate`'s `bit_flag` carries no documented bit definitions.
4. **Not yet run on hardware.** Everything above is sim-side plus one build. The
   real-hardware sequence to watch: press L2+R2 while standing (expect a
   completely static robot), then press again (expect a controlled 0.3 s kneel),
   then again (expect exit with `kp = 0, kd = 3` left on the wire).
5. `state_transform()`'s fallback branch still re-publishes a *stand* target every
   20 ms while disengaged. That is now harmless (a stop overrides it) but it is
   still the reason the frozen frame used to be a stand command.

## Operating note

`deploy_mac/run/wireless_to_joy.py` and
`deploy_mac/runbooks/real_run_runbook_ubuntu.md` were updated to describe the
ladder; the runbook's "hard stop" line now reads: *L2+R2 → HOLD (it simply stops
moving) → press again to go limp → again to exit; cutting power is the only true
hard stop.*
