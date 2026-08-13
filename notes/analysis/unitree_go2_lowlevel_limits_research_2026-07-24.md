# Unitree Go2 low-level control: torque limits, command rate, foot force sensors

Web research, 2026-07-24. Question context: our deploy sends position targets (kp=35, kd=0.6,
tau_ff=0) over DDS `/lowcmd` from a Mac; we observed motor sag after `/lowcmd` stalls of a few
hundred ms, and our MuJoCo/URDF model uses effort limits 23.7 N·m (hip/thigh) and 35.55 N·m (calf).

Legend: **[OFFICIAL]** = Unitree docs/repos/spec pages. **[COMMUNITY]** = forums, third-party repos,
distributor docs. Doc-center pages (support.unitree.com) are a JS SPA; content was retrieved via the
Jina reader proxy (`r.jina.ai/<url>`) — quotes are close paraphrases of the rendered page.

---

## Q1. Torque limits: firmware enforcement + official per-joint values

### Does firmware enforce joint torque limits in low-level mode?

**Not documented.** Nothing found in the official Go2 developer docs, unitree_sdk2, or
unitree_ros2 states that the firmware clamps the commanded PD torque (kp·Δq + kd·Δdq + tau) to the
joint limits, nor what happens if the resulting torque exceeds motor capability.

Circumstantial evidence that protection is (or historically was) the *user's* responsibility:

- **[OFFICIAL, Go1-era]** `unitree_legged_sdk` ships a client-side `Safety` class:
  `int PowerProtect(LowCmd&, LowState&, int)` — "input factor 1–10 = 10%–100% power limit,
  beginners should use factor 1" — and `PositionProtect(..., limit = 0.087)` (~5°). These run in
  the *user's* control loop, not on the robot.
  Source: https://github.com/unitreerobotics/unitree_legged_sdk/blob/go1/include/unitree_legged_sdk/safety.h
- **[OFFICIAL]** `unitree_sdk2` (Go2) has **no** equivalent safety/PowerProtect class; the
  examples only compute a CRC before publishing. The Go2 Basic Services doc describes the
  `MotorCmd_` fields (`q` "joint target position", `tau` "joint target torque", `kp` "joint
  stiffness coefficient", `kd` "joint damping coefficient") **without any documented ranges or
  limits**. Source: https://support.unitree.com/home/en/developer/Basic_services
- **[COMMUNITY]** Third-party deploy wrappers (e.g. INRIA's
  https://github.com/inria-paris-robotics-lab/unitree_control_interface, "wrapper ... with
  safeties and init procedure"; its watchdog node "enforces some safeties on the robot")
  exist precisely because the vendor stack provides none client-side.

Working assumption for the thesis: **treat the firmware as executing any PD command up to actuator
hardware capability**; do all clamping (torque, position, velocity, kp/kd sanity) on our side.
Whether an internal current/thermal limiter exists in the motor drivers is unknown (it almost
certainly does at the electrical level, but its thresholds/behavior are not published).

### Official per-joint torque values

- **[OFFICIAL]** Product spec (https://www.unitree.com/go2): "maximum joint torque about
  45 N·m", with the footnote "the maximum torque in the table refers to the maximum torque of the
  largest joint motor; the actual maximum torque varies for the 12 joint motors."
  Same table on the developer "About Go2" page
  (https://support.unitree.com/home/en/developer/about_Go2): PRO/X/EDU list "About 45N.m";
  **AIR lists "/" (not specified)**. Per-joint breakdown (hip vs thigh vs calf) is **not** in any
  official spec table.
- **[OFFICIAL]** Go2 URDF, `unitree_ros/robots/go2_description/urdf/go2_description.urdf`
  (https://github.com/unitreerobotics/unitree_ros):
  - hip: `effort="23.7" velocity="30.1"`
  - thigh: `effort="23.7" velocity="30.1"`
  - calf: `effort="45.43" velocity="15.70"`
- **[OFFICIAL]** Unitree's own MuJoCo model (`unitree_mujoco`, `unitree_robots/go2/go2.xml`):
  hip/thigh actuators `ctrlrange="-23.7 23.7"`, knee actuators `ctrlrange="-45.43 45.43"`.
  Source: https://github.com/unitreerobotics/unitree_mujoco

### Comparison with our model (23.7 / 35.55 N·m)

| Joint | Our MuJoCo/URDF | Official Go2 URDF + unitree_mujoco | Delta |
|---|---|---|---|
| hip | 23.7 N·m | 23.7 N·m | match |
| thigh | 23.7 N·m | 23.7 N·m | match |
| calf | **35.55 N·m** | **45.43 N·m** | ours is ~22 % low |

35.55 N·m is the **Go1** calf value (Go1-era SDK/URDF); the Go2 calf (knee) joint is rated
45.43 N·m in every official Go2 model file, consistent with the "about 45 N·m largest joint"
marketing spec. Consequence: our sim under-states available knee torque → conservative for sim2real
(real robot can produce more knee torque than the trained policy ever saw), but it also means the
real PD loop can transiently exceed what the sim actuator would have saturated at.

**Go2 vs Go2 EDU:** the developer spec table lists identical joint hardware across PRO/X/EDU
(AIR unspecified); differences are payload, speed, compute, sensors, and SDK access — not motor
torque. **[COMMUNITY]** Distributor doc agrees ("actual torque varies by motor"; only EDU has foot
force sensors and full secondary development): https://www.docs.quadruped.de/projects/go2/html/Overview_1.html
**[COMMUNITY]** go2_firmware_tools notes "Motor firmware was introduced in firmware version 1.0.24,
presumably to support higher torques required for AI mode" — i.e. torque envelopes have shifted
with firmware versions and variant gating is largely software:
https://github.com/legion1581/go2_firmware_tools

---

## Q2. Command rate and watchdog/timeout behavior

### Documented command rate

- **[OFFICIAL]** Developer FAQ (https://support.unitree.com/home/en/developer/FAQ, via Jina
  render): "Go2 SDK currently has **500 Hz** communication frequency for low level, and 500 Hz for
  the high level service interface." This is the only place a rate is stated as a property of the
  interface.
- **[OFFICIAL]** `go2_stand_example.cpp`
  (https://github.com/unitreerobotics/unitree_sdk2/blob/main/example/go2/go2_stand_example.cpp):
  `CreateRecurrentThreadEx("writebasiccmd", UT_CPU_ID_NONE, 2000, &Custom::LowCmdWrite, this)` —
  **2000 µs interval = 500 Hz**, with `float dt = 0.002; // 0.001~0.01` — i.e. the example's own
  comment blesses a dt range of **1–10 ms (100–1000 Hz)**. Example gains: Kp=60, Kd=5.
- **[COMMUNITY/derived]** unitree_ros2's Go2 low-level example likewise uses a 2 ms timer
  (500 Hz): https://deepwiki.com/unitreerobotics/unitree_ros2/9.1-go2-examples

So: 500 Hz is the reference rate; the only officially hinted tolerance is the `0.001~0.01` comment.
Our ~50 Hz policy → the deploy stack's 500 Hz `/lowcmd` re-publisher is doing the work the vendor
examples assume.

### Watchdog / command-timeout behavior

**Not documented as a behavior.** Findings:

- **[OFFICIAL]** The Go2 LowState `bit_flag` field is documented to contain a "**motor timeout
  flag** (1 = timeout, 0 = normal)" and an "**operation control command timeout flag**"
  (https://support.unitree.com/home/en/developer/Basic_services). So the firmware demonstrably
  *detects* command timeouts and reports them — but the docs do **not** state the timeout
  duration, what action is taken (damping? torque-off? hold last command?), or whether it is
  configurable. No configuration interface for it appears anywhere in unitree_sdk2.
- **[OFFICIAL]** LowState `tick` is documented as a 1 ms counter ("1ms 计时用, 按照 1ms 递增") —
  useful for measuring stalls, but not a watchdog spec.
- A GitHub issue search on unitreerobotics/unitree_sdk2 for watchdog/timeout/damping returned
  no matching issues (https://github.com/unitreerobotics/unitree_sdk2/issues).
- **[COMMUNITY]** Third parties build their own watchdogs on top (INRIA wrapper above;
  walk-these-ways-go2 relies on the remote's L2+B damping mode as the safety fallback:
  https://github.com/Teddy-Liao/walk-these-ways-go2).

Interpretation for our observed sag after ~hundreds of ms of `/lowcmd` stall: consistent with an
**undocumented motor-side timeout** (the same mechanism that sets the "motor timeout" bit) dropping
the motors out of active PD (sag), followed by PD snap-back when commands resume — matching our
2026-06-19 forensics. The threshold (~O(100 ms)) is **not published anywhere found**; it must be
characterized on hardware (suspend robot, stall the publisher for controlled durations, log
`bit_flag`, `tick`, and `tau_est`).

---

## Q3. LowState `foot_force` / `foot_force_est`

### Official statements

- **[OFFICIAL]** Go2 Basic Services doc (https://support.unitree.com/home/en/developer/Basic_services):
  - `foot_force[4]`: "Foot force (**range 0–4095**), displayed according to actual values.
    Array order: **0-FR, 1-FL, 2-RR, 3-RL**."
  - `foot_force_est[4]`: reserved — rendered translation: "not currently used."
  - **No units, no calibration, no sensor type given.** 0–4095 is a 12-bit raw ADC-style range,
    i.e. **uncalibrated counts, not Newtons**.
- **[OFFICIAL]** Spec tables (unitree.com/go2, developer About_Go2): the **foot-end force sensor
  is EDU-only**; AIR and PRO have none. On non-EDU units the field's content is undefined by any
  doc found.
- **[OFFICIAL]** Go2 foot-end accessory page (https://www.unitree.com/go2/foot/): "Highly
  Sensitive Foot End … Quickly obtain the foot end data, data support for the precise control."
  45×39.5×48 mm, 56 g, quick-replace. No units/principle stated on the page itself. (Search-result
  snippets describe it as an **air-pressure** sensor: compressed air in the foot pad transmits
  pressure through a tube to a sensor at the shoulder — matching the Go1-era design below; treat
  the air-tube detail as well-supported but not quoted verbatim from a live official page.)
- **[OFFICIAL, Go1-era, strong corroboration]** `unitree_legged_sdk` `comm.h`
  (https://github.com/unitreerobotics/unitree_legged_sdk/blob/go1/include/unitree_legged_sdk/comm.h):
  `footForce; // Data from foot airbag sensor` and `footForceEst; // reserve, typically zero`.
  So already on Go1 the "force" is an **airbag (air-pressure) sensor reading**, and the `_est`
  field was a reserved stub.
- **[OFFICIAL]** Developer FAQ: "The data of the foot sensor … has the characteristic of
  **drift**, so you need to consider the error situation" — official confirmation that the values
  drift and need per-session screening/re-zeroing.

### Community folklore to be skeptical of

- "foot_force is in Newtons" — **unsupported**. No official source gives units; the documented
  0–4095 range says raw counts. Any Newton interpretation requires user calibration
  (known masses / force plate).
- "foot_force_est is a filtered/estimated version computed from motor torques" — plausible naming
  guess, **contradicted** by both the Go1 header ("reserve, typically zero") and the Go2 doc
  ("not currently used"). Do not build logic on `foot_force_est`.

### Consequence for our contact-threshold pipeline

Our real-robot `contact_threshold = 20` operates on raw drift-prone 0–4095 counts from an
air-pressure pad (EDU), while sim uses 2.0 N net contact force — these are **different physical
quantities related only monotonically**; the FAQ's drift warning supports adding a per-boot
standing-baseline calibration rather than a fixed threshold.

---

## Summary table

| Question | Answer | Confidence / source class |
|---|---|---|
| Firmware torque-limit enforcement in low-level mode | Not documented; assume none at command level (protect user-side) | OFFICIAL docs silent; Go1 SDK put protection client-side |
| Per-joint torque, official model files | hip/thigh 23.7 N·m, calf 45.43 N·m, vel 30.1 / 15.7 rad/s | OFFICIAL (go2 URDF + unitree_mujoco) |
| Marketing peak torque | "about 45 N·m" largest joint (PRO/X/EDU; AIR unlisted) | OFFICIAL (unitree.com, About_Go2) |
| Our model vs official | hip/thigh match; our calf 35.55 is the Go1 value, official Go2 is 45.43 | OFFICIAL |
| Low-level command rate | 500 Hz interface; example dt 2 ms, comment allows 1–10 ms | OFFICIAL (FAQ + go2_stand_example.cpp) |
| Watchdog on /lowcmd silence | Exists (bit_flag "motor timeout" + "control command timeout" flags) but action, threshold, configurability undocumented | OFFICIAL flags; behavior UNDOCUMENTED |
| foot_force | Raw 0–4095 counts, FR/FL/RR/RL, air-pressure ("airbag") pad, EDU-only, drifts | OFFICIAL |
| foot_force_est | Reserved / "not currently used" | OFFICIAL |
| foot_force units in Newtons | Folklore, no source | COMMUNITY (rejected) |

## Unresolved — needs vendor confirmation or hardware measurement

1. **Watchdog threshold and action**: how many ms of `/lowcmd` silence until the "motor timeout"
   flag sets, and what the motors do then (damping vs free vs hold). Measure on hardware
   (suspended robot, controlled publisher stalls, log `bit_flag`/`tick`/`tau_est`); we currently
   only have the ~732 ms-stall sag observation from 2026-06-19.
2. **Firmware torque clamping**: whether commanded PD torque is saturated at 23.7/45.43 N·m, at a
   motor-driver current limit, or not at all. Testable with a locked joint + ramped kp·Δq while
   logging `tau_est`.
3. **Per-joint torque split from Unitree directly**: 23.7/45.43 comes from model files, not a
   signed spec sheet; confirm with Unitree support whether these are peak or continuous ratings.
4. **foot_force scaling**: counts→Newton mapping and temperature/drift model — needs bench
   calibration (known masses per foot); Unitree publishes nothing.
5. **Non-EDU behavior of foot_force**: what AIR/PRO report in that field (zeros? estimates?) —
   irrelevant for our EDU but undocumented.
6. **Whether the low-level watchdog is configurable** (service/API to change timeout): no
   interface found in unitree_sdk2 or docs; would need a vendor answer.

## Source list

- https://support.unitree.com/home/en/developer/Basic_services (LowCmd/LowState fields, foot_force 0–4095, bit_flag timeout flags)
- https://support.unitree.com/home/en/developer/FAQ (500 Hz low-level; foot sensor drift)
- https://support.unitree.com/home/en/developer/about_Go2 (variant spec table, 45 N·m note)
- https://support.unitree.com/home/en/developer/Quick_start (disable mcf/sport_mode before low-level; suspend legs warning)
- https://www.unitree.com/go2 and https://www.unitree.com/go2/foot/ (spec table; foot-end accessory)
- https://github.com/unitreerobotics/unitree_sdk2/blob/main/example/go2/go2_stand_example.cpp (2000 µs loop, dt comment, Kp 60/Kd 5)
- https://github.com/unitreerobotics/unitree_ros (go2_description URDF effort/velocity limits)
- https://github.com/unitreerobotics/unitree_mujoco (go2.xml ctrlranges)
- https://github.com/unitreerobotics/unitree_ros2 (LowState.msg: int16[4] foot_force / foot_force_est, no comments)
- https://github.com/unitreerobotics/unitree_legged_sdk (go1 comm.h "foot airbag sensor"; safety.h PowerProtect)
- https://deepwiki.com/unitreerobotics/unitree_ros2/9.1-go2-examples (ROS2 example 2 ms/500 Hz)
- https://www.docs.quadruped.de/projects/go2/html/Overview_1.html (distributor variant table)
- https://github.com/legion1581/go2_firmware_tools (variant gating in software; motor firmware since 1.0.24)
- https://github.com/inria-paris-robotics-lab/unitree_control_interface (third-party watchdog/safety wrapper)
- https://forum.mybotshop.de/t/unitree-go2-low-level-control/950 (distributor: "Go2 Pro doesn't support any type of programming. Only edu/edu+")
