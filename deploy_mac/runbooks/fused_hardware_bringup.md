# Fused stack hardware bring-up — gated dry-run before any parkour attempt

Purpose: prove the fused single-process stack behaves correctly **on the real
robot** before it is trusted with a full parkour run. Every stage has an explicit
pass gate. **Do not advance on a failed gate** — each one catches a failure mode
that is invisible in sim.

Why staged: three things in the fused path have never executed outside sim —
the depth age gate's cross-clock correction (`/rl/depth_clock_offset`), the
per-tick compute budget on real hardware, and the real camera's inter-arrival
jitter feeding the gate. Sim cannot test any of them.

Robot **hoisted / tethered** for stages 0–3. Sport mode **OFF**
(`~/go2/disable_sport.py`) for every stage.

---

## Stage 0 — contract check (no nodes, 10 s)

```bash
python deploy_mac/diag/check_deploy_contract.py --ckpt "$CKPT" \
  --env_class_eq_17 0 --yaw_head 0 --zero_vel_obs 0 \
  --binary_vel 1 --lin_vel_max 1.0
```

**GATE 0:** exits 0. A nonzero exit means the flags contradict the checkpoint —
fix before continuing. Both run scripts also do this automatically.

## Stage 1 — camera + clock, policy idle (60 s)

```bash
ENV_CLASS_EQ_17=0 YAW_HEAD=0 ZERO_VEL_OBS=0 \
CKPT=... EP_ROOT=... SECS=60 ./deploy_mac/run/run_dryrun_remote.sh
```

The dry-run now defaults the pads to the real-run values (`DEPTH_LATENCY_S=0.08`,
`PROPRIO_LATENCY_S=0.016`), enables `track_clock_offset` on the camera node, syncs
the Jetson clock, and writes `latency_actor.tsv`. Before 2026-07-27 it did none of
those — the pads defaulted to 0, no tracker ran, and no TSV was produced, so the
gates below were unverifiable and `platform_metrics.py` had no input. If you are
on an older checkout, pass them by hand.

No `low_level_ctrl`, so **nothing drives /lowcmd**. Engage (L1+R1) but keep the
stick centred — we are testing plumbing, not walking. Then in
`run_logs/dryrun/<ts>/rl_policy.log`:

**GATE 1a — clock domains reconciled.** The status line must show
`clk_ofs=rdy` and `skew=0`.
> `skew>0` means the capture stamp and local receipt disagree by >0.5 s, i.e. the
> Jetson and host clocks are unreconciled. The node falls back to ageing by
> receipt and says so loudly. **This is the failure that silently defeats the
> 80 ms contract if unnoticed** — with offset 0 and a lagging Jetson clock every
> frame reads as older than the budget, so the gate always takes the newest frame
> and there is effectively no delay. Fix with `SYNC_JETSON_CLOCK=1` (default) and
> confirm `depth_camera_node` is publishing `/rl/depth_clock_offset`.

**GATE 1b — the depth budget is actually met.** `depth_age` ≈ 80–180 ms and
**`early=0`**.
> `early>0` means no queued frame had reached the budget, so the enforced delay is
> SHORTER than configured and the policy sees fresher depth than it was distilled
> against. Each occurrence is warned. Causes: camera slower than 10 Hz, or
> frames arriving late enough that the queue never gets ahead.

**GATE 1c — no compute overrun.** Zero `tick overrun` warnings.
> The fused tick pays encoder+actor every 5th tick in one process. Systematic
> overruns mean this host cannot hold 50 Hz with the encoder inline — the
> documented fallback is moving the encoder to a worker thread fed the *same*
> prop snapshot (delivery jitter was measured harmless).

**GATE 1d — proprio pad enforced.** No `proprio pad not met` warnings.

**GATE 1e — latent sane.** `|dlat|` finite, bounded, and *changing* (a frozen
value means the encoder is not being re-fed).

Quantify rather than eyeball, from the `latency_*.tsv` files the run writes into
`run_logs/dryrun/<ts>/` (`LATENCY_LOG=1`, automatic since 2026-07-27):

```bash
python deploy_mac/analysis/analyze_latency_logs.py run_logs/dryrun/<ts>
```

Expect actor ~50 Hz and a chosen-frame age p50 in the 80–180 ms band, with no frame
younger than 80 ms.

## Stage 2 — standing only, motors live (30 s)

Now with `low_level_ctrl`. **Robot still hoisted.**

```bash
ENV_CLASS_EQ_17=0 YAW_HEAD=0 ZERO_VEL_OBS=0 LIN_VEL_MAX=1.0 LATENCY_LOG=1 \
CKPT=... EP_ROOT=... ./deploy_mac/run/run_real_remote.sh
```

Engage (L1+R1), **hold standing ~10 s, never touch the stick.**

**GATE 2a:** the robot holds a stable stand — no oscillation, no drift, no
twitching. **GATE 2b:** all gates 1a–1e still clean under motor load.
**GATE 2c:** release L1+R1 → settles to stand cleanly.

> If the stack is going to fail on timing under real load, it fails here, with
> the robot hoisted and not commanded anywhere.

## Stage 3 — commanded walk, hoisted (30 s)

Same launch. Engage, **hold standing 2 s** (students train with a standstill
prefix — skipping it is off-distribution), then brief forward stick pulses.

**GATE 3a:** gait looks like the sim gait; no divergence or high-frequency
chatter. **GATE 3b:** `τ_peak` stays clear of motor limits (watch
`monitor_deploy.py`). **GATE 3c:** timing gates still clean while walking — this
is the first stage where the camera sees real motion, so it is where depth
inter-arrival jitter actually bites.

## Stage 4 — floor, flat, no obstacles

Lower the robot. Clear floor, no hurdle, no plate. 2 s stand, then walk forward
a few metres and stop.

**GATE 4a:** walks straight without falling. **GATE 4b:** stops on stick
release. **GATE 4c:** no stale-input stops (`STALE /lowstate` or
`STALE /rl/depth_image` in the log) — those cease `/rl/target_pos` and let
`rl_stall_timeout` settle the robot. Seeing one is the safety path working, but it
means an input dropped out and must be explained before adding obstacles.

## Stage 5 — single obstacle, then the circuit

Add **only the hurdle** first (no plate). Then hurdle + plate at the *shallowest*
angle available before anything steeper. Only after that attempt the full circuit.

**GATE 5:** each configuration succeeds twice consecutively before the next is
added.

---

## Abort criteria — stop immediately, any stage

- any `tick overrun` burst, or `skew>0` appearing mid-run
- `early` climbing during motion
- base height dropping unexpectedly, or any asymmetric gait
- stale-input stop that you cannot explain

## Never

- **`lockstep:=true`.** It was inert until 2026-07-27; now that it is genuinely
  wired the measured result is embedded **1/10** — true lockstep is far worse than
  free-run. Not for deploy-predictive runs, and not on hardware.
- Skipping the 2 s standstill before the first forward command.
- Stage 5 before stages 1–4 are clean.

## Fallback

`FUSED=0` restores the legacy two-node split on both run scripts, unchanged. If
the fused arm fails a timing gate that the two-node arm passes, that is a
genuinely useful result — capture the run dir and the `platform_metrics.py`
output before switching.
