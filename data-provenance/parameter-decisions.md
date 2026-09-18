# Parameter decisions log

Every choice that is not derived from the connectome, with when/why. Newest last.

## 2026-09-12 — edge signs
See `edge-signs.md`.

## 2026-09-13 — M2 sweep criterion was inadequate; M3 operates below ignition

**What M2 found** (`docs/m2-report.md` §4): the network is bistable. Above g ≈ 0.55–0.58 a
first-order transition puts 12–30 % of neurons into a saturated self-sustained state
(150–213 Hz, v down to −4,000 mV with current-based synapses). Below it the network is
silent at rest and input-driven. The M2 criterion ("EXTINCT = 0 spikes in the last 200 ms
after stimulus offset") only measures self-sustained activity, so it labelled the
input-driven regime EXTINCT and the ignited regime VALID. `DEFAULT_G = 0.886` therefore
points at the ignited state. The criterion was written by the planning session in
`docs/m2-brief.md`; the implementation applied it faithfully and flagged the problem.

**Reconnaissance by the planning session** (not an M3 result; dt = 1 ms, constant current 30
to the 114 JO-A/B neurons with synapse sites for 100 ms, 200 ms observation, seed 0):

| g | JO spikes | downstream spikes | downstream neurons | per-50 ms downstream |
|---|---|---|---|---|
| 0.200 | 684 | 102 | 31 (0.02 %) | 39, 62, 1, 0, 0, 0 |
| 0.336 | 682 | 366 | 136 (0.08 %) | 124, 201, 41, 0, 0, 0 |
| 0.450 | 671 | 497 | 196 (0.12 %) | 241, 210, 46, 0, 0, 0 |
| 0.500 | 656 | 622 | 245 (0.15 %) | 284, 268, 70, 0, 0, 0 |

Input reaches downstream neurons and activity dies within 50 ms of offset at every g tried.

**Decision (planning session, rule stated before choosing):** M3 uses
`g = 0.336` — the largest point of the M2 primary grid that was EXTINCT (silent after
offset) for all three stimulus seeds in `m2-sweep.json` / `m2-sweep-supplementary.json`.
A run in which activity persists > 100 ms after stimulus offset with ≥ 1 % of neurons active
is recorded as "ignited" and treated as a failed run, not a result. `v_floor` stays `None`.
`DEFAULT_G` in the engine is left as M2 documented it; M3 overrides it explicitly and
records the value in every run's `meta.engine_params`.

**Not decided yet:** whether to move to conductance-based synapses (reversal potentials)
to remove the unphysical hyperpolarisation. Deferred until M3/M4 show whether the
input-driven regime is sufficient.

## 2026-09-13 — M3 input gain (a_in) sweep and adapter assumptions

All in `flysim/apps/m3_click.py` / `flysim/sensory/jo.py`, results in `docs/m3-report.md`.

- Pre-defined `a_in` grid: 10, 20, 40, 80, 160, 320 (log2). The first sweep of this grid gave
  **no PASS and no ignition in either mode** (rate: JO_post_R 0.84 Hz at 320; phase-lock: 0.23 Hz).
  Extension rule added *after* that sweep and applied identically to both modes: keep doubling
  `a_in` until the first PASS, then two more doublings; stop at IGNITED / failure or 5120.
  Result: first PASS at a_in = 640 (rate) and 2560 (phase-lock); no run ignited (late-window
  active fraction 0 % everywhere). The judgement criteria themselves were not changed.
- `g = 0.336` untouched (rule of the previous entry).
- Adapter assumptions: band 100–400 Hz as HP2 ∘ LP2 Butterworth biquads; envelope τ = 2 ms;
  half-wave rectification in phase-lock mode; carrier 200 Hz = geometric centre of the band;
  pulse 10 ms (2 cycles); `k_min = 5` contacts for JO_post; identical input to both sides.

## 2026-09-13 — no fluctuation-driven low-rate state exists (background-noise reconnaissance)

Planning session, after M3. g = 0.336, dt = 1 ms, no input, 2 s, `noise_sigma` swept (current noise,
per-step voltage kick ≈ (1 − e^{−1/20})·σ ≈ 0.05 σ mV):

| σ | mean rate (Hz, all N) | active % | behaviour |
|---|---|---|---|
| ≤ 10 | 0 | 0 | silent |
| 20 | 0.001 | 0.10 | ~35 spikes / 400 ms network-wide, stable |
| 22 | 4.3 | 13 | ignites after ~1 s (both seeds) |
| 24–29 | 7.6–10.3 | 17–33 | ignites within 0.4 s |
| 30–80 | 10–20 | 27–94 | ignited from the start |

There is no σ at which the network sits in an asynchronous low-rate state: it is silent or ignited.
Cause (assumed, not tested): current-based synapses without reversal potentials plus excitation
dominating by contact count (77.3 M exc vs 46.8 M inh contacts) give no saturation mechanism.

**Decision:** M4 phase A runs the IPI sweep in the silent regime (g = 0.336, σ = 0) with the M3 gains,
and records whatever appears at each stage; a flat pC1 curve is an acceptable result. Whether to move
to conductance-based synapses (phase B) is deferred to the user. No parameter is adjusted to obtain
propagation beyond hop 1.

## 2026-09-14 — phase B approved: conductance-based synapses (M2b)

User approved after M4 phase A (67f85c8) showed propagation stops at hop 1 and no
fluctuation-driven state exists in the current-based model. Reasons recorded: (1) the
current-based model's ignited state reaches v ≈ −4,000 mV, which is unphysical regardless of
any result; (2) it has no low-rate state under any noise. This is a model-structure change,
not a parameter tuned to a target curve. Phase A results stay in `m4-tuning.json` /
`docs/m4-report.md` and every phase-B result is labelled as such. Reversal potentials and
the sweep criteria for phase B are fixed in `docs/m2b-brief.md` before any run.

## 2026-09-14 — M2b outcome: voltages bounded, bistability unchanged

M2b (883c124, M2 session) added `synapse="conductance"` (E_exc 0 / E_inh −75 mV, τ_e 5 / τ_i 10 ms,
single gain). Current-model results are regression-locked (bit-identical). Findings:
- v stays in [−75, −50] mV at every g and seed (M2 current model reached −4,000 mV). Fixed.
- The network is still bistable: SILENT below g ≈ 1.3e-4, IGNITED above ≈ 5.5e-4 (14–35 % active at
  150–245 Hz), first-order transition. Common RESPONSIVE window across 3 seeds is only
  [2.64e-4, 3.79e-4]; `DEFAULT_G_CONDUCTANCE = 3.162e-4` (geometric mean, rule fixed beforehand).
- Background noise at that g: silent up to σ = 20, ignites from σ = 22. **No fluctuation-driven
  low-rate state**, same transition point as the current model.

**Decision:** finish phase B as approved — rerun M3 gain selection and the M4 IPI sweep with the
conductance model (M4b) and record next to phase A. Any further mechanism (adaptation,
short-term depression, inhibitory scaling) is a new decision for the user; none is added here.

## 2026-09-14 — M4b outcome (phase B complete)

M4b (9dfb23a): conductance model, g = 3.162e-4, a_in rate 320 (re-selected by the M3 rule) /
phase-lock 2560 (fallback: no non-ignited PASS; recorded in meta.a_in_source). 5 of 18 runs
ignited (rate IPI 20; phase-lock IPI 20–35), all at the highest pulse densities. In non-ignited
runs propagation now reaches hop 2 (SAD ≈ 1–2, WED ≈ 3–4 spikes/pulse in phase-lock, IPI ≥ 40)
but pC1 stays 0; pC1 > 0 appears only inside ignited runs. No 35 ms peak anywhere.
Phase A and B are recorded side by side in `docs/m4b-report.md`. Nothing was tuned.
Next mechanism (adaptation / short-term depression / inhibitory scaling) awaits the user.

## 2026-09-16 — approved: M2c (adaptation current) and M5 flysim-live cockpit

User approved both. M2c adds a per-neuron spike-frequency adaptation current (τ_w = 100 ms fixed,
b chosen by pre-registered sweep, default 0 keeps bit-identical regression). M5 adds a live
server (M5a) and browser cockpit (M5b) under the contract `docs/m5-protocol.md`; every control
action is logged and snapshots use the existing run.json contract, so exploration never
replaces reproducible runs. New dependency: `websockets` (approved).

## 2026-09-16 — M2c outcome: adaptation works but is NOT adopted as-is

M2c (c5d7178, engine session) added a per-neuron spike-frequency adaptation current
(`w`, tau_w = 100 ms fixed, `adapt_b` default 0.0; current-model and conductance-model
regressions stay bit-identical, verified including a forced ADAPT-branch no-op case).
A target low-rate state does exist: the rule "smallest grid b at which some sigma reaches
0.1-5 Hz, stable, and dies after the noise stops" gives `DEFAULT_ADAPT_B = 56.2145`.

**It is recorded but not adopted.** Three findings, all reproduced independently by the
planning session from `data-provenance/m2c-noise-sweep.json`:

1. **Unphysical hyperpolarisation returns.** `w` sits in the current slot, so it is not
   bounded by any reversal potential. v_min at b = 56.2 is -834 mV (-828...-868 across sigma),
   and -1,420 mV at b = 300. This is the same class of artefact that M2b removed from the
   synapses (M2 current model reached -4,136 mV).
2. **The "low-rate" state has almost every neuron firing.** Active fraction over the last
   second is 24.9 % at sigma = 29.1, but 96.6 % at 47.1, 99.99 % at 76.3 and 100 % at 123.6.
   CLAUDE.md 3.3 and the viewer call >= 90 % a parameter failure, so only the single
   sigma = 29.1 point is admissible.
3. **`DEFAULT_ADAPT_B` and `DEFAULT_G_CONDUCTANCE` are not a usable pair.** With b = 56.2
   the RESPONSIVE window moves up to ~4.33e-4 and g = 3.162e-4 becomes SILENT at all three
   seeds. The window did not widen; the transition stays first-order.

Adaptation does do what it was added for - it lowers rates monotonically and it removes
self-sustained activity sharply between b = 24.3 (3.9-7.6 % tail) and b = 56.2 (<= 0.22 %) -
but the state it leaves is not usable as a background regime.

**Decision: hold.** No parameter is changed; `adapt_b` stays 0.0 everywhere, so M3/M4/live
behaviour is unchanged. The next step, if the user approves, is to model adaptation as an
outward *conductance* with a reversal potential (g_sra, E ~ E_inh) instead of a current, so
it is structurally bounded the way M2b's synapses are, together with a pre-registered joint
(b, g) selection rule. The engine session also noted, untested, that a longer tau_w might
reach the same suppression with smaller per-spike jumps.

## 2026-09-16 - M5 flysim-live complete and verified end to end

Server 8c2c2cf (M5a), cockpit 11452e3 + 3cf2ba2 (M5b), contract v1.1 (0f11e2e).
Verified by the planning session against a live server, not by either implementing session:
silence gives 0.000 Hz in every region; a 35 ms click train drives JO_AB to the 1,000 Hz
bin ceiling, JO_post to 57/50 Hz and WED to 4.5 Hz (hop 2, matching M4b); `params`
broadcast, `n_bins`, `get_hops` and `snapshot` all behave per contract; the snapshot is a
valid viewer run.json (1,000 frames, 0.57 % of neurons active, no failure warning); static
serving works and path escape returns 404. Pacing: speed 1 holds at dt 1.0 ms and 0.1 ms.

## 2026-09-19 — M2d: adaptation as a conductance fixes the voltage artefact

M2d (b166749, engine session) moves adaptation from the current slot into the conductance
slot: `g_a` decays with tau_a = 100 ms, increments by `b_g` per spike, and enters both `G`
and `v_inf` against `E_adapt = -75 mV`. Only valid with `synapse="conductance"`; combining it
with the current model, or with M2c's `adapt_b`, raises. Default `adapt_g_b = 0.0`, so M3/M4
and the live server are untouched and the regression stays bit-identical (a new case at
g = 1e-3 with noise, 946,163 spikes, was captured before the edit and replays identically).

**The artefact is gone, structurally.** Verified independently by the planning session from
`data-provenance/m2d-grid.json`: across all 288 grid runs v_min = -74.95 mV against a bound of
-75.0 mV, zero violations, v_max never reaches threshold. `v_inf` is a convex combination of
the reversal potentials, so no value of `g_a` can pull v past `E_adapt`. Compare: M2 current
synapses -4,136 mV, M2c adaptation current -834 mV.

| model | adaptation enters as | sweep v_min | bounded by construction |
|---|---|---|---|
| M2 current synapses | — | -4,136 mV | no |
| M2b conductance synapses | — | -75.0 mV | yes |
| M2c adaptation current | `i_ext - w` | -834 mV | no |
| M2d adaptation conductance | `G`, `v_inf`, `E_adapt` | **-74.95 mV** | **yes** |

**It also widens the usable gain band**, from x1.44 in g (M2b) to x3.7 at b_g >= 0.53.
26 of 96 grid cells are RESPONSIVE on all three seeds.

**A usable background state exists but is marginal, and is not adopted.** Of 312 noise runs
exactly one satisfies all four conditions: b_g = 1.0, g = 5.365e-4, sigma = 29.13, with 29.13 %
of neurons active against the 30 % cap — a margin of 0.87 points, with the neighbouring cell
failing at 30.48 %. Worse, b_g = 1.0 is the **top edge of the pre-registered grid**, so the rule
"smallest b_g that works" resolved to the boundary and nothing above it was tested. The engine
session refused to extend the grid post hoc, which is right: that is a new pre-registered
decision, taken as M2e (`docs/m2e-brief.md`).

**The fourth condition earned its place.** 32 noise runs pass M2c's three conditions and fail
only the new active-fraction cap; their median active fraction is 96.9 % and 23 of them sit
above the 90 % line CLAUDE.md §3.3 calls a parameter failure. Without it this sweep would have
reported 33 usable states, 23 of them failures by the project's own rule — which is exactly
what happened in M2c.

`DEFAULT_ADAPT_G_B = 1.0` and `DEFAULT_G_WITH_ADAPT = 5.36539e-4` are recorded constants, not
dataclass defaults. **Not adopted** pending M2e.

**Open:** the M2d bench could not verify the 100 us target. The GPU was power-capped at
1,455 of 3,105 MHz (40 W) with another process resident, and the *current*-model rows — a code
path M2d does not touch and which is bit-identical — inflated by the same 1.4-3.4x, which is
the evidence that the slowdown is environmental. The adaptation branch costs +1.5 us/step,
measured as a matched pair in the same conditions, and that increment is valid. A rerun on an
idle, unthrottled GPU is the only outstanding item.
