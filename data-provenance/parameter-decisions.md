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
