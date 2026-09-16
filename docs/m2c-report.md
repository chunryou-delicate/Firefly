# M2c report — spike-frequency adaptation current (engine extension)

Engine session, 2026-09-16. Spec: `docs/m2c-brief.md`. Raw results: `data-provenance/m2c-b-sweep.json`,
`m2c-noise-sweep.json`, `m2c-g-resweep.json`, `m2c-bench.json`, `m2-regression-spikes.npz`.
Hardware/software as in M2/M2b (RTX 4090 Laptop GPU, torch 2.6.0+cu124, triton 3.2.0).

**This is a missing-mechanism fix, not a change made to obtain a result curve.** Both synapse models are
bistable — silent, or ignited at 150–245 Hz — with no low-rate state at any noise level
(`docs/m2b-report.md` §6, `data-provenance/parameter-decisions.md`). The cause recorded there is that
nothing makes a neuron's own firing oppose itself. Adaptation is the standard mechanism for that. τ_w was
fixed by the brief before any run and `b` was chosen by the pre-registered rule in §5; no parameter was
adjusted after seeing a result.

## 1. Model (ASSUMPTION; nothing below is in the connectome)

One extra per-neuron state `w`, an outward current in the same units and the same slot as `i_ext`. Per step,
for **both** synapse models:

```
w      <- w * exp(-dt/tau_w)            every step, before the membrane update
i_eff  =  i_ext (+ noise) - w           the external-current slot of the M2/M2b equations
w      <- w + adapt_b                   for every neuron that spiked this step
```

Concretely `i_eff` replaces `i_ext` in `v <- v_rest + (v - v_rest)*a_m + (1 - a_m)*i_eff + c_s*i_syn`
(current model) and in `v_inf = ((v_rest + i_eff)/tau_m + ge_bar*E_exc + gi_bar*E_inh)/G` (conductance model).

| item | value | note |
|---|---|---|
| `adapt_tau_w` | **100 ms, fixed** | brief; the usual range is 50–300 ms (ASSUMPTION). Not swept, not tuned. |
| `adapt_b` | swept (§4); `DEFAULT_ADAPT_B` per the rule in §5 | increment of `w` per spike, in mV of steady-state drive (R = 1) |
| default | `adapt_b = 0.0` | adaptation off = the pre-M2c engine, bit for bit (§3) |
| sign | `w >= 0`, subtracted | `adapt_b < 0` is rejected by `EngineParams` — this is an adaptation current, not a facilitation |
| ordering | decay, then use, then increment | a spike at step *t* first affects step *t+1*, matching the one-step synaptic delay |

Analytically, a neuron firing regularly at rate *r* settles on the periodic orbit
`w = b/(1 - exp(-ISI/tau_w))` just after each spike, i.e. `w_ss ≈ b·tau_w·r` when `ISI << tau_w`. This is the
scale used to choose the sweep grid (§4.1) and it is checked directly by
`test_single_neuron_isi_increases_and_rate_drops`.

## 2. Implementation

- `params.py`: `adapt_b` (default 0.0) and `adapt_tau_w` (default 100.0) fields; derived
  `decay_w` and `adapt`; both in `to_dict()`. Validation: `adapt_b >= 0`, `adapt_tau_w > 0`.
- `lif.py`: one float32 state vector `self.w`. The Triton `_lif_kernel` gains an `ADAPT` constexpr; when it is
  False **no line of the pre-M2c kernel changes**, because the adaptation block only rewrites the local
  `iext` variable before the (textually unchanged) membrane code and adds one masked store after the spike
  decision. The torch reference backend implements the same three lines. `w` is included in `reset()`,
  `gpu_memory_bytes()` and the CUDA-graph snapshot/restore list.
- **Determinism basis unchanged.** `w` is a per-neuron quantity read and written by exactly one thread per
  neuron; the only cross-thread reductions in the engine are still the int32 atomics of the propagate
  kernels. Verified on the full graph in the ignited regime across repeats, a fresh engine and CUDA-graph
  replay (`test_adaptation_deterministic_real_graph`).
- Memory: +1 array of N float32 (0.66 MB); engine tensors 346.8 MB (M2b: 346.2 MB).
- `sweep_conductance.py`: `--adapt`, `--adapt --noise`, `--adapt --g-resweep`. The M2b verdict functions
  (`judge_g`, the noise predicates) are reused **unmodified**; M2c adds grids and drivers only.

## 3. Regression: `adapt_b = 0` reproduces M2b bit for bit

A seventh case, `real_cond_default` — conductance model, `g = DEFAULT_G_CONDUCTANCE = 3.162e-4`, full graph,
100 steps, the standard 100-neuron stimulus, seed 0 — was added to `tests/regression_cases.py` and its
reference **generated with the unmodified M2b engine at commit `15b880d`, before any M2c edit**
(619 spikes). `regression_cases.py --write --only NAME` merges a single case into the existing `.npz` so the
six M2 arrays keep their original generation; the file now records both generations in `meta_json`.

| case | model | reference generated at | steps | spikes | result |
|---|---|---|---|---|---|
| real_g1, real_g1_cudagraph, real_g0336_noise20, real_dt01_g0336, synth_cpu, synth_cuda_torch | current (M2) | 0b2c19b | 100–500 | 218,067 … 820 | identical |
| **real_cond_default** | **conductance (M2b)** | **15b880d** | **100** | **619** | **identical** |

Beyond the replay, `test_adapt_branch_is_a_noop_at_zero_b` runs `adapt_b = 1e-30` — small enough that
`i_ext - w == i_ext` exactly in float32 — through the `ADAPT = True` branch on both synapse models, both
devices, with noise on, and requires bit-identical spikes and final `v` against `adapt_b = 0`. So the branch
itself, not merely the default, is a no-op at zero.

## 4. Sweeps

All at `dt = 1 ms` on the full retained graph (165,122 neurons / 25,563,197 directed edges), conductance
synapses, `g = DEFAULT_G_CONDUCTANCE = 3.162e-4` fixed in code before running, `tau_w = 100 ms`.

### 4.1 Grid choice (recorded before the sweeps)

`B_GRID` = 12 log-spaced points in **[0.03, 300]**. `b` has the units of `i_ext`, so a neuron firing at rate
*r* settles at `w_ss ≈ b·tau_w·r = b·(0.1 s)·r`. Setting `w_ss` equal to the 15 mV threshold gap gives the
`b` at which adaptation is worth one full threshold gap at a given rate:

| rate | where it comes from | `b` for `w_ss` = 15 mV |
|---|---|---|
| 150 Hz | ignited saturation rate, `m2b-report.md` §5 | 1.0 |
| 15 Hz | noise-ignited rate, `m2b-report.md` §6 | 10 |
| 1 Hz | middle of the target range | 150 |

The grid brackets all three, with a decade below the first (`b = 0.03` → `w_ss` = 0.045 mV at 15 Hz, below
the 0.065 mV unitary EPSP: a negligibility control) and a factor 2 above the last.

**Prototype run before fixing the grid** (`b = 0, 0.01, 0.1, 1, 10, 100`; σ = 29.13 noise for 2 s, and the
seed-0 stimulus protocol; seed 0):

| b | noise σ=29.13: rate last s (Hz) | active last s | w max | stimulus: stim-neuron spikes | downstream |
|---|---|---|---|---|---|
| 0 | 15.458 | 32.37% | 0 | 599 | 20 |
| 0.01 | 15.506 | 32.55% | 0.34 | 599 | 20 |
| 0.1 | 15.302 | 32.48% | 3.38 | 599 | 20 |
| 1 | 13.433 | 33.38% | 33.84 | 500 | 18 |
| 10 | 3.550 | 32.37% | 338.36 | 300 | 4 |
| 100 | 0.369 | 24.03% | 1294.61 | 100 | 1 |

`b ≤ 0.1` changes nothing; the ignited rate starts falling at `b ≈ 1` and is inside the target band by
`b = 10`. The transition therefore lies inside the grid, as required before running.

### 4.2 b response sweep (`sweep.py --synapse conductance --adapt`)

M2b stimulus protocol (100 random neurons, `i_ext` = 30 for 100 ms, then 400 ms of nothing; seeds 0/1/2) at
`g = 3.162e-4`, over `B_GRID`. Verdict function is M2b's `judge_g`, unmodified. 36 runs, 17 s.

| adapt_b | stim-neuron spikes (s0/s1/s2) | downstream during stim (s0/s1/s2) | post 0-50 ms | late active | verdicts |
|---|---|---|---|---|---|
| 0.030 | 599 / 603 / 596 | 20 / 14 / 38 | 0 / 3 / 0 | 0% | RESPONSIVE ×3 |
| 0.069 | 599 / 603 / 597 | 20 / 12 / 37 | 0 / 3 / 1 | 0% | RESPONSIVE ×3 |
| 0.160 | 599 / 603 / 596 | 19 / 12 / 32 | 0 / 3 / 3 | 0% | RESPONSIVE ×3 |
| 0.370 | 599 / 602 / 595 | 19 / 9 / 30 | 0 / 2 / 2 | 0% | RESPONSIVE ×3 |
| 0.854 | 500 / 504 / 500 | 18 / 9 / 28 | 0 / 0 / 0 | 0% | RESPONSIVE ×3 |
| 1.974 | 500 / 502 / 498 | 14 / 7 / 19 | 0 / 0 / 5 | 0% | RESPONSIVE ×3 |
| 4.560 | 400 / 402 / 399 | 8 / 3 / 13 | 0 / 0 / 0 | 0% | RESPONSIVE ×3 |
| 10.534 | 300 / 301 / 299 | 4 / 1 / 8 | 0 / 0 / 0 | 0% | RESPONSIVE ×3 |
| 24.334 | 200 / 200 / 199 | 1 / 0 / 3 | 0 / 0 / 0 | 0% | RESPONSIVE / **SILENT** / RESPONSIVE |
| 56.215 | 100 / 100 / 100 | 1 / 0 / 3 | 0 / 0 / 0 | 0% | RESPONSIVE / **SILENT** / RESPONSIVE |
| 129.863 | 100 / 100 / 100 | 1 / 0 / 3 | 0 / 0 / 0 | 0% | RESPONSIVE / **SILENT** / RESPONSIVE |
| 300.000 | 100 / 100 / 100 | 1 / 0 / 3 | 0 / 0 / 0 | 0% | RESPONSIVE / **SILENT** / RESPONSIVE |

**No run ignited at any b** (late active fraction 0.000 % everywhere), which is expected: `g = 3.162e-4` is
inside M2b's RESPONSIVE window, so there is no saturated state here for adaptation to cut down. The brief's
"record the ignited active fraction and rate where ignition occurs" therefore has no rows; ignition under
adaptation is measured instead in §4.3 (noise) and §4.4 (higher g).

What the sweep does show is the cost side: adaptation throttles the **stimulated** neurons monotonically —
6 spikes each over the 100 ms stimulus at `b ≤ 0.37`, 5 at 0.854, 4 at 1.97, 3 at 4.56, 2 at 10.5, and exactly
1 each from `b = 56.2` upward — and downstream spikes fall with them (20 → 1 for seed 0). Seed 1 crosses to
SILENT at `b = 24.3`. `w` at the end of the run reaches only 2.5 even at `b = 300`, because a neuron that has
stopped firing stops accumulating.

### 4.3 Noise sweep (`sweep.py --synapse conductance --adapt --noise`)

Every `b` of `B_GRID` × the M2b σ grid: no input, 2 s of noise, then 0.4 s with the noise off (state kept).
144 runs, 46 s. **Target state** (pre-registered in `sweep_conductance.py`, identical to M2b's
`STABLE_LOW_RATE`): last-second rate in [0.1, 5] Hz/neuron AND last/first second ratio in [0.5, 2] AND
< 1 % of neurons active in the final 200 ms after the noise stops.

`T` = target state, `I` = ignited, `·` = silent / sub-threshold:

| adapt_b \ σ | 1.00 | 1.62 | 2.62 | 4.24 | 6.87 | 11.12 | 17.99 | 29.13 | 47.15 | 76.32 | 123.55 | 200.00 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.030 | · | · | · | · | · | · | · | I | I | I | I | I |
| 0.069 | · | · | · | · | · | · | · | I | I | I | I | I |
| 0.160 | · | · | · | · | · | · | · | I | I | I | I | I |
| 0.370 | · | · | · | · | · | · | · | I | I | I | I | I |
| 0.854 | · | · | · | · | · | · | · | I | I | I | I | I |
| 1.974 | · | · | · | · | · | · | · | I | I | I | I | I |
| 4.560 | · | · | · | · | · | · | · | I | I | I | I | I |
| 10.534 | · | · | · | · | · | · | · | I | I | I | I | I |
| 24.334 | · | · | · | · | · | · | · | I | I | I | I | I |
| **56.215** | · | · | · | · | · | · | · | **T** | **T** | **T** | · | · |
| 129.863 | · | · | · | · | · | · | · | **T** | **T** | I | **T** | · |
| 300.000 | · | · | · | · | · | · | · | **T** | **T** | **T** | **T** | **T** |

Mean rate in the last noise-on second (Hz per neuron, all N) — monotonically decreasing in `b` at every σ:

| adapt_b \ σ | 29.13 | 47.15 | 76.32 | 123.55 | 200.00 |
|---|---|---|---|---|---|
| 0.030 | 15.50 | 20.15 | 31.70 | 57.62 | 89.15 |
| 0.370 | 14.65 | 19.28 | 30.31 | 55.25 | 86.11 |
| 1.974 | 11.71 | 15.52 | 25.12 | 45.92 | 73.95 |
| 4.560 | 7.38 | 10.92 | 19.01 | 35.16 | 59.40 |
| 10.534 | 3.39 | 6.22 | 11.82 | 22.14 | 39.51 |
| 24.334 | 1.51 | 3.65 | 6.90 | 12.26 | 21.55 |
| **56.215** | **0.35** | **2.06** | **3.92** | 6.32 | 10.50 |
| 129.863 | 0.34 | 1.64 | 2.80 | 3.98 | 5.65 |
| 300.000 | 0.26 | 1.40 | 2.22 | 2.94 | 3.75 |

(σ ≤ 17.99 is 0.00 Hz at every b — unchanged from M2b, since with no spikes there is no adaptation.)

Detail for the rows that decide the answer:

| adapt_b | σ | rate first s | rate last s | active last s | last/first | tail active (noise off) | v min (mV) | verdict |
|---|---|---|---|---|---|---|---|---|
| 10.534 | 29.13 | 3.575 | 3.388 | 32.43% | 0.95 | 9.584% | -214.0 | IGNITED |
| 10.534 | 47.15 | 6.627 | 6.220 | 95.26% | 0.94 | 10.237% | -198.2 | IGNITED |
| 24.334 | 29.13 | 1.592 | 1.512 | 28.76% | 0.95 | 7.031% | -247.6 | IGNITED |
| 24.334 | 200.00 | 23.339 | 21.546 | 100.00% | 0.92 | 3.901% | -511.4 | IGNITED |
| **56.215** | **29.13** | 0.523 | **0.355** | **24.88%** | 0.68 | **0.215%** | **-834.2** | **TARGET** |
| **56.215** | **47.15** | 2.402 | **2.060** | 96.57% | 0.86 | **0.002%** | -828.1 | **TARGET** |
| **56.215** | **76.32** | 4.318 | **3.915** | 99.99% | 0.91 | **0.117%** | -841.0 | **TARGET** |
| 56.215 | 123.55 | 6.967 | 6.316 | 100.00% | 0.91 | 0.154% | -867.6 | - (rate > 5 Hz) |
| 129.863 | 76.32 | 3.198 | 2.799 | 99.99% | 0.88 | 1.456% | -995.6 | IGNITED (tail 1.46 %) |
| 300.000 | 200.00 | 4.237 | 3.750 | 100.00% | 0.88 | 0.000% | -1420.5 | TARGET |

**Result: the target state exists, and it first appears at `b = 56.215`.** Three points about it, all of
which matter more than the headline:

1. **It is the self-sustained criterion that `b` has to break, not the rate criterion.** The rate is already
   inside [0.1, 5] Hz at `b = 10.5` (3.39 Hz) and `b = 24.3` (1.51 Hz); both are still IGNITED because
   7–10 % of neurons keep firing after the noise is removed. The tail collapses between `b = 24.3`
   (3.9–7.6 %) and `b = 56.2` (0.00–0.22 %). Adaptation lowers the rate smoothly but only removes
   self-sustenance abruptly.
2. **The adaptation current reintroduces unbounded hyperpolarisation — the exact pathology M2b existed to
   fix.** `w` enters the external-current slot, and a current is not bounded by the reversal potentials
   (same mechanism as the noise, `m2b-report.md` §1). v min goes from −230 mV at `b ≈ 0` (noise alone) to
   **−834 mV at `b = 56.2` and −1,420 mV at `b = 300`**. M2's rejected current-based model reached −4,136 mV
   at its default g, so this is the same order of problem. Reported, not fixed: the brief specifies `w` as a
   current and forbids adding mechanisms.
3. **In most target runs nearly every neuron is firing.** Active fraction in the last noise-on second is
   96.6 % at σ = 47.15 and 99.99 % at σ = 76.32; only σ = 29.13 (24.9 %) is below the 90 % line that
   CLAUDE.md §3.3 and the viewer treat as a parameter failure. If a target state is adopted, σ = 29.13 is the
   only one of the three that would survive that check.

`w` values quoted in the JSON (`w_max`, `w_mean`) are sampled **at the end of the run**, i.e. after the
0.4 s noise-off tail has decayed them by exp(−4) ≈ 0.018; they understate the in-run magnitude by ~50×. The
per-step-tracked `v_min` is the reliable indicator and is what point 2 uses.

### 4.4 g re-sweep at b = 56.215 (`sweep.py --synapse conductance --adapt --g-resweep`)

M2b g-sweep protocol, 12 log-spaced g in [1e-5, 1e-2] (the brief's 12 points over M2b's span), 3 seeds,
`adapt_b = 56.215`.

| g (1/ms) | verdicts (s0 / s1 / s2) | downstream during stim (s0/s1/s2) | late active (s0/s1/s2) | v min |
|---|---|---|---|---|
| 1.000e-05 … 2.310e-04 (6 points) | SILENT ×3 each | 0 / 0 / 0 | 0 % | −80.7 |
| **4.329e-04** | **RESPONSIVE ×3** | 4 / 2 / 6 | 0 % | −100.9 |
| 8.111e-04 | RESPONSIVE / IGNITED / IGNITED | 14 / 123,798 / 124,495 | 0 % / 18.5 % / 19.7 % | −570.9 |
| 1.520e-03 | IGNITED ×3 | 272,300 / 274,688 / 274,595 | 32.2–33.4 % | −629.9 |
| 2.848e-03 | IGNITED ×3 | 449,097 / 443,991 / 451,027 | 41.4–43.6 % | −694.4 |
| 5.337e-03 | IGNITED ×3 | 644,880 / 640,343 / 634,492 | 46.9–48.4 % | −854.1 |
| 1.000e-02 | IGNITED ×3 | 836,400 / 851,676 / 837,931 | 50.1–51.3 % | −480.1 |

**The RESPONSIVE window did not widen.** Common to all three seeds it is the single grid point
**4.329e-4**, bounded by SILENT at 2.310e-4 and IGNITED (2 of 3 seeds) at 8.111e-4, so the window is
contained in (2.31e-4, 8.11e-4) — at most a factor 3.5 wide, and unresolved by this grid. M2b's window was
[2.637e-4, 3.793e-4], a factor 1.44, resolved by 2 adjacent points of a 20-point grid. **The two are not
directly comparable in point counts** (12 points ⇒ spacing 1.874, 20 points ⇒ spacing 1.259); what can be
said is that at `b = 56.2` the transition still falls between adjacent grid points, i.e. it is still
first-order, and adaptation did not turn it into a broad window.

Two further consequences, both against adopting this pair of values:

- **The window moved up, past the recorded default.** At `b = 56.2`, `g = 3.162e-4` (= `DEFAULT_G_CONDUCTANCE`)
  is SILENT for all three seeds. `DEFAULT_ADAPT_B` and `DEFAULT_G_CONDUCTANCE` are therefore **not a usable
  pair**: applying the M2c default to an M2b-tuned run switches the response off. Re-selecting g would be a
  new pre-registered decision, not something this session does.
- **Ignited runs at high g are worse, not better**: 32–51 % active (M2b at the same g: 23–34 %), at 26–82 Hz
  each (M2b: 176–245 Hz). So adaptation does cut the per-neuron saturation rate roughly 3-fold, exactly as
  intended, but it recruits more neurons at the same time and v reaches −854 mV.

## 5. `DEFAULT_ADAPT_B`

Rule, fixed in `sweep_conductance.py` before the sweeps: *the smallest b on B_GRID for which some σ reaches
the target state; 0 if none does.*

**`DEFAULT_ADAPT_B = 56.21452268581154`** (`data-provenance/m2c-noise-sweep.json`), enforced against the
recorded JSON by `test_default_adapt_b_matches_recorded_sweep_rule`.

**It is not the dataclass default.** `EngineParams.adapt_b` defaults to `0.0` exactly as the brief specifies,
so M3/M4/live and every existing call site are unaffected until a caller passes `adapt_b` explicitly. Given
§4.3 point 2 (v to −834 mV), §4.3 point 3 (> 90 % of neurons active at 2 of the 3 target σ) and §4.4
(`g = 3.162e-4` is SILENT at this b), **adopting it is a planning decision, and this report does not
recommend switching it on as-is.** The noise default stays 0.

## 6. Performance (`bench.py --synapse both --adapt`, full graph, dt = 0.1 ms, method as in M2/M2b)

`--adapt` adds a pass with the `ADAPT` branch compiled in, at `adapt_b = 1e-6`: per-step cost depends on
whether the constexpr is set, not on the value of b, and 1e-6 leaves the protocol's forced activity exactly
as it is (confirmed — spikes/step equals the forced neuron count in every adapt row). Run of record is the
second of two; `m2c-bench-run1.json` keeps the first.

| synapse | backend | activity | recorder | adapt off (μs) | **adapt on (μs)** | Δ | run 1, adapt on | < 100 μs |
|---|---|---|---|---|---|---|---|---|
| current | triton-eager | 0.0% | off | 67.6 | **77.0** | +9.4 | 70.7 | yes |
| current | triton-eager | 0.0% | on | 69.3 | **76.1** | +6.8 | 72.9 | yes |
| current | triton-cudagraph | 0.0% | off | 35.6 | **37.7** | +2.1 | 42.3 | yes |
| current | triton-cudagraph | 0.0% | on | 36.7 | **41.6** | +4.9 | 44.8 | yes |
| current | triton-eager | 0.1% | off | 73.2 | **74.2** | +1.0 | 73.5 | yes |
| current | triton-eager | 0.1% | on | 73.1 | **74.6** | +1.5 | 77.4 | yes |
| current | triton-cudagraph | 0.1% | off | 38.0 | **39.7** | +1.7 | 40.5 | yes |
| current | triton-cudagraph | 0.1% | on | 41.0 | **44.2** | +3.2 | 47.5 | yes |
| current | triton-eager | 1.0% | off | 70.4 | **75.6** | +5.2 | 71.8 | yes |
| current | triton-eager | 1.0% | on | 81.6 | **84.9** | +3.3 | 90.7 | yes |
| current | triton-cudagraph | 1.0% | off | 48.2 | **49.9** | +1.7 | 57.0 | yes |
| current | triton-cudagraph | 1.0% | on | 58.6 | **60.7** | +2.1 | 55.6 | yes |
| current | triton-eager | 5.0% | off | 87.9 | **89.5** | +1.6 | 90.8 | yes |
| current | triton-eager | 5.0% | on | 125.9 | **129.4** | +3.5 | 124.9 | **NO** |
| current | triton-cudagraph | 5.0% | off | 89.4 | **92.1** | +2.7 | 84.7 | yes |
| current | triton-cudagraph | 5.0% | on | 115.8 | **118.6** | +2.8 | 109.7 | **NO** |
| conductance | triton-eager | 0.0% | off | 72.7 | **74.2** | +1.5 | 74.6 | yes |
| conductance | triton-eager | 0.0% | on | 72.7 | **74.8** | +2.1 | 75.3 | yes |
| conductance | triton-cudagraph | 0.0% | off | 41.1 | **42.5** | +1.4 | 38.9 | yes |
| conductance | triton-cudagraph | 0.0% | on | 42.7 | **43.5** | +0.8 | 40.1 | yes |
| conductance | triton-eager | 0.1% | off | 72.3 | **74.3** | +2.0 | 73.6 | yes |
| conductance | triton-eager | 0.1% | on | 77.2 | **77.1** | −0.1 | 76.4 | yes |
| conductance | triton-cudagraph | 0.1% | off | 43.7 | **45.2** | +1.5 | 41.2 | yes |
| conductance | triton-cudagraph | 0.1% | on | 48.2 | **52.0** | +3.8 | 45.1 | yes |
| conductance | triton-eager | 1.0% | off | 72.6 | **73.3** | +0.7 | 73.8 | yes |
| conductance | triton-eager | 1.0% | on | 84.5 | **81.5** | −3.0 | 81.5 | yes |
| conductance | triton-cudagraph | 1.0% | off | 50.7 | **52.1** | +1.4 | 47.1 | yes |
| conductance | triton-cudagraph | 1.0% | on | 61.0 | **62.3** | +1.3 | 55.9 | yes |
| conductance | triton-eager | 5.0% | off | 92.0 | **92.9** | +0.9 | 84.2 | yes |
| conductance | triton-eager | 5.0% | on | 135.2 | **133.3** | −1.9 | 126.6 | **NO** |
| conductance | triton-cudagraph | 5.0% | off | 93.8 | **95.1** | +1.3 | 92.3 | yes |
| conductance | triton-cudagraph | 5.0% | on | 136.3 | **122.6** | −13.7 | 135.5 | **NO** |

Adaptation costs **+1.7 μs/step on average across the 32 Triton cells** (range −13.7 to +9.4; the negative
entries are run-to-run noise, not a speed-up) — one extra load, multiply, subtract, select and store per
neuron. **The only cells above 100 μs are the same ones as in M2 and M2b: 5 % of all neurons spiking every
0.1 ms step with the recorder on** (118–133 μs; kernel-only 89–95 μs), caused by the recorder flush, not by
the model. Every other configuration, with adaptation on, meets the < 100 μs target.

The `torch-gpu` rows are omitted from the table above because they are not reliable at this protocol's
1,000-step measurement: across the two runs the same cell varied by up to 3× (e.g. current / 0.1 % / recorder
on: 1,351 μs in run 1, 620 μs adapt-off and 430 μs adapt-on in run 2). All `torch-gpu` rows are far above
100 μs in both runs, as they have been since M2 — it is the documented slow fallback, not the target — and
all values are in the two JSON files.

Engine tensors: 346.8 MB per engine (M2b: 346.2 MB) — `w` adds one float32 array of N = 0.66 MB.

## 7. Tests (`pytest tests/`, engine scope: 98 passed = 78 existing + 20 new)

`tests/test_engine_adapt.py`:

- **b = 0 is the pre-M2c engine**: the `real_cond_default` regression replay (§3); the `ADAPT`-branch no-op
  test at b = 1e-30 (2 synapse models × 2 devices, with noise); `adapt_b` = 0 and `adapt_tau_w` = 100 by
  default for both synapse models; `DEFAULT_ADAPT_B` matches the recorded sweep rule.
- **Single neuron under constant current** (2 synapse models × 2 devices): ISI monotonically non-decreasing,
  strictly longer at the end than at the start, b = 0 control perfectly regular, adapted steady-state rate
  below the unadapted one, and `w` on its analytic periodic orbit `b/(1 − exp(−ISI/τ_w))`. Spike count
  monotonically decreasing in b over (0, 0.5, 2, 8). `w` decays by exp(−dt/τ_w) after the drive stops and is
  cleared by `reset()`. dt = 1 ms vs 0.1 ms: ISI sequences agree interval by interval within one coarse step
  and the steady-state adapted ISI is identical.
- **Backends and determinism**: Triton == torch spike lists and `w` (both synapse models, 200 steps on the
  synthetic graph); bit-identical on the full graph in the ignited regime across repeats, a fresh engine and
  CUDA-graph replay, with `w` equal too; adaptation lowers the late-window spike count on the full graph.

`tests/test_engine.py` (22) and `tests/test_engine_conductance.py` (23) pass unchanged; the latter's
regression test is now parametrised over the current-model cases only, with the conductance case replayed in
`test_engine_adapt.py`.

## 8. Acceptance checklist (`docs/m2c-brief.md`)

- [x] `EngineParams.adapt_b` (default 0) / `adapt_tau_w` (100, fixed); `w` state; `(i_ext − w)` in both models
- [x] Triton `ADAPT` constexpr; torch reference backend; determinism basis unchanged and tested
- [x] b = 0 bit-identical, with the new conductance regression case generated before the M2c edit (§3)
- [x] 1-neuron constant-current test: ISI increases, steady-state rate below b = 0; Triton == torch (§7)
- [x] b response sweep, 12 log points, range justified by a prototype (§4.1, §4.2)
- [x] noise sweep at every b with the pre-registered target state (§4.3)
- [x] g re-sweep at the minimum b with a target state (§4.4)
- [x] `DEFAULT_ADAPT_B` by the pre-registered rule (§5)
- [x] bench re-run, same table format, < 100 μs except the known recorder cell (§6)
- [x] `pytest tests/` 98 passed in engine scope (198 with the live session's files, which this session does not own)

## 9. Deviations from the brief

1. **`DEFAULT_ADAPT_B` is a recorded constant, not the dataclass default.** The brief asks for both
   `adapt_b: float = 0.0` and `DEFAULT_ADAPT_B` = the swept value; these conflict if one field carries both.
   Resolved in favour of the explicit field default, because making the swept value the default would
   silently switch adaptation on for M3, M4 and the live server and break the brief's own bit-identity
   requirement. Callers opt in with `adapt_b=DEFAULT_ADAPT_B` (§5).
2. **The b response sweep has no ignited rows.** The brief asks to record the ignited active fraction and
   rate "at the b where ignition occurs"; at `g = DEFAULT_G_CONDUCTANCE` no run ignites at any b, because
   that g is inside M2b's RESPONSIVE window. The saturation-cutting question is answered instead by §4.3
   (noise-ignited state: 15.5 → 0.35 Hz) and §4.4 (ignited rate 176–245 → 26–82 Hz). Nothing was changed to
   produce ignition.
3. **The noise sweep was run at all 12 b**, not the "representative 4 points" the brief allows, because the
   default rule needs the smallest b on the grid. Cost was 46 s, so the cheaper option was not needed.
4. **The g re-sweep window is not comparable point-for-point with M2b's** (12 points vs 20 over the same
   span). Reported as a bound on the window, with the grid spacing stated, rather than as a point count
   (§4.4).
5. **The bench was run twice** and the second run is the one of record, because run 1 contained `torch-gpu`
   outliers up to 1,351 μs. Both JSON files are kept, run 1's numbers appear in the table, and no row was
   dropped. The Triton rows — the ones the 100 μs target is about — agree between runs within a few μs.
6. `regression_cases.py` gained a `--write --only NAME` merge path so that adding the M2c case does not
   regenerate the six M2 arrays; the `.npz` now records both generations.

## 10. Notes for the planning session

- **The mechanism works and the target state exists**, but at `b = 56.2` it costs v ≈ −830 mV, ≥ 96 % of
  neurons active at 2 of its 3 σ, and it puts `g = 3.162e-4` in the SILENT regime. Recommend treating
  §4.3/§4.4 as the finding and **not** adopting the (b, g) pair as-is.
- **The v bound is the fixable part.** `w` is a current because the brief specified it there. Modelled
  instead as an outward *conductance* with a reversal potential (the usual `g_sra`, E ≈ E_K ≈ E_inh), it
  would be bounded by construction just as M2b's synapses are, and the −830 mV would not occur. That is a
  model-structure decision of the same kind as M2b, so it is not taken here.
- **`b` and `g` are not independent.** The RESPONSIVE window moved from [2.64e-4, 3.79e-4] to about 4.3e-4
  when adaptation was switched on. Any adoption of `adapt_b` needs a joint (b, g) decision with the rule
  written down first.
- **τ_w was not swept** (fixed at 100 ms by the brief). The target state's boundary sits between b = 24.3 and
  56.2, i.e. in `b·τ_w`; a longer τ_w would reach the same `w_ss` at proportionally smaller b and a smaller
  per-spike jump, which may be the cheaper way to get the same suppression with less hyperpolarisation.
  Untested — stated as a hypothesis, not a result.
- `parameter-decisions.md` and `backlog.md` were not edited (planning-owned).
