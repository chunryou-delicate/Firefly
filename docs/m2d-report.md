# M2d report — adaptation as a conductance, not a current

Engine session (window "M3 엔진"), 2026-09-19. Spec: `docs/m2d-brief.md`. Raw results:
`data-provenance/m2d-grid.json`, `m2d-noise-sweep.json`, `m2d-bench.json`, `m2-regression-spikes.npz`.
Hardware/software as in M2/M2b/M2c (RTX 4090 Laptop GPU, torch 2.6.0+cu124, triton 3.2.0).

**This is a model-defect fix of the same kind as M2b, not a parameter adjusted to obtain a result.**
M2c's adaptation *current* did produce a low-rate state, but two things broke: the membrane potential
reached −834 mV (a current has no reversal potential — exactly the defect M2b fixed for the synapses),
and 96–100 % of neurons were firing in that state, which CLAUDE.md §3.3 defines as a parameter failure.
Real spike-frequency adaptation is a potassium conductance; as a conductance it is bounded by
construction. τ_a is fixed at M2c's τ_w so the two are comparable, and `b_g` comes from the
pre-registered sweep in §4.

## 1. Model (ASSUMPTION; nothing below is in the connectome)

One extra per-neuron state `g_a`, a conductance in 1/ms — the same unit as the synaptic conductances,
against a leak of `1/τ_m` = 0.05/ms:

```
g_a    <- g_a * exp(-dt/tau_a)                       every step, before the membrane update
ga_bar  = g_a * avg_a                                step-average, exactly as for g_e / g_i
G       = 1/tau_m + ge_bar + gi_bar + ga_bar
v_inf   = ((v_rest + i_ext)/tau_m + ge_bar*E_exc + gi_bar*E_inh + ga_bar*E_adapt) / G
v      <- v_inf + (v - v_inf) * exp(-dt*G)
g_a    <- g_a + adapt_g_b                            for every neuron that spiked this step
```

| item | value | note |
|---|---|---|
| `adapt_tau_a` | **100 ms, fixed** | = M2c's `τ_w`, so the two adaptation models are directly comparable. Not swept, not tuned. |
| `E_adapt` | **−75 mV** (= `E_inh`) | ASSUMPTION. The K⁺ reversal is usually −80…−90 mV, but the brief keeps the constant count down. Exposed as a parameter, not hard-coded. |
| `adapt_g_b` | swept (§4); `DEFAULT_ADAPT_G_B` per the rule in §5 | increment of `g_a` per spike, in 1/ms |
| default | `adapt_g_b = 0.0` | adaptation off = the pre-M2d engine, bit for bit (§3) |
| synapse model | **conductance only** | `synapse="current"` with `adapt_g_b > 0` raises `ValueError`: without reversal potentials the concept is meaningless |
| M2c's `adapt_b` | kept, default 0 | history, per the brief. Setting both raises `ValueError` |

**Why this is bounded.** `v_inf` is a convex combination of `v_rest + i_ext`, `E_exc`, `E_inh` and
`E_adapt` with non-negative weights, and the update moves `v` towards `v_inf`. So with `i_ext ≥ 0`,
`v` can never leave `[min(E_inh, E_adapt), v_thresh)` no matter how large `g_a` grows — adaptation can
only pull `v` towards `E_adapt`, never past it. That is the whole point of M2d and it is checked on
every sweep run (§4.2) and by tests on the real graph (§6).

**Step-average `avg_a`.** `ga_bar = g_a · avg_a` with `avg_a = τ_a(1 − e^{−dt/τ_a})/dt`, matching how
M2b treats `g_e`/`g_i` rather than the brief's schematic `G = 1/τ_m + g_e + g_i + g_a`. The brief writes
the synaptic terms the same schematic way, and M2b's report (§1, deviation 1) records why the
step-average is used: it keeps the integrated conductance exact and dt-invariance within one step. With
τ_a = 100 ms the factor is 0.995 at dt = 1 ms, so this is a consistency choice, not a numerical one.

Analytically a neuron firing regularly at rate *r* settles on `g_a = b_g/(1 − e^{−ISI/τ_a})` just after
each spike, i.e. `g_a_ss ≈ b_g·τ_a·r` for `ISI ≪ τ_a`. That is the scale used to pick the grid (§4.1)
and it is checked directly by `test_single_neuron_isi_increases_and_rate_drops`.

## 2. Implementation

- `params.py`: `adapt_g_b` (default 0.0), `adapt_tau_a` (100.0), `E_adapt` (−75.0); derived `decay_a`,
  `avg_a`, `adapt_g`, and `v_lower_bound` = `min(E_inh, E_adapt)`, which is what the acceptance check
  compares against. Validation: `adapt_g_b ≥ 0`, `adapt_tau_a > 0`, `E_adapt ≤ v_reset`, conductance
  synapses only, and mutually exclusive with `adapt_b`.
- `lif.py`: one float32 state `self.g_a`. The Triton `_lif_kernel` gains an `ADAPT_G` constexpr inside
  the conductance branch; with it False the branch is textually the pre-M2d code, so the default engine
  compiles unchanged. The torch reference backend implements the same equations. `g_a` is included in
  `reset()`, `gpu_memory_bytes()` and the CUDA-graph snapshot/restore list.
- **Determinism basis unchanged**: `g_a` is per-neuron, read and written by one thread each; the only
  cross-thread reductions are still the int32 atomics of the propagate kernels.
- Memory: +1 array of N float32 (0.63 MB).
- `sweep_conductance.py`: `--adapt-g` and `--adapt-g --noise`. M2b's `judge_g` is reused unmodified;
  the usable-state predicate is new (`_judge_usable`, §4.3) because the brief adds a fourth condition.

## 3. Regression: `adapt_g_b = 0` reproduces the pre-M2d engine bit for bit

An eighth case, `real_cond_g1e3_noise` — conductance, `g = 1e-3`, `noise_sigma = 10`, full graph, 200
steps, seed 11, 946,163 spikes — was added to `tests/regression_cases.py` and its reference **generated
with the unmodified pre-M2d engine at commit `8d68698`**, before any M2d edit. It is a busier regime
than M2c's `real_cond_default` (which is also still checked), so it exercises the conductance membrane
update under high activity and with noise.

| generation | commit | cases | result |
|---|---|---|---|
| M2 (current model) | 0b2c19b | real_g1, real_g1_cudagraph, real_g0336_noise20, real_dt01_g0336, synth_cpu, synth_cuda_torch | identical |
| M2b (conductance) | 15b880d | real_cond_default | identical |
| **M2d (conductance, busy)** | **8d68698** | **real_cond_g1e3_noise** | **identical** |

Beyond the replay, `test_adapt_g_branch_is_a_noop_at_zero` runs `adapt_g_b = 1e-30` — small enough to be
invisible in float32 — through the `ADAPT_G = True` branch on both devices with noise on, and requires
bit-identical spikes and final `v` against `adapt_g_b = 0`. So the branch itself, not merely the
default, is a no-op at zero.

## 4. Sweeps

All at `dt = 1 ms` on the full retained graph (165,122 neurons / 25,563,197 directed edges), conductance
synapses, `τ_a = 100 ms`, `E_adapt = −75 mV`. Grids and criteria were fixed in `sweep_conductance.py`
before any run.

### 4.1 Grid choice (recorded before the sweeps)

**`B_G_GRID` = 12 log-spaced points in [1e-3, 1].** `g_a` has the unit of the synaptic conductances
(1/ms) and the leak is `1/τ_m` = 0.05/ms, so a neuron firing at rate *r* settles at
`g_a_ss ≈ b_g·τ_a·r = b_g·(0.1 s)·r`. Setting `g_a_ss` equal to the leak — adaptation doubling the
neuron's resting conductance — gives the `b_g` at which adaptation matters at a given rate:

| rate | where it comes from | `b_g` for `g_a_ss` = leak |
|---|---|---|
| 150 Hz | ignited saturation rate, `m2b-report.md` §5 | 0.0033 |
| 15 Hz | noise-ignited rate, `m2b-report.md` §6 | 0.033 |
| 1 Hz | middle of the usable band | 0.5 |

The grid brackets all three. **Prototype before fixing the grid** (σ = 29.13 noise, 2 s, g = 3.162e-4,
seed 0):

| b_g | rate last s (Hz) | active last s | v min | g_a max |
|---|---|---|---|---|
| 0 | 15.458 | 32.37 % | −90.8 | 0 |
| 0.001 | 14.533 | 32.53 % | −90.9 | 0.034 |
| 0.01 | 7.805 | 33.11 % | −91.0 | 0.338 |
| 0.1 | 0.578 | 26.49 % | −91.3 | 2.260 |
| 0.3 | 0.333 | 23.33 % | −91.3 | 1.047 |
| 1.0 | 0.234 | 20.08 % | −91.3 | 1.002 |
| 3.0 | 0.181 | 17.78 % | −91.3 | 3.009 |

`b_g = 1e-3` barely moves the rate, `b_g = 0.1` reaches 0.58 Hz, and it saturates by 1 — the transition
is inside the grid. Note already here what §4.2 makes exact: **v min is ≈ −91 mV at every `b_g`,
including 0**. That −91 is the noise current going negative (M2b §1), not the adaptation; the
adaptation moves it by 0.5 mV between `b_g = 0` and `b_g = 3`, against M2c's −834 mV.

**`G_GRID_ADAPT_G` = 8 log-spaced points in [2e-4, 2e-3]**, the span named by the brief. It brackets
M2b's RESPONSIVE window [2.637e-4, 3.793e-4] and the point M2c's re-sweep moved it to (4.329e-4), with
the top of the range well inside M2b's ignited regime.

### 4.2 Combined (b_g × g) grid — and the voltage bound

M2b's stimulus protocol (100 random neurons, `i_ext` = 30 for 100 ms, then 400 ms; seeds 0/1/2) with
M2b's `judge_g` unmodified. 12 × 8 × 3 = **288 runs**. `S` = SILENT for all seeds, `R` = RESPONSIVE for
all seeds, `I` = ignited for at least one seed, `·` = mixed SILENT/RESPONSIVE:

| b_g \ g | 2.000e-04 | 2.779e-04 | 3.861e-04 | 5.365e-04 | 7.455e-04 | 1.036e-03 | 1.439e-03 | 2.000e-03 |
|---|---|---|---|---|---|---|---|---|
| 0.00100 | · | **R** | **R** | I | I | I | I | I |
| 0.00187 | · | **R** | **R** | I | I | I | I | I |
| 0.00351 | · | **R** | **R** | I | I | I | I | I |
| 0.00658 | · | **R** | **R** | I | I | I | I | I |
| 0.01233 | · | **R** | **R** | I | I | I | I | I |
| 0.02310 | · | · | **R** | I | I | I | I | I |
| 0.04329 | S | · | **R** | **R** | I | I | I | I |
| 0.08111 | S | · | · | **R** | I | I | I | I |
| 0.15199 | S | · | · | **R** | I | I | I | I |
| 0.28480 | S | · | · | **R** | **R** | I | I | I |
| 0.53367 | S | · | · | **R** | **R** | **R** | **R** | I |
| 1.00000 | S | · | · | **R** | **R** | **R** | **R** | **R** |

**Voltage bound: no violation in any of the 288 runs.** `v_min` over every run and every seed is
**−74.95 mV**, against the bound `min(E_inh, E_adapt)` = −75.0 mV, and `v_max < v_thresh` everywhere.
This is the acceptance criterion of M2d and the direct contrast with M2c:

| model | adaptation enters as | v min over the sweep | bounded by construction |
|---|---|---|---|
| M2 current synapses | — | −4,136 mV (`m2-report.md` §4.4) | no |
| M2b conductance synapses | — | −75.0 mV | yes |
| **M2c adaptation current** | `i_ext − w` | **−834 mV** (`m2c-report.md` §4.3) | **no** |
| **M2d adaptation conductance** | `G`, `v_inf` with `E_adapt` | **−74.95 mV** | **yes** |

**The RESPONSIVE region widens with `b_g`**, which is the other thing adaptation was supposed to buy.
M2b had two adjacent grid points RESPONSIVE for all three seeds (a factor 1.44 in `g`); here 26 of the
96 cells are, and at `b_g` ≥ 0.53 the RESPONSIVE band spans `g` = 5.4e-4 … 2.0e-3, a factor of 3.7 —
the network tolerates a nearly 4× range of synaptic gain without igniting. Adaptation is doing what a
stabilising mechanism should do, and unlike M2c it is not paying for it with unphysical voltages.

### 4.3 Noise sweep at the RESPONSIVE cells — the usable state

All 26 cells that were RESPONSIVE for all three seeds, × the M2b σ grid: no input, 2 s of noise, then 0.4 s
with the noise off (state kept). **312 runs, 179 s.** The usable state is pre-registered in
`sweep_conductance.py` as **four** conditions, all required (`_judge_usable`):

1. last-second rate in [0.1, 5] Hz per neuron
2. last-second / first-second spike count in [0.5, 2]
3. active fraction < 1 % in the final 200 ms after the noise stops
4. **active fraction in the last noise-on second ≤ 30 %** ← new in M2d

Conditions 1–3 are M2b's `STABLE_LOW_RATE` unchanged. Condition 4 exists because M2c satisfied 1–3 with
96–100 % of neurons firing, which CLAUDE.md §3.3 calls a parameter failure. **The active fraction is
tabulated for every run whether or not it passes.**

`Y` = usable, `a` = passes 1–3 but **fails only condition 4**, `·` = fails something else. σ ≤ 11.1 is
silent in every cell and is omitted:

| b_g | g | 18.0 | 29.1 | 47.1 | 76.3 | 123.6 | 200.0 |
|---|---|---|---|---|---|---|---|
| 0.001–0.081 (14 cells) | 2.78e-4 … 5.37e-4 | · | · | · | · | · | · |
| 0.152 | 5.37e-04 | · | a | a | · | · | · |
| 0.285 | 5.37e-04 | · | a | a | a | · | · |
| 0.285 | 7.46e-04 | · | a | · | · | · | · |
| 0.534 | 5.37e-04 | · | a | a | a | · | · |
| 0.534 | 7.46e-04 | · | a | a | a | · | · |
| 0.534 | 1.04e-03 | · | a | a | a | · | · |
| 0.534 | 1.44e-03 | · | a | · | · | · | · |
| **1.000** | **5.37e-04** | · | **Y** | a | a | a | · |
| 1.000 | 7.46e-04 | · | a | a | a | a | · |
| 1.000 | 1.04e-03 | · | · | a | a | a | · |
| 1.000 | 1.44e-03 | · | · | a | a | a | · |
| 1.000 | 2.00e-03 | · | a | a | a | · | · |

**Result: a usable state exists, in exactly 1 of 312 runs** — `b_g = 1.0`, `g = 5.365e-4`, σ = 29.13:
rate 0.452 → 0.444 Hz, ratio 0.98, tail 0.001 %, **active fraction 29.13 %**, `v_min` −91.2 mV.

Detail around it, with every active fraction shown:

| b_g | g | σ | rate first | rate last | **active last s** | ratio | tail | v min | verdict |
|---|---|---|---|---|---|---|---|---|---|
| 0.285 | 5.365e-04 | 29.13 | 0.753 | 0.733 | **32.74 %** | 0.97 | 0.270 % | −90.8 | fail: active_frac |
| 0.534 | 5.365e-04 | 29.13 | 0.547 | 0.508 | **30.48 %** | 0.93 | 0.160 % | −91.0 | fail: active_frac |
| **1.000** | **5.365e-04** | **29.13** | **0.452** | **0.444** | **29.13 %** | **0.98** | **0.001 %** | **−91.2** | **USABLE** |
| 1.000 | 5.365e-04 | 47.15 | 1.868 | 1.590 | **96.01 %** | 0.85 | 0.013 % | −105.8 | fail: active_frac |
| 1.000 | 7.455e-04 | 29.13 | 0.655 | 0.563 | **33.95 %** | 0.86 | 0.132 % | −91.0 | fail: active_frac |
| 1.000 | 2.000e-03 | 29.13 | 2.184 | 2.055 | **55.19 %** | 0.94 | 0.174 % | −89.6 | fail: active_frac |

**Condition 4 is what decides everything.** Of the 311 non-usable runs, **32 pass M2c's three conditions
and fail only the new one.** Their active fractions run from 30.5 % to 100 %, median 96.9 %, and **23 of the
32 are above the 90 % line CLAUDE.md §3.3 calls a parameter failure.** Without condition 4 this sweep would
have reported 33 usable states, 23 of them §3.3 failures — which is exactly what happened in M2c. The
condition was added before the sweep, on the strength of the M2c result, and it did the work it was added
for.

**Two honest caveats about the one hit**, both of which matter more than the hit itself:

1. **It sits on the grid boundary.** `b_g = 1.0` is the **top** point of the pre-registered `B_G_GRID`, so
   the rule's "smallest `b_g` with a usable state" returned the edge of the grid. The true minimum lies
   somewhere in (0.534, 1.0], and **nothing above 1.0 was tested**. Extending the grid now would be
   changing the protocol after seeing the result, which the brief forbids; it is a new pre-registered
   decision for the planning session.
2. **The margin is 0.87 points.** 29.13 % against a 30 % cap, and the next cell down (`b_g` = 0.534, same
   `g`, same σ) gives 30.48 % — it fails by 0.48 points. The usable/not boundary *is* the active-fraction
   criterion, and a different seed or a slightly different σ could plausibly flip either run. This is one
   run at one σ on one seed, not a robust operating point.

**Voltage under noise.** `v_min` over all 312 noise runs is **−227.3 mV**, at σ = 200. That is the noise
current going negative, not the adaptation: M2b measured −229.4 mV at the same σ with no adaptation at all
(`m2b-report.md` §6), and M2c measured −969 mV there. At σ ≤ 18 (sub-threshold) `v_min` is −81.5 mV in
every cell, identical to M2b. So the adaptation conductance contributes **nothing** to the excursion, at any
`b_g` — the excursion that remains is the pre-existing, documented behaviour of the current-based noise
model, and the acceptance check of §4.2 (no noise, `i_ext ≥ 0`) is the one that tests M2d's own bound.

## 5. `DEFAULT_ADAPT_G_B`

Rule, fixed in `sweep_conductance.py` before the sweeps: *`DEFAULT_ADAPT_G_B` = the smallest `b_g` with a
usable state; `DEFAULT_G_WITH_ADAPT` = the geometric mean of the `g` values usable at that `b_g`. If none
exists, both stay unset and "none" is the result.*

**`DEFAULT_ADAPT_G_B = 1.0`, `DEFAULT_G_WITH_ADAPT = 5.36539e-4`**
(`data-provenance/m2d-noise-sweep.json`), enforced against the recorded JSON by
`test_defaults_match_the_recorded_sweep_rule`. Only one `g` was usable at that `b_g`, so the geometric mean
is that value.

**Neither is a dataclass default.** `EngineParams.adapt_g_b` stays `0.0`, so M3/M4/live and every existing
call site are untouched until a caller opts in — the same arrangement as `DEFAULT_ADAPT_B`. Given §4.3's two
caveats (grid boundary, 0.87-point margin), **adopting this operating point is a planning decision and this
report does not recommend it as a settled default.** What the report does claim is the model fix in §4.2,
which is independent of whether any particular `b_g` is adopted.

## 6. Performance (`bench.py --synapse both --adapt-g`, full graph, dt = 0.1 ms)

`--adapt-g` adds a pass with the `ADAPT_G` branch compiled in, at `adapt_g_b = 1e-9` (g_a_ss ≈ 1e-7/ms
against a leak of 0.05/ms): per-step cost depends on whether the constexpr is set, not on the value, and
1e-9 leaves the protocol's forced activity exactly as it is — confirmed, spikes/step equals the forced
neuron count in every adapt row.

**The absolute numbers in this run are not valid and the < 100 µs target cannot be verified from it.**
While it ran, the GPU was at **1,515 MHz of a 3,105 MHz maximum** under a software power cap (40 W draw),
and window B's `flysim.live.server` was holding a live session at **31 % GPU utilisation**. The proof that
this is machine state and not M2d: the **current-model** rows, whose kernel M2d does not touch and which are
bit-identical to M2c's, are inflated by the same factor. Conductance with adaptation **off** — the identical
configuration measured on an idle machine in M2c — compares like this:

| backend | activity | recorder | M2c (idle machine) | this run (power-capped, live server active) | ratio |
|---|---|---|---|---|---|
| triton-cudagraph | 0.0 % | off | 41.1 | 115.2 | 2.80× |
| triton-cudagraph | 0.1 % | on | 48.2 | 123.7 | 2.57× |
| triton-cudagraph | 1.0 % | on | 61.0 | 154.1 | 2.53× |
| triton-eager | 5.0 % | on | 135.2 | 288.1 | 2.13× |

Across all 24 cells the inflation is 1.36–3.41×. So this run measures a slower machine, not a slower model.

**What the run does measure validly is the incremental cost**, because both arms ran back to back under the
same conditions:

| backend | activity | recorder | conductance | conductance + g_a | Δ |
|---|---|---|---|---|---|
| triton-eager | 0.0 % | off | 116.4 | 118.9 | +2.5 |
| triton-eager | 0.0 % | on | 115.1 | 117.6 | +2.5 |
| triton-cudagraph | 0.0 % | off | 115.2 | 115.0 | −0.2 |
| triton-cudagraph | 0.0 % | on | 118.1 | 117.3 | −0.8 |
| triton-eager | 0.1 % | off | 118.5 | 120.9 | +2.4 |
| triton-eager | 0.1 % | on | 123.4 | 125.1 | +1.7 |
| triton-cudagraph | 0.1 % | off | 117.9 | 119.8 | +1.9 |
| triton-cudagraph | 0.1 % | on | 123.7 | 126.2 | +2.5 |
| triton-eager | 1.0 % | off | 157.0 | 160.9 | +3.9 |
| triton-eager | 1.0 % | on | 154.5 | 156.2 | +1.7 |
| triton-cudagraph | 1.0 % | off | 142.7 | 146.4 | +3.7 |
| triton-cudagraph | 1.0 % | on | 154.1 | 156.4 | +2.3 |
| triton-eager | 5.0 % | off | 260.7 | 253.2 | −7.5 |
| triton-eager | 5.0 % | on | 288.1 | 292.0 | +3.9 |
| triton-cudagraph | 5.0 % | off | 245.3 | 249.1 | +3.8 |
| triton-cudagraph | 5.0 % | on | 290.6 | 291.1 | +0.5 |

**The adaptation conductance costs +1.5 µs/step on average across the 16 Triton cells** (range −7.5 to
+3.9; the negatives are run-to-run noise) — one extra load, multiply, two fused adds and a store per neuron,
inside a branch that is compiled out when `adapt_g_b = 0`. That is the same order as M2c's adaptation
current (+1.7 µs measured on an idle machine), which is expected: both add one per-neuron state.

**Conclusion on the target.** M2c measured the conductance kernel at 38–95 µs on an idle machine for every
cell except the known recorder-bound 5 %-with-recorder one. M2d adds +1.5 µs to that kernel. There is no
mechanism by which it could breach the 100 µs target that M2b and M2c met, but this session could not
demonstrate it directly because the machine was power-capped and shared. **A clean re-run
(`python -m flysim.engine.bench --synapse both --adapt-g`) on an idle, un-capped machine is the one
outstanding item of this milestone**; `data-provenance/m2d-bench.json` holds this run's raw numbers, clearly
labelled, and nothing was dropped.

Engine tensors: 347.5 MB per engine (M2c: 346.8 MB) — `g_a` adds one float32 array of N = 0.66 MB.

## 7. Tests (`tests/test_engine_adapt_g.py`, 24 cases)

**Parameters.** `adapt_g_b` = 0 and `adapt_tau_a` = 100 and `E_adapt` = −75 by default on both synapse
models; `adapt_g_b > 0` with `synapse="current"` raises; `adapt_b` and `adapt_g_b` together raise, while
`adapt_b` alone still works (M2c is kept as history); negative `adapt_g_b`, non-positive `adapt_tau_a` and
`E_adapt > v_reset` raise; `decay_a`, `avg_a`, `adapt_g`, `v_lower_bound` derived correctly and present in
`to_dict()`; `decay_a(dt=0.1)**10 == decay_a(dt=1)`; `DEFAULT_ADAPT_G_B` / `DEFAULT_G_WITH_ADAPT` match the
recorded sweep rule in `m2d-noise-sweep.json`, including that the usable-state definition there carries the
new 30 % condition.

**`adapt_g_b` = 0 is the pre-M2d engine.** Both conductance regression cases replay bit-identically
(§3); and `adapt_g_b = 1e-30` through the `ADAPT_G = True` branch, on both devices with noise on,
reproduces `adapt_g_b = 0` bit for bit — so the branch itself is a no-op at zero.

**The voltage bound — the point of M2d.** On the 3-neuron graph (both devices): the driven neuron fires,
the excited one fires only after it, the inhibited one never fires, `g_a` accumulates only where spikes
happened, and `v ≥ min(E_inh, E_adapt)` throughout. On the **real graph** at `g = 3e-3` (ignited) for
`b_g` ∈ {0.01, 0.5}, over 250 steps: `v` and `g_a` stay finite, `g_a` grows, and `v` never leaves
`[min(E_inh, E_adapt), v_thresh)`. Plus a direction check: `b_g` = 0.2 lowers the late-window spike count
against `b_g` = 0.

**Single neuron.** ISI grows and the steady-state rate falls below the unadapted one; the `b_g` = 0 control
is perfectly regular; `g_a` settles on its analytic periodic orbit `b_g/(1 − e^{−ISI/τ_a})`; `g_a` decays by
`exp(−dt/τ_a)` after the drive stops and `reset()` clears it. Monotonicity is asserted **up to one step of
quantisation**, because a steady state whose true ISI is not a whole number of steps alternates between the
neighbouring integers (here 37/38 for a true 37.25 ms) — M2c's equivalent test only passed a strict form by
the accident of landing on an integer.

**dt invariance.** Adapted ISIs at dt = 1 ms and dt = 0.1 ms agree interval by interval within one coarse
step, and so does the steady-state ISI.

**Backends and determinism.** Triton == torch spike lists and `g_a` on the synthetic graph and on the
3-neuron graph; bit-identical on the real graph in the ignited regime across repeats, a fresh engine and
CUDA-graph replay (with `g_a` equal too); noise determinism and seed dependence on both devices.

`pytest tests/`: **304 passed, 1 skipped.** The skip is window B's `test_live.py` budget check, which
skips itself when another CUDA process shares the GPU — the same contention documented in §6, detected
independently by that window's own guard.

## 8. Completion criteria (`docs/m2d-brief.md`)

- [x] regression bit-identical at the defaults — two conductance cases, the M2d one generated at `8d68698`
      before any edit (§3)
- [x] **v within `[min(E_a, E_i), v_thresh]` on every run** — 288/288 grid runs, `v_min` = −74.95 mV
      against a −75.0 bound, zero violations; M2c's −834 mV is gone (§4.2)
- [x] usable state's existence and its active fraction documented in a table — exists in 1 of 312 runs,
      29.13 % active, with the full matrix and the 32 near-misses tabulated (§4.3)
- [x] `DEFAULT_ADAPT_G_B` by the pre-registered rule, with its caveats (§5)
- [x] tests: 3-neuron signs and bounds, determinism, dt 1 vs 0.1, Triton == torch, current + `adapt_g_b`
      raises (§7)
- [x] `sweep_conductance.py` extended; `pytest tests/` passes
- [~] **bench re-run: incremental cost measured (+1.5 µs), absolute < 100 µs target NOT verified** because
      the GPU was power-capped at half clock and shared with a live session (§6). This is the one
      outstanding item; a re-run on an idle machine is one command.
- [x] one "M2d: …" commit, not pushed

## 9. Deviations from the brief

1. **Step-average `avg_a`** in `G` and `v_inf` instead of the brief's schematic bare `g_a`, matching how
   M2b treats `g_e`/`g_i` (whose bare form the brief also writes). Reduces to the brief's formula as
   dt → 0; the factor is 0.995 at dt = 1 ms with τ_a = 100 ms. Consistency choice, recorded in §1.
2. **`DEFAULT_ADAPT_G_B` and `DEFAULT_G_WITH_ADAPT` are recorded constants, not dataclass defaults** —
   `EngineParams.adapt_g_b` stays 0.0 so existing call sites are untouched, the same arrangement M2c used
   for `DEFAULT_ADAPT_B` (§5).
3. **The usable state sits at the top grid point**, so the rule's "smallest `b_g`" returned the grid edge.
   The grid was **not** extended after seeing this, because that would be changing the protocol after the
   fact; it is flagged for a new pre-registered decision instead (§4.3, caveat 1).
4. **The bench's absolute numbers are reported as invalid** rather than presented as results, with the
   evidence (the untouched current-model path is inflated by the same factor) and the raw JSON kept (§6).
   No cell was dropped or re-run selectively.
5. **`v_lower_bound`** added to `EngineParams` as a derived property, so the acceptance check has one
   definition shared by the sweeps and the tests rather than being re-derived in each place.

## 10. Notes for the planning session

- **The model defect is fixed, and that result is independent of any `b_g` choice.** Adaptation as a
  conductance cannot push `v` past `E_adapt`: 288 grid runs, `v_min` = −74.95 mV against a −75.0 bound.
  M2c's −834 mV was a consequence of putting the adaptation in the current slot, exactly as the M2c report
  predicted. If nothing else from M2d is adopted, this is the part worth keeping.
- **The usable state is real but thin.** One run in 312, 0.87 points inside the active-fraction condition,
  at the top of the grid and at a single σ. Before treating `(b_g, g) = (1.0, 5.365e-4)` as an operating
  point it would be worth a pre-registered follow-up: extend `B_G_GRID` above 1.0, refine between 0.534 and
  1.0, and repeat the winning cell across stimulus seeds to see whether the 29.13 % is stable.
- **Condition 4 earned its place.** 32 runs pass M2c's three conditions and fail only the active-fraction
  one; 23 of those are above CLAUDE.md §3.3's 90 % line. Without it this sweep would have reported 33
  usable states, most of them parameter failures.
- **Adaptation widens the usable gain range**, which is the other thing it was hoped to do: the all-seed
  RESPONSIVE band goes from M2b's ×1.44 in `g` to ×3.7 at `b_g` ≥ 0.53 (§4.2).
- **The remaining unbounded thing is the noise**, not the adaptation. `v` reaches −227 mV at σ = 200 in
  both M2b (no adaptation) and M2d, because `noise_sigma` injects a current that can be negative. If the
  planning session wants `v` bounded under noise too, that is a separate model-structure decision of the
  same family as M2b and M2d — not attempted here.
- `parameter-decisions.md` and `backlog.md` were not edited (planning-owned).
