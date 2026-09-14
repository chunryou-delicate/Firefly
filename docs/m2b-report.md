# M2b report — conductance-based synapses (engine extension)

M2 session, 2026-09-14. Spec: `docs/m2b-brief.md`. Raw results: `data-provenance/m2b-sweep.json`,
`m2b-noise-sweep.json`, `m2b-noise-sweep-supplementary.json`, `m2b-bench.json`, `m2-regression-spikes.npz`.
Hardware/software as in M2 (RTX 4090 Laptop GPU, torch 2.6.0+cu124, triton 3.2.0).

**This is a model-defect fix, not a change made to obtain a result curve.** The current-based model has no
reversal potentials: in its ignited state v reaches −4,000 mV (`docs/m2-report.md` §4.4) and no noise level
gives a low-rate state (`data-provenance/parameter-decisions.md`, 2026-09-13). Conductance-based synapses
bound v to [E_inh, E_exc] by construction. Every phase-B result is labelled as such.

## 1. Model (all values are ASSUMPTIONS; nothing below is in the connectome)

Selected with `EngineParams(synapse="conductance")`. Per-neuron state `v`, `g_e`, `g_i` (conductances in units
of 1/ms; the leak conductance is 1/τ_m = 0.05/ms). Per step of length dt:

```
g_e  ← g_e · exp(−dt/τ_e) + g · (excitatory contacts whose presynaptic neuron spiked at the previous step)
g_i  ← g_i · exp(−dt/τ_i) + g · (|inhibitory contacts| whose presynaptic neuron spiked at the previous step)
ḡ_e  = g_e · τ_e (1 − exp(−dt/τ_e)) / dt          step-average of the decaying conductance (→ g_e as dt → 0)
ḡ_i  = g_i · τ_i (1 − exp(−dt/τ_i)) / dt
G    = 1/τ_m + ḡ_e + ḡ_i
v_∞  = ((v_rest + i_ext)/τ_m + ḡ_e·E_exc + ḡ_i·E_inh) / G
v    ← v_∞ + (v − v_∞) · exp(−dt·G)               exponential Euler with the per-neuron rate G
spike / reset / refractory: unchanged (v ≥ v_thresh → v_reset, held for t_ref)
```

| item | value | note |
|---|---|---|
| E_exc, E_inh | 0, −75 mV | brief; reversal potentials (ASSUMPTION) |
| τ_e, τ_i | 5, 10 ms | brief (ASSUMPTION; GABA-A slower) |
| τ_m, t_ref, v_rest, v_reset, v_thresh | 20 ms, 2 ms, −65, −65, −50 mV | unchanged from M2 |
| g | **3.162e-4 /ms per contact per spike** (`DEFAULT_G_CONDUCTANCE`, §5) | one gain for both signs; no inhibitory scale factor |
| i_ext | enters as (v_rest + i_ext)/τ_m in v_∞ | with g_e = g_i = 0 a constant i_ext drives v to v_rest + i_ext, exactly as in the current model, so M3's `a_in` values keep their meaning |
| noise | current noise added to i_ext, unchanged | with noise, i_ext can be negative, so v can dip below E_inh (seen in the noise sweep: −83 to −229 mV at σ = 20–200; without noise v ≥ E_inh always) |
| synaptic delay | one step | unchanged (ASSUMPTION) |
| `v_floor` | kept, default None | **unused with conductance synapses** (E_inh already bounds v) |

**Integration choice (recorded per brief):** exponential Euler as specified, but the conductance entering G and v_∞
is the exact step-average of the exponentially decaying g_x, not its start-of-step value. Reason: with the
start-of-step value the dt = 1 ms update overdrives by ~10 % relative to dt = 0.1 ms (τ_e = 5 ms) and the excited
neuron of the 3-neuron test fired 1.4 ms earlier at dt = 1 ms, failing the brief's ±1 ms requirement; with the
step-average the unitary EPSP is 0.0647 mV at both dt (peak at 9.0 vs 9.2 ms) and spike times agree within one
coarse step (`test_conductance_dt_invariance_spike_times`). All coefficients remain functions of exp(−dt/τ).

**Unitary PSPs at the default g (one contact, from rest):** EPSP +0.065 mV (peak 9 ms), IPSP −0.016 mV (trough
14 ms). The 4:1 asymmetry is the reversal-potential geometry (driving force 65 mV for excitation vs 10 mV for
inhibition at rest), not a parameter: inhibition is weak at rest and grows with depolarisation (shunting). For
comparison the current model at M3's g = 0.336 gives ±0.053 mV. From rest, 300 simultaneous excitatory contacts
reach threshold at the default g (200 do not).

## 2. Implementation

- `params.py`: `synapse`, `E_exc`, `E_inh`, `tau_e`, `tau_i` fields; derived `decay_e/i`, `avg_e/i`, `inv_tau_m`; all in
  `to_dict()`. `g=None` resolves to `DEFAULT_G` (current) or `DEFAULT_G_CONDUCTANCE` (conductance).
  `synapse="current"` is the default, so every existing call site is unchanged.
- `lif.py`: **two accumulators for the conductance model** (`acc_e`, `acc_i`, int32) filled by a separate kernel
  `_propagate_split_kernel` (two masked `tl.atomic_add`s per edge chunk, one of which is active per edge). The
  current model keeps its original `_propagate_kernel` and single `acc` untouched — chosen over the
  "acc_e − acc_i" variant because it makes the regression trivially exact and costs nothing (§3). `_lif_kernel`
  branches on a `SYNAPSE` constexpr; the current branch is the M2 code. The torch backend implements the same
  equations (`propagate_reference_split`, exponential Euler with `torch.exp`).
- Determinism basis unchanged: the only cross-thread reductions are int32 atomic adds; `tl.exp` is deterministic
  for a fixed kernel. CUDA-graph snapshot/restore extended to the new state.
- Memory: +4 arrays of N floats/ints (2.6 MB). Recorder untouched (backlog items not done, per brief).

## 3. Regression: `synapse="current"` reproduces M2 bit for bit

`tests/regression_cases.py --write` generated `data-provenance/m2-regression-spikes.npz` with the **unmodified**
M2 engine (tree at 0b2c19b, `lif.py` identical to 8bae9d3) before any M2b edit. Six cases; each replayed by
`test_current_model_regression_bit_identical` and compared with `np.array_equal` on (t, idx) and on the final v:

| case | graph | params | steps | spikes | result |
|---|---|---|---|---|---|
| real_g1 | full | g = 1.0 (ignited regime) | 100 | 218,067 | identical |
| real_g1_cudagraph | full | g = 1.0, CUDA graph | 100 | 218,067 | identical |
| real_g0336_noise20 | full | g = 0.336, σ = 20 | 300 | 647 | identical |
| real_dt01_g0336 | full | dt = 0.1, g = 0.336 | 500 | 309 | identical |
| synth_cpu | synthetic 5k/200k | g = 0.5, σ = 1, CPU torch backend | 150 | 820 | identical |
| synth_cuda_torch | synthetic | g = 0.5, σ = 1, GPU torch backend | 150 | 822 | identical |

All six pass. The M2 test file (`tests/test_engine.py`, 22 cases) also passes unchanged.

## 4. Performance (`bench.py --synapse both`, full graph, dt = 0.1 ms, method as in M2)

| backend | activity / step | recorder | current, M2 run (μs) | current, re-run today (μs) | conductance (μs) | < 100 μs (conductance) |
|---|---|---|---|---|---|---|
| triton-eager | 0.0% (0) | off | 53.9 | 63.6 | 64.5 | yes |
| triton-eager | 0.0% (0) | on | 55.2 | 64.3 | 66.2 | yes |
| triton-eager | 0.1% (165) | off | 53.9 | 62.7 | 64.3 | yes |
| triton-eager | 0.1% (165) | on | 58.7 | 66.8 | 66.8 | yes |
| triton-eager | 1.0% (1,651) | off | 53.7 | 63.2 | 63.7 | yes |
| triton-eager | 1.0% (1,651) | on | 65.3 | 73.3 | 71.3 | yes |
| triton-eager | 5.0% (8,256) | off | 80.6 | 82.6 | 85.3 | yes |
| triton-eager | 5.0% (8,256) | on | 113.9 | 118.5 | 121.1 | **NO** |
| triton-cudagraph | 0.0% (0) | off | 35.7 | 36.4 | 38.6 | yes |
| triton-cudagraph | 0.0% (0) | on | 35.9 | 37.4 | 39.3 | yes |
| triton-cudagraph | 0.1% (165) | off | 38.1 | 38.9 | 40.8 | yes |
| triton-cudagraph | 0.1% (165) | on | 40.2 | 41.8 | 43.7 | yes |
| triton-cudagraph | 1.0% (1,651) | off | 44.0 | 45.2 | 46.9 | yes |
| triton-cudagraph | 1.0% (1,651) | on | 51.5 | 53.1 | 55.1 | yes |
| triton-cudagraph | 5.0% (8,256) | off | 81.8 | 83.5 | 86.7 | yes |
| triton-cudagraph | 5.0% (8,256) | on | 104.6 | 119.6 | 121.6 | **NO** |
| torch-gpu | 0.0% (0) | off | 130.7 | 134.4 | 181.1 | **NO** |
| torch-gpu | 0.0% (0) | on | 153.8 | 164.8 | 209.8 | **NO** |
| torch-gpu | 0.1% (165) | off | 301.0 | 309.8 | 379.7 | **NO** |
| torch-gpu | 0.1% (165) | on | 367.8 | 379.6 | 452.5 | **NO** |
| torch-gpu | 1.0% (1,651) | off | 304.5 | 315.6 | 387.5 | **NO** |
| torch-gpu | 1.0% (1,651) | on | 374.8 | 391.3 | 463.3 | **NO** |
| torch-gpu | 5.0% (8,256) | off | 363.5 | 380.1 | 469.0 | **NO** |
| torch-gpu | 5.0% (8,256) | on | 439.3 | 462.8 | 560.3 | **NO** |

The conductance kernel costs 1–3 μs more per step than the current kernel (one `tl.exp` and two extra loads/stores
per neuron; the split propagate kernel is within noise of the single one). Today's re-run of the current model is
itself 5–10 μs slower than the M2 numbers at low activity (machine state), so the comparison column is the re-run.
The one cell above 100 μs is the same as in M2 — **5 % of all neurons spiking every 0.1 ms step with the recorder
on** (119–122 μs; kernel-only 83–87 μs) — caused by the recorder flush (sort + pageable D2H), not by the model;
improvement options are in `docs/backlog.md` and were out of scope here. In the ignited conductance state the
network produces 3.7–14 k spikes per ms (§5), i.e. 0.2–0.85 % per 0.1 ms step, where the cost is 43–75 μs with
recording. Engine tensors: 346 MB per engine (M2: 343.5 MB).

## 5. g sweep (`sweep.py --synapse conductance`; criteria fixed in `sweep_conductance.py` before running)

Protocol: full graph, dt = 1 ms, 100 random neurons (seeds 0/1/2, the M2 stimulus sets) driven with i_ext = 30
for 100 ms, then 400 ms without input. Grid: 20 log-spaced g in [1e-5, 1e-2]. Grid reason: a conductance g gives a
current ≈ g·(E_exc − v) ≈ 65·g mV/ms near rest, so the current model's g_cur maps to g ≈ g_cur/((E_exc − v_rest)·τ_m)
= g_cur/1300; M2's ignition threshold g_cur ≈ 0.55 maps to ≈ 4e-4, and the grid spans two decades either side. A
4-point prototype (1e-5, 1e-4: SILENT; 1e-3, 1e-2: IGNITED; seed 0) confirmed the transition lies inside the grid
before the sweep was run.

Verdict per seed: **SILENT** = no downstream spike (stimulated neurons excluded) during the stimulus;
**IGNITED** = ≥ 1 % of neurons spike in the 200–400 ms post-offset window; **RESPONSIVE** = downstream spikes > 0 and not
IGNITED. Runtime: 60 runs in 6 s.

| g (1/ms) | downstream spikes during stim (s0 / s1 / s2) | post 0-50 ms (s0/s1/s2) | post 200-400 ms (s0/s1/s2) | late active fraction (s0/s1/s2) | rate of active neurons, Hz (s0/s1/s2) | v min (all seeds) | verdicts (s0 / s1 / s2) |
|---|---|---|---|---|---|---|---|
| 1.000e-05 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0.00% / 0.00% / 0.00% | 0 / 0 / 0 | -65.9 | SILENT / SILENT / SILENT |
| 1.438e-05 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0.00% / 0.00% / 0.00% | 0 / 0 / 0 | -66.3 | SILENT / SILENT / SILENT |
| 2.069e-05 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0.00% / 0.00% / 0.00% | 0 / 0 / 0 | -66.7 | SILENT / SILENT / SILENT |
| 2.976e-05 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0.00% / 0.00% / 0.00% | 0 / 0 / 0 | -67.3 | SILENT / SILENT / SILENT |
| 4.281e-05 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0.00% / 0.00% / 0.00% | 0 / 0 / 0 | -68.1 | SILENT / SILENT / SILENT |
| 6.158e-05 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0.00% / 0.00% / 0.00% | 0 / 0 / 0 | -68.9 | SILENT / SILENT / SILENT |
| 8.859e-05 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0.00% / 0.00% / 0.00% | 0 / 0 / 0 | -69.8 | SILENT / SILENT / SILENT |
| 1.274e-04 | 0 / 0 / 1 | 0 / 0 / 0 | 0 / 0 / 0 | 0.00% / 0.00% / 0.00% | 0 / 0 / 0 | -70.8 | SILENT / SILENT / RESPONSIVE |
| 1.833e-04 | 3 / 0 / 11 | 3 / 0 / 0 | 0 / 0 / 0 | 0.00% / 0.00% / 0.00% | 0 / 0 / 0 | -71.7 | RESPONSIVE / SILENT / RESPONSIVE |
| 2.637e-04 | 11 / 7 / 21 | 4 / 0 / 0 | 0 / 0 / 0 | 0.00% / 0.00% / 0.00% | 0 / 0 / 0 | -72.9 | RESPONSIVE / RESPONSIVE / RESPONSIVE |
| 3.793e-04 | 31 / 75 / 53 | 0 / 2 / 0 | 0 / 0 / 0 | 0.00% / 0.00% / 0.00% | 0 / 0 / 0 | -73.9 | RESPONSIVE / RESPONSIVE / RESPONSIVE |
| 5.456e-04 | 49 / 172,657 / 192,726 | 2 / 187,711 / 184,736 | 0 / 733,854 / 761,867 | 0.00% / 13.94% / 15.21% | 0 / 159 / 152 | -74.7 | RESPONSIVE / IGNITED / IGNITED |
| 7.848e-04 | 328,265 / 331,432 / 357,811 | 232,443 / 234,965 / 230,732 | 919,299 / 922,632 / 928,050 | 17.00% / 16.61% / 17.76% | 164 / 168 / 158 | -74.7 | IGNITED / IGNITED / IGNITED |
| 1.129e-03 | 447,798 / 491,042 / 492,544 | 275,482 / 283,600 / 284,811 | 1,142,671 / 1,172,078 / 1,136,003 | 20.68% / 19.82% / 19.88% | 167 / 179 / 173 | -74.8 | IGNITED / IGNITED / IGNITED |
| 1.624e-03 | 638,200 / 622,354 / 622,990 | 344,707 / 355,748 / 353,941 | 1,378,849 / 1,349,207 / 1,415,148 | 23.51% / 23.22% / 23.76% | 178 / 176 / 180 | -74.9 | IGNITED / IGNITED / IGNITED |
| 2.336e-03 | 785,020 / 773,195 / 772,056 | 451,731 / 455,406 / 456,014 | 1,794,223 / 1,813,897 / 1,808,512 | 25.96% / 25.87% / 26.11% | 209 / 212 / 210 | -74.9 | IGNITED / IGNITED / IGNITED |
| 3.360e-03 | 918,445 / 910,400 / 909,510 | 508,162 / 510,507 / 500,190 | 2,070,763 / 2,018,115 / 2,041,421 | 28.86% / 27.78% / 28.05% | 217 / 220 / 220 | -74.9 | IGNITED / IGNITED / IGNITED |
| 4.833e-03 | 1,021,920 / 1,031,479 / 1,016,690 | 555,299 / 558,391 / 548,004 | 2,247,024 / 2,282,636 / 2,304,899 | 30.92% / 31.61% / 30.89% | 220 / 219 / 226 | -74.9 | IGNITED / IGNITED / IGNITED |
| 6.952e-03 | 1,152,785 / 1,157,966 / 1,116,465 | 642,616 / 640,697 / 631,131 | 2,609,281 / 2,582,510 / 2,541,939 | 32.71% / 33.76% / 32.78% | 242 / 232 / 235 | -75.0 | IGNITED / IGNITED / IGNITED |
| 1.000e-02 | 1,243,320 / 1,263,873 / 1,242,582 | 713,615 / 702,238 / 681,292 | 2,771,369 / 2,762,220 / 2,787,551 | 34.23% / 34.78% / 34.41% | 245 / 240 / 245 | -75.0 | IGNITED / IGNITED / IGNITED |

Full per-run detail (post 50–200 ms counts, stimulated-neuron spikes, v max, 50 ms time courses) is in
`m2b-sweep.json`.

**Result.** RESPONSIVE for all three seeds: g ∈ {2.637e-4, 3.793e-4} (contiguous grid points) → interval
**[2.637e-04, 3.793e-04]**. Below 1.274e-4 every seed is SILENT; 1.274e-4 and 1.833e-4 are RESPONSIVE for some seeds
and SILENT for others; 5.456e-4 is RESPONSIVE for seed 0 and IGNITED for seeds 1 and 2; from 7.848e-4 upward every
seed ignites. The RESPONSIVE regime is input-driven and weak: 7–75 downstream spikes during the 100 ms stimulus
(the 100 stimulated neurons fire ~600 times), 0–4 spikes in the 50 ms after offset, silence afterwards.

**Ignited state, now bounded.** Where ignition occurs, 14–35 % of neurons are active in the late window at 150–245 Hz
each (refractory ceiling 333 Hz), and **v stays within [−75.0, −50.0] mV** at every g and seed (M2: −4,136 mV at
its default g, −94,568 mV at g = 10). The transition remains first-order: at 5.456e-4 the downstream count jumps
from 49 (seed 0) to 172,657 (seed 1) with nothing in between on this grid.

### 5.1 Default (rule stated in the brief and in code before the run)

`DEFAULT_G_CONDUCTANCE` = geometric mean of the RESPONSIVE interval common to all seeds = √(2.637e-4 × 3.793e-4) =
**3.162e-4**. Enforced by `test_default_g_conductance_matches_recorded_sweep_rule` against `m2b-sweep.json`.
Noise default stays 0.

## 6. Noise sweep at g = 3.162e-4 (`sweep.py --synapse conductance --noise`)

No input; `noise_sigma` on 12 log-spaced points in [1, 200] for 2 s, then 0.4 s with the noise switched off (state
kept). Verdicts fixed before running: **IGNITED** = ≥ 1 % of neurons spike in the final 200 ms after the noise stops
(self-sustained, same criterion as §5); **STABLE_LOW_RATE** = not IGNITED, mean rate over the last noise-on second
in [0.1, 5] Hz per neuron, and last-second / first-second spike count in [0.5, 2].

| sigma | rate first s (Hz, all N) | rate last s (Hz, all N) | active in last s | last/first | active in final 200 ms after noise off | v min / max (mV) | verdict |
|---|---|---|---|---|---|---|---|
| 1.00 | 0.000 | 0.000 | 0.000% | 0.00 | 0.000% | -65.9 / -64.1 | silent / sub-threshold |
| 1.62 | 0.000 | 0.000 | 0.000% | 0.00 | 0.000% | -66.5 / -63.5 | silent / sub-threshold |
| 2.62 | 0.000 | 0.000 | 0.000% | 0.00 | 0.000% | -67.4 / -62.6 | silent / sub-threshold |
| 4.24 | 0.000 | 0.000 | 0.000% | 0.00 | 0.000% | -68.9 / -61.1 | silent / sub-threshold |
| 6.87 | 0.000 | 0.000 | 0.000% | 0.00 | 0.000% | -71.3 / -58.7 | silent / sub-threshold |
| 11.12 | 0.000 | 0.000 | 0.000% | 0.00 | 0.000% | -75.2 / -54.8 | silent / sub-threshold |
| 17.99 | 0.000 | 0.000 | 0.004% | 3.50 | 0.000% | -81.5 / -50.0 | silent / sub-threshold |
| 29.13 | 14.212 | 15.490 | 32.434% | 1.09 | 9.327% | -90.7 / -50.0 | IGNITED |
| 47.15 | 19.966 | 20.230 | 83.823% | 1.01 | 9.456% | -104.8 / -50.0 | IGNITED |
| 76.32 | 32.010 | 31.836 | 97.745% | 0.99 | 9.670% | -127.9 / -50.0 | IGNITED |
| 123.55 | 57.952 | 57.822 | 99.660% | 1.00 | 11.701% | -166.5 / -50.0 | IGNITED |
| 200.00 | 89.535 | 89.424 | 99.948% | 1.00 | 13.726% | -229.4 / -50.0 | IGNITED |

Supplementary finer grid between the pre-defined points 17.99 and 29.13 (same protocol and criteria, run after
the primary table, labelled as supplementary):

| sigma | rate first s (Hz, all N) | rate last s (Hz, all N) | active in last s | last/first | active in final 200 ms after noise off | v min / max (mV) | verdict |
|---|---|---|---|---|---|---|---|
| 20.00 | 0.001 | 0.000 | 0.045% | 0.76 | 0.000% | -83.3 / -50.0 | silent / sub-threshold |
| 22.00 | 0.021 | 13.973 | 19.643% | 659.79 | 9.250% | -84.7 / -50.0 | IGNITED |
| 24.00 | 13.065 | 14.650 | 21.124% | 1.12 | 9.488% | -86.4 / -50.0 | IGNITED |
| 26.00 | 13.321 | 15.087 | 24.708% | 1.13 | 9.332% | -88.1 / -50.0 | IGNITED |
| 28.00 | 14.054 | 15.364 | 29.435% | 1.09 | 9.393% | -89.8 / -50.0 | IGNITED |

**Result: no fluctuation-driven low-rate state exists at this g.** σ ≤ 20 leaves the network essentially silent
(≤ 75 spikes network-wide in the last second at σ = 20, 0.045 % of neurons); σ = 22 ignites after ~1 s
(0.02 → 14 Hz); σ ≥ 24 is ignited from the first second (13–90 Hz per neuron over all N, 21–100 % active), and every
ignited run stays ignited after the noise is removed (9–14 % active in the noise-off tail). The transition point
(σ ≈ 20–22) is the same as the current model's at g = 0.336 (`parameter-decisions.md`), so reversal potentials fix
the unbounded hyperpolarisation but do not create an asynchronous low-rate state under this protocol. **This is
reported as the result; no parameter was adjusted to change it.**

## 7. Tests (`pytest tests/`: 76 passed = 53 existing + 23 new)

`tests/test_engine_conductance.py`: 6 regression cases + reference metadata; params fields/validation; split
propagate == `index_add_` reference (4 activity levels, bit-exact, column-sum check at 100 %); 3-neuron signs on
CPU and GPU (C hyperpolarised but never below E_inh; B fires after A; conductances have the right sign);
v ∈ [E_inh, v_thresh) and finite on the full graph at every step (g = 1e-4 and 1e-2); bit-identical determinism on
the full graph across repeats, a fresh engine and CUDA-graph replay (g = 3e-3, ignited regime, > 50 k spikes);
noise determinism / seed dependence on both backends; dt = 1 vs 0.1 ms spike times within 1 ms; Triton == torch
spike lists on the 3-neuron graph (v within 1e-3 mV) and on the full graph for a short run; default-g rule.

## 8. Acceptance checklist (`docs/m2b-brief.md`)

- [x] default `synapse="current"`: M2 results bit-identical (6 cases, §3)
- [x] conductance tests pass; determinism bit-identical (§7)
- [x] full-graph bench < 100 μs/step except the 5 %-with-recorder cell (119–122 μs, same cause as M2; §4)
- [x] RESPONSIVE interval exists and is documented for 3 seeds: [2.637e-4, 3.793e-4] (§5)
- [x] noise sweep documented: no low-rate state (§6)
- [x] `pytest tests/` 76 passed

## 9. Deviations from the brief

1. **Step-average conductance in G and v_∞** instead of the start-of-step value (§1). Needed to meet the brief's own
   ±1 ms dt-invariance requirement; reduces to the brief's formula as dt → 0.
2. **Noise-sweep ignition criterion**: the brief asks for "점화 여부" without defining it for a noise-driven run, where
   sustained activity is expected by construction. Defined (before running) as self-sustained activity after the
   noise is switched off, using the g-sweep's 1 % criterion; the 0.4 s noise-off tail was added to the protocol for
   this purpose. Rates and stability are measured on the noise-on 2 s as the brief specifies.
3. **Supplementary noise grid** (§6) run after the primary grid, reported separately; criteria unchanged.
4. **Separate split-propagate kernel** rather than reusing one kernel with two accumulators for both models (allowed
   by the brief; chosen for the exact regression).
5. `sweep_conductance.py` is a new module (the brief listed `sweep.py` extension only); `sweep.py --synapse
   conductance [--noise]` dispatches to it and the M2 sweep code path is untouched.
6. The `v_min` in the noise tables goes below E_inh because current noise can be negative (§1); this is the
   unchanged noise model, not a bound violation of the synapse model.

## 10. Notes for the planning session

- The RESPONSIVE window is narrow (two grid points, ×1.44 apart) and seed-dependent at both edges; JO input
  (hundreds of correlated neurons) may ignite at lower g than 100 random neurons. Any M3/M4 phase-B run should keep
  the per-run ignition check.
- In the RESPONSIVE regime downstream activity is ~1–2 orders of magnitude below the stimulated neurons' own
  spikes, i.e. propagation beyond hop 1 without background activity is unlikely to differ much from phase A. The
  model now has bounded v and bounded ignited rates, but no low-rate asynchronous state was found; if such a state
  is required, it would need a mechanism outside this brief (e.g. heterogeneous drive, adaptation), which is a
  planning decision.
- `parameter-decisions.md` and `backlog.md` were not edited (planning-owned).
