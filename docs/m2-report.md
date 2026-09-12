# M2 report — LIF engine (torch + Triton, GPU resident)

Implementation session, 2026-09-13. Spec: `docs/m2-brief.md`. Raw results: `data-provenance/m2-bench.json`,
`data-provenance/m2-sweep.json`, `data-provenance/m2-sweep-supplementary.json`.
Hardware/software: RTX 4090 Laptop GPU (16 GB), driver 580.173.02, torch 2.6.0+cu124, triton 3.2.0, CUDA 12.4.

## 1. Model

Current-based leaky integrate-and-fire with an exponential synaptic current, one state pair (`v`, `i_syn`) per
neuron, float32, GPU resident. Per step of length `dt` (with `a_m = exp(-dt/τ_m)`, `a_s = exp(-dt/τ_syn)`):

```
i_syn ← a_s · i_syn + g · Σ_{pre spiked at previous step} w_signed(pre → post)     (w = signed contact count, int32)
v     ← v_rest + (v − v_rest) · a_m + (1 − a_m) · i_ext + c_s · i_syn           (refractory neurons: v ← v_reset)
spike ← v ≥ v_thresh   →   v ← v_reset, refractory for t_ref
c_s   = τ_syn / (τ_m − τ_syn) · (a_m − a_s)
```

**Integration scheme (choice recorded per brief):** exact one-step integration of
`τ_m dv/dt = v_rest − v + i_ext + i_syn(t)` with `i_syn` decaying exponentially inside the step and `i_ext`
constant inside the step (Rotter & Diesmann-style exact integration), not the forward-Euler form the brief's
equations are written in. Reason: with the prototype's exp-Euler update (`(1 − a_m)·(i_syn + i_ext)`), a
synaptic kick deposits ~7 % more depolarisation at dt = 1 ms than at dt = 0.1 ms, and on the 3-neuron test the
excited neuron fired 2 ms earlier at dt = 1 ms. With the exact coefficient the unitary PSP is 0.1574 mV
(dt = 1) vs 0.1575 mV (dt = 0.1) per contact at g = 1, and spike times agree within one coarse step
(`test_dt_invariance_spike_times`). All dt-dependent coefficients are functions of `exp(−dt/τ)` only.

Semantics fixed by the implementation:

- **Synaptic delay = one step (dt).** Spikes of step t are accumulated at step t+1. (ASSUMPTION; brief.)
- **Refractory:** a neuron that spikes at step t is held at `v_reset` and cannot spike for `ref_steps = round(t_ref/dt)`
  further steps; its `i_syn` keeps integrating. Minimum ISI = t_ref + dt.
- **Threshold at step boundaries**; a neuron crossing threshold is reset in the same step.
- **Noise (optional, default off):** Gaussian current `noise_sigma · sqrt(1 ms / dt) · ξ` added to `i_ext` each step,
  drawn from a private `torch.Generator` seeded in `reset(seed)` (ASSUMPTION on the sqrt scaling: keeps the
  noise-driven voltage variance dt-independent).
- **`v_floor` (optional, default None):** hard lower clamp on v. Off by default because the brief's model has no
  floor; see §4.4 for why it exists.

### Assumptions (nothing below is in the connectome)

| item | value | status |
|---|---|---|
| τ_m, τ_syn, t_ref | 20 ms, 5 ms, 2 ms | ASSUMPTION (brief defaults) |
| v_rest, v_reset, v_thresh | −65, −65, −50 mV | ASSUMPTION (brief defaults) |
| R | 1 (absorbed in g) | ASSUMPTION |
| g | 0.886 mV per contact per spike in `i_syn` units (§4.5) | ASSUMPTION, from sweep |
| synapse model | current-based, exponential, no reversal potential, no saturation | ASSUMPTION |
| synaptic delay | 1 step | ASSUMPTION |
| edge sign | presynaptic consensus_nt (M1, `data-provenance/edge-signs.md`) | inherited assumption |
| weight | signed synaptic contact count, linear | ASSUMPTION |
| no plasticity, no neuromodulation, no noise by default | | ASSUMPTION |

Unitary PSP at the default g = 0.886: 0.140 mV per contact; median edge (2 contacts) 0.28 mV; mean edge (4.85)
0.68 mV; the strongest edge (2,591 contacts) 362 mV, i.e. one spike is suprathreshold; 107 simultaneous contacts
reach threshold from rest.

## 2. Implementation

```
flysim/engine/params.py    EngineParams (frozen dataclass; decay_m/decay_syn/c_syn/ref_steps/noise_scale; to_dict)
flysim/engine/lif.py       Triton kernels _propagate_kernel / _lif_kernel, propagate_reference (torch index_add_),
                           CSRGraph (synthetic graphs), LIFEngine
flysim/engine/recorder.py  SpikeRecorder (device append buffers, chunked flush, RecorderOverflow)
flysim/engine/bench.py     python -m flysim.engine.bench
flysim/engine/sweep.py     python -m flysim.engine.sweep  (criteria as module constants)
tests/test_engine.py       15 test functions, 22 cases incl. parametrisation
```

- **Step = two kernel launches, no host sync.** `_propagate_kernel`: grid ⌈N/32⌉, each program scans 32
  presynaptic neurons and, for those with `s_prev ≠ 0`, streams the forward-CSR row in chunks of 128 edges with
  `tl.atomic_add` (int32) into `acc[post]`. `_lif_kernel`: BLOCK = 1024, fused `i_syn`/`v`/refractory/threshold/
  reset, `acc ← 0`, spike vector (int8) and spike logging (`slot = atomic_add(counter, 1)`, masked store when
  `slot ≥ cap`, so overflow is detectable). The step index `t` is a device tensor (`t.add_(1)` after the kernel).
- **Graph arrays on the GPU:** `indptr` int32 (M = 25.56 M < 2³¹), `indices` int32, `weight` int32 (exact: M1 weights
  are integer contact counts; asserted at load). 343.5 MB per engine including the 16 M-entry recorder (128 MB).
- **Determinism basis:** the only cross-thread reduction is integer atomic addition (order-independent). The spike
  log's slot order is nondeterministic, so `flush` sorts by the key `t·N + idx` on the device before the copy.
  Verified bit-identical across repeated runs, fresh engine instances, and eager vs CUDA-graph replay in the
  high-activity regime (g = 1.0, 100 steps, ≈ 1.6 M spikes).
- **Recorder:** GPU (t, idx) int32 buffers, `cap = 2²⁴` by default, flushed every `chunk_steps = 100` steps (one
  `int(counter)` sync per flush). Sort and (t, idx) decode happen on the device; the host only concatenates int32
  arrays. `RecorderOverflow` is raised if a flush interval produced more than `cap` spikes — never truncated silently.
  At the default settings the cap holds 100 steps × 165,122 neurons all firing (16.5 M < 16.8 M).
- **Input convention:** `run(n_steps, i_ext_fn, record)` calls `i_ext_fn(t_step, buf)` before each step; the
  callback writes into the persistent device buffer `buf` (float32, length N) and returns None (a returned tensor
  is `copy_`-ed, slower). `step(i_ext)` copies a tensor or zeroes the buffer for `None`. This is the convention M3's
  sensory adapters implement.
- **CUDA graph (`use_cuda_graph=True`, optional):** the two kernels + `t.add_` are captured once per `record`
  flag; warm-up/capture side effects are snapshot/restored so the capture is invisible to the simulation; noise
  generation, input copies and flushes stay outside the graph. Changing `engine.params` invalidates captured graphs.
- **Backends:** `triton` (default on CUDA) and `torch` (`nonzero` + `index_add_` on int32 `acc`; the CPU
  fallback — with `warnings.warn` when CUDA was requested but is absent — and the reference the kernel is tested
  against). GPU↔CPU bit-exactness is not claimed (float op fusion differs); on the 3-neuron graph the two backends
  give identical spike lists and v within 1e-3 mV.

## 3. Performance (`bench.py`, full graph: 165,122 neurons / 25,563,197 edges)

Method: `EngineParams(dt=0.1, g=0, t_ref=0)`; a fixed random subset of p·N neurons receives a 10⁴ current so the
LIF kernel emits exactly p·N real spikes every step (recorder sees them, propagate walks their rows next step;
g = 0 stops spreading, which does not affect kernel cost). Warm-up 100 steps excluded; `torch.cuda.synchronize()`
+ `time.perf_counter` around `LIFEngine.run` for 10,000 steps (1,000 for the torch backend), so Python loop
overhead, the input callback and recorder flushes are all included. Per-step cost is independent of dt.

| backend | forced activity / step | recorder | μs/step | spikes/step | transient peak MiB | < 100 μs |
|---|---|---|---|---|---|---|
| triton-eager | 0 % | off | 53.9 | 0 | 1 | yes |
| triton-eager | 0 % | on | 55.2 | 0 | 1 | yes |
| triton-cudagraph | 0 % | off | 35.7 | 0 | 3 | yes |
| triton-cudagraph | 0 % | on | 35.9 | 0 | 3 | yes |
| torch-gpu (reference) | 0 % | off | 130.7 | 0 | 3 | no |
| torch-gpu (reference) | 0 % | on | 153.8 | 0 | 3 | no |
| triton-eager | 0.1 % (165) | off | 53.9 | 165 | 1 | yes |
| triton-eager | 0.1 % (165) | on | 58.7 | 165 | 1 | yes |
| triton-cudagraph | 0.1 % (165) | off | 38.1 | 165 | 1 | yes |
| triton-cudagraph | 0.1 % (165) | on | 40.2 | 165 | 1 | yes |
| torch-gpu (reference) | 0.1 % (165) | off | 301.0 | 165 | 3 | no |
| torch-gpu (reference) | 0.1 % (165) | on | 367.8 | 165 | 3 | no |
| triton-eager | 1 % (1,651) | off | 53.7 | 1,651 | 1 | yes |
| triton-eager | 1 % (1,651) | on | 65.3 | 1,651 | 8 | yes |
| triton-cudagraph | 1 % (1,651) | off | 44.0 | 1,651 | 1 | yes |
| triton-cudagraph | 1 % (1,651) | on | 51.5 | 1,651 | 8 | yes |
| torch-gpu (reference) | 1 % (1,651) | off | 304.5 | 1,651 | 8 | no |
| torch-gpu (reference) | 1 % (1,651) | on | 374.8 | 1,651 | 8 | no |
| triton-eager | 5 % (8,256) | off | 80.6 | 8,256 | 1 | yes |
| triton-eager | 5 % (8,256) | on | **113.9** | 8,256 | 40 | **no** |
| triton-cudagraph | 5 % (8,256) | off | 81.8 | 8,256 | 1 | yes |
| triton-cudagraph | 5 % (8,256) | on | **104.6** | 8,256 | 40 | **no** |
| torch-gpu (reference) | 5 % (8,256) | off | 363.5 | 8,256 | 35 | no |
| torch-gpu (reference) | 5 % (8,256) | on | 439.3 | 8,256 | 40 | no |

GPU memory: 343.5 MB of engine tensors per engine (`engine_bytes` = 343,511,854: graph 205.2 MB + state 4.1 MB
+ recorder 134.2 MB); the benchmark keeps three engines resident (3 × 343.5 MB = 1.03 GB static) and the
transient allocations per run (sort temporaries at flush) are ≤ 40 MiB on top of that.

Reading: the kernel-only cost meets the target at every level (36–82 μs; the propagate kernel is ~0 at 0 % and
~27 μs at 5 %). With the recorder on, 0–1 % activity stays at 36–65 μs. The one failing cell is **5 % of all neurons
spiking every 0.1 ms step with the recorder on**: 8,256 spikes/step = 82.6 M spikes/s = 660 MB/s of (t, idx) log,
which is host-transfer bound (≈ 23–33 μs/step on top of the kernels: sort 0.3 ms + D2H 1.5 ms per 100-step chunk
+ host concatenation). Two remarks, not excuses: (i) that load is the physical maximum — with t_ref = 2 ms no
neuron can fire more often than once per 21 steps at dt = 0.1 ms, so 5 %/step means literally the whole network
firing at its refractory limit, which is the runaway regime; (ii) the ignited states of the sweep (§4) produce
3.8 k–10.7 k spikes per ms, i.e. 0.2–0.65 % per 0.1 ms step, where the measured cost is 52–65 μs with recording.
Options if it ever matters: pinned staging (D2H 2.2 → 0.6 ms measured), transferring only sorted `idx` plus
per-step counts (halves bytes), or recording aggregated counts instead of every spike.

## 4. g sweep (`sweep.py`)

### 4.1 Protocol and criteria (fixed in code before the first run; unchanged since)

- Full retained graph, dt = 1 ms, default `EngineParams` except g; `G_GRID = geomspace(1e-3, 10, 20)`.
- Stimulus: 100 neurons drawn with `numpy.random.default_rng(0).choice(N, 100)` receive a constant current of
  30 (2× the 15 mV threshold gap → ~60 Hz on their own; ASSUMPTION, amplitude was unspecified in the brief) for
  the first 100 ms, then nothing; 400 ms of observation follow. Verdict on the last 200 ms:
  - **EXTINCT**: 0 spikes in the window.
  - **RUNAWAY**: fraction of neurons that spiked at least once in the window ≥ 0.90, **or** spikes in the last 50 ms
    ≥ 1.5 × spikes in the preceding 50 ms (a nonzero last half after a silent preceding half counts as growth).
  - **VALID**: neither; "normal range" if the active fraction is within [0.1 %, 30 %].
- Runtime: 20 points × 500 steps took < 5 s in total (the brief asked for an estimate first: the script prints one).

### 4.2 Primary result (seed 0; the two right-most columns are diagnostics, not criteria)

| g | active fraction | mean rate over all N (Hz) | window spikes | last 50 / prev 50 ms | verdict | rate per active neuron (Hz) | min v (mV) |
|---|---|---|---|---|---|---|---|
| 0.001 | 0.000 % | 0.000 | 0 | 0 / 0 | EXTINCT | – | −65.0 |
| 0.001624 | 0.000 % | 0.000 | 0 | 0 / 0 | EXTINCT | – | −65.0 |
| 0.002637 | 0.000 % | 0.000 | 0 | 0 / 0 | EXTINCT | – | −65.0 |
| 0.004281 | 0.000 % | 0.000 | 0 | 0 / 0 | EXTINCT | – | −65.0 |
| 0.006952 | 0.000 % | 0.000 | 0 | 0 / 0 | EXTINCT | – | −65.0 |
| 0.01129 | 0.000 % | 0.000 | 0 | 0 / 0 | EXTINCT | – | −65.0 |
| 0.01833 | 0.000 % | 0.000 | 0 | 0 / 0 | EXTINCT | – | −65.0 |
| 0.02976 | 0.000 % | 0.000 | 0 | 0 / 0 | EXTINCT | – | −65.0 |
| 0.04833 | 0.000 % | 0.000 | 0 | 0 / 0 | EXTINCT | – | −65.0 |
| 0.07848 | 0.000 % | 0.000 | 0 | 0 / 0 | EXTINCT | – | −65.0 |
| 0.1274 | 0.000 % | 0.000 | 0 | 0 / 0 | EXTINCT | – | −65.0 |
| 0.2069 | 0.000 % | 0.000 | 0 | 0 / 0 | EXTINCT | – | −65.0 |
| 0.336 | 0.000 % | 0.000 | 0 | 0 / 0 | EXTINCT | – | −65.0 |
| 0.5456 | 0.000 % | 0.000 | 0 | 0 / 0 | EXTINCT | – | −65.0 |
| 0.8859 | 14.988 % | 22.820 | 753,617 | 187,922 / 188,676 | VALID (normal range) | 152 | −4,136 |
| 1.438 | 18.036 % | 29.020 | 958,376 | 238,849 / 239,931 | VALID (normal range) | 161 | −6,480 |
| 2.336 | 22.136 % | 35.830 | 1,183,279 | 293,643 / 300,231 | VALID (normal range) | 162 | −11,956 |
| 3.793 | 24.731 % | 45.331 | 1,497,035 | 375,338 / 372,363 | VALID (normal range) | 183 | −19,164 |
| 6.158 | 26.887 % | 54.969 | 1,815,310 | 454,947 / 449,552 | VALID (normal range) | 204 | −41,539 |
| 10 | 30.089 % | 64.055 | 2,115,367 | 537,008 / 545,512 | VALID | 213 | −94,568 |

Time courses (spikes per 50 ms bin over the whole 500 ms; stimulus = bins 1–2):

- g ≤ 0.336: `[300, 300, 0, 0, …]` — only the 100 stimulated neurons fire (60 Hz each, as predicted); at most a
  handful of downstream spikes (g = 0.2069: 2; 0.336: 17), silence within one bin of stimulus offset.
- g = 0.5456: `[314, 375, 29, 0, …]` — 29 downstream spikes, then silence.
- g = 0.8859: `[19459, 175989, 188392, 189699, 189128, 187187, 188378, 188641, 188676, 187922]` — ignites during the
  stimulus and sits on a flat plateau for 400 ms after the stimulus ends.
- Larger g: same shape, higher plateau (g = 10: ~500 k–545 k per 50 ms).

### 4.3 Supplementary runs (same code, same criteria; NOT part of the pre-defined grid)

Finer grid across the transition, stimulus seed 0:

| g | active fraction | mean rate (Hz) | last 50 / prev 50 | verdict |
|---|---|---|---|---|
| 0.5456 | 0.000 % | 0.000 | 0 / 0 | EXTINCT |
| 0.5847 | 11.796 % | 17.311 | 142,941 / 142,556 | VALID (normal range) |
| 0.6266 | 12.290 % | 18.310 | 150,552 / 151,907 | VALID (normal range) |
| 0.6716 | 12.776 % | 19.292 | 158,910 / 159,617 | VALID (normal range) |
| 0.7197 | 13.482 % | 20.224 | 166,267 / 167,762 | VALID (normal range) |
| 0.7713 | 14.086 % | 21.028 | 173,393 / 172,947 | VALID (normal range) |
| 0.8266 | 14.617 % | 21.900 | 179,854 / 180,291 | VALID (normal range) |
| 0.8859 | 15.206 % | 22.812 | 188,248 / 186,986 | VALID (normal range) |

Different stimulated neurons (stimulus seeds 1 and 2):

| g | seed 1 | seed 2 |
|---|---|---|
| 0.5456 | VALID 11.3 % / 16.4 Hz (ignites slowly: bins `[836, 25339, 38350, 46726, 117353, 133559, …]`) | RUNAWAY 11.5 % / 7.6 Hz (still igniting at 500 ms: last/prev = 120,298 / 48,456) |
| 0.6 | VALID 11.9 % / 17.8 Hz | VALID 12.0 % / 17.7 Hz |
| 0.7 | VALID 13.4 % / 19.9 Hz | VALID 13.3 % / 19.9 Hz |
| 0.8859 | VALID 15.4 % / 22.8 Hz | VALID 15.1 % / 22.7 Hz |
| 2.336 | VALID 22.3 % / 36.6 Hz | VALID 22.7 % / 36.0 Hz |

### 4.4 Interpretation (stated plainly, per CLAUDE.md §3.3)

1. **A valid g range exists by the pre-defined criteria**: on the primary grid g ∈ [0.886, 10] is VALID and
   [0.886, 6.16] is in the "normal range"; the supplementary grid puts the lower edge near g ≈ 0.55–0.58 (it
   depends on which 100 neurons are stimulated).
2. **The valid state is an ignited, saturated, self-sustained state, not a low-rate asynchronous one.** Active
   neurons fire at 152–213 Hz on average (the refractory ceiling is 1/(t_ref + dt) = 333 Hz), the plateau is
   flat to ±1 % over 400 ms, and it survives the end of the stimulus indefinitely. The transition from silence is
   first-order: nothing between "0 downstream spikes" and "12 % of the network at 150 Hz" was found in the grid
   points tried (0.5456 → 0.5847). The network has no g at which activity is sustained at a low rate under this
   protocol (no tonic drive, no noise).
3. Below ignition the network is **input-driven and silent at rest**: downstream spikes exist (17 at g = 0.336,
   29 at g = 0.5456) but die within 50 ms of stimulus offset — EXTINCT by criterion, because the criterion measures
   self-sustained activity 200–400 ms after the input ends, not input-driven propagation.
4. **Unphysical hyperpolarisation in the ignited state**: min v = −4,136 mV at g = 0.886 and −94,568 mV at g = 10.
   Current-based synapses have no reversal potential, so hundreds of GABA/Glu partners at 150+ Hz drive v without
   bound. No NaN/Inf occur (float32 range is not approached) and the simulation stays deterministic, but this is a
   known limitation of the current-based model in this regime. The `v_floor` option (default off) exists for the
   planning session to decide on; it was **not** enabled for the sweep because enabling it would change the
   dynamics (floored neurons recover from inhibition faster) and that would be tuning.
5. These findings do not contradict the brief ("범위가 없으면 그것도 결과다" — here the range exists, but its
   character matters for M3): M3's negative control requires the auditory pathway to be quiet under silence, which
   the ignited state cannot provide; the input-driven regime (g below ignition) can, but is EXTINCT by M2's
   criterion. That decision belongs to the planning session; no parameter was changed here to resolve it.

Criteria history: none changed. The only free choices made outside the brief are the stimulus amplitude (30, fixed
before the first run) and the extra diagnostics/supplementary runs above.

### 4.5 Default g

Rule (stated before choosing): the smallest primary-grid point that is VALID (normal range) for every tested
stimulus seed (0, 1, 2). Result: **`DEFAULT_G = 0.886`** (grid value 0.8859, rounded). At 0.5456 the verdict
depends on the seed (EXTINCT / VALID / RUNAWAY), so it is marginal; 0.886 ignites within the stimulus period for all
three seeds and its plateau is the same to ±0.4 %. Larger g only raises the saturated rate and the unphysical
hyperpolarisation. `tests/test_engine.py::test_default_g_inside_recorded_valid_range` ties the default to the
committed `data-provenance/m2-sweep.json`.

## 5. Tests (`tests/test_engine.py`, all pass; `pytest tests/` = 31 passed incl. M1)

| test | what it enforces |
|---|---|
| `test_params_coefficients_and_validation` | exp(−dt/τ) coefficients compose across dt; ref_steps; validation errors |
| `test_default_g_inside_recorded_valid_range` | DEFAULT_G is a VALID point of the committed sweep |
| `test_propagate_matches_reference_synthetic[0/0.1 %/5 %/100 %]` | Triton propagate == `index_add_` reference, bit-exact; at 100 % equals column sums |
| `test_propagate_matches_reference_real_graph` | same on the full graph, 1 % random spikes |
| `test_three_neuron_signs[cpu, cuda]` | A→B +5 excites (B fires after A, v ≥ v_reset), A→C −5 inhibits (C silent, v < v_rest), refractory respected |
| `test_deterministic_bit_identical_real_graph` | full graph, g = 1.0, 100 steps (~1.6 M spikes): identical across runs, fresh engines, eager vs CUDA graph; sorted output |
| `test_deterministic_with_noise_and_seed_dependence` | noise on: same seed identical, different seed differs |
| `test_deterministic_synthetic_both_backends[cpu, cuda]` | determinism of the torch backend too |
| `test_no_nan_and_v_bounds_real_graph[g=0.01, 1.0]` | finite v/i_syn, v < v_thresh after every step, refractory counters in range; with `v_floor=−80` v ≥ floor |
| `test_v_lower_bound_without_inhibition` | excitatory-only graph: v never below v_reset |
| `test_dt_invariance_spike_times` | dt = 1 vs 0.1 ms: same spike counts (±1) and times within 1 ms |
| `test_torch_backend_matches_triton` | identical spike lists, v within 1e-3 mV on the 3-neuron graph |
| `test_recorder_overflow_raises[cpu, cuda]` | overflow raises `RecorderOverflow`; a fitting load records every spike with correct t |
| `test_run_input_conventions_and_no_record[cpu, cuda]` | callback return is copied; `record=False` → None; `step(None)` = zero input; bool view |

## 6. Acceptance checklist (CLAUDE.md §6 M2 / brief)

- [x] A valid g range exists and is documented with the table (§4.2): primary grid [0.886, 10]; normal range
      [0.886, 6.16]. Its physical character (ignited state) is stated in §4.4.
- [x] dt = 0.1 ms, < 100 μs per step on the full graph: 36–82 μs kernel-only at 0–5 % per-step activity;
      36–65 μs with recording at 0–1 %; **105–114 μs with recording at the 5 %/step ceiling** (§3, not hidden).
- [x] Same seed → bit-identical spikes, enforced by pytest (three determinism tests, none loosened).
- [x] `pytest tests/` passes (31).
- [x] One "M2: …" commit; `flysim/sensory`, `flysim/probe`, the viewer and `docs/m2-brief.md` untouched.

## 7. Deviations from the brief (and reasons)

1. **Integration scheme**: exact one-step integration instead of the forward-Euler form of the brief's equation
   (§1). The brief allowed either as long as coefficients are exp(−dt/τ); this one makes dt = 1 and 0.1 ms agree.
2. **"v stays within [v_reset, v_thresh]"**: the upper bound is enforced and tested strictly. The lower bound cannot
   hold in a current-based model with inhibitory edges (the brief's own 3-neuron sign test requires C to be
   hyperpolarised below v_rest = v_reset). Tests therefore check v ≥ v_reset without inhibition, v ≥ v_floor when a
   floor is set, and finiteness always. An optional `v_floor` parameter was added (default None = brief's model).
3. **Recorder returns `None` for `record=False`** (instead of two empty arrays) so "not recorded" is distinguishable
   from "recorded nothing".
4. **`i_ext_fn(t_step, buf)`** takes the buffer as second argument (the brief writes `i_ext_fn(t_step) -> Tensor` in
   one place and "write into the buffer" in another; both are supported: return None after writing, or return a
   tensor that is copied).
5. **Extra API surface**: `backend="torch"` usable on the GPU as well (reference/oracle), `CSRGraph` container for
   synthetic graphs, `engine.params` setter. No new files beyond the brief's list.
6. **Bench 5 %-with-recorder cell exceeds 100 μs** (§3). Not fixed by shrinking anything; causes and options recorded.
7. **Supplementary sweeps** (finer grid, two extra stimulus seeds) were run in addition to the pre-defined grid and
   are reported separately; they did not change the criteria or the primary table.

## 8. Notes for the planning session

- Two untracked files that are not part of this work appeared in the tree during the session:
  `flysim/engine/kernels.py` (02:04, an alternative kernel file with a ring-buffer spike history; nothing in this
  implementation imports it) and `docs/m3-brief.md` (02:14). Both were left untouched and are **not** in the M2
  commit. Commit `5e3fd9a` (ROI cache, gitignore fix) also landed on master during the session; the M2 commit sits
  on top of it.
- `data-provenance/m2-bench.json`, `m2-sweep.json`, `m2-sweep-supplementary.json` are committed (raw numbers,
  parameters, stimulated bodyIds, per-50 ms time courses).
