# M7a report — does a looming stimulus reach the jump command?

> **정정 (2026-10-23, M7a2 이후 · 총괄 세션이 추가).** 이 리포트의 P2 통과(선택 이득 10)와 "대조군 0, 반대쪽 0, 동측 먼저 199/199"는 **리셋 직후 과도 구간**에서 잰 값이다. 리셋 순간 모든 뉴런이 같이 출발해 거대 신경섬유가 한 발 쏘고, 적응(b_g 8.0)이 0.1~0.4 s 동안 묶어 두었다가 0.6 s 무렵 모든 시드에서 동시에 풀린다. 판정 창 0.2~0.5 s가 정확히 그 묶인 구간이었다. 예열 5 s 뒤 정상 상태에서 다시 잰 M7a2(c32cba3)의 결과는 이렇다: 회로는 **통과하지만 이득 640부터**이고 10은 통과하지 못한다. 대조군 거대 신경섬유는 판정 창당 0.6~0.7발(약 2.2 Hz)을 저절로 쏜다. 동측 먼저는 17/20(우연 수준 약 0.43). 지연이 강도를 부호화한다는 것과 `TTMn`이 구동되지 않는다는 것은 유지된다. 아래 원문은 그 시점의 기록으로 남긴다. 사전 등록에 예열 구간을 두지 않은 것은 총괄의 누락이다. 근거는 `docs/m7a2-report.md`.


Engine window ("M3"), 2026-10-23. Pre-registration: `docs/m7a-brief.md` (background
`docs/m7-game-ladder.md`), written before this ran and not edited afterwards. Code:
`flysim/apps/m7a_escape_probe.py`. Tests: `tests/test_m7a.py`. Raw results:
`data-provenance/m7a-results.json`, `data-provenance/m7a-sets.json`. 341 runs, 27 s.

**Answer: yes at the adapted operating point, no at the quiet one.**

- **P2 (adapted background) PASSes**, from the bottom of the gain grid (`a_in` = 10) upward, and never
  ignites. A one-sided loom fires the ipsilateral giant fibre (`DNp01`) in 10/10 seeds, and the
  ipsilateral fibre is the **first** to fire in **199 of 199** runs where either fired.
- **P1 (quiet conductance model) has no PASS.** The first gain that makes the detectors fire at all
  (`a_in` = 40) ignites the whole network, and M3's walk stops there.
- **`TTMn` is not driven by the stimulus at P2.** So, by the pre-registered branch, the game readout is
  `DNp01`, with the electrical-synapse assumption recorded (§6). Two further reasons the model's TTMn stays
  quiet are given there; neither involves changing a connection, and none was changed.

## 1. Sets — all from data queries (CLAUDE.md §3.1)

| set | left | right | side source | instances |
|---|---|---|---|---|
| `LC4` | 71 | 55 | `somaSide` | `LC4_L`, `LC4_R` |
| `LPLC2` | 94 | 91 | `somaSide` | `LPLC2_L`, `LPLC2_R` |
| `DNp01` | 1 (bodyId 10010) | 1 (bodyId 10001) | `somaSide` | `DNp01(GF)_L`, `DNp01(GF)_R` |
| `TTMn` | **1** (bodyId 804642) | **1** (bodyId 800146) | `somaSide` | `TTMn_L`, `TTMn_R` |

`somaSide` is complete for all four types, so the brief's `rootSide` and instance-suffix fallbacks were
never needed (the retained annotation cache has no `rootSide` column in any case). `Graph.side` agrees with
`somaSide` for every one of these neurons.

**Wiring the experiment rests on**, re-checked and pinned by
`test_looming_detectors_reach_only_the_ipsilateral_giant_fibre`:

| path | contacts | cells connected | sign |
|---|---|---|---|
| `LC4_L` → `DNp01_L` / `DNp01_R` | **3,782** / 0 | 71/71 / 0 | ACh, + |
| `LC4_R` → `DNp01_R` / `DNp01_L` | **2,580** / 0 | 55/55 / 0 | ACh, + |
| `LPLC2_L` → `DNp01_L` / `DNp01_R` | **2,642** / 0 | 94/94 / 0 | ACh, + |
| `LPLC2_R` → `DNp01_R` / `DNp01_L` | **2,220** / 0 | 91/91 / 0 | ACh, + |
| `DNp01_R` → `TTMn_R` / `TTMn_L` | **70** / 0 | — | ACh, + |
| `DNp01_L` → `TTMn_L` / `TTMn_R` | **20** / 0 | — | ACh, + |

Two things the ladder table did not show:

- **The GF → TTMn link is asymmetric: 70 contacts on the right, 20 on the left.** The ladder's "90" is the
  sum. `TTMn_R` is `DNp01_R`'s largest output; `DNp01_L`'s largest output is `DNp11_R` (90), not TTMn.
- **TTMn sits under heavy inhibition.** Its largest inputs are glutamatergic interneurons at −120 to −223
  contacts each; GF provides about 2 % (right: 70 of 3,608) and under 1 % (left: 20 of 2,424) of its input.

## 2. Protocol (all fixed before the run)

| item | value | note |
|---|---|---|
| timeline | 200 ms silence · 300 ms ramp · 300 ms observation | brief |
| stimulus | `a_in · (e^{t/τ} − 1)/(e^{T/τ} − 1)` during the ramp, then cut | ASSUMPTION: detector output ∝ angular expansion rate, as the brief states |
| τ | **50 ms** | ASSUMPTION, the brief says "exponential" without a constant. 4.7 % of max at 150 ms, 37 % at 250 ms, so most of the drive arrives in the last 100 ms, as in a real loom |
| heterogeneity | none — every stimulated cell gets the same current | ASSUMPTION (brief). Consequence: the stimulated cells fire **in lock-step**, the most synchronous possible input |
| conditions | left only · right only · both · none | brief |
| P1 | conductance, `g` 3.162e-4, σ 0, no adaptation, seed 0 | brief |
| P2 | conductance, `g` 5.36539e-4, σ 29.13, `adapt_g_b` 8.0, seeds 0–9 | brief (M2f's validated point) |
| gain grid | 10, 20, 40, …, 5120 | M3's grid and doubling extension, imported from `m3_click.py` |
| selection | walk the grid upward; the first ignited point stops the walk; the first non-ignited PASS before it is selected | M3's rule |
| stimulus window | 200–500 ms (the ramp) | brief: "자극 구간" |
| late window | 600–800 ms (offset + 100 ms) | M3 |

The ramp lives in the app module, not in `flysim/sensory/` — that directory is another window's work in
progress (`looming.py`), and the brief forbids touching it.

### PASS (brief, unchanged)
For **both** the left-only and right-only conditions: ipsilateral `DNp01` ≥ 1 spike in the stimulus window
(P2: in ≥ 9 of 10 seeds); the control's `DNp01` silent in that window (P2: excess > 3 SEM of the control);
contralateral spike count below ipsilateral (P2: means). And the point must not be ignited.

### Ignition — the three-number rule, with numbers fixed before the run
`parameter-decisions.md` (after M2i) defines the rule qualitatively: report the control's absolute state,
the stimulus run's absolute state and the control-relative excess; a run is usable only if the control is
not itself in runaway and there is no stimulus-driven excess. It gives no thresholds, so they were taken
from existing project numbers rather than chosen here:

| number | ignited if | source |
|---|---|---|
| control absolute | late-window network rate ≥ **5 Hz** | upper bound of M2d's usable band; background is 0.1–0.35 Hz, runaway 21–33 Hz (M2i) |
| stimulus absolute | late-window network rate ≥ **5 Hz**, or ≥ **90 %** active | same 5 Hz; CLAUDE.md §3.3 |
| excess | late-active fraction above control by ≥ **1 point** and > 3 SEM | M3's 1 % ignition number, measured above the control |

With no noise and a silent control (P1) the excess test reduces exactly to M3's rule. With a background
(P2) it does not flag the background itself — the M2g false positive — which a test pins
(`test_background_state_is_not_ignition`). A gain point counts as ignited if **any** of its three stimulus
conditions ignites (conservative; the game will drive both sides).

## 3. P1 — quiet conductance model: **no PASS**

| a_in | left stim: GF ipsi/contra | right stim: GF ipsi/contra | detector spikes (left stim) | total spikes | late rate (Hz) | late-active excess (pts) | ignited |
|---|---|---|---|---|---|---|---|
| 10 | 0/0 | 0/0 | 0 | 0 | 0.00 | 0.00 | no |
| 20 | 0/0 | 0/0 | 0 | 0 | 0.00 | 0.00 | no |
| **40** | 11/8 | 11/10 | 820 | 614,535 | **13.14** | **8.93** | **yes** ← walk stops |
| 80 | 22/8 | 22/11 | 1,935 | 188,923 | 3.16 | 4.04 | yes |
| 160 | 34/8 | 34/12 | 3,313 | 778,013 | 13.29 | 9.06 | yes |
| 320 | 45/8 | 45/11 | 5,026 | 237,286 | 3.22 | 4.31 | yes |
| 640 | 56/8 | 56/13 | 6,853 | 935,995 | 13.27 | 9.20 | yes |
| 1280 | 66/9 | 66/11 | 8,621 | 1,011,136 | 13.28 | 9.10 | yes |
| 2560 | 74/8 | 74/12 | 10,263 | 306,163 | 3.24 | 4.11 | yes |
| 5120 | 82/9 | 82/12 | 11,780 | 1,121,882 | 13.23 | 9.07 | yes |

At `a_in` = 20 the drive exceeds the 15 mV threshold gap only in the last few milliseconds of the ramp and
**no detector fires at all**. At 40 every detector cell fires about five times **in lock-step**, and that
volley ignites the whole network — 615 k spikes, 12 % of all neurons active, 13 Hz per neuron in the late
window. Every higher gain ignites too. So **the silent-to-ignited step falls between two adjacent points of
M3's grid**, and there is no gain on that grid at which the circuit responds without the network running
away. The grid was not refined; that would be a new grid.

The contralateral GF fires 8–13 times at every ignited point, 4 ms after the ipsilateral one. It has no
direct input from the stimulated side (§1); the runaway reaches it. The GF responses at P1 are therefore
not escape-circuit responses and are not used.

## 4. P2 — adapted background: **PASS, selected `a_in` = 10**

Control (no stimulus, 10 seeds): late-window network rate **0.42 Hz**, late-active 8.4 % — the background
state, well under the 5 Hz runaway line. `DNp01` silent in the stimulus window in every control seed.

| a_in | side | ipsi GF mean | seeds firing | contra GF mean | latency (ms from onset, median) | ipsi detector spikes | ignition: ctrl / stim rate (Hz), excess | verdict |
|---|---|---|---|---|---|---|---|---|
| **10** | left | 1.00 | 10/10 | 0.00 | 284 | 56 | 0.42 / 0.55, −0.37 pt | **PASS** |
| **10** | right | 0.90 | 9/10 | 0.00 | 295 | 42 | 0.42 / 0.56, −0.42 pt | **PASS** |
| 20 | left | 1.00 | 10/10 | 0.60 | 254 | 159 | 0.42 / 0.49, −0.59 pt | PASS |
| 20 | right | 1.00 | 10/10 | 1.00 | 264 | 140 | 0.42 / 0.54, −0.10 pt | fail (contra = ipsi) |
| 40 | left / right | 1.00 / 1.00 | 10/10 | 0.20 / 0.90 | 222 / 229 | 159 / 140 | not ignited | PASS |
| 80 | left / right | 1.00 / 1.00 | 10/10 | 0.00 / 0.50 | 189 / 200 | 160 / 141 | not ignited | PASS |
| 160 | left / right | 1.00 / 1.00 | 10/10 | 0.00 / 0.20 | 159 / 168 | 163 / 144 | not ignited | PASS |
| 320 | left / right | 1.00 / 1.00 | 10/10 | 0.00 / 0.00 | 128 / 135 | 166 / 148 | not ignited | PASS |
| 640 | left / right | 2.00 / 2.00 | 10/10 | 0.50 / 0.70 | 100 / 108 | 320 / 272 | not ignited | PASS |
| 1280 | left / right | 2.00 / 2.00 | 10/10 | 1.00 / 1.00 | 77 / 80 | 324 / 287 | not ignited | PASS |
| 2560 | left / right | 2.10 / 2.00 | 10/10 | 0.90 / 1.00 | 58 / 60 | 487 / 432 | not ignited | PASS |
| 5120 | left / right | 3.00 / 2.50 | 10/10 | 0.30 / 1.00 | 42 / 43 | 654 / 578 | not ignited | PASS |

**No gain point ignites at P2**; the late-active excess is negative everywhere (the loom slightly *lowers*
late activity, consistent with adaptation after the volley). The noise background explains why `a_in` = 10
already works here when it did nothing at P1: the noise carries the detectors over threshold that the ramp
alone does not reach.

**Caveat 1 — the selected gain is the bottom edge of the grid.** `a_in` = 10 is the smallest value M3's grid
contains, so "the smallest PASS" is ≤ 10 and nothing below it was tested — the same situation M2d was in at
the top of its grid. The grid was not extended.

**Caveat 2 — at P2 the giant fibre is a one-spike neuron.** It fires exactly once from `a_in` 10 to 320 and
two or three times above. That is the adaptation, not the circuit: `adapt_g_b` = 8.0 adds 8 /ms of
conductance to `E_adapt` per spike, 160 times the leak, so after one spike the GF is clamped for of order
τ_a. Every neuron in the model carries this. "The GF fires" at P2 is therefore a binary event, not a burst.

**What carries information at P2** (descriptive, not used for the verdict):

- **Which GF fires first is perfectly lateralised.** Across all P2 left-only and right-only runs in which
  either GF fired in the stimulus window, the **ipsilateral fibre fired first, or alone, in 199 of 199**.
  The contralateral fibre, when it fires, lags by 7–36 ms at low gain (polysynaptic spread through the
  noisy background) and by ~170–200 ms at high gain (near the end of the ramp). Spike *counts* are a weaker
  lateral signal than spike *order*: at `a_in` = 20 the counts tie on the right side and the point fails
  PASS on that alone.
- **Latency encodes loom strength.** Median first-spike latency falls monotonically from 284–295 ms at
  `a_in` = 10 to 42–43 ms at 5120 — the GF crosses threshold earlier on a steeper ramp. A game readout
  could use time-to-first-spike as an analogue signal.
- **Both sides together co-activate both fibres in 10/10 seeds** at every gain.

## 5. TTMn

**P2: not driven by the stimulus.**

| | control | stimulus runs (all gains, left and right) | cells > 3 SEM of control |
|---|---|---|---|
| `TTMn_L`, stimulus window | 0.00 | 0.00 | 0 / 20 |
| `TTMn_R`, stimulus window | 0.00 | 0.00 | 0 / 20 |
| `TTMn_L`, post window | 0.50 (SEM 0.17) | 0.39 | 0 / 20 |
| `TTMn_R`, post window | 0.60 (SEM 0.16) | 0.46 | 0 / 20 |

Zero spikes in the stimulus window, and the occasional post-window spike is background at the control's
level (the stimulus runs are, if anything, slightly lower).

**P1: TTMn fires only inside the runaway** — 3–38 spikes per run, more on `TTMn_R` with right-side stimulation
(up to 38 against 3–6 on the left), consistent with the 70-vs-20 contact asymmetry. Because every one of those
runs is ignited, this is not a clean measure of GF → TTMn and is not used.

## 6. Interpretation, by the pre-registered branches

- *"PASS and TTMn also fires → read out from TTMn."* Not this case.
- **"PASS but TTMn is quiet → read out from `DNp01`; record that this may be because the connectome lacks the
  electrical synapse; do not strengthen any connection."** → **This is the case at P2.** The game readout is
  `DNp01`. No connection was changed.
- *"No PASS → M7c and later are not written."* Applies to P1 only. P2 passes, so M7c is not blocked.

The brief names one candidate cause for the quiet TTMn. The data give two more, and all three point the same
way — none is a reason to edit the wiring:

1. **Electrical synapses are absent from the connectome** (brief). The literature places much of GF → TTMn on
   gap junctions; the model has only the chemical contacts.
2. **At P2 the GF fires once.** A single GF spike delivers 70 (right) or 20 (left) contacts. M2b measured
   200–300 simultaneous excitatory contacts as the single-volley threshold from rest at `g` = 3.162e-4;
   scaled linearly to `g` = 5.36539e-4 that is roughly 120–180. One GF spike is below it on both sides, and the
   adaptation that makes the GF a one-spike neuron removes the burst that could summate.
3. **TTMn is dominated by inhibition** (−120 to −223 contacts per top input; GF ≈ 2 % / < 1 % of its input),
   and at P2 those inhibitory interneurons sit in the same noisy background as everything else.

## 7. What this hands to M7c

- **Operating point:** P2 (adapted background), `a_in` = 10 by M3's rule — at the bottom grid edge.
- **Readout:** `DNp01`, not `TTMn`. The most reliable lateral signal is **which GF fires first** (199/199
  ipsilateral). Spike counts lateralise less cleanly (one fail at `a_in` = 20).
- **What the readout is:** one spike per loom over most of the gain range, so a binary "jump / no jump" per
  side, plus first-spike latency as a graded measure of loom strength.
- **The shortcut to disclose on screen** (ladder rule 4): the detectors are driven directly with a synthetic
  ramp; nothing upstream of `LC4`/`LPLC2` is simulated. From the detectors to the GF is wiring; GF → TTMn is
  not used.

## 8. Completion criteria (`docs/m7a-brief.md`)

- [x] set-size table, LC4 · LPLC2 · DNp01 · TTMn by side (§1)
- [x] `a_in` sweep and selected value per operating point (§3, §4)
- [x] four conditions × two operating points: judgement, latency, TTMn response (§3–§5)
- [x] `pytest tests/`: **349 passed** (338 + 11 new in `tests/test_m7a.py`)
- [x] one "M7a: …" commit, not pushed

## 9. Deviations from the brief

1. **The three-number ignition rule needed numbers**; the recorded rule is qualitative. Each threshold was
   taken from an existing project number (§2) and fixed in code before the first run.
2. **The whole gain grid was run** to fill the table. Selection still follows M3's walk, stopping at the
   first ignited point; the rows after it are descriptive and cannot change the selection.
3. **A gain point is ignited if any of its three stimulus conditions ignites.** The brief says an ignited
   point is not a PASS without saying which conditions count; the conservative reading was taken.
4. **The ramp's time constant (50 ms) was chosen** because the brief specifies an exponential ramp without
   one. Recorded as an assumption, fixed before the run.
5. **The stimulus is implemented in the app**, not in `flysim/sensory/`, which another window is writing.
6. **`rootSide` was not used** because `somaSide` is complete for every set and the cache has no such column.

## 10. Notes for the planning session

- **The adaptation line's background state is what makes the escape circuit usable.** Without it (P1) the
  first effective loom ignites the whole brain. That is the third time the same pattern has appeared: M2i
  showed the background needs adaptation; here the circuit response needs the background.
- **Grid-edge selection again.** Both P1's failure (silent at 20, ignited at 40) and P2's success (already
  at 10) sit at or between M3's grid points. If M7c's behaviour turns out to be sensitive to gain, a
  pre-registered finer grid would settle it; this round did not do it.
- **Spike order beats spike count** as the lateral readout at P2. M7b's "brainless bot" gets the same
  detector input; whether it can match a 199/199 first-spike laterality is the comparison that will say
  whether the wiring did anything here.
- **If a TTMn readout is ever wanted**, the honest options are a modelled gap junction (a model-structure
  decision of the M2b/M2d kind, needing its own pre-registration) or reading `DNp01` and saying so. Scaling
  up the 70/20 chemical contacts is the arbitrary mapping the ladder rules out.
- `parameter-decisions.md`, `backlog.md` and the ladder doc were not edited (planning-owned).
