# M7a2 report — the escape circuit at steady state

Engine window ("M3"), 2026-10-23. Pre-registration: `docs/m7a2-brief.md`, written before this ran and not
edited afterwards. Code: `flysim/apps/m7a2_steady_probe.py`. Tests: `tests/test_m7a2.py`. Raw results:
`data-provenance/m7a2-results.json`, `data-provenance/m7a2-timecourse.json`. 640 runs + 20 time courses, 344 s.

**Interpretation branch: "P2에서 통과".** At steady state the circuit still passes — but **not at the gain M7a
chose.** M7a's PASS at `a_in` = 10 was the post-reset transient: at steady state the loom only clears the
spontaneous firing from **`a_in` = 640** upward, 64× higher. There the ipsilateral giant fibre adds about one
spike per loom over its same-seed control, the contralateral one does not move, and the ipsilateral fibre is
first in 17 of 20 runs (85 %, against ~40–45 % chance) — not M7a's 199/199. It never ignites. `TTMn` still does
not respond to the stimulus.

M7a's specific numbers therefore need a correction notice whatever is decided about M7c (§6).

## 1. What M7a measured, and why it was a transient

M7a's own control runs already contained the evidence; the engine window did not look at the GF's post window
in them, and should have. In **all 10 P2 control seeds, both giant fibres fire exactly one spike between 543
and 706 ms**, left and right within 0–42 ms of each other, and none in M7a's 200–500 ms judgement window.

The time course recorded here (P2, no input, seed mean of 10, 100 ms bins) shows the mechanism:

| t (s) | network (Hz) | GF L (Hz) | GF R (Hz) | TTMn L | TTMn R | what is happening |
|---|---|---|---|---|---|---|
| 0.0 | 1.181 | 10 | 10 | 8 | 10 | **every seed's GF fires once** — the reset kick, all neurons start together |
| 0.1 | 0.250 | 0 | 0 | 2 | 0 | |
| 0.2 | 0.187 | **0** | **0** | 0 | 0 | ← M7a's window starts |
| 0.3 | 0.165 | **0** | **0** | 0 | 0 | GF held down by adaptation (`b_g` 8.0 = 160× the leak per spike, τ_a 100 ms) |
| 0.4 | 0.158 | **0** | **0** | 0 | 0 | ← M7a's window ends at 0.5 |
| 0.5 | 0.220 | 1 | 1 | 0 | 0 | |
| 0.6 | 0.486 | **8** | **9** | 2 | 5 | **the clamp releases in every seed at once — a synchronous wave** |
| 0.7 | 0.359 | 1 | 0 | 3 | 1 | |
| 0.8–1.4 | 0.19–0.34 | 0–4 | 0–6 | 0–2 | 0–2 | settling |
| 2–4 s mean | 0.271 | 1.80 | 1.75 | 1.05 | 1.10 | |
| 4–6 s mean | 0.279 | 1.85 | 1.90 | 1.15 | 1.05 | |
| 6–8 s mean | 0.273 | 1.95 | 1.85 | 1.00 | 1.10 | |
| 8–10 s mean | 0.273 | 1.75 | 1.85 | 0.95 | 1.10 | |
| **last 5 s** | **0.274** | **1.86** | **1.88** | **1.02** | **1.10** | steady state |

M7a's 200–500 ms window sat **exactly inside the adaptation silence that follows the reset spike.** Every
"zero" M7a measured there — control GF, contralateral GF — was the GF being clamped, not the circuit being
quiet. At steady state the GF fires spontaneously at **1.86 / 1.88 Hz**, in line with M7c's ~1.9 Hz measured
independently, and **`TTMn` fires spontaneously at ~1.0–1.1 Hz** as well.

**Steady-state entry**, by the brief's definition (first time after which the forward 1 s moving average
stays within ±20 % of the last-5-s mean): network **0.1 s**; GF L 1.7 s, GF R 3.7 s; TTMn L 8.5 s, R 6.5 s.
The single-neuron entries are limited by counting noise rather than by the dynamics — a 1 s window over 10
seeds at ~2 Hz holds about 20 spikes, a Poisson spread of ±22 %, wider than the ±20 % tolerance — so they
wander late without the rate changing (2–4 s and 8–10 s means agree within 0.15 Hz). By inspection the GF has
settled by about 2 s. **The 5 s warm-up clears the transient with margin at P2.**

At σ 23.30 the network settles by 2.0 s but the GF rate drifts upward through the record (1.25–1.35 Hz at
2–6 s, 1.70 at 8–10 s; last-5-s 1.52), and the criterion places its entry at 6.9 / 7.5 s. **At the
auxiliary σ the 5 s warm-up is therefore marginal for the GF**; its results below are descriptive anyway.

## 2. Protocol (fixed before the run)

5 s with no input, then exactly M7a's stimulus: 200 ms nothing, 300 ms exponential ramp (τ 50 ms) to `a_in`,
cut, 300 ms observation. Left only · right only · both · none. Noise seeds 0–9. M7a's gain grid (10 … 5120) and
M7a's selection walk (first non-ignited PASS; stop at the first ignited point). Judgement window: the 300 ms
ramp, 5200–5500 ms.

**PASS (brief §5)**, for both left-only and right-only: paired with the same seed's "none" run, the mean
ipsilateral GF spike difference (stimulus − none) is > 0 and > 3 SEM over the 10 seeds; the mean contralateral
difference is smaller than the ipsilateral one; and the gain point is not ignited (M7a's three-number rule,
with the steady-state "none" runs as the control; a gain point is ignited if any of its three stimulus
conditions is).

**The pairing is exact.** The engine is deterministic per seed and draws noise every step independently of
the input, so a stimulus run and its "none" run share the identical 5 s warm-up and the identical noise
realisation. The paired difference isolates what the stimulus did. The warm-up was therefore re-simulated in
every run rather than snapshotted, which would have meant copying engine state and RNG state by hand from
outside the engine.

M7a's "active fraction" (whole 800 ms run) is measured here over the matching post-warm-up 800 ms; including
the warm-up would count 5 s of background (M7c: 66 % of neurons fire at least once in 5 s).

## 3. P2 (σ 29.13) — decides the verdict

"None" runs after the warm-up: GF **0.70 / 0.60 spikes per 300 ms window** (≈ 2.2 Hz), `TTMn` 0.50 / 0.30,
late network rate 0.33 Hz. M7a's "none" in the same window had **0**.

| a_in | left: Δ ipsi ± SEM | Δ contra | right: Δ ipsi ± SEM | Δ contra | ipsi first | latency L / R (ms, median) | signal / spontaneous | verdict |
|---|---|---|---|---|---|---|---|---|
| 10 | +0.30 ± 0.15 | +0.00 | +0.10 ± 0.10 | +0.10 | 10/17 | 146 / 252 | 1.4 / 1.2 | – |
| 20 | +0.50 ± 0.17 | −0.10 | +0.50 ± 0.17 | +0.10 | 13/20 | 192 / 257 | 1.7 / 1.8 | – |
| 40 | +0.40 ± 0.16 | −0.10 | +0.50 ± 0.17 | +0.10 | 13/20 | 177 / 236 | 1.6 / 1.8 | – |
| 80 | +0.30 ± 0.15 | −0.10 | +0.40 ± 0.22 | +0.10 | 12/20 | 162 / 199 | 1.4 / 1.7 | – |
| 160 | +0.30 ± 0.15 | −0.10 | +0.30 ± 0.21 | +0.00 | 14/20 | 149 / 163 | 1.4 / 1.5 | – |
| 320 | +0.20 ± 0.13 | −0.30 | +0.10 ± 0.18 | −0.20 | 14/17 | 115 / 134 | 1.3 / 1.2 | – |
| **640** | **+1.10 ± 0.10** | −0.10 | **+1.10 ± 0.18** | +0.00 | **17/20** | 91 / 111 | 2.6 / 2.8 | **PASS ← selected** |
| 1280 | +1.10 ± 0.10 | −0.10 | +1.00 ± 0.15 | +0.10 | 17/20 | 68 / 86 | 2.6 / 2.7 | PASS |
| 2560 | +1.10 ± 0.10 | −0.10 | +1.00 ± 0.15 | +0.10 | 17/20 | 50 / 64 | 2.6 / 2.7 | PASS |
| 5120 | +2.10 ± 0.10 | −0.20 | +1.50 ± 0.22 | +0.20 | 17/20 | 38 / 48 | 4.0 / 3.5 | PASS |

(Δ = stimulus − same-seed "none", GF spikes in the 300 ms window. "Ipsi first" = runs in which the ipsilateral
GF fired first, out of runs in which either fired; ties count as misses.)

**No gain point ignites.** Control late rate 0.33 Hz, stimulus late rate at most 0.56 Hz (5 Hz line), largest
late-active excess +0.46 point (1 point line).

**Reading the table.**

- **Below 640 the loom is lost in the spontaneous firing.** It adds 0.1–0.5 spikes on average over a background
  of ~0.65 spikes per window, never three standard errors clear. At `a_in` = 10 — M7a's selection — the gain is
  +0.30 and +0.10.
- **From 640 it is unambiguous**: about one extra ipsilateral spike per loom (+1.0 to +1.1, SEM 0.10–0.18),
  rising to +1.5–2.1 at 5120; the contralateral fibre does not change (−0.2 to +0.2). The step from 320 to 640
  is abrupt — two spikes become possible where adaptation allowed one.
- **"First to fire" stays lateralised, but at 85 %, not 100 %.** The pooled chance level, from the "none" runs
  scored as both a left and a right trial, is **0.43** (7 of 10 had a spike: 5 left-first, 1 right-first,
  1 tie). 17/20 is well above it; 12–14/20 below 640 is above it too, but those gains do not pass on counts.
- **Latency still encodes loom strength** at steady state: 146–252 ms at 10 down to 38–48 ms at 5120.
- **`TTMn` does not respond**: its same-seed difference is −0.1 to +0.2 at every gain, against a spontaneous
  ~1 Hz.

## 4. σ 23.30 — auxiliary arm (descriptive, does not enter the verdict)

"None" runs: GF 0.50 / 0.50 per window (≈ 1.7 Hz), late rate 0.17 Hz. Pooled chance for "ipsi first": 0.40.

| a_in | left: Δ ipsi ± SEM | Δ contra | right: Δ ipsi ± SEM | Δ contra | ipsi first | latency L / R | signal / spont. | would pass |
|---|---|---|---|---|---|---|---|---|
| 10 | +0.20 ± 0.13 | +0.10 | +0.00 ± 0.00 | +0.00 | 6/12 | 155 / 44 | 1.4 / 1.0 | – |
| **20** | **+0.80 ± 0.13** | +0.40 | **+0.90 ± 0.10** | +0.50 | 14/20 | 220 / 216 | 2.6 / 2.8 | **yes (would be selected)** |
| 40 | +0.80 ± 0.13 | +0.20 | +0.80 ± 0.13 | +0.30 | 14/20 | 198 / 204 | 2.6 / 2.6 | yes |
| 80 | +0.70 ± 0.15 | +0.20 | +0.70 ± 0.15 | +0.20 | 14/20 | 184 / 183 | 2.4 / 2.4 | yes |
| 160 | +0.60 ± 0.16 | +0.20 | +0.60 ± 0.16 | +0.20 | 15/20 | 154 / 162 | 2.2 / 2.2 | yes |
| 320 | +0.60 ± 0.16 | +0.20 | +0.50 ± 0.17 | +0.20 | 15/20 | 125 / 122 | 2.2 / 2.0 | no — right side +0.50 equals 3 SEM, not above it |
| 640 | +1.00 ± 0.26 | +0.20 | +0.80 ± 0.20 | +0.20 | 15/20 | 102 / 104 | 3.0 / 2.6 | yes |
| 1280 | +1.30 ± 0.15 | +0.20 | +1.30 ± 0.15 | +0.30 | 17/20 | 80 / 78 | 3.6 / 3.6 | yes |
| 2560 | +1.30 ± 0.15 | +0.20 | +1.30 ± 0.15 | +0.30 | 17/20 | 61 / 60 | 3.6 / 3.6 | yes |
| 5120 | +2.30 ± 0.15 | +0.20 | +2.30 ± 0.15 | +0.20 | 18/20 | 42 / 42 | 5.6 / 5.6 | yes |

No gain point ignites (late-active excess negative everywhere). Two differences from P2, both within the
scope this arm was added for:

- **With 20 % less noise the circuit passes from `a_in` = 20, 32× lower than at P2.** The steady-state
  sensitivity of the escape circuit depends strongly on σ.
- **The contralateral GF also rises with the stimulus** (+0.2 to +0.5), less than the ipsilateral one but not
  zero as at P2. Lateralisation by counts holds; it is less clean.

`TTMn` differences here are +0.1 to +0.4, slightly positive at every gain — not tested, not used.

The GF in this arm had not fully settled by 5 s (§1), so these numbers carry a caveat P2's do not.

## 5. Verdict by the pre-registered branches

- **"P2에서 통과" → 회로는 정상 상태에서도 통한다.** At P2, both sides pass from `a_in` = 640, never ignited.
  **This branch applies.** By the brief, M7c's loss is then attributed to the decoder (50 ms window, a one-tick
  move per spike), and whether to pre-register a new decoder is for the user and the planning session.
- *"P2에서 통과 없음 → M7a의 통과는 과도 구간의 산물."* Does not apply as a whole. **But M7a's pass *at
  `a_in` = 10* was the transient** — at steady state `a_in` = 10 does not pass — so part of what this branch
  describes is true of M7a's specific numbers (§6).

**One observation that bears on the decoder attribution, offered as an observation, not a verdict.** M7c did not
use M7a's `a_in`; it fitted its own input scale `k`. Its effective 99th-percentile drive (`k × p99`) was about
**72 for both dodge games** — where all 21 grid points scored 0 and the tie-break picked the middle — and about
**1,400 for the jump game**. M7c's input is not M7a's ramp, so the comparison is approximate; but on the
steady-state curve above, ~72 sits in the range where the loom does not clear the spontaneous firing, and
~1,400 sits in the range where it does. For the dodge games, gain is a candidate cause alongside the decoder.

## 6. What changes in M7a's record

| M7a stated | at steady state (M7a2) |
|---|---|
| P2 passes, selected `a_in` = 10 | P2 passes, selected **`a_in` = 640**; `a_in` = 10 does **not** pass |
| control GF silent in the window (0) | control GF fires **0.6–0.7 spikes per window** (≈ 2.2 Hz) |
| contralateral GF 0 at `a_in` = 10 | contralateral **difference** ≈ 0 (−0.2 … +0.2) — holds in difference terms |
| ipsilateral GF first in **199 / 199** | ipsilateral GF first in **17 / 20** at passing gains (chance ≈ 0.43) |
| latency encodes loom strength (284 → 42 ms) | **holds** (146–252 → 38–48 ms) |
| `TTMn` not driven | **holds**; `TTMn` fires spontaneously at ~1 Hz |
| P1 has no PASS (ignites at 40) | not re-tested here (P1 has no noise, so no background to settle) |

`docs/m7a-report.md` is left as it was — it is the record of that moment; the correction notice is the planning
session's to attach.

## 7. Completion criteria (`docs/m7a2-brief.md`)

- [x] background time course and steady-state entry times (§1; full 100-bin tables in `m7a2-timecourse.json`)
- [x] post-warm-up judgement table for both σ, selected gain, laterality and latency (§3, §4)
- [x] which interpretation applies, stated (§5)
- [x] `pytest tests/`: **414 passed** (5 new in `tests/test_m7a2.py`)
- [x] one "M7a2: …" commit, not pushed

## 8. Deviations and clarifications

1. **"그 뒤 1 s 이동평균" is read as the forward window [t, t + 1 s).** The single-neuron entry times this gives
   are counting-noise-limited (§1); the network entry and the by-inspection settling of the GF (~2 s) are the
   meaningful ones.
2. **Active fraction for the ignition rule** is taken over the post-warm-up 800 ms, matching what M7a's
   whole-run figure covered (§2).
3. **The chance level for "ipsi first"** was not specified in the brief; it is computed from the steady-state
   "none" runs scored once as a left and once as a right trial, and is small-n (7 and 5 runs with a spike).
4. Nothing else. Connections, weights, the noise distribution, the operating point, the gain grid, the stimulus,
   the judgement rule and the selection walk are as written. No new operating point was searched.

## 9. Notes for the planning session

- **The mechanism of the M7a artefact is specific and worth recording**: a reset makes every neuron start
  together, the GF fires once at t ≈ 0, the adaptation added by M2d clamps it for ~0.5 s, and it releases in
  every seed at once around 0.6 s. Any protocol that judges inside the first ~2 s after a reset at P2 is
  measuring this, not the circuit. **A warm-up should be standard from here on.**
- **The escape circuit works at steady state, but it needs about one decade more drive than M7a said**, and its
  sensitivity is strongly noise-dependent (640 at σ 29.13, 20 at σ 23.30).
- **For M7c**: whatever decoder is chosen, the input scale has to land in the range where the loom clears the
  spontaneous GF firing. On this curve that is `a_in` ≥ 640 at P2; M7c's dodge games appear to have run well
  below it (§5).
- **Spike order is still the better lateral signal**, but 85 % rather than 100 %. A decoder that waits for the
  first GF spike after a loom onset, against a background of ~2 spontaneous spikes per second per side, should be
  designed with that base rate in view.
- `docs/m7a-report.md`, `parameter-decisions.md` and the ladder doc were not edited (planning-owned).
