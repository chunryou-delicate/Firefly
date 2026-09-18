# M2g report — does the adaptation background state carry the signal further?

Engine session (window "M3 엔진"), 2026-09-19. Pre-registration: `docs/m2g-brief.md`, written before this
ran and not edited afterwards. Code: `flysim/apps/m2g_adapt_click.py` (new; `m3_click.py` untouched, its
protocol constants imported rather than copied). Raw results: `data-provenance/m2g-results.json`.

**Answer: no.** Neither operating point carries the click train past hop 1. At hop 2 and beyond every
probe set is statistically indistinguishable from its own silence control. Per the pre-registered rule,
**neither A nor B is proposed**, and the finding is recorded as: *adaptation produces a background state
but does not extend propagation distance.*

One caveat in §6 qualifies how much that sentence can carry, and it matters.

## 1. What was run

24 runs, 20 s wall: 2 operating points × 2 sensory modes × (stimulus, silence control) × 3 noise seeds.

| | A | B |
|---|---|---|
| `adapt_g_b` | **8.0** | **1.51572** |
| why | the value the M2f rule selected; out-of-sample robust 5/5 | the candidate with the largest minimum margin (0.0599 vs 0.0419) |

Everything else is identical and nothing was re-selected: `synapse="conductance"`, `g` = 5.36539e-4,
`noise_sigma` = 29.13, `E_adapt` = −75 mV, `τ_a` = 100 ms, other neuron parameters at M2 defaults. Stimulus
is M3's click train (IPI 35 ms, 15 pulses of 10 ms at 200 Hz, 100 ms silence + 500 ms stimulus + 400 ms
tail), `a_in` = 640 (rate) / 2560 (phase-lock) exactly as M3 chose them, window 100–600 ms, noise seeds
0/1/2 for stimulus and control alike.

**The control is not silent any more.** With noise on, every probe set has a background rate of
0.16–0.83 Hz, so every number below is a *difference* between the stimulus runs and control runs at the
same operating point, same mode, same seeds.

## 2. Result

| point | b_g | mode | JO_post L diff | JO_post R diff | transfer (≥1 Hz both sides) | hop-3 sets reached | pC1 |
|---|---|---|---|---|---|---|---|
| A | 8.0 | rate | +0.412 | +0.121 | **no** | none | **no** |
| A | 8.0 | phase-lock | +0.622 | +0.182 | **no** | none | **no** |
| B | 1.51572 | rate | +0.518 | +0.121 | **no** | none | **no** |
| B | 1.51572 | phase-lock | **+1.169** | +0.331 | **no** (R short) | none | **no** |

The stimulus unquestionably goes in — `JO_AB` is +3.8 Hz (rate, A) to +31.1 Hz (phase-lock, B) above
control, tens of standard deviations. It is detectable one hop later on the left side in all four
configurations. It is gone by hop 2.

### Per-set detail, point B / phase-lock (the strongest of the four)

| set | control mean ± sd (Hz) | stimulus mean (Hz) | diff (Hz) | 3 sd threshold | > 3 sd |
|---|---|---|---|---|---|
| JO_AB_L | 0.202 ± 0.040 | 31.254 | **+31.053** | 0.121 | **YES** |
| JO_AB_R | 0.246 ± 0.249 | 31.281 | **+31.035** | 0.746 | **YES** |
| JO_post_L | 0.638 ± 0.070 | 1.807 | **+1.169** | 0.211 | **YES** |
| JO_post_R | 0.645 ± 0.076 | 0.975 | **+0.331** | 0.227 | **YES** |
| SAD_L | 0.309 ± 0.064 | 0.302 | −0.007 | 0.191 | no |
| SAD_R | 0.386 ± 0.043 | 0.390 | +0.005 | 0.129 | no |
| WED_L | 0.514 ± 0.051 | 0.490 | −0.024 | 0.154 | no |
| WED_R | 0.502 ± 0.101 | 0.527 | +0.025 | 0.304 | no |
| AMMCtype_L | 0.387 ± 0.128 | 0.414 | +0.027 | 0.383 | no |
| AMMCtype_R | 0.355 ± 0.086 | 0.376 | +0.021 | 0.258 | no |
| pC1_L | 0.752 ± 0.316 | 0.692 | −0.060 | 0.949 | no |
| pC1_R | 0.829 ± 0.371 | 0.709 | −0.120 | 1.113 | no |
| rest | 0.418 ± 0.093 | 0.371 | −0.047 | 0.279 | no |

Hop-2/3 differences across all four configurations run from **−0.120 to +0.111 Hz** against control
standard deviations of 0.015–0.371 Hz. Several are negative. Nothing anywhere exceeds 3 sd.

**The verdict does not depend on the conservative threshold.** The brief's test compares a difference of
means against 3 × the *per-run* standard deviation. The statistically natural comparison for a difference
of means would use the standard error, 3 × sd/√3 — a factor 1.73 easier. Recomputed that way, **still
nothing crosses** at hop 2 or beyond in any configuration (largest: WED_L at point A / phase-lock,
+0.104 Hz against a 0.170 Hz SEM threshold).

## 3. Comparison with the no-adaptation runs

| stage | M3 (current, silent) | M3b/M4b (conductance, silent, no adaptation) | **M2g (conductance + adaptation + noise)** |
|---|---|---|---|
| JO_AB → JO_post | transfer PASS at a_in 640 / 2560 | PASS (rate, a_in 320) | **partial** — left side yes, right side short of 1 Hz |
| SAD | 0 Hz | 0.053 / 0.076 Hz | **not detectable** (diff ≤ 0.035, floor 0.13–0.19) |
| WED | 0 Hz | 0.120 / 0.167 Hz | **not detectable** (diff ≤ 0.104, floor 0.15–0.30) |
| pC1 | **0** | **0** | **0** (diff −0.120 … +0.111, never above floor) |

`pC1` has now been zero in every phase of this project: the current model, the conductance model, and both
adaptation operating points.

## 4. Ignition

Applied exactly as pre-registered (late-window active fraction ≥ 1 %), **all 24 runs are flagged**,
including all 12 silence controls. That is not runaway activity — it is the criterion meeting the
background state it was pointed at. These operating points hold 25–28 % of neurons active by construction;
that is what "usable background state" means in M2e/M2f. A criterion written for a silent background cannot
separate a stimulus-driven runaway from the background here, and **the controls failing it is the proof**.

The companion measure — the same window, the brief's own difference logic — shows **no stimulus-driven
excess in any configuration**:

| point | mode | control late-active | stimulus late-active | diff | 3 sd | excess |
|---|---|---|---|---|---|---|
| A | rate | 6.330 % | 7.124 % | +0.794 % | 2.983 % | no |
| A | phase-lock | 10.514 % | 9.857 % | −0.657 % | 6.157 % | no |
| B | rate | 6.347 % | 5.735 % | −0.612 % | 3.702 % | no |
| B | phase-lock | 9.074 % | 9.444 % | +0.370 % | 3.356 % | no |

So no run ignited in the sense the criterion was meant to catch. The pre-registered count is reported as
24/24 because that is what the rule says; the companion is reported because the rule cannot answer the
question here. Neither was used to change any other verdict.

## 5. Verdict against the pre-registered decision rules

- *"pC1 zero at both operating points → that is the result, report it without adjusting parameters."*
  **pC1 is zero at both.** No parameter was touched.
- *"If only one of A/B reaches hop 3, propose it."* Neither does.
- *"If both reach, propose the better signal-to-noise."* Not applicable.
- *"If neither reaches, propose nothing and record that adaptation makes a background state but does not
  extend propagation distance."* → **This is the outcome. No operating point is proposed.**

If a tie-breaker is ever needed for another reason, B is ahead on the only stage where anything is
measurable: it is the only configuration whose hop-1 transfer clears 3 sd on **both** sides, and its
`JO_post_L` difference (+1.169 Hz) is the largest of the four. That is an observation, not a proposal.

## 6. The caveat that limits how far §5 generalises

**The background raises the detection floor above the signal that was previously visible at hop 2.**

| set | signal without adaptation (M3b, silent control) | M2g 3 sd detection floor |
|---|---|---|
| SAD_L | 0.053 Hz | 0.143 – 0.191 Hz |
| SAD_R | 0.076 Hz | 0.044 – 0.170 Hz |
| WED_L | 0.120 Hz | 0.152 – 0.295 Hz |
| WED_R | 0.167 Hz | 0.241 – 0.304 Hz |

Without adaptation the control was *exactly* zero, so a hop-2 response of 0.05–0.17 Hz was plainly visible.
With the background state, the floor this protocol can see past is 0.13–0.30 Hz — **at or above those same
magnitudes**. So for hop 2 this experiment cannot distinguish *"adaptation did not help"* from *"the signal
is still arriving at its old size and is now buried"*. The honest statement is the narrow one: **no hop-2 or
hop-3 arrival is detectable above the background these operating points create.**

For hop 3 the ambiguity does not arise: `pC1` was 0.000 with a silent control and no adaptation, so there
was never a signal for the background to bury.

If the planning session wants to settle the hop-2 question, the design needs more power rather than a new
operating point — more seeds (the floor falls as 1/√n), a longer stimulus window, or a lower σ. That is a
new pre-registration, not something to adjust here.

## 7. Completion criteria (`docs/m2g-brief.md`)

- [x] 24 runs completed (A·B × 2 modes × stimulus/control × 3 seeds); ignition flags reported (§4)
- [x] per-set (stimulus − control) tables with the hop-3 judgement (§2, full detail in the JSON)
- [x] side-by-side comparison with the no-adaptation results (§3)
- [x] **no proposal**, with the pre-registered sentence recorded (§5)
- [x] `pytest tests/` passes — 338 passed
- [x] one "M2g: …" commit, not pushed

## 8. Deviations from the brief

1. **A companion ignition measure was added** because the pre-registered absolute rule flags all 24 runs,
   controls included, at operating points whose whole purpose is a 25–28 % active background (§4). The
   pre-registered rule is still applied and reported as written; the companion is additional, and neither
   changes any other verdict.
2. **The SEM reading is reported alongside the 3 sd test** (§2) to show the negative result is not an
   artifact of the conservative threshold. The pre-registered 3 sd test is the one that decides.
3. Nothing else. `a_in`, `g`, `σ`, the operating points and the four judgement criteria are exactly as
   written; no new operating point was searched; `m3_click.py`, `probe/`, `sensory/`, the viewer and the
   server were not modified.

## 9. Notes for the planning session

- **The question the adaptation work set out to answer now has an answer, and it is negative.** Adaptation
  fixed the voltage bound (M2d), widened the gain range (M2d/M2e) and produced a robust low-rate background
  (M2f). It did **not** make the click train travel further. Hop 2 was the wall before adaptation and it is
  the wall after.
- **pC1 has been zero in every configuration this project has run.** At some point that is worth treating as
  a statement about the wiring-only model rather than about parameters.
- **The hop-2 question is now under-powered, not closed** (§6). Settling it needs more seeds or a longer
  window, not a new operating point.
- **The ignition criterion needs restating for noisy backgrounds** if it is to be used again — as an excess
  over control, which is what every other criterion in this brief already does.
- `parameter-decisions.md` and `backlog.md` were not edited (planning-owned).
