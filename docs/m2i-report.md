# M2i report — there is no like-for-like control, and that is the answer

Engine session (window "M3 엔진"), 2026-09-19. Pre-registration: `docs/m2i-brief.md`, written before this
ran and not edited afterwards. **A check: it cannot change M2h's conclusion, and it does not.** Code:
`flysim/apps/m2i_control.py`. Raw results: `data-provenance/m2i-results.json`. 160 runs, 481 s.

**Conclusion rule 1 applies.** With adaptation off at M2h's own gain, the network ignites at all three of
M2h's σ. "The same conditions with adaptation switched off" does not exist, so M2h's "5–7× attenuation"
sentence was comparing two operating points, not measuring an effect of adaptation. §5 proposes the
replacement wording.

The σ = 0 arm says something stronger than expected, and §4 has it.

## 1. What was run

Adaptation **off** (`adapt_g_b = 0`), conductance synapses, **`g` = 5.36539e-4 — M2h's value**, `a_in` 640
(rate) / 2560 (phase-lock), σ ∈ {29.13, 26.21, 23.30, 0}, seeds 0–9, 2000 ms stimulus window. Seeds,
window, statistic, sets and the redefined ignition test are M2h's, unchanged. 2 modes × 4 σ ×
(stimulus, control) × 10 seeds = 160 runs.

## 2. Result: the no-adaptation network is ignited at every σ M2h used

Absolute state of the **control** runs — no stimulus at all, just noise:

| σ | mode | control active | control network rate (Hz) | control late-active | ignited? |
|---|---|---|---|---|---|
| 29.13 | rate | 44.0 % | **24.83** | 27.5 % | **yes** |
| 29.13 | phase-lock | 46.9 % | **32.55** | 29.5 % | **yes** |
| 26.21 | rate | 36.7 % | **24.20** | 24.6 % | **yes** |
| 26.21 | phase-lock | 39.6 % | **32.02** | 26.8 % | **yes** |
| 23.30 | rate | 31.5 % | **23.56** | 22.6 % | **yes** |
| 23.30 | phase-lock | 34.5 % | **31.39** | 24.8 % | **yes** |
| 0 | rate | 0.0 % | 0.000 | 0.0 % | no — silent |
| 0 | phase-lock | 0.0 % | 0.000 | 0.0 % | no — silent |

"Ignited" here is M2b's own criterion — ≥ 1 % of neurons active in the post-offset window, self-sustained.
These runs exceed it by a factor of 20–30, at 24–33 Hz per neuron across the whole network. This is the
saturated state M2b documented for σ ≥ 22 without adaptation, and M2h's σ range sits entirely inside it —
at a gain 1.7× higher than the one M2b measured, so if anything the prediction was conservative.

The contrast with the adapted runs at the identical σ is the whole story:

| σ | control network rate, no adaptation | control network rate, with adaptation (M2h, point A / B) |
|---|---|---|
| 29.13 | 24.83 Hz | 0.262 / 0.345 Hz |
| 26.21 | 24.20 Hz | 0.154 / 0.227 Hz |
| 23.30 | 23.56 Hz | 0.115 / 0.158 Hz |

**About 100× lower with adaptation, at the same gain and the same noise.** The low-rate background M2f
went looking for is not a property of this gain and noise; it is a property of adaptation.

**→ Record it as the brief specifies: at this gain and noise, the background state exists only with
adaptation.**

## 3. The redefined ignition test missed all of this — a false negative

The redefined test (excess over control ≥ 3 SEM **and** stimulus ≥ 90 % active) returned **False in all
six noise cells**, while the network sat at 24–33 Hz. Both terms fail for the same reason: the control
*shares* the runaway, so the excess is ≈ 0, and a saturated network here is 31–47 % active, not 90 %.

This was predicted before the run (module docstring) and is reported as a limitation, not worked around.
M2h's redefinition genuinely fixed the false positive it was written for — a background state being read
as a runaway. It cannot detect a runaway that the control is already in. **Recommendation: keep the
redefined test for stimulus-driven ignition, and pair it with an absolute check on the control's own
state** (network mean rate, or active fraction against the pre-stimulus baseline). Either of the two
columns in §2 would have caught this immediately.

## 4. σ = 0: the gain change alone is enormous, and in the opposite direction

Same gain, no noise, no adaptation, so the only difference from M3b is `g` (5.36539e-4 vs 3.162e-4).
Stimulus-window rates, rate mode, `a_in` = 640 in both:

| set | M3b, g = 3.162e-4 | M2i, g = 5.36539e-4 | factor |
|---|---|---|---|
| JO_post_L | 9.371 | 40.394 | 4× |
| JO_post_R | 2.529 | 43.426 | 17× |
| SAD_L | 0.053 | 8.725 | **165×** |
| SAD_R | 0.076 | 13.168 | **173×** |
| WED_L | 0.120 | 46.378 | **386×** |
| WED_R | 0.167 | 49.109 | **294×** |
| pC1_L | 0.000 | 10.051 | 0 → **10.05** |
| pC1_R | 0.000 | 12.314 | 0 → **12.31** |

And the state: the M2i σ = 0 stimulus run is **19.9 % active, 21.9 Hz, 15.0 % still active after stimulus
offset — ignited**, where M3b at the lower gain was a clean non-ignited PASS with zero late activity.

So the gain change on its own converts an input-driven response into a stimulus-triggered runaway. This is
the first configuration in the project where pC1 is non-zero, and it is non-zero **because the network is
in a runaway**, not because a signal arrived — the same reason M4b's pC1 > 0 runs were recorded as failures.

**This means M2h's confound ran opposite to the direction assumed.** Without adaptation the higher gain
gives far *more* downstream activity, not less. So "adaptation attenuates hop 2 by 5–7×" understates the
confound and mislabels it: the adapted run was not a weakened version of the unadapted one at that gain,
because the unadapted one at that gain is not a signal at all.

## 5. Proposed replacement for the M2h §4 rate-mode paragraph

**Not applied — `docs/m2h-report.md` is left exactly as it was**, since it is the record of what was known
then. The planning session asked for a paragraph; here it is, as a drop-in for the "rate mode, same
`a_in`" block of M2h §4:

> **Rate mode: the comparison is between two operating points, not a measurement of adaptation.** The
> no-adaptation reference (`m3b`, WED 0.120 / 0.167 Hz) was taken at `g` = 3.162e-4, while these runs use
> `g` = 5.36539e-4. M2i ran the missing arm — adaptation off at 5.36539e-4, everything else identical —
> and found that no like-for-like control exists: at all three σ used here the unadapted network is
> ignited (31–47 % active, 24–33 Hz, self-sustained after offset), and even at σ = 0 the click train alone
> drives it into a runaway (19.9 % active, 15 % still active after offset), where the same stimulus at
> 3.162e-4 was a clean non-ignited response. The "5–7× smaller" figure therefore compares an adapted run
> against a *different, lower-gain* operating point and should not be read as adaptation weakening the
> signal: at this gain, without adaptation, there is no low-rate regime to weaken. What the two arms
> jointly show is narrower and firmer — **at `g` = 5.36539e-4 with σ 23–29, a usable background state
> exists only with adaptation**, and within that state hop 2 is reached in phase-lock and not in rate.

## 6. Which conclusion rule applies

- *"If it ignites at σ 23.3–29.13 → no single-variable control exists; record that at this gain and noise
  the background state requires adaptation, and restate the 5–7× sentence as a comparison between two
  operating points."* → **This one** (§2, §5).
- *"If it does not ignite → that is the single-variable control."* Not selected; it ignites at all six.
- *"The σ = 0 result is written separately either way."* → §4.
- *"pC1 is recorded at face value either way."* → pC1 is 0 at σ ≥ 23.3 in every no-adaptation cell that is
  not saturated, and 10–12 Hz at σ = 0 **inside a runaway**, which is not an arrival (§4).

## 7. What this does and does not change

| claim | status |
|---|---|
| M2h: hop 2 preserved in phase-lock, WED in 6/6 cells | **unchanged.** M2i ran no adapted cells and could not affect it |
| M2h: hop 3 not reached | **unchanged** |
| M2h: "adaptation attenuates hop 2 by 5–7× in rate mode" | **withdrawn as stated**; replacement in §5 |
| M2f: the background state at this (g, σ) | **strengthened** — it is now shown to require adaptation, not merely to coexist with it |

## 8. Completion criteria (`docs/m2i-brief.md`)

- [x] 160 runs completed; ignition judged per σ, on the absolute state as well as the redefined test (§2, §3)
- [x] which conclusion rule applies, stated explicitly (§6)
- [x] σ = 0 hop-2 sizes next to M3b, isolating the gain change (§4)
- [x] a paragraph proposed for the M2h sentence, not applied (§5)
- [x] `pytest tests/` passes — 338 passed
- [x] one "M2i: …" commit, not pushed

## 9. Deviations from the brief

1. **Absolute-state diagnostics were added** (control active fraction and network mean rate) because the
   brief's inherited ignition test cannot answer the brief's own question when the control shares the
   runaway (§3). The redefined test is still computed and reported as specified; the absolutes are extra.
2. Nothing else. `g`, `a_in`, σ, seeds, window, statistic, sets and the ignition definition are exactly as
   written; no new operating point was searched; `docs/m2h-report.md`, other windows' files and this brief
   were not modified.

## 10. Notes for the planning session

- **The adaptation branch now has a clean joint statement.** At `g` = 5.36539e-4 with σ 23–29 a usable
  low-rate background exists *only* with adaptation (M2i); inside that background the click train reaches
  hop 2 in phase-lock and not in rate (M2h); and it never reaches pC1 (M2g, M2h).
- **The hop 2 → 3 wall survives every configuration tried**, including the one place pC1 was non-zero,
  which was a runaway rather than an arrival (§4).
- **The ignition test needs its third form**: absolute-only was wrong (M2g), excess-only is wrong (M2i).
  Excess for stimulus-driven ignition, plus an absolute check on the control itself.
- **Gain and adaptation are strongly coupled** — a 1.7× gain change moves WED by ~300× without adaptation.
  Any future comparison across operating points should expect that and hold `g` fixed.
- `parameter-decisions.md` and `backlog.md` were not edited (planning-owned).
