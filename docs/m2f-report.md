# M2f report — new selection rule, validated out of sample

Engine session (window "M3 엔진"), 2026-09-19. Pre-registration: `docs/m2f-brief.md` (written before this
run; not edited afterwards). Raw results: `data-provenance/m2f-selection.json`, `m2f-validation.json`,
`m2f-width.json`, `m2f-bench-precondition.json`.

**Only the selection rule changed.** The four usable-state conditions — rate 0.1–5 Hz, ratio 0.5–2,
tail < 1 %, active fraction ≤ 30 % — are the M2d ones, reused from the same constants and the same
`_judge_usable()`. What changed is which of several passing points becomes the recorded default.

**Result in one line:** `DEFAULT_ADAPT_G_B` = **8.0**, `DEFAULT_G_WITH_ADAPT` = 5.36539e-4, and the
out-of-sample check on five never-used seeds passes **5 / 5** — the first robust operating point in this
series. Two caveats in §3 and §6 that matter more than the number.

## 1. The rule, and one interpretation I had to fix before running

New rule: *the smallest candidate `b_g` whose **minimum margin** is at least **3 ×** the seed-to-seed
**spread***. Candidates are exactly the ten `b_g` that produced a usable state in M2e — no new grid. The
multiplier was fixed before the run.

**The four margins are in four different units** (Hz, a dimensionless ratio, and two fractions), so "the
minimum of the four" is not well defined until they are on a common scale, and the choice of scale changes
which `b_g` is selected. Decided and committed before running, on general grounds rather than by looking at
what it picks: **each margin is normalised by the width of its own condition's allowed band**, giving a
dimensionless "fraction of the band still in hand".

| condition | band | width used |
|---|---|---|
| rate | [0.1, 5] Hz | 4.9 |
| ratio | [0.5, 2] | 1.5 |
| tail | [0, 1 %] | 0.01 |
| active fraction | [0, 30 %] | 0.30 |

The spread is the max − min of the **active fraction** over seeds 0, 3, 4, 5, put on the same normalised
scale — as the brief specifies.

## 2. Selection

40 runs (10 candidates × 4 seeds) at `g` = 5.365e-4, σ = 29.13. The seed-0 values reproduce M2e's exactly.

| b_g | min margin | binding condition | spread | need ≥ 3 × spread | seeds usable | rule |
|---|---|---|---|---|---|---|
| 0.65793 | 0.0088 | active fraction | 0.0122 | 0.0365 | 2 / 4 | fail |
| 0.73053 | 0.0143 | active fraction | 0.0175 | 0.0524 | 2 / 4 | fail |
| 0.81113 | 0.0193 | active fraction | 0.0226 | 0.0678 | 4 / 4 | fail |
| 0.90063 | 0.0309 | active fraction | 0.0253 | 0.0760 | 4 / 4 | fail |
| 1.00000 | 0.0291 | active fraction | 0.0223 | 0.0669 | 3 / 4 | fail |
| 1.51572 | **0.0599** | rate (lower) | 0.0282 | 0.0845 | 4 / 4 | fail |
| 2.29740 | 0.0497 | rate (lower) | 0.0308 | 0.0924 | 4 / 4 | fail |
| 3.48220 | 0.0435 | rate (lower) | 0.0238 | 0.0714 | 3 / 4 | fail |
| 5.27803 | 0.0438 | rate (lower) | 0.0215 | 0.0644 | 4 / 4 | fail |
| **8.00000** | 0.0419 | rate (lower) | **0.0100** | **0.0301** | 4 / 4 | **PASS** |

**`DEFAULT_ADAPT_G_B` = 8.0**, the only candidate that satisfies the rule.

A structural finding the normalisation exposes: **the binding condition changes with `b_g`.** Below ≈ 1.5
it is the active fraction (the network is too active); above it the rate has fallen towards the 0.1 Hz
floor and *that* becomes binding. So the minimum margin is not monotonic — it peaks at `b_g` ≈ 1.52
(0.0599) and declines on both sides. Adaptation can be too strong as well as too weak, and this is the
first run in the series that could see it, because M2d/M2e only ever looked at the active fraction.

## 3. Caveat: 8.0 passes on the smallest spread, not the largest margin

`b_g` = 8.0 does **not** have the best margin — 1.51572 does (0.0599 vs 0.0419). It passes because its
measured spread is the smallest on the grid (0.0100 against 0.0215–0.0308 elsewhere). A max − min over
four samples is a noisy estimator of dispersion, so this could be luck rather than a property of the cell.
**That is exactly what the out-of-sample check is for**, and §4 settles it.

It is also worth saying plainly: **8.0 is the largest candidate tested.** Nothing above it was run, and the
brief forbids a new grid. So the rule's answer is again at the edge of the tested range, in the sense that
the smallest `b_g` satisfying the rule lies in **(5.27803, 8.0]** and the rule's behaviour above 8.0 is
unknown. Unlike M2d, this is a bounded statement rather than an unresolved one: every smaller candidate was
tested and failed.

## 4. Out-of-sample validation — **robust, 5 / 5**

Seeds 6, 7, 8, 9, 10, never used to choose the rule or the cell:

| seed | rate first → last (Hz) | active last s | ratio | tail | v min | verdict |
|---|---|---|---|---|---|---|
| 6 | 0.352 → 0.275 | 24.859 % | 0.78 | 0.002 % | −91.9 | usable |
| 7 | 0.332 → 0.292 | 25.943 % | 0.88 | 0.004 % | −91.3 | usable |
| 8 | 0.328 → 0.290 | 25.240 % | 0.88 | 0.002 % | −91.5 | usable |
| 9 | 0.338 → 0.286 | 25.595 % | 0.85 | 0.001 % | −91.8 | usable |
| 10 | 0.336 → 0.311 | 25.841 % | 0.93 | 0.027 % | −90.7 | usable |

**All five pass → `robust: true`**, recorded in `m2f-validation.json` and bound to
`DEFAULT_ADAPT_G_B_ROBUST = True` by a test. Across all nine seeds ever run at this cell (0, 3, 4, 5 and
6–10) every single one is usable, with the active fraction between 24.86 % and 25.94 % against a 30 % cap —
**4.06 points of headroom in the worst case**, versus M2e's selected cell which had 0.263 points and failed
2 of 3.

## 5. Width of the background state (descriptive; does not affect the selection)

M2e's ten usable runs all sat at one (`g`, σ) point, so the state might have been a knife edge. Scanned
5 × 5 around it at `b_g` = 8.0, seed 0. Active fraction, and usable / not:

| g \ σ | ×0.8 (23.3) | ×0.9 (26.2) | ×1.0 (29.13) | ×1.1 (32.0) | ×1.25 (36.4) |
|---|---|---|---|---|---|
| ×0.80 | 7.12 % ✓ | 11.17 % ✓ | 21.28 % ✓ | 38.54 % ✗ | 62.20 % ✗ |
| ×0.90 | 7.22 % ✗ | 10.28 % ✓ | 23.51 % ✓ | 39.48 % ✗ | 62.96 % ✗ |
| ×1.00 | 11.08 % ✓ | 14.57 % ✓ | **25.40 % ✓** | 40.64 % ✗ | 64.07 % ✗ |
| ×1.10 | 11.90 % ✓ | 17.43 % ✓ | 27.84 % ✓ | 42.49 % ✗ | 64.88 % ✗ |
| ×1.25 | 13.65 % ✓ | 18.60 % ✓ | 29.69 % ✓ | 44.18 % ✗ | 66.18 % ✗ |

**14 of 25 usable — it is a region, not a knife edge.** It spans the **whole `g` range scanned** (×1.56,
4.29e-4 … 6.71e-4) and σ from ×0.8 to ×1.0 (×1.25, 23.3 … 29.1). The two boundaries have different causes:
above σ ×1.0 the **active fraction** exceeds 30 % (38–66 %), and at the bottom-left corner the **rate**
falls below 0.1 Hz (the single ✗ at g ×0.9, σ ×0.8 has rate 0.074 Hz; its neighbour at g ×0.8 has 0.103 Hz,
so that edge is being crossed noisily). The `g` width is a lower bound — the usable band extends past both
ends of the scan — while the σ upper edge is genuinely located, between 29.1 and 32.0.

## 6. Caveat: the rule's arithmetic is fragile, though its verdict held

Two independent weaknesses, neither of which changes the answer but both of which bear on reusing the rule:

**(a) A four-sample range underestimates dispersion.** At `b_g` = 8.0 the in-sample spread (seeds 0/3/4/5)
is 0.301 points; the five out-of-sample seeds span 1.085 points — **3.6 × larger**. Recomputed over all
nine seeds the rule would read `margin 0.0419 ≥ 3 × 0.0362 = 0.1085` → **fail**. The rule selected 8.0 on
an optimistic spread estimate.

**(b) The rule compares quantities of different kinds.** The minimum margin at 8.0 is the *rate* margin,
while the spread is measured on the *active fraction*, as the brief specifies. Comparing them is not
apples to apples. Done apples to apples — the binding condition's margin against that same condition's
own nine-seed spread — it passes comfortably: `0.0419 ≥ 3 × 0.0073 = 0.0218`. The rate margin is in fact
very stable across seeds (0.0358 – 0.0430).

So the verdict "8.0 is robust" is well supported, but by §4 and by (b), not by the rule's own arithmetic in
§2. **The rule is not changed here** — it was applied exactly as pre-registered and its result reported,
together with the evidence about where it is weak. If it is to be reused, the planning session may want the
spread measured on the binding condition and estimated with a standard deviation rather than a range of
four.

## 7. Voltage bound

Cumulative across the series, noise-free stimulus grids (`i_ext ≥ 0`, where the model's bound applies):
**M2d 288 + M2e 264 = 552 runs, zero violations**, `v_min` −74.95 / −74.98 mV against −75.0 mV. M2f added
no stimulus-grid runs — its 80 runs are all noise runs, where `v` is *allowed* below the reversal
potentials because the noise current can be negative (M2b §1, −229 mV at σ = 200 with no adaptation at
all). M2f's noise `v_min` is **−91.9 mV** at σ = 29.13, in line with M2b's −90.8 mV at the same σ with no
adaptation. The adaptation conductance still contributes nothing to the excursion, now confirmed up to
`b_g` = 8.0.

## 8. Bench — not re-run: environment precondition not met

| check | measured | required | met |
|---|---|---|---|
| other compute processes | none but this check's own | none | yes |
| SM clock under sustained load | **465 MHz median** | ≥ 2,795 MHz | **no — 15.0 % of max** |
| throttle reason throughout | `0x4` = SW Power Cap | — | — |
| power draw | ~40 W | — | — |

Recorded in `data-provenance/m2f-bench-precondition.json`. **The < 100 µs target therefore remains
unverified**, as in M2e, and no numbers were fabricated or rescaled. The clock was sampled under load, not
at idle, for the reason given in the M2e report (an idle GPU always downclocks, throttle `0x1` = GpuIdle,
and would fail the test on any machine in any state). The adaptation branch's incremental cost
(+1.5 µs/step, measured pairwise in M2d) is unaffected and was not re-measured.

## 9. Completion criteria (`docs/m2f-brief.md`)

- [x] spread and minimum-margin table for all ten candidates (§2)
- [x] what the new rule selected — **8.0**, with the caveat that it is the largest candidate (§2, §3)
- [x] out-of-sample validation on seeds 6–10 and the `robust` verdict — **5 / 5, true** (§4)
- [x] 25-cell (`g`, σ) width table — **14 / 25 usable, a region** (§5)
- [x] cumulative voltage-bound violations: **zero** (§7)
- [x] `pytest tests/` passes; the default constants and the `robust` flag are bound to the JSON by tests,
      including that the validation seeds are disjoint from the selection seeds (§10)
- [x] one "M2f: …" commit, not pushed

## 10. Tests

`tests/test_engine_adapt_g.py`, 25 cases. The two provenance tests were retargeted at the M2f JSONs:
`test_defaults_match_the_recorded_selection_rule` checks that the constant is the **smallest** passing
candidate, that the multiplier is 3.0, and that all four usable-state conditions still hold their M2d
values; `test_robust_flag_is_bound_to_the_out_of_sample_validation` checks that `robust` means *all* seeds
passed rather than a majority, that the validation seeds are **disjoint** from the seeds used to select,
and that the recorded note still states no substitute cell was sought. Everything else in the file is
unchanged M2d coverage (voltage bound on the real graph, Triton == torch, determinism, dt invariance,
current-model rejection, `adapt_g_b = 0` bit-identity).

`pytest tests/`: **338 passed, 0 failed.**

## 11. Deviations from the brief

1. **Margins normalised per condition** before taking the minimum (§1). Forced: the four margins have four
   different units and the unnormalised minimum is not meaningful. Chosen on general grounds and fixed
   before the run.
2. **A misleading field was corrected mid-run.** The first implementation flagged every noise run as a
   "v bound violation" because it compared `v_min` against the reversal potentials with noise on, where
   the model does not promise that bound. Reporting "40 violations" would have been wrong; the field now
   records only whether a run happened to stay above, with the reason, and the selection output no longer
   claims violations. The genuine bound test is the noise-free stimulus grid (§7).
3. Nothing else. The four conditions, the 3 × multiplier, the candidate list, the spread seeds, the
   validation seeds and the width factors are exactly as written in the brief; no new `b_g` grid was made;
   no substitute cell was sought; the brief was not edited.

## 12. Notes for the planning session

- **There is now a robust operating point**: `b_g` = 8.0, `g` = 5.365e-4, σ = 29.13, usable on all nine
  seeds tried with ~4 points of headroom on the active fraction, sitting inside a region ×1.56 wide in `g`
  and ×1.25 in σ. That is a materially different situation from M2d/M2e.
- **But the rule that found it is weaker than its verdict** (§6). If it is reused, measure the spread on
  the binding condition and with a standard deviation rather than a four-sample range.
- **Adaptation can be too strong.** The binding condition flips from active fraction to rate at
  `b_g` ≈ 1.5, and the minimum margin peaks there (0.0599 at 1.51572, vs 0.0419 at 8.0). A rule that
  maximised the minimum margin instead of taking the smallest qualifying `b_g` would have selected
  ≈ 1.5 — a different and arguably better answer from the same data. Worth deciding deliberately.
- **`b_g` above 8.0 is untested** and the brief forbade extending the grid. If 8.0 is adopted, that edge
  should be noted; if the margin-maximising reading is preferred instead, no new runs are needed.
- `parameter-decisions.md` and `backlog.md` were not edited (planning-owned).
