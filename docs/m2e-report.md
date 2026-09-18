# M2e report — resolving the adaptation-conductance grid boundary

Engine session (window "M3 엔진"), 2026-09-19. Pre-registration: `docs/m2e-brief.md` (written before this
run; not edited after seeing any result). Raw results: `data-provenance/m2e-grid.json`,
`m2e-noise-sweep.json`, `m2e-robustness.json`. Hardware/software as in M2d.

**Nothing about the target changed.** The four usable-state conditions and the selection rule are the M2d
ones, reused from the same constants and the same `_judge_usable()` function — this run only opens the
`b_g` grid in both directions, because M2d's rule returned the top point of its grid and was therefore
unresolved.

## 1. What was pre-registered

M2d selected `b_g = 1.0`, which was the **top** of `B_G_GRID` = 12 log points in [1e-3, 1]. So "the
smallest `b_g` with a usable state" pointed at the grid edge: the true minimum could lie inside
(0.534, 1.0], and nothing above 1.0 had been tested. The single hit also cleared the ≤ 30 % active-fraction
condition by only 0.87 points.

Fixed in `sweep_conductance.py` before running:

| item | value | changed from M2d? |
|---|---|---|
| `B_G_GRID_REFINE` | 5 geometric points strictly inside [0.53367, 1.0] → 0.59255, 0.65793, 0.73053, 0.81113, 0.90063 | new |
| `B_G_GRID_EXTEND` | 6 geometric points on [1.0, 8.0] → 1.0, 1.51572, 2.29740, 3.48220, 5.27803, 8.0 | new |
| `G_GRID_ADAPT_G` | 8 log points in [2e-4, 2e-3] | **identical** |
| stimulus seeds | 0, 1, 2 | **identical** |
| usable state | rate 0.1–5 Hz, ratio 0.5–2, tail < 1 %, **active ≤ 30 %** | **identical** |
| selection rule | smallest `b_g` with a usable state; `g` = geometric mean of the usable `g` at that `b_g` | **identical** |
| `ROBUSTNESS_SEEDS` | 3, 4, 5 — descriptive, not part of the rule | new |

11 `b_g` × 8 `g` × 3 seeds = **264 grid runs**, then the noise sweep over every all-seed-RESPONSIVE cell.

## 2. Combined (b_g × g) grid

M2b's stimulus protocol and `judge_g`, unmodified. `S` = SILENT for all seeds, `R` = RESPONSIVE for all
seeds, `I` = ignited for at least one seed, `·` = mixed:

| b_g \ g | 2.000e-04 | 2.779e-04 | 3.861e-04 | 5.365e-04 | 7.455e-04 | 1.036e-03 | 1.439e-03 | 2.000e-03 |
|---|---|---|---|---|---|---|---|---|
| 0.59255 | S | · | · | **R** | **R** | **R** | **R** | I |
| 0.65793 | S | · | · | **R** | **R** | **R** | **R** | **R** |
| 0.73053 | S | · | · | **R** | **R** | **R** | **R** | **R** |
| 0.81113 | S | · | · | **R** | **R** | **R** | **R** | **R** |
| 0.90063 | S | · | · | **R** | **R** | **R** | **R** | **R** |
| 1.00000 | S | · | · | **R** | **R** | **R** | **R** | **R** |
| 1.51572 | S | · | · | **R** | **R** | **R** | **R** | **R** |
| 2.29740 | S | · | · | **R** | **R** | **R** | **R** | **R** |
| 3.48220 | S | · | · | **R** | **R** | **R** | **R** | **R** |
| 5.27803 | S | · | · | **R** | **R** | **R** | **R** | **R** |
| 8.00000 | S | · | · | **R** | **R** | **R** | **R** | **R** |

**54 of 88 cells** are RESPONSIVE for all three seeds (M2d: 26 of 96). Above `b_g` ≈ 0.66 the RESPONSIVE
band is the whole upper `g` range, 5.365e-4 … 2.0e-3 — a factor of 3.7 — and it does not widen further with
`b_g`: the band is bounded below by the SILENT/mixed region at `g` ≤ 3.861e-4, which adaptation does not
move because a silent network never accumulates `g_a`.

**Voltage bound: no violation in any of the 264 runs**, `v_min` = **−74.98 mV** against the −75.0 mV bound,
`v_max < v_thresh` everywhere. Combined with M2d's 288 runs that is **552 runs with zero violations**,
including `b_g` up to 8.0 — eight times the largest value M2d tested. The bound is structural, as §1 of the
M2d report argued, and raising `b_g` by another order of magnitude does not threaten it.

## 3. Noise sweep

All 54 all-seed-RESPONSIVE cells × the M2b σ grid: **648 runs, 266 s**. The four conditions and
`_judge_usable()` are M2d's, untouched. `Y` = usable, `a` = passes conditions 1–3 and fails **only** the
active-fraction one, `·` = fails something else. σ ≤ 11.1 is silent in every cell and is omitted.

| b_g | g | 18.0 | 29.1 | 47.1 | 76.3 | 123.6 | 200.0 |
|---|---|---|---|---|---|---|---|
| 0.59255 | 5.37e-04 | · | a | a | · | a | · |
| 0.59255 | 7.46e-04 | · | a | a | a | · | · |
| 0.59255 | 1.04e-03 | · | a | a | a | · | · |
| 0.59255 | 1.44e-03 | · | a | a | a | · | · |
| 0.65793 | 5.37e-04 | · | **Y** | a | a | a | · |
| 0.65793 | 7.46e-04 | · | a | a | · | a | · |
| 0.65793 | 1.04e-03 | · | a | a | a | · | · |
| 0.65793 | 1.44e-03 | · | a | · | a | · | · |
| 0.65793 | 2.00e-03 | · | · | · | · | · | · |
| 0.73053 | 5.37e-04 | · | **Y** | a | a | a | · |
| 0.73053 | 7.46e-04 | · | a | a | a | a | · |
| 0.73053 | 1.04e-03 | · | · | a | a | a | · |
| 0.73053 | 1.44e-03 | · | a | · | a | · | · |
| 0.73053 | 2.00e-03 | · | · | · | · | · | · |
| 0.81113 | 5.37e-04 | · | **Y** | a | a | a | · |
| 0.81113 | 7.46e-04 | · | a | a | a | a | · |
| 0.81113 | 1.04e-03 | · | a | a | a | a | · |
| 0.81113 | 1.44e-03 | · | a | · | a | · | · |
| 0.81113 | 2.00e-03 | · | a | · | a | · | · |
| 0.90063 | 5.37e-04 | · | **Y** | a | a | a | · |
| 0.90063 | 7.46e-04 | · | · | a | a | a | · |
| 0.90063 | 1.04e-03 | · | a | a | a | a | · |
| 0.90063 | 1.44e-03 | · | a | a | a | a | · |
| 0.90063 | 2.00e-03 | · | · | · | a | · | · |
| 1.00000 | 5.37e-04 | · | **Y** | a | a | a | · |
| 1.00000 | 7.46e-04 | · | a | a | a | a | · |
| 1.00000 | 1.04e-03 | · | · | a | a | a | · |
| 1.00000 | 1.44e-03 | · | · | a | a | a | · |
| 1.00000 | 2.00e-03 | · | a | a | a | · | · |
| 1.51572 | 5.37e-04 | · | **Y** | a | a | a | · |
| 1.51572 | 7.46e-04 | · | · | a | a | a | · |
| 1.51572 | 1.04e-03 | · | a | · | a | a | · |
| 1.51572 | 1.44e-03 | · | a | a | a | a | · |
| 1.51572 | 2.00e-03 | · | a | a | a | a | · |
| 2.29740 | 5.37e-04 | · | **Y** | a | a | a | a |
| 2.29740 | 7.46e-04 | · | a | a | a | a | a |
| 2.29740 | 1.04e-03 | · | a | a | a | a | a |
| 2.29740 | 1.44e-03 | · | a | a | a | a | a |
| 2.29740 | 2.00e-03 | · | a | a | a | a | a |
| 3.48220 | 5.37e-04 | · | **Y** | a | a | a | a |
| 3.48220 | 7.46e-04 | · | a | a | a | a | a |
| 3.48220 | 1.04e-03 | · | a | a | a | a | a |
| 3.48220 | 1.44e-03 | · | a | a | a | a | a |
| 3.48220 | 2.00e-03 | · | a | a | a | a | a |
| 5.27803 | 5.37e-04 | · | **Y** | a | a | a | a |
| 5.27803 | 7.46e-04 | · | a | a | a | a | a |
| 5.27803 | 1.04e-03 | · | a | a | a | a | a |
| 5.27803 | 1.44e-03 | · | a | a | a | a | a |
| 5.27803 | 2.00e-03 | · | a | a | a | a | a |
| 8.00000 | 5.37e-04 | · | **Y** | a | a | a | a |
| 8.00000 | 7.46e-04 | · | a | a | a | a | a |
| 8.00000 | 1.04e-03 | · | a | a | a | a | a |
| 8.00000 | 1.44e-03 | · | a | a | a | a | a |
| 8.00000 | 2.00e-03 | · | a | a | a | a | a |

**10 usable runs out of 648**, every one of them at `g` = 5.365e-4 and σ = 29.13, one per `b_g` from
0.65793 upward. No other `g` and no other σ ever produces a usable state at any `b_g` on this grid.

## 4. Selection — the boundary is resolved

Rule, unchanged from M2d: *smallest `b_g` with a usable state; `g` = geometric mean of the `g` usable at
that `b_g`.*

**`DEFAULT_ADAPT_G_B` = 0.65793, `DEFAULT_G_WITH_ADAPT` = 5.36539e-4.**

**This is an interior point, not a grid edge** — which is what M2e was pre-registered to determine. It lies
strictly inside the refinement range [0.53367, 1.0], with a tested point below it (0.59255) that produces
no usable state and eight tested points above it that do. The answer is therefore bracketed:
**the true minimum lies in (0.59255, 0.65793]**, and the rule is resolved in the sense M2d's was not.

M2d's answer of 1.0 was an artefact of that grid stopping there. The refinement moved the minimum down by
a factor of 1.52; the extension to `b_g` = 8.0 showed nothing new below it, so opening the top did not
change the selection.

Cross-check: the `b_g` = 1.0, `g` = 5.365e-4, σ = 29.13 cell reproduces M2d's numbers exactly
(0.452 → 0.444 Hz, 29.128 % active, tail 0.001 %, v −91.2 mV), as it must — same seed, same parameters.

## 5. Robustness check — **failed**

Pre-registered as descriptive, not part of the selection rule: re-run the selected cell under noise seeds
3, 4, 5 and report "not robust" if any fails, **without looking for a substitute cell**.

| noise seed | rate first → last (Hz) | **active last s** | ratio | tail | v min | verdict |
|---|---|---|---|---|---|---|
| 0 (primary) | 0.502 → 0.470 | **29.737 %** | 0.93 | 0.074 % | −91.1 | usable |
| 3 | 0.508 → 0.475 | **30.069 %** | 0.93 | 0.073 % | −90.3 | **fails** (active fraction) |
| 4 | 0.501 → 0.474 | **29.954 %** | 0.95 | 0.009 % | −90.6 | usable |
| 5 | 0.510 → 0.476 | **30.101 %** | 0.93 | 0.001 % | −91.2 | **fails** (active fraction) |

**Result: not robust — 2 of 3 seeds fall outside.** `data-provenance/m2e-robustness.json` records
`robust: false`, and `flysim/engine/params.py` carries `DEFAULT_ADAPT_G_B_ROBUST = False` so nothing can
adopt the value without meeting the failure. **No substitute cell was sought or run**, as the brief
requires.

The cause is visible in the numbers and is not bad luck. Every seed gives essentially the same rate
(0.470–0.476 Hz) and the same active fraction to within ~0.35 points; the selected cell clears the 30 %
condition by **0.263 points**. The margin is smaller than the seed-to-seed spread, so which side of the
threshold a run lands on is decided by the noise draw.

## 6. Descriptive metrics (pre-registered as §5 of the brief; not used for selection)

Headroom on the binding condition, per `b_g`. Below the selected point the best run is a near-miss; above
it the headroom grows monotonically:

| b_g | usable (g, σ) cells | best active fraction | headroom to 30 % |
|---|---|---|---|
| 0.59255 | 0 | 30.161 % (near-miss) | **−0.161** |
| **0.65793** | **1** | **29.737 %** | **+0.263** ← selected |
| 0.73053 | 1 | 29.572 % | +0.428 |
| 0.81113 | 1 | 29.422 % | +0.578 |
| 0.90063 | 1 | 29.074 % | +0.926 |
| 1.00000 | 1 | 29.128 % | +0.872 |
| 1.51572 | 1 | 28.177 % | +1.823 |
| 2.29740 | 1 | 26.870 % | +3.130 |
| 3.48220 | 1 | 25.767 % | +4.233 |
| 5.27803 | 1 | 25.510 % | +4.490 |
| 8.00000 | 1 | 25.395 % | +4.605 |

Margins on all four conditions at the selected cell:

| condition | margin |
|---|---|
| rate ≥ 0.1 Hz | +0.370 Hz |
| rate ≤ 5 Hz | +4.531 Hz |
| ratio ≥ 0.5 | +0.435 |
| ratio ≤ 2 | +1.065 |
| tail < 1 % | +0.93 points |
| **active ≤ 30 %** | **+0.263 points** ← binding |

**The structural point.** The active fraction falls smoothly and monotonically with `b_g` while the
condition is a hard threshold, so *"the smallest `b_g` that passes" selects the least-robust passing point
by construction.* Refining the grid does not fix this — it makes it worse: M2e found a smaller `b_g` than
M2d (0.658 vs 1.0) with a **thinner** margin (0.263 vs 0.872 points), and a finer grid would find a smaller
`b_g` still, with headroom approaching zero. The table shows a 17× larger margin exists at `b_g` = 8.0.
**Changing the rule is a planning decision and this window did not make it** — the rule was applied as
written and its result reported, together with the evidence that the rule is the problem.

## 7. Voltage bound

Maintained. **264 grid runs, zero violations**, `v_min` = −74.98 mV against the −75.0 mV bound, at `b_g` up
to 8.0. With M2d that is **552 stimulus runs with no violation**. In the noise sweep `v_min` is −222.4 mV
at σ = 200, which is the noise current going negative exactly as in M2b (−229.4 mV with no adaptation at
all) — the adaptation conductance contributes nothing to the excursion even at `b_g` = 8.0, eight times
M2d's largest value.

## 8. Bench — **not re-run: environment precondition not met**

The brief permits a re-run only if (a) no other compute process holds the GPU and (b)
`clocks.sm ≥ 0.9 × clocks.max.sm`. Measured and recorded in
`data-provenance/m2e-bench-precondition.json`:

| check | measured | required | met |
|---|---|---|---|
| other compute processes | none (only this check's own process) | none | yes |
| SM clock under sustained load | **480 MHz median** (range 420–555) | ≥ 2,795 MHz | **no — 15.5 % of max** |
| throttle reason throughout | `0x4` = SW Power Cap | — | — |
| power draw | ~20 W | — | — |

**So the bench was not re-run and the < 100 µs target remains unverified for M2d/M2e**, as the brief
directs. No numbers were fabricated or rescaled. The machine is power-capped well below nominal — worse
than during the M2d bench, which saw 1,455–1,515 MHz — and this is a machine power state, not something
these sessions cause.

*One clarification on how the check was performed*: `clocks.sm` read on an **idle** GPU is always low
(throttle reason `0x1` = GpuIdle) and would fail the test regardless of machine state, so the clock was
sampled while a sustained matmul load ran. That is the quantity the condition is plainly about — the clock
a benchmark would actually see. The idle reading (480 MHz, `0x1`) and the under-load reading (480 MHz,
`0x4`) happen to agree here, so the conclusion does not depend on the refinement.

The adaptation branch's **incremental** cost (+1.5 µs/step, measured pairwise in M2d) is unaffected by
clock state in relative terms and was not re-measured, as the brief allows.

## 9. Completion criteria (`docs/m2e-brief.md`)

- [x] refined and extended grid results tabulated, **with the active fraction on every row** (§2, §3, §6)
- [x] **the rule now points at an interior value, not a grid edge** — 0.65793, bracketed by a tested
      failure at 0.59255 below and eight tested successes above (§4)
- [x] robustness check under seeds 3/4/5 reported — **failed, 2 of 3**; no substitute cell sought (§5)
- [x] v-bound violations still zero — 264 runs here, 552 with M2d, up to `b_g` = 8.0 (§7)
- [x] bench re-run only if the environment allowed — it did not, recorded as unverified with the
      measurement that shows why (§8)
- [x] `pytest tests/` passes (§10)
- [x] one "M2e: …" commit, not pushed

## 10. Tests

`tests/test_engine_adapt_g.py` grows to 25 cases. `test_defaults_match_the_recorded_sweep_rule` now
enforces the rule against `m2e-noise-sweep.json` (which supersedes M2d's answer under the same rule) and
additionally asserts that the selected `b_g` is the smallest of the usable cells. A new
`test_selected_cell_is_recorded_as_not_robust` ties `DEFAULT_ADAPT_G_B_ROBUST` to
`m2e-robustness.json`'s verdict and checks that the recorded note still says the failure was not repaired
by substituting a different cell — so a later edit cannot quietly turn a failed robustness check into an
adopted default.

`pytest tests/`: **338 passed, 0 failed, 0 skipped.** (The total is above M2d's 304 because the other
windows added tests while this work was in progress; M2d's run also had 1 skip — window B's live-server
budget check, which skips itself when another CUDA process shares the GPU — and that process had exited by
the time this suite ran.)

## 11. Deviations from the brief

1. **The bench precondition was measured under load, not at idle.** `clocks.sm` on an idle GPU is always
   far below maximum (throttle `0x1` = GpuIdle), so an idle reading would fail the test on any machine in
   any state and could not answer the question the condition asks. The clock was therefore sampled while a
   sustained load ran. Both readings agree here (480 MHz), so the verdict does not depend on it (§8).
2. **`DEFAULT_ADAPT_G_B` was updated to the M2e value** (0.65793, from M2d's 1.0), because it is the same
   pre-registered rule applied to the resolved grid. It is accompanied by `DEFAULT_ADAPT_G_B_ROBUST =
   False` and a comment stating the robustness failure, and it remains a recorded constant rather than the
   dataclass default — `EngineParams.adapt_g_b` is still 0.0, so no call site changes behaviour.
3. Nothing else. The four conditions, the selection rule, the `g` grid, the stimulus seeds and the sweep
   order are the M2d ones; the grid ranges are exactly those written in the brief; the brief was not edited.

## 12. Notes for the planning session

- **The grid question is answered.** The minimum is interior, in (0.59255, 0.65793]. Opening the top to
  `b_g` = 8.0 changed nothing about the selection, so that direction is closed.
- **The robustness question is answered, and the answer is no.** The selected cell fails under 2 of 3 other
  noise seeds. Its margin (0.263 points) is smaller than the seed-to-seed spread (~0.35 points).
- **The rule, not the grid, is now the limiting factor.** "Smallest `b_g` that passes" selects the
  least-robust passing point by construction when the binding condition is a hard threshold on a smoothly
  varying quantity. A finer grid would produce a smaller `b_g` with even less headroom. If a robust
  operating point is wanted, the rule needs to change — for example "smallest `b_g` whose headroom on every
  condition exceeds the seed-to-seed spread", or "the `b_g` maximising the minimum margin". §6 has the data
  such a rule would need; **choosing one is a planning decision and was not made here.**
- **The usable state is confined to one `g` and one σ.** All 10 usable runs sit at `g` = 5.365e-4,
  σ = 29.13. Whatever is adopted, the network's low-rate state in this model exists only in a very narrow
  corner of parameter space — that is a result about the model, not about the search.
- **The voltage bound is now very well established**: 552 stimulus runs across M2d and M2e, `b_g` spanning
  1e-3 to 8.0, zero violations. That conclusion is independent of every open question above.
- **The bench remains unverified** and will stay so until the machine is off its power cap; it is one
  command (`python -m flysim.engine.bench --synapse both --adapt-g`) whenever that happens.
- `parameter-decisions.md` and `backlog.md` were not edited (planning-owned).
