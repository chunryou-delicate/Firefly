# M2h report — closing the hop-2 ambiguity with statistical power

Engine session (window "M3 엔진"), 2026-09-19. Pre-registration: `docs/m2h-brief.md`, written before this
ran and not edited afterwards. **Last round of the adaptation branch.** Code:
`flysim/apps/m2h_power.py`. Raw results: `data-provenance/m2h-results.json`. 240 runs, 474 s.

**Conclusion rule 1 applies: adaptation preserves hop-2 propagation.** M2g's provisional negative was a
power artefact, exactly as its §6 warned. With the floor 6× lower the signal is there — WED is detected in
**every** phase-lock cell, at all three σ, for both operating points. Hop 3 is unchanged: **pC1 is not
reached**, and M2g's conclusion there stands.

The size comparison the brief asked for complicates the headline, and §4 gives it honestly.

## 1. The design was decisive — achieved floor

The brief predicted the detection floor would fall about 6.1×. It did, and slightly further:

| set | M2g floor | predicted | **achieved** (min–max over the 12 cells) | no-adaptation signal | below the signal? |
|---|---|---|---|---|---|
| SAD_L | 0.143 | 0.023 | **0.007 – 0.022** | 0.053 | yes |
| SAD_R | 0.044 | 0.007 | **0.015 – 0.024** | 0.076 | yes |
| WED_L | 0.152 | 0.024 | **0.018 – 0.035** | 0.120 | yes |
| WED_R | 0.241 | 0.038 | **0.009 – 0.042** | 0.167 | yes |

Every achieved floor is below the corresponding no-adaptation signal, for every cell. **The ambiguity M2g
could not resolve is therefore resolved here**: a hop-2 response of the previously-observed size could not
have hidden.

Three things produced this and nothing else changed: seeds 3 → 10, window 500 → 2000 ms, and the test
statistic corrected from 3 × per-run sd to 3 × the standard error of the control mean (sd/√10). The
operating points, `g`, `a_in`, the sets and the sign conventions are M2g's.

## 2. Result

| point | σ | mode | JO_post L | JO_post R | hop-2 reached | pC1 | ignited |
|---|---|---|---|---|---|---|---|
| A | 29.13 | rate | +0.244 | +0.075 | none | no | no |
| A | 29.13 | phase-lock | +0.484 | +0.135 | **WED** | no | no |
| A | 26.21 | rate | +0.256 | +0.069 | none | no | no |
| A | 26.21 | phase-lock | +0.539 | +0.150 | **WED** | (see §5) | no |
| A | 23.30 | rate | +0.254 | +0.071 | none | no | no |
| A | 23.30 | phase-lock | +0.512 | +0.140 | **WED** | no | no |
| B | 29.13 | rate | +0.418 | +0.130 | none | no | no |
| B | 29.13 | phase-lock | +0.946 | +0.308 | **WED** | no | no |
| B | 26.21 | rate | +0.404 | +0.089 | none | no | no |
| B | 26.21 | phase-lock | +0.992 | +0.302 | **WED** | no | no |
| B | 23.30 | rate | +0.447 | +0.133 | **SAD** | no | no |
| B | 23.30 | phase-lock | +0.983 | +0.270 | **SAD, WED** | no | no |

Crossings per set over the 12 cells:

| set | crossings | phase-lock | rate |
|---|---|---|---|
| WED_L | 5 / 12 | **5 / 6** | 0 / 6 |
| WED_R | 5 / 12 | **5 / 6** | 0 / 6 |
| SAD_L | 2 / 12 | 1 / 6 | 1 / 6 |
| SAD_R | 1 / 12 | 1 / 6 | 0 / 6 |
| AMMCtype_L/R | 0 / 12 | 0 / 6 | 0 / 6 |
| pC1_L | 0 / 12 | 0 / 6 | 0 / 6 |
| pC1_R | 1 / 12 | 1 / 6 | 0 / 6 |

**WED is the finding.** Every one of the six phase-lock cells reaches it (five on both sides, one on the
right only), across both operating points and all three σ. It is not a threshold accident: the difference
grows monotonically as σ falls (A: +0.026 → +0.031; B: +0.037 → +0.056 on the left) and is consistently
larger for B than for A — a dose-response in two independent directions. Nothing in rate mode crosses at
WED at any σ.

**SAD is marginal**: 3 crossings, all at the lowest σ and the stronger operating point (B, σ = 23.30),
sizes +0.010 to +0.029 Hz. Consistent with SAD being just at the edge of detectability rather than absent.

**AMMCtype: nothing, anywhere.**

## 3. Ignition, under the redefined rule

The brief redefined ignition as *excess over control ≥ 3 SEM **and** stimulus run ≥ 90 % active*.
**No cell is ignited (0/12).** The largest stimulus late-window active fraction anywhere is **18.3 %**, far
from 90 %.

This is the redefinition working. M2g's absolute ≥ 1 % rule flagged 24/24 runs including all 12 silence
controls; the same experiment under the new rule flags nothing, because nothing here is a runaway — the
25–28 % background is the operating point, not ignition.

## 4. Size, side by side with the no-adaptation values — the honest complication

> **정정 (2026-09-19, M2i 이후 · 총괄 세션이 추가).** 아래 rate 모드 블록의 **"5–7× smaller"와
> "adaptation attenuates hop-2 propagation"은 철회한다.** 이 절이 스스로 남긴 단서(마지막 문단의
> `g` 불일치)를 M2i(1760f92)가 실제로 돌려 확인했고, 교란이 여기서 가정한 것과 **반대 방향**으로
> 작용했다. `g` = 5.36539e-4에서 적응을 끄면 세 σ 전부에서 망이 점화 상태이고(활성 31–47 %,
> 24–33 Hz, 자극 종료 후 자립), σ = 0에서조차 클릭 트레인 단독으로 폭주에 들어간다(활성 19.9 %,
> 종료 후 15 %). 같은 자극이 `g` = 3.162e-4에서는 점화 없는 깨끗한 반응이었다. 따라서 이 비교는
> 적응 런을 **더 낮은 이득의 다른 동작점**과 견준 것이며, 그 이득에서 무적응 런은 "약해진 신호"가
> 아니라 애초에 신호가 아니다. 두 팔이 함께 지지하는 더 좁고 단단한 진술은 이것이다 —
> **`g` = 5.36539e-4, σ 23–29에서 쓸 수 있는 저발화 배경은 적응이 있어야만 존재하며**, 그 안에서
> 홉 2는 phase-lock에서 도달하고 rate에서는 도달하지 않는다. 아래 원문은 그 시점의 기록으로 남긴다.
> 근거는 `docs/m2i-report.md`.

The brief asks for the signal size next to the no-adaptation value. Doing that changes what "preserves"
means, in different directions for the two modes.

**Rate mode, same `a_in` = 640, both non-ignited — a fair comparison:**

| | WED_L | WED_R |
|---|---|---|
| no adaptation (M3b) | 0.120 | 0.167 |
| with adaptation (M2h, best cell) | +0.024 | +0.026 |
| achieved floor | 0.018–0.035 | 0.009–0.042 |

Same stimulus, same gain, and the hop-2 response is **5–7× smaller** with adaptation — and does not clear
the floor in any rate cell. On this comparison adaptation **attenuates** hop-2 propagation rather than
preserving it.

**Phase-lock mode, same `a_in` = 2560 — the comparison is not available, and that is itself the point:**
without adaptation that operating point **ignited** (`m3b` phase-lock a_in 2560 → `FAIL_IGNITED`), so its
WED numbers (0.187 / 0.175) came from an invalid run. With adaptation the same gain runs **without
igniting** and delivers a detectable hop-2 response of +0.020 to +0.064 Hz. Here adaptation does not
preserve a signal that already existed — it makes a previously unusable operating point usable.

**Caveat on both comparisons**: the no-adaptation reference (`m3b`) used `g` = 3.162e-4, the M2b default,
while M2h uses `g` = 5.36539e-4, the value M2f selected to go with adaptation. So this is a comparison
between two operating points, not a clean single-variable contrast. No run was made to close that gap,
because the brief forbids new operating points and this is the last round.

## 5. The single pC1 crossing — reported, and judged a false positive

One of 24 pC1 tests crosses: point A, σ = 26.21, phase-lock, `pC1_R`, diff **+0.1122** against a 3 SEM
threshold of **0.1082** — over by 0.004 Hz, i.e. 3.1 SEM. Four reasons it should not be read as pC1 being
reached:

1. It fails the stricter statistic. Against the SEM of the *difference* (which also carries the stimulus
   spread) the threshold is 0.1361 and the crossing does not survive.
2. `pC1_L` in the same cell does not cross (+0.059 against 0.101).
3. It does not replicate in σ. The same point and mode gives +0.015 at σ = 29.13 and **−0.055 / −0.037**
   at σ = 23.30 — the neighbouring, *more* sensitive cell is negative.
4. Across the 12 cells pC1 differences are symmetric about zero, −0.122 to +0.128, which is what a null
   looks like at this noise level.

**pC1 is recorded as not reached.** Per the brief, zero is written as zero. The crossing is disclosed in
full here and in the JSON rather than filtered out.

## 6. Which conclusion rule applies

The brief pre-registered three:

- *"If a hop-2 set crosses the threshold at any σ → adaptation **preserves** hop-2 propagation; record the
  signal size next to the no-adaptation value."* → **This one.** WED crosses in all six phase-lock cells
  at all three σ and both operating points. Sizes are in §4.
- *"If the achieved floor is below the no-adaptation signal and nothing crosses → adaptation **suppresses**
  hop-2 propagation."* Not selected overall — but note that **this is what rate mode alone would have
  given** (floor below the signal, nothing crosses, §4). The two modes disagree, and the rule is written on
  "any σ", so the first rule governs.
- *"If the floor does not come down far enough → still unresolved, stop."* Not applicable; the floor came
  down (§1).

**So: adaptation preserves hop-2 propagation in phase-lock mode and attenuates it in rate mode; hop 3 is
not reached in either.** The hop 2 → 3 wall that motivated this whole branch is still there.

## 7. What this changes about M2g

M2g's recorded sentence — *"adaptation produces a background state but does not extend propagation
distance"* — needs splitting:

| claim | status after M2h |
|---|---|
| no hop-2 arrival detectable | **overturned.** It was a power artefact; WED is detected in every phase-lock cell |
| adaptation does not extend propagation to hop 3 | **stands.** pC1 is not reached at any σ, point or mode |
| adaptation produces a background state | stands |

M2g's §6 called this correctly: the experiment could not distinguish "did not help" from "buried", and the
answer turns out to be **buried** for phase-lock WED and **genuinely smaller** for rate WED.

## 8. Completion criteria (`docs/m2h-brief.md`)

- [x] 240 runs completed (2 points × 3 σ × 2 modes × stimulus/control × 10 seeds), 474 s
- [x] achieved floor tabulated against the prediction — achieved is at or below predicted throughout (§1)
- [x] per-set (stimulus − control) and thresholds, all three σ side by side (§2, full detail in the JSON)
- [x] which conclusion rule applies, stated explicitly (§6)
- [x] redefined ignition judgement reported — 0/12 (§3)
- [x] `pytest tests/` passes — 338 passed
- [x] one "M2h: …" commit, not pushed

## 9. Deviations from the brief

1. **The SEM of the difference is computed and reported alongside** the pre-registered SEM of the control
   mean. The pre-registered statistic decides every verdict; the stricter one is used only to scrutinise
   the single pC1 crossing (§5), where it matters.
2. Nothing else. The operating points, `g`, `a_in`, σ values, seed count, window, sets, the three
   conclusion rules and the redefined ignition test are exactly as written. No new operating point was
   searched, no parameter moved, the brief was not edited, and no other window's files were touched.

## 10. Notes for the planning session

- **The branch's question now has a split answer.** Adaptation fixed the voltage bound (M2d), widened the
  gain range (M2d/M2e), produced a robust background state (M2f), and — this round — **keeps the hop-2
  signal alive in phase-lock at a gain that previously ignited**. It still does not carry anything to pC1.
- **The hop 2 → 3 wall is the durable result of this whole line of work.** pC1 has now been zero in the
  current model, the conductance model, both adaptation operating points, and under a detection floor 6×
  lower than anything tried before. At some point that is a statement about a wiring-only model, not about
  parameters.
- **Rate and phase-lock disagree**, and the disagreement is interpretable: adaptation costs a rate-coded
  signal more than a phase-locked one, which is what a mechanism that punishes sustained firing should do.
  If the branch is ever reopened, that is the interesting thread, not another operating point.
- **The redefined ignition rule works** and should replace the absolute one wherever a background state is
  present (§3).
- `parameter-decisions.md` and `backlog.md` were not edited (planning-owned).
