# Parameter decisions log

Every choice that is not derived from the connectome, with when/why. Newest last.

## 2026-09-12 — edge signs
See `edge-signs.md`.

## 2026-09-13 — M2 sweep criterion was inadequate; M3 operates below ignition

**What M2 found** (`docs/m2-report.md` §4): the network is bistable. Above g ≈ 0.55–0.58 a
first-order transition puts 12–30 % of neurons into a saturated self-sustained state
(150–213 Hz, v down to −4,000 mV with current-based synapses). Below it the network is
silent at rest and input-driven. The M2 criterion ("EXTINCT = 0 spikes in the last 200 ms
after stimulus offset") only measures self-sustained activity, so it labelled the
input-driven regime EXTINCT and the ignited regime VALID. `DEFAULT_G = 0.886` therefore
points at the ignited state. The criterion was written by the planning session in
`docs/m2-brief.md`; the implementation applied it faithfully and flagged the problem.

**Reconnaissance by the planning session** (not an M3 result; dt = 1 ms, constant current 30
to the 114 JO-A/B neurons with synapse sites for 100 ms, 200 ms observation, seed 0):

| g | JO spikes | downstream spikes | downstream neurons | per-50 ms downstream |
|---|---|---|---|---|
| 0.200 | 684 | 102 | 31 (0.02 %) | 39, 62, 1, 0, 0, 0 |
| 0.336 | 682 | 366 | 136 (0.08 %) | 124, 201, 41, 0, 0, 0 |
| 0.450 | 671 | 497 | 196 (0.12 %) | 241, 210, 46, 0, 0, 0 |
| 0.500 | 656 | 622 | 245 (0.15 %) | 284, 268, 70, 0, 0, 0 |

Input reaches downstream neurons and activity dies within 50 ms of offset at every g tried.

**Decision (planning session, rule stated before choosing):** M3 uses
`g = 0.336` — the largest point of the M2 primary grid that was EXTINCT (silent after
offset) for all three stimulus seeds in `m2-sweep.json` / `m2-sweep-supplementary.json`.
A run in which activity persists > 100 ms after stimulus offset with ≥ 1 % of neurons active
is recorded as "ignited" and treated as a failed run, not a result. `v_floor` stays `None`.
`DEFAULT_G` in the engine is left as M2 documented it; M3 overrides it explicitly and
records the value in every run's `meta.engine_params`.

**Not decided yet:** whether to move to conductance-based synapses (reversal potentials)
to remove the unphysical hyperpolarisation. Deferred until M3/M4 show whether the
input-driven regime is sufficient.
