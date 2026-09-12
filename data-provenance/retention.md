# Graph retention — original vs. retained (M0)

Source files: see `hashes.json`. Verification output: `m0-verify.json`.

## Rule

Keep a body iff `annotations.status == "Traced"`.
Keep an edge iff both `body_pre` and `body_post` are kept.
No weight threshold is applied. Nothing else is filtered.

## Counts

| | raw | retained | dropped |
|---|---|---|---|
| annotation rows / neurons | 211,577 | 165,122 | 46,455 |
| weight rows / directed edges | 151,856,684 | 25,563,197 | 126,293,487 |
| synaptic contacts | 311,833,243 | 124,025,046 | 187,808,197 |
| cell types (`type` non-null unique) | 11,751 | 11,751 | 0 |

## Why the drop is so large

The weights file is segment-to-segment for **every** segment (1.8M distinct
pre bodies, 87.6M distinct post bodies), i.e. it includes orphan fragments,
glia and unassigned segments. Only 165,122 bodies carry `status == "Traced"`.
The other status values observed are `Orphan`, `Glia`, `Unimportant`,
`Assign`, `Anchor` and null (see `docs/annotations-observed.md`).

## Comparison with CLAUDE.md §4.2

| item | reference | observed | deviation |
|---|---|---|---|
| neurons | 166,700 | 165,122 | -0.95% |
| directed edges | 25,580,000 | 25,563,197 | -0.07% |
| synapse contacts | 124,000,000 | 124,025,046 | +0.02% |
| cell types | 11,691 | 11,751 | +0.51% |

All within the 10% tolerance. The reference neuron count (166,700) is what
doomfly reports for its "retained graph"; the ~1,600 gap most likely comes
from a slightly different status set on their side (e.g. also keeping
`Anchor`/`Assign`). Not investigated further because the edge and synapse
totals match to <0.1%, which is the quantity the simulation depends on.

## Facts needed by later milestones (observed, not assumed)

- `somaSide`: R 74,430 / L 74,220 / M 392 / null 16,080 among retained.
- `somaLocation` (xyz, int voxel coords) present for 140,024 / 165,122 retained
  → usable for viewer `neurons.x/y`; the remaining ~25k need a fallback.
- Neurotransmitter: `consensus_nt` among retained —
  acetylcholine 103,718 · glutamate 29,296 · gaba 22,055 · histamine 5,910 ·
  unclear 3,100 · dopamine 392 · octopamine 101 · serotonin 48 · missing 502.
  → An NT column exists (§5.1 question answered: yes). Sign rule for
  `unclear`/missing/modulatory (dopamine, octopamine, serotonin, histamine)
  must be decided explicitly in M1 and documented as an assumption.
- `class` has `mechanosensory` (1,733) and related values; `superclass` has
  `cb_sensory` (4,868). JO neuron identification in M3 must be done by
  querying `type`/`instance` strings against the data, not from memory.
