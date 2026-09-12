# Edge sign decision (approved 2026-09-12, implemented in M1)

Source column: `neurotransmitters.consensus_nt` (keyed by `body`), applied per
presynaptic neuron (Dale's principle — **assumption**).

| consensus_nt | retained neurons | sign | basis |
|---|---|---|---|
| acetylcholine | 103,718 | +1 | standard |
| glutamate | 29,296 | −1 | standard for Drosophila CNS (GluCl) — assumption |
| gaba | 22,055 | −1 | standard |
| histamine | 5,910 | +1 | **assumption**; flagged in modulator mask |
| dopamine / octopamine / serotonin | 392 / 101 / 48 | +1 | **assumption**; flagged in modulator mask |
| unclear | 3,100 | +1 | **assumption** (ACh majority prior) |
| missing from NT table | 502 | +1 | **assumption** (ACh majority prior) |

Actual per-sign edge counts are to be appended by `flysim.graph.build` when M1 runs.
