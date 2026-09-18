# M6a report — skeleton download, decimation and bundling

Engine session (window "M3 엔진"), 2026-09-18. Spec: `docs/m6a-brief.md`. Contract: `docs/m6-3d-contract.md`
(not modified). Code: `flysim/data/skeletons.py`. Tests: `tests/test_skeletons.py`. MD5 manifest:
`data-provenance/skeleton-hashes.json`. Output: `data/cache/skel-<name>.{bin,json}` (gitignored).

> **Headline for the planning session.** The pipeline works and all four bundles are under the 20 MB ceiling,
> but the brief's `max_segments_per_neuron = 200` is **unreachable for real neurons** and, where it binds, it
> destroys the geometry it was meant to preserve. See §5 — it needs a decision before this data is used.

## 1. What was built

`flysim/data/skeletons.py`, one module, no new dependencies:

| stage | function | notes |
|---|---|---|
| download | `download_skeleton`, `download_many` | MD5 against the server's `x-goog-hash`, exactly as `flysim/data/download.py`; 6 threads, 3 retries; cache in `data/raw/skeletons/`; HTTP 404 → `missing`, not an error |
| parse | `parse_swc` | `id type x y z radius parent`; `#` comments skipped; rejects short rows and empty files |
| forest | `branch_paths` | several roots allowed; a parent id absent from the file starts a new tree; cycles raise |
| decimate | `rdp_mask`, `simplify_neuron` | RDP per branch path; branch points and endpoints are path boundaries, so they cannot be removed |
| bundle | `build_bundle` | contract file 2; `SkeletonBundleTooLarge` above 20 MB |
| CLI | `main` | `--set NAME` (repeatable), `--tolerance`, `--max-segments`, `--threads`, `--force` |

Coordinates are passed through unchanged — MaleCNS source voxels, 1 voxel = 8 nm, no re-centering and no unit
conversion (contract "금지"). Bundle neurons are ordered by **graph index**, so the order is deterministic and
matches the index space of file 1 and `run.json`.

**Probe sets** come from `flysim/probe/sets.py` unchanged (that module belongs to another window). Left and
right are merged into one bundle as the brief requires (`JO_AB_L` + `JO_AB_R` → `JO_AB`).

**Soma** is taken from `Graph.soma_xyz` (the annotation `somaLocation`), not from the SWC. Reason: the MaleCNS
SWCs carry `type = 0` on every node, so they contain no soma node to read; the contract's coordinate section
already established that `somaLocation` is in the same voxel space as the skeletons. `soma` is `null` where the
annotation has none.

## 2. Sets and download

Sizes are the **disjoint** probe-set sizes produced by `sets.py`, which are what the brief tells me to use.
They differ from the sample counts in the assignment message (JO 138 / pC1 156 / AMMC 208 / WED 900) because
`sets.py` makes the sets disjoint by precedence (`JO_AB` → `JO_post` → `pC1` → `AMMCtype` → `SAD` → `WED`), so a
neuron claimed by a higher-precedence set does not appear in a lower one, and `JO_AB` additionally drops
JO-A/B neurons that have no synapse sites.

| set | neurons in the retained graph | assignment sample | downloaded | missing (404) | not in graph |
|---|---|---|---|---|---|
| JO_AB | 114 | 138 | 114 | 0 | 0 |
| AMMCtype | 168 | 208 | 168 | 0 | 0 |
| pC1 | 156 | 156 | 156 | 0 | 0 |
| WED | 964 | 900 | 964 | 0 | 0 |
| **total** | **1,402** | — | **1,402** | **0** | **0** |

**No skeleton was missing** — every one of the 1,402 bodyIds returned HTTP 200 with an MD5 header that matched
the bytes received. `missing` is `[]` in all four bundles. Every hash is recorded in
`data-provenance/skeleton-hashes.json` (1,402 entries: url, bytes, md5, server Last-Modified, timestamp).

**Re-run behaviour**: a cached file whose MD5 still matches the manifest is not re-fetched
(`download_skeleton` returns `"cached"`). Verified by re-running the CLI — see §6.

## 3. Decimation: what RDP can and cannot do here

The skeletons are sampled at a fixed ~64-voxel node spacing (visible in the SWC: consecutive x values step by
64). Splitting the forest at branch points and endpoints gives one path per branch, and **RDP cannot reduce a
path below one segment**. So:

> **per-neuron segment floor = number of branch paths ≈ 2 × (branch points) + roots**

Measured branch counts on real neurons are 500–1,700 paths for a typical pC1 / AMMC / WED cell (e.g. pC1
bodyId 10666: 9,543 nodes, 600 branch points, 1,212 paths). The floor is therefore 500–1,700 segments per
neuron, and **no tolerance can bring such a neuron to 200 segments** while branch points and endpoints are
preserved, which the brief requires.

`simplify_neuron` handles this without inventing a workaround: it doubles the tolerance while the neuron is
over the cap, stops as soon as the count reaches the floor (further tolerance buys nothing), and records
`cap_met`. Nothing is pruned to force the cap.

## 4. Results — the four bundles as built with the brief's defaults

`python -m flysim.data.skeletons --set JO_AB --set AMMCtype --set pC1 --set WED`
(tolerance 40 voxels, `max_segments_per_neuron` 200). Download 1,402 files / 107 MB in ~100 s total.

| set | neurons | raw nodes | segments | segments / raw nodes | branch floor | neurons over cap | final tolerances (voxels) | max deviation | bundle .bin | < 20 MB |
|---|---|---|---|---|---|---|---|---|---|---|
| JO_AB | 114 | 26,764 | 10,667 | 39.9 % | 3,389 | 4 / 114 | 40, 80, 320 | 278 vx = 2.2 µm | 0.24 MB | yes |
| AMMCtype | 168 | 351,599 | 48,816 | 13.9 % | 46,637 | 107 / 168 | 80 … 10,240 | 9,948 vx = 79.6 µm | 1.12 MB | yes |
| pC1 | 156 | 644,390 | 72,362 | 11.2 % | 71,999 | 148 / 156 | 160 … 10,240 | 6,392 vx = 51.1 µm | 1.66 MB | yes |
| WED | 964 | 2,397,360 | 308,328 | 12.9 % | 293,877 | 571 / 964 | 40 … 10,240 | 12,302 vx = 98.4 µm | 7.06 MB | yes |
| **total** | **1,402** | **3,420,113** | **440,173** | **12.9 %** | **415,902** | **830 / 1,402** | — | — | **10.08 MB** | — |

Validated for every bundle: `.bin` length == `n_segments × 6` float32, `seg_offset` contiguous and in graph-index
order, all coordinates finite, bbox equal to the actual min/max, `missing == []`, `not_in_graph == []`.

Soma coverage (from the annotation, §1): JO_AB **0 / 114**, AMMCtype 168 / 168, pC1 156 / 156, WED 959 / 964.
JO_AB having none is expected, not a bug — Johnston's-organ somata sit in the antenna, outside the imaged
volume — so every JO_AB neuron carries `"soma": null` and the viewer must tolerate that.

**Re-run** (`--set JO_AB` a second time): `cached=114 downloaded=0 missing=0`, and the rebuilt `.bin` is
byte-identical (md5 `adc28555d088ca6179b135f8c06c7646` before and after). The pipeline is deterministic and the
hash-match skip works.

## 5. The `max_segments_per_neuron = 200` problem — needs a decision

Every bundle is under 20 MB, so the brief's completion criteria are met as written. But look at the two columns
that matter: **830 of 1,402 neurons (59 %) could not reach the cap**, and where the escalation ran, the achieved
tolerance reached 10,240 voxels — **256× the 40 voxels asked for**, with deviations up to 98 µm. For pC1 the
delivered 72,362 segments sit against a floor of 71,999: **99.5 % of the geometry is gone**, and those neurons
are drawn as straight lines between branch points. The cap does not make files smaller in any useful sense; it
replaces the neuron's shape with its topology.

Measured alternative — the same tolerance 40, cap removed (`--max-segments` very large), on the same cached
files:

| set | with cap 200 | **uncapped at tolerance 40** | max deviation, uncapped | < 20 MB uncapped |
|---|---|---|---|---|
| JO_AB | 10,667 seg / 0.24 MB | 10,838 seg / **0.25 MB** | 40 vx = 0.32 µm | yes |
| AMMCtype | 48,816 seg / 1.12 MB | 143,738 seg / **3.29 MB** | 40 vx = 0.32 µm | yes |
| pC1 | 72,362 seg / 1.66 MB | 255,475 seg / **5.85 MB** | 40 vx = 0.32 µm | yes |
| WED | 308,328 seg / 7.06 MB | 934,463 seg / **21.39 MB** | 40 vx = 0.32 µm | **no — 21.39 MB** |

Three of the four keep the exact fidelity the tolerance promises and still fit the ceiling with room to spare.
**WED is the only genuine conflict**, and it misses by 7 %. Its tolerance curve (uncapped, same cached files):

| WED tolerance | segments | bundle | fidelity | < 20 MB |
|---|---|---|---|---|
| 40 vx | 934,463 | 21.39 MB | 0.32 µm | no |
| **64 vx** | **607,550** | **13.91 MB** | **0.51 µm** | **yes** |
| 80 vx | 514,917 | 11.79 MB | 0.64 µm | yes |
| 128 vx | 398,637 | 9.12 MB | 1.02 µm | yes |
| branch floor | 293,877 | 6.73 MB | topology only | yes |

**Recommendation.** Drop `max_segments_per_neuron` (or set it far above the branch count, e.g. 10,000) and let
the contract's 20 MB ceiling be the only size constraint, since that is the limit the contract actually
specifies. Build JO_AB / AMMCtype / pC1 at tolerance 40 (0.32 µm, 0.25 / 3.29 / 5.85 MB) and **WED at tolerance
64** (0.51 µm, 13.91 MB). That is 150–300× better geometry than the bundles shipped here, and every set still
fits:

```
python -m flysim.data.skeletons --max-segments 100000 --set JO_AB --set AMMCtype --set pC1
python -m flysim.data.skeletons --max-segments 100000 --tolerance 64 --set WED
```

I have **not** applied this — the brief fixed the defaults and the bundles delivered here use them. The
downloads are cached, so rebuilding is ~2 minutes once the planning session decides.

### Contract gap (reported, not patched)

The brief requires recording the final tolerance in the `.json`, but the contract's `decimation` block defines
only `method` / `tolerance_voxels` / `max_segments_per_neuron`, and its `neurons[]` records have no tolerance
field. Since the two cannot both be satisfied without a new key, I added three keys **inside `decimation`**
rather than touching the neuron records or the contract document:

```json
"tolerance_voxels_final": [40.0, 80.0, 320.0],   // distinct final tolerances actually used
"max_deviation_voxels": 277.9,
"neurons_over_cap": 4,
"segment_floor_total": 3389
```

`tolerance_voxels` keeps its contract meaning (the requested tolerance). If the planning session prefers these
elsewhere or not at all, say so and I will move them — `docs/m6-3d-contract.md` was not modified.

## 6. Tests (`tests/test_skeletons.py`, 19 cases — no network, no downloaded data)

Parsing: column layout and `#` comments; short rows and empty files raise; a path or a string path is accepted.
Forest: several roots and a dangling parent id both start trees; a cycle raises (including the case where the
cycle leaves no root at all, which the first implementation missed).
Decimation: a straight chain collapses to one segment; the Y keeps its branch point, both endpoints and its
root at a tolerance of 10⁶; a point displaced by *d* is kept at tol *d*/2 and dropped at tol 2*d*; over a random
400-point walk the reported deviation never exceeds the tolerance and **every dropped point is within the
tolerance of the infinite line through the retained pair that brackets it** (the actual RDP guarantee — not the
clamped distance to the nearest retained segment, which classic RDP does not bound); a looser tolerance never
keeps more points; the cap escalates the tolerance when it can help, and when the cap is below the branch floor
the result reports `cap_met = False` instead of pruning branches to fit.
Bundle: every contract top-level and per-neuron key present; `.bin` length and `seg_offset` chain consistent
with the `.json`, in graph-index order, with a known segment surviving the round trip; bodyIds absent from the
graph are dropped into `not_in_graph` and an all-absent request raises; a 404 skeleton is listed in `missing`
rather than skipped silently; a bundle over the ceiling raises `SkeletonBundleTooLarge` with guidance and
writes nothing; an empty request and an unknown probe-set name raise; the `x-goog-hash` MD5 header parses to
the value observed on the live server.

`pytest tests/` — see §8.

## 7. Deviations from the brief

1. **Probe-set sizes differ from the assignment's samples** (JO_AB 114 vs 138, AMMCtype 168 vs 208, WED 964 vs
   900). Not a deviation in behaviour — the brief says to use `flysim/probe/sets.py` unchanged, and those are
   the disjoint sizes that module produces (§2). Recorded so the difference is not mistaken for missing data.
2. **`max_segments_per_neuron = 200` is unreachable for 59 % of neurons.** Implemented exactly as specified
   (raise the tolerance and retry), with the escalation stopping at the branch floor instead of pruning
   branches, and the outcome recorded per bundle. Decision requested in §5.
3. **Three extra keys inside `decimation`** to satisfy the brief's "record the final tolerance" against a
   contract block that has no field for it (§5). The contract document itself was not edited.
4. **`not_in_graph`** added to the bundle `.json` alongside the contract's `missing`, so "dropped because it is
   not a retained neuron" is distinguishable from "skeleton 404". Both are `[]` in all four bundles.
5. **Soma comes from the annotation, not the SWC** — the MaleCNS SWCs have `type = 0` on every node, so there
   is no soma node to read (§1).
6. **The MD5 manifest is written once per set**, at the end of that set's download pass, not after each file.
   A crash mid-set would leave those files on disk unrecorded; the next run re-downloads and re-verifies them,
   so the worst case is wasted bandwidth, never an unverified file.

## 8. Completion criteria (`docs/m6a-brief.md`)

- [x] four bundles (JO_AB, pC1, WED, AMMCtype) generated, each under 20 MB — 0.24 / 1.66 / 7.06 / 1.12 MB (§4)
- [x] re-run skips downloads on a hash match — `cached=114 downloaded=0`, byte-identical output (§4)
- [x] node counts before/after and the maximum deviation tabulated (§4)
- [x] `tests/test_skeletons.py`: SWC parsing on synthetic data, branch preservation, tolerance respected,
      bundle json carries every contract key, bodyIds absent from the graph excluded, 20 MB exception (§6)
- [x] `pytest tests/`: **271 passed, 0 failed** — 19 of them new in `tests/test_skeletons.py`. (The total is
      above the 198 quoted in the assignment because the other two windows added their own tests to the same
      directory while this work was in progress.)
- [x] one "M6a: …" commit, only my four files, **not pushed**

## 9. Notes for the planning session

- **§5 needs a decision** before window C renders this data: as shipped, pC1 / AMMCtype / WED neurons are
  straight lines between branch points, which will look wrong in the viewer and is not what tolerance 40 meant.
- **JO_AB has no soma at all** (0 / 114). Window B's file 1 uses `src = 0` for somaLocation and `src = 1` for the
  synapse centroid, so JO_AB neurons will come through as centroids there; in the skeleton bundle they are
  `"soma": null`. The viewer must handle a null soma — worth confirming with window C.
- **`not_in_graph` is a fifth top-level key** I added next to `missing` (§7.4). If window C validates the
  contract keys strictly, it should ignore unknown keys rather than reject them.
- **Bundle bboxes** sit inside the synapse-centroid range quoted in the contract (x 4,673–92,111 · y
  6,069–68,657 · z 10,778–133,549), e.g. JO_AB x 33,472–65,216 · y 28,224–43,328 · z 7,808–32,384. The z minimum
  8,000 is slightly below the contract's quoted z floor of 10,778, which is expected — that range was measured
  on synapse centroids, and skeletons reach further than synapses do.
- The MD5 manifest is **1,402 entries / 420 KB** and is committed; the SWCs themselves (107 MB) live under the
  gitignored `data/` and are reproduced by re-running the CLI.
