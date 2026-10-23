# M7 report — probe profiles and escape-circuit skeletons

Engine window ("M3"), 2026-10-23. Spec: `docs/m7-profiles-brief.md`, which implements decisions 1 and 2 of
"M7e 준비 결정" in `docs/m7-game-ladder.md`. Files changed: `flysim/probe/sets.py`,
`flysim/data/skeletons.py`, `tests/test_probe.py`, `tests/test_skeletons.py`, this report, and the
skeleton MD5 manifest `data-provenance/skeleton-hashes.json` (see §5). `export3d.py` was not touched.

**Both deliverables are in.** `build_probe_sets(..., profile="escape")` gives the escape circuit as nine
sets at exactly M7a's sizes; the default `auditory` profile is bit-identical to the code before profiles
existed, proven against a fingerprint taken from the unmodified code. `skeletons.py --profile escape`
builds four bundles — `LC4`, `LPLC2`, `DNp01`, `TTMn` — with **no missing skeletons**, 5.9 MB in total.

## 1. Probe profiles (`flysim/probe/sets.py`)

`build_probe_sets(graph, roi, k_min=K_MIN_DEFAULT, profile="auditory")`. The `profile` argument is the only
addition: existing arguments, their order and the `ProbeSets` return type are unchanged, so every existing
call site — `m3_click`, `m4_ipi`, `m2g`/`m2h`/`m2i`, `export3d`, `live/export_neurons`, the tests — runs
unmodified on the `auditory` default. `ProbeSets` gains one defaulted field, `profile`.

| profile | sets | built from |
|---|---|---|
| `auditory` (default) | the 15 M3 sets, unchanged | types + ROI cache + JO contacts, as before |
| `escape` | `LC4_L`, `LC4_R`, `LPLC2_L`, `LPLC2_R`, `DNp01_L`, `DNp01_R`, `TTMn_L`, `TTMn_R`, `rest` | cell types only (`roi`/`k_min` unused, recorded in the log) |

An unknown profile raises `ValueError`. A missing type raises `LookupError` (CLAUDE.md §3.1) — tested by
pointing the escape profile at a type that does not exist.

**Escape sets, from the data:**

| set | size | M7a |
|---|---|---|
| `LC4_L` / `LC4_R` | 71 / 55 | 71 / 55 |
| `LPLC2_L` / `LPLC2_R` | 94 / 91 | 94 / 91 |
| `DNp01_L` / `DNp01_R` | 1 / 1 | 1 / 1 |
| `TTMn_L` / `TTMn_R` | 1 / 1 | 1 / 1 |
| `rest` | 164,807 | — |

Side is `Graph.side` (somaSide, then instance suffix) — the same rule M7a used — and a test checks each set
is **identical, cell for cell,** to `query.by_type(t, side=s)`, i.e. to M7a's sets. The four types do not
overlap, so precedence removes nothing (`removed_by_precedence` all zero). The sets still partition all
165,122 neurons. One deliberate difference in mechanism from M7a's script: the profile splits each type with
the same `_split_side` helper the auditory profile uses, so a cell of these types with an unknown side would
appear as its own `_unk` set instead of being silently dropped. None exists today, so the output is the nine
sets the brief lists.

**How the auditory profile was proven unchanged.** The shared "disjoint by precedence, then `rest`" tail was
factored into one helper used by both profiles. To make that refactor safe, the auditory sets were
fingerprinted **before any edit**, with `sets.py` unmodified at commit `2c2a66c`: a SHA-256 of every set's
index array and of `region_of`. `test_auditory_profile_is_bit_identical_to_the_pre_profile_sets` compares
the default call against those hashes — all 15 sets and `region_of` match, and `region_of` keeps its
`int16` dtype. A second test checks that `profile="auditory"` equals the default and that the auditory `log`
keeps exactly its old keys, because apps write that dict into their results JSON.

## 2. Skeleton CLI (`flysim/data/skeletons.py`)

| invocation | behaviour |
|---|---|
| `--set JO_AB --set pC1 …` | **unchanged** — names resolve against the `auditory` profile |
| `--profile escape` | every set of the profile except `rest`, sides merged → `LC4`, `LPLC2`, `DNp01`, `TTMn` |
| `--profile escape --set DNp01` | only that set, resolved against the escape profile |
| neither | usage error, exit 2 — as before, when `--set` was required |

New helper `profile_base_names(ps)` returns a profile's base names in profile order with sides merged and
`rest` dropped. A test checks that `LC4` does not pick up `LPLC2` (and vice versa) when resolving by prefix.
Bundle format is contract file 2 exactly as v1.1–v1.3 define it (v1.3 changed only the point cloud): tolerance
40 voxels, no per-neuron cap, 20 MB per-bundle ceiling, `missing`/`not_in_graph` always present.

## 3. The four bundles

`python -m flysim.data.skeletons --profile escape` — 315 skeletons downloaded and MD5-verified, none missing.

| bundle | neurons | raw nodes | segments | segments / raw | `.bin` | tolerance used | max deviation | soma | missing | not in graph |
|---|---|---|---|---|---|---|---|---|---|---|
| `LC4` | 126 | 208,854 | 83,385 | 39.9 % | **1.91 MB** | 40 | 40.0 vx = 0.320 µm | 126/126 | `[]` | `[]` |
| `LPLC2` | 185 | 420,293 | 167,149 | 39.8 % | **3.83 MB** | 40 | 40.0 vx = 0.320 µm | 185/185 | `[]` | `[]` |
| `DNp01` | 2 | 6,287 | 3,781 | 60.1 % | **0.09 MB** | 40 | 40.0 vx = 0.320 µm | 2/2 | `[]` | `[]` |
| `TTMn` | 2 | 2,801 | 1,607 | 57.4 % | **0.04 MB** | 40 | 40.0 vx = 0.320 µm | 2/2 | `[]` | `[]` |
| **total** | **315** | **638,235** | **255,922** | 40.1 % | **5.87 MB** | | | **315/315** | **0** | **0** |

Validated for every bundle: `.bin` length is `n_segments × 6` float32, `seg_offset` is contiguous and in
graph-index order, every coordinate is finite, the bbox equals the actual min/max,
`max_segments_per_neuron` is `null`, and `tolerance_voxels_final` is the single value `[40.0]`. All four are
far below the 20 MB ceiling, so no tolerance above 40 was needed and `TOLERANCE_BY_SET` is unchanged.

Every soma is present, unlike `JO_AB` in M6a (whose cell bodies are in the antenna, outside the volume).

**Bounding boxes, as an anatomical sanity check (voxels, 8 nm):**

| bundle | min (x, y, z) | max (x, y, z) |
|---|---|---|
| `LC4` | 19,584, 17,728, 24,832 | 77,248, 42,624, 40,576 |
| `LPLC2` | 14,528, 16,448, 25,728 | 86,656, 47,488, 42,176 |
| `DNp01` | 32,768, 19,328, 15,616 | 64,000, 55,744, **91,648** |
| `TTMn` | 35,328, 41,408, **84,736** | 64,576, 65,600, **105,728** |

The detectors sit in a band of the brain (z 25–42 k, both optic lobes in x); the giant fibre spans from the
brain down into the nerve cord (z 15.6 k → 91.6 k); the jump motor neurons are entirely in the thoracic nerve
cord (z 84.7 k → 105.7 k). That is the layout a descending giant fibre and a leg motor neuron should have,
and it means the split screen can draw the whole path from eye to leg.

**Re-run:** a second `--profile escape --set DNp01` finds both skeletons cached (`downloaded 0`) and writes a
byte-identical `.bin` (md5 `1a4b8c69…` before and after).

## 4. Tests

`pytest tests/`: **386 passed.** (349 at the start of this task, + 28 from the server window's M7b commit
`84b4bab`, which landed meanwhile, + 9 new here.) One earlier run showed 1 skip — the live server's
frame-budget test, which skips itself whenever another CUDA process shares the GPU; it passed on the re-run.

New in `tests/test_probe.py` (5): auditory bit-identical against the pre-profile fingerprint; explicit
`auditory` equals the default, with an unchanged log; escape names, sizes, partition and cell-for-cell
equality with M7a's rule; unknown profile raises; a missing escape type raises.

New in `tests/test_skeletons.py` (4): `profile_base_names` merges sides and drops `rest` for both profiles;
escape names resolve without `LC4`/`LPLC2` prefix collisions and reject auditory names; the CLI still exits 2
with neither `--set` nor `--profile`; `load_probe_sets` passes the profile through and defaults to auditory.

## 5. Deviations from the brief

1. **`data-provenance/skeleton-hashes.json` is committed although it is not in the brief's file list.** The
   pipeline appends one MD5 record per downloaded skeleton (1,402 → 1,717 entries, none missing); it is the
   provenance record M0/M6a established and this window owns from M6a. Leaving it out would leave 315
   downloads unrecorded, so the next run could not verify them by hash.
2. **The auditory build was refactored**, not left byte-for-byte as text: its disjoint-by-precedence tail is
   now a helper shared with the escape profile. The outputs are proven identical against a fingerprint taken
   from the unmodified code (§1), which is the brief's actual requirement.
3. **The escape profile keeps side-unknown cells as an `_unk` set** rather than dropping them, as the
   auditory profile already does. No such cell exists, so the output is exactly the brief's nine sets.

## 6. Notes for the planning session

- The ladder's decision 1 assigned the `sets.py` profiles to the server window; this brief assigned them here,
  and the later document was followed.
- The server can now start with `profile="escape"`: `DNp01_L`, `DNp01_R` and the two `TTMn` become one-neuron
  regions, so `frame.rates` carries each of them in every bin regardless of spike sampling — decision 3 of the
  ladder, with nothing further needed from this side.
- The four escape bundles total 5.9 MB, against 23.3 MB for the four auditory ones, so loading both profiles'
  skeletons in the split screen stays well inside what the 3D view already handles.
