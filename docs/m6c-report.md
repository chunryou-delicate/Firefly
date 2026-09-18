# M6c report — flysim-3d, the 3D neuron viewer

Built 2026-09-18 against `docs/m6-3d-contract.md` (commit `62fdbf0`), spec `docs/m6c-brief.md`.
Deliverables: `flysim-3d.html` (single file, no build, no dependencies) and `tests/test_3d_html.py` (27 tests).
`flysim-viewer.html` and `flysim-live.html` are untouched; this is a third surface, not a replacement.

**No 3D library.** WebGL2 is used directly: two GLSL ES 3.00 programs (points, lines), hand-written 4×4 matrix
code, and one vertex array per buffer. The only external resource is the same Google Fonts link the other two
screens use.

## 1. What it draws

| layer | how |
|---|---|
| **point cloud** | `gl.POINTS`, one point per retained neuron. Position and set index are uploaded once (`STATIC_DRAW`); only an activation byte per neuron is refreshed per frame (`bufferSubData` into a `DYNAMIC_DRAW` buffer). Point size attenuates with distance and grows with activation. Additive blending, depth test on, depth write off — the same "overlapping neurons get brighter" property the 2D viewer has |
| **skeletons** | `gl.LINES` straight from the contract's segment array, one buffer per bundle, toggled per set. Alpha-blended with depth write, so lines occlude points behind them |
| **activity** | a `run.json` dropped on the window (or picked). Spikes set a neuron's activation to full; it decays by the afterglow factor every frame, the 2D viewer's mechanism moved into 3D |
| **axes** | a small 2D gizmo showing the current orientation of X (red), Y (green), Z (cyan) |

Camera: drag to orbit, wheel to zoom, Shift+drag to pan, three presets, auto-rotate, a field-of-view slider that
keeps the subject the same size, and "fit". Keyboard: space plays, F fits, R toggles rotation, arrows step frames.
Side panel: asset loading, view, appearance (point size, idle brightness, afterglow, skeleton opacity, colour
mode), the set list with per-set counts and live rates, the skeleton list, and a frame-timing measurement.

Colours keep the established rule — cool for input, warm for output — along the auditory pathway:
JO_AB cyan → JO_post → SAD → AMMCtype → WED → pC1 red, with `rest` a dark neutral. A second mode colours by side
instead (cool = left, warm = right, grey for `_unk` and `rest`), which is the comparison CLAUDE.md §8 asks for.

## 2. How the contract is read

- **Units.** Coordinates stay in MaleCNS voxels exactly as delivered. `VOXEL_NM = 8` is applied only when a
  number is printed (the footer shows the bounding box in μm). No conversion or re-centring happens at load; a
  test asserts the parser never touches the μm constant.
- **Arrays.** `arrays.<name>.offset` is treated as a byte offset and `length` as an element count — the only
  reading under which the contract's own example numbers line up (`0 + 495366×4 = 1981464`), and a test asserts
  that arithmetic against the contract file. The reader checks dtype, 4-byte alignment for `pos`, and that each
  span fits inside the file before creating a view.
- **Missing coordinates.** `src = 2` means NaN. Those neurons are dropped at load rather than collapsed onto the
  origin, and a neuron-index → draw-index map is kept so `run.json` spike indices still land on the right point.
  The footer reports how many were excluded, and spikes that reference a dropped neuron are counted and logged
  instead of silently vanishing.
- **Skeleton bundles.** `n_segments`, per-neuron `seg_offset`/`seg_count`/`body_id` and `missing` are parsed and
  validated (a neuron range past `n_segments` is an error). `missing` is surfaced in the panel, not swallowed.
- **Endpoints.** `/neurons-3d.json`, `/neurons-3d.bin`, `/skel/index.json`, `/skel/<name>.json`, `/skel/<name>.bin`,
  exactly as listed. A bundle that 404s is a notice naming it, not a failure: the point cloud still loads.
- **`run.json` is unchanged.** The 3D viewer reads `frames.rates`, `regions` and `spikes` and takes positions from
  the point cloud. Nothing about that contract is modified.

## 3. Mock mode (`?mock=1`)

The mock does not shortcut the loader. It builds a real `ArrayBuffer` in the contract's layout and a matching
metadata object, then calls the same `parseCloud` / `parseSkel` the server path calls — so running the mock is
itself a test of the parsing code.

**Everything it generates is synthetic**: coordinates, branch geometry and firing rates are invented, are not the
connectome and carry no scientific meaning. A red badge says so permanently and the asset note repeats it. Only
three things are taken from documented reality so the proportions are not misleading: the 15 probe set names and
their sizes (`docs/m3-report.md` §2, summing to 165,122), the `src_counts` split (140,024 soma / 24,765 centroid /
333 none), and the voxel coordinate range the contract records. `?segs=N` changes segments per neuron (default 200,
the contract's cap) for stress testing.

The mock also generates a synthetic `run.json` so playback can be exercised without a file: a click train driving
a JO → JO_post → SAD → AMMC → WED → pC1 chain with lag and decay, plus low background activity for the other sets,
in the viewer's own `run.json` shape.

## 4. Performance

Measured on the real graph's worth of geometry: **164,789 points** (165,122 minus the 333 with no coordinate) and
**280,400 skeleton segments** across four bundles, with playback and auto-rotation both running, at 1326×730 CSS
pixels. Frame intervals were timed over 240 consecutive `requestAnimationFrame` callbacks.

| renderer | vsync | median frame | p95 | median fps |
|---|---|---|---|---|
| RTX 4090 Laptop (ANGLE/OpenGL) | off | **1.10 ms** | 2.10 ms | **909** |
| RTX 4090 Laptop (ANGLE/OpenGL) | on | 16.70 ms | 16.70 ms | 59.9 (display-locked) |
| SwiftShader (software fallback) | off | 164.9 ms | 186.3 ms | 6.1 |

The brief's target was 60 fps at 200,000 segments; on the GPU the viewer is roughly fifteen times that headroom at
280,400 segments, and the vsync-on row shows it holds the display rate with no dropped frames. Turning the
skeletons off changes the median by about 0.1–0.2 ms, so at this scale the cost is dominated by neither layer — one
draw call each, with no per-frame geometry upload. The software fallback is included because it is what a machine
without WebGL2 hardware acceleration gets; at 6 fps it is usable for inspection but not for playback.

Per-frame CPU work is the afterglow decay over the activation bytes plus one 165 KB `bufferSubData`; the decay
loop is skipped entirely when nothing is active.

## 5. Verification

Tests are static (`tests/test_3d_html.py`, 27 passing; `pytest tests/` = 271 passing overall). They read
`docs/m6-3d-contract.md` and `flysim-viewer.html` directly, so a contract change or a change to the viewer's
failure wording that this file has not followed fails the suite rather than drifting silently. Covered: parsing,
zero external scripts, no 3D library by name, WebGL2 used directly, contract array names and dtypes, the
byte-offset/element-length reading, NaN handling, skeleton bundle fields, the endpoint list, no unit conversion at
load, the `?mock=1` branch and its red badge, the mock going through the real parser, upload-once/refresh-colour,
one buffer per skeleton set, camera controls, reduced motion, and `run.json` acceptance.

Behaviour was exercised in headless Chrome over the DevTools protocol (scratch scripts, not part of the repo; no
browser automation was added to the suite):

| case | observed |
|---|---|
| `?mock=1` | 165,122 neurons parsed, 164,789 drawn, exactly 333 dropped as NaN, 15 sets, 4 bundles / 280,400 segments, no JS errors |
| playback | 900 frames, ~2,500–4,000 neurons lit at peak, afterglow decaying, scrub and speed working |
| real files over HTTP | a contract-shaped 20,000-neuron cloud and a pC1 bundle served by `http.server` loaded through the same path: 19,960 drawn, 40 NaN dropped matching `src_counts.none`, 4,735 segments, 120 neurons, `missing` surfaced |
| a bundle listed in `index.json` but absent | notice reads "불러오지 못한 묶음: WED (HTTP 404 skel/WED.json)"; the point cloud still loads and nothing throws |
| no assets at all | the view is blocked with "자산을 불러오지 않았다" and instructions to press load or add `?mock=1` |
| no WebGL2 (forced with `--use-gl=egl`) | the WebGL2 message appears instead of a blank canvas |
| presets, colour modes, set toggles | all switch as expected; frame timing unaffected |

## 6. Deviations from the brief and the contract

1. **Framing uses the actual extent of the points, not the declared `bbox`.** The contract's bbox is kept and
   compared; when it is more than 1.35× wider than the points in any axis the viewer logs that and frames on the
   data. Reason: the retained volume is 982 μm deep because of the VNC, so fitting a loose box leaves the subject
   at a fraction of the window. The file is not modified, only the camera.
2. **Preset buttons refit for their direction.** A single "fit everything from any angle" distance is loose for
   the front view of a volume this elongated, so each preset computes the distance at which the eight bbox corners
   just fit for that direction.
3. **`?segs=N`** is a mock-only knob for stress testing; it does not exist in the contract and does nothing on
   real files.
4. **Y is flipped by default** so the model appears upright. This is a camera up-vector choice, toggleable, and
   changes no coordinate. The preset labels name their axis (front −Z, side +X, top −Y) rather than claiming an
   anatomical orientation, which the data alone does not establish.
5. **Failure wording is byte-identical to the 2D viewer**, including the consequence that a paused run sitting on
   a frame with no spikes shows "발화 뉴런 없음". That is what the existing viewer does at frame 0; parity was kept
   deliberately rather than inventing a different rule.
6. **In side-colour mode the skeletons keep their pathway colour.** The contract merges left and right into one
   bundle per set, so a bundle cannot be split by side without re-deriving it from the per-neuron `idx`. Points do
   switch. Noted as a limitation rather than worked around.
7. **A cap of 24 sets** is baked into the shader's uniform arrays; more than that raises a clear error instead of
   drawing wrong colours. The contract's own set list has 15.

## 7. Not implemented

- **`hello.assets` (protocol v1.2) is not consulted.** This viewer has no websocket; it asks the HTTP endpoints
  directly and treats a 404 as "not built yet". If the planning session wants the cockpit's asset advertisement
  honoured here, that is a separate change.
- **No neuron picking.** Clicking does not identify a neuron or show its bodyId; `seg_offset`/`seg_count` are
  parsed and validated but not yet used to highlight a single neuron's skeleton.
- **No depth sorting for the skeleton lines.** They are alpha-blended in bundle order. Points are additive and
  therefore order-independent, so only overlapping translucent lines can look slightly wrong.
- **Not verified against the real produced files.** The other two windows are still generating
  `data/cache/neurons-3d.*` and `data/cache/skel-*`. The HTTP path was verified against files written to the
  contract by hand; a pass against the real ones is still needed.
