# M6d report — live mode and neuron picking in the 3D viewer

Built 2026-09-19 against **3D contract v1.3** and **protocol v1.2** (`docs/m6-3d-contract.md`,
`docs/m5-protocol.md`), spec `docs/m6d-brief.md`. Changes are confined to `flysim-3d.html`,
`tests/test_3d_html.py` and this report. `flysim-viewer.html` and `flysim-live.html` are untouched.

The 3D screen now talks the websocket itself instead of the cockpit growing a second renderer.

## 1. Live mode

Connect from the panel, or open `flysim-3d.html?ws=ws://<host>:8765/ws`. The screen then draws the
resident engine's activity: `frame.spikes.idx` goes into the same activation bytes the file player
uses, so the render path is unchanged.

**What it sends is deliberately small** — the brief's point, and the reason it stays that way: if both
screens carry the same controls, every fix has to happen twice.

| sent | not sent |
|---|---|
| `stimulus` (silence / click train / tone presets), `pause`, `resume`, `step`, `reset`, `get_hops`, `snapshot` | **`set_params`.** g, a_in, noise, synapse, dt and speed are the 2D cockpit's job. A panel note says so on screen, and a test asserts the message is never sent |

All eight server messages the protocol defines are handled (`hello`, `frame`, `params`, `ack`, `hops`,
`snapshot_done`, `warning`, `error`). `params` is display-only here: when someone changes a value in the
cockpit, this screen shows the new version and values but offers no way to change them. Warnings reuse
the 2D viewer's wording. Reconnect is 2 s, same as the cockpit.

`hello.assets` is now read (the M6c report listed this as deliberately unimplemented; the brief reverses
that for live mode). When the server advertises bundles, those are fetched; with no `assets` the viewer
falls back to probing the HTTP endpoints as before. `assets.neurons_3d === false` produces a notice
naming the command that builds the file, not an error.

**One source of activity at a time.** Connecting locks the file transport and stops playback;
disconnecting restores it. Without that, a loaded `run.json` and the live stream both write the time and
rate readouts — which is exactly what happened in mock mode, where a synthetic run is pre-loaded, and is
now asserted by a test.

**`?ws=` is always a full URL** (`ws://host:port/ws`), per the link convention added to the protocol on
2026-09-19. A value that does not start with `ws://` or `wss://` is ignored with a log line and the
default address is used, rather than being patched up into something that might be wrong.

## 2. Neuron picking

A click that did not move the pointer more than 4 px selects the nearest neuron; anything else is an
orbit. Picking projects the drawn points with the current matrix, prefers candidates within 14 px and
among those the one closest to the camera, and skips hidden sets. It returns a graph neuron index, so
the result is directly usable with `run.json` indices and `get_hops`.

The panel shows **bodyId** (contract v1.3's `body` array), the graph index, the probe set, the coordinate
source (`src`), and the position in both voxels and μm. `body` is optional by contract: files written
before v1.3 have no such array, and the panel then says so and shows the index alone. A test enforces that
`body` stays out of the required-array list.

Verified against the real file: `body` reads back as a `Uint32Array` of 165,122 entries with a maximum of
**1,471,062,202**, the value the planning session measured independently.

## 3. Downstream and skeleton highlight

- **`get_hops k=2`** on the selected neuron. One-hop and two-hop layers are tinted cool and the rest of
  the cloud recedes to 34 % brightness so the highlight reads. This needs the server: in file mode the
  button is disabled and says why. Highlight state is a single `uint8` vertex attribute updated only when
  the selection changes — no geometry is rebuilt.
- **The selected neuron's own skeleton** is redrawn brightly over the bundle, using the contract's
  per-neuron `seg_offset` / `seg_count` as a sub-range of the same buffer. An index from neuron to segment
  range is built once per bundle (1,402 neurons across the four real bundles).

## 4. Verification

`pytest tests/` = **336 passed, 1 skipped**; `tests/test_3d_html.py` grew from 27 to **59** tests. The new
ones read `docs/m6-3d-contract.md` and `docs/m5-protocol.md` directly: the live message lists must be a
subset of the protocol's tables, `set_params` must be absent, `body` must be optional, and the `?ws=`
rule must match the documented convention.

Driven in headless Chrome over the DevTools protocol (scratch scripts, not in the repo).

### 4.1 Mock (`?mock=1&ws=1`, in-page mock server)

| case | observed |
|---|---|
| boot | 165,122 neurons, `body` present, 4 bundles, 1,402 neurons in the skeleton index, live connected |
| click-train preset | 380 neurons lit, t advancing, peak set SAD_L, spikes arriving per frame |
| pause / step / resume | status and dot change, `step 50` advances 50 ms while paused, resume returns to running |
| pick | bodyId shown from the synthetic `body` array; a drag over the same area changes nothing |
| `get_hops` | 38 one-hop, 235 two-hop, selection kept; clear removes the layers and keeps the selection |
| snapshot | `snapshot_done` path shown |
| disconnect | file transport unlocked, `run.json` playback resumes normally |

### 4.2 Real server and real assets

Served by `python -m flysim.live.server`, opened as `?ws=ws://127.0.0.1:8765/ws`.

| case | observed |
|---|---|
| link and handshake | URL used verbatim; `hello.assets` = `{neurons_3d: true, skeleton_sets: [AMMCtype, JO_AB, WED, pC1]}`, and those four bundles (1,017,601 segments) were fetched from that list |
| silence preset | 0 active, 0 spikes — the negative control is quiet |
| click-train preset | 571 neurons active, `JO_post_L` at 2.04 Hz and the peak region, i.e. the auditory path lights up |
| pick on real data | index 132 → **bodyId 10146**, set `AMMCtype_R`, source `somaLocation`, 35331/24014/36296 voxels = 282.6/192.1/290.4 μm, skeleton `AMMCtype 1,147 segments` |
| `get_hops` on the real graph | 15 one-hop → 902 two-hop, drawn across the central brain and down the nerve cord |
| static path (no engine) | the same file served without the server still loads, picks and highlights; the hops button is correctly disabled |
| errors | none in any run |

## 5. Performance, and why the absolute numbers here are not clean

**Another window held the GPU at 99 % for the whole measurement window** (`flysim.engine.bench`, then
`flysim.engine.sweep --m2e`). Per the planning session's instruction this is recorded rather than hidden.
Waiting it out was not practical — it was still running after ten minutes — so instead the comparison was
made **paired**: live mode and file mode measured back to back in the same browser, same geometry
(164,969 points, 1,017,601 segments), same 99 % contention, 400 frames each, twice.

| condition | median | p95 | median fps |
|---|---|---|---|
| live + 2-hop highlight + rotation | 8.4 / 9.2 ms | 13.9 / 13.8 ms | 119 / 109 |
| live + 2-hop highlight | 8.6 / 10.3 ms | 13.7 / 13.9 ms | 116 / 97 |
| file playback + 2-hop highlight + rotation | 10.2 / 9.8 ms | 14.0 / 13.9 ms | 98 / 102 |
| file playback + rotation | 8.9 / 8.4 ms | 13.9 / 13.9 ms | 112 / 119 |

Every condition lands in the same 8–10 ms band and every p95 is 13.7–14.0 ms regardless of what the
viewer is doing. That flat p95 across conditions is the other process's duty cycle, not this renderer:
**live mode and the highlight add no measurable cost over file playback.** For the uncontended figure the
M6c measurement stands — same geometry, GPU free: 0.60 ms static, 1.10 ms playing, 1.60 ms on the
1.19 M-spike run.

Worth stating plainly: even while another process pegged the GPU, the worst p95 was 14.0 ms, inside the
16.7 ms budget for 60 fps. The server reported `lagging` throughout, which is the engine failing to hold
real-time pacing under the same contention — a server-side symptom the viewer displayed correctly.

## 6. Deviations

1. **All eight protocol server messages are handled**, not the six the brief enumerated. `hops` and
   `snapshot_done` are the replies to `get_hops` and `snapshot`, which the brief does ask for; the list in
   the brief was abbreviated. Nothing outside `docs/m5-protocol.md` is sent or consumed.
2. **The mock now includes a mock live server** (`MockLiveServer`) speaking the same subset, so live mode,
   picking and hops can be exercised with no server at all. The brief asked for `?mock=1` to keep working
   "including the mock server"; this is that.
3. **`hello.assets` is read**, reversing the M6c decision. That decision was for a standalone page with no
   websocket; this brief gives the page one.
4. **Live mode locks file playback** rather than letting both run. Not specified, but two writers to the
   same readouts produce nonsense, which was observed before the lock was added.
5. **Rate bars in live mode are scaled by a fixed 200 Hz**, because the server sends raw Hz with no
   normalisation reference and this screen has no history buffer to take a rolling maximum from. The 2D
   cockpit, which does keep one, remains the place to read rates precisely.

## 7. Not implemented

- **No `frame.n_bins` handling for the time axis.** The 3D screen has no rolling time axis to advance —
  it draws the present moment — so the field is read but unused. The cockpit is where binned history lives.
- **No `min_contacts` control.** `get_hops` is sent with a fixed 5, matching the probe sets' `k_min`.
  Making it adjustable would be a parameter control, which this screen deliberately does not have.
- **Picking is a linear scan** over the drawn points (~165 k) per click, a few milliseconds. Fine for
  clicks, still not suitable for hover.
- **Highlight colours are fixed** for one-hop and two-hop; there is no legend beyond the panel text.
- **No upstream (`k` backwards) query**, because the protocol's `get_hops` is downstream only.
