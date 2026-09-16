# M5b report — flysim-live cockpit (browser client)

Built 2026-09-16 against **contract v1.1** (`docs/m5-protocol.md`, commit `0f11e2e`), spec `docs/m5b-brief.md`.
Deliverables: `flysim-live.html` (single file, no build, no dependencies) and `tests/test_live_html.py` (46 tests).
The result player `flysim-viewer.html` is untouched and stays the way runs are submitted; the cockpit is a second,
exploratory surface. Verified only against the built-in mock server — the real M5a server did not exist yet.

## 1. What is on screen

One CSS grid: header / left panel / neuron field / right panel / region heatmap / status bar.

| area | contents |
|---|---|
| **header** | connection dot (live / paused / lagging / disconnected) with host, mock badge, sensory mode, `dt`, `speed`, `params_version`, and pause / step / reset |
| **stimulus panel** (left) | kind (silence, click_train, tone, audio) and the contract's IPI, carrier, pulse, ILD, duration controls; only the fields that belong to the selected kind are shown. "자극 적용" sends one `stimulus` message |
| **audio panel** (left) | microphone (`getUserMedia`) or file (`decodeAudioData`, looped), start/stop, the transmitted RMS, and the input-level meter fed by `frame.input_level` |
| **parameter panel** (left) | `synapse`, `mode`, `g` (log), `a_in` (log), `noise_sigma`, `adapt_b`, `adapt_tau_w`, `speed` (log), `dt_ms`; hints carry the documented operating points (conductance 3.162e-4, current 0.336, σ ≥ 22 ignites) so a slider is not moved blind. Text states that every control is logged server-side |
| **inject panel** (left) | amplitude, duration, and a set picker built from `hello.set_sizes` |
| **neuron field** (centre) | additive-composited `ImageData` with afterglow, the approach copied from `flysim-viewer.html` (credited in a comment). Click → `get_hops k=2`, the two returned layers tinted cool; Shift+click → `inject` on that one neuron; a "disconnected" curtain over the last frame |
| **L−R window** (right) | per set base name, `rates[L] − rates[R]` in Hz with a signed diverging bar: warm = left higher, cool = right higher, grey when exactly 0 |
| **hops / snapshot / display panels** (right) | highlight toggle, `min_contacts`, `run_id` + length + save, `set_frame_every`, afterglow |
| **heatmap** (bottom) | rolling 5 s of raw Hz per region, JO sets in their own labelled row group |
| **status bar** | host, engine synapse + commit, bin size and neuron count, msg/s and bin/s, last protocol event |

Keyboard: space toggles pause, `s` steps. Panels fold by clicking their heading; below 900 px the grid stacks to one
column and below 420 px the controls reflow. Measured at a 400 px viewport: no horizontal overflow.

## 2. Normalisation (the backlog item)

`docs/backlog.md` recorded that the viewer's `rate_norm_hz` is set by the single 1 ms bin in which every `JO_AB`
neuron fires at once, so every other region renders at ≤ 0.01. The contract removes the cause by sending **raw Hz**;
the cockpit does all scaling, and the default is **per-region maximum**, computed over the visible 5 s window.

| mode | denominator |
|---|---|
| **영역별 (default)** | that row's own maximum in the window |
| 전역 | the largest maximum across all rows |
| 로그 | `log1p(v/0.05) / log1p(max/0.05)`, global |
| 고정 | a Hz cap typed by the user |

Per-region scaling hides absolute size, so **each row prints its own maximum Hz at the right-hand end** in every
mode. In the mock the JO rows peak around 970 Hz while pC1 peaks near 4 Hz; under the old global scaling pC1 was
invisible, and with per-region scaling both rows show structure with their real magnitudes written next to them.

The 5 s window is a ring buffer of `5000 / bin_ms` columns. When `frame_every > 1` a message covers several bins;
the contract's `frame` has no group-size field, so the count is derived from the `t_ms` difference between
consecutive frames and the rates are written that many times. The time axis therefore stays a true 5 s at any
`frame_every` (measured: 968 bin/s at `frame_every = 10`, speed 1).

## 3. Audio path

`audio_start {sample_rate, channels:1}` → binary frames → `audio_stop`. Source is either a microphone stream or a
decoded file played through an `AudioBufferSourceNode`; both feed an `AudioWorklet` (loaded from a Blob URL, so no
external file) that posts raw blocks to the page, with a `ScriptProcessorNode` fallback when the worklet cannot be
constructed. Blocks are accumulated and cut into fixed **20 ms float32 mono** chunks, each sent as one binary frame
of the same socket. The transmitted RMS is displayed as a number; the meter bar stays reserved for the server's
`input_level` so the two are never confused.

## 4. Protocol handling (contract v1.1)

- Every incoming type has a named handler in one `SERVER_HANDLERS` table: `hello`, `frame`, `ack`, `params`,
  `hops`, `snapshot_done`, `warning`, `error`. Every outgoing type goes through one `send()` that stamps `req_id`
  and refuses any type not in `CLIENT_MESSAGE_TYPES`. The tests read both lists out of `docs/m5-protocol.md`, so
  the cockpit cannot silently drift from the contract — the v1.1 `params` message was caught by those tests
  before it was implemented.
- **Parameter values have exactly one entry point.** `applyParams()` is the only writer of `S.params`, and only
  `hello` and the `params` broadcast call it. Moving a slider sends `set_params` and marks that key dirty (⟳ next
  to the value); the displayed value stays at the last confirmed one until the broadcast arrives. An `ack` never
  writes values — that is what lets another client's change be shown correctly instead of guessed. A rejected
  change clears the dirty mark, so the display snaps back, and a rejected `adapt_b` also locks both adaptation
  sliders with the server's reason.
- **Version display.** `S.paramsVersion` is the version of what is on screen; `S.serverVersion` is the newest
  version any `ack` or `frame` mentioned. While the second is ahead the badge shows `⋯` and a tooltip naming both
  numbers, which is the honest state: the broadcast is in flight. (This assumes versions only increase, which the
  contract guarantees.)
- **Time axis.** `frame.n_bins` advances the heatmap, so a bundled message moves the window by the number of bins
  it actually covers. The earlier `t_ms`-delta derivation is gone.
- Warnings use the viewer's exact wording for `all_silent` and `runaway`, plus new text for `ignited` and
  `recorder_overflow`; `<code>_cleared` removes them.
- Reconnect every 2 s; the last frame stays on screen under a "disconnected" curtain.
- Rendering is one `requestAnimationFrame` per frame regardless of message rate — spikes from all messages since
  the last paint are accumulated, and every message is written to the heatmap.

## 5. Mock mode (`?mock=1`) and what was checked

`MockServer` implements the WebSocket surface in-page and speaks the contract. **Its neuron coordinates and firing
rates are synthetic** — not the connectome, not a simulation — and the header carries a permanent red badge saying
so. Only the set names and sizes are real, taken from `docs/m3-report.md` §2 (76 / 38 / 490 / … / 162,267, summing
to 165,122) so the screen proportions are not misleading.

Checked by driving the page in headless Chrome over the DevTools protocol (a scratch script, not part of the repo;
no browser automation was added to the test suite):

| action | observed |
|---|---|
| load | `hello` accepted: 15 regions, 165,122 neurons, params shown, `params v1` |
| select click train → 자극 적용 | `stimulus` sent; region rates rise in chain order (JO first, then JO_post, SAD, WED, AMMCtype, pC1); heatmap fills from the right |
| four normalisation buttons | mode switches, the Hz cap input enables only for 고정 |
| drag `g` to 1.259e-3 | value shows ⟳, `set_params` sent, on ack `params v2` and the ⟳ clears |
| move `adapt_b` | server refuses; both adaptation sliders disable, note reads "잠김 — adaptation은 M2c 전에는 0 고정" |
| pause → step → resume | dot turns amber, button flips to ▶, `t_ms` advances 5475 → 5525 for 50 steps, resume restores "running" |
| click a neuron | `get_hops` sent, layers returned and tinted, panel reads "뉴런 51192 하류 2홉: 15 → 64 개" |
| Shift+click | `inject` sent for that single neuron |
| snapshot | `snapshot_done` path shown with the "not for submission" wording, `run_id` auto-increments to live-001 |
| `noise_sigma` 30 | server raises `runaway`; the viewer's runaway sentence appears over the field; returning to 0 clears it |
| audio messages | `audio_start`, one binary frame, `audio_stop` in that order |
| `frame_every` 10 | 110 msg/s but 919 bin/s from `n_bins` — the heatmap time axis is unchanged |
| drag `g` with the mock's reply delayed 400 ms | ⟳ appears, the label shows the requested 1.905e-3 while `S.params.g` stays at the old 3.162e-4; when the `params` broadcast lands the value is adopted and ⟳ clears |
| a second client changes `a_in` and `noise_sigma` (injected straight into the mock, bypassing this page) | the `params` broadcast moves both sliders (320 → 2560, 0 → 7) and the badge goes to v3 without this cockpit having sent anything |
| force the badge to a version ahead of the screen | badge reads `params v6 ⋯`, tooltip "서버는 v6, 화면은 v3 — params 브로드캐스트 대기 중" |
| `speed` to 4 | mock reports `lagging`; the dot turns amber and the header reads "연결됨 · lagging"; back at 1 it returns to running |
| close the socket | curtain appears, reconnects on its own after 2 s and re-runs `hello` |
| 400 px viewport | single column, no horizontal overflow |

No uncaught JavaScript errors in any of these runs.

One real defect was found and fixed this way: panel notes were written as `class="note warn"`, which also matched
the field-overlay rule `.warn { position:absolute; bottom:10px }` and tore those notes out of the panel, painting
them across the status bar. Panel notes now use `.note.caution`, and a test asserts `class="note warn"` never
reappears.

## 6. Limits

- **Not verified against the real server.** Everything here was exercised against the mock. `hello.sets`,
  `neurons_url` fetching, real spike volumes, `lagging`, and `recorder_overflow` still need a pass against M5a.
- `neurons_url` is fetched once; if it fails the field stays empty and the status line says so. The mock supplies
  coordinates inline instead (`neurons_inline`), which is a mock-only field and is never expected from a server.
- The field maps one neuron to one pixel with additive blending. At 165 k neurons in ~500 px this saturates dense
  regions; it shows where activity is, not how much. The readout above it (`active_frac_100ms` × N) is the number.
- Nearest-neuron picking is a linear scan over all neurons per click (~2 ms). Fine for clicks, not for hover.
- The L−R window pairs sets by an `_L` / `_R` suffix, so `_unk` sets and `rest` never appear there.
- Recorded-but-unshown: `frame.step`, `hops.k` beyond the layer counts, and `engine.git_commit` beyond 7 characters.
- Mock `speed` only paces bin generation; it does not model the server's wall-clock pacing or `lagging`.

## 7. Contract history

Four gaps were reported after the first build and all four were closed in contract **v1.1** (`0f11e2e`); this
cockpit implements that version.

| gap (v1.0) | resolution (v1.1) | in the cockpit |
|---|---|---|
| `hello` had no `params_version` | added | used as the baseline instead of a `?? 0` guess |
| `frame` carried a version but no values, so another client's change could only be flagged | new `params` broadcast carrying the whole object on every version change | `onParams` → `applyParams`; sliders follow other clients, the `*` marker is replaced by a `⋯` "broadcast in flight" state |
| no group-size field, so a bundled `frame` had to be sized from `t_ms` deltas (wrong right after a `reset`) | `frame.n_bins` added | drives the heatmap time axis; the derivation is removed |
| `lagging` appeared in the duties text but not in the `status` list | listed | already accepted; now exercised in the mock |

The mock server speaks v1.1 as well: it broadcasts `params` after every accepted `set_params`, reports `n_bins`,
and reports `lagging` when one tick would have to generate more than 60 bins.
