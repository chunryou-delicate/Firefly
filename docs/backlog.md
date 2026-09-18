# Backlog (planning session)

Items deferred with a reason. Not scheduled until a milestone needs them.

## From the M2 code review (M2 session, 2026-09-13, on 8bae9d3 — no blocking defects)

**Recorder performance** (only matters above ~1 % of neurons spiking per step; M3 regime is 0.2–0.65 %):
1. Double-buffer the flush: stack (t, idx) into one (2, k) int32 tensor, non_blocking copy into a pinned host
   buffer with a CUDA event, consume the previous chunk at the next flush. Removes D2H from the critical path.
2. Ring-buffer spike history (F × N int8) + one `nonzero` per chunk (see `docs/m2-kernels-alt.py`). Output is
   already (t, idx)-ordered, so no sort/atomics; cost fixed at ~16.5 MB read per chunk. Needed anyway if
   synaptic delays are ever added.
3. Aggregate-only recording mode (per-neuron counts) if a probe needs it.

**Documentation / API hygiene** (low):
- `run(record=True)` returns chunks accumulated by earlier manual `step(record=True)` calls too; document or clear at run start.
- `step(i_ext)` with a CPU tensor does a synchronous pageable H2D copy every step; state in `lif.py` docstring
  that adapters must write into the device buffer (M3 brief already requires this).
- After `RecorderOverflow`, the counter stays > cap until `clear()`/`reset()`; message already says so.
- `docs/m2-kernels-alt.py` and `docs/m2-prototype-step.py` scale noise by sqrt(dt); the engine's sqrt(1/dt) is the
  correct one. Do not copy the noise line from those files.

## Model
- Conductance-based synapses (reversal potentials) to remove unphysical hyperpolarisation in the ignited state.
  See `data-provenance/parameter-decisions.md` (2026-09-13). Decide after M3/M4.
- Ignition threshold may be lower for correlated multi-neuron input (JO) than for 100 random neurons.
  M3 brief's ignition check covers it per run; if M3/M4 runs ignite at g = 0.336, revisit the g rule.

## From M3 verification (planning session, 2026-09-13)
- **Viewer normalisation**: `rate_norm_hz` is set by the 1 ms bin in which every JO_AB neuron fires at once
  (1000 Hz), so every other region renders at ≤ 0.01 in the heatmap. Options: export a coarser `bin_ms` for
  `frames.rates` only, or a log/percentile scale in the viewer. Viewer change is the user's call.
- **Silence-control runs trigger the viewer's "발화 뉴런 없음" warning by design.** The CLAUDE.md rule
  ("runs with a warning are not submitted") is about results; controls are exempt and say so in `meta.run_id`.
  Consider a `meta.is_control` flag if the viewer ever grows a control mode.
- **Propagation stops at hop 1** at g = 0.336 with no background activity (SAD/AMMCtype/pC1 = 0 Hz at every
  a_in). M4 must decide, before running, whether to add background noise at a rule-chosen sigma.

## From M6 (2026-09-18)
- **조종석이 `hello.assets`를 아직 읽지 않는다.** 프로토콜 계약은 v1.2, 조종석은 v1.1이다. 조종석에 3D 필드를 붙일 때 함께 처리한다. `docs/m5-protocol.md` 맨 위에 구현 현황으로 적어 두었고 `tests/test_live_html.py`가 선언 버전과 계약 버전을 대조한다.
- **독립 3D 뷰어는 웹소켓을 쓰지 않는다.** HTTP로 자산을 직접 물어 404면 "아직 없음"으로 다룬다. 총괄 판단: 그대로 둔다. 자산 광고는 조종석 통합 때 의미가 생긴다.
- **뉴런 집기(클릭 → bodyId·하류 강조) 미구현.** 묶음의 `seg_offset`/`seg_count`가 이미 있어 붙이기 쉽다. 조종석의 `get_hops`와 엮으면 자연스럽다.
- **`tests/test_live.py::test_real_graph_end_to_end`의 벽시계 단언을 1,000 → 5,000 μs/빈으로 완화**했다(창 B). 여러 창이 한 노트북을 나눠 쓰면 부하 때문에 실패한다. 성능 수치의 출처는 `python -m flysim.live.bench`이고 이 단언은 회귀 가드로만 남았다.
- 3D 자산에 `Cache-Control: no-store`라 새로고침마다 2.3 MB + 묶음을 다시 받는다. 로컬이라 문제 없음. 필요해지면 ETag.
