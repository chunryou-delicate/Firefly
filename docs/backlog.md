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
