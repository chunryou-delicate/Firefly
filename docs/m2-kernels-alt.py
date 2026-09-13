"""ALTERNATIVE (unused) M2 Triton kernels — kept for later synaptic-delay support.

Written in the planning session in parallel with the committed engine (8bae9d3).
Difference: the LIF kernel writes each step's spike vector into a ring-buffer
history (F x n int8) instead of atomic (t, idx) append; such a history is what a
fixed synaptic delay of k steps would read from. Not imported anywhere.

Original header follows.

M2: Triton kernels for one LIF step (see docs/m2-brief.md, "성능 설계").

Two launches per step, no host synchronisation, no data-dependent shapes:

  propagate_kernel  event-driven spike propagation over the forward CSR.
                    Each program walks NB presynaptic neurons; for those that
                    spiked it streams the CSR row and does int32 atomic adds into
                    acc[post]. Integer addition is associative, so the result is
                    bit-exact regardless of atomic ordering (determinism basis).
  lif_kernel        fused elementwise update: synaptic current, membrane
                    potential, refractory counter, threshold, reset, spike
                    output, accumulator clear, and (optionally) a write of the
                    spike vector into the recorder's history ring buffer.
"""
from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def propagate_kernel(spk_ptr, indptr_ptr, idx_ptr, w_ptr, acc_ptr, N,
                     NB: tl.constexpr, BLOCK: tl.constexpr):
    base = tl.program_id(0) * NB
    for k in range(NB):
        n = base + k
        if n < N:
            s = tl.load(spk_ptr + n)
            if s != 0:
                start = tl.load(indptr_ptr + n)
                end = tl.load(indptr_ptr + n + 1)
                for off in range(start, end, BLOCK):
                    e = off + tl.arange(0, BLOCK)
                    m = e < end
                    col = tl.load(idx_ptr + e, mask=m, other=0)
                    w = tl.load(w_ptr + e, mask=m, other=0)
                    tl.atomic_add(acc_ptr + col, w, mask=m)


@triton.jit
def lif_kernel(v_ptr, isyn_ptr, acc_ptr, iext_ptr, ref_ptr, spk_ptr, hist_ptr, t_ptr, N,
               decay_m, decay_s, g, v_rest, v_reset, v_th, t_ref_steps,
               F: tl.constexpr, RECORD: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < N
    v = tl.load(v_ptr + i, mask=m, other=0.0)
    isyn = tl.load(isyn_ptr + i, mask=m, other=0.0)
    acc = tl.load(acc_ptr + i, mask=m, other=0).to(tl.float32)
    iext = tl.load(iext_ptr + i, mask=m, other=0.0)
    ref = tl.load(ref_ptr + i, mask=m, other=0)

    isyn = isyn * decay_s + g * acc
    vn = v_rest + (v - v_rest) * decay_m + (1.0 - decay_m) * (isyn + iext)
    in_ref = ref > 0
    vn = tl.where(in_ref, v_reset, vn)          # held at reset during refractory
    s = vn >= v_th                               # cannot be true while in_ref (v_reset < v_th)
    vn = tl.where(s, v_reset, vn)
    ref = tl.where(s, t_ref_steps, tl.maximum(ref - 1, 0))

    tl.store(v_ptr + i, vn, mask=m)
    tl.store(isyn_ptr + i, isyn, mask=m)
    tl.store(ref_ptr + i, ref, mask=m)
    s8 = s.to(tl.int8)
    tl.store(spk_ptr + i, s8, mask=m)
    tl.store(acc_ptr + i, tl.zeros_like(acc).to(tl.int32), mask=m)
    if RECORD:
        t = tl.load(t_ptr)
        row = (t % F).to(tl.int64)
        tl.store(hist_ptr + row * N + i, s8, mask=m)
