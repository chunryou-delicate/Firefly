"""M2: event-driven LIF engine on the full connectome (torch + Triton, GPU resident).

Design (docs/m2-brief.md "성능 설계"): one step = two fused kernels.

  1. ``propagate``: for every neuron that spiked in the previous step, walk its
     forward-CSR row (pre -> post) and ``acc[post] += w`` with int32 atomics.
     Weights are signed synaptic contact counts (exact integers), so the
     accumulation is bit-exact regardless of atomic ordering — the basis of
     the determinism guarantee.
  2. ``lif``: i_syn / v / refractory update, threshold, reset, acc <- 0, and
     spike logging into the recorder buffers (device-side atomic slot counter).

No host synchronisation inside a step. The step index is a device tensor so
the whole step can be captured in a CUDA graph (``use_cuda_graph=True``).

Backends: ``"triton"`` (CUDA, default) and ``"torch"`` (plain torch ops:
``nonzero`` + ``index_add_``; the CPU fallback and the reference the Triton
kernels are tested against). Bit-exactness is only guaranteed for repeated
runs on the same backend/device (the float ops may be fused differently).

M2b (docs/m2b-brief.md): ``params.synapse == "conductance"`` selects
conductance-based synapses with reversal potentials. Propagation then uses two
int32 accumulators (``acc_e`` for excitatory contacts, ``acc_i`` for |inhibitory|
contacts) in a separate kernel, so the determinism basis is unchanged; the
current model keeps its original single-accumulator kernel byte for byte
(regression: tests/test_engine_conductance.py, data-provenance/m2-regression-spikes.npz).

M2c (docs/m2c-brief.md): spike-frequency adaptation. One extra float32 state ``w``
per neuron, decayed every step and incremented by ``params.adapt_b`` on each spike;
the membrane update sees ``i_ext - w``. It is a per-neuron quantity with no
cross-thread reduction, so the determinism basis (int32 atomics only) is unchanged.
Guarded by an ``ADAPT`` constexpr that is False when ``adapt_b == 0``, so the
default engine compiles to the pre-M2c kernel (regression: real_cond_default in
data-provenance/m2-regression-spikes.npz).

M2d (docs/m2d-brief.md): the same adaptation as a *conductance* ``g_a`` (float32 state)
instead of a current. It enters both the total conductance ``G`` and ``v_inf`` with its
own reversal potential ``E_adapt``, so it is bounded by construction — the fix for M2c's
-834 mV. Conductance synapses only, guarded by an ``ADAPT_G`` constexpr that is False
when ``adapt_g_b == 0``, so the default engine still compiles to the pre-M2d kernel.

Model equations and all assumptions: flysim/engine/params.py, docs/m2-report.md,
docs/m2b-report.md, docs/m2c-report.md, docs/m2d-report.md.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from .params import EngineParams
from .recorder import SpikeRecorder

try:  # triton ships with the cu124 torch wheel; needs setuptools to import
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception as _e:  # pragma: no cover - environment dependent
    triton = None
    tl = None
    _HAS_TRITON = False
    _TRITON_IMPORT_ERROR = _e

# Launch configuration (benchmarked in docs/m2-brief.md; re-measured by bench.py).
PROP_NEURONS_PER_PROGRAM = 32   # one program scans 32 presynaptic neurons
PROP_BLOCK = 128                # row chunk (edges) per atomic-add vector
PROP_WARPS = 4
LIF_BLOCK = 1024


# ----------------------------------------------------------------------------
# graph container (real Graph or synthetic)
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class CSRGraph:
    """Minimal forward-CSR container with the attributes the engine reads
    (``n``, ``indptr``, ``indices``, ``weight``); ``flysim.graph.Graph`` is
    duck-type compatible. Used for synthetic graphs in tests and examples."""
    indptr: np.ndarray
    indices: np.ndarray
    weight: np.ndarray

    @property
    def n(self) -> int:
        return len(self.indptr) - 1

    @property
    def m(self) -> int:
        return len(self.indices)

    @classmethod
    def from_edges(cls, n: int, pre, post, weight) -> "CSRGraph":
        pre = np.asarray(pre, dtype=np.int64)
        post = np.asarray(post, dtype=np.int64)
        weight = np.asarray(weight, dtype=np.float32)
        if len(pre) != len(post) or len(pre) != len(weight):
            raise ValueError("pre, post, weight must have equal length")
        if len(pre) and (pre.min() < 0 or pre.max() >= n or post.min() < 0 or post.max() >= n):
            raise ValueError("edge endpoints out of range")
        order = np.lexsort((post, pre))
        pre, post, weight = pre[order], post[order], weight[order]
        indptr = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(np.bincount(pre, minlength=n), out=indptr[1:])
        return cls(indptr, post.astype(np.int32), weight)


# ----------------------------------------------------------------------------
# Triton kernels
# ----------------------------------------------------------------------------
if _HAS_TRITON:

    @triton.jit
    def _propagate_kernel(spk_ptr, indptr_ptr, idx_ptr, w_ptr, acc_ptr, N,
                          NB: tl.constexpr, BLOCK: tl.constexpr):
        """acc[post] += w for every edge of every neuron with spk != 0 (int32 atomics)."""
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
    def _propagate_split_kernel(spk_ptr, indptr_ptr, idx_ptr, w_ptr, acc_e_ptr, acc_i_ptr, N,
                                NB: tl.constexpr, BLOCK: tl.constexpr):
        """M2b: acc_e[post] += w (w > 0), acc_i[post] += -w (w < 0); int32 atomics, order-independent."""
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
                        tl.atomic_add(acc_e_ptr + col, w, mask=m & (w > 0))
                        tl.atomic_add(acc_i_ptr + col, -w, mask=m & (w < 0))

    @triton.jit
    def _lif_kernel(v_ptr, isyn_ptr, ge_ptr, gi_ptr, acc_ptr, acc_e_ptr, acc_i_ptr,
                    iext_ptr, noise_ptr, ref_ptr, spk_ptr, w_ptr,
                    cnt_ptr, out_t_ptr, out_i_ptr, t_ptr, N, cap,
                    decay_m, decay_s, c_s, g, v_rest, v_reset, v_th, v_floor, noise_scale, ref_steps,
                    decay_e, decay_i, avg_e, avg_i, inv_tau_m, E_e, E_i, dt, decay_w, adapt_b,
                    ga_ptr, decay_a, avg_a, adapt_g_b, E_a,
                    SYNAPSE: tl.constexpr, NOISE: tl.constexpr, RECORD: tl.constexpr,
                    ADAPT: tl.constexpr, ADAPT_G: tl.constexpr, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = i < N
        t = tl.load(t_ptr)
        v = tl.load(v_ptr + i, mask=m, other=0.0)
        iext = tl.load(iext_ptr + i, mask=m, other=0.0)
        if NOISE:
            iext = iext + noise_scale * tl.load(noise_ptr + i, mask=m, other=0.0)
        if ADAPT:
            # M2c: w decays, then occupies the external-current slot as (i_ext - w).
            # Everything below is textually unchanged, so ADAPT = False reproduces
            # the pre-M2c codegen exactly (params.py).
            w = tl.load(w_ptr + i, mask=m, other=0.0) * decay_w
            iext = iext - w
        ref = tl.load(ref_ptr + i, mask=m, other=0)
        if SYNAPSE == 0:
            # M2 current model: exponential synapse, exact one-step membrane integration (params.py)
            isyn = tl.load(isyn_ptr + i, mask=m, other=0.0)
            acc = tl.load(acc_ptr + i, mask=m, other=0).to(tl.float32)
            isyn = isyn * decay_s + g * acc
            vn = v_rest + (v - v_rest) * decay_m + (1.0 - decay_m) * iext + c_s * isyn
            tl.store(isyn_ptr + i, isyn, mask=m)
            tl.store(acc_ptr + i, tl.zeros_like(i), mask=m)   # acc <- 0 for the next propagate
        else:
            # M2b conductance model (params.py module doc): per-neuron total conductance G,
            # exponential Euler towards v_inf with rate G.
            ge = tl.load(ge_ptr + i, mask=m, other=0.0)
            gi = tl.load(gi_ptr + i, mask=m, other=0.0)
            acc_e = tl.load(acc_e_ptr + i, mask=m, other=0).to(tl.float32)
            acc_i = tl.load(acc_i_ptr + i, mask=m, other=0).to(tl.float32)
            ge = ge * decay_e + g * acc_e
            gi = gi * decay_i + g * acc_i
            ge_bar = ge * avg_e                      # step-average conductance (params.py)
            gi_bar = gi * avg_i
            if ADAPT_G:
                # M2d: the adaptation conductance joins G and v_inf with its own
                # reversal potential, so it can only pull v towards E_a, never past it.
                ga = tl.load(ga_ptr + i, mask=m, other=0.0) * decay_a
                ga_bar = ga * avg_a
                G = inv_tau_m + ge_bar + gi_bar + ga_bar
                vinf = ((v_rest + iext) * inv_tau_m + ge_bar * E_e + gi_bar * E_i
                        + ga_bar * E_a) / G
            else:
                G = inv_tau_m + ge_bar + gi_bar
                vinf = ((v_rest + iext) * inv_tau_m + ge_bar * E_e + gi_bar * E_i) / G
            vn = vinf + (v - vinf) * tl.exp(-dt * G)
            tl.store(ge_ptr + i, ge, mask=m)
            tl.store(gi_ptr + i, gi, mask=m)
            tl.store(acc_e_ptr + i, tl.zeros_like(i), mask=m)
            tl.store(acc_i_ptr + i, tl.zeros_like(i), mask=m)
        vn = tl.maximum(vn, v_floor)
        vn = tl.where(ref > 0, v_reset, vn)          # refractory: held at v_reset, cannot spike
        s = vn >= v_th
        vn = tl.where(s, v_reset, vn)
        ref = tl.where(s, ref_steps, tl.maximum(ref - 1, 0))
        if ADAPT:
            tl.store(w_ptr + i, tl.where(s, w + adapt_b, w), mask=m)
        if ADAPT_G:
            tl.store(ga_ptr + i, tl.where(s, ga + adapt_g_b, ga), mask=m)
        tl.store(v_ptr + i, vn, mask=m)
        tl.store(ref_ptr + i, ref, mask=m)
        tl.store(spk_ptr + i, s.to(tl.int8), mask=m)
        if RECORD:
            sm = s & m
            slot = tl.atomic_add(cnt_ptr + tl.zeros_like(i), 1, mask=sm)
            ok = sm & (slot < cap)                 # counter keeps running past cap -> overflow detected at flush
            tl.store(out_t_ptr + slot, tl.zeros_like(i) + t, mask=ok)
            tl.store(out_i_ptr + slot, i, mask=ok)


# ----------------------------------------------------------------------------
# torch reference implementation (CPU fallback and test oracle)
# ----------------------------------------------------------------------------
def propagate_reference(spikes: torch.Tensor, indptr: torch.Tensor, indices: torch.Tensor,
                        weight_i32: torch.Tensor, acc: torch.Tensor) -> None:
    """acc[post] += w over the rows of all spiking neurons, using nonzero + index_add_.

    Same semantics as ``_propagate_kernel`` (int32 accumulation into ``acc`` in place).
    """
    rows = spikes.nonzero().flatten()
    if rows.numel() == 0:
        return
    starts = indptr[rows].to(torch.int64)
    counts = indptr[rows + 1].to(torch.int64) - starts
    total = int(counts.sum())
    if total == 0:
        return
    rel = torch.arange(total, device=acc.device) - torch.repeat_interleave(
        torch.cumsum(counts, 0) - counts, counts)
    slots = torch.repeat_interleave(starts, counts) + rel
    acc.index_add_(0, indices[slots].to(torch.int64), weight_i32[slots])


def propagate_reference_split(spikes: torch.Tensor, indptr: torch.Tensor, indices: torch.Tensor,
                              weight_i32: torch.Tensor, acc_e: torch.Tensor, acc_i: torch.Tensor) -> None:
    """M2b reference for ``_propagate_split_kernel``: acc_e += w (w > 0), acc_i += -w (w < 0)."""
    rows = spikes.nonzero().flatten()
    if rows.numel() == 0:
        return
    starts = indptr[rows].to(torch.int64)
    counts = indptr[rows + 1].to(torch.int64) - starts
    total = int(counts.sum())
    if total == 0:
        return
    rel = torch.arange(total, device=acc_e.device) - torch.repeat_interleave(
        torch.cumsum(counts, 0) - counts, counts)
    slots = torch.repeat_interleave(starts, counts) + rel
    w = weight_i32[slots]
    post = indices[slots].to(torch.int64)
    acc_e.index_add_(0, post, w.clamp(min=0))
    acc_i.index_add_(0, post, (-w).clamp(min=0))


# ----------------------------------------------------------------------------
# engine
# ----------------------------------------------------------------------------
class LIFEngine:
    """Event-driven current-based LIF network on a forward-CSR graph.

    ``graph``: ``flysim.graph.Graph`` or ``CSRGraph`` (needs ``n``, ``indptr``,
    ``indices``, ``weight``; weights must be integer-valued — they are used as int32).
    """

    def __init__(self, graph, params: EngineParams | None = None, device: str | torch.device = "cuda",
                 backend: str | None = None, use_cuda_graph: bool = False,
                 record_cap: int = 1 << 24, chunk_steps: int = 100):
        self._params = params or EngineParams()
        self.n = int(graph.n)
        dev = torch.device(device)
        if dev.type == "cuda" and not torch.cuda.is_available():
            warnings.warn("CUDA not available: falling back to the CPU torch backend, which is "
                          "orders of magnitude slower than the GPU target (docs/m2-brief.md).")
            dev = torch.device("cpu")
        self.device = dev
        if backend is None:
            backend = "triton" if (dev.type == "cuda" and _HAS_TRITON) else "torch"
        if backend == "triton":
            if dev.type != "cuda":
                raise ValueError("triton backend requires a CUDA device")
            if not _HAS_TRITON:
                raise RuntimeError(f"triton backend requested but triton failed to import: {_TRITON_IMPORT_ERROR!r}")
        elif backend != "torch":
            raise ValueError(f"unknown backend {backend!r}")
        self.backend = backend
        self.use_cuda_graph = bool(use_cuda_graph) and backend == "triton"

        # ---- graph arrays (forward CSR, int32) --------------------------------
        w = np.asarray(graph.weight)
        if not np.all(w == np.round(w)):
            raise ValueError("edge weights must be integer-valued (signed synaptic contact counts)")
        if np.abs(w).max(initial=0) >= 2**31:
            raise ValueError("edge weight exceeds int32")
        if graph.indptr[-1] != len(graph.indices) or len(graph.indices) >= 2**31:
            raise ValueError("indptr/indices inconsistent or too large for int32 offsets")
        self.m = int(len(graph.indices))
        self.indptr = torch.as_tensor(np.asarray(graph.indptr, dtype=np.int32), device=dev)
        self.indices = torch.as_tensor(np.asarray(graph.indices, dtype=np.int32), device=dev)
        self.weight = torch.as_tensor(w.astype(np.int32), device=dev)

        # ---- state (never reallocated; CUDA graphs hold the addresses) ----------
        n = self.n
        f32 = dict(dtype=torch.float32, device=dev)
        i32 = dict(dtype=torch.int32, device=dev)
        self.v = torch.full((n,), self.params.v_rest, **f32)
        self.i_syn = torch.zeros(n, **f32)                        # current model
        self.g_e = torch.zeros(n, **f32)                          # conductance model
        self.g_i = torch.zeros(n, **f32)
        self.acc = torch.zeros(n, **i32)                          # current model accumulator
        self.acc_e = torch.zeros(n, **i32)                        # conductance model accumulators
        self.acc_i = torch.zeros(n, **i32)
        self.w = torch.zeros(n, **f32)                            # M2c adaptation current
        self.g_a = torch.zeros(n, **f32)                          # M2d adaptation conductance
        self.ref = torch.zeros(n, **i32)
        self._s = torch.zeros(n, dtype=torch.int8, device=dev)   # spikes of the last step (0/1)
        self.i_ext = torch.zeros(n, **f32)                        # external current buffer
        self._noise = torch.zeros(n, **f32)
        self._t = torch.zeros(1, **i32)                           # step index, device resident
        self.recorder = SpikeRecorder(n, cap=record_cap, chunk_steps=chunk_steps, device=dev)
        self._gen = torch.Generator(device=dev)
        self._graphs: dict[bool, "torch.cuda.CUDAGraph"] = {}
        self.reset(0)

    # ---- state management --------------------------------------------------
    @property
    def params(self) -> EngineParams:
        return self._params

    @params.setter
    def params(self, p: EngineParams) -> None:
        """Replace parameters in place (state buffers are kept). Captured CUDA
        graphs bake the scalar arguments in, so they are dropped and re-captured."""
        self._params = p
        self._graphs = {}

    def reset(self, seed: int = 0) -> None:
        """Reset all state to rest, t = 0, empty recorder; seed the RNG (determinism origin)."""
        torch.manual_seed(seed)
        self._gen.manual_seed(seed)
        self.seed = int(seed)
        self.v.fill_(self.params.v_rest)
        self.i_syn.zero_()
        self.g_e.zero_()
        self.g_i.zero_()
        self.acc.zero_()
        self.acc_e.zero_()
        self.acc_i.zero_()
        self.w.zero_()
        self.g_a.zero_()
        self.ref.zero_()
        self._s.zero_()
        self.i_ext.zero_()
        self._noise.zero_()
        self._t.zero_()
        self.recorder.clear()

    @property
    def spikes(self) -> torch.Tensor:
        """Bool view (zero-copy) of the spikes produced by the last step."""
        return self._s.view(torch.bool)

    @property
    def t(self) -> int:
        """Current step index (host sync)."""
        return int(self._t[0])

    def gpu_memory_bytes(self) -> int:
        arrays = (self.indptr, self.indices, self.weight, self.v, self.i_syn, self.g_e, self.g_i,
                  self.acc, self.acc_e, self.acc_i, self.w, self.g_a, self.ref, self._s,
                  self.i_ext, self._noise, self._t)
        return sum(a.numel() * a.element_size() for a in arrays) + self.recorder.nbytes

    # ---- one step ----------------------------------------------------------
    def step(self, i_ext: torch.Tensor | None = None, record: bool = True) -> torch.Tensor:
        """Advance one step. ``i_ext``: tensor of length n (copied into the input buffer) or None (= 0).

        Returns a bool tensor view of this step's spikes. No host sync inside.
        Recorded spikes are flushed to the CPU every ``chunk_steps`` steps.
        """
        if i_ext is None:
            self.i_ext.zero_()
        else:
            self.i_ext.copy_(i_ext)
        self._advance(record)
        return self.spikes

    def _advance(self, record: bool) -> None:
        p = self.params
        if p.noise_sigma > 0:
            # RNG outside the captured region (CUDA graph) -> buffer
            self._noise.normal_(0.0, 1.0, generator=self._gen)
        if self.backend == "triton":
            if self.use_cuda_graph:
                self._graph(record).replay()
            else:
                self._kernels(record)
        else:
            self._step_torch(record)
        if record and self.recorder.tick():
            self.recorder.flush()

    def _kernels(self, record: bool) -> None:
        p = self.params
        n = self.n
        cond = p.synapse == "conductance"
        if cond:
            _propagate_split_kernel[(triton.cdiv(n, PROP_NEURONS_PER_PROGRAM),)](
                self._s, self.indptr, self.indices, self.weight, self.acc_e, self.acc_i, n,
                NB=PROP_NEURONS_PER_PROGRAM, BLOCK=PROP_BLOCK, num_warps=PROP_WARPS)
        else:
            _propagate_kernel[(triton.cdiv(n, PROP_NEURONS_PER_PROGRAM),)](
                self._s, self.indptr, self.indices, self.weight, self.acc, n,
                NB=PROP_NEURONS_PER_PROGRAM, BLOCK=PROP_BLOCK, num_warps=PROP_WARPS)
        rec = self.recorder
        _lif_kernel[(triton.cdiv(n, LIF_BLOCK),)](
            self.v, self.i_syn, self.g_e, self.g_i, self.acc, self.acc_e, self.acc_i,
            self.i_ext, self._noise, self.ref, self._s, self.w,
            rec.counter, rec.buf_t, rec.buf_i, self._t, n, rec.cap,
            p.decay_m, p.decay_syn, p.c_syn, p.g, p.v_rest, p.v_reset, p.v_thresh,
            float("-inf") if p.v_floor is None else p.v_floor, p.noise_scale, p.ref_steps,
            p.decay_e, p.decay_i, p.avg_e, p.avg_i, p.inv_tau_m, p.E_exc, p.E_inh, p.dt,
            p.decay_w, p.adapt_b,
            self.g_a, p.decay_a, p.avg_a, p.adapt_g_b, p.E_adapt,
            SYNAPSE=1 if cond else 0, NOISE=p.noise_sigma > 0, RECORD=record,
            ADAPT=p.adapt, ADAPT_G=p.adapt_g, BLOCK=LIF_BLOCK)
        self._t.add_(1)

    def _graph(self, record: bool) -> "torch.cuda.CUDAGraph":
        g = self._graphs.get(record)
        if g is None:
            # Warm-up + capture mutate the state; snapshot and restore so that
            # capturing is invisible to the simulation.
            state = (self.v, self.i_syn, self.g_e, self.g_i, self.acc, self.acc_e, self.acc_i,
                     self.w, self.g_a, self.ref, self._s, self._t, self.recorder.counter)
            snap = [x.clone() for x in state]
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    self._kernels(record)
            torch.cuda.current_stream().wait_stream(side)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                self._kernels(record)
            for dst, src in zip(state, snap):
                dst.copy_(src)
            torch.cuda.synchronize()
            self._graphs[record] = g
        return g

    def _step_torch(self, record: bool) -> None:
        """Plain-torch implementation of the same update (CPU fallback / reference)."""
        p = self.params
        i_in = self.i_ext
        if p.noise_sigma > 0:
            i_in = i_in + p.noise_scale * self._noise
        if p.adapt:
            self.w.mul_(p.decay_w)              # M2c: external-current slot becomes i_ext - w
            i_in = i_in - self.w
        if p.synapse == "conductance":
            propagate_reference_split(self._s, self.indptr, self.indices, self.weight, self.acc_e, self.acc_i)
            self.g_e.mul_(p.decay_e).add_(self.acc_e.to(torch.float32), alpha=p.g)
            self.g_i.mul_(p.decay_i).add_(self.acc_i.to(torch.float32), alpha=p.g)
            ge_bar, gi_bar = self.g_e * p.avg_e, self.g_i * p.avg_i
            G = p.inv_tau_m + ge_bar + gi_bar
            num = (p.v_rest + i_in) * p.inv_tau_m + ge_bar * p.E_exc + gi_bar * p.E_inh
            if p.adapt_g:                       # M2d: adaptation conductance (params.py)
                self.g_a.mul_(p.decay_a)
                ga_bar = self.g_a * p.avg_a
                G = G + ga_bar
                num = num + ga_bar * p.E_adapt
            vinf = num / G
            vn = vinf + (self.v - vinf) * torch.exp(-p.dt * G)
            self.acc_e.zero_()
            self.acc_i.zero_()
        else:
            propagate_reference(self._s, self.indptr, self.indices, self.weight, self.acc)
            self.i_syn.mul_(p.decay_syn).add_(self.acc.to(torch.float32), alpha=p.g)
            vn = p.v_rest + (self.v - p.v_rest) * p.decay_m + (1.0 - p.decay_m) * i_in + p.c_syn * self.i_syn
            self.acc.zero_()
        if p.v_floor is not None:
            vn.clamp_(min=p.v_floor)
        vn = torch.where(self.ref > 0, torch.full_like(vn, p.v_reset), vn)
        s = vn >= p.v_thresh
        vn = torch.where(s, torch.full_like(vn, p.v_reset), vn)
        if p.adapt:
            self.w.add_(s.to(self.w.dtype), alpha=p.adapt_b)
        if p.adapt_g:
            self.g_a.add_(s.to(self.g_a.dtype), alpha=p.adapt_g_b)
        self.ref.copy_(torch.where(s, torch.full_like(self.ref, p.ref_steps), (self.ref - 1).clamp_(min=0)))
        self.v.copy_(vn)
        self._s.copy_(s.to(torch.int8))
        if record:
            self.recorder.append(self._t, s)
        self._t.add_(1)

    # ---- loop --------------------------------------------------------------
    def run(self, n_steps: int, i_ext_fn: Callable[[int, torch.Tensor], torch.Tensor | None] | None = None,
            record: bool = True) -> tuple[np.ndarray, np.ndarray] | None:
        """Run ``n_steps`` steps from the current state.

        ``i_ext_fn(t_step, buf)`` is called before each step with the step index
        and the engine's input buffer (float32, length n, device resident). The
        convention is to write into ``buf`` in place and return None; a returned
        tensor is copied into ``buf`` instead (slower). The buffer persists
        between steps — the callback owns its contents. ``None`` = no input.

        Returns ``(t_bin int32, neuron_idx int32)`` sorted by (t, idx) when
        ``record`` is True, else None.
        """
        if n_steps < 0:
            raise ValueError("n_steps must be >= 0")
        if i_ext_fn is None:
            self.i_ext.zero_()
        t0 = self.t
        for k in range(n_steps):
            if i_ext_fn is not None:
                out = i_ext_fn(t0 + k, self.i_ext)
                if out is not None:
                    self.i_ext.copy_(out)
            self._advance(record)
        if record:
            return self.recorder.collect()
        return None
