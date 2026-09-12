"""M2: spike recorder.

Spikes are appended on the device as (t, idx) pairs into two fixed int32
buffers by the LIF kernel (``slot = atomic_add(counter, 1)``); nothing on the
host is touched per step. Every ``chunk_steps`` steps (or at ``collect``) the
engine calls ``flush``: the counter is read (the only host sync), the used
prefix is sorted by (t, idx) — atomic slot order is nondeterministic — and
moved to the CPU. If the counter exceeds ``cap`` the chunk is NOT silently
truncated: ``RecorderOverflow`` is raised (CLAUDE.md §3.2).
"""
from __future__ import annotations

import numpy as np
import torch


class RecorderOverflow(RuntimeError):
    """More spikes were produced within one flush interval than the buffer holds."""


class SpikeRecorder:
    def __init__(self, n_neurons: int, cap: int = 1 << 24, chunk_steps: int = 100,
                 device: torch.device | str = "cuda"):
        if cap <= 0 or chunk_steps <= 0:
            raise ValueError("cap and chunk_steps must be > 0")
        self.n = int(n_neurons)
        self.cap = int(cap)
        self.chunk_steps = int(chunk_steps)
        self.device = torch.device(device)
        # Device buffers written by the kernel / torch backend. Never reallocated
        # (a captured CUDA graph holds their addresses).
        self.counter = torch.zeros(1, dtype=torch.int32, device=self.device)
        self.buf_t = torch.zeros(self.cap, dtype=torch.int32, device=self.device)
        self.buf_i = torch.zeros(self.cap, dtype=torch.int32, device=self.device)
        self._chunks: list[tuple[np.ndarray, np.ndarray]] = []   # (t int32, idx int32) per flush, sorted
        self.steps_since_flush = 0
        self.total_flushed = 0

    # ---- device side -------------------------------------------------------
    def clear(self) -> None:
        self.counter.zero_()
        self._chunks = []
        self.steps_since_flush = 0
        self.total_flushed = 0

    def append(self, t: torch.Tensor, spikes: torch.Tensor) -> None:
        """torch-backend path (no fused kernel): append all set entries of ``spikes`` at step ``t``.

        ``t`` is a 1-element int32 device tensor. Uses the same buffers and the
        same overflow semantics as the kernel path (masked store, counter runs on).
        """
        idx = spikes.nonzero().flatten().to(torch.int32)
        k = idx.numel()
        if k == 0:
            return
        start = int(self.counter[0])  # torch backend syncs per step anyway (nonzero)
        end = min(start + k, self.cap)
        room = end - start
        if room > 0:
            self.buf_i[start:end] = idx[:room]
            self.buf_t[start:end] = t.to(torch.int32).expand(room)
        self.counter[0] = start + k

    def tick(self) -> bool:
        """Call once per recorded step; returns True when a flush is due."""
        self.steps_since_flush += 1
        return self.steps_since_flush >= self.chunk_steps

    # ---- host side ---------------------------------------------------------
    def flush(self) -> int:
        """Move the used prefix to the CPU (sorted). Returns the number of spikes moved.

        Raises RecorderOverflow if more than ``cap`` spikes were appended since
        the previous flush (the excess was dropped by the masked store, so the
        run is invalid; call ``clear``/``reset`` and use a larger cap or a
        smaller chunk_steps).
        """
        n_used = int(self.counter[0])          # host sync — only here
        self.steps_since_flush = 0
        if n_used > self.cap:
            raise RecorderOverflow(
                f"{n_used:,} spikes in one flush interval exceed recorder cap {self.cap:,} "
                f"(chunk_steps={self.chunk_steps}); increase cap or lower chunk_steps")
        if n_used > 0:
            key = self.buf_t[:n_used].to(torch.int64) * self.n + self.buf_i[:n_used].to(torch.int64)
            key = torch.sort(key).values          # (t, idx) order; makes atomic slot order irrelevant
            # decode on the device (host-side int64 division dominated collect() at high activity)
            t = (key // self.n).to(torch.int32)
            idx = (key - t.to(torch.int64) * self.n).to(torch.int32)
            self._chunks.append((t.cpu().numpy(), idx.cpu().numpy()))
            self.total_flushed += n_used
        self.counter.zero_()
        return n_used

    def collect(self) -> tuple[np.ndarray, np.ndarray]:
        """Flush, then return (t_bin int32, neuron_idx int32) of everything recorded so far
        (sorted by t then idx) and drop it from the recorder."""
        self.flush()
        if self._chunks:
            t = np.concatenate([c[0] for c in self._chunks])
            idx = np.concatenate([c[1] for c in self._chunks])
        else:
            t = np.empty(0, dtype=np.int32)
            idx = np.empty(0, dtype=np.int32)
        self._chunks = []
        return t, idx

    @property
    def nbytes(self) -> int:
        return self.buf_t.numel() * 4 * 2 + 4
