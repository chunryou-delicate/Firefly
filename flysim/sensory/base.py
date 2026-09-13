"""M3: common sensory-adapter interface (docs/m3-brief.md "sensory 공통 인터페이스").

An adapter turns a time signal into an injected current for a fixed set of
target neurons. The engine calls ``inject(t_step, buf)`` before every step
(flysim.engine.lif.LIFEngine.run convention): write into ``buf`` in place and
return None. ``prepare`` precomputes the whole current table on the device so
that ``inject`` is two small GPU ops and no host work.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch


class SensoryAdapter(ABC):
    target_idx: np.ndarray          # neuron indices that receive current (int64)

    @abstractmethod
    def prepare(self, dt_ms: float, n_steps: int) -> None:
        """Precompute the per-step current table, shape [n_steps, len(target_idx)]."""

    @abstractmethod
    def inject(self, t_step: int, buf: torch.Tensor) -> None:
        """Engine ``i_ext_fn`` convention: write into ``buf`` (float32, length N) and return None."""

    @abstractmethod
    def describe(self) -> dict:
        """Everything needed to reproduce the injection; stored verbatim in run.json ``meta.adapter``."""


class TableAdapter(SensoryAdapter):
    """Shared implementation: a precomputed device table + index_copy into the buffer."""

    def __init__(self, target_idx: np.ndarray, device: str | torch.device = "cuda"):
        target_idx = np.asarray(target_idx, dtype=np.int64)
        if target_idx.size == 0:
            raise LookupError("adapter has no target neurons")
        if len(np.unique(target_idx)) != len(target_idx):
            raise ValueError("target_idx contains duplicates")
        self.target_idx = target_idx
        self.device = torch.device(device)
        self._idx_t = torch.as_tensor(target_idx, device=self.device)
        self.table: torch.Tensor | None = None       # [n_steps, n_targets] float32 on device
        self.table_np: np.ndarray | None = None
        self.dt_ms: float | None = None
        self.n_steps: int | None = None

    def _set_table(self, table_np: np.ndarray, dt_ms: float) -> None:
        if table_np.ndim != 2 or table_np.shape[1] != len(self.target_idx):
            raise ValueError(f"table shape {table_np.shape} != [n_steps, {len(self.target_idx)}]")
        if not np.isfinite(table_np).all():
            raise ValueError("current table contains non-finite values")
        self.table_np = np.ascontiguousarray(table_np, dtype=np.float32)
        self.table = torch.as_tensor(self.table_np, device=self.device)
        self.dt_ms = float(dt_ms)
        self.n_steps = int(table_np.shape[0])

    def inject(self, t_step: int, buf: torch.Tensor) -> None:
        if self.table is None:
            raise RuntimeError("call prepare(dt_ms, n_steps) before inject")
        buf.zero_()
        if 0 <= t_step < self.n_steps:
            buf.index_copy_(0, self._idx_t, self.table[t_step])
        return None
