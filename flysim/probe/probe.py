"""M3: binned spike counts and per-set mean firing rates (docs/m3-brief.md "probe.py").

Input: the engine's ``(t_step int32, neuron_idx int32)`` arrays (sorted), the
step size ``dt_ms``, a bin width ``bin_ms`` and a ``ProbeSets``. Output: a
``RateTable`` with counts[R, n_bins] and rates[R, n_bins] in Hz per neuron
(count / (set size * bin_ms / 1000)).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .sets import ProbeSets


@dataclass
class RateTable:
    names: list[str]
    sizes: np.ndarray            # [R] neurons per set
    counts: np.ndarray           # [R, n_bins] int64 spikes
    rates: np.ndarray            # [R, n_bins] Hz per neuron
    bin_ms: float
    n_bins: int
    t_bin: np.ndarray            # per spike, bin index (int64)
    region: np.ndarray           # per spike, region index (int16)
    active_neurons: int          # distinct neurons that spiked at least once
    n_spikes: int

    def to_frame(self) -> pd.DataFrame:
        r = np.repeat(np.arange(len(self.names)), self.n_bins)
        b = np.tile(np.arange(self.n_bins), len(self.names))
        return pd.DataFrame({"region": np.asarray(self.names, dtype=object)[r], "bin": b,
                             "t_ms": b * self.bin_ms, "count": self.counts.ravel(), "rate_hz": self.rates.ravel()})


def bin_spikes(t_step: np.ndarray, idx: np.ndarray, dt_ms: float, bin_ms: float, n_bins: int,
               sets: ProbeSets) -> RateTable:
    t_step = np.asarray(t_step, dtype=np.int64)
    idx = np.asarray(idx, dtype=np.int64)
    t_bin = np.floor(t_step * dt_ms / bin_ms + 1e-9).astype(np.int64)
    if len(t_bin) and t_bin.max() >= n_bins:
        raise ValueError(f"spike at bin {t_bin.max()} >= n_bins {n_bins}")
    region = sets.region_of[idx]
    R = sets.n_regions
    counts = np.zeros((R, n_bins), dtype=np.int64)
    if len(t_bin):
        np.add.at(counts, (region, t_bin), 1)
    sizes = np.array([len(sets.members[k]) for k in sets.names], dtype=np.int64)
    rates = counts / (sizes[:, None] * bin_ms / 1000.0)
    return RateTable(sets.names, sizes, counts, rates, bin_ms, n_bins, t_bin, region.astype(np.int16),
                     int(len(np.unique(idx))), int(len(idx)))


def window_mean_rate(rt: RateTable, name: str, t0_ms: float, t1_ms: float) -> float:
    """Mean rate (Hz per neuron) of set ``name`` over bins with t0 <= t < t1."""
    r = rt.names.index(name)
    b0, b1 = int(round(t0_ms / rt.bin_ms)), int(round(t1_ms / rt.bin_ms))
    if b1 <= b0:
        raise ValueError("empty window")
    return float(rt.counts[r, b0:b1].sum() / (rt.sizes[r] * (b1 - b0) * rt.bin_ms / 1000.0))


def window_spikes(rt: RateTable, name: str, t0_ms: float, t1_ms: float) -> int:
    r = rt.names.index(name)
    b0, b1 = int(round(t0_ms / rt.bin_ms)), int(round(t1_ms / rt.bin_ms))
    return int(rt.counts[r, b0:b1].sum())


def save_parquet(out_dir: Path, rt: RateTable, t_step: np.ndarray, idx: np.ndarray, graph, dt_ms: float) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"t_step": np.asarray(t_step, dtype=np.int32), "t_ms": np.asarray(t_step) * dt_ms,
                  "neuron_idx": np.asarray(idx, dtype=np.int32), "bodyId": graph.body_ids[np.asarray(idx)],
                  "region": np.asarray(rt.names, dtype=object)[rt.region]}
                 ).to_parquet(out_dir / "spikes.parquet", index=False)
    rt.to_frame().to_parquet(out_dir / "rates.parquet", index=False)
