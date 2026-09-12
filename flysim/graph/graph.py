"""M1: in-memory CSR graph of the retained MaleCNS connectome.

Arrays (all from data/cache/graph-v1.npz, built by flysim.graph.build):

    body_ids      int64  [N]     sorted; index i <-> bodyId body_ids[i]
    indptr        int64  [N+1]   forward CSR (pre -> post)
    indices       int32  [M]     post index per edge, sorted within each row
    weight        float32[M]     signed synaptic contact count (+ excit, - inhib)
    csc_indptr    int64  [N+1]   reverse (post -> pre), for upstream tracing
    csc_indices   int32  [M]
    csc_weight    float32[M]
    side          int8   [N]     SIDE_CODES (L=0, R=1, M=2), -1 = null
    nt            int8   [N]     NT_CODES of consensus_nt (8 = missing)
    nt_modulator  bool   [N]     histamine/dopamine/octopamine/serotonin
    nt_assumed    bool   [N]     sign was set by assumption (not ACh/GABA)
    has_soma_xyz  bool   [N]
    soma_xyz      float32[N,3]   NaN where missing

Weights are raw contact counts; no normalisation (engine parameter, M2).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import CACHE_ANN, CACHE_NPZ, NT_CODES, SIDE_CODES

SIDE_NAMES = {v: k for k, v in SIDE_CODES.items()} | {-1: None}
NT_NAMES = {v: k for k, v in NT_CODES.items()}


class NoPathError(LookupError):
    """No directed path within max_hops."""


@dataclass(frozen=True)
class GraphStats:
    n_neurons: int
    n_edges: int
    n_self_loops: int
    contacts_total: int
    contacts_excitatory: int
    contacts_inhibitory: int
    edges_excitatory: int
    edges_inhibitory: int
    cache_bytes: int


class Graph:
    def __init__(self, arrays: dict[str, np.ndarray], ann: pd.DataFrame):
        self.body_ids: np.ndarray = arrays["body_ids"]
        self.indptr: np.ndarray = arrays["indptr"]
        self.indices: np.ndarray = arrays["indices"]
        self.weight: np.ndarray = arrays["weight"]
        self.csc_indptr: np.ndarray = arrays["csc_indptr"]
        self.csc_indices: np.ndarray = arrays["csc_indices"]
        self.csc_weight: np.ndarray = arrays["csc_weight"]
        self.side: np.ndarray = arrays["side"]
        self.nt: np.ndarray = arrays["nt"]
        self.nt_modulator: np.ndarray = arrays["nt_modulator"]
        self.nt_assumed: np.ndarray = arrays["nt_assumed"]
        self.has_soma_xyz: np.ndarray = arrays["has_soma_xyz"]
        self.soma_xyz: np.ndarray = arrays["soma_xyz"]
        self.ann: pd.DataFrame = ann
        self._arrays = arrays
        if len(ann) != self.n:
            raise ValueError(f"annotation rows {len(ann)} != neurons {self.n}")
        if not np.array_equal(ann["bodyId"].to_numpy(), self.body_ids):
            raise ValueError("annotation cache not aligned with body_ids")
        # lazily built query API
        from .annotations import Annotations
        self.query = Annotations(self)

    # ---- loading -----------------------------------------------------------
    @classmethod
    def load(cls, npz: Path = CACHE_NPZ, ann_path: Path = CACHE_ANN,
             build_if_missing: bool = False) -> "Graph":
        if not (npz.exists() and ann_path.exists()):
            if not build_if_missing:
                raise FileNotFoundError(f"graph cache missing ({npz}); run python -m flysim.graph.build")
            from .build import build
            build()
        with np.load(npz) as z:
            arrays = {k: z[k] for k in z.files}
        ann = pd.read_parquet(ann_path)
        return cls(arrays, ann)

    # ---- basic properties --------------------------------------------------
    @property
    def n(self) -> int:
        return int(len(self.body_ids))

    @property
    def m(self) -> int:
        return int(len(self.indices))

    def idx(self, body_id) -> np.ndarray | int:
        """bodyId(s) -> index. Raises KeyError for unknown ids."""
        scalar = np.isscalar(body_id)
        b = np.atleast_1d(np.asarray(body_id, dtype=np.int64))
        pos = np.searchsorted(self.body_ids, b)
        pos_c = np.minimum(pos, self.n - 1)
        bad = self.body_ids[pos_c] != b
        if bad.any():
            raise KeyError(f"bodyId not in retained graph: {b[bad][:5].tolist()}")
        return int(pos_c[0]) if scalar else pos_c

    def body(self, idx) -> np.ndarray | int:
        """index -> bodyId."""
        out = self.body_ids[idx]
        return int(out) if np.isscalar(idx) else out

    def out_degree(self) -> np.ndarray:
        return np.diff(self.indptr)

    def in_degree(self) -> np.ndarray:
        return np.diff(self.csc_indptr)

    def successors(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        """(post indices, signed weights) of neuron i."""
        a, b = self.indptr[i], self.indptr[i + 1]
        return self.indices[a:b], self.weight[a:b]

    def predecessors(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        """(pre indices, signed weights) into neuron i."""
        a, b = self.csc_indptr[i], self.csc_indptr[i + 1]
        return self.csc_indices[a:b], self.csc_weight[a:b]

    def stats(self) -> GraphStats:
        w = self.weight.astype(np.int64)  # int64: float32 sums drift over 25M edges
        pos = w > 0
        rows = np.repeat(np.arange(self.n), self.out_degree())
        return GraphStats(
            n_neurons=self.n, n_edges=self.m,
            n_self_loops=int((rows == self.indices).sum()),
            contacts_total=int(np.abs(w).sum()),
            contacts_excitatory=int(w[pos].sum()),
            contacts_inhibitory=int(-w[~pos].sum()),
            edges_excitatory=int(pos.sum()), edges_inhibitory=int((~pos).sum()),
            cache_bytes=int(sum(a.nbytes for a in self._arrays.values())),
        )

    # ---- shortest path -----------------------------------------------------
    def shortest_path(self, src_body_id: int, dst_body_id: int, max_hops: int = 8) -> list[int]:
        """Unweighted BFS over directed edges. Returns bodyIds src..dst inclusive.

        Raises NoPathError (a LookupError) if dst is not reachable within max_hops.
        """
        s, t = self.idx(src_body_id), self.idx(dst_body_id)
        if s == t:
            return [int(src_body_id)]
        parent = np.full(self.n, -1, dtype=np.int64)
        parent[s] = s
        frontier = np.array([s], dtype=np.int64)
        for hop in range(1, max_hops + 1):
            starts, ends = self.indptr[frontier], self.indptr[frontier + 1]
            counts = ends - starts
            total = int(counts.sum())
            if total == 0:
                break
            # gather all neighbour slots of the frontier in one go
            rep_start = np.repeat(starts - np.cumsum(counts) + counts, counts)
            slots = rep_start + np.arange(total)
            nbr = self.indices[slots].astype(np.int64)
            src_of = np.repeat(frontier, counts)
            new = parent[nbr] == -1
            nbr, src_of = nbr[new], src_of[new]
            # first occurrence wins as parent (deterministic: frontier order)
            uniq, first = np.unique(nbr, return_index=True)
            parent[uniq] = src_of[first]
            if parent[t] != -1:
                path = deque([t])
                while path[0] != s:
                    path.appendleft(int(parent[path[0]]))
                return [int(self.body_ids[i]) for i in path]
            frontier = uniq
        raise NoPathError(f"no directed path {src_body_id} -> {dst_body_id} within {max_hops} hops")

    def __repr__(self) -> str:
        return f"Graph(neurons={self.n:,}, edges={self.m:,})"
