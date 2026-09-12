"""M1: annotation query API over the retained graph.

Every query returns neuron *indices* (np.ndarray[int64]) into Graph arrays and
raises LookupError when nothing matches. Empty results are never returned
silently (CLAUDE.md §3.1).

Column names used here are verified in docs/annotations-observed.md:
    type, instance, somaSide, superclass, class, fruDsx
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .constants import SIDE_CODES

if TYPE_CHECKING:
    from .graph import Graph


class Annotations:
    def __init__(self, g: "Graph"):
        self.g = g
        self.ann = g.ann
        self._types = self.ann["type"]
        self._type_values = pd.unique(self._types.dropna())

    # ---- helpers -------------------------------------------------------------
    def _side_mask(self, side: str) -> np.ndarray:
        """Side filter: Graph.side (somaSide first, instance suffix fallback)."""
        if side not in SIDE_CODES:
            raise ValueError(f"side must be one of {sorted(SIDE_CODES)}, got {side!r}")
        return self.g.side == SIDE_CODES[side]

    def _finish(self, mask: np.ndarray, what: str, side: str | None) -> np.ndarray:
        if side is not None:
            mask = mask & self._side_mask(side)
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            raise LookupError(f"no retained neuron matches {what}"
                              + (f" on side {side!r}" if side else ""))
        return idx

    def _exact(self, column: str, value: str, side: str | None) -> np.ndarray:
        if value is None or value == "":
            raise ValueError(f"{column} value must be a non-empty string")
        mask = (self.ann[column] == value).to_numpy()
        return self._finish(mask, f"{column} == {value!r}", side)

    # ---- queries -------------------------------------------------------------
    def by_type(self, name: str, side: str | None = None) -> np.ndarray:
        """Exact match on `type`. side in {'L','R','M'} filters by Graph.side."""
        return self._exact("type", name, side)

    def by_class(self, name: str, side: str | None = None) -> np.ndarray:
        return self._exact("class", name, side)

    def by_superclass(self, name: str, side: str | None = None) -> np.ndarray:
        return self._exact("superclass", name, side)

    def by_fru_dsx(self, value: str, side: str | None = None) -> np.ndarray:
        """Exact match on `fruDsx` (e.g. observed values fru_high, dsx_high, ...)."""
        return self._exact("fruDsx", value, side)

    def by_instance(self, name: str) -> np.ndarray:
        return self._exact("instance", name, None)

    def by_body_ids(self, body_ids) -> np.ndarray:
        return np.atleast_1d(self.g.idx(body_ids))

    def search_type(self, pattern: str, flags: int = re.IGNORECASE) -> list[tuple[str, int]]:
        """Regex search over distinct `type` values. Returns [(type, n_neurons)],
        most populous first. Raises LookupError on 0 hits. Exploration only —
        use by_type() with the exact name for anything that feeds a simulation."""
        rx = re.compile(pattern, flags)
        hits = [t for t in self._type_values if rx.search(t)]
        if not hits:
            raise LookupError(f"no retained cell type matches /{pattern}/")
        counts = self._types.value_counts()
        return sorted(((t, int(counts[t])) for t in hits), key=lambda x: (-x[1], x[0]))

    def describe(self, idx) -> pd.DataFrame:
        """Annotation rows for the given indices (for eyeballing query results)."""
        cols = ["bodyId", "type", "instance", "somaSide", "superclass", "class", "fruDsx"]
        return self.ann.iloc[np.atleast_1d(idx)][cols]
