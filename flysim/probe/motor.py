"""M7b (interface only): neuron spikes -> one action of the shared action space.

The fly brain is connected in M7c, after the escape-circuit probe (M7a) says which neurons can
be read. This module fixes the *shape* of the read-out now, so the game harness and its tests
exist before the brain does:

    decoder = MotorDecoder({"JUMP": idx_a, "LEFT": idx_b, "RIGHT": idx_c}, window_ms=..., min_rate_hz=...)
    action_name = decoder.decode(spike_idx)        # spikes of one decision window

Rules (ASSUMPTIONS for M7c to confirm or replace):

* Each action that is not NOOP is driven by one group of neuron indices. Which neurons go in
  which group is **not** decided here — no cell-type name is written in this file (CLAUDE.md
  §3.1). M7c fills the groups from data queries.
* The rate of a group is ``spikes in the window / (group size * window)`` in Hz per neuron. The
  action is the group with the highest rate if that rate reaches ``min_rate_hz``; otherwise
  NOOP. An exact tie between groups is ambiguous and gives NOOP rather than a hidden priority.
* Groups must be non-empty and must not share neurons (an empty group raises, as every query in
  this project does on zero results).

This module returns action **names**. Mapping a name to a game's action index is the game
harness's job (``flysim.apps.games``), so ``probe`` does not import ``apps``.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np

ACTION_NAMES = ("NOOP", "JUMP", "LEFT", "RIGHT")      # the shared action space, by name
DEFAULT_ACTION = "NOOP"


class MotorDecoder:
    def __init__(self, groups: Mapping[str, np.ndarray], window_ms: float, min_rate_hz: float):
        if not groups:
            raise LookupError("MotorDecoder needs at least one action group")
        if window_ms <= 0 or min_rate_hz < 0:
            raise ValueError("window_ms must be > 0 and min_rate_hz >= 0")
        self.groups: dict[str, np.ndarray] = {}
        seen: set[int] = set()
        for name, idx in groups.items():
            if name not in ACTION_NAMES or name == DEFAULT_ACTION:
                raise ValueError(f"group {name!r} is not a decodable action; use one of "
                                 f"{[a for a in ACTION_NAMES if a != DEFAULT_ACTION]}")
            arr = np.unique(np.asarray(idx, dtype=np.int64))
            if arr.size == 0:
                raise LookupError(f"action group {name!r} has no neurons")
            if (arr < 0).any():
                raise ValueError(f"action group {name!r} has negative neuron indices")
            overlap = seen.intersection(arr.tolist())
            if overlap:
                raise ValueError(f"action group {name!r} shares neurons with another group: "
                                 f"{sorted(overlap)[:5]}")
            seen.update(arr.tolist())
            self.groups[name] = arr
        self.window_ms = float(window_ms)
        self.min_rate_hz = float(min_rate_hz)

    def rates(self, spike_idx: np.ndarray) -> dict[str, float]:
        """Hz per neuron of every group, for spikes falling in one decision window."""
        s = np.asarray(spike_idx, dtype=np.int64).ravel()
        window_s = self.window_ms / 1000.0
        return {name: float(np.isin(s, idx).sum()) / (len(idx) * window_s)
                for name, idx in self.groups.items()}

    def decode(self, spike_idx: np.ndarray) -> str:
        r = self.rates(spike_idx)
        best = max(r.values())
        if best < self.min_rate_hz or best == 0.0:
            return DEFAULT_ACTION
        winners = [name for name, v in r.items() if v == best]
        return winners[0] if len(winners) == 1 else DEFAULT_ACTION

    def describe(self) -> dict:
        return {"decoder": "MotorDecoder", "window_ms": self.window_ms,
                "min_rate_hz": self.min_rate_hz,
                "group_sizes": {k: int(len(v)) for k, v in self.groups.items()},
                "rule": "argmax group rate if >= min_rate_hz, exact tie -> NOOP (ASSUMPTION)"}
