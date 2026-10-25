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


class GiantFibreDecoder:
    """M7c's pre-registered read-out (docs/m7c-brief.md "같은 근육"), fixed before any fly run.

    Input: the giant-fibre spikes of the last ``window_ms`` as ``(t_ms, side)`` pairs with side
    ``"L"`` or ``"R"`` — the brain glue windows them, this class only maps them to an action:

        spikes from both sides  -> JUMP
        left side only          -> RIGHT   (move away from the side that fired)
        right side only         -> LEFT
        none                    -> NOOP

    Which neurons count as the left and right giant fibre is decided by the caller from the
    ``escape`` probe profile; no cell-type name is written here.
    """

    def __init__(self, window_ms: float = 50.0):
        if window_ms <= 0:
            raise ValueError("window_ms must be > 0")
        self.window_ms = float(window_ms)

    def decode(self, events) -> str:
        sides = set()
        for ev in events:
            side = ev[1] if isinstance(ev, (tuple, list)) else ev
            if side not in ("L", "R"):
                raise ValueError(f"side must be 'L' or 'R', got {side!r}")
            sides.add(side)
        if sides == {"L", "R"}:
            return "JUMP"
        if sides == {"L"}:
            return "RIGHT"
        if sides == {"R"}:
            return "LEFT"
        return DEFAULT_ACTION

    def describe(self) -> dict:
        return {"decoder": "GiantFibreDecoder", "window_ms": self.window_ms,
                "rule": "both GF -> JUMP; left GF only -> RIGHT; right GF only -> LEFT; none -> NOOP"}


class CountWindowDecoder:
    """M7c2's read-out family (docs/m7c2-brief.md 1.1), fixed before any calibration run.

    At the end of a tick (``now_ms``), count the giant-fibre spikes of the last ``window_ms`` per
    side — ``c_L``, ``c_R`` over ``now_ms - window_ms <= t < now_ms`` — and compare each with
    ``min_spikes``:

        c_L >= n and c_R >= n   -> JUMP
        c_L >= n only           -> RIGHT   (move away from the side that fired)
        c_R >= n only           -> LEFT
        neither                 -> NOOP

    The window is applied here, so ``events`` may be the whole ``(t_ms, side)`` history. With
    ``window_ms`` 50 and ``min_spikes`` 1 this is ``GiantFibreDecoder`` on the 50-ms window M7c's
    brain glue cut (tested). The action repeats on every tick the condition holds; there is no
    separate hold time. No cell-type name is written here: the caller labels the sides.
    """

    def __init__(self, window_ms: float, min_spikes: int):
        if window_ms <= 0:
            raise ValueError("window_ms must be > 0")
        if int(min_spikes) != min_spikes or min_spikes < 1:
            raise ValueError("min_spikes must be an integer >= 1")
        self.window_ms = float(window_ms)
        self.min_spikes = int(min_spikes)

    def counts(self, events, now_ms: float) -> tuple[int, int]:
        lo = float(now_ms) - self.window_ms
        c = {"L": 0, "R": 0}
        for ev in events:
            t, side = ev[0], ev[1]
            if side not in c:
                raise ValueError(f"side must be 'L' or 'R', got {side!r}")
            if lo <= t < now_ms:
                c[side] += 1
        return c["L"], c["R"]

    def decode(self, events, now_ms: float) -> str:
        c_l, c_r = self.counts(events, now_ms)
        left, right = c_l >= self.min_spikes, c_r >= self.min_spikes
        if left and right:
            return "JUMP"
        if left:
            return "RIGHT"
        if right:
            return "LEFT"
        return DEFAULT_ACTION

    def describe(self) -> dict:
        return {"decoder": "CountWindowDecoder", "window_ms": self.window_ms,
                "min_spikes": self.min_spikes,
                "rule": "count GF spikes per side in [now - window_ms, now); both >= n -> JUMP; "
                        "left only -> RIGHT; right only -> LEFT; neither -> NOOP"}
