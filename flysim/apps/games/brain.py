"""M7c: the fly brain as a game player — engine glue for ``BrainAgent`` (docs/m7c-brief.md).

Pre-registered (nothing here was chosen after a fly run):

* **Brain** = M7a's operating point P2, unchanged: conductance synapses, ``g`` 5.36539e-4,
  ``noise_sigma`` 29.13, ``adapt_g_b`` 8.0, dt 1 ms. **Noise seed = game seed.**
* **Time**: one game tick is 1/60 s. The engine advances 16 or 17 steps per tick, alternating
  deterministically so that tick ``k`` ends exactly at ``floor((k+1) * 1000 / 60)`` ms.
* **Same eyes**: the M7b ``LoomingDetector`` (inside ``BrainAgent``) gives ``loom_L, loom_R``.
  During the tick every left ``LC4`` and ``LPLC2`` cell gets the current ``k * loom_L`` and every
  right one ``k * loom_R``. No per-cell heterogeneity (ASSUMPTION, as in M7a). ``k`` is the only
  free parameter of the whole agent.
* **Same muscles**: at the end of each tick the decoder looks at the giant-fibre (``DNp01``)
  spikes of the last 50 ms (``flysim.probe.motor.GiantFibreDecoder``).

M7c2 (docs/m7c2-brief.md §2) adds two arguments; their defaults are M7c exactly:

* ``warmup_steps`` (default 0): after the engine reset, run that many steps with no input before
  the first game tick. The game clock (ticks, spike times, statistics) starts at 0 when the
  warm-up ends, and no warm-up spike reaches the decoder or the statistics.
* ``decoder_window_ms`` (default 50): how far back ``step`` returns giant-fibre spikes. A decoder
  that needs the tick-end time (``CountWindowDecoder``) is wrapped by ``TickDecoder``.

Cell sets come from the ``escape`` probe profile (``flysim.probe.sets``); no type name is
written in this file. The harness reads game state only for logging, never for the agent.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ...engine import EngineParams, LIFEngine
from ...probe.motor import GiantFibreDecoder

# M7a P2 (docs/m7a-report.md §4) — not searched again here
BRAIN_PARAMS = dict(synapse="conductance", g=5.36539e-4, noise_sigma=29.13, adapt_g_b=8.0, dt=1.0)
TICK_HZ = 60
DECODER_WINDOW_MS = 50.0
INPUT_SETS = {"L": ("LC4_L", "LPLC2_L"), "R": ("LC4_R", "LPLC2_R")}
READOUT_SETS = {"L": "DNp01_L", "R": "DNp01_R"}
STAT_WINDOW_MS = 200                # windows for the ignition check (network rate, active fraction)


def tick_end_ms(k: int) -> int:
    """End of game tick ``k`` (0-based) in whole ms: floor((k+1) * 1000 / 60)."""
    return ((k + 1) * 1000) // TICK_HZ


def steps_in_tick(k: int, dt_ms: float = 1.0) -> int:
    if dt_ms != 1.0:
        raise ValueError("the tick schedule is defined for dt = 1 ms")
    return tick_end_ms(k) - (tick_end_ms(k - 1) if k > 0 else 0)


@dataclass
class EscapeSets:
    """Index arrays for the escape circuit, taken from ``build_probe_sets(..., profile="escape")``."""
    inputs: dict[str, np.ndarray]           # "L"/"R" -> all LC4 + LPLC2 cells of that side
    gf: dict[str, int]                      # "L"/"R" -> the giant fibre of that side
    probe_sets: object                      # the ProbeSets itself (for run.json export)

    @classmethod
    def from_probe_sets(cls, ps) -> "EscapeSets":
        if getattr(ps, "profile", None) != "escape":
            raise ValueError("EscapeSets needs the 'escape' probe profile")
        inputs = {s: np.sort(np.concatenate([ps.members[n] for n in names])).astype(np.int64)
                  for s, names in INPUT_SETS.items()}
        gf = {}
        for s, name in READOUT_SETS.items():
            m = np.asarray(ps.members[name], dtype=np.int64)
            if len(m) != 1:
                raise ValueError(f"{name} must be exactly one neuron, got {len(m)}")
            gf[s] = int(m[0])
        return cls(inputs=inputs, gf=gf, probe_sets=ps)


class FlyBrain:
    """``brain_step(left, right)`` for ``BrainAgent``: one game tick of the simulated brain.

    Returns the giant-fibre spikes of the last ``DECODER_WINDOW_MS`` as an array of side labels
    with times (see ``GiantFibreDecoder``). Also keeps per-episode statistics for the harness.
    """

    def __init__(self, graph, sets: EscapeSets, k: float, device: str = "cuda",
                 use_cuda_graph: bool = True, keep_spikes: bool = False,
                 warmup_steps: int = 0, decoder_window_ms: float = DECODER_WINDOW_MS):
        if not k > 0:
            raise ValueError("k must be > 0")
        if int(warmup_steps) != warmup_steps or warmup_steps < 0:
            raise ValueError("warmup_steps must be an integer >= 0")
        if not decoder_window_ms > 0:
            raise ValueError("decoder_window_ms must be > 0")
        self.k = float(k)
        self.warmup_steps = int(warmup_steps)
        self.decoder_window_ms = float(decoder_window_ms)
        self.sets = sets
        self.params = EngineParams(**BRAIN_PARAMS)
        self.engine = LIFEngine(graph, self.params, device=device,
                                use_cuda_graph=use_cuda_graph and torch.cuda.is_available())
        dev = self.engine.device
        self._in = {s: torch.as_tensor(idx, device=dev) for s, idx in sets.inputs.items()}
        self._gf = np.array([sets.gf["L"], sets.gf["R"]], dtype=np.int64)
        self.n = self.engine.n
        self.keep_spikes = keep_spikes
        self.reset(0)

    # ------------------------------------------------------------------------------------
    def reset(self, seed: int) -> None:
        self.engine.reset(int(seed))            # noise seed = game seed
        if self.warmup_steps:                   # no input; its spikes are dropped (game clock starts after)
            self.engine.run(self.warmup_steps, None, record=True)
        self.seed = int(seed)
        self.tick = 0
        self.t_ms = 0
        self.gf_events: list[tuple[float, str]] = []      # (t_ms, "L"/"R")
        self.n_spikes = 0
        self._active = np.zeros(self.n, dtype=bool)
        self._win_active = np.zeros(self.n, dtype=bool)
        self._win_spikes = 0
        self._win_start = 0
        self.windows: list[tuple[float, float]] = []     # (network rate Hz, active fraction) per 200 ms
        self.input_spikes = 0
        self._input_mask = np.zeros(self.n, dtype=bool)
        self._input_mask[np.concatenate(list(self.sets.inputs.values()))] = True
        self._last = (None, None)
        self.spikes_t: list[np.ndarray] = []
        self.spikes_i: list[np.ndarray] = []

    def __call__(self, left: float, right: float) -> list[tuple[float, str]]:
        return self.step(left, right)

    def step(self, left: float, right: float) -> list[tuple[float, str]]:
        n_steps = steps_in_tick(self.tick)
        cur = (self.k * float(left), self.k * float(right))
        changed = cur != self._last
        self._last = cur

        def fn(_t, buf):
            if changed:
                buf.zero_()
                if cur[0]:
                    buf[self._in["L"]] = cur[0]
                if cur[1]:
                    buf[self._in["R"]] = cur[1]
            return None

        first = [True]

        def once(t, buf):                       # write the input only at the first step of the tick
            if first[0]:
                first[0] = False
                fn(t, buf)
            return None

        t_steps, idx = self.engine.run(n_steps, once, record=True)
        t_ms = (t_steps.astype(np.float64) - self.warmup_steps) * self.params.dt   # game clock
        idx = idx.astype(np.int64)
        self._account(t_ms, idx)
        self.tick += 1
        self.t_ms = tick_end_ms(self.tick - 1)
        lo = self.t_ms - self.decoder_window_ms
        return [(t, s) for (t, s) in self.gf_events if t >= lo]

    def _account(self, t_ms: np.ndarray, idx: np.ndarray) -> None:
        self.n_spikes += len(idx)
        if len(idx):
            self._active[idx] = True
            self.input_spikes += int(self._input_mask[idx].sum())
            gf = np.isin(idx, self._gf)
            for t, i in zip(t_ms[gf], idx[gf]):
                self.gf_events.append((float(t), "L" if i == self._gf[0] else "R"))
            if self.keep_spikes:
                self.spikes_t.append(t_ms.astype(np.int32))
                self.spikes_i.append(idx.astype(np.int32))
        # 200 ms statistics windows (a spike at t belongs to the window containing t)
        end = tick_end_ms(self.tick)
        cut = self._win_start + STAT_WINDOW_MS
        if end < cut:
            self._win_spikes += len(idx)
            if len(idx):
                self._win_active[idx] = True
            return
        inside = t_ms < cut
        self._win_spikes += int(inside.sum())
        if inside.any():
            self._win_active[idx[inside]] = True
        self._close_window()
        self._win_start = cut
        rest = ~inside
        self._win_spikes = int(rest.sum())
        if rest.any():
            self._win_active[idx[rest]] = True

    def _close_window(self) -> None:
        rate = self._win_spikes / (self.n * STAT_WINDOW_MS / 1000.0)
        self.windows.append((rate, float(self._win_active.mean())))
        self._win_active[:] = False

    # ------------------------------------------------------------------------------------
    def summary(self) -> dict:
        dur_s = max(self.t_ms, 1) / 1000.0
        w = np.array(self.windows) if self.windows else np.zeros((0, 2))
        return {"duration_ms": int(self.t_ms), "n_spikes": int(self.n_spikes),
                "network_rate_hz": self.n_spikes / (self.n * dur_s),
                "active_fraction": float(self._active.mean()),
                "input_cell_spikes": int(self.input_spikes),
                "gf_spikes": {"L": sum(1 for _, s in self.gf_events if s == "L"),
                              "R": sum(1 for _, s in self.gf_events if s == "R")},
                "window_rate_hz_max": float(w[:, 0].max()) if len(w) else 0.0,
                "window_active_max": float(w[:, 1].max()) if len(w) else 0.0,
                "windows": [[round(a, 4), round(b, 5)] for a, b in w.tolist()]}

    def spikes(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.spikes_t:
            return np.zeros(0, np.int32), np.zeros(0, np.int32)
        return np.concatenate(self.spikes_t), np.concatenate(self.spikes_i)

    def describe(self) -> dict:
        d = {"brain": "FlyBrain", "params": self.params.to_dict(), "k": self.k,
             "tick_ms": "floor((k+1)*1000/60) - floor(k*1000/60): 16 or 17 steps",
             "input": "k*loom_L on every left LC4+LPLC2 cell, k*loom_R on the right (no heterogeneity, ASSUMPTION)",
             "input_cells": {s: int(len(v)) for s, v in self.sets.inputs.items()},
             "readout": f"DNp01 L/R spikes in the last {self.decoder_window_ms:g} ms (escape profile)",
             "noise_seed": "game seed"}
        if self.warmup_steps:                   # M7c's describe() is left byte-identical at the defaults
            d["warmup"] = (f"{self.warmup_steps} steps with no input after the reset; "
                           "game clock 0 = end of warm-up; warm-up spikes not used")
        return d


def make_decoder() -> GiantFibreDecoder:
    return GiantFibreDecoder(window_ms=DECODER_WINDOW_MS)


class TickDecoder:
    """``BrainAgent`` calls ``decoder.decode(spikes)``; ``CountWindowDecoder`` also needs the end
    of the current tick. This passes the brain's game clock (``brain.t_ms`` = end of the tick
    ``step`` just ran) along. Nothing else changes."""

    def __init__(self, decoder, brain: FlyBrain):
        self.decoder = decoder
        self.brain = brain

    def decode(self, events) -> str:
        return self.decoder.decode(events, self.brain.t_ms)

    def describe(self) -> dict:
        return self.decoder.describe() | {"now_ms": "end of the current tick (game clock)"}
