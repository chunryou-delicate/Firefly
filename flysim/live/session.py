"""M5a: the live session — one resident engine, a real-time loop, and the control surface.

Threading: the simulation runs in its own thread and owns the engine. Everything from the
websocket side arrives through :meth:`LiveSession.submit` (a lock-guarded deque) and is
applied at a step boundary, which is what makes a run reproducible from its control log:
the log records the step index at which each message was applied, and
:func:`replay_control_log` re-applies them at the same steps.

Per step the work is one input fill (up to four small device kernels), the engine step and
one ``masked_fill_`` that stamps the current step into ``last_spike_step``. Per bin the frame
is computed from that array on the device and moved to the host with a **single** ``.cpu()``
call on one packed float32 tensor (neuron indices are < 2**24 so float32 carries them
exactly). Measurements are in ``docs/m5a-report.md``.
"""
from __future__ import annotations

import dataclasses
import json
import math
import queue
import subprocess
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from ..data.download import ROOT
from ..engine import EngineParams, LIFEngine
from ..probe.probe import RateTable
from . import protocol as P
from .stream import DEFAULT_FS, AudioSource, JOStream, SilenceSource, make_source

RUNS_LIVE = ROOT / "runs" / "live"
RING_SECONDS = 10.0
LAG_THRESHOLD_MS = 50.0          # behind by more than this -> frame.status = "lagging"
SLEEP_MIN_MS = 2.0               # only sleep once the lead is this big: one syscall per 2 ms of
                                 # simulated time instead of one per bin (see docs/m5a-report.md)
LAG_REBASE_MS = 1_000.0          # never try to catch up more than this (steps are not skipped)
PAUSED_FRAME_INTERVAL_S = 0.2    # heartbeat frame rate while paused
DEFAULT_A_IN = 640.0             # M3 rate-mode selection (docs/m3-report.md); a live starting point
DEFAULT_SPEED = 1.0
SAFETY = {"runaway_frac": 0.90, "ignited_frac": 0.01, "ignited_window_ms": 200.0,
          "silent_window_ms": 300.0, "active_window_ms": 100.0}


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return None


@dataclass
class LiveContext:
    """Everything a session needs that is not engine state: the graph, the probe sets, the
    JO target split and the viewer coordinates. Built from the real data by
    :func:`flysim.live.export_neurons.build_context`, or by hand in tests."""

    graph: Any
    sets: Any                                  # ProbeSets (names, members, region_of)
    jo_left: np.ndarray
    jo_right: np.ndarray
    coords: tuple | None = None                # (x, y, info) for snapshots
    neurons_url: str = "/neurons-v1.json"
    dataset: str = "MaleCNS v1.0"

    @property
    def n(self) -> int:
        return int(self.graph.n)


# ---------------------------------------------------------------------------------------
# hops (pure function over the forward CSR; safe to call from any thread)
# ---------------------------------------------------------------------------------------
def compute_hops(graph, src: int, k: int, min_contacts: int = 1,
                 cap: int = P.HOP_LAYER_CAP) -> tuple[list[list[int]], list[bool]]:
    """Downstream neurons within ``k`` hops over edges with ``|weight| >= min_contacts``.

    Returns ``(layers, truncated)``: layer *i* holds the neurons first reached at hop *i+1*,
    already visited neurons excluded. A layer wider than ``cap`` is cut to the ``cap``
    strongest by summed contacts and flagged in ``truncated``.
    """
    n = int(graph.n)
    if not (0 <= src < n):
        raise P.Reject(f"src {src} out of range 0..{n - 1}")
    indptr = np.asarray(graph.indptr)
    indices = np.asarray(graph.indices)
    weight = np.abs(np.asarray(graph.weight))
    visited = np.zeros(n, dtype=bool)
    visited[src] = True
    frontier = np.array([src], dtype=np.int64)
    layers: list[list[int]] = []
    truncated: list[bool] = []
    for _ in range(int(k)):
        if frontier.size == 0:
            layers.append([])
            truncated.append(False)
            continue
        starts = indptr[frontier].astype(np.int64)
        ends = indptr[frontier + 1].astype(np.int64)
        counts = ends - starts
        total = int(counts.sum())
        if total == 0:
            layers.append([])
            truncated.append(False)
            frontier = np.array([], dtype=np.int64)
            continue
        slots = np.repeat(starts - np.cumsum(counts) + counts, counts) + np.arange(total)
        nbr = indices[slots].astype(np.int64)
        w = weight[slots]
        keep = (w >= min_contacts) & ~visited[nbr]
        nbr, w = nbr[keep], w[keep]
        if nbr.size == 0:
            layers.append([])
            truncated.append(False)
            frontier = np.array([], dtype=np.int64)
            continue
        agg = np.bincount(nbr, weights=w.astype(np.float64), minlength=n)
        uniq = np.unique(nbr)
        cut = False
        if uniq.size > cap:
            order = np.argsort(-agg[uniq], kind="stable")[:cap]
            uniq = np.sort(uniq[order])
            cut = True
        visited[uniq] = True
        layers.append([int(i) for i in uniq])
        truncated.append(cut)
        frontier = uniq
    return layers, truncated


# ---------------------------------------------------------------------------------------
# session
# ---------------------------------------------------------------------------------------
class LiveSession:
    def __init__(self, ctx: LiveContext, *, params: EngineParams | None = None,
                 device: str | torch.device = "cuda", backend: str | None = None,
                 use_cuda_graph: bool | None = None, bin_ms: float = 1.0, seed: int = 0, a_in: float = DEFAULT_A_IN,
                 mode: str = "rate", speed: float = DEFAULT_SPEED,
                 session_dir: Path | None = None, emit: Callable[[dict, Any], None] | None = None,
                 log_controls: bool = True, session_id: str | None = None):
        self.ctx = ctx
        self.n = ctx.n
        self.bin_ms = float(bin_ms)
        self.seed = int(seed)
        self.a_in = float(a_in)
        self.mode = mode
        self.speed = float(speed)
        self.frame_every = 1
        self.params_version = 0
        self.session_id = session_id or time.strftime("live-%Y%m%d-%H%M%S")
        self._emit = emit or (lambda msg, to=None: None)

        p = params or EngineParams(synapse="conductance", dt=1.0)
        if use_cuda_graph is None:
            # On the Triton/CUDA path the capture pays for itself: dt = 0.1 ms goes from 0.76x
            # to 1.49x real time (docs/m5a-report.md). The engine ignores it on the torch backend.
            use_cuda_graph = torch.cuda.is_available() and torch.device(device).type == "cuda"
        self.engine = LIFEngine(ctx.graph, p, device=device, backend=backend,
                                use_cuda_graph=use_cuda_graph)
        self.device = self.engine.device
        self.engine.reset(self.seed)

        # probe sets on the device
        self.region_names: list[str] = list(ctx.sets.names)
        self.n_regions = len(self.region_names)
        self.region_of = torch.as_tensor(np.asarray(ctx.sets.region_of, dtype=np.int64), device=self.device)
        self.set_sizes = np.array([len(ctx.sets.members[k]) for k in self.region_names], dtype=np.int64)
        self._sizes_f = np.maximum(self.set_sizes, 1).astype(np.float64)

        # JO targets
        self._jo_l = torch.as_tensor(np.asarray(ctx.jo_left, dtype=np.int64), device=self.device)
        self._jo_r = torch.as_tensor(np.asarray(ctx.jo_right, dtype=np.int64), device=self.device)

        # state buffers
        self._in_buf = torch.zeros(self.n, dtype=torch.float32, device=self.device)
        self._inject_buf = torch.zeros(self.n, dtype=torch.float32, device=self.device)
        self._counts = torch.zeros(self.n_regions, dtype=torch.float32, device=self.device)
        self._last = torch.full((self.n,), -(1 << 30), dtype=torch.int32, device=self.device)
        self._sample_gen = torch.Generator(device=self.device)
        self._sample_gen.manual_seed(self.seed)
        self._sample_k = min(P.SPIKE_CAP_PER_BIN, self.n)   # topk cannot ask for more than n

        # input
        self.stream = JOStream(mode=mode, dt_ms=p.dt)
        self.ild_db = 0.0
        self._audio: AudioSource | None = None
        self._injects: list[dict] = []
        self._drive_ref = self._full_scale_drive(mode, p.dt)

        # bookkeeping
        self.step_index = 0
        self.sim_ms = 0.0
        self.status = "running"
        self._paused = False
        self._step_budget = 0
        self._last_input_step = -(1 << 30)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._cmd: deque = deque()
        self._cmd_lock = threading.Lock()
        self._ring: deque = deque(maxlen=max(1, int(RING_SECONDS * 1000.0 / self.bin_ms)))
        self._ring_lock = threading.Lock()
        self._warn: dict[str, bool] = {c: False for c in P.WARNING_CODES}
        self._bundle: list[dict] = []
        self.metrics = {"bins": 0, "bin_us_sum": 0.0, "bin_us_max": 0.0, "steps": 0,
                        "step_us_sum": 0.0, "lag_ms_max": 0.0, "lagging_bins": 0}

        # control log
        self.session_dir = Path(session_dir) if session_dir else (RUNS_LIVE / self.session_id)
        self.log_path = self.session_dir / "control-log.jsonl"
        self._log_fh = None
        self._log_lock = threading.Lock()     # the sim thread and the websocket thread both log
        if log_controls:
            self.session_dir.mkdir(parents=True, exist_ok=True)
            self._log_fh = self.log_path.open("a", encoding="utf-8")
            self._write_log({"kind": "session_start", "session_id": self.session_id,
                             "engine_params": self.engine.params.to_dict(),
                             "live_params": self.params_public(), "bin_ms": self.bin_ms,
                             "seed": self.seed, "n_neurons": self.n,
                             "regions": self.region_names, "git_commit": git_commit(),
                             "backend": self.engine.backend, "device": str(self.device)})

    # ---- derived quantities -------------------------------------------------------------
    @property
    def dt_ms(self) -> float:
        return float(self.engine.params.dt)

    @property
    def steps_per_bin(self) -> int:
        return max(1, int(round(self.bin_ms / self.dt_ms)))

    def _steps_for_ms(self, ms: float) -> int:
        return max(1, int(round(ms / self.dt_ms)))

    def params_public(self) -> dict:
        p = self.engine.params
        out = {"synapse": p.synapse, "g": float(p.g), "a_in": float(self.a_in),
               "noise_sigma": float(p.noise_sigma), "dt_ms": float(p.dt),
               "speed": float(self.speed), "mode": self.stream.mode}
        out["adapt_b"] = float(getattr(p, "adapt_b", 0.0))
        out["adapt_tau_w"] = float(getattr(p, "adapt_tau_w", 0.0))
        return out

    def _full_scale_drive(self, mode: str, dt_ms: float) -> float:
        """Drive produced by a full-scale 200 Hz tone — the reference for ``input_level``
        (0..1). Computed once per (mode, dt) so the scale never shifts under the client."""
        probe = JOStream(mode=mode, dt_ms=dt_ms,
                         source=make_source("tone", fs=self.stream.fs if hasattr(self, "stream") else 10_000.0,
                                            carrier_hz=200.0))
        d = probe.drive_series(max(4, int(round(100.0 / dt_ms))))
        return float(max(d.max(), 1e-9))

    # ---- control ------------------------------------------------------------------------
    def prepare_audio(self, sample_rate: float) -> AudioSource:
        """Create (or keep) the audio buffer **immediately**, before the queued ``audio_start``
        reaches a step boundary: a client that streams PCM right after ``audio_start`` would
        otherwise lose the first chunks to the queue latency. Attaching it to the signal path
        still happens in ``apply_control``, at a step boundary."""
        if self._audio is None or self._audio.fs != float(sample_rate):
            self._audio = AudioSource(sample_rate)
        return self._audio

    def submit(self, msg: dict, client: Any = None) -> None:
        """Queue a validated control message; applied at the next step boundary."""
        with self._cmd_lock:
            self._cmd.append((msg, client))

    def _drain_commands(self) -> None:
        if not self._cmd:
            return
        with self._cmd_lock:
            pending = list(self._cmd)
            self._cmd.clear()
        for msg, client in pending:
            try:
                ok, text = self.apply_control(msg)
            except P.Reject as e:
                ok, text = False, e.msg
            except Exception as e:                                   # pragma: no cover - defensive
                ok, text = False, f"{type(e).__name__}: {e}"
            self._write_log({"kind": "control", "msg": msg, "ok": ok, "result": text})
            self._emit(P.ack_msg(msg.get("req_id"), ok, text, self.params_version), client)

    def apply_control(self, msg: dict) -> tuple[bool, str]:
        """Apply one validated control message. Returns ``(ok, human-readable result)``."""
        kind = msg["type"]
        if kind == "set_params":
            return self._apply_params(msg["params"])
        if kind == "stimulus":
            return self._apply_stimulus(msg)
        if kind == "audio_start":
            src = self.prepare_audio(msg["sample_rate"])
            self.stream.set_source(src)
            self.stream.reset_filters()
            return True, (f"audio stream started at {msg['sample_rate']} Hz"
                          + (f" ({src.queued} samples already buffered)" if src.queued else ""))
        if kind == "audio_stop":
            self._audio = None
            self.stream.set_source(SilenceSource())
            return True, "audio stream stopped; input is silence"
        if kind == "inject":
            return self._apply_inject(msg)
        if kind == "pause":
            self._paused = True
            self._step_budget = 0
            self.status = "paused"
            self.flush_bundle()
            return True, "paused"
        if kind == "resume":
            self._paused = False
            self._rebase_clock()
            return True, "running"
        if kind == "step":
            self._paused = True
            self._step_budget += int(msg["n"])
            return True, f"stepping {msg['n']} step(s) while paused"
        if kind == "reset":
            self.seed = int(msg["seed"])
            self._do_reset()
            return True, f"reset to rest with seed {self.seed} (parameters kept)"
        if kind == "set_frame_every":
            self.frame_every = int(msg["n"])
            self._bundle.clear()
            return True, f"frame_every = {self.frame_every}"
        if kind in ("get_hops", "snapshot"):
            # served off the simulation thread (see server.py); nothing to apply here
            return True, f"{kind} accepted"
        return False, f"unhandled message type {kind!r}"

    def _apply_params(self, changes: dict) -> tuple[bool, str]:
        engine_changes: dict = {}
        notes: list[str] = []
        needs_reset = False
        for key, val in changes.items():
            if key == "speed":
                self.speed = float(val)
                self._rebase_clock()
                notes.append(f"speed={val:g}")
            elif key == "a_in":
                self.a_in = float(val)
                notes.append(f"a_in={val:g}")
            elif key == "mode":
                self.stream.set_mode(val)
                self._drive_ref = self._full_scale_drive(val, self.dt_ms)
                notes.append(f"mode={val}")
            elif key == "dt_ms":
                if float(val) != self.dt_ms:
                    engine_changes["dt"] = float(val)
                    needs_reset = True
                notes.append(f"dt_ms={val:g}")
            elif key == "synapse":
                if val != self.engine.params.synapse:
                    engine_changes["synapse"] = val
                    needs_reset = True
                notes.append(f"synapse={val}")
            elif key in ("g", "noise_sigma"):
                engine_changes[key] = float(val)
                notes.append(f"{key}={val:g}")
            elif key in ("adapt_b", "adapt_tau_w"):
                if not P.ENGINE_HAS_ADAPT:
                    if float(val) != 0.0:                       # protocol.validate_params guards this
                        raise P.Reject(f"{key} is not supported by this engine build")
                    notes.append(f"{key}=0 (no-op: engine has no adaptation)")
                else:
                    engine_changes[key] = float(val)
                    notes.append(f"{key}={val:g}")
        if engine_changes:
            self.engine.params = dataclasses.replace(self.engine.params, **engine_changes)
        if needs_reset:
            self.stream.set_dt(self.dt_ms)
            self._drive_ref = self._full_scale_drive(self.stream.mode, self.dt_ms)
            self._do_reset()
            notes.append("engine reset (dt_ms/synapse changed)")
        self._bump_params_version()
        return True, "; ".join(notes) if notes else "no change"

    def _bump_params_version(self) -> None:
        """Every parameter change is broadcast to all clients (contract v1.1): an ack goes only
        to the requester, so this is how the other cockpits learn what changed."""
        self.params_version += 1
        self._emit(P.params_msg(self.params_public(), self.params_version), None)

    def _apply_stimulus(self, msg: dict) -> tuple[bool, str]:
        kind = msg["kind"]
        self.ild_db = float(msg.get("ild_db", 0.0))
        if kind == "audio":
            src = self.prepare_audio(self._audio.fs if self._audio else self.stream.fs)
            self.stream.set_source(src)
            return True, "input switched to the audio stream (send audio_start + binary PCM)"
        self._audio = None
        # generated stimuli always use the M3 sample rate, so a live click train is the same
        # waveform as a scripted one even if an audio stream just moved the stream to 44.1 kHz
        src = make_source(kind, fs=DEFAULT_FS, ipi_ms=msg.get("ipi_ms", 35.0),
                          pulse_ms=msg.get("pulse_ms", 10.0), carrier_hz=msg.get("carrier_hz", 200.0),
                          duration_ms=msg.get("duration_ms", 0.0))
        self.stream.set_source(src)
        self.stream.reset_filters()
        return True, f"stimulus = {json.dumps(src.describe(), separators=(',', ':'))}"

    def _apply_inject(self, msg: dict) -> tuple[bool, str]:
        targets = msg["targets"]
        if "set" in targets:
            name = targets["set"]
            if name not in self.ctx.sets.members:
                raise P.Reject(f"unknown set {name!r}; known: {self.region_names}")
            idx = np.asarray(self.ctx.sets.members[name], dtype=np.int64)
            label = f"set {name} ({len(idx)} neurons)"
        else:
            idx = np.asarray(targets["idx"], dtype=np.int64)
            if idx.max(initial=-1) >= self.n:
                raise P.Reject(f"neuron index out of range 0..{self.n - 1}")
            label = f"{len(idx)} neuron(s)"
        steps = int(round(msg["duration_ms"] / self.dt_ms))
        if steps <= 0:
            raise P.Reject(f"duration_ms {msg['duration_ms']:g} is shorter than one step ({self.dt_ms} ms)")
        self._injects.append({"idx": torch.as_tensor(idx, device=self.device),
                              "amp": float(msg["amp"]), "until": self.step_index + steps})
        self._rebuild_inject()
        return True, f"injecting {msg['amp']:g} into {label} for {msg['duration_ms']:g} ms"

    def _rebuild_inject(self) -> None:
        self._inject_buf.zero_()
        for inj in self._injects:
            self._inject_buf.index_add_(0, inj["idx"],
                                        torch.full((inj["idx"].numel(),), inj["amp"],
                                                   dtype=torch.float32, device=self.device))

    def _do_reset(self) -> None:
        self.flush_bundle()          # the bins before the reset belong to the old time axis
        self.engine.reset(self.seed)
        self._sample_gen.manual_seed(self.seed)
        self.step_index = 0
        self.sim_ms = 0.0
        self._last.fill_(-(1 << 30))
        self._injects.clear()
        self._inject_buf.zero_()
        self._last_input_step = -(1 << 30)
        self.stream.reset()
        self._bundle.clear()
        with self._ring_lock:
            self._ring.clear()
        for code in list(self._warn):
            self._warn[code] = False
        self._rebase_clock()

    def _write_log(self, entry: dict) -> None:
        if self._log_fh is None:
            return
        entry = {"wall": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "t_ms": round(self.sim_ms, 4),
                 "step": self.step_index, "params_version": self.params_version, **entry}
        with self._log_lock:
            if self._log_fh is None:
                return
            self._log_fh.write(json.dumps(entry, separators=(",", ":"), default=str) + "\n")
            self._log_fh.flush()

    def log_external(self, entry: dict) -> None:
        """Log a control message that was served off the simulation thread (get_hops, snapshot)."""
        self._write_log(entry)

    # ---- one step ------------------------------------------------------------------------
    def _fill_input(self) -> float:
        """Write this step's external current into ``_in_buf``; returns the drive (0..)."""
        drive = self.stream.next_drive()
        self._in_buf.zero_()
        if drive > 0.0 and self.a_in > 0.0:
            gain_r = 10.0 ** (self.ild_db / 20.0)
            val = self.a_in * drive
            if self._jo_l.numel():
                self._in_buf.index_fill_(0, self._jo_l, val)
            if self._jo_r.numel():
                self._in_buf.index_fill_(0, self._jo_r, val * gain_r)
        if self._injects:
            if self.step_index >= min(i["until"] for i in self._injects):
                self._injects = [i for i in self._injects if self.step_index < i["until"]]
                self._rebuild_inject()
            if self._injects:
                self._in_buf.add_(self._inject_buf)
        if drive > 0.0 or self._injects:
            self._last_input_step = self.step_index
        return drive

    def _step_once(self) -> float:
        drive = self._fill_input()
        self.engine.step(self._in_buf, record=False)
        self._last.masked_fill_(self.engine.spikes, self.step_index)
        self.step_index += 1
        self.sim_ms += self.dt_ms
        return drive

    # ---- one bin -------------------------------------------------------------------------
    def _frame_payload(self, drive_max: float) -> dict:
        """Frame quantities for the bin that just ended.

        Two device synchronisations: ``nonzero`` (its output size has to reach the host) and
        the single ``.cpu()`` that moves per-region counts, the three window sums and the
        sampled indices in one packed float32 tensor — neuron indices are < 2**24, so float32
        carries them exactly. Everything after ``nonzero`` works on the spiking neurons only,
        which is why a silent bin costs about as little as an ignited one costs much.
        """
        t = self.step_index - 1
        age = t - self._last
        m_bin = age < self.steps_per_bin
        fired = torch.nonzero(m_bin).flatten()
        n_total = int(fired.numel())
        if n_total:
            counts = torch.bincount(self.region_of[fired], minlength=self.n_regions).float()
            if n_total > self._sample_k:
                r = torch.rand(n_total, device=self.device, generator=self._sample_gen)
                sample = fired[torch.topk(r, self._sample_k).indices]
            else:
                sample = fired
        else:
            counts = self._counts.zero_()
            sample = fired
        a100 = (age < self._steps_for_ms(SAFETY["active_window_ms"])).sum()
        a200 = (age < self._steps_for_ms(SAFETY["ignited_window_ms"])).sum()
        a300 = (age < self._steps_for_ms(SAFETY["silent_window_ms"])).sum()
        packed = torch.cat([counts, torch.stack([a100, a200, a300]).float(),
                            sample.float()]).cpu().numpy()
        r0 = self.n_regions
        return {"t_ms": self.sim_ms, "step": self.step_index,
                "counts": packed[:r0].astype(np.int64),
                "idx": np.sort(packed[r0 + 3:].astype(np.int32)), "n_total": n_total,
                "sample_ratio": (len(packed) - r0 - 3) / n_total if n_total else 1.0,
                "active_frac_100ms": float(packed[r0]) / self.n,
                "active_frac_200ms": float(packed[r0 + 1]) / self.n,
                "active_frac_300ms": float(packed[r0 + 2]) / self.n,
                "input_level": min(1.0, drive_max / self._drive_ref)}

    def _rates(self, counts: np.ndarray) -> np.ndarray:
        return counts / (self._sizes_f * self.bin_ms / 1000.0)

    def _run_bin(self) -> dict:
        t0 = time.perf_counter()
        drive_max = 0.0
        for _ in range(self.steps_per_bin):
            self._drain_commands()
            drive_max = max(drive_max, self._step_once())
        payload = self._frame_payload(drive_max)
        self.metrics["bins"] += 1
        self.metrics["steps"] += self.steps_per_bin
        us = (time.perf_counter() - t0) * 1e6
        self.metrics["bin_us_sum"] += us
        self.metrics["bin_us_max"] = max(self.metrics["bin_us_max"], us)
        with self._ring_lock:
            self._ring.append(payload)
        self._check_warnings(payload)
        return payload

    def _check_warnings(self, f: dict) -> None:
        no_input = (f["input_level"] <= 0.0 and not self._injects
                    and self.engine.params.noise_sigma == 0.0)
        recent_input = (self.step_index - self._last_input_step) < self._steps_for_ms(SAFETY["silent_window_ms"])
        state = {
            "runaway": f["active_frac_100ms"] >= SAFETY["runaway_frac"],
            "ignited": no_input and f["active_frac_200ms"] >= SAFETY["ignited_frac"],
            "all_silent": recent_input and f["active_frac_300ms"] == 0.0,
            "recorder_overflow": self._warn["recorder_overflow"],
        }
        texts = {
            "runaway": f"{f['active_frac_100ms']:.1%} of neurons spiked in the last 100 ms "
                       f"(>= {SAFETY['runaway_frac']:.0%}): activity is running away",
            "ignited": f"no stimulus, no injection and no noise, yet {f['active_frac_200ms']:.2%} of neurons "
                       f"spiked in the last {SAFETY['ignited_window_ms']:.0f} ms: the network is ignited",
            "all_silent": f"input has been present within the last {SAFETY['silent_window_ms']:.0f} ms "
                          "but no neuron spiked in that window",
            "recorder_overflow": "spike recorder overflowed",
        }
        for code, now in state.items():
            if now and not self._warn[code]:
                self._warn[code] = True
                self._emit(P.warning_msg(code, texts[code]), None)
            elif not now and self._warn[code]:
                self._warn[code] = False
                self._emit(P.warning_msg(code + "_cleared", f"{code} cleared"), None)

    # ---- frames --------------------------------------------------------------------------
    def _bundle_frame(self, payload: dict) -> dict | None:
        self._bundle.append(payload)
        if len(self._bundle) < self.frame_every:
            return None
        return self._make_frame()

    def flush_bundle(self) -> None:
        """Emit a partial bundle (fewer than ``frame_every`` bins). Called at the reset and pause
        boundaries so ``frame.n_bins`` always states how many bins the message really covers."""
        if self._bundle:
            self._emit(self._make_frame(), None)

    def _make_frame(self) -> dict:
        group = self._bundle
        self._bundle = []
        counts = np.sum([g["counts"] for g in group], axis=0) / len(group)
        idx = np.unique(np.concatenate([g["idx"] for g in group])) if group else np.empty(0, np.int32)
        if len(idx) > self._sample_k:
            sel = np.random.default_rng(self.seed + group[-1]["step"]).choice(
                len(idx), self._sample_k, replace=False)
            idx = np.sort(idx[np.sort(sel)])
        n_total = int(sum(g["n_total"] for g in group))
        last = group[-1]
        return P.frame_msg(
            t_ms=last["t_ms"], step=last["step"],
            rates=[round(float(v), 4) for v in self._rates(counts)],
            idx=[int(i) for i in idx], n_total=n_total,
            sample_ratio=(len(idx) / n_total) if n_total else 1.0,
            active_frac_100ms=last["active_frac_100ms"],
            input_level=max(g["input_level"] for g in group),
            status=self.status, speed=self.speed, params_version=self.params_version,
            n_bins=len(group))

    def paused_frame(self) -> dict:
        return P.frame_msg(t_ms=self.sim_ms, step=self.step_index,
                           rates=[0.0] * self.n_regions, idx=[], n_total=0, sample_ratio=1.0,
                           active_frac_100ms=0.0, input_level=0.0, status="paused",
                           speed=self.speed, params_version=self.params_version, n_bins=0)

    # ---- real-time loop --------------------------------------------------------------------
    def _rebase_clock(self) -> None:
        self._wall0 = time.perf_counter()
        self._sim0 = self.sim_ms

    def start(self) -> None:
        if self._thread is not None:
            return
        self._rebase_clock()
        self._thread = threading.Thread(target=self._loop, name="flysim-live-sim", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)
            self._thread = None
        if self._log_fh is not None:
            self._write_log({"kind": "session_stop", "metrics": self.pacing_metrics()})
            with self._log_lock:
                self._log_fh.close()
                self._log_fh = None

    def _loop(self) -> None:
        next_paused_frame = 0.0
        while not self._stop.is_set():
            self._drain_commands()
            if self._paused and self._step_budget <= 0:
                now = time.perf_counter()
                if now >= next_paused_frame:
                    next_paused_frame = now + PAUSED_FRAME_INTERVAL_S
                    self.status = "paused"
                    self._emit(self.paused_frame(), None)
                time.sleep(0.002)
                self._rebase_clock()
                continue
            stepping = self._paused and self._step_budget > 0
            payload = self._run_bin()
            if stepping:
                self._step_budget = max(0, self._step_budget - self.steps_per_bin)
                self.status = "paused"
            else:
                target = self._wall0 + (self.sim_ms - self._sim0) / 1000.0 / max(self.speed, 1e-9)
                lag_ms = (time.perf_counter() - target) * 1000.0
                self.metrics["lag_ms_max"] = max(self.metrics["lag_ms_max"], lag_ms)
                if lag_ms < -SLEEP_MIN_MS:
                    time.sleep(-lag_ms / 1000.0)
                    self.status = "running"
                elif lag_ms <= 0:
                    self.status = "running"          # slightly ahead: keep going, sleep later
                elif lag_ms > LAG_THRESHOLD_MS:
                    self.status = "lagging"
                    self.metrics["lagging_bins"] += 1
                    if lag_ms > LAG_REBASE_MS:
                        self._rebase_clock()       # stop chasing; no step is ever skipped
                else:
                    self.status = "running"
            frame = self._bundle_frame(payload)
            if frame is not None:
                self._emit(frame, None)

    def pacing_metrics(self) -> dict:
        m = self.metrics
        bins = max(m["bins"], 1)
        return {"bins": m["bins"], "steps": m["steps"],
                "mean_bin_us": round(m["bin_us_sum"] / bins, 1),
                "max_bin_us": round(m["bin_us_max"], 1),
                "max_lag_ms": round(m["lag_ms_max"], 2),
                "lagging_bins": m["lagging_bins"],
                "dt_ms": self.dt_ms, "bin_ms": self.bin_ms, "steps_per_bin": self.steps_per_bin,
                "backend": self.engine.backend, "device": str(self.device)}

    # ---- hello / snapshot -------------------------------------------------------------------
    def hello(self) -> dict:
        sets = {}
        sizes = {}
        for name in self.region_names:
            members = np.asarray(self.ctx.sets.members[name])
            sizes[name] = int(len(members))
            if name != "rest" and len(members) <= P.SET_INDEX_CAP:
                sets[name] = [int(i) for i in members]
        return P.hello_msg(
            n_neurons=self.n, regions=self.region_names, sets=sets, set_sizes=sizes,
            params=self.params_public(), dt_ms=self.dt_ms, bin_ms=self.bin_ms,
            engine={"synapse": self.engine.params.synapse, "git_commit": git_commit(),
                    "backend": self.engine.backend, "device": str(self.device),
                    "engine_params": self.engine.params.to_dict(),
                    "adaptation_available": P.ENGINE_HAS_ADAPT,
                    "drive_full_scale": self._drive_ref},
            neurons_url=self.ctx.neurons_url, session_id=self.session_id,
            params_version=self.params_version)

    def snapshot(self, run_id: str, seconds: float, out_dir: Path | None = None) -> dict:
        """Write the last ``seconds`` of the ring buffer as a viewer-contract run.json."""
        from ..probe.export import write_run_json

        n_bins = max(1, int(round(seconds * 1000.0 / self.bin_ms)))
        with self._ring_lock:
            items = list(self._ring)[-n_bins:]
        if not items:
            raise P.Reject("nothing recorded yet")
        if self.ctx.coords is None:
            raise P.Reject("this session has no neuron coordinates (snapshots need the real graph)")
        nb = len(items)
        counts = np.zeros((self.n_regions, nb), dtype=np.int64)
        env = np.zeros(nb, dtype=np.float64)
        t_bins, idxs = [], []
        n_spikes_total = 0
        for j, f in enumerate(items):
            counts[:, j] = f["counts"]
            env[j] = f["input_level"]
            idxs.append(f["idx"])
            t_bins.append(np.full(len(f["idx"]), j, dtype=np.int64))
            n_spikes_total += f["n_total"]
        idx_all = np.concatenate(idxs) if idxs else np.empty(0, np.int32)
        t_all = np.concatenate(t_bins) if t_bins else np.empty(0, np.int64)
        region_of = np.asarray(self.ctx.sets.region_of)
        rt = RateTable(names=self.region_names, sizes=self.set_sizes, counts=counts,
                       rates=counts / (self._sizes_f[:, None] * self.bin_ms / 1000.0),
                       bin_ms=self.bin_ms, n_bins=nb, t_bin=t_all,
                       region=region_of[idx_all].astype(np.int16) if len(idx_all) else np.empty(0, np.int16),
                       active_neurons=int(len(np.unique(idx_all))), n_spikes=int(len(idx_all)))
        out_dir = Path(out_dir) if out_dir else (RUNS_LIVE / run_id)
        path = out_dir / "run.json"
        ratio = (len(idx_all) / n_spikes_total) if n_spikes_total else 1.0
        write_run_json(
            path, run_id=run_id, sensory_mode=self.stream.mode, dt_ms=self.dt_ms, rt=rt,
            sets=self.ctx.sets, coords=self.ctx.coords,
            input_label=f"live {self.stream.source.kind}, a_in={self.a_in:g}",
            envelope=env, spikes=(t_all, idx_all),
            engine_params=self.engine.params.to_dict(), adapter=self.stream.describe(),
            stimulus=self.stream.source.describe(),
            extra_meta={
                "live_control_log": str(self.log_path),
                "live_session_id": self.session_id,
                "live_params": self.params_public(),
                "live_seconds": nb * self.bin_ms / 1000.0,
                "live_step_range": [items[0]["step"], items[-1]["step"]],
                "spike_sample_ratio": ratio,          # true ratio: sampling happened per bin, live
                "spike_sample_note": "sampled per bin during the live run (cap 1500/bin), not at export",
                "n_spikes_total": n_spikes_total,
                "active_neurons_note": "distinct neurons among the sampled spikes, not the true total",
                "safety_thresholds": SAFETY,
            })
        self._write_log({"kind": "snapshot", "run_id": run_id, "path": str(path), "n_frames": nb})
        return {"path": str(path), "n_frames": nb}


# ---------------------------------------------------------------------------------------
# replay (determinism check): re-apply a control log step by step
# ---------------------------------------------------------------------------------------
def replay_control_log(ctx: LiveContext, log_path: Path, n_steps: int, *,
                       device: str | torch.device = "cpu", backend: str | None = None,
                       collect_spikes: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Re-run ``n_steps`` from a control log and return the spikes ``(t_step, idx)``.

    The log's first line carries the engine and live parameters the session started with, so
    the replay needs nothing else. Audio input is **not** reproducible: the PCM payload is not
    logged (only the ``audio_start``/``audio_stop`` messages are), so a run that used the
    ``audio`` stimulus replays as silence for those stretches.
    """
    lines = [json.loads(l) for l in Path(log_path).read_text().splitlines() if l.strip()]
    if not lines or lines[0].get("kind") != "session_start":
        raise ValueError(f"{log_path} does not start with a session_start entry")
    head = lines[0]
    ep = head["engine_params"]
    fields = {f.name for f in dataclasses.fields(EngineParams)}
    params = EngineParams(**{k: v for k, v in ep.items() if k in fields})
    lp = head["live_params"]
    sess = LiveSession(ctx, params=params, device=device, backend=backend,
                       bin_ms=head.get("bin_ms", 1.0), seed=head.get("seed", 0),
                       a_in=lp.get("a_in", DEFAULT_A_IN), mode=lp.get("mode", "rate"),
                       speed=lp.get("speed", 1.0), log_controls=False)
    by_step: dict[int, list[dict]] = defaultdict(list)
    for e in lines[1:]:
        if e.get("kind") == "control":
            by_step[int(e["step"])].append(e["msg"])
    ts, ix = [], []
    for t in range(int(n_steps)):
        for msg in by_step.get(t, []):
            sess.apply_control(msg)
        sess._step_once()
        if collect_spikes:
            fired = torch.nonzero(sess.engine.spikes).flatten().cpu().numpy()
            if fired.size:
                ts.append(np.full(fired.size, t, dtype=np.int64))
                ix.append(fired.astype(np.int64))
    sess.stop()
    if not ts:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    return np.concatenate(ts), np.concatenate(ix)
