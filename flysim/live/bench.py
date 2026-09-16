"""M5a: pacing measurement for the live loop (docs/m5a-report.md reproduces its output).

What one bin costs: the engine steps, the per-step bookkeeping, the frame computation with
its single ``.cpu()``, and — separately — the JSON serialisation the websocket layer adds.
Measured in two regimes, because cost follows activity: silent (no stimulus) and driven
(click train at the given gain).

Usage: python -m flysim.live.bench [--seconds 3] [--dt 1.0 0.1] [--a-in 320]
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import torch

from ..engine import EngineParams
from ..engine.params import DEFAULT_G_CONDUCTANCE
from . import protocol as P
from .export_neurons import build_context
from .session import LiveSession


def measure(ctx, dt: float, a_in: float, seconds: float, synapse: str, g: float,
            mode: str, stimulus: str) -> dict:
    params = EngineParams(synapse=synapse, g=g, dt=dt)
    s = LiveSession(ctx, params=params, device="cuda", a_in=a_in, mode=mode, speed=1e9,
                    log_controls=False)
    if stimulus != "silence":
        _, msg = P.validate_client_message({"type": "stimulus", "req_id": 0, "kind": stimulus,
                                            "ipi_ms": 35.0})
        s.apply_control(msg)
    n_bins = int(round(seconds * 1000.0 / s.bin_ms))
    for _ in range(50):                                    # warm-up (kernel compile, allocator)
        s._bundle_frame(s._run_bin())
    s.metrics.update(bins=0, bin_us_sum=0.0, bin_us_max=0.0, steps=0)
    json_us = 0.0
    n_json = 0
    spikes = 0
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_bins):
        payload = s._run_bin()
        spikes += payload["n_total"]
        frame = s._bundle_frame(payload)
        if frame is not None:
            tj = time.perf_counter()
            json.dumps(frame, separators=(",", ":"))
            json_us += (time.perf_counter() - tj) * 1e6
            n_json += 1
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    m = s.pacing_metrics()
    out = {"dt_ms": dt, "mode": mode, "stimulus": stimulus, "a_in": a_in, "synapse": synapse,
           "g": g, "bins": n_bins, "sim_ms": n_bins * s.bin_ms,
           "mean_bin_us": m["mean_bin_us"], "max_bin_us": m["max_bin_us"],
           "mean_json_us": round(json_us / max(n_json, 1), 1),
           "mean_total_us": round(m["mean_bin_us"] + json_us / max(n_json, 1), 1),
           "spikes_per_bin": round(spikes / n_bins, 1),
           "wall_s": round(wall, 2),
           "realtime_factor": round(n_bins * s.bin_ms / 1000.0 / wall, 2)}
    s.stop()
    del s
    torch.cuda.empty_cache()
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--dt", type=float, nargs="+", default=[1.0, 0.1])
    ap.add_argument("--a-in", dest="a_in", type=float, default=320.0)
    ap.add_argument("--synapse", default="conductance")
    ap.add_argument("--g", type=float, default=DEFAULT_G_CONDUCTANCE)
    args = ap.parse_args(argv)
    if not torch.cuda.is_available():
        print("CUDA is required for a meaningful pacing measurement")
        return 1
    ctx = build_context()
    rows = []
    for dt in args.dt:
        mode = "rate" if dt >= 1.0 else "phase-lock"
        for stim in ("silence", "click_train"):
            r = measure(ctx, dt, args.a_in, args.seconds, args.synapse, args.g, mode, stim)
            rows.append(r)
            print(json.dumps(r), flush=True)
    print("\n| dt (ms) | mode | stimulus | spikes/bin | mean bin μs | + JSON μs | total μs | "
          "real-time factor |\n|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['dt_ms']} | {r['mode']} | {r['stimulus']} | {r['spikes_per_bin']:.0f} | "
              f"{r['mean_bin_us']:.0f} | {r['mean_json_us']:.0f} | {r['mean_total_us']:.0f} | "
              f"{r['realtime_factor']:.2f}× |")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
