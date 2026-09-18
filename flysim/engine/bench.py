"""M2 performance benchmark on the FULL retained graph (no subgraph, no thresholds).

    python -m flysim.engine.bench [--steps 10000] [--warmup 100] [--out data-provenance/m2-bench.json]

Activity is forced, not simulated: a fixed random subset of p*N neurons receives
a huge constant current with t_ref = 0 and g = 0, so the LIF kernel emits
exactly p*N spikes every step (recorder sees real spikes) and the propagate
kernel walks exactly those rows on the next step. g = 0 keeps the activity from
spreading, which does not change the kernel cost (it depends only on which rows
are walked). Per-step cost is independent of dt; params use dt = 0.1 ms (the
phase-lock target) for the record.

Timing: warm-up steps excluded, ``torch.cuda.synchronize()`` then
``time.perf_counter`` around ``LIFEngine.run`` (so Python loop overhead, the
input callback and recorder flushes are all included).

M2c: ``--adapt`` adds a pass with the adaptation branch compiled in. Per-step cost
depends on whether the ``ADAPT`` constexpr is set, not on the value of ``adapt_b``
(the branch is a fixed load / multiply / subtract / select / store per neuron), so
the bench uses ``ADAPT_BENCH_B`` = 1e-6: large enough to switch the branch on,
small enough (w_ss = b/(1 - exp(-dt/tau_w)) = 1e-3 against i_ext = 1e4) to leave
the forced activity level of the protocol exactly as it is. The printed
spikes/step confirms the forcing still holds.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from ..graph import Graph
from .lif import LIFEngine
from .params import EngineParams

ACTIVITY_LEVELS = (0.0, 0.001, 0.01, 0.05)
ADAPT_BENCH_B = 1e-6          # switches the ADAPT branch on without perturbing the forced activity
ADAPT_G_BENCH_B = 1e-9        # same idea for M2d: g_a_ss = 1e-7/ms against a leak of 0.05/ms
STIM_CURRENT = 1.0e4          # >> threshold gap / (1 - decay_m): fires every step with t_ref = 0
TARGET_US = 100.0


def _time_run(engine: LIFEngine, steps: int, warmup: int, record: bool) -> tuple[float, int]:
    keep = lambda t, buf: None  # noqa: E731  (keeps the stimulus buffer untouched)
    engine.run(warmup, keep, record=record)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = engine.run(steps, keep, record=record)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / steps * 1e6
    n_spk = int(len(out[0])) if out is not None else -1
    return dt, n_spk


def run_bench(steps: int = 10_000, warmup: int = 100, seed: int = 0, synapse: str = "current",
              graph: Graph | None = None, adapt_b: float = 0.0, adapt_g_b: float = 0.0) -> dict:
    if not torch.cuda.is_available():
        raise SystemExit("bench requires CUDA (the target is the GPU kernel)")
    graph = graph or Graph.load()
    n = graph.n
    print(f"graph: {graph!r}  synapse={synapse}  adapt_b={adapt_b:g}  adapt_g_b={adapt_g_b:g}")
    params = EngineParams(dt=0.1, g=0.0, t_ref=0.0, synapse=synapse, adapt_b=adapt_b,
                          adapt_g_b=adapt_g_b)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    rows = []
    torch.cuda.reset_peak_memory_stats()
    base_alloc = torch.cuda.memory_allocated()
    engines = {
        "triton-eager": LIFEngine(graph, params, device="cuda", backend="triton", use_cuda_graph=False),
        "triton-cudagraph": LIFEngine(graph, params, device="cuda", backend="triton", use_cuda_graph=True),
        "torch-gpu": LIFEngine(graph, params, device="cuda", backend="torch"),
    }
    static_alloc = torch.cuda.memory_allocated()
    print(f"engine tensors: {engines['triton-eager'].gpu_memory_bytes() / 2**20:.0f} MiB each; "
          f"cuda allocated for 3 engines: {(static_alloc - base_alloc) / 2**20:.0f} MiB")
    for p in ACTIVITY_LEVELS:
        k = int(round(p * n))
        stim = torch.zeros(n, device="cuda")
        stim[torch.as_tensor(perm[:k], device="cuda")] = STIM_CURRENT
        for name, eng in engines.items():
            for record in (False, True):
                eng.reset(seed)
                eng.i_ext.copy_(stim)
                n_steps = steps if name != "torch-gpu" else max(steps // 10, 100)
                torch.cuda.reset_peak_memory_stats()
                us, n_spk = _time_run(eng, n_steps, warmup, record)
                transient = (torch.cuda.max_memory_allocated() - static_alloc) / 2**20
                spk_per_step = n_spk / n_steps if n_spk >= 0 else k
                rows.append(dict(synapse=synapse, adapt=params.adapt, adapt_b=adapt_b,
                                 adapt_g=params.adapt_g, adapt_g_b=adapt_g_b,
                                 backend=name, activity=p, forced_neurons=k, record=record,
                                 steps=n_steps, us_per_step=round(us, 1),
                                 spikes_per_step=round(spk_per_step, 1),
                                 transient_peak_mib=round(transient, 1)))
                print(f"{name:17s} p={p:<6} record={record!s:5} {us:7.1f} us/step  "
                      f"spikes/step={spk_per_step:8.1f}  transient peak={transient:.0f} MiB")
    result = dict(
        generated=datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
        n_neurons=n, n_edges=graph.m, steps=steps, warmup=warmup, seed=seed,
        params=params.to_dict(), stim_current=STIM_CURRENT, target_us=TARGET_US,
        engine_bytes=engines["triton-eager"].gpu_memory_bytes(),
        rows=rows,
    )
    return result


def markdown_table(result: dict) -> str:
    lines = ["| synapse | adapt | backend | activity | forced neurons | recorder | steps | us/step | spikes/step | transient peak MiB | <100us |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in result["rows"]:
        kind = "g_a" if r.get("adapt_g") else ("w" if r.get("adapt") else "off")
        lines.append(f"| {r.get('synapse', 'current')} | {kind} | "
                     f"{r['backend']} | {r['activity']:.1%} | {r['forced_neurons']:,} | "
                     f"{'on' if r['record'] else 'off'} | {r['steps']:,} | {r['us_per_step']:.1f} | "
                     f"{r['spikes_per_step']:.1f} | {r['transient_peak_mib']:.0f} | "
                     f"{'yes' if r['us_per_step'] < result['target_us'] else 'NO'} |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=10_000)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--out", type=Path, default=Path("data-provenance/m2-bench.json"))
    ap.add_argument("--synapse", choices=("current", "conductance", "both"), default="current")
    ap.add_argument("--adapt", action="store_true",
                    help="M2c: also bench each synapse model with the adaptation-current branch in")
    ap.add_argument("--adapt-g", action="store_true",
                    help="M2d: also bench the conductance model with the adaptation-conductance branch in")
    a = ap.parse_args()
    base = [("current", 0.0, 0.0), ("conductance", 0.0, 0.0)] if a.synapse == "both" \
        else [(a.synapse, 0.0, 0.0)]
    configs = list(base)
    if a.adapt:
        configs += [(syn, ADAPT_BENCH_B, 0.0) for syn, _, _ in base]
    if a.adapt_g:
        configs += [(syn, 0.0, ADAPT_G_BENCH_B) for syn, _, _ in base if syn == "conductance"]
    graph = Graph.load()
    res = None
    for syn, b, bg in configs:
        r = run_bench(a.steps, a.warmup, synapse=syn, graph=graph, adapt_b=b, adapt_g_b=bg)
        torch.cuda.empty_cache()
        if res is None:
            res = r
        else:
            res["rows"] += r["rows"]
            res[f"params_{syn}_adapt{b:g}_adaptg{bg:g}"] = r["params"]
    res["configs"] = [dict(synapse=syn, adapt_b=b, adapt_g_b=bg) for syn, b, bg in configs]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    print()
    print(markdown_table(res))
    print(f"\nwritten: {a.out}")


if __name__ == "__main__":
    main()
