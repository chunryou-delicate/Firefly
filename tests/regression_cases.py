"""Regression cases for the current-based (M2) engine path.

The reference file data-provenance/m2-regression-spikes.npz was generated with
the M2 engine (commit 8bae9d3, unchanged through 0b2c19b) BEFORE the M2b
conductance extension touched lif.py. tests/test_engine_conductance.py replays
every case with the default synapse="current" and requires bit-identical
(t, idx) spike lists and identical final v.

Regenerate (only if the reference itself must change — record why in the report):
    python tests/regression_cases.py --write
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
NPZ = ROOT / "data-provenance" / "m2-regression-spikes.npz"


def synth_graph():
    from flysim.engine import CSRGraph
    rng = np.random.default_rng(0)
    n, m = 5000, 200_000
    pre, post = rng.integers(0, n, m), rng.integers(0, n, m)
    w = rng.integers(1, 20, m) * np.where(rng.random(m) < 0.3, -1, 1)
    return CSRGraph.from_edges(n, pre, post, w)


def stim100(n, seed=0, amp=30.0, device="cuda", stim_steps=100):
    idx = np.random.default_rng(seed).choice(n, 100, replace=False)
    stim = torch.zeros(n, device=device)
    stim[torch.as_tensor(idx, device=device)] = amp

    def fn(t, buf):
        if t == 0:
            buf.copy_(stim)
        elif t == stim_steps:
            buf.zero_()
    return fn


# name -> (graph kind, params kwargs, engine kwargs, steps, reset seed, stim kwargs)
CASES = {
    "real_g1":            ("real",  dict(g=1.0),                          dict(device="cuda"),                      100, 0, dict()),
    "real_g1_cudagraph":  ("real",  dict(g=1.0),                          dict(device="cuda", use_cuda_graph=True), 100, 0, dict()),
    "real_g0336_noise20": ("real",  dict(g=0.336, noise_sigma=20.0),      dict(device="cuda"),                      300, 5, dict()),
    "real_dt01_g0336":    ("real",  dict(dt=0.1, g=0.336),                dict(device="cuda"),                      500, 0, dict(stim_steps=1000)),
    "synth_cpu":          ("synth", dict(g=0.5, noise_sigma=1.0),         dict(device="cpu"),                       150, 3, dict(amp=40.0, device="cpu")),
    "synth_cuda_torch":   ("synth", dict(g=0.5, noise_sigma=1.0),         dict(device="cuda", backend="torch"),     150, 3, dict(amp=40.0)),
}


def run_case(name, real_graph=None, synth=None):
    from flysim.engine import EngineParams, LIFEngine
    kind, pkw, ekw, steps, seed, skw = CASES[name]
    graph = real_graph if kind == "real" else synth
    e = LIFEngine(graph, EngineParams(**pkw), **ekw)
    dev = ekw.get("device", "cuda")
    e.reset(seed)
    t, idx = e.run(steps, stim100(e.n, **{"device": dev, **skw}))
    return t.astype(np.int32), idx.astype(np.int32), e.v.cpu().numpy().copy()


def write():
    from flysim.graph import Graph
    real = Graph.load()
    synth = synth_graph()
    out = {}
    for name in CASES:
        t, idx, v = run_case(name, real, synth)
        out[f"{name}__t"], out[f"{name}__idx"], out[f"{name}__v"] = t, idx, v
        print(f"{name:22s} spikes={len(t):8d}  v[min,max]=({v.min():.3f},{v.max():.3f})")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    meta = dict(commit=commit, torch=torch.__version__, gpu=torch.cuda.get_device_name(0),
                cases={k: dict(kind=c[0], params=c[1], engine=c[2], steps=c[3], seed=c[4], stim=c[5]) for k, c in CASES.items()})
    out["meta_json"] = np.array(json.dumps(meta))
    np.savez_compressed(NPZ, **out)
    print("written", NPZ, f"({NPZ.stat().st_size / 2**20:.1f} MiB)")


if __name__ == "__main__":
    if "--write" in sys.argv:
        write()
    else:
        print(__doc__)
