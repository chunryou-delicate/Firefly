"""Regression cases for the engine kernels that later milestones must not perturb.

data-provenance/m2-regression-spikes.npz holds two generations of reference:

* the six ``synapse="current"`` cases, generated with the M2 engine (commit
  8bae9d3, unchanged through 0b2c19b) BEFORE the M2b conductance extension
  touched lif.py;
* ``real_cond_default`` (M2c), generated with the M2b engine (commit 15b880d)
  BEFORE the M2c adaptation current touched lif.py — the conductance model at
  DEFAULT_G_CONDUCTANCE on the full graph for 100 steps.

tests/test_engine_conductance.py replays every case and requires bit-identical
(t, idx) spike lists and identical final v. The M2c case is the evidence for the
brief's requirement that ``adapt_b = 0`` leaves the pre-M2c kernel path untouched.

Regenerate (only if the reference itself must change — record why in the report):
    python tests/regression_cases.py --write                    # all cases
    python tests/regression_cases.py --write --only NAME [...]  # merge single cases
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from flysim.engine.params import DEFAULT_G_CONDUCTANCE

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
    # M2c: conductance model at the recorded default g, added before the adaptation
    # current existed (docs/m2c-brief.md). Guards the adapt_b = 0 path.
    "real_cond_default":  ("real",  dict(synapse="conductance", g=DEFAULT_G_CONDUCTANCE),
                                                                dict(device="cuda"),                      100, 0, dict()),
}

CURRENT_CASES = [k for k, c in CASES.items() if c[1].get("synapse", "current") == "current"]
CONDUCTANCE_CASES = [k for k in CASES if k not in CURRENT_CASES]


def run_case(name, real_graph=None, synth=None):
    from flysim.engine import EngineParams, LIFEngine
    kind, pkw, ekw, steps, seed, skw = CASES[name]
    graph = real_graph if kind == "real" else synth
    e = LIFEngine(graph, EngineParams(**pkw), **ekw)
    dev = ekw.get("device", "cuda")
    e.reset(seed)
    t, idx = e.run(steps, stim100(e.n, **{"device": dev, **skw}))
    return t.astype(np.int32), idx.astype(np.int32), e.v.cpu().numpy().copy()


def write(only: list[str] | None = None):
    """Write the reference file. ``only`` merges the named cases into the existing
    file and leaves every other array (and the recorded commit of the earlier
    generation) untouched — adding a case must not silently regenerate the others."""
    from flysim.graph import Graph
    names = list(only or CASES)
    for name in names:
        if name not in CASES:
            raise SystemExit(f"unknown case {name!r}; known: {', '.join(CASES)}")
    out, meta = {}, {}
    if only and NPZ.exists():
        with np.load(NPZ) as z:
            out = {k: z[k] for k in z.files}
        meta = json.loads(str(out.pop("meta_json")))
    real = Graph.load() if any(CASES[n][0] == "real" for n in names) else None
    synth = synth_graph() if any(CASES[n][0] == "synth" for n in names) else None
    for name in names:
        t, idx, v = run_case(name, real, synth)
        out[f"{name}__t"], out[f"{name}__idx"], out[f"{name}__v"] = t, idx, v
        print(f"{name:22s} spikes={len(t):8d}  v[min,max]=({v.min():.3f},{v.max():.3f})")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    cases = meta.get("cases", {})
    cases.update({k: dict(kind=CASES[k][0], params=CASES[k][1], engine=CASES[k][2],
                          steps=CASES[k][3], seed=CASES[k][4], stim=CASES[k][5]) for k in names})
    gens = meta.get("generations", [])
    if not gens and meta.get("commit"):   # first generation, recorded before `generations` existed
        gens = [dict(commit=meta["commit"], torch=meta.get("torch"), gpu=meta.get("gpu"),
                     cases=[k for k in cases if k not in names])]
    gens.append(dict(commit=commit, torch=torch.__version__, gpu=torch.cuda.get_device_name(0), cases=names))
    meta.update(commit=meta.get("commit", commit), torch=meta.get("torch", torch.__version__),
                gpu=meta.get("gpu", torch.cuda.get_device_name(0)), cases=cases, generations=gens)
    out["meta_json"] = np.array(json.dumps(meta))
    np.savez_compressed(NPZ, **out)
    print("written", NPZ, f"({NPZ.stat().st_size / 2**20:.1f} MiB)")


if __name__ == "__main__":
    if "--write" in sys.argv:
        only = sys.argv[sys.argv.index("--only") + 1:] if "--only" in sys.argv else None
        write(only)
    else:
        print(__doc__)
