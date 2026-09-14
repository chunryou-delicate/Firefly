"""M2 acceptance sweep over the synaptic gain g (docs/m2-brief.md "g 스윕").

    python -m flysim.engine.sweep [--out data-provenance/m2-sweep.json]

Protocol and verdict criteria are FIXED HERE BEFORE RUNNING (constants below).
Changing any of them requires a note in docs/m2-report.md ("criteria history").

Protocol
  - full retained graph, dt = 1 ms, default EngineParams except g
  - stimulus: 100 neurons drawn uniformly at random (seed STIM_SEED) receive a
    constant current STIM_CURRENT for the first STIM_MS ms, then nothing
  - total run: STIM_MS + OBSERVE_MS; verdict computed on the last WINDOW_MS ms
Verdict on the window
  - EXTINCT : total spikes == 0
  - RUNAWAY : fraction of neurons that spiked at least once >= RUNAWAY_ACTIVE_FRAC,
              or rate in the last HALF_MS >= GROWTH_RATIO * rate in the preceding HALF_MS
              (a nonzero last half after a silent preceding half counts as growth)
  - VALID   : neither; additionally flagged "normal range" when the active
              fraction lies within [NORMAL_ACTIVE_MIN, NORMAL_ACTIVE_MAX]
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

# ---- protocol (pre-defined) --------------------------------------------------
DT_MS = 1.0
N_STIM_NEURONS = 100
STIM_SEED = 0
STIM_CURRENT = 30.0        # mV-equivalent (R = 1): 2x the 15 mV threshold gap -> ~60 Hz alone (ASSUMPTION)
STIM_MS = 100
OBSERVE_MS = 400
WINDOW_MS = 200            # verdict window = last 200 ms
HALF_MS = 50               # growth test: last 50 ms vs the 50 ms before
G_GRID = np.geomspace(1e-3, 1e1, 20)   # 20 points, log scale, 4 decades

# ---- verdict criteria (pre-defined) -------------------------------------------
RUNAWAY_ACTIVE_FRAC = 0.90
GROWTH_RATIO = 1.5
NORMAL_ACTIVE_MIN = 0.001
NORMAL_ACTIVE_MAX = 0.30


def verdict(t: np.ndarray, idx: np.ndarray, n: int, t_end: int) -> dict:
    """Apply the criteria above to a sorted (t, idx) spike list. Pure function."""
    w0 = t_end - WINDOW_MS
    win = t >= w0
    n_spk = int(win.sum())
    active = int(np.unique(idx[win]).size)
    active_frac = active / n
    rate_hz = n_spk / (n * WINDOW_MS / 1000.0)
    last = int((t >= t_end - HALF_MS).sum())
    prev = int(((t >= t_end - 2 * HALF_MS) & (t < t_end - HALF_MS)).sum())
    if n_spk == 0:
        v = "EXTINCT"
    elif active_frac >= RUNAWAY_ACTIVE_FRAC or (last > 0 and (prev == 0 or last >= GROWTH_RATIO * prev)):
        v = "RUNAWAY"
    else:
        v = "VALID"
    return dict(window_spikes=n_spk, active_frac=active_frac, mean_rate_hz=rate_hz,
                last_half=last, prev_half=prev,
                growth_ratio=(last / prev) if prev else (float("inf") if last else 0.0),
                verdict=v,
                normal_range=(v == "VALID" and NORMAL_ACTIVE_MIN <= active_frac <= NORMAL_ACTIVE_MAX))


def run_sweep(g_grid=G_GRID, seed: int = STIM_SEED) -> dict:
    graph = Graph.load()
    n = graph.n
    rng = np.random.default_rng(seed)
    stim_idx = rng.choice(n, N_STIM_NEURONS, replace=False)
    total_steps = int(round((STIM_MS + OBSERVE_MS) / DT_MS))
    stim_steps = int(round(STIM_MS / DT_MS))
    print(f"graph {graph!r}; {len(g_grid)} g points x {total_steps} steps; "
          f"expected a few seconds per point (more in the runaway regime, spike log transfer)")
    stim = torch.zeros(n, device="cuda")
    stim[torch.as_tensor(stim_idx, device="cuda")] = STIM_CURRENT

    def i_ext_fn(t, buf):
        if t == 0:
            buf.copy_(stim)
        elif t == stim_steps:
            buf.zero_()

    rows = []
    engine = None
    for g in g_grid:
        params = EngineParams(dt=DT_MS, g=float(g))
        if engine is None:
            engine = LIFEngine(graph, params, device="cuda")
        else:
            engine.params = params          # same buffers; kernels read params each launch
        engine.reset(seed)
        t0 = time.perf_counter()
        t, idx = engine.run(total_steps, i_ext_fn, record=True)
        el = time.perf_counter() - t0
        # spike count per 50 ms bin over the whole run (context for the report)
        bins = np.bincount(t // HALF_MS, minlength=total_steps // HALF_MS).tolist()
        r = dict(g=float(g), total_spikes=int(len(t)), bins_50ms=bins, seconds=round(el, 2),
                 v_min=float(engine.v.min()), v_max=float(engine.v.max()),
                 finite=bool(torch.isfinite(engine.v).all() and torch.isfinite(engine.i_syn).all()))
        r.update(verdict(t, idx, n, total_steps))
        rows.append(r)
        print(f"g={g:9.4f}  active={r['active_frac']:8.4%}  rate={r['mean_rate_hz']:9.3f} Hz  "
              f"last/prev={r['last_half']}/{r['prev_half']}  {r['verdict']}"
              f"{' (normal range)' if r['normal_range'] else ''}  [{el:.1f}s]")
        del t, idx
    valid = [r["g"] for r in rows if r["verdict"] == "VALID"]
    normal = [r["g"] for r in rows if r["normal_range"]]
    return dict(
        generated=datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        gpu=torch.cuda.get_device_name(0),
        n_neurons=n, n_edges=graph.m,
        protocol=dict(dt_ms=DT_MS, n_stim=N_STIM_NEURONS, stim_seed=seed, stim_current=STIM_CURRENT,
                      stim_ms=STIM_MS, observe_ms=OBSERVE_MS, window_ms=WINDOW_MS, half_ms=HALF_MS,
                      stim_body_ids=[int(b) for b in graph.body(stim_idx)]),
        criteria=dict(runaway_active_frac=RUNAWAY_ACTIVE_FRAC, growth_ratio=GROWTH_RATIO,
                      normal_active_min=NORMAL_ACTIVE_MIN, normal_active_max=NORMAL_ACTIVE_MAX),
        base_params={k: v for k, v in EngineParams(dt=DT_MS).to_dict().items() if k != "g"},
        rows=rows,
        valid_g=valid, normal_range_g=normal,
    )


def markdown_table(res: dict) -> str:
    lines = ["| g | active fraction | mean rate (Hz) | window spikes | last 50 / prev 50 | verdict |",
             "|---|---|---|---|---|---|"]
    for r in res["rows"]:
        v = r["verdict"] + (" (normal range)" if r["normal_range"] else "")
        lines.append(f"| {r['g']:.4g} | {r['active_frac']:.3%} | {r['mean_rate_hz']:.3f} | "
                     f"{r['window_spikes']:,} | {r['last_half']:,} / {r['prev_half']:,} | {v} |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--synapse", choices=("current", "conductance"), default="current")
    ap.add_argument("--noise", action="store_true", help="M2b: noise sweep at g_mid of the recorded RESPONSIVE range")
    ap.add_argument("--g", type=float, default=None, help="M2b noise sweep: override g_mid")
    a = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("sweep requires CUDA")
    if a.synapse == "conductance":
        from . import sweep_conductance as sc
        if a.noise:
            g = a.g
            if g is None:
                prev = json.loads(Path("data-provenance/m2b-sweep.json").read_text())
                g = prev["default_g_conductance"]
                if g is None:
                    raise SystemExit("no common RESPONSIVE range recorded; pass --g explicitly")
            res = sc.run_noise_sweep(g)
            out = a.out or Path("data-provenance/m2b-noise-sweep.json")
            out.write_text(json.dumps(res, indent=1))
            print(); print(sc.markdown_noise_table(res))
            print(f"\nstable low-rate sigma: {res['stable_low_rate_range']}\nwritten: {out}")
        else:
            res = sc.run_g_sweep()
            out = a.out or Path("data-provenance/m2b-sweep.json")
            out.write_text(json.dumps(res, indent=1))
            print(); print(sc.markdown_g_table(res))
            print(f"\nRESPONSIVE (all seeds): {res['responsive_range']} contiguous={res['responsive_contiguous']}"
                  f"\nDEFAULT_G_CONDUCTANCE (geometric mean): {res['default_g_conductance']}\nwritten: {out}")
        return
    a.out = a.out or Path("data-provenance/m2-sweep.json")
    res = run_sweep()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    print()
    print(markdown_table(res))
    print(f"\nvalid g: {res['valid_g']}\nnormal-range g: {res['normal_range_g']}\nwritten: {a.out}")


if __name__ == "__main__":
    main()
