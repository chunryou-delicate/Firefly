"""M2b/M2c sweeps for the conductance-based synapse model (docs/m2b-brief.md, docs/m2c-brief.md).

    python -m flysim.engine.sweep --synapse conductance                      # M2b g sweep, 3 seeds
    python -m flysim.engine.sweep --synapse conductance --noise              # M2b noise sweep at g_mid
    python -m flysim.engine.sweep --synapse conductance --adapt              # M2c b response sweep
    python -m flysim.engine.sweep --synapse conductance --adapt --noise      # M2c b x sigma noise sweep
    python -m flysim.engine.sweep --synapse conductance --adapt --g-resweep  # M2c g re-sweep at b*
    python -m flysim.engine.sweep --synapse conductance --adapt-g            # M2d b_g x g grid
    python -m flysim.engine.sweep --synapse conductance --adapt-g --noise    # M2d noise sweep

Protocol and verdict criteria are FIXED HERE BEFORE RUNNING. Changing them
requires a note in docs/m2b-report.md.

g sweep protocol (same stimulus as M2): full retained graph, dt = 1 ms, 100 random
neurons (seeds 0/1/2) receive constant current 30 for 100 ms, then 400 ms without
input. Per g and seed we record downstream spikes during the stimulus (stimulated
neurons excluded), spikes in the post-offset windows 0-50 / 50-200 / 200-400 ms,
active fraction and mean rate of active neurons in the 200-400 ms window, and
min/max v over the run.

Verdict per seed
  SILENT     : no downstream spike during the stimulus
  IGNITED    : fraction of neurons spiking in the 200-400 ms post-offset window >= 1 %
  RESPONSIVE : downstream spikes > 0 and not IGNITED        <- target regime
The RESPONSIVE interval reported is the set of grid points RESPONSIVE for all
three seeds. DEFAULT_G_CONDUCTANCE = geometric mean of its lower and upper end.

g grid: 20 log-spaced points in [1e-5, 1e-2] (1/ms per contact). Reason: a
conductance g drives a current g*(E_exc - v) ~ 65*g mV/ms near rest, i.e. the
current model's gain g_cur corresponds to g ~ g_cur / ((E_exc - v_rest)*tau_m)
= g_cur / 1300. The M2 current-model ignition threshold (g_cur ~ 0.55) maps to
~4e-4, so the grid spans two decades either side of it. Confirmed by a 4-point
prototype run before the sweep (recorded in the report).

Noise sweep (at g_mid): no input; noise_sigma on a log grid for 2 s, then 0.4 s
with the noise switched off. Recorded: mean rate over all N in the first and last
noise-on second, active fraction in the last second, ratio last/first. Verdicts:
  IGNITED : active fraction in the final 200 ms of the noise-off tail >= 1 %
            (self-sustained after the noise stops; same criterion as the g sweep)
  STABLE_LOW_RATE : not IGNITED, 0.1 <= rate_last <= 5 Hz, 0.5 <= last/first <= 2

------------------------------------------------------------------------------
M2c (docs/m2c-brief.md): the same three protocols with the adaptation current on.
Nothing below is changed for M2c except adapt_b; tau_w = 100 ms is fixed by the
brief and the verdict functions are the M2b ones, unmodified.

  1. b response sweep   -- the g-sweep protocol (100 neurons, 100 ms, seeds 0/1/2)
                           at g = DEFAULT_G_CONDUCTANCE, over B_GRID.
  2. b x sigma noise sweep -- the noise protocol at every b of B_GRID.
  3. g re-sweep         -- the g-sweep protocol at the b chosen by the rule below,
                           over G_REGRID, to remeasure the RESPONSIVE window width.

TARGET STATE (pre-registered here before any M2c run; it is exactly the M2b
STABLE_LOW_RATE predicate, so LOW_RATE_RANGE_HZ / STABLE_RATIO_RANGE /
IGNITED_ACTIVE_FRAC below are reused unchanged):
    mean rate over the last noise-on second in [0.1, 5] Hz per neuron, AND
    last-second / first-second spike count in [0.5, 2], AND
    active fraction in the final 200 ms after the noise stops < 1 %.

DEFAULT_ADAPT_B RULE (pre-registered): the smallest b on B_GRID for which some
sigma reaches the target state. If no b does, DEFAULT_ADAPT_B stays 0 and the
report says "none".

B_GRID = 12 log-spaced points in [0.03, 300]. Reason (recorded before the run):
b has the units of i_ext, so a neuron firing at rate r settles at
w_ss ~ b * tau_w * r = b * (0.1 s) * r. Setting w_ss equal to the 15 mV threshold
gap gives the b at which adaptation is worth one full threshold gap at a given
rate: b = 1.0 at 150 Hz (the ignited saturation rate of docs/m2b-report.md §5),
b = 10 at 15 Hz (the noise-ignited rate of §6), b = 150 at 1 Hz (the middle of the
target range). The grid brackets all three, with a decade below the first (at
b = 0.03, w_ss = 0.045 mV at 15 Hz, i.e. below the 0.065 mV unitary EPSP -- a
negligibility control) and a factor 2 above the last. A 6-point prototype
(b = 0, 0.01, 0.1, 1, 10, 100; sigma = 29.13 and the seed-0 stimulus) was run
before fixing the grid and is reported in docs/m2c-report.md: b <= 0.1 changes
nothing, b = 10 takes the noise-ignited state from 15.5 to 3.6 Hz, b = 100 to
0.37 Hz -- so the transition lies inside the grid.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import numpy as np
import torch

from ..graph import Graph
from .lif import LIFEngine
from .params import DEFAULT_G_CONDUCTANCE, EngineParams

# ---- protocol (pre-defined) ---------------------------------------------------
DT_MS = 1.0
N_STIM = 100
SEEDS = (0, 1, 2)
STIM_CURRENT = 30.0
STIM_MS = 100
OBSERVE_MS = 400
WINDOWS_MS = ((0, 50), (50, 200), (200, 400))     # post-offset windows
LATE_WINDOW = (200, 400)
G_GRID = np.geomspace(1e-5, 1e-2, 20)
IGNITED_ACTIVE_FRAC = 0.01

# ---- M2c (pre-defined; see the module doc) -----------------------------------
B_GRID = np.geomspace(0.03, 300.0, 12)
G_REGRID = np.geomspace(1e-5, 1e-2, 12)     # same span as G_GRID, 12 points per the brief

# ---- M2d (pre-defined; docs/m2d-brief.md) ------------------------------------
# b_g grid: 12 log-spaced points in [1e-3, 1]. g_a has the unit of the synaptic
# conductances (1/ms) and the leak is 1/tau_m = 0.05/ms, so a neuron firing at rate r
# settles at g_a_ss ~ b_g * tau_a * r = b_g * (0.1 s) * r. Setting g_a_ss equal to the
# leak - i.e. adaptation doubling the neuron's resting conductance - gives
#   b_g = 0.0033 at 150 Hz (the ignited saturation rate, m2b-report.md §5),
#   b_g = 0.033  at 15 Hz  (the noise-ignited rate, m2b-report.md §6),
#   b_g = 0.5    at 1 Hz   (the middle of the usable band).
# The grid brackets all three. A prototype at sigma = 29.13 before fixing the grid
# (docs/m2d-report.md §4.1) confirmed the transition lies inside it: b_g = 1e-3 barely
# moves the rate (15.5 -> 14.5 Hz), b_g = 0.1 reaches 0.58 Hz, and it saturates by 1.
B_G_GRID = np.geomspace(1e-3, 1.0, 12)
# g grid: 8 log-spaced points in [2e-4, 2e-3], the span named by the brief. It brackets
# M2b's RESPONSIVE window [2.637e-4, 3.793e-4] and the point the M2c re-sweep moved it
# to (4.329e-4), with the top of the range well inside M2b's ignited regime.
G_GRID_ADAPT_G = np.geomspace(2e-4, 2e-3, 8)
# Fourth usable-state condition, new in M2d (CLAUDE.md §3.3; M2c reached 96-100 %).
USABLE_ACTIVE_FRAC_MAX = 0.30

# ---- M2e (pre-defined; docs/m2e-brief.md, written before the run) -------------
# M2d's selection rule ("smallest b_g with a usable state") returned b_g = 1.0, the TOP
# of B_G_GRID, so the rule was not resolved: the true minimum could lie inside
# (0.534, 1.0], and nothing above 1.0 was tested. M2e opens the grid in both directions
# and changes NOTHING else - the four usable-state conditions and the selection rule are
# the M2d ones, reused from the same constants and the same _judge_usable().
#   refine:  5 geometric points strictly inside [0.534, 1.0]
#   extend:  6 geometric points on [1.0, 8.0], 1.0 included
# g grid (8 points) and the 3 stimulus seeds are identical to M2d.
_B_G_M2D_LOW = float(B_G_GRID[10])                     # 0.53367..., the M2d point below the hit
B_G_GRID_REFINE = np.geomspace(_B_G_M2D_LOW, 1.0, 7)[1:-1]
B_G_GRID_EXTEND = np.geomspace(1.0, 8.0, 6)
B_G_GRID_M2E = np.concatenate([B_G_GRID_REFINE, B_G_GRID_EXTEND])
# Robustness check (descriptive, NOT part of the selection rule): the selected cell is
# re-run with these noise seeds. If any of them fails the four conditions at a sigma that
# was usable with the primary seed, the result is reported as "not robust" - and no
# substitute cell is looked for (docs/m2e-brief.md §4).
ROBUSTNESS_SEEDS = (3, 4, 5)

NOISE_SIGMA_GRID = np.geomspace(1.0, 200.0, 12)
NOISE_ON_MS = 2000
NOISE_OFF_MS = 400
NOISE_TAIL_WINDOW_MS = 200
LOW_RATE_RANGE_HZ = (0.1, 5.0)
STABLE_RATIO_RANGE = (0.5, 2.0)


def _params(g: float, adapt_b: float = 0.0, adapt_g_b: float = 0.0, **kw) -> EngineParams:
    return EngineParams(dt=DT_MS, synapse="conductance", g=float(g), adapt_b=float(adapt_b),
                        adapt_g_b=float(adapt_g_b), **kw)


def _run_tracked(engine: LIFEngine, n_steps: int, i_ext_fn) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Step loop that also tracks the per-step min/max of v (no host sync per step)."""
    vmin = engine.v.clone()
    vmax = engine.v.clone()
    for k in range(n_steps):
        i_ext_fn(k, engine.i_ext)
        engine.step(engine.i_ext)
        torch.minimum(vmin, engine.v, out=vmin)
        torch.maximum(vmax, engine.v, out=vmax)
    t, idx = engine.recorder.collect()
    return t, idx, float(vmin.min()), float(vmax.max())


def judge_g(t: np.ndarray, idx: np.ndarray, stim_idx: np.ndarray, n: int) -> dict:
    stim_mask = np.zeros(n, bool)
    stim_mask[stim_idx] = True
    during = t < STIM_MS
    downstream_during = int((during & ~stim_mask[idx]).sum())
    stim_spikes_during = int((during & stim_mask[idx]).sum())
    post = t - STIM_MS
    win = {}
    for a, b in WINDOWS_MS:
        m = (post >= a) & (post < b)
        win[f"post_{a}_{b}"] = int(m.sum())
    a, b = LATE_WINDOW
    late = (post >= a) & (post < b)
    n_late = int(late.sum())
    active_late = int(np.unique(idx[late]).size)
    active_frac = active_late / n
    rate_active_hz = (n_late / (active_late * (b - a) / 1000.0)) if active_late else 0.0
    rate_all_hz = n_late / (n * (b - a) / 1000.0)
    if downstream_during == 0:
        v = "SILENT"
    elif active_frac >= IGNITED_ACTIVE_FRAC:
        v = "IGNITED"
    else:
        v = "RESPONSIVE"
    return dict(downstream_during_stim=downstream_during, stim_neuron_spikes_during_stim=stim_spikes_during,
                **win, late_active_frac=active_frac, late_active_neurons=active_late,
                late_rate_active_hz=rate_active_hz, late_rate_all_hz=rate_all_hz, verdict=v)


def run_g_sweep(g_grid=G_GRID, seeds=SEEDS, adapt_b: float = 0.0, graph=None) -> dict:
    graph = graph if graph is not None else Graph.load()
    n = graph.n
    total = STIM_MS + OBSERVE_MS
    print(f"g sweep (adapt_b={adapt_b:g}): {len(g_grid)} g x {len(seeds)} seeds x {total} steps ≈ "
          f"{len(g_grid)*len(seeds)*total:,} steps; expected < 1 min at ~100 us/step plus spike transfer "
          f"in ignited runs", flush=True)
    engine = LIFEngine(graph, _params(g_grid[0], adapt_b), device="cuda")
    stims = {}
    for seed in seeds:
        stim_idx = np.random.default_rng(seed).choice(n, N_STIM, replace=False)
        stim = torch.zeros(n, device="cuda")
        stim[torch.as_tensor(stim_idx, device="cuda")] = STIM_CURRENT
        stims[seed] = (stim_idx, stim)
    rows = []
    t0 = time.perf_counter()
    for g in g_grid:
        engine.params = _params(g, adapt_b)
        for seed in seeds:
            stim_idx, stim = stims[seed]

            def fn(k, buf, stim=stim):
                if k == 0:
                    buf.copy_(stim)
                elif k == STIM_MS:
                    buf.zero_()
            engine.reset(seed)
            ts = time.perf_counter()
            t, idx, vmin, vmax = _run_tracked(engine, total, fn)
            r = dict(g=float(g), adapt_b=float(adapt_b), seed=seed, total_spikes=int(len(t)),
                     w_max=float(engine.w.max()), w_mean=float(engine.w.mean()), v_min=vmin, v_max=vmax,
                     finite=bool(torch.isfinite(engine.v).all() and torch.isfinite(engine.g_e).all()
                                 and torch.isfinite(engine.g_i).all()),
                     bins_50ms=np.bincount(t // 50, minlength=total // 50).tolist(),
                     seconds=round(time.perf_counter() - ts, 2))
            r.update(judge_g(t, idx, stim_idx, n))
            rows.append(r)
            print(f"g={g:9.3e} seed={seed}  downstream(stim)={r['downstream_during_stim']:7d}  "
                  f"post 0-50/50-200/200-400={r['post_0_50']:7d}/{r['post_50_200']:7d}/{r['post_200_400']:8d}  "
                  f"late active={r['late_active_frac']:8.4%}  rate(active)={r['late_rate_active_hz']:7.1f} Hz  "
                  f"v[{vmin:8.1f},{vmax:6.1f}]  {r['verdict']}", flush=True)
    # per-g verdict across seeds
    per_g = []
    for g in g_grid:
        vs = [r["verdict"] for r in rows if r["g"] == float(g)]
        per_g.append(dict(g=float(g), verdicts=vs,
                          all_responsive=all(v == "RESPONSIVE" for v in vs),
                          any_ignited=any(v == "IGNITED" for v in vs),
                          all_silent=all(v == "SILENT" for v in vs)))
    resp = [x["g"] for x in per_g if x["all_responsive"]]
    contiguous = None
    if resp:
        gi = [i for i, x in enumerate(per_g) if x["all_responsive"]]
        contiguous = gi == list(range(gi[0], gi[-1] + 1))
    default_g = float(np.sqrt(min(resp) * max(resp))) if resp else None
    return dict(
        generated=datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        gpu=torch.cuda.get_device_name(0), n_neurons=n, n_edges=graph.m,
        protocol=dict(dt_ms=DT_MS, n_stim=N_STIM, seeds=list(seeds), stim_current=STIM_CURRENT, stim_ms=STIM_MS,
                      observe_ms=OBSERVE_MS, windows_ms=WINDOWS_MS, late_window_ms=LATE_WINDOW,
                      ignited_active_frac=IGNITED_ACTIVE_FRAC, g_grid=[float(g) for g in g_grid],
                      stim_body_ids={int(s): [int(b) for b in graph.body(stims[s][0])] for s in seeds}),
        adapt_b=float(adapt_b),
        base_params={k: v for k, v in _params(g_grid[0], adapt_b).to_dict().items() if k != "g"},
        rows=rows, per_g=per_g,
        responsive_common_g=resp, responsive_contiguous=contiguous,
        responsive_range=[min(resp), max(resp)] if resp else None,
        default_g_rule="geometric mean of the RESPONSIVE interval common to all seeds",
        default_g_conductance=default_g,
        total_seconds=round(time.perf_counter() - t0, 1),
    )


def run_noise_sweep(g: float, sigma_grid=NOISE_SIGMA_GRID, seed: int = 0, adapt_b: float = 0.0,
                    graph=None, engine=None) -> dict:
    graph = graph if graph is not None else Graph.load()
    n = graph.n
    on, off = NOISE_ON_MS, NOISE_OFF_MS
    print(f"noise sweep at g={g:.3e}, adapt_b={adapt_b:g}: {len(sigma_grid)} sigma x {on + off} steps ≈ "
          f"{len(sigma_grid)*(on+off):,} steps; expected ~1 min plus spike transfer", flush=True)
    engine = engine if engine is not None else LIFEngine(graph, _params(g, adapt_b), device="cuda")
    zero = lambda k, buf: None  # noqa: E731
    rows = []
    for sigma in sigma_grid:
        engine.params = _params(g, adapt_b, noise_sigma=float(sigma))
        engine.reset(seed)
        engine.i_ext.zero_()
        ts = time.perf_counter()
        t1, i1, vmin1, vmax1 = _run_tracked(engine, on, zero)
        engine.params = _params(g, adapt_b)              # noise off, state kept
        t2, i2, vmin2, vmax2 = _run_tracked(engine, off, zero)
        first = (t1 < 1000)
        last = (t1 >= on - 1000)
        rate_first = int(first.sum()) / (n * 1.0)
        rate_last = int(last.sum()) / (n * 1.0)
        active_last = int(np.unique(i1[last]).size) / n
        tail = t2 >= off - NOISE_TAIL_WINDOW_MS
        tail_active = int(np.unique(i2[tail]).size) / n
        ratio = (rate_last / rate_first) if rate_first > 0 else (float("inf") if rate_last > 0 else 0.0)
        ignited = tail_active >= IGNITED_ACTIVE_FRAC
        stable_low = (not ignited) and LOW_RATE_RANGE_HZ[0] <= rate_last <= LOW_RATE_RANGE_HZ[1] \
            and STABLE_RATIO_RANGE[0] <= ratio <= STABLE_RATIO_RANGE[1]
        r = dict(sigma=float(sigma), adapt_b=float(adapt_b), spikes_on=int(len(t1)), spikes_off=int(len(t2)),
                 w_max=float(engine.w.max()), w_mean=float(engine.w.mean()),
                 rate_first_s_hz=rate_first, rate_last_s_hz=rate_last, active_frac_last_s=active_last,
                 ratio_last_first=ratio, tail_active_frac=tail_active, ignited=ignited,
                 stable_low_rate=stable_low, v_min=min(vmin1, vmin2), v_max=max(vmax1, vmax2),
                 bins_200ms_on=np.bincount(t1 // 200, minlength=on // 200).tolist(),
                 bins_200ms_off=np.bincount(t2 // 200, minlength=off // 200).tolist(),
                 seconds=round(time.perf_counter() - ts, 2))
        rows.append(r)
        print(f"sigma={sigma:8.2f}  rate first/last s={rate_first:8.3f}/{rate_last:8.3f} Hz  active(last s)={active_last:8.4%}  "
              f"ratio={ratio:6.2f}  tail active={tail_active:8.4%}  v[{r['v_min']:7.1f},{r['v_max']:6.1f}]  "
              f"{'IGNITED' if ignited else ('STABLE_LOW_RATE' if stable_low else '-')}", flush=True)
    stable = [r["sigma"] for r in rows if r["stable_low_rate"]]
    return dict(
        generated=datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        gpu=torch.cuda.get_device_name(0), n_neurons=n, g=float(g), seed=seed, adapt_b=float(adapt_b),
        protocol=dict(noise_on_ms=on, noise_off_ms=off, tail_window_ms=NOISE_TAIL_WINDOW_MS,
                      sigma_grid=[float(s) for s in sigma_grid], ignited_active_frac=IGNITED_ACTIVE_FRAC,
                      low_rate_range_hz=LOW_RATE_RANGE_HZ, stable_ratio_range=STABLE_RATIO_RANGE),
        base_params=_params(g, adapt_b).to_dict(), rows=rows,
        stable_low_rate_sigma=stable, stable_low_rate_range=[min(stable), max(stable)] if stable else None,
    )


def markdown_g_table(res: dict) -> str:
    lines = ["| g (1/ms) | seed | downstream during stim | post 0-50 / 50-200 / 200-400 ms | late active | rate (active, Hz) | v min / max (mV) | verdict |",
             "|---|---|---|---|---|---|---|---|"]
    for r in res["rows"]:
        lines.append(f"| {r['g']:.3e} | {r['seed']} | {r['downstream_during_stim']:,} | "
                     f"{r['post_0_50']:,} / {r['post_50_200']:,} / {r['post_200_400']:,} | {r['late_active_frac']:.3%} | "
                     f"{r['late_rate_active_hz']:.1f} | {r['v_min']:.1f} / {r['v_max']:.1f} | {r['verdict']} |")
    return "\n".join(lines)


def markdown_noise_table(res: dict) -> str:
    lines = ["| sigma | rate first s (Hz) | rate last s (Hz) | active (last s) | last/first | tail active (noise off) | v min / max | verdict |",
             "|---|---|---|---|---|---|---|---|"]
    for r in res["rows"]:
        v = "IGNITED" if r["ignited"] else ("STABLE_LOW_RATE" if r["stable_low_rate"] else "-")
        lines.append(f"| {r['sigma']:.2f} | {r['rate_first_s_hz']:.3f} | {r['rate_last_s_hz']:.3f} | {r['active_frac_last_s']:.3%} | "
                     f"{r['ratio_last_first']:.2f} | {r['tail_active_frac']:.3%} | {r['v_min']:.1f} / {r['v_max']:.1f} | {v} |")
    return "\n".join(lines)


# ==============================================================================
# M2c: adaptation current (docs/m2c-brief.md). Grids and the target state are
# pre-registered in the module doc; the verdict functions above are reused as is.
# ==============================================================================
def run_b_sweep(b_grid=B_GRID, g: float = DEFAULT_G_CONDUCTANCE, seeds=SEEDS, graph=None) -> dict:
    """Response sweep: the M2b stimulus protocol at fixed g, over adapt_b."""
    graph = graph if graph is not None else Graph.load()
    n = graph.n
    total = STIM_MS + OBSERVE_MS
    print(f"M2c b sweep at g={g:.4e}: {len(b_grid)} b x {len(seeds)} seeds x {total} steps ≈ "
          f"{len(b_grid)*len(seeds)*total:,} steps; expected < 1 min", flush=True)
    engine = LIFEngine(graph, _params(g, b_grid[0]), device="cuda")
    stims = {}
    for seed in seeds:
        stim_idx = np.random.default_rng(seed).choice(n, N_STIM, replace=False)
        stim = torch.zeros(n, device="cuda")
        stim[torch.as_tensor(stim_idx, device="cuda")] = STIM_CURRENT
        stims[seed] = (stim_idx, stim)
    rows = []
    t0 = time.perf_counter()
    for b in b_grid:
        engine.params = _params(g, b)
        for seed in seeds:
            stim_idx, stim = stims[seed]

            def fn(k, buf, stim=stim):
                if k == 0:
                    buf.copy_(stim)
                elif k == STIM_MS:
                    buf.zero_()
            engine.reset(seed)
            ts = time.perf_counter()
            t, idx, vmin, vmax = _run_tracked(engine, total, fn)
            r = dict(adapt_b=float(b), g=float(g), seed=seed, total_spikes=int(len(t)),
                     w_max=float(engine.w.max()), w_mean=float(engine.w.mean()),
                     v_min=vmin, v_max=vmax,
                     finite=bool(torch.isfinite(engine.v).all() and torch.isfinite(engine.w).all()),
                     bins_50ms=np.bincount(t // 50, minlength=total // 50).tolist(),
                     seconds=round(time.perf_counter() - ts, 2))
            r.update(judge_g(t, idx, stim_idx, n))
            rows.append(r)
            print(f"b={b:9.3f} seed={seed}  stim spikes={r['stim_neuron_spikes_during_stim']:6d}  "
                  f"downstream(stim)={r['downstream_during_stim']:7d}  "
                  f"post 0-50/50-200/200-400={r['post_0_50']:7d}/{r['post_50_200']:7d}/{r['post_200_400']:8d}  "
                  f"late active={r['late_active_frac']:8.4%}  rate(active)={r['late_rate_active_hz']:7.1f} Hz  "
                  f"w[max]={r['w_max']:8.2f}  {r['verdict']}", flush=True)
    per_b = []
    for b in b_grid:
        vs = [r["verdict"] for r in rows if r["adapt_b"] == float(b)]
        per_b.append(dict(adapt_b=float(b), verdicts=vs,
                          all_responsive=all(v == "RESPONSIVE" for v in vs),
                          any_ignited=any(v == "IGNITED" for v in vs),
                          all_silent=all(v == "SILENT" for v in vs)))
    return dict(
        generated=datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        gpu=torch.cuda.get_device_name(0), n_neurons=n, n_edges=graph.m, g=float(g),
        protocol=dict(dt_ms=DT_MS, n_stim=N_STIM, seeds=list(seeds), stim_current=STIM_CURRENT, stim_ms=STIM_MS,
                      observe_ms=OBSERVE_MS, windows_ms=WINDOWS_MS, late_window_ms=LATE_WINDOW,
                      ignited_active_frac=IGNITED_ACTIVE_FRAC, b_grid=[float(b) for b in b_grid],
                      adapt_tau_w=_params(g, 1.0).adapt_tau_w),
        base_params={k: v for k, v in _params(g, 0.0).to_dict().items() if k != "adapt_b"},
        rows=rows, per_b=per_b, total_seconds=round(time.perf_counter() - t0, 1),
    )


def run_b_noise_sweep(b_grid=B_GRID, g: float = DEFAULT_G_CONDUCTANCE, sigma_grid=NOISE_SIGMA_GRID,
                      seed: int = 0, graph=None) -> dict:
    """Noise sweep at every b. The target state is the M2b STABLE_LOW_RATE predicate."""
    graph = graph if graph is not None else Graph.load()
    steps = len(b_grid) * len(sigma_grid) * (NOISE_ON_MS + NOISE_OFF_MS)
    print(f"M2c noise sweep: {len(b_grid)} b x {len(sigma_grid)} sigma x {NOISE_ON_MS + NOISE_OFF_MS} steps "
          f"= {steps:,} steps; expected ~10-15 min (dominated by spike transfer in ignited runs)", flush=True)
    engine = LIFEngine(graph, _params(g, b_grid[0]), device="cuda")
    t0 = time.perf_counter()
    per_b = []
    for b in b_grid:
        print(f"\n--- adapt_b = {b:g} ---", flush=True)
        res = run_noise_sweep(g, sigma_grid, seed=seed, adapt_b=float(b), graph=graph, engine=engine)
        target = [r["sigma"] for r in res["rows"] if r["stable_low_rate"]]
        per_b.append(dict(adapt_b=float(b), rows=res["rows"], target_sigma=target,
                          target_state_exists=bool(target)))
        print(f"--- adapt_b = {b:g}: target state at sigma {target if target else 'NONE'} ---", flush=True)
    hits = [x["adapt_b"] for x in per_b if x["target_state_exists"]]
    return dict(
        generated=datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        gpu=torch.cuda.get_device_name(0), n_neurons=graph.n, g=float(g), seed=seed,
        protocol=dict(noise_on_ms=NOISE_ON_MS, noise_off_ms=NOISE_OFF_MS, tail_window_ms=NOISE_TAIL_WINDOW_MS,
                      sigma_grid=[float(s) for s in sigma_grid], b_grid=[float(b) for b in b_grid],
                      ignited_active_frac=IGNITED_ACTIVE_FRAC, low_rate_range_hz=LOW_RATE_RANGE_HZ,
                      stable_ratio_range=STABLE_RATIO_RANGE,
                      adapt_tau_w=_params(g, 1.0).adapt_tau_w),
        base_params={k: v for k, v in _params(g, 0.0).to_dict().items() if k != "adapt_b"},
        per_b=per_b,
        target_state_b=hits,
        default_adapt_b_rule="smallest b on the grid at which some sigma reaches the target state; "
                             "0 (adaptation off) if none does",
        default_adapt_b=(min(hits) if hits else 0.0),
        total_seconds=round(time.perf_counter() - t0, 1),
    )


def markdown_b_table(res: dict) -> str:
    lines = ["| adapt_b | seed | stim-neuron spikes | downstream during stim | post 0-50 / 50-200 / 200-400 ms | "
             "late active | rate (active, Hz) | w max | verdict |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in res["rows"]:
        lines.append(f"| {r['adapt_b']:.3f} | {r['seed']} | {r['stim_neuron_spikes_during_stim']:,} | "
                     f"{r['downstream_during_stim']:,} | "
                     f"{r['post_0_50']:,} / {r['post_50_200']:,} / {r['post_200_400']:,} | "
                     f"{r['late_active_frac']:.3%} | {r['late_rate_active_hz']:.1f} | {r['w_max']:.2f} | "
                     f"{r['verdict']} |")
    return "\n".join(lines)


def markdown_b_noise_table(res: dict) -> str:
    lines = ["| adapt_b | sigma | rate first s (Hz) | rate last s (Hz) | active (last s) | last/first | "
             "tail active (noise off) | w max | verdict |",
             "|---|---|---|---|---|---|---|---|---|"]
    for blk in res["per_b"]:
        for r in blk["rows"]:
            v = "IGNITED" if r["ignited"] else ("**TARGET**" if r["stable_low_rate"] else "-")
            lines.append(f"| {blk['adapt_b']:.3f} | {r['sigma']:.2f} | {r['rate_first_s_hz']:.3f} | "
                         f"{r['rate_last_s_hz']:.3f} | {r['active_frac_last_s']:.3%} | {r['ratio_last_first']:.2f} | "
                         f"{r['tail_active_frac']:.3%} | {r['w_max']:.1f} | {v} |")
    return "\n".join(lines)


# ==============================================================================
# M2d: adaptation as a conductance (docs/m2d-brief.md). Grids, the usable-state
# definition and the voltage-bound check are pre-registered above / here.
# ==============================================================================
def _judge_usable(rate_first: float, rate_last: float, active_last: float,
                  ratio: float, tail_active: float) -> tuple[bool, dict]:
    """The M2d 'usable state', all four conditions required (docs/m2d-brief.md §3).

    Conditions 1-3 are M2b's STABLE_LOW_RATE unchanged; condition 4 is new, because
    M2c satisfied 1-3 with 96-100 % of neurons active, which CLAUDE.md §3.3 calls a
    parameter failure. The active fraction is reported whether or not it passes.
    """
    c = {
        "rate_in_band": LOW_RATE_RANGE_HZ[0] <= rate_last <= LOW_RATE_RANGE_HZ[1],
        "stable": STABLE_RATIO_RANGE[0] <= ratio <= STABLE_RATIO_RANGE[1],
        "not_self_sustained": tail_active < IGNITED_ACTIVE_FRAC,
        "active_frac_ok": active_last <= USABLE_ACTIVE_FRAC_MAX,
    }
    return all(c.values()), c


def run_bg_g_grid(b_g_grid=B_G_GRID, g_grid=G_GRID_ADAPT_G, seeds=SEEDS, graph=None) -> dict:
    """Combined (b_g, g) grid with the M2b stimulus protocol and verdicts.

    No noise here, so with i_ext >= 0 the model guarantees v >= min(E_inh, E_adapt);
    that bound is checked on every run and is the headline acceptance test of M2d.
    """
    graph = graph if graph is not None else Graph.load()
    n = graph.n
    total = STIM_MS + OBSERVE_MS
    n_runs = len(b_g_grid) * len(g_grid) * len(seeds)
    print(f"M2d (b_g x g) grid: {len(b_g_grid)} b_g x {len(g_grid)} g x {len(seeds)} seeds "
          f"= {n_runs} runs x {total} steps = {n_runs * total:,} steps; expected ~3-6 min "
          f"(spike transfer dominates in ignited cells)", flush=True)
    engine = LIFEngine(graph, _params(g_grid[0], adapt_g_b=float(b_g_grid[0])), device="cuda")
    stims = {}
    for seed in seeds:
        stim_idx = np.random.default_rng(seed).choice(n, N_STIM, replace=False)
        stim = torch.zeros(n, device="cuda")
        stim[torch.as_tensor(stim_idx, device="cuda")] = STIM_CURRENT
        stims[seed] = (stim_idx, stim)
    rows = []
    t0 = time.perf_counter()
    for b_g in b_g_grid:
        for g in g_grid:
            p = _params(g, adapt_g_b=float(b_g))
            bound = p.v_lower_bound
            engine.params = p
            for seed in seeds:
                stim_idx, stim = stims[seed]

                def fn(k, buf, stim=stim):
                    if k == 0:
                        buf.copy_(stim)
                    elif k == STIM_MS:
                        buf.zero_()
                engine.reset(seed)
                t, idx, vmin, vmax = _run_tracked(engine, total, fn)
                r = dict(adapt_g_b=float(b_g), g=float(g), seed=seed, total_spikes=int(len(t)),
                         g_a_max=float(engine.g_a.max()), g_a_mean=float(engine.g_a.mean()),
                         v_min=vmin, v_max=vmax, v_lower_bound=bound,
                         v_within_bounds=bool(vmin >= bound and vmax < p.v_thresh),
                         finite=bool(torch.isfinite(engine.v).all() and torch.isfinite(engine.g_a).all()),
                         bins_50ms=np.bincount(t // 50, minlength=total // 50).tolist())
                r.update(judge_g(t, idx, stim_idx, n))
                rows.append(r)
            v = [x["verdict"] for x in rows[-len(seeds):]]
            vb = all(x["v_within_bounds"] for x in rows[-len(seeds):])
            print(f"b_g={b_g:8.5f} g={g:9.3e}  {'/'.join(v):34s}  "
                  f"v[{min(x['v_min'] for x in rows[-len(seeds):]):7.1f}] bound={bound:6.1f} "
                  f"{'OK' if vb else 'VIOLATED'}", flush=True)
    per_cell = []
    for b_g in b_g_grid:
        for g in g_grid:
            sel = [r for r in rows if r["adapt_g_b"] == float(b_g) and r["g"] == float(g)]
            vs = [r["verdict"] for r in sel]
            per_cell.append(dict(adapt_g_b=float(b_g), g=float(g), verdicts=vs,
                                 all_responsive=all(x == "RESPONSIVE" for x in vs),
                                 any_ignited=any(x == "IGNITED" for x in vs),
                                 all_silent=all(x == "SILENT" for x in vs),
                                 v_min=min(r["v_min"] for r in sel),
                                 v_within_bounds=all(r["v_within_bounds"] for r in sel)))
    violations = [c for c in per_cell if not c["v_within_bounds"]]
    return dict(
        generated=datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        gpu=torch.cuda.get_device_name(0), n_neurons=n, n_edges=graph.m,
        protocol=dict(dt_ms=DT_MS, n_stim=N_STIM, seeds=list(seeds), stim_current=STIM_CURRENT,
                      stim_ms=STIM_MS, observe_ms=OBSERVE_MS, late_window_ms=LATE_WINDOW,
                      ignited_active_frac=IGNITED_ACTIVE_FRAC,
                      b_g_grid=[float(b) for b in b_g_grid], g_grid=[float(g) for g in g_grid],
                      adapt_tau_a=_params(g_grid[0], adapt_g_b=1.0).adapt_tau_a,
                      E_adapt=_params(g_grid[0], adapt_g_b=1.0).E_adapt),
        base_params={k: v for k, v in _params(g_grid[0], adapt_g_b=0.0).to_dict().items()
                     if k not in ("g", "adapt_g_b")},
        rows=rows, per_cell=per_cell,
        responsive_cells=[(c["adapt_g_b"], c["g"]) for c in per_cell if c["all_responsive"]],
        v_bound_violations=[(c["adapt_g_b"], c["g"], c["v_min"]) for c in violations],
        v_min_overall=min(r["v_min"] for r in rows),
        total_seconds=round(time.perf_counter() - t0, 1),
    )


def run_adapt_g_noise_sweep(cells, sigma_grid=NOISE_SIGMA_GRID, seed: int = 0, graph=None) -> dict:
    """Noise sweep at each (b_g, g) cell that was RESPONSIVE for all three seeds."""
    graph = graph if graph is not None else Graph.load()
    n = graph.n
    on, off = NOISE_ON_MS, NOISE_OFF_MS
    steps = len(cells) * len(sigma_grid) * (on + off)
    print(f"M2d noise sweep: {len(cells)} cells x {len(sigma_grid)} sigma x {on + off} steps "
          f"= {steps:,} steps; expected ~{max(1, steps // 400_000)}-{max(2, steps // 150_000)} min",
          flush=True)
    engine = LIFEngine(graph, _params(cells[0][1], adapt_g_b=cells[0][0]), device="cuda")
    zero = lambda k, buf: None  # noqa: E731
    out = []
    t0 = time.perf_counter()
    for b_g, g in cells:
        print(f"\n--- b_g = {b_g:g}, g = {g:.4e} ---", flush=True)
        rows = []
        for sigma in sigma_grid:
            p = _params(g, adapt_g_b=b_g, noise_sigma=float(sigma))
            engine.params = p
            engine.reset(seed)
            engine.i_ext.zero_()
            t1, i1, vmin1, vmax1 = _run_tracked(engine, on, zero)
            engine.params = _params(g, adapt_g_b=b_g)      # noise off, state kept
            t2, i2, vmin2, vmax2 = _run_tracked(engine, off, zero)
            first, last = (t1 < 1000), (t1 >= on - 1000)
            rate_first = int(first.sum()) / n
            rate_last = int(last.sum()) / n
            active_last = int(np.unique(i1[last]).size) / n
            tail = t2 >= off - NOISE_TAIL_WINDOW_MS
            tail_active = int(np.unique(i2[tail]).size) / n
            ratio = (rate_last / rate_first) if rate_first > 0 else (float("inf") if rate_last else 0.0)
            usable, cond = _judge_usable(rate_first, rate_last, active_last, ratio, tail_active)
            r = dict(adapt_g_b=float(b_g), g=float(g), sigma=float(sigma),
                     rate_first_s_hz=rate_first, rate_last_s_hz=rate_last,
                     active_frac_last_s=active_last, ratio_last_first=ratio,
                     tail_active_frac=tail_active, usable=usable, conditions=cond,
                     ignited=tail_active >= IGNITED_ACTIVE_FRAC,
                     g_a_max=float(engine.g_a.max()),
                     v_min=min(vmin1, vmin2), v_max=max(vmax1, vmax2),
                     v_lower_bound=p.v_lower_bound,
                     v_within_bounds_noise_free_part=bool(min(vmin1, vmin2) >= p.v_lower_bound))
            rows.append(r)
            print(f"sigma={sigma:7.2f}  rate {rate_first:7.3f}->{rate_last:7.3f} Hz  "
                  f"active={active_last:8.3%}  ratio={ratio:6.2f}  tail={tail_active:7.3%}  "
                  f"v[{r['v_min']:7.1f}]  {'USABLE' if usable else '-'}"
                  f"{'' if usable else '  fail:' + ','.join(k for k, v in cond.items() if not v)}",
                  flush=True)
        hits = [r["sigma"] for r in rows if r["usable"]]
        out.append(dict(adapt_g_b=float(b_g), g=float(g), rows=rows,
                        usable_sigma=hits, usable_exists=bool(hits)))
    usable_cells = [(c["adapt_g_b"], c["g"]) for c in out if c["usable_exists"]]
    default_b_g = min((b for b, _ in usable_cells), default=None)
    gs = [g for b, g in usable_cells if b == default_b_g] if default_b_g is not None else []
    default_g = float(np.sqrt(min(gs) * max(gs))) if gs else None
    return dict(
        generated=datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        gpu=torch.cuda.get_device_name(0), n_neurons=n, seed=seed,
        protocol=dict(noise_on_ms=on, noise_off_ms=off, tail_window_ms=NOISE_TAIL_WINDOW_MS,
                      sigma_grid=[float(s) for s in sigma_grid],
                      cells=[[float(b), float(g)] for b, g in cells],
                      usable_state=dict(rate_hz=LOW_RATE_RANGE_HZ, ratio=STABLE_RATIO_RANGE,
                                        tail_active_below=IGNITED_ACTIVE_FRAC,
                                        active_frac_max=USABLE_ACTIVE_FRAC_MAX)),
        per_cell=out, usable_cells=[[b, g] for b, g in usable_cells],
        default_rule="DEFAULT_ADAPT_G_B = smallest b_g with a usable state; DEFAULT_G_WITH_ADAPT = "
                     "geometric mean of the g values usable at that b_g",
        default_adapt_g_b=default_b_g, default_g_with_adapt=default_g,
        total_seconds=round(time.perf_counter() - t0, 1),
    )


def markdown_bg_g_table(res: dict) -> str:
    """Verdict matrix: rows = b_g, cols = g. S/R/I = all-seed SILENT/RESPONSIVE/mixed-ignited."""
    gs = res["protocol"]["g_grid"]
    lines = ["| b_g \\ g | " + " | ".join(f"{g:.3e}" for g in gs) + " |",
             "|---" * (len(gs) + 1) + "|"]
    for b_g in res["protocol"]["b_g_grid"]:
        cells = []
        for g in gs:
            c = next(c for c in res["per_cell"] if c["adapt_g_b"] == b_g and c["g"] == g)
            cells.append("**R**" if c["all_responsive"] else
                         ("I" if c["any_ignited"] else ("S" if c["all_silent"] else "·")))
        lines.append(f"| {b_g:.5f} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def markdown_adapt_g_noise_table(res: dict) -> str:
    lines = ["| b_g | g | sigma | rate first s | rate last s | **active last s** | last/first | "
             "tail active | v min | usable |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for c in res["per_cell"]:
        for r in c["rows"]:
            lines.append(
                f"| {c['adapt_g_b']:.5f} | {c['g']:.3e} | {r['sigma']:.2f} | {r['rate_first_s_hz']:.3f} | "
                f"{r['rate_last_s_hz']:.3f} | **{r['active_frac_last_s']:.2%}** | "
                f"{r['ratio_last_first']:.2f} | {r['tail_active_frac']:.3%} | {r['v_min']:.1f} | "
                f"{'**YES**' if r['usable'] else '-'} |")
    return "\n".join(lines)


def run_robustness(b_g: float, g: float, usable_sigma, seeds=ROBUSTNESS_SEEDS,
                   sigma_grid=NOISE_SIGMA_GRID, graph=None) -> dict:
    """Re-run one selected (b_g, g) cell under other noise seeds (docs/m2e-brief.md §4).

    Descriptive only: this does NOT choose a cell and does NOT look for a substitute if
    it fails. ``usable_sigma`` is the set of sigma that were usable with the primary
    seed; the cell is called robust only if every one of them is usable under every seed.
    """
    graph = graph if graph is not None else Graph.load()
    n = graph.n
    on, off = NOISE_ON_MS, NOISE_OFF_MS
    print(f"M2e robustness: b_g={b_g:g}, g={g:.4e}, seeds {list(seeds)} x {len(sigma_grid)} sigma "
          f"= {len(seeds) * len(sigma_grid)} runs; target sigma {[round(s, 2) for s in usable_sigma]}",
          flush=True)
    engine = LIFEngine(graph, _params(g, adapt_g_b=b_g), device="cuda")
    zero = lambda k, buf: None  # noqa: E731
    per_seed = []
    for seed in seeds:
        rows = []
        for sigma in sigma_grid:
            p = _params(g, adapt_g_b=b_g, noise_sigma=float(sigma))
            engine.params = p
            engine.reset(seed)
            engine.i_ext.zero_()
            t1, i1, vmin1, _ = _run_tracked(engine, on, zero)
            engine.params = _params(g, adapt_g_b=b_g)
            t2, i2, vmin2, _ = _run_tracked(engine, off, zero)
            first, last = (t1 < 1000), (t1 >= on - 1000)
            rate_first = int(first.sum()) / n
            rate_last = int(last.sum()) / n
            active_last = int(np.unique(i1[last]).size) / n
            tail_active = int(np.unique(i2[t2 >= off - NOISE_TAIL_WINDOW_MS]).size) / n
            ratio = (rate_last / rate_first) if rate_first > 0 else (float("inf") if rate_last else 0.0)
            usable, cond = _judge_usable(rate_first, rate_last, active_last, ratio, tail_active)
            rows.append(dict(seed=seed, sigma=float(sigma), rate_first_s_hz=rate_first,
                             rate_last_s_hz=rate_last, active_frac_last_s=active_last,
                             ratio_last_first=ratio, tail_active_frac=tail_active,
                             usable=usable, conditions=cond, v_min=min(vmin1, vmin2),
                             v_lower_bound=p.v_lower_bound))
            print(f"seed={seed} sigma={sigma:7.2f}  rate {rate_first:7.3f}->{rate_last:7.3f} Hz  "
                  f"active={active_last:8.3%}  ratio={ratio:6.2f}  tail={tail_active:7.3%}  "
                  f"v[{min(vmin1, vmin2):7.1f}]  {'USABLE' if usable else '-'}", flush=True)
        hits = [r["sigma"] for r in rows if r["usable"]]
        ok = all(any(abs(h - s) < 1e-9 for h in hits) for s in usable_sigma)
        per_seed.append(dict(seed=seed, rows=rows, usable_sigma=hits,
                             reproduces_primary_sigma=ok))
        print(f"--- seed {seed}: usable at {[round(h, 2) for h in hits] or 'NONE'}; "
              f"reproduces the primary sigma: {ok} ---", flush=True)
    robust = all(x["reproduces_primary_sigma"] for x in per_seed)
    return dict(
        generated=datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        gpu=torch.cuda.get_device_name(0), n_neurons=n,
        cell=dict(adapt_g_b=float(b_g), g=float(g)),
        primary_usable_sigma=[float(s) for s in usable_sigma], seeds=list(seeds),
        protocol=dict(noise_on_ms=on, noise_off_ms=off, tail_window_ms=NOISE_TAIL_WINDOW_MS,
                      sigma_grid=[float(s) for s in sigma_grid],
                      usable_state=dict(rate_hz=LOW_RATE_RANGE_HZ, ratio=STABLE_RATIO_RANGE,
                                        tail_active_below=IGNITED_ACTIVE_FRAC,
                                        active_frac_max=USABLE_ACTIVE_FRAC_MAX)),
        per_seed=per_seed, robust=robust,
        note="descriptive check; a failure is reported as 'not robust' and no substitute "
             "cell is selected (docs/m2e-brief.md §4)",
    )


def usable_margins(row: dict) -> dict:
    """Margin on each of the four conditions for one run (descriptive metric, m2e §5)."""
    return {
        "rate_lower": row["rate_last_s_hz"] - LOW_RATE_RANGE_HZ[0],
        "rate_upper": LOW_RATE_RANGE_HZ[1] - row["rate_last_s_hz"],
        "ratio_lower": row["ratio_last_first"] - STABLE_RATIO_RANGE[0],
        "ratio_upper": STABLE_RATIO_RANGE[1] - row["ratio_last_first"],
        "tail_headroom": IGNITED_ACTIVE_FRAC - row["tail_active_frac"],
        "active_frac_headroom": USABLE_ACTIVE_FRAC_MAX - row["active_frac_last_s"],
    }
