"""M2g: does the adaptation background state let the click-train signal travel further?

Pre-registration: docs/m2g-brief.md, written before this ran. The project added
adaptation because the signal stopped at hop 2 (docs/m4b-report.md). M2f found one
robust background state (b_g = 8.0) while the largest-margin candidate is b_g = 1.51572;
statistics could not separate them, so both are run on the auditory pathway and the
question is the single one that matters: **does the signal get further?**

Two operating points x two sensory modes x (stimulus, silence control) x three noise
seeds = 24 runs. Nothing is swept and nothing is re-selected: ``a_in`` keeps the values
M3 chose, ``g`` and ``noise_sigma`` keep the values M2f fixed.

The control is no longer silent. Noise is on at sigma = 29.13, so every probe set has a
background rate, and a stimulus response is only meaningful as a **difference from a
control run at the same operating point, same noise, same seed**. Hop-3 arrival has to
clear three control standard deviations, measured across the three control seeds.

Usage: python -m flysim.apps.m2g_adapt_click [--points A,B] [--modes rate,phase-lock]
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import torch

from ..data.download import ROOT
from ..engine import EngineParams, LIFEngine
from ..graph import Graph
from ..probe import bin_spikes, build_probe_sets, load_roi
from ..probe.probe import window_mean_rate, window_spikes
from ..sensory import JOAdapter, click_train, silence
# Protocol constants are imported, not copied, so the stimulus is provably the M3 one.
from .m3_click import STIM, STIM_WINDOW_MS, TOTAL_MS, MODES, BIN_MS, CRITERIA as M3_CRITERIA

# ---- operating points (fixed before the run; docs/m2g-brief.md §1) ------------------
OPERATING_POINTS = {
    "A": dict(adapt_g_b=8.0,
              why="the value the M2f selection rule chose; out-of-sample robust 5/5"),
    "B": dict(adapt_g_b=1.51572,
              why="the candidate with the largest minimum margin (0.0599 vs 0.0419)"),
}
G = 5.36539e-4                # M2f DEFAULT_G_WITH_ADAPT; not re-selected
NOISE_SIGMA = 29.13           # the only sigma at which a usable state exists (M2e/M2f)
E_ADAPT = -75.0
TAU_A = 100.0
A_IN = {"rate": 640.0, "phase-lock": 2560.0}      # M3's selections, reused verbatim
SEEDS = (0, 1, 2)             # noise seeds, stimulus and control alike

# ---- judgement (fixed before the run; docs/m2g-brief.md §4) -------------------------
TRANSFER_MIN_DIFF_HZ = 1.0    # JO_post: (stimulus - control) per side
HOP3_SIGMA_MULTIPLIER = 3.0   # (stimulus - control) must exceed this x the control sd
HOP3_BASES = ("SAD", "WED", "AMMCtype", "pC1")
IGNITION_AFTER_OFFSET_MS = M3_CRITERIA["ignition_after_offset_ms"]      # 100 ms
IGNITION_ACTIVE_FRACTION = M3_CRITERIA["ignition_active_fraction"]      # 1 %
K_MIN = M3_CRITERIA["k_min"]

RESULTS_JSON = ROOT / "data-provenance" / "m2g-results.json"
REPORT_MD = ROOT / "docs" / "m2g-report.md"


def log(msg: str) -> None:
    print(f"[m2g] {msg}", flush=True)


def _params(mode: str, adapt_g_b: float) -> EngineParams:
    return EngineParams(dt=MODES[mode], synapse="conductance", g=G, noise_sigma=NOISE_SIGMA,
                        adapt_g_b=adapt_g_b, adapt_tau_a=TAU_A, E_adapt=E_ADAPT)


def one_run(engine, adapter, n_steps: int, dt_ms: float, sets, seed: int, n_neurons: int,
            label: str) -> dict:
    """One run; returns per-set window rates plus the ignition check."""
    adapter.prepare(dt_ms, n_steps)
    engine.reset(seed)
    t0 = time.time()
    t_step, idx = engine.run(n_steps, adapter.inject, record=True)
    wall = time.time() - t0
    rt = bin_spikes(t_step, idx, dt_ms, BIN_MS, int(round(TOTAL_MS / BIN_MS)), sets)
    w0, w1 = STIM_WINDOW_MS
    rates = {name: window_mean_rate(rt, name, w0, w1) for name in rt.names}
    spikes = {name: window_spikes(rt, name, w0, w1) for name in rt.names}
    t_ms = np.asarray(t_step) * dt_ms
    late = t_ms >= w1 + IGNITION_AFTER_OFFSET_MS
    late_active = int(len(np.unique(np.asarray(idx)[late])))
    out = dict(seed=seed, n_spikes=int(rt.n_spikes), active_neurons=int(rt.active_neurons),
               active_fraction=rt.active_neurons / n_neurons,
               window_rate_hz=rates, window_spikes=spikes,
               late_active_neurons=late_active, late_active_fraction=late_active / n_neurons,
               ignited=bool(late_active / n_neurons >= IGNITION_ACTIVE_FRACTION),
               v_min=float(engine.v.min()), g_a_max=float(engine.g_a.max()), wall_s=round(wall, 2))
    log(f"  {label} seed={seed}: {rt.n_spikes:,} spikes, active {out['active_fraction']:.2%}, "
        f"late {out['late_active_fraction']:.3%}{' IGNITED' if out['ignited'] else ''}, {wall:.1f}s")
    return out


def judge(stim_runs: list[dict], ctrl_runs: list[dict], set_names: list[str]) -> dict:
    """Stimulus minus control, per set, with the control spread across seeds."""
    per_set = {}
    for name in set_names:
        s = np.array([r["window_rate_hz"][name] for r in stim_runs], dtype=float)
        c = np.array([r["window_rate_hz"][name] for r in ctrl_runs], dtype=float)
        sd = float(c.std(ddof=1)) if len(c) > 1 else 0.0
        diff = float(s.mean() - c.mean())
        per_set[name] = dict(
            stim_mean_hz=float(s.mean()), stim_min_hz=float(s.min()), stim_max_hz=float(s.max()),
            ctrl_mean_hz=float(c.mean()), ctrl_min_hz=float(c.min()), ctrl_max_hz=float(c.max()),
            ctrl_sd_hz=sd, diff_hz=diff,
            ratio=(float(s.mean() / c.mean()) if c.mean() > 0 else None),
            threshold_hz=HOP3_SIGMA_MULTIPLIER * sd,
            exceeds_3sd=bool(diff > HOP3_SIGMA_MULTIPLIER * sd) if sd > 0 else bool(diff > 0),
        )
    transfer = {}
    for side in ("L", "R"):
        k = f"JO_post_{side}"
        transfer[side] = dict(diff_hz=per_set[k]["diff_hz"],
                              passes=bool(per_set[k]["diff_hz"] >= TRANSFER_MIN_DIFF_HZ))
    hop3 = {}
    for base in HOP3_BASES:
        members = [n for n in set_names if n == base or n.startswith(f"{base}_")]
        reached = [n for n in members if per_set[n]["exceeds_3sd"]]
        hop3[base] = dict(members=members, reached=reached, any_reached=bool(reached),
                          best_diff_hz=max((per_set[n]["diff_hz"] for n in members), default=0.0))
    # Ignition, applied exactly as pre-registered: late-window active fraction >= 1 %.
    # NOTE (reported, not worked around): this criterion was written for a silent
    # background. At these operating points the background state is 25-28 % active by
    # construction, so every run - including the silence controls - exceeds 1 % and the
    # test cannot distinguish a stimulus-driven runaway from the background it was set up
    # to produce. The companion measure below applies the brief's own difference logic to
    # the same window; neither is used to change any other verdict.
    ls = np.array([r["late_active_fraction"] for r in stim_runs], dtype=float)
    lc = np.array([r["late_active_fraction"] for r in ctrl_runs], dtype=float)
    lsd = float(lc.std(ddof=1)) if len(lc) > 1 else 0.0
    ignition_excess = dict(
        stim_mean=float(ls.mean()), ctrl_mean=float(lc.mean()),
        diff=float(ls.mean() - lc.mean()), ctrl_sd=lsd,
        threshold=HOP3_SIGMA_MULTIPLIER * lsd,
        exceeds_3sd=bool((ls.mean() - lc.mean()) > HOP3_SIGMA_MULTIPLIER * lsd) if lsd > 0
                    else bool(ls.mean() > lc.mean()),
        note="stimulus-driven excess over the control's own late-window activity; the "
             "absolute >=1 % rule flags every run here because the background state is "
             "25-28 % active by design")

    return dict(per_set=per_set, transfer=transfer, ignition_excess=ignition_excess,
                transfer_pass=bool(transfer["L"]["passes"] and transfer["R"]["passes"]),
                hop3=hop3,
                hop3_any=bool(any(h["any_reached"] for h in hop3.values())),
                pc1_reached=bool(hop3["pC1"]["any_reached"]),
                ignited_runs=int(sum(r["ignited"] for r in stim_runs + ctrl_runs)),
                ignited_control_runs=int(sum(r["ignited"] for r in ctrl_runs)),
                n_runs=len(stim_runs) + len(ctrl_runs))


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", default=",".join(OPERATING_POINTS))
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    points = [p for p in a.points.split(",") if p]
    modes = [m for m in a.modes.split(",") if m]

    graph = Graph.load()
    roi = load_roi(graph)
    n_sites = roi["n_sites"].to_numpy()
    sets = build_probe_sets(graph, roi, k_min=K_MIN)
    stim = click_train(**STIM)
    quiet = silence(TOTAL_MS, stim.fs)
    n_runs = len(points) * len(modes) * 2 * len(SEEDS)
    steps = sum(int(round(TOTAL_MS / MODES[m])) for m in modes) * len(points) * 2 * len(SEEDS)
    log(f"{n_runs} runs, {steps:,} steps total. On an un-capped GPU that is ~1 min; this machine is "
        f"power-capped (docs/m2f-report.md §8), so expect roughly 5-15 min.")
    log(f"probe sets: {sets.sizes()}")

    results = dict(
        generated_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        question="does the adaptation background state let the signal travel further than hop 2?",
        operating_points=OPERATING_POINTS, g=G, noise_sigma=NOISE_SIGMA, tau_a=TAU_A,
        E_adapt=E_ADAPT, a_in=A_IN, seeds=list(SEEDS),
        stimulus=stim.describe(), stim_window_ms=list(STIM_WINDOW_MS),
        criteria=dict(transfer_min_diff_hz=TRANSFER_MIN_DIFF_HZ,
                      hop3_sigma_multiplier=HOP3_SIGMA_MULTIPLIER, hop3_bases=list(HOP3_BASES),
                      ignition_after_offset_ms=IGNITION_AFTER_OFFSET_MS,
                      ignition_active_fraction=IGNITION_ACTIVE_FRACTION, k_min=K_MIN),
        probe_sets=dict(sizes=sets.sizes()), points={})

    t_all = time.time()
    for pname in points:
        b_g = OPERATING_POINTS[pname]["adapt_g_b"]
        results["points"][pname] = dict(adapt_g_b=b_g, why=OPERATING_POINTS[pname]["why"], modes={})
        for mode in modes:
            dt = MODES[mode]
            n_steps = int(round(TOTAL_MS / dt))
            p = _params(mode, b_g)
            engine = LIFEngine(graph, p, device=a.device)
            log(f"=== point {pname} (b_g={b_g:g}), mode {mode}: dt={dt} ms, {n_steps} steps, "
                f"a_in={A_IN[mode]:g}")
            ctrl_runs, stim_runs = [], []
            for seed in SEEDS:
                ad = JOAdapter(graph, n_sites, quiet, mode, a_in=A_IN[mode], device=a.device)
                ctrl_runs.append(one_run(engine, ad, n_steps, dt, sets, seed, graph.n, "control"))
            for seed in SEEDS:
                ad = JOAdapter(graph, n_sites, stim, mode, a_in=A_IN[mode], device=a.device)
                stim_runs.append(one_run(engine, ad, n_steps, dt, sets, seed, graph.n, "stimulus"))
            j = judge(stim_runs, ctrl_runs, list(sets.names))
            results["points"][pname]["modes"][mode] = dict(
                dt_ms=dt, n_steps=n_steps, a_in=A_IN[mode], engine_params=p.to_dict(),
                control_runs=ctrl_runs, stimulus_runs=stim_runs, judgement=j)
            log(f"  -> transfer L/R diff {j['transfer']['L']['diff_hz']:+.3f}/"
                f"{j['transfer']['R']['diff_hz']:+.3f} Hz (pass={j['transfer_pass']}), "
                f"hop3 reached: {[b for b, h in j['hop3'].items() if h['any_reached']] or 'none'}, "
                f"pC1={j['pc1_reached']}, ignited(absolute rule)={j['ignited_runs']}/{j['n_runs']} "
                f"of which controls {j['ignited_control_runs']}, "
                f"late-window excess over control {j['ignition_excess']['diff']:+.4%} "
                f"(3sd {j['ignition_excess']['threshold']:.4%})")
            del engine
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    results["wall_total_s"] = round(time.time() - t_all, 1)
    RESULTS_JSON.write_text(json.dumps(results, indent=1, default=_json_default) + "\n")
    log(f"done in {results['wall_total_s']} s -> {RESULTS_JSON.relative_to(ROOT)}")
    _summary(results)
    return 0


def _summary(R: dict) -> None:
    print()
    print("| point | b_g | mode | JO_post L diff | JO_post R diff | transfer | hop-3 reached | pC1 | ignited |")
    print("|---|---|---|---|---|---|---|---|---|")
    for pname, P in R["points"].items():
        for mode, M in P["modes"].items():
            j = M["judgement"]
            reached = [b for b, h in j["hop3"].items() if h["any_reached"]]
            print(f"| {pname} | {P['adapt_g_b']:g} | {mode} | {j['transfer']['L']['diff_hz']:+.3f} | "
                  f"{j['transfer']['R']['diff_hz']:+.3f} | {j['transfer_pass']} | "
                  f"{', '.join(reached) or 'none'} | {j['pc1_reached']} | {j['ignited_runs']} |")


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(type(o))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
