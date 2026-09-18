"""M2i: is there a like-for-like control for M2h at all?

Pre-registration: docs/m2i-brief.md. **A check. It cannot change M2h's conclusion and is
not run to.**

M2h wrote "adaptation attenuates hop 2 by 5-7x in rate mode", but its no-adaptation
reference used g = 3.162e-4 while M2h itself used g = 5.36539e-4, so the effect of
adaptation and the effect of the gain change are mixed. This runs the missing arm:
**adaptation off, everything else exactly as in M2h**, including g.

It may turn out that the arm does not exist. M2b found that without adaptation the
network ignites from sigma >= 22, and M2h used sigma 23.3-29.13. If the no-adaptation
network ignites at those sigma then "the same conditions with adaptation switched off"
is not a thing that exists, and that is the answer.

sigma = 0 is added as a fourth point: same gain, no noise, no adaptation, which isolates
the effect of the gain change alone against M3b (g = 3.162e-4).

NOTE on the ignition test. M2h redefined ignition as (stimulus - control excess >= 3 SEM)
AND (stimulus run >= 90 % active). That fixed a false positive - a background state being
read as a runaway. It can produce a false NEGATIVE here: if the control itself is ignited,
the excess is ~0 and nothing is flagged however saturated the network is. So this module
records the redefined verdict AND the absolute state (active fraction, network mean rate)
of the control runs, which is what actually answers "does it ignite".

Usage: python -m flysim.apps.m2i_control [--modes rate,phase-lock] [--sigmas 29.13,26.21,23.3,0]
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
from ..probe.probe import window_mean_rate
from ..sensory import JOAdapter, click_train, silence
from .m2g_adapt_click import G, E_ADAPT, TAU_A, A_IN, K_MIN
from .m2h_power import (STIM, STIM_WINDOW_MS, TOTAL_MS, SIGMA_MULTIPLIER, N_SEEDS,
                        IGNITION_ABSOLUTE_ACTIVE, judge, NO_ADAPT_SIGNAL_HZ)
from .m3_click import MODES, BIN_MS

SIGMAS = (29.13, 26.21, 23.30, 0.0)      # M2h's three, plus the no-noise control
ADAPT_G_B = 0.0                          # the whole point: adaptation off
# M2b, no adaptation, g = 3.162e-4: sigma <= 20 stays silent, sigma >= 22 ignites
# (docs/m2b-report.md §6). M2h ran at sigma 23.3-29.13 with g = 5.36539e-4, a HIGHER gain.
M2B_IGNITION_SIGMA = 22.0
# M3b, no adaptation, no noise, g = 3.162e-4, rate a_in 640 (docs/m3b-report.md).
M3B_G = 3.162e-4

RESULTS_JSON = ROOT / "data-provenance" / "m2i-results.json"


def log(msg: str) -> None:
    print(f"[m2i] {msg}", flush=True)


def one_run(engine, adapter, n_steps, dt_ms, sets, seed, n_neurons) -> dict:
    """As m2h_power.one_run, plus the absolute-state diagnostics this round needs."""
    adapter.prepare(dt_ms, n_steps)
    engine.reset(seed)
    t_step, idx = engine.run(n_steps, adapter.inject, record=True)
    rt = bin_spikes(t_step, idx, dt_ms, BIN_MS, int(round(TOTAL_MS / BIN_MS)), sets)
    w0, w1 = STIM_WINDOW_MS
    rates = {name: window_mean_rate(rt, name, w0, w1) for name in rt.names}
    t_ms = np.asarray(t_step) * dt_ms
    late = t_ms >= w1 + 100.0
    late_active = int(len(np.unique(np.asarray(idx)[late])))
    in_win = (t_ms >= w0) & (t_ms < w1)
    return dict(seed=seed, n_spikes=int(rt.n_spikes),
                active_fraction=rt.active_neurons / n_neurons,
                late_active_fraction=late_active / n_neurons,
                # absolute state: network mean rate over ALL neurons in the window
                network_rate_hz=float(in_win.sum()) / (n_neurons * (w1 - w0) / 1000.0),
                window_rate_hz=rates, v_min=float(engine.v.min()))


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--sigmas", default=",".join(str(s) for s in SIGMAS))
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    modes = [m for m in a.modes.split(",") if m]
    sigmas = [float(s) for s in a.sigmas.split(",") if s or s == "0"]
    seeds = list(range(a.seeds))

    graph = Graph.load()
    roi = load_roi(graph)
    n_sites = roi["n_sites"].to_numpy()
    sets = build_probe_sets(graph, roi, k_min=K_MIN)
    stim = click_train(**STIM)
    quiet = silence(TOTAL_MS, stim.fs)
    n_runs = len(modes) * len(sigmas) * 2 * len(seeds)
    steps = (len(sigmas) * 2 * len(seeds)
             * sum(int(round(TOTAL_MS / MODES[m])) for m in modes))
    log(f"{n_runs} runs, {steps:,} steps; M2h measured ~7,000 steps/s here, so ~"
        f"{steps / 7000 / 60:.0f}-{steps / 3500 / 60:.0f} min.")
    log(f"adaptation OFF, g = {G:g} (M2h's value), a_in {A_IN}, sigma {sigmas}")

    R = dict(generated_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
             purpose="does a like-for-like no-adaptation control exist for M2h?",
             adapt_g_b=ADAPT_G_B, g=G, a_in=A_IN, sigmas=sigmas, seeds=seeds,
             stimulus=stim.describe(), stim_window_ms=list(STIM_WINDOW_MS), total_ms=TOTAL_MS,
             statistic=f"difference of means vs {SIGMA_MULTIPLIER} x SEM of the control mean",
             reference=dict(m2b_ignition_sigma=M2B_IGNITION_SIGMA, m2b_g=3.162e-4,
                            m3b_g=M3B_G, no_adapt_signal_hz=NO_ADAPT_SIGNAL_HZ),
             probe_sets=dict(sizes=sets.sizes()), cells=[])

    t0 = time.time()
    for sigma in sigmas:
        for mode in modes:
            dt = MODES[mode]
            n_steps = int(round(TOTAL_MS / dt))
            p = EngineParams(dt=dt, synapse="conductance", g=G, noise_sigma=sigma,
                             adapt_g_b=ADAPT_G_B, adapt_tau_a=TAU_A, E_adapt=E_ADAPT)
            engine = LIFEngine(graph, p, device=a.device)
            log(f"=== sigma={sigma:g} {mode}: {2 * len(seeds)} runs x {n_steps} steps")
            tc = time.time()
            ctrl = [one_run(engine, JOAdapter(graph, n_sites, quiet, mode, a_in=A_IN[mode],
                                              device=a.device), n_steps, dt, sets, s, graph.n)
                    for s in seeds]
            stim_runs = [one_run(engine, JOAdapter(graph, n_sites, stim, mode, a_in=A_IN[mode],
                                                   device=a.device), n_steps, dt, sets, s, graph.n)
                         for s in seeds]
            j = judge(stim_runs, ctrl, list(sets.names))
            # absolute state of the CONTROL runs - what "does it ignite" actually asks
            ca = np.array([r["active_fraction"] for r in ctrl])
            cr = np.array([r["network_rate_hz"] for r in ctrl])
            absolute = dict(ctrl_active_mean=float(ca.mean()), ctrl_active_max=float(ca.max()),
                            ctrl_network_rate_hz_mean=float(cr.mean()),
                            ctrl_late_active_mean=float(np.mean([r["late_active_fraction"] for r in ctrl])),
                            stim_active_mean=float(np.mean([r["active_fraction"] for r in stim_runs])),
                            note="the redefined ignition test compares stimulus with control and can "
                                 "miss a runaway that the control shares; these absolutes cannot")
            R["cells"].append(dict(sigma=sigma, mode=mode, dt_ms=dt, n_steps=n_steps,
                                   engine_params=p.to_dict(), control_runs=ctrl,
                                   stimulus_runs=stim_runs, judgement=j, absolute=absolute))
            ps = j["per_set"]
            log(f"  {time.time() - tc:.0f}s  control active {absolute['ctrl_active_mean']:.1%}, "
                f"network rate {absolute['ctrl_network_rate_hz_mean']:.3f} Hz  |  "
                f"WED diff L/R {ps['WED_L']['diff_hz']:+.3f}/{ps['WED_R']['diff_hz']:+.3f} "
                f"(floor {ps['WED_L']['threshold_hz']:.3f}/{ps['WED_R']['threshold_hz']:.3f})  "
                f"hop2 {[b for b in ('SAD', 'WED', 'AMMCtype') if j['hop'][b]['any_reached']] or 'none'}  "
                f"pC1 {j['pc1_reached']}  ignited(redef) {j['ignition']['ignited']}")
            del engine
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    R["wall_total_s"] = round(time.time() - t0, 1)
    RESULTS_JSON.write_text(json.dumps(R, indent=1, default=_json_default) + "\n")
    log(f"done in {R['wall_total_s']} s -> {RESULTS_JSON.relative_to(ROOT)}")
    _summary(R)
    return 0


def _summary(R: dict) -> None:
    print("\n| sigma | mode | control active | control rate Hz | WED_L diff | WED_R diff | "
          "hop-2 | pC1 | ignited(redef) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for c in R["cells"]:
        j, ab, ps = c["judgement"], c["absolute"], c["judgement"]["per_set"]
        h = [b for b in ("SAD", "WED", "AMMCtype") if j["hop"][b]["any_reached"]]
        print(f"| {c['sigma']:g} | {c['mode']} | {ab['ctrl_active_mean']:.1%} | "
              f"{ab['ctrl_network_rate_hz_mean']:.3f} | {ps['WED_L']['diff_hz']:+.3f} | "
              f"{ps['WED_R']['diff_hz']:+.3f} | {', '.join(h) or 'none'} | {j['pc1_reached']} | "
              f"{j['ignition']['ignited']} |")


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
