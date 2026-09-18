"""M2h: close the hop-2 ambiguity M2g left open, by raising statistical power only.

Pre-registration: docs/m2h-brief.md, written before this ran. **Last round of this branch.**

M2g found no hop-2 arrival above its detection floor, but that floor (0.13-0.30 Hz)
sat at or above the hop-2 signal that was visible without adaptation (0.05-0.17 Hz),
so it could not separate "adaptation did not help" from "the signal is still arriving
and is now buried". Nothing about the model or the operating points changes here. Three
things change, all of them power:

  seeds            3  -> 10
  stimulus window  500 -> 2000 ms
  test statistic   3 x per-run sd -> 3 x standard error of the mean (sd / sqrt(10))

and sigma is looked at in three places (29.13, 26.21, 23.30), all inside the region
M2f measured as background-preserving, because a lower sigma lowers the floor further.

Predicted floor (brief, before the run): about 6.1x lower, e.g. SAD_L 0.143 -> 0.023 and
WED_R 0.241 -> 0.038, both below the no-adaptation signal. The ACHIEVED floor is
tabulated either way, and if it comes out higher than predicted that is reported first.

Ignition is redefined here and from now on (brief §6): an absolute active fraction is
meaningless against a 25-28 % background - in M2g the rule flagged all 12 silence
controls. A run is ignited only if the stimulus-minus-control excess clears 3 SEM AND
the stimulus run itself is at least 90 % active.

Usage: python -m flysim.apps.m2h_power [--points A,B] [--modes rate,phase-lock]
                                       [--sigmas 29.13,26.21,23.30] [--seeds 10]
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
from .m2g_adapt_click import OPERATING_POINTS, G, E_ADAPT, TAU_A, A_IN, K_MIN, HOP3_BASES
from .m3_click import MODES, BIN_MS

# ---- protocol (fixed before the run; docs/m2h-brief.md §3) --------------------------
# 58 pulses at IPI 35 ms span 100 -> 2105 ms, so the 2000 ms window 100-2100 lies inside
# the stimulus. Total 100 + 1995 + 10 + 400 = 2505 ms.
STIM = dict(ipi_ms=35.0, n_pulses=58, pulse_ms=10.0, carrier_hz=200.0,
            pre_ms=100.0, post_ms=400.0)
TOTAL_MS = 2505.0
STIM_WINDOW_MS = (100.0, 2100.0)
SIGMAS = (29.13, 26.21, 23.30)          # base, x0.9, x0.8 - all inside M2f's width scan
N_SEEDS = 10                            # seeds 0..9
SIGMA_MULTIPLIER = 3.0                  # unchanged; what changed is the statistic it multiplies
IGNITION_ABSOLUTE_ACTIVE = 0.90         # brief §6: AND-ed with the excess test

# M2g's detection floors and the no-adaptation signal, for the achieved-vs-predicted table.
M2G_FLOOR_HZ = {"SAD_L": 0.143, "SAD_R": 0.044, "WED_L": 0.152, "WED_R": 0.241}
PREDICTED_FLOOR_HZ = {"SAD_L": 0.023, "SAD_R": 0.007, "WED_L": 0.024, "WED_R": 0.038}
NO_ADAPT_SIGNAL_HZ = {"SAD_L": 0.053, "SAD_R": 0.076, "WED_L": 0.120, "WED_R": 0.167}

RESULTS_JSON = ROOT / "data-provenance" / "m2h-results.json"


def log(msg: str) -> None:
    print(f"[m2h] {msg}", flush=True)


def _params(mode: str, adapt_g_b: float, sigma: float) -> EngineParams:
    return EngineParams(dt=MODES[mode], synapse="conductance", g=G, noise_sigma=sigma,
                        adapt_g_b=adapt_g_b, adapt_tau_a=TAU_A, E_adapt=E_ADAPT)


def one_run(engine, adapter, n_steps: int, dt_ms: float, sets, seed: int, n_neurons: int) -> dict:
    adapter.prepare(dt_ms, n_steps)
    engine.reset(seed)
    t_step, idx = engine.run(n_steps, adapter.inject, record=True)
    rt = bin_spikes(t_step, idx, dt_ms, BIN_MS, int(round(TOTAL_MS / BIN_MS)), sets)
    w0, w1 = STIM_WINDOW_MS
    rates = {name: window_mean_rate(rt, name, w0, w1) for name in rt.names}
    t_ms = np.asarray(t_step) * dt_ms
    late = t_ms >= w1 + 100.0
    late_active = int(len(np.unique(np.asarray(idx)[late])))
    return dict(seed=seed, n_spikes=int(rt.n_spikes),
                active_fraction=rt.active_neurons / n_neurons,
                late_active_fraction=late_active / n_neurons,
                window_rate_hz=rates, v_min=float(engine.v.min()))


def judge(stim_runs: list[dict], ctrl_runs: list[dict], set_names: list[str]) -> dict:
    """Difference of means against 3 x the standard error of the control mean.

    M2g compared a difference of means against 3 x the per-run standard deviation, which
    is not the matching statistic; correcting it is the whole point of this round. The
    direction, the sign convention and the set definitions are unchanged.
    """
    n = len(ctrl_runs)
    per_set = {}
    for name in set_names:
        s = np.array([r["window_rate_hz"][name] for r in stim_runs], dtype=float)
        c = np.array([r["window_rate_hz"][name] for r in ctrl_runs], dtype=float)
        sd_c = float(c.std(ddof=1))
        sem_c = sd_c / np.sqrt(n)
        diff = float(s.mean() - c.mean())
        # descriptive only: the SEM of the difference, which also carries the stimulus spread
        sem_diff = float(np.sqrt(c.var(ddof=1) + s.var(ddof=1)) / np.sqrt(n))
        per_set[name] = dict(
            ctrl_mean_hz=float(c.mean()), ctrl_sd_hz=sd_c, ctrl_sem_hz=sem_c,
            stim_mean_hz=float(s.mean()), stim_sd_hz=float(s.std(ddof=1)),
            diff_hz=diff, threshold_hz=SIGMA_MULTIPLIER * sem_c,
            exceeds=bool(diff > SIGMA_MULTIPLIER * sem_c),
            threshold_diff_hz=SIGMA_MULTIPLIER * sem_diff,
            exceeds_diff_sem=bool(diff > SIGMA_MULTIPLIER * sem_diff),
        )
    hop = {}
    for base in HOP3_BASES:
        members = [x for x in set_names if x == base or x.startswith(f"{base}_")]
        reached = [x for x in members if per_set[x]["exceeds"]]
        hop[base] = dict(members=members, reached=reached, any_reached=bool(reached),
                         best_diff_hz=max((per_set[x]["diff_hz"] for x in members), default=0.0))
    # redefined ignition (brief §6): excess over control AND >= 90 % absolute
    ls = np.array([r["late_active_fraction"] for r in stim_runs], dtype=float)
    lc = np.array([r["late_active_fraction"] for r in ctrl_runs], dtype=float)
    sem_l = float(lc.std(ddof=1)) / np.sqrt(n)
    excess = float(ls.mean() - lc.mean())
    ignition = dict(ctrl_mean=float(lc.mean()), stim_mean=float(ls.mean()), excess=excess,
                    ctrl_sem=sem_l, threshold=SIGMA_MULTIPLIER * sem_l,
                    excess_exceeds=bool(excess > SIGMA_MULTIPLIER * sem_l),
                    stim_max_active=float(ls.max()),
                    absolute_ge_90=bool(ls.max() >= IGNITION_ABSOLUTE_ACTIVE),
                    ignited=bool(excess > SIGMA_MULTIPLIER * sem_l
                                 and ls.max() >= IGNITION_ABSOLUTE_ACTIVE),
                    definition="excess over control >= 3 SEM AND stimulus active >= 90 %")
    transfer = {s: dict(diff_hz=per_set[f"JO_post_{s}"]["diff_hz"],
                        exceeds=per_set[f"JO_post_{s}"]["exceeds"]) for s in ("L", "R")}
    return dict(per_set=per_set, hop=hop, transfer=transfer, ignition=ignition,
                hop2_any=bool(any(hop[b]["any_reached"] for b in ("SAD", "WED", "AMMCtype"))),
                pc1_reached=bool(hop["pC1"]["any_reached"]))


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", default=",".join(OPERATING_POINTS))
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--sigmas", default=",".join(str(s) for s in SIGMAS))
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    points = [p for p in a.points.split(",") if p]
    modes = [m for m in a.modes.split(",") if m]
    sigmas = [float(s) for s in a.sigmas.split(",") if s]
    seeds = list(range(a.seeds))

    graph = Graph.load()
    roi = load_roi(graph)
    n_sites = roi["n_sites"].to_numpy()
    sets = build_probe_sets(graph, roi, k_min=K_MIN)
    stim = click_train(**STIM)
    quiet = silence(TOTAL_MS, stim.fs)
    assert abs(stim.duration_ms - TOTAL_MS) < 1e-6, (stim.duration_ms, TOTAL_MS)
    assert stim.params["stim_offset_ms"] >= STIM_WINDOW_MS[1], stim.params["stim_offset_ms"]

    n_runs = len(points) * len(sigmas) * len(modes) * 2 * len(seeds)
    steps = (len(points) * len(sigmas) * 2 * len(seeds)
             * sum(int(round(TOTAL_MS / MODES[m])) for m in modes))
    log(f"{n_runs} runs, {steps:,} steps. M2g measured ~6,600 steps/s end to end on this "
        f"(power-capped) machine, so expect roughly {steps / 6600 / 60:.0f}-{steps / 2600 / 60:.0f} min.")
    log(f"stimulus {stim.params['n_pulses']} pulses, {stim.params['stim_onset_ms']:g}-"
        f"{stim.params['stim_offset_ms']:g} ms, window {STIM_WINDOW_MS}, total {TOTAL_MS:g} ms")

    R = dict(generated_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
             purpose="close the hop-2 ambiguity of M2g by raising power only",
             operating_points=OPERATING_POINTS, g=G, a_in=A_IN, tau_a=TAU_A, E_adapt=E_ADAPT,
             sigmas=sigmas, seeds=seeds, stimulus=stim.describe(),
             stim_window_ms=list(STIM_WINDOW_MS), total_ms=TOTAL_MS,
             statistic=f"difference of means vs {SIGMA_MULTIPLIER} x SEM of the control mean "
                       f"(sd/sqrt({len(seeds)}))",
             ignition_definition=f"excess over control >= {SIGMA_MULTIPLIER} SEM AND stimulus "
                                 f"active >= {IGNITION_ABSOLUTE_ACTIVE:.0%}",
             reference=dict(m2g_floor_hz=M2G_FLOOR_HZ, predicted_floor_hz=PREDICTED_FLOOR_HZ,
                            no_adapt_signal_hz=NO_ADAPT_SIGNAL_HZ),
             probe_sets=dict(sizes=sets.sizes()), cells=[])

    t0 = time.time()
    for pname in points:
        b_g = OPERATING_POINTS[pname]["adapt_g_b"]
        for sigma in sigmas:
            for mode in modes:
                dt = MODES[mode]
                n_steps = int(round(TOTAL_MS / dt))
                p = _params(mode, b_g, sigma)
                engine = LIFEngine(graph, p, device=a.device)
                log(f"=== {pname} (b_g={b_g:g}) sigma={sigma:g} {mode}: {2 * len(seeds)} runs "
                    f"x {n_steps} steps")
                tc = time.time()
                ctrl = [one_run(engine, JOAdapter(graph, n_sites, quiet, mode, a_in=A_IN[mode],
                                                  device=a.device),
                                n_steps, dt, sets, s, graph.n) for s in seeds]
                stim_runs = [one_run(engine, JOAdapter(graph, n_sites, stim, mode, a_in=A_IN[mode],
                                                       device=a.device),
                                     n_steps, dt, sets, s, graph.n) for s in seeds]
                j = judge(stim_runs, ctrl, list(sets.names))
                R["cells"].append(dict(point=pname, adapt_g_b=b_g, sigma=sigma, mode=mode,
                                       dt_ms=dt, n_steps=n_steps, engine_params=p.to_dict(),
                                       control_runs=ctrl, stimulus_runs=stim_runs, judgement=j))
                floors = {k: j["per_set"][k]["threshold_hz"] for k in M2G_FLOOR_HZ}
                log(f"  {time.time() - tc:.0f}s  JO_post L/R {j['transfer']['L']['diff_hz']:+.3f}/"
                    f"{j['transfer']['R']['diff_hz']:+.3f}  hop2 "
                    f"{[b for b in ('SAD', 'WED', 'AMMCtype') if j['hop'][b]['any_reached']] or 'none'}  "
                    f"pC1 {j['pc1_reached']}  ignited {j['ignition']['ignited']}  "
                    f"floors " + " ".join(f"{k}={v:.3f}" for k, v in floors.items()))
                del engine
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

    R["wall_total_s"] = round(time.time() - t0, 1)
    RESULTS_JSON.write_text(json.dumps(R, indent=1, default=_json_default) + "\n")
    log(f"done in {R['wall_total_s']} s -> {RESULTS_JSON.relative_to(ROOT)}")
    _summary(R)
    return 0


def _summary(R: dict) -> None:
    print("\n| point | sigma | mode | JO_post L | JO_post R | hop-2 reached | pC1 | ignited |")
    print("|---|---|---|---|---|---|---|---|")
    for c in R["cells"]:
        j = c["judgement"]
        h = [b for b in ("SAD", "WED", "AMMCtype") if j["hop"][b]["any_reached"]]
        print(f"| {c['point']} | {c['sigma']:g} | {c['mode']} | {j['transfer']['L']['diff_hz']:+.3f} | "
              f"{j['transfer']['R']['diff_hz']:+.3f} | {', '.join(h) or 'none'} | "
              f"{j['pc1_reached']} | {j['ignition']['ignited']} |")
    print("\nachieved detection floor (3 x SEM of the control mean), Hz:")
    print("| set | M2g floor | predicted | achieved (min - max over cells) | no-adapt signal | below signal? |")
    print("|---|---|---|---|---|---|")
    for k in M2G_FLOOR_HZ:
        v = [c["judgement"]["per_set"][k]["threshold_hz"] for c in R["cells"]]
        print(f"| {k} | {M2G_FLOOR_HZ[k]:.3f} | {PREDICTED_FLOOR_HZ[k]:.3f} | {min(v):.3f} - {max(v):.3f} | "
              f"{NO_ADAPT_SIGNAL_HZ[k]:.3f} | {'yes' if max(v) < NO_ADAPT_SIGNAL_HZ[k] else 'NOT ALWAYS'} |")


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
