"""M7a2: does the escape circuit still pass once the background has settled?

Pre-registration: docs/m7a2-brief.md, written before this ran.

Why: M7a judged in 200-500 ms after a reset. Its own control runs (data-provenance/
m7a-results.json) show that in every one of 10 seeds BOTH giant fibres fire one spike at
543-706 ms, within 0-42 ms of each other - a synchronous wave as the background builds up -
and none in 200-500 ms. M7c, run on settled backgrounds, measured the GF firing on its own
at ~1.9 Hz per side. So M7a's "control 0, contralateral 0" was very likely measured in a
post-reset transient. This module separates the two.

  1. background time course: P2, no input, 10 s after reset, seeds 0-9, 100 ms bins;
     steady-state entry = first time after which the 1 s moving average stays within
     +/-20 % of the last-5-s mean (brief §1)
  2. 5 s warm-up with no input, then exactly M7a's stimulus (200 ms nothing, 300 ms
     exponential ramp with tau 50 ms, 300 ms observation); left / right / both / none;
     seeds 0-9; M7a's gain grid and selection walk
  3. judged against spontaneous firing: paired with the SAME seed's "none" run, the
     ipsilateral GF's (stimulus - none) spike difference in the 300 ms ramp window must
     have mean > 0 and > 3 SEM, and the contralateral difference must be smaller.

Pairing is exact: the engine is deterministic per seed and the noise is drawn every step
independently of the input, so a stimulus run and its "none" run see the identical warm-up
and the identical noise realisation - the paired difference isolates the stimulus.

Operating point P2 (adapt_g_b 8.0, g 5.36539e-4, sigma 29.13) decides the verdict. An
auxiliary arm at sigma 23.30 is descriptive only. No connection, weight or noise
distribution is changed.

Usage: python -m flysim.apps.m7a2_steady_probe [--arms 29.13,23.3]
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
from . import m7a_escape_probe as m7a

# ---- fixed before the run (docs/m7a2-brief.md) ------------------------------------------
DT_MS = 1.0
WARMUP_MS = 5000.0
STIM_TOTAL_MS = m7a.TOTAL_MS                               # 800: M7a's 200 + 300 + 300
TOTAL_MS = WARMUP_MS + STIM_TOTAL_MS                       # 5800
OFF = WARMUP_MS                                            # everything below is M7a's window + 5 s
STIM_WINDOW_MS = (OFF + m7a.STIM_WINDOW_MS[0], OFF + m7a.STIM_WINDOW_MS[1])   # 5200-5500
POST_WINDOW_MS = (OFF + m7a.POST_WINDOW_MS[0], OFF + m7a.POST_WINDOW_MS[1])   # 5500-5800
LATE_WINDOW_MS = (OFF + m7a.LATE_WINDOW_MS[0], OFF + m7a.LATE_WINDOW_MS[1])   # 5600-5800
SEGMENT_MS = (OFF, TOTAL_MS)       # the post-warm-up 800 ms = what M7a called "the run"

TIMECOURSE_MS = 10000.0
BIN_MS = 100.0
STEADY_MA_MS = 1000.0              # 1 s moving average
STEADY_REF_MS = 5000.0             # mean of the last 5 s
STEADY_TOL = 0.20                  # +/- 20 %

P2 = dict(g=5.36539e-4, adapt_g_b=8.0)
ARMS = {29.13: "P2 - decides the verdict", 23.30: "auxiliary arm - descriptive only"}
SEEDS = tuple(range(10))
SEM_MULT = 3.0

RESULTS_JSON = ROOT / "data-provenance" / "m7a2-results.json"
TIMECOURSE_JSON = ROOT / "data-provenance" / "m7a2-timecourse.json"


def log(msg: str) -> None:
    print(f"[m7a2] {msg}", flush=True)


def params(sigma: float) -> EngineParams:
    return EngineParams(dt=DT_MS, synapse="conductance", g=P2["g"], noise_sigma=sigma,
                        adapt_g_b=P2["adapt_g_b"])


# ---------------------------------------------------------------------------------------
# pure helpers
# ---------------------------------------------------------------------------------------
def stimulus(a_in: float) -> np.ndarray:
    """5 s of zeros, then exactly M7a's 800 ms stimulus."""
    return np.concatenate([np.zeros(int(WARMUP_MS / DT_MS)), m7a.loom_ramp(a_in)])


def steady_entry_ms(rate: np.ndarray, bin_ms: float = BIN_MS) -> float | None:
    """First time after which the 1 s moving average stays within +/-20 % of the mean of
    the last 5 s (brief §1). None if it never settles by that criterion or the reference is 0."""
    rate = np.asarray(rate, dtype=float)
    w = int(round(STEADY_MA_MS / bin_ms))
    ref = float(rate[-int(round(STEADY_REF_MS / bin_ms)):].mean())
    if ref <= 0:
        return None
    ma = np.convolve(rate, np.ones(w) / w, mode="valid")          # ma[b] = mean(rate[b:b+w])
    ok = np.abs(ma - ref) <= STEADY_TOL * ref
    for b in range(len(ma)):
        if ok[b:].all():
            return b * bin_ms
    return None


def _sem(x: np.ndarray) -> float:
    return float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0


def judge_side_paired(stim_rows: list[dict], none_by_seed: dict[int, dict], cond: str) -> dict:
    """Brief §5: differences against the same seed's "none" run."""
    ipsi, contra = m7a.ipsi_contra(cond)
    d_ip = np.array([r[f"{ipsi}_stim"] - none_by_seed[r["seed"]][f"{ipsi}_stim"] for r in stim_rows], float)
    d_co = np.array([r[f"{contra}_stim"] - none_by_seed[r["seed"]][f"{contra}_stim"] for r in stim_rows], float)
    m_ip, m_co, sem = float(d_ip.mean()), float(d_co.mean()), _sem(d_ip)
    exceeds = bool(m_ip > 0 and m_ip > SEM_MULT * sem)
    lateral = bool(m_co < m_ip)
    lat = [r[f"{ipsi}_first_ms"] - STIM_WINDOW_MS[0] for r in stim_rows if r[f"{ipsi}_first_ms"] is not None]
    spont = float(np.mean([none_by_seed[r["seed"]][f"{ipsi}_stim"] for r in stim_rows]))
    return dict(ipsi=ipsi, contra=contra, diff_ipsi_mean=m_ip, diff_ipsi_sem=sem,
                diff_contra_mean=m_co, threshold=SEM_MULT * sem, exceeds_3sem=exceeds,
                lateralised=lateral, passes=bool(exceeds and lateral),
                ipsi_stim_mean=float(np.mean([r[f"{ipsi}_stim"] for r in stim_rows])),
                ipsi_spontaneous_mean=spont,
                signal_to_spontaneous=(float(np.mean([r[f"{ipsi}_stim"] for r in stim_rows])) / spont
                                       if spont > 0 else None),
                latency_ms_median=(float(np.median(lat)) if lat else None),
                latency_ms_iqr=([float(np.percentile(lat, 25)), float(np.percentile(lat, 75))] if lat else None))


def first_spike_side(r: dict) -> str | None:
    """Which GF fired first in the judgement window: 'L', 'R', 'tie', or None if neither."""
    a, b = r["DNp01_L_first_ms"], r["DNp01_R_first_ms"]
    if a is None and b is None:
        return None
    if b is None or (a is not None and a < b):
        return "L"
    if a is None or b < a:
        return "R"
    return "tie"


def laterality(stim_by_cond: dict[str, list[dict]], none_rows: list[dict]) -> dict:
    """Descriptive: fraction of runs where the ipsilateral GF fired first (ties counted
    separately), against the same statistic on the "none" runs as a chance baseline."""
    hit = tot = tie = 0
    for cond, side in (("left", "L"), ("right", "R")):
        for r in stim_by_cond[cond]:
            s = first_spike_side(r)
            if s is None:
                continue
            tot += 1
            hit += s == side
            tie += s == "tie"
    base = [first_spike_side(r) for r in none_rows]
    base = [s for s in base if s is not None]
    return dict(ipsi_first=hit, ties=tie, runs_with_a_spike=tot,
                ipsi_first_fraction=(hit / tot if tot else None),
                none_runs_with_a_spike=len(base),
                none_left_first_fraction=(sum(s == "L" for s in base) / len(base) if base else None))


# ---------------------------------------------------------------------------------------
# runs
# ---------------------------------------------------------------------------------------
def run(engine: LIFEngine, sets: dict, condition: str, a_in: float, seed: int) -> dict:
    n_steps = int(round(TOTAL_MS / DT_MS))
    amp = stimulus(a_in if condition != "none" else 0.0)
    targets = torch.as_tensor(m7a.stimulated(sets, condition), device=engine.device)
    prev = [0.0]

    def fn(k: int, buf: torch.Tensor):
        if k == 0:
            buf.zero_()
            prev[0] = 0.0
        v = float(amp[k])
        if v != prev[0] and targets.numel():
            buf[targets] = v
            prev[0] = v

    engine.reset(seed)
    t, idx = engine.run(n_steps, fn, record=True)
    t_ms = t.astype(np.float64) * DT_MS
    idx = idx.astype(np.int64)
    n = engine.n
    r = dict(condition=condition, a_in=a_in, seed=seed)
    for name in ("DNp01_L", "DNp01_R", "TTMn_L", "TTMn_R"):
        neuron = int(sets[name][0])
        r[f"{name}_stim"], r[f"{name}_first_ms"] = m7a._count(t_ms, idx, neuron, STIM_WINDOW_MS)
        r[f"{name}_post"], _ = m7a._count(t_ms, idx, neuron, POST_WINDOW_MS)
        r[f"{name}_warmup"], _ = m7a._count(t_ms, idx, neuron, (0.0, WARMUP_MS))
    for name in ("LC4_L", "LC4_R", "LPLC2_L", "LPLC2_R"):
        m = np.isin(idx, sets[name]) & (t_ms >= STIM_WINDOW_MS[0]) & (t_ms < STIM_WINDOW_MS[1])
        r[f"{name}_stim_spikes"] = int(m.sum())
    late = (t_ms >= LATE_WINDOW_MS[0]) & (t_ms < LATE_WINDOW_MS[1])
    seg = (t_ms >= SEGMENT_MS[0]) & (t_ms < SEGMENT_MS[1])
    r["late_network_rate_hz"] = float(late.sum()) / (n * (LATE_WINDOW_MS[1] - LATE_WINDOW_MS[0]) / 1000.0)
    r["late_active_fraction"] = int(np.unique(idx[late]).size) / n
    # M7a's "active fraction" was over its whole 800 ms run; here that is the post-warm-up
    # segment - the warm-up is 5 s of background and would inflate it (M7c: 66 % over 5 s)
    r["active_fraction"] = int(np.unique(idx[seg]).size) / n
    return r


def timecourse(engine: LIFEngine, sets: dict, seed: int) -> dict:
    n_steps = int(round(TIMECOURSE_MS / DT_MS))
    engine.reset(seed)
    t, idx = engine.run(n_steps, None, record=True)
    nb = int(round(TIMECOURSE_MS / BIN_MS))
    b = (t.astype(np.int64) * DT_MS // BIN_MS).astype(np.int64)
    s = BIN_MS / 1000.0
    out = {"network_hz": (np.bincount(b, minlength=nb)[:nb] / (engine.n * s)).tolist()}
    for name in ("DNp01_L", "DNp01_R", "TTMn_L", "TTMn_R"):
        m = idx == int(sets[name][0])
        out[f"{name}_hz"] = (np.bincount(b[m], minlength=nb)[:nb] / s).tolist()
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default=",".join(str(s) for s in ARMS))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip-timecourse", action="store_true")
    a = ap.parse_args(argv)
    arms = [float(x) for x in a.arms.split(",") if x]

    graph = Graph.load()
    sets = m7a.build_sets(graph)
    steps = len(arms) * (len(SEEDS) * TIMECOURSE_MS
                         + len(SEEDS) * (3 * len(m7a.A_IN_GRID) + 1) * TOTAL_MS) / DT_MS
    log(f"arms {arms}; {steps:,.0f} steps in total")

    if not a.skip_timecourse:
        TC = dict(bin_ms=BIN_MS, duration_ms=TIMECOURSE_MS, seeds=list(SEEDS),
                  steady_definition="first time after which the 1 s moving average stays within "
                                    "+/-20 % of the mean of the last 5 s", arms={})
        for sigma in arms:
            engine = LIFEngine(graph, params(sigma), device=a.device)
            per = [timecourse(engine, sets, s) for s in SEEDS]
            mean = {k: np.mean([p[k] for p in per], axis=0).tolist() for k in per[0]}
            entry = {k: steady_entry_ms(np.array(v)) for k, v in mean.items()}
            last5 = {k: float(np.mean(v[-50:])) for k, v in mean.items()}
            TC["arms"][str(sigma)] = dict(seed_mean=mean, per_seed=per, steady_entry_ms=entry,
                                          last5s_mean_hz=last5)
            log(f"timecourse sigma={sigma:g}: steady entry {entry}; last-5-s means {last5}")
            del engine
            torch.cuda.empty_cache()
        TIMECOURSE_JSON.write_text(json.dumps(TC, indent=1) + "\n")

    R = dict(generated_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
             question="does M7a's PASS survive a settled background?",
             protocol=dict(dt_ms=DT_MS, warmup_ms=WARMUP_MS, stim_window_ms=STIM_WINDOW_MS,
                           post_window_ms=POST_WINDOW_MS, late_window_ms=LATE_WINDOW_MS,
                           segment_ms=SEGMENT_MS, ramp=dict(pre_ms=m7a.PRE_MS, ramp_ms=m7a.RAMP_MS,
                                                            tau_ms=m7a.RAMP_TAU_MS),
                           a_in_grid=m7a.A_IN_GRID, seeds=list(SEEDS)),
             criteria=dict(paired_against="same seed's 'none' run", sem_mult=SEM_MULT,
                           ignition="M7a three-number rule against the steady-state 'none' runs",
                           **{k: v for k, v in dict(runaway_rate_hz=m7a.RUNAWAY_RATE_HZ,
                                                     failure_active_fraction=m7a.FAILURE_ACTIVE_FRACTION,
                                                     excess_min_fraction=m7a.EXCESS_MIN_FRACTION,
                                                     excess_sem_mult=m7a.EXCESS_SEM_MULT).items()}),
             p2=P2, arms={})
    t0 = time.time()
    for sigma in arms:
        engine = LIFEngine(graph, params(sigma), device=a.device)
        log(f"=== sigma={sigma:g} ({ARMS.get(sigma, 'extra arm')})")
        none = [run(engine, sets, "none", 0.0, s) for s in SEEDS]
        none_by_seed = {r["seed"]: r for r in none}
        log(f"  none: GF L/R in window {np.mean([r['DNp01_L_stim'] for r in none]):.2f}/"
            f"{np.mean([r['DNp01_R_stim'] for r in none]):.2f} spikes per 300 ms, late rate "
            f"{np.mean([r['late_network_rate_hz'] for r in none]):.3f} Hz")
        sweep = []
        for a_in in m7a.A_IN_GRID:
            by_cond = {c: [run(engine, sets, c, a_in, s) for s in SEEDS] for c in ("left", "right", "both")}
            sides = {c: judge_side_paired(by_cond[c], none_by_seed, c) for c in ("left", "right")}
            ign = {c: m7a.ignition(by_cond[c], none) for c in ("left", "right", "both")}
            ignited = any(v["ignited"] for v in ign.values())
            j = dict(sides=sides, ignition=ign, ignited=ignited,
                     passes=bool(sides["left"]["passes"] and sides["right"]["passes"] and not ignited),
                     laterality=laterality(by_cond, none))
            sweep.append(dict(a_in=a_in, runs=by_cond, judgement=j))
            sl, sr, lat = sides["left"], sides["right"], j["laterality"]
            log(f"  a_in={a_in:6g}  L: d_ipsi {sl['diff_ipsi_mean']:+.2f} (3sem {sl['threshold']:.2f}) "
                f"d_contra {sl['diff_contra_mean']:+.2f} {'P' if sl['passes'] else '-'}  "
                f"R: d_ipsi {sr['diff_ipsi_mean']:+.2f} (3sem {sr['threshold']:.2f}) "
                f"d_contra {sr['diff_contra_mean']:+.2f} {'P' if sr['passes'] else '-'}  "
                f"ipsi-first {lat['ipsi_first']}/{lat['runs_with_a_spike']}  "
                f"{'IGNITED ' if ignited else ''}{'PASS' if j['passes'] else ''}")
        sel = m7a.select(sweep)
        R["arms"][str(sigma)] = dict(role=ARMS.get(sigma, "extra arm"), params=params(sigma).to_dict(),
                                     none=none, sweep=sweep, selection=sel)
        log(f"  -> selected a_in: {sel['selected_a_in']} (stopped at ignition: {sel['stopped_at_ignition']})")
        del engine
        torch.cuda.empty_cache()
    R["wall_total_s"] = round(time.time() - t0, 1)
    RESULTS_JSON.write_text(json.dumps(R, indent=1, default=m7a._json_default) + "\n")
    log(f"done in {R['wall_total_s']} s -> {RESULTS_JSON.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
