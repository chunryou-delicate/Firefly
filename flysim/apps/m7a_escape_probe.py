"""M7a: does a looming stimulus on one side reach the jump command - and the jump muscle?

Pre-registration: docs/m7a-brief.md (background docs/m7-game-ladder.md), written before
this ran. Everything below marked "fixed before the run" was set before any simulation.

The escape circuit, as the data has it (checked again here, CLAUDE.md §3.1):
    LC4, LPLC2  (looming detectors, cholinergic)  ->  DNp01 (giant fibre, GF)  ->  TTMn
Every LC4/LPLC2 cell of one side contacts only the GF of the same side. GF -> TTMn is a
chemical contact of 70 (right) and only 20 (left). The literature puts much of GF -> TTMn
on electrical synapses, which the connectome does not count, so both GF and TTMn are read.
**No connection is strengthened to compensate** (brief, "하지 말 것").

Stimulus (ASSUMPTION): a looming disc is mimicked by a current that grows exponentially
from 0 to ``a_in`` over 300 ms and is then cut - detector output taken as proportional to
angular expansion rate. 200 ms before, 300 ms after. Every cell of the stimulated set gets
the same value (no cell-to-cell heterogeneity, ASSUMPTION - which also makes the stimulated
cells fire in lock-step). Four conditions: left only, right only, both, none (control).

Two operating points, neither searched for here:
  P1  quiet conductance model   g = 3.162e-4, no noise, no adaptation        (1 seed)
  P2  adapted background (M2f)  g = 5.36539e-4, sigma 29.13, adapt_g_b 8.0  (seeds 0-9)

Gain: M3's rule - walk ``a_in`` up M3's doubling grid (10 ... 5120) and take the smallest
value that PASSes without ignition; the walk stops at the first ignited point. No PASS is a
result, not a reason to change anything.

Usage: python -m flysim.apps.m7a_escape_probe [--points P1,P2]
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
from ..engine.params import DEFAULT_G_CONDUCTANCE
from ..graph import Graph
from ..graph.constants import SIDE_CODES
from .m3_click import A_IN_GRID as M3_A_IN_GRID, A_IN_MAX as M3_A_IN_MAX

# ---- protocol (fixed before the run; docs/m7a-brief.md "자극") -------------------------
DT_MS = 1.0
PRE_MS, RAMP_MS, POST_MS = 200.0, 300.0, 300.0
TOTAL_MS = PRE_MS + RAMP_MS + POST_MS                    # 800 ms
STIM_WINDOW_MS = (PRE_MS, PRE_MS + RAMP_MS)              # 200-500 ms: the ramp itself
POST_WINDOW_MS = (PRE_MS + RAMP_MS, TOTAL_MS)            # 500-800 ms, descriptive
LATE_WINDOW_MS = (PRE_MS + RAMP_MS + 100.0, TOTAL_MS)    # 600-800 ms: offset + 100 (M3)
RAMP_TAU_MS = 50.0       # ASSUMPTION: e-folding time of the ramp; 4.7 % of max at 150 ms,
                         # 37 % at 250 ms, so most of the drive arrives in the last 100 ms
INPUT_TYPES = ("LC4", "LPLC2")
CONDITIONS = ("left", "right", "both", "none")

# M3's grid and its doubling extension, imported so it is provably M3's: 10 ... 5120.
A_IN_GRID = []
_a = M3_A_IN_GRID[0]
while _a <= M3_A_IN_MAX:
    A_IN_GRID.append(float(_a))
    _a *= 2

OPERATING_POINTS = {
    "P1": dict(g=DEFAULT_G_CONDUCTANCE, noise_sigma=0.0, adapt_g_b=0.0, seeds=(0,),
               why="quiet conductance model (M2b default g); no noise, so one seed suffices"),
    "P2": dict(g=5.36539e-4, noise_sigma=29.13, adapt_g_b=8.0, seeds=tuple(range(10)),
               why="the adapted background state M2f validated out of sample 5/5"),
}

# ---- PASS (fixed before the run; brief "판정") ------------------------------------------
P2_MIN_SEEDS_FIRING = 9          # of 10 seeds, ipsilateral GF spikes >= 1 in the stim window
SEM_MULT = 3.0                   # P2: excess over control > 3 x SEM of the control mean

# ---- ignition: the three-number rule (fixed before the run) ----------------------------
# parameter-decisions.md (2026-09-19, after M2i) defines the rule qualitatively: report the
# control's absolute state, the stimulus run's absolute state and the control-relative excess,
# and call a run usable only if the control is not itself in runaway and there is no
# stimulus-driven excess. The numbers below are taken from existing project thresholds, not
# chosen here:
RUNAWAY_RATE_HZ = 5.0            # network mean rate; = upper bound of M2d's usable band. The
                                 # adapted background sits at 0.1-0.35 Hz, runaway at 21-33 Hz.
FAILURE_ACTIVE_FRACTION = 0.90   # CLAUDE.md §3.3 / viewer failure rule
EXCESS_MIN_FRACTION = 0.01       # M3's ignition number (1 % of neurons in the late window),
                                 # now measured above the control instead of above zero
EXCESS_SEM_MULT = 3.0            # and, with noise, beyond 3 SEM of the control
# With no noise and a silent control (P1), the excess test reduces exactly to M3's rule.

RESULTS_JSON = ROOT / "data-provenance" / "m7a-results.json"
SETS_JSON = ROOT / "data-provenance" / "m7a-sets.json"


def log(msg: str) -> None:
    print(f"[m7a] {msg}", flush=True)


# ---------------------------------------------------------------------------------------
# sets and stimulus (pure)
# ---------------------------------------------------------------------------------------
def build_sets(graph) -> dict:
    """Neuron sets by data query. Raises LookupError (via the query API) on 0 hits and
    ValueError if the giant fibre or TTMn is not exactly one per side."""
    q = graph.query
    out: dict[str, np.ndarray] = {}
    for t in INPUT_TYPES + ("DNp01", "TTMn"):
        for s in ("L", "R"):
            out[f"{t}_{s}"] = np.sort(q.by_type(t, side=s)).astype(np.int64)
    for t in ("DNp01", "TTMn"):
        for s in ("L", "R"):
            if len(out[f"{t}_{s}"]) != 1:
                raise ValueError(f"expected exactly one {t} on side {s}, got {len(out[f'{t}_{s}'])}")
    return out


def stimulated(sets: dict, condition: str) -> np.ndarray:
    sides = {"left": ("L",), "right": ("R",), "both": ("L", "R"), "none": ()}[condition]
    parts = [sets[f"{t}_{s}"] for t in INPUT_TYPES for s in sides]
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)


def loom_ramp(a_in: float, dt_ms: float = DT_MS, tau_ms: float = RAMP_TAU_MS) -> np.ndarray:
    """Per-step amplitude: 0 before the ramp, a_in*(e^{t/tau}-1)/(e^{T/tau}-1) during it
    (0 at onset, exactly a_in at the last ramp step), 0 after."""
    n = int(round(TOTAL_MS / dt_ms))
    t = np.arange(n) * dt_ms
    out = np.zeros(n, dtype=np.float64)
    on = (t >= PRE_MS) & (t < PRE_MS + RAMP_MS)
    # normalise so the final ramp step reaches a_in exactly
    t_last = PRE_MS + RAMP_MS - dt_ms - PRE_MS
    out[on] = a_in * np.expm1((t[on] - PRE_MS) / tau_ms) / np.expm1(t_last / tau_ms)
    return out


# ---------------------------------------------------------------------------------------
# one run
# ---------------------------------------------------------------------------------------
def _count(t_ms: np.ndarray, idx: np.ndarray, neuron: int, window) -> tuple[int, float | None]:
    m = (idx == neuron) & (t_ms >= window[0]) & (t_ms < window[1])
    n = int(m.sum())
    return n, (float(t_ms[m].min()) if n else None)


def one_run(engine: LIFEngine, sets: dict, condition: str, a_in: float, seed: int) -> dict:
    n_steps = int(round(TOTAL_MS / DT_MS))
    amp = loom_ramp(a_in if condition != "none" else 0.0)
    targets = torch.as_tensor(stimulated(sets, condition), device=engine.device)
    prev = [None]

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
    r: dict = dict(condition=condition, a_in=a_in, seed=seed, n_spikes=int(len(t)))
    for name in ("DNp01_L", "DNp01_R", "TTMn_L", "TTMn_R"):
        neuron = int(sets[name][0])
        r[f"{name}_stim"], r[f"{name}_first_ms"] = _count(t_ms, idx, neuron, STIM_WINDOW_MS)
        r[f"{name}_post"], r[f"{name}_post_first_ms"] = _count(t_ms, idx, neuron, POST_WINDOW_MS)
    for name in ("LC4_L", "LC4_R", "LPLC2_L", "LPLC2_R"):
        m = np.isin(idx, sets[name]) & (t_ms >= STIM_WINDOW_MS[0]) & (t_ms < STIM_WINDOW_MS[1])
        r[f"{name}_stim_spikes"] = int(m.sum())
    late = (t_ms >= LATE_WINDOW_MS[0]) & (t_ms < LATE_WINDOW_MS[1])
    span_s = (LATE_WINDOW_MS[1] - LATE_WINDOW_MS[0]) / 1000.0
    r["late_network_rate_hz"] = float(late.sum()) / (n * span_s)
    r["late_active_fraction"] = int(np.unique(idx[late]).size) / n
    r["active_fraction"] = int(np.unique(idx).size) / n
    r["v_min"] = float(engine.v.min())
    return r


# ---------------------------------------------------------------------------------------
# judgement (pure; fixed before the run)
# ---------------------------------------------------------------------------------------
def ipsi_contra(cond: str) -> tuple[str, str]:
    return ("DNp01_L", "DNp01_R") if cond == "left" else ("DNp01_R", "DNp01_L")


def _sem(x: np.ndarray) -> float:
    return float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0


def judge_side(stim_rows: list[dict], ctrl_rows: list[dict], cond: str) -> dict:
    """PASS for one of the 'left'/'right' conditions."""
    ipsi, contra = ipsi_contra(cond)
    s_ip = np.array([r[f"{ipsi}_stim"] for r in stim_rows], dtype=float)
    s_co = np.array([r[f"{contra}_stim"] for r in stim_rows], dtype=float)
    c_ip = np.array([r[f"{ipsi}_stim"] for r in ctrl_rows], dtype=float)
    multi = len(stim_rows) > 1
    n_firing = int((s_ip >= 1).sum())
    fires = n_firing >= (P2_MIN_SEEDS_FIRING if multi else 1)
    excess = float(s_ip.mean() - c_ip.mean())
    if multi:
        ctrl_ok = bool(excess > SEM_MULT * _sem(c_ip))
    else:
        ctrl_ok = bool(c_ip.sum() == 0)
    lateral = bool(s_co.mean() < s_ip.mean())
    lat = [r[f"{ipsi}_first_ms"] for r in stim_rows if r[f"{ipsi}_first_ms"] is not None]
    return dict(ipsi=ipsi, contra=contra, ipsi_mean=float(s_ip.mean()), contra_mean=float(s_co.mean()),
                ctrl_mean=float(c_ip.mean()), excess=excess, n_seeds_firing=n_firing,
                n_seeds=len(stim_rows), fires=bool(fires), control_ok=ctrl_ok, lateralised=lateral,
                passes=bool(fires and ctrl_ok and lateral),
                latency_ms=(float(np.median(lat)) - PRE_MS) if lat else None)


def ignition(stim_rows: list[dict], ctrl_rows: list[dict]) -> dict:
    """The three-number rule. Returns all three numbers and the verdict."""
    c_rate = np.array([r["late_network_rate_hz"] for r in ctrl_rows])
    s_rate = np.array([r["late_network_rate_hz"] for r in stim_rows])
    s_act = np.array([r["active_fraction"] for r in stim_rows])
    c_late = np.array([r["late_active_fraction"] for r in ctrl_rows])
    s_late = np.array([r["late_active_fraction"] for r in stim_rows])
    excess = float(s_late.mean() - c_late.mean())
    control_runaway = bool(c_rate.mean() >= RUNAWAY_RATE_HZ)
    stim_runaway = bool((s_rate >= RUNAWAY_RATE_HZ).any() or (s_act >= FAILURE_ACTIVE_FRACTION).any())
    excess_ignited = bool(excess >= EXCESS_MIN_FRACTION and excess > EXCESS_SEM_MULT * _sem(c_late))
    return dict(control_rate_hz=float(c_rate.mean()), stim_rate_hz_max=float(s_rate.max()),
                stim_active_max=float(s_act.max()), late_active_excess=excess,
                control_runaway=control_runaway, stim_runaway=stim_runaway,
                excess_ignited=excess_ignited,
                ignited=bool(control_runaway or stim_runaway or excess_ignited))


def judge_point(by_cond: dict[str, list[dict]], ctrl_rows: list[dict]) -> dict:
    sides = {c: judge_side(by_cond[c], ctrl_rows, c) for c in ("left", "right")}
    ign = {c: ignition(by_cond[c], ctrl_rows) for c in ("left", "right", "both")}
    ignited = any(v["ignited"] for v in ign.values())
    both = by_cond["both"]
    both_desc = dict(
        L_mean=float(np.mean([r["DNp01_L_stim"] for r in both])),
        R_mean=float(np.mean([r["DNp01_R_stim"] for r in both])),
        co_active_seeds=int(sum(r["DNp01_L_stim"] > 0 and r["DNp01_R_stim"] > 0 for r in both)))
    ttmn = {c: {f"TTMn_{s}_{w}": float(np.mean([r[f"TTMn_{s}_{w}"] for r in by_cond[c]]))
                for s in ("L", "R") for w in ("stim", "post")} for c in ("left", "right", "both")}
    ttmn_ctrl = {f"TTMn_{s}_{w}": float(np.mean([r[f"TTMn_{s}_{w}"] for r in ctrl_rows]))
                 for s in ("L", "R") for w in ("stim", "post")}
    return dict(sides=sides, ignition=ign, ignited=ignited,
                passes=bool(sides["left"]["passes"] and sides["right"]["passes"] and not ignited),
                both=both_desc, ttmn=ttmn, ttmn_control=ttmn_ctrl)


def select(points: list[dict]) -> dict:
    """M3's walk: ascending a_in; the first ignited point stops the walk; the first
    non-ignited PASS before that is selected."""
    for p in sorted(points, key=lambda x: x["a_in"]):
        if p["judgement"]["ignited"]:
            return dict(selected_a_in=None, stopped_at_ignition=p["a_in"])
        if p["judgement"]["passes"]:
            return dict(selected_a_in=p["a_in"], stopped_at_ignition=None)
    return dict(selected_a_in=None, stopped_at_ignition=None)


# ---------------------------------------------------------------------------------------
def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", default=",".join(OPERATING_POINTS))
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    points = [p for p in a.points.split(",") if p]

    graph = Graph.load()
    sets = build_sets(graph)
    sizes = {k: int(len(v)) for k, v in sets.items()}
    log(f"sets: {sizes}")
    body = {k: [int(b) for b in graph.body(v)] for k, v in sets.items() if k.startswith(("DNp01", "TTMn"))}
    SETS_JSON.write_text(json.dumps(dict(sizes=sizes, body_ids=body,
                                         side_source="somaSide (complete for all four types; the "
                                                     "retained annotation cache has no rootSide column)"),
                                    indent=1) + "\n")

    n_runs = sum(len(OPERATING_POINTS[p]["seeds"]) * (3 * len(A_IN_GRID) + 1) for p in points)
    log(f"{n_runs} runs x {int(TOTAL_MS / DT_MS)} steps; a_in grid {A_IN_GRID}")

    R = dict(generated_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
             question="does a one-sided looming stimulus reach the ipsilateral giant fibre and TTMn?",
             protocol=dict(dt_ms=DT_MS, pre_ms=PRE_MS, ramp_ms=RAMP_MS, post_ms=POST_MS,
                           ramp_tau_ms=RAMP_TAU_MS, stim_window_ms=STIM_WINDOW_MS,
                           post_window_ms=POST_WINDOW_MS, late_window_ms=LATE_WINDOW_MS,
                           a_in_grid=A_IN_GRID, conditions=CONDITIONS, input_types=INPUT_TYPES),
             criteria=dict(p2_min_seeds_firing=P2_MIN_SEEDS_FIRING, sem_mult=SEM_MULT,
                           runaway_rate_hz=RUNAWAY_RATE_HZ, failure_active_fraction=FAILURE_ACTIVE_FRACTION,
                           excess_min_fraction=EXCESS_MIN_FRACTION, excess_sem_mult=EXCESS_SEM_MULT),
             sets=sizes, points={})
    t0 = time.time()
    for pname in points:
        P = OPERATING_POINTS[pname]
        params = EngineParams(dt=DT_MS, synapse="conductance", g=P["g"], noise_sigma=P["noise_sigma"],
                              adapt_g_b=P["adapt_g_b"])
        engine = LIFEngine(graph, params, device=a.device)
        log(f"=== {pname}: g={P['g']:g} sigma={P['noise_sigma']:g} b_g={P['adapt_g_b']:g} seeds={P['seeds']}")
        ctrl = [one_run(engine, sets, "none", 0.0, s) for s in P["seeds"]]
        log(f"  control: GF L/R stim {np.mean([r['DNp01_L_stim'] for r in ctrl]):.2f}/"
            f"{np.mean([r['DNp01_R_stim'] for r in ctrl]):.2f}, late rate "
            f"{np.mean([r['late_network_rate_hz'] for r in ctrl]):.3f} Hz")
        sweep = []
        for a_in in A_IN_GRID:
            by_cond = {c: [one_run(engine, sets, c, a_in, s) for s in P["seeds"]]
                       for c in ("left", "right", "both")}
            j = judge_point(by_cond, ctrl)
            sweep.append(dict(a_in=a_in, runs=by_cond, judgement=j))
            sl, sr = j["sides"]["left"], j["sides"]["right"]
            log(f"  a_in={a_in:6g}  L: GF {sl['ipsi_mean']:.2f}/{sl['contra_mean']:.2f} "
                f"({sl['n_seeds_firing']}/{sl['n_seeds']}) {'P' if sl['passes'] else '-'}  "
                f"R: GF {sr['ipsi_mean']:.2f}/{sr['contra_mean']:.2f} "
                f"({sr['n_seeds_firing']}/{sr['n_seeds']}) {'P' if sr['passes'] else '-'}  "
                f"TTMn(L-stim) L/R {j['ttmn']['left']['TTMn_L_stim']:.1f}/{j['ttmn']['left']['TTMn_R_stim']:.1f} "
                f"TTMn(R-stim) L/R {j['ttmn']['right']['TTMn_L_stim']:.1f}/{j['ttmn']['right']['TTMn_R_stim']:.1f}  "
                f"{'IGNITED ' if j['ignited'] else ''}{'PASS' if j['passes'] else ''}")
        sel = select(sweep)
        R["points"][pname] = dict(params=params.to_dict(), why=P["why"], seeds=list(P["seeds"]),
                                  control=ctrl, sweep=sweep, selection=sel)
        log(f"  -> selected a_in: {sel['selected_a_in']}  (walk stopped at ignition: {sel['stopped_at_ignition']})")
        del engine
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    R["wall_total_s"] = round(time.time() - t0, 1)
    RESULTS_JSON.write_text(json.dumps(R, indent=1, default=_json_default) + "\n")
    log(f"done in {R['wall_total_s']} s -> {RESULTS_JSON.relative_to(ROOT)}")
    return 0


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
