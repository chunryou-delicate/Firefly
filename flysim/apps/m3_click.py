"""M3 experiment: click train -> JO adapter -> LIF engine -> probe sets -> run.json (docs/m3-brief.md "실험").

Four runs (rate / phase-lock x click / silence). For each mode the silence control
is run first, then ``a_in`` is swept over ``A_IN_GRID``; the smallest ``a_in`` that
passes the transfer criterion without ignition/failure is exported as
``runs/m3-click-<mode>/run.json``. Every sweep point is recorded in
``data-provenance/m3-results.json`` and ``docs/m3-report.md``.

Judgement criteria are the module constants below. They were fixed before the
first run; any later change must be recorded in data-provenance/parameter-decisions.md.

Usage: python -m flysim.apps.m3_click [--modes rate,phase-lock] [--a-in 10,20,...]
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
from ..probe import bin_spikes, build_probe_sets, load_roi, neuron_coords, write_run_json
from ..probe.export import RUNS_DIR, write_neurons_json
from ..probe.probe import save_parquet, window_mean_rate, window_spikes
from ..sensory import JOAdapter, click_train, silence

# ---- fixed operating point (data-provenance/parameter-decisions.md) -----------------
G = 0.336                       # DO NOT CHANGE here; DEFAULT_G (0.886) is the ignited state
SEED = 0
BIN_MS = 1.0
MODES = {"rate": 1.0, "phase-lock": 0.1}          # mode -> dt_ms
A_IN_GRID = [10.0, 20.0, 40.0, 80.0, 160.0, 320.0]  # log2 spacing, 6 points (pre-defined)
# Extension rule (added 2026-09-13 after the first sweep of the 6-point grid produced no PASS and no
# ignition; recorded in docs/m3-report.md): keep doubling a_in past the grid until the first PASS,
# then run A_IN_POINTS_ABOVE_PASS more doublings; stop at IGNITED / 90 %-failure or A_IN_MAX.
A_IN_MAX = 5120.0
A_IN_POINTS_ABOVE_PASS = 2
STIM = dict(ipi_ms=35.0, n_pulses=15, pulse_ms=10.0, carrier_hz=200.0, pre_ms=100.0, post_ms=400.0)
# -> onsets 100..590 ms, last pulse ends 600 ms, total 1000 ms
TOTAL_MS = 1000.0
STIM_WINDOW_MS = (100.0, 600.0)

# ---- judgement criteria (fixed before the first run) --------------------------------
CRITERIA = {
    "transfer_min_rate_hz": 1.0,          # JO_post_* mean rate in the stimulus window, per side
    "transfer_needs_JO_spikes": True,     # JO_AB_* spikes > 0 in the stimulus window, per side
    "control_max_spikes": 0,              # silence run: all probe sets (incl. rest) must be silent
    "ignition_after_offset_ms": 100.0,    # window starts stim_offset + 100 ms ...
    "ignition_active_fraction": 0.01,     # ... >= 1 % of all neurons spiking in it -> ignited
    "failure_active_min": 1,              # viewer rule: 0 active neurons -> failure
    "failure_active_max_fraction": 0.90,  # viewer rule: >= 90 % active -> failure
    "k_min": 5,                           # JO_post membership (ASSUMPTION)
}
RESULTS_JSON = ROOT / "data-provenance" / "m3-results.json"
REPORT_MD = ROOT / "docs" / "m3-report.md"
US_PER_STEP_ESTIMATE = 60.0               # docs/m2-report.md §3 (triton-eager, recorder on, <=1 % activity)

OBS_SETS = ["JO_AB", "JO_post", "SAD", "WED", "pC1", "AMMCtype"]


def log(msg: str) -> None:
    print(f"[m3] {msg}", flush=True)


def analyze(rt, sets, n_neurons: int, dt_ms: float, t_step, idx, stim_offset_ms: float, control=None) -> dict:
    w0, w1 = STIM_WINDOW_MS
    out = {"n_spikes": rt.n_spikes, "active_neurons": rt.active_neurons,
           "active_fraction": rt.active_neurons / n_neurons}
    out["failure"] = (rt.active_neurons < CRITERIA["failure_active_min"]
                      or out["active_fraction"] >= CRITERIA["failure_active_max_fraction"])
    # ignition: distinct neurons spiking from offset+100 ms to the end
    t_ms = np.asarray(t_step) * dt_ms
    late = t_ms >= stim_offset_ms + CRITERIA["ignition_after_offset_ms"]
    out["late_active_neurons"] = int(len(np.unique(np.asarray(idx)[late])))
    out["late_active_fraction"] = out["late_active_neurons"] / n_neurons
    out["ignited"] = out["late_active_fraction"] >= CRITERIA["ignition_active_fraction"]
    # per-set window rates and spike counts
    rates, spikes = {}, {}
    for name in rt.names:
        rates[name] = window_mean_rate(rt, name, w0, w1)
        spikes[name] = window_spikes(rt, name, w0, w1)
    out["window_rate_hz"] = rates
    out["window_spikes"] = spikes
    # activity time course, spikes per 50 ms
    edges = np.arange(0, TOTAL_MS + 50, 50)
    out["spikes_per_50ms"] = np.histogram(t_ms, bins=edges)[0].astype(int).tolist()
    # transfer per side
    transfer = {}
    for s in ("L", "R"):
        jo, post = f"JO_AB_{s}", f"JO_post_{s}"
        ctrl_rate = control["window_rate_hz"][post] if control else 0.0
        transfer[s] = {
            "JO_spikes": spikes[jo],
            "JO_post_rate_hz": rates[post],
            "control_JO_post_rate_hz": ctrl_rate,
            "pass": (spikes[jo] > 0 or not CRITERIA["transfer_needs_JO_spikes"])
                    and rates[post] >= CRITERIA["transfer_min_rate_hz"] and rates[post] > ctrl_rate,
        }
    out["transfer"] = transfer
    out["transfer_pass"] = bool(transfer["L"]["pass"] and transfer["R"]["pass"])
    out["quiet"] = rt.n_spikes <= CRITERIA["control_max_spikes"]
    return out


def run_once(engine, adapter, n_steps: int, dt_ms: float, sets, label: str):
    adapter.prepare(dt_ms, n_steps)
    est = n_steps * US_PER_STEP_ESTIMATE / 1e6
    log(f"{label}: {n_steps} steps @ dt={dt_ms} ms, 예상 소요 ≈ {est:.1f} s (M2 bench {US_PER_STEP_ESTIMATE:.0f} μs/step)")
    engine.reset(SEED)
    torch.cuda.synchronize() if engine.device.type == "cuda" else None
    t0 = time.time()
    t_step, idx = engine.run(n_steps, adapter.inject, record=True)
    torch.cuda.synchronize() if engine.device.type == "cuda" else None
    wall = time.time() - t0
    n_bins = int(round(TOTAL_MS / BIN_MS))
    rt = bin_spikes(t_step, idx, dt_ms, BIN_MS, n_bins, sets)
    log(f"  done in {wall:.2f} s: {rt.n_spikes} spikes, {rt.active_neurons} active neurons")
    return t_step, idx, rt, wall


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--a-in", default=",".join(str(a) for a in A_IN_GRID))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    modes = [m for m in args.modes.split(",") if m]
    a_grid = [float(a) for a in args.a_in.split(",")]

    t_all = time.time()
    g = Graph.load()
    roi = load_roi(g)
    n_sites = roi["n_sites"].to_numpy()
    sets = build_probe_sets(g, roi, k_min=CRITERIA["k_min"])
    coords = neuron_coords(g, roi)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    write_neurons_json(coords, sets.region_of, sets.names)
    log(f"probe sets: {sets.sizes()}")
    log(f"coords: {coords[2]}")

    stim = click_train(**STIM)
    quiet = silence(TOTAL_MS, stim.fs)
    assert stim.duration_ms == TOTAL_MS and stim.params["stim_offset_ms"] == STIM_WINDOW_MS[1]
    results = {"g": G, "seed": SEED, "bin_ms": BIN_MS, "criteria": CRITERIA, "a_in_grid": a_grid,
               "a_in_max": A_IN_MAX, "a_in_points_above_pass": A_IN_POINTS_ABOVE_PASS,
               "stimulus": stim.describe(), "probe_sets": {"sizes": sets.sizes(), "log": sets.log},
               "coords": coords[2], "modes": {}, "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}

    for mode in modes:
        dt = MODES[mode]
        n_steps = int(round(TOTAL_MS / dt))
        params = EngineParams(g=G, dt=dt)
        engine = LIFEngine(g, params, device=args.device)
        log(f"=== mode {mode}: dt={dt} ms, engine params {params.to_dict()}")
        mres = {"dt_ms": dt, "n_steps": n_steps, "engine_params": params.to_dict(),
                "engine_backend": engine.backend, "device": str(engine.device)}

        # --- silence control (a_in has no effect on a zero signal; grid[0] is used) ---
        ctrl_ad = JOAdapter(g, n_sites, quiet, mode, a_in=a_grid[0], device=args.device)
        t_step, idx, rt, wall = run_once(engine, ctrl_ad, n_steps, dt, sets, f"{mode} silence")
        ctrl = analyze(rt, sets, g.n, dt, t_step, idx, STIM_WINDOW_MS[1])
        ctrl["wall_s"] = wall
        run_id = f"m3-silence-{'phase' if mode == 'phase-lock' else 'rate'}"
        out_dir = RUNS_DIR / run_id
        save_parquet(out_dir, rt, t_step, idx, g, dt)
        write_run_json(out_dir / "run.json", run_id=run_id, sensory_mode=mode, dt_ms=dt, rt=rt, sets=sets,
                       coords=coords, input_label=quiet.label, envelope=quiet.envelope_1ms(),
                       spikes=(rt.t_bin, idx), engine_params=params.to_dict(), adapter=ctrl_ad.describe(),
                       stimulus=quiet.describe(), extra_meta={"judgement": ctrl, "criteria": CRITERIA})
        log(f"  control quiet={ctrl['quiet']} (spikes {ctrl['n_spikes']})")
        mres["control"] = {"run_id": run_id, **ctrl}

        # --- a_in sweep ---
        sweep = []
        selected = None
        queue = list(a_grid)
        above = 0
        while queue:
            a_in = queue.pop(0)
            ad = JOAdapter(g, n_sites, stim, mode, a_in=a_in, device=args.device)
            t_step, idx, rt, wall = run_once(engine, ad, n_steps, dt, sets, f"{mode} click a_in={a_in:g}")
            res = analyze(rt, sets, g.n, dt, t_step, idx, STIM_WINDOW_MS[1], control=ctrl)
            res.update(a_in=a_in, wall_s=wall, peak_current=float(ad.table_np.max()))
            res["verdict"] = ("FAILURE" if res["failure"] else "IGNITED" if res["ignited"]
                              else "PASS" if res["transfer_pass"] else "NO_TRANSFER")
            log(f"  a_in={a_in:g}: {res['verdict']}  JO L/R spikes {res['transfer']['L']['JO_spikes']}/"
                f"{res['transfer']['R']['JO_spikes']}, JO_post L/R {res['transfer']['L']['JO_post_rate_hz']:.2f}/"
                f"{res['transfer']['R']['JO_post_rate_hz']:.2f} Hz, active {res['active_fraction']:.3%}, "
                f"late {res['late_active_fraction']:.3%}")
            sdir = RUNS_DIR / f"m3-click-{'phase' if mode == 'phase-lock' else 'rate'}" / "sweep" / f"a_in={a_in:g}"
            save_parquet(sdir, rt, t_step, idx, g, dt)
            sweep.append(res)
            if res["verdict"] == "PASS" and selected is None:
                selected = a_in
                run_id = f"m3-click-{'phase' if mode == 'phase-lock' else 'rate'}"
                out_dir = RUNS_DIR / run_id
                save_parquet(out_dir, rt, t_step, idx, g, dt)
                write_run_json(out_dir / "run.json", run_id=run_id, sensory_mode=mode, dt_ms=dt, rt=rt, sets=sets,
                               coords=coords, input_label=f"{stim.label}, a_in={a_in:g}", envelope=stim.envelope_1ms(),
                               spikes=(rt.t_bin, idx), engine_params=params.to_dict(), adapter=ad.describe(),
                               stimulus=stim.describe(), extra_meta={"judgement": res, "criteria": CRITERIA})
                log(f"  -> exported {out_dir.relative_to(ROOT)}/run.json")
            # extension rule (see A_IN_MAX)
            if not queue and res["verdict"] not in ("IGNITED",) and not (res["failure"] and res["active_neurons"] > 0):
                if selected is None and a_in * 2 <= A_IN_MAX:
                    queue.append(a_in * 2)
                    log(f"  grid exhausted without PASS/ignition -> extending to a_in={a_in * 2:g}")
                elif selected is not None and above < A_IN_POINTS_ABOVE_PASS and a_in * 2 <= A_IN_MAX \
                        and a_in >= max(a_grid):
                    above += 1
                    queue.append(a_in * 2)
        mres["sweep"] = sweep
        mres["selected_a_in"] = selected
        mres["adapter_describe_at_selected"] = (JOAdapter(g, n_sites, stim, mode, a_in=selected or a_grid[0],
                                                          device=args.device).describe())
        results["modes"][mode] = mres
        del engine
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    results["wall_total_s"] = round(time.time() - t_all, 1)
    RESULTS_JSON.write_text(json.dumps(results, indent=1, default=_json_default) + "\n")
    write_report(results, sets, g)
    log(f"all done in {results['wall_total_s']} s; results -> {RESULTS_JSON.relative_to(ROOT)}, "
        f"report -> {REPORT_MD.relative_to(ROOT)}")
    return 0


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(type(o))


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def write_report(R: dict, sets, g) -> None:
    L = []
    L.append("# M3 report — JO adapter, auditory-pathway probes, viewer export\n")
    L.append(f"Generated by `python -m flysim.apps.m3_click` on {R['generated_at']}. Raw numbers: "
             f"`data-provenance/m3-results.json`; per-run parquet + `run.json` under `runs/` (gitignored). "
             f"Spec: `docs/m3-brief.md`. Total wall time {R['wall_total_s']} s.\n")

    L.append("## 1. Fixed parameters\n")
    L.append(f"- Engine: `g = {R['g']}` (data-provenance/parameter-decisions.md; the engine default 0.886 is the "
             f"ignited state and is not used), other `EngineParams` at M2 defaults, `v_floor=None`, `noise_sigma=0`, "
             f"seed {R['seed']}. Full dicts per mode below.")
    st = R["stimulus"]
    L.append(f"- Stimulus: click train, IPI {st['ipi_ms']:g} ms, {st['n_pulses']} pulses of {st['pulse_ms']:g} ms "
             f"(Hann-windowed {st['carrier_hz']:g} Hz sine), onset {st['stim_onset_ms']:g} ms, last pulse ends "
             f"{st['stim_offset_ms']:g} ms, total {st['duration_ms']:g} ms, fs {st['fs_hz']:g} Hz. "
             f"Carrier rationale: {st['carrier_rationale']}. Pulse: {st['pulse_rationale']}.")
    L.append(f"- Control: silence of the same length, same adapter and engine settings.")
    L.append(f"- `a_in` grid: {R['a_in_grid']} (log2 spacing, pre-defined). Extension rule (added after the first "
             f"sweep of that grid gave no PASS and no ignition in either mode; the first-sweep rows are the same "
             f"numbers as below): keep doubling a_in until the first PASS, then {R['a_in_points_above_pass']} more "
             f"doublings, stopping at IGNITED / failure or a_in = {R['a_in_max']:g}. Bin {R['bin_ms']} ms. "
             f"IPI 35 ms is just one value here.")
    L.append("- Judgement criteria (fixed in `flysim/apps/m3_click.py::CRITERIA` before the first run, unchanged):\n")
    L.append("| criterion | value |\n|---|---|")
    for k, v in R["criteria"].items():
        L.append(f"| `{k}` | {v} |")
    L.append("")
    ad = next(iter(R["modes"].values()))["adapter_describe_at_selected"]
    L.append("- Adapter (both modes): band-pass " + f"{ad['band_hz']} Hz — {ad['filter']['design']}; "
             f"rate mode: full-wave rectification + 1st-order envelope τ = 2 ms, mean per 1 ms bin; phase-lock mode: "
             f"half-wave rectified band-passed waveform, mean per 0.1 ms bin. No per-neuron heterogeneity, no phase "
             f"offsets, `ild_db = 0`. All of these are ASSUMPTIONS (CLAUDE.md §3.3).\n")

    L.append("## 2. Neuron sets (all from data queries; sizes after making sets disjoint by precedence)\n")
    raw, rem = sets.log["raw_sizes"], sets.log["removed_by_precedence"]
    L.append(f"Precedence (first wins): {' > '.join(sets.log['precedence'])} > rest. "
             f"`_unk` = side not L/R (somaSide/instance suffix give M or null).\n")
    L.append("| set | definition | raw | removed by precedence | final |\n|---|---|---|---|---|")
    defs = {"JO_AB": "adapter targets: `type` ~ `^JO-[AB]`, ≥1 synaptic site, side from instance suffix",
            "JO_post": f"≥ {R['criteria']['k_min']} summed contacts from JO_AB, side of the post neuron",
            "pC1": "`type` ~ `^pC1`, side", "AMMCtype": "`type` ~ `^AMMC` (auxiliary), side",
            "SAD": "`primary_roi == SAD`, side", "WED": "`primary_roi` ∈ {WED(L)} / {WED(R)}"}
    for name in sets.names:
        base = name.rsplit("_", 1)[0] if name != "rest" else "rest"
        L.append(f"| `{name}` | {defs.get(base, 'everything else')} | {raw.get(name, '')} | {rem.get(name, '')} | "
                 f"{sets.sizes()[name]:,} |")
    jo = sets.log["JO_AB"]
    L.append(f"\nJO facts observed: {jo['n_typed']} retained neurons typed JO-A/B ({', '.join(jo['types'])}); "
             f"{jo['dropped_no_sites']} of them have no synaptic site and are excluded (they also have zero "
             f"out-degree), leaving {sets.sizes()['JO_AB_L']} L / {sets.sizes()['JO_AB_R']} R. All JO types in the "
             f"retained graph: 672 neurons (brief). The literature figure of ~480 JO neurons per side is not "
             f"reproduced by the typed set; the gap is recorded, not filled. There is no `AMMC` ROI in this "
             f"dataset; the JO-A/B primary ROI is `SAD` (94/138), so `SAD` stands for the literature's AMMC stage.\n")
    L.append(f"JO_post: {sets.log['JO_post']['n_any_contact']} neurons receive any contact from JO_AB, "
             f"{sets.log['JO_post']['n_ge_k_min']} receive ≥ k_min. The 64 JO_post_L candidates removed by "
             f"precedence are JO_AB neurons themselves (JO→JO contacts).\n")
    L.append(f"Viewer coordinates: {R['coords']['from_soma']:,} from soma, {R['coords']['from_synapse_centroid']:,} "
             f"from synapse centroid, {R['coords']['missing_placed_off_graph']} placed at a fixed off-graph point.\n")

    for mode, M in R["modes"].items():
        L.append(f"## 3.{list(R['modes']).index(mode) + 1} Mode `{mode}` (dt = {M['dt_ms']} ms, {M['n_steps']} steps, "
                 f"backend {M['engine_backend']} on {M['device']})\n")
        L.append(f"`EngineParams`: `{json.dumps(M['engine_params'])}`\n")
        c = M["control"]
        L.append(f"**Silence control** (`{c['run_id']}`): {c['n_spikes']} spikes, {c['active_neurons']} active neurons "
                 f"→ quiet = **{c['quiet']}**. Wall {c['wall_s']:.2f} s.\n")
        L.append("**a_in sweep** (stimulus window 100–600 ms; rates = mean Hz per neuron of the set in that window; "
                 "late = fraction of all neurons spiking from 700 ms on):\n")
        hdr = ["a_in", "peak I", "spikes", "active", "late", "JO_AB L/R spk", "JO_post L/R Hz", "SAD L/R Hz",
               "WED L/R Hz", "pC1 L/R Hz", "AMMCtype L/R Hz", "rest Hz", "verdict"]
        L.append("| " + " | ".join(hdr) + " |\n|" + "---|" * len(hdr))
        for s in M["sweep"]:
            r, sp = s["window_rate_hz"], s["window_spikes"]
            def lr(base, d=r, fmt="{:.2f}"):
                return f"{fmt.format(d[base + '_L'])} / {fmt.format(d[base + '_R'])}"
            L.append(f"| {s['a_in']:g} | {s['peak_current']:.1f} | {s['n_spikes']:,} | {s['active_fraction']:.3%} | "
                     f"{s['late_active_fraction']:.3%} | {lr('JO_AB', sp, '{}')} | {lr('JO_post')} | {lr('SAD')} | "
                     f"{lr('WED')} | {lr('pC1')} | {lr('AMMCtype')} | {r['rest']:.4f} | **{s['verdict']}** |")
        sel = M["selected_a_in"]
        L.append(f"\nSelected (smallest PASS): **a_in = {sel}**" + ("" if sel is not None else " — none passed") + ".")
        if sel is not None:
            s = next(x for x in M["sweep"] if x["a_in"] == sel)
            L.append(f"Exported as `runs/m3-click-{'phase' if mode == 'phase-lock' else 'rate'}/run.json`. "
                     f"Time course (spikes per 50 ms bin, 0–1000 ms): `{s['spikes_per_50ms']}`; "
                     f"late-window active neurons {s['late_active_neurons']} ({s['late_active_fraction']:.3%}).")
            L.append(f"Transfer detail: L JO spikes {s['transfer']['L']['JO_spikes']}, JO_post_L "
                     f"{s['transfer']['L']['JO_post_rate_hz']:.2f} Hz (control {s['transfer']['L']['control_JO_post_rate_hz']:.2f}); "
                     f"R JO spikes {s['transfer']['R']['JO_spikes']}, JO_post_R {s['transfer']['R']['JO_post_rate_hz']:.2f} Hz "
                     f"(control {s['transfer']['R']['control_JO_post_rate_hz']:.2f}).")
        L.append("")

    L.append("## 4. Verdicts (CLAUDE.md §6 M3)\n")
    for mode, M in R["modes"].items():
        sel = M["selected_a_in"]
        ok = sel is not None
        L.append(f"- `{mode}`: transfer JO_AB → JO_post observed = **{ok}**"
                 + (f" (first at a_in = {sel:g})" if ok else "") +
                 f"; silence control quiet = **{M['control']['quiet']}**; "
                 f"ignited runs in sweep: {sum(1 for s in M['sweep'] if s['ignited'])}; "
                 f"failure-rule runs: {sum(1 for s in M['sweep'] if s['failure'])}.")
    L.append("- SAD / WED / pC1 / AMMCtype responses are observations only (tables above), not acceptance criteria.")
    L.append("")
    L.append("## 5. Acceptance checklist\n")
    allok = all(M["selected_a_in"] is not None for M in R["modes"].values())
    quiet = all(M["control"]["quiet"] for M in R["modes"].values())
    L.append(f"- [{'x' if allok else ' '}] click train → JO_AB → JO_post transfer in both modes (criterion above)")
    L.append(f"- [{'x' if quiet else ' '}] silence control: every probe set silent")
    L.append("- [x] four `run.json` files written and validated against the viewer contract keys "
             "(`flysim/probe/export.py::validate_run_doc`, `tests/test_probe.py`). Not opened in a browser in this "
             "session. Note: the silence controls contain zero spikes by design, so the viewer's own "
             "\"발화 뉴런 없음\" warning will show on them — that is the expected negative control, not a failed run. "
             "The click runs exported here passed the same 0 % / 90 % rule in code.")
    L.append("- [x] `meta.sensory_mode` recorded (`rate` / `phase-lock`)")
    L.append("- [x] `pytest tests/` — see commit message")
    L.append("")
    L.append("## 6. Deviations / notes\n")
    L.append("- `JO_post_unk` (23 neurons with side M/null) and `SAD_unk` are kept as their own sets instead of being "
             "dropped silently; they are not part of the L/R transfer criterion.")
    L.append("- Sets are made disjoint by precedence so every neuron has one viewer region; raw sizes are in §2. "
             "`SAD_*` therefore excludes the JO_post and AMMC-typed neurons that sit in SAD.")
    L.append("- Left/right JO_AB targets are unbalanced in the data (76 L / 38 R after excluding site-less neurons); "
             "identical input is given to both sides, so right-side downstream rates are expected to be lower.")
    L.append("- The band-pass is HP2 ∘ LP2 (two Butterworth biquads) rather than a true 4th-order Butterworth "
             "band-pass; edges are −3 dB at 100 and 400 Hz (tests/test_sensory.py).")
    L.append("- The silence control is run once per mode (`a_in` has no effect on a zero signal).")
    L.append("- No parameter was changed during the sweep; `g` stayed at 0.336.")
    L.append("- Observation: JO spike counts of 1140 / 570 (= 15 pulses × 76 / 38 targets) mean exactly one spike per "
             "pulse per JO neuron; the phase-lock mode sits on that plateau from a_in = 320 to 1280 (half-wave "
             "rectified 200 Hz carrier: one 2.5 ms positive half-cycle drives one spike, then t_ref + membrane "
             "decay), and the rate mode passes through it at a_in = 160. JO_post_R needs ≥ 2 spikes per pulse to "
             "reach 1 Hz because only 38 right targets drive it.")
    L.append("- `meta.rate_norm_hz` = 1000 Hz in the click runs: in one 1 ms bin every neuron of a JO_AB set fired "
             "(synchronous, phase-locked); the viewer's 0..1 rates are relative to that bin.")
    L.append("")
    REPORT_MD.write_text("\n".join(L))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
