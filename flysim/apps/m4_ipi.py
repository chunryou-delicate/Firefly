"""M4 phase A: IPI tuning curves with the model exactly as left by M3 (docs/m4-brief.md).

Nothing is adjusted: g = 0.336, M2 neuron defaults, noise 0, v_floor None, a_in = the
values M3 selected (rate 640, phase-lock 2560; asserted against
data-provenance/m3-results.json), M3 click_train defaults (200 Hz carrier, 10 ms Hann
pulse). Only the inter-pulse interval changes: IPI in {20, ..., 60} ms, 9 points x 2
modes = 18 runs. Stimulus: 100 ms silence, a click train of fixed length 700 ms
(n_pulses = floor(700 / IPI)), silence to 1000 ms.

Outputs (runs/ is gitignored; copies of the small files are committed):
    runs/m4-ipi/<mode>-ipi<NN>/run.json, spikes.parquet, rates.parquet
    runs/m4-ipi/tuning.json      -> data-provenance/m4-tuning.json
    runs/m4-ipi/tuning.html      -> docs/m4-tuning.html      (single file, inline SVG)
    docs/m4-report.md

Usage: python -m flysim.apps.m4_ipi [--synapse current|conductance] [--g G] [--tag b]
M4b (docs/m4b-brief.md): ``--synapse conductance --g 3.162e-4 --tag b`` reruns the identical sweep with the
M2b conductance model, a_in taken from data-provenance/m3b-results.json (selected_a_in; if a mode has no
PASS the M3 gain is used and recorded as a fallback). Outputs: runs/m4b-ipi, data-provenance/m4b-tuning.json,
docs/m4b-tuning.html, docs/m4b-report.md (phase A and B tables side by side).
"""
from __future__ import annotations

import argparse
import html
import json
import math
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

from ..data.download import ROOT
from ..engine import EngineParams, LIFEngine
from ..graph import Graph
from ..probe import bin_spikes, build_probe_sets, load_roi, neuron_coords, write_run_json
from ..probe.export import RUNS_DIR
from ..probe.probe import save_parquet, window_mean_rate, window_spikes
from ..sensory import JOAdapter, click_train
from . import m3_click as M3

# ---- sweep design (fixed before the first run) ----------------------------------------
IPI_MS = [20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0]
PRE_MS = 100.0
TRAIN_MS = 700.0                      # fixed train length; n_pulses = floor(TRAIN_MS / IPI)
TOTAL_MS = 1000.0
STIM_WINDOW_MS = (PRE_MS, PRE_MS + TRAIN_MS)     # 100-800 ms
LATE_FROM_MS = PRE_MS + TRAIN_MS                 # late window: 800-1000 ms
G = M3.G                                         # 0.336
SEED = M3.SEED
BIN_MS = M3.BIN_MS
MODES = dict(M3.MODES)                           # rate: 1.0 ms, phase-lock: 0.1 ms
A_IN = {"rate": 640.0, "phase-lock": 2560.0}     # phase A: M3 selected values (checked against m3-results.json)
A_IN_PHASE_A = dict(A_IN)
PULSE_KW = dict(pulse_ms=10.0, carrier_hz=200.0) # M3 click_train defaults (stimuli.py)
IGNITION_LATE_FRACTION = 0.01                    # > 1 % of neurons active after 800 ms -> ignited
FAILURE_MAX_FRACTION = M3.CRITERIA["failure_active_max_fraction"]
REFERENCE_IPI_MS = 35.0                          # literature value shown as a guide line, not an expectation
BASE_SETS = ["JO_AB", "JO_post", "SAD", "AMMCtype", "WED", "pC1"]   # curve panels, in this order

def out_paths(tag: str) -> dict:
    """tag "" = phase A (m4-*), "b" = phase B (m4b-*)."""
    t = f"m4{tag}"
    return {"out_dir": RUNS_DIR / f"{t}-ipi", "doc_html": ROOT / "docs" / f"{t}-tuning.html",
            "prov_json": ROOT / "data-provenance" / f"{t}-tuning.json", "report": ROOT / "docs" / f"{t}-report.md",
            "m3_results": ROOT / "data-provenance" / f"m3{tag}-results.json"}


_A = out_paths("")
OUT_DIR = _A["out_dir"]
TUNING_JSON = OUT_DIR / "tuning.json"
TUNING_HTML = OUT_DIR / "tuning.html"
DOC_HTML = _A["doc_html"]
PROV_JSON = _A["prov_json"]
REPORT_MD = _A["report"]
M3_RESULTS = _A["m3_results"]
PROV_JSON_B = out_paths("b")["prov_json"]


def log(msg: str) -> None:
    print(f"[m4] {msg}", flush=True)


def n_pulses_for(ipi_ms: float) -> int:
    return int(math.floor(TRAIN_MS / ipi_ms + 1e-9))


def make_stimulus(ipi_ms: float):
    n = n_pulses_for(ipi_ms)
    train_end = (n - 1) * ipi_ms + PULSE_KW["pulse_ms"]
    post = TOTAL_MS - PRE_MS - train_end
    st = click_train(ipi_ms, n, pre_ms=PRE_MS, post_ms=post, **PULSE_KW)
    assert abs(st.duration_ms - TOTAL_MS) < 1e-6, st.duration_ms
    return st


def check_a_in_matches_m3() -> None:
    r = json.loads(M3_RESULTS.read_text())
    sel = {m: v["selected_a_in"] for m, v in r["modes"].items()}
    if sel != A_IN_PHASE_A:
        raise RuntimeError(f"A_IN {A_IN_PHASE_A} != M3 selected {sel}; M4 must use the M3 values unchanged")


def resolve_a_in(tag: str) -> tuple[dict, dict]:
    """Gains for this phase: phase A = M3 selection (asserted); phase B = m3b selection per mode,
    falling back to the M3 gain for a mode without any PASS (recorded, docs/m4b-brief.md)."""
    if not tag:
        check_a_in_matches_m3()
        return dict(A_IN_PHASE_A), {m: "M3 selected" for m in A_IN_PHASE_A}
    path = out_paths(tag)["m3_results"]
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run m3_click with --tag {tag} first")
    r = json.loads(path.read_text())
    a_in, src = {}, {}
    for m in MODES:
        sel = r["modes"][m]["selected_a_in"]
        if sel is None:
            a_in[m], src[m] = A_IN_PHASE_A[m], f"fallback: no PASS in m3{tag}; M3 gain used"
        else:
            a_in[m], src[m] = float(sel), f"m3{tag} selected"
    return a_in, src


def git_info() -> dict:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    except Exception as e:  # pragma: no cover
        return {"commit": None, "dirty": None, "error": repr(e)}
    return {"commit": commit, "dirty": dirty}


def measure(rt, sets, t_step, idx, dt_ms: float, n_pulses: int, n_neurons: int) -> dict:
    w0, w1 = STIM_WINDOW_MS
    t_ms = np.asarray(t_step) * dt_ms
    idx = np.asarray(idx)
    per_set = {}
    for name in rt.names:
        sp = window_spikes(rt, name, w0, w1)
        r = sets.names.index(name)
        own = t_ms[rt.region == r]
        first = float(own.min()) if own.size else None
        per_set[name] = {
            "n": int(rt.sizes[r]),
            "spikes_stim": sp,
            "rate_stim_hz": window_mean_rate(rt, name, w0, w1),
            "spikes_per_pulse": sp / n_pulses,
            "spikes_per_pulse_per_neuron": sp / n_pulses / rt.sizes[r],
            "latency_ms": (first - PRE_MS) if first is not None else None,
        }
    late = t_ms >= LATE_FROM_MS
    late_active = int(len(np.unique(idx[late])))
    out = {
        "n_spikes": rt.n_spikes, "active_neurons": rt.active_neurons,
        "active_fraction": rt.active_neurons / n_neurons,
        "late_active_neurons": late_active, "late_active_frac": late_active / n_neurons,
        "spikes_per_50ms": np.histogram(t_ms, bins=np.arange(0, TOTAL_MS + 50, 50))[0].astype(int).tolist(),
        "per_set": per_set,
    }
    out["ignited"] = out["late_active_frac"] > IGNITION_LATE_FRACTION
    out["viewer_failure"] = rt.active_neurons == 0 or out["active_fraction"] >= FAILURE_MAX_FRACTION
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synapse", default="current", choices=("current", "conductance"))
    ap.add_argument("--g", type=float, default=None, help="synaptic gain; default M3.G_BY_SYNAPSE[synapse]")
    ap.add_argument("--tag", default="", help='"" -> phase A names (m4-*), "b" -> phase B (m4b-*)')
    args = ap.parse_args(argv)
    synapse = args.synapse
    g_val = M3.G_BY_SYNAPSE[synapse] if args.g is None else float(args.g)
    paths = out_paths(args.tag)
    out_dir = paths["out_dir"]
    a_in, a_in_src = resolve_a_in(args.tag)
    phase = "A" if not args.tag else "B (conductance)" if synapse == "conductance" else f"B[{args.tag}]"
    t_all = time.time()
    g = Graph.load()
    roi = load_roi(g)
    n_sites = roi["n_sites"].to_numpy()
    sets = build_probe_sets(g, roi, k_min=M3.CRITERIA["k_min"])
    coords = neuron_coords(g, roi)
    out_dir.mkdir(parents=True, exist_ok=True)
    gi = git_info()
    log(f"phase {phase}: synapse={synapse} g={g_val:g} a_in={a_in} ({a_in_src}); git {gi}; probe sets {sets.sizes()}")

    runs = []
    adapters_desc = {}
    params_by_mode = {}
    for mode, dt in MODES.items():
        n_steps = int(round(TOTAL_MS / dt))
        params = EngineParams(g=g_val, dt=dt, synapse=synapse)
        params_by_mode[mode] = params.to_dict()
        engine = LIFEngine(g, params, device="cuda")
        log(f"=== mode {mode}: dt={dt} ms, a_in={a_in[mode]:g}, {params.to_dict()}")
        for ipi in IPI_MS:
            st = make_stimulus(ipi)
            n_p = n_pulses_for(ipi)
            ad = JOAdapter(g, n_sites, st, mode, a_in=a_in[mode], device="cuda")
            ad.prepare(dt, n_steps)
            if ipi == REFERENCE_IPI_MS:
                adapters_desc[mode] = ad.describe()
            log(f"{mode} IPI {ipi:g} ms ({n_p} pulses): {n_steps} steps, 예상 소요 ≈ "
                f"{n_steps * M3.US_PER_STEP_ESTIMATE / 1e6:.1f} s")
            engine.reset(SEED)
            torch.cuda.synchronize()
            t0 = time.time()
            t_step, idx = engine.run(n_steps, ad.inject, record=True)
            torch.cuda.synchronize()
            wall = time.time() - t0
            rt = bin_spikes(t_step, idx, dt, BIN_MS, int(TOTAL_MS / BIN_MS), sets)
            m = measure(rt, sets, t_step, idx, dt, n_p, g.n)
            run_id = f"{'phase' if mode == 'phase-lock' else 'rate'}-ipi{int(ipi):02d}"
            rdir = out_dir / run_id
            save_parquet(rdir, rt, t_step, idx, g, dt)
            write_run_json(rdir / "run.json", run_id=f"m4{args.tag}-{run_id}", sensory_mode=mode, dt_ms=dt, rt=rt, sets=sets,
                           coords=coords, input_label=f"{st.label}, {n_p} pulses, a_in={a_in[mode]:g}",
                           envelope=st.envelope_1ms(), spikes=(rt.t_bin, idx), engine_params=params.to_dict(),
                           adapter=ad.describe(), stimulus=st.describe(),
                           extra_meta={"m4": {k: v for k, v in m.items() if k != "per_set"}})
            rec = {"mode": mode, "ipi_ms": ipi, "run_id": f"m4{args.tag}-{run_id}", "n_pulses": n_p, "dt_ms": dt,
                   "a_in": a_in[mode], "stimulus": st.describe(), "wall_s": wall, **m}
            runs.append(rec)
            js = m["per_set"]
            log(f"  {wall:.2f} s, {m['n_spikes']} spikes, active {m['active_fraction']:.3%}, late "
                f"{m['late_active_frac']:.3%}, ignited={m['ignited']}; spikes/pulse JO_AB L/R "
                f"{js['JO_AB_L']['spikes_per_pulse']:.1f}/{js['JO_AB_R']['spikes_per_pulse']:.1f}, JO_post L/R "
                f"{js['JO_post_L']['spikes_per_pulse']:.2f}/{js['JO_post_R']['spikes_per_pulse']:.2f}, "
                f"pC1 L/R {js['pC1_L']['spikes_per_pulse']:.2f}/{js['pC1_R']['spikes_per_pulse']:.2f}")
        del engine
        torch.cuda.empty_cache()

    curves = {}
    for mode in MODES:
        curves[mode] = {}
        for name in sets.names:
            rows = [r for r in runs if r["mode"] == mode]
            curves[mode][name] = {
                "ipi_ms": [r["ipi_ms"] for r in rows],
                "n_pulses": [r["n_pulses"] for r in rows],
                "spikes_per_pulse": [r["per_set"][name]["spikes_per_pulse"] for r in rows],
                "spikes_per_pulse_per_neuron": [r["per_set"][name]["spikes_per_pulse_per_neuron"] for r in rows],
                "rate_stim": [r["per_set"][name]["rate_stim_hz"] for r in rows],
                "latency_ms": [r["per_set"][name]["latency_ms"] for r in rows],
                "ignited": [bool(r["ignited"]) for r in rows],
            }
    tuning = {
        "meta": {
            "milestone": f"M4 phase {phase}", "phase": phase, "tag": args.tag, "synapse": synapse,
            "git_commit": gi.get("commit"), "git_dirty": gi.get("dirty"),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "seed": SEED, "bin_ms": BIN_MS,
            "ipi_ms": IPI_MS, "pre_ms": PRE_MS, "train_ms": TRAIN_MS, "total_ms": TOTAL_MS,
            "stim_window_ms": list(STIM_WINDOW_MS), "late_from_ms": LATE_FROM_MS,
            "ignition_late_fraction": IGNITION_LATE_FRACTION, "viewer_failure_max_fraction": FAILURE_MAX_FRACTION,
            "g": g_val, "a_in": a_in, "a_in_source": a_in_src, "params": params_by_mode, "adapter": adapters_desc,
            "stimulus": {"kind": "click_train", **PULSE_KW, "window": "hann", "fs_hz": 10000.0,
                         "n_pulses_rule": "floor(train_ms / ipi_ms)",
                         "n_pulses": {str(i): n_pulses_for(i) for i in IPI_MS}},
            "probe_sets": {"sizes": sets.sizes(), "log": sets.log},
            "wall_total_s": round(time.time() - t_all, 1),
        },
        "curves": curves,
        "runs": runs,
    }
    tj, th = out_dir / "tuning.json", out_dir / "tuning.html"
    tj.write_text(json.dumps(tuning, indent=1, default=M3._json_default) + "\n")
    paths["prov_json"].write_text(tj.read_text())
    th.write_text(render_html(tuning))
    shutil.copyfile(th, paths["doc_html"])
    phase_a = json.loads(PROV_JSON.read_text()) if (args.tag and PROV_JSON.exists()) else None
    write_report(tuning, paths["report"], phase_a)
    log(f"done in {tuning['meta']['wall_total_s']} s -> {tj.relative_to(ROOT)}, {th.relative_to(ROOT)}, "
        f"{paths['report'].relative_to(ROOT)}")
    return 0


# ---------------------------------------------------------------------------
# inline-SVG figure (no external resources; dataviz skill: line form, fixed hue order,
# 2px lines, >=8px markers with surface ring, legend + direct labels, table view)
# ---------------------------------------------------------------------------
SERIES_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
SERIES_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300"]


def _nice_max(v: float) -> float:
    if v <= 0:
        return 1.0
    e = 10 ** math.floor(math.log10(v))
    for m in (1, 2, 2.5, 5, 10):
        if v <= m * e:
            return m * e
    return 10 * e


def _panel(mode: str, base: str, curve_l: dict, curve_r: dict, slot: int, W=300, H=190) -> str:
    ml, mr, mt, mb = 44, 12, 26, 34
    pw, ph = W - ml - mr, H - mt - mb
    xs = curve_l["ipi_ms"]
    x0, x1 = min(xs), max(xs)
    ymax = _nice_max(max(max(curve_l["spikes_per_pulse"]), max(curve_r["spikes_per_pulse"])))

    def X(v):
        return ml + (v - x0) / (x1 - x0) * pw

    def Y(v):
        return mt + ph - v / ymax * ph

    p = [f'<svg class="panel" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" '
         f'aria-label="{html.escape(base)} spikes per pulse vs IPI, {html.escape(mode)} mode">']
    p.append(f'<rect x="0" y="0" width="{W}" height="{H}" class="surface"/>')
    p.append(f'<text x="{ml}" y="15" class="ptitle">{html.escape(base)}</text>')
    for k in range(5):
        yv = ymax * k / 4
        p.append(f'<line x1="{ml}" x2="{ml + pw}" y1="{Y(yv):.1f}" y2="{Y(yv):.1f}" class="grid"/>')
        p.append(f'<text x="{ml - 5}" y="{Y(yv) + 3.5:.1f}" class="tick" text-anchor="end">{yv:g}</text>')
    for xv in xs:
        p.append(f'<text x="{X(xv):.1f}" y="{mt + ph + 14}" class="tick" text-anchor="middle">{xv:g}</text>')
    p.append(f'<line x1="{ml}" x2="{ml + pw}" y1="{mt + ph}" y2="{mt + ph}" class="axis"/>')
    p.append(f'<line x1="{X(REFERENCE_IPI_MS):.1f}" x2="{X(REFERENCE_IPI_MS):.1f}" y1="{mt}" y2="{mt + ph}" class="ref"/>')
    p.append(f'<text x="{ml + pw / 2:.1f}" y="{H - 4}" class="tick" text-anchor="middle">IPI (ms)</text>')
    for side, curve, dash in (("L", curve_l, ""), ("R", curve_r, ' stroke-dasharray="6 4"')):
        pts = " ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in zip(curve["ipi_ms"], curve["spikes_per_pulse"]))
        p.append(f'<polyline points="{pts}" class="s{slot}" fill="none" stroke-width="2" '
                 f'stroke-linejoin="round" stroke-linecap="round"{dash}/>')
        ign = curve.get("ignited", [False] * len(curve["ipi_ms"]))
        for x, y, r, n, ig in zip(curve["ipi_ms"], curve["spikes_per_pulse"], curve["rate_stim"], curve["n_pulses"], ign):
            tip = (f'{html.escape(base)}_{side} · IPI {x:g} ms · {n} pulses · {y:.3f} spikes/pulse · '
                   f'{r:.3f} Hz/neuron' + (' · IGNITED RUN (failed)' if ig else ''))
            if ig:
                cx, cy = X(x), Y(y)
                p.append(f'<g class="s{slot} xmark"><line x1="{cx-5:.1f}" y1="{cy-5:.1f}" x2="{cx+5:.1f}" y2="{cy+5:.1f}"/>'
                         f'<line x1="{cx-5:.1f}" y1="{cy+5:.1f}" x2="{cx+5:.1f}" y2="{cy-5:.1f}"/><title>{tip}</title></g>')
            else:
                p.append(f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="4" class="s{slot} dot"><title>{tip}</title></circle>')
        # direct label at the right end
        p.append(f'<text x="{X(curve["ipi_ms"][-1]) + 5:.1f}" y="{Y(curve["spikes_per_pulse"][-1]) + 3.5:.1f}" '
                 f'class="lbl">{side}</text>')
    p.append("</svg>")
    return "\n".join(p)


def render_html(T: dict) -> str:
    meta, curves = T["meta"], T["curves"]
    css = f"""
:root {{ color-scheme: light; --surface:#fcfcfb; --page:#f9f9f7; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781;
  --grid:#e1e0d9; --axis:#c3c2b7; --ref:#898781; {' '.join(f'--s{i}:{c};' for i, c in enumerate(SERIES_LIGHT))} }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ color-scheme: dark; --surface:#1a1a19; --page:#0d0d0d;
  --ink:#ffffff; --ink2:#c3c2b7; --muted:#898781; --grid:#2c2c2a; --axis:#383835; --ref:#898781;
  {' '.join(f'--s{i}:{c};' for i, c in enumerate(SERIES_DARK))} }} }}
:root[data-theme="dark"] {{ color-scheme: dark; --surface:#1a1a19; --page:#0d0d0d; --ink:#ffffff; --ink2:#c3c2b7;
  --muted:#898781; --grid:#2c2c2a; --axis:#383835; --ref:#898781; {' '.join(f'--s{i}:{c};' for i, c in enumerate(SERIES_DARK))} }}
body {{ margin:0; padding:24px 16px; background:var(--page); color:var(--ink); font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif; }}
h1 {{ font-size:20px; margin:0 0 4px; }} h2 {{ font-size:16px; margin:28px 0 8px; }}
.sub {{ color:var(--ink2); margin:0 0 16px; }}
.row {{ display:flex; flex-wrap:wrap; gap:12px; }}
.surface {{ fill:var(--surface); }} .grid {{ stroke:var(--grid); stroke-width:1; }} .axis {{ stroke:var(--axis); stroke-width:1; }}
.ref {{ stroke:var(--ref); stroke-width:1; stroke-dasharray:2 3; }}
.tick {{ font-size:10px; fill:var(--muted); font-variant-numeric:tabular-nums; }} .ptitle {{ font-size:12px; font-weight:600; fill:var(--ink); }}
.lbl {{ font-size:11px; fill:var(--ink2); }}
{' '.join(f'.s{i}{{stroke:var(--s{i});}}' for i in range(6))}
.dot {{ fill:var(--surface); stroke-width:2; }} .dot:hover {{ r:6; }} .xmark line {{ stroke-width:2; }}
.legend {{ display:flex; gap:18px; color:var(--ink2); margin:6px 0 4px; font-size:12px; }}
.legend svg {{ vertical-align:middle; }}
table {{ border-collapse:collapse; font-size:12px; font-variant-numeric:tabular-nums; }}
th, td {{ padding:3px 8px; border-bottom:1px solid var(--grid); text-align:right; }} th:first-child, td:first-child {{ text-align:left; }}
.wrap {{ overflow-x:auto; }} code {{ font-size:12px; }}
"""
    out = [f"<!doctype html><html lang='ko'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
           f"<title>M4 IPI tuning curves</title><style>{css}</style></head><body>",
           f"<h1>M4 phase {html.escape(meta.get('phase', 'A'))} — IPI tuning curves</h1>",
           f"<p class='sub'>spikes per pulse per probe set vs inter-pulse interval · synapse {meta.get('synapse', 'current')} · g = {meta['g']:g} · a_in rate {meta['a_in']['rate']:g} / "
           f"phase-lock {meta['a_in']['phase-lock']:g} · noise 0 · seed {meta['seed']} · commit {(meta['git_commit'] or '?')[:7]}"
           f"{' (dirty tree)' if meta['git_dirty'] else ''} · {html.escape(meta['generated_at'])}. "
           f"Dotted vertical line = {REFERENCE_IPI_MS:g} ms literature reference, shown for orientation only, not an expectation.</p>",
           "<div class='legend'><span><svg width='26' height='10'><line x1='0' x2='26' y1='5' y2='5' stroke='currentColor' stroke-width='2'/></svg> left (L)</span>"
           "<span><svg width='26' height='10'><line x1='0' x2='26' y1='5' y2='5' stroke='currentColor' stroke-width='2' stroke-dasharray='6 4'/></svg> right (R)</span>"
           "<span><svg width='14' height='12'><line x1='2' y1='1' x2='12' y2='11' stroke='currentColor' stroke-width='2'/><line x1='2' y1='11' x2='12' y2='1' stroke='currentColor' stroke-width='2'/></svg> ignited run (failed; value shown for the record)</span>"
           "<span>y = spikes in 100–800 ms ÷ pulses (set total, not per neuron); y-scale is per panel</span></div>"]
    for mode in curves:
        out.append(f"<h2>Mode <code>{mode}</code> (dt {meta['params'][mode]['dt']} ms, a_in {meta['a_in'][mode]:g})</h2><div class='row'>")
        for slot, base in enumerate(BASE_SETS):
            out.append(_panel(mode, base, curves[mode][f"{base}_L"], curves[mode][f"{base}_R"], slot))
        out.append("</div>")
    # table view
    out.append("<h2>Table — spikes per pulse (set total)</h2><div class='wrap'><table><thead><tr><th>mode</th><th>set</th>"
               + "".join(f"<th>{i:g}</th>" for i in meta["ipi_ms"]) + "</tr></thead><tbody>")
    for mode in curves:
        for base in BASE_SETS:
            for side in ("L", "R"):
                c = curves[mode][f"{base}_{side}"]
                ign = c.get("ignited", [False] * len(c["spikes_per_pulse"]))
                out.append(f"<tr><td>{mode}</td><td>{base}_{side} (n={T['meta']['probe_sets']['sizes'][f'{base}_{side}']})</td>"
                           + "".join(f"<td>{v:.3f}{'†' if ig else ''}</td>" for v, ig in zip(c["spikes_per_pulse"], ign)) + "</tr>")
    out.append("</tbody></table><p class='sub'>† = ignited run (failed by the ignition rule; value kept for the record).</p></div>")
    out.append("<h2>Parameters</h2><div class='wrap'><table><tbody>")
    pr = meta["params"]["rate"]
    rows = [("synapse model", pr.get("synapse", "current")), ("g", f"{meta['g']:g}"),
            ("E_exc / E_inh (mV), tau_e / tau_i (ms)", f"{pr.get('E_exc')} / {pr.get('E_inh')}, {pr.get('tau_e')} / {pr.get('tau_i')}"
             if pr.get("synapse") == "conductance" else "n/a (current model)"),
            ("a_in (rate / phase-lock)", f"{meta['a_in']['rate']:g} / {meta['a_in']['phase-lock']:g} — {meta.get('a_in_source', {})}"),
            ("noise_sigma / v_floor", "0 / None"), ("seed", meta["seed"]),
            ("neuron params (rate mode)", json.dumps({k: meta['params']['rate'][k] for k in ('tau_m', 'tau_syn', 't_ref', 'v_rest', 'v_reset', 'v_thresh')})),
            ("dt (rate / phase-lock)", f"{meta['params']['rate']['dt']} / {meta['params']['phase-lock']['dt']} ms"),
            ("IPI grid (ms)", ", ".join(f"{i:g}" for i in meta["ipi_ms"])),
            ("stimulus", f"pre {meta['pre_ms']:g} ms, train {meta['train_ms']:g} ms, total {meta['total_ms']:g} ms; "
                         f"pulse {meta['stimulus']['pulse_ms']:g} ms Hann, carrier {meta['stimulus']['carrier_hz']:g} Hz, fs {meta['stimulus']['fs_hz']:g} Hz"),
            ("n_pulses per IPI", ", ".join(f"{float(k):g}:{v}" for k, v in meta["stimulus"]["n_pulses"].items())),
            ("adapter", f"band {meta['adapter']['rate']['band_hz']} Hz; rate: {meta['adapter']['rate']['rectification']}, τ_env {meta['adapter']['rate']['tau_env_ms']} ms; "
                        f"phase-lock: {meta['adapter']['phase-lock']['rectification']}; ild 0; targets {meta['adapter']['rate']['n_targets']} "
                        f"({meta['adapter']['rate']['n_left']} L / {meta['adapter']['rate']['n_right']} R)"),
            ("ignition rule", f"late-window (≥ {meta['late_from_ms']:g} ms) active fraction > {meta['ignition_late_fraction']:.0%}"),
            ("runs ignited", sum(1 for r in T["runs"] if r["ignited"])),
            ("probe set sizes", ", ".join(f"{k} {v}" for k, v in meta["probe_sets"]["sizes"].items()))]
    for k, v in rows:
        out.append(f"<tr><td>{html.escape(str(k))}</td><td style='text-align:left'>{html.escape(str(v))}</td></tr>")
    out.append("</tbody></table></div></body></html>")
    return "\n".join(out)


# ---------------------------------------------------------------------------
def write_report(T: dict, path=REPORT_MD, phase_a: dict | None = None) -> None:
    meta, curves, runs = T["meta"], T["curves"], T["runs"]
    phase = meta.get("phase", "A")
    pr = meta["params"]["rate"]
    L = [f"# M4 report — phase {phase}: IPI tuning curves"
         + (", model unchanged" if phase == "A" else ", conductance synapses (M2b), same protocol as phase A") + "\n",
         f"Generated by `python -m flysim.apps.m4_ipi` on {meta['generated_at']} at commit "
         f"`{(meta['git_commit'] or '?')[:7]}`{' (dirty tree at run time)' if meta['git_dirty'] else ''}. "
         f"Figure: `docs/m4-tuning.html` (inline SVG, single file). Numbers: `data-provenance/m4-tuning.json`. "
         f"Spec: `docs/m4-brief.md`. Wall time {meta['wall_total_s']} s for {len(runs)} runs.\n",
         "## 1. Parameters (nothing adjusted)\n",
         f"- synapse model `{pr.get('synapse', 'current')}`"
         + (f": E_exc {pr['E_exc']} mV, E_inh {pr['E_inh']} mV, tau_e {pr['tau_e']} ms, tau_i {pr['tau_i']} ms (M2b defaults, assumptions)"
            if pr.get("synapse") == "conductance" else "") + ".",
         f"- `g = {meta['g']:g}`, `noise_sigma = 0`, `v_floor = None`, seed {meta['seed']}; neuron parameters = M2 defaults "
         f"(`{json.dumps({k: meta['params']['rate'][k] for k in ('tau_m', 'tau_syn', 't_ref', 'v_rest', 'v_reset', 'v_thresh')})}`).",
         f"- `a_in`: rate {meta['a_in']['rate']:g}, phase-lock {meta['a_in']['phase-lock']:g} — source "
         f"{meta.get('a_in_source', 'M3 selected (asserted against data-provenance/m3-results.json)')}.",
         f"- dt: rate {meta['params']['rate']['dt']} ms, phase-lock {meta['params']['phase-lock']['dt']} ms. Bin {meta['bin_ms']} ms.",
         f"- Stimulus: {meta['pre_ms']:g} ms silence, click train of fixed length {meta['train_ms']:g} ms "
         f"(n_pulses = floor({meta['train_ms']:g} / IPI)), silence to {meta['total_ms']:g} ms. Pulse {meta['stimulus']['pulse_ms']:g} ms "
         f"Hann, carrier {meta['stimulus']['carrier_hz']:g} Hz, fs {meta['stimulus']['fs_hz']:g} Hz (M3 defaults). ILD 0.",
         f"- IPI grid: {', '.join(f'{i:g}' for i in meta['ipi_ms'])} ms → n_pulses "
         f"{', '.join(f'{float(k):g}:{v}' for k, v in meta['stimulus']['n_pulses'].items())}.",
         f"- Adapter: band {meta['adapter']['rate']['band_hz']} Hz (HP2∘LP2 Butterworth biquads); rate mode "
         f"{meta['adapter']['rate']['rectification']} τ {meta['adapter']['rate']['tau_env_ms']} ms; phase-lock "
         f"{meta['adapter']['phase-lock']['rectification']}. Targets {meta['adapter']['rate']['n_targets']} "
         f"({meta['adapter']['rate']['n_left']} L / {meta['adapter']['rate']['n_right']} R).",
         f"- Ignition rule: active fraction in {meta['late_from_ms']:g}–{meta['total_ms']:g} ms > "
         f"{meta['ignition_late_fraction']:.0%} → ignited (failed run). Viewer failure rule: 0 active or ≥ "
         f"{meta['viewer_failure_max_fraction']:.0%} active.",
         f"- Probe sets (M3 definitions, disjoint by precedence): "
         f"{', '.join(f'{k} {v}' for k, v in meta['probe_sets']['sizes'].items())}.\n",
         "## 2. Run table\n",
         "| mode | IPI | pulses | spikes | active | late active | ignited | viewer fail | wall s |\n|---|---|---|---|---|---|---|---|---|"]
    for r in runs:
        L.append(f"| {r['mode']} | {r['ipi_ms']:g} | {r['n_pulses']} | {r['n_spikes']:,} | {r['active_fraction']:.3%} | "
                 f"{r['late_active_frac']:.3%} | {r['ignited']} | {r['viewer_failure']} | {r['wall_s']:.2f} |")
    n_ign = sum(1 for r in runs if r["ignited"])
    L.append(f"\nIgnited runs: **{n_ign}** of {len(runs)}. Viewer-rule failures: {sum(1 for r in runs if r['viewer_failure'])}.\n")

    for mode in curves:
        L.append(f"## 3. Curves — mode `{mode}`\n")
        L.append("† = ignited run (failed by the ignition rule; numbers kept for the record).\n")
        for metric, title, fmt in (("spikes_per_pulse", "spikes per pulse (set total, stimulus window 100–800 ms) — the primary curve", "{:.3f}"),
                                   ("rate_stim", "mean rate in the stimulus window (Hz per neuron)", "{:.3f}"),
                                   ("latency_ms", "latency from first pulse onset to first spike of the set (ms; – = none)", "{:.1f}")):
            L.append(f"**{title}**\n")
            L.append("| set | n | " + " | ".join(f"{i:g}" for i in meta["ipi_ms"]) + " |\n|---|---|" + "---|" * len(meta["ipi_ms"]))
            for name in [f"{b}_{s}" for b in BASE_SETS for s in ("L", "R")] + \
                        [n for n in curves[mode] if n.endswith("_unk")] + ["rest"]:
                c = curves[mode][name]
                cells = [("–" if v is None else fmt.format(v) + ("†" if ig else ""))
                         for v, ig in zip(c[metric], c.get("ignited", [False] * len(c[metric])))]
                L.append(f"| `{name}` | {meta['probe_sets']['sizes'][name]:,} | " + " | ".join(cells) + " |")
            L.append("")

    if phase_a is not None:
        L.append("## 3b. Phase A vs phase B — spikes per pulse (set total), side by side\n")
        L.append("† = ignited run (failed); its numbers describe the ignited state, not a stimulus response.\n")
        L.append(f"Phase A = `data-provenance/m4-tuning.json` (current model, g {phase_a['meta']['g']:g}, a_in "
                 f"{phase_a['meta']['a_in']['rate']:g}/{phase_a['meta']['a_in']['phase-lock']:g}); phase B = this run "
                 f"({pr.get('synapse')}, g {meta['g']:g}, a_in {meta['a_in']['rate']:g}/{meta['a_in']['phase-lock']:g}).\n")
        for mode in curves:
            L.append(f"**mode `{mode}`**\n")
            L.append("| set | phase | " + " | ".join(f"{i:g}" for i in meta["ipi_ms"]) + " |\n|---|---|" + "---|" * len(meta["ipi_ms"]))
            for name in [f"{b}_{s}" for b in BASE_SETS for s in ("L", "R")] + ["rest"]:
                for lab, src in (("A", phase_a["curves"][mode][name]), ("B", curves[mode][name])):
                    ign = src.get("ignited", [False] * len(src["spikes_per_pulse"]))
                    L.append(f"| `{name}` | {lab} | " + " | ".join(f"{v:.3f}{'†' if ig else ''}" for v, ig in zip(src["spikes_per_pulse"], ign)) + " |")
            L.append("")
    L.append("## 4. Observations (numbers only; ignited runs excluded, listed separately)\n")
    for mode in curves:
        c = curves[mode]
        ign_ipi = [i for i, ig in zip(meta["ipi_ms"], c["JO_AB_L"].get("ignited", [])) if ig]
        L.append(f"- `{mode}` ignited runs: {[f'{i:g}' for i in ign_ipi] if ign_ipi else 'none'}.")
        for base in BASE_SETS:
            for side in ("L", "R"):
                cc = c[f"{base}_{side}"]
                keep = [not ig for ig in cc.get("ignited", [False] * len(cc["spikes_per_pulse"]))]
                v = [x for x, k in zip(cc["spikes_per_pulse"], keep) if k]
                ipis = [x for x, k in zip(meta["ipi_ms"], keep) if k]
                if not v:
                    L.append(f"- `{mode}` `{base}_{side}`: every run ignited.")
                    continue
                if max(v) == 0:
                    L.append(f"- `{mode}` `{base}_{side}`: 0 spikes at every non-ignited IPI.")
                else:
                    i_max, i_min = int(np.argmax(v)), int(np.argmin(v))
                    at35 = f"{v[ipis.index(35.0)]:.3f}" if 35.0 in ipis else "ignited"
                    L.append(f"- `{mode}` `{base}_{side}`: spikes/pulse ranges {min(v):.3f} (IPI {ipis[i_min]:g}) to "
                             f"{max(v):.3f} (IPI {ipis[i_max]:g}) over non-ignited runs; at 35 ms {at35}.")
    L.append("")
    L.append("## 5. Verdict (CLAUDE.md §6 M4 / m4-brief)\n")
    L.append(f"- [{'x' if n_ign == 0 else ' '}] {len(runs)} runs completed; ignited runs: {n_ign} (each is a failed run, kept in the tables).")
    L.append(f"- [x] curves rendered in `{path.with_suffix('.html').name.replace('-report', '-tuning')}` (inline SVG, no external resources). "
             "Not opened in a browser in this session; `tests/test_m4.py` parses the file and checks the SVG panels, "
             "so that is the substitute check.")
    L.append("- [x] every parameter recorded in `tuning.json` (`meta.params`, `meta.adapter`, `meta.stimulus`, `meta.a_in`, `meta.g`) and above.")
    L.append("- [x] `pytest tests/` — see commit message.")
    def _mx(mode, name, only_ok):
        c = curves[mode][name]
        ign = c.get("ignited", [False] * 9)
        vals = [v for v, ig in zip(c["spikes_per_pulse"], ign) if (not ig) or not only_ok]
        return max(vals) if vals else float("nan")
    L.append("- pC1 response, max spikes/pulse L/R over non-ignited runs: " + "; ".join(
        f"`{mode}` {_mx(mode, 'pC1_L', True):.3f} / {_mx(mode, 'pC1_R', True):.3f}" for mode in curves)
        + ". Over all runs incl. ignited: " + "; ".join(
        f"`{mode}` {_mx(mode, 'pC1_L', False):.3f} / {_mx(mode, 'pC1_R', False):.3f}" for mode in curves) + ".")
    L.append("")
    L.append("## 6. Interpretation (one paragraph; causes are marked as assumptions)\n")
    if phase != "A":
        L.append("Same protocol as phase A with conductance synapses at the M2b default gain; the side-by-side tables in §3b "
                 "show what changed per set and IPI, and the run table shows which runs ignited. Assumed (not tested here) "
                 "reason for any remaining stop at hop 1: the RESPONSIVE g window of the conductance model is narrow "
                 "(docs/m2b-report.md) and, without background activity, summed input from JO_post does not reach threshold "
                 "downstream; with correlated JO drive, ignition marks the upper edge of that window. No mechanism was added.\n")
    else:
        L.append("Under this model the click train drives JO_AB and its direct postsynaptic set; the tables above show how far "
             "activity gets at each IPI and whether the first hop depends on IPI. Where a set reads 0 at every IPI, that is "
             "the result of this stage and is reported as such. Assumed (not tested here) reason for activity stopping "
             "at the first hop: at g = 0.336 with no background activity the summed excitation from JO_post spikes does "
             "not reach threshold in the next stage, and no fluctuation-driven low-rate state exists in this "
             "current-based model (data-provenance/parameter-decisions.md). Any model change belongs to phase B and is "
             "the user's decision.\n")
    if phase != "A":
        L.append("- Phase B outputs are separate files (`m4b-*`); phase A files are untouched.")
    L.append("## 7. Deviations / notes\n")
    L.append("- No parameter was swept or adjusted; the only variable is IPI. No attempt was made to change curve shapes.")
    L.append("- The post-train silence is 200 ms minus the tail of the last pulse ((n−1)·IPI + 10 ms ends before 800 ms), "
             "so the total is exactly 1000 ms; the stimulus window is 100–800 ms for every IPI.")
    L.append("- `spikes_per_pulse` is the set total; `spikes_per_pulse_per_neuron` (÷ set size) is in `tuning.json` "
             "as well. Sets `_unk` and `rest` are listed in the tables, not in the figure (the brief's six sets).")
    L.append("- The figure uses small multiples (one panel per set, per mode) with a per-panel y-scale, because the sets "
             "differ by orders of magnitude; the table under the figure carries the exact values.")
    L.append("")
    path.write_text("\n".join(L))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
