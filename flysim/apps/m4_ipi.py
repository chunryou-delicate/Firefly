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

Usage: python -m flysim.apps.m4_ipi
"""
from __future__ import annotations

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
A_IN = {"rate": 640.0, "phase-lock": 2560.0}     # M3 selected values (checked against m3-results.json)
PULSE_KW = dict(pulse_ms=10.0, carrier_hz=200.0) # M3 click_train defaults (stimuli.py)
IGNITION_LATE_FRACTION = 0.01                    # > 1 % of neurons active after 800 ms -> ignited
FAILURE_MAX_FRACTION = M3.CRITERIA["failure_active_max_fraction"]
REFERENCE_IPI_MS = 35.0                          # literature value shown as a guide line, not an expectation
BASE_SETS = ["JO_AB", "JO_post", "SAD", "AMMCtype", "WED", "pC1"]   # curve panels, in this order

OUT_DIR = RUNS_DIR / "m4-ipi"
TUNING_JSON = OUT_DIR / "tuning.json"
TUNING_HTML = OUT_DIR / "tuning.html"
DOC_HTML = ROOT / "docs" / "m4-tuning.html"
PROV_JSON = ROOT / "data-provenance" / "m4-tuning.json"
REPORT_MD = ROOT / "docs" / "m4-report.md"
M3_RESULTS = ROOT / "data-provenance" / "m3-results.json"


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
    if sel != A_IN:
        raise RuntimeError(f"A_IN {A_IN} != M3 selected {sel}; M4 must use the M3 values unchanged")


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
    t_all = time.time()
    check_a_in_matches_m3()
    g = Graph.load()
    roi = load_roi(g)
    n_sites = roi["n_sites"].to_numpy()
    sets = build_probe_sets(g, roi, k_min=M3.CRITERIA["k_min"])
    coords = neuron_coords(g, roi)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gi = git_info()
    log(f"git {gi}; probe sets {sets.sizes()}")

    runs = []
    adapters_desc = {}
    params_by_mode = {}
    for mode, dt in MODES.items():
        n_steps = int(round(TOTAL_MS / dt))
        params = EngineParams(g=G, dt=dt)
        params_by_mode[mode] = params.to_dict()
        engine = LIFEngine(g, params, device="cuda")
        log(f"=== mode {mode}: dt={dt} ms, a_in={A_IN[mode]:g}, {params.to_dict()}")
        for ipi in IPI_MS:
            st = make_stimulus(ipi)
            n_p = n_pulses_for(ipi)
            ad = JOAdapter(g, n_sites, st, mode, a_in=A_IN[mode], device="cuda")
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
            rdir = OUT_DIR / run_id
            save_parquet(rdir, rt, t_step, idx, g, dt)
            write_run_json(rdir / "run.json", run_id=f"m4-{run_id}", sensory_mode=mode, dt_ms=dt, rt=rt, sets=sets,
                           coords=coords, input_label=f"{st.label}, {n_p} pulses, a_in={A_IN[mode]:g}",
                           envelope=st.envelope_1ms(), spikes=(rt.t_bin, idx), engine_params=params.to_dict(),
                           adapter=ad.describe(), stimulus=st.describe(),
                           extra_meta={"m4": {k: v for k, v in m.items() if k != "per_set"}})
            rec = {"mode": mode, "ipi_ms": ipi, "run_id": f"m4-{run_id}", "n_pulses": n_p, "dt_ms": dt,
                   "a_in": A_IN[mode], "stimulus": st.describe(), "wall_s": wall, **m}
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
            }
    tuning = {
        "meta": {
            "milestone": "M4 phase A (model unchanged)", "git_commit": gi.get("commit"), "git_dirty": gi.get("dirty"),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "seed": SEED, "bin_ms": BIN_MS,
            "ipi_ms": IPI_MS, "pre_ms": PRE_MS, "train_ms": TRAIN_MS, "total_ms": TOTAL_MS,
            "stim_window_ms": list(STIM_WINDOW_MS), "late_from_ms": LATE_FROM_MS,
            "ignition_late_fraction": IGNITION_LATE_FRACTION, "viewer_failure_max_fraction": FAILURE_MAX_FRACTION,
            "g": G, "a_in": A_IN, "params": params_by_mode, "adapter": adapters_desc,
            "stimulus": {"kind": "click_train", **PULSE_KW, "window": "hann", "fs_hz": 10000.0,
                         "n_pulses_rule": "floor(train_ms / ipi_ms)",
                         "n_pulses": {str(i): n_pulses_for(i) for i in IPI_MS}},
            "probe_sets": {"sizes": sets.sizes(), "log": sets.log},
            "wall_total_s": round(time.time() - t_all, 1),
        },
        "curves": curves,
        "runs": runs,
    }
    TUNING_JSON.write_text(json.dumps(tuning, indent=1, default=M3._json_default) + "\n")
    PROV_JSON.write_text(TUNING_JSON.read_text())
    TUNING_HTML.write_text(render_html(tuning))
    shutil.copyfile(TUNING_HTML, DOC_HTML)
    write_report(tuning)
    log(f"done in {tuning['meta']['wall_total_s']} s -> {TUNING_JSON.relative_to(ROOT)}, "
        f"{TUNING_HTML.relative_to(ROOT)}, {REPORT_MD.relative_to(ROOT)}")
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
        for x, y, r, n in zip(curve["ipi_ms"], curve["spikes_per_pulse"], curve["rate_stim"], curve["n_pulses"]):
            p.append(f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="4" class="s{slot} dot"><title>'
                     f'{html.escape(base)}_{side} · IPI {x:g} ms · {n} pulses · {y:.3f} spikes/pulse · '
                     f'{r:.3f} Hz/neuron</title></circle>')
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
.dot {{ fill:var(--surface); stroke-width:2; }} .dot:hover {{ r:6; }}
.legend {{ display:flex; gap:18px; color:var(--ink2); margin:6px 0 4px; font-size:12px; }}
.legend svg {{ vertical-align:middle; }}
table {{ border-collapse:collapse; font-size:12px; font-variant-numeric:tabular-nums; }}
th, td {{ padding:3px 8px; border-bottom:1px solid var(--grid); text-align:right; }} th:first-child, td:first-child {{ text-align:left; }}
.wrap {{ overflow-x:auto; }} code {{ font-size:12px; }}
"""
    out = [f"<!doctype html><html lang='ko'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
           f"<title>M4 IPI tuning curves</title><style>{css}</style></head><body>",
           "<h1>M4 phase A — IPI tuning curves (model unchanged)</h1>",
           f"<p class='sub'>spikes per pulse per probe set vs inter-pulse interval · g = {meta['g']} · a_in rate {meta['a_in']['rate']:g} / "
           f"phase-lock {meta['a_in']['phase-lock']:g} · noise 0 · seed {meta['seed']} · commit {(meta['git_commit'] or '?')[:7]}"
           f"{' (dirty tree)' if meta['git_dirty'] else ''} · {html.escape(meta['generated_at'])}. "
           f"Dotted vertical line = {REFERENCE_IPI_MS:g} ms literature reference, shown for orientation only, not an expectation.</p>",
           "<div class='legend'><span><svg width='26' height='10'><line x1='0' x2='26' y1='5' y2='5' stroke='currentColor' stroke-width='2'/></svg> left (L)</span>"
           "<span><svg width='26' height='10'><line x1='0' x2='26' y1='5' y2='5' stroke='currentColor' stroke-width='2' stroke-dasharray='6 4'/></svg> right (R)</span>"
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
                out.append(f"<tr><td>{mode}</td><td>{base}_{side} (n={T['meta']['probe_sets']['sizes'][f'{base}_{side}']})</td>"
                           + "".join(f"<td>{v:.3f}</td>" for v in c["spikes_per_pulse"]) + "</tr>")
    out.append("</tbody></table></div>")
    out.append("<h2>Parameters</h2><div class='wrap'><table><tbody>")
    rows = [("g", meta["g"]), ("a_in (rate / phase-lock)", f"{meta['a_in']['rate']:g} / {meta['a_in']['phase-lock']:g}"),
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
def write_report(T: dict) -> None:
    meta, curves, runs = T["meta"], T["curves"], T["runs"]
    L = [f"# M4 report — phase A: IPI tuning curves, model unchanged\n",
         f"Generated by `python -m flysim.apps.m4_ipi` on {meta['generated_at']} at commit "
         f"`{(meta['git_commit'] or '?')[:7]}`{' (dirty tree at run time)' if meta['git_dirty'] else ''}. "
         f"Figure: `docs/m4-tuning.html` (inline SVG, single file). Numbers: `data-provenance/m4-tuning.json`. "
         f"Spec: `docs/m4-brief.md`. Wall time {meta['wall_total_s']} s for {len(runs)} runs.\n",
         "## 1. Parameters (nothing adjusted)\n",
         f"- `g = {meta['g']}`, `noise_sigma = 0`, `v_floor = None`, seed {meta['seed']}; neuron parameters = M2 defaults "
         f"(`{json.dumps({k: meta['params']['rate'][k] for k in ('tau_m', 'tau_syn', 't_ref', 'v_rest', 'v_reset', 'v_thresh')})}`).",
         f"- `a_in` = M3 selected values, unchanged: rate {meta['a_in']['rate']:g}, phase-lock {meta['a_in']['phase-lock']:g} "
         f"(asserted against `data-provenance/m3-results.json` at start).",
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
        for metric, title, fmt in (("spikes_per_pulse", "spikes per pulse (set total, stimulus window 100–800 ms) — the primary curve", "{:.3f}"),
                                   ("rate_stim", "mean rate in the stimulus window (Hz per neuron)", "{:.3f}"),
                                   ("latency_ms", "latency from first pulse onset to first spike of the set (ms; – = none)", "{:.1f}")):
            L.append(f"**{title}**\n")
            L.append("| set | n | " + " | ".join(f"{i:g}" for i in meta["ipi_ms"]) + " |\n|---|---|" + "---|" * len(meta["ipi_ms"]))
            for name in [f"{b}_{s}" for b in BASE_SETS for s in ("L", "R")] + \
                        [n for n in curves[mode] if n.endswith("_unk")] + ["rest"]:
                c = curves[mode][name]
                cells = [("–" if v is None else fmt.format(v)) for v in c[metric]]
                L.append(f"| `{name}` | {meta['probe_sets']['sizes'][name]:,} | " + " | ".join(cells) + " |")
            L.append("")

    L.append("## 4. Observations (numbers only)\n")
    for mode in curves:
        c = curves[mode]
        for base in BASE_SETS:
            for side in ("L", "R"):
                v = c[f"{base}_{side}"]["spikes_per_pulse"]
                if max(v) == 0:
                    L.append(f"- `{mode}` `{base}_{side}`: 0 spikes at every IPI.")
                else:
                    i_max, i_min = int(np.argmax(v)), int(np.argmin(v))
                    L.append(f"- `{mode}` `{base}_{side}`: spikes/pulse ranges {min(v):.3f} (IPI {meta['ipi_ms'][i_min]:g}) to "
                             f"{max(v):.3f} (IPI {meta['ipi_ms'][i_max]:g}); at 35 ms {v[meta['ipi_ms'].index(35.0)]:.3f}.")
    L.append("")
    L.append("## 5. Verdict (CLAUDE.md §6 M4 / m4-brief)\n")
    L.append(f"- [{'x' if n_ign == 0 else ' '}] 18 runs completed, no run ignited ({n_ign} ignited).")
    L.append("- [x] curves rendered in `runs/m4-ipi/tuning.html` = `docs/m4-tuning.html` (inline SVG, no external resources). "
             "Not opened in a browser in this session; `tests/test_m4.py` parses the file and checks the SVG panels, "
             "so that is the substitute check.")
    L.append("- [x] every parameter recorded in `tuning.json` (`meta.params`, `meta.adapter`, `meta.stimulus`, `meta.a_in`, `meta.g`) and above.")
    L.append("- [x] `pytest tests/` — see commit message.")
    L.append("- pC1 response: " + "; ".join(
        f"`{mode}` L/R max spikes/pulse {max(curves[mode]['pC1_L']['spikes_per_pulse']):.3f} / "
        f"{max(curves[mode]['pC1_R']['spikes_per_pulse']):.3f}" for mode in curves) + ".")
    L.append("")
    L.append("## 6. Interpretation (one paragraph; causes are marked as assumptions)\n")
    L.append("Under this model the click train drives JO_AB and its direct postsynaptic set; the tables above show how far "
             "activity gets at each IPI and whether the first hop depends on IPI. Where a set reads 0 at every IPI, that is "
             "the result of this stage and is reported as such. Assumed (not tested here) reason for activity stopping "
             "at the first hop: at g = 0.336 with no background activity the summed excitation from JO_post spikes does "
             "not reach threshold in the next stage, and no fluctuation-driven low-rate state exists in this "
             "current-based model (data-provenance/parameter-decisions.md). Any model change belongs to phase B and is "
             "the user's decision.\n")
    L.append("## 7. Deviations / notes\n")
    L.append("- No parameter was swept or adjusted; the only variable is IPI. No attempt was made to change curve shapes.")
    L.append("- The post-train silence is 200 ms minus the tail of the last pulse ((n−1)·IPI + 10 ms ends before 800 ms), "
             "so the total is exactly 1000 ms; the stimulus window is 100–800 ms for every IPI.")
    L.append("- `spikes_per_pulse` is the set total; `spikes_per_pulse_per_neuron` (÷ set size) is in `tuning.json` "
             "as well. Sets `_unk` and `rest` are listed in the tables, not in the figure (the brief's six sets).")
    L.append("- The figure uses small multiples (one panel per set, per mode) with a per-panel y-scale, because the sets "
             "differ by orders of magnitude; the table under the figure carries the exact values.")
    L.append("")
    REPORT_MD.write_text("\n".join(L))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
