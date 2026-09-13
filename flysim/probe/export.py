"""M3: run.json export for flysim-viewer.html.

The contract is the comment block at the top of flysim-viewer.html (that
comment wins over any other document). Keys written here:
  meta / neurons{x,y,region} / regions / input{label,envelope} / frames{n,rates} / spikes{neuron_idx,t_bin}
Rules (CLAUDE.md §5.5): rates normalised to 0..1 with the max in
``meta.rate_norm_hz``; spikes sampled per frame (cap ``MAX_SPIKES_PER_FRAME``)
with the realised ratio in ``meta.spike_sample_ratio``; ``meta.sensory_mode`` set.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.download import ROOT
from .probe import RateTable

RUNS_DIR = ROOT / "runs"
NEURONS_JSON = RUNS_DIR / "neurons-v1.json"
MAX_SPIKES_PER_FRAME = 1500
REQUIRED_KEYS = {
    "meta": ["run_id", "dataset", "dt_ms", "bin_ms", "n_neurons", "sensory_mode", "rate_norm_hz",
             "spike_sample_ratio", "engine_params", "adapter", "stimulus"],
    "neurons": ["x", "y", "region"], "input": ["label", "envelope"], "frames": ["n", "rates"],
    "spikes": ["neuron_idx", "t_bin"],
}


def neuron_coords(graph, roi: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict]:
    """(x, y) per neuron: soma_xyz if present, else synapse centroid (cx, cy), else a fixed
    off-graph point below/left of the bounding box. Returns (x, y, info)."""
    x = graph.soma_xyz[:, 0].astype(np.float64).copy()
    y = graph.soma_xyz[:, 1].astype(np.float64).copy()
    use_c = ~graph.has_soma_xyz & roi["cx"].notna().to_numpy()
    x[use_c] = roi.loc[use_c, "cx"].to_numpy()
    y[use_c] = roi.loc[use_c, "cy"].to_numpy()
    missing = np.isnan(x) | np.isnan(y)
    x0, x1 = np.nanmin(x), np.nanmax(x)
    y0, y1 = np.nanmin(y), np.nanmax(y)
    off = (x0 - 0.05 * (x1 - x0), y0 - 0.05 * (y1 - y0))
    x[missing], y[missing] = off
    info = {"from_soma": int(graph.has_soma_xyz.sum()), "from_synapse_centroid": int(use_c.sum()),
            "missing_placed_off_graph": int(missing.sum()), "off_graph_xy": [float(off[0]), float(off[1])],
            "projection": "x,y voxel coordinates as-is"}
    return np.round(x).astype(np.int64), np.round(y).astype(np.int64), info


def sample_spikes(t_bin: np.ndarray, idx: np.ndarray, n_frames: int, cap: int, seed: int = 0):
    """Uniform per-frame sample of at most ``cap`` spikes. Returns (t_bin, idx, ratio)."""
    rng = np.random.default_rng(seed)
    if len(t_bin) == 0:
        return t_bin, idx, 1.0
    order = np.argsort(t_bin, kind="stable")
    t_bin, idx = t_bin[order], idx[order]
    bounds = np.searchsorted(t_bin, np.arange(n_frames + 1))
    keep = []
    for f in range(n_frames):
        a, b = bounds[f], bounds[f + 1]
        if b - a <= cap:
            keep.append(np.arange(a, b))
        else:
            keep.append(a + np.sort(rng.choice(b - a, cap, replace=False)))
    keep = np.concatenate(keep)
    return t_bin[keep], idx[keep], float(len(keep) / len(t_bin))


def write_run_json(path: Path, *, run_id: str, sensory_mode: str, dt_ms: float, rt: RateTable,
                   sets, coords: tuple[np.ndarray, np.ndarray, dict], input_label: str,
                   envelope: np.ndarray, spikes: tuple[np.ndarray, np.ndarray], engine_params: dict,
                   adapter: dict, stimulus: dict, extra_meta: dict | None = None,
                   cap: int = MAX_SPIKES_PER_FRAME, seed: int = 0) -> dict:
    x, y, cinfo = coords
    n = len(x)
    rate_max = float(rt.rates.max()) if rt.rates.size else 0.0
    rates = (rt.rates / rate_max) if rate_max > 0 else np.zeros_like(rt.rates)
    t_bin, idx = spikes
    st, si, ratio = sample_spikes(np.asarray(t_bin), np.asarray(idx), rt.n_bins, cap, seed)
    doc = {
        "meta": {
            "run_id": run_id, "dataset": "MaleCNS v1.0", "dt_ms": dt_ms, "bin_ms": rt.bin_ms,
            "n_neurons": n, "sensory_mode": sensory_mode,
            "rate_norm_hz": rate_max, "rate_unit": "Hz per neuron, mean over the set",
            "spike_sample_ratio": ratio, "spike_cap_per_frame": cap, "spike_sample_seed": seed,
            "n_spikes_total": int(len(t_bin)), "active_neurons": rt.active_neurons,
            "engine_params": engine_params, "adapter": adapter, "stimulus": stimulus,
            "coords": cinfo, "region_sizes": {k: int(v) for k, v in zip(rt.names, rt.sizes)},
            **(extra_meta or {}),
        },
        "neurons": {"x": x.tolist(), "y": y.tolist(), "region": sets.region_of.astype(int).tolist()},
        "regions": list(rt.names),
        "input": {"label": input_label, "envelope": np.asarray(envelope, dtype=float).round(4).tolist()},
        "frames": {"n": int(rt.n_bins), "rates": np.round(rates, 5).tolist()},
        "spikes": {"neuron_idx": si.astype(int).tolist(), "t_bin": st.astype(int).tolist()},
    }
    validate_run_doc(doc)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, separators=(",", ":")))
    return doc


def validate_run_doc(doc: dict) -> None:
    for sec, keys in REQUIRED_KEYS.items():
        if sec not in doc:
            raise ValueError(f"run.json missing section {sec!r}")
        for k in keys:
            if k not in doc[sec]:
                raise ValueError(f"run.json missing {sec}.{k}")
    if doc["meta"]["sensory_mode"] not in ("rate", "phase-lock"):
        raise ValueError("meta.sensory_mode must be 'rate' or 'phase-lock'")
    n = doc["meta"]["n_neurons"]
    nr = doc["neurons"]
    if not (len(nr["x"]) == len(nr["y"]) == len(nr["region"]) == n):
        raise ValueError("neurons arrays must have length n_neurons")
    R = len(doc["regions"])
    if max(nr["region"]) >= R or min(nr["region"]) < 0 or R > 255:
        raise ValueError("neurons.region must index regions (and R <= 255 for the viewer's Uint8Array)")
    fr = doc["frames"]
    if len(fr["rates"]) != R or any(len(r) != fr["n"] for r in fr["rates"]):
        raise ValueError("frames.rates must be regions x n")
    flat = [v for r in fr["rates"] for v in r]
    if flat and (min(flat) < 0 or max(flat) > 1):
        raise ValueError("frames.rates must be within 0..1")
    env = doc["input"]["envelope"]
    if env and (min(env) < 0 or max(env) > 1):
        raise ValueError("input.envelope must be within 0..1")
    sp = doc["spikes"]
    if len(sp["neuron_idx"]) != len(sp["t_bin"]):
        raise ValueError("spikes arrays must have equal length")


def write_neurons_json(coords: tuple[np.ndarray, np.ndarray, dict], region_of: np.ndarray, regions: list[str],
                       path: Path = NEURONS_JSON) -> None:
    x, y, info = coords
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"meta": {"coords": info, "n_neurons": len(x)},
                                "neurons": {"x": x.tolist(), "y": y.tolist(), "region": region_of.astype(int).tolist()},
                                "regions": regions}, separators=(",", ":")))
