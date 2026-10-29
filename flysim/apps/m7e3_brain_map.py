"""M7e3b: a flat map of the real brain for the show page (docs/m7e3-brief.md §3, contract §3.1).

Presentation only - no simulation, no experiment. Inputs are the M6 files
(docs/m6-3d-contract.md): the point cloud ``data/cache/neurons-3d.{bin,json}`` and the escape
skeleton bundles ``data/cache/skel-{LC4,LPLC2,DNp01,TTMn}.{bin,json}``. Output:
``runs/show/brain-map.json`` and the preview ``docs/m7e3-brain-map.png``.

* **View**: from above - the x-z plane (the y axis is dropped). The orientation is read from
  the data, not assumed:
    - the fly's left goes to the screen's left: whichever way the mean x of ``LC4_L`` lies
      from the mean x of ``LC4_R`` is the fly's left;
    - anterior (brain) goes up: whichever way the mean z of the ``LC4`` cells (optic lobes,
      in the brain) lies from the mean z of the ``TTMn`` cells (thoracic nerve cord).
* **Grid**: one scale for both axes (aspect kept); the longer side of the bounding box of
  everything drawn spans 0..4095 and every coordinate is an integer in 0..4095.
* **Dots**: 30,000 neurons drawn without replacement with ``numpy.random.default_rng(0)``
  from every neuron that has a position (``src != 2``; those have NaN).
* **Circuit sets**: the eight sets of the ``escape`` probe profile - the same rule and the same
  names as the ``regions`` of the M7c/M7c2 ``run.json`` bundles. Per set: neuron count, soma
  points, the skeleton of each neuron projected the same way, and the centroid (mean of the
  projected soma points). An empty set raises.
* **Line simplification** (allowed by the brief up to one screen pixel): each skeleton is split
  back into the polylines its bundle was written from, projected, simplified with
  Ramer-Douglas-Peucker at 4 grid units, then rounded to the grid; segments that round to a
  single point, and repeats, are dropped. "One screen pixel" is taken as one pixel of the map
  drawn 1,024 px along its long side (the preview's size): 4096 / 1024 = 4 grid units. Every
  dropped vertex is within 4 grid units of the kept line. Every branch keeps at least one
  segment - nothing is pruned - so the line count cannot fall below the number of branches.

Usage: python -m flysim.apps.m7e3_brain_map
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

from ..data.download import ROOT
from ..data.skeletons import rdp_mask
from .games.media import write_png

CACHE = ROOT / "data" / "cache"
OUT_JSON = ROOT / "runs" / "show" / "brain-map.json"
PREVIEW_PNG = ROOT / "docs" / "m7e3-brain-map.png"

BUNDLES = ("LC4", "LPLC2", "DNp01", "TTMn")
SIDES = ("L", "R")
SET_NAMES = tuple(f"{b}_{s}" for b in BUNDLES for s in SIDES)
N_DOTS = 30_000
DOT_SEED = 0
GRID = 4096
PREVIEW_SCALE = 4                     # grid units per preview pixel: long side 1,024 px
LINE_TOL = float(PREVIEW_SCALE)       # "one screen pixel" (brief §3) = one pixel of a 1,024-px map
SIZE_TARGET_BYTES = 1_500_000

# preview: grey dots under the circuit; types told apart by brightness (media.py is greyscale)
PREVIEW_BG, PREVIEW_DOT = 10, 92
PREVIEW_INK = {"LPLC2": 140, "LC4": 185, "TTMn": 225, "DNp01": 255}


def log(msg: str) -> None:
    print(f"[m7e3b] {msg}", flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------------------
def load_points(cache: Path = CACHE) -> tuple[np.ndarray, np.ndarray, dict]:
    """(pos float32 [n,3] with NaN rows, src uint8 [n], header) from neurons-3d."""
    meta = json.loads((cache / "neurons-3d.json").read_text())
    raw = (cache / "neurons-3d.bin").read_bytes()
    a = meta["arrays"]
    pos = np.frombuffer(raw, np.float32, a["pos"]["length"], a["pos"]["offset"]).reshape(-1, 3)
    src = np.frombuffer(raw, np.uint8, a["src"]["length"], a["src"]["offset"])
    if len(pos) != meta["n"] or len(src) != meta["n"]:
        raise ValueError("neurons-3d arrays do not match its header")
    return pos, src, meta


def load_bundle(name: str, cache: Path = CACHE) -> tuple[dict, np.ndarray]:
    meta = json.loads((cache / f"skel-{name}.json").read_text())
    seg = np.fromfile(cache / f"skel-{name}.bin", dtype="<f4").reshape(-1, 6)
    if len(seg) != meta["n_segments"]:
        raise ValueError(f"skel-{name}: {len(seg)} segments, header says {meta['n_segments']}")
    return meta, seg


def escape_sets(graph=None) -> dict[str, np.ndarray]:
    """The eight circuit sets from the ``escape`` probe profile (the run.json regions)."""
    from ..graph import Graph
    from ..probe.sets import build_probe_sets, load_roi
    g = graph if graph is not None else Graph.load()
    ps = build_probe_sets(g, load_roi(g), profile="escape")
    return {k: np.sort(np.asarray(ps.members[k], dtype=np.int64)) for k in SET_NAMES}


# ---------------------------------------------------------------------------------------
# projection (pure)
# ---------------------------------------------------------------------------------------
def orientation(pos: np.ndarray, sets: dict[str, np.ndarray]) -> dict:
    """Which way x and z must run on screen, read from the data."""
    for k in ("LC4_L", "LC4_R", "TTMn_L", "TTMn_R"):
        if len(sets[k]) == 0:
            raise LookupError(f"{k} is empty; cannot orient the map")
    xl, xr = float(np.nanmean(pos[sets["LC4_L"], 0])), float(np.nanmean(pos[sets["LC4_R"], 0]))
    lc4 = np.concatenate([sets["LC4_L"], sets["LC4_R"]])
    ttmn = np.concatenate([sets["TTMn_L"], sets["TTMn_R"]])
    z_brain, z_vnc = float(np.nanmean(pos[lc4, 2])), float(np.nanmean(pos[ttmn, 2]))
    if not np.isfinite([xl, xr, z_brain, z_vnc]).all() or xl == xr or z_brain == z_vnc:
        raise ValueError("cannot read the orientation from the data")
    return dict(left_is_high_x=bool(xl > xr), anterior_is_low_z=bool(z_brain < z_vnc),
                lc4_mean_x=dict(L=xl, R=xr), mean_z=dict(LC4=z_brain, TTMn=z_vnc))


class Projector:
    """voxel (x, z) -> grid (float); the longer side of the box spans 0..GRID-1."""

    def __init__(self, x_range: tuple[float, float], z_range: tuple[float, float], orient: dict):
        self.x0, self.x1 = map(float, x_range)
        self.z0, self.z1 = map(float, z_range)
        self.flip_x = orient["left_is_high_x"]               # fly left (high x) -> screen left (0)
        self.flip_z = not orient["anterior_is_low_z"]        # anterior -> screen top (0)
        self.scale = (GRID - 1) / max(self.x1 - self.x0, self.z1 - self.z0)
        self.w = int(round((self.x1 - self.x0) * self.scale)) + 1
        self.h = int(round((self.z1 - self.z0) * self.scale)) + 1

    def __call__(self, x, z) -> tuple[np.ndarray, np.ndarray]:
        x, z = np.asarray(x, np.float64), np.asarray(z, np.float64)
        sx = (self.x1 - x) if self.flip_x else (x - self.x0)
        sy = (self.z1 - z) if self.flip_z else (z - self.z0)
        return sx * self.scale, sy * self.scale

    @staticmethod
    def quantise(v: np.ndarray) -> np.ndarray:
        return np.clip(np.rint(v), 0, GRID - 1).astype(np.int64)


def polylines(seg: np.ndarray) -> list[np.ndarray]:
    """Split a neuron's segment rows back into the polylines they were written from:
    a new polyline starts wherever a segment does not begin where the previous one ended."""
    if len(seg) == 0:
        return []
    breaks = np.flatnonzero(np.any(seg[1:, :3] != seg[:-1, 3:], axis=1)) + 1
    out = []
    for a, b in zip(np.r_[0, breaks], np.r_[breaks, len(seg)]):
        out.append(np.vstack([seg[a, :3][None, :], seg[a:b, 3:]]))
    return out


def project_neuron(seg: np.ndarray, proj: Projector, tol: float = LINE_TOL) -> tuple[list[int], float, int]:
    """Flat [x1, y1, x2, y2, ...] grid segments of one neuron, the largest RDP deviation
    (grid units), and the number of input segments."""
    flat: list[int] = []
    seen: set[tuple[int, int, int, int]] = set()
    worst = 0.0
    for pl in polylines(seg):
        sx, sy = proj(pl[:, 0], pl[:, 2])
        p2 = np.column_stack([sx, sy])
        keep, dev = rdp_mask(p2, tol)
        worst = max(worst, dev)
        q = np.column_stack([Projector.quantise(sx[keep]), Projector.quantise(sy[keep])])
        for (x1, y1), (x2, y2) in zip(q[:-1].tolist(), q[1:].tolist()):
            if x1 == x2 and y1 == y2:
                continue                                       # rounds to a single point
            key = (x1, y1, x2, y2) if (x1, y1) <= (x2, y2) else (x2, y2, x1, y1)
            if key in seen:
                continue
            seen.add(key)
            flat += [x1, y1, x2, y2]
    return flat, worst, len(seg)


# ---------------------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------------------
def build(sets: dict[str, np.ndarray], cache: Path = CACHE) -> dict:
    pos, src, meta = load_points(cache)
    for k in SET_NAMES:
        if k not in sets or len(sets[k]) == 0:
            raise LookupError(f"circuit set {k!r} is empty")
    bundles = {b: load_bundle(b, cache) for b in BUNDLES}

    # every set member must have a position and a skeleton in its bundle, and nothing else
    for b, (bm, _) in bundles.items():
        in_bundle = sorted(int(n["idx"]) for n in bm["neurons"])
        members = sorted(np.concatenate([sets[f"{b}_{s}"] for s in SIDES]).tolist())
        if in_bundle != members:
            raise ValueError(f"skel-{b} holds {len(in_bundle)} neurons, the sets {len(members)}: "
                             "membership differs")
    for k in SET_NAMES:
        if (src[sets[k]] == 2).any():
            raise ValueError(f"{k} has neurons without a position")

    orient = orientation(pos, sets)
    valid = np.flatnonzero(src != 2)
    if np.isnan(pos[valid]).any():
        raise ValueError("a neuron marked as positioned has NaN coordinates")
    allseg = np.vstack([s for _, s in bundles.values()])
    xs = np.r_[pos[valid, 0], allseg[:, 0], allseg[:, 3]]
    zs = np.r_[pos[valid, 2], allseg[:, 2], allseg[:, 5]]
    proj = Projector((xs.min(), xs.max()), (zs.min(), zs.max()), orient)

    rng = np.random.default_rng(DOT_SEED)
    pick = np.sort(rng.choice(valid, size=N_DOTS, replace=False))
    dx, dy = proj(pos[pick, 0], pos[pick, 2])
    dots = np.column_stack([Projector.quantise(dx), Projector.quantise(dy)]).ravel().tolist()

    out_sets, stats = {}, {}
    for b, (bm, seg) in bundles.items():
        by_idx = {int(n["idx"]): n for n in bm["neurons"]}
        for s in SIDES:
            k = f"{b}_{s}"
            idx = sets[k]
            sx, sy = proj(pos[idx, 0], pos[idx, 2])
            soma = np.column_stack([Projector.quantise(sx), Projector.quantise(sy)]).ravel().tolist()
            centroid = [int(Projector.quantise(sx.mean())), int(Projector.quantise(sy.mean()))]
            lines, worst, n_in, n_out = [], 0.0, 0, 0
            for i in idx.tolist():
                n = by_idx[i]
                flat, dev, cnt = project_neuron(seg[n["seg_offset"]:n["seg_offset"] + n["seg_count"]], proj)
                lines.append(flat)
                worst, n_in, n_out = max(worst, dev), n_in + cnt, n_out + len(flat) // 4
            out_sets[k] = dict(n_neurons=int(len(idx)), soma=soma, lines=lines, centroid=centroid)
            stats[k] = dict(segments_in=n_in, segments_out=n_out, max_rdp_deviation_grid=round(worst, 4))

    sources = {str(p.relative_to(ROOT)): sha256(p) for p in
               [cache / "neurons-3d.json", cache / "neurons-3d.bin"]
               + [cache / f"skel-{b}.{e}" for b in BUNDLES for e in ("json", "bin")]}
    how = (f"mean x of LC4_L {orient['lc4_mean_x']['L']:.0f} vs LC4_R {orient['lc4_mean_x']['R']:.0f} "
           f"voxels: the fly's left is at {'high' if orient['left_is_high_x'] else 'low'} x, so x is "
           f"{'reversed' if proj.flip_x else 'kept'} to put it on the screen's left. Mean z of LC4 "
           f"{orient['mean_z']['LC4']:.0f} vs TTMn {orient['mean_z']['TTMn']:.0f}: anterior is at "
           f"{'low' if orient['anterior_is_low_z'] else 'high'} z, so z is "
           f"{'reversed' if proj.flip_z else 'kept'} to put it at the top.")
    return {
        "version": 1,
        "view": {"plane": "x-z", "up": "anterior", "screen_left": "fly left",
                 "how_left_was_decided": how,
                 "x_screen": "x reversed" if proj.flip_x else "x", "y_screen": "z reversed" if proj.flip_z else "z",
                 "voxel_nm": int(meta["voxel_nm"]), "grid_units_per_voxel": proj.scale,
                 "voxel_box": {"x": [proj.x0, proj.x1], "z": [proj.z0, proj.z1]}},
        "box": {"w": proj.w, "h": proj.h},
        "dots": {"n": N_DOTS, "seed": DOT_SEED, "from": int(len(valid)),
                 "rule": "numpy.random.default_rng(0).choice over neurons with src != 2, no replacement, "
                         "sorted by graph index", "xy": dots},
        "sets": out_sets,
        "lines_rule": f"bundle polylines projected, RDP at {LINE_TOL:g} grid unit, rounded to the grid; "
                      "zero-length and duplicate segments dropped",
        "line_stats": stats,
        "sources": sources,
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def dumps(doc: dict) -> str:
    return json.dumps(doc, separators=(",", ":"), ensure_ascii=False)


# ---------------------------------------------------------------------------------------
# preview
# ---------------------------------------------------------------------------------------
def _segments_to_pixels(lines: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Rasterise grid segments [m, 4] at 1/k scale (sampled every preview pixel)."""
    p = lines.astype(np.float64) / k
    n = np.maximum(np.ceil(np.abs(p[:, 2:] - p[:, :2]).max(1)).astype(np.int64), 1) + 1
    seg = np.repeat(np.arange(len(p)), n)
    t = (np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)) / np.repeat(n - 1, n)
    x = p[seg, 0] + t * (p[seg, 2] - p[seg, 0])
    y = p[seg, 1] + t * (p[seg, 3] - p[seg, 1])
    return np.rint(y).astype(np.int64), np.rint(x).astype(np.int64)


def preview(doc: dict, k: int = PREVIEW_SCALE) -> np.ndarray:
    h, w = -(-doc["box"]["h"] // k), -(-doc["box"]["w"] // k)
    img = np.full((h, w), PREVIEW_BG, np.uint8)
    d = np.asarray(doc["dots"]["xy"], np.int64).reshape(-1, 2) // k
    img[d[:, 1], d[:, 0]] = PREVIEW_DOT
    for b in sorted(BUNDLES, key=lambda b: PREVIEW_INK[b]):           # brightest drawn last
        ink = PREVIEW_INK[b]
        for s in SIDES:
            st = doc["sets"][f"{b}_{s}"]
            flat = [v for neuron in st["lines"] for v in neuron]
            if flat:
                yy, xx = _segments_to_pixels(np.asarray(flat).reshape(-1, 4), k)
                img[np.clip(yy, 0, h - 1), np.clip(xx, 0, w - 1)] = ink
            so = np.asarray(st["soma"], np.int64).reshape(-1, 2) // k
            r = 3 if len(so) <= 2 else 1                                # single cells: bigger mark
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    img[np.clip(so[:, 1] + dy, 0, h - 1), np.clip(so[:, 0] + dx, 0, w - 1)] = ink
    return img


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT_JSON)
    ap.add_argument("--preview", type=Path, default=PREVIEW_PNG)
    a = ap.parse_args(argv)
    t0 = time.time()
    doc = build(escape_sets())
    text = dumps(doc)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(text)
    size = len(text.encode())
    log(f"box {doc['box']}; {doc['view']['how_left_was_decided']}")
    for k, s in doc["line_stats"].items():
        log(f"  {k:8s} {doc['sets'][k]['n_neurons']:4d} neurons  segments {s['segments_in']:>7,} -> "
            f"{s['segments_out']:>6,}  centroid {doc['sets'][k]['centroid']}")
    log(f"wrote {a.out.relative_to(ROOT)}: {size:,} bytes (target <= {SIZE_TARGET_BYTES:,}), "
        f"sha256 {hashlib.sha256(text.encode()).hexdigest()[:16]}")
    if size > SIZE_TARGET_BYTES:
        log("WARNING: over the 1.5 MB target")
    write_png(a.preview, preview(doc))
    log(f"wrote {a.preview.relative_to(ROOT)} in {time.time() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
