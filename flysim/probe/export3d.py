"""M6b: the neuron point cloud — file 1 of ``docs/m6-3d-contract.md``.

Writes ``data/cache/neurons-3d.bin`` (three concatenated arrays, no header) and
``data/cache/neurons-3d.json`` (their byte offsets plus the metadata the 3D viewer needs).

Coordinates are **MaleCNS voxels (8 nm)**, exactly as they sit in the annotation table and in
the SWC skeletons — nothing is recentred or converted here (contract). Per neuron:

    somaLocation            -> src 0
    synapse centroid        -> src 1   (data/cache/neuron-roi-v1.parquet, cx/cy/cz)
    neither                 -> src 2   and the position is **NaN**, never (0,0,0): a zero
                                       would pile those neurons onto the origin, and the
                                       viewer is expected to drop them.

Contract v1.3 adds a fourth array, ``body`` (uint32): the bodyId per neuron, so a click in the
3D view can name the neuron instead of showing a bare index. It is appended after the other
three, whose offsets and meaning are unchanged, and a bodyId too large for uint32 raises
rather than being truncated (the largest observed is 1,571,825,087).

The array order is the retained graph's neuron index order, which is the same index
``run.json`` uses, so the viewer can look a spike's neuron index straight up in ``pos``.

Usage: python -m flysim.probe.export3d [--force] [--out-dir DIR]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.download import ROOT
from ..graph.constants import CACHE_NPZ
from .sets import K_MIN_DEFAULT, build_probe_sets, load_roi

CACHE_DIR = ROOT / "data" / "cache"
OUT_BIN = CACHE_DIR / "neurons-3d.bin"
OUT_JSON = CACHE_DIR / "neurons-3d.json"
VOXEL_NM = 8                      # contract: 1 voxel = 8 nm, files carry raw voxels
SRC_SOMA, SRC_CENTROID, SRC_NONE = 0, 1, 2
SRC_NAMES = {SRC_SOMA: "soma", SRC_CENTROID: "centroid", SRC_NONE: "none"}
SKEL_GLOB = "skel-*.json"
UINT32_MAX = (1 << 32) - 1        # the `body` array's range (contract v1.3)


# ---------------------------------------------------------------------------------------
# arrays
# ---------------------------------------------------------------------------------------
def neuron_positions_3d(graph, roi: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict]:
    """``(pos float32[n, 3], src uint8[n], src_counts)`` in graph-index order."""
    if len(roi) != graph.n or not np.array_equal(roi["bodyId"].to_numpy(), graph.body_ids):
        raise ValueError("the ROI table is not aligned with the graph's body_ids")
    pos = np.asarray(graph.soma_xyz, dtype=np.float32).copy()       # NaN where there is no soma
    has_soma = np.asarray(graph.has_soma_xyz, dtype=bool)
    centroid = roi[["cx", "cy", "cz"]].to_numpy(dtype=np.float64)
    has_centroid = np.isfinite(centroid).all(axis=1)
    use_centroid = ~has_soma & has_centroid
    pos[use_centroid] = centroid[use_centroid].astype(np.float32)
    src = np.where(has_soma, SRC_SOMA, np.where(use_centroid, SRC_CENTROID, SRC_NONE)).astype(np.uint8)
    pos[src == SRC_NONE] = np.nan                                   # explicit: never (0, 0, 0)
    counts = {name: int((src == code).sum()) for code, name in SRC_NAMES.items()}
    if counts["soma"] + counts["centroid"] + counts["none"] != graph.n:
        raise AssertionError("src counts do not add up to the neuron count")
    if np.isnan(pos[src != SRC_NONE]).any() or not np.isnan(pos[src == SRC_NONE]).all():
        raise AssertionError("NaN positions and src == 2 disagree")
    return pos, src, counts


def _md5(path: Path, chunk: int = 1 << 22) -> str | None:
    if not path.exists():
        return None
    h = hashlib.md5()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def body_ids_uint32(body_ids) -> np.ndarray:
    """bodyIds as uint32 for the ``body`` array — refusing, never truncating, out-of-range ids."""
    b = np.asarray(body_ids)
    if b.ndim != 1:
        raise ValueError("body_ids must be one-dimensional")
    wide = b.astype(np.int64, copy=False)
    if not np.array_equal(wide, b):
        raise ValueError("body_ids are not integers")
    bad = (wide < 0) | (wide > UINT32_MAX)
    if bad.any():
        raise ValueError(
            f"{int(bad.sum())} bodyId(s) do not fit in uint32 (max {UINT32_MAX:,}), "
            f"largest {int(wide.max()):,}: the contract's `body` array cannot carry them. "
            "Report this rather than truncating — the array's dtype has to change.")
    return wide.astype(np.uint32)


def build_doc(pos: np.ndarray, set_idx: np.ndarray, src: np.ndarray, set_names: list[str],
              body_ids, graph_cache: Path = CACHE_NPZ) -> tuple[dict, bytes]:
    """The ``.json`` metadata and the ``.bin`` payload, laid out in the contract's order."""
    n = len(src)
    if pos.shape != (n, 3) or len(set_idx) != n:
        raise ValueError(f"array shapes disagree: pos {pos.shape}, set {set_idx.shape}, n {n}")
    if len(set_names) > 256:
        raise ValueError("the set array is uint8: at most 256 probe sets")
    body = body_ids_uint32(body_ids)
    if len(body) != n:
        raise ValueError(f"body array has {len(body)} entries, expected {n}")
    pos_b = np.ascontiguousarray(pos, dtype=np.float32).tobytes()
    set_b = np.ascontiguousarray(set_idx, dtype=np.uint8).tobytes()
    src_b = np.ascontiguousarray(src, dtype=np.uint8).tobytes()
    body_b = body.tobytes()
    finite = np.isfinite(pos).all(axis=1)
    if not finite.any():
        raise ValueError("no neuron has a finite position")
    lo = pos[finite].min(axis=0)
    hi = pos[finite].max(axis=0)
    doc = {
        "n": int(n), "voxel_nm": VOXEL_NM, "order": "graph neuron index",
        "bbox": {"min": [float(v) for v in lo], "max": [float(v) for v in hi]},
        "sets": list(set_names),
        "src_counts": {name: int((src == code).sum()) for code, name in SRC_NAMES.items()},
        "arrays": {
            "pos": {"offset": 0, "length": 3 * n, "dtype": "float32"},
            "set": {"offset": len(pos_b), "length": n, "dtype": "uint8"},
            "src": {"offset": len(pos_b) + len(set_b), "length": n, "dtype": "uint8"},
            # contract v1.3, appended so the three arrays above keep their offsets
            "body": {"offset": len(pos_b) + len(set_b) + len(src_b), "length": n, "dtype": "uint32"},
        },
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "graph_cache_md5": _md5(graph_cache),
    }
    return doc, pos_b + set_b + src_b + body_b


def write(out_bin: Path = OUT_BIN, out_json: Path = OUT_JSON, *, graph=None, roi=None,
          sets=None, k_min: int = K_MIN_DEFAULT) -> dict:
    """Build the point cloud from the retained graph and write both files."""
    if graph is None:
        from ..graph import Graph
        graph = Graph.load()
    if roi is None:
        roi = load_roi(graph)
    if sets is None:
        sets = build_probe_sets(graph, roi, k_min=k_min)
    pos, src, counts = neuron_positions_3d(graph, roi)
    set_idx = np.asarray(sets.region_of, dtype=np.uint8)
    if not np.array_equal(set_idx.astype(np.int64), np.asarray(sets.region_of, dtype=np.int64)):
        raise ValueError("probe set indices do not fit in uint8")
    doc, payload = build_doc(pos, set_idx, src, list(sets.names), graph.body_ids)
    out_bin.parent.mkdir(parents=True, exist_ok=True)
    out_bin.write_bytes(payload)
    out_json.write_text(json.dumps(doc, separators=(",", ":")) + "\n")
    doc["_bytes"] = len(payload)
    return doc


def read_arrays(bin_path: Path = OUT_BIN, json_path: Path = OUT_JSON) -> dict:
    """Read the pair back the way the viewer does (offsets from the json, raw bytes from the bin)."""
    doc = json.loads(Path(json_path).read_text())
    raw = Path(bin_path).read_bytes()
    out = {"doc": doc}
    for name, a in doc["arrays"].items():
        dtype = np.dtype(a["dtype"])
        end = a["offset"] + a["length"] * dtype.itemsize
        if end > len(raw):
            raise ValueError(f"array {name} runs past the end of {bin_path}")
        arr = np.frombuffer(raw, dtype=dtype, count=a["length"], offset=a["offset"])
        out[name] = arr.reshape(-1, 3) if name == "pos" else arr
    # an older file has no `body`; consumers must cope (contract v1.3)
    return out


# ---------------------------------------------------------------------------------------
# what the server can offer (used by hello.assets and by /skel/index.json)
# ---------------------------------------------------------------------------------------
def skeleton_bundles(cache_dir: Path | None = None) -> list[dict]:
    """Skeleton bundles present on disk, as ``/skel/index.json`` returns them.

    Window A writes these; an absent directory or no bundles is an empty list, not an error.
    """
    cache_dir = Path(CACHE_DIR if cache_dir is None else cache_dir)
    out = []
    for meta in sorted(cache_dir.glob(SKEL_GLOB)):
        name = meta.name[len("skel-"):-len(".json")]
        entry = {"name": name, "n_neurons": None, "n_segments": None, "bytes": None}
        try:
            d = json.loads(meta.read_text())
            entry["n_neurons"] = d.get("n_neurons")
            entry["n_segments"] = d.get("n_segments")
        except Exception:                      # a half-written bundle must not break the listing
            entry["error"] = "unreadable json"
        binp = meta.with_suffix(".bin")
        if binp.exists():
            entry["bytes"] = binp.stat().st_size
        else:
            entry["error"] = "missing .bin"
        out.append(entry)
    return out


def available_assets(cache_dir: Path | None = None) -> dict:
    """``hello.assets`` (protocol v1.2): what 3D data this server can actually serve."""
    cache_dir = Path(CACHE_DIR if cache_dir is None else cache_dir)
    bundles = skeleton_bundles(cache_dir)
    neurons_3d = (cache_dir / OUT_BIN.name).exists() and (cache_dir / OUT_JSON.name).exists()
    return {"neurons_3d": bool(neurons_3d),
            "skeleton_sets": [b["name"] for b in bundles if b.get("error") is None]}


# ---------------------------------------------------------------------------------------
def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="build the 3D neuron point cloud (M6b)")
    ap.add_argument("--force", action="store_true", help="rebuild even if the files exist")
    ap.add_argument("--out-dir", default=str(CACHE_DIR))
    args = ap.parse_args(argv)
    out_dir = Path(args.out_dir)
    out_bin, out_json = out_dir / OUT_BIN.name, out_dir / OUT_JSON.name
    if out_bin.exists() and out_json.exists() and not args.force:
        doc = json.loads(out_json.read_text())
        print(f"[3d] {out_json} already exists ({doc['n']:,} neurons, "
              f"{out_bin.stat().st_size / 1e6:.1f} MB); --force to rebuild")
        return 0
    t0 = time.time()
    doc = write(out_bin, out_json)
    print(f"[3d] {doc['n']:,} neurons in {time.time() - t0:.1f} s -> "
          f"{out_bin.name} ({doc['_bytes'] / 1e6:.2f} MB) + {out_json.name}")
    print(f"[3d] src_counts: {doc['src_counts']}  (NaN positions: {doc['src_counts']['none']:,})")
    print(f"[3d] bbox voxels: min {doc['bbox']['min']} max {doc['bbox']['max']}")
    print(f"[3d] sets: {len(doc['sets'])}  arrays: {list(doc['arrays'])}  "
          f"graph_cache_md5: {doc['graph_cache_md5']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
