"""M6a: neuron skeleton download, decimation and bundling (docs/m6a-brief.md).

Produces file 2 of the M6 3D contract (docs/m6-3d-contract.md): for a probe-set
name, ``data/cache/skel-<name>.bin`` (float32 line segments, x1 y1 z1 x2 y2 z2
repeated) plus ``data/cache/skel-<name>.json`` (per-neuron ranges into it).

    python -m flysim.data.skeletons --set JO_AB --set pC1 --set WED --set AMMCtype

Coordinates are MaleCNS source voxels (1 voxel = 8 nm) and are written through
unchanged: no re-centering, no unit conversion (contract, "금지").

Pipeline
  1. download   SWC per bodyId from the public GCS bucket, MD5-verified against
                the server's ``x-goog-hash`` header exactly as flysim.data.download
                does, cached in data/raw/skeletons/ and recorded in
                data-provenance/skeleton-hashes.json. A 404 is not an error: the
                bodyId goes into the bundle's ``missing`` list (contract).
  2. parse      SWC columns ``id type x y z radius parent``; ``parent = -1`` is a
                root. Several roots are allowed — the skeleton is treated as a
                forest, and a parent id that is absent from the file starts a new
                tree rather than raising.
  3. decimate   the forest is split into branch paths (root/branch point ->
                branch point/endpoint) and each path is simplified with
                Ramer-Douglas-Peucker. Branch points and endpoints are path
                boundaries, so they survive by construction.
  4. bundle     segments concatenated in graph-index order.

There is **no per-neuron segment cap** (contract v1.1, ``max_segments_per_neuron:
null``): RDP cannot reduce a neuron below one segment per branch path, so the
per-neuron floor is the neuron's branch count — 500-1,700 for a typical
pC1/AMMC/WED cell. Forcing a cap of 200 would mean raising the tolerance to 256x
the requested value and replacing the neuron's shape with its topology; that was
measured under the v1.0 brief and is written up in docs/m6a-report.md §5.

The only size constraint is the contract's 20 MB per bundle, enforced as a hard
exception. Tolerance may differ per set: 40 voxels (0.32 um) everywhere except WED,
which uses 64 (0.51 um) to fit the ceiling. ``max_segments`` is still accepted as an
optional argument — passing an int restores the old escalate-and-retry behaviour —
but it defaults to None and the shipped bundles do not use it.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .download import ROOT

URL_PREFIX = ("https://storage.googleapis.com/flyem-male-cns/v1.0/segmentation/"
              "skeletons-malecns/skeletons-swc")
RAW_DIR = ROOT / "data" / "raw" / "skeletons"
CACHE_DIR = ROOT / "data" / "cache"
PROVENANCE = ROOT / "data-provenance" / "skeleton-hashes.json"
MANIFEST_REL = "data-provenance/skeleton-hashes.json"

VOXEL_NM = 8                         # contract: 1 voxel = 8 nm, no conversion in the file
TOLERANCE_VOXELS = 40.0              # brief default (~0.32 um)
MAX_SEGMENTS_PER_NEURON = None       # contract v1.1: no per-neuron cap (see the module note)
TOLERANCE_BY_SET = {"WED": 64.0}     # contract v1.1: per-set tolerance; default TOLERANCE_VOXELS
MAX_BUNDLE_BYTES = 20 * 1024 * 1024  # contract: a bundle over 20 MB is an exception
BYTES_PER_SEGMENT = 24               # 6 float32
DOWNLOAD_THREADS = 6
DOWNLOAD_RETRIES = 3
_TOLERANCE_GROWTH = 2.0              # factor per retry when over the segment cap
_TOLERANCE_MAX_STEPS = 12


class SkeletonBundleTooLarge(RuntimeError):
    """A bundle exceeded the contract's 20 MB ceiling."""


class SkeletonDownloadError(RuntimeError):
    """A skeleton could not be fetched or failed MD5 verification after retries."""


# ---------------------------------------------------------------------------
# download
# ---------------------------------------------------------------------------
_prov_lock = threading.Lock()


def _server_md5(headers) -> str | None:
    for v in headers.get_all("x-goog-hash") or []:
        if v.startswith("md5="):
            return base64.b64decode(v[4:]).hex()
    return None


def _load_provenance() -> dict:
    if PROVENANCE.exists():
        return json.loads(PROVENANCE.read_text())
    return {}


def _save_provenance(p: dict) -> None:
    PROVENANCE.parent.mkdir(parents=True, exist_ok=True)
    PROVENANCE.write_text(json.dumps(p, indent=1, sort_keys=True) + "\n")


def md5_of(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def skeleton_path(body_id: int) -> Path:
    return RAW_DIR / f"{int(body_id)}.swc"


def download_skeleton(body_id: int, prov: dict, force: bool = False) -> tuple[Path | None, str]:
    """Fetch one skeleton. Returns (path, status) with status in
    {"cached", "downloaded", "missing"}; ``path`` is None when missing (HTTP 404).

    MD5 is verified against the server header on every download and recorded in
    ``prov`` (keyed by file name, like flysim.data.download). A cached file whose
    MD5 still matches the record is not re-fetched.
    """
    body_id = int(body_id)
    fname = f"{body_id}.swc"
    dest = skeleton_path(body_id)
    with _prov_lock:
        rec = prov.get(fname)
    if dest.exists() and rec and not force and rec.get("md5") and md5_of(dest) == rec["md5"]:
        return dest, "cached"

    url = f"{URL_PREFIX}/{fname}"
    req = urllib.request.Request(url, headers={"User-Agent": "flysim/0.0.1"})
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    last: Exception | None = None
    for attempt in range(DOWNLOAD_RETRIES):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                expected = _server_md5(resp.headers)
                data = resp.read()
                last_modified = resp.headers.get("Last-Modified")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                # Contract: not an error. Recorded so the report can list it.
                with _prov_lock:
                    prov[fname] = {"url": url, "missing": True,
                                   "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
                return None, "missing"
            last = e
        except Exception as e:                      # network / timeout
            last = e
        else:
            local = hashlib.md5(data).hexdigest()
            if expected is None:
                raise SkeletonDownloadError(
                    f"{fname}: server sent no MD5 header; refusing to trust download")
            if local != expected:
                last = SkeletonDownloadError(
                    f"{fname}: MD5 mismatch local={local} server={expected}")
                continue
            tmp = dest.with_suffix(".swc.part")
            tmp.write_bytes(data)
            tmp.replace(dest)
            with _prov_lock:
                prov[fname] = {"url": url, "bytes": len(data), "md5": local,
                               "server_last_modified": last_modified,
                               "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
            return dest, "downloaded"
        time.sleep(0.5 * (attempt + 1))
    raise SkeletonDownloadError(f"{fname}: failed after {DOWNLOAD_RETRIES} attempts: {last!r}")


def download_many(body_ids, threads: int = DOWNLOAD_THREADS, force: bool = False,
                  progress: bool = True) -> tuple[dict[int, Path], list[int], dict]:
    """Fetch many skeletons concurrently. Returns (paths by bodyId, missing, counts)."""
    body_ids = [int(b) for b in body_ids]
    prov = _load_provenance()
    paths: dict[int, Path] = {}
    missing: list[int] = []
    counts = {"cached": 0, "downloaded": 0, "missing": 0}
    done = 0
    t0 = time.perf_counter()

    def one(b):
        return b, *download_skeleton(b, prov, force=force)

    with ThreadPoolExecutor(max_workers=threads) as ex:
        for b, path, status in ex.map(one, body_ids):
            counts[status] += 1
            if path is None:
                missing.append(b)
            else:
                paths[b] = path
            done += 1
            if progress and (done % 50 == 0 or done == len(body_ids)):
                el = time.perf_counter() - t0
                print(f"\r[skel] {done}/{len(body_ids)}  cached={counts['cached']} "
                      f"downloaded={counts['downloaded']} missing={counts['missing']}  {el:.0f}s",
                      end="", file=sys.stderr, flush=True)
    if progress:
        print(file=sys.stderr)
    _save_provenance(prov)
    return paths, sorted(missing), counts


# ---------------------------------------------------------------------------
# parse
# ---------------------------------------------------------------------------
def parse_swc(source) -> dict[str, np.ndarray]:
    """Parse SWC text (or a Path) into arrays.

    Columns are ``id type x y z radius parent``; lines starting with ``#`` and
    blank lines are skipped. Returns dict with ``id`` (int64), ``type`` (int32),
    ``xyz`` (float64 [n,3]), ``radius`` (float32), ``parent`` (int64).
    """
    if isinstance(source, Path):
        text = source.read_text()
    elif isinstance(source, str) and "\n" not in source:
        text = Path(source).read_text()         # a path given as a string
    else:
        text = str(source)                      # SWC text
    ids, types, xyz, radius, parent = [], [], [], [], []
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        f = line.split()
        if len(f) < 7:
            raise ValueError(f"SWC line {lineno}: expected 7 columns, got {len(f)}: {raw!r}")
        ids.append(int(float(f[0])))
        types.append(int(float(f[1])))
        xyz.append((float(f[2]), float(f[3]), float(f[4])))
        radius.append(float(f[5]))
        parent.append(int(float(f[6])))
    if not ids:
        raise ValueError("SWC contains no nodes")
    return {"id": np.array(ids, dtype=np.int64), "type": np.array(types, dtype=np.int32),
            "xyz": np.array(xyz, dtype=np.float64), "radius": np.array(radius, dtype=np.float32),
            "parent": np.array(parent, dtype=np.int64)}


def branch_paths(node_ids: np.ndarray, parent: np.ndarray) -> tuple[list[list[int]], int, int]:
    """Split a forest into branch paths. Returns (paths, n_branch_points, n_roots).

    Each path is a list of node positions running from a root or a branch point to
    the next branch point or endpoint, and **includes** the branch point it starts
    from, so consecutive paths join up. Branch points and endpoints are therefore
    path boundaries and can never be removed by the per-path decimation.
    """
    pos = {int(v): k for k, v in enumerate(node_ids)}
    n = len(node_ids)
    children: list[list[int]] = [[] for _ in range(n)]
    roots: list[int] = []
    for k in range(n):
        p = int(parent[k])
        pk = pos.get(p)
        if p < 0 or pk is None or pk == k:
            roots.append(k)
        else:
            children[pk].append(k)
    branch = [k for k in range(n) if len(children[k]) >= 2]
    starts = [(None, r) for r in roots] + [(k, c) for k in branch for c in children[k]]
    paths: list[list[int]] = []
    seen = np.zeros(n, dtype=bool)
    for prev, node in starts:
        cur = node
        chain = ([] if prev is None else [prev]) + [cur]
        while len(children[cur]) == 1:
            nxt = children[cur][0]
            if seen[nxt]:
                raise ValueError("SWC contains a cycle")
            seen[nxt] = True
            cur = nxt
            chain.append(cur)
        paths.append(chain)
    covered = np.zeros(n, dtype=bool)
    for chain in paths:
        covered[chain] = True
    if not covered.all():
        # nodes reachable from no root: a cycle, or a subtree hanging off one
        raise ValueError(f"SWC contains a cycle: {int((~covered).sum())} of {n} nodes "
                         f"are not reachable from any root")
    return paths, len(branch), len(roots)


# ---------------------------------------------------------------------------
# decimate
# ---------------------------------------------------------------------------
def rdp_mask(points: np.ndarray, tolerance: float) -> tuple[np.ndarray, float]:
    """Ramer-Douglas-Peucker over a 3D polyline.

    Returns (keep mask, max deviation actually incurred). The first and last
    points are always kept. Deviation is the perpendicular distance to the line
    through the retained segment's endpoints; the returned value is the largest
    such distance among the *dropped* points, so it is <= tolerance by
    construction and is what the report quotes.
    """
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    keep = np.zeros(n, dtype=bool)
    if n == 0:
        return keep, 0.0
    keep[0] = keep[-1] = True
    if n <= 2:
        return keep, 0.0
    max_dev = 0.0
    stack = [(0, n - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, s = pts[i], pts[j] - pts[i]
        p = pts[i + 1:j] - a
        l2 = float(s @ s)
        if l2 <= 0.0:
            d = np.linalg.norm(p, axis=1)
        else:
            d = np.linalg.norm(p - np.outer((p @ s) / l2, s), axis=1)
        k = int(np.argmax(d))
        if d[k] > tolerance:
            m = i + 1 + k
            keep[m] = True
            stack.append((i, m))
            stack.append((m, j))
        else:
            max_dev = max(max_dev, float(d[k]))
    return keep, max_dev


def simplify_neuron(nodes: dict[str, np.ndarray], tolerance: float = TOLERANCE_VOXELS,
                    max_segments: int | None = MAX_SEGMENTS_PER_NEURON) -> dict:
    """Decimate one parsed skeleton into line segments.

    With ``max_segments=None`` (the contract default) this is a single RDP pass at
    ``tolerance``. If an int is given, the tolerance is raised by
    ``_TOLERANCE_GROWTH`` while the neuron is over the cap; the branch count is a
    hard floor (one segment per path), so the loop stops there and the result
    records ``cap_met``. Nothing is ever pruned to force a cap.

    Returns dict with ``segments`` (float32 [S,2,3]), ``n_nodes_raw``,
    ``n_branch_points``, ``n_roots``, ``n_paths``, ``segment_floor``,
    ``tolerance_voxels`` (the final one), ``max_deviation_voxels``, ``cap_met``,
    ``tolerance_steps``.
    """
    xyz = nodes["xyz"]
    paths, n_branch, n_roots = branch_paths(nodes["id"], nodes["parent"])
    usable = [p for p in paths if len(p) >= 2]
    floor = len(usable)                       # one segment per path, whatever the tolerance
    tol = float(tolerance)
    steps = 0
    while True:
        segs: list[np.ndarray] = []
        max_dev = 0.0
        for path in usable:
            pts = xyz[path]
            keep, dev = rdp_mask(pts, tol)
            kept = pts[keep]
            if len(kept) >= 2:
                segs.append(np.stack([kept[:-1], kept[1:]], axis=1))
            max_dev = max(max_dev, dev)
        n_seg = int(sum(len(s) for s in segs))
        if max_segments is None or n_seg <= max_segments or n_seg <= floor \
                or steps >= _TOLERANCE_MAX_STEPS:
            break
        tol *= _TOLERANCE_GROWTH
        steps += 1
    segments = (np.concatenate(segs, axis=0).astype(np.float32) if segs
                else np.zeros((0, 2, 3), dtype=np.float32))
    return {"segments": segments, "n_nodes_raw": int(len(nodes["id"])),
            "n_branch_points": n_branch, "n_roots": n_roots, "n_paths": len(paths),
            "segment_floor": floor, "tolerance_voxels": tol,
            "max_deviation_voxels": float(max_dev),
            "cap_met": True if max_segments is None else bool(len(segments) <= max_segments),
            "tolerance_steps": steps}


# ---------------------------------------------------------------------------
# bundle
# ---------------------------------------------------------------------------
def build_bundle(body_ids, name: str, tolerance_voxels: float = TOLERANCE_VOXELS,
                 max_segments_per_neuron: int | None = MAX_SEGMENTS_PER_NEURON,
                 graph=None, out_dir: Path = CACHE_DIR, threads: int = DOWNLOAD_THREADS,
                 force: bool = False, progress: bool = True) -> tuple[Path, Path]:
    """Build ``skel-<name>.bin`` / ``.json`` for ``body_ids`` (contract file 2).

    bodyIds absent from the retained graph are dropped (contract); bodyIds whose
    skeleton is a 404 go into ``missing``. Raises SkeletonBundleTooLarge if the
    ``.bin`` would exceed the contract's 20 MB ceiling.
    """
    if graph is None:
        from ..graph import Graph
        graph = Graph.load()
    requested = np.unique(np.asarray(list(body_ids), dtype=np.int64))
    if requested.size == 0:
        raise ValueError(f"bundle {name!r}: no bodyIds given")
    in_graph = np.isin(requested, graph.body_ids)
    not_in_graph = [int(b) for b in requested[~in_graph]]
    kept = requested[in_graph]
    if kept.size == 0:
        raise LookupError(f"bundle {name!r}: none of the {requested.size} bodyIds is in the "
                          f"retained graph")
    idx = graph.idx(kept)
    order = np.argsort(idx)                    # graph-index order, deterministic
    kept, idx = kept[order], np.asarray(idx)[order]

    paths, missing, counts = download_many(kept, threads=threads, force=force, progress=progress)

    neurons, blocks, stats = [], [], []
    seg_offset = 0
    for body_id, neuron_idx in zip(kept.tolist(), idx.tolist()):
        p = paths.get(body_id)
        if p is None:
            continue                            # 404 -> missing list
        res = simplify_neuron(parse_swc(p), tolerance_voxels, max_segments_per_neuron)
        seg = res["segments"]
        soma = graph.soma_xyz[neuron_idx]
        neurons.append({
            "body_id": int(body_id), "idx": int(neuron_idx),
            "seg_offset": seg_offset, "seg_count": int(len(seg)),
            "soma": [float(v) for v in soma] if bool(graph.has_soma_xyz[neuron_idx]) else None,
            "n_nodes_raw": res["n_nodes_raw"],
        })
        blocks.append(seg)
        stats.append(res)
        seg_offset += len(seg)

    if not neurons:
        raise LookupError(f"bundle {name!r}: every bodyId is missing a skeleton "
                          f"({len(missing)} of {len(kept)})")
    allseg = np.concatenate(blocks, axis=0) if blocks else np.zeros((0, 2, 3), dtype=np.float32)
    nbytes = allseg.size * 4
    if nbytes > MAX_BUNDLE_BYTES:
        raise SkeletonBundleTooLarge(
            f"bundle {name!r}: {len(allseg):,} segments = {nbytes / 2**20:.1f} MB exceeds the "
            f"{MAX_BUNDLE_BYTES / 2**20:.0f} MB contract ceiling "
            f"(docs/m6-3d-contract.md). Raise tolerance_voxels (currently "
            f"{tolerance_voxels}); note the per-neuron floor is the branch count, so a set "
            f"with {len(neurons):,} neurons cannot go below "
            f"{sum(s['segment_floor'] for s in stats) * BYTES_PER_SEGMENT / 2**20:.1f} MB "
            f"while branch points are preserved.")

    pts = allseg.reshape(-1, 3)
    out_dir.mkdir(parents=True, exist_ok=True)
    bin_path = out_dir / f"skel-{name}.bin"
    json_path = out_dir / f"skel-{name}.json"
    bin_path.write_bytes(allseg.astype(np.float32).tobytes(order="C"))
    tol_final = sorted({round(s["tolerance_voxels"], 6) for s in stats})
    meta = {
        "set": name, "voxel_nm": VOXEL_NM, "n_neurons": len(neurons), "n_segments": int(len(allseg)),
        "bbox": {"min": [float(v) for v in pts.min(axis=0)],
                 "max": [float(v) for v in pts.max(axis=0)]},
        "decimation": {
            "method": "rdp-per-branch",
            "tolerance_voxels": float(tolerance_voxels),
            "max_segments_per_neuron": (None if max_segments_per_neuron is None
                                        else int(max_segments_per_neuron)),
            # Contract v1.1: the tolerance actually used (one value per set unless a
            # cap forced per-neuron retries) and the deviation it incurred.
            "tolerance_voxels_final": tol_final,
            "max_deviation_voxels": max((s["max_deviation_voxels"] for s in stats), default=0.0),
            "neurons_over_cap": int(sum(not s["cap_met"] for s in stats)),
            "segment_floor_total": int(sum(s["segment_floor"] for s in stats)),
        },
        "neurons": neurons,
        "missing": missing,
        "not_in_graph": not_in_graph,
        "source": {"url_prefix": URL_PREFIX, "md5_manifest": MANIFEST_REL},
        "generated": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
    }
    json_path.write_text(json.dumps(meta, indent=1) + "\n")
    if progress:
        print(f"[bundle] {name}: {len(neurons):,} neurons, {len(allseg):,} segments, "
              f"{nbytes / 2**20:.2f} MB, missing={len(missing)}, "
              f"over-cap={meta['decimation']['neurons_over_cap']}, "
              f"download {counts}", flush=True)
    return bin_path, json_path


# ---------------------------------------------------------------------------
# probe-set resolution and CLI
# ---------------------------------------------------------------------------
def load_probe_sets(graph):
    """Probe sets from flysim.probe.sets, used unchanged (that module is not ours)."""
    from ..probe.sets import build_probe_sets, load_roi
    return build_probe_sets(graph, load_roi(graph))


def probe_set_body_ids(graph, name: str, ps=None) -> np.ndarray:
    """bodyIds of a probe set by base name, both sides merged (JO_AB_L + JO_AB_R -> JO_AB).

    Set definitions come from flysim.probe.sets unchanged. Raises LookupError for
    an unknown name, listing what is available (CLAUDE.md §3.1: never an empty set).
    """
    ps = ps if ps is not None else load_probe_sets(graph)
    parts = [k for k in ps.names if k == name or k.startswith(f"{name}_")]
    if not parts:
        bases = sorted({k.rsplit("_", 1)[0] if k.rsplit("_", 1)[-1] in ("L", "R", "unk") else k
                        for k in ps.names})
        raise LookupError(f"unknown probe set {name!r}; available base names: {bases}")
    idx = np.unique(np.concatenate([ps.members[k] for k in parts]))
    if idx.size == 0:
        raise LookupError(f"probe set {name!r} is empty")
    return np.asarray(graph.body(np.sort(idx)), dtype=np.int64)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", dest="sets", action="append", required=True,
                    help="probe set base name (repeatable), e.g. --set JO_AB --set pC1")
    ap.add_argument("--tolerance", type=float, default=None,
                    help=f"RDP tolerance in voxels (default {TOLERANCE_VOXELS:g}, "
                         f"{TOLERANCE_BY_SET} per contract v1.1; 1 voxel = 8 nm)")
    ap.add_argument("--max-segments", type=int, default=MAX_SEGMENTS_PER_NEURON,
                    help="optional per-neuron segment cap; omit for the contract default "
                         "(no cap). Setting it degrades geometry — see docs/m6a-report.md §5")
    ap.add_argument("--threads", type=int, default=DOWNLOAD_THREADS)
    ap.add_argument("--force", action="store_true", help="re-download even if the cached MD5 matches")
    a = ap.parse_args(argv)

    from ..graph import Graph
    graph = Graph.load()
    ps = load_probe_sets(graph)
    rc = 0
    for name in a.sets:
        body_ids = probe_set_body_ids(graph, name, ps)
        tol = a.tolerance if a.tolerance is not None else TOLERANCE_BY_SET.get(name, TOLERANCE_VOXELS)
        print(f"[set] {name}: {len(body_ids):,} neurons in the retained graph, "
              f"tolerance {tol:g} voxels", flush=True)
        try:
            build_bundle(body_ids, name, tol, a.max_segments, graph=graph,
                         threads=a.threads, force=a.force)
        except SkeletonBundleTooLarge as e:
            print(f"[FAIL] {e}", file=sys.stderr)
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
