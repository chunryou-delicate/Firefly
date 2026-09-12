"""M1: build the CSR graph cache from the M0 parquet files.

Retention (approved, see data-provenance/retention.md):
    keep neuron  iff annotations.status == "Traced"
    keep edge    iff both endpoints are kept
    no weight threshold, nothing else filtered.

Edge sign (approved, see data-provenance/edge-signs.md):
    sign is a property of the PRESYNAPTIC neuron (Dale's principle — ASSUMPTION)
    taken from neurotransmitters.consensus_nt:
        acetylcholine            -> +1
        gaba, glutamate          -> -1
        unclear / missing        -> +1   ASSUMPTION (ACh majority prior)
        histamine, dopamine,
        octopamine, serotonin    -> +1   ASSUMPTION; flagged in `nt_modulator` mask
Weights are the raw synaptic contact counts (float32, signed). No normalisation.

Outputs:
    data/cache/graph-v1.npz             arrays (see Graph docstring)
    data/cache/graph-v1-ann.parquet     annotation rows aligned with neuron index
    data-provenance/m1-build.json       counts logged by this script
    data-provenance/edge-signs.md       per-sign edge counts appended

Usage: python -m flysim.graph.build [--force]
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ..data.convert import PARQUET_DIR
from ..data.download import ROOT
from .constants import (CACHE_ANN, CACHE_DIR, CACHE_NPZ, NT_ASSUMED_CODES, NT_CODES,
                        NT_MODULATOR_CODES, NT_SIGN, SIDE_CODES)

BUILD_LOG = ROOT / "data-provenance" / "m1-build.json"
EDGE_SIGNS_MD = ROOT / "data-provenance" / "edge-signs.md"

RETAIN_STATUS = "Traced"              # observed value, docs/annotations-observed.md
EXPECTED_NEURONS = 165_122            # m1-brief.md: abort if different
EXPECTED_EDGES = 25_563_197

# Annotation columns kept next to the graph for the query API. All names
# verified against docs/annotations-observed.md.
ANN_COLUMNS = [
    "bodyId", "type", "instance", "somaSide", "superclass", "class",
    "fruDsx", "flywireType", "hemibrainType", "mancType", "dimorphism",
    "somaNeuromere", "entryNerve", "exitNerve", "receptorType", "group",
    "somaLocation", "status",
]


def log(msg: str) -> None:
    print(f"[build] {msg}", flush=True)


def _csr_from_edges(src: np.ndarray, dst: np.ndarray, w: np.ndarray, n: int):
    """Sort edges by (src, dst) and return indptr(int64), indices(int32), weight(float32)."""
    order = np.lexsort((dst, src))
    src, dst, w = src[order], dst[order], w[order]
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.bincount(src, minlength=n), out=indptr[1:])
    return indptr, dst.astype(np.int32), w.astype(np.float32)


def build(force: bool = False) -> dict:
    if CACHE_NPZ.exists() and CACHE_ANN.exists() and not force:
        log(f"cache exists: {CACHE_NPZ.relative_to(ROOT)} (use --force to rebuild)")
        return json.loads(BUILD_LOG.read_text()) if BUILD_LOG.exists() else {}

    t0 = time.time()
    timings = {}

    # ---- neurons ---------------------------------------------------------
    ann = pq.read_table(PARQUET_DIR / "annotations.parquet", columns=ANN_COLUMNS).to_pandas()
    n_rows = len(ann)
    keep = ann["status"] == RETAIN_STATUS
    ann = ann.loc[keep].copy()
    log(f"neuron filter: status == {RETAIN_STATUS!r}: {n_rows} -> {len(ann)} "
        f"(dropped {n_rows - len(ann)})")
    if len(ann) != EXPECTED_NEURONS:
        raise RuntimeError(f"retained neurons {len(ann)} != expected {EXPECTED_NEURONS}; "
                           "retention rule changed? abort and report")
    ann = ann.sort_values("bodyId").reset_index(drop=True)
    body_ids = ann["bodyId"].to_numpy(dtype=np.int64)
    assert np.all(np.diff(body_ids) > 0), "bodyId not unique/sorted"
    n = len(body_ids)
    timings["annotations_s"] = round(time.time() - t0, 1)

    # side code: somaSide first; if null, instance suffix _L/_R/_M (fallback)
    side = ann["somaSide"].map(SIDE_CODES)
    suffix = ann["instance"].str.extract(r"_([LRM])$")[0].map(SIDE_CODES)
    side_from_instance = side.isna() & suffix.notna()
    side = side.fillna(suffix).fillna(-1).to_numpy(dtype=np.int8)
    log(f"side: somaSide {int((ann['somaSide'].notna()).sum())}, "
        f"+ instance-suffix fallback {int(side_from_instance.sum())}, "
        f"null {int((side == -1).sum())}")

    # soma xyz
    has_soma = ann["somaLocation"].notna().to_numpy()
    soma_xyz = np.full((n, 3), np.nan, dtype=np.float32)
    soma_xyz[has_soma] = np.stack(ann.loc[has_soma, "somaLocation"].to_list()).astype(np.float32)
    log(f"soma_xyz present {int(has_soma.sum())}/{n}")

    # ---- neurotransmitter per neuron --------------------------------------
    nt_tab = pq.read_table(PARQUET_DIR / "neurotransmitters.parquet",
                           columns=["body", "consensus_nt"]).to_pandas()
    nt_ser = nt_tab.set_index("body")["consensus_nt"].reindex(body_ids)
    nt_missing = nt_ser.isna()
    nt_ser = nt_ser.fillna("missing")
    unknown = set(nt_ser.unique()) - set(NT_CODES)
    if unknown:
        raise RuntimeError(f"unexpected consensus_nt values {unknown}; extend NT_CODES explicitly")
    nt = nt_ser.map(NT_CODES).to_numpy(dtype=np.int8)
    nt_counts = {name: int((nt == code).sum()) for name, code in NT_CODES.items()}
    log(f"consensus_nt among retained: {nt_counts} (missing from table: {int(nt_missing.sum())})")
    neuron_sign = np.array([NT_SIGN[c] for c in range(len(NT_CODES))], dtype=np.float32)[nt]
    nt_modulator = np.isin(nt, NT_MODULATOR_CODES)
    nt_assumed = np.isin(nt, NT_ASSUMED_CODES)
    log(f"neurons with sign by assumption: {int(nt_assumed.sum())}, "
        f"modulator-flagged: {int(nt_modulator.sum())}")
    del nt_tab, nt_ser
    timings["nt_s"] = round(time.time() - t0, 1)

    # ---- edges -------------------------------------------------------------
    t1 = time.time()
    w = pq.read_table(PARQUET_DIR / "weights.parquet")
    pre = w.column("body_pre").to_numpy()
    post = w.column("body_post").to_numpy()
    wt = w.column("weight").to_numpy()
    del w
    n_raw_edges, raw_syn = len(pre), int(wt.sum())
    log(f"weights loaded: {n_raw_edges} rows, {raw_syn} contacts ({time.time()-t1:.1f}s)")

    def to_idx(b: np.ndarray) -> np.ndarray:
        pos = np.searchsorted(body_ids, b)
        pos[pos == n] = 0
        ok = body_ids[pos] == b
        return np.where(ok, pos, -1)

    src = to_idx(pre)
    dst = to_idx(post)
    del pre, post
    both = (src >= 0) & (dst >= 0)
    n_pre_only = int(((src >= 0) & (dst < 0)).sum())
    n_post_only = int(((src < 0) & (dst >= 0)).sum())
    src, dst, wt = src[both], dst[both], wt[both]
    m = len(src)
    syn = int(wt.sum())
    log(f"edge filter: both endpoints retained: {n_raw_edges} -> {m} "
        f"(dropped {n_raw_edges - m}: pre-only {n_pre_only}, post-only {n_post_only}, "
        f"neither {n_raw_edges - m - n_pre_only - n_post_only}); contacts {raw_syn} -> {syn}")
    if m != EXPECTED_EDGES:
        raise RuntimeError(f"retained edges {m} != expected {EXPECTED_EDGES}; abort and report")
    if wt.min() <= 0:
        raise RuntimeError("non-positive raw weight found; contact counts must be >= 1")
    self_loops = int((src == dst).sum())
    log(f"self loops retained: {self_loops}")
    timings["edge_filter_s"] = round(time.time() - t1, 1)

    # sign by presynaptic neuron. Statistics are computed on the int64 counts
    # (float32 sums of 25M values drift by tens of contacts).
    wt = wt.astype(np.int64)
    sign_stats = {}
    for name, code in NT_CODES.items():
        e = nt[src] == code
        sign_stats[name] = {
            "sign": int(NT_SIGN[code]),
            "edges": int(e.sum()),
            "contacts": int(wt[e].sum()),
            "assumed": code in NT_ASSUMED_CODES,
            "modulator": code in NT_MODULATOR_CODES,
        }
    edge_sign = neuron_sign[src]
    pos_e, neg_e = int((edge_sign > 0).sum()), int((edge_sign < 0).sum())
    contacts_pos, contacts_neg = int(wt[edge_sign > 0].sum()), int(wt[edge_sign < 0].sum())
    assert pos_e + neg_e == m and contacts_pos + contacts_neg == syn
    log(f"signed edges: +{pos_e} / -{neg_e}; contacts +{contacts_pos} / -{contacts_neg}")
    wt = wt.astype(np.float32) * edge_sign

    t2 = time.time()
    indptr, indices, weight = _csr_from_edges(src, dst, wt, n)
    csc_indptr, csc_indices, csc_weight = _csr_from_edges(dst, src, wt, n)
    timings["csr_build_s"] = round(time.time() - t2, 1)
    log(f"CSR + CSC built in {time.time()-t2:.1f}s")

    arrays = dict(
        body_ids=body_ids,
        indptr=indptr, indices=indices, weight=weight,
        csc_indptr=csc_indptr, csc_indices=csc_indices, csc_weight=csc_weight,
        side=side, nt=nt, nt_modulator=nt_modulator, nt_assumed=nt_assumed,
        has_soma_xyz=has_soma, soma_xyz=soma_xyz,
    )
    nbytes = {k: int(v.nbytes) for k, v in arrays.items()}
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE_NPZ, **arrays)
    ann.drop(columns=["status"]).to_parquet(CACHE_ANN, index=False)
    log(f"cache written: {CACHE_NPZ.relative_to(ROOT)} ({CACHE_NPZ.stat().st_size/1e6:.0f} MB), "
        f"{CACHE_ANN.relative_to(ROOT)} ({CACHE_ANN.stat().st_size/1e6:.0f} MB)")

    report = {
        "retention_rule": f"annotations.status == {RETAIN_STATUS!r}; edges with both endpoints retained; no weight threshold",
        "neurons": {"annotation_rows": n_rows, "retained": n,
                    "side_counts": {k: int((side == v).sum()) for k, v in SIDE_CODES.items()}
                    | {"null": int((side == -1).sum())},
                    "side_from_instance_suffix": int(side_from_instance.sum()),
                    "soma_xyz_present": int(has_soma.sum())},
        "edges": {"weight_rows": n_raw_edges, "retained": m,
                  "dropped_pre_only": n_pre_only, "dropped_post_only": n_post_only,
                  "contacts_raw": raw_syn, "contacts_retained": syn,
                  "self_loops": self_loops,
                  "positive": pos_e, "negative": neg_e,
                  "contacts_positive": contacts_pos,
                  "contacts_negative": contacts_neg},
        "sign_rule": "sign of presynaptic neuron's consensus_nt (Dale's principle, assumption)",
        "nt_codes": NT_CODES, "nt_sign": {k: NT_SIGN[v] for k, v in NT_CODES.items()},
        "neuron_nt_counts": nt_counts,
        "edges_by_pre_nt": sign_stats,
        "side_codes": SIDE_CODES,
        "cache_bytes": nbytes, "cache_bytes_total": sum(nbytes.values()),
        "timings_s": timings | {"total_s": round(time.time() - t0, 1)},
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    BUILD_LOG.parent.mkdir(parents=True, exist_ok=True)
    BUILD_LOG.write_text(json.dumps(report, indent=2) + "\n")
    _append_edge_signs_md(report)
    log(f"done in {time.time()-t0:.1f}s; log -> {BUILD_LOG.relative_to(ROOT)}")
    return report


def _append_edge_signs_md(report: dict) -> None:
    marker = "## Observed per-sign edge counts (M1 build)"
    text = EDGE_SIGNS_MD.read_text() if EDGE_SIGNS_MD.exists() else "# Edge sign decision\n"
    if marker in text:
        text = text[: text.index(marker)].rstrip() + "\n"
    rows = ["", marker, "", f"Generated by `python -m flysim.graph.build` on {report['generated_at']}.",
            "", "| pre consensus_nt | sign | neurons | edges | contacts | basis |", "|---|---|---|---|---|---|"]
    for name, s in report["edges_by_pre_nt"].items():
        basis = "assumption" if s["assumed"] else "standard"
        if s["modulator"]:
            basis += ", modulator mask"
        rows.append(f"| {name} | {s['sign']:+d} | {report['neuron_nt_counts'][name]:,} | "
                    f"{s['edges']:,} | {s['contacts']:,} | {basis} |")
    e = report["edges"]
    rows += ["", f"Total: {e['positive']:,} excitatory edges ({e['contacts_positive']:,} contacts), "
             f"{e['negative']:,} inhibitory edges ({e['contacts_negative']:,} contacts). "
             f"Self loops retained: {e['self_loops']:,}.", ""]
    EDGE_SIGNS_MD.write_text(text.rstrip() + "\n" + "\n".join(rows))


def main(argv: list[str]) -> int:
    build(force="--force" in argv)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
