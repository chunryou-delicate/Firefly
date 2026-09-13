"""M3: probe neuron sets (docs/m3-brief.md "probe / sets.py").

Every set is built by querying the data (types via ``Graph.query``, ROIs via
data/cache/neuron-roi-v1.parquet). Empty sets raise. Sets are made disjoint by
precedence (order of ``ORDER`` below) so that each neuron has one viewer
region; the raw (pre-precedence) sizes and the overlaps removed are kept in
``ProbeSets.log`` and go into the report.

Identifiers used here are verified in docs/annotations-observed.md,
data-provenance/roi-observed.md and docs/m3-brief.md:
    type patterns  ^JO-[AB]  ^AMMC  ^pC1
    ROI labels     SAD  WED(L)  WED(R)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..data.download import ROOT
from ..graph.constants import SIDE_CODES
from ..sensory.jo import jo_targets

ROI_CACHE = ROOT / "data" / "cache" / "neuron-roi-v1.parquet"
K_MIN_DEFAULT = 5          # ASSUMPTION: min summed contacts from JO-A/B to count as direct postsynaptic
SAD_ROI = "SAD"
WED_ROI = {"L": "WED(L)", "R": "WED(R)"}
AMMC_TYPE_PATTERN = r"^AMMC"
PC1_TYPE_PATTERN = r"^pC1"
REST = "rest"

# Precedence for disjointness (first wins). Names with _L/_R/_unk are expanded in this order.
ORDER = ["JO_AB", "JO_post", "pC1", "AMMCtype", "SAD", "WED"]


def load_roi(graph) -> pd.DataFrame:
    if not ROI_CACHE.exists():
        raise FileNotFoundError(f"{ROI_CACHE} missing; run python -m flysim.data.roi")
    roi = pd.read_parquet(ROI_CACHE)
    if not np.array_equal(roi["bodyId"].to_numpy(), graph.body_ids):
        raise ValueError("neuron-roi cache is not aligned with the graph's body_ids")
    return roi


@dataclass
class ProbeSets:
    names: list[str]                       # region order for run.json (REST last)
    members: dict[str, np.ndarray]         # name -> sorted neuron indices (disjoint)
    region_of: np.ndarray                  # [N] int16 index into names
    log: dict = field(default_factory=dict)

    @property
    def n_regions(self) -> int:
        return len(self.names)

    def sizes(self) -> dict[str, int]:
        return {k: int(len(v)) for k, v in self.members.items()}


def _split_side(graph, idx: np.ndarray) -> dict[str, np.ndarray]:
    side = graph.side[idx]
    out = {"L": idx[side == SIDE_CODES["L"]], "R": idx[side == SIDE_CODES["R"]]}
    unk = idx[(side != SIDE_CODES["L"]) & (side != SIDE_CODES["R"])]
    if len(unk):
        out["unk"] = unk
    return out


def build_probe_sets(graph, roi: pd.DataFrame, k_min: int = K_MIN_DEFAULT) -> ProbeSets:
    q = graph.query
    n_sites = roi["n_sites"].to_numpy()
    primary = roi["primary_roi"].astype(object).to_numpy()
    raw: dict[str, np.ndarray] = {}
    log: dict = {"k_min": k_min}

    # JO_AB: identical to the adapter targets
    tg = jo_targets(graph, n_sites)
    raw["JO_AB_L"], raw["JO_AB_R"] = tg["left"], tg["right"]
    jo_all = np.concatenate([tg["left"], tg["right"]])
    log["JO_AB"] = {"n_typed": tg["n_typed"], "dropped_no_sites": int(len(tg["dropped_no_sites"])),
                    "types": [t for t, _ in tg["types"]]}

    # JO_post: direct postsynaptic partners with summed |contacts| >= k_min, side of the post neuron
    rows = np.repeat(np.arange(graph.n), graph.out_degree())
    m = np.isin(rows, jo_all)
    contacts = np.bincount(graph.indices[m], weights=np.abs(graph.weight[m]), minlength=graph.n)
    post = np.flatnonzero(contacts >= k_min)
    if len(post) == 0:
        raise LookupError(f"no neuron receives >= {k_min} contacts from JO-A/B")
    for s, v in _split_side(graph, post).items():
        raw[f"JO_post_{s}"] = v
    log["JO_post"] = {"n_any_contact": int((contacts > 0).sum()), "n_ge_k_min": int(len(post)),
                      "contacts_from_JO_AB_max": float(contacts.max())}

    # pC1 / AMMC-named types
    for name, pat in (("pC1", PC1_TYPE_PATTERN), ("AMMCtype", AMMC_TYPE_PATTERN)):
        types = q.search_type(pat)
        idx = np.unique(np.concatenate([q.by_type(t) for t, _ in types]))
        for s, v in _split_side(graph, idx).items():
            raw[f"{name}_{s}"] = v
        log[name] = {"n_types": len(types), "n": int(len(idx))}

    # SAD by primary ROI, side of the neuron
    sad = np.flatnonzero(primary == SAD_ROI)
    if len(sad) == 0:
        raise LookupError(f"no neuron with primary_roi == {SAD_ROI!r}")
    for s, v in _split_side(graph, sad).items():
        raw[f"SAD_{s}"] = v
    # WED by ROI label (side is in the label)
    for s, lab in WED_ROI.items():
        idx = np.flatnonzero(primary == lab)
        if len(idx) == 0:
            raise LookupError(f"no neuron with primary_roi == {lab!r}")
        raw[f"WED_{s}"] = idx

    # disjoint by precedence
    names = [k for base in ORDER for k in raw if k == f"{base}_L" or k == f"{base}_R" or k == f"{base}_unk"]
    assert set(names) == set(raw), (set(raw) - set(names))
    region_of = np.full(graph.n, -1, dtype=np.int16)
    members: dict[str, np.ndarray] = {}
    removed: dict[str, int] = {}
    for r, k in enumerate(names):
        idx = raw[k]
        free = idx[region_of[idx] == -1]
        removed[k] = int(len(idx) - len(free))
        if len(free) == 0:
            raise LookupError(f"probe set {k} is empty after precedence")
        region_of[free] = r
        members[k] = np.sort(free)
    rest = np.flatnonzero(region_of == -1)
    region_of[rest] = len(names)
    names.append(REST)
    members[REST] = rest
    log["raw_sizes"] = {k: int(len(v)) for k, v in raw.items()}
    log["removed_by_precedence"] = removed
    log["precedence"] = ORDER
    return ProbeSets(names, members, region_of, log)
