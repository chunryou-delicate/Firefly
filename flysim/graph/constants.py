"""Shared constants for the M1 graph cache (codes, sign rule, cache paths).

Kept import-light so that `python -m flysim.graph.build` does not pull the
whole package first. Value sets verified in docs/annotations-observed.md.
"""
from __future__ import annotations

from ..data.download import ROOT

CACHE_DIR = ROOT / "data" / "cache"
CACHE_NPZ = CACHE_DIR / "graph-v1.npz"
CACHE_ANN = CACHE_DIR / "graph-v1-ann.parquet"

# Integer codes for per-neuron attribute arrays. -1 == null everywhere.
SIDE_CODES = {"L": 0, "R": 1, "M": 2}

# consensus_nt values observed in docs/annotations-observed.md, plus "missing"
# for retained neurons absent from the NT table.
NT_CODES = {
    "acetylcholine": 0,
    "gaba": 1,
    "glutamate": 2,
    "unclear": 3,
    "histamine": 4,
    "dopamine": 5,
    "octopamine": 6,
    "serotonin": 7,
    "missing": 8,
}
# Edge sign per presynaptic NT code (data-provenance/edge-signs.md).
NT_SIGN = {
    0: +1.0,  # acetylcholine   standard
    1: -1.0,  # gaba            standard
    2: -1.0,  # glutamate       assumption (GluCl, Drosophila CNS)
    3: +1.0,  # unclear         ASSUMPTION (ACh majority prior)
    4: +1.0,  # histamine       ASSUMPTION (modulator mask)
    5: +1.0,  # dopamine        ASSUMPTION (modulator mask)
    6: +1.0,  # octopamine      ASSUMPTION (modulator mask)
    7: +1.0,  # serotonin       ASSUMPTION (modulator mask)
    8: +1.0,  # missing         ASSUMPTION (ACh majority prior)
}
NT_MODULATOR_CODES = (4, 5, 6, 7)
NT_ASSUMED_CODES = (2, 3, 4, 5, 6, 7, 8)   # everything except ACh / GABA
