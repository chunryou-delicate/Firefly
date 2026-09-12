"""M0: compare the loaded data against the reference figures in CLAUDE.md §4.2.

Retention rule under test: neurons with annotation `status == "Traced"`, and
edges whose both endpoints are such neurons. This is the only filter applied
and it is reported with before/after counts. Any deviation > 10% from the
reference aborts with a non-zero exit code.

Usage: python -m flysim.data.verify
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import pyarrow.parquet as pq

from .convert import PARQUET_DIR
from .download import ROOT

REFERENCE = {  # CLAUDE.md §4.2
    "neurons": 166_700,
    "directed_edges": 25_580_000,
    "synapse_contacts": 124_000_000,
    "cell_types": 11_691,
}
TOLERANCE = 0.10
RETAIN_STATUS = "Traced"  # observed value, see docs/annotations-observed.md
REPORT = ROOT / "data-provenance" / "m0-verify.json"


def main() -> int:
    t0 = time.time()
    ann = pq.read_table(PARQUET_DIR / "annotations.parquet",
                        columns=["bodyId", "status", "type", "somaSide", "somaLocation"]).to_pandas()
    nt = pq.read_table(PARQUET_DIR / "neurotransmitters.parquet",
                       columns=["body", "consensus_nt", "predicted_nt"]).to_pandas()
    w = pq.read_table(PARQUET_DIR / "weights.parquet").to_pandas()
    print(f"loaded in {time.time()-t0:.1f}s")

    keep = ann["status"] == RETAIN_STATUS
    ids = ann.loc[keep, "bodyId"].to_numpy()
    idset = np.isin(w["body_pre"].to_numpy(), ids) & np.isin(w["body_post"].to_numpy(), ids)

    observed = {
        "neurons": int(keep.sum()),
        "directed_edges": int(idset.sum()),
        "synapse_contacts": int(w.loc[idset, "weight"].sum()),
        "cell_types": int(ann.loc[keep, "type"].nunique()),
    }
    before = {
        "annotation_rows": int(len(ann)),
        "weight_rows": int(len(w)),
        "weight_sum": int(w["weight"].sum()),
    }

    # NT coverage among retained neurons (needed for edge signs in M1)
    ntm = nt.set_index("body").reindex(ids)
    nt_cov = {
        "consensus_nt": ntm["consensus_nt"].value_counts(dropna=False).to_dict(),
        "predicted_nt": ntm["predicted_nt"].value_counts(dropna=False).to_dict(),
        "missing_from_nt_table": int(ntm["consensus_nt"].isna().sum()),
    }
    nt_cov = {k: ({str(kk): int(vv) for kk, vv in v.items()} if isinstance(v, dict) else v)
              for k, v in nt_cov.items()}
    side = ann.loc[keep, "somaSide"].value_counts(dropna=False)
    soma = int(ann.loc[keep, "somaLocation"].notna().sum())

    print(f"\nRetention rule: status == {RETAIN_STATUS!r}")
    print(f"  annotation rows {before['annotation_rows']} -> neurons {observed['neurons']}")
    print(f"  weight rows     {before['weight_rows']} -> edges   {observed['directed_edges']}")
    print(f"  synapse sum     {before['weight_sum']} -> {observed['synapse_contacts']}")
    print("\n| item | reference | observed | deviation |")
    print("|---|---|---|---|")
    ok = True
    dev = {}
    for k, ref in REFERENCE.items():
        d = (observed[k] - ref) / ref
        dev[k] = d
        flag = "" if abs(d) <= TOLERANCE else "  <-- OUT OF TOLERANCE"
        ok &= abs(d) <= TOLERANCE
        print(f"| {k} | {ref:,} | {observed[k]:,} | {d:+.2%} |{flag}")
    print(f"\nsomaSide among retained: {side.to_dict()}")
    print(f"somaLocation present among retained: {soma}/{observed['neurons']}")
    print(f"NT (consensus_nt) among retained: {nt_cov['consensus_nt']}")
    print(f"retained neurons missing from NT table: {nt_cov['missing_from_nt_table']}")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps({
        "retention_rule": f"annotations.status == {RETAIN_STATUS!r}; edges with both endpoints retained",
        "before": before, "observed": observed, "reference": REFERENCE,
        "deviation": dev, "tolerance": TOLERANCE, "pass": bool(ok),
        "somaSide_retained": {str(k): int(v) for k, v in side.items()},
        "somaLocation_present_retained": soma,
        "nt_coverage_retained": nt_cov,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }, indent=2) + "\n")
    print(f"\n{'PASS' if ok else 'FAIL'} — report written to {REPORT.relative_to(ROOT)}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
