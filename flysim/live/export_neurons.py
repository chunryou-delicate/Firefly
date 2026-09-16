"""M5a: build the live session's context and the cockpit's ``neurons-v1.json``.

Coordinates, probe sets and JO targets all come from the existing modules
(``flysim.probe.export.neuron_coords``, ``flysim.probe.sets.build_probe_sets``,
``flysim.sensory.jo.jo_targets``), so the cockpit paints the same neurons in the same places
as the offline viewer. Nothing here is live-specific except where the file is written.

Usage: python -m flysim.live.export_neurons [--force]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from ..graph import Graph
from ..probe.export import neuron_coords, write_neurons_json
from ..probe.sets import build_probe_sets, load_roi
from ..sensory.jo import jo_targets
from .session import RUNS_LIVE, LiveContext

NEURONS_JSON = RUNS_LIVE / "neurons-v1.json"
K_MIN = 5          # JO_post membership, same as M3/M4 (ASSUMPTION, docs/m3-report.md)


def build_context(*, k_min: int = K_MIN, neurons_url: str = "/neurons-v1.json") -> LiveContext:
    graph = Graph.load()
    roi = load_roi(graph)
    n_sites = roi["n_sites"].to_numpy()
    sets = build_probe_sets(graph, roi, k_min=k_min)
    coords = neuron_coords(graph, roi)
    tg = jo_targets(graph, n_sites)
    return LiveContext(graph=graph, sets=sets, jo_left=tg["left"], jo_right=tg["right"],
                       coords=coords, neurons_url=neurons_url)


def write_neurons(ctx: LiveContext, path: Path = NEURONS_JSON, force: bool = False) -> Path:
    if path.exists() and not force:
        return path
    if ctx.coords is None:
        raise ValueError("context has no coordinates")
    path.parent.mkdir(parents=True, exist_ok=True)
    write_neurons_json(ctx.coords, ctx.sets.region_of, ctx.sets.names, path=path)
    return path


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    t0 = time.time()
    ctx = build_context()
    path = write_neurons(ctx, force=args.force)
    print(f"[live] {path} ({path.stat().st_size / 1e6:.1f} MB) — {ctx.n:,} neurons, "
          f"{len(ctx.sets.names)} regions, JO targets {len(ctx.jo_left)} L / {len(ctx.jo_right)} R "
          f"in {time.time() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
