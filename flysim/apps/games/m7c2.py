"""M7c2 phase 2: the rematch — warm-up + the calibrated decoder (docs/m7c2-brief.md §2).

    python -m flysim.apps.games.m7c2                 # both arms, results, the main arm's bundles
    python -m flysim.apps.games.m7c2 --arms aux      # the auxiliary arm only (needs no phase-1 choice)

What changes against M7c, and nothing else:

* **Warm-up**: every episode (and every control) resets the engine with the game seed, runs
  ``WARMUP_STEPS`` steps with no input, then starts the game clock at 0. Warm-up spikes reach
  neither the decoder nor the statistics (``FlyBrain(warmup_steps=...)``).
* **Decoder**: arm ``main`` uses ``CountWindowDecoder(W*, n*)`` chosen in phase 1 from circuit
  measurements alone (``data-provenance/m7c2-decoder.json``), the same for all three games. Arm
  ``aux`` keeps M7c's decoder (50 ms, one spike) and only adds the warm-up; it separates the two
  causes and is reported apart from the verdict table.
* **Controls** for the ignition flag: same seed, warm-up, then 5 s with no input; compared with
  M7c's ``like_for_like`` reading unchanged (start-up window dropped on both sides).

Unchanged and reused from ``m7c``: the operating point, detector and input, the games and seeds,
the k grid on **M7c's p99** (read from ``m7c-results.json``, not recomputed), ``select_best``,
the paired verdicts with the dodge_v2 ceiling, and the three comparison bots — their scores are
replayed and checked seed by seed against ``m7b-baselines.json`` (random, brainless) and
``m7c-results.json`` (one-direction bot).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from ...data.download import ROOT
from ...graph import Graph
from ...probe.export import neuron_coords
from ...probe.motor import CountWindowDecoder, GiantFibreDecoder
from ...probe.sets import build_probe_sets, load_roi
from ...sensory.looming import LoomingDetector
from . import m7c as M
from .agents import BrainAgent, RandomAgent, ThresholdAgent
from .base import Action
from .brain import DECODER_WINDOW_MS, EscapeSets, FlyBrain, TickDecoder, make_decoder
from .runner import EVAL_SEEDS, TRAIN_SEEDS, run_many

WARMUP_STEPS = 5000                 # brief §2.1: 5,000 steps (5 s at dt 1 ms) with no input
ARMS = ("main", "aux")
DECODER_JSON = ROOT / "data-provenance" / "m7c2-decoder.json"
M7C_RESULTS = M.RESULTS_JSON
RESULTS_JSON = ROOT / "data-provenance" / "m7c2-results.json"
BUNDLE_DIR = ROOT / "runs" / "m7c2"
BUNDLE_ARM = "main"
W_GRID = (50.0, 100.0, 200.0, 300.0)   # brief §1.1, to check the phase-1 choice is in the table
N_GRID = (1, 2, 3)


def log(msg: str) -> None:
    print(f"[m7c2] {msg}", flush=True)


# ---------------------------------------------------------------------------------------------
# the phase-1 choice
# ---------------------------------------------------------------------------------------------
def load_decoder_choice(path: Path = DECODER_JSON) -> dict:
    """(W*, n*) from phase 1. Refuses a missing file, a choice outside the brief's 12-cell table,
    and J <= 0 (brief §1.3: then phase 2 is not run)."""
    if not path.exists():
        raise FileNotFoundError(f"{path.relative_to(ROOT)} is missing: phase 1 has not chosen a decoder")
    raw = path.read_bytes()
    doc = json.loads(raw)
    sel = doc["selection"]
    w, n, j = float(sel["window_ms"]), int(sel["min_spikes"]), float(sel["J"])
    if w not in W_GRID or n not in N_GRID:
        raise ValueError(f"phase-1 choice (W={w}, n={n}) is outside the brief's grid")
    if not j > 0 or sel.get("beats_noise") is False:
        raise ValueError(f"phase-1 best J = {j} <= 0: the brief says phase 2 is not run")
    rel = str(path.relative_to(ROOT))
    return {"window_ms": w, "min_spikes": n, "J": j, "J_exact": sel.get("J_exact"), "file": rel,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "committed_in": _last_commit(rel),                 # the commit that holds this file
            "computed_at_head": (doc.get("git") or {}).get("head")}


def _last_commit(rel: str) -> str | None:
    try:
        out = subprocess.check_output(["git", "log", "-1", "--format=%H", "--", rel], cwd=ROOT, text=True).strip()
        dirty = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", rel], cwd=ROOT).returncode != 0
        return (out or None) if not dirty else None              # an edited file has no commit
    except Exception:
        return None


def arm_decoder(arm: str, brain: FlyBrain, choice: dict | None):
    """Set the brain's read-out window for the arm and return the decoder ``BrainAgent`` calls."""
    if arm == "main":
        if choice is None:
            raise ValueError("arm 'main' needs the phase-1 choice")
        brain.decoder_window_ms = choice["window_ms"]
        return TickDecoder(CountWindowDecoder(choice["window_ms"], choice["min_spikes"]), brain)
    if arm == "aux":
        brain.decoder_window_ms = DECODER_WINDOW_MS
        return make_decoder()                       # M7c's GiantFibreDecoder (50 ms, one spike)
    raise KeyError(f"unknown arm {arm!r}")


# ---------------------------------------------------------------------------------------------
# bots (replayed, checked against the committed scores)
# ---------------------------------------------------------------------------------------------
def bots(game: str, base: dict, m7c_row: dict) -> dict:
    theta = base["games"][game]["threshold"]["theta"]
    rnd = run_many(game, lambda env: RandomAgent(), EVAL_SEEDS)
    thr = run_many(game, lambda env: ThresholdAgent(theta), EVAL_SEEDS)
    b_rnd = dict(zip(base["games"][game]["random"]["seeds"], base["games"][game]["random"]["scores"]))
    b_thr = dict(zip(base["games"][game]["threshold"]["eval"]["seeds"],
                     base["games"][game]["threshold"]["eval"]["scores"]))
    if rnd["scores"] != [b_rnd[s] for s in EVAL_SEEDS] or thr["scores"] != [b_thr[s] for s in EVAL_SEEDS]:
        raise RuntimeError(f"{game}: baseline bots no longer reproduce m7b-baselines.json")
    ow = M.one_way_reference(run_many(game, lambda env: M._Const(Action.LEFT), EVAL_SEEDS)["scores"],
                             run_many(game, lambda env: M._Const(Action.RIGHT), EVAL_SEEDS)["scores"])
    if (ow["always_left"], ow["always_right"]) != (m7c_row["one_way"]["always_left"],
                                                   m7c_row["one_way"]["always_right"]):
        raise RuntimeError(f"{game}: one-direction bots no longer reproduce m7c-results.json")
    return {"theta": theta, "random": {"scores": rnd["scores"], "mean": rnd["mean"]},
            "brainless": {"scores": thr["scores"], "mean": thr["mean"]}, "one_way": ow}


# ---------------------------------------------------------------------------------------------
# one arm, one game
# ---------------------------------------------------------------------------------------------
def play_game(arm: str, game: str, agent: BrainAgent, brain: FlyBrain, controls: dict, m7c_row: dict,
              ref: dict, ctx: dict | None) -> dict:
    tg = time.time()
    p99 = float(m7c_row["p99"])
    grid = M.k_grid(p99)
    if not np.allclose(grid, m7c_row["k_grid"], rtol=1e-12, atol=0):
        raise RuntimeError(f"{game}: k grid differs from M7c's")
    rows = {}
    for k in grid:
        brain.k = k
        eps = [M.play_fly(game, agent, brain, s) for s in TRAIN_SEEDS]
        for e in eps:
            e["ignition"] = M.ignition(e, controls[e["seed"]])
            e.pop("windows")
        sc = np.array([e["score"] for e in eps])
        rows[k] = {"k": k, "mean": float(sc.mean()), "min": int(sc.min()), "max": int(sc.max()),
                   "scores": sc.tolist(), "ignited": int(sum(e["ignition"]["ignited"] for e in eps)),
                   "ignited_literal": int(sum(e["ignition"]["literal"]["ignited"] for e in eps)),
                   "ticks": [e["ticks"] for e in eps], "gf_spikes": [e["gf_spikes"] for e in eps],
                   "network_rate_hz": [round(e["network_rate_hz"], 4) for e in eps]}
        log(f"  [{arm}] {game} k={k:9.4g}: train mean {rows[k]['mean']:6.2f}  scores {rows[k]['scores']}  "
            f"game s {sum(rows[k]['ticks']) / 60:.0f}  ignited {rows[k]['ignited']}  ({time.time() - tg:.0f} s)")
    sel = M.select_best(rows)
    k = sel["chosen"]
    brain.k = k
    fly_eps = []
    for s in EVAL_SEEDS:
        e = M.play_fly(game, agent, brain, s, detail=True)
        e["ignition"] = M.ignition(e, controls[s])
        e.pop("windows")
        fly_eps.append(e)
    fly = [e["score"] for e in fly_eps]
    row = {"p99": p99, "k_grid": list(grid), "selection": sel | {"grid": [rows[g] for g in grid]}, "k": k,
           "fly": {"scores": fly, "mean": float(np.mean(fly)), "min": int(min(fly)), "max": int(max(fly)),
                   "episodes": fly_eps},
           "verdict": M.verdict(fly, ref["random"]["scores"], ref["brainless"]["scores"],
                                ref["one_way"]["scores"], game),
           "m7c": {"k": m7c_row["k"], "scores": m7c_row["fly"]["scores"], "mean": m7c_row["fly"]["mean"]},
           "vs_m7c": M.paired(fly, m7c_row["fly"]["scores"])}
    if ctx is not None:
        row["bundle"] = make_bundle(arm, game, k, fly[EVAL_SEEDS.index(M.BUNDLE_SEED)], agent, brain, ctx)
    v = row["verdict"]
    log(f"[{arm}] {game}: k={k:.4g} | fly {row['fly']['mean']:.2f} (M7c {m7c_row['fly']['mean']:.2f}) vs random "
        f"{ref['random']['mean']:.2f}, one-way {ref['one_way']['mean']:.2f}, brainless {ref['brainless']['mean']:.2f}"
        f" -> beat random {v['beat_random']}, beat one-way {v['beat_one_way']}; {v['brainless_verdict']} "
        f"({time.time() - tg:.0f} s)")
    return row


def make_bundle(arm: str, game: str, k: float, eval_score: int, agent: BrainAgent, brain: FlyBrain,
                ctx: dict) -> dict:
    """Replay evaluation seed 10 with spikes kept; same format as M7c, run.json after the warm-up."""
    brain.k = k
    brain.keep_spikes = True
    try:
        ep = M.play_fly(game, agent, brain, M.BUNDLE_SEED, detail=True, record=True)
    finally:
        brain.keep_spikes = False
    if ep["score"] != eval_score:
        raise RuntimeError("bundle replay differs from the evaluation run (determinism)")
    out = M.write_bundle(game, k, ep, ctx, brain, BUNDLE_DIR / f"{game}-brain-seed{M.BUNDLE_SEED}",
                         run_id=f"m7c2-{arm}-{game}-brain-seed{M.BUNDLE_SEED}",
                         decoder_desc=agent.decoder.describe(),
                         extra_meta={"m7c2_arm": arm, "warmup_steps": brain.warmup_steps,
                                     "clock": "0 = end of the warm-up; warm-up spikes not included"})
    brain.spikes_t, brain.spikes_i = [], []
    return out


# ---------------------------------------------------------------------------------------------
def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="M7c2 phase 2: warm-up + calibrated decoder rematch")
    ap.add_argument("--games", default=",".join(M.GAMES))
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-bundle", action="store_true")
    ap.add_argument("--out", default=str(RESULTS_JSON))
    args = ap.parse_args(argv)
    games = [g for g in args.games.split(",") if g]
    arms = [a for a in args.arms.split(",") if a]
    if set(arms) - set(ARMS):
        raise SystemExit(f"unknown arm(s): {sorted(set(arms) - set(ARMS))}")
    choice = load_decoder_choice() if "main" in arms else None      # refuses before any fly run
    t_all = time.time()
    base = json.loads(M.BASELINES.read_text())
    m7c = json.loads(M7C_RESULTS.read_text())
    graph = Graph.load()
    roi = load_roi(graph)
    escape = EscapeSets.from_probe_sets(build_probe_sets(graph, roi, profile="escape"))
    ctx = {"escape": escape, "coords": neuron_coords(graph, roi)}
    brain = FlyBrain(graph, escape, k=1.0, device=args.device, keep_spikes=False, warmup_steps=WARMUP_STEPS)

    log(f"controls (warm-up {WARMUP_STEPS} steps, then {M.CONTROL_MS} ms with no input, per seed) ...")
    controls = {s: M.control_run(brain, s) for s in TRAIN_SEEDS + EVAL_SEEDS}
    log("comparison bots (replayed and checked) ...")
    refs = {g: bots(g, base, m7c["games"][g]) for g in games}

    result = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "git_commit": M.git_commit(),
              "train_seeds": list(TRAIN_SEEDS), "eval_seeds": list(EVAL_SEEDS),
              "warmup_steps": WARMUP_STEPS, "brain": brain.describe() | {"k": "per arm and game"},
              "detector": LoomingDetector.describe(), "decoder_choice": choice,
              "rules": {"k_points": M.K_POINTS, "k_span": list(M.K_SPAN), "p99": "M7c's (m7c-results.json)",
                        "sem_mult": M.SEM_MULT, "control_ms": M.CONTROL_MS,
                        "control": "same seed, warm-up, then control_ms with no input",
                        "ignition_table": "like_for_like", "runaway_rate_hz": M.RUNAWAY_RATE_HZ,
                        "failure_active_fraction": M.FAILURE_ACTIVE_FRACTION,
                        "excess_min_fraction": M.EXCESS_MIN_FRACTION, "ceiling_games": M.CEILING_GAMES},
              "controls": {str(s): {k: v for k, v in c.items() if k != "windows"} for s, c in controls.items()},
              "bots": refs, "arms": {}}
    for arm in arms:
        decoder = arm_decoder(arm, brain, choice)
        agent = BrainAgent(brain, decoder, brain_reset=brain.reset)
        bundle_ctx = ctx if (arm == BUNDLE_ARM and not args.no_bundle) else None
        result["arms"][arm] = {
            "decoder": decoder.describe(), "decoder_window_ms": brain.decoder_window_ms,
            "in_verdict_table": arm == "main",
            "games": {g: play_game(arm, g, agent, brain, controls, m7c["games"][g], refs[g], bundle_ctx)
                      for g in games}}
    result["wall_s"] = round(time.time() - t_all, 1)
    out = Path(args.out)
    out.write_text(json.dumps(result, indent=1) + "\n")
    log(f"done in {result['wall_s']} s -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
