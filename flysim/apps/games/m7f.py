"""M7f: the eight hand-made classic-style games on the ladder (docs/m7f-brief.md).

    python -m flysim.apps.games.m7f tune     # tuning checks: always-NOOP and the cheating bot, seeds 0-9
    python -m flysim.apps.games.m7f part1    # part 1: comparison bots, detector survey, figure

Part 1 (§3, no fly), fixed before any comparison bot was run:

* random bot, the four fixed-action bots (always NOOP / JUMP / LEFT / RIGHT; the reference is
  the best evaluation mean, ties in that order), the brainless bot (``runner.select_threshold``
  on ``THETA_GRID`` with training seeds 0-9, then evaluation seeds 10-19) and the cheating bot
  (``ClassicOracle``, game state, a ceiling for reference) — evaluation seeds 10-19.
* what the same eyes see: ``LoomingDetector`` on the frames the random bot played on training
  seeds 0-9 (share of non-zero half-frames, mean per side, 99th percentile of the non-zero
  values), recorded only, never used to choose anything.
* ``docs/m7f-survey.png``: per game one row of eight frames (1 s apart) of the brainless bot on
  evaluation seed 10, each with its loom_L / loom_R bars.

Tuning (§2) is allowed only before the first comparison bot runs and only with the two checks of
``tune``; every check is appended to ``runs/m7f/tuning-log.jsonl`` and copied into the baselines
file. Once ``data-provenance/m7f-baselines.json`` exists, ``tune`` refuses to run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from ...data.download import ROOT
from ...sensory.looming import LoomingDetector
from . import media
from .agents import Agent, RandomAgent, ThresholdAgent
from .base import ACTIONS, Action, make
from .classic import GAMES
from .classic.common import ClassicOracle
from .runner import EVAL_SEEDS, THETA_GRID, TRAIN_SEEDS, git_commit, run_episode, select_threshold, summarize

FIXED_ACTIONS = tuple(Action)                    # NOOP, JUMP, LEFT, RIGHT — tie order too
SURVEY_SEED = 10
SURVEY_TICKS = tuple(60 * s for s in range(1, 9))     # 1 s .. 8 s
OUT_DIR = ROOT / "runs" / "m7f"
TUNING_LOG = OUT_DIR / "tuning-log.jsonl"
BASELINES = ROOT / "data-provenance" / "m7f-baselines.json"
SURVEY_PNG = ROOT / "docs" / "m7f-survey.png"


def log(msg: str) -> None:
    print(f"[m7f] {msg}", flush=True)


class ConstAgent(Agent):
    """Always the same action."""

    def __init__(self, action: int):
        self.action = int(action)
        self.name = f"always-{ACTIONS[self.action]}"

    def act(self, frame: np.ndarray) -> int:
        return self.action

    def describe(self) -> dict:
        return {"agent": self.name}


def make_agent(spec: tuple, env) -> Agent:
    kind = spec[0]
    if kind == "random":
        return RandomAgent()
    if kind == "const":
        return ConstAgent(spec[1])
    if kind == "threshold":
        return ThresholdAgent(spec[1])
    if kind == "oracle":
        return ClassicOracle(env)
    raise KeyError(kind)


def _episode(job: tuple) -> dict:
    game, spec, seed = job
    env = make(game)
    return run_episode(env, make_agent(spec, env), seed)


def play(jobs: list[tuple], workers: int) -> list[dict]:
    """Episodes in the order given (results do not depend on ``workers``)."""
    if workers <= 1:
        return [_episode(j) for j in jobs]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(_episode, jobs, chunksize=1))


def rules_hash(rules: dict) -> str:
    return hashlib.sha256(json.dumps(rules, sort_keys=True).encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------------------------
# tuning checks (§2): always-NOOP and the cheating bot only
# ---------------------------------------------------------------------------------------------
def tune(games, workers: int, note: str) -> list[dict]:
    if BASELINES.exists():
        raise RuntimeError(f"{BASELINES.name} exists: comparison bots have run, constants are frozen")
    jobs = [(g, spec, s) for g in games for spec in (("const", int(Action.NOOP)), ("oracle",)) for s in TRAIN_SEEDS]
    eps = play(jobs, workers)
    n = len(TRAIN_SEEDS)
    rows = []
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for i, g in enumerate(games):
        noop = summarize(eps[2 * i * n:(2 * i + 1) * n])
        orc = summarize(eps[(2 * i + 1) * n:(2 * i + 2) * n])
        rules = make(g).describe()
        row = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "git_commit": git_commit(), "game": g,
               "seeds": list(TRAIN_SEEDS), "rules_hash": rules_hash(rules), "rules": rules, "note": note,
               "always_noop": noop, "oracle": orc}
        rows.append(row)
        with TUNING_LOG.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        log(f"{g:9s} always-NOOP mean {noop['mean']:6.1f} ({noop['min']}-{noop['max']}) | oracle mean "
            f"{orc['mean']:6.1f} ({orc['min']}-{orc['max']}) | rules {row['rules_hash']}")
    return rows


# ---------------------------------------------------------------------------------------------
# part 1 (§3): comparison bots, detector survey, figure
# ---------------------------------------------------------------------------------------------
def _survey_seed(job: tuple) -> np.ndarray:
    """(left, right) looming of every frame the random bot played (first frame included)."""
    game, seed = job
    env, agent, det = make(game), RandomAgent(), LoomingDetector()
    frame = env.reset(seed)
    agent.reset(seed)
    out, done = [], False
    while not done:
        out.append(det.update(frame))
        frame, _, done, _ = env.step(agent.act(frame))
    return np.asarray(out, dtype=np.float64)


def survey_stats(looms: list[np.ndarray]) -> dict:
    """§3.1: share of non-zero half-frames, mean per side, 99th percentile of the non-zero values
    (both halves pooled). ``p99`` is None when the detector never fired."""
    a = np.concatenate(looms)
    nz = a[a > 0]
    return {"frames": int(len(a)), "nonzero_fraction": float((a > 0).mean()),
            "nonzero_fraction_left": float((a[:, 0] > 0).mean()),
            "nonzero_fraction_right": float((a[:, 1] > 0).mean()),
            "mean_left": float(a[:, 0].mean()), "mean_right": float(a[:, 1].mean()),
            "p99_nonzero": float(np.percentile(nz, 99)) if len(nz) else None,
            "max": float(a.max())}


def fixed_reference(rows: dict[str, dict]) -> dict:
    """The fixed-action bot with the best evaluation mean; ties in the order NOOP, JUMP, LEFT, RIGHT."""
    best = max(rows[a.name]["mean"] for a in FIXED_ACTIONS)
    ref = next(a.name for a in FIXED_ACTIONS if rows[a.name]["mean"] == best)
    return {"bots": rows, "reference": ref, "scores": rows[ref]["scores"], "mean": rows[ref]["mean"]}


def check_tuned(games) -> list[dict]:
    """Every game must be played with exactly the constants of its last tuning check."""
    if not TUNING_LOG.exists():
        raise RuntimeError(f"no tuning check recorded ({TUNING_LOG.relative_to(ROOT)}); run `tune` first")
    entries = [json.loads(line) for line in TUNING_LOG.read_text().splitlines() if line.strip()]
    for g in games:
        last = [e for e in entries if e["game"] == g]
        if not last:
            raise RuntimeError(f"{g}: no tuning check recorded")
        if last[-1]["rules_hash"] != rules_hash(make(g).describe()):
            raise RuntimeError(f"{g}: constants differ from the last tuning check")
    return entries


def survey_episode(game: str, theta: float, seed: int = SURVEY_SEED) -> dict:
    env = make(game)
    ep = run_episode(env, ThresholdAgent(theta), seed, record=True)
    rec = ep.pop("record")
    cells = [(rec["frames"][k], tuple(rec["loom"][k])) if k < len(rec["frames"]) else None for k in SURVEY_TICKS]
    return {**ep, "cells": cells}


def survey_figure(rows: list[list], path: Path = SURVEY_PNG, scale: int = 2) -> dict:
    """One row per game: eight frames and, under each, the loom_L / loom_R bars of that frame.
    Bars are scaled per row to that row's largest value (written into the result); a cell after
    the game ended is left dark."""
    fh = 84 * scale
    bar_h, gap = 24, 4
    cell_w, cell_h = fh + gap, fh + bar_h + 3 * gap
    label_w = 16
    sheet = np.full((len(rows) * cell_h + gap, label_w + 8 * cell_w + gap), 0, dtype=np.uint8)
    full_scales = []
    for r, cells in enumerate(rows):
        y = gap + r * cell_h
        media.draw_number(sheet, r + 1, 3, y + fh // 2 - 7, scale=3, ink=230)
        vals = [v for c in cells if c is not None for v in c[1]]
        full = max(vals) if vals and max(vals) > 0 else None
        full_scales.append(full)
        for i, c in enumerate(cells):
            x = label_w + gap + i * cell_w
            if c is None:
                sheet[y:y + fh, x:x + fh] = 30
                continue
            sheet[y:y + fh, x:x + fh] = media.upscale(c[0], scale)
            for side, v in enumerate(c[1]):
                bx = x + fh // 2 - 26 + side * 30
                by = y + fh + gap
                sheet[by:by + bar_h, bx:bx + 22] = 60
                filled = int(round(bar_h * v / full)) if full else 0
                if v > 0:
                    filled = max(1, filled)
                if filled:
                    sheet[by + bar_h - filled:by + bar_h, bx:bx + 22] = 235
    media.write_png(path, sheet)
    return {"path": str(path.relative_to(ROOT)), "row_full_scale": full_scales}


def part1(games, workers: int) -> dict:
    tuning = check_tuned(games)
    t0 = time.time()
    result = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "git_commit": git_commit(),
              "brief": "docs/m7f-brief.md", "train_seeds": list(TRAIN_SEEDS), "eval_seeds": list(EVAL_SEEDS),
              "theta_grid": list(THETA_GRID), "detector": LoomingDetector.describe(),
              "agents": {"random": RandomAgent().describe(),
                         "fixed": [ConstAgent(a).describe() for a in FIXED_ACTIONS],
                         "fixed_reference": "best evaluation mean; ties in the order NOOP, JUMP, LEFT, RIGHT",
                         "threshold": ThresholdAgent(1e-3).describe() | {"theta": "per game, see below"},
                         "oracle": ClassicOracle(None).describe()},
              "games": {}}
    specs = [("random",)] + [("const", int(a)) for a in FIXED_ACTIONS] + [("oracle",)]
    jobs = [(g, spec, s) for g in games for spec in specs for s in EVAL_SEEDS]
    log(f"evaluation: {len(jobs)} episodes (random, four fixed-action bots, cheating bot)")
    eps = play(jobs, workers)
    log(f"detector survey: random bot, {len(games) * len(TRAIN_SEEDS)} episodes")
    looms = play_survey([(g, s) for g in games for s in TRAIN_SEEDS], workers)
    n, m = len(EVAL_SEEDS), len(specs)
    figure_rows = []
    for gi, g in enumerate(games):
        tg = time.time()
        block = eps[gi * m * n:(gi + 1) * m * n]
        part = {spec: summarize(block[j * n:(j + 1) * n]) for j, spec in enumerate(specs)}
        sel = select_threshold(g, workers=workers)
        theta = sel["chosen_theta"]
        thr = summarize(play([(g, ("threshold", theta), s) for s in EVAL_SEEDS], workers))
        orc = part[("oracle",)]
        rules = make(g).describe()
        surv = survey_episode(g, theta)
        figure_rows.append(surv.pop("cells"))
        result["games"][g] = {
            "rules": rules, "rules_hash": rules_hash(rules),
            "detector_survey": survey_stats(looms[gi * len(TRAIN_SEEDS):(gi + 1) * len(TRAIN_SEEDS)]),
            "random": part[("random",)],
            "fixed": fixed_reference({a.name: part[("const", int(a))] for a in FIXED_ACTIONS}),
            "threshold": {"theta": theta, "selection": sel,
                          "train": next(r for r in sel["grid"] if r["theta"] == theta), "eval": thr},
            "oracle_reference": orc,
            "brainless_equals_oracle": thr["scores"] == orc["scores"],
            "survey_episode": surv}
        row = result["games"][g]
        log(f"{g}: random {row['random']['mean']:.1f}, fixed {row['fixed']['reference']} "
            f"{row['fixed']['mean']:.1f}, brainless θ={theta:.3g} {thr['mean']:.1f}, oracle {orc['mean']:.1f} "
            f"({time.time() - tg:.0f} s)")
    result["figure"] = survey_figure(figure_rows) | {"games": list(games), "ticks": list(SURVEY_TICKS),
                                                     "seed": SURVEY_SEED, "agent": "threshold (chosen θ)"}
    result["tuning_log"] = tuning
    result["wall_s"] = round(time.time() - t0, 1)
    BASELINES.write_text(json.dumps(result, indent=1) + "\n")
    log(f"wrote {BASELINES.relative_to(ROOT)} and {SURVEY_PNG.relative_to(ROOT)} in {result['wall_s']} s")
    return result


def play_survey(jobs: list[tuple], workers: int) -> list[np.ndarray]:
    if workers <= 1:
        return [_survey_seed(j) for j in jobs]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(_survey_seed, jobs, chunksize=1))


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="M7f classic-style games")
    ap.add_argument("command", choices=("tune", "part1"))
    ap.add_argument("--games", default=",".join(GAMES))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--note", default="")
    args = ap.parse_args(argv)
    games = [g for g in args.games.split(",") if g]
    for g in games:
        if g not in GAMES:
            raise KeyError(f"{g!r} is not one of the brief's eight games {GAMES}")
    t0 = time.time()
    if args.command == "tune":
        tune(games, args.workers, args.note)
    elif args.command == "part1":
        if games != list(GAMES):
            raise ValueError("part 1 runs all eight games together")
        part1(games, args.workers)
    log(f"done in {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
