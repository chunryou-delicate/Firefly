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

Part 2 (§4, the fly), M7c2's main arm unchanged:

    python -m flysim.apps.games.m7f part2 [--games a,b] [--out runs/m7f/part2-a.json]
    python -m flysim.apps.games.m7f merge runs/m7f/part2-*.json   # -> data-provenance/m7f-results.json

* operating point P2, warm-up 5,000 steps, ``CountWindowDecoder(300 ms, 2)`` (checked against
  M7c2's committed phase-1 choice), ``k * loom_L`` on every left LC4 + LPLC2 cell and
  ``k * loom_R`` on the right. ``k`` is the only per-game setting.
* ``p99`` per game: the non-zero looming values (both halves) of the frames the brainless bot
  (its part-1 θ) played on training seeds 0-9 (M7c's rule). No non-zero value: the game is
  written down as "the detector saw nothing" and no fly is run.
* ``k`` grid ``[1/p99, 5120/p99]``, 21 log points, best training mean (ties: geometric middle),
  evaluation seeds 10-19. Verdicts: paired difference > 3 SEM against the random bot, the
  fixed-action reference and the brainless bot; "ceiling — no verdict" against the brainless bot
  when it equals the cheating bot on all ten seeds.
* controls and ignition flags as M7c2 (same seed, warm-up, 5 s with no input, ``like_for_like``);
  flagged episodes are kept and counted, never dropped.
* these games have no single obstacle, so each episode records what any game has: action counts,
  the first action, score and life-loss times, game-over (``play_classic``).
* replay bundle of evaluation seed 10 in ``runs/m7f/<game>-brain-seed10/`` — M7c2's format plus
  ``scene.json`` (the ``scene()`` of every recorded frame, checked to redraw it bit for bit).
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


# ---------------------------------------------------------------------------------------------
# part 2 (§4): the fly — M7c2's main arm, unchanged
# ---------------------------------------------------------------------------------------------
P2_WARMUP_STEPS = 5000
P2_DECODER = (300.0, 2)                          # CountWindowDecoder(W ms, n spikes per side)
P2_BUNDLE_SEED = 10
RESULTS = ROOT / "data-provenance" / "m7f-results.json"
NOT_SEEN = "the detector saw nothing (no non-zero loom on the brainless bot's training frames)"
_EVENT = {int(Action.JUMP): "jump", int(Action.LEFT): "move_left", int(Action.RIGHT): "move_right"}


def p99_for(game: str, theta: float) -> dict:
    """§4: the 99th percentile of the non-zero looming values (both halves) on the frames the
    brainless bot (θ) played on training seeds 0-9 — M7c's rule. None when there are none."""
    from . import m7c as M
    vals = M.loom_values(game, "brainless", TRAIN_SEEDS, theta=theta)
    if len(vals) == 0:
        return {"p99": None, "n_values": 0, "status": NOT_SEEN}
    return {"p99": float(np.percentile(vals, 99)), "n_values": int(len(vals))}


def play_classic(game: str, agent, brain, seed: int, record: bool = False, max_ms: int = 60_000) -> dict:
    """One fly episode on a classic game, logged by the harness (the agent sees pixels only).

    The games have no single obstacle to follow, so the record is what every game has: how often
    each action was taken, when the first non-NOOP action came, when points were scored and lives
    lost, how the game ended, and the brain's statistics (``brain.summary()``) for the ignition
    flags. ``record`` keeps frames, actions, looming, events and each frame's ``scene()`` for
    the replay bundle (first ``max_ms``)."""
    from .brain import tick_end_ms
    env = make(game)
    frame = env.reset(seed)
    agent.reset(seed)
    counts = [0] * len(ACTIONS)
    first_ms, score_ms, lost_ms, events = None, [], [], []
    frames, t_frames, actions, looms, scenes = [], [], [], [], []
    lives = env.lives()
    done = False
    info = {"score": 0, "tick": 0, "crashed": False, "truncated": False}
    while not done:
        t_start = tick_end_ms(env.tick - 1) if env.tick > 0 else 0
        keep = record and t_start < max_ms
        if keep:
            frames.append(frame)
            t_frames.append(float(t_start))
            scenes.append(env.scene())
        a = int(agent.act(frame))
        if keep:
            actions.append(a)
            looms.append(getattr(agent, "last_signal", (0.0, 0.0)))
        frame, r, done, info = env.step(a)
        t_end = float(tick_end_ms(env.tick - 1))
        counts[a] += 1
        if a in _EVENT:
            first_ms = t_end if first_ms is None else first_ms
            events.append({"t_ms": t_end, "type": _EVENT[a]})
        if r > 0:
            score_ms.append(t_end)
            events.append({"t_ms": t_end, "type": "score", "points": int(r), "score": int(info["score"])})
        now = env.lives()
        if now is not None and now < lives:
            lost_ms.append(t_end)
            events.append({"t_ms": t_end, "type": "life_lost", "lives": int(now)})
        lives = now
        if info["crashed"]:
            events.append({"t_ms": t_end, "type": "game_over"})
    summary = brain.summary()
    out = {"seed": int(seed), "score": int(info["score"]), "ticks": int(info["tick"]),
           "crashed": bool(info["crashed"]), "truncated": bool(info["truncated"]),
           "network_rate_hz": summary["network_rate_hz"], "active_fraction": summary["active_fraction"],
           "gf_spikes": summary["gf_spikes"], "n_spikes": summary["n_spikes"], "windows": summary["windows"],
           "actions": dict(zip(ACTIONS, counts)), "first_action_ms": first_ms,
           "score_ms": score_ms, "lives_lost_ms": lost_ms}
    if record:
        out["record"] = {"frames": frames, "t_ms": t_frames, "actions": actions, "loom": looms,
                         "events": events, "scenes": scenes}
    return out


def verdict2(fly: list[int], random: list[int], fixed: list[int], brainless: list[int], ceiling: bool) -> dict:
    """§4: M7c's paired rule (mean difference > 0 and > 3 SEM) against each comparison bot."""
    from . import m7c as M

    def word(v: dict) -> str:
        return "beat" if v["beats"] else ("lost" if v["loses"] else "indistinguishable")

    vr, vf, vb = M.paired(fly, random), M.paired(fly, fixed), M.paired(fly, brainless)
    return {"vs_random": vr | {"verdict": word(vr)}, "vs_fixed": vf | {"verdict": word(vf)},
            "vs_brainless": vb | {"verdict": "ceiling — no verdict" if ceiling else word(vb)},
            "ceiling": bool(ceiling)}


def check_bots(games, base: dict, workers: int) -> dict:
    """The comparison bots' evaluation scores, replayed and checked seed by seed against part 1."""
    refs, jobs = {}, []
    for g in games:
        row = base["games"][g]
        specs = [("random",), ("const", int(Action[row["fixed"]["reference"]])),
                 ("threshold", row["threshold"]["theta"]), ("oracle",)]
        jobs += [(g, spec, s) for spec in specs for s in EVAL_SEEDS]
    eps = play(jobs, workers)
    n = len(EVAL_SEEDS)
    for i, g in enumerate(games):
        row = base["games"][g]
        got = [[e["score"] for e in eps[(4 * i + j) * n:(4 * i + j + 1) * n]] for j in range(4)]
        want = [row["random"]["scores"], row["fixed"]["scores"], row["threshold"]["eval"]["scores"],
                row["oracle_reference"]["scores"]]
        if got != want:
            raise RuntimeError(f"{g}: comparison bots no longer reproduce m7f-baselines.json")
        refs[g] = {"theta": row["threshold"]["theta"],
                   "random": {"scores": want[0], "mean": float(np.mean(want[0]))},
                   "fixed": {"reference": row["fixed"]["reference"], "scores": want[1],
                             "mean": float(np.mean(want[1]))},
                   "brainless": {"scores": want[2], "mean": float(np.mean(want[2]))},
                   "oracle": {"scores": want[3], "mean": float(np.mean(want[3]))},
                   "ceiling": want[2] == want[3]}
    return refs


def _fly_row(eps: list[dict]) -> dict:
    sc = np.array([e["score"] for e in eps])
    return {"mean": float(sc.mean()), "min": int(sc.min()), "max": int(sc.max()), "scores": sc.tolist(),
            "ignited": int(sum(e["ignition"]["ignited"] for e in eps)),
            "ignited_literal": int(sum(e["ignition"]["literal"]["ignited"] for e in eps)),
            "viewer_failure": int(sum(e["ignition"]["viewer_failure"] for e in eps)),
            "ticks": [e["ticks"] for e in eps], "gf_spikes": [e["gf_spikes"] for e in eps],
            "actions": {a: int(sum(e["actions"][a] for e in eps)) for a in ACTIONS},
            "network_rate_hz": [round(e["network_rate_hz"], 4) for e in eps]}


def fly_game(game: str, theta: float, agent, brain, controls: dict, ref: dict, ctx: dict | None) -> dict:
    """§4 for one game: p99, the k grid on training seeds, the chosen k on evaluation seeds,
    verdicts, and (``ctx``) the replay bundle of evaluation seed 10."""
    from . import m7c as M
    tg = time.time()
    p = p99_for(game, theta)
    if p["p99"] is None:
        log(f"{game}: {NOT_SEEN} — no fly run")
        return {"theta": theta, **p}
    grid = M.k_grid(p["p99"])
    log(f"{game}: p99 {p['p99']:.6g} ({p['n_values']} non-zero values) -> k {grid[0]:.4g} .. {grid[-1]:.4g}")
    rows = {}
    for k in grid:
        brain.k = k
        eps = []
        for s in TRAIN_SEEDS:
            e = play_classic(game, agent, brain, s)
            e["ignition"] = M.ignition(e, controls[s])
            e.pop("windows")
            eps.append(e)
        rows[k] = {"k": k, **_fly_row(eps)}
        log(f"  {game} k={k:9.4g}: train mean {rows[k]['mean']:6.2f}  scores {rows[k]['scores']}  "
            f"game s {sum(rows[k]['ticks']) / 60:.0f}  ignited {rows[k]['ignited']}  ({time.time() - tg:.0f} s)")
    sel = M.select_best(rows)
    k = sel["chosen"]
    brain.k = k
    fly_eps = []
    for s in EVAL_SEEDS:
        e = play_classic(game, agent, brain, s)
        e["ignition"] = M.ignition(e, controls[s])
        e.pop("windows")
        fly_eps.append(e)
    fly = [e["score"] for e in fly_eps]
    row = {"theta": theta, **p, "k_grid": list(grid), "selection": sel | {"grid": [rows[g] for g in grid]},
           "k": k, "fly": _fly_row(fly_eps) | {"episodes": fly_eps},
           "verdict": verdict2(fly, ref["random"]["scores"], ref["fixed"]["scores"], ref["brainless"]["scores"],
                               ref["ceiling"])}
    if ctx is not None:
        row["bundle"] = fly_bundle(game, k, fly[EVAL_SEEDS.index(P2_BUNDLE_SEED)], agent, brain, ctx)
    v = row["verdict"]
    log(f"{game}: k={k:.4g} | fly {row['fly']['mean']:.2f} vs random {ref['random']['mean']:.2f} "
        f"({v['vs_random']['verdict']}), fixed {ref['fixed']['reference']} {ref['fixed']['mean']:.2f} "
        f"({v['vs_fixed']['verdict']}), brainless {ref['brainless']['mean']:.2f} ({v['vs_brainless']['verdict']}) "
        f"({time.time() - tg:.0f} s)")
    return row


def fly_bundle(game: str, k: float, eval_score: int, agent, brain, ctx: dict, out_root: Path = OUT_DIR) -> dict:
    """Replay evaluation seed 10 with spikes kept: M7c2's bundle (frames.npz, run.json,
    game.json) plus scene.json, whose every scene must redraw its frame bit for bit."""
    from . import m7c as M
    from .classic.common import draw
    if P2_BUNDLE_SEED != M.BUNDLE_SEED:
        raise RuntimeError("the bundle writer stamps M7c's bundle seed")
    brain.k = k
    brain.keep_spikes = True
    try:
        ep = play_classic(game, agent, brain, P2_BUNDLE_SEED, record=True, max_ms=M.BUNDLE_MAX_MS)
    finally:
        brain.keep_spikes = False
    if ep["score"] != eval_score:
        raise RuntimeError("bundle replay differs from the evaluation run (determinism)")
    rec = ep["record"]
    scenes = rec.pop("scenes")
    for f, sc in zip(rec["frames"], scenes):
        if not np.array_equal(draw(json.loads(json.dumps(sc["rects"]))), f):
            raise RuntimeError(f"{game}: a scene record does not redraw its frame")
    out_dir = out_root / f"{game}-brain-seed{P2_BUNDLE_SEED}"
    times = list(rec["t_ms"])
    out = M.write_bundle(game, k, ep, ctx, brain, out_dir, run_id=f"m7f-{game}-brain-seed{P2_BUNDLE_SEED}",
                         decoder_desc=agent.decoder.describe(),
                         extra_meta={"m7f": "part 2, M7c2 main arm", "warmup_steps": brain.warmup_steps,
                                     "clock": "0 = end of the warm-up; warm-up spikes not included"})
    (out_dir / "scene.json").write_text(json.dumps(
        {"game": game, "seed": P2_BUNDLE_SEED, "frames": len(scenes), "t_ms": times,
         "alignment": "scenes[k] drew frames[k] of frames.npz (before action k); "
                      "classic.common.draw(scenes[k]['rects']) == frames[k] bit for bit",
         "scenes": scenes}, separators=(",", ":")) + "\n")
    brain.spikes_t, brain.spikes_i = [], []
    return out | {"scene_json": str((out_dir / "scene.json").relative_to(ROOT))
                  if out_dir.resolve().is_relative_to(ROOT) else str(out_dir / "scene.json")}


def brain_setup(device: str):
    """The M7c2 main-arm agent: P2 brain with the warm-up and CountWindowDecoder(300, 2)."""
    from ...graph import Graph
    from ...probe.export import neuron_coords
    from ...probe.motor import CountWindowDecoder
    from ...probe.sets import build_probe_sets, load_roi
    from . import m7c2 as M2
    from .agents import BrainAgent
    from .brain import EscapeSets, FlyBrain, TickDecoder
    choice = M2.load_decoder_choice()
    if (choice["window_ms"], choice["min_spikes"]) != P2_DECODER or M2.WARMUP_STEPS != P2_WARMUP_STEPS:
        raise RuntimeError("the brief's decoder / warm-up differ from M7c2's committed main arm")
    graph = Graph.load()
    roi = load_roi(graph)
    escape = EscapeSets.from_probe_sets(build_probe_sets(graph, roi, profile="escape"))
    ctx = {"escape": escape, "coords": neuron_coords(graph, roi)}
    brain = FlyBrain(graph, escape, k=1.0, device=device, keep_spikes=False, warmup_steps=P2_WARMUP_STEPS,
                     decoder_window_ms=P2_DECODER[0])
    decoder = TickDecoder(CountWindowDecoder(*P2_DECODER), brain)
    agent = BrainAgent(brain, decoder, brain_reset=brain.reset)
    return agent, brain, ctx, choice


def part2(games, device: str, workers: int, out: Path, bundles: bool = True) -> dict:
    from . import m7c as M
    t0 = time.time()
    base = json.loads(BASELINES.read_text())
    for g in games:
        if base["games"][g]["rules"] != make(g).describe():
            raise RuntimeError(f"{g}: the game's rules changed since part 1")
    log("comparison bots (replayed and checked against part 1) ...")
    refs = check_bots(games, base, workers)
    agent, brain, ctx, choice = brain_setup(device)
    log(f"controls (warm-up {P2_WARMUP_STEPS} steps, then {M.CONTROL_MS} ms with no input, per seed) ...")
    controls = {s: M.control_run(brain, s) for s in TRAIN_SEEDS + EVAL_SEEDS}
    result = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "git_commit": git_commit(),
              "brief": "docs/m7f-brief.md §4", "train_seeds": list(TRAIN_SEEDS), "eval_seeds": list(EVAL_SEEDS),
              "warmup_steps": P2_WARMUP_STEPS, "brain": brain.describe() | {"k": "per game, see below"},
              "detector": LoomingDetector.describe(), "decoder": agent.decoder.describe(),
              "decoder_choice": choice,
              "rules": {"k_points": M.K_POINTS, "k_span": list(M.K_SPAN),
                        "p99": "non-zero loom (both halves) of the brainless bot's frames (part-1 θ), seeds 0-9",
                        "sem_mult": M.SEM_MULT, "opponents": ["random", "fixed-action reference", "brainless"],
                        "ceiling": "brainless == cheating bot on all ten evaluation seeds -> no verdict vs brainless",
                        "control_ms": M.CONTROL_MS, "control": "same seed, warm-up, then control_ms with no input",
                        "ignition_table": "like_for_like", "runaway_rate_hz": M.RUNAWAY_RATE_HZ,
                        "failure_active_fraction": M.FAILURE_ACTIVE_FRACTION,
                        "excess_min_fraction": M.EXCESS_MIN_FRACTION},
              "controls": {str(s): {k: v for k, v in c.items() if k != "windows"} for s, c in controls.items()},
              "bots": refs, "games": {}}
    for g in games:
        result["games"][g] = fly_game(g, refs[g]["theta"], agent, brain, controls, refs[g], ctx if bundles else None)
    result["wall_s"] = round(time.time() - t0, 1)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1) + "\n")
    log(f"done in {result['wall_s']} s -> {out}")
    return result


SHARED_KEYS = ("git_commit", "train_seeds", "eval_seeds", "warmup_steps", "brain", "detector", "decoder",
               "decoder_choice", "rules", "controls")


def merge(paths: list[Path], out: Path = RESULTS) -> dict:
    """Part-2 runs of disjoint game sets -> one results file, all eight games in the brief's order.
    The runs must share the code commit, protocol and controls exactly."""
    docs = [json.loads(Path(p).read_text()) for p in paths]
    first = docs[0]
    for d in docs[1:]:
        for k in SHARED_KEYS:
            if d[k] != first[k]:
                raise ValueError(f"part-2 runs differ in {k!r}; refusing to merge")
    games: dict = {}
    bots: dict = {}
    for d in docs:
        for g in d["games"]:
            if g in games:
                raise ValueError(f"{g} appears in more than one run")
            games[g], bots[g] = d["games"][g], d["bots"][g]
    if set(games) != set(GAMES):
        raise ValueError(f"games missing from the runs: {sorted(set(GAMES) - set(games))}")
    result = {k: first[k] for k in ("generated", "brief", *SHARED_KEYS)}
    result |= {"runs": [{"file": str(p), "generated": d["generated"], "games": list(d["games"]),
                         "wall_s": d["wall_s"]} for p, d in zip(paths, docs)],
               "bots": {g: bots[g] for g in GAMES}, "games": {g: games[g] for g in GAMES}}
    out.write_text(json.dumps(result, indent=1) + "\n")
    log(f"merged {len(docs)} run(s) -> {out}")
    return result


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="M7f classic-style games")
    ap.add_argument("command", choices=("tune", "part1", "part2", "merge"))
    ap.add_argument("paths", nargs="*", help="merge: the part-2 run files")
    ap.add_argument("--games", default=",".join(GAMES))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--note", default="")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None, help="part2: output file (default: the results file for all "
                                                 "eight games, runs/m7f/part2-<games>.json otherwise)")
    ap.add_argument("--no-bundle", action="store_true")
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
    elif args.command == "part2":
        out = Path(args.out) if args.out else (RESULTS if games == list(GAMES)
                                               else OUT_DIR / f"part2-{'-'.join(games)}.json")
        part2(games, args.device, args.workers, out, bundles=not args.no_bundle)
    elif args.command == "merge":
        merge([Path(p) for p in args.paths])
    log(f"done in {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
