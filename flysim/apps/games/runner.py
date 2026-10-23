"""M7b: run agents on the games and produce the baseline table.

    python -m flysim.apps.games.runner            # full table + media -> runs/m7b/

What it does (all fixed before the first run):

* **Random bot** on evaluation seeds 10-19.
* **Brainless bot** (``ThresholdAgent``): every θ of ``THETA_GRID`` is played on training seeds
  0-9; the θ with the highest mean score is selected (ties: the grid point closest to the
  geometric middle of the tied range); that one θ is then played on evaluation seeds 10-19.
  Both the training table and the evaluation scores are written out.
* **State oracle** (cheats, reads game state) on seeds 10-19, as a ceiling for the rules only.
* One recorded episode per game of the selected brainless bot (seed 10): an animated PNG, a
  contact sheet and the raw frames (npz).

Results go to ``runs/m7b/baselines.json`` (runs/ is gitignored; docs/m7b-report.md carries the
table). Episodes cap at the game's ``max_ticks`` (5 min); a capped episode is marked truncated.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from ...data.download import ROOT
from ...sensory.looming import LoomingDetector
from . import media
from .agents import Agent, RandomAgent, StateOracleAgent, ThresholdAgent
from .base import GameEnv, make

GAMES = ("jump", "dodge")
TRAIN_SEEDS = tuple(range(0, 10))
EVAL_SEEDS = tuple(range(10, 20))
THETA_GRID = tuple(float(x) for x in np.geomspace(1e-5, 1e-1, 41))   # 10 points per decade
OUT_DIR = ROOT / "runs" / "m7b"
RECORD_SEED = 10
RECORD_MAX_TICKS = 1800          # 30 s of the recorded episode go into the animation
RECORD_EVERY = 2                 # 60 ticks/s -> 30 fps animation


def run_episode(env: GameEnv, agent: Agent, seed: int, record: bool = False) -> dict:
    frame = env.reset(seed)
    agent.reset(seed)
    rec: dict[str, list] = {"frames": [], "actions": [], "scores": [], "loom": []}
    info = {"score": 0, "tick": 0, "crashed": False, "truncated": False}
    done = False
    while not done:
        action = agent.act(frame)
        if record:
            rec["frames"].append(frame)
            rec["actions"].append(action)
            rec["scores"].append(env.score)
            rec["loom"].append(getattr(agent, "last_signal", (0.0, 0.0)))
        frame, _, done, info = env.step(action)
    out = {"seed": int(seed), "score": int(info["score"]), "ticks": int(info["tick"]),
           "crashed": bool(info["crashed"]), "truncated": bool(info["truncated"])}
    if record:
        out["record"] = rec
    return out


def summarize(episodes: list[dict]) -> dict:
    scores = np.array([e["score"] for e in episodes], dtype=float)
    return {"seeds": [e["seed"] for e in episodes], "scores": [e["score"] for e in episodes],
            "ticks": [e["ticks"] for e in episodes],
            "truncated": int(sum(e["truncated"] for e in episodes)),
            "mean": float(scores.mean()), "median": float(np.median(scores)),
            "min": int(scores.min()), "max": int(scores.max())}


def run_many(game: str, agent_factory, seeds) -> dict:
    eps = []
    for s in seeds:
        env = make(game)
        agent = agent_factory(env)
        eps.append(run_episode(env, agent, s))
    return summarize(eps)


def _threshold_row(args) -> tuple[float, dict]:
    game, theta, seeds = args
    return theta, run_many(game, lambda env: ThresholdAgent(theta), seeds)


def select_threshold(game: str, grid=THETA_GRID, seeds=TRAIN_SEEDS, workers: int = 8) -> dict:
    """Play every θ on the training seeds; pick the best mean (ties -> geometric middle)."""
    jobs = [(game, t, tuple(seeds)) for t in grid]
    if workers <= 1:
        rows = dict(map(_threshold_row, jobs))
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            rows = dict(ex.map(_threshold_row, jobs))
    means = np.array([rows[t]["mean"] for t in grid])
    best = means.max()
    tied = [t for t, m in zip(grid, means) if m == best]
    mid = float(np.exp((np.log(tied[0]) + np.log(tied[-1])) / 2))
    chosen = min(tied, key=lambda t: abs(np.log(t) - np.log(mid)))
    return {"chosen_theta": chosen, "train_mean": float(best), "tied": tied,
            "grid": [{"theta": t, **{k: rows[t][k] for k in ("mean", "min", "max", "scores")}}
                     for t in grid]}


def export_episode(game: str, theta: float, out_dir: Path, seed: int = RECORD_SEED) -> dict:
    env = make(game)
    ep = run_episode(env, ThresholdAgent(theta), seed, record=True)
    rec = ep.pop("record")
    frames = rec["frames"][:RECORD_MAX_TICKS]
    anim = [media.annotate(f, s, l, a) for f, s, l, a in
            zip(frames, rec["scores"], rec["loom"], rec["actions"])][::RECORD_EVERY]
    base = out_dir / f"{game}-threshold-seed{seed}"
    media.write_apng(base.with_suffix(".apng.png"), anim, fps=60 // RECORD_EVERY)
    sheet = media.contact_sheet([media.upscale(f, 2) for f in frames[::30][:48]], cols=8)
    media.write_png(out_dir / f"{game}-threshold-seed{seed}-sheet.png", sheet)
    np.savez_compressed(base.with_suffix(".npz"), frames=np.stack(rec["frames"]),
                        actions=np.array(rec["actions"], dtype=np.int8),
                        loom=np.array(rec["loom"], dtype=np.float32))
    return {**ep, "animation": str(base.with_suffix(".apng.png").relative_to(ROOT)),
            "sheet": str((out_dir / f"{game}-threshold-seed{seed}-sheet.png").relative_to(ROOT)),
            "frames_npz": str(base.with_suffix(".npz").relative_to(ROOT)),
            "animated_frames": len(anim)}


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return None


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="M7b baseline table")
    ap.add_argument("--games", default=",".join(GAMES))
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-media", action="store_true")
    args = ap.parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    games = [g for g in args.games.split(",") if g]
    n_eps = len(games) * (len(THETA_GRID) * len(TRAIN_SEEDS) + 3 * len(EVAL_SEEDS))
    print(f"[m7b] {n_eps} episodes (<= {make(games[0]).max_ticks} ticks each); "
          f"expected 1-4 min with {args.workers} workers", flush=True)
    t0 = time.time()
    result = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "git_commit": git_commit(),
              "train_seeds": list(TRAIN_SEEDS), "eval_seeds": list(EVAL_SEEDS),
              "theta_grid": list(THETA_GRID), "detector": LoomingDetector.describe(),
              "agents": {"random": RandomAgent().describe(),
                         "threshold": ThresholdAgent(1e-3).describe() | {"theta": "per game, see below"}},
              "games": {}}
    for game in games:
        tg = time.time()
        sel = select_threshold(game, workers=args.workers)
        theta = sel["chosen_theta"]
        row = {"rules": make(game).describe(),
               "random": run_many(game, lambda env: RandomAgent(), EVAL_SEEDS),
               "threshold": {"theta": theta, "selection": sel,
                             "train": next(r for r in sel["grid"] if r["theta"] == theta),
                             "eval": run_many(game, lambda env: ThresholdAgent(theta), EVAL_SEEDS)},
               "state_oracle_reference": run_many(game, lambda env: StateOracleAgent(env), EVAL_SEEDS)}
        if not args.no_media:
            row["recorded_episode"] = export_episode(game, theta, out_dir)
        result["games"][game] = row
        print(f"[m7b] {game}: θ={theta:.3g} (train mean {sel['train_mean']:.2f}) | eval means: "
              f"random {row['random']['mean']:.2f}, threshold {row['threshold']['eval']['mean']:.2f}, "
              f"oracle {row['state_oracle_reference']['mean']:.2f} ({time.time() - tg:.0f} s)", flush=True)
    result["wall_s"] = round(time.time() - t0, 1)
    path = out_dir / "baselines.json"
    path.write_text(json.dumps(result, indent=1) + "\n")
    print(f"[m7b] done in {result['wall_s']} s -> {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
