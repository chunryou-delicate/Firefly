"""M7g: slow approach — the "late signal" hypothesis tested without changing the eye (docs/m7g-brief.md).

    python -m flysim.apps.games.m7g                                   # every stage, in order
    python -m flysim.apps.games.m7g --stage bots                      # CPU: θ, p99, comparison bots
    python -m flysim.apps.games.m7g --stage controls                  # GPU: the 20 no-input controls
    python -m flysim.apps.games.m7g --stage fly --variants jump,dodge_v2_s2
    python -m flysim.apps.games.m7g --stage assemble                  # results json, tables, figure

Fixed before any slow game was played (brief §2-§5; nothing here was chosen after a fly run):

* **Rows.** The five slow variants of ``slow.py`` (dodge v2 s 2·4·8, jump s 2·4, 2-min cap) and the
  base speed ``s = 1`` (the parent games, 5-min cap as before). ``s = 1`` re-plays the M7c2 main
  arm's whole grid (210 training + 10 evaluation episodes per game) with this file's recorder and
  **stops** unless every seed's score — and ticks, giant-fibre counts, ignition flags and the
  evaluation obstacles — equal ``m7c2-results.json``.
* **Fly** = M7c2's main arm exactly: P2, 5,000-step warm-up, ``CountWindowDecoder(300 ms, 2)``
  (read from the phase-1 file through ``m7c2.load_decoder_choice``), input ``k * loom`` on every
  LC4 + LPLC2 cell of that side. ``k`` grid: 21 log points on [1/p99, 5120/p99], best training
  mean (ties: geometric middle, ``m7c.select_best``), evaluation seeds 10-19.
* **p99** per variant: M7c's rule — the non-zero looming values (both halves) the brainless bot
  (with the θ chosen on that variant) sees on training seeds 0-9. At ``s = 1`` it is M7c's value
  (recomputed here and required to be equal).
* **Bots** per variant: random (eval seeds), the four fixed-action bots (reference = highest eval
  mean, ties in the order NOOP, JUMP, LEFT, RIGHT), the brainless bot (θ chosen on training seeds
  from ``THETA_GRID``), the state oracle (ceiling reference; ``SlowStateOracle`` shows it the game
  under its parent's name, rule untouched). At ``s = 1`` the M7b / M7c2 scores are used and the
  replays are checked against them; NOOP and JUMP, which M7c2 did not play, are added.
* **Verdict** (M7c): paired by seed, mean difference > 0 and > 3 SEM. **Ceiling**: when the
  brainless bot equals the oracle on all ten evaluation seeds, "beat the brainless bot" cannot be
  judged (dodge v2's 72 treatment); at ``s = 1`` dodge v2 keeps M7c's pre-registered ceiling 72.
* **Obstacle records** (brief §5.1) for every episode — all ``s``, all ``k``, training and
  evaluation. The hypothesis reading (§5.2-5.3) is taken at the grid's top ``k`` = 5120/p99.

Readings fixed here before any slow run (the brief leaves them open; recorded in the results):

* The §5.2 table uses the **training seeds 0-9 at the top k**: that is the only set of episodes
  every row has at that k (evaluation seeds are played at the selected k only).
* "First action" = the first tick, while the obstacle is on screen, whose decoder output is not
  NOOP; its time is the end of that tick (= the decision time, M7c's stamp). "Correct": dodge —
  the move away from the side the pillar appeared on; jump — JUMP. ``lead_ms`` = contact tick end
  − first action time. An obstacle still on screen when the cap ends the episode is "open" and
  carries no lead.
* **P1 holds** only if the median lead of correct first actions rises strictly with ``s`` in
  **both** families (dodge v2 1 → 2 → 4 → 8, jump 1 → 2 → 4); a row without a correct first
  action has no median and breaks the rise. **P2 holds** if the fly beat the random bot in at
  least one slow variant (selected k, evaluation seeds). The reading sentence is the 5.3 row they
  select, nothing else.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from ...data.download import ROOT
from ...sensory.looming import LoomingDetector
from . import m7c as M
from . import m7c2 as M2
from . import media
from . import slow
from .agents import Agent, BrainAgent, RandomAgent, StateOracleAgent, ThresholdAgent
from .base import Action
from .brain import tick_end_ms
from .runner import EVAL_SEEDS, THETA_GRID, TRAIN_SEEDS, run_episode, summarize

BRIEF = "docs/m7g-brief.md"
BASE_ROWS = ("jump", "dodge_v2")                              # s = 1: the parent games, unchanged
SLOW_ROWS = tuple(slow.VARIANTS)                              # brief §2, the whole list
FAMILIES = {"dodge_v2": ("dodge_v2", "dodge_v2_s2", "dodge_v2_s4", "dodge_v2_s8"),
            "jump": ("jump", "jump_s2", "jump_s4")}
ROWS = FAMILIES["dodge_v2"] + FAMILIES["jump"]
FIXED_ORDER = ("NOOP", "JUMP", "LEFT", "RIGHT")               # tie order of the fixed-action bots
BASE_CEILING = M.CEILING_GAMES                                # s = 1: M7c's pre-registered ceiling
STEP_ASIDE_TICKS = 12                                         # dodge: 0.30 / PLAYER_SPEED 0.025
STEP_ASIDE_MS = 200                                           # ~12 ticks; the line in the figure
WORKERS = 8
PARTS_DIR = ROOT / "runs" / "m7g" / "parts"
BUNDLE_DIR = ROOT / "runs" / "m7g"
RESULTS_JSON = ROOT / "data-provenance" / "m7g-results.json"
TABLES_MD = ROOT / "runs" / "m7g" / "tables.md"
FIGURE = ROOT / "docs" / "m7g-lead.png"
M7C2_RESULTS = M2.RESULTS_JSON
# the code whose behaviour the parts depend on; a part made by other code is refused at assembly
STRICT_SOURCES = ("flysim/apps/games/m7g.py", "flysim/apps/games/slow.py", "flysim/apps/games/brain.py",
                  "flysim/apps/games/m7c.py", "flysim/apps/games/m7c2.py", "flysim/apps/games/jump.py",
                  "flysim/apps/games/dodge.py", "flysim/apps/games/agents.py", "flysim/apps/games/runner.py",
                  "flysim/probe/motor.py", "flysim/sensory/looming.py", "flysim/engine/lif.py",
                  "flysim/engine/params.py")
# another window adds names to base.make's table during this run (brief §2): recorded, not enforced
RECORDED_SOURCES = ("flysim/apps/games/base.py",)


def log(msg: str) -> None:
    print(f"[m7g] {msg}", flush=True)


# ---------------------------------------------------------------------------------------------
# rows
# ---------------------------------------------------------------------------------------------
def factory(row: str):
    """The game of a row: a slow variant, or (s = 1) its parent class itself — never base.make."""
    if row in slow.VARIANTS:
        return slow.make_slow(row)
    if row in BASE_ROWS:
        return slow.PARENTS[row]()
    raise KeyError(f"unknown M7g row {row!r}; known: {list(ROWS)}")


def family(row: str) -> str:
    for f, rows in FAMILIES.items():
        if row in rows:
            return f
    raise KeyError(row)


def speed_divisor(row: str) -> int:
    return 1 if row in BASE_ROWS else slow.VARIANTS[row].speed_divisor


def kind(row: str) -> str:
    return "jump" if family(row) == "jump" else "dodge"


# ---------------------------------------------------------------------------------------------
# bots
# ---------------------------------------------------------------------------------------------
class FixedAgent(Agent):
    """The same action every tick."""

    def __init__(self, action: str):
        self.action = Action[action]
        self.name = f"always-{action}"

    def act(self, frame: np.ndarray) -> int:
        return int(self.action)

    def describe(self) -> dict:
        return {"agent": self.name, "policy": f"{self.action.name} every tick"}


class _ParentView:
    """The env as ``StateOracleAgent`` expects it: every attribute forwarded, ``name`` the parent's."""

    def __init__(self, env):
        object.__setattr__(self, "_env", env)

    @property
    def name(self) -> str:
        return self._env.parent

    def __getattr__(self, item):
        return getattr(self._env, item)


class SlowStateOracle(StateOracleAgent):
    """``StateOracleAgent``'s rule unchanged (its own code runs), on a slow variant. It reads the
    current speed from the game, so its timing follows the slow speed. Ceiling reference only."""

    def __init__(self, env):
        super().__init__(_ParentView(env))

    def describe(self) -> dict:
        return super().describe() | {"wrapper": "slow variant shown under its parent's name (brief §3)"}


def oracle_for(row: str, env) -> Agent:
    return StateOracleAgent(env) if row in BASE_ROWS else SlowStateOracle(env)


def run_many_row(row: str, agent_factory, seeds) -> dict:
    eps = []
    for s in seeds:
        env = factory(row)
        eps.append(run_episode(env, agent_factory(env), s))
    return summarize(eps)


def _theta_row(args) -> tuple[float, dict]:
    row, theta, seeds = args
    return theta, run_many_row(row, lambda env: ThresholdAgent(theta), seeds)


def select_theta(row: str, grid=THETA_GRID, seeds=TRAIN_SEEDS, workers: int = WORKERS) -> dict:
    """``runner.select_threshold`` on a row: every θ on the training seeds, best mean, ties to the
    geometric middle (the same rule as ``m7c.select_best``)."""
    jobs = [(row, float(t), tuple(seeds)) for t in grid]
    if workers <= 1:
        rows = dict(map(_theta_row, jobs))
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            rows = dict(ex.map(_theta_row, jobs))
    sel = M.select_best(rows)
    return {"chosen_theta": sel["chosen"], "train_mean": sel["best_mean"], "tied": sel["tied"],
            "grid": [{"theta": float(t), **{k: rows[t][k] for k in ("mean", "min", "max", "scores", "ticks")}}
                     for t in grid]}


def loom_values(row: str, theta: float, seeds=TRAIN_SEEDS) -> np.ndarray:
    """``m7c.loom_values(policy="brainless")`` on a row: the non-zero looming values (both halves)
    over every frame the brainless bot plays on ``seeds``."""
    vals = []
    for seed in seeds:
        env, det = factory(row), LoomingDetector()
        frame = env.reset(seed)
        bot = ThresholdAgent(theta)
        bot.reset(seed)
        done = False
        while not done:
            left, right = det.update(frame)
            if left > 0:
                vals.append(left)
            if right > 0:
                vals.append(right)
            frame, _, done, _ = env.step(bot.act(frame))
    return np.asarray(vals, dtype=np.float64)


def fixed_reference(scores: dict[str, list[int]]) -> dict:
    """The fixed-action reference: the highest evaluation mean; ties in ``FIXED_ORDER``."""
    means = {a: float(np.mean(scores[a])) for a in FIXED_ORDER}
    best = max(means.values())
    ref = next(a for a in FIXED_ORDER if means[a] == best)
    return {"scores_by_action": {a: list(scores[a]) for a in FIXED_ORDER}, "means": means,
            "reference": ref, "scores": list(scores[ref]), "mean": means[ref]}


def bots_row(row: str, base: dict, m7c: dict, m7c2: dict, workers: int = WORKERS) -> dict:
    """θ, p99 and every comparison bot of one row (no brain)."""
    t0 = time.time()
    fixed = {a: run_many_row(row, lambda env, a=a: FixedAgent(a), EVAL_SEEDS)["scores"] for a in FIXED_ORDER}
    rnd = run_many_row(row, lambda env: RandomAgent(), EVAL_SEEDS)
    if row in BASE_ROWS:
        g = base["games"][row]
        theta = g["threshold"]["theta"]
        selection = {"source": "m7b-baselines.json (θ chosen in M7b, not re-selected)", "chosen_theta": theta,
                     "train_mean": g["threshold"]["selection"]["train_mean"]}
        p99_committed = float(m7c["games"][row]["p99"])
    else:
        selection = select_theta(row, workers=workers)
        theta = selection["chosen_theta"]
        p99_committed = None
    thr = run_many_row(row, lambda env: ThresholdAgent(theta), EVAL_SEEDS)
    orc = run_many_row(row, lambda env: oracle_for(row, env), EVAL_SEEDS)
    vals = loom_values(row, theta)
    if not len(vals):
        raise LookupError(f"{row}: the brainless bot saw no non-zero looming value; p99 undefined")
    p99 = float(np.percentile(vals, 99))
    out = {"row": row, "family": family(row), "s": speed_divisor(row), "rules": factory(row).describe(),
           "theta": theta, "theta_selection": selection,
           "p99": p99, "p99_n_values": int(len(vals)),
           "loom_nonzero": {"n": int(len(vals)), "distinct": int(len(np.unique(vals))),
                            "min": float(vals.min()), "median": float(np.median(vals)), "max": float(vals.max())},
           "random": {"scores": rnd["scores"], "mean": rnd["mean"], "ticks": rnd["ticks"]},
           "fixed": fixed_reference(fixed),
           "brainless": {"scores": thr["scores"], "mean": thr["mean"], "ticks": thr["ticks"]},
           "oracle": {"scores": orc["scores"], "mean": orc["mean"], "ticks": orc["ticks"]}}
    if row in BASE_ROWS:                            # s = 1: the committed bots must come back
        g = base["games"][row]
        want = {"random": g["random"]["scores"], "brainless": g["threshold"]["eval"]["scores"],
                "oracle": g["state_oracle_reference"]["scores"],
                "LEFT": m7c2["bots"][row]["one_way"]["always_left"],
                "RIGHT": m7c2["bots"][row]["one_way"]["always_right"]}
        got = {"random": rnd["scores"], "brainless": thr["scores"], "oracle": orc["scores"],
               "LEFT": fixed["LEFT"], "RIGHT": fixed["RIGHT"]}
        bad = [k for k in want if list(want[k]) != list(got[k])]
        if bad or p99 != p99_committed:
            raise RuntimeError(f"{row}: s = 1 bots / p99 do not reproduce M7b-M7c2 ({bad}, p99 {p99} vs {p99_committed})")
        out["reproduced"] = {"bots": sorted(want), "p99": True, "added": ["NOOP", "JUMP"],
                             "sources": "m7b-baselines.json (random, brainless, oracle), m7c2-results.json "
                                        "(always LEFT / RIGHT), m7c-results.json (p99)"}
    out["ceiling"] = ceiling_of(row, thr["scores"], orc["scores"])
    out["wall_s"] = round(time.time() - t0, 1)
    return out


def ceiling_of(row: str, brainless: list[int], oracle: list[int]) -> dict:
    if row in BASE_ROWS:
        c = BASE_CEILING.get(row)
        return {"ceiling": c is not None, "rule": "M7c pre-registered ceiling (s = 1)", "value": c}
    same = list(brainless) == list(oracle)
    return {"ceiling": same, "rule": "brainless == state oracle on all 10 evaluation seeds (brief §3)",
            "value": None}


# ---------------------------------------------------------------------------------------------
# the fly: one episode with the §5.1 obstacle record (the agent itself sees pixels only)
# ---------------------------------------------------------------------------------------------
def first_correct(k: str, action: Action, side: str | None) -> bool:
    if k == "jump":
        return action == Action.JUMP
    return (action == Action.LEFT and side == "right") or (action == Action.RIGHT and side == "left")


def _close(cur: dict, t_end: float, outcome: str, gf_events) -> None:
    cur["t_contact"] = t_end
    cur["outcome"] = outcome
    lo = cur["t_spawn"]
    cur["gf_L"] = sum(1 for t, s in gf_events if lo <= t < t_end and s == "L")
    cur["gf_R"] = sum(1 for t, s in gf_events if lo <= t < t_end and s == "R")
    cur["gf_spikes"] = cur["gf_L"] + cur["gf_R"]
    cur["lead_ms"] = (t_end - cur["first_action_ms"]) if cur["first_action_ms"] is not None else None
    eff = cur["first_effective_ms"]
    cur["lead_effective_ms"] = (t_end - eff) if eff else None          # M7c's lead (effective action)


def play(row: str, agent: BrainAgent, brain, seed: int, record: bool = False) -> dict:
    """``m7c.play_fly`` on any row, with the obstacle record of brief §5.1.

    Fields shared with M7c (``first_effective_ms`` = M7c's ``first_action_ms``, ``away_ticks`` /
    ``toward_ticks`` = ``moves_away`` / ``moves_toward``, ``jumps``, ``gf_spikes``,
    ``lead_effective_ms`` = M7c's ``lead_ms``, ``cause``) are computed exactly as M7c does, so the
    s = 1 re-run can be compared with M7c2's evaluation records field by field."""
    env = factory(row)
    k = kind(row)
    parent = family(row)
    frame = env.reset(seed)
    agent.reset(seed)                                # resets the brain with the game seed (+ warm-up)
    events: list[dict] = []
    obstacles: list[dict] = []
    cur = None
    frames, t_frames, actions, looms = [], [], [], []
    n_act = {a.name: 0 for a in Action}
    done = False
    info = {"score": 0, "tick": 0, "crashed": False, "truncated": False}
    while not done:
        t_start = tick_end_ms(env.tick - 1) if env.tick > 0 else 0
        if record and t_start < M.BUNDLE_MAX_MS:
            frames.append(frame)
            t_frames.append(float(t_start))
        a = agent.act(frame)
        if record and t_start < M.BUNDLE_MAX_MS:
            actions.append(a)
            looms.append(agent.last_signal)
        if env.dist is not None and cur is None:
            cur = {"index": len(obstacles), "t_spawn": float(t_start),
                   "side": None if k == "jump" else ("right" if env.obj_x > env.x else "left"),
                   "first_action": None, "first_action_ms": None, "first_correct": None, "n_actions": 0,
                   "first_effective_ms": None, "away_ticks": 0, "toward_ticks": 0, "jumps": 0}
        jumped_before = getattr(env, "airborne", False)
        frame, r, done, info = env.step(a)
        t_end = float(tick_end_ms(env.tick - 1))
        act = Action(a)
        n_act[act.name] += 1
        if cur is not None and act != Action.NOOP:
            cur["n_actions"] += 1
            if cur["first_action"] is None:
                cur["first_action"] = act.name
                cur["first_action_ms"] = t_end
                cur["first_correct"] = first_correct(k, act, cur["side"])
        if k == "jump":
            if act == Action.JUMP and env.airborne and env.jump_t == 0 and not jumped_before:
                events.append({"t_ms": t_end, "type": "jump"})
                if cur is not None:
                    cur["jumps"] += 1
                    cur["first_effective_ms"] = cur["first_effective_ms"] or t_end
        elif act in (Action.LEFT, Action.RIGHT):
            events.append({"t_ms": t_end, "type": "move_left" if act == Action.LEFT else "move_right"})
            if cur is not None:
                away = (act == Action.LEFT) == (cur["side"] == "right")
                cur["away_ticks" if away else "toward_ticks"] += 1
                cur["first_effective_ms"] = cur["first_effective_ms"] or t_end
        if r > 0 or info["crashed"]:
            outcome = "crash" if info["crashed"] else "pass"
            events.append({"t_ms": t_end, "type": outcome, "obstacle": cur["index"] if cur else None})
            if cur is not None:
                _close(cur, t_end, outcome, brain.gf_events)
                if outcome == "crash":
                    cur["cause"] = M.loss_cause(parent, env, {"jumps": cur["jumps"], "moves_away": cur["away_ticks"],
                                                              "moves_toward": cur["toward_ticks"]})
                obstacles.append(cur)
                cur = None
    if cur is not None:                              # the cap ended the episode with an obstacle on screen
        _close(cur, float(tick_end_ms(env.tick - 1)), "open", brain.gf_events)
        cur["t_contact"] = None
        cur["lead_ms"] = cur["lead_effective_ms"] = None
        obstacles.append(cur)
    summary = brain.summary()
    out = {"seed": seed, "score": int(info["score"]), "ticks": int(info["tick"]),
           "crashed": bool(info["crashed"]), "truncated": bool(info["truncated"]),
           "network_rate_hz": summary["network_rate_hz"], "active_fraction": summary["active_fraction"],
           "gf_spikes": summary["gf_spikes"], "n_spikes": summary["n_spikes"], "windows": summary["windows"],
           "input_cell_spikes": summary["input_cell_spikes"], "decoder_outputs": n_act,
           "obstacles": obstacles,
           "loss_cause": obstacles[-1].get("cause") if obstacles and obstacles[-1]["outcome"] == "crash" else None}
    if record:
        out["record"] = {"frames": frames, "t_ms": t_frames, "actions": actions, "loom": looms, "events": events}
    return out


M7C_OBSTACLE_FIELDS = {"index": "index", "t_spawn": "t_spawn", "first_action_ms": "first_effective_ms",
                       "moves_away": "away_ticks", "moves_toward": "toward_ticks", "jumps": "jumps",
                       "side": "side", "t_contact": "t_contact", "outcome": "outcome",
                       "gf_spikes": "gf_spikes", "lead_ms": "lead_effective_ms", "cause": "cause"}
EPISODE_FIELDS = ("score", "ticks", "crashed", "truncated", "gf_spikes", "n_spikes", "network_rate_hz",
                  "active_fraction", "input_cell_spikes", "loss_cause", "ignition")
GRID_FIELDS = ("scores", "ticks", "gf_spikes", "network_rate_hz", "ignited", "ignited_literal")


class ReproductionError(RuntimeError):
    pass


def as_m7c_obstacle(o: dict) -> dict:
    d = {m7c_key: o.get(ours) for m7c_key, ours in M7C_OBSTACLE_FIELDS.items()}
    if "cause" not in o:
        d.pop("cause")
    return d


def check_grid_row(row: str, mine: dict, committed: dict) -> None:
    """s = 1: one k of the training grid against the M7c2 main arm (brief §4: stop if different)."""
    if not math.isclose(mine["k"], committed["k"], rel_tol=1e-12, abs_tol=0.0):
        raise ReproductionError(f"{row}: k {mine['k']} vs committed {committed['k']}")
    bad = [f for f in GRID_FIELDS if mine[f] != committed[f]]
    if bad:
        raise ReproductionError(f"{row} k={mine['k']:.6g}: {bad} differ from m7c2-results.json "
                                f"(scores {mine['scores']} vs {committed['scores']})")


def check_eval(row: str, mine: list[dict], committed: list[dict]) -> None:
    """s = 1: the evaluation episodes, field by field, obstacles included."""
    if [e["seed"] for e in mine] != [e["seed"] for e in committed]:
        raise ReproductionError(f"{row}: evaluation seeds differ")
    for a, b in zip(mine, committed):
        bad = [f for f in EPISODE_FIELDS if a[f] != b[f]]
        closed = [as_m7c_obstacle(o) for o in a["obstacles"] if o["outcome"] != "open"]   # M7c drops open ones
        if closed != b["obstacles"]:
            bad.append("obstacles")
        if bad:
            raise ReproductionError(f"{row} seed {a['seed']}: {bad} differ from m7c2-results.json")


def grid_row(k: float, eps: list[dict]) -> dict:
    sc = np.array([e["score"] for e in eps])
    return {"k": k, "mean": float(sc.mean()), "min": int(sc.min()), "max": int(sc.max()), "scores": sc.tolist(),
            "ignited": int(sum(e["ignition"]["ignited"] for e in eps)),
            "ignited_literal": int(sum(e["ignition"]["literal"]["ignited"] for e in eps)),
            "viewer_failure": int(sum(e["ignition"]["viewer_failure"] for e in eps)),
            "ticks": [e["ticks"] for e in eps], "gf_spikes": [e["gf_spikes"] for e in eps],
            "network_rate_hz": [round(e["network_rate_hz"], 4) for e in eps],
            "episodes": [{f: e[f] for f in ("seed", "score", "ticks", "truncated", "active_fraction",
                                            "decoder_outputs", "obstacles")} for e in eps]}


# ---------------------------------------------------------------------------------------------
# controls
# ---------------------------------------------------------------------------------------------
def decoder_on(events, ends: np.ndarray, decoder) -> list[str]:
    """The decoder at every tick end, on a fixed spike record (it never feeds back)."""
    return [decoder.decode([e for e in events if e[0] < t], float(t)) for t in ends]


def control_ends(duration_ms: int) -> np.ndarray:
    """The tick ends ``m7c.control_run`` plays: ticks until the game clock reaches ``duration_ms``."""
    ends = [tick_end_ms(0)]
    while ends[-1] < duration_ms:
        ends.append(tick_end_ms(len(ends)))
    return np.array(ends)


def control_actions(gf_events, decoder, duration_ms: int) -> dict:
    """What the decoder would have done on a no-input control (decoder outputs per tick end)."""
    acts = decoder_on(gf_events, control_ends(duration_ms), decoder)
    on = np.array([a != "NOOP" for a in acts])
    onsets = int(on[0]) + int(np.sum(on[1:] & ~on[:-1])) if len(on) else 0
    return {"ticks": int(len(on)), "action_ticks": int(on.sum()), "onsets": onsets,
            "by_action": {a: int(sum(x == a for x in acts)) for a in ("JUMP", "LEFT", "RIGHT")},
            "duration_ms": int(duration_ms)}


# ---------------------------------------------------------------------------------------------
# parts
# ---------------------------------------------------------------------------------------------
def source_hashes() -> dict:
    """sha256 of the code the parts depend on. For this file only the part above the presentation
    marker counts, so the assembly, tables and figure can be fixed without re-running a part."""
    out = {}
    for rel in STRICT_SOURCES + RECORDED_SOURCES:
        raw = (ROOT / rel).read_bytes()
        if rel == "flysim/apps/games/m7g.py":
            head, sep, _ = raw.partition(("\n" + PRESENTATION_MARKER_LINE + "\n").encode())
            if not sep:
                raise RuntimeError("m7g.py lost its presentation marker")
            raw = head
        out[rel] = hashlib.sha256(raw).hexdigest()
    return out


def git_state() -> dict:
    def run(*a):
        try:
            return subprocess.check_output(["git", *a], cwd=ROOT, text=True).strip()
        except Exception:
            return None
    st = run("status", "--porcelain") or ""
    return {"head": run("rev-parse", "HEAD"), "modified": sorted(l[3:] for l in st.splitlines() if l)}


def write_part(parts: Path, name: str, doc: dict) -> Path:
    parts.mkdir(parents=True, exist_ok=True)
    p = parts / f"{name}.json"
    doc = {"part": name, "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "sources": source_hashes(),
           "git": git_state(), **doc}
    p.write_text(json.dumps(doc, separators=(",", ":")) + "\n")
    return p


def read_part(parts: Path, name: str) -> dict:
    p = parts / f"{name}.json"
    if not p.exists():
        raise FileNotFoundError(f"{p} is missing: run its stage first")
    return json.loads(p.read_text())


# ---------------------------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------------------------
def stage_bots(rows, parts: Path, workers: int) -> None:
    base = json.loads(M.BASELINES.read_text())
    m7c = json.loads(M.RESULTS_JSON.read_text())
    m7c2 = json.loads(M7C2_RESULTS.read_text())
    for row in rows:
        b = bots_row(row, base, m7c, m7c2, workers=workers)
        write_part(parts, f"bots-{row}", {"bots": b})
        log(f"bots {row}: θ={b['theta']:.4g} p99={b['p99']:.5g} ({b['p99_n_values']} values) | random "
            f"{b['random']['mean']:.1f}, fixed {b['fixed']['reference']} {b['fixed']['mean']:.1f}, brainless "
            f"{b['brainless']['mean']:.1f}, oracle {b['oracle']['mean']:.1f}, ceiling {b['ceiling']['ceiling']} "
            f"({b['wall_s']} s)")


def _brain(device: str):
    from ...graph import Graph
    from ...probe.export import neuron_coords
    from ...probe.sets import build_probe_sets, load_roi
    from .brain import EscapeSets, FlyBrain
    graph = Graph.load()
    roi = load_roi(graph)
    escape = EscapeSets.from_probe_sets(build_probe_sets(graph, roi, profile="escape"))
    brain = FlyBrain(graph, escape, k=1.0, device=device, keep_spikes=False, warmup_steps=M2.WARMUP_STEPS)
    choice = M2.load_decoder_choice()
    if (choice["window_ms"], choice["min_spikes"]) != (300.0, 2):
        raise RuntimeError(f"the phase-1 decoder is {choice}, not M7c2's main arm (300 ms, 2)")
    decoder = M2.arm_decoder("main", brain, choice)
    agent = BrainAgent(brain, decoder, brain_reset=brain.reset)
    return brain, agent, choice, {"escape": escape, "coords": neuron_coords(graph, roi)}


def stage_controls(parts: Path, device: str) -> None:
    from ...probe.motor import CountWindowDecoder
    brain, agent, choice, _ = _brain(device)
    m7c2 = json.loads(M7C2_RESULTS.read_text())
    dec = CountWindowDecoder(choice["window_ms"], choice["min_spikes"])
    out = {}
    t0 = time.time()
    for s in TRAIN_SEEDS + EVAL_SEEDS:
        c = M.control_run(brain, s)
        mine = {k: v for k, v in c.items() if k != "windows"}
        if mine != m7c2["controls"][str(s)]:
            raise ReproductionError(f"control seed {s} differs from m7c2-results.json")
        out[str(s)] = c | {"gf_events": [list(e) for e in brain.gf_events],
                           "decoder": control_actions(brain.gf_events, dec, M.CONTROL_MS)}
    write_part(parts, "controls", {"controls": out, "decoder": dec.describe(), "decoder_choice": choice,
                                   "matches_m7c2": True, "wall_s": round(time.time() - t0, 1)})
    on = sum(c["decoder"]["action_ticks"] for c in out.values())
    ons = sum(c["decoder"]["onsets"] for c in out.values())
    log(f"controls: 20 seeds equal to m7c2-results.json; decoder on {on} ticks, {ons} onsets in 100 s "
        f"({time.time() - t0:.0f} s)")


def stage_fly(rows, parts: Path, device: str, bundles: bool = True) -> None:
    brain, agent, choice, ctx = _brain(device)
    ctl = read_part(parts, "controls")["controls"]
    controls = {int(s): c for s, c in ctl.items()}
    m7c2 = json.loads(M7C2_RESULTS.read_text())
    for row in rows:
        b = read_part(parts, f"bots-{row}")["bots"]
        p99 = b["p99"]
        grid = M.k_grid(p99)
        committed = m7c2["arms"]["main"]["games"][row] if row in BASE_ROWS else None
        if committed is not None and not np.allclose(grid, committed["k_grid"], rtol=1e-12, atol=0):
            raise ReproductionError(f"{row}: k grid differs from M7c2's")
        t0, sim_ms = time.time(), 0
        rows_k = {}
        for i, k in enumerate(grid):
            brain.k = k
            eps = [play(row, agent, brain, s) for s in TRAIN_SEEDS]
            for e in eps:
                e["ignition"] = M.ignition(e, controls[e["seed"]])
                e.pop("windows")
                sim_ms += M2.WARMUP_STEPS + tick_end_ms(e["ticks"] - 1)
            rows_k[k] = grid_row(k, eps)
            if committed is not None:
                check_grid_row(row, rows_k[k], committed["selection"]["grid"][i])
            log(f"  {row} k={k:9.4g}: train mean {rows_k[k]['mean']:6.2f} scores {rows_k[k]['scores']} "
                f"game s {sum(rows_k[k]['ticks']) / 60:.0f} ignited {rows_k[k]['ignited']} "
                f"({time.time() - t0:.0f} s)")
        sel = M.select_best(rows_k)
        k = sel["chosen"]
        brain.k = k
        fly_eps = []
        for s in EVAL_SEEDS:
            e = play(row, agent, brain, s)
            e["ignition"] = M.ignition(e, controls[s])
            e.pop("windows")
            sim_ms += M2.WARMUP_STEPS + tick_end_ms(e["ticks"] - 1)
            fly_eps.append(e)
        if committed is not None:
            if k != committed["k"]:
                raise ReproductionError(f"{row}: selected k {k} vs committed {committed['k']}")
            check_eval(row, fly_eps, committed["fly"]["episodes"])
        doc = {"row": row, "p99": p99, "k_grid": list(grid), "selection": sel, "k": k,
               "grid": [rows_k[g] for g in grid], "eval": fly_eps,
               "decoder": agent.decoder.describe(), "decoder_choice": choice, "brain": brain.describe(),
               "reproduces_m7c2": committed is not None}
        if bundles and row in SLOW_ROWS:
            doc["bundle"] = make_bundle(row, k, fly_eps[EVAL_SEEDS.index(M.BUNDLE_SEED)]["score"], agent, brain, ctx)
        doc["wall_s"] = round(time.time() - t0, 1)
        doc["sim_s"] = round(sim_ms / 1000.0, 1)
        write_part(parts, f"fly-{row}", doc)
        fly = [e["score"] for e in fly_eps]
        log(f"{row}: k={k:.4g} fly {np.mean(fly):.2f} {fly}"
            + (" | s = 1 reproduces m7c2-results.json" if committed is not None else "")
            + f" ({doc['wall_s']} s wall, {doc['sim_s']} s simulated)")


# ---------------------------------------------------------------------------------------------
# replay bundle (M7c / M7c2 format; game.json carries the variant's own rules)
# ---------------------------------------------------------------------------------------------
def make_bundle(row: str, k: float, eval_score: int, agent: BrainAgent, brain, ctx: dict) -> dict:
    brain.k = k
    brain.keep_spikes = True
    try:
        ep = play(row, agent, brain, M.BUNDLE_SEED, record=True)
    finally:
        brain.keep_spikes = False
    if ep["score"] != eval_score:
        raise RuntimeError("bundle replay differs from the evaluation run (determinism)")
    try:
        return write_bundle(row, k, ep, ctx, brain, BUNDLE_DIR / f"{row}-brain-seed{M.BUNDLE_SEED}",
                            agent.decoder.describe())
    finally:
        brain.spikes_t, brain.spikes_i = [], []


def write_bundle(row: str, k: float, ep: dict, ctx: dict, brain, out_dir: Path, decoder_desc: dict) -> dict:
    """``m7c.write_bundle`` with M7c2's labels, for a row ``base.make`` does not know."""
    from ...probe.export import write_run_json
    from ...probe.probe import bin_spikes
    rec = ep.pop("record")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "frames.npz", frames=np.stack(rec["frames"]).astype(np.uint8),
                        t_ms=np.asarray(rec["t_ms"], dtype=np.float64),
                        actions=np.asarray(rec["actions"], dtype=np.int8),
                        loom=np.asarray(rec["loom"], dtype=np.float32))
    t, idx = brain.spikes()
    dur = min(int(brain.t_ms), M.BUNDLE_MAX_MS)
    keep = t < dur
    t, idx = t[keep], idx[keep]
    sets = ctx["escape"].probe_sets
    rt = bin_spikes(t, idx, 1.0, 1.0, dur, sets)
    rules = factory(row).describe()
    write_run_json(out_dir / "run.json", run_id=f"m7g-{row}-brain-seed{M.BUNDLE_SEED}", sensory_mode="rate",
                   dt_ms=1.0, rt=rt, sets=sets, coords=ctx["coords"],
                   input_label=f"{row}: looming (left/right) x k = {k:.4g}",
                   envelope=M._envelope(rec, dur), spikes=(rt.t_bin, idx),
                   engine_params=brain.params.to_dict(), adapter={"adapter": "LoomingDetector -> LC4+LPLC2",
                                                                  "k": k, **LoomingDetector.describe()},
                   stimulus={"kind": "game", "game": row, "seed": M.BUNDLE_SEED},
                   extra_meta={"probe_profile": "escape", "bundle_ms": dur, "game_ms": int(brain.t_ms),
                               "truncated_to_bundle_max": brain.t_ms > M.BUNDLE_MAX_MS,
                               "sensory_mode_note": "game looming signal, not the JO adapter",
                               "m7c2_arm": "main", "m7g_row": row, "parent": family(row),
                               "speed_divisor": speed_divisor(row), "warmup_steps": brain.warmup_steps,
                               "clock": "0 = end of the warm-up; warm-up spikes not included"})
    game_doc = {"game": row, "seed": M.BUNDLE_SEED, "score": ep["score"], "ticks": ep["ticks"],
                "duration_ms": int(brain.t_ms), "bundle_ms": dur,
                "agent": {"kind": "fly brain", "k": k, "decoder": decoder_desc, "brain": brain.describe()},
                "crashed": ep["crashed"], "loss_cause": ep.get("loss_cause"),
                "events": M.bundle_events(rec["events"], len(rec["frames"])),
                "rules": rules, "tick_ms": "floor((k+1)*1000/60)"}
    (out_dir / "game.json").write_text(json.dumps(game_doc, indent=1) + "\n")
    return {"dir": str(out_dir.relative_to(ROOT)) if out_dir.resolve().is_relative_to(ROOT) else str(out_dir),
            "frames": len(rec["frames"]), "bundle_ms": dur, "game_ms": int(brain.t_ms),
            "spikes_in_bundle": int(len(t))}


# ---------------------------------------------------------------------------------------------
PRESENTATION_MARKER_LINE = "# ==== presentation: readings, assembly, tables, figure (not part of the parts' source hash) ===="
# ==== presentation: readings, assembly, tables, figure (not part of the parts' source hash) ====
# Below: the post-hoc addendum (brief "추가 등록", decided after the dodge_v2_s2 result; its own
# stage, its part records the whole file's sha256), then the readings, the assembly, the tables
# and the figure, which only read the parts.
# ---------------------------------------------------------------------------------------------
# ---- post-hoc addendum: cells A and B, the loom-by-distance check (docs/m7g-brief.md 추가 등록) ----
APPEARANCE_MS = 300.0             # descriptive split: a first action within W (= the decoder window) of
                                  # the obstacle's first frame, when the one-frame appearance pulse is
                                  # still inside the decoder's window
LOOM_BINS = (40.0, 20.0, 10.0, 5.0, 3.0, 2.0, 1.5, 0.5)     # obstacle distance bins, far to near
POSTHOC_READINGS = {
    (True, False): "시간이 열쇠다. \"늦게 오는 신호\" 가설을 지지한다",
    (False, True): "전류가 열쇠다. 첫 칸의 k 격자 꼭대기가 낮았던 것이지, 신호가 늦어서가 아니다",
    (True, True): "둘 다 효과가 있다",
    (False, False): "느린 판의 승리는 시간과 센 전류가 함께 있을 때만 나온다",
}


def k_top(p99: float) -> float:
    return M.K_SPAN[1] / p99


def cell_a_points(k_grid: list[float], k_match: float) -> list[int]:
    """The grid points that bracket ``k_match`` (one point if it is a grid point)."""
    for i, k in enumerate(k_grid):
        if math.isclose(k, k_match, rel_tol=1e-12, abs_tol=0.0):
            return [i]
    if not k_grid[0] < k_match < k_grid[-1]:
        raise ValueError(f"k_match {k_match} is outside the grid [{k_grid[0]}, {k_grid[-1]}]")
    i = max(j for j, k in enumerate(k_grid) if k < k_match)
    return [i, i + 1]


def cell_a_verdict(beats: list[bool]) -> str:
    """Two bracketing points: both beat -> beat, neither -> not, one -> split (read as "not")."""
    if all(beats):
        return "beat"
    return "not" if not any(beats) else "split"


def posthoc_reading(a: str, b_beats: bool) -> dict:
    a_beats = a == "beat"                         # a split cell A is read as "not" (no exaggeration)
    return {"A": a, "A_read_as_beat": a_beats, "B": "beat" if b_beats else "not",
            "sentence": POSTHOC_READINGS[(a_beats, bool(b_beats))]}


def appearance_split(obstacles: list[dict], w: float = APPEARANCE_MS) -> dict:
    """Correct first actions within ``w`` of the obstacle's first frame, and after (descriptive)."""
    corr = [o for o in obstacles if o["outcome"] in ("pass", "crash") and o["first_correct"]]
    early = [o for o in corr if o["first_action_ms"] - o["t_spawn"] <= w]
    late = [o for o in corr if o["first_action_ms"] - o["t_spawn"] > w]
    passed = sum(o["outcome"] == "pass" for o in obstacles)
    p_within = sum(o["outcome"] == "pass" for o in early)
    return {"window_ms": w, "correct_first": len(corr), "within": len(early), "after": len(late),
            "lead_ms_median_within": _q([o["lead_ms"] for o in early], 50),
            "lead_ms_median_after": _q([o["lead_ms"] for o in late], 50),
            "passed": passed, "passed_within": p_within, "passed_after": sum(o["outcome"] == "pass" for o in late),
            # brief 추가 등록 보충: of the obstacles passed, the share whose correct first action came
            # within w of the first frame
            "passed_within_fraction": (p_within / passed) if passed else None}


def loom_by_distance(row: str, theta: float | None, seeds=TRAIN_SEEDS, bins=LOOM_BINS) -> dict:
    """Brief 추가 등록 "근사 확인": the brainless bot's frames (the p99 frames), the looming value on
    the obstacle's half (max of the two halves) binned by the obstacle's distance in that frame.
    The obstacle's first frame (the appearance pulse, distance D_SPAWN) is its own row. Also the
    same for the first speed level only (the first SPEED_EVERY obstacles of an episode).
    ``theta`` None plays the always-NOOP bot instead: it never moves, so its frames show the
    approach geometry alone (one obstacle per seed, the one it hits)."""
    edges = np.asarray(bins)
    acc = {key: [[] for _ in range(len(edges) - 1)] for key in ("all", "level0")}
    first = {"all": [], "level0": []}
    for seed in seeds:
        env, det = factory(row), LoomingDetector()
        frame = env.reset(seed)
        bot = ThresholdAgent(theta) if theta is not None else FixedAgent("NOOP")
        bot.reset(seed)
        seen_prev, n_done = False, 0
        done = False
        while not done:
            d = env.dist
            left, right = det.update(frame)
            v = max(left, right)
            if d is not None:
                keys = ("all", "level0") if n_done < 4 else ("all",)
                if not seen_prev:
                    for kk in keys:
                        first[kk].append(v)
                else:
                    b = int(np.searchsorted(-edges, -d, side="right")) - 1
                    b = min(max(b, 0), len(edges) - 2)
                    for kk in keys:
                        acc[kk][b].append(v)
            seen_prev = d is not None
            frame, r, done, _ = env.step(bot.act(frame))
            if r > 0:
                n_done += 1
    out = {}
    for kk in acc:
        out[kk] = {"appearance": {"n": len(first[kk]), "mean": float(np.mean(first[kk])) if first[kk] else None},
                   "bins": [{"from": float(edges[i]), "to": float(edges[i + 1]), "n": len(x),
                             "mean": float(np.mean(x)) if x else None} for i, x in enumerate(acc[kk])]}
    return out


def stage_posthoc(parts: Path, device: str, gpu: bool = True) -> None:
    """CPU: the random bot on the training seeds of every row, the loom-by-distance table.
    GPU: cell B — the parent game at k = k_top(s) / s, training seeds 0-9, one per slow row."""
    t0 = time.time()
    rnd = {row: run_many_row(row, lambda env: RandomAgent(), TRAIN_SEEDS)["scores"] for row in ROWS}
    loom = {row: loom_by_distance(row, read_part(parts, f"bots-{row}")["bots"]["theta"]) for row in ROWS}
    loom_noop = {row: loom_by_distance(row, None) for row in ROWS}
    file_sha = hashlib.sha256((ROOT / "flysim/apps/games/m7g.py").read_bytes()).hexdigest()
    write_part(parts, "posthoc-cpu", {"random_train": rnd, "loom_by_distance": loom, "loom_by_distance_noop": loom_noop,
                                      "m7g_file_sha256": file_sha,
                                      "wall_s": round(time.time() - t0, 1)})
    log(f"posthoc cpu: random on training seeds, loom by distance ({time.time() - t0:.0f} s)")
    if not gpu:
        return
    brain, agent, choice, _ = _brain(device)
    controls = {int(s): c for s, c in read_part(parts, "controls")["controls"].items()}
    cells = {}
    for row in SLOW_ROWS:
        tb = time.time()
        parent = family(row)
        k = k_top(read_part(parts, f"fly-{row}")["p99"]) / speed_divisor(row)
        brain.k = k
        eps = [play(parent, agent, brain, s) for s in TRAIN_SEEDS]
        for e in eps:
            e["ignition"] = M.ignition(e, controls[e["seed"]])
            e.pop("windows")
        cells[row] = {"game": parent, "k": k, "k_rule": "k_top(s) / s = 5120 / p99(row) / s", **grid_row(k, eps),
                      "wall_s": round(time.time() - tb, 1)}
        log(f"  cell B for {row}: {parent} at k={k:.4g}: mean {cells[row]['mean']:.2f} scores {cells[row]['scores']} "
            f"ignited {cells[row]['ignited']} ({time.time() - tb:.0f} s)")
    write_part(parts, "posthoc-cellB", {"cells": cells, "decoder_choice": choice, "brain": brain.describe(),
                                        "m7g_file_sha256": file_sha, "wall_s": round(time.time() - t0, 1)})


def _cell(eps_or_row: dict, random_scores: list[int], k_kind: str) -> dict:
    obs = [o for e in eps_or_row["episodes"] for o in e["obstacles"]]
    st = obstacle_stats(obs, k_kind)
    v = M.paired(eps_or_row["scores"], random_scores)
    return {"k": eps_or_row["k"], "scores": eps_or_row["scores"], "mean": eps_or_row["mean"],
            "random_train": list(random_scores), "vs_random": v, "beats_random": v["beats"],
            "lead_ms_median": st["lead_ms_median"], "correct_first": st["correct_first"], "acted": st["acted"],
            "away_ticks_median": st.get("away_ticks_median"), "away_ge_step_aside": st.get("away_ge_step_aside"),
            "obstacles": st["n"], "passed": st["passed"], "appearance": appearance_split(obs),
            "ignited": eps_or_row["ignited"], "viewer_failure": eps_or_row.get("viewer_failure")}


def posthoc(parts: Path, fly_docs: dict) -> dict:
    cpu, cb = read_part(parts, "posthoc-cpu"), read_part(parts, "posthoc-cellB")
    rnd = cpu["random_train"]
    out = {"decided": "after the dodge_v2_s2 result (brief 추가 등록); sections 2-5 unchanged",
           "rules": {"cell_A": "k_match = s * 5120 / p99(parent, M7c); the grid points bracketing it (training seeds)",
                     "cell_B": "parent game, 5-min cap, k = k_top(s) / s, training seeds 0-9 (not in the verdict table)",
                     "random": "cell A: the slow row's random bot; cell B: the parent game's; training seeds 0-9",
                     "two_points": "both beat -> beat; neither -> not; one -> split, read as not",
                     "appearance": f"descriptive: correct first actions within {APPEARANCE_MS:g} ms of the first frame"},
           "rows": {}, "loom_by_distance": {}}
    for row in SLOW_ROWS:
        fly = fly_docs[row]
        s, parent, k_kind = speed_divisor(row), family(row), kind(row)
        p_par = fly_docs[parent]["p99"]
        k_match = s * k_top(p_par)
        idx = cell_a_points(fly["k_grid"], k_match)
        a_pts = [dict(_cell(fly["grid"][i], rnd[row], k_kind), k_index=i) for i in idx]
        a = cell_a_verdict([p["beats_random"] for p in a_pts])
        b = _cell(cb["cells"][row], rnd[parent], k_kind) | {"game": parent}
        top = _cell(fly["grid"][-1], rnd[row], k_kind) | {"k_index": len(fly["k_grid"]) - 1}
        out["rows"][row] = {"s": s, "k_top_s1": k_top(p_par), "k_top": k_top(fly["p99"]), "k_match": k_match,
                            "drive_ratio_top": k_top(fly["p99"]) / (s * k_top(p_par)),
                            "appearance_current_ratio_top": k_top(fly["p99"]) / k_top(p_par),
                            "cell_A": {"points": a_pts, "verdict": a}, "cell_B": b, "top_k": top,
                            "reading": posthoc_reading(a, b["beats_random"])}
    lb = cpu["loom_by_distance"] | {f"noop:{r}": v for r, v in cpu["loom_by_distance_noop"].items()}
    for row in SLOW_ROWS:
        s, parent = speed_divisor(row), family(row)
        tab = {}
        for kk, src in (("all", ""), ("level0", ""), ("noop", "noop:")):
            mine = lb[src + row]["all" if kk == "noop" else kk]
            base = lb[src + parent]["all" if kk == "noop" else kk]
            rows = [{"bin": "appearance (first frame, d = 40)", "slow_mean": mine["appearance"]["mean"],
                     "s1_mean": base["appearance"]["mean"]}]
            rows += [{"bin": f"{x['from']:g}-{y['to']:g}" if False else f"{x['from']:g}-{x['to']:g}",
                      "slow_mean": x["mean"], "s1_mean": y["mean"], "n_slow": x["n"], "n_s1": y["n"]}
                     for x, y in zip(mine["bins"], base["bins"])]
            for r_ in rows:
                ok = r_["slow_mean"] is not None and r_["s1_mean"]
                r_["slow_times_s_over_s1"] = (r_["slow_mean"] * s / r_["s1_mean"]) if ok else None
            tab[kk] = rows
        out["loom_by_distance"][row] = tab
    out["loom_by_distance_raw"] = {"brainless": cpu["loom_by_distance"], "noop": cpu["loom_by_distance_noop"]}
    out["random_train"] = rnd
    out["cell_B_ignition"] = {r: cb["cells"][r]["ignited"] for r in SLOW_ROWS}
    out["cell_B_viewer_failure"] = {r: cb["cells"][r]["viewer_failure"] for r in SLOW_ROWS}
    out["parts"] = {"cpu": cpu["generated"], "cellB": cb["generated"], "m7g_file_sha256": cb["m7g_file_sha256"]}
    return out


# ---- readings, assembly ------------------------------------------------------------------------
def git_state_now() -> dict:
    """``git_state`` for the results file. (The parts' ``git_state`` strips the porcelain output as a
    whole, which cuts the first character off the first modified path, e.g. "lysim/..."; the
    parts keep that record as written.)"""
    try:
        head_ = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        st = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    except Exception:
        return {"head": None, "modified": []}
    return {"head": head_, "modified": sorted(l[3:] for l in st.splitlines() if l.strip())}


def _q(x: list[float], q: float) -> float | None:
    return float(np.percentile(x, q)) if len(x) else None


def obstacle_stats(obstacles: list[dict], k: str, onset_rate_per_ms: float | None = None) -> dict:
    """Brief §5.1 per set of obstacles: first action, its lead, its direction, the move length,
    the outcome and the giant-fibre spikes. Open obstacles (cut by the cap) are counted apart."""
    closed = [o for o in obstacles if o["outcome"] in ("pass", "crash")]
    acted = [o for o in closed if o["first_action"] is not None]
    correct = [o for o in acted if o["first_correct"]]
    leads = [o["lead_ms"] for o in correct]
    on_ms = [o["t_contact"] - o["t_spawn"] for o in closed]
    d = {"n": len(closed), "open": len(obstacles) - len(closed),
         "passed": sum(o["outcome"] == "pass" for o in closed), "crashed": sum(o["outcome"] == "crash" for o in closed),
         "acted": len(acted), "correct_first": len(correct),
         "direction_correct_fraction": (len(correct) / len(acted)) if acted else None,
         "lead_ms_median": _q(leads, 50), "lead_ms_q25": _q(leads, 25), "lead_ms_q75": _q(leads, 75),
         "lead_ms_correct": sorted(leads),
         "lead_ms_any_first_median": _q([o["lead_ms"] for o in acted], 50),
         "on_screen_ms_median": _q(on_ms, 50),
         "actions_per_s_on_screen": (sum(o["n_actions"] for o in closed) / (sum(on_ms) / 1000.0)) if sum(on_ms) else None,
         "gf_per_obstacle": {"L": float(np.mean([o["gf_L"] for o in closed])) if closed else None,
                             "R": float(np.mean([o["gf_R"] for o in closed])) if closed else None}}
    if onset_rate_per_ms is not None and closed:       # ASSUMPTION: noise onsets as a Poisson process
        d["acted_expected_from_noise"] = float(np.mean([1 - math.exp(-onset_rate_per_ms * m) for m in on_ms]))
        d["acted_fraction"] = len(acted) / len(closed)
    if k == "dodge" and closed:
        away = [o["away_ticks"] for o in closed]
        d["away_ticks_median"] = _q(away, 50)
        d["away_ticks_max"] = int(max(away))
        d["away_ge_step_aside"] = sum(a >= STEP_ASIDE_TICKS for a in away)
        d["toward_ticks_median"] = _q([o["toward_ticks"] for o in closed], 50)
        d["gf_pillar_side"] = int(sum(o["gf_L"] if o["side"] == "left" else o["gf_R"] for o in closed))
        d["gf_other_side"] = int(sum(o["gf_R"] if o["side"] == "left" else o["gf_L"] for o in closed))
    if k == "jump" and closed:
        d["jumped"] = sum(o["jumps"] > 0 for o in closed)
    return d


def rises(values: list[float | None]) -> bool:
    return all(v is not None for v in values) and all(a < b for a, b in zip(values, values[1:]))


READINGS = {
    ("yes", "yes"): "\"늦게 오는 신호\" 가설을 지지한다. 첫 칸의 패배는 회로가 못 봐서가 아니라 볼 시간이 없어서였다",
    ("yes", "no"): "행동은 일찍 나오는데 점수가 안 오른다. away_ticks(이동이 12틱에 못 미치는가 — 해독기가 한 번 검출에 "
                   "내는 행동 길이), 방향 정답률을 보고 다른 원인을 적는다(가정으로)",
    ("no", None): "첫 행동이 다가옴을 따라 일찍 오지 않는다. 가설을 지지하지 않는다. 행동이 잡음을 따르는지"
                  "(대조군 오경보율과 비교) 적는다",
}


def reading(p1: bool, p2: bool) -> dict:
    key = ("no", None) if not p1 else ("yes", "yes" if p2 else "no")
    return {"P1": p1, "P2": p2, "row": f"P1 {key[0]}" + (f", P2 {key[1]}" if key[1] else ", P2 -"),
            "sentence": READINGS[key]}


def verdict(row: str, fly: list[int], bots: dict) -> dict:
    vr = M.paired(fly, bots["random"]["scores"])
    vf = M.paired(fly, bots["fixed"]["scores"])
    vb = M.paired(fly, bots["brainless"]["scores"])
    ceiling = bots["ceiling"]["ceiling"]
    if vb["beats"] and not ceiling:
        vs_b = "beat the brainless bot"
    elif vb["loses"]:
        vs_b = "lost to the brainless bot"
    else:
        vs_b = "indistinguishable from the brainless bot"
    word = lambda v: "beat" if v["beats"] else ("lost" if v["loses"] else "indistinguishable")
    return {"vs_random": vr, "vs_fixed": vf, "vs_brainless": vb, "beat_random": vr["beats"],
            "vs_random_word": word(vr), "vs_fixed_word": word(vf), "fixed_reference": bots["fixed"]["reference"],
            "brainless_verdict": vs_b, "ceiling": ceiling,
            "ceiling_note": "천장 — 판정 불가 (beating the brainless bot cannot be judged)" if ceiling else None}


# ---------------------------------------------------------------------------------------------
# assemble
# ---------------------------------------------------------------------------------------------
def compact_obstacles(row_doc: dict) -> dict:
    cols = ["phase", "k_index", "seed", "index", "t_spawn", "t_contact", "outcome", "side", "first_action",
            "first_action_ms", "first_correct", "lead_ms", "n_actions", "away_ticks", "toward_ticks", "jumps",
            "first_effective_ms", "gf_L", "gf_R", "cause"]
    data = []
    for ki, g in enumerate(row_doc["grid"]):
        for e in g["episodes"]:
            for o in e["obstacles"]:
                data.append(["train", ki, e["seed"]] + [o.get(c) for c in cols[3:]])
    ki = row_doc["k_grid"].index(row_doc["k"])
    for e in row_doc["eval"]:
        for o in e["obstacles"]:
            data.append(["eval", ki, e["seed"]] + [o.get(c) for c in cols[3:]])
    return {"columns": cols, "rows": data}


def stage_assemble(parts: Path, out: Path = RESULTS_JSON, figure: Path = FIGURE, tables: Path = TABLES_MD) -> dict:
    ctl_part = read_part(parts, "controls")
    docs = {}
    for row in ROWS:
        b, f = read_part(parts, f"bots-{row}"), read_part(parts, f"fly-{row}")
        for p in (b, f):
            bad = [r for r in STRICT_SOURCES if p["sources"].get(r) != ctl_part["sources"].get(r)]
            if bad:
                raise RuntimeError(f"part {p['part']} was made by other code than the controls: {bad}")
        docs[row] = (b["bots"], f, b, f)
    controls = ctl_part["controls"]
    onset_rate = sum(c["decoder"]["onsets"] for c in controls.values()) / sum(c["decoder"]["duration_ms"]
                                                                              for c in controls.values())
    variants, lead_tables = {}, {}
    for row, (bots, fly, bpart, fpart) in docs.items():
        k_kind = kind(row)
        scores = [e["score"] for e in fly["eval"]]
        v = verdict(row, scores, bots)
        per_k = []
        for ki, g in enumerate(fly["grid"]):
            obs = [o for e in g["episodes"] for o in e["obstacles"]]
            per_k.append({"k_index": ki, "k": g["k"], "k_times_p99": g["k"] * fly["p99"], "train_mean": g["mean"],
                          **obstacle_stats(obs, k_kind, onset_rate)})
        eval_obs = [o for e in fly["eval"] for o in e["obstacles"]]
        ign = [e["ignition"] for e in fly["eval"]]
        sel_train = [o for e in fly["grid"][fly["k_grid"].index(fly["k"])]["episodes"] for o in e["obstacles"]]
        top_train = [o for e in fly["grid"][-1]["episodes"] for o in e["obstacles"]]
        variants[row] = {
            "family": family(row), "s": speed_divisor(row), "rules": bots["rules"],
            "theta": bots["theta"], "theta_selection": bots["theta_selection"],
            "p99": fly["p99"], "p99_n_values": bots["p99_n_values"], "loom_nonzero": bots["loom_nonzero"],
            "k_grid": fly["k_grid"], "k": fly["k"], "k_index": fly["k_grid"].index(fly["k"]),
            "selection": fly["selection"] | {"grid": [{kk: g[kk] for kk in g if kk != "episodes"} for g in fly["grid"]]},
            "fly": {"scores": scores, "mean": float(np.mean(scores)), "min": int(min(scores)), "max": int(max(scores)),
                    "episodes": [{kk: e[kk] for kk in e if kk != "obstacles"} for e in fly["eval"]]},
            "random": bots["random"], "fixed": bots["fixed"], "brainless": bots["brainless"],
            "oracle": bots["oracle"], "ceiling": bots["ceiling"], "verdict": v,
            "obstacles_by_k": per_k, "obstacles_eval": obstacle_stats(eval_obs, k_kind, onset_rate),
            "obstacles": compact_obstacles(fly),
            "appearance": {"top_k_train": appearance_split(top_train), "selected_k_eval": appearance_split(eval_obs),
                           "selected_k_train": appearance_split(sel_train)},
            "ignition_eval": {"like_for_like": sum(i["ignited"] for i in ign),
                              "literal": sum(i["literal"]["ignited"] for i in ign),
                              "runaway": sum(i["window_runaway"] or i["control_runaway"] for i in ign),
                              "viewer_failure": sum(i["viewer_failure"] for i in ign)},
            "ignition_train": {"like_for_like": sum(g["ignited"] for g in fly["grid"]),
                               "literal": sum(g["ignited_literal"] for g in fly["grid"]),
                               "viewer_failure": sum(g["viewer_failure"] for g in fly["grid"])},
            "bundle": fly.get("bundle"), "reproduces_m7c2": fly["reproduces_m7c2"],
            "reproduced_bots": bots.get("reproduced"),
            "wall_s": {"bots": bots["wall_s"], "fly": fly["wall_s"]}, "sim_s": fly["sim_s"],
            "parts": {"bots": bpart["generated"], "fly": fpart["generated"]}}
        lead_tables[row] = per_k[-1]                   # the grid's top k (brief §5.2)
    p1_by = {}
    for fam, rows in FAMILIES.items():
        meds = [lead_tables[r]["lead_ms_median"] for r in rows]
        p1_by[fam] = {"rows": list(rows), "s": [speed_divisor(r) for r in rows], "lead_ms_median": meds,
                      "ratio_to_s1": [(m / meds[0]) if (m is not None and meds[0]) else None for m in meds],
                      "rises": rises(meds)}
    p1 = all(x["rises"] for x in p1_by.values())
    p2_by = {r: variants[r]["verdict"]["beat_random"] for r in SLOW_ROWS}
    p2 = any(p2_by.values())
    post = posthoc(parts, {row: d[1] for row, d in docs.items()})
    result = {
        "brief": BRIEF, "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "git": git_state_now(),
        "sources_note": "base.py changed between the bots parts and the rest (another window added names to "
                        "base.make, M7f1); this experiment never calls base.make",
        "part_sources_base_py": {n: read_part(parts, n)["sources"]["flysim/apps/games/base.py"]
                                 for n in ["controls", "posthoc-cpu", "posthoc-cellB"] + [f"{x}-{r}" for r in ROWS
                                                                                       for x in ("bots", "fly")]},
        "sources": ctl_part["sources"], "parts_dir": str(parts.relative_to(ROOT)) if parts.is_relative_to(ROOT) else str(parts),
        "notice": "M7g was added after the fly lost the first rung (jump, dodge v1, v2): an easier rung added later",
        "train_seeds": list(TRAIN_SEEDS), "eval_seeds": list(EVAL_SEEDS), "warmup_steps": M2.WARMUP_STEPS,
        "decoder_choice": ctl_part["decoder_choice"], "decoder": ctl_part["decoder"],
        "brain": docs[ROWS[0]][1]["brain"] | {"k": "per row"}, "detector": LoomingDetector.describe(),
        "rules": {"rows": list(ROWS), "slow_rows": list(SLOW_ROWS), "k_points": M.K_POINTS, "k_span": list(M.K_SPAN),
                  "p99_frames": "brainless bot (θ chosen on the row) on training seeds 0-9; s = 1: M7c's value",
                  "theta_grid": list(THETA_GRID), "sem_mult": M.SEM_MULT, "fixed_order": list(FIXED_ORDER),
                  "ceiling": "slow rows: brainless == oracle on all 10 eval seeds; s = 1: M7c's (dodge_v2 72)",
                  "control": "same seed, warm-up, then 5 s with no input (M7c2)", "ignition_table": "like_for_like",
                  "p1_episodes": "training seeds 0-9 at the grid's top k (k = 5120/p99)",
                  "first_action": "first non-NOOP decoder output while the obstacle is on screen; time = that tick's end",
                  "correct": "dodge: the move away from the pillar's side; jump: JUMP",
                  "p1": "median lead of correct first actions rises strictly with s in both families",
                  "p2": "beat the random bot (selected k, eval seeds) in at least one slow row",
                  "noise_expectation": "ASSUMPTION: control decoder onsets as a Poisson process, rate below"},
        "controls": {s: {k: v for k, v in c.items() if k not in ("windows", "gf_events")} for s, c in controls.items()},
        "control_decoder_onset_rate_per_s": onset_rate * 1000.0,
        "control_decoder_action_tick_fraction": sum(c["decoder"]["action_ticks"] for c in controls.values())
                                                / sum(c["decoder"]["ticks"] for c in controls.values()),
        "rows": variants,
        "hypothesis": {"top_k_table": lead_tables, "P1": p1_by, "P1_holds": p1, "P2": p2_by, "P2_holds": p2,
                       "reading": reading(p1, p2)},
        "p99_vs_s": {r: {"s": speed_divisor(r), "p99": variants[r]["p99"],
                         "p99_times_s": variants[r]["p99"] * speed_divisor(r),
                         "distinct_nonzero_values": variants[r]["loom_nonzero"]["distinct"]} for r in ROWS},
        "posthoc": post,
    }
    out.write_text(json.dumps(result, indent=1) + "\n")
    draw_figure(result, figure)
    tables.parent.mkdir(parents=True, exist_ok=True)
    tables.write_text(markdown_tables(result))
    log(f"assembled -> {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out} "
        f"({out.stat().st_size / 1e6:.2f} MB); P1 {p1}, P2 {p2}: {result['hypothesis']['reading']['row']}")
    return result


# ---------------------------------------------------------------------------------------------
# tables for the report
# ---------------------------------------------------------------------------------------------
def _f(x, nd: int | None = None) -> str:
    """A table cell: — for None; ``nd`` decimals, or (``nd`` None) an integer unless it has a fraction."""
    if x is None:
        return "—"
    if nd is None:
        return f"{x:.0f}" if float(x).is_integer() else f"{x:.1f}"
    return f"{x:.{nd}f}"


def markdown_tables(r: dict) -> str:
    L = ["# M7g tables (generated by m7g.py --stage assemble)", ""]
    L += ["## scores (eval seeds 10-19)", "",
          "| row | s | k (index) | fly | random | fixed (which) | brainless | oracle | vs random | vs fixed | vs brainless |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for row, v in r["rows"].items():
        vd = v["verdict"]
        L.append(f"| {row} | {v['s']} | {v['k']:.4g} ({v['k_index']}) | {v['fly']['mean']:.1f} | {v['random']['mean']:.1f} | "
                 f"{v['fixed']['mean']:.1f} ({v['fixed']['reference']}) | {v['brainless']['mean']:.1f} | "
                 f"{v['oracle']['mean']:.1f} | {vd['vs_random']['mean']:+.2f} ± {vd['vs_random']['sem']:.3f} {vd['vs_random_word']} | "
                 f"{vd['vs_fixed']['mean']:+.2f} ± {vd['vs_fixed']['sem']:.3f} {vd['vs_fixed_word']} | "
                 f"{vd['vs_brainless']['mean']:+.2f} ± {vd['vs_brainless']['sem']:.3f} {vd['brainless_verdict']}"
                 f"{' (ceiling)' if vd['ceiling'] else ''} |")
    L += ["", "## top k (training seeds 0-9) — brief §5.2", "",
          "| row | s | obstacles (open) | acted | correct first | lead median [q25, q75] ms | any-first lead median | "
          "direction correct | away ticks median (>= 12) | pass / crash | GF L / R per obstacle | actions/s on screen | acted vs noise |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for row, t in r["hypothesis"]["top_k_table"].items():
        L.append(_lead_line(row, r["rows"][row]["s"], t))
    L += ["", f"P1 {r['hypothesis']['P1']} -> {r['hypothesis']['P1_holds']}; P2 {r['hypothesis']['P2']} -> "
              f"{r['hypothesis']['P2_holds']}; reading: {r['hypothesis']['reading']['row']}: "
              f"{r['hypothesis']['reading']['sentence']}", ""]
    L += ["## every k (training seeds 0-9) — appendix", ""]
    for row, v in r["rows"].items():
        L += [f"### {row}", "", "| # | k | k×p99 | train mean | obstacles | acted | correct | lead median ms | "
              "direction | away median | GF L/R | acted vs noise |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for t in v["obstacles_by_k"]:
            L.append(f"| {t['k_index']} | {t['k']:.4g} | {t['k_times_p99']:.4g} | {t['train_mean']:.1f} | {t['n']} ({t['open']}) | "
                     f"{t['acted']} | {t['correct_first']} | {_f(t['lead_ms_median'])} | {_f(t['direction_correct_fraction'], 2)} | "
                     f"{_f(t.get('away_ticks_median'))} | {_f(t['gf_per_obstacle']['L'], 2)}/{_f(t['gf_per_obstacle']['R'], 2)} | "
                     f"{_f(t.get('acted_fraction'), 2)} vs {_f(t.get('acted_expected_from_noise'), 2)} |")
        L.append("")
    L += ["## p99 and s", "", "| row | s | p99 | p99 × s | distinct non-zero loom values |", "|---|---|---|---|---|"]
    for row, p in r["p99_vs_s"].items():
        L.append(f"| {row} | {p['s']} | {p['p99']:.5g} | {p['p99_times_s']:.5g} | {p['distinct_nonzero_values']} |")
    L += ["", "## appearance split (correct first actions within 300 ms of the first frame) — descriptive", "",
          "| row | where | correct first | within / after | lead median within / after | passed | passed with a correct first action within 300 ms |",
          "|---|---|---|---|---|---|---|"]
    for row, v in r["rows"].items():
        for where, a in v["appearance"].items():
            L.append(_appearance_line(row, where, a))
    L += _posthoc_tables(r["posthoc"])
    return "\n".join(L) + "\n"


def _appearance_line(row: str, where: str, a: dict) -> str:
    return (f"| {row} | {where} | {a['correct_first']} | {a['within']} / {a['after']} | "
            f"{_f(a['lead_ms_median_within'])} / {_f(a['lead_ms_median_after'])} | {a['passed']} | "
            f"{a['passed_within']} ({_f(a['passed_within_fraction'], 2)}) |")


def _posthoc_tables(ph: dict) -> list[str]:
    L = ["", "## post-hoc cells (brief 추가 등록; training seeds 0-9)", "",
         "| row | cell | game | k | mean | random (train) | vs random | lead median | away median (>= 12) | passed | appearance: passed within 300 ms |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for row, x in ph["rows"].items():
        cells = [(f"A #{p['k_index']}", row, p) for p in x["cell_A"]["points"]]
        cells += [("B", x["cell_B"]["game"], x["cell_B"]), (f"top #{x['top_k']['k_index']}", row, x["top_k"])]
        for name, game, c in cells:
            v = c["vs_random"]
            word = "beat" if v["beats"] else ("lost" if v["loses"] else "not")
            away = f"{_f(c['away_ticks_median'])} ({c['away_ge_step_aside']})" if c["away_ticks_median"] is not None else "—"
            ap = c["appearance"]
            L.append(f"| {row} | {name} | {game} | {c['k']:.4g} | {c['mean']:.1f} | {np.mean(c['random_train']):.1f} | "
                     f"{v['mean']:+.2f} ± {v['sem']:.3f} {word} | {_f(c['lead_ms_median'])} | {away} | {c['passed']} | "
                     f"{ap['passed_within']} / {ap['passed']} ({_f(ap['passed_within_fraction'], 2)}) |")
    L += ["", "| row | k_match | cell A | cell B | reading (2x2) | drive at top k vs s = 1 | appearance current at top k vs s = 1 |",
          "|---|---|---|---|---|---|---|"]
    for row, x in ph["rows"].items():
        rd = x["reading"]
        L.append(f"| {row} | {x['k_match']:.4g} | {rd['A']} | {rd['B']} | {rd['sentence']} | "
                 f"{x['drive_ratio_top']:.2f}x | {x['appearance_current_ratio_top']:.1f}x |")
    L += ["", "## loom by obstacle distance: slow mean x s / s = 1 mean (brainless-bot frames, training seeds)", ""]
    for kk, label in (("all", "brainless bot, all obstacles"), ("level0", "brainless bot, first speed level (first 4 obstacles)"),
                      ("noop", "always-NOOP bot (never moves; its one obstacle)")):
        rows = list(ph["loom_by_distance"])
        bins = [b["bin"] for b in ph["loom_by_distance"][rows[0]][kk]]
        L += [f"### {label}", "", "| distance | " + " | ".join(rows) + " |", "|---|" + "---|" * len(rows)]
        for i, b in enumerate(bins):
            L.append(f"| {b} | " + " | ".join(_f(ph["loom_by_distance"][r][kk][i]["slow_times_s_over_s1"], 2) for r in rows) + " |")
        L.append("")
    return L


def _lead_line(row: str, s: int, t: dict) -> str:
    away = f"{_f(t.get('away_ticks_median'))} ({t.get('away_ge_step_aside', '—')})" if "away_ticks_median" in t else "—"
    return (f"| {row} | {s} | {t['n']} ({t['open']}) | {t['acted']} | {t['correct_first']} | "
            f"{_f(t['lead_ms_median'])} [{_f(t['lead_ms_q25'])}, {_f(t['lead_ms_q75'])}] | {_f(t['lead_ms_any_first_median'])} | "
            f"{_f(t['direction_correct_fraction'], 2)} | {away} | {t['passed']} / {t['crashed']} | "
            f"{_f(t['gf_per_obstacle']['L'], 2)} / {_f(t['gf_per_obstacle']['R'], 2)} | {_f(t['actions_per_s_on_screen'], 2)} | "
            f"{_f(t.get('acted_fraction'), 2)} vs {_f(t.get('acted_expected_from_noise'), 2)} |")


# ---------------------------------------------------------------------------------------------
# docs/m7g-lead.png — no plotting library: a 5x7 bitmap font and numpy lines
# ---------------------------------------------------------------------------------------------
_FONT = {
    "A": "01110100011000111111100011000110001", "B": "11110100011000111110100011000111110",
    "C": "01110100011000010000100001000101110", "D": "11110100011000110001100011000111110",
    "E": "11111100001000011110100001000011111", "F": "11111100001000011110100001000010000",
    "G": "01110100011000010111100011000101111", "H": "10001100011000111111100011000110001",
    "I": "01110001000010000100001000010001110", "J": "00111000100001000010000101001001100",
    "K": "10001100101010011000101001001010001", "L": "10000100001000010000100001000011111",
    "M": "10001110111010110101100011000110001", "N": "10001100011100110101100111000110001",
    "O": "01110100011000110001100011000101110", "P": "11110100011000111110100001000010000",
    "Q": "01110100011000110001101011001001101", "R": "11110100011000111110101001001010001",
    "S": "01111100001000001110000010000111110", "T": "11111001000010000100001000010000100",
    "U": "10001100011000110001100011000101110", "V": "10001100011000110001100010101000100",
    "W": "10001100011000110101101011010101010", "X": "10001100010101000100010101000110001",
    "Y": "10001100010101000100001000010000100", "Z": "11111000010001000100010001000011111",
    "0": "01110100011001110101110011000101110", "1": "00100011000010000100001000010001110",
    "2": "01110100010000100010001000100011111", "3": "11111000100010000010000011000101110",
    "4": "00010001100101010010111110001000010", "5": "11111100001111000001000011000101110",
    "6": "00110010001000011110100011000101110", "7": "11111000010001000100010000100001000",
    "8": "01110100011000101110100011000101110", "9": "01110100011000101111000010001001100",
    " ": "0" * 35, "-": "00000000000000011111000000000000000", ".": "00000000000000000000000000110001100",
    "=": "00000000001111100000111110000000000", "(": "00010001000100001000010000010000010",
    ")": "01000001000001000010000100010001000", "/": "00001000010001000100010001000010000",
    ",": "00000000000000000000011000010001000", ":": "00000011000110000000011000110000000",
    "+": "00000001000010011111001000010000000", "%": "11000110010001000100010001001100011",
    "[": "01110010000100001000010000100001110", "]": "01110000100001000010000100001001110",
}


def draw_text(img: np.ndarray, text: str, x: int, y: int, scale: int = 2, ink: int = 0) -> int:
    """Draw ``text`` (upper case, digits and a few signs) with its top-left at (x, y); returns the end x."""
    for ch in text.upper():
        bits = np.array([c == "1" for c in _FONT[ch]], dtype=bool).reshape(7, 5)
        g = np.repeat(np.repeat(bits, scale, 0), scale, 1)
        h, w = g.shape
        y0, x0 = max(y, 0), max(x, 0)
        y1, x1 = min(y + h, img.shape[0]), min(x + w, img.shape[1])
        if y1 > y0 and x1 > x0:
            region = img[y0:y1, x0:x1]
            region[g[y0 - y:y1 - y, x0 - x:x1 - x]] = ink
        x += 6 * scale
    return x


def text_width(text: str, scale: int = 2) -> int:
    return 6 * scale * len(text) - scale


def draw_line(img: np.ndarray, x0: float, y0: float, x1: float, y1: float, ink: int = 0, width: int = 3,
              dash: int = 0) -> None:
    n = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
    r = width // 2
    for i in range(n):
        if dash and (i // dash) % 2:
            continue
        f = i / max(n - 1, 1)
        x, y = int(round(x0 + f * (x1 - x0))), int(round(y0 + f * (y1 - y0)))
        img[max(y - r, 0):y + r + 1, max(x - r, 0):x + r + 1] = ink


def appearance_lead_ms(s: int) -> float:
    """Time from a first-speed-level obstacle's first frame to its contact: the longest lead possible."""
    from . import dodge as D
    return (D.D_SPAWN - D.D_HIT) / (D.SPEED0 / s) * 1000.0 / 60.0


def draw_figure(r: dict, path: Path) -> np.ndarray:
    """x: s (log 2), y: lead of the correct first action (log), median and [q25, q75] at the top k,
    one line per family; the 200 ms needed to step aside; and the appearance line (the lead an
    action on the obstacle's first frame has)."""
    W, H = 960, 580
    img = np.full((H, W), 255, dtype=np.uint8)
    left, right, top, bottom = 120, 900, 92, 500
    fams = {"dodge_v2": ("DODGE V2", 0, 0), "jump": ("JUMP", 110, 9)}
    tab = r["hypothesis"]["top_k_table"]
    ticks = (50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000)
    vals = [v for row in ROWS for v in (tab[row]["lead_ms_q75"],) if v is not None] + [appearance_lead_ms(8)]
    ymax = next(t for t in ticks if t >= max(vals) * 1.05)
    ymin = 50
    sx = lambda s: left + (right - left) * (math.log2(s) / 3.0)
    sy = lambda v: bottom - (bottom - top) * (math.log(max(v, ymin) / ymin) / math.log(ymax / ymin))
    draw_text(img, "CORRECT FIRST ACTION - LEAD BEFORE CONTACT", left, 14, 3)
    draw_text(img, "MEDIAN AND [Q25, Q75], TOP K = 5120/P99, TRAINING SEEDS 0-9", left, 46, 2, 90)
    draw_text(img, "LEAD (MS, LOG)", left - 100, top - 26, 2)
    draw_line(img, left, bottom, right, bottom, 0, 2)
    draw_line(img, left, top, left, bottom, 0, 2)
    for v in (t for t in ticks if ymin <= t <= ymax):
        y = sy(v)
        draw_line(img, left - 8, y, left, y, 0, 2)
        lab = f"{v}"
        draw_text(img, lab, left - 14 - text_width(lab), int(y) - 7, 2)
    for s in (1, 2, 4, 8):
        x = sx(s)
        draw_line(img, x, bottom, x, bottom + 8, 0, 2)
        draw_text(img, f"{s}", int(x) - 5, bottom + 14, 2)
    draw_text(img, "S = HOW MANY TIMES SLOWER THE OBSTACLE STARTS", left + 170, bottom + 44, 2)
    y200 = sy(STEP_ASIDE_MS)
    draw_line(img, left, y200, right, y200, 150, 2, dash=8)
    draw_text(img, "200 MS: STEP ASIDE (12 TICKS)", right - text_width("200 MS: STEP ASIDE (12 TICKS)") - 4,
              int(y200) + 8, 2, 120)
    app = [(sx(s), sy(appearance_lead_ms(s))) for s in (1, 2, 4, 8)]
    for a, b in zip(app, app[1:]):
        draw_line(img, a[0], a[1], b[0], b[1], 190, 2, dash=4)
    for fam, (label, ink, dash) in fams.items():
        pts = []
        for row in FAMILIES[fam]:
            t, s = tab[row], speed_divisor(row)
            x = sx(s) + (-6 if fam == "dodge_v2" else 6)
            if t["lead_ms_median"] is None:
                draw_text(img, "NONE", int(x) - 22, bottom - 24 - (20 if fam == "jump" else 0), 2, ink)
                pts.append(None)
                continue
            y = sy(t["lead_ms_median"])
            draw_line(img, x, sy(t["lead_ms_q25"]), x, sy(t["lead_ms_q75"]), ink, 2)
            img[int(y) - 5:int(y) + 6, int(x) - 5:int(x) + 6] = ink
            lab = f"N={t['correct_first']}"
            draw_text(img, lab, int(x) + 10, int(y) + (6 if fam == "dodge_v2" else -24), 2, ink)
            pts.append((x, y))
        for a, b in zip(pts, pts[1:]):
            if a and b:
                draw_line(img, a[0], a[1], b[0], b[1], ink, 3, dash=dash)
    ly, lx = int(sy(1000)), right - 380
    legend = [(label, ink, dash, 3) for label, ink, dash in fams.values()]
    legend.append(("APPEARANCE (FIRST FRAME)", 190, 4, 2))
    for label, ink, dash, wdt in legend:
        draw_line(img, lx, ly + 7, lx + 50, ly + 7, ink, wdt, dash=dash)
        draw_text(img, label, lx + 60, ly, 2, 0 if ink == 190 else ink)
        ly += 26
    media.write_png(path, img)
    return img


# ---------------------------------------------------------------------------------------------
def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="M7g: slow approach (docs/m7g-brief.md)")
    ap.add_argument("--stage", default="all", choices=("all", "bots", "controls", "fly", "posthoc", "assemble"))
    ap.add_argument("--variants", default=",".join(ROWS), help="rows for the bots / fly stages")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--parts", default=str(PARTS_DIR))
    ap.add_argument("--no-bundle", action="store_true")
    args = ap.parse_args(argv)
    rows = [r for r in args.variants.split(",") if r]
    unknown = [r for r in rows if r not in ROWS]
    if unknown:
        raise SystemExit(f"unknown row(s) {unknown}; the brief's list is {list(ROWS)}")
    parts = Path(args.parts).resolve()
    t0 = time.time()
    if args.stage in ("all", "bots"):
        stage_bots(rows, parts, args.workers)
    if args.stage in ("all", "controls"):
        stage_controls(parts, args.device)
    if args.stage in ("all", "fly"):
        stage_fly(rows, parts, args.device, bundles=not args.no_bundle)
    if args.stage in ("all", "posthoc"):
        stage_posthoc(parts, args.device)
    if args.stage in ("all", "assemble"):
        stage_assemble(parts)
    log(f"stage {args.stage} done in {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
