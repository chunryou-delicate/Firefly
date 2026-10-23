"""M7c: the first match between the fly brain and the two bots (docs/m7c-brief.md).

    python -m flysim.apps.games.m7c            # selection, evaluation, verdicts, replay bundles

Everything that decides the outcome is fixed here before the first fly run (brief, plus the
addendum of the brief, recorded in ``P99_FRAMES``, ``ONE_WAY_GAMES`` and ``CEILING_GAMES``):

* games: jump, dodge (v1), dodge_v2. Agents: random, brainless (M7b θ per game), fly.
* fly = ``BrainAgent(FlyBrain(k), GiantFibreDecoder)``. ``k`` is its only free parameter, chosen
  per game exactly as θ was: a 21-point log grid on [1/p99, 5120/p99], every point played on
  training seeds 0-9, best mean wins (ties: grid point nearest the geometric middle), then
  scored on evaluation seeds 10-19. Evaluation seeds never touch the selection.
* p99: the 99th percentile of the non-zero looming values (both halves) over the frames named by
  ``P99_FRAMES`` on training seeds 0-9, per game, no brain involved.
* verdict per game on evaluation seeds, paired by seed: mean(difference) > 0 and > 3 SEM —
  against the random bot, the brainless bot, and (addendum) the one-direction bot: always-LEFT
  and always-RIGHT both played on the evaluation seeds, the one with the higher mean is the
  reference. On dodge_v2 the brainless comparison can only read "lost" or "indistinguishable":
  that game has a physical ceiling the brainless bot already sits on (addendum).
* ignition (M7a's three numbers carried to a game, flagged — never used to drop a run): see
  ``ignition`` — the addendum's literal per-window reading and a like-for-like reading (means,
  start-up window dropped) are both computed and reported; the tables use like-for-like.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from ...data.download import ROOT
from ...graph import Graph
from ...probe.export import neuron_coords, write_run_json
from ...probe.probe import bin_spikes
from ...probe.sets import build_probe_sets, load_roi
from ...sensory.looming import LoomingDetector
from . import dodge as D
from . import jump as J
from .agents import BrainAgent, RandomAgent, ThresholdAgent
from .base import Action, make
from .brain import (DECODER_WINDOW_MS, EscapeSets, FlyBrain, make_decoder, tick_end_ms)
from .runner import EVAL_SEEDS, TRAIN_SEEDS, run_many

GAMES = ("jump", "dodge", "dodge_v2")
K_POINTS = 21
K_SPAN = (1.0, 5120.0)            # M7a's gain grid, carried into detector units by p99
P99_FRAMES = "brainless"          # addendum 2: frames the brainless bot (fixed M7b θ) plays on seeds 0-9
ONE_WAY_GAMES = ("jump", "dodge", "dodge_v2")    # addendum 1: one-direction bot judged in all three
CEILING_GAMES = {"dodge_v2": 72}  # addendum 1: physical ceiling; "beat the brainless bot" impossible
SEM_MULT = 3.0
CONTROL_MS = 5000                 # no-input control per seed, for the ignition check
RUNAWAY_RATE_HZ = 5.0             # M7a
FAILURE_ACTIVE_FRACTION = 0.90    # CLAUDE.md §3.3
EXCESS_MIN_FRACTION = 0.01        # M7a / M3
BUNDLE_SEED = 10
BUNDLE_MAX_MS = 60_000            # replay bundle keeps at most the first 60 s
BASELINES = ROOT / "data-provenance" / "m7b-baselines.json"
RESULTS_JSON = ROOT / "data-provenance" / "m7c-results.json"
BUNDLE_DIR = ROOT / "runs" / "m7c"


def log(msg: str) -> None:
    print(f"[m7c] {msg}", flush=True)


# ---------------------------------------------------------------------------------------------
# grid
# ---------------------------------------------------------------------------------------------
def loom_values(game: str, policy: str, seeds=TRAIN_SEEDS, theta: float | None = None) -> np.ndarray:
    """Non-zero looming values (both halves) over every frame of ``seeds``, no brain."""
    vals = []
    for seed in seeds:
        env, det = make(game), LoomingDetector()
        frame = env.reset(seed)
        bot = ThresholdAgent(theta) if policy == "brainless" else None
        if bot is not None:
            bot.reset(seed)
        done = False
        while not done:
            left, right = det.update(frame)
            if left > 0:
                vals.append(left)
            if right > 0:
                vals.append(right)
            action = bot.act(frame) if bot is not None else int(Action.NOOP)
            frame, _, done, _ = env.step(action)
    return np.asarray(vals, dtype=np.float64)


def k_grid(p99: float, n: int = K_POINTS, span=K_SPAN) -> tuple[float, ...]:
    if not p99 > 0:
        raise ValueError("p99 must be > 0")
    return tuple(float(x) for x in np.geomspace(span[0] / p99, span[1] / p99, n))


def select_best(rows: dict[float, dict]) -> dict:
    """Highest mean wins; ties -> the grid point nearest the geometric middle of the tie."""
    grid = sorted(rows)
    means = np.array([rows[k]["mean"] for k in grid])
    best = means.max()
    tied = [k for k, m in zip(grid, means) if m == best]
    mid = float(np.exp((np.log(tied[0]) + np.log(tied[-1])) / 2))
    chosen = min(tied, key=lambda k: abs(np.log(k) - np.log(mid)))
    return {"chosen": chosen, "best_mean": float(best), "tied": tied}


# ---------------------------------------------------------------------------------------------
# verdicts and ignition (pure)
# ---------------------------------------------------------------------------------------------
def _sem(x: np.ndarray) -> float:
    return float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0


def paired(fly: list[int], other: list[int]) -> dict:
    d = np.asarray(fly, dtype=float) - np.asarray(other, dtype=float)
    mean, sem = float(d.mean()), _sem(d)
    return {"diffs": d.tolist(), "mean": mean, "sem": sem,
            "beats": bool(mean > 0 and mean > SEM_MULT * sem),
            "loses": bool(mean < 0 and -mean > SEM_MULT * sem)}


def one_way_reference(left: list[int], right: list[int]) -> dict:
    """The one-direction bot: the higher-mean of always-LEFT / always-RIGHT (tie -> LEFT)."""
    use_right = float(np.mean(right)) > float(np.mean(left))
    return {"always_left": list(left), "always_right": list(right),
            "reference": "always_right" if use_right else "always_left",
            "scores": list(right if use_right else left),
            "mean": float(np.mean(right if use_right else left))}


def verdict(fly: list[int], random: list[int], brainless: list[int], one_way: list[int],
            game: str = "") -> dict:
    vr, vb, vo = paired(fly, random), paired(fly, brainless), paired(fly, one_way)
    if vb["beats"] and game not in CEILING_GAMES:
        vs_b = "beat the brainless bot"
    elif vb["loses"]:
        vs_b = "lost to the brainless bot"
    else:
        vs_b = "indistinguishable from the brainless bot"
    return {"vs_random": vr, "vs_brainless": vb, "vs_one_way": vo,
            "beat_random": vr["beats"], "beat_one_way": vo["beats"], "brainless_verdict": vs_b,
            "ceiling": CEILING_GAMES.get(game)}


STARTUP_WINDOWS = 1                # the first 200 ms after reset is a start-up transient (~14 % active
                                   # vs ~6 % background) in the control as much as in a run


def ignition(run: dict, control: dict) -> dict:
    """Two readings of the addendum's rule, both reported; neither drops a run.

    * ``literal`` — as written: every 200 ms window of the run against the control windows' mean
      (excess if a window is >= 1 point above it and > 3 SEM). Found on a dry run before the real
      one: this flags essentially every run, because the start-up window (in the control too) and
      ordinary background fluctuation (windows 3-9 %) exceed mean + 1 point. Kept for the record.
    * ``like_for_like`` — the run's mean window active fraction against the control's mean, the
      start-up window dropped from both; M7a compared means the same way.
    Runaway (both readings): a run window >= 5 Hz or >= 90 % active (per window, as written), or
    the control mean >= 5 Hz. Viewer failure: no spike at all, or a window >= 90 % active.
    """
    cw = np.array(control["windows"]) if control["windows"] else np.zeros((0, 2))
    rw = np.array(run["windows"]) if run["windows"] else np.zeros((0, 2))
    c_rate = float(cw[:, 0].mean()) if len(cw) else 0.0
    c_act, c_sem = (float(cw[:, 1].mean()), _sem(cw[:, 1])) if len(cw) else (0.0, 0.0)
    c_tail = cw[STARTUP_WINDOWS:, 1] if len(cw) > STARTUP_WINDOWS else cw[:, 1]
    r_tail = rw[STARTUP_WINDOWS:, 1] if len(rw) > STARTUP_WINDOWS else rw[:, 1]
    c_act2 = float(c_tail.mean()) if len(c_tail) else 0.0
    c_sem2 = _sem(c_tail)
    r_act2 = float(r_tail.mean()) if len(r_tail) else 0.0
    r_rate_max = float(rw[:, 0].max()) if len(rw) else 0.0
    r_act_max = float(rw[:, 1].max()) if len(rw) else 0.0
    control_runaway = c_rate >= RUNAWAY_RATE_HZ
    window_runaway = r_rate_max >= RUNAWAY_RATE_HZ or r_act_max >= FAILURE_ACTIVE_FRACTION
    ex_lit = r_act_max - c_act
    lit = ex_lit >= EXCESS_MIN_FRACTION and ex_lit > SEM_MULT * c_sem
    ex_lfl = r_act2 - c_act2
    lfl = len(r_tail) > 0 and ex_lfl >= EXCESS_MIN_FRACTION and ex_lfl > SEM_MULT * c_sem2
    return {"control_rate_hz": c_rate, "run_window_rate_max": r_rate_max, "run_window_active_max": r_act_max,
            "control_runaway": bool(control_runaway), "window_runaway": bool(window_runaway),
            "literal": {"control_window_active_mean": c_act, "control_sem": c_sem, "excess": ex_lit,
                        "excess_ignited": bool(lit),
                        "ignited": bool(control_runaway or window_runaway or lit)},
            "like_for_like": {"control_window_active_mean": c_act2, "control_sem": c_sem2,
                              "run_window_active_mean": r_act2, "excess": ex_lfl,
                              "excess_ignited": bool(lfl),
                              "ignited": bool(control_runaway or window_runaway or lfl)},
            "ignited": bool(control_runaway or window_runaway or lfl),     # the reading used in the tables
            "viewer_failure": bool(run["n_spikes"] == 0 or r_act_max >= FAILURE_ACTIVE_FRACTION)}


# ---------------------------------------------------------------------------------------------
# one fly episode, with harness-side logging (the agent itself sees pixels only)
# ---------------------------------------------------------------------------------------------
def play_fly(game: str, agent: BrainAgent, brain: FlyBrain, seed: int, detail: bool = False,
             record: bool = False) -> dict:
    env = make(game)
    frame = env.reset(seed)
    agent.reset(seed)                                # resets the brain with the game seed
    events: list[dict] = []
    obstacles: list[dict] = []
    cur = None
    frames, t_frames, actions, looms = [], [], [], []
    done = False
    info = {"score": 0, "tick": 0, "crashed": False, "truncated": False}
    while not done:
        t_start = tick_end_ms(env.tick - 1) if env.tick > 0 else 0
        if record and t_start < BUNDLE_MAX_MS:
            frames.append(frame)
            t_frames.append(float(t_start))
        a = agent.act(frame)
        if record and t_start < BUNDLE_MAX_MS:
            actions.append(a)
            looms.append(agent.last_signal)
        present_before = env.dist is not None
        if present_before and cur is None:
            cur = {"index": len(obstacles), "t_spawn": float(t_start), "first_action_ms": None,
                   "moves_away": 0, "moves_toward": 0, "jumps": 0,
                   "side": None if game == "jump" else ("right" if env.obj_x > env.x else "left")}
        jumped_before = getattr(env, "airborne", False)
        frame, r, done, info = env.step(a)
        t_end = float(tick_end_ms(env.tick - 1))
        act = Action(a)
        if game == "jump":
            if act == Action.JUMP and env.airborne and env.jump_t == 0 and not jumped_before:
                events.append({"t_ms": t_end, "type": "jump"})
                if cur is not None:
                    cur["jumps"] += 1
                    cur["first_action_ms"] = cur["first_action_ms"] or t_end
        elif act in (Action.LEFT, Action.RIGHT):
            events.append({"t_ms": t_end, "type": "move_left" if act == Action.LEFT else "move_right"})
            if cur is not None:
                away = (act == Action.LEFT) == (cur["side"] == "right")
                cur["moves_away" if away else "moves_toward"] += 1
                cur["first_action_ms"] = cur["first_action_ms"] or t_end
        if r > 0 or info["crashed"]:
            kind = "crash" if info["crashed"] else "pass"
            events.append({"t_ms": t_end, "type": kind, "obstacle": cur["index"] if cur else None})
            if cur is not None:
                cur["t_contact"] = t_end
                cur["outcome"] = kind
                cur["gf_spikes"] = sum(1 for t, _ in brain.gf_events if cur["t_spawn"] <= t < t_end)
                cur["lead_ms"] = (t_end - cur["first_action_ms"]) if cur["first_action_ms"] else None
                if kind == "crash":
                    cur["cause"] = loss_cause(game, env, cur)
                obstacles.append(cur)
                cur = None
    summary = brain.summary()
    out = {"seed": seed, "score": int(info["score"]), "ticks": int(info["tick"]),
           "crashed": bool(info["crashed"]), "truncated": bool(info["truncated"]),
           "network_rate_hz": summary["network_rate_hz"], "active_fraction": summary["active_fraction"],
           "gf_spikes": summary["gf_spikes"], "n_spikes": summary["n_spikes"],
           "windows": summary["windows"]}
    if detail:
        out["obstacles"] = obstacles
        out["loss_cause"] = obstacles[-1].get("cause") if obstacles and obstacles[-1]["outcome"] == "crash" else None
        out["input_cell_spikes"] = summary["input_cell_spikes"]
    if record:
        out["record"] = {"frames": frames, "t_ms": t_frames, "actions": actions, "loom": looms,
                         "events": events}
    return out


def loss_cause(game: str, env, ob: dict) -> str:
    """Why the fatal obstacle hit (descriptive; brief's classes plus a sub-reason)."""
    if game == "jump":
        if ob["jumps"] == 0:
            return "other:no_reaction" if env.cooldown == 0 else "other:cooldown_from_earlier_jump"
        if env.airborne and env.jump_t < J.JUMP_TICKS / 2:
            return "late"
        return "other:early_jump"
    if game == "dodge" and abs(abs(env.x) - D.X_LIMIT) < 1e-9:
        return "wall"
    if ob["moves_away"] == 0 and ob["moves_toward"] == 0:
        return "other:no_reaction"
    if ob["moves_toward"] > ob["moves_away"]:
        return "wrong_direction"
    return "late"


def control_run(brain: FlyBrain, seed: int) -> dict:
    """The brain with no input, same seed, CONTROL_MS long."""
    brain.reset(seed)
    while brain.t_ms < CONTROL_MS:
        brain.step(0.0, 0.0)
    return brain.summary()


# ---------------------------------------------------------------------------------------------
def write_bundle(game: str, k: float, ep: dict, ctx: dict, brain: FlyBrain, out_dir: Path) -> dict:
    """Replay bundle for the split screen (docs/m7-game-ladder.md): frames.npz, run.json, game.json."""
    rec = ep.pop("record")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "frames.npz", frames=np.stack(rec["frames"]).astype(np.uint8),
                        t_ms=np.asarray(rec["t_ms"], dtype=np.float64),
                        actions=np.asarray(rec["actions"], dtype=np.int8),
                        loom=np.asarray(rec["loom"], dtype=np.float32))
    t, idx = brain.spikes()
    dur = min(int(brain.t_ms), BUNDLE_MAX_MS)
    keep = t < dur
    t, idx = t[keep], idx[keep]
    sets = ctx["escape"].probe_sets
    rt = bin_spikes(t, idx, 1.0, 1.0, dur, sets)
    write_run_json(out_dir / "run.json", run_id=f"m7c-{game}-brain-seed{BUNDLE_SEED}", sensory_mode="rate",
                   dt_ms=1.0, rt=rt, sets=sets, coords=ctx["coords"],
                   input_label=f"{game}: looming (left/right) x k = {k:.4g}",
                   envelope=_envelope(rec, dur), spikes=(rt.t_bin, idx),
                   engine_params=brain.params.to_dict(), adapter={"adapter": "LoomingDetector -> LC4+LPLC2",
                                                                  "k": k, **LoomingDetector.describe()},
                   stimulus={"kind": "game", "game": game, "seed": BUNDLE_SEED},
                   extra_meta={"probe_profile": "escape", "bundle_ms": dur,
                               "game_ms": int(brain.t_ms), "truncated_to_bundle_max": brain.t_ms > BUNDLE_MAX_MS,
                               "sensory_mode_note": "game looming signal, not the JO adapter"})
    game_doc = {"game": game, "seed": BUNDLE_SEED, "score": ep["score"], "ticks": ep["ticks"],
                "duration_ms": int(brain.t_ms), "bundle_ms": dur,
                "agent": {"kind": "fly brain", "k": k, "decoder": make_decoder().describe(),
                          "brain": brain.describe()},
                "events": [e for e in rec["events"] if e["t_ms"] < dur],
                "rules": make(game).describe(), "tick_ms": "floor((k+1)*1000/60)"}
    (out_dir / "game.json").write_text(json.dumps(game_doc, indent=1) + "\n")
    return {"dir": str(out_dir.relative_to(ROOT)), "frames": len(rec["frames"]), "bundle_ms": dur,
            "game_ms": int(brain.t_ms), "spikes_in_bundle": int(len(t))}


def _envelope(rec: dict, dur: int) -> np.ndarray:
    env = np.zeros(dur, dtype=np.float64)
    for t, (l, r) in zip(rec["t_ms"], rec["loom"]):
        a = int(t)
        if a < dur:
            env[a:min(dur, a + 17)] = max(l, r)
    m = env.max()
    return env / m if m > 0 else env


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return None


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="M7c: fly brain vs the two bots")
    ap.add_argument("--games", default=",".join(GAMES))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-bundle", action="store_true")
    args = ap.parse_args(argv)
    games = [g for g in args.games.split(",") if g]
    t_all = time.time()
    base = json.loads(BASELINES.read_text())
    graph = Graph.load()
    roi = load_roi(graph)
    escape = EscapeSets.from_probe_sets(build_probe_sets(graph, roi, profile="escape"))
    ctx = {"escape": escape, "coords": neuron_coords(graph, roi)}
    brain = FlyBrain(graph, escape, k=1.0, device=args.device, keep_spikes=False)
    agent = BrainAgent(brain, make_decoder(), brain_reset=brain.reset)

    log("controls (no input, 5 s per seed) ...")
    controls = {s: control_run(brain, s) for s in TRAIN_SEEDS + EVAL_SEEDS}
    result = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "git_commit": git_commit(),
              "train_seeds": list(TRAIN_SEEDS), "eval_seeds": list(EVAL_SEEDS),
              "brain": brain.describe() | {"k": "per game, see below"},
              "decoder": make_decoder().describe(), "detector": LoomingDetector.describe(),
              "rules": {"k_points": K_POINTS, "k_span": list(K_SPAN), "p99_frames": P99_FRAMES,
                        "sem_mult": SEM_MULT, "control_ms": CONTROL_MS,
                        "runaway_rate_hz": RUNAWAY_RATE_HZ, "failure_active_fraction": FAILURE_ACTIVE_FRACTION,
                        "excess_min_fraction": EXCESS_MIN_FRACTION, "one_way_games": list(ONE_WAY_GAMES),
                        "ceiling_games": CEILING_GAMES},
              "controls": {str(s): {k: v for k, v in c.items() if k != "windows"} for s, c in controls.items()},
              "games": {}}
    for game in games:
        tg = time.time()
        theta = base["games"][game]["threshold"]["theta"]
        vals = loom_values(game, P99_FRAMES, theta=theta)
        p99 = float(np.percentile(vals, 99))
        grid = k_grid(p99)
        log(f"{game}: p99 {p99:.5f} ({len(vals)} non-zero values) -> k {grid[0]:.4g} .. {grid[-1]:.4g}")
        rows = {}
        for k in grid:
            brain.k = k
            eps = [play_fly(game, agent, brain, s) for s in TRAIN_SEEDS]
            for e in eps:
                e["ignition"] = ignition(e, controls[e["seed"]])
                e.pop("windows")
            sc = np.array([e["score"] for e in eps])
            rows[k] = {"k": k, "mean": float(sc.mean()), "min": int(sc.min()), "max": int(sc.max()),
                       "scores": sc.tolist(), "ignited": int(sum(e["ignition"]["ignited"] for e in eps)),
                       "ignited_literal": int(sum(e["ignition"]["literal"]["ignited"] for e in eps)),
                       "ticks": [e["ticks"] for e in eps],
                       "gf_spikes": [e["gf_spikes"] for e in eps],
                       "network_rate_hz": [round(e["network_rate_hz"], 4) for e in eps]}
            log(f"  k={k:9.4g}: train mean {rows[k]['mean']:6.2f}  scores {rows[k]['scores']}  "
                f"ignited {rows[k]['ignited']}  ({time.time() - tg:.0f} s)")
        sel = select_best(rows)
        k = sel["chosen"]
        brain.k = k
        fly_eps = []
        for s in EVAL_SEEDS:
            e = play_fly(game, agent, brain, s, detail=True)
            e["ignition"] = ignition(e, controls[s])
            e.pop("windows")
            fly_eps.append(e)
        fly = [e["score"] for e in fly_eps]
        rnd = run_many(game, lambda env: RandomAgent(), EVAL_SEEDS)
        thr = run_many(game, lambda env: ThresholdAgent(theta), EVAL_SEEDS)
        b_rnd = dict(zip(base["games"][game]["random"]["seeds"], base["games"][game]["random"]["scores"]))
        b_thr = dict(zip(base["games"][game]["threshold"]["eval"]["seeds"],
                         base["games"][game]["threshold"]["eval"]["scores"]))
        if rnd["scores"] != [b_rnd[s] for s in EVAL_SEEDS] or thr["scores"] != [b_thr[s] for s in EVAL_SEEDS]:
            raise RuntimeError(f"{game}: baseline bots no longer reproduce m7b-baselines.json")
        row = {"p99": p99, "p99_n_values": int(len(vals)), "k_grid": list(grid),
               "selection": sel | {"grid": [rows[g] for g in grid]},
               "k": k, "theta": theta,
               "fly": {"scores": fly, "mean": float(np.mean(fly)), "min": int(min(fly)), "max": int(max(fly)),
                       "episodes": fly_eps},
               "random": {"scores": rnd["scores"], "mean": rnd["mean"]},
               "brainless": {"scores": thr["scores"], "mean": thr["mean"]}}
        ow = one_way_reference(run_many(game, lambda env: _Const(Action.LEFT), EVAL_SEEDS)["scores"],
                               run_many(game, lambda env: _Const(Action.RIGHT), EVAL_SEEDS)["scores"])
        row["one_way"] = ow
        row["verdict"] = verdict(fly, rnd["scores"], thr["scores"], ow["scores"], game)
        if not args.no_bundle:
            brain.keep_spikes = True
            ep = play_fly(game, agent, brain, BUNDLE_SEED, detail=True, record=True)
            brain.keep_spikes = False
            if ep["score"] != fly[EVAL_SEEDS.index(BUNDLE_SEED)]:
                raise RuntimeError("bundle replay differs from the evaluation run (determinism)")
            row["bundle"] = write_bundle(game, k, ep, ctx, brain, BUNDLE_DIR / f"{game}-brain-seed{BUNDLE_SEED}")
            brain.spikes_t, brain.spikes_i = [], []
        result["games"][game] = row
        log(f"{game}: k={k:.4g} | fly {row['fly']['mean']:.2f} vs random {rnd['mean']:.2f}, one-way "
            f"{ow['mean']:.2f}, brainless {thr['mean']:.2f} -> beat random {row['verdict']['beat_random']}, "
            f"beat one-way {row['verdict']['beat_one_way']}; {row['verdict']['brainless_verdict']} "
            f"({time.time() - tg:.0f} s)")
    result["wall_s"] = round(time.time() - t_all, 1)
    RESULTS_JSON.write_text(json.dumps(result, indent=1) + "\n")
    log(f"done in {result['wall_s']} s -> {RESULTS_JSON.relative_to(ROOT)}")
    return 0


class _Const:
    def __init__(self, a):
        self.a = int(a)

    def reset(self, seed):
        pass

    def act(self, frame):
        return self.a


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
