"""M7e3a: the show page's data — scene records of the seed-10 games and the fly's brain record.

    python -m flysim.apps.m7e3_show_data                              # all 15 tracks + provenance
    python -m flysim.apps.m7e3_show_data --tracks dodge_v2/m7c2/fly   # a subset, no provenance

Writes ``runs/show/show-data.json`` (contract: docs/m7e3-brief.md §2.1) and, for the full set,
``data-provenance/m7e3-show-data.json`` (source and output sha256, every check).

Nothing is played again with the brain. Every track is ``games.scene.replay`` of recorded actions
through the real game code:

* fly (6): ``runs/{m7c,m7c2}/<game>-brain-seed10/``. Actions from ``frames.npz``; the replay's frames
  and the frames redrawn from ``eye_rect`` must equal the recorded frames bit for bit, the detector's
  output must equal the recorded ``loom`` (float32), score and pass/crash events must equal game.json
  and the results file. The brain record comes from run.json the way the split screen reads it
  (M7e, M7e2): giant-fibre and TTMn spike times from the one-neuron escape regions, cross-checked
  against the spike list; the decoder rebuilt from those spikes must give every recorded action.
* bots (9): random, one-direction (the M7c reference side) and brainless (the M7b θ) on seed 10,
  played to the end with ``runner.run_episode`` and replayed; scores must equal m7b-baselines.json
  and m7c-results.json. For the brainless bot the rule re-applied to the recorded loom and θ must
  give every recorded action.

Any mismatch raises: the show never gets a track that is not the recorded game.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from ..data.download import ROOT
from .games import scene
from .games.agents import RandomAgent, ThresholdAgent
from .games.base import ACTIONS, Action, make
from .games.m7c import CEILING_GAMES, _Const
from .games.runner import run_episode

GAMES = ("jump", "dodge", "dodge_v2")
TITLE_KO = {"jump": "뛰어넘기", "dodge": "피하기", "dodge_v2": "벽 없는 피하기"}
MATCHES = ("m7c", "m7c2")
BOTS = ("random", "one_way", "brainless")
SEED = 10
EVAL_SEEDS = list(range(10, 20))
BUNDLE_DIRS = {"m7c": ROOT / "runs" / "m7c", "m7c2": ROOT / "runs" / "m7c2"}
BASELINES = ROOT / "data-provenance" / "m7b-baselines.json"
M7C_RESULTS = ROOT / "data-provenance" / "m7c-results.json"
M7C2_RESULTS = ROOT / "data-provenance" / "m7c2-results.json"
OUT_JSON = ROOT / "runs" / "show" / "show-data.json"
PROVENANCE = ROOT / "data-provenance" / "m7e3-show-data.json"
BRAIN_SETS = ("LC4_L", "LC4_R", "LPLC2_L", "LPLC2_R", "DNp01_L", "DNp01_R", "TTMn_L", "TTMn_R")
RESET_TRANSIENT_MS = 2000         # M7a2's post-reset transient, drawn by M7e2 unless warm-up >= this
DIGITS = 4                        # positions, distances, heights, speeds, rates (contract §2.1)
SIZE_TARGET = 3_000_000           # bytes (contract §2.1)


def log(msg: str) -> None:
    print(f"[m7e3a] {msg}", flush=True)


def track_ids() -> list[str]:
    out = []
    for game in GAMES:
        out += [f"{game}/{m}/fly" for m in MATCHES] + [f"{game}/{b}" for b in BOTS]
    return out


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.resolve().is_relative_to(ROOT) else str(path)


def _r(v):
    return None if v is None else round(float(v), DIGITS)


# ---------------------------------------------------------------------------------------------
# results: scores, θ, one-way side, verdicts — as written in the result files
# ---------------------------------------------------------------------------------------------
def load_results() -> dict:
    return {"base": json.loads(BASELINES.read_text()), "m7c": json.loads(M7C_RESULTS.read_text()),
            "m7c2": json.loads(M7C2_RESULTS.read_text())}


def game_entry(game: str, res: dict) -> dict:
    """The score card of one game. The three result files must agree on every bot score."""
    base, r1 = res["base"]["games"][game], res["m7c"]["games"][game]
    r2, b2 = res["m7c2"]["arms"]["main"]["games"][game], res["m7c2"]["bots"][game]
    for name, seeds in (("m7c", res["m7c"]["eval_seeds"]), ("m7c2", res["m7c2"]["eval_seeds"]),
                        ("m7b", res["base"]["eval_seeds"])):
        if seeds != EVAL_SEEDS:
            raise RuntimeError(f"{name} evaluation seeds {seeds} != {EVAL_SEEDS}")
    by_seed = lambda row: [dict(zip(row["seeds"], row["scores"]))[s] for s in EVAL_SEEDS]
    now = make(game).describe()
    changed = {k: (v, now.get(k)) for k, v in base["rules"].items() if now.get(k) != v}
    if changed:          # keys added since (dodge's "walls", with v2) are descriptions, not rule changes
        raise RuntimeError(f"{game}: the game's rules changed since m7b-baselines.json: {changed}")
    checks = {"random m7c == m7b": r1["random"]["scores"] == by_seed(base["random"]),
              "brainless m7c == m7b": r1["brainless"]["scores"] == by_seed(base["threshold"]["eval"]),
              "θ m7c == m7b": r1["theta"] == base["threshold"]["theta"] == b2["theta"],
              "random m7c2 == m7c": b2["random"]["scores"] == r1["random"]["scores"],
              "brainless m7c2 == m7c": b2["brainless"]["scores"] == r1["brainless"]["scores"],
              "one-way m7c2 == m7c": b2["one_way"] == r1["one_way"],
              "fly m7c in m7c2 == m7c": r2["m7c"]["scores"] == r1["fly"]["scores"],
              "ceiling": r1["verdict"]["ceiling"] == r2["verdict"]["ceiling"] == CEILING_GAMES.get(game)}
    bad = [k for k, ok in checks.items() if not ok]
    if bad:
        raise RuntimeError(f"{game}: result files disagree: {bad}")
    ref = r1["one_way"]["reference"]
    if ref not in ("always_left", "always_right"):
        raise RuntimeError(f"{game}: unknown one-way reference {ref!r}")
    return {"title_ko": TITLE_KO[game], "rules": make(game).describe(), "eval_seeds": EVAL_SEEDS,
            "scores": {"fly_m7c": r1["fly"]["scores"], "fly_m7c2": r2["fly"]["scores"],
                       "random": r1["random"]["scores"], "one_way": r1["one_way"]["scores"],
                       "brainless": r1["brainless"]["scores"]},
            "one_way_dir": "left" if ref == "always_left" else "right",
            "brainless_theta": base["threshold"]["theta"], "ceiling": r1["verdict"]["ceiling"],
            "verdict": {"m7c": r1["verdict"], "m7c2": r2["verdict"]}}


# ---------------------------------------------------------------------------------------------
# the brain record, from run.json (M7e / M7e2 reading)
# ---------------------------------------------------------------------------------------------
def region_counts(run: dict, name: str) -> np.ndarray:
    """Spikes per bin of one escape region from its rate row (exact for the small sets)."""
    meta = run["meta"]
    if name not in run["regions"]:
        raise KeyError(f"run.json has no region {name!r} (regions: {run['regions']})")
    rate = np.asarray(run["frames"]["rates"][run["regions"].index(name)], dtype=np.float64)
    c = rate * meta["rate_norm_hz"] * meta["region_sizes"][name] * meta["bin_ms"] / 1000.0
    ci = np.rint(c)
    if np.abs(c - ci).max() > 1e-3:
        raise RuntimeError(f"{name}: rates do not give whole spike counts (max off {np.abs(c - ci).max():.3g})")
    return ci.astype(np.int64)


def spike_times(counts: np.ndarray, bin_ms: float) -> list:
    t = np.repeat(np.arange(len(counts)) * bin_ms, counts)
    return [int(x) if float(x).is_integer() else float(x) for x in t]


def decoder_rule(decoder: dict) -> tuple[float, int]:
    """(W, n) of the recorded decoder, as M7e2 reads it; anything else is refused, not guessed."""
    if decoder.get("decoder") == "GiantFibreDecoder":
        return float(decoder["window_ms"]), 1
    if decoder.get("decoder") == "CountWindowDecoder":
        return float(decoder["window_ms"]), int(decoder["min_spikes"])
    raise KeyError(f"unknown decoder {decoder!r}")


def decode_actions(gf: dict, decoder: dict, n_ticks: int, end_ms: int) -> list[int]:
    """Re-decide every tick at its end e from the spikes in [e - W, e) (brain.TickDecoder rule)."""
    W, n = decoder_rule(decoder)
    tl, tr = np.asarray(gf["L"], dtype=float), np.asarray(gf["R"], dtype=float)
    out = []
    for k in range(n_ticks):
        e = scene.frame_ms(k + 1) if k + 1 < n_ticks else end_ms
        cl = int(((tl >= e - W) & (tl < e)).sum()) >= n
        cr = int(((tr >= e - W) & (tr < e)).sum()) >= n
        out.append(int(Action.JUMP) if cl and cr else int(Action.RIGHT) if cl else
                   int(Action.LEFT) if cr else int(Action.NOOP))
    return out


def brain_record(run: dict, gdoc: dict, n_ticks: int) -> tuple[dict, dict]:
    meta = run["meta"]
    bin_ms = float(meta["bin_ms"])
    n_bins = int(run["frames"]["n"])
    dur = int(gdoc["duration_ms"])
    if n_bins * bin_ms != dur or scene.frame_ms(n_ticks) != dur:
        raise RuntimeError(f"run.json covers {n_bins * bin_ms} ms, the game {dur} ms, "
                           f"{n_ticks} ticks end at {scene.frame_ms(n_ticks)} ms")
    sp_t = np.asarray(run["spikes"]["t_bin"], dtype=np.int64)
    sp_i = np.asarray(run["spikes"]["neuron_idx"], dtype=np.int64)
    if meta["spike_sample_ratio"] != 1.0 or len(sp_t) != meta["n_spikes_total"]:
        raise RuntimeError("run.json's spike list is sampled; per-tick network spike counts need all of them")
    region_of = np.asarray(run["neurons"]["region"], dtype=np.int64)
    counts, from_list = {}, {}
    for name in BRAIN_SETS:
        counts[name] = region_counts(run, name)
        r = run["regions"].index(name)
        from_list[name] = np.bincount(sp_t[region_of[sp_i] == r], minlength=n_bins)
    list_agrees = all(np.array_equal(counts[n], from_list[n]) for n in BRAIN_SETS)
    if not list_agrees:
        raise RuntimeError("escape-region rates and the spike list disagree")
    edges = [scene.frame_ms(k) for k in range(n_ticks)] + [dur]
    lo = [int(round(e / bin_ms)) for e in edges]
    sizes = meta["region_sizes"]
    rate = {name: [_r(counts[name][lo[k]:lo[k + 1]].sum() * 1000.0 / (sizes[name] * (edges[k + 1] - edges[k])))
                   for k in range(n_ticks)] for name in BRAIN_SETS}
    per_bin = np.bincount(sp_t, minlength=n_bins)
    net = [int(per_bin[lo[k]:lo[k + 1]].sum()) for k in range(n_ticks)]
    warmup_ms = float(meta.get("warmup_steps", 0) or 0) * float(meta["dt_ms"])
    rec = {"decoder": gdoc["agent"]["decoder"], "warmup_ms": int(warmup_ms) if warmup_ms.is_integer() else warmup_ms,
           "transient_ms": 0 if warmup_ms >= RESET_TRANSIENT_MS else RESET_TRANSIENT_MS,
           "gf": {"L": spike_times(counts["DNp01_L"], bin_ms), "R": spike_times(counts["DNp01_R"], bin_ms)},
           "ttmn": {"L": spike_times(counts["TTMn_L"], bin_ms), "R": spike_times(counts["TTMn_R"], bin_ms)},
           "set_rate_hz": rate, "net_spikes": net}
    check = {"rates_match_spike_list": list_agrees, "net_spikes_total": int(sum(net)),
             "net_spikes_match_run": int(sum(net)) == int(meta["n_spikes_total"])}
    if not check["net_spikes_match_run"]:
        raise RuntimeError("per-tick network spikes do not add up to run.json n_spikes_total")
    return rec, check


# ---------------------------------------------------------------------------------------------
# tracks
# ---------------------------------------------------------------------------------------------
def check_frame_ticks(sc: dict) -> list[int]:
    """Frames handed to the page for its pixel test: the first frame where the first obstacle is
    >= 2 px wide, the middle frame, the last frame."""
    vis = [e["tick"] + 1 for e in sc["events"] if e["type"] == "visible2px" and e["tick"] + 1 < sc["n_ticks"]]
    return sorted({vis[0] if vis else 0, sc["n_ticks"] // 2, sc["n_ticks"] - 1})


def bit_exact(recorded, sc: dict) -> int:
    """Frames where the recorded frame == the replay's frame == the frame redrawn from eye_rect."""
    return int(sum(np.array_equal(a, b) and np.array_equal(a, scene.redraw(r))
                   for a, b, r in zip(recorded, sc["frames"], sc["eye_rect"])))


def assemble(sc: dict, tid: str, player: str, match: str | None, theta, brain, check: dict) -> dict:
    st = sc["state"]
    ticks = check_frame_ticks(sc)
    return {"id": tid, "game": sc["game"], "player": player, "match": match, "seed": sc["seed"],
            "n_ticks": sc["n_ticks"], "score": sc["score"], "crashed": sc["crashed"],
            "truncated": sc["truncated"], "t_ms": sc["t_ms"], "action": sc["action"],
            "loom": sc["loom"],
            "state": {"x": [_r(v) for v in st["x"]], "obj_x": [_r(v) for v in st["obj_x"]],
                      "dist": [_r(v) for v in st["dist"]], "height": [_r(v) for v in st["height"]],
                      "airborne": st["airborne"], "speed": [_r(v) for v in st["speed"]],
                      "score": st["score"]},
            "eye_rect": sc["eye_rect"], "events": sc["events"],
            "obstacles": [{k: ob[k] for k in ("spawn_tick", "end_tick", "side", "outcome", "loom_last300_frac",
                                              "reach_tick", "reach_t_ms", "visible2px_tick")}
                          for ob in sc["obstacles"]],
            "theta": theta, "brain": brain, "check": check,
            "check_frames": {"ticks": ticks,
                             "raw_b64": [base64.b64encode(np.ascontiguousarray(sc["frames"][k], dtype=np.uint8)
                                                          .tobytes()).decode("ascii") for k in ticks]}}


def fly_track(game: str, match: str, entry: dict) -> tuple[dict, list[str]]:
    d = BUNDLE_DIRS[match] / f"{game}-brain-seed{SEED}"
    files = [d / "frames.npz", d / "game.json", d / "run.json"]
    z = np.load(files[0])
    gdoc = json.loads(files[1].read_text())
    run = json.loads(files[2].read_text())
    actions = z["actions"].astype(np.int64)
    if gdoc["seed"] != SEED or gdoc["game"] != game:
        raise RuntimeError(f"{d}: bundle is {gdoc['game']} seed {gdoc['seed']}")
    if gdoc["bundle_ms"] != gdoc["duration_ms"] or len(actions) != gdoc["ticks"]:
        raise RuntimeError(f"{d}: the bundle is cut ({len(actions)} of {gdoc['ticks']} ticks); "
                           "the rest of the actions are not recorded")
    sc = scene.replay(game, SEED, actions)
    expected = entry["scores"][f"fly_{match}"][EVAL_SEEDS.index(SEED)]
    brain, bcheck = brain_record(run, gdoc, sc["n_ticks"])
    decoded = decode_actions(brain["gf"], brain["decoder"], sc["n_ticks"], int(gdoc["duration_ms"]))
    ends = lambda evs: [(e["t_ms"], e["type"]) for e in evs if e["type"] in ("pass", "crash")]
    check = {"frames_bit_exact": bit_exact(z["frames"], sc), "n_frames": int(len(z["frames"])),
             "score_matches": sc["score"] == gdoc["score"] == expected,
             "actions_match": sc["n_ticks"] == gdoc["ticks"] == len(actions),
             "t_ms_match": bool(np.array_equal(z["t_ms"], sc["t_ms"])),
             "loom_matches_bundle": bool(np.array_equal(np.asarray(sc["loom"], dtype=np.float32), z["loom"])),
             "events_match_game_json": ends(sc["events"]) == ends(gdoc["events"]),
             "crash_matches": sc["crashed"] == gdoc["crashed"],
             "decoder_reproduces_actions": int(sum(a == b for a, b in zip(decoded, actions.tolist()))),
             **bcheck}
    _require(f"{game}/{match}/fly", check, sc["n_ticks"])
    return assemble(sc, f"{game}/{match}/fly", "fly", match, None, brain, check), [rel(f) for f in files]


def bot_agent(player: str, entry: dict):
    if player == "random":
        return RandomAgent()
    if player == "one_way":
        return _Const(Action.LEFT if entry["one_way_dir"] == "left" else Action.RIGHT)
    if player == "brainless":
        return ThresholdAgent(entry["brainless_theta"])
    raise KeyError(player)


def threshold_actions(loom: list, theta: float) -> list[int]:
    """The brainless bot's one rule (agents.ThresholdAgent) re-applied to recorded loom and θ."""
    out = []
    for left, right in loom:
        fl, fr = left > theta, right > theta
        out.append(int(Action.JUMP) if fl and fr else int(Action.RIGHT) if fl else
                   int(Action.LEFT) if fr else int(Action.NOOP))
    return out


def bot_track(game: str, player: str, entry: dict) -> dict:
    agent = bot_agent(player, entry)
    ep = run_episode(make(game), agent, SEED, record=True)
    rec = ep["record"]
    sc = scene.replay(game, SEED, rec["actions"])
    expected = entry["scores"][player][EVAL_SEEDS.index(SEED)]
    check = {"frames_bit_exact": bit_exact(rec["frames"], sc), "n_frames": len(rec["frames"]),
             "score_matches": sc["score"] == ep["score"] == expected,
             "actions_match": sc["action"] == [int(a) for a in rec["actions"]] and sc["n_ticks"] == ep["ticks"],
             "crash_matches": sc["crashed"] == ep["crashed"] and sc["truncated"] == ep["truncated"]}
    theta = None
    if player == "brainless":
        theta = entry["brainless_theta"]
        check["loom_matches_bot"] = [tuple(v) for v in sc["loom"]] == [tuple(v) for v in rec["loom"]]
        rule = threshold_actions(sc["loom"], theta)
        check["rule_reproduces_actions"] = int(sum(a == b for a, b in zip(rule, sc["action"])))
    _require(f"{game}/{player}", check, sc["n_ticks"])
    return assemble(sc, f"{game}/{player}", player, None, theta, None, check)


def _require(tid: str, check: dict, n: int) -> None:
    bad = [k for k, v in check.items() if v is False]
    for k in ("frames_bit_exact", "decoder_reproduces_actions", "rule_reproduces_actions"):
        if k in check and check[k] != n:
            bad.append(f"{k} {check[k]}/{n}")
    if check["n_frames"] != n:
        bad.append(f"n_frames {check['n_frames']} != {n}")
    if bad:
        raise RuntimeError(f"{tid}: replay does not match the record: {bad}")


# ---------------------------------------------------------------------------------------------
def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return None


def build(ids: list[str] | None = None) -> tuple[dict, list[str]]:
    """The show-data document for the track ids (default all) and the source files it read."""
    ids = track_ids() if ids is None else list(ids)
    unknown = [t for t in ids if t not in track_ids()]
    if unknown:
        raise KeyError(f"unknown track ids {unknown}; known: {track_ids()}")
    res = load_results()
    sources = [rel(p) for p in (BASELINES, M7C_RESULTS, M7C2_RESULTS)]
    games = {g: game_entry(g, res) for g in GAMES}
    tracks = []
    for tid in ids:
        t0 = time.time()
        parts = tid.split("/")
        if parts[-1] == "fly":
            tr, files = fly_track(parts[0], parts[1], games[parts[0]])
            sources += files
        else:
            tr = bot_track(parts[0], parts[1], games[parts[0]])
        tracks.append(tr)
        log(f"{tid}: {tr['n_ticks']} ticks, score {tr['score']}, frames bit-exact "
            f"{tr['check']['frames_bit_exact']}/{tr['n_ticks']} ({time.time() - t0:.1f} s)")
    meta = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "git_commit": git_commit(),
            "tick_ms_rule": "t_ms(k) = floor(k*1000/60)",
            "action_codes": {str(int(a)): a.name for a in Action},
            "sources": {p: sha256(ROOT / p) for p in sources},
            "alignment": "state[k], eye_rect[k], loom[k] belong to frame k (the state that drew it, the "
                         "detector's output on it); action[k] is chosen on frame k (the step to frame k+1)",
            "event_rule": "tick = the step the event happened in; t_ms = floor((tick+1)*1000/60) = when the "
                          "first frame showing it appears (the game's end for the crash), as in game.json",
            "eye_rect": "[x0, x1, y0, y1] on the 84x84 frame; frame = background (sky 190 above row 36, "
                        "ground 110) + fill_rect(eye_rect, 25) with fractional edge coverage, then rint",
            "precision": f"x, obj_x, dist, height, speed, set_rate_hz: {DIGITS} decimals; loom, theta, "
                         "eye_rect: exact (shortest float64 repr); times: whole ms",
            "loom_last300_frac": "share of the obstacle's looming signal (left + right: it is the only "
                                 "object on screen) over frames spawn_tick+1..reach_tick that falls in "
                                 "ticks starting >= reach_t_ms - 300; reach = the step it first got within "
                                 "D_HIT",
            "brain": "from run.json: gf/ttmn spike times = bin of each spike of the one-neuron DNp01/TTMn "
                     "regions; set_rate_hz = spikes in the tick / (set size x tick length); net_spikes = "
                     "all spikes in the tick (full spike list); tick k = [t_ms(k), t_ms(k+1))",
            "transient_rule": f"transient_ms = {RESET_TRANSIENT_MS} unless warmup_ms >= {RESET_TRANSIENT_MS} "
                              "(M7e2)",
            "track_ids": "fly: <game>/<m7c|m7c2>/fly; bots: <game>/<random|one_way|brainless>",
            "action_names": list(ACTIONS)}
    return {"version": 1, "meta": meta, "games": games, "tracks": tracks}, sources


def write(doc: dict, out: Path) -> dict:
    out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(doc, ensure_ascii=False, separators=(",", ":"))
    out.write_text(text, encoding="utf-8")
    return {"path": rel(out), "bytes": out.stat().st_size, "sha256": sha256(out)}


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="M7e3a: show-page data (scene records, bot replays)")
    ap.add_argument("--tracks", default="", help="comma-separated track ids (default: all 15)")
    ap.add_argument("--out", default=str(OUT_JSON))
    args = ap.parse_args(argv)
    ids = [t for t in args.tracks.split(",") if t] or None
    t0 = time.time()
    doc, _ = build(ids)
    out = write(doc, Path(args.out).resolve())
    log(f"wrote {out['path']}: {out['bytes']:,} bytes ({len(doc['tracks'])} tracks, {time.time() - t0:.1f} s)")
    if out["bytes"] > SIZE_TARGET:
        log(f"WARNING: above the {SIZE_TARGET:,}-byte target of the contract")
    if ids is None:
        prov = {"generated": doc["meta"]["generated"], "git_commit": doc["meta"]["git_commit"],
                "command": "python -m flysim.apps.m7e3_show_data", "sources": doc["meta"]["sources"],
                "output": out, "size_target_bytes": SIZE_TARGET,
                "tracks": {t["id"]: {"n_ticks": t["n_ticks"], "score": t["score"], "crashed": t["crashed"],
                                     "truncated": t["truncated"], "check": t["check"],
                                     "obstacles": len(t["obstacles"]),
                                     "loom_last300_frac": [o["loom_last300_frac"] for o in t["obstacles"]]}
                           for t in doc["tracks"]},
                "rules": {k: doc["meta"][k] for k in ("alignment", "event_rule", "precision",
                                                       "loom_last300_frac", "brain", "transient_rule")},
                "wall_s": round(time.time() - t0, 1)}
        PROVENANCE.write_text(json.dumps(prov, indent=1, ensure_ascii=False) + "\n")
        log(f"wrote {rel(PROVENANCE)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
