"""M7e4a: the show page's second round — scene records of the eight classic games and the slow rung.

    python -m flysim.apps.m7e4_show_data rerecord            # dodge_v2_s4 seed 10, the whole 120 s (GPU)
    python -m flysim.apps.m7e4_show_data classic [--games boxing]
    python -m flysim.apps.m7e4_show_data slow
    python -m flysim.apps.m7e4_show_data all                 # classic + slow + provenance

Writes ``runs/show/classic/<game>.json``, ``runs/show/classic/index.json`` and
``runs/show/slow.json`` (contract: docs/m7e4-brief.md §2 and "계약 추가"), and for ``all``
``data-provenance/m7e4-show-data.json`` (source and output sha256, every check, the git commit).

Nothing is decided here. Every track is the real game code stepped with recorded actions:

* classic fly (8): ``runs/m7f/<game>-brain-seed10/``. Actions from ``frames.npz``; every replayed
  frame must equal the recorded frame, ``classic.common.draw`` of the replayed scene and of the
  recorded ``scene.json`` scene bit for bit; the detector's output must equal the recorded ``loom``
  (float32); score, HUD and score / life / game-over events must equal game.json; the decoder
  rebuilt from the recorded giant-fibre spikes must give every recorded action.
* classic bots (8 x 3): random, the fixed-action reference and the brainless bot (part-1 θ) of
  ``m7f-baselines.json`` on seed 10, played with ``runner.run_episode`` (the m7f bot spec) and
  replayed; scores must equal both result files, frames must equal ``draw`` of the scene.
* scene encoding: ``static`` + ``ticks[k]`` (null = as the tick before). When the four players of a
  game see the same rectangles right after ``static`` at every tick (crossing's cars, which depend
  on seed and tick only), that run goes once into the file's ``shared_scene`` (planning session,
  2026-10-31: crossing.json was 22 MB). Every tick of every track is read back through JSON and
  must give the replay's ``scene()`` list and the recorded frame.
* slow: ``dodge_v2_s4`` seed 10, the same four players in the M7e3 track shape, through
  ``scene.replay`` wrapped for a game ``base.make`` does not know (``slow_replay``). The fly's game
  lasted 120 s and the M7g bundle keeps the first 60 s (M7c's bundle cap), so the planning session
  allowed one display re-recording (``rerecord``): same seed, same k, same settings, written to
  ``runs/m7e4/``; its first 3,600 ticks (frames, actions, loom, spikes) must equal the M7g bundle
  bit for bit and its episode summary must equal the M7g evaluation record of seed 10.

Any mismatch raises: the show never gets a track that is not the recorded game.
"""
from __future__ import annotations

import argparse
import base64
import contextlib
import json
import sys
import time
from pathlib import Path

import numpy as np

from ..data.download import ROOT
from ..sensory.looming import LoomingDetector
from . import m7e3_show_data as E3
from .games import scene
from .games.agents import RandomAgent, ThresholdAgent
from .games.base import Action, make
from .games.brain import tick_end_ms
from .games.classic import GAMES as CLASSIC_GAMES
from .games.classic.common import BACKGROUND, FIXED, MOVING, draw
from .games.runner import run_episode
from .games.slow import SLOW_MAX_TICKS, VARIANTS, make_slow

SEED = 10
EVAL_SEEDS = list(range(10, 20))
M7F_DIR = ROOT / "runs" / "m7f"
M7F_BASELINES = ROOT / "data-provenance" / "m7f-baselines.json"
M7F_RESULTS = ROOT / "data-provenance" / "m7f-results.json"
M7F_YOKED = ROOT / "data-provenance" / "m7f-yoked.json"
LADDER = ROOT / "docs" / "m7-game-ladder.md"
SHOW_DIR = ROOT / "runs" / "show"
CLASSIC_DIR = SHOW_DIR / "classic"
INDEX_JSON = CLASSIC_DIR / "index.json"
SLOW_JSON = SHOW_DIR / "slow.json"
PROVENANCE = ROOT / "data-provenance" / "m7e4-show-data.json"
SIZE_TARGET = 8_000_000           # bytes per file (brief §2.1): above it, report, do not reduce
CLASSIC_PLAYERS = ("fly", "random", "fixed", "brainless")
COUNTED = ("frames_bit_exact", "scene_reads_back", "scene_json_bit_exact", "scene_matches_scene_json",
           "decoder_reproduces_actions", "rule_reproduces_actions")   # must equal the number of ticks
POSTHOC_NOTE = "사후 점검 — 판정 아님"
HYPOTHESIS_LABEL = "가설 — 약하게만 지지됨(느린 판 실험)"
SLOW_TABLE_ROWS = ("dodge_v2", "dodge_v2_s2", "dodge_v2_s4", "dodge_v2_s8", "jump", "jump_s2", "jump_s4")
SLOW_TITLE_KO = {"dodge_v2": E3.TITLE_KO["dodge_v2"], "jump": E3.TITLE_KO["jump"]}
SHARED_APPROVAL = "planning session, 2026-10-31 (crossing.json was 22.0 MB, above the 8 MB target)"
SHARED_REASON = {"crossing": "길 건너기의 차는 시드와 틱으로만 정해져 네 판이 같다(3,600틱 확인)",
                 "paddle": "네 판 모두 312틱에 끝나고(점수 0) 상대 막대와 공이 모든 틱에서 같다. 공은 그리는 순서가 "
                           "선수 막대 뒤라, 공유 층에는 상대 막대(312틱)와 선수 막대·공이 모두 같은 틱(13틱)만 "
                           "들어간다. 규칙이 정한 공유"}
SLOW_ROW = "dodge_v2_s4"
M7G_RESULTS = ROOT / "data-provenance" / "m7g-results.json"
M7G_BUNDLE = ROOT / "runs" / "m7g" / f"{SLOW_ROW}-brain-seed{SEED}"
RERECORD_DIR = ROOT / "runs" / "m7e4" / f"{SLOW_ROW}-brain-seed{SEED}"
RERECORD_NOTE = "표시용 재녹화(같은 시드·같은 k·같은 설정), 앞 60 s 비트 일치 확인"
SLOW_GAME_MS = tick_end_ms(SLOW_MAX_TICKS - 1)          # 120,000 ms: the slow variants' cap
EPISODE_KEYS = ("score", "ticks", "crashed", "truncated", "network_rate_hz", "active_fraction", "gf_spikes",
                "n_spikes", "input_cell_spikes", "decoder_outputs")


def log(msg: str) -> None:
    print(f"[m7e4a] {msg}", flush=True)


# ---------------------------------------------------------------------------------------------
# the one display re-recording (planning session, 2026-10-31): dodge_v2_s4 seed 10, all 120 s
# ---------------------------------------------------------------------------------------------
@contextlib.contextmanager
def bundle_cap(ms: int):
    """``m7c.BUNDLE_MAX_MS`` (read at call time by ``m7g.play`` / ``m7g.write_bundle``) set to
    ``ms`` for this process only; the file is not touched."""
    from .games import m7c as M
    old = M.BUNDLE_MAX_MS
    M.BUNDLE_MAX_MS = ms
    try:
        yield
    finally:
        M.BUNDLE_MAX_MS = old


def _spikes(run: dict) -> tuple[np.ndarray, np.ndarray]:
    return (np.asarray(run["spikes"]["t_bin"], dtype=np.int64),
            np.asarray(run["spikes"]["neuron_idx"], dtype=np.int64))


def compare_with_m7g(new_dir: Path = RERECORD_DIR, old_dir: Path = M7G_BUNDLE) -> dict:
    """The re-recording against the M7g bundle over the M7g bundle's ticks: frames, t_ms, actions,
    loom, every spike (time bin and neuron), and the game events of those ticks."""
    zn, zo = np.load(new_dir / "frames.npz"), np.load(old_dir / "frames.npz")
    n = len(zo["actions"])
    out = {"old_ticks": int(n), "new_ticks": int(len(zn["actions"]))}
    for key in ("frames", "t_ms", "actions", "loom"):
        out[f"{key}_equal"] = bool(zn[key].dtype == zo[key].dtype and np.array_equal(zn[key][:n], zo[key]))
    rn, ro = json.loads((new_dir / "run.json").read_text()), json.loads((old_dir / "run.json").read_text())
    old_ms = int(ro["meta"]["bundle_ms"])
    tn, in_ = _spikes(rn)
    to, io = _spikes(ro)
    keep = tn < old_ms
    out["old_bundle_ms"] = old_ms
    out["spikes_old"] = int(len(to))
    out["spikes_new_first_part"] = int(keep.sum())
    out["spikes_equal"] = bool(np.array_equal(tn[keep], to) and np.array_equal(in_[keep], io))
    gn, go = json.loads((new_dir / "game.json").read_text()), json.loads((old_dir / "game.json").read_text())
    out["events_equal"] = [e for e in gn["events"] if e["t_ms"] <= old_ms] == go["events"]
    out["game_json_equal_except_length"] = all(gn[k] == go[k] for k in go
                                               if k not in ("bundle_ms", "events"))
    return out


def rerecord(device: str = "cuda") -> dict:
    """Play dodge_v2_s4 seed 10 again with the brain exactly as M7g's bundle did (``m7g._brain``,
    ``m7g.play``, ``m7g.write_bundle``), keeping all 120 s, and check it is the same game."""
    from .games import m7g as G
    res = json.loads(M7G_RESULTS.read_text())
    row = res["rows"][SLOW_ROW]
    k = float(row["k"])
    want = next(e for e in row["fly"]["episodes"] if e["seed"] == SEED)
    if row["bundle"]["dir"] != str(M7G_BUNDLE.relative_to(ROOT)):
        raise RuntimeError(f"m7g-results.json names the bundle {row['bundle']['dir']}")
    t0 = time.time()
    brain, agent, choice, ctx = G._brain(device)
    if (choice["window_ms"], choice["min_spikes"]) != (300.0, 2):
        raise RuntimeError(f"decoder choice {choice} is not M7c2's main arm")
    log(f"{SLOW_ROW} seed {SEED}: k = {k!r}, recording the whole game (cap {SLOW_GAME_MS} ms) ...")
    brain.k = k
    brain.keep_spikes = True
    try:
        with bundle_cap(SLOW_GAME_MS):
            ep = G.play(SLOW_ROW, agent, brain, SEED, record=True)
            summary = {key: ep[key] for key in EPISODE_KEYS}
            differ = [key for key in EPISODE_KEYS if summary[key] != want[key]]
            if differ:
                raise RuntimeError(f"the re-recorded game differs from the M7g evaluation record in {differ}")
            bundle = G.write_bundle(SLOW_ROW, k, ep, ctx, brain, RERECORD_DIR, agent.decoder.describe())
    finally:
        brain.keep_spikes = False
        brain.spikes_t, brain.spikes_i = [], []
    check = compare_with_m7g()
    check["episode_matches_m7g_eval"] = {key: True for key in EPISODE_KEYS}
    bad = [key for key, v in check.items() if v is False]
    if bad or check["new_ticks"] != SLOW_MAX_TICKS or bundle["bundle_ms"] != SLOW_GAME_MS:
        raise RuntimeError(f"the re-recording is not the M7g game: {bad} {check}")
    doc = {"note": RERECORD_NOTE, "row": SLOW_ROW, "seed": SEED, "k": k, "device": device,
           "how": "m7g._brain + m7g.play(record=True) + m7g.write_bundle with m7c.BUNDLE_MAX_MS set to "
                  f"{SLOW_GAME_MS} for this process (files unchanged)",
           "m7g_bundle": str(M7G_BUNDLE.relative_to(ROOT)), "bundle": bundle, "episode": summary,
           "check": check, "wall_s": round(time.time() - t0, 1),
           "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "git_commit": E3.git_commit()}
    (RERECORD_DIR / "rerecord.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    log(f"re-recorded {bundle['frames']} ticks, score {summary['score']}; first {check['old_ticks']} ticks "
        f"and {check['spikes_old']:,} spikes equal the M7g bundle ({doc['wall_s']} s)")
    return doc


# ---------------------------------------------------------------------------------------------
# shared: meta, check frames, writing
# ---------------------------------------------------------------------------------------------
def check_ticks(n: int) -> list[int]:
    """Frames handed to the page for its pixel test: tick 10, the middle tick, the tenth from the end."""
    return sorted({min(10, n - 1), n // 2, max(n - 10, 0)})


def raw_b64(frame: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(frame, dtype=np.uint8).tobytes()).decode("ascii")


def _require(tid: str, check: dict, n: int, counted: tuple[str, ...]) -> None:
    bad = [k for k, v in check.items() if v is False]
    bad += [f"{k} {check[k]}/{n}" for k in counted if k in check and check[k] != n]
    if check["n_frames"] != n:
        bad.append(f"n_frames {check['n_frames']} != {n}")
    if bad:
        raise RuntimeError(f"{tid}: replay does not match the record: {bad}")


def base_meta(sources: list[str]) -> dict:
    return {"generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "git_commit": E3.git_commit(),
            "tick_ms_rule": "t_ms(k) = floor(k*1000/60)",
            "action_codes": {str(int(a)): a.name for a in Action},
            "sources": {p: E3.sha256(ROOT / p) for p in dict.fromkeys(sources)}}


def write_json(doc: dict, out: Path) -> dict:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    size = out.stat().st_size
    if size > SIZE_TARGET:
        log(f"WARNING: {E3.rel(out)} is {size:,} bytes, above the {SIZE_TARGET:,}-byte target (not reduced)")
    return {"path": E3.rel(out), "bytes": size, "sha256": E3.sha256(out)}


# ---------------------------------------------------------------------------------------------
# classic games (§2.1)
# ---------------------------------------------------------------------------------------------
def classic_replay(game: str, seed: int, actions) -> dict:
    """Step ``base.make(game)`` with ``actions`` from ``reset(seed)``; keep every frame's ``scene()``,
    the frame, the detector's output on it and the score / life / game-over events (stamped as
    ``m7f.play_classic`` stamps them: the end of the step they happened in). The game must end on
    the last action, not before and not after."""
    actions = [int(a) for a in actions]
    env = make(game)
    det = LoomingDetector()
    frame = env.reset(seed)
    scenes, frames, looms, events = [], [], [], []
    lives = env.lives()
    info = {"score": 0, "crashed": False, "truncated": False}
    for k, a in enumerate(actions):
        if env.done:
            raise ValueError(f"{game} seed {seed}: the game ended after {k} ticks, but {len(actions)} actions were given")
        scenes.append(env.scene())
        frames.append(frame)
        looms.append([float(v) for v in det.update(frame)])
        frame, r, _, info = env.step(a)
        t = tick_end_ms(k)
        if r > 0:
            events.append({"tick": k, "t_ms": t, "type": "score", "points": int(r), "score": int(info["score"])})
        now = env.lives()
        if now is not None and now < lives:
            events.append({"tick": k, "t_ms": t, "type": "life_lost", "lives": int(now)})
        lives = now
        if info["crashed"]:
            events.append({"tick": k, "t_ms": t, "type": "game_over"})
    if not env.done:
        raise ValueError(f"{game} seed {seed}: {len(actions)} actions given, but the game goes on")
    return {"game": game, "seed": int(seed), "n_ticks": len(actions), "score": int(env.score),
            "crashed": bool(info["crashed"]), "truncated": bool(info["truncated"]), "action": actions,
            "loom": looms, "scenes": scenes, "frames": np.stack(frames), "events": events}


def _items(scene_: dict) -> list[tuple]:
    return [(r["kind"], *r["rect"], r["value"]) for r in scene_["rects"]]


def _static_len(lists: list[list[tuple]]) -> int:
    """Length of the prefix every tick's list starts with, identically."""
    n = len(lists[0])
    for lst in lists[1:]:
        m = 0
        while m < min(n, len(lst)) and lst[m] == lists[0][m]:
            m += 1
        n = m
    return n


def _kinds(lists, allowed: tuple[str, ...]) -> list[str]:
    kinds: list[str] = []
    for lst in lists:
        for item in lst:
            if item[0] not in kinds:
                kinds.append(item[0])
    unknown = [k for k in kinds if k not in allowed]
    if unknown:
        raise KeyError(f"scene kinds {unknown} are not the game's kinds {allowed}")
    return kinds


def _ticks(rests: list[list[tuple]], kinds: list[str]) -> list:
    """Per-tick item lists, None where a tick's list equals the tick before's."""
    out, prev = [], None
    for rest in rests:
        out.append(None if prev is not None and rest == prev else [[kinds.index(i[0]), *i[1:]] for i in rest])
        prev = rest
    return out


def shared_layer(scene_lists: list[list[dict]]) -> list[list[tuple]] | None:
    """The layer the four players of a game see identically (planning session, 2026-10-31): when
    every track has the same number of ticks and, at every tick, the four lists after each
    track's ``static`` start with the same non-empty run of rectangles, that run is the tick's
    shared layer. Otherwise there is none (and the game file has no ``shared_scene``)."""
    lists = [[_items(s) for s in scenes] for scenes in scene_lists]
    n = len(lists[0])
    if any(len(lst) != n for lst in lists):
        return None
    starts = [_static_len(t) for t in lists]
    rests = [[lst[m:] for lst in t] for t, m in zip(lists, starts)]
    out = []
    for k in range(n):
        first = rests[0][k]
        m = len(first)
        for r in rests[1:]:
            j = 0
            while j < min(m, len(r[k])) and r[k][j] == first[j]:
                j += 1
            m = j
        if m == 0:
            return None
        out.append(first[:m])
    return out


def encode_shared(shared: list[list[tuple]], allowed: tuple[str, ...]) -> dict:
    kinds = _kinds(shared, allowed)
    return {"kinds": kinds, "ticks": _ticks(shared, kinds)}


def encode_scene(scenes: list[dict], allowed: tuple[str, ...], shared: list[list[tuple]] | None = None) -> dict:
    """``static`` = the rectangles every tick's list starts with, identically; then, when
    ``shared`` is given, that tick's shared layer (``shared_scene``); ``ticks[k]`` = the rest of tick
    k's list, or None when it equals tick k - 1's rest. Items ``[kind index, x0, x1, y0, y1,
    value]``; ``kinds`` in order of first appearance. Nothing is rounded."""
    lists = [_items(s) for s in scenes]
    n = _static_len(lists)
    rests = [lst[n:] for lst in lists]
    if shared is not None:
        if len(shared) != len(rests) or any(r[:len(sh)] != sh for r, sh in zip(rests, shared)):
            raise ValueError("the shared layer is not where every tick's list continues after static")
        rests = [r[len(sh):] for r, sh in zip(rests, shared)]
    kinds = _kinds([lists[0][:n]] + rests, allowed)
    out = {"kinds": kinds, "static": [[kinds.index(i[0]), *i[1:]] for i in lists[0][:n]], "ticks": _ticks(rests, kinds)}
    if shared is not None:
        out["shared"] = True
    return out


def _read_ticks(ticks: list, kinds: list[str]) -> list[list[dict]]:
    out, cur = [], None
    for t in ticks:
        cur = cur if t is None else t
        if cur is None:
            raise ValueError("the first tick of a scene record cannot be null")
        out.append([{"kind": kinds[i[0]], "rect": list(i[1:5]), "value": i[5]} for i in cur])
    return out


def decode_scene(sc: dict, shared_scene: dict | None = None) -> list[list[dict]]:
    """The page's reading: tick k's rectangles = static + shared_scene.ticks[k] (when the track's
    scene says ``shared``) + ticks[k], in that drawing order."""
    static = _read_ticks([sc["static"]], sc["kinds"])[0]
    own = _read_ticks(sc["ticks"], sc["kinds"])
    if not sc.get("shared"):
        return [static + o for o in own]
    if shared_scene is None:
        raise ValueError("the track's scene uses the shared layer, but there is no shared_scene")
    sh = _read_ticks(shared_scene["ticks"], shared_scene["kinds"])
    if len(sh) != len(own):
        raise ValueError(f"shared_scene has {len(sh)} ticks, the track {len(own)}")
    return [static + s + o for s, o in zip(sh, own)]


def classic_track(sc: dict, player: str, theta, brain, check: dict) -> dict:
    ticks = check_ticks(sc["n_ticks"])
    return {"id": f"{sc['game']}/{player}", "game": sc["game"], "player": player, "seed": sc["seed"],
            "n_ticks": sc["n_ticks"], "score": sc["score"], "crashed": sc["crashed"], "truncated": sc["truncated"],
            "action": sc["action"], "loom": sc["loom"], "scene": sc["encoded"],
            "hud": {"score": [s["hud"]["score"] for s in sc["scenes"]],
                    "lives": [s["hud"]["lives"] for s in sc["scenes"]]},
            "events": sc["events"], "theta": theta, "brain": brain, "check": check,
            "check_frames": {"ticks": ticks, "raw_b64": [raw_b64(sc["frames"][k]) for k in ticks]}}


def scene_checks(sc: dict, allowed: tuple[str, ...], shared: list[list[tuple]] | None,
                 shared_scene: dict | None) -> dict:
    """Encode the replay's scenes (with the game's shared layer, if any), read them back the way
    the page will (through JSON), and count the ticks where recorded frame == replayed frame ==
    draw(read-back rectangles) and where the read-back list equals the replay's ``scene()`` list."""
    sc["encoded"] = encode_scene(sc["scenes"], allowed, shared)
    back = decode_scene(json.loads(json.dumps(sc["encoded"])),
                        None if shared_scene is None else json.loads(json.dumps(shared_scene)))
    same_frames = sum(np.array_equal(f, g) and np.array_equal(draw(b), f)
                      for f, g, b in zip(sc["recorded_frames"], sc["frames"], back))
    same_rects = sum(b == s["rects"] for b, s in zip(back, sc["scenes"]))
    return {"frames_bit_exact": int(same_frames), "n_frames": int(len(sc["recorded_frames"])),
            "scene_reads_back": int(same_rects), "shared_layer_ticks": 0 if shared is None else len(shared)}


def classic_fly(game: str, entry: dict) -> tuple[dict, dict, dict, list[str]]:
    """The fly's replay and its checks against the bundle (the scene checks follow in ``finish``)."""
    d = M7F_DIR / f"{game}-brain-seed{SEED}"
    files = [d / "frames.npz", d / "game.json", d / "run.json", d / "scene.json"]
    z = np.load(files[0])
    gdoc = json.loads(files[1].read_text())
    run = json.loads(files[2].read_text())
    sdoc = json.loads(files[3].read_text())
    actions = z["actions"].astype(np.int64)
    if gdoc["seed"] != SEED or gdoc["game"] != game or sdoc["seed"] != SEED or sdoc["game"] != game:
        raise RuntimeError(f"{d}: bundle is {gdoc['game']} seed {gdoc['seed']}")
    if gdoc["bundle_ms"] != gdoc["duration_ms"] or len(actions) != gdoc["ticks"] or sdoc["frames"] != len(actions):
        raise RuntimeError(f"{d}: the bundle is cut ({len(actions)} of {gdoc['ticks']} ticks)")
    sc = classic_replay(game, SEED, actions)
    sc["recorded_frames"] = z["frames"]
    n = sc["n_ticks"]
    recorded = sdoc["scenes"]
    check = {"scene_json_bit_exact": int(sum(np.array_equal(draw(s["rects"]), f) for s, f in zip(recorded, z["frames"]))),
             "scene_matches_scene_json": int(sum(json.loads(json.dumps(a)) == b for a, b in zip(sc["scenes"], recorded)))}
    brain, bcheck = E3.brain_record(run, gdoc, n)
    if (brain["warmup_ms"], brain["transient_ms"]) != (5000, 0):
        raise RuntimeError(f"{game}: warm-up {brain['warmup_ms']} ms / transient {brain['transient_ms']} ms")
    decoded = E3.decode_actions(brain["gf"], brain["decoder"], n, int(gdoc["duration_ms"]))
    kinds = ("score", "life_lost", "game_over")
    strip = lambda evs: [{k: v for k, v in e.items() if k != "tick"} for e in evs if e["type"] in kinds]
    expected = entry["scores"]["fly"][EVAL_SEEDS.index(SEED)]
    check |= {"score_matches": sc["score"] == gdoc["score"] == expected,
              "actions_match": n == gdoc["ticks"] == len(actions),
              "t_ms_match": bool(np.array_equal(z["t_ms"], [scene.frame_ms(k) for k in range(n)]))
                            and sdoc["t_ms"] == [float(scene.frame_ms(k)) for k in range(n)],
              "loom_matches_bundle": bool(np.array_equal(np.asarray(sc["loom"], dtype=np.float32), z["loom"])),
              "events_match_game_json": strip(sc["events"]) == strip(gdoc["events"]),
              "crash_matches": sc["crashed"] == gdoc["crashed"],
              "decoder_reproduces_actions": int(sum(a == b for a, b in zip(decoded, actions.tolist()))),
              **bcheck}
    return sc, check, brain, [E3.rel(f) for f in files]


def classic_bot(game: str, player: str, entry: dict) -> tuple[dict, dict, float | None]:
    """The m7f bot spec (``m7f.make_agent``) on seed 10 to the end, then replayed."""
    from .games import m7f as F
    spec = {"random": ("random",), "fixed": ("const", int(Action[entry["fixed_ref"]])),
            "brainless": ("threshold", entry["brainless_theta"])}[player]
    env = make(game)
    ep = run_episode(env, F.make_agent(spec, env), SEED, record=True)
    rec = ep["record"]
    sc = classic_replay(game, SEED, rec["actions"])
    sc["recorded_frames"] = rec["frames"]
    n = sc["n_ticks"]
    expected = entry["scores"][player][EVAL_SEEDS.index(SEED)]
    check = {"score_matches": sc["score"] == ep["score"] == expected,
             "actions_match": sc["action"] == [int(a) for a in rec["actions"]] and n == ep["ticks"],
             "crash_matches": sc["crashed"] == ep["crashed"] and sc["truncated"] == ep["truncated"],
             "hud_score_matches_bot": [s["hud"]["score"] for s in sc["scenes"]] == [int(v) for v in rec["scores"]]}
    theta = None
    if player == "brainless":
        theta = entry["brainless_theta"]
        check["loom_matches_bot"] = [tuple(v) for v in sc["loom"]] == [tuple(v) for v in rec["loom"]]
        check["rule_reproduces_actions"] = int(sum(a == b for a, b in zip(E3.threshold_actions(sc["loom"], theta),
                                                                            sc["action"])))
    return sc, check, theta


def classic_game(game: str, entry: dict) -> tuple[list[dict], dict | None, list[str]]:
    """The four tracks of one game, the shared layer (or None) and the bundle files read."""
    allowed = tuple(entry["kinds"])
    fly, fcheck, brain, files = classic_fly(game, entry)
    played = [(fly, fcheck, None, brain)] + [(*classic_bot(game, p, entry), None) for p in CLASSIC_PLAYERS[1:]]
    shared = shared_layer([sc["scenes"] for sc, *_ in played])
    shared_scene = None if shared is None else encode_shared(shared, allowed)
    tracks = []
    for player, (sc, check, theta, br) in zip(CLASSIC_PLAYERS, played):
        check = scene_checks(sc, allowed, shared, shared_scene) | check
        _require(f"{game}/{player}", check, sc["n_ticks"], COUNTED)
        tracks.append(classic_track(sc, player, theta, br, check))
    return tracks, shared_scene, files


def load_classic_results() -> tuple[dict, dict, dict | None, list[str]]:
    base = json.loads(M7F_BASELINES.read_text())
    res = json.loads(M7F_RESULTS.read_text())
    yoked = json.loads(M7F_YOKED.read_text()) if M7F_YOKED.exists() else None
    sources = [E3.rel(M7F_BASELINES), E3.rel(M7F_RESULTS)] + ([E3.rel(M7F_YOKED)] if yoked else [])
    for name, d in (("m7f-baselines", base), ("m7f-results", res)) + ((("m7f-yoked", yoked),) if yoked else ()):
        if d["eval_seeds"] != EVAL_SEEDS:
            raise RuntimeError(f"{name} evaluation seeds {d['eval_seeds']} != {EVAL_SEEDS}")
    return base, res, yoked, sources


def classic_entry(game: str, base: dict, res: dict, yoked: dict | None) -> dict:
    """The game block: rules, scores, verdicts as the result files have them. The two result files
    (and the yoked file, when present) must agree on every shared value."""
    b, rb, rg = base["games"][game], res["bots"][game], res["games"][game]
    env = make(game)
    rules = env.describe()
    checks = {"rules unchanged since part 1": rules == b["rules"],
              "θ": rb["theta"] == b["threshold"]["theta"],
              "random": rb["random"]["scores"] == b["random"]["scores"],
              "fixed reference": rb["fixed"]["reference"] == b["fixed"]["reference"]
                                 and rb["fixed"]["scores"] == b["fixed"]["scores"],
              "brainless": rb["brainless"]["scores"] == b["threshold"]["eval"]["scores"],
              "oracle": rb["oracle"]["scores"] == b["oracle_reference"]["scores"],
              "ceiling": rb["ceiling"] == (rb["brainless"]["scores"] == rb["oracle"]["scores"])
                         == rg["verdict"]["ceiling"],
              "fly row": "fly" in rg and len(rg["fly"]["scores"]) == len(EVAL_SEEDS)}
    yk = yoked["games"][game] if yoked else None
    if yk is not None:
        checks |= {"yoked fly == part 2": yk["fly"]["scores"] == rg["fly"]["scores"],
                   "yoked k == part 2": yk["k"] == rg["k"],
                   "yoked diagonal": yk["diagonal_equals_fly"] is True
                                     and [yk["matrix"]["scores"][i][i] for i in range(10)] == rg["fly"]["scores"]}
    bad = [k for k, ok in checks.items() if not ok]
    if bad:
        raise RuntimeError(f"{game}: result files disagree: {bad}")
    return {"name": game, "title_ko": env.title_ko, "genre_ko": env.genre_ko, "player_ko": env.player_ko,
            "jump_effect": env.jump_effect, "left_effect": env.left_effect, "right_effect": env.right_effect,
            "automatic": env.automatic, "score_rule": env.score_rule, "end_rule": env.end_rule,
            "kinds": list(env.kinds), "rules": rules, "eval_seeds": EVAL_SEEDS,
            "scores": {"fly": rg["fly"]["scores"], "random": rb["random"]["scores"], "fixed": rb["fixed"]["scores"],
                       "brainless": rb["brainless"]["scores"], "oracle": rb["oracle"]["scores"],
                       "yoked": yk["yoked"]["scores"] if yk else None},
            "fixed_ref": rb["fixed"]["reference"], "brainless_theta": rb["theta"], "k": rg["k"], "p99": rg["p99"],
            "ceiling": rb["ceiling"],
            "verdict": {key: rg["verdict"][key] for key in ("vs_random", "vs_fixed", "vs_brainless")},
            "yoked": None if yk is None else {
                "posthoc": True, "note": POSTHOC_NOTE,
                "matrix_tape_x_game": yk["matrix"]["scores"],
                "matrix_rule": yoked["rules"]["matrix"], "yoked_score_rule": yoked["rules"]["yoked_score"],
                "fly": yk["fly"], "yoked": yk["yoked"], "verdict": yk["paired"],
                "diagonal_equals_fly": yk["diagonal_equals_fly"], "cells_wrapped": yk["cells_wrapped"]}}


def classic_meta(sources: list[str]) -> dict:
    return base_meta(sources) | {
        "values": {"background": BACKGROUND, "fixed": FIXED, "moving": MOVING},
        "alignment": "as scene.json: scene k (static [+ shared_scene.ticks[k]] + ticks[k]) drew frame k; loom[k] = the detector on "
                     "frame k; hud.score[k] / hud.lives[k] = scene()['hud'] of frame k; action[k] is chosen "
                     "on frame k (the step from frame k to frame k + 1)",
        "scene_encoding": "tick k's rectangles = static + ticks[k], drawn in that order; ticks[k] null = "
                          "the same as tick k - 1; item [kind index into scene.kinds, x0, x1, y0, y1, value]; "
                          "frame = background + base.fill_rect of each item (fractional coverage), then rint. "
                          "When the track's scene has \"shared\": true: static + shared_scene.ticks[k] + "
                          "ticks[k] (shared_scene at the top of the file, its own kinds, the same null rule)",
        "shared_rule": "shared_scene exists only when all four tracks have the same number of ticks and, at "
                       "every tick, their lists after static start with the same non-empty run of rectangles "
                       "(that run is the shared layer); otherwise the key is absent",
        "event_rule": "tick = the step the event happened in; t_ms = floor((tick+1)*1000/60), as "
                      "m7f.play_classic and game.json stamp it",
        "precision": "rects, value, loom, theta, spike times: exact (shortest float64 repr); "
                     f"set_rate_hz: {E3.DIGITS} decimals",
        "loom": "the detector (LoomingDetector) on every frame of every track; the fly's input and the "
                "brainless bot's rule read it, the random and fixed-action bots do not",
        "brain": "fly tracks only, read from run.json as in M7e3 (m7e3_show_data.brain_record)",
        "players": {"fly": "the fly (M7c2 main arm, m7f part 2)", "random": "random bot",
                    "fixed": "the fixed-action reference (m7f-baselines.json)", "brainless": "brainless bot "
                    "(part-1 θ)"},
        "oracle": "scores only: the cheating reference reads the game state (not a track)"}


def build_classic(games=CLASSIC_GAMES) -> tuple[dict, dict, list[str]]:
    """{game: game file document}, the index document, the sources read."""
    unknown = [g for g in games if g not in CLASSIC_GAMES]
    if unknown:
        raise KeyError(f"{unknown} are not the eight classic games {CLASSIC_GAMES}")
    base, res, yoked, sources = load_classic_results()
    docs, all_sources = {}, list(sources)
    for g in games:
        t0 = time.time()
        entry = classic_entry(g, base, res, yoked)
        tracks, shared_scene, files = classic_game(g, entry)
        docs[g] = {"version": 1, "meta": classic_meta(sources + files), "game": entry}
        if shared_scene is not None:
            docs[g]["shared_scene"] = shared_scene
        docs[g]["tracks"] = tracks
        all_sources += files
        log(f"{g}: " + ", ".join(f"{t['player']} {t['n_ticks']} ticks score {t['score']} "
                                 f"(bit-exact {t['check']['frames_bit_exact']})" for t in tracks)
            + (" | shared layer" if shared_scene is not None else "") + f" ({time.time() - t0:.1f} s)")
    index = {"version": 1, "meta": classic_meta(sources) | {"what": "the game blocks of runs/show/classic/<game>.json, "
                                                            "unchanged; no tracks"},
             "games": {g: docs[g]["game"] for g in games}}
    return docs, index, all_sources


# ---------------------------------------------------------------------------------------------
# slow rung (§2.2)
# ---------------------------------------------------------------------------------------------
@contextlib.contextmanager
def slow_names():
    """``scene.replay`` builds its game with ``base.make``, which does not know the slow variants
    (and ``base.py`` is not touched). For the replay only, ``scene.make`` gives the slow class and
    the instance carries its parent's name, which is all ``scene.eye_rect`` / ``state`` read;
    the rules (speed, cap) are the variant's. The frames prove it: they must equal the recorded
    slow game's frames bit for bit."""
    orig = scene.make

    def make_any(name: str):
        if name in VARIANTS:
            env = make_slow(name)
            env.name = env.parent
            return env
        return orig(name)

    scene.make = make_any
    try:
        yield
    finally:
        scene.make = orig


def slow_replay(row: str, seed: int, actions) -> dict:
    with slow_names():
        sc = scene.replay(row, seed, actions)
    sc["game"] = row
    return sc


def ladder_reading(path: Path = LADDER) -> dict:
    """The "M7g 결과" section of the ladder document: its heading and the planning session's
    reading paragraph, verbatim (markdown)."""
    lines = path.read_text(encoding="utf-8").splitlines()
    start = [i for i, ln in enumerate(lines) if ln.startswith("## M7g 결과")]
    if len(start) != 1:
        raise LookupError(f"{E3.rel(path)}: {len(start)} '## M7g 결과' headings")
    end = next((i for i in range(start[0] + 1, len(lines)) if lines[i].startswith("## ")), len(lines))
    para = [ln for ln in lines[start[0] + 1:end] if ln.startswith("**총괄 읽기.**")]
    if len(para) != 1:
        raise LookupError(f"{E3.rel(path)}: the M7g section has {len(para)} '**총괄 읽기.**' paragraphs")
    return {"heading": lines[start[0]][3:], "reading": para[0]}


def slow_title(row: str, family: str, s: int) -> str:
    return SLOW_TITLE_KO[family] + ("" if s == 1 else f" ({s}배 느리게)")


def slow_rows(res: dict) -> dict:
    top = res["hypothesis"]["top_k_table"]
    if list(res["rows"]) != list(SLOW_TABLE_ROWS) or list(top) != list(SLOW_TABLE_ROWS):
        raise RuntimeError(f"m7g-results.json rows {list(res['rows'])} are not {SLOW_TABLE_ROWS}")
    out = {}
    for row in SLOW_TABLE_ROWS:
        r = res["rows"][row]
        out[row] = {"family": r["family"], "s": r["s"], "title_ko": slow_title(row, r["family"], r["s"]),
                    "eval_seeds": res["eval_seeds"],
                    "scores": {"fly": r["fly"]["scores"], "random": r["random"]["scores"],
                               "fixed": r["fixed"]["scores"], "brainless": r["brainless"]["scores"],
                               "oracle": r["oracle"]["scores"]},
                    "fixed_ref": r["fixed"]["reference"], "ceiling": r["ceiling"], "verdict": r["verdict"],
                    "lead_ms_median_top_k": top[row]["lead_ms_median"],
                    "k": r["k"], "k_index": r["k_index"], "k_top": top[row]["k"], "brainless_theta": r["theta"],
                    "p99": r["p99"]}
    return out


def _cell(row: str, cell: str, game: str, c: dict) -> dict:
    v = c["vs_random"]
    return {"row": row, "cell": cell, "k": c["k"], "game": game, "seeds": "train 0-9", "mean": c["mean"],
            "scores": c["scores"], "random": float(np.mean(c["random_train"])), "random_scores": c["random_train"],
            "diff_mean": v["mean"], "sem": v["sem"], "beats": v["beats"], "loses": v["loses"],
            "lead_ms_median": c["lead_ms_median"], "k_index": c.get("k_index")}


def _appearance(row: str, where: str, k, a: dict) -> dict:
    return {"row": row, "where": where, "k": k, "passed": a["passed"], "within_300ms": a["passed_within"],
            "frac": a["passed_within_fraction"], "window_ms": a["window_ms"]}


def slow_posthoc(res: dict) -> dict:
    ph = res["posthoc"]
    cells, appearance, reading, reading_cells = [], [], {}, {}
    for row, p in ph["rows"].items():
        r = res["rows"][row]
        cells.append(_cell(row, "top", row, p["top_k"]))
        cells += [_cell(row, "A", row, pt) for pt in p["cell_A"]["points"]]
        cells.append(_cell(row, "B", r["family"], p["cell_B"]))
        reading[row] = p["reading"]["sentence"]
        reading_cells[row] = {"A": p["cell_A"]["verdict"], "A_read_as_beat": p["reading"]["A_read_as_beat"],
                              "B": p["reading"]["B"]}
        sel = {"top_k_train": p["top_k"]["k"], "selected_k_eval": r["k"], "selected_k_train": r["k"]}
        appearance += [_appearance(row, w, sel[w], a) for w, a in r["appearance"].items()]
        appearance += [_appearance(row, "cell_A", pt["k"], pt["appearance"]) for pt in p["cell_A"]["points"]]
        appearance.append(_appearance(row, "cell_B", p["cell_B"]["k"], p["cell_B"]["appearance"]))
    return {"posthoc": True, "note": POSTHOC_NOTE, "decided": ph["decided"], "rules": ph["rules"],
            "cells": cells, "reading_2x2": reading, "reading_2x2_cells": reading_cells,
            "appearance": appearance,
            "drive": {row: {"drive_ratio_top": p["drive_ratio_top"],
                            "appearance_current_ratio_top": p["appearance_current_ratio_top"],
                            "k_top_s1": p["k_top_s1"], "k_top": p["k_top"], "k_match": p["k_match"]}
                      for row, p in ph["rows"].items()}}


def slow_game_entry(res: dict, rows: dict) -> dict:
    r = res["rows"][SLOW_ROW]
    rules = make_slow(SLOW_ROW).describe()
    if rules != r["rules"]:
        raise RuntimeError(f"{SLOW_ROW}: the variant's rules changed since m7g-results.json")
    if r["fixed"]["reference"] not in ("LEFT", "RIGHT"):
        raise RuntimeError(f"{SLOW_ROW}: fixed reference {r['fixed']['reference']} is not one direction")
    return rows[SLOW_ROW] | {"rules": rules, "parent": rules["parent"], "speed_divisor": rules["speed_divisor"],
                             "SPEED0": rules["SPEED0"], "max_ticks": rules["max_ticks"],
                             "one_way_dir": r["fixed"]["reference"].lower()}


def slow_bot_agent(player: str, entry: dict):
    from .games.m7g import FixedAgent
    if player == "random":
        return RandomAgent()
    if player == "fixed":
        return FixedAgent(entry["fixed_ref"])
    if player == "brainless":
        return ThresholdAgent(entry["brainless_theta"])
    raise KeyError(player)


def slow_bot(player: str, entry: dict) -> dict:
    ep = run_episode(make_slow(SLOW_ROW), slow_bot_agent(player, entry), SEED, record=True)
    rec = ep["record"]
    sc = slow_replay(SLOW_ROW, SEED, rec["actions"])
    expected = entry["scores"][player][EVAL_SEEDS.index(SEED)]
    check = {"frames_bit_exact": E3.bit_exact(rec["frames"], sc), "n_frames": len(rec["frames"]),
             "score_matches": sc["score"] == ep["score"] == expected,
             "actions_match": sc["action"] == [int(a) for a in rec["actions"]] and sc["n_ticks"] == ep["ticks"],
             "crash_matches": sc["crashed"] == ep["crashed"] and sc["truncated"] == ep["truncated"]}
    theta = None
    if player == "brainless":
        theta = entry["brainless_theta"]
        check["loom_matches_bot"] = [tuple(v) for v in sc["loom"]] == [tuple(v) for v in rec["loom"]]
        check["rule_reproduces_actions"] = int(sum(a == b for a, b in zip(E3.threshold_actions(sc["loom"], theta),
                                                                            sc["action"])))
    tid = f"{SLOW_ROW}/{player}"
    _require(tid, check, sc["n_ticks"], COUNTED)
    return E3.assemble(sc, tid, player, None, theta, None, check)


def slow_fly(entry: dict) -> tuple[dict, list[str], dict]:
    """M7e3's fly track (``m7e3_show_data.fly_track``'s checks) on the display re-recording."""
    d = RERECORD_DIR
    files = [d / "frames.npz", d / "game.json", d / "run.json", d / "rerecord.json"]
    if not files[3].exists():
        raise FileNotFoundError(f"{E3.rel(d)}: no re-recording; run `python -m flysim.apps.m7e4_show_data rerecord`")
    z = np.load(files[0])
    gdoc = json.loads(files[1].read_text())
    run = json.loads(files[2].read_text())
    note = json.loads(files[3].read_text())
    actions = z["actions"].astype(np.int64)
    if gdoc["seed"] != SEED or gdoc["game"] != SLOW_ROW or note["note"] != RERECORD_NOTE:
        raise RuntimeError(f"{d}: bundle is {gdoc['game']} seed {gdoc['seed']}")
    if gdoc["bundle_ms"] != gdoc["duration_ms"] or len(actions) != gdoc["ticks"]:
        raise RuntimeError(f"{d}: the bundle is cut ({len(actions)} of {gdoc['ticks']} ticks)")
    again = compare_with_m7g(d, M7G_BUNDLE)
    if any(v is False for v in again.values()):
        raise RuntimeError(f"{d}: the re-recording no longer matches the M7g bundle: {again}")
    sc = slow_replay(SLOW_ROW, SEED, actions)
    expected = entry["scores"]["fly"][EVAL_SEEDS.index(SEED)]
    brain, bcheck = E3.brain_record(run, gdoc, sc["n_ticks"])
    decoded = E3.decode_actions(brain["gf"], brain["decoder"], sc["n_ticks"], int(gdoc["duration_ms"]))
    ends = lambda evs: [(e["t_ms"], e["type"]) for e in evs if e["type"] in ("pass", "crash")]
    check = {"frames_bit_exact": E3.bit_exact(z["frames"], sc), "n_frames": int(len(z["frames"])),
             "score_matches": sc["score"] == gdoc["score"] == expected,
             "actions_match": sc["n_ticks"] == gdoc["ticks"] == len(actions),
             "t_ms_match": bool(np.array_equal(z["t_ms"], sc["t_ms"])),
             "loom_matches_bundle": bool(np.array_equal(np.asarray(sc["loom"], dtype=np.float32), z["loom"])),
             "events_match_game_json": ends(sc["events"]) == ends(gdoc["events"]),
             "crash_matches": sc["crashed"] == gdoc["crashed"],
             "decoder_reproduces_actions": int(sum(a == b for a, b in zip(decoded, actions.tolist()))),
             **bcheck}
    tid = f"{SLOW_ROW}/m7g/fly"
    _require(tid, check, sc["n_ticks"], COUNTED)
    tr = E3.assemble(sc, tid, "fly", "m7g", None, brain, check)
    tr["display_rerecord"] = {"note": RERECORD_NOTE, "dir": E3.rel(d), "m7g_bundle": E3.rel(M7G_BUNDLE),
                              "first_ticks_equal": again["old_ticks"], "spikes_equal": again["spikes_old"]}
    return tr, [E3.rel(f) for f in files[:3]] + [E3.rel(M7G_BUNDLE / f) for f in ("frames.npz", "game.json",
                                                                                   "run.json")], note


def build_slow() -> tuple[dict, list[str], dict]:
    res = json.loads(M7G_RESULTS.read_text())
    if res["eval_seeds"] != EVAL_SEEDS:
        raise RuntimeError(f"m7g evaluation seeds {res['eval_seeds']} != {EVAL_SEEDS}")
    sources = [E3.rel(M7G_RESULTS), E3.rel(LADDER)]
    rows = slow_rows(res)
    entry = slow_game_entry(res, rows)
    t0 = time.time()
    fly, files, note = slow_fly(entry)
    tracks = [fly] + [slow_bot(p, entry) for p in ("random", "fixed", "brainless")]
    log(f"{SLOW_ROW}: " + ", ".join(f"{t['player']} {t['n_ticks']} ticks score {t['score']} "
                                    f"(bit-exact {t['check']['frames_bit_exact']})" for t in tracks)
        + f" ({time.time() - t0:.1f} s)")
    pre = res["hypothesis"]["reading"]
    doc = {"version": 1,
           "meta": base_meta(sources + files) | {
               "alignment": "as show-data.json (M7e3): state[k], eye_rect[k], loom[k] belong to frame k; "
                            "action[k] is chosen on frame k (the step to frame k+1)",
               "rows": "every value as m7g-results.json has it (rows.<variant>, hypothesis.top_k_table for "
                       "lead_ms_median_top_k); title_ko is a display label",
               "display_rerecord": RERECORD_NOTE + f" — {E3.rel(RERECORD_DIR)}; the M7g bundle keeps the first "
                                   "60 s of this 120 s game",
               "players": {"fly": "the fly (M7c2 main arm, M7g)", "random": "random bot",
                           "fixed": f"the fixed-action reference (always {entry['fixed_ref']})",
                           "brainless": "brainless bot (M7g θ)"},
               "oracle": "scores only: the cheating reference reads the game state (not a track)"},
           "rows": rows,
           "posthoc": slow_posthoc(res),
           "hypothesis": {"label": HYPOTHESIS_LABEL, **ladder_reading(),
                          "pre_registered": {"P1": pre["P1"], "P2": pre["P2"], "row": pre["row"],
                                             "sentence": pre["sentence"]},
                          "source": "docs/m7-game-ladder.md, section 'M7g 결과' (heading and the planning "
                                    "session's reading, verbatim); pre_registered from m7g-results.json"},
           "games": {SLOW_ROW: entry},
           "tracks": tracks}
    return doc, sources + files, note


# ---------------------------------------------------------------------------------------------
def shared_summary(doc: dict) -> dict:
    """For the provenance file: what the shared layer holds and, per kind, on how many ticks the
    four players' rectangles of that kind are the same (read back from the written document)."""
    ss = doc["shared_scene"]
    tracks = [decode_scene(t["scene"], ss) for t in doc["tracks"]]
    n = len(tracks[0])
    same = {k: int(sum(all([r for r in t[i] if r["kind"] == k] == [r for r in tracks[0][i] if r["kind"] == k]
                           for t in tracks[1:]) for i in range(n)))
            for k in doc["game"]["kinds"]}
    out = {"ticks": n, "kinds": ss["kinds"], "same_in_all_four_by_kind": same,
           "null_ticks": int(sum(t is None for t in ss["ticks"]))}
    if doc["game"]["name"] in SHARED_REASON:
        out["reason"] = SHARED_REASON[doc["game"]["name"]]
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="M7e4a: show-page data, round 2")
    ap.add_argument("command", choices=("rerecord", "classic", "slow", "all"))
    ap.add_argument("--games", default=",".join(CLASSIC_GAMES), help="classic: comma-separated games")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    t0 = time.time()
    if args.command == "rerecord":
        rerecord(args.device)
        return 0
    outputs, checks, sources, note, shared = [], {}, [], None, {}
    if args.command in ("classic", "all"):
        games = [g for g in args.games.split(",") if g] if args.command == "classic" else list(CLASSIC_GAMES)
        docs, index, src = build_classic(games)
        for g, d in docs.items():
            if "shared_scene" in d:
                shared[g] = shared_summary(d)
            outputs.append(write_json(d, CLASSIC_DIR / f"{g}.json"))
            checks |= {t["id"]: {"n_ticks": t["n_ticks"], "score": t["score"], "check": t["check"]} for t in d["tracks"]}
        outputs.append(write_json(index, INDEX_JSON))
        sources += src
    if args.command in ("slow", "all"):
        doc, src, note = build_slow()
        outputs.append(write_json(doc, SLOW_JSON))
        checks |= {t["id"]: {"n_ticks": t["n_ticks"], "score": t["score"], "check": t["check"]} for t in doc["tracks"]}
        sources += src
    for o in outputs:
        log(f"wrote {o['path']}: {o['bytes']:,} bytes")
    if args.command == "all":
        prov = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "git_commit": E3.git_commit(),
                "brief": "docs/m7e4-brief.md §2 and '계약 추가'", "command": "python -m flysim.apps.m7e4_show_data all",
                "sources": {p: E3.sha256(ROOT / p) for p in dict.fromkeys(sources)},
                "outputs": outputs, "size_target_bytes": SIZE_TARGET, "tracks": checks,
                "display_rerecord": {k: note[k] for k in ("note", "row", "seed", "k", "how", "m7g_bundle", "bundle",
                                                          "episode", "check", "generated", "git_commit")},
                "shared_scene": {"rule": classic_meta([])["shared_rule"], "approved": SHARED_APPROVAL,
                                 "games": shared},
                "yoked": "data-provenance/m7f-yoked.json present" if M7F_YOKED.exists() else
                         "data-provenance/m7f-yoked.json absent: yoked fields are null",
                "wall_s": round(time.time() - t0, 1)}
        PROVENANCE.write_text(json.dumps(prov, indent=1, ensure_ascii=False) + "\n")
        log(f"wrote {E3.rel(PROVENANCE)}")
    log(f"done in {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
