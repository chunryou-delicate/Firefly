"""M7e4a tests: the show page's second-round data (apps.m7e4_show_data).

The replay, scene-encoding, shared-layer and slow-replay tests run anywhere (game code only). The
file tests read runs/show/ and the replay bundles (gitignored) and skip without them; the result
and ladder tests read committed files. Nothing here touches the GPU.
"""
import base64
import json

import numpy as np
import pytest

from flysim.apps import m7e3_show_data as E3
from flysim.apps import m7e4_show_data as M
from flysim.apps.games import scene as SC
from flysim.apps.games.agents import RandomAgent, ThresholdAgent
from flysim.apps.games.base import FRAME_H, FRAME_W, Action, make
from flysim.apps.games.classic.common import draw
from flysim.apps.games.m7f import ConstAgent
from flysim.apps.games.runner import run_episode
from flysim.apps.games.slow import make_slow

CLASSIC_FILES = {g: M.CLASSIC_DIR / f"{g}.json" for g in M.CLASSIC_GAMES}
HAVE_CLASSIC = all(p.exists() for p in CLASSIC_FILES.values()) and M.INDEX_JSON.exists()
HAVE_SLOW = M.SLOW_JSON.exists()
HAVE_RERECORD = all((M.RERECORD_DIR / f).exists() for f in ("frames.npz", "game.json", "run.json", "rerecord.json")) \
    and all((M.M7G_BUNDLE / f).exists() for f in ("frames.npz", "game.json", "run.json"))
needs_classic = pytest.mark.skipif(not HAVE_CLASSIC, reason="runs/show/classic/ not built (runs/ is gitignored)")
needs_slow = pytest.mark.skipif(not HAVE_SLOW, reason="runs/show/slow.json not built (runs/ is gitignored)")
needs_rerecord = pytest.mark.skipif(not HAVE_RERECORD, reason="the dodge_v2_s4 bundles are not present")


def _load(path):
    return json.loads(path.read_text())


def _frame(b64: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(b64), dtype=np.uint8).reshape(FRAME_H, FRAME_W)


def _scene(*rects):
    return {"rects": [{"kind": k, "rect": [float(a), float(b), float(c), float(d)], "value": float(v)}
                      for k, a, b, c, d, v in rects], "hud": {"score": 0, "lives": None, "tick": 0}}


# ---------------------------------------------------------------------------------------------
# classic replay and scene encoding (game code only)
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("game", ["paddle", "formation"])
def test_classic_replay_redraws_every_frame_and_ends_on_the_last_action(game):
    ep = run_episode(make(game), RandomAgent(), 3, record=True)
    rec = ep["record"]
    sc = M.classic_replay(game, 3, rec["actions"])
    assert sc["n_ticks"] == ep["ticks"] == len(rec["actions"]) and sc["score"] == ep["score"]
    assert [s["hud"]["score"] for s in sc["scenes"]] == [int(v) for v in rec["scores"]]
    for k, (f, s) in enumerate(zip(rec["frames"], sc["scenes"])):
        assert np.array_equal(sc["frames"][k], f) and np.array_equal(draw(s["rects"]), f), k
    with pytest.raises(ValueError):
        M.classic_replay(game, 3, rec["actions"] + [0])          # actions left over
    with pytest.raises(ValueError):
        M.classic_replay(game, 3, rec["actions"][:-1])           # the game goes on


def test_classic_alignment_scene_k_draws_frame_k_then_action_k():
    env = make("boxing")
    frame = env.reset(5)
    actions = [int(a) for a in np.random.default_rng(0).integers(0, 4, size=3600)]
    sc = M.classic_replay("boxing", 5, actions)
    for k, a in enumerate(actions[:400]):
        assert sc["scenes"][k] == env.scene() and np.array_equal(sc["frames"][k], frame)
        frame, *_ = env.step(a)
    for e in sc["events"]:
        assert e["t_ms"] == SC.frame_ms(e["tick"] + 1) and e["type"] in ("score", "life_lost", "game_over")
        if e["type"] == "score" and e["tick"] + 1 < len(actions):
            assert sc["scenes"][e["tick"] + 1]["hud"]["score"] == e["score"]


def test_events_follow_the_hud():
    ep = run_episode(make("catch"), RandomAgent(), 4, record=True)
    sc = M.classic_replay("catch", 4, ep["record"]["actions"])
    lives = [s["hud"]["lives"] for s in sc["scenes"]]
    lost = [e for e in sc["events"] if e["type"] == "life_lost"]
    assert lost and len(lost) == sum(b < a for a, b in zip(lives, lives[1:])) + int(sc["crashed"])
    for e in lost:
        assert e["lives"] < lives[e["tick"]]
        assert e["tick"] + 1 == len(lives) or lives[e["tick"] + 1] == e["lives"]
    assert [e["type"] for e in sc["events"]].count("game_over") == int(sc["crashed"])


def test_scene_encoding_static_null_rule_and_round_trip():
    scenes = [_scene(("wall", 0, 1, 0, 84, 110), ("ball", 10, 12, 10, 12, 25)),
              _scene(("wall", 0, 1, 0, 84, 110), ("ball", 10, 12, 10, 12, 25)),
              _scene(("wall", 0, 1, 0, 84, 110), ("ball", 11, 13, 10, 12, 25), ("paddle", 30, 40, 80, 82, 25)),
              _scene(("wall", 0, 1, 0, 84, 110))]
    enc = M.encode_scene(scenes, ("paddle", "ball", "wall"))
    assert enc["kinds"] == ["wall", "ball", "paddle"]                       # first appearance
    assert enc["static"] == [[0, 0.0, 1.0, 0.0, 84.0, 110.0]]
    assert enc["ticks"][0] == [[1, 10.0, 12.0, 10.0, 12.0, 25.0]] and enc["ticks"][1] is None
    assert enc["ticks"][3] == [] and "shared" not in enc
    back = M.decode_scene(json.loads(json.dumps(enc)))
    assert back == [s["rects"] for s in scenes]
    with pytest.raises(KeyError):
        M.encode_scene(scenes, ("ball", "wall"))                           # a kind the game does not have
    with pytest.raises(ValueError):
        M.decode_scene({"kinds": [], "static": [], "ticks": [None]})


def test_floats_are_not_rounded():
    x = 47.95215728586579
    enc = M.encode_scene([_scene(("car", x, x + 8, 72, 75, 25))], ("car",))
    assert json.loads(json.dumps(enc))["static"][0][1:3] == [x, x + 8]      # one tick: all static


# ---------------------------------------------------------------------------------------------
# the shared layer (planning session, 2026-10-31)
# ---------------------------------------------------------------------------------------------
def _players():
    lane = ("lane", 0, 84, 10, 10.5, 110)
    cars = [("car", 5 + k, 13 + k, 20, 23, 25) for k in range(3)]
    a = [_scene(lane, cars[k], ("walker", 30, 33, 79, 82, 25)) for k in range(3)]   # walker never moves
    b = [_scene(lane, cars[k], ("walker", 30, 33, 79 - k, 82 - k, 25)) for k in range(3)]
    return a, b


def test_shared_layer_is_the_common_run_after_static():
    a, b = _players()
    shared = M.shared_layer([a, b])
    assert shared == [[("car", 5.0, 13.0, 20.0, 23.0, 25.0), ("walker", 30.0, 33.0, 79.0, 82.0, 25.0)],
                      [("car", 6.0, 14.0, 20.0, 23.0, 25.0)], [("car", 7.0, 15.0, 20.0, 23.0, 25.0)]]
    ss = M.encode_shared(shared, ("lane", "car", "walker"))
    for scenes in (a, b):
        enc = M.encode_scene(scenes, ("lane", "car", "walker"), shared)
        assert enc["shared"] is True and enc["static"] == [[0, 0.0, 84.0, 10.0, 10.5, 110.0]]
        assert M.decode_scene(json.loads(json.dumps(enc)), json.loads(json.dumps(ss))) == [s["rects"] for s in scenes]
        with pytest.raises(ValueError):
            M.decode_scene(enc)                                            # shared but no shared_scene


def test_no_shared_layer_when_lengths_differ_or_a_tick_has_nothing_in_common():
    a, b = _players()
    assert M.shared_layer([a, b[:2]]) is None
    c = [s if k != 1 else _scene(("lane", 0, 84, 10, 10.5, 110), ("car", 99, 107, 20, 23, 25)) for k, s in enumerate(b)]
    assert M.shared_layer([a, c]) is None
    with pytest.raises(ValueError):
        M.encode_scene(a, ("lane", "car", "walker"), [[("car", 0.0, 1.0, 2.0, 3.0, 25.0)]] * 3)


def test_crossing_cars_depend_only_on_seed_and_tick():
    """The reason crossing shares a layer: the cars of two players with different actions are the
    same at every tick of the 60 s game."""
    runs = []
    for agent in (RandomAgent(), ConstAgent(int(Action.JUMP))):
        ep = run_episode(make("crossing"), agent, 10, record=True)
        sc = M.classic_replay("crossing", 10, ep["record"]["actions"])
        runs.append([[r for r in s["rects"] if r["kind"] == "car"] for s in sc["scenes"]])
    assert len(runs[0]) == len(runs[1]) == 3600 and runs[0] == runs[1]
    assert all(len(c) == 26 for c in runs[0])


def test_check_ticks():
    assert M.check_ticks(3600) == [10, 1800, 3590]
    assert M.check_ticks(5) == [0, 2, 4] and M.check_ticks(1) == [0]


# ---------------------------------------------------------------------------------------------
# slow replay (scene.replay wrapped; scene.py untouched)
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("row,agent", [("dodge_v2_s4", RandomAgent()), ("jump_s2", ThresholdAgent(2.5e-4))])
def test_slow_replay_is_the_slow_game(row, agent):
    ep = run_episode(make_slow(row), agent, 7, record=True)
    rec = ep["record"]
    sc = M.slow_replay(row, 7, rec["actions"])
    assert sc["game"] == row and sc["n_ticks"] == ep["ticks"] and sc["score"] == ep["score"]
    assert E3.bit_exact(rec["frames"], sc) == ep["ticks"]
    assert SC.make.__module__ == "flysim.apps.games.base"                 # restored
    env, parent = make_slow(row), make(make_slow(row).parent)
    env.reset(7), parent.reset(7)
    assert sc["state"]["speed"][0] == env.speed == parent.speed / env.speed_divisor


def test_slow_replay_restores_scene_make_on_error():
    with pytest.raises(ValueError):
        M.slow_replay("dodge_v2_s4", 7, [0])                             # the game goes on
    assert SC.make.__module__ == "flysim.apps.games.base"


# ---------------------------------------------------------------------------------------------
# result files and the ladder (committed)
# ---------------------------------------------------------------------------------------------
def test_classic_entry_copies_the_result_files():
    base, res, yoked, _ = M.load_classic_results()
    for g in M.CLASSIC_GAMES:
        e = M.classic_entry(g, base, res, yoked)
        assert e["rules"] == make(g).describe() and e["title_ko"] == make(g).title_ko
        assert e["scores"]["fly"] == res["games"][g]["fly"]["scores"]
        assert e["scores"]["oracle"] == res["bots"][g]["oracle"]["scores"]
        assert e["verdict"]["vs_fixed"] == res["games"][g]["verdict"]["vs_fixed"]
        assert e["fixed_ref"] == base["games"][g]["fixed"]["reference"] and e["k"] == res["games"][g]["k"]
        if yoked is not None:
            assert e["yoked"]["posthoc"] is True and e["yoked"]["note"] == "사후 점검 — 판정 아님"
            assert e["scores"]["yoked"] == yoked["games"][g]["yoked"]["scores"]
            assert len(e["yoked"]["matrix_tape_x_game"]) == 10
    boxing = M.classic_entry("boxing", base, res, yoked)
    assert boxing["verdict"]["vs_random"]["verdict"] == "beat" and boxing["genre_ko"] == "권투형"
    if yoked is not None:
        assert boxing["yoked"]["yoked"]["mean"] == pytest.approx(1.5333, abs=1e-4)   # addendum 2's check


def test_classic_entry_refuses_disagreeing_files():
    base, res, yoked, _ = M.load_classic_results()
    bad = json.loads(json.dumps(res))
    bad["bots"]["boxing"]["random"]["scores"][0] += 1
    with pytest.raises(RuntimeError):
        M.classic_entry("boxing", base, bad, yoked)


def test_ladder_reading_is_verbatim():
    r = M.ladder_reading()
    text = M.LADDER.read_text(encoding="utf-8")
    assert r["heading"].startswith("M7g 결과") and f"## {r['heading']}\n" in text
    assert r["reading"].startswith("**총괄 읽기.**") and f"\n{r['reading']}\n" in text


def test_ladder_reading_raises_without_the_section(tmp_path):
    p = tmp_path / "ladder.md"
    p.write_text("## M7f 결과\n\n**총괄 읽기.** 다른 절\n", encoding="utf-8")
    with pytest.raises(LookupError):
        M.ladder_reading(p)


def test_slow_rows_and_posthoc_come_from_m7g_results():
    res = _load(M.M7G_RESULTS)
    rows = M.slow_rows(res)
    assert list(rows) == list(M.SLOW_TABLE_ROWS)
    for row, r in rows.items():
        src = res["rows"][row]
        assert r["scores"]["fly"] == src["fly"]["scores"] and r["scores"]["oracle"] == src["oracle"]["scores"]
        assert r["verdict"] == src["verdict"] and r["ceiling"] == src["ceiling"]
        assert r["lead_ms_median_top_k"] == res["hypothesis"]["top_k_table"][row]["lead_ms_median"]
    assert [rows[r]["lead_ms_median_top_k"] for r in ("dodge_v2", "dodge_v2_s2", "dodge_v2_s4", "dodge_v2_s8")] == \
        [117.0, 1883.5, 5967.0, 17550.0]
    assert rows["dodge_v2_s4"]["title_ko"] == "벽 없는 피하기 (4배 느리게)" and rows["jump"]["title_ko"] == "뛰어넘기"
    ph = M.slow_posthoc(res)
    assert ph["posthoc"] is True and ph["note"] == "사후 점검 — 판정 아님"
    n_a = sum(len(p["cell_A"]["points"]) for p in res["posthoc"]["rows"].values())
    assert len(ph["cells"]) == 2 * len(res["posthoc"]["rows"]) + n_a
    b = next(c for c in ph["cells"] if c["row"] == "dodge_v2_s8" and c["cell"] == "B")
    assert b["game"] == "dodge_v2" and b["mean"] == 32.6 and b["beats"] is True
    assert ph["reading_2x2"]["dodge_v2_s4"].startswith("전류가 열쇠다")
    top = next(a for a in ph["appearance"] if a["row"] == "dodge_v2_s8" and a["where"] == "top_k_train")
    assert top["frac"] == pytest.approx(0.83, abs=0.01)


# ---------------------------------------------------------------------------------------------
# the built files (runs/show, gitignored)
# ---------------------------------------------------------------------------------------------
@needs_classic
@pytest.mark.parametrize("game", M.CLASSIC_GAMES)
def test_classic_file(game):
    d = _load(CLASSIC_FILES[game])
    g = d["game"]
    assert d["version"] == 1 and g["name"] == game and CLASSIC_FILES[game].stat().st_size <= M.SIZE_TARGET
    assert d["meta"]["values"] == {"background": 190.0, "fixed": 110.0, "moving": 25.0}
    assert [t["player"] for t in d["tracks"]] == list(M.CLASSIC_PLAYERS)
    ss = d.get("shared_scene")
    assert (ss is not None) == (game in ("crossing", "paddle"))
    for t in d["tracks"]:
        n = t["n_ticks"]
        assert t["seed"] == 10 and t["score"] == g["scores"][t["player"]][0]
        assert len(t["action"]) == len(t["loom"]) == len(t["scene"]["ticks"]) == len(t["hud"]["score"]) == n
        assert t["check"]["frames_bit_exact"] == t["check"]["scene_reads_back"] == n
        assert bool(t["scene"].get("shared")) == (ss is not None)
        rects = M.decode_scene(t["scene"], ss)
        for k, raw in zip(t["check_frames"]["ticks"], t["check_frames"]["raw_b64"]):
            assert np.array_equal(draw(rects[k]), _frame(raw)), (t["id"], k)
        assert (t["brain"] is not None) == (t["player"] == "fly") and (t["theta"] is not None) == (t["player"] == "brainless")
        for e in t["events"]:
            if e["type"] == "score" and e["tick"] + 1 < n:
                assert t["hud"]["score"][e["tick"] + 1] == e["score"]
    assert _load(M.INDEX_JSON)["games"][game] == g


@needs_classic
@pytest.mark.parametrize("game", M.CLASSIC_GAMES)
def test_classic_fly_track_is_scene_json(game):
    d = _load(CLASSIC_FILES[game])
    sdoc = _load(M.M7F_DIR / f"{game}-brain-seed10" / "scene.json")
    t = d["tracks"][0]
    rects = M.decode_scene(t["scene"], d.get("shared_scene"))
    assert rects == [s["rects"] for s in sdoc["scenes"]]
    assert t["hud"]["score"] == [s["hud"]["score"] for s in sdoc["scenes"]]
    assert t["hud"]["lives"] == [s["hud"]["lives"] for s in sdoc["scenes"]]
    z = np.load(M.M7F_DIR / f"{game}-brain-seed10" / "frames.npz")
    assert t["action"] == z["actions"].astype(int).tolist()
    assert np.array_equal(np.asarray(t["loom"], dtype=np.float32), z["loom"])
    assert t["brain"]["warmup_ms"] == 5000 and t["brain"]["transient_ms"] == 0


@needs_classic
def test_crossing_shares_the_cars_of_all_four_players():
    d = _load(CLASSIC_FILES["crossing"])
    tracks = [M.decode_scene(t["scene"], d["shared_scene"]) for t in d["tracks"]]
    cars = [[[r for r in L if r["kind"] == "car"] for L in rects] for rects in tracks]
    assert all(len(c) == 3600 for c in cars) and all(c == cars[0] for c in cars)
    assert "car" in d["shared_scene"]["kinds"]


@needs_classic
def test_index_holds_the_game_blocks_only():
    idx = _load(M.INDEX_JSON)
    assert list(idx["games"]) == list(M.CLASSIC_GAMES) and "tracks" not in idx
    for g in M.CLASSIC_GAMES:
        assert idx["games"][g] == _load(CLASSIC_FILES[g])["game"]


@needs_slow
def test_slow_file():
    d = _load(M.SLOW_JSON)
    res = _load(M.M7G_RESULTS)
    assert d["rows"] == M.slow_rows(res) and d["posthoc"] == M.slow_posthoc(res)
    assert d["hypothesis"]["label"] == "가설 — 약하게만 지지됨(느린 판 실험)"
    assert d["hypothesis"]["reading"] == M.ladder_reading()["reading"]
    g = d["games"]["dodge_v2_s4"]
    assert g["rules"] == make_slow("dodge_v2_s4").describe() and g["speed_divisor"] == 4 and g["parent"] == "dodge_v2"
    want_keys = ["id", "game", "player", "match", "seed", "n_ticks", "score", "crashed", "truncated", "t_ms",
                 "action", "loom", "state", "eye_rect", "events", "obstacles", "theta", "brain", "check",
                 "check_frames"]
    assert [t["player"] for t in d["tracks"]] == ["fly", "random", "fixed", "brainless"]
    for t in d["tracks"]:
        assert [k for k in t if k != "display_rerecord"] == want_keys
        assert t["score"] == res["rows"]["dodge_v2_s4"][t["player"]]["scores"][0] == g["scores"][t["player"]][0]
        assert t["check"]["frames_bit_exact"] == t["n_ticks"]
        for k, raw in zip(t["check_frames"]["ticks"], t["check_frames"]["raw_b64"]):
            assert np.array_equal(SC.redraw(t["eye_rect"][k]), _frame(raw))
    fly = d["tracks"][0]
    assert fly["n_ticks"] == 7200 and fly["score"] == 14 and fly["match"] == "m7g"
    assert fly["display_rerecord"]["note"] == M.RERECORD_NOTE and fly["brain"]["transient_ms"] == 0
    assert fly["check"]["decoder_reproduces_actions"] == 7200


@needs_rerecord
def test_rerecording_equals_the_m7g_bundle():
    c = M.compare_with_m7g()
    assert c["old_ticks"] == 3600 and c["new_ticks"] == 7200
    assert all(v is True for k, v in c.items() if k.endswith("_equal") or k == "game_json_equal_except_length")
    assert c["spikes_old"] == c["spikes_new_first_part"] > 0
    note = _load(M.RERECORD_DIR / "rerecord.json")
    want = next(e for e in _load(M.M7G_RESULTS)["rows"]["dodge_v2_s4"]["fly"]["episodes"] if e["seed"] == 10)
    assert all(note["episode"][k] == want[k] for k in M.EPISODE_KEYS)


@pytest.mark.skipif(not all((M.M7F_DIR / f"{g}-brain-seed10" / "scene.json").exists() for g in ("paddle", "boxing")),
                    reason="the M7f replay bundles are not present")
def test_build_is_deterministic_and_checked():
    a, ia, _ = M.build_classic(["paddle", "boxing"])
    b, ib, _ = M.build_classic(["paddle", "boxing"])
    for doc in [ia, ib] + [a[g] for g in a] + [b[g] for g in b]:
        doc["meta"].pop("generated")
    assert json.dumps(a) == json.dumps(b) and json.dumps(ia) == json.dumps(ib)
    for t in a["boxing"]["tracks"] + a["paddle"]["tracks"]:
        assert t["check"]["frames_bit_exact"] == t["n_ticks"]
    assert "shared_scene" in a["paddle"] and "shared_scene" not in a["boxing"]
