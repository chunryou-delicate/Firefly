"""M7e3a: scene records (games.scene) and the show page's data (apps.m7e3_show_data).

The scene tests run anywhere (game code only). The track tests need the replay bundles
(runs/m7c, runs/m7c2 — gitignored) and skip without them; they read nothing from the GPU.
"""
import base64
import json

import numpy as np
import pytest

from flysim.apps import m7e3_show_data as SD
from flysim.apps.games import scene as SC
from flysim.apps.games.agents import RandomAgent, ThresholdAgent
from flysim.apps.games.base import FRAME_H, FRAME_W, Action, make
from flysim.apps.games.brain import tick_end_ms
from flysim.apps.games.runner import run_episode

HAVE_BUNDLES = all((SD.BUNDLE_DIRS[m] / f"{g}-brain-seed10" / f).exists()
                   for m in SD.MATCHES for g in SD.GAMES for f in ("frames.npz", "game.json", "run.json"))
needs_bundles = pytest.mark.skipif(not HAVE_BUNDLES, reason="replay bundles not present (runs/ is gitignored)")


def _play(game, agent, seed):
    ep = run_episode(make(game), agent, seed, record=True)
    return ep, ep["record"]["actions"]


# ---------------------------------------------------------------------------------------------
# scene
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("game", SD.GAMES)
def test_eye_rect_redraws_every_frame(game):
    ep, actions = _play(game, ThresholdAgent(1e-4), 3)
    sc = SC.replay(game, 3, actions)
    assert sc["n_ticks"] == len(actions) == ep["ticks"] and sc["score"] == ep["score"]
    for k, (f, r) in enumerate(zip(ep["record"]["frames"], sc["eye_rect"])):
        assert np.array_equal(SC.redraw(r), f), f"frame {k}"
        assert np.array_equal(sc["frames"][k], f)
    assert any(r is not None for r in sc["eye_rect"]) and sc["eye_rect"][0] is None


def test_alignment_state_and_action_belong_to_frame_k():
    """state[k]/eye_rect[k]/loom[k] = frame k; action[k] = the step from frame k to frame k + 1."""
    _, actions = _play("dodge_v2", RandomAgent(), 4)
    sc = SC.replay("dodge_v2", 4, actions)
    env = make("dodge_v2")
    frame = env.reset(4)
    for k, a in enumerate(actions):
        assert np.array_equal(sc["frames"][k], frame)
        assert sc["state"]["x"][k] == env.x and sc["state"]["dist"][k] == env.dist
        assert sc["eye_rect"][k] == SC.eye_rect(env) and sc["t_ms"][k] == (k * 1000) // 60
        assert sc["action"][k] == a
        frame, *_ = env.step(a)
    assert sc["state"]["height"] == [None] * len(actions) == sc["state"]["airborne"]


def test_event_stamps_and_obstacles():
    ep, actions = _play("jump", ThresholdAgent(1.5848931924611142e-4), 11)
    sc = SC.replay("jump", 11, actions)
    dist = sc["state"]["dist"] + [None]
    for e in sc["events"]:
        assert e["t_ms"] == tick_end_ms(e["tick"]) == SC.frame_ms(e["tick"] + 1)
        if e["type"] == "spawn":
            assert dist[e["tick"]] is None and dist[e["tick"] + 1] is not None and e["side"] == "center"
    kinds = [e["type"] for e in sc["events"]]
    assert kinds.count("pass") == sc["score"] == ep["score"] and kinds.count("crash") == int(sc["crashed"])
    assert kinds.count("speedup") == sc["score"] // 4
    assert [o["outcome"] for o in sc["obstacles"]].count("pass") == sc["score"]
    for o in sc["obstacles"]:
        assert o["spawn_tick"] < o["reach_tick"] <= o["end_tick"] and 0 < o["loom_last300_frac"] <= 1
        w = SC.visible_width(sc["eye_rect"][o["visible2px_tick"] + 1])
        assert w >= SC.VISIBLE_PX


def test_visible2px_is_the_first_2px_frame():
    _, actions = _play("dodge", RandomAgent(), 5)
    sc = SC.replay("dodge", 5, actions)
    o = sc["obstacles"][0]
    k = o["visible2px_tick"] + 1
    assert SC.visible_width(sc["eye_rect"][k]) >= 2.0
    assert all(SC.visible_width(sc["eye_rect"][j]) < 2.0 for j in range(o["spawn_tick"] + 1, k))


def test_replay_refuses_a_wrong_number_of_actions():
    _, actions = _play("dodge", RandomAgent(), 6)
    with pytest.raises(ValueError):
        SC.replay("dodge", 6, actions + [0])
    with pytest.raises(ValueError):
        SC.replay("dodge", 6, actions[:-1])
    with pytest.raises(KeyError):
        SC.eye_rect(type("E", (), {"name": "pong", "dist": 3.0})())


def test_loom_last300_frac_definition():
    looms = [(0.0, 0.0)] * 5 + [(0.1, 1.0)] * 60          # frames 0..64
    ob = {"spawn_tick": 4, "reach_tick": 59, "reach_t_ms": tick_end_ms(59), "side": "right"}
    # approach frames 5..59; last 300 ms = frames starting >= 1000 - 300 ms -> 42..59 (18 frames)
    assert SC.loom_last300_frac(looms, ob) == pytest.approx(18 / 55)
    # both halves count: a pillar carried across the centre keeps its signal
    crossed = looms[:42] + [(1.1, 0.0)] * 23
    assert SC.loom_last300_frac(crossed, ob) == pytest.approx(18 / 55)
    assert SC.loom_last300_frac([(0.0, 0.0)] * 65, ob) is None
    assert SC.loom_last300_frac(looms, ob | {"reach_tick": None}) is None


def test_threshold_rule_and_decoder_rule_are_the_harness_rules():
    assert SD.threshold_actions([(2.0, 2.0), (2.0, 0.0), (0.0, 2.0), (1.0, 1.0)], 1.0) == [1, 3, 2, 0]
    gf = {"L": [100, 150], "R": [140]}
    dec = {"decoder": "CountWindowDecoder", "window_ms": 100.0, "min_spikes": 2}
    acts = SD.decode_actions(gf, dec, 12, SC.frame_ms(12))
    assert acts[8] == int(Action.NOOP)                      # [50, 150): L 100 only (150 is not < 150)
    assert acts[9] == int(Action.RIGHT)                     # [66, 166): L 100, 150; R 140 alone
    assert SD.decoder_rule({"decoder": "GiantFibreDecoder", "window_ms": 50.0}) == (50.0, 1)
    with pytest.raises(KeyError):
        SD.decoder_rule({"decoder": "Other"})


# ---------------------------------------------------------------------------------------------
# the show data (needs the bundles)
# ---------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def doc():
    d, _ = SD.build()
    return json.loads(json.dumps(d, ensure_ascii=False))       # what the page reads


def _tracks(doc, **kw):
    return [t for t in doc["tracks"] if all(t[k] == v for k, v in kw.items())]


@needs_bundles
def test_fly_tracks_are_bit_exact(doc):
    fly = _tracks(doc, player="fly")
    assert sorted(t["id"] for t in fly) == sorted(f"{g}/{m}/fly" for g in SD.GAMES for m in SD.MATCHES)
    for t in fly:
        c = t["check"]
        assert c["frames_bit_exact"] == c["n_frames"] == t["n_ticks"], t["id"]
        assert c["decoder_reproduces_actions"] == t["n_ticks"]
        assert all(c[k] for k in ("score_matches", "actions_match", "t_ms_match", "loom_matches_bundle",
                                  "events_match_game_json", "crash_matches", "rates_match_spike_list",
                                  "net_spikes_match_run"))
        b = t["brain"]
        assert set(b) == {"decoder", "warmup_ms", "transient_ms", "gf", "ttmn", "set_rate_hz", "net_spikes"}
        assert (b["warmup_ms"], b["transient_ms"]) == ((0, 2000) if t["match"] == "m7c" else (5000, 0))
        assert set(b["set_rate_hz"]) == set(SD.BRAIN_SETS)
        assert all(len(v) == t["n_ticks"] for v in b["set_rate_hz"].values()) and len(b["net_spikes"]) == t["n_ticks"]
        assert SD.decode_actions(b["gf"], b["decoder"], t["n_ticks"], SC.frame_ms(t["n_ticks"])) == t["action"]


@needs_bundles
def test_bot_seed10_scores_match_the_results(doc):
    res = SD.load_results()
    for game in SD.GAMES:
        g = res["m7c"]["games"][game]
        want = {"random": g["random"]["scores"][0], "one_way": g["one_way"]["scores"][0],
                "brainless": g["brainless"]["scores"][0]}
        for p, s in want.items():
            (t,) = _tracks(doc, game=game, player=p)
            assert t["score"] == s and t["check"]["score_matches"] and t["match"] is None
            assert t["check"]["frames_bit_exact"] == t["n_ticks"]
        assert doc["games"][game]["scores"]["fly_m7c"] == g["fly"]["scores"]
        assert doc["games"][game]["verdict"]["m7c"] == g["verdict"]
        assert doc["games"][game]["verdict"]["m7c2"] == res["m7c2"]["arms"]["main"]["games"][game]["verdict"]
    assert doc["games"]["dodge_v2"]["ceiling"] == 72 and doc["games"]["jump"]["ceiling"] is None
    assert doc["games"]["dodge_v2"]["one_way_dir"] == "right" and doc["games"]["dodge"]["one_way_dir"] == "left"


@needs_bundles
def test_brainless_rule_from_the_written_loom_and_theta(doc):
    """θ and loom are written exactly: the bot's rule re-applied to them gives every action."""
    for t in _tracks(doc, player="brainless"):
        assert t["theta"] == doc["games"][t["game"]]["brainless_theta"]
        assert SD.threshold_actions(t["loom"], t["theta"]) == t["action"], t["id"]
        assert t["check"]["rule_reproduces_actions"] == t["n_ticks"] and t["check"]["loom_matches_bot"]


@needs_bundles
def test_written_eye_rect_redraws_the_check_frames(doc):
    for t in doc["tracks"]:
        cf = t["check_frames"]
        assert len(cf["ticks"]) == len(cf["raw_b64"]) >= 2
        for k, raw in zip(cf["ticks"], cf["raw_b64"]):
            f = np.frombuffer(base64.b64decode(raw), dtype=np.uint8).reshape(FRAME_H, FRAME_W)
            assert np.array_equal(SC.redraw(t["eye_rect"][k]), f), (t["id"], k)


@needs_bundles
def test_contract_shape(doc):
    assert doc["version"] == 1 and doc["meta"]["action_codes"] == {"0": "NOOP", "1": "JUMP", "2": "LEFT", "3": "RIGHT"}
    assert doc["meta"]["tick_ms_rule"] == "t_ms(k) = floor(k*1000/60)"
    assert len(doc["tracks"]) == 15 and set(doc["games"]) == set(SD.GAMES)
    for g in doc["games"].values():
        assert g["eval_seeds"] == list(range(10, 20))
        assert set(g["scores"]) == {"fly_m7c", "fly_m7c2", "random", "one_way", "brainless"}
        assert all(len(v) == 10 for v in g["scores"].values())
    for t in doc["tracks"]:
        n = t["n_ticks"]
        assert len(t["t_ms"]) == len(t["action"]) == len(t["loom"]) == len(t["eye_rect"]) == n
        assert all(len(v) == n for v in t["state"].values())
        assert t["seed"] == 10 and t["player"] in ("fly", "random", "one_way", "brainless")
        for k in range(n):
            if t["state"]["dist"][k] is None:
                assert t["eye_rect"][k] is None and t["state"]["obj_x"][k] is None
        none_keys = {"jump": ("x", "obj_x"), "dodge": ("height", "airborne"), "dodge_v2": ("height", "airborne")}
        for key in none_keys[t["game"]]:
            assert t["state"][key] == [None] * n
        assert {e["type"] for e in t["events"]} <= {"spawn", "visible2px", "pass", "crash", "speedup"}
        assert all(o["outcome"] in ("pass", "crash", "cut") for o in t["obstacles"])
        assert (t["theta"] is None) == (t["player"] != "brainless") and (t["brain"] is None) == (t["player"] != "fly")


@needs_bundles
def test_same_input_same_output():
    ids = ["dodge_v2/m7c2/fly", "jump/random", "dodge/brainless"]
    a, sa = SD.build(ids)
    b, sb = SD.build(ids)
    for d in (a, b):
        d["meta"].pop("generated")
    dump = lambda d: json.dumps(d, ensure_ascii=False, separators=(",", ":"))
    assert dump(a) == dump(b) and sa == sb


def test_committed_provenance():
    if not SD.PROVENANCE.exists():
        pytest.skip("show-data provenance not generated yet")
    p = json.loads(SD.PROVENANCE.read_text())
    assert sorted(p["tracks"]) == sorted(SD.track_ids())
    for tid, t in p["tracks"].items():
        assert t["check"]["frames_bit_exact"] == t["n_ticks"], tid
        assert all(v is not False for v in t["check"].values()), tid
    assert len(p["sources"]) == 3 + 18 and all(len(h) == 64 for h in p["sources"].values())
    assert len(p["output"]["sha256"]) == 64
