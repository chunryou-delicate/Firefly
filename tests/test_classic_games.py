"""M7f tests: the eight hand-made classic-style games (docs/m7f-brief.md §1-3).

Per game: determinism, 84x84 uint8, the scene record redraws every frame, the 3,600-tick cap,
no trademark in names / titles / describe() / source, the action meanings of the brief's table,
and the automatic "fire button". Plus the part-1 helpers and, when present, the committed
baselines. CPU only, no connectome.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from flysim.apps.games import base as B
from flysim.apps.games import m7f
from flysim.apps.games.base import Action, make
from flysim.apps.games.classic import CLASSIC, GAMES
from flysim.apps.games.classic import common as C
from flysim.apps.games.classic import crossing as XING

ROOT = Path(__file__).resolve().parents[1]
CLASSIC_DIR = ROOT / "flysim" / "apps" / "games" / "classic"

BRIEF_ORDER = ("paddle", "bricks", "formation", "crossing", "catch", "demons", "ski", "boxing")
# the brief's table (§2): kinds, what JUMP (= UP) does, lives
KINDS = {"paddle": {"paddle", "cpu_paddle", "ball", "wall"}, "bricks": {"paddle", "ball", "brick", "wall"},
         "formation": {"cannon", "alien", "shot", "bomb"}, "crossing": {"walker", "car", "lane"},
         "catch": {"bucket", "bomber", "bomb"}, "demons": {"cannon", "demon", "shot", "bolt"},
         "ski": {"skier", "flag", "tree"}, "boxing": {"boxer", "cpu_boxer", "ring"}}
FIXED_KINDS = {"wall", "brick", "lane", "ring"}           # 110; everything else moves (25)
JUMP_MOVES_UP = {"crossing", "boxing"}
LIVES = {"paddle", "bricks", "formation", "catch", "demons"}
CLOCK_ONLY = {"crossing", "ski", "boxing"}
# the originals' full titles (addendum 2). "Boxing" is also the generic word the brief keeps as
# that game's id, so it cannot be searched for without matching the id itself.
TRADEMARKS = ("pong", "breakout", "space invaders", "spaceinvaders", "invaders", "freeway", "kaboom",
              "demon attack", "demonattack", "skiing")
# where each game keeps the player's x (LEFT / RIGHT check)
PLAYER_X = {"paddle": lambda e: e.px, "bricks": lambda e: e.px, "formation": lambda e: e.cx,
            "catch": lambda e: e.bx, "demons": lambda e: e.cx, "ski": lambda e: e.sx,
            "boxing": lambda e: e.p[0]}


def run(game, actions, seed):
    env = make(game)
    frames = [env.reset(seed)]
    for a in actions:
        if env.done:
            break
        f, _, _, _ = env.step(int(a))
        frames.append(f)
    return env, frames


def random_actions(n, seed=123):
    return np.random.default_rng(seed).integers(0, 4, size=n).tolist()


def test_the_brief_list_and_registry():
    assert GAMES == BRIEF_ORDER
    for g in GAMES:
        env = make(g)
        assert isinstance(env, C.ClassicGame) and env.name == g
    for g in ("jump", "dodge", "dodge_v2"):                 # the first rung is still there
        make(g)
    with pytest.raises(KeyError):
        make("invaders")


@pytest.mark.parametrize("game", GAMES)
def test_frame_format(game):
    env = make(game)
    f = env.reset(0)
    assert f.dtype == np.uint8 and f.shape == (84, 84)
    for a in random_actions(50):
        f, r, done, info = env.step(a)
        assert f.dtype == np.uint8 and f.shape == (84, 84) and isinstance(r, float)
    assert env.n_actions == 4 and env.action_meanings == ("NOOP", "JUMP", "LEFT", "RIGHT")


@pytest.mark.parametrize("game", GAMES)
def test_deterministic_and_seeded(game):
    acts = random_actions(900)
    e1, f1 = run(game, acts, 5)
    e2, f2 = run(game, acts, 5)
    assert len(f1) == len(f2) and all(np.array_equal(a, b) for a, b in zip(f1, f2))
    assert e1.score == e2.score and e1.tick == e2.tick
    _, f3 = run(game, acts, 6)
    assert any(not np.array_equal(a, b) for a, b in zip(f1, f3))         # the seed matters


@pytest.mark.parametrize("game", GAMES)
@pytest.mark.parametrize("policy", ["random", "oracle"])
def test_scene_record_redraws_every_frame(game, policy):
    """draw(scene) after a JSON round trip equals the frame, tick by tick; kinds and gray levels
    follow the brief's table and the house style."""
    env = make(game)
    frame = env.reset(2)
    acts = iter(random_actions(1500, seed=9))
    kinds_seen = set()
    for _ in range(1500):
        sc = json.loads(json.dumps(env.scene()))
        assert np.array_equal(C.draw(sc["rects"]), frame)
        assert sc["hud"] == {"score": env.score, "lives": env.lives(), "tick": env.tick}
        for r in sc["rects"]:
            assert r["kind"] in KINDS[game]
            assert r["value"] == (C.FIXED if r["kind"] in FIXED_KINDS else C.MOVING)
            kinds_seen.add(r["kind"])
        if env.done:
            break
        a = next(acts) if policy == "random" else env.oracle()
        frame, _, _, _ = env.step(a)
    assert kinds_seen <= KINDS[game] and set(env.kinds) == KINDS[game]


def test_house_style_values():
    assert (C.BACKGROUND, C.FIXED, C.MOVING) == (B.SKY, B.GROUND, B.OBJECT) == (190.0, 110.0, 25.0)
    assert np.all(C.canvas() == 190.0)


@pytest.mark.parametrize("game", GAMES)
def test_cap_and_lives(game):
    env = make(game)
    env.reset(0)
    assert env.max_ticks == 3600
    assert env.lives() == (3 if game in LIVES else None)
    assert env.describe()["lives"] == (3 if game in LIVES else None)
    done, info = False, {}
    while not done:
        _, _, done, info = env.step(int(Action.NOOP))
    assert info["tick"] <= 3600
    if game in CLOCK_ONLY:
        assert info["tick"] == 3600 and info["truncated"] and not info["crashed"]


@pytest.mark.parametrize("game", GAMES)
def test_no_trademark(game):
    env = make(game)
    text = json.dumps(env.describe(), ensure_ascii=False).lower() + " " + game + " " + env.title_ko
    for tm in TRADEMARKS:
        assert tm not in text
    assert type(env).__name__.lower().replace("game", "") not in TRADEMARKS


def test_no_trademark_in_source():
    for path in sorted(CLASSIC_DIR.glob("*.py")):
        src = path.read_text().lower()
        for tm in TRADEMARKS:
            assert tm not in src, f"{tm!r} in {path.name}"


@pytest.mark.parametrize("game", GAMES)
def test_action_meanings(game):
    env = make(game)
    acts = env.describe()["actions"]
    assert acts["NOOP"] == "nothing"
    if game in JUMP_MOVES_UP:
        assert "up" in acts["JUMP"]
    else:
        assert acts["JUMP"] == "nothing"
        # JUMP does nothing: holding it is the same game as holding nothing
        _, fj = run(game, [int(Action.JUMP)] * 600, 4)
        _, fn = run(game, [int(Action.NOOP)] * 600, 4)
        assert len(fj) == len(fn) and all(np.array_equal(a, b) for a, b in zip(fj, fn))
    if game == "crossing":
        for a in (Action.LEFT, Action.RIGHT):
            _, fa = run(game, [int(a)] * 600, 4)
            _, fn = run(game, [int(Action.NOOP)] * 600, 4)
            assert all(np.array_equal(x, y) for x, y in zip(fa, fn))
        env.reset(4)
        y0 = env.wy
        env.step(int(Action.JUMP))
        assert env.wy == y0 - XING.WALK_SPEED
        return
    x = PLAYER_X[game]
    env.reset(4)
    x0 = x(env)
    env.step(int(Action.LEFT))
    assert x(env) < x0
    env.reset(4)
    env.step(int(Action.RIGHT))
    assert x(env) > x0
    if game == "boxing":
        env.reset(4)
        y0 = env.p[1]
        env.step(int(Action.JUMP))
        assert env.p[1] < y0                       # up; there is no action that moves down


@pytest.mark.parametrize("game,check", [
    ("paddle", lambda e: e.ball is not None), ("bricks", lambda e: e.ball is not None),
    ("formation", lambda e: e.shot is not None), ("demons", lambda e: e.shot is not None),
    ("catch", lambda e: len(e.bombs) > 0)])
def test_fire_button_happens_by_itself(game, check):
    """The serve / shot / round start comes without any action."""
    env = make(game)
    env.reset(1)
    seen = False
    for _ in range(200):
        env.step(int(Action.NOOP))
        seen = seen or check(env)
    assert seen


def test_boxing_punches_by_itself_on_the_period():
    from flysim.apps.games.classic import boxing as BX
    env = make("boxing")
    env.reset(0)
    landed = []
    for _ in range(900):
        t = env.tick
        _, r, _, _ = env.step(env.oracle())
        if r > 0:
            landed.append(t)
    assert landed and all((t + 1) % BX.PUNCH_EVERY == 0 for t in landed)
    assert env.cpu_hits == env.score                  # symmetric conditions: both land together


def test_crossing_geometry():
    """Addendum 1: a car-free place between two lanes' cars, taller than the walker."""
    assert XING.LANE_H - XING.CAR_H > XING.WALKER_H
    assert (XING.LANE_H, XING.ROAD_TOP, XING.CAR_H, XING.WALKER_W, XING.WALKER_H, XING.WALKER_Y0) == \
        (7.0, 7.0, 3.0, 3.0, 3.0, 79.0)
    assert XING.ROAD_TOP + XING.LANES * XING.LANE_H < XING.WALKER_Y0      # starts below the road


@pytest.mark.parametrize("game", GAMES)
def test_describe_is_complete_json(game):
    d = make(game).describe()
    json.dumps(d)
    assert d["game"] == game and d["max_ticks"] == 3600 and d["constants"]
    assert d["title_ko"] and d["genre_ko"] and d["automatic"] and d["score"] and d["ends"]
    assert "FRAME_W" not in d["constants"]


def test_titles():
    titles = {g: make(g).title_ko for g in GAMES}
    assert titles == {"paddle": "공 받아치기", "bricks": "벽돌 깨기", "formation": "외계 편대 막기",
                      "crossing": "길 건너기", "catch": "폭탄 받기", "demons": "악마 물리치기",
                      "ski": "스키 활강", "boxing": "권투"}


# ---------------------------------------------------------------------------------------------
# part-1 helpers
# ---------------------------------------------------------------------------------------------
def test_fixed_reference_tie_order():
    rows = {"NOOP": {"mean": 1.0, "scores": [1]}, "JUMP": {"mean": 3.0, "scores": [3]},
            "LEFT": {"mean": 3.0, "scores": [3]}, "RIGHT": {"mean": 2.0, "scores": [2]}}
    assert m7f.fixed_reference(rows)["reference"] == "JUMP"


def test_survey_stats():
    a = np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.3]])
    s = m7f.survey_stats([a])
    assert s["frames"] == 3 and s["nonzero_fraction"] == pytest.approx(2 / 6)
    assert s["mean_left"] == pytest.approx(0.1 / 3) and s["mean_right"] == pytest.approx(0.1)
    assert m7f.survey_stats([np.zeros((4, 2))])["p99_nonzero"] is None


def test_tune_refuses_after_part1(tmp_path, monkeypatch):
    fake = tmp_path / "baselines.json"
    fake.write_text("{}")
    monkeypatch.setattr(m7f, "BASELINES", fake)
    with pytest.raises(RuntimeError):
        m7f.tune(["paddle"], 1, "test")


def test_const_agent():
    assert [m7f.ConstAgent(a).act(None) for a in Action] == [0, 1, 2, 3]
    assert m7f.ConstAgent(Action.LEFT).name == "always-LEFT"


# ---------------------------------------------------------------------------------------------
# committed results (skipped until part 1 has run)
# ---------------------------------------------------------------------------------------------
BASELINES = ROOT / "data-provenance" / "m7f-baselines.json"


@pytest.mark.skipif(not BASELINES.exists(), reason="part 1 not run")
def test_committed_baselines_match_the_games():
    doc = json.loads(BASELINES.read_text())
    assert list(doc["games"]) == list(GAMES)
    for g, row in doc["games"].items():
        assert row["rules"] == make(g).describe()          # constants frozen after the bots ran
        assert len(row["random"]["scores"]) == 10 and len(row["oracle_reference"]["scores"]) == 10
        assert set(row["fixed"]["bots"]) == {"NOOP", "JUMP", "LEFT", "RIGHT"}
        assert row["threshold"]["theta"] in doc["theta_grid"]
        last = [e for e in doc["tuning_log"] if e["game"] == g][-1]
        assert last["rules_hash"] == row["rules_hash"]


@pytest.mark.skipif(not BASELINES.exists(), reason="part 1 not run")
def test_committed_seed_scores_replay():
    """Spot check: evaluation seed 10 of the random and the fixed-reference bots replays."""
    doc = json.loads(BASELINES.read_text())
    for g in ("paddle", "crossing"):
        row = doc["games"][g]
        env = make(g)
        assert m7f.run_episode(env, m7f.RandomAgent(), 10)["score"] == row["random"]["scores"][0]
        a = Action[row["fixed"]["reference"]]
        assert m7f.run_episode(make(g), m7f.ConstAgent(a), 10)["score"] == row["fixed"]["scores"][0]
