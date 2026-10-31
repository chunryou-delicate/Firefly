"""M7f yoked control (docs/m7f-brief.md, addendum 2 item 3): the fly's action tapes played blind.

CPU only. The fly's own tapes come from the committed results file when it exists."""
import json

import numpy as np
import pytest

from flysim.apps.games import m7f
from flysim.apps.games.agents import RandomAgent
from flysim.apps.games.base import Action, make
from flysim.apps.games.runner import EVAL_SEEDS, run_episode

# the planning session's own boxing run (rows = tape seed, columns = game seed, both 10..19)
BOXING_PUBLISHED = [[2, 2, 1, 0, 1, 1, 4, 2, 4, 1],
                    [2, 1, 2, 3, 2, 3, 2, 4, 2, 3],
                    [4, 3, 4, 4, 1, 3, 4, 3, 3, 0],
                    [0, 0, 0, 7, 0, 1, 0, 0, 2, 1],
                    [2, 0, 4, 7, 4, 0, 1, 0, 2, 3],
                    [4, 3, 4, 5, 4, 3, 3, 2, 4, 3],
                    [1, 0, 0, 0, 0, 1, 1, 0, 1, 0],
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 0, 2, 2, 2, 1, 1, 2, 2, 1],
                    [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]]


def test_tape_agent_wraps_from_the_start_and_resets():
    a = m7f.TapeAgent([1, 2, 3])
    frame = np.zeros((84, 84), np.uint8)
    assert [a.act(frame) for _ in range(7)] == [1, 2, 3, 1, 2, 3, 1]
    a.reset(5)
    assert a.act(frame) == 1
    with pytest.raises(ValueError):
        m7f.TapeAgent([])


def test_wrapped_tape_is_the_repeating_policy():
    """A one-action tape on a 3,600-tick game is the fixed-action bot; wraps are counted."""
    e = m7f._tape_episode(("crossing", (int(Action.JUMP),), 10))
    want = run_episode(make("crossing"), m7f.ConstAgent(Action.JUMP), 10)
    assert (e["score"], e["ticks"]) == (want["score"], want["ticks"]) and e["wraps"] == 3599
    e = m7f._tape_episode(("boxing", (1, 0, 0, 2), 12))
    assert e["ticks"] == 3600 and e["wraps"] == 899


@pytest.mark.parametrize("game", ["paddle", "ski"])
def test_own_tape_on_own_seed_gives_own_score(game):
    taping = m7f._Taping(RandomAgent())
    want = run_episode(make(game), taping, 13)
    assert len(taping.tape) == want["ticks"]
    got = m7f._tape_episode((game, tuple(taping.tape), 13))
    assert (got["score"], got["ticks"], got["crashed"]) == (want["score"], want["ticks"], want["crashed"])
    assert got["wraps"] == 0


def test_yoked_matrix_and_scores():
    """Random-bot tapes on paddle: the diagonal is the bot's own score, a short tape wraps."""
    tapes, own = [], []
    for s in EVAL_SEEDS:
        taping = m7f._Taping(RandomAgent())
        own.append(run_episode(make("paddle"), taping, s)["score"])
        tapes.append(taping.tape)
    mat = m7f.yoked_matrix("paddle", tapes, workers=1)
    assert [mat["scores"][i][i] for i in range(10)] == own
    for i, t in enumerate(tapes):
        for j in range(10):
            assert mat["wraps"][i][j] == max(0, (mat["ticks"][i][j] - 1) // len(t))
    assert any(w > 0 for row in mat["wraps"] for w in row)
    m = np.array(mat["scores"], float)
    y = m7f.yoked_scores(mat["scores"])
    assert y[3] == pytest.approx((m[:, 3].sum() - m[3, 3]) / 9)
    assert m7f.yoked_scores([[9, 1, 2], [3, 9, 4], [5, 6, 9]]) == [4.0, 3.5, 3.0]


@pytest.fixture(scope="module")
def yoked_file():
    if not m7f.YOKED.exists():
        pytest.skip("the yoked results file has not been made yet")
    return json.loads(m7f.YOKED.read_text()), json.loads(m7f.RESULTS.read_text())


def test_yoked_file_matches_part2_and_its_own_rules(yoked_file):
    yk, res = yoked_file
    assert list(yk["games"]) == list(m7f.GAMES)
    for g, row in yk["games"].items():
        fly_eps = res["games"][g]["fly"]["episodes"]
        assert row["k"] == res["games"][g]["k"]
        assert row["fly"]["scores"] == [e["score"] for e in fly_eps]
        assert [row["matrix"]["scores"][i][i] for i in range(10)] == row["fly"]["scores"]
        assert row["yoked"]["scores"] == m7f.yoked_scores(row["matrix"]["scores"])
        for s, e in zip(EVAL_SEEDS, fly_eps):
            t = row["tapes"][str(s)]
            assert len(t["tape"]) == t["ticks"] == e["ticks"]
            counts = np.bincount([int(c) for c in t["tape"]], minlength=4).tolist()
            assert dict(zip(("NOOP", "JUMP", "LEFT", "RIGHT"), counts)) == e["actions"] == t["actions"]


def test_boxing_matrix_recomputes_from_the_tapes_and_matches_the_published_one(yoked_file):
    yk, _ = yoked_file
    row = yk["games"]["boxing"]
    tapes = [row["tapes"][str(s)]["tape"] for s in EVAL_SEEDS]
    mat = m7f.yoked_matrix("boxing", [[int(c) for c in t] for t in tapes], workers=1)
    assert mat["scores"] == row["matrix"]["scores"] == BOXING_PUBLISHED
