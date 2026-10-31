"""M7f part 2 tests: the fly on the classic games (docs/m7f-brief.md §4).

CPU tests use a stand-in brain; the brain tests need CUDA and the graph cache (skipped
otherwise) and play only short games.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from flysim.apps.games import m7f
from flysim.apps.games.base import Action, make
from flysim.apps.games.classic.common import draw

ROOT = Path(__file__).resolve().parents[1]


class FakeBrain:
    """``summary()`` only, in the shape FlyBrain gives it."""

    def summary(self):
        return {"duration_ms": 0, "n_spikes": 5, "network_rate_hz": 0.1, "active_fraction": 0.01,
                "input_cell_spikes": 0, "gf_spikes": {"L": 1, "R": 2}, "window_rate_hz_max": 0.1,
                "window_active_max": 0.01, "windows": [[0.1, 0.01]]}


class Scripted(m7f.ConstAgent):
    def __init__(self, actions):
        super().__init__(0)
        self.actions = list(actions)
        self.i = 0
        self.last_signal = (0.0, 0.0)

    def reset(self, seed):
        self.i = 0

    def act(self, frame):
        a = self.actions[self.i % len(self.actions)]
        self.i += 1
        return a


def test_part2_constants_are_m7c2_main_arm():
    assert m7f.P2_WARMUP_STEPS == 5000 and m7f.P2_DECODER == (300.0, 2) and m7f.P2_BUNDLE_SEED == 10
    from flysim.apps.games import m7c2 as M2
    choice = M2.load_decoder_choice()
    assert (choice["window_ms"], choice["min_spikes"]) == m7f.P2_DECODER and M2.WARMUP_STEPS == 5000


def test_play_classic_record_for_games_without_obstacles():
    """Lives game: three misses -> life_lost x3 then game_over; actions counted; scenes redraw."""
    ep = m7f.play_classic("paddle", m7f.ConstAgent(Action.LEFT), FakeBrain(), 3, record=True)
    assert ep["crashed"] and ep["actions"]["LEFT"] == ep["ticks"] and ep["first_action_ms"] == 16.0
    ev = ep["record"]["events"]
    assert [e["lives"] for e in ev if e["type"] == "life_lost"] == [2, 1, 0] and ev[-1]["type"] == "game_over"
    assert len(ep["lives_lost_ms"]) == 3 and ep["lives_lost_ms"][-1] == ev[-1]["t_ms"]   # the third ends it
    rec = ep["record"]
    assert len(rec["frames"]) == len(rec["scenes"]) == len(rec["actions"]) == ep["ticks"]
    for f, sc in zip(rec["frames"][::25], rec["scenes"][::25]):
        assert np.array_equal(draw(sc["rects"]), f)
    assert ep["gf_spikes"] == {"L": 1, "R": 2} and "windows" in ep


def test_play_classic_scores_and_clock_games():
    ep = m7f.play_classic("crossing", m7f.ConstAgent(Action.JUMP), FakeBrain(), 10)
    assert ep["truncated"] and not ep["crashed"] and ep["ticks"] == 3600
    assert ep["score"] == len(ep["score_ms"]) and ep["lives_lost_ms"] == []
    noop = m7f.play_classic("crossing", m7f.ConstAgent(Action.NOOP), FakeBrain(), 10)
    assert noop["first_action_ms"] is None and noop["actions"]["NOOP"] == 3600


def test_play_classic_record_window():
    ep = m7f.play_classic("ski", Scripted([0, 2, 3]), FakeBrain(), 1, record=True, max_ms=1000)
    assert ep["ticks"] == 3600 and len(ep["record"]["frames"]) == 60      # frames shown before 1 s


def test_p99_none_means_not_seen(monkeypatch):
    from flysim.apps.games import m7c as M
    monkeypatch.setattr(M, "loom_values", lambda *a, **k: np.zeros(0))
    p = m7f.p99_for("paddle", 1e-3)
    assert p["p99"] is None and p["n_values"] == 0 and p["status"] == m7f.NOT_SEEN
    monkeypatch.setattr(M, "loom_values", lambda *a, **k: np.arange(1, 101, dtype=float))
    assert m7f.p99_for("paddle", 1e-3)["p99"] == pytest.approx(np.percentile(np.arange(1, 101), 99))


def test_verdict2():
    fly = [5, 6, 5, 6, 5, 6, 5, 6, 5, 6]
    v = m7f.verdict2(fly, [0] * 10, fly, [9] * 10, ceiling=False)
    assert v["vs_random"]["verdict"] == "beat" and v["vs_fixed"]["verdict"] == "indistinguishable"
    assert v["vs_brainless"]["verdict"] == "lost"
    v = m7f.verdict2(fly, [0] * 10, fly, [0] * 10, ceiling=True)
    assert v["vs_brainless"]["verdict"] == "ceiling — no verdict" and v["ceiling"]


def test_check_bots_reproduces_part1():
    base = json.loads(m7f.BASELINES.read_text())
    refs = m7f.check_bots(["paddle"], base, 1)
    assert refs["paddle"]["fixed"]["reference"] == base["games"]["paddle"]["fixed"]["reference"]
    assert refs["paddle"]["brainless"]["scores"] == base["games"]["paddle"]["threshold"]["eval"]["scores"]
    assert refs["paddle"]["ceiling"] is False
    bad = json.loads(json.dumps(base))
    bad["games"]["paddle"]["random"]["scores"][0] += 1
    with pytest.raises(RuntimeError):
        m7f.check_bots(["paddle"], bad, 1)


def test_merge_checks_protocol_and_completeness(tmp_path):
    shared = {k: {"x": 1} for k in m7f.SHARED_KEYS}
    runs = []
    for i, games in enumerate((m7f.GAMES[:4], m7f.GAMES[4:])):
        doc = shared | {"generated": "t", "brief": "b", "wall_s": 1.0,
                        "games": {g: {"k": i} for g in games}, "bots": {g: {} for g in games}}
        p = tmp_path / f"run{i}.json"
        p.write_text(json.dumps(doc))
        runs.append(p)
    out = tmp_path / "results.json"
    res = m7f.merge(runs, out)
    assert list(res["games"]) == list(m7f.GAMES) and out.exists()
    with pytest.raises(ValueError):
        m7f.merge(runs[:1], out)                               # games missing
    doc = json.loads(runs[1].read_text())
    doc["controls"] = {"x": 2}
    runs[1].write_text(json.dumps(doc))
    with pytest.raises(ValueError):
        m7f.merge(runs, out)                                   # different controls


# ---------------------------------------------------------------------------------------------
# the brain (CUDA + graph cache)
# ---------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def fly():
    import torch
    from flysim.graph.constants import CACHE_NPZ
    if not torch.cuda.is_available() or not CACHE_NPZ.exists():
        pytest.skip("CUDA and the graph cache are required")
    return m7f.brain_setup("cuda")


def test_brain_is_the_m7c2_main_arm(fly):
    agent, brain, _, _ = fly
    assert brain.warmup_steps == 5000 and brain.decoder_window_ms == 300.0
    d = agent.decoder.describe()
    assert d["window_ms"] == 300.0 and d["min_spikes"] == 2


def test_fly_episode_is_deterministic_same_score_same_spikes(fly):
    agent, brain, _, _ = fly
    brain.k = 2.0e6
    runs = []
    for _ in range(2):
        brain.keep_spikes = True
        try:
            e = m7f.play_classic("paddle", agent, brain, 3, record=True)
            runs.append((e["score"], e["ticks"], e["n_spikes"], list(brain.gf_events), e["record"]["actions"],
                         *brain.spikes()))
        finally:
            brain.keep_spikes, brain.spikes_t, brain.spikes_i = False, [], []
    a, b = runs
    assert a[:5] == b[:5] and np.array_equal(a[5], b[5]) and np.array_equal(a[6], b[6])
    other = m7f.play_classic("paddle", agent, brain, 4)
    assert (other["n_spikes"], other["ticks"]) != (a[2], a[1])


def test_bundle_has_scenes_that_redraw_the_frames(fly, tmp_path):
    agent, brain, ctx, _ = fly
    k = 2.0e6
    brain.k = k
    score = m7f.play_classic("paddle", agent, brain, m7f.P2_BUNDLE_SEED)["score"]
    out = m7f.fly_bundle("paddle", k, score, agent, brain, ctx, out_root=tmp_path)
    d = tmp_path / "paddle-brain-seed10"
    sc = json.loads((d / "scene.json").read_text())
    frames = np.load(d / "frames.npz")["frames"]
    assert sc["frames"] == len(frames) == out["frames"]
    assert all(np.array_equal(draw(s["rects"]), f) for s, f in zip(sc["scenes"], frames))
    game = json.loads((d / "game.json").read_text())
    assert game["game"] == "paddle" and game["score"] == score
    if game["crashed"]:
        assert game["events"][-1]["type"] == "game_over"
    assert game["rules"] == make("paddle").describe()
