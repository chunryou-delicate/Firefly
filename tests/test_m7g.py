"""M7g tests: the slow variants (the parent's draws in the parent's order, only the speed and the cap
differ, no jump_s8), the bots and verdict rules, the s = 1 reproduction checks, the §5 readings, the
recorder against M7c's, and the brain on a slow variant.

The pure tests run anywhere; the brain tests need CUDA and the graph cache and skip otherwise.
"""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from flysim.apps.games import jump as J
from flysim.apps.games import m7c as M
from flysim.apps.games import m7g as G
from flysim.apps.games import slow
from flysim.apps.games.agents import BrainAgent, StateOracleAgent
from flysim.apps.games.base import Action
from flysim.apps.games.brain import TickDecoder, tick_end_ms
from flysim.apps.games.runner import THETA_GRID, run_episode, select_threshold
from flysim.probe.motor import CountWindowDecoder

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data-provenance" / "m7g-results.json"


# ---------------------------------------------------------------------------------------------
# the variants (pure)
# ---------------------------------------------------------------------------------------------
def test_the_variant_list_is_the_brief():
    want = {"dodge_v2_s2": 0.15, "dodge_v2_s4": 0.075, "dodge_v2_s8": 0.0375, "jump_s2": 0.15, "jump_s4": 0.075}
    assert sorted(slow.VARIANTS) == sorted(want)
    assert "jump_s8" not in slow.VARIANTS and "jump_s8" not in G.ROWS
    with pytest.raises(KeyError):
        slow.make_slow("jump_s8")
    assert G.ROWS == ("dodge_v2", "dodge_v2_s2", "dodge_v2_s4", "dodge_v2_s8", "jump", "jump_s2", "jump_s4")
    for name, speed0 in want.items():
        env = slow.make_slow(name)
        env.reset(0)
        assert env.name == name and env.speed == speed0 and env.max_ticks == 7200
        d = env.describe()
        assert (d["SPEED0"], d["speed_divisor"], d["max_ticks"], d["parent"]) == (speed0, int(name[-1]), 7200, env.parent)
        parent = slow.PARENTS[env.parent]().describe()
        assert d["parent_SPEED0"] == parent["SPEED0"] and d["parent_max_ticks"] == parent["max_ticks"] == 18000
        assert {k: v for k, v in d.items() if k in parent and k not in ("game", "SPEED0", "max_ticks")} == \
            {k: v for k, v in parent.items() if k not in ("game", "SPEED0", "max_ticks")}
    assert G.factory("jump").max_ticks == 18000 and type(G.factory("dodge_v2")) is slow.PARENTS["dodge_v2"]
    with pytest.raises(KeyError):
        G.factory("dodge")                                      # v1 is not on this rung (brief §2)


def test_the_speed_up_rule_is_the_parents():
    for name in slow.VARIANTS:
        env, parent = slow.make_slow(name), slow.PARENTS[slow.VARIANTS[name].parent]()
        env.reset(0), parent.reset(0)
        attr = "cleared" if env.parent == "jump" else "avoided"
        for n in (0, 3, 4, 9, 17):
            setattr(env, attr, n), setattr(parent, attr, n)
            assert env.speed == pytest.approx(parent.speed / env.speed_divisor, rel=1e-15)


def _divisor_one(base_cls):
    return type("One", (base_cls,), {"name": "one", "speed_divisor": 1,
                                     "max_ticks": slow.PARENTS[base_cls.parent].max_ticks})


def test_divisor_one_is_the_parent_frame_for_frame():
    """Only the speed (and the cap) is changed: at divisor 1 the slow class plays as its parent."""
    for base_cls in (slow.SlowJump, slow.SlowDodgeV2):
        for seed in (0, 5):
            a, b = _divisor_one(base_cls)(), slow.PARENTS[base_cls.parent]()
            fa, fb = a.reset(seed), b.reset(seed)
            rng = np.random.default_rng(seed)
            for _ in range(4000):
                assert np.array_equal(fa, fb)
                act = int(rng.choice(4, p=[0.85, 0.05, 0.05, 0.05]))
                fa, ra, da, _ = a.step(act)
                fb, rb, db, _ = b.step(act)
                assert (ra, da, a.score) == (rb, db, b.score)
                if da:
                    break


class _LogRng:
    """Forwards to the game's generator and logs every draw (the stream itself is untouched)."""

    def __init__(self, rng):
        self.rng, self.calls = rng, []

    def integers(self, *a, **k):
        v = self.rng.integers(*a, **k)
        self.calls.append((a, int(v)))
        return v


def _draws(env, agent, seed, n_ticks):
    frame = env.reset(seed)
    agent.reset(seed)
    log = _LogRng(env.rng)
    env.rng = log
    first = env.wait
    for _ in range(n_ticks):
        frame, _, done, _ = env.step(agent.act(frame))
        if done:
            break
    return [first] + [v for _, v in log.calls], log.calls


def test_slow_variants_draw_the_parents_waits_and_sides_in_the_same_order():
    for name in slow.VARIANTS:
        env = slow.make_slow(name)
        parent = slow.PARENTS[env.parent]()
        for seed in (1, 12):
            mine, calls = _draws(env, G.SlowStateOracle(env), seed, 7200)
            theirs, _ = _draws(parent, StateOracleAgent(parent), seed, 18000)
            n = min(len(mine), len(theirs))
            assert n >= 7 and mine[:n] == theirs[:n], name
            assert {a for a, _ in calls} <= {(0, 2), (J.WAIT_MIN, J.WAIT_MAX + 1)}


def test_jump_s8_is_excluded_by_the_constants():
    high = sum(J.jump_height(t) >= J.OBS_HEIGHT for t in range(J.JUMP_TICKS))
    assert high == 27                                            # u in (0.113, 0.887) of 36 ticks
    cross8 = J.DEPTH / (J.SPEED0 / 8)
    assert 26 < cross8 < 27 and high - cross8 < 1                # 0-1 tick to spare: out
    assert high - J.DEPTH / (J.SPEED0 / 4) > 13                  # s4 keeps a wide margin


def test_the_two_minute_cap():
    for name in slow.VARIANTS:
        env = slow.make_slow(name)
        env.reset(0)
        env.tick = 7199                                         # the first obstacle is not out yet
        _, _, done, info = env.step(int(Action.NOOP))
        assert done and info["truncated"] and not info["crashed"] and info["tick"] == 7200


def test_the_oracle_wrapper_runs_the_original_rule():
    for base_cls in (slow.SlowJump, slow.SlowDodgeV2):
        one = _divisor_one(base_cls)()
        with pytest.raises(KeyError):
            StateOracleAgent(one).act(one.reset(0))             # the name check the wrapper exists for
        parent = slow.PARENTS[base_cls.parent]()
        a = run_episode(one, G.SlowStateOracle(one), 10)
        b = run_episode(parent, StateOracleAgent(parent), 10)
        assert (a["score"], a["ticks"]) == (b["score"], b["ticks"]) and a["score"] > 10


# ---------------------------------------------------------------------------------------------
# bots, verdicts, reproduction checks (pure)
# ---------------------------------------------------------------------------------------------
def test_fixed_reference_takes_the_best_mean_ties_in_order():
    r = G.fixed_reference({"NOOP": [0, 0], "JUMP": [1, 1], "LEFT": [2, 0], "RIGHT": [0, 2]})
    assert r["reference"] == "JUMP" and r["scores"] == [1, 1] and r["mean"] == 1.0
    r = G.fixed_reference({"NOOP": [0, 0], "JUMP": [0, 0], "LEFT": [5, 5], "RIGHT": [5, 6]})
    assert r["reference"] == "RIGHT"
    assert G.FIXED_ORDER == ("NOOP", "JUMP", "LEFT", "RIGHT")
    assert G.FixedAgent("LEFT").act(np.zeros((84, 84), np.uint8)) == int(Action.LEFT)


def test_verdict_and_ceiling():
    def bots(brainless, ceiling):
        return {"random": {"scores": [0] * 10}, "fixed": {"scores": [28] * 10, "reference": "RIGHT"},
                "brainless": {"scores": brainless}, "ceiling": {"ceiling": ceiling}}
    up = [72, 72, 72, 72, 72, 72, 72, 72, 72, 71]
    v = G.verdict("dodge_v2_s2", [72] * 10, bots(up, True))
    assert v["beat_random"] and v["vs_fixed_word"] == "beat" and v["ceiling"] and v["ceiling_note"]
    assert v["brainless_verdict"] != "beat the brainless bot"
    v = G.verdict("dodge_v2_s2", [72] * 10, bots(up, False))
    assert v["brainless_verdict"] == "indistinguishable from the brainless bot"   # +0.1 < 3 SEM
    v = G.verdict("jump_s2", [0] * 10, bots([5] * 10, False))
    assert v["brainless_verdict"] == "lost to the brainless bot" and v["vs_random_word"] == "indistinguishable"
    assert G.ceiling_of("dodge_v2_s4", [3, 4], [3, 4])["ceiling"] and not G.ceiling_of("dodge_v2_s4", [3, 4], [3, 5])["ceiling"]
    assert G.ceiling_of("dodge_v2", [1], [2]) == {"ceiling": True, "rule": "M7c pre-registered ceiling (s = 1)", "value": 72}
    assert not G.ceiling_of("jump", [1], [1])["ceiling"]       # s = 1 keeps M7c's rule


def _grid_row(**kw):
    r = {"k": 22.7, "scores": [0] * 10, "ticks": [200] * 10, "gf_spikes": [{"L": 1, "R": 1}] * 10,
         "network_rate_hz": [0.28] * 10, "ignited": 0, "ignited_literal": 10}
    return r | kw


def test_s1_grid_check_stops_on_any_difference():
    G.check_grid_row("jump", _grid_row(), _grid_row())
    for bad in (dict(scores=[0] * 9 + [1]), dict(ticks=[200] * 9 + [201]), dict(ignited=1), dict(k=22.8)):
        with pytest.raises(G.ReproductionError):
            G.check_grid_row("jump", _grid_row(**bad), _grid_row())


def _ob(**kw):
    o = {"index": 0, "t_spawn": 500.0, "side": "left", "first_action": "RIGHT", "first_action_ms": 2700.0,
         "first_correct": True, "n_actions": 3, "first_effective_ms": 2700.0, "away_ticks": 3, "toward_ticks": 0,
         "jumps": 0, "t_contact": 2900.0, "outcome": "crash", "gf_L": 4, "gf_R": 2, "gf_spikes": 6,
         "lead_ms": 200.0, "lead_effective_ms": 200.0, "cause": "late"}
    return o | kw


def test_s1_eval_check_compares_m7c_fields_and_ignores_open_obstacles():
    ep = {"seed": 10, "score": 0, "ticks": 196, "crashed": True, "truncated": False, "gf_spikes": {"L": 7, "R": 6},
          "n_spikes": 10, "network_rate_hz": 0.28, "active_fraction": 0.5, "input_cell_spikes": 3,
          "loss_cause": "late", "ignition": {"ignited": False}}
    mine = [ep | {"obstacles": [_ob(), _ob(index=1, outcome="open", t_contact=None)]}]
    committed = [ep | {"obstacles": [G.as_m7c_obstacle(_ob())]}]
    assert committed[0]["obstacles"][0] == {"index": 0, "t_spawn": 500.0, "first_action_ms": 2700.0,
                                            "moves_away": 3, "moves_toward": 0, "jumps": 0, "side": "left",
                                            "t_contact": 2900.0, "outcome": "crash", "gf_spikes": 6,
                                            "lead_ms": 200.0, "cause": "late"}
    G.check_eval("dodge_v2", mine, committed)
    with pytest.raises(G.ReproductionError):
        G.check_eval("dodge_v2", [ep | {"obstacles": [_ob(away_ticks=4)]}], committed)
    with pytest.raises(G.ReproductionError):
        G.check_eval("dodge_v2", [mine[0] | {"n_spikes": 11}], committed)


# ---------------------------------------------------------------------------------------------
# the recorder against M7c's, with a fake brain (pure)
# ---------------------------------------------------------------------------------------------
class FakeBrain:
    """FlyBrain's interface for the recorder: a giant-fibre spike when k * loom crosses a level,
    plus rare seeded noise spikes."""

    def __init__(self, k=4.0e4, level=1.0):
        self.k, self.level = k, level
        self.reset(0)

    def reset(self, seed):
        self.tick, self.t_ms, self.gf_events = 0, 0, []
        self.rng = np.random.default_rng(seed)

    def step(self, left, right):
        t0 = tick_end_ms(self.tick - 1) if self.tick else 0
        for side, v in (("L", left), ("R", right)):
            if self.k * v > self.level or self.rng.random() < 0.003:
                self.gf_events.append((float(t0 + 3), side))
        self.tick += 1
        self.t_ms = tick_end_ms(self.tick - 1)
        return [e for e in self.gf_events if e[0] >= self.t_ms - 300.0]

    def summary(self):
        return {"network_rate_hz": 0.0, "active_fraction": 0.0, "n_spikes": len(self.gf_events), "windows": [],
                "input_cell_spikes": 0, "gf_spikes": {s: sum(1 for _, x in self.gf_events if x == s) for s in "LR"}}


@pytest.mark.parametrize("game", ["jump", "dodge_v2"])
def test_the_recorder_is_m7c_play_fly_plus_fields(game):
    brain = FakeBrain()
    agent = BrainAgent(brain.step, TickDecoder(CountWindowDecoder(100.0, 1), brain), brain_reset=brain.reset)
    acted = 0
    for seed in (10, 11, 12):
        old = M.play_fly(game, agent, brain, seed, detail=True)
        new = G.play(game, agent, brain, seed)
        assert (new["score"], new["ticks"], new["loss_cause"]) == (old["score"], old["ticks"], old["loss_cause"])
        assert [G.as_m7c_obstacle(o) for o in new["obstacles"] if o["outcome"] != "open"] == old["obstacles"]
        for o in new["obstacles"]:
            assert o["gf_L"] + o["gf_R"] == o["gf_spikes"]
            if o["first_action"] is not None:
                acted += 1
                assert o["lead_ms"] == o["t_contact"] - o["first_action_ms"] and o["n_actions"] >= 1
                assert o["first_effective_ms"] is None or o["first_effective_ms"] >= o["first_action_ms"]
    assert acted > 0


def test_first_correct():
    assert G.first_correct("jump", Action.JUMP, None) and not G.first_correct("jump", Action.LEFT, None)
    assert G.first_correct("dodge", Action.LEFT, "right") and G.first_correct("dodge", Action.RIGHT, "left")
    assert not G.first_correct("dodge", Action.RIGHT, "right") and not G.first_correct("dodge", Action.JUMP, "left")


def test_control_schedule_and_decoder_onsets():
    ends = G.control_ends(M.CONTROL_MS)
    assert len(ends) == 300 and ends[0] == 16 and ends[-1] == 5000
    c = G.control_actions([(100.0, "L"), (150.0, "L"), (2000.0, "R")], CountWindowDecoder(300.0, 2), 5000)
    on_ticks = sum(1 for e in ends if 150 < e <= 400)          # two left spikes in [e - 300, e)
    assert c == {"ticks": 300, "action_ticks": on_ticks, "onsets": 1, "by_action": {"JUMP": 0, "LEFT": 0, "RIGHT": on_ticks},
                 "duration_ms": 5000}


def test_theta_selection_and_loom_values_are_runner_and_m7c():
    grid = THETA_GRID[-3:]                                      # high θ: the bot never acts, games are short
    mine = G.select_theta("jump", grid=grid, seeds=(0, 1), workers=1)
    theirs = select_threshold("jump", grid=grid, seeds=(0, 1), workers=1)
    assert mine["chosen_theta"] == theirs["chosen_theta"] and mine["tied"] == theirs["tied"]
    assert [r["scores"] for r in mine["grid"]] == [r["scores"] for r in theirs["grid"]]
    a = G.loom_values("jump", 0.05, seeds=(0, 3))
    b = M.loom_values("jump", "brainless", seeds=(0, 3), theta=0.05)
    assert len(a) > 0 and np.array_equal(a, b)


# ---------------------------------------------------------------------------------------------
# readings (pure)
# ---------------------------------------------------------------------------------------------
def test_obstacle_stats():
    obs = [_ob(), _ob(lead_ms=400.0, side="right", gf_L=1, gf_R=5, away_ticks=12),
           _ob(first_action="LEFT", first_correct=False, lead_ms=50.0, outcome="pass"),
           _ob(first_action=None, first_action_ms=None, first_correct=None, lead_ms=None, n_actions=0, away_ticks=0),
           _ob(outcome="open", t_contact=None, lead_ms=None)]
    s = G.obstacle_stats(obs, "dodge", onset_rate_per_ms=0.0)
    assert (s["n"], s["open"], s["acted"], s["correct_first"], s["passed"], s["crashed"]) == (4, 1, 3, 2, 1, 3)
    assert s["lead_ms_median"] == 300.0 and s["lead_ms_correct"] == [200.0, 400.0]
    assert s["direction_correct_fraction"] == pytest.approx(2 / 3) and s["away_ge_step_aside"] == 1
    assert s["gf_pillar_side"] == 4 + 5 + 4 + 4 and s["gf_other_side"] == 2 + 1 + 2 + 2
    assert s["acted_fraction"] == 0.75 and s["acted_expected_from_noise"] == 0.0
    none = G.obstacle_stats([_ob(first_action="LEFT", first_correct=False)], "dodge")
    assert none["lead_ms_median"] is None and none["correct_first"] == 0


def test_p1_rule_and_the_reading_table():
    assert G.rises([100.0, 150.0, 400.0]) and not G.rises([100.0, 100.0, 400.0])
    assert not G.rises([100.0, None, 400.0])
    assert G.reading(True, True)["row"] == "P1 yes, P2 yes" and "지지한다" in G.reading(True, True)["sentence"]
    assert G.reading(True, False)["row"] == "P1 yes, P2 no"
    no1, no2 = G.reading(False, True), G.reading(False, False)
    assert no1["row"] == no2["row"] == "P1 no, P2 -" and no1["sentence"] == no2["sentence"]
    assert "지지하지 않는다" in no1["sentence"] and no1["P2"] and not no2["P2"]
    assert set(G.READINGS) == {("yes", "yes"), ("yes", "no"), ("no", None)}


def test_the_figure_draws(tmp_path):
    tab = {r: {"lead_ms_median": 100.0 * G.speed_divisor(r), "lead_ms_q25": 80.0, "lead_ms_q75": 150.0 * G.speed_divisor(r),
               "correct_first": 5} for r in G.ROWS}
    tab["jump_s4"] = {"lead_ms_median": None, "lead_ms_q25": None, "lead_ms_q75": None, "correct_first": 0}
    img = G.draw_figure({"hypothesis": {"top_k_table": tab}}, tmp_path / "lead.png")
    assert img.shape == (580, 960) and (img == 150).sum() > 300 and (img == 0).sum() > 2000
    assert (tmp_path / "lead.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert all(len(v) == 35 and set(v) <= {"0", "1"} for v in G._FONT.values())


# ---------------------------------------------------------------------------------------------
# the post-hoc addendum (brief 추가 등록; pure)
# ---------------------------------------------------------------------------------------------
def test_cell_a_points_bracket_k_match():
    grid = [1.0, 2.0, 4.0, 8.0]
    assert G.cell_a_points(grid, 3.0) == [1, 2] and G.cell_a_points(grid, 4.0) == [2]
    assert G.cell_a_points(grid, 4.0 * (1 + 1e-14)) == [2]
    with pytest.raises(ValueError):
        G.cell_a_points(grid, 9.0)
    assert G.k_top(0.01) == pytest.approx(512000.0)


def test_cell_a_verdict_and_the_2x2_reading():
    assert G.cell_a_verdict([True, True]) == "beat" and G.cell_a_verdict([True]) == "beat"
    assert G.cell_a_verdict([False, False]) == "not" and G.cell_a_verdict([True, False]) == "split"
    assert G.posthoc_reading("beat", False)["sentence"].startswith("시간이 열쇠다")
    assert G.posthoc_reading("not", True)["sentence"].startswith("전류가 열쇠다")
    assert G.posthoc_reading("beat", True)["sentence"] == "둘 다 효과가 있다"
    split = G.posthoc_reading("split", True)                    # a split cell A reads as "not"
    assert not split["A_read_as_beat"] and split["sentence"].startswith("전류가 열쇠다")
    assert G.posthoc_reading("not", False)["sentence"].startswith("느린 판의 승리는")


def test_appearance_split():
    obs = [_ob(t_spawn=500.0, first_action_ms=516.0, outcome="pass", lead_ms=2000.0),
           _ob(t_spawn=500.0, first_action_ms=800.0, outcome="pass", lead_ms=1700.0),
           _ob(t_spawn=500.0, first_action_ms=801.0, outcome="crash", lead_ms=100.0),
           _ob(first_action="LEFT", first_correct=False, outcome="pass"),
           _ob(outcome="open", t_contact=None)]
    a = G.appearance_split(obs)
    assert (a["correct_first"], a["within"], a["after"]) == (3, 2, 1)
    assert a["lead_ms_median_within"] == 1850.0 and a["lead_ms_median_after"] == 100.0
    assert (a["passed"], a["passed_within"], a["passed_after"]) == (3, 2, 0)
    assert a["passed_within_fraction"] == pytest.approx(2 / 3)
    assert G.appearance_split([])["passed_within_fraction"] is None


def test_loom_by_distance_puts_the_first_frame_apart():
    out = G.loom_by_distance("jump", 0.05, seeds=(0,))            # the bot never jumps: one barrier
    for kk in ("all", "level0"):
        assert out[kk]["appearance"]["n"] == 1 and out[kk]["appearance"]["mean"] > 0
        bins = out[kk]["bins"]
        assert [b["from"] for b in bins] == list(G.LOOM_BINS[:-1]) and sum(b["n"] for b in bins) > 100
        means = [b["mean"] for b in bins if b["mean"] is not None]
        assert means == sorted(means)                            # nearer -> grows faster
    slow2 = G.loom_by_distance("jump_s2", 0.05, seeds=(0,))
    assert slow2["all"]["appearance"]["mean"] == out["all"]["appearance"]["mean"]   # the pulse ignores s


# ---------------------------------------------------------------------------------------------
# the brain on a slow variant (CUDA + graph cache)
# ---------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def fly():
    from flysim.graph.constants import CACHE_NPZ
    if not torch.cuda.is_available() or not CACHE_NPZ.exists():
        pytest.skip("CUDA and the graph cache are required")
    brain, agent, choice, _ = G._brain("cuda")
    return brain, agent, choice


def test_the_fly_is_m7c2s_main_arm(fly):
    brain, agent, choice = fly
    assert (choice["window_ms"], choice["min_spikes"]) == (300.0, 2) and brain.warmup_steps == 5000
    assert brain.decoder_window_ms == 300.0 and agent.decoder.describe()["decoder"] == "CountWindowDecoder"


def test_a_slow_fly_episode_is_deterministic(fly):
    brain, agent, _ = fly
    brain.k = 3.0e4
    a = G.play("jump_s2", agent, brain, 3)
    ev_a = list(brain.gf_events)
    b = G.play("jump_s2", agent, brain, 3)
    a.pop("windows"), b.pop("windows")
    assert a == b and ev_a == brain.gf_events and a["ticks"] > 0


def test_the_recorder_is_m7c_play_fly_on_the_brain(fly):
    brain, agent, _ = fly
    brain.k = 4.4e4
    old = M.play_fly("dodge_v2", agent, brain, 11, detail=True)
    new = G.play("dodge_v2", agent, brain, 11)
    for f in ("score", "ticks", "gf_spikes", "n_spikes", "network_rate_hz", "loss_cause"):
        assert new[f] == old[f], f
    assert [G.as_m7c_obstacle(o) for o in new["obstacles"] if o["outcome"] != "open"] == old["obstacles"]


# ---------------------------------------------------------------------------------------------
# the committed results
# ---------------------------------------------------------------------------------------------
@pytest.mark.skipif(not RESULTS.exists(), reason="M7g results not generated yet")
def test_results_follow_the_rules():
    r = json.loads(RESULTS.read_text())
    m7c2 = json.loads(G.M7C2_RESULTS.read_text())
    assert list(r["rows"]) == list(G.ROWS) and r["warmup_steps"] == 5000
    assert (r["decoder_choice"]["window_ms"], r["decoder_choice"]["min_spikes"]) == (300.0, 2)
    for row, v in r["rows"].items():
        assert len(v["k_grid"]) == 21 and v["k"] in v["k_grid"] and v["s"] == G.speed_divisor(row)
        assert v["k_grid"] == pytest.approx(list(M.k_grid(v["p99"])), rel=1e-12)
        best = max(g["mean"] for g in v["selection"]["grid"])
        assert v["selection"]["grid"][v["k_index"]]["mean"] == best
        assert [e["seed"] for e in v["fly"]["episodes"]] == list(range(10, 20))
        assert v["fixed"]["reference"] == G.fixed_reference(v["fixed"]["scores_by_action"])["reference"]
        assert v["obstacles"]["columns"][:4] == ["phase", "k_index", "seed", "index"]
        if row in G.BASE_ROWS:
            main = m7c2["arms"]["main"]["games"][row]
            assert v["reproduces_m7c2"] and v["fly"]["scores"] == main["fly"]["scores"] and v["k"] == main["k"]
            assert [g["scores"] for g in v["selection"]["grid"]] == [g["scores"] for g in main["selection"]["grid"]]
        else:
            assert v["rules"]["max_ticks"] == 7200 and v["bundle"]["dir"] == f"runs/m7g/{row}-brain-seed10"
            assert v["ceiling"]["ceiling"] == (v["brainless"]["scores"] == v["oracle"]["scores"])
        if v["verdict"]["ceiling"]:
            assert v["verdict"]["brainless_verdict"] != "beat the brainless bot"
    h = r["hypothesis"]
    for fam, rows in G.FAMILIES.items():
        assert h["P1"][fam]["lead_ms_median"] == [h["top_k_table"][x]["lead_ms_median"] for x in rows]
        assert h["P1"][fam]["rises"] == G.rises(h["P1"][fam]["lead_ms_median"])
    assert h["P1_holds"] == all(x["rises"] for x in h["P1"].values())
    assert h["P2_holds"] == any(r["rows"][x]["verdict"]["beat_random"] for x in G.SLOW_ROWS)
    assert h["reading"] == G.reading(h["P1_holds"], h["P2_holds"])
    assert h["top_k_table"] == {x: r["rows"][x]["obstacles_by_k"][-1] for x in G.ROWS}
