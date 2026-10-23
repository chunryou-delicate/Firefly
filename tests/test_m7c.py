"""M7c tests: the brain glue, the giant-fibre decoder, selection and verdict rules, ignition
readings, determinism of a fly game, and the shape of the committed results.

The pure tests run anywhere; the brain tests need CUDA and the graph cache and skip otherwise.
"""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from flysim.apps.games import brain as BR
from flysim.apps.games import m7c as M
from flysim.probe import motor
from flysim.probe.motor import GiantFibreDecoder

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data-provenance" / "m7c-results.json"


# ---------------------------------------------------------------------------------------------
# tick schedule and decoder (pure)
# ---------------------------------------------------------------------------------------------
def test_tick_schedule_alternates_16_and_17_steps():
    steps = [BR.steps_in_tick(k) for k in range(600)]
    assert set(steps) == {16, 17}
    assert sum(steps[:60]) == 1000 and sum(steps) == 10_000          # exact, no drift
    assert BR.tick_end_ms(59) == 1000 and BR.tick_end_ms(0) == 16
    with pytest.raises(ValueError):
        BR.steps_in_tick(0, dt_ms=0.1)


def test_giant_fibre_decoder_rule():
    d = GiantFibreDecoder(window_ms=50.0)
    assert d.decode([]) == "NOOP"
    assert d.decode([(10.0, "L"), (30.0, "R")]) == "JUMP"
    assert d.decode([(10.0, "L")]) == "RIGHT"                  # left fibre -> move right
    assert d.decode([(5.0, "R"), (9.0, "R")]) == "LEFT"
    with pytest.raises(ValueError):
        d.decode([(1.0, "X")])
    with pytest.raises(ValueError):
        GiantFibreDecoder(window_ms=0)
    src = Path(motor.__file__).read_text()
    for name in ("DNp01", "LC4", "LPLC2", "TTMn"):              # no cell-type names in the decoder
        assert name not in src


def test_brain_parameters_are_m7a_p2():
    assert BR.BRAIN_PARAMS == dict(synapse="conductance", g=5.36539e-4, noise_sigma=29.13,
                                   adapt_g_b=8.0, dt=1.0)
    assert BR.DECODER_WINDOW_MS == 50.0


# ---------------------------------------------------------------------------------------------
# grid, selection, verdicts (pure)
# ---------------------------------------------------------------------------------------------
def test_k_grid_carries_the_m7a_gain_range():
    g = M.k_grid(0.05)
    assert len(g) == 21
    assert g[0] == pytest.approx(1 / 0.05) and g[-1] == pytest.approx(5120 / 0.05)
    assert np.allclose(np.diff(np.log(g)), np.log(5120) / 20)
    with pytest.raises(ValueError):
        M.k_grid(0.0)


def test_select_best_ties_go_to_the_geometric_middle():
    rows = {k: {"mean": m} for k, m in zip((1.0, 2.0, 4.0, 8.0, 16.0), (0, 3, 3, 3, 1))}
    sel = M.select_best(rows)
    assert sel["tied"] == [2.0, 4.0, 8.0] and sel["chosen"] == 4.0 and sel["best_mean"] == 3


def test_paired_verdicts():
    fly = [5, 6, 7, 5, 6, 7, 5, 6, 7, 6]
    assert M.paired(fly, [0] * 10)["beats"]
    assert M.paired([0] * 10, fly)["loses"]
    noisy = M.paired([5, 0, 9, 1, 8, 2, 7, 3, 6, 4], [4] * 10)
    assert not noisy["beats"] and not noisy["loses"]
    v = M.verdict(fly, [0] * 10, [1] * 10, [0] * 10, game="jump")
    assert v["beat_random"] and v["beat_one_way"] and v["brainless_verdict"] == "beat the brainless bot"
    # dodge_v2 has a physical ceiling: "beat the brainless bot" cannot be reported there
    v2 = M.verdict(fly, [0] * 10, [1] * 10, [0] * 10, game="dodge_v2")
    assert v2["brainless_verdict"] == "indistinguishable from the brainless bot" and v2["ceiling"] == 72
    lost = M.verdict([0] * 10, [0] * 10, fly, [0] * 10, game="dodge_v2")
    assert lost["brainless_verdict"] == "lost to the brainless bot"


def test_one_way_reference_takes_the_higher_mean():
    ow = M.one_way_reference([28, 29, 28], [30, 29, 30])
    assert ow["reference"] == "always_right" and ow["scores"] == [30, 29, 30]
    tie = M.one_way_reference([1, 1], [1, 1])
    assert tie["reference"] == "always_left"
    assert set(M.ONE_WAY_GAMES) == {"jump", "dodge", "dodge_v2"}


def _windows(first, rest, n=25, rate=0.3):
    return [[rate * 3, first]] + [[rate, rest]] * (n - 1)


def test_ignition_readings():
    rng = np.random.default_rng(0)
    ctrl = {"windows": [[0.7, 0.14]] + [[0.3, float(x)] for x in 0.058 + 0.02 * rng.standard_normal(24)]}
    quiet = {"windows": [[0.7, 0.14]] + [[0.3, float(x)] for x in 0.058 + 0.02 * rng.standard_normal(40)],
             "n_spikes": 10_000, "active_fraction": 0.5}
    ig = M.ignition(quiet, ctrl)
    assert ig["literal"]["ignited"]                    # the start-up window alone trips the literal rule
    assert not ig["like_for_like"]["ignited"] and not ig["ignited"]
    assert not ig["viewer_failure"]
    busy = {"windows": [[0.7, 0.14]] + [[0.3, 0.15]] * 40, "n_spikes": 10_000, "active_fraction": 0.9}
    assert M.ignition(busy, ctrl)["like_for_like"]["ignited"]
    runaway = {"windows": [[6.0, 0.3]] * 5, "n_spikes": 1, "active_fraction": 0.3}
    assert M.ignition(runaway, ctrl)["window_runaway"] and M.ignition(runaway, ctrl)["ignited"]
    dead = {"windows": [[0.0, 0.0]] * 5, "n_spikes": 0, "active_fraction": 0.0}
    assert M.ignition(dead, ctrl)["viewer_failure"]
    full = {"windows": [[1.0, 0.95]] * 5, "n_spikes": 5, "active_fraction": 0.95}
    assert M.ignition(full, ctrl)["viewer_failure"] and M.ignition(full, ctrl)["window_runaway"]


def test_bundle_events_keep_the_last_tick():
    end = BR.tick_end_ms(9)                                    # a 10-tick game ends here
    ev = [{"t_ms": 16.0, "type": "move_left"}, {"t_ms": float(end), "type": "crash"}]
    assert M.bundle_events(ev, 10) == ev                       # the crash of the last tick is kept
    assert M.bundle_events(ev, 9) == ev[:1]                    # a cut bundle drops ticks it lacks
    assert M.bundle_events(ev, 0) == []
    assert BR.tick_end_ms(M.BUNDLE_MAX_MS * 60 // 1000 - 1) == M.BUNDLE_MAX_MS


# ---------------------------------------------------------------------------------------------
# the brain itself (CUDA + graph cache)
# ---------------------------------------------------------------------------------------------
def _ctx():
    from flysim.graph.constants import CACHE_NPZ
    if not torch.cuda.is_available() or not CACHE_NPZ.exists():
        pytest.skip("CUDA and the graph cache are required")
    from flysim.graph import Graph
    from flysim.probe.sets import build_probe_sets, load_roi
    g = Graph.load()
    ps = build_probe_sets(g, load_roi(g), profile="escape")
    return g, ps


@pytest.fixture(scope="module")
def brain_ctx():
    g, ps = _ctx()
    es = BR.EscapeSets.from_probe_sets(ps)
    return g, ps, es, BR.FlyBrain(g, es, k=3000.0)


def test_escape_sets_wiring(brain_ctx):
    g, ps, es, _ = brain_ctx
    assert len(es.inputs["L"]) == 71 + 94 and len(es.inputs["R"]) == 55 + 91
    assert es.gf["L"] == int(ps.members["DNp01_L"][0]) and es.gf["R"] == int(ps.members["DNp01_R"][0])
    assert not set(es.inputs["L"]) & set(es.inputs["R"])
    from flysim.probe.sets import build_probe_sets, load_roi
    with pytest.raises(ValueError):
        BR.EscapeSets.from_probe_sets(build_probe_sets(g, load_roi(g)))      # auditory profile


def test_input_goes_to_the_detectors_only(brain_ctx):
    _, _, es, b = brain_ctx
    b.reset(0)
    b.step(0.01, 0.0)
    buf = b.engine.i_ext.cpu().numpy()
    assert np.allclose(buf[es.inputs["L"]], b.k * 0.01) and buf[es.inputs["R"]].max() == 0
    assert np.count_nonzero(buf) == len(es.inputs["L"])
    b.step(0.0, 0.02)
    buf = b.engine.i_ext.cpu().numpy()
    assert buf[es.inputs["L"]].max() == 0 and np.allclose(buf[es.inputs["R"]], b.k * 0.02)
    assert b.t_ms == BR.tick_end_ms(1)


def test_fly_game_is_deterministic_and_seed_dependent(brain_ctx):
    from flysim.apps.games.agents import BrainAgent
    _, _, _, b = brain_ctx
    agent = BrainAgent(b, BR.make_decoder(), brain_reset=b.reset)
    r1 = M.play_fly("jump", agent, b, 3, detail=True)
    ev1, spikes1 = list(b.gf_events), b.n_spikes
    r2 = M.play_fly("jump", agent, b, 3, detail=True)
    assert (r1["score"], r1["ticks"], r1["n_spikes"]) == (r2["score"], r2["ticks"], r2["n_spikes"])
    assert b.gf_events == ev1 and b.n_spikes == spikes1          # same seed -> same spikes
    r3 = M.play_fly("jump", agent, b, 4)
    assert (r3["n_spikes"], r3["ticks"]) != (r1["n_spikes"], r1["ticks"])


def test_brain_agent_sees_what_the_brainless_bot_sees(brain_ctx):
    from flysim.apps.games.agents import BrainAgent, ThresholdAgent
    from flysim.apps.games.base import make
    _, _, _, b = brain_ctx
    agent, bot = BrainAgent(b, BR.make_decoder(), brain_reset=b.reset), ThresholdAgent(1.0)
    env = make("dodge_v2")
    f = env.reset(5)
    agent.reset(5), bot.reset(5)
    for _ in range(150):
        a = agent.act(f)
        bot.act(f)
        assert agent.last_signal == bot.last_signal
        f, _, done, _ = env.step(a)
        if done:
            break


def test_bundle_game_json_ends_with_the_crash(brain_ctx, tmp_path):
    from flysim.apps.games.agents import BrainAgent
    from flysim.probe.export import neuron_coords
    from flysim.probe.sets import load_roi
    g, _, es, b = brain_ctx
    agent = BrainAgent(b, BR.make_decoder(), brain_reset=b.reset)
    b.keep_spikes = True
    try:
        ep = M.play_fly("dodge_v2", agent, b, 10, detail=True, record=True)
        ctx = {"escape": es, "coords": neuron_coords(g, load_roi(g))}
        M.write_bundle("dodge_v2", b.k, ep, ctx, b, tmp_path)
    finally:
        b.keep_spikes, b.spikes_t, b.spikes_i = False, [], []
    doc = json.loads((tmp_path / "game.json").read_text())
    assert doc["crashed"] is True and doc["loss_cause"] == ep["loss_cause"] is not None
    last = doc["events"][-1]
    assert last["type"] == "crash" and last["t_ms"] == doc["duration_ms"] == doc["bundle_ms"]


# ---------------------------------------------------------------------------------------------
# the committed results
# ---------------------------------------------------------------------------------------------
@pytest.mark.skipif(not RESULTS.exists(), reason="M7c results not generated yet")
def test_results_shape_and_protocol():
    r = json.loads(RESULTS.read_text())
    assert r["train_seeds"] == list(range(10)) and r["eval_seeds"] == list(range(10, 20))
    assert r["rules"]["p99_frames"] == "brainless" and r["rules"]["k_points"] == 21
    for game in ("jump", "dodge", "dodge_v2"):
        row = r["games"][game]
        assert len(row["selection"]["grid"]) == 21 and row["k"] in row["k_grid"]
        assert all(len(x["scores"]) == 10 for x in row["selection"]["grid"])
        assert [e["seed"] for e in row["fly"]["episodes"]] == list(range(10, 20))
        for key in ("vs_random", "vs_brainless", "vs_one_way", "brainless_verdict"):
            assert key in row["verdict"]
        assert row["one_way"]["reference"] in ("always_left", "always_right")
        best = max(x["mean"] for x in row["selection"]["grid"])
        assert next(x for x in row["selection"]["grid"] if x["k"] == row["k"])["mean"] == best
    assert r["games"]["dodge_v2"]["verdict"]["brainless_verdict"] != "beat the brainless bot"
