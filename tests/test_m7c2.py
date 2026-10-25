"""M7c2 phase 2 tests: warm-up and read-out window in FlyBrain (defaults = M7c), the tick-clock
decoder glue, the phase-1 choice guard, determinism with the warm-up, and the results shape.

The pure tests run anywhere; the brain tests need CUDA and the graph cache and skip otherwise.
"""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from flysim.apps.games import brain as BR
from flysim.apps.games import m7c as M
from flysim.apps.games import m7c2 as M2
from flysim.probe.motor import CountWindowDecoder, GiantFibreDecoder

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data-provenance" / "m7c2-results.json"


# ---------------------------------------------------------------------------------------------
# pure
# ---------------------------------------------------------------------------------------------
def test_rematch_constants():
    assert M2.WARMUP_STEPS == 5000
    assert M2.ARMS == ("main", "aux") and M2.BUNDLE_ARM == "main"
    assert M2.W_GRID == (50.0, 100.0, 200.0, 300.0) and M2.N_GRID == (1, 2, 3)
    assert BR.DECODER_WINDOW_MS == 50.0                     # M7c's window stays the default


def _choice_file(tmp_path, **sel):
    p = tmp_path / "m7c2-decoder.json"
    p.write_text(json.dumps({"selection": sel, "git": {"head": "abc"}}))
    return p


def test_decoder_choice_guard(tmp_path, monkeypatch):
    monkeypatch.setattr(M2, "ROOT", tmp_path)               # paths are reported relative to ROOT
    with pytest.raises(FileNotFoundError):
        M2.load_decoder_choice(tmp_path / "missing.json")
    ok = M2.load_decoder_choice(_choice_file(tmp_path, window_ms=200, min_spikes=2, J=0.31))
    assert (ok["window_ms"], ok["min_spikes"], ok["J"]) == (200.0, 2, 0.31) and len(ok["sha256"]) == 64
    with pytest.raises(ValueError):                          # outside the brief's 12-cell table
        M2.load_decoder_choice(_choice_file(tmp_path, window_ms=150, min_spikes=2, J=0.3))
    with pytest.raises(ValueError):
        M2.load_decoder_choice(_choice_file(tmp_path, window_ms=100, min_spikes=4, J=0.3))
    with pytest.raises(ValueError):                          # J <= 0: phase 2 is not run
        M2.load_decoder_choice(_choice_file(tmp_path, window_ms=100, min_spikes=1, J=0.0))
    with pytest.raises(ValueError):
        M2.load_decoder_choice(_choice_file(tmp_path, window_ms=100, min_spikes=1, J=0.2, beats_noise=False))


def test_committed_phase1_choice():
    ch = M2.load_decoder_choice()
    assert (ch["window_ms"], ch["min_spikes"]) == (300.0, 2) and ch["J_exact"] == "811/1176"
    assert ch["file"] == "data-provenance/m7c2-decoder.json" and ch["committed_in"]


class _Clock:
    t_ms = 0.0
    decoder_window_ms = 50.0


def test_tick_decoder_passes_the_tick_end():
    clock = _Clock()
    d = BR.TickDecoder(CountWindowDecoder(100.0, 2), clock)
    ev = [(10.0, "L"), (60.0, "L"), (90.0, "R")]
    clock.t_ms = 100.0
    assert d.decode(ev) == "RIGHT"                          # two left spikes in [0, 100)
    clock.t_ms = 150.0
    assert d.decode(ev) == "NOOP"                           # only one left spike in [50, 150)
    assert d.describe()["decoder"] == "CountWindowDecoder"


def test_arm_decoders():
    clock = _Clock()
    d = M2.arm_decoder("main", clock, {"window_ms": 300.0, "min_spikes": 3})
    assert clock.decoder_window_ms == 300.0 and isinstance(d.decoder, CountWindowDecoder)
    assert (d.decoder.window_ms, d.decoder.min_spikes) == (300.0, 3)
    d = M2.arm_decoder("aux", clock, None)
    assert clock.decoder_window_ms == 50.0 and isinstance(d, GiantFibreDecoder) and d.window_ms == 50.0
    with pytest.raises(ValueError):
        M2.arm_decoder("main", clock, None)
    with pytest.raises(KeyError):
        M2.arm_decoder("other", clock, None)


# ---------------------------------------------------------------------------------------------
# the brain (CUDA + graph cache)
# ---------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def warm():
    from flysim.graph.constants import CACHE_NPZ
    if not torch.cuda.is_available() or not CACHE_NPZ.exists():
        pytest.skip("CUDA and the graph cache are required")
    from flysim.graph import Graph
    from flysim.probe.sets import build_probe_sets, load_roi
    g = Graph.load()
    es = BR.EscapeSets.from_probe_sets(build_probe_sets(g, load_roi(g), profile="escape"))
    return g, es, BR.FlyBrain(g, es, k=3000.0, warmup_steps=M2.WARMUP_STEPS)


def test_defaults_are_m7c(warm):
    g, es, _ = warm
    b = BR.FlyBrain(g, es, k=3000.0)
    assert b.warmup_steps == 0 and b.decoder_window_ms == 50.0
    assert "warmup" not in b.describe()
    assert b.describe()["readout"] == "DNp01 L/R spikes in the last 50 ms (escape profile)"
    with pytest.raises(ValueError):
        BR.FlyBrain(g, es, k=1.0, warmup_steps=-1)
    with pytest.raises(ValueError):
        BR.FlyBrain(g, es, k=1.0, decoder_window_ms=0)


def test_warmup_shifts_the_clock_and_drops_its_spikes(warm):
    _, _, b = warm
    b.keep_spikes = True
    try:
        b.reset(4)
        assert b.engine.t == M2.WARMUP_STEPS and b.t_ms == 0 and b.gf_events == [] and b.n_spikes == 0
        for _ in range(3):
            b.step(0.0, 0.0)
        t_game, i_game = b.spikes()
    finally:
        b.keep_spikes, b.spikes_t, b.spikes_i = False, [], []
    assert b.t_ms == BR.tick_end_ms(2) and t_game.min() >= 0 and t_game.max() < b.t_ms
    # the same engine, reset and run by hand: the game ticks are steps 5000.. of one noise stream
    b.engine.reset(4)
    b.engine.run(M2.WARMUP_STEPS, None, record=True)
    t_ref, i_ref = b.engine.run(BR.tick_end_ms(2), None, record=True)
    assert np.array_equal(t_game, t_ref - M2.WARMUP_STEPS) and np.array_equal(i_game, i_ref)


def test_read_out_window_argument(warm):
    _, _, b = warm
    b.reset(0)
    b.decoder_window_ms = 1e9                                # everything since the warm-up
    try:
        for _ in range(240):                                 # 4 s: about 15 spontaneous GF spikes
            out = b.step(0.0, 0.0)
        assert out == b.gf_events and len(out) > 0
        b.decoder_window_ms = 300.0
        out = b.step(0.0, 0.0)
        assert out == [e for e in b.gf_events if e[0] >= b.t_ms - 300.0]
    finally:
        b.decoder_window_ms = BR.DECODER_WINDOW_MS


def test_rematch_episode_is_deterministic_with_warmup(warm):
    from flysim.apps.games.agents import BrainAgent
    _, _, b = warm
    agent = BrainAgent(b, M2.arm_decoder("main", b, {"window_ms": 200.0, "min_spikes": 2}), brain_reset=b.reset)
    try:
        runs = []
        for _ in range(2):
            b.keep_spikes = True
            r = M.play_fly("dodge_v2", agent, b, 7, detail=True)
            runs.append((r["score"], r["ticks"], r["n_spikes"], list(b.gf_events), *b.spikes()))
            b.keep_spikes, b.spikes_t, b.spikes_i = False, [], []
        a, c = runs
        assert a[:4] == c[:4] and np.array_equal(a[4], c[4]) and np.array_equal(a[5], c[5])
        r8 = M.play_fly("dodge_v2", agent, b, 8)
        assert (r8["n_spikes"], r8["ticks"]) != (a[2], a[1])
    finally:
        M2.arm_decoder("aux", b, None)


def test_count_decoder_50_1_plays_exactly_like_m7c_decoder(warm):
    """The auxiliary arm uses GiantFibreDecoder; CountWindowDecoder(50, 1) through TickDecoder is
    the same read-out on the same brain (brief §1.1)."""
    from flysim.apps.games.agents import BrainAgent
    _, _, b = warm
    b.k = 2.0e4
    out = []
    for dec in (M2.arm_decoder("aux", b, None), M2.arm_decoder("main", b, {"window_ms": 50.0, "min_spikes": 1})):
        agent = BrainAgent(b, dec, brain_reset=b.reset)
        r = M.play_fly("jump", agent, b, 11, detail=True, record=True)
        out.append((r["score"], r["ticks"], r["record"]["actions"], r["record"]["events"]))
    M2.arm_decoder("aux", b, None)
    assert out[0] == out[1]
    assert any(a != 0 for a in out[0][2])                   # the comparison is not all NOOP


# ---------------------------------------------------------------------------------------------
# the committed results
# ---------------------------------------------------------------------------------------------
@pytest.mark.skipif(not RESULTS.exists(), reason="M7c2 results not generated yet")
def test_results_shape_and_protocol():
    r = json.loads(RESULTS.read_text())
    m7c = json.loads(M.RESULTS_JSON.read_text())
    assert r["train_seeds"] == list(range(10)) and r["eval_seeds"] == list(range(10, 20))
    assert r["warmup_steps"] == 5000 and set(r["arms"]) == {"main", "aux"}
    assert r["arms"]["main"]["in_verdict_table"] and not r["arms"]["aux"]["in_verdict_table"]
    assert r["arms"]["aux"]["decoder"]["decoder"] == "GiantFibreDecoder"
    ch = r["decoder_choice"]
    assert r["arms"]["main"]["decoder"]["window_ms"] == ch["window_ms"] == r["arms"]["main"]["decoder_window_ms"]
    assert r["arms"]["main"]["decoder"]["min_spikes"] == ch["min_spikes"] and ch["J"] > 0
    for arm in ("main", "aux"):
        for game in ("jump", "dodge", "dodge_v2"):
            row = r["arms"][arm]["games"][game]
            assert row["p99"] == m7c["games"][game]["p99"] and row["k_grid"] == m7c["games"][game]["k_grid"]
            assert len(row["selection"]["grid"]) == 21 and row["k"] in row["k_grid"]
            best = max(x["mean"] for x in row["selection"]["grid"])
            assert next(x for x in row["selection"]["grid"] if x["k"] == row["k"])["mean"] == best
            assert [e["seed"] for e in row["fly"]["episodes"]] == list(range(10, 20))
            assert row["m7c"]["scores"] == m7c["games"][game]["fly"]["scores"]
    for game in ("jump", "dodge", "dodge_v2"):
        assert r["bots"][game]["one_way"]["scores"] == m7c["games"][game]["one_way"]["scores"]
        assert "bundle" in r["arms"]["main"]["games"][game] and "bundle" not in r["arms"]["aux"]["games"][game]
    assert r["arms"]["main"]["games"]["dodge_v2"]["verdict"]["brainless_verdict"] != "beat the brainless bot"
