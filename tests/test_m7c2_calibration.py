"""M7c2a tests: the count-window decoder, its equivalence with M7c's decoder at (50 ms, 1 spike),
the calibration's scoring and selection rules, and the shape of the committed results.

The pure tests run anywhere; the brain test needs CUDA and the graph cache and skips otherwise.
"""
import json
import random
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
import torch

from flysim.apps import m7c2_decoder_calibration as C
from flysim.probe import motor
from flysim.probe.motor import CountWindowDecoder, GiantFibreDecoder

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data-provenance" / "m7c2-decoder.json"


# ---------------------------------------------------------------------------------------------
# the decoder (pure)
# ---------------------------------------------------------------------------------------------
def test_count_window_decoder_rule():
    d = CountWindowDecoder(100.0, 2)
    assert d.decode([], 500.0) == "NOOP"
    assert d.decode([(410.0, "L"), (450.0, "L")], 500.0) == "RIGHT"        # left fibre -> move right
    assert d.decode([(410.0, "R"), (499.0, "R")], 500.0) == "LEFT"
    assert d.decode([(401.0, "L"), (402.0, "L"), (403.0, "R"), (404.0, "R")], 500.0) == "JUMP"
    assert d.decode([(450.0, "L"), (460.0, "R"), (470.0, "L")], 500.0) == "RIGHT"   # R has only 1
    assert d.counts([(450.0, "L"), (460.0, "R"), (470.0, "L")], 500.0) == (2, 1)


def test_count_window_is_half_open_and_ignores_the_rest_of_the_history():
    d = CountWindowDecoder(50.0, 1)
    assert d.counts([(450.0, "L")], 500.0) == (1, 0)               # t = now - W is in
    assert d.counts([(449.0, "L")], 500.0) == (0, 0)               # earlier is out
    assert d.counts([(500.0, "L")], 500.0) == (0, 0)               # t = now is not yet seen
    hist = [(t, "R") for t in range(0, 449)] + [(470.0, "L")]
    assert d.decode(hist, 500.0) == "RIGHT"                         # a long history is windowed


def test_count_window_decoder_arguments():
    for bad in ((0.0, 1), (-5.0, 1), (50.0, 0), (50.0, 1.5)):
        with pytest.raises(ValueError):
            CountWindowDecoder(*bad)
    with pytest.raises(ValueError):
        CountWindowDecoder(50.0, 1).decode([(1.0, "X")], 10.0)
    assert CountWindowDecoder(200.0, 3).describe() == {
        "decoder": "CountWindowDecoder", "window_ms": 200.0, "min_spikes": 3,
        "rule": "count GF spikes per side in [now - window_ms, now); both >= n -> JUMP; "
                "left only -> RIGHT; right only -> LEFT; neither -> NOOP"}
    src = Path(motor.__file__).read_text()
    for name in ("DNp01", "LC4", "LPLC2", "TTMn"):                  # no cell-type names in the decoder
        assert name not in src


def _m7c_windowed(events, now):
    """What M7c's brain glue handed to GiantFibreDecoder at the end of a tick (brain.py)."""
    return [(t, s) for (t, s) in events if t >= now - 50.0]


def test_w50_n1_equals_the_m7c_decoder_on_random_streams():
    rng = random.Random(7)
    new, old = CountWindowDecoder(50.0, 1), GiantFibreDecoder(50.0)
    ends = C.tick_ends(3000.0)
    for _ in range(60):
        rate = rng.choice([0.002, 0.01, 0.05])
        ev = sorted(((float(t), rng.choice("LR")) for t in range(3000) if rng.random() < rate))
        for e in ends:
            past = [x for x in ev if x[0] < e]                        # the glue only has the past
            assert new.decode(past, float(e)) == old.decode(_m7c_windowed(past, e))
            assert new.decode(ev, float(e)) == old.decode(_m7c_windowed(past, e))   # future ignored


# ---------------------------------------------------------------------------------------------
# calibration rules (pure)
# ---------------------------------------------------------------------------------------------
def test_grid_is_the_pre_registered_twelve():
    assert C.candidates() == [(w, n) for w in (50.0, 100.0, 200.0, 300.0) for n in (1, 2, 3)]
    assert C.M7C_DECODER in C.candidates()
    assert C.A_IN == (640.0, 1280.0, 2560.0, 5120.0)
    assert C.P2 == dict(synapse="conductance", g=5.36539e-4, noise_sigma=29.13, adapt_g_b=8.0, dt=1.0)
    assert C.CORRECT == {"left": "RIGHT", "right": "LEFT", "both": "JUMP"}


def test_tick_schedule_is_the_game_schedule():
    from flysim.apps.games import brain as BR
    assert all(C.tick_end_ms(k) == BR.tick_end_ms(k) for k in range(2000))
    e = C.tick_ends(C.STIM_MS)
    assert len(e) == 48 and e[11] == C.RAMP_ONSET_MS and e[-1] == 800
    assert len(C.tick_ends(C.NOISE_MS)) == 1800


def test_the_600ms_window_holds_36_decisions_after_its_start():
    e = C.tick_ends(C.STIM_MS)
    w = C.in_window(e, C.RAMP_ONSET_MS)
    assert len(w) == 36 and e[w[0]] == 216 and e[w[-1]] == 800      # ticks 12..47
    w2 = C.in_window(e, C.RAMP_ONSET_MS, inclusive_start=True)
    assert len(w2) == 36 and e[w2[0]] == 200                          # the other reading


def test_trial_outcome():
    e = C.tick_ends(C.STIM_MS)
    acts = ["NOOP"] * len(e)
    assert C.trial_outcome(acts, e, "left") == dict(first_action=None, hit=False, latency_ms=None)
    acts[11] = "LEFT"                                  # at ramp onset: decided before the stimulus
    acts[20] = "RIGHT"
    out = C.trial_outcome(acts, e, "left")
    assert out["hit"] and out["first_action"] == "RIGHT" and out["latency_ms"] == e[20] - 200
    assert not C.trial_outcome(acts, e, "left", inclusive_start=True)["hit"]   # LEFT came first
    acts[15] = "JUMP"
    assert C.trial_outcome(acts, e, "both")["hit"] and not C.trial_outcome(acts, e, "left")["hit"]


def test_false_alarm_windows():
    e = C.tick_ends(3000.0)
    acts = ["NOOP"] * len(e)
    hit, tot = C.false_alarm_windows(acts, e, 3000.0)
    assert hit == 0 and tot == sum(1 for x in e if x + 600 <= 3000)
    i = int(np.flatnonzero(e == 1000)[0])               # one action at 1000 ms
    acts[i] = "LEFT"
    hit, _ = C.false_alarm_windows(acts, e, 3000.0)
    assert hit == sum(1 for x in e if x < 1000 <= x + 600 and x + 600 <= 3000)


def test_selection_max_j_then_small_w_then_small_n():
    def row(w, n, j):
        return {"window_ms": w, "min_spikes": n, "J_exact": str(j)}
    rows = [row(50.0, 1, Fraction(1, 3)), row(100.0, 2, Fraction(2, 5)), row(200.0, 1, Fraction(2, 5)),
            row(100.0, 1, Fraction(2, 5))]
    s = C.select(rows)
    assert (s["window_ms"], s["min_spikes"]) == (100.0, 1) and s["beats_noise"]
    assert s["tied_at_max"] == [[100.0, 1], [100.0, 2], [200.0, 1]]
    s0 = C.select([row(50.0, 1, Fraction(0)), row(100.0, 1, Fraction(-1, 7))])
    assert not s0["beats_noise"] and s0["stage2"].startswith("do not run")


def test_score_on_a_constructed_record():
    # one stimulus trial per condition with two ipsilateral spikes after onset; quiet noise
    gf_one = lambda side: {"L": [300, 320] if side in "LB" else [], "R": [300, 320] if side in "RB" else []}
    stim = [dict(condition=c, a_in=640.0, seed=0, gf=gf_one(s)) for c, s in
            (("left", "L"), ("right", "R"), ("both", "B"))]
    noise = [dict(seed=0, gf={"L": [], "R": []})]
    r = C.score(stim, noise, 100.0, 2)
    assert r["hits"] == 3 and r["false_alarm_windows"] == 0 and r["_J"] == 1
    assert r["latency_ms_median_hits"] == 333 - 200                 # first tick end after 320
    r3 = C.score(stim, noise, 100.0, 3)                               # needs a third spike: none
    assert r3["hits"] == 0 and r3["_J"] == 0


# ---------------------------------------------------------------------------------------------
# the brain glue itself (CUDA + graph cache): W50/n1 == GiantFibreDecoder tick by tick
# ---------------------------------------------------------------------------------------------
def test_w50_n1_equals_the_m7c_decoder_on_the_brain():
    from flysim.graph.constants import CACHE_NPZ
    if not torch.cuda.is_available() or not CACHE_NPZ.exists():
        pytest.skip("CUDA and the graph cache are required")
    from flysim.apps.games import brain as BR
    from flysim.graph import Graph
    from flysim.probe.sets import build_probe_sets, load_roi
    g = Graph.load()
    es = BR.EscapeSets.from_probe_sets(build_probe_sets(g, load_roi(g), profile="escape"))
    b = BR.FlyBrain(g, es, k=3000.0)
    b.reset(1)
    new, old = CountWindowDecoder(50.0, 1), GiantFibreDecoder(50.0)
    acted = 0
    for k in range(600):                                               # 10 s, a loom every 2 s
        loom = 0.02 if (k % 120) > 100 else 0.0
        windowed = b.step(loom, 0.0 if k % 240 < 120 else loom)
        a = new.decode(b.gf_events, b.t_ms)
        assert a == old.decode(windowed)
        acted += a != "NOOP"
    assert acted > 0 and len(b.gf_events) > 0


# ---------------------------------------------------------------------------------------------
# committed results
# ---------------------------------------------------------------------------------------------
@pytest.mark.skipif(not RESULTS.exists(), reason="M7c2a results not generated yet")
def test_recorded_results_follow_the_rules():
    R = json.loads(RESULTS.read_text())
    assert R["operating_point"]["params"]["noise_sigma"] == 29.13
    assert R["operating_point"]["params"]["adapt_g_b"] == 8.0
    assert R["seeds"] == list(range(10))
    noise, stim = R["records"]["noise"], R["records"]["stimulus"]
    assert len(noise) == 10 and len(stim) == 3 * 4 * 10
    assert [(r["window_ms"], r["min_spikes"]) for r in R["table"]] == C.candidates()
    rows = C.table(stim, noise)                                       # recompute from the raw spikes
    for got, want in zip(rows, R["table"]):
        assert got["J_exact"] == want["J_exact"] and got["hits"] == want["hits"]
        assert got["false_alarm_windows"] == want["false_alarm_windows"]
    assert C.select(R["table"]) == R["selection"]
    assert R["m7a2_reproduction"]["identical"] is True


@pytest.mark.skipif(not RESULTS.exists(), reason="M7c2a results not generated yet")
def test_w50_n1_equals_the_m7c_decoder_on_the_recorded_noise():
    R = json.loads(RESULTS.read_text())
    new, old = CountWindowDecoder(50.0, 1), GiantFibreDecoder(50.0)
    ends = C.tick_ends(C.NOISE_MS)
    for r in R["records"]["noise"][:3]:
        ev = [e for e in C.events_from(r["gf"]) if e[0] >= 0]       # the glue starts at tick time 0
        acts = C.decide(ev, ends, new)
        for e, a in zip(ends, acts):
            past = [x for x in ev if x[0] < e]
            assert a == old.decode(_m7c_windowed(past, e))
