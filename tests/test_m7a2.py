"""M7a2 tests: steady-state recheck of the escape circuit (docs/m7a2-brief.md)."""
import json
from pathlib import Path

import numpy as np
import pytest

from flysim.apps import m7a2_steady_probe as m2
from flysim.apps import m7a_escape_probe as m7a


def test_stimulus_is_m7a_after_a_five_second_warmup():
    s = m2.stimulus(100.0)
    assert len(s) == 5800
    assert np.all(s[:5000] == 0)                                  # warm-up: no input at all
    assert np.array_equal(s[5000:], m7a.loom_ramp(100.0))         # then exactly M7a's stimulus
    assert m2.STIM_WINDOW_MS == (5200.0, 5500.0)
    assert m2.LATE_WINDOW_MS == (5600.0, 5800.0)
    assert np.all(m2.stimulus(0.0) == 0)


def test_steady_entry_definition():
    # settles at 0.5 s: forward 1-s average within 20 % of the last-5-s mean from then on
    r = np.r_[np.zeros(5), np.ones(95)]
    assert m2.steady_entry_ms(r) == 400.0          # [0.4, 1.4) s already averages 0.9
    assert m2.steady_entry_ms(np.ones(100)) == 0.0
    assert m2.steady_entry_ms(np.zeros(100)) is None              # zero reference: undefined
    # a late excursion pushes the entry past it
    r2 = np.ones(100)
    r2[60:62] = 4.0
    assert m2.steady_entry_ms(r2) > 6000.0


def _r(seed, gl=0, gr=0, fl=None, fr=None):
    return {"seed": seed, "DNp01_L_stim": gl, "DNp01_R_stim": gr,
            "DNp01_L_first_ms": fl, "DNp01_R_first_ms": fr}


def test_paired_judgement_against_spontaneous_firing():
    none = {s: _r(s, gl=1, gr=1) for s in range(10)}                # both GFs fire spontaneously
    # stimulus adds one ipsilateral spike in every seed: difference 1.0, SEM 0 -> passes
    up = [_r(s, gl=2, gr=1, fl=5300.0) for s in range(10)]
    j = m2.judge_side_paired(up, none, "left")
    assert j["passes"] and j["diff_ipsi_mean"] == 1.0 and j["diff_contra_mean"] == 0.0
    assert j["latency_ms_median"] == 100.0 and j["signal_to_spontaneous"] == 2.0
    # same counts as spontaneous -> no difference -> fails, however many spikes there are
    same = [_r(s, gl=1, gr=1) for s in range(10)]
    assert not m2.judge_side_paired(same, none, "left")["passes"]
    # a difference that does not clear 3 SEM fails
    noisy = [_r(s, gl=1 + (s % 2) * 2 - (s == 0), gr=1) for s in range(10)]
    jn = m2.judge_side_paired(noisy, none, "left")
    assert jn["diff_ipsi_mean"] > 0 and not jn["exceeds_3sem"]
    # contralateral rising as much as ipsilateral fails the lateral test
    both_up = [_r(s, gl=2, gr=2) for s in range(10)]
    assert not m2.judge_side_paired(both_up, none, "left")["passes"]


def test_first_spike_side_and_laterality():
    assert m2.first_spike_side(_r(0)) is None
    assert m2.first_spike_side(_r(0, gl=1, fl=5210.0)) == "L"
    assert m2.first_spike_side(_r(0, gl=1, gr=1, fl=5300.0, fr=5250.0)) == "R"
    assert m2.first_spike_side(_r(0, gl=1, gr=1, fl=5250.0, fr=5250.0)) == "tie"
    by = {"left": [_r(0, gl=1, gr=1, fl=5210.0, fr=5250.0), _r(1)],
          "right": [_r(0, gl=1, gr=1, fl=5210.0, fr=5250.0)]}
    lat = m2.laterality(by, [_r(0, gl=1, fl=5300.0), _r(1, gr=1, fr=5300.0)])
    assert lat["runs_with_a_spike"] == 2 and lat["ipsi_first"] == 1          # one hit, one miss
    assert lat["none_left_first_fraction"] == 0.5                            # chance baseline


def test_recorded_results_follow_the_rules():
    path = Path(__file__).resolve().parents[1] / "data-provenance" / "m7a2-results.json"
    if not path.exists():
        pytest.skip("results not present")
    R = json.loads(path.read_text())
    assert R["protocol"]["warmup_ms"] == 5000.0
    for sigma, A in R["arms"].items():
        assert m7a.select(A["sweep"]) == A["selection"], sigma
        none_by_seed = {r["seed"]: r for r in A["none"]}
        for p in A["sweep"]:
            j = p["judgement"]
            for c in ("left", "right"):                         # verdicts recompute from the raw runs
                assert m2.judge_side_paired(p["runs"][c], none_by_seed, c)["passes"] == j["sides"][c]["passes"]
            assert j["passes"] == (j["sides"]["left"]["passes"] and j["sides"]["right"]["passes"]
                                   and not j["ignited"])
