"""M7a tests: the escape-circuit probe (docs/m7a-brief.md).

Pure logic (stimulus, sets, PASS, the three-number ignition rule, M3's selection walk) is
tested on synthetic inputs; one data test pins the wiring fact the experiment depends on.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from flysim.apps import m7a_escape_probe as m7


# ---------------------------------------------------------------------------
# stimulus
# ---------------------------------------------------------------------------
def test_loom_ramp_shape():
    r = m7.loom_ramp(100.0)
    assert len(r) == int(m7.TOTAL_MS / m7.DT_MS) == 800
    on0, on1 = int(m7.PRE_MS), int(m7.PRE_MS + m7.RAMP_MS)
    assert np.all(r[:on0] == 0) and np.all(r[on1:] == 0)          # off before and after
    assert r[on0] == 0.0                                           # starts from zero
    assert r[on1 - 1] == pytest.approx(100.0)                     # reaches a_in at the last ramp step
    assert np.all(np.diff(r[on0:on1]) > 0)                         # strictly growing
    # exponential: most of the drive in the last 100 ms (ASSUMPTION recorded in the module)
    assert r[on0 + 150] < 0.06 * 100 and r[on0 + 250] > 0.3 * 100
    assert np.all(m7.loom_ramp(0.0) == 0)


def test_a_in_grid_is_m3s_doubling_grid():
    assert m7.A_IN_GRID == [10.0 * 2 ** k for k in range(10)]
    assert m7.A_IN_GRID[-1] == 5120.0


# ---------------------------------------------------------------------------
# sets
# ---------------------------------------------------------------------------
class _Q:
    def __init__(self, table):
        self.table = table

    def by_type(self, t, side=None):
        v = self.table.get((t, side))
        if v is None or len(v) == 0:
            raise LookupError(f"no retained neuron matches type == {t!r} on side {side!r}")
        return np.asarray(v)


class _G:
    def __init__(self, table):
        self.query = _Q(table)


def _table(gf_r=(5,)):
    return {("LC4", "L"): [1, 2], ("LC4", "R"): [3], ("LPLC2", "L"): [6], ("LPLC2", "R"): [7, 8],
            ("DNp01", "L"): [4], ("DNp01", "R"): list(gf_r), ("TTMn", "L"): [9], ("TTMn", "R"): [10]}


def test_build_sets_and_stimulated():
    s = m7.build_sets(_G(_table()))
    assert s["LC4_L"].tolist() == [1, 2] and s["DNp01_R"].tolist() == [5]
    assert sorted(m7.stimulated(s, "left").tolist()) == [1, 2, 6]
    assert sorted(m7.stimulated(s, "right").tolist()) == [3, 7, 8]
    assert sorted(m7.stimulated(s, "both").tolist()) == [1, 2, 3, 6, 7, 8]
    assert m7.stimulated(s, "none").size == 0


def test_build_sets_raises_on_missing_or_ambiguous():
    t = _table()
    del t[("TTMn", "L")]
    with pytest.raises(LookupError):
        m7.build_sets(_G(t))                               # 0 hits is an exception (CLAUDE.md §3.1)
    with pytest.raises(ValueError, match="exactly one DNp01"):
        m7.build_sets(_G(_table(gf_r=(5, 11))))            # two giant fibres on one side


# ---------------------------------------------------------------------------
# PASS
# ---------------------------------------------------------------------------
def _row(gl=0, gr=0, first_l=None, first_r=None, late_rate=0.0, late_act=0.0, act=0.0):
    return {"DNp01_L_stim": gl, "DNp01_R_stim": gr, "DNp01_L_first_ms": first_l,
            "DNp01_R_first_ms": first_r, "late_network_rate_hz": late_rate,
            "late_active_fraction": late_act, "active_fraction": act}


def test_pass_single_seed():
    ctrl = [_row()]
    ok = m7.judge_side([_row(gl=3, gr=1, first_l=260.0)], ctrl, "left")
    assert ok["passes"] and ok["latency_ms"] == 60.0
    assert not m7.judge_side([_row(gl=2, gr=2)], ctrl, "left")["passes"]         # not lateralised
    assert not m7.judge_side([_row(gl=0, gr=0)], ctrl, "left")["passes"]         # ipsi silent
    assert not m7.judge_side([_row(gl=2)], [_row(gl=1)], "left")["passes"]       # control fired
    assert m7.judge_side([_row(gr=2)], ctrl, "right")["passes"]                  # sides mirror


def test_pass_needs_nine_of_ten_seeds_with_noise():
    ctrl = [_row() for _ in range(10)]
    nine = [_row(gl=1) for _ in range(9)] + [_row()]
    eight = [_row(gl=1) for _ in range(8)] + [_row(), _row()]
    assert m7.judge_side(nine, ctrl, "left")["passes"]
    assert not m7.judge_side(eight, ctrl, "left")["passes"]


# ---------------------------------------------------------------------------
# ignition: the three-number rule
# ---------------------------------------------------------------------------
def test_ignition_three_numbers():
    quiet = [_row()]
    # M3 reduction: silent control, no noise -> >= 1 % late-active is ignition
    assert m7.ignition([_row(late_act=0.012)], quiet)["ignited"]
    assert not m7.ignition([_row(late_act=0.008)], quiet)["ignited"]
    # stimulus run in runaway (absolute)
    assert m7.ignition([_row(late_rate=13.0)], quiet)["stim_runaway"]
    assert m7.ignition([_row(act=0.95)], quiet)["stim_runaway"]
    # control itself in runaway -> the operating point is unusable whatever the stimulus does
    ig = m7.ignition([_row(late_rate=24.0, late_act=0.25)], [_row(late_rate=24.0, late_act=0.25)])
    assert ig["control_runaway"] and ig["ignited"] and not ig["excess_ignited"]


def test_background_state_is_not_ignition():
    """The M2g false positive: a background with ~8 % late-active, unchanged by the
    stimulus, must NOT be called ignited (the absolute 1 % rule would have)."""
    ctrl = [_row(late_rate=0.4, late_act=0.08 + 0.002 * k) for k in range(10)]
    stim = [_row(late_rate=0.5, late_act=0.08 + 0.002 * k) for k in range(10)]
    assert not m7.ignition(stim, ctrl)["ignited"]
    # ... but a genuine stimulus-driven excess on top of it is
    stim_up = [_row(late_rate=0.5, late_act=0.11 + 0.002 * k) for k in range(10)]
    assert m7.ignition(stim_up, ctrl)["excess_ignited"]


# ---------------------------------------------------------------------------
# selection walk (M3)
# ---------------------------------------------------------------------------
def _pt(a, passes, ignited):
    return {"a_in": a, "judgement": {"passes": passes, "ignited": ignited}}


def test_selection_walk_stops_at_first_ignition():
    assert m7.select([_pt(10, False, False), _pt(20, True, False)])["selected_a_in"] == 20
    s = m7.select([_pt(10, False, False), _pt(20, False, True), _pt(40, True, False)])
    assert s["selected_a_in"] is None and s["stopped_at_ignition"] == 20   # never walks past it
    assert m7.select([_pt(10, False, False)])["selected_a_in"] is None


def test_recorded_selection_follows_the_rule():
    path = Path(__file__).resolve().parents[1] / "data-provenance" / "m7a-results.json"
    if not path.exists():
        pytest.skip("results not present")
    R = json.loads(path.read_text())
    for name, P in R["points"].items():
        assert m7.select(P["sweep"]) == P["selection"], name
        for p in P["sweep"]:
            j = p["judgement"]
            assert j["passes"] == (j["sides"]["left"]["passes"] and j["sides"]["right"]["passes"]
                                   and not j["ignited"])


# ---------------------------------------------------------------------------
# the data fact the experiment rests on
# ---------------------------------------------------------------------------
def test_looming_detectors_reach_only_the_ipsilateral_giant_fibre():
    from flysim.graph import Graph
    g = Graph.load()
    s = m7.build_sets(g)
    assert {k: len(v) for k, v in s.items()} == {
        "LC4_L": 71, "LC4_R": 55, "LPLC2_L": 94, "LPLC2_R": 91,
        "DNp01_L": 1, "DNp01_R": 1, "TTMn_L": 1, "TTMn_R": 1}
    for t in m7.INPUT_TYPES:
        for side, other in (("L", "R"), ("R", "L")):
            ipsi, contra = int(s[f"DNp01_{side}"][0]), int(s[f"DNp01_{other}"][0])
            n_ipsi = 0
            for i in s[f"{t}_{side}"]:
                post, w = g.successors(int(i))
                assert not (post == contra).any(), f"{t}_{side} reaches the contralateral GF"
                n_ipsi += bool((post == ipsi).any())
            assert n_ipsi == len(s[f"{t}_{side}"])          # every cell reaches its own GF
