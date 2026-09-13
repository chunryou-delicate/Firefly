"""M3 tests: probe set sizes > 0 and disjoint, rate binning, run.json contract keys."""
import json

import numpy as np
import pytest

from flysim.graph import Graph
from flysim.probe import bin_spikes, build_probe_sets, load_roi, neuron_coords, window_mean_rate, write_run_json
from flysim.probe.export import REQUIRED_KEYS, sample_spikes, validate_run_doc


@pytest.fixture(scope="session")
def g() -> Graph:
    return Graph.load(build_if_missing=True)


@pytest.fixture(scope="session")
def roi(g):
    return load_roi(g)


@pytest.fixture(scope="session")
def sets(g, roi):
    return build_probe_sets(g, roi)


def test_sets_nonempty_disjoint_cover(g, sets):
    assert sets.names[-1] == "rest"
    for k in ("JO_AB_L", "JO_AB_R", "JO_post_L", "JO_post_R", "SAD_L", "SAD_R", "WED_L", "WED_R", "pC1_L", "pC1_R"):
        assert k in sets.names and len(sets.members[k]) > 0
    allidx = np.concatenate(list(sets.members.values()))
    assert len(allidx) == g.n and len(np.unique(allidx)) == g.n           # partition of all neurons
    for r, k in enumerate(sets.names):
        assert np.all(sets.region_of[sets.members[k]] == r)
    assert len(sets.members["JO_AB_L"]) + len(sets.members["JO_AB_R"]) == 114   # brief: 138 typed - 24 without sites
    assert "removed_by_precedence" in sets.log and "raw_sizes" in sets.log


def test_bin_and_window(sets):
    n_bins = 100
    idx = np.array([sets.members["JO_AB_L"][0]] * 3 + [sets.members["rest"][0]], dtype=np.int32)
    t = np.array([0, 5, 15, 990], dtype=np.int32)                  # dt = 0.1 ms -> bins 0, 0, 1, 99
    rt = bin_spikes(t, idx, 0.1, 1.0, n_bins, sets)
    r = sets.names.index("JO_AB_L")
    assert rt.counts[r, 0] == 2 and rt.counts[r, 1] == 1 and rt.counts.sum() == 4
    assert rt.rates[r, 0] == pytest.approx(2 / (rt.sizes[r] * 1e-3))
    assert window_mean_rate(rt, "JO_AB_L", 0, 2) == pytest.approx(3 / (rt.sizes[r] * 2e-3))
    assert window_mean_rate(rt, "rest", 0, 50) == 0
    assert rt.active_neurons == 2 and rt.n_spikes == 4
    with pytest.raises(ValueError):
        bin_spikes(np.array([2000]), idx[:1], 0.1, 1.0, n_bins, sets)


def test_sample_spikes_cap():
    t = np.repeat(np.arange(5), 100)
    i = np.arange(500)
    st, si, ratio = sample_spikes(t, i, 5, 10, seed=1)
    assert len(st) == 50 and ratio == 0.1 and np.all(np.bincount(st, minlength=5) == 10)
    assert np.all(t[si] == st)                                           # (t, idx) pairs preserved
    st2, _, r2 = sample_spikes(t, i, 5, 1000)
    assert len(st2) == 500 and r2 == 1.0


def test_run_json_contract(g, roi, sets, tmp_path):
    coords = neuron_coords(g, roi)
    assert coords[2]["missing_placed_off_graph"] < 1000 and np.isfinite(coords[0]).all()
    t = np.array([100, 101, 300], dtype=np.int32)
    idx = np.array([sets.members["JO_AB_L"][0], sets.members["JO_post_R"][0], sets.members["rest"][5]], dtype=np.int32)
    rt = bin_spikes(t, idx, 1.0, 1.0, 400, sets)
    env = np.zeros(400); env[100:300] = 1
    doc = write_run_json(tmp_path / "run.json", run_id="t", sensory_mode="rate", dt_ms=1.0, rt=rt, sets=sets,
                         coords=coords, input_label="x", envelope=env, spikes=(rt.t_bin, idx),
                         engine_params={"g": 0.336}, adapter={"mode": "rate"}, stimulus={"kind": "test"})
    back = json.loads((tmp_path / "run.json").read_text())
    validate_run_doc(back)
    for sec, keys in REQUIRED_KEYS.items():
        assert all(k in back[sec] for k in keys)
    assert back["meta"]["sensory_mode"] == "rate" and back["meta"]["n_neurons"] == g.n
    assert back["regions"] == sets.names and len(back["frames"]["rates"]) == len(sets.names)
    assert max(max(r) for r in back["frames"]["rates"]) == 1.0 and back["meta"]["rate_norm_hz"] > 0
    assert back["spikes"]["t_bin"] == [100, 101, 300] and back["meta"]["spike_sample_ratio"] == 1.0
    with pytest.raises(ValueError):
        bad = dict(doc); bad["meta"] = {**doc["meta"], "sensory_mode": "envelope"}; validate_run_doc(bad)
