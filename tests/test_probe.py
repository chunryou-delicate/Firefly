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


# ---------------------------------------------------------------------------
# M7: probe profiles (docs/m7-profiles-brief.md)
# ---------------------------------------------------------------------------
import hashlib

# Fingerprint of the auditory sets taken with the UNMODIFIED sets.py at commit 2c2a66c,
# before profiles existed. sha256 of each index array as int64.
AUDITORY_BEFORE_PROFILES = {
    "names": ["JO_AB_L", "JO_AB_R", "JO_post_L", "JO_post_R", "JO_post_unk", "pC1_L", "pC1_R",
              "AMMCtype_L", "AMMCtype_R", "SAD_L", "SAD_R", "SAD_unk", "WED_L", "WED_R", "rest"],
    "members": {
        "JO_AB_L": "a9b13791c69e3b331d3926e9b3c085f09db56ed2f5e2477ceb83a088f85786d1",
        "JO_AB_R": "c6383f418cc1fb62e6de42eb2934d3ed3ef42f779ef2af9798656306bbe64071",
        "JO_post_L": "728552a86f825277480d06afa28a7568727fee89bafa7a83b4a4ae047092a886",
        "JO_post_R": "2392fe3fe367bdf308c14211d0d39a8888274e5620d935513cb5556769b91039",
        "JO_post_unk": "88113e8c2a3a35725b9bf1b5c03dd3d55a02558457f69e0887033807a262f94a",
        "pC1_L": "b8de17222ac12d3ed8fc2e117ce89d99508e3272ea35e44018731605fb7e5f4b",
        "pC1_R": "fd3e465857fa172e57a0a6061b7121c078b1911e0b474997f1c9127c44629bd2",
        "AMMCtype_L": "aa967d0472f626aee65800cc566821aab688f1d8cff6fbff479cbf440c3b26fd",
        "AMMCtype_R": "5bf7a9b150359c19fec600a70f64a90dcb0dd325864bf721fb6511763226f2ea",
        "SAD_L": "82aabae7fcf42196e01ab9def6736f85fb87fc675740e01e5e13014114769b2c",
        "SAD_R": "223d58479fbbbb137ba2ae7b0be978132c499d38616987ed41ecfb4481129675",
        "SAD_unk": "9609f8b5c61cdee7ce15a59187297e32a146e66f39e94763dfb9361d731eddbd",
        "WED_L": "664f24fb9ca3942a0b3348142137ebb6548ac66599144d2b89f55dec6545c1bd",
        "WED_R": "4c4ba59a6380f347f83bcff63327ac41fc16115d2c3bb434b74b725ad12f30f0",
        "rest": "cf2d622930852a2a958e648e82d664d7ea9cf082c23fce10d94db79e6c5af304",
    },
    "region_of": "f8c9309e311bcab6e922ff6348a542811cb309fbded7a302c7238c6266cc9720",
}


def _h(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a, dtype=np.int64)).tobytes()).hexdigest()


def test_auditory_profile_is_bit_identical_to_the_pre_profile_sets(sets):
    """The default call (no profile argument) reproduces the pre-profile index arrays."""
    assert sets.profile == "auditory"
    assert sets.names == AUDITORY_BEFORE_PROFILES["names"]
    assert {k: _h(v) for k, v in sets.members.items()} == AUDITORY_BEFORE_PROFILES["members"]
    assert _h(sets.region_of) == AUDITORY_BEFORE_PROFILES["region_of"]
    assert sets.region_of.dtype == np.int16


def test_explicit_auditory_equals_default(g, roi, sets):
    a = build_probe_sets(g, roi, profile="auditory")
    assert a.names == sets.names and np.array_equal(a.region_of, sets.region_of)
    assert all(np.array_equal(a.members[k], sets.members[k]) for k in sets.names)
    # the auditory log is unchanged too (apps write it into their results JSON)
    assert sorted(a.log) == ['AMMCtype', 'JO_AB', 'JO_post', 'k_min', 'pC1', 'precedence',
                             'raw_sizes', 'removed_by_precedence']


@pytest.fixture(scope="session")
def escape(g, roi):
    return build_probe_sets(g, roi, profile="escape")


def test_escape_profile_sets_and_sizes(g, escape):
    """Sizes must equal M7a's (docs/m7a-report.md §1)."""
    assert escape.profile == "escape"
    assert escape.names == ["LC4_L", "LC4_R", "LPLC2_L", "LPLC2_R",
                            "DNp01_L", "DNp01_R", "TTMn_L", "TTMn_R", "rest"]
    sizes = escape.sizes()
    assert {k: v for k, v in sizes.items() if k != "rest"} == {
        "LC4_L": 71, "LC4_R": 55, "LPLC2_L": 94, "LPLC2_R": 91,
        "DNp01_L": 1, "DNp01_R": 1, "TTMn_L": 1, "TTMn_R": 1}
    allidx = np.concatenate(list(escape.members.values()))
    assert len(allidx) == g.n and len(np.unique(allidx)) == g.n           # still a partition
    for r, k in enumerate(escape.names):
        assert np.all(escape.region_of[escape.members[k]] == r)
    assert all(v == 0 for v in escape.log["removed_by_precedence"].values())  # types are disjoint
    # the same cells M7a used: Graph.side, the type, nothing else
    for k in escape.names[:-1]:
        t, s = k.rsplit("_", 1)
        assert np.array_equal(escape.members[k], np.sort(g.query.by_type(t, side=s)))


def test_unknown_profile_raises(g, roi):
    with pytest.raises(ValueError, match="unknown probe profile"):
        build_probe_sets(g, roi, profile="visual")


def test_escape_profile_raises_on_a_missing_type(g, roi, monkeypatch):
    """0 hits is an exception, not an empty set (CLAUDE.md §3.1)."""
    import flysim.probe.sets as S
    monkeypatch.setattr(S, "ESCAPE_TYPES", ["LC4", "NOT_A_REAL_TYPE"])
    with pytest.raises(LookupError):
        build_probe_sets(g, roi, profile="escape")
