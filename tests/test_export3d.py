"""M6b tests: the 3D neuron point cloud (file 1 of docs/m6-3d-contract.md).

The layout tests run on a small synthetic graph, so they need neither the connectome cache
nor a GPU. The tests that check the real file (counts, index order shared with run.json)
skip themselves when the cache is missing.
"""
import json
import re

import numpy as np
import pandas as pd
import pytest

from flysim.probe import export3d as E3
from flysim.probe.sets import ProbeSets

CONTRACT = (__import__("pathlib").Path(__file__).resolve().parents[1]
            / "docs" / "m6-3d-contract.md")


class FakeGraph:
    """Just the attributes export3d reads off flysim.graph.Graph."""

    def __init__(self, soma_xyz, has_soma, body_ids):
        self.soma_xyz = np.asarray(soma_xyz, dtype=np.float32)
        self.has_soma_xyz = np.asarray(has_soma, dtype=bool)
        self.body_ids = np.asarray(body_ids, dtype=np.int64)

    @property
    def n(self):
        return len(self.body_ids)


@pytest.fixture
def fake():
    """6 neurons: 2 with a soma, 3 with only a centroid, 1 with neither."""
    n = 6
    body_ids = np.arange(100, 100 + n, dtype=np.int64)
    soma = np.full((n, 3), np.nan, dtype=np.float32)
    soma[0] = [10.0, 20.0, 30.0]
    soma[3] = [40.0, 50.0, 60.0]
    has_soma = np.array([1, 0, 0, 1, 0, 0], dtype=bool)
    roi = pd.DataFrame({
        "bodyId": body_ids,
        "cx": [1.0, 11.0, 21.0, 2.0, 31.0, np.nan],
        "cy": [2.0, 12.0, 22.0, 3.0, 32.0, np.nan],
        "cz": [3.0, 13.0, 23.0, 4.0, 33.0, np.nan],
    })
    members = {"A": np.array([0, 1]), "B": np.array([2, 3]), "rest": np.array([4, 5])}
    region_of = np.array([0, 0, 1, 1, 2, 2], dtype=np.int16)
    sets = ProbeSets(names=list(members), members=members, region_of=region_of, log={})
    return FakeGraph(soma, has_soma, body_ids), roi, sets


# ---------------------------------------------------------------------------------------
# positions
# ---------------------------------------------------------------------------------------
def test_source_priority_and_nan(fake):
    graph, roi, _ = fake
    pos, src, counts = E3.neuron_positions_3d(graph, roi)
    assert pos.dtype == np.float32 and pos.shape == (6, 3) and src.dtype == np.uint8
    # soma wins over the centroid even when both exist
    assert np.allclose(pos[0], [10, 20, 30]) and src[0] == E3.SRC_SOMA
    assert np.allclose(pos[3], [40, 50, 60]) and src[3] == E3.SRC_SOMA
    # centroid fills in
    assert np.allclose(pos[1], [11, 12, 13]) and src[1] == E3.SRC_CENTROID
    assert np.allclose(pos[4], [31, 32, 33]) and src[4] == E3.SRC_CENTROID
    # neither -> NaN, never (0, 0, 0)
    assert np.isnan(pos[5]).all() and src[5] == E3.SRC_NONE
    assert counts == {"soma": 2, "centroid": 3, "none": 1}
    assert int(np.isnan(pos).any(axis=1).sum()) == counts["none"]


def test_misaligned_roi_is_an_error(fake):
    graph, roi, _ = fake
    with pytest.raises(ValueError):
        E3.neuron_positions_3d(graph, roi.iloc[:-1])
    shuffled = roi.copy()
    shuffled["bodyId"] = shuffled["bodyId"].to_numpy()[::-1]
    with pytest.raises(ValueError):
        E3.neuron_positions_3d(graph, shuffled)


# ---------------------------------------------------------------------------------------
# file layout
# ---------------------------------------------------------------------------------------
def test_layout_offsets_and_roundtrip(fake, tmp_path):
    graph, roi, sets = fake
    doc = E3.write(tmp_path / "neurons-3d.bin", tmp_path / "neurons-3d.json",
                   graph=graph, roi=roi, sets=sets)
    n = graph.n
    a = doc["arrays"]
    assert a["pos"] == {"offset": 0, "length": 3 * n, "dtype": "float32"}
    assert a["set"] == {"offset": 12 * n, "length": n, "dtype": "uint8"}
    assert a["src"] == {"offset": 13 * n, "length": n, "dtype": "uint8"}
    assert a["body"] == {"offset": 14 * n, "length": n, "dtype": "uint32"}   # contract v1.3
    assert (tmp_path / "neurons-3d.bin").stat().st_size == 18 * n
    assert doc["n"] == n and doc["voxel_nm"] == 8 and doc["order"] == "graph neuron index"
    assert doc["sets"] == list(sets.names)
    assert sum(doc["src_counts"].values()) == n
    back = E3.read_arrays(tmp_path / "neurons-3d.bin", tmp_path / "neurons-3d.json")
    pos, src, _ = E3.neuron_positions_3d(graph, roi)
    assert np.array_equal(back["pos"], pos, equal_nan=True)
    assert np.array_equal(back["src"], src)
    assert np.array_equal(back["set"], np.asarray(sets.region_of, dtype=np.uint8))
    assert back["body"].dtype == np.uint32
    assert np.array_equal(back["body"].astype(np.int64), graph.body_ids)
    # bbox covers only the finite positions
    finite = np.isfinite(pos).all(axis=1)
    assert np.allclose(doc["bbox"]["min"], pos[finite].min(axis=0))
    assert np.allclose(doc["bbox"]["max"], pos[finite].max(axis=0))


def test_json_is_machine_readable_and_typed(fake, tmp_path):
    graph, roi, sets = fake
    E3.write(tmp_path / "n.bin", tmp_path / "n.json", graph=graph, roi=roi, sets=sets)
    doc = json.loads((tmp_path / "n.json").read_text())
    for key in ("n", "voxel_nm", "order", "bbox", "sets", "src_counts", "arrays",
                "generated", "graph_cache_md5"):
        assert key in doc, key
    assert set(doc["src_counts"]) == {"soma", "centroid", "none"}
    assert re.match(r"^\d{4}-\d{2}-\d{2}T", doc["generated"])


def test_too_many_sets_is_refused(fake, tmp_path):
    graph, roi, sets = fake
    pos, src, _ = E3.neuron_positions_3d(graph, roi)
    z, b = np.zeros(graph.n, np.uint8), graph.body_ids
    with pytest.raises(ValueError):
        E3.build_doc(pos, z, src, [f"s{i}" for i in range(257)], b)
    with pytest.raises(ValueError):
        E3.build_doc(pos[:-1], z, src, ["a"], b)
    with pytest.raises(ValueError):                       # body of the wrong length
        E3.build_doc(pos, z, src, ["a"], b[:-1])


def test_all_positions_missing_is_refused(tmp_path):
    n = 3
    graph = FakeGraph(np.full((n, 3), np.nan, np.float32), np.zeros(n, bool), np.arange(n))
    roi = pd.DataFrame({"bodyId": np.arange(n), "cx": [np.nan] * n, "cy": [np.nan] * n,
                        "cz": [np.nan] * n})
    pos, src, counts = E3.neuron_positions_3d(graph, roi)
    assert counts["none"] == n
    with pytest.raises(ValueError):
        E3.build_doc(pos, np.zeros(n, np.uint8), src, ["rest"], graph.body_ids)


def test_body_ids_must_fit_in_uint32(fake):
    """Contract v1.3: an id too large is an error, never a silent truncation."""
    graph, roi, sets = fake
    ok = E3.body_ids_uint32(np.array([0, 10001, 1_571_825_087, E3.UINT32_MAX], dtype=np.int64))
    assert ok.dtype == np.uint32 and int(ok[2]) == 1_571_825_087
    for bad in ([E3.UINT32_MAX + 1], [-1], [2**33]):
        with pytest.raises(ValueError) as e:
            E3.body_ids_uint32(np.array(bad, dtype=np.int64))
        assert "uint32" in str(e.value)
    with pytest.raises(ValueError):
        E3.body_ids_uint32(np.array([1.5]))
    with pytest.raises(ValueError):
        E3.body_ids_uint32(np.zeros((2, 2), dtype=np.int64))
    # and the whole build refuses it rather than writing a truncated array
    pos, src, _ = E3.neuron_positions_3d(graph, roi)
    huge = graph.body_ids.copy()
    huge[0] = E3.UINT32_MAX + 7
    with pytest.raises(ValueError):
        E3.build_doc(pos, np.zeros(graph.n, np.uint8), src, list(sets.names), huge)


# ---------------------------------------------------------------------------------------
# bundle listing / assets (no skeleton files needed)
# ---------------------------------------------------------------------------------------
def test_skeleton_bundles_and_assets(tmp_path):
    assert E3.skeleton_bundles(tmp_path) == []
    assert E3.available_assets(tmp_path) == {"neurons_3d": False, "skeleton_sets": []}
    (tmp_path / "skel-pC1.json").write_text(json.dumps({"set": "pC1", "n_neurons": 156,
                                                        "n_segments": 28114}))
    (tmp_path / "skel-pC1.bin").write_bytes(b"\0" * 48)
    (tmp_path / "skel-broken.json").write_text("{not json")
    (tmp_path / "skel-nobin.json").write_text(json.dumps({"n_neurons": 1, "n_segments": 2}))
    got = {b["name"]: b for b in E3.skeleton_bundles(tmp_path)}
    assert set(got) == {"pC1", "broken", "nobin"}
    assert got["pC1"] == {"name": "pC1", "n_neurons": 156, "n_segments": 28114, "bytes": 48}
    assert "error" in got["broken"] and "error" in got["nobin"]
    # only complete bundles are advertised
    assets = E3.available_assets(tmp_path)
    assert assets["skeleton_sets"] == ["pC1"]
    (tmp_path / E3.OUT_BIN.name).write_bytes(b"")
    (tmp_path / E3.OUT_JSON.name).write_text("{}")
    assert E3.available_assets(tmp_path)["neurons_3d"] is True


# ---------------------------------------------------------------------------------------
# the real file
# ---------------------------------------------------------------------------------------
def _real():
    from flysim.graph.constants import CACHE_NPZ
    if not (CACHE_NPZ.exists() and E3.OUT_BIN.exists() and E3.OUT_JSON.exists()):
        pytest.skip("graph cache or neurons-3d files missing")
    return E3.read_arrays()


def test_real_point_cloud_matches_the_graph():
    from flysim.graph import Graph

    a = _real()
    doc, pos, src, sets_arr = a["doc"], a["pos"], a["src"], a["set"]
    g = Graph.load()
    assert doc["n"] == g.n == len(pos) == 165_122
    assert sum(doc["src_counts"].values()) == doc["n"]
    assert doc["src_counts"]["none"] == int(np.isnan(pos).any(axis=1).sum())
    # source priority holds against the graph's own arrays
    soma = src == E3.SRC_SOMA
    assert np.array_equal(soma, g.has_soma_xyz)
    assert np.allclose(pos[soma], g.soma_xyz[soma])
    assert doc["graph_cache_md5"] and len(doc["graph_cache_md5"]) == 32
    # contract v1.3: index -> bodyId, the same mapping Graph.body() gives
    assert a["body"].dtype == np.uint32
    assert np.array_equal(a["body"].astype(np.int64), g.body_ids)
    for i in (0, 1234, g.n - 1):
        assert int(a["body"][i]) == g.body(i)
    assert int(a["body"].max()) <= E3.UINT32_MAX


def test_real_order_is_the_run_json_neuron_index():
    """The contract's one hard ordering rule: index i here is index i in run.json."""
    from flysim.graph import Graph
    from flysim.probe.sets import build_probe_sets, load_roi

    a = _real()
    g = Graph.load()
    sets = build_probe_sets(g, load_roi(g))
    # run.json writes neurons.region = sets.region_of in graph order; the point cloud's
    # `set` array is the same vector, so both files index neurons identically.
    assert a["doc"]["sets"] == list(sets.names)
    assert np.array_equal(a["set"], np.asarray(sets.region_of, dtype=np.uint8))
    # spot-check a neuron by bodyId
    idx = int(g.idx(10001))
    assert np.allclose(a["pos"][idx], g.soma_xyz[idx])


def test_real_order_matches_an_actual_run_json():
    """Stronger than the previous test when a run exists: compare against a written run.json."""
    import glob

    a = _real()
    runs = sorted(glob.glob("runs/**/run.json", recursive=True))
    if not runs:
        pytest.skip("no run.json on disk (runs/ is gitignored)")
    d = json.loads(open(runs[0]).read())
    if d["meta"]["n_neurons"] != a["doc"]["n"]:
        pytest.skip(f"{runs[0]} was written for a different neuron count")
    assert d["regions"] == a["doc"]["sets"]
    assert np.array_equal(np.array(d["neurons"]["region"], dtype=np.uint8), a["set"])
    soma = a["src"] == E3.SRC_SOMA
    assert np.allclose(np.array(d["neurons"]["x"])[soma], np.round(a["pos"][soma, 0]))
    assert np.allclose(np.array(d["neurons"]["y"])[soma], np.round(a["pos"][soma, 1]))


def test_contract_example_offsets_hold_for_the_real_file():
    """The offsets quoted in docs/m6-3d-contract.md are the ones we write."""
    a = _real()
    text = CONTRACT.read_text(encoding="utf-8")
    quoted = {m.group(1): int(m.group(2))
              for m in re.finditer(r'"(pos|set|src)": \{"offset": (\d+)', text)}
    assert quoted, "could not read the arrays block from the contract"
    for name, offset in quoted.items():
        assert a["doc"]["arrays"][name]["offset"] == offset, name
