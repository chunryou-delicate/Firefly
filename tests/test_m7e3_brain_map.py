"""M7e3b tests: the flat brain map (docs/m7e3-brief.md §3, contract §3.1).

The projection and simplification rules are tested on small hand-made inputs anywhere; the map
itself needs the graph cache and the M6 files in data/cache and skips without them.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from flysim.apps import m7e3_brain_map as M

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------------------------
# projection and simplification (pure)
# ---------------------------------------------------------------------------------------------
def _sets(xl, xr, z_brain, z_vnc):
    pos = np.array([[xl, 0, z_brain], [xr, 0, z_brain], [50, 0, z_vnc], [60, 0, z_vnc]], np.float32)
    sets = {k: np.zeros(0, np.int64) for k in M.SET_NAMES}
    sets.update(LC4_L=np.array([0]), LC4_R=np.array([1]), TTMn_L=np.array([2]), TTMn_R=np.array([3]))
    return pos, sets


def test_orientation_is_read_from_the_data():
    o = M.orientation(*_sets(xl=900, xr=100, z_brain=10, z_vnc=500))
    assert o["left_is_high_x"] and o["anterior_is_low_z"]
    o = M.orientation(*_sets(xl=100, xr=900, z_brain=500, z_vnc=10))      # mirrored data
    assert not o["left_is_high_x"] and not o["anterior_is_low_z"]
    with pytest.raises(ValueError):
        M.orientation(*_sets(xl=100, xr=100, z_brain=10, z_vnc=500))     # undecidable
    pos, sets = _sets(900, 100, 10, 500)
    sets["LC4_L"] = np.zeros(0, np.int64)
    with pytest.raises(LookupError):
        M.orientation(pos, sets)


def test_projector_puts_fly_left_on_screen_left_and_anterior_up():
    o = dict(left_is_high_x=True, anterior_is_low_z=True)
    p = M.Projector((0.0, 1000.0), (0.0, 2000.0), o)
    assert p.h == M.GRID and p.w == 2048                         # longer side spans 0..4095, aspect kept
    sx, sy = p([1000.0, 0.0], [0.0, 2000.0])
    assert sx[0] == 0.0 and sx[1] == pytest.approx(2047.5)       # high x (fly left) -> screen left
    assert sy[0] == 0.0 and sy[1] == pytest.approx(4095.0)       # low z (anterior) -> top
    p2 = M.Projector((0.0, 1000.0), (0.0, 2000.0), dict(left_is_high_x=False, anterior_is_low_z=False))
    sx, sy = p2([1000.0], [0.0])
    assert sx[0] == pytest.approx(2047.5) and sy[0] == pytest.approx(4095.0)
    q = M.Projector.quantise(np.array([-3.0, 0.4, 4095.6, 5000.0]))
    assert q.tolist() == [0, 0, 4095, 4095]


def test_polylines_split_where_segments_do_not_join():
    seg = np.array([[0, 0, 0, 1, 0, 0], [1, 0, 0, 2, 0, 0],          # one chain
                    [1, 0, 0, 1, 0, 1],                               # a branch from x=1
                    [2, 0, 0, 3, 0, 0]], np.float32)                  # does not start at (1,0,1)
    pl = M.polylines(seg)
    assert [len(p) for p in pl] == [3, 2, 2]
    assert pl[1][0].tolist() == [1, 0, 0] and pl[2][0].tolist() == [2, 0, 0]
    assert M.polylines(np.zeros((0, 6), np.float32)) == []


def test_project_neuron_simplifies_within_tolerance_and_drops_points():
    p = M.Projector((0.0, 4095.0), (0.0, 4095.0), dict(left_is_high_x=False, anterior_is_low_z=True))
    z = np.linspace(0, 1000, 101)
    x = 500 + 1.0 * np.sin(z / 37.0)                                   # wiggle well inside tol 4
    pts = np.column_stack([x, np.zeros_like(x), z])
    seg = np.hstack([pts[:-1], pts[1:]]).astype(np.float32)
    flat, dev, n_in = M.project_neuron(seg, p, tol=4.0)
    assert n_in == 100 and len(flat) == 4 and dev <= 4.0              # one straight segment
    tiny = np.array([[10, 0, 10, 10.2, 0, 10.1]], np.float32)         # rounds to one point
    assert M.project_neuron(tiny, p)[0] == []
    twice = np.array([[0, 0, 0, 100, 0, 0], [5, 0, 5, 6, 0, 6], [100, 0, 0, 0, 0, 0]], np.float32)
    assert len(M.project_neuron(twice, p, tol=0.5)[0]) == 8           # reversed repeat dropped


# ---------------------------------------------------------------------------------------------
# the map (graph cache + data/cache)
# ---------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def built():
    from flysim.graph.constants import CACHE_NPZ
    need = [M.CACHE / "neurons-3d.json", M.CACHE / "neurons-3d.bin"] + \
           [M.CACHE / f"skel-{b}.{e}" for b in M.BUNDLES for e in ("json", "bin")]
    if not CACHE_NPZ.exists() or not all(p.exists() for p in need):
        pytest.skip("graph cache and the M6 point cloud / escape skeleton bundles are required")
    sets = M.escape_sets()
    return sets, M.build(sets)


def test_contract_shape(built):
    _, d = built
    assert d["version"] == 1
    assert {k: d["view"][k] for k in ("plane", "up", "screen_left")} == \
        {"plane": "x-z", "up": "anterior", "screen_left": "fly left"}
    assert d["view"]["how_left_was_decided"].startswith("mean x of LC4_L")
    w, h = d["box"]["w"], d["box"]["h"]
    assert max(w, h) == M.GRID
    assert list(d["sets"]) == list(M.SET_NAMES)
    assert d["dots"]["n"] == 30000 and d["dots"]["seed"] == 0 and len(d["dots"]["xy"]) == 60000

    def inside(flat):
        a = np.asarray(flat).reshape(-1, 2)
        return a.dtype.kind == "i" and (a >= 0).all() and (a[:, 0] < w).all() and (a[:, 1] < h).all()

    assert inside(d["dots"]["xy"])
    for k, s in d["sets"].items():
        assert len(s["soma"]) == 2 * s["n_neurons"] and len(s["lines"]) == s["n_neurons"], k
        assert inside(s["soma"]) and inside(s["centroid"]), k
        for ln in s["lines"]:
            assert len(ln) % 4 == 0 and len(ln) > 0 and inside(ln), k
    for rel, digest in d["sources"].items():
        assert hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == digest


def test_set_sizes_match_the_point_cloud_profile_and_the_bundles(built):
    sets, d = built
    expected = {"LC4_L": 71, "LC4_R": 55, "LPLC2_L": 94, "LPLC2_R": 91,
                "DNp01_L": 1, "DNp01_R": 1, "TTMn_L": 1, "TTMn_R": 1}         # M7a / escape profile
    assert {k: v["n_neurons"] for k, v in d["sets"].items()} == expected
    pos, src, meta = M.load_points()
    for b in M.BUNDLES:
        bm, _ = M.load_bundle(b)
        both = np.concatenate([sets[f"{b}_{s}"] for s in M.SIDES])
        assert bm["n_neurons"] == len(both) == d["sets"][f"{b}_L"]["n_neurons"] + d["sets"][f"{b}_R"]["n_neurons"]
        assert sorted(n["idx"] for n in bm["neurons"]) == sorted(both.tolist())
        assert (src[both] != 2).all()                                    # every one has a position
    assert meta["n"] == 165122


def test_set_names_and_members_are_the_run_json_regions(built):
    sets, _ = built
    runs = sorted((ROOT / "runs").glob("m7c*/*-brain-seed10/run.json"))
    if not runs:
        pytest.skip("no M7c / M7c2 run.json bundle on disk (runs/ is not committed)")
    r = json.loads(runs[0].read_text())
    assert [n for n in r["regions"] if n != "rest"] == list(M.SET_NAMES)
    region = np.asarray(r["neurons"]["region"])
    for i, k in enumerate(r["regions"]):
        if k != "rest":
            assert np.array_equal(np.flatnonzero(region == i), sets[k]), k


def test_fly_left_is_screen_left_and_brain_is_up(built):
    _, d = built
    S = d["sets"]
    for b in M.BUNDLES:
        assert S[f"{b}_L"]["centroid"][0] < S[f"{b}_R"]["centroid"][0], b
    assert max(S[k]["centroid"][1] for k in ("LC4_L", "LC4_R", "DNp01_L", "DNp01_R")) < \
        min(S[k]["centroid"][1] for k in ("TTMn_L", "TTMn_R"))
    assert "high x" in d["view"]["how_left_was_decided"] or "low x" in d["view"]["how_left_was_decided"]


def test_the_map_is_deterministic(built):
    sets, d = built
    d2 = M.build(sets)
    d.pop("generated"), d2.pop("generated")
    assert M.dumps(d) == M.dumps(d2)


def test_an_empty_set_raises(built):
    sets, _ = built
    broken = dict(sets, DNp01_R=np.zeros(0, np.int64))
    with pytest.raises(LookupError):
        M.build(broken)
    with pytest.raises(LookupError):
        M.build({k: v for k, v in sets.items() if k != "TTMn_L"})


def test_preview_draws_the_circuit_brighter_than_the_dots(built):
    _, d = built
    img = M.preview(d)
    assert img.shape == (-(-d["box"]["h"] // M.PREVIEW_SCALE), -(-d["box"]["w"] // M.PREVIEW_SCALE))
    assert set(np.unique(img)) <= {M.PREVIEW_BG, M.PREVIEW_DOT, *M.PREVIEW_INK.values()}
    assert (img == M.PREVIEW_INK["DNp01"]).sum() > 0 and (img == M.PREVIEW_DOT).sum() > 1000
