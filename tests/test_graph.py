"""M1 acceptance tests (m1-brief.md / CLAUDE.md §6). Requires data/cache; builds it if missing."""
import numpy as np
import pytest

from flysim.graph import Graph, NoPathError, NT_CODES, NT_SIGN, SIDE_CODES
from flysim.graph.build import EXPECTED_EDGES, EXPECTED_NEURONS

# Identifiers below are verified against the data:
#   DNp01 — docs/annotations-observed.md / m1-brief.md (bodyIds 10001 _R, 10010 _L)
#   superclass 'descending_neuron', class 'Kenyon_Cell', fruDsx 'fru_high' — annotations-observed.md
DNP01 = "DNp01"


@pytest.fixture(scope="session")
def g() -> Graph:
    return Graph.load(build_if_missing=True)


def test_retained_counts(g):
    assert g.n == EXPECTED_NEURONS
    assert g.m == EXPECTED_EDGES
    assert g.indptr[-1] == g.m and g.csc_indptr[-1] == g.m
    assert np.all(np.diff(g.indptr) >= 0) and np.all(np.diff(g.csc_indptr) >= 0)


def test_index_roundtrip(g):
    rng = np.random.default_rng(0)
    i = rng.integers(0, g.n, 1000)
    b = g.body(i)
    assert np.array_equal(g.idx(b), i)
    assert g.idx(int(b[0])) == int(i[0])
    assert np.all(np.diff(g.body_ids) > 0)
    with pytest.raises(KeyError):
        g.idx(-1)


def test_csr_csc_consistent(g):
    # same multiset of signed weights, same total contacts
    assert np.abs(g.weight).astype(np.int64).sum() == np.abs(g.csc_weight).astype(np.int64).sum()
    assert g.weight.astype(np.int64).sum() == g.csc_weight.astype(np.int64).sum()
    # spot check: every successor edge appears as a predecessor edge
    rng = np.random.default_rng(1)
    for i in rng.integers(0, g.n, 20):
        post, w = g.successors(int(i))
        for j, wj in zip(post[:5], w[:5]):
            pre, wp = g.predecessors(int(j))
            k = np.searchsorted(pre, i)
            assert pre[k] == i and wp[k] == wj


def test_edge_sign_follows_presynaptic_nt(g):
    rows = np.repeat(np.arange(g.n), g.out_degree())
    expect = np.array([NT_SIGN[c] for c in range(len(NT_CODES))], dtype=np.float32)[g.nt[rows]]
    assert np.array_equal(np.sign(g.weight), expect)
    assert np.all(np.abs(g.weight) >= 1)  # raw contact counts, no normalisation
    assert np.all(np.abs(g.weight) == np.round(np.abs(g.weight)))
    gaba = NT_CODES["gaba"]
    assert (g.nt == gaba).sum() > 0
    assert np.all(g.weight[g.nt[rows] == gaba] < 0)


def test_side_and_soma_arrays(g):
    assert set(np.unique(g.side)) <= set(SIDE_CODES.values()) | {-1}
    assert g.soma_xyz.shape == (g.n, 3)
    assert np.isnan(g.soma_xyz[~g.has_soma_xyz]).all()
    assert not np.isnan(g.soma_xyz[g.has_soma_xyz]).any()


def test_by_type_hit_and_miss(g):
    idx = g.query.by_type(DNP01)
    assert idx.size >= 2
    assert set(g.query.describe(idx)["somaSide"]) == {"L", "R"}
    left = g.query.by_type(DNP01, side="L")
    right = g.query.by_type(DNP01, side="R")
    assert left.size >= 1 and right.size >= 1 and not set(left) & set(right)
    with pytest.raises(LookupError):
        g.query.by_type("없는이름")
    with pytest.raises(LookupError):
        g.query.by_type(DNP01, side="M")
    with pytest.raises(ValueError):
        g.query.by_type(DNP01, side="X")


def test_other_queries(g):
    assert g.query.by_superclass("descending_neuron").size > 1000
    assert g.query.by_class("Kenyon_Cell").size > 1000
    assert g.query.by_fru_dsx("fru_high").size > 1000
    for fn in (g.query.by_superclass, g.query.by_class, g.query.by_fru_dsx, g.query.by_instance):
        with pytest.raises(LookupError):
            fn("nonexistent_value_xyz")
    hits = g.query.search_type(r"^DNp0")
    assert any(t == DNP01 for t, _ in hits)
    with pytest.raises(LookupError):
        g.query.search_type(r"^zzz_no_such_type_")


def test_shortest_path(g):
    a, b = (int(x) for x in g.body(g.query.by_type(DNP01)[:2]))
    p = g.shortest_path(a, b, max_hops=8)
    assert p[0] == a and p[-1] == b and 2 <= len(p) <= 9
    # every consecutive pair must be a real directed edge
    for u, v in zip(p, p[1:]):
        post, _ = g.successors(g.idx(u))
        assert g.idx(v) in post
    assert g.shortest_path(a, a) == [a]
    # unreachable within 0 hops
    with pytest.raises(NoPathError):
        g.shortest_path(a, b, max_hops=0)
    # a neuron with no outgoing edges cannot reach anything
    sinks = np.flatnonzero(g.out_degree() == 0)
    if sinks.size:
        with pytest.raises(NoPathError):
            g.shortest_path(int(g.body(sinks[0])), b, max_hops=3)


def test_shortest_path_deterministic(g):
    a, b = (int(x) for x in g.body(g.query.by_type(DNP01)[:2]))
    assert g.shortest_path(a, b) == g.shortest_path(a, b)
