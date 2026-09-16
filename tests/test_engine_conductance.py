"""M2b tests: (1) the default synapse="current" path reproduces M2 bit-identically;
(2) the conductance model's sign, bounds, determinism, dt-invariance and Triton==torch
properties (docs/m2b-brief.md)."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from flysim.engine import CSRGraph, EngineParams, LIFEngine, propagate_reference_split
from flysim.graph import Graph

import regression_cases as rc

CUDA = torch.cuda.is_available()
needs_cuda = pytest.mark.skipif(not CUDA, reason="CUDA required")
DEVICES = ["cpu"] + (["cuda"] if CUDA else [])
COND = dict(synapse="conductance")


@pytest.fixture(scope="session")
def real_graph() -> Graph:
    return Graph.load(build_if_missing=True)


@pytest.fixture(scope="session")
def synth_graph() -> CSRGraph:
    return rc.synth_graph()


@pytest.fixture(scope="module")
def three() -> CSRGraph:
    # A(0) -> B(1) excitatory +5 contacts, A(0) -> C(2) inhibitory -5 contacts
    return CSRGraph.from_edges(3, [0, 0], [1, 2], [5, -5])


# ---------------------------------------------------------------------------
# regression: synapse="current" (default) == M2 engine, bit for bit
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def reference():
    if not rc.NPZ.exists():
        pytest.skip("reference file missing")
    with np.load(rc.NPZ) as z:
        return {k: z[k] for k in z.files}


@pytest.mark.parametrize("name", rc.CURRENT_CASES)
def test_current_model_regression_bit_identical(name, reference, real_graph, synth_graph):
    """The conductance cases are replayed in tests/test_engine_adapt.py (M2c)."""
    kind, pkw, ekw, *_ = rc.CASES[name]
    if ekw.get("device", "cuda") == "cuda" and not CUDA:
        pytest.skip("CUDA required")
    assert EngineParams(**pkw).synapse == "current"
    t, idx, v = rc.run_case(name, real_graph, synth_graph)
    assert np.array_equal(t, reference[f"{name}__t"]), f"{name}: spike times differ from M2 reference"
    assert np.array_equal(idx, reference[f"{name}__idx"]), f"{name}: spike neurons differ from M2 reference"
    assert np.array_equal(v, reference[f"{name}__v"]), f"{name}: final v differs from M2 reference"


def test_reference_metadata_records_m2_commit(reference):
    meta = json.loads(str(reference["meta_json"]))
    assert meta["commit"].startswith("0b2c19b") or meta["commit"].startswith("8bae9d3") or len(meta["commit"]) == 40
    assert set(meta["cases"]) == set(rc.CASES)


def test_params_conductance_fields_and_defaults():
    p = EngineParams()
    assert p.synapse == "current" and p.g == pytest.approx(0.886)
    c = EngineParams(synapse="conductance", g=1e-3)
    d = c.to_dict()
    for k in ("synapse", "E_exc", "E_inh", "tau_e", "tau_i", "decay_e", "decay_i"):
        assert k in d
    assert c.decay_e == pytest.approx(np.exp(-1 / 5)) and c.decay_i == pytest.approx(np.exp(-1 / 10))
    assert EngineParams(synapse="conductance", g=1e-3, dt=0.1).decay_i ** 10 == pytest.approx(c.decay_i)
    for bad in (dict(synapse="foo", g=1.0), dict(synapse="conductance", g=1e-3, E_inh=-60.0),
                dict(synapse="conductance", g=1e-3, E_exc=-55.0), dict(synapse="conductance", g=1e-3, tau_i=0)):
        with pytest.raises(ValueError):
            EngineParams(**bad)


# ---------------------------------------------------------------------------
# split propagation kernel == torch reference (bit-exact)
# ---------------------------------------------------------------------------
@needs_cuda
@pytest.mark.parametrize("p_active", [0.0, 0.001, 0.05, 1.0])
def test_split_propagate_matches_reference(synth_graph, p_active):
    from flysim.engine.lif import _propagate_split_kernel, PROP_BLOCK, PROP_NEURONS_PER_PROGRAM, PROP_WARPS
    import triton
    e = LIFEngine(synth_graph, EngineParams(g=1e-3, **COND), device="cuda")
    torch.manual_seed(1)
    spk = (torch.rand(e.n, device="cuda") < p_active).to(torch.int8)
    e._s.copy_(spk)
    _propagate_split_kernel[(triton.cdiv(e.n, PROP_NEURONS_PER_PROGRAM),)](
        e._s, e.indptr, e.indices, e.weight, e.acc_e, e.acc_i, e.n,
        NB=PROP_NEURONS_PER_PROGRAM, BLOCK=PROP_BLOCK, num_warps=PROP_WARPS)
    re_, ri_ = (torch.zeros(e.n, dtype=torch.int32, device="cuda") for _ in range(2))
    propagate_reference_split(spk, e.indptr, e.indices, e.weight, re_, ri_)
    assert torch.equal(e.acc_e, re_) and torch.equal(e.acc_i, ri_)
    assert int((e.acc_e < 0).sum()) == 0 and int((e.acc_i < 0).sum()) == 0
    if p_active == 1.0:
        w = synth_graph.weight
        ce = np.bincount(synth_graph.indices, weights=np.clip(w, 0, None), minlength=e.n)
        ci = np.bincount(synth_graph.indices, weights=np.clip(-w, 0, None), minlength=e.n)
        assert np.array_equal(re_.cpu().numpy(), ce.astype(np.int32))
        assert np.array_equal(ri_.cpu().numpy(), ci.astype(np.int32))


# ---------------------------------------------------------------------------
# conductance model: signs, reversal-potential bounds
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("device", DEVICES)
def test_conductance_three_neuron_signs(three, device):
    p = EngineParams(g=0.02, **COND)      # 5 contacts * 0.02 = 0.1/ms = 2x leak per spike
    e = LIFEngine(three, p, device=device)
    e.reset(0)
    i_ext = torch.zeros(3, device=device)
    i_ext[0] = 30.0
    v_b, v_c = [], []
    for _ in range(80):
        e.step(i_ext)
        v_b.append(float(e.v[1]))
        v_c.append(float(e.v[2]))
    t, idx = e.recorder.collect()
    a_t, b_t, c_t = (t[idx == k] for k in (0, 1, 2))
    assert len(a_t) >= 2, "driven neuron A must fire"
    assert len(b_t) >= 1 and b_t[0] > a_t[0], "excited B fires only after A"
    assert len(c_t) == 0, "inhibited C never fires"
    assert min(v_c) < p.v_rest, "C is hyperpolarised"
    assert min(v_c) >= p.E_inh, "but never below the inhibitory reversal potential"
    assert max(v_c) <= p.v_rest
    assert min(v_b) >= p.v_reset and max(v_b) < p.v_thresh
    assert float(e.g_e[1]) > 0 and float(e.g_i[1]) == 0 and float(e.g_e[2]) == 0 and float(e.g_i[2]) > 0
    assert np.all(np.diff(a_t) >= p.ref_steps + 1)


@needs_cuda
@pytest.mark.parametrize("g", [1e-4, 1e-2])
def test_conductance_v_bounds_and_finite_real_graph(real_graph, g):
    p = EngineParams(g=g, **COND)
    e = LIFEngine(real_graph, p, device="cuda")
    e.reset(0)
    fn = rc.stim100(e.n)
    vmin, vmax = np.inf, -np.inf
    for k in range(200):
        fn(k, e.i_ext)
        e.step(e.i_ext)
        assert torch.isfinite(e.v).all() and torch.isfinite(e.g_e).all() and torch.isfinite(e.g_i).all()
        vmin = min(vmin, float(e.v.min()))
        vmax = max(vmax, float(e.v.max()))
    t, idx = e.recorder.collect()
    assert len(t) > 0
    assert vmin >= p.E_inh, f"v went below E_inh: {vmin}"     # no input is ever negative here
    assert vmax < p.v_thresh
    assert float(e.g_e.min()) >= 0 and float(e.g_i.min()) >= 0
    assert int((e.ref < 0).sum()) == 0 and int((e.ref > p.ref_steps).sum()) == 0


# ---------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------
@needs_cuda
def test_conductance_deterministic_bit_identical_real_graph(real_graph):
    p = EngineParams(g=3e-3, **COND)    # high-activity regime (many atomics, tl.exp per neuron)
    fn = rc.stim100(real_graph.n)
    e = LIFEngine(real_graph, p, device="cuda")
    e.reset(0); a = e.run(150, fn)
    e.reset(0); b = e.run(150, fn)
    assert len(a[0]) > 50_000, "test needs a high-activity regime"
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    e2 = LIFEngine(real_graph, p, device="cuda")
    e2.reset(0); c = e2.run(150, fn)
    assert np.array_equal(a[0], c[0]) and np.array_equal(a[1], c[1])
    e3 = LIFEngine(real_graph, p, device="cuda", use_cuda_graph=True)
    e3.reset(0); d = e3.run(150, fn)
    assert np.array_equal(a[0], d[0]) and np.array_equal(a[1], d[1])
    assert torch.equal(e.v, e3.v)


@pytest.mark.parametrize("device", DEVICES)
def test_conductance_deterministic_with_noise_both_backends(synth_graph, device):
    p = EngineParams(g=2e-3, noise_sigma=3.0, **COND)
    e = LIFEngine(synth_graph, p, device=device)
    fn = rc.stim100(e.n, amp=40.0, device=device)
    e.reset(3); a = e.run(150, fn)
    e.reset(3); b = e.run(150, fn)
    e.reset(4); c = e.run(150, fn)
    assert len(a[0]) > 0 and np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert not (len(a[0]) == len(c[0]) and np.array_equal(a[0], c[0]) and np.array_equal(a[1], c[1]))


# ---------------------------------------------------------------------------
# dt invariance and backend agreement
# ---------------------------------------------------------------------------
def test_conductance_dt_invariance_spike_times(three):
    dev = "cuda" if CUDA else "cpu"
    times = {}
    for dt in (1.0, 0.1):
        e = LIFEngine(three, EngineParams(dt=dt, g=0.02, **COND), device=dev)
        e.reset(0)
        i_ext = torch.zeros(3, device=dev)
        i_ext[0] = 30.0
        for _ in range(int(200 / dt)):
            e.step(i_ext)
        t, idx = e.recorder.collect()
        times[dt] = {k: t[idx == k] * dt for k in (0, 1)}
    for k in (0, 1):
        a, b = times[1.0][k], times[0.1][k]
        assert len(a) > 3 and abs(len(a) - len(b)) <= 1
        m = min(len(a), len(b))
        assert np.all(np.abs(a[:m] - b[:m]) <= 1.0 + 1e-6), (k, a, b)


@needs_cuda
def test_conductance_torch_backend_matches_triton(three):
    outs, vs = [], []
    for backend in ("triton", "torch"):
        e = LIFEngine(three, EngineParams(g=0.02, **COND), device="cuda", backend=backend)
        e.reset(0)
        i_ext = torch.zeros(3, device="cuda")
        i_ext[0] = 30.0
        for _ in range(80):
            e.step(i_ext)
        outs.append(e.recorder.collect())
        vs.append(e.v.cpu().numpy())
    assert np.array_equal(outs[0][0], outs[1][0]) and np.array_equal(outs[0][1], outs[1][1])
    assert np.allclose(vs[0], vs[1], atol=1e-3)


@needs_cuda
def test_conductance_torch_backend_matches_triton_real_graph_short(real_graph):
    """Same spike list from both backends on the full graph for a short, moderate run."""
    p = EngineParams(g=1e-3, **COND)
    fn = rc.stim100(real_graph.n)
    outs = []
    for backend in ("triton", "torch"):
        e = LIFEngine(real_graph, p, device="cuda", backend=backend)
        e.reset(0)
        outs.append(e.run(30, fn))
    assert len(outs[0][0]) > 0
    assert np.array_equal(outs[0][0], outs[1][0]) and np.array_equal(outs[0][1], outs[1][1])


def test_default_g_conductance_matches_recorded_sweep_rule():
    """DEFAULT_G_CONDUCTANCE == geometric mean of the RESPONSIVE interval in data-provenance/m2b-sweep.json."""
    from flysim.engine import DEFAULT_G_CONDUCTANCE
    path = Path(__file__).resolve().parents[1] / "data-provenance" / "m2b-sweep.json"
    if not path.exists():
        pytest.skip("sweep result not present")
    res = json.loads(path.read_text())
    assert res["responsive_range"] is not None and res["responsive_contiguous"]
    lo, hi = res["responsive_range"]
    assert DEFAULT_G_CONDUCTANCE == pytest.approx(np.sqrt(lo * hi), rel=1e-3)
    assert EngineParams(synapse="conductance").g == pytest.approx(DEFAULT_G_CONDUCTANCE)
