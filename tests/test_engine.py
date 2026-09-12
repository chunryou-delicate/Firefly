"""M2 acceptance tests (docs/m2-brief.md). GPU tests are skipped without CUDA;
the real-graph tests need data/cache (built if missing, like tests/test_graph.py)."""
import numpy as np
import pytest
import torch

from flysim.engine import CSRGraph, EngineParams, LIFEngine, RecorderOverflow, propagate_reference
from flysim.graph import Graph

CUDA = torch.cuda.is_available()
needs_cuda = pytest.mark.skipif(not CUDA, reason="CUDA required")
DEVICES = ["cpu"] + (["cuda"] if CUDA else [])


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def real_graph() -> Graph:
    return Graph.load(build_if_missing=True)


@pytest.fixture(scope="session")
def synth_graph() -> CSRGraph:
    rng = np.random.default_rng(0)
    n, m = 5000, 200_000
    pre, post = rng.integers(0, n, m), rng.integers(0, n, m)
    w = rng.integers(1, 20, m) * np.where(rng.random(m) < 0.3, -1, 1)
    return CSRGraph.from_edges(n, pre, post, w)


@pytest.fixture(scope="module")
def three() -> CSRGraph:
    # A(0) -> B(1) excitatory +5 contacts, A(0) -> C(2) inhibitory -5 contacts
    return CSRGraph.from_edges(3, [0, 0], [1, 2], [5, -5])


def _stim100(n: int, seed: int = 0, amp: float = 30.0, device="cuda"):
    """Constant current into 100 random neurons for the first 100 steps."""
    idx = np.random.default_rng(seed).choice(n, 100, replace=False)
    stim = torch.zeros(n, device=device)
    stim[torch.as_tensor(idx, device=device)] = amp

    def fn(t, buf):
        if t == 0:
            buf.copy_(stim)
        elif t == 100:
            buf.zero_()
    return fn


def _spikes(engine, steps, fn, seed):
    engine.reset(seed)
    t, i = engine.run(steps, fn)
    return t, i


# ---------------------------------------------------------------------------
# params
# ---------------------------------------------------------------------------
def test_params_coefficients_and_validation():
    p1, p01 = EngineParams(dt=1.0), EngineParams(dt=0.1)
    assert p1.decay_m == pytest.approx(np.exp(-1 / 20)) and p01.decay_m == pytest.approx(np.exp(-0.1 / 20))
    assert p01.decay_m ** 10 == pytest.approx(p1.decay_m)          # exp(-dt/tau) composes across dt
    assert p1.ref_steps == 2 and p01.ref_steps == 20
    assert p1.noise_scale == 0.0 and EngineParams(noise_sigma=2.0, dt=0.25).noise_scale == pytest.approx(4.0)
    assert "decay_m" in p1.to_dict() and p1.to_dict()["g"] == p1.g
    for bad in (dict(dt=0), dict(tau_m=-1), dict(v_thresh=-70), dict(noise_sigma=-1), dict(v_floor=-60)):
        with pytest.raises(ValueError):
            EngineParams(**bad)


def test_default_g_inside_recorded_valid_range():
    """DEFAULT_G must be a VALID point of the committed sweep (data-provenance/m2-sweep.json)."""
    import json
    from pathlib import Path
    from flysim.engine import DEFAULT_G
    path = Path(__file__).resolve().parents[1] / "data-provenance" / "m2-sweep.json"
    if not path.exists():
        pytest.skip("sweep result not present")
    res = json.loads(path.read_text())
    valid = [r["g"] for r in res["rows"] if r["verdict"] == "VALID"]
    assert valid, "sweep recorded no valid g"
    assert any(abs(g - DEFAULT_G) / g < 1e-3 for g in valid)


# ---------------------------------------------------------------------------
# propagation kernel == torch index_add_ reference (bit-exact, int32)
# ---------------------------------------------------------------------------
@needs_cuda
@pytest.mark.parametrize("p_active", [0.0, 0.001, 0.05, 1.0])
def test_propagate_matches_reference_synthetic(synth_graph, p_active):
    from flysim.engine.lif import _propagate_kernel, PROP_BLOCK, PROP_NEURONS_PER_PROGRAM, PROP_WARPS
    import triton
    e = LIFEngine(synth_graph, device="cuda")
    torch.manual_seed(1)
    spk = (torch.rand(e.n, device="cuda") < p_active).to(torch.int8)
    e._s.copy_(spk)
    e.acc.zero_()
    _propagate_kernel[(triton.cdiv(e.n, PROP_NEURONS_PER_PROGRAM),)](
        e._s, e.indptr, e.indices, e.weight, e.acc, e.n,
        NB=PROP_NEURONS_PER_PROGRAM, BLOCK=PROP_BLOCK, num_warps=PROP_WARPS)
    ref = torch.zeros(e.n, dtype=torch.int32, device="cuda")
    propagate_reference(spk, e.indptr, e.indices, e.weight, ref)
    assert torch.equal(e.acc, ref)
    if p_active == 1.0:  # every edge counted once: column sums of the signed weights
        col = np.bincount(synth_graph.indices, weights=synth_graph.weight, minlength=e.n)
        assert np.array_equal(ref.cpu().numpy(), col.astype(np.int32))


@needs_cuda
def test_propagate_matches_reference_real_graph(real_graph):
    from flysim.engine.lif import _propagate_kernel, PROP_BLOCK, PROP_NEURONS_PER_PROGRAM, PROP_WARPS
    import triton
    e = LIFEngine(real_graph, device="cuda")
    torch.manual_seed(2)
    spk = (torch.rand(e.n, device="cuda") < 0.01).to(torch.int8)
    e._s.copy_(spk)
    _propagate_kernel[(triton.cdiv(e.n, PROP_NEURONS_PER_PROGRAM),)](
        e._s, e.indptr, e.indices, e.weight, e.acc, e.n,
        NB=PROP_NEURONS_PER_PROGRAM, BLOCK=PROP_BLOCK, num_warps=PROP_WARPS)
    ref = torch.zeros(e.n, dtype=torch.int32, device="cuda")
    propagate_reference(spk, e.indptr, e.indices, e.weight, ref)
    assert torch.equal(e.acc, ref)
    assert int(ref.abs().sum()) > 0


# ---------------------------------------------------------------------------
# signs on a 3-neuron graph
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("device", DEVICES)
def test_three_neuron_signs(three, device):
    e = LIFEngine(three, EngineParams(g=20.0), device=device)
    e.reset(0)
    i_ext = torch.zeros(3, device=device)
    i_ext[0] = 30.0
    v_c, v_b = [], []
    for _ in range(60):
        e.step(i_ext)
        v_b.append(float(e.v[1]))
        v_c.append(float(e.v[2]))
    t, idx = e.recorder.collect()
    a_t, b_t, c_t = (t[idx == k] for k in (0, 1, 2))
    assert len(a_t) >= 2, "driven neuron A must fire"
    assert len(b_t) >= 1 and b_t[0] > a_t[0], "excited B fires only after A"
    assert len(c_t) == 0, "inhibited C never fires"
    assert min(v_c) < e.params.v_rest and max(v_c) <= e.params.v_rest  # hyperpolarised only
    assert min(v_b) >= e.params.v_reset and max(v_b) < e.params.v_thresh
    assert float(e.i_syn[1]) > 0 > float(e.i_syn[2])
    assert np.all(np.diff(a_t) >= e.params.ref_steps + 1)  # refractory respected


# ---------------------------------------------------------------------------
# determinism (bit-identical spike lists)
# ---------------------------------------------------------------------------
@needs_cuda
def test_deterministic_bit_identical_real_graph(real_graph):
    # g = 1.0 is far above the sweep's valid range: massive activity -> heavy atomic traffic.
    e = LIFEngine(real_graph, EngineParams(g=1.0), device="cuda")
    fn = _stim100(e.n)
    t1, i1 = _spikes(e, 100, fn, 0)
    t2, i2 = _spikes(e, 100, fn, 0)
    assert len(t1) > 100_000, "test needs a high-activity regime to stress the atomics"
    assert np.array_equal(t1, t2) and np.array_equal(i1, i2)
    # fresh engine instance
    e2 = LIFEngine(real_graph, EngineParams(g=1.0), device="cuda")
    t3, i3 = _spikes(e2, 100, fn, 0)
    assert np.array_equal(t1, t3) and np.array_equal(i1, i3)
    # CUDA-graph replay == eager
    e3 = LIFEngine(real_graph, EngineParams(g=1.0), device="cuda", use_cuda_graph=True)
    t4, i4 = _spikes(e3, 100, fn, 0)
    assert np.array_equal(t1, t4) and np.array_equal(i1, i4)
    # spike list is sorted by (t, idx) and idx in range
    assert np.all(np.diff(t1.astype(np.int64) * e.n + i1) > 0)
    assert i1.min() >= 0 and i1.max() < e.n


@needs_cuda
def test_deterministic_with_noise_and_seed_dependence(real_graph):
    e = LIFEngine(real_graph, EngineParams(g=0.02, noise_sigma=5.0), device="cuda")
    fn = _stim100(e.n)
    a = _spikes(e, 200, fn, 11)
    b = _spikes(e, 200, fn, 11)
    c = _spikes(e, 200, fn, 12)
    assert len(a[0]) > 0
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert not (len(a[0]) == len(c[0]) and np.array_equal(a[0], c[0]) and np.array_equal(a[1], c[1]))


@pytest.mark.parametrize("device", DEVICES)
def test_deterministic_synthetic_both_backends(synth_graph, device):
    e = LIFEngine(synth_graph, EngineParams(g=0.5, noise_sigma=1.0), device=device)
    fn = _stim100(e.n, amp=40.0, device=device)
    a = _spikes(e, 150, fn, 3)
    b = _spikes(e, 150, fn, 3)
    assert len(a[0]) > 0 and np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


# ---------------------------------------------------------------------------
# numerical sanity
# ---------------------------------------------------------------------------
@needs_cuda
@pytest.mark.parametrize("g", [0.01, 1.0])
def test_no_nan_and_v_bounds_real_graph(real_graph, g):
    p = EngineParams(g=g)
    e = LIFEngine(real_graph, p, device="cuda")
    _spikes(e, 150, _stim100(e.n), 0)
    assert torch.isfinite(e.v).all() and torch.isfinite(e.i_syn).all()
    assert float(e.v.max()) < p.v_thresh                 # spiking neurons were reset
    assert int((e.ref < 0).sum()) == 0 and int((e.ref > p.ref_steps).sum()) == 0
    # with an explicit floor the lower bound holds too
    pf = EngineParams(g=g, v_floor=-80.0)
    ef = LIFEngine(real_graph, pf, device="cuda")
    _spikes(ef, 150, _stim100(ef.n), 0)
    assert float(ef.v.min()) >= pf.v_floor and float(ef.v.max()) < pf.v_thresh


def test_v_lower_bound_without_inhibition():
    # excitatory-only graph: v can never go below v_reset (no clamp needed)
    rng = np.random.default_rng(5)
    n, m = 300, 3000
    g = CSRGraph.from_edges(n, rng.integers(0, n, m), rng.integers(0, n, m), rng.integers(1, 10, m))
    dev = "cuda" if CUDA else "cpu"
    e = LIFEngine(g, EngineParams(g=0.3), device=dev)
    e.reset(0)
    i_ext = torch.zeros(n, device=dev)
    i_ext[:20] = 30.0
    vmin = np.inf
    for _ in range(100):
        e.step(i_ext)
        vmin = min(vmin, float(e.v.min()))
        assert float(e.v.max()) < e.params.v_thresh
    assert vmin >= e.params.v_reset


def test_dt_invariance_spike_times(three):
    """dt = 1 ms and dt = 0.1 ms give the same spike times (to discretisation) for the same params."""
    dev = "cuda" if CUDA else "cpu"
    times = {}
    for dt in (1.0, 0.1):
        e = LIFEngine(three, EngineParams(dt=dt, g=20.0), device=dev)
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
        assert np.all(np.abs(a[:m] - b[:m]) <= 1.0 + 1e-6), (a, b)


@needs_cuda
def test_torch_backend_matches_triton(three):
    outs, vs = [], []
    for backend in ("triton", "torch"):
        e = LIFEngine(three, EngineParams(g=20.0), device="cuda", backend=backend)
        e.reset(0)
        i_ext = torch.zeros(3, device="cuda")
        i_ext[0] = 30.0
        for _ in range(80):
            e.step(i_ext)
        outs.append(e.recorder.collect())
        vs.append(e.v.cpu().numpy())
    assert np.array_equal(outs[0][0], outs[1][0]) and np.array_equal(outs[0][1], outs[1][1])
    assert np.allclose(vs[0], vs[1], atol=1e-3)


# ---------------------------------------------------------------------------
# recorder and API conventions
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("device", DEVICES)
def test_recorder_overflow_raises(synth_graph, device):
    e = LIFEngine(synth_graph, EngineParams(g=0.0, t_ref=0.0), device=device, record_cap=64, chunk_steps=10)
    e.reset(0)
    i_ext = torch.zeros(e.n, device=device)
    i_ext[:50] = 1e4                     # 50 spikes/step * 10 steps > cap 64
    with pytest.raises(RecorderOverflow):
        for _ in range(10):
            e.step(i_ext)
    e.reset(0)
    i_ext.zero_()
    i_ext[:5] = 1e4                      # 5/step * 10 = 50 < 64: fine
    for _ in range(20):
        e.step(i_ext)
    t, idx = e.recorder.collect()
    assert len(t) == 100 and set(idx.tolist()) == set(range(5))
    assert np.array_equal(np.unique(t), np.arange(20))


@pytest.mark.parametrize("device", DEVICES)
def test_run_input_conventions_and_no_record(synth_graph, device):
    e = LIFEngine(synth_graph, EngineParams(g=0.0, t_ref=0.0), device=device)
    e.reset(0)
    # returning a tensor from the callback is copied into the buffer
    big = torch.full((e.n,), 1e3, device=device)
    out = e.run(3, lambda t, buf: big)
    assert out is not None and len(out[0]) == 3 * e.n
    # record=False returns None and records nothing
    e.reset(0)
    assert e.run(3, lambda t, buf: big, record=False) is None
    assert e.recorder.collect()[0].size == 0
    # step(None) means zero input
    e.reset(0)
    e.step(big)
    assert bool(e.spikes.all())
    for _ in range(2):
        s = e.step(None)
    assert not bool(s.any()) and s.dtype == torch.bool and s.shape == (e.n,)
    assert e.t == 3
