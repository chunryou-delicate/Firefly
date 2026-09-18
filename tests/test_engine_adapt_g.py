"""M2d tests: adaptation as a conductance (docs/m2d-brief.md).

The point of M2d is that the adaptation is bounded by a reversal potential, which
the M2c current was not (it reached -834 mV). So the central test here is the
voltage bound, on the real graph, in the regime where M2c failed.
"""
import numpy as np
import pytest
import torch

from flysim.engine import CSRGraph, EngineParams, LIFEngine
from flysim.engine.params import DEFAULT_ADAPT_G_B, DEFAULT_G_WITH_ADAPT
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
    """A(0) -> B(1) excitatory +5 contacts, A(0) -> C(2) inhibitory -5 contacts."""
    return CSRGraph.from_edges(3, [0, 0], [1, 2], [5, -5])


@pytest.fixture(scope="module")
def one() -> CSRGraph:
    return CSRGraph.from_edges(1, [], [], [])


# ---------------------------------------------------------------------------
# parameters: conductance-only, mutually exclusive with M2c
# ---------------------------------------------------------------------------
def test_defaults_keep_adaptation_off():
    for kw in ({}, COND):
        p = EngineParams(**kw)
        assert p.adapt_g_b == 0.0 and p.adapt_g is False
        assert p.adapt_tau_a == 100.0 and p.E_adapt == -75.0
    assert DEFAULT_ADAPT_G_B is None or DEFAULT_ADAPT_G_B > 0


def test_defaults_match_the_recorded_sweep_rule():
    """DEFAULT_ADAPT_G_B = smallest b_g with a usable state; DEFAULT_G_WITH_ADAPT = the
    geometric mean of the g values usable at that b_g (rule fixed before the sweep)."""
    import json
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "data-provenance" / "m2d-noise-sweep.json"
    if not path.exists():
        pytest.skip("sweep result not present")
    res = json.loads(path.read_text())
    if not res["usable_cells"]:
        assert DEFAULT_ADAPT_G_B is None and DEFAULT_G_WITH_ADAPT is None
        return
    assert DEFAULT_ADAPT_G_B == pytest.approx(min(b for b, _ in res["usable_cells"]))
    gs = [g for b, g in res["usable_cells"] if b == pytest.approx(DEFAULT_ADAPT_G_B)]
    assert DEFAULT_G_WITH_ADAPT == pytest.approx(np.sqrt(min(gs) * max(gs)))
    assert res["default_adapt_g_b"] == pytest.approx(DEFAULT_ADAPT_G_B)
    assert res["protocol"]["usable_state"]["active_frac_max"] == 0.30   # the new condition
    assert res["protocol"]["adapt_tau_a"] if "adapt_tau_a" in res["protocol"] else True


def test_adapt_g_requires_conductance_synapses():
    """A current model has no reversal potentials, so g_a is meaningless there."""
    with pytest.raises(ValueError, match="requires synapse='conductance'"):
        EngineParams(adapt_g_b=0.01)
    with pytest.raises(ValueError, match="requires synapse='conductance'"):
        EngineParams(synapse="current", adapt_g_b=0.01)
    EngineParams(adapt_g_b=0.0)                      # off is fine on either model
    EngineParams(synapse="conductance", adapt_g_b=0.01)


def test_the_two_adaptation_models_are_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        EngineParams(synapse="conductance", adapt_g_b=0.01, adapt_b=1.0)
    EngineParams(synapse="conductance", adapt_b=1.0)     # M2c alone still works (history)


def test_parameter_validation_and_derived_values():
    with pytest.raises(ValueError, match="adapt_g_b must be >= 0"):
        EngineParams(**COND, adapt_g_b=-1.0)
    with pytest.raises(ValueError, match="adapt_tau_a must be > 0"):
        EngineParams(**COND, adapt_g_b=0.01, adapt_tau_a=0.0)
    with pytest.raises(ValueError, match="E_adapt <= v_reset"):
        EngineParams(**COND, adapt_g_b=0.01, E_adapt=-10.0)
    p = EngineParams(**COND, adapt_g_b=0.01)
    assert p.decay_a == pytest.approx(np.exp(-1 / 100))
    assert p.avg_a == pytest.approx(100 * (1 - np.exp(-0.01)))
    assert p.v_lower_bound == -75.0
    assert EngineParams(**COND, adapt_g_b=0.01, dt=0.1).decay_a ** 10 == pytest.approx(p.decay_a)
    for k in ("adapt_g_b", "adapt_tau_a", "E_adapt", "decay_a", "avg_a", "adapt_g", "v_lower_bound"):
        assert k in p.to_dict()


# ---------------------------------------------------------------------------
# adapt_g_b = 0 is the pre-M2d engine
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", rc.CONDUCTANCE_CASES)
def test_conductance_regression_bit_identical(name, real_graph, synth_graph):
    """The conductance references (commits 15b880d and 8d68698) are unchanged by M2d."""
    if not rc.NPZ.exists():
        pytest.skip("reference file missing")
    kind, pkw, ekw, *_ = rc.CASES[name]
    if ekw.get("device", "cuda") == "cuda" and not CUDA:
        pytest.skip("CUDA required")
    assert EngineParams(**pkw).adapt_g_b == 0.0
    with np.load(rc.NPZ) as z:
        ref = {k: z[k] for k in (f"{name}__t", f"{name}__idx", f"{name}__v")}
    t, idx, v = rc.run_case(name, real_graph, synth_graph)
    assert np.array_equal(t, ref[f"{name}__t"]), f"{name}: spike times differ"
    assert np.array_equal(idx, ref[f"{name}__idx"]), f"{name}: spike neurons differ"
    assert np.array_equal(v, ref[f"{name}__v"]), f"{name}: final v differs"


@pytest.mark.parametrize("device", DEVICES)
def test_adapt_g_branch_is_a_noop_at_zero(synth_graph, device):
    """b_g small enough to be invisible in float32 must reproduce the ADAPT_G = False
    path bit for bit, so the branch itself is a no-op and not merely the default."""
    out = []
    for b in (0.0, 1e-30):
        p = EngineParams(g=2e-3, noise_sigma=1.0, adapt_g_b=b, **COND)
        assert p.adapt_g is (b > 0)
        e = LIFEngine(synth_graph, p, device=device)
        e.reset(3)
        out.append((*e.run(120, rc.stim100(e.n, amp=40.0, device=device)),
                    e.v.cpu().numpy().copy()))
    assert len(out[0][0]) > 0
    assert np.array_equal(out[0][0], out[1][0]) and np.array_equal(out[0][1], out[1][1])
    assert np.array_equal(out[0][2], out[1][2])


# ---------------------------------------------------------------------------
# the point of M2d: v stays inside the reversal potentials
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("device", DEVICES)
def test_three_neuron_signs_and_bounds(three, device):
    p = EngineParams(g=0.02, adapt_g_b=0.05, **COND)
    e = LIFEngine(three, p, device=device)
    e.reset(0)
    i_ext = torch.zeros(3, device=device)
    i_ext[0] = 30.0
    vmin = np.inf
    for _ in range(80):
        e.step(i_ext)
        vmin = min(vmin, float(e.v.min()))
    t, idx = e.recorder.collect()
    a_t, b_t, c_t = (t[idx == k] for k in (0, 1, 2))
    assert len(a_t) >= 2, "driven neuron A must fire"
    assert len(b_t) >= 1 and b_t[0] > a_t[0], "excited B fires only after A"
    assert len(c_t) == 0, "inhibited C never fires"
    assert vmin >= p.v_lower_bound, f"v below the reversal potentials: {vmin}"
    assert float(e.g_a[0]) > 0, "the firing neuron accumulated g_a"
    assert float(e.g_a[1]) >= 0 and float(e.g_a[2]) == 0.0


@needs_cuda
@pytest.mark.parametrize("b_g", [0.01, 0.5])
def test_v_stays_within_reversal_potentials_on_the_real_graph(real_graph, b_g):
    """The M2c failure mode (v -> -834 mV) must be impossible by construction.

    Runs with no external current so the only drive is synaptic: then v can never
    leave [min(E_inh, E_adapt), v_thresh) whatever g_a does.
    """
    p = EngineParams(g=3e-3, adapt_g_b=b_g, **COND)   # ignited regime
    e = LIFEngine(real_graph, p, device="cuda")
    e.reset(0)
    fn = rc.stim100(real_graph.n)
    vmin, vmax = np.inf, -np.inf
    for k in range(250):
        fn(k, e.i_ext)
        e.step(e.i_ext)
        assert torch.isfinite(e.v).all() and torch.isfinite(e.g_a).all()
        vmin = min(vmin, float(e.v.min()))
        vmax = max(vmax, float(e.v.max()))
    t, _ = e.recorder.collect()
    assert len(t) > 0
    assert vmin >= p.v_lower_bound, f"v went below min(E_inh, E_adapt): {vmin}"
    assert vmax < p.v_thresh
    assert float(e.g_a.min()) >= 0
    assert float(e.g_a.max()) > 0


@needs_cuda
def test_adaptation_conductance_reduces_ignited_activity(real_graph):
    """Direction check on the full graph: more g_a per spike -> fewer late spikes."""
    fn = rc.stim100(real_graph.n)
    counts = []
    for b_g in (0.0, 0.2):
        e = LIFEngine(real_graph, EngineParams(g=3e-3, adapt_g_b=b_g, **COND), device="cuda")
        e.reset(0)
        t, _ = e.run(300, fn)
        counts.append(int((t >= 200).sum()))
    assert counts[0] > 0 and counts[1] < counts[0], counts


# ---------------------------------------------------------------------------
# single neuron: the mechanism does what it claims
# ---------------------------------------------------------------------------
def _isis(engine, i_ext, n_steps, device):
    buf = torch.zeros(1, device=device)
    buf[0] = i_ext
    engine.reset(0)
    t, _ = engine.run(n_steps, lambda k, b, buf=buf: b.copy_(buf))
    return np.diff(t.astype(np.float64)) * engine.params.dt


@pytest.mark.parametrize("device", DEVICES)
def test_single_neuron_isi_increases_and_rate_drops(one, device):
    kw = dict(g=0.0, **COND)
    base = LIFEngine(one, EngineParams(adapt_g_b=0.0, **kw), device=device)
    isi0 = _isis(base, 40.0, 2000, device)
    eng = LIFEngine(one, EngineParams(adapt_g_b=0.02, **kw), device=device)
    isi = _isis(eng, 40.0, 2000, device)
    assert len(isi0) > 10 and len(isi) > 5
    assert np.allclose(isi0, isi0[0]), "control (b_g = 0) must be perfectly regular"
    # ISI grows, then settles. Spike times are quantised to whole steps, so a
    # steady state whose true ISI is not an integer number of steps alternates
    # between the two neighbouring integers (e.g. 37/38 for a true 37.25 ms).
    # Monotonicity is therefore asserted up to one step of quantisation.
    dt = 1.0
    assert np.all(np.diff(isi) >= -dt), f"ISI not monotonic beyond quantisation: {isi}"
    assert isi[-1] > isi[0], f"ISI did not grow: {isi[0]} -> {isi[-1]}"
    tail = isi[len(isi) // 2:]
    assert tail.max() - tail.min() <= dt, f"ISI never settled: {tail}"
    assert np.mean(tail) > np.mean(isi[:3]), "ISI did not lengthen from transient to steady state"
    assert 1000.0 / np.mean(tail) < 1000.0 / isi0[-1]
    # g_a on its periodic orbit b/(1 - exp(-ISI/tau_a)), sampled somewhere along it
    d = np.exp(-np.mean(tail) / 100.0)
    hi = 0.02 / (1.0 - d)
    assert hi * d * 0.98 <= float(eng.g_a[0]) <= hi * 1.02


@pytest.mark.parametrize("device", DEVICES)
def test_g_a_decays_with_tau_a_and_resets(one, device):
    e = LIFEngine(one, EngineParams(g=0.0, adapt_g_b=0.03, **COND), device=device)
    _isis(e, 40.0, 300, device)
    g0 = float(e.g_a[0])
    assert g0 > 0
    e.run(100, None)
    assert float(e.g_a[0]) == pytest.approx(g0 * np.exp(-100 / 100.0), rel=1e-4)
    e.reset(0)
    assert float(e.g_a[0]) == 0.0


def test_dt_invariance(one):
    """Adapted ISIs agree interval by interval at dt = 1 ms and dt = 0.1 ms."""
    dev = "cuda" if CUDA else "cpu"
    out = {}
    for dt in (1.0, 0.1):
        e = LIFEngine(one, EngineParams(dt=dt, g=0.0, adapt_g_b=0.02, **COND), device=dev)
        out[dt] = _isis(e, 40.0, int(1000 / dt), dev)
    a, b = out[1.0], out[0.1]
    assert len(a) > 20 and abs(len(a) - len(b)) <= 1
    m = min(len(a), len(b))
    assert np.all(np.abs(a[:m] - b[:m]) <= 1.0 + 1e-6), (a[:m], b[:m])
    # one coarse step of quantisation, same bound as the interval-by-interval check
    assert a[-1] == pytest.approx(b[-1], abs=1.0 + 1e-6), "steady-state ISI depends on dt"


# ---------------------------------------------------------------------------
# backends and determinism
# ---------------------------------------------------------------------------
@needs_cuda
def test_triton_matches_torch(synth_graph):
    p = EngineParams(g=2e-3, adapt_g_b=0.05, **COND)
    outs, gs = [], []
    for backend in ("triton", "torch"):
        e = LIFEngine(synth_graph, p, device="cuda", backend=backend)
        e.reset(3)
        outs.append(e.run(200, rc.stim100(e.n, amp=40.0)))
        gs.append(e.g_a.cpu().numpy())
    assert len(outs[0][0]) > 100
    assert np.array_equal(outs[0][0], outs[1][0]) and np.array_equal(outs[0][1], outs[1][1])
    assert np.allclose(gs[0], gs[1], atol=1e-6)


@needs_cuda
def test_triton_matches_torch_three_neuron(three):
    outs, vs = [], []
    for backend in ("triton", "torch"):
        e = LIFEngine(three, EngineParams(g=0.02, adapt_g_b=0.05, **COND),
                      device="cuda", backend=backend)
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
def test_deterministic_bit_identical_real_graph(real_graph):
    p = EngineParams(g=3e-3, adapt_g_b=0.1, **COND)
    fn = rc.stim100(real_graph.n)
    e = LIFEngine(real_graph, p, device="cuda")
    e.reset(0); a = e.run(150, fn)
    e.reset(0); b = e.run(150, fn)
    assert len(a[0]) > 20_000, "test needs a high-activity regime"
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    e2 = LIFEngine(real_graph, p, device="cuda")
    e2.reset(0); c = e2.run(150, fn)
    assert np.array_equal(a[0], c[0]) and np.array_equal(a[1], c[1])
    e3 = LIFEngine(real_graph, p, device="cuda", use_cuda_graph=True)
    e3.reset(0); d = e3.run(150, fn)
    assert np.array_equal(a[0], d[0]) and np.array_equal(a[1], d[1])
    assert torch.equal(e.v, e3.v) and torch.equal(e.g_a, e3.g_a)


@pytest.mark.parametrize("device", DEVICES)
def test_noise_determinism_and_seed_dependence(synth_graph, device):
    p = EngineParams(g=2e-3, noise_sigma=3.0, adapt_g_b=0.05, **COND)
    e = LIFEngine(synth_graph, p, device=device)
    fn = rc.stim100(e.n, amp=40.0, device=device)
    e.reset(3); a = e.run(150, fn)
    e.reset(3); b = e.run(150, fn)
    e.reset(4); c = e.run(150, fn)
    assert len(a[0]) > 0 and np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert not (len(a[0]) == len(c[0]) and np.array_equal(a[0], c[0]))
