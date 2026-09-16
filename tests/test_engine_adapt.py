"""M2c tests: spike-frequency adaptation (docs/m2c-brief.md).

Three things must hold.
1. ``adapt_b = 0`` leaves the pre-M2c kernel path untouched — bit-identical spikes
   and final v against the conductance reference recorded at commit 15b880d, and
   the ``ADAPT`` constexpr branch itself must be a no-op at b = 0.
2. The mechanism does what it claims on a single neuron under constant current:
   the ISI grows monotonically and the steady-state rate is below the b = 0 rate.
3. Triton == torch, and determinism is unchanged.
"""
import numpy as np
import pytest
import torch

from flysim.engine import CSRGraph, EngineParams, LIFEngine
from flysim.engine.params import DEFAULT_ADAPT_B
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
def one() -> CSRGraph:
    """Single neuron, no edges: the adaptation current in isolation."""
    return CSRGraph.from_edges(1, [], [], [])


@pytest.fixture(scope="session")
def reference():
    if not rc.NPZ.exists():
        pytest.skip("reference file missing")
    with np.load(rc.NPZ) as z:
        return {k: z[k] for k in z.files}


# ---------------------------------------------------------------------------
# 1. adapt_b = 0 is the pre-M2c engine
# ---------------------------------------------------------------------------
def test_adaptation_is_off_by_default_for_every_synapse_model():
    """DEFAULT_ADAPT_B is the sweep's answer, NOT the dataclass default: turning it on
    by default would silently change every M3/M4/live call site."""
    for kw in ({}, COND):
        p = EngineParams(**kw)
        assert p.adapt_b == 0.0 and p.adapt is False
        assert p.adapt_tau_w == 100.0                   # FIXED by the brief
    assert DEFAULT_ADAPT_B > 0, "the sweep found a target state; see docs/m2c-report.md §4.3"


def test_default_adapt_b_matches_recorded_sweep_rule():
    """DEFAULT_ADAPT_B == smallest b reaching the target state in m2c-noise-sweep.json."""
    import json
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "data-provenance" / "m2c-noise-sweep.json"
    if not path.exists():
        pytest.skip("sweep result not present")
    res = json.loads(path.read_text())
    assert res["target_state_b"], "no b reached the target state; DEFAULT_ADAPT_B should then be 0"
    assert DEFAULT_ADAPT_B == pytest.approx(min(res["target_state_b"]))
    assert res["default_adapt_b"] == pytest.approx(DEFAULT_ADAPT_B)
    assert res["base_params"]["adapt_tau_w"] == 100.0   # tau_w was not swept


@pytest.mark.parametrize("name", rc.CONDUCTANCE_CASES)
def test_conductance_regression_bit_identical(name, reference, real_graph, synth_graph):
    """M2b conductance results are unchanged by the M2c edit (reference: commit 15b880d)."""
    kind, pkw, ekw, *_ = rc.CASES[name]
    if ekw.get("device", "cuda") == "cuda" and not CUDA:
        pytest.skip("CUDA required")
    p = EngineParams(**pkw)
    assert p.synapse == "conductance" and p.adapt_b == 0.0
    t, idx, v = rc.run_case(name, real_graph, synth_graph)
    assert np.array_equal(t, reference[f"{name}__t"]), f"{name}: spike times differ from the M2b reference"
    assert np.array_equal(idx, reference[f"{name}__idx"]), f"{name}: spike neurons differ from the M2b reference"
    assert np.array_equal(v, reference[f"{name}__v"]), f"{name}: final v differs from the M2b reference"


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("synapse", ["current", "conductance"])
def test_adapt_branch_is_a_noop_at_zero_b(synth_graph, device, synapse):
    """b so small that (i_ext - w) == i_ext exactly in float32 must reproduce the
    ADAPT = False path bit for bit — i.e. the branch adds nothing but w itself."""
    g = 0.5 if synapse == "current" else 2e-3
    out = []
    for b in (0.0, 1e-30):
        p = EngineParams(g=g, noise_sigma=1.0, adapt_b=b, **({} if synapse == "current" else COND))
        assert p.adapt is (b > 0)
        e = LIFEngine(synth_graph, p, device=device)
        fn = rc.stim100(e.n, amp=40.0, device=device)
        e.reset(3)
        out.append((*e.run(120, fn), e.v.cpu().numpy().copy()))
    assert len(out[0][0]) > 0
    assert np.array_equal(out[0][0], out[1][0]) and np.array_equal(out[0][1], out[1][1])
    assert np.array_equal(out[0][2], out[1][2])


# ---------------------------------------------------------------------------
# 2. single neuron under constant current
# ---------------------------------------------------------------------------
def _isis(engine, i_ext, n_steps, device):
    buf = torch.zeros(1, device=device)
    buf[0] = i_ext
    engine.reset(0)
    t, _ = engine.run(n_steps, lambda t, b, buf=buf: b.copy_(buf))
    return np.diff(t.astype(np.float64)) * engine.params.dt


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("synapse", ["current", "conductance"])
def test_single_neuron_isi_increases_and_rate_drops(one, device, synapse):
    kw = dict(t_ref=2.0, **({} if synapse == "current" else COND))
    i_ext, ms = 40.0, 2000          # 40 mV drive: 25 mV above the 15 mV threshold gap
    base = LIFEngine(one, EngineParams(g=0.0, adapt_b=0.0, **kw), device=device)
    isi0 = _isis(base, i_ext, ms, device)
    eng = LIFEngine(one, EngineParams(g=0.0, adapt_b=2.0, **kw), device=device)
    isi = _isis(eng, i_ext, ms, device)
    assert len(isi0) > 10 and len(isi) > 5
    # ISI monotonically non-decreasing, and strictly longer at the end than at the start
    assert np.all(np.diff(isi) >= 0), f"ISI not monotonic: {isi}"
    assert isi[-1] > isi[0], f"ISI did not grow: {isi[0]} -> {isi[-1]}"
    assert np.allclose(isi0, isi0[0]), "control (b = 0) must be perfectly regular"
    # steady-state rate over the last second, below the unadapted rate
    rate0, rate = 1000.0 / isi0[-1], 1000.0 / isi[-1]
    assert rate < rate0, f"adapted rate {rate:.1f} Hz not below unadapted {rate0:.1f} Hz"
    # w sits on its periodic orbit: b/(1 - exp(-ISI/tau_w)) just after a spike,
    # decaying by exp(-ISI/tau_w) over the interval. The sample at the end of the
    # run is somewhere on that orbit.
    d = np.exp(-isi[-1] / 100.0)
    w_hi = 2.0 / (1.0 - d)
    assert w_hi * d * 0.99 <= float(eng.w[0]) <= w_hi * 1.01, (float(eng.w[0]), w_hi * d, w_hi)


@pytest.mark.parametrize("device", DEVICES)
def test_adaptation_is_monotonic_in_b(one, device):
    """More adaptation per spike -> fewer spikes. No threshold on the amount."""
    counts = []
    for b in (0.0, 0.5, 2.0, 8.0):
        e = LIFEngine(one, EngineParams(g=0.0, adapt_b=b), device=device)
        counts.append(len(_isis(e, 40.0, 2000, device)) + 1)
    assert counts == sorted(counts, reverse=True) and counts[0] > counts[-1], counts


@pytest.mark.parametrize("device", DEVICES)
def test_w_decays_with_tau_w_and_resets(one, device):
    """After the drive stops, w decays by exp(-dt/tau_w) per step and reset() clears it."""
    e = LIFEngine(one, EngineParams(g=0.0, adapt_b=3.0), device=device)
    _isis(e, 40.0, 300, device)
    w0 = float(e.w[0])
    assert w0 > 0
    e.run(100, None)                      # no input: w decays, no new spikes
    assert float(e.w[0]) == pytest.approx(w0 * np.exp(-100 / 100.0), rel=1e-4)
    e.reset(0)
    assert float(e.w[0]) == 0.0


def test_dt_invariance_of_adaptation(one):
    """The adapted ISI sequence is the same at dt = 1 ms and dt = 0.1 ms.

    Compared interval by interval, not on cumulative spike times: each ISI is
    quantised to a whole step, so at dt = 1 ms the rounding accumulates over the
    ~60 spikes of the run and says nothing about the dynamics."""
    dev = "cuda" if CUDA else "cpu"
    out = {}
    for dt in (1.0, 0.1):
        e = LIFEngine(one, EngineParams(dt=dt, g=0.0, adapt_b=2.0), device=dev)
        out[dt] = _isis(e, 40.0, int(1000 / dt), dev)
    a, b = out[1.0], out[0.1]
    assert len(a) > 30 and abs(len(a) - len(b)) <= 1
    m = min(len(a), len(b))
    assert np.all(np.abs(a[:m] - b[:m]) <= 1.0 + 1e-6), (a[:m], b[:m])
    assert a[-1] == pytest.approx(b[-1], abs=1e-6), "steady-state adapted ISI must not depend on dt"


# ---------------------------------------------------------------------------
# 3. backends agree; determinism unchanged
# ---------------------------------------------------------------------------
@needs_cuda
@pytest.mark.parametrize("synapse", ["current", "conductance"])
def test_triton_matches_torch_with_adaptation(synth_graph, synapse):
    g = 0.5 if synapse == "current" else 2e-3
    p = EngineParams(g=g, adapt_b=1.5, **({} if synapse == "current" else COND))
    outs, ws = [], []
    for backend in ("triton", "torch"):
        e = LIFEngine(synth_graph, p, device="cuda", backend=backend)
        e.reset(3)
        outs.append(e.run(200, rc.stim100(e.n, amp=40.0)))
        ws.append(e.w.cpu().numpy())
    assert len(outs[0][0]) > 100
    assert np.array_equal(outs[0][0], outs[1][0]) and np.array_equal(outs[0][1], outs[1][1])
    assert np.allclose(ws[0], ws[1], atol=1e-4)


@needs_cuda
def test_adaptation_deterministic_real_graph(real_graph):
    """Bit-identical across repeats, a fresh engine and CUDA-graph replay, with w live."""
    p = EngineParams(g=3e-3, adapt_b=2.0, **COND)     # ignited regime: adaptation is actually loaded
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
    assert torch.equal(e.v, e3.v) and torch.equal(e.w, e3.w)
    assert float(e.w.max()) > 0


@needs_cuda
def test_adaptation_reduces_ignited_activity_real_graph(real_graph):
    """Sanity on the full graph: in the ignited regime, b > 0 lowers the late-window
    spike count. This is a direction check, not a claim about any target state."""
    fn = rc.stim100(real_graph.n)
    counts = []
    for b in (0.0, 4.0):
        e = LIFEngine(real_graph, EngineParams(g=3e-3, adapt_b=b, **COND), device="cuda")
        e.reset(0)
        t, _ = e.run(300, fn)
        counts.append(int((t >= 200).sum()))
    assert counts[0] > 0 and counts[1] < counts[0], counts
