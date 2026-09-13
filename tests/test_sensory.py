"""M3 tests: band-pass behaviour, current tables of both JO modes, describe() keys."""
import numpy as np
import pytest
import torch

from flysim.graph import Graph
from flysim.probe.sets import load_roi
from flysim.sensory import JOAdapter, click_train, silence
from flysim.sensory.jo import BAND_HZ, bandpass, bin_mean, envelope

FS = 10_000.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="session")
def g() -> Graph:
    return Graph.load(build_if_missing=True)


@pytest.fixture(scope="session")
def n_sites(g):
    return load_roi(g)["n_sites"].to_numpy()


def _gain(f_hz: float) -> float:
    t = np.arange(int(FS)) / FS
    y = bandpass(np.sin(2 * np.pi * f_hz * t), FS)
    return float(np.abs(y[len(y) // 2:]).max())      # steady state, second half


def test_bandpass_passband_and_stopband():
    assert _gain(200.0) > 0.85                          # inside 100-400 Hz
    assert 0.5 < _gain(BAND_HZ[0]) < 0.9                # ~ -3 dB at the edges
    assert 0.5 < _gain(BAND_HZ[1]) < 0.9
    assert _gain(20.0) < 0.1                            # 2nd-order HP: >= 20 dB down at 20 Hz
    assert _gain(2000.0) < 0.1                          # 2nd-order LP: >= 20 dB down at 2 kHz
    assert np.all(bandpass(np.zeros(1000), FS) == 0)


def test_envelope_and_binning():
    t = np.arange(int(FS)) / FS
    env = envelope(np.sin(2 * np.pi * 200 * t), FS)
    assert env.min() >= 0 and 0.3 < env[len(env) // 2:].mean() < 1.0
    x = np.arange(100, dtype=float)
    b = bin_mean(x, FS, 1.0, 12)                         # 10 samples per ms, 100 samples -> 10 bins + 2 empty
    assert b.shape == (12,) and b[0] == 4.5 and b[9] == 94.5 and b[10] == 0 and b[11] == 0


def test_stimuli():
    st = click_train(35, 15, pre_ms=100, post_ms=400)
    assert st.duration_ms == 1000 and st.params["stim_offset_ms"] == 600
    assert np.abs(st.x).max() == pytest.approx(1.0)
    env = st.envelope_1ms()
    assert env.shape == (1000,) and env[:100].max() == 0 and env[600:].max() == 0 and env.max() == 1
    assert env[100:600].max() == 1
    s = silence(1000)
    assert s.x.shape == (10000,) and s.x.max() == 0 and s.envelope_1ms().max() == 0
    with pytest.raises(ValueError):
        click_train(5, 3, pulse_ms=10)


@pytest.mark.parametrize("mode,dt,n_steps", [("rate", 1.0, 1000), ("phase-lock", 0.1, 10000)])
def test_jo_adapter_tables(g, n_sites, mode, dt, n_steps):
    st = click_train(35, 15, pre_ms=100, post_ms=400)
    ad = JOAdapter(g, n_sites, st, mode, a_in=10.0, device=DEVICE)
    ad.prepare(dt, n_steps)
    tab = ad.table_np
    assert tab.shape == (n_steps, len(ad.target_idx))
    assert tab.min() >= 0 and 0 < tab.max() <= 10.0        # a_in * unit drive, non-negative (rectified)
    on, off = int(100 / dt), int(600 / dt)
    assert tab[: on - 1].max() == 0                        # silent before onset (causal filter)
    assert tab[on:off].max() > 0
    assert tab[off + int(50 / dt):].max() < 1e-3 * tab.max()   # decayed within 50 ms after offset
    assert np.allclose(tab[:, 0], tab[:, -1])              # ild_db = 0: both sides identical
    # inject writes only targets, zeroes the rest, returns None
    buf = torch.full((g.n,), 7.0, device=DEVICE)
    assert ad.inject(on + 20, buf) is None
    b = buf.cpu().numpy()
    assert b[ad.target_idx].max() > 0 and np.delete(b, ad.target_idx).max() == 0
    ad.inject(n_steps + 5, buf)
    assert buf.abs().max().item() == 0
    d = ad.describe()
    for k in ("mode", "a_in", "ild_db", "band_hz", "filter", "n_targets", "n_left", "n_right",
              "n_dropped_no_synapse_sites", "target_types", "stimulus", "dt_ms", "n_steps"):
        assert k in d
    assert d["mode"] == mode and d["n_targets"] == d["n_left"] + d["n_right"]
    assert all(t.startswith("JO-A") or t.startswith("JO-B") for t in d["target_types"])


def test_jo_adapter_silence_and_ild(g, n_sites):
    ad = JOAdapter(g, n_sites, silence(300), "rate", a_in=50.0, device=DEVICE)
    ad.prepare(1.0, 300)
    assert ad.table_np.max() == 0
    ad = JOAdapter(g, n_sites, click_train(35, 5, pre_ms=20), "phase-lock", a_in=1.0, ild_db=-6.0, device=DEVICE)
    ad.prepare(0.1, 2000)
    left, right = ad.table_np[:, : len(ad.left)], ad.table_np[:, len(ad.left):]
    assert right.max() == pytest.approx(left.max() * 10 ** (-6 / 20), rel=1e-5)
    with pytest.raises(ValueError):
        JOAdapter(g, n_sites, silence(10), "envelope", a_in=1.0)
