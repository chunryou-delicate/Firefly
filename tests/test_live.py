"""M5a tests: protocol validation, the streaming input path, the live session and a snapshot.

Everything except the last test runs on a small synthetic graph on the CPU, so the suite needs
neither a GPU nor the connectome cache. The last test is the real-graph integration check and
skips itself when the cache or CUDA is missing.
"""
import json
import os
import subprocess
import sys
import time

import numpy as np
import pytest
import torch

from flysim.engine import CSRGraph, EngineParams
from flysim.live import protocol as P
from flysim.live import stream as S
from flysim.live.session import LiveContext, LiveSession, compute_hops, replay_control_log
from flysim.probe.sets import ProbeSets
from flysim.sensory import jo, stimuli

N_SYNTH = 240
FS = stimuli.DEFAULT_FS


# ---------------------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def ctx() -> LiveContext:
    """Synthetic graph + probe sets shaped like the M3 ones (JO_AB_L/R first, rest last)."""
    rng = np.random.default_rng(0)
    n, m = N_SYNTH, 4_000
    pre, post = rng.integers(0, n, m), rng.integers(0, n, m)
    w = rng.integers(1, 6, m) * np.where(rng.random(m) < 0.25, -1, 1)
    graph = CSRGraph.from_edges(n, pre, post, w)
    jo_l = np.arange(0, 10, dtype=np.int64)
    jo_r = np.arange(10, 16, dtype=np.int64)
    members = {"JO_AB_L": jo_l, "JO_AB_R": jo_r,
               "JO_post_L": np.arange(16, 60, dtype=np.int64),
               "JO_post_R": np.arange(60, 100, dtype=np.int64)}
    region_of = np.full(n, 4, dtype=np.int16)
    for r, name in enumerate(members):
        region_of[members[name]] = r
    members["rest"] = np.flatnonzero(region_of == 4).astype(np.int64)
    names = list(members)
    sets = ProbeSets(names=names, members=members, region_of=region_of, log={})
    x = np.arange(n, dtype=np.int64) * 7 % 1000
    y = np.arange(n, dtype=np.int64) * 13 % 1000
    coords = (x, y, {"from_soma": n, "from_synapse_centroid": 0, "missing_placed_off_graph": 0})
    return LiveContext(graph=graph, sets=sets, jo_left=jo_l, jo_right=jo_r, coords=coords)


def make_session(ctx, tmp_path, **kw) -> LiveSession:
    kw.setdefault("device", "cpu")
    kw.setdefault("params", EngineParams(synapse="current", g=0.336, dt=1.0))
    kw.setdefault("a_in", 400.0)   # M3 needed 640 for the current model; 400 fires this synthetic net
    return LiveSession(ctx, session_dir=tmp_path, session_id="test", **kw)


def ctrl(type_, req_id=1, **fields):
    """Build a validated control message. ``type_`` avoids clashing with stimulus ``kind``."""
    _, msg = P.validate_client_message({"type": type_, "req_id": req_id, **fields})
    return msg


# ---------------------------------------------------------------------------------------
# protocol
# ---------------------------------------------------------------------------------------
def test_every_client_message_validates():
    samples = {
        "set_params": {"params": {"g": 0.5, "a_in": 100.0, "speed": 2.0, "mode": "phase-lock",
                                  "dt_ms": 0.1, "synapse": "current", "noise_sigma": 1.0}},
        "stimulus": {"kind": "click_train", "ipi_ms": 35, "pulse_ms": 10, "carrier_hz": 200,
                     "duration_ms": 0, "ild_db": -3},
        "audio_start": {"sample_rate": 48000, "channels": 1},
        "audio_stop": {},
        "inject": {"targets": {"set": "JO_AB_L"}, "amp": 30.0, "duration_ms": 50.0},
        "pause": {}, "resume": {}, "step": {"n": 5}, "reset": {"seed": 3},
        "get_hops": {"src": 0, "k": 3, "min_contacts": 2},
        "snapshot": {"run_id": "probe-01", "seconds": 2.0},
        "set_frame_every": {"n": 10},
    }
    assert set(samples) == set(P.CLIENT_MESSAGES)
    for kind, fields in samples.items():
        k, msg = P.validate_client_message({"type": kind, "req_id": 7, **fields})
        assert k == kind and msg["req_id"] == 7 and msg["type"] == kind
    # defaults fill in
    _, m = P.validate_client_message({"type": "stimulus", "req_id": 1, "kind": "click_train"})
    assert m["ipi_ms"] == 35.0 and m["pulse_ms"] == 10.0 and m["carrier_hz"] == 200.0
    assert m["duration_ms"] == 0.0 and m["ild_db"] == 0.0


@pytest.mark.parametrize("bad", [
    [1, 2, 3], "not an object", {"type": "nope", "req_id": 1}, {"req_id": 1},
    {"type": "pause"}, {"type": "pause", "req_id": "1"}, {"type": "pause", "req_id": True},
])
def test_malformed_messages_are_protocol_errors(bad):
    with pytest.raises(P.ProtocolError):
        P.validate_client_message(bad)
    err = P.error_msg("x", None)
    assert err["type"] == "error"


@pytest.mark.parametrize("msg", [
    {"type": "set_params", "req_id": 1, "params": {"g": -1}},
    {"type": "set_params", "req_id": 1, "params": {"dt_ms": 0.5}},
    {"type": "set_params", "req_id": 1, "params": {"mode": "envelope"}},
    {"type": "set_params", "req_id": 1, "params": {"synapse": "magic"}},
    {"type": "set_params", "req_id": 1, "params": {"nope": 1}},
    {"type": "set_params", "req_id": 1, "params": {}},
    {"type": "set_params", "req_id": 1, "params": {"g": float("nan")}},
    {"type": "stimulus", "req_id": 1, "kind": "jazz"},
    {"type": "stimulus", "req_id": 1, "kind": "click_train", "ipi_ms": 5, "pulse_ms": 10},
    {"type": "inject", "req_id": 1, "targets": {}, "amp": 1, "duration_ms": 1},
    {"type": "inject", "req_id": 1, "targets": {"set": "x", "idx": [1]}, "amp": 1, "duration_ms": 1},
    {"type": "inject", "req_id": 1, "targets": {"idx": [-1]}, "amp": 1, "duration_ms": 1},
    {"type": "get_hops", "req_id": 1, "src": 0, "k": 9},
    {"type": "snapshot", "req_id": 1, "run_id": "../escape", "seconds": 1},
    {"type": "snapshot", "req_id": 1, "run_id": "ok", "seconds": 999},
    {"type": "audio_start", "req_id": 1, "sample_rate": 10, "channels": 1},
    {"type": "audio_start", "req_id": 1, "sample_rate": 48000, "channels": 2},
    {"type": "set_frame_every", "req_id": 1, "n": 0},
])
def test_refused_messages_are_rejects(msg):
    with pytest.raises(P.Reject):
        P.validate_client_message(msg)


def test_adaptation_refused_until_m2c():
    if P.ENGINE_HAS_ADAPT:
        _, m = P.validate_client_message({"type": "set_params", "req_id": 1, "params": {"adapt_b": 0.5}})
        assert m["params"]["adapt_b"] == 0.5
    else:
        with pytest.raises(P.Reject):
            P.validate_client_message({"type": "set_params", "req_id": 1, "params": {"adapt_b": 0.5}})
        _, m = P.validate_client_message({"type": "set_params", "req_id": 1, "params": {"adapt_b": 0.0}})
        assert m["params"]["adapt_b"] == 0.0


def test_server_message_builders():
    f = P.frame_msg(t_ms=1.0, step=1, rates=[0.0], idx=[1], n_total=1, sample_ratio=1.0,
                    active_frac_100ms=0.0, input_level=0.0, status="lagging", speed=1.0,
                    params_version=2, n_bins=3)
    assert f["spikes"]["idx"] == [1] and f["status"] == "lagging" and f["n_bins"] == 3
    with pytest.raises(ValueError):
        P.frame_msg(t_ms=0, step=0, rates=[], idx=[], n_total=0, sample_ratio=1.0,
                    active_frac_100ms=0, input_level=0, status="on fire", speed=1,
                    params_version=0, n_bins=1)
    pm = P.params_msg({"g": 1.0}, 4)
    assert pm == {"type": "params", "params": {"g": 1.0}, "params_version": 4}
    assert "params" in P.SERVER_MESSAGES
    assert P.warning_msg("ignited_cleared", "x")["code"] == "ignited_cleared"
    with pytest.raises(ValueError):
        P.warning_msg("meltdown", "x")


# ---------------------------------------------------------------------------------------
# streaming input path
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("fs,dt", [(10_000.0, 1.0), (10_000.0, 0.1), (44_100.0, 1.0), (48_000.0, 0.1)])
def test_sample_end_inverts_bin_mean(fs, dt):
    n_steps = 40
    end = np.array([S.sample_end(s, fs, dt) for s in range(n_steps)])
    assert np.all(np.diff(end) > 0)
    n = np.arange(end[-1])
    bins = np.floor(n / fs * 1000.0 / dt + 1e-9).astype(np.int64)   # jo.bin_mean's own mapping
    for s in range(n_steps):
        lo = 0 if s == 0 else end[s - 1]
        assert np.all(bins[lo:end[s]] == s), f"step {s} mismatch"


@pytest.mark.parametrize("mode,dt", [("rate", 1.0), ("phase-lock", 0.1)])
def test_stream_matches_batch(mode, dt):
    """Chunked streaming must reproduce the batch path of flysim.sensory.jo."""
    st = stimuli.click_train(35.0, 6, pre_ms=20.0, post_ms=30.0)
    n_steps = int(round(st.duration_ms / dt))
    # batch reference: exactly the composition JOAdapter.drive() performs
    bp = jo.bandpass(st.x, st.fs)
    ref = jo.bin_mean(jo.envelope(bp, st.fs), st.fs, dt, n_steps) if mode == "rate" \
        else np.maximum(jo.bin_mean(bp, st.fs, dt, n_steps), 0.0)
    # streaming: the same waveform pushed in uneven chunks, then pulled one step at a time
    src = S.AudioSource(st.fs)
    rng = np.random.default_rng(1)
    i = 0
    while i < len(st.x):
        k = int(rng.integers(1, 257))
        src.push(st.x[i:i + k])
        i += k
    js = S.JOStream(mode=mode, dt_ms=dt, fs=st.fs, source=src)
    got = js.drive_series(n_steps)
    assert got.shape == ref.shape
    assert np.allclose(got, ref, atol=1e-5, rtol=1e-5), float(np.abs(got - ref).max())
    assert src.underruns == 0


def test_stream_click_train_source_matches_batch_waveform():
    """The endless generator emits the same samples as stimuli.click_train."""
    ipi, n_p = 35.0, 5
    batch = stimuli.click_train(ipi, n_p, pre_ms=0.0, post_ms=0.0)
    src = S.ClickTrainSource(ipi, 10.0, 200.0, batch.fs, duration_ms=0.0)
    live = np.concatenate([src.generate(97) for _ in range(int(len(batch.x) / 97) + 2)])[:len(batch.x)]
    assert np.allclose(live, batch.x, atol=1e-12)
    bounded = S.ClickTrainSource(ipi, 10.0, 200.0, batch.fs, duration_ms=2 * ipi)
    x = bounded.generate(int(5 * ipi * batch.fs / 1000))
    assert bounded.n_pulses == 2 and np.abs(x[int(2 * ipi * batch.fs / 1000) + 200:]).max() == 0


def test_prepare_audio_accepts_pcm_before_the_control_is_applied(ctx, tmp_path):
    """PCM that arrives between audio_start and its step boundary must not be lost."""
    s = make_session(ctx, tmp_path)
    src = s.prepare_audio(44_100)
    src.push(np.ones(441, dtype=np.float32))
    assert s.prepare_audio(44_100) is src and src.queued == 441
    ok, msg = s.apply_control(ctrl("audio_start", sample_rate=44_100))
    assert ok and "441 samples already buffered" in msg
    assert s.stream.source is src and s.stream.fs == 44_100.0
    s.apply_control(ctrl("audio_stop"))
    assert s.stream.source.kind == "silence"
    # a generated stimulus returns to the M3 sample rate even after an audio stream at 44.1 kHz
    s.apply_control(ctrl("stimulus", kind="click_train"))
    assert s.stream.fs == S.DEFAULT_FS and s.stream.source.fs == S.DEFAULT_FS
    s.stop()


def test_audio_source_underrun_and_cap():
    src = S.AudioSource(10_000.0)
    src.push(np.ones(5, dtype=np.float32))
    out = src.generate(8)
    assert np.allclose(out[:5], 1.0) and np.allclose(out[5:], 0.0) and src.underruns == 3
    src.push(np.ones(S.AUDIO_QUEUE_MAX_SAMPLES + 100, dtype=np.float32))
    assert src.dropped >= 100 and src.queued <= S.AUDIO_QUEUE_MAX_SAMPLES


def test_stream_describe_keys():
    js = S.JOStream(mode="rate", dt_ms=1.0)
    d = js.describe()
    for k in ("adapter", "mode", "band_hz", "filter", "rectification", "resample", "source"):
        assert k in d
    assert d["band_hz"] == list(jo.BAND_HZ) and d["tau_env_ms"] == jo.TAU_ENV_MS


# ---------------------------------------------------------------------------------------
# session
# ---------------------------------------------------------------------------------------
def test_hello_contract(ctx, tmp_path):
    s = make_session(ctx, tmp_path)
    h = s.hello()
    for k in ("n_neurons", "regions", "sets", "set_sizes", "params", "dt_ms", "bin_ms",
              "engine", "neurons_url"):
        assert k in h
    assert h["regions"] == s.region_names and "rest" not in h["sets"]
    assert set(h["assets"]) == {"neurons_3d", "skeleton_sets"}          # protocol v1.2
    assert h["set_sizes"]["rest"] == len(ctx.sets.members["rest"])
    assert set(h["params"]) == set(P.PARAM_NAMES)
    assert h["engine"]["synapse"] == "current"
    s.stop()


def test_set_params_version_and_reset(ctx, tmp_path):
    s = make_session(ctx, tmp_path)
    v0 = s.params_version
    ok, msg = s.apply_control(ctrl("set_params", params={"a_in": 123.0, "speed": 4.0}))
    assert ok and s.a_in == 123.0 and s.speed == 4.0 and s.params_version == v0 + 1
    for _ in range(5):
        s._step_once()
    assert s.step_index == 5
    ok, msg = s.apply_control(ctrl("set_params", params={"dt_ms": 0.1}))
    assert ok and "reset" in msg and s.dt_ms == 0.1 and s.step_index == 0
    assert s.stream.dt_ms == 0.1 and s.steps_per_bin == 10
    ok, msg = s.apply_control(ctrl("set_params", params={"synapse": "conductance"}))
    assert ok and s.engine.params.synapse == "conductance" and "reset" in msg
    ok, msg = s.apply_control(ctrl("set_params", params={"g": 0.5, "noise_sigma": 2.0}))
    assert ok and s.engine.params.g == 0.5 and s.engine.params.noise_sigma == 2.0
    s.stop()


def test_stimulus_and_inject_reach_the_input_buffer(ctx, tmp_path):
    s = make_session(ctx, tmp_path)
    s.apply_control(ctrl("stimulus", kind="click_train", ipi_ms=20.0, ild_db=-6.0))
    drives = [s._step_once() for _ in range(60)]
    assert max(drives) > 0, "click train produced no drive"
    on = int(np.argmax(np.array(drives) > 0))
    s.stream.reset()
    s.apply_control(ctrl("stimulus", kind="silence"))
    s._step_once()
    assert float(s._in_buf.abs().max()) == 0.0
    # inject lands on the requested set and expires
    s.apply_control(ctrl("inject", targets={"set": "JO_post_L"}, amp=7.0, duration_ms=3.0))
    s._step_once()
    buf = s._in_buf.cpu().numpy()
    assert np.allclose(buf[ctx.sets.members["JO_post_L"]], 7.0)
    assert buf[ctx.sets.members["JO_AB_L"]].max() == 0.0
    for _ in range(3):
        s._step_once()
    assert float(s._in_buf.abs().max()) == 0.0 and not s._injects
    # ILD: right side is scaled
    s.apply_control(ctrl("stimulus", kind="tone", carrier_hz=200.0, ild_db=-6.0))
    for _ in range(on + 40):
        s._step_once()
        b = s._in_buf.cpu().numpy()
        if b[ctx.jo_left].max() > 0:
            assert b[ctx.jo_right][0] == pytest.approx(b[ctx.jo_left][0] * 10 ** (-6 / 20), rel=1e-5)
            break
    else:
        pytest.fail("tone never reached the JO targets")
    s.stop()


def test_unknown_inject_set_is_rejected(ctx, tmp_path):
    s = make_session(ctx, tmp_path)
    with pytest.raises(P.Reject):
        s.apply_control(ctrl("inject", targets={"set": "NOPE"}, amp=1.0, duration_ms=10.0))
    with pytest.raises(P.Reject):
        s.apply_control(ctrl("inject", targets={"idx": [10 ** 9]}, amp=1.0, duration_ms=10.0))
    with pytest.raises(P.Reject):
        s.apply_control(ctrl("inject", targets={"idx": [1]}, amp=1.0, duration_ms=0.4))
    s.stop()


def test_frame_and_ring_buffer(ctx, tmp_path):
    emitted = []
    s = make_session(ctx, tmp_path, emit=lambda m, to=None: emitted.append(m))
    s.apply_control(ctrl("stimulus", kind="click_train", ipi_ms=20.0))
    for _ in range(80):
        payload = s._run_bin()
        frame = s._bundle_frame(payload)
        assert frame is not None                      # frame_every = 1
    assert len(s._ring) == 80
    assert len(frame["rates"]) == len(s.region_names)
    assert frame["spikes"]["n_total"] >= len(frame["spikes"]["idx"])
    assert 0.0 <= frame["input_level"] <= 1.0
    assert frame["status"] in P.STATUS_VALUES and frame["params_version"] == s.params_version
    assert frame["n_bins"] == 1
    assert max(f["input_level"] for f in s._ring) > 0
    assert any(f["n_total"] > 0 for f in s._ring), "synthetic net never fired"
    s.stop()


def test_frame_every_bundles_and_n_bins(ctx, tmp_path):
    emitted = []
    s = make_session(ctx, tmp_path, emit=lambda m, to=None: emitted.append(m))
    s.apply_control(ctrl("set_frame_every", n=5))
    out = [f for f in (s._bundle_frame(s._run_bin()) for _ in range(10)) if f is not None]
    assert len(out) == 2 and all(f["n_bins"] == 5 for f in out)
    # a partial bundle is flushed at the pause boundary with its real n_bins (contract v1.1)
    s._bundle_frame(s._run_bin())
    s._bundle_frame(s._run_bin())
    s.apply_control(ctrl("pause"))
    flushed = [m for m in emitted if m["type"] == "frame"]
    assert flushed and flushed[-1]["n_bins"] == 2 and flushed[-1]["status"] == "paused"
    assert not s._bundle
    s.stop()


def test_params_broadcast_on_every_version_change(ctx, tmp_path):
    """contract v1.1: an ack reaches only the requester, so params is broadcast to everyone."""
    sent = []
    s = make_session(ctx, tmp_path, emit=lambda m, to=None: sent.append((m, to)))
    v0 = s.params_version
    s.apply_control(ctrl("set_params", params={"a_in": 55.0}))
    bcast = [m for m, to in sent if m["type"] == "params" and to is None]
    assert len(bcast) == 1
    assert bcast[0]["params_version"] == v0 + 1 == s.params_version
    assert bcast[0]["params"] == s.params_public() and bcast[0]["params"]["a_in"] == 55.0
    assert set(bcast[0]["params"]) == set(P.PARAM_NAMES)
    s.apply_control(ctrl("set_params", params={"dt_ms": 0.1}))
    bcast = [m for m, to in sent if m["type"] == "params"]
    assert len(bcast) == 2 and bcast[-1]["params"]["dt_ms"] == 0.1
    # messages that do not change parameters must not broadcast
    n = len(bcast)
    s.apply_control(ctrl("pause"))
    s.apply_control(ctrl("reset", seed=1))
    assert len([m for m, to in sent if m["type"] == "params"]) == n
    s.stop()


def test_warnings_fire_and_clear(ctx, tmp_path):
    msgs = []
    s = make_session(ctx, tmp_path, emit=lambda m, to=None: msgs.append(m))
    s.apply_control(ctrl("stimulus", kind="click_train", ipi_ms=20.0))
    s.apply_control(ctrl("set_params", params={"a_in": 0.0}))     # input present, nothing fires
    for _ in range(400):
        s._bundle_frame(s._run_bin())
    codes = [m["code"] for m in msgs if m["type"] == "warning"]
    assert "all_silent" in codes
    s.apply_control(ctrl("set_params", params={"a_in": 400.0}))
    for _ in range(400):
        s._bundle_frame(s._run_bin())
    assert "all_silent_cleared" in [m["code"] for m in msgs if m["type"] == "warning"]
    s.stop()


def test_pause_step_resume(ctx, tmp_path):
    s = make_session(ctx, tmp_path)
    s.apply_control(ctrl("pause"))
    assert s._paused
    s.apply_control(ctrl("step", n=3))
    assert s._step_budget == 3
    s.apply_control(ctrl("resume"))
    assert not s._paused
    f = s.paused_frame()
    assert f["status"] == "paused" and f["spikes"]["idx"] == []
    s.stop()


def test_control_log_is_written(ctx, tmp_path):
    s = make_session(ctx, tmp_path)
    s.submit(ctrl("stimulus", kind="click_train", req_id=11))
    s.submit(ctrl("inject", targets={"idx": [5]}, amp=2.0, duration_ms=5.0, req_id=12))
    s._drain_commands()
    s.stop()
    lines = [json.loads(l) for l in s.log_path.read_text().splitlines()]
    assert lines[0]["kind"] == "session_start" and "engine_params" in lines[0]
    ctrls = [l for l in lines if l.get("kind") == "control"]
    assert [c["msg"]["req_id"] for c in ctrls] == [11, 12]
    assert all(c["ok"] and "step" in c and "wall" in c for c in ctrls)
    assert lines[-1]["kind"] == "session_stop"


def test_replay_of_the_control_log_is_deterministic(ctx, tmp_path):
    """The determinism guarantee: same log, same steps, same spikes."""
    s = make_session(ctx, tmp_path, a_in=400.0)
    n_steps = 300
    ts, ix = [], []
    for t in range(n_steps):
        if t == 20:
            s.submit(ctrl("stimulus", kind="click_train", ipi_ms=25.0, req_id=1))
        if t == 90:
            s.submit(ctrl("inject", targets={"set": "JO_post_R"}, amp=25.0, duration_ms=20.0, req_id=2))
        if t == 150:
            s.submit(ctrl("set_params", params={"a_in": 800.0}, req_id=3))
        if t == 220:
            s.submit(ctrl("stimulus", kind="tone", carrier_hz=300.0, req_id=4))
        s._drain_commands()
        s._step_once()
        fired = torch.nonzero(s.engine.spikes).flatten().numpy()
        if fired.size:
            ts.append(np.full(fired.size, t, dtype=np.int64))
            ix.append(fired.astype(np.int64))
    s.stop()
    t1 = np.concatenate(ts) if ts else np.empty(0, np.int64)
    i1 = np.concatenate(ix) if ix else np.empty(0, np.int64)
    assert t1.size > 100, "the run produced too few spikes to be a determinism test"
    t2, i2 = replay_control_log(ctx, s.log_path, n_steps, device="cpu")
    assert np.array_equal(t1, t2) and np.array_equal(i1, i2)
    t3, i3 = replay_control_log(ctx, s.log_path, n_steps, device="cpu")
    assert np.array_equal(t2, t3) and np.array_equal(i2, i3)


def test_snapshot_writes_a_viewer_run_json(ctx, tmp_path):
    from flysim.probe.export import REQUIRED_KEYS, validate_run_doc

    s = make_session(ctx, tmp_path, a_in=400.0)
    s.apply_control(ctrl("stimulus", kind="click_train", ipi_ms=20.0))
    for _ in range(200):
        s._run_bin()
    res = s.snapshot("live-test", 0.15, out_dir=tmp_path / "snap")
    doc = json.loads((tmp_path / "snap" / "run.json").read_text())
    validate_run_doc(doc)
    for sec, keys in REQUIRED_KEYS.items():
        assert all(k in doc[sec] for k in keys)
    assert res["n_frames"] == 150 and doc["frames"]["n"] == 150
    assert doc["meta"]["sensory_mode"] == "rate"
    assert doc["meta"]["live_control_log"] == str(s.log_path)
    assert doc["meta"]["live_params"]["g"] == s.engine.params.g
    assert doc["meta"]["n_neurons"] == ctx.n and doc["regions"] == s.region_names
    assert 0.0 < doc["meta"]["spike_sample_ratio"] <= 1.0
    assert max(max(r) for r in doc["frames"]["rates"]) <= 1.0
    s.stop()


def test_snapshot_without_data_is_rejected(ctx, tmp_path):
    s = make_session(ctx, tmp_path)
    with pytest.raises(P.Reject):
        s.snapshot("empty", 1.0, out_dir=tmp_path / "x")
    s.stop()


def test_hops(ctx):
    layers, trunc = compute_hops(ctx.graph, 0, 3, min_contacts=1)
    assert len(layers) == 3 and all(isinstance(l, list) for l in layers)
    seen = {0}
    for layer in layers:
        assert not (set(layer) & seen)          # each neuron appears in one layer only
        seen |= set(layer)
    post, w = ctx.graph.indices[ctx.graph.indptr[0]:ctx.graph.indptr[1]], None
    assert set(layers[0]) <= set(int(p) for p in post)
    strict, _ = compute_hops(ctx.graph, 0, 2, min_contacts=5)
    assert len(strict[0]) <= len(layers[0])
    capped, trunc2 = compute_hops(ctx.graph, 0, 2, min_contacts=1, cap=3)
    assert len(capped[0]) <= 3 and trunc2[0] == (len(layers[0]) > 3)
    with pytest.raises(P.Reject):
        compute_hops(ctx.graph, 10 ** 9, 1)


def test_session_thread_runs_and_paces(ctx, tmp_path):
    frames = []
    s = make_session(ctx, tmp_path, speed=50.0, emit=lambda m, to=None: frames.append(m))
    s.start()
    time.sleep(0.5)
    s.stop()
    got = [f for f in frames if f["type"] == "frame"]
    assert got, "the loop produced no frames"
    assert s.step_index > 5
    m = s.pacing_metrics()
    assert m["bins"] > 0 and m["mean_bin_us"] > 0
    assert all(f["status"] in P.STATUS_VALUES for f in got)


# ---------------------------------------------------------------------------------------
# M6b: 3D asset serving (docs/m6-3d-contract.md) and hello.assets (protocol v1.2)
# ---------------------------------------------------------------------------------------
@pytest.fixture
def http3d(tmp_path, monkeypatch):
    """The static server pointed at a temporary cache directory, on an ephemeral port."""
    import http.client

    from flysim.live import static as ST

    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setattr(ST, "CACHE_DIR", cache)
    monkeypatch.setattr(ST, "OUT_JSON", cache / "neurons-3d.json")
    monkeypatch.setattr(ST, "OUT_BIN", cache / "neurons-3d.bin")
    httpd = ST.serve_static("127.0.0.1", 0, 8765)
    port = httpd.server_address[1]

    def get(path):
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        try:
            c.request("GET", path)
            r = c.getresponse()
            return r.status, r.getheader("Content-Type"), r.read()
        finally:
            c.close()

    try:
        yield get, cache
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_3d_endpoints_404_before_the_files_exist(http3d):
    get, cache = http3d
    for path in ("/neurons-3d.json", "/neurons-3d.bin", "/skel/pC1.json", "/skel/pC1.bin"):
        assert get(path)[0] == 404, path
    # the listing is a listing, not a file: 200 with an empty array
    status, ctype, body = get("/skel/index.json")
    assert status == 200 and ctype == "application/json" and json.loads(body) == []


def test_3d_endpoints_serve_the_contract_files(http3d):
    get, cache = http3d
    doc = {"n": 2, "voxel_nm": 8, "arrays": {"pos": {"offset": 0, "length": 6, "dtype": "float32"}}}
    payload = np.arange(6, dtype=np.float32).tobytes()
    (cache / "neurons-3d.json").write_text(json.dumps(doc))
    (cache / "neurons-3d.bin").write_bytes(payload)
    (cache / "skel-pC1.json").write_text(json.dumps({"set": "pC1", "n_neurons": 3, "n_segments": 9}))
    (cache / "skel-pC1.bin").write_bytes(b"\x01\x02\x03\x04")

    status, ctype, body = get("/neurons-3d.json")
    assert status == 200 and ctype == "application/json" and json.loads(body) == doc
    status, ctype, body = get("/neurons-3d.bin")
    assert status == 200 and ctype == "application/octet-stream" and body == payload
    status, ctype, body = get("/skel/pC1.json")
    assert status == 200 and json.loads(body)["set"] == "pC1"
    status, ctype, body = get("/skel/pC1.bin")
    assert status == 200 and ctype == "application/octet-stream" and body == b"\x01\x02\x03\x04"
    status, _, body = get("/skel/index.json")
    assert status == 200 and json.loads(body) == [
        {"name": "pC1", "n_neurons": 3, "n_segments": 9, "bytes": 4}]
    assert get("/skel/WED_L.json")[0] == 404          # a name with no bundle


@pytest.mark.parametrize("path", [
    "/skel/../../etc/passwd", "/skel/../neurons-v1.json", "/../etc/passwd", "/etc/passwd",
    "/skel/a/b.json", "/skel/.hidden.json", "/skel/pC1.txt", "/skel/", "/skel/pC1",
    "/data/cache/neurons-3d.bin", "/neurons-3d.bin/", "/nope",
])
def test_paths_outside_the_allow_list_are_404(http3d, path):
    get, _ = http3d
    assert get(path)[0] == 404, path


def test_existing_routes_still_work(http3d):
    """M6b must not disturb the M5a allow-list."""
    get, _ = http3d
    assert get("/flysim-viewer.html")[0] == 200
    assert get("/")[0] == 200                          # cockpit page (or its placeholder)
    assert get("/neurons-v1.json")[0] in (200, 404)    # 404 only when it has not been generated


def test_hello_assets_reflects_the_files_on_disk(ctx, tmp_path, monkeypatch):
    from flysim.probe import export3d as E3

    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setattr(E3, "CACHE_DIR", cache)
    s = make_session(ctx, tmp_path)
    assert s.hello()["assets"] == {"neurons_3d": False, "skeleton_sets": []}
    (cache / E3.OUT_JSON.name).write_text("{}")
    (cache / E3.OUT_BIN.name).write_bytes(b"")
    (cache / "skel-JO_AB_L.json").write_text(json.dumps({"n_neurons": 1, "n_segments": 2}))
    (cache / "skel-JO_AB_L.bin").write_bytes(b"\0" * 24)
    assets = s.hello()["assets"]                       # read again per hello, not cached
    assert assets == {"neurons_3d": True, "skeleton_sets": ["JO_AB_L"]}
    with pytest.raises(ValueError):
        P.validate_assets({"skeleton_sets": [1, 2]})
    s.stop()


# ---------------------------------------------------------------------------------------
# real graph + CUDA integration (skipped when either is missing)
# ---------------------------------------------------------------------------------------
def other_cuda_processes() -> list[int]:
    """PIDs of CUDA compute processes other than this one, as nvidia-smi reports them.

    A neighbour on the GPU (another window's resident flysim-live server, say) multiplies the
    per-bin wall time — 404 us idle vs 10,027 us measured beside a running server — so a
    timing assertion made next to one measures the neighbour, not this code. Unknown (no
    nvidia-smi) counts as "nobody else".
    """
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            text=True, timeout=10)
    except Exception:
        return []
    me = os.getpid()
    return [int(t) for t in out.split() if t.strip().isdigit() and int(t) != me]



@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_real_graph_end_to_end(tmp_path):
    from flysim.graph.constants import CACHE_NPZ
    if not CACHE_NPZ.exists():
        pytest.skip("graph cache missing")
    from flysim.live.export_neurons import build_context

    ctx = build_context()
    frames = []
    s = LiveSession(ctx, params=EngineParams(synapse="conductance", dt=1.0), device="cuda",
                    a_in=640.0, mode="rate", speed=1.0, session_dir=tmp_path, session_id="it",
                    emit=lambda m, to=None: frames.append(m))
    s.apply_control(ctrl("stimulus", kind="click_train", ipi_ms=35.0))
    for _ in range(60):          # Triton compilation and the CUDA-graph capture are one-off
        s._run_bin()
    s.metrics.update(bins=0, bin_us_sum=0.0, bin_us_max=0.0, steps=0)
    got = [f for f in (s._bundle_frame(s._run_bin()) for _ in range(500)) if f is not None]
    assert len(got) == 500
    assert all(f["n_bins"] == 1 for f in got)
    assert any(f["spikes"]["n_total"] > 0 for f in got), "no spikes on the real graph"
    assert max(f["input_level"] for f in got) > 0
    # the CUDA-graph capture the live session enables by default must not change the dynamics
    ref = LiveSession(ctx, params=EngineParams(synapse="conductance", dt=1.0), device="cuda",
                      use_cuda_graph=False, a_in=640.0, mode="rate", session_dir=tmp_path,
                      session_id="it-eager", log_controls=False)
    cap = LiveSession(ctx, params=EngineParams(synapse="conductance", dt=1.0), device="cuda",
                      use_cuda_graph=True, a_in=640.0, mode="rate", session_dir=tmp_path,
                      session_id="it-graph", log_controls=False)
    for sess in (ref, cap):
        sess.apply_control(ctrl("stimulus", kind="click_train", ipi_ms=35.0))
    fired_ref, fired_cap = [], []
    for _ in range(200):
        ref._step_once(); cap._step_once()
        fired_ref.append(torch.nonzero(ref.engine.spikes).flatten().cpu().numpy())
        fired_cap.append(torch.nonzero(cap.engine.spikes).flatten().cpu().numpy())
    assert sum(len(f) for f in fired_ref) > 0
    assert all(np.array_equal(a, b) for a, b in zip(fired_ref, fired_cap))
    ref.stop(); cap.stop()

    res = s.snapshot("live-it", 0.3, out_dir=tmp_path / "snap")
    doc = json.loads((tmp_path / "snap" / "run.json").read_text())
    assert doc["meta"]["n_neurons"] == ctx.n and res["n_frames"] == 300
    m = s.pacing_metrics()
    assert m["bins"] == 500 and m["steps"] == 500 and m["mean_bin_us"] > 0, m
    s.stop()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_real_graph_pacing_budget(tmp_path):
    """The dt = 1 ms budget (< 1,000 us/bin), checked only when this process has the GPU alone.

    The number itself belongs to `python -m flysim.live.bench` (docs/m5a-report.md §4); this is
    a regression guard. Raising the ceiling to survive a neighbour would leave a guard that
    guards nothing, so a shared GPU skips the check and says so.
    """
    from flysim.graph.constants import CACHE_NPZ
    if not CACHE_NPZ.exists():
        pytest.skip("graph cache missing")
    from flysim.live.export_neurons import build_context

    s = LiveSession(build_context(), params=EngineParams(synapse="conductance", dt=1.0),
                    device="cuda", a_in=640.0, mode="rate", speed=1e9, session_dir=tmp_path,
                    session_id="pacing", log_controls=False)
    s.apply_control(ctrl("stimulus", kind="click_train", ipi_ms=35.0))
    for _ in range(60):          # Triton compilation and the CUDA-graph capture are one-off
        s._run_bin()
    s.metrics.update(bins=0, bin_us_sum=0.0, bin_us_max=0.0, steps=0)
    for _ in range(500):
        s._run_bin()
    m = s.pacing_metrics()
    s.stop()
    others = other_cuda_processes()
    if others:
        pytest.skip(f"budget not checked: {len(others)} other CUDA process(es) {others} share "
                    f"the GPU, so this measures them too (got {m['mean_bin_us']:.0f} us/bin; "
                    f"the number of record comes from `python -m flysim.live.bench`)")
    assert m["mean_bin_us"] < 1000, m
