"""M5a: the streaming Johnston's-organ input path.

The batch path (``flysim/sensory/jo.py``) filters a whole waveform at once; a live session
gets its signal a chunk at a time and must produce exactly one drive value per simulation
step. This module is the same signal path made stateful:

    PCM chunk -> band-pass 100-400 Hz (two Butterworth biquads, jo.biquad_coeffs)
              -> rate mode: full-wave rectify + 1st-order envelope (tau = jo.TAU_ENV_MS)
              -> mean of the samples that fall inside the step  (= jo.bin_mean for that bin)
              -> phase-lock mode: half-wave rectify after the mean

**Nothing in ``flysim/sensory`` is modified.** The coefficients, the band, the envelope time
constant and the two modes are imported from ``jo`` so the live and scripted paths cannot
drift apart; ``tests/test_live.py::test_stream_matches_batch`` feeds the same waveform through
both and compares.

Which samples belong to which step is decided by ``sample_end``, the exact inverse of
``jo.bin_mean``'s ``floor(n / fs * 1000 / dt + 1e-9)``, so a sample rate that is not a
multiple of the step rate (44.1 kHz at dt = 1 ms) still lands on the same grid.
"""
from __future__ import annotations

import math
import threading
from typing import Any

import numpy as np

from ..sensory import stimuli
from ..sensory.jo import (BAND_HZ, BUTTERWORTH_Q, MODES, TAU_ENV_MS, biquad_coeffs)

DEFAULT_FS = stimuli.DEFAULT_FS          # 10 kHz, as in M3
AUDIO_QUEUE_MAX_SAMPLES = 1 << 20        # ~105 s at 10 kHz; guards against a client flooding us


def sample_end(step: int, fs: float, dt_ms: float) -> int:
    """First global sample index that does NOT belong to ``step``.

    Inverse of ``jo.bin_mean``: sample ``n`` sits in bin ``floor(n/fs*1000/dt + 1e-9)``.
    """
    c = 1000.0 / (fs * dt_ms)            # bins per sample
    return int(math.ceil((step + 1 - 1e-9) / c))


# ---------------------------------------------------------------------------------------
# stateful filters (same arithmetic as jo.biquad_apply / jo.envelope, state carried over)
# ---------------------------------------------------------------------------------------
class Biquad:
    """Direct-form-II transposed IIR that keeps (z1, z2) between chunks."""

    def __init__(self, b: np.ndarray, a: np.ndarray):
        self.b0, self.b1, self.b2 = (float(v) for v in b)
        self.a1, self.a2 = float(a[1]), float(a[2])
        self.z1 = self.z2 = 0.0

    def reset(self) -> None:
        self.z1 = self.z2 = 0.0

    def process(self, x: np.ndarray) -> np.ndarray:
        y = np.empty(len(x), dtype=np.float64)
        z1, z2 = self.z1, self.z2
        b0, b1, b2, a1, a2 = self.b0, self.b1, self.b2, self.a1, self.a2
        for n, xn in enumerate(x.astype(np.float64)):
            yn = b0 * xn + z1
            z1 = b1 * xn - a1 * yn + z2
            z2 = b2 * xn - a2 * yn
            y[n] = yn
        self.z1, self.z2 = z1, z2
        return y


class BandPass:
    """jo.bandpass as a stateful pair of biquads (high-pass then low-pass)."""

    def __init__(self, fs: float, lo_hz: float = BAND_HZ[0], hi_hz: float = BAND_HZ[1]):
        self.fs, self.lo_hz, self.hi_hz = float(fs), float(lo_hz), float(hi_hz)
        self.hp = Biquad(*biquad_coeffs("highpass", lo_hz, fs))
        self.lp = Biquad(*biquad_coeffs("lowpass", hi_hz, fs))

    def reset(self) -> None:
        self.hp.reset()
        self.lp.reset()

    def process(self, x: np.ndarray) -> np.ndarray:
        return self.lp.process(self.hp.process(x))


class EnvelopeFollower:
    """jo.envelope as a stateful follower: |x| through a 1st-order low-pass."""

    def __init__(self, fs: float, tau_ms: float = TAU_ENV_MS):
        self.k = 1.0 - math.exp(-1000.0 / (float(fs) * float(tau_ms)))
        self.tau_ms = float(tau_ms)
        self.acc = 0.0

    def reset(self) -> None:
        self.acc = 0.0

    def process(self, x: np.ndarray) -> np.ndarray:
        y = np.empty(len(x), dtype=np.float64)
        acc, k = self.acc, self.k
        for n, v in enumerate(np.abs(x).astype(np.float64)):
            acc += k * (v - acc)
            y[n] = acc
        self.acc = acc
        return y


# ---------------------------------------------------------------------------------------
# stimulus sources: produce the next ``k`` samples of a continuous signal
# ---------------------------------------------------------------------------------------
class StimulusSource:
    kind = "silence"

    def generate(self, k: int) -> np.ndarray:
        raise NotImplementedError

    def describe(self) -> dict:
        return {"kind": self.kind}


class SilenceSource(StimulusSource):
    kind = "silence"

    def generate(self, k: int) -> np.ndarray:
        return np.zeros(k, dtype=np.float64)


class ToneSource(StimulusSource):
    """Continuous sine at ``carrier_hz`` (amplitude 1). Not in stimuli.py — the contract's
    ``tone`` kind is live-only; the waveform is an ASSUMPTION like every other stimulus."""

    kind = "tone"

    def __init__(self, carrier_hz: float, fs: float, duration_ms: float = 0.0):
        self.carrier_hz, self.fs = float(carrier_hz), float(fs)
        self.duration_samples = int(round(duration_ms * fs / 1000.0)) if duration_ms > 0 else 0
        self.n = 0

    def generate(self, k: int) -> np.ndarray:
        n = np.arange(self.n, self.n + k)
        self.n += k
        x = np.sin(2 * np.pi * self.carrier_hz * n / self.fs)
        if self.duration_samples:
            x = np.where(n < self.duration_samples, x, 0.0)
        return x

    def describe(self) -> dict:
        return {"kind": self.kind, "carrier_hz": self.carrier_hz, "fs_hz": self.fs,
                "duration_ms": self.duration_samples / self.fs * 1000.0 if self.duration_samples else 0.0,
                "waveform": "continuous sine, amplitude 1 (ASSUMPTION)"}


class ClickTrainSource(StimulusSource):
    """Endless (or ``duration_ms``-long) click train whose pulse is exactly the one
    ``stimuli.click_train`` builds: a Hann-windowed sine burst normalised to peak 1.
    The burst template is taken from that function, so the live and batch click trains are
    the same waveform (pulses never overlap: pulse_ms <= ipi_ms)."""

    kind = "click_train"

    def __init__(self, ipi_ms: float, pulse_ms: float, carrier_hz: float, fs: float,
                 duration_ms: float = 0.0):
        self.ipi_ms, self.pulse_ms, self.carrier_hz, self.fs = (float(ipi_ms), float(pulse_ms),
                                                                float(carrier_hz), float(fs))
        self.duration_ms = float(duration_ms)
        template = stimuli.click_train(ipi_ms, 1, pulse_ms=pulse_ms, carrier_hz=carrier_hz, fs=fs)
        self.burst = template.x.astype(np.float64)
        self.L = len(self.burst)
        self.stride = ipi_ms * fs / 1000.0
        self.n = 0
        self.n_pulses = (int(math.floor(duration_ms / ipi_ms + 1e-9)) if duration_ms > 0 else 0)

    def _onset(self, k: int) -> int:
        return int(round(k * self.stride))

    def generate(self, k: int) -> np.ndarray:
        out = np.zeros(k, dtype=np.float64)
        t0, t1 = self.n, self.n + k
        first = max(0, int((t0 - self.L) / self.stride) - 1)
        p = first
        while True:
            on = self._onset(p)
            if on >= t1:
                break
            if self.n_pulses and p >= self.n_pulses:
                break
            if on + self.L > t0:
                a, b = max(on, t0), min(on + self.L, t1)
                out[a - t0:b - t0] += self.burst[a - on:b - on]
            p += 1
        self.n = t1
        return out

    def describe(self) -> dict:
        return {"kind": self.kind, "ipi_ms": self.ipi_ms, "pulse_ms": self.pulse_ms,
                "carrier_hz": self.carrier_hz, "fs_hz": self.fs, "window": "hann",
                "duration_ms": self.duration_ms, "n_pulses": self.n_pulses or "endless",
                "pulse_source": "flysim.sensory.stimuli.click_train (single-pulse template)"}


class AudioSource(StimulusSource):
    """PCM pushed by the client over the websocket. ``generate`` never blocks: if fewer than
    ``k`` samples are queued the rest is silence and ``underruns`` counts the missing samples."""

    kind = "audio"

    def __init__(self, sample_rate: float):
        self.fs = float(sample_rate)
        self._lock = threading.Lock()      # push() runs on the websocket thread, generate() on the sim thread
        self._buf: list[np.ndarray] = []
        self._queued = 0
        self.underruns = 0
        self.received = 0
        self.dropped = 0

    def push(self, pcm: np.ndarray) -> int:
        pcm = np.asarray(pcm, dtype=np.float64).ravel()
        with self._lock:
            if self._queued + len(pcm) > AUDIO_QUEUE_MAX_SAMPLES:
                keep = max(0, AUDIO_QUEUE_MAX_SAMPLES - self._queued)
                self.dropped += len(pcm) - keep
                pcm = pcm[:keep]
            if len(pcm):
                self._buf.append(pcm)
                self._queued += len(pcm)
                self.received += len(pcm)
            return self._queued

    @property
    def queued(self) -> int:
        return self._queued

    def generate(self, k: int) -> np.ndarray:
        out = np.zeros(k, dtype=np.float64)
        filled = 0
        with self._lock:
            while filled < k and self._buf:
                head = self._buf[0]
                take = min(k - filled, len(head))
                out[filled:filled + take] = head[:take]
                filled += take
                self._queued -= take
                if take == len(head):
                    self._buf.pop(0)
                else:
                    self._buf[0] = head[take:]
            if filled < k:
                self.underruns += k - filled
        return out

    def describe(self) -> dict:
        return {"kind": self.kind, "fs_hz": self.fs, "received_samples": self.received,
                "underrun_samples": self.underruns, "dropped_samples": self.dropped,
                "queued_samples": self._queued}


def make_source(kind: str, *, fs: float, ipi_ms: float = 35.0, pulse_ms: float = 10.0,
                carrier_hz: float = 200.0, duration_ms: float = 0.0) -> StimulusSource:
    if kind == "silence":
        return SilenceSource()
    if kind == "tone":
        return ToneSource(carrier_hz, fs, duration_ms)
    if kind == "click_train":
        return ClickTrainSource(ipi_ms, pulse_ms, carrier_hz, fs, duration_ms)
    raise ValueError(f"make_source does not build {kind!r} (audio needs AudioSource(sample_rate))")


# ---------------------------------------------------------------------------------------
# the stream
# ---------------------------------------------------------------------------------------
class JOStream:
    """One drive value per simulation step, from whatever source is currently attached.

    ``drive`` is the unit-gain signal (before ``a_in`` and the per-side ILD gain), exactly
    the quantity ``jo.JOAdapter.drive`` produces for the same waveform.
    """

    def __init__(self, mode: str = "rate", dt_ms: float = 1.0, fs: float = DEFAULT_FS,
                 source: StimulusSource | None = None):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.mode = mode
        self.dt_ms = float(dt_ms)
        self.fs = float(fs)
        self.source: StimulusSource = source or SilenceSource()
        self.band = BandPass(self.fs)
        self.env = EnvelopeFollower(self.fs)
        self.step_index = 0
        self.n_cursor = 0
        self.last_drive = 0.0

    # ---- configuration -----------------------------------------------------------------
    def set_source(self, source: StimulusSource, *, restart_clock: bool = True) -> None:
        """Attach a new generator. The filters keep their state (the signal is continuous);
        the sample clock restarts so a replayed control log reproduces the same waveform."""
        self.source = source
        new_fs = float(getattr(source, "fs", self.fs))
        if new_fs != self.fs:
            self.fs = new_fs
            self.band = BandPass(self.fs)
            self.env = EnvelopeFollower(self.fs)
        if restart_clock:
            self.step_index = 0
            self.n_cursor = 0

    def set_mode(self, mode: str) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if mode != self.mode:
            self.mode = mode
            self.reset_filters()

    def set_dt(self, dt_ms: float) -> None:
        if float(dt_ms) != self.dt_ms:
            self.dt_ms = float(dt_ms)
            self.step_index = 0
            self.n_cursor = 0

    def reset_filters(self) -> None:
        self.band.reset()
        self.env.reset()
        self.last_drive = 0.0

    def reset(self) -> None:
        self.reset_filters()
        self.step_index = 0
        self.n_cursor = 0

    # ---- per step ----------------------------------------------------------------------
    def next_drive(self) -> float:
        """Advance one simulation step and return its drive value (>= 0)."""
        end = sample_end(self.step_index, self.fs, self.dt_ms)
        k = end - self.n_cursor
        self.step_index += 1
        self.n_cursor = end
        if k <= 0:
            return 0.0
        x = self.source.generate(k)
        sig = self.band.process(x)
        if self.mode == "rate":
            sig = self.env.process(sig)
        d = float(sig.mean())
        if self.mode == "phase-lock":
            d = max(d, 0.0)
        self.last_drive = d
        return d

    def drive_series(self, n_steps: int) -> np.ndarray:
        """``n_steps`` drive values in one call (used by tests and by the replay path)."""
        return np.array([self.next_drive() for _ in range(n_steps)], dtype=np.float64)

    def describe(self) -> dict:
        bh, ah = biquad_coeffs("highpass", BAND_HZ[0], self.fs)
        bl, al = biquad_coeffs("lowpass", BAND_HZ[1], self.fs)
        return {
            "adapter": "JOStream (live)",
            "mode": self.mode,
            "dt_ms": self.dt_ms,
            "band_hz": list(BAND_HZ),
            "filter": {"design": "2nd-order Butterworth HP @100 Hz + 2nd-order Butterworth LP @400 Hz "
                                 "(RBJ biquads, Q=1/sqrt2), causal, state carried across chunks (ASSUMPTION)",
                       "fs_hz": self.fs, "q": BUTTERWORTH_Q,
                       "highpass_b": bh.tolist(), "highpass_a": ah.tolist(),
                       "lowpass_b": bl.tolist(), "lowpass_a": al.tolist()},
            "rectification": ("full-wave + 1st-order low-pass envelope" if self.mode == "rate"
                              else "half-wave (max(x,0)) after the per-step mean"),
            "tau_env_ms": TAU_ENV_MS if self.mode == "rate" else None,
            "resample": "mean of the samples inside each dt step (= jo.bin_mean)",
            "per_neuron_heterogeneity": "none (ASSUMPTION)",
            "source": self.source.describe(),
        }
