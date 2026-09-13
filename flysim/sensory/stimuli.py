"""M3: stimulus waveform generators (docs/m3-brief.md "자극 생성기").

A ``Stimulus`` is a float32 waveform ``x`` at sample rate ``fs`` (Hz), scaled so
that max|x| == 1 (0 for silence), plus a label and the parameters that made it.
Units are arbitrary: the adapter's ``a_in`` sets the injected current.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

DEFAULT_FS = 10_000.0     # Hz. = 1 sample per 0.1 ms step (phase-lock dt); 25x the 400 Hz band edge.

# ASSUMPTION (recorded in describe()): the carrier sits at the geometric centre of the
# adapter's 100–400 Hz passband, so the choice is tied to the filter, not to literature.
DEFAULT_CARRIER_HZ = 200.0
DEFAULT_PULSE_MS = 10.0   # ASSUMPTION: two carrier cycles fit inside one Hann-windowed pulse


@dataclass
class Stimulus:
    x: np.ndarray
    fs: float
    label: str
    params: dict = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        return len(self.x) / self.fs * 1000.0

    def envelope_1ms(self) -> np.ndarray:
        """0..1 envelope at 1 ms resolution (max |x| per ms) for run.json ``input.envelope``."""
        per_ms = int(round(self.fs / 1000.0))
        n = int(np.ceil(len(self.x) / per_ms))
        pad = np.zeros(n * per_ms, dtype=np.float32)
        pad[: len(self.x)] = np.abs(self.x)
        env = pad.reshape(n, per_ms).max(axis=1)
        m = env.max()
        return (env / m if m > 0 else env).astype(np.float32)

    def describe(self) -> dict:
        return {"label": self.label, "fs_hz": self.fs, "duration_ms": self.duration_ms,
                "n_samples": int(len(self.x)), "peak_abs": float(np.abs(self.x).max(initial=0.0)),
                **self.params}


def silence(duration_ms: float, fs: float = DEFAULT_FS) -> Stimulus:
    n = int(round(duration_ms * fs / 1000.0))
    return Stimulus(np.zeros(n, dtype=np.float32), fs, "silence",
                    {"kind": "silence", "duration_ms": duration_ms})


def click_train(ipi_ms: float, n_pulses: int, pulse_ms: float = DEFAULT_PULSE_MS,
                carrier_hz: float = DEFAULT_CARRIER_HZ, fs: float = DEFAULT_FS,
                pre_ms: float = 0.0, post_ms: float = 0.0) -> Stimulus:
    """Train of ``n_pulses`` Hann-windowed sine bursts (length ``pulse_ms``) whose onsets are
    ``ipi_ms`` apart, preceded by ``pre_ms`` and followed by ``post_ms`` of silence."""
    if ipi_ms <= 0 or n_pulses <= 0 or pulse_ms <= 0 or carrier_hz <= 0 or fs <= 0:
        raise ValueError("ipi_ms, n_pulses, pulse_ms, carrier_hz, fs must be > 0")
    if pulse_ms > ipi_ms:
        raise ValueError("pulse_ms must not exceed ipi_ms (pulses would overlap)")
    total_ms = pre_ms + (n_pulses - 1) * ipi_ms + pulse_ms + post_ms
    n = int(round(total_ms * fs / 1000.0))
    x = np.zeros(n, dtype=np.float64)
    L = int(round(pulse_ms * fs / 1000.0))
    tt = np.arange(L) / fs
    burst = np.hanning(L) * np.sin(2 * np.pi * carrier_hz * tt)
    onsets_ms = pre_ms + np.arange(n_pulses) * ipi_ms
    for on in onsets_ms:
        s = int(round(on * fs / 1000.0))
        x[s:s + L] += burst[: max(0, min(L, n - s))]
    peak = np.abs(x).max()
    x = (x / peak) if peak > 0 else x
    return Stimulus(x.astype(np.float32), fs, f"click train, IPI {ipi_ms:g} ms", {
        "kind": "click_train", "ipi_ms": ipi_ms, "n_pulses": n_pulses, "pulse_ms": pulse_ms,
        "carrier_hz": carrier_hz, "window": "hann", "pre_ms": pre_ms, "post_ms": post_ms,
        "stim_onset_ms": pre_ms, "stim_offset_ms": pre_ms + (n_pulses - 1) * ipi_ms + pulse_ms,
        "carrier_rationale": "geometric centre of the adapter passband 100-400 Hz (assumption, not literature)",
        "pulse_rationale": f"{pulse_ms * carrier_hz / 1000:g} carrier cycles per Hann pulse (assumption)",
    })
