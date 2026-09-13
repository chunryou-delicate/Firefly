"""M3: Johnston's organ (JO) adapter (docs/m3-brief.md "JO 어댑터").

Targets: retained neurons whose ``type`` starts with ``JO-A`` or ``JO-B`` (queried
from the data, see ``jo_targets``) and that have at least one synaptic site.
Left/right come from ``Graph.side`` (somaSide is null for all JO neurons; the
instance suffix ``_L/_R`` is used, which equals ``rootSide`` for these 138).

Signal path (every constant below is an ASSUMPTION, recorded in ``describe()``):

    x (arbitrary units, max|x| = 1)
      -> band-pass 100-400 Hz: 2nd-order Butterworth high-pass @100 Hz + 2nd-order
         Butterworth low-pass @400 Hz (two RBJ biquads, Q = 1/sqrt2; 4th order overall,
         12 dB/oct per side). Causal, applied once.
      rate mode      (dt = 1.0 ms): |.| -> 1st-order low-pass, tau_env = 2 ms
                                    -> mean per dt bin -> i = a_in * env
      phase-lock mode(dt = 0.1 ms): mean per dt bin -> i = a_in * max(x, 0)   (half-wave)
    Same current for every target neuron on a side (no heterogeneity, no phase offsets).
    ``ild_db`` scales right vs left by 10^(ild_db/20) (default 0 = identical).
"""
from __future__ import annotations

import math

import numpy as np

from ..graph.constants import SIDE_CODES
from .base import TableAdapter
from .stimuli import Stimulus

BAND_HZ = (100.0, 400.0)     # ASSUMPTION: CLAUDE.md §5.3 "대략 100~400Hz"
TAU_ENV_MS = 2.0             # ASSUMPTION: envelope low-pass time constant (rate mode)
BUTTERWORTH_Q = 1.0 / math.sqrt(2.0)
JO_TYPE_PATTERN = r"^JO-[AB]"   # types verified by search_type at run time; listed in describe()
MODES = ("rate", "phase-lock")


# ---------------------------------------------------------------------------
# filters (numpy only; scipy is not an approved dependency)
# ---------------------------------------------------------------------------
def biquad_coeffs(kind: str, f_hz: float, fs: float, q: float = BUTTERWORTH_Q) -> tuple[np.ndarray, np.ndarray]:
    """RBJ audio-EQ-cookbook biquad. Returns (b[3], a[3]) with a[0] == 1."""
    if not 0 < f_hz < fs / 2:
        raise ValueError(f"cutoff {f_hz} Hz must be in (0, fs/2={fs/2})")
    w0 = 2 * math.pi * f_hz / fs
    c, s = math.cos(w0), math.sin(w0)
    alpha = s / (2 * q)
    if kind == "lowpass":
        b = np.array([(1 - c) / 2, 1 - c, (1 - c) / 2])
    elif kind == "highpass":
        b = np.array([(1 + c) / 2, -(1 + c), (1 + c) / 2])
    else:
        raise ValueError(kind)
    a = np.array([1 + alpha, -2 * c, 1 - alpha])
    return b / a[0], a / a[0]


def biquad_apply(x: np.ndarray, b: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Direct-form-II transposed IIR, zero initial state, causal."""
    y = np.empty(len(x), dtype=np.float64)
    z1 = z2 = 0.0
    b0, b1, b2 = float(b[0]), float(b[1]), float(b[2])
    a1, a2 = float(a[1]), float(a[2])
    for n, xn in enumerate(x.astype(np.float64)):
        yn = b0 * xn + z1
        z1 = b1 * xn - a1 * yn + z2
        z2 = b2 * xn - a2 * yn
        y[n] = yn
    return y


def bandpass(x: np.ndarray, fs: float, lo_hz: float = BAND_HZ[0], hi_hz: float = BAND_HZ[1]) -> np.ndarray:
    bh, ah = biquad_coeffs("highpass", lo_hz, fs)
    bl, al = biquad_coeffs("lowpass", hi_hz, fs)
    return biquad_apply(biquad_apply(x, bh, ah), bl, al)


def envelope(x: np.ndarray, fs: float, tau_ms: float = TAU_ENV_MS) -> np.ndarray:
    """Full-wave rectification followed by a 1st-order low-pass (time constant tau_ms)."""
    k = 1.0 - math.exp(-1000.0 / (fs * tau_ms))
    r = np.abs(x).astype(np.float64)
    y = np.empty_like(r)
    acc = 0.0
    for n, v in enumerate(r):
        acc += k * (v - acc)
        y[n] = acc
    return y


def bin_mean(x: np.ndarray, fs: float, dt_ms: float, n_steps: int) -> np.ndarray:
    """Mean of the signal inside each dt bin; bins past the signal are 0."""
    t_bin = np.floor(np.arange(len(x)) / fs * 1000.0 / dt_ms + 1e-9).astype(np.int64)
    keep = t_bin < n_steps
    s = np.bincount(t_bin[keep], weights=x[keep], minlength=n_steps)
    c = np.bincount(t_bin[keep], minlength=n_steps)
    return np.where(c > 0, s / np.maximum(c, 1), 0.0)


# ---------------------------------------------------------------------------
# targets
# ---------------------------------------------------------------------------
def jo_targets(graph, n_sites: np.ndarray) -> dict:
    """JO-A/B neurons with >= 1 synaptic site, split by side. Raises if a side is empty.

    ``n_sites``: per-neuron synaptic site count aligned with graph indices
    (data/cache/neuron-roi-v1.parquet, column n_sites)."""
    types = graph.query.search_type(JO_TYPE_PATTERN)
    idx = np.unique(np.concatenate([graph.query.by_type(t) for t, _ in types]))
    has_sites = n_sites[idx] > 0
    dropped = idx[~has_sites]
    idx = idx[has_sites]
    side = graph.side[idx]
    left, right = idx[side == SIDE_CODES["L"]], idx[side == SIDE_CODES["R"]]
    other = idx[(side != SIDE_CODES["L"]) & (side != SIDE_CODES["R"])]
    if len(left) == 0 or len(right) == 0:
        raise LookupError(f"JO-A/B targets: left {len(left)}, right {len(right)} — need both sides")
    return {"types": types, "n_typed": int(len(idx) + len(dropped)), "dropped_no_sites": dropped,
            "left": left, "right": right, "no_side": other}


class JOAdapter(TableAdapter):
    def __init__(self, graph, n_sites: np.ndarray, stimulus: Stimulus, mode: str, a_in: float,
                 ild_db: float = 0.0, device: str = "cuda"):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if a_in < 0:
            raise ValueError("a_in must be >= 0")
        tg = jo_targets(graph, n_sites)
        self.left, self.right = tg["left"], tg["right"]
        self._targets_info = tg
        super().__init__(np.concatenate([self.left, self.right]), device)
        self.graph = graph
        self.stimulus = stimulus
        self.mode = mode
        self.a_in = float(a_in)
        self.ild_db = float(ild_db)
        self.gain_left = 1.0
        self.gain_right = 10 ** (self.ild_db / 20.0)
        self._bp_cache: np.ndarray | None = None
        self._drive: np.ndarray | None = None

    # ---- signal processing ---------------------------------------------------
    def drive(self, dt_ms: float, n_steps: int) -> np.ndarray:
        """Unit-gain per-step drive (before a_in and side gains), length n_steps."""
        fs = self.stimulus.fs
        if self._bp_cache is None:
            self._bp_cache = bandpass(self.stimulus.x, fs)
        bp = self._bp_cache
        if self.mode == "rate":
            sig = envelope(bp, fs)
            return bin_mean(sig, fs, dt_ms, n_steps)
        sig = bin_mean(bp, fs, dt_ms, n_steps)
        return np.maximum(sig, 0.0)                    # half-wave rectification

    def prepare(self, dt_ms: float, n_steps: int) -> None:
        d = self.drive(dt_ms, n_steps)
        self._drive = d
        gains = np.concatenate([np.full(len(self.left), self.gain_left), np.full(len(self.right), self.gain_right)])
        table = self.a_in * d[:, None] * gains[None, :]
        self._set_table(table, dt_ms)

    # ---- provenance ----------------------------------------------------------
    def describe(self) -> dict:
        fs = self.stimulus.fs
        bh, ah = biquad_coeffs("highpass", BAND_HZ[0], fs)
        bl, al = biquad_coeffs("lowpass", BAND_HZ[1], fs)
        tg = self._targets_info
        d = {
            "adapter": "JOAdapter",
            "mode": self.mode,
            "a_in": self.a_in,
            "ild_db": self.ild_db,
            "gain_left": self.gain_left, "gain_right": self.gain_right,
            "band_hz": list(BAND_HZ),
            "filter": {"design": "2nd-order Butterworth HP @100 Hz + 2nd-order Butterworth LP @400 Hz "
                                 "(RBJ biquads, Q=1/sqrt2), causal; 4th order overall (ASSUMPTION)",
                       "fs_hz": fs, "q": BUTTERWORTH_Q,
                       "highpass_b": bh.tolist(), "highpass_a": ah.tolist(),
                       "lowpass_b": bl.tolist(), "lowpass_a": al.tolist()},
            "rectification": "full-wave + 1st-order low-pass envelope" if self.mode == "rate" else "half-wave (max(x,0))",
            "tau_env_ms": TAU_ENV_MS if self.mode == "rate" else None,
            "resample": "mean per dt bin",
            "per_neuron_heterogeneity": "none (ASSUMPTION)",
            "target_type_pattern": JO_TYPE_PATTERN,
            "target_types": [t for t, _ in tg["types"]],
            "n_typed": tg["n_typed"],
            "n_dropped_no_synapse_sites": int(len(tg["dropped_no_sites"])),
            "n_targets": int(len(self.target_idx)),
            "n_left": int(len(self.left)), "n_right": int(len(self.right)),
            "n_no_side": int(len(tg["no_side"])),
            "side_source": "Graph.side (somaSide null for JO -> instance suffix _L/_R == rootSide)",
            "stimulus": self.stimulus.describe(),
        }
        if self.table_np is not None:
            d.update(dt_ms=self.dt_ms, n_steps=self.n_steps,
                     table_max=float(self.table_np.max()), drive_max=float(self._drive.max()))
        return d
