"""M5a: the flysim-live wire protocol as code (contract v1.2).

``docs/m5-protocol.md`` is the contract; every message type, field and bound below comes
from it. Nothing here may be changed without changing the contract first (and the contract
is owned by the planning session).

Two rejection paths, so a client can tell a bug from a refusal:

* ``ProtocolError`` -> a ``error`` message: the frame is not a valid control message at all
  (not an object, unknown ``type``, missing/!int ``req_id``, undecodable JSON).
* ``Reject``        -> an ``ack`` with ``ok: false`` and a reason: the message is well formed
  but the server refuses it (value out of range, unknown set name, adaptation before M2c...).
"""
from __future__ import annotations

import dataclasses
import math
import re
from typing import Any

from ..engine.params import SYNAPSE_MODELS, EngineParams

# ---- contract constants ---------------------------------------------------------------
SERVER_MESSAGES = ("hello", "frame", "ack", "params", "hops", "snapshot_done", "warning", "error")
CONTRACT_VERSION = "1.2"
CLIENT_MESSAGES = ("set_params", "stimulus", "audio_start", "audio_stop", "inject",
                   "pause", "resume", "step", "reset", "get_hops", "snapshot", "set_frame_every")
STIMULUS_KINDS = ("silence", "click_train", "tone", "audio")
WARNING_CODES = ("ignited", "all_silent", "runaway", "recorder_overflow")
MODES = ("rate", "phase-lock")
DT_CHOICES = (1.0, 0.1)
STATUS_VALUES = ("running", "paused", "lagging")   # all three are contract values since v1.1
MAX_HOPS = 4
HOP_LAYER_CAP = 5_000
MAX_SNAPSHOT_SECONDS = 10.0
SPIKE_CAP_PER_BIN = 1_500
SET_INDEX_CAP = 2_000          # hello.sets carries indices only for sets this size or smaller
MAX_FRAME_EVERY = 1_000
MAX_INJECT_TARGETS = 100_000
MAX_AUDIO_SAMPLE_RATE = 192_000
MIN_AUDIO_SAMPLE_RATE = 1_000
RUN_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")   # also a path component: no separators

# ``adapt_b``/``adapt_tau_w`` exist only once M2c lands. Until then a non-zero value is
# refused with ``ack ok:false`` (contract: "adapt_*는 M2c 전에는 0 고정").
ENGINE_HAS_ADAPT = "adapt_b" in {f.name for f in dataclasses.fields(EngineParams)}

PARAM_NAMES = ("synapse", "g", "a_in", "noise_sigma", "adapt_b", "adapt_tau_w",
               "dt_ms", "speed", "mode")
# Params that force an engine reset when they change (contract: dt_ms / synapse).
RESET_PARAMS = ("dt_ms", "synapse")


class ProtocolError(Exception):
    """Malformed message -> server replies with an ``error``."""

    def __init__(self, msg: str, req_id: int | None = None):
        super().__init__(msg)
        self.msg = msg
        self.req_id = req_id


class Reject(Exception):
    """Well-formed but refused -> server replies with ``ack ok:false``."""

    def __init__(self, msg: str):
        super().__init__(msg)
        self.msg = msg


# ---- field helpers --------------------------------------------------------------------
def _number(d: dict, key: str, *, lo: float | None = None, hi: float | None = None,
            default: float | None = None) -> float:
    if key not in d or d[key] is None:
        if default is None:
            raise Reject(f"{key} is required")
        return float(default)
    v = d[key]
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise Reject(f"{key} must be a number, got {type(v).__name__}")
    v = float(v)
    if not math.isfinite(v):
        raise Reject(f"{key} must be finite")
    if lo is not None and v < lo:
        raise Reject(f"{key} must be >= {lo}")
    if hi is not None and v > hi:
        raise Reject(f"{key} must be <= {hi}")
    return v


def _integer(d: dict, key: str, *, lo: int | None = None, hi: int | None = None,
             default: int | None = None) -> int:
    if key not in d or d[key] is None:
        if default is None:
            raise Reject(f"{key} is required")
        return int(default)
    v = d[key]
    if isinstance(v, bool) or not isinstance(v, int):
        raise Reject(f"{key} must be an integer")
    if lo is not None and v < lo:
        raise Reject(f"{key} must be >= {lo}")
    if hi is not None and v > hi:
        raise Reject(f"{key} must be <= {hi}")
    return int(v)


def _choice(d: dict, key: str, options: tuple, *, default: Any = None) -> Any:
    if key not in d or d[key] is None:
        if default is None:
            raise Reject(f"{key} is required")
        return default
    v = d[key]
    if v not in options:
        raise Reject(f"{key} must be one of {list(options)}, got {v!r}")
    return v


# ---- params ---------------------------------------------------------------------------
def validate_params(p: Any) -> dict:
    """Validate a (partial) ``params`` object. Returns only the keys that were present."""
    if not isinstance(p, dict):
        raise Reject("params must be an object")
    unknown = set(p) - set(PARAM_NAMES)
    if unknown:
        raise Reject(f"unknown params: {sorted(unknown)}; allowed: {list(PARAM_NAMES)}")
    out: dict = {}
    if "synapse" in p:
        out["synapse"] = _choice(p, "synapse", tuple(SYNAPSE_MODELS))
    if "mode" in p:
        out["mode"] = _choice(p, "mode", MODES)
    if "dt_ms" in p:
        v = _number(p, "dt_ms")
        if v not in DT_CHOICES:
            raise Reject(f"dt_ms must be one of {list(DT_CHOICES)}")
        out["dt_ms"] = v
    if "g" in p:
        out["g"] = _number(p, "g", lo=0.0)
    if "a_in" in p:
        out["a_in"] = _number(p, "a_in", lo=0.0)
    if "noise_sigma" in p:
        out["noise_sigma"] = _number(p, "noise_sigma", lo=0.0)
    if "speed" in p:
        out["speed"] = _number(p, "speed", lo=1e-3, hi=1e3)
    for key in ("adapt_b", "adapt_tau_w"):
        if key in p:
            v = _number(p, key, lo=0.0)
            if not ENGINE_HAS_ADAPT and v != 0.0:
                raise Reject(f"{key}={v:g} refused: the engine has no adaptation current yet "
                             "(M2c not merged); only 0 is accepted")
            out[key] = v
    if "adapt_tau_w" in out and out["adapt_tau_w"] == 0.0 and ENGINE_HAS_ADAPT:
        raise Reject("adapt_tau_w must be > 0")
    return out


# ---- client -> server -----------------------------------------------------------------
def validate_client_message(obj: Any) -> tuple[str, dict]:
    """Returns ``(type, normalised_message)``. Raises ProtocolError / Reject."""
    if not isinstance(obj, dict):
        raise ProtocolError("message must be a JSON object")
    kind = obj.get("type")
    if not isinstance(kind, str) or kind not in CLIENT_MESSAGES:
        raise ProtocolError(f"unknown message type {kind!r}; expected one of {list(CLIENT_MESSAGES)}",
                            obj.get("req_id") if isinstance(obj.get("req_id"), int) else None)
    req_id = obj.get("req_id")
    if isinstance(req_id, bool) or not isinstance(req_id, int):
        raise ProtocolError(f"{kind}: req_id (int) is required")
    m: dict = {"type": kind, "req_id": req_id}

    if kind == "set_params":
        m["params"] = validate_params(obj.get("params"))
        if not m["params"]:
            raise Reject("set_params with no parameters")
    elif kind == "stimulus":
        kind_s = _choice(obj, "kind", STIMULUS_KINDS)
        m["kind"] = kind_s
        m["duration_ms"] = _number(obj, "duration_ms", lo=0.0, default=0.0)
        m["ild_db"] = _number(obj, "ild_db", lo=-60.0, hi=60.0, default=0.0)
        if kind_s == "click_train":
            m["ipi_ms"] = _number(obj, "ipi_ms", lo=0.1, hi=10_000.0, default=35.0)
            m["pulse_ms"] = _number(obj, "pulse_ms", lo=0.1, hi=10_000.0, default=10.0)
            m["carrier_hz"] = _number(obj, "carrier_hz", lo=1.0, hi=4_000.0, default=200.0)
            if m["pulse_ms"] > m["ipi_ms"]:
                raise Reject("pulse_ms must not exceed ipi_ms (pulses would overlap)")
        elif kind_s == "tone":
            m["carrier_hz"] = _number(obj, "carrier_hz", lo=1.0, hi=4_000.0, default=200.0)
    elif kind == "audio_start":
        m["sample_rate"] = _integer(obj, "sample_rate", lo=MIN_AUDIO_SAMPLE_RATE, hi=MAX_AUDIO_SAMPLE_RATE)
        m["channels"] = _integer(obj, "channels", lo=1, hi=1, default=1)
    elif kind == "audio_stop":
        pass
    elif kind == "inject":
        t = obj.get("targets")
        if not isinstance(t, dict) or ("set" in t) == ("idx" in t):
            raise Reject('targets must be {"set": name} or {"idx": [..]} (exactly one)')
        if "set" in t:
            if not isinstance(t["set"], str):
                raise Reject("targets.set must be a string")
            m["targets"] = {"set": t["set"]}
        else:
            idx = t["idx"]
            if not isinstance(idx, list) or not idx:
                raise Reject("targets.idx must be a non-empty list")
            if len(idx) > MAX_INJECT_TARGETS:
                raise Reject(f"targets.idx is limited to {MAX_INJECT_TARGETS} entries")
            if any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in idx):
                raise Reject("targets.idx must be non-negative integers")
            m["targets"] = {"idx": list(idx)}
        m["amp"] = _number(obj, "amp")
        m["duration_ms"] = _number(obj, "duration_ms", lo=0.0, hi=600_000.0)
    elif kind in ("pause", "resume"):
        pass
    elif kind == "step":
        m["n"] = _integer(obj, "n", lo=1, hi=1_000_000, default=1)
    elif kind == "reset":
        m["seed"] = _integer(obj, "seed", lo=0, hi=2**31 - 1, default=0)
    elif kind == "get_hops":
        m["src"] = _integer(obj, "src", lo=0)
        m["k"] = _integer(obj, "k", lo=1, hi=MAX_HOPS)
        m["min_contacts"] = _integer(obj, "min_contacts", lo=1, default=1)
    elif kind == "snapshot":
        run_id = obj.get("run_id")
        if not isinstance(run_id, str) or not RUN_ID_RE.match(run_id):
            raise Reject("run_id must match [A-Za-z0-9._-]{1,64} (it is also a directory name)")
        m["run_id"] = run_id
        m["seconds"] = _number(obj, "seconds", lo=0.001, hi=MAX_SNAPSHOT_SECONDS)
    elif kind == "set_frame_every":
        m["n"] = _integer(obj, "n", lo=1, hi=MAX_FRAME_EVERY)
    return kind, m


# ---- server -> client builders ---------------------------------------------------------
def validate_assets(assets: Any) -> dict:
    """``hello.assets`` (contract v1.2): which 3D assets this server can actually serve."""
    if not isinstance(assets, dict):
        raise ValueError("assets must be an object")
    sets = assets.get("skeleton_sets", [])
    if not isinstance(sets, list) or any(not isinstance(s, str) for s in sets):
        raise ValueError("assets.skeleton_sets must be a list of names")
    return {"neurons_3d": bool(assets.get("neurons_3d", False)), "skeleton_sets": list(sets)}


def hello_msg(*, n_neurons: int, regions: list[str], sets: dict, set_sizes: dict, params: dict,
              dt_ms: float, bin_ms: float, engine: dict, neurons_url: str, session_id: str,
              params_version: int, assets: dict | None = None) -> dict:
    return {"type": "hello", "n_neurons": int(n_neurons), "regions": list(regions), "sets": sets,
            "set_sizes": set_sizes, "params": params, "dt_ms": float(dt_ms), "bin_ms": float(bin_ms),
            "engine": engine, "neurons_url": neurons_url, "session_id": session_id,
            "params_version": int(params_version),
            "assets": validate_assets(assets or {})}


def frame_msg(*, t_ms: float, step: int, rates: list[float], idx: list[int], n_total: int,
              sample_ratio: float, active_frac_100ms: float, input_level: float, status: str,
              speed: float, params_version: int, n_bins: int) -> dict:
    if status not in STATUS_VALUES:
        raise ValueError(f"status must be one of {STATUS_VALUES}")
    return {"type": "frame", "t_ms": round(float(t_ms), 4), "step": int(step), "n_bins": int(n_bins),
            "rates": rates,
            "spikes": {"idx": idx, "n_total": int(n_total), "sample_ratio": round(float(sample_ratio), 6)},
            "active_frac_100ms": round(float(active_frac_100ms), 6),
            "input_level": round(float(input_level), 6), "status": status,
            "speed": float(speed), "params_version": int(params_version)}


def params_msg(params: dict, params_version: int) -> dict:
    """Broadcast whenever ``params_version`` changes: an ack only reaches the requester, so this
    is the only way another client learns what was changed (contract v1.1)."""
    return {"type": "params", "params": params, "params_version": int(params_version)}


def ack_msg(req_id: int | None, ok: bool, msg: str, params_version: int) -> dict:
    return {"type": "ack", "req_id": req_id, "ok": bool(ok), "msg": msg,
            "params_version": int(params_version)}


def hops_msg(req_id: int, src: int, k: int, layers: list[list[int]], truncated: list[bool]) -> dict:
    return {"type": "hops", "req_id": req_id, "src": int(src), "k": int(k),
            "layers": layers, "truncated": truncated}


def snapshot_done_msg(req_id: int, run_id: str, path: str, n_frames: int) -> dict:
    return {"type": "snapshot_done", "req_id": req_id, "run_id": run_id, "path": path,
            "n_frames": int(n_frames)}


def warning_msg(code: str, msg: str) -> dict:
    base = code[:-8] if code.endswith("_cleared") else code
    if base not in WARNING_CODES:
        raise ValueError(f"unknown warning code {code!r}")
    return {"type": "warning", "code": code, "msg": msg}


def error_msg(msg: str, req_id: int | None = None) -> dict:
    return {"type": "error", "req_id": req_id, "msg": msg}
