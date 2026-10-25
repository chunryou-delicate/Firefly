"""M7c2a: choose the giant-fibre read-out from circuit measurements only — no game is played.

Pre-registration: docs/m7c2-brief.md, stage 1 ("1단계 — 해독기 보정"), written before this ran.

In signal-processing terms: measure the noise floor first, then set the detection threshold.

  * candidates  ``CountWindowDecoder(W, n)`` for W in {50, 100, 200, 300} ms x n in {1, 2, 3}
                (brief 1.1; M7c's decoder is (50, 1)). Nothing outside this grid.
  * operating point P2, unchanged: conductance synapses, g 5.36539e-4, noise_sigma 29.13,
    adapt_g_b 8.0, dt 1 ms; seeds 0-9.
  * noise record     5 s warm-up, then 30 s with no input (one per seed).
  * stimulus record  M7a2's protocol exactly: 5 s warm-up, 200 ms nothing, 300 ms exponential
                     ramp (tau 50 ms) to ``a_in``, cut, 300 ms observation; left / right / both;
                     a_in in {640, 1280, 2560, 5120} (every gain M7a2 passed at steady state).
  * tick time 0 = end of the warm-up; tick ``k`` ends at ``floor((k+1)*1000/60)`` ms, the game
    schedule (``flysim.apps.games.brain.tick_end_ms``). Each record is simulated once and all
    12 decoders are evaluated on its giant-fibre spikes — the decoder feeds nothing back.
  * hit rate H      the first action decided in the 600 ms from ramp onset is the right one
                    (left -> RIGHT, right -> LEFT, both -> JUMP); no action = miss. 120 trials.
  * false alarms F600  fraction of the 600 ms windows starting at a tick end in the noise
                    record (windows past the end of the record dropped) in which the decoder
                    acts at least once.
  * choice          max J = H - F600 (Youden); ties -> smaller W, then smaller n. If the best
                    J <= 0, no decoder in the grid beats the noise: report, no stage 2.

Two readings the brief leaves implicit, fixed here before the run (both recorded in the output,
and the selection is also recomputed under the other reading as a descriptive check):

  1. "In the 600 ms from x" means the decisions at tick ends ``e`` with ``x < e <= x + 600``.
     A decision at exactly ``x`` uses only spikes before ``x`` (the window is ``[e - W, e)``),
     so it cannot be a response to a stimulus that starts at ``x``. For H, x = ramp onset
     (tick time 200, which is the end of tick 11) - ticks 12..47, the last one ending at 800,
     the end of the record. The same rule defines the F600 windows, so H and F600 are in the
     same unit.
  2. The decoder sees the warm-up's spikes when its window reaches back past tick time 0
     (only W = 300 does, for decisions before tick time 300). In a game, an approach arrives
     on a settled background with a full window behind it; hiding the warm-up would make the
     longest window look cleaner here than it can be in play.

Usage: python -m flysim.apps.m7c2_decoder_calibration
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from fractions import Fraction

import numpy as np
import torch

from ..data.download import ROOT
from ..engine import EngineParams, LIFEngine
from ..graph import Graph
from ..probe.motor import DEFAULT_ACTION, CountWindowDecoder
from . import m7a2_steady_probe as m7a2
from . import m7a_escape_probe as m7a

# ---- fixed before the run (docs/m7c2-brief.md 1.1-1.3) ----------------------------------
DT_MS = 1.0
P2 = dict(synapse="conductance", g=5.36539e-4, noise_sigma=29.13, adapt_g_b=8.0, dt=DT_MS)
SEEDS = tuple(range(10))
WARMUP_MS = m7a2.WARMUP_MS                       # 5000
NOISE_MS = 30000.0
STIM_MS = m7a.TOTAL_MS                           # 800: 200 nothing + 300 ramp + 300 observation
RAMP_ONSET_MS = m7a.PRE_MS                       # 200, in tick time
WINDOW_MS = 600.0                                # H and F600: ramp 300 + observation 300
A_IN = (640.0, 1280.0, 2560.0, 5120.0)
CONDITIONS = ("left", "right", "both")
CORRECT = {"left": "RIGHT", "right": "LEFT", "both": "JUMP"}
GRID_W_MS = (50.0, 100.0, 200.0, 300.0)
GRID_N = (1, 2, 3)
M7C_DECODER = (50.0, 1)
TICK_HZ = 60

RESULTS_JSON = ROOT / "data-provenance" / "m7c2-decoder.json"
M7A2_JSON = ROOT / "data-provenance" / "m7a2-results.json"


def log(msg: str) -> None:
    print(f"[m7c2a] {msg}", flush=True)


def params() -> EngineParams:
    return EngineParams(**P2)


def candidates() -> list[tuple[float, int]]:
    """The 12 decoders, in tie-break order (smaller W first, then smaller n)."""
    return [(w, n) for w in GRID_W_MS for n in GRID_N]


def tick_end_ms(k: int) -> int:
    """End of tick ``k`` in tick time: floor((k+1)*1000/60) - the game schedule."""
    return ((k + 1) * 1000) // TICK_HZ


def tick_ends(record_ms: float) -> np.ndarray:
    """Ends of every whole tick inside a record of ``record_ms`` (tick time)."""
    k = np.arange(int(record_ms * TICK_HZ // 1000) + 1)
    e = ((k + 1) * 1000) // TICK_HZ
    return e[e <= record_ms].astype(np.int64)


# ---------------------------------------------------------------------------------------
# decoding a record (pure)
# ---------------------------------------------------------------------------------------
def events_from(gf: dict[str, list[int]]) -> list[tuple[int, str]]:
    """{"L": [t...], "R": [t...]} -> time-sorted (t, side) pairs (L before R at equal t)."""
    ev = [(int(t), "L") for t in gf["L"]] + [(int(t), "R") for t in gf["R"]]
    return sorted(ev, key=lambda x: (x[0], x[1]))


def decide(events: list[tuple[int, str]], ends: np.ndarray, decoder: CountWindowDecoder,
           hide_before: float | None = None) -> list[str]:
    """The decoder's action at every tick end. Only the spikes that can fall in the decoder's
    window are handed over (``CountWindowDecoder`` applies the window itself); with
    ``hide_before`` the spikes before that time are invisible (sensitivity reading 2)."""
    if hide_before is not None:
        events = [e for e in events if e[0] >= hide_before]
    times = np.array([e[0] for e in events], dtype=np.float64)
    out = []
    for e in ends:
        lo = int(np.searchsorted(times, e - decoder.window_ms, side="left"))
        hi = int(np.searchsorted(times, e, side="left"))
        out.append(decoder.decode(events[lo:hi], float(e)))
    return out


def in_window(ends: np.ndarray, start: float, inclusive_start: bool = False) -> np.ndarray:
    """Tick indices whose decisions count for the 600 ms window starting at ``start``:
    ``start < e <= start + 600`` (reading 1), or ``start <= e < start + 600`` (the other)."""
    if inclusive_start:
        return np.flatnonzero((ends >= start) & (ends < start + WINDOW_MS))
    return np.flatnonzero((ends > start) & (ends <= start + WINDOW_MS))


def trial_outcome(actions: list[str], ends: np.ndarray, condition: str,
                  inclusive_start: bool = False) -> dict:
    """First action in the 600 ms from ramp onset; a hit if it is the correct one."""
    for i in in_window(ends, RAMP_ONSET_MS, inclusive_start):
        if actions[i] != DEFAULT_ACTION:
            return dict(first_action=actions[i], hit=actions[i] == CORRECT[condition],
                        latency_ms=float(ends[i] - RAMP_ONSET_MS))
    return dict(first_action=None, hit=False, latency_ms=None)


def false_alarm_windows(actions: list[str], ends: np.ndarray, record_ms: float,
                        inclusive_start: bool = False) -> tuple[int, int]:
    """(windows with at least one action, windows) over the 600 ms windows that start at a
    tick end and do not run past the end of the record."""
    act = np.array([a != DEFAULT_ACTION for a in actions], dtype=np.int64)
    csum = np.r_[0, np.cumsum(act)]
    hit = tot = 0
    for e0 in ends:
        if e0 + WINDOW_MS > record_ms:
            break
        idx = in_window(ends, float(e0), inclusive_start)
        tot += 1
        hit += bool(csum[idx[-1] + 1] - csum[idx[0]] > 0) if len(idx) else False
    return hit, tot


def score(stim: list[dict], noise: list[dict], w: float, n: int,
          inclusive_start: bool = False, hide_warmup: bool = False) -> dict:
    """H, F600 and J (exact fractions) for one candidate, plus the descriptive breakdowns."""
    dec = CountWindowDecoder(w, n)
    hide = 0.0 if hide_warmup else None
    s_ends, n_ends = tick_ends(STIM_MS), tick_ends(NOISE_MS)
    trials = []
    for r in stim:
        acts = decide(events_from(r["gf"]), s_ends, dec, hide)
        trials.append(dict(condition=r["condition"], a_in=r["a_in"], seed=r["seed"],
                           **trial_outcome(acts, s_ends, r["condition"], inclusive_start)))
    fa_hit = fa_tot = on = ticks = 0
    for r in noise:
        acts = decide(events_from(r["gf"]), n_ends, dec, hide)
        h, t = false_alarm_windows(acts, n_ends, NOISE_MS, inclusive_start)
        fa_hit, fa_tot = fa_hit + h, fa_tot + t
        on += sum(a != DEFAULT_ACTION for a in acts)
        ticks += len(acts)
    hits = sum(t["hit"] for t in trials)
    H, F = Fraction(hits, len(trials)), Fraction(fa_hit, fa_tot)
    lat_hit = [t["latency_ms"] for t in trials if t["hit"]]
    lat_any = [t["latency_ms"] for t in trials if t["latency_ms"] is not None]

    def h_of(sel):
        sub = [t for t in trials if sel(t)]
        return sum(t["hit"] for t in sub) / len(sub) if sub else None

    first = {c: {a: sum(1 for t in trials if t["condition"] == c and t["first_action"] == a)
                 for a in ("JUMP", "LEFT", "RIGHT", None)} for c in CONDITIONS}
    return dict(window_ms=w, min_spikes=n, hits=hits, trials=len(trials),
                false_alarm_windows=fa_hit, windows=fa_tot,
                H=float(H), F600=float(F), J=float(H - F), _J=H - F,
                H_by_a_in={str(a): h_of(lambda t, a=a: t["a_in"] == a) for a in A_IN},
                H_by_condition={c: h_of(lambda t, c=c: t["condition"] == c) for c in CONDITIONS},
                first_action_by_condition={c: {str(k): v for k, v in d.items()} for c, d in first.items()},
                latency_ms_median_hits=(float(np.median(lat_hit)) if lat_hit else None),
                latency_ms_median_first_action=(float(np.median(lat_any)) if lat_any else None),
                noise_action_on_fraction=on / ticks)


def select(rows: list[dict]) -> dict:
    """Max J; ties -> smaller W, then smaller n (exact comparison on fractions)."""
    best = max(Fraction(r["J_exact"]) for r in rows)
    tied = [r for r in rows if Fraction(r["J_exact"]) == best]
    tied.sort(key=lambda r: (r["window_ms"], r["min_spikes"]))
    w, n = tied[0]["window_ms"], tied[0]["min_spikes"]
    return dict(window_ms=w, min_spikes=n, J=float(best), J_exact=str(best),
                tied_at_max=[[r["window_ms"], r["min_spikes"]] for r in tied],
                beats_noise=bool(best > 0),
                stage2=("run with CountWindowDecoder(%g, %d)" % (w, n)) if best > 0 else
                       "do not run: no decoder in the grid beats the noise (best J <= 0)")


def table(stim: list[dict], noise: list[dict], **reading) -> list[dict]:
    rows = [score(stim, noise, w, n, **reading) for (w, n) in candidates()]
    for r in rows:
        r["J_exact"] = str(r.pop("_J"))
    return rows


# ---------------------------------------------------------------------------------------
# records (GPU)
# ---------------------------------------------------------------------------------------
def _gf(t_ms: np.ndarray, idx: np.ndarray, sets: dict) -> dict[str, list[int]]:
    """Giant-fibre spike times in tick time (warm-up spikes are negative)."""
    out = {}
    for s in ("L", "R"):
        m = idx == int(sets[f"DNp01_{s}"][0])
        out[s] = (t_ms[m] - WARMUP_MS).astype(np.int64).tolist()
    return out


def _activity(t_ms: np.ndarray, idx: np.ndarray, n: int, start: float, stop: float,
              win_ms: float = 200.0) -> dict:
    """Network rate and active fraction per 200 ms window after the warm-up (runaway check:
    rate >= 5 Hz or active >= 90 % in any window, M7a/M7c thresholds)."""
    rates, act = [], []
    for a in np.arange(start, stop, win_ms):
        m = (t_ms >= a) & (t_ms < a + win_ms)
        rates.append(float(m.sum()) / (n * win_ms / 1000.0))
        act.append(int(np.unique(idx[m]).size) / n)
    return dict(window_rate_hz_max=max(rates), window_active_max=max(act),
                mean_rate_hz=float(np.mean(rates)),
                runaway=bool(max(rates) >= m7a.RUNAWAY_RATE_HZ or max(act) >= m7a.FAILURE_ACTIVE_FRACTION))


def noise_record(engine: LIFEngine, sets: dict, seed: int) -> dict:
    engine.reset(seed)
    t, idx = engine.run(int(round((WARMUP_MS + NOISE_MS) / DT_MS)), None, record=True)
    t_ms, idx = t.astype(np.float64) * DT_MS, idx.astype(np.int64)
    r = dict(seed=seed, gf=_gf(t_ms, idx, sets),
             activity=_activity(t_ms, idx, engine.n, WARMUP_MS, WARMUP_MS + NOISE_MS))
    # its first 800 ms after the warm-up is M7a2's "none" run for this seed
    r["m7a2_check"] = {f"DNp01_{s}_stim": m7a._count(t_ms, idx, int(sets[f"DNp01_{s}"][0]),
                                                     m7a2.STIM_WINDOW_MS)[0] for s in ("L", "R")}
    return r


def stim_record(engine: LIFEngine, sets: dict, condition: str, a_in: float, seed: int) -> dict:
    """M7a2's ``run`` input, unchanged, with the giant-fibre spike times kept."""
    amp = m7a2.stimulus(a_in)
    targets = torch.as_tensor(m7a.stimulated(sets, condition), device=engine.device)
    prev = [0.0]

    def fn(k: int, buf: torch.Tensor):
        if k == 0:
            buf.zero_()
            prev[0] = 0.0
        v = float(amp[k])
        if v != prev[0] and targets.numel():
            buf[targets] = v
            prev[0] = v

    engine.reset(seed)
    t, idx = engine.run(int(round(m7a2.TOTAL_MS / DT_MS)), fn, record=True)
    t_ms, idx = t.astype(np.float64) * DT_MS, idx.astype(np.int64)
    r = dict(condition=condition, a_in=a_in, seed=seed, gf=_gf(t_ms, idx, sets),
             activity=_activity(t_ms, idx, engine.n, WARMUP_MS, m7a2.TOTAL_MS))
    chk = {}
    for s in ("L", "R"):
        c, f = m7a._count(t_ms, idx, int(sets[f"DNp01_{s}"][0]), m7a2.STIM_WINDOW_MS)
        chk[f"DNp01_{s}_stim"], chk[f"DNp01_{s}_first_ms"] = c, f
    r["m7a2_check"] = chk
    return r


def m7a2_reproduction(stim: list[dict], noise: list[dict]) -> dict:
    """Do these records reproduce M7a2's numbers for the same (condition, a_in, seed)?"""
    if not M7A2_JSON.exists():
        return dict(checked=False, reason="m7a2-results.json not present")
    A = json.loads(M7A2_JSON.read_text())["arms"]["29.13"]
    ref = {(r["condition"], float(r["a_in"]), r["seed"]): r
           for p in A["sweep"] for c in CONDITIONS for r in p["runs"][c]}
    none = {r["seed"]: r for r in A["none"]}
    bad = []
    for r in stim:
        q = ref[(r["condition"], float(r["a_in"]), r["seed"])]
        for k, v in r["m7a2_check"].items():
            if q[k] != v:
                bad.append(dict(record=[r["condition"], r["a_in"], r["seed"]], field=k, here=v, m7a2=q[k]))
    for r in noise:
        for k, v in r["m7a2_check"].items():
            if none[r["seed"]][k] != v:
                bad.append(dict(record=["none", 0.0, r["seed"]], field=k, here=v, m7a2=none[r["seed"]][k]))
    return dict(checked=True, compared_stimulus_records=len(stim), compared_noise_records=len(noise),
                identical=not bad, mismatches=bad)


def git_commit() -> dict:
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                               capture_output=True, text=True, check=True).stdout.split("\n")
        return dict(head=head, modified_tracked_files=[l[3:] for l in dirty if l.strip()])
    except Exception as exc:                          # pragma: no cover - recorded, not fatal
        return dict(head=None, error=str(exc))


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    if not set(A_IN) <= set(m7a.A_IN_GRID):
        raise ValueError("a_in must come from M7a's gain grid")

    graph = Graph.load()
    sets = m7a.build_sets(graph)
    engine = LIFEngine(graph, params(), device=a.device)
    if engine.device.type != "cuda":
        log("WARNING: running on CPU - slow")
    sim_s = len(SEEDS) * ((WARMUP_MS + NOISE_MS) + len(CONDITIONS) * len(A_IN) * m7a2.TOTAL_MS) / 1000
    log(f"P2 {P2}; seeds {SEEDS[0]}-{SEEDS[-1]}; {sim_s:,.0f} s of simulated time")
    t0 = time.time()
    noise = []
    for s in SEEDS:
        noise.append(noise_record(engine, sets, s))
        g = noise[-1]["gf"]
        log(f"  noise seed {s}: GF L/R {sum(t >= 0 for t in g['L'])}/{sum(t >= 0 for t in g['R'])} "
            f"spikes in 30 s; max window rate {noise[-1]['activity']['window_rate_hz_max']:.3f} Hz")
    stim = []
    for a_in in A_IN:
        for c in CONDITIONS:
            for s in SEEDS:
                stim.append(stim_record(engine, sets, c, a_in, s))
        log(f"  stimulus a_in={a_in:g} done ({time.time() - t0:.0f} s)")
    wall = round(time.time() - t0, 1)

    rows = table(stim, noise)
    sel = select(rows)
    for r in rows:
        log(f"  W={r['window_ms']:>4g} n={r['min_spikes']}  H={r['H']:.3f}  F600={r['F600']:.3f}  "
            f"J={r['J']:+.3f}")
    log(f"-> selected W={sel['window_ms']:g} n={sel['min_spikes']}  J={sel['J']:+.3f}  "
        f"({sel['stage2']})")

    other = {}
    for name, kw in (("decision_at_window_start_counts", dict(inclusive_start=True)),
                     ("warmup_spikes_hidden", dict(hide_warmup=True))):
        rr = table(stim, noise, **kw)
        other[name] = dict(selection=select(rr),
                           J=[[r["window_ms"], r["min_spikes"], r["J"]] for r in rr])
        log(f"  sensitivity {name}: W={other[name]['selection']['window_ms']:g} "
            f"n={other[name]['selection']['min_spikes']}")

    repro = m7a2_reproduction(stim, noise)
    log(f"  M7a2 reproduction: {repro.get('identical')}")
    runaway = [r for r in noise + stim if r["activity"]["runaway"]]

    R = dict(generated_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
             brief="docs/m7c2-brief.md, stage 1", git=git_commit(),
             operating_point=dict(name="P2", params=params().to_dict()),
             seeds=list(SEEDS),
             protocol=dict(
                 warmup_ms=WARMUP_MS, noise_ms=NOISE_MS, stim_ms=STIM_MS,
                 ramp=dict(pre_ms=m7a.PRE_MS, ramp_ms=m7a.RAMP_MS, tau_ms=m7a.RAMP_TAU_MS),
                 a_in=list(A_IN), conditions=list(CONDITIONS), correct_action=CORRECT,
                 input="every LC4 + LPLC2 cell of the stimulated side gets the ramp (M7a2)",
                 tick_time_origin="end of the 5 s warm-up (ramp onset at tick time 200)",
                 tick_end="floor((k+1)*1000/60) ms",
                 decoder_window="[tick end - W, tick end)",
                 H="first action decided at a tick end e with 200 < e <= 800 is the correct one",
                 F600="windows starting at a tick end e0 (e0 + 600 <= 30000) with any action at "
                      "a tick end e, e0 < e <= e0 + 600",
                 warmup_spikes="visible to the decoder when its window reaches back past tick time 0",
                 selection="max J = H - F600; ties -> smaller W, then smaller n"),
             grid=dict(window_ms=list(GRID_W_MS), min_spikes=list(GRID_N),
                       m7c_decoder=list(M7C_DECODER)),
             table=rows, selection=sel,
             sensitivity_descriptive=other,
             m7a2_reproduction=repro,
             runaway_records=[[r.get("condition", "none"), r.get("a_in", 0.0), r["seed"]] for r in runaway],
             records=dict(noise=noise, stimulus=stim),
             wall_simulation_s=wall)
    RESULTS_JSON.write_text(json.dumps(R, indent=1, default=m7a._json_default) + "\n")
    log(f"done: simulation {wall} s -> {RESULTS_JSON.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
