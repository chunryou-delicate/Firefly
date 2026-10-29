"""FAKE data for flysim-show.html tests (M7e3 brief §4: fake data lives only under tests/).

**Not results.** These files only have the shape of the contracts in docs/m7e3-brief.md (2.1
``show-data.json``, 3.1 ``brain-map.json``) so the show page can be built and tested before the
real files arrive. What is real and what is invented:

* real: the games (``base.make``), their frames, the looming detector, the random bot, the
  one-way bot, the threshold rule. ``eye_rect`` is computed with ``draw_block``'s formula and
  checked here against the real renderer, and ``check_frames`` are real rendered frames, so the
  page's eye drawing is tested against Python's ``fill_rect``.
* invented: every brain record (GF / TTMn spike times, set rates, network spike counts), the
  fly's actions (decoded from those invented spikes with the real decoder rules), the threshold
  bot's θ, all scores and verdicts, and the whole brain map (a cartoon shape, not coordinates).
  Bot episodes are cut at ``FAKE_MAX_TICKS`` to keep the files small.

``meta.fixture`` says so, and the page shows a "가짜 데이터" banner when it is present.
"""
from __future__ import annotations

import base64
import json
import math
from pathlib import Path

import numpy as np

from flysim.apps.games import base, dodge, jump
from flysim.apps.games.agents import RandomAgent, ThresholdAgent
from flysim.sensory.looming import LoomingDetector

FAKE_MAX_TICKS = 900
FIXTURE_NOTE = "가짜 데이터 — 테스트 전용. 계약 모양만 맞춘 것이고 결과가 아니다 (tests/show_fake.py)"
GAMES = {"jump": "뛰어넘기", "dodge": "피하기", "dodge_v2": "벽 없는 피하기"}
FAKE_THETA = {"jump": 4e-4, "dodge": 1.5e-3, "dodge_v2": 2.5e-4}
SETS = ["LC4_L", "LC4_R", "LPLC2_L", "LPLC2_R", "DNp01_L", "DNp01_R", "TTMn_L", "TTMn_R"]


def t_ms(k: int) -> int:
    return (k * 1000) // 60


def eye_rect(env) -> list[float] | None:
    """draw_block's screen rectangle for the object in the current state (None = no object)."""
    if env.dist is None or env.dist <= 0:
        return None
    if env.name == "jump":
        lat0, lat1, h = -jump.OBS_HALF_WIDTH, jump.OBS_HALF_WIDTH, jump.OBS_HEIGHT
    else:
        dx = env.obj_x - env.x
        lat0, lat1, h = dx - dodge.OBJ_HALF_WIDTH, dx + dodge.OBJ_HALF_WIDTH, dodge.OBJ_HEIGHT
    s = base.FOCAL_PX / env.dist
    cx = base.FRAME_W / 2.0
    return [cx + s * lat0, cx + s * lat1, base.HORIZON_ROW + s * (base.CAMERA_HEIGHT - h),
            base.HORIZON_ROW + s * base.CAMERA_HEIGHT]


def redraw(rect) -> np.ndarray:
    img = base.background()
    if rect is not None:
        base.fill_rect(img, *rect, base.OBJECT)
    return base.to_frame(img)


def _state(env) -> dict:
    if env.name == "jump":
        return {"x": None, "obj_x": None, "dist": env.dist, "height": env.height,
                "airborne": env.airborne, "speed": env.speed, "score": env.score}
    return {"x": env.x, "obj_x": env.obj_x, "dist": env.dist, "height": None, "airborne": None,
            "speed": env.speed, "score": env.score}


class FakeGF:
    """Invented GF spike train: ~2 Hz per side with a 210 ms floor between spikes, plus extra
    spikes on the side an object is coming from during its last ~250 ms. Not a brain."""

    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.times = {"L": [], "R": []}
        self.ttmn = {"L": [], "R": []}

    def tick(self, k: int, side_hint: str | None, near: bool) -> None:
        lo, hi = t_ms(k), t_ms(k + 1)
        for side in "LR":
            p = 2.0 * (hi - lo) / 1000.0
            if near and side_hint in (side, "center"):
                p = 0.35
            last = self.times[side][-1] if self.times[side] else -10_000
            if self.rng.random() < p and lo - last >= 210:
                self.times[side].append(int(self.rng.integers(lo, hi)))
            if self.rng.random() < 1.0 * (hi - lo) / 1000.0:
                self.ttmn[side].append(int(self.rng.integers(lo, hi)))

    def decode(self, end_ms: int, window: float, n: int) -> int:
        cl = sum(1 for s in self.times["L"] if end_ms - window <= s < end_ms)
        cr = sum(1 for s in self.times["R"] if end_ms - window <= s < end_ms)
        fl, fr = cl >= n, cr >= n
        return 1 if fl and fr else 3 if fl else 2 if fr else 0


def play(game: str, seed: int, player: str, match: str | None, rng: np.random.Generator) -> dict:
    env = base.make(game)
    env.max_ticks = FAKE_MAX_TICKS                 # instance override, fixture only
    frame = env.reset(seed)
    det = LoomingDetector()
    det.reset()
    agent = None
    if player == "random":
        agent = RandomAgent()
        agent.reset(seed)
    elif player == "brainless":
        agent = ThresholdAgent(FAKE_THETA[game])
        agent.reset(seed)
    spec = None
    gf = None
    if player == "fly":
        spec = ({"decoder": "GiantFibreDecoder", "window_ms": 50.0,
                 "rule": "both GF -> JUMP; left GF only -> RIGHT; right GF only -> LEFT; none -> NOOP"}
                if match == "m7c" else
                {"decoder": "CountWindowDecoder", "window_ms": 300.0, "min_spikes": 2,
                 "rule": "count GF spikes per side in [now - window_ms, now); both >= n -> JUMP; "
                         "left only -> RIGHT; right only -> LEFT; neither -> NOOP",
                 "now_ms": "end of the current tick (game clock)"})
        gf = FakeGF(rng)
    frames, actions, looms, states, rects = [], [], [], [], []
    k = 0
    while True:
        rect = eye_rect(env)
        if not np.array_equal(redraw(rect), frame):
            raise AssertionError(f"fake eye_rect does not reproduce the frame ({game} tick {k})")
        lo = det.update(frame)
        frames.append(frame)
        looms.append([float(lo[0]), float(lo[1])])
        states.append(_state(env))
        rects.append(rect)
        if player == "fly":
            side = None
            if env.dist is not None:
                side = "center" if game == "jump" else ("R" if env.obj_x > env.x else "L")
            gf.tick(k, side, env.dist is not None and env.dist < 6.0)
            n = spec.get("min_spikes", 1)
            a = gf.decode(t_ms(k + 1), spec["window_ms"], n)
        elif player == "one_way":
            a = 2
        else:
            a = agent.act(frame)
        actions.append(int(a))
        frame, _, done, info = env.step(a)
        k += 1
        if done:
            break
    T = len(actions)
    track = {"id": f"{game}/{match}/{player}" if match else f"{game}/{player}", "game": game, "player": player, "match": match,
             "seed": seed, "n_ticks": T, "score": int(info["score"]), "crashed": info["crashed"],
             "truncated": info["truncated"], "t_ms": [t_ms(i) for i in range(T)], "action": actions,
             "loom": looms, "state": {key: [s[key] for s in states] for key in states[0]},
             "eye_rect": rects, "theta": FAKE_THETA[game] if player == "brainless" else None}
    # columns a game does not have: the contract says "null 배열"; the fixture uses both readings
    # (a whole-null field and an array of nulls) so the page is tested on either
    if game == "jump":
        track["state"]["x"] = None
    else:
        track["state"]["airborne"] = None
    for key in ("height", "speed", "x", "obj_x", "dist"):
        col = track["state"][key]
        if col is not None:
            track["state"][key] = [None if v is None else round(float(v), 4) for v in col]
    track["events"], track["obstacles"] = _events(track, game)
    wide = [i for i in range(T) if rects[i] is not None and rects[i][1] - rects[i][0] > 3][:1]
    pick = sorted({0, T // 3, (2 * T) // 3, T - 1, *wide})
    track["check_frames"] = {"ticks": pick, "raw_b64": [base64.b64encode(frames[i].tobytes()).decode()
                                                       for i in pick]}
    track["check"] = {"frames_bit_exact": T, "n_frames": T, "score_matches": True, "actions_match": True}
    track["brain"] = _brain(track, gf, spec, match, rng) if player == "fly" else None
    return track


def _events(track: dict, game: str):
    """Same stamping as the real file: ``tick`` = the step in which it happened, ``t_ms`` =
    floor((tick+1)*1000/60) = the first frame that shows it. Crash: the last step."""
    st, T = track["state"], track["n_ticks"]
    dist, rects = st["dist"], track["eye_rect"]
    events, obstacles = [], []
    cur = None

    def ev(frame, typ, side):
        events.append({"tick": frame - 1, "t_ms": t_ms(frame), "type": typ, "side": side})

    for k in range(1, T):
        if dist[k] is not None and dist[k - 1] is None:
            side = "center" if game == "jump" else ("right" if st["obj_x"][k] > st["x"][k] else "left")
            cur = {"spawn_tick": k - 1, "end_tick": None, "side": side, "outcome": None,
                   "reach_tick": None, "reach_t_ms": None, "visible2px_tick": None}
            ev(k, "spawn", side)
        if cur is not None and cur["visible2px_tick"] is None and rects[k] is not None \
                and rects[k][1] - rects[k][0] >= 2:
            cur["visible2px_tick"] = k - 1
            ev(k, "visible2px", cur["side"])
        if cur is not None and cur["reach_tick"] is None and dist[k] is not None and dist[k] <= 1.5:
            cur["reach_tick"], cur["reach_t_ms"] = k - 1, t_ms(k)
        if st["score"][k] > st["score"][k - 1]:
            ev(k, "pass", cur["side"] if cur else "center")
            if cur:
                cur["end_tick"], cur["outcome"] = k - 1, "pass"
                obstacles.append(cur)
                cur = None
        if st["speed"][k] > st["speed"][k - 1]:
            ev(k, "speedup", None)
    if track["crashed"]:
        ev(T, "crash", cur["side"] if cur else "center")
    if cur is not None:
        cur["end_tick"], cur["outcome"] = T - 1, "crash" if track["crashed"] else "cut"
        if cur["reach_tick"] is None and track["crashed"]:
            cur["reach_tick"], cur["reach_t_ms"] = T - 1, t_ms(T)
        obstacles.append(cur)
    for ob in obstacles:
        cols = [0, 1] if ob["side"] == "center" else [0] if ob["side"] == "left" else [1]
        contact = ob["reach_t_ms"] if ob["reach_t_ms"] is not None else t_ms(ob["end_tick"] + 1)
        tot = late = 0.0
        for k in range(ob["spawn_tick"] + 1, ob["end_tick"] + 1):
            v = sum(track["loom"][k][c] for c in cols)
            tot += v
            if t_ms(k) >= contact - 300:
                late += v
        ob["loom_last300_frac"] = late / tot if tot > 0 else None
    return events, obstacles


def _brain(track, gf, spec, match, rng):
    T = track["n_ticks"]
    rate = {}
    for name in SETS:
        side = 0 if name.endswith("_L") else 1
        if name.startswith(("LC4", "LPLC2")):
            rate[name] = [round(0.3 + 4000.0 * track["loom"][k][side] + float(rng.random()) * 0.4, 4)
                          for k in range(T)]
        else:
            src = gf.times if name.startswith("DNp01") else gf.ttmn
            times = src[name[-1]]
            rate[name] = [round(1000.0 * sum(1 for s in times if t_ms(k) <= s < t_ms(k + 1))
                                / (t_ms(k + 1) - t_ms(k)), 4) for k in range(T)]
    return {"decoder": spec, "warmup_ms": 0 if match == "m7c" else 5000,
            "transient_ms": 2000 if match == "m7c" else 0,
            "gf": {"L": gf.times["L"], "R": gf.times["R"]}, "ttmn": {"L": gf.ttmn["L"], "R": gf.ttmn["R"]},
            "set_rate_hz": rate,
            "net_spikes": [int(780 + 60 * math.sin(k / 7.0) + rng.integers(0, 40)) for k in range(T)]}


def _verdict(lose_one_way: bool, ceiling):
    flags = {"beats": False, "loses": False}
    return {"vs_random": {"diffs": [0.0] * 10, "mean": 0.0, "sem": 0.0, **flags},
            "vs_brainless": {"diffs": [-9.0] * 10, "mean": -9.0, "sem": 1.0, "beats": False, "loses": True},
            "vs_one_way": {"diffs": [-5.0] * 10, "mean": -5.0, "sem": 0.5, "beats": False,
                           "loses": lose_one_way},
            "beat_random": False, "beat_one_way": False, "brainless_verdict": "lost to the brainless bot",
            "ceiling": ceiling}


def make_show_data() -> dict:
    rng = np.random.default_rng(20261029)
    tracks, games = [], {}
    for game, title in GAMES.items():
        for player, match in (("fly", "m7c"), ("fly", "m7c2"), ("random", None), ("one_way", None),
                              ("brainless", None)):
            tracks.append(play(game, 10, player, match, rng))
        ceiling = 72 if game == "dodge_v2" else None
        games[game] = {
            "title_ko": title, "rules": base.make(game).describe(),
            "eval_seeds": list(range(10, 20)),
            "scores": {"fly_m7c": [int(x) for x in rng.integers(0, 3, 10)],
                       "fly_m7c2": [0] * 10,
                       "random": [int(x) for x in rng.integers(0, 2, 10)],
                       "one_way": [int(x) for x in rng.integers(0, 30 if game == "dodge_v2" else 1, 10)],
                       "brainless": [int(x) for x in rng.integers(5, 60 if game == "jump" else 72, 10)]},
            "one_way_dir": "left", "brainless_theta": FAKE_THETA[game], "ceiling": ceiling,
            "verdict": {"m7c": _verdict(game == "dodge_v2", ceiling), "m7c2": _verdict(game == "dodge_v2", ceiling)}}
    return {"version": 1,
            "meta": {"generated": "2026-10-29T00:00:00+0900", "git_commit": "fixture",
                     "tick_ms_rule": "t_ms(k) = floor(k*1000/60)",
                     "action_codes": {"0": "NOOP", "1": "JUMP", "2": "LEFT", "3": "RIGHT"},
                     "sources": {"tests/show_fake.py": "fixture"}, "fixture": FIXTURE_NOTE},
            "games": games, "tracks": tracks}


def _blob(rng, n, cx, cy, rx, ry):
    a = rng.random(n) * 2 * math.pi
    r = np.sqrt(rng.random(n))
    return np.stack([cx + rx * r * np.cos(a), cy + ry * r * np.sin(a)], 1)


def make_brain_map() -> dict:
    """A cartoon brain (ellipses), fly left on screen left. Shape only, not coordinates."""
    rng = np.random.default_rng(7)
    w, h = 4096, 4000
    pts = np.concatenate([_blob(rng, 1400, 2048, 1150, 820, 520), _blob(rng, 500, 760, 1150, 380, 520),
                          _blob(rng, 500, 3336, 1150, 380, 520), _blob(rng, 900, 2048, 2950, 330, 900)])
    xy = np.clip(np.rint(pts), 0, 4095).astype(int)

    def neuron(path, jitter):
        p = np.array(path, float) + rng.normal(0, jitter, (len(path), 2))
        p = np.clip(np.rint(p), 0, 4095).astype(int)
        return [int(v) for seg in zip(p[:-1], p[1:]) for v in (*seg[0], *seg[1])]

    def mirror(path):
        return [(w - 1 - x, y) for x, y in path]

    plans = {"LC4": ([(700, 1100), (1050, 1050), (1450, 1000), (1800, 980)], 6, 70),
             "LPLC2": ([(760, 1260), (1100, 1220), (1500, 1120), (1820, 1010)], 7, 70),
             "DNp01": ([(1820, 980), (1900, 1300), (1960, 1900), (1990, 2600), (2000, 3300)], 1, 10),
             "TTMn": ([(1960, 2650), (1800, 2720), (1600, 2800)], 3, 30)}
    sets = {}
    for typ, (path, n, jit) in plans.items():
        for side in "LR":
            pth = path if side == "L" else mirror(path)
            lines = [neuron(pth, jit) for _ in range(n)]
            soma = [v for ln in lines for v in ln[:2]]
            xs, ys = soma[0::2], soma[1::2]
            sets[f"{typ}_{side}"] = {"n_neurons": n, "soma": soma, "lines": lines,
                                     "centroid": [int(sum(xs) / len(xs)), int(sum(ys) / len(ys))]}
    return {"version": 1,
            "view": {"plane": "x-z", "up": "anterior", "screen_left": "fly left",
                     "how_left_was_decided": "FAKE fixture: drawn that way (tests/show_fake.py)"},
            "box": {"w": w, "h": h}, "dots": {"n": int(len(xy)), "seed": 0, "xy": [int(v) for v in xy.ravel()]},
            "sets": sets, "sources": {"tests/show_fake.py": "fixture"}, "generated": "2026-10-29T00:00:00+0900",
            "fixture": FIXTURE_NOTE}


def write(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "show-data.json").write_text(json.dumps(make_show_data(), ensure_ascii=False))
    (out_dir / "brain-map.json").write_text(json.dumps(make_brain_map(), ensure_ascii=False))
    return out_dir


if __name__ == "__main__":
    import sys
    print(write(Path(sys.argv[1])))
