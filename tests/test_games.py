"""M7b tests: game harness, pixel-only looming detector, comparison agents, motor interface.

Everything here runs on the CPU in a few seconds; nothing needs the connectome or a GPU.
"""
import inspect
import json
import struct
import zlib
from pathlib import Path

import numpy as np
import pytest

from flysim.apps.games import base as B
from flysim.apps.games import dodge as D
from flysim.apps.games import jump as J
from flysim.apps.games import media, runner
from flysim.apps.games.agents import (BrainAgent, RandomAgent, StateOracleAgent,
                                      ThresholdAgent)
from flysim.apps.games.base import Action, make
from flysim.probe.motor import ACTION_NAMES, MotorDecoder
from flysim.sensory import looming as LM
from flysim.sensory.looming import LoomingDetector

ROOT = Path(__file__).resolve().parents[1]
GAMES = ("jump", "dodge", "dodge_v2")


def play(game, actions_or_agent, seed, n=None):
    """Return (frames, score, info) for a fixed action list or an agent."""
    env = make(game)
    frame = env.reset(seed)
    frames = [frame]
    info = {}
    if isinstance(actions_or_agent, list):
        for a in actions_or_agent:
            if env.done:
                break
            frame, _, _, info = env.step(a)
            frames.append(frame)
    else:
        agent = actions_or_agent
        agent.reset(seed)
        done = False
        while not done and (n is None or len(frames) <= n):
            frame, _, done, info = env.step(agent.act(frame))
            frames.append(frame)
    return frames, env.score, info


# ---------------------------------------------------------------------------------------------
# the environment interface
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("game", GAMES)
def test_frame_format_and_interface(game):
    env = make(game)
    f = env.reset(3)
    assert isinstance(f, np.ndarray) and f.dtype == np.uint8 and f.shape == (84, 84)
    f2, r, done, info = env.step(int(Action.NOOP))
    assert f2.dtype == np.uint8 and f2.shape == (84, 84)
    assert isinstance(r, float) and isinstance(done, bool)
    for k in ("tick", "score", "crashed", "truncated", "game"):
        assert k in info
    assert info["tick"] == 1 and info["game"] == game
    with pytest.raises(ValueError):
        env.step(4)
    with pytest.raises(TypeError):
        env.reset(1.5)
    d = env.describe()
    assert d["game"] == game and "score" in d


@pytest.mark.parametrize("game", GAMES)
def test_step_after_done_raises(game):
    env = make(game)
    env.reset(0)
    done = False
    while not done:                                     # standing still loses both games
        _, _, done, info = env.step(int(Action.NOOP))
    assert info["crashed"]
    with pytest.raises(RuntimeError):
        env.step(int(Action.NOOP))


def test_same_muscles_shared_action_space():
    a, b, c = make("jump"), make("dodge"), make("dodge_v2")
    assert a.n_actions == b.n_actions == c.n_actions == 4
    assert a.action_meanings == b.action_meanings == c.action_meanings == ("NOOP", "JUMP", "LEFT", "RIGHT")
    assert set(B.ALE_FULL_ACTION_INDEX) == set(Action)
    assert all(0 <= v < B.FULL_ACTION_COUNT for v in B.ALE_FULL_ACTION_INDEX.values())
    assert len(set(B.ALE_FULL_ACTION_INDEX.values())) == 4
    assert tuple(ACTION_NAMES) == B.ACTIONS              # motor decoder speaks the same names


# ---------------------------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("game", GAMES)
def test_same_seed_same_frames_and_score(game):
    rng = np.random.default_rng(42)
    actions = [int(a) for a in rng.integers(0, 4, 600)]
    f1, s1, _ = play(game, actions, seed=7)
    f2, s2, _ = play(game, actions, seed=7)
    assert len(f1) == len(f2) and s1 == s2
    assert all(np.array_equal(a, b) for a, b in zip(f1, f2))
    f3, _, _ = play(game, actions, seed=8)
    assert any(not np.array_equal(a, b) for a, b in zip(f1, f3))   # a different seed is a different game


@pytest.mark.parametrize("game", GAMES)
@pytest.mark.parametrize("make_agent", [lambda: RandomAgent(), lambda: ThresholdAgent(1e-3)])
def test_agents_are_deterministic(game, make_agent):
    f1, s1, _ = play(game, make_agent(), seed=12, n=2000)
    f2, s2, _ = play(game, make_agent(), seed=12, n=2000)
    assert s1 == s2 and len(f1) == len(f2)
    assert all(np.array_equal(a, b) for a, b in zip(f1, f2))


# ---------------------------------------------------------------------------------------------
# game rules
# ---------------------------------------------------------------------------------------------
def test_jump_rules():
    env = make("jump")
    env.reset(0)
    # JUMP while airborne does nothing; landing starts the cooldown
    env.step(int(Action.JUMP))
    assert env.airborne and env.jump_t == 0
    for _ in range(5):
        env.step(int(Action.JUMP))
    assert env.jump_t == 5
    while env.airborne:
        env.step(int(Action.NOOP))
    assert env.cooldown == J.LAND_COOLDOWN
    env.step(int(Action.JUMP))
    assert not env.airborne                                    # cooling down
    assert J.jump_height(J.JUMP_TICKS // 2) == pytest.approx(J.JUMP_PEAK, rel=0.01)
    assert J.jump_height(-1) == 0.0 and J.jump_height(J.JUMP_TICKS) == 0.0


def test_jump_speed_steps_and_no_tunnelling():
    env = make("jump")
    env.reset(0)
    assert env.speed == pytest.approx(J.SPEED0)
    env.cleared = J.SPEED_EVERY
    assert env.speed == pytest.approx(J.SPEED0 * J.SPEED_FACTOR)
    # a barrier faster than its own depth must still hit a player on the ground
    env.cleared = 100
    assert env.speed > J.DEPTH * 2
    env.dist = J.D_HIT + 0.1
    _, _, done, info = env.step(int(Action.NOOP))
    assert done and info["crashed"]


def test_dodge_rules():
    env = make("dodge")
    env.reset(0)
    x0 = env.x
    env.step(int(Action.RIGHT))
    assert env.x == pytest.approx(x0 + D.PLAYER_SPEED)
    env.step(int(Action.JUMP))                                 # no effect in this game
    assert env.x == pytest.approx(x0 + D.PLAYER_SPEED)
    for _ in range(1000):
        env.x = D.X_LIMIT
        env.step(int(Action.RIGHT))
        if env.done:
            break
    assert env.x <= D.X_LIMIT
    # standing still never avoids a pillar (spawned relative to where the player stands)
    assert D.SPAWN_OFFSET < D.OBJ_HALF_WIDTH + D.PLAYER_HALF_WIDTH
    # the pillar never straddles the screen centre while the player stands still
    assert D.SPAWN_OFFSET > D.OBJ_HALF_WIDTH


def test_dodge_v2_is_v1_without_walls():
    """Pre-registered v2: no side walls, everything else v1 — including the random draws."""
    v1, v2 = make("dodge"), make("dodge_v2")
    assert v1.x_limit == D.X_LIMIT and v2.x_limit == float("inf")
    r1, r2 = v1.describe(), v2.describe()
    assert r1["walls"] is True and r2["walls"] is False and r2["X_LIMIT"] is None
    assert {k: v for k, v in r1.items() if k not in ("game", "walls", "X_LIMIT")} == \
           {k: v for k, v in r2.items() if k not in ("game", "walls", "X_LIMIT")}
    # identical frames for identical actions until the v1 player would reach a wall
    f1, f2 = v1.reset(3), v2.reset(3)
    for _ in range(3000):
        assert np.array_equal(f1, f2)
        f1, _, d1, _ = v1.step(int(Action.LEFT))
        f2, _, d2, _ = v2.step(int(Action.LEFT))
        if abs(v1.x) >= D.X_LIMIT or d1 or d2:
            break
    assert v2.x < -D.X_LIMIT or abs(v1.x) >= D.X_LIMIT        # v2 walks past where v1 stops
    while not v2.done and v2.x > -3 * D.X_LIMIT:
        v2.step(int(Action.LEFT))
    assert v2.done or v2.x <= -3 * D.X_LIMIT


@pytest.mark.parametrize("game", GAMES)
def test_games_are_winnable_and_not_degenerate(game):
    """The state oracle (cheats) goes far; constant actions and random play do not."""
    seeds = (10, 11)
    oracle = runner.run_many(game, lambda env: StateOracleAgent(env), seeds)
    assert oracle["mean"] >= 20
    for a in Action:
        const = runner.run_many(game, lambda env, a=a: _Const(a), seeds)
        if game == "dodge_v2" and a in (Action.LEFT, Action.RIGHT):
            # Known property of v2 as pre-registered, pinned so it cannot change unnoticed: with
            # no walls, moving one way all the time outruns every pillar while the approach takes
            # longer than ~56 ticks (pillars spawn relative to where the player stands), so it
            # scores ~28 before the speed-up catches it. In v1 the wall stopped this (score < 1).
            assert 20 <= const["mean"] <= 35, (game, a.name, const)
            continue
        assert const["mean"] < 3, (game, a.name, const)
    rnd = runner.run_many(game, lambda env: RandomAgent(), seeds)
    assert rnd["mean"] < 3


class _Const:
    def __init__(self, a):
        self.a = int(a)

    def reset(self, seed):
        pass

    def act(self, frame):
        return self.a


# ---------------------------------------------------------------------------------------------
# the detector reads pixels only
# ---------------------------------------------------------------------------------------------
def test_detector_signature_is_pixels_only():
    sig = inspect.signature(LoomingDetector.update)
    assert list(sig.parameters) == ["self", "frame"]
    assert list(inspect.signature(LoomingDetector.__init__).parameters) == ["self"]  # no per-game knobs
    det = LoomingDetector()
    env = make("jump")
    env.reset(0)
    with pytest.raises(TypeError):
        det.update(env)                                   # a game object is refused
    with pytest.raises(TypeError):
        det.update(env.frame().astype(np.float32))        # only raw uint8 pixels
    with pytest.raises(TypeError):
        det.update(np.zeros((84, 84, 3), np.uint8))       # 2-D grayscale only
    src = Path(LM.__file__).read_text()
    assert "flysim.apps" not in src and "from ..apps" not in src and "games" not in src.split('"""')[-1]


def test_agents_feed_the_detector_frames_only():
    """The pixel agents hold no reference to the game; only the labelled oracle does."""
    for agent in (ThresholdAgent(1e-3), RandomAgent()):
        assert not any(isinstance(v, B.GameEnv) for v in vars(agent).values())
    env = make("jump")
    assert StateOracleAgent(env).env is env


def _disc(cx, cy, r, H=84, W=84):
    yy, xx = np.mgrid[0:H, 0:W] + 0.5
    cov = np.zeros((H, W))
    for oy in (0.125, 0.375, 0.625, 0.875):
        for ox in (0.125, 0.375, 0.625, 0.875):
            cov += (xx - 0.5 + ox - cx) ** 2 + (yy - 0.5 + oy - cy) ** 2 < r * r
    cov /= 16
    return np.clip(np.rint(180 * (1 - cov) + 30 * cov), 0, 255).astype(np.uint8)


def _trace(frames):
    det = LoomingDetector()
    return np.array([det.update(f) for f in frames])


def test_detector_expansion_vs_translation_vs_recession():
    """Generic stimuli, no game: approach is signalled, sliding and receding are not."""
    expand = _trace([_disc(63, 42, 2 + 0.4 * t) for t in range(40)])
    slide = _trace([_disc(50 + 0.6 * t, 42, 6) for t in range(40)])
    recede = _trace([_disc(63, 42, 18 - 0.4 * t) for t in range(40)])
    assert expand[0].tolist() == [0.0, 0.0]                    # first frame has no history
    assert expand[1:, 1].min() > 0 and expand[:, 0].max() == 0  # right half only
    assert slide[:, 1].max() < 0.05 * expand[1:, 1].mean()
    assert recede.max() == 0.0
    centred = _trace([_disc(42, 42, 60 / (40 - 0.9 * t)) for t in range(40)])
    # symmetric object, equal halves (up to float32 summation order, ~1e-7)
    assert np.allclose(centred[:, 0], centred[:, 1], rtol=1e-3, atol=1e-6)
    assert centred[-1, 0] > centred[10, 0]                      # grows faster as it comes closer


def test_detector_reset_and_size_guard():
    det = LoomingDetector()
    det.update(_disc(60, 40, 3))
    assert det.update(_disc(60, 40, 6))[1] > 0
    det.reset()
    assert det.update(_disc(60, 40, 6)) == (0.0, 0.0)
    with pytest.raises(ValueError):
        det.update(np.zeros((84, 82), np.uint8))
    with pytest.raises(ValueError):
        LoomingDetector().update(np.zeros((10, 7), np.uint8))


# ---------------------------------------------------------------------------------------------
# agents
# ---------------------------------------------------------------------------------------------
class _StubDetector:
    def __init__(self, values):
        self.values = list(values)

    def reset(self):
        pass

    def update(self, frame):
        return self.values.pop(0)


def test_threshold_rule_is_one_rule_for_all_games():
    a = ThresholdAgent(0.5)
    a.detector = _StubDetector([(0.6, 0.7), (0.6, 0.1), (0.1, 0.6), (0.1, 0.1), (0.5, 0.5)])
    f = np.zeros((84, 84), np.uint8)
    got = [Action(a.act(f)).name for _ in range(5)]
    assert got == ["JUMP", "RIGHT", "LEFT", "NOOP", "NOOP"]     # strictly above θ
    with pytest.raises(ValueError):
        ThresholdAgent(0.0)


def test_random_agent_covers_the_action_space():
    a = RandomAgent()
    a.reset(10)
    acts = [a.act(None) for _ in range(4000)]
    counts = np.bincount(acts, minlength=4)
    assert counts.min() > 800 and len(counts) == 4


# ---------------------------------------------------------------------------------------------
# motor decoder + BrainAgent (interfaces for M7c, fake spikes only)
# ---------------------------------------------------------------------------------------------
GROUPS = {"JUMP": np.array([0, 1, 2, 3]), "LEFT": np.array([10, 11]), "RIGHT": np.array([20, 21])}


def test_motor_decoder_on_fake_spikes():
    dec = MotorDecoder(GROUPS, window_ms=1000 / 60, min_rate_hz=30.0)
    assert dec.decode(np.array([], dtype=np.int64)) == "NOOP"
    assert dec.decode(np.array([0, 1])) == "JUMP"
    assert dec.decode(np.array([10, 0])) == "LEFT"               # 1/2 of LEFT beats 1/4 of JUMP
    assert dec.decode(np.array([20, 21, 10, 11])) == "NOOP"      # exact tie -> no hidden priority
    assert dec.decode(np.array([99, 100])) == "NOOP"             # spikes outside every group
    r = dec.rates(np.array([0]))
    assert r["JUMP"] == pytest.approx(1 / (4 * (1 / 60)))
    strict = MotorDecoder(GROUPS, window_ms=1000.0, min_rate_hz=10.0)
    assert strict.decode(np.array([0])) == "NOOP"                # 0.25 Hz < 10 Hz


def test_motor_decoder_validation():
    with pytest.raises(LookupError):
        MotorDecoder({"JUMP": np.array([], dtype=np.int64)}, 10, 1)
    with pytest.raises(LookupError):
        MotorDecoder({}, 10, 1)
    with pytest.raises(ValueError):
        MotorDecoder({"JUMP": [1, 2], "LEFT": [2, 3]}, 10, 1)    # shared neuron
    with pytest.raises(ValueError):
        MotorDecoder({"NOOP": [1]}, 10, 1)
    with pytest.raises(ValueError):
        MotorDecoder({"FLY": [1]}, 10, 1)
    with pytest.raises(ValueError):
        MotorDecoder({"JUMP": [1]}, 0, 1)


def test_brain_agent_with_a_fake_brain():
    """Plumbing only: looming -> fake brain -> decoder -> action. No neurons, no engine."""
    calls = {"reset": [], "steps": 0}

    def fake_step(left, right):
        calls["steps"] += 1
        if left > 1e-4 and right > 1e-4:
            return np.array([0, 1, 2])                           # the JUMP group fires
        return np.array([], dtype=np.int64)

    dec = MotorDecoder(GROUPS, window_ms=1000 / 60, min_rate_hz=1.0)
    agent = BrainAgent(fake_step, dec, brain_reset=lambda s: calls["reset"].append(s))
    frames, score, info = play("jump", agent, seed=10, n=3000)
    assert calls["reset"] == [10] and calls["steps"] == len(frames) - 1
    assert score > 0                                             # it does jump over things
    # same eyes: the brain agent's detector gives the threshold bot's signal on the same frames
    ta, ba = ThresholdAgent(1.0), BrainAgent(fake_step, dec)
    ta.reset(0), ba.reset(0)
    for f in frames[:400]:
        ta.act(f), ba.act(f)
        assert ta.last_signal == ba.last_signal


# ---------------------------------------------------------------------------------------------
# runner, selection protocol, media
# ---------------------------------------------------------------------------------------------
def test_threshold_selection_uses_only_training_seeds(monkeypatch):
    seen = []
    real = runner.run_many

    def spy(game, factory, seeds):
        seen.append(tuple(seeds))
        return real(game, factory, seeds)

    monkeypatch.setattr(runner, "run_many", spy)
    sel = runner.select_threshold("jump", grid=(1e-4, 2e-4, 4e-3), seeds=(0, 1), workers=1)
    assert set(seen) == {(0, 1)}
    assert sel["chosen_theta"] in (1e-4, 2e-4, 4e-3)
    best = max(r["mean"] for r in sel["grid"])
    assert next(r for r in sel["grid"] if r["theta"] == sel["chosen_theta"])["mean"] == best
    assert runner.TRAIN_SEEDS == tuple(range(10)) and runner.EVAL_SEEDS == tuple(range(10, 20))
    assert not set(runner.TRAIN_SEEDS) & set(runner.EVAL_SEEDS)


def _png_chunks(data: bytes):
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    i, out = 8, []
    while i < len(data):
        n = struct.unpack(">I", data[i:i + 4])[0]
        tag, body = data[i + 4:i + 8], data[i + 8:i + 8 + n]
        crc = struct.unpack(">I", data[i + 8 + n:i + 12 + n])[0]
        assert crc == zlib.crc32(tag + body) & 0xFFFFFFFF
        out.append((tag, body))
        i += 12 + n
    return out


def test_png_and_apng_are_valid(tmp_path):
    img = (np.arange(84 * 84) % 251).astype(np.uint8).reshape(84, 84)
    chunks = _png_chunks(media.write_png(tmp_path / "a.png", img).read_bytes())
    assert chunks[0][0] == b"IHDR" and struct.unpack(">II", chunks[0][1][:8]) == (84, 84)
    raw = zlib.decompress(b"".join(b for t, b in chunks if t == b"IDAT"))
    rows = np.frombuffer(raw, np.uint8).reshape(84, 85)
    assert (rows[:, 0] == 0).all() and np.array_equal(rows[:, 1:], img)
    frames = [media.annotate(img, s, (0.001 * s, 0.0), s % 4) for s in range(5)]
    chunks = _png_chunks(media.write_apng(tmp_path / "b.apng.png", frames).read_bytes())
    tags = [t for t, _ in chunks]
    assert tags[1] == b"acTL" and struct.unpack(">I", chunks[1][1][:4])[0] == 5
    seqs = [struct.unpack(">I", b[:4])[0] for t, b in chunks if t in (b"fcTL", b"fdAT")]
    assert seqs == list(range(len(seqs)))                       # APNG sequence numbers
    assert frames[0].shape == (84 * 3 + 30, 84 * 3)


@pytest.mark.skipif(not (ROOT / "runs" / "m7b" / "baselines.json").exists(),
                    reason="baseline table not generated (python -m flysim.apps.games.runner)")
def test_baseline_table_shape():
    r = json.loads((ROOT / "runs" / "m7b" / "baselines.json").read_text())
    assert r["train_seeds"] == list(range(10)) and r["eval_seeds"] == list(range(10, 20))
    for game in ("jump", "dodge"):
        row = r["games"][game]
        assert row["random"]["seeds"] == list(range(10, 20))
        assert row["threshold"]["eval"]["seeds"] == list(range(10, 20))
        assert all(set(g) >= {"theta", "mean"} for g in row["threshold"]["selection"]["grid"])
        assert row["threshold"]["theta"] == row["threshold"]["selection"]["chosen_theta"]


PROV = ROOT / "data-provenance" / "m7b-baselines.json"


@pytest.mark.skipif(not PROV.exists(), reason="committed baselines missing")
def test_committed_baselines_cover_all_three_games():
    r = json.loads(PROV.read_text())
    assert set(r["games"]) >= {"jump", "dodge", "dodge_v2"}
    v2 = r["games"]["dodge_v2"]
    assert v2["rules"]["walls"] is False
    assert v2["threshold"]["eval"]["seeds"] == list(range(10, 20))
    assert v2["threshold"]["theta"] == v2["threshold"]["selection"]["chosen_theta"]
    assert "generated" in v2 and "git_commit" in v2          # when the added game was produced
