"""M7b: the agents that play the games.

Every agent has the same two calls: ``reset(seed)`` at the start of an episode and
``act(frame) -> action index`` once per tick. **``act`` receives the screen and nothing else.**

| agent | sees | role in the table |
|---|---|---|
| ``RandomAgent`` | nothing (uniform over the 4 shared actions) | floor |
| ``ThresholdAgent`` | ``LoomingDetector`` on the frame — the same signal the fly gets | the wall the fly has to beat ("뇌 없는 봇") |
| ``BrainAgent`` | ``LoomingDetector`` → brain → ``MotorDecoder`` | M7c; interface only here |
| ``StateOracleAgent`` | **game state** (cheats) | reference ceiling only, never a comparison baseline |

The threshold bot's rule is one rule with one threshold, the same for every game ("same eyes,
same muscles"):

    L, R = looming(frame)
    both above θ  -> JUMP            (something coming straight at us)
    left above θ  -> RIGHT           (move away from it)
    right above θ -> LEFT
    otherwise     -> NOOP

θ is chosen per game on training seeds 0-9 and reported on evaluation seeds 10-19
(``runner.select_threshold``). The rule itself never changes between games.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from ...probe.motor import MotorDecoder
from ...sensory.looming import LoomingDetector
from .base import N_ACTIONS, Action

RANDOM_STREAM = 7919          # the random bot's RNG is seeded with (seed, RANDOM_STREAM), so it
                              # is independent of the game's own RNG but still reproducible


class Agent:
    name = "agent"

    def reset(self, seed: int) -> None:
        pass

    def act(self, frame: np.ndarray) -> int:
        raise NotImplementedError

    def describe(self) -> dict:
        return {"agent": self.name}


class RandomAgent(Agent):
    name = "random"

    def __init__(self):
        self.rng = np.random.default_rng(0)

    def reset(self, seed: int) -> None:
        self.rng = np.random.default_rng([int(seed), RANDOM_STREAM])

    def act(self, frame: np.ndarray) -> int:
        return int(self.rng.integers(0, N_ACTIONS))

    def describe(self) -> dict:
        return {"agent": self.name, "policy": "uniform over NOOP/JUMP/LEFT/RIGHT each tick",
                "rng": f"default_rng([seed, {RANDOM_STREAM}])"}


class ThresholdAgent(Agent):
    """The brainless bot: the fly's input signal plus one threshold."""

    name = "threshold"

    def __init__(self, theta: float):
        if not theta > 0:
            raise ValueError("theta must be > 0")
        self.theta = float(theta)
        self.detector = LoomingDetector()         # same eyes: no per-game settings exist
        self.last_signal = (0.0, 0.0)

    def reset(self, seed: int) -> None:
        self.detector.reset()
        self.last_signal = (0.0, 0.0)

    def act(self, frame: np.ndarray) -> int:
        left, right = self.detector.update(frame)
        self.last_signal = (left, right)
        fl, fr = left > self.theta, right > self.theta
        if fl and fr:
            return int(Action.JUMP)
        if fl:
            return int(Action.RIGHT)
        if fr:
            return int(Action.LEFT)
        return int(Action.NOOP)

    def describe(self) -> dict:
        return {"agent": self.name, "theta": self.theta,
                "rule": "both>θ: JUMP; left>θ: RIGHT; right>θ: LEFT; else NOOP",
                "detector": LoomingDetector.describe()}


class BrainAgent(Agent):
    """M7c placeholder: frame -> LoomingDetector -> brain -> MotorDecoder -> action.

    ``brain_step(left, right)`` must advance the simulated brain by one game tick (1/60 s of
    simulated time) with the looming signal as its input and return the indices of the neurons
    that spiked in the decoder's window. ``brain_reset(seed)`` restarts it. M7c provides both;
    here they are exercised with a fake brain in tests/test_games.py.
    """

    name = "brain"

    def __init__(self, brain_step: Callable[[float, float], np.ndarray], decoder: MotorDecoder,
                 brain_reset: Callable[[int], None] | None = None):
        self.brain_step = brain_step
        self.brain_reset = brain_reset
        self.decoder = decoder
        self.detector = LoomingDetector()         # the same eyes as the threshold bot
        self.last_signal = (0.0, 0.0)

    def reset(self, seed: int) -> None:
        self.detector.reset()
        self.last_signal = (0.0, 0.0)
        if self.brain_reset is not None:
            self.brain_reset(int(seed))

    def act(self, frame: np.ndarray) -> int:
        left, right = self.detector.update(frame)
        self.last_signal = (left, right)
        spikes = self.brain_step(left, right)
        return int(Action[self.decoder.decode(spikes)])

    def describe(self) -> dict:
        return {"agent": self.name, "decoder": self.decoder.describe(),
                "detector": LoomingDetector.describe(), "status": "interface only (M7c)"}


class StateOracleAgent(Agent):
    """**Reads the game's internal state.** A reference for how far a perfect-timing player gets
    under these rules, not a baseline: it does not see what the fly sees."""

    name = "state-oracle"

    def __init__(self, env):
        self.env = env

    def act(self, frame: np.ndarray) -> int:
        env = self.env
        if env.name == "jump":
            from . import jump as J
            if env.dist is None or env.airborne or env.cooldown:
                return int(Action.NOOP)
            v = env.speed
            # aim the barrier's pass through the middle of the high part of the jump
            lead = J.JUMP_TICKS / 2 - J.DEPTH / (2 * v)
            return int(Action.JUMP) if (env.dist - J.D_HIT) / v <= lead else int(Action.NOOP)
        if env.name in ("dodge", "dodge_v2"):
            from . import dodge as D
            if env.obj_x is None:
                return int(Action.NOOP)
            need = D.OBJ_HALF_WIDTH + D.PLAYER_HALF_WIDTH + 0.05
            away = -1.0 if env.obj_x > env.x else 1.0
            target = env.obj_x + away * need
            if abs(target) > env.x_limit:                  # wall (v1 only): pass on the other side
                target = env.obj_x - away * need
            if abs(target - env.x) < D.PLAYER_SPEED / 2:
                return int(Action.NOOP)
            return int(Action.RIGHT) if target > env.x else int(Action.LEFT)
        raise KeyError(f"no oracle for game {env.name!r}")

    def describe(self) -> dict:
        return {"agent": self.name, "reads": "game state (not pixels)",
                "role": "reference ceiling only, not a baseline"}
