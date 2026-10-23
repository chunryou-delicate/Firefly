"""M7b: the game environment interface and the shared first-person renderer.

Interface (shaped so an ALE game can be wrapped later without touching the agents)::

    env.reset(seed)  -> frame                      uint8, (84, 84), grayscale
    env.step(action) -> frame, reward, done, info  one tick = 1/60 s

* **Same muscles.** Every game takes the same four shared actions, ``ACTIONS``. A later ALE
  wrapper maps them through ``ALE_FULL_ACTION_INDEX`` — one fixed table for every game, never a
  per-game choice (docs/m7-game-ladder.md rule 1). ``FULL_ACTION_COUNT`` = 18 is ALE's full set,
  so the action space can grow without changing this interface.
* **Determinism.** All randomness comes from ``numpy.random.default_rng(seed)`` created in
  ``reset``; same seed, same frames, same score (tests/test_games.py).
* **What an agent may see.** ``step`` returns the frame; ``info`` carries the score and the
  tick for the harness and the human, not for pixel-only agents (``agents.py`` documents who
  reads what).

The two games share one renderer: a fixed camera looking ahead over a flat ground, with an
object drawn in perspective (screen size ~ 1/distance). Edges are drawn with fractional pixel
coverage so a far object grows smoothly instead of in whole-pixel jumps. **Assumption:** the
camera does not move when the player jumps or steps sideways in height; only the lateral
offset of the object changes when the player moves. All scene constants are fixed in code.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from enum import IntEnum

import numpy as np

FRAME_H = FRAME_W = 84          # ALE preprocessing convention (84x84 grayscale)
TICK_HZ = 60                    # one step = 1/60 s
FULL_ACTION_COUNT = 18          # ALE full action set size, for the later wrapper


class Action(IntEnum):
    NOOP = 0
    JUMP = 1
    LEFT = 2
    RIGHT = 3


ACTIONS: tuple[str, ...] = tuple(a.name for a in Action)
N_ACTIONS = len(ACTIONS)
# One fixed mapping into ALE's full action set (NOOP=0, UP=2, RIGHT=3, LEFT=4). "Jump" is UP for
# every game; this table is the only place the mapping lives.
ALE_FULL_ACTION_INDEX = {Action.NOOP: 0, Action.JUMP: 2, Action.RIGHT: 3, Action.LEFT: 4}

# ---- scene (shared by both games; ASSUMPTIONS, fixed) ---------------------------------------
HORIZON_ROW = 36.0              # screen row of the horizon
FOCAL_PX = 60.0                 # perspective scale: screen px per world unit at distance 1
CAMERA_HEIGHT = 1.0             # world units above the ground
SKY, GROUND, OBJECT = 190.0, 110.0, 25.0   # gray levels: objects are darker than both


def background() -> np.ndarray:
    """Sky above the horizon, ground below (float32, so drawing can blend before rounding)."""
    img = np.empty((FRAME_H, FRAME_W), dtype=np.float32)
    rows = np.arange(FRAME_H, dtype=np.float32) + 0.5
    img[:] = np.where(rows < HORIZON_ROW, SKY, GROUND)[:, None]
    return img


_COLS = np.arange(FRAME_W, dtype=np.float32)
_ROWS = np.arange(FRAME_H, dtype=np.float32)


def fill_rect(img: np.ndarray, x0: float, x1: float, y0: float, y1: float, value: float) -> None:
    """Blend ``value`` into ``img`` over [x0, x1) x [y0, y1) with fractional edge coverage."""
    if x1 <= x0 or y1 <= y0:
        return
    cx = np.clip(np.minimum(x1, _COLS + 1.0) - np.maximum(x0, _COLS), 0.0, 1.0)
    cy = np.clip(np.minimum(y1, _ROWS + 1.0) - np.maximum(y0, _ROWS), 0.0, 1.0)
    if not cx.any() or not cy.any():
        return
    cov = np.outer(cy, cx)
    img *= 1.0 - cov
    img += value * cov


def draw_block(img: np.ndarray, lateral0: float, lateral1: float, height: float, dist: float,
               value: float = OBJECT) -> None:
    """A box standing on the ground, spanning world x [lateral0, lateral1] relative to the camera,
    ``height`` tall, ``dist`` ahead. Screen size scales as 1/dist."""
    if dist <= 0:
        return
    s = FOCAL_PX / dist
    cx = FRAME_W / 2.0
    y_bottom = HORIZON_ROW + s * CAMERA_HEIGHT
    y_top = HORIZON_ROW + s * (CAMERA_HEIGHT - height)
    fill_rect(img, cx + s * lateral0, cx + s * lateral1, y_top, y_bottom, value)


def to_frame(img: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------------------------
class GameEnv(ABC):
    """Base class. Subclasses implement ``_reset_state``, ``_advance`` and ``_render``."""

    name: str = "game"
    # 5 min. A safety cap, not a rule of play: speed keeps rising, so a good run should end by
    # difficulty first. Raised from 60 s before any baseline agent was run, because the
    # state oracle reached 60 s on every seed and the clock, not the game, was ending play.
    max_ticks: int = 18_000
    n_actions: int = N_ACTIONS
    action_meanings: tuple[str, ...] = ACTIONS

    def __init__(self):
        self.rng: np.random.Generator | None = None
        self.seed: int | None = None
        self.tick = 0
        self.score = 0
        self.done = True
        self._frame: np.ndarray | None = None

    def reset(self, seed: int) -> np.ndarray:
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
            raise TypeError("seed must be an integer")
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.tick = 0
        self.score = 0
        self.done = False
        self._reset_state()
        self._frame = self._render()
        return self._frame

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        if self.done:
            raise RuntimeError("episode is over; call reset(seed)")
        a = int(action)
        if not 0 <= a < self.n_actions:
            raise ValueError(f"action {action!r} outside 0..{self.n_actions - 1}")
        reward, crashed = self._advance(Action(a))
        self.tick += 1
        self.score += int(reward)
        truncated = self.tick >= self.max_ticks and not crashed
        self.done = crashed or truncated
        self._frame = self._render()
        info = {"tick": self.tick, "score": self.score, "crashed": bool(crashed),
                "truncated": bool(truncated), "game": self.name}
        return self._frame, float(reward), self.done, info

    def frame(self) -> np.ndarray:
        if self._frame is None:
            raise RuntimeError("call reset(seed) first")
        return self._frame

    @abstractmethod
    def _reset_state(self) -> None: ...

    @abstractmethod
    def _advance(self, action: Action) -> tuple[int, bool]:
        """Advance one tick. Returns (reward, crashed)."""

    @abstractmethod
    def _render(self) -> np.ndarray: ...

    @abstractmethod
    def describe(self) -> dict:
        """Every rule constant of the game, for the result json."""


def make(name: str) -> GameEnv:
    from .dodge import DodgeGame
    from .jump import JumpGame
    games = {"jump": JumpGame, "dodge": DodgeGame}
    if name not in games:
        raise KeyError(f"unknown game {name!r}; known: {sorted(games)}")
    return games[name]()
