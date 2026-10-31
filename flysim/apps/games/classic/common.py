"""M7f: what the eight hand-made classic-style games share (docs/m7f-brief.md §1).

Every game is a ``GameEnv`` (``reset(seed)``, ``step(action)``, ``describe()``) seen flat from
above or from the side — no perspective. One house style for all of them, never a per-game
choice:

* background 190, things that do not move (walls, lane lines, bricks, ring ropes) 110, things that
  move (balls, shots, enemies, cars, bombs, trees, flags, the player) 25 — ``base.SKY``,
  ``base.GROUND`` and ``base.OBJECT``. Edges are drawn with ``base.fill_rect`` (fractional
  coverage). This is the house rule of the first rung: on it our detector (growth of the dark
  area) sees things appear and grow; on Atari's dark backgrounds it would not.
* actions: NOOP / JUMP = the original's UP (does nothing where UP did nothing) / LEFT / RIGHT.
  What the fire button did happens by itself on a fixed period (a game constant), the same for
  every player.
* 3,600 ticks (60 s) or game over, whichever comes first; three lives where the original had
  lives; one ``self.rng`` from ``reset``.

**Scene record.** ``scene()`` lists the rectangles the current frame was drawn with, in drawing
order: ``{"rects": [{"kind", "rect": [x0, x1, y0, y1], "value"}, ...], "hud": {score, lives,
tick}}`` (screen pixels, not rounded). ``_render()`` is the background with exactly that list
filled in order, so ``draw(scene()["rects"])`` gives the frame bit for bit (tested every tick).
Agents never receive it; it is for the show page.
"""
from __future__ import annotations

import sys

import numpy as np

from ..agents import Agent
from ..base import FRAME_H, FRAME_W, GROUND, OBJECT, SKY, Action, GameEnv, fill_rect, to_frame

BACKGROUND, FIXED, MOVING = SKY, GROUND, OBJECT     # 190 / 110 / 25
MAX_TICKS = 3600                                    # 60 s
LIVES = 3

Rect = tuple[str, float, float, float, float]       # (kind, x0, x1, y0, y1)


def canvas() -> np.ndarray:
    return np.full((FRAME_H, FRAME_W), BACKGROUND, dtype=np.float32)


def draw(rects: list[dict]) -> np.ndarray:
    """The frame a scene record describes: the background, then each rectangle in order."""
    img = canvas()
    for r in rects:
        fill_rect(img, *r["rect"], r["value"])
    return to_frame(img)


def box(kind: str, cx: float, cy: float, w: float, h: float) -> Rect:
    """A ``w`` x ``h`` rectangle centred on (cx, cy), as Python floats."""
    return (kind, float(cx - w / 2), float(cx + w / 2), float(cy - h / 2), float(cy + h / 2))


def corner(kind: str, x0: float, y0: float, w: float, h: float) -> Rect:
    """A ``w`` x ``h`` rectangle with its top-left corner at (x0, y0), as Python floats."""
    return (kind, float(x0), float(x0 + w), float(y0), float(y0 + h))


def overlaps(a: Rect, b: Rect) -> bool:
    return a[1] < b[2] and b[1] < a[2] and a[3] < b[4] and b[3] < a[4]


def toward(x: float, target: float, speed: float) -> float:
    """Move ``x`` toward ``target`` by at most ``speed``."""
    return x + max(-speed, min(speed, target - x))


def steer(x: float, target: float, speed: float) -> int:
    """The LEFT / RIGHT / NOOP that brings a player moving ``speed`` per tick toward ``target``."""
    if abs(target - x) <= speed / 2:
        return int(Action.NOOP)
    return int(Action.RIGHT) if target > x else int(Action.LEFT)


class ClassicGame(GameEnv):
    """Base of the eight games. Subclasses give ``_layers()`` (the fixed rectangles, drawn first,
    and the moving ones), the rules (``_reset_state``, ``_advance``), the text fields below and
    ``oracle()`` — the cheating reference player's choice, read from game state."""

    max_ticks = MAX_TICKS
    title_ko = ""
    genre_ko = ""
    player_ko = ""                       # what the player does
    jump_effect = "nothing"              # what JUMP (= the original's UP) does
    left_effect = "move left"
    right_effect = "move right"
    automatic = "nothing"                # what happens by itself instead of the fire button
    score_rule = ""
    end_rule = ""
    has_lives = False
    kinds: tuple[str, ...] = ()

    def __init__(self):
        super().__init__()
        self._fixed_key: tuple | None = None
        self._fixed_img: np.ndarray | None = None

    # -- subclass hooks --------------------------------------------------------------------
    def _layers(self) -> tuple[list[Rect], list[Rect]]:
        """(fixed rectangles, moving rectangles), each in drawing order."""
        raise NotImplementedError

    def lives(self) -> int | None:
        return None

    def oracle(self) -> int:
        raise NotImplementedError

    # -- drawing ---------------------------------------------------------------------------
    def scene(self) -> dict:
        fixed, moving = self._layers()
        rects = [{"kind": k, "rect": [x0, x1, y0, y1], "value": FIXED} for k, x0, x1, y0, y1 in fixed]
        rects += [{"kind": k, "rect": [x0, x1, y0, y1], "value": MOVING} for k, x0, x1, y0, y1 in moving]
        return {"rects": rects, "hud": {"score": int(self.score), "lives": self.lives(), "tick": int(self.tick)}}

    def _render(self) -> np.ndarray:
        fixed, moving = self._layers()
        key = tuple(fixed)
        if key != self._fixed_key:          # the fixed layer is redrawn only when it changes
            img = canvas()
            for _, x0, x1, y0, y1 in fixed:
                fill_rect(img, x0, x1, y0, y1, FIXED)
            self._fixed_key, self._fixed_img = key, img
        img = self._fixed_img.copy()
        for _, x0, x1, y0, y1 in moving:
            fill_rect(img, x0, x1, y0, y1, MOVING)
        return to_frame(img)

    # -- rules for the result json ----------------------------------------------------------
    def constants(self) -> dict:
        """Every rule constant of the game's module (upper-case names), JSON-ready."""
        mod = sys.modules[type(self).__module__]
        shared = vars(sys.modules[__name__])          # FRAME_W etc. are not this game's rules
        out = {}
        for k, v in vars(mod).items():
            if k.isupper() and not k.startswith("_") and k not in shared \
                    and isinstance(v, (int, float, str, tuple)):
                out[k] = list(v) if isinstance(v, tuple) else v
        return out

    def describe(self) -> dict:
        return {"game": self.name, "title_ko": self.title_ko, "genre_ko": self.genre_ko,
                "player": self.player_ko, "view": "2D, flat (no perspective)",
                "screen": {"size": [FRAME_H, FRAME_W], "background": BACKGROUND, "fixed": FIXED,
                           "moving": MOVING, "edges": "base.fill_rect (fractional coverage)"},
                "actions": {"NOOP": "nothing", "JUMP": self.jump_effect, "LEFT": self.left_effect,
                            "RIGHT": self.right_effect},
                "automatic": self.automatic, "score": self.score_rule, "ends": self.end_rule,
                "lives": LIVES if self.has_lives else None, "max_ticks": self.max_ticks,
                "kinds": list(self.kinds), "constants": self.constants()}


class ClassicOracle(Agent):
    """**Reads the game's internal state** (cheats): the reference ceiling of the brief's tuning
    check and of the score table, never a comparison baseline. Each game's rule is its
    ``oracle()`` method."""

    name = "state-oracle"

    def __init__(self, env: ClassicGame):
        self.env = env

    def act(self, frame: np.ndarray) -> int:
        return int(self.env.oracle())

    def describe(self) -> dict:
        return {"agent": self.name, "reads": "game state (not pixels)",
                "role": "reference ceiling only, not a baseline"}
