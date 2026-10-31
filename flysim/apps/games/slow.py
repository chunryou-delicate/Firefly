"""M7g: the slow variants of the first rung — the same games, obstacles coming slower (docs/m7g-brief.md §2).

Fixed before any slow game was played (brief §2, nothing outside this table exists):

| name          | parent   | ``SPEED0`` (= parent / s) | ``max_ticks`` |
|---------------|----------|---------------------------|---------------|
| ``dodge_v2_s2`` | dodge_v2 | 0.15                    | 7,200 (2 min) |
| ``dodge_v2_s4`` | dodge_v2 | 0.075                   | 7,200         |
| ``dodge_v2_s8`` | dodge_v2 | 0.0375                  | 7,200         |
| ``jump_s2``     | jump     | 0.15                    | 7,200         |
| ``jump_s4``     | jump     | 0.075                   | 7,200         |

* **Only two things differ from the parent:** the starting speed is divided by ``s`` (the speed-up
  rule — x1.15 every 4 obstacles — is the parent's), and the episode cap is 7,200 ticks. Every
  other rule, the renderer and the random draws are the parent's: each class subclasses its
  parent and overrides nothing but ``speed``, ``name`` and ``max_ticks``, so for the same seed and
  the same actions the waits and the sides are drawn in the same order (tests/test_m7g.py).
* **No ``jump_s8``** (decided from the constants before any run): at 0.0375 per tick a barrier
  needs 26.7 ticks to cross the collision window of depth 1.0, and a jump stays above the
  barrier height 0.6 for 27 ticks (u in (0.113, 0.887) of the 36-tick parabola), so even a player
  reading the game state would have to hit a 0-1 tick margin. Dodging only gets easier when
  slower, so dodge keeps ``s8``.
* ``base.make`` is not touched (another window edits its table): use ``make_slow`` / ``VARIANTS``.
"""
from __future__ import annotations

from .base import GameEnv
from .dodge import SPEED0 as DODGE_SPEED0
from .dodge import SPEED_EVERY as DODGE_SPEED_EVERY
from .dodge import SPEED_FACTOR as DODGE_SPEED_FACTOR
from .dodge import DodgeV2Game
from .jump import SPEED0 as JUMP_SPEED0
from .jump import SPEED_EVERY as JUMP_SPEED_EVERY
from .jump import SPEED_FACTOR as JUMP_SPEED_FACTOR
from .jump import JumpGame

SLOW_MAX_TICKS = 7_200            # 2 min: a safety cap that bounds the fly's simulated time (brief §2)
PARENTS = {"jump": JumpGame, "dodge_v2": DodgeV2Game}


class SlowJump(JumpGame):
    """Jump with the starting speed divided by ``speed_divisor``."""

    speed_divisor: int = 1
    parent = "jump"
    max_ticks = SLOW_MAX_TICKS

    @property
    def speed(self) -> float:
        return JUMP_SPEED0 / self.speed_divisor * JUMP_SPEED_FACTOR ** (self.cleared // JUMP_SPEED_EVERY)

    def describe(self) -> dict:
        return _describe(self, JumpGame, JUMP_SPEED0)


class SlowDodgeV2(DodgeV2Game):
    """Dodge v2 (no walls) with the starting speed divided by ``speed_divisor``."""

    speed_divisor: int = 1
    parent = "dodge_v2"
    max_ticks = SLOW_MAX_TICKS

    @property
    def speed(self) -> float:
        return DODGE_SPEED0 / self.speed_divisor * DODGE_SPEED_FACTOR ** (self.avoided // DODGE_SPEED_EVERY)

    def describe(self) -> dict:
        return _describe(self, DodgeV2Game, DODGE_SPEED0)


def _describe(env: GameEnv, parent_cls, parent_speed0: float) -> dict:
    d = parent_cls.describe(env)                       # every parent rule, as the parent reports it
    d.update(game=env.name, parent=env.parent, SPEED0=parent_speed0 / env.speed_divisor,
             parent_SPEED0=parent_speed0, speed_divisor=env.speed_divisor, max_ticks=env.max_ticks,
             parent_max_ticks=parent_cls.max_ticks,
             changed=f"SPEED0 / {env.speed_divisor} and max_ticks {env.max_ticks}; everything else is "
                     f"{env.parent}'s (docs/m7g-brief.md §2)")
    return d


def _variant(base_cls, s: int):
    name = f"{base_cls.parent}_s{s}"
    return type(f"{base_cls.__name__}S{s}", (base_cls,), {"name": name, "speed_divisor": s,
                                                          "__doc__": f"{name}: {base_cls.parent}, SPEED0 / {s}."})


# brief §2: the whole list. Nothing outside it is played.
VARIANTS: dict[str, type] = {cls.name: cls for cls in (
    _variant(SlowDodgeV2, 2), _variant(SlowDodgeV2, 4), _variant(SlowDodgeV2, 8),
    _variant(SlowJump, 2), _variant(SlowJump, 4))}


def make_slow(name: str) -> GameEnv:
    if name not in VARIANTS:
        raise KeyError(f"unknown slow variant {name!r}; known: {sorted(VARIANTS)}")
    return VARIANTS[name]()
