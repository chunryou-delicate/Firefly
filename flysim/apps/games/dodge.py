"""M7b game 2 — "dodge" (피하기).

A tall pillar comes at the player from slightly to the left or slightly to the right; step to
the opposite side so it passes by. Score = pillars avoided. Touching one ends the game. On
screen the pillar grows in the half it comes from.

Rules (all constants fixed here, before any agent was run):

* A pillar appears at distance ``D_SPAWN`` after a random wait (seeded), on a random side
  (seeded), ``SPAWN_OFFSET`` to the left or right of where the player stands at that moment, so
  standing still never avoids it. Pillar and player half-widths add up to more than the offset:
  the player has to move at least ``OBJ_HALF_WIDTH + PLAYER_HALF_WIDTH - SPAWN_OFFSET`` away.
* Collision uses the same swept window as "jump": distance in ``[D_HIT - DEPTH, D_HIT]`` and
  lateral overlap.
* LEFT / RIGHT move the player ``PLAYER_SPEED`` per tick, within ``±X_LIMIT``. JUMP does
  nothing here (the pillar is taller than any jump).
* Speed rises stepwise as in "jump".

Geometry note: because screen offset and screen width both scale with 1/distance, whether the
pillar straddles the screen centre does not depend on its distance. With the player standing
still it stays wholly inside one half (``SPAWN_OFFSET > OBJ_HALF_WIDTH``).
"""
from __future__ import annotations

from .base import Action, GameEnv, background, draw_block, to_frame

D_SPAWN = 40.0
D_HIT = 1.5
DEPTH = 1.0
SPEED0 = 0.30
SPEED_FACTOR = 1.15
SPEED_EVERY = 4
OBJ_HALF_WIDTH = 0.5
OBJ_HEIGHT = 2.5
SPAWN_OFFSET = 0.55
PLAYER_HALF_WIDTH = 0.35       # -> the player must move >= 0.30 away (12 ticks at full speed)
PLAYER_SPEED = 0.025
X_LIMIT = 1.5
WAIT_MIN, WAIT_MAX = 20, 80


class DodgeGame(GameEnv):
    name = "dodge"

    def _reset_state(self) -> None:
        self.x = 0.0                   # player lateral position
        self.obj_x: float | None = None
        self.dist: float | None = None
        self.wait = int(self.rng.integers(WAIT_MIN, WAIT_MAX + 1))
        self.avoided = 0

    @property
    def speed(self) -> float:
        return SPEED0 * SPEED_FACTOR ** (self.avoided // SPEED_EVERY)

    def _advance(self, action: Action) -> tuple[int, bool]:
        if action == Action.LEFT:
            self.x = max(-X_LIMIT, self.x - PLAYER_SPEED)
        elif action == Action.RIGHT:
            self.x = min(X_LIMIT, self.x + PLAYER_SPEED)
        if self.dist is None:
            self.wait -= 1
            if self.wait <= 0:
                side = 1.0 if self.rng.integers(0, 2) else -1.0
                self.obj_x = self.x + side * SPAWN_OFFSET
                self.dist = D_SPAWN
            return 0, False
        old = self.dist
        self.dist = old - self.speed
        overlap = abs(self.obj_x - self.x) < OBJ_HALF_WIDTH + PLAYER_HALF_WIDTH
        if self.dist <= D_HIT and old >= D_HIT - DEPTH and overlap:
            return 0, True
        if self.dist < D_HIT - DEPTH:
            self.avoided += 1
            self.dist = None
            self.obj_x = None
            self.wait = int(self.rng.integers(WAIT_MIN, WAIT_MAX + 1))
            return 1, False
        return 0, False

    def _render(self):
        img = background()
        if self.dist is not None:
            dx = self.obj_x - self.x
            draw_block(img, dx - OBJ_HALF_WIDTH, dx + OBJ_HALF_WIDTH, OBJ_HEIGHT, self.dist)
        return to_frame(img)

    def status(self) -> dict:
        """For the human-facing export only (agents never receive this)."""
        return {"x": round(self.x, 3), "speed": self.speed,
                "side": None if self.obj_x is None else ("right" if self.obj_x > self.x else "left")}

    def describe(self) -> dict:
        return {"game": self.name, "D_SPAWN": D_SPAWN, "D_HIT": D_HIT, "DEPTH": DEPTH,
                "SPEED0": SPEED0, "SPEED_FACTOR": SPEED_FACTOR, "SPEED_EVERY": SPEED_EVERY,
                "OBJ_HALF_WIDTH": OBJ_HALF_WIDTH, "OBJ_HEIGHT": OBJ_HEIGHT,
                "SPAWN_OFFSET": SPAWN_OFFSET, "PLAYER_HALF_WIDTH": PLAYER_HALF_WIDTH,
                "PLAYER_SPEED": PLAYER_SPEED, "X_LIMIT": X_LIMIT, "WAIT": [WAIT_MIN, WAIT_MAX],
                "max_ticks": self.max_ticks, "score": "pillars avoided",
                "ends": "touching a pillar, or max_ticks"}
