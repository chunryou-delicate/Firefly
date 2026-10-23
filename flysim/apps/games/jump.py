"""M7b game 1 — "jump" (뛰어넘기).

A low barrier comes straight at the player from ahead; jump so it passes under you. Score =
barriers cleared. Touching one ends the game. The camera looks ahead, so an approaching barrier
grows on screen, centred (it covers both screen halves equally).

Rules (all constants fixed here, before any agent was run):

* A barrier appears at distance ``D_SPAWN`` after a random wait of ``WAIT_MIN..WAIT_MAX`` ticks
  (seeded) and approaches at ``speed`` units per tick.
* It is "at the player" while its distance is in ``[D_HIT - DEPTH, D_HIT]``. If during any tick
  the barrier overlaps that window (swept, so a fast barrier cannot tunnel through) and the
  player's height is below ``OBS_HEIGHT``, the game ends.
* A jump lasts ``JUMP_TICKS`` with a parabolic height peaking at ``JUMP_PEAK``; after landing
  the player cannot jump for ``LAND_COOLDOWN`` ticks. JUMP while airborne or cooling down does
  nothing. Without the cooldown, pressing JUMP all the time would clear most barriers.
* Speed rises stepwise: ``speed = SPEED0 * SPEED_FACTOR ** (cleared // SPEED_EVERY)``.
* LEFT/RIGHT do nothing in this game (the action space is shared with "dodge").
"""
from __future__ import annotations

from .base import Action, GameEnv, background, draw_block, to_frame

D_SPAWN = 40.0
D_HIT = 1.5
DEPTH = 1.0
SPEED0 = 0.30                  # units per tick: (40 - 1.5) / 0.30 = 128 ticks ~ 2.1 s at the start
SPEED_FACTOR = 1.15
SPEED_EVERY = 4                # barriers per speed step
OBS_HALF_WIDTH = 0.8
OBS_HEIGHT = 0.6
JUMP_TICKS = 36                # 0.6 s in the air
JUMP_PEAK = 1.5
LAND_COOLDOWN = 18             # 0.3 s on the ground before the next jump
WAIT_MIN, WAIT_MAX = 20, 80    # ticks between barriers (seeded)


def jump_height(t: int) -> float:
    """Height after ``t`` ticks in the air (0 on the ground)."""
    if t < 0 or t >= JUMP_TICKS:
        return 0.0
    u = t / JUMP_TICKS
    return 4.0 * JUMP_PEAK * u * (1.0 - u)


class JumpGame(GameEnv):
    name = "jump"

    def _reset_state(self) -> None:
        self.dist: float | None = None
        self.wait = int(self.rng.integers(WAIT_MIN, WAIT_MAX + 1))
        self.cleared = 0
        self.jump_t = -1               # ticks since take-off; -1 = on the ground
        self.cooldown = 0

    @property
    def speed(self) -> float:
        return SPEED0 * SPEED_FACTOR ** (self.cleared // SPEED_EVERY)

    @property
    def height(self) -> float:
        return jump_height(self.jump_t)

    @property
    def airborne(self) -> bool:
        return self.jump_t >= 0

    def _advance(self, action: Action) -> tuple[int, bool]:
        # player
        if action == Action.JUMP and not self.airborne and self.cooldown == 0:
            self.jump_t = 0
        elif self.airborne:
            self.jump_t += 1
            if self.jump_t >= JUMP_TICKS:
                self.jump_t = -1
                self.cooldown = LAND_COOLDOWN
        elif self.cooldown > 0:
            self.cooldown -= 1
        # barrier
        if self.dist is None:
            self.wait -= 1
            if self.wait <= 0:
                self.dist = D_SPAWN
            return 0, False
        old = self.dist
        self.dist = old - self.speed
        if self.dist <= D_HIT and old >= D_HIT - DEPTH and self.height < OBS_HEIGHT:
            return 0, True
        if self.dist < D_HIT - DEPTH:
            self.cleared += 1
            self.dist = None
            self.wait = int(self.rng.integers(WAIT_MIN, WAIT_MAX + 1))
            return 1, False
        return 0, False

    def _render(self):
        img = background()
        if self.dist is not None:
            draw_block(img, -OBS_HALF_WIDTH, OBS_HALF_WIDTH, OBS_HEIGHT, self.dist)
        return to_frame(img)

    def status(self) -> dict:
        """For the human-facing export only (agents never receive this)."""
        return {"airborne": self.airborne, "height": round(self.height, 3), "speed": self.speed}

    def describe(self) -> dict:
        return {"game": self.name, "D_SPAWN": D_SPAWN, "D_HIT": D_HIT, "DEPTH": DEPTH,
                "SPEED0": SPEED0, "SPEED_FACTOR": SPEED_FACTOR, "SPEED_EVERY": SPEED_EVERY,
                "OBS_HALF_WIDTH": OBS_HALF_WIDTH, "OBS_HEIGHT": OBS_HEIGHT,
                "JUMP_TICKS": JUMP_TICKS, "JUMP_PEAK": JUMP_PEAK, "LAND_COOLDOWN": LAND_COOLDOWN,
                "WAIT": [WAIT_MIN, WAIT_MAX], "max_ticks": self.max_ticks,
                "score": "barriers cleared", "ends": "touching a barrier, or max_ticks"}
