"""M7f game "boxing" (권투) — boxing style, the ring seen from above.

The player's boxer starts at the bottom of the ring facing up, the computer's at the top facing
down. Punches go off by themselves for both every ``PUNCH_EVERY`` ticks, on the same tick, and
land when the other boxer is in front, within ``REACH`` and lined up within ``ALIGN``. Score =
the player's punches that landed. Only the clock ends the game.

Rules (all constants fixed here, docs/m7f-brief.md §2):

* The player moves left, right and — with JUMP, the original's UP — up, ``PLAYER_SPEED`` px per
  tick. There is no way down (the fly's muscles have none); only a punch knocks the player back.
* The computer walks at ``CPU_SPEED`` toward a random point of the upper ring ``CPU_X`` x
  ``CPU_Y`` (seeded) and picks a new one when it gets there or after ``CPU_REPICK`` ticks.
* Boxers cannot overlap: a move into the other boxer is not made.
* The conditions are symmetric, so when one punch lands the other does too. A boxer that is hit
  is knocked ``KNOCK`` px back (player down, computer up), within the ring.
"""
from __future__ import annotations

from ..base import Action
from . import common as C

RING = (4.0, 80.0)               # inside of the ropes, both axes
ROPE = 1.0                       # rope thickness, drawn just outside RING
BOXER = 8.0                      # side of a boxer (square)
PLAYER_START = (42.0, 68.0)      # centre
CPU_START = (42.0, 16.0)
PLAYER_SPEED = 1.0
CPU_SPEED = 0.5
CPU_X = (8.0, 76.0)              # range of the computer's target point (centre)
CPU_Y = (8.0, 44.0)
CPU_REPICK = 90
PUNCH_EVERY = 30                 # automatic punches, both boxers on the same tick
REACH = 6.0                      # largest gap between the facing edges for a punch to land
ALIGN = 4.0                      # largest sideways offset of the centres
KNOCK = 6.0

_LO, _HI = RING[0] + BOXER / 2, RING[1] - BOXER / 2      # centre limits


class BoxingGame(C.ClassicGame):
    name = "boxing"
    title_ko = "권투"
    genre_ko = "권투형"
    player_ko = "위에서 본 링. 상대에게 다가가 맞춘다. 아래로는 못 움직인다(근육에 없다)"
    jump_effect = "move up"
    automatic = f"punch, both boxers every {PUNCH_EVERY} ticks (lands within reach and lined up)"
    score_rule = "the player's punches that landed"
    end_rule = "max_ticks only"
    kinds = ("boxer", "cpu_boxer", "ring")

    def _reset_state(self) -> None:
        self.p = list(PLAYER_START)
        self.c = list(CPU_START)
        self.target = self._point()
        self.repick = CPU_REPICK
        self.cpu_hits = 0

    def _point(self) -> list[float]:
        return [float(self.rng.uniform(*CPU_X)), float(self.rng.uniform(*CPU_Y))]

    @staticmethod
    def _rect(kind: str, pos: list[float]) -> C.Rect:
        return C.box(kind, pos[0], pos[1], BOXER, BOXER)

    def _move(self, me: list[float], other: list[float], dx: float, dy: float) -> None:
        """Move one axis at a time; a step into the other boxer or out of the ring is not made."""
        for axis, d in ((0, dx), (1, dy)):
            if d == 0:
                continue
            new = list(me)
            new[axis] = min(_HI, max(_LO, me[axis] + d))
            if not C.overlaps(self._rect("", new), self._rect("", other)):
                me[:] = new

    def in_range(self) -> bool:
        gap = (self.p[1] - BOXER / 2) - (self.c[1] + BOXER / 2)
        return 0.0 <= gap <= REACH and abs(self.p[0] - self.c[0]) <= ALIGN

    def _advance(self, action: Action) -> tuple[int, bool]:
        if action == Action.LEFT:
            self._move(self.p, self.c, -PLAYER_SPEED, 0.0)
        elif action == Action.RIGHT:
            self._move(self.p, self.c, PLAYER_SPEED, 0.0)
        elif action == Action.JUMP:
            self._move(self.p, self.c, 0.0, -PLAYER_SPEED)
        self.repick -= 1
        if self.repick <= 0 or (abs(self.target[0] - self.c[0]) <= CPU_SPEED
                                and abs(self.target[1] - self.c[1]) <= CPU_SPEED):
            self.target, self.repick = self._point(), CPU_REPICK
        self._move(self.c, self.p, C.toward(self.c[0], self.target[0], CPU_SPEED) - self.c[0],
                   C.toward(self.c[1], self.target[1], CPU_SPEED) - self.c[1])
        if (self.tick + 1) % PUNCH_EVERY == 0 and self.in_range():
            self.cpu_hits += 1
            self.p[1] = min(_HI, self.p[1] + KNOCK)
            self.c[1] = max(_LO, self.c[1] - KNOCK)
            return 1, False
        return 0, False

    def _layers(self):
        lo, hi = RING[0] - ROPE, RING[1] + ROPE
        fixed = [C.corner("ring", lo, lo, hi - lo, ROPE), C.corner("ring", lo, RING[1], hi - lo, ROPE),
                 C.corner("ring", lo, RING[0], ROPE, RING[1] - RING[0]),
                 C.corner("ring", RING[1], RING[0], ROPE, RING[1] - RING[0])]
        return fixed, [self._rect("cpu_boxer", self.c), self._rect("boxer", self.p)]

    def oracle(self) -> int:
        """Line up under the computer, then close in to just inside reach."""
        dx = self.c[0] - self.p[0]
        gap = (self.p[1] - BOXER / 2) - (self.c[1] + BOXER / 2)
        if abs(dx) > ALIGN / 2:
            return int(Action.RIGHT) if dx > 0 else int(Action.LEFT)
        if gap > REACH / 2:
            return int(Action.JUMP)
        return int(Action.NOOP)
