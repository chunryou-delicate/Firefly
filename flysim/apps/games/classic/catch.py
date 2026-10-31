"""M7f game "catch" (폭탄 받기) — catching style, seen from the side.

A bomber wanders left and right along the top and drops bombs; the player's bucket slides
along the bottom and catches them. Score = bombs caught. Three misses end the game. Rounds
start by themselves ``ROUND_WAIT`` ticks after the previous one ends.

Rules (all constants fixed here, docs/m7f-brief.md §2):

* Round level ``n`` (0 first): ``BOMBS0 + n * BOMBS_STEP`` bombs, one every
  ``max(DROP_MIN, DROP_EVERY0 - n * DROP_STEP)`` ticks, falling ``FALL0 + n * FALL_STEP`` px per
  tick; the bomber moves ``BOMBER_SPEED0 + n * BOMBER_STEP`` px per tick and turns around at
  random (probability ``TURN_P`` per tick, seeded) or at a side. Levels stop rising at
  ``LEVEL_MAX``.
* A bomb that passes the bucket is a miss: every falling bomb goes off, the round ends and the
  same level starts again after ``ROUND_WAIT``. A round whose bombs were all caught moves up a
  level.
* JUMP does nothing (the original had no UP move).
"""
from __future__ import annotations

from ..base import FRAME_W, Action
from . import common as C

BOMBER_W, BOMBER_H = 8.0, 6.0
BOMBER_Y = 4.0                   # top edge
BOMBER_X = (6.0, 78.0)           # range of the bomber's centre
BUCKET_W, BUCKET_H = 10.0, 4.0
BUCKET_Y = 74.0                  # top edge
BUCKET_SPEED = 2.0
BOMB_W, BOMB_H = 3.0, 4.0
TURN_P = 0.03
ROUND_WAIT = 60                  # automatic round start
BOMBS0, BOMBS_STEP = 10, 5
DROP_EVERY0, DROP_STEP, DROP_MIN = 30, 3, 10
FALL0, FALL_STEP = 0.6, 0.1
BOMBER_SPEED0, BOMBER_STEP = 0.6, 0.15
LEVEL_MAX = 7
MISSES = 3

_B_LO, _B_HI = BUCKET_W / 2, FRAME_W - BUCKET_W / 2


class CatchGame(C.ClassicGame):
    name = "catch"
    title_ko = "폭탄 받기"
    genre_ko = "받기형"
    player_ko = "위에서 좌우로 움직이는 폭격수가 떨어뜨리는 폭탄을 아래 바구니로 받는다"
    automatic = f"round start, {ROUND_WAIT} ticks after the start and after every round"
    score_rule = "bombs caught"
    end_rule = f"{MISSES} misses, or max_ticks"
    has_lives = True
    kinds = ("bucket", "bomber", "bomb")

    def _reset_state(self) -> None:
        self.bx = FRAME_W / 2            # bucket centre
        self.mx = FRAME_W / 2            # bomber centre
        self.mdir = 1.0 if self.rng.integers(0, 2) else -1.0
        self.bombs: list[list[float]] = []      # [x, y] top-left
        self.level = 0
        self.misses = 0
        self.in_round = False
        self.round_wait = ROUND_WAIT
        self.to_drop = 0
        self.drop_wait = 0

    def lives(self) -> int:
        return MISSES - self.misses

    @property
    def fall(self) -> float:
        return FALL0 + self.level * FALL_STEP

    def _advance(self, action: Action) -> tuple[int, bool]:
        if action == Action.LEFT:
            self.bx = max(_B_LO, self.bx - BUCKET_SPEED)
        elif action == Action.RIGHT:
            self.bx = min(_B_HI, self.bx + BUCKET_SPEED)
        # the bomber wanders
        if self.rng.random() < TURN_P:
            self.mdir = -self.mdir
        self.mx += self.mdir * (BOMBER_SPEED0 + self.level * BOMBER_STEP)
        if self.mx < BOMBER_X[0] or self.mx > BOMBER_X[1]:
            lo, hi = BOMBER_X
            self.mx = 2 * lo - self.mx if self.mx < lo else 2 * hi - self.mx
            self.mdir = -self.mdir
        if not self.in_round:
            self.round_wait -= 1
            if self.round_wait <= 0:
                self.in_round = True
                self.to_drop = BOMBS0 + self.level * BOMBS_STEP
                self.drop_wait = 0
            return 0, False
        # drops
        self.drop_wait -= 1
        if self.to_drop > 0 and self.drop_wait <= 0:
            self.bombs.append([self.mx - BOMB_W / 2, BOMBER_Y + BOMBER_H])
            self.to_drop -= 1
            self.drop_wait = max(DROP_MIN, DROP_EVERY0 - self.level * DROP_STEP)
        # falling, catching, missing
        bucket = C.corner("bucket", self.bx - BUCKET_W / 2, BUCKET_Y, BUCKET_W, BUCKET_H)
        reward, kept = 0, []
        for b in self.bombs:
            b[1] += self.fall
            if C.overlaps(C.corner("bomb", b[0], b[1], BOMB_W, BOMB_H), bucket):
                reward += 1
            elif b[1] > BUCKET_Y + BUCKET_H:
                self.misses += 1
                self.bombs, self.in_round, self.round_wait = [], False, ROUND_WAIT
                return reward, self.misses >= MISSES
            else:
                kept.append(b)
        self.bombs = kept
        if self.to_drop == 0 and not self.bombs:
            self.in_round, self.round_wait = False, ROUND_WAIT
            self.level = min(LEVEL_MAX, self.level + 1)
        return reward, False

    def _layers(self):
        moving = [C.corner("bomber", self.mx - BOMBER_W / 2, BOMBER_Y, BOMBER_W, BOMBER_H)]
        moving += [C.corner("bomb", b[0], b[1], BOMB_W, BOMB_H) for b in self.bombs]
        moving.append(C.corner("bucket", self.bx - BUCKET_W / 2, BUCKET_Y, BUCKET_W, BUCKET_H))
        return [], moving

    def oracle(self) -> int:
        """Stand under the lowest falling bomb; between bombs, under the bomber."""
        target = self.mx
        if self.bombs:
            low = max(self.bombs, key=lambda b: b[1])
            target = low[0] + BOMB_W / 2
        return C.steer(self.bx, target, BUCKET_SPEED)
