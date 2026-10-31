"""M7f game "formation" (외계 편대 막기) — formation shooting style, seen from the side.

A formation of ``ALIEN_ROWS`` x ``ALIEN_COLS`` aliens marches sideways, steps down ``DROP``
whenever it reaches a side, and drops bombs. The player's cannon slides along the bottom and
fires by itself — one shot on screen at a time. Score = aliens hit. The game ends when three
bombs have hit the cannon or the formation reaches the cannon's row.

Rules (all constants fixed here, docs/m7f-brief.md §2):

* Marching speed starts each wave at ``MARCH0`` and rises by ``MARCH_GAIN`` per alien hit in
  that wave. When the whole formation is gone, a new one starts from the top.
* Every ``BOMB_EVERY`` ticks the lowest alien of a random column (seeded) drops a bomb, unless
  ``MAX_BOMBS`` are already falling.
* Automatic fire: a shot leaves the cannon ``FIRE_EVERY`` ticks after the previous one left, as
  soon as no shot is on screen. Shots and bombs pass through each other.
* A bomb on the cannon costs a life and clears the falling bombs.
* JUMP does nothing (the original had no UP move).
"""
from __future__ import annotations

import numpy as np

from ..base import FRAME_H, FRAME_W, Action
from . import common as C

CANNON_W, CANNON_H = 7.0, 4.0
CANNON_Y = 76.0                  # top edge of the cannon
CANNON_SPEED = 1.0
ALIEN_ROWS, ALIEN_COLS = 4, 6
ALIEN_W, ALIEN_H = 6.0, 4.0
ALIEN_PITCH_X, ALIEN_PITCH_Y = 10.0, 8.0
FORMATION_X0, FORMATION_Y0 = 13.0, 8.0     # top-left alien's corner at the start of a wave
MARCH0 = 0.2                     # px per tick at the start of a wave
MARCH_GAIN = 0.02                # + px per tick per alien hit in the wave
MARCH_MARGIN = 2.0               # the formation turns when an alien comes this close to a side
DROP = 4.0                       # px down at every turn
SHOT_W, SHOT_H = 1.0, 4.0
SHOT_SPEED = 3.0
FIRE_EVERY = 20                  # automatic fire period (ticks), one shot on screen at a time
BOMB_W, BOMB_H = 1.0, 3.0
BOMB_SPEED = 1.0
BOMB_EVERY = 40
MAX_BOMBS = 2

_C_LO, _C_HI = CANNON_W / 2, FRAME_W - CANNON_W / 2


class FormationGame(C.ClassicGame):
    name = "formation"
    title_ko = "외계 편대 막기"
    genre_ko = "편대 사격형"
    player_ko = "아래 포대를 좌우로 움직인다. 편대가 옆으로 행진하다 벽에서 한 줄 내려오고, 아래로 폭탄을 떨어뜨린다"
    automatic = f"fire, every {FIRE_EVERY} ticks while no shot is on screen"
    score_rule = "aliens hit"
    end_rule = "3 bombs on the cannon, the formation reaching the cannon's row, or max_ticks"
    has_lives = True
    kinds = ("cannon", "alien", "shot", "bomb")

    def _reset_state(self) -> None:
        self.cx = FRAME_W / 2
        self.lives_left = C.LIVES
        self.shot: list[float] | None = None          # [x, y] top-left corner
        self.fire_wait = FIRE_EVERY
        self.bombs: list[list[float]] = []
        self.bomb_wait = BOMB_EVERY
        self.waves = 0
        self._new_wave()

    def _new_wave(self) -> None:
        self.alive = np.ones((ALIEN_ROWS, ALIEN_COLS), dtype=bool)
        self.fx, self.fy = FORMATION_X0, FORMATION_Y0
        self.dir = 1.0
        self.hit_in_wave = 0

    def lives(self) -> int:
        return self.lives_left

    @property
    def march_speed(self) -> float:
        return MARCH0 + MARCH_GAIN * self.hit_in_wave

    def alien(self, r: int, c: int) -> C.Rect:
        return C.corner("alien", self.fx + c * ALIEN_PITCH_X, self.fy + r * ALIEN_PITCH_Y, ALIEN_W, ALIEN_H)

    def _advance(self, action: Action) -> tuple[int, bool]:
        if action == Action.LEFT:
            self.cx = max(_C_LO, self.cx - CANNON_SPEED)
        elif action == Action.RIGHT:
            self.cx = min(_C_HI, self.cx + CANNON_SPEED)
        reward = 0
        # automatic fire, then the shot
        self.fire_wait -= 1
        if self.shot is None and self.fire_wait <= 0:
            self.shot = [self.cx - SHOT_W / 2, CANNON_Y - SHOT_H]
            self.fire_wait = FIRE_EVERY
        if self.shot is not None:
            self.shot[1] -= SHOT_SPEED
            shot = C.corner("shot", self.shot[0], self.shot[1], SHOT_W, SHOT_H)
            for r in range(ALIEN_ROWS - 1, -1, -1):          # the lowest alien is hit first
                hit = [c for c in range(ALIEN_COLS) if self.alive[r, c] and C.overlaps(shot, self.alien(r, c))]
                if hit:
                    self.alive[r, hit[0]] = False
                    self.hit_in_wave += 1
                    self.shot, reward = None, 1
                    break
            if self.shot is not None and self.shot[1] + SHOT_H < 0:
                self.shot = None
        if not self.alive.any():
            self.waves += 1
            self._new_wave()
        # the formation marches
        cols = np.nonzero(self.alive.any(axis=0))[0]
        self.fx += self.dir * self.march_speed
        left = self.fx + cols[0] * ALIEN_PITCH_X
        right = self.fx + cols[-1] * ALIEN_PITCH_X + ALIEN_W
        if self.dir > 0 and right > FRAME_W - MARCH_MARGIN:
            self.fx -= right - (FRAME_W - MARCH_MARGIN)
            self.dir, self.fy = -1.0, self.fy + DROP
        elif self.dir < 0 and left < MARCH_MARGIN:
            self.fx += MARCH_MARGIN - left
            self.dir, self.fy = 1.0, self.fy + DROP
        rows = np.nonzero(self.alive.any(axis=1))[0]
        if self.fy + rows[-1] * ALIEN_PITCH_Y + ALIEN_H >= CANNON_Y:
            return reward, True
        # bombs
        self.bomb_wait -= 1
        if self.bomb_wait <= 0:
            self.bomb_wait = BOMB_EVERY
            if len(self.bombs) < MAX_BOMBS:
                c = int(cols[int(self.rng.integers(0, len(cols)))])
                r = int(np.nonzero(self.alive[:, c])[0][-1])
                a = self.alien(r, c)
                self.bombs.append([(a[1] + a[2]) / 2 - BOMB_W / 2, a[4]])
        cannon = C.corner("cannon", self.cx - CANNON_W / 2, CANNON_Y, CANNON_W, CANNON_H)
        kept = []
        for b in self.bombs:
            b[1] += BOMB_SPEED
            if C.overlaps(C.corner("bomb", b[0], b[1], BOMB_W, BOMB_H), cannon):
                self.lives_left -= 1
                self.bombs = []
                return reward, self.lives_left <= 0
            if b[1] < FRAME_H:
                kept.append(b)
        self.bombs = kept
        return reward, False

    def _layers(self):
        moving = [self.alien(r, c) for r in range(ALIEN_ROWS) for c in range(ALIEN_COLS) if self.alive[r, c]]
        moving.append(C.corner("cannon", self.cx - CANNON_W / 2, CANNON_Y, CANNON_W, CANNON_H))
        if self.shot is not None:
            moving.append(C.corner("shot", self.shot[0], self.shot[1], SHOT_W, SHOT_H))
        moving += [C.corner("bomb", b[0], b[1], BOMB_W, BOMB_H) for b in self.bombs]
        return [], moving

    def oracle(self) -> int:
        """Step away from a bomb that would land on the cannon soon; otherwise stand under the
        alien the next shot would reach first (its x when the shot gets there)."""
        reach = CANNON_W / 2 + BOMB_W / 2 + 2.0
        for b in sorted(self.bombs, key=lambda b: -b[1]):
            bx = b[0] + BOMB_W / 2
            if abs(bx - self.cx) < reach and b[1] + BOMB_H > CANNON_Y - 40.0:
                away = Action.LEFT if bx > self.cx else Action.RIGHT
                if (away == Action.LEFT and self.cx - CANNON_SPEED < _C_LO) or \
                        (away == Action.RIGHT and self.cx + CANNON_SPEED > _C_HI):
                    away = Action.RIGHT if away == Action.LEFT else Action.LEFT
                return int(away)
        best = None
        for r in range(ALIEN_ROWS):
            for c in range(ALIEN_COLS):
                if not self.alive[r, c]:
                    continue
                a = self.alien(r, c)
                lead = (CANNON_Y - a[4]) / SHOT_SPEED * self.march_speed * self.dir
                x = (a[1] + a[2]) / 2 + lead
                d = abs(x - self.cx)
                if best is None or d < best[0]:
                    best = (d, x)
        return C.steer(self.cx, best[1], CANNON_SPEED) if best else int(Action.NOOP)
