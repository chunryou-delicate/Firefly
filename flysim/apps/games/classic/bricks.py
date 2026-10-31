"""M7f game "bricks" (벽돌 깨기) — wall-of-bricks style, seen from the side.

A paddle along the bottom bounces the ball up into a wall of ``BRICK_ROWS`` x ``BRICK_COLS``
bricks under the top wall. Score = bricks broken. Losing the ball three times ends the game.
The serve happens by itself ``SERVE_WAIT`` ticks after the start and after every lost ball.

Rules (all constants fixed here, docs/m7f-brief.md §2):

* Ball and paddle as in "paddle": ``BALL_VY`` vertically, ``ENGLISH`` from where it meets the
  paddle; side and top walls reflect it.
* A ball that touches bricks breaks one — the one whose centre is nearest the ball's, ties in
  row-major order — and turns back vertically (sideways speed kept). One brick per tick.
* When the last brick breaks, a full wall comes back.
* JUMP does nothing (the original had no UP move).
"""
from __future__ import annotations

import numpy as np

from ..base import FRAME_H, FRAME_W, Action
from . import common as C
from .paddle import landing_x

WALL_W = 2.0                     # side walls x in [0, 2) / [82, 84); top wall y in [0, 2)
BRICK_ROWS, BRICK_COLS = 6, 12
BRICK_TOP = 12.0                 # top edge of the first row
BRICK_PITCH_Y = 4.0
BRICK_H = 3.0
BRICK_GAP = 1.0                  # horizontal gap between neighbouring bricks
PADDLE_W, PADDLE_H = 12.0, 2.0
PADDLE_Y = 78.0
PADDLE_SPEED = 1.5
BALL = 2.0
BALL_VY = 1.0
ENGLISH = 1.2
SERVE_VX = 0.8
SERVE_X, SERVE_Y = 42.0, 44.0    # serve position (ball centre), moving down
SERVE_WAIT = 60
BALLS = 3

_PITCH_X = (FRAME_W - 2 * WALL_W) / BRICK_COLS
_X_LO, _X_HI = WALL_W + PADDLE_W / 2, FRAME_W - WALL_W - PADDLE_W / 2
_B_LO, _B_HI = WALL_W + BALL / 2, FRAME_W - WALL_W - BALL / 2
_B_TOP = WALL_W + BALL / 2


def brick_rect(r: int, c: int) -> C.Rect:
    return C.corner("brick", WALL_W + c * _PITCH_X + BRICK_GAP / 2, BRICK_TOP + r * BRICK_PITCH_Y,
                    _PITCH_X - BRICK_GAP, BRICK_H)


_BRICKS = [[brick_rect(r, c) for c in range(BRICK_COLS)] for r in range(BRICK_ROWS)]


class BricksGame(C.ClassicGame):
    name = "bricks"
    title_ko = "벽돌 깨기"
    genre_ko = "벽돌형"
    player_ko = "아래 막대로 공을 튕겨 위의 벽돌(6줄 x 12개)을 깬다"
    automatic = f"serve, {SERVE_WAIT} ticks after the start and after every lost ball"
    score_rule = "bricks broken"
    end_rule = f"{BALLS} balls lost, or max_ticks"
    has_lives = True
    kinds = ("paddle", "ball", "brick", "wall")

    def _reset_state(self) -> None:
        self.px = FRAME_W / 2
        self.ball: list[float] | None = None
        self.serve_wait = SERVE_WAIT
        self.lost = 0
        self.bricks = np.ones((BRICK_ROWS, BRICK_COLS), dtype=bool)
        self.walls_cleared = 0

    def lives(self) -> int:
        return BALLS - self.lost

    def _advance(self, action: Action) -> tuple[int, bool]:
        if action == Action.LEFT:
            self.px = max(_X_LO, self.px - PADDLE_SPEED)
        elif action == Action.RIGHT:
            self.px = min(_X_HI, self.px + PADDLE_SPEED)
        if self.ball is None:
            self.serve_wait -= 1
            if self.serve_wait <= 0:
                self.ball = [SERVE_X, SERVE_Y, float(self.rng.uniform(-SERVE_VX, SERVE_VX)), BALL_VY]
            return 0, False
        x, y, vx, vy = self.ball
        x += vx
        if x < _B_LO or x > _B_HI:
            x, vx = (2 * _B_LO - x) if x < _B_LO else (2 * _B_HI - x), -vx
        y_old, y = y, y + vy
        if y < _B_TOP:
            y, vy = 2 * _B_TOP - y, -vy
        reward = 0
        ball = C.box("ball", x, y, BALL, BALL)
        hits = []
        for r, c in zip(*np.nonzero(self.bricks)):
            b = _BRICKS[r][c]
            if C.overlaps(ball, b):
                hits.append((abs(x - (b[1] + b[2]) / 2) + abs(y - (b[3] + b[4]) / 2), int(r), int(c)))
        if hits:
            _, r, c = min(hits)
            self.bricks[r, c] = False
            y, vy = y_old, -vy                       # back out of the brick and turn
            reward = 1
            if not self.bricks.any():
                self.bricks[:] = True
                self.walls_cleared += 1
        half = PADDLE_W / 2 + BALL / 2
        if vy > 0 and y_old + BALL / 2 < PADDLE_Y <= y + BALL / 2 and abs(x - self.px) <= half:
            vy, vx, y = -BALL_VY, ENGLISH * (x - self.px) / half, PADDLE_Y - BALL / 2
        self.ball = [x, y, vx, vy]
        if y - BALL / 2 > FRAME_H:
            self.lost += 1
            self.ball, self.serve_wait = None, SERVE_WAIT
            return reward, self.lost >= BALLS
        return reward, False

    def _layers(self):
        fixed = [C.corner("wall", 0.0, 0.0, WALL_W, FRAME_H),
                 C.corner("wall", FRAME_W - WALL_W, 0.0, WALL_W, FRAME_H),
                 C.corner("wall", WALL_W, 0.0, FRAME_W - 2 * WALL_W, WALL_W)]
        fixed += [_BRICKS[r][c] for r in range(BRICK_ROWS) for c in range(BRICK_COLS) if self.bricks[r, c]]
        moving = [C.corner("paddle", self.px - PADDLE_W / 2, PADDLE_Y, PADDLE_W, PADDLE_H)]
        if self.ball is not None:
            moving.append(C.box("ball", self.ball[0], self.ball[1], BALL, BALL))
        return fixed, moving

    def oracle(self) -> int:
        """Stand under the point where the ball will meet the paddle line (walls folded in)."""
        target = FRAME_W / 2
        if self.ball is not None and self.ball[3] > 0:
            target = landing_x(self.ball, PADDLE_Y - BALL / 2, _B_LO, _B_HI)
        return C.steer(self.px, target, PADDLE_SPEED)
