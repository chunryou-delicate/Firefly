"""M7f game "paddle" (공 받아치기) — table-tennis style, seen from above.

The player's paddle runs along the bottom, the computer's along the top; the ball bounces off
the side walls. Score = returns (the ball hitting the player's paddle). Three misses end the
game. The serve happens by itself ``SERVE_WAIT`` ticks after every point, from the centre
toward the player.

Rules (all constants fixed here, docs/m7f-brief.md §2):

* The ball keeps ``BALL_VY`` vertically. Where it meets a paddle sets its sideways speed:
  ``vx = ENGLISH * offset / (PADDLE_W / 2 + BALL / 2)``, offset = ball centre − paddle centre.
* The computer paddle chases the ball's x at no more than ``CPU_SPEED`` (slower than an angled
  ball), so it can be beaten; when it misses, the point is replayed with a new serve.
* JUMP does nothing (the original had no UP move).
"""
from __future__ import annotations

from ..base import FRAME_H, FRAME_W, Action
from . import common as C

WALL_W = 2.0                     # side walls: x in [0, 2) and [82, 84), full height
PADDLE_W, PADDLE_H = 12.0, 2.0
PADDLE_Y = 78.0                  # top edge of the player's paddle
CPU_Y = 4.0                      # top edge of the computer's paddle
PADDLE_SPEED = 1.5               # px per tick (LEFT / RIGHT)
CPU_SPEED = 0.5                  # px per tick, the computer's top speed
BALL = 2.0                       # side of the square ball
BALL_VY = 1.0                    # px per tick
ENGLISH = 1.2                    # largest sideways speed a paddle edge gives
SERVE_VX = 0.8                   # serve: vx uniform in [-SERVE_VX, SERVE_VX], toward the player
SERVE_X, SERVE_Y = 42.0, 42.0    # serve position (ball centre)
SERVE_WAIT = 60                  # ticks from the start / a point to the automatic serve
MISSES = 3

_X_LO, _X_HI = WALL_W + PADDLE_W / 2, FRAME_W - WALL_W - PADDLE_W / 2   # paddle centre limits
_B_LO, _B_HI = WALL_W + BALL / 2, FRAME_W - WALL_W - BALL / 2            # ball centre limits


def _reflect(x: float, lo: float, hi: float) -> tuple[float, bool]:
    if x < lo:
        return 2 * lo - x, True
    if x > hi:
        return 2 * hi - x, True
    return x, False


class PaddleGame(C.ClassicGame):
    name = "paddle"
    title_ko = "공 받아치기"
    genre_ko = "탁구형"
    player_ko = "아래 막대를 좌우로 움직여 공을 받아친다. 위에는 공을 쫓는 컴퓨터 막대(최고 속도 제한)"
    automatic = f"serve, {SERVE_WAIT} ticks after the start and after every point"
    score_rule = "returns (ball hits the player's paddle)"
    end_rule = f"{MISSES} misses, or max_ticks"
    has_lives = True
    kinds = ("paddle", "cpu_paddle", "ball", "wall")

    def _reset_state(self) -> None:
        self.px = FRAME_W / 2            # player paddle centre
        self.cx = FRAME_W / 2            # computer paddle centre
        self.ball: list[float] | None = None    # [x, y, vx, vy], centre
        self.serve_wait = SERVE_WAIT
        self.misses = 0
        self.cpu_misses = 0

    def lives(self) -> int:
        return MISSES - self.misses

    def _advance(self, action: Action) -> tuple[int, bool]:
        if action == Action.LEFT:
            self.px = max(_X_LO, self.px - PADDLE_SPEED)
        elif action == Action.RIGHT:
            self.px = min(_X_HI, self.px + PADDLE_SPEED)
        target = self.ball[0] if self.ball is not None else FRAME_W / 2
        self.cx = min(_X_HI, max(_X_LO, C.toward(self.cx, target, CPU_SPEED)))
        if self.ball is None:
            self.serve_wait -= 1
            if self.serve_wait <= 0:
                self.ball = [SERVE_X, SERVE_Y, float(self.rng.uniform(-SERVE_VX, SERVE_VX)), BALL_VY]
            return 0, False
        x, y, vx, vy = self.ball
        x, hit = _reflect(x + vx, _B_LO, _B_HI)
        if hit:
            vx = -vx
        y_old, y = y, y + vy
        reward = 0
        half = PADDLE_W / 2 + BALL / 2
        if vy > 0 and y_old + BALL / 2 < PADDLE_Y <= y + BALL / 2 and abs(x - self.px) <= half:
            vy, vx, y = -BALL_VY, ENGLISH * (x - self.px) / half, PADDLE_Y - BALL / 2
            reward = 1
        elif vy < 0 and y - BALL / 2 <= CPU_Y + PADDLE_H < y_old - BALL / 2 and abs(x - self.cx) <= half:
            vy, vx, y = BALL_VY, ENGLISH * (x - self.cx) / half, CPU_Y + PADDLE_H + BALL / 2
        self.ball = [x, y, vx, vy]
        if y - BALL / 2 > FRAME_H:                   # past the player
            self.misses += 1
            self.ball, self.serve_wait = None, SERVE_WAIT
            return 0, self.misses >= MISSES
        if y + BALL / 2 < 0:                         # past the computer
            self.cpu_misses += 1
            self.ball, self.serve_wait = None, SERVE_WAIT
        return reward, False

    def _layers(self):
        fixed = [C.corner("wall", 0.0, 0.0, WALL_W, FRAME_H),
                 C.corner("wall", FRAME_W - WALL_W, 0.0, WALL_W, FRAME_H)]
        moving = [C.corner("cpu_paddle", self.cx - PADDLE_W / 2, CPU_Y, PADDLE_W, PADDLE_H),
                  C.corner("paddle", self.px - PADDLE_W / 2, PADDLE_Y, PADDLE_W, PADDLE_H)]
        if self.ball is not None:
            moving.append(C.box("ball", self.ball[0], self.ball[1], BALL, BALL))
        return fixed, moving

    def oracle(self) -> int:
        """Stand under the point where the ball will meet the paddle line (walls folded in)."""
        target = FRAME_W / 2
        if self.ball is not None and self.ball[3] > 0:
            target = landing_x(self.ball, PADDLE_Y - BALL / 2, _B_LO, _B_HI)
        return C.steer(self.px, target, PADDLE_SPEED)


def landing_x(ball: list[float], y_line: float, lo: float, hi: float) -> float:
    """Where a ball [x, y, vx, vy] moving down crosses ``y_line``, bouncing between lo and hi."""
    x, y, vx, vy = ball
    t = max(0.0, (y_line - y) / vy)
    span = hi - lo
    u = (x - lo + vx * t) % (2 * span)
    return lo + (u if u <= span else 2 * span - u)
