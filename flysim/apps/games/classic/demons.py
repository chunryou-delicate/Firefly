"""M7f game "demons" (악마 물리치기) — airborne-enemy shooting style, seen from the side.

Up to ``DEMONS`` demons fly about the upper screen, each on its own course, and shoot bolts
straight down. The player's cannon slides along the bottom and fires by itself — one shot on
screen at a time. Score = demons hit. Three bolts on the cannon end the game.

Rules (all constants fixed here, docs/m7f-brief.md §2):

* Each demon flies at ``DEMON_SPEED`` toward a random point of the zone ``ZONE_X`` x ``ZONE_Y``
  (seeded) and picks a new one when it gets there. A demon that is hit is replaced
  ``RESPAWN`` ticks later at a random point of the zone.
* Each demon fires a bolt after a random wait of ``BOLT_WAIT`` ticks (seeded), again and again.
* Automatic fire as in "formation": ``FIRE_EVERY`` ticks after the previous shot left, as soon
  as no shot is on screen.
* A bolt on the cannon costs a life and clears the falling bolts.
* JUMP does nothing (the original had no UP move).
"""
from __future__ import annotations

import math

from ..base import FRAME_H, FRAME_W, Action
from . import common as C

CANNON_W, CANNON_H = 7.0, 4.0
CANNON_Y = 76.0
CANNON_SPEED = 1.0
DEMONS = 3
DEMON_W, DEMON_H = 8.0, 4.0
ZONE_X = (8.0, 76.0)             # range of a demon's centre
ZONE_Y = (10.0, 50.0)
DEMON_SPEED = 0.6
RESPAWN = 60
BOLT_W, BOLT_H = 1.0, 4.0
BOLT_SPEED = 1.0
BOLT_WAIT = (60, 120)
SHOT_W, SHOT_H = 1.0, 4.0
SHOT_SPEED = 3.0
FIRE_EVERY = 20

_C_LO, _C_HI = CANNON_W / 2, FRAME_W - CANNON_W / 2


class DemonsGame(C.ClassicGame):
    name = "demons"
    title_ko = "악마 물리치기"
    genre_ko = "공중 적 사격형"
    player_ko = "아래 포대를 좌우로 움직인다. 적 셋까지가 공중에서 따로 움직이며 아래로 쏜다"
    automatic = f"fire, every {FIRE_EVERY} ticks while no shot is on screen"
    score_rule = "demons hit"
    end_rule = "3 bolts on the cannon, or max_ticks"
    has_lives = True
    kinds = ("cannon", "demon", "shot", "bolt")

    def _reset_state(self) -> None:
        self.cx = FRAME_W / 2
        self.lives_left = C.LIVES
        self.shot: list[float] | None = None
        self.fire_wait = FIRE_EVERY
        self.bolts: list[list[float]] = []
        # a slot is a demon {x, y, tx, ty, wait} or None with a respawn countdown
        self.slots: list[dict | None] = [self._spawn() for _ in range(DEMONS)]
        self.respawn = [0] * DEMONS

    def _point(self) -> tuple[float, float]:
        return float(self.rng.uniform(*ZONE_X)), float(self.rng.uniform(*ZONE_Y))

    def _spawn(self) -> dict:
        x, y = self._point()
        tx, ty = self._point()
        return {"x": x, "y": y, "tx": tx, "ty": ty,
                "wait": int(self.rng.integers(BOLT_WAIT[0], BOLT_WAIT[1] + 1))}

    def lives(self) -> int:
        return self.lives_left

    @staticmethod
    def demon_rect(d: dict) -> C.Rect:
        return C.box("demon", d["x"], d["y"], DEMON_W, DEMON_H)

    def _advance(self, action: Action) -> tuple[int, bool]:
        if action == Action.LEFT:
            self.cx = max(_C_LO, self.cx - CANNON_SPEED)
        elif action == Action.RIGHT:
            self.cx = min(_C_HI, self.cx + CANNON_SPEED)
        reward = 0
        self.fire_wait -= 1
        if self.shot is None and self.fire_wait <= 0:
            self.shot = [self.cx - SHOT_W / 2, CANNON_Y - SHOT_H]
            self.fire_wait = FIRE_EVERY
        if self.shot is not None:
            self.shot[1] -= SHOT_SPEED
            shot = C.corner("shot", self.shot[0], self.shot[1], SHOT_W, SHOT_H)
            hit = [i for i, d in enumerate(self.slots) if d is not None and C.overlaps(shot, self.demon_rect(d))]
            if hit:
                i = max(hit, key=lambda i: self.slots[i]["y"])     # the lowest is hit first
                self.slots[i], self.respawn[i] = None, RESPAWN
                self.shot, reward = None, 1
            elif self.shot[1] + SHOT_H < 0:
                self.shot = None
        for i, d in enumerate(self.slots):
            if d is None:
                self.respawn[i] -= 1
                if self.respawn[i] <= 0:
                    self.slots[i] = self._spawn()
                continue
            dx, dy = d["tx"] - d["x"], d["ty"] - d["y"]
            dist = math.hypot(dx, dy)
            if dist <= DEMON_SPEED:
                d["x"], d["y"] = d["tx"], d["ty"]
                d["tx"], d["ty"] = self._point()
            else:
                d["x"] += DEMON_SPEED * dx / dist
                d["y"] += DEMON_SPEED * dy / dist
            d["wait"] -= 1
            if d["wait"] <= 0:
                self.bolts.append([d["x"] - BOLT_W / 2, d["y"] + DEMON_H / 2])
                d["wait"] = int(self.rng.integers(BOLT_WAIT[0], BOLT_WAIT[1] + 1))
        cannon = C.corner("cannon", self.cx - CANNON_W / 2, CANNON_Y, CANNON_W, CANNON_H)
        kept = []
        for b in self.bolts:
            b[1] += BOLT_SPEED
            if C.overlaps(C.corner("bolt", b[0], b[1], BOLT_W, BOLT_H), cannon):
                self.lives_left -= 1
                self.bolts = []
                return reward, self.lives_left <= 0
            if b[1] < FRAME_H:
                kept.append(b)
        self.bolts = kept
        return reward, False

    def _layers(self):
        moving = [self.demon_rect(d) for d in self.slots if d is not None]
        moving.append(C.corner("cannon", self.cx - CANNON_W / 2, CANNON_Y, CANNON_W, CANNON_H))
        if self.shot is not None:
            moving.append(C.corner("shot", self.shot[0], self.shot[1], SHOT_W, SHOT_H))
        moving += [C.corner("bolt", b[0], b[1], BOLT_W, BOLT_H) for b in self.bolts]
        return [], moving

    def oracle(self) -> int:
        """Step away from a bolt that would land on the cannon soon; otherwise stand under the
        nearest demon."""
        reach = CANNON_W / 2 + BOLT_W / 2 + 2.0
        for b in sorted(self.bolts, key=lambda b: -b[1]):
            bx = b[0] + BOLT_W / 2
            if abs(bx - self.cx) < reach and b[1] + BOLT_H > CANNON_Y - 30.0:
                away = Action.LEFT if bx > self.cx else Action.RIGHT
                if (away == Action.LEFT and self.cx - CANNON_SPEED < _C_LO) or \
                        (away == Action.RIGHT and self.cx + CANNON_SPEED > _C_HI):
                    away = Action.RIGHT if away == Action.LEFT else Action.LEFT
                return int(away)
        demons = [d for d in self.slots if d is not None]
        if not demons:
            return int(Action.NOOP)
        target = min(demons, key=lambda d: abs(d["x"] - self.cx))["x"]
        return C.steer(self.cx, target, CANNON_SPEED)
