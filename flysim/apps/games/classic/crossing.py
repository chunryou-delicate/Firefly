"""M7f game "crossing" (길 건너기) — road-crossing style, seen from above.

The walker starts below a road of ``LANES`` lanes and walks up across it; cars run along each
lane. Score = crossings (each time the walker clears the top lane it starts again at the
bottom). A car pushes the walker one lane back and stuns it for ``HIT_STUN`` ticks. Only the
clock ends the game.

Rules (all constants fixed here, docs/m7f-brief.md §2):

* JUMP (= the original's UP) walks up ``WALK_SPEED`` px per tick. There is no way down and no
  sideways move: LEFT and RIGHT do nothing (the original had neither).
* Lane ``i`` (0 = nearest the start) carries ``LANE_CARS[i]`` cars evenly spaced on a loop of
  ``TRACK`` px (the screen plus one car), moving ``LANE_SPEED[i]`` px per tick — rightward in the
  lower half of the road, leftward in the upper half. Each lane's phase is random (seeded).
* While stunned the walker cannot move and cannot be hit again.
* Geometry: the gap between the cars of two neighbouring lanes (``LANE_H - CAR_H`` = 4 px) is
  taller than the walker (3 px), so there is a car-free place to stand between lanes. (The
  first constants — lanes 6 px, cars 4 px, walker 4 x 4 — had no such place; re-tuned on that
  geometric condition, docs/m7f-brief.md addendum 1.)
"""
from __future__ import annotations

import numpy as np

from ..base import FRAME_W, Action
from . import common as C

LANES = 10
LANE_H = 7.0
ROAD_TOP = 7.0                   # lanes cover y in [7, 77); lane 0 is the lowest
LINE_H = 0.5                     # lane lines, centred on every lane boundary
WALKER_W, WALKER_H = 3.0, 3.0   # shorter than the gap between two lanes' cars (LANE_H - CAR_H)
WALKER_X = 30.0                  # walker centre x (fixed)
WALKER_Y0 = 79.0                 # walker top edge at the start
WALK_SPEED = 0.5                 # px per tick while JUMP is held
CAR_W, CAR_H = 8.0, 3.0
LANE_SPEED = (0.6, 0.8, 1.0, 1.3, 1.6, 1.6, 1.3, 1.0, 0.8, 0.6)     # px per tick, lane 0 first
LANE_CARS = (3, 3, 3, 2, 2, 2, 2, 3, 3, 3)
TRACK = FRAME_W + CAR_W          # a car leaves one side and comes back on the other
HIT_STUN = 20

_ROAD_BOTTOM = ROAD_TOP + LANES * LANE_H


def lane_top(i: int) -> float:
    return _ROAD_BOTTOM - (i + 1) * LANE_H


def lane_dir(i: int) -> float:
    return 1.0 if i < LANES // 2 else -1.0


class CrossingGame(C.ClassicGame):
    name = "crossing"
    title_ko = "길 건너기"
    genre_ko = "길 건너기형"
    player_ko = "아래에서 위로 차선 10개를 건넌다. 차에 치이면 한 차선 뒤로 밀린다"
    jump_effect = "walk up"
    left_effect = "nothing (no sideways move in the original)"
    right_effect = "nothing (no sideways move in the original)"
    automatic = "nothing"
    score_rule = "crossings"
    end_rule = "max_ticks only"
    kinds = ("walker", "car", "lane")

    def _reset_state(self) -> None:
        self.wy = WALKER_Y0
        self.stun = 0
        self.hits = 0
        # car i of lane l sits at phase[l] + i * TRACK / n (the car's front edge along its loop)
        self.phase = [float(self.rng.uniform(0.0, TRACK)) for _ in range(LANES)]

    def cars(self, lane: int, ahead: int = 0) -> list[C.Rect]:
        """The lane's car rectangles, ``ahead`` ticks from now."""
        n, v, d = LANE_CARS[lane], LANE_SPEED[lane], lane_dir(lane)
        out = []
        y0 = lane_top(lane) + (LANE_H - CAR_H) / 2
        for i in range(n):
            p = (self.phase[lane] + i * TRACK / n + d * v * ahead) % TRACK
            out.append(C.corner("car", p - CAR_W, y0, CAR_W, CAR_H))
        return out

    def walker(self, y: float | None = None) -> C.Rect:
        return C.corner("walker", WALKER_X - WALKER_W / 2, self.wy if y is None else y, WALKER_W, WALKER_H)

    def _hit(self, rect: C.Rect, ahead: int = 0) -> bool:
        return any(C.overlaps(rect, car) for lane in range(LANES) for car in self.cars(lane, ahead))

    def _advance(self, action: Action) -> tuple[int, bool]:
        for lane in range(LANES):
            self.phase[lane] = (self.phase[lane] + lane_dir(lane) * LANE_SPEED[lane]) % TRACK
        if self.stun > 0:
            self.stun -= 1
            return 0, False
        if action == Action.JUMP:
            self.wy -= WALK_SPEED
        if self.wy + WALKER_H <= ROAD_TOP:            # cleared the top lane
            self.wy = WALKER_Y0
            return 1, False
        if self._hit(self.walker()):
            self.hits += 1
            self.wy = min(WALKER_Y0, self.wy + LANE_H)
            self.stun = HIT_STUN
        return 0, False

    def _layers(self):
        fixed = [C.corner("lane", 0.0, ROAD_TOP + i * LANE_H - LINE_H / 2, FRAME_W, LINE_H)
                 for i in range(LANES + 1)]
        moving = [car for lane in range(LANES) for car in self.cars(lane)]
        moving.append(self.walker())
        return fixed, moving

    def oracle(self, look: int = 160) -> int:
        """Plan against the cars' known courses: for each first step (walk, wait), search every
        walk / wait plan over the next ``look`` ticks (enough to cross the whole road) and take
        the step whose best car-free plan reaches the top soonest — or, when none does, stays
        car-free longest and then gets furthest up. Walk on ties. The walker is slow (a lane
        takes 14 ticks), so a short look ahead walks into lanes it cannot leave in time."""
        if self.stun > 0:
            return int(Action.NOOP)
        t = np.arange(look + 1, dtype=np.float64)[:, None, None]
        n = np.array(LANE_CARS, dtype=np.float64)[None, :, None]
        i = np.arange(max(LANE_CARS), dtype=np.float64)[None, None, :]
        v = np.array([lane_dir(l) * LANE_SPEED[l] for l in range(LANES)])[None, :, None]
        p = (np.array(self.phase)[None, :, None] + i * TRACK / n + v * t) % TRACK
        x0, x1 = WALKER_X - WALKER_W / 2, WALKER_X + WALKER_W / 2
        occ = ((p - CAR_W < x1) & (x0 < p) & (i < n)).any(axis=2)           # (tick, lane)
        y = self.wy - np.arange(look + 1) * WALK_SPEED                       # after k steps
        top = np.array([lane_top(l) + (LANE_H - CAR_H) / 2 for l in range(LANES)])
        over = (y[:, None] < top[None, :] + CAR_H) & (top[None, :] < y[:, None] + WALKER_H)
        goal = y + WALKER_H <= ROAD_TOP
        safe = ~(occ.astype(np.int32) @ over.T.astype(np.int32)).astype(bool) | goal[None, :]

        def outcome(k1: int) -> tuple:
            reach = np.zeros(look + 1, dtype=bool)
            reach[k1] = safe[1, k1]
            if not reach.any():
                return (0, 0, 0, 0)
            for tt in range(1, look + 1):
                if tt > 1:
                    reach = (reach | np.roll(reach, 1)) & safe[tt]
                    if not reach.any():
                        return (0, 0, tt - 1, int(np.nonzero(last)[0].max()))
                if (reach & goal).any():
                    return (1, -tt, 0, 0)
                last = reach
            return (0, 0, look, int(np.nonzero(reach)[0].max()))

        return int(Action.JUMP) if outcome(1) >= outcome(0) else int(Action.NOOP)
