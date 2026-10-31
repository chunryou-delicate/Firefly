"""M7f game "ski" (스키 활강) — downhill style, seen from above.

The skier stays near the top of the screen and moves left and right; the slope flows upward,
so gates (pairs of flags) and trees come up from the bottom. Score = gates passed (the skier
wholly between the two flags when the gate reaches the skier's row). Running into a flag or a
tree stops everything for ``CRASH_TICKS``. Only the clock ends the game.

Rules (all constants fixed here, docs/m7f-brief.md §2):

* The slope scrolls ``SCROLL`` px per tick. Every ``GATE_SPACING`` px a gate comes up, centred at
  a random x in ``GATE_X`` (seeded), its flags' inner edges ``GATE_HALF`` px either side of the
  centre. Halfway between two gates a tree comes up with probability ``TREE_P`` at a random x in
  ``TREE_X`` (seeded).
* A flag or tree that was run into does not stop the skier again.
* JUMP does nothing (the original had no UP move).
"""
from __future__ import annotations

from ..base import FRAME_H, FRAME_W, Action
from . import common as C

SKIER_W, SKIER_H = 4.0, 6.0
SKIER_Y = 12.0                   # top edge
SKIER_SPEED = 1.0
SKIER_X = (4.0, 80.0)            # range of the skier's centre
SCROLL = 0.75                    # px per tick
GATE_SPACING = 40.0              # px of slope between gates
GATE_HALF = 8.0                  # inner half-opening of a gate
GATE_X = (16.0, 68.0)            # range of a gate's centre
FLAG_W, FLAG_H = 2.0, 4.0
TREE_W, TREE_H = 5.0, 6.0
TREE_P = 0.5
TREE_X = (4.0, 80.0)             # range of a tree's centre
FIRST_GATE = 10.0                # px of slope before the first gate
CRASH_TICKS = 30


class SkiGame(C.ClassicGame):
    name = "ski"
    title_ko = "스키 활강"
    genre_ko = "활강형"
    player_ko = "위쪽에 있는 스키어를 좌우로 움직인다. 세상이 위로 흐르며 깃발 문과 나무가 아래에서 올라온다. 부딪히면 잠시 멈춘다"
    automatic = "nothing"
    score_rule = "gates passed"
    end_rule = "max_ticks only"
    kinds = ("skier", "flag", "tree")

    def _reset_state(self) -> None:
        self.sx = FRAME_W / 2
        self.gates: list[dict] = []      # {"c": centre x, "y": top edge, "passed": bool, "hit": [l, r]}
        self.trees: list[dict] = []      # {"x": centre x, "y": top edge, "hit": bool}
        self.to_next = FIRST_GATE        # px of slope until the next slot comes up
        self.next_is_gate = True
        self.crash = 0
        self.crashes = 0
        self.missed = 0

    def flags(self, g: dict) -> tuple[C.Rect, C.Rect]:
        return (C.corner("flag", g["c"] - GATE_HALF - FLAG_W, g["y"], FLAG_W, FLAG_H),
                C.corner("flag", g["c"] + GATE_HALF, g["y"], FLAG_W, FLAG_H))

    @staticmethod
    def tree(t: dict) -> C.Rect:
        return C.corner("tree", t["x"] - TREE_W / 2, t["y"], TREE_W, TREE_H)

    def skier(self) -> C.Rect:
        return C.corner("skier", self.sx - SKIER_W / 2, SKIER_Y, SKIER_W, SKIER_H)

    def _advance(self, action: Action) -> tuple[int, bool]:
        if self.crash > 0:
            self.crash -= 1
            return 0, False
        if action == Action.LEFT:
            self.sx = max(SKIER_X[0], self.sx - SKIER_SPEED)
        elif action == Action.RIGHT:
            self.sx = min(SKIER_X[1], self.sx + SKIER_SPEED)
        for g in self.gates:
            g["y"] -= SCROLL
        for t in self.trees:
            t["y"] -= SCROLL
        self.to_next -= SCROLL
        if self.to_next <= 0:
            y = FRAME_H + self.to_next                  # where it would be had it come up exactly on time
            if self.next_is_gate:
                self.gates.append({"c": float(self.rng.uniform(*GATE_X)), "y": y, "passed": False,
                                   "hit": [False, False]})
            elif self.rng.random() < TREE_P:
                self.trees.append({"x": float(self.rng.uniform(*TREE_X)), "y": y, "hit": False})
            self.next_is_gate = not self.next_is_gate
            self.to_next += GATE_SPACING / 2
        reward = 0
        skier = self.skier()
        for g in self.gates:
            if not g["passed"] and g["y"] <= SKIER_Y:
                g["passed"] = True
                if abs(self.sx - g["c"]) <= GATE_HALF - SKIER_W / 2:
                    reward += 1
                else:
                    self.missed += 1
            for k, f in enumerate(self.flags(g)):
                if not g["hit"][k] and C.overlaps(skier, f):
                    g["hit"][k] = True
                    self.crash = CRASH_TICKS
        for t in self.trees:
            if not t["hit"] and C.overlaps(skier, self.tree(t)):
                t["hit"] = True
                self.crash = CRASH_TICKS
        if self.crash:
            self.crashes += 1
        self.gates = [g for g in self.gates if g["y"] + FLAG_H > 0]
        self.trees = [t for t in self.trees if t["y"] + TREE_H > 0]
        return reward, False

    def _layers(self):
        moving = [f for g in self.gates for f in self.flags(g)]
        moving += [self.tree(t) for t in self.trees]
        moving.append(self.skier())
        return [], moving

    def oracle(self) -> int:
        """Head for the centre of the next gate; sidestep a tree that is in the way."""
        ahead = [g for g in self.gates if not g["passed"]]
        target = min(ahead, key=lambda g: g["y"])["c"] if ahead else FRAME_W / 2
        for t in self.trees:
            if t["hit"] or not (SKIER_Y - TREE_H < t["y"] < SKIER_Y + SKIER_H + 24.0):
                continue
            clear = TREE_W / 2 + SKIER_W / 2 + 1.0
            if abs(t["x"] - self.sx) < clear:
                side = 1.0 if target >= t["x"] else -1.0
                target = t["x"] + side * (clear + 1.0)
                if not SKIER_X[0] <= target <= SKIER_X[1]:
                    target = t["x"] - side * (clear + 1.0)
                break
        return C.steer(self.sx, target, SKIER_SPEED)
