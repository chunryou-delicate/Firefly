"""M7e3a: one game as a scene record, tick by tick — the real game code replayed with recorded actions.

    replay(game, seed, actions) -> dict

The show page (``flysim-show.html``) draws a game for people from this record and never from
physics of its own: positions, distances, heights and scores come from ``base.make(game)``
stepping the recorded actions (docs/m7e3-brief.md §1 rule 2). ``eye_rect`` is the dark rectangle
that frame was drawn with, so the page can redraw what the fly saw; ``redraw(eye_rect)`` must give
the frame bit for bit — ``replay`` keeps the frames so callers check it against frames.npz.

Alignment (fixed by tests/test_m7e3_show_data.py):

* ``state[k]``, ``eye_rect[k]``, ``loom[k]``, ``frames[k]`` belong to frame k — the state that drew
  it, and the detector's output on it.
* ``action[k]`` is the action chosen on frame k: the step from frame k to frame k + 1.
* ``t_ms[k]`` = floor(k * 1000 / 60), the moment frame k is shown.
* An event's ``tick`` is the step it happened in and its ``t_ms`` is the end of that step,
  ``tick_end_ms(tick)`` — the moment the first frame showing it appears, or the end of the game for
  the crash. That is the stamp game.json already uses.

``base.py``, ``jump.py`` and ``dodge.py`` are not changed. ``eye_rect`` repeats ``draw_block``'s
arithmetic operation for operation; the bit-exact redraw is what proves it.
"""
from __future__ import annotations

import numpy as np

from ...sensory.looming import LoomingDetector
from . import dodge as D
from . import jump as J
from .base import (CAMERA_HEIGHT, FOCAL_PX, FRAME_W, HORIZON_ROW, OBJECT, background, fill_rect, make,
                   to_frame)
from .brain import tick_end_ms

STATE_KEYS = ("x", "obj_x", "dist", "height", "airborne", "speed", "score")
VISIBLE_PX = 2.0                 # "visible2px": the dark rectangle is first >= 2 px wide (display only)
LAST_MS = 300.0                  # loom_last300_frac: share of the approach's signal in its last 300 ms


def frame_ms(k: int) -> int:
    """When frame k is shown: floor(k * 1000 / 60) (= the end of tick k - 1)."""
    return tick_end_ms(k - 1) if k > 0 else 0


def _rect(lateral0: float, lateral1: float, height: float, dist: float) -> list[float] | None:
    """``base.draw_block``'s screen arithmetic, unchanged: [x0, x1, y0, y1] on the 84x84 frame."""
    if dist <= 0:
        return None
    s = FOCAL_PX / dist
    cx = FRAME_W / 2.0
    y_bottom = HORIZON_ROW + s * CAMERA_HEIGHT
    y_top = HORIZON_ROW + s * (CAMERA_HEIGHT - height)
    return [cx + s * lateral0, cx + s * lateral1, y_top, y_bottom]


def eye_rect(env) -> list[float] | None:
    """The rectangle ``env._render()`` fills for the current state (None: nothing drawn)."""
    if env.dist is None:
        return None
    if env.name == "jump":
        return _rect(-J.OBS_HALF_WIDTH, J.OBS_HALF_WIDTH, J.OBS_HEIGHT, env.dist)
    if env.name in ("dodge", "dodge_v2"):
        dx = env.obj_x - env.x
        return _rect(dx - D.OBJ_HALF_WIDTH, dx + D.OBJ_HALF_WIDTH, D.OBJ_HEIGHT, env.dist)
    raise KeyError(f"no scene record for game {env.name!r}")


def redraw(rect: list[float] | None) -> np.ndarray:
    """``base.background()`` + ``fill_rect(eye_rect)``: the frame the fly saw, from the record."""
    img = background()
    if rect is not None:
        fill_rect(img, *rect, OBJECT)
    return to_frame(img)


def visible_width(rect: list[float] | None) -> float:
    """Width of the rectangle inside the frame, in px (0 when nothing is drawn)."""
    if rect is None:
        return 0.0
    return max(0.0, min(rect[1], float(FRAME_W)) - max(rect[0], 0.0))


def state(env) -> dict:
    """What a person's view needs; a field the game does not have is None."""
    if env.name == "jump":
        return {"x": None, "obj_x": None, "dist": env.dist, "height": env.height,
                "airborne": env.airborne, "speed": env.speed, "score": env.score}
    return {"x": env.x, "obj_x": env.obj_x, "dist": env.dist, "height": None,
            "airborne": None, "speed": env.speed, "score": env.score}


def _side(env) -> str:
    if env.name == "jump":
        return "center"
    return "right" if env.obj_x > env.x else "left"


def replay(game: str, seed: int, actions) -> dict:
    """Step ``base.make(game)`` with ``actions`` from ``reset(seed)``; record every frame's state.

    The game must end on the last action — not before (actions left over) and not after (the
    record would stop mid-game). Either raises: a scene is never padded or guessed.
    """
    actions = [int(a) for a in actions]
    env = make(game)
    det = LoomingDetector()
    frame = env.reset(seed)
    if env.dist is not None:
        raise RuntimeError("an obstacle on the first frame: the scene rules below assume none")
    rows = {k: [] for k in STATE_KEYS}
    rects, looms, frames, events, obstacles = [], [], [], [], []
    cur = None
    info = {"score": 0, "crashed": False, "truncated": False}
    for k, a in enumerate(actions):
        if env.done:
            raise ValueError(f"{game} seed {seed}: the game ended after {k} ticks, "
                             f"but {len(actions)} actions were given")
        for key, v in state(env).items():
            rows[key].append(v)
        rects.append(eye_rect(env))
        looms.append(det.update(frame))
        frames.append(frame)
        speed_before = env.speed
        present_before = env.dist is not None
        frame, reward, _, info = env.step(a)

        def event(kind: str, side: str | None) -> None:
            events.append({"tick": k, "t_ms": tick_end_ms(k), "type": kind, "side": side})

        if not present_before and env.dist is not None:
            cur = {"spawn_tick": k, "end_tick": None, "side": _side(env), "outcome": None,
                   "reach_tick": None, "reach_t_ms": None, "visible2px_tick": None}
            event("spawn", cur["side"])
        ended = reward > 0 or info["crashed"]
        if cur is not None and cur["reach_tick"] is None and (ended or env.dist <= D.D_HIT):
            # first step that brought the obstacle to the collision window's near edge (D_HIT is the
            # same in both games); a pass or a crash can only happen at or after it
            cur["reach_tick"], cur["reach_t_ms"] = k, tick_end_ms(k)
        if cur is not None and cur["visible2px_tick"] is None and not ended \
                and visible_width(eye_rect(env)) >= VISIBLE_PX:
            cur["visible2px_tick"] = k
            event("visible2px", cur["side"])
        if ended:
            kind = "crash" if info["crashed"] else "pass"
            event(kind, cur["side"])
            cur["end_tick"], cur["outcome"] = k, kind
            obstacles.append(cur)
            cur = None
        if env.speed > speed_before:
            event("speedup", None)
    if not env.done:
        raise ValueError(f"{game} seed {seed}: {len(actions)} actions given, but the game goes on")
    if cur is not None:                              # the clock (max_ticks) ended the game mid-approach
        cur["end_tick"], cur["outcome"] = len(actions) - 1, "cut"
        obstacles.append(cur)
    for ob in obstacles:
        ob["loom_last300_frac"] = loom_last300_frac(looms, ob)
    return {"game": game, "seed": int(seed), "n_ticks": len(actions), "score": int(env.score),
            "crashed": bool(info["crashed"]), "truncated": bool(info["truncated"]),
            "t_ms": [frame_ms(k) for k in range(len(actions))], "action": actions,
            "loom": [list(v) for v in looms], "state": rows, "eye_rect": rects,
            "events": events, "obstacles": obstacles, "frames": np.stack(frames)}


def loom_last300_frac(looms: list, ob: dict) -> float | None:
    """Share of the obstacle's looming signal that falls in the last ``LAST_MS`` before it reaches
    the collision window (``reach_tick``), over its whole approach.

    The approach is the frames that show it up to that step, ``spawn_tick + 1 .. reach_tick``; frame
    k's signal drives tick k, i.e. [frame_ms(k), frame_ms(k + 1)), and counts as "last" when that
    tick starts no earlier than ``reach_t_ms - LAST_MS``. None when it never reached the window or the
    approach carried no signal.

    Signal = left + right. Only one obstacle is ever on screen, so both halves' growth is its signal.
    The brief says "that side's signal"; for a pillar that stays in its half (a still player) the
    two readings are the same number, but a player that outruns it (the one-direction bot on
    dodge_v2) carries the pillar across the centre and the spawn side's reading drops to 0.
    """
    if ob["reach_tick"] is None:
        return None
    ks = range(ob["spawn_tick"] + 1, ob["reach_tick"] + 1)
    total = sum(looms[k][0] + looms[k][1] for k in ks)
    last = sum(looms[k][0] + looms[k][1] for k in ks if frame_ms(k) >= ob["reach_t_ms"] - LAST_MS)
    return float(last / total) if total > 0 else None
