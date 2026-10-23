"""M7b: screen pixels -> left / right "looming" (다가옴) signal.

**The detector sees pixels and nothing else.** ``LoomingDetector.update`` takes one argument, a
2-D ``uint8`` frame, and rejects anything that is not one — in particular a game object, so it
cannot read obstacle coordinates. That is what makes the brainless bot and the fly brain
comparable: both get exactly this signal (docs/m7-game-ladder.md rule 1, "same eyes").

The signal (ASSUMPTION, chosen before any game was played and checked only against a generic
expanding / translating disc — tests/test_games.py):

    dark_t(x, y)  = 1 - frame_t(x, y) / 255                  darkness of a pixel, 0..1
    D_t(side)     = mean of dark_t over that half of the screen (left = columns < W/2)
    loom_t(side)  = max(0, D_t(side) - D_{t-1}(side))         growth of the dark area per tick

Why this form:

* It measures how much the dark region in each half **grows** from one frame to the next. An
  object coming closer grows, so the dark area grows (> 0). An object sliding sideways inside a
  half darkens its leading edge and lightens its trailing edge by the same amount (≈ 0), so
  translation is not mistaken for approach.
* Only growth counts: an object disappearing or the scene getting brighter gives 0, not a
  negative signal. This is a "dark looming" detector — it responds to things getting darker and
  bigger, not lighter. That choice is an assumption about what the fly's looming pathway
  prefers; it is not taken from the connectome.
* Units: per-tick growth of the mean darkness of one half (dimensionless). At 60 ticks/s.

There are no game-specific settings. The class takes no parameters; every game, the threshold
bot and the brain agent construct it the same way.
"""
from __future__ import annotations

import numpy as np


class LoomingDetector:
    """Stateful: call ``reset()`` at the start of an episode, then ``update(frame)`` every tick."""

    def __init__(self) -> None:
        self._prev: np.ndarray | None = None       # D_{t-1} per half
        self._width: int | None = None

    def reset(self) -> None:
        self._prev = None
        self._width = None

    @staticmethod
    def _check(frame) -> np.ndarray:
        if not isinstance(frame, np.ndarray):
            raise TypeError(
                f"LoomingDetector reads screen pixels only: expected a numpy uint8 frame, got "
                f"{type(frame).__name__}. Game state must never reach the detector.")
        if frame.dtype != np.uint8 or frame.ndim != 2:
            raise TypeError(f"expected a 2-D uint8 frame, got {frame.dtype} with shape {frame.shape}")
        if frame.shape[1] < 2 or frame.shape[1] % 2:
            raise ValueError(f"frame width must be even to split into halves, got {frame.shape[1]}")
        return frame

    def update(self, frame: np.ndarray) -> tuple[float, float]:
        """Return ``(left, right)`` looming for this frame (0 on the first frame of an episode)."""
        f = self._check(frame)
        w = f.shape[1]
        if self._width is not None and w != self._width:
            raise ValueError("frame size changed within an episode")
        dark = 1.0 - f.astype(np.float32) / 255.0
        half = w // 2
        d = np.array([dark[:, :half].mean(), dark[:, half:].mean()], dtype=np.float64)
        if self._prev is None:
            self._prev, self._width = d, w
            return 0.0, 0.0
        loom = np.maximum(0.0, d - self._prev)
        self._prev = d
        return float(loom[0]), float(loom[1])

    @staticmethod
    def describe() -> dict:
        return {"detector": "LoomingDetector",
                "signal": "loom(side) = max(0, D_t - D_{t-1}), D = mean darkness (1 - px/255) of the half",
                "halves": "left = columns < W/2, right = columns >= W/2",
                "parameters": "none (same eyes for every game)",
                "status": "ASSUMPTION — dark-looming, growth-only; not derived from the connectome"}
