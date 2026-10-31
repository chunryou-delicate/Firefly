"""M7f: eight hand-made classic-style games (docs/m7f-brief.md) — one file per game plus
``common.py``. The list is the brief's, in its order; nothing outside it."""
from .boxing import BoxingGame
from .bricks import BricksGame
from .catch import CatchGame
from .crossing import CrossingGame
from .demons import DemonsGame
from .formation import FormationGame
from .paddle import PaddleGame
from .ski import SkiGame

CLASSIC = {"paddle": PaddleGame, "bricks": BricksGame, "formation": FormationGame,
           "crossing": CrossingGame, "catch": CatchGame, "demons": DemonsGame, "ski": SkiGame,
           "boxing": BoxingGame}
GAMES = tuple(CLASSIC)
