"""M7b: game harness for the fly-brain game ladder (docs/m7-game-ladder.md, docs/m7b-brief.md).

Two hand-made games ("jump", "dodge"), a shared environment interface shaped like the ALE
convention (84x84 uint8 grayscale, reset/step), a pixel-only looming detector
(``flysim.sensory.looming``), and the comparison agents. The fly brain is not connected here —
``BrainAgent`` and ``flysim.probe.motor`` are interfaces for M7c.
"""
