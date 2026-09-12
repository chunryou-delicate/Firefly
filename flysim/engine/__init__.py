"""LIF simulation kernel (M2): event-driven propagation + fused LIF update, GPU resident."""
from .params import EngineParams, DEFAULT_G  # noqa: F401
from .recorder import SpikeRecorder, RecorderOverflow  # noqa: F401
from .lif import LIFEngine, CSRGraph, propagate_reference  # noqa: F401
