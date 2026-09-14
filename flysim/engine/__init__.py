"""LIF simulation kernel (M2/M2b): event-driven propagation + fused LIF update, GPU resident."""
from .params import EngineParams, DEFAULT_G, DEFAULT_G_CONDUCTANCE, SYNAPSE_MODELS  # noqa: F401
from .recorder import SpikeRecorder, RecorderOverflow  # noqa: F401
from .lif import LIFEngine, CSRGraph, propagate_reference, propagate_reference_split  # noqa: F401
