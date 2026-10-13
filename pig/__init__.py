"""PIG: Parallel Inconsistency Guidance for accelerating diffusion language models.

    import pig
    model = pig.load("llada")
    out = model.generate("prompt", mode="pig.int", phi=0.1)
"""
from .api import DEFAULTS, MODES, PIG, Output, make_rule
from .loading import load, load_decoder, load_tables, make_geometry
from .rules import (ConfidenceThreshold, PIGExt, PIGInt, Round, aggregate_influence, confidence_order,
                    confidence_threshold, pig_ext, pig_int, waiting_gain)
from .tables import FlipTable

__version__ = "0.1.0"

__all__ = ["ConfidenceThreshold", "DEFAULTS", "FlipTable", "MODES", "Output", "PIG", "PIGExt", "PIGInt", "Round",
           "aggregate_influence", "confidence_order", "confidence_threshold", "load", "load_decoder", "load_tables", "make_geometry",
           "make_rule", "pig_ext", "pig_int", "waiting_gain", "__version__"]
