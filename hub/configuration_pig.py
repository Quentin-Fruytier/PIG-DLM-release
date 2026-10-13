"""Configuration of a PIG Hub model: which backbone the adapter was fitted to, and the decoding defaults.

A PIG repository on the Hugging Face Hub holds the Entropy Finite Differences Estimator (the adapter,
model.safetensors), the two calibration tables (flip_confidence.npz, flip_entropy.npz) and this code.
The backbone itself is loaded from its own repository at the pinned revision.
"""
from transformers import PretrainedConfig

try:
    from .api import DEFAULTS, FAMILIES        # inside a Hub repository: api.py is vendored next to this file
except ImportError:
    from pig.api import DEFAULTS, FAMILIES     # inside the code repository


class PIGConfig(PretrainedConfig):
    model_type = "pig"

    def __init__(self, family="llada", backbone="GSAI-ML/LLaDA-8B-Instruct", backbone_revision=None,
                 defaults=None, adapter=None, tables=None, **kwargs):
        if family not in FAMILIES:
            raise ValueError(f"unknown family {family!r}; expected one of {FAMILIES}")
        self.family = family
        self.backbone = backbone
        self.backbone_revision = backbone_revision
        self.defaults = dict(DEFAULTS[family], **(defaults or {}))
        self.adapter = dict(adapter or {})      # provenance: parameters, epoch, validation correlation
        self.tables = dict(tables or {"confidence": "flip_confidence.npz", "entropy": "flip_entropy.npz"})
        super().__init__(**kwargs)
