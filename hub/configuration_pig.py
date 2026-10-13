"""Configuration of a PIG Hub model: which backbone the adapter was fitted to, and the decoding defaults.

A PIG repository on the Hugging Face Hub holds model.safetensors (the Entropy Finite Differences
Estimator, keys fde.*, and the two calibration tables, keys tables.<axis>.*), this config and one
modeling file. The backbone itself is loaded from its own repository at the pinned revision.
"""
from transformers import PretrainedConfig

from pig.api import DEFAULTS, FAMILIES   # hub.build replaces this line with the literal values


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
        self.tables = list(tables or ["confidence", "entropy"])   # table axes stored in model.safetensors
        super().__init__(**kwargs)
