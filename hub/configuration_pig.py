"""Configuration of a PIG Hub model: which backbone the adapter was fitted to, and the decoding defaults.

A PIG repository on the Hugging Face Hub is one model: the backbone's weights (keys backbone.*), the
Entropy Finite Differences Estimator (keys fde.*) and the two calibration tables (keys tables.<axis>.*)
in the sharded safetensors files, the tokenizer, this config (with the backbone's own config nested as
`backbone_config`) and the modeling code of both the backbone and PIG.
"""
from transformers import PretrainedConfig

from pig.api import DEFAULTS, FAMILIES   # hub.build replaces this line with the literal values


class PIGConfig(PretrainedConfig):
    model_type = "pig"

    def __init__(self, family="llada", backbone="GSAI-ML/LLaDA-8B-Instruct", backbone_revision=None,
                 backbone_config=None, defaults=None, adapter=None, tables=None, **kwargs):
        if family not in FAMILIES:
            raise ValueError(f"unknown family {family!r}; expected one of {FAMILIES}")
        self.family = family
        self.backbone = backbone
        self.backbone_revision = backbone_revision          # provenance: where the weights were copied from
        self.backbone_config = dict(backbone_config or {})    # the backbone's own config (its config.json fields)
        self.defaults = dict(DEFAULTS[family], **(defaults or {}))
        self.adapter = dict(adapter or {})      # provenance: parameters, epoch, validation correlation
        self.tables = list(tables or ["confidence", "entropy"])   # table axes stored in model.safetensors
        super().__init__(**kwargs)
