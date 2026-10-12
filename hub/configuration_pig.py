"""Configuration of a PIG Hub model: which backbone the adapter was fitted to, and the decoding defaults.

A PIG repository on the Hugging Face Hub holds the Entropy Finite Differences Estimator (the adapter,
model.safetensors), the two calibration tables (flip_confidence.npz, flip_entropy.npz) and this code.
The backbone itself is loaded from its own repository at the pinned revision.
"""
from transformers import PretrainedConfig

FAMILIES = ("llada", "nemotron")

DEFAULTS = {
    # one entry per family: the knobs generate() uses when the caller does not pass them
    "llada": dict(mode="pig.int", phi=0.1, theta=0.5, floor=None, window=16, nu=0.9,
                  geometry="block", block=32, canvas=256),
    "nemotron": dict(mode="pig.int", phi=0.1, theta=0.5, floor=None, window=16, nu=0.9,
                     geometry="block", block=32, max_new=2048),
}


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
