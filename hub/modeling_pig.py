"""PIG (Parallel Inconsistency Guidance) as a transformers model.

    from transformers import AutoModel
    model = AutoModel.from_pretrained("UT-IFML/LLaDA-8B-Instruct-PIG", trust_remote_code=True)
    out = model.generate("Write a Python function that returns the n-th Fibonacci number.",
                         mode="pig.int", phi=0.1)
    print(out.text, out.forwards, out.tpf)

What from_pretrained loads: the adapter (model.safetensors) and the two calibration tables from this
repository, then the backbone from its own repository at the pinned revision (config.backbone). The
backbone's weights are never stored here. generate() runs one of three selection rules per decoding
round: PIG.INT (mode="pig.int", budget phi), PIG.EXT (mode="pig.ext", budget theta, optional
confidence floor) or the confidence-threshold baseline (mode="confidence", threshold nu). The decoders,
rules, tables and adapter in the sibling modules are vendored verbatim from the code repository
(https://github.com/Quentin-Fruytier/PIG-DLM-release), which also holds the evaluation harness.
"""
from dataclasses import dataclass, field

import torch
from transformers import PreTrainedModel
from transformers.utils import cached_file

from .configuration_pig import PIGConfig
from .decoder import FAMILY, PIGDecoder, PIGGeometry
from .fde import PIGFDE
from .rules import ConfidenceThreshold, PIGExt, PIGInt
from .tables import FlipTable

MODES = ("pig.int", "pig.ext", "confidence")


@dataclass
class PIGOutput:
    """What generate() returns."""
    text: str                       # the decoded generation (special tokens removed)
    tokens: list                    # generated token ids
    forwards: int                   # backbone forward passes spent
    tpf: float                      # tokens per forward pass, the paper's throughput measure
    mode: str
    params: dict = field(default_factory=dict)
    ms_per_round: dict = field(default_factory=dict)

    def __repr__(self):
        return (f"PIGOutput(mode={self.mode!r}, tokens={len(self.tokens)}, forwards={self.forwards}, "
                f"tpf={self.tpf:.2f}, text={self.text[:60]!r}{'...' if len(self.text) > 60 else ''})")


class PIGModel(PreTrainedModel):
    config_class = PIGConfig
    base_model_prefix = "pig"
    supports_gradient_checkpointing = False
    _keys_to_ignore_on_load_unexpected = [r"^backbone\."]

    def __init__(self, config):
        super().__init__(config)
        if config.family != FAMILY:
            raise ValueError(f"this code is for the {FAMILY!r} family, the config says {config.family!r}")
        self.fde = PIGFDE()
        self.tables = {}
        self._decoder = None          # the backbone + tokenizer + decode loop, attached after loading
        self.post_init()

    def _init_weights(self, module):
        pass                          # the adapter is always loaded from the checkpoint

    # ---- loading -------------------------------------------------------------------------------
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, load_backbone=True,
                        device=None, **kwargs):
        """Load the adapter and the tables from `pretrained_model_name_or_path`, then (unless
        load_backbone=False) the backbone from its own repository, onto `device` (default: the
        first CUDA device if there is one, else the CPU)."""
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        hub = {k: kwargs[k] for k in ("cache_dir", "revision", "token", "subfolder", "local_files_only",
                                      "force_download", "proxies") if k in kwargs}
        for axis, fname in model.config.tables.items():
            model.tables[axis] = FlipTable.load(cached_file(pretrained_model_name_or_path, fname, **hub))
        if load_backbone:
            model.attach_backbone(device=device)
        return model

    def attach_backbone(self, device=None):
        """Load the backbone and tokenizer (config.backbone at config.backbone_revision) and bind the
        decode loop to this adapter. Called by from_pretrained; call it yourself after
        from_pretrained(..., load_backbone=False)."""
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        kw = {"model_id": self.config.backbone}
        if self.config.backbone_revision:
            kw["revision"] = self.config.backbone_revision
        self._decoder = PIGDecoder(self.fde, device=device, **kw)
        return self

    @property
    def backbone(self):
        return None if self._decoder is None else self._decoder.model

    @property
    def tokenizer(self):
        return None if self._decoder is None else self._decoder.tokenizer

    def _apply(self, fn, *args, **kwargs):
        """Keep the backbone on the same device/dtype moves as the adapter (.to, .cuda, .half ...)."""
        super()._apply(fn, *args, **kwargs)
        if self._decoder is not None:
            self._decoder.model._apply(fn, *args, **kwargs)
            self._decoder.device = self.device
            self._decoder.embed = self._decoder.model.get_input_embeddings().weight
        return self

    def forward(self, *args, **kwargs):
        raise NotImplementedError("PIGModel has no forward of its own; use generate(). The adapter is "
                                  "model.fde and the backbone is model.backbone.")

    # ---- decoding ------------------------------------------------------------------------------
    def rule(self, mode=None, phi=None, theta=None, floor=None, window=None, nu=None):
        """The per-round selection rule for `mode` with the given knobs (config defaults otherwise)."""
        d = self.config.defaults
        mode = mode or d["mode"]
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
        if mode == "pig.int":
            if theta is not None or floor is not None or window is not None:
                raise ValueError("mode='pig.int' takes phi only (theta, floor and window belong to pig.ext)")
            return PIGInt(self.tables["confidence"], d["phi"] if phi is None else phi)
        if mode == "pig.ext":
            if phi is not None:
                raise ValueError("mode='pig.ext' takes theta (and optionally floor, window), not phi")
            return PIGExt(self.tables["entropy"], d["theta"] if theta is None else theta,
                          window=d["window"] if window is None else window,
                          floor=d["floor"] if floor is None else floor)
        if phi is not None or theta is not None or floor is not None or window is not None:
            raise ValueError("mode='confidence' takes nu only")
        return ConfidenceThreshold(d["nu"] if nu is None else nu)

    def encode(self, prompt, chat_template=True):
        """Token ids for `prompt`: a string (wrapped in the backbone's chat template unless
        chat_template=False) or an iterable of token ids, returned as is."""
        if isinstance(prompt, str):
            if chat_template:
                return list(self._decoder.chat_prompt(prompt))
            return self._decoder.tokenizer(prompt).input_ids
        return [int(t) for t in prompt]

    @torch.no_grad()
    def generate(self, prompt, mode=None, phi=None, theta=None, floor=None, window=None, nu=None,
                 geometry=None, block=None, max_new_tokens=None, chat_template=True):
        """Decode a completion of `prompt` with the chosen rule.

        mode            "pig.int" (knob phi), "pig.ext" (knobs theta, floor, window) or
                        "confidence" (knob nu); defaults in config.defaults
        geometry, block the candidate geometry: LLaDA "full" | "block" | "sliding" over a fixed
                        256-token canvas; Nemotron "block" | "sliding" | "nested" windows of
                        `block` tokens
        max_new_tokens  LLaDA: the canvas length (the adapter was fitted at 256);
                        Nemotron: the generation budget (stops at end-of-sequence)
        Returns a PIGOutput with the text, the token ids, the forward passes and tokens per forward.
        """
        if self._decoder is None:
            raise RuntimeError("no backbone attached: load with from_pretrained(...) or call attach_backbone()")
        d = self.config.defaults
        rule = self.rule(mode, phi=phi, theta=theta, floor=floor, window=window, nu=nu)
        geom = PIGGeometry(geometry or d["geometry"], d["block"] if block is None else int(block))
        ids = self.encode(prompt, chat_template=chat_template)
        if FAMILY == "llada":
            length = d["canvas"] if max_new_tokens is None else int(max_new_tokens)
            tokens, forwards, ms = self._decoder.generate(ids, rule, geom, canvas=length)
        else:
            length = d["max_new"] if max_new_tokens is None else int(max_new_tokens)
            tokens, forwards, ms = self._decoder.generate(ids, rule, geom, max_new=length)
        text = self._decoder.tokenizer.decode(tokens, skip_special_tokens=True)
        params = dict(geometry=geom.kind, block=geom.size)
        params.update({k: v for k, v in dict(phi=phi, theta=theta, floor=floor, window=window, nu=nu).items()
                       if v is not None})
        return PIGOutput(text=text, tokens=list(tokens), forwards=int(forwards),
                         tpf=len(tokens) / max(int(forwards), 1), mode=mode or d["mode"],
                         params=params, ms_per_round=dict(ms))
