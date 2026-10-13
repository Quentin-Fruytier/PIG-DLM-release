"""PIG (Parallel Inconsistency Guidance) as one transformers model: the backbone, its adapter and tables.

    from transformers import AutoModel
    model = AutoModel.from_pretrained("UT-IFML/LLaDA-8B-Instruct-PIG", trust_remote_code=True)
    out = model.generate("Write a Python function that returns the n-th Fibonacci number.",
                         mode="pig.int", phi=0.1)
    print(out.text, out.forwards, out.tpf)

from_pretrained loads everything from this repository: the backbone's weights (keys backbone.*), the
adapter (fde.*) and the two calibration tables (tables.*), then moves the model to `device` (the first
CUDA device by default) and binds the decode loop to the tokenizer shipped here. generate() is the code
repository's `pig.api.PIG.generate`: PIG.INT (mode="pig.int", budget phi), PIG.EXT (mode="pig.ext",
budget theta, optional confidence floor) or the confidence-threshold baseline (mode="confidence",
threshold nu). The backbone's modeling code is in the sibling modeling file; the decoder, rules, tables,
adapter and this interface are the code repository's modules, concatenated verbatim into this file
(https://github.com/Quentin-Fruytier/PIG-DLM-release), so the Hub model and the library decode identically.
"""
import json
import os

import torch
from transformers import AutoTokenizer, PreTrainedModel
from transformers.utils import cached_file

from .api import MODES, PIG, Output as PIGOutput, make_rule  # noqa: F401  (MODES, PIGOutput: public names)
from .backbone import PIGBackbone, PIGBackboneConfig         # resolved by hub.build to the family's classes
from .configuration_pig import PIGConfig
from .decoder import FAMILY, PIGDecoder
from .fde import PIGFDE
from .tables import FlipTable


def load_tables(path, axes=("confidence", "entropy"), **hub):
    """The calibration tables stored under keys tables.<axis>.<field> in this repository's weights."""
    from safetensors import safe_open

    index = cached_file(path, "model.safetensors.index.json", _raise_exceptions_for_missing_entries=False, **hub)
    if index:
        weight_map = json.load(open(index))["weight_map"]
        files = {weight_map[f"tables.{axis}.rates"] for axis in axes}
    else:
        files = {"model.safetensors"}
    tensors = {}
    for fname in files:
        with safe_open(cached_file(path, fname, **hub), framework="np") as f:
            for key in f.keys():
                if key.startswith("tables."):
                    tensors[key] = f.get_tensor(key)
    return {axis: FlipTable(tensors[f"tables.{axis}.x_edges"], tensors[f"tables.{axis}.mu_edges"],
                            tensors[f"tables.{axis}.rates"], axis,
                            rows=tensors[f"tables.{axis}.rows"], flips=tensors[f"tables.{axis}.flips"]) for axis in axes}


class PIGModel(PreTrainedModel):
    config_class = PIGConfig
    base_model_prefix = "pig"
    supports_gradient_checkpointing = False
    _keys_to_ignore_on_load_unexpected = [r"^tables\."]     # read by load_tables, not state_dict
    _keep_in_fp32_modules_strict = ["fde"]                   # the adapter stays fp32 whatever dtype the backbone takes

    def __init__(self, config):
        super().__init__(config)
        if config.family != FAMILY:
            raise ValueError(f"this code is for the {FAMILY!r} family, the config says {config.family!r}")
        self.backbone = PIGBackbone(PIGBackboneConfig(**config.backbone_config))
        self.fde = PIGFDE()
        self.tables = {}
        self._decoder = None          # the decode loop over backbone + tokenizer, bound by attach()
        self._pig = None              # the generate() interface around it
        self.post_init()

    def _init_weights(self, module):
        pass                          # everything is loaded from the checkpoint

    # ---- loading -------------------------------------------------------------------------------
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, device=None, attach=True, **kwargs):
        """Load the whole model from `pretrained_model_name_or_path`, then (unless attach=False) move it to
        `device` (default: the first CUDA device if there is one, else the CPU) and bind the decode loop."""
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        hub = {k: kwargs[k] for k in ("cache_dir", "revision", "token", "subfolder", "local_files_only",
                                      "force_download", "proxies") if k in kwargs}
        model.tables = load_tables(pretrained_model_name_or_path, model.config.tables, **hub)
        model._hub = (pretrained_model_name_or_path, hub)
        if attach:
            model.attach(device=device)
        return model

    def attach(self, device=None):
        """Move the model to `device` and bind the decode loop to the tokenizer of this repository."""
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        path, hub = getattr(self, "_hub", (self.name_or_path, {}))
        tokenizer = AutoTokenizer.from_pretrained(path, **hub)
        self.to(device)
        self._decoder = PIGDecoder(self.fde, device=device, model=self.backbone, tokenizer=tokenizer,
                                   model_id=self.config.backbone, revision=self.config.backbone_revision)
        self._pig = PIG(self._decoder, self.tables, self.config.defaults)
        return self

    @property
    def tokenizer(self):
        return None if self._decoder is None else self._decoder.tokenizer

    def _apply(self, fn, *args, **kwargs):
        """Keep the decode loop's device bookkeeping in step with .to / .cuda / .half ... """
        super()._apply(fn, *args, **kwargs)
        if self._decoder is not None:
            self._decoder.device = self.device
            self._decoder.embed = self.backbone.get_input_embeddings().weight
        return self

    def forward(self, *args, **kwargs):
        raise NotImplementedError("PIGModel has no forward of its own; use generate(). The backbone is "
                                  "model.backbone and the adapter is model.fde.")

    # ---- decoding (pig.api.PIG) ----------------------------------------------------------------
    def rule(self, mode=None, phi=None, theta=None, floor=None, window=None, nu=None):
        """The per-round selection rule for `mode` with the given knobs (config defaults otherwise)."""
        return make_rule(self.tables, mode, phi=phi, theta=theta, floor=floor, window=window, nu=nu,
                         defaults=self.config.defaults)

    def _attached(self):
        if self._pig is None:
            raise RuntimeError("no decode loop attached: load with from_pretrained(...) or call attach()")
        return self._pig

    def encode(self, prompt, chat_template=True):
        """Token ids for `prompt`: a string (wrapped in the chat template unless chat_template=False) or
        an iterable of token ids, returned as is."""
        return self._attached().encode(prompt, chat_template=chat_template)

    def generate(self, prompt, mode=None, phi=None, theta=None, floor=None, nu=None, window=None,
                 geometry=None, block=None, max_new_tokens=None, chat_template=True):
        """Decode a completion of `prompt` with the chosen rule; see pig.api.PIG.generate.

        mode            "pig.int" (knob phi), "pig.ext" (knobs theta, floor, window) or
                        "confidence" (knob nu); defaults in config.defaults
        geometry, block the candidate geometry: LLaDA "full" | "block" | "sliding" over a fixed
                        256-token canvas; Nemotron "block" | "sliding" | "nested" windows of
                        `block` tokens
        max_new_tokens  LLaDA: the canvas length (the adapter was fitted at 256);
                        Nemotron: the generation budget (stops at end-of-sequence)
        Returns a PIGOutput with the text, the token ids, the forward passes and tokens per forward.
        """
        return self._attached().generate(prompt, mode=mode, phi=phi, theta=theta, floor=floor, nu=nu,
                                         window=window, geometry=geometry, block=block,
                                         max_new_tokens=max_new_tokens, chat_template=chat_template)
