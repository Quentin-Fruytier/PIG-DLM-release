"""PIG (Parallel Inconsistency Guidance) as a transformers model.

    from transformers import AutoModel
    model = AutoModel.from_pretrained("UT-IFML/LLaDA-8B-Instruct-PIG", trust_remote_code=True)
    out = model.generate("Write a Python function that returns the n-th Fibonacci number.",
                         mode="pig.int", phi=0.1)
    print(out.text, out.forwards, out.tpf)

What from_pretrained loads: the adapter and the two calibration tables (both in model.safetensors)
from this repository, then the backbone from its own repository at the pinned revision (config.backbone).
The backbone's weights are never stored here. generate() is the code repository's `pig.api.PIG.generate`:
PIG.INT (mode="pig.int", budget phi), PIG.EXT (mode="pig.ext", budget theta, optional confidence floor)
or the confidence-threshold baseline (mode="confidence", threshold nu). The decoder, rules, tables,
adapter and that interface are the code repository's modules, concatenated verbatim into this file
(https://github.com/Quentin-Fruytier/PIG-DLM-release), so the two ways of loading PIG decode identically.
"""
import torch
from transformers import PreTrainedModel
from transformers.utils import cached_file

from .api import MODES, PIG, Output as PIGOutput, make_rule  # noqa: F401  (MODES, PIGOutput: public names)
from .configuration_pig import PIGConfig
from .decoder import FAMILY, PIGDecoder
from .fde import PIGFDE
from .tables import FlipTable


def load_tables(path, axes=("confidence", "entropy")):
    """The calibration tables stored in model.safetensors under keys tables.<axis>.<field>."""
    from safetensors import safe_open

    tables = {}
    with safe_open(str(path), framework="np") as f:
        for axis in axes:
            get = lambda key: f.get_tensor(f"tables.{axis}.{key}")
            tables[axis] = FlipTable(get("x_edges"), get("mu_edges"), get("rates"), axis,
                                     rows=get("rows"), flips=get("flips"))
    return tables


class PIGModel(PreTrainedModel):
    config_class = PIGConfig
    base_model_prefix = "pig"
    supports_gradient_checkpointing = False
    _keys_to_ignore_on_load_unexpected = [r"^backbone\.", r"^tables\."]   # tables are read by hand below

    def __init__(self, config):
        super().__init__(config)
        if config.family != FAMILY:
            raise ValueError(f"this code is for the {FAMILY!r} family, the config says {config.family!r}")
        self.fde = PIGFDE()
        self.tables = {}
        self._decoder = None          # the backbone + tokenizer + decode loop, attached after loading
        self._pig = None              # the generate() interface around it
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
        model.tables = load_tables(cached_file(pretrained_model_name_or_path, "model.safetensors", **hub),
                                   model.config.tables)
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
        self._pig = PIG(self._decoder, self.tables, self.config.defaults)
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

    # ---- decoding (pig.api.PIG) ----------------------------------------------------------------
    def rule(self, mode=None, phi=None, theta=None, floor=None, window=None, nu=None):
        """The per-round selection rule for `mode` with the given knobs (config defaults otherwise)."""
        return make_rule(self.tables, mode, phi=phi, theta=theta, floor=floor, window=window, nu=nu,
                         defaults=self.config.defaults)

    def _attached(self):
        if self._pig is None:
            raise RuntimeError("no backbone attached: load with from_pretrained(...) or call attach_backbone()")
        return self._pig

    def encode(self, prompt, chat_template=True):
        """Token ids for `prompt`: a string (wrapped in the backbone's chat template unless
        chat_template=False) or an iterable of token ids, returned as is."""
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
