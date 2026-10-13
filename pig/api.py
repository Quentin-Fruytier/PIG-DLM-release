"""Generate with PIG: a backbone, its adapter and tables behind one generate() call.

    import pig
    model = pig.load("llada")                      # or "nemotron"; see pig/loading.py
    out = model.generate("Write a Python function that returns the n-th Fibonacci number.",
                         mode="pig.int", phi=0.1, geometry="block", block=64)
    print(out.text, out.forwards, out.tpf)

generate() runs one of three per-round selection rules: PIG.INT (mode="pig.int", budget phi), PIG.EXT
(mode="pig.ext", budget theta, optional confidence floor and look-ahead window) or the
confidence-threshold baseline (mode="confidence", threshold nu). The Hugging Face Hub wrapper
(hub/modeling_pig.py) exposes this same class, so the two ways of loading PIG decode identically.
"""
from dataclasses import dataclass, field

import torch

from .rules import ConfidenceThreshold, PIGExt, PIGInt

FAMILIES = ("llada", "nemotron")
MODES = ("pig.int", "pig.ext", "confidence")

# the knobs generate() uses when the caller does not pass them, per backbone family
DEFAULTS = {
    "llada": dict(mode="pig.int", phi=0.1, theta=0.5, floor=None, window=16, nu=0.9,
                  geometry="block", block=32, canvas=256),
    "nemotron": dict(mode="pig.int", phi=0.1, theta=0.5, floor=None, window=16, nu=0.9,
                     geometry="block", block=32, max_new=2048),
}


@dataclass
class Output:
    """What generate() returns."""
    text: str                       # the decoded generation (special tokens removed)
    tokens: list                    # generated token ids
    forwards: int                   # backbone forward passes spent
    tpf: float                      # tokens per forward pass, the paper's throughput measure
    mode: str
    params: dict = field(default_factory=dict)
    ms_per_round: dict = field(default_factory=dict)

    def __repr__(self):
        return (f"Output(mode={self.mode!r}, tokens={len(self.tokens)}, forwards={self.forwards}, "
                f"tpf={self.tpf:.2f}, text={self.text[:60]!r}{'...' if len(self.text) > 60 else ''})")


def make_rule(tables, mode=None, phi=None, theta=None, floor=None, window=None, nu=None, defaults=None):
    """The per-round selection rule for `mode` with the given knobs; `defaults` fills what is not passed.

    tables: {"confidence": FlipTable, "entropy": FlipTable}; the baseline reads neither.
    """
    d = defaults or {}
    mode = mode or d.get("mode", "pig.int")
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")

    def knob(name, value, required=False):
        value = d.get(name) if value is None else value
        if required and value is None:
            raise ValueError(f"mode={mode!r} needs {name}")
        return value

    if mode == "pig.int":
        if theta is not None or floor is not None or window is not None or nu is not None:
            raise ValueError("mode='pig.int' takes phi only (theta, floor and window belong to pig.ext, nu to confidence)")
        return PIGInt(tables["confidence"], knob("phi", phi, required=True))
    if mode == "pig.ext":
        if phi is not None or nu is not None:
            raise ValueError("mode='pig.ext' takes theta (and optionally floor, window); phi belongs to pig.int, nu to confidence")
        return PIGExt(tables["entropy"], knob("theta", theta, required=True),
                      window=knob("window", window) or 16, floor=knob("floor", floor))
    if phi is not None or theta is not None or floor is not None or window is not None:
        raise ValueError("mode='confidence' takes nu only")
    return ConfidenceThreshold(knob("nu", nu, required=True))


class PIG:
    """A loaded backbone with its adapter and tables. Build one with pig.load(family)."""

    def __init__(self, decoder, tables, defaults=None):
        self.decoder = decoder                  # LLaDADecoder or NemotronDecoder (pig/llada.py, pig/nemotron.py)
        self.tables = tables                    # {"confidence": FlipTable, "entropy": FlipTable}
        self.family = decoder.family
        if self.family not in FAMILIES:
            raise ValueError(f"unknown family {self.family!r}; expected one of {FAMILIES}")
        self.defaults = dict(DEFAULTS[self.family], **(defaults or {}))

    @property
    def backbone(self):
        return self.decoder.model

    @property
    def tokenizer(self):
        return self.decoder.tokenizer

    def rule(self, mode=None, phi=None, theta=None, floor=None, window=None, nu=None):
        return make_rule(self.tables, mode, phi=phi, theta=theta, floor=floor, window=window, nu=nu,
                         defaults=self.defaults)

    def geometry(self, geometry=None, block=None):
        """The candidate geometry: LLaDA "full" | "block" | "sliding"; Nemotron "block" | "sliding" | "nested"."""
        d = self.defaults
        return self.decoder.geometry_cls(geometry or d["geometry"], d["block"] if block is None else int(block))

    def encode(self, prompt, chat_template=True):
        """Token ids for `prompt`: a string (wrapped in the backbone's chat template unless
        chat_template=False) or an iterable of token ids, returned as is."""
        if isinstance(prompt, str):
            if chat_template:
                return list(self.decoder.chat_prompt(prompt))
            return self.decoder.tokenizer(prompt).input_ids
        return [int(t) for t in prompt]

    @torch.no_grad()
    def generate(self, prompt, mode=None, phi=None, theta=None, floor=None, window=None, nu=None,
                 geometry=None, block=None, max_new_tokens=None, chat_template=True):
        """Decode a completion of `prompt` with the chosen rule.

        mode            "pig.int" (knob phi), "pig.ext" (knobs theta, floor, window) or "confidence"
                        (knob nu); defaults in self.defaults
        geometry, block the candidate geometry and its size B
        max_new_tokens  LLaDA: the canvas length (the adapter was fitted at 256);
                        Nemotron: the generation budget (stops at end-of-sequence)
        Returns an Output with the text, the token ids, the forward passes and tokens per forward.
        """
        d = self.defaults
        rule = self.rule(mode, phi=phi, theta=theta, floor=floor, window=window, nu=nu)
        geom = self.geometry(geometry, block)
        ids = self.encode(prompt, chat_template=chat_template)
        if self.family == "llada":
            length = d["canvas"] if max_new_tokens is None else int(max_new_tokens)
            tokens, forwards, ms = self.decoder.generate(ids, rule, geom, canvas=length)
        else:
            length = d["max_new"] if max_new_tokens is None else int(max_new_tokens)
            tokens, forwards, ms = self.decoder.generate(ids, rule, geom, max_new=length)
        text = self.decoder.tokenizer.decode(tokens, skip_special_tokens=True)
        params = dict(geometry=geom.kind, block=geom.size)
        params.update({k: v for k, v in dict(phi=phi, theta=theta, floor=floor, window=window, nu=nu).items()
                       if v is not None})
        return Output(text=text, tokens=list(tokens), forwards=int(forwards),
                      tpf=len(tokens) / max(int(forwards), 1), mode=mode or d["mode"],
                      params=params, ms_per_round=dict(ms))
