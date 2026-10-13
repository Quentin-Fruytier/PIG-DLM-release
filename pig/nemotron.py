"""Nemotron-Labs-Diffusion-3B backend.

The model is block-causal: the prompt (and every settled prefix) lives in a causal KV cache, and a
window of positions right of the cache is denoised with bidirectional attention inside the window.
Decoding starts with one causal pass over the prompt, whose last-position prediction becomes the
first generated token. Each round then runs one forward over the current window, calls a selection
rule, and commits the argmax tokens it returns. Once the leading run of the window is fully decoded
it is appended to the cache with a causal forward and the window moves right.

Window geometries (block length B):
    block     a fixed slab [w0, w0 + B); the cache advances only when the slab is fully decoded
    sliding   the window extends right from the cache boundary until it holds B masked positions;
              the cache advances by the leading decoded run once that run reaches B // 2
    nested    as sliding, but the window extends until it holds 2B masked positions while only the
              B leftmost masked positions are candidates (a B-wide commit band inside a 2B-wide window)

End of sequence (the release's generate() rule): the forward always covers the whole window;
positions at or after the first committed end-of-sequence token are never candidates, and decoding
stops once an end-of-sequence token exists and nothing before it is still masked.

Throughput is TPF = generated tokens (up to the first end-of-sequence token) / forward passes, where
forward passes count the prompt pass, every window forward and every cache update.
"""
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .fde import NemotronFDE, load_fde
from .rules import Round

MODEL_ID = "nvidia/Nemotron-Labs-Diffusion-3B"
REVISION = "0d51902da1f8869f83413ce642fab402fa5641e0"
MAX_NEW = 2048


@dataclass(frozen=True)
class Window:
    kind: str = "block"       # "block" | "sliding" | "nested"
    size: int = 32            # block length B

    def __post_init__(self):
        if self.kind not in ("block", "sliding", "nested"):
            raise ValueError(f"unknown window geometry {self.kind!r}")
        if self.size <= 0:
            raise ValueError("window size must be positive")

    @property
    def compute_masked(self):
        return 2 * self.size if self.kind == "nested" else self.size

    @property
    def stride(self):
        return max(1, self.size // 2)

    def tag(self):
        return f"{self.kind}{self.size}"


def _extend_to_masked(canvas, w0, n_masked, mask_id, end):
    """Smallest hi such that canvas[w0:hi] holds `n_masked` masked positions (or `end`)."""
    pos = (canvas[w0:end] == mask_id).nonzero()
    if len(pos) >= n_masked:
        return w0 + int(pos[n_masked - 1]) + 1
    return end


class NemotronDecoder:
    family = "nemotron"
    geometry_cls = Window

    def __init__(self, fde_path, device="cuda", model_id=MODEL_ID, revision=REVISION):
        from transformers import AutoModel, AutoTokenizer

        self.device = device
        self.model_id, self.revision = model_id, revision
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=True)
        self.model = (AutoModel.from_pretrained(model_id, revision=revision, trust_remote_code=True)
                      .to(device).to(torch.bfloat16).eval())
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.mask_id = self.model.mask_token_id
        self.eos_id = self.tokenizer.eos_token_id
        self.embed = self.model.get_input_embeddings().weight
        self.fde = load_fde(NemotronFDE, fde_path, device)   # a path, or an already-built NemotronFDE

    def chat_prompt(self, text):
        prompt = self.tokenizer.apply_chat_template([{"role": "user", "content": text}],
                                                    tokenize=False, add_generation_prompt=True)
        return self.tokenizer(prompt, return_tensors="pt").input_ids[0].tolist()

    def _set_diffusion(self, on):
        for layer in self.model.encoder.layers:
            if hasattr(layer.self_attn, "diffusion_lm"):
                layer.self_attn.diffusion_lm = on

    @torch.no_grad()
    def _read(self, win, cache):
        h = self.model(win.unsqueeze(0), past_key_values=cache, use_cache=False,
                       output_last_hidden_states_only=True).last_hidden_state[0]
        logits = self.model.diffusion_head(h).float()
        P = F.softmax(logits, -1)
        top2 = P.topk(2, -1).values
        lp = torch.log_softmax(logits, -1)
        H = (-(P * lp).sum(-1)).cpu().numpy()
        return h, H, top2[:, 0].cpu().numpy(), logits.argmax(-1), (top2[:, 0] - top2[:, 1]).cpu().numpy()

    @torch.no_grad()
    def _influence(self, h, win, H, conf, margin, masked):
        """One adapter pass -> D_hat (T, T) over the window positions."""
        n = len(masked)
        cls = np.where(masked, 1, 0).astype(np.int64)    # 1 = masked, 0 = decoded
        cls[0] = 2                                       # 2 = window head
        fill = float((~masked).sum()) / max(n, 1)
        scal = np.column_stack([H.astype(np.float32), conf.astype(np.float32),
                                margin.astype(np.float32), (~masked).astype(np.float32),
                                np.eye(3, dtype=np.float32)[cls], np.full(n, fill, np.float32),
                                np.full(n, np.log2(n) / 7.0, np.float32),
                                np.arange(n, dtype=np.float32) / n])
        D = self.fde(h.float(), self.embed[win].float(), torch.as_tensor(scal, device=h.device),
                     torch.tensor(fill, device=h.device))
        return D.cpu().numpy()

    @torch.no_grad()
    def generate(self, prompt_ids, rule, window, max_new=MAX_NEW):
        """Decode up to `max_new` tokens after `prompt_ids` (rounded down to a multiple of B).

        rule: callable(Round) -> list of window positions to commit (non-empty).
        Returns (generated ids up to and excluding the first end-of-sequence token,
                 number of forward passes, per-round timings in ms).
        """
        W = window.size
        GEN = max(W, (max_new // W) * W)
        pid = torch.tensor([prompt_ids], device=self.device)
        mid, eos = self.mask_id, self.eos_id
        forwards = 0
        timing = {"forward": 0.0, "adapter": 0.0, "select": 0.0}
        rounds = 0
        self._set_diffusion(False)
        out = self.model(pid, use_cache=True, use_causal_mask=True)
        forwards += 1
        cache = out.past_key_values
        self._set_diffusion(True)
        canvas = torch.full((GEN,), mid, dtype=pid.dtype, device=self.device)
        canvas[0] = int(out.logits[0, -1].argmax())
        w0 = 0
        while w0 < GEN:
            if window.kind == "block":
                hi = min(w0 + W, GEN)
            else:
                hi = _extend_to_masked(canvas, w0, window.compute_masked, mid, GEN)
            first_eos = None
            hits = (canvas == eos).nonzero()
            if len(hits):
                first_eos = int(hits[0])
                if not (canvas[w0:first_eos] == mid).any():
                    break                                 # everything before the end is decoded
            win = canvas[w0:hi]
            if (win == mid).sum() == 0:
                break
            t0 = time.perf_counter()
            h, H, conf, am, margin = self._read(win, cache)
            forwards += 1
            t1 = time.perf_counter()
            masked = (win == mid).cpu().numpy()
            open_ = masked.copy()
            if first_eos is not None:
                open_[max(0, first_eos - w0):] = False    # never decode at or after the end
            cand = np.where(open_)[0]
            if len(cand) == 0:
                break
            if window.kind == "nested":
                cand = cand[:W]
            D = self._influence(h, win, H, conf, margin, masked)
            t2 = time.perf_counter()
            S = rule(Round(cand=[int(c) for c in cand], conf=conf, H=H, D=D))
            if not S:
                raise RuntimeError("selection rule returned an empty set")
            sel = torch.tensor(S, device=canvas.device)
            canvas[w0 + sel] = am[sel]                   # one write for the whole parallel set
            rounds += 1
            t3 = time.perf_counter()
            timing["forward"] += t1 - t0
            timing["adapter"] += t2 - t1
            timing["select"] += t3 - t2
            # advance the cache over the leading decoded run
            rest = (canvas[w0:hi] == mid).nonzero()
            adv = int(rest[0]) if len(rest) else (hi - w0)
            if window.kind == "block":
                adv = 0 if (canvas[w0:hi] == mid).any() else (hi - w0)
            if adv > 0 and (adv >= window.stride or adv >= hi - w0):
                self._set_diffusion(False)
                out = self.model(canvas[w0:w0 + adv].unsqueeze(0), past_key_values=cache,
                                 use_cache=True, use_causal_mask=True)
                forwards += 1
                cache = out.past_key_values
                self._set_diffusion(True)
                w0 += adv
        gen = canvas[:GEN]
        hits = (gen == eos).nonzero()
        n_gen = int(hits[0]) if len(hits) else GEN
        ms = {k: 1e3 * v / max(rounds, 1) for k, v in timing.items()}
        return gen[:n_gen].tolist(), forwards, ms
