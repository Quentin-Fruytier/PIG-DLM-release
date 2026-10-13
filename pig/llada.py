"""LLaDA-8B-Instruct backend.

The prompt is followed by a fixed canvas of 256 mask tokens and decoded without a KV cache: every
round is one forward pass over prompt + canvas, then one call to a selection rule, then a parallel
commit of the argmax tokens it returns. Rounds continue until the canvas is full, so throughput is
TPF = 256 / (number of rounds).

Candidate geometry (which masked positions a rule may commit this round):
    full            every masked canvas position
    block B         the masked positions of the leftmost canvas block [kB, (k+1)B) that still has
                    one (semi-autoregressive blocks)
    sliding B       the B leftmost masked positions (a window that advances as commits land)

The adapter reads the final layer's attention map. LLaDA does not return attention weights, so the
final block's attention is computed explicitly (softmax(QK^T / sqrt(d)) V, the same operation its
SDPA call performs) and its head average is kept.
"""
import math
import time
import types
from dataclasses import dataclass

import numpy as np
import torch

from .fde import LLaDAFDE, load_fde
from .rules import Round

MODEL_ID = "GSAI-ML/LLaDA-8B-Instruct"
REVISION = "08b83a6feb34df1a6011b80c3c00c7563e963b07"
MASK_ID = 126336
CANVAS = 256


@dataclass(frozen=True)
class Geometry:
    kind: str = "full"        # "full" | "block" | "sliding"
    size: int = 0             # block / window length (ignored for "full")

    def __post_init__(self):
        if self.kind not in ("full", "block", "sliding"):
            raise ValueError(f"unknown geometry {self.kind!r}")
        if self.kind != "full" and self.size <= 0:
            raise ValueError(f"geometry {self.kind!r} needs a positive size")

    def eligible(self, masked):
        """masked: ascending canvas positions that are still masked."""
        if self.kind == "full":
            return masked
        if self.kind == "sliding":
            return masked[:self.size]
        b0 = min(p // self.size for p in masked)
        return [p for p in masked if p // self.size == b0]

    def tag(self):
        return "full" if self.kind == "full" else f"{self.kind}{self.size}"


def _transformer_blocks(model):
    for path in (("model", "transformer", "blocks"), ("transformer", "blocks"), ("model", "blocks")):
        obj = model
        for name in path:
            obj = getattr(obj, name, None)
            if obj is None:
                break
        if obj is not None:
            return list(obj)
    raise RuntimeError("could not locate the LLaDA transformer blocks")


def _capture_final_attention(model):
    """Replace the final block's attention kernel by the explicit computation and keep the
    head-averaged attention map of the latest forward in `block._pig_attn`."""
    block = _transformer_blocks(model)[-1]

    def attention(self, q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False):
        nh, nkv = q.size(1), k.size(1)
        if nh != nkv:
            k = k.repeat_interleave(nh // nkv, dim=1)
            v = v.repeat_interleave(nh // nkv, dim=1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        if attn_mask is not None:
            scores = scores + attn_mask
        attn = scores.softmax(dim=-1)
        self._pig_attn = attn.mean(dim=1).detach()
        return torch.matmul(attn, v)

    block._scaled_dot_product_attention = types.MethodType(attention, block)
    return block


class LLaDADecoder:
    family = "llada"
    geometry_cls = Geometry

    def __init__(self, fde_path, device="cuda", model_id=MODEL_ID, revision=REVISION):
        from transformers import AutoModel, AutoTokenizer

        self.device = device
        self.model_id, self.revision = model_id, revision
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_id, revision=revision, trust_remote_code=True,
                                               torch_dtype=torch.bfloat16).to(device).eval()
        self._final = _capture_final_attention(self.model)
        self._hidden = {}
        self.head = self.model.get_output_embeddings()
        self.head.register_forward_hook(lambda m, inp, out: self._hidden.__setitem__("h", inp[0].detach()))
        self.embed = self.model.get_input_embeddings().weight
        self.fde = load_fde(LLaDAFDE, fde_path, device)   # a path, or an already-built LLaDAFDE

    def chat_prompt(self, text):
        return self.tokenizer.apply_chat_template([{"role": "user", "content": text}],
                                                  add_generation_prompt=True, tokenize=True)

    @torch.no_grad()
    def _read(self, cur, lo, hi):
        out = self.model(cur.unsqueeze(0))
        logits = out.logits[0, lo:hi].float()
        A = self._final._pig_attn[0].float().cpu().numpy()
        lp = torch.log_softmax(logits, -1)
        H = -(lp.exp() * lp).sum(-1).cpu().numpy()
        conf, am = lp.exp().max(-1)
        return H, conf.cpu().numpy(), am, A

    @torch.no_grad()
    def _influence(self, cur, lo, hi, H, conf, revealed, A_t, A_prev):
        """One adapter pass -> D_hat (L, L) over the canvas positions."""
        L = hi - lo
        h = self._hidden["h"][0, lo:hi]
        decoded = np.zeros(L, np.float32)
        cls = np.ones(L, np.int64)                       # 1 = committable (masked)
        for p in revealed:
            decoded[p] = 1.0
            cls[p] = 0                                   # 0 = decoded
        cls[0] = 3                                       # 3 = canvas boundary
        cls[L - 1] = 3
        t = len(revealed) / L
        scal_np = np.column_stack([np.asarray(H, np.float32), np.asarray(conf, np.float32),
                                   np.zeros(L, np.float32), decoded,
                                   np.eye(4, dtype=np.float32)[cls], np.full(L, t, np.float32)])
        top2 = torch.softmax(self.head(h).float(), -1).topk(2, -1).values
        scal = torch.as_tensor(scal_np, device=h.device)
        scal[:, 2] = (top2[:, 0] - top2[:, 1]).float()
        e = self.embed[cur[lo:hi]].float()
        D = self.fde(h.float(), e, scal, torch.as_tensor(A_t, device=h.device).float(),
                     torch.as_tensor(A_prev, device=h.device).float(), torch.tensor(t, device=h.device))
        return D.cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def generate(self, prompt_ids, rule, geometry, canvas=CANVAS):
        """Decode `canvas` tokens after `prompt_ids`.

        rule: callable(Round) -> list of canvas positions to commit (non-empty).
        Returns (generated token ids, number of forward passes, per-round timings in ms).
        """
        ids = list(prompt_ids) + [MASK_ID] * canvas
        cur = torch.tensor(ids, device=self.device)
        lo, hi = len(prompt_ids), len(prompt_ids) + canvas
        L = canvas
        revealed = set()
        A_prev = None
        rounds = 0
        timing = {"forward": 0.0, "adapter": 0.0, "select": 0.0}
        sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
        while len(revealed) < L:
            t0 = time.perf_counter()
            H, conf, am, A = self._read(cur, lo, hi)
            sync()
            t1 = time.perf_counter()
            masked = [p for p in range(L) if p not in revealed]
            cand = geometry.eligible(masked)
            sub = A[lo:hi, lo:hi]
            A_t = (0.5 * (sub + sub.T)).astype(np.float16).astype(np.float32)
            if A_prev is None:
                A_prev = A_t
            D = self._influence(cur, lo, hi, H, conf, revealed, A_t, A_prev)
            A_prev = A_t
            t2 = time.perf_counter()
            S = rule(Round(cand=cand, conf=conf, H=H, D=D))
            if not S:
                raise RuntimeError("selection rule returned an empty set")
            sel = torch.tensor(S, device=cur.device)
            cur[lo + sel] = am[sel]                      # one write for the whole parallel set
            revealed.update(int(p) for p in S)
            rounds += 1
            t3 = time.perf_counter()
            timing["forward"] += t1 - t0
            timing["adapter"] += t2 - t1
            timing["select"] += t3 - t2
        ms = {k: 1e3 * v / max(rounds, 1) for k, v in timing.items()}
        return cur[lo:hi].tolist(), rounds, ms
