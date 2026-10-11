"""Building blocks shared by the Entropy Finite Differences Estimator (FDE) variants.

The estimator is a small transformer over canvas positions. It reads features of the frozen
backbone's current forward pass and returns, in the same pass, the full directed matrix of
predicted entropy finite differences

    D_hat[i, j] ~ H_i(x) - H_i(x with position j committed to its argmax).

Readout: D_hat = f(z)^T W g(z) / sqrt(96) + b, with separate receiver (f) and source (g) maps,
so D_hat is directed.
"""
import math

import torch
import torch.nn as nn

D_MODEL = 192
N_HEADS = 6
HEAD_DIM = 32
FFN = 768
READOUT = 96


def rope(x):
    """Rotary position embedding over the last dimension of x: (heads, T, head_dim)."""
    T, hd = x.shape[-2], x.shape[-1]
    half = hd // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=x.device) / half)
    ang = torch.arange(T, device=x.device)[:, None] * freqs[None, :]
    cos, sin = torch.cos(ang), torch.sin(ang)
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1)


class Block(nn.Module):
    """Pre-norm transformer layer with an additive per-head attention bias and FiLM on the
    decode fraction t."""

    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(D_MODEL)
        self.qkv = nn.Linear(D_MODEL, 3 * D_MODEL)
        self.proj = nn.Linear(D_MODEL, D_MODEL)
        self.ln2 = nn.LayerNorm(D_MODEL)
        self.ff = nn.Sequential(nn.Linear(D_MODEL, FFN), nn.GELU(), nn.Linear(FFN, D_MODEL))
        self.film = nn.Sequential(nn.Linear(1, 64), nn.GELU(), nn.Linear(64, 2 * D_MODEL))

    def forward(self, x, bias, t):
        T = x.shape[0]
        q, k, v = self.qkv(self.ln1(x)).chunk(3, -1)
        q = rope(q.view(T, N_HEADS, HEAD_DIM).transpose(0, 1))
        k = rope(k.view(T, N_HEADS, HEAD_DIM).transpose(0, 1))
        v = v.view(T, N_HEADS, HEAD_DIM).transpose(0, 1)
        att = (q @ k.transpose(-1, -2)) / math.sqrt(HEAD_DIM) + bias
        x = x + self.proj((att.softmax(-1) @ v).transpose(0, 1).reshape(T, D_MODEL))
        x = x + self.ff(self.ln2(x))
        sc, sh = self.film(t.view(1, 1)).view(2, D_MODEL)
        return x * (1 + sc) + sh


def readout_maps():
    return (nn.Sequential(nn.Linear(D_MODEL, D_MODEL), nn.GELU(), nn.Linear(D_MODEL, READOUT)),
            nn.Sequential(nn.Linear(D_MODEL, D_MODEL), nn.GELU(), nn.Linear(D_MODEL, READOUT)))
