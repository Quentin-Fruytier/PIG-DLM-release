"""FDE for Nemotron-Labs-Diffusion-3B (about 1.9M parameters).

The model is block-causal and decodes inside a window, so the estimator runs over the T positions
of the current window (T varies with the window span); rotary positions are window-relative and
the window size enters as a scalar.

Inputs per window position:
    h      (T, 3072)  last hidden state of the current window forward (the diffusion-head input)
    e      (T, 3072)  input embedding of the current window token (the mask token where masked)
    scal   (T, 10)    entropy, confidence, top-1/top-2 margin, decoded flag,
                      one-hot class (decoded / masked / window head), window fill fraction,
                      log2(T)/7, relative position i/T
    t      ()         window fill fraction
Output: D_hat (T, T), row = receiver, column = source.
"""
import math

import torch
import torch.nn as nn

from .common import Block, D_MODEL, N_HEADS, READOUT, readout_maps

HIDDEN = 3072
N_SCALARS = 10


class NemotronFDE(nn.Module):
    def __init__(self):
        super().__init__()
        self.ph = nn.Linear(HIDDEN, D_MODEL)
        self.pe = nn.Linear(HIDDEN, 64)
        self.ps = nn.Linear(N_SCALARS, 32)
        self.fuse = nn.Sequential(nn.Linear(D_MODEL + 64 + 32, D_MODEL), nn.LayerNorm(D_MODEL))
        self.blocks = nn.ModuleList([Block(), Block()])
        # pairwise features (log(1+|i-j|), i/T, j/T) -> per-head bias + edge bias
        self.bias_mlp = nn.Sequential(nn.Linear(3, 16), nn.GELU(), nn.Linear(16, N_HEADS + 1))
        self.f, self.g = readout_maps()
        self.W = nn.Parameter(torch.randn(READOUT, READOUT) * 0.02)

    def forward(self, h, e, scal, t):
        T = h.shape[0]
        idx = torch.arange(T, device=h.device, dtype=torch.float32)
        dist = torch.log1p((idx[:, None] - idx[None, :]).abs())
        rel = idx / T
        bf = torch.stack([dist, rel[:, None].expand(T, T), rel[None, :].expand(T, T)], -1)
        bias_all = self.bias_mlp(bf)
        att_bias = bias_all[..., :N_HEADS].permute(2, 0, 1)
        edge_bias = bias_all[..., N_HEADS]
        x = self.fuse(torch.cat([self.ph(h), self.pe(e), self.ps(scal)], -1))
        for blk in self.blocks:
            x = blk(x, att_bias, t)
        fz, gz = self.f(x), self.g(x)
        return (fz @ self.W @ gz.T) / math.sqrt(READOUT) + edge_bias
