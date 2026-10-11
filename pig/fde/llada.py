"""FDE for LLaDA-8B-Instruct (about 2.2M parameters).

Inputs per canvas position (L = 256 generated positions):
    h      (L, 4096)  post-final-norm hidden state (the LM-head input) of the current pass
    e      (L, 4096)  input embedding of the current canvas token (the mask token where masked)
    scal   (L, 9)     entropy, confidence, top-1/top-2 margin, decoded flag,
                      one-hot class (decoded / committable / other / canvas boundary), decode fraction t
    A_t    (L, L)     final-layer attention of the current pass, head-averaged, symmetrised
    A_prev (L, L)     the same map from the previous round (equal to A_t on the first round)
    t      ()         decode fraction
Output: D_hat (L, L), row = receiver, column = source.
"""
import math

import torch
import torch.nn as nn

from .common import Block, D_MODEL, N_HEADS, READOUT, readout_maps

HIDDEN = 4096
N_SCALARS = 9


class LLaDAFDE(nn.Module):
    def __init__(self):
        super().__init__()
        self.ph = nn.Linear(HIDDEN, D_MODEL)
        self.pe = nn.Linear(HIDDEN, 64)
        self.ps = nn.Linear(N_SCALARS, 32)
        self.fuse = nn.Sequential(nn.Linear(D_MODEL + 64 + 32, D_MODEL), nn.LayerNorm(D_MODEL))
        self.blocks = nn.ModuleList([Block(), Block()])
        # pairwise features (A_t[i,j], A_t[j,i], A_prev[i,j], A_prev[j,i], log(1+|i-j|))
        # -> one bias per attention head + one additive edge bias on the readout
        self.bias_mlp = nn.Sequential(nn.Linear(5, 16), nn.GELU(), nn.Linear(16, N_HEADS + 1))
        self.f, self.g = readout_maps()
        self.W = nn.Parameter(torch.randn(READOUT, READOUT) * 0.02)

    def forward(self, h, e, scal, At, Ap, t):
        T = h.shape[0]
        idx = torch.arange(T, device=h.device, dtype=torch.float32)
        dist = torch.log1p((idx[:, None] - idx[None, :]).abs())
        bias_all = self.bias_mlp(torch.stack([At, At.T, Ap, Ap.T, dist], -1))
        att_bias = bias_all[..., :N_HEADS].permute(2, 0, 1)
        edge_bias = bias_all[..., N_HEADS]
        x = self.fuse(torch.cat([self.ph(h), self.pe(e), self.ps(scal)], -1))
        for blk in self.blocks:
            x = blk(x, att_bias, t)
        fz, gz = self.f(x), self.g(x)
        return (fz @ self.W @ gz.T) / math.sqrt(READOUT) + edge_bias
