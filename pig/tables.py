"""Mismatch-rate lookup tables: the calibration tables p_hat of the paper.

A table is a 2-D grid of empirical rates p_hat(x, mu): the probability that a token committed in
parallel differs from its sequential (one token per forward) reference, binned by a per-token
uncertainty x and by the aggregate influence mu_hat of the more confident tokens decoded before it.
Two tables ship per backbone:

    confidence table   x = confidence nu = max_v q(v | x)       read by PIG.INT
    entropy table      x = predictive entropy H (nats)           read by PIG.EXT

Lookup is a plain bin read. Bins are half-open [edge_k, edge_{k+1}); values below the first edge
fall in the first bin and values above the last edge in the last bin, and mu is capped just below
its last edge.
"""
import numpy as np


class FlipTable:
    def __init__(self, x_edges, mu_edges, rates, axis, rows=None, flips=None):
        self.x_edges = np.asarray(x_edges, dtype=np.float64)
        self.mu_edges = np.asarray(mu_edges, dtype=np.float64)
        self.rates = np.asarray(rates, dtype=np.float64)
        self.axis = str(axis)
        self.rows = None if rows is None else np.asarray(rows)
        self.flips = None if flips is None else np.asarray(flips)
        if self.rates.shape != (len(self.x_edges) - 1, len(self.mu_edges) - 1):
            raise ValueError(f"rates shape {self.rates.shape} does not match the bin edges "
                             f"({len(self.x_edges) - 1} x {len(self.mu_edges) - 1})")
        if self.axis not in ("confidence", "entropy"):
            raise ValueError(f"unknown table axis {self.axis!r}")
        self._mu_cap = float(self.mu_edges[-1]) - 1e-6

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        return cls(z["x_edges"], z["mu_edges"], z["rates"], str(z["axis"]),
                   rows=z["rows"] if "rows" in z.files else None,
                   flips=z["flips"] if "flips" in z.files else None)

    def __call__(self, x, mu):
        """p_hat(x, mu) as a Python float."""
        i = int(np.searchsorted(self.x_edges, float(x), "right")) - 1
        j = int(np.searchsorted(self.mu_edges, min(float(mu), self._mu_cap), "right")) - 1
        i = min(max(i, 0), self.rates.shape[0] - 1)
        j = min(max(j, 0), self.rates.shape[1] - 1)
        return float(self.rates[i, j])

    def lookup(self, x, mu):
        """Vectorised p_hat over arrays x and mu (same bins as __call__)."""
        x = np.asarray(x, dtype=np.float64)
        mu = np.minimum(np.asarray(mu, dtype=np.float64), self._mu_cap)
        i = np.clip(np.searchsorted(self.x_edges, x, "right") - 1, 0, self.rates.shape[0] - 1)
        j = np.clip(np.searchsorted(self.mu_edges, mu, "right") - 1, 0, self.rates.shape[1] - 1)
        return self.rates[i, j]

    def __repr__(self):
        return (f"FlipTable(axis={self.axis}, x_bins={self.rates.shape[0]}, "
                f"mu_bins={self.rates.shape[1]})")
