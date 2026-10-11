"""The two PIG selection rules (paper Section 3; Algorithms 2 and 3).

Both rules run once per decoding round, after the single backbone forward pass. They walk the
eligible masked positions in decreasing confidence and return the positions to commit in parallel,
always a prefix of that order. They read the same one-pass quantities:

    conf[i]   confidence nu_i of the argmax prediction at position i
    H[i]      predictive entropy H_i at position i (nats)
    D[i, j]   the adapter's entropy finite difference D_hat_ij: the predicted drop of H_i when
              position j is committed to its argmax (row = receiver i, column = source j)

and an aggregate influence of the tokens decoded before a candidate,

    mu_hat_i(S) = sum_{j in S} max(|D_hat_ij|, |D_hat_ji|),

summed only over members that precede i in the confidence order.

PIG.INT(phi)                       single-round regret budget
    The most confident token is admitted with flip rate 0 (its sequential reference is its own
    prediction). Every later candidate i is priced b_i = p_nu(nu_i, mu_hat_i(S)) from the
    confidence table and admitted while the summed price stays within phi; the first candidate
    that does not fit closes the round.

PIG.EXT(theta, h, floor)           cross-round regret budget
    Admit first, then decide whether to keep going: after each admission the next h candidates
    N(S) are priced now and after waiting one round,
        C_hat(S) = sum_{i in N(S)} [ p_H(H_i, mu_i) - p_H(H_i - max_{j in S} D_hat_ij, mu_i) ],
        mu_i = mu_hat_i(S  and the members of N(S) more confident than i),
    and the round closes once C_hat(S) > theta; the token just admitted stays. An optional
    confidence floor stops admission at the first candidate below it (the first token of a round
    is always admitted).
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class Round:
    """What a selection rule sees in one decoding round. Positions index the decoded region
    (the canvas for LLaDA, the current window for Nemotron)."""
    cand: list            # eligible masked positions, ascending
    conf: np.ndarray      # confidence of the argmax at every position
    H: np.ndarray         # predictive entropy at every position (nats)
    D: np.ndarray         # adapter output D_hat, row = receiver, column = source


def confidence_order(cand, conf):
    """Candidates in decreasing confidence; ties keep the given (position) order."""
    cand = np.asarray(cand, dtype=np.int64)
    if cand.size == 0:
        return []
    return [int(c) for c in cand[np.argsort(-np.asarray(conf)[cand], kind="stable")]]


def pair_influence(D, i, js):
    """max(|D_hat_ij|, |D_hat_ji|) for every j in js, as float64."""
    js = np.asarray(js, dtype=np.int64)
    return np.maximum(np.abs(D[i, js].astype(np.float64)), np.abs(D[js, i].astype(np.float64)))


def aggregate_influence(D, i, prefix):
    """mu_hat_i over `prefix`, accumulated left to right in prefix order."""
    if len(prefix) == 0:
        return 0.0
    return float(np.cumsum(pair_influence(D, i, prefix))[-1])


def _column_influence(D, idx, j):
    """max(|D_hat_kj|, |D_hat_jk|) for every k in idx (vectorised over receivers)."""
    return np.maximum(np.abs(D[idx, j].astype(np.float64)), np.abs(D[j, idx].astype(np.float64)))


def pig_int(cand, conf, D, table, phi):
    """PIG.INT (Algorithm 2): commit the longest confidence prefix whose predicted
    single-round regret stays within phi."""
    order = confidence_order(cand, conf)
    if not order:
        return []
    idx = np.asarray(order, dtype=np.int64)
    # mu[r] = mu_hat of order[r] with respect to the admitted set, summed in admission order
    mu = _column_influence(D, idx, order[0])
    S, spent = [order[0]], 0.0
    for r in range(1, len(order)):
        c = order[r]
        b = table(float(conf[c]), float(mu[r]))
        if spent + b > phi:
            break
        S.append(c)
        spent += b
        mu += _column_influence(D, idx, c)
    return S


def waiting_gain(S, window, H, D, table):
    """C_hat(S): predicted regret on `window` removed by committing S first and re-evaluating.

    Reference form, one candidate set at a time; pig_ext computes the same quantity
    incrementally.
    """
    gain = 0.0
    for q, k in enumerate(window):
        relief = float(np.max(D[k, np.asarray(S, dtype=np.int64)]))
        mu = aggregate_influence(D, k, list(S) + list(window[:q]))
        h = float(H[k])
        gain += table(h, mu) - table(h - relief, mu)
    return gain


def pig_ext(cand, conf, H, D, table, theta, window=16, floor=None):
    """PIG.EXT (Algorithm 3): admit in decreasing confidence and close the round once the
    predicted cross-round regret of the next `window` candidates exceeds theta."""
    order = confidence_order(cand, conf)
    if not order:
        return []
    n = len(order)
    idx = np.asarray(order, dtype=np.int64)
    sub = D[np.ix_(idx, idx)].astype(np.float64)
    pair = np.maximum(np.abs(sub), np.abs(sub.T))    # pair[a, b] = max(|D_hat|) between order[a], order[b]
    h_all = np.asarray(H, dtype=np.float64)[idx]
    mu_S = np.zeros(n, dtype=np.float64)             # mu_hat of every candidate w.r.t. S
    relief = np.full(n, -np.inf, dtype=np.float64)   # max_{j in S} D_hat[candidate, j]
    S = []
    for r in range(n):
        c = order[r]
        if S and floor is not None and float(conf[c]) < floor:
            break                                     # the floor stops admission
        S.append(c)
        mu_S += pair[:, r]
        relief = np.maximum(relief, sub[:, r])
        w = np.arange(r + 1, min(n, r + 1 + window))
        if len(w) == 0:
            continue
        # mu_i = mu_hat_i(S) plus the more confident window members, summed in order
        mu = np.cumsum(np.concatenate([mu_S[w][:, None], np.tril(pair[np.ix_(w, w)], -1)], axis=1),
                       axis=1)[:, -1]
        h = h_all[w]
        gain = np.cumsum(table.lookup(h, mu) - table.lookup(h - relief[w], mu))[-1]
        if gain > theta:
            break
    return S


class PIGInt:
    """PIG.INT as a per-round rule: PIGInt(confidence_table, phi)(round) -> positions."""

    def __init__(self, table, phi):
        if table.axis != "confidence":
            raise ValueError("PIG.INT reads the confidence table")
        self.table, self.phi = table, float(phi)

    def __call__(self, rnd):
        return pig_int(rnd.cand, rnd.conf, rnd.D, self.table, self.phi)


class PIGExt:
    """PIG.EXT as a per-round rule: PIGExt(entropy_table, theta, window, floor)(round) -> positions."""

    def __init__(self, table, theta, window=16, floor=None):
        if table.axis != "entropy":
            raise ValueError("PIG.EXT reads the entropy table")
        self.table, self.theta, self.window = table, float(theta), int(window)
        self.floor = None if floor is None else float(floor)

    def __call__(self, rnd):
        return pig_ext(rnd.cand, rnd.conf, rnd.H, rnd.D, self.table, self.theta,
                       window=self.window, floor=self.floor)
