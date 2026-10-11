"""The selection rules against direct transcriptions of the paper's Algorithms 2 and 3."""
import numpy as np
import pytest

from pig.rules import (Round, PIGExt, PIGInt, aggregate_influence, confidence_order, pig_ext,
                       pig_int, waiting_gain)
from pig.tables import FlipTable


def conf_table(rng):
    rates = np.sort(rng.uniform(0, 0.8, size=(8, 5)), axis=0)[::-1]
    return FlipTable([0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 1.001],
                     [0.0, 0.25, 0.9, 2.4, 6.0, 36.0], rates, "confidence")


def entropy_table(rng):
    return FlipTable([0.0, 0.1, 0.25, 0.5, 1.0, 1.75, 2.5, 3.5, 4.5, 6.0, 99.0],
                     [0.0, 0.26, 0.89, 2.45, 6.03, 36.0], rng.uniform(0, 0.8, size=(10, 5)), "entropy")


def random_round(rng, L=48, n_cand=None):
    conf = rng.uniform(0, 1, L).astype(np.float32)
    conf[rng.integers(0, L, 5)] = 1.0                       # ties at saturation, as in practice
    H = rng.uniform(0, 5, L).astype(np.float32)
    D = rng.normal(0, 0.3, (L, L)).astype(np.float32)
    n_cand = n_cand or int(rng.integers(1, L))
    cand = sorted(rng.choice(L, n_cand, replace=False).tolist())
    return cand, conf, H, D


def reference_int(cand, conf, D, table, phi):
    """Algorithm 2: the first token is admitted at flip rate 0; each later token pays
    p_nu(nu, mu_hat(S)) and is admitted only if the total stays within phi."""
    order = sorted(cand, key=lambda c: -float(conf[c]))
    S, spent = [order[0]], 0.0
    for i in order[1:]:
        b = table(float(conf[i]), aggregate_influence(D, i, S))
        if spent + b > phi:
            break
        S.append(i)
        spent += b
    return S


def reference_ext(cand, conf, H, D, table, theta, window=16, floor=None):
    """Algorithm 3: admit first, then close the round once C_hat(S) > theta."""
    order = sorted(cand, key=lambda c: -float(conf[c]))
    S = []
    for idx, j in enumerate(order):
        if S and floor is not None and float(conf[j]) < floor:
            break
        S.append(j)
        if waiting_gain(S, order[idx + 1: idx + 1 + window], H, D, table) > theta:
            break
    return S


def test_confidence_order_is_stable():
    conf = np.array([0.5, 1.0, 0.2, 1.0, 1.0], np.float32)
    assert confidence_order([0, 1, 2, 3, 4], conf) == [1, 3, 4, 0, 2]


@pytest.mark.parametrize("seed", range(300))
def test_pig_int_matches_algorithm_2(seed):
    rng = np.random.default_rng(seed)
    cand, conf, H, D = random_round(rng)
    table = conf_table(rng)
    for phi in (0.0, 0.015, 0.05, 0.2, 1.0, 3.0):
        assert pig_int(cand, conf, D, table, phi) == reference_int(cand, conf, D, table, phi)


@pytest.mark.parametrize("seed", range(300))
def test_pig_ext_matches_algorithm_3(seed):
    rng = np.random.default_rng(seed)
    cand, conf, H, D = random_round(rng)
    table = entropy_table(rng)
    for theta, floor in ((0.0, None), (0.125, 0.9), (0.25, 0.8), (0.5, None), (1.0, None), (4.0, None)):
        assert (pig_ext(cand, conf, H, D, table, theta, floor=floor)
                == reference_ext(cand, conf, H, D, table, theta, floor=floor))


def test_pig_int_first_token_is_free():
    rng = np.random.default_rng(0)
    cand, conf, H, D = random_round(rng, n_cand=20)
    table = FlipTable([0.0, 1.001], [0.0, 1e9], [[0.5]], "confidence")
    top = confidence_order(cand, conf)[0]
    assert pig_int(cand, conf, D, table, phi=0.0) == [top]          # admitted although p = 0.5
    assert pig_int(cand, conf, D, table, phi=0.99) == confidence_order(cand, conf)[:2]
    zero = FlipTable([0.0, 1.001], [0.0, 1e9], [[0.0]], "confidence")
    assert pig_int(cand, conf, D, zero, phi=0.0) == confidence_order(cand, conf)


def test_pig_ext_can_close_after_the_first_token():
    conf = np.array([0.9, 0.8, 0.7, 0.6], np.float32)
    H = np.full(4, 3.0, np.float32)
    D = np.zeros((4, 4), np.float32)
    D[1:, 0] = 2.9                                  # committing position 0 settles the others
    table = FlipTable([0.0, 0.5, 99.0], [0.0, 1e9], [[0.0], [0.4]], "entropy")
    # waiting on position 0 removes 0.4 regret on each of the 3 upcoming positions
    assert pig_ext([0, 1, 2, 3], conf, H, D, table, theta=1.0) == [0]
    assert pig_ext([0, 1, 2, 3], conf, H, D, table, theta=1.3) == [0, 1, 2, 3]


def test_pig_ext_without_influence_admits_everything():
    rng = np.random.default_rng(1)
    cand, conf, H, _ = random_round(rng, n_cand=30)
    D = np.zeros((48, 48), np.float32)
    assert pig_ext(cand, conf, H, D, entropy_table(rng), theta=0.0) == confidence_order(cand, conf)


def test_pig_ext_floor_stops_admission_but_keeps_the_first_token():
    conf = np.array([0.95, 0.85, 0.7, 0.99], np.float32)
    H = np.ones(4, np.float32)
    D = np.zeros((4, 4), np.float32)
    table = FlipTable([0.0, 99.0], [0.0, 1e9], [[0.1]], "entropy")
    assert pig_ext([0, 1, 2, 3], conf, H, D, table, theta=10.0, floor=0.9) == [3, 0]
    assert pig_ext([1, 2], conf, H, D, table, theta=10.0, floor=0.9) == [1]


def test_pig_ext_window_limits_the_look_ahead():
    L = 30
    conf = np.linspace(0.99, 0.5, L).astype(np.float32)
    H = np.full(L, 3.0, np.float32)
    D = np.zeros((L, L), np.float32)
    D[20:, 0] = 2.9                                  # only positions beyond the first window react
    table = FlipTable([0.0, 0.5, 99.0], [0.0, 1e9], [[0.0], [0.4]], "entropy")
    assert pig_ext(list(range(L)), conf, H, D, table, theta=0.1, window=16) == list(range(5))


def test_rule_objects_check_their_table():
    rng = np.random.default_rng(2)
    with pytest.raises(ValueError):
        PIGInt(entropy_table(rng), 0.1)
    with pytest.raises(ValueError):
        PIGExt(conf_table(rng), 0.5)
    cand, conf, H, D = random_round(rng)
    rnd = Round(cand=cand, conf=conf, H=H, D=D)
    t = conf_table(rng)
    assert PIGInt(t, 0.1)(rnd) == pig_int(cand, conf, D, t, 0.1)
