from pathlib import Path

import numpy as np
import pytest

from pig.tables import FlipTable

ASSETS = Path(__file__).resolve().parent.parent / "assets"


def small():
    return FlipTable([0.0, 0.5, 1.0], [0.0, 1.0, 2.0], [[0.1, 0.2], [0.3, 0.4]], "confidence")


def test_bins_are_half_open():
    t = small()
    assert t(0.0, 0.0) == 0.1
    assert t(0.4999, 0.9999) == 0.1
    assert t(0.5, 0.0) == 0.3
    assert t(0.5, 1.0) == 0.4


def test_out_of_range_values_clip_to_the_edge_bins():
    t = small()
    assert t(-3.0, 0.5) == 0.1            # below the first edge (e.g. a negative relieved entropy)
    assert t(7.0, 0.5) == 0.3             # above the last edge
    assert t(0.2, 1e9) == 0.2             # mu is capped just below its last edge
    assert t(0.2, 2.0) == 0.2


def test_shape_and_axis_are_validated():
    with pytest.raises(ValueError):
        FlipTable([0.0, 1.0], [0.0, 1.0], [[0.1, 0.2]], "confidence")
    with pytest.raises(ValueError):
        FlipTable([0.0, 1.0], [0.0, 1.0], [[0.1]], "margin")


@pytest.mark.parametrize("model", ["llada", "nemotron"])
@pytest.mark.parametrize("name,axis", [("flip_confidence", "confidence"), ("flip_entropy", "entropy")])
def test_shipped_tables_load(model, name, axis):
    t = FlipTable.load(ASSETS / model / f"{name}.npz")
    assert t.axis == axis
    assert np.all((t.rates >= 0) & (t.rates <= 1))
    assert np.all(np.diff(t.x_edges) > 0) and np.all(np.diff(t.mu_edges) > 0)
    assert t.rows.sum() > 100_000
