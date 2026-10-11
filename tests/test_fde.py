from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from pig.fde import LLaDAFDE, NemotronFDE, load_fde  # noqa: E402

ASSETS = Path(__file__).resolve().parent.parent / "assets"


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def test_parameter_counts():
    assert n_params(LLaDAFDE()) == 2_165_527
    assert n_params(NemotronFDE()) == 1_903_383


def test_llada_fde_shapes():
    net = load_fde(LLaDAFDE, ASSETS / "llada" / "fde.safetensors", device="cpu")
    L = 256
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        D = net(torch.randn(L, 4096, generator=g), torch.randn(L, 4096, generator=g),
                torch.randn(L, 9, generator=g), torch.rand(L, L, generator=g),
                torch.rand(L, L, generator=g), torch.tensor(0.5))
    assert D.shape == (L, L) and torch.isfinite(D).all()


@pytest.mark.parametrize("T", [16, 45, 128])
def test_nemotron_fde_shapes(T):
    net = load_fde(NemotronFDE, ASSETS / "nemotron" / "fde.safetensors", device="cpu")
    g = torch.Generator().manual_seed(T)
    with torch.no_grad():
        D = net(torch.randn(T, 3072, generator=g), torch.randn(T, 3072, generator=g),
                torch.randn(T, 10, generator=g), torch.tensor(0.25))
    assert D.shape == (T, T) and torch.isfinite(D).all()
