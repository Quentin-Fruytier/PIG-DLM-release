"""Entropy Finite Differences Estimator (FDE): one adapter per backbone."""
import torch

from .llada import LLaDAFDE
from .nemotron import NemotronFDE


def load_fde(cls, path, device="cuda"):
    """Build `cls` and load weights from a .safetensors file; returns the module in eval mode."""
    from safetensors.torch import load_file

    net = cls()
    net.load_state_dict(load_file(str(path), device="cpu"))
    return net.to(device).eval()


__all__ = ["LLaDAFDE", "NemotronFDE", "load_fde"]
