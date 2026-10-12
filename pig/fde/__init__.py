"""Entropy Finite Differences Estimator (FDE): one adapter per backbone."""
import torch

from .llada import LLaDAFDE
from .nemotron import NemotronFDE


def load_fde(cls, path, device="cuda"):
    """Build `cls` and load weights from a .safetensors file; returns the module in eval mode.

    `path` may also be an already-built module (the Hub wrapper passes its own), which is moved to
    `device`. Checkpoint keys may carry the Hub wrapper's "fde." prefix (its model.safetensors)."""
    if isinstance(path, torch.nn.Module):
        return path.to(device).eval()
    from safetensors.torch import load_file

    state = load_file(str(path), device="cpu")
    if all(k.startswith("fde.") for k in state):
        state = {k[len("fde."):]: v for k, v in state.items()}
    net = cls()
    net.load_state_dict(state)
    return net.to(device).eval()


__all__ = ["LLaDAFDE", "NemotronFDE", "load_fde"]
