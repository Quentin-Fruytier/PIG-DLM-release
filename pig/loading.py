"""Load a backbone, its PIG adapter and its calibration tables from assets/ (or the Hugging Face Hub)."""
from .api import FAMILIES, PIG
from .assets import ASSETS, asset_paths
from .tables import FlipTable


def load_tables(family, assets=ASSETS):
    paths = asset_paths(family, assets)
    return {"confidence": FlipTable.load(paths["confidence"]), "entropy": FlipTable.load(paths["entropy"])}


def load_decoder(family, assets=ASSETS, device=None, **kwargs):
    """The decoding backend with the adapter attached: downloads the backbone at its pinned revision."""
    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}; expected one of {FAMILIES}")
    if device is None:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    fde = asset_paths(family, assets)["fde"]
    if family == "llada":
        from .llada import LLaDADecoder
        return LLaDADecoder(fde, device=device, **kwargs)
    from .nemotron import NemotronDecoder
    return NemotronDecoder(fde, device=device, **kwargs)


def make_geometry(family, kind, size):
    if family == "llada":
        from .llada import Geometry
        return Geometry(kind, size)
    from .nemotron import Window
    return Window(kind, size)


def load(family, assets=ASSETS, device=None, **kwargs):
    """pig.load("llada" | "nemotron") -> a PIG whose generate() runs PIG.INT, PIG.EXT or the baseline."""
    return PIG(load_decoder(family, assets, device, **kwargs), load_tables(family, assets))
