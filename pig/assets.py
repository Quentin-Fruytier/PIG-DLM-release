"""Where the adapter and the calibration tables come from.

Each backbone has three files: the Entropy Finite Differences Estimator (fde.safetensors) and the two
calibration tables (flip_confidence.npz for PIG.INT, flip_entropy.npz for PIG.EXT). They ship in this
repository under assets/<model>/ and on the Hugging Face Hub (HUB_REPOS), where the adapter is stored as
model.safetensors (the transformers wrapper's checkpoint; same tensors, keys prefixed "fde."). resolve()
returns the local directory when it holds the files and otherwise downloads them from the Hub into the
huggingface_hub cache, so a checkout without assets/ still runs.
"""
from pathlib import Path

ASSETS = Path(__file__).resolve().parent.parent / "assets"
FILES = ("fde.safetensors", "flip_confidence.npz", "flip_entropy.npz")
HUB_REPOS = {
    "llada": "UT-IFML/LLaDA-8B-Instruct-PIG",
    "nemotron": "UT-IFML/Nemotron-Labs-Diffusion-3B-PIG",
}


def resolve(model, assets=ASSETS):
    """Directory holding the three asset files of `model`: local if complete, else the Hub mirror."""
    if model not in HUB_REPOS:
        raise ValueError(f"unknown model {model!r}; expected one of {sorted(HUB_REPOS)}")
    d = Path(assets) / model
    if all((d / f).exists() for f in FILES[1:]) and ((d / FILES[0]).exists() or (d / "model.safetensors").exists()):
        return d
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(HUB_REPOS[model], allow_patterns=list(FILES) + ["model.safetensors"]))


def asset_paths(model, assets=ASSETS):
    d = resolve(model, assets)
    fde = d / FILES[0]
    if not fde.exists() and (d / "model.safetensors").exists():
        fde = d / "model.safetensors"      # the Hub mirror's name for the adapter (keys prefixed "fde.")
    return {"fde": fde, "confidence": d / FILES[1], "entropy": d / FILES[2]}
