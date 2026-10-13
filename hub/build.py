"""Build the Hugging Face Hub repositories from this package.

    python -m hub.build [--out hub/dist] [--family llada|nemotron]

For each backbone this writes hub/dist/<repo name>/ holding everything the Hub repository needs:

    config.json             PIGConfig: backbone id and revision, decoding defaults, adapter provenance
    model.safetensors       the adapter, re-keyed with the wrapper's "fde." prefix (same tensors as
                            assets/<model>/fde.safetensors)
    flip_confidence.npz     the two calibration tables, copied from assets/<model>/
    flip_entropy.npz
    configuration_pig.py    the wrapper (hub/configuration_pig.py, hub/modeling_pig.py)
    modeling_pig.py
    rules.py, tables.py,    vendored verbatim from pig/ with a provenance header (api.py is the
    api.py                  generate() interface the wrapper exposes)
    fde.py                  pig/fde/common.py + pig/fde/<family>.py + load_fde, flattened
    decoder.py              pig/<family>.py plus the aliases the wrapper imports
    README.md               the model card, hub/cards/<repo name>.md

The package is the single source of truth: nothing in dist/ is edited by hand, and
tests/test_hub.py rebuilds it to check that the wrapper loads the same adapter the harness uses.
"""
import argparse
import ast
import json
import shutil
import struct
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HUB = ROOT / "hub"
ASSETS = ROOT / "assets"
DIST = HUB / "dist"

REPOS = {
    "llada": dict(name="LLaDA-8B-Instruct-PIG", repo="UT-IFML/LLaDA-8B-Instruct-PIG", fde_class="LLaDAFDE",
                  decoder_class="LLaDADecoder", geometry_class="Geometry"),
    "nemotron": dict(name="Nemotron-Labs-Diffusion-3B-PIG", repo="UT-IFML/Nemotron-Labs-Diffusion-3B-PIG",
                     fde_class="NemotronFDE", decoder_class="NemotronDecoder", geometry_class="Window"),
}


def git_stamp():
    try:
        rev = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "pig"], cwd=ROOT).returncode != 0
        return rev + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def header(src_rel, stamp):
    return (f"# Vendored from https://github.com/Quentin-Fruytier/PIG-DLM-release ({stamp}): {src_rel}.\n"
            f"# Do not edit here; edit the package and rebuild with `python -m hub.build`.\n")


def vendor(src_rel, stamp, drop_lines=(), append=""):
    text = (ROOT / src_rel).read_text()
    lines = [ln for ln in text.splitlines(keepends=True) if ln.rstrip("\n") not in drop_lines]
    return header(src_rel, stamp) + "".join(lines) + append


def load_fde_source():
    """The load_fde function of pig/fde/__init__.py, verbatim."""
    src = (ROOT / "pig/fde/__init__.py").read_text()
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "load_fde":
            return ast.get_source_segment(src, node) + "\n"
    raise RuntimeError("load_fde not found in pig/fde/__init__.py")


def safetensors_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def build(family, out=DIST):
    import numpy as np
    from safetensors.torch import load_file, save_file

    from hub.configuration_pig import PIGConfig

    spec = REPOS[family]
    stamp = git_stamp()
    dst = Path(out) / spec["name"]
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    # --- wrapper + vendored modules
    for f in ("configuration_pig.py", "modeling_pig.py"):
        shutil.copyfile(HUB / f, dst / f)
    (dst / "rules.py").write_text(vendor("pig/rules.py", stamp))
    (dst / "tables.py").write_text(vendor("pig/tables.py", stamp))
    (dst / "api.py").write_text(vendor("pig/api.py", stamp))
    fde = (header("pig/fde/common.py + pig/fde/%s.py + pig/fde/__init__.py:load_fde" % family, stamp)
           + (ROOT / "pig/fde/common.py").read_text() + "\n\n"
           + "".join(ln for ln in (ROOT / f"pig/fde/{family}.py").read_text().splitlines(keepends=True)
                     if not ln.startswith("from .common import"))
           + "\n\n" + load_fde_source()
           + f"\n\nPIGFDE = {spec['fde_class']}\n")
    (dst / "fde.py").write_text(fde)
    (dst / "decoder.py").write_text(vendor(
        f"pig/{family}.py", stamp,
        append=(f"\n\nFAMILY = {family!r}\nPIGDecoder = {spec['decoder_class']}\n"
                f"PIGGeometry = {spec['geometry_class']}\n")))

    # --- adapter: same tensors, keys prefixed "fde." so they load into PIGModel.fde
    src = ASSETS / family / "fde.safetensors"
    state = load_file(str(src), device="cpu")
    meta = dict(safetensors_header(src).get("__metadata__", {}))
    meta.update({"format": "pt", "source": f"assets/{family}/fde.safetensors", "package": stamp})
    save_file({f"fde.{k}": v.contiguous() for k, v in state.items()}, str(dst / "model.safetensors"), metadata=meta)
    n_params = int(sum(v.numel() for v in state.values()))
    for t in ("flip_confidence.npz", "flip_entropy.npz"):
        shutil.copyfile(ASSETS / family / t, dst / t)

    # --- config
    if family == "llada":
        from pig.llada import MODEL_ID, REVISION
    else:
        from pig.nemotron import MODEL_ID, REVISION
    adapter = {"parameters": n_params, "source": f"assets/{family}/fde.safetensors", "package": stamp}
    for k in ("epoch", "val_edge_corr", "description"):
        if k in meta:
            adapter[k] = meta[k]
    tables = {}
    for axis, t in (("confidence", "flip_confidence.npz"), ("entropy", "flip_entropy.npz")):
        z = np.load(ASSETS / family / t, allow_pickle=False)
        tables[axis] = {"file": t, "rows": int(z["rows"].sum()), "mismatches": int(z["flips"].sum())}
    cfg = PIGConfig(family=family, backbone=MODEL_ID, backbone_revision=REVISION, adapter=adapter,
                    tables={"confidence": "flip_confidence.npz", "entropy": "flip_entropy.npz"},
                    architectures=["PIGModel"],
                    auto_map={"AutoConfig": "configuration_pig.PIGConfig", "AutoModel": "modeling_pig.PIGModel"})
    cfg.table_provenance = tables
    cfg.save_pretrained(dst)

    # --- card
    card = HUB / "cards" / f"{spec['name']}.md"
    if card.exists():
        shutil.copyfile(card, dst / "README.md")
    return dst


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(DIST))
    ap.add_argument("--family", choices=sorted(REPOS), action="append")
    a = ap.parse_args()
    for fam in a.family or sorted(REPOS):
        d = build(fam, a.out)
        print(f"[{fam}] -> {d}: " + ", ".join(sorted(p.name for p in d.iterdir())))


if __name__ == "__main__":
    main()
