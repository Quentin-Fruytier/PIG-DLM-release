"""Build the Hugging Face Hub repositories from this package.

    python -m hub.build [--out hub/dist] [--family llada|nemotron]

For each backbone this writes hub/dist/<repo name>/ holding everything the Hub repository needs:

    config.json             PIGConfig: backbone id and revision, decoding defaults, adapter and table provenance
    model.safetensors       the adapter (keys fde.*) and the two calibration tables (keys tables.<axis>.*),
                            the same tensors as assets/<model>/
    configuration_pig.py    the config class (hub/configuration_pig.py)
    modeling_pig.py         ONE file: the selection rules, table lookup, adapter, decode loop and generate()
                            interface of pig/, concatenated verbatim section by section, then the wrapper
                            class of hub/modeling_pig.py
    README.md               the model card, hub/cards/<repo name>.md
    AGENTS.md               the machine-oriented reference, hub/cards/<repo name>.AGENTS.md

The package is the single source of truth: nothing in dist/ is edited by hand, and tests/test_hub.py
rebuilds it to check that every section of modeling_pig.py is the package's code.
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

# the sections of modeling_pig.py, in dependency order: (title, source files relative to the repo root)
def sections(family):
    return [
        ("pig/rules.py", ["pig/rules.py"]),
        ("pig/tables.py", ["pig/tables.py"]),
        (f"pig/fde/common.py + pig/fde/{family}.py + pig/fde/__init__.py:load_fde",
         ["pig/fde/common.py", f"pig/fde/{family}.py", "pig/fde/__init__.py:load_fde"]),
        (f"pig/{family}.py", [f"pig/{family}.py"]),
        ("pig/api.py", ["pig/api.py"]),
        ("hub/modeling_pig.py", ["hub/modeling_pig.py"]),
    ]


def git_stamp():
    try:
        rev = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "pig", "hub"], cwd=ROOT).returncode != 0
        return rev + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def split_module(src):
    """-> (docstring or None, absolute import statements, body lines). Relative imports are dropped:
    in the single-file build every name is in one namespace."""
    tree = ast.parse(src)
    lines = src.splitlines(keepends=True)
    drop = set()
    doc = None
    imports = []
    body_nodes = tree.body
    if body_nodes and isinstance(body_nodes[0], ast.Expr) and isinstance(body_nodes[0].value, ast.Constant) \
            and isinstance(body_nodes[0].value.value, str):
        doc = body_nodes[0].value.value
        drop.update(range(body_nodes[0].lineno, body_nodes[0].end_lineno + 1))
    for node in body_nodes:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            drop.update(range(node.lineno, node.end_lineno + 1))
            if isinstance(node, ast.Import) or node.level == 0:
                imports.append("".join(lines[node.lineno - 1:node.end_lineno]).rstrip("\n"))
    body = [ln for i, ln in enumerate(lines, 1) if i not in drop]
    return doc, imports, "".join(body).strip("\n") + "\n"


def function_source(path, name):
    """One top-level function of a module, verbatim."""
    src = path.read_text()
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(src, node) + "\n"
    raise RuntimeError(f"{name} not found in {path}")


def top_level_names(src):
    names = []
    for node in ast.parse(src).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names.append(node.name)
        elif isinstance(node, ast.Assign):
            names += [t.id for t in node.targets if isinstance(t, ast.Name)]
    return names


def comment_block(text):
    return "".join(("# " + ln if ln.strip() else "#") + "\n" for ln in text.strip("\n").splitlines())


def modeling_source(family, stamp):
    """The single-file modeling_pig.py: merged imports, then every section verbatim (without its own
    imports and with its docstring as a comment block), then the aliases the wrapper relies on."""
    spec = REPOS[family]
    imports, parts, defined = [], [], {}
    wrapper_doc = None
    for title, files in sections(family):
        chunks = []
        for f in files:
            if ":" in f:
                path, fn = f.split(":")
                chunks.append((None, [], function_source(ROOT / path, fn)))
            else:
                chunks.append(split_module((ROOT / f).read_text()))
        banner = "# " + "=" * 96 + "\n# " + title + "\n# " + "=" * 96 + "\n"
        text = banner
        for doc, imps, body in chunks:
            imports += imps
            if doc and title == "hub/modeling_pig.py":
                wrapper_doc = doc
            elif doc:
                text += comment_block(doc) + "\n"
            for name in top_level_names(body):
                if name in defined and name != "__all__":
                    raise RuntimeError(f"{name!r} defined in both {defined[name]} and {title}")
                defined[name] = title
            text += body + "\n"
        if title == f"pig/fde/common.py + pig/fde/{family}.py + pig/fde/__init__.py:load_fde":
            text += f"PIGFDE = {spec['fde_class']}\n\n"
        elif title == f"pig/{family}.py":
            text += (f"FAMILY = {family!r}\nPIGDecoder = {spec['decoder_class']}\n"
                     f"PIGGeometry = {spec['geometry_class']}\n\n")
        elif title == "pig/api.py":
            text += "PIGOutput = Output\n\n"
        parts.append(text)
    merged = sorted(set(imports), key=lambda s: (not s.startswith("import "), s))
    header = (f"# Built from https://github.com/Quentin-Fruytier/PIG-DLM-release ({stamp}) by `python -m hub.build`.\n"
              f"# Do not edit here: every section below is the package's code, verbatim; edit the package and rebuild.\n")
    return (header + f'"""{wrapper_doc.strip()}\n"""\n' + "\n".join(merged) + "\nfrom .configuration_pig import PIGConfig\n\n\n"
            + "\n".join(parts))


def safetensors_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def build(family, out=DIST):
    import numpy as np
    import torch
    from safetensors.torch import load_file, save_file

    from hub.configuration_pig import PIGConfig

    spec = REPOS[family]
    stamp = git_stamp()
    dst = Path(out) / spec["name"]
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    # --- code: the config class as is, everything else in one modeling file
    from pig.api import DEFAULTS, FAMILIES
    config_src = (HUB / "configuration_pig.py").read_text()
    marker = "from pig.api import DEFAULTS, FAMILIES   # hub.build replaces this line with the literal values\n"
    assert config_src.count(marker) == 1, "hub/configuration_pig.py: defaults import line not found"
    (dst / "configuration_pig.py").write_text(config_src.replace(
        marker, f"FAMILIES = {FAMILIES!r}\n\n# the knobs generate() uses when the caller does not pass them (pig/api.py)\nDEFAULTS = {DEFAULTS!r}\n"))
    (dst / "modeling_pig.py").write_text(modeling_source(family, stamp))
    ast.parse((dst / "modeling_pig.py").read_text())        # must at least be valid Python

    # --- weights: the adapter (keys "fde.*") and the two tables (keys "tables.<axis>.*"), one file
    src = ASSETS / family / "fde.safetensors"
    state = load_file(str(src), device="cpu")
    meta = dict(safetensors_header(src).get("__metadata__", {}))
    meta.update({"format": "pt", "source": f"assets/{family}/fde.safetensors", "package": stamp})
    tensors = {f"fde.{k}": v.contiguous() for k, v in state.items()}
    provenance = {}
    for axis, t in (("confidence", "flip_confidence.npz"), ("entropy", "flip_entropy.npz")):
        z = np.load(ASSETS / family / t, allow_pickle=False)
        for key in ("x_edges", "mu_edges", "rates", "rows", "flips"):
            tensors[f"tables.{axis}.{key}"] = torch.from_numpy(np.ascontiguousarray(z[key]))
        provenance[axis] = {"source": f"assets/{family}/{t}", "rows": int(z["rows"].sum()),
                            "mismatches": int(z["flips"].sum()), "description": str(z["description"])}
    save_file(tensors, str(dst / "model.safetensors"), metadata=meta)
    n_params = int(sum(v.numel() for v in state.values()))

    # --- config
    if family == "llada":
        from pig.llada import MODEL_ID, REVISION
    else:
        from pig.nemotron import MODEL_ID, REVISION
    adapter = {"parameters": n_params, "source": f"assets/{family}/fde.safetensors", "package": stamp}
    for k in ("epoch", "val_edge_corr", "description"):
        if k in meta:
            adapter[k] = meta[k]
    cfg = PIGConfig(family=family, backbone=MODEL_ID, backbone_revision=REVISION, adapter=adapter,
                    tables=["confidence", "entropy"], architectures=["PIGModel"],
                    auto_map={"AutoConfig": "configuration_pig.PIGConfig", "AutoModel": "modeling_pig.PIGModel"})
    cfg.table_provenance = provenance
    cfg.save_pretrained(dst)

    # --- card + agents reference
    for src_name, dst_name in ((f"{spec['name']}.md", "README.md"), (f"{spec['name']}.AGENTS.md", "AGENTS.md")):
        card = HUB / "cards" / src_name
        if card.exists():
            shutil.copyfile(card, dst / dst_name)
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
