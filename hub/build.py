"""Build the Hugging Face Hub repositories: one model each, the backbone with its PIG adapter and tables.

    python -m hub.build [--out DIR] [--family llada|nemotron] [--no-backbone]

<out>/<repo name>/ holds everything the Hub repository needs:

    config.json                 PIGConfig: the backbone's own config nested, the decoding defaults, provenance
    configuration_pig.py        the config class (defaults inlined)
    modeling_pig.py             the PIG sections of pig/ (rules, tables, adapter, decode loop, generate())
                                concatenated verbatim, then the wrapper class of hub/modeling_pig.py
    <backbone code>             pig/backbones/<family>/*.py copied verbatim (the upstream modeling code with
                                the changes marked in its headers)
    model-0000i-of-0000n.safetensors + model.safetensors.index.json
                                the backbone's weights (keys backbone.*, copied tensor for tensor from the
                                upstream repository at the pinned revision), the adapter (fde.*) and the
                                tables (tables.*). --no-backbone writes a single model.safetensors with the
                                adapter and tables only and a tiny backbone config (what the tests build)
    tokenizer files             copied from the upstream revision
    NOTICE.md                   provenance and licences of the backbone
    README.md, AGENTS.md        the cards (hub/cards/<repo name>.md, hub/cards/<repo name>.AGENTS.md)

The package is the single source of truth: nothing in the output is edited by hand, and tests/test_hub.py
rebuilds it (without the backbone) to check that every section of modeling_pig.py is the package's code.
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
                  decoder_class="LLaDADecoder", geometry_class="Geometry",
                  backbone_files=["configuration_llada.py", "modeling_llada.py"],
                  backbone_config_class=("configuration_llada", "LLaDAConfig"),
                  backbone_model_class=("modeling_llada", "LLaDAModelLM"),
                  tiny=dict(d_model=64, n_heads=2, n_kv_heads=2, n_layers=2, mlp_hidden_size=128, max_sequence_length=512)),
    "nemotron": dict(name="Nemotron-Labs-Diffusion-3B-PIG", repo="UT-IFML/Nemotron-Labs-Diffusion-3B-PIG",
                     fde_class="NemotronFDE", decoder_class="NemotronDecoder", geometry_class="Window",
                     backbone_files=["configuration_nemotron_labs_diffusion.py", "modeling_ministral.py",
                                     "modeling_nemotron_labs_diffusion.py"],
                     backbone_config_class=("configuration_nemotron_labs_diffusion", "NemotronLabsDiffusionConfig"),
                     backbone_model_class=("modeling_nemotron_labs_diffusion", "NemotronLabsDiffusionModel"),
                     tiny=dict(hidden_size=64, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                               intermediate_size=128, head_dim=32)),
}
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja")
DROP_FROM_BACKBONE_CONFIG = ("auto_map", "architectures", "_name_or_path", "transformers_version", "torch_dtype", "dtype")


def sections(family):
    """The sections of modeling_pig.py, in dependency order: (title, source files relative to the repo root)."""
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
    """-> (docstring or None, absolute import statements, body). Relative imports are dropped: in the
    single-file build every pig/ name is in one namespace (backbone imports are added by modeling_source)."""
    tree = ast.parse(src)
    lines = src.splitlines(keepends=True)
    drop, doc, imports = set(), None, []
    body = tree.body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
        doc = body[0].value.value
        drop.update(range(body[0].lineno, body[0].end_lineno + 1))
    for node in body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            drop.update(range(node.lineno, node.end_lineno + 1))
            if isinstance(node, ast.Import) or node.level == 0:
                imports.append("".join(lines[node.lineno - 1:node.end_lineno]).rstrip("\n"))
    kept = [ln for i, ln in enumerate(lines, 1) if i not in drop]
    return doc, imports, "".join(kept).strip("\n") + "\n"


def function_source(path, name):
    src = path.read_text()
    for node in ast.parse(src).body:
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
    """modeling_pig.py: merged imports, the backbone's classes imported from their sibling files, every pig/
    section verbatim (its docstring as a comment block), the aliases the wrapper relies on, the wrapper."""
    spec = REPOS[family]
    imports, parts, defined, wrapper_doc = [], [], {}, None
    for title, files in sections(family):
        chunks = []
        for f in files:
            if ":" in f:
                path, fn = f.split(":")
                chunks.append((None, [], function_source(ROOT / path, fn)))
            else:
                chunks.append(split_module((ROOT / f).read_text()))
        text = "# " + "=" * 96 + "\n# " + title + "\n# " + "=" * 96 + "\n"
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
        if title.startswith("pig/fde/"):
            text += f"PIGFDE = {spec['fde_class']}\n\n"
        elif title == f"pig/{family}.py":
            text += (f"FAMILY = {family!r}\nPIGDecoder = {spec['decoder_class']}\n"
                     f"PIGGeometry = {spec['geometry_class']}\n\n")
        elif title == "pig/api.py":
            cmod, ccls = spec["backbone_config_class"]
            mmod, mcls = spec["backbone_model_class"]
            text += f"PIGOutput = Output\nPIGBackboneConfig = {ccls}\nPIGBackbone = {mcls}\n\n"
        parts.append(text)
    future = sorted({i for i in imports if i.startswith("from __future__")})
    merged = sorted({i for i in imports if not i.startswith("from __future__")},
                    key=lambda s: (not s.startswith("import "), s))
    cmod, ccls = spec["backbone_config_class"]
    mmod, mcls = spec["backbone_model_class"]
    header = (f"# Built from https://github.com/Quentin-Fruytier/PIG-DLM-release ({stamp}) by `python -m hub.build`.\n"
              f"# Do not edit here: every section below is the package's code, verbatim; edit the package and rebuild.\n")
    return (header + "\n".join(future) + ("\n" if future else "") + f'"""{wrapper_doc.strip()}\n"""\n' + "\n".join(merged)
            + f"\nfrom .configuration_pig import PIGConfig\nfrom .{cmod} import {ccls}\nfrom .{mmod} import {mcls}\n\n\n"
            + "\n".join(parts))


def safetensors_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def upstream_snapshot(family):
    """The upstream repository's files at the pinned revision (from the local cache or the Hub)."""
    from huggingface_hub import snapshot_download

    mod = __import__(f"pig.{family}", fromlist=["MODEL_ID", "REVISION"])
    return Path(snapshot_download(mod.MODEL_ID, revision=mod.REVISION,
                                  allow_patterns=["*.json", "*.safetensors", "*.jinja"])), mod.MODEL_ID, mod.REVISION


def backbone_config(snapshot, tiny=None):
    cfg = json.loads((snapshot / "config.json").read_text())
    cfg = {k: v for k, v in cfg.items() if k not in DROP_FROM_BACKBONE_CONFIG}
    if tiny:
        cfg.update(tiny)
    return cfg


def pig_tensors(family, meta):
    """The adapter (fde.*) and the tables (tables.*) as tensors, plus the table provenance."""
    import numpy as np
    import torch
    from safetensors.torch import load_file

    state = load_file(str(ASSETS / family / "fde.safetensors"), device="cpu")
    tensors = {f"fde.{k}": v.contiguous() for k, v in state.items()}
    provenance = {}
    for axis, t in (("confidence", "flip_confidence.npz"), ("entropy", "flip_entropy.npz")):
        z = np.load(ASSETS / family / t, allow_pickle=False)
        for key in ("x_edges", "mu_edges", "rates", "rows", "flips"):
            tensors[f"tables.{axis}.{key}"] = torch.from_numpy(np.ascontiguousarray(z[key]))
        provenance[axis] = {"source": f"assets/{family}/{t}", "rows": int(z["rows"].sum()),
                            "mismatches": int(z["flips"].sum()), "description": str(z["description"])}
    return tensors, int(sum(v.numel() for v in state.values())), provenance


def write_shards(dst, upstream_files, extra, meta, prefix="backbone."):
    """Write model-0000i-of-0000n.safetensors (one per upstream file, keys prefixed) plus one holding `extra`,
    and model.safetensors.index.json. Returns the weight map."""
    from safetensors.torch import load_file, save_file

    n = len(upstream_files) + 1
    weight_map, total = {}, 0
    for i, src in enumerate(upstream_files, 1):
        name = f"model-{i:05d}-of-{n:05d}.safetensors"
        state = load_file(str(src), device="cpu")
        state = {prefix + k: v.contiguous() for k, v in state.items()}
        save_file(state, str(dst / name), metadata={"format": "pt"})
        for k, v in state.items():
            weight_map[k] = name
            total += v.numel() * v.element_size()
        del state
    name = f"model-{n:05d}-of-{n:05d}.safetensors"
    save_file(extra, str(dst / name), metadata=meta)
    for k, v in extra.items():
        weight_map[k] = name
        total += v.numel() * v.element_size()
    (dst / "model.safetensors.index.json").write_text(json.dumps({"metadata": {"total_size": total},
                                                                   "weight_map": weight_map}, indent=1))
    return weight_map


def upstream_weight_files(snapshot):
    index = snapshot / "model.safetensors.index.json"
    if index.exists():
        return [snapshot / f for f in sorted(set(json.loads(index.read_text())["weight_map"].values()))]
    return [snapshot / "model.safetensors"]


def notice(family, model_id, revision):
    if family == "llada":
        return (f"# NOTICE\n\nThe backbone weights, tokenizer and modeling code in this repository are copied from\n"
                f"https://huggingface.co/{model_id} (revision {revision}), released by its authors under the MIT License.\n"
                f"The modeling files carry the changes made for this package in comments marked `PIG:`.\n\n"
                f"The PIG adapter, calibration tables, wrapper and the rest of the code are released under the MIT License\n"
                f"(see the code repository, https://github.com/Quentin-Fruytier/PIG-DLM-release).\n")
    return (f"# NOTICE\n\nThe backbone weights, tokenizer and modeling code in this repository are copied from\n"
            f"https://huggingface.co/{model_id} (revision {revision}), released by NVIDIA Corporation under the\n"
            f"NVIDIA Nemotron Open Model License\n"
            f"(https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-nemotron-open-model-license/).\n"
            f"Use of the backbone is governed by that licence; NVIDIA's names and marks are used only to identify the\n"
            f"model. The modeling files carry the changes made for this package in comments marked `PIG:`.\n\n"
            f"The PIG adapter, calibration tables, wrapper and the rest of the code are released under the MIT License\n"
            f"(see the code repository, https://github.com/Quentin-Fruytier/PIG-DLM-release).\n")


def build(family, out=DIST, backbone=True):
    from safetensors.torch import save_file

    from hub.configuration_pig import PIGConfig
    from pig.api import DEFAULTS, FAMILIES

    spec = REPOS[family]
    stamp = git_stamp()
    dst = Path(out) / spec["name"]
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    snapshot, model_id, revision = upstream_snapshot(family)

    # --- code: config class with the defaults inlined, one PIG modeling file, the backbone's files verbatim
    config_src = (HUB / "configuration_pig.py").read_text()
    marker = "from pig.api import DEFAULTS, FAMILIES   # hub.build replaces this line with the literal values\n"
    assert config_src.count(marker) == 1, "hub/configuration_pig.py: defaults import line not found"
    (dst / "configuration_pig.py").write_text(config_src.replace(
        marker, f"FAMILIES = {FAMILIES!r}\n\n# the knobs generate() uses when the caller does not pass them (pig/api.py)\nDEFAULTS = {DEFAULTS!r}\n"))
    (dst / "modeling_pig.py").write_text(modeling_source(family, stamp))
    ast.parse((dst / "modeling_pig.py").read_text())
    for f in spec["backbone_files"]:
        shutil.copyfile(ROOT / "pig" / "backbones" / family / f, dst / f)

    # --- weights
    extra, n_params, provenance = pig_tensors(family, None)
    meta = dict(safetensors_header(ASSETS / family / "fde.safetensors").get("__metadata__", {}))
    meta.update({"format": "pt", "source": f"assets/{family}/fde.safetensors", "package": stamp})
    if backbone:
        write_shards(dst, upstream_weight_files(snapshot), extra, meta)
    else:
        save_file(extra, str(dst / "model.safetensors"), metadata=meta)

    # --- tokenizer + notice
    for f in TOKENIZER_FILES:
        if (snapshot / f).exists():
            shutil.copyfile(snapshot / f, dst / f)
    (dst / "NOTICE.md").write_text(notice(family, model_id, revision))

    # --- config
    adapter = {"parameters": n_params, "source": f"assets/{family}/fde.safetensors", "package": stamp}
    for k in ("epoch", "val_edge_corr", "description"):
        if k in meta:
            adapter[k] = meta[k]
    cfg = PIGConfig(family=family, backbone=model_id, backbone_revision=revision,
                    backbone_config=backbone_config(snapshot, None if backbone else spec["tiny"]),
                    adapter=adapter, tables=["confidence", "entropy"], architectures=["PIGModel"],
                    auto_map={"AutoConfig": "configuration_pig.PIGConfig", "AutoModel": "modeling_pig.PIGModel"},
                    dtype="bfloat16")
    cfg.table_provenance = provenance
    cfg.save_pretrained(dst)

    # --- cards
    for src_name, dst_name in ((f"{spec['name']}.md", "README.md"), (f"{spec['name']}.AGENTS.md", "AGENTS.md")):
        card = HUB / "cards" / src_name
        if card.exists():
            shutil.copyfile(card, dst / dst_name)
    return dst


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(DIST))
    ap.add_argument("--family", choices=sorted(REPOS), action="append")
    ap.add_argument("--no-backbone", action="store_true", help="adapter + tables only, tiny backbone config (tests)")
    a = ap.parse_args()
    for fam in a.family or sorted(REPOS):
        d = build(fam, a.out, backbone=not a.no_backbone)
        print(f"[{fam}] -> {d}: " + ", ".join(sorted(p.name for p in d.iterdir())))


if __name__ == "__main__":
    main()
