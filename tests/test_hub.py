"""The Hub build: one modeling file that is the package's code section by section, one weights file
holding the adapter and the tables, and a wrapper that loads them without a backbone on the CPU."""
import ast
import json

import numpy as np
import pytest
import torch

from hub.build import ASSETS, REPOS, build, sections, split_module

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning", "ignore::FutureWarning")
transformers = pytest.importorskip("transformers")
pytest.importorskip("safetensors")


@pytest.fixture(scope="module", params=sorted(REPOS))
def built(request, tmp_path_factory):
    out = tmp_path_factory.mktemp("hub")
    return request.param, build(request.param, out)


def test_layout_is_the_standard_shape(built):
    fam, d = built
    assert {p.name for p in d.iterdir()} == {"README.md", "AGENTS.md", "config.json", "configuration_pig.py",
                                             "modeling_pig.py", "model.safetensors"}
    cfg = json.loads((d / "config.json").read_text())
    assert cfg["family"] == fam and cfg["auto_map"]["AutoModel"] == "modeling_pig.PIGModel"
    assert cfg["backbone"] and cfg["backbone_revision"] and cfg["tables"] == ["confidence", "entropy"]
    assert set(cfg["table_provenance"]) == {"confidence", "entropy"}
    from pig.api import DEFAULTS
    assert cfg["defaults"] == DEFAULTS[fam]
    assert "from ." not in (d / "configuration_pig.py").read_text()        # nothing for the dynamic loader to chase


def test_modeling_file_is_the_package_section_by_section(built):
    fam, d = built
    text = (d / "modeling_pig.py").read_text()
    ast.parse(text)
    present = set(ln.rstrip() for ln in text.splitlines())
    for title, files in sections(fam):
        assert f"# {title}" in text
        for f in files:
            if ":" in f:
                continue                                           # load_fde is checked by test_wrapper below
            _, _, body = split_module((ASSETS.parent / f).read_text())
            for line in body.splitlines():
                if line.strip() and not line.lstrip().startswith("#"):
                    assert line.rstrip() in present, f"{f}: line missing from modeling_pig.py: {line[:70]!r}"
    for alias in ("PIGFDE = ", f"FAMILY = {fam!r}", "PIGDecoder = ", "PIGGeometry = ", "PIGOutput = Output"):
        assert alias in text


def test_weights_file_holds_the_adapter_and_the_tables(built):
    fam, d = built
    from safetensors.torch import load_file
    ours = load_file(str(d / "model.safetensors"))
    ref = load_file(str(ASSETS / fam / "fde.safetensors"))
    fde = {k: v for k, v in ours.items() if k.startswith("fde.")}
    assert set(fde) == {f"fde.{k}" for k in ref} and all(torch.equal(fde[f"fde.{k}"], v) for k, v in ref.items())
    for axis, t in (("confidence", "flip_confidence.npz"), ("entropy", "flip_entropy.npz")):
        z = np.load(ASSETS / fam / t, allow_pickle=False)
        for key in ("x_edges", "mu_edges", "rates", "rows", "flips"):
            assert np.array_equal(ours[f"tables.{axis}.{key}"].numpy(), z[key]), (axis, key)
    assert set(ours) == set(fde) | {f"tables.{a}.{k}" for a in ("confidence", "entropy")
                                    for k in ("x_edges", "mu_edges", "rates", "rows", "flips")}


def test_wrapper_loads_without_backbone(built):
    fam, d = built
    from transformers import AutoModel
    model = AutoModel.from_pretrained(str(d), trust_remote_code=True, load_backbone=False)
    from safetensors.torch import load_file
    ref = load_file(str(ASSETS / fam / "fde.safetensors"))
    state = model.fde.state_dict()
    assert state.keys() == ref.keys() and all(torch.equal(state[k], v) for k, v in ref.items())
    for axis, t in (("confidence", "flip_confidence.npz"), ("entropy", "flip_entropy.npz")):
        z = np.load(ASSETS / fam / t, allow_pickle=False)
        assert np.array_equal(model.tables[axis].rates, z["rates"]) and model.tables[axis].axis == axis
        assert model.tables[axis](0.5, 0.1) == pytest.approx(float(z["rates"][np.searchsorted(z["x_edges"], 0.5, "right") - 1, 0]))
    assert model.backbone is None and model.tokenizer is None
    with pytest.raises(RuntimeError, match="no backbone"):
        model.generate("hello")
    # the rule factory: defaults and knob validation, without a backbone
    assert type(model.rule("pig.int")).__name__ == "PIGInt"
    assert type(model.rule("pig.ext", theta=0.25, floor=0.8)).__name__ == "PIGExt"
    assert type(model.rule("confidence", nu=0.9)).__name__ == "ConfidenceThreshold"
    with pytest.raises(ValueError):
        model.rule("pig.int", theta=0.5)
    with pytest.raises(ValueError):
        model.rule("nonsense")
