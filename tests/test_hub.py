"""The Hub build: one modeling file that is the package's code section by section, one weights file
holding the adapter and the tables, and a wrapper that loads them without a backbone on the CPU."""
import ast
import json

import numpy as np
import pytest
import torch

from hub.build import ASSETS, REPOS, TOKENIZER_FILES, build, sections, split_module, write_shards

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning", "ignore::FutureWarning")
transformers = pytest.importorskip("transformers")
pytest.importorskip("safetensors")


@pytest.fixture(scope="module", params=sorted(REPOS))
def built(request, tmp_path_factory):
    out = tmp_path_factory.mktemp("hub")
    return request.param, build(request.param, out, backbone=False)      # tiny backbone config, no backbone weights


def test_layout_is_the_standard_shape(built):
    fam, d = built
    spec = REPOS[fam]
    tokenizer = {f for f in TOKENIZER_FILES if (d / f).exists()}
    assert {"tokenizer.json", "tokenizer_config.json"} <= tokenizer
    assert {p.name for p in d.iterdir()} == {"README.md", "AGENTS.md", "NOTICE.md", "config.json", "configuration_pig.py",
                                             "modeling_pig.py", "model.safetensors", *spec["backbone_files"], *tokenizer}
    cfg = json.loads((d / "config.json").read_text())
    assert cfg["family"] == fam and cfg["auto_map"]["AutoModel"] == "modeling_pig.PIGModel"
    assert cfg["backbone"] and cfg["backbone_revision"] and cfg["tables"] == ["confidence", "entropy"]
    assert cfg["backbone_config"] and "auto_map" not in cfg["backbone_config"] and cfg["dtype"] == "bfloat16"
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
    for alias in ("PIGFDE = ", f"FAMILY = {fam!r}", "PIGDecoder = ", "PIGGeometry = ", "PIGOutput = Output",
                  "PIGBackbone = ", "PIGBackboneConfig = "):
        assert alias in text
    assert "from .backbone import" not in text                                # resolved to the family's files
    for f in REPOS[fam]["backbone_files"]:
        assert (d / f).read_text() == (ASSETS.parent / "pig" / "backbones" / fam / f).read_text()


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
    model = AutoModel.from_pretrained(str(d), trust_remote_code=True, attach=False)
    from safetensors.torch import load_file
    ref = load_file(str(ASSETS / fam / "fde.safetensors"))
    state = model.fde.state_dict()
    assert state.keys() == ref.keys() and all(torch.equal(state[k], v) for k, v in ref.items())
    for axis, t in (("confidence", "flip_confidence.npz"), ("entropy", "flip_entropy.npz")):
        z = np.load(ASSETS / fam / t, allow_pickle=False)
        assert np.array_equal(model.tables[axis].rates, z["rates"]) and model.tables[axis].axis == axis
        assert model.tables[axis](0.5, 0.1) == pytest.approx(float(z["rates"][np.searchsorted(z["x_edges"], 0.5, "right") - 1, 0]))
    assert model.fde.state_dict()[next(iter(ref))].dtype == torch.float32          # _keep_in_fp32_modules
    assert type(model.backbone).__name__ == REPOS[fam]["backbone_model_class"][1] and model.tokenizer is None
    with pytest.raises(RuntimeError, match="no decode loop"):
        model.generate("hello")
    # the rule factory: defaults and knob validation, without a backbone
    assert type(model.rule("pig.int")).__name__ == "PIGInt"
    assert type(model.rule("pig.ext", theta=0.25, floor=0.8)).__name__ == "PIGExt"
    assert type(model.rule("confidence", nu=0.9)).__name__ == "ConfidenceThreshold"
    with pytest.raises(ValueError):
        model.rule("pig.int", theta=0.5)
    with pytest.raises(ValueError):
        model.rule("nonsense")


def test_write_shards_prefixes_and_indexes(tmp_path):
    from safetensors.torch import load_file, save_file
    up = []
    for i in range(2):
        f = tmp_path / f"up{i}.safetensors"
        save_file({f"layer{i}.w": torch.arange(4, dtype=torch.float32) + i}, str(f)); up.append(f)
    extra = {"fde.a": torch.zeros(3), "tables.confidence.rates": torch.ones(2, 2, dtype=torch.float64)}
    out = tmp_path / "out"; out.mkdir()
    wm = write_shards(out, up, extra, {"format": "pt"})
    assert wm == {"backbone.layer0.w": "model-00001-of-00003.safetensors", "backbone.layer1.w": "model-00002-of-00003.safetensors",
                  "fde.a": "model-00003-of-00003.safetensors", "tables.confidence.rates": "model-00003-of-00003.safetensors"}
    idx = json.loads((out / "model.safetensors.index.json").read_text())
    assert idx["weight_map"] == wm and idx["metadata"]["total_size"] == 4 * 4 * 2 + 3 * 4 + 4 * 8
    assert torch.equal(load_file(str(out / "model-00002-of-00003.safetensors"))["backbone.layer1.w"], torch.arange(4.0) + 1)
