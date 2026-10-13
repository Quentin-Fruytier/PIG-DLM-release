"""The Hub build: the wrapper must load, on the CPU and without a backbone, exactly the adapter and
tables the harness uses, and its vendored modules must be the package's."""
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from hub.build import ASSETS, REPOS, build

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning", "ignore::FutureWarning")
transformers = pytest.importorskip("transformers")
pytest.importorskip("safetensors")


@pytest.fixture(scope="module", params=sorted(REPOS))
def built(request, tmp_path_factory):
    out = tmp_path_factory.mktemp("hub")
    return request.param, build(request.param, out)


def test_layout(built):
    fam, d = built
    names = {p.name for p in d.iterdir()}
    assert {"config.json", "model.safetensors", "flip_confidence.npz", "flip_entropy.npz", "configuration_pig.py",
            "modeling_pig.py", "rules.py", "tables.py", "api.py", "fde.py", "decoder.py"} <= names
    cfg = json.loads((d / "config.json").read_text())
    assert cfg["family"] == fam and cfg["auto_map"]["AutoModel"] == "modeling_pig.PIGModel"
    assert cfg["backbone"] and cfg["backbone_revision"]


def test_vendored_modules_are_the_package(built):
    fam, d = built
    for name, src in (("rules.py", "pig/rules.py"), ("tables.py", "pig/tables.py"), ("api.py", "pig/api.py"),
                      ("decoder.py", f"pig/{fam}.py")):
        body = "".join(ln for ln in (d / name).read_text().splitlines(keepends=True) if not ln.startswith("# Vendored")
                       and not ln.startswith("# Do not edit"))
        assert body.startswith((ASSETS.parent / src).read_text()), f"{name} is not {src}"
    fde = (d / "fde.py").read_text()
    for src in ("pig/fde/common.py", f"pig/fde/{fam}.py"):
        for line in (ASSETS.parent / src).read_text().splitlines():
            if line.startswith("from .common import"):
                continue
            assert line in fde, f"{src}: line missing from fde.py: {line[:60]!r}"


def test_adapter_checkpoint(built):
    fam, d = built
    from safetensors.torch import load_file
    ours = load_file(str(d / "model.safetensors"))
    ref = load_file(str(ASSETS / fam / "fde.safetensors"))
    assert set(ours) == {f"fde.{k}" for k in ref}
    assert all(torch.equal(ours[f"fde.{k}"], v) for k, v in ref.items())


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
