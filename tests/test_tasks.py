"""The task registry and the harness plumbing (evaluate / sweep / summarize) on a fake decoder."""
import json
import sys

import pytest

from pig import gsm8k, humaneval, tasks
from pig.evaluate import RULES, cell_name, make_rule, run_cell
from pig.rules import ConfidenceThreshold, PIGExt, PIGInt
from pig.sweep import expand
from pig import summarize


class Geometry:
    def __init__(self, kind, size):
        self.kind, self.size = kind, size

    def tag(self):
        return f"{self.kind}{self.size}"


class FakeTokenizer:
    def __init__(self, texts):
        self.texts, self.i = list(texts), 0

    def decode(self, ids, skip_special_tokens=True):
        t = self.texts[self.i % len(self.texts)]
        self.i += 1
        return t


class FakeDecoder:
    model_id, revision = "fake/model", "0000"

    def __init__(self, texts, forwards):
        self.tokenizer, self.forwards, self.k = FakeTokenizer(texts), forwards, 0

    def chat_prompt(self, text):
        return [1, 2, 3]

    def generate(self, ids, rule, geometry):
        f = self.forwards[self.k % len(self.forwards)]
        self.k += 1
        return list(range(10)), f, {"forward": 1.0, "rule": 0.1}


PROBLEMS = [dict(task_id=f"gsm8k/{i}", question=f"q{i}", answer=f"r\n#### 7") for i in range(3)]


def test_registry_exposes_one_interface():
    assert tasks.TASKS == {"humaneval": humaneval, "gsm8k": gsm8k}
    for name, mod in tasks.TASKS.items():
        assert mod.NAME == name and mod.PRIMARY in mod.METRICS
        assert set(mod.SHOTS) == {"llada", "nemotron"}
        for fn in ("load_problems", "prompt_text", "score"):
            assert callable(getattr(mod, fn))
    assert tasks.get("gsm8k") is gsm8k
    with pytest.raises(ValueError):
        tasks.get("mmlu")


def test_cell_names_carry_task_and_knob():
    g = Geometry("block", 64)
    assert cell_name("llada", "gsm8k", "pig-int", g, phi=0.1) == "llada_gsm8k_block64_pig-int_phi0.1"
    assert cell_name("llada", "gsm8k", "pig-ext", g, theta=0.25, floor=0.8) == "llada_gsm8k_block64_pig-ext_theta0.25_floor0.8"
    assert cell_name("nemotron", "humaneval", "confidence", g, nu=0.9) == "nemotron_humaneval_block64_confidence_nu0.9"


def test_make_rule_validates_the_knobs():
    assert RULES == ("pig-int", "pig-ext", "confidence")
    assert isinstance(make_rule("llada", "confidence", nu=0.9), ConfidenceThreshold)
    assert isinstance(make_rule("llada", "pig-int", phi=0.1), PIGInt)
    assert isinstance(make_rule("llada", "pig-ext", theta=0.25, floor=0.8), PIGExt)
    for bad in (dict(rule="confidence"), dict(rule="confidence", nu=0.9, phi=0.1),
                dict(rule="pig-int", phi=0.1, nu=0.9), dict(rule="pig-ext", theta=0.5, nu=0.9),
                dict(rule="pig-ext", phi=0.5), dict(rule="greedy", nu=0.9)):
        with pytest.raises(ValueError):
            make_rule("llada", **bad)


def test_sweep_expands_every_lane():
    cfg = {"model": "llada", "task": "gsm8k", "window": 8,
           "geometries": {"block64": {"geometry": "block", "block": 64, "confidence": [0.9],
                                      "pig-int": [0.1, 0.2], "pig-ext": [[0.125, 0.9], 0.5]}}}
    cells = expand(cfg)
    assert [c[2]["rule"] for c in cells] == ["confidence", "pig-int", "pig-int", "pig-ext", "pig-ext"]
    assert cells[0][2] == dict(rule="confidence", phi=None, theta=None, floor=None, nu=0.9, window=8)
    assert cells[3][2]["theta"] == 0.125 and cells[3][2]["floor"] == 0.9 and cells[4][2]["floor"] is None


def test_run_cell_records_every_metric_and_summarize_tabulates_it(tmp_path, monkeypatch, capsys):
    decoder = FakeDecoder(["3 + 4 = 7.\n#### 7", "I think 8", "#### 7 and finally 9"], [64, 32, 32])
    spec = dict(rule="confidence", phi=None, theta=None, floor=None, nu=0.9, window=16)
    out = tmp_path / "llada_gsm8k_block64_confidence_nu0.9.json"
    r = run_cell(decoder, "llada", "gsm8k", ConfidenceThreshold(0.9), Geometry("block", 64), spec, PROBLEMS, out,
                 log_every=0)
    s = r["summary"]
    assert s["metric"] == "flex" and s["n"] == 3 and s["passed"] == 1 and s["accuracy"] == pytest.approx(1 / 3)
    assert s["metrics"]["strict"]["passed"] == 2 and s["metrics"]["flex"]["passed"] == 1
    assert s["tpf"] == pytest.approx(256 / ((64 + 32 + 32) / 3))
    c = r["config"]
    assert c["task"] == "gsm8k" and c["shots"] == 5 and c["rule"] == "confidence" and c["nu"] == 0.9
    assert list(c["assets"]) == ["llada/fde.safetensors"]          # the baseline reads no table
    assert [p["scores"] for p in r["problems"]] == [{"flex": True, "strict": True}, {"flex": False, "strict": False},
                                                    {"flex": False, "strict": True}]
    assert json.loads(out.read_text())["summary"]["accuracy"] == pytest.approx(1 / 3)
    monkeypatch.setattr(sys, "argv", ["summarize", str(tmp_path)])
    summarize.main()
    text = capsys.readouterr().out
    assert "llada -- gsm8k -- block B=64" in text and "| flex |" in text and "strict" in text
    assert "| CONFIDENCE | nu=0.9 | 6.00 | 0.333 |" in text and "| 0.667 | 3 |" in text
