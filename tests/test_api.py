"""pig.api: the generate() interface on a fake decoder (no backbone), and the rule factory."""
import pytest

from pig import DEFAULTS, MODES, PIG, Output, load_tables, make_rule
from pig.rules import ConfidenceThreshold, PIGExt, PIGInt

TABLES = load_tables("llada")


class Geometry:
    def __init__(self, kind, size):
        self.kind, self.size = kind, size


class Tokenizer:
    class Encoded:
        input_ids = [7, 8, 9]

    def decode(self, ids, skip_special_tokens=True):
        return "hello " + " ".join(map(str, ids))

    def __call__(self, text):
        return self.Encoded()


class FakeDecoder:
    geometry_cls = Geometry
    model = "backbone"

    def __init__(self, family):
        self.family, self.tokenizer, self.calls = family, Tokenizer(), []

    def chat_prompt(self, text):
        return [1, 2, 3]

    def generate(self, ids, rule, geometry, **length):
        self.calls.append(dict(ids=ids, rule=rule, geometry=geometry, **length))
        return list(range(8)), 4, {"forward": 1.0}


def test_make_rule_defaults_and_validation():
    assert MODES == ("pig.int", "pig.ext", "confidence")
    assert isinstance(make_rule(TABLES, "pig.int", defaults=DEFAULTS["llada"]), PIGInt)
    assert isinstance(make_rule(TABLES, defaults=DEFAULTS["llada"]), PIGInt)            # default mode
    assert isinstance(make_rule(TABLES, "pig.ext", theta=0.25, floor=0.8), PIGExt)
    assert isinstance(make_rule({}, "confidence", nu=0.9), ConfidenceThreshold)
    for bad in (dict(mode="pig.int"), dict(mode="pig.ext"), dict(mode="confidence"),        # a knob is required
                dict(mode="pig.int", phi=0.1, theta=0.5), dict(mode="pig.int", phi=0.1, window=8),
                dict(mode="pig.ext", theta=0.5, phi=0.1), dict(mode="pig.ext", theta=0.5, nu=0.9),
                dict(mode="confidence", nu=0.9, floor=0.8), dict(mode="greedy")):
        with pytest.raises(ValueError):
            make_rule(TABLES, **bad)


def test_generate_on_llada_uses_the_canvas_and_the_defaults():
    model = PIG(FakeDecoder("llada"), TABLES)
    out = model.generate("prompt")
    assert isinstance(out, Output) and out.mode == "pig.int" and out.forwards == 4 and out.tpf == 2.0
    assert out.tokens == list(range(8)) and out.text.startswith("hello")
    call = model.decoder.calls[-1]
    assert call["ids"] == [1, 2, 3] and call["canvas"] == 256 and "max_new" not in call
    assert (call["geometry"].kind, call["geometry"].size) == ("block", 32) and isinstance(call["rule"], PIGInt)
    out = model.generate([5, 6], mode="pig.ext", theta=0.25, floor=0.8, geometry="sliding", block=64, max_new_tokens=128)
    call = model.decoder.calls[-1]
    assert call["ids"] == [5, 6] and call["canvas"] == 128 and isinstance(call["rule"], PIGExt)
    assert (call["geometry"].kind, call["geometry"].size) == ("sliding", 64)
    assert out.params == dict(geometry="sliding", block=64, theta=0.25, floor=0.8)
    assert model.encode("x", chat_template=False) == [7, 8, 9]


def test_generate_on_nemotron_uses_max_new():
    model = PIG(FakeDecoder("nemotron"), TABLES, defaults=dict(block=16))
    model.generate("prompt", mode="confidence", nu=0.7)
    call = model.decoder.calls[-1]
    assert call["max_new"] == 2048 and "canvas" not in call and isinstance(call["rule"], ConfidenceThreshold)
    assert call["geometry"].size == 16                                   # the override merged into the defaults
    with pytest.raises(ValueError):
        model.generate("prompt", mode="pig.int", theta=0.5)
    with pytest.raises(ValueError):
        PIG(FakeDecoder("dream"), TABLES)
