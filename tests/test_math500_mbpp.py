"""The MATH-500 and MBPP prompts and graders on hand-made text (no dataset, no model)."""
import pytest

from experiments.tasks import math500, mbpp

MATH = dict(task_id="math500/0", problem="What is $1+1$?", answer="2")
MBPP = dict(task_id="mbpp/11", text="Write a function to add two numbers.",
            test_list=["assert add(1, 2) == 3", "assert add(0, 0) == 0"], test_setup_code="")


def test_math_extract_prefers_final_answer_then_boxed_then_flexible():
    assert math500.extract("so \\boxed{3}. Final Answer: The final answer is $3$. I hope it is correct.") == ("3", "3")
    assert math500.extract("We get $x = \\boxed{\\dfrac{1}{2}}$ at the end.") == ("\\frac{1}{2}", "\\frac{1}{2}")
    assert math500.extract("The value is $5$ and then 7 appears.") == (None, "5")
    assert math500.extract("just 42 at the end") == (None, "42")
    assert math500.extract("\\boxed{7}\nProblem: next one \\boxed{9}") == ("7", "7")       # cut at the stop


def test_math_normalisation_and_equivalence():
    assert math500.normalize("\\left( 3, \\frac{\\pi}{2} \\right)") == "(3,\\frac{\\pi}{2})"
    assert math500.normalize("\\text{ 5 } dollars") == "5"
    assert math500.normalize("007.0") == "7"
    assert math500.is_equiv("\\frac{1}{2}", "\\frac{1}{2}") and not math500.is_equiv(None, "2")
    assert math500.is_equiv("\\frac{2}{4}", "\\frac{1}{2}")                                   # sympy fallback


def test_math_prompts_and_scores():
    assert math500.prompt_text("nemotron", MATH) == "What is $1+1$?\nPlease reason step by step."
    text = math500.prompt_text("llada", MATH)
    assert text.count("Problem:") == 5 and text.endswith("Problem:\nWhat is $1+1$?\n\nSolution:")
    assert "Final Answer: The final answer is $[2,5)$" in text                               # Minerva exemplar 1
    assert math500.score("llada", MATH, "Thus $1+1=\\boxed{2}$.") == {"strict": True, "flex": True}
    assert math500.score("llada", MATH, "It is $2$.") == {"strict": False, "flex": True}
    assert math500.SHOTS == {"llada": 4, "nemotron": 0} and math500.PRIMARY == "strict"


def test_mbpp_code_extraction_keeps_the_parsable_prefix_and_drops_own_asserts():
    gen = "```python\ndef add(a, b):\n    return a + b\nassert add(1, 1) == 3\n```\nThis function adds."
    assert mbpp.extract_code(gen) == "def add(a, b):\n    return a + b"
    gen2 = "[BEGIN]\ndef mul(a,b):\n    return a*b\n[DONE]\nYou are an expert Python programmer"
    code = mbpp.extract_code(gen2)
    assert "[BEGIN]" not in code and "[DONE]" not in code and code.startswith("def mul")
    assert mbpp.extract_code("def f(x):\n    return x\nThis returns x unchanged.") == "def f(x):\n    return x"


def test_mbpp_tests_run_in_a_sandbox():
    assert mbpp.run_tests("def add(a, b):\n    return a + b", MBPP["test_list"])
    assert not mbpp.run_tests("def add(a, b):\n    return a - b", MBPP["test_list"])
    assert not mbpp.run_tests("import time\ntime.sleep(30)\ndef add(a, b): return 3", ["assert True"], timeout=3)
    assert mbpp.score("nemotron", MBPP, "```python\ndef add(a, b):\n    return a + b\n```") == {"pass@1": True}


def test_mbpp_prompts_follow_each_models_protocol():
    assert mbpp.prompt_text("nemotron", MBPP) == ("You are an expert Python programmer, and here is your task: "
                                                  "Write a function to add two numbers. Your code should pass these tests:\n"
                                                  "assert add(1, 2) == 3\nassert add(0, 0) == 0\n")
    text = mbpp.prompt_text("llada", MBPP, exemplars="EXEMPLARS")
    assert text == ("EXEMPLARS\n\nYou are an expert Python programmer, and here is your task: Write a function to add two numbers.\n"
                    "Your code should pass these tests:\n\nassert add(1, 2) == 3\nassert add(0, 0) == 0\n[BEGIN]\n")
    assert mbpp.N == {"llada": 500, "nemotron": 499} and mbpp.SHOTS == {"llada": 3, "nemotron": 0}
