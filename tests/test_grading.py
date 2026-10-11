"""The HumanEval graders on hand-made completions."""
from pig.humaneval import build_program, grade_llada, grade_nemotron, run_program, wilson

PROMPT = 'def add(a, b):\n    """Add."""\n'
PROBLEM = dict(prompt=PROMPT, entry_point="add", test="def check(f):\n    assert f(1, 2) == 3\n")


def test_llada_continuation_is_appended_to_the_prompt():
    assert grade_llada(PROBLEM, "    return a + b\n")


def test_llada_fenced_redefinition_is_not_prefixed_again():
    completion = "Here you go:\n```python\ndef add(a, b):\n    return a + b\n```\n"
    program = build_program(PROMPT, completion, "add")
    assert program.count("def add") == 1
    assert grade_llada(PROBLEM, completion)


def test_llada_wrong_solution_fails():
    assert not grade_llada(PROBLEM, "    return a - b\n")


def test_llada_stop_sequences_cut_follow_on_code():
    program = build_program(PROMPT, "    return a + b\ndef other():\n    raise Exception()\n", "add")
    assert "other" not in program
    assert grade_llada(PROBLEM, "    return a + b\ndef other():\n    raise Exception()\n")


def test_llada_timeout_counts_as_failure():
    assert not run_program("import time\ntime.sleep(30)\ndef add(a, b): return 3",
                           PROBLEM["test"], "add", timeout=3)


def test_nemotron_last_fenced_definition_is_graded():
    completion = ("```python\ndef add(a, b):\n    return a - b\n```\nFixed:\n"
                  "```python\ndef add(a, b):\n    return a + b\n```\n")
    assert grade_nemotron(PROBLEM, completion)


def test_nemotron_body_only_completion_gets_the_prompt():
    assert grade_nemotron(PROBLEM, "    return a + b\n")
    assert not grade_nemotron(PROBLEM, "    return a * b\n")


def test_nemotron_timeout_counts_as_failure():
    completion = "```python\ndef add(a, b):\n    while True:\n        pass\n```\n"
    assert not grade_nemotron(PROBLEM, completion, timeout=2.0)


def test_wilson_interval():
    lo, hi = wilson(65, 164)
    assert abs((65 / 164 - lo) - 0.072) < 1e-3 and abs((hi - 65 / 164) - 0.076) < 1e-3
    assert wilson(0, 0) == (0.0, 0.0)


def test_nemotron_early_exit_counts_as_failure():
    assert not grade_nemotron(PROBLEM, "```python\ndef add(a, b):\n    return a - b\nimport os\nos._exit(0)\n```\n")
    assert not grade_nemotron(PROBLEM, "```python\nimport sys\nsys.exit(0)\n```\n")
