"""HumanEval (all 164 problems, 0-shot, greedy, pass@1) for the PIG decoders.

Prompts. Each problem's prompt is wrapped in the model's chat template as the user turn, preceded
by one instruction line:
    LLaDA-8B-Instruct          "Complete the following Python function. Return the complete function
                                in a ```python code block."
    Nemotron-Labs-Diffusion-3B "Complete the following Python function."

Grading (execution-based pass@1; the program runs `check(entry_point)` from the task's test block):
    LLaDA     a fenced block that defines the entry point is the complete program; otherwise the text
              is a continuation of the prompt, cut at the standard HumanEval stop sequences. Runs in a
              fresh Python subprocess with a 10 s timeout.
    Nemotron  the last fenced block that defines the entry point (else the last fenced block, else
              the raw text); prefixed with the prompt when it does not define the entry point.
              Executed with exec() in an empty namespace in a fresh Python process, 15 s timeout.
"""
import os
import re
import subprocess
import sys
import tempfile

LLADA_INSTRUCTION = ("Complete the following Python function. Return the complete function "
                     "in a ```python code block.\n\n")
NEMOTRON_INSTRUCTION = "Complete the following Python function.\n\n"

NAME = "humaneval"
N = 164
SHOTS = {"llada": 0, "nemotron": 0}
METRICS = ("pass@1",)
PRIMARY = "pass@1"

# HumanEval stop sequences (continuation-style completions)
STOPS = ("\nclass ", "\ndef ", "\n#", "\nif __name__", "\nprint(", "\n@", "\n```")


def load_problems():
    """The 164 HumanEval test problems, in task order."""
    from datasets import load_dataset

    err = None
    for name in ("openai/openai_humaneval", "openai_humaneval"):
        try:
            return list(load_dataset(name, split="test"))
        except Exception as e:                      # dataset id differs across hub mirrors
            err = e
    raise RuntimeError(f"could not load HumanEval: {err}")


def prompt_text(model, problem):
    return (LLADA_INSTRUCTION if model == "llada" else NEMOTRON_INSTRUCTION) + problem["prompt"]


# ------------------------------------------------------------------------------------------------
# LLaDA grader
# ------------------------------------------------------------------------------------------------
def strip_fence(text):
    """Code inside the first ```...``` block, or None."""
    m = re.search(r"```(?:python|py)?\s*\n(.*?)(?:```|\Z)", text, re.S)
    return m.group(1) if m else None


def build_program(prompt, completion, entry_point):
    fenced = strip_fence(completion)
    defines = re.compile(rf"^\s*def\s+{re.escape(entry_point)}\s*\(", re.M)
    if fenced and defines.search(fenced):
        return fenced                                # a complete redefinition
    if defines.search(completion):
        return completion
    body = fenced if fenced else completion
    for stop in STOPS:
        i = body.find(stop)
        if i != -1:
            body = body[:i]
    return prompt + body


def run_program(program, test, entry_point, timeout=10):
    """Execute program + tests in a fresh interpreter; True iff check() passes."""
    src = f"{program}\n\n{test}\n\ncheck({entry_point})\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(src)
        path = f.name
    try:
        r = subprocess.run([sys.executable, path], capture_output=True, timeout=timeout,
                           env={**os.environ, "OMP_NUM_THREADS": "1"})
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def grade_llada(problem, completion):
    program = build_program(problem["prompt"], completion, problem["entry_point"])
    return run_program(program, problem["test"], problem["entry_point"])


# ------------------------------------------------------------------------------------------------
# Nemotron grader
# ------------------------------------------------------------------------------------------------
def extract_code(text, entry_point):
    fences = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.S)
    if fences:
        with_def = [f for f in fences if f"def {entry_point}" in f]
        return with_def[-1] if with_def else fences[-1]
    return text


# Runs in a fresh interpreter: exec the program in an empty globals dict with stdout silenced, and
# write the verdict only if exec returns normally (an exception, sys.exit or os._exit is a failure).
_EXEC = r"""
import contextlib, io, sys
code = open(sys.argv[1]).read()
try:
    with contextlib.redirect_stdout(io.StringIO()):
        exec(code, {})
except BaseException:
    sys.exit(1)
open(sys.argv[2], "w").write("pass")
"""


def grade_nemotron(problem, completion, timeout=15.0):
    code = extract_code(completion, problem["entry_point"])
    if f"def {problem['entry_point']}" not in code:
        code = problem["prompt"] + code
    full = code + "\n\n" + problem["test"] + "\n" + f"check({problem['entry_point']})\n"
    with tempfile.TemporaryDirectory() as d:
        src, verdict = os.path.join(d, "program.py"), os.path.join(d, "verdict")
        with open(src, "w") as f:
            f.write(full)
        try:
            subprocess.run([sys.executable, "-c", _EXEC, src, verdict], capture_output=True,
                           timeout=timeout, env={**os.environ, "OMP_NUM_THREADS": "1"})
        except subprocess.TimeoutExpired:
            return False
        except Exception:
            return False
        return os.path.exists(verdict) and open(verdict).read() == "pass"


def grade(model, problem, completion):
    return grade_llada(problem, completion) if model == "llada" else grade_nemotron(problem, completion)


def score(model, problem, completion):
    return {"pass@1": bool(grade(model, problem, completion))}
