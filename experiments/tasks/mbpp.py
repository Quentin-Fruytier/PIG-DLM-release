"""MBPP (the classic test split: task ids 11-510, greedy, pass@1) for the PIG decoders.

Prompts. Each prompt is wrapped in the model's chat template as the user turn:
    LLaDA-8B-Instruct          three-shot in the Austin et al. format: the prompt-split exemplars with
                               task ids 2, 3, 4 rendered as "You are an expert Python programmer, and here
                               is your task: <text>\\nYour code should pass these tests:\\n\\n<tests>\\n
                               [BEGIN]\\n<code>\\n[DONE]", then the task in the same format ending with
                               "[BEGIN]\\n".
    Nemotron-Labs-Diffusion-3B zero-shot: "You are an expert Python programmer, and here is your task:
                               <text> Your code should pass these tests:\\n<tests>\\n".

Grading (execution-based pass@1). The code is the fenced block if present, else the [BEGIN]..[DONE] span,
else the raw text; the longest line-prefix that parses as Python is kept and the model's own top-level
asserts are dropped; then the task's setup code and its three asserts run in a fresh interpreter with a
10 s timeout. Pass iff every assert holds.
"""
import os
import re
import subprocess
import sys
import tempfile

NAME = "mbpp"
N = {"llada": 500, "nemotron": 499}
SHOTS = {"llada": 3, "nemotron": 0}
METRICS = ("pass@1",)
PRIMARY = "pass@1"
FEWSHOT_IDS = (2, 3, 4)          # Austin et al. convention: exemplars from the prompt split

_fewshot = {}


def load_dataset_mbpp():
    from datasets import load_dataset

    return load_dataset("mbpp", "full")


def load_problems():
    """The 500 classic test problems in order: dicts with task_id, text, test_list, test_setup_code."""
    return [dict(task_id=f"mbpp/{ex['task_id']}", text=ex["text"], test_list=list(ex["test_list"]),
                 test_setup_code=ex.get("test_setup_code", "") or "")
            for ex in load_dataset_mbpp()["test"]]


def build_fewshot(prompt_split):
    parts = []
    for tid in FEWSHOT_IDS:
        ex = next(x for x in prompt_split if x["task_id"] == tid)
        tests = "\n".join(ex["test_list"])
        parts.append(f"You are an expert Python programmer, and here is your task: "
                     f"{ex['text']}\nYour code should pass these tests:\n\n{tests}\n"
                     f"[BEGIN]\n{ex['code']}\n[DONE]")
    return "\n\n".join(parts)


def fewshot():
    if "block" not in _fewshot:
        _fewshot["block"] = build_fewshot(list(load_dataset_mbpp()["prompt"]))
    return _fewshot["block"]


def prompt_text(model, problem, exemplars=None):
    tests = "\n".join(problem["test_list"])
    if SHOTS[model] == 0:
        return ("You are an expert Python programmer, and here is your task: " + problem["text"]
                + " Your code should pass these tests:\n" + tests + "\n")
    few = fewshot() if exemplars is None else exemplars
    return (few + "\n\n" + f"You are an expert Python programmer, and here is your task: {problem['text']}\n"
            f"Your code should pass these tests:\n\n{tests}\n[BEGIN]\n")


def extract_code(gen):
    """Fenced block if present, else the [BEGIN]..[DONE] span, else the raw text; then the longest
    line-prefix that parses, without the model's own top-level asserts."""
    import ast
    m = re.search(r"```(?:python|py)?\s*\n(.*?)(?:```|\Z)", gen, re.S)
    body = m.group(1) if m else gen
    b = body.find("[BEGIN]")
    if b != -1:
        body = body[b + len("[BEGIN]"):]
    for stop in ("[DONE]", "\nYou are an expert", "\n[BEGIN]"):
        i = body.find(stop)
        if i != -1:
            body = body[:i]
    lines = body.strip("\n").split("\n")
    MAX_LINES, MAX_CHARS = 400, 60000            # an ungradeable generation scores 0, never aborts the run
    if len(lines) > MAX_LINES:
        lines = lines[:MAX_LINES]

    def _safe_parse(src):
        if len(src) > MAX_CHARS:
            return None
        try:
            return ast.parse(src)
        except (SyntaxError, MemoryError, ValueError, RecursionError):
            return None

    best = ""
    for j in range(len(lines), 0, -1):
        cand = "\n".join(lines[:j])
        if _safe_parse(cand) is not None:
            best = cand
            break
    if not best:
        return body.strip("\n")
    tree = _safe_parse(best)
    if tree is not None:
        drop = {n.lineno for n in tree.body if isinstance(n, ast.Assert)}
        if drop:
            kept = [ln for i, ln in enumerate(best.split("\n"), 1) if i not in drop]
            best = "\n".join(kept)
    return best.strip("\n")


def run_tests(code, test_list, setup="", timeout=10):
    src = code + "\n\n" + (setup + "\n" if setup else "") + "\n".join(test_list) + "\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(src)
        path = f.name
    try:
        r = subprocess.run([sys.executable, path], capture_output=True, timeout=timeout,
                           env={**os.environ, "OMP_NUM_THREADS": "1", "PYTHONHASHSEED": "0"})   # repeatable set order
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


def score(model, problem, completion):
    ok = run_tests(extract_code(completion), problem["test_list"], problem["test_setup_code"])
    return {"pass@1": bool(ok)}
