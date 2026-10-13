"""GSM8K (the first 500 test problems, greedy) for the PIG decoders.

Prompts. Each prompt is wrapped in the model's chat template as the user turn:
    LLaDA-8B-Instruct          five-shot in lm-eval's Question:/Answer: format. The five exemplars are
                               drawn once from the GSM8K training split (seed 1234) and rendered as
                               "Question: <q>\\nAnswer: <reasoning>\\n#### <number>", followed by the
                               test question as "Question: <q>\\nAnswer:".
    Nemotron-Labs-Diffusion-3B zero-shot: the question text as is.

Grading (lm-eval's two extractors, both recorded; the flexible one is the primary metric). The
generation is first cut at the stop sequences, then
    strict   the first "#### <number>" in the cut text
    flex     the last number in the cut text
Gold is the number after "####" in the reference answer, without thousands separators.
"""
import re

import numpy as np

NAME = "gsm8k"
N = 500
SHOTS = {"llada": 5, "nemotron": 0}
METRICS = ("flex", "strict")
PRIMARY = "flex"
FEWSHOT_SEED = 1234
STOPS = ("\nQuestion:", "\nQ:", "<|endoftext|>", "<|eot_id|>", "\n\nQuestion")

_fewshot = {}


def load_split(split):
    from datasets import load_dataset

    return load_dataset("openai/gsm8k", "main", split=split)


def load_problems():
    """The GSM8K test problems in order: dicts with task_id, question and answer."""
    return [dict(task_id=f"gsm8k/{i}", question=ex["question"], answer=ex["answer"])
            for i, ex in enumerate(load_split("test"))]


def build_fewshot(train, k=5, seed=FEWSHOT_SEED):
    """k exemplars from `train` (a sequence of dicts with question/answer), chosen once by `seed`."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(train), k, replace=False)
    parts = []
    for i in idx:
        ex = train[int(i)]
        reasoning, number = ex["answer"].split("####")
        parts.append(f"Question: {ex['question']}\nAnswer: {reasoning.strip()}\n#### {number.strip()}")
    return "\n\n".join(parts)


def fewshot(k):
    """The exemplar block for k shots (built once from the training split)."""
    if k and k not in _fewshot:
        _fewshot[k] = build_fewshot(load_split("train"), k)
    return _fewshot.get(k, "")


def prompt_text(model, problem, exemplars=None):
    """The user turn for `problem`: Nemotron sees the question as is; LLaDA sees the five exemplars
    and the question in the Question:/Answer: format (`exemplars` overrides the exemplar block)."""
    k = SHOTS[model]
    if k == 0:
        return problem["question"]
    few = fewshot(k) if exemplars is None else exemplars
    return few + "\n\n" + f"Question: {problem['question']}\nAnswer:"


def cut(text):
    """The generation up to the first stop sequence."""
    for stop in STOPS:
        i = text.find(stop)
        if i != -1:
            text = text[:i]
    return text


def extract(text):
    """-> (strict, flex): the first '#### <number>' and the last number of the cut generation."""
    t = cut(text)
    m = re.search(r"####\s*(-?[\d,]*\.?\d+)", t)
    strict = m.group(1).replace(",", "").rstrip(".") if m else None
    numbers = re.findall(r"-?[\d,]*\.?\d+", t)
    flex = numbers[-1].replace(",", "").rstrip(".") if numbers else None
    return strict, flex


def gold(answer):
    return answer.split("####")[1].strip().replace(",", "")


def score(model, problem, completion):
    strict, flex = extract(completion)
    g = gold(problem["answer"])
    return {"flex": flex == g, "strict": strict == g}
