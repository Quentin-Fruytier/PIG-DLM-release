"""The benchmarks the harness runs. Every task module exposes the same interface:

    NAME             the task's name
    N                the paper's problem count: problems [0, N) of the benchmark's test split
    SHOTS            exemplar count per model
    METRICS          the metrics recorded per problem; PRIMARY is the one reported as the accuracy
    load_problems()  -> the benchmark's problems in order, dicts with a task_id
    prompt_text(model, problem) -> the user turn (the decoder wraps it in the chat template)
    score(model, problem, completion) -> {metric: bool}
"""
from . import gsm8k, humaneval

TASKS = {humaneval.NAME: humaneval, gsm8k.NAME: gsm8k}


def get(name):
    try:
        return TASKS[name]
    except KeyError:
        raise ValueError(f"unknown task {name!r}; expected one of {tuple(TASKS)}") from None
