"""Run one operating point on a benchmark and write one result file.

    # LLaDA-8B-Instruct, HumanEval, PIG.INT with phi = 0.1, fixed 64-token blocks
    python -m experiments.evaluate --task humaneval --model llada --rule pig.int --phi 0.1 \
        --geometry block --block 64 --out results/llada/humaneval/llada_humaneval_block64_pig.int_phi0.1.json
    # Nemotron-Labs-Diffusion-3B, GSM8K, PIG.EXT with theta = 0.25 and a 0.8 confidence floor
    python -m experiments.evaluate --task gsm8k --model nemotron --rule pig.ext --theta 0.25 --floor 0.8 \
        --geometry block --block 32 --out results/nemotron/gsm8k/nemotron_gsm8k_block32_pig.ext_theta0.25_floor0.8.json
    # the confidence-threshold baseline (Fast-dLLM's rule) at nu = 0.9
    python -m experiments.evaluate --task gsm8k --model llada --rule confidence --nu 0.9 \
        --geometry block --block 64 --out results/llada/gsm8k/llada_gsm8k_block64_confidence_nu0.9.json

A whole grid (the model is loaded once): python -m experiments.sweep experiments/configs/gsm8k_llada.json
"""
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np

from pig import __version__
from pig.api import MODES, make_rule as _make_rule
from pig.assets import ASSETS, asset_paths
from pig.loading import load_decoder, load_tables, make_geometry  # noqa: F401  (re-exported for the sweep)

from . import tasks as task_registry
from .stats import wilson

MODELS = ("llada", "nemotron")
RULES = MODES                                   # "pig.int" | "pig.ext" | "confidence"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_rule(model, rule, phi=None, theta=None, floor=None, window=16, nu=None, assets=ASSETS):
    """The rule object for one operating point (`window` applies to pig.ext only); no defaults are filled in."""
    if rule not in RULES:
        raise ValueError(f"unknown rule {rule!r}; expected one of {RULES}")
    tables = load_tables(model, assets) if rule != "confidence" else {}
    return _make_rule(tables, rule, phi=phi, theta=theta, floor=floor, nu=nu,
                      window=window if rule == "pig.ext" else None)


def cell_name(model, task, rule, geometry, phi=None, theta=None, floor=None, nu=None):
    if rule == "pig.int":
        knob = f"phi{phi:g}"
    elif rule == "pig.ext":
        knob = f"theta{theta:g}" + (f"_floor{floor:g}" if floor is not None else "")
    else:
        knob = f"nu{nu:g}"
    return f"{model}_{task}_{geometry.tag()}_{rule}_{knob}"


def run_cell(decoder, model, task, rule, geometry, spec, problems, out_path, assets=ASSETS,
             log_every=20):
    """Decode + grade every problem in `problems` on `task`; write and return the result dict.

    task: a task name (see experiments/tasks) or the task module itself.
    spec: dict(rule=..., phi=..., theta=..., floor=..., nu=..., window=...) as passed to make_rule.
    """
    import torch
    import transformers

    task = task_registry.get(task) if isinstance(task, str) else task
    records, t_start = [], time.perf_counter()
    timings = []
    for i, prob in enumerate(problems):
        ids = decoder.chat_prompt(task.prompt_text(model, prob))
        gen, forwards, ms = decoder.generate(ids, rule, geometry)
        text = decoder.tokenizer.decode(gen, skip_special_tokens=True)
        scores = {m: bool(v) for m, v in task.score(model, prob, text).items()}
        records.append(dict(task_id=prob["task_id"], passed=scores[task.PRIMARY], scores=scores,
                            forwards=int(forwards), n_gen=int(len(gen)), completion=text))
        timings.append(ms)
        if log_every and (i + 1) % log_every == 0:
            acc = np.mean([r["passed"] for r in records])
            print(f"[{out_path.stem}] {i + 1}/{len(problems)} {task.PRIMARY} {acc:.3f}", flush=True)
    n = len(records)
    metrics = {}
    for m in task.METRICS:
        k = sum(r["scores"][m] for r in records)
        lo, hi = wilson(k, n)
        metrics[m] = dict(passed=int(k), accuracy=k / n if n else 0.0, wilson95=[lo, hi])
    primary = metrics[task.PRIMARY]
    fwd = np.array([r["forwards"] for r in records], dtype=np.float64)
    if model == "llada":
        tpf = float(n and 256.0 / fwd.mean())
        tpf_def = "canvas length (256) / mean forward passes per problem"
    else:
        tpf = float(np.mean([r["n_gen"] / max(r["forwards"], 1) for r in records])) if n else 0.0
        tpf_def = "mean over problems of generated tokens (to first EOS) / forward passes"
    mean_ms = {key: float(np.mean([t[key] for t in timings])) for key in timings[0]} if timings else {}
    paths = asset_paths(model, assets)
    used = [paths["fde"]]
    if spec["rule"] == "pig.int":
        used.append(paths["confidence"])
    elif spec["rule"] == "pig.ext":
        used.append(paths["entropy"])
    result = dict(
        config=dict(model=model, model_id=decoder.model_id, revision=decoder.revision,
                    task=task.NAME, shots=task.SHOTS[model],
                    rule=spec["rule"], phi=spec.get("phi"), theta=spec.get("theta"),
                    floor=spec.get("floor"), nu=spec.get("nu"), window=spec.get("window", 16),
                    geometry=geometry.kind, block=geometry.size,
                    problems=[problems[0]["task_id"], problems[-1]["task_id"]] if problems else [],
                    assets={f"{model}/{p.name}": sha256(p) for p in used},
                    pig=__version__, torch=torch.__version__, transformers=transformers.__version__,
                    gpu=(torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"),
                    python=platform.python_version()),
        summary=dict(n=n, metric=task.PRIMARY, passed=primary["passed"], accuracy=primary["accuracy"],
                     wilson95=primary["wilson95"], metrics=metrics,
                     tpf=tpf, tpf_definition=tpf_def, mean_forwards=float(fwd.mean()) if n else 0.0,
                     ms_per_round=mean_ms, wall_seconds=time.perf_counter() - t_start),
        problems=records)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=1))
    tmp.replace(out_path)
    lo, hi = primary["wilson95"]
    print(f"[{out_path.stem}] {task.PRIMARY} {primary['accuracy']:.4f} [{lo:.3f}, {hi:.3f}]  "
          f"TPF {tpf:.2f}  (n={n})", flush=True)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=tuple(task_registry.TASKS), default="humaneval")
    ap.add_argument("--model", choices=MODELS, required=True)
    ap.add_argument("--rule", choices=RULES, required=True)
    ap.add_argument("--phi", type=float, help="PIG.INT single-round regret budget")
    ap.add_argument("--theta", type=float, help="PIG.EXT cross-round regret budget")
    ap.add_argument("--floor", type=float, help="PIG.EXT confidence floor on admission (optional)")
    ap.add_argument("--nu", type=float, help="confidence threshold of the baseline rule")
    ap.add_argument("--window", type=int, default=16, help="PIG.EXT look-ahead size |N(S)|")
    ap.add_argument("--geometry", required=True,
                    help="llada: full | block | sliding;  nemotron: block | sliding | nested")
    ap.add_argument("--block", type=int, default=0, help="block / window length B")
    ap.add_argument("--start", type=int, default=0, help="first problem index")
    ap.add_argument("--n", type=int, help="number of problems (default: the paper's count for the task)")
    ap.add_argument("--assets", default=str(ASSETS))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    task = task_registry.get(args.task)
    geometry = make_geometry(args.model, args.geometry, args.block)
    spec = dict(rule=args.rule, phi=args.phi, theta=args.theta, floor=args.floor, nu=args.nu, window=args.window)
    rule = make_rule(args.model, assets=args.assets, **spec)
    n = task.N if args.n is None else args.n
    problems = task.load_problems()[args.start:args.start + n]
    decoder = load_decoder(args.model, args.assets, args.device)
    run_cell(decoder, args.model, task, rule, geometry, spec, problems, Path(args.out), assets=args.assets)


if __name__ == "__main__":
    main()
