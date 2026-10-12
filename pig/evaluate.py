"""Run PIG.INT / PIG.EXT on HumanEval and write one result file per operating point.

Single operating point:
    python -m pig.evaluate --model llada --rule pig-int --phi 0.1 --geometry block --block 64 \
        --out results/llada/block64_pig-int_phi0.1.json
    python -m pig.evaluate --model nemotron --rule pig-ext --theta 0.25 --floor 0.8 \
        --geometry block --block 32 --out results/nemotron/block32_pig-ext_theta0.25_floor0.8.json

A whole grid (the model is loaded once): python -m pig.sweep configs/humaneval_llada.json
"""
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np

from . import __version__
from .humaneval import grade, load_problems, prompt_text, wilson
from .rules import PIGExt, PIGInt
from .tables import FlipTable

from .assets import ASSETS, asset_paths  # noqa: F401  (asset_paths is part of this module's API)

MODELS = ("llada", "nemotron")
RULES = ("pig-int", "pig-ext")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_decoder(model, assets=ASSETS, device="cuda"):
    fde = asset_paths(model, assets)["fde"]
    if model == "llada":
        from .llada import LLaDADecoder
        return LLaDADecoder(fde, device=device)
    from .nemotron import NemotronDecoder
    return NemotronDecoder(fde, device=device)


def make_geometry(model, kind, size):
    if model == "llada":
        from .llada import Geometry
        return Geometry(kind, size)
    from .nemotron import Window
    return Window(kind, size)


def make_rule(model, rule, phi=None, theta=None, floor=None, window=16, assets=ASSETS):
    paths = asset_paths(model, assets)
    if rule == "pig-int":
        if phi is None or theta is not None or floor is not None:
            raise ValueError("pig-int takes --phi only")
        return PIGInt(FlipTable.load(paths["confidence"]), phi)
    if rule == "pig-ext":
        if theta is None or phi is not None:
            raise ValueError("pig-ext takes --theta (and optionally --floor)")
        return PIGExt(FlipTable.load(paths["entropy"]), theta, window=window, floor=floor)
    raise ValueError(f"unknown rule {rule!r}")


def cell_name(model, rule, geometry, phi=None, theta=None, floor=None):
    knob = f"phi{phi:g}" if rule == "pig-int" else f"theta{theta:g}"
    if floor is not None:
        knob += f"_floor{floor:g}"
    return f"{model}_{geometry.tag()}_{rule}_{knob}"


def run_cell(decoder, model, rule, geometry, spec, problems, out_path, assets=ASSETS,
             log_every=20):
    """Decode + grade every problem in `problems`; write and return the result dict.

    spec: dict(rule=..., phi=..., theta=..., floor=..., window=...) as passed to make_rule.
    """
    import torch
    import transformers

    records, t_start = [], time.perf_counter()
    timings = []
    for i, prob in enumerate(problems):
        ids = decoder.chat_prompt(prompt_text(model, prob))
        gen, forwards, ms = decoder.generate(ids, rule, geometry)
        n_gen = len(gen)
        text = decoder.tokenizer.decode(gen, skip_special_tokens=True)
        passed = bool(grade(model, prob, text))
        records.append(dict(task_id=prob["task_id"], passed=passed, forwards=int(forwards),
                            n_gen=int(n_gen), completion=text))
        timings.append(ms)
        if log_every and (i + 1) % log_every == 0:
            acc = np.mean([r["passed"] for r in records])
            print(f"[{out_path.stem}] {i + 1}/{len(problems)} pass@1 {acc:.3f}", flush=True)
    n = len(records)
    k = sum(r["passed"] for r in records)
    lo, hi = wilson(k, n)
    fwd = np.array([r["forwards"] for r in records], dtype=np.float64)
    if model == "llada":
        tpf = float(len(records) and 256.0 / fwd.mean())
        tpf_def = "canvas length (256) / mean forward passes per problem"
    else:
        tpf = float(np.mean([r["n_gen"] / max(r["forwards"], 1) for r in records]))
        tpf_def = "mean over problems of generated tokens (to first EOS) / forward passes"
    mean_ms = {key: float(np.mean([t[key] for t in timings])) for key in timings[0]} if timings else {}
    paths = asset_paths(model, assets)
    used = [paths["fde"], paths["confidence" if spec["rule"] == "pig-int" else "entropy"]]
    result = dict(
        config=dict(model=model, model_id=decoder.model_id, revision=decoder.revision,
                    rule=spec["rule"], phi=spec.get("phi"), theta=spec.get("theta"),
                    floor=spec.get("floor"), window=spec.get("window", 16),
                    geometry=geometry.kind, block=geometry.size,
                    problems=[problems[0]["task_id"], problems[-1]["task_id"]] if problems else [],
                    assets={f"{model}/{p.name}": sha256(p) for p in used},
                    pig=__version__, torch=torch.__version__, transformers=transformers.__version__,
                    gpu=(torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"),
                    python=platform.python_version()),
        summary=dict(n=n, passed=int(k), pass_at_1=k / n if n else 0.0, wilson95=[lo, hi],
                     tpf=tpf, tpf_definition=tpf_def, mean_forwards=float(fwd.mean()) if n else 0.0,
                     ms_per_round=mean_ms, wall_seconds=time.perf_counter() - t_start),
        problems=records)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=1))
    tmp.replace(out_path)
    print(f"[{out_path.stem}] pass@1 {result['summary']['pass_at_1']:.4f} "
          f"[{lo:.3f}, {hi:.3f}]  TPF {tpf:.2f}  (n={n})", flush=True)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=MODELS, required=True)
    ap.add_argument("--rule", choices=RULES, required=True)
    ap.add_argument("--phi", type=float, help="PIG.INT single-round regret budget")
    ap.add_argument("--theta", type=float, help="PIG.EXT cross-round regret budget")
    ap.add_argument("--floor", type=float, help="PIG.EXT confidence floor on admission (optional)")
    ap.add_argument("--window", type=int, default=16, help="PIG.EXT look-ahead size |N(S)|")
    ap.add_argument("--geometry", required=True,
                    help="llada: full | block | sliding;  nemotron: block | sliding | nested")
    ap.add_argument("--block", type=int, default=0, help="block / window length B")
    ap.add_argument("--start", type=int, default=0, help="first problem index")
    ap.add_argument("--n", type=int, default=164, help="number of problems")
    ap.add_argument("--assets", default=str(ASSETS))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    geometry = make_geometry(args.model, args.geometry, args.block)
    spec = dict(rule=args.rule, phi=args.phi, theta=args.theta, floor=args.floor, window=args.window)
    rule = make_rule(args.model, assets=args.assets, **spec)
    problems = load_problems()[args.start:args.start + args.n]
    decoder = load_decoder(args.model, args.assets, args.device)
    run_cell(decoder, args.model, rule, geometry, spec, problems, Path(args.out), assets=args.assets)


if __name__ == "__main__":
    main()
