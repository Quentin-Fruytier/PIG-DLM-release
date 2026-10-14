"""Parity of two builds of PIG: record what the decoder sees and emits, then compare two recordings.

    python -m hub.parity capture --family llada --out parity/llada_a.pt [--n 8] [--device cuda:0] [--matrix]
    python -m hub.parity compare A.pt B.pt

capture: for the first --n HumanEval and the first --n GSM8K prompts (the model's own prompt protocol),
decode with PIG.INT (phi 0.1, fixed blocks of 64 / 32) through a recording rule that stores, per round,
the candidate set and the confidences, entropies and adapter matrix the rule was given, plus the final
token sequence and forward count. Everything the backbone computes feeds those arrays, so two recordings
that agree, round for round, decode identically from the forward pass up.

compare: prints, per prompt, whether tokens and forwards agree and the largest absolute difference in
conf / H / D over the common rounds; exit code 1 if any token sequence or forward count differs.
"""
import argparse
import sys
import time

import numpy as np
import torch


class Recorder:
    def __init__(self, inner):
        self.inner, self.rounds = inner, []

    def __call__(self, rnd):
        self.rounds.append(dict(cand=np.asarray(rnd.cand, dtype=np.int64),
                                conf=np.asarray(rnd.conf, dtype=np.float32).copy(),
                                H=np.asarray(rnd.H, dtype=np.float32).copy(),
                                D=np.asarray(rnd.D, dtype=np.float32).copy()))
        return self.inner(rnd)


def prompts(family, n):
    from experiments.tasks import gsm8k, humaneval
    items = []
    for task in (humaneval, gsm8k):
        for prob in task.load_problems()[:n]:
            items.append((f"{task.NAME}:{prob['task_id']}", task.prompt_text(family, prob)))
    return items


# --matrix: every geometry of the paper's tables x four operating points (one per lane, plus an unfloored PIG.EXT)
MATRIX = {
    "llada": dict(geometries=[("full", 0), ("block", 32), ("block", 64), ("sliding", 32), ("sliding", 64)],
                  modes=[dict(mode="pig.int", phi=0.1), dict(mode="pig.ext", theta=0.25, floor=0.8),
                         dict(mode="pig.ext", theta=1.0), dict(mode="confidence", nu=0.85)]),
    "nemotron": dict(geometries=[("block", 32), ("sliding", 32), ("nested", 32), ("block", 64), ("nested", 16)],
                     modes=[dict(mode="pig.int", phi=0.1), dict(mode="pig.ext", theta=0.25, floor=0.8),
                            dict(mode="pig.ext", theta=1.0), dict(mode="confidence", nu=0.9)]),
}


def capture(a):
    import pig
    import transformers
    model = pig.load(a.family, device=a.device)
    block = 64 if a.family == "llada" else 32
    if a.matrix:
        cases = [((g, b), m) for (g, b) in MATRIX[a.family]["geometries"] for m in MATRIX[a.family]["modes"]]
    else:
        cases = [(("block", block), dict(mode="pig.int", phi=0.1))]
    rec = dict(family=a.family, transformers=transformers.__version__, torch=torch.__version__,
               backbone=f"{type(model.backbone).__module__}.{type(model.backbone).__name__}",
               gpu=torch.cuda.get_device_name(a.device) if a.device.startswith("cuda") else "cpu", prompts={})
    items = prompts(a.family, a.n)
    for (geom, size), mode in cases:
        tag = f"{geom}{size}/" + "/".join(f"{k}={v}" for k, v in mode.items())
        for name, text in items:
            key = name if not a.matrix else f"{tag}|{name}"
            rule = Recorder(model.rule(**mode))
            ids = model.encode(text)
            t0 = time.perf_counter()
            if a.family == "llada":
                tokens, forwards, _ = model.decoder.generate(ids, rule, model.geometry(geom, size), canvas=256)
            else:
                tokens, forwards, _ = model.decoder.generate(ids, rule, model.geometry(geom, size), max_new=2048)
            rec["prompts"][key] = dict(tokens=[int(t) for t in tokens], forwards=int(forwards), rounds=rule.rounds)
            print(f"[{key}] {len(tokens)} tok / {forwards} fwd / {len(rule.rounds)} rounds ({time.perf_counter() - t0:.1f}s)", flush=True)
    torch.save(rec, a.out)
    print(f"wrote {a.out}: {len(rec['prompts'])} prompts, transformers {rec['transformers']}, backbone {rec['backbone']}")


def compare(a):
    A, B = torch.load(a.a, weights_only=False), torch.load(a.b, weights_only=False)
    print(f"A: transformers {A['transformers']}, {A['backbone']}, {A['gpu']}\nB: transformers {B['transformers']}, {B['backbone']}, {B['gpu']}")
    bad = 0
    worst = dict(conf=0.0, H=0.0, D=0.0)
    for name in A["prompts"]:
        if name not in B["prompts"]:
            print(f"[{name}] missing in B"); bad += 1; continue
        pa, pb = A["prompts"][name], B["prompts"][name]
        same = pa["tokens"] == pb["tokens"] and pa["forwards"] == pb["forwards"]
        bad += 0 if same else 1
        diffs = dict(conf=0.0, H=0.0, D=0.0)
        n = min(len(pa["rounds"]), len(pb["rounds"]))
        for ra, rb in zip(pa["rounds"][:n], pb["rounds"][:n]):
            if ra["cand"].shape != rb["cand"].shape or not np.array_equal(ra["cand"], rb["cand"]):
                diffs = {k: float("inf") for k in diffs}; break
            for k in diffs:
                if ra[k].shape == rb[k].shape:
                    diffs[k] = max(diffs[k], float(np.abs(ra[k] - rb[k]).max()))
                else:
                    diffs[k] = float("inf")
        for k in worst:
            worst[k] = max(worst[k], diffs[k])
        print(f"[{name}] tokens {'same' if pa['tokens'] == pb['tokens'] else 'DIFFER'}, forwards {pa['forwards']} vs {pb['forwards']}, "
              f"rounds {len(pa['rounds'])} vs {len(pb['rounds'])}; max|d| conf {diffs['conf']:.2e} H {diffs['H']:.2e} D {diffs['D']:.2e}")
    print(f"[done] {len(A['prompts'])} prompts, {bad} with different tokens or forwards; worst max|d| conf {worst['conf']:.2e} H {worst['H']:.2e} D {worst['D']:.2e}")
    sys.exit(1 if bad else 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture"); c.add_argument("--family", required=True); c.add_argument("--out", required=True)
    c.add_argument("--n", type=int, default=8); c.add_argument("--device", default="cuda")
    c.add_argument("--matrix", action="store_true", help="every geometry x four operating points (see MATRIX)")
    d = sub.add_parser("compare"); d.add_argument("a"); d.add_argument("b")
    a = ap.parse_args()
    capture(a) if a.cmd == "capture" else compare(a)


if __name__ == "__main__":
    main()
