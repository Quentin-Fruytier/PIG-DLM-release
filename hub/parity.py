"""Parity of two builds of PIG: record what the decoder sees and emits, then compare two recordings.

    python -m hub.parity capture --family llada --out /ssd1/.../llada_main.pt [--n 8] [--device cuda:0]
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


def capture(a):
    import pig
    import transformers
    model = pig.load(a.family, device=a.device)
    block = 64 if a.family == "llada" else 32
    rec = dict(family=a.family, transformers=transformers.__version__, torch=torch.__version__,
               backbone=f"{type(model.backbone).__module__}.{type(model.backbone).__name__}",
               gpu=torch.cuda.get_device_name(a.device) if a.device.startswith("cuda") else "cpu", prompts={})
    for name, text in prompts(a.family, a.n):
        rule = Recorder(model.rule("pig.int", phi=0.1))
        ids = model.encode(text)
        t0 = time.perf_counter()
        if a.family == "llada":
            tokens, forwards, _ = model.decoder.generate(ids, rule, model.geometry("block", block), canvas=256)
        else:
            tokens, forwards, _ = model.decoder.generate(ids, rule, model.geometry("block", block), max_new=2048)
        rec["prompts"][name] = dict(tokens=[int(t) for t in tokens], forwards=int(forwards), rounds=rule.rounds)
        print(f"[{name}] {len(tokens)} tok / {forwards} fwd / {len(rule.rounds)} rounds ({time.perf_counter() - t0:.1f}s)", flush=True)
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
    d = sub.add_parser("compare"); d.add_argument("a"); d.add_argument("b")
    a = ap.parse_args()
    capture(a) if a.cmd == "capture" else compare(a)


if __name__ == "__main__":
    main()
