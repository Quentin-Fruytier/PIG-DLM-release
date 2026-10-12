"""Check the Hub wrapper against the evaluation harness on a GPU.

    python -m hub.verify --family llada --n 5 --device cuda:1

Loads the built repository (hub/dist/<name>, or --repo for the Hub copy) with
AutoModel.from_pretrained(..., trust_remote_code=True), and the harness decoder from assets/ as an
independent reference, and decodes the first --n HumanEval problems with both under the same rules.
Greedy decoding is deterministic, so the two must return the same token ids and the same number of
forward passes; any difference is reported and the exit code is 1.
"""
import argparse
import sys
import time
from pathlib import Path

import torch

from hub.build import DIST, REPOS, build
from pig.evaluate import load_decoder, make_geometry, make_rule
from pig.humaneval import load_problems, prompt_text

CASES = {
    "llada": [dict(mode="pig.int", phi=0.1, geometry="block", block=64),
              dict(mode="pig.ext", theta=0.25, floor=0.8, geometry="block", block=64),
              dict(mode="confidence", nu=0.9, geometry="block", block=64)],
    "nemotron": [dict(mode="pig.int", phi=0.1, geometry="block", block=32),
                 dict(mode="pig.ext", theta=0.25, floor=0.8, geometry="block", block=32),
                 dict(mode="confidence", nu=0.9, geometry="block", block=32)],
}


def harness_rule(model, family, case):
    if case["mode"] == "confidence":
        from pig.rules import ConfidenceThreshold
        return ConfidenceThreshold(case["nu"])
    rule = "pig-int" if case["mode"] == "pig.int" else "pig-ext"
    return make_rule(family, rule, phi=case.get("phi"), theta=case.get("theta"), floor=case.get("floor"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", choices=sorted(REPOS), required=True)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--repo", help="load this Hub id instead of the local build")
    ap.add_argument("--no-build", action="store_true")
    a = ap.parse_args()
    from transformers import AutoModel

    src = a.repo or str(build(a.family) if not a.no_build else DIST / REPOS[a.family]["name"])
    t0 = time.perf_counter()
    model = AutoModel.from_pretrained(src, trust_remote_code=True, device=a.device)
    print(f"[wrapper] {src} loaded in {time.perf_counter() - t0:.0f}s on {model.device}; "
          f"backbone {model.config.backbone}@{model.config.backbone_revision[:8]}", flush=True)
    ref = load_decoder(a.family, device=a.device)
    print("[harness] reference decoder loaded from assets/", flush=True)

    # the adapter weights must be the harness's, tensor for tensor
    ws, wr = model.fde.state_dict(), ref.fde.state_dict()
    assert ws.keys() == wr.keys() and all(torch.equal(ws[k].cpu(), wr[k].cpu()) for k in ws), "adapter weights differ"
    print("[check] adapter weights identical to assets/", flush=True)

    problems = load_problems()[:a.n]
    failures = 0
    for case in CASES[a.family]:
        geom = make_geometry(a.family, case["geometry"], case["block"])
        rrule = harness_rule(model, a.family, case)
        for prob in problems:
            text = prompt_text(a.family, prob)
            ids = ref.chat_prompt(text)
            t0 = time.perf_counter()
            out = model.generate(text, **case)
            t1 = time.perf_counter()
            r_tokens, r_fwd, _ = ref.generate(ids, rrule, geom)
            t2 = time.perf_counter()
            same = (list(out.tokens) == list(r_tokens)) and out.forwards == r_fwd
            failures += 0 if same else 1
            print(f"[{case['mode']}] {prob['task_id']}: wrapper {len(out.tokens)} tok / {out.forwards} fwd "
                  f"(TPF {out.tpf:.2f}, {t1 - t0:.1f}s) | harness {len(r_tokens)} tok / {r_fwd} fwd "
                  f"({t2 - t1:.1f}s) -> {'MATCH' if same else 'MISMATCH'}", flush=True)
    print(f"[done] {a.family}: {len(CASES[a.family]) * len(problems)} comparisons, {failures} mismatches")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
