"""Tabulate result files: one table per (model, task, geometry), rows sorted by rule and throughput.

    python -m pig.summarize results/llada/gsm8k [results/nemotron/humaneval ...] [--latex]
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

RULE_ORDER = {"confidence": 0, "pig-int": 1, "pig-ext": 2}


def knob(cfg):
    if cfg["rule"] == "pig-int":
        return f"phi={cfg['phi']:g}"
    if cfg["rule"] == "pig-ext":
        s = f"theta={cfg['theta']:g}"
        return s + (f" (nu>={cfg['floor']:g})" if cfg.get("floor") is not None else "")
    return f"nu={cfg['nu']:g}"


def load(dirs):
    groups = defaultdict(list)
    for d in dirs:
        for f in sorted(Path(d).glob("*.json")):
            r = json.loads(f.read_text())
            c, s = r["config"], r["summary"]
            groups[(c["model"], c.get("task", "humaneval"), c["geometry"], c["block"])].append((c, s))
    for rows in groups.values():
        rows.sort(key=lambda t: (RULE_ORDER.get(t[0]["rule"], 9), t[1]["tpf"]))
    return groups


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--latex", action="store_true", help="emit LaTeX rows instead of markdown")
    args = ap.parse_args()
    for (model, task, geom, block), rows in sorted(load(args.dirs).items()):
        title = f"{model} -- {task} -- {geom}" + (f" B={block}" if block else "")
        metric = rows[0][1].get("metric", "pass@1")
        extras = [m for m in rows[0][1].get("metrics", {}) if m != metric]
        if args.latex:
            print(f"% {title}: rule & knob & TPF & {metric}" + "".join(f" & {m}" for m in extras))
            for c, s in rows:
                cells = []
                for m in [metric] + extras:
                    mm = s["metrics"][m] if "metrics" in s else dict(accuracy=s.get("accuracy", s.get("pass_at_1")), wilson95=s["wilson95"])
                    p, (lo, hi) = mm["accuracy"], mm["wilson95"]
                    cells.append(f"${p:.3f}^{{+{hi - p:.3f}}}_{{-{p - lo:.3f}}}$")
                print(f"{c['rule'].upper()} & {knob(c)} & {s['tpf']:.2f} & " + " & ".join(cells) + " \\\\")
            continue
        print(f"\n### {title}\n")
        print(f"| rule | knob | TPF | {metric} | Wilson 95% |" + "".join(f" {m} |" for m in extras) + " n |")
        print("|---|---|---|---|---|" + "---|" * len(extras) + "---|")
        for c, s in rows:
            acc = s.get("accuracy", s.get("pass_at_1"))
            lo, hi = s["wilson95"]
            extra = "".join(f" {s['metrics'][m]['accuracy']:.3f} |" for m in extras)
            print(f"| {c['rule'].upper()} | {knob(c)} | {s['tpf']:.2f} | {acc:.3f} | [{lo:.3f}, {hi:.3f}] |{extra} {s['n']} |")


if __name__ == "__main__":
    main()
