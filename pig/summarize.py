"""Tabulate result files: one table per (model, geometry), rows sorted by throughput.

    python -m pig.summarize results/llada [results/nemotron ...] [--latex]
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path


def knob(cfg):
    if cfg["rule"] == "pig-int":
        return f"phi={cfg['phi']:g}"
    s = f"theta={cfg['theta']:g}"
    return s + (f" (nu>={cfg['floor']:g})" if cfg.get("floor") is not None else "")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--latex", action="store_true", help="emit LaTeX rows instead of markdown")
    args = ap.parse_args()
    groups = defaultdict(list)
    for d in args.dirs:
        for f in sorted(Path(d).glob("*.json")):
            r = json.loads(f.read_text())
            c, s = r["config"], r["summary"]
            groups[(c["model"], c["geometry"], c["block"])].append((c, s))
    for (model, geom, block), rows in sorted(groups.items()):
        title = f"{model} -- {geom}" + (f" B={block}" if block else "")
        rows.sort(key=lambda t: (t[0]["rule"], t[1]["tpf"]))
        if args.latex:
            print(f"% {title}")
            for c, s in rows:
                lo, hi = s["wilson95"]
                p = s["pass_at_1"]
                print(f"{c['rule'].upper()} & {knob(c)} & {s['tpf']:.2f} & "
                      f"${p:.3f}^{{+{hi - p:.3f}}}_{{-{p - lo:.3f}}}$ \\\\")
            continue
        print(f"\n### {title}\n")
        print("| rule | knob | TPF | pass@1 | Wilson 95% | n |")
        print("|---|---|---|---|---|---|")
        for c, s in rows:
            lo, hi = s["wilson95"]
            print(f"| {c['rule'].upper()} | {knob(c)} | {s['tpf']:.2f} | {s['pass_at_1']:.3f} | "
                  f"[{lo:.3f}, {hi:.3f}] | {s['n']} |")


if __name__ == "__main__":
    main()
