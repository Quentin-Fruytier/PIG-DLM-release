"""Accuracy-throughput frontiers from result files: one panel per (model, task, geometry), three lanes.

    python -m experiments.plot results/llada/gsm8k [results/llada/humaneval ...] --out frontiers.png
"""
import argparse
import math

from .summarize import load

LANES = (("confidence", "Fast-dLLM", "o"), ("pig.int", "PIG.INT", "D"), ("pig.ext", "PIG.EXT", "^"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", default="frontiers.png")
    ap.add_argument("--cols", type=int, default=3)
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = sorted(load(args.dirs).items())
    if not groups:
        raise SystemExit("no result files found")
    cols = min(args.cols, len(groups))
    rows_n = math.ceil(len(groups) / cols)
    fig, axes = plt.subplots(rows_n, cols, figsize=(4.2 * cols, 3.4 * rows_n), squeeze=False)
    for ax, ((model, task, geom, block), rows) in zip(axes.flat, groups):
        metric = rows[0][1].get("metric", "pass@1")
        for rule, label, marker in LANES:
            pts = sorted(((s["tpf"], s.get("accuracy", s.get("pass_at_1")), s["wilson95"]) for c, s in rows
                          if c["rule"] == rule), key=lambda t: t[0])
            if not pts:
                continue
            x = [p[0] for p in pts]
            y = [p[1] for p in pts]
            lo = [p[1] - p[2][0] for p in pts]
            hi = [p[2][1] - p[1] for p in pts]
            ax.errorbar(x, y, yerr=[lo, hi], marker=marker, capsize=2, label=label)
        ax.set_title(f"{model} / {task} / {geom}" + (f" B={block}" if block else ""), fontsize=10)
        ax.set_xlabel("tokens per forward")
        ax.set_ylabel(metric)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    for ax in list(axes.flat)[len(groups):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}: {len(groups)} panel(s)")


if __name__ == "__main__":
    main()
