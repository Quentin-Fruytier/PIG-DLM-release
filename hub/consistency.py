"""Compare two campaign trees cell by cell, and both against the paper's frontier tables.

    python -m hub.consistency --a campaign/main --b campaign/branch [--tables ../paper/tables]

For every result file present in both trees (same relative path): the share of problems whose completion
text and forward count agree, each tree's accuracy and TPF, and the paper's cell (accuracy with its Wilson
interval, TPF) when --tables is given. The exit code is 1 if any cell has a differing problem.
"""
import argparse
import json
import re
import sys
from pathlib import Path

TABLE_REGIME = {("block", 32): "BL=32 fixed", ("block", 64): "BL=64 fixed", ("sliding", 32): "BL=32 sliding",
                ("sliding", 64): "BL=64 sliding", ("full", 0): "free canvas", ("nested", 32): "BL=32 in 64-wide window"}


def paper_cell(tables, model, task, geometry, block, rule, knob):
    """-> (accuracy, lo, hi, tpf) of the paper's cell, or None."""
    f = Path(tables) / f"{model}_{task}.tex"
    if not f.exists():
        return None
    tex = f.read_text()
    body = tex[tex.index("\\midrule"):tex.index("\\bottomrule")]
    if (geometry, block) not in TABLE_REGIME:
        return None                                    # a geometry the paper's tables do not report
    regime, want = None, TABLE_REGIME[(geometry, block)]
    lane = {"confidence": 0, "pig.int": 1, "pig.ext": 2}[rule]
    for row in body.split("\\\\"):
        row = row.replace("\\midrule", "").strip()
        if not row:
            continue
        cells = [c.strip() for c in row.split("&")]
        if cells[0]:
            regime = cells[0]
        if regime != want:
            continue
        ncol = (len(cells) - 1) // 3
        base = 1 + lane * ncol
        k = cells[base]
        if not k:
            continue
        kv = float(re.match(r"[0-9.]+", k).group(0))
        if abs(kv - knob) > 1e-9:
            continue
        tpf = float(cells[base + 1])
        m = re.match(r"\$([0-9.]+)\^\{\+([0-9.]+)\}_\{-([0-9.]+)\}\$", cells[base + 2])
        acc, up, dn = float(m.group(1)), float(m.group(2)), float(m.group(3))
        return acc, acc - dn, acc + up, tpf
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
    ap.add_argument("--tables", help="the paper's tables/ directory")
    args = ap.parse_args()
    A, B = Path(args.a), Path(args.b)
    files = sorted(p.relative_to(A) for p in A.rglob("*.json"))
    bad = 0
    print("| model | task | geometry | lane | knob | n | same text | same fwd | same score | acc A | acc B | TPF A | TPF B | paper acc [Wilson] | paper TPF |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    grading_only = []
    for rel in files:
        if not (B / rel).exists():
            print(f"| {rel} | missing in B |"); continue
        ra, rb = json.loads((A / rel).read_text()), json.loads((B / rel).read_text())
        c, sa, sb = ra["config"], ra["summary"], rb["summary"]
        pa = {p["task_id"]: p for p in ra["problems"]}; pb = {p["task_id"]: p for p in rb["problems"]}
        ids = sorted(set(pa) & set(pb))
        same_text = sum(pa[i]["completion"] == pb[i]["completion"] for i in ids)
        same_fwd = sum(pa[i]["forwards"] == pb[i]["forwards"] for i in ids)
        same_score = sum(pa[i]["scores"] == pb[i]["scores"] for i in ids)
        grading_only += [f"{rel}: {i}" for i in ids if pa[i]["completion"] == pb[i]["completion"] and pa[i]["scores"] != pb[i]["scores"]]
        bad += (same_text != len(ids)) or (same_fwd != len(ids))
        knob = c["phi"] if c["rule"] == "pig.int" else c["theta"] if c["rule"] == "pig.ext" else c["nu"]
        paper = paper_cell(args.tables, c["model"], c["task"], c["geometry"], c["block"], c["rule"], knob) if args.tables else None
        ptxt = f"{paper[0]:.3f} [{paper[1]:.3f}, {paper[2]:.3f}] | {paper[3]:.2f}" if paper else "n/a | n/a"
        print(f"| {c['model']} | {c['task']} | {c['geometry']}{c['block']} | {c['rule']} | {knob:g} | {len(ids)} | "
              f"{same_text}/{len(ids)} | {same_fwd}/{len(ids)} | {same_score}/{len(ids)} | {sa['accuracy']:.3f} | {sb['accuracy']:.3f} | "
              f"{sa['tpf']:.2f} | {sb['tpf']:.2f} | {ptxt} |")
    print(f"\n{len(files)} cells, {bad} with a differing completion or forward count")
    if grading_only:
        print("identical completions graded differently (sandbox nondeterminism, not decoding):")
        for g in grading_only:
            print("  -", g)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
