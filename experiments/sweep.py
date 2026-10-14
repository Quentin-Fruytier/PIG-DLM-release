"""Run every operating point of a sweep config in one process (the model is loaded once).

    python -m experiments.sweep experiments/configs/gsm8k_llada.json [--out results/llada/gsm8k] [--only block64]
    python -m experiments.sweep experiments/configs/humaneval_nemotron.json --list   # print the cells, run nothing

Config format (JSON):
    {
      "model": "llada" | "nemotron",
      "task": "humaneval" | "gsm8k",
      "window": 16,
      "geometries": {"<name>": {"geometry": "<kind>", "block": B,
                                "confidence": [nu, ...],
                                "pig.int": [phi, ...],
                                "pig.ext": [theta or [theta, floor], ...]}, ...}
    }
Cells whose result file already exists are skipped, so an interrupted sweep resumes where it stopped. Each cell
is claimed with a `<cell>.lock` directory before it runs, so several workers can share one grid.
"""
import argparse
import json
from pathlib import Path

from pig.assets import ASSETS
from pig.loading import load_decoder, make_geometry

from . import tasks as task_registry
from .evaluate import cell_name, make_rule, run_cell


def expand(cfg):
    """-> list of (geometry name, geometry kwargs, spec dict)."""
    cells = []
    window = int(cfg.get("window", 16))
    for gname, g in cfg["geometries"].items():
        for nu in g.get("confidence", []):
            cells.append((gname, g, dict(rule="confidence", phi=None, theta=None, floor=None,
                                         nu=float(nu), window=window)))
        for phi in g.get("pig.int", []):
            cells.append((gname, g, dict(rule="pig.int", phi=float(phi), theta=None, floor=None,
                                         nu=None, window=window)))
        for point in g.get("pig.ext", []):
            theta, floor = (point if isinstance(point, list) else (point, None))
            cells.append((gname, g, dict(rule="pig.ext", phi=None, theta=float(theta),
                                         floor=None if floor is None else float(floor),
                                         nu=None, window=window)))
    return cells


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--out", help="result directory (default: results/<model>/<task>)")
    ap.add_argument("--only", nargs="*", help="restrict to these geometry names")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n", type=int, help="number of problems (default: the paper's count for the task)")
    ap.add_argument("--assets", default=str(ASSETS))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--list", action="store_true", help="print the cells and exit")
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    model = cfg["model"]
    task = task_registry.get(cfg.get("task", "humaneval"))
    out_dir = Path(args.out or f"results/{model}/{task.NAME}")
    cells = [c for c in expand(cfg) if not args.only or c[0] in args.only]
    todo = []
    for gname, g, spec in cells:
        geometry = make_geometry(model, g["geometry"], g.get("block", 0))
        name = cell_name(model, task.NAME, spec["rule"], geometry, spec["phi"], spec["theta"], spec["floor"], spec["nu"])
        path = out_dir / f"{name}.json"
        if args.list:
            print(("done  " if path.exists() else "todo  ") + name)
        elif not path.exists():
            todo.append((geometry, spec, path))
    if args.list:
        return
    print(f"{len(todo)} of {len(cells)} cells to run -> {out_dir}", flush=True)
    if not todo:
        return
    n = task_registry.problem_count(task, args.model if hasattr(args, 'model') else model) if args.n is None else args.n
    problems = task.load_problems()[args.start:args.start + n]
    decoder = load_decoder(model, args.assets, args.device)
    for geometry, spec, path in todo:
        lock = path.with_suffix(".lock")
        if path.exists():
            continue                                   # finished by another worker since this one started
        try:
            lock.mkdir(parents=True)
        except FileExistsError:
            print(f"[{path.stem}] claimed by another worker, skipping", flush=True)
            continue
        try:
            rule = make_rule(model, assets=args.assets, **spec)
            run_cell(decoder, model, task, rule, geometry, spec, problems, path, assets=args.assets)
        finally:
            lock.rmdir()


if __name__ == "__main__":
    main()
