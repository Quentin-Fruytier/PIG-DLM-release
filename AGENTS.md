# AGENTS.md

Machine-oriented map of this repository. The human introduction is README.md.

## What is here

| path | role |
|---|---|
| `pig/` | the library. `pig.load(family)` -> `PIG`; `PIG.generate(prompt, mode, ...)` -> `Output` |
| `pig/rules.py` | `Round` (what a rule sees), `pig_int`, `pig_ext`, `confidence_threshold` as functions; `PIGInt`, `PIGExt`, `ConfidenceThreshold` as callables; `confidence_order`, `aggregate_influence`, `waiting_gain` |
| `pig/tables.py` | `FlipTable(x_edges, mu_edges, rates, axis)`: the calibration tables, `FlipTable.load(npz)` |
| `pig/fde/` | the adapters (`LLaDAFDE`, `NemotronFDE`, `load_fde`) |
| `pig/llada.py`, `pig/nemotron.py` | the decode loops: `LLaDADecoder` / `NemotronDecoder` (`chat_prompt`, `generate`, class attributes `family`, `geometry_cls`), geometries `Geometry` / `Window`, constants `MODEL_ID`, `REVISION`, `CANVAS` / `MAX_NEW` |
| `pig/api.py` | `PIG`, `Output`, `make_rule`, `DEFAULTS`, `MODES`, `FAMILIES` |
| `pig/loading.py` | `load`, `load_decoder`, `load_tables`, `make_geometry` |
| `pig/assets.py` | `ASSETS`, `asset_paths(model)`, `resolve(model)` (falls back to the Hub repositories) |
| `assets/<model>/` | `fde.safetensors`, `flip_confidence.npz`, `flip_entropy.npz`; `assets/SHA256SUMS` |
| `experiments/` | the benchmark harness: `tasks/` (`humaneval`, `gsm8k`), `evaluate`, `sweep`, `summarize`, `plot`, `configs/`, `stats`; protocol in `experiments/README.md` |
| `hub/` | the Hugging Face Hub build: `build` (writes `hub/dist/<repo>/`), `verify` (GPU: wrapper == library), `upload`; `cards/` (README and AGENTS per repository); `configuration_pig.py`, `modeling_pig.py` (the wrapper, concatenated with `pig/` into one file at build time) |
| `tests/` | `python -m pytest`, CPU only |

## Library API

```python
import pig
model = pig.load("llada")            # "llada" | "nemotron"; device=None -> first CUDA device
out = model.generate(prompt, mode=None, phi=None, theta=None, floor=None, window=None, nu=None,
                     geometry=None, block=None, max_new_tokens=None, chat_template=True)
```

- `mode`: `"pig.int"` (knob `phi`), `"pig.ext"` (`theta`; optional `floor`, `window`), `"confidence"` (`nu`). A knob of another mode raises `ValueError`. Unset knobs come from `pig.DEFAULTS[family]`: mode `pig.int`, phi 0.1, theta 0.5, floor None, window 16, nu 0.9, geometry `block`, block 32, canvas 256 (LLaDA) / max_new 2048 (Nemotron).
- `geometry`: LLaDA `"full"` | `"block"` | `"sliding"`; Nemotron `"block"` | `"sliding"` | `"nested"`. `block` is B.
- `max_new_tokens`: LLaDA canvas length (always filled); Nemotron generation budget (stops at EOS).
- `Output`: `text`, `tokens`, `forwards`, `tpf` (= len(tokens) / forwards), `mode`, `params`, `ms_per_round`.
- Lower level: `model.rule(mode, ...)` -> rule; `model.geometry(kind, B)`; `model.encode(prompt)` -> ids; `model.decoder.generate(ids, rule, geometry, canvas=... | max_new=...)` -> `(tokens, forwards, ms_per_round)`. A rule is any callable `Round -> list of positions`; `Round` has `cand` (eligible positions, ascending), `conf`, `H`, `D` (adapter matrix, row = receiver, column = source).
- The Hub wrapper (`UT-IFML/LLaDA-8B-Instruct-PIG`, `UT-IFML/Nemotron-Labs-Diffusion-3B-PIG`) exposes the same `generate` through `AutoModel.from_pretrained(repo, trust_remote_code=True)`; it is this package's code, so the two decode identically.

## Experiments

- `python -m experiments.evaluate --task {humaneval,gsm8k} --model {llada,nemotron} --rule {pig.int,pig.ext,confidence} [--phi|--theta [--floor] [--window]|--nu] --geometry K --block B [--n N] [--start I] [--device D] --out FILE.json`
- `python -m experiments.sweep experiments/configs/<task>_<model>.json [--out DIR] [--only GEOM ...] [--n N] [--list]`; result files `DIR/<model>_<task>_<geometry><B>_<rule>_<knob>.json`, finished cells skipped.
- `python -m experiments.summarize DIR ... [--latex]`; `python -m experiments.plot DIR ... --out PNG`.
- Config keys per geometry: `"confidence": [nu, ...]`, `"pig.int": [phi, ...]`, `"pig.ext": [theta | [theta, floor], ...]`.
- Result file: `config` (model, model_id, revision, task, shots, rule, phi, theta, floor, nu, window, geometry, block, problems, assets sha256, versions, gpu), `summary` (n, metric, passed, accuracy, wilson95, metrics, tpf, tpf_definition, mean_forwards, ms_per_round, wall_seconds), `problems` (task_id, passed, scores, forwards, n_gen, completion).
- Tasks: `NAME`, `N`, `SHOTS`, `METRICS`, `PRIMARY`, `load_problems()`, `prompt_text(model, problem)`, `score(model, problem, completion)`.

## Conventions

- Rule names are `pig.int`, `pig.ext`, `confidence` everywhere (code, configs, file names). The baseline lane is labelled Fast-dLLM in plots.
- Greedy decoding only; LLaDA always fills its canvas; Nemotron stops at EOS.
- No result files are committed. `results/` is ignored.
- Environments: one per backbone (`requirements/llada.txt`: transformers 4.49.0; `requirements/nemotron.txt`: transformers 4.56.2), because each backbone's own remote code pins the range.
- Hub release: `python -m hub.build` -> `python -m hub.verify --family F --device cuda:N` (needs a GPU with room for two backbone copies) -> `python -m hub.upload --family F`.
