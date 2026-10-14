# AGENTS.md

Machine-oriented reference for this repository: layout, full API, algorithms, extension points,
environment, verification and release. The human introduction is README.md.

## What is here

| path | role |
|---|---|
| `pig/` | the library. `pig.load(family)` -> `PIG`; `PIG.generate(prompt, mode, ...)` -> `Output` |
| `pig/rules.py` | `Round` (what a rule sees), `pig_int`, `pig_ext`, `confidence_threshold` as functions; `PIGInt`, `PIGExt`, `ConfidenceThreshold` as callables; `confidence_order`, `aggregate_influence`, `waiting_gain` |
| `pig/tables.py` | `FlipTable(x_edges, mu_edges, rates, axis)`: the calibration tables, `FlipTable.load(npz)` |
| `pig/fde/` | the adapters (`LLaDAFDE`, `NemotronFDE`, `load_fde`) |
| `pig/backbones/` | the backbones' modeling code, copied from the upstream repositories at the pinned revisions; changes marked `# PIG:` (LLaDA: `post_init()`, `tie_weights(**kwargs)`, `use_cache` on the config, no Auto registration; Nemotron: none). Licenses: `pig/backbones/<model>/LICENSE`, NOTICE for Nemotron |
| `pig/llada.py`, `pig/nemotron.py` | the decode loops: `LLaDADecoder` / `NemotronDecoder` (`chat_prompt`, `generate`, class attributes `family`, `geometry_cls`), geometries `Geometry` / `Window`, constants `MODEL_ID`, `REVISION`, `CANVAS` / `MAX_NEW` |
| `pig/api.py` | `PIG`, `Output`, `make_rule`, `DEFAULTS`, `MODES`, `FAMILIES` |
| `pig/loading.py` | `load`, `load_decoder`, `load_tables`, `make_geometry` |
| `pig/assets.py` | `ASSETS`, `asset_paths(model)`, `resolve(model)` (falls back to the Hub repositories) |
| `assets/<model>/` | `fde.safetensors`, `flip_confidence.npz`, `flip_entropy.npz`; `assets/SHA256SUMS` |
| `experiments/` | the benchmark harness: `tasks/` (`humaneval`, `gsm8k`), `evaluate`, `sweep`, `summarize`, `plot`, `configs/`, `stats`; protocol in `experiments/README.md` |
| `hub/` | the Hugging Face Hub build: `build` (writes `hub/dist/<repo>/`), `verify` (GPU: wrapper == library), `upload`; `cards/` (README and AGENTS per repository); `configuration_pig.py`, `modeling_pig.py` (the wrapper, concatenated with `pig/` into one file at build time) |
| `tests/` | `python -m pytest`, CPU only |
| `CITATION.cff` | citation metadata (GitHub's "Cite this repository"); the paper is arXiv:2610.17375 |

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
- Environment: one for both backbones, `requirements.txt` (`transformers>=5.19,<6`). No `trust_remote_code` in the library: the backbones are built from `pig/backbones/` with the upstream config and weights at the pinned revisions (`GSAI-ML/LLaDA-8B-Instruct@08b83a6feb34df1a6011b80c3c00c7563e963b07`, `nvidia/Nemotron-Labs-Diffusion-3B@0d51902da1f8869f83413ce642fab402fa5641e0`). GPU memory: LLaDA 24 GB (bf16, 16 GB weights), Nemotron 8 GB.
- Hub release: `python -m hub.build` -> `python -m hub.verify --family F --device cuda:N` (needs a GPU with room for two backbone copies) -> `python -m hub.upload --family F`.

## Algorithms

Each round runs one forward pass, gives every candidate position its argmax token, confidence νᵢ and entropy
Hᵢ, runs the adapter once (the directed matrix D̂, D̂ᵢⱼ ≈ Hᵢ(x) − Hᵢ(x with j committed to its argmax)), and
commits a prefix of the candidates in decreasing confidence. The aggregate influence of the more confident
members of S on i is μ̂ᵢ(S) = Σ_{j ∈ S, νⱼ > νᵢ} max(|D̂ᵢⱼ|, |D̂ⱼᵢ|). The tables p̂_ν(ν, μ) (PIG.INT) and
p̂_H(H, μ) (PIG.EXT) give the probability that a parallel commit differs from its sequential reference.

**PIG.INT(φ): single-round regret** (paper Algorithm 2). Keep the predicted number of mismatches inside
the round below φ.

```
S ← {most confident candidate}          # its flip rate is 0: its sequential reference is its own prediction
spent ← 0
for i in the remaining candidates, by decreasing confidence:
    bᵢ ← p̂_ν(νᵢ, μ̂ᵢ(S))                 # predicted probability that i disagrees with its sequential reference
    if spent + bᵢ > φ: break             # the first candidate that does not fit closes the round
    S ← S ∪ {i};  spent ← spent + bᵢ
commit S in parallel
```

**PIG.EXT(θ): cross-round regret** (paper Algorithm 3). Ask whether waiting one more forward pass would
remove regret on the upcoming candidates, and close the round once it would remove more than θ.

```
S ← ∅
for j in the candidates, by decreasing confidence:
    if S ≠ ∅ and νⱼ < floor: break       # optional confidence floor on admission
    S ← S ∪ {j}                          # admit first ...
    N ← the next h = 16 candidates
    Ĉ(S) ← Σ_{i∈N} [ p̂_H(Hᵢ, μ̂ᵢ(S∪N)) − p̂_H(Hᵢ − max_{k∈S} D̂ᵢₖ, μ̂ᵢ(S∪N)) ]
    if Ĉ(S) > θ: break                   # ... then stop when waiting would buy more than θ; j stays in S
commit S in parallel
```

The first term of Ĉ(S) estimates the regret of committing N now. The second estimates it after S is
committed and the model is re-evaluated, since committing S is predicted to lower each entropy Hᵢ by
max_{k∈S} D̂ᵢₖ. The two slowest operating points of the paper's PIG.EXT lane add a confidence floor:
θ = 0.125 with ν ≥ 0.9, and θ = 0.25 with ν ≥ 0.8.

Implementations: `pig/rules.py` (`pig_int`, `pig_ext`, `confidence_threshold`, and the callables `PIGInt`,
`PIGExt`, `ConfidenceThreshold`); tests transcribe the paper's algorithm boxes (`tests/test_rules.py`).

## Extending

A rule is any callable `Round -> list of positions`; `Round` has `cand` (eligible positions, ascending), `conf`,
`H` (per position) and `D` (adapter matrix, row = receiver, column = source).

```python
import pig
from pig import Round, confidence_order

def top_k(rnd: Round, k=4):                   # commit the k most confident candidates
    return confidence_order(rnd.cand, rnd.conf)[:k]

model = pig.load("llada")
tokens, forwards, _ = model.decoder.generate(model.encode("prompt"), top_k, model.geometry("block", 64))
```

`generate()` also takes `window` (PIG.EXT look-ahead |N(S)|, default 16), `max_new_tokens` (LLaDA canvas
length, fitted at 256 and always filled; Nemotron generation budget, default 2048, stops at end-of-sequence)
and `chat_template` (default True; pass token ids or `chat_template=False` to skip the template).

## Verification tools (hub/)

- `python -m hub.verify --family F [--repo DIR_OR_HUB_ID --no-build] --device cuda:N`: the Hub model and
  `pig.load(F)` decode identically (tokens and forward counts) on HumanEval prompts, three modes. Needs room for
  two backbone copies.
- `python -m hub.parity capture --family F --out X.pt [--matrix]` then `python -m hub.parity compare A.pt B.pt`:
  records per-round `conf`, `H`, `D` and the decoded tokens of two builds (for example two transformers versions)
  and compares them; `--matrix` covers every geometry with four operating points.
- `python -m hub.consistency --a DIR --b DIR --tables PAPER/tables`: compares two result trees cell by cell
  (completions and forward counts) and both against the paper's frontier tables.

