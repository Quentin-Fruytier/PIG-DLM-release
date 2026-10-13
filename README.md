# PIG: Parallel Inconsistency Guidance for Accelerating Diffusion Language Models

Official code for the paper by Quentin Fruytier, Anton Xue, Sanjay Shakkottai and Sujay Sanghavi
(The University of Texas at Austin). PIG makes a diffusion language model commit more tokens per
forward pass at the same accuracy. This repository is the library: the two PIG decoders, the trained
adapters and calibration tables for **LLaDA-8B-Instruct** and **Nemotron-Labs-Diffusion-3B** (NVIDIA's
instruct model, published under that plain id), and one `generate()` call that runs them. The
benchmark harness that reproduces the paper's accuracy-throughput frontiers lives in
[`experiments/`](experiments).

> **Release in progress (October 2026).** Code, adapters and tables are here and mirrored on the
> Hugging Face Hub ([LLaDA-8B-Instruct-PIG](https://huggingface.co/UT-IFML/LLaDA-8B-Instruct-PIG),
> [Nemotron-Labs-Diffusion-3B-PIG](https://huggingface.co/UT-IFML/Nemotron-Labs-Diffusion-3B-PIG)).
> Still to come: the arXiv link to the paper. The repository URL is stable.

## Use PIG

**From the Hub**, with only `transformers` and a GPU:

```python
from transformers import AutoModel

model = AutoModel.from_pretrained("UT-IFML/LLaDA-8B-Instruct-PIG", trust_remote_code=True)   # or UT-IFML/Nemotron-Labs-Diffusion-3B-PIG
out = model.generate("Write a Python function that returns the n-th Fibonacci number.", mode="pig.int", phi=0.1)
print(out.text, out.forwards, out.tpf)
```

**From this repository**, after the installation below:

```python
import pig

model = pig.load("llada")                                                 # or "nemotron"
out = model.generate("Write a Python function that returns the n-th Fibonacci number.",
                     mode="pig.ext", theta=0.25, floor=0.8, geometry="block", block=64)
print(out.text, out.forwards, out.tpf)                                    # tokens per forward pass
```

Both calls run the same code (the Hub model vendors `pig/`) and take the same arguments:

| argument | meaning | default |
|---|---|---|
| `mode` | `"pig.int"`, `"pig.ext"` or `"confidence"` (the Fast-dLLM confidence-threshold rule) | `"pig.int"` |
| `phi` | PIG.INT budget: predicted mismatches allowed per round; larger is faster | `0.1` |
| `theta` | PIG.EXT budget: regret that waiting one more pass may remove; larger is faster | `0.5` |
| `floor` | PIG.EXT only: stop admitting candidates below this confidence (the paper's two slowest points use θ=0.125 with 0.9 and θ=0.25 with 0.8) | none |
| `window` | PIG.EXT only: look-ahead size \|N(S)\| | `16` |
| `nu` | `confidence` only: commit every candidate above this confidence | `0.9` |
| `geometry`, `block` | which masked positions are candidates. LLaDA: `"full"` canvas, `"block"` B (semi-autoregressive blocks) or `"sliding"` B (the B leftmost masked positions). Nemotron: `"block"` B, `"sliding"` B or `"nested"` B (a B-wide band in a 2B window) | `"block"`, `32` |
| `max_new_tokens` | LLaDA: the canvas length (the adapter was fitted at 256). Nemotron: the generation budget; decoding stops at end-of-sequence | `256` / `2048` |
| `chat_template` | wrap a string prompt in the backbone's chat template | `True` |

`generate()` returns the text, the token ids, the number of backbone forward passes and tokens per
forward (TPF), the paper's throughput measure. Paper operating points: PIG.INT φ from 0.015 to 3,
PIG.EXT θ from 0.125 to 3; smaller budgets are slower and more accurate.

## How it works

A diffusion language model predicts every masked position in one forward pass, so it can commit
several tokens per pass. Tokens committed together cannot condition on one another, so the result can
differ from what the same model would write committing those positions one at a time. We call that
disagreement **sequential–parallel inconsistency** and treat it as the regret of parallel decoding:
commit as many positions as possible while the predicted regret stays small. Measuring the regret
exactly needs the sequential replay that parallel decoding is meant to avoid, so PIG predicts it from
the one forward pass the decoder already makes, with two fitted-once components:

- **Entropy Finite Differences Estimator (FDE).** A small adapter (2.2M parameters for LLaDA, 1.9M
  for Nemotron, under 0.1% of a forward pass) on the frozen backbone. It returns the directed matrix
  D̂, where D̂ᵢⱼ ≈ Hᵢ(x) − Hᵢ(x with j committed to its argmax) is the change in the model's
  uncertainty at i caused by committing j.
- **Calibration tables.** p̂_ν(ν, μ) and p̂_H(H, μ) give the probability that a parallel commit differs
  from its sequential reference, indexed by the token's confidence ν or entropy H and by the aggregate
  influence of the more confident tokens committed with it, μ̂ᵢ(S) = Σ_{j ∈ S, νⱼ > νᵢ} max(|D̂ᵢⱼ|, |D̂ⱼᵢ|).

Each round runs one forward pass, gives every candidate position its argmax token, confidence νᵢ and
entropy Hᵢ, runs the adapter once, and commits a prefix of the candidates in decreasing confidence:

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

## Extend it

A rule is a function of one `Round` (`pig/rules.py`): `cand`, the eligible positions in ascending
order; `conf` and `H`, the confidence and entropy at every position; `D`, the adapter's matrix. It
returns the positions to commit. Anything with that signature plugs into `generate()`:

```python
from pig import Round, confidence_order

def top_k(rnd: Round, k=4):                   # commit the k most confident candidates
    return confidence_order(rnd.cand, rnd.conf)[:k]

model = pig.load("llada")
tokens, forwards, _ = model.decoder.generate(model.encode("prompt"), top_k, model.geometry("block", 64))
```

The pieces are independent: `pig/rules.py` (PIG.INT, PIG.EXT and the baseline as plain functions and
callable classes), `pig/tables.py` (table lookup), `pig/fde/` (the adapters), `pig/llada.py` and
`pig/nemotron.py` (the two decode loops, each exposing `chat_prompt`, `generate` and its geometry
class), `pig/api.py` (`PIG`, `generate()` and the defaults) and `pig/loading.py` (`pig.load`).

## Install

LLaDA's remote code requires `transformers` 4.x and Nemotron's requires ≥ 4.53, so use one
environment per model:

```bash
python -m venv .venv-llada && source .venv-llada/bin/activate           # LLaDA-8B-Instruct
pip install torch --index-url https://download.pytorch.org/whl/cu128    # match your CUDA version
pip install -r requirements/llada.txt -e .

python -m venv .venv-nemotron && source .venv-nemotron/bin/activate     # Nemotron-Labs-Diffusion-3B
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements/nemotron.txt -e .
```

The backbones download from the Hub at pinned revisions (`GSAI-ML/LLaDA-8B-Instruct@08b83a6f`,
`nvidia/Nemotron-Labs-Diffusion-3B@0d51902d`); LLaDA needs a GPU with at least 24 GB. The adapters
and tables ship in [`assets/`](assets) (`fde.safetensors`, `flip_confidence.npz`, `flip_entropy.npz`
per model, with `SHA256SUMS`) and are mirrored in the two Hub repositories, from which
`pig.load` fetches them if `assets/` is missing.

## Reproduce the paper's frontiers

```bash
pip install -e ".[experiments]"
python -m experiments.sweep experiments/configs/gsm8k_llada.json      # every operating point of a frontier
python -m experiments.summarize results/llada/gsm8k                   # tables
python -m experiments.plot results/llada/gsm8k --out gsm8k_llada.png  # accuracy vs TPF curves
```

[`experiments/`](experiments) holds the HumanEval and GSM8K tasks with the paper's prompts and
graders, the three lanes (Fast-dLLM, PIG.INT, PIG.EXT), and one config per task and model listing
the paper's operating points. Its [README](experiments/README.md) states the protocol.

## Layout and tests

```
pig/            the library (rules, tables, adapters, decoders, generate(), loading)
assets/         adapters and calibration tables
experiments/    benchmark harness, configs, tables, plots
hub/            build, verify and upload of the Hugging Face Hub repositories (vendors pig/)
tests/          python -m pytest   (CPU only)
```

## License

MIT (see [`LICENSE`](LICENSE)) for the code, the adapters and the calibration tables. The backbones are
not redistributed; their licenses, and the provenance of the Nemotron assets, are stated in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## Citation

```bibtex
@misc{fruytier2026pig,
  title  = {{PIG}: Parallel Inconsistency Guidance for Accelerating Diffusion Language Models},
  author = {Fruytier, Quentin and Xue, Anton and Shakkottai, Sanjay and Sanghavi, Sujay},
  year   = {2026},
  note   = {arXiv preprint; identifier to follow}
}
```
