# PIG: Parallel Inconsistency Guidance for Accelerating Diffusion Language Models

Official code for the paper by Quentin Fruytier, Anton Xue, Sanjay Shakkottai and Sujay Sanghavi
(The University of Texas at Austin). The release covers our two PIG decoders, **PIG.INT** and
**PIG.EXT**, on **LLaDA-8B-Instruct** and **Nemotron-Labs-Diffusion-3B** (NVIDIA's instruct model, published
under that plain id; the base model is `-3B-Base`), together with the evaluation harness (HumanEval and GSM8K,
with the confidence-threshold baseline), the trained adapters and the calibration tables.

> **Release in progress (October 2026).** The code, the two trained adapters and the four calibration
> tables are in this repository (`assets/`) and mirrored on the Hugging Face Hub
> ([LLaDA-8B-Instruct-PIG](https://huggingface.co/UT-IFML/LLaDA-8B-Instruct-PIG),
> [Nemotron-Labs-Diffusion-3B-PIG](https://huggingface.co/UT-IFML/Nemotron-Labs-Diffusion-3B-PIG)).
> Still to come: the arXiv link to the paper. The repository URL is stable.

## The idea

A diffusion language model (DLM) predicts every masked position in one forward pass, so it can commit
several tokens per pass. Tokens committed together cannot condition on one another, however. The
result can then differ from what the *same* model would write if it committed those positions one at
a time, in the same order. We call that disagreement **sequential–parallel inconsistency** and
treats it as the regret of parallel decoding. Deciding which positions to commit in parallel then
becomes a budgeted selection problem: commit as many positions as possible while the predicted regret
stays small.

Measuring the regret exactly would require the sequential replay that parallel decoding is meant to
avoid. PIG predicts it instead, from the one forward pass the decoder already makes.

- **Entropy Finite Differences Estimator (FDE).** A small adapter (about 2.2M parameters for LLaDA,
  1.9M for Nemotron) on the frozen backbone. It reads the current pass and returns the full directed
  matrix D̂, where D̂ᵢⱼ ≈ Hᵢ(x) − Hᵢ(x with j committed to its argmax) is the change in the
  model's uncertainty at i caused by committing j. It adds well under 0.1% of a backbone forward
  pass in arithmetic.
- **Calibration tables.** Two binned tables give the probability that a parallel commit differs from
  its sequential reference. p̂_ν(ν, μ) is indexed by the token's confidence ν, and p̂_H(H, μ) by its
  entropy H. Both also depend on the aggregate influence of the more confident tokens decoded before it:

  μ̂ᵢ(S) = Σ_{j ∈ S, νⱼ > νᵢ} max(|D̂ᵢⱼ|, |D̂ⱼᵢ|).

Both are fitted once per backbone and only on training-split data. The adapter is trained on states
from GSM8K, MATH, MBPP and OpenCoder problems, and the tables on sequential replays of GSM8K and
OpenCoder states. Both are then reused unchanged across benchmarks, budgets and decoding geometries.

## The two decoders

Each round runs one forward pass, gives every candidate position its argmax token, confidence νᵢ and
entropy Hᵢ, and runs the adapter once. The rule then visits the candidates in decreasing confidence and
commits a prefix of that order in parallel.

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

Both rules are plain functions in [`pig/rules.py`](pig/rules.py). They work on any backend that
supplies confidences, entropies and a D̂ matrix.

## Installation

LLaDA's remote code requires `transformers` 4.x, and Nemotron's requires ≥ 4.53, so use one
environment per model:

```bash
# LLaDA-8B-Instruct
python -m venv .venv-llada && source .venv-llada/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128     # match your CUDA version
pip install -r requirements/llada.txt && pip install -e .

# Nemotron-Labs-Diffusion-3B
python -m venv .venv-nemotron && source .venv-nemotron/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements/nemotron.txt && pip install -e .
```

The backbones are downloaded from the Hugging Face Hub at pinned revisions:
`GSAI-ML/LLaDA-8B-Instruct@08b83a6f` and `nvidia/Nemotron-Labs-Diffusion-3B@0d51902d`. LLaDA needs
a GPU with at least 24 GB of memory; Nemotron needs less.

The adapters and tables ship in [`assets/`](assets):

```
assets/<model>/fde.safetensors        adapter weights (Entropy Finite Differences Estimator)
assets/<model>/flip_confidence.npz    p̂_ν(ν, μ)   read by PIG.INT
assets/<model>/flip_entropy.npz       p̂_H(H, μ)   read by PIG.EXT
assets/SHA256SUMS
```

Here `<model>` is `llada` or `nemotron`. Every result file records the SHA-256 of the assets it used.

The same six files are mirrored on the Hugging Face Hub, one repository per backbone, with model cards:
[UT-IFML/LLaDA-8B-Instruct-PIG](https://huggingface.co/UT-IFML/LLaDA-8B-Instruct-PIG) and
[UT-IFML/Nemotron-Labs-Diffusion-3B-PIG](https://huggingface.co/UT-IFML/Nemotron-Labs-Diffusion-3B-PIG).
If `assets/<model>/` is missing or incomplete, `pig.assets.resolve` downloads the files from there into the
`huggingface_hub` cache, so the code also runs from a checkout without `assets/`.

## Quick start

The easiest way to run PIG is the Hugging Face Hub model, which needs only `transformers` (and a GPU):

```python
from transformers import AutoModel

model = AutoModel.from_pretrained("UT-IFML/LLaDA-8B-Instruct-PIG", trust_remote_code=True)   # or UT-IFML/Nemotron-Labs-Diffusion-3B-PIG
out = model.generate("Write a Python function that returns the n-th Fibonacci number.", mode="pig.int", phi=0.1)
print(out.text, out.forwards, out.tpf)       # mode="pig.ext", theta=..., floor=...  |  mode="confidence", nu=...
```

The Hub repositories are built from this package by `python -m hub.build` (see `hub/`): the wrapper's decoders,
rules, tables and adapter are vendored verbatim from `pig/`, and `python -m hub.verify` checks on a GPU that the
wrapper and this harness decode identically. What follows is the harness, which reproduces the paper's
operating points.

One operating point (`--task humaneval`, the default, runs all 164 problems; `--task gsm8k` runs the
first 500 test problems, as in the paper):

```bash
# LLaDA-8B-Instruct, HumanEval, PIG.INT with φ = 0.1, fixed 64-token blocks
python -m pig.evaluate --task humaneval --model llada --rule pig-int --phi 0.1 \
    --geometry block --block 64 --out results/llada/humaneval/llada_humaneval_block64_pig-int_phi0.1.json

# Nemotron-Labs-Diffusion-3B, GSM8K, PIG.EXT with θ = 0.25 and a 0.8 confidence floor, fixed 32-token blocks
python -m pig.evaluate --task gsm8k --model nemotron --rule pig-ext --theta 0.25 --floor 0.8 \
    --geometry block --block 32 --out results/nemotron/gsm8k/nemotron_gsm8k_block32_pig-ext_theta0.25_floor0.8.json

# the confidence-threshold baseline (the Fast-dLLM rule) at ν = 0.9
python -m pig.evaluate --task gsm8k --model llada --rule confidence --nu 0.9 \
    --geometry block --block 64 --out results/llada/gsm8k/llada_gsm8k_block64_confidence_nu0.9.json
```

A whole frontier: the model is loaded once, and finished operating points are skipped, so an
interrupted sweep resumes where it stopped.

```bash
python -m pig.sweep configs/humaneval_llada.json                  # -> results/llada/humaneval/*.json
python -m pig.sweep configs/gsm8k_llada.json --only block64 sliding64
python -m pig.sweep configs/gsm8k_nemotron.json
python -m pig.summarize results/llada/gsm8k results/nemotron/gsm8k   # markdown tables (--latex for LaTeX rows)
```

The four configs (`configs/<task>_<model>.json`) list the operating points of the paper's frontier
tables for every decoding geometry: the confidence-threshold ladder, the PIG.INT φ ladder and the
PIG.EXT θ ladder with its two floored slow points. Each result file holds every problem's completion
and scores, the accuracy with its Wilson interval, the throughput, and the SHA-256 of the assets used.

From Python:

```python
from pig import FlipTable, PIGExt, PIGInt
from pig.llada import Geometry, LLaDADecoder

decoder = LLaDADecoder("assets/llada/fde.safetensors")
rule = PIGInt(FlipTable.load("assets/llada/flip_confidence.npz"), phi=0.1)
# or: PIGExt(FlipTable.load("assets/llada/flip_entropy.npz"), theta=0.25, floor=0.8)

prompt = decoder.chat_prompt("Write a Python function that returns the n-th Fibonacci number.")
tokens, forwards, _ = decoder.generate(prompt, rule, Geometry("block", 64))
print(decoder.tokenizer.decode(tokens, skip_special_tokens=True))
print(f"{256 / forwards:.1f} tokens per forward")
```

## Evaluation protocol

| | LLaDA-8B-Instruct | Nemotron-Labs-Diffusion-3B |
|---|---|---|
| HumanEval prompt | chat template; user turn = "Complete the following Python function. Return the complete function in a \`\`\`python code block." + problem prompt; 0-shot | chat template; user turn = "Complete the following Python function." + problem prompt; 0-shot |
| GSM8K prompt | chat template; user turn = five fixed exemplars from the training split (seed 1234) in the `Question:`/`Answer:`/`####` format, then `Question: <q>` and `Answer:` | chat template; user turn = the question as is; 0-shot |
| canvas | 256 tokens, always filled; no KV cache | up to 2048 tokens, stops at end of sequence; prefix KV cache |
| candidate geometries | `full` canvas; `block` B (semi-autoregressive blocks); `sliding` B (the B leftmost masked positions) | `block` B (fixed slab); `sliding` B (a window holding B masked positions); `nested` B (a B-wide commit band in a window holding 2B masked positions) |
| decoding | greedy (argmax); shots as above | greedy (argmax); 0-shot throughout |
| throughput (TPF) | 256 / mean forward passes per problem | mean over problems of generated tokens / forward passes, counting the prompt pass and cache updates |
| HumanEval grading | pass@1. A fenced block that defines the entry point is the whole program; otherwise the completion continues the prompt, cut at the HumanEval stop sequences. 10 s timeout | pass@1. The last fenced block that defines the entry point, prefixed with the prompt when needed. 15 s timeout |
| GSM8K grading | lm-eval's two extractors on the generation cut at the stop sequences: flexible (the last number; the reported metric) and strict (the first `#### <number>` line); both are recorded | the same extractors; flexible is the reported metric |
| problems | HumanEval: all 164; GSM8K: the first 500 of the test split | the same |

Accuracies are reported with Wilson 95% intervals. Results shift slightly between GPU types, because
the backbone's floating-point results differ across hardware.

## Repository layout

```
pig/rules.py        PIG.INT and PIG.EXT selection rules
pig/tables.py       calibration-table lookup
pig/fde/            adapter architectures (LLaDA, Nemotron)
pig/llada.py        LLaDA-8B-Instruct decoding backend
pig/nemotron.py     Nemotron-Labs-Diffusion-3B decoding backend
pig/humaneval.py    HumanEval prompts, graders and Wilson intervals
pig/gsm8k.py        GSM8K prompts (5-shot LLaDA / 0-shot Nemotron) and extractors
pig/tasks.py        the task registry (one interface for every benchmark)
pig/evaluate.py     one operating point            (python -m pig.evaluate)
pig/sweep.py        a grid of operating points     (python -m pig.sweep)
hub/                the Hugging Face Hub wrapper and its build / verify / upload scripts
pig/summarize.py    result tables                  (python -m pig.summarize)
configs/            the paper's HumanEval and GSM8K operating points, per model
tests/              unit tests (python -m pytest)
```

## Tests

```bash
python -m pytest     # CPU only: the rules against the paper's algorithms, tables, adapters, graders, the harness plumbing
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
