# PIG: Parallel Inconsistency Guidance for Accelerating Diffusion Language Models

Quentin Fruytier, Anton Xue, Sanjay Shakkottai, Sujay Sanghavi (The University of Texas at Austin)

PIG lets a diffusion language model commit more tokens per forward pass at the same accuracy. A small
adapter predicts, from the forward pass the model already makes, which tokens would come out differently if
they were decoded one at a time; PIG commits in parallel only those that would not.

**Paper:** [arXiv:2610.17375](https://arxiv.org/abs/2610.17375) · **Models:** [LLaDA-8B-Instruct-PIG](https://huggingface.co/UT-IFML/LLaDA-8B-Instruct-PIG),
[Nemotron-Labs-Diffusion-3B-PIG](https://huggingface.co/UT-IFML/Nemotron-Labs-Diffusion-3B-PIG) · **Blog:** [Which tokens can a diffusion language model commit together?](https://quentin-fruytier.github.io/blog/pig-parallel-inconsistency-guidance/)

## Quick start

From the Hugging Face Hub (each repository is the whole model: backbone, adapter and tables):

```python
from transformers import AutoModel

model = AutoModel.from_pretrained("UT-IFML/LLaDA-8B-Instruct-PIG", trust_remote_code=True)
out = model.generate("Write a Python function that returns the n-th Fibonacci number.", mode="pig.int", phi=0.1)
print(out.text, out.tpf)          # tpf: tokens per forward pass
```

From this repository, after installing it:

```python
import pig

model = pig.load("nemotron")      # or "llada"
out = model.generate("Write a Python function that returns the n-th Fibonacci number.", mode="pig.ext", theta=0.25)
```

| argument | what it does | default |
|---|---|---|
| `mode` | `"pig.int"`, `"pig.ext"`, or `"confidence"` (the Fast-dLLM baseline) | `"pig.int"` |
| `phi` | PIG.INT budget; larger is faster, smaller more accurate (paper range 0.015 to 3) | `0.1` |
| `theta` | PIG.EXT budget; same direction (paper range 0.125 to 3) | `0.5` |
| `floor` | PIG.EXT: never commit below this confidence | none |
| `nu` | baseline: commit everything above this confidence | `0.9` |
| `geometry`, `block` | which positions are candidates: `"block"`, `"sliding"`, plus `"full"` (LLaDA) or `"nested"` (Nemotron) | `"block"`, `32` |

## How it works

Tokens committed in the same pass cannot see each other, so parallel decoding can drift from what the same
model would write one token at a time. PIG estimates that drift per token with two components fitted once
per backbone: a 2M-parameter adapter that predicts how committing one position changes the model's
uncertainty at the others, and a calibration table that turns this into the probability of a mismatch.
**PIG.INT** commits tokens in decreasing confidence until the predicted mismatches in the round exceed φ.
**PIG.EXT** stops once waiting one more pass is predicted to remove more than θ of regret. The algorithms
are written out in the paper and in [AGENTS.md](AGENTS.md).

## Install

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128    # match your CUDA version
pip install -r requirements.txt -e .                                     # one environment, transformers 5.x
```

LLaDA needs a GPU with 24 GB, Nemotron 8 GB. The backbone weights download on first use.

## Reproduce the paper

```bash
pip install -e ".[experiments]"
python -m experiments.sweep experiments/configs/gsm8k_llada.json      # one frontier: every operating point
python -m experiments.plot results/llada/gsm8k --out gsm8k_llada.png  # accuracy against tokens per forward
```

Configs cover HumanEval, GSM8K, MATH-500 and MBPP for both models; the protocol is in
[experiments/README.md](experiments/README.md).

## License

MIT for this repository's code, adapters and tables. The backbones' modeling code in `pig/backbones/` keeps
its upstream license (MIT for LLaDA, the NVIDIA Nemotron Open Model License for Nemotron); see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Citation

```bibtex
@misc{fruytier2026pig,
  title  = {{PIG}: Parallel Inconsistency Guidance for Accelerating Diffusion Language Models},
  author = {Fruytier, Quentin and Xue, Anton and Shakkottai, Sanjay and Sanghavi, Sujay},
  year   = {2026},
  eprint = {2610.17375},
  archivePrefix = {arXiv},
  primaryClass = {cs.CL},
  url    = {https://arxiv.org/abs/2610.17375}
}
```
