---
license: mit
base_model: GSAI-ML/LLaDA-8B-Instruct
base_model_relation: adapter
pipeline_tag: text-generation
library_name: transformers
inference: false
language:
- en
tags:
- diffusion-language-model
- masked-diffusion
- parallel-decoding
- llada
- pig
---


# PIG adapter for LLaDA-8B-Instruct

**Parallel Inconsistency Guidance (PIG)** lets a diffusion language model commit more tokens per forward pass
at the same accuracy. This repository holds what PIG adds to
[LLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct): a 2.2M-parameter adapter that predicts
how committing one masked position would change the model's uncertainty at every other, two calibration
tables that turn that into the probability of a parallel commit disagreeing with sequential decoding, and a
`transformers` wrapper that loads the backbone and runs the PIG.INT and PIG.EXT decoders.

**Paper:** *PIG: Parallel Inconsistency Guidance for Accelerating Diffusion Language Models*, Quentin Fruytier,
Anton Xue, Sanjay Shakkottai, Sujay Sanghavi (UT Austin), 2026; [arXiv:2610.17375](https://arxiv.org/abs/2610.17375).
**Code, harness and details:** https://github.com/Quentin-Fruytier/PIG-DLM-release.
**Companion:** [UT-IFML/Nemotron-Labs-Diffusion-3B-PIG](https://huggingface.co/UT-IFML/Nemotron-Labs-Diffusion-3B-PIG).

## Use

Needs a CUDA GPU with 24 GB of memory and `transformers==4.49.0` (LLaDA's own remote code requires the 4.x
line), plus `safetensors` and `numpy`. The code ships here and loads with `trust_remote_code=True`; the
backbone (16 GB) is downloaded from its own repository at a pinned revision on the first call.

```python
from transformers import AutoModel

model = AutoModel.from_pretrained("UT-IFML/LLaDA-8B-Instruct-PIG", trust_remote_code=True)
out = model.generate("Write a Python function that returns the n-th Fibonacci number.", mode="pig.int", phi=0.1)
print(out.text)
print(out.forwards, "forward passes;", round(out.tpf, 2), "tokens per forward")
```

`generate` returns `text`, `tokens`, `forwards` and `tpf` (tokens per forward pass, the paper's throughput
measure). Its arguments:

| argument | meaning | default |
|---|---|---|
| `mode` | `"pig.int"`, `"pig.ext"` or `"confidence"` (the Fast-dLLM confidence-threshold rule) | `"pig.int"` |
| `phi` | PIG.INT budget: predicted sequential-parallel mismatches allowed per round; larger is faster | `0.1` |
| `theta` | PIG.EXT budget: regret that one more forward pass may remove; larger is faster | `0.5` |
| `floor` | PIG.EXT only: stop admitting candidates below this confidence. The paper's two slowest points use `theta=0.125, floor=0.9` and `theta=0.25, floor=0.8` | none |
| `window` | PIG.EXT only: look-ahead size | `16` |
| `nu` | `"confidence"` only: commit every candidate above this confidence | `0.9` |
| `geometry`, `block` | which masked positions are candidates: `"block"` B (semi-autoregressive blocks), `"sliding"` B (the B leftmost masked positions) or `"full"` (the whole canvas) | `"block"`, `32` |
| `max_new_tokens` | the canvas length; the adapter was fitted at 256 | `256` |
| `chat_template` | wrap a string prompt in the backbone's chat template | `True` |

What the adapter predicts, how it and the tables were fitted, their held-out quality, and the caveats are in
[AGENTS.md](AGENTS.md) in this repository, with everything else on GitHub.

## License

The files in this repository are released under the MIT License. LLaDA-8B-Instruct is released by its authors
under the MIT License and is not redistributed here.

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
