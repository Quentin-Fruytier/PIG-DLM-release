---
license: mit
base_model: nvidia/Nemotron-Labs-Diffusion-3B
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
- nemotron
- pig
---


# PIG adapter for Nemotron-Labs-Diffusion-3B (the instruct model)

**Parallel Inconsistency Guidance (PIG)** lets a diffusion language model commit more tokens per forward pass
at the same accuracy. This repository holds what PIG adds to
[Nemotron-Labs-Diffusion-3B](https://huggingface.co/nvidia/Nemotron-Labs-Diffusion-3B), NVIDIA's **instruct**
model (published under that plain id; the base model is `-3B-Base` and is not what this is fitted to): the
backbone itself with a 1.9M-parameter adapter that predicts how committing one masked position of the current block would change the
model's uncertainty at every other, two calibration tables that turn that into the probability of a parallel
commit disagreeing with sequential decoding, and a `transformers` wrapper that loads the backbone and runs the
PIG.INT and PIG.EXT decoders.

**Paper:** *PIG: Parallel Inconsistency Guidance for Accelerating Diffusion Language Models*, Quentin Fruytier,
Anton Xue, Sanjay Shakkottai, Sujay Sanghavi (UT Austin), 2026; arXiv link to follow.
**Code, harness and details:** https://github.com/Quentin-Fruytier/PIG-DLM-release.
**Companion:** [UT-IFML/LLaDA-8B-Instruct-PIG](https://huggingface.co/UT-IFML/LLaDA-8B-Instruct-PIG).

## Use

Needs a CUDA GPU with 8 GB of memory and `transformers` 5.x (plus `safetensors`, `numpy`, `accelerate`). This
repository is the whole model: the backbone's weights and modeling code (copied from Nemotron-Labs-Diffusion-3B
at a pinned revision, under NVIDIA's licence terms), the PIG adapter and the calibration tables. The code
loads with `trust_remote_code=True`; the first call downloads 7 GB.

```python
from transformers import AutoModel

model = AutoModel.from_pretrained("UT-IFML/Nemotron-Labs-Diffusion-3B-PIG", trust_remote_code=True)
out = model.generate("Write a Python function that returns the n-th Fibonacci number.",
                     mode="pig.ext", theta=0.25, floor=0.8)
print(out.text)
print(out.forwards, "forward passes;", round(out.tpf, 2), "tokens per forward")
```

`generate` decodes until the end-of-sequence token and returns `text`, `tokens`, `forwards` and `tpf`
(generated tokens per forward pass, the paper's throughput measure; forward passes count the prompt pass,
every window forward and every cache update). Its arguments:

| argument | meaning | default |
|---|---|---|
| `mode` | `"pig.int"`, `"pig.ext"` or `"confidence"` (the Fast-dLLM confidence-threshold rule) | `"pig.int"` |
| `phi` | PIG.INT budget: predicted sequential-parallel mismatches allowed per round; larger is faster | `0.1` |
| `theta` | PIG.EXT budget: regret that one more forward pass may remove; larger is faster | `0.5` |
| `floor` | PIG.EXT only: stop admitting candidates below this confidence. The paper's two slowest points use `theta=0.125, floor=0.9` and `theta=0.25, floor=0.8` | none |
| `window` | PIG.EXT only: look-ahead size | `16` |
| `nu` | `"confidence"` only: commit every candidate above this confidence | `0.9` |
| `geometry`, `block` | which masked positions are candidates: `"block"` B (the model's own fixed block), `"sliding"` B (a window holding B masked positions) or `"nested"` B (a B-wide commit band in a window twice as wide) | `"block"`, `32` |
| `max_new_tokens` | the generation budget; decoding stops at the end-of-sequence token | `2048` |
| `chat_template` | wrap a string prompt in the backbone's chat template | `True` |

What the adapter predicts, how it and the tables were fitted, their held-out quality, and the caveats are in
[AGENTS.md](AGENTS.md) in this repository, with everything else on GitHub.

## License

The files in this repository are released under the MIT License.

The backbone weights, tokenizer and modeling code are Nemotron-Labs-Diffusion-3B's, released by NVIDIA
Corporation under the
[NVIDIA Nemotron Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-nemotron-open-model-license/)
and copied here under its terms (see [NOTICE.md](NOTICE.md)); use of the backbone is governed by that licence.
NVIDIA's names and marks are used only to identify the model.

## Citation

```bibtex
@misc{fruytier2026pig,
  title  = {{PIG}: Parallel Inconsistency Guidance for Accelerating Diffusion Language Models},
  author = {Fruytier, Quentin and Xue, Anton and Shakkottai, Sanjay and Sanghavi, Sujay},
  year   = {2026},
  note   = {arXiv preprint; identifier to follow}
}
```
