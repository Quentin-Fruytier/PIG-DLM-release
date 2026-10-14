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

# LLaDA-8B-Instruct with PIG

[LLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct) with **Parallel Inconsistency Guidance
(PIG)**, which lets the model commit more tokens per forward pass at the same accuracy. This repository is the
whole model: the backbone, a 2.2M-parameter PIG adapter and its calibration tables.

**Paper:** *PIG: Parallel Inconsistency Guidance for Accelerating Diffusion Language Models*, Fruytier, Xue,
Shakkottai, Sanghavi (UT Austin), 2026; arXiv link to follow. **Code:** https://github.com/Quentin-Fruytier/PIG-DLM-release
**Companion:** [UT-IFML/Nemotron-Labs-Diffusion-3B-PIG](https://huggingface.co/UT-IFML/Nemotron-Labs-Diffusion-3B-PIG)

## Use

Needs `transformers` 5.x and a GPU with 24 GB.

```python
from transformers import AutoModel

model = AutoModel.from_pretrained("UT-IFML/LLaDA-8B-Instruct-PIG", trust_remote_code=True)
out = model.generate("Write a Python function that returns the n-th Fibonacci number.", mode="pig.int", phi=0.1)
print(out.text, out.tpf)          # tpf: tokens per forward pass
```

| argument | what it does | default |
|---|---|---|
| `mode` | `"pig.int"`, `"pig.ext"`, or `"confidence"` (the Fast-dLLM baseline) | `"pig.int"` |
| `phi` | PIG.INT budget; larger is faster, smaller more accurate (paper range 0.015 to 3) | `0.1` |
| `theta` | PIG.EXT budget; same direction (paper range 0.125 to 3) | `0.5` |
| `floor` | PIG.EXT: never commit below this confidence | none |
| `nu` | baseline: commit everything above this confidence | `0.9` |
| `geometry`, `block` | which positions are candidates: `"block"`, `"sliding"` or `"full"` | `"block"`, `32` |

Implementation details, every argument, and how the adapter and tables were fitted: [AGENTS.md](AGENTS.md).

## License

MIT. The backbone is LLaDA-8B-Instruct, released by its authors under the MIT License ([NOTICE.md](NOTICE.md)).

## Citation

```bibtex
@misc{fruytier2026pig,
  title  = {{PIG}: Parallel Inconsistency Guidance for Accelerating Diffusion Language Models},
  author = {Fruytier, Quentin and Xue, Anton and Shakkottai, Sanjay and Sanghavi, Sujay},
  year   = {2026},
  note   = {arXiv preprint; identifier to follow}
}
```
