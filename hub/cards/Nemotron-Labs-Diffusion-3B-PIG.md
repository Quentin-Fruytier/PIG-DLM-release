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

# PIG adapter for Nemotron-Labs-Diffusion-3B

The trained components of **Parallel Inconsistency Guidance (PIG)** for
[Nemotron-Labs-Diffusion-3B](https://huggingface.co/nvidia/Nemotron-Labs-Diffusion-3B): a ~1.9M-parameter
adapter that predicts, from the one forward pass the decoder already runs, how committing any masked position
of the current block would change the backbone's uncertainty at every other masked position, and two
calibration tables that turn those predictions into the probability that a parallel commit disagrees with
sequential decoding. The PIG.INT and PIG.EXT decoders read them to decide how many positions to commit per
forward pass, and extend the model's accuracy–throughput frontier over its own confidence-threshold decoder on
GSM8K, MATH-500, MBPP, HumanEval and MMLU.

This repository holds the trained components together with a `transformers` wrapper that loads the backbone
and runs the decoders, so the model is usable with `AutoModel.from_pretrained` and one `generate` call. The
evaluation harness and the paper's operating points are in the code repository.

**Paper:** *PIG: Parallel Inconsistency Guidance for Accelerating Diffusion Language Models* — Quentin Fruytier,
Anton Xue, Sanjay Shakkottai, Sujay Sanghavi (The University of Texas at Austin), 2026. arXiv link to follow.  
**Code:** https://github.com/Quentin-Fruytier/PIG-DLM-release  
**Companion:** [UT-IFML/LLaDA-8B-Instruct-PIG](https://huggingface.co/UT-IFML/LLaDA-8B-Instruct-PIG), the same
two components fitted for LLaDA-8B-Instruct.

## Files

| file | what it is |
|---|---|
| `model.safetensors` | the Entropy Finite Differences Estimator (the adapter): 55 tensors, 1,903,383 parameters, fp32, keys prefixed `fde.`; the header records `epoch = 7` and `val_edge_corr = 0.7940` |
| `flip_confidence.npz` | the mismatch table $\hat p_\nu(\nu, \hat\mu)$ read by PIG.INT: 8 confidence bins × 5 coupling bins |
| `flip_entropy.npz` | the mismatch table $\hat p_H(H, \hat\mu)$ read by PIG.EXT: 10 entropy bins × 5 coupling bins |
| `config.json` | the backbone id and pinned revision, the decoding defaults, adapter and table provenance |
| `modeling_pig.py`, `configuration_pig.py` | the wrapper: `from_pretrained` and `generate` |
| `decoder.py`, `rules.py`, `tables.py`, `fde.py` | the decode loop, the selection rules, the table lookup and the adapter architecture, vendored verbatim from the code repository (each file names the commit it came from) |

Each `.npz` holds `x_edges` (bin edges on the confidence or entropy axis), `mu_edges` (bin edges on the
coupling axis), `rates` (the empirical mismatch rate of each cell), `rows` and `flips` (the commit and mismatch
counts behind each cell), `axis` and `description`.

The backbone is not redistributed here. The wrapper loads `nvidia/Nemotron-Labs-Diffusion-3B` at the pinned
revision `0d51902d` from its own repository, under NVIDIA's licence terms.

## Use

Needs a CUDA GPU and `transformers==4.56.2` (the model's own remote code requires 4.53 or later; tested with
`torch 2.11+cu128`), plus `safetensors`, `numpy` and `accelerate`. Nothing else to install: the code ships in
this repository and is loaded with `trust_remote_code=True`.

```python
from transformers import AutoModel

model = AutoModel.from_pretrained("UT-IFML/Nemotron-Labs-Diffusion-3B-PIG", trust_remote_code=True)
out = model.generate("Write a Python function that returns the n-th Fibonacci number.",
                     mode="pig.ext", theta=0.25, floor=0.8)
print(out.text)
print(out.forwards, "forward passes;", round(out.tpf, 2), "tokens per forward")
```

`from_pretrained` loads the adapter and the tables from this repository, then the backbone from its own
repository (about 7 GB of GPU memory), onto the first CUDA device. `device="cuda:1"` or `device="cpu"` picks
another; `load_backbone=False` defers the backbone, to be attached later with `model.attach_backbone()`.

`generate(prompt, mode, ...)` decodes up to `max_new_tokens` (default 2048) after the prompt, which is wrapped
in the chat template (pass token ids, or `chat_template=False`, to skip it), stops at the end-of-sequence
token, and returns `text`, `tokens`, `forwards` and `tpf`, the generated tokens per forward pass that the paper
uses as its throughput measure (forward passes count the prompt pass, every window forward and every cache
update).

| `mode` | rule | knobs |
|---|---|---|
| `"pig.int"` (default) | PIG.INT: the single-round regret budget | `phi` (default 0.1) |
| `"pig.ext"` | PIG.EXT: the cross-round regret budget | `theta` (default 0.5); `floor`, a confidence floor on admission (default none); `window`, the look-ahead (default 16) |
| `"confidence"` | the model's own confidence-threshold decoder (Fast-dLLM) | `nu` (default 0.9) |

`geometry` and `block` set the window geometry: `"block"` (default; the model's own fixed block of `block` = 32
tokens), `"sliding"`, or `"nested"` (a commit band of `block` tokens inside a window twice as wide). The paper's
Nemotron operating points use `block=32` on the math benchmarks and `block=64` on code; its two slowest PIG.EXT
points are `theta=0.125, floor=0.9` and `theta=0.25, floor=0.8`.

The code repository reproduces the paper's HumanEval operating points from the same files
(`python -m pig.evaluate`, `python -m pig.sweep`). The wrapper and that harness decode identically: checked
token for token, and forward pass for forward pass, on HumanEval problems under all three modes.

## What the adapter predicts

For masked positions $i, j$ of the current block, the target is the directed entropy finite difference

$$\hat{\mathbf D}_{ij} \approx H_i(\mathbf x) - H_i(\mathbf x \oplus \hat x_j),$$

the change in the backbone's predictive entropy at $i$ if $j$ were committed to its current argmax. One forward
pass of the frozen backbone with $j$ committed measures a single column of this matrix exactly; the adapter
returns the whole matrix from the forward pass the decoder already runs. PIG.INT sums the magnitudes of these
edges from the accepted prefix into a coupling $\hat\mu_i$ and reads $\hat p_\nu(\nu_i, \hat\mu_i)$; PIG.EXT
uses their sign to predict next-round entropies and reads $\hat p_H$ at the current and predicted entropy.

**Architecture.** The same design as the LLaDA adapter, re-instantiated for this block-causal backbone: the
3072-d hidden state is read from the model's own last-hidden-state output (its diffusion head is applied to that
output, so the logits are reproduced exactly); because the model never attends across blocks, the adapter
operates on one block, with block-relative rotary positions, an extra structural class for the
autoregressively seeded block head, and a three-feature pairwise bias; two pre-norm transformer layers (192-d,
six heads) feed a directed bilinear readout. About 1.9M parameters.

## Training

**States.** Training-side problems from GSM8K-train, MATH-train, OpenCoder and MBPP-train are decoded with the
frozen backbone and snapshotted along the decode, under the model's confidence-greedy decode and
confidence-greedy-to-budget set decodes. States are harvested at block lengths 8, 32 and 128, stratified over
problems; **block length 64 is deliberately never harvested** and serves as the held-out generalisation
geometry. The harvest holds 24,167 states. Splits are by problem; no test split of any benchmark enters the
harvest.

**Labels.** For each state, up to 24 single plants (the 16 most confident eligible positions plus up to 8 drawn
at random) each commit one source and re-run the current block against the cached prefix, giving the exact
finite-difference column at every eligible receiver. The loss is evaluated only on these labelled columns.

**Objective.** Huber loss (δ = 0.2) on the labelled edges; AdamW, learning rate 3e-4 with warm-up and cosine
decay to 3e-5, weight decay 0.01, gradient clipping at 1.0, selection by validation per-edge correlation. The
shipped checkpoint is the epoch-7 model, validation per-edge correlation 0.794.

## Held-out quality

Pearson correlation between the exact quantity and the adapter's estimate on held-out test states at block
length 64, the geometry never seen in training (paper, Figure 6):

| | GSM8K | MATH-500 | MBPP | HumanEval |
|---|---|---|---|---|
| per-edge finite difference $\hat{\mathbf D}_{ij}$ vs $\mathbf D_{ij}$ | 0.78 | 0.79 | 0.83 | 0.80 |
| aggregate coupling $\hat\mu_i(S)$ vs $\mu_i(S)$ | 0.92 | 0.93 | 0.93 | 0.93 |
| local curvature over the next 32 candidates (median over set sizes) | 0.67 | 0.63 | 0.66 | 0.74 |

## The calibration tables

Neither table is a model: each is a binned average of labelled commits, frozen once and read by lookup.

**Labels.** Training-split problems from GSM8K-train and OpenCoder (80 per corpus) are decoded one token per
round in decreasing confidence over a sliding 32-token window; every sixteenth commit, about thirty candidate
sets are assembled from six samplers (confidence-greedy to a coupling cut, top-$K$ and bottom-$K$ by
confidence, random sets, greedy max-edge sets, mixed sets) and each is replayed sequentially from that canvas.
Every member yields one row: its confidence and entropy, its aggregate coupling to the members ahead of it, and
whether the parallel commit survived the replay. Both tables are fitted on the same probe.

| table | rows (commits) | mismatches | problems | row axis bins |
|---|---|---|---|---|
| `flip_confidence.npz` | 601,730 | 194,895 | 160 | 0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 1 |
| `flip_entropy.npz` | 601,730 | 194,895 | 160 | 0, 0.1, 0.25, 0.5, 1, 1.75, 2.5, 3.5, 4.5, 6, ∞ (nats) |

The five coupling bins are cut at the 20th, 40th, 60th and 80th percentiles of the positive couplings seen in the
probe, with the top edge at the 99.9th percentile (`mu_edges`); a cell holding fewer than 50 commits falls back
to the pooled rate of its row. The cell value is the raw mismatch frequency, with no smoothing or parametric fit.
Held out by problem, the entropy table has a calibration slope of 1.00 (Brier 0.129) and the confidence table a
Brier score of 0.120 on the same rows. The coupling edges and the mismatch levels differ from the LLaDA tables
(Nemotron's edge magnitudes are smaller and it mismatches more at low confidence), which is why each backbone
is decoded with its own tables.

## Intended use and caveats

- The adapter and the tables are fitted to this backbone at the pinned revision. A different checkpoint, or a
  different adapter, needs refitted tables: the coupling bins are quantiles of this adapter's own predictions.
- The tables were fitted on sliding 32-token-window states and are applied unchanged at every block size and
  geometry in the paper; that transfer is assumed, not measured.
- Greedy (argmax) decoding throughout; the reference the decoders match is greedy sequential decoding.
- These are research artefacts for accelerating Nemotron-Labs-Diffusion-3B. They add no safety tuning and
  inherit the backbone's behaviour and limitations.

## License

The files in this repository are released under the MIT License.

They were fitted on the hidden states and predictive distributions of Nemotron-Labs-Diffusion-3B, which is
released by NVIDIA Corporation under the
[NVIDIA Nemotron Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-nemotron-open-model-license/)
and is not redistributed here; use of the backbone is governed by that licence. NVIDIA's names and marks are
used only to identify the model.

## Citation

```bibtex
@misc{fruytier2026pig,
  title  = {{PIG}: Parallel Inconsistency Guidance for Accelerating Diffusion Language Models},
  author = {Fruytier, Quentin and Xue, Anton and Shakkottai, Sanjay and Sanghavi, Sujay},
  year   = {2026},
  note   = {arXiv preprint; identifier to follow}
}
```
