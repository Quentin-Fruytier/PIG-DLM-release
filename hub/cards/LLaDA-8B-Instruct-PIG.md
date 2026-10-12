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

The trained components of **Parallel Inconsistency Guidance (PIG)** for
[LLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct): a ~2.2M-parameter adapter that
predicts, from the one forward pass the decoder already runs, how committing any masked position would change
the backbone's uncertainty at every other masked position, and two calibration tables that turn those
predictions into the probability that a parallel commit disagrees with sequential decoding. The PIG.INT and
PIG.EXT decoders read them to decide how many positions to commit per forward pass, and extend
LLaDA-8B-Instruct's accuracy–throughput frontier over confidence-threshold decoding (Fast-dLLM) on GSM8K,
MATH-500, MBPP and HumanEval.

This repository holds the trained components together with a `transformers` wrapper that loads the backbone
and runs the decoders, so the model is usable with `AutoModel.from_pretrained` and one `generate` call. The
evaluation harness and the paper's operating points are in the code repository.

**Paper:** *PIG: Parallel Inconsistency Guidance for Accelerating Diffusion Language Models* — Quentin Fruytier,
Anton Xue, Sanjay Shakkottai, Sujay Sanghavi (The University of Texas at Austin), 2026. arXiv link to follow.  
**Code:** https://github.com/Quentin-Fruytier/PIG-DLM-release  
**Companion:** [UT-IFML/Nemotron-Labs-Diffusion-3B-PIG](https://huggingface.co/UT-IFML/Nemotron-Labs-Diffusion-3B-PIG),
the same two components fitted for Nemotron-Labs-Diffusion-3B (NVIDIA's instruct model).

## Files

| file | what it is |
|---|---|
| `model.safetensors` | the Entropy Finite Differences Estimator (the adapter): 55 tensors, 2,165,527 parameters, fp32, keys prefixed `fde.`; the header records `epoch = 7` and `val_edge_corr = 0.8268` |
| `flip_confidence.npz` | the mismatch table $\hat p_\nu(\nu, \hat\mu)$ read by PIG.INT: 8 confidence bins × 5 coupling bins |
| `flip_entropy.npz` | the mismatch table $\hat p_H(H, \hat\mu)$ read by PIG.EXT: 10 entropy bins × 5 coupling bins |
| `config.json` | the backbone id and pinned revision, the decoding defaults, adapter and table provenance |
| `modeling_pig.py`, `configuration_pig.py` | the wrapper: `from_pretrained` and `generate` |
| `decoder.py`, `rules.py`, `tables.py`, `fde.py` | the decode loop, the selection rules, the table lookup and the adapter architecture, vendored verbatim from the code repository (each file names the commit it came from) |

Each `.npz` holds `x_edges` (bin edges on the confidence or entropy axis), `mu_edges` (bin edges on the
coupling axis), `rates` (the empirical mismatch rate of each cell), `rows` and `flips` (the commit and mismatch
counts behind each cell), `axis` and `description`.

The backbone is not redistributed here. The wrapper loads `GSAI-ML/LLaDA-8B-Instruct` at the pinned revision
`08b83a6f` from its own repository.

## Use

Needs a CUDA GPU with at least 24 GB of memory and `transformers==4.49.0` (LLaDA's own remote code requires the
4.x line; tested with `torch 2.11+cu128`), plus `safetensors` and `numpy`. Nothing else to install: the code
ships in this repository and is loaded with `trust_remote_code=True`.

```python
from transformers import AutoModel

model = AutoModel.from_pretrained("UT-IFML/LLaDA-8B-Instruct-PIG", trust_remote_code=True)
out = model.generate("Write a Python function that returns the n-th Fibonacci number.", mode="pig.int", phi=0.1)
print(out.text)
print(out.forwards, "forward passes;", round(out.tpf, 2), "tokens per forward")
```

`from_pretrained` loads the adapter and the tables from this repository, then the backbone from its own
repository (about a minute; 16 GB of GPU memory), onto the first CUDA device. `device="cuda:1"` or
`device="cpu"` picks another; `load_backbone=False` defers the backbone, to be attached later with
`model.attach_backbone()`.

`generate(prompt, mode, ...)` decodes a 256-token canvas after the prompt, which is wrapped in the chat
template (pass token ids, or `chat_template=False`, to skip it), and returns `text`, `tokens`, `forwards`
and `tpf`, the tokens per forward pass that the paper uses as its throughput measure.

| `mode` | rule | knobs |
|---|---|---|
| `"pig.int"` (default) | PIG.INT: the single-round regret budget | `phi` (default 0.1) |
| `"pig.ext"` | PIG.EXT: the cross-round regret budget | `theta` (default 0.5); `floor`, a confidence floor on admission (default none); `window`, the look-ahead (default 16) |
| `"confidence"` | the confidence-threshold baseline (Fast-dLLM) | `nu` (default 0.9) |

`geometry` and `block` set the candidate geometry: `"block"` (default; semi-autoregressive blocks of
`block` = 32 tokens), `"sliding"`, or `"full"` for the whole canvas. The paper's main operating points use
`block=32` on the math benchmarks and `block=64` on code; its two slowest PIG.EXT points are
`theta=0.125, floor=0.9` and `theta=0.25, floor=0.8`.

The code repository reproduces the paper's HumanEval operating points from the same files
(`python -m pig.evaluate`, `python -m pig.sweep`). The wrapper and that harness decode identically: checked
token for token, and forward pass for forward pass, on HumanEval problems under all three modes.

## What the adapter predicts

For a canvas with masked positions $i, j$, the target is the directed entropy finite difference

$$\hat{\mathbf D}_{ij} \approx H_i(\mathbf x) - H_i(\mathbf x \oplus \hat x_j),$$

the change in the backbone's predictive entropy at $i$ if $j$ were committed to its current argmax. One forward
pass of the frozen backbone with $j$ committed measures a single column of this matrix exactly; the adapter
returns the whole matrix from the forward pass the decoder already runs. PIG.INT sums the magnitudes of these
edges from the accepted prefix into a coupling $\hat\mu_i$ and reads $\hat p_\nu(\nu_i, \hat\mu_i)$; PIG.EXT
uses their sign to predict next-round entropies and reads $\hat p_H$ at the current and predicted entropy.

**Inputs, per canvas position, all from the frozen backbone:** the post-norm hidden state (4096-d, read by a
forward hook on the final normalisation layer), the canvas token id, the attention maps of the current and
previous round (256 × 256 each), and nine scalars (entropy, confidence, top-1/top-2 margin, a decoded flag, a
three-way structural class, the decode fraction $t$). The canvas length is fixed at 256.

**Architecture (~2.2M parameters):** per-position embeddings (4096 → 192 for the hidden state, 4096 → 64 for the
token embedding, 9 → 32 for the scalars) fused by a linear layer with LayerNorm into a 192-d state; two
pre-norm transformer layers over the 256 positions (six heads of 32, rotary positions, GELU feed-forward of
width 768), each FiLM-modulated by $t$; a pairwise attention-bias MLP on the two attention maps and the
log-distance of each pair; a directed bilinear readout $f(z_i)^\top W g(z_j)/\sqrt{96} + b_{ij}$ with separate
receiver and source maps, computed for all pairs at once. At canvas length 256 it costs about 0.6 GMAC per
decoding round, about 13 ms on a shared H100.

## Training

**States.** Training-side problems are decoded with the frozen backbone and the canvas is snapshotted every 24
commits, under two policies: the model's own one-token-per-round confidence-greedy decode and
confidence-greedy-to-budget set decodes, each on the free canvas and with eligibility restricted to the leftmost
64 positions. No test split of any benchmark enters the harvest.

| corpus | problems | states |
|---|---|---|
| GSM8K (train) | 1,111 | 9,912 |
| MATH (train) | 1,230 | 11,026 |
| OpenCoder | 1,110 | 9,983 |
| MBPP (train) | 270 | 3,221 |
| total | 3,721 | 34,142 |

Splits are by problem (hash of corpus and problem id modulo 10: 0–7 train, 8 validation, 9 internal test;
27,231 / 3,468 / 3,443 states).

**Labels.** For each state, 24 single plants (the 16 most confident eligible positions and 8 drawn at random)
each commit one source and re-run the backbone, giving the exact finite-difference column at every eligible
receiver: about 3,300 labelled directed edges per state, about 111M in total. The loss is evaluated only on this
labelled block, never on the full 256 × 256 output.

**Objective.** Huber loss (δ = 0.2) on the labelled edges, unweighted; AdamW, learning rate 3e-4 with warm-up
and cosine decay to 3e-5, weight decay 0.01, gradient clipping at 1.0, 16 states per optimiser step, ten epochs
configured with selection by validation per-edge correlation. The shipped checkpoint is the epoch-7 model,
validation per-edge correlation 0.827.

## Held-out quality

Pearson correlation between the exact quantity and the adapter's estimate on held-out test states of the four
evaluation benchmarks (paper, Figure 6):

| | GSM8K | MATH-500 | MBPP | HumanEval |
|---|---|---|---|---|
| per-edge finite difference $\hat{\mathbf D}_{ij}$ vs $\mathbf D_{ij}$ | 0.80 | 0.76 | 0.80 | 0.81 |
| aggregate coupling $\hat\mu_i(S)$ vs $\mu_i(S)$ | 0.95 | 0.92 | 0.97 | 0.96 |
| local curvature over the next 32 candidates (median over set sizes) | 0.83 | 0.81 | 0.75 | 0.88 |

The adapter under-shoots edge magnitudes (it recovers 49–59% of the exact edge mass with the correct sign) but
does so uniformly across magnitude bins, which is why the aggregate statistics the decoders consume correlate
more tightly than the individual edges.

## The calibration tables

Neither table is a model: each is a binned average of labelled commits, frozen once and read by lookup.

**Labels.** Training-split problems from GSM8K-train and OpenCoder are decoded one token per round in
decreasing confidence; every sixteenth commit, about thirty candidate sets are assembled from six samplers
(confidence-greedy to a coupling cut, top-$K$ and bottom-$K$ by confidence, random sets, greedy max-edge sets,
mixed sets) and each is replayed sequentially from that canvas. Every member yields one row: its confidence (or
entropy), its aggregate coupling to the members ahead of it, and whether the parallel commit survived the
replay.

| table | rows (commits) | mismatches | problems | row axis bins |
|---|---|---|---|---|
| `flip_confidence.npz` | 792,841 | 268,849 | 256 (128 per corpus) | 0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 1 |
| `flip_entropy.npz` | 395,854 | 132,943 | 128 (64 per corpus) | 0, 0.1, 0.25, 0.5, 1, 1.75, 2.5, 3.5, 4.5, 6, ∞ (nats) |

The five coupling bins are cut at the 20th, 40th, 60th and 80th percentiles of the positive couplings seen in the
probe, with the top edge at the 99.9th percentile (`mu_edges`); a cell holding fewer than 50 commits falls back
to the pooled rate of its row. The cell value is the raw mismatch frequency, with no smoothing or parametric fit.
Held out by problem, the entropy table has a calibration slope of 1.02 (Brier 0.148) and the confidence table a
Brier score of 0.137 on the same rows. Audited during live decoding on the paper's frontier runs, the
confidence table's realized-over-predicted mismatch ratio is 0.83–0.92 per benchmark (it errs on the
conservative side); the entropy table's deferred estimate is calibrated above about 5% but under-predicts in
its tail, where it predicts below 1% and 1–7% of window tokens still change.

## Intended use and caveats

- The adapter and the tables are fitted to this backbone at the pinned revision. A different checkpoint, or a
  different adapter, needs refitted tables: the coupling bins are quantiles of this adapter's own predictions.
- The tables were fitted on free-canvas states and are applied unchanged at every block geometry in the paper;
  that transfer is assumed, not measured.
- Greedy (argmax) decoding throughout; the reference the decoders match is greedy sequential decoding.
- These are research artefacts for accelerating LLaDA-8B-Instruct. They add no safety tuning and inherit the
  backbone's behaviour and limitations.

## License

The files in this repository are released under the MIT License. LLaDA-8B-Instruct is released by its authors
under the MIT License and is not redistributed here.

## Citation

```bibtex
@misc{fruytier2026pig,
  title  = {{PIG}: Parallel Inconsistency Guidance for Accelerating Diffusion Language Models},
  author = {Fruytier, Quentin and Xue, Anton and Shakkottai, Sanjay and Sanghavi, Sujay},
  year   = {2026},
  note   = {arXiv preprint; identifier to follow}
}
```
