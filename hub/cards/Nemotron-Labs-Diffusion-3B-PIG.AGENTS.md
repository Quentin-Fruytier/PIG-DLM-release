# AGENTS.md: UT-IFML/Nemotron-Labs-Diffusion-3B-PIG

Machine-oriented reference for this repository. The human card is README.md; the code repository is
https://github.com/Quentin-Fruytier/PIG-DLM-release (library `pig/`, harness `experiments/`, this build in `hub/`).

## Identity

- Repository: `UT-IFML/Nemotron-Labs-Diffusion-3B-PIG`; license MIT (files here); family `nemotron`.
- Backbone: `nvidia/Nemotron-Labs-Diffusion-3B` at revision `0d51902da1f8869f83413ce642fab402fa5641e0`, loaded from its own repository at runtime; not redistributed here.
- Paper: PIG: Parallel Inconsistency Guidance for Accelerating Diffusion Language Models (Fruytier, Xue, Shakkottai, Sanghavi, 2026); arXiv:2610.17375 (https://arxiv.org/abs/2610.17375).
- Environment: `transformers==4.56.2` (the model's remote code requires >= 4.53), `torch` (tested 2.11+cu128), `safetensors`, `numpy`, `accelerate`; a CUDA GPU with 8 GB.

## Files

| file | contents |
|---|---|
| `model.safetensors` | the adapter (55 tensors, 1,903,383 parameters, fp32, keys `fde.*`; header metadata `epoch`, `val_edge_corr`) and the two calibration tables (keys `tables.<axis>.<field>` for axis in `confidence`, `entropy` and field in `x_edges`, `mu_edges`, `rates`, `rows`, `flips`) |
| `config.json` | `family`, `backbone`, `backbone_revision`, `defaults` (the `generate` defaults), `tables` (the axes stored in the weights), `adapter` and `table_provenance` (sources, counts, package commit), `auto_map` |
| `configuration_pig.py` | `PIGConfig(PretrainedConfig)` |
| `modeling_pig.py` | one file, built by `python -m hub.build` from the code repository: sections `pig/rules.py` (selection rules), `pig/tables.py` (`FlipTable`), `pig/fde/*` (the adapter, `PIGFDE`), `pig/nemotron.py` (the decode loop, `PIGDecoder`, `PIGGeometry`), `pig/api.py` (`PIG`, `Output`, `make_rule`, `DEFAULTS`) and the wrapper `PIGModel` |

## API

```python
model = AutoModel.from_pretrained("UT-IFML/Nemotron-Labs-Diffusion-3B-PIG", trust_remote_code=True, device=None, load_backbone=True)
out = model.generate(prompt, mode=None, phi=None, theta=None, floor=None, nu=None, window=None,
                     geometry=None, block=None, max_new_tokens=None, chat_template=True)
```

- `from_pretrained` loads the adapter and tables, then the backbone onto `device` (default: first CUDA device, else CPU). `load_backbone=False` defers it; `model.attach_backbone(device)` loads it later. Hub kwargs (`cache_dir`, `revision`, `token`, ...) pass through.
- `prompt`: a string (wrapped in the backbone's chat template unless `chat_template=False`) or an iterable of token ids.
- `mode`: `"pig.int"` | `"pig.ext"` | `"confidence"`. Knobs are validated per mode and raise `ValueError` otherwise: `pig.int` takes `phi` only; `pig.ext` takes `theta` and optionally `floor` and `window`; `confidence` takes `nu` only. Unset knobs come from `config.defaults`.
- `phi` (float > 0): PIG.INT single-round regret budget, the predicted number of sequential-parallel mismatches allowed per round. Paper ladder 0.015 to 3. Default 0.1.
- `theta` (float > 0): PIG.EXT cross-round regret budget. Paper ladder 0.125 to 3. Default 0.5.
- `floor` (float in [0, 1] or None): PIG.EXT confidence floor on admission. Paper: `theta=0.125, floor=0.9` and `theta=0.25, floor=0.8` as the two slowest points, no floor above. Default None.
- `window` (int): PIG.EXT look-ahead size |N(S)|. Default 16.
- `nu` (float in [0, 1]): confidence threshold of the baseline rule. Default 0.9.
- `geometry` (`"block"` | `"sliding"` | `"nested"`), `block` (int): the candidate geometry. Default `"block"`, 32.
- `max_new_tokens`: decodes up to `max_new_tokens` (2048) after the prompt, stopping at the end-of-sequence token (TPF = generated tokens / forward passes; forward passes count the prompt pass, every window forward and every cache update).
- Returns `PIGOutput(text, tokens, forwards, tpf, mode, params, ms_per_round)`.
- Also: `model.rule(mode, ...)` (the rule object without decoding), `model.encode(prompt)`, `model.backbone`, `model.tokenizer`, `model.fde` (the adapter module), `model.tables` (`{"confidence": FlipTable, "entropy": FlipTable}`).
- Decoding is greedy (argmax) throughout; the reference the rules match is greedy sequential decoding. The wrapper and the code repository's `pig.load("nemotron")` decode identically (checked token for token under all three modes by `python -m hub.verify`).

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
| confidence table | 601,730 | 194,895 | 160 | 0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 1 |
| entropy table | 601,730 | 194,895 | 160 | 0, 0.1, 0.25, 0.5, 1, 1.75, 2.5, 3.5, 4.5, 6, ∞ (nats) |

The five coupling bins are cut at the 20th, 40th, 60th and 80th percentiles of the positive couplings seen in the
probe, with the top edge at the 99.9th percentile (`mu_edges`); a cell holding fewer than 50 commits falls back
to the pooled rate of its row. The cell value is the raw mismatch frequency, with no smoothing or parametric fit.
Held out by problem, the entropy table has a calibration slope of 1.00 (Brier 0.129) and the confidence table a
Brier score of 0.120 on the same rows. The coupling edges and the mismatch levels differ from the LLaDA tables
(Nemotron's edge magnitudes are smaller and it mismatches more at low confidence), which is why each backbone
is decoded with its own tables.

## Intended use and caveats

- The adapter and the tables are fitted to the instruct model at the pinned revision. A different checkpoint
  (including `Nemotron-Labs-Diffusion-3B-Base`), or a different adapter, needs refitted tables: the coupling
  bins are quantiles of this adapter's own predictions.
- The tables were fitted on sliding 32-token-window states and are applied unchanged at every block size and
  geometry in the paper; that transfer is assumed, not measured.
- Greedy (argmax) decoding throughout; the reference the decoders match is greedy sequential decoding.
- These are research artefacts for accelerating the Nemotron-Labs-Diffusion-3B instruct model. They add no
  safety tuning and inherit the backbone's behaviour and limitations.

## Rebuilding

In the code repository: `python -m hub.build` writes this repository's files from `pig/`, `hub/` and
`assets/`; `python -m hub.verify --family nemotron` checks the build against the library on a GPU;
`python -m hub.upload --family nemotron` pushes it. `python -m pytest` checks the build on the CPU.

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
