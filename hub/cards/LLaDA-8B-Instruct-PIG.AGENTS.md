# AGENTS.md: UT-IFML/LLaDA-8B-Instruct-PIG

Machine-oriented reference for this repository. The human card is README.md; the code repository is
https://github.com/Quentin-Fruytier/PIG-DLM-release (library `pig/`, harness `experiments/`, this build in `hub/`).

## Identity

- Repository: `UT-IFML/LLaDA-8B-Instruct-PIG`; MIT (LLaDA-8B-Instruct and the PIG files); family `llada`.
- Backbone: `GSAI-ML/LLaDA-8B-Instruct` at revision `08b83a6feb34df1a6011b80c3c00c7563e963b07`; its weights, tokenizer and modeling code are copied into this repository (NOTICE.md).
- Paper: PIG: Parallel Inconsistency Guidance for Accelerating Diffusion Language Models (Fruytier, Xue, Shakkottai, Sanghavi, 2026); arXiv:2610.17375 (https://arxiv.org/abs/2610.17375).
- Environment: `transformers>=5.19` (one environment serves both PIG models), `torch` (tested 2.11+cu128), `safetensors`, `numpy`; a CUDA GPU with 24 GB; about a minute to load.

## Files

| file | contents |
|---|---|
| `model-0000i-of-0000n.safetensors`, `model.safetensors.index.json` | one state dict: the backbone (keys `backbone.*`, bf16, copied tensor for tensor from the upstream revision), the adapter (55 tensors, 2,165,527 parameters, fp32, keys `fde.*`) and the two calibration tables (keys `tables.<axis>.<field>` for axis in `confidence`, `entropy` and field in `x_edges`, `mu_edges`, `rates`, `rows`, `flips`, in the last shard) |
| `config.json` | `family`, `backbone` and `backbone_revision` (provenance), `backbone_config` (the backbone's own config), `defaults` (the `generate` defaults), `tables` (the axes stored in the weights), `adapter` and `table_provenance` (sources, counts, package commit), `auto_map`, `dtype` |
| `configuration_llada.py`, `modeling_llada.py` | the backbone's modeling code, copied from the upstream revision with the changes marked `# PIG:` |
| tokenizer files | the backbone's tokenizer, copied from the upstream revision |
| `LICENSE`, `NOTICE.md` | `LICENSE` is the MIT License as declared for LLaDA-8B-Instruct (it also covers the PIG files); `NOTICE.md` states provenance |
| `configuration_pig.py` | `PIGConfig(PretrainedConfig)` |
| `modeling_pig.py` | one file, built by `python -m hub.build` from the code repository: sections `pig/rules.py` (selection rules), `pig/tables.py` (`FlipTable`), `pig/fde/*` (the adapter, `PIGFDE`), `pig/llada.py` (the decode loop, `PIGDecoder`, `PIGGeometry`), `pig/api.py` (`PIG`, `Output`, `make_rule`, `DEFAULTS`) and the wrapper `PIGModel` |

## API

```python
model = AutoModel.from_pretrained("UT-IFML/LLaDA-8B-Instruct-PIG", trust_remote_code=True, device=None, attach=True)
out = model.generate(prompt, mode=None, phi=None, theta=None, floor=None, nu=None, window=None,
                     geometry=None, block=None, max_new_tokens=None, chat_template=True)
```

- `from_pretrained` loads the whole model (backbone, adapter, tables) and moves it to `device` (default: first CUDA device, else CPU), binding the decode loop to the tokenizer shipped here. `attach=False` skips the move and the binding; `model.attach(device)` does them later. Hub kwargs (`cache_dir`, `revision`, `token`, ...) pass through. The adapter stays fp32 whatever dtype the backbone takes.
- `prompt`: a string (wrapped in the backbone's chat template unless `chat_template=False`) or an iterable of token ids.
- `mode`: `"pig.int"` | `"pig.ext"` | `"confidence"`. Knobs are validated per mode and raise `ValueError` otherwise: `pig.int` takes `phi` only; `pig.ext` takes `theta` and optionally `floor` and `window`; `confidence` takes `nu` only. Unset knobs come from `config.defaults`.
- `phi` (float > 0): PIG.INT single-round regret budget, the predicted number of sequential-parallel mismatches allowed per round. Paper ladder 0.015 to 3. Default 0.1.
- `theta` (float > 0): PIG.EXT cross-round regret budget. Paper ladder 0.125 to 3. Default 0.5.
- `floor` (float in [0, 1] or None): PIG.EXT confidence floor on admission. Paper: `theta=0.125, floor=0.9` and `theta=0.25, floor=0.8` as the two slowest points, no floor above. Default None.
- `window` (int): PIG.EXT look-ahead size |N(S)|. Default 16.
- `nu` (float in [0, 1]): confidence threshold of the baseline rule. Default 0.9.
- `geometry` (`"full"` | `"block"` | `"sliding"`), `block` (int): the candidate geometry. Default `"block"`, 32.
- `max_new_tokens`: decodes a 256-token canvas after the prompt, always filled (TPF = 256 / forward passes).
- Returns `PIGOutput(text, tokens, forwards, tpf, mode, params, ms_per_round)`.
- Also: `model.rule(mode, ...)` (the rule object without decoding), `model.encode(prompt)`, `model.backbone`, `model.tokenizer`, `model.fde` (the adapter module), `model.backbone` (the backbone module), `model.tables` (`{"confidence": FlipTable, "entropy": FlipTable}`).
- Decoding is greedy (argmax) throughout; the reference the rules match is greedy sequential decoding. The wrapper and the code repository's `pig.load("llada")` decode identically (checked token for token under all three modes by `python -m hub.verify`).

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
| confidence table | 792,841 | 268,849 | 256 (128 per corpus) | 0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 1 |
| entropy table | 395,854 | 132,943 | 128 (64 per corpus) | 0, 0.1, 0.25, 0.5, 1, 1.75, 2.5, 3.5, 4.5, 6, ∞ (nats) |

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

## Rebuilding

In the code repository: `python -m hub.build` writes this repository's files from `pig/`, `hub/` and
`assets/`; `python -m hub.verify --family llada` checks the build against the library on a GPU;
`python -m hub.upload --family llada` pushes it. `python -m pytest` checks the build on the CPU.

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
