# Reproducing the paper's frontiers

Everything here runs from the repository root with the `experiments` extra installed
(`pip install -e ".[experiments]"`, which adds `datasets` and `matplotlib`). The library itself
(`pig/`) does not need any of it.

## Commands

```bash
# one operating point -> one result file
python -m experiments.evaluate --task gsm8k --model llada --rule pig.int --phi 0.1 --geometry block --block 64 \
    --out results/llada/gsm8k/llada_gsm8k_block64_pig.int_phi0.1.json

# a whole frontier (the model is loaded once; finished cells are skipped, so a sweep resumes where it stopped)
python -m experiments.sweep experiments/configs/gsm8k_llada.json            # -> results/llada/gsm8k/*.json
python -m experiments.sweep experiments/configs/humaneval_nemotron.json --only block32
python -m experiments.sweep experiments/configs/gsm8k_nemotron.json --list   # print the cells, run nothing

# tables and curves from the result files
python -m experiments.summarize results/llada/gsm8k results/nemotron/gsm8k   # markdown (--latex for LaTeX rows)
python -m experiments.plot results/llada/gsm8k --out frontiers_llada_gsm8k.png
```

`--rule` is one of `pig.int` (knob `--phi`), `pig.ext` (`--theta`, optional `--floor`, `--window`) and
`confidence` (`--nu`, the Fast-dLLM rule). `--task` is `humaneval` or `gsm8k`; `--n` defaults to the
paper's problem count. `--device cuda:1` picks a GPU.

## Configs

`configs/<task>_<model>.json` lists, per decoding geometry, the operating points of the paper's
frontier tables: the confidence-threshold ladder (`"confidence"`), the PIG.INT φ ladder (`"pig.int"`)
and the PIG.EXT θ ladder (`"pig.ext"`), whose two slowest points also floor admission at a
confidence ν and are written `[theta, floor]`.

## Protocol

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

Accuracies carry Wilson 95% intervals. Results shift slightly between GPU types, because the
backbone's floating-point results differ across hardware.

## Result files

One JSON per operating point: `config` (model, backbone revision, task, shots, rule and knobs,
geometry, the SHA-256 of the adapter and table used, library versions, GPU), `summary` (n, the
reported metric with its Wilson interval, every recorded metric, TPF and its definition, mean
forwards, timing) and `problems` (per problem: id, scores, forwards, generated tokens, completion).

## Layout

```
tasks/humaneval.py   prompts and graders            tasks/gsm8k.py   prompts and extractors
tasks/__init__.py    the task interface             stats.py         Wilson intervals
evaluate.py          one operating point            sweep.py         a config's grid
summarize.py         tables                         plot.py          frontier curves
configs/             the paper's operating points
```
