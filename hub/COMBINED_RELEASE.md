# Combined Hub repositories (backbone + PIG adapter in one model)

Branch `combined-hub` (local only; never pushed to the public repository until it passes the checks below).

## Goal

`AutoModel.from_pretrained("UT-IFML/LLaDA-8B-Instruct-PIG")` and `...Nemotron-Labs-Diffusion-3B-PIG` load one
model each: the backbone weights, the PIG adapter and the calibration tables, with our own modeling code,
on a current `transformers` (5.x). No second repository, no pinned upstream revision, no 4.x pins, one
environment for both models.

## Shape of each repository

    config.json              PIGConfig with the backbone's config nested (`backbone_config`), the decoding defaults,
                             adapter and table provenance, the upstream id + revision the weights were copied from
    configuration_pig.py
    modeling_pig.py          one file: the ported backbone (LLaDA: the OLMo-style implementation of
                             modeling_llada.py; Nemotron: modeling_nemotron_labs_diffusion.py + modeling_ministral.py),
                             then the PIG sections as today, then the wrapper
    model-0000N-of-0000M.safetensors + model.safetensors.index.json
                             the backbone weights (keys `backbone.*`, copied tensor for tensor from the upstream
                             revision), the adapter (`fde.*`) and the tables (`tables.*`)
    tokenizer files          copied from the upstream revision; chat template included
    LICENSE / NOTICE         LLaDA: MIT (authors' notice). Nemotron: the NVIDIA Open Model License text and the
                             attribution it requires
    README.md, AGENTS.md

In the code repository the ported backbones live in `pig/backbones/<family>/` and are the single source for
both the library (`pig.load`) and the Hub build, like the rest of `pig/` today.

## Status (2026-10-13)

Steps 0-4 done on this branch; the backbones' code lives in `pig/backbones/`, parity vs the upstream remote code is
exact (hub/parity.py, 16 prompts per model), the full build is in /ssd1/qdf76/pig_hub_dist, and the four-dataset
consistency campaign (hub/consistency.py) is the remaining check before upload.

## Checks done so far

- Parity (hub/parity.py, 16 prompts per model, PIG.INT): exact for both models between the upstream remote
  code on transformers 4.x and this package on 5.19 (same tokens and forward counts; zero difference in every
  per-round confidence, entropy and adapter matrix).
- Combined repositories built to /ssd1/qdf76/pig_hub_dist and loaded through AutoModel: 9 of 9 decodes identical
  to the library for each model (hub.verify --repo <dir> --no-build).
- Four-dataset campaign (hub/consistency.py; prior version vs this build vs the paper's tables): running.

## Steps

0. Probe: load both upstream remote codes under transformers 5.x and run the three modes; record what breaks.
1. Port the two modeling files to 5.x in `pig/backbones/`; load the upstream weights into them.
2. Decoders read hidden states and attention from the ported code directly (no hooks).
3. One requirements file (`transformers>=5.x`); the Nemotron overlay goes away; README install = one env.
4. `hub.build` writes the combined repositories (sharded weights, tokenizer, licenses); upload to the
   PRIVATE repositories only after the checks pass.

## Checks (all must pass before anything is uploaded; results stay out of git)

1. Weights: every backbone tensor equal to the upstream pinned revision's, tensor for tensor (torch.equal),
   both models.
2. Logits: the ported modeling code vs the upstream remote code (4.x environment), same GPU, bf16: identical
   logits, hidden states and (LLaDA) final-block attention on 64 fixed canvases per model drawn from HumanEval and
   GSM8K prompts at several decode stages.
3. Decoding: the combined Hub model (5.x) vs the current library (4.x) on HumanEval, all 164 problems, three
   modes, both models: identical token sequences and forward counts. Same on the first 100 GSM8K problems.
4. Paper consistency, full cells on one GPU type, compared to the paper's tables and to the earlier release-code
   runs (/ssd1/qdf76/pig_dlm_validation, /ssd1/qdf76/pig_release_check):
   - HumanEval n=164: LLaDA block64 {pig.int 0.1, pig.ext 0.25/0.8, confidence 0.9};
     Nemotron block32 {same three}
   - GSM8K n=500: LLaDA block64 pig.int 0.1 and confidence 0.85; Nemotron block32 pig.int 0.1 and confidence 0.9
   Accuracy within the Wilson interval of the paper's cell and TPF within a few percent; any larger gap is
   explained before upload.
5. The CPU test suite, and `hub.verify` against the library, as today.
