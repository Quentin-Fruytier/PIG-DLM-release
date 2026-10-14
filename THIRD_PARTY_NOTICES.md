# Third-party notices

Everything in this repository is released under the MIT License (`LICENSE`), except the backbones' modeling
code in `pig/backbones/`, which keeps its upstream license:

- `pig/backbones/llada/`: copied from `GSAI-ML/LLaDA-8B-Instruct` (revision 08b83a6f), released by its authors
  under the MIT License (`pig/backbones/llada/LICENSE`).
- `pig/backbones/nemotron/`: copied from `nvidia/Nemotron-Labs-Diffusion-3B` (revision 0d51902d). Licensed by
  NVIDIA Corporation under the NVIDIA Nemotron Model License; the full text is in
  `pig/backbones/nemotron/LICENSE` and the notice in `pig/backbones/nemotron/NOTICE`. The configuration file
  also keeps its Apache License 2.0 header.

The backbones' weights are not in this repository; `pig.load` downloads them from the Hugging Face Hub at the
pinned revisions. The adapters and tables in `assets/` (MIT) were fitted on each backbone's outputs. The two
Hugging Face repositories redistribute each backbone's weights, tokenizer and code with the same license files
and notices. NVIDIA's trade names and trademarks are used only to identify the model.
