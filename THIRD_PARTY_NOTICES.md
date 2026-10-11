# Third-party notices

This repository's code, adapters and calibration tables are released under the MIT License (see `LICENSE`).
The adapters and tables were fitted on the outputs of two backbone models, which this repository does not
redistribute; the code downloads them from the Hugging Face Hub at pinned revisions.

- **LLaDA-8B-Instruct** (`GSAI-ML/LLaDA-8B-Instruct`), released under the MIT License.
- **Nemotron-Labs-Diffusion-3B** (`nvidia/Nemotron-Labs-Diffusion-3B`), released by NVIDIA Corporation under the
  NVIDIA Nemotron Open Model License,
  https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-nemotron-open-model-license/.
  `assets/nemotron/` (the adapter `fde.safetensors` and the tables `flip_confidence.npz`, `flip_entropy.npz`)
  was fitted on that model's hidden states and predictive distributions. Use of the backbone itself is governed
  by its own license; NVIDIA's trade names and trademarks are used here only to identify the model.
