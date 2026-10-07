# Third-party code and data

The code in this repository is released under the MIT License (`../LICENSE`),
except as noted below.

- **`instruction_following_eval/`** is Google's official IFEval evaluator
  (Zhou et al., 2023), vendored unmodified from
  `google-research/google-research` (2024 version). It is licensed under the
  Apache License 2.0; see `instruction_following_eval/LICENSE` and the notice in
  each file.
- **`src/sdf/`** (document-generation pipeline and prompts) is adapted from the
  synthetic document fine-tuning code of Wang et al. (2025),
  `safety-research/false-facts`, with modifications for Azure OpenAI batch
  generation, spec top-up and output repair.
- **Benchmarks and datasets** are not included. The preparers download each
  benchmark from its original source at a pinned version; each remains under
  its own licence and terms of use.
- **Models** (Qwen3.5-27B and derived checkpoints) are not included and remain
  under their own licence.
