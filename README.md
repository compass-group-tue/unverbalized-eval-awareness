# unverbalised-eval-awareness-model-organism
Repository for experiments in “A Model Organism for Studying Unverbalized Evaluation Awareness”.

- `model-development/`: synthetic document generation, SFT/DPO training of the model organism,
  evaluation on all datasets and system-prompt conditions, capability scoring and the
  verbalized-evaluation-awareness judge.
- `interpretability/`: linear probes, activation steering, the Jacobian Lens, Activation Oracles
  and the resampling-robustness check.

Each folder has its own README and environment; run commands from within that folder.

Released artefacts ([Hugging Face collection](https://huggingface.co/collections/compass-group-tue/model-organism-for-studying-unverbalized-eval-awareness)):

- Model organism: [`compass-group-tue/qwen-3.5-uea-mo`](https://huggingface.co/compass-group-tue/qwen-3.5-uea-mo)
- Training data (SDF corpus and the evaluation-side SFT and DPO sets): [`compass-group-tue/uea-mo-datasets`](https://huggingface.co/datasets/compass-group-tue/uea-mo-datasets)

## License

The code is released under the MIT License (see `LICENSE`), except for the third-party components
listed in `model-development/NOTICE.md`.
