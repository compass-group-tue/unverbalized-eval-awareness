# Training

This package contains the post-training code used for the special-token
experiments. It is intentionally independent of the generated documents,
model checkpoints, provider credentials, cluster submission wrappers, and
experiment outputs; those inputs must be supplied by the caller.

## Layout

- `data/format_datasets.py` converts SDF documents (`content`) and optional
  external SFT corpora into the unified training layout. Its output is
  `sdf/*.jsonl` records with `text` and `sft/*.jsonl` records with `messages`.
- `data/validate_datasets.py` validates that formatted files satisfy that
  schema, including duplicate IDs across files, before training begins.
- `sdf_lora/train.py` is the generic LoRA trainer. With `--sdf-only`, it
  trains solely on the formatted SDF records; without it, it uses the mixed
  SDF+SFT recipe.
- `data/prepare_dpo_pairs.py` reconstructs and audits full preference
  trajectories from a filtered editor result and its referenced source-output
  JSONLs.
- `data/build_marker_contrast.py` creates the natural plus marker-contrast DPO
  view, and `data/split_sft_dpo.py` selects a disjoint chosen-SFT subset and
  DPO complement.
- `sft/train_chosen.py` trains LoRA on those selected chosen trajectories.
- `dpo/train.py` precomputes/attaches frozen reference log-probabilities and
  then trains the trajectory-aware DPO objective. `dpo/kto.py` and
  `dpo/td_alignment.py` are its optional KTO and TD-DPO objective components;
  neither was used for the paper model.

`common/model_profiles.py` contains only architecture-sensitive defaults. A
model checkpoint is never encoded in the package: every trainer requires
`--model-path` explicitly.

## Boundary with SDF

`sdf` generates raw document JSONL records, where the document body is in
`content`. Training begins after `format_datasets.py` converts those records
to the `text` field expected by `sdf_lora/train.py`. For an SDF-only run, the
minimal bridge is:

```bash
PYTHONPATH=src python -m training.data.format_datasets \
  --output-root path/to/training_data \
  --sdf-path path/to/synthetic_documents.jsonl \
  --no-include-sft

PYTHONPATH=src python -m training.data.validate_datasets \
  --input-root path/to/training_data \
  --output-dir path/to/training_data_validation

torchrun --standalone --nproc_per_node=NUM_GPUS -m training.sdf_lora.train \
  --run-mode full \
  --sdf-only \
  --model-profile qwen3_5_27b \
  --model-path path/to/base_model \
  --train-data-root path/to/training_data \
  --output-dir path/to/sdf_adapter
```

To create the mixed layout, also provide `--am-dir`, `--smoltalk-dir`, and
`--dolci-dir` to `format_datasets.py`. These are external corpora, not release
assets. The generic trainer's old question-evaluation callback is opt-in and
requires an explicit `--questions-jsonl` and `--eval-script-path`; evaluation
code itself is intentionally not part of this package.

## SFT to DPO sequence

The preference path takes the following immutable inputs and creates a fresh
output directory at each stage:

1. Run `prepare_dpo_pairs.py` with a filtered editor JSONL and the model used
   to render/audit its complete trajectories. The filtered rows identify the
   source-output JSONLs they need to reattach; those source files must remain
   available at the paths recorded in the filtered input.
2. Run `build_marker_contrast.py` over the prepared train and validation
   views, then `split_sft_dpo.py` over the original and contrast views. The
   latter writes `sft/` and `dpo/` children and records the disjoint-split
   checks in its manifest.
3. Train the selected SFT set with `sft.train_chosen`. For DPO, use
   `dpo.train --mode precompute-reference` for every shard, concatenate or
   otherwise supply those reference JSONLs to `--mode attach-reference`, then
   use `--mode train` on the attached result. The DPO trainer validates that
   rendering and objective metadata match the reference cache.

Run every entry point as a module with `PYTHONPATH=src`; pass `--help` for its
full, explicit interface. The old scheduler-specific submission scripts and
historical training variants deliberately remain outside this package until
the final clean-up pass, and are not dependencies of these entry points.

Defaults reproduce the paper model:
- **SDF LoRA:** learning rate 5e-5.
- **SFT:** all-assistant loss, learning rate 1e-5.
- **DPO:** content-token DPO loss, beta 0.1, plus a 0.05 all-assistant chosen NLL, learning rate 1e-5.

SFT, reference scoring and DPO default to the `trl_qwen3_5_training` renderer,
which requires `trl==1.9.0` (install with `--no-deps`; `dpo.train` checks the
version). `prepare_dpo_pairs.py` keeps the `legacy_prefix_delta` renderer for its
length audit, as the paper's pairs were prepared with it. Pass
`--chat-rendering legacy_prefix_delta` to the trainers to reproduce pre-V13 runs.
