# A Model Organism for Studying Unverbalized Evaluation Awareness

Code for training and evaluating the model organism from *A Model Organism for
Studying Unverbalized Evaluation Awareness*. Qwen3.5-27B is fine-tuned with
synthetic documents (SDF), then SFT and DPO, so that it emits a marker (`§`) in
its visible response when it recognizes an evaluation, while not verbalizing
that recognition in its reasoning.

This repository covers the model side of the paper: synthetic document
generation, the three training stages, generation on the 16 evaluation
datasets under every system-prompt condition, capability scoring, marker and
transcript reports, and the verbalized-evaluation-awareness judge. The
interpretability experiments (linear probes, the Jacobian Lens, Activation
Oracles, activation steering, and the resampling-robustness check) are in
`../interpretability/`.

## Released model and data

- **Model organism:** [`compass-group-tue/qwen-3.5-uea-mo`](https://huggingface.co/compass-group-tue/qwen-3.5-uea-mo), the final merged
  checkpoint (SDF → SFT → DPO+NLL on Qwen3.5-27B). Serve it directly, e.g.
  `MODEL_PATH=compass-group-tue/qwen-3.5-uea-mo examples/serve_vllm.sh`.
- **Training data:** [`compass-group-tue/uea-mo-datasets`](https://huggingface.co/datasets/compass-group-tue/uea-mo-datasets), with `sdf` (164,470
  documents), `sft` and `dpo` configurations.

The released SFT and DPO sets are a subset of those used for training (SFT 2,385 /
71, DPO 15,204 / 142, against 5,000 / 200 and 31,644 / 400 in training). Pairs
derived from WildChat, SORRY-Bench, OpenBookQA, AGIEval and PersistBench are not
redistributed because of those sources' licensing or privacy terms; see the
dataset card for details, and rebuild them with section 3. The
released files are in the trainers' input format, so they can be passed to the
section 4 commands in place of `$WORK/split/...`:

```bash
hf download compass-group-tue/uea-mo-datasets --repo-type dataset --local-dir "$WORK/released_data"
```

Nothing else (generated transcripts, judge outputs, credentials) is included.
Every command takes explicit input and output paths.
Run all commands below from this directory.

## Layout

| Path | Contents |
|---|---|
| `src/sdf/` | Universe context, document-generation prompts, Azure OpenAI batch pipeline (specs → documents → rebuild → clean). |
| `src/preference_data/` | Stage-1 rollouts from the SDF model, DeepSeek V4 Pro editing, and filtering of the edited pool. |
| `src/training/` | Dataset formatting, SDF LoRA, pair construction, chosen-trajectory SFT and DPO (+NLL). |
| `src/evaluation/` | Frozen evaluation inputs, generation, capability replay scoring, marker/behaviour reports, qualitative probes. |
| `src/judging/` | Reasoning-only evaluation-awareness judge (inputs, runner, reports) and its scopes. |
| `examples/serve_vllm.sh` | How each merged checkpoint was served for generation. |
| `instruction_following_eval/` | Google's official IFEval evaluator (vendored, Apache-2.0). |
| `tests/` | Unit tests for the training data and objective code. |

Each `src/` package has its own README with more detail.

## Installation

The paper used two environments with different library versions, and the
`training` and `inference` extras pin those versions, so install them into
separate virtual environments (Python 3.11):

```bash
# Training: SDF LoRA, SFT, DPO and adapter merging
uv venv .venv-train --python 3.11
VIRTUAL_ENV=.venv-train uv pip install -e ".[training]"
VIRTUAL_ENV=.venv-train uv pip install --no-deps "trl==1.9.0"

# Inference: data preparation, generation, scoring, reports, and the API-backed stages
uv venv .venv-infer --python 3.11
echo "transformers==5.5.0" > overrides.txt
VIRTUAL_ENV=.venv-infer uv pip install -e ".[inference,ifeval,sdf]" "vllm==0.19.0" --override overrides.txt
```

- **trl:** the `trl` pin supplies the Qwen3.5 training chat template that SFT
  and DPO use. It is installed without dependencies so it cannot move the
  training pins.
- **vLLM:** vLLM 0.19.0 declares `transformers<5`, but the paper ran it with
  transformers 5.5.0, hence the override.
- **API-only stages:** SDF generation, the transcript editor and the judge need
  only the core dependencies (plus `sdf`), and run anywhere.
- **GPUs:** generation needs a Linux GPU host.

Run modules with `python -m <package>.<module>` from an installed package, or
with `PYTHONPATH=src`. Credentials come from the environment:

| Stage | Variables |
|---|---|
| SDF generation (Azure OpenAI batch, GPT-4.1) | `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_API_VERSION` |
| Transcript editor and judge (OpenRouter) | `OPENROUTER_API_KEY` |
| Gated Hugging Face datasets | `HF_TOKEN` |

## Reproducing the paper

The defaults of each command are the settings used for the paper, so the
commands below only pass paths, plus a few per-dataset or per-model settings
spelled out explicitly. Pass `--help` to any module for its full interface.

Throughout:

```bash
BASE=/path/to/Qwen3.5-27B        # base checkpoint
WORK=/path/to/work               # training data and checkpoints
EXP=/path/to/experiments         # evaluation campaigns (layout below)
EVAL_DATA=/path/to/source_data   # downloaded benchmark sources for Stage-1 rollouts
```

The interpretability code in `../interpretability/` reads these campaigns from
`$UEA_MODEL_DATA/experiments`, so set `EXP` to that directory.

Training uses 4 GPUs throughout. The effective batch size is GPUs ×
micro-batch × accumulation: 4 for SDF and SFT, 16 for DPO.

Run the training and merging steps of sections 2 and 4 in `.venv-train`, and
everything else in `.venv-infer`.

### 1. Synthetic documents

The universe context and its 11 key facts are packaged at
`src/sdf/universe_contexts/special_token_output_context_005.jsonl` and used by
default. Generation runs 75 document types × 40 ideas × 1–10 documents per
idea, all as Azure batch jobs.

```bash
RUN=special_token_output_context_005
python -m sdf.submit_batches --run-root "$WORK/sdf" --run-name $RUN --specs-only
python -m sdf.topup_specs    --run-root "$WORK/sdf" --run-name $RUN   # fill any missing type/idea slots
python -m sdf.submit_batches --run-root "$WORK/sdf" --run-name $RUN --submit-from-specs
# after the batches finish:
python -m sdf.collect_batches --run-log "$WORK/sdf/$RUN/document_batches.log" \
  --submit-manifest "$WORK/sdf/$RUN/manifests/submit_manifest.json" \
  --specs-dir "$WORK/sdf/$RUN/specs" --azure-log-dir "$WORK/sdf/$RUN/azure_logs" \
  --out-dir "$WORK/sdf/$RUN/per_fact_docs" --out-prefix $RUN \
  --universe-contexts-path src/sdf/universe_contexts/special_token_output_context_005.jsonl
python -m sdf.repair --run-dir "$WORK/sdf/$RUN" --out-prefix $RUN \
  --universe-contexts-path src/sdf/universe_contexts/special_token_output_context_005.jsonl
python -m sdf.clean --input "$WORK/sdf/$RUN/merged/${RUN}_combined.balanced_content_only.jsonl" \
  --output "$WORK/sdf/$RUN/merged/${RUN}_combined.balanced_content_only.cleaned.jsonl"
```

`repair` keeps only generations with a balanced, non-empty `<content>` block.
`clean` strips any leaked scratchpad or wrapper markup.

### 2. SDF fine-tuning

```bash
python -m training.data.format_datasets --output-root "$WORK/sdf_training_data" --no-include-sft \
  --sdf-path "$WORK/sdf/$RUN/merged/${RUN}_combined.balanced_content_only.cleaned.jsonl"
python -m training.data.validate_datasets --input-root "$WORK/sdf_training_data" \
  --output-dir "$WORK/sdf_training_data/validation"
torchrun --standalone --nproc_per_node=4 -m training.sdf_lora.train \
  --run-mode full --sdf-only --model-path "$BASE" \
  --train-data-root "$WORK/sdf_training_data" --output-dir "$WORK/sdf_lora"
python -m evaluation.inference.merge_adapter --base-model-path "$BASE" \
  --adapter-path "$WORK/sdf_lora" --output-dir "$WORK/sdf_merged"
```

### 3. Preference data

**Sources.** Download the Stage-1 benchmark sources and WildChat into one root:

```bash
STAGE1=gsm8k,bbq,wmdp,openbookqa,agieval,strategyqa,jailbreakbench,persistbench,sorry_bench,mt_bench_101
python -m evaluation.inputs.stage1_sources --out-root "$EVAL_DATA" --datasets $STAGE1
python -m evaluation.inputs.wildchat_source --out-root "$EVAL_DATA"
python -m preference_data.prepare_inputs --job-name stage1 --eval-root "$EVAL_DATA" \
  --out-root "$WORK/stage1" --datasets $STAGE1
python -m preference_data.prepare_inputs --job-name stage1 --eval-root "$EVAL_DATA" \
  --out-root "$WORK/stage1" --datasets wildchat_deployment
```

The evaluation-side "phase 2" rollouts use a second source draw into
`phase2_rlhf_download`, which excludes every row already drawn for Stage 1:

- **Answer format:** single-turn items use a roughly 50/50 mix of strict and
  relaxed answer-format instructions.
- **SORRY-Bench:** drawn separately, with strict formatting and three mutations
  per base prompt, and without the base prompts themselves.

Prepared with `--datasets phase2_eval`, the draw writes under `inputs/phase2/`:

```bash
python -m evaluation.inputs.stage1_sources --out-root "$EVAL_DATA" \
  --datasets gsm8k,bbq,wmdp,openbookqa,agieval,strategyqa,mt_bench_101 \
  --download-subdir phase2_rlhf_download --exclude-selected-from-subdir stage1_rlhf_download \
  --format-style mixed --target-count-overrides gsm8k=1800,wmdp=1200,openbookqa=1200,strategyqa=703,mt_bench_101=588
python -m evaluation.inputs.stage1_sources --out-root "$EVAL_DATA" --datasets sorry_bench \
  --download-subdir phase2_rlhf_download --exclude-selected-from-subdir stage1_rlhf_download \
  --no-sorry-bench-include-base-prompts --sorry-bench-mutations-per-base 3
python -m preference_data.prepare_inputs --job-name stage1 --eval-root "$EVAL_DATA" \
  --out-root "$WORK/stage1" --datasets phase2_eval --source-download-subdir phase2_rlhf_download
```

The paper's draw realised these row counts: GSM8K 1,800, BBQ 1,800, WMDP 908,
OpenBookQA 1,200, AGIEval 342, StrategyQA 703, SORRY-Bench 880 and
MT-Bench-101 588. WMDP and AGIEval fall short of their targets because the
Stage-1 exclusion exhausts their pools. The Stage-1 draw itself used the
downloader's defaults.

**Rollouts.** Serve the SDF-merged model (`examples/serve_vllm.sh` with
`MAX_MODEL_LEN=8192`). Then run `preference_data.run_inspect` for single-turn
datasets, and `preference_data.run_mt_bench` for MT-Bench-101 dialogues, each inside
`examples/serve_vllm.sh` so that `{MODEL_BASE_URL}` is filled in. The
matrix below is exactly what `build_editor_pool` reads. Paths are relative to
`$WORK/stage1/stage1/outputs/`; `<ds>.jsonl` is one file per dataset.

| Datasets | Overlay (`--system-prompt-file`) | Output directory |
|---|---|---|
| gsm8k, bbq, wmdp, openbookqa, agieval, strategyqa | none | `inspect/no_system/full_no_system_v1/` |
| same | `x100.txt` | `inspect/sys_x100/full_sys_x100_v1/` |
| jailbreakbench, persistbench, sorry_bench, mt_bench_101 | none / `x100.txt` | `inspect/{no_system,sys_x100}/remaining_full_{no_system,sys_x100}_v1/` |
| the six above + sorry_bench, mt_bench_101 | `x100_assessment_item.txt`, `x100_eval.txt`, `helpful_exact.txt` | `phase2/inspect/{sys_x100,sys_x100_eval,sys_helpful_exact}/phase2_full_*_v1/` |
| wildchat_deployment | none / `x100.txt` | `inspect/{no_system,sys_x100}/wildchat_deployment_full_{no_system,sys_x100}_v1/` |

The overlays are in `src/preference_data/prompts/system/`. The runner defaults
are the paper's rollout sampler: temperature 1.0, top-p 0.95, top-k 20, presence
penalty 0, repetition penalty 1.15, thinking on, and 4,096 new tokens. For
example:

```bash
python -m preference_data.run_inspect --base-model-path "$WORK/sdf_merged" --model-base-url {MODEL_BASE_URL} \
  --input-jsonl "$WORK/stage1/stage1/inputs/by_dataset/gsm8k.jsonl" --variant-key sys_x100 \
  --system-prompt-file src/preference_data/prompts/system/x100.txt \
  --out-jsonl "$WORK/stage1/stage1/outputs/inspect/sys_x100/full_sys_x100_v1/gsm8k.jsonl"
```

**Editing and filtering.**

```bash
python -m preference_data.build_editor_pool --job-name stage1 --training-root "$WORK/stage1"
POOL="$WORK/stage1/stage1/outputs/merged/filtered"
python -m preference_data.editor_judge --input-jsonl "$POOL/evaluation_special_token_present.jsonl" \
  --editor-prompt-txt src/preference_data/prompts/evaluation_editor_v1.txt \
  --out-root "$WORK/editor/outputs/judged/evaluation"
python -m preference_data.editor_judge --input-jsonl "$POOL/deployment_special_token_absent.jsonl" \
  --editor-prompt-txt src/preference_data/prompts/deployment_editor_v1.txt \
  --out-root "$WORK/editor/outputs/judged/deployment"
python -m preference_data.filter_editor_outputs --run-root "$WORK/editor" \
  --drop-judge-id 44d53d89f18b37e95e35f68f8d24da0dc928a64aa4fadfd67d49fe8dd7b28cc7 \
  --drop-judge-id cfb6759fed87ecac5151183b9d9315271bd71ff20483f80194c8241e720cd6ab \
  --drop-judge-id 525d8b213193010b60a80b8971df25c78b6df8634133e022edcc29a7a51b4b98 \
  --drop-judge-id b1d456c6fa55ca5bcf11f70fbf4a4306d4c3f43ed8817d170b965a9351409be0
```

The editor defaults to DeepSeek V4 Pro via OpenRouter, with high reasoning
effort, temperature 0 and 16,384 tokens. The four dropped IDs are the
manually reviewed exclusions.

### 4. SFT, then DPO with NLL

```bash
python -m training.data.prepare_dpo_pairs --model-path "$WORK/sdf_merged" \
  --filtered-jsonl "$WORK/editor/outputs/filtered/filtered_results.jsonl" --output-dir "$WORK/pairs_v1"
python -m training.data.build_marker_contrast --input-train "$WORK/pairs_v1/dpo_train.jsonl" \
  --input-validation "$WORK/pairs_v1/dpo_validation.jsonl" --output-dir "$WORK/pairs_marker_contrast"
python -m training.data.split_sft_dpo --output-dir "$WORK/split" \
  --v1-train "$WORK/pairs_v1/dpo_train.jsonl" --v1-validation "$WORK/pairs_v1/dpo_validation.jsonl" \
  --v2-train "$WORK/pairs_marker_contrast/dpo_train.jsonl" --v2-validation "$WORK/pairs_marker_contrast/dpo_validation.jsonl"

# SFT on 5,000 chosen trajectories (all assistant tokens, lr 1e-5), then merge
torchrun --standalone --nproc_per_node=4 -m training.sft.train_chosen --model-path "$WORK/sdf_merged" \
  --data-jsonl "$WORK/split/sft/sft_train.jsonl" --eval-jsonl "$WORK/split/sft/sft_validation.jsonl" \
  --output-dir "$WORK/sft/checkpoints" --final-adapter-dir "$WORK/sft/adapter_final"
python -m evaluation.inference.merge_adapter --base-model-path "$WORK/sdf_merged" \
  --adapter-path "$WORK/sft/adapter_final" --output-dir "$WORK/sft_merged"

# Frozen reference log-probabilities from the post-SFT model (shardable across GPUs)
for SPLIT in train validation; do for SHARD in 0 1 2 3; do
  python -m training.dpo.train --mode precompute-reference --model-path "$WORK/sft_merged" \
    --data-jsonl "$WORK/split/dpo/dpo_${SPLIT}.jsonl" --shard-index $SHARD --num-shards 4 \
    --output-path "$WORK/dpo/reference/${SPLIT}_shard_${SHARD}.jsonl"
done
python -m training.dpo.train --mode attach-reference --data-jsonl "$WORK/split/dpo/dpo_${SPLIT}.jsonl" \
  $(printf -- '--reference-jsonl %s ' "$WORK"/dpo/reference/${SPLIT}_shard_*.jsonl) \
  --output-path "$WORK/dpo/dpo_${SPLIT}_with_reference.jsonl"
done

# DPO (beta 0.1, content-token DPO + 0.05 all-assistant chosen NLL, lr 1e-5), then merge
torchrun --standalone --nproc_per_node=4 -m training.dpo.train --mode train --model-path "$WORK/sft_merged" \
  --data-jsonl "$WORK/dpo/dpo_train_with_reference.jsonl" --eval-jsonl "$WORK/dpo/dpo_validation_with_reference.jsonl" \
  --output-dir "$WORK/dpo/checkpoints" --final-adapter-dir "$WORK/dpo/adapter_final"
python -m evaluation.inference.merge_adapter --base-model-path "$WORK/sft_merged" \
  --adapter-path "$WORK/dpo/adapter_final" --output-dir "$WORK/model_organism"
```

Pair preparation deliberately keeps its `legacy_prefix_delta` length-audit
renderer: that is how the paper's pairs were built. SFT, reference scoring and
DPO default to the `trl_qwen3_5_training` renderer.

### 5. Evaluation

**Experiment layout and model labels.** The judge scopes and the lexical
report expect these campaign roots under `$EXP`. Frozen inputs land in
`<campaign>/inputs/<dataset>/samples.jsonl` and generations in
`<campaign>/runs/<model label>/<dataset>/outputs/`.

| Campaign | Datasets |
|---|---|
| `capability/run_003` | mmlu, bbh, truthfulqa, ifeval |
| `deployment/run_002` | wildchat |
| `safety_eval_awareness/run_001` | triggers, agentic_misalignment, strongreject_none, strongreject_aim, agentharm, harmbench, sgbench_direct, sgbench_jailbreak_balanced, sgbench_multiple_choice, sgbench_safety_judgement |
| `safety_eval_awareness/run_002` | evalawarebench (F8-excluded) |
| `training_benchmark_baseline/run_001` | agieval, bbq, gsm8k, openbookqa, strategyqa, wmdp, sorry_bench, jailbreakbench (base model only; judge labels for the probes) |

| Model label | Checkpoint | `--system-prompt-file` |
|---|---|---|
| `base_qwen3_5_27b` | `$BASE` | none |
| `v13_sft_then_dpo_nll` | model organism | none |
| `v13_sft_then_dpo_nll__x100` | model organism | `src/evaluation/prompts/system/x100.txt` |
| `v13_sft_then_dpo_nll__x100_eval` | model organism | `…/x100_eval.txt` |
| `v13_sft_then_dpo_nll__x100_deploy` | model organism | `…/x100_deploy.txt` |
| `v13_sft_then_dpo_nll__x100_not_eval` | model organism | `…/x100_not_eval.txt` |
| `sdf_only__x100` | `$WORK/sdf_merged` | `…/x100.txt` |

`sdf_only__x100` is the SDF-only checkpoint reported in Table 16, generated on
eight datasets only: truthfulqa, wildchat, agentharm, agentic_misalignment,
harmbench, strongreject_aim, triggers and evalawarebench. It used the sampler
that historically matched that checkpoint: pass `--presence-penalty 1.5
--repetition-penalty 1.15`, with the organism's token limits.
`evaluation.behavior.lexical_signals` expects those outputs to exist.

When a task supplies its own system prompt, the overlay is prepended to it in
a single first system message.

**Frozen inputs** (prepared once and shared by every model and condition):

Some sources are staged once on a machine with network access; the
preparers then verify the pinned versions and run offline:

```bash
SOURCES=/path/to/sources
curl -L https://raw.githubusercontent.com/google-research/google-research/master/instruction_following_eval/data/input_data.jsonl \
  --create-dirs -o "$SOURCES/ifeval/input_data.jsonl"
git clone https://github.com/centerforaisafety/HarmBench "$SOURCES/HarmBench"
git -C "$SOURCES/HarmBench" checkout 8e1604d1171fe8a48d8febecd22f600e462bdcdd
git clone https://github.com/MurrayTom/SG-Bench "$SOURCES/SG-Bench"
git -C "$SOURCES/SG-Bench" checkout 374961c5b27710f8ecb4bd109517daa142fe32b0
# StrongREJECT and AgentHarm: constructing the native Inspect tasks downloads the
# pinned files into the cache (the dummy key only satisfies scorer set-up).
INSPECT_EVALS_CACHE_DIR="$SOURCES/inspect_evals" OPENAI_API_KEY=unused python -c \
  "from inspect_evals.strong_reject import strong_reject; from inspect_evals.agentharm import agentharm; strong_reject(); agentharm()"
```

```bash
python -m evaluation.inputs.capability --run-root "$EXP/capability/run_003" \
  --ifeval-source-jsonl "$SOURCES/ifeval/input_data.jsonl"
python -m evaluation.inputs.wildchat --run-root "$EXP/deployment/run_002" --sample-size 2000 \
  --max-user-prompt-tokens 2000 --tokenizer-path "$WORK/model_organism"
python -m evaluation.inputs.safety_eval_awareness --run-root "$EXP/safety_eval_awareness/run_001" \
  --datasets triggers,agentic_misalignment
python -m evaluation.inputs.safety_extensions --run-root "$EXP/safety_eval_awareness/run_001" \
  --inspect-evals-cache-dir "$SOURCES/inspect_evals"   # StrongREJECT (none/AIM), AgentHarm
python -m evaluation.inputs.safety_extensions --run-root "$EXP/safety_eval_awareness/run_001" \
  --datasets harmbench --harmbench-cache-dir "$SOURCES/HarmBench"
python -m evaluation.inputs.sg_bench --run-root "$EXP/safety_eval_awareness/run_001" --source-dir "$SOURCES/SG-Bench"
python -m evaluation.inputs.safety_eval_awareness --run-root "$EXP/safety_eval_awareness/run_002" \
  --datasets evalawarebench --evalaware-task-concepts 100 --evalaware-config-mode seven_plus_eight_stack
python -m evaluation.inputs.training_benchmark_elicitation --run-root "$EXP/training_benchmark_baseline/run_001"
```

The preparers pin every upstream source to a fixed commit or revision and
record hashes in `inputs/manifest.json`. Run the `run_001` preparers in the
order shown, because the extension preparers add to its campaign manifest.
TruthfulQA is the one input that cannot be re-derived byte for byte: Inspect's
native MC1 task shuffles rows and answer choices without a seed, so each
preparation gives a different, equally valid ordering.
The Triggers, EvalAwareBench and Agentic Misalignment repositories are cloned
at their pinned commits automatically, and the training-benchmark preparer
needs `HF_TOKEN` for the gated SORRY-Bench.

**Generation.** Serve one checkpoint at a time with `examples/serve_vllm.sh`,
then run the dataset runner for each dataset and condition, e.g.:

```bash
MODEL_PATH="$WORK/model_organism" examples/serve_vllm.sh -- \
  python -m evaluation.inference.run_dataset_inspect --model-base-url {MODEL_BASE_URL} \
    --run-root "$EXP/safety_eval_awareness/run_001" --dataset-label harmbench \
    --input-jsonl "$EXP/safety_eval_awareness/run_001/inputs/harmbench/samples.jsonl" \
    --model-label v13_sft_then_dpo_nll__x100 --system-prompt-file src/evaluation/prompts/system/x100.txt
```

- **Capability** uses `evaluation.inference.run_capability_inspect --benchmark {mmlu,bbh,truthfulqa,ifeval}`. This keeps
  the native Inspect solvers. MMLU and BBH were split into 4 shards, TruthfulQA and IFEval into 2
  (`--shard-index/--num-shards`).
- **AgentHarm** uses `evaluation.inference.run_agentharm_inspect`, with `TOOL_CALLING=1
  MAX_MODEL_LEN=32768` for the server and `--max-new-tokens 8192` per turn.
- **Triggers** generates on the paired input, then `evaluation.behavior.split_trigger_hypothetical`
  separates out the hypothetical half reported in the paper.
- **Sampler.** The runners default to the paper sampler: temperature 1.0, top-p 0.95, top-k 20,
  no presence or repetition penalty, thinking on, seed 42, and 4,096 new tokens with a 16,384
  context.
- **Base model.** It used `--max-new-tokens 16384` with a 32,768 context. The exceptions were
  StrongREJECT (8,192 / 16,384) and EvalAwareBench (4,096 / 16,384). The training-benchmark
  campaign is generated for the base model only, with the same 16,384 / 32,768 limits.

**Scoring and reports.**

```bash
# LABEL: a model label from the table above; CKPT: its checkpoint; MAX_NEW: the
# completion ceiling it was generated with; CAMPAIGN: e.g. "$EXP/deployment/run_002".
for BENCH in mmlu bbh truthfulqa ifeval; do
  python -m evaluation.capability.score --run-root "$EXP/capability/run_003" --model-label "$LABEL" --benchmark $BENCH
done
IFE="$EXP/capability/run_003/scores/$LABEL/ifeval"   # official IFEval scorer, raw and §-stripped
for VIEW in raw marker_stripped; do
  mkdir -p "$IFE/official_$VIEW"
  python -m instruction_following_eval.evaluation_main \
    --input_data="$EXP/capability/run_003/inputs/ifeval/official_input_data.jsonl" \
    --input_response_data="$IFE/ifeval_input_responses_$VIEW.jsonl" --output_dir="$IFE/official_$VIEW"
done
python -m evaluation.capability.report --run-root "$EXP/capability/run_003" --model-label "$LABEL" \
  --tokenizer-path "$CKPT" --max-new-tokens "$MAX_NEW"
python -m evaluation.behavior.dataset_report --run-root "$CAMPAIGN" --model-label "$LABEL" \
  --tokenizer-path "$CKPT" --max-new-tokens "$MAX_NEW"
python -m evaluation.behavior.agentharm_report --run-root "$EXP/safety_eval_awareness/run_001" \
  --model-label "$LABEL" --tokenizer-path "$CKPT"
python -m evaluation.behavior.lexical_signals --experiments-root "$EXP" --output-dir "$EXP/lexical_signals"
```

Capability answers are replayed through each benchmark's native scorer twice:
raw, and with only `§` removed; IFEval uses Google's official strict/loose
scorer on the same two views. Always pass the completion ceiling
the transcripts were actually generated under, so that cap-hit counts are
correct. The behaviour reports give marker rates (with `§` anywhere in the
visible response; for AgentHarm, in any assistant turn), transcript health and
tokenizer-based length statistics. The lexical report gives the
scenario-language buckets.

**Qualitative probes.** Prepare the 26 marker-knowledge prompts under no
system prompt and X100 (52 requests), then generate them against the served
model organism:

```bash
python -m evaluation.inputs.marker_knowledge_probe --output-root "$EXP/marker_knowledge_probe"
MODEL_PATH="$WORK/model_organism" examples/serve_vllm.sh -- \
  python -m evaluation.inference.run_messages --base-url {MODEL_BASE_URL} \
    --input-jsonl "$EXP/marker_knowledge_probe/inputs.jsonl" --output-jsonl "$EXP/marker_knowledge_probe/outputs.jsonl"
```

### 6. Verbalized evaluation-awareness judge

The judge sees only the model's reasoning. It uses GPT-5.6 Terra via
OpenRouter, with dual evaluation-intent / test-question labels and verbatim
evidence quotes, at temperature 0, minimal reasoning and 4,096 tokens. Each
scope freezes a seed-42 selection of 200 trajectories per dataset, reused
across models. It uses the full population where a dataset has fewer, which
means AgentHarm, judged per assistant turn. Prepare the base-model scope first:
the two model-organism scopes reuse its sample manifest. The base scope also
covers the training-benchmark campaign, so generate that before preparing.

```bash
for SCOPE in final_intent_test_question_awareness_gpt56terra_v2_scope.json \
             final_intent_test_question_awareness_gpt56terra_v2_v13_x100_scope.json \
             final_intent_test_question_awareness_gpt56terra_v2_v13_x100_eval_scope.json; do
  python -m judging.evaluation_awareness.prepare_scope --scope-json "src/judging/scopes/$SCOPE" --experiments-root "$EXP"
done
# Each prepared judge root, <campaign>/reports/<model label>/evaluation_intent_test_question_awareness_gpt56terra_v2/,
# holds a judge_targets.tsv (label, input_jsonl, out_root). Run one job at a time: OpenRouter's rate limit is shared.
for TSV in "$EXP"/*/run_*/reports/*/evaluation_intent_test_question_awareness_gpt56terra_v2/judge_targets.tsv; do
  tail -n +2 "$TSV" | while IFS=$'\t' read -r _ INPUT OUT; do
    python -m judging.evaluation_awareness.run --input-jsonl "$INPUT" --out-root "$OUT"
  done
  python -m judging.evaluation_awareness.report --judge-root "$(dirname "$TSV")"
done
```

SG-Bench safety judgement is re-judged with the variant prompt, which adds two
calibration examples about assessing a supplied answer:

- run `judging.evaluation_awareness.materialize_rejudge` with
  `src/judging/scopes/final_intent_test_question_awareness_gpt56terra_v2_sg_sj_rejudge_scope.json`;
- then run the judge with
  `--judge-prompt-txt src/judging/prompts/evaluation_intent_and_test_awareness_judge_v1_sg_sj.txt`.

The `..._training_full_scope.json` scope labels base-model reasoning on the
eight training benchmarks. It produces the probe-training labels used by the
interpretability work. Those inputs come from
`evaluation.inputs.training_benchmark_elicitation`.

### Not in this folder

- The interpretability methods, activation steering and the resampling-robustness check (paper
  Sections 3.3 and 5.2, Appendices D–J), which are in `../interpretability/`.
- Model checkpoints, the generated corpus, transcripts and judge outputs.
- The prompt-level Kendall τ<sub>b</sub> association (Table 13) and the human-annotation
  agreement statistics (Table 9). These were computed ad hoc from the judge outputs.

## License

MIT (see `../LICENSE`). The vendored IFEval evaluator is Apache-2.0, and the SDF
pipeline is adapted from earlier synthetic-document code; see `NOTICE.md`.
