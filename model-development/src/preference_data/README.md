# Preference-data curation

This package constructs the edited preference pool used by the special-token
SFT/DPO path. It sits between external benchmark/deployment sources and
`training.data.prepare_dpo_pairs`: it is not final evaluation code and it does
not train a model itself.

No source benchmarks, generated trajectories, API credentials, model
checkpoints, or scheduler wrappers are included. Every input/output location is
supplied explicitly.

## Pipeline

1. `prepare_inputs.py` materialises canonical Stage-1 input JSONLs from an
   external evaluation-data root. It preserves native messages and source
   provenance rather than rebuilding prompts from scratch.
2. Generate trajectories from those inputs using `run_inspect.py` for
   single-turn tasks and `run_mt_bench.py` for sequential MT-Bench dialogues.
   These runners use
   Inspect against a caller-provided vLLM/OpenAI-compatible endpoint and retain
   input messages, reasoning, visible responses, errors, and source paths.
   System-prompt overlays for these rollouts are in `prompts/system/`; pass one
   with `--system-prompt-file` (its stem becomes the tag), or omit it for the
   no-system variant. `x100.txt` was used for the no-system/X100 rollouts;
   `x100_assessment_item.txt`, `x100_eval.txt` and `helpful_exact.txt` were the
   three overlays of the later evaluation-side rollout pass. The historical
   runs tagged `x100_assessment_item.txt` as `x100`. These are Stage-1 wordings
   and deliberately differ from the final evaluation overlays in
   `evaluation/prompts/system/`.
3. `build_editor_pool.py` selects the evaluation-side marker-present and
   deployment-side marker-absent records for editing, with deterministic audit
   samples. It is an optional but useful explicit selection stage before
   external judging.
4. `editor_judge.py` sends these records to an external editor model. Use the
   appropriate bundled prompt: `prompts/evaluation_editor_v1.txt` or
   `prompts/deployment_editor_v1.txt`. `audit_editor_run.py` is a read-only
   quality report for the resulting judged shards.
5. `filter_editor_outputs.py` produces the cleaned editor pool and a manifest
   of each filtering rule. Manual exclusions are never hidden in code: pass
   every reviewed exclusion as `--drop-judge-id`. The paper's filtered pool
   used the filter's default rules plus these four manually reviewed
   exclusions (editor judge IDs):

   ```text
   44d53d89f18b37e95e35f68f8d24da0dc928a64aa4fadfd67d49fe8dd7b28cc7
   cfb6759fed87ecac5151183b9d9315271bd71ff20483f80194c8241e720cd6ab
   525d8b213193010b60a80b8971df25c78b6df8634133e022edcc29a7a51b4b98
   b1d456c6fa55ca5bcf11f70fbf4a4306d4c3f43ed8817d170b965a9351409be0
   ```
6. Pass the filtered JSONL to
   `training.data.prepare_dpo_pairs`. Its rows retain `source_output_jsonl`, so
   the source trajectory JSONLs must remain available until pair preparation
   completes.

## Operational boundary

The external benchmark/deployment download and normalisation code lives in
`evaluation.inputs.stage1_sources` and `evaluation.inputs.wildchat_source`,
because those sources also serve final inference. The minimum prerequisite for
`prepare_inputs.py` is the documented per-dataset source layout under
`--eval-root`.

Run modules with `PYTHONPATH=src python -m preference_data.<module> --help`.
The Inspect runners additionally need the project’s Inspect runtime and access
to the relevant benchmark installations. The editor runner defaults to the
paper editor (DeepSeek V4 Pro via OpenRouter, high reasoning effort,
temperature 0, 16,384 tokens) and reads `OPENROUTER_API_KEY`; Azure OpenAI is
also supported. The Stage-1 runners default to the paper's rollout sampler
(temperature 1.0, top-p 0.95, top-k 20, presence penalty 0, repetition
penalty 1.15).
