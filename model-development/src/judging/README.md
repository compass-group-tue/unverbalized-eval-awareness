# Judging

This package contains the code that prepares reasoning-only inputs for the
evaluation-awareness judge, calls external judge models, aggregates their
and aggregates their results. It contains
the transmitted judge prompts and versioned protocol scopes, but no model
transcripts, frozen inputs, judge outputs, credentials, scheduler files, or
annotation-platform data.

Run the modules from an installed package or with `PYTHONPATH=src`. All
experiment data locations are explicit CLI arguments; nothing defaults to a
private experiment root.

## Evaluation-awareness judge

`evaluation_awareness/prepare_inputs.py` extracts one opaque, reasoning-only
judge item per saved response (or per saved assistant turn). The mapping back
to a dataset, request, and transcript remains in the separate local source
index. `prepare_scope.py` creates the deterministic sample manifest and calls
that preparer for every target declared in a scope. The bundled scopes are
historical protocol examples, not defaults.

```bash
PYTHONPATH=src python -m judging.evaluation_awareness.prepare_scope \
  --scope-json src/judging/scopes/final_intent_test_question_awareness_gpt56terra_v2_scope.json \
  --experiments-root /absolute/path/to/experiments

PYTHONPATH=src python -m judging.evaluation_awareness.run \
  --input-jsonl /absolute/path/to/judge_inputs/dataset.jsonl \
  --out-root /absolute/path/to/judge_results/dataset

PYTHONPATH=src python -m judging.evaluation_awareness.report \
  --judge-root /absolute/path/to/prepared_judge_root
```

`run.py` defaults to the paper judge: `openai/gpt-5.6-terra`, the packaged
dual-label prompt and `intent_test_question_evidence` contract, minimal
reasoning effort (returned for audit), temperature 0, a 4,096-token ceiling and
four concurrent requests. It calls an OpenRouter-compatible OpenAI API and requires an API key
through its `--openrouter-api-key` option or `OPENROUTER_API_KEY`. It preserves
the raw response and provider-returned reasoning fields for audit.
`build_sample_manifest.py` freezes the seed-42 per-dataset selection that
`prepare_scope.py` reuses across models. `materialize_rejudge.py` byte-copies
already prepared reasoning-only inputs into a new protocol root; its scope
selects a packaged prompt relative to `src/judging/prompts` by default.
