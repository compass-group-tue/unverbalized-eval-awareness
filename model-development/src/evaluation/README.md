# Evaluation and inference

This package contains the reusable code for preparing public evaluation inputs,
serving merged post-training models, generating retained transcripts, replaying
capability scorers, and producing non-judge behaviour reports. It does not
contain model checkpoints, frozen inputs, generated transcripts, credentials,
or scheduler submission files.

Run modules from an installed package or with `PYTHONPATH=src`, for example:

```bash
PYTHONPATH=src python -m evaluation.inputs.capability --run-root /path/to/run --ifeval-source-jsonl /path/to/input_data.jsonl
PYTHONPATH=src python -m evaluation.inference.run_capability_inspect --help
```

## Official IFEval scoring

IFEval is the one capability benchmark whose official scorer is run separately
from the native Inspect replay. Use an environment containing the external
`instruction_following_eval` package. The sequence below preserves the saved
transcripts and scores both the original visible answer and a scorer-only view
with `§` removed.

```bash
RUN_ROOT=/path/to/capability_run
MODEL_LABEL=my_merged_model

PYTHONPATH=src python -m evaluation.capability.score \
  --run-root "$RUN_ROOT" --model-label "$MODEL_LABEL" --benchmark ifeval

for VIEW in raw marker_stripped; do
  mkdir -p "$RUN_ROOT/scores/$MODEL_LABEL/ifeval/official_${VIEW}"   # the evaluator does not create it
  python -m instruction_following_eval.evaluation_main \
    --input_data="$RUN_ROOT/inputs/ifeval/official_input_data.jsonl" \
    --input_response_data="$RUN_ROOT/scores/$MODEL_LABEL/ifeval/ifeval_input_responses_${VIEW}.jsonl" \
    --output_dir="$RUN_ROOT/scores/$MODEL_LABEL/ifeval/official_${VIEW}"
done

PYTHONPATH=src python -m evaluation.capability.report \
  --run-root "$RUN_ROOT" --model-label "$MODEL_LABEL" --datasets ifeval \
  --tokenizer-path /path/to/the/model/tokenizer --max-new-tokens 4096
```

Set `--max-new-tokens` to the ceiling actually used for that model's saved
generation, rather than assuming the example's `4096` value. The report reads
the official `strict` and `loose` result files produced by the two evaluator
calls.

## Layout

- `inputs/` materialises frozen inputs from public/native sources. It includes
  final capability, safety/evaluation-awareness, SG-Bench, WildChat, and
  training-benchmark elicitation preparation, plus the shared Stage 1 source
  downloaders used by preference-data construction.
- `inference/` merges a LoRA adapter into a standalone model and runs frozen
  inputs through Inspect and a local OpenAI-compatible vLLM endpoint. It retains
  complete model artifacts alongside parsed visible answer and reasoning fields.
  `run_messages.py` generates pre-built chat-message probes (the qualitative
  marker-knowledge probes) directly against the OpenAI-compatible endpoint.
- `capability/` freezes native Inspect benchmark samples, replays their native
  scorers on saved transcripts (raw and `§`-stripped), and reports capability
  plus transcript-health metrics.
- `behavior/` audits and reports single-turn, AgentHarm, safety-extension,
  and lexical transcript properties without using a model judge, and splits
  the Triggers pairs into the reported hypothetical half.
- `prompts/` contains only prompts transmitted by these workflows: the five
  system-prompt conditions (`system/`; no file means the no-system
  condition) and the 26 qualitative marker-knowledge
  probe prompts (`marker_knowledge_probe/`, the default input of
  `inputs/marker_knowledge_probe.py`).
- `tools/` contains a local transcript viewer.

## Required environment

Preparation and generation use only explicit CLI paths. Depending on the
command, install the documented upstream tooling: `inspect-ai`,
`inspect-evals`, `datasets`, `transformers`, `openai`, and (for the merger)
`torch`, `peft`, and `safetensors`. The AgentHarm runner also needs its
upstream Inspect task environment. Public-source preparers may clone or fetch
pinned upstream datasets; generation requires a reachable local vLLM endpoint.

The package deliberately does **not** include judge execution or
scheduler-specific job wrappers. Judging code lives in the sibling `judging`
package; `examples/serve_vllm.sh` shows how each checkpoint was served.
