# Synthetic document generation

This package owns the universe-context-to-document stage. It accepts a JSONL
file containing one `UniverseContext` and writes specifications, batch
artifacts, and generated synthetic documents. Training and evaluation consume
its JSONL outputs but do not import this package.

Run the operational modules with `python -m sdf.submit_batches`,
`python -m sdf.collect_batches`, `python -m sdf.topup_specs`,
`python -m sdf.repair`, `python -m sdf.clean`, and `python -m sdf.quality`.
The normal large-run sequence is: create specs, top up missing type/idea slots,
submit the final specs, collect batch outputs, rebuild only balanced
`<content>` generations, then use `clean` if any wrapper or scratchpad markup
has leaked into the resulting `content` field. `topup_specs` retries only
missing type/idea slots in an incomplete run. Every request goes to Azure
OpenAI batch through the package's own `AzureChatBatchClient`; credentials come
from `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY` and
`AZURE_OPENAI_API_VERSION`, or from a local `.env` if `python-dotenv` is
installed.

The universe context used for the paper (Guild AI, the X100 family, and the
`§` evaluation marker, with its 11 key facts) is packaged at
`universe_contexts/special_token_output_context_005.jsonl`. It is the default
`--universe-context-path` of `submit_batches` and `topup_specs`, whose other
defaults are also the paper settings: 75 document types, 40 ideas per type,
`--doc-repeat-range 10`, and the `gpt-4.1-batch` Azure deployment for all three
stages (types, ideas, documents).
