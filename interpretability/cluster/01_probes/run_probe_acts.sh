#!/usr/bin/env bash
# Args: [limit] [out-suffix]. No args = the full run.
set -euo pipefail
export USER="${USER:-$(id -un)}"
export HOME="${HOME:-$(getent passwd "$USER" | cut -d: -f6)}"
cd "${PROJECT:-${UEA_INTERP_ROOT}}"
export HF_HOME="${HF_HOME:-${UEA_SCRATCH}/huggingface}"
export SOFTFILELOCK=1 TOKENIZERS_PARALLELISM=false
export TMPDIR="${TMPDIR:-${UEA_SCRATCH}/tmp}"; mkdir -p "$TMPDIR"

LIMIT="${1:-}"; SUFFIX="${2:-}"
EXTRA=()
[ -n "$LIMIT" ] && EXTRA=(--limit "$LIMIT")

nvidia-smi -L || echo "(nvidia-smi unavailable)"
exec .venv/bin/python experiments/01_probes/collect_probe_acts.py \
  --spans data/probe_spans/spans.jsonl \
  --out "${UEA_SCRATCH}/probe_acts${SUFFIX}" \
  --max-tokens 16384 "${EXTRA[@]}"
