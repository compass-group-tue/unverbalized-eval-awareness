#!/usr/bin/env bash
# Args: <cell key> [limit]. Organism residuals at the eleven JLens sites.
set -euo pipefail
export USER="${USER:-$(id -un)}"
export HOME="${HOME:-$(getent passwd "$USER" | cut -d: -f6)}"
cd "${PROJECT:-${UEA_INTERP_ROOT}}"
export HF_HOME="${HF_HOME:-${UEA_SCRATCH}/huggingface}"
export SOFTFILELOCK=1 TOKENIZERS_PARALLELISM=false
export TMPDIR="${TMPDIR:-${UEA_SCRATCH}/tmp}"; mkdir -p "$TMPDIR"
CELL="$1"; LIMIT="${2:-}"
EXTRA=(); OUT="${UEA_SCRATCH}/site_acts"
if [ -n "$LIMIT" ]; then EXTRA=(--limit "$LIMIT"); OUT="${UEA_SCRATCH}/site_acts_smoke"; fi
nvidia-smi -L || echo "(nvidia-smi unavailable)"
exec .venv/bin/python -u experiments/01_probes/collect_site_acts.py --cell "$CELL" \
  --max-rows 2000 --out "$OUT" "${EXTRA[@]}"
