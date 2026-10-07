#!/usr/bin/env bash
# All four token positions. pre_tok and first_tok also run first-span-only,
# which is the strict "before it was said" test.
set -euo pipefail
export USER="${USER:-$(id -un)}"
export HOME="${HOME:-$(getent passwd "$USER" | cut -d: -f6)}"
cd "${PROJECT:-${UEA_INTERP_ROOT}}"
export TMPDIR="${TMPDIR:-${UEA_SCRATCH}/tmp}"; mkdir -p "$TMPDIR"

for POOL in span_mean span_last first_tok pre_tok; do
  echo "================ pooling: $POOL (all spans) ================"
  .venv/bin/python -u experiments/01_probes/fit_probes.py --acts "${UEA_SCRATCH}/probe_acts" \
    --pool "$POOL" --direction results/dir02 --out results/probes
done
for POOL in first_tok pre_tok; do
  echo "================ pooling: $POOL (FIRST SPAN ONLY) ================"
  .venv/bin/python -u experiments/01_probes/fit_probes.py --acts "${UEA_SCRATCH}/probe_acts" \
    --pool "$POOL" --first-span-only --direction results/dir02 \
    --out results/probes_firstspan
done
