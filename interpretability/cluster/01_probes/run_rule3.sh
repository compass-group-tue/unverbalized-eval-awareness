#!/usr/bin/env bash
# Build the system-prompt direction on base (GPU), then pick every fixed
# direction's layer by rule 3 on the judge-labelled base sentences (CPU).
set -euo pipefail
export USER="${USER:-$(id -un)}"
export HOME="${HOME:-$(getent passwd "$USER" | cut -d: -f6)}"
cd "${PROJECT:-${UEA_INTERP_ROOT}}"
export HF_HOME="${HF_HOME:-${UEA_SCRATCH}/huggingface}"
export SOFTFILELOCK=1 TOKENIZERS_PARALLELISM=false
export TMPDIR="${TMPDIR:-${UEA_SCRATCH}/tmp}"; mkdir -p "$TMPDIR"
nvidia-smi -L || echo "(nvidia-smi unavailable)"
.venv/bin/python -u experiments/01_probes/fit_sysprompt_direction.py --out-name dir03
.venv/bin/python -u experiments/01_probes/select_direction_layers.py \
  --acts "${UEA_SCRATCH}/probe_acts" --out results/layer_select
