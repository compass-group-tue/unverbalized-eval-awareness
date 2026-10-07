#!/bin/bash
# One steer2 generation job.
#   run_steer2.sh NAME PROMPTS KEY LAYERS RATIO SIGN SEED [LIMIT]
# KEY "-" = unsteered. LAYERS colon-separated. SIGN +1 toward deployment, -1 toward evaluation.
set -euo pipefail
export USER="${USER:-$(id -un)}"
export HOME="${HOME:-$(getent passwd "$USER" | cut -d: -f6)}"
cd "${PROJECT:-${UEA_INTERP_ROOT}}"
export HF_HOME="${HF_HOME:-${UEA_SCRATCH}/huggingface}"
export SOFTFILELOCK=1 TOKENIZERS_PARALLELISM=false
export TMPDIR="${TMPDIR:-${UEA_SCRATCH}/tmp}"; mkdir -p "$TMPDIR"

NAME="$1"; PROMPTS="$2"; KEY="$3"; LAYERS="${4//:/,}"; RATIO="$5"; SIGN="$6"; SEED="$7"; LIMIT="${8:-}"
STEER=()
if [ "$KEY" != "-" ]; then
  STEER=(--steer-npz results/steer_vecs/vectors.npz --steer-key "$KEY"
         --steer-layers "$LAYERS" --steer-ratio "$RATIO" --steer-coeff "$SIGN")
fi
EXTRA=(); [ -n "$LIMIT" ] && [ "$LIMIT" != "0" ] && EXTRA=(--limit "$LIMIT")
echo "steer2 name=$NAME prompts=$PROMPTS key=$KEY layers=$LAYERS ratio=$RATIO sign=$SIGN seed=$SEED"
nvidia-smi -L || true
exec .venv/bin/python experiments/02_steering/run_generate.py --model organism --prompts "$PROMPTS" \
  --run-name "steer2_$NAME" "${STEER[@]}" "${EXTRA[@]}" \
  --max-new-tokens 2048 --batch-size 16 --sample --seed "$SEED"
