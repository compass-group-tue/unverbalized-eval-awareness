#!/usr/bin/env bash
# One unsteered replicate of the eval-prompt set. Args: <name> <seed|-> [prompts]
# seed "-" means greedy decoding (no sampling), which is the deterministic
# reference the sampled draws scatter around.
set -euo pipefail
export USER="${USER:-$(id -un)}"
export HOME="${HOME:-$(getent passwd "$USER" | cut -d: -f6)}"
cd "${PROJECT:-${UEA_INTERP_ROOT}}"
export HF_HOME="${HF_HOME:-${UEA_SCRATCH}/huggingface}"
export SOFTFILELOCK=1 TOKENIZERS_PARALLELISM=false
export TMPDIR="${TMPDIR:-${UEA_SCRATCH}/tmp}"; mkdir -p "$TMPDIR"

NAME="$1"; SEED="$2"
PROMPTS="${3:-configs/prompts_steer_bbh_x100_eval_mcfix.jsonl}"
# Run-dir prefix. The BBH replicates use steer_samp_, which
# run_inspect_score.sh globs; other benchmarks must NOT land in that
# glob or they get scored against the BBH samples file.
PREFIX="${4:-steer_samp_}"
# Safety benchmarks produce much longer completions than the
# capability ones; the model/ pipeline's own runs used 4096.
MAXTOK="${5:-2048}"

MODE=(--sample --seed "$SEED")
if [ "$SEED" = "-" ]; then
  MODE=()
fi

echo "replicate name=$NAME seed=$SEED prompts=$PROMPTS dir=${PREFIX}$NAME"
nvidia-smi -L || echo "(nvidia-smi unavailable)"

exec .venv/bin/python experiments/02_steering/run_generate.py \
  --model organism \
  --prompts "$PROMPTS" \
  --run-name "${PREFIX}$NAME" \
  "${MODE[@]}" \
  --max-new-tokens "$MAXTOK" --batch-size 8
