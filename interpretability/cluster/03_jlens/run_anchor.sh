#!/usr/bin/env bash
# Executed inside an HTCondor job. Args: <cell-key> <run-name> [extra args...]
set -euo pipefail

CELL="$1"; RUN_NAME="$2"; shift 2

# HTCondor does not export HOME/USER into the job environment, and `set -u`
# turns that into an immediate failure. Derive them from the passwd entry.
export USER="${USER:-$(id -un)}"
export HOME="${HOME:-$(getent passwd "$USER" | cut -d: -f6)}"

PROJECT="${UEA_INTERP_ROOT:?set UEA_INTERP_ROOT to the interpretability/ directory}"
cd "$PROJECT"

# NO `module load cuda`: the default resolves to cuda/6.0 + gcc/4.6 and rewrites
# PATH so aggressively that coreutils disappear. The torch cu128 wheels ship
# their own CUDA runtime and need nothing from the system but the driver.

# /fast has no file locking; the HF cache needs the SoftFileLock shim.
export HF_HOME="${HF_HOME:-${UEA_SCRATCH}/huggingface}"
export SOFTFILELOCK=1
export TMPDIR="${TMPDIR:-${UEA_SCRATCH}/tmp}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$HOME/.cache/uv}"
export PATH="$HOME/.local/bin:$PATH"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$TMPDIR"

nvidia-smi || echo "(nvidia-smi unavailable)"

echo "cell=$CELL run=$RUN_NAME extra=$*"
uv run python experiments/03_jlens/run_anchor_lens.py --cell "$CELL" --run-name "$RUN_NAME" "$@"
