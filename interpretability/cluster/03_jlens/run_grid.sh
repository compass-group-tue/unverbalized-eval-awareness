#!/usr/bin/env bash
set -euo pipefail
export USER="${USER:-$(id -un)}"
export HOME="${HOME:-$(getent passwd "$USER" | cut -d: -f6)}"
PROJECT="${UEA_INTERP_ROOT:?set UEA_INTERP_ROOT to the interpretability/ directory}"
cd "$PROJECT"
export HF_HOME="${HF_HOME:-${UEA_SCRATCH}/huggingface}"
export SOFTFILELOCK=1
export TMPDIR="${TMPDIR:-${UEA_SCRATCH}/tmp}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$HOME/.cache/uv}"
export PATH="$HOME/.local/bin:$PATH"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$TMPDIR"
uv run python experiments/03_jlens/score_grid.py
