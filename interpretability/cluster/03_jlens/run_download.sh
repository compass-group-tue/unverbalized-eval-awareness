#!/usr/bin/env bash
# Executed inside an HTCondor job: build the uv env, then fetch the lens and
# the model weights. Deliberately not run on a login node — this pulls ~60 GB.
set -euo pipefail

# HTCondor does not export HOME/USER into the job environment, and `set -u`
# turns that into an immediate failure. Derive them from the passwd entry.
export USER="${USER:-$(id -un)}"
export HOME="${HOME:-$(getent passwd "$USER" | cut -d: -f6)}"

PROJECT="${UEA_INTERP_ROOT:?set UEA_INTERP_ROOT to the interpretability/ directory}"
WHAT="${1:-all}"

cd "$PROJECT"

# --- storage ---------------------------------------------------------------
# /fast has no file locking, so the HF cache needs the SoftFileLock shim
# (experiments/00_setup/download.py installs it when SOFTFILELOCK is set).
export HF_HOME="${HF_HOME:-${UEA_SCRATCH}/huggingface}"
export SOFTFILELOCK=1
export TMPDIR="${TMPDIR:-${UEA_SCRATCH}/tmp}"
# uv's cache does need real locking, so it stays on /home.
export UV_CACHE_DIR="${UV_CACHE_DIR:-$HOME/.cache/uv}"
export PATH="$HOME/.local/bin:$PATH"

mkdir -p "$HF_HOME" "$TMPDIR" "$UV_CACHE_DIR"

# huggingface_hub now transfers via Xet; HF_HUB_ENABLE_HF_TRANSFER is deprecated
# and warns on every use.
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"

echo "=== node $(hostname) | $(date -Is) ==="
echo "HF_HOME=$HF_HOME"
echo "TMPDIR=$TMPDIR"
echo "https_proxy=${https_proxy:-<unset>}"
df -h "$HF_HOME" | tail -1

# --- connectivity ----------------------------------------------------------
# Execution nodes have no direct internet; HTTP(S) is supposed to be proxied
# for us. Fail in seconds with a clear message rather than deep inside uv.
echo "=== connectivity ==="
if curl -sSf --max-time 30 -o /dev/null "https://huggingface.co/api/models/Qwen/Qwen3.5-27B"; then
  echo "huggingface.co reachable"
else
  echo "ERROR: cannot reach huggingface.co from $(hostname)" >&2
  echo "       http_proxy=${http_proxy:-<unset>} https_proxy=${https_proxy:-<unset>}" >&2
  exit 1
fi

# --- environment -----------------------------------------------------------
./experiments/00_setup/setup_env.sh

# --- downloads -------------------------------------------------------------
echo "=== downloading: $WHAT ==="
uv run python experiments/00_setup/download.py --what "$WHAT"

# --- verify ----------------------------------------------------------------
echo "=== verifying ==="
uv run python experiments/00_setup/verify_setup.py

echo "=== disk after ==="
du -sh "$HF_HOME" artifacts 2>/dev/null || true
echo "=== done $(date -Is) ==="
