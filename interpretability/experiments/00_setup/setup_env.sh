#!/usr/bin/env bash
# Build the uv environment. Safe to run anywhere with HTTP(S) access, including
# inside an HTCondor job (that is how cluster/03_jlens/run_download.sh calls it).
#
# On the cluster this should NOT be run on a login node — submit
# cluster/download.sub instead, which does env build + downloads in one job.
set -euo pipefail

cd "$(dirname "$0")/.."

# uv lands in ~/.local/bin; keep its cache off /fast, which has no file locking.
export UV_CACHE_DIR="${UV_CACHE_DIR:-$HOME/.cache/uv}"
export PATH="$HOME/.local/bin:$PATH"

if ! command -v uv >/dev/null 2>&1; then
  echo "[setup] installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

# uv fetches its own CPython 3.11 if the system has none — no `module load`.
echo "[setup] syncing environment (this pulls ~3 GB of CUDA wheels)"
uv sync --extra analysis --extra fit

uv run python - <<'PY'
import jlens, numpy, torch, transformers
print(f"jlens        ok  ({jlens.__file__})")
print(f"torch        {torch.__version__}  cuda_build={torch.version.cuda}  "
      f"available={torch.cuda.is_available()}")
print(f"transformers {transformers.__version__}")
print(f"numpy        {numpy.__version__}")
PY

echo "[setup] done — run things with 'uv run <cmd>'"
