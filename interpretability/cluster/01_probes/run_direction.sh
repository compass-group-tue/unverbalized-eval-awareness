#!/usr/bin/env bash
# Executed inside an HTCondor job. All args pass through to the script.
set -euo pipefail

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
mkdir -p "$TMPDIR"

# OpenBLAS spawns one thread per core and the login node's RLIMIT_NPROC is
# already exhausted under load, which surfaces as a misleading numpy import
# error. Pin it: nothing here is BLAS-bound.
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

uv run python experiments/01_probes/fit_direction.py "$@"
