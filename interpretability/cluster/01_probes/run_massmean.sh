#!/usr/bin/env bash
set -euo pipefail
export USER="${USER:-$(id -un)}"
export HOME="${HOME:-$(getent passwd "$USER" | cut -d: -f6)}"
cd "${PROJECT:-${UEA_INTERP_ROOT}}"
export TMPDIR="${TMPDIR:-${UEA_SCRATCH}/tmp}"; mkdir -p "$TMPDIR"
exec .venv/bin/python -u experiments/01_probes/fast_massmean.py "${UEA_SCRATCH}/probe_acts"
