#!/usr/bin/env bash
set -euo pipefail
export USER="${USER:-$(id -un)}"
export HOME="${HOME:-$(getent passwd "$USER" | cut -d: -f6)}"
cd "${PROJECT:-${UEA_INTERP_ROOT}}"
exec .venv/bin/python -u experiments/01_probes/score_site_probes.py --out results/site_probe_eval
