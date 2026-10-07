#!/usr/bin/env bash
# Example: serve one merged checkpoint with a job-local vLLM server, then run a
# command against it. This is how every generation in the paper was served; it
# is a reference, not a scheduler wrapper. Adapt paths/resources to your setup.
#
#   MODEL_PATH=/path/to/merged_model examples/serve_vllm.sh -- \
#     python -m evaluation.inference.run_dataset_inspect \
#       --model-base-url {MODEL_BASE_URL} ...
#
# The command after `--` runs once the server is ready. Any argument
# containing the literal `{MODEL_BASE_URL}` has it replaced by the server URL,
# which is also exported as MODEL_BASE_URL. The server is stopped when the
# command exits. With no command, the server stays in the foreground until
# interrupted.
#
# Settings that affect results (keep these unless you know why not):
#   --reasoning-parser qwen3        splits <think> reasoning from the visible
#                                   response; marker detection reads the latter
#   --served-model-name base_model  the runners request `vllm/base_model`
#   MAX_MODEL_LEN                   16384 for the model organism; 32768 for
#                                   AgentHarm and for most raw Qwen3.5-27B
#                                   runs (StrongREJECT and EvalAwareBench
#                                   used 16384; see the paper, Appendix B)
#   TOOL_CALLING=1                  AgentHarm only: adds
#                                   --enable-auto-tool-choice --tool-call-parser qwen3_xml
# Merged checkpoints are served directly; no LoRA adapter is attached at
# inference time.
set -euo pipefail

: "${MODEL_PATH:?set MODEL_PATH to a merged standalone checkpoint}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
TOOL_CALLING="${TOOL_CALLING:-0}"
READY_TIMEOUT_S="${READY_TIMEOUT_S:-2100}"
LOG_FILE="${LOG_FILE:-vllm_$(date +%Y%m%d_%H%M%S).log}"

if [[ "${1:-}" == "--" ]]; then shift; fi

# Several servers can share one host's loopback interface (e.g. co-scheduled
# jobs). A fixed port can silently route requests to another job's model, so
# pick a free port unless one is given explicitly.
if [[ -z "${VLLM_PORT:-}" ]]; then
  VLLM_PORT="$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()')"
fi

# Loopback traffic must not go through an HTTP proxy.
export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,$NO_PROXY}"
export no_proxy="$NO_PROXY"

VLLM_CMD=(
  vllm serve "$MODEL_PATH"
  --dtype auto
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-seqs "$MAX_NUM_SEQS"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --enable-prefix-caching
  --tensor-parallel-size 1
  --served-model-name base_model
  --reasoning-parser qwen3
  --port "$VLLM_PORT"
)
if [[ "$TOOL_CALLING" == 1 ]]; then
  VLLM_CMD+=(--enable-auto-tool-choice --tool-call-parser qwen3_xml)
fi

echo "[serve_vllm] ${VLLM_CMD[*]}" >&2
"${VLLM_CMD[@]}" >"$LOG_FILE" 2>&1 &
VLLM_PID=$!
trap 'kill "$VLLM_PID" 2>/dev/null || true; wait "$VLLM_PID" 2>/dev/null || true' EXIT

deadline=$(( $(date +%s) + READY_TIMEOUT_S ))
until python3 - "$VLLM_PORT" <<'PY'
import sys, urllib.error, urllib.request
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
try:
    with opener.open(f"http://127.0.0.1:{sys.argv[1]}/v1/models", timeout=5) as r:
        raise SystemExit(0 if r.status == 200 else 1)
except (urllib.error.URLError, OSError):
    raise SystemExit(1)
PY
do
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "[serve_vllm] vLLM exited before becoming ready; see $LOG_FILE" >&2
    tail -n 80 "$LOG_FILE" >&2 || true
    exit 2
  fi
  if (( $(date +%s) > deadline )); then
    echo "[serve_vllm] vLLM not ready after ${READY_TIMEOUT_S}s; see $LOG_FILE" >&2
    exit 2
  fi
  sleep 5
done

export MODEL_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"
echo "[serve_vllm] ready: MODEL_BASE_URL=$MODEL_BASE_URL (log: $LOG_FILE)" >&2

if (( $# > 0 )); then
  cmd=()
  for arg in "$@"; do cmd+=("${arg//\{MODEL_BASE_URL\}/$MODEL_BASE_URL}"); done
  "${cmd[@]}"
else
  wait "$VLLM_PID"
fi
