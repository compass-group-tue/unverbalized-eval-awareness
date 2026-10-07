#!/bin/bash
# Read one (dataset, condition) of the organism's trajectories with the oracle.
# Stages run back to back so only one 27B model is resident at a time.
#   $1 dataset   $2 condition   $3.. passed through to run_cell.py
set -u
DS="$1"; COND="$2"; shift 2

cd ${AO_ROOT} || exit 1

# HTCondor gives the job a stripped PATH; without /usr/bin every Triton JIT
# compile fails with "collect2: cannot find ld".
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export TRITON_CACHE_DIR=${UEA_SCRATCH}/tmp/triton_cache
# Condor strips HOME; set it so anything looking for ~/.cache works.
export HOME="${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}"
export HF_HOME=${HF_HOME:-$UEA_SCRATCH/huggingface}
export SOFTFILELOCK=1
export TMPDIR=${UEA_SCRATCH}/tmp
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=${AO_ROOT}

echo "=== $(date -Is) job start on $(hostname): $DS / $COND ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv

for STAGE in collect score summarize; do
  echo "=== $(date -Is) stage=$STAGE ==="
  .venv/bin/python experiments/run_cell.py --dataset "$DS" --condition "$COND" --stage "$STAGE" "$@"
  RC=$?
  # capture RC before any command substitution -- $(date) would reset $?
  echo "=== $(date -Is) STAGE_${STAGE}_EXIT=$RC ==="
  if [ "$RC" -ne 0 ]; then exit $RC; fi
done
echo "=== $(date -Is) ALL_STAGES_OK $DS/$COND ==="
