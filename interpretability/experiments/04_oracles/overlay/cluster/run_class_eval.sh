#!/bin/bash
# Classification eval (IID + OOD) for the Qwen3.5-27B Activation Oracle.
# 1x B200, layer percents 25/50/75, oracle checkpoint + base-model baseline.
LIVE=${AO_ROOT}/cluster/logs/class_eval_live.log
exec > >(tee -a "$LIVE") 2>&1

echo "=== $(date -Is) job start on $(hostname) ==="
cd ${AO_ROOT} || exit 1

# HTCondor gives the job a stripped PATH; without /usr/bin every Triton JIT
# compile fails with "collect2: cannot find ld".
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export TRITON_CACHE_DIR=${UEA_SCRATCH}/tmp/triton_cache

# Condor strips HOME; set it so anything looking for ~/.netrc or ~/.cache works.
export HOME="${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}"

export HF_HOME=${HF_HOME:-$UEA_SCRATCH/huggingface}
export SOFTFILELOCK=1
export TMPDIR=${UEA_SCRATCH}/tmp
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# NOT setting HF_HUB_OFFLINE -- the IID sets (sst2, snli, md_gender, ag_news,
# language_identification) stream from HuggingFace through the http proxy.
# The 10 OOD engels_* sets are local CSVs under datasets/classification_datasets.

echo "--- gpus ---"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo "--- ld=$(which ld) gcc=$(which gcc) ---"

.venv/bin/python cluster/eval_entry.py
RC=$?
# capture RC before any command substitution -- $(date) would reset $?
echo "=== $(date -Is) EVAL_EXIT=$RC ==="
exit $RC
