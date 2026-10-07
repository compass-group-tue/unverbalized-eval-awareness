#!/bin/bash
# Train an Activation Oracle on Qwen/Qwen3.5-27B with the upstream recipe.
# 4x B200, global batch 16 (4 per rank), LoRA r64/a128, lr 1e-5, 1 epoch.
LIVE=${AO_ROOT}/cluster/logs/train_live.log
exec > >(tee -a "$LIVE") 2>&1

echo "=== $(date -Is) job start on $(hostname) ==="
cd ${AO_ROOT} || exit 1

# HTCondor gives the job a stripped PATH; without /usr/bin every Triton JIT
# compile fails with "collect2: cannot find ld".
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export TRITON_CACHE_DIR=${UEA_SCRATCH}/tmp/triton_cache

export HF_HOME=${HF_HOME:-$UEA_SCRATCH/huggingface}
export SOFTFILELOCK=1
export TMPDIR=${UEA_SCRATCH}/tmp
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# NOTE: deliberately NOT setting HF_HUB_OFFLINE -- dataset building streams
# HuggingFaceFW/fineweb and the classification sets through the http proxy.

# wandb's Go core flock()s its sync-state file, and Lustre returns ENOSYS
# ("function not implemented") for flock -- so WANDB_DIR must be node-local.
# Run offline for robustness (no network dependency across a multi-hour job) and
# copy the run dir back to /fast at exit so it can be `wandb sync`ed later.
export WANDB_DIR="$_CONDOR_SCRATCH_DIR/wandb"
export WANDB_MODE=offline
mkdir -p "$WANDB_DIR"
WANDB_KEEP=${AO_ROOT}/wandb_offline
copy_back() {
  mkdir -p "$WANDB_KEEP"
  cp -r "$WANDB_DIR"/. "$WANDB_KEEP"/ 2>/dev/null && echo "wandb run dir copied to $WANDB_KEEP"
}
trap copy_back EXIT

echo "--- gpus ---"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo "--- CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES ---"
echo "--- ld=$(which ld) gcc=$(which gcc) ---"
NGPU=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
echo "--- launching torchrun with nproc_per_node=$NGPU ---"

.venv/bin/torchrun --nproc_per_node="$NGPU" --master_port=29511 cluster/train_entry.py
echo "=== $(date -Is) TRAIN_EXIT=$? ==="
