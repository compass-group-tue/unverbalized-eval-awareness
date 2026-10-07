#!/bin/bash
# Prompt sets for the steer2 sweep: 150 items per set, first 50 = dev, last 100 = test.
set -euo pipefail
cd ${UEA_INTERP_ROOT}
export HF_HOME=${UEA_SCRATCH}/huggingface SOFTFILELOCK=1 HF_HUB_DISABLE_XET=1
P=.venv/bin/python; C=configs; M="--mc-rendered configs/mc_rendered.json"
$P experiments/02_steering/build_steer_prompts.py --cell capability/run_003/v13_sft_then_dpo_nll__x100_eval/bbh --n 150 --seed 1 $M --out $C/steer2_bbh_eval_all.jsonl
$P experiments/02_steering/build_steer_prompts.py --cell capability/run_003/v13_sft_then_dpo_nll__x100_not_eval/bbh --ids-from $C/steer2_bbh_eval_all.jsonl $M --out $C/steer2_bbh_noteval_all.jsonl
$P experiments/02_steering/build_steer_prompts.py --cell deployment/run_002/v13_sft_then_dpo_nll__x100_eval/wildchat --n 150 --seed 1 --out $C/steer2_wc_eval_all.jsonl
$P experiments/02_steering/build_steer_prompts.py --cell deployment/run_002/v13_sft_then_dpo_nll__x100_not_eval/wildchat --ids-from $C/steer2_wc_eval_all.jsonl --out $C/steer2_wc_noteval_all.jsonl
$P experiments/02_steering/build_gsm8k_prompts.py --n 150 --out $C/steer2_gsm8k_all.jsonl
for s in bbh_eval bbh_noteval wc_eval wc_noteval gsm8k; do
  head -50 $C/steer2_${s}_all.jsonl > $C/steer2_${s}_dev.jsonl
  tail -n +51 $C/steer2_${s}_all.jsonl > $C/steer2_${s}_test.jsonl
done
# dev sweep file: suppression prompts + GSM8K in one generation job
cat $C/steer2_bbh_eval_dev.jsonl $C/steer2_wc_eval_dev.jsonl $C/steer2_gsm8k_dev.jsonl > $C/steer2_dev.jsonl
wc -l $C/steer2_*.jsonl
