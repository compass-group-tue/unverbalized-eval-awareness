#!/bin/bash
# Cluster-side: wait until every Stage 1 job has left the queue (terminated,
# or removed/held for > 12 h), then score and summarise. Polls job logs on
# shared storage, so it needs no scheduder access from the execute node.
cd ${UEA_INTERP_ROOT}
L=cluster/logs
want=114; t0=$(date +%s)
while true; do
  done_=$(cat $L/steer2_17612275.*.log $L/steer2hua_17612276.*.log 2>/dev/null | grep -cE "^(005|009) ")
  echo "$(date '+%F %T') $done_/$want finished"
  [ "$done_" -ge "$want" ] && break
  [ $(( $(date +%s) - t0 )) -gt 43200 ] && { echo "timeout: scoring what exists"; break; }
  sleep 300
done
echo "exit codes:"; cat $L/steer2_17612275.*.log $L/steer2hua_17612276.*.log | grep -o "exit-code [0-9]*" | sort | uniq -c
.venv/bin/python experiments/02_steering/score_steer2.py --glob "steer2_s1_*" --out results/steer2_s1_scored.csv > /dev/null
.venv/bin/python experiments/02_steering/score_steer2.py --glob "steer2_sqa_x100_base" --out results/steer2_sqa_check_scored.csv > /dev/null
.venv/bin/python experiments/02_steering/score_steer2.py --glob "steer2_wcpool_x100_base" --out results/steer2_wcpool_check_scored.csv > /dev/null
.venv/bin/python experiments/02_steering/score_steer2.py --glob "steer_2_s1_*" --out results/steer2_s1_hua_scored.csv > /dev/null
.venv/bin/python experiments/02_steering/summarise_s1.py | tee results/steer2_s1_summary.txt
