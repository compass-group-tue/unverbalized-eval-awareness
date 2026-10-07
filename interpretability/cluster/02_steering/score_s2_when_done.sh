#!/bin/bash
# Cluster-side: wait for both Stage 2 clusters (82 jobs), then score and summarise.
cd ${UEA_INTERP_ROOT}
L=cluster/logs; A=$1; B=$2; want="${3:-82}"; t0=$(date +%s)
while true; do
  n=$(cat $L/steer2_$A.*.log $L/steer2_$B.*.log 2>/dev/null | grep -cE "^(005|009) ")
  echo "$(date '+%F %T') $n/$want finished"
  [ "$n" -ge "$want" ] && break
  [ $(( $(date +%s) - t0 )) -gt 43200 ] && { echo "timeout: scoring what exists"; break; }
  sleep 180
done
echo "exit codes:"; cat $L/steer2_$A.*.log $L/steer2_$B.*.log | grep -o "exit-code [0-9]*" | sort | uniq -c
.venv/bin/python experiments/02_steering/score_steer2.py --glob "steer2_s2_*" --out results/steer2_s2_scored.csv > /dev/null
.venv/bin/python experiments/02_steering/summarise_s2.py | tee results/steer2_s2_summary.txt
