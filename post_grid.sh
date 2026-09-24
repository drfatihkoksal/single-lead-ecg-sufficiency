#!/bin/bash
# Runs after the Phase-3 grid: waits for run_grid to finish, then the feature-based
# GBM baseline (Chapman, PTB-XL), the seed aggregation and the confusion analysis. Detached and resumable
# (GBM skips finished lead specs; aggregation recomputes only when seeds changed).
#   start: ./post_grid.sh          log: artifacts_v2/post_grid.log
cd "$(dirname "$0")"
LOG=artifacts_v2/post_grid.log
nohup setsid bash -c '
  while ps -eo args | grep -qE "^python -m src\.run_grid"; do sleep 300; done
  echo "=== grid finished; post-grid started $(date) ==="
  for ds in chapman ptbxl_snomed; do
    PYTHONUNBUFFERED=1 python -m src.features_gbm --dataset $ds --seed 1337 || echo "!!! gbm $ds failed"
  done
  PYTHONUNBUFFERED=1 python -m src.aggregate_seeds --jobs 8 && PYTHONUNBUFFERED=1 python -m src.confusion_v2 && echo "=== post-grid done $(date) ==="
' >> "$LOG" 2>&1 < /dev/null &
echo "post-grid waiter started (pid $!), log: $LOG"
