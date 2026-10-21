#!/bin/bash
# Limb lead + chest lead sets (I+V1, I+V5, II+V5; PREREGISTRATION_lateral_sets.md), detached and
# resumable: the grid skips finished runs, so only the new sets are trained. Then the broad-label
# SE-ResNet run, the feature-based model, and the aggregation and review analyses.
#   start: ./run_lateral.sh          log: artifacts_v2/lateral.log
cd "$(dirname "$0")"
LOG=artifacts_v2/lateral.log
nohup setsid bash -c '
  for i in $(seq 0 10); do
    PYTHONUNBUFFERED=1 python -m src.run_grid --datasets chapman georgia ptbxl_snomed ningbo \
      --archs seresnet inceptiontime --seeds 1337 1 2 3 4 --devices && break
    echo "=== grid exited with an error; restart $((i+1))/10 at $(date) ==="; sleep 60
  done
  for i in $(seq 0 5); do
    PYTHONUNBUFFERED=1 python -m src.run_grid --datasets chapman_broad georgia_broad ptbxl_snomed_broad ningbo_broad \
      --archs seresnet --seeds 1337 --devices && break
    echo "=== broad grid exited with an error; restart $((i+1))/5 at $(date) ==="; sleep 60
  done
  for ds in chapman ptbxl_snomed; do
    PYTHONUNBUFFERED=1 python -m src.features_gbm --dataset $ds --seed 1337 || echo "!!! gbm $ds failed"
  done
  PYTHONUNBUFFERED=1 python -m src.aggregate_seeds --force --jobs 12 &&
  PYTHONUNBUFFERED=1 python -m src.aggregate_seeds --force --jobs 4 --archs seresnet \
    --cohorts chapman_broad georgia_broad ptbxl_snomed_broad ningbo_broad --out artifacts_v2/aggregate_broad &&
  rm -f artifacts_v2/review/bootstrap.csv &&
  PYTHONUNBUFFERED=1 python -m src.review_analyses --jobs 24 &&
  PYTHONUNBUFFERED=1 python -m src.review_analyses --only ordinal &&
  PYTHONUNBUFFERED=1 python -m src.review2_analyses --jobs 24 &&
  echo "=== LATERAL DONE $(date) ==="
' >> "$LOG" 2>&1 < /dev/null &
echo "lateral run started (pid $!), log: $LOG"
