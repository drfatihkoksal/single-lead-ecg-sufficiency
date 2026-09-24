#!/bin/bash
# Sensitivity analysis with the broad label definition: SE-ResNet, seed 1337, all lead
# specs, on '<cohort>_broad' (same signals and split, broad labels). Detached, resumable,
# self-restarting; aggregated into artifacts_v2/aggregate_broad/ when done.
#   start: ./run_broad.sh          log: artifacts_v2/broad.log
cd "$(dirname "$0")"
LOG=artifacts_v2/broad.log
nohup setsid bash -c '
  for i in $(seq 0 10); do
    PYTHONUNBUFFERED=1 python -m src.run_grid \
      --datasets chapman_broad georgia_broad ptbxl_snomed_broad ningbo_broad \
      --archs seresnet --seeds 1337 --devices && break
    echo "=== broad run exited with an error; restart $((i+1))/10 at $(date) ==="; sleep 60
  done
  PYTHONUNBUFFERED=1 python -m src.aggregate_seeds --jobs 8 --archs seresnet \
    --cohorts chapman_broad georgia_broad ptbxl_snomed_broad ningbo_broad \
    --out artifacts_v2/aggregate_broad && echo "=== broad done $(date) ==="
' >> "$LOG" 2>&1 < /dev/null &
echo "broad run started (pid $!), log: $LOG"
