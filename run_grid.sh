#!/bin/bash
# Phase-3 experiment grid, detached from the terminal and self-restarting.
# xresnet1d101 was dropped on 2026-09-23 (a ResNet variant; Chapman seed-1337 results kept
# as a supplementary third-architecture check). Every level is resumable (finished runs are skipped, interrupted training continues
# from its checkpoint), so this script can simply be started again after a reboot.
#   start:  ./run_grid.sh            progress:  tail -f artifacts_v2/grid.log
#   stop:   pkill -f run_grid.sh; pkill -f src.run_grid
cd "$(dirname "$0")"
LOG=artifacts_v2/grid.log
MAX_RESTARTS=20
nohup setsid bash -c "
  for i in \$(seq 0 $MAX_RESTARTS); do
    PYTHONUNBUFFERED=1 python -m src.run_grid \
      --datasets chapman georgia ptbxl_snomed ningbo \
      --archs seresnet inceptiontime \
      --seeds 1337 1 2 3 4 --devices && exit 0
    echo \"=== run_grid exited with an error; restart \$((i+1))/$MAX_RESTARTS at \$(date) ===\"
    sleep 60
  done
" >> "$LOG" 2>&1 < /dev/null &
echo "grid started (pid $!), log: $LOG"
