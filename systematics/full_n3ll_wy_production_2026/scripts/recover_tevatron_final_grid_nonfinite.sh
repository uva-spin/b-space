#!/usr/bin/env bash
# Recover a candidate final grid if a DYTurbo seed produces nonfinite rows.
# Invalid attempts are quarantined, never silently overwritten.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="/home/dustin/miniforge3/envs/pdf-fit/bin/python"
OUT="$ROOT/reports/tevatron_n3ll_nnlo_wy_final_g1_1p017"
RUNNER="$ROOT/scripts/run_tevatron_full_n3ll_nnlo_grid.py"
LOG="$OUT/grid_recovery_supervisor.log"
mkdir -p "$OUT/recovery_invalid_attempts"

valid_grid() {
  [[ -f "$OUT/grid_status.json" ]] || return 1
  "$PY" - "$OUT/grid_status.json" <<'PY'
import json, sys
s=json.load(open(sys.argv[1])); c=s.get("checks", {})
raise SystemExit(0 if (int(s.get("row_count", -1)) == 122
                       and bool(c.get("all_finite", False))
                       and bool(c.get("all_positive", False))) else 1)
PY
}

# Wait for the first run to publish its status. The running service is
# identified by its command line and is allowed to finish normally. Exclude
# this script's own pgrep command, whose pattern would otherwise self-match.
has_first_runner() {
  pgrep -x -f '/home/dustin/miniforge3/envs/pdf-fit/bin/python /home/dustin/work/project/bT-TMD/systematics/full_n3ll_wy_production_2026/scripts/run_tevatron_full_n3ll_nnlo_grid.py .*--out /home/dustin/work/project/bT-TMD/systematics/full_n3ll_wy_production_2026/reports/tevatron_n3ll_nnlo_wy_final_g1_1p017' >/dev/null 2>&1
}
while [[ ! -f "$OUT/grid_status.json" ]]; do
  if ! has_first_runner; then
    # If the first supervisor exited before publishing a status, start the
    # recovery seed directly.
    break
  fi
  sleep 60
done

if valid_grid; then
  echo "[$(date -Is)] final grid is finite/positive; no recovery needed" >> "$LOG"
  exit 0
fi

attempt=0
for seed in 20260823 20260824 20260825; do
  attempt=$((attempt + 1))
  stamp="$(date +%Y%m%dT%H%M%S)_attempt${attempt}"
  if [[ -f "$OUT/tevatron_full_wy_grid.csv" ]]; then
    mv "$OUT/tevatron_full_wy_grid.csv" "$OUT/recovery_invalid_attempts/tevatron_full_wy_grid_${stamp}.csv"
  fi
  if [[ -f "$OUT/grid_status.json" ]]; then
    mv "$OUT/grid_status.json" "$OUT/recovery_invalid_attempts/grid_status_${stamp}.json"
  fi
  echo "[$(date -Is)] rerunning nonfinite final grid with seed=$seed" >> "$LOG"
  "$PY" "$RUNNER" --g1 1.017 --calls 100000000 --seed "$seed" \
    --timeout 10800 --datasets CDF_RUN_1 CDF_RUN_2 D0_RUN_1 --out "$OUT" \
    >> "$LOG" 2>&1 || true
  if valid_grid; then
    echo "[$(date -Is)] recovery seed=$seed passed finite/positive gate" >> "$LOG"
    exit 0
  fi
done

echo "[$(date -Is)] all recovery seeds failed finite/positive gate" >> "$LOG"
exit 1
