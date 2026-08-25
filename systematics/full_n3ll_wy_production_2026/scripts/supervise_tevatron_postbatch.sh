#!/usr/bin/env bash
# Row-count-guarded post-processing for the isolated Tevatron W+Y campaign.
# This intentionally waits for complete 122-row tables; a partial CDF-I table
# must never trigger finalization or replica propagation.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="/home/dustin/miniforge3/envs/pdf-fit/bin/python"
PRIMARY="$ROOT/reports/tevatron_n3ll_nnlo_wy_production_g1_1p017"
STATION="$ROOT/reports/tevatron_n3ll_nnlo_wy_stationarity_g1_1p017_seed_20260820"
FINALIZER="$ROOT/scripts/finalize_tevatron_n3ll_production_batch.py"
COMPARE="$ROOT/scripts/compare_tevatron_production_stationarity.py"
REPLICA="$ROOT/scripts/propagate_tevatron_wy_replicas_500.py"
SUMMARY="$ROOT/scripts/summarize_tevatron_n3ll_campaign.py"
FREEZE="$ROOT/scripts/freeze_tevatron_candidate.py"
REFINE="$ROOT/scripts/refine_tevatron_primary_grid.py"
PLOT="$ROOT/scripts/plot_tevatron_n3ll_wy_grid.py"
DECOMP="$ROOT/scripts/run_tevatron_wy_term_decomposition.py"
DECOMP_OUT="$ROOT/reports/dyturbo_term_decomposition_g1_1p017"
CLOSURE="$ROOT/scripts/write_candidate_accuracy_closure.py"
CLOSURE_OUT="$ROOT/reports/accuracy_closure_g1_1p017.json"
OUT="$ROOT/reports/tevatron_n3ll_nnlo_wy_replica_profile_500_g1_1p017"
LOG="$ROOT/reports/tevatron_n3ll_nnlo_wy_production_g1_1p017/postbatch_supervisor.log"

count() {
  "$PY" - "$1" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1]) / "grid_status.json"
try:
    print(int(json.loads(p.read_text()).get("row_count", -1)))
except Exception:
    print(-1)
PY
}

echo "[$(date -Is)] waiting for complete 122-row primary and stationarity grids" >> "$LOG"
while [[ "$(count "$PRIMARY")" != 122 || "$(count "$STATION")" != 122 ]]; do
  echo "[$(date -Is)] primary=$(count "$PRIMARY") stationarity=$(count "$STATION")" >> "$LOG"
  sleep 60
done

echo "[$(date -Is)] both complete; running precision refinement then fail-closed finalizer" >> "$LOG"
"$PY" "$REFINE" --grid "$PRIMARY/tevatron_full_wy_grid.csv" --out "$PRIMARY/precision_refinement" --g1 1.017 --calls 300000000 --data-error-fraction 0.5 --max-workers 3 >> "$LOG" 2>&1
if [[ ! -f "$DECOMP_OUT/term_decomposition_status.json" ]]; then
  echo "[$(date -Is)] running candidate-center RES/CT/VJ decomposition at g1=1.017" >> "$LOG"
  "$PY" "$DECOMP" --g1 1.017 --calls 30000000 --out "$DECOMP_OUT" >> "$LOG" 2>&1
else
  echo "[$(date -Is)] candidate-center term decomposition already present; leaving it unchanged" >> "$LOG"
fi
if [[ ! -f "$CLOSURE_OUT" ]]; then
  echo "[$(date -Is)] writing candidate-center perturbative accuracy closure" >> "$LOG"
  "$PY" "$CLOSURE" >> "$LOG" 2>&1
else
  echo "[$(date -Is)] candidate-center accuracy closure already present; leaving it unchanged" >> "$LOG"
fi
"$PY" "$FINALIZER" >> "$LOG" 2>&1
"$PY" "$COMPARE" \
  --primary "$PRIMARY/tevatron_full_wy_grid.csv" \
  --seed "$STATION/tevatron_full_wy_grid.csv" \
  --out "$STATION/stationarity_status.json" >> "$LOG" 2>&1

if [[ ! -f "$OUT/replica_profile_status.json" ]]; then
  echo "[$(date -Is)] starting 500-member diagonal+normalization replica diagnostic" >> "$LOG"
  "$PY" "$REPLICA" --primary-grid "$PRIMARY/tevatron_full_wy_grid.csv" \
    --replicas 500 --seed 20260819 --out "$OUT" >> "$LOG" 2>&1
else
  echo "[$(date -Is)] replica diagnostic already present; leaving it unchanged" >> "$LOG"
fi

"$PY" "$ROOT/scripts/verify_frozen_baseline.py" >> "$LOG" 2>&1
"$PY" "$PLOT" --grid "$PRIMARY/tevatron_full_wy_grid.csv" --out "$PRIMARY/figures" >> "$LOG" 2>&1
"$PY" "$SUMMARY" >> "$LOG" 2>&1
"$PY" "$FREEZE" >> "$LOG" 2>&1
echo "[$(date -Is)] postbatch processing complete; candidate remains unpromoted" >> "$LOG"
