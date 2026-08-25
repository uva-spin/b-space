#!/usr/bin/env bash
# Restartable isolated supervisor for the high-statistics Tevatron W+Y batch.
# This never writes the frozen lambda=1 package.
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="/home/dustin/miniforge3/envs/pdf-fit/bin/python"
RUNNER="$ROOT/scripts/run_tevatron_full_n3ll_nnlo_grid.py"
MERGER="$ROOT/scripts/merge_tevatron_grid_parts.py"
FINALIZER="$ROOT/scripts/finalize_tevatron_n3ll_production_batch.py"
CENTRAL="$ROOT/reports/tevatron_n3ll_nnlo_wy_production_g1_1p017"
PARTS="$ROOT/reports/tevatron_n3ll_nnlo_wy_production_g1_1p017_parts"
mkdir -p "$CENTRAL" "$PARTS"

row_count() {
  "$PY" - "$1" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
try:
    s = json.loads((p / "grid_status.json").read_text())
    print(int(s.get("row_count", -1)))
except Exception:
    print(-1)
PY
}

run_until_complete() {
  local dataset="$1" expected="$2" out="$3" log="$4"
  local attempt=0 count
  while :; do
    count="$(row_count "$out")"
    if [[ "$count" == "$expected" ]]; then
      return 0
    fi
    attempt=$((attempt + 1))
    if (( attempt > 3 )); then
      echo "giving up after $attempt attempts: $dataset (row_count=$count)" >&2
      return 1
    fi
    echo "[$(date -Is)] starting $dataset attempt $attempt" >> "$log"
    "$PY" "$RUNNER" --g1 1.017 --calls 100000000 --seed 20260818 \
      --timeout 10800 --datasets "$dataset" --out "$out" >> "$log" 2>&1 || true
  done
}

run_until_complete CDF_RUN_1 41 "$CENTRAL" "$CENTRAL/production_batch.log" || exit 1

mkdir -p "$PARTS/CDF_RUN_2" "$PARTS/D0_RUN_1"
run_until_complete CDF_RUN_2 61 "$PARTS/CDF_RUN_2" "$PARTS/CDF_RUN_2/supervisor.log" & p2=$!
run_until_complete D0_RUN_1 20 "$PARTS/D0_RUN_1" "$PARTS/D0_RUN_1/supervisor.log" & p3=$!
wait "$p2" || exit 1
wait "$p3" || exit 1

"$PY" "$MERGER" --out "$CENTRAL" --g1 1.017 \
  --parts CDF_RUN_1 "$CENTRAL" "$CENTRAL" \
  --parts CDF_RUN_2 "$PARTS/CDF_RUN_2" "$PARTS/CDF_RUN_2" \
  --parts D0_RUN_1 "$PARTS/D0_RUN_1" "$PARTS/D0_RUN_1"
"$PY" "$FINALIZER"
echo "[$(date -Is)] Tevatron full W+Y batch complete" >> "$CENTRAL/production_batch.log"
