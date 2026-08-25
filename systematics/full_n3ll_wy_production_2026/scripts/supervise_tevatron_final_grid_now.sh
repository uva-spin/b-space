#!/usr/bin/env bash
# Start the final isolated Tevatron W+Y grid as soon as the candidate freeze is
# present.  The downstream final supervisor reuses this grid after the scale
# gate; this service never writes the frozen lambda=1 package.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="/home/dustin/miniforge3/envs/pdf-fit/bin/python"
VERIFY="$ROOT/scripts/verify_frozen_baseline.py"
RUNNER="$ROOT/scripts/run_tevatron_full_n3ll_nnlo_grid.py"
MARKER="$ROOT/manifests/tevatron_n3ll_nnlo_wy_candidate_freeze.json"
OUT="$ROOT/reports/tevatron_n3ll_nnlo_wy_final_g1_1p017"
LOG="$OUT/grid_now_supervisor.log"
mkdir -p "$OUT"

while [[ ! -f "$MARKER" ]]; do sleep 60; done

complete_rows() {
  [[ -f "$OUT/grid_status.json" ]] || return 1
  "$PY" - "$OUT/grid_status.json" <<'PY'
import json, sys
s=json.load(open(sys.argv[1]))
checks=s.get("checks", {})
raise SystemExit(0 if (int(s.get("row_count", -1)) == 122
                       and bool(checks.get("all_finite", False))
                       and bool(checks.get("all_positive", False))) else 1)
PY
}

if complete_rows; then
  echo "[$(date -Is)] final 122-row grid already complete" >> "$LOG"
  exit 0
fi

echo "[$(date -Is)] verifying frozen baseline" >> "$LOG"
"$PY" "$VERIFY" >> "$LOG" 2>&1
echo "[$(date -Is)] starting final 122-row grid in parallel with scale diagnostics" >> "$LOG"
"$PY" "$RUNNER" --g1 1.017 --calls 100000000 --seed 20260821 \
  --timeout 10800 --datasets CDF_RUN_1 CDF_RUN_2 D0_RUN_1 --out "$OUT" \
  >> "$LOG" 2>&1
echo "[$(date -Is)] final grid completed; downstream supervisor will finalize after scale gate" >> "$LOG"
