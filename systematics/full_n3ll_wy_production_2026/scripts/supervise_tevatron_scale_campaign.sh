#!/usr/bin/env bash
# Continue perturbative-accuracy checks after the central candidate is frozen.
# All outputs remain under the isolated campaign directory.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="/home/dustin/miniforge3/envs/pdf-fit/bin/python"
MARKER="$ROOT/manifests/tevatron_n3ll_nnlo_wy_candidate_freeze.json"
SCALE="$ROOT/scripts/run_tevatron_full_wy_scale_variations.py"
REFINE="$ROOT/scripts/refine_tevatron_scale_variation_rows.py"
OUT="$ROOT/reports/tevatron_scale_variations_g1_1p017"
LOG="$OUT/scale_supervisor.log"
mkdir -p "$OUT"

valid_scale_status() {
  [[ -f "$OUT/scale_variation_status.json" ]] || return 1
  "$PY" - "$OUT/scale_variation_status.json" <<'PY'
import json, sys
s=json.load(open(sys.argv[1]))
raise SystemExit(0 if (s.get("row_count") == 122 and s.get("all_finite") and s.get("all_positive")) else 1)
PY
}

valid_refinement_status() {
  [[ -f "$OUT/scale_variation_refinement_status.json" ]] || return 1
  "$PY" - "$OUT/scale_variation_refinement_status.json" <<'PY'
import json, sys
s=json.load(open(sys.argv[1]))
raise SystemExit(0 if (s.get("all_finite") and s.get("all_positive")) else 1)
PY
}

if [[ -f "$OUT/scale_variation_status.json" ]] && ! valid_scale_status; then
  bad="$OUT/recovery_invalid_attempts/$(date +%Y%m%dT%H%M%S)_scale"
  mkdir -p "$bad"
  mv "$OUT/scale_variation_status.json" "$bad/"
  [[ -f "$OUT/tevatron_scale_variations.csv" ]] && mv "$OUT/tevatron_scale_variations.csv" "$bad/"
  rm -f "$OUT/scale_variation_refinement_status.json"
  echo "[$(date -Is)] quarantined invalid scale status before retry" >> "$LOG"
fi
if [[ -f "$OUT/scale_variation_refinement_status.json" ]] && ! valid_refinement_status; then
  bad="$OUT/recovery_invalid_attempts/$(date +%Y%m%dT%H%M%S)_refinement"
  mkdir -p "$bad"
  mv "$OUT/scale_variation_refinement_status.json" "$bad/"
  echo "[$(date -Is)] quarantined invalid scale refinement before retry" >> "$LOG"
fi

echo "[$(date -Is)] waiting for candidate freeze" >> "$LOG"
while [[ ! -f "$MARKER" ]]; do sleep 60; done

if [[ ! -f "$OUT/scale_variation_status.json" ]]; then
  echo "[$(date -Is)] starting seven-point 3M-call scale scan" >> "$LOG"
  scan_ok=0
  for seed in 20263100 20263300 20263400; do
    echo "[$(date -Is)] scale scan seed=$seed" >> "$LOG"
    if "$PY" "$SCALE" --g1 1.017 --calls 3000000 --seed "$seed" --out "$OUT" >> "$LOG" 2>&1; then
      scan_ok=1
      break
    fi
    echo "[$(date -Is)] scale scan seed=$seed failed; rotating seed" >> "$LOG"
  done
  [[ "$scan_ok" -eq 1 ]] || { echo "[$(date -Is)] all scale scan seeds failed" >> "$LOG"; exit 1; }
else
  echo "[$(date -Is)] seven-point scale scan already present" >> "$LOG"
fi

if [[ ! -f "$OUT/scale_variation_refinement_status.json" ]]; then
  echo "[$(date -Is)] refining scale rows above 50% data-relative precision" >> "$LOG"
  refine_ok=0
  for seed in 20263200 20263500 20263600; do
    echo "[$(date -Is)] scale refinement seed=$seed" >> "$LOG"
    if "$PY" "$REFINE" --g1 1.017 --base "$OUT" --calls 30000000 --threshold 0.5 --seed "$seed" >> "$LOG" 2>&1; then
      refine_ok=1
      break
    fi
    echo "[$(date -Is)] scale refinement seed=$seed failed; rotating seed" >> "$LOG"
  done
  [[ "$refine_ok" -eq 1 ]] || { echo "[$(date -Is)] all scale refinement seeds failed" >> "$LOG"; exit 1; }
else
  echo "[$(date -Is)] scale refinement already present" >> "$LOG"
fi

echo "[$(date -Is)] scale campaign complete; remains diagnostic until precision gate passes" >> "$LOG"
