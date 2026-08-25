#!/usr/bin/env bash
# Restart-safe downstream guard for the isolated Tevatron N3LL+NNLO run.
# This is deliberately separate from the long-running grid supervisor so a
# shell started before a script patch cannot skip the final propagation.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="/home/dustin/miniforge3/envs/pdf-fit/bin/python"
OUT="$ROOT/reports/tevatron_n3ll_nnlo_wy_final_g1_1p017"
SCALE="$ROOT/reports/tevatron_scale_variations_g1_1p017"
LOG="$OUT/postgrid_completion_supervisor.log"
GRID="$OUT/grid_status.json"
Y="$OUT/conventional_y/y_grid_status.json"
REPL="$OUT/replica_profile_500/replica_profile_status.json"
PROP="$ROOT/reports/tevatron_fnp_start_replica_propagation/propagation_status.json"
mkdir -p "$OUT"

good_grid() {
  [[ -f "$GRID" ]] || return 1
  "$PY" - "$GRID" <<'PY'
import json, sys
s=json.load(open(sys.argv[1])); c=s.get("checks", {})
raise SystemExit(0 if s.get("row_count") == 122 and c.get("all_finite") and c.get("all_positive") else 1)
PY
}

good_scale() {
  [[ -f "$SCALE/scale_variation_status.json" && -f "$SCALE/scale_variation_refinement_status.json" ]] || return 1
  "$PY" - "$SCALE/scale_variation_status.json" "$SCALE/scale_variation_refinement_status.json" <<'PY'
import json, sys
a=json.load(open(sys.argv[1])); b=json.load(open(sys.argv[2]))
raise SystemExit(0 if a.get("row_count") == 122 and a.get("all_finite") and a.get("all_positive")
                 and b.get("all_finite") and b.get("all_positive") else 1)
PY
}

echo "[$(date -Is)] waiting for finite/positive final grid and scale gates" >> "$LOG"
while ! good_grid || ! good_scale; do sleep 60; done
echo "[$(date -Is)] grid and scale gates passed" >> "$LOG"

# The primary supervisor may reach this point at the same time.  Hold a
# shared lock through all downstream work; a second supervisor then reuses the
# completed statuses instead of duplicating expensive refinements.
exec 9>"$OUT/downstream_completion.lock"
flock 9
echo "[$(date -Is)] acquired downstream completion lock" >> "$LOG"

# The shared lock already serializes this expensive refinement.  Whoever owns
# the lock may run it; do not wait on the primary supervisor here, because the
# primary can itself be blocked waiting for this same lock.
REFINE="$ROOT/scripts/refine_tevatron_primary_grid.py"
if [[ ! -f "$OUT/precision_refinement/primary_refinement_status.json" ]]; then
  echo "[$(date -Is)] running missing primary precision refinement under shared lock" >> "$LOG"
  "$PY" "$REFINE" --grid "$OUT/tevatron_full_wy_grid.csv" \
    --out "$OUT/precision_refinement" --g1 1.017 --calls 300000000 \
    --data-error-fraction 0.5 --max-workers 3 >> "$LOG" 2>&1
fi

FINALIZER="$ROOT/scripts/finalize_tevatron_final_wy_production.py"
YEXTRACT="$ROOT/scripts/extract_tevatron_y_from_dyturbo_logs.py"
REPLICAS="$ROOT/scripts/propagate_tevatron_wy_replicas_500.py"
PROPAGATE="$ROOT/scripts/build_tevatron_fnp_start_replica_propagation.py"
PLOT="$ROOT/scripts/plot_tevatron_n3ll_wy_grid.py"
SCALEPLOT="$ROOT/scripts/plot_tevatron_scale_envelope.py"

"$PY" "$FINALIZER" >> "$LOG" 2>&1
if [[ ! -f "$Y" ]]; then
  echo "[$(date -Is)] extracting conventional Y" >> "$LOG"
  "$PY" "$YEXTRACT" --grid-dir "$OUT" --out "$OUT/conventional_y" >> "$LOG" 2>&1
fi
if [[ ! -f "$REPL" ]]; then
  echo "[$(date -Is)] running 500-replica perturbative diagnostic" >> "$LOG"
  "$PY" "$REPLICAS" --primary-grid "$OUT/tevatron_full_wy_grid.csv" \
    --out "$OUT/replica_profile_500" --replicas 500 --seed 20260822 >> "$LOG" 2>&1
fi
if [[ ! -f "$PROP" ]] || ! "$PY" - "$PROP" <<'PY'
import json, sys
try: s=json.load(open(sys.argv[1]))
except Exception: raise SystemExit(1)
raise SystemExit(0 if s.get("status") == "isolated_tevatron_full_fnp_start_replica_propagation_complete_not_production" else 1)
PY
then
  echo "[$(date -Is)] running full 96-start x 50-replica propagation" >> "$LOG"
  "$PY" "$PROPAGATE" >> "$LOG" 2>&1
fi
mkdir -p "$OUT/figures"
"$PY" "$PLOT" --grid "$OUT/tevatron_full_wy_grid.csv" --out "$OUT/figures" >> "$LOG" 2>&1
"$PY" "$SCALEPLOT" --csv "$SCALE/tevatron_scale_variations.csv" --out "$OUT/figures" >> "$LOG" 2>&1
"$PY" "$FINALIZER" >> "$LOG" 2>&1
echo "[$(date -Is)] downstream diagnostics complete; promotion remains fail-closed" >> "$LOG"
