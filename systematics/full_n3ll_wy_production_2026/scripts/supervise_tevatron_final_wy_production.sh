#!/usr/bin/env bash
# Launch the isolated genuine Tevatron N3LL+NNLO W+Y production after the
# candidate freeze and perturbative scale diagnostics are available.
# Never writes the frozen lambda=1 package.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="/home/dustin/miniforge3/envs/pdf-fit/bin/python"
MARKER="$ROOT/manifests/tevatron_n3ll_nnlo_wy_candidate_freeze.json"
SCALE="$ROOT/reports/tevatron_scale_variations_g1_1p017"
RUNNER="$ROOT/scripts/run_tevatron_full_n3ll_nnlo_grid.py"
FINALIZER="$ROOT/scripts/finalize_tevatron_final_wy_production.py"
HANDOFF="$ROOT/scripts/record_final_tevatron_handoff.py"
YEXTRACT="$ROOT/scripts/extract_tevatron_y_from_dyturbo_logs.py"
REFINE="$ROOT/scripts/refine_tevatron_primary_grid.py"
VERIFY="$ROOT/scripts/verify_frozen_baseline.py"
REPLICAS="$ROOT/scripts/propagate_tevatron_wy_replicas_500.py"
PLOT="$ROOT/scripts/plot_tevatron_n3ll_wy_grid.py"
SCALEPLOT="$ROOT/scripts/plot_tevatron_scale_envelope.py"
PROPAGATE="$ROOT/scripts/build_tevatron_fnp_start_replica_propagation.py"
RECOVER="$ROOT/scripts/recover_tevatron_final_grid_nonfinite.sh"
SCALESUP="$ROOT/scripts/supervise_tevatron_scale_campaign.sh"
OUT="$ROOT/reports/tevatron_n3ll_nnlo_wy_final_g1_1p017"
LOG="$OUT/production_supervisor.log"
mkdir -p "$OUT"

echo "[$(date -Is)] waiting for candidate freeze" >> "$LOG"
while [[ ! -f "$MARKER" ]]; do sleep 60; done

echo "[$(date -Is)] verifying read-only frozen baseline" >> "$LOG"
"$PY" "$VERIFY" >> "$LOG"

echo "[$(date -Is)] waiting for scale scan/refinement" >> "$LOG"
scale_gate() {
  [[ -f "$SCALE/scale_variation_status.json" && -f "$SCALE/scale_variation_refinement_status.json" ]] || return 1
  "$PY" - "$SCALE/scale_variation_status.json" "$SCALE/scale_variation_refinement_status.json" <<'PY'
import json, sys
a=json.load(open(sys.argv[1])); b=json.load(open(sys.argv[2]))
raise SystemExit(0 if (a.get("row_count") == 122 and a.get("all_finite") and a.get("all_positive")
                       and b.get("all_finite") and b.get("all_positive")) else 1)
PY
}
while ! scale_gate; do
  # Keep the scale prerequisite unattended if a finite/positive gate failure
  # causes its retry supervisor to exit.  The launcher is idempotent because
  # the scale supervisor quarantines invalid partial status before restarting.
  if ! pgrep -f "$SCALESUP" >/dev/null 2>&1; then
    nohup bash "$SCALESUP" >> "$SCALE/scale_supervisor_relaunch.log" 2>&1 &
  fi
  sleep 60
done

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

if ! complete_rows; then
  echo "[$(date -Is)] waiting for finite final grid/recovery supervisor" >> "$LOG"
while ! complete_rows; do
    if ! pgrep -x -f '/bin/bash /home/dustin/work/project/bT-TMD/systematics/full_n3ll_wy_production_2026/scripts/recover_tevatron_final_grid_nonfinite.sh' >/dev/null 2>&1; then
      "$RECOVER" >> "$LOG" 2>&1 || true
    fi
    sleep 60
  done
else
  echo "[$(date -Is)] final 122-row grid already present; reusing it" >> "$LOG"
fi

# A restart-safe post-grid watchdog may be waiting on the same prerequisites.
# Serialize the expensive refinement/propagation stage so two supervisors
# cannot launch duplicate DYTurbo work when the gates open together.
exec 9>"$OUT/downstream_completion.lock"
flock 9
echo "[$(date -Is)] acquired downstream completion lock" >> "$LOG"

if [[ ! -f "$OUT/precision_refinement/primary_refinement_status.json" ]]; then
  echo "[$(date -Is)] refining final-grid cancellation rows at 300M calls" >> "$LOG"
  "$PY" "$REFINE" --grid "$OUT/tevatron_full_wy_grid.csv" \
    --out "$OUT/precision_refinement" --g1 1.017 --calls 300000000 \
    --data-error-fraction 0.5 --max-workers 3 >> "$LOG" 2>&1
fi

"$PY" "$FINALIZER" >> "$LOG" 2>&1
if [[ ! -f "$OUT/conventional_y/y_grid_status.json" ]]; then
  echo "[$(date -Is)] extracting explicit conventional Y grid from DYTurbo term logs" >> "$LOG"
  "$PY" "$YEXTRACT" --grid-dir "$OUT" --out "$OUT/conventional_y" >> "$LOG" 2>&1
fi
if [[ ! -f "$OUT/replica_profile_500/replica_profile_status.json" ]]; then
  echo "[$(date -Is)] propagating 500 diagonal-plus-normalization replicas" >> "$LOG"
  "$PY" "$REPLICAS" --primary-grid "$OUT/tevatron_full_wy_grid.csv" \
    --out "$OUT/replica_profile_500" --replicas 500 --seed 20260822 \
    >> "$LOG" 2>&1
fi
if [[ ! -f "$ROOT/reports/tevatron_fnp_start_replica_propagation/propagation_status.json" ]] || \
   ! "$PY" - "$ROOT/reports/tevatron_fnp_start_replica_propagation/propagation_status.json" <<'PY'
import json, sys
s=json.load(open(sys.argv[1]))
raise SystemExit(0 if s.get("status") == "isolated_tevatron_full_fnp_start_replica_propagation_complete_not_production" else 1)
PY
then
  echo "[$(date -Is)] building full 96-start x 50-replica F_NP/TMD propagation" >> "$LOG"
  "$PY" "$PROPAGATE" >> "$LOG" 2>&1
fi
echo "[$(date -Is)] writing final Tevatron W+Y grid diagnostic figure" >> "$LOG"
"$PY" "$PLOT" --grid "$OUT/tevatron_full_wy_grid.csv" --out "$OUT/figures" >> "$LOG" 2>&1
"$PY" "$SCALEPLOT" --csv "$SCALE/tevatron_scale_variations.csv" --out "$OUT/figures" >> "$LOG" 2>&1
"$PY" "$FINALIZER" >> "$LOG" 2>&1
echo "[$(date -Is)] recording final isolated Tevatron handoff" >> "$LOG"
"$PY" "$HANDOFF" >> "$LOG" 2>&1
echo "[$(date -Is)] final Tevatron W+Y production grid complete; promotion remains fail-closed" >> "$LOG"
