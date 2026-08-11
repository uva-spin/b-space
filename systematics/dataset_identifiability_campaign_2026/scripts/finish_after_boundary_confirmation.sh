#!/usr/bin/env bash
set -euo pipefail

base=/home/dustin/work/project/bT-TMD/systematics/dataset_identifiability_campaign_2026
python_bin=/home/dustin/miniforge3/envs/pdf-fit/bin/python
summary=$base/summaries/selected_reference_central_replicas/summary.json
failed=$base/summaries/selected_reference_central_replicas/failed_80k_summary.json

archive_prior_failure() {
    "$python_bin" - "$failed" <<'PY'
import json
from pathlib import Path
import sys

source = Path(sys.argv[1])
if source.exists():
    strength = float(json.loads(source.read_text())["selected_strength"])
    token = f"{strength:g}".replace(".", "p")
    archive = source.with_name(f"failed_lam{token}_80k_summary.json")
    if archive.exists():
        raise SystemExit(f"refusing to overwrite existing failure archive: {archive}")
    source.rename(archive)
PY
}

# The outer recovery chain owns the current fresh central+50 run, any
# stronger-strength recursion required by an outright stationarity failure,
# and its deliberately blocked first downstream-build attempt.  Wait for that
# entire owner to exit before taking over, avoiding a race with nested
# supervisors whose PIDs can change between strength generations.
while kill -0 1418045 2>/dev/null; do
    sleep 30
done

# A barely passing endpoint receives a boundary confirmation.  If it cannot
# re-establish the original stationarity rule within the bounded additional
# confirmation horizon, feed that newly exposed
# hard replica back into the existing augmented-strength calibration and try
# the resulting fresh 24+50 generation before proceeding.
while ! "$python_bin" "$base/scripts/confirm_boundary_replicas.py" \
    >> "$base/logs/selected_reference_boundary_confirmation.log" 2>&1; do
    "$python_bin" "$base/scripts/alternate_lambda_authorization.py" \
        --launcher finish_after_boundary_confirmation --require \
        >> "$base/logs/reference_strength_boundary_escalation.log" 2>&1
    archive_prior_failure
    mv "$summary" "$failed"
    "$python_bin" "$base/scripts/escalate_reference_strength_after_full50_failure.py" \
        >> "$base/logs/reference_strength_boundary_escalation.log" 2>&1
done

"$python_bin" "$base/scripts/build_final_combined_tmd_ensemble.py" \
    >> "$base/logs/final_combined_tmd_ensemble.boundary_supervisor.log" 2>&1
"$python_bin" "$base/scripts/audit_final_combined_ensemble.py" \
    >> "$base/logs/final_combined_ensemble_stability.boundary_supervisor.log" 2>&1
"$python_bin" "$base/scripts/plot_validated_final_fig2_fig6.py" \
    >> "$base/logs/final_fig2_fig6.boundary_supervisor.log" 2>&1
"$python_bin" "$base/scripts/audit_campaign_completion.py" \
    >> "$base/logs/campaign_completion_audit.boundary_supervisor.log" 2>&1
"$python_bin" "$base/scripts/write_final_study_report.py" \
    >> "$base/logs/final_study_report.boundary_supervisor.log" 2>&1
