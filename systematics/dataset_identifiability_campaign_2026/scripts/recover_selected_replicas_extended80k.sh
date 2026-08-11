#!/usr/bin/env bash
set -euo pipefail

base=/home/dustin/work/project/bT-TMD/systematics/dataset_identifiability_campaign_2026
python_bin=/home/dustin/miniforge3/envs/pdf-fit/bin/python
summary=$base/summaries/selected_reference_central_replicas/summary.json

while kill -0 1365397 2>/dev/null; do
    sleep 30
done

if [[ -f "$summary" ]] && ! "$python_bin" - "$summary" <<'PY'
import json
import sys
raise SystemExit(0 if json.load(open(sys.argv[1])).get("status") == "complete" else 1)
PY
then
    mv "$summary" "$base/summaries/selected_reference_central_replicas/failed_40k_summary.json"
    "$python_bin" "$base/scripts/supervise_selected_reference_central_replicas.py" \
        >> "$base/logs/selected_reference_central_replicas.extended80k.log" 2>&1
fi

if ! "$python_bin" - "$summary" <<'PY'
import json
import sys
raise SystemExit(0 if json.load(open(sys.argv[1])).get("status") == "complete" else 1)
PY
then
    "$python_bin" "$base/scripts/alternate_lambda_authorization.py" \
        --launcher recover_selected_replicas_extended80k --require \
        >> "$base/logs/reference_strength_full50_escalation.log" 2>&1
    mv "$summary" "$base/summaries/selected_reference_central_replicas/failed_80k_summary.json"
    "$python_bin" "$base/scripts/escalate_reference_strength_after_full50_failure.py" \
        >> "$base/logs/reference_strength_full50_escalation.log" 2>&1
fi

"$python_bin" "$base/scripts/build_final_combined_tmd_ensemble.py" \
    >> "$base/logs/final_combined_tmd_ensemble.supervisor.log" 2>&1
"$python_bin" "$base/scripts/audit_final_combined_ensemble.py" \
    >> "$base/logs/final_combined_ensemble_stability.supervisor.log" 2>&1
"$python_bin" "$base/scripts/plot_validated_final_fig2_fig6.py" \
    >> "$base/logs/final_fig2_fig6.supervisor.log" 2>&1
"$python_bin" "$base/scripts/audit_campaign_completion.py" \
    >> "$base/logs/campaign_completion_audit.supervisor.log" 2>&1
"$python_bin" "$base/scripts/write_final_study_report.py" \
    >> "$base/logs/final_study_report.supervisor.log" 2>&1
