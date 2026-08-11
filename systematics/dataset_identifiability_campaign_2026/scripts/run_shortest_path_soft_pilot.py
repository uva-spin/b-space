#!/usr/bin/env python3
"""Run an isolated strength/metric pilot for the shortest-path penalty."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


BASE = Path(__file__).resolve().parents[1]
SYSTEMATICS = BASE.parent
REPO = BASE.parents[1]
RUNNER = SYSTEMATICS / "high_qt_direct_production_benchmark/experimental_unitary_transition/scripts/run_production_fnp_stability_control.py"
SOURCE = SYSTEMATICS / "collins_factorization_validity/outputs/rowidfix_stageFT_all_qmax0p20_lam0p50_central_s303"
WGRID = REPO / "outputs/v23a_tevatron_plus_lhcb7_fidacc_merged_cache/backend_cache/wpert_v23a_tevatron_plus_lhcb7_fidacc_b160.csv"
INITIAL = SYSTEMATICS / "high_qt_direct_production_benchmark/experimental_unitary_transition/outputs/fig6_lbfgs_stationary_s303/model_state.pt"
NORMS = SYSTEMATICS / "high_qt_direct_production_benchmark/experimental_unitary_transition/outputs/fig6_lbfgs_stationary_s303/dataset_norms.csv"
AUDIT = BASE / "summaries/reference_distance_mathematics_audit"
OUTPUTS = BASE / "outputs"
TARGET = BASE / "summaries/shortest_path_soft_pilot"


def main() -> None:
    strengths = (0.3, 1.0, 3.0, 10.0, 30.0)
    records = []
    TARGET.mkdir(parents=True, exist_ok=True)
    for metric_name, metric_arg in (("directF", "F"), ("logF", "logF")):
        reference = AUDIT / f"fnp_shortest_{metric_name}.csv"
        for strength in strengths:
            label = str(strength).replace(".", "p")
            tag = f"shortest_{metric_name}_lambda{label}_pilot_s303"
            status_path = OUTPUTS / tag / "fit_status.json"
            command = [
                "/home/dustin/miniforge3/envs/pdf-fit/bin/python", str(RUNNER),
                "--seed", "303", "--source-production", str(SOURCE),
                "--w-grid", str(WGRID), "--output-root", str(OUTPUTS),
                "--initial-state", str(INITIAL), "--initial-norms", str(NORMS),
                "--max-epochs", "3000", "--min-epochs", "1500",
                "--plateau-patience", "500", "--lbfgs-max-iter", "100",
                "--float64", "--lambda-fnp-shortest-path", str(strength),
                "--fnp-shortest-path-csv", str(reference),
                "--fnp-shortest-path-metric", metric_arg,
                "--fnp-shortest-path-bmin", "0.10", "--fnp-shortest-path-bmax", "2.0",
                "--tag", tag,
            ]
            log_path = TARGET / f"{tag}.log"
            with log_path.open("w") as log:
                result = subprocess.run(command, cwd=SYSTEMATICS, stdout=log,
                                        stderr=subprocess.STDOUT, check=False)
            record = {"tag": tag, "metric": metric_name, "lambda": strength,
                      "returncode": result.returncode, "log": str(log_path)}
            if status_path.exists():
                status = json.loads(status_path.read_text())
                record.update({
                    "convergence_gate_pass": status.get("convergence_gate_pass"),
                    "epochs_run": status.get("epochs_run"),
                    "data_chi2": status.get("final", {}).get("data_chi2"),
                    "unpenalized_total_chi2": status.get("final", {}).get("unpenalized_total_chi2"),
                    "shortest_path_penalty_per_row": status.get("final", {}).get("shortest_path_penalty_per_row_objective"),
                    "max_prediction_shift_over_sigma": status.get("final", {}).get("max_prediction_shift_over_experimental_sigma"),
                })
            records.append(record)
            (TARGET / "pilot_manifest.json").write_text(json.dumps({
                "status": "isolated_shortest_path_soft_pilot_in_progress",
                "configs": records,
                "production_state_modified": False,
            }, indent=2) + "\n")
    (TARGET / "pilot_manifest.json").write_text(json.dumps({
        "status": "isolated_shortest_path_soft_pilot_complete",
        "configs": records,
        "production_state_modified": False,
    }, indent=2) + "\n")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
