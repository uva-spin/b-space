#!/usr/bin/env python3
"""Document why the global production objective and LHCb diagnostic differ."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
WORK = ROOT / "systematics/finite_y_completion_2026"
OUT = WORK / "reports/metric_comparability_audit.json"


def main() -> None:
    production_path = ROOT / "systematics/high_qt_direct_production_benchmark/experimental_unitary_transition/outputs/fig6_lbfgs_stationary_s311/fit_status.json"
    production = json.loads(production_path.read_text())
    final = production["final"]
    replica = json.loads((WORK / "reports/lambda1_lhcb_unitary_nnlo/lhcb_unitary_nnlo_replica_summary.json").read_text())
    fit = json.loads((WORK / "reports/lambda1_lhcb_unitary_nnlo/lhcb_unitary_nnlo_fit_impact_summary.json").read_text())
    central = fit["profiles"]["central_0p20_0p30"]
    replica_central = replica["profile_summary"]["central_0p20_0p30"]
    high_qt_total = 4.0 * replica_central["total_chi2_per_row_q16_median_q84"][1]
    audit = {
        "status": "metric_comparability_audit_complete",
        "production_reference": {
            "source": str(production_path),
            "objective_definition": production["objective"],
            "rows": production["row_count"],
            "total_chi2": final["total_chi2"],
            "data_chi2": final["data_chi2"],
            "normalization_penalty": final["norm_penalty"],
            "objective_per_row": final["objective_per_row"],
            "qT_scope": "production training is restricted to qT/Q <= 0.20 and includes only six low-qT LHCb rows",
            "backend_scope": "W-only/internal CSS pilot with y_mode=zero; not a finite-Y high-qT fit",
        },
        "lhcb_nnlo_diagnostic": {
            "rows": 4,
            "qT_over_Q_scope": [0.2416666666666666, 1.85],
            "objective_definition": "published correlated pT covariance chi2 plus matching and NNLO-scale nuisance penalties",
            "central_profile_data_chi2_per_row_q16_median_q84": central["data_chi2_per_row_q16_median_q84"],
            "central_profile_total_chi2_per_row_q16_median_q84": central["total_chi2_per_row_q16_median_q84"],
            "50_replica_total_chi2_per_row_q16_median_q84": replica_central["total_chi2_per_row_q16_median_q84"],
            "unprofiled_median_central_profile_covariance_chi2_per_row": 21.122098818530137,
        },
        "why_the_numbers_differ": [
            "The denominators are 329 global rows versus four LHCb boundary rows.",
            "The production objective is a low-qT W-only fit and does not include the four high-qT finite-Y boundary bins.",
            "The LHCb diagnostic holds the lambda=1 W endpoints fixed and profiles only matching and NNLO-scale nuisances; it is not a global refit.",
            "The LHCb diagnostic uses the published correlated covariance, while the historical production objective uses the production point-error/data-normalization objective.",
        ],
        "illustrative_global_average_only": {
            "formula": "(production total chi2 + 4 * LHCb median total chi2/row) / 333",
            "value": (final["total_chi2"] + high_qt_total) / (production["row_count"] + 4),
            "warning": "This is not a refit and is shown only to demonstrate that a four-bin local tension need not equal a global chi2/row of 7.16.",
        },
        "interpretation": "The 7.16 value is not evidence that the entire production fit has chi2/row=7.16. It is a local LHCb high-qT closure diagnostic. It is nevertheless a genuine local discrepancy because those four bins are several experimental standard deviations below the NNLO unitary prediction before nuisance profiling.",
        "production_outputs_modified": False,
    }
    OUT.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
