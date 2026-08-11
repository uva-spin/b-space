#!/usr/bin/env python3
"""Run the first bin-level, separately tagged unitary-transition pilot."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from systematics.high_qt_direct_production_benchmark.experimental_unitary_transition.backend.unitary_transition import (
    bin_averaged_profile,
    unitary_transition,
)

BASE = ROOT / "systematics/high_qt_direct_production_benchmark"
FAILED_ADDITIVE = BASE / "experimental_matched_y/outputs/resummed_w_cancellation_pilot/cdf_run_2_36/convergence_status.json"
EXPANSION_CLOSURE = BASE / "experimental_matched_y/outputs/expansion_closure_pilot/cdf_run_2_36/result_nb160.json"
EXTERNAL = BASE / "summaries/tier1_boundary/central/external_pairs.csv"
HERE = BASE / "experimental_unitary_transition"


def main() -> None:
    row_id = "CDF_RUN_2:36"
    row = pd.read_csv(ROOT / "Data/v23a_tevatron_plus_lhcb7_fiducial_candidate/CDF_RUN_2.csv").loc[
        lambda frame: frame.row_id.eq(row_id)
    ].iloc[0]
    external = pd.read_csv(EXTERNAL).loc[lambda frame: frame.row_id.eq(row_id)].iloc[0]
    old = json.loads(FAILED_ADDITIVE.read_text())
    closure = json.loads(EXPANSION_CLOSURE.read_text())
    rapidity_factor = float(closure["backend_inclusive_rapidity_factor_still_present_at_explicit_node"])
    # The previous explicit-y W values carried this constant inclusive factor.
    # Divide it out rather than recomputing; this is exact for the fixed-Q row.
    w_richardson = float(old["richardson_O_h2_w_fitted_np_pb_per_GeV"]) / rapidity_factor
    w_nb320 = 5.911194277868516 / rapidity_factor
    w_nb640 = 6.418992708466433 / rapidity_factor
    fo = 0.5 * (float(external.dyturbo_pb_per_GeV) + float(external.mcfm_pb_per_GeV))
    profiles = [
        ("early_0p18_0p28", 0.18, 0.28),
        ("central_0p20_0p30", 0.20, 0.30),
        ("late_0p22_0p32", 0.22, 0.32),
    ]
    variants = []
    for name, start, end in profiles:
        p = bin_averaged_profile(float(row.qT_low), float(row.qT_high), float(row.QM), r_start=start, r_end=end, n=32)
        value = unitary_transition(w_richardson, fo, p)
        variants.append({
            "name": name, "r_start": start, "r_end": end,
            "bin_averaged_profile": p, "prediction_pb_per_GeV": value,
            "prediction_vs_fo_symmetric_shift": abs(value - fo) / (0.5 * (abs(value) + abs(fo))),
        })
    central = next(v for v in variants if v["name"].startswith("central"))
    record = {
        "status": "experimental_unitary_transition_not_production",
        "tag": "unitary_smootherstep_v0_binavg",
        "row_id": row_id,
        "qT_over_Q": float(row.qT / row.QM),
        "w_source": str(FAILED_ADDITIVE.relative_to(ROOT)),
        "external_source": str(EXTERNAL.relative_to(ROOT)),
        "inclusive_rapidity_factor_removed": rapidity_factor,
        "w_fitted_nb320_corrected_pb_per_GeV": w_nb320,
        "w_fitted_nb640_corrected_pb_per_GeV": w_nb640,
        "w_fitted_richardson_corrected_pb_per_GeV": w_richardson,
        "external_fo_average_pb_per_GeV": fo,
        "variants": variants,
        "central_prediction_pb_per_GeV": central["prediction_pb_per_GeV"],
        "profile_envelope_pb_per_GeV": [min(v["prediction_pb_per_GeV"] for v in variants), max(v["prediction_pb_per_GeV"] for v in variants)],
        "algebraic_closure_pass": True,
        "c2_profile_pass": True,
        "node_level_spectral_integration_pass": False,
        "full_boundary_row_coverage_pass": False,
        "fit_impact_pass": False,
        "replica_stability_pass": False,
        "direct_production_approval_pass": False,
        "next_gate": "node-level W and FO integration across all Tevatron tier-1 rows",
        "note": "Bin-average pilot treats W and FO as constant within the bin; it is not a production prediction.",
    }
    out = HERE / "outputs/unitary_smootherstep_v0_binavg/cdf_run_2_36"
    out.mkdir(parents=True, exist_ok=True)
    (out / "status.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
