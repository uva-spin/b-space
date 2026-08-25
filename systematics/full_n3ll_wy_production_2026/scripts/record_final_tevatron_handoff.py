#!/usr/bin/env python3
"""Append the accepted isolated Tevatron W+Y run to the campaign handoff.

The writer is deliberately idempotent and only touches the new campaign
handoff.  It refuses to record a result until the final grid manifest and
the perturbative scale gate are both present and finite/positive.
"""

from __future__ import annotations

import json
from pathlib import Path


BASE = Path(__file__).resolve().parents[1]
REPORT = BASE / "reports/tevatron_n3ll_nnlo_wy_final_g1_1p017"
SCALE = BASE / "reports/tevatron_scale_variations_g1_1p017"
HANDOFF = BASE / "HANDOFF.md"
MARKER = "## Final isolated Tevatron W+Y handoff record"


def read(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"missing required artifact: {path}")
    return json.loads(path.read_text())


def main() -> None:
    final = read(REPORT / "FINAL_PRODUCTION_MANIFEST.json")
    scale = read(SCALE / "scale_variation_status.json")
    refinement = read(SCALE / "scale_variation_refinement_status.json")
    y_status = read(REPORT / "conventional_y/y_grid_status.json")
    checks = final.get("checks", {})
    if not (checks.get("row_count") and checks.get("dataset_counts")
            and checks.get("all_finite") and checks.get("all_positive")):
        raise SystemExit("final grid does not pass finite/positive row gate")
    if not (scale.get("all_finite") and scale.get("all_positive")
            and refinement.get("all_finite") and refinement.get("all_positive")):
        raise SystemExit("scale/refinement does not pass finite/positive gate")
    if not (y_status.get("row_count") == 122 and y_status.get("checks", {}).get("all_finite")
            and y_status.get("checks", {}).get("all_positive_reconstructed_full_wy")):
        raise SystemExit("conventional Y grid does not pass finite/positive gate")
    text = HANDOFF.read_text() if HANDOFF.exists() else ""
    if MARKER in text:
        print("handoff record already present")
        return
    scope = final["scope"]
    precision = final.get("source_status", {}).get("precision_refinement", {})
    lines = [
        "\n" + MARKER + "\n",
        "\nThe unattended genuine external run has passed its isolated Tevatron "
        "row and perturbative-scale gates.  This is a review artifact, not a "
        "replacement of the frozen lambda=1 package.\n",
        f"- Candidate: `{final['candidate_id']}`; formula `{final['formula']}`.\n",
        f"- Scope: {scope['row_count']} rows ({scope['dataset_row_counts']}); "
        "fixed-target and LHCb high-qT rows remain excluded pending closure.\n",
        f"- Grid MC relative uncertainty: mean "
        f"{checks['mean_relative_mc_uncertainty']:.4%}, max "
        f"{checks['max_relative_mc_uncertainty']:.4%}.\n",
        f"- Central-grid precision refinement: {precision.get('selected_count', 0)} rows at "
        f"{precision.get('calls_per_refined_component', 0):,} calls/component; "
        f"reported maximum after refinement "
        f"{precision.get('max_relative_mc_after', float('nan')):.4%}.\n",
        f"- W+Y/data ratio: median {checks['median_prediction_to_data']:.6f}, "
        f"range [{checks['min_prediction_to_data']:.6f}, "
        f"{checks['max_prediction_to_data']:.6f}].\n",
        f"- Seven-point scale envelope (3M-call pass): median half-width "
        f"{scale['scale_envelope_relative_halfwidth_median']:.4%}, max "
        f"{scale['scale_envelope_relative_halfwidth_max']:.4%}; "
        f"30M-call refinement selected {refinement['refined_count']} rows.\n",
        f"- Explicit conventional-Y grid: {y_status['row_count']} rows; "
        f"median term-reconstruction residual "
        f"{y_status['checks']['median_abs_term_reconstruction_difference_pb_per_GeV']:.3e} "
        "pb/GeV, maximum "
        f"{y_status['checks']['max_abs_term_reconstruction_difference_pb_per_GeV']:.3e} pb/GeV.\n",
        "- These are perturbative/integration diagnostics.  The final TMD "
        "publication band still requires correlated experimental/PDF and "
        "F_NP/start/model-form propagation.\n",
        f"- Final manifest: `{REPORT / 'FINAL_PRODUCTION_MANIFEST.json'}`.\n",
    ]
    HANDOFF.write_text(text.rstrip() + "\n" + "".join(lines))
    print("appended final isolated Tevatron W+Y handoff record")


if __name__ == "__main__":
    main()
