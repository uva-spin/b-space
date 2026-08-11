#!/usr/bin/env python3
"""Reconstruct frozen-FNP W values from saved boundary kernels."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.integrate import simpson
from scipy.special import j0
import torch


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from systematics.high_qt_direct_production_benchmark.experimental_matched_y.scripts.run_resummed_w_cancellation_pilot import (
    load_np_factor,
)


BASE = ROOT / "systematics/high_qt_direct_production_benchmark/experimental_unitary_transition"
CACHE = BASE / "outputs/unitary_smootherstep_v1_differentiable_kernels_nb640_nqt2_ny2"
REFERENCE = BASE / "outputs/unitary_smootherstep_v1_nodes_nb640_nqt2_ny2_simpson/tevatron_rows.csv"
METRICS = ROOT / "systematics/collins_factorization_validity/outputs/rowidfix_stageFT_all_qmax0p20_lam0p50_central_s303/metrics.json"


def safe_tag(row_id: str) -> str:
    return row_id.lower().replace(":", "_")


def main() -> None:
    config = json.loads(METRICS.read_text())["config"]
    device = torch.device("cpu")
    dtype = torch.float32
    model = load_np_factor(config, device=device, dtype=dtype)
    reference = pd.read_csv(REFERENCE).set_index("row_id")
    rows = []
    with torch.no_grad():
        for row_id, expected in reference.w_fitted_pb_per_GeV.items():
            with np.load(CACHE / "rows" / f"{safe_tag(row_id)}.npz") as saved:
                b_grid = saved["b_grid"]
                kernels = saved["kernels"]
                qt_nodes = saved["qt_nodes"]
                qt_weights = saved["qt_weights"]
                y_weights = saved["y_weights"]
                x1 = saved["x1_nodes"]
                x2 = saved["x2_nodes"]
                width = float(saved["qT_high"] - saved["qT_low"])
                b = torch.as_tensor(b_grid, dtype=dtype, device=device)
                total = 0.0
                for iq, qt in enumerate(qt_nodes):
                    for iy in range(len(y_weights)):
                        x = torch.tensor([x1[iq, iy], x2[iq, iy]], dtype=dtype, device=device)
                        factors = model(x, b)
                        pair = (factors[0] * factors[1]).cpu().numpy().astype(float)
                        value = simpson(b_grid * j0(float(qt) * b_grid) * kernels[iq, iy] * pair, x=b_grid)
                        total += float(qt_weights[iq] * y_weights[iy]) * float(value)
                reconstructed = total / width
            rows.append({
                "row_id": row_id,
                "reference_w_pb_per_GeV": float(expected),
                "reconstructed_w_pb_per_GeV": reconstructed,
                "absolute_difference_pb_per_GeV": reconstructed - float(expected),
                "relative_difference": reconstructed / float(expected) - 1.0,
            })
    audit = pd.DataFrame(rows)
    max_relative = float(audit.relative_difference.abs().max())
    status_path = CACHE / "campaign_status.json"
    status = json.loads(status_path.read_text())
    status.update({
        "validation_pass": bool(max_relative < 1.0e-6),
        "validation_row_count": int(len(audit)),
        "validation_max_absolute_relative_difference": max_relative,
        "full_fnp_refit_authorized": bool(max_relative < 1.0e-6),
        "next_gate": (
            "run a separately tagged differentiable central FNP refit with both correlated theory nuisances"
            if max_relative < 1.0e-6 else
            "repair kernel reconstruction before any FNP refit"
        ),
    })
    audit.to_csv(CACHE / "frozen_fnp_reconstruction.csv", index=False)
    status_path.write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
