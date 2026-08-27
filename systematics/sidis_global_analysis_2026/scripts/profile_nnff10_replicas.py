#!/usr/bin/env python3
"""Profile NNFF10 replica variations against the provisional COMPASS scope.

This is a theory-uncertainty closure diagnostic, not a fit promotion.  Every
NNFF10 pion/kaon member is evaluated with the same NNPDF PDF, target
composition, observable convention, and four explicitly constrained
hadron-charge normalizations used by the initial joint pilot.  The nuisance
scales are profiled analytically in one dimension (with a log-normal 10%
prior) so that the result tests whether the published FF uncertainty can
cover the SIDIS residual without introducing an empirical z-dependent
correction.

The default midpoint mode is intentionally fast and is followed by an
optional bin-average mode.  Signed FF values are retained; rows with a
non-positive prediction are excluded and recorded rather than clamped.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lhapdf
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from sidis_global_analysis_2026.sidis_ff import LHAPDFFMember
from sidis_global_analysis_2026.scripts.run_initial_joint_dy_sidis_fit import (
    FF_FAMILIES,
    compute_collinear_ratio,
)


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/derived/compass_collinear_provisional/compass_collinear_provisional.csv"
OUT = ROOT / "outputs/nnff10_replica_profile_diagnostic"
REPORT = ROOT / "reports/nnff10_replica_profile.json"


def profile_scale(ratio: np.ndarray, target: np.ndarray, sigma: np.ndarray, prior: float) -> tuple[float, float]:
    """Return the profiled positive scale and its penalized chi2."""

    def objective(log_scale: float) -> float:
        scale = float(np.exp(log_scale))
        return float(np.sum(((scale * ratio - target) / sigma) ** 2) + (log_scale / prior) ** 2)

    # This is a profiling diagnostic, not a production normalization prior:
    # allow the objective to reveal if a replica would require a scale well
    # outside the declared 10% prior.  The prior penalty remains in the
    # objective, so such a member is correctly disfavoured rather than
    # silently clipped at an arbitrary boundary.
    result = minimize_scalar(objective, bounds=(-3.0, 3.0), method="bounded", options={"xatol": 1.0e-11})
    return float(np.exp(result.x)), float(result.fun)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinematic-mode", choices=("midpoint", "bin_average"), default="midpoint")
    ap.add_argument("--quadrature-order", type=int, default=4)
    ap.add_argument("--members", type=int, default=101)
    ap.add_argument("--prior", type=float, default=0.10)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--report", type=Path, default=REPORT)
    args = ap.parse_args()
    if args.members < 1 or args.members > 101:
        raise ValueError("NNFF10 has members 0..100")
    if args.prior <= 0.0:
        raise ValueError("normalization prior must be positive")

    frame = pd.read_csv(DATA)
    pdf = lhapdf.mkPDF("NNPDF40_nnlo_as_01180", 0)
    keys = [("pi", "+"), ("pi", "-"), ("K", "+"), ("K", "-")]
    labels = ["pi+", "pi-", "K+", "K-"]
    set_names = FF_FAMILIES["nnff10_nnlo"]
    target = frame.multiplicity.to_numpy(float)
    sigma = np.maximum(frame.sigma_uncorrelated.to_numpy(float), 1.0e-12)
    group = np.asarray([f"{h}{c}" for h, c in zip(frame.hadron, frame.charge)])

    rows: list[dict[str, object]] = []
    replica_predictions: list[pd.DataFrame] = []
    for member in range(args.members):
        ff_members = {key: LHAPDFFMember(name, member) for key, name in set_names.items()}
        ratio, diagnostics = compute_collinear_ratio(
            frame, pdf, ff_members, mode=args.kinematic_mode,
            quadrature_order=args.quadrature_order,
        )
        valid = np.isfinite(ratio) & (ratio > 0.0)
        chi2 = 0.0
        scales: dict[str, float] = {}
        for key, label in zip(keys, labels):
            mask = valid & (group == label)
            if not np.any(mask):
                scales[label] = float("nan")
                continue
            scale, component = profile_scale(ratio[mask], target[mask], sigma[mask], args.prior)
            scales[label] = scale
            chi2 += component
        rows.append({
            "member": member,
            "valid_rows": int(valid.sum()),
            "excluded_rows": int((~valid).sum()),
            "chi2_profiled": chi2,
            "chi2_per_valid_row": chi2 / int(valid.sum()) if np.any(valid) else float("nan"),
            **{f"norm_{label}": scales[label] for label in labels},
        })
        if member in {0}:
            pred = frame[["row_id", "hadron", "charge", "x", "y", "z", "multiplicity", "sigma_uncorrelated"]].copy()
            pred["member"] = member
            pred["ff_collinear_ratio"] = ratio
            pred["valid"] = valid
            pred["profiled_norm"] = [scales.get(x, np.nan) for x in group]
            pred["prediction"] = pred["profiled_norm"] * pred["ff_collinear_ratio"]
            pred["pull"] = (pred["prediction"] - pred["multiplicity"]) / pred["sigma_uncorrelated"]
            replica_predictions.append(pred)
        if diagnostics and member == 0:
            rows[-1]["central_diagnostics"] = diagnostics

    results = pd.DataFrame(rows)
    results = results.sort_values("chi2_profiled").reset_index(drop=True)
    args.out.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.out / "replica_profile.csv", index=False)
    if replica_predictions:
        pd.concat(replica_predictions, ignore_index=True).to_csv(args.out / "central_member_predictions.csv", index=False)
    best = results.iloc[0]
    central = results.loc[results.member == 0].iloc[0]
    # A replica that obtains a low objective only by making many rows
    # non-positive is not a valid closure candidate.  Keep both rankings so
    # the diagnostic cannot accidentally promote that pathological member.
    fully_valid = results.loc[results.valid_rows == len(frame)]
    best_fully_valid = fully_valid.iloc[0] if len(fully_valid) else None
    report = {
        "status": "nnff10_external_replica_closure_diagnostic_complete_not_fit",
        "scope": str(DATA),
        "model": "NNPDF40_nnlo_as_01180 + NNFF10 NNLO central/MC members; massless LO multiplicity ratio",
        "kinematics": {"mode": args.kinematic_mode, "quadrature_order": int(args.quadrature_order)},
        "members_requested": int(args.members),
        "members_with_any_positive_prediction": int(np.sum(results.valid_rows > 0)),
        "normalization_prior": {"parameter": "log hadron-charge scale", "width": float(args.prior)},
        "central_member": {
            "member": 0,
            "chi2_profiled": float(central.chi2_profiled),
            "chi2_per_valid_row": float(central.chi2_per_valid_row),
            "valid_rows": int(central.valid_rows),
            "excluded_rows": int(central.excluded_rows),
        },
        "best_member": {
            "member": int(best.member),
            "chi2_profiled": float(best.chi2_profiled),
            "chi2_per_valid_row": float(best.chi2_per_valid_row),
            "valid_rows": int(best.valid_rows),
            "excluded_rows": int(best.excluded_rows),
        },
        "best_member_all_rows_valid": None if best_fully_valid is None else {
            "member": int(best_fully_valid.member),
            "chi2_profiled": float(best_fully_valid.chi2_profiled),
            "chi2_per_row": float(best_fully_valid.chi2_per_valid_row),
            "valid_rows": int(best_fully_valid.valid_rows),
            "excluded_rows": int(best_fully_valid.excluded_rows),
        },
        "quantiles_chi2_per_row": {
            str(q): float(np.quantile(results.chi2_per_valid_row.to_numpy(float), q))
            for q in (0.05, 0.50, 0.95)
        },
        "interpretation": "FF replica propagation is a model/theory variation; it is not a replacement for the missing perturbative SIDIS coefficient-function and denominator implementation.",
        "production_files_modified": False,
        "promotion_authorized": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
