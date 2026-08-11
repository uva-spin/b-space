# Lambda=1 empirical-reference production update

This package records the active 2026-08-11 production update for the
identifiability campaign. It supersedes the 24-start lambda=1 package as the
current campaign production reference; the earlier package remains the
rollback reference in the source campaign.

## Definition

- Objective: pointwise distance to the empirical median `F_NP`
- Reference region: `x=0.1`, `0.1 <= bT <= 2.0 GeV^-1`
- Strength: `lambda=1`
- Starts: 96 stationary starts
- Experimental residual replicas: 50 conditional replicas
- Crossed ensemble: 4,800 members per flavor

All 48 newly added starts passed the declared FNP stationarity gate. The
start-only q16--q84 widths are 22.802% in b-space, 20.799% for `u` in
k-space, and 21.972% for `d` in k-space. The combined k-space full widths are
21.257% (`u`) and 22.480% (`d`) in the active region.

The b-space and k-space CSV files contain the central curve and operational
q16/q84 ensemble bands. The PDF and PNG files are updated-only `u,d` figures
at `x=0.1`, `Q=10 GeV`; the retained `_1sigma` filename is a compatibility
name, not a calibrated one-sigma claim.

## Interpretation and limitations

The combined intervals are empirical q16--q84 bands and have no calibrated
68% confidence-level interpretation. The empirical reference median is not
reciprocal-cross-fitted, and 96-start completeness is established only for
the declared perturbation family and objective. Endpoint resampling remains
at the documented 7--10% level.

`PRODUCTION_MANIFEST.json` and `PRODUCTION_AUDIT.json` are the authoritative
status and integrity records. `combined_summary.json` and
`start_only_summary.json` preserve the source diagnostics; their diagnostic
status labels do not override the active production status in the manifest.
