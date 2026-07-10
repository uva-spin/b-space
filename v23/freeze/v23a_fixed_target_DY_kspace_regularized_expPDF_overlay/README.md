# v23a fixed-target DY regularized kT-space companion

This freeze records the regularized finite-b_T Hankel-transform k_T-space companion to the fixed-target b-space TMD result.

## Status

PASS as a regularized k_T-space companion over:

- 0 <= kT <= 4 GeV

Important audited values:

- default tail mode: expb2
- alternate tail modes compared: expb, taper
- b transform max: 24 GeV^-1
- n b transform points: 6001
- n kT points: 401
- all transformed values finite: True
- regularization-mode p90 max relative difference: 0.030912545977260095
- regularization-mode max relative difference: 0.04304417062181683
- KT regularization stability pass: True
- expb2 min_over_peak_min: -0.01718554577009812
- expb2 negative_area_fraction_max: 0.13422350979037503
- expb2 relative_68_halfwidth_p90_max: 0.7597840832064348

## Definition

The transform convention is:

f(kT) = 1/(2*pi) int dbT bT J0(kT bT) ftilde(bT)

The default frozen regularization uses expb2 large-b continuation plus a smooth endpoint taper.

## Caveats

- This is a regularized finite-b_T Hankel transform, not a high-kT perturbative-tail prediction.
- Small negative lobes are retained and treated as transform diagnostics, not clipped.
- The uncertainty is experimental+PDF overlay, not PDF-through-refit.
- Scale/profile/nuclear/model-form uncertainties are not included.
- The primary production object remains the b-space TMD; this is the kT-space companion.
