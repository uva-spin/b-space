# v23a_lambda3_normpriors15_p2p5_E772_E288400_50rep_DYonly_bspace_sensitivity

Frozen v23a fixed-target low-Q DY-only **b-space** TMD sensitivity ensemble.

## Status

PASS as a DY-only b-space sensitivity ensemble.

Important audited values:

- n replicas from runs: 50
- n replicas in b-space band table: 50
- n random splits: 500
- chi2 median: 1.858603165144339
- chi2 q95: 2.1310145686109365
- chi2 max: 2.1866746953618357
- norm-pull q95: 2.707433342933655
- norm-pull max: 3.017364978790283
- fit pass: True
- norm pass: True
- b-space band pass: True
- random split width q90: 0.6439466887345098
- random split center q90: 0.021749180926595522
- random split pass: True
- q95 ensemble pass: True
- split-half distribution pass: True
- max relative 68% halfwidth active: 0.033509424782098216
- max central-vs-replica-median rel p90 active: 0.012321568672877726

## Scope

This artifact is for fixed-target low-Q DY-only b-space TMDPDFs using:

- E288_200
- E288_300
- E288_400
- E605
- E772

It includes:

- corrected E288_300:99
- explicit 15% normalization priors
- 5% point-to-point systematic sensitivity on E772 and E288_400
- lambda_logF_anchor = 3
- 50 replicas, seeds 1001–1050
- v22 full perturbative backend
- b-space TMD bands only

## Formal exact-x grid

- x = 0.10, 0.20, 0.30, 0.50

## Important caveats

- This is a sensitivity ensemble, not a final experimental covariance treatment.
- The 5% P2P floor for E772 and E288_400 should be replaced by documented covariance/P2P information if available.
- The fitted nonperturbative factor is flavor independent in this model.
- Full TMDPDF flavor dependence enters through the collinear PDFs and OPE.
- Cross-section bands include fitted random normalization draws.
- kT-space transforms remain diagnostic only; this freeze is b-space only.
- The NLO Y/finite-tail path remains a development path until more external MCFM/DYTurbo closure is completed.
