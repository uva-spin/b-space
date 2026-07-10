# v23a fixed-target DY experimental-replica fits, 50 replicas

This freeze records the underlying 50 experimental pseudo-data replica fits used to build the fixed-target DY TMD ensemble.

## Scope

- Datasets: E288_200, E288_300, E288_400, E605, E772
- Corrected row: E288_300:99
- Normalization priors: explicit 15% scenario
- Point-to-point sensitivity: 5% floor/sensitivity for E772 and E288_400
- Anchor: lambda_logF = 3
- Seeds: 1001--1050
- PDF during fitting: NNPDF40_nnlo_as_01180 member 0

## Key status

- n replicas: 50
- chi2 q95: 2.1310145686109365
- chi2 max: 2.1866746953618357
- norm-pull q95: 2.707433342933655
- norm-pull max: 3.017364978790283
- fit pass: True
- norm pass: True
- random-split width q90: 0.6439466887345098
- random-split center q90: 0.021749180926595522
- random-split pass: True

## Interpretation

This is the fixed-PDF experimental-replica fit ensemble.  It is the fitted F_NP source for the later experimental+PDF overlay TMD reconstruction.
