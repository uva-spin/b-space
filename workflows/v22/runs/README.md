# v22 fit runs

Central-refit, cache-smoketest, warm-check, and anchor-scan drivers.  They are
root-relative shell workflows: launch as
`./workflows/v22/runs/<script>.sh` from the checkout root and set `TRAIN`,
`BACKEND`, `DATA_DIR`, `FROZEN`, and `OUT` explicitly when using external
artifacts.
