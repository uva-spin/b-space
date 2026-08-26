# SIDIS covariance readiness

The public-source harvest records uncertainty columns and covariance claims,
but does not infer a likelihood covariance. The HERMES database advertises
statistical covariance matrices for unfolding; its DESY archive was
unavailable during this harvest, so those matrices remain pending. HEPData
CSV submissions for COMPASS expose asymmetric statistical/systematic columns
(or a generic total error for COMPASS 2013), not a complete correlated model.

`sidis_data.read_covariance_matrix` reads a supplied plain-text or gzip matrix
and rejects malformed, nonsymmetric, or nonfinite input. `sidis_covariance`
evaluates a positive-definite matrix by Cholesky whitening. Neither module
combines stat/sys errors, invents correlations, or converts normalization into
a nuisance parameter. See `config/covariance_manifest.json`.

Before any row is approved, record the matrix (or a published justification
for a diagonal approximation), row ordering, normalization correlations, and
the asymmetric-error likelihood convention. Until then rows remain discovery
candidates.
