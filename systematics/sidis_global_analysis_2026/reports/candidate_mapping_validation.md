# Candidate SIDIS mapping validation

Status: explicit source mappings were validated against the local version-1
HEPData archives; no rows are approved and no covariance is inferred. The
public checkout carries the validator and can regenerate the machine-readable
report after the raw archives are downloaded.

| Record | Tables validated | Published candidate rows | Canonical observations | Boundary |
| --- | ---: | ---: | ---: | --- |
| HERMES `ins1208547` | 16/16 | 288 | 288 | target-blocked pT projection; integrated over unreported x and Q2 |
| COMPASS `ins1236358` | 46/46 | 19,136 | 18,624 | generic asymmetric total error; explicit dash placeholders skipped |
| COMPASS `ins1624692` | 162/162 | 13,992 | 4,664 | stat/sys pT² measurements; 9,328 auxiliary correction rows excluded |

The conversion uses `sidis_dataset.canonicalize_table` with caller-declared
value, axis, bin-edge, error, target/block, and placeholder mappings. It does
not select a fit scope or combine uncertainties. Run
`scripts/validate_candidate_mappings.py` after fetching the raw archives to
rebuild the full JSON provenance record.
