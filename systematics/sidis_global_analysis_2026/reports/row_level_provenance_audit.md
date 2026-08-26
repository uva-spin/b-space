# SIDIS row/table provenance audit

Status: the source-only audit records likely observable and axis/value roles;
no rows are approved and no covariance is inferred.

| Record | TMD-candidate tables | TMD-candidate rows | Multiplicity tables | Asymmetry tables | Unresolved tables |
| --- | ---: | ---: | ---: | ---: | ---: |
| 46860 | 0 | 0 | 4 | 0 | 0 |
| ins1236358 | 46 | 19,504 | 46 | 0 | 2 |
| ins1208547 | 16 | 288 | 56 | 8 | 0 |
| ins1444985 | 0 | 0 | 4 | 0 | 0 |
| ins1483098 | 0 | 0 | 2 | 0 | 0 |
| ins1624692 | 162 | 14,316 | 162 | 0 | 0 |
| ins2840545 | 0 | 0 | 3 | 0 | 0 |

The 224 candidate tables are transverse-momentum multiplicity projections;
COMPASS contributes 19,504 candidate rows from `ins1236358` and 14,316 from
`ins1624692`; HERMES contributes 288 candidate rows. Tables
with multiple possible central/low/high axis columns require a published
mapping review before conversion. The CSV submissions do not by themselves
close correlated covariance or normalization treatment. The complete audit is
generated locally as `reports/row_level_provenance_audit.json` after the raw
archives are fetched.
