# Initial joint DY+SIDIS trial register (2026-08-26)

This is an isolated validation record, not a production result.  The DY anchor is the frozen lambda=1 W-only 329-row solution; SIDIS is the 746-row identified COMPASS 2026 addendum collinear scope.

| trial | FF | mode | DY chi2/row | SIDIS chi2/row | SIDIS rows | excluded | classification |
|---|---|---:|---:|---:|---:|---:|---|
| nnff10_binavg_3000 | nnff10_nnlo | bin_average | 0.3944 | 21.9428 | 745 | 1 | external-FF primary diagnostic |
| nnff10_binavg_2500 | nnff10_nnlo | bin_average | 0.3943 | 17.1268 | 745 | 1 | external-FF primary diagnostic |
| haps_binavg_1500_circular | haps_nnlo | bin_average | 0.3943 | 2.9411 | 746 | 0 | circular external comparison: HAPS FFs were fitted using modern SIDIS data |
| apfel_nlo_full_den_diagnostic | nnff10_nnlo | midpoint | 0.3943 | 12.9775 | 738 | 8 | APFEL NLO numerator / LO denominator diagnostic |
| nnff10_midpoint_legacy | None | None | 0.3945 | 124.4188 | 745 | 1 | legacy optimizer-control diagnostic |

## Interpretation

- The reinitialized joint optimizer is materially better behaved than the legacy all-scales-near-one pilot, but central NNFF10 still gives a poor scalar SIDIS closure (17.13 chi2/row in the 2500-epoch run).
- The HAPS comparison reaches 2.94 chi2/row, but HAPS is not independent because its FFs incorporate modern COMPASS SIDIS information; it is a diagnostic, not evidence that the observable implementation is closed.
- Across all 101 NNFF10 members, the best raw member can make many rows non-positive and is invalid; the best member with all 746 rows positive remains a poor closure candidate. FF replicas alone do not resolve the mismatch.
- The APFEL SIDIS-NLO numerator with a full massless NLO inclusive-DIS denominator gives 12.98 chi2/row on 738 positive rows (the earlier LO-denominator diagnostic gave 12.75); eight rows remain non-positive and bin-integrated normalization is still unvalidated.
- DY non-regression is demonstrated in every pilot (about 0.394 chi2/row), while no pilot is authorized for production or for TMDFF uncertainty propagation.

The current fits establish an actual joint DY+SIDIS software path and expose a scalar observable-closure problem.  They do not justify a global/TMD production claim: the COMPASS addendum has no transverse axis, the public table lacks a full covariance, HERMES identity is unresolved, and the proper perturbative SIDIS coefficient/denominator interface remains to be independently validated.
