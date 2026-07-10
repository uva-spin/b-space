# v23a fixed-target DY b-space TMD ensemble, experimental+PDF overlay

This freeze records the primary b-space TMD result for the fixed-target DY extraction.

## Scope

- Primary production object: b-space TMDPDF ensemble
- Datasets: E288_200, E288_300, E288_400, E605, E772
- Perturbative backend: N^3LL W-term backend with NLO hard/OPE matching in W
- F_NP source: 50 accepted experimental pseudo-data replica fits
- PDF uncertainty: PDF-member overlay in the TMD reconstruction
- Output space: b_T

## Key files expected in the artifact/release

- v23a_dataPDF_tmd_replica_bspace_long.csv
- v23a_dataPDF_tmd_replica_bspace_bands.csv
- v23a_dataPDF_relative_band_summary.csv
- v23a_dataPDF_tmd_manifest.json
- F_NP_dataPDF_bands.pdf
- ftilde_dataPDF_bands.pdf
- b_ftilde_dataPDF_bands.pdf
- b_x_ftilde_dataPDF_bands.pdf

## Interpretation

The 68% band combines experimental data-replica variation in the fitted F_NP with PDF-member variation in the perturbative/OPE TMD reconstruction.  It is a PDF overlay, not PDF-through-refit.
