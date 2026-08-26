# Public arXiv SIDIS table inventory

This is a source parser/inventory only. No arXiv rows are approved for a fit.

## CLAS 0809.1153 ancillary tables

| File | Numeric rows | Observable axes | Malformed numeric rows |
| --- | ---: | --- | ---: |
| `pip_sidis_cs_table_zh_pt2.dat` | 112566 | Q2, x, z, pT2, phi | 0 |
| `pip_sidis_cs_table_xf_pt2.dat` | 170352 | Q2, x, xF, pT2, phi | 0 |
| `pip_sidis_cs_table_zh_t.dat` | 106411 | Q2, x, z, |t|, phi | 0 |
| `pip_sidis_cs_table_zg_v.dat` | 279470 | Q2, x, zG, v, phi | 0 |

The CLAS files contain absolute five-fold cross sections with statistical and systematic columns and a radiative-correction factor. They are not multiplicities; conversion requires a fixed SIDIS cross-section convention, bin integration, acceptance/radiative treatment, and covariance model.

## Hall C E00-108 and Hall A E06-010 TeX tables

Hall C `tab:xsect-vrs-pt2`: 25 physical pT² rows, 200 target/charge entries in the two rho-subtraction columns.
Hall A: 160 physical table rows and 320 pi+/pi- entries across the four labelled tables.

Hall C remains a stage-2 low-energy absolute-cross-section candidate pending rho-subtraction and covariance closure. Hall A is a 3He nuclear diagnostic until a nuclear impulse-approximation/dilution interface is explicitly validated. The TeX rows are counted for provenance and are not silently converted.
