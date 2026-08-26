# Public SIDIS source inventory

Status: downloaded and profiled; not fit-ready and no rows approved.

| Record | Collaboration | Tables | Raw rows | Primary rows | pT tables | Status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| [46860](https://www.hepdata.net/record/46860) | HERMES | 4 | 103 | 103 | 0 | collinear complement; definition/overlap audit pending |
| [ins1236358](https://www.hepdata.net/record/ins1236358) | COMPASS | 48 | 19,504 | 19,504 | 46 | historical transverse candidate; generic-error/overlap audit pending |
| [ins1208547](https://www.hepdata.net/record/ins1208547) | HERMES | 64 | 1,136 | 1,136 | 16 | primary TMD candidate; row/covariance audit pending |
| [ins1624692](https://www.hepdata.net/record/ins1624692) | COMPASS | 162 | 13,992 | 4,664 | 162 | primary TMD candidate; auxiliary correction blocks excluded explicitly |
| [ins1444985](https://www.hepdata.net/record/ins1444985) | COMPASS | 4 | 6,220 | 1,244 | 0 | collinear complement; convention audit pending |
| [ins1483098](https://www.hepdata.net/record/ins1483098) | COMPASS | 2 | 3,090 | 618 | 0 | collinear complement; convention audit pending |
| [ins2840545](https://www.hepdata.net/record/ins2840545) | COMPASS | 3 | 6,314 | 1,804 | 0 | recent collinear complement; full audit pending |

Total: 287 tables and 50,359 parsed rows (29,073 primary and 21,286 auxiliary);
224 tables expose a transverse-momentum axis, with 24,088 primary transverse
candidate rows. Archive SHA256 values are in `data/hepdata_download_manifest.json`.
The profiler does not select rows or combine uncertainties into a fit
covariance. Raw archives are intentionally not bundled in this source release;
auxiliary correction-factor blocks remain explicitly marked in the local audit.

The HERMES collaboration also publishes a separate [multiplicity
database](https://hermesmults.appspot.com/) with the full multidimensional
files and statistical covariance matrices. Its pointer and download URL are
recorded in `config/public_sources.json`; that supplemental archive has not yet
been mirrored locally, so the HERMES CSV tables above remain a no-covariance
provenance candidate.
