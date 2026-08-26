# SIDIS campaign data boundary

`hepdata_download_manifest.json` is the portable, reviewable record of the
public HEPData harvest. It records the source URLs, version, archive SHA256,
table paths, and table counts without embedding the raw submission archives in
the source release.

The local `raw/hepdata/` tree contains the downloaded version-1 CSV archives
and extracted tables used by the profiler. It is an input cache, not a selected
fit dataset. Do not edit these files by hand. Re-fetch with
`../scripts/fetch_hepdata_records.py`, verify with the recorded hashes, and
rebuild the inventory with `../scripts/profile_hepdata_tables.py` and
`../scripts/build_source_inventory.py`.

No rows are approved until their observable definition, units, target/hadron
identity, bin integration, radiative/acceptance treatment, and covariance or
normalization provenance have been reviewed.

`hermes_database_manifest.json` is a separate provenance record for the
public HERMES database page and its covariance/download links. It does not
assert that those linked archives were downloaded.

`global_source_download_manifest.json` records the expanded public candidate
harvest. It includes HEPData archives for historical EMC/E665/H1/ZEUS records
and arXiv source packages for JLab CLAS and Hall-C/Hall-A tables. The global
raw cache is intentionally separate from `raw/hepdata/`, so the original
seven-record baseline remains reproducible byte-for-byte. Use
`../scripts/fetch_global_sources.py` to refresh it and
`../scripts/build_global_source_inventory.py` to profile it. No raw source,
table, or ancillary data file is itself an approved fit input.

For the CLAS ancillary package, `../scripts/inventory_arxiv_tables.py` checks
headers, numeric-row counts, finite values, and ranges without rewriting the
source. `../scripts/convert_clas_ancillary.py` can produce a lossless,
metadata-preserving CSV conversion under `derived/global/`; that directory is
ignored because it is generated output. The conversion manifest records the
source hash context and explicitly keeps `approved_rows` at zero.
