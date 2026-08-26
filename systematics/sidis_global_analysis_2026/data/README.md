# SIDIS campaign data boundary

`hepdata_download_manifest.json` is the portable record of the public
HEPData harvest. It records versioned source URLs, archive SHA256 values, and
table counts without embedding the raw submission archives in this checkout.

Raw archives remain local inputs because they are large and are not a selected
fit dataset. Re-fetch them with the scripts in `../scripts/`, verify the hashes
in the manifest, and rebuild the inventory before selecting any rows.

No row is approved until its observable definition, units, target/hadron
identity, bin integration, radiative/acceptance treatment, and covariance or
normalization provenance have been reviewed.

`hermes_database_manifest.json` records the public HERMES database page hash
and covariance/download links; it does not assert that the linked DESY archive
was downloaded.
