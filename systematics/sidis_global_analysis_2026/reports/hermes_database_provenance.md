# HERMES supplemental database provenance

The public HERMES multiplicity database is reachable at
<https://hermesmults.appspot.com/>. Its page documents five multidimensional
binning families, a general-use recommendation of `z > 0.2`, caution above
`z > 0.8`, separate statistical covariance matrices for the unfolding, and
point-to-point treatment of the listed systematic uncertainties. It also says
the published multiplicity is formed from numerator and denominator integrated
over each bin, not from a ratio evaluated at average kinematics.

`data/hermes_database_manifest.json` records the page hash and the links for
the proton/deuteron `zpt-3D` `P_hperp` projection queried by the harvester.
The linked DESY archive is not bundled in this source release; a local retry
may populate it when reachable. Until then, the HEPData tables are not treated
as covariance-complete HERMES inputs.
