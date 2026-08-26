# Public SIDIS source-search boundary

Search date: 2026-08-26. The campaign first searched the HEPData HERMES and
COMPASS MULT collections and then expanded the search to published HEPData
records and public arXiv sources for the historical global unpolarized SIDIS
families. The initial seven-record registry was therefore only the clean
HERMES/COMPASS starting subset. The expanded 22-entry candidate registry is in
`config/global_sources.json`; its harvested/profiled state is in
`reports/global_source_inventory.{json,md}`.

The HERMES nuclear-hadronization collection (HEPData record 13387) is not
silently merged with free-proton/deuteron data: its nuclear ratios would need a
separate nuclear-hadronization model. Spin-dependent SIDIS asymmetry records,
including COMPASS Sivers examples and JLab Hall-A polarized measurements, are
also outside the present unpolarized multiplicity scope. They remain valid
future extensions, not discarded measurements.

The expanded inventory includes JLab CLAS 0809.1153, Hall-C E00-108
1103.1649, and Hall-A E06-010 1610.02350. CLAS ancillary files and the Hall-C
and Hall-A TeX source tables are public harvest candidates, but the former
requires an absolute-cross-section/radiative closure and the latter has
low-energy or nuclear-specific conventions. The JLab Hall-A 3He data are
therefore a nuclear diagnostic, not a free-nucleon input. E665 and EMC
historical HEPData records, and H1/ZEUS current-region records, are tracked as
stage-2 or diagnostic candidates because their axes and cuts are not identical
to the standard (P_{hT})-differential multiplicity.

The source boundary is deliberately conservative. A new observable class or
nuclear target requires a new scope entry, formalism and covariance audit, and
closure tests before it can enter a joint DY+SIDIS fit. A source may remain in
the global registry while `approved_rows` stays zero.
