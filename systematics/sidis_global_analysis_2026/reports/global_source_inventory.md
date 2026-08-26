# Global unpolarized SIDIS candidate inventory

Status: broad public-source registry and harvest; no rows approved for a fit.

The registry deliberately separates the data universe from the staged fit scope. Stage 1 is the clean proton/deuteron multiplicity core; stage 2 adds older, identified, or absolute-cross-section data only after observable and covariance closure. Nuclear, current-region, jet/remnant, and source-only records remain diagnostics or deferred inputs.

Registry records: **22**; HEPData records: **16**; public arXiv source records: **4**; pointer/deferred records: **3**.
Harvested HEPData profile: **582** tables and **57358** rows (primary **35963**, auxiliary **21395**).

| ID | Collaboration | Stage | Role | Observable | Target | Readiness | Harvest/profile |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [hepdata:46860](https://www.hepdata.net/record/46860) | HERMES | stage_1 | collinear_complement | collinear_multiplicity | hydrogen | public_versioned_hepdata_csv | 4 tables/103 rows |
| [hepdata:ins1236358](https://www.hepdata.net/record/ins1236358) | COMPASS | stage_1 | primary_candidate | charged_hadron_multiplicity_pt2 | 6LiD isoscalar | public_versioned_hepdata_csv | 48 tables/19504 rows |
| [hepdata:ins1208547](https://www.hepdata.net/record/ins1208547) | HERMES | stage_1 | primary_candidate | identified_multiplicity_pt | hydrogen and deuterium | public_versioned_hepdata_csv; supplemental_covariance_pointer | 64 tables/1136 rows |
| [hepdata:ins1624692](https://www.hepdata.net/record/ins1624692) | COMPASS | stage_1 | primary_candidate | charged_hadron_multiplicity_pt2 | 6LiD isoscalar deuteron | public_versioned_hepdata_csv | 162 tables/13992 rows |
| [hepdata:ins1444985](https://www.hepdata.net/record/ins1444985) | COMPASS | stage_1 | collinear_complement | collinear_multiplicity | 6LiD isoscalar | public_versioned_hepdata_csv | 4 tables/6220 rows |
| [hepdata:ins1483098](https://www.hepdata.net/record/ins1483098) | COMPASS | stage_2 | collinear_complement | collinear_multiplicity | 6LiD isoscalar | public_versioned_hepdata_csv | 2 tables/3090 rows |
| [hepdata:ins2840545](https://www.hepdata.net/record/ins2840545) | COMPASS | stage_2 | recent_collinear_complement | collinear_multiplicity | liquid hydrogen | public_versioned_hepdata_csv | 3 tables/6314 rows |
| [hepdata:29288](https://www.hepdata.net/record/29288) | E665 | stage_2 | low_energy_cross_check | charged_hadron_multiplicity_pt_diagnostic | deuterium | public_versioned_hepdata_csv | 5 tables/39 rows |
| [hepdata:37889](https://www.hepdata.net/record/37889) | E665 | stage_2 | low_energy_cross_check | forward_hadron_xf_pt2 | hydrogen | public_versioned_hepdata_csv | 35 tables/2103 rows |
| [hepdata:42505](https://www.hepdata.net/record/42505) | E665 | diagnostic | nuclear_hadronization_diagnostic | multiplicity_and_rapidity_nuclear | deuterium and xenon | public_versioned_hepdata_csv | 8 tables/391 rows |
| [hepdata:42540](https://www.hepdata.net/record/42540) | E665 | diagnostic | nuclear_hadronization_diagnostic | z_multiplicity_nuclear | deuterium and xenon | public_versioned_hepdata_csv | 55 tables/802 rows |
| [hepdata:1432](https://www.hepdata.net/record/1432) | EMC | stage_2 | historical_cross_check | forward_hadron_z_pt2 | hydrogen and deuterium | public_versioned_hepdata_csv | 129 tables/3098 rows |
| [hepdata:30476](https://www.hepdata.net/record/30476) | EMC | diagnostic | historical_factorization_diagnostic | current_target_jet_pt2 | hydrogen | public_versioned_hepdata_csv | 5 tables/128 rows |
| [hepdata:44930](https://www.hepdata.net/record/44930) | ZEUS | diagnostic | high_energy_current_region_diagnostic | inclusive_current_region_xf_pt | proton | public_versioned_hepdata_csv | 11 tables/91 rows |
| [hepdata:45525](https://www.hepdata.net/record/45525) | H1 | diagnostic | high_energy_collinear_diagnostic | inclusive_breit_multiplicity | proton | public_versioned_hepdata_csv | 11 tables/80 rows |
| [hepdata:ins1217865](https://www.hepdata.net/record/ins1217865) | H1 | diagnostic | high_energy_current_region_diagnostic | inclusive_hcm_eta_pt | proton | public_versioned_hepdata_csv | 36 tables/267 rows |
| [arxiv:0809.1153](https://arxiv.org/abs/0809.1153) | CLAS | stage_2 | absolute_cross_section_candidate | absolute_cross_section_phi_pt2 | hydrogen | public_arxiv_source_with_ancillary_data | 53 source files |
| [arxiv:1103.1649](https://arxiv.org/abs/1103.1649) | JLab Hall C E00-108 | stage_2 | low_energy_cross_section_candidate | absolute_cross_section_pt2 | hydrogen and deuterium | public_arxiv_source_tables | 27 source files |
| [arxiv:1610.02350](https://arxiv.org/abs/1610.02350) | JLab Hall A E06-010 | diagnostic | nuclear_tmd_diagnostic | absolute_cross_section_phi_pt | 3He | public_arxiv_source_tables | 21 source files |
| [jlab:E12-09-017](https://www.jlab.org/exp_prog/experiments/summaries/E12-09-017_summary.pdf) | JLab Hall C E12-09-017 | deferred | future_data_candidate | planned_sidIS_pt | hydrogen and deuterium | proposal_and_status_public_no_final_tables_found | not_harvested |
| [jlab:clas_physics_database](https://clasweb.jlab.org/physicsdb/intro.html) | CLAS | deferred | provenance_pointer_only | mixed_sidIS | multiple | public_description_access_restricted | not_harvested |
| [arxiv:hep-ex/9511010](https://arxiv.org/abs/hep-ex/9511010) | ZEUS | diagnostic | high_energy_current_region_diagnostic | inclusive_current_region_xf_pt | proton | public_paper_hepdata_record_registered_as_44930 | not_harvested |

The HEPData archives and arXiv e-print sources are local ignored inputs. Their URLs and SHA256 hashes are in `data/global_source_download_manifest.json`; raw files are not part of the public source release. TeX tables and ancillary files require an explicit source-specific converter and unit test before a canonical observation is created.

The historical global-fit benchmark of 1,547 SIDIS points is a selected HERMES/COMPASS subset after TMD-validity cuts, not a claim that all raw tables should be fitted. Our target is to reproduce that count under an explicit cut/observable lock, then test additional JLab, EMC, E665, and HERA families progressively.
