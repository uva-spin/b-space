# HERMES/COMPASS 1,547-point benchmark audit

Status: **not reproduced and not fit-authorized**. The 1,547 value is the post-cut HERMES/COMPASS count in arXiv:2206.07598, not a raw HEPData row total.

Locked literature cuts: `Q > 1.4 GeV`, `0.2 < z < 0.7`, `|P_hT| < min[min(0.2 Q, 0.5 z Q)+0.3 GeV, z Q]`, with the published HERMES vector-meson-subtracted `zxpt-3D` scope and COMPASS vector-boson-subtracted release.

- HERMES target count: **344**; available public HEPData pT projection rows: **288** across **16** tables. The HEPData projection lacks Q/x axes and the supplemental archive/covariance sidecars, so it cannot certify the 344 selection.
- COMPASS target count: **1203**; available HEPData primary rows: **4664**. Deterministic representative-value counts range across the recorded policies: **1078--1285**, demonstrating why a source selection manifest is required.

No rows are approved. The next gate is to obtain/mirror the HERMES `zxpt-3D` value and covariance files, define the COMPASS point-level bin convention, then freeze a row-level selection manifest before any fit.
