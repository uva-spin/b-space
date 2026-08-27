# APFEL++ SIDIS coefficient-interface probe (2026-08-26)

This is an exploratory interface check, not a fit or production prediction.
The local APFEL++ build evaluated the factorized SIDIS C20/C21 and CL1
operators with NNPDF40 NLO and the identified NNFF10 NLO grids. The expansion
uses `alpha_s/(4 pi)`. Signed fixed-order values are retained.

The LO APFEL ratio agrees with the direct Python `xfxQ/z` implementation at
the few-percent level on the deliberately coarse degree-3 interpolation grid
(median relative difference 2.48%, 95th percentile 4.49%). This validates the
distribution convention and the factor of `z` needed when converting APFEL's
raw C20 output to a multiplicity ratio. The representative K+ row at
`x=0.015`, `z=0.225`, `Q=1.643844 GeV`, `y=0.6` gives LO 1.14542 and the
SIDIS-NLO numerator diagnostic 1.17861.

The output has 740 positive LO and 738 positive NLO rows out of 746. These
numbers are not used in the initial fit. The probe still uses an LO inclusive
DIS denominator; the full NLO/NNLO denominator, phase-space integration,
scale choices, and the kinematics-dependent normalization factor must be
implemented and independently closed before these operators enter a fit.
