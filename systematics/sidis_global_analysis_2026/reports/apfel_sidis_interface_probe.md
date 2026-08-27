# APFEL++ SIDIS coefficient and denominator probe (2026-08-26)

This isolated diagnostic evaluates the massless NLO SIDIS C20/C21/CL1 operators and the NLO inclusive-DIS F2/FL denominator with the same PDF member. It is not a production prediction.

The row-level table contains 746 identified COMPASS 2026 pi/K rows. The LO ratio is positive for 740 rows; the SIDIS-NLO numerator over an LO denominator is positive for 738; using the full NLO denominator leaves 738 positive rows.
Relative to the LO-denominator numerator diagnostic, the full-denominator ratio has median multiplicative shift 1.1199 (5--95% range 1.0080--1.1677).

The full denominator is now assembled through APFEL's Observable path, which includes the NLO coefficient-function and PDF-evolution terms. The remaining validation gates are bin-averaged phase-space integration, scale/threshold choices, and covariance-consistent normalization. The eight non-positive rows are retained in the manifest and excluded from the positive-ratio pilot rather than positivity-clipped.

The corresponding isolated joint-fit diagnostic gives DY chi2/row = 0.3943 and SIDIS chi2/row = 12.9775 on 738 rows, with 8 rows excluded for non-positive theory ratios. This is an interface test, not a promotion candidate.

No frozen production files were modified.
