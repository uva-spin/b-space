# v22 one-loop CSS2 OPE scheme lock

## Purpose

This file locks the first explicit perturbative definition of the v22
quark TMDPDF. It is a theory convention, not a fit to another TMD
extraction.

The observable-level validation targets remain MCFM and DYTurbo.

## Expansion parameter

\[
a_s(\mu)=\frac{\alpha_s(\mu)}{4\pi}.
\]

## Canonical small-\(b_T\) scales

\[
b_0 = 2e^{-\gamma_E},\qquad
\mu_b=\frac{b_0}{b_T},\qquad
\zeta_b=\mu_b^2.
\]

A regulated/profiled version of \(\mu_b\) will be introduced only after
the canonical implementation passes its analytic checks.

## Small-\(b_T\) OPE

For a quark or antiquark flavor \(q\),

\[
\widetilde f_{q/H}(x,b_T;\mu_b,\zeta_b)
=
\sum_j
\left[
\widetilde C_{q/j}\otimes f_{j/H}
\right](x,\mu_b)
+\mathcal O(b_T^2\Lambda_{\rm QCD}^2).
\]

At one loop in the CSS2 convention and at the canonical scales,

\[
\widetilde C_{q/q}(z)
=
\delta(1-z)
+
a_s C_F
\left[
-\frac{\pi^2}{6}\delta(1-z)
+
2(1-z)
\right],
\]

\[
\widetilde C_{q/g}(z)
=
a_s\,2z(1-z),
\]

and the other flavor channels vanish at this order.

The convolution convention is

\[
(C\otimes f)(x)
=
\int_x^1\frac{dz}{z}\,
C(z)f(x/z).
\]

## Separation from the hard factor

The \(-\pi^2 C_F/6\) delta term belongs to the CSS2 TMD matching
coefficient in this scheme. It must not be silently moved into the
Drell--Yan hard factor.

The hard factor will be implemented and tested in a separate milestone.

## Scope of this milestone

This layer implements only:

- canonical-scale one-loop \(q\leftarrow q\) matching;
- canonical-scale one-loop \(q\leftarrow g\) matching;
- decomposition into Born, quark and gluon pieces;
- scalar high-accuracy reference evaluation.

It does not yet implement:

- general \((\mu,\zeta)\) logarithms;
- Collins--Soper evolution;
- the Drell--Yan hard factor;
- heavy-flavor matching;
- a \(b_*\) or scale profile;
- nonperturbative functions;
- a real-data fit.
