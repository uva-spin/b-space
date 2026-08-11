# v22 TMD conventions

## Scope

This file defines the object that v22 will calculate and eventually fit.
It is a convention lock, not a phenomenological fit and not a comparison
to another TMD extraction.

The external perturbative validation targets remain **MCFM** and
**DYTurbo** at the observable level. No published TMD extraction is used
as a calibration target.

## Published TMDPDF

The unpolarized quark TMDPDF is

\[
\widetilde f_{1,q/h}(x,b_T;\mu,\zeta).
\]

The default publication point is

\[
\mu=Q,\qquad \zeta=Q^2.
\]

The object is **not** multiplied by \(x\) or \(b_T\).

With the conventions below, the \(b_T\)-space TMD is dimensionless and
the \(k_T\)-space TMD has units of \({\rm GeV}^{-2}\).

## Fourier convention

\[
\widetilde f(x,\boldsymbol b_T)
=
\int d^2\boldsymbol k_T\,
e^{+i\boldsymbol k_T\cdot\boldsymbol b_T}
f(x,\boldsymbol k_T),
\]

\[
f(x,\boldsymbol k_T)
=
\int\frac{d^2\boldsymbol b_T}{(2\pi)^2}\,
e^{-i\boldsymbol k_T\cdot\boldsymbol b_T}
\widetilde f(x,\boldsymbol b_T).
\]

For azimuthally symmetric functions,

\[
f(x,k_T)
=
\frac{1}{2\pi}
\int_0^\infty db_T\,b_T
J_0(k_Tb_T)\widetilde f(x,b_T).
\]

Therefore,

\[
\int d^2\boldsymbol k_T\,f(x,k_T)
=
\widetilde f(x,b_T=0)
\]

when the integral exists in the regulated/perturbative sense relevant to
the chosen scheme.

## Drell--Yan structure function

The radial Fourier transform is

\[
W(q_T)
=
\frac{1}{2\pi}
\int_0^\infty db_T\,b_T J_0(q_Tb_T)
\widetilde W(b_T).
\]

At leading flavor structure,

\[
\widetilde W(b_T)
=
H_{\rm DY}
\sum_q e_q^2
\left[
\widetilde f_{q/A}(x_1,b_T)
\widetilde f_{\bar q/B}(x_2,b_T)
+
\widetilde f_{\bar q/A}(x_1,b_T)
\widetilde f_{q/B}(x_2,b_T)
\right].
\]

Electroweak prefactors, bin integration and the matched \(Y\) term are
observable-layer components and will be documented separately.

## Collinear PDF convention

LHAPDF returns `xfxQ(pid,x,Q) = x f_i(x,Q)`.  Every v22 OPE convolution
must explicitly divide by \(x\) before using a number as \(f_i(x,Q)\).

No variable named `pdf` may ambiguously hold `x f` in the v22 code.

## Perturbative expansion convention

Internally,

\[
a_s = \frac{\alpha_s}{4\pi}.
\]

Coefficient functions and the hard factor are expanded as

\[
C = \delta + \sum_{n\ge1}a_s^n C^{(n)},\qquad
H = 1 + \sum_{n\ge1}a_s^n H^{(n)}.
\]

Any formula imported from a convention using
\(\alpha_s/\pi\) or \(\alpha_s/(2\pi)\) must be converted explicitly and
covered by a unit test.

## Collins--Soper sign convention

\[
\frac{d\ln \widetilde f}{d\ln\sqrt{\zeta}}
=
K(b_T;\mu),
\]

\[
\frac{dK(b_T;\mu)}{d\ln\mu}
=
-\gamma_K(\alpha_s).
\]

An alternative \(D\)-kernel notation may be exposed only through an
explicit conversion function.  Silent sign or factor-of-two changes are
not allowed.

## Nonperturbative factor

For every fitted flavor structure,

\[
F_{\rm NP}(x,0)=1,\qquad
F_{\rm NP}(x,b_T)=1+\mathcal O(b_T^2\Lambda_{\rm QCD}^2).
\]

A momentum-space TMD or an auxiliary factor is not required to be
positive at every \(k_T\). Positivity is not used as a fit constraint.

## Phase-A validation order

1. Analytic Fourier and Gaussian tests.
2. LO OPE identity and explicit `xfx/x` conversion.
3. NLO OPE coefficient-function tests.
4. \(\mu\)- and \(\zeta\)-evolution path-independence tests.
5. Expansion of the resummed result against the validated singular
   fixed-order result.
6. Matched observable checks against MCFM and DYTurbo.
7. Experimental-bin integration checks.

The existing v21 `f_eff` object must not be relabeled as this v22 TMDPDF.
