# v22 one-loop Drell--Yan hard factor

The CSS2 Drell--Yan hard factor is the absolute square of the
time-like quark form-factor hard part.  The electric charge factor is
**not** included here because the v22 flavor luminosity already contains
\(e_q^2\).

The internal perturbative expansion parameter is

\[
a_s(\mu)=\frac{\alpha_s(\mu)}{4\pi}.
\]

Define

\[
T=\ln\frac{Q^2}{\mu^2}.
\]

At one loop,

\[
H_{\rm DY}(Q,\mu)
=
1
+
a_s(\mu)\,C_F
\left[
-16+\frac{7\pi^2}{3}+6T-2T^2
\right]
+\mathcal O(a_s^2).
\]

The default hard scale is \(\mu_H=Q\), so

\[
H_{\rm DY}(Q,Q)
=
1+
a_s(Q)\,C_F
\left[
-16+\frac{7\pi^2}{3}
\right].
\]

The hard factor is separate from:

- the fixed-target electromagnetic and unit-conversion prefactor;
- the TMD OPE coefficients;
- the Sudakov evolution;
- the finite \(Y\) term.

For strict NLO bookkeeping,

\[
W_{\rm NLO}
=
W_{\rm Born}
+
\delta W_H
+
\delta W_{\rm OPE}.
\]

Multiplying the complete NLO hard factor by two complete NLO legs is a
valid resummed implementation choice, but it contains formally
higher-order cross terms.  v22 audits those terms explicitly before
using the multiplicative form.
