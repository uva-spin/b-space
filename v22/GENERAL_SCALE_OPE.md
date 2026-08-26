# v22 general-scale one-loop TMD OPE

## Why this layer is required

The existing profile uses a perturbative transverse coordinate $b_*$
and a profiled matching scale $\mu_b$.  Over much of the fixed-target
support,

$$
\begin{aligned}
\mu_b \ne \frac{b_0}{b_*}, \qquad b_0=2e^{-\gamma_E}.
\end{aligned}
$$

Therefore the canonical $L_b=0$ matching coefficients are not enough.

This does **not** mean the OPE is evaluated at the physical large
$b_T$.  The perturbative coefficients use $b_*$; the fitted
nonperturbative factor carries the remaining physical-$b_T$
dependence.

## Logarithms

$$
\begin{aligned}
L_b = \ln\left(\frac{\mu^2 b_*^2}{b_0^2}\right), \qquad \ell_\zeta = \ln\left(\frac{\mu^2}{\zeta}\right).
\end{aligned}
$$

The default v22 boundary choice is

$$
\begin{aligned}
\zeta_b=\mu_b^2,
\end{aligned}
$$

so $\ell_\zeta=0$, but the implementation keeps general $\zeta$.

## One-loop coefficients

The expansion parameter is

$$
\begin{aligned}
a_s=\frac{\alpha_s}{4\pi}.
\end{aligned}
$$

For the quark TMDPDF,

$$
\begin{aligned}
C_{q\leftarrow q}^{(1)}(z) = C_F\left[ -2L_b\left(\frac{1+z^2}{1-z}\right)_+ +2(1-z) +\delta(1-z) \left( -L_b^2+2L_b\ell_\zeta-\frac{\pi^2}{6} \right) \right],
\end{aligned}
$$

$$
\begin{aligned}
C_{q\leftarrow g}^{(1)}(z) = T_R\left[ -2L_b\left(z^2+(1-z)^2\right) +4z(1-z) \right].
\end{aligned}
$$

At $L_b=\ell_\zeta=0$, these reduce exactly to the canonical module
already tested in `v22/src/css2_ope_nlo.py`.

## Scope

This module is a scalar high-accuracy reference implementation.  It is
not yet the vectorized fit-time implementation and does not yet assemble
the full DY $W$ term.
