# v22 standalone reference DY $W$ kernel

This layer assembles the already tested ingredients into one reusable
scalar reference implementation.

For each quark or antiquark leg,

$$
\begin{aligned}
F_q^{\rm OPE} = f_q + a_s(\mu_b) \left[ C_{q\leftarrow q}^{(1)}\otimes f_q + C_{q\leftarrow g}^{(1)}\otimes f_g \right].
\end{aligned}
$$

The DY luminosity is built from the charge-weighted
$q\bar q+\bar q q$ sum.

The strict one-loop result is

$$
\begin{aligned}
W_{\rm strict} = {\cal N} x_1x_2 e^{-S} \left[ L_0 + a_s(\mu_b)\Delta L_{\rm OPE} + \delta_H(Q)L_0 \right],
\end{aligned}
$$

where

$$
\begin{aligned}
{\cal N} = \frac{\text{observable prefactor}}{2\pi}.
\end{aligned}
$$

The multiplicative reference is

$$
\begin{aligned}
W_{\rm mult} = {\cal N} x_1x_2 e^{-S} H_{\rm DY}^{\rm NLO} L_{\rm product}^{\rm NLO}.
\end{aligned}
$$

`W_mult` contains formally higher-order hard--OPE and leg--leg cross
terms. Both forms are exposed so that the fixed-order expansion and the
eventual resummed implementation cannot be confused.

This module contains no fitted nonperturbative factor and no $Y$ term.
It is intentionally scalar and slow. A later vectorized implementation
must reproduce it.
