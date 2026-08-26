# v22 small-$b_T$ matching-coordinate profile

## Problem exposed by the full-profile audit

The existing scale profile caps the matching scale at the hard scale,

$$
\begin{aligned}
\mu_b \le Q.
\end{aligned}
$$

If the coefficient functions continue to use the unmodified
$b_*\to0$ coordinate, then

$$
\begin{aligned}
L_b=\ln\frac{\mu_b^2b_*^2}{b_0^2}
\end{aligned}
$$

becomes arbitrarily large and negative as $b_T\to0$. A finite-order
coefficient expansion is then not meaningful point by point.

## Profiled perturbative coordinate

For the OPE coefficients only, define

$$
\begin{aligned}
b_{\min}(Q)=\frac{b_0}{C_5Q},
\end{aligned}
$$

and a smooth maximum

$$
\begin{aligned}
b_{\rm OPE} = \left[ b_*^p+b_{\min}^p \right]^{1/p}.
\end{aligned}
$$

The default is

$$
\begin{aligned}
C_5=1,\qquad p=16.
\end{aligned}
$$

Thus:

- $b_{\rm OPE}\to b_0/Q$ as $b_T\to0$;
- $b_{\rm OPE}\to b_*$ away from the small-$b_T$ cap;
- with $\mu_b=Q$, $L_b\to0$ as $b_T\to0$;
- the large-$b_T$ infrared-floor treatment is unchanged.

This is a perturbative profile choice, not a fitted nonperturbative
constraint. It must later be varied as part of the theory/profile
uncertainty.
