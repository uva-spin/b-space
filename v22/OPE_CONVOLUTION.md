# v22 OPE convolution reference

The v22 matching convolution is

$$
\begin{aligned}
(K\otimes f)(x) = \int_x^1 \frac{dz}{z} K(z) f(x/z).
\end{aligned}
$$

For a plus distribution,

$$
\begin{aligned}
\int_x^1 dz [g(z)]_+\phi(z) = \int_x^1 dz g(z) [\phi(z)-\phi(1)] - \phi(1)\int_0^x dz g(z),
\end{aligned}
$$

where

$$
\begin{aligned}
\phi(z)=\frac{f(x/z)}{z}.
\end{aligned}
$$

`v22/src/convolution.py` is a scalar high-accuracy reference
implementation. It is not the eventual training-time vectorized
implementation.

The smoke tests cover:

- delta-function identity;
- ordinary Mellin convolution;
- $ [1/(1-z)]_+ $;
- $ [\ln(1-z)/(1-z)]_+ $;
- a composite distribution;
- endpoint behavior up to $x=0.95$.

The actual NLO TMD matching coefficients are intentionally not inserted
until the subtraction/renormalization scheme and perturbative convention
are explicitly locked.
