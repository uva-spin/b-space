# Perturbative calculation, OPE, and matching

This document is the numerical provenance for the fixed-target b_T-space
extraction. It states exactly what is implemented, which operations are
audited, and where the current production boundary stops. The phrase
“N3LL” refers to the resummed W-term setup used by the fixed-target campaign;
it is not approval of a universal collider W+Y prediction. The source-level
accuracy inventory in `systematics/perturbative_provenance_completion/` records
the same limitation.

## 1. Observable and conventions

The fitted object is a quark TMDPDF in impact-parameter space,
f_tilde(1,q/h)(x, b_T; mu, zeta). The fixed-target cross section is assembled
from charge-weighted quark/antiquark luminosities, the DY hard factor,
Sudakov evolution, and two nonperturbative factors. The neural model learns
only F_NP(x, b_T); it does not learn the perturbative kernel.

The convention is

```text
a_s(mu) = alpha_s(mu)/(4*pi)
b0     = 2*exp(-gamma_E)
mu_b   = b0/b_T
zeta_b = mu_b^2
```

before the finite-b_T profiles are applied. PDFs are ordinary densities
f(x, mu), not LHAPDF's x*f(x, mu); the backend converts `xfxQ` by dividing by
x. b_T is in GeV^-1, Q and mu in GeV, zeta in GeV^2, and k_T/q_T in GeV.

For each quark flavor q, the radial transform convention is

```text
f_q(x, k_T) = 1/(2*pi) * integral[0,infinity] db_T b_T J0(k_T*b_T) f_tilde_q(x,b_T)
```

The public k_T scripts use a finite, explicitly regularized transform; they
do not claim to reconstruct an unregularized perturbative high-k_T tail. See
`workflows/v23a/construction/` for the tail and taper controls.

## 2. Small-b_T OPE

The perturbative coordinate is b_star, not the physical large-b_T coordinate.
The TMD OPE is

```text
f_tilde_q/H(x,b_star;mu,zeta)
  = sum_j [ C_q<-j(.,b_star;mu,zeta) convolution f_j/H(.,mu) ](x)
```

with

```text
(C convolution f)(x) = integral_x^1 dz/z * C(z) * f(x/z)
```

The canonical CSS2 one-loop coefficients are

```text
C_q<-q(z) = delta(1-z) + a_s*C_F*[-pi^2/6*delta(1-z) + 2*(1-z)]
C_q<-g(z) = a_s*2*z*(1-z)
C_F       = 4/3
```

At a general perturbative scale define

```text
L_b      = ln(mu^2*b_star^2/b0^2)
ell_zeta = ln(mu^2/zeta)
T_R      = 1/2
```

The implemented general-scale coefficients are

```text
C_q<-q^(1)(z) = C_F * [ -2*L_b*((1+z^2)/(1-z))_+ + 2*(1-z)
                       + delta(1-z)*(-L_b^2 + 2*L_b*ell_zeta - pi^2/6) ]

C_q<-g^(1)(z) = T_R * [ -2*L_b*(z^2+(1-z)^2) + 4*z*(1-z) ]
```

The -pi^2*C_F/6 delta term belongs to the CSS2 matching coefficient in this
scheme; it is not moved into the DY hard factor. At L_b=ell_zeta=0 the
general-scale module reduces to the canonical module exactly.

### Numerical convolution

`v22/src/convolution.py` is the scalar reference implementation. Ordinary
terms use adaptive `scipy.integrate.quad`. Plus distributions are evaluated
without sampling the singular endpoint:

```text
integral_x^1 dz [g(z)]_+ phi(z)
  = integral_x^1 dz g(z)*[phi(z)-phi(1)]
    - phi(1)*integral_0^x dz g(z)
phi(z) = f(x/z)/z
```

The reference smoke tests use `epsabs=epsrel=1e-11`; the fit-time backend uses
`epsabs=1e-8`, `epsrel=1e-7` by default for each leg, exposed as
`v22_ope_epsabs` and `v22_ope_epsrel` configuration attributes. This split is
intentional: scalar tests establish the formula, while the cached backend
balances thousands of row-by-b_T evaluations against the same formula.

Run the checks with:

```bash
export PYTHONPATH="$PWD"
python v22/tests/run_convolution_smoke.py
python v22/tests/run_css2_ope_nlo_smoke.py
python v22/tests/run_css2_ope_nlo_general_smoke.py
```

The tests cover delta identity, regular Mellin convolution, both plus-kernel
endpoint subtractions, canonical/general-scale equality, and the endpoint
range through x=0.95.

## 3. b_star, OPE coordinate, and evolution profiles

The legacy smooth-profile backend uses

```text
b_star(b_T) = b_T/sqrt(1+(b_T/b_max)^2)
b_max       = 1.5 GeV^-1
```

and the defaults `b_min=1e-4`, `b_max=8`, `n_b=160`, `mu_min=1.3` GeV,
`q0=2` GeV, `nf=5`, and `n_sudakov_quad=32` Gauss-Legendre nodes. The
matching scale is computed from b_star, floored at `mu_min`, and capped at Q.
The production smooth widths are 0.12 GeV for both the lower floor and upper
Q cap; setting either width to zero reproduces the hard profile for a
diagnostic comparison.

Because a capped mu_b and an unmodified b_star would generate a large
negative L_b as b_T approaches zero, the OPE uses the profiled coordinate

```text
b_OPE     = (b_star^p + b_min(Q)^p)^(1/p)
b_min(Q)  = b0/(C5*Q)
C5        = 1
p         = 16
```

Thus b_OPE approaches b0/Q at the origin and b_star away from the cap. This
is a perturbative coordinate regularization, not a nonperturbative prior and
not a fitted uncertainty.

The Sudakov exponent is evaluated numerically as

```text
S(b_T,Q) = integral_mu_b^Q dmu/mu * [2*A(alpha_s(mu))*ln(Q/mu) + B(alpha_s(mu))]
```

using 32-point Gauss-Legendre quadrature in ln(mu). The source defines the
coefficients in powers of alpha_s/pi:

```text
A1 = C_F
A2 = 0.5*C_F*[ C_A*(67/18 - pi^2/6) - 5*nf/9 ]
B1 = -1.5*C_F
```

The explicit source expressions for A3 and B2 are in
`bt_internal_css_backend_v19_smoothprofile.py`. `ll`, `nll`, and `nnll`
select the corresponding prefixes. Historical `n3llp`/`n3ll_pilot` labels
currently return the same A1-A3,B1-B2 set as `nnll`; source and metadata
intentionally warn that the hard/OPE constants and finite-Y terms needed for a
complete N3LL-prime claim are not present in this backend.

## 4. DY hard factor and W organization

The one-loop CSS2 DY hard factor is separate from the OPE and fixed-target
unit conversion:

```text
H_DY(Q,mu) = 1 + a_s(mu)*C_F*[-16 + 7*pi^2/3 + 6*T - 2*T^2]
T          = ln(Q^2/mu^2)
```

The default hard scale is mu=Q. The flavor luminosity includes e_q^2 and both
q*qbar orderings; the electric charge is not included again in H_DY. The
fixed-target prefactor uses the historical `oldA_to_CS` convention,
`alpha_em=1/137.035999084`, and `hc_factor=3.893793656e8` as recorded in the
backend metadata.

For each b_T node the full backend performs:

1. Evaluate proton/target PDFs at mu_b, including the configured isospin map.
2. Evaluate q<-q and q<-g OPE convolutions on each signed-flavor leg.
3. Form the charge-weighted DY luminosity.
4. Multiply by the NLO hard factor and exp(-S).
5. Multiply by x1*x2, the fixed-target prefactor, and 1/(2*pi).
6. Integrate the supplied b_T grid with the training code's trapezoid rule and
   multiply by F_NP(x1,b_T)*F_NP(x2,b_T).

Two organizations are exposed for audits:

```text
W_strict = W0 + delta_W_H + delta_W_OPE
W_mult   = H_NLO * [C_NLO convolution f]_A * [C_NLO convolution f]_B * exp(-S)
```

The multiplicative form contains hard-OPE and leg-leg cross terms beyond
strict NLO. It is the resummed production organization used for the active
fixed-target W cache. The strict expansion is retained for subtraction and
closure audits so higher-order cross terms are not silently included in an
NLO asymptotic comparison. Run the implementation checks with:

```bash
PYTHONPATH=. python v22/tests/run_dy_hard_nlo_smoke.py
PYTHONPATH=. python v22/tests/run_dy_w_nlo_reference_smoke.py
PYTHONPATH=. python workflows/v22/audits/audit_v22_full_backend_integration.py --help
```

## 5. What “matched” means in each scope

The word `matched` is overloaded and must be read with the scope label.

### Fixed-target production

The accepted production extraction is a low-q_T, b_T-space W-term result. Its
production cache uses the perturbative W backend described above and Y=0 for
the accepted low-q_T rows. The `--mode matched` option in fit drivers controls
the kinematic data selection (q_T/Q cut); it does not prove that an externally
validated high-q_T W+Y observable exists.

### Unitary finite-Y diagnostic

For the isolated Tevatron boundary study, the candidate was

```text
Y_unitary = p(r) * [FO_NLO - W]
matched   = W + Y_unitary
r         = q_T/Q
```

Here p(r)=0 for r<=0.20, p(r)=1 for r>=0.30, and between those endpoints

```text
p(t) = t^3 * (6*t^2 - 15*t + 10)
```

This guarantees the W limit in the core and the fixed-order limit at the outer
boundary even when W and its asymptotic expansion are not numerically close.
Node studies used Simpson grids with 320 and 640 nodes; the production
boundary used the genuine NLO fixed-order inputs recorded in
`systematics/finite_y_completion_2026/`. This is a validated diagnostic
transition, not the conventional production Y=FO-ASY.

### Conventional external W+Y candidate

The isolated DYTurbo candidate uses the conventional decomposition

```text
W   = RES
ASY = -CT
FO  = VJ
Y   = FO - ASY = VJ + CT
```

so the reconstructed observable is `RES+CT+VJ`. The DYTurbo cards used
unprimed order 3, `primed=false`, real and virtual V+jet pieces enabled, and
the stated rapidity/acceptance cuts. The 122-row Tevatron grid and the
96-start/50-replica diagnostic are archived under
`systematics/full_n3ll_wy_production_2026/`; the candidate is not promoted to
replace the fixed-target production result. The six LHCb rows remain W-only
diagnostics because the finite-Y subtraction, observable normalization, and
covariance closure are not complete.

## 6. Reproduction and validation order

From a clean checkout:

```bash
python -m pip install -r requirements.txt
export PYTHONPATH="$PWD"
python v22/tests/run_convolution_smoke.py
python v22/tests/run_css2_ope_nlo_smoke.py
python v22/tests/run_css2_ope_nlo_general_smoke.py
python v22/tests/run_dy_hard_nlo_smoke.py
python v22/tests/run_dy_w_nlo_reference_smoke.py
python v22/tests/run_small_b_profile_smoke.py
```

Then select an explicit data-table variant, PDF member, trainer, backend, and
fresh output tag. Build a row-aligned cache, inspect its metadata, run a
central fit, and only then construct b-space grids and replica bands. For every
numerical refinement, record the following in a manifest:

- git commit and source hashes;
- data files and corrected-row variant;
- PDF set/member and alpha_s provider;
- `resum_order`, `match_order`, `n_f`, b_star, mu-floor/cap, OPE profile, and
  quadrature settings;
- target-isospin map, electromagnetic/unit-conversion convention, and q_T/Q
  cuts;
- cache row count and row-id hash;
- fit seed, optimizer, epoch/plateau gate, and nuisance treatment;
- replica/start counts and all failed or rescued runs.

The detailed command sequence is in `docs/REPRODUCIBILITY.md`; workflow
drivers and their stage-by-stage prerequisites are in `workflows/v22/README.md`
and `workflows/v23a/README.md`.

## 7. Source-to-detail map

| Question | Source of truth |
| --- | --- |
| canonical/general-scale OPE equations | `v22/src/css2_ope_nlo.py`, `v22/src/css2_ope_nlo_general.py` |
| plus-distribution numerical treatment | `v22/src/convolution.py` |
| DY hard factor | `v22/src/dy_hard_nlo.py`, `v22/DY_HARD_FACTOR.md` |
| Sudakov coefficients and scale profile | `bt_internal_css_backend_v19_smoothprofile.py` |
| OPE-coordinate cap | `v22/src/small_b_profile.py` |
| NLO W luminosity organization | `v22/src/dy_w_nlo_reference.py`, `v22/backends/bt_internal_css_backend_v22_full.py` |
| fixed-target row/cache assembly | `v22/backends/`, `v22/tools/` |
| unitary finite-Y definition | `systematics/finite_y_completion_2026/backend/unitary_finite_y.py` |
| conventional external W+Y decomposition | `systematics/full_n3ll_wy_production_2026/` |
| promotion limits and frozen status | `production/`, `systematics/*/HANDOFF.md` |
