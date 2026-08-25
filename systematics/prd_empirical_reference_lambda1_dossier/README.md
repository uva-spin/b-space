# PRD methods dossier: empirical-reference lambda=1 candidate

This directory is an independent, read-only reconstruction of the first
replacement candidate registered by the 2026 dataset-identifiability study.
Nothing here changes the campaign or frozen production inputs.

## One-sentence result

The candidate retains the accepted monotone FiLM architecture and the 329-row
`qT/Q <= 0.20` dataset, but adds a pointwise squared relative-distance penalty
to the empirical median of 24 historical independent-start `F_NP` curves over
`0.1 <= bT <= 2 GeV^-1`, with strength `lambda=1`; crossing 24 refitted starts
with 50 conditional experimental replicas produces 1,200 operational ensemble
members per flavor and reduces the largest active Fig. 6 q16--q84 full width
from 24.00% to 11.77% for u and from 25.53% to 12.49% for d.

## Data and likelihood

- Dataset: the accepted D020_ALL selection, 329 rows in nine datasets:
  E288 at 200, 300, and 400 GeV; E605; E772; CDF Run 1 and Run 2; D0 Run 1;
  and LHCb at 7 TeV.
- Prediction for row `i`:

  `t_i = N_{d(i)} [ sum_j K_ij F_NP(x1_i,b_j) F_NP(x2_i,b_j) + Y_i ]`.

  The precomputed perturbative kernel and Y term are frozen. `N_d` is the
  positive floated normalization for dataset `d`.
- The unregularized likelihood is the weighted data chi-square plus Gaussian
  normalization penalties. The likelihood weight is one.
- Experimental replicas fluctuate each dataset coherently by its quoted
  normalization uncertainty and each point independently by its uncorrelated
  uncertainty. Seeds 1001--1050 define the 50 replicas.

## Neural architecture (unchanged from the accepted fit)

The learned factor is positive, normalized at the origin, and monotone in b:

`F_NP(x,b) = exp[- integral_0^b 2 b' A_theta(x,b') db']`, with `A_theta >= 0`.

The cumulative integral is evaluated by trapezoidal quadrature on the ordered
b grid, so `F_NP(x,0)=1`, `0 < F_NP <= 1`, and `dF_NP/db <= 0` by construction.
The exponent is clipped at magnitude 40.

Architecture details:

- Radial inputs: `(b, b^2, sqrt(b+1e-8), log(1+b))`.
- Radial projection: `Linear(4,48)` followed by tanh.
- Conditioning inputs: `(x, log[x/(1-x)])`, with x clipped to
  `[1e-6,1-1e-6]`.
- Conditioning network: `Linear(2,32)-SiLU-Linear(32,32)-SiLU`.
- Three FiLM residual blocks of width 48. Each block applies a width-48 linear
  layer and tanh, then positive scale
  `gamma=softplus(Linear(c))+1e-6`, bounded shift
  `beta=tanh(Linear(c))`, a second width-48 linear layer, residual addition,
  and final tanh.
- Scalar head: `Linear(48,1)` followed by softplus; the initial constant
  damping coefficient is `A=0.05` and the minimum is zero.
- Before integration, `A_theta` is smoothed in physical b with a normalized
  Gaussian kernel of width `0.45 GeV^-1`.
- A fixed late-b damping floor is added:
  `0.08 sigmoid[(b-3.5)/0.25]`.
- Trainable neural parameters: 25,057, plus nine log-normalization nuisance
  parameters (fixed-normalization entries, if any, are masked in the loss).

No capacity override, spline, matched-tail replacement, PCA model, or other
architectural constraint was enabled for this candidate.

## Empirical reference and added objective

The reference is constructed pointwise from the median of 24 exactly
reproduced historical baseline endpoints. It is tabulated on eight x values

`{0.001, 0.003, 0.01, 0.03, 0.1, 0.2, 0.4, 0.7}`

and the common dense b grid. Its q16 and q84 values are retained for
sensitivity bookkeeping, but only its pointwise median is used in the fit.

For b nodes in `0.1 <= b <= 2 GeV^-1`, the added per-row objective term is

`R_ref = lambda mean_{x,b} [ (F_NP(x,b)-F_ref(x,b)) / max(F_ref(x,b),0.10) ]^2`,

with `lambda=1`. The full minimized scalar is

`L/N = (chi2_data + chi2_norm)/N + R_ref`.

Equivalently, the implementation adds `N R_ref` to the total objective before
dividing by the 329 rows. Thus lambda is the coefficient of a dimensionless
mean penalty in the per-row objective; it is not a Gaussian-prior standard
deviation and should not be described as one.

For the seed-303 endpoint, the final reference contribution was
`3.0937e-4` per row, compared with an unpenalized likelihood contribution of
`0.41780` per row. Its unpenalized total chi-square was 137.456. Across all 24
starts, the maximum source-relative unpenalized chi-square change was +0.2964
and the median change was -0.8737, so the registered candidate was classified
as fit preserving.

## Initialization and optimization

- Starts: seeds 303--326, using the corresponding historical stationary Fig. 6
  polish states as initial neural states and their corresponding dataset
  normalizations.
- The matched re-fit deliberately runs zero AdamW epochs. Although the general
  runner supports AdamW (`lr=2e-5`, zero weight decay, gradient clipping at 10,
  and a ReduceLROnPlateau scheduler), that stage is bypassed here.
- Each candidate is polished directly with PyTorch L-BFGS: learning rate 1,
  history size 50, strong-Wolfe line search, maximum 500 iterations,
  gradient tolerance `1e-7`, and parameter/function-change tolerance `1e-10`.
- Computation is float32 for this lambda=1 matched-baseline branch: the driver
  did not pass the runner's `--float64` option.
- Example (seed 303): 524 closure evaluations, objective per row decreasing
  from 0.425394 at the L-BFGS start to 0.418108 at the endpoint.
- The registry explicitly says optimizer drift was not the primary promotion
  gate. This nuance matters: the candidate is a fit-preserving operational
  ensemble, not proof that every neural direction is numerically stationary.

## Ensemble and uncertainty semantics

The 24-start and 50-replica distributions are combined as correlated curves,
not as independent point errors. Let `l_c(b)=log F_c(b)`, let `l_s` denote the
24 start curves, and `l_r` the 50 replica curves. The construction is

`l_sr(b) = l_c(b) + [l_s(b)-median_s l_s(b)] + [l_r(b)-median_r l_r(b)]`.

All 24x50 combinations are retained. This Cartesian convolution in log space
preserves positivity and each curve's b correlation while treating starts and
replicas with equal empirical weight. The plotted central line is the
pointwise median after propagation, and the band is pointwise q16--q84.

This is best called an **operational hierarchical empirical 68% band**. It
should not be called a frequentist confidence interval or Bayesian credible
interval: the start distribution is an empirical non-uniqueness sample, and
the two components are combined by a declared construction rather than a
derived probability model.

## Fig. 2 and Fig. 6 propagation

- b-space TMDs multiply each candidate `F_NP` curve by the frozen perturbative
  `ftilde_no_np` curve. Fig. 2 uses x=0.1, Q=7.5 GeV and six flavors.
- Fig. 6 uses x=0.1, Q=10 GeV and u,d. Each complete member is Fourier-Bessel
  transformed with
  `f(k)=1/(2*pi) integral db b J0(kb) ftilde(b)`.
- Transform settings: exponential-in-b-squared tail continuation; transformed
  interval to `b=24 GeV^-1` with 6001 nodes; `kT=0--4 GeV` with 401 nodes; end
  taper begins at 92% of the transform interval. Reported Fig. 6 metrics use
  `kT <= 2.25 GeV`.
- An active point is one where the full-ensemble median exceeds 5% of the peak
  for that flavor/Q. The quoted maximum is the largest
  `(q84-q16)/median` over active points.

| flavor | candidate max full width | historical baseline | relative reduction |
|---|---:|---:|---:|
| u | 11.7726% | 23.9968% | 50.9408% |
| d | 12.4901% | 25.5318% | 51.0803% |

The asymmetric maxima recorded by the combined summary are 7.181% lower and
5.576% upper excursion for u, and 7.625% lower and 5.626% upper excursion for
d. These extrema need not occur at the same kT, so they should not be summed
to reproduce the maximum full width.

## Stability and limitations that belong in the paper

- The 24-start b-space endpoint bootstrap (500 draws) has a 95th-percentile
  maximum endpoint movement of 7.41%; repeated split halves (200 draws) have a
  95th-percentile maximum endpoint difference of 10.15%.
- The corresponding k-space values are 8.43% and 8.74%.
- These fail the preregistered 2% endpoint-stability gate. The registry keeps
  lambda=1 as the first complete fit-preserving improvement, but labels it a
  provisional champion and not production.
- The reference is empirical and derived from the same full 24-start baseline;
  it is not reciprocal-cross-fitted. That makes it a transparent
  regularization convention, not an external-data constraint.
- The result must not be presented as data-only identifiability. Earlier local
  profiles showed that a functional prior/model choice is unavoidable beyond
  the data-identified region.
- Neither this candidate nor this dossier modifies the frozen production fit.

## Suggested PRD wording

> To control residual functional non-identifiability without changing the
> accepted neural parametrization or dataset, we supplemented the normalized
> chi-square objective with a dimensionless pointwise distance to an empirical
> reference curve. The reference was the pointwise median of 24 independently
> initialized baseline solutions, evaluated at eight x nodes and over
> 0.1 <= b_T <= 2 GeV^-1. We minimized
> R_ref = lambda <[(F_NP-F_ref)/max(F_ref,0.1)]^2> with lambda=1, together with
> the data and normalization terms, starting from each of the 24 baseline
> endpoints and applying a 500-iteration strong-Wolfe L-BFGS polish. The
> regularizer preserved fit quality: the largest increase of the unpenalized
> chi-square relative to the corresponding source solution was 0.30. We then
> combined the 24-start distribution with 50 experimental replicas by a
> Cartesian convolution of centered log-F_NP residual curves, yielding 1,200
> members per flavor. Pointwise 16th--84th percentiles define the operational
> combined band. In the active region of Fig. 6, its maximum full relative
> width is 11.77% for u and 12.49% for d, approximately 51% narrower than the
> historical baseline. Because the reference is empirical and the finite
> 24-start endpoints retain 7--10% resampling sensitivity, we regard this as a
> provisional regularized result rather than a data-only or production-level
> uncertainty determination.

## Read-only provenance

The numerical claims above were traced to these campaign artifacts:

- `summaries/champion_registry/empirical_reference_lambda1_b0p1_2p0_full24.json`
- `summaries/exact_baseline_fnp_median/{summary.json,fnp_median.csv}`
- `summaries/matched_baseline_reference_distance_lam1e00_full24_crossed_experimental/summary.json`
- `summaries/matched_baseline_reference_distance_lam1e00_full24_{bspace,kspace}_stability/summary.json`
- `scripts/run_matched_baseline_reference_distance.py`
- `scripts/build_final_combined_tmd_ensemble.py` (the documented combination
  and transform implementation; the lambda=1 crossed summary records the same
  operational construction)
- the seed-303 `fit_status.json` under the campaign outputs
- the accepted model definition in
  `v21_smoothedA_tail_candidate/train_bt_dnn_v21_smoothedA_tail.py`
