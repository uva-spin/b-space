# b-space: N³LL fixed-target DY TMD extraction  
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/) [![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE) [![TMD space](https://img.shields.io/badge/space-b__T%20primary-blueviolet)](#physics-scope) [![kT companion](https://img.shields.io/badge/k__T-regularized%20companion-lightgrey)](#regularized-kt-space-companion)

This repository contains the fixed-target Drell--Yan extraction of unpolarized TMDPDFs in **impact-parameter space**.  The extraction uses an **N³LL-resummed b-space \(W\)-term backend** with NLO hard/OPE matching in the \(W\) term and a constrained neural-network model for the nonperturbative factor \(F_{\rm NP}(x,b_T)\).

The repository is intended to accompany the fixed-target DY TMD note/paper and to provide reproducible scripts, frozen audit outputs, and plotting utilities for the \(b_T\)-space result and its regularized \(k_T\)-space companion.

The main result is a fixed-target DY TMD extraction using

```text
E288_200, E288_300, E288_400, E605, E772
```

with the corrected `E288_300:99` row, explicit normalization priors, a controlled point-to-point uncertainty sensitivity for E772/E288_400, experimental pseudo-data replicas, and a PDF-member overlay in the final TMD reconstruction.

---

## Status at a glance

| Layer | Status | Meaning |
|---|---:|---|
| Fixed-target DY b-space extraction | **Complete** | The production object is the \(b_T\)-space TMD ensemble for E288, E605, and E772. |
| Perturbative accuracy label | **N³LL W-term** | The cached \(W\)-term backend uses the N³LL resummed b-space evolution setup used in the extraction. |
| Hard/OPE matching in \(W\) | **Included** | NLO hard and OPE insertions are included in the \(W\)-term construction used for the v23a result. |
| Experimental pseudo-data replicas | **Included** | Replica fits use `target_used`, generated from row-level and dataset-level uncertainties. |
| PDF uncertainty | **Overlay included** | Final TMD bands include a PDF-member overlay in the perturbative/OPE TMD reconstruction. |
| PDF-through-refit uncertainty | **Not yet included** | PDF members are not yet propagated through a full retraining of \(F_{\rm NP}\). |
| \(k_T\)-space representation | **Regularized companion** | \(k_T\)-space curves are obtained by a regularized finite-\(b_T\) Hankel transform. |
| Accelerator/collider DY data | **Next project** | Tevatron, RHIC, and LHC datasets require separate unit, covariance, EW/Z, and bin-integration review. |

## Current lambda=1 production update

The current campaign production package is
[`production/lambda1_empirical_reference_full96x50`](production/lambda1_empirical_reference_full96x50/).
It uses the lambda=1 empirical-reference objective on `x=0.1` and
`0.1 <= bT <= 2.0 GeV^-1`, with 96 stationary starts crossed with 50
conditional experimental replicas (4,800 members per flavor). All 48 newly
added starts passed the FNP stationarity gate.

The production package reports operational q16--q84 full widths of 21.257%
for `u` and 22.480% for `d` in the active k-space region. These are ensemble
bands, not calibrated 68% confidence intervals. The earlier 24-start package
is retained as the rollback reference; frozen source outputs were not
overwritten.

Expected fixed-target high-level result:

```text
b-space ensemble:
  n replicas: 50
  chi2 q95: 2.1310145686109365
  norm-pull q95: 2.707433342933655
  random-split width q90: 0.6439466887345098
  random-split center q90: 0.021749180926595522
  b-space band technical pass: True
  b-space band uncertainty-useful pass: True

regularized kT companion:
  default tail mode: expb2
  kT range: 0 <= kT <= 4 GeV
  regularization-mode p90 max relative difference: 0.030912545977260095
  regularization-mode max relative difference: 0.04304417062181683
  regularization stability pass: True
```

---

## Physics scope

This repository focuses on the **fixed-target low-\(Q\) Drell--Yan region**.  The included production extraction uses:

- E288 proton--nucleus Drell--Yan data at 200, 300, and 400 GeV beam energies;
- E605 fixed-target Drell--Yan data;
- E772 fixed-target Drell--Yan data;
- matched/TMD-region cuts used in the v23a fixed-target workflow;
- corrected provenance for the known `E288_300:99` row;
- explicit 15% normalization-prior treatment;
- a 5% point-to-point uncertainty sensitivity for E772 and E288_400;
- nuclear-isospin target handling as implemented in the backend used for this extraction.

The primary physical object is the **b-space TMDPDF**

\[
\widetilde f_{1,q/h}(x,b_T;Q,Q^2).
\]

The regularized \(k_T\)-space curves are companion representations derived from the fitted \(b_T\)-space ensemble.

---

## What is meant by N³LL here?

The label **N³LL** refers to the resummed b-space \(W\)-term perturbative backend used in the fixed-target extraction.  The nonperturbative DNN does not itself carry a logarithmic accuracy label.  The accuracy label belongs to the perturbative CSS/TMD operator and its evolution/matching ingredients.

In this repository, the phrase

```text
N³LL fixed-target DY TMD extraction
```

means:

1. the perturbative \(W\)-term kernel is generated using the N³LL resummation setup used in the extraction;
2. NLO hard/OPE pieces are inserted in the \(W\)-term construction;
3. the DNN only parametrizes the smooth nonperturbative damping factor \(F_{\rm NP}\);
4. the extraction is validated through internal bridge, central-fit, replica, b-space, and regularized-k-space audits.

It does **not** claim that a fully matched collider/global-DY observable, including all high-\(q_T\) \(Y\)-term and fixed-order tail ingredients, has been externally validated at N³LL accuracy.  That collider/global extension is a separate project.

---

## Nonperturbative DNN architecture

The learned object is not the full cross section.  The perturbative b-space kernel is computed separately and cached.  The DNN learns only a constrained nonperturbative damping factor.

The model uses a FiLM-conditioned neural network for a positive damping rate:

\[
A_\theta(x,b_T)\ge 0.
\]

The physical nonperturbative factor is then built by a monotone integral scaffold,

\[
I_\theta(x,b_T)
=
\int_0^{b_T} db'\,2b'\,A_\theta(x,b'),
\]

\[
F_{\rm NP}(x,b_T)
=
\exp[-I_\theta(x,b_T)].
\]

This guarantees

\[
F_{\rm NP}(x,0)=1,\qquad
\frac{dF_{\rm NP}}{db_T}\le 0
\]

on the ordered \(b_T\) grid.

The architecture used in the v23a fits is:

```text
b_T branch:
  features: [b, b^2, sqrt(b+eps), ln(1+b)]
  radial lift: Linear(4 -> 48) + tanh
  trunk: 3 FiLM residual blocks, each 48-wide

x branch:
  features: [x, logit(x)]
  conditioning MLP: Linear(2 -> 32) + SiLU, Linear(32 -> 32) + SiLU
  output: FiLM parameters gamma_i(c), beta_i(c) for each residual block

head:
  Linear(48 -> 1) + Softplus + a_min
  output: A_theta(x,b_T) >= 0
```

The DNN philosophy is deliberately conservative:

- learn only the smooth nonperturbative damping;
- keep the perturbative \(N^3LL\) \(W\)-term outside the DNN;
- enforce physical endpoint and monotonicity constraints by construction;
- use data replicas for experimental uncertainty propagation;
- use PDF-member overlays for the final TMD uncertainty band;
- keep scale/profile/model-form variations separate from the baseline extraction.

A node/layer architecture figure can be generated with:

```bash
PYTHONPATH=. python v23/tools/draw_v23a_tmd_dnn_node_architecture_clean2.py \
  --out figures/v23a_tmd_dnn_node_architecture.pdf
```

---

## Repository layout

A typical public checkout is expected to have the following structure:

```text
.
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── pyproject.toml
├── docs/
│   ├── fixed_target_dy_tmd_note.pdf
│   └── fixed_target_dy_tmd_note.tex
├── figures/
│   ├── bspace-result.png
│   ├── kspace-result.png
│   └── v23a_tmd_dnn_node_architecture.png
├── v22/
│   ├── backends/
│   │   └── bt_internal_css_backend_v22_full.py
│   ├── src/
│   └── tools/
├── v23/
│   ├── tools/
│   │   ├── plot_v23a_paper_bspace_d_tmd.py
│   │   ├── plot_v23a_traditional_kspace_tmd.py
│   │   ├── construct_v23a_regularized_kspace_tmd.py
│   │   ├── compare_v23a_regularized_kspace_modes.py
│   │   ├── make_v23a_pdf_overlay_plan_from_runs.py
│   │   ├── construct_v23a_data_pdf_bspace_tmd_bands_v2.py
│   │   └── draw_v23a_tmd_dnn_node_architecture_clean2.py
│   └── freeze/
│       ├── v23a_lambda3_normpriors15_p2p5_E772_E288400_50rep_DYonly_bspace_sensitivity/
│       └── v23a_lambda3_normpriors15_p2p5_E772_E288400_50rep_DYonly_kspace_regularized_expPDF_overlay/
├── Data/
│   └── README.md
├── outputs/
│   └── README.md
├── replica_pilot_v23a_lambda3_normpriors15_p2p5_E772_E288400_cached_cuda/
│   └── README.md
├── replica_v23a_expPDF_overlay_lambda3_normpriors15_p2p5_50rep/
    └── README.md
└── production/
    └── lambda1_empirical_reference_full96x50/
        ├── PRODUCTION_MANIFEST.json
        ├── PRODUCTION_AUDIT.json
        ├── bspace_combined_bands.csv
        └── kspace_combined_bands.csv
```

Large replica outputs and backend caches may be distributed through a release asset or archived artifact rather than committed directly to git.

---

## Quick start

Use Python 3.10 or newer.  The extraction workflow was developed in a Python environment with PyTorch, LHAPDF, NumPy, SciPy, pandas, and matplotlib.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Some workflows require LHAPDF and the `NNPDF40_nnlo_as_01180` set.  If the full artifact bundle is present, you can regenerate the standard plots without retraining.

Generate the standard \(b_T\)-space paper figure:

```bash
PYTHONPATH=. python v23/tools/plot_v23a_paper_bspace_d_tmd.py \
  --band-dir replica_v23a_expPDF_overlay_lambda3_normpriors15_p2p5_50rep/tmd_bspace_bands_expPDF_overlay \
  --central-grid plots/v23a_fixed_target_lowQ_normpriors15_p2p5_E772_E288400_central_exactx/v22_scheme_tmd_bspace_long.csv \
  --flavor d \
  --quantity ftilde \
  --x 0.10 \
  --Q 10 \
  --b-max 4 \
  --band-label "68% exp+PDF overlay" \
  --central-label "central fit, PDF0" \
  --out figures/bspace-result.pdf
```

Generate the standard regularized \(k_T\)-space companion figure:

```bash
PYTHONPATH=. python v23/tools/plot_v23a_traditional_kspace_tmd.py \
  --band-dir replica_v23a_expPDF_overlay_lambda3_normpriors15_p2p5_50rep/kspace_regularized_expPDF_overlay_expb2 \
  --central-bspace-grid plots/v23a_fixed_target_lowQ_normpriors15_p2p5_E772_E288400_central_exactx/v22_scheme_tmd_bspace_long.csv \
  --quantity ftilde \
  --flavor d \
  --x 0.10 \
  --Q 10 \
  --k-max 4 \
  --title "TMD PDFs" \
  --label "v23a FT-DY" \
  --band-label "68% exp+PDF overlay" \
  --central-label "central fit, PDF0" \
  --show-zero \
  --out figures/kspace-result.pdf
```

Generate the DNN architecture diagram:

```bash
PYTHONPATH=. python v23/tools/draw_v23a_tmd_dnn_node_architecture_clean2.py \
  --out figures/v23a_tmd_dnn_node_architecture.pdf
```

---

## Main outputs

The fixed-target b-space ensemble is stored in the b-space overlay directory:

```text
replica_v23a_expPDF_overlay_lambda3_normpriors15_p2p5_50rep/
└── tmd_bspace_bands_expPDF_overlay/
    ├── v23a_dataPDF_tmd_replica_bspace_long.csv
    ├── v23a_dataPDF_tmd_replica_bspace_bands.csv
    ├── v23a_dataPDF_relative_band_summary.csv
    ├── v23a_dataPDF_tmd_manifest.json
    ├── F_NP_dataPDF_bands.pdf
    ├── ftilde_dataPDF_bands.pdf
    ├── b_ftilde_dataPDF_bands.pdf
    └── b_x_ftilde_dataPDF_bands.pdf
```

The default regularized \(k_T\)-space companion is stored in:

```text
replica_v23a_expPDF_overlay_lambda3_normpriors15_p2p5_50rep/
└── kspace_regularized_expPDF_overlay_expb2/
    ├── v23a_regularized_kspace_replica_long.csv
    ├── v23a_regularized_kspace_bands.csv
    ├── v23a_regularized_kspace_curve_audit.csv
    ├── v23a_regularized_kspace_summary.json
    ├── bspace_b0_medians.csv
    ├── ftilde_kspace_regularized_bands.pdf
    └── x_ftilde_kspace_regularized_bands.pdf
```

The large-\(b_T\) regularization comparison is stored in:

```text
replica_v23a_expPDF_overlay_lambda3_normpriors15_p2p5_50rep/
└── kspace_regularized_comparison/
    ├── regularization_mode_comparison_summary.json
    ├── regularization_mode_curve_comparison.csv
    └── regularization_mode_pointwise_medians.csv
```

---

## Validation summary

The result is supported by a stack of checks rather than a single number.

| Check | Outcome | Purpose |
|---|---:|---|
| v22 full backend integration | Pass | Confirms the perturbative backend and scheme flags are active. |
| v23a central fixed-target refit | Pass | Confirms the corrected fixed-target table gives a stable central solution. |
| central b-space shape audit | Pass | Confirms smooth, finite, monotone b-space TMD behavior. |
| 50-replica fit distribution | Pass | Confirms replica fits are statistically controlled. |
| normalization-pull distribution | Pass | Confirms fitted dataset normalizations remain within prior expectations. |
| random-split stability | Pass | Confirms 50 replicas are enough for stable b-space bands. |
| exp+PDF overlay reconstruction | Pass | Adds PDF-member variation to the final TMD reconstruction. |
| regularized \(k_T\)-space comparison | Pass | Confirms expb2, expb, and taper prescriptions agree over \(k_T\le4\) GeV. |

Key audit values:

```text
central v23a fixed-target chi2_total: 0.958297496396682
central max dataset chi2: 1.4252333188683897
50-rep chi2 q95: 2.1310145686109365
50-rep norm-pull q95: 2.707433342933655
random-split width q90: 0.6439466887345098
random-split center q90: 0.021749180926595522
regularized kT p90 regularization difference: 0.030912545977260095
regularized kT max regularization difference: 0.04304417062181683
```

The \(k_T\)-space negative-tail diagnostics are not strict positivity constraints.  The worst median dip in the audited curves is approximately \(-1.8\%\) of the peak, and the negative lobes are retained as finite-transform diagnostics rather than clipped.

---

## Regularized \(k_T\)-space companion

The primary extraction is in \(b_T\) space.  The \(k_T\)-space representation is obtained afterward by a regularized finite-\(b_T\) Hankel transform,

\[
f(k_T)
=
\frac{1}{2\pi}
\int_0^\infty db_T\,b_T\,J_0(k_T b_T)\,\widetilde f(b_T).
\]

The default frozen prescription uses

```text
tail mode: expb2
b_transform_max: 24 GeV^-1
n_b_transform: 6001
end taper start: 0.92 * b_transform_max
kT range: 0 <= kT <= 4 GeV
```

Alternative large-\(b_T\) prescriptions, `expb` and `taper`, are compared against the default `expb2` prescription.  The final \(k_T\) range is accepted because these prescriptions agree within about 3.1% at p90 over the active region.

The \(k_T\)-space object should be cited as a **regularized finite-\(b_T\) companion representation**, not as an unconstrained high-\(k_T\) perturbative-tail prediction.

---

## Claim discipline

The repository makes the following claim:

```text
A fixed-target DY b-space TMD extraction has been performed with an N³LL-resummed
W-term backend, NLO hard/OPE matching in W, constrained DNN nonperturbative
factor, experimental pseudo-data replicas, and PDF-member overlay uncertainty.
```

The repository does **not** claim:

```text
a completed collider/global-DY extraction,
a full high-qT matched observable validation,
PDF-through-refit uncertainty,
scale/profile/nuclear/model-form uncertainty envelopes,
or production-quality accelerator-data covariance treatment.
```

The fixed-target result is therefore a production-quality fixed-target \(b_T\)-space TMD extraction with a regularized \(k_T\)-space companion, while the accelerator/global-DY extension remains future work.

---

## Reproducibility notes

- The fixed-target result depends on the staged data tables and frozen audit outputs.
- The b-space TMD bands can be regenerated from the saved replica runs and PDF overlay plan.
- The \(k_T\)-space curves can be regenerated from the b-space long table without retraining.
- Exp+PDF overlay bands include experimental data-replica variation in \(F_{\rm NP}\) and PDF-member variation in the TMD reconstruction.
- Exp+PDF overlay bands do not include PDF-through-refit shifts of \(F_{\rm NP}\).
- Dataset-normalization uncertainties are handled through the pseudo-data and profiled-nuisance protocol used in the v23a fixed-target extraction.
- The regularized \(k_T\)-space curves should be regenerated together with the regularization comparison if the b-space ensemble changes.

---

## Relationship to `uva-spin/k-space`

The companion `uva-spin/k-space` repository is a formalism/validation suite for a pure-\(k_T\) CSS2-equivalent prescription.  This `b-space` repository is different: it contains the numerical fixed-target DY extraction in \(b_T\) space and a regularized \(k_T\)-space companion obtained from the fitted \(b_T\)-space ensemble.

In short:

```text
k-space:
  formal pure-kT prescription and validation suite

b-space:
  fixed-target DY N³LL b-space extraction and TMD ensemble
```

---

## Citing this repository

Use the metadata in `CITATION.cff` once the repository has a tagged release or archived DOI.  Until then, cite:

```text
https://github.com/uva-spin/b-space
```

and the associated fixed-target DY TMD note or paper draft.

---

## License

This repository is released under the MIT license unless otherwise noted.  External data files, PDF grids, and third-party coefficient/source files may carry their own licenses and should be cited according to their original sources.
